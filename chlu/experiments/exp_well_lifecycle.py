"""⭐ Experiment WELL-LIFECYCLE (C2W8) — **stage 1: the rig, the instrument, the
census.** The kill comes before the build.

**What this is.** The `exp_cl_entry` Class-IL stream ported onto the **full CLU
system** (:class:`~chlu.core.clu_system.CluSystem`, learned ``V_theta``), run
until it is at least **2x over-dug** (``overdig = n_admitted / well_budget >=
2.0``, prereg §3.4), and then censused:

* how deep are the wells this rig actually digs (the `cluformer-pilot` warning:
  a ``CluSystem`` fed a stream may dig **no wells at all** — an inert store makes
  a census vacuous for a reason that is *not* K1's reason, and the two must never
  be confused);
* ``theta_att``, **measured** by SC-6 bisection on this rig;
* ``P`` (live attractors, never read, not protected) and ``M`` (near-duplicate
  pairs), with the two well populations reported **separately** (mechanic 1);
* every depth curve **RAW and NETTED** (B1), the netting replayed exactly from
  the controller's own decay log and audited with C2W6's residual instrument
  (imported from ``exp_anti_erosion``, which this wave does not edit);
* the ``U`` telemetry (B2), item-id-keyed, primary proxy ``read_hits``.

The deliverable is ``census.json`` with a **mechanically computed**
``stage2_unlock`` (prereg §5 K1: ``UNLOCK iff P >= 0.05 or M >= 0.05`` on the
seed mean; ``KILL iff both < 0.05 on every seed``). ⛔ If it is ``false``,
stage 2 is **not built** and the census IS the wave's finding.

⛔ **Declared NOT-RUNs of this file** (never reported as nulls): no I2 verdict,
no tier-ii verdict, no full-CLU verdict, no cross-stream criterion.

Runnable::

    uv run python -m chlu.experiments.exp_well_lifecycle --quick
    chlu exp-well-lifecycle [--project N] [--seeds 0,1,2] [--quick]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional

import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.soft_certificate import population_median_nn
from chlu.core.well_lifecycle import census, gate_addr_verdict, unlock_verdict
from chlu.experiments.usage_telemetry import (
    UsageTelemetry,
    attach_reads,
    loo_loss_contribution,
)

#: read-only imports from the sibling spoke's files (`exp_cl_entry` is NOT ours)
from chlu.experiments.exp_cl_entry import (  # noqa: E402  (documented ownership)
    PHI_PRIMARY,
    RingBufferKNN,
    build_cl_stream,
    build_phi,
)

#: the C2W6 residual-vs-decay-law instrument, IMPORTED (that file is C2W6's)
from chlu.experiments.exp_anti_erosion import _interference_audit  # noqa: E402


# ---------------------------------------------------------------------------
# the rig
# ---------------------------------------------------------------------------
class PhiAddress:
    """``x -> (phi(x) scaled into the address ball | payload channels = 0)``.

    The scale is fixed **once** from the ``phi`` fit pool (task-1 only, disjoint
    from every stored item), so no stream item and no test item ever influences
    the store's geometry.
    """

    def __init__(self, phi, dim: int, addr_dim: int, scale: float):
        self.phi = phi
        self.dim = int(dim)
        self.addr_dim = int(addr_dim)
        self.scale = float(scale)

    def __call__(self, x):
        import jax.numpy as jnp

        x = np.asarray(x, dtype=np.float32)
        # ⚠ The system re-embeds points that are ALREADY store-space (every
        # internal probe: `_relaxed_sites`, `self_probe`, `place_pass`). phi must
        # be idempotent on those or the store re-reads its own sites through a
        # 784-dim read-in and crashes. dim (=addr+payload) can never collide with
        # the image dimension, so the test is unambiguous.
        if x.shape[-1] == self.dim:
            return jnp.asarray(x)
        f = np.asarray(self.phi(x), dtype=np.float32)
        if f.ndim == 1:
            f = f[None, :]
        out = np.zeros((f.shape[0], self.dim), dtype=np.float32)
        out[:, : self.addr_dim] = f[:, : self.addr_dim] * self.scale
        return jnp.asarray(out)

    def keys(self, x) -> np.ndarray:
        """The raw address vectors (for the launder and the geometry rules)."""
        f = np.asarray(self.phi(np.asarray(x, dtype=np.float32)), dtype=np.float32)
        return f[:, : self.addr_dim] * self.scale


def _median_nn(keys: np.ndarray) -> float:
    k = np.asarray(keys, dtype=float)
    if k.shape[0] < 3:
        return float("nan")
    d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.min(d, axis=1)))


def cl_config(cfg: CHLUConfig):
    """The CL-entry config the stream builder wants, at this wave's phi_dim.

    ⚠ ``phi_dim`` is forced to ``addr_dim``: the store's address dimension and
    the read-in's output dimension are one number, not two.
    """
    w = cfg.experiment_well_lifecycle
    cl = copy.deepcopy(cfg.experiment_cl_entry)
    cl.dataset = w.dataset
    cl.phi_arm = w.phi_arm
    cl.phi_dim = int(w.addr_dim)
    return cl


def store_config(cfg: CHLUConfig, seed: int, d_safe: float,
                 overrides: Optional[Dict[str, Any]] = None) -> CluSystemConfig:
    """⭐ ``overrides`` (C2W8 pass 2, additive): extra ``CluSystemConfig`` fields
    an *arm* sets on top of the census's own store config, so both pass-2 arms
    re-run **this** rig rather than a copy of it. ``None`` (the default) is the
    pass-1 path, unchanged.
    """
    w = cfg.experiment_well_lifecycle
    return CluSystemConfig(
        addr_dim=int(w.addr_dim), payload_dim=int(w.payload_dim),
        capacity=int(w.capacity), budget=int(w.capacity),
        seed=int(seed), leak=float(w.leak), stage_lifetimes=True,
        d_safe_override=float(d_safe),
        write_steps=int(w.write_steps), read_steps=int(w.read_steps),
        address_steps=int(w.address_steps),
        n_query_per_item=int(w.n_query_per_item),
        quick=bool(w.quick),
        **dict(overrides or {}),
    )


#: ⛔ Bound ONCE at import, so the width guard below can recover the census's own
#: (un-substituted) store config even while an arm has substituted `store_config`.
_BASE_STORE_CONFIG = store_config


class UnselectedAtomWidth(RuntimeError):
    """⭐⭐ **C2W8 close-out item (vi.5)** — the census REFUSED to run.

    Arm A's banked pass-2/pass-3 runs used ``atom_width_frac_spacing = 1.5``,
    passed on the CLI, while the shipped :class:`ExperimentCaptureArmAConfig`
    default is **0.5** — a width `c2w8p3-gate-addr` measured **does not clear the
    pass-2 gate**. A census that silently runs at a width nobody selected
    produces numbers **nobody can attribute**, and the spine came within one
    explicit declaration of scoring a different store than the one it reports.

    So the census now recovers the width it is *about* to run at as a fraction of
    the cell's own **measured** key spacing and refuses, loudly, unless that
    fraction is **explicitly declared** somewhere in config.
    """


def _declared_atom_width_fractions(cfg: CHLUConfig) -> Dict[str, float]:
    """Every atom-width fraction that is EXPLICITLY declared in config.

    Resolution order (``experiment_well_lifecycle.atom_width_selection`` takes
    absolute priority when set): the census's own pin, then each arm's own
    declared fraction. An arm that co-scales its width without writing the
    fraction into a config dataclass is exactly the unattributable case.
    """
    w = cfg.experiment_well_lifecycle
    if w.atom_width_selection is not None:
        return {"experiment_well_lifecycle.atom_width_selection":
                float(w.atom_width_selection)}
    out: Dict[str, float] = {}
    for name in ("experiment_capture_arm_a", "experiment_capture_strong_phi"):
        arm = getattr(cfg, name, None)
        frac = getattr(arm, "atom_width_frac_spacing", None) if arm else None
        if frac is not None:
            out[f"{name}.atom_width_frac_spacing"] = float(frac)
    return out


def _assert_selected_atom_width(cfg: CHLUConfig, scfg: CluSystemConfig,
                                spacing: float, *, seed: int,
                                overrides: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
    """⭐ Refuse to census a store whose atom width nobody selected (item vi.5).

    The width is **not** refused for being unusual — it is refused for being
    **unattributable**. Three cases:

    * the width equals the census's own un-substituted store config ⇒ the pass-1
      path, nothing was co-scaled, **allowed**;
    * the width equals a **declared** fraction of the measured key spacing ⇒
      attributable, **allowed** (the source is recorded in the flag table);
    * anything else ⇒ :class:`UnselectedAtomWidth`, naming the effective
      fraction, every declared fraction, and how to declare this one.
    """
    w = cfg.experiment_well_lifecycle
    base = _BASE_STORE_CONFIG(cfg, seed, float(scfg.d_safe_override or 0.0),
                              overrides=overrides)
    frac_eff = (float(scfg.atom_width) / float(spacing)
                if float(spacing) > 0 else float("nan"))
    declared = _declared_atom_width_fractions(cfg)
    rtol = float(w.atom_width_selection_rtol)
    source, matched = None, None
    if abs(float(scfg.atom_width) - float(base.atom_width)) <= rtol * max(
            abs(float(base.atom_width)), 1e-12):
        source = "census default (not co-scaled; the pass-1 path)"
    else:
        for name, frac in declared.items():
            if np.isfinite(frac_eff) and abs(frac_eff - frac) <= rtol * max(abs(frac), 1e-12):
                source, matched = name, float(frac)
                break
    block = {
        "atom_width": float(scfg.atom_width),
        "key_spacing_used": float(spacing),
        "atom_width_frac_spacing_effective": frac_eff,
        "declared_selections": declared,
        "selection_source": source,
        "selection_value": matched,
        "refuse_unselected_atom_width": bool(w.refuse_unselected_atom_width),
        "rule": ("C2W8 close-out item (vi.5): the census REFUSES to run at a "
                 "width nobody selected — the effective width fraction must "
                 "match a fraction explicitly declared in config, or equal the "
                 "census's own un-substituted default"),
    }
    if source is None and bool(w.refuse_unselected_atom_width):
        raise UnselectedAtomWidth(
            "⛔ THE CENSUS REFUSES TO RUN AT AN UNSELECTED ATOM WIDTH.\n"
            f"  effective atom_width           = {float(scfg.atom_width):.6g}\n"
            f"  measured key spacing           = {float(spacing):.6g}\n"
            f"  => atom_width_frac_spacing     = {frac_eff:.6g}\n"
            f"  census default atom_width      = {float(base.atom_width):.6g}\n"
            f"  declared selections in config  = {declared or '{} (NONE)'}\n"
            "  A census run at a width nobody selected produces numbers nobody "
            "can attribute (charter §A31.6; arm A's banked 1.5 vs the shipped "
            "default 0.5). Declare it: set "
            "`experiment_well_lifecycle.atom_width_selection` to the fraction "
            "you mean, or set the arm's own `atom_width_frac_spacing`.")
    return block


def label_to_payload(label: int, scale: float) -> float:
    """``(label - 4.5) / scale`` — a bounded, monotone, class-separating value.

    At the default ``scale = 9`` adjacent classes sit ``0.111`` apart, i.e.
    **above** ``payload_tol = 0.1``: two items are payload-mergeable iff they
    carry the **same class**, which is exactly the near-duplicate population
    over-digging is supposed to produce.
    """
    return float((int(label) - 4.5) / float(scale))


# ---------------------------------------------------------------------------
# one census cell
# ---------------------------------------------------------------------------
def run_census_cell(cfg: CHLUConfig, seed: int, *, data=None,
                    verbose: bool = True,
                    clu_overrides: Optional[Dict[str, Any]] = None,
                    post_build: Optional[Any] = None) -> Dict[str, Any]:
    """Stream -> full-CLU store -> reads -> census, for one seed.

    ⭐ **C2W8 pass-2 arm seams (additive; both ``None`` = the pass-1 path,
    unchanged).** ``clu_overrides`` are extra :class:`CluSystemConfig` fields the
    arm sets (its own flag); ``post_build(system, stream=..., embed=..., cfg=...,
    seed=...)`` runs once, after the system is built and **before the first
    write**, which is where an arm that must train something (arm B's emission
    head) pays its amortised cost. The census itself is untouched: pass 2's race
    is only a race if both arms are censused by this same function.
    """
    w = cfg.experiment_well_lifecycle
    cl = cl_config(cfg)
    t0 = time.time()

    stream = build_cl_stream(cl, seed, data=data)
    phi, phi_prov = build_phi(w.phi_regime or PHI_PRIMARY, stream, cl, seed)

    # --- the address scale, fixed from the phi fit pool only ---
    pool = np.asarray(stream[f"fit_pool_{w.phi_regime}"], dtype=np.float32)
    f_pool = np.asarray(phi(pool), dtype=float)
    r95 = float(np.percentile(np.linalg.norm(f_pool, axis=1), 95.0))
    # ⭐ C2W8 pass 3, the SCALE-ONLY control (PREREG-C2W8-PASS3 §4): identical
    # phi, address scale x a declared constant. 1.0 (the default) is the pass-1/2
    # path bit-for-bit. Every geometric quantity downstream (d_safe, the arms'
    # co-scaled atom widths, the G-ADDR cue jitter) is MEASURED from the keys, so
    # it co-scales; if G-ADDR still moves, the leg is reading the scale.
    scale = float(1.0 / max(r95, 1e-9)) * float(w.addr_scale_mult)
    embed = PhiAddress(phi, dim=int(w.addr_dim + w.payload_dim), addr_dim=int(w.addr_dim),
                       scale=scale)

    # --- the admission radius: the CL entry's own sizing rule, frozen on task 1 ---
    # ⭐⭐ C2W8 close-out item (v) + (vi.6) — ONE defect at three sites: the
    # ~200-key SIZING spacing standing in for the ~16-item STORE spacing. It sized
    # `d_safe` (=> monitor #3's 0.000 refusal rate was arithmetic), it normalised
    # the G-ADDR cue jitter (=> arm-dependent cue difficulty, 30 % spread), and it
    # produced the RETRACTED §A29.5 mechanism. Fixing the population choice fixes
    # all three; `d_safe_population = "sizing"` restores the old behaviour exactly.
    task1_keys = embed.keys(
        stream["train_X"][0][: min(int(w.d_safe_sizing_n), len(stream["train_X"][0]))])
    med_nn = _median_nn(task1_keys)
    # ⛔ the population is the STORE's, i.e. `capacity` (16 on the shipped rig —
    # the number the Advisor's erratum names), never the ~200-key sizing set.
    spacing_pop = population_median_nn(task1_keys, int(w.capacity),
                                       n_draws=int(w.d_safe_population_draws),
                                       seed=int(seed))
    med_nn_store = float(spacing_pop["median_nn_population"])
    use_store_pop = str(w.d_safe_population).lower() == "store"
    med_nn_for_d_safe = (med_nn_store if use_store_pop and np.isfinite(med_nn_store)
                         else med_nn)
    d_safe = float(w.d_safe_frac) * med_nn_for_d_safe

    scfg = store_config(cfg, seed, d_safe, overrides=clu_overrides)
    # ⭐ item (vi.5): the width the store will actually run at is recovered from
    # the SAME spacing the store-config factory recovers it from
    # (`d_safe / d_safe_frac`), and refused unless someone selected it.
    width_block = _assert_selected_atom_width(cfg, scfg, med_nn_for_d_safe,
                                              seed=seed, overrides=clu_overrides)
    sysm = build_system(scfg, key=jax.random.PRNGKey(seed), phi=embed, loud=False)
    if post_build is not None:
        post_build(sysm, stream=stream, embed=embed, cfg=cfg, seed=seed)
    tel = UsageTelemetry()
    launder = RingBufferKNN(int(w.well_budget))

    # --- the B1 audit arrays (C2W6's residual instrument is fed, not re-written) ---
    K = int(w.capacity)
    site_depth: List[np.ndarray] = []
    live_hist, slot_hist, adm_hist, gscale_hist = [], [], [], []
    depth_trace: List[Dict[str, Any]] = []

    def snapshot() -> np.ndarray:
        from chlu.core.well_lifecycle import own_foreign_site_depth

        out = np.zeros((K, 2), dtype=float)
        for r in sysm.controller.allocator.records.values():
            z = np.zeros((sysm.store.dim,), dtype=float)
            z[: sysm.store.addr_dim] = r.center
            z[sysm.store.addr_dim] = r.payload
            out[int(r.slot)] = own_foreign_site_depth(sysm.store, int(r.slot), z)
        return out

    def live_mask() -> np.ndarray:
        m = np.zeros((K,), dtype=float)
        for r in sysm.controller.allocator.records.values():
            m[int(r.slot)] = 1.0
        return m

    n_admitted = n_refused = n_offered = 0
    read_events = 0
    target = int(np.ceil(float(w.overdig_target) * int(w.well_budget)))
    depth_first_writes: Optional[List[float]] = None

    for t in range(int(cl.n_tasks)):
        Xt = np.asarray(stream["train_X"][t], dtype=np.float32)
        yt = np.asarray(stream["train_y"][t], dtype=int)
        n_task = min(int(w.n_offer_per_task), len(Xt))
        for j in range(n_task):
            if n_admitted >= target and t >= 1:
                break
            x = Xt[j: j + 1]
            key = embed.keys(x)[0]
            payload = label_to_payload(int(yt[j]), float(w.payload_scale))
            permanent = j < int(w.permanent_per_task)
            iid = 1000 * t + j
            before_live = live_mask()
            site_depth.append(snapshot())
            n_offered += 1
            amps_before = np.asarray(sysm.controller.allocator.store.amps, dtype=float).copy()
            rep = sysm.write_stream([{
                "item_id": iid, "address": key, "payload": payload,
                "permanent": bool(permanent),
                "leak": 0.0 if permanent else float(w.leak),
            }])
            ok = bool(rep.admitted)
            amps_after = np.asarray(sysm.controller.allocator.store.amps, dtype=float)
            gs = np.ones((K,), dtype=float)
            nz = amps_before > 0
            gs[nz] = np.sqrt(np.clip(amps_after[nz] / amps_before[nz], 0.0, None))
            slot = -1
            if ok:
                n_admitted += 1
                tel.note_admitted(iid, sysm._t)
                launder.offer(key, int(yt[j]))
                slot = int(sysm._slot_of(iid))
            else:
                n_refused += 1
            live_hist.append(before_live)
            slot_hist.append(slot)
            adm_hist.append(1.0 if ok else 0.0)
            gscale_hist.append(gs)

            if ok and depth_first_writes is None and n_admitted >= 3:
                d, _ = sysm.well_fits()
                depth_first_writes = [float(x) for x in d]
                if verbose:
                    print(f"  [pilot check] depth after {n_admitted} writes: "
                          f"{np.round(depth_first_writes, 4).tolist()}", flush=True)

            # --- a READ EVENT: this is what makes "never read" computable ---
            if ok and (n_admitted % int(w.read_every) == 0):
                qX, qy = _query_batch(stream, t, int(w.read_batch), seed + read_events)
                q0 = np.asarray(embed(qX))
                res = sysm.read(q0)
                attach_reads(sysm, tel, res, sysm._t)
                read_events += 1
                d, _ = sysm.well_fits()
                depth_trace.append({
                    "t": int(sysm._t), "n_admitted": n_admitted,
                    "n_live": int(sysm.controller.allocator.n_live),
                    "depth_raw_median": float(np.median(d)) if d.size else float("nan"),
                    "read_event": read_events,
                    "read_acc": _read_acc(res, sysm, qy, float(w.payload_scale)),
                    "knn_acc": float(np.mean(launder.predict(embed.keys(qX)) == qy)),
                })
        if n_admitted >= target and t >= 1:
            break

    site_depth.append(snapshot())
    audit = _interference_audit(site_depth, np.asarray(live_hist),
                                np.asarray(slot_hist, dtype=int),
                                np.asarray(adm_hist), np.asarray(gscale_hist))

    cen = census(sysm, tel, well_budget=int(w.well_budget), n_admitted=n_admitted,
                 seed=seed, n_dirs=int(w.capture_dirs),
                 bisect_steps=int(w.capture_bisect_steps),
                 measure_capture=bool(w.measure_capture),
                 drift_floor_frac=float(w.gdrift_floor_frac_spacing),
                 drift_ceil_frac=float(w.gdrift_ceil_frac_spacing))
    ids, _, _ = sysm.codebook()
    loo = (loo_loss_contribution(sysm, [int(i) for i in ids],
                                 repeats=int(w.loo_repeats), seed=seed)
           if w.run_loo and len(ids) else
           {"status": "NOT RUN", "role": "SECONDARY — reported, never deciding"})

    probe = sysm.self_probe()
    trips = _trip_state(sysm)
    usage = tel.summary(live_ids=[int(i) for i in ids])
    g_addr = _gate_addr_block(sysm, cfg, cen, depth_trace, med_nn, seed, usage,
                              codebook_spacing=float(cen.get("codebook_spacing",
                                                             float("nan"))))
    out = {
        "seed": int(seed),
        "census": cen,
        # ⭐ C2W8 PASS 3 (charter §A30.1): the addressability leg the pass-2 gate
        # was blind to. Additive — the `census` dict above is untouched, and both
        # pass-2 arms inherit it because both are censused by THIS function.
        "g_addr": g_addr,
        "usage": usage,
        "loo": loo,
        "decay_netting_audit": audit,
        "depth_trace": depth_trace,
        "depth_after_first_writes": depth_first_writes,
        "self_probe": {k: probe[k] for k in ("acq", "strict", "decode", "chance",
                                             "n_probed") if k in probe},
        "monitor_trips": trips,
        "stream": {
            "n_offered": n_offered, "n_admitted": n_admitted, "n_refused": n_refused,
            "refusal_rate": float(n_refused / max(n_offered, 1)),
            "n_read_events": read_events,
            "overdig": cen["overdig"], "well_budget": int(w.well_budget),
            "n_live_end": cen["n_live"],
        },
        "geometry": {"phi_scale": scale, "median_nn_task1": med_nn, "d_safe": d_safe,
                     "r95_phi_norm": r95,
                     # ⭐ item (v): the population the admission radius was sized
                     # on, stated beside the sizing-set number it replaces.
                     "median_nn_store_population": med_nn_store,
                     "d_safe_population": str(w.d_safe_population),
                     "d_safe_spacing_used": float(med_nn_for_d_safe),
                     "spacing_population_block": spacing_pop},
        # ⭐ item (vi.5): the width this cell actually ran at, and who selected it
        "atom_width_selection": width_block,
        "phi_provenance": phi_prov,
        "flags": _flag_table(cfg, sysm, seed, d_safe),
        "bytes": _byte_ledger(sysm, launder, int(w.addr_dim)),
        "wall_s": float(time.time() - t0),
    }
    if verbose and isinstance(g_addr.get("A1"), dict):
        # ⭐ item (ii): the margin-in-SE and reads-to-flip travel with the boolean
        print(f"  seed {seed}: G-ADDR A1={g_addr['A1']['correct_basin_rate']:.4f} "
              f"(thr {g_addr['A1']['threshold']:.4f}, {g_addr['A1']['pass']}, "
              f"margin {g_addr['A1']['margin_in_se_vs_threshold']:+.2f} SE, "
              f"{g_addr['A1']['reads_to_flip']} read(s) from flipping) "
              f"A2={g_addr['A2']['never_addressed_frac']:.4f} ({g_addr['A2']['pass']}) "
              f"| A3 [DIAGNOSTIC, not in the pass condition] "
              f"a={g_addr['A3']['A3a_cue_margin']:+.4f} "
              f"b={g_addr['A3']['A3b_stream_margin']} "
              f"-> gate_addr_pass={g_addr['gate_addr_pass']}", flush=True)
    if verbose:
        gd = cen["G_DRIFT_two_sided"]
        pc = cen["P_comparability"]
        print(f"  seed {seed}: P={cen['P']:.4f} (n_non_capturing="
              f"{pc['n_non_capturing']}, comparable={pc['P_comparable_across_arms']}) "
              f"M={cen['M']:.4f} "
              f"overdig={cen['overdig']:.2f} n_live={cen['n_live']} "
              f"theta_att={cen['theta_att_block']['theta_att']:.4g} "
              f"depth_med(raw/netted)={cen['depth_raw_median']:.4g}/"
              f"{cen['depth_netted_median']:.4g} [{out['wall_s']:.0f}s]", flush=True)
        print(f"  seed {seed}: G-DRIFT two-sided ratio={gd['ratio']:.4g} in "
              f"[{gd['floor_frac_spacing']:g}, {gd['ceil_frac_spacing']:g}) x "
              f"codebook spacing -> {gd['pass']} "
              f"(fails_low_D2a={gd['fails_low_D2a_table_expressible']}, "
              f"fails_high={gd['fails_high_cannot_address']})", flush=True)
    return out


def _gate_addr_block(system, cfg: CHLUConfig, cen: Dict[str, Any],
                     depth_trace: List[Dict[str, Any]], med_nn: float,
                     seed: int, usage: Dict[str, Any],
                     codebook_spacing: float = float("nan")) -> Dict[str, Any]:
    """⭐ G-ADDR on this cell (C2W8 pass 3) — wiring only; the leg lives in
    :func:`chlu.core.well_lifecycle.gate_addr`.

    Two ingredients are handed in rather than re-measured, so the leg costs one
    extra read and nothing else:

    * the **measured** capture radii, straight off the census's own wells
      (same order: both iterate the sorted codebook);
    * ``A3b``, the stream's own launder margin — the mean over read events of
      ``read_acc - knn_acc``, i.e. the census's held-out class accuracy against
      the ring-buffer kNN-in-``phi`` launder, **matched items**.

    ⭐⭐ **C2W8 close-out item (vi.6): the cue jitter is normalised on the
    CODEBOOK spacing.** ``kappa_q`` used to multiply ``median_nn_task1``, the
    ~200-key **sizing** spacing, while the read must beat the **codebook**
    spacing — which made the cue's real difficulty **arm-dependent** (0.927 /
    0.875 / **0.710** across the pass-3 arms, a 30 % spread, so no cross-arm A1
    comparison was scale-matched). ``gaddr_spacing_population = "sizing"``
    restores the banked behaviour exactly. **Both** ratios are emitted on every
    cell either way, so the comparison can never again be made blind to it.

    ⭐⭐ **C2W8 close-out item (iv): the banked telemetry's caption is CORRECTED
    here** (charter §A31.1). ``n_never_read`` was gated on ``covered``, a
    LAUNCH-POINT test (``min_j |q0 - c_j| <= 1/2 min-sep``) — store-invariant by
    construction, which is the mechanical explanation of the digit-identical
    "58 / 62 / 62 of 64 unassigned reads" that was read as decisive and was
    **vacuous**. The telemetry is now gated on the **settle-side**
    ``settle_covered`` (:func:`chlu.experiments.usage_telemetry.attach_reads`),
    and the launch-side quantity is emitted here **under its own name**,
    ``launch_coverage_*``, so the retired sentence cannot be re-derived from it.
    """
    from chlu.core.well_lifecycle import gate_addr

    w = cfg.experiment_well_lifecycle
    if not bool(w.run_gate_addr):
        return {"status": "NOT RUN — run_gate_addr = False (declared, not a null)"}
    margins = [float(t["read_acc"]) - float(t["knn_acc"]) for t in depth_trace]
    stream_margin = float(np.mean(margins)) if margins else None
    # pooled binomial SE over ALL stream queries (ERRATA §1: the per-event SE at
    # n_events = 4 is unstable enough to flip a verdict).
    n_stream = int(len(depth_trace) * int(w.read_batch))
    p_s = float(np.mean([t["read_acc"] for t in depth_trace])) if depth_trace else 0.0
    p_l = float(np.mean([t["knn_acc"] for t in depth_trace])) if depth_trace else 0.0
    se_stream = (float(np.sqrt((p_s * (1 - p_s) + p_l * (1 - p_l)) / max(n_stream, 1)))
                 if depth_trace else None)
    cap = [float(x["capture_radius"]) for x in cen["wells"]]
    use_codebook = str(w.gaddr_spacing_population).lower() == "codebook"
    sp = (float(codebook_spacing) if use_codebook and np.isfinite(codebook_spacing)
          else float(med_nn))
    g = gate_addr(system, spacing=sp, kappa_q=float(w.gaddr_kappa_q),
                  n_query_per_item=int(w.gaddr_n_query_per_item), seed=int(seed),
                  capture=cap, stream_margin=stream_margin,
                  stream_margin_se=se_stream)
    g["telemetry_launch_side"] = {
        "launch_coverage_n_unassigned": usage.get("n_unassigned"),
        "launch_coverage_n_read_events": usage.get("n_read_events"),
        "label": ("LAUNCH-SIDE — a property of the query distribution against "
                  "the codebook, ⛔ NOT of the store"),
        "caveat": ("`covered` is a LAUNCH-POINT test on q0 "
                   "(min_j |q0 - c_j| <= 1/2 min-sep): same phi + same admitted "
                   "codebook ⇒ the same number whatever the store does. That is "
                   "the mechanical explanation of the digit-identical "
                   "58/62/62-of-64 'unassigned reads' reading, which is RETIRED "
                   "(§A31.1). ⛔ It is NOT the A2 leg and never was."),
    }
    g["telemetry_settle_side"] = {
        "frac_never_read": usage.get("frac_never_read"),
        "n_never_read": usage.get("n_never_read"),
        "label": ("SETTLE-SIDE from the C2W8 close-out onward (gated on "
                  "`settle_covered`); ⚠ values banked BEFORE it were "
                  "launch-gated and are not comparable"),
        "coverage_side": usage.get("coverage_side"),
    }
    g["A3"]["A3b_n_stream_queries"] = n_stream
    g["A3"]["A3b_read_acc_by_event"] = [float(t["read_acc"]) for t in depth_trace]
    g["A3"]["A3b_knn_acc_by_event"] = [float(t["knn_acc"]) for t in depth_trace]
    # ⭐ item (vi.6): which population the cue jitter was normalised on, and BOTH
    # ratios, on every cell — so no cross-arm comparison is made blind to it.
    g["spacing_source"] = ("codebook_spacing (the live store's own median-NN — "
                           "the resolution the read must beat)" if use_codebook
                           else "median_nn_task1 (the ~200-key SIZING spacing)")
    g["gaddr_spacing_population"] = str(w.gaddr_spacing_population)
    g["median_nn_task1_sizing"] = float(med_nn)
    g["cue_sigma_over_sizing_spacing"] = (
        float(g["cue_sigma"] / med_nn) if med_nn > 0 else float("nan"))
    g["addr_scale_mult"] = float(w.addr_scale_mult)
    return g


def full_state_coscaled_config(cfg: CHLUConfig, alpha: float) -> CHLUConfig:
    """⭐⭐ **C2W8 close-out item (iii): the LEGAL rescale of this rig.**

    Head-ratified (§A31.6): **address-only rescaling is NOT a symmetry of the
    system — the payload channel is absolute.** An address-only rescale walks the
    store across its own payload wall, which is why it moved arm A's ``A1`` by
    −0.125 at ``a = 0.8`` and moved the *rig* (self-probe ``acq`` 0.4844 → 0.3203,
    ``G-DEC`` 0.1484 → 0.1094, ``G-DRIFT`` ratio ×3.8) — a finding about the rig,
    not a leg failure.

    The legal rescale is **FULL-STATE co-scaling**: address **and** payload
    together. On this rig that is exactly two knobs — ``addr_scale_mult x= a``
    (the address ball) and ``payload_scale /= a`` (payloads are
    ``(label - 4.5) / payload_scale``, so dividing the scale multiplies the
    payload). Everything else geometric is **measured** from the keys and
    co-scales for free (``d_safe``, the arms' co-scaled atom widths, the G-ADDR
    cue jitter, the codebook spacing).

    Returns a deep copy; ``cfg`` is untouched. Feed the resulting legs to
    :func:`chlu.core.well_lifecycle.scale_guard`, whose pass condition is
    **verdict stability**, not bounded metric movement.
    """
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"full-state co-scaling needs alpha > 0, got {alpha!r}")
    out = copy.deepcopy(cfg)
    w = out.experiment_well_lifecycle
    w.addr_scale_mult = float(w.addr_scale_mult) * a
    w.payload_scale = float(w.payload_scale) / a
    return out


def _query_batch(stream, t: int, n: int, seed: int):
    """Held-out test queries from every task seen so far (a Class-IL read)."""
    rng = np.random.default_rng(int(seed) + 7717)
    Xs, ys = [], []
    for k in range(t + 1):
        Xs.append(np.asarray(stream["test_X"][k], dtype=np.float32))
        ys.append(np.asarray(stream["test_y"][k], dtype=int))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    idx = rng.choice(len(X), size=int(min(n, len(X))), replace=False)
    return X[idx], y[idx]


def _read_acc(res, system, y_true, payload_scale: float) -> float:
    """Class accuracy of the settled read-out (nearest label in payload space)."""
    val = np.asarray(res.value).reshape(len(y_true), -1)[:, 0]
    labels = np.arange(10)
    pays = (labels - 4.5) / float(payload_scale)
    pred = labels[np.argmin(np.abs(val[:, None] - pays[None, :]), axis=1)]
    return float(np.mean(pred == np.asarray(y_true)))


def _trip_state(system) -> Dict[str, Any]:
    """Monitor trip state, with #9 and #12 named — a starved write reads as a
    capacity result when it is not (w26)."""
    try:
        readings = system.observe(stage="census")
    except Exception as e:  # a monitor must never break a measurement
        return {"error": repr(e)}
    out = {}
    for r in readings:
        out[str(r.name)] = {
            "mode": int(getattr(r, "mode", -1)),
            "tripped": bool(r.tripped),
            "applicable": bool(getattr(r, "applicable", True)),
            "value": _jsonable(getattr(r, "value", None)),
        }
    return out


def _flag_table(cfg: CHLUConfig, system, seed: int, d_safe: float) -> Dict[str, Any]:
    w = cfg.experiment_well_lifecycle
    return {
        "seed": int(seed),
        "clu_system_non_defaults": system.cfg.as_flag_table(),
        "addr_dim": int(w.addr_dim), "phi_arm": w.phi_arm,
        "phi_regime": w.phi_regime, "phi_dim": int(w.addr_dim),
        "well_budget": int(w.well_budget), "capacity": int(w.capacity),
        "leak": float(w.leak), "stage_lifetimes": True,
        "d_safe_override": float(d_safe), "write_steps": int(w.write_steps),
        "read_steps": int(w.read_steps), "address_steps": int(w.address_steps),
        "kinetic_mode": system.cfg.kinetic_mode,
        "n_atoms": int(system.cfg.n_atoms),
        "payload_scale": float(w.payload_scale),
        "promotable": False,
        "why_not_promotable": (
            "census cell: phi_dim = addr_dim = %d is below the CL entry's binding "
            "phi_dim >= 16 (ERRATA-C2W8 §3: the learned store is measurably inert "
            "at 16 on this rig), and the store runs at demoted write budget"
            % int(w.addr_dim)
        ),
    }


def _byte_ledger(system, launder, addr_dim: int) -> Dict[str, int]:
    """Every arm's bytes, launder included (§3.5). ``gamma_phi`` holes would be
    added here the moment the trash region ships — a trash region off the ledger
    is a hidden capacity increase (§A9.6)."""
    n_knn = len(getattr(launder, "keys", []))
    trash = int(system.trash_bytes())
    # ⭐ C2W8 pass 2 arm B: head parameters are bytes and must not be silently
    # booked as "codebook". 0 with the flag off => this ledger is unchanged.
    head = int(getattr(system, "emission_bytes", lambda: 0)())
    return {
        "clu_store_bytes": int(system.store.n_bytes()),
        "clu_codebook_bytes": int(system.n_bytes() - system.store.n_bytes()
                                  - trash - head),
        "emission_head_bytes": head,
        "clu_total_bytes": int(system.n_bytes()),
        "knn_launder_bytes": int(n_knn * (addr_dim + 1) * 4),
        # ⚠ the trash region's holes ARE bytes; 0 only because no hole is placed
        # in the census (the routing verb is stage 2's).
        "gamma_phi_hole_bytes": trash,
        "gamma_phi_enabled": bool(system.cfg.gamma_phi),
    }


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


# ---------------------------------------------------------------------------
# the stage-1 deliverable
# ---------------------------------------------------------------------------
def run_experiment_well_lifecycle(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "plots",
    seeds: Optional[List[int]] = None,
    quick: bool = False,
    data=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the census over every seed and write ``census.json``.

    ⛔ The returned ``stage2_unlock`` is computed by :func:`unlock_verdict`, i.e.
    by the prereg's arithmetic — never by a judgement call at review time.
    """
    cfg = config or get_default_config()
    if quick:
        apply_quick(cfg)
    w = cfg.experiment_well_lifecycle
    seeds = [int(s) for s in (seeds if seeds is not None else w.seeds)]

    cells = []
    for s in seeds:
        if verbose:
            print(f"[well-lifecycle] census seed {s} ...", flush=True)
        cells.append(run_census_cell(cfg, s, data=data, verbose=verbose))

    verdict = unlock_verdict([c["census"]["P"] for c in cells],
                             [c["census"]["M"] for c in cells])
    legs = [c["g_addr"] for c in cells if isinstance(c.get("g_addr", {}).get("A1"), dict)]
    results = {
        "experiment": "well_lifecycle_census",
        "stage": 1,
        "seeds": seeds,
        "cells": cells,
        "k1": verdict,
        "stage2_unlock": bool(verdict["stage2_unlock"]),
        # ⭐ C2W8 pass 3: the arm-level G-ADDR verdict, computed mechanically
        "g_addr": (gate_addr_verdict(legs) if legs else
                   {"status": "NOT RUN — run_gate_addr = False (declared)"}),
        # ⭐ C2W8 close-out: the hardened legs, per seed, at the top level
        "g_drift_two_sided_by_seed": [c["census"]["G_DRIFT_two_sided"] for c in cells],
        "P_comparability_by_seed": [c["census"]["P_comparability"] for c in cells],
        "atom_width_selection_by_seed": [c["atom_width_selection"] for c in cells],
        "overdig_by_seed": [c["census"]["overdig"] for c in cells],
        "depth_raw_median_by_seed": [c["census"]["depth_raw_median"] for c in cells],
        "depth_netted_median_by_seed": [c["census"]["depth_netted_median"] for c in cells],
        "theta_att_by_seed": [c["census"]["theta_att_block"]["theta_att"] for c in cells],
        "declared_not_runs": [
            "I2 correlation test (deferred to C2W10, §A23.5) — NOT RUN, not a null",
            "cross-stream / persistent-store criterion (C2W10)",
            "tier-ii verdict, full-CLU verdict (§A28.4)",
            "stage-2 verbs (merge / prune / gamma_phi) — gated on stage2_unlock",
        ],
        "wall_s": float(sum(c["wall_s"] for c in cells)),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "census.json")
    with open(path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    results["census_json"] = path
    if verbose:
        print(f"\n[well-lifecycle] K1: {verdict['rule']}\n"
              f"  P_mean={verdict['P_mean']:.4f} M_mean={verdict['M_mean']:.4f} "
              f"-> stage2_unlock={verdict['stage2_unlock']} kill={verdict['kill']}\n"
              f"  wrote {path}", flush=True)
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Smoke mode: a real census on a tiny stream (never a claim cell)."""
    w = config.experiment_well_lifecycle
    w.quick = True
    w.seeds = [0]
    w.addr_dim = 4
    w.capacity = 6
    w.well_budget = 3
    w.n_offer_per_task = 4
    w.read_batch = 6
    w.read_every = 2
    w.write_steps = 40
    w.read_steps = 120
    w.address_steps = 60
    w.n_query_per_item = 2
    w.capture_dirs = 6
    w.capture_bisect_steps = 4
    w.loo_repeats = 2
    w.gaddr_n_query_per_item = 4
    cl = config.experiment_cl_entry
    cl.n_tasks = 3
    cl.n_train_per_task = 60
    cl.n_test_per_task = 40
    cl.n_fit_region = 600
    cl.n_fit_pool = 200


def main():
    parser = argparse.ArgumentParser(description="C2W8 well-lifecycle census (stage 1)")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="e.g. 0,1,2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save-dir", type=str, default="plots")
    args = parser.parse_args()

    config = get_default_config()
    save_dir = args.save_dir
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        paths = pm.get_paths(args.project)
        save_dir = str(paths["plots"])
        cfg_path = paths["config"] / "config.yaml"
        if cfg_path.exists():
            from chlu.config import load_config

            config = load_config(cfg_path)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else None)
    run_experiment_well_lifecycle(config, save_dir=save_dir, seeds=seeds,
                                  quick=args.quick)


if __name__ == "__main__":
    main()


__all__ = [
    "PhiAddress", "cl_config", "store_config", "label_to_payload",
    "run_census_cell", "run_experiment_well_lifecycle", "apply_quick",
    # ⭐ C2W8 close-out (charter §A32.3)
    "UnselectedAtomWidth", "full_state_coscaled_config",
]
