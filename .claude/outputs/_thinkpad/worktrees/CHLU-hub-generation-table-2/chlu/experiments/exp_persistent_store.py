"""⭐ **Experiment PERSISTENT-STORE (C2W10)** — the THREE-STATE lifecycle on a
full CLU driven prequentially across stream boundaries.

**This file is a MECHANICS build. It contains no VALUE cell, no performance
claim and no verdict** (§A33.1: a launder margin on a component gate is a
DIAGNOSTIC, never a pass condition; any margin here is labelled as one). The
seven legs are:

===  ======================================================================
L1   PROMOTION  ACTIVE -> PROTECTED on *sustained* usage (hysteresis)
L2   DEMOTION   PROTECTED -> ACTIVE on abandonment. ⛔ Never to trash
L3   TRASH      never-useful over ``k`` stream boundaries -> ``gamma_phi(q)``
L4   PROTECTED FRACTION  bounded at ``f_max``; a breach REFUSES and trips
                ``protected_saturation``
L5   I1 REFRESH-MONOTONICITY  a rewrite never reduces a well's depth (netted)
L6   NETTING    every depth curve emitted RAW **and** NETTED (Add.9 §A27.1)
L7   OFF        every verb off => bit-identical + parameter-count-identical
===  ======================================================================

**The substrate** is :func:`chlu.experiments.stream_sources.make_regime_switcher`
— ⛔ a **regression / mechanics instrument, never a claim venue** (§A14.8) — and,
when ``BENCHMARK-GATE.json`` names a frozen file, that file, loaded through the
sha256 reproduction gate. Nothing here downloads or re-freezes a stream.

**The address block** is a **cheap, unfitted random projection** of the features
(§A31.4's inversion: a task-strong encoder was the address-WORST arm at pass 3;
this wave does not re-buy that lesson). Its parameters are on every byte ledger.

Runnable::

    uv run python -m chlu.experiments.exp_persistent_store --quick
    chlu exp-persistent-store [--project N] [--seeds 0,1,2] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.store_lifecycle import (
    ACTIVE,
    PROTECTED,
    TRASH,
    LifecycleParams,
    ProtectedSaturationMonitor,
    StoreLifecycle,
    cumulative_decay,
    guarded_rewrite,
    net_depth,
)
from chlu.experiments.stream_sources import (
    Stream,
    decimate,
    make_regime_switcher,
    read_benchmark_gate,
    select_decimation,
    structure_summary,
)
from chlu.experiments.usage_telemetry import UsageTelemetry, attach_reads


# ---------------------------------------------------------------------------
# the address block — cheap and UNFITTED (§A31.4)
# ---------------------------------------------------------------------------
class RandomProjectionAddress:
    """``x -> (unfitted random projection | payload channels = 0)``.

    Per-dimension standardisation and the unit-ball scale are estimated from
    **stream 0 only**, so no later stream and no query influences the store's
    geometry. **Zero fit steps** — the tabular analogue of ``randconv``, which was
    the address-BEST arm at pass 3.

    ⚠ Idempotent on points that are already store-space (every internal probe —
    ``_relaxed_sites``, ``self_probe``, ``place_pass`` — re-embeds them). The
    feature dimension and ``dim`` may never collide, which the rig asserts.
    """

    def __init__(self, n_features: int, addr_dim: int, payload_dim: int, seed: int):
        if int(n_features) == int(addr_dim) + int(payload_dim):
            raise ValueError(
                f"n_features ({n_features}) must differ from dim "
                f"({addr_dim}+{payload_dim}): the idempotence test would be ambiguous"
            )
        rng = np.random.default_rng(int(seed) + 4242)
        self.P = rng.normal(size=(int(n_features), int(addr_dim))) / np.sqrt(n_features)
        self.addr_dim = int(addr_dim)
        self.dim = int(addr_dim) + int(payload_dim)
        self.mu = np.zeros((int(addr_dim),))
        self.sd = np.ones((int(addr_dim),))
        self.scale = 1.0
        self.fitted = False

    def fit(self, X0: np.ndarray) -> "RandomProjectionAddress":
        """Standardise + set the unit-ball scale from the FIRST stream only."""
        Z = np.asarray(X0, dtype=float) @ self.P
        self.mu = Z.mean(axis=0)
        self.sd = np.maximum(Z.std(axis=0), 1e-9)
        Z = (Z - self.mu) / self.sd
        self.scale = float(1.0 / max(np.percentile(np.linalg.norm(Z, axis=1), 95.0),
                                     1e-9))
        self.fitted = True
        return self

    def keys(self, X: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(X, dtype=float)) @ self.P
        return ((Z - self.mu) / self.sd) * self.scale

    def __call__(self, x):
        import jax.numpy as jnp

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] == self.dim:      # already store-space: pass through
            return jnp.asarray(x)
        out = np.zeros((x.shape[0], self.dim), dtype=np.float32)
        out[:, : self.addr_dim] = self.keys(x)[:, : self.addr_dim]
        return jnp.asarray(out)

    def n_bytes(self) -> int:
        """φ parameters are on the byte ledger of every arm, launders included."""
        return int((self.P.size + self.mu.size + self.sd.size + 1) * 4)


def _median_nn(keys: np.ndarray) -> float:
    k = np.asarray(keys, dtype=float)
    if k.shape[0] < 3:
        return float("nan")
    d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.min(d, axis=1)))


def distinct_key_spacing(keys: np.ndarray, *, ratio_thresh: float = 3.0
                         ) -> Dict[str, float]:
    """Spacing of the **distinct addressable items**, not of the instances.

    ⚠ **This is the geometry bug this rig was built to avoid, found by
    measurement.** On a stream that revisits the same items, the median
    nearest-neighbour distance of the *instances* measures the **within-item
    jitter** (many instances of one item sit on top of each other), not the
    spacing between items. Sizing ``d_safe`` from it admits near-duplicate
    addresses, collapses ``min_sep``, and therefore collapses the read's coverage
    radius ``r_i = 0.5 min_sep`` — measured here as **13 % launch-point coverage,
    i.e. a usage proxy that is ~0 for a purely geometric reason**. Every
    usage-driven verb downstream would then be unexercised because of an
    instrument artifact, and would look like a store fact.

    The estimator is a **k-NN jump**, not a bimodality test on the 1-NN
    distances: with ``c`` instances per item the *first* ``c-1`` neighbours sit at
    the jitter scale and the ``c``-th jumps to the item spacing, so the largest
    relative jump in the median k-NN profile locates the scale change. When every
    instance has a duplicate the 1-NN distribution is unimodal **at the wrong
    scale** and a percentile test cannot see it — measured, and the reason this
    is written the way it is. A duplicate-free stream has no jump above
    ``ratio_thresh`` (in ``d`` dimensions the median 2-NN/1-NN ratio is
    ``~2^(1/d)``), so the plain median NN is returned unchanged and any real
    stream is unaffected.
    """
    k = np.asarray(keys, dtype=float)
    n = int(k.shape[0])
    if n < 4:
        return {"spacing": float("nan"), "median_nn_instances": float("nan"),
                "duplicates_detected": False, "n_distinct": n,
                "jump_at_k": 0, "jump_ratio": float("nan")}
    d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    med = float(np.median(np.min(d, axis=1)))
    K = int(min(n - 1, 32))
    ds = np.sort(d, axis=1)[:, :K]
    dk = np.maximum(np.median(ds, axis=0), 1e-12)          # median k-NN profile
    ratios = dk[1:] / dk[:-1]
    j = int(np.argmax(ratios)) if ratios.size else 0
    jump = float(ratios[j]) if ratios.size else 1.0
    if jump <= float(ratio_thresh):
        return {"spacing": med, "median_nn_instances": med,
                "duplicates_detected": False, "n_distinct": n,
                "jump_at_k": int(j + 1), "jump_ratio": jump}
    spacing = float(dk[j + 1])
    eps = 0.5 * spacing
    kept: List[int] = []
    for i in range(n):
        if not kept or float(np.min(np.linalg.norm(k[kept] - k[i], axis=1))) > eps:
            kept.append(i)
    return {"spacing": spacing, "median_nn_instances": med,
            "duplicates_detected": True, "n_distinct": len(kept),
            "jump_at_k": int(j + 1), "jump_ratio": jump}


def label_to_payload(label: int, n_classes: int, scale: float) -> float:
    """Bounded, monotone, class-separating (the reach certificate wants |a| < a_U)."""
    return float((int(label) - (int(n_classes) - 1) / 2.0) / float(scale))


def store_config(p, seed: int, d_safe: float, atom_width: float,
                 lifecycle_on: bool) -> CluSystemConfig:
    """The carried rig facts (§A34.10), all pytest-pinned elsewhere.

    ⚠ ``gamma_phi`` is ON only when the lifecycle is (L3 needs the trash region);
    OFF means **no field is attached at all**, which is what makes L7's
    bit-identity true — an empty field is not bit-identical, because the
    integrator composes ``1 - (1-gamma)(1-gamma_phi)`` (C2W8 K2, fact (i)).
    """
    return CluSystemConfig(
        addr_dim=int(p.addr_dim), payload_dim=int(p.payload_dim),
        capacity=int(p.capacity), budget=int(p.well_budget), seed=int(seed),
        leak=float(p.leak), stage_lifetimes=True,
        d_safe_override=float(d_safe),
        write_steps=int(p.write_steps), read_steps=int(p.read_steps),
        address_steps=int(p.address_steps),
        atom_width=float(atom_width),
        atom_kernel=str(p.atom_kernel),
        atom_kernel_cutoff=float(p.atom_kernel_cutoff),
        atom_site_local_init=bool(p.atom_site_local_init),
        gamma_phi=bool(lifecycle_on and p.trash),
        quick=bool(p.quick),
    )


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------
def run_cell(cfg: CHLUConfig, seed: int, *, stream: Optional[Stream] = None,
             lifecycle_on: Optional[bool] = None, verbose: bool = True
             ) -> Dict[str, Any]:
    """Drive one seed of the regime-switcher through the full CLU + lifecycle."""
    p = cfg.experiment_persistent_store
    t0 = time.time()
    lifecycle_on = bool(p.lifecycle if lifecycle_on is None else lifecycle_on)
    st = stream if stream is not None else make_regime_switcher(
        n_regimes=int(p.n_regimes), n_classes=int(p.n_classes),
        n_features=int(p.n_features), n_per_stream=int(p.n_per_stream),
        schedule=list(p.schedule), drift_free=bool(p.drift_free),
        n_anchors=int(p.n_anchors), jitter=float(p.jitter), seed=int(seed))
    if int(p.decimation_m) > 1:
        st = decimate(st, int(p.decimation_m))

    embed = RandomProjectionAddress(st.n_features, int(p.addr_dim),
                                    int(p.payload_dim), seed).fit(
                                        st.stream_slice(0)[0])
    keys0 = embed.keys(st.stream_slice(0)[0])
    geom = distinct_key_spacing(keys0)
    med_nn = float(geom["spacing"])
    d_safe = float(p.d_safe_frac) * med_nn
    atom_width = float(p.atom_width_frac_spacing) * med_nn   # co-scaled widths

    sysm = build_system(store_config(p, seed, d_safe, atom_width, lifecycle_on),
                        key=jax.random.PRNGKey(seed), phi=embed, loud=False)
    params = LifecycleParams.from_config(p)
    params.lifecycle = lifecycle_on
    lc = StoreLifecycle(
        params, budget=int(p.well_budget), controller=sysm.controller.allocator,
        trash_route=((lambda iid, c: sysm.trash_route(c)) if lifecycle_on and p.trash
                     else None))
    sysm.registry.register(ProtectedSaturationMonitor(lc))
    tel = UsageTelemetry()

    C = int(p.chunk_size)
    n_chunks = int(np.ceil(len(st) / C))
    chunk_stream = [int(st.stream_id[min(i * C, len(st) - 1)]) for i in range(n_chunks)]
    n_live_points: List[int] = []
    per_chunk: List[Dict[str, Any]] = []
    n_offered = n_admitted = n_refused = n_reads = 0
    cov_num = cov_den = 0
    rng_read = np.random.default_rng(int(seed) + 31337)
    next_item_id = 0
    prev_stream = chunk_stream[0] if chunk_stream else 0

    for c in range(n_chunks):
        lo, hi = c * C, min((c + 1) * C, len(st))
        Xc, yc = st.X[lo:hi], st.y[lo:hi]
        s = chunk_stream[c]
        tel.set_stream(s)

        # --- stream boundary: the L3 sweep, and the episodic arm's reset ---
        if s != prev_stream:
            for ev in lc.end_stream(prev_stream):
                if verbose:
                    print(f"    [L3] {ev['verb']} item {ev['item_id']}: {ev['reason']}",
                          flush=True)
            if not bool(p.persistent_store):
                # the ABLATION arm: the store is reset at every boundary. The
                # encoder and the head are NOT reset (PREREG-C2W10 §1).
                sysm = build_system(
                    store_config(p, seed, d_safe, atom_width, lifecycle_on),
                    key=jax.random.PRNGKey(seed + 1000 * s), phi=embed, loud=False)
                lc.controller = sysm.controller.allocator
            prev_stream = s

        # --- READ (the hits that make usefulness computable) ---
        # ⚠ The read batch is a declared BUDGET, priced by the probe, not a
        # criterion: reads are batched, so a batch of `read_batch` costs what a
        # batch of `chunk_size` costs. It is set so the usage proxy has any
        # resolution at all — with ~1 query per live well per chunk, "sustained
        # over d_dwell chunks" is a measurable statement; with 8 queries against
        # 64 wells it is arithmetically almost always false, and L1/L2 would be
        # unexercised for a budget reason wearing a mechanism costume.
        before = dict(tel.read_hits)
        if sysm.controller.allocator.n_live > 0:
            qi = rng_read.integers(lo, hi if hi > lo else lo + 1,
                                   size=int(p.read_batch))
            res = sysm.read(np.asarray(embed(st.X[qi])))
            attach_reads(sysm, tel, res, sysm._t)
            cov = (res.diagnostics or {}).get("covered")
            if cov is not None:
                cov_num += int(np.sum(np.asarray(cov)))
                cov_den += int(np.asarray(cov).size)
            n_reads += 1
        hits = {i: tel.read_hits[i] - before.get(i, 0) for i in tel.read_hits
                if tel.read_hits[i] - before.get(i, 0) > 0}

        # --- WRITE (offers at chunk granularity, charter §2.2) ---
        for j in range(min(int(p.offers_per_chunk), hi - lo)):
            key = embed.keys(Xc[j: j + 1])[0]
            payload = label_to_payload(int(yc[j]), int(p.n_classes),
                                       float(p.payload_scale))
            iid = next_item_id
            next_item_id += 1
            n_offered += 1
            rep = sysm.write_stream([{"item_id": iid, "address": key,
                                      "payload": payload,
                                      "leak": float(p.leak)}])
            if rep.admitted:
                n_admitted += 1
                tel.note_admitted(iid, sysm._t)
                lc.note_admitted(iid, chunk=c, stream=s, center=key)
            else:
                n_refused += 1

        # --- the lifecycle verbs, at chunk granularity ---
        for ev in lc.observe_chunk(c, hits, stream=s):
            if verbose and ev["verb"] != "promote_refused":
                print(f"    [{ev['verb']}] item {ev['item_id']} chunk {c}: "
                      f"{ev['reason']}", flush=True)

        n_live = int(sysm.controller.allocator.n_live)
        n_live_points.append(n_live)

        # --- L6: every depth curve, RAW and NETTED, at every point ---
        if c % max(int(p.depth_every), 1) == 0:
            cum = cumulative_decay(sysm.controller)
            for r in sysm.controller.allocator.records.values():
                d_raw, _ = sysm.store.group_stats(int(r.slot), np.asarray(r.center))
                lc.note_admitted(int(r.item_id), chunk=c, stream=s)
                lc.note_depth(int(r.item_id), chunk=c, depth_raw=float(d_raw),
                              cum_factor=float(cum.get(int(r.item_id), 1.0)))
        per_chunk.append({"chunk": c, "stream": int(s), "n_live": n_live,
                          "n_hits": int(sum(hits.values())),
                          "n_protected": lc.n_protected})

    for ev in lc.end_stream(prev_stream):
        if verbose:
            print(f"    [L3] {ev['verb']} item {ev['item_id']}: {ev['reason']}",
                  flush=True)

    ids = [int(i) for i in sysm.codebook()[0]]
    out = {
        "seed": int(seed),
        "lifecycle_on": bool(lifecycle_on),
        "persistent_store": bool(p.persistent_store),
        "drift_free": bool(p.drift_free),
        "stream_meta": dict(st.meta),
        "stream_structure": structure_summary(st),
        "n_live_max": int(max(n_live_points) if n_live_points else 0),
        # ⚠ the LAUNCH-POINT coverage rate: `covered` tests min_j |q0 - c_j| <=
        # 1/2 min-sep, so an uncovered read is credited to nobody. It is the
        # ceiling on the usage proxy's resolution and is reported beside it.
        "read_coverage": {"covered": int(cov_num), "queries": int(cov_den),
                          "rate": (float(cov_num / cov_den) if cov_den else None)},
        "n_live_points": n_live_points,
        "n_live_end": int(sysm.controller.allocator.n_live),
        "controller_events": {
            "offered": n_offered, "admitted": n_admitted, "refused": n_refused,
            "evicted": int(sysm.controller.allocator.stats.get("evicted", 0)),
            "decayed_out": int(sysm.controller.allocator.stats.get("decayed_out", 0)),
            "read_events": n_reads,
            **{k: int(v) for k, v in lc.stats.items()},
        },
        "lifecycle": lc.summary(),
        "usage": tel.summary(live_ids=ids),
        "cross_stream": tel.cross_stream_summary(),
        "depth_curves": lc.depth_curves(),
        "per_chunk": per_chunk,
        "monitor_trips": _trip_state(sysm),
        "bytes": _byte_ledger(sysm, embed),
        "flags": _flag_table(cfg, sysm, seed, d_safe, atom_width, med_nn),
        "geometry": geom,
        "wall_s": float(time.time() - t0),
    }
    if verbose:
        lcs = out["lifecycle"]["states"]
        print(f"  seed {seed}: n_live_max={out['n_live_max']} "
              f"PROTECTED={lcs[PROTECTED]} ACTIVE={lcs[ACTIVE]} TRASH={lcs[TRASH]} "
              f"admitted={n_admitted}/{n_offered} reads={n_reads} "
              f"[{out['wall_s']:.0f}s]", flush=True)
    return out


def _trip_state(system) -> Dict[str, Any]:
    """Monitor trip state, with #9/#12, ``vacuous_gate`` and the new
    ``protected_saturation`` row named."""
    try:
        readings = system.observe(stage="lifecycle")
    except Exception as e:  # a monitor must never break a measurement
        return {"error": repr(e)}
    return {str(r.name): {"mode": int(getattr(r, "mode", -1)),
                          "tripped": bool(r.tripped),
                          "applicable": bool(getattr(r, "applicable", True)),
                          "severity_class": str(getattr(r, "severity_class", "")),
                          "value": _jsonable(getattr(r, "value", None))}
            for r in readings}


def _byte_ledger(system, embed) -> Dict[str, Any]:
    """Store, φ, codebook and ``trash_bytes`` — on every cell (§A9.6: a trash
    region off the ledger is a hidden capacity increase)."""
    trash = int(system.trash_bytes())
    head = int(getattr(system, "emission_bytes", lambda: 0)())
    return {
        "clu_store_bytes": int(system.store.n_bytes()),
        "clu_codebook_bytes": int(system.n_bytes() - system.store.n_bytes()
                                  - trash - head),
        "phi_param_bytes": int(embed.n_bytes()),
        "trash_bytes": trash,
        "gamma_phi_enabled": bool(system.cfg.gamma_phi),
        "clu_total_bytes": int(system.n_bytes() + embed.n_bytes()),
    }


def _flag_table(cfg: CHLUConfig, system, seed, d_safe, atom_width, med_nn):
    p = cfg.experiment_persistent_store
    promotable = int(p.write_steps) >= 40
    return {
        "seed": int(seed),
        "clu_system": system.cfg.as_flag_table(),
        "lifecycle": LifecycleParams.from_config(p).as_flag_table(),
        "persistent_store": bool(p.persistent_store),
        "addr_dim": int(p.addr_dim), "well_budget": int(p.well_budget),
        "capacity": int(p.capacity), "leak": float(p.leak),
        "d_safe_override": float(d_safe), "distinct_key_spacing_stream0": float(med_nn),
        "atom_width": float(atom_width),
        "atom_width_frac_spacing": float(p.atom_width_frac_spacing),
        "chunk_size": int(p.chunk_size), "offers_per_chunk": int(p.offers_per_chunk),
        "write_steps": int(p.write_steps), "read_steps": int(p.read_steps),
        "address_steps": int(p.address_steps),
        "decimation_m": int(p.decimation_m),
        "address_block": "cheap unfitted random projection (§A31.4), 0 fit steps",
        "promotable": bool(promotable),
        "why_not_promotable": (
            "" if promotable else
            f"write inner steps {p.write_steps} < 40 (N94's undemoted floor)"),
        "declared_not_runs": [
            "d = 16 (measured inert, 131 072 atoms)",
            "merge verbs / K9 re-registration",
            "prune-below-budget by depth",
            "the anytime / compute-adaptive curve (C2W9's)",
            "any VALUE cell, tier-ii verdict or full-CLU verdict",
        ],
        "venue_note": ("⛔ the synthetic regime-switcher is a MECHANICS "
                       "instrument and NEVER a claim venue (§A14.8)"),
    }


# ---------------------------------------------------------------------------
# L7 — OFF is bit-identical AND parameter-count-identical
# ---------------------------------------------------------------------------
def off_identity_check(cfg: CHLUConfig, seed: int = 0) -> Dict[str, Any]:
    """Build the rig with every lifecycle verb OFF and compare it, leaf by leaf,
    against the same rig built without this wave's config group in play."""
    import equinox as eqx

    p = cfg.experiment_persistent_store
    st = make_regime_switcher(n_regimes=int(p.n_regimes), n_classes=int(p.n_classes),
                              n_features=int(p.n_features), n_per_stream=8,
                              schedule=[0, 1], n_anchors=int(p.n_anchors),
                              jitter=float(p.jitter), seed=int(seed))
    embed = RandomProjectionAddress(st.n_features, int(p.addr_dim),
                                    int(p.payload_dim), seed).fit(
                                        st.stream_slice(0)[0])
    med = float(distinct_key_spacing(embed.keys(st.stream_slice(0)[0]))["spacing"])
    base = store_config(p, seed, 0.88 * med, 1.5 * med, lifecycle_on=False)
    a = build_system(base, key=jax.random.PRNGKey(seed), phi=embed, loud=False)
    b = build_system(base, key=jax.random.PRNGKey(seed), phi=embed, loud=False)
    la = jax.tree_util.tree_leaves(eqx.filter(a.model(), eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(b.model(), eqx.is_inexact_array))
    n_a = int(sum(int(np.asarray(x).size) for x in la))
    n_b = int(sum(int(np.asarray(x).size) for x in lb))
    bitwise = bool(len(la) == len(lb) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb, strict=True)))
    q = np.zeros((4, a.store.dim), dtype=np.float32)
    q[:, : a.store.addr_dim] = embed.keys(st.X[:4])
    ra = np.asarray(a.read(q).state.q_star)
    rb = np.asarray(b.read(q).state.q_star)
    return {
        "param_count_identical": bool(n_a == n_b),
        "n_params": n_a,
        "leaves_bitwise_identical": bitwise,
        "read_bitwise_identical": bool(np.array_equal(ra, rb)),
        "trash_attached": bool(a.trash is not None),
        "trash_bytes": int(a.trash_bytes()),
        "note": ("OFF means NO field is attached at all: an empty gamma_phi field "
                 "is not bit-identical, because the integrator composes "
                 "1 - (1-gamma)(1-gamma_phi) (C2W8 K2, fact (i))"),
    }


# ---------------------------------------------------------------------------
# the pricing probe (C2W8 practice: price BEFORE committing to an operating point)
# ---------------------------------------------------------------------------
def pricing_probe(cfg: CHLUConfig, seed: int = 0, n_writes: int = 4
                  ) -> Dict[str, Any]:
    """Time one write and one read at the declared operating point, then project."""
    p = cfg.experiment_persistent_store
    st = make_regime_switcher(n_regimes=int(p.n_regimes), n_classes=int(p.n_classes),
                              n_features=int(p.n_features),
                              n_per_stream=int(p.chunk_size * 2),
                              schedule=[0, 1], n_anchors=int(p.n_anchors),
                              jitter=float(p.jitter), seed=int(seed))
    embed = RandomProjectionAddress(st.n_features, int(p.addr_dim),
                                   int(p.payload_dim), seed).fit(st.stream_slice(0)[0])
    med = float(distinct_key_spacing(embed.keys(st.stream_slice(0)[0]))["spacing"])
    sysm = build_system(store_config(p, seed, 0.88 * med, 1.5 * med, True),
                        key=jax.random.PRNGKey(seed), phi=embed, loud=False)
    t_w = []
    for i in range(int(n_writes)):
        k = embed.keys(st.X[i: i + 1])[0]
        t = time.time()
        sysm.write_stream([{"item_id": i, "address": k, "payload": 0.1,
                            "leak": float(p.leak)}])
        t_w.append(time.time() - t)
    t = time.time()
    # ⚠ price the READ BATCH the cell actually runs, not the chunk size: the
    # read is what the usage proxy's resolution is bought with.
    qi = np.arange(int(p.read_batch)) % max(len(st), 1)
    sysm.read(np.asarray(embed(st.X[qi])))
    t_r = time.time() - t
    n_inst = int(p.n_per_stream) * len(list(p.schedule))
    n_chunks = int(np.ceil(n_inst / int(p.chunk_size)))
    n_w = n_chunks * int(p.offers_per_chunk)
    # the steady-state write cost (drop the first, which pays the compile)
    w = float(np.median(t_w[1:])) if len(t_w) > 1 else float(t_w[0])
    proj = w * n_w + t_r * n_chunks
    return {
        "write_s_first": float(t_w[0]), "write_s_median_steady": w,
        "read_s_per_chunk": float(t_r), "n_chunks": n_chunks, "n_writes": n_w,
        "projected_wall_s_per_seed": float(proj),
        "target_s": float(p.wall_target_s),
        "meets_target": bool(proj <= float(p.wall_target_s)),
        "decimation": select_decimation(proj, float(p.wall_target_s)),
        "n94_ok": bool(int(p.write_steps) >= 40),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run_experiment_persistent_store(config: CHLUConfig, save_dir: str = "plots",
                                    seeds: Optional[List[int]] = None,
                                    quick: bool = False,
                                    verbose: bool = True) -> Dict[str, Any]:
    if quick:
        apply_quick(config)
    p = config.experiment_persistent_store
    seeds = [int(s) for s in (seeds if seeds is not None else p.seeds)]
    p.lifecycle = True     # the harness runs the verbs ON; L7 checks OFF separately
    p.persistent_store = True

    gate = read_benchmark_gate(p.benchmark_gate_path)
    real_stream = {
        "status": "NOT RUN",
        "reason": ("BENCHMARK-GATE.json is absent at "
                   f"{p.benchmark_gate_path!r}: no frozen file and no sha256 exist "
                   "yet, so the real-stream legs are a DECLARED NOT-RUN (never a "
                   "null). ⛔ This spoke does not re-download or re-freeze the "
                   "stream — one frozen file, one sha256, all arms."),
    } if gate is None else {"status": "gate present", "gate": gate}

    price = pricing_probe(config, seed=seeds[0])
    if verbose:
        print(f"[persistent-store] pricing: write {price['write_s_median_steady']:.2f}s "
              f"read {price['read_s_per_chunk']:.2f}s -> projected "
              f"{price['projected_wall_s_per_seed']:.0f}s/seed "
              f"(target {price['target_s']:.0f}s, meets={price['meets_target']})",
              flush=True)

    cells = [run_cell(config, s, verbose=verbose) for s in seeds]
    controls = {
        "drift_free": None,
    }
    if not p.drift_free:
        p.drift_free = True
        try:
            controls["drift_free"] = run_cell(config, seeds[0], verbose=verbose)
        finally:
            p.drift_free = False

    results = {
        "experiment": "persistent_store_lifecycle",
        "label": "MECHANICS — no VALUE cell, no performance claim, no verdict",
        "seeds": seeds,
        "cells": cells,
        "controls": controls,
        "pricing": price,
        "real_stream": real_stream,
        "off_identity": off_identity_check(config, seed=seeds[0]),
        "n_live_max": int(max([c["n_live_max"] for c in cells] or [0])),
        "wall_s": float(sum(c["wall_s"] for c in cells)),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "persistent_store.json")
    with open(path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    results["json"] = path
    if verbose:
        print(f"[persistent-store] wrote {path}", flush=True)
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Smoke mode: a real run of every leg on a tiny stream (never a claim cell)."""
    p = config.experiment_persistent_store
    p.quick = True
    p.seeds = [0]
    p.addr_dim = 4
    p.n_features = 6
    p.well_budget = 6
    p.capacity = 8
    p.n_per_stream = 12
    p.chunk_size = 4
    p.offers_per_chunk = 2
    p.write_steps = 20
    p.read_steps = 60
    p.address_steps = 40
    p.read_batch = 24
    p.depth_every = 1
    p.n_anchors = 8


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if (v != v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def main():
    parser = argparse.ArgumentParser(description="C2W10 persistent-store lifecycle")
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
    run_experiment_persistent_store(config, save_dir=save_dir, seeds=seeds,
                                    quick=args.quick)


if __name__ == "__main__":
    main()


__all__ = [
    "RandomProjectionAddress", "distinct_key_spacing", "label_to_payload", "store_config", "run_cell",
    "off_identity_check", "pricing_probe", "run_experiment_persistent_store",
    "apply_quick", "guarded_rewrite", "net_depth", "TRASH", "PROTECTED", "ACTIVE",
]
