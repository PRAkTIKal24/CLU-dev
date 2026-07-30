"""Experiment MEMORY-GYM: Track 1, launder-native, the dividend as the sole KPI.

Runs the four gym families of :mod:`chlu.experiments.memory_gym` — one per
structural opening of charter §2.1 — **on the harness ``full-clu-harness``
landed**, with

* the **three harness-native controls firing automatically on every cell**
  (settle-deleted launder · same-keys null · blank/empty store) — a cell that
  cannot produce all three does not report;
* **one strong classical substitute per family**, three of which cost **+0 B**
  (k-NN mean · insertion order · echo). This is not optional politeness: for a
  non-metric-native query the frozen three are not the strongest classical
  method, and a positive dividend a +0 B substitute matches is an artefact of the
  frozen control's byte allocation, not a dynamics dividend;
* an explicit **byte ledger** on both sides of every dividend, with the
  ``matched`` flag. ⚠ A ``matched=False`` cell may be reported, clearly labelled,
  but **may not be quoted as a dividend**;
* the full **13-monitor registry** on every consolidation window, so the four
  monitors the harness left UNTESTED (#3, #4, #6, #11) finally get a stream long
  enough to exercise them — and *exercising a monitor for the first time is a
  reportable result*;
* **multi-seed** aggregation (mean ± sample sd, ``ddof = 1``) before any number
  leaves the experiment.

⭐ **A dividend of ≈0 or negative at v0 is the charter's own stated expectation
and is a SUCCESSFUL outcome for this task** (charter §6.2). A *positive* cell is
suspicious: it is reported as ``unexplained-pending-controls`` with every control
and its multi-seed spread beside it, never as a result.

Runnable directly::

    uv run python -m chlu.experiments.exp_memory_gym --quick

or via the CLI: ``chlu exp-memory-gym [--project N] [--seed I] [--quick]``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.eval.dividend import (
    ByteAccount,
    blank_store_control,
    byte_account,
    dividend,
    echo_launder,
    fit_shared_metric,
    knn_mean_launder,
    order_aware_launder,
    same_keys_null,
    settle_deleted_launder,
    shared_metric_launder,
)
from chlu.experiments.memory_gym import (
    FAMILIES,
    PRIMARY_METRIC,
    GymConfig,
    QuerySet,
    byte_ratio_law,
    gym_config,
    make_gym_stream,
    queries_aggregate,
    queries_manifold,
    queries_overload,
    queries_recency,
    readout_occupancy,
    readout_point_assign,
    readout_settled,
    readout_spectator,
    readout_tail_mean,
    score,
)

#: The declared compute order (PREREG §6). ``(family, arm, seeds)`` per cell.
DEFAULT_PLAN: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    # 2. the byte-parity frontier first — it answers the Hub's hard problem
    ("overload", "ref3", (0,)),
    ("overload", "ref8", (0,)),
    ("overload", "ref16", (0,)),
    # 2b. the 1x-LOAD reference (added after the 3x frontier came out flat, so the
    # curve is attributable to the load rather than to the atom budget alone —
    # declared as a post-hoc addition in the report, not as a pre-registered cell)
    ("overload", "load1x", (0, 1, 2)),
    ("overload", "load1x_ref8", (0,)),
    ("overload", "load1x_ref3", (0,)),
    ("overload", "load1x_shipped", (0, 1, 2)),
    # 3. the acceptance criterion: all four families, three seeds
    ("overload", "base", (0, 1, 2)),
    ("aggregate", "base", (0, 1, 2)),
    ("recency", "base", (0, 1, 2)),
    ("manifold", "base", (0, 1, 2)),
    # 4. the mechanism arm: overlapping basins
    ("aggregate", "tight", (0, 1, 2)),
    # 5. the named blockers / monitor probes
    ("manifold", "ridge", (0,)),
    ("overload", "reach_free", (0,)),
)

QUICK_PLAN: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    ("overload", "base", (0,)),
    ("aggregate", "base", (0,)),
    ("recency", "base", (0,)),
    ("manifold", "base", (0,)),
)

#: ``--quick`` shrinks the store and the stream, never the code path.
QUICK_GYM = dict(n_offer=6, capacity=6, budget=6, reference_capacity=3,
                 n_query_per_item=2, n_query_per_pair=3, consolidate_every=2,
                 min_consolidations=4, n_manifold_launch=4)
QUICK_CLU = dict(write_steps=30, address_steps=80, read_steps=120,
                 n_query_per_item=2, quick=True)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _insertion_order(system) -> Dict[int, int]:
    """Item id -> insertion index, from the controller's **public** verb log.

    This is the recency ground truth, and it is also exactly the information a
    table gets for free in its row order (see ``order_aware_launder``).
    """
    out: Dict[int, int] = {}
    n = 0
    for vr in system.controller.log:
        if vr.verb == "admit" and vr.applied:
            iid = int(vr.detail.get("item_id", -1))
            if iid >= 0:
                out[iid] = n
                n += 1
    return out


def _build_queries(gcfg: GymConfig, ccfg: CluSystemConfig, stream, system,
                   centers: np.ndarray, pays: np.ndarray, born: np.ndarray,
                   rng: np.random.Generator) -> QuerySet:
    fam = gcfg.family
    if fam == "overload":
        return queries_overload(gcfg, ccfg, stream, rng)
    if fam == "aggregate":
        return queries_aggregate(gcfg, ccfg, centers, pays, rng)
    if fam == "recency":
        return queries_recency(gcfg, ccfg, centers, born, rng)
    if fam == "manifold":
        return queries_manifold(gcfg, ccfg, centers, rng)
    raise ValueError(fam)


def _clu_predictions(gcfg: GymConfig, ccfg: CluSystemConfig, res,
                     centers: np.ndarray, sep: float) -> Dict[str, np.ndarray]:
    """The CLU's own read-outs. ⭐ Point vs trajectory is an *internal ablation*."""
    fam = gcfg.family
    if fam == "manifold":
        return {"clu": readout_spectator(res, ccfg)}
    if fam == "recency":
        radius = float(gcfg.occupancy_radius_frac) * (sep if np.isfinite(sep) else 1.0)
        occ = readout_occupancy(res, centers, radius)
        return {"clu": np.argmax(occ, axis=1),               # trajectory read-out
                "clu_point": readout_point_assign(res, centers),  # settled point
                "_occupancy": occ}
    return {"clu": readout_settled(res, ccfg),
            "clu_traj_tail": readout_tail_mean(res, ccfg)}


def _launder_predictions(qs: QuerySet, centers: np.ndarray, pays: np.ndarray,
                         born: np.ndarray, rng: np.random.Generator,
                         shared_M: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Every control's prediction, scored later by the SAME family scorer.

    The first two keys are the frozen harness-native controls and are present on
    every cell by construction; the rest are the family's strong substitutes.
    """
    out: Dict[str, np.ndarray] = {}
    if qs.kind == "value":
        out["settle_deleted"] = settle_deleted_launder(centers, pays, qs.keys)
        out["same_keys_null"] = same_keys_null(centers, pays, qs.keys, rng)
        out["knn2_mean_+0B"] = knn_mean_launder(centers, pays, qs.keys, k=2)
        out["knn2_idw_+0B"] = knn_mean_launder(centers, pays, qs.keys, k=2,
                                               weighting="inverse_distance")
        if shared_M is not None:
            out["shared_metric_+40B"] = shared_metric_launder(centers, pays, qs.keys,
                                                              shared_M)
    elif qs.kind == "index":
        out["settle_deleted"] = settle_deleted_launder(centers, pays, qs.keys,
                                                       metric="assign")
        # ⚠ same-keys null = same keys, PERMUTED PAYLOADS (Hub ruling). For an
        # index-valued question the permutation cannot change an arg-min over
        # keys, so this control provably COINCIDES with the settle-deleted launder
        # here. Reported, not hidden: a control that cannot differ is a control
        # that carries no information for this family.
        perm = rng.permutation(pays.shape[0])
        out["same_keys_null"] = settle_deleted_launder(centers, pays[perm], qs.keys,
                                                       metric="assign")
        out["order_aware_+0B"] = order_aware_launder(centers, born, qs.keys, k=2)
    else:  # coord
        n = len(qs)
        # a table stores ONE point per item, and the spectator coordinate it
        # stored is the written one (zero) — so its manifold prediction is constant.
        out["settle_deleted"] = np.zeros((n, 1), dtype=float)
        out["same_keys_null"] = np.zeros((n, 1), dtype=float)
        out["echo_+0B"] = echo_launder(qs.target)
    return out


def _ridge_write(system, gcfg: GymConfig, ccfg: CluSystemConfig, slot: int,
                 address: np.ndarray, payload: float, key) -> dict:
    """F4's blocker arm: write **collinear targets along the spectator axis**.

    ``train_memory_landscape`` accepts ``(K, dim)`` targets, so a *valley* is
    expressible even though the harness's own write path passes a single row. The
    controller has **no verb** for it — that is the named blocker, and this
    function measures it rather than asserting it.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from chlu.core.memory_potentials import atom_write_mask_fn
    from chlu.training.train_memory import train_memory_landscape

    j = ccfg.addr_dim + ccfg.payload_dim
    n = int(gcfg.ridge_targets)
    grid = np.linspace(-gcfg.manifold_launch_span, gcfg.manifold_launch_span, n)
    z = np.zeros((n, ccfg.dim), dtype=np.float32)
    z[:, : ccfg.addr_dim] = np.asarray(address)[: ccfg.addr_dim]
    z[:, ccfg.addr_dim: ccfg.addr_dim + ccfg.payload_dim] = payload
    z[:, j] = grid
    V, hist = train_memory_landscape(
        system.store.V, jnp.asarray(z), key,
        steps=int(ccfg.write_steps), lr=float(ccfg.write_lr),
        weight_decay=float(ccfg.write_weight_decay),
        loss_kwargs=dict(n_perturb=int(ccfg.write_n_perturb),
                         sigma_addr=float(ccfg.write_sigma_addr),
                         sigma_pay=float(ccfg.write_sigma_pay),
                         margin=float(ccfg.write_margin),
                         barrier=0.0,  # the ridge's own rows must NOT repel
                         payload_index=int(ccfg.addr_dim)),
        update_mask_fn=atom_write_mask_fn(system.store.group_rows(int(slot))),
    )
    system.store = eqx.tree_at(lambda s: s.V, system.store, V)
    return {"n_targets": n, "final_loss": (float(hist[-1]) if hist else float("nan")),
            "slot": int(slot),
            "note": ("a ridge write is NOT a controller verb; it is applied here "
                     "directly to the item's own atom mask to measure the blocker")}


def _hessian_spectrum(system, centers: np.ndarray, pays: np.ndarray,
                      ccfg: CluSystemConfig) -> dict:
    """Eigen-spectrum of ``Hess V`` at each site + the softest direction's
    spectator participation — the flat-direction diagnostic (pillar 1/(d))."""
    import jax
    import jax.numpy as jnp

    V = system.model().potential_net
    j = ccfg.addr_dim + ccfg.payload_dim
    lam_min, lam_gap, spec_part = [], [], []
    for c, a in zip(centers, pays, strict=True):
        z = np.zeros((ccfg.dim,), dtype=np.float32)
        z[: ccfg.addr_dim] = c
        z[ccfg.addr_dim: ccfg.addr_dim + ccfg.payload_dim] = a
        H = np.asarray(jax.hessian(lambda q: V(q))(jnp.asarray(z)))
        w, U = np.linalg.eigh(H)
        lam_min.append(float(w[0]))
        lam_gap.append(float(w[1] - w[0]))
        if ccfg.n_spectator > 0:
            spec_part.append(float(np.sum(U[j: j + ccfg.n_spectator, 0] ** 2)))
    return {"lambda_min": lam_min, "lambda_gap": lam_gap,
            "softest_spectator_participation": spec_part}


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
def run_cell(family: str, arm: str = "base", seed: int = 0,
             gym_overrides: Optional[dict] = None, quick: bool = False,
             loud: bool = True) -> dict:
    """Run one gym cell end-to-end and return its record (dividend + controls)."""
    import jax

    over = dict(gym_overrides or {})
    if quick:
        qk = dict(QUICK_GYM)
        clu = dict(qk.pop("clu_overrides", {}))
        clu.update(QUICK_CLU)
        clu.update(dict(over.pop("clu_overrides", {})))
        qk.update(over)
        over = dict(qk, clu_overrides=clu)
    gcfg = gym_config(family, arm, seed=seed, **over)
    ccfg = gcfg.build_clu()
    label = f"{family}/{arm}@s{seed}"

    t0 = time.time()
    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=loud)
    stream = make_gym_stream(gcfg, ccfg)

    # --- the write stream, with consolidation windows interleaved -----------
    key = jax.random.PRNGKey(seed + 1)
    admitted: List[int] = []
    refused: List[int] = []
    evicted: List[int] = []
    deleted: List[int] = []
    losses: List[float] = []
    consolidations: List[dict] = []
    prev = 0
    for b in stream.chunks:
        if b > prev:
            key, k_w = jax.random.split(key)
            wrep = system.write_stream(stream.items[prev:b], key=k_w)
            admitted += wrep.admitted
            refused += wrep.refused
            evicted += wrep.evicted
            deleted += wrep.deleted
            losses += wrep.losses
            prev = b
        crep = system.consolidate()
        consolidations.append({
            "n_live": int(np.size(crep.self_probe.get("retention", []))),
            "acq": float(crep.self_probe.get("acq", float("nan"))),
            "decode": float(crep.self_probe.get("decode", float("nan"))),
            "write_loss": float(crep.self_probe.get("write_loss", float("nan"))),
            "trips": [r.name for r in crep.readings if r.tripped],
        })
    write_s = time.time() - t0

    ids, centers, pays = system.codebook()
    order_map = _insertion_order(system)
    born = np.asarray([order_map.get(int(i), -1) for i in ids], dtype=float)

    ridge = None
    if gcfg.ridge_write and len(ids) > 0:
        key, k_r = jax.random.split(key)
        slot = system._slot_of(int(ids[0]))
        ridge = _ridge_write(system, gcfg, ccfg, slot, centers[0], float(pays[0, 0]),
                             k_r)

    degenerate = int(len(ids)) < 2
    if degenerate:
        print(f"⛔ DEGENERATE CELL {label}: n_live={len(ids)} — fewer than 2 live "
              f"items cannot support ANY reported metric; a clean monitor table "
              f"here is the empty-store artefact, not an acceptance.")
        return {"cell": label, "family": family, "arm": arm, "seed": seed,
                "degenerate": True, "n_live": int(len(ids)),
                "admitted": admitted, "refused": refused,
                "gym_config_non_default": gcfg.as_flag_table(),
                "clu_config_non_default": ccfg.as_flag_table()}

    # --- the queries and the read ------------------------------------------
    rng = np.random.default_rng(seed + 7717)
    qs = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)
    sep = float(system.certificates().get("sep", float("nan")))
    t1 = time.time()
    res = system.read(qs.q0)
    read_s = time.time() - t1

    preds = _clu_predictions(gcfg, ccfg, res, centers, sep)
    occupancy = preds.pop("_occupancy", None)

    # ⭐ SECOND READ VARIANT, at zero extra bytes and (to within one step) the same
    # compute: the **annealed read**, applied through the designed `anneal` verb.
    # It is the harness's own cleanest mechanism result (acquisition 0.828 -> 1.000,
    # decode 0.906 -> 1.000, clearing modes #5 and #9 simultaneously), i.e. the
    # KNOWN PRODUCTIVE BAND of the read lever (charter §3.1). Measuring every
    # family only at the shipped read would contaminate all four numbers with mode
    # #5, and measuring only at the annealed read would break the PREREG; so both
    # are reported, the shipped read is the pre-registered primary, and the write
    # (which dominates the cost) is shared.
    res_a = preds_a = None
    try:
        system.controller.anneal([4.0, 2.0, 1.0])
        res_a = system.read(qs.q0)
        preds_a = _clu_predictions(gcfg, ccfg, res_a, centers, sep)
        occupancy_a = preds_a.pop("_occupancy", None)
    except Exception as exc:  # a refused verb is reported, never silently skipped
        anneal_error = repr(exc)
        occupancy_a = None
    else:
        anneal_error = None

    # --- the shared-metric launder (doctrine I-12; never run before) --------
    shared_M = None
    metric_fit = None
    if qs.kind == "value" and centers.shape[0] >= 2:
        fit_rng = np.random.default_rng(seed + 991)
        n_fit = max(4 * centers.shape[0], 32)
        lab_fit = fit_rng.integers(0, centers.shape[0], size=n_fit)
        q_fit = centers[lab_fit] + fit_rng.normal(size=(n_fit, ccfg.addr_dim)) * ccfg.query_sigma
        shared_M = fit_shared_metric(centers, q_fit, lab_fit)
        metric_fit = {"n_fit": int(n_fit), "det": float(np.linalg.det(shared_M)),
                      "eigvals": np.linalg.eigvalsh(shared_M).tolist(),
                      "bytes": int(ccfg.addr_dim * (ccfg.addr_dim + 1) / 2 * 4),
                      "note": ("isotropic query law => M ~ I => this launder "
                               "necessarily TIES plain arg-min; run because "
                               "doctrine I-12 asks for it and it had never run")}
    launders = _launder_predictions(qs, centers, pays, born, rng, shared_M)

    # --- the blank / empty-store control (harness-native #3 of 3) ----------
    blank_sys = build_system(replace(ccfg, seed=ccfg.seed + 991),
                             key=jax.random.PRNGKey(seed + 991), loud=False)
    b_res = blank_sys.read(qs.q0)
    b_preds = _clu_predictions(gcfg, ccfg, b_res, centers, sep)
    b_preds.pop("_occupancy", None)
    blank_primary = score(qs, b_preds["clu"])[PRIMARY_METRIC[family]]
    # the auxiliary, family-independent decode the harness uses (comparable to
    # its 0.125-0.167 numbers), scored through the FROZEN blank_store_control
    aux_labels = np.argmin(
        np.linalg.norm(qs.keys[:, None, :] - centers[None, :, :], axis=-1), axis=1)
    b_val = readout_settled(b_res, ccfg)
    b_dec = np.argmin(np.linalg.norm(b_val[:, None, :] - pays[None, :, :], axis=-1),
                      axis=1)
    blank_aux = blank_store_control(lambda _q: b_dec, qs.q0, labels=aux_labels,
                                    representation="settled_point")

    # --- score everything with the SAME scorer -----------------------------
    metric = PRIMARY_METRIC[family]
    scores = {name: score(qs, p) for name, p in preds.items()}
    if preds_a is not None:
        scores.update({f"{n}@anneal": score(qs, p) for n, p in preds_a.items()})
    scores.update({name: score(qs, p) for name, p in launders.items()})
    full = float(scores["clu"][metric])
    lnd = float(scores["settle_deleted"][metric])

    # --- the byte ledger, both sides ---------------------------------------
    ba: ByteAccount = byte_account(system, centers, pays)
    atoms_per_live = system.store.V.learned.centers.shape[0] / max(len(ids), 1)
    ledger = dict(ba.as_dict())
    ledger.update({
        "n_atoms": int(system.store.V.learned.centers.shape[0]),
        "atoms_per_live_item": float(atoms_per_live),
        "closed_form_ratio": byte_ratio_law(atoms_per_live, ccfg.addr_dim,
                                            ccfg.payload_dim, ccfg.n_spectator),
        "controller_state_bytes_reported_not_counted": int(len(ids) * 4 * 4),
        "control_extra_bytes": {"knn2_mean": 0, "knn2_idw": 0, "order_aware": 0,
                                "echo": 0,
                                "shared_metric": (metric_fit or {}).get("bytes", 0)},
        "floor_note": ("one atom group per item (what makes the write masked / "
                       "C3-local) forces atoms_per_live_item >= 1, hence "
                       f"ratio >= {byte_ratio_law(1.0, ccfg.addr_dim, ccfg.payload_dim, ccfg.n_spectator):.2f}x "
                       "at this geometry: matched bytes is UNREACHABLE by "
                       "construction, not merely unachieved"),
    })

    # --- the monitors, on the cell's final state ---------------------------
    probe = system.self_probe()
    certs = system.certificates()
    blank_dict = dict(blank_aux)
    blank_dict["family_primary_score"] = float(blank_primary)
    readings = system.observe(stage=label, self_probe=probe, certificates=certs,
                              blank=blank_dict, reads=res.diagnostics,
                              extras={"gym_family": family, "gym_arm": arm})
    trips = [r.name for r in readings if r.tripped]
    # the same registry, on the annealed read — so the trip table records whether
    # the designed restoring verb actually clears what it is supposed to clear
    readings_a: List[Any] = []
    trips_a: List[str] = []
    probe_a: Dict[str, Any] = {}
    if res_a is not None:
        probe_a = system.self_probe()
        readings_a = system.observe(stage=f"{label}/annealed", self_probe=probe_a,
                                    certificates=certs, blank=blank_dict,
                                    reads=res_a.diagnostics,
                                    extras={"gym_family": family, "gym_arm": arm,
                                            "read_variant": "annealed"})
        trips_a = [r.name for r in readings_a if r.tripped]

    controls = {"same_keys_null": float(scores["same_keys_null"][metric]),
                "blank_store": float(blank_primary),
                "blank_aux_decode": float(blank_aux["score"]),
                "blank_aux_bar": float(blank_aux["bar"]),
                "chance": float(scores["clu"].get("chance", float("nan")))}
    for name in launders:
        if name not in ("settle_deleted", "same_keys_null"):
            controls[name] = float(scores[name][metric])
    div = dividend(full, lnd, metric=metric, controls=controls, bytes_account=ba,
                   flags={"cell": label, "family": family, "arm": arm, "seed": seed,
                          **gcfg.as_flag_table(), **ccfg.as_flag_table()})

    # --- the trivial-substitute audit (the genuine-win bar) ----------------
    zero_byte = {k: v for k, v in controls.items() if k.endswith("+0B")}
    best_zero = max(zero_byte.values()) if zero_byte else float("nan")
    audit = {
        "zero_byte_substitutes": zero_byte,
        "best_zero_byte": best_zero,
        "clu_minus_best_zero_byte": (float(full - best_zero)
                                     if np.isfinite(best_zero) else float("nan")),
        "verdict": ("CLU beats every +0 B substitute" if np.isfinite(best_zero)
                    and full > best_zero else
                    ("a +0 B substitute matches or beats the CLU => the dividend "
                     "against the frozen launder is an artefact of that control's "
                     "byte allocation" if np.isfinite(best_zero) else "no +0 B "
                     "substitute defined for this family")),
    }

    rec = {
        "cell": label, "family": family, "arm": arm, "seed": seed,
        "degenerate": False,
        "gym_config_non_default": gcfg.as_flag_table(),
        "clu_config_non_default": ccfg.as_flag_table(),
        "n_offered": len(stream.offered), "n_stream_rows": len(stream.items),
        "n_live": int(len(ids)),
        "overload_factor": gcfg.overload_factor,
        "admitted": admitted, "refused": refused, "evicted": evicted,
        "deleted": deleted, "write_losses": losses,
        "n_consolidations": len(consolidations), "consolidations": consolidations,
        "n_queries": int(len(qs)), "primary_metric": metric,
        "scores": {k: {m: float(v) for m, v in s.items()} for k, s in scores.items()},
        "sep": sep, "sep_over_sigma_q": float(sep / max(ccfg.query_sigma, 1e-12)),
        "certificates": {k: (float(v) if isinstance(v, (int, float, np.floating))
                             else v) for k, v in certs.items()},
        "self_probe": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in probe.items()
                       if k in ("acq", "strict", "decode", "chance",
                                "delta_read_basin_conditioned", "retention",
                                "payload_abs", "write_loss")},
        "blank": blank_dict,
        "byte_ledger": ledger,
        "shared_metric_fit": metric_fit,
        "dividend": div.as_dict(),
        "trivial_substitute_audit": audit,
        "monitors": [r.as_dict() for r in readings],
        "trips": trips,
        "read_variants": {
            "shipped": {"primary": full, "trips": trips,
                        "acq": float(probe.get("acq", float("nan")))},
            "annealed": ({"primary": float(scores["clu@anneal"][metric]),
                          "dividend": float(scores["clu@anneal"][metric] - lnd),
                          "trips": trips_a,
                          "acq": float(probe_a.get("acq", float("nan"))),
                          "schedule": [4.0, 2.0, 1.0],
                          "extra_bytes": 0,
                          "n_steps": int(res_a.n_steps)}
                         if res_a is not None else {"error": anneal_error}),
        },
        "monitors_annealed": [r.as_dict() for r in readings_a],
        "monitor_summary": system.registry.summary(),
        "guard_counts": system.controller.guard_fire_counts(),
        "verb_counts": system.controller.verb_counts(),
        "n_steps_per_read": int(res.n_steps),
        "wall_write_s": write_s, "wall_read_s": read_s,
        "ridge_write": ridge,
    }
    if family == "recency":
        rec["point_vs_trajectory"] = {
            "traj": float(scores["clu"][metric]),
            "point": float(scores["clu_point"][metric]),
            "delta_traj_minus_point": float(scores["clu"][metric]
                                            - scores["clu_point"][metric]),
            "traj_annealed": (float(scores["clu@anneal"][metric])
                              if preds_a is not None else float("nan")),
            "point_annealed": (float(scores["clu_point@anneal"][metric])
                               if preds_a is not None else float("nan")),
            "occupancy_mean_max": (float(np.mean(np.max(occupancy, axis=1)))
                                   if occupancy is not None else float("nan")),
            "occupancy_mean_max_annealed": (
                float(np.mean(np.max(occupancy_a, axis=1)))
                if occupancy_a is not None else float("nan")),
            "note": ("handcrafted psi (soft time-occupancy). The learned "
                     "trajectory read-out is `trainability-spike`'s; scoring "
                     "pillar (c) with a handcrafted psi is the declared v0 limit"),
        }
    if family == "manifold":
        rec["manifold_diagnostics"] = _hessian_spectrum(system, centers, pays, ccfg)
    return rec


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate(records: Sequence[dict]) -> Dict[str, dict]:
    """Group cells by ``family/arm`` and reduce over seeds.

    Mean ± **sample** sd (``ddof = 1``) and ``SE = sd/sqrt(n)``. ⚠ The convention
    is declared because the repo carries a cross-wave sd-convention split
    (population vs sample) that has already caused one reconciliation.
    """
    out: Dict[str, dict] = {}
    for rec in records:
        if rec.get("degenerate"):
            continue
        key = f"{rec['family']}/{rec['arm']}"
        row = out.setdefault(key, {"family": rec["family"], "arm": rec["arm"],
                                   "seeds": [], "metric": rec["primary_metric"],
                                   "_full": [], "_lnd": [], "_div": [],
                                   "_controls": {}, "_ratio": [], "_sep": [],
                                   "_zero": [], "_full_a": [], "_div_a": []})
        row["seeds"].append(rec["seed"])
        row["_full"].append(rec["dividend"]["full"])
        row["_lnd"].append(rec["dividend"]["launder"])
        row["_div"].append(rec["dividend"]["dividend"])
        rv = (rec.get("read_variants") or {}).get("annealed") or {}
        row["_full_a"].append(rv.get("primary", float("nan")))
        row["_div_a"].append(rv.get("dividend", float("nan")))
        row["_ratio"].append(rec["byte_ledger"]["ratio"])
        row["_sep"].append(rec["sep_over_sigma_q"])
        row["_zero"].append(rec["trivial_substitute_audit"]["best_zero_byte"])
        for k, v in rec["dividend"]["controls"].items():
            row["_controls"].setdefault(k, []).append(v)

    def _ms(v):
        a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
        if a.size == 0:
            return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                    "n": 0}
        sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        return {"mean": float(np.mean(a)), "sd": sd,
                "se": (sd / np.sqrt(a.size) if a.size > 1 else 0.0), "n": int(a.size)}

    for row in out.values():
        row["full"] = _ms(row.pop("_full"))
        row["launder_settle_deleted"] = _ms(row.pop("_lnd"))
        row["dividend"] = _ms(row.pop("_div"))
        row["full_annealed"] = _ms(row.pop("_full_a"))
        row["dividend_annealed"] = _ms(row.pop("_div_a"))
        row["byte_ratio"] = _ms(row.pop("_ratio"))
        row["sep_over_sigma_q"] = _ms(row.pop("_sep"))
        row["best_zero_byte_substitute"] = _ms(row.pop("_zero"))
        row["controls"] = {k: _ms(v) for k, v in row.pop("_controls").items()}
        row["matched_bytes"] = bool(abs(row["byte_ratio"]["mean"] - 1.0) <= 0.05)
        row["quotable_as_dividend"] = bool(row["matched_bytes"])
        d, se, n = (row["dividend"]["mean"], row["dividend"]["se"],
                    row["dividend"]["n"])
        sign = ("positive" if d > max(2.0 * se, 1e-9) else
                ("negative" if d < -max(2.0 * se, 1e-9) else "~zero"))
        # ⚠ a single-seed cell has no spread, so its sign is not a claim
        row["sign"] = sign if n >= 2 else f"{sign} (SINGLE SEED — not a claim)"
    return out


def monitor_first_fires(records: Sequence[dict]) -> Dict[str, dict]:
    """Which monitors fired, where, and which are still UNTESTED.

    A monitor that never fires anywhere is **untested, not green** — the harness
    left #3, #4, #6 and #11 in that state and a gym stream is long enough to
    exercise them.
    """
    table: Dict[str, dict] = {}
    for rec in records:
        for key, suffix in (("monitors", ""), ("monitors_annealed", "/annealed")):
            for m in rec.get(key, []):
                row = table.setdefault(m["name"], {"mode": m["mode"], "cells": {},
                                                   "n_applicable": 0, "n_trips": 0})
                state = ("TRIP" if m["tripped"]
                         else ("inapplicable" if not m["applicable"] else "clear"))
                row["cells"][rec["cell"] + suffix] = state
                row["n_applicable"] += int(bool(m["applicable"]))
                row["n_trips"] += int(bool(m["tripped"]))
        for c in rec.get("consolidations", []):
            for name in c.get("trips", []):
                row = table.setdefault(name, {"mode": -1, "cells": {},
                                              "n_applicable": 0, "n_trips": 0})
                row["n_trips"] += 1
                row["cells"].setdefault(rec["cell"] + ":consolidate", "TRIP")
    for row in table.values():
        row["ever_tripped"] = row["n_trips"] > 0
        row["untested"] = (row["n_applicable"] == 0) or (row["n_trips"] == 0)
    return table


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_memory_gym(
    config=None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    families: Optional[Sequence[str]] = None,
    arms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    quick: bool = False,
    overrides: Optional[dict] = None,
) -> dict:
    """Run the gym plan and write the dividend/byte/monitor artifact.

    ``config`` is a :class:`~chlu.config.CHLUConfig`, used only for project paths
    and the default seed; the gym's own knobs live in
    :class:`~chlu.experiments.memory_gym.GymConfig`, read from a ``memory_gym:``
    block of the project YAML if present. That keeps the gym config-driven
    **without touching ``chlu/config.py``** (C1W27 owns two blocks of it).
    """
    os.makedirs(save_dir, exist_ok=True)
    base_seed = int(seed if seed is not None else getattr(
        getattr(config, "project", None), "seed", 0) or 0)
    over = dict(overrides or _project_overrides(config))

    plan = list(QUICK_PLAN if quick else DEFAULT_PLAN)
    if families:
        plan = [c for c in plan if c[0] in set(families)]
    if arms:
        plan = [c for c in plan if c[1] in set(arms)]
    if seeds:
        plan = [(f, a, tuple(int(s) for s in seeds)) for f, a, _ in plan]
    elif base_seed:
        plan = [(f, a, tuple(int(s) + base_seed for s in ss)) for f, a, ss in plan]

    results: Dict[str, Any] = {
        "base_seed": base_seed,
        "quick": bool(quick),
        "plan": [{"family": f, "arm": a, "seeds": list(ss)} for f, a, ss in plan],
        "sd_convention": "sample sd (ddof=1); se = sd/sqrt(n)",
        "declared_query_law": (
            "sigma_q = 0.15 ISOTROPIC (the harness's shipped value); d=4, m=1, "
            "ball_radius 1.0 (0.45 on aggregate/tight); sep/sigma_q reported per cell"
        ),
        "dividend_rule": (
            "dividend = (full CLU) - (its own settle-deleted launder), same "
            "harness / same phi. A matched=False cell may be reported, labelled, "
            "but MAY NOT be quoted as a dividend."
        ),
        "cells": [],
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_memory_gym_metrics.json")

    def _dump():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_json_default)

    for family, arm, cell_seeds in plan:
        for s in cell_seeds:
            t0 = time.time()
            try:
                rec = run_cell(family, arm, int(s), gym_overrides=over, quick=quick)
            except Exception as exc:  # a failed cell is reported, never silent
                import traceback

                traceback.print_exc()
                rec = {"cell": f"{family}/{arm}@s{s}", "family": family, "arm": arm,
                       "seed": int(s), "degenerate": True, "error": repr(exc)}
            rec["wall_s"] = time.time() - t0
            results["cells"].append(rec)
            d = rec.get("dividend", {})
            print(f"[{rec['cell']}] metric={rec.get('primary_metric')} "
                  f"full={d.get('full')} launder={d.get('launder')} "
                  f"dividend={d.get('dividend')} "
                  f"bytes={rec.get('byte_ledger', {}).get('ratio')}x "
                  f"matched={rec.get('byte_ledger', {}).get('matched')} "
                  f"trips={rec.get('trips')} ({rec['wall_s']:.0f}s)")
            _dump()

    results["aggregate"] = aggregate(results["cells"])
    results["monitor_table"] = monitor_first_fires(results["cells"])
    results["byte_frontier"] = [
        {"cell": r["cell"], "arm": r["arm"],
         "atoms_per_live_item": r["byte_ledger"]["atoms_per_live_item"],
         "byte_ratio": r["byte_ledger"]["ratio"],
         "n_atoms": r["byte_ledger"]["n_atoms"],
         "primary": r["dividend"]["full"], "metric": r["primary_metric"],
         "n_live": r["n_live"]}
        for r in results["cells"]
        if r.get("family") == "overload" and not r.get("degenerate")
    ]
    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        results["figures"] = []
        results["figure_error"] = repr(exc)
    _dump()
    results["metrics_path"] = out_path
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _project_overrides(config) -> dict:
    """Read a ``memory_gym:`` block from the project YAML, if any."""
    path = getattr(getattr(config, "project", None), "config_path", None)
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return dict(raw.get("memory_gym", {}))
    except Exception:
        return {}


def _plot(results: dict, save_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = results["aggregate"]
    keys = list(agg)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (1) the dividend per family/arm, with its zero-byte substitute
    ax = axes[0][0]
    x = np.arange(len(keys))
    d = [agg[k]["dividend"]["mean"] for k in keys]
    e = [agg[k]["dividend"]["se"] for k in keys]
    da = [agg[k]["dividend_annealed"]["mean"] for k in keys]
    ea = [agg[k]["dividend_annealed"]["se"] for k in keys]
    ax.bar(x - 0.2, d, 0.4, yerr=e, capsize=3, label="shipped read",
           color=["tab:green" if v > 0 else "tab:red" for v in d])
    ax.bar(x + 0.2, da, 0.4, yerr=ea, capsize=3, label="annealed read (+0 B)",
           color=["tab:green" if v > 0 else "tab:red" for v in da], alpha=0.55)
    ax.axhline(0, color="k", lw=1)
    ax.legend(fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("dividend (full - settle-deleted launder)")
    ax.set_title("dynamics dividend — <=0 at v0 is the honest start\n"
                 "(no cell is byte-matched; none is quotable as a dividend)",
                 fontsize=9)
    ax.grid(alpha=0.3)

    # (2) full vs every control
    ax = axes[0][1]
    names = ["full", "launder_settle_deleted"]
    ctrl_names = sorted({c for k in keys for c in agg[k]["controls"]})
    w = 0.8 / (len(names) + len(ctrl_names))
    for i, n in enumerate(names):
        ax.bar(x + i * w, [agg[k][n]["mean"] for k in keys], w, label=n)
    for j, c in enumerate(ctrl_names):
        ax.bar(x + (len(names) + j) * w,
               [agg[k]["controls"].get(c, {}).get("mean", np.nan) for k in keys],
               w, label=c)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("primary metric (family-specific)")
    ax.set_title("the CLU against every control, same scorer", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # (3) the byte frontier — quote the CURVE, not the endpoint
    ax = axes[1][0]
    fr = sorted(results.get("byte_frontier", []), key=lambda r: r["byte_ratio"])
    if fr:
        ax.plot([r["byte_ratio"] for r in fr], [r["primary"] for r in fr], "o-")
        for r in fr:
            ax.annotate(r["arm"], (r["byte_ratio"], r["primary"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("byte ratio full/launder (log)  — 1.0 = matched")
    ax.set_ylabel("decode (overload family)")
    ax.set_title("byte-parity frontier: matched bytes is UNREACHABLE\n"
                 "(one atom group per item forces ratio >= 2.2x at d=4,m=1)",
                 fontsize=9)
    ax.grid(alpha=0.3)

    # (4) point vs trajectory (pillar c) and the monitor first-fires
    ax = axes[1][1]
    pv = [(r["cell"], r["point_vs_trajectory"]) for r in results["cells"]
          if r.get("point_vs_trajectory")]
    if pv:
        xx = np.arange(len(pv))
        ax.bar(xx - 0.2, [p[1]["point"] for p in pv], 0.4, label="settled point psi")
        ax.bar(xx + 0.2, [p[1]["traj"] for p in pv], 0.4, label="trajectory psi")
        ax.axhline(0.5, color="k", ls=":", lw=1, label="chance (2-way)")
        ax.set_xticks(xx)
        ax.set_xticklabels([p[0] for p in pv], rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7)
    ax.set_ylabel("recency accuracy")
    ax.set_title("pillar (c): point vs trajectory read-out, handcrafted psi",
                 fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(save_dir, "exp_memory_gym.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment MEMORY-GYM: Track 1, the dividend as the sole KPI."
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Base seed offset")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES),
                        help="Run only these families (default: all)")
    parser.add_argument("--arms", nargs="+", help="Run only these arms")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Override the per-cell seed list")
    args = parser.parse_args()

    config = None
    save_dir, models_dir = "results", None
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        os.makedirs(save_dir, exist_ok=True)

    res = run_experiment_memory_gym(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed,
        families=args.families, arms=args.arms, seeds=args.seeds, quick=args.quick,
    )
    print(json.dumps({k: {"dividend": v["dividend"], "sign": v["sign"],
                          "byte_ratio": v["byte_ratio"]["mean"],
                          "matched": v["matched_bytes"]}
                      for k, v in res["aggregate"].items()}, indent=2))


if __name__ == "__main__":
    main()
