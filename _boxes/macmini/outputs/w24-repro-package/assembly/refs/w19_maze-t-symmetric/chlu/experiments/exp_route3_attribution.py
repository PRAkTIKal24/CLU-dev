"""⭐ Experiment ROUTE-3 STAGE 1 — **the store-attribution curve** (charter §A10).

Measurement only. **No new store, no new objective, no claim of a dividend** —
this experiment decides whether Route 3 stage 2 gets built at all, by applying the
Head's pre-registered §A9.4 bar arithmetically:

> **Stage 2 unlocks iff the per-slot store-attributable discriminability (full −
> settle-deleted launder, per slot ``t``) clears the launch-noise floor BEYOND
> 2 SE, at >= 3 seeds, on >= 1 family, at ANY ``t`` — q-slots and p-slots scored
> separately (a live p-channel unlocks even with a dead q-channel).**

⛔ **And it is overridden by §A9.5**, which is built here in stage 1 rather than
deferred: if a **per-slot matched-bytes table launder** reproduces the slotted
read, Route 3 has degenerated into ``K`` time-indexed lookup tables and fails
regardless of dividend (intervention §8.2).

The rig is the shipped one: :func:`chlu.experiments.exp_memory_gym.run_cell`'s own
write path, verbatim (same key chain, same query RNG, same blank-store control),
so the numbers are comparable to `memory-gym-v0` and to both C2W2 routes. Four
reads per cell:

============================  =================================================
``full``                      real store, real launch
``launder`` (settle-deleted)  **store-deleted** system, same launch
``floor`` (launch noise)      store-deleted, launch re-drawn within its own cloud
``perturbed``                 real store, launch re-drawn — the §A8.2 Jacobian
============================  =================================================

⭐ **Part 2 (C2W4 `bprime-c6`) — the third-party attribution probe.** ``--part
thirdparty`` deletes a stored item the query did **not** select and measures the
change in every slot's content: the one coupling a per-slot table gives **exactly
0** for by construction (`bprime-theory` Prop T5.4), swept across ``d/s``. ⛔ It is
**protocol evidence for the audit paper (§A14.1), not a Route-3 revival**: §A9.5's
kill stands and no inference-read claim is made anywhere in this module.

Runnable::

    uv run chlu exp-route3-attribution --quick
    uv run chlu exp-route3-attribution --part thirdparty --seeds 0 1 2
    uv run python -m chlu.experiments.exp_route3_attribution --families overload
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chlu.eval.attribution import (
    SLOT_GRID,
    THIRDPARTY_SLOT_GRID,
    CurveBundle,
    a95_verdict,
    address_block_curve,
    apply_a94_bar,
    attribution_curve,
    coupling_law_fit,
    jacobian_curves,
    slot_deltas,
    slot_index_table,
    table_third_party_delta,
    third_party_curve,
)
from chlu.experiments.exp_memory_gym import _build_queries
from chlu.experiments.memory_gym import gym_config, make_gym_stream

#: ``(family, gym arm)``. ⚠ ``overload`` is quoted **only** at ``load1x_shipped``
#: (the 478x anchor): at the gym's base atom budget it went 0/18 admissible
#: *including the Gaussian control* (C2W2 reconciliation 6, Hub-accepted).
FAMILY_ARMS: Tuple[Tuple[str, str], ...] = (
    ("overload", "load1x_shipped"),
    ("aggregate", "base"),
    ("manifold", "base"),
)

#: Head ruling (i) admissibility, the Route-1 convention (``race.WriteRecord``).
ENDPOINT_LOSS_TOL = 0.05

#: The one bounded budget escalation a 0-admissible family gets before it may be
#: called "not testable this wave" (C2W2 ruling (i) counterweight).
ESCALATION_WRITE_STEPS = 900


def _answer_channel(family: str, ccfg) -> Tuple[int, str]:
    """The family's own answer channel inside the latent, and its name.

    ``overload``/``aggregate`` answer with the **payload** block; ``manifold``
    answers with the **spectator** block (its target is the launch spectator
    coordinate). The instrument reads exactly the channel the family is scored on
    — it is not a new read-out.
    """
    if family == "manifold":
        return int(ccfg.addr_dim + ccfg.payload_dim), "spectator"
    return int(ccfg.addr_dim), "payload"


def _labels(qs) -> np.ndarray:
    """Item labels for the §A8.2 separation curve (pair id where there is no item)."""
    lab = np.asarray(qs.label).ravel()
    if np.all(lab >= 0):
        return lab
    pairs = np.asarray(qs.meta.get("pairs"))
    if pairs is not None and pairs.size:
        uniq = {tuple(p): i for i, p in enumerate(np.unique(pairs, axis=0))}
        return np.asarray([uniq[tuple(p)] for p in pairs], dtype=int)
    return np.zeros_like(lab)


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
@dataclass
class WrittenCell:
    """The shipped write path's product, shared by both parts of this module.

    Extracted verbatim from :func:`run_attribution_cell` (C2W4 `bprime-c6`) so the
    third-party probe writes the store **exactly** the way stage 1 did — same key
    chain, same query RNG, same admissibility rule — rather than re-deriving it.
    """

    system: Any
    gcfg: Any
    ccfg: Any
    stream: Any
    qs: Any
    ids: np.ndarray
    centers: np.ndarray
    pays: np.ndarray
    endpoint_loss: float
    lam_min: float
    admissible: bool
    reason: str


def _write_and_query(family: str, arm: str = "base", seed: int = 0, *,
                     quick: bool = False, write_steps: Optional[int] = None,
                     clu_extra: Optional[Dict[str, Any]] = None,
                     loud: bool = False) -> Optional[WrittenCell]:
    """Build the gym cell, run the shipped write stream, build its queries.

    ``clu_extra`` (C2W4) merges extra ``CluSystemConfig`` overrides — the
    third-party sweep uses it to move ``ball_radius`` (and the matching
    ``d_safe_override``) and nothing else. ``None`` => the shipped cell, verbatim.
    Returns ``None`` for a degenerate (``n_live < 2``) cell.
    """
    import jax

    from chlu.core.clu_system import build_system

    over: Dict[str, Any] = {}
    clu_over: Dict[str, Any] = {}
    if write_steps is not None:
        clu_over["write_steps"] = int(write_steps)
    if quick:
        over.update(n_offer=6, capacity=6, budget=6, reference_capacity=3,
                    n_query_per_item=2, n_query_per_pair=3, consolidate_every=2,
                    min_consolidations=4, n_manifold_launch=4)
        clu_over.update(write_steps=int(write_steps or 30), address_steps=80,
                        read_steps=120, n_query_per_item=2, quick=True)
    if clu_extra:
        clu_over.update({k: v for k, v in clu_extra.items() if v is not None})
    if clu_over:
        over["clu_overrides"] = clu_over
    gcfg = gym_config(family, arm, seed=seed, **over)
    ccfg = gcfg.build_clu()

    # --- the shipped write path, verbatim (same key chain as run_cell) -----
    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=loud)
    stream = make_gym_stream(gcfg, ccfg)
    key = jax.random.PRNGKey(seed + 1)
    prev = 0
    for b in stream.chunks:
        if b > prev:
            key, k_w = jax.random.split(key)
            system.write_stream(stream.items[prev:b], key=k_w)
            prev = b
        system.consolidate()

    ids, centers, pays = system.codebook()
    if len(ids) < 2:
        return None

    eps = [float(x) for x in (system._endpoint_losses or []) if np.isfinite(x)]
    endpoint_loss = float(max(eps)) if eps else float("nan")
    lam_min = float(system.certificates().get("lambda_min", float("nan")))
    admissible = bool(np.isfinite(endpoint_loss) and endpoint_loss <= ENDPOINT_LOSS_TOL
                      and np.isfinite(lam_min) and lam_min >= 0.0)
    reason = "" if admissible else (
        f"endpoint write loss {endpoint_loss:.4f} > {ENDPOINT_LOSS_TOL}"
        if not (np.isfinite(endpoint_loss) and endpoint_loss <= ENDPOINT_LOSS_TOL)
        else f"lambda_min<0 ({lam_min:+.4f})")

    # --- the queries: the family's own, the gym's own RNG -------------------
    rng = np.random.default_rng(seed + 7717)
    born = np.zeros((len(ids),), dtype=float)
    qs = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)
    return WrittenCell(system=system, gcfg=gcfg, ccfg=ccfg, stream=stream, qs=qs,
                       ids=ids, centers=centers, pays=pays,
                       endpoint_loss=endpoint_loss, lam_min=lam_min,
                       admissible=admissible, reason=reason)


def run_attribution_cell(family: str, arm: str = "base", seed: int = 0, *,
                         quick: bool = False, write_steps: Optional[int] = None,
                         slots: Sequence[int] = SLOT_GRID,
                         loud: bool = False) -> CurveBundle:
    """Write the gym stream on the shipped rig, then measure the four arms."""
    import jax

    from chlu.core.clu_system import build_system

    cell = _write_and_query(family, arm, seed, quick=quick,
                            write_steps=write_steps, loud=loud)
    if cell is None:
        return CurveBundle(family=family, seed=seed, admissible=False,
                           reason="degenerate cell: n_live<2")
    system, gcfg, ccfg, qs = cell.system, cell.gcfg, cell.ccfg, cell.qs
    ids, centers = cell.ids, cell.centers
    endpoint_loss, lam_min = cell.endpoint_loss, cell.lam_min
    admissible, reason = cell.admissible, cell.reason

    # --- the four arms ------------------------------------------------------
    # the store-deleted system is the harness's OWN blank/empty-store control
    blank = build_system(replace(ccfg, seed=ccfg.seed + 991),
                         key=jax.random.PRNGKey(seed + 991), loud=False)
    # the launch-noise draw: the query law itself (+N(0, sigma_q) on the address
    # block), an INDEPENDENT draw of the same magnitude
    nrng = np.random.default_rng(seed + 20260731)
    q0 = np.asarray(qs.q0, dtype=np.float32)
    delta = np.zeros_like(q0)
    delta[:, : ccfg.addr_dim] = nrng.normal(
        size=(q0.shape[0], ccfg.addr_dim)) * float(ccfg.query_sigma)
    q0_pert = q0 + delta

    t0 = time.time()
    res_full = system.read(q0)
    res_launder = blank.read(q0)
    res_floor = blank.read(q0_pert)
    res_pert = system.read(q0_pert)
    read_s = time.time() - t0

    idx, chan_name = _answer_channel(family, ccfg)
    rows = attribution_curve(
        res_full, res_launder, res_floor, np.asarray(qs.target, dtype=float),
        idx, slots=slots, keys=np.asarray(qs.keys), centers=np.asarray(centers),
        traj_stride=int(ccfg.traj_stride), dt=float(ccfg.dt),
        address_steps=int(ccfg.address_steps))
    jac = jacobian_curves(res_full, res_pert, delta[:, : ccfg.addr_dim],
                          _labels(qs), slots=slots, dim=int(ccfg.dim))
    # ⚠ POST-HOC secondary diagnostic (declared): the address block, where q0 IS
    # the query — the only place §A8.1's "position at small t is almost pure
    # query" can be tested, because the ANSWER channel of q0 is zeroed by the
    # shipped read. Never enters the §A9.4 bar.
    addr_rows = address_block_curve(res_full, res_launder, res_floor, _labels(qs),
                                    int(ccfg.addr_dim), slots=slots)

    return CurveBundle(
        family=family, seed=seed, admissible=admissible, reason=reason, rows=rows,
        jacobian=jac,
        flags={"address_block_posthoc": addr_rows, "gym_arm": arm, "n_live": int(len(ids)), "n_queries": int(len(qs)),
               "answer_channel": chan_name, "channel_index": int(idx),
               "endpoint_write_loss_max": endpoint_loss, "lambda_min": lam_min,
               "write_steps": int(ccfg.write_steps),
               "address_steps": int(ccfg.address_steps),
               "read_steps": int(ccfg.read_steps),
               "traj_stride": int(ccfg.traj_stride), "dt": float(ccfg.dt),
               "query_sigma": float(ccfg.query_sigma),
               "gamma_address": float(ccfg.gamma_address),
               "gamma_read": float(ccfg.gamma_read),
               "n_traj_points": int(np.asarray(res_full.traj).shape[1]),
               "retries": int(res_full.retries), "read_wall_s": read_s,
               **gcfg.as_flag_table(), **ccfg.as_flag_table()})


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_route3_attribution(
    config=None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    families: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0, 1, 2),
    quick: bool = False,
    escalate: bool = True,
    out_json: Optional[str] = None,
) -> dict:
    """Run the curve on every family, apply §A9.4, then §A9.5 (which overrides)."""
    os.makedirs(save_dir, exist_ok=True)
    fam_arms = [fa for fa in FAMILY_ARMS if not families or fa[0] in set(families)]
    if quick:
        seeds = tuple(seeds)[:1]

    bundles: Dict[str, Dict[int, CurveBundle]] = {}
    coverage: Dict[str, dict] = {}
    t_all = time.time()
    for family, arm in fam_arms:
        bundles[family] = {}
        excl: List[dict] = []
        for s in seeds:
            b = run_attribution_cell(family, arm, int(s), quick=quick)
            bundles[family][int(s)] = b
            if not b.admissible:
                excl.append({"seed": int(s), "reason": b.reason})
            print(f"[{family}/{arm} s{s}] admissible={b.admissible} "
                  f"({b.reason or 'ok'}) n_rows={len(b.rows)}", flush=True)
        adm = [s for s, b in bundles[family].items() if b.admissible]
        escalated = False
        if not adm and escalate and not quick:
            # ⭐ the ONE bounded budget escalation (C2W2 ruling (i) counterweight)
            escalated = True
            for s in seeds:
                b = run_attribution_cell(family, arm, int(s),
                                         write_steps=ESCALATION_WRITE_STEPS)
                bundles[family][int(s)] = b
                excl.append({"seed": int(s), "escalated_write_steps":
                             ESCALATION_WRITE_STEPS, "reason": b.reason})
                print(f"[{family}/{arm} s{s} ESCALATED {ESCALATION_WRITE_STEPS}] "
                      f"admissible={b.admissible} ({b.reason or 'ok'})", flush=True)
            adm = [s for s, b in bundles[family].items() if b.admissible]
        coverage[family] = {
            "gym_arm": arm, "n_cells": len(seeds), "n_admissible": len(adm),
            "coverage": float(len(adm) / max(len(seeds), 1)),
            "admissible_seeds": sorted(int(x) for x in adm),
            "escalated": escalated, "escalation_write_steps": (
                ESCALATION_WRITE_STEPS if escalated else None),
            "excluded": excl,
            "verdict": ("VOTES" if adm else
                        "ABSTAINS (0 admissibly-written cells after the one "
                        "bounded escalation — neither unlocks stage 2 nor blocks it)"),
        }

    a94: Dict[str, Any] = {}
    a95: Dict[str, Any] = {}
    for family, per_seed in bundles.items():
        adm = coverage[family]["admissible_seeds"]
        rows = {s: b.rows for s, b in per_seed.items()}
        a94[family] = apply_a94_bar(rows, family=family, admissible_seeds=adm,
                                    min_seeds=1 if quick else 3)
        clearing = [(r["channel"], r["slot"]) for r in a94[family]["clearing_set"]]
        a95[family] = a95_verdict(rows, admissible_seeds=adm,
                                  clearing_slots=clearing or None)

    unlock = any(v["unlock"] for v in a94.values())
    killed = any(v.get("fires") for f, v in a95.items() if a94[f]["unlock"])
    out = {
        "wall_s": time.time() - t_all,
        "seeds": [int(s) for s in seeds],
        "slot_grid": list(SLOT_GRID),
        "quick": bool(quick),
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); clears iff mean-2SE>0",
        "coverage_per_family": coverage,
        "a94_unlock_bar": a94,
        "a95_table_launder": a95,
        "unlock": bool(unlock),
        "a95_fires_on_clearing_slots": bool(killed),
        "stage2_verdict": (
            "⛔ NO STAGE 2 — §A9.5 fires and overrides the unlock" if killed else
            ("unlock = true" if unlock else "unlock = false")),
        "curves": {f: [b.as_dict() for b in per.values()]
                   for f, per in bundles.items()},
    }
    path = out_json or os.path.join(save_dir, "exp_route3_attribution.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    out["metrics_path"] = path
    try:
        out["figures"] = _plot(out, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        out["figures"] = []
        out["figure_error"] = repr(exc)
    print(f"\nunlock = {out['unlock']}  |  §A9.5 fires on clearing slots = "
          f"{out['a95_fires_on_clearing_slots']}  |  {out['stage2_verdict']}")
    return out


# --------------------------------------------------------------------------
# ⭐⭐ PART 2 (C2W4 `bprime-c6`) — THIRD-PARTY STORE ATTRIBUTION
# --------------------------------------------------------------------------
#: The ``d/s`` sweep: ``ball_radius`` scales the designed site geometry linearly
#: (``designed_sites`` is a scaled point set), so it moves ``d`` at fixed
#: ``atom_width`` and fixed ``query_sigma`` — which is exactly the axis T5.5's
#: exchange rate is a function of. ``d_safe_override`` is scaled with it so the
#: gate-to-geometry ratio is INVARIANT across the sweep and ``R = 1.0`` is the
#: shipped cell, bit-for-bit.
BALL_RADIUS_GRID: Tuple[float, ...] = (0.42, 0.55, 0.64, 0.80, 1.00, 1.20)

#: The shipped ``overload`` admission gate, and the radius it was declared at.
SHIPPED_D_SAFE_OVERRIDE = 0.58
SHIPPED_BALL_RADIUS = 1.0


def _sweep_overrides(ball_radius: Optional[float]) -> Dict[str, Any]:
    """``ball_radius`` + the matching admission gate. ``None`` => shipped cell."""
    if ball_radius is None:
        return {}
    r = float(ball_radius)
    return {"ball_radius": r,
            "d_safe_override": SHIPPED_D_SAFE_OVERRIDE * r / SHIPPED_BALL_RADIUS}


def _deleted_system(system, slot: int):
    """The same system with item ``slot``'s own atom group **deleted**.

    ``A_j = amp_j**2``, so scaling the group's amplitude by 0 removes that item's
    contribution to ``V_theta`` **exactly** and moves nothing else — the
    landscape analogue of deleting one row of the table. ⚠ Deliberately NOT the
    shipped ``evict`` path, which *re-draws* the freed group from the init
    distribution (a membership-leak repair, `LearnedVStore.reinit_group`): a
    re-draw substitutes a random row, it does not delete one, and the induced
    ``Delta`` would then be the re-draw's, not the item's. The shipped path is
    reported beside it as a robustness arm.
    """
    sysk = copy.copy(system)
    sysk.store = system.store.scale_group_amplitude(int(slot), 0.0)
    return sysk


def run_thirdparty_cell(family: str = "overload", arm: str = "load1x_shipped",
                        seed: int = 0, *, quick: bool = False,
                        ball_radius: Optional[float] = None,
                        slots: Sequence[int] = THIRDPARTY_SLOT_GRID,
                        write_steps: Optional[int] = None) -> dict:
    """⭐⭐ **C6** — delete a stored item the query did **not** select, and measure
    the change in every slot's content.

    Four objects, and they are deliberately four different things:

    ``third``  delete the query's **second-nearest** stored key (T5.4's
        non-selected row). ⭐ **This is the measurement.**
    ``own``    delete the query's **nearest** stored key — the denominator that
        makes the coupling dimensionless and comparable to T5.5's
        ``||grad V_j|| / own``.
    ``blank``  the same zeroing applied to an unwritten group of the harness's
        own blank store — the control that says the probe itself is not the
        signal.
    ``launch`` the launch-noise floor: an independent re-draw of the query
        jitter, propagated through the same slots.

    ⛔ Plus the **table's** third-party Delta, computed (not assumed) at every
    slot: **exactly 0 by construction (Prop T5.4)**.
    """
    import jax
    import jax.numpy as jnp

    from chlu.core.clu_system import _grad_vecs, build_system

    cell = _write_and_query(family, arm, seed, quick=quick,
                            write_steps=write_steps,
                            clu_extra=_sweep_overrides(ball_radius))
    if cell is None:
        return {"family": family, "arm": arm, "seed": int(seed),
                "ball_radius": ball_radius, "admissible": False,
                "reason": "degenerate cell: n_live<2", "rows": []}
    system, ccfg, qs = cell.system, cell.ccfg, cell.qs
    ids, centers = cell.ids, cell.centers
    K = int(len(ids))

    # --- who the query selects, and who it provably does NOT ----------------
    keys = np.asarray(qs.keys, dtype=float)
    dist = np.linalg.norm(keys[:, None, :] - centers[None, :, :], axis=-1)
    order = np.argsort(dist, axis=1)
    sel, third = order[:, 0].astype(int), order[:, 1].astype(int)
    d_third = dist[np.arange(dist.shape[0]), third]
    d_own = dist[np.arange(dist.shape[0]), sel]
    sep = float(_min_sep(centers))

    q0 = np.asarray(qs.q0, dtype=np.float32)
    nrng = np.random.default_rng(seed + 20260731)
    delta = np.zeros_like(q0)
    delta[:, : ccfg.addr_dim] = nrng.normal(
        size=(q0.shape[0], ccfg.addr_dim)) * float(ccfg.query_sigma)

    t0 = time.time()
    res_full = system.read(q0, allow_retry=False)
    res_launch = system.read(q0 + delta, allow_retry=False)
    slot_map = {r["slot"]: r for r in slot_index_table(
        res_full, slots, traj_stride=int(ccfg.traj_stride), dt=float(ccfg.dt),
        address_steps=int(ccfg.address_steps))}

    # --- the deletions: one read per live item ------------------------------
    deltas: Dict[int, Dict[str, np.ndarray]] = {}
    grads: Dict[int, np.ndarray] = {}
    m_full = system.model()
    g_full = np.asarray(_grad_vecs(m_full, jnp.asarray(q0)))
    for k in range(K):
        slot = system._slot_of(int(ids[k]))
        sysk = _deleted_system(system, slot)
        resk = sysk.read(q0, allow_retry=False)
        deltas[k] = slot_deltas(res_full, resk, slots=slots, dim=int(ccfg.dim))
        # the item's OWN gradient contribution: grad(V_full - V_{-k}) at q0
        grads[k] = np.linalg.norm(
            g_full - np.asarray(_grad_vecs(sysk.model(), jnp.asarray(q0))), axis=-1)

    # --- the blank-store delete control -------------------------------------
    blank = build_system(replace(ccfg, seed=ccfg.seed + 991),
                         key=jax.random.PRNGKey(seed + 991), loud=False)
    res_blank = blank.read(q0, allow_retry=False)
    res_blank_del = _deleted_system(blank, 0).read(q0, allow_retry=False)
    blank_d = slot_deltas(res_blank, res_blank_del, slots=slots, dim=int(ccfg.dim))
    launch_d = slot_deltas(res_full, res_launch, slots=slots, dim=int(ccfg.dim))

    # --- ⛔ the table's third-party Delta: computed, and exactly 0 -----------
    idx, _chan = _answer_channel(family, ccfg)
    tbl = []
    for j in slots:
        if j >= np.asarray(res_full.traj).shape[1]:
            continue
        v = np.asarray(res_full.traj)[:, int(j), int(idx)]
        for k in np.unique(third):
            tbl.append(table_third_party_delta(v, keys, centers, drop=int(k)))
    tbl_max = float(np.nanmax([r["max_abs_delta"] for r in tbl])) if tbl else float("nan")

    rows = third_party_curve(deltas, sel, third, blank=blank_d, launch=launch_d,
                             table_delta=tbl_max, sigma_q=float(ccfg.query_sigma),
                             slot_meta=slot_map)

    # --- P4: the BALLISTIC prefactor, with the damping correction -----------
    # |dq(n)| = (dt^2 F / (M gamma)) [n - (1-gamma)(1-(1-gamma)^n)/gamma], which
    # -> (tau^2/2M) F as gamma -> 0. The theorist's bare-ballistic version missed
    # its own 20 % bar by 0.61-0.73x at t = 10 (the residual is free-fall toward
    # the item's own well), so both the bare and the damped predictions are
    # reported and the free-fall residual is what is left over.
    mass = float(np.mean(np.asarray(m_full.mass_vector())))
    ball = []
    g_third_q = np.asarray([grads[int(third[i])][i] for i in range(len(sel))])
    for si, j in enumerate(deltas[0]["slots"]):
        n = int(slot_map.get(int(j), {}).get("step", 0))
        if n <= 0:
            continue
        tau = n * float(ccfg.dt)
        bare = (tau ** 2 / (2.0 * mass)) * g_third_q
        damp = _damped_over_bare(n, float(ccfg.gamma_address))
        meas = np.asarray([deltas[int(third[i])]["q"][si, i] for i in range(len(sel))])
        ok_b = bare > 0
        r_bare = meas[ok_b] / bare[ok_b]
        ball.append({"slot": int(j), "step": n, "t": tau,
                     "measured_over_bare_ballistic": float(np.median(r_bare))
                     if r_bare.size else float("nan"),
                     "damped_over_bare_predicted": damp,
                     "free_fall_residual": (float(np.median(r_bare)) / damp
                                            if r_bare.size and damp > 0
                                            else float("nan"))})

    # --- the static form of the same quantity (T5.5's own object) -----------
    # ⚠ MEDIAN over queries is the primary estimator (the denominator is a
    # per-query gradient magnitude, so the mean is dominated by the queries whose
    # own-well gradient is near zero). Fixed on --quick plumbing runs, before the
    # registered sweep; the mean is reported beside it.
    g_third = np.asarray([grads[int(third[i])][i] for i in range(len(sel))])
    g_own = np.asarray([grads[int(sel[i])][i] for i in range(len(sel))])
    ok = g_own > 0
    gr = g_third[ok] / g_own[ok] if np.any(ok) else np.zeros((0,))
    grad_ratio = float(np.median(gr)) if gr.size else float("nan")

    # --- the two x-axis conventions, both reported --------------------------
    Ds, ss = system.well_fits()
    s_fit = float(np.median(ss))
    s_proxy = float(ccfg.atom_width)
    d_geo = float(np.mean(d_third))

    return {
        "family": family, "arm": arm, "seed": int(seed),
        "ball_radius": float(ccfg.ball_radius),
        "d_safe_override": (None if ccfg.d_safe_override is None
                            else float(ccfg.d_safe_override)),
        "admissible": bool(cell.admissible), "reason": cell.reason,
        "n_live": K, "n_queries": int(len(sel)),
        "endpoint_write_loss_max": cell.endpoint_loss, "lambda_min": cell.lam_min,
        "sep": sep, "d_third_mean": d_geo, "d_own_mean": float(np.mean(d_own)),
        "s_proxy_atom_width": s_proxy, "s_fitted_well": s_fit,
        "well_depth_median": float(np.median(Ds)),
        "d_over_s_proxy": d_geo / s_proxy, "d_over_s_fitted": d_geo / s_fit,
        "d_over_s_gate": (float(ccfg.d_safe_override) / s_proxy
                          if ccfg.d_safe_override else float("nan")),
        "grad_ratio": grad_ratio,
        "grad_ratio_mean": float(np.mean(gr)) if gr.size else float("nan"),
        "grad_ratio_se_queries": (float(1.2533 * np.std(gr, ddof=1) / np.sqrt(gr.size))
                                  if gr.size > 1 else float("nan")),
        "t5_5_closed_form": _t55(d_geo, s_proxy, float(ccfg.query_sigma)),
        "ballistic": ball, "mass_mean": mass,
        "table_third_party_max_abs_delta": tbl_max,
        "table_exactly_zero": bool(all(r["exactly_zero"] for r in tbl)) if tbl else False,
        "sel_agrees_with_own_item": _sel_agreement(qs, ids, sel),
        "wall_s": time.time() - t0,
        "rows": [r.as_dict() for r in rows],
        "flags": {"slots": list(slots), "allow_retry": False,
                  "delete_mode": "amplitude_zero (A = amp**2 -> 0)",
                  "sigma_q": float(ccfg.query_sigma),
                  "atom_width": float(ccfg.atom_width),
                  "write_steps": int(ccfg.write_steps),
                  "address_steps": int(ccfg.address_steps),
                  "read_steps": int(ccfg.read_steps),
                  "traj_stride": int(ccfg.traj_stride), "dt": float(ccfg.dt),
                  "gamma_address": float(ccfg.gamma_address),
                  "gamma_read": float(ccfg.gamma_read),
                  "kinetic_mode": ccfg.kinetic_mode,
                  "atoms_per_item": int(ccfg.atoms_per_item)},
    }


def _damped_over_bare(n: int, gamma: float) -> float:
    """``|dq_damped(n)| / |dq_bare_ballistic(n)|`` for a constant force at ``p0 = 0``.

    Verlet with ``p <- (1-gamma) p`` per step gives
    ``dq(n) = (dt^2 F/(M gamma))[n - (1-gamma)(1-(1-gamma)^n)/gamma]`` while the
    bare ballistic form is ``(dt^2 F/M) n(n+1)/2``. At ``gamma = 0.05`` this is
    **0.864 at n = 10** and **0.744 at n = 20** — a pre-registered correction, not
    a fitted one.
    """
    g = float(gamma)
    if g <= 0:
        return 1.0
    num = n - (1.0 - g) * (1.0 - (1.0 - g) ** n) / g
    return float(num / (g * n * (n + 1) / 2.0))


def _sel_agreement(qs, ids: np.ndarray, sel: np.ndarray) -> float:
    """How often the query's **nearest live key** is its own item's row.

    ⚠ ``QuerySet.label`` indexes the *offered* stream, not the live codebook
    (which is sorted by ``item_id``), so the two orderings are mapped through
    ``meta["item_ids"]`` before they are compared. A sanity number only: the
    probe's selection is geometric by design, because "the row a per-slot table
    selects" is exactly ``argmin`` over the stored keys (Prop T5.4).
    """
    lab = np.asarray(getattr(qs, "label", []), dtype=int).ravel()
    item_ids = list(qs.meta.get("item_ids", []))
    if lab.size == 0 or not item_ids or np.any(lab < 0):
        return float("nan")
    pos = {int(i): k for k, i in enumerate(np.asarray(ids, dtype=int))}
    want = np.asarray([pos.get(int(item_ids[j]), -1) for j in lab], dtype=int)
    live = want >= 0
    return float(np.mean(sel[live] == want[live])) if np.any(live) else float("nan")


def _t55(d: float, s: float, sigma_q: float) -> float:
    """T5.5's exchange rate in closed form (reproduces its table to the digit)."""
    return float((d / sigma_q) * np.exp(-(d ** 2 - sigma_q ** 2) / (2.0 * s ** 2)))


def _min_sep(centers: np.ndarray) -> float:
    c = np.asarray(centers, dtype=float)
    if c.shape[0] < 2:
        return float("nan")
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


def run_experiment_thirdparty(config=None, save_dir: str = "results",
                              models_dir: Optional[str] = None,
                              seed: Optional[int] = None,
                              seeds: Sequence[int] = (0, 1, 2),
                              radii: Sequence[float] = BALL_RADIUS_GRID,
                              family: str = "overload",
                              arm: str = "load1x_shipped",
                              quick: bool = False,
                              out_json: Optional[str] = None) -> dict:
    """⭐⭐ The C6 probe across the ``d/s`` sweep, multi-seed, both x-axis conventions."""
    os.makedirs(save_dir, exist_ok=True)
    if quick:
        seeds, radii = tuple(seeds)[:1], tuple(radii)[:2]
    t_all = time.time()
    cells: List[dict] = []
    for r in radii:
        for s in seeds:
            c = run_thirdparty_cell(family, arm, int(s), quick=quick,
                                    ball_radius=float(r))
            cells.append(c)
            print(f"[R={r} s{s}] admissible={c['admissible']} ({c['reason'] or 'ok'}) "
                  f"sep={c.get('sep', float('nan')):.4f} "
                  f"d/s(proxy)={c.get('d_over_s_proxy', float('nan')):.2f} "
                  f"grad_ratio={c.get('grad_ratio', float('nan')):.3e}", flush=True)

    # --- aggregate per radius, over ADMISSIBLE seeds only -------------------
    per_radius: List[dict] = []
    for r in radii:
        grp = [c for c in cells if abs(c["ball_radius"] - float(r)) < 1e-9]
        adm = [c for c in grp if c["admissible"]]
        row: Dict[str, Any] = {
            "ball_radius": float(r), "n_cells": len(grp), "n_admissible": len(adm),
            "coverage": float(len(adm) / max(len(grp), 1)),
            "admissible_seeds": [c["seed"] for c in adm],
            "excluded": [{"seed": c["seed"], "reason": c["reason"]}
                         for c in grp if not c["admissible"]],
        }
        use = adm or []
        for k in ("sep", "d_third_mean", "s_fitted_well", "d_over_s_proxy",
                  "d_over_s_fitted", "grad_ratio", "t5_5_closed_form",
                  "lambda_min", "endpoint_write_loss_max"):
            v = [c[k] for c in use if np.isfinite(c.get(k, np.nan))]
            row[k] = float(np.mean(v)) if v else float("nan")
            if k == "grad_ratio":
                row["grad_ratio_sd"] = (float(np.std(v, ddof=1)) if len(v) > 1
                                        else float("nan"))
                row["grad_ratio_se"] = (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                                        if len(v) > 1 else float("nan"))
        # the per-slot curve, pooled over admissible seeds
        curve: Dict[Tuple[int, str], List[dict]] = {}
        for c in use:
            for rw in c["rows"]:
                curve.setdefault((rw["slot"], rw["channel"]), []).append(rw)
        row["curve"] = []
        for (slot, ch), rws in sorted(curve.items()):
            def _m(key, rws=rws):
                v = [x[key] for x in rws if np.isfinite(x[key])]
                return (float(np.mean(v)) if v else float("nan"),
                        float(np.std(v, ddof=1) / np.sqrt(len(v)))
                        if len(v) > 1 else float("nan"))
            third_m, third_se = _m("delta_third")
            own_m, _ = _m("delta_own")
            blank_m, blank_se = _m("delta_blank")
            launch_m, _ = _m("delta_launch")
            coup_m, coup_se = _m("coupling")
            marg = [x["margin_blank"] for x in rws if np.isfinite(x["margin_blank"])]
            mm = float(np.mean(marg)) if marg else float("nan")
            mse = (float(np.std(marg, ddof=1) / np.sqrt(len(marg)))
                   if len(marg) > 1 else float("nan"))
            row["curve"].append({
                "slot": slot, "channel": ch, "step": rws[0]["step"], "t": rws[0]["t"],
                "n_seeds": len(rws), "delta_third": third_m, "delta_third_se": third_se,
                "delta_own": own_m, "delta_blank": blank_m, "delta_blank_se": blank_se,
                "delta_launch": launch_m, "delta_table": 0.0,
                "coupling": coup_m, "coupling_se": coup_se,
                "margin_blank": mm, "margin_blank_se": mse,
                "clears_blank_2se": bool(len(marg) >= 3 and np.isfinite(mse)
                                         and (mm - 2.0 * mse) > 0.0),
                "clears_launch": bool(np.isfinite(third_m) and np.isfinite(launch_m)
                                      and third_m > launch_m),
            })
        # the peak coupling over the grid, per channel (the reported scalar)
        for ch in ("q", "p"):
            sub = [x for x in row["curve"] if x["channel"] == ch]
            row[f"coupling_peak_{ch}"] = (max((x["coupling"] for x in sub
                                               if np.isfinite(x["coupling"])),
                                              default=float("nan")))
            row[f"coupling_slot0_{ch}"] = next(
                (x["coupling"] for x in sub if x["slot"] == 0), float("nan"))
            row[f"clears_blank_{ch}"] = bool(any(x["clears_blank_2se"] for x in sub))
        per_radius.append(row)

    ok = [r for r in per_radius if np.isfinite(r["grad_ratio"]) and r["n_admissible"]]
    fits = {
        "gradient_ratio": coupling_law_fit(
            [r["d_over_s_proxy"] for r in ok], [r["grad_ratio"] for r in ok],
            sigma_q=0.15, d=[r["d_third_mean"] for r in ok]),
        "slot_coupling_q": coupling_law_fit(
            [r["d_over_s_proxy"] for r in ok], [r["coupling_slot0_q"] for r in ok],
            sigma_q=0.15, d=[r["d_third_mean"] for r in ok]),
    }
    out = {
        "probe": "C6 third-party store attribution (bprime-theory T5.4/T5.5)",
        "wall_s": time.time() - t_all, "seeds": [int(s) for s in seeds],
        "radii": [float(r) for r in radii], "family": family, "arm": arm,
        "quick": bool(quick), "slot_grid": list(THIRDPARTY_SLOT_GRID),
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); clears iff mean-2SE>0",
        "table_third_party_delta": "0 by construction (Prop T5.4) — computed, see cells",
        "per_radius": per_radius, "law_fits": fits, "cells": cells,
    }
    path = out_json or os.path.join(save_dir, "exp_route3_thirdparty.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    out["metrics_path"] = path
    try:
        out["figures"] = _plot_thirdparty(out, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        out["figures"] = []
        out["figure_error"] = repr(exc)
    return out


def _plot_thirdparty(res: dict, save_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in res["per_radius"] if r["n_admissible"]]
    if not rows:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    x = [r["d_over_s_proxy"] for r in rows]
    xf = [r["d_over_s_fitted"] for r in rows]
    ax.errorbar(x, [max(r["grad_ratio"], 1e-12) for r in rows],
                yerr=[2 * (r["grad_ratio_se"] or 0) for r in rows], marker="o",
                color="tab:blue", capsize=3, label="measured ∇V ratio (s = atom_width)")
    ax.plot(xf, [max(r["grad_ratio"], 1e-12) for r in rows], "s--", color="tab:cyan",
            label="same, x = d / fitted well width")
    ax.plot(x, [max(r["t5_5_closed_form"], 1e-12) for r in rows], "^:", color="k",
            label="T5.5 closed form exp(−½(d/s)²)")
    ax.plot(x, [1e-12] * len(x), "x", color="tab:red",
            label="per-slot TABLE: 0 by construction (T5.4)")
    for xv, lbl in ((1.95, "override-rig label"), (2.9, "soft cert"), (4.4, "designed gate")):
        ax.axvline(xv, color="grey", ls=":", lw=1)
        ax.text(xv, 1e-9, lbl, rotation=90, fontsize=6, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("d / s")
    ax.set_ylabel("third-party coupling (÷ the item's own)")
    ax.set_title("⭐ what a per-slot table structurally cannot do,\nand where our own "
                 "admission gate puts it", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        for ch, ls in (("q", "-"), ("p", "--")):
            sub = sorted([c for c in r["curve"] if c["channel"] == ch],
                         key=lambda c: c["step"])
            ax.plot([c["step"] for c in sub], [max(c["delta_third"], 1e-12) for c in sub],
                    ls, ms=3, marker="o" if ch == "q" else "s",
                    label=f"R={r['ball_radius']} {ch}")
    ax.set_yscale("log")
    ax.set_xlabel("integrator step t")
    ax.set_ylabel("|Δ slot| / σ_q  (delete a NON-selected item)")
    ax.set_title("the curve, not the endpoint (C7: t ∈ [1, 240])", fontsize=9)
    ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[2]
    r0 = rows[0]
    for ch, col in (("q", "tab:blue"), ("p", "tab:red")):
        sub = sorted([c for c in r0["curve"] if c["channel"] == ch],
                     key=lambda c: c["step"])
        ax.errorbar([c["step"] for c in sub], [c["delta_third"] for c in sub],
                    yerr=[2 * (c["delta_third_se"] or 0) for c in sub], marker="o",
                    ms=3, color=col, capsize=2, label=f"{ch}: third-party Δ")
        ax.plot([c["step"] for c in sub], [c["delta_blank"] for c in sub], ":",
                color=col, alpha=0.6, label=f"{ch}: blank-store delete control")
    ax.set_yscale("log")
    ax.set_xlabel("integrator step t")
    ax.set_ylabel("|Δ slot| / σ_q")
    ax.set_title(f"R = {r0['ball_radius']} (d/s = {r0['d_over_s_proxy']:.2f}): "
                 "signal vs its own control", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(save_dir, "exp_route3_thirdparty.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _plot(res: dict, save_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [f for f in res["curves"] if res["coverage_per_family"][f]["admissible_seeds"]]
    if not fams:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"q": "tab:blue", "p": "tab:red"}
    fam = fams[0]
    rows = res["a94_unlock_bar"][fam]["rows"]
    ax = axes[0][0]
    for ch in ("q", "p"):
        sub = sorted([r for r in rows if r["channel"] == ch], key=lambda r: r["t"])
        ax.errorbar([r["t"] for r in sub], [r["margin_mean"] for r in sub],
                    yerr=[2 * (r["margin_se"] or 0) for r in sub], marker="o",
                    ms=3, color=colors[ch], capsize=2,
                    label=f"{ch}-slots  (margin ± 2 SE)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("t (time units, log)")
    ax.set_ylabel("(full − settle-deleted) − launch-noise floor")
    ax.set_title(f"⭐ store-attribution curve — {fam}\n§A9.4: clears iff mean−2SE>0",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for ch in ("q", "p"):
        sub = sorted([r for r in rows if r["channel"] == ch], key=lambda r: r["t"])
        ax.plot([r["t"] for r in sub], [r["full_mean"] for r in sub], "o-", ms=3,
                color=colors[ch], label=f"{ch}: full")
        ax.plot([r["t"] for r in sub], [r["dividend_mean"] for r in sub], "s--",
                ms=3, color=colors[ch], alpha=0.5, label=f"{ch}: full − launder")
    ax.set_xscale("log")
    ax.set_xlabel("t (log)")
    ax.set_ylabel("discriminability |rho|")
    ax.set_title("the curve, not the endpoint", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    a95 = res["a95_table_launder"][fam]["rows"]
    for ch in ("q", "p"):
        sub = [r for r in a95 if r["channel"] == ch]
        ax.plot([r["slot"] for r in sub], [r["table_margin_mean"] for r in sub],
                "o-", ms=3, color=colors[ch], label=f"{ch}: read − table")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("slot index")
    ax.set_ylabel("read − per-slot table launder")
    ax.set_title("⛔ §A9.5 kill-condition (<= 0 ⇒ K time-indexed tables)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    jac = res["curves"][fam][0]["jacobian"]
    for ch in ("q", "p"):
        c = jac["channels"][ch]
        ax.plot(jac["slots"], c["contraction"], "o-", ms=3, color=colors[ch],
                label=f"{ch}: contraction ‖Δs_t‖/‖δ‖")
        ax.plot(jac["slots"], c["fisher"], "s--", ms=3, color=colors[ch], alpha=0.5,
                label=f"{ch}: separation (between/within)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("slot index")
    ax.set_title("§A8.2 flow-map Jacobian: contractive within, separated across",
                 fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(save_dir, "exp_route3_attribution.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def main():
    ap = argparse.ArgumentParser(
        description="Route 3 stage 1: the store-attribution curve (§A10) + the "
                    "§A9.4 unlock bar + the §A9.5 per-slot table launder.")
    ap.add_argument("--project")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--part", choices=["curve", "thirdparty"], default="curve")
    ap.add_argument("--radii", nargs="+", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()

    config, save_dir, models_dir = None, "results", None
    if a.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(a.project)
        paths = pm.get_paths(a.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        os.makedirs(save_dir, exist_ok=True)
    if a.part == "thirdparty":
        run_experiment_thirdparty(
            config=config, save_dir=save_dir, models_dir=models_dir, seed=a.seed,
            seeds=a.seeds, radii=tuple(a.radii or BALL_RADIUS_GRID),
            quick=a.quick, out_json=a.out)
        return
    run_experiment_route3_attribution(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=a.seed,
        families=a.families, seeds=a.seeds, quick=a.quick,
        escalate=not a.no_escalate, out_json=a.out)


if __name__ == "__main__":
    main()
