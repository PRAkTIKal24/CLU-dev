"""⭐⭐ **`c2w6-anti-erosion`** — does the store still get eroded by its own
outer loss? (charter §A20.6 / §A21 C2W6; `PREREG-AntiErosion.md`.)

**The mechanism under attack (N223, measured).** ``CluStoreCell.write`` is
differentiably unrolled inside the outer step, so the byte-LM loss reaches the
store's **initial-atom leaves** through the write (measured at the CSF3 run-2
config, toy: ``||dL/d amp|| = 9.3e-03``). Under a **net-cost** store the
optimiser uses that channel to teach the writer to stop writing: fresh-write
depth ``0.0288 -> 4.95e-63`` after 200 outer steps
(`pilot-placement-probe` §7 R3). Forgetting must happen through **designed**
channels — the decay law, eviction, the trash region — never as an optimiser
side effect.

**What this module measures**, all on the FULL block (the same system the CSF3
runs use) with exactly one capability toggled (ADDENDUM 6's binding design rule):

* **the erosion curve** — per-well fitted depth at the item's own site on the
  launch manifold vs outer step, sampled every ``monitor_every`` steps, each
  well tagged with its **last-write chunk** so a fresh write and post-write
  designed decay are separable (designed decay is not erosion);
* **P1** (``StreamMemoryConfig.erosion_partition``) — the stop-gradient partition
  at the write boundary, ± on paired seeds;
* **I1** (``refresh_monotonic``) — the refresh-on-rewrite monotonicity guard;
  first the baseline **audit** with the guard OFF (I1-a's event rate), then the
  guard (I1-b: zero violations, bit-identical when no violation would occur);
* **I2** — the usage-vs-erosion telemetry: per well, per monitor window, the
  fitted depth, the last-write chunk, a read-selection count, the outer loss's
  gradient magnitude into the well's own atoms, and — at a few checkpoints — a
  **leave-one-well-out** probe-batch bpc (the loss-contribution measurement).
  The Head's registered hypothesis is that the MOST useful wells erode FASTEST.

⛔ **Adjudication is mechanical**: :func:`aggregate` applies ``PREREG-
AntiErosion.md`` §4 verbatim and prints the run-3 verdict
(``EARNS_SLOT`` / ``FAILS_K3`` / ``FAILS_FLATTEN`` / ``K4_RELOCATED``) **either
way**, with the run-3 flag block beside it. The Advisor decides promotion; this
module only measures and applies the registered rule.

⚠ Monitor #13 / N94 travels with every ``w4`` reading (4 inner write steps
against a floor of 40): every w4 number here is formally NON-PROMOTABLE, and the
``w40_*`` pair is the undemoted confirmation. ⛔ TOY scale (0.16 M). ⛔ Depth is
not quotable as feature importance until I2 reports (charter §A21).

Runnable directly::

    PYTHONPATH=. python -m chlu.experiments.exp_anti_erosion \\
        --cells p1_off p1_on --seeds 0 1 2 --steps 1000
    PYTHONPATH=. python -m chlu.experiments.exp_anti_erosion --aggregate-only \\
        --out .claude/outputs/c2w6-anti-erosion
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.blocks import WritePlan, fitted_well_depth, make_memory_cell
from chlu.data.enwik8 import bits_per_character, contiguous_batches, random_batches
from chlu.experiments.exp_placement_probe import _prepare, multi_lane_self_probe
from chlu.training.train_cluformer import (
    _block_chunk_latents,
    _block_forward,
    _embed_stream,
    _eval_loss,
    build_arm,
    cell_group_depth,
    evaluate,
    loss_fn,
    plan_pass,
    save_json,
    solve_arms,
    train_arm,
)

# ==========================================================================
# the registered rig (PREREG-AntiErosion.md §1)
# ==========================================================================
#: ⭐ Every cell is the **CSF3 run-2 config** — `atom_place_radius=0.3`
#: (`h1b_m0.6`'s ``mem``), `write_margin=0.6` (its ``store``),
#: `psi_payload_residual=True`, `psi_residual_source=q_star`, all stage flags
#: TRUE — with **exactly one capability toggled**. No isolated-arm studies: the
#: residual-off pair is a labelled DIAGNOSTIC rider inside the full-system rig
#: (intervention §8.1), never a claim cell.
CELLS: Dict[str, Dict[str, Any]] = {
    "p1_off": dict(base="h1b_m0.6", mem={}, residual=True,
                   note="run-2 config, the partition OFF (the N223 baseline)"),
    "p1_on": dict(base="h1b_m0.6", mem={"erosion_partition": True},
                  residual=True, note="run-2 config + P1"),
    "p1_on_i1_on": dict(base="h1b_m0.6",
                        mem={"erosion_partition": True,
                             "refresh_monotonic": True},
                        residual=True,
                        note="run-2 config + P1 + I1 — the SHIPPED CANDIDATE"),
    # ⭐ the w40 pair: N94's maturity floor, shorter horizon. The only readings
    # monitor #13 does not demote.
    "w40_p1_off": dict(base="h1b_m0.6", mem={"write_inner_steps": 40},
                       residual=True,
                       note="w40 confirmation: partition OFF at N94's floor"),
    "w40_p1_on": dict(base="h1b_m0.6",
                      mem={"write_inner_steps": 40, "erosion_partition": True},
                      residual=True,
                      note="w40 confirmation: partition ON at N94's floor"),
    # ⛔ DIAGNOSTIC RIDER ONLY (§A20.6's "erosion intrinsic vs symptom" corner).
    # The psi payload residual measurably RESISTS R3 (psires §5: paired depth
    # 1.9x/4.7x/3.2x on 3/3 seeds), i.e. the one channel P1 severs is also the
    # one that gave the wells a reason to exist. These two cells price that
    # tension. They are NEVER a claim cell and never a null.
    "resoff_p1_off": dict(base="h1b_m0.6", mem={}, residual=False,
                          note="DIAGNOSTIC: residual OFF, partition OFF"),
    "resoff_p1_on": dict(base="h1b_m0.6", mem={"erosion_partition": True},
                         residual=False,
                         note="DIAGNOSTIC: residual OFF, partition ON"),
}

#: The claim cells (the diagnostics ride, they do not adjudicate).
CLAIM_CELLS: Tuple[str, ...] = ("p1_off", "p1_on", "p1_on_i1_on")
W40_CELLS: Tuple[str, ...] = ("w40_p1_off", "w40_p1_on")
DIAGNOSTIC_CELLS: Tuple[str, ...] = ("resoff_p1_off", "resoff_p1_on")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
#: Registered horizon and sampling cadence (prereg §1).
DEFAULT_STEPS: int = 1000
DEFAULT_W40_STEPS: int = 400
MONITOR_EVERY: int = 25

# -- the registered bands (prereg §3), as code -----------------------------
#: E1 — partition-OFF depth at step 1000 <= 0.3x its own step-200 value on >=2/3
#: seeds (band 0.02x-0.5x).
E1_POINT: float = 0.30
E1_BAND: Tuple[float, float] = (0.02, 0.50)
#: E2 — partition-ON depth at step 1000 >= 0.7x the step-200 value (band
#: [0.5, 1.05]), 3/3 seeds.
E2_POINT: float = 0.70
E2_BAND: Tuple[float, float] = (0.50, 1.05)
#: E3 / K3 — paired |Delta bpc(ON - OFF)| within +-0.01 at 1000 toy steps.
E3_TOL: float = 0.01
#: I1-a — rewrites reduce the fitted depth in 10-40 % of rewrite events
#: (band 2-60 %).
I1A_POINT: Tuple[float, float] = (0.10, 0.40)
I1A_BAND: Tuple[float, float] = (0.02, 0.60)
#: I2 — Spearman rho(usefulness, erosion rate) >= +0.5 (refutation branch
#: <= -0.3; |rho| < 0.3 is "no usage structure").
I2_POINT: float = 0.50
I2_REFUTE: float = -0.30
#: The residual-only banked final depth the P-residual interaction is scored
#: against (psires §5, residual_on at 200 steps).
P_RESIDUAL_BANKED_DEPTH: float = 0.1321

#: The gate's legs (prereg §4), in the order the verdict applies them.
GATE_LEGS: Tuple[str, ...] = ("E2_on_arm_flattens", "E1_off_arm_decays",
                              "K3_bpc_not_worse", "K4_not_relocated")
#: K4's float32 floor: below this a bpc margin is not distinguishable from zero.
K4_FLOAT32_FLOOR: float = 1e-6
#: ADDENDUM 1: a well below this fitted depth is at the collapse floor; ratios
#: taken against it are 0/0 and are scored as "already collapsed", not as noise.
DEPTH_COLLAPSE_FLOOR: float = 1e-6


# ==========================================================================
# small numeric helpers
# ==========================================================================
def _mean_se(v) -> Tuple[float, float, int]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return float(np.mean(a)), float(sd / np.sqrt(a.size)), int(a.size)


def _rank(v: Sequence[float]) -> np.ndarray:
    """Average ranks (ties shared) — the Spearman convention."""
    a = np.asarray(v, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(1, a.size + 1, dtype=float)
    # average the ranks of tied values
    for val in np.unique(a):
        m = a == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation — I2's registered statistic.

    ``nan`` when fewer than 3 finite pairs remain or either side is constant
    (a constant column has no rank structure; reporting 0.0 there would be a
    claim we did not measure).
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / den) if den > 0 else float("nan")


def _log_slope(steps: Sequence[float], depth: Sequence[float]) -> float:
    """Least-squares slope of ``ln(depth)`` vs step (``nan`` if under-determined).

    The **erosion rate** is ``-slope``: positive = the well is being erased.
    Non-positive depths are dropped (a well at 1e-63 is already dead and its
    log is dominated by float noise); fewer than 3 usable points => ``nan``.
    """
    s = np.asarray(steps, dtype=float)
    d = np.asarray(depth, dtype=float)
    m = np.isfinite(s) & np.isfinite(d) & (d > 0)
    if m.sum() < 3:
        return float("nan")
    s, d = s[m], np.log(d[m])
    if np.all(s == s[0]):
        return float("nan")
    return float(np.polyfit(s, d, 1)[0])


def _at_step(curve: List[Dict[str, Any]], step: int, key: str = "depth_median"
             ) -> float:
    """The reading nearest ``step`` (the cadence may not land on it exactly)."""
    rows = [r for r in curve if np.isfinite(float(r.get(key, np.nan)))]
    if not rows:
        return float("nan")
    r = min(rows, key=lambda r: abs(int(r["at_step"]) - int(step)))
    return float(r[key])


# ==========================================================================
# preparation — the run-2 rig + exactly one toggled capability
# ==========================================================================
def _prepare_erosion(cell: str, seed: int, *, steps: Optional[int] = None,
                     eval_batches: int = 4, monitor_every: int = MONITOR_EVERY):
    """`pilot-placement-probe`'s ``_prepare`` + the run-2 residual + the lever.

    ⭐ Paired by construction: the shell, the ``phi`` gain and the data order
    depend only on ``seed`` and are fixed BEFORE the lever is applied, so a
    partition-on and a partition-off cell at the same seed differ by one static
    flag and nothing else — same parameters, same batches, same plans at step 0.
    """
    spec = CELLS[cell]
    pcfg, data, k_solve, k_model, prov = _prepare(
        spec["base"], seed, steps=steps, eval_batches=eval_batches)
    pcfg.memory = dict(pcfg.memory)
    if bool(spec.get("residual", True)):
        pcfg.memory.update({"psi_payload_residual": True,
                            "psi_residual_source": "q_star",
                            "psi_residual_gain": 1.0,
                            "psi_residual_trainable": True})
    pcfg.memory.update(dict(spec.get("mem") or {}))
    pcfg.monitor_every = int(monitor_every)
    prov["base_cell"] = spec["base"]
    prov["cell_mem_overrides"] = dict(spec.get("mem") or {})
    prov["psi_payload_residual"] = bool(spec.get("residual", True))
    prov["monitor_every"] = int(monitor_every)
    return pcfg, data, k_solve, k_model, prov


# ==========================================================================
# ⭐⭐ I2 — the usage-vs-erosion telemetry (decision-inert to the model)
# ==========================================================================
@eqx.filter_jit
def _write_and_diag(cell, state, z, plan_c):
    """One chunk-write **and** its I1 audit, in one traced pass and one compile.

    ⚠ Calling ``write`` and ``write_diag`` separately would pay the inner write
    twice — at ``write_inner_steps = 40`` that is the whole telemetry budget.
    """
    return cell.write_and_diag(state, z, plan_c)


@eqx.filter_jit
def _site_depths(cell, state, sites, addr_dim: int):
    """``(K, 2)`` — each slot's fitted depth at its OWN site, two ways.

    ``[:, 0]`` = the slot's **own** atom rows (the published convention,
    :func:`~chlu.training.train_cluformer.cell_group_depth`); ``[:, 1]`` = every
    **foreign** row. The fit is linear in the row mask, so the total landscape
    depth at the site is exactly their sum.

    ⭐ The split is the **interference instrument**, and the split is the point.
    The masked write is C3-local in parameter space, so writing item B leaves
    item A's own rows untouched *except* for the designed decay law — column 0
    is therefore predicted exactly by ``D * group_scale^2`` and any deviation is
    a locality bug. Column 1 is the real interference channel (#9/#12): B's
    atoms dug NEAR A's site change the landscape A sits in without touching a
    single one of A's parameters. That is the channel I1's guard was written
    for, and it is the one that is measurable here.
    """
    gm = jnp.asarray(cell.group_matrix, dtype=jnp.float32)

    def one(z, rows):
        return jnp.stack([
            fitted_well_depth(state.centers, state.log_width, state.amp, rows,
                              z, addr_dim),
            fitted_well_depth(state.centers, state.log_width, state.amp,
                              1.0 - rows, z, addr_dim)])

    return jax.vmap(one)(sites, gm)


def _plan_without_slot(plan: WritePlan, slot: int) -> WritePlan:
    """The **leave-one-well-out** plan: the same stream with every chunk that
    would be admitted into ``slot`` refused instead.

    ⭐ This is the loss-contribution instrument. It is a *plan* edit, not a model
    edit, so the arm is the same trained block reading a store that never
    received one item — the honest form of "delete this well".
    """
    keep = (jnp.asarray(plan.slot) != int(slot)).astype(plan.admitted.dtype)
    return plan._replace(admitted=plan.admitted * keep)


def _read_selection_counts(pl0, z_lane: np.ndarray, addr_dim: int, K: int
                           ) -> List[int]:
    """⭐ The read-selection proxy: which live well each chunk's read is nearest.

    The CLU read is a *relaxation*, not a discrete lookup, so there is no
    selection event to count. The registered proxy (PREREG ADDENDUM 1) is the
    **nearest live site in address space to the launch point** — the well the
    read is launched into the basin of — evaluated against the store as it
    stood BEFORE that chunk's write (read-before-write, so chunk ``c`` sees
    ``0..c-1``).
    """
    live = np.asarray(pl0.live)
    sites = np.asarray(pl0.sites)
    cnt = [0] * int(K)
    for c in range(1, int(z_lane.shape[0])):
        lv = live[c - 1]
        idx = [i for i in range(int(K)) if lv[i] > 0.5]
        if not idx:
            continue
        q = np.asarray(z_lane[c])[:addr_dim]
        d = [float(np.linalg.norm(sites[c - 1, i, :addr_dim] - q)) for i in idx]
        cnt[idx[int(np.argmin(d))]] += 1
    return cnt


def post_guard_violations(events: Sequence[Dict[str, float]],
                          tol: float = 1e-6) -> int:
    """⭐ I1-b's actual statistic: rewrite events whose **final** (post-guard)
    depth is below the pre-write reference.

    ⚠ The write's own ``violation`` flag is the *pre-guard* verdict — it says
    "the inner write reduced the depth", which is exactly the event the guard
    exists to repair, so on a guard-ON arm it counts REPAIRS, not failures.
    I1-b is about what is left afterwards, and that is this. Computed from the
    per-event depths stored in the artifact, so it is uniform across cells that
    ran before this function existed.
    """
    return int(sum(1 for e in events
                   if float(e.get("rewrite", 0.0)) > 0.5
                   and float(e.get("depth_guarded", float("nan")))
                   < float(e.get("depth_before", float("nan"))) * (1.0 - tol)))


def _record_events(rec: Dict[str, Any]) -> List[Dict[str, float]]:
    """Every rewrite event of one run, pooled over its telemetry readings."""
    out: List[Dict[str, float]] = []
    for t in rec.get("telemetry", []) or []:
        out.extend(t.get("rewrite_events", []) or [])
    return out


def _interference_audit(site_depth: List[np.ndarray], live: np.ndarray,
                        slot: np.ndarray, admitted: np.ndarray,
                        group_scale: np.ndarray, tol: float = 1e-4
                        ) -> Dict[str, Any]:
    """⭐ **The measurable form of I1-a**: does writing item B disturb item A?

    An **event** is a pair ``(A, c)``: item ``A`` was live before chunk ``c``,
    and chunk ``c`` was admitted into a DIFFERENT slot. Two legs, and the split
    is what makes the reading honest:

    * ``own`` — A's own atom rows. The write is masked to B's rows, so the ONLY
      designed change is the decay law: the prediction is
      ``D_after = D_before * group_scale^2`` **exactly**. ``n_down_own`` counts
      violations of that prediction, i.e. depth lost beyond the designed decay.
      A non-zero count is a C3-locality bug and this leg is a live regression
      check, not a result.
    * ``foreign`` — every other row's contribution at A's site. This is the real
      interference channel (#9/#12): B's atoms dug near A change the landscape A
      sits in **without touching one of A's parameters**. Reported signed —
      ``n_up_foreign`` (a neighbour crowding in) and the median relative change.

    ``site_depth[c]`` is the ``(K, 2)`` reading taken BEFORE chunk ``c``'s write
    (index 0 = before any write), so ``c`` and ``c + 1`` bracket it.
    """
    n = min(len(site_depth) - 1, int(live.shape[0]))
    out: Dict[str, Any] = {"n_events": 0, "n_down_own": 0, "n_up_foreign": 0,
                           "n_down_foreign": 0}
    d_own, d_for, res = [], [], []
    for c in range(1, n):
        if admitted[c] <= 0.5:
            continue
        s_c = int(slot[c])
        before, after = site_depth[c], site_depth[c + 1]
        for k in range(int(live.shape[1])):
            if k == s_c or live[c - 1, k] <= 0.5:
                continue
            out["n_events"] += 1
            gs2 = float(group_scale[c, k]) ** 2
            b0, a0 = float(before[k, 0]), float(after[k, 0])
            pred = b0 * gs2
            if pred > 0:
                res.append((pred - a0) / pred)
                if a0 < pred * (1.0 - tol):
                    out["n_down_own"] += 1
                d_own.append((b0 - a0) / b0 if b0 > 0 else float("nan"))
            b1, a1 = float(before[k, 1]), float(after[k, 1])
            if a1 > b1:
                out["n_up_foreign"] += 1
            elif a1 < b1:
                out["n_down_foreign"] += 1
            if abs(b1) > 0:
                d_for.append((a1 - b1) / abs(b1))
    ne = out["n_events"]
    out["rate_down_own_beyond_decay"] = (out["n_down_own"] / ne if ne
                                         else float("nan"))
    out["rate_up_foreign"] = out["n_up_foreign"] / ne if ne else float("nan")
    out["median_own_residual_vs_decay_law"] = (float(np.median(res)) if res
                                               else float("nan"))
    out["max_abs_own_residual_vs_decay_law"] = (float(np.max(np.abs(res)))
                                                if res else float("nan"))
    out["median_rel_drop_own"] = float(np.median(d_own)) if d_own else float("nan")
    out["median_rel_change_foreign"] = (float(np.median(d_for)) if d_for
                                        else float("nan"))
    out["max_rel_change_foreign"] = (float(np.max(np.abs(d_for))) if d_for
                                     else float("nan"))
    return out


def well_telemetry(model, pcfg, tokens, *, layer: int = 0, loo: bool = False,
                   loo_batches: Optional[List] = None) -> Dict[str, Any]:
    """⭐⭐ **I2's artifact**: one per-well reading of the running block.

    Per well (atom group / slot) of **lane 0**, at this outer step:

    ==========================  ==============================================
    ``depth``                   fitted depth at the item's own site
    ``last_write_chunk``        the chunk the slot was last admitted at — the
                                tag that separates a FRESH write from designed
                                post-write decay
    ``n_writes``                admitted writes into the slot this stream
    ``read_selection``          the nearest-live-site read proxy
    ``grad_atoms``              ``||dL_outer/d(this slot's atom rows)||`` — the
                                erosion DRIVER (exactly 0.0 under P1)
    ``loo_delta_bpc``           leave-one-well-out probe bpc minus the full
                                store's (only at LOO checkpoints)
    ==========================  ==============================================

    plus the stream's **I1 rewrite audit** (event count, violation count, the
    per-event depth stages) — the same numbers the guard acts on, from the same
    code path.

    ⛔ **Decision-inert to the model**: nothing here is fed back into training.
    """
    from chlu.core.blocks import CluStoreCell

    t0 = time.time()
    blk = model.blocks[layer]
    cell = blk.cell
    if not isinstance(cell, CluStoreCell):
        return {"applicable": False, "why": type(cell).__name__}
    scfg = pcfg.store_cfg()
    d, K = int(scfg.addr_dim), int(scfg.capacity)
    tk = jnp.asarray(tokens, dtype=jnp.int32)[:1]
    plans, _ = plan_pass(model, tk, pcfg)
    h = _embed_stream(model, tk)
    for i in range(layer):
        h = _block_forward(model.blocks[i], h, plans[i])
    z = _block_chunk_latents(blk, h)
    pl0 = jax.tree_util.tree_map(lambda a: a[0], plans[layer])

    # -- the stream, with the I1 audit taken from the write itself ----------
    st = cell.init_state()
    events: List[Dict[str, float]] = []
    last_write = [-1] * K
    n_writes = [0] * K
    all_sites = np.asarray(pl0.sites)
    all_live = np.asarray(pl0.live)
    all_slot = np.asarray(pl0.slot)
    all_adm = np.asarray(pl0.admitted)
    site_depth = [np.asarray(_site_depths(cell, st, jnp.asarray(all_sites[0]), d))]
    for c in range(int(z.shape[1])):
        pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl0)
        st, dg = _write_and_diag(cell, st, z[0, c], pc)
        row = {k: float(np.asarray(v)) for k, v in dg.items()}
        row["chunk"] = int(c)
        events.append(row)
        s = int(np.asarray(pc.slot))
        if float(np.asarray(pc.admitted)) > 0.5:
            last_write[s] = int(c)
            n_writes[s] += 1
        # ⭐ the interference reading: every live item's own well, refitted
        # AFTER this chunk's write, group-restricted and globally.
        site_depth.append(np.asarray(
            _site_depths(cell, st, jnp.asarray(all_sites[c]), d)))
    inter = _interference_audit(site_depth, all_live, all_slot, all_adm,
                                np.asarray(pl0.group_scale))

    live = np.asarray(pl0.live)[-1]
    sites = np.asarray(pl0.sites)[-1]
    sel = _read_selection_counts(pl0, np.asarray(z[0]), d, K)

    # -- the outer loss's gradient into each well's own atoms ---------------
    tg = jnp.concatenate([tk[:, 1:], tk[:, -1:]], axis=1)
    _l, grads = eqx.filter_value_and_grad(loss_fn)(model, tk, tg, plans)
    ga = grads.blocks[layer].cell.clu.potential_net.learned
    g_c = np.asarray(ga.centers, dtype=float)
    g_w = np.asarray(ga.log_width, dtype=float)
    g_a = np.asarray(ga.amp, dtype=float)
    gm = np.asarray(cell.group_matrix, dtype=bool)

    # -- leave-one-well-out probe bpc (the loss-contribution measurement) ---
    loo_delta: Dict[int, float] = {}
    if loo and loo_batches:
        base, per_slot = [], {i: [] for i in range(K)}
        for x, y in loo_batches:
            bx = jnp.asarray(x, dtype=jnp.int32)
            by = jnp.asarray(y, dtype=jnp.int32)
            pls, _ = plan_pass(model, bx, pcfg)
            base.append(float(_eval_loss(model, bx, by, pls, None, None)))
            for i in range(K):
                pl_i = list(pls)
                pl_i[layer] = _plan_without_slot(pls[layer], i)
                per_slot[i].append(
                    float(_eval_loss(model, bx, by, pl_i, None, None)))
        b = bits_per_character(float(np.mean(base)))
        for i in range(K):
            loo_delta[i] = bits_per_character(float(np.mean(per_slot[i]))) - b

    wells = []
    for i in range(K):
        rows = gm[i]
        D, s_eff = cell_group_depth(cell, st, i, sites[i, :d])
        gn = float(np.sqrt((g_c[rows] ** 2).sum() + (g_w[rows] ** 2).sum()
                           + (g_a[rows] ** 2).sum()))
        wells.append({
            "slot": i, "live": float(live[i]), "depth": float(D),
            "s_eff": float(s_eff), "last_write_chunk": int(last_write[i]),
            "n_writes": int(n_writes[i]), "read_selection": int(sel[i]),
            "grad_atoms": gn,
            "site_addr": [float(v) for v in sites[i, :d]],
            "loo_delta_bpc": float(loo_delta.get(i, float("nan"))),
        })
    dep = np.asarray([w["depth"] for w in wells if w["live"] > 0.5], dtype=float)
    ev_r = [e for e in events if e["rewrite"] > 0.5]
    return {
        "applicable": True,
        "n_live": int((live > 0.5).sum()),
        "n_chunks": int(z.shape[1]),
        "depth_median": float(np.median(dep)) if dep.size else float("nan"),
        "depth_mean": float(np.mean(dep)) if dep.size else float("nan"),
        "wells": wells,
        "grad_atoms_total": float(np.sqrt((g_c ** 2).sum() + (g_w ** 2).sum()
                                          + (g_a ** 2).sum())),
        # ⭐ I1-a: the rewrite audit, from the write's own code path
        "n_rewrite_events": len(ev_r),
        "n_rewrite_violations": int(sum(1 for e in ev_r
                                        if e["violation"] > 0.5)),
        "n_rewrite_violations_post_guard": post_guard_violations(ev_r),
        "rewrite_violation_rate": (float(np.mean([e["violation"] for e in ev_r]))
                                   if ev_r else float("nan")),
        "rewrite_events": ev_r,
        # ⚠ the audit's decomposition: how many admitted writes landed on an
        # occupied slot at all, and how many of those were EVICTION re-draws (a
        # designed channel, excluded from the event set by construction).
        "n_admitted": int(sum(1 for e in events if e["admitted"] > 0.5)),
        "n_occupied_target": int(sum(1 for e in events
                                     if e["admitted"] > 0.5
                                     and e["occupied"] > 0.5)),
        "n_evicting_target": int(sum(1 for e in events
                                     if e["admitted"] > 0.5
                                     and e["evicting"] > 0.5)),
        # ⭐ the interference channel I1-a's rate was a proxy for (#9/#12)
        "interference": inter,
        "loo_ran": bool(loo and loo_batches),
        "wall_s": time.time() - t0,
    }


# ==========================================================================
# the erosion-curve harness
# ==========================================================================
def _memory_deleted_bpc(model, pcfg, ev) -> float:
    """The **memory-deleted** arm as an EVAL-time swap, not a retrained arm.

    ⚠ Declared: K4's third column here is the trained block with its memory
    cell replaced by :class:`~chlu.core.blocks.NullMemoryCell` (the read returns
    zeros), NOT a separately-trained ``none`` arm. It is the tier-appropriate
    *detector* asked for by §A20.6's caveat — "did the collapse relocate into
    the block's other weights?" — measured on the same weights; a retrained
    ``none`` arm is additionally reported by :func:`run_cell` when it is run.
    """
    cells = [make_memory_cell("none", latent_dim=b.cell.latent_dim,
                              key=jax.random.PRNGKey(0))
             for b in model.blocks]
    m = eqx.tree_at(lambda mm: [b.cell for b in mm.blocks], model,
                    replace=cells, is_leaf=lambda x: x is None)
    return evaluate(m, pcfg, iter(ev))["bpc"]


def run_cell(cell: str, seed: int, *, steps: Optional[int] = None,
             eval_batches: int = 4, loo_batches: int = 2,
             monitor_every: int = MONITOR_EVERY, with_none: bool = True,
             n_loo_checkpoints: int = 4, store_watch: bool = True
             ) -> Dict[str, Any]:
    """One (cell, seed): train ``steps`` outer steps and return the curve.

    The record is the prereg's unit of adjudication — the depth curve, the I1
    audit, the I2 series, and the three bpc columns (live / blank-store /
    memory-deleted) K4 needs.
    """
    t0 = time.time()
    pcfg, (tr, va, te), k_solve, k_model, prov = _prepare_erosion(
        cell, seed, steps=steps, eval_batches=eval_batches,
        monitor_every=monitor_every)
    # ⚠ declared cut: the trainer's own store_watch is a SECOND depth series on
    # different tokens; at w40 it costs a full 40-step write replay per window
    # for a quantity this module already measures. Off => `store_health` is
    # empty and the curve comes from the telemetry alone.
    pcfg.store_watch = bool(store_watch)
    specs, ledger = solve_arms(pcfg, k_solve)
    batches = list(random_batches(tr, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                  n_batches=pcfg.steps, seed=pcfg.seed))
    ev = list(contiguous_batches(te, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                 n_batches=int(eval_batches)))
    x0, _y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                           seq_len=pcfg.seq_len, n_batches=1)))
    loo_ev = ev[: max(1, int(loo_batches))]
    # LOO is the expensive column (K forwards per eval batch), so it is taken at
    # a few evenly-spaced checkpoints, declared, not at every window.
    n_ck = max(1, int(n_loo_checkpoints))
    ck = {0} | {int(round(j * (pcfg.steps - 1) / (n_ck - 1))) if n_ck > 1
                else pcfg.steps - 1 for j in range(1, n_ck)}
    ck = {int(monitor_every * round(c / monitor_every)) for c in ck} | {0}

    model = build_arm("clu_store", pcfg, specs, key=k_model)
    series: List[Dict[str, Any]] = []

    def probe(m, step, tag):
        return well_telemetry(m, pcfg, x0,
                              loo=(step in ck or step == pcfg.steps - 1),
                              loo_batches=loo_ev)

    model, hist = train_arm("clu_store", model, pcfg, iter(batches),
                            probe=probe, probe_out=series)

    live = evaluate(model, pcfg, iter(ev))
    blank = evaluate(model, pcfg, iter(ev), blank=True)
    mdel = _memory_deleted_bpc(model, pcfg, ev)
    pooled = multi_lane_self_probe(model, pcfg, x0)

    none_bpc = float("nan")
    if with_none:
        # the retrained memory-deleted arm (the probe's own convention). It does
        # not depend on any flag under test, so at a fixed seed it is the SAME
        # number in every cell — which is a free internal consistency check.
        nm = build_arm("none", pcfg, specs, key=k_model)
        nm, _ = train_arm("none", nm, pcfg, iter(batches))
        none_bpc = evaluate(nm, pcfg, iter(ev))["bpc"]

    curve = [{"at_step": r["at_step"], "tag": r["tag"],
              "depth_median": r.get("depth_median", float("nan")),
              "n_live": r.get("n_live", 0),
              "grad_atoms_total": r.get("grad_atoms_total", float("nan")),
              "n_rewrite_events": r.get("n_rewrite_events", 0),
              "n_rewrite_violations": r.get("n_rewrite_violations", 0)}
             for r in series]
    d0 = _at_step(curve, 0)
    d200 = _at_step(curve, 200)
    dfin = curve[-1]["depth_median"] if curve else float("nan")
    n_ev = int(sum(r.get("n_rewrite_events", 0) for r in series))
    n_vi = int(sum(r.get("n_rewrite_violations", 0) for r in series))
    n_vp = int(sum(post_guard_violations(r.get("rewrite_events", []) or [])
                   for r in series))
    ia = [r.get("interference", {}) for r in series if r.get("interference")]
    inter = {
        "n_events_total": int(sum(int(x.get("n_events", 0)) for x in ia)),
        "n_down_own_total": int(sum(int(x.get("n_down_own", 0)) for x in ia)),
        "n_up_foreign_total": int(sum(int(x.get("n_up_foreign", 0))
                                      for x in ia)),
        "rate_down_own_beyond_decay_mean": float(np.nanmean(
            [x.get("rate_down_own_beyond_decay", np.nan) for x in ia])) if ia
        else float("nan"),
        "rate_up_foreign_mean": float(np.nanmean(
            [x.get("rate_up_foreign", np.nan) for x in ia])) if ia
        else float("nan"),
        "max_abs_own_residual_vs_decay_law": float(np.nanmax(
            [x.get("max_abs_own_residual_vs_decay_law", np.nan) for x in ia]))
        if ia else float("nan"),
        "median_rel_change_foreign_mean": float(np.nanmean(
            [x.get("median_rel_change_foreign", np.nan) for x in ia])) if ia
        else float("nan"),
        "per_reading": ia,
    }

    rec = {
        "cell": cell, "seed": int(seed), "tier": "erosion",
        "provenance": prov, "steps": int(pcfg.steps),
        "monitor_every": int(monitor_every),
        "store_watch": bool(store_watch),
        "store_health": hist.get("store_health", []),
        "cell_ledger": ledger.get("clu_store"),
        "curve": curve,
        "telemetry": series,
        "depth_untrained": d0,
        "depth_at_200": d200,
        "depth_final": dfin,
        "depth_ratio_1000_over_200": (dfin / d200) if (np.isfinite(d200)
                                                       and d200 > 0)
        else float("nan"),
        "depth_ratio_final_over_untrained": (dfin / d0) if (np.isfinite(d0)
                                                            and d0 > 0)
        else float("nan"),
        "bpc_live": live["bpc"], "bpc_blank": blank["bpc"],
        "bpc_memory_deleted_eval": mdel,
        "bpc_none_retrained": none_bpc,
        "bpc_live_minus_blank": live["bpc"] - blank["bpc"],
        "bpc_memory_deleted_minus_live": mdel - live["bpc"],
        "bpc_none_minus_live": none_bpc - live["bpc"],
        "acq": float(pooled.get("acq_live", float("nan"))),
        "acq_blank": float(pooled.get("acq_blank", float("nan"))),
        "chance": float(pooled.get("chance", float("nan"))),
        # I1-a / I1-b
        "n_rewrite_events": n_ev,
        "n_rewrite_violations": n_vi,
        "rewrite_violation_rate": (n_vi / n_ev) if n_ev else float("nan"),
        "n_rewrite_violations_post_guard": n_vp,
        "rewrite_violation_rate_post_guard": ((n_vp / n_ev) if n_ev
                                              else float("nan")),
        "n_admitted": int(sum(r.get("n_admitted", 0) for r in series)),
        "n_occupied_target": int(sum(r.get("n_occupied_target", 0)
                                     for r in series)),
        "n_evicting_target": int(sum(r.get("n_evicting_target", 0)
                                     for r in series)),
        "interference": inter,
        # I2
        "i2": i2_correlations(series),
        "plan_pass_frac": hist["plan_pass_frac"],
        "wall_s": time.time() - t0,
    }
    print(f"[erosion {cell} s{seed}] depth {d0:.4g} -> {dfin:.4g} "
          f"(x{rec['depth_ratio_final_over_untrained']:.3g} vs untrained, "
          f"x{rec['depth_ratio_1000_over_200']:.3g} vs step200) | bpc "
          f"{live['bpc']:.4f} (live-blank {rec['bpc_live_minus_blank']:+.2e}, "
          f"memdel-live {rec['bpc_memory_deleted_minus_live']:+.4f}) | rewrites "
          f"{n_vi}/{n_ev} violated | rho_sel "
          f"{rec['i2'].get('rho_read_selection', float('nan')):+.3f} | "
          f"{rec['wall_s']:.0f}s", flush=True)
    return rec


# ==========================================================================
# ⭐ I2 — the correlation the Head's hypothesis is about
# ==========================================================================
def i2_correlations(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``rho(well usefulness, well erosion rate)`` over the wells of one run.

    * **erosion rate** = ``-`` the least-squares slope of ``ln(depth)`` vs outer
      step, per well, over the run's readings (positive = eroding).
    * **usefulness**, both registered proxies, reported separately and never
      pooled: the mean **read-selection** count, and the mean **leave-one-
      well-out** ``Delta bpc`` over the LOO checkpoints (a well whose deletion
      COSTS bpc is useful, so this column is used as-is: larger = more useful).

    ⚠ Provisional by construction: the analyst adjudicates ``rho`` on the raw
    per-well series (which this returns alongside), this is the harness's own
    computation of the same statistic.
    """
    rows = [r for r in series if r.get("applicable") and r.get("wells")]
    if len(rows) < 3:
        return {"n_readings": len(rows), "why": "fewer than 3 readings"}
    K = len(rows[0]["wells"])
    steps = [int(r["at_step"]) for r in rows]
    out_wells = []
    for i in range(K):
        dep = [float(r["wells"][i]["depth"]) for r in rows]
        sel = [float(r["wells"][i]["read_selection"]) for r in rows]
        gr = [float(r["wells"][i]["grad_atoms"]) for r in rows]
        loo = [float(r["wells"][i]["loo_delta_bpc"]) for r in rows
               if np.isfinite(r["wells"][i]["loo_delta_bpc"])]
        livef = float(np.mean([float(r["wells"][i]["live"]) for r in rows]))
        out_wells.append({
            "slot": i, "live_frac": livef,
            "erosion_rate": -_log_slope(steps, dep),
            "depth_first": dep[0], "depth_last": dep[-1],
            "depth_ratio": (dep[-1] / dep[0]) if dep[0] > 0 else float("nan"),
            "mean_read_selection": float(np.mean(sel)),
            "mean_grad_atoms": float(np.mean(gr)),
            "mean_loo_delta_bpc": float(np.mean(loo)) if loo else float("nan"),
            "n_loo": len(loo),
        })
    use = [w for w in out_wells if w["live_frac"] > 0.5]
    rate = [w["erosion_rate"] for w in use]
    return {
        "n_readings": len(rows), "n_wells": len(use),
        "rho_read_selection": spearman([w["mean_read_selection"] for w in use],
                                       rate),
        "rho_loo_delta_bpc": spearman([w["mean_loo_delta_bpc"] for w in use],
                                      rate),
        "rho_grad_atoms": spearman([w["mean_grad_atoms"] for w in use], rate),
        "wells": out_wells,
    }


# ==========================================================================
# ⭐⭐ the gate — prereg §4, applied verbatim and mechanically
# ==========================================================================
def _paired(records: List[Dict[str, Any]], on: str, off: str, key: str
            ) -> Tuple[List[float], List[int]]:
    """Per-seed ``on - off`` differences for ``key`` (paired seeds only)."""
    a = {int(r["seed"]): r for r in records if r["cell"] == on}
    b = {int(r["seed"]): r for r in records if r["cell"] == off}
    seeds = sorted(set(a) & set(b))
    return ([float(a[s][key]) - float(b[s][key]) for s in seeds], seeds)


def gate_verdict(records: List[Dict[str, Any]], *, on: str = "p1_on",
                 off: str = "p1_off") -> Dict[str, Any]:
    """⭐⭐ **The CSF3 run-3 gate** (PREREG-AntiErosion.md §4, verbatim).

    *The component earns the run-3 config slot iff the erosion curve flattens
    (E2 met: >= 0.5x band, 3/3 seeds, while the OFF arm decays per E1) with bpc
    not worse beyond 2 SE (K3 green), multi-seed, K4 not fired.*

    Returns the verdict EITHER WAY — ``EARNS_SLOT`` / ``FAILS_FLATTEN`` /
    ``FAILS_K3`` / ``K4_RELOCATED`` (or ``INSUFFICIENT_DATA``). The Advisor
    decides promotion; this only applies the registered rule.
    """
    ra = [r for r in records if r["cell"] == on]
    rb = [r for r in records if r["cell"] == off]
    seeds = sorted({int(r["seed"]) for r in ra} & {int(r["seed"]) for r in rb})
    legs: Dict[str, Any] = {}
    if len(seeds) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "on": on, "off": off,
                "n_paired_seeds": len(seeds),
                "legs": {k: None for k in GATE_LEGS}}

    ron = {int(r["seed"]): r for r in ra}
    roff = {int(r["seed"]): r for r in rb}
    r_on = [float(ron[s]["depth_ratio_1000_over_200"]) for s in seeds]
    r_off = [float(roff[s]["depth_ratio_1000_over_200"]) for s in seeds]
    legs["E2_on_arm_flattens"] = {
        "rule": f"depth(final)/depth(200) >= {E2_BAND[0]} on ALL seeds",
        "per_seed": r_on, "n_met": int(sum(x >= E2_BAND[0] for x in r_on)),
        "passed": bool(all(np.isfinite(x) and x >= E2_BAND[0] for x in r_on)),
    }
    # ⚠ ADDENDUM 1: an OFF arm already at the collapse floor by step 200 has a
    # ratio of 0/0. That is a STRONGER form of E1, not a failure of it, so it is
    # scored as met and labelled.
    d200_off = [float(roff[s]["depth_at_200"]) for s in seeds]
    coll = [bool(np.isfinite(x) and x < DEPTH_COLLAPSE_FLOOR) for x in d200_off]
    met_off = [bool((np.isfinite(r) and r <= E1_BAND[1]) or c)
               for r, c in zip(r_off, coll, strict=True)]
    legs["E1_off_arm_decays"] = {
        "rule": f"depth(final)/depth(200) <= {E1_BAND[1]} on >= 2/3 seeds, OR "
                f"already at the collapse floor ({DEPTH_COLLAPSE_FLOOR}) by "
                f"step 200 (a stronger form of E1)",
        "per_seed": r_off, "depth_at_200_per_seed": d200_off,
        "collapsed_by_200": coll,
        "n_met": int(sum(met_off)),
        "passed": bool(sum(met_off) >= max(2, len(seeds) - 1)),
    }
    dif, _s = _paired(records, on, off, "bpc_live")
    mu, se, _n = _mean_se(dif)
    legs["K3_bpc_not_worse"] = {
        "rule": f"paired Delta bpc(ON-OFF) not worse than 2 SE AND within "
                f"+-{E3_TOL} (E3's registered equivalence band)",
        "delta_per_seed": dif, "delta_mean": mu, "delta_se": se,
        "passed": bool(np.isfinite(mu) and not (mu > 2.0 * se and mu > E3_TOL)),
    }
    lb = [abs(float(ron[s]["bpc_live_minus_blank"])) for s in seeds]
    md = [float(ron[s]["bpc_memory_deleted_minus_live"]) for s in seeds]
    fired = bool(np.mean(lb) < K4_FLOAT32_FLOOR
                 and np.mean(md) <= K4_FLOAT32_FLOOR)
    legs["K4_not_relocated"] = {
        "rule": "K4 FIRES if the ON arm's |live-blank| is at the float32 floor "
                "AND the memory-deleted margin is not positive: depth would be "
                "protected while the store stayed useless, i.e. the collapse "
                "relocated into the block's other weights",
        "abs_live_minus_blank": lb, "memory_deleted_minus_live": md,
        "fired": fired, "passed": bool(not fired),
    }
    if not (legs["E2_on_arm_flattens"]["passed"]
            and legs["E1_off_arm_decays"]["passed"]):
        verdict = "FAILS_FLATTEN"
    elif not legs["K3_bpc_not_worse"]["passed"]:
        verdict = "FAILS_K3"
    elif legs["K4_not_relocated"]["fired"]:
        verdict = "K4_RELOCATED"
    else:
        verdict = "EARNS_SLOT"
    return {"verdict": verdict, "on": on, "off": off, "seeds": seeds,
            "n_paired_seeds": len(seeds), "legs": legs,
            "caveats": ["toy scale (0.16 M) — no pilot-scale claim",
                        "monitor #13 / N94 demotes every w4 reading; the w40 "
                        "pair is the undemoted confirmation",
                        "the tier-appropriate control is the system-level swap "
                        "(K4); the settle-deleted launder is inherited "
                        "diagnostic only"]}


def prereg_scorecard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Registered -> measured -> verdict, for E1/E2/E3, I1-a/b and P-residual."""
    def cell(name):
        return [r for r in records if r["cell"] == name]

    out: Dict[str, Any] = {}
    off, on, i1 = cell("p1_off"), cell("p1_on"), cell("p1_on_i1_on")
    if off:
        v = [float(r["depth_ratio_1000_over_200"]) for r in off]
        out["E1"] = {"registered": f"<= {E1_POINT}x on >=2/3 seeds "
                                   f"(band {E1_BAND})",
                     "measured_per_seed": v,
                     "n_met_point": int(sum(x <= E1_POINT for x in v
                                            if np.isfinite(x))),
                     "n_met_band": int(sum(x <= E1_BAND[1] for x in v
                                           if np.isfinite(x))),
                     "n_seeds": len(v)}
    if on:
        v = [float(r["depth_ratio_1000_over_200"]) for r in on]
        out["E2"] = {"registered": f">= {E2_POINT}x (band {E2_BAND}), 3/3 seeds",
                     "measured_per_seed": v,
                     "n_met_point": int(sum(x >= E2_POINT for x in v
                                            if np.isfinite(x))),
                     "n_met_band": int(sum(x >= E2_BAND[0] for x in v
                                           if np.isfinite(x))),
                     "n_inside_band": int(sum(E2_BAND[0] <= x <= E2_BAND[1]
                                              for x in v if np.isfinite(x))),
                     "n_above_band": int(sum(x > E2_BAND[1] for x in v
                                             if np.isfinite(x))),
                     "n_seeds": len(v)}
    if on and off:
        dif, seeds = _paired(records, "p1_on", "p1_off", "bpc_live")
        mu, se, _n = _mean_se(dif)
        out["E3"] = {"registered": f"paired |Delta bpc| <= {E3_TOL}",
                     "delta_per_seed": dif, "seeds": seeds,
                     "mean": mu, "se": se,
                     "met": bool(np.isfinite(mu) and abs(mu) <= E3_TOL)}
        # ⭐ the P-residual interaction (prereg §3, last row)
        fin_on = [float(r["depth_final"]) for r in on]
        out["P_residual_interaction"] = {
            "registered": f"partition-ON final depth >= the residual-only "
                          f"banked {P_RESIDUAL_BANKED_DEPTH} on >=2/3 seeds; "
                          f"if partition-ON depth COLLAPSES BELOW the "
                          f"partition-OFF arm, P1 is disproved as specified",
            "on_final_per_seed": fin_on,
            "off_final_per_seed": [float(r["depth_final"]) for r in off],
            "n_ge_banked": int(sum(x >= P_RESIDUAL_BANKED_DEPTH
                                   for x in fin_on if np.isfinite(x))),
            "on_below_off_seeds": int(sum(
                1 for s in seeds
                if float([r for r in on if r["seed"] == s][0]["depth_final"])
                < float([r for r in off if r["seed"] == s][0]["depth_final"]))),
            "n_seeds": len(seeds),
        }
    if off:
        rt = [float(r["rewrite_violation_rate"]) for r in off]
        mu, se, n = _mean_se(rt)
        out["I1a"] = {"registered": f"{I1A_POINT[0]:.0%}-{I1A_POINT[1]:.0%} of "
                                    f"rewrite events (band {I1A_BAND})",
                      "measured_per_seed": rt, "mean": mu, "se": se, "n": n,
                      "n_events_per_seed": [int(r["n_rewrite_events"])
                                            for r in off],
                      "in_point": bool(np.isfinite(mu)
                                       and I1A_POINT[0] <= mu <= I1A_POINT[1]),
                      "in_band": bool(np.isfinite(mu)
                                      and I1A_BAND[0] <= mu <= I1A_BAND[1])}
    if i1:
        nv = [int(r["n_rewrite_violations"]) for r in i1]
        npost = [post_guard_violations(_record_events(r)) for r in i1]
        out["I1b"] = {"registered": "depth-reduction events = EXACTLY 0 by "
                                    "construction; a violation-free write is "
                                    "bit-identical",
                      "violations_pre_guard_per_seed": nv,
                      "violations_post_guard_per_seed": npost,
                      "events_per_seed": [int(r["n_rewrite_events"])
                                          for r in i1],
                      "note": "the pre-guard count is how often the guard "
                              "FIRED (the repairs); I1-b is the post-guard "
                              "count",
                      "met": bool(all(x == 0 for x in npost))}
    if on:
        rho = [float(r.get("i2", {}).get("rho_read_selection", float("nan")))
               for r in off] if off else []
        rho_l = [float(r.get("i2", {}).get("rho_loo_delta_bpc", float("nan")))
                 for r in off] if off else []
        mu, se, n = _mean_se(rho)
        mul, sel, nl = _mean_se(rho_l)
        out["I2"] = {"registered": f"on the partition-OFF arm, Spearman rho >= "
                                   f"{I2_POINT} (most-useful wells erode "
                                   f"fastest); <= {I2_REFUTE} is the registered "
                                   f"refutation branch; |rho| < 0.3 = no usage "
                                   f"structure",
                     "rho_read_selection_per_seed": rho,
                     "rho_read_selection_mean": mu, "se": se, "n": n,
                     "rho_loo_delta_bpc_per_seed": rho_l,
                     "rho_loo_delta_bpc_mean": mul, "loo_se": sel, "loo_n": nl,
                     "branch": ("CONFIRMED" if np.isfinite(mu) and mu >= I2_POINT
                                else "REFUTED_PRUNES_USELESS"
                                if np.isfinite(mu) and mu <= I2_REFUTE
                                else "NO_USAGE_STRUCTURE"
                                if np.isfinite(mu) and abs(mu) < 0.3
                                else "INTERMEDIATE"),
                     "note": "provisional — the analyst adjudicates rho on the "
                             "raw per-well series"}
    return out


def run3_flag_block(verdict: str, *, i1: bool = True) -> str:
    """⭐ The CSF3 **run-3** flag block: run-2's exact config + only this wave's
    flags, through the existing ``--mem/--store/--set`` path (§A20.4: **zero
    module edits on the cluster**). Emitted with the verdict attached, either
    way — the Advisor decides whether it is submitted.
    """
    mem = ("atom_place_radius=0.3 write_inner_steps=40 "
           "psi_payload_residual=True psi_residual_source=q_star "
           "erosion_partition=True")
    if i1:
        mem += " refresh_monotonic=True"
    return f"""# scripts/csf3/job_gpu_cluformer.sh — RUN 3 candidate
# ⛔ VERDICT AT EMISSION: {verdict}. Submitted only on the Advisor's decision.
sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,\\
MEM="{mem}",\\
STORE="write_margin=0.6",\\
SET="monitor_every=25 plan_workers=8" \\
       -c 12 --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
"""


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-mean +- SE per cell + the prereg scorecard + the mechanical gate."""
    out: Dict[str, Any] = {
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n_seeds)",
        "rig": "the CSF3 run-2 config (atom_place_radius=0.3, write_margin=0.6, "
               "psi_payload_residual=True, psi_residual_source=q_star, all "
               "stage flags TRUE) on local toy enwik8, +- ONE capability",
        "monitor_13": "w4 cells are NON-PROMOTABLE (N94); the w40_* pair is the "
                      "undemoted confirmation",
        "depth_caveat": "depth is NOT quotable as feature importance until I2 "
                        "reports (charter §A21)",
        "cells": {},
    }
    for cell in CELLS:
        rs = [r for r in records if r["cell"] == cell]
        if not rs:
            continue
        row: Dict[str, Any] = {"n_seeds": len(rs),
                               "seeds": [int(r["seed"]) for r in rs],
                               "note": CELLS[cell]["note"],
                               "diagnostic": cell in DIAGNOSTIC_CELLS}
        for k in ("depth_untrained", "depth_at_200", "depth_final",
                  "depth_ratio_1000_over_200", "depth_ratio_final_over_untrained",
                  "bpc_live", "bpc_live_minus_blank",
                  "bpc_memory_deleted_minus_live", "bpc_none_minus_live",
                  "acq", "acq_blank", "chance", "rewrite_violation_rate",
                  "n_rewrite_events", "n_rewrite_violations",
                  "n_rewrite_violations_post_guard", "n_admitted",
                  "n_occupied_target", "n_evicting_target", "wall_s"):
            v = [float(r.get(k, float("nan"))) for r in rs]
            mu, se, n = _mean_se(v)
            row[k], row[k + "_se"], row[k + "_n"] = mu, se, n
            row[k + "_per_seed"] = v
        for k in ("n_events_total", "n_down_own_total", "n_up_foreign_total",
                  "rate_down_own_beyond_decay_mean", "rate_up_foreign_mean",
                  "max_abs_own_residual_vs_decay_law",
                  "median_rel_change_foreign_mean"):
            v = [float(r.get("interference", {}).get(k, float("nan")))
                 for r in rs]
            mu, se, n = _mean_se(v)
            row["interference_" + k], row["interference_" + k + "_se"] = mu, se
            row["interference_" + k + "_per_seed"] = v
        for k in ("rho_read_selection", "rho_loo_delta_bpc", "rho_grad_atoms"):
            v = [float(r.get("i2", {}).get(k, float("nan"))) for r in rs]
            mu, se, n = _mean_se(v)
            row["i2_" + k], row["i2_" + k + "_se"] = mu, se
            row["i2_" + k + "_per_seed"] = v
        out["cells"][cell] = row
    out["prereg"] = prereg_scorecard(records)
    out["gate"] = gate_verdict(records)
    if any(r["cell"] in W40_CELLS for r in records):
        out["gate_w40"] = gate_verdict(records, on="w40_p1_on",
                                       off="w40_p1_off")
    out["run3_flag_block"] = run3_flag_block(out["gate"]["verdict"])
    return out


def plot_curves(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """The wave's headline figure: fitted depth vs outer step, per cell/seed.

    Log-y (the OFF arm's collapse spans dozens of decades), one colour per cell,
    one line per seed, with the step-200 reference the E1/E2 ratio is taken
    against marked. Returns ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                    # pragma: no cover
        return None
    cells = [c for c in CELLS if any(r["cell"] == c for r in records)]
    seeds = sorted({int(r["seed"]) for r in records})
    if not cells or not seeds:
        return None
    # ⭐ FACETED BY SEED. The arms are paired *within* a seed (the untrained
    # depth itself varies 5.4x across seeds at this rig), so a pooled panel
    # compares lines that were never comparable.
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, s in zip(axes, seeds, strict=True):
        for j, c in enumerate(cells):
            for r in [r for r in records
                      if r["cell"] == c and int(r["seed"]) == s]:
                xs = [int(p["at_step"]) for p in r["curve"]]
                ys = [max(float(p["depth_median"]), 1e-70) for p in r["curve"]]
                ax.plot(xs, ys, color=colors[j % 10], alpha=0.9, lw=1.4,
                        ls="--" if c in DIAGNOSTIC_CELLS else "-",
                        label=c if ax is axes[0] else None)
        ax.axvline(200, color="0.6", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("outer step")
        ax.set_title(f"seed {s}", fontsize=10)
    axes[0].set_ylabel("median fitted well depth (live items, own site)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("C2W6 erosion curve — toy (0.16 M), CSF3 run-2 config, ± P1 "
                 "  [w4 cells are monitor-#13 demoted; dashed = DIAGNOSTIC]",
                 fontsize=10)
    fig.tight_layout()
    p = Path(out) / "erosion_curves.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ==========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", nargs="*", default=None,
                    help=f"subset of {sorted(CELLS)}")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--steps", type=int, default=None,
                    help=f"outer steps (default {DEFAULT_STEPS}, "
                         f"{DEFAULT_W40_STEPS} for the w40 cells)")
    ap.add_argument("--monitor-every", type=int, default=MONITOR_EVERY)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--loo-batches", type=int, default=2)
    ap.add_argument("--loo-checkpoints", type=int, default=4)
    ap.add_argument("--no-store-watch", action="store_true",
                    help="skip the trainer's own second depth series (a "
                         "declared cut at w40, where it costs a 40-step write "
                         "replay per monitor window)")
    ap.add_argument("--no-none-arm", action="store_true",
                    help="skip the retrained memory-deleted arm (it is "
                         "flag-independent, so one per seed suffices)")
    ap.add_argument("--out", default=".claude/outputs/c2w6-anti-erosion")
    ap.add_argument("--tag", default="erosion")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="re-adjudicate existing *_records.json in --out")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.aggregate_only:
        recs: List[Dict[str, Any]] = []
        for p in sorted(out.glob("erosion_*_records.json")):
            recs.extend(json.loads(p.read_text()).get("records", []))
        agg = aggregate(recs)
        save_json(out / "erosion_aggregate.json",
                  {"n_records": len(recs), "aggregate": agg})
        print("plot:", plot_curves(recs, out))
        print(json.dumps(agg["gate"], indent=1, default=float))
        print(agg["run3_flag_block"])
        return 0

    cells = args.cells or list(CLAIM_CELLS)
    bad = [c for c in cells if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells {bad}; known: {sorted(CELLS)}")

    recs = []
    for cell in cells:
        steps = args.steps
        if steps is None:
            steps = DEFAULT_W40_STEPS if cell in W40_CELLS else DEFAULT_STEPS
        for s in args.seeds:
            recs.append(run_cell(cell, int(s), steps=int(steps),
                                 eval_batches=args.eval_batches,
                                 loo_batches=args.loo_batches,
                                 monitor_every=args.monitor_every,
                                 n_loo_checkpoints=args.loo_checkpoints,
                                 store_watch=not args.no_store_watch,
                                 with_none=not args.no_none_arm))
            save_json(out / f"erosion_{args.tag}_records.json",
                      {"records": recs, "aggregate": aggregate(recs)})
    agg = aggregate(recs)
    p = save_json(out / f"erosion_{args.tag}_records.json",
                  {"records": recs, "aggregate": agg,
                   "jax_version": jax.__version__,
                   "flags": {"scale": "toy", "cells": cells,
                             "seeds": list(args.seeds), "steps": args.steps,
                             "monitor_every": args.monitor_every}})
    print(json.dumps(agg["gate"], indent=1, default=float))
    print(agg["run3_flag_block"])
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
