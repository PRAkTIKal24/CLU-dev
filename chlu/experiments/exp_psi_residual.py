"""⭐ **`psi-payload-residual`** — does a payload-carrying residual close the
read-out's payload gap? (charter §A20.3(a); GATES CSF3 RUN 2.)

**The convicted component.** `pilot-placement-probe` §6 measured the two-phase
relaxation DELIVERING the payload — ``q*[payload]`` moves 30-50 % of the way to
the true value when the store is live and **exactly 0.000** when it is blank —
and ``psi`` then compressing the between-item spread a further **7-25x**
(``q*`` spread 0.053-0.114 -> decoded 0.0065-0.0117), which leaves the decode
near-constant and the nearest-stored-payload assignment a fixed permutation of a
nearly constant vector, i.e. ``acq = mean(1/n_live) = chance``, identically.

**The build under test** (`chlu.core.blocks.StreamMemoryConfig.
psi_payload_residual`): the read's own PAYLOAD coordinates reach the decode
without passing through psi's pooling, through a learned gate initialised to
pass-through. ⛔ **Payload coordinates only** — the read launches on the
payload-zero manifold, so the residual carries no information about ``phi(x)``
and cannot become N68's query bypass; the blank-store arm measures that, and it
is a blocking check, not a footnote.

**What this module measures — the PER-STAGE SPREAD LEDGER** (the probe's §6
instrument, promoted from one lane / one seed to every lane and 3 paired seeds):

=========================  ===================================================
stage                      quantity
=========================  ===================================================
``true``                   ``sites[i, a:a+m]`` — the written payload
``q_star``                 ``q*[payload]`` — what the dynamics deliver
``traj_mean``              the strided trajectory's mean payload slot
``psi_only``               the shipped decode (residual gate = 0)
``decoded``                ``psi_only + gate . source`` — the fixed decode
=========================  ===================================================

⭐ The residual is **additive and linear in the gate** by construction, so
``decode(g) = decode(0) + g . source`` holds exactly. The whole ledger, the gate
sweep and both leak controls therefore come out of **three reads per item**, not
one grid per gate — and the linearity itself is asserted at runtime
(``_LINEARITY_TOL``) rather than assumed.

⚠ Monitor #13 / N94 travels with every reading: 4 inner write steps against a
floor of 40 ⇒ every number here is formally NON-PROMOTABLE except the ``*_w40``
cells. ⛔ TOY scale (0.16 M). ⛔ No paper number: this is run-2 config evidence
and tier-ii design input.

Runnable directly::

    PYTHONPATH=. python -m chlu.experiments.exp_psi_residual --tier ledger
    PYTHONPATH=. python -m chlu.experiments.exp_psi_residual --tier trained \\
        --cells run1 --seeds 0 1 2 --steps 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.data.enwik8 import contiguous_batches, random_batches
from chlu.experiments.exp_placement_probe import CELLS as PROBE_CELLS
from chlu.experiments.exp_placement_probe import _prepare, multi_lane_self_probe
from chlu.training.train_cluformer import (
    build_arm,
    evaluate,
    plan_pass,
    save_json,
    solve_arms,
    train_arm,
)

#: ⭐ The acceptance bar, §A20.3(a) verbatim: *"decoded spread reaches ``q*``
#: spread (ratio >= 0.5 across cells, vs the current 0.04-0.15)"*.
ACCEPTANCE_RATIO: float = 0.5

#: The probe's measured ratios, recomputed CONSISTENTLY (PREREG ADDENDUM 1)
#: from `decode_dispersion.json` / `qstar_payload.json`, lane 0, seed 0. ⚠ The
#: task file's "current 0.04-0.15" is the reciprocal of §6's 7-25x compression
#: band, not a per-cell ratio; per cell the before-column is 0.058-0.191 (the
#: ratio is ddof-invariant, so this is the same number under either convention).
PROBE_BEFORE_RATIO: Dict[str, float] = {"baseline": 0.1600, "h1b_r0.3": 0.0583,
                                        "h1b_m1.0": 0.1909}

#: The residual is additive and linear in the gate BY CONSTRUCTION. Asserted, at
#: float32 tolerance, on every cell/seed — if it ever fails, the ledger's
#: arithmetic derivations are void and the run must be discarded.
_LINEARITY_TOL: float = 1e-5

#: The registered cells. ``base`` names a `pilot-placement-probe` cell (so the
#: shell, the phi gain and the data order stay bit-identical to §6's), ``mem`` /
#: ``store`` are extra overrides on top of it.
CELLS: Dict[str, Dict[str, Any]] = {
    "baseline": dict(base="baseline", note="the probe's own baseline (§6 row 1)"),
    "h1b_r0.3": dict(base="h1b_r0.3", note="placement 0.3 (§6 row 2)"),
    "h1b_m1.0": dict(base="h1b_m1.0", note="placement 0.3 x margin 1.0 (§6 row 3)"),
    # ⭐ the CSF3 RUN-1 config (probe §10): placement 0.3 x margin 0.6, and the
    # `_w40` variant at N94's maturity floor — the only cells monitor #13 does
    # not demote.
    "run1": dict(base="h1b_m0.6",
                 note="the run-1 config (placement 0.3 x margin 0.6) at w4"),
    "run1_w40": dict(base="h1b_m0.6", mem={"write_inner_steps": 40},
                     note="the run-1 config AS SUBMITTED (w40, N94's floor)"),
}

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: The gate sweep. ``0`` = the shipped read-out; ``1`` = pass-through; the rest
#: separate a SPREAD deficit from a SCALE deficit (see the report's §"the
#: second gap").
GATE_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def _std(v: np.ndarray) -> float:
    """Sample sd (ddof=1) — the spread convention of the whole ledger."""
    a = np.asarray(v, dtype=float).ravel()
    return float(np.std(a, ddof=1)) if a.size > 1 else float("nan")


def _rng(v: np.ndarray) -> float:
    a = np.asarray(v, dtype=float).ravel()
    return float(np.max(a) - np.min(a)) if a.size > 1 else float("nan")


def _acq(decoded: np.ndarray, pays: np.ndarray) -> float:
    """N110's honest metric: assign each read to the NEAREST STORED payload."""
    if decoded.shape[0] == 0:
        return float("nan")
    dist = np.linalg.norm(decoded[:, None, :] - pays[None, :, :], axis=-1)
    return float((np.argmin(dist, axis=1) == np.arange(decoded.shape[0])).mean())


def _mean_se(v) -> Tuple[float, float, int]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return float(np.mean(a)), float(sd / np.sqrt(a.size)), int(a.size)


def _with_gate(cell, g: np.ndarray):
    """The same cell with its residual gate replaced (the dial of the sweep)."""
    return eqx.tree_at(lambda c: c.psi_res_gate, cell,
                       jnp.asarray(g, dtype=jnp.float32))


def _gate_vec(cell, g: float, row: int = 0) -> np.ndarray:
    """A gate of the cell's own shape with source ``row`` set to ``g``."""
    out = np.zeros(tuple(int(v) for v in cell.psi_res_gate.shape),
                   dtype=np.float32)
    out[int(row)] = float(g)
    return out


def _model_at_gate(model, g: float, row: int = 0):
    """Every layer's gate pinned — the dial applied to the whole block stack."""
    gv = jnp.asarray(_gate_vec(model.blocks[0].cell, g, row), dtype=jnp.float32)
    return eqx.tree_at(lambda mm: [b.cell.psi_res_gate for b in mm.blocks],
                       model, replace=[gv for _ in model.blocks])


# --------------------------------------------------------------------------
# preparation: one model per (cell, seed), residual wired with BOTH sources
# --------------------------------------------------------------------------
def _prepare_residual(cell: str, seed: int, *, steps: Optional[int] = None,
                      eval_batches: int = 4, source: str = "both",
                      gain: float = 1.0, trainable: bool = True):
    """The probe's ``_prepare`` + the residual flags.

    ⭐ Paired by construction: the residual flags are applied AFTER the phi-gain
    calibration (which runs on the memory-deleted arm and cannot see them), so a
    residual-on and a residual-off cell at the same seed share the shell, the
    gain, the data order and every non-gate parameter **bit-identically**.
    """
    spec = CELLS[cell]
    pcfg, data, k_solve, k_model, prov = _prepare(
        spec["base"], seed, steps=steps, eval_batches=eval_batches)
    pcfg.memory = dict(pcfg.memory)
    pcfg.memory.update(dict(spec.get("mem") or {}))
    pcfg.memory.update({"psi_payload_residual": True,
                        "psi_residual_source": str(source),
                        "psi_residual_gain": float(gain),
                        "psi_residual_trainable": bool(trainable)})
    if spec.get("store"):
        pcfg.store = dict(pcfg.store or {})
        pcfg.store.update(dict(spec["store"]))
    prov["base_cell"] = spec["base"]
    prov["base_mem_overrides"] = dict(PROBE_CELLS[spec["base"]]["mem"])
    prov["base_store_overrides"] = dict(PROBE_CELLS[spec["base"]].get("store") or {})
    prov["cell_mem_overrides"] = dict(spec.get("mem") or {})
    prov["residual"] = {"source": source, "gain": gain, "trainable": trainable}
    return pcfg, data, k_solve, k_model, prov


def _written_states(model, pcfg, tokens):
    """Replay the plan and write every lane's stream. ``(z, plans, states)``."""
    blk = model.blocks[0]
    cell = blk.cell
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    plans, _ = plan_pass(model, tk, pcfg)
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    h = h + model.pos[: h.shape[1]][None]
    z = jax.vmap(blk.chunk_latents)(h)
    lanes = []
    for b in range(int(z.shape[0])):
        pl = jax.tree_util.tree_map(lambda a, i=b: a[i], plans[0])
        st = cell.init_state()
        for c in range(int(z.shape[1])):
            pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl)
            st = cell.write(st, z[b, c], pc)
        lanes.append((pl, st))
    return z, lanes


# --------------------------------------------------------------------------
# ⭐⭐ THE PER-STAGE SPREAD LEDGER
# --------------------------------------------------------------------------
def stage_ledger(model, pcfg, tokens) -> Dict[str, Any]:
    """The four stages, per lane, live **and** blank, from 3 reads per item.

    The residual is linear in the gate, so with ``source="both"`` the gate rows
    ``[[0],[0]] / [[1],[0]] / [[0],[1]]`` recover ``psi_only``, ``q*[payload]``
    and ``traj_mean[payload]`` exactly. ⭐ ``q*`` is cross-checked against
    ``cell.read_diag``'s own settled point, which is an independently-coded path
    — a disagreement there means the residual is not reading what §6 measured.
    """
    cell = model.blocks[0].cell
    a, m = int(cell.cfg.addr_dim), int(cell.cfg.payload_dim)
    if int(cell.psi_res_gate.shape[0]) != 2:
        raise ValueError("stage_ledger needs psi_residual_source='both' (two "
                         "gate rows) to separate q_star from traj_mean")
    c0 = _with_gate(cell, _gate_vec(cell, 0.0))
    cq = _with_gate(cell, _gate_vec(cell, 1.0, row=0))
    ct = _with_gate(cell, _gate_vec(cell, 1.0, row=1))

    z, lanes = _written_states(model, pcfg, tokens)
    blank = cell.init_state()
    lane_rows: List[Dict[str, Any]] = []
    lin_err = 0.0
    qstar_err = 0.0
    for lane, (pl, st) in enumerate(lanes):
        sites = np.asarray(pl.sites)[-1]
        live = np.asarray(pl.live)[-1]
        idx = [i for i in range(sites.shape[0]) if live[i] > 0.5]
        if len(idx) < 2:
            continue
        rows: Dict[str, List[np.ndarray]] = {k: [] for k in
                                             ("true", "q_star", "traj_mean",
                                              "psi_only", "q_star_blank",
                                              "psi_only_blank")}
        for i in idx:
            q = jnp.asarray(sites[i], dtype=jnp.float32)
            p0 = np.asarray(c0.read(st, q))[a: a + m]
            pq = np.asarray(cq.read(st, q))[a: a + m]
            pt = np.asarray(ct.read(st, q))[a: a + m]
            b0 = np.asarray(c0.read(blank, q))[a: a + m]
            bq = np.asarray(cq.read(blank, q))[a: a + m]
            rows["true"].append(sites[i, a: a + m])
            rows["psi_only"].append(p0)
            rows["q_star"].append(pq - p0)
            rows["traj_mean"].append(pt - p0)
            rows["psi_only_blank"].append(b0)
            rows["q_star_blank"].append(bq - b0)
            # the independent cross-check of the residual's source
            qs = np.asarray(cell.read_diag(st, q)["q_star"])[a: a + m]
            qstar_err = max(qstar_err, float(np.max(np.abs(qs - (pq - p0)))))
            # linearity: read(gate=2*e_q) must equal psi_only + 2*q*
            p2 = np.asarray(_with_gate(cell, _gate_vec(cell, 2.0)
                                       ).read(st, q))[a: a + m]
            lin_err = max(lin_err, float(np.max(np.abs(
                p2 - (p0 + 2.0 * (pq - p0))))))
        arr = {k: np.stack(v) for k, v in rows.items()}
        dec = arr["psi_only"] + arr["q_star"]          # gate = 1, source q_star
        dec_b = arr["psi_only_blank"] + arr["q_star_blank"]
        row = {
            "lane": lane, "n_live": len(idx),
            "chance": 1.0 / len(idx),
            "spread": {k: _std(arr[k]) for k in arr},
            "range": {k: _rng(arr[k]) for k in arr},
            "values": {k: arr[k].tolist() for k in arr},
            "spread_decoded": _std(dec), "range_decoded": _rng(dec),
            "spread_decoded_blank": _std(dec_b),
            # ⭐ THE ACCEPTANCE RATIO, per lane
            "ratio_psi_only_over_qstar": _std(arr["psi_only"]) / max(
                _std(arr["q_star"]), 1e-12),
            "ratio_decoded_over_qstar": _std(dec) / max(_std(arr["q_star"]), 1e-12),
            "ratio_qstar_over_true": _std(arr["q_star"]) / max(
                _std(arr["true"]), 1e-12),
            # the SCALE (not spread) gap: how far the delivered payload gets
            "frac_of_true_median": float(np.median(
                np.abs(arr["q_star"]) / np.maximum(np.abs(arr["true"]), 1e-12))),
            # acquisition, arithmetically, at every gate
            "acq_by_gate": {f"{g:g}": _acq(arr["psi_only"] + g * arr["q_star"],
                                           arr["true"]) for g in GATE_GRID},
            "acq_blank_by_gate": {
                f"{g:g}": _acq(arr["psi_only_blank"] + g * arr["q_star_blank"],
                               arr["true"]) for g in GATE_GRID},
        }
        lane_rows.append(row)

    n_tot = sum(r["n_live"] for r in lane_rows)
    pooled = {
        "n_lanes": len(lane_rows), "n_items": n_tot,
        "chance": float(np.average([r["chance"] for r in lane_rows],
                                   weights=[r["n_live"] for r in lane_rows]))
        if lane_rows else float("nan"),
        "linearity_maxabs": lin_err,
        "qstar_source_maxabs_vs_read_diag": qstar_err,
    }
    # ⛔ blocking: the ledger's derivations ARE the linearity. Never a warning.
    if lin_err > _LINEARITY_TOL or qstar_err > _LINEARITY_TOL:
        raise AssertionError(
            f"the payload residual is not linear-and-additive as designed "
            f"(linearity {lin_err:.3e}, q*-source vs read_diag {qstar_err:.3e}, "
            f"tol {_LINEARITY_TOL:g}) — every arithmetically-derived spread and "
            "swept acquisition in this record would be void")
    for k in ("ratio_decoded_over_qstar", "ratio_psi_only_over_qstar",
              "ratio_qstar_over_true", "frac_of_true_median",
              "spread_decoded", "spread_decoded_blank"):
        mu, se, n = _mean_se([r[k] for r in lane_rows])
        pooled[k], pooled[k + "_se"], pooled[k + "_n"] = mu, se, n
    for k in ("true", "q_star", "traj_mean", "psi_only", "q_star_blank",
              "psi_only_blank"):
        mu, se, n = _mean_se([r["spread"][k] for r in lane_rows])
        pooled["spread_" + k], pooled["spread_" + k + "_se"] = mu, se
    for g in GATE_GRID:
        key = f"{g:g}"
        w = [r["n_live"] for r in lane_rows]
        pooled.setdefault("acq_by_gate", {})[key] = float(np.average(
            [r["acq_by_gate"][key] for r in lane_rows], weights=w)) if lane_rows \
            else float("nan")
        pooled.setdefault("acq_blank_by_gate", {})[key] = float(np.average(
            [r["acq_blank_by_gate"][key] for r in lane_rows], weights=w)) \
            if lane_rows else float("nan")
    return {"pooled": pooled, "lanes": lane_rows}


def _probe_at_gate(model, pcfg, tokens, g: float) -> Dict[str, Any]:
    """``multi_lane_self_probe`` with the gate pinned — the FORWARD check.

    The ledger derives acquisition arithmetically; this runs the shipped probe
    through the real read at the same gate, so the derivation is verified rather
    than trusted.
    """
    out = multi_lane_self_probe(_model_at_gate(model, g), pcfg, tokens)
    return {k: v for k, v in out.items() if k != "depth_per_item"} | {
        "depth_median": out.get("depth_median", float("nan")),
        "depth_per_item": out.get("depth_per_item", []),
    }


# --------------------------------------------------------------------------
# the tiers
# --------------------------------------------------------------------------
def run_ledger(cell: str, seed: int, *, eval_batches: int = 4,
               forward_gates: Tuple[float, ...] = (0.0, 1.0)) -> Dict[str, Any]:
    """The UNTRAINED tier: the spread ledger + the blank-leak check."""
    t0 = time.time()
    pcfg, (tr, va, te), k_solve, k_model, prov = _prepare_residual(
        cell, seed, eval_batches=eval_batches)
    specs, ledger = solve_arms(pcfg, k_solve)
    model = build_arm("clu_store", pcfg, specs, key=k_model)
    x0, _y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                           seq_len=pcfg.seq_len, n_batches=1)))
    led = stage_ledger(model, pcfg, x0)

    ev = list(contiguous_batches(te, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                 n_batches=int(eval_batches)))
    fwd: Dict[str, Any] = {}
    for g in forward_gates:
        mdl = _model_at_gate(model, g)
        live = evaluate(mdl, pcfg, iter(ev))
        blank = evaluate(mdl, pcfg, iter(ev), blank=True)
        pr = _probe_at_gate(model, pcfg, x0, g)
        fwd[f"{g:g}"] = {
            "bpc_live": live["bpc"], "bpc_blank": blank["bpc"],
            "bpc_live_minus_blank": live["bpc"] - blank["bpc"],
            "acq": pr.get("acq_live", float("nan")),
            "acq_blank": pr.get("acq_blank", float("nan")),
            "acq_minus_blank": pr.get("acq_minus_blank", float("nan")),
            "chance": pr.get("chance", float("nan")),
            "se": pr.get("se", float("nan")),
            "n_probed": pr.get("n_probed", 0),
            "depth_median": pr.get("depth_median", float("nan")),
        }
    p = led["pooled"]
    rec = {
        "cell": cell, "seed": int(seed), "tier": "ledger",
        "provenance": prov, "cell_ledger": ledger.get("clu_store"),
        "ledger": led, "forward": fwd,
        "ratio_decoded_over_qstar": p["ratio_decoded_over_qstar"],
        "acceptance_met": bool(p["ratio_decoded_over_qstar"] >= ACCEPTANCE_RATIO),
        "wall_s": time.time() - t0,
    }
    print(f"[ledger {cell} s{seed}] ratio decoded/q* "
          f"{p['ratio_decoded_over_qstar']:.3f} (psi-only "
          f"{p['ratio_psi_only_over_qstar']:.3f}) | q*/true "
          f"{p['ratio_qstar_over_true']:.3f} | frac-of-true "
          f"{p['frac_of_true_median']:.3f} | acq g=1 "
          f"{fwd.get('1', {}).get('acq', float('nan')):.3f} "
          f"(blank {fwd.get('1', {}).get('acq_blank', float('nan')):.3f}, chance "
          f"{fwd.get('1', {}).get('chance', float('nan')):.3f}) | lin "
          f"{p['linearity_maxabs']:.1e} | {rec['wall_s']:.0f}s", flush=True)
    return rec


def run_trained(cell: str, seed: int, *, steps: Optional[int] = None,
                eval_batches: int = 4,
                arms: Tuple[str, ...] = ("residual_off", "residual_on")
                ) -> Dict[str, Any]:
    """The TRAINED tier: does the residual survive the outer loop?

    ⛔ ``residual_off`` is **not** a different model — it is the same model with
    the gate pinned to 0, which is bit-identical to the shipped read-out
    (asserted in ``tests/test_psi_residual.py``). So the two arms differ by ONE
    leaf and nothing else.
    """
    t0 = time.time()
    out: Dict[str, Any] = {"cell": cell, "seed": int(seed), "tier": "trained",
                           "arms": {}}
    for arm in arms:
        on = arm == "residual_on"
        pcfg, (tr, va, te), k_solve, k_model, prov = _prepare_residual(
            cell, seed, steps=steps, eval_batches=eval_batches,
            source="q_star", gain=(1.0 if on else 0.0), trainable=on)
        specs, ledger = solve_arms(pcfg, k_solve)
        batches = list(random_batches(tr, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                      n_batches=pcfg.steps, seed=pcfg.seed))
        ev = list(contiguous_batches(te, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                     n_batches=int(eval_batches)))
        x0, _y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                               seq_len=pcfg.seq_len, n_batches=1)))
        model = build_arm("clu_store", pcfg, specs, key=k_model)
        g_init = [float(x) for x in np.asarray(
            model.blocks[0].cell.psi_res_gate).ravel()]
        model, hist = train_arm("clu_store", model, pcfg, iter(batches))
        live = evaluate(model, pcfg, iter(ev))
        blank = evaluate(model, pcfg, iter(ev), blank=True)
        pooled = multi_lane_self_probe(model, pcfg, x0)
        ret = [x for x in pooled.get("depth_per_item", []) if np.isfinite(x)]
        # the ledger AFTER training, on the trained gate's own model
        led = _trained_stage_ledger(model, pcfg, x0)
        out["arms"][arm] = {
            "provenance": prov, "steps": int(pcfg.steps),
            "cell_ledger": ledger.get("clu_store"),
            "gate_init": g_init,
            "gate_trained": [float(x) for x in np.asarray(
                model.blocks[0].cell.psi_res_gate).ravel()],
            "gate_by_layer": [[float(v) for v in np.asarray(b.cell.psi_res_gate
                                                            ).ravel()]
                              for b in model.blocks],
            "bpc_live": live["bpc"], "bpc_blank": blank["bpc"],
            "bpc_live_minus_blank": live["bpc"] - blank["bpc"],
            "acq": float(pooled.get("acq_live", float("nan"))),
            "acq_blank": float(pooled.get("acq_blank", float("nan"))),
            "acq_minus_blank": float(pooled.get("acq_minus_blank", float("nan"))),
            "chance": float(pooled.get("chance", float("nan"))),
            "depth_median": float(np.median(ret)) if ret else float("nan"),
            "depth_per_item": [float(x) for x in ret],
            "ledger": led,
            "plan_pass_frac": hist["plan_pass_frac"], "wall_s": hist["wall_s"],
        }
        r = out["arms"][arm]
        print(f"[trained {cell} s{seed} {arm}] bpc {r['bpc_live']:.4f} | gate "
              f"{r['gate_trained']} | ratio {r['ledger']['pooled']['ratio_decoded_over_qstar']:.3f} "
              f"| acq {r['acq']:.3f}/{r['chance']:.3f} | depth "
              f"{r['depth_median']:.4g} | {r['wall_s']:.0f}s", flush=True)
    out["wall_s"] = time.time() - t0
    return out


def _trained_stage_ledger(model, pcfg, tokens) -> Dict[str, Any]:
    """``stage_ledger`` on a model whose gate has ONE row (``source="q_star"``).

    The trained tier runs the shipped single-source configuration, so the
    two-row extraction of :func:`stage_ledger` does not apply; this rebuilds the
    same stages with a 1-row gate.
    """
    cell = model.blocks[0].cell
    a, m = int(cell.cfg.addr_dim), int(cell.cfg.payload_dim)
    n_src = int(cell.psi_res_gate.shape[0])
    c0 = _with_gate(cell, np.zeros((n_src, m), dtype=np.float32))
    e0 = np.zeros((n_src, m), dtype=np.float32)
    e0[0] = 1.0
    cq = _with_gate(cell, e0)
    _z, lanes = _written_states(model, pcfg, tokens)
    blank = cell.init_state()
    lane_rows: List[Dict[str, Any]] = []
    for lane, (pl, st) in enumerate(lanes):
        sites = np.asarray(pl.sites)[-1]
        live = np.asarray(pl.live)[-1]
        idx = [i for i in range(sites.shape[0]) if live[i] > 0.5]
        if len(idx) < 2:
            continue
        true, qs, po, pob, qsb = [], [], [], [], []
        for i in idx:
            q = jnp.asarray(sites[i], dtype=jnp.float32)
            p0 = np.asarray(c0.read(st, q))[a: a + m]
            pq = np.asarray(cq.read(st, q))[a: a + m]
            b0 = np.asarray(c0.read(blank, q))[a: a + m]
            bq = np.asarray(cq.read(blank, q))[a: a + m]
            true.append(sites[i, a: a + m])
            po.append(p0)
            qs.append(pq - p0)
            pob.append(b0)
            qsb.append(bq - b0)
        true, qs, po = np.stack(true), np.stack(qs), np.stack(po)
        pob, qsb = np.stack(pob), np.stack(qsb)
        g = np.asarray(cell.psi_res_gate)[0]
        dec, dec_b = po + g * qs, pob + g * qsb
        lane_rows.append({
            "lane": lane, "n_live": len(idx), "chance": 1.0 / len(idx),
            "spread": {"true": _std(true), "q_star": _std(qs),
                       "psi_only": _std(po), "q_star_blank": _std(qsb),
                       "psi_only_blank": _std(pob)},
            "spread_decoded": _std(dec), "spread_decoded_blank": _std(dec_b),
            "ratio_decoded_over_qstar": _std(dec) / max(_std(qs), 1e-12),
            "ratio_psi_only_over_qstar": _std(po) / max(_std(qs), 1e-12),
            "ratio_qstar_over_true": _std(qs) / max(_std(true), 1e-12),
            "frac_of_true_median": float(np.median(
                np.abs(qs) / np.maximum(np.abs(true), 1e-12))),
            "acq_decoded": _acq(dec, true), "acq_psi_only": _acq(po, true),
            "acq_decoded_blank": _acq(dec_b, true),
            "acq_by_gate": {f"{gg:g}": _acq(po + gg * qs, true)
                            for gg in GATE_GRID},
        })
    pooled: Dict[str, Any] = {"n_lanes": len(lane_rows),
                              "n_items": sum(r["n_live"] for r in lane_rows)}
    for k in ("ratio_decoded_over_qstar", "ratio_psi_only_over_qstar",
              "ratio_qstar_over_true", "frac_of_true_median", "spread_decoded",
              "spread_decoded_blank", "acq_decoded", "acq_psi_only",
              "acq_decoded_blank"):
        mu, se, n = _mean_se([r[k] for r in lane_rows])
        pooled[k], pooled[k + "_se"] = mu, se
    for k in ("true", "q_star", "psi_only", "q_star_blank", "psi_only_blank"):
        mu, se, _n = _mean_se([r["spread"][k] for r in lane_rows])
        pooled["spread_" + k], pooled["spread_" + k + "_se"] = mu, se
    return {"pooled": pooled, "lanes": lane_rows}


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-mean +- SE per cell, with the §A20.3(a) bar adjudicated MECHANICALLY."""
    out: Dict[str, Any] = {
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); spread = sd over "
                         "the lane's live items",
        "acceptance": f"ratio decoded_spread / q*_spread >= {ACCEPTANCE_RATIO} "
                      f"(charter §A20.3(a)); probe's before-column "
                      f"{PROBE_BEFORE_RATIO}",
        "leak_check": "three clauses, all RESIDUAL-SPECIFIC: (1) the blank-store "
                      "decode stays at chance; (2) the residual's own blank "
                      "contribution sd(q*_blank)/sd(q*) < 0.05; (3) the residual "
                      "adds nothing to the blank decode's spread "
                      "(sd(decoded_blank) <= 1.05 * sd(psi_only_blank)). "
                      "⚠ psi's OWN query-driven spread on a blank store is "
                      "shipped behaviour, present at gate 0, and is reported "
                      "beside these clauses rather than charged to the residual.",
        "monitor_13": "4 write steps vs floor 40 => NON-PROMOTABLE (N94), except "
                      "the *_w40 cells",
        "cells": {},
    }
    for tier in sorted({r.get("tier", "ledger") for r in records}):
        for cell in CELLS:
            rs = [r for r in records if r["cell"] == cell
                  and r.get("tier", "ledger") == tier]
            if not rs:
                continue
            row: Dict[str, Any] = {"n_seeds": len(rs),
                                   "seeds": [r["seed"] for r in rs],
                                   "note": CELLS[cell]["note"]}
            if tier == "ledger":
                cols = {k: [r["ledger"]["pooled"][k] for r in rs] for k in
                        ("ratio_decoded_over_qstar", "ratio_psi_only_over_qstar",
                         "ratio_qstar_over_true", "frac_of_true_median",
                         "spread_true", "spread_q_star", "spread_psi_only",
                         "spread_traj_mean", "spread_decoded",
                         "spread_decoded_blank", "spread_q_star_blank",
                         "spread_psi_only_blank", "linearity_maxabs",
                         "qstar_source_maxabs_vs_read_diag")}
                for g in ("0", "1"):
                    for k in ("acq", "acq_blank", "acq_minus_blank", "chance",
                              "bpc_live_minus_blank", "depth_median"):
                        cols[f"g{g}_{k}"] = [r["forward"][g][k] for r in rs
                                             if g in r["forward"]]
                cols["acq_arith_g1"] = [r["ledger"]["pooled"]["acq_by_gate"]["1"]
                                        for r in rs]
                cols["acq_arith_blank_g1"] = [
                    r["ledger"]["pooled"]["acq_blank_by_gate"]["1"] for r in rs]
                for g in GATE_GRID:
                    cols[f"acq_arith_g{g:g}"] = [
                        r["ledger"]["pooled"]["acq_by_gate"][f"{g:g}"] for r in rs]
            else:
                cols = {}
                for arm in ("residual_off", "residual_on"):
                    if arm not in rs[0]["arms"]:
                        continue
                    for k in ("bpc_live", "bpc_live_minus_blank", "acq",
                              "acq_blank", "chance", "depth_median"):
                        cols[f"{arm}_{k}"] = [r["arms"][arm][k] for r in rs]
                    for k in ("ratio_decoded_over_qstar",
                              "ratio_psi_only_over_qstar", "frac_of_true_median",
                              "acq_decoded", "acq_psi_only", "acq_decoded_blank"):
                        cols[f"{arm}_{k}"] = [r["arms"][arm]["ledger"]["pooled"][k]
                                              for r in rs]
                    cols[f"{arm}_gate"] = [r["arms"][arm]["gate_trained"][0]
                                           for r in rs]
            for k, v in cols.items():
                mu, se, n = _mean_se(v)
                row[k], row[k + "_se"], row[k + "_n"] = mu, se, n
                row[k + "_per_seed"] = [float(x) for x in v]
            key = ("ratio_decoded_over_qstar" if tier == "ledger"
                   else "residual_on_ratio_decoded_over_qstar")
            row["ACCEPTANCE_MET"] = bool(np.isfinite(row.get(key, np.nan))
                                         and row[key] >= ACCEPTANCE_RATIO)
            if tier == "ledger" and "g1_acq_blank" in row:
                # ⛔ the blocking leak check, RESIDUAL-SPECIFIC (see "leak_check")
                row["leak_blank_acq_at_chance"] = bool(
                    np.isfinite(row["g1_acq_blank"])
                    and row["g1_acq_blank"] <= row["g1_chance"] + 1e-9)
                row["leak_residual_blank_share"] = float(
                    row["spread_q_star_blank"] / max(row["spread_q_star"], 1e-12))
                row["leak_residual_adds_no_blank_spread"] = bool(
                    row["spread_decoded_blank"]
                    <= 1.05 * row["spread_psi_only_blank"] + 1e-6)
                row["LEAK_CHECK_GREEN"] = bool(
                    row["leak_blank_acq_at_chance"]
                    and row["leak_residual_blank_share"] < 0.05
                    and row["leak_residual_adds_no_blank_spread"])
            out["cells"][f"{tier}/{cell}"] = row
    return out


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=("ledger", "trained"), default="ledger")
    ap.add_argument("--cells", nargs="*", default=None, help=f"subset of {sorted(CELLS)}")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--out", default=".claude/outputs/psi-payload-residual")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    cells = args.cells or ["baseline", "h1b_r0.3", "h1b_m1.0", "run1"]
    bad = [c for c in cells if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells {bad}; known: {sorted(CELLS)}")

    recs: List[Dict[str, Any]] = []
    out = Path(args.out)
    tag = args.tag or args.tier
    for cell in cells:
        for s in args.seeds:
            if args.tier == "ledger":
                recs.append(run_ledger(cell, int(s),
                                       eval_batches=args.eval_batches))
            else:
                recs.append(run_trained(cell, int(s), steps=args.steps,
                                        eval_batches=args.eval_batches))
            save_json(out / f"psires_{tag}_records.json",
                      {"records": recs, "aggregate": aggregate(recs)})
    agg = aggregate(recs)
    p = save_json(out / f"psires_{tag}_records.json",
                  {"records": recs, "aggregate": agg,
                   "jax_version": jax.__version__,
                   "flags": {"scale": "toy", "cells": cells,
                             "seeds": list(args.seeds), "tier": args.tier,
                             "gate_grid": list(GATE_GRID)}})
    print(json.dumps(agg["cells"], indent=1, default=float)[:4000])
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
