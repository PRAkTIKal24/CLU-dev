"""⭐ **`pilot-placement-probe`** — does LOCALIZED PLACEMENT wake the in-block store?

Charter ADDENDUM 4 §A19 task 5, per ruling **A18.4**. This is an
**instrument/diagnostic**: ⛔ nothing it produces is a paper number. It is config
evidence for the CSF3 submitted run and design evidence for tier ii.

**The finding it interrogates** (`cluformer-pilot` §5, registered as N196): the
full C2W1 store, run as a streaming block's memory, is measurably **inert** —
live-store and blank-store held-out NLL are equal to float32 round-off, the
in-block self-probe sits exactly at chance, and **the write budget is refuted as
the cause** (16x the inner steps buys 1.33x the depth and zero acquisition, with
depth saturating at ~0.045 against the shipped store's fitted ``D = 0.46-0.80``).
The surviving hypothesis is **atom placement at init**: 128 atoms per group
scattered at ``init_scale = 1.0`` in a ``dim = 3`` ball cannot be gathered into a
well at the target by a few unrolled steps.

**The two hypotheses, tested in the registered order** (§5.3 of the pilot):

* **H1 — localized atom init** (``atom_local_radius`` at its N98 designed band
  ``~2 * atom_width``, targets = the phi-image of the earliest chunks).
* **H2 — the trajectory write term** (``write_lambda_traj``, the C2W2 machinery
  as tier-ii organizer tooling, §A14.1).

(The psi payload residual is third-priority and is a **declared NOT-RUN** unless
H1 and H2 both fail with budget remaining.)

**The pre-registered success signal** (`PREREG.md` §3), reported per arm, never
binarized — a partial waking is the mechanistically informative outcome:

(a) the in-block acquisition self-probe comes off chance (``> chance + 2 SE``, 3
paired seeds); (b) live != blank at float32 resolution; (c) the fitted well depth
leaves the 0.045 saturation toward the shipped 0.46-0.80 band.

⭐⭐ **The laundering control that runs beside every acquisition number:** the
**blank store with the SAME localized init**. Localizing a group's atoms near a
site makes the read relax toward that site *whether or not anything was written*,
so a self-probe hit can be bought by the initialisation alone. Only the paired
``acq(live) - acq(blank)`` at the same init is evidence of a **write**.

⚠ **Monitor #13 / N94 travels with every reading here: 4 inner write steps
against a floor of 40 ⇒ every number in this module is formally NON-PROMOTABLE.**

⛔ TOY scale (0.16 M) only. "CLU-former" is a placeholder name and appears
nowhere; this is *the tier-iii block*.

Runnable directly::

    PYTHONPATH=. python -m chlu.experiments.exp_placement_probe --tier screen
    PYTHONPATH=. python -m chlu.experiments.exp_placement_probe --tier trained \\
        --cells baseline h1_r0.6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from chlu.data.enwik8 import contiguous_batches, load_enwik8, random_batches
from chlu.experiments.exp_cluformer_pilot import TOY, make_config
from chlu.training.train_cluformer import (
    PilotConfig,
    build_arm,
    calibrate_atom_group_centers,
    calibrate_phi_gain,
    evaluate,
    monitor_pass,
    plan_pass,
    save_json,
    solve_arms,
    train_arm,
)

#: The shipped store's fitted well-depth band (`full-clu-harness` §3.4) — the
#: target of success signal (c). The pilot's saturated value is 0.045.
SHIPPED_DEPTH_BAND: Tuple[float, float] = (0.46, 0.80)
PILOT_SATURATED_DEPTH: float = 0.045

#: ⭐ The registered cells. ``mem`` entries are :class:`StreamMemoryConfig`
#: overrides; ``needs_centers`` marks the arms whose init is localized (and which
#: therefore need the phi-image calibration pass **and** a blank control run at
#: the SAME init).
#:
#: The N98 designed band is ``radius ~ 2 * atom_width``; at this rig
#: ``atom_width = 0.3`` so ``r = 0.6`` is the designed point, ``r = 0.3`` is 1s
#: and ``r = 0.15`` (0.5s) is the declared extra. C2W2's measured ``lambda_traj``
#: band is ``{0.03, 0.3, 3, 30}``; ``{0.3, 3}`` is the registered primary pair.
CELLS: Dict[str, Dict[str, Any]] = {
    "baseline": dict(mem={}, needs_centers=False,
                     note="the pilot's own configuration, reproduced"),
    "h1_r0.15": dict(mem={"atom_local_radius": 0.15}, needs_centers=True,
                     note="H1 at 0.5s (declared extra)"),
    "h1_r0.3": dict(mem={"atom_local_radius": 0.3}, needs_centers=True,
                    note="H1 at 1s"),
    "h1_r0.6": dict(mem={"atom_local_radius": 0.6}, needs_centers=True,
                    note="H1 at the N98 DESIGNED band (2s)"),
    # ⭐ H1b — localized placement AT WRITE (the streaming form of H1; PREREG
    # ADDENDUM 1, filed before these cells ran). The static N98 lever localizes a
    # group around a target fixed BEFORE the stream starts, but a streaming
    # block's item sites are chosen by the controller when the chunk arrives.
    "h1b_r0.15": dict(mem={"atom_place_radius": 0.15}, needs_centers=False,
                      note="H1b at 0.5s"),
    "h1b_r0.3": dict(mem={"atom_place_radius": 0.3}, needs_centers=False,
                     note="H1b at 1s"),
    "h1b_r0.6": dict(mem={"atom_place_radius": 0.6}, needs_centers=False,
                     note="H1b at the N98 designed band (2s)"),
    "h2_lam0.3": dict(mem={"write_lambda_traj": 0.3}, needs_centers=False,
                      note="H2 at lambda_traj = 0.3"),
    "h2_lam3": dict(mem={"write_lambda_traj": 3.0}, needs_centers=False,
                    note="H2 at lambda_traj = 3"),
    "h1h2": dict(mem={"atom_local_radius": 0.6, "write_lambda_traj": 0.3},
                 needs_centers=True, note="interaction: H1(init) x H2"),
    "h1bh2": dict(mem={"atom_place_radius": 0.3, "write_lambda_traj": 0.3},
                  needs_centers=False, note="interaction: H1b(write) x H2"),
    # ⭐ The write-budget INTERACTION (PREREG ADDENDUM 2). The pilot refuted the
    # write budget as the cause of inertness *at the scattered init* (16x buys
    # nothing). The question the CSF3 config actually needs answered is whether
    # the budget becomes live ONCE THE ATOMS ARE IN THE RIGHT PLACE — i.e.
    # whether "steps" and "placement" are complements. 40 = monitor #13 / N94's
    # maturity floor, so these are the only two cells here whose readings are
    # not demoted by #13.
    "baseline_w40": dict(mem={"write_inner_steps": 40}, needs_centers=False,
                         note="control: N94's 40-step floor at the scattered init"),
    "h1b_r0.3_w40": dict(mem={"atom_place_radius": 0.3, "write_inner_steps": 40},
                         needs_centers=False,
                         note="H1b x N94's 40-step floor - the CSF3 candidate"),
}

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)


# --------------------------------------------------------------------------
# one cell, one seed
# --------------------------------------------------------------------------
def _prepare(cell: str, seed: int, *, steps: Optional[int] = None,
             eval_batches: int = 4) -> Tuple[PilotConfig, Any, Any, Any, Dict[str, Any]]:
    """Config + data + the calibrated phi gain / localization targets.

    ⭐ Paired by construction: the shell, the phi gain and the data order depend
    only on ``seed``, so two cells at the same seed differ **only** in the
    memory-slot overrides under test.
    """
    spec = CELLS[cell]
    over: Dict[str, Any] = {"eval_batches": int(eval_batches)}
    if steps is not None:
        over["steps"] = int(steps)
    pcfg = make_config("toy", seed, over)
    tr, va, te = load_enwik8(pcfg.data_root, n_bytes=pcfg.data_bytes)

    key = jax.random.PRNGKey(1000 + seed)      # identical to the pilot's chain
    k_cal, k_solve, k_model = jax.random.split(key, 3)
    calib_x = next(iter(random_batches(tr, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                       n_batches=1, seed=seed)))[0]
    gain = calibrate_phi_gain(pcfg, calib_x, key=k_cal)
    pcfg.memory = dict(pcfg.memory)
    pcfg.memory["phi_gain"] = gain
    prov: Dict[str, Any] = {"phi_gain": gain}
    if spec["needs_centers"]:
        centers = calibrate_atom_group_centers(pcfg, calib_x, key=k_cal)
        pcfg.memory["atom_group_centers"] = centers
        prov["atom_group_centers"] = [list(c) for c in centers]
    pcfg.memory.update(dict(spec["mem"]))
    prov["mem_overrides"] = dict(spec["mem"])
    return pcfg, (tr, va, te), k_solve, k_model, prov


def _read_output_delta(model, pcfg: PilotConfig, tokens) -> Dict[str, float]:
    """‖read(written state) − read(blank state)‖ at the same queries.

    The most direct form of "is the store inert": the held-out NLL gap is the
    same statement after two layers of a language model, and it is the one the
    pilot published at ``0.00e+00``.
    """
    blk = model.blocks[0]
    cell = blk.cell
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    plans, _ = plan_pass(model, tk, pcfg)
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    h = h + model.pos[: h.shape[1]][None]
    z = jax.vmap(blk.chunk_latents)(h)
    st = cell.init_state()
    pl0 = jax.tree_util.tree_map(lambda a: a[0], plans[0])
    for c in range(int(z.shape[1])):
        pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl0)
        st = cell.write(st, z[0, c], pc)
    blank = cell.init_state()
    d, n = [], []
    for c in range(int(z.shape[1])):
        r_live = np.asarray(cell.read(st, z[0, c]))
        r_blank = np.asarray(cell.read(blank, z[0, c]))
        d.append(float(np.linalg.norm(r_live - r_blank)))
        n.append(float(np.linalg.norm(r_blank)))
    # how far the WRITE moved the landscape parameters (state deviation)
    a0 = cell.init_state()
    dev = float(np.max(np.abs(np.asarray(st.amp) - np.asarray(a0.amp))))
    dev_c = float(np.max(np.abs(np.asarray(st.centers) - np.asarray(a0.centers))))
    return {"read_delta_median": float(np.median(d)),
            "read_delta_max": float(np.max(d)),
            "read_norm_median": float(np.median(n)),
            "amp_max_deviation": dev, "centers_max_deviation": dev_c}


def multi_lane_self_probe(model, pcfg: PilotConfig, tokens) -> Dict[str, Any]:
    """⭐ The self-probe pooled over **every lane of the batch**, live and blank.

    ``monitor_pass`` replays lane 0 only, which at this rig leaves 3-6 live items
    ⇒ ``chance = 1/3`` and ``SE ~ 0.27``: too weak to resolve the pre-registered
    ``chance + 2 SE`` bar. Pooling the four lanes quadruples the probed set at
    the same seed and the same plan. The chance level is the mean of the
    per-lane ``1/n_live`` because lanes may admit different numbers of items.

    ⭐⭐ **The blank column is computed on the SAME init** — that is the
    laundering control: a localized init can buy self-probe hits with no write at
    all, so only ``acq_live - acq_blank`` is evidence of a write.
    """
    from chlu.training.train_cluformer import store_self_probe

    blk = model.blocks[0]
    cell = blk.cell
    tk = jnp.asarray(tokens, dtype=jnp.int32)
    plans, _ = plan_pass(model, tk, pcfg)
    h = jax.vmap(lambda t: jax.vmap(model.embed)(t))(tk)
    h = h + model.pos[: h.shape[1]][None]
    z = jax.vmap(blk.chunk_latents)(h)
    tol = float(pcfg.store_cfg().payload_tol)

    hits_l = hits_b = strict_l = strict_b = 0.0
    n_tot = 0
    chances: List[float] = []
    depths: List[float] = []
    for b in range(int(z.shape[0])):
        pl = jax.tree_util.tree_map(lambda a, i=b: a[i], plans[0])
        st = cell.init_state()
        for c in range(int(z.shape[1])):
            pc = jax.tree_util.tree_map(lambda a, i=c: a[i], pl)
            st = cell.write(st, z[b, c], pc)
        sites = np.asarray(pl.sites)[-1]
        live = np.asarray(pl.live)[-1]
        pr = store_self_probe(cell, st, sites, live, payload_tol=tol)
        pb = store_self_probe(cell, cell.init_state(), sites, live,
                              payload_tol=tol)
        n = int(pr.get("n_probed", 0))
        if n == 0:
            continue
        hits_l += float(pr["acq"]) * n
        hits_b += float(pb["acq"]) * n
        strict_l += float(pr["strict"]) * n
        strict_b += float(pb["strict"]) * n
        chances.extend([float(pr["chance"])] * n)
        depths.extend([float(x) for x in pr.get("retention", [])])
        n_tot += n
    if n_tot == 0:
        return {"n_probed": 0}
    ch = float(np.mean(chances))
    return {
        "n_probed": n_tot, "n_lanes": int(z.shape[0]),
        "acq_live": hits_l / n_tot, "acq_blank": hits_b / n_tot,
        "strict_live": strict_l / n_tot, "strict_blank": strict_b / n_tot,
        "chance": ch,
        "se": float(np.sqrt(max(ch, 1e-9) * (1.0 - ch) / n_tot)),
        "acq_minus_blank": (hits_l - hits_b) / n_tot,
        "depth_median": float(np.median(depths)) if depths else float("nan"),
        "depth_per_item": depths,
    }


def run_screen(cell: str, seed: int, *, eval_batches: int = 4) -> Dict[str, Any]:
    """⭐ The SCREEN tier: everything the success signal needs, **untrained**.

    The pilot's mechanism (depth, acquisition, live-vs-blank) is a property of
    the *write*, not of the outer optimisation: the write runs at inference, the
    self-probe is label-free, and the pilot's own write-budget sweep was measured
    this way. Running the screen untrained buys a 7-cell x 3-seed grid for the
    price of one trained cell, and every arm that clears it is re-measured in the
    trained tier.

    ⚠ The held-out bpc of an untrained model is ~8 and is **not** a performance
    number; only the live-vs-blank GAP is read here.
    """
    t0 = time.time()
    pcfg, (tr, va, te), k_solve, k_model, prov = _prepare(
        cell, seed, eval_batches=eval_batches)
    specs, ledger = solve_arms(pcfg, k_solve)
    model = build_arm("clu_store", pcfg, specs, key=k_model)

    x0, y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                          seq_len=pcfg.seq_len, n_batches=1)))
    mp = monitor_pass(model, pcfg, x0)
    ev = list(contiguous_batches(te, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                 n_batches=int(eval_batches)))
    live = evaluate(model, pcfg, iter(ev))
    blank = evaluate(model, pcfg, iter(ev), blank=True)
    rd = _read_output_delta(model, pcfg, x0)

    pooled = multi_lane_self_probe(model, pcfg, x0)
    ret = [x for x in pooled.get("depth_per_item", []) if np.isfinite(x)]
    acq = float(pooled.get("acq_live", float("nan")))
    chance = float(pooled.get("chance", float("nan")))
    se = float(pooled.get("se", float("nan")))
    blank_acq = float(pooled.get("acq_blank", float("nan")))
    rec = {
        "cell": cell, "seed": int(seed), "tier": "screen",
        "provenance": prov,
        "cell_ledger": ledger.get("clu_store"),
        # -- success signal (a) --------------------------------------------
        "acq": acq, "acq_chance": chance, "acq_se": se,
        "acq_strict": float(pooled.get("strict_live", float("nan"))),
        "acq_strict_blank": float(pooled.get("strict_blank", float("nan"))),
        "n_probed": int(pooled.get("n_probed", 0)),
        "pooled_self_probe": {k: v for k, v in pooled.items()
                              if k != "depth_per_item"},
        "lane0_self_probe": {k: v for k, v in mp.get("self_probe", {}).items()
                             if k != "retention"},
        "blank_acq": blank_acq,
        "acq_minus_blank": acq - blank_acq,
        "acq_off_chance_2se": bool(np.isfinite(acq) and acq > chance + 2.0 * se),
        # -- success signal (b) --------------------------------------------
        "bpc_live": live["bpc"], "bpc_blank": blank["bpc"],
        "bpc_live_minus_blank": live["bpc"] - blank["bpc"],
        "read_output": rd,
        # -- success signal (c) --------------------------------------------
        "depth_median": float(np.median(ret)) if ret else float("nan"),
        "depth_max": float(np.max(ret)) if ret else float("nan"),
        "depth_per_item": [float(x) for x in ret],
        "depth_leaves_saturation": bool(
            ret and float(np.median(ret)) > 2.0 * PILOT_SATURATED_DEPTH),
        "depth_in_shipped_band": bool(
            ret and SHIPPED_DEPTH_BAND[0] <= float(np.median(ret))
            <= SHIPPED_DEPTH_BAND[1]),
        # -- context --------------------------------------------------------
        "monitors": {k: v for k, v in mp.items() if k != "readings"},
        "monitor_13_note": ("write_steps = 4 vs floor 40 => every reading here "
                            "is formally NON-PROMOTABLE under N94"),
        "wall_s": time.time() - t0,
    }
    print(f"[screen {cell} s{seed}] acq {acq:.3f} (chance {chance:.3f}, blank "
          f"{blank_acq:.3f}) | depth {rec['depth_median']:.4g} | "
          f"live-blank {rec['bpc_live_minus_blank']:+.3e} bpc | "
          f"read Δ {rd['read_delta_median']:.3e} | {rec['wall_s']:.0f}s", flush=True)
    return rec


def run_trained(cell: str, seed: int, *, steps: Optional[int] = None,
                eval_batches: int = 4, with_none: bool = True) -> Dict[str, Any]:
    """The TRAINED tier: the same three signals after the outer optimisation.

    Adds the **memory-deleted** arm (the pilot's ``none``) so the probe's own
    version of the pilot's decisive margin — *is the memory a net cost?* — is
    measured at the same seeds and the same data order.
    """
    t0 = time.time()
    pcfg, (tr, va, te), k_solve, k_model, prov = _prepare(
        cell, seed, steps=steps, eval_batches=eval_batches)
    specs, ledger = solve_arms(pcfg, k_solve)
    batches = list(random_batches(tr, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                  n_batches=pcfg.steps, seed=pcfg.seed))
    ev = list(contiguous_batches(te, batch=pcfg.batch, seq_len=pcfg.seq_len,
                                 n_batches=int(eval_batches)))
    x0, y0 = next(iter(contiguous_batches(va, batch=pcfg.batch,
                                          seq_len=pcfg.seq_len, n_batches=1)))

    out: Dict[str, Any] = {"cell": cell, "seed": int(seed), "tier": "trained",
                           "provenance": prov, "steps": int(pcfg.steps),
                           "cell_ledger": ledger.get("clu_store"), "arms": {}}
    arms = ["clu_store"] + (["none"] if with_none else [])
    for a in arms:
        m = build_arm(a, pcfg, specs, key=k_model)
        m, hist = train_arm(a, m, pcfg, iter(batches))
        row = {"static": evaluate(m, pcfg, iter(ev)),
               "plan_pass_frac": hist["plan_pass_frac"],
               "wall_s": hist["wall_s"]}
        if a == "clu_store":
            row["blank_store"] = evaluate(m, pcfg, iter(ev), blank=True)
            mp = monitor_pass(m, pcfg, x0,
                              write_loss_now=float(hist["loss_history"][-1]))
            pooled = multi_lane_self_probe(m, pcfg, x0)
            ret = [x for x in pooled.get("depth_per_item", []) if np.isfinite(x)]
            row["acq"] = float(pooled.get("acq_live", float("nan")))
            row["acq_chance"] = float(pooled.get("chance", float("nan")))
            row["acq_se"] = float(pooled.get("se", float("nan")))
            row["blank_acq"] = float(pooled.get("acq_blank", float("nan")))
            row["acq_minus_blank"] = float(pooled.get("acq_minus_blank",
                                                      float("nan")))
            row["n_probed"] = int(pooled.get("n_probed", 0))
            row["depth_median"] = float(np.median(ret)) if ret else float("nan")
            row["depth_per_item"] = [float(x) for x in ret]
            row["monitors"] = {k: v for k, v in mp.items() if k != "readings"}
            row["read_output"] = _read_output_delta(m, pcfg, x0)
        out["arms"][a] = row
        print(f"[trained {cell} s{seed} {a}] bpc {row['static']['bpc']:.4f} "
              f"({row['wall_s']:.0f}s)", flush=True)
    clu = out["arms"]["clu_store"]
    out["bpc_live_minus_blank"] = (clu["static"]["bpc"]
                                   - clu["blank_store"]["bpc"])
    if with_none:
        out["bpc_clu_minus_none"] = (clu["static"]["bpc"]
                                     - out["arms"]["none"]["static"]["bpc"])
    out["wall_s"] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _mean_se(v) -> Tuple[float, float, int]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return float(np.mean(a)), float(sd / np.sqrt(a.size)), int(a.size)


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-mean +- SE per cell, and the three success signals adjudicated.

    ⛔ Nothing is binarized away: a partial waking (depth moves, acquisition does
    not) is reported as the pattern it is.
    """
    out: Dict[str, Any] = {"sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n)",
                           "success_signal": {
                               "a": "acq > chance + 2 SE (paired seeds)",
                               "b": "live != blank at float32 resolution",
                               "c": f"depth leaves {PILOT_SATURATED_DEPTH} toward "
                                    f"{SHIPPED_DEPTH_BAND}"},
                           "monitor_13": "4 write steps vs floor 40 => "
                                         "NON-PROMOTABLE (N94)",
                           "cells": {}}
    tiers = {r.get("tier", "screen") for r in records}
    for tier in sorted(tiers):
        for cell in CELLS:
            rs = [r for r in records if r["cell"] == cell
                  and r.get("tier", "screen") == tier]
            if not rs:
                continue
            if tier == "screen":
                get = lambda r, k: r.get(k, float("nan"))  # noqa: E731
                cols = {
                    "acq": [get(r, "acq") for r in rs],
                    "acq_chance": [get(r, "acq_chance") for r in rs],
                    "blank_acq": [get(r, "blank_acq") for r in rs],
                    "acq_minus_blank": [get(r, "acq_minus_blank") for r in rs],
                    "depth_median": [get(r, "depth_median") for r in rs],
                    "bpc_live_minus_blank": [get(r, "bpc_live_minus_blank")
                                             for r in rs],
                    "read_delta_median": [r["read_output"]["read_delta_median"]
                                          for r in rs],
                    "amp_max_deviation": [r["read_output"]["amp_max_deviation"]
                                          for r in rs],
                }
            else:
                cols = {
                    "bpc_clu": [r["arms"]["clu_store"]["static"]["bpc"] for r in rs],
                    "bpc_blank": [r["arms"]["clu_store"]["blank_store"]["bpc"]
                                  for r in rs],
                    "bpc_live_minus_blank": [r["bpc_live_minus_blank"] for r in rs],
                    "bpc_clu_minus_none": [r.get("bpc_clu_minus_none", float("nan"))
                                           for r in rs],
                    "acq": [r["arms"]["clu_store"]["acq"] for r in rs],
                    "acq_chance": [r["arms"]["clu_store"]["acq_chance"] for r in rs],
                    "blank_acq": [r["arms"]["clu_store"]["blank_acq"] for r in rs],
                    "depth_median": [r["arms"]["clu_store"]["depth_median"]
                                     for r in rs],
                    "plan_pass_frac": [r["arms"]["clu_store"]["plan_pass_frac"]
                                       for r in rs],
                }
            row: Dict[str, Any] = {"n_seeds": len(rs),
                                   "seeds": [r["seed"] for r in rs],
                                   "note": CELLS[cell]["note"]}
            for k, v in cols.items():
                m, se, n = _mean_se(v)
                row[k] = m
                row[k + "_se"] = se
                row[k + "_per_seed"] = [float(x) for x in v]
                row[k + "_n"] = n
            # the three signals, adjudicated on the seed-mean
            acq_m, acq_se = row.get("acq"), row.get("acq_se")
            ch = row.get("acq_chance")
            paired_se = row.get("acq_minus_blank_se", acq_se)
            row["signal_a_acq_off_chance"] = bool(
                np.isfinite(acq_m) and np.isfinite(acq_se)
                and acq_m > ch + 2.0 * max(acq_se, 1e-12))
            row["signal_a_write_effect"] = bool(
                np.isfinite(row.get("acq_minus_blank", np.nan))
                and row["acq_minus_blank"] > 2.0 * max(paired_se or 0.0, 1e-12))
            row["signal_b_live_ne_blank"] = bool(
                abs(row.get("bpc_live_minus_blank", 0.0)) > 1e-6)
            row["signal_c_depth_off_saturation"] = bool(
                np.isfinite(row.get("depth_median", np.nan))
                and row["depth_median"] > 2.0 * PILOT_SATURATED_DEPTH)
            row["signal_c_depth_in_shipped_band"] = bool(
                np.isfinite(row.get("depth_median", np.nan))
                and SHIPPED_DEPTH_BAND[0] <= row["depth_median"]
                <= SHIPPED_DEPTH_BAND[1])
            out["cells"][f"{tier}/{cell}"] = row
    return out


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=("screen", "trained"), default="screen")
    ap.add_argument("--cells", nargs="*", default=None,
                    help=f"subset of {sorted(CELLS)}")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--steps", type=int, default=None,
                    help="outer steps (trained tier); default = TOY's 60")
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--out", default=".claude/outputs/pilot-placement-probe")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    cells = args.cells or list(CELLS)
    bad = [c for c in cells if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells {bad}; known: {sorted(CELLS)}")

    recs: List[Dict[str, Any]] = []
    out = Path(args.out)
    tag = args.tag or args.tier
    for cell in cells:
        for s in args.seeds:
            if args.tier == "screen":
                recs.append(run_screen(cell, int(s),
                                       eval_batches=args.eval_batches))
            else:
                recs.append(run_trained(cell, int(s), steps=args.steps,
                                        eval_batches=args.eval_batches))
            save_json(out / f"probe_{tag}_records.json",
                      {"records": recs, "aggregate": aggregate(recs)})
    agg = aggregate(recs)
    p = save_json(out / f"probe_{tag}_records.json",
                  {"records": recs, "aggregate": agg,
                   "jax_version": jax.__version__,
                   "flags": {"scale": "toy", "cells": cells,
                             "seeds": list(args.seeds), "tier": args.tier,
                             "toy": TOY}})
    print(json.dumps(agg["cells"], indent=1, default=float)[:4000])
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
