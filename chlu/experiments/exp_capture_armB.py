"""⭐ Experiment CAPTURE **ARM B** (C2W8 pass 2) — *the emission head*.

**What this arm is.** A standard MLP-class head on ``phi`` **emits the well
parameters** (center, width, depth, payload) and the item's atom block is *set*
to that designed well — **a forward pass instead of 300 gradient steps** — with
the well's functional form left exactly as it is now.

**How it is scored.** Pass 1's census, re-run **unchanged**:
:func:`chlu.experiments.exp_well_lifecycle.run_census_cell` drives the same
stream onto the same store through the same frozen instrument
(:mod:`chlu.core.well_lifecycle`, read-only this wave); the arm supplies only
(a) its ``CluSystemConfig`` flag and (b) a ``post_build`` hook that pays its
amortised training cost before the first write. The three legs are

======== ======================================================================
G-CAP    ``capture_radius > 0`` on a **majority** of live wells
G-DEC    self-probe ``decode`` above chance (0.0625) by more than 2 SE
G-DRIFT  median ``site_drift`` below the measured key spacing (per seed)
======== ======================================================================

⛔ **K8 travels with every number.** One well per item = private wells = explicit
per-item store parameters = laundered by construction, so every cell, the
aggregate and the JSON artifact carry ``wells_per_item``, ``vocabulary_shared``
and ``NO_TIER_II_CLAIM``.

⛔ **own/foreign is DIAGNOSTIC** — reported under **both** aggregations (median
canonical, mean beside it, ``ERRATA-C2W8-PASS2.md`` §1 Q2), never a target and
never a gate leg.

⛔ **Declared NOT-RUNs** (never reported as nulls): merge, prune, depth
restoration, any §2.7 claim cell, the factored store, the I2 correlation test,
any tier-ii / full-CLU / I2 verdict, CSF3.

Runnable::

    uv run python -m chlu.experiments.exp_capture_armB --quick
    chlu exp-capture-armb [--project N] [--seeds 0,1,2] [--quick]
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
from chlu.core.emission_head import (
    NO_TIER_II_CLAIM,
    emission_ledger,
    pretrain_emission_head,
)

#: the pass-1 rig and the FROZEN census, imported read-only
from chlu.experiments.exp_well_lifecycle import (  # noqa: E402
    _jsonable,
    apply_quick as apply_quick_census,
    run_census_cell,
)

#: pass-1 baselines the ledger is quoted against (`c2w8-well-lifecycle.md` §3.5)
PASS1_CLU_TOTAL_BYTES = 360_960
PASS1_KNN_LAUNDER_BYTES = 288


# ---------------------------------------------------------------------------
# the arm's two seams into the frozen rig
# ---------------------------------------------------------------------------
def clu_overrides(cfg: CHLUConfig, *, min_store: bool = False) -> Dict[str, Any]:
    """The arm's ``CluSystemConfig`` flag (and, for the secondary cell, the atom
    budget the emission head actually needs)."""
    b = cfg.experiment_capture_armb
    over: Dict[str, Any] = {
        "emission_head": True,
        "emission_head_hidden": int(b.head_hidden),
        "emission_head_layers": int(b.head_layers),
        "emission_width_min": float(b.width_min),
        "emission_width_max": float(b.width_max),
        "emission_depth_min": float(b.depth_min),
        "emission_depth_max": float(b.depth_max),
        "emission_payload_delta_max": float(b.payload_delta_max),
        "emission_center_skip_gain": float(b.center_skip_gain),
    }
    if min_store:
        # ⭐ the structural half of the byte finding: the emission head writes ONE
        # designed well per item, so the `min_atoms_base * sqrt2^d` co-scaling
        # (8 192 atoms at d=8, 131 072 at d=16) buys the arm nothing.
        # ⛔ Secondary cell only. The gate cell keeps pass 1's atom budget so the
        # race against arm A is a race.
        n = int(cfg.experiment_capture_armb.min_store_atoms_per_item)
        over.update({"atoms_per_item": n, "min_atoms": n,
                     "min_atoms_base": n, "min_atoms_c": 1.0})
    return over


def make_post_build(cfg: CHLUConfig, record: Dict[str, Any]):
    """The ``post_build`` hook: pretrain the head, once, before the first write.

    ⚠ **Provenance of the training data, declared.** The pool is the ``phi`` fit
    pool (``task1_only`` on this rig) — the same rows ``phi`` itself was fitted
    on, so **no stream item and no test item is seen**; payloads are drawn
    *synthetically* over the payload range, so the head is never shown a
    ``(phi, label)`` pair and no label can leak into placement.
    """
    b = cfg.experiment_capture_armb
    w = cfg.experiment_well_lifecycle

    def hook(system, *, stream, embed, cfg=None, seed=0):
        if system.emitter is None:
            raise RuntimeError("arm B post_build ran with no emission head")
        record["_system"] = system  # popped before serialization (see run_arm_b_cell)
        pool_raw = np.asarray(stream[f"fit_pool_{w.phi_regime}"], dtype=np.float32)
        pool_raw = pool_raw[: int(b.pretrain_pool)]
        phi_pool = np.asarray(embed.keys(pool_raw), dtype=np.float32)
        rng = np.random.default_rng(int(seed) + 4242)
        # synthetic payloads over the SAME range the stream's label map spans
        hi = 4.5 / float(w.payload_scale)
        pays = rng.uniform(-hi, hi, size=(phi_pool.shape[0], int(w.payload_dim)))
        t0 = time.time()
        head, hist = pretrain_emission_head(
            system.emitter, phi_pool, pays,
            jax.random.PRNGKey(int(seed) + 5150),
            dim=int(system.store.dim), confine=float(system.cfg.confine),
            addr_dim=int(system.store.addr_dim),
            steps=int(b.pretrain_steps), batch=int(b.pretrain_batch),
            lr=float(b.pretrain_lr), weight_decay=float(b.pretrain_weight_decay),
            reach_weight=float(b.reach_weight), reach_rho=float(b.reach_rho),
            attr_weight=float(b.attr_weight), attr_margin=float(b.attr_margin),
            loss_kwargs=dict(
                n_perturb=int(system.cfg.write_n_perturb),
                sigma_addr=float(system.cfg.write_sigma_addr),
                sigma_pay=float(system.cfg.write_sigma_pay),
                margin=float(system.cfg.write_margin),
                barrier=float(system.cfg.write_barrier),
                # the w24 crowding lever at the rig's OWN admission radius
                crowd_weight=float(b.crowd_weight),
                crowd_d_safe=float(getattr(system.controller.allocator,
                                            "d_safe", 0.0)),
            ),
        )
        system.emitter = head
        record["pretrain"] = {
            "pool_regime": w.phi_regime, "pool_rows": int(phi_pool.shape[0]),
            "payloads": "synthetic uniform over [-%.4f, %.4f] (NO label seen)" % (hi, hi),
            "loss_first": hist["loss_first"], "loss_last": hist["loss_last"],
            "steps": hist["steps"], "batch": hist["batch"],
            "objective_evals": hist["objective_evals"],
            "wall_s": float(time.time() - t0),
            "head_param_count": int(head.n_params()),
            "reach_weight": float(b.reach_weight), "reach_rho": float(b.reach_rho),
            "attr_weight": float(b.attr_weight), "attr_margin": float(b.attr_margin),
            "crowd_weight": float(b.crowd_weight),
            "center_skip_gain": float(b.center_skip_gain),
            "declared": ("the DESIGNED write->phi organization gradient "
                         "(charter §A28.1), routed through the shipped write "
                         "objective — a designed mechanism, never an inherited leak"),
        }

    return hook


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def gate_legs(cell: Dict[str, Any]) -> Dict[str, Any]:
    """G-CAP / G-DEC / G-DRIFT for one seed, exactly as PREREG-C2W8-PASS2 §3."""
    cen = cell["census"]
    wells = cen["wells"]
    cap = np.asarray([w["capture_radius"] for w in wells], dtype=float)
    n = int(cap.size)
    frac_cap = float(np.mean(cap > 0.0)) if n else float("nan")

    probe = cell.get("self_probe", {})
    dec = float(probe.get("decode", float("nan")))
    chance = float(probe.get("chance", float("nan")))
    n_probed = int(probe.get("n_probed", 0))
    se = (float(np.sqrt(chance * (1.0 - chance) / n_probed))
          if n_probed > 0 and np.isfinite(chance) else float("nan"))

    drift = np.asarray([w["site_drift"] for w in wells], dtype=float)
    med_drift = float(np.median(drift)) if n else float("nan")
    spacing = float(cell["geometry"]["median_nn_task1"])

    return {
        "G_CAP": {"frac_capture_positive": frac_cap, "n_live": n,
                  "n_positive": int(np.sum(cap > 0.0)),
                  "pass": bool(n > 0 and frac_cap > 0.5)},
        "G_DEC": {"decode": dec, "chance": chance, "se": se, "n_probed": n_probed,
                  "threshold_2se": (chance + 2.0 * se if np.isfinite(se) else float("nan")),
                  "pass": bool(np.isfinite(dec) and np.isfinite(se)
                               and dec > chance + 2.0 * se)},
        "G_DRIFT": {"median_site_drift": med_drift, "key_spacing": spacing,
                    "pass": bool(np.isfinite(med_drift) and np.isfinite(spacing)
                                 and med_drift < spacing)},
    }


def own_foreign(cell: Dict[str, Any]) -> Dict[str, Any]:
    """⛔ DIAGNOSTIC ONLY (prereg §3), under **both** aggregations (§1 Q2).

    The **median** is canonical; the mean is reported beside it so the two forms
    are reconciled rather than circulating side by side. No gate leg reads this.
    """
    wells = cell["census"]["wells"]
    own = np.asarray([w["own_atom_depth"] for w in wells], dtype=float)
    foreign = np.asarray([w["foreign_atom_depth"] for w in wells], dtype=float)
    n = int(own.size)
    return {
        "aggregation_canonical": "median",
        "own_median": float(np.median(own)) if n else float("nan"),
        "foreign_median": float(np.median(foreign)) if n else float("nan"),
        "own_mean": float(np.mean(own)) if n else float("nan"),
        "foreign_mean": float(np.mean(foreign)) if n else float("nan"),
        "n_wells_foreign_exceeds_own": int(np.sum(foreign > own)),
        "n_wells": n,
        "status": "DIAGNOSTIC — never a target, never a gate leg (prereg §3)",
    }


def byte_ledger(cell: Dict[str, Any]) -> Dict[str, Any]:
    """The arm's sharpest column: head parameters counted, against BOTH pass-1
    anchors. ⛔ Byte-blind gate (§1 Q3) — no performance number is quoted here."""
    b = dict(cell["bytes"])
    total = int(b["clu_total_bytes"])
    head = int(b.get("emission_head_bytes", 0))
    return {
        **b,
        "emission_head_bytes": head,
        "head_share_of_total": (float(head) / total if total else float("nan")),
        "vs_pass1_clu_total_360960": float(total) / float(PASS1_CLU_TOTAL_BYTES),
        "vs_knn_launder_288": (float(total) / float(b["knn_launder_bytes"])
                               if b.get("knn_launder_bytes") else float("nan")),
        "pass1_clu_total_bytes": PASS1_CLU_TOTAL_BYTES,
        "pass1_knn_launder_bytes": PASS1_KNN_LAUNDER_BYTES,
        "gate_is_byte_blind": True,
        "note": ("ERRATA-C2W8-PASS2 §1 Q3: no gate leg reads bytes and no "
                 "performance number is quoted at the 1 253x ratio"),
    }


def placement_diagnostics(cell: Dict[str, Any]) -> Dict[str, Any]:
    """``|c_emitted - phi|`` — **the anti-pin evidence, measured not asserted**.

    A pinned/snapped center would drive this to (near) zero by construction. It
    is reported, never targeted, and no term in the head's objective is a
    function of it (the reach hinge is zero once the launch is inside ``rho*s``).
    """
    d = cell.get("emission", {}).get("center_minus_phi", None)
    if not d:
        return {"status": "NOT MEASURED"}
    v = np.asarray(d, dtype=float)
    return {
        "median_abs_center_minus_phi": float(np.median(v)),
        "min": float(np.min(v)), "max": float(np.max(v)), "n": int(v.size),
        "key_spacing": float(cell["geometry"]["median_nn_task1"]),
        "ratio_to_key_spacing": float(np.median(v)
                                      / max(cell["geometry"]["median_nn_task1"], 1e-12)),
        "status": ("DIAGNOSTIC — the emitted center is a learned continuous "
                   "function of phi and is never pinned/snapped/regularized to it"),
    }


# ---------------------------------------------------------------------------
# one arm-B cell
# ---------------------------------------------------------------------------
def run_arm_b_cell(cfg: CHLUConfig, seed: int, *, data=None, min_store: bool = False,
                   verbose: bool = True) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    cell = run_census_cell(cfg, seed, data=data, verbose=verbose,
                           clu_overrides=clu_overrides(cfg, min_store=min_store),
                           post_build=make_post_build(cfg, record))
    system = record.pop("_system", None)
    cell["emission"] = dict(record)
    cell["emission"]["min_store_cell"] = bool(min_store)
    # ⭐ |c_emitted - phi| per live item — the anti-pin evidence (see
    # `placement_diagnostics`). Eval-only bookkeeping; the read never sees it.
    if system is not None:
        ids, centers, _ = system.codebook()
        gap = []
        for i, iid in enumerate(ids):
            src = system._emitted_from_phi.get(int(iid))
            if src is not None:
                gap.append(float(np.linalg.norm(np.asarray(centers[i]) - src)))
        cell["emission"]["center_minus_phi"] = gap
    return cell


def finish_cell(system_free_cell: Dict[str, Any], n_items: int) -> Dict[str, Any]:
    """Attach the gate legs, K8's ledger and the diagnostics to a raw cell."""
    cell = system_free_cell
    cell["gate"] = gate_legs(cell)
    cell["own_foreign"] = own_foreign(cell)
    cell["byte_ledger"] = byte_ledger(cell)
    cell["placement"] = placement_diagnostics(cell)
    head_params = int(cell.get("emission", {}).get("pretrain", {})
                      .get("head_param_count", 0))
    cell["k8"] = emission_ledger(None, n_items=int(n_items))
    cell["k8"]["head_param_count"] = head_params
    cell["k8"]["head_bytes"] = int(head_params * 4)
    cell["tier_ii_status"] = cell["k8"]["tier_ii_status"]
    return cell


# ---------------------------------------------------------------------------
# the deliverable
# ---------------------------------------------------------------------------
def run_experiment_capture_armb(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "plots",
    seeds: Optional[List[int]] = None,
    quick: bool = False,
    data=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run arm B over every seed and write ``census_armB.json``."""
    cfg = config or get_default_config()
    if quick:
        apply_quick(cfg)
    b = cfg.experiment_capture_armb
    seeds = [int(s) for s in (seeds if seeds is not None else b.seeds)]

    cells: List[Dict[str, Any]] = []
    for s in seeds:
        if verbose:
            print(f"[arm-B] emission-head census seed {s} ...", flush=True)
        cell = run_arm_b_cell(cfg, s, data=data, verbose=verbose)
        cells.append(finish_cell(cell, n_items=int(cell["census"]["n_live"])))
        if verbose:
            g = cells[-1]["gate"]
            print(f"  seed {s}: G-CAP {g['G_CAP']['frac_capture_positive']:.3f} "
                  f"({'PASS' if g['G_CAP']['pass'] else 'FAIL'}) · "
                  f"G-DEC {g['G_DEC']['decode']:.4f} vs "
                  f"{g['G_DEC']['threshold_2se']:.4f} "
                  f"({'PASS' if g['G_DEC']['pass'] else 'FAIL'}) · "
                  f"G-DRIFT {g['G_DRIFT']['median_site_drift']:.4f} vs "
                  f"{g['G_DRIFT']['key_spacing']:.4f} "
                  f"({'PASS' if g['G_DRIFT']['pass'] else 'FAIL'})", flush=True)

    min_cells: List[Dict[str, Any]] = []
    if b.run_min_store_cell and seeds:
        s = seeds[0]
        if verbose:
            print(f"[arm-B] SECONDARY min-store cell (seed {s}) ...", flush=True)
        c = run_arm_b_cell(cfg, s, data=data, min_store=True, verbose=verbose)
        min_cells.append(finish_cell(c, n_items=int(c["census"]["n_live"])))

    legs = {k: [bool(c["gate"][k]["pass"]) for c in cells]
            for k in ("G_CAP", "G_DEC", "G_DRIFT")}
    results = {
        "experiment": "capture_arm_b_emission_head",
        "arm": "B — emission head (MLP-class head on phi emits the well parameters)",
        "pass": 2,
        "seeds": seeds,
        "cells": cells,
        "secondary_min_store_cells": min_cells,
        "gate": {
            "legs_by_seed": legs,
            "G_CAP_all_seeds": bool(legs["G_CAP"] and all(legs["G_CAP"])),
            "G_DEC_all_seeds": bool(legs["G_DEC"] and all(legs["G_DEC"])),
            "G_DRIFT_all_seeds": bool(legs["G_DRIFT"] and all(legs["G_DRIFT"])),
            "ALL_THREE_ALL_SEEDS": bool(
                legs["G_CAP"] and all(legs["G_CAP"]) and all(legs["G_DEC"])
                and all(legs["G_DRIFT"])),
            "rule": ("PREREG-C2W8-PASS2 §3: G-CAP majority capture_radius > 0 · "
                     "G-DEC decode above chance by 2 SE · G-DRIFT median "
                     "site_drift below the measured key spacing; >= 3 seeds"),
        },
        "k8": (cells[0]["k8"] if cells else
               emission_ledger(None, n_items=0)),
        "tier_ii_status": NO_TIER_II_CLAIM,
        "shared_vocabulary_interface": {
            "built": False,
            "specified": True,
            "form": ("head(phi_i, a_i) -> coefficients w_i in R^V over a store-owned "
                     "well vocabulary {theta_v}; item i's contribution is "
                     "sum_v w_iv * well(theta_v)"),
            "degenerate_case": ("V = n_items, w_i = one-hot(i) => the private-well "
                                "configuration this arm ships; asserted bitwise in "
                                "tests/test_emission_head.py"),
            "code": "chlu.core.emission_head.{WellVocabulary,compose_wells,private_vocabulary}",
        },
        "declared_not_runs": [
            "merge / prune / depth restoration / any §2.7 claim cell — NOT BUILT",
            "the factored store (shared well vocabulary) — SPECIFIED, not built",
            "I2 correlation test (C2W10) · cross-stream criterion (C2W10)",
            "wormholes / learned p0 traversal (C2W9)",
            "any tier-ii, full-CLU or I2 verdict (§A28.4)",
            "CSF3 — untouched",
        ],
        "wall_s": float(sum(c["wall_s"] for c in cells + min_cells)),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "census_armB.json")
    with open(path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    results["census_json"] = path
    if verbose:
        print(f"\n[arm-B] gate: {results['gate']['rule']}\n"
              f"  G-CAP {legs['G_CAP']} · G-DEC {legs['G_DEC']} · "
              f"G-DRIFT {legs['G_DRIFT']}\n"
              f"  ALL THREE, ALL SEEDS = {results['gate']['ALL_THREE_ALL_SEEDS']}\n"
              f"  K8: wells_per_item={results['k8']['wells_per_item']} "
              f"vocabulary_shared={results['k8']['vocabulary_shared']} "
              f"-> {results['tier_ii_status']}\n"
              f"  wrote {path}", flush=True)
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Smoke mode: a real arm-B census on a tiny stream (never a claim cell)."""
    apply_quick_census(config)
    b = config.experiment_capture_armb
    b.quick = True
    b.seeds = [0]
    b.pretrain_steps = 25
    b.pretrain_batch = 8
    b.pretrain_pool = 48
    b.run_min_store_cell = False


def main():
    parser = argparse.ArgumentParser(
        description="C2W8 pass 2 ARM B — the emission head, scored on pass 1's census")
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
    run_experiment_capture_armb(config, save_dir=save_dir, seeds=seeds,
                                quick=args.quick)


if __name__ == "__main__":
    main()


__all__ = [
    "PASS1_CLU_TOTAL_BYTES", "PASS1_KNN_LAUNDER_BYTES", "clu_overrides",
    "make_post_build", "gate_legs", "own_foreign", "byte_ledger",
    "placement_diagnostics", "run_arm_b_cell", "finish_cell",
    "run_experiment_capture_armb", "apply_quick",
]
