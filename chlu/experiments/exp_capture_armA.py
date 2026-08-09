"""⭐ Experiment **CAPTURE ARM A** (C2W8 pass 2) — *make the store capture by
bounding how far each atom reaches.*

**What pass 1 measured, and what this changes.** C3 locality **holds in parameter
space** (a masked write's own-leg violation rate is 0.000, exact) and **fails in
function space** (78-84 % of writes raise the foreign contribution; foreign
exceeds own on 45 of 48 wells; `capture_radius` is exactly 0.000 on 47 of 48
wells while `lambda_min > 0` everywhere). **A write touches only its own atom
block, but atoms have TAILS.** This arm makes the atom influence profile
**compact** — identically zero beyond `R = cutoff * s` — with `s` co-scaled to
the **measured** task-1 key spacing of that seed's own run. It is the wave's own
K2 lesson (the trash region needed a compact gate, because a sigmoid tail makes a
"local" change global) applied one level down, to the atoms.

⛔ **The census is FROZEN and this file does not touch it.** Every cell is
:func:`chlu.experiments.exp_well_lifecycle.run_census_cell`, called unmodified on
the unmodified :mod:`chlu.core.well_lifecycle` instrument, so arm A and arm B are
raced on one arithmetic. The *only* thing this file substitutes is the **store
config factory** (`store_config`), which is what an arm is allowed to change; the
substitution is explicit, scoped by `try/finally`, and recorded in the artifact.

⛔⛔ **This arm does NOT pin attractors to designed anchors.** It changes *how far
each atom's influence reaches* and (as the companion lever the compact kernel
needs to be runnable at all) *where the admitted slot's atoms are initialised*.
Placement stays learned and continuous, the settled point stays free, and basins
stay free to interact. See :func:`chlu.core.memory_potentials.localize_group_atoms`.

⛔ **Declared NOT-RUNs** (never nulls): merge, prune, depth restoration and every
§2.7 claim cell (gated on the capture gate); the factored store; I2; cross-stream;
wormholes / learned p0; CSF3; any tier-ii, full-CLU or I2 verdict; **any
performance claim** — the pass-2 gate is retrievability and is **byte-blind**
(`ERRATA-C2W8-PASS2.md` §1 Q3), though the byte ledger is reported on every arm.

Runnable::

    uv run python -m chlu.experiments.exp_capture_armA --quick
    uv run python -m chlu.experiments.exp_capture_armA --seeds 0,1,2 --save-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.clu_system import CluSystemConfig
from chlu.experiments import exp_well_lifecycle as ewl

#: chance level of the self-probe `decode` leg is 1/n_live; the census reports it
CHANCE_KEY = "chance"

#: ⛔ Bound ONCE at import, so the substitution below can never see (or recurse
#: into) its own replacement, whatever else a caller has patched.
_FROZEN_STORE_CONFIG = ewl.store_config


# ---------------------------------------------------------------------------
# the arm: a store config, and nothing else
# ---------------------------------------------------------------------------
def arm_store_config(cfg: CHLUConfig, seed: int, d_safe: float,
                     overrides: Optional[Dict[str, Any]] = None) -> CluSystemConfig:
    """Pass 1's store config with **the reach lever** applied — nothing else.

    ⭐ The width co-scaling is to the **measured** key spacing of this seed's own
    run, never a hardcoded 0.14: `run_census_cell` computes
    `d_safe = d_safe_frac * median_nn_task1` on task-1 `phi` keys and hands it to
    the store-config factory, so the spacing is recovered exactly as
    `d_safe / d_safe_frac`.
    """
    # ⚠ C2W8 pass 3 (blocking-bug fix, reconciliation item): the `overrides`
    # keyword is the pass-2 arm-B seam that `run_census_cell` ALWAYS passes. Arm
    # A's substituted factory did not accept it, so `exp_capture_armA` raised
    # `TypeError: got an unexpected keyword argument 'overrides'` at
    # `main @ 1eda6a0` — i.e. **arm A could not run at all on main**; the two arms
    # were merged independently and neither merge re-ran the other. Arm A never
    # sets overrides, so forwarding them is behaviour-preserving and the pass-2
    # numbers are unaffected. Pinned by
    # `tests/test_gate_addr.py::test_arm_a_store_config_accepts_the_overrides_seam`.
    base = _FROZEN_STORE_CONFIG(cfg, seed, d_safe,
                                overrides=overrides)   # the FROZEN pass-1 config
    a = cfg.experiment_capture_arm_a
    w = cfg.experiment_well_lifecycle
    med_nn = float(d_safe) / float(w.d_safe_frac)
    s = (float(a.atom_width_frac_spacing) * med_nn
         if a.atom_width_frac_spacing is not None else float(base.atom_width))
    return dataclasses.replace(
        base,
        atom_width=float(s),
        atom_kernel=str(a.atom_kernel),
        atom_kernel_cutoff=float(a.atom_kernel_cutoff),
        atom_site_local_init=bool(a.site_local_init),
        atom_site_local_radius=float(a.site_local_radius_frac) * float(s),
    )


def run_cell(cfg: CHLUConfig, seed: int, *, data=None, verbose: bool = True) -> Dict[str, Any]:
    """One census cell on the arm's store — pass 1's cell, arm A's store.

    The `store_config` substitution is the whole diff and it is undone in a
    `finally`, so nothing leaks into a later cell or a later arm.
    """
    original = ewl.store_config
    ewl.store_config = lambda c, s, d, overrides=None: arm_store_config(
        c, s, d, overrides=overrides)
    try:
        cell = ewl.run_census_cell(cfg, seed, data=data, verbose=verbose)
    finally:
        ewl.store_config = original
    return cell


# ---------------------------------------------------------------------------
# the gate (PREREG-C2W8-PASS2 §3) — computed mechanically, never by judgement
# ---------------------------------------------------------------------------
def gate_legs(cell: Dict[str, Any]) -> Dict[str, Any]:
    """G-CAP / G-DEC / G-DRIFT for one seed, plus their ingredients.

    * **G-CAP** `capture_radius > 0` on a **majority** of live wells.
      ⚠ Reported beside the **stricter** `capture_radius >= sigma_q` count,
      because K7 pinned an instrument floor of `tol / expansion_rate`: a site
      whose relaxation barely moves reports a positive radius with no basin
      (`tests/test_compact_atoms.py::test_k7_capture_radius_floor_*`).
    * **G-DEC** self-probe `decode` above chance beyond **2 SE**
      (binomial SE at `n_probed`).
    * **G-DRIFT** median `site_drift` below the **measured** key spacing.
    """
    cen = cell["census"]
    wells = cen["wells"]
    cap = np.asarray([w["capture_radius"] for w in wells], dtype=float)
    drift = np.asarray([w["site_drift"] for w in wells], dtype=float)
    n = int(len(wells))
    sigma_q = float(cen["theta_att_block"]["sigma_q"])
    probe = cell["self_probe"]
    dec = float(probe.get("decode", float("nan")))
    chance = float(probe.get(CHANCE_KEY, float("nan")))
    n_probed = int(probe.get("n_probed", 0))
    se = float(np.sqrt(max(chance * (1.0 - chance), 0.0) / max(n_probed, 1)))
    spacing = float(cell["geometry"]["median_nn_task1"])
    frac_pos = float(np.mean(cap > 0.0)) if n else float("nan")
    med_drift = float(np.median(drift)) if n else float("nan")
    return {
        "n_live": n,
        "G_CAP": {
            "frac_capture_positive": frac_pos,
            "n_capture_positive": int(np.sum(cap > 0.0)),
            "n_capture_ge_sigma_q": int(np.sum(cap >= sigma_q)),
            "sigma_q": sigma_q,
            "capture_median": float(np.median(cap)) if n else float("nan"),
            "capture_max": float(np.max(cap)) if n else float("nan"),
            "pass": bool(frac_pos > 0.5),
        },
        "G_DEC": {
            "decode": dec, "chance": chance, "n_probed": n_probed, "se": se,
            "margin_in_se": float((dec - chance) / se) if se > 0 else float("nan"),
            "pass": bool(dec > chance + 2.0 * se),
        },
        "G_DRIFT": {
            "median_site_drift": med_drift, "key_spacing": spacing,
            "ratio": float(med_drift / spacing) if spacing > 0 else float("nan"),
            "pass": bool(med_drift < spacing),
        },
    }


def own_foreign(cell: Dict[str, Any]) -> Dict[str, Any]:
    """The own/foreign split — ⛔ **DIAGNOSTIC, never a target, never a gate leg.**

    Reported under **BOTH** aggregations, each labelled (`ERRATA-C2W8-PASS2.md`
    §1 Q2; the **median** is canonical). Under private wells a high foreign
    contribution is interference; in a factored store it is the **signal**
    (compositionality), so over-fitting to own-dominance now buys a reversal
    later. The invariant that survives both designs is **retrievability**, which
    is what the gate measures.

    ⚠ **Kernel mismatch, carried honestly.** `well_lifecycle.own_foreign_site_depth`
    hard-codes the **Gaussian** atom sum. That file is READ-ONLY for this arm
    (the race depends on one arithmetic), so under a compact kernel this
    diagnostic over-reads both legs by the Gaussian tail it assumes. Direction of
    the bias is known (up), the gate does not read it, and it is filed as a
    reconciliation item for the Hub.
    """
    wells = cell["census"]["wells"]
    own = np.asarray([w["own_atom_depth"] for w in wells], dtype=float)
    foreign = np.asarray([w["foreign_atom_depth"] for w in wells], dtype=float)
    return {
        "status": "DIAGNOSTIC — never a target, never a gate leg (prereg §3)",
        "estimator": ("frozen census `own_foreign_site_depth`, which assumes a "
                      "GAUSSIAN atom kernel; kernel-mismatched under arm A"),
        "own_median": float(np.median(own)), "foreign_median": float(np.median(foreign)),
        "own_mean": float(np.mean(own)), "foreign_mean": float(np.mean(foreign)),
        "n_foreign_exceeds_own": int(np.sum(foreign > own)),
        "n_wells": int(own.size),
    }


def byte_ledger(cell: Dict[str, Any], cfg: CHLUConfig) -> Dict[str, Any]:
    """K5: the ledger on every arm including the launder, `(d, atom budget)` as
    **ONE joint dial**, `gamma_phi` holes counted.

    ⛔ Byte-blind gate (`ERRATA-C2W8-PASS2.md` §1 Q3): no gate leg reads this and
    **no performance number is quoted at the ratio**.
    """
    b = dict(cell["bytes"])
    w = cfg.experiment_well_lifecycle
    d = int(w.addr_dim)
    n_atoms = int(cell["flags"]["n_atoms"])
    b.update({
        "joint_dial_(d, atom_budget)": {
            "addr_dim": d, "n_atoms": n_atoms,
            "rule": "n_atoms = max(atoms_per_item*K, min_atoms, round(512*sqrt(2)^d))",
            "bytes_per_atom": 4 * (d + int(w.payload_dim) + 2),
            "note": ("ONE dial: bytes/well grow with d while the compact reach "
                     "tightens as the co-scaled width; they are not independent"),
        },
        "ratio_clu_over_knn_launder": float(b["clu_total_bytes"] / max(b["knn_launder_bytes"], 1)),
        "gate_reads_bytes": False,
        "performance_claim": "NONE — pass 2 is a capture/instrument gate",
    })
    return b


# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------
def run_arm(cfg: CHLUConfig, seeds: List[int], *, label: str, data=None,
            verbose: bool = True) -> Dict[str, Any]:
    cells, legs = [], []
    for s in seeds:
        if verbose:
            print(f"[armA:{label}] census seed {s} ...", flush=True)
        c = run_cell(cfg, s, data=data, verbose=verbose)
        cells.append(c)
        lg = gate_legs(c)
        legs.append(lg)
        if verbose:
            print(f"  seed {s}: G-CAP {lg['G_CAP']['frac_capture_positive']:.3f} "
                  f"({lg['G_CAP']['pass']}) | G-DEC {lg['G_DEC']['decode']:.4f} vs "
                  f"{lg['G_DEC']['chance']:.4f} ({lg['G_DEC']['pass']}) | G-DRIFT "
                  f"{lg['G_DRIFT']['median_site_drift']:.4f} vs "
                  f"{lg['G_DRIFT']['key_spacing']:.4f} ({lg['G_DRIFT']['pass']})",
                  flush=True)
    n_seed = len(cells)
    all_pass = [bool(lg["G_CAP"]["pass"] and lg["G_DEC"]["pass"] and lg["G_DRIFT"]["pass"])
                for lg in legs]
    return {
        "label": label,
        "seeds": [int(s) for s in seeds],
        "arm_config": dataclasses.asdict(cfg.experiment_capture_arm_a),
        "store_flags": [c["flags"]["clu_system_non_defaults"] for c in cells],
        "gate": {
            "legs_by_seed": legs,
            "G_CAP_pass_seeds": int(sum(lg["G_CAP"]["pass"] for lg in legs)),
            "G_DEC_pass_seeds": int(sum(lg["G_DEC"]["pass"] for lg in legs)),
            "G_DRIFT_pass_seeds": int(sum(lg["G_DRIFT"]["pass"] for lg in legs)),
            "all_three_same_seed": int(sum(all_pass)),
            "n_seeds": n_seed,
            "gate_pass": bool(n_seed >= 3 and all(all_pass)),
            "rule": ("PREREG-C2W8-PASS2 §3: G-CAP majority of live wells with "
                     "capture_radius>0; G-DEC decode above chance beyond 2 SE; "
                     "G-DRIFT median site_drift below the measured key spacing; "
                     ">= 3 seeds"),
        },
        "own_foreign_by_seed": [own_foreign(c) for c in cells],
        "bytes_by_seed": [byte_ledger(c, cfg) for c in cells],
        "depth_raw_median_by_seed": [c["census"]["depth_raw_median"] for c in cells],
        "geometry_by_seed": [c["geometry"] for c in cells],
        "self_probe_by_seed": [c["self_probe"] for c in cells],
        "cells": cells,
        "wall_s": float(sum(c["wall_s"] for c in cells)),
    }


def run_experiment_capture_arm_a(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "plots",
    seeds: Optional[List[int]] = None,
    quick: bool = False,
    baseline: Optional[bool] = None,
    data=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run arm A (and, if asked, the pass-1 Gaussian baseline on this branch)."""
    cfg = config or get_default_config()
    if quick:
        ewl.apply_quick(cfg)
        cfg.experiment_capture_arm_a.quick = True
        cfg.experiment_capture_arm_a.seeds = [0]
    a = cfg.experiment_capture_arm_a
    seeds = [int(s) for s in (seeds if seeds is not None else a.seeds)]
    t0 = time.time()

    arms = {"armA_compact": run_arm(cfg, seeds, label="armA_compact", data=data,
                                    verbose=verbose)}
    if a.run_baseline_gaussian if baseline is None else baseline:
        base_cfg = copy.deepcopy(cfg)
        b = base_cfg.experiment_capture_arm_a
        b.atom_kernel = "gaussian"
        b.atom_width_frac_spacing = None
        b.site_local_init = False
        arms["baseline_gaussian"] = run_arm(base_cfg, seeds, label="baseline_gaussian",
                                            data=data, verbose=verbose)

    results = {
        "experiment": "capture_arm_a",
        "wave": "C2W8 pass 2",
        "dial_declaration": {
            "dial": "none as a new claim — instrument/mechanism repair on the write side",
            "laundering_control": ("kNN-in-phi launder carried with the byte ledger on "
                                   "every reading; gate is BYTE-BLIND (ERRATA §1 Q3)"),
            "falsifies": "the gate fails on >= 3 seeds => measured negative, reported as such",
            "does_not_falsify": ("losing to the launder (no performance claim); a high "
                                 "foreign contribution (diagnostic); losing to arm B"),
            "depth_is_not_feature_importance": "§A23.5 ACTIVE",
        },
        "seeds": seeds,
        "arms": arms,
        "declared_not_runs": [
            "merge / prune / depth restoration / §2.7 claim cells — NOT BUILT (gated on the capture gate)",
            "factored store / shared well vocabulary — specified elsewhere (K8), not built",
            "I2 correlation test (C2W10); cross-stream criterion (C2W10)",
            "wormholes / learned p0 traversal (C2W9); CSF3 untouched",
            "any tier-ii, full-CLU or I2 verdict; any performance claim",
        ],
        "wall_s": float(time.time() - t0),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "capture_armA.json")
    with open(path, "w") as f:
        json.dump(ewl._jsonable(results), f, indent=2)
    results["artifact"] = path
    if verbose:
        for name, arm in arms.items():
            g = arm["gate"]
            print(f"\n[armA] {name}: gate_pass={g['gate_pass']} "
                  f"(CAP {g['G_CAP_pass_seeds']}/{g['n_seeds']}, "
                  f"DEC {g['G_DEC_pass_seeds']}/{g['n_seeds']}, "
                  f"DRIFT {g['G_DRIFT_pass_seeds']}/{g['n_seeds']})", flush=True)
        print(f"  wrote {path}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description="C2W8 pass 2 — ARM A (compact atoms)")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None, help="e.g. 0,1,2")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--baseline", action="store_true",
                   help="also re-run the pass-1 Gaussian store on this branch")
    p.add_argument("--kernel", type=str, default=None)
    p.add_argument("--width-frac", type=float, default=None)
    p.add_argument("--cutoff", type=float, default=None)
    p.add_argument("--no-site-local-init", action="store_true")
    p.add_argument("--save-dir", type=str, default="plots")
    args = p.parse_args()

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
    a = config.experiment_capture_arm_a
    if args.kernel:
        a.atom_kernel = args.kernel
    if args.width_frac is not None:
        a.atom_width_frac_spacing = args.width_frac
    if args.cutoff is not None:
        a.atom_kernel_cutoff = args.cutoff
    if args.no_site_local_init:
        a.site_local_init = False
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else None)
    run_experiment_capture_arm_a(config, save_dir=save_dir, seeds=seeds,
                                 quick=args.quick, baseline=args.baseline or None)


if __name__ == "__main__":
    main()


__all__ = [
    "arm_store_config", "run_cell", "gate_legs", "own_foreign", "byte_ledger",
    "run_arm", "run_experiment_capture_arm_a",
]
