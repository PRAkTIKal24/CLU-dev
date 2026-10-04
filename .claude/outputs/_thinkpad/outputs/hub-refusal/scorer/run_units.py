"""hub/refusal -- the resumable unit runner.

ONE UNIT = (seed, budget rung, request set, inner arm).  Each unit is written to disk
the moment its chain finishes and is recorded in ``runs/LEDGER.jsonl``.  A kill costs
ONE UNIT, never the run (`scripts/csf3/job_cpu_gen.sh`'s discipline; compute is LOCAL
here, so resumability is the only protection there is).

Usage::

    python run_units.py --seeds 0 1 2 --budget A --sets val MS --arms full
    python run_units.py --seeds 0 1 2 --budget A --sets val A B MS C A7 A7D \
                        --arms full request_off zero_step

A unit already on disk is skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402

RUNS = R.OUT / "runs"
LEDGER = RUNS / "LEDGER.jsonl"

#: The sets that cost NO held-out budget: val is the tuning split, MS is drawn from
#: store_train.  Everything else is a scored set and is gated on the Head's answer.
FREE_SETS = ("val", "MS")
SCORED_SETS = ("A", "B", "C", "A7", "A7D")


def ledger_append(rec: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def unit_path(seed: int, budget: str, set_name: str, arm: str) -> Path:
    return RUNS / f"seed_{int(seed)}" / f"{budget}_{set_name}_{arm}.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", default=R.HEADLINE_BUDGET)
    ap.add_argument("--sets", nargs="+", default=list(FREE_SETS))
    ap.add_argument("--arms", nargs="+", default=["full"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-scored", action="store_true",
                    help="required to touch A / B / C / A7 / A7D -- the scored sets")
    ap.add_argument("--prereg", default="",
                    help="the PREREG file this run is executing; recorded in the ledger")
    a = ap.parse_args()

    wants_scored = [s for s in a.sets if s in SCORED_SETS]
    if wants_scored and not a.allow_scored:
        print(f"REFUSING: {wants_scored} are held-out scored sets and --allow-scored "
              "was not given. A scored run needs a PREREG and the Head's count.")
        return 2
    if wants_scored and not a.prereg:
        print("REFUSING: a scored run must name its PREREG file (--prereg).")
        return 2

    RUNS.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for seed in a.seeds:
        R.tlog(f"=== seed {seed} · budget {a.budget} ===")
        todo = [(s, m) for s in a.sets for m in a.arms
                if a.force or not unit_path(seed, a.budget, s, m).exists()]
        if not todo:
            R.tlog(f"  every unit for seed {seed} is already on disk -- skipped")
            continue
        t0 = time.time()
        ctx = R.build_context(int(seed))
        built = R.build_land(ctx, str(a.budget))
        reqs = R.request_sets(ctx, sorted({s for s, _ in todo}))
        R.tlog(f"  context+landscape ready in {time.time() - t0:.1f}s; "
               f"sets={ {k: len(v) for k, v in reqs.items()} }")
        # the build-time provenance, written once per (seed, budget)
        prov = {
            "seed": int(seed), "budget": str(a.budget),
            "coupling_certificate": built["coupling_certificate"],
            "byte_ledger": built["byte_ledger"],
            "byte_ledger_matches_prereg": built["byte_ledger_matches_prereg"],
            "atoms_are_train_rows": built["atoms_are_train_rows"],
            "sigma": built["sigma"],
            "reencode_check": ctx["reencode_check"],
            "cache_meta": ctx["cache_meta"],
            "row_split": ctx["row_split"],
            "grouping": ctx["grouping"],
            "encoder_param_hash": ctx["encoder_param_hash"],
            "n_rows": {"pool": int(ctx["pool"].size), "val": int(ctx["val_rows"].size),
                       "A": int(ctx["eval_a"].size), "B": int(ctx["eval_b"].size),
                       "MS": int(ctx["ms_rows"].size)},
        }
        pp = RUNS / f"seed_{int(seed)}" / f"{a.budget}_provenance.json"
        pp.parent.mkdir(parents=True, exist_ok=True)
        with open(pp, "w", encoding="utf-8") as fh:
            json.dump(prov, fh, indent=1, default=str)

        for set_name, arm in todo:
            p = unit_path(seed, a.budget, set_name, arm)
            t1 = time.time()
            try:
                rec = R.settle_unit(ctx, built, set_name, reqs[set_name], arm=arm,
                                    budget=str(a.budget))
                info = R.save_unit(rec, p)
                m = rec["meta"]
                ledger_append({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "ok", "seed": int(seed), "budget": str(a.budget),
                    "set": set_name, "arm": arm, "n": m["n"], "steps": m["steps"],
                    "wall_clock_s": m["wall_clock_s"], "path": info["path"],
                    "bytes": info["bytes"],
                    "experts_reproduction": m["experts_reproduction"],
                    "frac_still_descending":
                        m["pooled_convergence"]["frac_still_descending"],
                    "energy_first": m["pooled_convergence"]["energy_first"],
                    "energy_final": m["pooled_convergence"]["energy_final"],
                    "diverged": m["diverged"],
                    "prereg": a.prereg,
                    "scored_set": bool(set_name in SCORED_SETS),
                })
                R.tlog(f"  [{set_name}/{arm}] n={m['n']} {time.time() - t1:.1f}s -> "
                       f"{p.name}  still_descending="
                       f"{m['pooled_convergence']['frac_still_descending']}")
            except Exception as exc:  # a failed unit is a ledger row, never a silence
                ledger_append({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "FAILED", "seed": int(seed), "budget": str(a.budget),
                    "set": set_name, "arm": arm, "error": repr(exc),
                    "traceback": traceback.format_exc(), "prereg": a.prereg,
                })
                R.tlog(f"  [{set_name}/{arm}] FAILED: {exc!r}")
        del ctx, built, reqs
    R.tlog(f"ALL UNITS DONE in {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
