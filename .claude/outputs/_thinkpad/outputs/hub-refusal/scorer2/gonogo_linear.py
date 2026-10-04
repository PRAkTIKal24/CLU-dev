"""hub/refusal-2 -- THE GO / NO-GO (PREREG-hub-refusal-2 section 4).

Runs the pre-registered statistic  S = E_store(x_read)  at the LINEAR read on the
INHERITED fit-derived synthetic analogue, at one rung, seed 0, under both arms.

DECLARED AND BINDING:
  * bases, base count and RNG seed are the PREDECESSOR's (``synthetic_probe.build_sets``,
    ``N_BASES = 256``, ``RNG_SEED = 20260916``) -- adopted verbatim, NOT re-tuned;
  * ``c_invalid.json`` and ``a7_redundant.json`` are NOT opened by this script;
  * the threshold is the declared rule, unchanged: ``quantile(S on val, q = 0.95)`` with
    ``q`` the harness's own shipped ``refusal_valid_quantile``;
  * MECHANISM ONLY.  No threshold, criterion, control, companion or bar may be changed
    because of what this prints.  If any is, the zero-supervision claim is VOID.
  * fit rows are SEEN, so the valid side here is easier than ``B``.  These numbers are
    NOT the benchmark and may never be quoted as the refusal result.

THE GATE:  min(AUROC_oor_pooled, AUROC_dup_vs_a7) < 0.60  =>  DO NOT SPEND.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

import numpy as np  # noqa: E402

import refusal2_lib as L  # noqa: E402
import refusal_lib as R  # noqa: E402
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from chlu.eval.rivals import gen_knn as knn  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

Q = 0.95
ARMS = ("linear", "linear_store_removed")
OOR_SETS = ("syn_oor_1", "syn_oor_2", "syn_oor_3")


def au(neg, pos) -> float:
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    return float(binary_auroc(s, y))


def stat_block(vals: dict, key: str) -> dict:
    """The full row for ONE statistic, under ONE arm.  ``vals[set][key] -> (n,)``."""
    x = {k: np.asarray(v[key], np.float64) for k, v in vals.items()}
    thr = knn.threshold_from_valid(x["val"], Q)
    refused = {k: float((v > thr).mean()) for k, v in x.items()}
    oor_pooled = np.concatenate([x[s] for s in OOR_SETS])
    return {
        "threshold_from_val": float(thr),
        "mean": {k: float(v.mean()) for k, v in x.items()},
        "refused_fraction": refused,
        "accept_fraction": {k: float(1.0 - refused[k]) for k in refused},
        # THE TWO GO/NO-GO NUMBERS, declared before the run:
        "AUROC_oor_pooled_vs_syn_valid": au(x["syn_valid"], oor_pooled),
        "AUROC_dup_vs_syn_a7_THE_CONTROLLED_ONE": au(x["syn_a7"], x["syn_dup"]),
        # reported beside them, never in place of them:
        "AUROC_oor_by_severity": {s: au(x["syn_valid"], x[s]) for s in OOR_SETS},
        "AUROC_dup_vs_syn_valid": au(x["syn_valid"], x["syn_dup"]),
        "AUROC_a7_vs_syn_valid_MUST_BE_NEAR_HALF": au(x["syn_valid"], x["syn_a7"]),
        "AUROC_synvalid_vs_val_MUST_BE_NEAR_HALF": au(x["val"], x["syn_valid"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    grid = ctx["grid"]
    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    base = requests_from_rows(grid, _fit[_take])
    sets = build_sets(base, grid, rng)
    sets = {"val": requests_from_rows(grid, ctx["val_rows"]), **sets}

    report = {
        "schema": "hub-refusal-2/gonogo-linear/1",
        "DECLARED": ("FIT-DERIVED SYNTHETIC, PREREG-hub-refusal-2 section 4. MECHANISM "
                     "ONLY, the GO/NO-GO. NOT the benchmark; fit rows are SEEN so the "
                     "valid side is easier than B; a constructed negative is not a C "
                     "record. c_invalid.json and a7_redundant.json were NOT opened."),
        "object": "S = E_store(x_read), x_read = posterior_mean linear read",
        "seed": int(a.seed), "budget": str(a.budget), "q": Q,
        "n_bases": N_BASES, "rng_seed": RNG_SEED,
        "gate": "min(AUROC_oor_pooled, AUROC_dup_vs_a7) < 0.60 => DO NOT SPEND",
        "cache_meta": ctx["cache_meta"],
        "reencode_check": ctx["reencode_check"],
        "arms": {},
    }

    rundir = R.OUT / "runs2" / f"gonogo_seed{a.seed}_{a.budget}{a.tag}"
    for arm in ARMS:
        R.tlog(f"--- arm {arm} ---")
        vals, metas = {}, {}
        for name, reqs in sets.items():
            rec = L.linear_unit(ctx, built, f"gng_{name}", reqs, arm=arm,
                                budget=str(a.budget), log=lambda *_: None)
            L.save_linear_unit(rec, rundir / f"{arm}__{name}.npz")
            vals[name] = rec
            metas[name] = rec["meta"]
            R.tlog(f"    {name:<12} n={rec['meta']['n']:<5} "
                   f"meanS={rec['S'].mean():.4f} conf={rec['meta']['mean_posterior_confidence']:.4f} "
                   f"({rec['meta']['wall_clock_s']:.1f}s)")
        row = {
            "S_THE_PREREGISTERED_OBJECT": stat_block(vals, "S"),
            "M_physics_free_margin_CONTROL_2": stat_block(vals, "M"),
            "d_over_spread_COMPANION_REJECTED_IN_ADVANCE": stat_block(vals, "d_over_spread"),
            "S_at_the_ARGMAX_read_DIAGNOSTIC_the_dead_read": stat_block(vals, "S_argmax_read"),
            "mean_d_argmax_read_MUST_BE_ZERO": {
                k: float(np.mean(v["d_argmax_read"])) for k, v in vals.items()},
            "mean_posterior_confidence": {
                k: m["mean_posterior_confidence"] for k, m in metas.items()},
            "linear_read_reproduction": {
                k: m["linear_read_reproduction"] for k, m in metas.items()},
            "experts_reproduction": {
                k: m["experts_reproduction"] for k, m in metas.items()},
        }
        report["arms"][arm] = row

        s = row["S_THE_PREREGISTERED_OBJECT"]
        print(f"\n=== arm={arm}  budget={a.budget}  threshold(val, q={Q}) = "
              f"{s['threshold_from_val']:.4f} ===")
        print("  %-12s %9s %9s %12s" % ("set", "refused", "mean S", "AUROC vs syn_valid"))
        for k in ["val", "syn_valid", "syn_a7", "syn_dup", "syn_oor_1", "syn_oor_2", "syn_oor_3"]:
            auv = s["AUROC_oor_by_severity"].get(k)
            if k == "syn_dup":
                auv = s["AUROC_dup_vs_syn_valid"]
            if k == "syn_a7":
                auv = s["AUROC_a7_vs_syn_valid_MUST_BE_NEAR_HALF"]
            print("  %-12s %9.4f %9.4f %12s" % (
                k, s["refused_fraction"][k], s["mean"][k],
                ("%.4f" % auv) if auv is not None else "--"))
        print("  ** AUROC_oor (pooled 1-3) = %.4f   AUROC_dup (vs syn_a7, CONTROLLED) = %.4f"
              % (s["AUROC_oor_pooled_vs_syn_valid"],
                 s["AUROC_dup_vs_syn_a7_THE_CONTROLLED_ONE"]))
        m = row["M_physics_free_margin_CONTROL_2"]
        print("  control 2 (physics-free margin M): oor %.4f  dup %.4f"
              % (m["AUROC_oor_pooled_vs_syn_valid"],
                 m["AUROC_dup_vs_syn_a7_THE_CONTROLLED_ONE"]))

    full = report["arms"]["linear"]["S_THE_PREREGISTERED_OBJECT"]
    rem = report["arms"]["linear_store_removed"]["S_THE_PREREGISTERED_OBJECT"]
    gate_min = min(full["AUROC_oor_pooled_vs_syn_valid"],
                   full["AUROC_dup_vs_syn_a7_THE_CONTROLLED_ONE"])
    rem_min = min(rem["AUROC_oor_pooled_vs_syn_valid"],
                  rem["AUROC_dup_vs_syn_a7_THE_CONTROLLED_ONE"])
    rem_max = max(rem["AUROC_oor_pooled_vs_syn_valid"],
                  rem["AUROC_dup_vs_syn_a7_THE_CONTROLLED_ONE"])
    report["VERDICT"] = {
        "gate_min_full_store": gate_min,
        "gate_passes_0.60": bool(gate_min >= 0.60),
        "store_removed_min": rem_min, "store_removed_max": rem_max,
        "control_1_collapses": bool(abs(rem_max - 0.5) <= 0.10 and abs(rem_min - 0.5) <= 0.10),
        "note": ("control 1 (charter C.2) must collapse to ~0.5 or the result is VOID; "
                 "the gate and control 1 are BOTH required before any scored spend."),
    }
    p = R.OUT / f"GONOGO_LINEAR_seed{a.seed}_{a.budget}{a.tag}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("\n" + json.dumps(report["VERDICT"], indent=1))
    print(f"wrote {p}")
    print("FIT-DERIVED SYNTHETIC. MECHANISM ONLY. NOT the refusal result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
