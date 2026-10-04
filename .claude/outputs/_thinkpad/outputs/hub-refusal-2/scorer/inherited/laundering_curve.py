"""hub/refusal -- THE LAUNDERING CURVE (PREREG AMENDMENT 7).

The AMENDMENT-5 probe measured ``AUROC(syn_dup vs syn_a7)`` at **0.7511 at the launch**
and **0.5825 after 1000 settling steps**.  This script measures the same statistic at a
declared ladder of step budgets, so "the settling launders the unsatisfiability" is a
CURVE rather than two endpoints.

DECLARED AND BINDING, exactly as AMENDMENT 3:
  * same fit-derived bases (AMENDMENT 5 random sample), same RNG seed, same statistic
    ``S_max``, same threshold rule, same rung A.  ONLY the step budget is swept, and it
    is swept over a ladder fixed here before the run.
  * MECHANISM ONLY.  Nothing in PREREG-1 may change because of what this prints.
  * FIT-DERIVED SYNTHETIC.  NOT the benchmark.  Never quotable as the refusal result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402
import synthetic_probe as P  # noqa: E402
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from chlu.eval.rivals import gen_knn as knn  # noqa: E402

#: The declared ladder.  Fixed here, before the run.
STEPS = (0, 10, 25, 50, 100, 250, 500, 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    grid = ctx["grid"]
    rng = np.random.default_rng(P.RNG_SEED)
    fit = np.asarray(ctx["fit_rows"], np.int64)
    take = np.sort(rng.choice(fit.size, size=min(P.N_BASES, fit.size), replace=False))
    base = requests_from_rows(grid, fit[take])
    sets = P.build_sets(base, grid, rng)
    val_reqs = requests_from_rows(grid, ctx["val_rows"])

    rows = []
    for n_steps in STEPS:
        arm = "zero_step" if n_steps == 0 else "full"
        S = {}
        S["val"] = R.settle_unit(ctx, built, "curve_val", val_reqs, arm=arm,
                                 budget=str(a.budget), steps=n_steps,
                                 log=lambda *_: None)["settled"]["S_max"]
        for name in ("syn_valid", "syn_a7", "syn_dup", "syn_oor_3"):
            S[name] = R.settle_unit(ctx, built, f"curve_{name}", sets[name], arm=arm,
                                    budget=str(a.budget), steps=n_steps,
                                    log=lambda *_: None)["settled"]["S_max"]
        thr = knn.threshold_from_valid(S["val"], P.Q)
        rr = lambda x: float(knn.refusal_rates(x, thr)["refused_fraction"])  # noqa: E731
        row = {
            "steps": int(n_steps),
            "threshold_from_val": float(thr),
            "auroc_dup_vs_a7_CONTROLLED": P.au(S["syn_a7"], S["syn_dup"]),
            "auroc_oor3_vs_valid": P.au(S["syn_valid"], S["syn_oor_3"]),
            "auroc_a7_vs_valid_MUST_BE_CHANCE": P.au(S["syn_valid"], S["syn_a7"]),
            "mean_S_max": {k: float(np.mean(v)) for k, v in S.items()},
            "refused": {k: rr(v) for k, v in S.items()},
        }
        rows.append(row)
        print("steps=%5d  AUROC(dup vs a7)=%.4f  AUROC(oor3 vs valid)=%.4f  "
              "AUROC(a7 vs valid)=%.4f  meanS(dup)=%8.3f meanS(valid)=%7.3f  tau=%7.3f"
              % (n_steps, row["auroc_dup_vs_a7_CONTROLLED"], row["auroc_oor3_vs_valid"],
                 row["auroc_a7_vs_valid_MUST_BE_CHANCE"],
                 row["mean_S_max"]["syn_dup"], row["mean_S_max"]["syn_valid"], thr),
              flush=True)

    out = {"schema": "hub-refusal/laundering-curve/1",
           "DECLARED": ("FIT-DERIVED SYNTHETIC, PREREG AMENDMENT 7. MECHANISM ONLY. "
                        "NOT the benchmark."),
           "seed": int(a.seed), "budget": str(a.budget), "q": P.Q,
           "n_bases": P.N_BASES, "rng_seed": P.RNG_SEED, "ladder": list(STEPS),
           "rows": rows}
    p = R.OUT / f"LAUNDERING_CURVE_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {p}")
    print("FIT-DERIVED SYNTHETIC. MECHANISM ONLY. NOT the refusal result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
