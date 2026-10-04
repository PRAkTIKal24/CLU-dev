"""hub/refusal -- the FIT-DERIVED SYNTHETIC MECHANISM PROBE (PREREG AMENDMENT 3).

Answers the MECHANISM question -- does ONE landscape criterion separate BOTH kinds of
unsatisfiable request from satisfiable ones? -- using no held-out row and no C record.

DECLARED AND BINDING (AMENDMENT 3):
  * bases are the first 256 ``fit`` rows of the seed; ``c_invalid.json`` and
    ``a7_redundant.json`` are NOT opened by this script;
  * the threshold is the declared rule, unchanged: quantile(S_max on ``val``, q = 0.95),
    with q the harness's own shipped ``refusal_valid_quantile``;
  * MECHANISM ONLY.  No threshold, criterion, companion or bar may be changed because of
    what this prints.  If any is, the zero-supervision claim is VOID and says so.
  * fit rows are SEEN, so the valid side here is easier than ``B``.  These numbers are
    NOT the benchmark and may never be quoted as the refusal result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from chlu.eval.rivals import gen_knn as knn  # noqa: E402

N_BASES = 256
RNG_SEED = 20260916
Q = 0.95


def _terms(req):
    return [dict(t) for t in req["request"]]


def build_sets(base, grid, rng) -> dict:
    """The four declared constructions, all from the SAME bases and the SAME factor draw."""
    f_of = rng.integers(0, grid.n_factors, size=len(base))
    out = {"syn_valid": [], "syn_a7": [], "syn_dup": []}
    for k in (1, 2, 3):
        out[f"syn_oor_{k}"] = []
    for i, req in enumerate(base):
        f = int(f_of[i])
        vals = np.asarray(grid.values[f], np.float64)
        cur = next(float(t["value"]) for t in req["request"]
                   if int(t["factor_index"]) == f)
        step = float(vals[1] - vals[0]) if vals.size > 1 else 1.0

        out["syn_valid"].append({**req, "id": f"{req['id']}-syn-valid",
                                 "kind": "syn_valid", "request": _terms(req)})

        a7 = _terms(req)
        a7.append({"factor": grid.names[f], "factor_index": f, "value": cur})
        out["syn_a7"].append({**req, "id": f"{req['id']}-syn-a7", "kind": "syn_a7",
                              "request": a7})

        other = float(vals[-1] if abs(cur - float(vals[0])) < abs(cur - float(vals[-1]))
                      else vals[0])
        dup = _terms(req)
        dup.append({"factor": grid.names[f], "factor_index": f, "value": other})
        out["syn_dup"].append({**req, "id": f"{req['id']}-syn-dup", "kind": "syn_dup",
                               "request": dup})

        for k in (1, 2, 3):
            oor = _terms(req)
            for t in oor:
                if int(t["factor_index"]) == f:
                    t["value"] = float(vals[-1] + k * step)
            out[f"syn_oor_{k}"].append({**req, "id": f"{req['id']}-syn-oor{k}",
                                        "kind": f"syn_oor_{k}", "request": oor})
    return out


def au(neg, pos) -> float:
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    return float(binary_auroc(s, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--arms", nargs="+", default=["full", "zero_step", "store_removed"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    grid = ctx["grid"]
    rng = np.random.default_rng(RNG_SEED)
    # AMENDMENT 5: a UNIFORM RANDOM SAMPLE of fit rows, never the head of the sorted
    # array.  ``stratified_fit_val`` returns ``fit = sort(pool \ val)`` and on Shapes3D a
    # row index IS a lexicographic factor tuple, so ``fit_rows[:256]`` is a contiguous
    # CORNER of the factor grid -- the exact defect ``chlu/data/gen_rowsplit.py`` was
    # written to repair.  Measured on the first probe: AUROC(val vs syn_valid) = 0.7221
    # between two VALID sets.
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    base = requests_from_rows(grid, _fit[_take])
    sets = build_sets(base, grid, rng)
    val_reqs = requests_from_rows(grid, ctx["val_rows"])

    report = {"schema": "hub-refusal/synthetic-probe/1",
              "DECLARED": ("FIT-DERIVED SYNTHETIC, PREREG AMENDMENT 3. MECHANISM ONLY. "
                           "NOT the benchmark; fit rows are SEEN so the valid side is "
                           "easier than B; a constructed negative is not a C record."),
              "seed": int(a.seed), "budget": str(a.budget), "q": Q,
              "n_bases": N_BASES, "rng_seed": RNG_SEED, "arms": {}}

    for arm in a.arms:
        R.tlog(f"--- arm {arm} ---")
        S = {}
        vrec = R.settle_unit(ctx, built, "probe_val", val_reqs, arm=arm,
                             budget=str(a.budget), log=lambda *_: None)
        S["val"] = vrec["settled"]["S_max"]
        for name, reqs in sets.items():
            rec = R.settle_unit(ctx, built, f"probe_{name}", reqs, arm=arm,
                                budget=str(a.budget), log=lambda *_: None)
            S[name] = rec["settled"]["S_max"]
        thr = knn.threshold_from_valid(S["val"], Q)
        rr = lambda x: float(knn.refusal_rates(x, thr)["refused_fraction"])  # noqa: E731
        row = {"threshold_from_val": float(thr),
               "refused_fraction": {k: rr(v) for k, v in S.items()},
               "mean_S_max": {k: float(np.mean(v)) for k, v in S.items()},
               "auroc_vs_syn_valid": {k: au(S["syn_valid"], v)
                                      for k, v in S.items() if k != "syn_valid"},
               "auroc_dup_vs_a7_THE_CONTROLLED_ONE": au(S["syn_a7"], S["syn_dup"])}
        report["arms"][arm] = row
        print(f"\n=== arm={arm}  threshold(val,q={Q}) = {thr:.4f} ===")
        print("  %-14s %8s %10s %10s" % ("set", "refused", "mean S_max", "AUROC vs syn_valid"))
        for k in ["val", "syn_valid", "syn_a7", "syn_dup", "syn_oor_1", "syn_oor_2", "syn_oor_3"]:
            auv = row["auroc_vs_syn_valid"].get(k)
            print("  %-14s %8.4f %10.4f %10s" % (
                k, row["refused_fraction"][k], row["mean_S_max"][k],
                ("%.4f" % auv) if auv is not None else "--"))
        print("  A7-CONTROLLED analogue  AUROC(syn_dup vs syn_a7) = %.4f"
              % row["auroc_dup_vs_a7_THE_CONTROLLED_ONE"])

    p = R.OUT / f"SYNTHETIC_PROBE_seed{a.seed}_{a.budget}{a.tag}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {p}")
    print("⛔ FIT-DERIVED SYNTHETIC. MECHANISM ONLY. NOT the refusal result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
