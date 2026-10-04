"""hub/refusal -- PLUMBING SMOKE, on FIT-derived synthetic terms only.

WHAT THIS IS FOR, and what it deliberately is NOT.
The term-expert path has two branches a valid request never exercises: an off-grid value
(``_extrapolating_index`` with ``|w| > 0``) and a 7th term slot.  If either crashes or
produces a non-finite number, I want to find that out on rows I am allowed to touch, not
half way through the scored run.

DECLARED, AND BINDING ON ME:
  * the requests here are built from **fit** rows -- the training half of store_train.
    They are NOT the C set, NOT A7, NOT val, NOT A, NOT B.  ``c_invalid.json`` and
    ``a7_redundant.json`` are not opened by this script.
  * this script asserts SHAPE, FINITENESS and TERM COUNT and prints the sign convention
    on the VALID terms only.  It deliberately does NOT print, and does not compute, any
    separation, AUROC, threshold or ranking between the synthetic perturbed requests and
    the valid ones -- there is nothing here that could be used to choose or adjust
    anything, and that is the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402


def perturb_offgrid(req: dict, grid, factor_index: int = 0, steps_out: float = 3.0) -> dict:
    """One term moved OFF the grid, below the first value by ``steps_out`` grid steps."""
    out = {**req, "id": str(req["id"]) + "-smoke-oor", "kind": "smoke_offgrid",
           "request": [dict(t) for t in req["request"]]}
    vals = np.asarray(grid.values[int(factor_index)], np.float64)
    step = float(vals[1] - vals[0]) if vals.size > 1 else 1.0
    for t in out["request"]:
        if int(t["factor_index"]) == int(factor_index):
            t["value"] = float(vals[0] - steps_out * step)
    return out


def perturb_duplicate(req: dict, grid, factor_index: int = 0) -> dict:
    """A 7th term: the same factor named again, at the grid's OTHER end."""
    out = {**req, "id": str(req["id"]) + "-smoke-dup", "kind": "smoke_duplicate",
           "request": [dict(t) for t in req["request"]]}
    vals = np.asarray(grid.values[int(factor_index)], np.float64)
    cur = next(float(t["value"]) for t in out["request"]
               if int(t["factor_index"]) == int(factor_index))
    other = float(vals[-1] if abs(cur - float(vals[0])) < abs(cur - float(vals[-1]))
                  else vals[0])
    out["request"].append({"factor": grid.names[int(factor_index)],
                           "factor_index": int(factor_index), "value": other})
    return out


def main() -> int:
    from chlu.core.gen_request import requests_from_rows

    ctx = R.build_context(0)
    built = R.build_land(ctx, "A")
    grid = ctx["grid"]
    fit8 = np.asarray(ctx["fit_rows"], np.int64)[:8]
    base = requests_from_rows(grid, fit8)

    cases = {
        "fit_valid": base,
        "fit_offgrid": [perturb_offgrid(r, grid, 0, 3.0) for r in base],
        "fit_duplicate": [perturb_duplicate(r, grid, 0) for r in base],
    }
    ok = True
    for name, reqs in cases.items():
        log_w, n_terms, tinfo = R.term_experts(built["book"], grid, reqs, R.T_MAX)
        finite_w = bool(np.isfinite(log_w).all())
        rec = R.settle_unit(ctx, built, f"smoke_{name}", reqs, arm="full", budget="A",
                            steps=50, log=lambda *_: None)
        S = rec["settled"]["S_max"]
        res = rec["settled"]["residuals"]
        finite_s = bool(np.isfinite(S).all())
        n_active = int(np.isfinite(res).sum(axis=1).max())
        print(f"[{name:14s}] n={len(reqs):3d} n_terms={sorted(set(n_terms.tolist()))} "
              f"offgrid_terms={tinfo['n_offgrid_terms']:2d} "
              f"log_w finite={finite_w} S_max finite={finite_s} "
              f"active term slots max={n_active}")
        ok &= finite_w and finite_s
        if name == "fit_valid":
            r6 = res[:, :6]
            print(f"                 SIGN CONVENTION on VALID fit terms: "
                  f"frac(r_t < 0) = {float((r6 < 0).mean()):.4f}  "
                  f"(satisfied terms must be negative)")
    # the branch coverage assertions -- the only thing this script is allowed to judge
    assert sorted(set(R.term_experts(built["book"], grid, cases["fit_duplicate"],
                                     R.T_MAX)[1].tolist())) == [7], "7th term slot unused"
    assert R.term_experts(built["book"], grid, cases["fit_offgrid"],
                          R.T_MAX)[2]["n_offgrid_terms"] == 8, "extrapolation not exercised"
    assert R.term_experts(built["book"], grid, cases["fit_valid"],
                          R.T_MAX)[2]["n_offgrid_terms"] == 0, "valid must never extrapolate"
    print("\nPLUMBING OK" if ok else "\nPLUMBING FAILED")
    print("No separation, AUROC, threshold or ranking was computed here -- by design.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
