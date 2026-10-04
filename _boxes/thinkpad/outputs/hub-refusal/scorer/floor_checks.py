"""hub/refusal -- charter section C.4 and the settled-state geometry, from units on disk.

Section C.4: "Any arm whose settled state sits on its nearest atom (distance -> 0) is a
lookup table and is reported as one."  Computed on VALID sets only (val, MS) -- no
held-out row, no C record.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import refusal_lib as R

def nearest_atom_stats(store, X):
    X = np.asarray(X, np.float32)
    d2 = np.zeros((X.shape[0],), np.float64)
    ref = 0.0
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        z = X[:, cols].astype(np.float64)
        A = np.asarray(store.atoms[k], np.float64)
        dd = ((z * z).sum(1)[:, None] - 2.0 * (z @ A.T) + (A * A).sum(1)[None, :])
        d2 += np.maximum(dd, 0.0).min(axis=1)
        ref += float((A.var(axis=0).sum()))
    return {"rms_distance_to_nearest_atom": float(np.sqrt(d2.mean())),
            "median_distance": float(np.median(np.sqrt(d2))),
            "min_distance": float(np.sqrt(d2).min()),
            "atom_rms_spread": float(np.sqrt(ref)),
            "ratio_distance_over_spread": float(np.sqrt(d2.mean()) / max(np.sqrt(ref), 1e-12))}

def main() -> int:
    seed, budget = 0, "A"
    ctx = R.build_context(seed)
    built = R.build_land(ctx, budget)
    store = built["store"]
    out = {"schema": "hub-refusal/floor-checks/1", "seed": seed, "budget": budget,
           "C4_rule": ("charter C.4: a settled state ON its nearest atom (distance -> 0) "
                       "is a lookup table and is reported as one"), "sets": {}}
    for s in ("val", "MS"):
        for arm in ("full", "zero_step"):
            p = R.OUT / "runs" / f"seed_{seed}" / f"{budget}_{s}_{arm}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            st = nearest_atom_stats(store, z["x_settled"])
            st["n"] = int(z["x_settled"].shape[0])
            out["sets"][f"{s}/{arm}"] = st
            print("%-16s n=%4d  rms d(settled, nearest atom)=%9.4f  median=%9.4f  "
                  "min=%9.4f  atom rms spread=%9.4f  ratio=%.4f"
                  % (f"{s}/{arm}", st["n"], st["rms_distance_to_nearest_atom"],
                     st["median_distance"], st["min_distance"], st["atom_rms_spread"],
                     st["ratio_distance_over_spread"]))
    p = R.OUT / "FLOOR_CHECKS.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
