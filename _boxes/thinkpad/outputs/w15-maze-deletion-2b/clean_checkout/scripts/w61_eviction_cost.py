"""w6.1 retry -- THE BRIDGE: what would evicting ONE atom cost this read?

The mechanism this retry establishes is a resolution law: the read's answers form a
lattice whose one-factor spacing is rho_f, the judge's decision cells are that lattice's
Voronoi cells (it flips at s ~ 0.5 in every factor), and therefore a read like this
tolerates displacement only up to about half a lattice step.

That law is not about dynamics.  It is about **motion**, whatever causes it.  The
programme's central open problem is that the object which forgets exactly does so by
**evicting atoms**, and evicting an atom moves the state too.  So this script prices
that motion in the same units:

  for each well k and each atom j, rebuild the span WITHOUT atom j, reproject every
  scored request's answer into it, and measure how far the answer had to move --
  against rho_1 / 2, the distance at which the judge starts reading back a different
  tuple.

⛔ Re-analysis only: the content, the address book and the requests are the parent's,
no judge is constructed and nothing is scored.

    python scripts/w61_eviction_cost.py --leg DIR --seed S --rho1 R --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w6_compose_physics as P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="test")
    ap.add_argument("--rho1", type=float, required=True,
                    help="the measured median one-factor answer spacing for this seed")
    ap.add_argument("--budget", type=float, default=None,
                    help="the motion budget in the same absolute units: the "
                         "displacement at which the loss reaches the pre-registered "
                         "+-1.0 novel tuples per corner")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()

    P.install()
    R.set_condition(a.seed, a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    corners = (fw.val_corners(cond) if a.condition == "val"
               else fw.nway_corners(cond))
    with open(Path(a.leg) / "arms" / "pca_tensor" / "arm.pkl", "rb") as fh:
        obj = pickle.load(fh)
    store = obj["store"]

    X = []
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        f, _ = R.arm_features(obj, vals, grid)
        X.append(np.asarray(f, np.float64))
    X = np.concatenate(X, 0)
    n = X.shape[0]
    print(f"{n} requests; rho_1 = {a.rho1:.4f}, half = {a.rho1 / 2:.4f}", flush=True)

    half = 0.5 * float(a.rho1)
    bud = float(a.budget) if a.budget is not None else half
    out = {"leg": str(a.leg), "seed": a.seed, "condition": a.condition,
           "n_requests": int(n), "rho1": float(a.rho1), "half_rho1": half,
           "motion_budget": bud,
           "median_state_norm": float(np.median(np.linalg.norm(X, axis=1))),
           "wells": []}
    allmed = []
    for k, (idx, G) in enumerate(zip(store.well_channels, store.modes)):
        idx = np.asarray(idx, np.int64)
        A = np.asarray(G, np.float64)              # (n_atoms, d_k)
        xb = X[:, idx]
        med, frac = [], []
        for j in range(A.shape[0]):
            keep = np.delete(A, j, axis=0)
            U, s, _ = np.linalg.svd(keep.T, full_matrices=False)
            r = int((s > float(s[0]) * 1e-8).sum())
            U = U[:, :r]
            resid = xb - (xb @ U) @ U.T            # what the reduced span cannot hold
            d = np.linalg.norm(resid, axis=1)
            med.append(float(np.median(d)))
            frac.append(float((d > half).mean()))
        med = np.asarray(med)
        frac = np.asarray(frac)
        allmed.append(med)
        out["wells"].append({
            "well": k, "n_atoms": int(A.shape[0]), "d_k": int(idx.size),
            "median_displacement_per_evicted_atom_min": float(med.min()),
            "median_displacement_per_evicted_atom_median": float(np.median(med)),
            "median_displacement_per_evicted_atom_max": float(med.max()),
            "atoms_whose_eviction_moves_the_median_request_past_half_rho1":
                int((med > half).sum()),
            "mean_frac_of_requests_moved_past_half_rho1": float(frac.mean()),
            "atom_0_is_the_mean_median_displacement": float(med[0]),
            "atoms_past_the_motion_budget": int((med > bud).sum()),
        })
        print(f"  well {k}: {int((med > half).sum())}/{A.shape[0]} atoms are individually "
              f"load-bearing past rho_1/2  ({time.time() - t0:.0f}s)", flush=True)
    M = np.concatenate(allmed)
    out["all_atoms"] = {
        "n_atoms_total": int(M.size),
        "atoms_past_half_rho1": int((M > half).sum()),
        "frac_past_half_rho1": float((M > half).mean()),
        "median_over_atoms": float(np.median(M)),
        "median_in_half_rho1_units": float(np.median(M) / half),
        "atoms_past_the_motion_budget": int((M > bud).sum()),
        "frac_past_the_motion_budget": float((M > bud).mean()),
        "atoms_in_the_wells_whose_atoms_are_independent": int(sum(
            int(x.size) for x in allmed[:3])),   # wells 0-2: 64 atoms, rank 64
        "atoms_past_budget_among_the_independent_ones": int(
            (np.concatenate(allmed[:3]) > bud).sum()),
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["all_atoms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
