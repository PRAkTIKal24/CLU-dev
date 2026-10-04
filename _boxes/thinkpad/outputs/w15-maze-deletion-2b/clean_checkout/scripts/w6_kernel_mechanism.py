#!/usr/bin/env python
"""w6 hub/potentials -- WIDTH AND SEPARATION, as a property of the KERNEL.

⛔ MECHANISM ONLY. Renders no verdict, scores no row, contests no rival.
w6-common section 3.7: row 2 belongs to a size-matched continuous MLP on every seed under
two implementations, and this hub does not re-contest it. The task file's own scope limit
says to measure the floor as a property of the kernel and report it as such -- that is all
this script does.

⛔ AND: the shorts contain NO Mexican-hat width law and NO interpolation-floor derivation
(inventory section 1.1). Nothing here is attributed to them. The closed form below is
derived from the energy THIS REPOSITORY already uses:

    E(x) = -sigma^2 logsumexp_j [log life_j - ||x - a_j||^2 / (2 sigma^2)]
    Hess E = I - Cov_w(a)/sigma^2

from which, for two equal-weight atoms at separation Delta, the curvature along the
separation axis is exactly 1 - Delta^2/(4 sigma^2): ONE merged minimum below
Delta/sigma = 2, TWO minima above. The field store PRINTS basin_gap/well_sigma at build
time ("THE NUMBER REFUSAL READS"), so the prediction can be read off the store and then
checked against the landscape's actual minima.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def count_minima(A, sigma, life, n_starts=512, seed=0, tol=1e-8, merge=1e-4):
    """Distinct mean-shift fixed points of E, from starts drawn on the atoms' own hull.

    Mean shift is the exact zero-gradient iteration for this energy, so a fixed point IS
    a critical point of V. Starts are convex combinations of the atoms (Dirichlet), which
    covers the region the read actually visits.
    """
    from chlu.core.w6_potentials import mean_shift_settle
    rng = np.random.default_rng(seed)
    A = np.asarray(A, np.float64)
    n = A.shape[0]
    W = rng.dirichlet(np.ones(n) * 0.35, size=int(n_starts))
    starts = W @ A
    fps = []
    counts = []
    for s in starts:
        x = mean_shift_settle(s, A, sigma, life, max_iter=2000, tol=tol)["x"]
        hit = -1
        for i, f in enumerate(fps):
            if np.sqrt(np.sum((x - f) ** 2)) <= merge * max(1.0, np.sqrt(np.sum(f * f))):
                hit = i
                break
        if hit < 0:
            fps.append(x)
            counts.append(1)
        else:
            counts[hit] += 1
    return {"n_minima": len(fps), "basin_sizes": sorted(counts, reverse=True),
            "n_starts": int(n_starts)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, help="a leg dir holding field/field.pkl")
    ap.add_argument("--out", default="")
    ap.add_argument("--starts", type=int, default=256)
    a = ap.parse_args(argv)

    import pickle
    from chlu.core import w6_potentials as W6

    with open(Path(a.leg) / "field" / "field.pkl", "rb") as fh:
        field = pickle.load(fh)["field"]

    gap = np.asarray(field.view("basin_gap"), np.float64)
    bsig = np.asarray(field.view("basin_sigma"), np.float64)
    wsig = np.asarray(field.view("well_sigma"), np.float64)
    bn = np.asarray(field.view("basin_n"), np.int64)

    out = {"field_sha": field.sha256(), "wells": [],
           "law": "Hess E = I - Cov_w/sigma^2; two atoms at separation D merge below "
                  "D/sigma = 2 and split above. DERIVED HERE from gen_landscape."
                  "store_energy, NOT from the shorts (inventory section 1.1).",
           "scope": "MECHANISM ONLY -- row 2 belongs to a size-matched continuous MLP."}

    print(f"field {field.sha256()[:16]}   K={field.K} A={field.A} D={field.D}")
    print("\nwell   d   well_sigma   median gap/sigma   predicted   measured   n_basins")
    print("                                             minima      minima     w/ atoms")
    for k in range(field.K):
        v = W6.well_view(field, k)
        A, sig, life = v["atoms"], v["sigma"], v["life"]
        # ⚠ The store's own printed number, and it is PER WELL: only the terms of the
        # factors WELL k HOSTS have a gap in well k's channels.  Pooling every term's
        # gap against every well's sigma (my first version did) mixes factors that live
        # in other wells and produces a number that means nothing.
        toff = np.asarray(field.meta["term_offsets"], np.int64)
        hosts = []
        for fct in field.meta["host_factors"][k]:
            hosts += list(range(int(toff[fct]), int(toff[fct + 1])))
        gs = np.array([gap[t] / sig for t in hosts
                       if bn[t] > 0 and np.isfinite(gap[t]) and gap[t] > 0])
        med = float(np.median(gs)) if gs.size else float("nan")
        pred_split = bool(med > 2.0)
        cm = count_minima(A, sig, life, n_starts=a.starts, seed=k)
        pr = W6.sigma_provenance(A, sig)
        # ⭐ CHARTER SECTION C.4: "any arm whose settled state sits on its nearest atom
        # (distance -> 0) is a lookup table and is reported as one."  Measure it.
        st = W6.mean_shift_settle(A.mean(axis=0), A, sig, life, max_iter=2000, tol=1e-10)
        dmin = float(np.sqrt(np.min(np.sum((A - st["x"][None, :]) ** 2, axis=1))))
        wts = W6.kernel_weights(st["x"], A, sig, life)
        # participation ratio: how many atoms the settled read actually reads from
        pr_eff = float(1.0 / np.sum(wts ** 2))
        # what width WOULD split the nearest basins of this well
        sigma_to_split = float(np.nanmin(gs) * sig / 2.0) if gs.size else float("nan")
        row = {"well": k, "d": int(A.shape[1]), "well_sigma": sig,
               "median_gap_over_sigma": med,
               "predicted_multiple_minima": pred_split,
               "measured_n_minima": cm["n_minima"],
               "basin_sizes": cm["basin_sizes"][:6],
               "sigma_that_would_split": sigma_to_split,
               "dist_min_to_nearest_atom": dmin,
               "dist_over_sigma": dmin / sig,
               "atoms_effectively_read": pr_eff,
               "sigma_provenance": pr}
        out["wells"].append(row)
        row["n_terms_hosted"] = int(gs.size)
        print(f"{k:3d} {A.shape[1]:5d} {sig:11.4f} {med:17.4f} "
              f"{('SPLIT' if pred_split else 'MERGED'):>11} "
              f"{cm['n_minima']:9d} {int(gs.size):10d}")

    meds = [r["median_gap_over_sigma"] for r in out["wells"]]
    print(f"\nALL FIVE WELLS have median gap/sigma in "
          f"[{np.nanmin(meds):.3f}, {np.nanmax(meds):.3f}] -- "
          f"{'all BELOW' if np.nanmax(meds) < 2.0 else 'NOT all below'} the "
          f"Delta/sigma = 2 bifurcation.")
    print("Interpretation (mechanism, not a row): below 2 the stored values do not carry "
          "separate minima,\nso the landscape interpolates between them rather than "
          "snapping to them. The width would have to\nshrink to the "
          "'sigma_that_would_split' column for a value to own a basin of its own.")
    print("\nknn_density trap: well_sigma / nn_median = " + ", ".join(
        f"{r['sigma_provenance']['well_sigma_over_nn_median']:.3f}" for r in out["wells"]))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
