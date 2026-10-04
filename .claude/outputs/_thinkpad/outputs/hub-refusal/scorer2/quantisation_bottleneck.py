"""hub/refusal-2 -- WHICH LINK loses the factor: the features, the ATOM QUANTISATION, or the fit?

MECHANISM ONLY.  fit rows and their labels.  No val scoring, no C record, no
a7_redundant.json, no held-out row.  NOT scored, NOT adopted.

WHY THIS EXISTS -- it tests a claim of mine that a guard measurement already half-refuted.
``addressability_locus`` found the geometric factors 9-38x less LINEARLY separable (Fisher)
than the hues, and I wrote that they were "not in the representation".
``feature_decodability`` then showed they ARE: 1-NN on a well's own features recovers shape at
0.8307 against a 0.2500 chance rate and orientation at 0.4393 against 0.0667.  So the
information is present in the features and absent from the address book, and the loss happens
IN BETWEEN.

THE CHAIN, and this script measures the SAME quantity at each link, in nats:

    features  --(nearest-atom quantisation)-->  atom index  --(soft fit)-->  log P(atom | f=v)

  * ``I(factor ; ATOM INDEX)`` -- a HARD nearest-atom quantisation of the well's features.
    This is the channel capacity the address book is built on top of, and it involves no
    bandwidth and no smoothing at all.  Estimated as the plug-in mutual information of the
    empirical joint on fit rows, with its own shuffled-label floor reported beside it because
    a plug-in MI over N atoms is biased upward.
  * ``I(factor ; SOFT assignment)`` at the shipped bandwidth and at the tightened ones -- the
    same quantity the shipped book actually sees, from ``soft_atom_assignment``.

THE DISCRIMINATION.
  * If ``I(factor ; atom index)`` is high for shape and the shipped soft version is ~0, the
    loss is the BANDWIDTH and it is cheap to fix.
  * If ``I(factor ; atom index)`` is itself ~0 for shape while 1-NN on the raw features
    recovers shape at 0.83, then the ATOM QUANTISATION is the bottleneck: distance in the
    well's feature space is dominated by the hue directions, so the atom index is
    approximately a hue label and no book fitted on it can address shape at any bandwidth.
    ⭐ That is a store-construction finding, not a read finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402
from chlu.core.gen_landscape import soft_atom_assignment  # noqa: E402

RHOS = (1.0, 0.5, 0.25, 0.125)


def mi_from_joint(J: np.ndarray) -> float:
    """Plug-in mutual information in nats from an unnormalised joint count/mass matrix."""
    J = np.asarray(J, np.float64)
    tot = J.sum()
    if tot <= 0:
        return 0.0
    P = J / tot
    px = P.sum(1, keepdims=True)
    py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = P * (np.log(np.maximum(P, 1e-300)) - np.log(np.maximum(px * py, 1e-300)))
    return float(np.nansum(np.where(P > 0, t, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--n", type=int, default=8192)
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    land = built["land"]
    store = land.store
    sigma = np.asarray(land.sigma, np.float64)
    grid = ctx["grid"]
    names = list(grid.names)
    cards = [int(c) for c in grid.cardinalities]
    N, K = int(store.n_atoms), int(store.n_wells)

    X = np.asarray(ctx["X_store"], np.float32)
    X = X.reshape((X.shape[0], -1)).astype(np.float64)
    Y = np.asarray(ctx["lab_store"], np.int64)
    rng = np.random.default_rng(20260917)
    n = min(int(a.n), X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X, Y = X[idx], Y[idx]

    rep = {"schema": "hub-refusal-2/quantisation-bottleneck/1",
           "DECLARED": ("MECHANISM ONLY, fit rows only. NOT scored, NOT adopted. Tests a "
                        "claim of this hub's that a guard measurement half-refuted."),
           "seed": int(a.seed), "budget": str(a.budget), "n_rows": n,
           "n_atoms": N, "n_wells": K, "log_N_nats": float(np.log(N)),
           "names": names, "cardinalities": cards}

    # ---- the HARD nearest-atom quantisation ---------------------------------
    hard = np.zeros((n, K), np.int64)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        Z = X[:, cols]
        A = np.asarray(store.atoms[k], np.float64)
        d = (Z * Z).sum(1)[:, None] - 2.0 * (Z @ A.T) + (A * A).sum(1)[None, :]
        hard[:, k] = d.argmin(1)

    yperm = np.stack([rng.permutation(Y[:, f]) for f in range(len(names))], 1)
    I_hard = np.zeros((K, len(names)))
    I_floor = np.zeros((K, len(names)))
    occ = []
    for k in range(K):
        occ.append(int(np.unique(hard[:, k]).size))
        for f in range(len(names)):
            J = np.zeros((N, cards[f]), np.float64)
            np.add.at(J, (hard[:, k], Y[:, f]), 1.0)
            I_hard[k, f] = mi_from_joint(J)
            Jp = np.zeros((N, cards[f]), np.float64)
            np.add.at(Jp, (hard[:, k], yperm[:, f]), 1.0)
            I_floor[k, f] = mi_from_joint(Jp)
    rep["hard_quantisation"] = {
        "atoms_occupied_per_well": occ, "n_atoms": N,
        "I_atom_index_vs_factor_nats": I_hard.round(4).tolist(),
        "SHUFFLED_LABEL_FLOOR_nats": I_floor.round(4).tolist(),
        "excess_over_floor": (I_hard - I_floor).round(4).tolist(),
        "max_over_wells_excess": dict(zip(names, (I_hard - I_floor).max(0).round(4).tolist())),
        "note": ("a plug-in MI over N atoms is biased upward, so the shuffled-label floor is "
                 "reported beside it and the EXCESS is the honest figure."),
    }
    R.tlog("  hard-quantisation excess MI, max over wells: "
           + " ".join(f"{nm}={v:.4f}" for nm, v in
                      zip(names, (I_hard - I_floor).max(0))))

    # ---- the SOFT assignment, at the shipped and tightened bandwidths -------
    soft = {}
    for rho in RHOS:
        pis = soft_atom_assignment(store, X.astype(np.float32), sigma * float(rho))
        I_s = np.zeros((K, len(names)))
        I_f = np.zeros((K, len(names)))
        eff = []
        for k in range(K):
            p = np.asarray(pis[k], np.float64)
            H = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
            eff.append(float(np.exp(H).mean()))
            for f in range(len(names)):
                J = np.zeros((N, cards[f]), np.float64)
                for v in range(cards[f]):
                    sel = Y[:, f] == v
                    if sel.any():
                        J[:, v] = p[sel].sum(0)
                I_s[k, f] = mi_from_joint(J)
                Jp = np.zeros((N, cards[f]), np.float64)
                for v in range(cards[f]):
                    sel = yperm[:, f] == v
                    if sel.any():
                        Jp[:, v] = p[sel].sum(0)
                I_f[k, f] = mi_from_joint(Jp)
        soft[str(rho)] = {
            "rho": float(rho), "is_shipped": bool(rho == 1.0),
            "effective_atoms_per_well": [round(x, 2) for x in eff],
            "I_soft_vs_factor_nats": I_s.round(4).tolist(),
            "SHUFFLED_LABEL_FLOOR_nats": I_f.round(4).tolist(),
            "max_over_wells_excess": dict(zip(names, (I_s - I_f).max(0).round(4).tolist())),
        }
        R.tlog(f"  soft rho={rho}: excess MI max over wells "
               + " ".join(f"{nm}={v:.4f}" for nm, v in zip(names, (I_s - I_f).max(0))))
    rep["soft_assignment_by_bandwidth"] = soft

    p = R.OUT / f"QUANTISATION_BOTTLENECK_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n%-13s %14s %14s %14s %14s %14s" % (
        "factor", "HARD atom idx", "soft rho=1", "rho=0.5", "rho=0.25", "rho=0.125"))
    for f, nm in enumerate(names):
        print("%-13s %14.4f %14.4f %14.4f %14.4f %14.4f" % (
            nm, (I_hard - I_floor).max(0)[f],
            *[soft[str(r)]["max_over_wells_excess"][nm] for r in RHOS]))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
