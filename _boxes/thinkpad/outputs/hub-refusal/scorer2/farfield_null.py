"""hub/refusal-2 -- WHY the section C.2 control cannot collapse: the FAR-FIELD theorem, measured.

MECHANISM ONLY.  Fit-derived synthetic analogue + the val split.  No C record, no
a7_redundant.json, no held-out row.  Nothing here is scored and nothing here is adopted:
the go/no-go already returned NO-GO and the criterion is frozen.

THE CLAIM UNDER TEST (logged in LOG.md BEFORE this script was written).
``store_energy(land, x) = -sum_k sigma_k^2 * logsumexp_a(-||x_k - a_k||^2 / 2 sigma_k^2)``.
Write ``u_a = -||x_k - a_k||^2 / 2 sigma_k^2``.  For a point in the FAR FIELD -- one whose
distance to the cloud is large compared with the cloud's own spread, which the go/no-go
measured at ``d / spread ~= 21`` for EVERY request kind -- ``logsumexp`` is dominated by the
whole cloud rather than by one atom, and

    logsumexp_a(u_a) = log N + mean(u) + 0.5 * Var(u) + O(third cumulant)
    =>  E_store(x) = sum_k [ 0.5*||x_k - abar_k||^2 + 0.5*tr(Cov_k) - sigma_k^2 log N ]
                     - sum_k 0.5 * sigma_k^2 * Var_a(u_a)  + ...

⭐ The charter section C.2 control permutes each CHANNEL COLUMN independently among the N
atoms.  A column permutation preserves every column's MEAN and every column's VARIANCE
EXACTLY.  So it preserves ``abar_k`` exactly and ``tr(Cov_k)`` exactly.  ⛔ Therefore the
ENTIRE leading behaviour of ``E_store`` in the far field is INVARIANT under the control, and
no criterion of the form ``E_store(linear read)`` can collapse under it -- on any store, at
any bandwidth.  That is a theorem if the far-field expansion holds; this script measures
whether it holds, and by how much.

THREE MEASUREMENTS:
  1. FAR-FIELD FIT.  R^2 / Pearson / Spearman of ``E_store(x_read)`` against the surrogate
     ``Q(x) = sum_k 0.5*||x_k - abar_k||^2``, which contains NO atom identity at all --
     only the centroid.
  2. THE INVARIANT / RESIDUAL SPLIT.  ``E_store = Q + (E_store - Q)``.  The residual is the
     only part the atoms' JOINT structure can reach.  Its AUROC is the honest upper bound on
     what the store's CONTENT contributes to refusal through this read.
  3. THE PERMUTATION NULL.  ``R_SHUF`` independent section C.2 shuffles, each a legitimate
     instance of the shipped control.  The full-store AUROC is reported against that null
     distribution with a two-sided rank p-value.  ⭐ A criterion whose full-store value sits
     INSIDE its own landscape-removed null is decorative, and this measures it rather than
     asserting it.
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
from chlu.config import gen_dynamical_read_config  # noqa: E402
from chlu.core.gen_landscape import AtomLandscape, store_energy  # noqa: E402
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

OOR_SETS = ("syn_oor_1", "syn_oor_2", "syn_oor_3")
R_SHUF = 16


def au(neg, pos) -> float:
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    return float(binary_auroc(s, y))


def pair(x: dict) -> tuple:
    """(AUROC_oor pooled vs syn_valid, AUROC_dup vs syn_a7) -- the two PREREG numbers."""
    oor = np.concatenate([x[s] for s in OOR_SETS])
    return au(x["syn_valid"], oor), au(x["syn_a7"], x["syn_dup"])


def centroid_surrogate(store, X) -> np.ndarray:
    """``Q(x) = sum_k 0.5 * ||x_k - abar_k||^2`` -- the centroid, and NO atom identity."""
    X = np.asarray(X, np.float64)
    q = np.zeros(X.shape[0], np.float64)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        abar = np.asarray(store.atoms[k], np.float64).mean(axis=0)
        d = X[:, cols] - abar[None, :]
        q += 0.5 * (d * d).sum(axis=1)
    return q


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(np.asarray(a, np.float64))).astype(np.float64)
    rb = np.argsort(np.argsort(np.asarray(b, np.float64))).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    land0, book, grid = built["land"], built["book"], ctx["grid"]
    sig_ref = float(gen_dynamical_read_config(ctx["config"]).sigma_scale_end)

    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    base = requests_from_rows(grid, _fit[_take])
    sets = build_sets(base, grid, rng)
    sets = {"val": requests_from_rows(grid, ctx["val_rows"]), **sets}

    # ⭐ The posterior is computed ONCE and is arm-independent BY CONSTRUCTION: the address
    #   book is built from atom LABELS, never from atom content, so every section C.2 shuffle
    #   below leaves P bit-for-bit identical.  That is the whole reason this null is the
    #   right one -- the ONLY thing that varies across realisations is the atom geometry.
    P = {}
    for name, reqs in sets.items():
        log_w, n_terms, _ = R.term_experts(book, grid, reqs, R.T_MAX)
        P[name] = L.term_posterior(book, log_w, n_terms)
    R.tlog(f"  posteriors built for {len(P)} sets")

    def realisation(store):
        S, Q, D = {}, {}, {}
        for name in sets:
            x = L.linear_read(store, P[name])
            S[name] = np.asarray(store_energy(
                AtomLandscape(store, np.asarray(land0.sigma), land0.coupling,
                              float(land0.request_strength),
                              float(land0.coupling_strength)), x, sig_ref), np.float64)
            Q[name] = centroid_surrogate(store, x)
            D[name] = L.nearest_atom_distance(store, x)["d_over_spread"]
        return S, Q, D

    S0, Q0, D0 = realisation(land0.store)
    allS = np.concatenate([S0[k] for k in sets])
    allQ = np.concatenate([Q0[k] for k in sets])

    # -- 1. the far-field fit -------------------------------------------------
    c = np.corrcoef(allS, allQ)[0, 1]
    A = np.stack([allQ, np.ones_like(allQ)], 1)
    beta, *_ = np.linalg.lstsq(A, allS, rcond=None)
    resid_all = allS - A @ beta
    r2 = float(1.0 - resid_all.var() / allS.var())
    farfield = {
        "pearson_r": float(c), "r2_linear_fit": r2, "spearman": spearman(allS, allQ),
        "slope": float(beta[0]), "intercept": float(beta[1]),
        "note": ("Q contains the atom CENTROID and nothing else. A slope near 1 and an R^2 "
                 "near 1 say E_store(x_read) IS a centroid distance -- and a column "
                 "permutation preserves the centroid EXACTLY."),
        "mean_d_over_spread": {k: float(np.mean(v)) for k, v in D0.items()},
        "residual_std_over_S_std": float(resid_all.std() / allS.std()),
    }

    # -- 2. the invariant / residual split ------------------------------------
    off = {k: (S0[k] - (beta[0] * Q0[k] + beta[1])) for k in sets}
    s_pair, q_pair, r_pair = pair(S0), pair(Q0), pair(off)
    split = {
        "AUROC_pair_E_store_FULL": list(s_pair),
        "AUROC_pair_Q_centroid_only": list(q_pair),
        "AUROC_pair_RESIDUAL_the_only_part_atom_JOINT_structure_can_reach": list(r_pair),
        "note": ("(oor pooled vs syn_valid, dup vs syn_a7). If Q alone reproduces E_store's "
                 "pair, the criterion carries no atom identity; if the residual is at "
                 "chance, the atoms' joint structure contributes nothing to refusal."),
    }

    # -- 3. the permutation null ----------------------------------------------
    null = []
    for i in range(R_SHUF):
        st = R.shuffled_store(land0.store, int(a.seed) + 1000 * (i + 1))
        Si, Qi, Di = realisation(st)
        null.append({"shuffle_seed": int(a.seed) + 1000 * (i + 1),
                     "AUROC_pair": list(pair(Si)),
                     "mean_d_over_spread_syn_valid": float(np.mean(Di["syn_valid"])),
                     "pearson_r_S_vs_Q": float(np.corrcoef(
                         np.concatenate([Si[k] for k in sets]),
                         np.concatenate([Qi[k] for k in sets]))[0, 1])})
        R.tlog(f"  shuffle {i + 1}/{R_SHUF}: pair = "
               f"({null[-1]['AUROC_pair'][0]:.4f}, {null[-1]['AUROC_pair'][1]:.4f})")
    Noor = np.asarray([n["AUROC_pair"][0] for n in null])
    Ndup = np.asarray([n["AUROC_pair"][1] for n in null])

    def pval(obs, nul):
        return float((np.abs(nul - 0.5) >= abs(obs - 0.5)).mean())

    perm = {
        "R_shuffles": R_SHUF,
        "full_store_pair": list(s_pair),
        "null_oor": {"mean": float(Noor.mean()), "std": float(Noor.std(ddof=1)),
                     "min": float(Noor.min()), "max": float(Noor.max())},
        "null_dup": {"mean": float(Ndup.mean()), "std": float(Ndup.std(ddof=1)),
                     "min": float(Ndup.min()), "max": float(Ndup.max())},
        "full_inside_null_oor": bool(Noor.min() <= s_pair[0] <= Noor.max()),
        "full_inside_null_dup": bool(Ndup.min() <= s_pair[1] <= Ndup.max()),
        "two_sided_rank_p_oor": pval(s_pair[0], Noor),
        "two_sided_rank_p_dup": pval(s_pair[1], Ndup),
        "note": ("Each shuffle is a legitimate instance of the SHIPPED charter C.2 control "
                 "(refusal_lib.shuffled_store) at a different seed. A full-store value "
                 "inside its own landscape-removed null is decorative, measured not asserted."),
        "realisations": null,
    }

    rep = {"schema": "hub-refusal-2/farfield-null/1",
           "DECLARED": ("MECHANISM ONLY, fit-derived synthetic + val. NOT scored, NOT "
                        "adopted, NOT the benchmark. The go/no-go already returned NO-GO "
                        "and the criterion is frozen; this explains WHY."),
           "seed": int(a.seed), "budget": str(a.budget),
           "far_field_fit": farfield, "invariant_residual_split": split,
           "permutation_null": perm}
    p = R.OUT / f"FARFIELD_NULL_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n" + json.dumps({k: v for k, v in rep.items()
                             if k in ("far_field_fit", "invariant_residual_split")}, indent=1))
    print(json.dumps({k: v for k, v in perm.items() if k != "realisations"}, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
