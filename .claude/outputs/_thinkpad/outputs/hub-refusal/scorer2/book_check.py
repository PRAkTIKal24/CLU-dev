"""hub/refusal-2 -- two verifications of the diagnosis, both of them attempts to BREAK it.

MECHANISM ONLY.  val + fit rows.  No C record, no a7_redundant.json, no held-out row.

CHECK 1 -- IS THE ADDRESSABILITY DEFICIT MINE OR THE PIN'S?
  ``posterior_anatomy`` measured, at rung A, that a fully-specified VALID 6-factor request
  narrows 64 stored atoms to ``exp(H) = 42.6`` -- 11.3 % of ``log N`` -- going through MY
  ``term_experts`` path.  If that is an artefact of my code the whole diagnosis is worthless.
  So it is recomputed here straight out of the SHIPPED ``gen_landscape.atom_posterior`` on
  dense class indices, which is the path every pinned arm uses, and the two are compared.

CHECK 2 -- THE PARTICIPATION-RATIO IDENTITY (a claim of mine, tested to be refuted).
  LOG.md, 12:4x: "under the section C.2 shuffle the read is a clean read-out of the ADDRESS
  BOOK's concentration".  Argument: a column permutation gives ``E_pi[x_read] = abar``
  exactly (column means preserved, ``sum_a P_a = 1``) and leaves the spread about ``abar``
  proportional to the posterior's PARTICIPATION RATIO ``PR = sum_a P_a^2`` times a
  column-variance the permutation also preserves exactly.  Measured here as a correlation
  between ``Q(x_read)`` under shuffle and ``PR``, across requests, over several shuffles.
  ⛔ If it does not correlate, the claim is wrong and gets its own refutation bullet.
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
from chlu.core.gen_landscape import _softmax_last, atom_posterior  # noqa: E402
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from farfield_null import centroid_surrogate, spearman  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

R_SHUF = 8


def entropy(P) -> np.ndarray:
    P = np.asarray(P, np.float64)
    return -(P * np.log(np.maximum(P, 1e-300))).sum(axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    land, book, grid = built["land"], built["book"], ctx["grid"]
    N = int(land.store.n_atoms)

    # ---- CHECK 1: the SHIPPED path, on dense class indices -------------------
    val_reqs = requests_from_rows(grid, ctx["val_rows"])
    Rq = np.asarray([[int(c) for c in r["class_indices"]] for r in val_reqs], np.int64)
    P_ship = _softmax_last(atom_posterior(book, Rq))          # the pinned path, verbatim
    log_w, n_terms, _ = R.term_experts(book, grid, val_reqs, R.T_MAX)
    P_mine = L.term_posterior(book, log_w, n_terms)
    H_ship, H_mine = entropy(P_ship), entropy(P_mine)

    # The same question asked of a request that names NOTHING: the PRIOR.  This is the
    # "no address at all" floor -- how much of log N the book realises is H_prior - H_post.
    lp = np.asarray(book.log_prior, np.float64)
    P_prior = _softmax_last(lp[None])[0]
    H_prior = entropy(P_prior[None])[0]

    check1 = {
        "n_atoms": N, "log_N_nats": float(np.log(N)),
        "SHIPPED_atom_posterior": {
            "mean_entropy_nats": float(H_ship.mean()),
            "mean_effective_atoms_expH": float(np.exp(H_ship).mean()),
            "mean_max_posterior": float(P_ship.max(-1).mean()),
        },
        "MY_term_posterior": {
            "mean_entropy_nats": float(H_mine.mean()),
            "mean_effective_atoms_expH": float(np.exp(H_mine).mean()),
            "mean_max_posterior": float(P_mine.max(-1).mean()),
        },
        "max_abs_entropy_delta": float(np.max(np.abs(H_ship - H_mine))),
        "PRIOR_no_request_at_all": {
            "mean_entropy_nats": float(H_prior.mean()),
            "mean_effective_atoms_expH": float(np.exp(H_prior).mean()),
        },
        "nats_the_REQUEST_actually_buys_per_well": float(H_prior.mean() - H_ship.mean()),
        "verdict_note": ("If SHIPPED and MINE agree, the addressability deficit is the "
                         "PIN's and not this hub's. The prior row is the honest floor: "
                         "what the request buys is H_prior - H_posterior, not log N - H."),
    }
    R.tlog(f"  CHECK1 shipped expH={check1['SHIPPED_atom_posterior']['mean_effective_atoms_expH']:.3f} "
           f"mine={check1['MY_term_posterior']['mean_effective_atoms_expH']:.3f} "
           f"prior={check1['PRIOR_no_request_at_all']['mean_effective_atoms_expH']:.3f}")

    # ---- CHECK 2: the participation-ratio identity --------------------------
    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    sets = build_sets(requests_from_rows(grid, _fit[_take]), grid, rng)
    sets = {"val": val_reqs, **sets}
    P = {}
    for name, reqs in sets.items():
        lw, nt, _ = R.term_experts(book, grid, reqs, R.T_MAX)
        P[name] = L.term_posterior(book, lw, nt)
    PR = np.concatenate([(np.asarray(P[k], np.float64) ** 2).sum(-1).sum(-1) for k in sets])

    rows = []
    for i in range(R_SHUF):
        st = R.shuffled_store(land.store, int(a.seed) + 1000 * (i + 1))
        Q = np.concatenate([centroid_surrogate(st, L.linear_read(st, P[k])) for k in sets])
        rows.append({"shuffle_seed": int(a.seed) + 1000 * (i + 1),
                     "pearson_Q_vs_PR": float(np.corrcoef(Q, PR)[0, 1]),
                     "spearman_Q_vs_PR": spearman(Q, PR)})
        R.tlog(f"  CHECK2 shuffle {i + 1}/{R_SHUF}: pearson(Q, PR) = "
               f"{rows[-1]['pearson_Q_vs_PR']:+.4f}  spearman = {rows[-1]['spearman_Q_vs_PR']:+.4f}")
    Qf = np.concatenate([centroid_surrogate(land.store, L.linear_read(land.store, P[k]))
                         for k in sets])
    check2 = {
        "claim": ("under the section C.2 shuffle, Q(x_read) is a monotone read-out of the "
                  "posterior participation ratio PR = sum_a P_a^2, because the permutation "
                  "preserves the column means and variances exactly"),
        "predicted_sign": "NEGATIVE pearson/spearman: larger PR (more concentrated) => "
                          "x_read further from abar => larger Q",
        "shuffled_realisations": rows,
        "mean_pearson_shuffled": float(np.mean([r["pearson_Q_vs_PR"] for r in rows])),
        "mean_spearman_shuffled": float(np.mean([r["spearman_Q_vs_PR"] for r in rows])),
        "FULL_STORE_pearson_Q_vs_PR": float(np.corrcoef(Qf, PR)[0, 1]),
        "FULL_STORE_spearman_Q_vs_PR": spearman(Qf, PR),
    }

    rep = {"schema": "hub-refusal-2/book-check/1",
           "DECLARED": ("MECHANISM ONLY, val + fit-derived synthetic. NOT scored, NOT "
                        "adopted. Two attempts to BREAK this hub's own diagnosis."),
           "seed": int(a.seed), "budget": str(a.budget),
           "check1_is_the_deficit_mine_or_the_pins": check1,
           "check2_participation_ratio_identity": check2}
    p = R.OUT / f"BOOK_CHECK_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n" + json.dumps(rep, indent=1, default=str))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
