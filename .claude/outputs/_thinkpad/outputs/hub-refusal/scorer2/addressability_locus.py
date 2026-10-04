"""hub/refusal-2 -- WHERE the addressability deficit lives: the BOOK's fit, or the ENCODER?

MECHANISM ONLY.  fit + val rows and their labels, which the address book already declares as
interface supervision.  No C record, no a7_redundant.json, no held-out row.  NOT scored,
NOT adopted; the go/no-go returned NO-GO at both rungs and the criterion is frozen.

THE QUESTION THIS DECIDES, and it is the one the coordinator actually needs.
``posterior_anatomy`` measured that a complete valid six-factor request buys only 0.454 of
the 4.144 nats the prior leaves available, and that four of the six factors are effectively
invisible (scale 0.012, shape 0.014, orientation 0.046, object_hue 0.089 nats).  There are
exactly two places that deficit can live, and they have OPPOSITE consequences:

  (A) THE BOOK'S FIT is throwing information away.  ``fit_atom_address_book`` builds
      ``log P(atom | f = v)`` from ``soft_atom_assignment``, a softmax over atoms at the
      LANDSCAPE's bandwidth ``sigma``.  Too wide a bandwidth smears every train row over
      every atom and flattens every conditional.  ⭐ If this is the cause it is CHEAP to
      fix -- one bandwidth, inside charter section B's "everything else may change".
  (B) THE FROZEN ENCODER'S FEATURES do not carry those factors in a way 64 stored atoms can
      separate.  ⛔ If this is the cause, no address book over these atoms can ever address
      them, a refusal criterion on this store is unreachable at any budget, and the next
      move is an encoder question, not a read question.

FOUR MEASUREMENTS, none of which can be gamed by a threshold:

  1. ASSIGNMENT SHARPNESS.  The entropy of ``soft_atom_assignment`` itself, per train row,
     per well, against ``log N``.  Near-zero entropy = the assignment is already essentially
     a nearest-atom hard assignment and hypothesis (A) is DEAD on arrival.
  2. THE BANDWIDTH DOSE-RESPONSE.  Refit the book at ``sigma * rho`` for a ladder of ``rho``
     and re-measure the nats the request buys and the pre-registered statistic's AUROCs.
     ⛔⛔ NO POINT ON THIS LADDER IS ADOPTED.  ``rho = 1`` is the shipped book.
  3. THE FEATURE-SPACE CEILING, per (well, factor).  A Fisher-style separability ratio
     ``between-class variance / within-class variance`` of the WELL's own feature columns
     under that factor's labels, on fit rows.  This is a property of the ENCODER and the
     well partition ALONE -- it does not involve atoms, the book, or any read -- so it
     separates (A) from (B) cleanly.
  4. THE ATOM-COVERAGE CEILING.  How many distinct values of each factor the N stored atoms
     actually realise.  A factor whose atoms cover 3 of its 10 values cannot be addressed
     however the book is fitted.
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
from chlu.core.gen_landscape import (  # noqa: E402
    _softmax_last,
    fit_atom_address_book,
    soft_atom_assignment,
    store_energy,
)
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

OOR_SETS = ("syn_oor_1", "syn_oor_2", "syn_oor_3")
RHOS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def au(neg, pos) -> float:
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    return float(binary_auroc(s, y))


def pair(x: dict) -> tuple:
    oor = np.concatenate([x[s] for s in OOR_SETS])
    return au(x["syn_valid"], oor), au(x["syn_a7"], x["syn_dup"])


def entropy(P) -> np.ndarray:
    P = np.asarray(P, np.float64)
    return -(P * np.log(np.maximum(P, 1e-300))).sum(axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--n_fisher", type=int, default=8192)
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    land, book0, grid = built["land"], built["book"], ctx["grid"]
    store = land.store
    sigma = np.asarray(land.sigma, np.float64)
    sig_ref = 1.0
    N, K = int(store.n_atoms), int(store.n_wells)
    X_store = np.asarray(ctx["X_store"], np.float32)
    X_store = X_store.reshape((X_store.shape[0], -1))
    lab = np.asarray(ctx["lab_store"], np.int64)
    names = list(grid.names)
    cards = [int(c) for c in grid.cardinalities]

    rep = {"schema": "hub-refusal-2/addressability-locus/1",
           "DECLARED": ("MECHANISM ONLY, fit rows + val + the inherited synthetic analogue. "
                        "NOT scored, NOT adopted. NO POINT ON THE BANDWIDTH LADDER IS "
                        "ADOPTED: rho = 1 is the shipped book."),
           "seed": int(a.seed), "budget": str(a.budget),
           "n_atoms": N, "n_wells": K, "log_N_nats": float(np.log(N)),
           "sigma_per_well": sigma.round(4).tolist(),
           "factors": list(zip(names, cards))}

    # ---- 1. assignment sharpness -------------------------------------------
    sub = X_store[: min(4096, X_store.shape[0])]
    pis = soft_atom_assignment(store, sub, sigma)
    rep["assignment_sharpness"] = {
        "n_train_rows_sampled": int(sub.shape[0]),
        "log_N_nats": float(np.log(N)),
        "mean_entropy_nats_per_well": [float(entropy(p).mean()) for p in pis],
        "mean_effective_atoms_per_well": [float(np.exp(entropy(p)).mean()) for p in pis],
        "mean_max_responsibility_per_well": [float(np.asarray(p).max(-1).mean()) for p in pis],
        "reading": ("entropy ~ 0 and effective atoms ~ 1 means the shipped assignment is "
                    "ALREADY essentially nearest-atom, so hypothesis (A) 'the bandwidth "
                    "smears the conditionals' is dead and the deficit is upstream."),
    }
    R.tlog("  1. assignment effective atoms per well: "
           + " ".join("%.2f" % v for v in rep["assignment_sharpness"]
                      ["mean_effective_atoms_per_well"]))

    # ---- 3. the feature-space ceiling (encoder + partition only) ------------
    nF = min(int(a.n_fisher), X_store.shape[0])
    Xf = X_store[:nF].astype(np.float64)
    Yf = lab[:nF]
    fisher = np.zeros((K, len(names)), np.float64)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        Z = Xf[:, cols]
        gm = Z.mean(0)
        tot = float(((Z - gm) ** 2).sum(1).mean())
        for f in range(len(names)):
            bet = 0.0
            wit = 0.0
            for v in range(cards[f]):
                sel = Yf[:, f] == v
                n_v = int(sel.sum())
                if n_v < 2:
                    continue
                mu = Z[sel].mean(0)
                bet += n_v * float(((mu - gm) ** 2).sum())
                wit += float(((Z[sel] - mu) ** 2).sum())
            fisher[k, f] = bet / max(wit, 1e-12)
        R.tlog(f"  3. well {k}: total var/row {tot:.2f}  fisher "
               + " ".join("%.4f" % v for v in fisher[k]))
    rep["feature_space_ceiling_fisher_ratio"] = {
        "n_rows": nF, "names": names,
        "between_over_within_by_well_factor": fisher.round(5).tolist(),
        "max_over_wells_by_factor": dict(zip(names, fisher.max(0).round(5).tolist())),
        "reading": ("a property of the FROZEN ENCODER and the well partition alone -- no "
                    "atoms, no book, no read. A factor with a ratio near 0 is not present "
                    "in the well's features and cannot be addressed however the book is fit."),
    }

    # ---- 4. the atom-coverage ceiling ---------------------------------------
    atom_rows = np.asarray(store.atom_rows, np.int64)
    atom_lab = lab[atom_rows] if atom_rows.max() < lab.shape[0] else None
    cov = {}
    if atom_lab is not None:
        for f, nm in enumerate(names):
            u = np.unique(atom_lab[:, f])
            cov[nm] = {"values_realised_by_atoms": int(u.size), "cardinality": cards[f],
                       "fraction": float(u.size / cards[f])}
    rep["atom_coverage_ceiling"] = {
        "n_atoms": N, "per_factor": cov,
        "n_distinct_full_tuples": (int(np.unique(atom_lab, axis=0).shape[0])
                                   if atom_lab is not None else None),
        "reading": ("a factor whose stored atoms realise only a few of its values cannot be "
                    "addressed however the book is fitted."),
    }

    # ---- 2. the bandwidth dose-response -------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    sets = build_sets(requests_from_rows(grid, _fit[_take]), grid, rng)
    sets = {"val": requests_from_rows(grid, ctx["val_rows"]), **sets}

    lp0 = np.asarray(book0.log_prior, np.float64)
    H_prior0 = float(entropy(_softmax_last(lp0[None])[0]).mean())

    ladder = []
    for rho in RHOS:
        bk = (book0 if rho == 1.0 else
              fit_atom_address_book(store, X_store, lab, cards, sigma * float(rho)))
        Sd, Hd = {}, {}
        for name, reqs in sets.items():
            lw, nt, _ = R.term_experts(bk, grid, reqs, R.T_MAX)
            P = L.term_posterior(bk, lw, nt)
            Sd[name] = np.asarray(store_energy(land, L.linear_read(store, P), sig_ref),
                                  np.float64)
            Hd[name] = float(entropy(P).mean())
        Hp = float(entropy(_softmax_last(np.asarray(bk.log_prior, np.float64)[None])[0]).mean())
        po = pair(Sd)
        ladder.append({
            "rho": float(rho), "is_shipped": bool(rho == 1.0),
            "H_prior_nats": Hp, "H_posterior_val_nats": Hd["val"],
            "nats_the_request_buys": Hp - Hd["val"],
            "effective_atoms_val": float(np.exp(Hd["val"])),
            "AUROC_oor_pooled": po[0], "AUROC_dup_vs_a7": po[1],
            "min_of_the_two": float(min(po)),
        })
        R.tlog(f"  2. rho={rho:<7} buys {ladder[-1]['nats_the_request_buys']:.4f} nats  "
               f"expH={ladder[-1]['effective_atoms_val']:6.2f}  "
               f"oor={po[0]:.4f} dup={po[1]:.4f}")
    rep["bandwidth_dose_response_NOT_ADOPTED"] = {
        "shipped_H_prior_nats": H_prior0, "ladder": ladder,
        "reading": ("if tightening the assignment bandwidth restores the nats AND the "
                    "AUROCs, the deficit is the BOOK's fit (A) and is cheap to fix; if the "
                    "nats move and the AUROCs do not, the read is the problem; if neither "
                    "moves, the ENCODER is the problem (B)."),
    }

    p = R.OUT / f"ADDRESSABILITY_LOCUS_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n" + json.dumps({k: v for k, v in rep.items() if k not in ("factors",)},
                            indent=1, default=str))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
