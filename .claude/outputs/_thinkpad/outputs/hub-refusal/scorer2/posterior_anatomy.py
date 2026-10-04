"""hub/refusal-2 -- WHY the address book never sharpens, in information, not adjectives.

MECHANISM ONLY.  val + the inherited fit-derived synthetic analogue.  No C record, no
a7_redundant.json, no held-out row.  NOT scored, NOT adopted; the go/no-go already returned
NO-GO at both rungs and the criterion is frozen.

THE QUESTION.  The go/no-go measured ``mean max_a P(a|request, well k) = 0.0675`` at rung A
(uniform = 1/64 = 0.0156) and ``0.0187`` at rung B (uniform = 1/512 = 0.00195) -- for VALID
requests as much as invalid ones.  So the linear read averages most of the well's atoms for
every request, and the argmax read picks one of many near-ties.  WHY?

WHAT IS MEASURED, all of it from the SHIPPED address book (``fit_atom_address_book``):

  1. IDENTIFICATION BUDGET.  ``log N`` nats are available per well.  The posterior for a
     VALID request realises ``H`` of them; the deficit ``log N - H`` is what the address
     book fails to resolve.  ``exp(H)`` is the effective number of atoms the linear read
     actually averages -- the headline number.

  2. WHERE THE BUDGET GOES, per (well, factor).  ``I_kf = E_v KL( P(a | f=v, k) || P(a | k) )``
     -- the naive-Bayes information factor ``f`` carries about atom identity in well ``k``,
     in nats, straight out of ``book.log_cond`` / ``book.log_prior``.  ``sum_f I_kf`` is the
     naive-Bayes budget, which OVERCOUNTS because it ignores dependence between factors;
     the realised ``log N - H`` is the honest figure, and the gap between them measures how
     redundant the six factors are inside one well.

  3. THE SHARPENING CURVE.  Temper the posterior, ``P_tau ~ P^tau`` renormalised, and track
     ``d(x_read, nearest atom) / atom spread`` and both AUROCs as a function of ``tau``.
     This says what concentration WOULD be needed to bring the linear read out of the far
     field, and whether refusal information appears when it does.
     ⛔⛔ NO POINT ON THIS CURVE IS ADOPTED.  ``tau = 1`` is the shipped read and the
     pre-registered object; the curve exists to tell the coordinator whether a PREREG-3 on
     a sharpened address book could be worth a slot NEXT iteration, on a machine that can
     produce three seeds.  Reading a tau off it and scoring it would be criterion-shopping
     on the analogue and the zero-supervision claim would be VOID.
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
from chlu.core.gen_landscape import store_energy  # noqa: E402
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import binary_auroc  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

OOR_SETS = ("syn_oor_1", "syn_oor_2", "syn_oor_3")
TAUS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e9)   # 1e9 == the argmax limit


def au(neg, pos) -> float:
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    y = np.concatenate([np.zeros(len(neg), bool), np.ones(len(pos), bool)])
    return float(binary_auroc(s, y))


def pair(x: dict) -> tuple:
    oor = np.concatenate([x[s] for s in OOR_SETS])
    return au(x["syn_valid"], oor), au(x["syn_a7"], x["syn_dup"])


def entropy(P) -> np.ndarray:
    """``(n, K, N) -> (n, K)`` posterior entropy in nats."""
    P = np.asarray(P, np.float64)
    return -(P * np.log(np.maximum(P, 1e-300))).sum(axis=-1)


def temper(P, tau: float):
    """``P^tau`` renormalised.  ``tau -> inf`` is the argmax (one-hot)."""
    P = np.asarray(P, np.float64)
    if tau > 1e6:
        out = np.zeros_like(P)
        idx = P.argmax(axis=-1)
        np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
        return out
    lg = float(tau) * np.log(np.maximum(P, 1e-300))
    lg -= lg.max(axis=-1, keepdims=True)
    e = np.exp(lg)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    land, book, grid = built["land"], built["book"], ctx["grid"]
    sig_ref = float(gen_dynamical_read_config(ctx["config"]).sigma_scale_end)
    N = int(land.store.n_atoms)
    K = int(land.store.n_wells)

    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    base = requests_from_rows(grid, _fit[_take])
    sets = build_sets(base, grid, rng)
    sets = {"val": requests_from_rows(grid, ctx["val_rows"]), **sets}

    P = {}
    for name, reqs in sets.items():
        log_w, n_terms, _ = R.term_experts(book, grid, reqs, R.T_MAX)
        P[name] = L.term_posterior(book, log_w, n_terms)

    # -- 1. the identification budget -----------------------------------------
    budget = {"n_atoms": N, "n_wells": K, "log_N_nats": float(np.log(N))}
    for name in sets:
        H = entropy(P[name])                       # (n, K)
        budget[name] = {
            "mean_entropy_nats": float(H.mean()),
            "mean_deficit_log_N_minus_H": float(np.log(N) - H.mean()),
            "frac_of_log_N_realised": float((np.log(N) - H.mean()) / np.log(N)),
            "mean_effective_atoms_expH": float(np.exp(H).mean()),
            "mean_max_posterior": float(np.asarray(P[name]).max(axis=-1).mean()),
        }

    # -- 2. where the budget goes, per (well, factor) -------------------------
    lp = np.asarray(book.log_prior, np.float64)          # (K, N)
    prior = np.exp(lp - np.log(np.exp(lp).sum(-1, keepdims=True)))
    per_wf = np.zeros((K, len(book.log_cond)), np.float64)
    for f, lc in enumerate(book.log_cond):
        lcv = np.asarray(lc, np.float64)                 # (V, K, N)
        V = lcv.shape[0]
        for v in range(V):
            q = np.exp(lcv[v] - np.log(np.exp(lcv[v]).sum(-1, keepdims=True)))
            per_wf[:, f] += (q * (np.log(np.maximum(q, 1e-300))
                                  - np.log(np.maximum(prior, 1e-300)))).sum(-1) / V
    info = {
        "names": list(grid.names),
        "cardinalities": [int(c) for c in grid.cardinalities],
        "I_well_by_factor_nats": per_wf.round(4).tolist(),
        "naive_bayes_budget_per_well_nats": per_wf.sum(1).round(4).tolist(),
        "log_N_nats": float(np.log(N)),
        "realised_per_well_nats_on_val": (
            np.log(N) - entropy(P["val"]).mean(axis=0)).round(4).tolist(),
        "note": ("sum_f I_kf OVERCOUNTS (naive Bayes ignores dependence). The realised "
                 "log N - H is the honest figure; the gap is factor redundancy inside "
                 "one well."),
    }

    # -- 3. the sharpening curve ----------------------------------------------
    curve = []
    for tau in TAUS:
        Sd, Dd = {}, {}
        for name in sets:
            Pt = temper(P[name], tau)
            x = L.linear_read(land.store, Pt)
            Sd[name] = np.asarray(store_energy(land, x, sig_ref), np.float64)
            Dd[name] = L.nearest_atom_distance(land.store, x)["d_over_spread"]
        po = pair(Sd)
        curve.append({
            "tau": (None if tau > 1e6 else float(tau)),
            "is_argmax_limit": bool(tau > 1e6),
            "mean_max_posterior_syn_valid": float(temper(P["syn_valid"], tau).max(-1).mean()),
            "mean_d_over_spread_syn_valid": float(np.mean(Dd["syn_valid"])),
            "mean_d_over_spread_syn_oor_3": float(np.mean(Dd["syn_oor_3"])),
            "AUROC_oor_pooled": po[0], "AUROC_dup_vs_a7": po[1],
            "min_of_the_two": float(min(po)),
        })
        R.tlog(f"  tau={str(tau):>8}  d/spread={curve[-1]['mean_d_over_spread_syn_valid']:8.3f}  "
               f"oor={po[0]:.4f} dup={po[1]:.4f}")

    rep = {"schema": "hub-refusal-2/posterior-anatomy/1",
           "DECLARED": ("MECHANISM ONLY, fit-derived synthetic + val. NOT scored, NOT the "
                        "benchmark. NO POINT ON THE SHARPENING CURVE IS ADOPTED: tau = 1 "
                        "is the shipped read and the pre-registered object."),
           "seed": int(a.seed), "budget": str(a.budget),
           "identification_budget": budget,
           "information_per_well_factor": info,
           "sharpening_curve_NOT_ADOPTED": curve}
    p = R.OUT / f"POSTERIOR_ANATOMY_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n" + json.dumps({"identification_budget": budget,
                             "information_per_well_factor": info}, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
