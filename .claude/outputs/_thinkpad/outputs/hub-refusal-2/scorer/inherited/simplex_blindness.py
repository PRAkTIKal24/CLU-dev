"""hub/refusal -- IS `out_of_range` STRUCTURALLY INVISIBLE?  (PREREG AMENDMENT 8)

The AMENDMENT-5/6 probes put `out_of_range` at chance (0.48-0.56) in every arm, at both
rungs, at every step budget.  The report offers a STRUCTURAL reason: the address book is
`log P(atom | f = v)` -- a point on the SIMPLEX over the well's stored atoms -- so R8's
extrapolation of it is still a distribution over those same atoms, and the induced term
stays satisfiable by a real atom.  **The simplex has no outside.**

This script turns that argument into a measurement.  For the SAME request, on-grid and at
extrapolation severities w, it reports:
  * `max_a P(atom | term)` and the expert's entropy -- does the expert stay a sane
    distribution, or does extrapolation blow it up?
  * the perturbed TERM'S OWN residual `r_t` at the launch and at the settled state -- if an
    off-grid term is as satisfied as an on-grid one, the landscape literally cannot see it;
  * the distance from the settled state to the perturbed expert's ARGMAX atom, against the
    distance to the nearest atom at all.

DECLARED AND BINDING, as AMENDMENT 3: FIT-DERIVED SYNTHETIC, MECHANISM ONLY, seed 0.
NOT the benchmark.  Nothing in PREREG-1 changes because of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402
import synthetic_probe as P  # noqa: E402
from chlu.core.gen_landscape import _softmax_last  # noqa: E402
from chlu.core.gen_request import _extrapolating_index, requests_from_rows  # noqa: E402

SEVERITIES = (0, 1, 2, 3, 6, 12)   # grid steps beyond the end; 0 = on-grid (the control)


def expert_stats(book, grid, f: int, v: float) -> dict:
    """The induced expert, as a distribution over the well's atoms."""
    lp = np.asarray(book.log_prior, np.float64)
    vals = np.asarray(grid.values[int(f)], np.float64)
    a, b, w = _extrapolating_index(vals, float(v))
    lc = np.asarray(book.log_cond[int(f)], np.float64)
    L = lc[a] if (a == b or w == 0.0) else (1.0 - w) * lc[a] + w * lc[b]
    L = L - R._logsumexp_last(L)
    Pr = _softmax_last(lp + (L - lp))          # = softmax(L) -- the term's own posterior
    ent = -(Pr * np.log(np.maximum(Pr, 1e-300))).sum(axis=-1)
    return {"w": float(w), "max_prob": float(Pr.max(axis=-1).mean()),
            "entropy_nats": float(ent.mean()),
            "max_entropy_nats": float(np.log(Pr.shape[-1])),
            "argmax_atom": Pr.argmax(axis=-1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    grid, book, store = ctx["grid"], built["book"], built["store"]
    rng = np.random.default_rng(P.RNG_SEED)
    fit = np.asarray(ctx["fit_rows"], np.int64)
    take = np.sort(rng.choice(fit.size, size=min(a.n, fit.size), replace=False))
    base = requests_from_rows(grid, fit[take])
    f_of = rng.integers(0, grid.n_factors, size=len(base))

    rows = []
    for k in SEVERITIES:
        reqs = []
        for i, req in enumerate(base):
            f = int(f_of[i])
            vals = np.asarray(grid.values[f], np.float64)
            step = float(vals[1] - vals[0]) if vals.size > 1 else 1.0
            terms = [dict(t) for t in req["request"]]
            if k > 0:
                for t in terms:
                    if int(t["factor_index"]) == f:
                        t["value"] = float(vals[-1] + k * step)
            else:
                for t in terms:
                    if int(t["factor_index"]) == f:
                        t["value"] = float(vals[-1])     # ON-GRID control: the last value
            reqs.append({**req, "id": f"{req['id']}-blind{k}", "kind": f"sev{k}",
                         "request": terms})
        rec_s = R.settle_unit(ctx, built, f"blind_{k}", reqs, arm="full",
                              budget=str(a.budget), log=lambda *_: None)
        # the PERTURBED term's own residual, isolated -- term slot == factor index
        res_s = rec_s["settled"]["residuals"]
        res_l = rec_s["launch"]["residuals"]
        idx = np.arange(len(reqs))
        r_settled = res_s[idx, f_of]
        r_launch = res_l[idx, f_of]
        # the expert's shape, and where the settled state ended up relative to its argmax
        ws, mx, ents, d_arg, d_near = [], [], [], [], []
        X = rec_s["x_settled"]
        for i, req in enumerate(reqs):
            f = int(f_of[i])
            v = next(float(t["value"]) for t in req["request"]
                     if int(t["factor_index"]) == f)
            st = expert_stats(book, grid, f, v)
            ws.append(st["w"]); mx.append(st["max_prob"]); ents.append(st["entropy_nats"])
            # distance from the settled state to that expert's argmax atom, per well
            d2a = d2n = 0.0
            for kk, wch in enumerate(store.well_channels):
                cols = np.asarray(wch, np.int64)
                z = X[i, cols].astype(np.float64)
                A = np.asarray(store.atoms[kk], np.float64)
                dd = ((A - z[None, :]) ** 2).sum(axis=1)
                d2a += float(dd[int(st["argmax_atom"][kk])])
                d2n += float(dd.min())
            d_arg.append(np.sqrt(d2a)); d_near.append(np.sqrt(d2n))
        row = {"severity_grid_steps": int(k),
               "extrapolation_w_mean": float(np.mean(ws)),
               "expert_max_prob_mean": float(np.mean(mx)),
               "expert_entropy_nats_mean": float(np.mean(ents)),
               "expert_max_entropy_nats": float(np.log(store.n_atoms)),
               "r_perturbed_term_launch_mean": float(np.mean(r_launch)),
               "r_perturbed_term_settled_mean": float(np.mean(r_settled)),
               "frac_perturbed_term_SATISFIED_settled": float(np.mean(r_settled < 0)),
               "d_settled_to_expert_argmax_atom_mean": float(np.mean(d_arg)),
               "d_settled_to_nearest_atom_mean": float(np.mean(d_near)),
               "S_max_mean": float(np.mean(rec_s["settled"]["S_max"]))}
        rows.append(row)
        print("sev=%3d  w=%6.2f  maxP=%.4f  H=%5.3f/%5.3f  r_term launch=%9.3f "
              "settled=%9.3f  frac satisfied=%.3f  d(argmax)=%7.3f d(nearest)=%7.3f  "
              "S_max=%7.3f"
              % (k, row["extrapolation_w_mean"], row["expert_max_prob_mean"],
                 row["expert_entropy_nats_mean"], row["expert_max_entropy_nats"],
                 row["r_perturbed_term_launch_mean"], row["r_perturbed_term_settled_mean"],
                 row["frac_perturbed_term_SATISFIED_settled"],
                 row["d_settled_to_expert_argmax_atom_mean"],
                 row["d_settled_to_nearest_atom_mean"], row["S_max_mean"]), flush=True)

    out = {"schema": "hub-refusal/simplex-blindness/1",
           "DECLARED": ("FIT-DERIVED SYNTHETIC, PREREG AMENDMENT 8. MECHANISM ONLY. "
                        "NOT the benchmark."),
           "seed": int(a.seed), "budget": str(a.budget), "n_bases": int(a.n),
           "rng_seed": P.RNG_SEED, "severities_grid_steps": list(SEVERITIES),
           "n_atoms_per_well": int(store.n_atoms), "rows": rows}
    p = R.OUT / f"SIMPLEX_BLINDNESS_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
