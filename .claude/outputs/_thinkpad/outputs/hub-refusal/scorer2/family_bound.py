"""hub/refusal-2 -- from "THIS criterion fails section C.2" to "NO statistic of the read can pass it".

MECHANISM ONLY.  The inherited fit-derived synthetic analogue + val.  No C record, no
a7_redundant.json, no held-out row.  NOT scored, NOT adopted; the go/no-go returned NO-GO at
both rungs and the pre-registered criterion is frozen.

WHAT THIS ADDS.  Everything measured so far is about ONE statistic, ``S = E_store(x_read)``.
A reviewer's fair objection is "you picked a bad statistic".  This answers it WITHOUT picking
another one, by asking a question about the read point itself:

    does the population of x_read for VALID requests differ from the population for INVALID
    requests ANY MORE under the real store than under a section C.2 destroyed store?

Any refusal criterion whatever, of the form ``f(x_read)``, is a function of that read point.
If the two populations are no further apart under the real atoms than under shuffled atoms,
then no ``f`` can be a CHLU refusal claim -- because whatever separation ``f`` finds, the
landscape-removed control finds as much, and charter section C.2 voids it.

THE STATISTIC, chosen because it requires NO FITTING and so cannot be tuned:
the ENERGY DISTANCE, in its scale-free ratio form,

    E(X, Y)  = 2 E|x - y| - E|x - x'| - E|y - y'|,        R(X, Y) = E(X, Y) / (2 E|x - y|)

``R`` is in [0, 1], is 0 exactly when the two distributions coincide, and is invariant to a
global rescaling of the space -- which matters here because the section C.2 shuffle changes the
atom cloud's geometry and therefore the scale of x_read.  ⛔ No threshold, no classifier, no
hyper-parameter, nothing selected on the data.

Reported for the full store and for R_SHUF independent shuffles, on the A7-CONTROLLED
comparison (dup vs a7) and the out-of-range one (oor pooled vs syn_valid), with a two-sided
rank p-value of the full store against its own landscape-removed null.
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
from chlu.core.gen_request import requests_from_rows  # noqa: E402
from synthetic_probe import N_BASES, RNG_SEED, build_sets  # noqa: E402

OOR_SETS = ("syn_oor_1", "syn_oor_2", "syn_oor_3")
R_SHUF = 16


def _pdist_mean(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> float:
    """mean ||a - b|| over all pairs, chunked so no (n, m, d) tensor is formed."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    bb = (B * B).sum(1)[None, :]
    tot, cnt = 0.0, 0
    for i in range(0, A.shape[0], chunk):
        a = A[i: i + chunk]
        d2 = (a * a).sum(1)[:, None] - 2.0 * (a @ B.T) + bb
        tot += float(np.sqrt(np.maximum(d2, 0.0)).sum())
        cnt += d2.size
    return tot / max(cnt, 1)


def energy_ratio(X: np.ndarray, Y: np.ndarray) -> dict:
    """Scale-free energy distance ratio R = E(X,Y) / (2 E|x-y|), in [0, 1]."""
    exy = _pdist_mean(X, Y)
    exx = _pdist_mean(X, X)
    eyy = _pdist_mean(Y, Y)
    e = 2.0 * exy - exx - eyy
    return {"energy_distance": float(e), "E_xy": float(exy), "E_xx": float(exx),
            "E_yy": float(eyy),
            "ratio_scale_free": float(e / (2.0 * exy)) if exy > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    land, book, grid = built["land"], built["book"], ctx["grid"]

    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    sets = build_sets(requests_from_rows(grid, _fit[_take]), grid, rng)
    sets = {"val": requests_from_rows(grid, ctx["val_rows"]), **sets}

    # The posterior is arm-independent by construction (the book is built from atom LABELS),
    # so it is computed once and every store realisation below reuses it bit for bit.
    P = {}
    for name, reqs in sets.items():
        lw, nt, _ = R.term_experts(book, grid, reqs, R.T_MAX)
        P[name] = L.term_posterior(book, lw, nt)
    R.tlog(f"  posteriors built for {len(P)} sets")

    def measure(store) -> dict:
        x = {k: L.linear_read(store, P[k]) for k in sets}
        oor = np.concatenate([x[s] for s in OOR_SETS], 0)
        return {
            "dup_vs_a7_A7_CONTROLLED": energy_ratio(x["syn_a7"], x["syn_dup"]),
            "oor_pooled_vs_syn_valid": energy_ratio(x["syn_valid"], oor),
            # the two SELF checks: two valid populations must be ~indistinguishable
            "SELFCHECK_synvalid_vs_val": energy_ratio(x["val"], x["syn_valid"]),
        }

    full = measure(land.store)
    R.tlog(f"  full store: dup {full['dup_vs_a7_A7_CONTROLLED']['ratio_scale_free']:.5f}  "
           f"oor {full['oor_pooled_vs_syn_valid']['ratio_scale_free']:.5f}  "
           f"selfcheck {full['SELFCHECK_synvalid_vs_val']['ratio_scale_free']:.5f}")

    null = []
    for i in range(R_SHUF):
        st = R.shuffled_store(land.store, int(a.seed) + 1000 * (i + 1))
        m = measure(st)
        null.append({"shuffle_seed": int(a.seed) + 1000 * (i + 1),
                     "dup": m["dup_vs_a7_A7_CONTROLLED"]["ratio_scale_free"],
                     "oor": m["oor_pooled_vs_syn_valid"]["ratio_scale_free"],
                     "selfcheck": m["SELFCHECK_synvalid_vs_val"]["ratio_scale_free"]})
        R.tlog(f"  shuffle {i + 1}/{R_SHUF}: dup {null[-1]['dup']:.5f}  oor {null[-1]['oor']:.5f}")

    out = {"schema": "hub-refusal-2/family-bound/1",
           "DECLARED": ("MECHANISM ONLY, fit-derived synthetic + val. NOT scored, NOT "
                        "adopted. A question about the READ POINT, not about any criterion."),
           "statistic": ("scale-free energy-distance ratio R = E(X,Y) / (2 E|x-y|) in [0,1]; "
                         "no fitting, no threshold, no hyper-parameter"),
           "seed": int(a.seed), "budget": str(a.budget), "R_shuffles": R_SHUF,
           "full_store": full, "null_realisations": null}
    for key in ("dup", "oor"):
        v = np.asarray([n[key] for n in null], np.float64)
        obs = full["dup_vs_a7_A7_CONTROLLED" if key == "dup"
                   else "oor_pooled_vs_syn_valid"]["ratio_scale_free"]
        out[f"verdict_{key}"] = {
            "full_store_ratio": float(obs),
            "null_mean": float(v.mean()), "null_std": float(v.std(ddof=1)),
            "null_min": float(v.min()), "null_max": float(v.max()),
            "full_exceeds_every_shuffle": bool(obs > v.max()),
            "n_shuffles_at_least_as_separated": int((v >= obs).sum()),
            "one_sided_rank_p": float((v >= obs).mean()),
            "z_vs_null": float((obs - v.mean()) / v.std(ddof=1)) if v.std(ddof=1) > 0 else None,
        }
    out["READING"] = (
        "If the full store does NOT separate valid from invalid read points more than its own "
        "landscape-removed control does, then NO criterion f(x_read) can be a CHLU refusal "
        "claim on this read: whatever separation f finds, the section C.2 control finds as much, "
        "and the charter voids it.")
    p = R.OUT / f"FAMILY_BOUND_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\n" + json.dumps({k: v for k, v in out.items()
                             if k.startswith("verdict") or k in ("full_store", "READING")},
                            indent=1, default=str))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
