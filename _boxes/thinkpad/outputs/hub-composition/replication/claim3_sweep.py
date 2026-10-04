"""CLAIM 3c -- the two RANDOM controls are RNG-dependent, so sweep the RNG.

A single draw of a Gaussian control is not a number, it is a sample. This sweeps 12
draws of each random control and reports the distribution, so that "hub says 393" can
be judged against a spread rather than against one other draw.
"""
import json
import sys

import numpy as np

import claim3
from claim3 import exact_count, rel_err, log

N_DRAWS = 12


def main():
    S = claim3.setup()
    decoder, clf, store = S["decoder"], S["clf"], S["store"]
    X, true_lab, feat_shape, D, batch = (S["X"], S["true_lab"], S["feat_shape"],
                                         S["D"], S["batch"])
    from chlu.core import gen_store, gen_compose

    code = gen_store.address_codes(store, X)
    rt = np.asarray(gen_compose.read_compose(store, code), np.float32).reshape((-1, D))
    e = rt - X
    en = np.linalg.norm(e.astype(np.float64), axis=1)

    out = {"n": int(X.shape[0]), "n_draws": N_DRAWS, "gauss": [], "perm": [],
           "gauss_rel_err": [], "perm_rel_err": []}

    for s in range(N_DRAWS):
        rng = np.random.default_rng(10_000 + s)
        g = rng.standard_normal(X.shape)
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * en[:, None]
        Xg = (X.astype(np.float64) + g).astype(np.float32)
        r = exact_count(decoder, clf, Xg, feat_shape, true_lab, batch, "gauss s%d" % s)
        out["gauss"].append(r["exact"])
        out["gauss_rel_err"].append(rel_err(Xg, X)["mean_per_row"])

    for s in range(N_DRAWS):
        rng = np.random.default_rng(20_000 + s)
        perm = rng.permutation(X.shape[0])
        Xp = (X.astype(np.float64) + e[perm].astype(np.float64)).astype(np.float32)
        r = exact_count(decoder, clf, Xp, feat_shape, true_lab, batch, "perm s%d" % s)
        out["perm"].append(r["exact"])
        out["perm_rel_err"].append(rel_err(Xp, X)["mean_per_row"])

    for k in ("gauss", "perm"):
        a = np.asarray(out[k], float)
        out[k + "_stats"] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                             "min": int(a.min()), "max": int(a.max())}
        b = np.asarray(out[k + "_rel_err"], float)
        out[k + "_rel_err_stats"] = {"mean": float(b.mean()), "min": float(b.min()),
                                     "max": float(b.max())}
        log(k, out[k + "_stats"], "rel_err", out[k + "_rel_err_stats"])
    return out


if __name__ == "__main__":
    r = main()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(r, fh, indent=1, default=str)
        log("wrote", sys.argv[1])
