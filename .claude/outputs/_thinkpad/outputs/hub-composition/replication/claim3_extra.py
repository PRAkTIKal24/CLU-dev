"""CLAIM 3 -- adversarial extras.

1. The hub's permuted-residual number (95) is below every draw of my own permutation
   sweep (min 104). So: is 95 reachable under a DIFFERENT permutation convention?
   Tries cyclic roll, a fixed-point-free derangement, and the legacy global RNG.
2. A dose-response the hub did not run: X + alpha*e for alpha in [0,1]. If the store's
   collapse were a magnitude effect it would be monotone-and-smooth in alpha; the
   interesting question is where the curve leaves the Gaussian control's level.
3. Per-factor breakdown of the round-trip vs the norm-matched Gaussian, which is what
   actually decides whether "structured" is the right word for the residual.
"""
import json
import sys

import numpy as np

import claim3
from claim3 import exact_count, rel_err, log


def main():
    S = claim3.setup()
    decoder, clf, store = S["decoder"], S["clf"], S["store"]
    X, true_lab, feat_shape, D, batch = (S["X"], S["true_lab"], S["feat_shape"],
                                         S["D"], S["batch"])
    from chlu.core import gen_store, gen_compose

    code = gen_store.address_codes(store, X)
    rt = np.asarray(gen_compose.read_compose(store, code), np.float32).reshape((-1, D))
    e = rt - X
    N = int(X.shape[0])
    out = {"n": N}

    # ---- 1. alternative permutation conventions ----------------------------
    perms = {}
    perms["roll_1"] = np.roll(np.arange(N), 1)
    perms["roll_256"] = np.roll(np.arange(N), 256)
    perms["reverse"] = np.arange(N)[::-1].copy()
    rng = np.random.default_rng(0)
    d = rng.permutation(N)
    while np.any(d == np.arange(N)):                 # crude derangement
        bad = np.flatnonzero(d == np.arange(N))
        j = rng.integers(0, N, size=bad.size)
        d[bad], d[j] = d[j], d[bad]
    perms["derangement_rng0"] = d
    npr = np.random.RandomState(0)
    perms["legacy_randomstate_0"] = npr.permutation(N)
    npr = np.random.RandomState(20260914)
    perms["legacy_randomstate_20260914"] = npr.permutation(N)

    out["alt_perms"] = {}
    for k, p in perms.items():
        Xp = (X.astype(np.float64) + e[p].astype(np.float64)).astype(np.float32)
        r = exact_count(decoder, clf, Xp, feat_shape, true_lab, batch, "perm/" + k)
        r["n_fixed_points"] = int((p == np.arange(N)).sum())
        r["rel_err_mean"] = rel_err(Xp, X)["mean_per_row"]
        out["alt_perms"][k] = r

    # ---- 2. dose-response in the TRUE residual direction --------------------
    out["dose_response"] = {}
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        Xa = (X.astype(np.float64) + alpha * e.astype(np.float64)).astype(np.float32)
        r = exact_count(decoder, clf, Xa, feat_shape, true_lab, batch,
                        "dose a=%.3f" % alpha)
        r["rel_err_mean"] = rel_err(Xa, X)["mean_per_row"]
        out["dose_response"]["%.3f" % alpha] = r
    # matched Gaussian dose-response, same per-row norms
    out["dose_response_gauss"] = {}
    en = np.linalg.norm(e.astype(np.float64), axis=1)
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        rng = np.random.default_rng(777)
        g = rng.standard_normal(X.shape)
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * (alpha * en)[:, None]
        Xa = (X.astype(np.float64) + g).astype(np.float32)
        r = exact_count(decoder, clf, Xa, feat_shape, true_lab, batch,
                        "doseG a=%.3f" % alpha)
        r["rel_err_mean"] = rel_err(Xa, X)["mean_per_row"]
        out["dose_response_gauss"]["%.3f" % alpha] = r

    # ---- 3. is rt informative about its OWN row at all? --------------------
    rngp = np.random.default_rng(5)
    p = rngp.permutation(N)
    r = exact_count(decoder, clf, rt[p], feat_shape, true_lab, batch, "rt[perm]")
    out["rt_permuted_rows"] = r
    out["roundtrip_per_factor"] = exact_count(decoder, clf, rt, feat_shape, true_lab,
                                              batch, "rt (again)")
    return out


if __name__ == "__main__":
    res = main()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        log("wrote", sys.argv[1])
