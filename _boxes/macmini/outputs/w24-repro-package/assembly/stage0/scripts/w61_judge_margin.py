"""w6.1 retry -- HOW FAR CAN THE STATE MOVE BEFORE THE READ-BACK CHANGES?

The lattice measurement says the settled read loses exactly the factors whose answers
are packed most tightly, and that the loss tracks the displacement measured in units of
that factor's own answer spacing.  ⭐ That is still a correlation across legs.  This
script measures the underlying quantity directly and turns it into a prediction:

  * take the INCUMBENT's answer to a request,
  * push it a controlled distance TOWARD the nearest answer that differs only in
    factor ``f``, in units of that factor's spacing ``rho_f``,
  * ask the judge what it now reads back for factor ``f``.

That yields the **margin CDF** ``M_f(s) = P(the read-back of factor f changes | the
state moved s rho_f toward its f-neighbour)`` -- the judge's tolerance to motion, per
factor, in lattice units.  A **random-direction control** at matched lengths separates
"motion toward another answer" from "any motion at all".

Then the prediction: for a settle leg whose per-request fractional progress toward the
f-neighbour is ``p_f``, the new-error rate at factor ``f`` should be ``mean_i M_f(p_f)``.
⭐ That is compared against what the leg actually scored.

⛔ **`val` only, and nothing here is scored.** The val condition is the selection
condition (charter A.2); the held-out legs are read for confirmation and are never
touched by this probe.  It does construct the judge -- that is what makes it a
measurement of the judge's margin rather than a guess at it.

    python scripts/w61_judge_margin.py --leg DIR --seed S --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w6_compose_physics as P

#: the scales at which the directed probe is taken, in units of rho_f.  1.0 lands the
#: state exactly on the neighbouring answer.
SCALES = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
#: absolute lengths for the isotropic control, chosen to bracket every leg's
#: displacement (f05 ~0.27, onestep ~1.3, the arm ~10.2).
RAND_LENGTHS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="val")
    ap.add_argument("--every", type=int, default=6, help="subsample of requests")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    leg = Path(a.leg)
    t0 = time.time()

    P.install()
    R.set_condition(a.seed, a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    corners = (fw.val_corners(cond) if a.condition == "val"
               else fw.nway_corners(cond))
    cards = [int(c) for c in grid.cardinalities]
    gv = [np.asarray(v, np.float64) for v in grid.values]
    F = len(cards)
    cols = [(f, v) for f in range(F) for v in range(cards[f])]
    nnb = len(cols)

    with open(leg / "arms" / "pca_tensor" / "arm.pkl", "rb") as fh:
        obj_t = pickle.load(fh)
    store, basis = obj_t["store"], obj_t["basis"]
    rng = np.random.default_rng(int(fw.PINS["seed"]))

    # -- assemble the probe set -------------------------------------------
    Xr, Irq, U, Rho = [], [], [], []
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        sub = np.arange(0, vals.shape[0], int(a.every))
        vals = vals[sub]
        n = vals.shape[0]
        I = np.zeros((n, F), np.int64)
        for f in range(F):
            I[:, f] = np.abs(vals[:, f][:, None] - gv[f][None, :]).argmin(1)
        nb_vals = np.repeat(vals[:, None, :], nnb, axis=1)
        selfcol = np.zeros((n, nnb), bool)
        for j, (f, v) in enumerate(cols):
            nb_vals[:, j, f] = gv[f][v]
            selfcol[:, j] = I[:, f] == v
        X, _ = R.arm_features(obj_t, vals, grid)
        X = np.asarray(X, np.float64)
        Xnb, _ = R.arm_features(obj_t, nb_vals.reshape(-1, F), grid)
        Xnb = np.asarray(Xnb, np.float64).reshape(n, nnb, -1)
        D = Xnb - X[:, None, :]
        dn = np.where(selfcol, np.inf, np.linalg.norm(D, axis=2))
        u = np.zeros((n, F, X.shape[1]))
        rho = np.zeros((n, F))
        for f in range(F):
            m = np.flatnonzero([cf == f for cf, _v in cols])
            j = m[np.argmin(dn[:, m], axis=1)]
            d = D[np.arange(n), j]
            r = dn[np.arange(n), j]
            u[:, f] = d / r[:, None]
            rho[:, f] = r
        Xr.append(X)
        Irq.append(I)
        U.append(u)
        Rho.append(rho)
    X = np.concatenate(Xr, 0)
    I = np.concatenate(Irq, 0)
    U = np.concatenate(U, 0)
    Rho = np.concatenate(Rho, 0)
    N = X.shape[0]
    print(f"probe set: {N} requests x {F} factors x {len(SCALES)} scales "
          f"= {N * F * len(SCALES)} judged states, plus "
          f"{N * len(RAND_LENGTHS)} isotropic controls", flush=True)

    judge = fw.Judge(int(fw.PINS["seed"]), log=lambda m: print(m, flush=True))
    eb = int(fw.PINS["eval_batch"])

    base = judge.read_back(np.asarray(X, np.float32), eb)
    ok0 = base == I
    print(f"  incumbent per-factor agreement: "
          f"{np.round(ok0.mean(0), 4).tolist()}  ({time.time() - t0:.0f}s)", flush=True)

    directed = np.zeros((F, len(SCALES), N, F), np.int64)
    for f in range(F):
        for si, s in enumerate(SCALES):
            Xp = X + (float(s) * Rho[:, f])[:, None] * U[:, f]
            directed[f, si] = judge.read_back(np.asarray(Xp, np.float32), eb)
        print(f"  factor {f} probed  ({time.time() - t0:.0f}s)", flush=True)

    # isotropic control: a random direction INSIDE the span, matched lengths
    g = rng.standard_normal(X.shape)
    gin = np.zeros_like(g)
    for idx, Uk in zip(store.well_channels, basis):
        idx = np.asarray(idx, np.int64)
        Un = np.asarray(Uk, np.float64)
        gin[:, idx] = (g[:, idx] @ Un) @ Un.T
    gin /= np.linalg.norm(gin, axis=1, keepdims=True)
    rand = np.zeros((len(RAND_LENGTHS), N, F), np.int64)
    for li, L in enumerate(RAND_LENGTHS):
        rand[li] = judge.read_back(np.asarray(X + float(L) * gin, np.float32), eb)
    print(f"  isotropic control done  ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "leg": str(leg), "seed": a.seed, "condition": a.condition,
        "n_requests": int(N), "scales_in_rho_units": list(SCALES),
        "random_control_lengths": list(RAND_LENGTHS),
        "median_rho_per_factor": [float(np.median(Rho[:, f])) for f in range(F)],
        "incumbent_agreement_per_factor": [float(ok0[:, f].mean()) for f in range(F)],
        "margin_cdf_directed": {},
        "margin_cdf_isotropic": {},
        "median_state_norm": float(np.median(np.linalg.norm(X, axis=1))),
    }
    for f in range(F):
        m = ok0[:, f]
        out["margin_cdf_directed"][str(f)] = [
            float((directed[f, si][m, f] != I[m, f]).mean())
            for si in range(len(SCALES))]
        out["margin_cdf_isotropic"][str(f)] = [
            float((rand[li][m, f] != I[m, f]).mean())
            for li in range(len(RAND_LENGTHS))]
    out["whole_tuple_valid_directed_scale1"] = {
        str(f): float((directed[f, -1] == I).all(1).mean()) for f in range(F)}
    out["whole_tuple_valid_isotropic"] = [
        float((rand[li] == I).all(1).mean()) for li in range(len(RAND_LENGTHS))]
    out["whole_tuple_valid_incumbent"] = float(ok0.all(1).mean())

    np.savez_compressed(Path(a.out).with_suffix(".npz"),
                        I=I, base=base, rho=Rho, directed=directed, rand=rand,
                        scales=np.asarray(SCALES), lengths=np.asarray(RAND_LENGTHS))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
