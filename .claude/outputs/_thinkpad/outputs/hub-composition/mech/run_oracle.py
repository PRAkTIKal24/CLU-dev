"""⭐⭐ THE ORACLE-SELECTION CONTROL — does the ATOM BASIS represent the target at all?

⛔ NOT A CANDIDATE ARM and it can never become one: it reads the TRUE FEATURE to choose
the per-well index, so §B.3 (request-only reads) bars it permanently. It exists only to
put a NUMBER on what perfect addressing could ever buy, so that `x0`'s shortfall can be
attributed to representation or to addressing rather than argued about.

⛔ `val` and `MS` only. ⛔ B never enters.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from chlu.config import gen_dynamical_read_config
from chlu.experiments.exp_gen_dynamical_read import build_landscape
from mech_driver import build_context, arm_x0, score

ctx = build_context(0)
res = {}


def oracle_x0(store, X_true, mode="nearest"):
    """Per well, the atom minimising ||atom - target|| ON THAT WELL'S CHANNELS."""
    n = int(X_true.shape[0])
    x = np.zeros((n, int(store.dim)), np.float32)
    picks = np.zeros((n, len(store.well_channels)), np.int64)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)          # (N, |w|)
        T = np.asarray(X_true, np.float32)[:, cols]          # (n, |w|)
        d = (T * T).sum(1)[:, None] - 2.0 * (T @ A.T) + (A * A).sum(1)[None, :]
        idx = np.asarray(d.argmin(axis=1), np.int64)
        picks[:, k] = idx
        x[:, cols] = A[idx]
    return x, picks


for budget in ("A", "B"):
    t = time.time()
    built = build_landscape(ctx["cfg"], budget, ctx["X_store"], ctx["lab_store"], ctx["wc"],
                            ctx["feat_shape"], ctx["cards"], 0, log=lambda *a: None)
    store = built["store"]
    atoms = np.asarray(ctx["lab_store"], np.int64)[np.asarray(store.atom_rows, np.int64)]
    print(f"### landscape {budget} in {time.time()-t:.0f}s  "
          f"({len(store.well_channels)} wells, sizes {list(map(int, store.well_sizes))}, "
          f"{np.asarray(store.atoms[0]).shape[0]} atoms/well)", flush=True)
    for which in ("val", "MS"):
        rows, X, lab = ctx["sets"][which]
        # the oracle
        xo, picks = oracle_x0(store, X)
        rep = score(ctx, xo, lab, atoms)
        res[f"oracle_{budget}_{which}"] = rep
        # relative reconstruction error, the physical quantity behind the score
        err = np.linalg.norm(xo - X, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-9)
        # the arm, for the same rows in the same process
        lau = arm_x0(ctx, built["land"], built["book"], which)
        repa = score(ctx, lau["x0"], lab, atoms)
        res[f"x0_{budget}_{which}"] = repa
        erra = np.linalg.norm(np.asarray(lau["x0"]) - X, axis=1) / np.maximum(
            np.linalg.norm(X, axis=1), 1e-9)
        agree = float((picks == np.asarray(lau["atom_index"], np.int64)).mean())
        print(f"ORACLE[{budget}]/{which:3s} valid={rep['distinct_valid']:4d}/{rep['n']:<4d} "
              f"exact={rep['exact_match_frac']:.4f} relerr={err.mean():.4f} | "
              f"x0 valid={repa['distinct_valid']:4d} exact={repa['exact_match_frac']:.4f} "
              f"relerr={erra.mean():.4f} | index agreement={agree:.4f}", flush=True)
        res[f"relerr_{budget}_{which}"] = {"oracle_mean": float(err.mean()),
                                           "x0_mean": float(erra.mean()),
                                           "index_agreement": agree}
    # ⭐ the ceiling on the same rows, once
    if budget == "A":
        for which in ("val", "MS"):
            rows, X, lab = ctx["sets"][which]
            rep = score(ctx, X, lab, atoms); res[f"ceiling_{which}"] = rep
            print(f"CEILING /{which:3s}   valid={rep['distinct_valid']:4d}/{rep['n']:<4d} "
                  f"exact={rep['exact_match_frac']:.4f} relerr=0", flush=True)
json.dump(res, open(os.path.join(os.environ['OUT'], 'mech', 'oracle.json'), 'w'), indent=2)
print("done")
