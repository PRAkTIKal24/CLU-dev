"""w12 hub/composition-ladder -- AMENDMENT 1: the byte-matched learners, fitted by THIS hub.

Two learned arms per leg, both `mlp` (one-hot request -> 2048-d feature), both fitted on
**that rung's own fit pool** so the population caveat that hangs over spoke A's rival set
(`fit_population = montero_combination`) does not hang over these:

  learned_ref_ownpool   131,072 parameters  = the 524,288 B REFERENCE pin
  learned_1x            the MEASURED total bytes of `single_design_K5` on that leg / 4
                        (atoms + sufficient statistics + address book, never netted)

Width comes from `w4_learners.match_width`, which measures the built model rather than a
formula and REFUSES outside its own +-10% band -- a refusal is reported as the result.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chlu
from pathlib import Path as _P
if _P(__file__).resolve().parents[1] not in _P(chlu.__file__).resolve().parents:
    raise AssertionError("the package resolves outside this checkout: %s" % chlu.__file__)
from chlu.experiments import exp_gen_factor_wells as fw
from chlu.eval.rivals import w4_learners as WL
T0 = time.time()
def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


N_FIT_ROWS = 32768          # = exp_gen_factor_wells.PINS['n_store_train']
EPOCHS = 20
LR = 1e-3
WD = 0.0
def fit_one(target_params: int, pool: np.ndarray, grid, seed: int, tag: str) -> dict:
    import jax
    cards = [int(c) for c in grid.cardinalities]
    build, kind = WL.builders(cards, int(fw.PINS["D"]),
                              jax.random.PRNGKey(int(seed)))["mlp"]
    w, npar, err = WL.match_width(build, int(target_params), lo=1, hi=8192, log=log)
    model = build(w)
    rows = np.asarray(pool, np.int64)[:N_FIT_ROWS]
    Y = fw.gather(fw.features_mmap(int(seed)), rows)
    cls = grid.factor_indices_of(rows)
    Xin = WL.onehot_inputs(cls, cards)
    t = time.time()
    model, info = WL.train(model, Xin, Y, jax.random.PRNGKey(int(seed) + 1),
                           epochs=EPOCHS, lr=LR, wd=WD, batch=WL.BATCH, log=log,
                           rng=np.random.default_rng(int(seed)))
    del Y, Xin
    log("    %s: width %d, %d params (%.2f%% of target), fitted in %.1fs"
        % (tag, w, npar, 100.0 * err, time.time() - t))

    gv = [np.asarray(v, np.float64) for v in grid.values]

    def fn(values: np.ndarray):
        V = np.asarray(values, np.float64)
        idx = np.empty(V.shape, np.int64)
        oor = np.zeros(V.shape[0], bool)
        for k in range(V.shape[1]):
            g = gv[k]
            idx[:, k] = np.argmin(np.abs(V[:, k, None] - g[None, :]), axis=1)
            oor |= (V[:, k] < g[0]) | (V[:, k] > g[-1])
        X = WL.onehot_inputs(idx, cards)
        return WL.predict(model, X), oor

    return {"fn": fn, "width": int(w), "param_count": int(npar),
            "param_err_frac": float(err), "train_info": info,
            "n_fit_rows": int(rows.size), "fit_pool": "this rung's own pool",
            "target_params": int(target_params)}


