"""RIVAL: the CONTINUOUS-INPUT MLP.  ⭐ The named falsifier of this hub's dial.

It takes the six requested factor values as RAW CONTINUOUS NUMBERS -- so an off-grid
hue is a perfectly ordinary input to it, with no bracketing, no blending and no store
-- and regresses the 2048-dim feature.  If it matches the arm on the second judge,
row 2 belongs to a generic learner and the store is decorative (DIAL DECLARATION).

Parameters are matched to the arm's deployed total (store bytes + head bytes) / 4
within +/-10%, and the count is PRINTED, per w4-hub-common section 1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parent))
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402


class ContMLP(eqx.Module):
    layers: list

    def __init__(self, key, n_in: int, hidden: int, n_out: int, depth: int = 2):
        ks = jax.random.split(key, depth + 1)
        dims = [n_in] + [hidden] * depth + [n_out]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i])
                       for i in range(depth + 1)]

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.gelu(lin(x))
        return self.layers[-1](x)


def n_params(m) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))))


def pick_hidden(target_params: int, n_in: int, n_out: int, depth: int = 2) -> int:
    best, bestd = 8, None
    for h in range(8, 2048):
        p = n_in * h + h + (depth - 1) * (h * h + h) + h * n_out + n_out
        d = abs(p - target_params)
        if bestd is None or d < bestd:
            best, bestd = h, d
    return best


def normalise(values: np.ndarray, grid) -> np.ndarray:
    """Raw continuous factor values -> [0, 1] by each factor's realised range."""
    V = np.asarray(values, np.float64).copy()
    for k in range(V.shape[1]):
        g = np.asarray(grid.values[k], np.float64)
        V[:, k] = (V[:, k] - g[0]) / max(g[-1] - g[0], 1e-12)
    return V.astype(np.float32)


def fit(grid, pool, F, target_params: int, seed: int = 0, steps: int = 4000,
        batch: int = 512, lr: float = 1e-3, log=print):
    X = normalise(grid.values_of(pool), grid)
    Y = W.gather(F, pool)
    h = pick_hidden(target_params, X.shape[1], Y.shape[1])
    model = ContMLP(jax.random.PRNGKey(seed), X.shape[1], h, Y.shape[1])
    npar = n_params(model)
    log(f"  cont_mlp: hidden={h} params={npar:,} target={target_params:,} "
        f"({100.0*npar/target_params:.1f}% of matched)")
    opt = optax.adam(lr)
    state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, st, xb, yb):
        def loss(mm):
            return jnp.mean((jax.vmap(mm)(xb) - yb) ** 2)
        lv, g = eqx.filter_value_and_grad(loss)(m)
        up, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, up), st, lv

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(steps):
        idx = rng.integers(0, X.shape[0], batch)
        model, state, lv = step(model, state, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
        if i % 500 == 0 or i == steps - 1:
            log(f"    step {i+1}/{steps} mse={float(lv):.5f} ({time.time()-t0:.0f}s)")
    return model, {"hidden": h, "n_params": npar, "target_params": int(target_params),
                   "pct_of_matched": float(100.0 * npar / target_params),
                   "steps": steps, "final_mse": float(lv)}


def predict(model, values: np.ndarray, grid, batch: int = 2048) -> np.ndarray:
    X = normalise(values, grid)
    f = eqx.filter_jit(jax.vmap(model))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + batch])), np.float32)
                           for i in range(0, X.shape[0], batch)])
