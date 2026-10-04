"""⭐ How much does the FROZEN decoder care about each feature coordinate?

⛔ **Why this exists.**  The shipped store is fitted to minimise **feature-space**
distortion (``gen_store.train_store``: *"no labels, no decoder, no MSE against anything
but the store's own read"*).  But the number the generation table reads is a classifier's
accuracy on the **decoded image**, and the map from feature to image is the frozen
decoder.  A quantiser that spends its codewords where the *features* vary is spending
them where the *image* may not vary at all — and vice versa.

For a read ``x̂ = x + δ`` the decoded error is, to first order,
``‖dec(x̂) − dec(x)‖² ≈ δᵀ JᵀJ δ`` with ``J = ∂dec/∂x``.  This module estimates the
**diagonal** of ``JᵀJ`` by Hutchinson probing::

    E_u[ (Jᵀu)_d² ] = (JᵀJ)_dd      for  u ~ N(0, I) in image space

so ``n_probe`` reverse-mode passes give an unbiased per-coordinate sensitivity.  Those
are then used as **weights in the codebook's distortion**, which is a change to *how the
codebook is fitted* and ⛔ **not** a change to what is stored: the modes are still
``M`` float32 vectors per well, so ``store_bytes = M·D·4`` is untouched to the byte.

⛔ **Label-free.**  Nothing here sees a factor label, a request, a held-out row or an arm
outcome — only the frozen decoder and train-split features.  Charter §B.6 is unaffected.
"""
from __future__ import annotations

import numpy as np

__all__ = ["decoder_sensitivity", "sensitivity_report"]


def decoder_sensitivity(decoder, X, feat_shape, *, n_probe: int = 16, batch: int = 32,
                        seed: int = 0, log=None) -> dict:
    """Per-coordinate ``diag(JᵀJ)`` of the frozen decoder, averaged over ``X``.

    Args:
        decoder: the frozen decoder (``chlu.eval.gen_probe.decode_features``' model).
        X: ``(n, D)`` or ``(n,) + feat_shape`` features to probe at.
        feat_shape: the decoder's input shape.
        n_probe: Hutchinson probes per batch.  ⭐ The estimator is unbiased at any
            ``n_probe``; more only reduces variance.

    Returns:
        ``{"w": (D,), "n_rows", "n_probe", ...}`` — ⛔ ``w`` is a *fitting weight*, never
        a stored quantity.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    fs = tuple(int(x) for x in feat_shape)
    D = int(np.prod(fs))
    Z = np.asarray(X, np.float32).reshape((-1,) + fs)
    n = int(Z.shape[0])
    key = jax.random.PRNGKey(int(seed))

    params, static = eqx.partition(decoder, eqx.is_inexact_array)

    def fwd(z):
        # ⭐ exactly ``gen_probe.decode_features``' forward: ``jax.vmap(decoder)``.
        return jax.vmap(eqx.combine(params, static))(z)

    acc = np.zeros(D, np.float64)
    nb = 0
    for i in range(0, n, int(batch)):
        zb = jnp.asarray(Z[i: i + int(batch)])
        out, vjp = jax.vjp(fwd, zb)
        for _ in range(int(n_probe)):
            key, sk = jax.random.split(key)
            u = jax.random.normal(sk, out.shape, out.dtype)
            (g,) = vjp(u)
            acc += np.asarray(jnp.sum(jnp.square(g).reshape((zb.shape[0], -1)),
                                      axis=0), np.float64)
        nb += int(zb.shape[0]) * int(n_probe)
        if log is not None:
            log(f"    sensitivity: {min(i + batch, n)}/{n} rows")
    w = acc / max(nb, 1)
    return {
        "w": w.astype(np.float64),
        "n_rows": n,
        "n_probe": int(n_probe),
        "estimator": "Hutchinson diag(J^T J), reverse-mode, frozen decoder",
        "label_free": True,
        "stored": False,
        "note": ("⭐ a FITTING weight for the codebook's distortion; ⛔ not stored, "
                 "so store_bytes = M·D·4 is unchanged"),
    }


def sensitivity_report(w: np.ndarray, X=None) -> dict:
    """Summary numbers, so the weighting is described rather than asserted."""
    w = np.asarray(w, np.float64)
    out = {
        "D": int(w.size),
        "w_min": float(w.min()),
        "w_max": float(w.max()),
        "w_mean": float(w.mean()),
        "w_median": float(np.median(w)),
        "dynamic_range": float(w.max() / max(w.min(), 1e-300)),
        "frac_mass_in_top_10pct": float(
            np.sort(w)[::-1][: max(1, w.size // 10)].sum() / max(w.sum(), 1e-300)),
    }
    if X is not None:
        v = np.var(np.asarray(X, np.float64).reshape((int(np.asarray(X).shape[0]), -1)),
                   axis=0)
        out["corr_log_w_log_var"] = float(
            np.corrcoef(np.log(np.maximum(w, 1e-30)), np.log(np.maximum(v, 1e-30)))[0, 1])
        out["note_corr"] = (
            "⭐ if this is near 1 the decoder cares about exactly what varies and a "
            "weighted codebook cannot help; ⛔ if it is low, the shipped unweighted "
            "fit is spending codewords in the wrong place")
    return out
