# ⭐ INHERITED VERBATIM from hub/generation-table (predecessor, iteration 1):
#   .claude/outputs/hub-generation-table/dev/fast.py
# ⛔ NOT rebuilt (LAUNCH-hub-composition-2: the instrument is inherited and reused).
# ⛔ Not one line below is modified. hub/composition-2 adds its columns in its own modules.
"""hub/generation-table — cached, instrumented DEV primitives.

⭐ Everything expensive that does NOT depend on K is computed once and cached to disk:
the channel correlation, the dead-channel mask, and the ward/average dendrogram leaf
orders.  ⛔ Nothing cached here is a result; it is all train-activation geometry.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import ctx

CACHE = Path(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1)) / "dev" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

_TIMES = {}


class timer:
    def __init__(self, name, log=None):
        self.name, self.log = name, log

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        dt = time.time() - self.t
        _TIMES[self.name] = _TIMES.get(self.name, 0.0) + dt
        if self.log:
            self.log(f"    [{self.name}] {dt:.1f}s")


def times():
    return dict(sorted(_TIMES.items(), key=lambda kv: -kv[1]))


def channel_geometry(X_store, cfg, tag: str, log=ctx.log):
    """``(corr, dead, leaf_order_ward, leaf_order_average)`` — cached on disk."""
    if log is None:
        log = lambda *_a, **_k: None
    p = CACHE / f"chan_geom_{tag}.npz"
    if p.exists():
        z = np.load(p)
        log(f"  channel geometry loaded from {p.name}")
        return z["corr"], z["dead"], z["order_ward"], z["order_avg"]
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    from chlu.core.gen_store import channel_correlation, dead_channels

    Xc = np.asarray(X_store, np.float32)[: int(cfg.gen_store.n_corr)]
    with timer("corr", log):
        corr = channel_correlation(Xc)
    with timer("dead", log):
        dead = dead_channels(Xc)
    live = np.flatnonzero(~dead)
    A = np.abs(np.asarray(corr, np.float64)[np.ix_(live, live)])
    with timer("linkage_ward", log):
        order_ward = np.asarray(leaves_list(linkage(A, method="ward")), np.int64)
    with timer("linkage_avg", log):
        dist = 1.0 - A
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        order_avg = np.asarray(leaves_list(linkage(squareform(dist, checks=False),
                                                   method="average")), np.int64)
    np.savez_compressed(p, corr=corr, dead=dead, order_ward=order_ward, order_avg=order_avg)
    log(f"  channel geometry computed and cached -> {p.name}")
    return corr, dead, order_ward, order_avg


def balanced_labels(order, dead, K: int, D: int) -> np.ndarray:
    """EXACTLY equal contiguous blocks of the channel ordering.

    The ordering is the live channels in dendrogram leaf order with the DEAD channels
    re-inserted at evenly spaced positions, so no single well is spent on constants.
    ⛔ Requires ``K`` to divide ``D`` — a well that is one channel wider than another
    would make the store non-uniform and the (K, M, d) layout impossible.
    """
    D = int(D); K = int(K)
    if D % K:
        raise ValueError(f"balanced_labels needs K | D; {K} does not divide {D}")
    dead = np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    liv = live[np.asarray(order, np.int64)]
    dch = np.flatnonzero(dead)
    ordering = np.empty(D, np.int64)
    if dch.size:
        # evenly spaced insertion slots for the dead channels
        slots = np.linspace(0, D - 1, dch.size).round().astype(np.int64)
        slots = np.unique(slots)
        while slots.size < dch.size:                      # pathological D/K; pad
            extra = np.setdiff1d(np.arange(D), slots)[: dch.size - slots.size]
            slots = np.sort(np.concatenate([slots, extra]))
        mask = np.zeros(D, bool)
        mask[slots[: dch.size]] = True
        ordering[mask] = dch
        ordering[~mask] = liv
    else:
        ordering[:] = liv
    labels = np.empty(D, np.int64)
    labels[ordering] = np.repeat(np.arange(K), D // K)
    return labels


def ward_labels(corr, dead, K: int, log=None):
    """The SHIPPED grouping, for the like-for-like column."""
    from chlu.core.gen_store import correlation_grouping

    return correlation_grouping(corr, int(K), "auto", dead=dead, min_well_frac=0.10,
                                return_info=True)


# ---------------------------------------------------------------------------
# A vectorised equal-well product quantiser.  ⭐ Numerically the SAME object as
# GenStore; the layout is (K, d) so K=128 is one batched op, not 128 scatters.
# ---------------------------------------------------------------------------

class EqualPQ:
    """``modes (K, M, d)`` over an equal-size channel partition ``perm (K, d)``.

    ⛔ Not a replacement for :class:`chlu.core.gen_store.GenStore` — it is converted
    to one (:meth:`to_gen_store`) before any read that is scored, so every scored
    number goes through the shipped ``read_from_codes`` and the shipped
    ``assert_no_bypass``.
    """

    def __init__(self, perm: np.ndarray, modes: np.ndarray, feat_shape):
        self.perm = np.asarray(perm, np.int64)        # (K, d) channel ids
        self.modes = np.asarray(modes, np.float32)    # (K, M, d)
        self.feat_shape = tuple(int(x) for x in feat_shape)
        self.D = int(np.prod(self.feat_shape))

    @property
    def K(self):
        return int(self.perm.shape[0])

    @property
    def M(self):
        return int(self.modes.shape[1])

    @property
    def d(self):
        return int(self.perm.shape[1])

    def bytes(self) -> int:
        return int(self.modes.size * 4)

    def views(self, X):
        Z = np.asarray(X, np.float32).reshape((-1, self.D))
        return Z[:, self.perm]                        # (n, K, d)

    def codes(self, X, batch: int = 8192) -> np.ndarray:
        out = np.empty((int(np.asarray(X).shape[0]), self.K), np.int64)
        mm = np.einsum("kmd,kmd->km", self.modes, self.modes)     # (K, M)
        for i in range(0, out.shape[0], batch):
            V = self.views(np.asarray(X)[i: i + batch])           # (b, K, d)
            # d2 = ||m||^2 - 2 v.m   (||v||^2 dropped: constant in m)
            cross = np.einsum("bkd,kmd->bkm", V, self.modes)
            out[i: i + batch] = np.argmin(mm[None] - 2.0 * cross, axis=-1)
        return out

    def read(self, codes) -> np.ndarray:
        C = np.asarray(codes, np.int64)
        n = int(C.shape[0])
        out = np.zeros((n, self.D), np.float32)
        gathered = self.modes[np.arange(self.K)[None, :], C]      # (n, K, d)
        out[:, self.perm.reshape(-1)] = gathered.reshape((n, -1))
        return out.reshape((n,) + self.feat_shape)

    def to_gen_store(self):
        from chlu.core.gen_store import GenStore
        import jax

        wc = tuple(tuple(int(i) for i in row) for row in self.perm)
        init = [self.modes[k] for k in range(self.K)]
        return GenStore(wc, self.M, self.feat_shape, jax.random.PRNGKey(0), init=init)


def kmeans_pq(X, labels: np.ndarray, K: int, M: int, seed: int, *, iters: int = 25,
              log=None, batch: int = 8192) -> EqualPQ:
    """k-means++ init + Lloyd iterations per well.  ⭐ Label-free: only ``X``.

    ⛔ Deliberately NOT the shipped ``train_store`` (soft/commitment Adam objective) —
    this is plain Lloyd, which is the exact minimiser of the HARD read's distortion
    and therefore the honest strongest form of "the codebook is as good as it gets".
    The shipped trainer is run beside it for the like-for-like column.
    """
    from chlu.core.gen_store import _kmeanspp_init

    lab = np.asarray(labels, np.int64)
    D = lab.size
    sizes = np.bincount(lab, minlength=int(K))
    d = int(sizes.max())
    if int(sizes.min()) != d:
        raise ValueError(f"kmeans_pq needs equal wells; got sizes {sizes.tolist()}")
    perm = np.stack([np.flatnonzero(lab == k) for k in range(int(K))])   # (K, d)
    Z = np.asarray(X, np.float32).reshape((-1, D))
    rng = np.random.default_rng(int(seed) * 2_654_435_761 % (2 ** 32))
    modes = np.empty((int(K), int(M), d), np.float32)
    for k in range(int(K)):
        modes[k] = _kmeanspp_init(Z[:, perm[k]], int(M), rng)
    pq = EqualPQ(perm, modes, (D,))
    V = Z[:, perm]                                                       # (n, K, d)
    for it in range(int(iters)):
        mm = np.einsum("kmd,kmd->km", pq.modes, pq.modes)
        moved = 0
        newm = np.zeros_like(pq.modes)
        cnt = np.zeros((int(K), int(M)), np.int64)
        for i in range(0, V.shape[0], batch):
            Vb = V[i: i + batch]
            cross = np.einsum("bkd,kmd->bkm", Vb, pq.modes)
            c = np.argmin(mm[None] - 2.0 * cross, axis=-1)               # (b, K)
            for k in range(int(K)):
                np.add.at(newm[k], c[:, k], Vb[:, k, :])
                cnt[k] += np.bincount(c[:, k], minlength=int(M))
        empty = cnt == 0
        cnt_safe = np.maximum(cnt, 1)[:, :, None]
        newm /= cnt_safe
        newm[empty] = pq.modes[empty]     # ⛔ an empty mode keeps its position
        moved = float(np.mean(np.abs(newm - pq.modes)))
        pq.modes = newm.astype(np.float32)
        if log is not None and (it == 0 or (it + 1) % 5 == 0):
            log(f"    lloyd {it+1}/{iters} mean|Δmode|={moved:.6f} empty={int(empty.sum())}")
        if moved < 1e-7:
            break
    return pq


# ---------------------------------------------------------------------------
# ⭐ WEIGHTED product quantisation.  Same stored object, different DISTORTION.
# ---------------------------------------------------------------------------

def weighted_codes(pq, X, w=None, batch: int = 8192) -> np.ndarray:
    """argmin_m  sum_d w_d (v_d - m_md)^2   per well.

    ⭐ With ``w = 1`` this is exactly ``gen_store.address_codes``' argmin (the dropped
    ||v||^2 term is constant in m).  ⛔ Used only to FIT and to build the head's
    targets; the scored read is always ``gen_store.read_from_codes``.
    """
    K, M, d = pq.modes.shape
    if w is None:
        Wk = np.ones((K, d), np.float32)
    else:
        Wk = np.asarray(w, np.float32).reshape(-1)[pq.perm]          # (K, d)
    mmw = np.einsum("kd,kmd,kmd->km", Wk, pq.modes, pq.modes)        # (K, M)
    n = int(np.asarray(X).shape[0])
    out = np.empty((n, K), np.int64)
    for i in range(0, n, batch):
        V = pq.views(np.asarray(X)[i: i + batch])                    # (b, K, d)
        VW = V * Wk[None]
        cross = np.einsum("bkd,kmd->bkm", VW, pq.modes)
        out[i: i + batch] = np.argmin(mmw[None] - 2.0 * cross, axis=-1)
    return out


def kmeans_pq_weighted(X, labels: np.ndarray, K: int, M: int, seed: int, w=None, *,
                       iters: int = 25, log=None, batch: int = 8192) -> EqualPQ:
    """Lloyd with a per-coordinate weighted ASSIGNMENT.

    ⭐ The centroid UPDATE is unchanged and is still the plain mean: for a
    per-coordinate (not per-sample) weight, ``d/dm_d sum_i w_d (v_id - m_d)^2 = 0``
    gives ``m_d = mean_i v_id`` whatever ``w_d`` is.  Only which rows land in which
    mode changes.  ⛔ Therefore the stored object is bit-for-bit the same KIND of
    object and ``store_bytes = M*D*4`` is untouched.
    """
    from chlu.core.gen_store import _kmeanspp_init

    lab = np.asarray(labels, np.int64)
    D = lab.size
    sizes = np.bincount(lab, minlength=int(K))
    d = int(sizes.max())
    if int(sizes.min()) != d:
        raise ValueError(f"kmeans_pq_weighted needs equal wells; got {sizes.tolist()}")
    perm = np.stack([np.flatnonzero(lab == k) for k in range(int(K))])
    Z = np.asarray(X, np.float32).reshape((-1, D))
    rng = np.random.default_rng(int(seed) * 2_654_435_761 % (2 ** 32))
    modes = np.empty((int(K), int(M), d), np.float32)
    for k in range(int(K)):
        modes[k] = _kmeanspp_init(Z[:, perm[k]], int(M), rng)
    pq = EqualPQ(perm, modes, (D,))
    Wk = (np.ones((int(K), d), np.float32) if w is None
          else np.asarray(w, np.float32).reshape(-1)[perm])
    V = Z[:, perm]
    for it in range(int(iters)):
        mmw = np.einsum("kd,kmd,kmd->km", Wk, pq.modes, pq.modes)
        newm = np.zeros_like(pq.modes)
        cnt = np.zeros((int(K), int(M)), np.int64)
        for i in range(0, V.shape[0], batch):
            Vb = V[i: i + batch]
            cross = np.einsum("bkd,kmd->bkm", Vb * Wk[None], pq.modes)
            c = np.argmin(mmw[None] - 2.0 * cross, axis=-1)
            for k in range(int(K)):
                np.add.at(newm[k], c[:, k], Vb[:, k, :])
                cnt[k] += np.bincount(c[:, k], minlength=int(M))
        empty = cnt == 0
        newm /= np.maximum(cnt, 1)[:, :, None]
        newm[empty] = pq.modes[empty]
        moved = float(np.mean(np.abs(newm - pq.modes)))
        pq.modes = newm.astype(np.float32)
        if log is not None and (it == 0 or (it + 1) % 5 == 0):
            log(f"    wlloyd {it+1}/{iters} mean|dmode|={moved:.6f} empty={int(empty.sum())}")
        if moved < 1e-7:
            break
    return pq


# ---------------------------------------------------------------------------
# ⭐⭐ METRIC product quantisation: a FULL per-well metric A_k, not its diagonal.
# DEV-07 showed the codebook's residual is low-dimensional and aligned with the
# directions the decoder reads; a diagonal weight cannot see those directions.
# The Lloyd UPDATE is unchanged (the mean minimises  sum_i (v-m) A (v-m)^T  for any
# fixed positive-definite A), so only the ASSIGNMENT changes and ⛔ nothing extra is
# stored: store_bytes = M*D*4 exactly as before, and the metric is never deployed
# (at read time the codes come from the request, not from a feature).
# ---------------------------------------------------------------------------

def decoder_jacobian_samples(decoder, X, feat_shape, n_probe=8, batch=32, seed=0,
                             log=None):
    """Samples ``g = J^T u``, ``u ~ N(0, I)`` in image space.  ⭐ ``cov(g) = J^T J``."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    fs = tuple(int(x) for x in feat_shape)
    D = int(np.prod(fs))
    Z = np.asarray(X, np.float32).reshape((-1,) + fs)
    key = jax.random.PRNGKey(int(seed))
    params, static = eqx.partition(decoder, eqx.is_inexact_array)

    def fwd(z):
        return jax.vmap(eqx.combine(params, static))(z)

    out = []
    for i in range(0, Z.shape[0], int(batch)):
        zb = jnp.asarray(Z[i:i + int(batch)])
        y, vjp = jax.vjp(fwd, zb)
        for _ in range(int(n_probe)):
            key, sk = jax.random.split(key)
            u = jax.random.normal(sk, y.shape, y.dtype)
            (g,) = vjp(u)
            out.append(np.asarray(g, np.float32).reshape((zb.shape[0], D)))
        if log is not None:
            log(f"    jacobian samples {min(i+batch, Z.shape[0])}/{Z.shape[0]}")
    return np.concatenate(out, axis=0)


def well_metrics(G, perm, ridge: float = 1e-3):
    """``A_k = cov(G[:, well k]) + ridge*trace/d * I``, normalised to unit mean trace."""
    K, d = perm.shape
    A = np.empty((K, d, d), np.float64)
    for k in range(K):
        Gk = np.asarray(G[:, perm[k]], np.float64)
        Ak = (Gk.T @ Gk) / max(Gk.shape[0], 1)
        tr = float(np.trace(Ak)) / d
        Ak = Ak + float(ridge) * tr * np.eye(d)
        A[k] = Ak / max(tr, 1e-30)
    return A


def kmeans_pq_metric(X, labels: np.ndarray, K: int, M: int, seed: int, A=None, *,
                     iters: int = 25, log=None, batch: int = 8192) -> EqualPQ:
    """Lloyd under a full per-well metric ``A (K, d, d)``.  ``A=None`` -> Euclidean."""
    from chlu.core.gen_store import _kmeanspp_init

    lab = np.asarray(labels, np.int64)
    D = lab.size
    sizes = np.bincount(lab, minlength=int(K))
    d = int(sizes.max())
    if int(sizes.min()) != d:
        raise ValueError(f"kmeans_pq_metric needs equal wells; got {sizes.tolist()}")
    perm = np.stack([np.flatnonzero(lab == k) for k in range(int(K))])
    Z = np.asarray(X, np.float32).reshape((-1, D))
    rng = np.random.default_rng(int(seed) * 2_654_435_761 % (2 ** 32))
    modes = np.empty((int(K), int(M), d), np.float32)
    for k in range(int(K)):
        modes[k] = _kmeanspp_init(Z[:, perm[k]], int(M), rng)
    pq = EqualPQ(perm, modes, (D,))
    V = Z[:, perm].astype(np.float32)                                # (n, K, d)
    if A is None:
        VA = V
    else:
        Af = np.asarray(A, np.float32)
        VA = np.einsum("bkd,kde->bke", V, Af)                        # (n, K, d)
    for it in range(int(iters)):
        if A is None:
            mm = np.einsum("kmd,kmd->km", pq.modes, pq.modes)
        else:
            mA = np.einsum("kmd,kde->kme", pq.modes, np.asarray(A, np.float32))
            mm = np.einsum("kme,kme->km", mA, pq.modes)
        newm = np.zeros((int(K), int(M), d), np.float64)
        cnt = np.zeros((int(K), int(M)), np.int64)
        for i in range(0, V.shape[0], batch):
            cross = np.einsum("bkd,kmd->bkm", VA[i:i + batch], pq.modes)
            c = np.argmin(mm[None] - 2.0 * cross, axis=-1)
            Vb = V[i:i + batch]
            for k in range(int(K)):
                np.add.at(newm[k], c[:, k], Vb[:, k, :])
                cnt[k] += np.bincount(c[:, k], minlength=int(M))
        empty = cnt == 0
        newm /= np.maximum(cnt, 1)[:, :, None]
        newm = newm.astype(np.float32)
        newm[empty] = pq.modes[empty]
        moved = float(np.mean(np.abs(newm - pq.modes)))
        pq.modes = newm
        if log is not None and (it == 0 or (it + 1) % 5 == 0):
            log(f"    mlloyd {it+1}/{iters} mean|dmode|={moved:.6f} empty={int(empty.sum())}")
        if moved < 1e-7:
            break
    return pq


def metric_codes(pq, X, A=None, batch: int = 8192) -> np.ndarray:
    """argmin_m (v-m) A_k (v-m)^T per well.  ``A=None`` -> ``gen_store.address_codes``."""
    K, M, d = pq.modes.shape
    if A is None:
        mm = np.einsum("kmd,kmd->km", pq.modes, pq.modes)
    else:
        mA = np.einsum("kmd,kde->kme", pq.modes, np.asarray(A, np.float32))
        mm = np.einsum("kme,kme->km", mA, pq.modes)
    n = int(np.asarray(X).shape[0])
    out = np.empty((n, K), np.int64)
    for i in range(0, n, batch):
        V = pq.views(np.asarray(X)[i:i + batch]).astype(np.float32)
        VA = V if A is None else np.einsum("bkd,kde->bke", V, np.asarray(A, np.float32))
        cross = np.einsum("bkd,kmd->bkm", VA, pq.modes)
        out[i:i + batch] = np.argmin(mm[None] - 2.0 * cross, axis=-1)
    return out
