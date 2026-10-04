"""Cached store construction + the matched-bytes ledger arithmetic.

⭐ The store (k-means PQ over the ward channel partition) costs ~40 s per corner and does
not depend on the head at all, so it is built once per (seed, corner, K, M, grouping) and
cached.  ⛔ Nothing cached here is a result; it is train-activation geometry plus k-means.

THE BYTE BUDGET, stated once and used everywhere
------------------------------------------------
    store  = M · D · 4                                   (independent of K)
    head   = ( n_add·K·M  +  n_pair·r  +  r·K·M ) · 4     n_add = 1 + Σ_f C_f
    budget = knn_proto's deployed ledger = (Σ_f C_f + 1) · D · 4 = 475,136 B
⭐ `max_rank` returns the LARGEST r that fits, so every candidate design is compared at
the SAME ceiling and ⛔ no design gets bytes another one was denied.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import comp
import ctx
import fast

CACHE = Path(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1)) / "dev" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def budget(cards, D: int) -> int:
    return comp.knn_proto_bytes(cards, D)


def max_rank(cards, pairs, K: int, M: int, D: int) -> int:
    """Largest rank whose TOTAL ledger still fits inside `knn_proto`'s."""
    if not pairs:
        return 0
    n_add = 1 + int(sum(int(x) for x in cards))
    _, n_pair = comp.pair_slices(cards, pairs)
    room = budget(cards, D) - comp.store_bytes_of(M, D) - n_add * int(K) * int(M) * 4
    if room <= 0:
        return -1
    return int(room // (4 * (n_pair + int(K) * int(M))))


def max_modes(cards, pairs, K: int, D: int, rank: int) -> int:
    """Largest M whose TOTAL ledger fits, at a fixed (K, pairs, rank)."""
    n_add = 1 + int(sum(int(x) for x in cards))
    _, n_pair = comp.pair_slices(cards, pairs)
    r = int(rank) if pairs else 0
    fixed = 4 * n_pair * r
    per_M = int(D) * 4 + 4 * (n_add * int(K) + r * int(K))
    return int((budget(cards, D) - fixed) // per_M)


def fits(cards, pairs, K, M, rank, D) -> bool:
    return comp.total_bytes(cards, pairs, K, M, rank, D) <= budget(cards, D)


# ---------------------------------------------------------------------------
def build(c, ci: int, pool, X_store, K: int, M: int, seed: int, *, log=ctx.log):
    """`(perm, modes)` for the DEV store on corner `ci`. Cached on disk."""
    D = int(np.prod(c["feat_shape"]))
    p = CACHE / f"store_s{seed}_c{ci}_K{K}_M{M}_n{int(np.asarray(pool).size)}.npz"
    if p.exists():
        z = np.load(p)
        return z["perm"], z["modes"]
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{seed}_c{ci}_n{int(np.asarray(pool).size)}", log=None)
    labels = fast.balanced_labels(ow, dead, int(K), D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(int(K))])
    pq = fast.kmeans_pq(np.asarray(X_store, np.float32).reshape((-1, D)),
                        labels, int(K), int(M), int(seed), iters=25)
    modes = np.asarray(pq.modes, np.float32)
    np.savez_compressed(p, perm=perm, modes=modes)
    log(f"  store K={K} M={M} built and cached -> {p.name}")
    return perm, modes


def to_gen_store(perm, modes, feat_shape):
    return fast.EqualPQ(np.asarray(perm), np.asarray(modes, np.float32),
                        (int(np.prod(feat_shape)),)).to_gen_store()
