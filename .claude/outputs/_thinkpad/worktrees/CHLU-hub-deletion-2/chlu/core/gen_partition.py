"""⭐ Channel partitions for the generation store, at a chosen number of wells.

⛔ **Why a second partition function exists at all.**
:func:`chlu.core.gen_store.correlation_grouping` is the shipped one and is
**unchanged**; this module is additive.  It exists because the shipped one returns
**unequal** well sizes (ward on the shipped cache: ``[303, 305, 573, 501, 366]``), and
two things the hub needs are easier on an **equal** partition:

1. ⭐ A per-well allocation argument: with equal ``d_k`` the store's byte identity
   ``store_bytes = M·D·4`` holds *per well as well as in total*, so "same bytes, more
   wells" is visible without an allocation table.
2. ⚠ Engineering: with equal ``d_k`` the whole product quantiser is one ``(K, d)``
   tensor, so a ``K = 64`` store is one batched op rather than 64 unrolled scatters.
   At ``K = 5`` that is a convenience; at ``K = 128`` the unrolled form is the run.

⭐ **The correlation structure is kept, not thrown away.**  The inherited result is that
the ward/correlation grouping already beats a D-matched random floor by ~50×, so this
module does **not** invent a new criterion.  It reuses the *same* ward linkage on the
*same* ``|corr|`` matrix and simply cuts the **dendrogram leaf order** into equal
contiguous blocks, instead of cutting the tree at ``K`` clusters.  Correlated channels
are adjacent in leaf order, so each block is still a correlation-local group.

⛔ Nothing here sees a label, a request, a held-out row or an arm outcome.  The input is
the train-activation correlation matrix and nothing else — the same input, from the same
rows, that the shipped grouping takes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["balanced_correlation_partition", "partition_report"]


def balanced_correlation_partition(
    corr: np.ndarray,
    k: int,
    dead: Optional[np.ndarray] = None,
    *,
    method: str = "ward",
) -> tuple:
    """Partition ``D`` coordinates into ``k`` wells of (near-)equal size.

    The live coordinates are ordered by the dendrogram leaf order of the **same**
    linkage :func:`chlu.core.gen_store._cluster_live` uses, then cut into ``k``
    contiguous blocks whose sizes differ by at most one.  Dead coordinates are then
    placed into the smallest well, exactly as the shipped grouping does.

    Returns:
        ``(labels (D,), info)``.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    C = np.asarray(corr, np.float64)
    d = int(C.shape[0])
    dead = np.zeros(d, bool) if dead is None else np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    if live.size < int(k):
        raise ValueError(f"only {live.size} live coordinates for {k} wells")
    A = np.abs(C[np.ix_(live, live)])
    if method == "ward":
        Z = linkage(A, method="ward")
    elif method == "average":
        from scipy.spatial.distance import squareform

        dist = 1.0 - A
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        Z = linkage(squareform(dist, checks=False), method="average")
    else:
        raise ValueError(f"method must be 'ward' or 'average', got {method!r}")
    order = np.asarray(leaves_list(Z), np.int64)

    labels = np.zeros(d, np.int64)
    # near-equal contiguous blocks in leaf order: sizes differ by at most one
    base, rem = divmod(int(live.size), int(k))
    sizes = np.full(int(k), base, np.int64)
    sizes[: int(rem)] += 1
    start = 0
    for j, s in enumerate(sizes):
        labels[live[order[start: start + int(s)]]] = j
        start += int(s)

    counts = np.bincount(labels[live], minlength=int(k)).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(counts))
        labels[c] = j
        counts[j] += 1
    info = {
        "partition": "balanced_correlation (dendrogram leaf order, equal blocks)",
        "linkage_method": method,
        "k": int(k),
        "n_live_channels": int(live.size),
        "n_dead_channels": int(dead.sum()),
        "live_block_sizes": [int(x) for x in sizes],
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=int(k))],
        "equal_sized_live_blocks": bool(sizes.max() - sizes.min() <= 1),
        "note": (
            "⭐ same |corr| and same linkage as the shipped grouping; ⛔ the tree is cut "
            "by leaf ORDER into equal blocks rather than at k clusters"
        ),
    }
    return labels, info


def partition_report(labels: np.ndarray, corr: np.ndarray) -> dict:
    """⭐ How block-diagonal is this partition?  ⛔ Reported, never used to select.

    Reuses :func:`chlu.core.gen_store.block_diagonality` so the number is the same
    statistic the shipped grouping is described by.
    """
    from .gen_store import block_diagonality

    return block_diagonality(np.asarray(corr, np.float64), np.asarray(labels, np.int64))
