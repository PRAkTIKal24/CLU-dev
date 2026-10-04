"""shapes matched bytes."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
D = 2048


def ward_balanced_labels(corr, K, dead):
    from scipy.cluster.hierarchy import leaves_list, linkage

    live_idx = np.flatnonzero(~dead)
    Zl = linkage(np.abs(corr[np.ix_(live_idx, live_idx)]), method="ward")
    order = live_idx[leaves_list(Zl)]
    lab = np.zeros(D, np.int64)
    base = order.size // K
    cap = np.full(K, base, np.int64)
    cap[: order.size - base * K] += 1
    pos = 0
    for j, c in enumerate(cap):
        lab[order[pos : pos + int(c)]] = j
        pos += int(c)
    cnt = np.bincount(lab[live_idx], minlength=K).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(cnt))
        lab[c] = j
        cnt[j] += 1
    return lab
