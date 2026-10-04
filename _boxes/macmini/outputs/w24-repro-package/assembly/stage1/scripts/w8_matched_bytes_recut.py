"""w8 hub/single-design -- the MATCHED-BYTES re-cut.  ⛔ DEV only, NOT a scored run.

The single design carries both properties at the incumbent's own K=5 partition, but at
**12.787x the 524,288 B budget** -- so rows 3-4 against ``pca_tensor`` would not be
matched-bytes and my target file pre-commits me to print that rather than net it.

This script asks whether there is a partition at which the honest ledger
(atoms + float64 second-moment triangle + first moment + count) fits the pin.  The
criterion is NOT re-invented: it is ``factor_aligned_balanced_labels`` -- the SAME
inherited label-MI criterion with the SAME merge rule -- evaluated at a finer K.
``mi_bits`` and ``dead`` are read off the realised ``partitions.npz``, so nothing is
re-fitted and no held-out row is touched.

⛔ Charter note: charter §B's closing sentence lists **partition** among the things a hub
may change.  "Placement exhausted" is about LEARNED well PLACEMENT returning a null
against inherited grouping; re-cutting granularity with the inherited criterion is a
different dial.  The placement RULE is unchanged; only K moves.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
D = 2048
def ward_balanced_labels(corr, K, dead):
    """Equal wells at ANY K by cutting the correlation dendrogram's leaf order.

    ⭐ Reproduced verbatim from ``exp_gen_factor_wells.py:594-608`` -- the shipped
    ``ward_balanced`` construction -- so this is an inherited partition evaluated at a
    finer K, not a new one.  ⭐⭐ It sees NO LABEL: the single design built on it is
    strictly MORE label-free than the incumbent, whose partition is label-MI.
    """
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
        lab[order[pos:pos + int(c)]] = j
        pos += int(c)
    cnt = np.bincount(lab[live_idx], minlength=K).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(cnt))
        lab[c] = j
        cnt[j] += 1
    return lab


