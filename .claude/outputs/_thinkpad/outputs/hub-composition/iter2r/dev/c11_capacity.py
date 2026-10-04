"""C11 — ⭐ THE CAPACITY LAW **FOR THIS READ**. ⛔ DEV ONLY; `eval_b` is never opened.

    python c11_capacity.py <corner,corner,...>

WHY
---
R2 requires the capacity law reported beside the headline. The inherited law
`rel-err = e^1.346 · bytes^−0.1651` (R² 0.9960) was fitted to the **argmax quantiser's
reconstruction residual**, and iteration 1's "77 GB / 147,000×" headline is a law about
*that read*. ⛔ **This object does not use that read**, and at the shipped bytes the argmax
read renders 0 tuples where the linear read renders 42 on B. ⇒ quoting the inherited law as a
bound on this object would be wrong, and saying so is not enough — the law for **this** read
has to be measured.

WHAT IS VARIED
--------------
K = 32 and the reference pair design / rank / ridge are **held fixed**; only **M** moves, so
the only thing changing is how many atoms each well carries:
    store = M·D·4          head = (n_add·K·M + n_pair·r + r·K·M)·4
    total = 17,664·M + 10,240 bytes
⚠ **The legs above M = 26 are DECLARED OVER BUDGET** and are ⛔ never quoted as matched-bytes
rows — they exist only to fit the curve and to say what capacity would buy.
⭐ The decoder ceiling is measured on the same corners, so the curve has a target.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp  # noqa: E402
import ctx  # noqa: E402
import dev  # noqa: E402
import fast  # noqa: E402
import rivals  # noqa: E402
import stores  # noqa: E402

CORNERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1
                            else "0,2,4,6,12,16,20,24,38,44,50,56").split(",")]
SEED = 0
K, RANK, RIDGE = 32, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
M_GRID = [4, 8, 13, 26, 52, 104]
TAG = "cap"

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
BUD = comp.knn_proto_bytes(cards, D)
for M in M_GRID:
    tb = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
    ctx.log(f"  M={M:4d}  bytes={tb:10,d}  ({tb / BUD:6.3f} x knn_proto)"
            f"{'' if tb <= BUD else '   ⚠ DECLARED OVER BUDGET — curve only'}")

for ci in CORNERS:
    try:
        s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        ctx.log(f"[c{ci}] ⛔ REFUSED: {e}")
        continue
    todo = [M for M in M_GRID if not rivals.ledger_has(TAG, ci, f"M{M}")]
    need_ceiling = not rivals.ledger_has(TAG, ci, "ceiling")
    if not todo and not need_ceiling:
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    req = requests_from_labels(lab_dev, "full", 0, SEED)
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    labels = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])

    if need_ceiling:
        h = comp.headline(c, np.asarray(c["gather"](s["dev_rows"]), np.float32).reshape(
            (-1,) + tuple(c["feat_shape"])), lab_dev, pool_tup)
        rivals.ledger_put({"tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci],
                           "seed": SEED, "arm": "ceiling", "bytes": 0,
                           "bytes_vs_knn_proto": 0.0,
                           "config": {"K": K, "rank": RANK, "ridge": RIDGE},
                           "note": "⛔ NOT AN ARM — the decoder ceiling, the curve's target",
                           "headline": h})
        ctx.log(f"[c{ci}] ceiling = {h['tuples_valid_novel']}")

    for M in todo:
        tp = time.time()
        pq = fast.kmeans_pq(Zs, labels, K, int(M), SEED, iters=25)
        st = fast.EqualPQ(perm, pq.modes.astype(np.float32), (D,)).to_gen_store()
        head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
        feats = np.asarray(clu_wide_vector(st, head.coefficients(req)), np.float32)
        h = comp.headline(c, feats, lab_dev, pool_tup)
        # the LINEAR read's own reconstruction error on the held-out corner
        Xd = np.asarray(c["gather"](s["dev_rows"]), np.float32).reshape((-1, D))
        F = feats.reshape((-1, D))
        rel = float(np.linalg.norm(F - Xd) / max(float(np.linalg.norm(Xd)), 1e-30))
        tb = comp.total_bytes(cards, PAIRS, K, int(M), RANK, D)
        rivals.ledger_put({
            "tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci], "seed": SEED,
            "arm": f"M{M}", "bytes": int(tb), "bytes_vs_knn_proto": tb / BUD,
            "config": {"K": K, "M": int(M), "rank": RANK, "ridge": RIDGE,
                       "pairs": PAIR_NAMES, "over_budget": bool(tb > BUD)},
            "note": "C11 capacity curve for the LINEAR read"
                    + ("  ⚠ DECLARED OVER BUDGET — curve only" if tb > BUD else ""),
            "extra": {"linear_read_rel_err_on_the_held_out_corner": rel},
            "headline": h})
        ctx.log(f"[c{ci}] M={M:4d} bytes={tb:9,d} ({tb / BUD:5.2f}x)  "
                f"tuples={h['tuples_valid_novel']:3d}  rel_err={rel:.4f}  "
                f"({time.time() - tp:.0f}s)")
    ctx.log(f"[c{ci}] done in {time.time() - t0:.1f}s")

ctx.log(f"C11 done in {time.time() - T0:.1f}s")
