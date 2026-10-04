"""C07 — THE (K, M) STORE SWEEP at matched bytes.  ⛔ SELECTION ONLY.

    python c07_store_km.py <tag> <seed> <corner,corner,...>

C04 held the store fixed and varied the head, and found the reference head already optimal.
This unit does the opposite: it holds the reference PAIR DESIGN fixed and varies the store's
geometry — how many wells the channels are partitioned into (K) and how many modes each well
carries (M) — at the SAME byte ceiling.

⭐ The trade is real and it is forced by the ledger, not chosen:
    store = M·D·4                        grows with M only
    head  = (n_add·K·M + n_pair·r + r·K·M)·4   grows with BOTH K and M
so more wells costs modes.  For each K, `stores.max_modes` returns the largest M that still
fits `knn_proto`'s 475,136 B at the reference rank, and that is the M used.  ⛔ No point in
this sweep gets bytes another point was denied.

⚠ §E.4 is inherited and ⛔ not re-derived: learned well PLACEMENT is a null.  This sweep does
not touch placement — the partition is the shipped ward correlation grouping at every K.
It varies only how finely that same grouping is cut and how many atoms sit in each piece.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp  # noqa: E402
import ctx  # noqa: E402
import dev  # noqa: E402
import rivals  # noqa: E402
import stores  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "km"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CORNERS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3
                            else "0,2,4,6,10,12,14,16").split(",")]
RANK, RIDGE = 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
K_GRID = [8, 16, 32, 64, 128]        # ⛔ K must divide D = 2048; all of these do.

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_store import store_bytes  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
PLAN = []
for K in K_GRID:
    M = stores.max_modes(cards, PAIRS, K, D, RANK)
    if M < 2:
        ctx.log(f"  ⛔ K={K}: no M ≥ 2 fits the budget — dropped")
        continue
    tb = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
    assert tb <= stores.budget(cards, D), (K, M, tb)
    PLAN.append({"K": int(K), "M": int(M), "total_bytes": int(tb)})
    ctx.log(f"  K={K:4d}  M={M:3d}  bytes={tb:,} "
            f"({tb / stores.budget(cards, D):.4f} x knn_proto)  "
            f"[store {comp.store_bytes_of(M, D):,} + head "
            f"{comp.head_bytes(cards, PAIRS, K, M, RANK):,}]")
ctx.log(f"C07 tag={TAG} seed={SEED}: {len(PLAN)} (K, M) points x {len(CORNERS)} corners")

for ci in CORNERS:
    try:
        s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        ctx.log(f"[c{ci}] ⛔ REFUSED: {e}")
        continue
    todo = [p for p in PLAN
            if not rivals.ledger_has(TAG, ci, f"K{p['K']}_M{p['M']}")]
    if not todo:
        ctx.log(f"[c{ci}] all points already in the ledger — skipped")
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    req = requests_from_labels(lab_dev, "full", 0, SEED)
    for p in todo:
        tp = time.time()
        perm, modes = stores.build(c, ci, s["pool"], X_store, p["K"], p["M"], SEED)
        st = stores.to_gen_store(perm, modes, c["feat_shape"])
        head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
        feats = np.asarray(clu_wide_vector(st, head.coefficients(req)), np.float32)
        h = comp.headline(c, feats, lab_dev, pool_tup)
        rivals.ledger_put({
            "tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci], "seed": SEED,
            "arm": f"K{p['K']}_M{p['M']}", "bytes": p["total_bytes"],
            "bytes_vs_knn_proto": p["total_bytes"] / stores.budget(cards, D),
            "config": {**p, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES},
            "note": "C07 (K, M) store sweep at matched bytes; PINNED fit_factored_head",
            "extra": {"store_bytes_measured": int(store_bytes(st)["store_bytes"]),
                      "head_bytes_measured": int(head.byte_ledger()["bytes"])},
            "headline": h})
        ctx.log(f"[c{ci}] K={p['K']:4d} M={p['M']:3d}  "
                f"tuples_valid_novel={h['tuples_valid_novel']:3d}  "
                f"agg={h['aggregate']:.4f}  distinct={h['distinct']:3d}  "
                f"({time.time() - tp:.0f}s)")
    ctx.log(f"[c{ci}] done in {time.time() - t0:.1f}s")

ctx.log(f"C07 done in {time.time() - T0:.1f}s")
