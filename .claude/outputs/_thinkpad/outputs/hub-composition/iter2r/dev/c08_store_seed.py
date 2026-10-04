"""C08 — ⭐ HOW MUCH OF THE CORNER VARIANCE IS THE STORE's k-MEANS SEED?

    python c08_store_seed.py <store_seed> <spec>

⛔ SELECTION-SIDE ONLY: DEV corners, ⛔ B never entered.

WHY
---
Every number this hub reports is **bridge seed 0**, because seeds 1 and 2 have `scope="used"`
caches and covering the cell would mean encoding ~32,200 new rows — forbidden. ⛔ That
limitation stands and is not evaded here.

But the *store* carries its own seed: `kmeans_pq`'s k-means++ initialisation. That one is
free to vary, because it touches **no** cached feature — it only re-initialises the Lloyd
iterations on features that are already on disk. ⭐ So this unit measures something the hub
could otherwise only assert: **how much of the ±13.4-tuple corner-to-corner spread is the
store's initialisation, and how much is the corner itself.**

⚠ The CONTEXT is always built at bridge seed 0 (`ctx.build_context(0)`) — the pool, the
encoder, the decoder, the classifier and the cache are all seed 0 and never move. ⛔ Only the
k-means seed changes. `knn_proto` is not re-run: `FactorPrototypes.fit` is a deterministic
per-value mean and has no seed.
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

STORE_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SPEC = sys.argv[2] if len(sys.argv) > 2 else "sel"
BRIDGE_SEED = 0                     # ⛔ never moves: seeds 1/2 are cache-blocked
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
TAG = f"storeseed{STORE_SEED}"


def parse(spec):
    if spec.startswith("sel"):
        idx = comp.SELECTION_IDX
    elif spec.startswith("val"):
        idx = comp.VALIDATION_IDX
    elif spec.startswith("all"):
        idx = sorted(comp.SELECTION_IDX + comp.VALIDATION_IDX)
    else:
        return [int(x) for x in spec.split(",")]
    n = spec.split(":")
    return idx[: int(n[1])] if len(n) > 1 else idx


T0 = time.time()
IDX = parse(SPEC)
c = ctx.build_context(BRIDGE_SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
TOT = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
ctx.log(f"C08 store_seed={STORE_SEED} (bridge seed FIXED at {BRIDGE_SEED}) "
        f"spec={SPEC} -> {len(IDX)} corners; bytes {TOT:,}")

ok, refused = 0, []
for ci in IDX:
    if rivals.ledger_has(TAG, ci, "clu_linear"):
        ok += 1
        continue
    try:
        s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        ctx.log(f"[c{ci}] ⛔ REFUSED: {e}")
        refused.append(ci)
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    perm, modes = stores.build(c, ci, s["pool"], X_store, K, M, STORE_SEED)
    st = stores.to_gen_store(perm, modes, c["feat_shape"])
    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    feats = np.asarray(clu_wide_vector(
        st, head.coefficients(requests_from_labels(lab_dev, "full", 0, BRIDGE_SEED))),
        np.float32)
    h = comp.headline(c, feats, lab_dev, pool_tup)
    rivals.ledger_put({
        "tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci],
        "seed": BRIDGE_SEED, "store_seed": STORE_SEED, "arm": "clu_linear",
        "bytes": TOT, "bytes_vs_knn_proto": TOT / stores.budget(cards, D),
        "config": {"K": K, "M": M, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES,
                   "bridge_seed": BRIDGE_SEED, "store_kmeans_seed": STORE_SEED},
        "note": "C08 store-seed spread; bridge seed fixed at 0 (seeds 1/2 cache-blocked)",
        "headline": h})
    ctx.log(f"[c{ci}] store_seed={STORE_SEED}  "
            f"tuples_valid_novel={h['tuples_valid_novel']:3d}  "
            f"agg={h['aggregate']:.4f}  distinct={h['distinct']:3d}  "
            f"({time.time() - t0:.0f}s)")
    ok += 1

ctx.log(f"C08 done: {ok} corners, {len(refused)} refused {refused}, "
        f"{time.time() - T0:.1f}s")
