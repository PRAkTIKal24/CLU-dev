"""C12 — ⭐ THE SHIPPED Adam `train_store` vs plain Lloyd, as the MODE FITTER.
⛔ DEV ONLY. `eval_b` is never opened.

    python c12_trainstore.py <steps> <corner,corner,...>

WHY — and it is the replication spoke's call, not mine
------------------------------------------------------
`fast.kmeans_pq`'s own docstring says it is *"deliberately NOT the shipped `train_store`
(soft/commitment Adam objective) … The shipped trainer is run beside it for the like-for-like
column."*  ⛔ **This hub never ran that column.** The §A.5 spoke named it the highest-value
follow-up and could not finish it; it is the one store-fit variant C09 did not cover.

⭐ There is a real mechanism to test, not just a box to tick. `train_store` minimises
    L = mean_k ‖x_k − Σ_m π_km·mode_km‖²  +  β‖x_k − mode_{k,argmin}‖²
i.e. it optimises the **SOFT occupancy read** — which is much closer to this object's LINEAR
read than Lloyd's hard-read distortion is. ⇒ it is a priori the better-matched fitter, and
C05a's finding (the head uses the store as a signed basis) makes it worth measuring.

WHAT IS HELD FIXED — so only the fitter moves
---------------------------------------------
The equal-well partition is **my** `fast.balanced_labels` (identical to every other unit
here), K = 32, M = 26, rank 16, ridge 10, the reference pair design, seed 0, the same pool,
the same requests, the same instrument. ⛔ The ONLY difference from the `ref` rows already in
the ledger is `init_store` + `train_store` in place of `kmeans_pq`.

⚠ It is SLOW (the spoke measured ~3.3 s/step). Resumable per corner: a kill costs ONE CORNER.
"""
from __future__ import annotations

import copy
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

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
CORNERS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2
                            else "0,2,4,6,20,44").split(",")]
SEED = 0
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
TAG = f"trained{STEPS}"

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

import jax  # noqa: E402
from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_store import init_store, store_bytes, train_store  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
TOT = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
ctx.log(f"C12 shipped train_store, steps={STEPS}, corners={CORNERS}; bytes {TOT:,}")

for ci in CORNERS:
    if rivals.ledger_has(TAG, ci, "clu_linear"):
        ctx.log(f"[c{ci}] already in the ledger — skipped")
        continue
    try:
        s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        ctx.log(f"[c{ci}] ⛔ REFUSED: {e}")
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    req = requests_from_labels(lab_dev, "full", 0, SEED)

    # ⭐ the SAME partition every other unit in this hub used
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    labels = fast.balanced_labels(ow, dead, K, D)
    wc = tuple(tuple(int(i) for i in np.flatnonzero(labels == k)) for k in range(K))

    g2 = copy.copy(c["cfg"].gen_store)
    g2.train_steps = int(STEPS)
    key = jax.random.PRNGKey(int(SEED) * 10_007 + 20_260_911)
    key, ik, tk = jax.random.split(key, 3)
    st = init_store(X_store, wc, M, tuple(c["feat_shape"]), ik, seed=SEED)
    ctx.log(f"[c{ci}] init_store done ({time.time() - t0:.0f}s); training {STEPS} steps…")
    ts = time.time()
    st, hist, key = train_store(st, X_store, g2, tk, log=None)
    dt = time.time() - ts
    loss0 = float(hist[0]) if hist is not None and len(hist) else float("nan")
    loss1 = float(hist[-1]) if hist is not None and len(hist) else float("nan")
    ctx.log(f"[c{ci}] train_store {STEPS} steps in {dt:.0f}s "
            f"({dt / max(STEPS, 1):.2f} s/step); loss {loss0:.4f} -> {loss1:.4f} "
            f"⚠ tau anneals, so this pair is NOT a monotone curve")

    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    feats = np.asarray(clu_wide_vector(st, head.coefficients(req)), np.float32)
    h = comp.headline(c, feats, lab_dev, pool_tup)
    sb = int(store_bytes(st)["store_bytes"])
    hb = int(head.byte_ledger()["bytes"])
    assert sb + hb == TOT, f"byte ledger moved: {sb}+{hb} != {TOT}"
    rivals.ledger_put({
        "tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci], "seed": SEED,
        "arm": "clu_linear", "bytes": TOT,
        "bytes_vs_knn_proto": TOT / stores.budget(cards, D),
        "config": {"K": K, "M": M, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES,
                   "fitter": "SHIPPED init_store + train_store (Adam, soft+commitment)",
                   "train_steps": int(STEPS),
                   "partition": "fast.balanced_labels — identical to every other unit"},
        "note": "C12 shipped-trainer column; ONLY the mode fitter differs from tag `ref`",
        "extra": {"loss_first": loss0, "loss_last": loss1, "train_seconds": dt,
                  "store_bytes_measured": sb, "head_bytes_measured": hb},
        "headline": h})
    ctx.log(f"[c{ci}] ⭐ TRAINED tuples_valid_novel={h['tuples_valid_novel']:3d}  "
            f"agg={h['aggregate']:.4f}  distinct={h['distinct']:3d}  "
            f"(corner total {time.time() - t0:.0f}s)")

ctx.log(f"C12 done in {time.time() - T0:.1f}s")
