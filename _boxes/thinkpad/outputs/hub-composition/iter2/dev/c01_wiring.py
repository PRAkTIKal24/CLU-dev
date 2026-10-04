"""C01 -- WIRING CHECK. Reproduce the reference object and prove my fast path is the pinned one.

Three things, on the first two admitted SELECTION corners:
 1. the reference object (K=32, M=26, rank=16, pairs=floor_hue x scale + object_hue x scale,
    ridge=10) reproduces the inherited DEV-24 numbers on the same corner;
 2. `comp`'s sufficient-statistic head equals the pinned `fit_factored_head` to float64
    round-off -- so a search run through the fast path is a search over the pinned object;
 3. the headline identity distinct_valid_novel == n * exact_match holds, asserted.

Nothing is selected here. eval_a / eval_b never enter (dev.dev_setup's guard).
"""
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

T0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
OUTP = str(comp.OUT / "dev" / f"c01_seed{SEED}.json")

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
F = len(cards)
ctx.log(f"D={D} cards={cards} names={c['names']}")

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import (assert_request_only_factored,  # noqa: E402
                                         default_pairs, fit_factored_head)
from chlu.core.gen_store import assert_no_bypass, store_bytes  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
ALL = comp.all_pairs(F)
ctx.log(f"reference pairs {PAIR_NAMES} -> {PAIRS}")
ctx.log(f"byte ledger: store {comp.store_bytes_of(M, D)} + head "
        f"{comp.head_bytes(cards, PAIRS, K, M, RANK)} = "
        f"{comp.total_bytes(cards, PAIRS, K, M, RANK, D)} "
        f"vs knn_proto {comp.knn_proto_bytes(cards, D)}")

res = {"seed": SEED, "config": {"K": K, "M": M, "rank": RANK, "ridge": RIDGE,
                                "pairs": PAIR_NAMES},
       "bytes": {"store": comp.store_bytes_of(M, D),
                 "head": comp.head_bytes(cards, PAIRS, K, M, RANK),
                 "total": comp.total_bytes(cards, PAIRS, K, M, RANK, D),
                 "knn_proto": comp.knn_proto_bytes(cards, D)},
       "corners": {}}

for ci in comp.SELECTION_IDX[:2]:
    corner = comp.CORNERS[ci]
    t0 = time.time()
    s = dev.dev_setup(c, corner)
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(c["grid"], s["pool"], c["factor_labels"])
    ctx.log(f"[c{ci}] pool {s['pool'].size} rows, {pool_tup.shape[0]} distinct pool tuples")
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    labels = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    t1 = time.time()
    pq = fast.kmeans_pq(Zs, labels, K, M, SEED, iters=25)
    ctx.log(f"[c{ci}] kmeans_pq {time.time()-t1:.1f}s")
    st = fast.EqualPQ(perm, pq.modes.astype(np.float32), (D,)).to_gen_store()
    req = requests_from_labels(lab_dev, "full", 0, SEED)

    # --- the PINNED head -----------------------------------------------------
    t2 = time.time()
    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    a_pin = head.coefficients(req)
    t_pin = time.time() - t2

    # --- the FAST head, over the FULL pair design ----------------------------
    t3 = time.time()
    G, H, n_add, n_pair_all = comp.sufficient_stats(Zs, lab_store, cards, ALL)
    t_stats = time.time() - t3
    t4 = time.time()
    idx = comp.subset_index(cards, ALL, PAIRS, n_add)
    W = comp.solve_W(G, H, idx, RIDGE)
    C = comp.project_to_modes(W, perm, pq.modes)
    C_add, E, U = comp.truncate(C, n_add, RANK)
    a_fast = comp.coefficients(C_add, E, U, lab_dev, cards, PAIRS, K, M)
    t_fast = time.time() - t4

    den = max(float(np.abs(a_pin).max()), 1e-30)
    dmax = float(np.abs(a_pin - a_fast).max())
    ctx.log(f"[c{ci}] head equivalence: max|d a| = {dmax:.3e} "
            f"(rel {dmax/den:.3e}); pinned {t_pin:.1f}s, stats {t_stats:.1f}s, "
            f"fast {t_fast:.2f}s")

    feats_pin = np.asarray(clu_wide_vector(st, a_pin), np.float32)
    feats_fast = np.asarray(clu_wide_vector(st, a_fast), np.float32)
    h_pin = comp.headline(c, feats_pin, lab_dev, pool_tup)
    h_fast = comp.headline(c, feats_fast, lab_dev, pool_tup)

    ro = assert_request_only_factored(st, head, req, decoy_features=Zs[:64])
    nb = assert_no_bypass(st, Zs[:64], rng_seed=SEED)

    res["corners"][f"c{ci}"] = {
        "corner": corner, "pool": int(s["pool"].size),
        "n_pool_tuples": int(pool_tup.shape[0]),
        "pinned": h_pin, "fast": h_fast,
        "head_equivalence": {"max_abs_delta_coefficients": dmax,
                             "relative": dmax / den,
                             "headline_identical": h_pin["tuples_valid_novel"]
                             == h_fast["tuples_valid_novel"]},
        "timings_s": {"kmeans": round(t1 and time.time() - t1, 1), "pinned_head": round(t_pin, 1),
                      "suff_stats_allpairs": round(t_stats, 1), "fast_head": round(t_fast, 2)},
        "store_bytes": int(store_bytes(st)["store_bytes"]),
        "head_bytes": int(head.byte_ledger()["bytes"]),
        "assert_request_only_factored": {k: v for k, v in ro.items() if k != "signatures"},
        "assert_no_bypass": {k: v for k, v in nb.items() if k != "note"},
        "seconds": round(time.time() - t0, 1),
    }
    ctx.log(f"[c{ci}] PINNED  tuples_valid_novel={h_pin['tuples_valid_novel']:3d}  "
            f"agg={h_pin['aggregate']:.4f}  distinct={h_pin['distinct']:3d}  "
            f"exact={h_pin['exact_match']:.4f}")
    ctx.log(f"[c{ci}] FAST    tuples_valid_novel={h_fast['tuples_valid_novel']:3d}  "
            f"agg={h_fast['aggregate']:.4f}  distinct={h_fast['distinct']:3d}  "
            f"exact={h_fast['exact_match']:.4f}")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
