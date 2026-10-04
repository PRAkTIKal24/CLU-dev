"""C10 — ⭐ THE FRESH-CORNER TEST. Pre-registered in `LOG.md` at 2026-09-17T16:20Z.

    python c10_fresh.py <spec>            spec: "all" | "N" (first N) | "i,j,k"

⛔ DEV ONLY. `eval_b` is never opened; `dev.dev_setup` refuses any corner that touches it or
`eval_a`, and refusals are NAMED, ⛔ never trimmed.

WHY
---
Two questions, both declared before any fresh corner was scored:
**F1 (the valuable one).** Does the hub's DEV estimate — Δ = +1.40, CI95 [−2.31, +5.11], a
TIE — replicate at corner locations that were used for **no** selection in this hub?
**F2–F4 (the hypothesis).** The corner-signature slice was POST-HOC: after seeing that B
carries `scale_hi ∧ orient_hi`, that slice of the existing corners read Δ = +8.50. ⛔ Named
after the fact, over four candidates. Here it is a prediction on corners never looked at.

THE FRESH FAMILY
----------------
The declared 64 corners use `hue_lo ∈ {0.0, 0.2, 0.4, 0.6}`. The hue grid has ten values
0.0 … 0.9, so `hue_lo ∈ {0.1, 0.3, 0.5, 0.7}` is untouched. ⛔ `hue_lo = 0.8` is EXCLUDED —
that is B's own location. Same `ctx.montero_like_corner`, same thresholds, same guard.
Corner indices are offset by 1000 in the ledger so the original indices are untouched.
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

SPEC = sys.argv[1] if len(sys.argv) > 1 else "all"
SEED = 0
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
TAG = "fresh"
OFFSET = 1000

FRESH = [dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
         for h in (0.1, 0.3, 0.5, 0.7) for sh in (0.0, 1.0, 2.0, 3.0)
         for sc in (True, False) for o in (True, False)]
assert len(FRESH) == 64
assert all(c["hue_lo"] not in (0.0, 0.2, 0.4, 0.6, 0.8) for c in FRESH), \
    "⛔ a fresh corner collides with the declared family or with B's location"

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
TOT = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
KNNB = comp.knn_proto_bytes(cards, D)
idx = (range(len(FRESH)) if SPEC == "all"
       else range(int(SPEC)) if SPEC.isdigit()
       else [int(x) for x in SPEC.split(",")])
ctx.log(f"C10 FRESH corners, tag={TAG}, {len(list(idx))} candidates; bytes {TOT:,}")

import jax  # noqa: E402
from chlu.core.gen_store import GenStore  # noqa: E402

ok, refused = 0, []
for j in idx:
    ci = OFFSET + j
    corner = FRESH[j]
    if rivals.ledger_has(TAG, ci, "clu_linear"):
        ok += 1
        continue
    try:
        s = dev.dev_setup(c, corner, log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        ctx.log(f"[f{j}] ⛔ REFUSED {corner}: {e}")
        refused.append(j)
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    perm, modes = stores.build(c, ci, s["pool"], X_store, K, M, SEED)
    st = stores.to_gen_store(perm, modes, c["feat_shape"])
    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    a = head.coefficients(requests_from_labels(lab_dev, "full", 0, SEED))

    def put(arm, feats, b, note):
        try:
            h = comp.headline(c, np.asarray(feats, np.float32), lab_dev, pool_tup)
        except AssertionError as e:                 # ⛔ named, never papered over
            ctx.log(f"[f{j}] ⛔ IDENTITY FAILED for {arm}: {e}")
            raise
        rivals.ledger_put({"tag": TAG, "corner_idx": ci, "corner": corner, "seed": SEED,
                           "arm": arm, "bytes": int(b),
                           "bytes_vs_knn_proto": float(b) / KNNB,
                           "config": {"K": K, "M": M, "rank": RANK, "ridge": RIDGE,
                                      "pairs": PAIR_NAMES, "family": "FRESH",
                                      "fresh_index": j},
                           "note": note, "headline": h})
        return h

    h1 = put("clu_linear", np.asarray(clu_wide_vector(st, a), np.float32), TOT,
             "C10 fresh-corner test — the object")
    protos = FactorPrototypes.fit(grid, Zs, s["pool"])
    h2 = put("knn_proto", np.asarray(protos.embed_many(
        requests_from_rows(grid, s["dev_rows"])), np.float32).reshape(
        (-1,) + tuple(c["feat_shape"])), protos.byte_ledger()["bytes"],
        "C10 fresh-corner test — the rival, same pool, same requests")
    wc = tuple(tuple(int(i) for i in r) for r in st.well_channels)
    z = GenStore(wc, M, tuple(c["feat_shape"]), jax.random.PRNGKey(0),
                 init=[np.zeros_like(np.asarray(m)) for m in st.modes])
    h3 = put("store_removed_zero", np.asarray(clu_wide_vector(z, a), np.float32), TOT,
             "C10 fresh-corner test — §C.2 laundering control")
    h4 = put("ceiling", np.asarray(c["gather"](s["dev_rows"]), np.float32).reshape(
        (-1,) + tuple(c["feat_shape"])), 0,
        "⛔ NOT AN ARM — the decoder ceiling on this fresh corner")
    ctx.log(f"[f{j}] hue={corner['hue_lo']} sc={int(corner['scale_hi'])} "
            f"sh={int(corner['shape_value'])} o={int(corner['orient_hi'])}  "
            f"clu={h1['tuples_valid_novel']:3d} knn={h2['tuples_valid_novel']:3d} "
            f"ctrl={h3['tuples_valid_novel']:3d} ceil={h4['tuples_valid_novel']:3d}  "
            f"({time.time() - t0:.0f}s)")
    ok += 1

ctx.log(f"C10 done: {ok} fresh corners, {len(refused)} refused {refused}, "
        f"{time.time() - T0:.1f}s")
