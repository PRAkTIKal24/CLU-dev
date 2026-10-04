"""C04 — THE HEAD-DESIGN SEARCH, at matched bytes, on the SELECTION corners only.

    python c04_head.py <tag> <seed> <spec>

⛔ SELECTION ONLY.  The store is the reference store (K=32, M=26 k-means modes over the
ward channel partition) and is held FIXED; only the request→coefficient head varies.
That keeps the departure minimal: the *store* — the thing deletion and refusal need —
is bit-identical to the reference object in every row below.

⭐ Every candidate is given the LARGEST rank that fits inside `knn_proto`'s byte ledger
(`stores.max_rank`), so no design is compared at a byte advantage.
⭐ The search runs through the FAST head; C01 measured fast == pinned to 4.8e-07 with an
identical headline.  ⛔ Whatever this unit SELECTS is re-fitted through the pinned
`fit_factored_head` in C06 before it is reported.

C02 already measured `additive` and the reference design; they are re-emitted here under
this unit's own tag so every candidate in the table comes from one code path.
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
import rivals  # noqa: E402
import stores  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "head"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SPEC = sys.argv[3] if len(sys.argv) > 3 else "sel"
K, M = 32, 26

# ---------------------------------------------------------------------------
# The candidate designs. Named by WHICH factors are allowed to interact.
# cards = [floor_hue 10, wall_hue 10, object_hue 10, scale 8, shape 4, orientation 15]
# ---------------------------------------------------------------------------
DESIGNS = {
    "additive":            [],
    "ref_hue0hue2_x_scale": [(0, 3), (2, 3)],          # the reference object's pairs
    "allhue_x_scale":      [(0, 3), (1, 3), (2, 3)],
    "allhue_x_shape":      [(0, 4), (1, 4), (2, 4)],
    "allhue_x_orient":     [(0, 5), (1, 5), (2, 5)],
    "hue_x_hue":           [(0, 1), (0, 2), (1, 2)],
    "geom_x_geom":         [(3, 4), (3, 5), (4, 5)],
    "shape_x_orient":      [(4, 5)],
    "scale_x_orient":      [(3, 5)],
    "scale_x_shape":       [(3, 4)],
    "allhue_x_geom":       [(0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4),
                            (0, 5), (1, 5), (2, 5)],
    "geom_plus_refhue":    [(3, 4), (3, 5), (4, 5), (0, 3), (2, 3)],
    "all15":               [(f, g) for f in range(6) for g in range(f + 1, 6)],
}
RIDGES = [1.0, 10.0, 100.0]
# extra rank points for the reference design, to separate "pairs help" from "rank helps"
RANK_PROBE = {"ref_hue0hue2_x_scale": [4, 8], "additive": []}

LEDGER_TAGCOL = "head_design"


def parse(spec):
    if spec.startswith("sel"):
        idx = comp.SELECTION_IDX
    elif spec.startswith("val"):
        idx = comp.VALIDATION_IDX
    else:
        return [int(x) for x in spec.split(",")]
    n = spec.split(":")
    return idx[: int(n[1])] if len(n) > 1 else idx


T0 = time.time()
IDX = parse(SPEC)
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
ctx.log(f"C04 tag={TAG} seed={SEED} spec={SPEC} -> {len(IDX)} corners; "
        f"budget {stores.budget(cards, D):,} B")

# the (design, ridge, rank) plan, with each design's own maximal rank
PLAN = []
for name, pairs in DESIGNS.items():
    r = stores.max_rank(cards, pairs, K, M, D)
    if r < 0:
        ctx.log(f"  ⛔ {name}: does not fit the budget at ANY rank — dropped")
        continue
    ranks = [r] + [x for x in RANK_PROBE.get(name, []) if x < r]
    for rr in sorted(set(ranks), reverse=True):
        for ridge in RIDGES:
            tb = comp.total_bytes(cards, pairs, K, M, rr, D)
            assert tb <= stores.budget(cards, D), (name, rr, tb)
            PLAN.append({"design": name, "pairs": pairs, "rank": int(rr),
                         "ridge": float(ridge), "total_bytes": int(tb)})
ctx.log(f"plan: {len(PLAN)} (design, rank, ridge) points "
        f"[{len(DESIGNS)} designs x ranks x {len(RIDGES)} ridges]")
for p in PLAN:
    if p["ridge"] == RIDGES[0]:
        ctx.log(f"  {p['design']:22s} rank={p['rank']:3d} bytes={p['total_bytes']:,} "
                f"({p['total_bytes'] / stores.budget(cards, D):.4f} x knn_proto)")

ALLP = comp.all_pairs(len(cards))
ok, refused = 0, []
for ci in IDX:
    try:
        s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        ctx.log(f"[c{ci}] ⛔ REFUSED: {e}")
        refused.append(ci)
        continue
    todo = [p for p in PLAN
            if not rivals.ledger_has(TAG, ci, f"{p['design']}|r{p['rank']}|g{p['ridge']:g}")]
    if not todo:
        ctx.log(f"[c{ci}] all {len(PLAN)} head points already in the ledger — skipped")
        ok += 1
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(c["grid"], s["pool"], c["factor_labels"])
    perm, modes = stores.build(c, ci, s["pool"], X_store, K, M, SEED)
    st = stores.to_gen_store(perm, modes, c["feat_shape"])
    G, H, n_add, _ = comp.sufficient_stats(Zs, lab_store, cards, ALLP)
    ctx.log(f"[c{ci}] store + sufficient stats in {time.time() - t0:.1f}s; "
            f"{len(todo)} head points to score")

    from chlu.core.gen_compose import clu_wide_vector
    best = None
    for p in todo:
        arm = f"{p['design']}|r{p['rank']}|g{p['ridge']:g}"
        idx = comp.subset_index(cards, ALLP, p["pairs"], n_add)
        W = comp.solve_W(G, H, idx, p["ridge"])
        C = comp.project_to_modes(W, perm, modes)
        C_add, E, U = comp.truncate(C, n_add, p["rank"])
        a = comp.coefficients(C_add, E, U, lab_dev, cards, p["pairs"], K, M)
        feats = np.asarray(clu_wide_vector(st, a), np.float32)
        h = comp.headline(c, feats, lab_dev, pool_tup)
        rivals.ledger_put({"tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci],
                           "seed": SEED, "arm": arm, "bytes": p["total_bytes"],
                           "bytes_vs_knn_proto": p["total_bytes"] / stores.budget(cards, D),
                           "config": {"K": K, "M": M, **p},
                           "note": "C04 head-design search, FAST head (fast==pinned to 4.8e-07)",
                           "headline": h})
        if best is None or h["tuples_valid_novel"] > best[1]:
            best = (arm, h["tuples_valid_novel"])
    ctx.log(f"[c{ci}] {len(todo)} points in {time.time() - t0:.1f}s; "
            f"best here: {best[0]} = {best[1]}")
    ok += 1

ctx.log(f"C04 done: {ok} corners, {len(refused)} refused {refused}, "
        f"{time.time() - T0:.1f}s")
