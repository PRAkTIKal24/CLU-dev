"""C09 — ⭐⭐ THE REPLICATION'S REFUTATION, RUN DOWN. ⛔ SELECTION-SIDE ONLY, B never entered.

    python c09_construction.py <variant> <spec>

WHY THIS UNIT EXISTS
--------------------
The §A.5 replication spoke reproduced `knn_proto` (12), the landscape-removed control (0),
the decoder ceiling (67), both byte ledgers and the `72 × exact_match` identity **exactly**,
and **could not reproduce the object's own headline**: it measured **25** where this hub
measured **42**, and showed the number ranges **18–36** across nuisance choices the PREREG
left free. ⭐ It is right that the PREREG under-specified the store fit, and the criticism is
taken. This unit measures the size of the gap on DEV corners, where it costs nothing.

THE ACTUAL DIFFERENCE, found by reading both constructions
----------------------------------------------------------
Both cut the **same** ward linkage on the **same** |corr| by dendrogram **leaf order** into
equal blocks. They differ in **what happens to the 61 DEAD channels**:
- ⭐ `fast.balanced_labels` (what this hub used, inherited verbatim from
  `hub/generation-table`) **spreads the dead channels at evenly spaced slots** through the
  ordering — its docstring says why: *"so no single well is spent on constants."*
- `gen_partition.balanced_correlation_partition` (what the spoke used) **puts every dead
  channel into the smallest well**, which is what the shipped grouping does.
With 61 dead channels and 64-wide wells, the second concentrates roughly a whole well's worth
of constants into one or two wells. ⇒ **they are different stores at the same byte count**,
and ⛔ the PREREG did not say which one. That is the defect, and it is mine.

THE VARIANTS
------------
`genpart_25`   the spoke's partition, Lloyd 25 iters  — the closest like-for-like
`genpart_conv` the spoke's partition, Lloyd to convergence (200 max)
`mine_raw0`    MY partition, **0 Lloyd iters** — modes are the k-means++ seeds, i.e. actual
               train rows. ⭐ The spoke measured this variant HIGHEST (36) and argued it means
               the headline rides on unconverged noise. ⚠ C05a suggests the opposite reading:
               the head uses the store as a **signed basis**, and a basis wants DIVERSITY,
               not centroids. ⭐ Either way it is a measurable claim, so it is measured —
               across all 49 corners, against the same rival, ⛔ not on one cell.
(The hub's own construction is already in the ledger under tag `ref`.)
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
import fast  # noqa: E402
import rivals  # noqa: E402
import stores  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "genpart_25"
SPEC = sys.argv[2] if len(sys.argv) > 2 else "all"
SEED = 0
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
VARIANTS = {
    "genpart_25":   dict(partition="genpart", iters=25),
    "genpart_conv": dict(partition="genpart", iters=200),
    "mine_raw0":    dict(partition="mine", iters=0),
    "mine_conv":    dict(partition="mine", iters=200),
}
if VARIANT not in VARIANTS:
    sys.exit(f"variant must be one of {list(VARIANTS)}")
V = VARIANTS[VARIANT]
TAG = f"ctor_{VARIANT}"


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
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_partition import balanced_correlation_partition  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
TOT = comp.total_bytes(cards, PAIRS, K, M, RANK, D)
ctx.log(f"C09 variant={VARIANT} {V} spec={SPEC} -> {len(IDX)} corners; bytes {TOT:,}")

ok, refused, report_once = 0, [], True
for ci in IDX:
    if rivals.ledger_has(TAG, ci, "clu_linear"):
        ok += 1
        continue
    try:
        s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    except dev.HeldOutOverlap as e:
        refused.append(ci)
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)

    if V["partition"] == "mine":
        labels = fast.balanced_labels(ow, dead, K, D)
        pinfo = "fast.balanced_labels — dead channels SPREAD at evenly spaced slots"
    else:
        labels, info = balanced_correlation_partition(corr, K, dead=dead, method="ward")
        pinfo = f"gen_partition.balanced_correlation_partition — dead into smallest well; " \
                f"well sizes {info['well_sizes']}"
    sizes = np.bincount(labels, minlength=K)
    if int(sizes.min()) != int(sizes.max()):
        ctx.log(f"[c{ci}] ⛔ variant {VARIANT} gives ragged wells {sizes.tolist()} — "
                "the factored head needs equal wells; corner SKIPPED and NAMED")
        refused.append(ci)
        continue
    if report_once:
        ctx.log(f"  partition: {pinfo}; dead={int(dead.sum())}; "
                f"well sizes min {int(sizes.min())} max {int(sizes.max())}")
        report_once = False
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    pq = fast.kmeans_pq(Zs, labels, K, M, SEED, iters=int(V["iters"]))
    st = fast.EqualPQ(perm, pq.modes.astype(np.float32), (D,)).to_gen_store()
    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    feats = np.asarray(clu_wide_vector(
        st, head.coefficients(requests_from_labels(lab_dev, "full", 0, SEED))), np.float32)
    h = comp.headline(c, feats, lab_dev, pool_tup)
    rivals.ledger_put({
        "tag": TAG, "corner_idx": ci, "corner": comp.CORNERS[ci], "seed": SEED,
        "arm": "clu_linear", "bytes": TOT,
        "bytes_vs_knn_proto": TOT / stores.budget(cards, D),
        "config": {"K": K, "M": M, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES,
                   "variant": VARIANT, **V},
        "note": f"C09 construction sensitivity ({VARIANT}): {pinfo}",
        "headline": h})
    ctx.log(f"[c{ci}] {VARIANT:14s} tuples_valid_novel={h['tuples_valid_novel']:3d}  "
            f"agg={h['aggregate']:.4f}  distinct={h['distinct']:3d}  "
            f"({time.time() - t0:.0f}s)")
    ok += 1

ctx.log(f"C09 {VARIANT} done: {ok} corners, {len(refused)} refused/skipped {refused}, "
        f"{time.time() - T0:.1f}s")
