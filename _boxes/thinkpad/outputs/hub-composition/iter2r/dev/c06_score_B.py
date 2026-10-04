"""⛔⛔ C06 — THE SCORED RUN. This script READS `eval_b`. Everything else in this hub does not.

    SCORED_RUN_PREREG=<path to the PREREG file> python c06_score_B.py <seed>

⛔ It REFUSES to start unless `SCORED_RUN_PREREG` names an existing PREREG file, so a
scored run cannot happen by reflex, by a stray shell-history line, or by a resumed loop.
⭐ It writes ONE result file per (seed) and ⛔ never overwrites one: a second invocation on
a seed that already has a result aborts. A scored run is spent once.

WHAT IT DOES
------------
R3: *"Headline comparisons are same-device by construction: the scored cell re-runs every
arm, rivals included, on this box from Mac-encoded features."*  That is exactly this.
Every arm below is built HERE, from the Mac-written feature cache (shortfall 0, gate not
enforced — `coverage_seed0.json`), and scored by the SAME frozen instruments:
`exp_gen_cell.load_external_classifier`, `exp_gen_compose.resolve_decoder`,
`exp_gen_cell.render` / `score_images`, and the pinned `FactorPrototypes` for the rival.
⛔ No Mac-published row enters any paired comparison. Mac rows are cited beside, as priors.

THE STATISTIC, DECLARED BEFORE THE RUN
--------------------------------------
B has 72 items and is ONE cell, so the pairing unit is the ITEM:
    d_i = valid_arm(i) − valid_rival(i) ∈ {−1, 0, +1},  i = 1..72
    Δ = mean(d) = exact_match_arm − exact_match_rival  (an aggregate-unit quantity)
    CI95 = paired t on d (McNemar's exact test reported beside it)
By the inherited identity `distinct_valid_novel = 72 · exact_match` on B — all 72 targets
distinct, none in any train row, ASSERTED per render by `comp.headline` — the same Δ in
TUPLES is 72 · Δ. R1 is applied in aggregate units with m = 0.020 and printed in both.
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
import rivals  # noqa: E402
import stores  # noqa: E402

PREREG = os.environ.get("SCORED_RUN_PREREG", "")
if not PREREG or not os.path.exists(PREREG):
    sys.exit("⛔ REFUSED: set SCORED_RUN_PREREG to an existing PREREG file. "
             "A scored run is pre-registered BEFORE it runs (charter §A.1).")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = stores.CACHE.parent / f"SCORED_B_seed{SEED}.json"
if OUT.exists():
    sys.exit(f"⛔ REFUSED: {OUT} already exists. A scored run is spent ONCE.")

# the object, read out of the PREREG so the two cannot drift apart
CFG = json.loads(open(os.path.join(os.path.dirname(PREREG),
                                   "PREREG-object.json"), encoding="utf-8").read())
K, M, RANK, RIDGE = int(CFG["K"]), int(CFG["M"]), int(CFG["rank"]), float(CFG["ridge"])
PAIR_NAMES = list(CFG["pairs"])

print("=" * 78)
print("⛔⛔ SCORED RUN — eval_b IS ABOUT TO BE READ.  1 of 3 for this target.")
print(f"    PREREG : {PREREG}")
print(f"    object : K={K} M={M} rank={RANK} ridge={RIDGE} pairs={PAIR_NAMES}")
print(f"    seed   : {SEED}   (⛔ seed 0 only on this box — seeds 1/2 are cache-blocked)")
print("=" * 78, flush=True)

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]
B = np.asarray(c["rows"]["eval_b"], np.int64)
pool = np.asarray(c["pool"], np.int64)
assert int(np.intersect1d(B, pool).size) == 0, \
    "⛔ eval_b intersects the store pool — the split is broken, STOP"
lab_B = c["factor_labels"](grid, B)
pool_tup = comp.pool_tuple_array(grid, pool, c["factor_labels"])
X_store = c["gather"](pool)
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
lab_store = c["factor_labels"](grid, pool)
ctx.log(f"B: {B.size} rows · pool {pool.size} rows / {pool_tup.shape[0]} distinct tuples")


def per_item_valid(feats):
    """(valid mask over items, the full headline). ⭐ ONE instrument for every arm."""
    from chlu.eval.gen_probe import predict_factors
    from chlu.experiments.exp_gen_cell import render
    imgs = render(c["decoder"], np.asarray(feats, np.float32), c["feat_shape"], 256)
    P = np.asarray(predict_factors(c["external"], imgs, 256), np.int64)
    v = (P == np.asarray(lab_B, np.int64)).all(1)
    return v, comp.headline(c, np.asarray(feats, np.float32), lab_B, pool_tup)


res = {"seed": SEED, "prereg": PREREG, "object": CFG, "n_B": int(B.size),
       "machine": "Windows AMD64 (FORGIS_14) — R3 provenance; every arm below built HERE",
       "cache": {"n_rows": int(c["cman"]["n_rows"]), "scope": str(c["cman"]["scope"]),
                 "shortfall": 0},
       "arms": {}, "per_item_valid": {}}

# ---- every arm, through rivals.py's own constructions -----------------------
from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import (assert_request_only_factored,  # noqa: E402
                                         default_pairs, fit_factored_head)
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.core.gen_store import assert_no_bypass, store_bytes  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
perm, modes = stores.build(c, "B", pool, X_store, K, M, SEED)
st = stores.to_gen_store(perm, modes, c["feat_shape"])
head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
req = requests_from_labels(lab_B, "full", 0, SEED)
a = head.coefficients(req)
tot_b = comp.store_bytes_of(M, D) + comp.head_bytes(cards, PAIRS, K, M, RANK)
knn_b = comp.knn_proto_bytes(cards, D)


def put(arm, feats, bytes_, note, extra=None):
    v, h = per_item_valid(feats)
    res["arms"][arm] = {"bytes": int(bytes_), "bytes_vs_knn_proto": bytes_ / knn_b,
                        "note": note, "headline": h, **({"extra": extra} if extra else {})}
    res["per_item_valid"][arm] = [int(x) for x in v]
    ctx.log(f"{arm:22s} tuples_valid_novel={h['tuples_valid_novel']:3d}  "
            f"agg={h['aggregate']:.4f}  distinct={h['distinct']:3d}  "
            f"exact={h['exact_match']:.4f}  bytes={int(bytes_):,}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    return v


ro = assert_request_only_factored(st, head, req, decoy_features=Zs[:64])
nb = assert_no_bypass(st, Zs[:64], rng_seed=SEED)
res["assertions"] = {"assert_request_only_factored":
                     {k: v for k, v in ro.items() if k != "signatures"},
                     "assert_no_bypass": {k: v for k, v in nb.items() if k != "note"}}
put("clu_linear", np.asarray(clu_wide_vector(st, a), np.float32), tot_b,
    "THE OBJECT — request-addressed LINEAR read, pinned head",
    {"store_bytes": int(store_bytes(st)["store_bytes"]),
     "head_bytes": int(head.byte_ledger()["bytes"])})

protos = FactorPrototypes.fit(grid, Zs, pool)
put("knn_proto", np.asarray(protos.embed_many(requests_from_rows(grid, B)),
                            np.float32).reshape((-1,) + tuple(c["feat_shape"])),
    protos.byte_ledger()["bytes"],
    "THE RIVAL — pinned FactorPrototypes, fitted on the SAME pool")

import jax  # noqa: E402
from chlu.core.gen_store import GenStore  # noqa: E402
wc = tuple(tuple(int(i) for i in r) for r in st.well_channels)
put("store_removed_zero",
    np.asarray(clu_wide_vector(GenStore(wc, M, tuple(c["feat_shape"]),
                                        jax.random.PRNGKey(0),
                                        init=[np.zeros_like(np.asarray(m))
                                              for m in st.modes]), a), np.float32),
    tot_b, "§C.2 LAUNDERING CONTROL — identical read path, atoms ZEROED")

rng = np.random.default_rng(int(SEED) * 104_729 + 7)
allm = np.stack([np.asarray(m) for m in st.modes])
flat = allm.reshape(-1, allm.shape[-1])[rng.permutation(allm.shape[0] * allm.shape[1])]
put("store_removed_shuffle",
    np.asarray(clu_wide_vector(GenStore(wc, M, tuple(c["feat_shape"]),
                                        jax.random.PRNGKey(0),
                                        init=list(flat.reshape(allm.shape))), a),
               np.float32),
    tot_b, "§C.2 LAUNDERING CONTROL — atoms SHUFFLED across wells, bytes identical")

h0 = fit_factored_head(st, Zs, lab_store, cards, [], 0, RIDGE)
put("head_additive_only",
    np.asarray(clu_wide_vector(st, h0.coefficients(req)), np.float32),
    comp.store_bytes_of(M, D) + comp.head_bytes(cards, [], K, M, 0),
    "the OTHER half of §C.2 — atoms kept, pair interactions removed")

S = rivals.n_lookup_slots(tot_b, D)
stride = max(1, Zs.shape[0] // 20000)
sub_rows = np.arange(0, Zs.shape[0], stride, dtype=np.int64)
med_f = sub_rows[rivals._kmedoid_rows(Zs[sub_rows], S, SEED)]
put("lookup_feat", rivals.lookup_read(Zs, lab_store, lab_B, med_f, grid).reshape(
    (-1,) + tuple(c["feat_shape"])), int(S * D * 4),
    f"§C.1 MATCHED-BYTES LOOKUP — {S} real pool rows, FEATURE k-medoids")
med_t = rivals._kmedoid_rows(rivals._tuple_space(grid, lab_store), S, SEED)
put("lookup_tuple", rivals.lookup_read(Zs, lab_store, lab_B, med_t, grid).reshape(
    (-1,) + tuple(c["feat_shape"])), int(S * D * 4),
    f"§C.1 MATCHED-BYTES LOOKUP, STRONGER FORM — {S} rows covering the FACTOR space")

put("const_mu", np.repeat(Zs.mean(0)[None], B.size, 0).reshape(
    (-1,) + tuple(c["feat_shape"])), int(D * 4), "the CONSTANT-OUTPUT FLOOR")

put("ceiling", np.asarray(c["gather"](B), np.float32).reshape(
    (-1,) + tuple(c["feat_shape"])), 0,
    "⛔ NOT AN ARM — the DECODER CEILING on B: the true cached feature, decoded. "
    "R2 requires it reported beside the headline.")

# ---- the pre-declared paired statistic --------------------------------------
from scipy import stats  # noqa: E402
vr = np.asarray(res["per_item_valid"]["knn_proto"], float)
res["paired_vs_knn_proto"] = {}
for arm in res["arms"]:
    if arm in ("knn_proto", "ceiling"):
        continue
    va = np.asarray(res["per_item_valid"][arm], float)
    d = va - vr
    n = d.size
    m_ = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    b01 = int(((va == 1) & (vr == 0)).sum())
    b10 = int(((va == 0) & (vr == 1)).sum())
    mcn = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if (b01 + b10) else 1.0
    p = {"n": n, "delta_aggregate_units": m_, "se": se,
         "ci95_aggregate_units": [m_ - t * se, m_ + t * se],
         "delta_tuples": m_ * 72.0,
         "ci95_tuples": [(m_ - t * se) * 72.0, (m_ + t * se) * 72.0],
         "mcnemar_b01_arm_only": b01, "mcnemar_b10_rival_only": b10,
         "mcnemar_exact_p": mcn}
    p["R1_verdict"] = comp.r1_verdict(
        {"delta": m_, "ci95": p["ci95_aggregate_units"]}, m=0.020, scale=1.0)
    res["paired_vs_knn_proto"][arm] = p
    ctx.log(f"R1 {arm:22s} Δ={m_:+.4f} agg ({m_ * 72:+.2f} tuples) "
            f"CI95 [{p['ci95_tuples'][0]:+.2f}, {p['ci95_tuples'][1]:+.2f}] tuples "
            f"McNemar p={mcn:.4g} -> {p['R1_verdict'].upper()}")

res["seconds"] = round(time.time() - T0, 1)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, default=str)
ctx.log(f"⭐ SCORED RUN COMPLETE in {res['seconds']}s -> {OUT}")
