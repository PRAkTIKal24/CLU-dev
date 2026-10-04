"""DEV-24 — ⭐ ONE configuration, through the PRODUCTION objects, on TWELVE corners.

The configuration is fixed and inherited (DEV-16's best on the selection corners):

    K = 32 · M = 26 · rank = 16 · pairs = (floor_hue x scale, object_hue x scale) · ridge = 10

⛔ Nothing is selected in this script.  It exists to (a) put the arm through the objects a
scored run would use — ``GenStore``, ``gen_factored_read.fit_factored_head``, the shipped
``gen_compose.clu_wide_vector``, ``assert_request_only_factored``, ``assert_no_bypass`` —
and (b) tighten the estimate of Δ against `knn_proto` by widening from 6 corners to 12.

Every corner carries, at IDENTICAL bytes: the atom-basis §C.1 floor, the PCA and random
substitute bases, the §C.2 store-removed control, `knn_proto`, the shipped `clu` argmax
read, and the structural best-constant bound.

⛔ DEV corners only.  eval_a / eval_b are never touched.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx  # noqa: E402
import dev  # noqa: E402
import fast  # noqa: E402

T0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp24_seed{SEED}.json")

# 12 Montero-shaped corners.  c0–c5 are DEV-16's (c0–c2 were the selection set).
# ⭐ EVERY Montero-shaped corner the grid admits (4 hue bands × 4 shapes × 2 scale
# corners × 2 orientation corners).  Corners that do not realise 72 rows, or that touch
# eval_a / eval_b, are REFUSED by the guard and recorded as skipped — ⛔ never trimmed.
# The first 12 are kept in their original order so the earlier rows resume unchanged.
CORNERS = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=True, shape_value=1.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=3.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=False, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.2, scale_hi=False, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.4, scale_hi=False, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=True, shape_value=3.0, orient_hi=True),
] + [dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
     for h in (0.0, 0.2, 0.4, 0.6) for sh in (0.0, 1.0, 2.0, 3.0)
     for sc in (True, False) for o in (True, False)]
KINDS = ("modes", "atoms", "pca", "random", "mean")

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
F = len(cards)

from chlu.core.gen_compose import clu_wide_vector, compose_codes, request_posterior  # noqa: E402
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import (assert_request_only_factored,  # noqa: E402
                                         default_pairs, fit_factored_head)
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.core.gen_store import (address_codes, assert_no_bypass, delete_store,  # noqa: E402
                                 read_from_codes, store_bytes)

PAIRS = default_pairs(c["names"], PAIR_NAMES)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("config", {"K": K, "M": M, "rank": RANK, "ridge": RIDGE,
                          "pairs": PAIR_NAMES})
res.setdefault("corners", {})

_seen = set()
_CORNERS = []
for _c in CORNERS:
    _k = tuple(sorted(_c.items()))
    if _k not in _seen:
        _seen.add(_k)
        _CORNERS.append(_c)
CORNERS = _CORNERS
ctx.log(f"{len(CORNERS)} candidate corners after de-duplication")

for ci, corner in enumerate(CORNERS):
    ck = f"c{ci}"
    if ck in res["corners"]:
        ctx.log(f"corner {ci} already done")
        continue
    t0 = time.time()
    try:
        s = dev.dev_setup(c, corner)
    except (dev.HeldOutOverlap, AssertionError) as exc:
        ctx.log(f"corner {ci} SKIPPED — {exc}")
        res["corners_skipped"] = res.get("corners_skipped", {})
        res["corners_skipped"][ck] = {"corner": corner, "reason": str(exc)}
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        continue
    X_store = c["gather"](s["pool"])
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    labels = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    req = requests_from_labels(lab_dev, "full", 0, SEED)
    row = {"corner": corner,
           "best_constant": float(np.mean([1.0 / np.unique(lab_dev[:, j]).size
                                           for j in range(F)]))}

    bases = {}
    pq = fast.kmeans_pq(Zs, labels, K, M, SEED, iters=25)
    bases["modes"] = pq.modes.astype(np.float32)
    rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916 + 17 * ci)
    ar = rng.choice(Zs.shape[0], M, replace=False)
    bases["atoms"] = np.stack([Zs[ar][:, perm[k]] for k in range(K)]).astype(np.float32)
    o = []
    for k in range(K):
        Y = Zs[: min(8192, Zs.shape[0]), perm[k]]
        _, _, Vt = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
        o.append(Vt[:M])
    bases["pca"] = np.stack(o).astype(np.float32)
    bases["random"] = np.random.default_rng(int(SEED) * 31 + 5 + 17 * ci).normal(
        size=(K, M, perm.shape[1])).astype(np.float32)
    bases["mean"] = np.repeat(bases["modes"].mean(axis=1, keepdims=True), M, axis=1)

    for kind, modes in bases.items():
        st = fast.EqualPQ(perm, modes, (D,)).to_gen_store()
        head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
        feats = np.asarray(clu_wide_vector(st, head.coefficients(req)), np.float32)
        row[kind] = dev.score_features(c, feats, lab_dev)
        if kind == "modes":
            sb = store_bytes(st)
            hb = head.byte_ledger()
            row["store_bytes"] = int(sb["store_bytes"])
            row["head_bytes"] = int(hb["bytes"])
            row["total_bytes"] = int(sb["store_bytes"] + hb["bytes"])
            row["channel_width_reals"] = int(head.channel_ledger()["width"])
            if ci == 0:
                ro = assert_request_only_factored(st, head, req, decoy_features=Zs[:64])
                nb = assert_no_bypass(st, Zs[:64], rng_seed=SEED)
                res["assertions"] = {
                    "assert_request_only_factored": {k: v for k, v in ro.items()
                                                     if k != "signatures"},
                    "assert_no_bypass": {k: v for k, v in nb.items() if k != "note"}}
                ctx.log("⭐ assertions PASS: request-only (independent re-implementation "
                        f"max|delta| {ro['max_abs_delta_vs_independent_sum']:.2e}), "
                        f"no-bypass (pure codebook gather {nb['read_is_pure_codebook_gather']})")
            # ⭐ the SHIPPED argmax read on the SAME store, for the like-for-like column
            ct = address_codes(st, Zs)
            bk = fit_address_book(ct, lab_store, cards, 1.0)
            cd = compose_codes(request_posterior(bk, req))
            row["clu_argmax_same_store"] = dev.score_features(
                c, np.asarray(read_from_codes(st, cd["code"]), np.float32), lab_dev)

    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    row["knn_proto"] = dev.score_features(
        c, np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                      np.float32), lab_dev)
    row["knn_proto"]["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    row["seconds"] = round(time.time() - t0, 1)
    res["corners"][ck] = row
    ctx.log(f"[c{ci}] modes={row['modes']['aggregate']:.4f}({row['modes']['distinct_factor_tuples']}) "
            f"atoms={row['atoms']['aggregate']:.4f} pca={row['pca']['aggregate']:.4f} "
            f"random={row['random']['aggregate']:.4f} mean={row['mean']['aggregate']:.4f} "
            f"| clu_argmax={row['clu_argmax_same_store']['aggregate']:.4f}"
            f"({row['clu_argmax_same_store']['distinct_factor_tuples']}) "
            f"knn={row['knn_proto']['aggregate']:.4f}({row['knn_proto']['distinct_factor_tuples']}) "
            f"| total={row['total_bytes']} [{row['seconds']}s]")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

rows = list(res["corners"].values())
g = lambda k: np.array([r[k]["aggregate"] for r in rows])
t = lambda k: np.array([r[k]["distinct_factor_tuples"] for r in rows])
kn = g("knn_proto")
summ = {"n_corners": len(rows),
        "best_constant": float(np.mean([r["best_constant"] for r in rows]))}
for k in list(KINDS) + ["knn_proto", "clu_argmax_same_store"]:
    summ[k] = {"mean": float(g(k).mean()), "sd": float(g(k).std()),
               "tuples_mean": float(t(k).mean())}
summ["delta_modes_vs_knn"] = float(g("modes").mean() - kn.mean())
summ["paired_wins_vs_knn"] = int((g("modes") > kn).sum())
summ["delta_modes_vs_atoms"] = float(g("modes").mean() - g("atoms").mean())
summ["paired_wins_vs_atoms"] = int((g("modes") > g("atoms")).sum())
summ["delta_modes_vs_pca"] = float(g("modes").mean() - g("pca").mean())
summ["paired_wins_vs_pca"] = int((g("modes") > g("pca")).sum())
summ["delta_modes_vs_clu_argmax"] = float(g("modes").mean() - g("clu_argmax_same_store").mean())
summ["total_bytes"] = rows[0]["total_bytes"]
res["summary"] = summ
res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"[FINAL n={summ['n_corners']}] modes={summ['modes']['mean']:.4f}±{summ['modes']['sd']:.4f}"
        f"({summ['modes']['tuples_mean']:.1f}) | knn={summ['knn_proto']['mean']:.4f}"
        f"±{summ['knn_proto']['sd']:.4f}({summ['knn_proto']['tuples_mean']:.1f}) "
        f"Dknn={summ['delta_modes_vs_knn']:+.4f} ({summ['paired_wins_vs_knn']}/{summ['n_corners']}) "
        f"| atoms={summ['atoms']['mean']:.4f} Datoms={summ['delta_modes_vs_atoms']:+.4f} "
        f"({summ['paired_wins_vs_atoms']}/{summ['n_corners']}) "
        f"| pca={summ['pca']['mean']:.4f} random={summ['random']['mean']:.4f} "
        f"mean={summ['mean']['mean']:.4f} | clu_argmax={summ['clu_argmax_same_store']['mean']:.4f}"
        f"({summ['clu_argmax_same_store']['tuples_mean']:.1f}) "
        f"| best_const={summ['best_constant']:.4f} | total_bytes={summ['total_bytes']}")
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
