"""DEV-25 — the SAME configuration on seeds 1 and 2.

⚠ Seeds 1 and 2 were written by ``gen-feature-bridge`` at ``cache_scope="used"`` — 9,288
rows — while this cell's store pool is 32,768.  ``exp_gen_store.gather_features`` therefore
**extends coverage through the frozen encoder** for the missing rows.  ⭐ That is the
SHIPPED behaviour of the pinned harness (its own docstring: *"Measured constraint, not a
choice"*), it re-runs ``_assert_cache_reencodes`` at full strength at the point of
extension, and it writes **nothing** to the cache file on disk.
⛔ COMMON §4 still binds: the newly encoded rows are produced on THIS box, so a comparison
that mixes them with Mac-produced artefacts is void.  Every number here is produced in one
process from one gather, so the comparison is internally consistent — and the same is true
of the scored run, where the rivals are re-measured beside the arm.

⭐ The full-pool gather is cached to disk per seed so the encode is paid ONCE, not once per
corner.  Corners are then index subsets of it.

⛔ DEV corners only.  eval_a / eval_b are never touched.  Nothing is selected here.
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
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 6
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp25_seed{SEED}.json")
CACHE = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev", "cache")

CORNERS = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=True, shape_value=1.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=3.0, orient_hi=False),
][:max(NC, 1)]
KINDS = ("modes", "atoms", "pca", "random", "mean")

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
F = len(cards)
pool_full = c["pool"]

pth = os.path.join(CACHE, f"pool_feats_s{SEED}_{pool_full.size}.npy")
if os.path.exists(pth):
    Xpool = np.load(pth, mmap_mode="r")
    ctx.log(f"full-pool features loaded from cache {Xpool.shape}")
else:
    t = time.time()
    ctx.log(f"gathering {pool_full.size} pool rows for seed {SEED} "
            "(cache_scope='used' seeds pay an encode here — the SHIPPED path)")
    Xpool = np.asarray(c["gather"](pool_full), np.float32).reshape((-1, D))
    np.save(pth, Xpool)
    ctx.log(f"full-pool gather took {time.time() - t:.0f}s -> cached {Xpool.shape}")
pool_index = {int(r): i for i, r in enumerate(pool_full.tolist())}

from chlu.core.gen_compose import clu_wide_vector, compose_codes, request_posterior  # noqa: E402
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.core.gen_store import address_codes, read_from_codes, store_bytes  # noqa: E402

PAIRS = default_pairs(c["names"], PAIR_NAMES)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("config", {"K": K, "M": M, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES})
res.setdefault("corners", {})
res.setdefault("corners_skipped", {})

for ci, corner in enumerate(CORNERS):
    ck = f"c{ci}"
    if ck in res["corners"] or ck in res["corners_skipped"]:
        ctx.log(f"corner {ci} already handled")
        continue
    t0 = time.time()
    try:
        s = dev.dev_setup(c, corner)
    except dev.HeldOutOverlap as exc:
        ctx.log(f"corner {ci} SKIPPED — {exc}")
        res["corners_skipped"][ck] = {"corner": corner, "reason": str(exc)}
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        continue
    sel = np.array([pool_index[int(r)] for r in s["pool"].tolist()], np.int64)
    Zs = np.asarray(Xpool[sel], np.float32)
    Zdev = np.asarray(c["gather"](s["dev_rows"]), np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    corr, dead, ow, oa = fast.channel_geometry(
        Zs, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
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
            sb, hb = store_bytes(st), head.byte_ledger()
            row["store_bytes"] = int(sb["store_bytes"])
            row["head_bytes"] = int(hb["bytes"])
            row["total_bytes"] = int(sb["store_bytes"] + hb["bytes"])
            ct = address_codes(st, Zs)
            bk = fit_address_book(ct, lab_store, cards, 1.0)
            cd = compose_codes(request_posterior(bk, req))
            row["clu_argmax_same_store"] = dev.score_features(
                c, np.asarray(read_from_codes(st, cd["code"]), np.float32), lab_dev)

    protos = FactorPrototypes.fit(c["grid"], Zs.reshape((-1,) + c["feat_shape"]), s["pool"])
    row["knn_proto"] = dev.score_features(
        c, np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                      np.float32), lab_dev)
    row["knn_proto"]["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    row["seconds"] = round(time.time() - t0, 1)
    res["corners"][ck] = row
    ctx.log(f"[s{SEED} c{ci}] modes={row['modes']['aggregate']:.4f}"
            f"({row['modes']['distinct_factor_tuples']}) atoms={row['atoms']['aggregate']:.4f} "
            f"pca={row['pca']['aggregate']:.4f} random={row['random']['aggregate']:.4f} "
            f"mean={row['mean']['aggregate']:.4f} | "
            f"clu_argmax={row['clu_argmax_same_store']['aggregate']:.4f}"
            f"({row['clu_argmax_same_store']['distinct_factor_tuples']}) "
            f"knn={row['knn_proto']['aggregate']:.4f}"
            f"({row['knn_proto']['distinct_factor_tuples']}) [{row['seconds']}s]")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

rows = list(res["corners"].values())
if rows:
    g = lambda k: np.array([r[k]["aggregate"] for r in rows])
    t_ = lambda k: np.array([r[k]["distinct_factor_tuples"] for r in rows])
    dl = g("modes") - g("knn_proto")
    da = g("modes") - g("atoms")
    res["summary"] = {
        "n_corners": len(rows),
        **{k: {"mean": float(g(k).mean()), "sd": float(g(k).std()),
               "tuples_mean": float(t_(k).mean())}
           for k in list(KINDS) + ["knn_proto", "clu_argmax_same_store"]},
        "delta_vs_knn_mean": float(dl.mean()), "delta_vs_knn_sd": float(dl.std(ddof=1)) if len(dl) > 1 else 0.0,
        "wins_vs_knn": int((dl > 0).sum()),
        "delta_vs_atoms_mean": float(da.mean()),
        "wins_vs_atoms": int((da > 0).sum()),
        "total_bytes": rows[0]["total_bytes"],
    }
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    ss = res["summary"]
    ctx.log(f"[SEED {SEED} FINAL n={ss['n_corners']}] modes={ss['modes']['mean']:.4f}"
            f"±{ss['modes']['sd']:.4f}({ss['modes']['tuples_mean']:.1f}) "
            f"knn={ss['knn_proto']['mean']:.4f}±{ss['knn_proto']['sd']:.4f}"
            f"({ss['knn_proto']['tuples_mean']:.1f}) Dknn={ss['delta_vs_knn_mean']:+.4f} "
            f"({ss['wins_vs_knn']}/{ss['n_corners']}) | atoms={ss['atoms']['mean']:.4f} "
            f"Datoms={ss['delta_vs_atoms_mean']:+.4f} ({ss['wins_vs_atoms']}/{ss['n_corners']}) "
            f"| pca={ss['pca']['mean']:.4f} random={ss['random']['mean']:.4f} "
            f"mean={ss['mean']['mean']:.4f} | argmax={ss['clu_argmax_same_store']['mean']:.4f}"
            f"({ss['clu_argmax_same_store']['tuples_mean']:.1f})")
ctx.log(f"done in {time.time() - T0:.0f}s -> {OUTP}")
