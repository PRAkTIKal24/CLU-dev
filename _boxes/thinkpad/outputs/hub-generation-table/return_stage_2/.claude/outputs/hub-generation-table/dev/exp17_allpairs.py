"""DEV-16 — ⭐ the SHORT LIST, re-measured on MANY corners.  ⛔ No maxima over the search.

DEV-14 took a maximum over several hundred configurations on 72 items.  That number is
optimistically biased by construction.  This re-measures a short, fixed list of
configurations on **several Montero-shaped DEV corners**, and reports mean ± sd across
corners, with the basis controls at **matched configuration** on every row.

Configurations and the pair ranking are INHERITED and ⛔ not re-selected here.

⛔ DEV numbers.  eval_a / eval_b are never touched.
"""
import itertools
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
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp17_seed{SEED}.json")

# (K, M, rank, n_pairs, ridge) — ⭐ fixed before this script ran
# ⭐ ALL 15 pairs — ⛔ NO pair selection, so nothing is tuned to one corner.
# head = (n_add*K*M + n_pair*r + r*K*M)*4 ; store = M*D*4 ; both under knn_proto's 475,136 B.
CONFIGS = [
    (32, 26, 8, 15, 10.0),
    (32, 26, 8, 15, 100.0),
    (32, 24, 12, 15, 10.0),
    (64, 14, 8, 15, 10.0),
    (16, 30, 8, 15, 10.0),
    (32, 26, 4, 15, 10.0),
]
KINDS = ("modes", "atoms", "pca", "random", "mean")

CORNERS = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
]

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
names = c["names"]
F = len(cards)
ALL_PAIRS = list(itertools.combinations(range(F), 2))
name2pair = {f"{names[f]}x{names[g]}": (f, g) for f, g in ALL_PAIRS}
# ⭐ ALL pairs, in the DEV-13 ranking so a truncation is still meaningful.
RANKED = [name2pair[k] for k in
          ("floor_huexscale", "object_huexscale", "shapexorientation",
           "floor_huexshape", "wall_huexshape", "object_huexshape",
           "scalexorientation", "wall_huexorientation", "floor_huexobject_hue",
           "object_huexorientation", "floor_huexorientation", "floor_huexwall_hue",
           "wall_huexobject_hue", "wall_huexscale", "scalexshape")]

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("configs", [list(x) for x in CONFIGS])
res.setdefault("corners", {})


def blocks(lab, pairs):
    """(additive design, pair design) — ⛔ kept SEPARATE.

    DEV-15 measured that SVD-truncating the ADDITIVE block destroys the head (the same
    failure as DEV-05b). ⭐ Only the INTERACTION block is factorised; the additive block
    is deployed at full rank, exactly as in DEV-14.
    """
    N = lab.shape[0]
    offs = np.cumsum([0] + cards[:-1])
    A = np.zeros((N, 1 + int(sum(cards))))
    A[:, 0] = 1.0
    A[np.repeat(np.arange(N), F), 1 + (lab + offs[None, :]).reshape(-1)] = 1.0
    if not pairs:
        return A, np.zeros((N, 0))
    Bs = []
    for (f, g) in pairs:
        B = np.zeros((N, cards[f] * cards[g]))
        B[np.arange(N), lab[:, f] * cards[g] + lab[:, g]] = 1.0
        Bs.append(B)
    return A, np.concatenate(Bs, axis=1)


for ci, corner in enumerate(CORNERS):
    ckey = f"c{ci}"
    if ckey in res["corners"] and res["corners"][ckey].get("done"):
        ctx.log(f"corner {ci} already done — skipped")
        continue
    t0 = time.time()
    s = dev.dev_setup(c, corner)
    X_store = c["gather"](s["pool"])
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    Zs = np.asarray(X_store, np.float32).reshape((-1, D)).astype(np.float64)
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=ctx.log)
    row = res["corners"].get(ckey, {})
    row["corner"] = corner
    row["n_removed"] = s["n_removed_from_pool"]

    from chlu.core.gen_request import FactorPrototypes, requests_from_rows
    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    emb = np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                     np.float32)
    kp = dev.score_features(c, emb, lab_dev)
    kp["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    row["knn_proto"] = kp
    row["best_constant"] = float(np.mean([1.0 / np.unique(lab_dev[:, j]).size
                                          for j in range(F)]))
    ctx.log(f"corner {ci} {corner}: knn_proto={kp['aggregate']:.4f}"
            f"({kp['distinct_factor_tuples']}/72) best_const={row['best_constant']:.4f}")

    cache = {}

    def basis_of(kind, perm, M, labels):
        key = (kind, perm.shape[0], M)
        if key in cache:
            return cache[key]
        K = perm.shape[0]
        if kind == "modes":
            out = fast.kmeans_pq(Zs.astype(np.float32), labels, K, M, SEED,
                                 iters=25).modes.astype(np.float64)
        elif kind == "atoms":
            rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916 + 17 * ci)
            ar = rng.choice(Zs.shape[0], M, replace=False)
            out = np.stack([Zs[ar][:, perm[k]] for k in range(K)])
        elif kind == "pca":
            o = []
            for k in range(K):
                Y = Zs[: min(8192, Zs.shape[0]), perm[k]]
                _, _, Vt = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
                o.append(Vt[:M])
            out = np.stack(o)
        elif kind == "random":
            rng = np.random.default_rng(int(SEED) * 31 + 5 + 17 * ci)
            out = rng.normal(size=(K, M, perm.shape[1]))
        elif kind == "mean":
            b = basis_of("modes", perm, M, labels)
            out = np.repeat(b.mean(axis=1, keepdims=True), M, axis=1)
        else:
            raise ValueError(kind)
        cache[key] = out
        return out

    for (K, M, r, npairs, ridge) in CONFIGS:
        tag = f"K{K}_M{M}_r{r}_p{npairs}_ridge{ridge:g}"
        if tag in row:
            continue
        pairs = RANKED[:npairs]
        A, B = blocks(lab_store, pairs)
        Ad, Bd = blocks(lab_dev, pairs)
        Phi = np.concatenate([A, B], axis=1)
        Phid = np.concatenate([Ad, Bd], axis=1)
        na, nb = A.shape[1], B.shape[1]
        nc = na + nb
        W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(nc), Phi.T @ Zs)
        labels = fast.balanced_labels(ow, dead, K, D)
        perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
        # ⭐ additive block FULL rank; interaction block factorised to rank r
        head_b = int((na * K * M + nb * r + r * K * M) * 4)
        store_b = int(M * D * 4)
        entry = {"head_bytes": head_b, "store_bytes": store_b,
                 "total_bytes": head_b + store_b,
                 "under_knn_bytes": bool(head_b + store_b < 475_136),
                 "n_cols_additive": na, "n_cols_pairs": nb}
        for kind in KINDS:
            bas = basis_of(kind, perm, M, labels)
            C = np.zeros((nc, K, M))
            for k in range(K):
                Bk = bas[k]
                C[:, k, :] = np.linalg.solve(Bk @ Bk.T + 1e-8 * np.eye(M),
                                             Bk @ W[:, perm[k]].T).T
            Cb = C[na:].reshape(nb, K * M)
            U_, S_, Vt_ = np.linalg.svd(Cb, full_matrices=False)
            rr = min(r, S_.size)
            C[na:] = ((U_[:, :rr] * S_[:rr]) @ Vt_[:rr]).reshape(nb, K, M)
            a = (Phid @ C.reshape(nc, K * M)).reshape(-1, K, M)
            out = np.zeros((Phid.shape[0], D))
            for k in range(K):
                out[:, perm[k]] = a[:, k, :] @ bas[k]
            entry[kind] = dev.score_features(c, out.astype(np.float32), lab_dev)
        row[tag] = entry
        ctx.log(f"  [c{ci} {tag}] " + "  ".join(
            f"{k}={entry[k]['aggregate']:.4f}({entry[k]['distinct_factor_tuples']})"
            for k in KINDS) + f"  total={entry['total_bytes']}"
            + (" ⭐under" if entry["under_knn_bytes"] else " ⚠over"))
        res["corners"][ckey] = row
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    row["done"] = True
    row["seconds"] = round(time.time() - t0, 1)
    res["corners"][ckey] = row
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

# --- the summary across corners -------------------------------------------
summary = {}
for (K, M, r, npairs, ridge) in CONFIGS:
    tag = f"K{K}_M{M}_r{r}_p{npairs}_ridge{ridge:g}"
    acc = {k: [] for k in KINDS}
    tup = {k: [] for k in KINDS}
    kp, bc = [], []
    for ckey, row in res["corners"].items():
        if tag not in row:
            continue
        for k in KINDS:
            acc[k].append(row[tag][k]["aggregate"])
            tup[k].append(row[tag][k]["distinct_factor_tuples"])
        kp.append(row["knn_proto"]["aggregate"])
        bc.append(row["best_constant"])
    if not kp:
        continue
    summary[tag] = {
        "n_corners": len(kp),
        "knn_proto_mean": float(np.mean(kp)), "knn_proto_sd": float(np.std(kp)),
        "best_constant_mean": float(np.mean(bc)),
        **{f"{k}_mean": float(np.mean(acc[k])) for k in KINDS},
        **{f"{k}_sd": float(np.std(acc[k])) for k in KINDS},
        **{f"{k}_tuples_mean": float(np.mean(tup[k])) for k in KINDS},
        "delta_modes_vs_knn": float(np.mean(acc["modes"]) - np.mean(kp)),
        "delta_modes_vs_atoms": float(np.mean(acc["modes"]) - np.mean(acc["atoms"])),
        "delta_modes_vs_pca": float(np.mean(acc["modes"]) - np.mean(acc["pca"])),
        "paired_win_vs_knn": int(sum(1 for a, b in zip(acc["modes"], kp) if a > b)),
        "paired_win_vs_atoms": int(sum(1 for a, b in zip(acc["modes"], acc["atoms"]) if a > b)),
    }
    ss = summary[tag]
    ctx.log(f"[SUMMARY {tag}] n={ss['n_corners']} "
            f"modes={ss['modes_mean']:.4f}±{ss['modes_sd']:.4f}({ss['modes_tuples_mean']:.1f}) "
            f"atoms={ss['atoms_mean']:.4f}±{ss['atoms_sd']:.4f} "
            f"pca={ss['pca_mean']:.4f} random={ss['random_mean']:.4f} "
            f"mean={ss['mean_mean']:.4f} | knn={ss['knn_proto_mean']:.4f}±{ss['knn_proto_sd']:.4f} "
            f"| Dknn={ss['delta_modes_vs_knn']:+.4f} ({ss['paired_win_vs_knn']}/{ss['n_corners']}) "
            f"Datoms={ss['delta_modes_vs_atoms']:+.4f} ({ss['paired_win_vs_atoms']}/{ss['n_corners']})")
res["summary"] = summary
res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
