"""DEV-15 — ⭐⭐ THE CANDIDATE ARM, with its whole control set and exact byte ledger.

    request  --Phi-->  (1 + Sum_f V_f + pair cols)
             --E-->    r numbers            ⭐ the ADDRESS
             --U-->    K*M coefficients
             --store-> x = Sum_k Sum_m a_km * mode_km
             --frozen decoder--> image --frozen classifier--> factor tuple

    head_bytes  = (n_cols * r + r * K * M) * 4
    store_bytes = M * D * 4
    total       = head + store          ⭐ compared against knn_proto's 475,136 B

⭐ The read is the shipped ``gen_compose.clu_wide_vector(store, weights)`` verb: a plain
matmul of request-derived weights against the stored modes, with **no feature argument**.
⛔ No labels in the store: the modes are k-means centroids of train activations.

Five bases at IDENTICAL bytes on every row:
  modes   learned k-means centroids        (the arm)
  atoms   M real train rows per well       ⭐ §C.1 matched-bytes lookup floor
  pca     per-well PCA components          ⭐ the sharpest trivial substitute
  random  Gaussian directions              ⭐ the D-matched null
  mean    every mode -> its well mean      ⭐ §C.2 laundering control

⛔ DEV numbers.  eval_a / eval_b are never touched.  All selection happens here.
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
CORNER = sys.argv[2] if len(sys.argv) > 2 else "default"
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp15_seed{SEED}_{CORNER}.json")

corner = None
if CORNER != "default":
    hue, sh, sc_, orn = CORNER.split("_")
    corner = dict(hue_lo=float(hue), shape_value=float(sh),
                  scale_hi=(sc_ == "hi"), orient_hi=(orn == "hi"))

c = ctx.build_context(SEED)
s = dev.dev_setup(c, corner)
ctx.log(f"corner {s['corner']} · removed {s['n_removed_from_pool']} rows from the pool")
X_store = c["gather"](s["pool"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D)).astype(np.float64)
cards = c["cards"]
names = c["names"]
F = len(cards)
corr, dead, order_ward, order_avg = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)

ALL_PAIRS = list(itertools.combinations(range(F), 2))
name2pair = {f"{names[f]}x{names[g]}": (f, g) for f, g in ALL_PAIRS}
# ⭐ INHERITED from DEV-13 (seed 0, the default corner) and ⛔ NOT re-selected per corner
RANKED = [name2pair[k] for k in
          ("floor_huexscale", "object_huexscale", "shapexorientation", "floor_huexshape",
           "wall_huexshape", "object_huexshape")]


def blocks(lab, pairs):
    N = lab.shape[0]
    offs = np.cumsum([0] + cards[:-1])
    A = np.zeros((N, 1 + int(sum(cards))))
    A[:, 0] = 1.0
    A[np.repeat(np.arange(N), F), 1 + (lab + offs[None, :]).reshape(-1)] = 1.0
    if not pairs:
        return A
    Bs = [A]
    for (f, g) in pairs:
        B = np.zeros((N, cards[f] * cards[g]))
        B[np.arange(N), lab[:, f] * cards[g] + lab[:, g]] = 1.0
        Bs.append(B)
    return np.concatenate(Bs, axis=1)


_CACHE = {}


def basis_of(kind, perm, M, labels):
    K = perm.shape[0]
    key = (kind, K, M)
    if key in _CACHE:
        return _CACHE[key]
    if kind == "modes":
        out = fast.kmeans_pq(Zs.astype(np.float32), labels, K, M, SEED,
                             iters=25).modes.astype(np.float64)
    elif kind == "atoms":
        rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916)
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
        rng = np.random.default_rng(int(SEED) * 31 + 5)
        out = rng.normal(size=(K, M, perm.shape[1]))
    elif kind == "mean":
        b = basis_of("modes", perm, M, labels)
        out = np.repeat(b.mean(axis=1, keepdims=True), M, axis=1)
    else:
        raise ValueError(kind)
    _CACHE[key] = out
    return out


def coeffs(P, perm, basis):
    K, M, _ = basis.shape
    C = np.zeros((P.shape[0], K, M))
    for k in range(K):
        Bk = basis[k]
        C[:, k, :] = np.linalg.solve(Bk @ Bk.T + 1e-8 * np.eye(M), Bk @ P[:, perm[k]].T).T
    return C


def render(C, perm, basis, Phi):
    K, M, _ = basis.shape
    a = (Phi @ C.reshape(C.shape[0], K * M)).reshape(-1, K, M)
    out = np.zeros((Phi.shape[0], D))
    for k in range(K):
        out[:, perm[k]] = a[:, k, :] @ basis[k]
    return out.astype(np.float32)


res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("refs", {})
res.setdefault("arm", {})

# --- references on this corner --------------------------------------------
if "knn_proto" not in res["refs"]:
    from chlu.core.gen_request import FactorPrototypes, requests_from_rows
    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    emb = np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                     np.float32)
    sc = dev.score_features(c, emb, lab_dev)
    sc["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    res["refs"]["knn_proto"] = sc
    ctx.log(f"  [knn_proto] agg={sc['aggregate']:.4f} ({sc['distinct_factor_tuples']}/72) "
            f"bytes={sc['deployed_bytes']}")
if "best_constant" not in res["refs"]:
    res["refs"]["best_constant"] = {
        "aggregate": float(np.mean([1.0 / np.unique(lab_dev[:, j]).size for j in range(F)])),
        "distinct_factor_tuples": 1, "what": "structural bound for a constant output"}
    ctx.log(f"  [best_constant bound] {res['refs']['best_constant']['aggregate']:.4f}")

GRID = [(32, 48), (32, 30), (16, 48), (64, 24), (32, 64), (16, 64)]
RANKS = (4, 8, 12, 16, 24)
KINDS = ("modes", "atoms", "pca", "random", "mean")

for npairs in (2, 3):
    pairs = RANKED[:npairs]
    Phi = blocks(lab_store, pairs)
    Phid = blocks(lab_dev, pairs)
    ncols = Phi.shape[1]
    for ridge in (10.0, 100.0):
        W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(ncols), Phi.T @ Zs)
        key_raw = f"raw_p{npairs}_r{ridge:g}"
        if key_raw not in res["refs"]:
            sc = dev.score_features(c, (Phid @ W).astype(np.float32), lab_dev)
            sc["deployed_bytes"] = int(ncols * D * 4)
            res["refs"][key_raw] = sc
            ctx.log(f"  [{key_raw}] agg={sc['aggregate']:.4f} "
                    f"({sc['distinct_factor_tuples']}/72) ⛔ {sc['deployed_bytes']} B")
        for (K, M) in GRID:
            labels = fast.balanced_labels(order_ward, dead, K, D)
            perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
            for kind in KINDS:
                bas = basis_of(kind, perm, M, labels)
                C = coeffs(W, perm, bas).reshape(ncols, K * M)
                U_, S_, Vt_ = np.linalg.svd(C, full_matrices=False)
                for r in RANKS:
                    if r > S_.size:
                        continue
                    tag = f"p{npairs}_ridge{ridge:g}_K{K}_M{M}_r{r}_{kind}"
                    if tag in res["arm"]:
                        continue
                    head_b = int((ncols * r + r * K * M) * 4)
                    store_b = int(M * D * 4)
                    Cr = ((U_[:, :r] * S_[:r]) @ Vt_[:r]).reshape(ncols, K, M)
                    sc = dev.score_features(c, render(Cr, perm, bas, Phid), lab_dev)
                    sc.update({"K": K, "M": M, "rank": r, "basis": kind,
                               "n_pairs": npairs, "ridge": ridge, "n_cols": ncols,
                               "head_bytes": head_b, "store_bytes": store_b,
                               "total_bytes": head_b + store_b,
                               "under_knn_bytes": bool(head_b + store_b < 475_136)})
                    res["arm"][tag] = sc
                    with open(OUTP, "w", encoding="utf-8") as fh:
                        json.dump(res, fh, indent=2)
            # one compact line per (npairs, ridge, K, M): the best legal rank per basis
            msg = []
            for kind in KINDS:
                best = None
                for r in RANKS:
                    t = f"p{npairs}_ridge{ridge:g}_K{K}_M{M}_r{r}_{kind}"
                    v = res["arm"].get(t)
                    if v and v["under_knn_bytes"] and (best is None
                                                       or v["aggregate"] > best["aggregate"]):
                        best = v
                msg.append(f"{kind}={best['aggregate']:.4f}({best['distinct_factor_tuples']},r{best['rank']})"
                           if best else f"{kind}=--")
            ctx.log(f"  [p{npairs} ridge{ridge:g} K{K} M{M}] " + "  ".join(msg))

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
