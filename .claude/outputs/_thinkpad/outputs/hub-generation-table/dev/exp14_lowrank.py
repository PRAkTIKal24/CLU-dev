"""DEV-14 — the interaction head, made affordable by the store, with ALL the controls.

DEV-13 part B failed for a stateable reason: a FULL interaction head in the store's basis
costs ``n_cols·K·M·4``, so at a fixed total budget it starves the basis, and a small basis
destroys the interaction model's gain (raw 0.8773 -> 0.7616 projected).

The repair is to stop paying ``n_cols × K·M`` for the interactions.  Write the head as

    a(request) = A_add[request]                       (58 x K·M, the additive part)
               + (P_pair[request] @ E) @ U            (⭐ rank-r interaction part)

    head_bytes = (58·K·M  +  n_pair_cols·r  +  r·K·M) · 4
    store      = M·D·4
    total      = head + store

⭐ The interactions now cost ``r·K·M`` instead of ``n_pair_cols·K·M`` — and ⛔ in RAW
feature space the same trick costs ``r·D``, which does not help nearly as much because
``D = 2048`` is large and fixed, whereas ``K·M`` is the store's own compression.

Controls on every row, all at the SAME bytes:
  atom_basis   M real train rows per well          ⭐ §C.1 lookup floor
  pca_basis    per-well PCA components             ⭐ the sharpest trivial substitute for
                                                      a dictionary; ⛔ NOT charter-legal as
                                                      a store (§B.1 wants atoms/basins) but
                                                      it is exactly the laundering control
  random_basis per-well Gaussian random directions ⭐ the D-matched null
  store_removed every mode -> its well mean        ⭐ §C.2

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
MODE = sys.argv[2] if len(sys.argv) > 2 else "cap,arm"
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp14_seed{SEED}.json")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D)).astype(np.float64)
Zd = np.asarray(X_dev, np.float32).reshape((-1, D)).astype(np.float64)
cards = c["cards"]
names = c["names"]
F = len(cards)
corr, dead, order_ward, order_avg = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)

# ⭐ pair ranking is INHERITED from DEV-13 so nothing is re-selected here
rank13 = json.load(open(os.path.join(os.path.dirname(OUTP), f"exp13_seed{SEED}.json"),
                        encoding="utf-8"))["pairA"]
ALL_PAIRS = list(itertools.combinations(range(F), 2))
name2pair = {f"{names[f]}x{names[g]}": (f, g) for f, g in ALL_PAIRS}
RANKED = [name2pair[k] for k, _ in sorted(rank13.items(), key=lambda kv: -kv[1]["aggregate"])]
ctx.log("inherited pair ranking: " + ", ".join(f"{names[a]}x{names[b]}" for a, b in RANKED[:6]))


def blocks(lab, pairs):
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


def fit_raw(pairs, ridge):
    A, B = blocks(lab_store, pairs)
    Phi = np.concatenate([A, B], axis=1)
    W = np.linalg.solve(Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1]), Phi.T @ Zs)
    return W[: A.shape[1]], W[A.shape[1]:]


_MODE_CACHE = {}


def basis_of(kind, perm, M, rng, labels=None):
    K = perm.shape[0]
    if kind == "modes":
        key = (K, M)
        if key not in _MODE_CACHE:
            pq = fast.kmeans_pq(Zs.astype(np.float32), labels, K, M, SEED, iters=25)
            _MODE_CACHE[key] = pq.modes.astype(np.float64)
        return _MODE_CACHE[key]
    if kind == "atoms":
        ar = rng.choice(Zs.shape[0], M, replace=False)
        return np.stack([Zs[ar][:, perm[k]] for k in range(K)])
    if kind == "pca":
        out = []
        for k in range(K):
            Y = Zs[:, perm[k]]
            Yc = Y - Y.mean(axis=0)
            _, _, Vt = np.linalg.svd(Yc[: min(8192, Yc.shape[0])], full_matrices=False)
            out.append(Vt[:M])
        return np.stack(out)
    if kind == "random":
        return rng.normal(size=(K, M, perm.shape[1]))
    raise ValueError(kind)


def project(P, perm, basis):
    out = np.zeros_like(P)
    for k in range(perm.shape[0]):
        w = perm[k]
        Bk = basis[k]
        gram = Bk @ Bk.T + 1e-8 * np.eye(Bk.shape[0])
        out[:, w] = (Bk.T @ np.linalg.solve(gram, Bk @ P[:, w].T)).T
    return out


def coeffs(P, perm, basis):
    """P (cols, D) -> C (cols, K, M) coefficients in the basis."""
    cols = P.shape[0]
    K, M, _ = basis.shape
    C = np.zeros((cols, K, M))
    for k in range(K):
        Bk = basis[k]
        gram = Bk @ Bk.T + 1e-8 * np.eye(M)
        C[:, k, :] = np.linalg.solve(gram, Bk @ P[:, perm[k]].T).T
    return C


def render_from_coeffs(C, perm, basis, Phi_blocks):
    A, B = Phi_blocks
    X = np.concatenate([A, B], axis=1) if B.shape[1] else A
    K, M, d = basis.shape
    a = X @ C.reshape(C.shape[0], K * M)            # (n, K*M)
    a = a.reshape(-1, K, M)
    out = np.zeros((X.shape[0], D))
    for k in range(K):
        out[:, perm[k]] = a[:, k, :] @ basis[k]
    return out.astype(np.float32)


res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("cap", {})
res.setdefault("arm", {})
rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916)

CAP_GRID = [(32, 26), (32, 30), (16, 48), (32, 48), (64, 16), (16, 32)]
NP_LIST = [2, 3]

if "cap" in MODE:
    for npairs in NP_LIST:
        pairs = RANKED[:npairs]
        Wa, Wb = fit_raw(pairs, 10.0)
        Wfull = np.concatenate([Wa, Wb], axis=0)
        Ad, Bd = blocks(lab_dev, pairs)
        raw = dev.score_features(c, (np.concatenate([Ad, Bd], axis=1) @ Wfull).astype(np.float32),
                                 lab_dev)
        res["cap"][f"raw_p{npairs}"] = raw
        ctx.log(f"  [raw p{npairs}] agg={raw['aggregate']:.4f} "
                f"({raw['distinct_factor_tuples']}/72) cols={Wfull.shape[0]}")
        for (K, M) in CAP_GRID:
            tag = f"p{npairs}_K{K}_M{M}"
            if tag in res["cap"]:
                continue
            labels = fast.balanced_labels(order_ward, dead, K, D)
            perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
            bas = basis_of("modes", perm, M, rng, labels)
            Wp = project(Wfull, perm, bas)
            sc = dev.score_features(
                c, (np.concatenate([Ad, Bd], axis=1) @ Wp).astype(np.float32), lab_dev)
            sc["store_bytes"] = int(M * D * 4)
            sc["full_head_bytes"] = int(Wfull.shape[0] * K * M * 4)
            res["cap"][tag] = sc
            ctx.log(f"  [cap {tag}] agg={sc['aggregate']:.4f} "
                    f"({sc['distinct_factor_tuples']}/72) basis={K*M} "
                    f"store={sc['store_bytes']} full_head={sc['full_head_bytes']}")
            with open(OUTP, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2)

if "arm" in MODE:
    for npairs in NP_LIST:
        pairs = RANKED[:npairs]
        for ridge in (10.0, 100.0):
            Wa, Wb = fit_raw(pairs, ridge)
            Ad, Bd = blocks(lab_dev, pairs)
            for (K, M) in CAP_GRID:
                labels = fast.balanced_labels(order_ward, dead, K, D)
                perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
                for kind in ("modes", "atoms", "pca", "random", "mean"):
                    if kind == "mean":
                        bm = basis_of("modes", perm, M, np.random.default_rng(0), labels)
                        bas = np.repeat(bm.mean(axis=1, keepdims=True), M, axis=1)
                    else:
                        bas = basis_of(kind, perm, M, np.random.default_rng(int(SEED) * 31 + 5), labels)
                    Ca = coeffs(Wa, perm, bas)                # (58, K, M)
                    Cb = coeffs(Wb, perm, bas)                # (npair_cols, K, M)
                    Cbf = Cb.reshape(Cb.shape[0], K * M)
                    U_, S_, Vt_ = np.linalg.svd(Cbf, full_matrices=False)
                    for r in (4, 8, 16, 32):
                        if r > S_.size:
                            continue
                        head_b = int((Ca.size + Cb.shape[0] * r + r * K * M) * 4)
                        store_b = int(M * D * 4)
                        tot = head_b + store_b
                        Cbr = ((U_[:, :r] * S_[:r]) @ Vt_[:r]).reshape(Cb.shape)
                        C = np.concatenate([Ca, Cbr], axis=0)
                        tag = f"p{npairs}_r{ridge:g}_K{K}_M{M}_rank{r}_{kind}"
                        if tag in res["arm"]:
                            continue
                        sc = dev.score_features(
                            c, render_from_coeffs(C, perm, bas, (Ad, Bd)), lab_dev)
                        sc.update({"K": K, "M": M, "rank": r, "kind": kind,
                                   "n_pairs": npairs, "ridge": ridge,
                                   "store_bytes": store_b, "head_bytes": head_b,
                                   "total_bytes": tot,
                                   "under_knn_bytes": bool(tot < 475_136)})
                        res["arm"][tag] = sc
                        if kind == "modes" or sc["aggregate"] > 0.70:
                            ctx.log(f"  [{tag}] agg={sc['aggregate']:.4f}"
                                    f"({sc['distinct_factor_tuples']}) total={tot} "
                                    f"{'⭐under' if tot < 475136 else '⚠over'}")
                        with open(OUTP, "w", encoding="utf-8") as fh:
                            json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
