"""DEV-13 — which interactions carry the +0.21, and can the STORE afford them?

Part A (raw feature space, ⛔ unaffordable, a reference): additive + ONE factor pair at a
time, then additive + the top-j pairs.  Ranks the interactions by what they are worth.

Part B (the arm): the same head expressed in the STORE'S BASIS —
``x = Σ_k Σ_m a_{k,m}(request)·mode_{k,m}`` with ``a`` additive **plus the selected pairs**.
    store = M·D·4          head = n_cols·K·M·4        total = 4M(D + n_cols·K)
Every row carries its **§C.1 atom-basis floor** (the identical head over M real train rows)
and its **§C.2 laundering control** (every mode replaced by its well mean).

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
PART = sys.argv[2] if len(sys.argv) > 2 else "AB"
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp13_seed{SEED}.json")

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

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("pairA", {})
res.setdefault("topj", {})
res.setdefault("armB", {})

ALL_PAIRS = list(itertools.combinations(range(F), 2))


def design(lab, pairs):
    N = lab.shape[0]
    offs = np.cumsum([0] + cards[:-1])
    blocks = [np.ones((N, 1))]
    A = np.zeros((N, int(sum(cards))))
    A[np.repeat(np.arange(N), F), (lab + offs[None, :]).reshape(-1)] = 1.0
    blocks.append(A)
    for (f, g) in pairs:
        B = np.zeros((N, cards[f] * cards[g]))
        B[np.arange(N), lab[:, f] * cards[g] + lab[:, g]] = 1.0
        blocks.append(B)
    return np.concatenate(blocks, axis=1)


def ncols(pairs):
    return 1 + int(sum(cards)) + int(sum(cards[f] * cards[g] for f, g in pairs))


def fit_raw(pairs, ridge):
    Phi = design(lab_store, pairs)
    Phid = design(lab_dev, pairs)
    W = np.linalg.solve(Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1]), Phi.T @ Zs)
    return (Phid @ W).astype(np.float32), Phi.shape[1]


if "A" in PART:
    for (f, g) in ALL_PAIRS:
        key = f"{names[f]}x{names[g]}"
        if key in res["pairA"]:
            continue
        best = None
        for r in (10.0, 100.0):
            pred, nc = fit_raw([(f, g)], r)
            sc = dev.score_features(c, pred, lab_dev)
            sc["ridge"] = r
            sc["n_columns"] = nc
            if best is None or sc["aggregate"] > best["aggregate"]:
                best = sc
        res["pairA"][key] = best
        ctx.log(f"  [+{key}] agg={best['aggregate']:.4f} "
                f"({best['distinct_factor_tuples']}/72) ridge={best['ridge']:g} "
                f"cols={best['n_columns']}")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

    rank = sorted(res["pairA"].items(), key=lambda kv: -kv[1]["aggregate"])
    ctx.log("pair ranking: " + ", ".join(f"{k}={v['aggregate']:.4f}" for k, v in rank))
    name2pair = {f"{names[f]}x{names[g]}": (f, g) for f, g in ALL_PAIRS}
    for j in (1, 2, 3, 4, 5, 6, 8, 10):
        key = f"top{j}"
        if key in res["topj"]:
            continue
        pairs = [name2pair[k] for k, _ in rank[:j]]
        best = None
        for r in (10.0, 100.0, 300.0):
            pred, nc = fit_raw(pairs, r)
            sc = dev.score_features(c, pred, lab_dev)
            sc["ridge"] = r
            sc["n_columns"] = nc
            sc["pairs"] = [f"{names[a]}x{names[b]}" for a, b in pairs]
            if best is None or sc["aggregate"] > best["aggregate"]:
                best = sc
        res["topj"][key] = best
        ctx.log(f"  [top{j}] agg={best['aggregate']:.4f} "
                f"({best['distinct_factor_tuples']}/72) cols={best['n_columns']} "
                f"ridge={best['ridge']:g} bytes_raw={best['n_columns']*D*4}")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

# ---------------------------------------------------------------------------
if "B" in PART:
    rank = sorted(res["pairA"].items(), key=lambda kv: -kv[1]["aggregate"])
    name2pair = {f"{names[f]}x{names[g]}": (f, g) for f, g in ALL_PAIRS}
    rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916)

    def project(P, perm, basis):
        out = np.zeros_like(P)
        for k in range(perm.shape[0]):
            w = perm[k]
            Bk = basis[k]
            gram = Bk @ Bk.T + 1e-8 * np.eye(Bk.shape[0])
            out[:, w] = (Bk.T @ np.linalg.solve(gram, Bk @ P[:, w].T)).T
        return out

    GRID = []
    for npairs in (1, 2, 3, 4):
        pairs = [name2pair[k] for k, _ in rank[:npairs]]
        nc = ncols(pairs)
        for K in (8, 16, 32, 64):
            M = int(475_136 // (4 * (D + nc * K)))
            if M >= 4:
                GRID.append((K, min(M, 64), npairs))
    for (K, M, npairs) in GRID:
        tag = f"K{K}_M{M}_p{npairs}"
        if tag in res["armB"]:
            continue
        t = time.time()
        pairs = [name2pair[k] for k, _ in rank[:npairs]]
        nc = ncols(pairs)
        Phi = design(lab_store, pairs)
        Phid = design(lab_dev, pairs)
        labels = fast.balanced_labels(order_ward, dead, K, D)
        perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
        pq = fast.kmeans_pq(Zs.astype(np.float32), labels, K, M, SEED, iters=25)
        modes = pq.modes.astype(np.float64)
        ar = rng.choice(Zs.shape[0], M, replace=False)
        atoms = np.stack([Zs[ar][:, perm[k]] for k in range(K)])
        mm = np.repeat(modes.mean(axis=1, keepdims=True), M, axis=1)
        store_b = int(M * D * 4)
        head_b = int(nc * K * M * 4)
        tot = store_b + head_b
        row = {"K": K, "M": M, "n_pairs": npairs, "n_columns": nc,
               "pairs": [f"{names[a]}x{names[b]}" for a, b in pairs],
               "store_bytes": store_b, "head_bytes": head_b, "total_bytes": tot,
               "beats_knn_bytes": bool(tot < 475_136)}
        best = None
        for r in (10.0, 100.0, 300.0):
            W = np.linalg.solve(Phi.T @ Phi + r * np.eye(Phi.shape[1]), Phi.T @ Zs)
            armp = project(W, perm, modes)
            sc = dev.score_features(c, (Phid @ armp).astype(np.float32), lab_dev)
            sc["ridge"] = r
            if best is None or sc["aggregate"] > best["aggregate"]:
                best, bestW = sc, W
        row["arm"] = best
        row["atom_basis_floor"] = dev.score_features(
            c, (Phid @ project(bestW, perm, atoms)).astype(np.float32), lab_dev)
        row["store_removed"] = dev.score_features(
            c, (Phid @ project(bestW, perm, mm)).astype(np.float32), lab_dev)
        row["seconds"] = round(time.time() - t, 1)
        res["armB"][tag] = row
        ctx.log(f"  [{tag}] arm={best['aggregate']:.4f}({best['distinct_factor_tuples']}) "
                f"atomfloor={row['atom_basis_floor']['aggregate']:.4f}"
                f"({row['atom_basis_floor']['distinct_factor_tuples']}) "
                f"sr={row['store_removed']['aggregate']:.4f}"
                f"({row['store_removed']['distinct_factor_tuples']}) "
                f"| cols={nc} store={store_b} head={head_b} total={tot} "
                f"{'⭐under' if tot < 475136 else '⚠over'} [{row['seconds']}s]")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
