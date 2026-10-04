"""DEV-20 — ⭐⭐ a SMOOTH request basis: 58 one-hot columns -> ~23 structured ones.

The head costs ``n_add·K·M + n_pair·r + r·K·M``.  Every column of the request design is
paid for ``K·M`` times, so the design's WIDTH is the binding constraint on how much span
and how much interaction rank the budget can buy.

Shapes3D's factors are not arbitrary categories:
  floor/wall/object hue   10 values on a CIRCLE  -> a few Fourier harmonics
  scale                    8 ordered values      -> a low-degree polynomial
  orientation             15 ordered values      -> a low-degree polynomial
  shape                    4 CATEGORIES          -> ⛔ left as one-hot, it is not ordered

    additive columns   58 -> 23
    pair columns     1322 -> ~201   (tensor products of the marginal bases)

⭐ Two things follow.  (1) The same budget now buys a 1,280-dimension span AND interaction
rank 32, instead of an 832-dimension span at rank 16.  (2) A smooth basis **extrapolates**
into an unseen cell rather than interpolating a table of cells, which is what a
compositional hold-out asks for.

Also tested: fitting the head DIRECTLY in the store's coefficient space (so the ridge
penalises coefficients, not feature-space weights) instead of fitting in feature space and
projecting.

⛔ DEV numbers.  eval_a / eval_b are never touched.
⭐ Selection corners c0–c2; ⛔ HELD-OUT corners c3–c5 are reported separately and nothing
is chosen on them.
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
                    f"exp21_seed{SEED}.json")

CORNERS = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
]
# (K, M, rank, ridge, fit_space)
CONFIGS = [
    (256, 8, 24, 10.0, "feature"),
    (512, 4, 28, 10.0, "feature"),
    (256, 6, 40, 10.0, "feature"),
    (128, 10, 46, 10.0, "feature"),
    (256, 8, 24, 100.0, "feature"),
    (256, 8, 24, 10.0, "coeff"),
]
KINDS = ("modes", "atoms", "pca", "random", "mean")

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
names = c["names"]
F = len(cards)


def marginal_basis(v: np.ndarray, kind: str, card: int) -> np.ndarray:
    """Per-factor basis, ⛔ excluding the constant (it lives once, in the bias)."""
    x = np.asarray(v, np.float64)
    if kind == "circular":            # hues: 2 harmonics -> 4 columns
        t = 2.0 * np.pi * x / card
        return np.stack([np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)], axis=1)
    if kind == "poly3":               # scale
        u = 2.0 * x / (card - 1) - 1.0
        return np.stack([u, u ** 2, u ** 3], axis=1)
    if kind == "poly4":               # orientation
        u = 2.0 * x / (card - 1) - 1.0
        return np.stack([u, u ** 2, u ** 3, u ** 4], axis=1)
    if kind == "onehot":              # shape — ⛔ categorical, left alone
        B = np.zeros((x.size, card - 1))
        idx = x.astype(np.int64)
        m = idx > 0
        B[np.flatnonzero(m), idx[m] - 1] = 1.0
        return B
    raise ValueError(kind)


BASIS_KIND = ["circular", "circular", "circular", "poly3", "onehot", "poly4"]


def design(lab: np.ndarray, with_pairs: bool = True):
    N = lab.shape[0]
    margs = [marginal_basis(lab[:, f], BASIS_KIND[f], cards[f]) for f in range(F)]
    A = np.concatenate([np.ones((N, 1))] + margs, axis=1)
    if not with_pairs:
        return A, np.zeros((N, 0))
    Bs = []
    for f, g in itertools.combinations(range(F), 2):
        Bs.append((margs[f][:, :, None] * margs[g][:, None, :]).reshape(N, -1))
    return A, np.concatenate(Bs, axis=1)


_A0, _B0 = design(np.zeros((1, F), np.int64))
N_ADD, N_PAIR = _A0.shape[1], _B0.shape[1]
ctx.log(f"smooth design: additive {N_ADD} columns (was 58), pairs {N_PAIR} (was 1322)")

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("design", {"n_additive": int(N_ADD), "n_pairs": int(N_PAIR),
                          "basis_kind": BASIS_KIND})
res.setdefault("corners", {})

for ci, corner in enumerate(CORNERS):
    ck = f"c{ci}"
    row = res["corners"].get(ck, {})
    if row.get("done"):
        ctx.log(f"corner {ci} already done")
        continue
    s = dev.dev_setup(c, corner)
    X_store = c["gather"](s["pool"])
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    Zs = np.asarray(X_store, np.float32).reshape((-1, D)).astype(np.float64)
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    row["corner"] = corner

    from chlu.core.gen_request import FactorPrototypes, requests_from_rows
    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    kp = dev.score_features(
        c, np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                      np.float32), lab_dev)
    kp["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    row["knn_proto"] = kp
    ctx.log(f"corner {ci}: knn_proto={kp['aggregate']:.4f}({kp['distinct_factor_tuples']}/72)")

    A, B = design(lab_store)
    Ad, Bd = design(lab_dev)
    Phi = np.concatenate([A, B], axis=1)
    Phid = np.concatenate([Ad, Bd], axis=1)
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
            out = np.stack([Zs[rng.choice(Zs.shape[0], M, replace=False)][:, perm[k]]
                            for k in range(K)]) if False else None
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

    for (K, M, r, ridge, space) in CONFIGS:
        tag = f"K{K}_M{M}_r{r}_ridge{ridge:g}_{space}"
        if tag in row:
            continue
        labels = fast.balanced_labels(ow, dead, K, D)
        perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
        head_b = int((N_ADD * K * M + N_PAIR * r + r * K * M) * 4)
        store_b = int(M * D * 4)
        entry = {"head_bytes": head_b, "store_bytes": store_b,
                 "total_bytes": head_b + store_b,
                 "under_knn_bytes": bool(head_b + store_b < 475_136),
                 "span": int(K * M), "n_add": int(N_ADD), "n_pair": int(N_PAIR)}
        for kind in KINDS:
            bas = basis_of(kind, perm, M, labels)
            if space == "feature":
                W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ Zs)
                C = np.zeros((Phi.shape[1], K, M))
                for k in range(K):
                    Bk = bas[k]
                    C[:, k, :] = np.linalg.solve(Bk @ Bk.T + 1e-8 * np.eye(M),
                                                 Bk @ W[:, perm[k]].T).T
            else:                      # fit directly in coefficient space
                C = np.zeros((Phi.shape[1], K, M))
                G = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
                for k in range(K):
                    Bk = bas[k]
                    Tk = np.linalg.solve(Bk @ Bk.T + 1e-8 * np.eye(M),
                                         Bk @ Zs[:, perm[k]].T).T      # (n, M) targets
                    C[:, k, :] = np.linalg.solve(G, Phi.T @ Tk)
            Cb = C[N_ADD:].reshape(N_PAIR, K * M)
            U_, S_, Vt_ = np.linalg.svd(Cb, full_matrices=False)
            rr = min(r, S_.size)
            C[N_ADD:] = ((U_[:, :rr] * S_[:rr]) @ Vt_[:rr]).reshape(N_PAIR, K, M)
            a = (Phid @ C.reshape(C.shape[0], K * M)).reshape(-1, K, M)
            out = np.zeros((Phid.shape[0], D))
            for k in range(K):
                out[:, perm[k]] = a[:, k, :] @ bas[k]
            entry[kind] = dev.score_features(c, out.astype(np.float32), lab_dev)
        row[tag] = entry
        ctx.log(f"  [c{ci} {tag}] " + "  ".join(
            f"{k}={entry[k]['aggregate']:.4f}({entry[k]['distinct_factor_tuples']})"
            for k in KINDS) + f"  span={entry['span']} total={entry['total_bytes']}"
            + (" ⭐under" if entry["under_knn_bytes"] else " ⚠over"))
        res["corners"][ck] = row
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    row["done"] = True
    res["corners"][ck] = row
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

SEL, VAL = ["c0", "c1", "c2"], ["c3", "c4", "c5"]
summary = {}
for (K, M, r, ridge, space) in CONFIGS:
    tag = f"K{K}_M{M}_r{r}_ridge{ridge:g}_{space}"
    if not all(tag in res["corners"].get(k, {}) for k in SEL + VAL):
        continue
    g = lambda ks, kind: [res["corners"][k][tag][kind]["aggregate"] for k in ks]
    kn = lambda ks: [res["corners"][k]["knn_proto"]["aggregate"] for k in ks]
    tup = [res["corners"][k][tag]["modes"]["distinct_factor_tuples"] for k in VAL]
    summary[tag] = {
        "sel_modes": float(np.mean(g(SEL, "modes"))), "sel_knn": float(np.mean(kn(SEL))),
        "val_modes": float(np.mean(g(VAL, "modes"))),
        "val_modes_sd": float(np.std(g(VAL, "modes"))),
        "val_atoms": float(np.mean(g(VAL, "atoms"))),
        "val_pca": float(np.mean(g(VAL, "pca"))),
        "val_random": float(np.mean(g(VAL, "random"))),
        "val_mean": float(np.mean(g(VAL, "mean"))),
        "val_knn": float(np.mean(kn(VAL))),
        "val_tuples": float(np.mean(tup)),
        "delta_vs_knn": float(np.mean(g(VAL, "modes")) - np.mean(kn(VAL))),
        "wins_vs_knn": int(sum(1 for a, b in zip(g(VAL, "modes"), kn(VAL)) if a > b)),
        "delta_vs_atoms": float(np.mean(g(VAL, "modes")) - np.mean(g(VAL, "atoms"))),
        "total_bytes": res["corners"][VAL[0]][tag]["total_bytes"],
    }
    ss = summary[tag]
    ctx.log(f"[SUMMARY {tag}] SEL modes={ss['sel_modes']:.4f} knn={ss['sel_knn']:.4f} | "
            f"HELD-OUT modes={ss['val_modes']:.4f}±{ss['val_modes_sd']:.4f}"
            f"({ss['val_tuples']:.1f}) atoms={ss['val_atoms']:.4f} pca={ss['val_pca']:.4f} "
            f"random={ss['val_random']:.4f} mean={ss['val_mean']:.4f} knn={ss['val_knn']:.4f} "
            f"| Dknn={ss['delta_vs_knn']:+.4f} ({ss['wins_vs_knn']}/3) "
            f"Datoms={ss['delta_vs_atoms']:+.4f} total={ss['total_bytes']}")
res["summary"] = summary
res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
