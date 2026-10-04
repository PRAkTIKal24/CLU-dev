"""DEV-11 — ⭐⭐⭐ THE STORE AS A DICTIONARY: a request-addressed LINEAR read over the modes.

DEV-08 measured, on the SHIPPED K=5 / M=64 / 524,288 B store:

    nearest mode  (vertices)        0.4884 (11/72)
    convex hull   (softmax/simplex) 0.5509 (12/72)
    ⭐ mode SPAN  (linear)          0.8264 (42/72)     vs knn_proto 0.7801 (53/72)

⇒ the 64 modes of a well SPAN a subspace that carries the corner; ⛔ the defect is that
the shipped read is restricted to a **vertex** of that span.  The shipped
``gen_compose.clu_wide_vector(store, weights)`` already computes
``x_k = Σ_m w_{k,m}·mode_{k,m}`` as a plain matmul with **no simplex constraint**, so the
linear read needs no new read function.

The head is an ADDITIVE coefficient table of the shipped head's exact shape
``(1 + Σ_f V_f) × K × M`` — at K=5, M=64 that is **74,240 B, the shipped addressing head
to the byte**.  Fitting it by least squares is, for a block-diagonal basis, *identical* to
projecting the per-well prototype table onto each well's mode span, so this arm is
exactly **`knn_proto`, compressed into the store's own basis**:

    store_bytes = M·D·4          head_bytes = (1 + Σ_f V_f)·K·M·4
    total       = 4M(D + 58K)    ⭐ < knn_proto's 475,136 B for M ≤ 50 at K = 5

Rows per (K, M): the arm, the span capacity (⛔ diagnostic), the §C.1 ATOM-basis floor
(the same read with 64 real train rows as the basis instead of learned modes), and the
§C.2 laundering control.

⛔ DEV numbers.  eval_a / eval_b are never touched.
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
GRID = sys.argv[2] if len(sys.argv) > 2 else "8:16,8:32,8:47,16:16,16:32,16:39,32:16,32:30,8:64,16:64,32:64"
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp11_seed{SEED}.json")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D)).astype(np.float64)
Zd = np.asarray(X_dev, np.float32).reshape((-1, D)).astype(np.float64)
corr, dead, order_ward, order_avg = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)

cards = c["cards"]
V = int(sum(cards))
offs = np.cumsum([0] + cards[:-1]).astype(np.int64)
n = Zs.shape[0]
# ⭐ the design matrix: bias + one-hot per factor.  The ONLY place labels enter.
Phi = np.zeros((n, 1 + V), np.float64)
Phi[:, 0] = 1.0
Phi[np.repeat(np.arange(n), len(cards)),
    1 + (lab_store + offs[None, :]).reshape(-1)] = 1.0
Phid = np.zeros((lab_dev.shape[0], 1 + V), np.float64)
Phid[:, 0] = 1.0
Phid[np.repeat(np.arange(lab_dev.shape[0]), len(cards)),
     1 + (lab_dev + offs[None, :]).reshape(-1)] = 1.0
RIDGE = 1e-3
G = Phi.T @ Phi + RIDGE * np.eye(1 + V)
Gi = np.linalg.inv(G)
PhiTX = Phi.T @ Zs                      # (1+V, D)
Pfull = Gi @ PhiTX                      # ⭐ == knn_proto's table, up to the ridge

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("rows", {})

# the unconstrained additive model = knn_proto (ridge form), for reference
if "knn_proto_ridge" not in res["rows"]:
    sc = dev.score_features(c, (Phid @ Pfull).astype(np.float32), lab_dev)
    sc["head_bytes"] = int((1 + V) * D * 4)
    sc["store_bytes"] = 0
    sc["total_bytes"] = sc["head_bytes"]
    res["rows"]["knn_proto_ridge"] = sc
    ctx.log(f"  [knn_proto_ridge] agg={sc['aggregate']:.4f} "
            f"({sc['distinct_factor_tuples']}/72) bytes={sc['total_bytes']}")


def project_rows(P, perm, basis):
    """Project the prototype table onto each well's basis span (least squares)."""
    out = np.zeros_like(P)
    for k in range(perm.shape[0]):
        w = perm[k]
        Bk = basis[k]                                   # (M, d)
        gram = Bk @ Bk.T + 1e-8 * np.eye(Bk.shape[0])
        coef = np.linalg.solve(gram, Bk @ P[:, w].T)    # (M, 1+V)
        out[:, w] = (Bk.T @ coef).T
    return out


def span_capacity(X, perm, basis):
    out = np.zeros_like(X)
    for k in range(perm.shape[0]):
        w = perm[k]
        Bk = basis[k]
        gram = Bk @ Bk.T + 1e-8 * np.eye(Bk.shape[0])
        coef = np.linalg.solve(gram, Bk @ X[:, w].T)
        out[:, w] = (Bk.T @ coef).T
    return out


rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916)
for spec in GRID.split(","):
    K, M = (int(x) for x in spec.split(":"))
    tag = f"K{K}_M{M}"
    if tag in res["rows"]:
        ctx.log(f"{tag} already in ledger — skipped")
        continue
    t = time.time()
    labels = fast.balanced_labels(order_ward, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    pq = fast.kmeans_pq(Zs.astype(np.float32), labels, K, M, SEED, iters=25)
    modes = pq.modes.astype(np.float64)                          # (K, M, d)
    store_b = int(pq.modes.size * 4)
    head_b = int((1 + V) * K * M * 4)
    tot = store_b + head_b

    Pk = project_rows(Pfull, perm, modes)
    arm = dev.score_features(c, (Phid @ Pk).astype(np.float32), lab_dev)
    cap = dev.score_features(c, span_capacity(Zd, perm, modes).astype(np.float32), lab_dev)
    # §C.2 laundering control: the SAME coefficients, modes replaced by the well mean
    mm = np.repeat(modes.mean(axis=1, keepdims=True), M, axis=1)
    Psr = project_rows(Pfull, perm, mm)
    sr = dev.score_features(c, (Phid @ Psr).astype(np.float32), lab_dev)
    # §C.1 floor: the same read with REAL TRAIN ROWS as the basis, same bytes
    ar = rng.choice(Zs.shape[0], M, replace=False)
    atoms = np.stack([Zs[ar][:, perm[k]] for k in range(K)])
    Pa = project_rows(Pfull, perm, atoms)
    floor = dev.score_features(c, (Phid @ Pa).astype(np.float32), lab_dev)

    row = {"K": K, "M": M, "store_bytes": store_b, "head_bytes": head_b,
           "total_bytes": tot, "beats_knn_bytes": bool(tot < 475_136),
           "arm": arm, "span_capacity": cap, "store_removed": sr, "atom_basis_floor": floor,
           "seconds": round(time.time() - t, 1)}
    res["rows"][tag] = row
    ctx.log(f"  [{tag}] arm={arm['aggregate']:.4f}({arm['distinct_factor_tuples']}) "
            f"cap={cap['aggregate']:.4f}({cap['distinct_factor_tuples']}) "
            f"atomfloor={floor['aggregate']:.4f}({floor['distinct_factor_tuples']}) "
            f"sr={sr['aggregate']:.4f}({sr['distinct_factor_tuples']}) "
            f"| store={store_b} head={head_b} total={tot} "
            f"{'⭐ under knn bytes' if tot < 475136 else '⚠ over knn bytes'} [{row['seconds']}s]")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
