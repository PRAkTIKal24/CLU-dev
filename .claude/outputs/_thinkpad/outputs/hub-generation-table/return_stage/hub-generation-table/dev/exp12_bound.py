"""DEV-12 — ⭐⭐ IS `knn_proto` BEATABLE AT ALL by a request-only map?

Every arm on this table is some function of the request alone, fitted on the store pool.
`knn_proto` is the **best ADDITIVE** such function (μ + Σ_f prototype deviations).  Two
structural facts have now been measured:

* a LINEAR store read composed with an additive head is an additive model in a
  RESTRICTED family ⇒ it can only approach `knn_proto`, never exceed it (DEV-11);
* the non-linear (vertex) store read tops out near 0.70 (DEV-03/07/08).

⇒ the question that decides whether ANY arm can win is: **does the corner's feature have
structure an additive request model cannot represent?**  This measures it directly, with
no store involved, by widening the request→feature model's hypothesis class:

    additive        1 + Σ_f V_f                 = 58 columns      (⭐ == knn_proto)
    + pairwise      + Σ_{f<g} V_f·V_g           = 1380 columns
    + k-NN in factor space (nearest SEEN combination, and k-averaged)
    ⛔ oracle additive: the additive model REFITTED ON THE CORNER ITSELF — the
       estimation-vs-model-class split.  A cheat, reported as one, never an arm.

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

T0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp12_seed{SEED}.json")

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
F = len(cards)
n = Zs.shape[0]

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("rows", {})


def design(lab, pairs):
    N = lab.shape[0]
    cols = [np.ones((N, 1))]
    offs = np.cumsum([0] + cards[:-1])
    A = np.zeros((N, int(sum(cards))))
    A[np.repeat(np.arange(N), F), (lab + offs[None, :]).reshape(-1)] = 1.0
    cols.append(A)
    if pairs:
        for f in range(F):
            for g in range(f + 1, F):
                B = np.zeros((N, cards[f] * cards[g]))
                B[np.arange(N), lab[:, f] * cards[g] + lab[:, g]] = 1.0
                cols.append(B)
    return np.concatenate(cols, axis=1)


def fit_score(name, pairs, ridge, train_lab=None, train_X=None):
    if name in res["rows"]:
        ctx.log(f"{name} already in ledger — skipped")
        return
    t = time.time()
    L = lab_store if train_lab is None else train_lab
    X = Zs if train_X is None else train_X
    Phi = design(L, pairs)
    Phid = design(lab_dev, pairs)
    Gm = Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1])
    W = np.linalg.solve(Gm, Phi.T @ X)
    pred = (Phid @ W).astype(np.float32)
    sc = dev.score_features(c, pred, lab_dev)
    sc["n_columns"] = int(Phi.shape[1])
    sc["ridge"] = float(ridge)
    sc["bytes_if_deployed"] = int(Phi.shape[1] * D * 4)
    sc["seconds"] = round(time.time() - t, 1)
    res["rows"][name] = sc
    ctx.log(f"  [{name}] agg={sc['aggregate']:.4f} ({sc['distinct_factor_tuples']}/72) "
            f"cols={sc['n_columns']} ridge={ridge:g} bytes={sc['bytes_if_deployed']} "
            f"[{sc['seconds']}s]")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


fit_score("additive_ridge1e-3", False, 1e-3)
for r in (1.0, 10.0, 100.0, 1000.0):
    fit_score(f"pairwise_ridge{r:g}", True, r)

# --- k-NN in FACTOR space -------------------------------------------------
if "knn_factor" not in res["rows"]:
    vals = [np.asarray(c["grid"].values[k], float) for k in range(F)]
    spans = np.array([max(v[-1] - v[0], 1e-9) for v in vals])
    Ls = np.stack([vals[k][lab_store[:, k]] for k in range(F)], axis=1) / spans
    Ld = np.stack([vals[k][lab_dev[:, k]] for k in range(F)], axis=1) / spans
    d2 = ((Ld[:, None, :] - Ls[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d2, axis=1)
    for k in (1, 5, 20):
        pred = Zs[order[:, :k]].mean(axis=1).astype(np.float32)
        sc = dev.score_features(c, pred, lab_dev)
        sc["k"] = k
        res["rows"][f"knn_factor_k{k}"] = sc
        ctx.log(f"  [knn_factor k={k}] agg={sc['aggregate']:.4f} "
                f"({sc['distinct_factor_tuples']}/72)")
    res["rows"]["knn_factor"] = {"done": True}
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

# --- ⛔ THE CHEAT: the additive model refitted ON the corner ---------------
if "oracle_additive_on_corner" not in res["rows"]:
    Phid = design(lab_dev, False)
    Gm = Phid.T @ Phid + 1e-3 * np.eye(Phid.shape[1])
    W = np.linalg.solve(Gm, Phid.T @ Zd)
    sc = dev.score_features(c, (Phid @ W).astype(np.float32), lab_dev)
    sc["what"] = ("⛔ A CHEAT, reported as one: the additive model FITTED ON THE HELD-OUT "
                  "CORNER. It separates 'the additive class cannot express this corner' "
                  "from 'the additive class cannot be ESTIMATED from the pool'.")
    res["rows"]["oracle_additive_on_corner"] = sc
    ctx.log(f"  [oracle_additive_on_corner] agg={sc['aggregate']:.4f} "
            f"({sc['distinct_factor_tuples']}/72)  ⛔ cheat")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
