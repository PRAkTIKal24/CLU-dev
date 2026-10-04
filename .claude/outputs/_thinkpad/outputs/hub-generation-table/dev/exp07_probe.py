"""DEV-07 — three cheap probes that decide where the remaining loss lives.

(1) ⭐ THE NOISE CONTROL.  Decode ``x + ε``, ``ε ~ N(0, σ²)`` with σ² set to the
    quantiser's own MSE at each K.  If matched-magnitude ISOTROPIC noise costs about the
    same accuracy as quantisation, the codebook is at the information limit and a better
    codebook cannot help.  If isotropic noise costs far LESS, the quantisation error is
    STRUCTURED (off-manifold / decoder-adversarial) and a decoder-aware codebook can.

(2) ``clu_wide`` — the shipped ``gen_compose.clu_wide_vector``: the request posterior's
    weighted MEAN over modes rather than its argmax.  ⚠ A WIDER channel (K·M reals, not
    K integers) — declared, ⛔ never quoted in the same column as a 30-bit read.

(3) A HYBRID addresser: ``score = log P(m | request) + λ·(−‖x̂_k − m‖²)`` — the logit head
    and the launch distance, combined.  λ selected on DEV, ⛔ never on a held-out set.

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
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["32", "128"])]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp07_seed{SEED}.json")
CACHE = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev", "cache")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
Zd = np.asarray(X_dev, np.float32).reshape((-1, D))
corr, dead, order_ward, order_avg = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)

from chlu.core.gen_compose import (clu_wide_vector, compose_codes, read_compose,  # noqa: E402
                                   request_posterior, store_removed_read)
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_store import read_from_codes  # noqa: E402

req = requests_from_labels(lab_dev, "full", 0, SEED)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("noise_control", {})
res.setdefault("K", {})

# --- (1) the noise control -------------------------------------------------
mses = {"8": 0.0545, "16": 0.0517, "32": 0.0446, "64": 0.0381, "128": 0.0289}
rng = np.random.default_rng(int(SEED) * 13 + 7)
for k, mse in mses.items():
    if k in res["noise_control"]:
        continue
    sig = float(np.sqrt(mse))
    noisy = (Zd + rng.normal(0.0, sig, Zd.shape)).astype(np.float32)
    sc = dev.score_features(c, noisy, lab_dev)
    sc["sigma"] = sig
    sc["target_mse"] = mse
    res["noise_control"][k] = sc
    ctx.log(f"  [isotropic noise, mse={mse:.4f}] agg={sc['aggregate']:.4f} "
            f"({sc['distinct_factor_tuples']}/72)")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

# --- (2)+(3) --------------------------------------------------------------
cards = c["cards"]
V = int(sum(cards))
offs = np.cumsum([0] + cards[:-1]).astype(np.int64)
mu = Zs.mean(axis=0)
P = np.zeros((V, D), np.float64)
for f, cf in enumerate(cards):
    for v in range(cf):
        m = lab_store[:, f] == v
        P[offs[f] + v] = (Zs[m].mean(axis=0) - mu) if m.sum() else 0.0
cols = (lab_dev + offs[None, :]).astype(np.int64)
xh = np.repeat(mu[None].astype(np.float64), lab_dev.shape[0], axis=0)
for f in range(len(cards)):
    xh += P[cols[:, f]]
xh = xh.astype(np.float32)

for K in KS:
    res["K"].setdefault(str(K), {})
    pth = os.path.join(CACHE, f"pq_s{SEED}_K{K}.npz")
    if os.path.exists(pth):
        z = np.load(pth)
        pq = fast.EqualPQ(z["perm"], z["modes"], (D,))
    else:
        labels = fast.balanced_labels(order_ward, dead, K, D)
        pq = fast.kmeans_pq(Zs, labels, K, 64, SEED, iters=25)
        np.savez_compressed(pth, perm=pq.perm, modes=pq.modes)
    store = pq.to_gen_store()
    ct = pq.codes(Zs)
    nb = fit_address_book(ct, lab_store, cards, 1.0)
    post = request_posterior(nb, req)                     # (n, K, M)

    if "clu_wide" not in res["K"][str(K)]:
        wide = np.asarray(clu_wide_vector(store, post), np.float32)
        sc = dev.score_features(c, wide, lab_dev)
        sc["channel"] = "K*M reals (WIDE) — ⛔ not a 30-bit read"
        res["K"][str(K)]["clu_wide"] = sc
        ctx.log(f"  [K={K} clu_wide] agg={sc['aggregate']:.4f} "
                f"({sc['distinct_factor_tuples']}/72)")

    if "argmax" not in res["K"][str(K)]:
        cd = compose_codes(post)
        sc = dev.score_features(
            c, np.asarray(read_compose(store, cd["code"]), np.float32), lab_dev)
        res["K"][str(K)]["argmax"] = sc
        ctx.log(f"  [K={K} argmax] agg={sc['aggregate']:.4f} "
                f"({sc['distinct_factor_tuples']}/72)")

    # hybrid: logit head + launch distance
    if "hybrid" not in res["K"][str(K)]:
        lp = np.log(np.maximum(post, 1e-30))                       # (n, K, M)
        Vw = xh.reshape(-1, D)[:, pq.perm]                          # (n, K, d)
        mm = np.einsum("kmd,kmd->km", pq.modes, pq.modes)
        cross = np.einsum("bkd,kmd->bkm", Vw, pq.modes)
        d2 = mm[None] - 2.0 * cross                                 # (n, K, M) up to const
        d2 = d2 - d2.min(axis=-1, keepdims=True)
        hyb = {}
        for lam in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 1e6):
            code = np.argmax(lp - float(lam) * d2, axis=-1)
            sc = dev.score_features(
                c, np.asarray(read_from_codes(store, code), np.float32), lab_dev)
            hyb[str(lam)] = sc
            ctx.log(f"  [K={K} hybrid lam={lam:g}] agg={sc['aggregate']:.4f} "
                    f"({sc['distinct_factor_tuples']}/72)")
        res["K"][str(K)]["hybrid"] = hyb
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
