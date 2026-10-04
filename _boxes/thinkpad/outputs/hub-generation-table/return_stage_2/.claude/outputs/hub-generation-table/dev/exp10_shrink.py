"""DEV-10 — ⭐ the SHRINKAGE hypothesis: k-means centroids are systematically washed out.

A centroid is an average, so ``Var(centroid) < Var(data)`` by construction — the read is a
*contracted* version of the feature distribution.  A decoder trained on real features then
sees a systematically low-contrast input, which is exactly the kind of **coherent,
low-dimensional** error DEV-07 measured to be expensive (0.43 of accuracy at K=8, against
0.03 for isotropic noise of the same energy).

The repair costs **nothing**: expand each mode about its well's mean,

    mode  <-  mean_k + alpha * (mode - mean_k)

⭐ The store still holds ``M`` float32 vectors per well, so ``store_bytes = M·D·4 =
524,288 B`` is untouched, the read is still the shipped ``read_from_codes(store, codes)``,
and ``assert_no_bypass`` is unchanged.  ⛔ ``alpha`` is selected on DEV, never on B.

Two forms: one scalar ``alpha`` for everything, and a per-COORDINATE ``alpha_d`` chosen to
match the read's variance to the train features' variance (folded into the modes, so still
zero extra bytes).

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
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16", "32", "64", "128"])]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp10_seed{SEED}.json")
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

from chlu.core.gen_compose import compose_codes, read_compose, request_posterior, store_removed_read  # noqa: E402
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_store import read_from_codes, store_bytes  # noqa: E402

req = requests_from_labels(lab_dev, "full", 0, SEED)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("K", {})

ALPHAS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]

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
    base_modes = pq.modes.copy()
    perm = pq.perm
    # the codes and the head are fitted ONCE, on the unmodified codebook
    ct = pq.codes(Zs)
    cdv = pq.codes(Zd)
    nb = fit_address_book(ct, lab_store, c["cards"], 1.0)
    cd = compose_codes(request_posterior(nb, req))

    # how contracted IS the read?
    store0 = pq.to_gen_store()
    rec0 = np.asarray(read_from_codes(store0, cdv), np.float32).reshape((-1, D))
    var_ratio = float(np.mean(rec0.var(axis=0)) / max(float(np.mean(Zd.var(axis=0))), 1e-30))
    ctx.log(f"K={K}: Var(read)/Var(features) on DEV = {var_ratio:.4f}")
    res["K"][str(K)]["variance_ratio_read_over_features"] = var_ratio

    wmean = base_modes.mean(axis=1, keepdims=True)                      # (K, 1, d)
    for a in ALPHAS:
        tag = f"alpha={a}"
        if tag in res["K"][str(K)]:
            continue
        pq.modes = (wmean + a * (base_modes - wmean)).astype(np.float32)
        st = pq.to_gen_store()
        ref = dev.score_features(
            c, np.asarray(read_from_codes(st, cdv), np.float32), lab_dev)
        arm = dev.score_features(
            c, np.asarray(read_compose(st, cd["code"]), np.float32), lab_dev)
        res["K"][str(K)][tag] = {"nearest_mode_reference": ref, "clu_nb": arm,
                                 "store_bytes": int(store_bytes(st)["store_bytes"])}
        ctx.log(f"  [K={K} alpha={a}] ref={ref['aggregate']:.4f}"
                f"({ref['distinct_factor_tuples']}) arm={arm['aggregate']:.4f}"
                f"({arm['distinct_factor_tuples']}) bytes={res['K'][str(K)][tag]['store_bytes']}")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

    # per-coordinate variance matching, folded into the modes (still zero extra bytes)
    if "per_coord" not in res["K"][str(K)]:
        pq.modes = base_modes.copy()
        st = pq.to_gen_store()
        rtr = np.asarray(read_from_codes(st, ct), np.float32).reshape((-1, D))
        g = np.sqrt(np.maximum(Zs.var(axis=0), 1e-12) / np.maximum(rtr.var(axis=0), 1e-12))
        g = np.clip(g, 0.5, 4.0)
        gk = g[perm]                                                    # (K, d)
        newm = wmean + (base_modes - wmean) * gk[:, None, :]
        pq.modes = newm.astype(np.float32)
        st = pq.to_gen_store()
        ref = dev.score_features(
            c, np.asarray(read_from_codes(st, cdv), np.float32), lab_dev)
        arm = dev.score_features(
            c, np.asarray(read_compose(st, cd["code"]), np.float32), lab_dev)
        sr = dev.score_features(
            c, np.asarray(store_removed_read(st, cd["code"], "mean", SEED), np.float32),
            lab_dev)
        res["K"][str(K)]["per_coord"] = {
            "nearest_mode_reference": ref, "clu_nb": arm, "store_removed": sr,
            "store_bytes": int(store_bytes(st)["store_bytes"]),
            "gain_min": float(g.min()), "gain_max": float(g.max()),
            "gain_mean": float(g.mean())}
        ctx.log(f"  [K={K} per_coord g in [{g.min():.2f},{g.max():.2f}] mean {g.mean():.3f}] "
                f"ref={ref['aggregate']:.4f}({ref['distinct_factor_tuples']}) "
                f"arm={arm['aggregate']:.4f}({arm['distinct_factor_tuples']}) "
                f"sr={sr['aggregate']:.4f}({sr['distinct_factor_tuples']})")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    pq.modes = base_modes

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
