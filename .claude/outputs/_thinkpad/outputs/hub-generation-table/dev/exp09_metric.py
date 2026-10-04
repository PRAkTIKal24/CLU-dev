"""DEV-09 — ⭐⭐ quantise under the FULL per-well decoder metric.

DEV-07 measured that the quantiser's residual costs 0.14–0.43 of accuracy while
isotropic noise of the SAME energy costs ~0.03.  ⇒ the residual is low-dimensional and
aligned with the directions the decoder reads.  DEV-04 showed a *diagonal* decoder weight
does nothing, which is consistent: those directions are not coordinate-aligned.

Here the assignment metric is the **full** per-well block ``A_k = (JᵀJ)|_{well k}``,
estimated from Hutchinson probes ``g = Jᵀu`` (``cov(g) = JᵀJ``).
⭐ The Lloyd update is still the plain mean, so **nothing extra is stored** and
``store_bytes = M·D·4 = 524,288 B`` is unchanged.  ⛔ The metric is a FITTING quantity and
is never deployed: at read time the codes come from the request.

Rows per K: the oracle-code reference under each metric, the naive-Bayes-addressed arm,
the discriminatively-addressed arm, and the laundering control.

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
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16", "32", "64"])]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp09_seed{SEED}.json")
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

gpath = os.path.join(CACHE, f"jac_samples_s{SEED}.npy")
if os.path.exists(gpath):
    G = np.load(gpath)
    ctx.log(f"jacobian samples loaded {G.shape}")
else:
    with fast.timer("jacobian_samples", ctx.log):
        G = fast.decoder_jacobian_samples(c["decoder"], Zs[:512], c["feat_shape"],
                                          n_probe=16, batch=32, seed=SEED)
    np.save(gpath, G)
    ctx.log(f"jacobian samples {G.shape} cached")

from chlu.core.gen_address import as_address_book, fit_discriminative_address_book  # noqa: E402
from chlu.core.gen_compose import (compose_codes, read_compose, request_posterior,  # noqa: E402
                                   store_removed_read)
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_store import read_from_codes, store_bytes  # noqa: E402

req = requests_from_labels(lab_dev, "full", 0, SEED)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("K", {})

for K in KS:
    res["K"].setdefault(str(K), {})
    labels = fast.balanced_labels(order_ward, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    A = fast.well_metrics(G, perm, ridge=1e-2)
    ev = np.linalg.eigvalsh(A[0])
    ctx.log(f"K={K} metric well-0 eigenvalue spread: max/min = "
            f"{float(ev[-1]/max(ev[0],1e-30)):.1f}  "
            f"top-4 share {float(ev[-4:].sum()/ev.sum()):.3f} of {ev.size}")
    for tag, metric in (("euclidean", None), ("decoder_metric", A)):
        if tag in res["K"][str(K)]:
            ctx.log(f"K={K} {tag} already in ledger — skipped")
            continue
        t = time.time()
        pq = fast.kmeans_pq_metric(Zs, labels, K, 64, SEED, metric, iters=25)
        store = pq.to_gen_store()
        sb = store_bytes(store)
        cdv = fast.metric_codes(pq, Zd, metric)
        ct = fast.metric_codes(pq, Zs, metric)
        rec = np.asarray(read_from_codes(store, cdv), np.float32).reshape((-1, D))
        row = {"bytes": int(sb["store_bytes"]),
               "bits_per_read": float(sb["bits_per_read"]),
               "recon_mse_dev": float(np.mean((rec - Zd) ** 2)),
               "fit_seconds": round(time.time() - t, 1)}
        row["nearest_mode_reference"] = dev.score_features(c, rec, lab_dev)
        nb = fit_address_book(ct, lab_store, c["cards"], 1.0)
        cd_nb = compose_codes(request_posterior(nb, req))
        row["clu_nb"] = dev.score_features(
            c, np.asarray(read_compose(store, cd_nb["code"]), np.float32), lab_dev)
        dbk = fit_discriminative_address_book(ct, lab_store, c["cards"], l2=1e-3,
                                              steps=150, lr=0.5)
        cd_d = compose_codes(request_posterior(as_address_book(dbk), req))
        row["clu_disc"] = dev.score_features(
            c, np.asarray(read_compose(store, cd_d["code"]), np.float32), lab_dev)
        row["store_removed"] = dev.score_features(
            c, np.asarray(store_removed_read(store, cd_nb["code"], "mean", SEED),
                          np.float32), lab_dev)
        res["K"][str(K)][tag] = row
        ctx.log(f"  [K={K} {tag}] mse={row['recon_mse_dev']:.4f} "
                f"ref={row['nearest_mode_reference']['aggregate']:.4f}"
                f"({row['nearest_mode_reference']['distinct_factor_tuples']}) "
                f"nb={row['clu_nb']['aggregate']:.4f}({row['clu_nb']['distinct_factor_tuples']}) "
                f"disc={row['clu_disc']['aggregate']:.4f}({row['clu_disc']['distinct_factor_tuples']}) "
                f"sr={row['store_removed']['aggregate']:.4f} [{row['fit_seconds']}s]")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
