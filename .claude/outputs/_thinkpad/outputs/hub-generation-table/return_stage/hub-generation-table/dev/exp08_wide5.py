"""DEV-08 — the SHIPPED cell (K=5, ward grouping, ``train_store``) read three ways.

⭐ This is the one configuration whose byte ledger is **identical to the shipped `clu`
arm in every column**: store 524,288 B, addressing head 74,240 B.  Nothing here changes
K, M, the partition or the trainer.  Only the READ verb changes:

  argmax     ``read_from_codes(store, argmax_m posterior)``   — the shipped 30-bit read
  wide       ``clu_wide_vector(store, posterior)``            — ⭐ the shipped R27 arm:
             ``x_k = Σ_m π_{k,m} mode_{k,m}``.  ⚠ A WIDE channel (K·M = 320 reals, vs
             `knn_proto`'s 2048) — declared, ⛔ never quoted in the 30-bit column.
  wide_disc  the same, with the discriminatively-fitted head (⭐ same 74,240 B)

plus three references on the same corner:
  simplex_ceiling   the best convex combination of each well's modes, fitted to the TRUE
                    dev feature (⛔ DIAGNOSTIC — the capacity of the WIDE read)
  span_ceiling      the least-squares projection onto each well's mode span (⛔ likewise)
  argmax reference  nearest mode from the true feature

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
                    f"exp08_seed{SEED}.json")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
Zd = np.asarray(X_dev, np.float32).reshape((-1, D))

from chlu.core.gen_address import as_address_book, fit_discriminative_address_book  # noqa: E402
from chlu.core.gen_compose import (address_book_bytes, clu_wide_vector, compose_codes,  # noqa: E402
                                   deployed_byte_ledger, read_compose, request_posterior,
                                   store_removed_read)
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_store import address_codes, delete_store, read_from_codes  # noqa: E402

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("rows", {})

# ⭐ THE SHIPPED STORE: ward grouping at the plan's K, the shipped Adam trainer.
t = time.time()
store, sinfo = dev.build_store(X_store, c["feat_shape"], 5, 64, SEED, c["cfg"])
ctx.log(f"shipped store built in {time.time()-t:.1f}s wells={sinfo['well_sizes']} "
        f"bytes={sinfo['store_bytes']['store_bytes']} bits={sinfo['store_bytes']['bits_per_read']}")

req = requests_from_labels(lab_dev, "full", 0, SEED)
ct = address_codes(store, Zs)
nb = fit_address_book(ct, lab_store, c["cards"], 1.0)
led = deployed_byte_ledger(store, nb)
res["byte_ledger"] = {"store_bytes": int(led["deployed_bytes"]),
                      "addressing_head_bytes": int(led["addressing_head"]["bytes"]),
                      "total_bytes": int(led["deployed_bytes"]
                                         + led["addressing_head"]["bytes"]),
                      "bits_per_read_argmax": float(led["bits_per_read"])}
ctx.log("byte ledger: " + json.dumps(res["byte_ledger"]))

post = request_posterior(nb, req)
cd = compose_codes(post)


def rec(name, feats, extra=None):
    sc = dev.score_features(c, np.asarray(feats, np.float32), lab_dev)
    if extra:
        sc.update(extra)
    res["rows"][name] = sc
    ctx.log(f"  [{name}] agg={sc['aggregate']:.4f} ({sc['distinct_factor_tuples']}/72)"
            + (f"  {extra}" if extra else ""))
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return sc


rec("argmax", read_compose(store, cd["code"]),
    {"channel": "K*log2(M) = 30 bits", "head_bytes": int(led["addressing_head"]["bytes"])})
rec("wide", clu_wide_vector(store, post),
    {"channel": "K*M = 320 reals (WIDE)", "head_bytes": int(led["addressing_head"]["bytes"])})
rec("store_removed_argmax",
    store_removed_read(store, cd["code"], "mean", SEED))
sr_store = delete_store(store, "mean", SEED)
rec("store_removed_wide", clu_wide_vector(sr_store, post))

dbk = fit_discriminative_address_book(ct, lab_store, c["cards"], l2=1e-3, steps=300, lr=0.5)
post_d = request_posterior(as_address_book(dbk), req)
rec("argmax_disc", read_compose(store, compose_codes(post_d)["code"]),
    {"head_bytes": int(address_book_bytes(as_address_book(dbk))["bytes"])})
rec("wide_disc", clu_wide_vector(store, post_d),
    {"head_bytes": int(address_book_bytes(as_address_book(dbk))["bytes"])})

# --- the two capacity references (⛔ DIAGNOSTICS: they see the true feature) ---
rec("ref_argmax", read_from_codes(store, address_codes(store, Zd)))

modes = [np.asarray(m, np.float64) for m in store.modes]
wc = [np.asarray(w, np.int64) for w in store.well_channels]
span = np.zeros_like(Zd, dtype=np.float64)
simp = np.zeros_like(Zd, dtype=np.float64)
for k, (w, m) in enumerate(zip(wc, modes)):
    Y = Zd[:, w].astype(np.float64)                       # (n, d_k)
    coef, *_ = np.linalg.lstsq(m.T, Y.T, rcond=None)      # (M, n)
    span[:, w] = (m.T @ coef).T
    # projection onto the SIMPLEX hull, by a few steps of exponentiated gradient
    a = np.full((Y.shape[0], m.shape[0]), 1.0 / m.shape[0])
    G = m @ m.T
    b = Y @ m.T
    for _ in range(400):
        grad = a @ G - b
        a = a * np.exp(-0.05 * grad / max(np.abs(grad).max(), 1e-9))
        a /= a.sum(axis=1, keepdims=True)
    simp[:, w] = a @ m
rec("ref_span_projection", span, {"what": "⛔ DIAGNOSTIC: least-squares onto each well's mode span"})
rec("ref_simplex_projection", simp, {"what": "⛔ DIAGNOSTIC: best convex combination of each well's modes"})

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
