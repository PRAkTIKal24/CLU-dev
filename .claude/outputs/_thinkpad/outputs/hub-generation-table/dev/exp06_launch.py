"""DEV-06 — ⭐⭐ THE LAUNCH READ: request -> latent address -> nearest mode per well -> read.

The head is a rank-``r`` ADDITIVE request->feature map (the only place labels enter):

    x̂(r) = mu + sum_f  P[offset_f + v_f]          P ≈ A B,  A (V x r), B (r x D)
    head_bytes = (V*r + r*D + D) * 4              ⭐ INDEPENDENT of K

and the read is the shipped ``gen_store.read_from_codes(store, codes)`` with
``codes[k] = argmin_m ||x̂_well_k - mode_{k,m}||²``.  ⛔ No feature tensor enters: x̂ is a
function of the request and the head alone.

⭐⭐ The §C.1 test is exact here, because **the head at full rank IS ``knn_proto``**:

    head_alone   = decode(x̂)                     ⭐ the matched-head LOOKUP FLOOR
    arm          = decode(read_from_codes(store, nearest_mode(x̂)))
    store_removed= decode(read_from_codes(delete_store(store), same codes))

⇒ ``arm > head_alone`` is the ONLY way the 524,288 B store earns its bytes on this path.
⛔ If it does not, the store is decorative on this path and is reported as such.

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
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                       ["16", "32", "64", "128", "256"])]
RS = [x for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["full", "16", "7"])]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp06_seed{SEED}.json")

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

cards = c["cards"]
V = int(sum(cards))
offs = np.cumsum([0] + cards[:-1]).astype(np.int64)

# --- the additive prototype head, fitted on the pool (labels stop HERE) -----
mu = Zs.mean(axis=0)
P = np.zeros((V, D), np.float64)
for f, cf in enumerate(cards):
    for v in range(cf):
        m = lab_store[:, f] == v
        P[offs[f] + v] = (Zs[m].mean(axis=0) - mu) if m.sum() else 0.0
Uh, Sh, Vth = np.linalg.svd(P, full_matrices=False)
ctx.log("prototype-deviation singular values (top 12): "
        + ", ".join(f"{x:.2f}" for x in Sh[:12]))
for rr in (4, 7, 8, 16, 24, 32):
    ctx.log(f"  spectral mass in top {rr:2d} of {Sh.size}: "
            f"{float((Sh[:rr]**2).sum()/(Sh**2).sum()):.4f}")

req_lab = lab_dev
cols = (req_lab + offs[None, :]).astype(np.int64)


def xhat(rank):
    Pr = P if rank is None else (Uh[:, :rank] * Sh[:rank]) @ Vth[:rank]
    out = np.repeat(mu[None].astype(np.float64), req_lab.shape[0], axis=0)
    for f in range(len(cards)):
        out += Pr[cols[:, f]]
    return out.astype(np.float32)


def head_bytes(rank):
    if rank is None:
        return int((V * D + D) * 4)          # ⭐ == knn_proto's own ledger
    return int((V * rank + rank * D + D) * 4)


res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("prototype_singular_values", [float(x) for x in Sh])
res.setdefault("head_alone", {})
res.setdefault("K", {})

from chlu.core.gen_store import delete_store, read_from_codes, store_bytes  # noqa: E402

# --- ⭐ the §C.1 LOOKUP FLOOR at every rank: the head alone, no store -------
for rs in RS:
    rank = None if rs == "full" else int(rs)
    if rs in res["head_alone"]:
        continue
    xh = xhat(rank)
    sc = dev.score_features(c, xh, lab_dev)
    sc["head_bytes"] = head_bytes(rank)
    sc["is_knn_proto"] = bool(rank is None)
    res["head_alone"][rs] = sc
    ctx.log(f"  [head_alone r={rs}] agg={sc['aggregate']:.4f} "
            f"({sc['distinct_factor_tuples']}/72) head={sc['head_bytes']} B"
            + ("   ⭐ == knn_proto" if rank is None else ""))
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

for K in KS:
    res["K"].setdefault(str(K), {})
    labels = fast.balanced_labels(order_ward, dead, K, D)
    tag = f"K{K}"
    if "_store" not in res["K"][str(K)]:
        t = time.time()
        pq = fast.kmeans_pq(Zs, labels, K, 64, SEED, iters=25)
        np.savez_compressed(os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1),
                                         "dev", "cache", f"pq_s{SEED}_K{K}.npz"),
                            perm=pq.perm, modes=pq.modes)
        ctx.log(f"  [{tag}] codebook fitted in {time.time()-t:.1f}s")
    else:
        z = np.load(os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1),
                                 "dev", "cache", f"pq_s{SEED}_K{K}.npz"))
        pq = fast.EqualPQ(z["perm"], z["modes"], (D,))
    store = pq.to_gen_store()
    sb = store_bytes(store)
    res["K"][str(K)]["_store"] = {"bytes": int(sb["store_bytes"]),
                                  "bits_per_read": float(sb["bits_per_read"])}
    # the distortion-optimal reference, once per K
    if "_nearest_mode_reference" not in res["K"][str(K)]:
        rec = np.asarray(read_from_codes(store, pq.codes(Zd)), np.float32)
        res["K"][str(K)]["_nearest_mode_reference"] = dev.score_features(c, rec, lab_dev)
        ctx.log(f"  [{tag}] nearest_mode_reference="
                f"{res['K'][str(K)]['_nearest_mode_reference']['aggregate']:.4f}"
                f"({res['K'][str(K)]['_nearest_mode_reference']['distinct_factor_tuples']}/72)")
    for rs in RS:
        if rs in res["K"][str(K)]:
            continue
        rank = None if rs == "full" else int(rs)
        xh = xhat(rank)
        codes = pq.codes(xh)
        feats = np.asarray(read_from_codes(store, codes), np.float32)
        sc = dev.score_features(c, feats, lab_dev)
        sr = dev.score_features(
            c, np.asarray(read_from_codes(delete_store(store, "mean", SEED), codes),
                          np.float32), lab_dev)
        floor = res["head_alone"][rs]["aggregate"]
        sc["head_bytes"] = head_bytes(rank)
        sc["store_bytes"] = int(sb["store_bytes"])
        sc["total_bytes"] = sc["head_bytes"] + sc["store_bytes"]
        sc["n_distinct_codes"] = int(np.unique(codes, axis=0).shape[0])
        sc["store_removed"] = sr
        sc["delta_vs_head_alone"] = float(sc["aggregate"] - floor)
        sc["store_earns_its_bytes"] = bool(sc["aggregate"] > floor)
        res["K"][str(K)][rs] = sc
        ctx.log(f"  [{tag} r={rs}] agg={sc['aggregate']:.4f}({sc['distinct_factor_tuples']}/72) "
                f"vs head_alone {floor:.4f} -> Δ={sc['delta_vs_head_alone']:+.4f} "
                f"{'⭐ STORE EARNS' if sc['store_earns_its_bytes'] else '⛔ store loses'} "
                f"| sr={sr['aggregate']:.4f}({sr['distinct_factor_tuples']}) "
                f"| head={sc['head_bytes']} total={sc['total_bytes']}")
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
