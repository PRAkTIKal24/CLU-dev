"""DEV-05 — (a) the OPPONENTS on the DEV corner, (b) how much rank the head needs.

(a) ``knn_proto`` — R8's per-factor prototype embedding, decoded directly. The ceiling
    arm. ⛔ Without it on the SAME corner the K-sweep is a curve with no opponent.
    Plus a matched-bytes **atom** store (64 real train rows per well) read two ways:
    posterior-ARGMAX (the addressed read) and posterior-MEAN (⭐ the §C.1 lookup floor).

(b) The fitted head's weight tensor ``W (ΣV_f × K·M)`` is truncated by SVD to rank r and
    re-scored. ⭐ That answers, without refitting, whether ``K=32`` survives a head small
    enough to be byte-identical to the shipped 74,240 B one.

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
K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp05_seed{SEED}_K{K}.json")

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

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("K", K)
res.setdefault("corner", s["corner"])

from chlu.core.gen_address import as_address_book, fit_discriminative_address_book  # noqa: E402
from chlu.core.gen_compose import (address_book_bytes, compose_codes, read_compose,  # noqa: E402
                                   request_posterior, store_removed_read)
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.core.gen_store import read_from_codes, store_bytes  # noqa: E402

req = requests_from_labels(lab_dev, "full", 0, SEED)

# ---------------------------------------------------------------------------
# (a) the opponents
# ---------------------------------------------------------------------------
if "rivals" not in res:
    riv = {}
    t = time.time()
    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    led = protos.byte_ledger()
    rd = requests_from_rows(c["grid"], s["dev_rows"])
    emb = np.asarray(protos.embed_many(rd), np.float32)
    riv["knn_proto"] = dev.score_features(c, emb, lab_dev)
    riv["knn_proto"]["deployed_bytes"] = int(led["bytes"])
    riv["knn_proto"]["bytes_formula"] = led["bytes_formula"]
    ctx.log(f"  [knn_proto] agg={riv['knn_proto']['aggregate']:.4f} "
            f"({riv['knn_proto']['distinct_factor_tuples']}/72) bytes={led['bytes']} "
            f"[{time.time()-t:.1f}s]")

    # matched-bytes ATOM store: 64 real train rows, the SAME row index in every well
    rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916)
    labels = fast.balanced_labels(order_ward, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    atom_rows = rng.choice(Zs.shape[0], 64, replace=False)
    atoms = np.stack([Zs[atom_rows][:, perm[k]] for k in range(K)])   # (K, 64, d)
    apq = fast.EqualPQ(perm, atoms, (D,))
    astore = apq.to_gen_store()
    ct_a = apq.codes(Zs)
    abook = fit_address_book(ct_a, lab_store, c["cards"], 1.0)
    post_a = request_posterior(abook, req)                            # (n, K, 64)
    cd_a = compose_codes(post_a)
    riv["atom_argmax"] = dev.score_features(
        c, np.asarray(read_from_codes(astore, cd_a["code"]), np.float32), lab_dev)
    # ⭐ the §C.1 LOOKUP FLOOR: the posterior-weighted MEAN of the same atoms,
    # at exactly the same bytes
    mean_feats = np.zeros((req.shape[0], D), np.float32)
    for k in range(K):
        mean_feats[:, perm[k]] = post_a[:, k, :] @ atoms[k]
    riv["atom_mean_lookup"] = dev.score_features(c, mean_feats, lab_dev)
    for nm in ("atom_argmax", "atom_mean_lookup"):
        riv[nm]["deployed_bytes"] = int(store_bytes(astore)["store_bytes"])
        ctx.log(f"  [{nm}] agg={riv[nm]['aggregate']:.4f} "
                f"({riv[nm]['distinct_factor_tuples']}/72) bytes={riv[nm]['deployed_bytes']}")
    res["rivals"] = riv
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

# ---------------------------------------------------------------------------
# (b) the head's rank
# ---------------------------------------------------------------------------
if "rank" not in res:
    labels = fast.balanced_labels(order_ward, dead, K, D)
    with fast.timer("kmeans_pq", ctx.log):
        pq = fast.kmeans_pq(Zs, labels, K, 64, SEED, iters=25)
    store = pq.to_gen_store()
    sb = store_bytes(store)
    ct = fast.weighted_codes(pq, Zs, None)
    nb = fit_address_book(ct, lab_store, c["cards"], 1.0)
    full_head_bytes = int(address_book_bytes(nb)["bytes"])
    cd_nb = compose_codes(request_posterior(nb, req))
    base = dev.score_features(c, np.asarray(read_compose(store, cd_nb["code"]), np.float32),
                              lab_dev)
    ctx.log(f"  [K={K} full naive-Bayes head] agg={base['aggregate']:.4f} "
            f"({base['distinct_factor_tuples']}/72) head={full_head_bytes} B")

    cards = c["cards"]
    offs = np.cumsum([0] + cards[:-1])
    V = int(sum(cards))
    M = 64
    b0 = np.asarray(nb.log_prior, np.float64)                        # (K, M)
    Wfull = np.concatenate([np.asarray(lc, np.float64) - b0[None] for lc in nb.log_cond],
                           axis=0).reshape(V, K * M)                 # (V, K*M)
    U, S, Vt = np.linalg.svd(Wfull, full_matrices=False)
    ctx.log("  head singular values (top 12): " + ", ".join(f"{x:.3f}" for x in S[:12]))
    ctx.log(f"  spectral mass in top 8 of {S.size}: "
            f"{float((S[:8]**2).sum()/ (S**2).sum()):.4f}")

    rank_rows = {"full": {**base, "head_bytes": full_head_bytes,
                          "store_bytes": int(sb["store_bytes"])},
                 "singular_values": [float(x) for x in S],
                 "budget_bytes": 74240}
    from chlu.core.gen_dream import AddressBook
    for r in (2, 4, 7, 8, 12, 16, 24, 32, 48):
        if r > S.size:
            continue
        Wr = (U[:, :r] * S[:r]) @ Vt[:r]
        head_bytes = int((V * r + K * M * r + K * M) * 4)
        lc = [np.asarray(Wr.reshape(V, K, M)[offs[f]: offs[f] + cards[f]] + b0[None],
                         np.float32) for f in range(len(cards))]
        bk = AddressBook(np.asarray(b0, np.float32), lc)
        cd = compose_codes(request_posterior(bk, req))
        sc = dev.score_features(c, np.asarray(read_compose(store, cd["code"]), np.float32),
                                lab_dev)
        sc["head_bytes"] = head_bytes
        sc["fits_74240"] = bool(head_bytes <= 74240)
        sc["total_bytes"] = int(sb["store_bytes"]) + head_bytes
        rank_rows[str(r)] = sc
        ctx.log(f"  [rank {r:3d}] agg={sc['aggregate']:.4f} ({sc['distinct_factor_tuples']}/72) "
                f"head={head_bytes} B  fits_74240={sc['fits_74240']}  total={sc['total_bytes']}")
    res["rank"] = rank_rows
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
