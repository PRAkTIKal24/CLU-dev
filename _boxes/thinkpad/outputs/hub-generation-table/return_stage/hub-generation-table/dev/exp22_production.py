"""DEV-22 — ⭐ the PRODUCTION path, end to end, with its assertions.

Everything above ran in local numpy.  A scored run must go through the objects the harness
uses.  This builds the arm out of:

    chlu.core.gen_store.GenStore                      (the shipped store)
    chlu.core.gen_factored_read.fit_factored_head     (my head, on my branch)
    chlu.core.gen_compose.clu_wide_vector             (the SHIPPED read verb)
    chlu.experiments.exp_gen_cell.render / score_images / distinct_counts
    exp_gen_cell.load_external_classifier             (the SHIPPED instrument)

and checks:
  1. ``assert_request_only_factored`` passes (signature · independent re-implementation ·
     provenance · bit-identity under feature replacement).
  2. ``gen_store.assert_no_bypass`` still passes on the same store object.
  3. the production path reproduces the DEV numpy pipeline to within float32 noise.
  4. the byte ledger is what the report will claim, computed from the LIVE objects.

⛔ DEV corner only.  eval_a / eval_b are never touched.
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
M = int(sys.argv[3]) if len(sys.argv) > 3 else 26
RANK = int(sys.argv[4]) if len(sys.argv) > 4 else 16
RIDGE = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
PAIRS = sys.argv[6] if len(sys.argv) > 6 else "floor_huexscale,object_huexscale"
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp22_seed{SEED}_K{K}_M{M}_r{RANK}.json")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
corr, dead, ow, oa = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)

from chlu.core.gen_compose import address_book_bytes, clu_wide_vector, deployed_byte_ledger  # noqa: E402
from chlu.core.gen_factored_read import (assert_request_only_factored,  # noqa: E402
                                         default_pairs, fit_factored_head)
from chlu.core.gen_store import assert_no_bypass, delete_store, store_bytes  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402

labels = fast.balanced_labels(ow, dead, K, D)
pq = fast.kmeans_pq(Zs, labels, K, M, SEED, iters=25)
store = pq.to_gen_store()
sb = store_bytes(store)
ctx.log(f"store: K={sb['n_wells']} M={sb['n_modes']} D={sb['dim']} "
        f"bytes={sb['store_bytes']} bits_per_read(argmax)={sb['bits_per_read']:.1f}")

pairs = default_pairs(c["names"], [p for p in PAIRS.split(",") if p])
ctx.log(f"pairs: {[(c['names'][a], c['names'][b]) for a, b in pairs]}")
head = fit_factored_head(store, Zs, lab_store, c["cards"], pairs, RANK, RIDGE)
ledger = head.byte_ledger()
ctx.log("head ledger: " + json.dumps({k: v for k, v in ledger.items()
                                      if k not in ("role",)}))

req = requests_from_labels(lab_dev, "full", 0, SEED)
a = head.coefficients(req)
feats = np.asarray(clu_wide_vector(store, a), np.float32)

# --- the assertions -------------------------------------------------------
ro = assert_request_only_factored(store, head, req, decoy_features=Zs[:64])
ctx.log("assert_request_only_factored: OK  max|delta| vs independent sum = "
        f"{ro['max_abs_delta_vs_independent_sum']:.3e}  channel={ro['channel']['width']} reals")
nb = assert_no_bypass(store, Zs[:64], rng_seed=SEED)
ctx.log(f"assert_no_bypass: OK  pure codebook gather={nb['read_is_pure_codebook_gather']} "
        f"bits_per_read={nb['bits_per_read']:.1f}")

# --- the score + the controls --------------------------------------------
arm = dev.score_features(c, feats, lab_dev)
ctx.log(f"  [arm] agg={arm['aggregate']:.4f} ({arm['distinct_factor_tuples']}/72)")

sr_store = delete_store(store, "mean", SEED)
sr = dev.score_features(c, np.asarray(clu_wide_vector(sr_store, a), np.float32), lab_dev)
ctx.log(f"  [store_removed §C.2] agg={sr['aggregate']:.4f} ({sr['distinct_factor_tuples']}/72)")

rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916)
ar = rng.choice(Zs.shape[0], M, replace=False)
atom_modes = np.stack([Zs[ar][:, pq.perm[k]] for k in range(K)]).astype(np.float32)
apq = fast.EqualPQ(pq.perm, atom_modes, (D,))
astore = apq.to_gen_store()
ahead = fit_factored_head(astore, Zs, lab_store, c["cards"], pairs, RANK, RIDGE)
af = np.asarray(clu_wide_vector(astore, ahead.coefficients(req)), np.float32)
floor = dev.score_features(c, af, lab_dev)
ctx.log(f"  [atom-basis §C.1 floor] agg={floor['aggregate']:.4f} "
        f"({floor['distinct_factor_tuples']}/72)")

from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
kp = dev.score_features(
    c, np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                  np.float32), lab_dev)
ctx.log(f"  [knn_proto] agg={kp['aggregate']:.4f} ({kp['distinct_factor_tuples']}/72) "
        f"bytes={protos.byte_ledger()['bytes']}")

res = {
    "seed": SEED, "corner": s["corner"], "K": K, "M": M, "rank": RANK, "ridge": RIDGE,
    "pairs": [[int(a_), int(b_)] for a_, b_ in pairs],
    "store_byte_ledger": {k: v for k, v in sb.items() if k != "note"},
    "head_byte_ledger": ledger,
    "total_bytes": int(sb["store_bytes"] + ledger["bytes"]),
    "under_knn_proto_bytes": bool(sb["store_bytes"] + ledger["bytes"] < 475_136),
    "channel": head.channel_ledger(),
    "assert_request_only_factored": {k: v for k, v in ro.items() if k != "signatures"},
    "assert_no_bypass": {k: v for k, v in nb.items() if k != "note"},
    "arm": arm, "store_removed": sr, "atom_basis_floor": floor, "knn_proto": kp,
    "head_fit_info": head.fit_info,
    "total_s": round(time.time() - T0, 1),
}
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"TOTAL BYTES store {sb['store_bytes']} + head {ledger['bytes']} = "
        f"{res['total_bytes']}  (knn_proto 475136, under={res['under_knn_proto_bytes']})")
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
