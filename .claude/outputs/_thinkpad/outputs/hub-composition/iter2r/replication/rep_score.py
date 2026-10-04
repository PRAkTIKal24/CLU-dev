"""Independent replication spoke - the four numbers.

Written from the pinned library only. No hub script was read or imported.

R1  HEADLINE for the object  (clu_linear: K=32/M=26 store, request-addressed LINEAR
    read through gen_compose.clu_wide_vector, factored head rank 16 ridge 10.0)
R2  HEADLINE for knn_proto
R3  HEADLINE for the landscape-removed control (same read path, modes ZEROED)
R4  the decoder ceiling (decode the true cached feature of each eval_b row)
"""
import json, os, sys, time, argparse
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

import chlu, jax, numpy
import jax.numpy as jnp
log(f"chlu.__file__ = {chlu.__file__}")

from chlu.config import get_default_config
from chlu.core.gen_bridge import dataset_module
from chlu.core.gen_store import (GenStore, store_bytes, well_channels_from_labels,
                                 channel_correlation, block_diagonality)
from chlu.core.gen_partition import balanced_correlation_partition, partition_report
from chlu.core.gen_compose import clu_wide_vector
from chlu.core.gen_factored_read import (fit_factored_head, default_pairs,
                                         assert_request_only_factored)
from chlu.core.gen_dream import requests_from_labels
from chlu.core.gen_request import FactorPrototypes, requests_from_rows
from chlu.eval.gen_probe import predict_factors
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows
from chlu.experiments.exp_gen_compose import resolve_decoder
from chlu.experiments.exp_gen_cell import load_external_classifier, render, score_images

ap = argparse.ArgumentParser()
ap.add_argument("--kmeans-seed-rule", default="library",
                choices=["library", "independent"])
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--tag", default="main")
ap.add_argument("--partition", default="ward", choices=["ward", "average"])
ap.add_argument("--store-fit", default="lloyd", choices=["lloyd", "shipped"],
                help="'shipped' = gen_store.init_store + gen_store.train_store (adam)")
args = ap.parse_args()

REP = Path(os.environ["REP"])
SEED, K, M, RANK, RIDGE = 0, 32, 26, 16, 10.0
BATCH = 256

ctx = np.load(REP / "context.npz")
meta = json.loads((REP / "context_meta.json").read_text())
X_store = ctx["X_store"]; X_b = ctx["X_b"]
lab_store = ctx["lab_store"].astype(np.int64); lab_b = ctx["lab_b"].astype(np.int64)
labels_vec = ctx["labels_vec"]; store_train = ctx["store_train"]; eval_b = ctx["eval_b"]
names = meta["names"]; cards = meta["cardinalities"]
feat_shape = tuple(meta["cache_feat_shape"])
log(f"context: store {X_store.shape} B {X_b.shape}  K={K} M={M}")

# ---- the partition -----------------------------------------------------------
# gen_factored_read.mode_basis REFUSES a ragged store, and gen_store's shipped
# correlation_grouping at K=32 is ragged ([36..156]).  The pinned library's own
# equal-well partition over the SAME |corr| and the SAME ward linkage is
# gen_partition.balanced_correlation_partition -- that is the only construction in
# the pinned library that both (a) is the correlation grouping and (b) yields the
# equal wells the factored head requires.
dead = ctx["dead"].astype(bool)
corr = channel_correlation(X_store)
labels_bal, pinfo = balanced_correlation_partition(corr, K, dead, method=args.partition)
wc = well_channels_from_labels(labels_bal, K)
bd_bal = partition_report(labels_bal, corr)
log(f"partition: {pinfo['partition']} linkage={pinfo['linkage_method']} "
    f"dead={pinfo['n_dead_channels']} well sizes {pinfo['well_sizes']} "
    f"in-block |corr|^2={bd_bal['in_block_fraction']:.4f}")
bd_ragged = block_diagonality(corr, labels_vec)
log(f"  (for reference, the RAGGED shipped grouping at K=32 scores "
    f"in-block |corr|^2={bd_ragged['in_block_fraction']:.4f} and is REFUSED by mode_basis)")
if len(set(len(w) for w in wc)) != 1:
    raise SystemExit(f"partition is ragged: {sorted(set(len(w) for w in wc))}")


def kmeanspp(Z, m, rng):
    n = Z.shape[0]
    idx = [int(rng.integers(n))]
    d2 = np.sum((Z - Z[idx[0]]) ** 2, axis=1)
    for _ in range(1, m):
        tot = float(d2.sum())
        p = (d2 / tot) if tot > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))
    return Z[np.asarray(idx)].astype(np.float32)


def lloyd(Z, C, iters, chunk=8192):
    n, m = Z.shape[0], C.shape[0]
    C = C.astype(np.float32).copy()
    n_iter = 0
    for it in range(int(iters)):
        n_iter = it + 1
        cn = np.sum(C * C, axis=1)
        assign = np.empty(n, np.int64)
        for i in range(0, n, chunk):
            Zi = Z[i:i + chunk]
            d = np.sum(Zi * Zi, axis=1)[:, None] - 2.0 * (Zi @ C.T) + cn[None, :]
            assign[i:i + chunk] = np.argmin(d, axis=1)
        newC = C.copy()
        for c in range(m):
            sel = assign == c
            if sel.any():
                newC[c] = Z[sel].mean(axis=0)
        moved = float(np.max(np.abs(newC - C)))
        C = newC
        if moved == 0.0:
            break
    return C, n_iter


# per-well Lloyd k-means, seed 0.  The seeding RNG stream follows the pinned
# library's own (gen_store.init_store: ONE default_rng consumed well-by-well).
if args.kmeans_seed_rule == "library":
    rng = np.random.default_rng(int(SEED) * 2_654_435_761 % (2 ** 32))
    seeds_iter = [rng] * K
else:
    seeds_iter = [np.random.default_rng(1_000_003 * (j + 1) + int(SEED)) for j in range(K)]

if args.store_fit == "shipped":
    # the PINNED library's own store fit: k-means++ init then the store-only
    # adam objective (gen_store.train_store), exactly as exp_gen_compose does.
    from chlu.core.gen_store import init_store, train_store, reconstruction_mse
    key = jax.random.PRNGKey(int(SEED) * 10_007 + 20_260_911)
    key, ik, tk = jax.random.split(key, 3)
    store = init_store(X_store, wc, M, feat_shape, ik, seed=int(SEED))
    store, hist, key = train_store(store, X_store, get_default_config().gen_store, tk, log=log)
    modes = [np.asarray(m, np.float32) for m in store.modes]
    kinfo = [{"fit": "shipped init_store+train_store", "loss_first": float(hist[0]),
              "loss_final": float(hist[-1])}]
    log(f"shipped store fit done (loss {hist[0]:.5f} -> {hist[-1]:.5f})")
else:
    modes, kinfo = [], []
    for j, w in enumerate(wc):
        Zw = np.ascontiguousarray(X_store[:, np.asarray(w, np.int64)])
        C0 = kmeanspp(Zw, M, seeds_iter[j])
        C, ni = lloyd(Zw, C0, args.iters)
        modes.append(C)
        kinfo.append({"well": j, "d": int(Zw.shape[1]), "lloyd_iters": ni})
    log(f"k-means done: mean Lloyd iters {np.mean([k['lloyd_iters'] for k in kinfo]):.1f}")
    store = GenStore(wc, M, feat_shape, jax.random.PRNGKey(0), init=modes)
sled = store_bytes(store)
log(f"store_bytes = {sled['store_bytes']}  (K={sled['n_wells']} M={sled['n_modes']} D={sled['dim']})")

pairs = default_pairs(names, ["floor_huexscale", "object_huexscale"])
log(f"pairs = {pairs}  ({[(names[a], names[b]) for a, b in pairs]})")
head = fit_factored_head(store, X_store, lab_store, cards, pairs, rank=RANK, ridge=RIDGE)
hled = head.byte_ledger()
log(f"head bytes = {hled['bytes']}  (n_add={hled['n_additive_columns']} "
    f"n_pair={hled['n_pair_columns']} rank={hled['rank']})")
log(f"OBJECT BYTES = {sled['store_bytes']} + {hled['bytes']} = {sled['store_bytes'] + hled['bytes']}")

req_b = requests_from_labels(lab_b, "full", 0, 0)
assert np.array_equal(req_b, lab_b), "protocol 'full' must return the labels verbatim"

ro = assert_request_only_factored(store, head, req_b, decoy_features=X_b)
log("assert_request_only_factored: " +
    json.dumps({k: v for k, v in ro.items() if isinstance(v, (bool, int, float, str))})[:400])

coef = head.coefficients(req_b)
Z_clu = clu_wide_vector(store, coef)
log(f"clu_linear features {Z_clu.shape}  mean|.|={float(np.abs(Z_clu).mean()):.4f}")

zero_store = GenStore(wc, M, feat_shape, jax.random.PRNGKey(0),
                      init=[np.zeros_like(m) for m in modes])
Z_zero = clu_wide_vector(zero_store, coef)
log(f"store_removed_zero features  max|.| = {float(np.abs(Z_zero).max()):.6g} (expect 0)")

cfg = get_default_config()
cr = os.environ["CHLU_CLAUDE_ROOT"]
cfg.gen_cell.claude_root = cr; cfg.gen_compose.claude_root = cr
for f in ("decoder_source", "winner_root", "winner_decoder_pattern", "expected_decoder_recipe"):
    setattr(cfg.gen_compose, f, getattr(cfg.gen_cell, f))
split, values = dataset_module("shapes3d").build_split(cfg.gen_data.data_root)
grid = split.grid
protos = FactorPrototypes.fit(grid, X_store, store_train)
pled = protos.byte_ledger()
log(f"knn_proto bytes = {pled['bytes']}  ({pled['n_prototypes']} prototypes, D={pled['dim']})")
Z_knn = protos.embed_many(requests_from_rows(grid, eval_b))
log(f"knn_proto features {Z_knn.shape}")

bdir = bridge_dir(cfg, "shapes3d", "recon", SEED)
decoder, dec_info, bman = resolve_decoder(cfg, bdir, SEED, log=log)
clf, clf_info = load_external_classifier(cfg, SEED, log=log)

train_tuples = {tuple(int(x) for x in r) for r in lab_store}
log(f"store was fitted on {len(train_tuples)} distinct factor tuples ({lab_store.shape[0]} rows)")


def novel(t):
    return tuple(int(x) for x in t) not in train_tuples


def headline(tag, Z):
    imgs = render(decoder, Z, feat_shape, BATCH)
    sc = score_images(clf, imgs, lab_b, names, BATCH)
    preds = predict_factors(clf, imgs, BATCH)
    valid = (preds == req_b).all(axis=1)
    nov = np.asarray([novel(t) for t in req_b], bool)
    sel = valid & nov
    tuples = {tuple(int(x) for x in req_b[i]) for i in np.flatnonzero(sel)}
    hl = len(tuples)
    n_valid = int(valid.sum())
    ident = (hl == int(round(72 * sc["exact_match"])))
    log(f"  [{tag}] exact_match={sc['exact_match']:.6f}  aggregate={sc['aggregate']:.4f}  "
        f"n_valid={n_valid}  n_novel={int(nov.sum())}  HEADLINE={hl}  "
        f"72*exact_match={72 * sc['exact_match']:.4f}  identity_holds={ident}")
    return {"arm": tag, "HEADLINE": hl, "n_valid": n_valid,
            "n_novel_requests": int(nov.sum()),
            "exact_match": float(sc["exact_match"]),
            "aggregate": float(sc["aggregate"]),
            "per_factor": sc["per_factor"],
            "72x_exact_match": float(72 * sc["exact_match"]),
            "identity_holds": bool(ident),
            "n_distinct_predicted_tuples": int(np.unique(preds, axis=0).shape[0]),
            "valid_mask": [int(b) for b in valid]}


log("=== SCORING ===")
res = {}
res["R1_clu_linear"] = headline("R1 clu_linear", Z_clu)
res["R2_knn_proto"] = headline("R2 knn_proto", Z_knn)
res["R3_store_removed_zero"] = headline("R3 store_removed_zero", Z_zero)
res["R4_decoder_ceiling"] = headline("R4 decoder_ceiling", X_b)

d = (np.asarray(res["R1_clu_linear"]["valid_mask"], float)
     - np.asarray(res["R2_knn_proto"]["valid_mask"], float))
delta = float(d.mean()); sd = float(d.std(ddof=1))
se = sd / np.sqrt(d.size)
ci = (delta - 1.99346 * se, delta + 1.99346 * se)
log(f"paired Delta (R1-R2) = {delta:+.4f} aggregate = {72 * delta:+.2f} tuples; "
    f"CI95 [{ci[0]:+.4f},{ci[1]:+.4f}] = [{72 * ci[0]:+.2f},{72 * ci[1]:+.2f}] tuples")

out = {
    "spoke": "independent replication (hub/composition-2 iter2r, seed 0, eval_b)",
    "tag": args.tag, "kmeans_seed_rule": args.kmeans_seed_rule, "lloyd_iters_max": args.iters,
    "partition_linkage": args.partition, "store_fit": args.store_fit,
    "results": res,
    "bytes": {
        "store": sled, "head": hled, "object_total": int(sled["store_bytes"] + hled["bytes"]),
        "knn_proto": pled,
        "ratio_object_over_knn": float((sled["store_bytes"] + hled["bytes"]) / pled["bytes"]),
    },
    "paired": {"delta_aggregate": delta, "delta_tuples": 72 * delta,
               "ci95_aggregate": list(ci), "ci95_tuples": [72 * ci[0], 72 * ci[1]]},
    "kmeans": kinfo,
    "partition_info": pinfo,
    "partition_block_diagonality": bd_bal,
    "ragged_grouping_block_diagonality": bd_ragged,
    "n_train_tuples": len(train_tuples),
    "n_eval_b": int(eval_b.size),
    "all_requests_distinct": bool(np.unique(req_b, axis=0).shape[0] == req_b.shape[0]),
    "all_requests_novel": bool(all(novel(t) for t in req_b)),
    "assert_request_only_factored": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                     for k, v in ro.items()},
    "context_meta": meta,
}
p = REP / f"RESULTS_{args.tag}.json"
p.write_text(json.dumps(out, indent=2, default=str))
log(f"written -> {p}")
