"""⭐⭐ DOES *COMPOSITION* COST ANYTHING ON TOP OF THE FIDELITY BOUND?

Every number in this hub so far is on rows the store has SEEN, yet the target is about
UNSEEN combinations.  ⛔ I cannot use B for this.  So: build a SURROGATE held-out cell —
the OPPOSITE corner of the factor grid, same shape and same size (2*2*2*3*1*3 = 72) as
Montero's — hold it out of the store fit, and compare round-trip + request-read on
  (a) rows the store was fitted on   vs   (b) the surrogate cell it has never seen.
⛔ The surrogate is drawn from `split.train_rows` ONLY. ⛔ B never enters, not even as labels.
⛔ Reads features from the frozen cache only (seed 0, scope=full). Nothing is encoded.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np, jax
from chlu.core.gen_store import (address_codes, channel_correlation, dead_channels,
                                 grouping_labels, init_store, train_store,
                                 well_channels_from_labels)
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior
from chlu.core.gen_dream import fit_address_book, requests_from_labels
from chlu.core.gen_bridge import features_for_rows, load_feature_cache
from chlu.eval.gen_probe import factor_labels
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows
from mech_driver import build_context, score

ctx = build_context(0)
cfg, grid = ctx["cfg"], ctx["grid"]
names, cards = ctx["names"], ctx["cards"]
split = ctx["split"]
bdir = bridge_dir(cfg, "shapes3d", "recon", 0)
crows, feats, cman = load_feature_cache(bdir, verify=False)
D = int(cman["D"])
take = lambda r: features_for_rows(crows, feats, np.asarray(r, np.int64).ravel()).reshape((-1, D))

# ---- the SURROGATE cell: the OPPOSITE corner, same shape as Montero's --------
train = np.asarray(split.train_rows, np.int64)
Ltr = factor_labels(grid, train)
lo = {0: (0, 1), 1: (0, 1), 2: (0, 1), 3: (0, 1, 2), 4: (0,), 5: (0, 1, 2)}
m = np.ones(train.size, bool)
for j, vals in lo.items():
    m &= np.isin(Ltr[:, j], np.asarray(vals))
sur = train[m]
print(f"surrogate cell: {sur.size} rows, {np.unique(factor_labels(grid, sur), axis=0).shape[0]} "
      f"distinct tuples (Montero's B is 72)", flush=True)

pool = np.asarray(select_store_rows(split, cfg, 0)["store_train"], np.int64)
sur_set = set(sur.tolist())
sur_tup = set(map(tuple, factor_labels(grid, sur).tolist()))
Lp = factor_labels(grid, pool)
keep = np.array([t not in sur_tup for t in map(tuple, Lp.tolist())], bool)
pool_seen = pool[keep]
print(f"store pool: {pool.size} -> {pool_seen.size} after removing every row whose TUPLE is in the "
      f"surrogate cell ({int((~keep).sum())} removed)", flush=True)

rng = np.random.default_rng(4242)
heldin = pool_seen[rng.permutation(pool_seen.size)[:512]]
X_fit, X_in, X_sur = take(pool_seen), take(heldin), take(sur)
L_in, L_sur, L_fit = (factor_labels(grid, heldin), factor_labels(grid, sur),
                      factor_labels(grid, pool_seen))
gs = cfg.gen_store
Xc = X_fit[: int(gs.n_corr)]
lab5, _ = grouping_labels(channel_correlation(Xc), 5, "correlation", 0, gs.grouping_method,
                          dead=dead_channels(Xc), min_well_frac=float(gs.min_well_frac))
wc = well_channels_from_labels(lab5, 5)
k1, k2 = jax.random.split(jax.random.PRNGKey(20260916))
t = time.time()
store = init_store(X_fit, wc, 64, ctx["feat_shape"], k1, seed=0)
store, _, _ = train_store(store, X_fit, gs, k2, log=None)
print(f"(store fitted on the reduced pool in {time.time()-t:.0f}s)", flush=True)
book = fit_address_book(address_codes(store, X_fit), L_fit, cards,
                        float(cfg.gen_cell.address_smoothing))

out = {"surrogate_rows": int(sur.size), "pool_seen": int(pool_seen.size),
       "rows_removed": int((~keep).sum())}
for tag, Xs, Ls in (("SEEN (fitted on)", X_in, L_in), ("SURROGATE (never seen)", X_sur, L_sur)):
    ce = score(ctx, Xs, Ls, None)
    code = address_codes(store, Xs)
    rt = np.asarray(read_compose(store, code), np.float32).reshape(Xs.shape)
    rr = score(ctx, rt, Ls, None)
    rel = float(np.mean(np.linalg.norm(rt - Xs, axis=1) /
                        np.maximum(np.linalg.norm(Xs, axis=1), 1e-9)))
    R = requests_from_labels(Ls, "full", 0, 0)
    rd = np.asarray(read_compose(store, compose_codes(request_posterior(book, R))["code"]),
                    np.float32)
    rq = score(ctx, rd, Ls, None)
    out[tag] = {"n": ce["n"], "ceiling": ce["distinct_valid"], "ceiling_exact": ce["exact_match_frac"],
                "roundtrip": rr["distinct_valid"], "roundtrip_exact": rr["exact_match_frac"],
                "roundtrip_relerr": rel, "request_read": rq["distinct_valid"],
                "request_read_exact": rq["exact_match_frac"]}
    print(f"{tag:24s} n={ce['n']:4d} | CEILING {ce['distinct_valid']:4d} "
          f"({ce['exact_match_frac']:.4f}) | ROUND-TRIP {rr['distinct_valid']:4d} "
          f"({rr['exact_match_frac']:.4f}, relerr {rel:.4f}) | REQUEST-READ "
          f"{rq['distinct_valid']:4d} ({rq['exact_match_frac']:.4f})", flush=True)
json.dump(out, open(os.path.join(os.environ['OUT'], 'mech', 'surrogate.json'), 'w'), indent=2)
print("done")
