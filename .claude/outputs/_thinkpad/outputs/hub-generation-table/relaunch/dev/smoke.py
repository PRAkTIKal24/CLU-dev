"""RELAUNCH smoke: one DEV corner, one variant — a timing probe, nothing else."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx, dev, fast, arm2
T0 = time.time()
c = ctx.build_context(0)
ctx.log(f"context built in {time.time()-T0:.1f}s; pool={c['pool'].size} feat_shape={c['feat_shape']}")
D = int(np.prod(c["feat_shape"])); cards = c["cards"]
from chlu.core.gen_dream import requests_from_labels
from chlu.core.gen_factored_read import default_pairs
t = time.time()
s = dev.dev_setup(c, dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True))
X_store = c["gather"](s["pool"]); ctx.log(f"gather {X_store.shape} in {time.time()-t:.1f}s")
lab_store = c["factor_labels"](c["grid"], s["pool"])
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
t = time.time()
_, dead, ow, _ = fast.channel_geometry(X_store, c["cfg"], "smoke_s0", log=ctx.log)
ctx.log(f"channel geometry in {time.time()-t:.1f}s  dead={int(dead.sum())}")
K = 32
labels = fast.balanced_labels(ow, dead, K, D)
perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
req = requests_from_labels(s["lab_dev"], "full", 0, 0)
t = time.time(); views = arm2.well_views(Zs, perm); ctx.log(f"views {time.time()-t:.1f}s")
PAIRS = default_pairs(c["names"], ["floor_huexscale", "object_huexscale"])
N_PAIR = sum(cards[f]*cards[g] for f, g in PAIRS)
N_ADD_RED = 1 + sum(x-1 for x in cards)
A = K*26
R = arm2.budget_rank(K, A, N_ADD_RED, N_PAIR, D//K)
ctx.log(f"K={K} A={A} n_pair={N_PAIR} n_add_red={N_ADD_RED} rank={R}")
t = time.time()
out = arm2.run_variant(c, s, Zs, lab_store, perm, req, s["lab_dev"], views=views,
                       method="kmeans", K=K, A=A, rank=R, pairs=PAIRS, ridge_add=10.0,
                       ridge_pair=100.0, reduced=True, metric_correct=True,
                       alloc="uniform", seed=0, n_add=N_ADD_RED, n_pair=N_PAIR)
ctx.log(f"variant in {time.time()-t:.1f}s -> agg={out['aggregate']:.4f} "
        f"tuples={out['distinct_factor_tuples']} bytes={out['total_bytes']}")
ctx.log(f"TOTAL {time.time()-T0:.1f}s")
