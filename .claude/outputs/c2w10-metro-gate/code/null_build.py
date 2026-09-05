"""PART B.4 -- the RE-CONSTRUCTED drift-free NULL.

`out-of-control` (the null PREREG-C2W10 6.2 registered) has no data source.  The control is
re-constructed here, never quietly dropped: a fixed-seed permutation of the PAIR SEQUENCE.
Each pair (32 features, target) is kept intact; only the stream order is destroyed.
=> the joint marginal P(X, y) is preserved EXACTLY; all regime/ordering structure is gone.
Positive control: persistence and both seasonal-naives are functions of a pair's OWN features,
so their scores are invariant under the shuffle by construction.
"""
import hashlib, json, os
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
SEED = 1
d = np.load(os.path.join(OUT, "pairs.npz"))
X, Y, tgt = d["X"], d["y"], d["tgt"]
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(Y))
Xs, Ys, ts = X[perm], Y[perm], tgt[perm]
h = hashlib.sha256(); h.update(Xs.tobytes()); h.update(Ys.tobytes()); h.update(ts.astype(np.int64).tobytes())
path = os.path.join(OUT, "pairs_shuffled_seed1.npz")
np.savez(path, X=Xs, y=Ys, tgt=ts, perm=perm)
meta = dict(construction=("fixed-seed (numpy default_rng(1)) uniform permutation of the 34,848 "
                          "prequential pairs; each pair's 32-D feature vector and its target are "
                          "kept intact, only the stream ORDER is permuted"),
            path=path, sha256=h.hexdigest(), seed=SEED, n_pairs=int(len(Y)),
            perm_sha256=hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest(),
            limits=["P(X,y) marginal preserved exactly; conditional P(y|X) preserved exactly; "
                    "ONLY the temporal ordering is destroyed",
                    "it is NOT a drift-free DATA SOURCE (the sampling distribution is the pooled "
                    "6-year mixture, not a stationary generating process observed in the wild)",
                    "autocorrelation ACROSS pairs is destroyed but autocorrelation WITHIN a pair's "
                    "24-lag window is retained, by design -- persistence must be invariant",
                    "not a substitute for INSECTS out-of-control: no class-emergence structure, "
                    "different task type (regression), our construction not the literature's"])
with open(os.path.join(OUT, "null_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
