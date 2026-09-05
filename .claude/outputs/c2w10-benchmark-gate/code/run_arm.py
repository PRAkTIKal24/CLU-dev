"""Prequential runner. Saves per-instance predictions; ALL metrics are computed later from these.

usage: run_arm.py <stream.npy> <arm> <tag> [seed]
arms: nochange | arf10 | arf100 | arf<N> | samknn_<L>_<raw|std> | knns_<L>_<raw|std>
"""
import sys, time, json, os
import numpy as np

stream, arm, tag = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

D = np.load(stream)
X = D[:, :-1].copy()
y = D[:, -1].astype(np.int64)
n = len(y)
t0 = time.time()


def causal_standardise(X):
    """Prequential (causal) per-feature standardisation: statistics use only instances < t.
    The instance at t is scaled with the running mean/var of the PAST, then folded in."""
    n, d = X.shape
    out = np.empty_like(X)
    mean = np.zeros(d)
    m2 = np.zeros(d)
    cnt = 0
    for i in range(n):
        if cnt < 2:
            out[i] = 0.0
        else:
            sd = np.sqrt(m2 / (cnt - 1))
            sd[sd < 1e-12] = 1.0
            out[i] = (X[i] - mean) / sd
        cnt += 1
        delta = X[i] - mean
        mean += delta / cnt
        m2 += delta * (X[i] - mean)
    return out


if arm == "nochange":
    pred = np.empty(n, dtype=np.int64)
    pred[0] = -1
    pred[1:] = y[:-1]
    meta = {}

elif arm.startswith("arf"):
    from river import forest
    nm = int(arm[3:])
    model = forest.ARFClassifier(n_models=nm, seed=seed)
    pred = np.empty(n, dtype=np.int64)
    cols = [f"f{j}" for j in range(33)]
    for i in range(n):
        d = dict(zip(cols, X[i]))
        p = model.predict_one(d)
        pred[i] = -1 if p is None else p
        model.learn_one(d, y[i])
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    meta = {"n_models": nm}

elif arm.startswith(("samknn_", "knns_")):
    from samknn_port import SAMKNN
    kind, L, scal = arm.split("_")
    L = int(L)
    if scal == "std":
        X = causal_standardise(X)
    if kind == "samknn":
        model = SAMKNN(n_neighbors=5, knnWeights="distance", maxSize=L,
                       LTMSizeProportion=0.4, minSTMSize=50,
                       recalculateSTMError=False, useLTM=True)
    else:
        model = SAMKNN(n_neighbors=5, knnWeights="distance", maxSize=L,
                       recalculateSTMError=None, useLTM=False)
    pred = model.alternateFitPredict(X, y, progress=10000).astype(np.int64)
    meta = {"L_max": L, "scaling": scal,
            "mean_STM": float(np.mean(model.STMSizes)), "mean_LTM": float(np.mean(model.LTMSizes)),
            "final_STM": int(model.STMSizes[-1]), "final_LTM": int(model.LTMSizes[-1]),
            "classifierChoice_counts": np.bincount(
                np.array(model.classifierChoice, dtype=np.int64), minlength=3).tolist()}
else:
    raise SystemExit("unknown arm " + arm)

wall = time.time() - t0
os.makedirs("preds", exist_ok=True)
np.savez_compressed(f"preds/{tag}.npz", pred=pred, y=y)
json.dump({"arm": arm, "tag": tag, "seed": seed, "stream": os.path.basename(stream),
           "wall_s": wall, "acc_all": float(np.mean(pred == y)),
           "acc_from1": float(np.mean(pred[1:] == y[1:])), **meta},
          open(f"preds/{tag}.json", "w"), indent=1)
print(f"DONE {tag} acc_from1={np.mean(pred[1:]==y[1:])*100:.4f}% wall={wall:.0f}s", flush=True)
