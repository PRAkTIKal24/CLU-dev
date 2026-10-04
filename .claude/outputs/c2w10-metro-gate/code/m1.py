"""STEP 3 -- M1: the loader/protocol control. Naive baselines ONLY, reported before M2."""
import json, os
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
d = np.load(os.path.join(OUT, "pairs.npz"))
X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
n = len(y)

preds = {}
preds["persistence"] = X[:, 0]                       # y_i  == y_{j-24}
preds["seasonal_naive_24"] = X[:, 0]                 # y_{j-24}  (registered degeneracy)
preds["seasonal_naive_168"] = X[:, 25]               # y_{i-144} == y_{j-168}
preds["seasonal_naive_168_smooth3"] = X[:, 24:27].mean(1)
preds["mean_so_far"] = np.concatenate([[y[0]], np.cumsum(y)[:-1] / np.arange(1, n)])
preds["window_mean_24"] = X[:, :24].mean(1)
preds["global_mean"] = np.full(n, y.mean())          # oracle-mean reference (not causal; labelled)


def mets(p):
    e = p - y
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                mape_pct=float(np.mean(np.abs(e) / np.maximum(y, 1.0)) * 100),
                bias=float(e.mean()))


res = {k: mets(v) for k, v in preds.items()}
res["_n_scored"] = n
res["_y_stats"] = dict(mean=float(y.mean()), std=float(y.std()), min=float(y.min()), max=float(y.max()))
res["_degeneracy_check_abs_diff"] = float(np.abs(preds["persistence"] - preds["seasonal_naive_24"]).max())
with open(os.path.join(OUT, "m1.json"), "w") as f:
    json.dump(res, f, indent=2)
np.savez_compressed(os.path.join(OUT, "preds_m1.npz"), **preds)
for k, v in res.items():
    if not k.startswith("_"):
        print(f"{k:30s} MAE {v['mae']:9.3f}  RMSE {v['rmse']:9.3f}  bias {v['bias']:+8.2f}")
print("n =", n, " y mean/std:", res["_y_stats"]["mean"], res["_y_stats"]["std"])
print("persistence vs seasonal_naive_24 max |diff| =", res["_degeneracy_check_abs_diff"])
