"""Calibrate a PRE-REGISTERED prediction for B from the 25 corners that carry a ceiling.
⛔ Read-only over the DEV ledger. ⛔ B is not touched. Run BEFORE the scored run."""
import json
import numpy as np
import rivals
rows = [r for r in rivals.ledger_rows() if r.get("tag") == "ref"]
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["corner_idx"]] = r["headline"]["tuples_valid_novel"]
ks = sorted(set(by["ceiling"]) & set(by["clu_linear"]) & set(by["knn_proto"]))
C = np.array([by["ceiling"][k] for k in ks], float)
A = np.array([by["clu_linear"][k] for k in ks], float)
R = np.array([by["knn_proto"][k] for k in ks], float)
out = {"n_corners_with_ceiling": len(ks), "corners": ks,
       "ceiling_mean": float(C.mean()), "ceiling_sd": float(C.std(ddof=1)),
       "B_ceiling_inherited": 67.67, "B_ceiling_inherited_sd": 2.49}
for name, Y in (("clu_linear", A), ("knn_proto", R), ("delta", A - R)):
    b, a0 = np.polyfit(C, Y, 1)
    yh = a0 + b * C
    resid = Y - yh
    s = float(resid.std(ddof=2))
    pred = float(a0 + b * 67.67)
    n = len(ks); Cb = C.mean(); Sxx = float(((C - Cb) ** 2).sum())
    se_pred = s * np.sqrt(1 + 1 / n + (67.67 - Cb) ** 2 / Sxx)
    from scipy import stats
    t = float(stats.t.ppf(0.975, n - 2))
    out[name] = {"slope": float(b), "intercept": float(a0),
                 "r": float(np.corrcoef(C, Y)[0, 1]), "resid_sd": s,
                 "prediction_at_B_ceiling_67.67": pred,
                 "pred_interval_95": [pred - t * se_pred, pred + t * se_pred]}
    print(f"{name:12s} slope={b:+.4f} r={out[name]['r']:+.3f} resid_sd={s:5.2f} "
          f"-> at B's ceiling 67.67: {pred:6.2f}  PI95 "
          f"[{pred - t * se_pred:6.2f}, {pred + t * se_pred:6.2f}]")
# unconditional paired prediction for a single new corner
d = A - R
from scipy import stats
n = len(d); t = float(stats.t.ppf(0.975, n - 1))
se = float(d.std(ddof=1) * np.sqrt(1 + 1 / n))
out["delta_unconditional"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                              "pred_interval_95_for_ONE_new_corner":
                                  [d.mean() - t * se, d.mean() + t * se]}
print(f"\nunconditional Δ on ONE new corner: {d.mean():+.2f} "
      f"PI95 [{d.mean() - t * se:+.2f}, {d.mean() + t * se:+.2f}]  (n={n})")
json.dump(out, open(str(rivals.OUT / "dev" / "predict_B.json"), "w"), indent=2)
