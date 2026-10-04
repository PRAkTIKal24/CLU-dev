"""⭐ THE POOLED DEV ESTIMATE — both corner families, disjoint in hue_lo.
⛔ Read-only over the DEV ledger. ⛔ B is not touched."""
import json
import numpy as np
from scipy import stats
import comp, rivals
rows = rivals.ledger_rows()
def col(tag, arm):
    return {r["corner_idx"]: r["headline"]["tuples_valid_novel"]
            for r in rows if r.get("tag") == tag and r["arm"] == arm}
fams = {}
a0, r0 = col("ref", "clu_linear"), col("ref", "knn_proto")
k0 = sorted(set(a0) & set(r0))
fams["declared (hue_lo 0.0/0.2/0.4/0.6)"] = np.array([a0[k]-r0[k] for k in k0], float)
a1, r1 = col("fresh", "clu_linear"), col("fresh", "knn_proto")
k1 = sorted(set(a1) & set(r1))
fams["FRESH (hue_lo 0.1/0.3/0.5/0.7)"] = np.array([a1[k]-r1[k] for k in k1], float)
def rep(x, label):
    n = x.size; se = float(x.std(ddof=1)/np.sqrt(n)); t = float(stats.t.ppf(0.975, n-1))
    lo, hi = x.mean()-t*se, x.mean()+t*se
    v = comp.r1_verdict({"delta": float(x.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
    print(f"{label:46s} n={n:3d} delta={x.mean():+7.3f} CI95[{lo:+7.3f},{hi:+7.3f}] "
          f"sd={x.std(ddof=1):6.2f}  {v}")
    return {"n": int(n), "delta": float(x.mean()), "ci95": [lo, hi],
            "sd": float(x.std(ddof=1)), "R1": v}
out = {"families": {}}
for k, v in fams.items():
    out["families"][k] = rep(v, k)
allv = np.concatenate(list(fams.values()))
out["POOLED"] = rep(allv, "⭐ POOLED, both families")
print(f"\nmeans /72 pooled: clu_linear "
      f"{np.mean([a0[k] for k in k0]+[a1[k] for k in k1]):.2f}  knn_proto "
      f"{np.mean([r0[k] for k in k0]+[r1[k] for k in k1]):.2f}")
out["means"] = {"clu_linear": float(np.mean([a0[k] for k in k0]+[a1[k] for k in k1])),
                "knn_proto": float(np.mean([r0[k] for k in k0]+[r1[k] for k in k1]))}
json.dump(out, open(str(rivals.OUT / "dev" / "pooled_dev.json"), "w"), indent=2)
