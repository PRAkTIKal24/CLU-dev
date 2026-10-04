"""⭐ The construction gap, measured on ALL DEV corners. ⛔ Read-only; B not touched."""
import json
import numpy as np
from scipy import stats
import comp, rivals
rows = rivals.ledger_rows()
def col(tag, arm="clu_linear"):
    return {r["corner_idx"]: r["headline"]["tuples_valid_novel"]
            for r in rows if r.get("tag") == tag and r["arm"] == arm}
mine = col("ref"); knn = col("ref", "knn_proto"); gp = col("ctor_genpart_25")
ks = sorted(set(mine) & set(knn) & set(gp))
A = np.array([mine[k] for k in ks], float)
G = np.array([gp[k] for k in ks], float)
R = np.array([knn[k] for k in ks], float)
out = {"n": len(ks), "corners": ks, "B": {"mine": 42, "genpart": 25, "knn": 12}}
def paired(x, y, label):
    d = x - y; n = d.size
    se = float(d.std(ddof=1)/np.sqrt(n)); t = float(stats.t.ppf(0.975, n-1))
    lo, hi = d.mean()-t*se, d.mean()+t*se
    v = comp.r1_verdict({"delta": float(d.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
    print(f"{label:46s} n={n:2d} delta={d.mean():+7.3f} CI95[{lo:+7.3f},{hi:+7.3f}] "
          f"W/T/L={int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}  {v}")
    return {"n": n, "delta": float(d.mean()), "ci95": [lo, hi], "R1": v,
            "wins": int((d>0).sum()), "ties": int((d==0).sum()), "losses": int((d<0).sum()),
            "sd": float(d.std(ddof=1))}
print(f"corners with all three: n={len(ks)}")
print(f"mean/72  mine {A.mean():.2f}   genpart {G.mean():.2f}   knn_proto {R.mean():.2f}\n")
out["means"] = {"mine": float(A.mean()), "genpart": float(G.mean()), "knn": float(R.mean())}
out["mine_vs_genpart"] = paired(A, G, "the CONSTRUCTION gap (mine - genpart)")
out["mine_vs_knn"] = paired(A, R, "mine vs knn_proto  (THE BAR)")
out["genpart_vs_knn"] = paired(G, R, "genpart vs knn_proto  (THE BAR, spoke's build)")
print(f"\nOn B: mine 42, genpart 25 -> construction gap +17. "
      f"On {len(ks)} DEV corners the same gap is {A.mean()-G.mean():+.2f}.")
json.dump(out, open(str(rivals.OUT / "dev" / "construction_gap.json"), "w"), indent=2)
