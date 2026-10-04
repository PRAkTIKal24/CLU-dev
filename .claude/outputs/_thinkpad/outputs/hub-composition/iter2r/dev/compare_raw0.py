"""⭐ Does the object's headline ride on UNCONVERGED codebook noise?

The §A.5 replication spoke's second criticism, stated in its own words:
    "My least converged store (k-means++ seeds, zero Lloyd steps -- 26 raw training rows
     per well, no fitting at all) scores the HIGHEST R1 (36), and every step of honest
     k-means fitting makes the headline worse. R1 is substantially measuring how much
     unconverged noise is left in the codebook, not composition."
It measured that on ONE cell. This tests it on all 49 DEV corners against the same rival.
⛔ Read-only over the DEV ledger; B is not touched.
"""
import json

import numpy as np
from scipy import stats

import comp
import rivals

rows = rivals.ledger_rows()


def col(tag, arm="clu_linear"):
    return {r["corner_idx"]: r["headline"]["tuples_valid_novel"]
            for r in rows if r.get("tag") == tag and r["arm"] == arm}


mine = col("ref")
knn = col("ref", "knn_proto")
raw0 = col("ctor_mine_raw0")
gp = col("ctor_genpart_25")
ks = sorted(set(mine) & set(knn) & set(raw0) & set(gp))
A = np.array([mine[k] for k in ks], float)
Z = np.array([raw0[k] for k in ks], float)
R = np.array([knn[k] for k in ks], float)
G = np.array([gp[k] for k in ks], float)


def paired(x, y, label):
    d = x - y
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    lo, hi = d.mean() - t * se, d.mean() + t * se
    v = comp.r1_verdict({"delta": float(d.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
    print(f"{label:54s} n={n:2d} delta={d.mean():+7.3f} CI95[{lo:+7.3f},{hi:+7.3f}] "
          f"W/T/L={int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())}  {v}")
    return {"n": n, "delta": float(d.mean()), "ci95": [lo, hi], "R1": v,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum()), "sd": float(d.std(ddof=1))}


print(f"n={len(ks)} corners")
print(f"mean/72   mine (Lloyd 25) {A.mean():.2f}   raw0 (no Lloyd) {Z.mean():.2f}   "
      f"genpart (Lloyd 25) {G.mean():.2f}   knn_proto {R.mean():.2f}\n")
out = {"n": len(ks), "corners": ks,
       "means": {"mine_lloyd25": float(A.mean()), "mine_raw0": float(Z.mean()),
                 "genpart_25": float(G.mean()), "knn_proto": float(R.mean())},
       "B_single_cell": {"mine": 42, "genpart": 25, "spoke_raw0": 36, "knn": 12}}
out["raw0_vs_mine"] = paired(Z, A, "raw0 (no Lloyd) vs mine (Lloyd 25)  <- THE CRITICISM")
out["raw0_vs_knn"] = paired(Z, R, "raw0 vs knn_proto  (THE BAR)")
out["mine_vs_knn"] = paired(A, R, "mine vs knn_proto  (THE BAR)")
out["genpart_vs_knn"] = paired(G, R, "genpart vs knn_proto  (THE BAR)")
with open(str(rivals.OUT / "dev" / "raw0_test.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\n-> {rivals.OUT / 'dev' / 'raw0_test.json'}")
