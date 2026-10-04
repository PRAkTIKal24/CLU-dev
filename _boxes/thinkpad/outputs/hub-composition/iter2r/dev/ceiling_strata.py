"""⭐ Is B's +30 explained by B being an EASY cell? Stratify the 49 DEV corners by ceiling.
⛔ Read-only over the DEV ledger. ⛔ B is not touched (its measured ceiling, 67, is quoted
from the scored run that already happened)."""
import json
import numpy as np
from scipy import stats
import comp, rivals
rows = [r for r in rivals.ledger_rows() if r.get("tag") == "ref"]
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["corner_idx"]] = r["headline"]["tuples_valid_novel"]
ks = sorted(set(by["ceiling"]) & set(by["clu_linear"]) & set(by["knn_proto"]))
C = np.array([by["ceiling"][k] for k in ks], float)
A = np.array([by["clu_linear"][k] for k in ks], float)
R = np.array([by["knn_proto"][k] for k in ks], float)
out = {"n": len(ks), "B_measured_ceiling": 67, "B_measured_delta": 30.0,
       "B_clu": 42, "B_knn": 12, "strata": []}
print(f"n={len(ks)} corners; ceiling {C.mean():.2f} +- {C.std(ddof=1):.2f}; "
      f"B's measured ceiling = 67 (percentile "
      f"{100*(C<67).mean():.1f} of the DEV distribution)\n")
print(f"{'stratum':22s} {'n':>3s} {'ceiling':>8s} {'clu':>7s} {'knn':>7s} {'delta':>7s} "
      f"{'CI95':>19s}  R1")
for name, m in (("ALL", C > -1), ("ceiling >= 55", C >= 55), ("ceiling >= 60", C >= 60),
                ("ceiling >= 65", C >= 65), ("ceiling < 55", C < 55)):
    if m.sum() < 3:
        print(f"{name:22s} {int(m.sum()):3d}  (too few)"); continue
    d = A[m] - R[m]; n = int(m.sum())
    se = float(d.std(ddof=1)/np.sqrt(n)); t = float(stats.t.ppf(0.975, n-1))
    lo, hi = d.mean()-t*se, d.mean()+t*se
    v = comp.r1_verdict({"delta": float(d.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
    out["strata"].append({"stratum": name, "n": n, "ceiling_mean": float(C[m].mean()),
                          "clu": float(A[m].mean()), "knn": float(R[m].mean()),
                          "delta": float(d.mean()), "ci95": [lo, hi], "R1": v,
                          "delta_sd": float(d.std(ddof=1))})
    print(f"{name:22s} {n:3d} {C[m].mean():8.2f} {A[m].mean():7.2f} {R[m].mean():7.2f} "
          f"{d.mean():+7.2f}  [{lo:+7.2f},{hi:+7.2f}]  {v}")
# where does B's delta sit in the DEV delta distribution?
d = A - R
z = (30.0 - d.mean())/d.std(ddof=1)
out["B_delta_z_in_DEV_distribution"] = float(z)
out["B_delta_percentile_in_DEV"] = float(100*(d < 30).mean())
out["B_clu_percentile_in_DEV"] = float(100*(A < 42).mean())
print(f"\nB's Δ = +30 sits at z = {z:+.2f} of the DEV Δ distribution "
      f"(mean {d.mean():+.2f}, sd {d.std(ddof=1):.2f}); "
      f"{100*(d<30).mean():.1f}% of DEV corners have a smaller Δ.")
print(f"B's clu_linear = 42 is above {100*(A<42).mean():.1f}% of DEV corners "
      f"(DEV mean {A.mean():.2f}).")
print(f"B's knn_proto = 12 is above {100*(R<12).mean():.1f}% of DEV corners "
      f"(DEV mean {R.mean():.2f}).")
json.dump(out, open(str(rivals.OUT / "dev" / "ceiling_strata.json"), "w"), indent=2)
