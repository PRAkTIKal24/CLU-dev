"""⭐ The STORE-SEED spread: three k-means initialisations of the same object, same corners.

⛔ Read-only over the DEV ledger; B is not touched.
⚠ This is **store-init** seed spread, ⛔ NOT bridge-seed spread. Bridge seeds 1 and 2 are
cache-blocked on this box (`scope="used"`; covering the cell means encoding ~32,200 new rows,
which is forbidden). ⛔ Nothing below may be quoted as bridge-seed spread.
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


S = {0: col("ref"), 1: col("storeseed1"), 2: col("storeseed2")}
S = {k: v for k, v in S.items() if v}
knn = col("ref", "knn_proto")
ks = sorted(set.intersection(*[set(v) for v in S.values()]) & set(knn))
R = np.array([knn[k] for k in ks], float)
M = np.stack([np.array([S[s][k] for k in ks], float) for s in sorted(S)])   # (S, n)
print(f"store seeds present: {sorted(S)}   n corners = {len(ks)}\n")
out = {"store_seeds": sorted(S), "n_corners": len(ks), "corners": ks,
       "WARNING": "store-INIT seed spread only; bridge seeds 1/2 are cache-blocked and "
                  "this may NOT be quoted as bridge-seed spread",
       "per_seed": {}, "variance_decomposition": {}}


def paired(x, y, label):
    d = x - y
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    lo, hi = d.mean() - t * se, d.mean() + t * se
    v = comp.r1_verdict({"delta": float(d.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
    print(f"{label:44s} mean={x.mean():6.2f}  delta={d.mean():+7.3f} "
          f"CI95[{lo:+7.3f},{hi:+7.3f}]  W/T/L={int((d > 0).sum())}/"
          f"{int((d == 0).sum())}/{int((d < 0).sum())}  {v}")
    return {"mean": float(x.mean()), "delta": float(d.mean()), "ci95": [lo, hi], "R1": v,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "losses": int((d < 0).sum())}


for i, s in enumerate(sorted(S)):
    out["per_seed"][str(s)] = paired(M[i], R, f"store seed {s} vs knn_proto (THE BAR)")
print(f"\n{'knn_proto (deterministic, no seed)':44s} mean={R.mean():6.2f}")

# variance decomposition on the PAIRED difference d = arm - rival
Dm = M - R[None]                                       # (S, n)
per_corner_mean = Dm.mean(0)                           # (n,)
within = float(Dm.var(0, ddof=1).mean())               # store-seed variance, avg over corners
between = float(per_corner_mean.var(ddof=1))           # corner variance of the seed-mean
tot = float(Dm.reshape(-1).var(ddof=1))
out["variance_decomposition"] = {
    "delta_mean_over_seeds_and_corners": float(Dm.mean()),
    "within_corner_store_seed_variance": within,
    "within_corner_store_seed_sd": float(np.sqrt(within)),
    "between_corner_variance_of_the_seed_mean": between,
    "between_corner_sd": float(np.sqrt(between)),
    "total_variance": tot,
    "fraction_of_variance_that_is_the_CORNER": float(between / max(between + within, 1e-30)),
}
print(f"\nVariance of the paired difference Δ = arm − knn_proto, decomposed:")
print(f"  store-seed (within corner)  sd = {np.sqrt(within):6.2f} tuples")
print(f"  corner     (between corner) sd = {np.sqrt(between):6.2f} tuples")
print(f"  ⇒ the CORNER accounts for "
      f"{100 * between / max(between + within, 1e-30):.1f}% of the variance in Δ.")

# the 3-store-seed averaged estimate
d3 = per_corner_mean
n = d3.size
se = float(d3.std(ddof=1) / np.sqrt(n))
t = float(stats.t.ppf(0.975, n - 1))
lo, hi = d3.mean() - t * se, d3.mean() + t * se
v = comp.r1_verdict({"delta": float(d3.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
out["three_store_seed_estimate"] = {"n_corners": n, "delta": float(d3.mean()),
                                    "ci95": [lo, hi], "R1": v}
print(f"\n⭐ 3-store-seed averaged estimate over {n} corners: "
      f"Δ = {d3.mean():+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  -> {v.upper()}")
with open(str(rivals.OUT / "dev" / "store_seed_spread.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"-> {rivals.OUT / 'dev' / 'store_seed_spread.json'}")
