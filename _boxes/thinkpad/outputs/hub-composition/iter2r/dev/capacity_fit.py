"""⭐ THE CAPACITY CURVE FOR THE **LINEAR** READ. ⛔ Read-only over the DEV ledger.

R2 requires the capacity law reported beside the headline. The inherited law
`rel-err = e^1.346 · bytes^-0.1651` (R² 0.9960) was fitted to the ARGMAX quantiser's
reconstruction residual. This fits the same two quantities for THIS read, on the same
corners, so the two can be compared honestly instead of one standing in for the other.
⚠ Legs above M = 26 are DECLARED OVER BUDGET and are never matched-bytes rows.
"""
import json

import numpy as np
from scipy import stats

import rivals

rows = [r for r in rivals.ledger_rows() if r.get("tag") == "cap"]
M_GRID = [4, 8, 13, 26, 52, 104]
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["corner_idx"]] = r
ceil = {k: v["headline"]["tuples_valid_novel"] for k, v in by.get("ceiling", {}).items()}
ks = sorted(set.intersection(*[set(by[f"M{m}"]) for m in M_GRID if f"M{m}" in by]))
print(f"corners with the full curve: n = {len(ks)}  {ks}")
print(f"decoder ceiling on these corners: {np.mean([ceil[k] for k in ks]):.2f} "
      f"+- {np.std([ceil[k] for k in ks], ddof=1):.2f}\n")

out = {"n_corners": len(ks), "corners": ks,
       "ceiling_mean": float(np.mean([ceil[k] for k in ks])),
       "ceiling_sd": float(np.std([ceil[k] for k in ks], ddof=1)), "curve": []}
hdr = (f"{'M':>5s} {'bytes':>10s} {'xknn':>6s} {'tuples mean':>12s} {'sd':>6s} "
       f"{'CI95':>17s} {'rel_err':>8s}")
print(hdr)
print("-" * len(hdr))
B, T, E = [], [], []
for m in M_GRID:
    key = f"M{m}"
    if key not in by:
        continue
    t = np.array([by[key][k]["headline"]["tuples_valid_novel"] for k in ks], float)
    e = np.array([by[key][k]["extra"]["linear_read_rel_err_on_the_held_out_corner"]
                  for k in ks], float)
    b = int(by[key][ks[0]]["bytes"])
    xk = float(by[key][ks[0]]["bytes_vs_knn_proto"])
    n = t.size
    se = float(t.std(ddof=1) / np.sqrt(n))
    tt = float(stats.t.ppf(0.975, n - 1))
    flag = "  ⚠ OVER BUDGET" if xk > 1.0 else ""
    print(f"{m:5d} {b:10,d} {xk:6.3f} {t.mean():12.2f} {t.std(ddof=1):6.2f} "
          f"[{t.mean() - tt * se:7.2f},{t.mean() + tt * se:7.2f}] {e.mean():8.4f}{flag}")
    out["curve"].append({"M": m, "bytes": b, "x_knn_proto": xk,
                         "tuples_mean": float(t.mean()), "tuples_sd": float(t.std(ddof=1)),
                         "ci95": [t.mean() - tt * se, t.mean() + tt * se],
                         "rel_err_mean": float(e.mean()),
                         "over_budget": bool(xk > 1.0),
                         "per_corner": {int(k): float(by[key][k]["headline"]
                                                     ["tuples_valid_novel"]) for k in ks}})
    B.append(b)
    T.append(t.mean())
    E.append(e.mean())

B, T, E = np.array(B, float), np.array(T, float), np.array(E, float)
# the inherited law's form: rel-err = A * bytes^p
sl, ic = np.polyfit(np.log(B), np.log(E), 1)
r2 = float(np.corrcoef(np.log(B), np.log(E))[0, 1] ** 2)
out["rel_err_power_law_for_the_LINEAR_read"] = {
    "form": "rel_err = exp(intercept) * bytes**slope", "slope": float(sl),
    "intercept": float(ic), "R2": r2,
    "inherited_argmax_law": {"intercept": 1.346, "slope": -0.1651, "R2": 0.9960}}
print(f"\nrel-err power law for the LINEAR read:  rel_err = e^{ic:.3f} · bytes^{sl:.4f}  "
      f"(R² = {r2:.4f})")
print(f"  inherited ARGMAX law, for comparison:  rel_err = e^1.346 · bytes^-0.1651  "
      f"(R² = 0.9960)")

imax = int(np.argmax(T))
out["argmax_of_the_tuple_curve"] = {"M": M_GRID[imax], "bytes": int(B[imax]),
                                    "tuples": float(T[imax]),
                                    "x_knn_proto": float(B[imax] / 475136)}
print(f"\n⭐ The TUPLE curve peaks at M = {M_GRID[imax]} "
      f"({int(B[imax]):,} B = {B[imax] / 475136:.3f} x knn_proto) "
      f"at {T[imax]:.2f} tuples.")
print(f"⭐ Monotone in bytes? {'YES' if np.all(np.diff(T) >= 0) else 'NO — it turns over'}")
out["tuple_curve_monotone_in_bytes"] = bool(np.all(np.diff(T) >= 0))
with open(str(rivals.OUT / "dev" / "capacity_linear_read.json"), "w",
          encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\n-> {rivals.OUT / 'dev' / 'capacity_linear_read.json'}")
