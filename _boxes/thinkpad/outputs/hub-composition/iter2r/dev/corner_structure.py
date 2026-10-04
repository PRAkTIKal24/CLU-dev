"""⭐ B is Montero's own cell: hue_lo = 0.8 (the TOP of the hue grid), scale_hi, orient_hi.
My 49 DEV corners span hue_lo in {0.0, 0.2, 0.4, 0.6} only -- B is one step BEYOND the
furthest DEV corner, at the grid EDGE. Does Delta trend with hue_lo, or with the
(scale_hi, orient_hi) signature that B carries? ⛔ Read-only over the DEV ledger."""
import json
import numpy as np
from scipy import stats
import comp, rivals
rows = [r for r in rivals.ledger_rows() if r.get("tag") == "ref"]
by, cor = {}, {}
for r in rows:
    by.setdefault(r["arm"], {})[r["corner_idx"]] = r["headline"]["tuples_valid_novel"]
    cor[r["corner_idx"]] = r["corner"]
ks = sorted(set(by["clu_linear"]) & set(by["knn_proto"]))
A = np.array([by["clu_linear"][k] for k in ks], float)
R = np.array([by["knn_proto"][k] for k in ks], float)
d = A - R
H = np.array([cor[k]["hue_lo"] for k in ks], float)
S = np.array([bool(cor[k]["scale_hi"]) for k in ks])
O = np.array([bool(cor[k]["orient_hi"]) for k in ks])
out = {"n": len(ks), "B": {"hue_lo": 0.8, "scale_hi": True, "orient_hi": True,
                           "clu": 42, "knn": 12, "delta": 30.0}, "by_hue": [], "by_sig": []}
def ci(x):
    n = len(x)
    if n < 3: return (float(np.mean(x)), float("nan"), float("nan"), n)
    se = float(np.std(x, ddof=1)/np.sqrt(n)); t = float(stats.t.ppf(0.975, n-1))
    return (float(np.mean(x)), float(np.mean(x))-t*se, float(np.mean(x))+t*se, n)
print(f"{'hue_lo':>7s} {'n':>3s} {'clu':>7s} {'knn':>7s} {'delta':>7s} {'CI95':>19s}")
for h in sorted(set(H.tolist())):
    m = H == h
    mu, lo, hi, n = ci(d[m])
    out["by_hue"].append({"hue_lo": h, "n": n, "clu": float(A[m].mean()),
                          "knn": float(R[m].mean()), "delta": mu, "ci95": [lo, hi]})
    print(f"{h:7.1f} {n:3d} {A[m].mean():7.2f} {R[m].mean():7.2f} {mu:+7.2f} "
          f"[{lo:+7.2f},{hi:+7.2f}]")
b, a0 = np.polyfit(H, d, 1)
r_ = float(np.corrcoef(H, d)[0, 1])
out["delta_vs_hue_lo"] = {"slope": float(b), "intercept": float(a0), "r": r_,
                          "extrapolated_to_hue_lo_0.8": float(a0 + b*0.8)}
print(f"\nDelta vs hue_lo: slope {b:+.3f}/unit, r = {r_:+.3f}; "
      f"linear extrapolation to B's hue_lo = 0.8 gives Delta = {a0 + b*0.8:+.2f} "
      f"(measured on B: +30.00)")
print(f"\n{'scale_hi':>9s} {'orient_hi':>10s} {'n':>3s} {'clu':>7s} {'knn':>7s} {'delta':>7s}")
for s_ in (True, False):
    for o_ in (True, False):
        m = (S == s_) & (O == o_)
        if m.sum() == 0: continue
        mu, lo, hi, n = ci(d[m])
        out["by_sig"].append({"scale_hi": s_, "orient_hi": o_, "n": n,
                              "clu": float(A[m].mean()), "knn": float(R[m].mean()),
                              "delta": mu, "ci95": [lo, hi]})
        star = "  <- B's signature" if (s_ and o_) else ""
        print(f"{str(s_):>9s} {str(o_):>10s} {n:3d} {A[m].mean():7.2f} {R[m].mean():7.2f} "
              f"{mu:+7.2f}{star}")
json.dump(out, open(str(rivals.OUT / "dev" / "corner_structure.json"), "w"), indent=2)
