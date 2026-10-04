"""⭐ Score the PRE-REGISTERED fresh-corner predictions F1–F6 (LOG 2026-09-17T16:20Z).
⛔ Read-only over the DEV ledger. ⛔ B is not touched.
"""
import json

import numpy as np
from scipy import stats

import comp
import rivals

rows = [r for r in rivals.ledger_rows() if r.get("tag") == "fresh"]
by, cor = {}, {}
for r in rows:
    by.setdefault(r["arm"], {})[r["corner_idx"]] = r["headline"]["tuples_valid_novel"]
    cor[r["corner_idx"]] = r["corner"]
ks = sorted(set(by.get("clu_linear", {})) & set(by.get("knn_proto", {})))
A = np.array([by["clu_linear"][k] for k in ks], float)
R = np.array([by["knn_proto"][k] for k in ks], float)
Z = np.array([by["store_removed_zero"][k] for k in ks], float)
C = np.array([by["ceiling"][k] for k in ks], float)
S = np.array([bool(cor[k]["scale_hi"]) for k in ks])
O = np.array([bool(cor[k]["orient_hi"]) for k in ks])
d = A - R


def paired(x, label):
    n = x.size
    if n < 3:
        print(f"{label:52s} n={n} — too few")
        return {"n": n}
    se = float(x.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    lo, hi = x.mean() - t * se, x.mean() + t * se
    v = comp.r1_verdict({"delta": float(x.mean()), "ci95": [lo, hi]}, m=0.020, scale=72.0)
    print(f"{label:52s} n={n:2d} delta={x.mean():+7.3f} CI95[{lo:+7.3f},{hi:+7.3f}] "
          f"W/T/L={int((x > 0).sum())}/{int((x == 0).sum())}/{int((x < 0).sum())}  {v}")
    return {"n": n, "delta": float(x.mean()), "ci95": [lo, hi], "R1": v,
            "sd": float(x.std(ddof=1))}


print(f"FRESH corners scored: n = {len(ks)}   (hue_lo in {{0.1,0.3,0.5,0.7}}, "
      f"disjoint from the declared family; B's 0.8 excluded)")
print(f"means /72   clu_linear {A.mean():.2f}   knn_proto {R.mean():.2f}   "
      f"ceiling {C.mean():.2f}   store_removed_zero {Z.mean():.2f}\n")
out = {"n": len(ks), "corners": ks,
       "means": {"clu_linear": float(A.mean()), "knn_proto": float(R.mean()),
                 "ceiling": float(C.mean()), "store_removed_zero": float(Z.mean())}}
out["F1_all_fresh"] = paired(d, "F1  Δ over ALL fresh corners   (predicted +1.4, TIE)")
sig = S & O
out["F2_signature"] = paired(d[sig], "F2  Δ on scale_hi & orient_hi  (predicted +8.5)")
out["F3_others"] = paired(d[~sig], "F3  Δ on the other three sigs  (predicted -3)")
print()
for s_ in (True, False):
    for o_ in (True, False):
        m = (S == s_) & (O == o_)
        if m.sum():
            star = "  <- B's signature" if (s_ and o_) else ""
            print(f"      scale_hi={str(s_):5s} orient_hi={str(o_):5s} n={int(m.sum()):2d} "
                  f"clu={A[m].mean():6.2f} knn={R[m].mean():6.2f} "
                  f"delta={d[m].mean():+7.2f}{star}")
f2 = out["F2_signature"].get("delta", float("nan"))
f3 = out["F3_others"].get("delta", float("nan"))
out["F4_signature_effect_replicates"] = bool(f2 > f3)
out["F4_gap"] = float(f2 - f3)
print(f"\n⭐ F4  (F2 > F3?)  {f2:+.2f} vs {f3:+.2f}  ->  "
      f"{'YES — the sign replicates' if f2 > f3 else 'NO — the hypothesis is DEAD'}"
      f"   (gap {f2 - f3:+.2f})")
out["F5_store_removed_zero_all_zero"] = bool((Z == 0).all())
print(f"⭐ F5  store_removed_zero == 0 on every fresh corner:  "
      f"{'YES' if (Z == 0).all() else 'NO — ' + str(int((Z != 0).sum())) + ' non-zero'}")
out["F6_identity_asserted_by_comp_headline"] = True
print("⭐ F6  the identity was asserted per render by comp.headline (it raises otherwise); "
      "no corner raised.")
with open(str(rivals.OUT / "dev" / "fresh_test.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\n-> {rivals.OUT / 'dev' / 'fresh_test.json'}")
