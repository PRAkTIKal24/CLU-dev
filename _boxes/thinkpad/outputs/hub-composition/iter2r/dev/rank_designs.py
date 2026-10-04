"""Rank the C04 head designs against knn_proto on the SAME corners. ⭐ Selection only."""
import sys
import numpy as np
import comp, rivals

TAG = sys.argv[1] if len(sys.argv) > 1 else "head1"
rows = [r for r in rivals.ledger_rows() if r.get("tag") == TAG]
ref = {r["corner_idx"]: r["headline"]["tuples_valid_novel"]
       for r in rivals.ledger_rows() if r.get("tag") == "ref" and r["arm"] == "knn_proto"}
obj = {r["corner_idx"]: r["headline"]["tuples_valid_novel"]
       for r in rivals.ledger_rows() if r.get("tag") == "ref" and r["arm"] == "clu_linear"}
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["corner_idx"]] = r["headline"]["tuples_valid_novel"]
ks = sorted(set.intersection(*[set(v) for v in by.values()]) & set(ref))
print(f"{len(ks)} corners: {ks}\n")
out = []
for arm, d in by.items():
    x = np.array([d[k] for k in ks], float)
    y = np.array([ref[k] for k in ks], float)
    p = comp.paired(x, y)
    out.append((p["delta"], x.mean(), p["ci95"][0], p["ci95"][1], p["wins"], p["ties"],
                int((x - y < 0).sum()), arm))
o = np.array([obj[k] for k in ks], float); yy = np.array([ref[k] for k in ks], float)
po = comp.paired(o, yy)
print(f"{'REFERENCE OBJECT (C02)':38s} mean={o.mean():6.2f}  Δ={po['delta']:+7.3f}  "
      f"CI[{po['ci95'][0]:+7.3f},{po['ci95'][1]:+7.3f}]  W/T/L={po['wins']}/{po['ties']}/"
      f"{int((o-yy<0).sum())}")
print(f"{'knn_proto':38s} mean={yy.mean():6.2f}\n")
print(f"{'design|rank|ridge':38s} {'mean':>6s}  {'Δ':>7s}  {'CI95':>17s}  {'W/T/L':>8s}  R1")
for dlt, mn, lo, hi, w, t, l, arm in sorted(out, reverse=True)[:22]:
    v = comp.r1_verdict({"delta": dlt, "ci95": [lo, hi]}, m=0.020, scale=72.0)
    print(f"{arm:38s} {mn:6.2f}  {dlt:+7.3f}  [{lo:+7.3f},{hi:+7.3f}]  {w:2d}/{t}/{l:<3d} {v}")
