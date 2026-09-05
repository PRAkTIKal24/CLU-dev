"""Derived quantities for the report — paired arm contrasts + the D2a correlation."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/CAPTURE-STRONG-PHI.json"))
A = d["arms"]
order = ["simclr", "randconv", "pca"]
get = lambda a, f: np.array([f(c) for c in A[a]["cells"]], float)

A1 = {a: get(a, lambda c: c["g_addr"]["A1"]["correct_basin_rate"]) for a in order}
A2 = {a: get(a, lambda c: c["g_addr"]["A2"]["never_addressed_frac"]) for a in order}
A3a = {a: get(a, lambda c: c["g_addr"]["A3"]["A3a_cue_margin"]) for a in order}
A3b = {a: get(a, lambda c: c["g_addr"]["A3"]["A3b_stream_margin"]) for a in order}
DR = {a: np.array([lg["G_DRIFT"]["ratio"] for lg in A[a]["gate"]["legs_by_seed"]], float)
      for a in order}
ZB = {a: get(a, lambda c: c["g_addr"]["A1"]["n_items_with_zero_basin"]) for a in order}
CAP = {a: np.array([lg["G_CAP"]["frac_capture_positive"]
                    for lg in A[a]["gate"]["legs_by_seed"]], float) for a in order}
AG = {a: get(a, lambda c: c["d2a"]["agreement_rate"]) for a in order}
NQ = int(A["randconv"]["cells"][0]["g_addr"]["n_queries"])

print(f"n_queries per cell = {NQ}; A1 threshold = "
      f"{A['randconv']['cells'][0]['g_addr']['A1']['threshold']:.4f} "
      f"= {A['randconv']['cells'][0]['g_addr']['A1']['threshold']*NQ:.0f}/{NQ}\n")

print(f"{'arm':<10} {'A1 mean±SE':<20} {'A1 counts/128':<18} {'A2 mean':<10} "
      f"{'A3a mean±SE':<20} {'A3b mean±SE':<20} {'zero-basin items/16':<20} {'G-CAP frac'}")
for a in order:
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
    print(f"{a:<10} {np.mean(A1[a]):.4f}±{se(A1[a]):.4f}      "
          f"{str([int(round(x*NQ)) for x in A1[a]]):<18} {np.mean(A2[a]):.4f}     "
          f"{np.mean(A3a[a]):+.4f}±{se(A3a[a]):.4f}     "
          f"{np.mean(A3b[a]):+.4f}±{se(A3b[a]):.4f}     "
          f"{str([int(x) for x in ZB[a]]):<20} {np.round(CAP[a],3).tolist()}")

print("\nPAIRED arm contrasts on A1 (same seed, same stream fingerprint):")
for x, y in (("simclr", "randconv"), ("simclr", "pca"), ("randconv", "pca")):
    dlt = A1[x] - A1[y]
    se = float(np.std(dlt, ddof=1) / np.sqrt(len(dlt)))
    print(f"  {x:<9} - {y:<9} = {np.mean(dlt):+.4f} ± {se:.4f} (2SE {2*se:.4f}) "
          f"per seed {np.round(dlt,4).tolist()} "
          f"| {int(np.sum(dlt>0))}/3 seeds {x} better")

print("\n⚠ D2a: across ALL 9 cells, A1 vs G-DRIFT ratio and vs agreement")
a1 = np.concatenate([A1[a] for a in order])
dr = np.concatenate([DR[a] for a in order])
ag = np.concatenate([AG[a] for a in order])
rk = lambda v: np.argsort(np.argsort(v)).astype(float)
print(f"  Pearson  r(A1, drift_ratio)     = {np.corrcoef(a1, dr)[0,1]:+.4f}")
print(f"  Spearman rho(A1, drift_ratio)   = {np.corrcoef(rk(a1), rk(dr))[0,1]:+.4f}")
print(f"  Pearson  r(A1, agreement)       = {np.corrcoef(a1, ag)[0,1]:+.4f}")
print(f"  Spearman rho(A1, agreement)     = {np.corrcoef(rk(a1), rk(ag))[0,1]:+.4f}")
o = np.argsort(-a1)
lbl = [f"{a}:{s}" for a in order for s in A[a]["seeds"]]
print("  cells by A1 (best first): " + ", ".join(
    f"{lbl[i]} A1={a1[i]:.4f} drift={dr[i]:.4f} agree={ag[i]:.4f}" for i in o))
