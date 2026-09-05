"""Re-derive every number in the c2w8p3-phi-geometry report from PHI-GEOMETRY.json.

Usage: python render.py [path/to/PHI-GEOMETRY.json]
"""
import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/PHI-GEOMETRY.json"
R = json.load(open(path))


def ms(vals):
    a = np.asarray(vals, float)
    if a.size == 0:
        return float("nan"), float("nan")
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return float(a.mean()), float(se)


rows = R["geometry_rows"]
arms = sorted({r["arm"] for r in rows})
dims = sorted({r["addr_dim"] for r in rows})
ns = sorted({r["n_keys"] for r in rows})
n_pri = R["verdict"]["n_keys_primary"]

print("=" * 100)
print("SUBSTRATE:", R["substrate"], "| seeds:", R["seeds"],
      "| stream fingerprints:", R["stream_fingerprints"])
print("geometry_go =", R["geometry_go"],
      "| d_favoured_by_geometry =", R["d_favoured_by_geometry"],
      "| d_recommended_operational =", R["d_recommended_operational"])
print("store inert by d:", R["store_inert_by_d"])
print("launder reads projected phi:", R["mapping"]["launder_reads_projected_phi"],
      "| bit-identical to store addresses:",
      R["mapping"]["launder_bit_identical_to_store"])
print("=" * 100)

print("\n### 1. sigma_q / spacing  (mean +/- SE over seeds), n_keys =", n_pri)
hdr = f"{'arm':34s}" + "".join(f"  d={d:<3d}(atoms {a:>7d})"
                               for d, a in [(d, next(r['n_atoms'] for r in rows
                                                     if r['addr_dim'] == d))
                                            for d in dims])
print(hdr)
for arm in arms:
    line = f"{arm:34s}"
    for d in dims:
        v = [r["sigma_q_over_spacing"] for r in rows
             if r["arm"] == arm and r["addr_dim"] == d and r["n_keys"] == n_pri]
        m, s = ms(v)
        line += f"   {m:8.3f}+-{s:6.3f}"
    print(line)

print("\n### 2. median-NN key spacing (unit-r95 address ball), n_keys =", n_pri)
for arm in arms:
    line = f"{arm:34s}"
    for d in dims:
        v = [r["median_nn_spacing"] for r in rows
             if r["arm"] == arm and r["addr_dim"] == d and r["n_keys"] == n_pri]
        m, s = ms(v)
        line += f"   {m:8.4f}+-{s:6.4f}"
    print(line)

print("\n### 3. spacing / uniform-ball spacing   (1.0 = fills the ball uniformly)")
for arm in arms:
    line = f"{arm:34s}"
    for d in dims:
        v = [r["spacing_over_uniform_ball"] for r in rows
             if r["arm"] == arm and r["addr_dim"] == d and r["n_keys"] == n_pri]
        m, s = ms(v)
        line += f"   {m:8.3f}+-{s:6.3f}"
    print(line)

print("\n### 4. participation ratio / d   (effective fraction of address dims used)")
for arm in arms:
    line = f"{arm:34s}"
    for d in dims:
        v = [r["participation_fraction"] for r in rows
             if r["arm"] == arm and r["addr_dim"] == d and r["n_keys"] == n_pri]
        m, s = ms(v)
        line += f"   {m:8.3f}+-{s:6.3f}"
    print(line)

print("\n### 5. d_safe / spacing (rig pricing), n_keys =", n_pri)
for arm in arms:
    line = f"{arm:34s}"
    for d in dims:
        v = [r["d_safe_over_spacing"] for r in rows
             if r["arm"] == arm and r["addr_dim"] == d and r["n_keys"] == n_pri]
        m, s = ms(v)
        line += f"   {m:8.3f}+-{s:6.3f}"
    print(line)

print("\n### 6. sigma_q/spacing vs n (the (n,d) geometry check), strong arm")
strong = R["verdict"]["strong_arm"]
ref = R["verdict"]["reference_arm"]
for arm in (strong, ref):
    print(f"  {arm}")
    for d in dims:
        line = f"    d={d:<3d}"
        for n in ns:
            v = [r["sigma_q_over_spacing"] for r in rows
                 if r["arm"] == arm and r["addr_dim"] == d and r["n_keys"] == n]
            m, _ = ms(v)
            line += f"   n={n}: {m:7.3f}"
        print(line)

print("\n### 7. THE REGISTERED READING (mechanical)")
print("  ", R["verdict"]["rule"])
for d in sorted(R["verdict"]["by_d"], key=int):
    b = R["verdict"]["by_d"][d]
    print(f"  d={d:>3s} atoms={b['n_atoms']:>8d}  strong={np.round(b['sigma_q_over_spacing_strong'],4).tolist()}"
          f"  ref={np.round(b['sigma_q_over_spacing_reference'],4).tolist()}")
    print(f"        paired improvement {b['paired_improvement_mean']:+.4f} "
          f"+- {b['paired_improvement_se']:.4f} SE, {b['n_seeds_positive']}/"
          f"{b['n_seeds']} seeds positive, ratio strong/ref="
          f"{b['ratio_strong_over_reference']:.4f}  ->  GO={b['go']}")

print("\n### 8. RIDER (a) — the (d, atom-budget) depth cells")
for dp in R["depth_probe"]:
    print(f"  d={dp['addr_dim']:<3d} n_atoms={dp['n_atoms']:>7d} "
          f"(honoured={dp['atom_budget_honoured']}) writes={dp['n_writes']} "
          f"admitted={dp['n_admitted']} depth_median={dp['depth_median']:.4g} "
          f"depth_max={dp['depth_max']} INERT={dp['inert']} "
          f"self_probe_strict={dp['self_probe_strict']:.3f} [{dp['wall_s']:.0f}s]")
    print(f"        depths={np.round(dp['fitted_depths'], 6).tolist()}")

print("\n### 9. RIDER (b) — d_safe pricing + the MEASURED refusal rate")
seen = set()
for r in R["d_safe_rider"]:
    if r["arm"] not in (strong, ref):
        continue
    k = (r["arm"], r["addr_dim"])
    agg = [x for x in R["d_safe_rider"]
           if x["arm"] == r["arm"] and x["addr_dim"] == r["addr_dim"]]
    if k in seen:
        continue
    seen.add(k)
    m_rig, _ = ms([x["refusal_rig"]["refusal_rate"] for x in agg])
    m_new, _ = ms([x["refusal_repriced"]["refusal_rate"] for x in agg])
    m_r1, _ = ms([x["d_safe_rig_over_population_spacing"] for x in agg])
    print(f"  {r['arm']:30s} d={r['addr_dim']:<3d} "
          f"d_safe_rig/pop_spacing={m_r1:.3f}  refusal(rig)={m_rig:.3f}  "
          f"refusal(repriced @0.88)={m_new:.3f}")

print("\n### 10. BYTE LEDGER (floats; the map rides on EVERY arm incl. the launder)")
byd = defaultdict(list)
for L in R["byte_ledger"]:
    byd[(L["arm"], L["addr_dim"])].append(L)
for (arm, d), v in sorted(byd.items()):
    if arm not in (strong, ref, R["verdict"]["strong_arm"]):
        continue
    L = v[0]
    print(f"  {arm:30s} d={d:<3d} encoder={L['encoder_param_floats']:>7d} "
          f"map={L['projection_param_floats']:>6d} "
          f"phi_total={L['phi_param_floats_total']:>7d} "
          f"launder_total={L['knn_launder_total_floats']:>7d}")

print("\n### 11. JOINT DIAL (d, atom budget)")
for r in R["joint_dial_d_atom_budget"]:
    print(f"  d={r['addr_dim']:>4d}  n_atoms={r['n_atoms']:>12d}")

print("\n### 12. FLAGS")
print(json.dumps(R["flags"], indent=2))
print("\nwall_s =", R["wall_s"])
