"""Re-derive every number in cl-encoder.md from the shipped JSON (w25 precedent)."""

import json
import os

import numpy as np

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    with open(os.path.join(S, name)) as f:
        return json.load(f)


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def ms(vals):
    vals = [float(v) for v in vals]
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{np.mean(vals):.4f} ± {sd:.4f}", np.mean(vals)


print("=" * 78)
print("GATE = ring-buffer kNN-in-φ, Class-IL end-of-stream ACC, Split-CIFAR-10")
print("reduced protocol, 200-item matched memory, task1_only φ.  Gate value 0.35")
print("=" * 78)

print("\n-- Stage A: the cheap arms (seed 0, default 2027-image pool) --")
rows = load("stageA.json") + load("stageA_control.json")
for r in sorted(rows, key=lambda r: (r["arm"], r["phi_dim"], r["regime"])):
    print(f"  {r['arm']:>4}-{r['phi_dim']:<4} {r['regime']:<15} "
          f"ring={r['acc_ring']:.4f}  full-stream(5k)={r['acc_full_stream']:.4f}")

print("\n-- Stage B: the conv arms at the OLD defaults (seed 0, 2000 steps, d=64) --")
for r in load("stageB_seed0.json"):
    print(f"  {r['arm']:>8}-{r['phi_dim']:<4} ring={r['acc_ring']:.4f}  "
          f"full={r['acc_full_stream']:.4f}  loss "
          f"{r['provenance'].get('loss_first')} -> {r['provenance'].get('loss_final')}")

print("\n-- PCA-32 control, 3 seeds (harness validity vs w25's 0.219 ± 0.014) --")
c = load("control_pca_3seeds.json")
for arm_dim in (32, 256):
    t1 = [r["acc_ring"] for r in sel(c, arm="pca", phi_dim=arm_dim,
                                     regime="task1_only")]
    gf = [r["acc_ring"] for r in sel(c, arm="pca", phi_dim=arm_dim,
                                     regime="generic_frozen")]
    print(f"  pca-{arm_dim:<4} task1_only     {ms(t1)[0]}   {['%.4f' % v for v in t1]}")
    print(f"  pca-{arm_dim:<4} generic_frozen {ms(gf)[0]}")
    print(f"  pca-{arm_dim:<4} strict-φ cost  {ms(gf)[1] - ms(t1)[1]:+.4f}")
b = load("control_pca_bigpool.json")
for arm_dim in (32, 256):
    t1 = [r["acc_ring"] for r in sel(b, arm="pca", phi_dim=arm_dim)]
    print(f"  pca-{arm_dim:<4} task1_only, 4966-image pool {ms(t1)[0]}")

print("\n-- ⭐ THE GATE: simclr, 8000 steps, 4966-image task-1 pool, 3 seeds --")
files = ["sweep_simclr8k.json", "sweep_simclr8k_seed1.json", "sweep_simclr8k_seed2.json"]
for d in (32, 64, 128, 256):
    vals = []
    for f in files:
        r = sel(load(f), spatial="keep", head="pca", phi_dim=d, l2=True, budget=200)
        if r:
            vals.append(r[0]["acc_ring"])
    if vals:
        txt, mean = ms(vals)
        print(f"  phi_dim={d:<4} {txt}  {['%.4f' % v for v in vals]}  "
              f"{'CLEARS' if mean >= 0.35 else 'MISSES'} the 0.35 gate")

print("\n-- the read-out sweep at 8k steps, seed 0 (28 configs) --")
for r in sorted(load("sweep_simclr8k.json"), key=lambda r: -r["acc_ring"])[:6]:
    print(f"  {r['spatial']:>4} {r['head']:>10} d={r['phi_dim']:<4} l2={int(r['l2'])} "
          f"ring={r['acc_ring']:.4f}")
print("  ... worst:")
for r in sorted(load("sweep_simclr8k.json"), key=lambda r: r["acc_ring"])[:2]:
    print(f"  {r['spatial']:>4} {r['head']:>10} d={r['phi_dim']:<4} l2={int(r['l2'])} "
          f"ring={r['acc_ring']:.4f}")

print("\n-- compute scaling of the SAME arm (seed 0, keep/pca/l2, d=256) --")
for f, label in (("sweep_simclr2k.json", "2000 steps, 2027 imgs, seed 0"),
                 ("sweep_simclr8k.json", "8000 steps, 4966 imgs, seed 0"),
                 ("sweep_simclr20k.json", "20000 steps, 4966 imgs, seed 0"),
                 ("sweep_simclr20k_seed1.json", "20000 steps, seed 1"),
                 ("sweep_simclr20k_seed2.json", "20000 steps, seed 2")):
    try:
        r = sel(load(f), spatial="keep", head="pca", phi_dim=256, l2=True, budget=200)
    except FileNotFoundError:
        print(f"  {label:<26} NOT RUN")
        continue
    if r:
        print(f"  {label:<26} ring={r[0]['acc_ring']:.4f}  "
              f"full={r[0]['acc_full_stream']:.4f}  "
              f"NT-Xent {r[0]['loss_first']:.3f} -> {r[0]['loss_final']:.3f}")

print("\n-- memory-budget sweep at the gate config (second cause #1) --")
for f in files + ["sweep_simclr20k.json", "sweep_simclr20k_seed1.json",
                  "sweep_simclr20k_seed2.json"]:
    try:
        rows = [r for r in load(f) if r.get("budget_sweep")]
    except FileNotFoundError:
        continue
    base = sel(load(f), spatial="keep", head="pca", phi_dim=256, l2=True, budget=200)
    if base:
        rows = [{"budget": 200, "acc_ring": base[0]["acc_ring"]}] + rows
    if rows:
        print(f"  {f:<30} " + "  ".join(
            f"{r['budget']}:{r['acc_ring']:.4f}" for r in rows))

print("\n-- strict-φ cost at the WORKING arm (generic_frozen − task1_only) --")
for tag in ("", "matched_"):
  print(f"  [{'UNMATCHED pool 6000 vs 4966' if not tag else 'MATCHED pool 4966 vs 4966'}]")
  for sd in (0, 1, 2):
    f = f"sweep_simclr8k_generic_{tag}seed{sd}.json"
    t1f = "sweep_simclr8k.json" if sd == 0 else f"sweep_simclr8k_seed{sd}.json"
    try:
        g = sel(load(f), spatial="keep", head="pca", phi_dim=256, l2=True, budget=200)
        t = sel(load(t1f), spatial="keep", head="pca", phi_dim=256, l2=True, budget=200)
    except FileNotFoundError:
        print(f"    seed {sd}: NOT RUN")
        continue
    if g and t:
        print(f"    seed {sd}: generic {g[0]['acc_ring']:.4f}  task1 {t[0]['acc_ring']:.4f}"
              f"  cost {g[0]['acc_ring'] - t[0]['acc_ring']:+.4f}  "
              f"(pools {g[0]['n_fit']} vs {t[0]['n_fit']})")

print("\n-- store geometry at the gate config (no CLU settle; addresses only) --")
try:
    g = load("geometry_simclr8k_seed0.json")
    geo = g["geometry"]
    print(f"  arm={g['arm']} phi_dim={g['phi_dim']} l2={g['l2']} seed={g['seed']}")
    print(f"  median-NN {geo['median_nn_addresses']:.4f}  s {geo['well_width_s']:.4f}  "
          f"σ_q {geo['sigma_q_norm']:.4f}  slack {geo['packing_slack_corrected']:.4f}")
    print(f"  n_stored {geo['n_stored']}  kNN over the STORE's own keys "
          f"{g['acc_knn_same_keys']:.4f}")
    for pt in g["per_task"]:
        print(f"    task {pt['task']}: offered {pt['offered']} admitted "
              f"{pt['admitted']} frac {pt['admitted_fraction']:.3f} "
              f"s {pt['well_width_s']:.4f} live {pt['n_live']}")
except FileNotFoundError:
    print("  NOT RUN")
