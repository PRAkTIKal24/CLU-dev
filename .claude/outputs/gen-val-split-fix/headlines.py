"""The MOVED/UNCHANGED headline table: the numbers the phase actually quotes."""
import json
from pathlib import Path
import numpy as np
R = Path("/Users/user/Desktop/CHLU/.claude/outputs")
N = R / "gen-val-split-fix/runs"

def g(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None

def dig(d, path):
    for k in path.split("."):
        if d is None: return None
        d = d.get(k) if isinstance(d, dict) else None
    return d

def row(label, oldp, newp, path, seeds=(0,1,2)):
    o = [dig(g(oldp.format(seed=s)), path) for s in seeds]
    n = [dig(g(newp.format(seed=s)), path) for s in seeds]
    if any(x is None for x in o) or any(x is None for x in n):
        print(f"| {label} | {'/'.join('—' if x is None else f'{x:.4f}' for x in o)} "
              f"| {'/'.join('—' if x is None else f'{x:.4f}' for x in n)} | (incomplete) |")
        return
    om, nm = float(np.mean(o)), float(np.mean(n))
    print(f"| {label} | {om:.4f} ± {np.std(o):.4f} | {nm:.4f} ± {np.std(n):.4f} | "
          f"{nm-om:+.4f} | " + " · ".join(f"{a:.4f}→{b:.4f}" for a,b in zip(o,n)) + " |")

H_OLD = str(R/"gen-arms-a7-wiring/runs/historical_688128/seed_{seed}/baselines_manifest.json")
H_NEW = str(N/"historical_688128/seed_{seed}/baselines_manifest.json")
D_OLD = str(R/"gen-arms-a7-wiring/runs/deployed_524288/seed_{seed}/baselines_manifest.json")
D_NEW = str(N/"deployed_524288/seed_{seed}/baselines_manifest.json")

print("\n### LACE (leg H, 688,128 B) — primary sampler `ode`, and its two Langevin companions")
print("| row | broken split | repaired split | Δmean | per-seed |")
print("|---|---|---|---|---|")
for smp in ("ode","ld_published","ld_corrected"):
    row(f"lace/{smp} in-corner balacc", H_OLD, H_NEW, f"arms.lace.samplers.{smp}.in_corner_discrimination.balanced_accuracy")
    row(f"lace/{smp} composition B agg", H_OLD, H_NEW, f"arms.lace.samplers.{smp}.composition_B.aggregate")
    row(f"lace/{smp} composition A agg", H_OLD, H_NEW, f"arms.lace.samplers.{smp}.composition_A.aggregate")
    row(f"lace/{smp} refusal AUROC(C vs B)", H_OLD, H_NEW, f"arms.lace.samplers.{smp}.refusal.auroc_c_vs_b")
    row(f"lace/{smp} A7 dup_factor AUROC", H_OLD, H_NEW, f"arms.lace.samplers.{smp}.refusal.a7_controlled.auroc_by_kind.duplicate_factor.auroc")
    row(f"lace/{smp} A7 out_of_range AUROC", H_OLD, H_NEW, f"arms.lace.samplers.{smp}.refusal.a7_controlled.auroc_by_kind.out_of_range.auroc")

print("\n### kNN family (leg H, 688,128 B)")
print("| row | broken split | repaired split | Δmean | per-seed |")
print("|---|---|---|---|---|")
for v in ("knn","knn_proto"):
    row(f"{v} in-corner balacc", H_OLD, H_NEW, f"arms.knn.variants.{v}.in_corner_discrimination.balanced_accuracy")
    row(f"{v} composition B agg", H_OLD, H_NEW, f"arms.knn.variants.{v}.composition_B.aggregate")
    row(f"{v} composition A agg", H_OLD, H_NEW, f"arms.knn.variants.{v}.composition_A.aggregate")
    row(f"{v} refusal AUROC(C vs B)", H_OLD, H_NEW, f"arms.knn.variants.{v}.refusal.auroc_c_vs_b")
    row(f"{v} A7 dup_factor AUROC", H_OLD, H_NEW, f"arms.knn.variants.{v}.refusal.a7_controlled.auroc_by_kind.duplicate_factor.auroc")
    row(f"{v} A7 out_of_range AUROC", H_OLD, H_NEW, f"arms.knn.variants.{v}.refusal.a7_controlled.auroc_by_kind.out_of_range.auroc")

print("\n### kNN family (leg D, deployed 524,288 B — the SHIPPED budget)")
print("| row | broken split | repaired split | Δmean | per-seed |")
print("|---|---|---|---|---|")
for v in ("knn","knn_proto"):
    row(f"{v} in-corner balacc", D_OLD, D_NEW, f"arms.knn.variants.{v}.in_corner_discrimination.balanced_accuracy")
    row(f"{v} composition B agg", D_OLD, D_NEW, f"arms.knn.variants.{v}.composition_B.aggregate")
    row(f"{v} composition A agg", D_OLD, D_NEW, f"arms.knn.variants.{v}.composition_A.aggregate")
    row(f"{v} refusal AUROC(C vs B)", D_OLD, D_NEW, f"arms.knn.variants.{v}.refusal.auroc_c_vs_b")
    row(f"{v} A7 dup_factor AUROC", D_OLD, D_NEW, f"arms.knn.variants.{v}.refusal.a7_controlled.auroc_by_kind.duplicate_factor.auroc")
    row(f"{v} A7 out_of_range AUROC", D_OLD, D_NEW, f"arms.knn.variants.{v}.refusal.a7_controlled.auroc_by_kind.out_of_range.auroc")

print("\n### structural invariants (P7: must be UNCHANGED)")
for lbl, path in [("no_leak n_holdout_leaked","no_leak.n_holdout_leaked"),
                  ("rows.fit","rows.fit"), ("rows.val","rows.val"), ("rows.A","rows.A"),
                  ("rows.B","rows.B"), ("rows.MS","rows.MS"),
                  ("prototype table bytes","arms.knn.prototype_table_ledger.bytes"),
                  ("byte target total","arms.knn.byte_target.total_bytes")]:
    o=[dig(g(H_OLD.format(seed=s)),path) for s in (0,1,2)]
    n=[dig(g(H_NEW.format(seed=s)),path) for s in (0,1,2)]
    print(f"  {lbl:28s} old={o} new={n}  {'IDENTICAL' if o==n else 'MOVED'}")
