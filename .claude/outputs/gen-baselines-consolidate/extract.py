"""gen-baselines-consolidate: extract every reported number from the on-disk artefacts.
READ-ONLY. Writes tables.json + tables.md under .claude/outputs/gen-baselines-consolidate/."""
import json, math, os
from statistics import mean, stdev, pstdev

RUNS = ".claude/outputs/gen-baselines/runs/shapes3d_recon"
OUT  = ".claude/outputs/gen-baselines-consolidate"
SEEDS = (0, 1, 2)

man = {s: json.load(open(f"{RUNS}/seed_{s}/baselines_manifest.json")) for s in SEEDS}
agg = json.load(open(f"{RUNS}/baselines_results.json"))

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return (mean(vals), pstdev(vals), stdev(vals) if len(vals) > 1 else 0.0, len(vals))

def fmt(t, nd=4):
    if t is None: return "NOT RUN"
    m, s0, s1, n = t
    return f"{m:.{nd}f} ± {s0:.{nd}f} [ddof1 {s1:.{nd}f}] (n={n})"

# ---- arm accessor: returns dict of the per-seed record for a named arm
def arm_record(s, arm):
    m = man[s]
    if arm in ("knn", "knn_proto", "knn_compsel"):
        v = m["arms"]["knn"]["variants"]
        if arm == "knn_compsel" and arm not in v:
            return v["knn"], "IDENTICAL-TO-knn (selection rules coincide on this seed)"
        return v[arm], None
    if arm.startswith("lace_"):
        return m["arms"]["lace"]["samplers"][arm[len("lace_"):]], None
    raise KeyError(arm)

ARMS = ["knn", "knn_proto", "knn_compsel",
        "lace_ode", "lace_ld_published", "lace_ld_corrected"]

table = {}
for arm in ARMS:
    per = {}
    for s in SEEDS:
        r, note = arm_record(s, arm)
        ref = r["refusal"]; bk = ref["auroc_c_vs_b_by_kind"]
        per[s] = dict(
            note=note,
            in_corner_balacc=r["in_corner_discrimination"]["balanced_accuracy"],
            in_corner_auroc=r["in_corner_discrimination"]["auroc"],
            in_corner_tpr=r["in_corner_discrimination"]["tpr"],
            in_corner_tnr=r["in_corner_discrimination"]["tnr"],
            B_shape=r["composition_B"]["per_factor"]["shape"],
            B_aggregate=r["composition_B"]["aggregate"],
            B_exact=r["composition_B"]["exact_match"],
            A_aggregate=r["composition_A"]["aggregate"],
            A_exact=r["composition_A"]["exact_match"],
            auroc_c_vs_b=ref["auroc_c_vs_b"],
            auroc_c_vs_a=ref["auroc_c_vs_a"],
            auroc_c_vs_aub=ref["auroc_c_vs_a_union_b"],
            dup=bk["duplicate_factor"]["auroc"], n_dup=bk["duplicate_factor"]["n_c"],
            oor=bk["out_of_range"]["auroc"],     n_oor=bk["out_of_range"]["n_c"],
            refused_A=ref["refused_fraction"]["A"], refused_B=ref["refused_fraction"]["B"],
            refused_C=ref["refused_fraction"]["C"], refused_MS=ref["refused_fraction"]["MS"],
        )
    table[arm] = per

# ---- oracle bounds (per-arm family, not per-sampler)
oracle = {}
for s in SEEDS:
    oracle[s] = dict(
        knn_c_vs_b=man[s]["arms"]["knn"]["tuning"]["ORACLE_UPPER_BOUND_auroc_c_vs_b"]["value"],
        knn_c_vs_aub=man[s]["arms"]["knn"]["tuning"]["ORACLE_UPPER_BOUND_auroc_c_vs_b"]["max_auroc_c_vs_a_union_b_over_all_trials"],
        lace_c_vs_b=man[s]["arms"]["lace"]["tuning"]["ORACLE_UPPER_BOUND_auroc_c_vs_b"]["value"],
        lace_c_vs_aub=man[s]["arms"]["lace"]["tuning"]["ORACLE_UPPER_BOUND_auroc_c_vs_b"]["max_auroc_c_vs_a_union_b_over_all_trials"],
    )

# ---- ceilings
ceil = {s: man[s]["ceilings"] for s in SEEDS}

out = dict(table=table, oracle=oracle,
           store_ceiling=agg["store_ceiling_balanced_accuracy"],
           decoder_ceiling_balacc=agg["summary"]["_ceiling_decoder"]["in_corner_balacc"],
           decoder_ceiling_A=agg["summary"]["_ceiling_decoder"]["A_aggregate"])
json.dump(out, open(f"{OUT}/tables.json", "w"), indent=1)

# ---- markdown
L = []
L.append("# gen-baselines-consolidate — extracted tables (all numbers from on-disk artefacts)\n")
L.append("⚠ **± is the POPULATION sd (ddof=0), the convention the shipped aggregate and the R26/R27 ledger use**; the sample sd (ddof=1) is given in brackets.\n")
L.append(f"Source: `{RUNS}/seed_{{0,1,2}}/baselines_manifest.json` and `{RUNS}/baselines_results.json`\n")

def block(title, key, nd=4):
    L.append(f"\n## {title}\n")
    L.append("| arm | seed 0 | seed 1 | seed 2 | mean ± sd |")
    L.append("|---|---|---|---|---|")
    for arm in ARMS:
        v = [table[arm][s][key] for s in SEEDS]
        L.append(f"| `{arm}` | " + " | ".join(f"{x:.{nd}f}" for x in v) + f" | {fmt(ms(v), nd)} |")

block("Composition — in-corner `shape` balanced accuracy (R23 gate statistic)", "in_corner_balacc")
block("Composition — in-corner AUROC (companion, NEVER the gate)", "in_corner_auroc")
block("Composition — B `shape` probe accuracy", "B_shape")
block("Composition — B aggregate probe accuracy", "B_aggregate")
block("Composition — A aggregate probe accuracy", "A_aggregate")
block("Refusal — AUROC(B vs C) POOLED  [⛔ CONTAMINATED: C-dup carries 7 terms vs B's 6]", "auroc_c_vs_b")
block("Refusal — AUROC(B vs C), kind = `duplicate_factor`  [⛔ CONTAMINATED — needs A7]", "dup")
block("Refusal — AUROC(B vs C), kind = `out_of_range`  [✅ A7-CLEAN: control is 6-term valid, detector 0.5000]", "oor")
block("Refusal — AUROC(A vs C) pooled [⛔ contaminated on the dup half]", "auroc_c_vs_a")
block("Refusal — AUROC(A∪B vs C) pooled [⛔ contaminated on the dup half]", "auroc_c_vs_aub")
block("Refused fraction on C (threshold = A-only 95th pct)", "refused_C")
block("Refused fraction on B (false refusal of valid-unseen)", "refused_B")

L.append("\n## ORACLE-selected upper bounds (max over all 36 trials, using C — ⛔ not achievable)\n")
L.append("| family | metric | seed 0 | seed 1 | seed 2 | mean ± sd |")
L.append("|---|---|---|---|---|---|")
for fam, met in [("knn","c_vs_b"),("knn","c_vs_aub"),("lace","c_vs_b"),("lace","c_vs_aub")]:
    v=[oracle[s][f"{fam}_{met}"] for s in SEEDS]
    L.append(f"| `{fam}` | {met} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {fmt(ms(v))} |")

L.append("\n## Ceilings (re-measured in-run, real features through THE frozen decoder)\n")
L.append("| ceiling | value |")
L.append("|---|---|")
dc=out["decoder_ceiling_balacc"]; da=out["decoder_ceiling_A"]
L.append(f"| decoder ceiling, in-corner `shape` balacc | {dc['mean']:.4f} ± {dc['std']:.4f} (n={dc['n_seeds']}) |")
L.append(f"| decoder ceiling, A aggregate | {da['mean']:.4f} ± {da['std']:.4f} (n={da['n_seeds']}) |")
L.append(f"| STORE ceiling, in-corner balacc (R22/R23, carried constant) | {out['store_ceiling']:.4f} |")

open(f"{OUT}/tables.md","w").write("\n".join(L)+"\n")
print("\n".join(L))
