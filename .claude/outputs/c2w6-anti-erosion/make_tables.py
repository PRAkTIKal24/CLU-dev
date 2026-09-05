"""Build TABLES.md for the c2w6-anti-erosion report from the run artifacts."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/user/Desktop/CHLU-c2w6")
from chlu.experiments.exp_anti_erosion import (  # noqa: E402
    _record_events, aggregate, post_guard_violations, spearman,
)

OUT = Path("/Users/user/Desktop/CHLU/.claude/outputs/c2w6-anti-erosion")


def load():
    recs = []
    for p in sorted(OUT.glob("erosion_*_records.json")):
        if "smoke" in p.name:
            continue
        recs.extend(json.loads(p.read_text()).get("records", []))
    return recs


def f(x, n=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.{n}g}"


def main():
    recs = load()
    agg = aggregate(recs)
    L = []
    L.append("# c2w6-anti-erosion — TABLES (generated)\n")
    L.append(f"records: {len(recs)}  "
             f"cells: {sorted({r['cell'] for r in recs})}\n")

    # ---- T1 the erosion curve ------------------------------------------
    L.append("\n## T1 — the erosion curve (median fitted depth, lane 0, val batch)\n")
    L.append("| cell | seed | step0 | step200 | final | final/step0 | final/step200 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in recs:
        L.append(f"| {r['cell']} | {r['seed']} | {f(r['depth_untrained'])} | "
                 f"{f(r['depth_at_200'])} | {f(r['depth_final'])} | "
                 f"{f(r['depth_ratio_final_over_untrained'],3)} | "
                 f"{f(r['depth_ratio_1000_over_200'],3)} |")

    L.append("\n### T1b — seed mean ± SE per cell\n")
    L.append("| cell | n | depth final | ratio final/200 | ratio final/untrained |")
    L.append("|---|---|---|---|---|")
    for c, row in agg["cells"].items():
        L.append(f"| {c} | {row['n_seeds']} | {f(row['depth_final'])} ± "
                 f"{f(row['depth_final_se'],2)} | "
                 f"{f(row['depth_ratio_1000_over_200'],3)} ± "
                 f"{f(row['depth_ratio_1000_over_200_se'],2)} | "
                 f"{f(row['depth_ratio_final_over_untrained'],3)} ± "
                 f"{f(row['depth_ratio_final_over_untrained_se'],2)} |")

    # ---- T2 bpc / K4 -----------------------------------------------------
    L.append("\n## T2 — bpc, live vs blank vs memory-deleted (K4's columns)\n")
    L.append("| cell | seed | bpc live | live−blank | memdel(eval)−live | "
             "none(retrained)−live | acq / chance |")
    L.append("|---|---|---|---|---|---|---|")
    for r in recs:
        L.append(f"| {r['cell']} | {r['seed']} | {f(r['bpc_live'])} | "
                 f"{f(r['bpc_live_minus_blank'],3)} | "
                 f"{f(r['bpc_memory_deleted_minus_live'],3)} | "
                 f"{f(r['bpc_none_minus_live'],3)} | "
                 f"{f(r['acq'],3)} / {f(r['chance'],3)} |")

    # ---- T3 I1 ------------------------------------------------------------
    L.append("\n## T3 — I1: the rewrite audit\n")
    L.append("| cell | seed | admits | occupied-target | evicting | events | "
             "pre-guard viol | post-guard viol | rate(pre) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        ev = _record_events(r)
        L.append(f"| {r['cell']} | {r['seed']} | {r.get('n_admitted','—')} | "
                 f"{r.get('n_occupied_target','—')} | "
                 f"{r.get('n_evicting_target','—')} | {r['n_rewrite_events']} | "
                 f"{r['n_rewrite_violations']} | {post_guard_violations(ev)} | "
                 f"{f(r['rewrite_violation_rate'],3)} |")

    L.append("\n### T3b — the interference audit (the measurable #9/#12 channel)\n")
    L.append("| cell | seed | events | own-leg viol vs decay law | max own residual "
             "| foreign-up rate | median rel Δforeign |")
    L.append("|---|---|---|---|---|---|---|")
    for r in recs:
        i = r.get("interference", {})
        L.append(f"| {r['cell']} | {r['seed']} | {i.get('n_events_total','—')} | "
                 f"{i.get('n_down_own_total','—')} | "
                 f"{f(i.get('max_abs_own_residual_vs_decay_law'),3)} | "
                 f"{f(i.get('rate_up_foreign_mean'),3)} | "
                 f"{f(i.get('median_rel_change_foreign_mean'),3)} |")

    # ---- T4 I2 -----------------------------------------------------------
    L.append("\n## T4 — I2: usage vs erosion (ρ, Spearman over live wells)\n")
    L.append("| cell | seed | n wells | ρ(read-selection, erosion) | "
             "ρ(LOO Δbpc, erosion) | ρ(grad, erosion) |")
    L.append("|---|---|---|---|---|---|")
    for r in recs:
        i2 = r.get("i2", {})
        L.append(f"| {r['cell']} | {r['seed']} | {i2.get('n_wells','—')} | "
                 f"{f(i2.get('rho_read_selection'),3)} | "
                 f"{f(i2.get('rho_loo_delta_bpc'),3)} | "
                 f"{f(i2.get('rho_grad_atoms'),3)} |")

    L.append("\n### T4b — pooled per-well rows on the partition-OFF arm\n")
    L.append("| seed | slot | erosion rate | mean read-sel | mean LOO Δbpc | "
             "mean ‖∂L/∂atoms‖ | depth first→last |")
    L.append("|---|---|---|---|---|---|---|")
    pool_u, pool_l, pool_r = [], [], []
    for r in [r for r in recs if r["cell"] == "p1_off"]:
        for w in r.get("i2", {}).get("wells", []):
            if w["live_frac"] <= 0.5:
                continue
            L.append(f"| {r['seed']} | {w['slot']} | {f(w['erosion_rate'],3)} | "
                     f"{f(w['mean_read_selection'],3)} | "
                     f"{f(w['mean_loo_delta_bpc'],3)} | "
                     f"{f(w['mean_grad_atoms'],3)} | "
                     f"{f(w['depth_first'],3)} → {f(w['depth_last'],3)} |")
            if np.isfinite(w["erosion_rate"]):
                pool_u.append(w["mean_read_selection"])
                pool_l.append(w["mean_loo_delta_bpc"])
                pool_r.append(w["erosion_rate"])
    if pool_r:
        L.append(f"\nPOOLED over seeds (n={len(pool_r)} wells): "
                 f"ρ(read-sel, erosion) = {f(spearman(pool_u, pool_r),3)}, "
                 f"ρ(LOO Δbpc, erosion) = {f(spearman(pool_l, pool_r),3)}")

    # ---- T5 the gate ------------------------------------------------------
    L.append("\n## T5 — the mechanical gate (prereg §4)\n```json")
    L.append(json.dumps(agg["gate"], indent=1, default=float))
    L.append("```\n")
    if "gate_w40" in agg:
        L.append("### the w40 gate\n```json")
        L.append(json.dumps(agg["gate_w40"], indent=1, default=float))
        L.append("```\n")
    L.append("## T6 — the prereg scorecard\n```json")
    L.append(json.dumps(agg["prereg"], indent=1, default=float))
    L.append("```\n")
    L.append("## T7 — the run-3 flag block\n```bash")
    L.append(agg["run3_flag_block"])
    L.append("```\n")

    (OUT / "TABLES.md").write_text("\n".join(L))
    (OUT / "erosion_aggregate.json").write_text(
        json.dumps({"n_records": len(recs), "aggregate": agg}, default=float,
                   indent=1))
    print(f"wrote {OUT / 'TABLES.md'}  ({len(recs)} records)")
    print(json.dumps(agg["gate"], indent=1, default=float))


if __name__ == "__main__":
    main()
