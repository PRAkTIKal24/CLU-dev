"""Merge the per-(seed, budget) frontier JSONs and re-derive every number in the
report with the repo's own aggregation/verdict functions. Nothing is hand-typed.

    PYTHONPATH=<worktree> .venv/bin/python render.py
"""
import glob
import json
import os

import numpy as np

from chlu.config import get_default_config
from chlu.experiments import exp_cl_entry as cle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

rows, sat, acct, tuned = [], [], None, None
for p in sorted(glob.glob(os.path.join(RES, "frontier_seed*_B*.json"))):
    d = json.load(open(p))
    rows += d["rows"]
    sat += d["store_saturation"]
    acct = acct or d["byte_accounting"]
    tuned = tuned or d.get("tuned_hypers")

cfg = get_default_config().experiment_cl_entry
table = cle.frontier_table(rows)
verdict = cle.frontier_verdict(table, cfg)
budgets = sorted({r["budget_floats"] for r in table if r["budget_floats"]})
seeds = sorted({r["seed"] for r in rows})

print(f"# seeds {seeds}  budgets {budgets}")
print("\n## byte accounting (floats/item)")
print(json.dumps(acct["floats_per_stored_item"]))
print("## fixed state (floats)")
print(json.dumps(acct["fixed_state_floats"]))
print("## tuned hypers:", json.dumps(tuned))

order = ["clu_entry", "knn_phi_ringbuffer", "knn_phi_same_keys", "icarl", "gdumb",
         "derpp", "er"]
for metric in ("forgetting", "ACC", "BWT", "LA"):
    print(f"\n## {metric} vs BYTES (mean ± sd over {len(seeds)} seeds)")
    hdr = "| method | " + " | ".join(f"{b:,}" for b in budgets) + " |"
    print(hdr)
    print("|---" * (len(budgets) + 1) + "|")
    for m in order:
        cells = []
        for b in budgets:
            r = [x for x in table if x["method"] == m and x["budget_floats"] == b]
            cells.append(f"{r[0][metric]:.3f} ± {r[0][metric + '_sd']:.3f}"
                         if r else "—")
        print(f"| {m} | " + " | ".join(cells) + " |")
    for m in sorted({r["method"] for r in table if r["budget_independent"]}):
        r = [x for x in table if x["method"] == m][0]
        print(f"| *{m}* (budget-free) | "
              + " | ".join([f"{r[metric]:.3f} ± {r[metric + '_sd']:.3f}"] * len(budgets))
              + " |")

print("\n## items actually held")
print("| method | " + " | ".join(f"{b:,}" for b in budgets) + " |")
print("|---" * (len(budgets) + 1) + "|")
for m in order:
    cells = []
    for b in budgets:
        r = [x for x in table if x["method"] == m and x["budget_floats"] == b]
        cells.append(f"{r[0]['memory_items']:.0f}" if r else "—")
    print(f"| {m} | " + " | ".join(cells) + " |")

print("\n## store saturation (mean over seeds)")
print("| B (floats) | budget items | live at end | admitted frac (per task) | "
      "refused_full | fill fraction | saturated |")
print("|---|---|---|---|---|---|---|")
for b in budgets:
    s = [x for x in sat if x["budget_floats"] == b]
    live = np.mean([x["n_live_end"] for x in s])
    bud = np.mean([x["budget_items"] for x in s])
    af = np.mean([x["admitted_fraction_per_task"] for x in s], axis=0)
    fill = live / max(1.0, bud)
    print(f"| {b:,} | {bud:.0f} | {live:.0f} | "
          + ", ".join(f"{a:.2f}" for a in af)
          + f" | {int(np.sum([x['refused_full'] for x in s]))} | "
          + f"{fill:.3f} | {'YES' if fill < 0.98 else 'no'} |")

print("\n## geometry (packing slack, mean over seeds)")
for b in budgets:
    s = [x for x in sat if x["budget_floats"] == b]
    g = [x["geometry"] for x in s if "packing_slack_corrected" in x["geometry"]]
    if g:
        print(f"  B={b:,}: median-NN {np.mean([x['median_nn_addresses'] for x in g]):.3f} "
              f"s {np.mean([x['well_width_s'] for x in g]):.3f} "
              f"sigma_q {np.mean([x['sigma_q_norm'] for x in g]):.3f} "
              f"slack {np.mean([x['packing_slack_corrected'] for x in g]):.3f}")

print("\n## VERDICT")
print(verdict["reading"])
print("dominates_at_budgets:", verdict["dominates_at_budgets"])
print("beats_replay_only_at_budgets:", verdict["beats_replay_only_at_budgets"])
print("launder_never_beaten:", verdict["launder_never_beaten"])
for b, d in verdict["per_budget"].items():
    vr, vl = d["vs_replay"], d["vs_launder"]
    print(f"  B={int(b):>7,}: CLU F {d['clu_forgetting']:.3f}±{d['clu_forgetting_sd']:.3f} "
          f"ACC {d['clu_ACC']:.3f} LA {d['clu_LA']:.3f} (best LA {d['best_LA_here']:.3f}, "
          f"in band {d['la_within_band']}) | vs replay best {vr['best_method']} "
          f"{vr['best_forgetting']:.3f} Δ{vr['clu_minus_best']:+.3f} lower={vr['clu_lower']} "
          f"sep={vr['separated_by_sd']} | vs launder {vl['best_method']} "
          f"{vl['best_forgetting']:.3f} Δ{vl['clu_minus_best']:+.3f} lower={vl['clu_lower']} "
          f"sep={vl['separated_by_sd']}")

# ---- the secondary "all-fixed-state-charged" frontier (PREREG §1b) ----------
print("\n## secondary frontier: TOTAL bytes = fixed state + stored items")
print("| method | " + " | ".join(f"{b:,}" for b in budgets) + " |")
print("|---" * (len(budgets) + 1) + "|")
for m in order:
    cells = []
    for b in budgets:
        r = [x for x in table if x["method"] == m and x["budget_floats"] == b]
        cells.append(f"{r[0]['memory_floats'] + r[0]['fixed_state_floats']:,.0f}"
                     if r else "—")
    print(f"| {m} | " + " | ".join(cells) + " |")

figs = cle._plots(
    {"config": {"dataset": "mnist"},
     "frontier": {"table": table, "store_saturation": sat, "bytes_per_float": 4}},
    HERE,
)
print("\nfigures:", figs)

json.dump({"table": table, "verdict": verdict, "byte_accounting": acct,
           "store_saturation": sat, "tuned_hypers": tuned, "seeds": seeds},
          open(os.path.join(RES, "frontier_merged.json"), "w"),
          indent=2, default=float)
print("\nmerged ->", os.path.join(RES, "frontier_merged.json"))
