"""gen-read-cell-arm — aggregate the scored cell and adjudicate, ⛔ by computation.

⭐ Also PROVES the arm sits on the same cell as the five rivals: the decoder hash, the
external-classifier hash, the row split and the store provenance recorded on the
dynread rows are compared against the ones the COPIED rival rows carry.  ⛔ If the
cell moved, the rivals' numbers are not comparable and this says so.
"""
import json, sys
from pathlib import Path

import numpy as np

from chlu.experiments.exp_gen_cell import (
    DYNREAD_CEILING_ARM, DYNREAD_LIKE_FOR_LIKE, aggregate_runs, dynread_verdict,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/Users/user/Desktop/CHLU/.claude/outputs/gen-read-cell-arm/runs")
SRC = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-val-split-fix/runs/cell")
dirs = [OUT / "cell" / f"seed_{s}" / f"seed_{s}" for s in (0, 1, 2)]
mans = [json.load(open(d / "cell_manifest.json")) for d in dirs if (d / "cell_manifest.json").exists()]

# ---- ⛔ the SAME-CELL proof ------------------------------------------------
same = {"checked": [], "ok": True}
for m in mans:
    s = int(m["seed"])
    src = json.load(open(SRC / f"seed_{s}" / f"seed_{s}" / "cell_manifest.json"))
    checks = {
        "decoder_param_hash": (m["decoder"]["decoder_param_hash"], src["decoder"]["decoder_param_hash"]),
        "decoder_recipe": (m["decoder"]["decoder_recipe"], src["decoder"]["decoder_recipe"]),
        "external_classifier_param_hash": (m["external_classifier"]["param_hash"], src["external_classifier"]["param_hash"]),
        "encoder_param_hash": (m["provenance"]["encoder_param_hash"], src["provenance"]["encoder_param_hash"]),
        "feature_cache_sha256": (m["provenance"]["feature_cache_sha256"], src["provenance"]["feature_cache_sha256"]),
        "store_path": (m["provenance"]["store"]["path"], src["provenance"]["store"]["path"]),
        "row_split_mode": (m["row_split"]["mode"], src["row_split"]["mode"]),
        "n_fit": (m["rows"]["fit"], src["rows"]["fit"]),
        "n_val": (m["rows"]["val"], src["rows"]["val"]),
        "n_B": (m["rows"]["B"], src["rows"]["B"]),
        "n_MS": (m["rows"]["MS"], src["rows"]["MS"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    # the rival rows must be the PUBLISHED ones, bit-identical
    rivals = {}
    for a in ("clu", "store_removed", "knn_proto", "lace", "ebm_compose"):
        if a in m["arms"] and a in src["arms"]:
            rivals[a] = (float(m["arms"][a]["external"]["composition_B"]["aggregate"])
                         == float(src["arms"][a]["external"]["composition_B"]["aggregate"]))
    same["checked"].append({"seed": s, "mismatches": bad,
                            "rival_rows_bit_identical": rivals})
    same["ok"] &= (not bad) and all(rivals.values())

agg = aggregate_runs(dirs, log=print)
agg["same_cell_proof"] = same
(OUT / "cell_results.json").write_text(json.dumps(agg, indent=1, default=float))

print("\n" + "=" * 100)
print("⭐ THE GENERATION TABLE — composition-B aggregate, external classifier, 3 seeds")
print("=" * 100)
order = ["store_removed", "ebm_compose", "lace", "clu", "dynread", "dynread/B",
         "dynread/A_request_off", "dynread/B_request_off", "knn_proto"]
order += [a for a in agg["arms"] if a not in order]
print(f"{'arm':26s} {'composition-B':>18s} {'per-seed':>26s} {'tuples/72':>12s} "
      f"{'still-desc @budget':>19s} {'bytes':>10s}")
for a in order:
    r = agg["arms"].get(a)
    if r is None:
        continue
    ms = r["external_B_aggregate"]
    dt = r["distinct_factor_tuples_B"]
    conv = ""
    for m in mans:
        row = m["arms"].get(a, {})
        d = row.get("dynread")
        if d:
            b = d["request_sets"]["B"]
            conv += f"{b['dynamics']['frac_still_descending']:.2f} "
    print(f"{a:26s} {ms['mean']:9.4f} ± {ms['std']:.4f} "
          f"{str([round(x,4) for x in ms['per_seed']]):>26s} "
          f"{dt['mean']:6.2f} ± {dt['std']:.2f} {conv:>19s} "
          f"{str(r['byte_ledger'].get('deployed_bytes')):>10s}")

v = agg["dynread_verdict"]
print("\n" + "=" * 100)
print("⭐⭐ VERDICT — PREREG §2b as hardened by PREREG-addendum-0")
print("=" * 100)
print(f"m = {v['margin_m']}")
for rival, r in v["against"].items():
    if "delta" not in r:
        print(f"  {rival:16s} {r['verdict']}")
        continue
    print(f"  {rival:16s} Δ={r['delta']:+.4f}  paired {[round(x,4) for x in r['paired_per_seed'].values()]}"
          f"  3/3={r['same_sign_3_of_3']}  overlap={r['intervals_overlap_mean_pm_sd']}"
          f"  ⇒ {r['verdict']}")
print(f"\n  CEILING VERDICT: {v['verdict']}   like-for-like all beaten: {v['like_for_like_all_beaten']}   PASS: {v['PASS']}")
print(f"\n  same-cell proof OK: {same['ok']}")

# ---- the steering read ----------------------------------------------------
print("\n⭐ STEERING (the falsifier): full vs request-off, per rung")
for rung, off in (("dynread", "dynread/A_request_off"), ("dynread/B", "dynread/B_request_off")):
    if rung not in agg["arms"] or off not in agg["arms"]:
        continue
    a, b = agg["arms"][rung]["external_B_aggregate"], agg["arms"][off]["external_B_aggregate"]
    per = [x - y for x, y in zip(a["per_seed"], b["per_seed"])]
    print(f"  {rung:12s} {a['mean']:.4f} ± {a['std']:.4f}  vs request-off {b['mean']:.4f} ± {b['std']:.4f}"
          f"  Δ={a['mean']-b['mean']:+.4f}  per-seed {[round(x,4) for x in per]}"
          f"  tuples {agg['arms'][rung]['distinct_factor_tuples_B']['mean']:.2f} vs "
          f"{agg['arms'][off]['distinct_factor_tuples_B']['mean']:.2f}")

print("\n⭐ wall clock per seed:", [round(m.get("wall_clock_s", float('nan'))) for m in mans])
