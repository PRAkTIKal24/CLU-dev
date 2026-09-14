"""Aggregate the per-seed gen-r0-rescore JSONs into tables. Scratch analysis script."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("RESCORE_OUT", "."))
STORE_RUNS = Path(os.environ["STORE_RUNS"])
FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
ARMS = ["correlation", "random"]


def ms(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "std": float(v.std()), "values": [float(x) for x in v],
            "n_seeds": int(v.size)}


def fmt(d, p=4):
    return f"{d['mean']:.{p}f} ± {d['std']:.{p}f}"


seeds = sorted(int(p.stem.split("_")[-1]) for p in OUT.glob("rescore_seed_*.json"))
R = {s: json.load(open(OUT / f"rescore_seed_{s}.json")) for s in seeds}
S = {s: json.load(open(STORE_RUNS / f"seed_{s}" / "store_manifest.json")) for s in seeds}
print(f"seeds scored: {seeds}")

out = {"seeds": seeds, "n_seeds": len(seeds)}

# ---- 1. kill rule (i) corrected, REAL features (the decoder-gate layer) ----
g = {}
for dec in ("baseline_decoder", "v1_decoder"):
    g[dec] = {k: ms([R[s]["gate_real_features"][dec][k] for s in seeds])
              for k in ("balanced_accuracy", "auroc", "tpr", "tnr", "raw_accuracy")}
    g[dec]["n_pos"] = [R[s]["gate_real_features"][dec]["n_pos"] for s in seeds]
    g[dec]["n_neg"] = [R[s]["gate_real_features"][dec]["n_neg"] for s in seeds]
    g[dec]["pos_called_cylinder"] = [R[s]["gate_real_features"][dec]["pos_called_cylinder"] for s in seeds]
    g[dec]["neg_called_cylinder"] = [R[s]["gate_real_features"][dec]["neg_called_cylinder"] for s in seeds]
out["kill_rule_i_corrected_real_features"] = g

# ---- 2. the corner metric on the STORE READ, correlation vs random ----
c = {}
for arm in ARMS:
    c[arm] = {dec: {k: ms([R[s]["arms"][arm]["corner_store_read"][dec][k] for s in seeds])
                    for k in ("balanced_accuracy", "auroc", "tpr", "tnr")}
              for dec in ("baseline_decoder", "v1_decoder")}
c["margin_corr_minus_rand"] = {
    dec: {k: ms([R[s]["arms"]["correlation"]["corner_store_read"][dec][k]
                 - R[s]["arms"]["random"]["corner_store_read"][dec][k] for s in seeds])
          for k in ("balanced_accuracy", "auroc")}
    for dec in ("baseline_decoder", "v1_decoder")}
out["corner_metric_on_store_read"] = c

# ---- 3. read, old decoder vs V1 decoder ----
r = {}
for arm in ARMS:
    r[arm] = {}
    for which, key in (("old_decoder", "read_old_decoder"), ("v1_decoder", "read_new_decoder")):
        r[arm][which] = {"aggregate": ms([R[s]["arms"][arm][key]["aggregate"] for s in seeds]),
                         "per_factor": {f: ms([R[s]["arms"][arm][key]["per_factor"][f] for s in seeds])
                                        for f in FACTORS}}
r["margin_corr_minus_rand"] = {}
for which, key in (("old_decoder", "read_old_decoder"), ("v1_decoder", "read_new_decoder")):
    r["margin_corr_minus_rand"][which] = {
        "aggregate": ms([R[s]["arms"]["correlation"][key]["aggregate"]
                         - R[s]["arms"]["random"][key]["aggregate"] for s in seeds]),
        "shape": ms([R[s]["arms"]["correlation"][key]["per_factor"]["shape"]
                     - R[s]["arms"]["random"][key]["per_factor"]["shape"] for s in seeds])}
r["ceiling_no_store"] = {
    "old_decoder": ms([R[s]["ceiling_no_store"]["old_decoder"]["aggregate"] for s in seeds]),
    "v1_decoder": ms([R[s]["ceiling_no_store"]["new_decoder"]["aggregate"] for s in seeds]),
    "old_decoder_shape": ms([R[s]["ceiling_no_store"]["old_decoder"]["per_factor"]["shape"] for s in seeds]),
    "v1_decoder_shape": ms([R[s]["ceiling_no_store"]["new_decoder"]["per_factor"]["shape"] for s in seeds])}
out["read_rescored"] = r

# ---- 4. corrected leakage ----
lk = {}
for arm in ARMS:
    pooled = [R[s]["arms"][arm]["leakage"]["pooled"] for s in seeds]
    lk[arm] = {
        "pooled_leakage_aggregate": ms([p["leakage_aggregate"] for p in pooled]),
        "pooled_leakage_shape": ms([p["leakage_per_factor"]["shape"] for p in pooled]),
        "pooled_leakage_per_factor": {f: ms([p["leakage_per_factor"][f] for p in pooled]) for f in FACTORS},
        "mean_of_per_well_leakage": ms([p["mean_of_per_well_leakage"] for p in pooled]),
        "per_well_leakage_aggregate": {
            j: ms([R[s]["arms"][arm]["leakage"]["per_well"][j]["corrected"]["leakage_aggregate"] for s in seeds])
            for j in range(len(R[seeds[0]]["arms"][arm]["leakage"]["per_well"]))},
        "per_well_leakage_shape": {
            j: ms([R[s]["arms"][arm]["leakage"]["per_well"][j]["corrected"]["leakage_per_factor"]["shape"]
                   for s in seeds])
            for j in range(len(R[seeds[0]]["arms"][arm]["leakage"]["per_well"]))},
        "old_pooled_leakage_aggregate": ms([S[s]["arms"][arm]["leakage"]["pooled"]["leakage_aggregate"] for s in seeds]),
        "old_pooled_leakage_shape": ms([S[s]["arms"][arm]["leakage"]["pooled"]["leakage_per_factor"]["shape"] for s in seeds]),
    }
lk["margin_rand_minus_corr"] = {
    "pooled_leakage_aggregate": ms([R[s]["arms"]["random"]["leakage"]["pooled"]["leakage_aggregate"]
                                    - R[s]["arms"]["correlation"]["leakage"]["pooled"]["leakage_aggregate"]
                                    for s in seeds]),
    "pooled_leakage_shape": ms([R[s]["arms"]["random"]["leakage"]["pooled"]["leakage_per_factor"]["shape"]
                                - R[s]["arms"]["correlation"]["leakage"]["pooled"]["leakage_per_factor"]["shape"]
                                for s in seeds]),
    "old_pooled_leakage_aggregate": ms([S[s]["arms"]["random"]["leakage"]["pooled"]["leakage_aggregate"]
                                        - S[s]["arms"]["correlation"]["leakage"]["pooled"]["leakage_aggregate"]
                                        for s in seeds]),
}

# ---- 5. ⭐ NON-DEGENERACY PROOF: every (well, factor) cell, both arms, all seeds ----
new_cells, old_cells, new_nan, old_nan, cell_rows = [], [], 0, 0, []
for s in seeds:
    for arm in ARMS:
        for pw in R[s]["arms"][arm]["leakage"]["per_well"]:
            for f in FACTORS:
                v = pw["corrected"]["leakage_per_factor"][f]
                o = pw["old"]["leakage_per_factor"][f]
                cell_rows.append({"seed": s, "arm": arm, "well": pw["well"], "factor": f,
                                  "corrected": v, "old": o})
                (new_nan := new_nan) if v == v else None
                if v != v: new_nan += 1
                else: new_cells.append(v)
                if o != o: old_nan += 1
                else: old_cells.append(o)
lk["non_degeneracy_proof"] = {
    "n_cells_total": len(cell_rows),
    "corrected": {"min": float(np.min(new_cells)), "max": float(np.max(new_cells)),
                  "n_nan": new_nan, "n_finite": len(new_cells),
                  "n_outside_pm1": int(np.sum(np.abs(new_cells) > 1.0)),
                  "argmin": min(cell_rows, key=lambda r: r["corrected"] if r["corrected"] == r["corrected"] else 1e9),
                  "argmax": max(cell_rows, key=lambda r: r["corrected"] if r["corrected"] == r["corrected"] else -1e9)},
    "old": {"min": float(np.min(old_cells)), "max": float(np.max(old_cells)),
            "n_nan": old_nan, "n_finite": len(old_cells),
            "n_outside_pm1": int(np.sum(np.abs(old_cells) > 1.0)),
            "argmin": min(cell_rows, key=lambda r: r["old"] if r["old"] == r["old"] else 1e9)},
}
out["leakage_corrected"] = lk
json.dump({"cells": cell_rows}, open(OUT / "leakage_cells.json", "w"), indent=1, default=float)

# ---- 6. companions that the corrections leave UNCHANGED (from the shipped manifests) ----
comp = {}
for arm in ARMS:
    comp[arm] = {
        "block_diagonality": ms([S[s]["arms"][arm]["companions"]["block_diagonality"]["in_block_fraction"] for s in seeds]),
        "mig": ms([S[s]["arms"][arm]["companions"]["mig"]["mig"] for s in seeds]),
        "dci_D": ms([S[s]["arms"][arm]["companions"]["dci"]["disentanglement"] for s in seeds]),
        "dci_C": ms([S[s]["arms"][arm]["companions"]["dci"]["completeness"] for s in seeds]),
        "dci_I": ms([S[s]["arms"][arm]["companions"]["dci"]["informativeness"] for s in seeds]),
    }
out["companions_unchanged"] = comp

json.dump(out, open(OUT / "summary.json", "w"), indent=2, sort_keys=True, default=float)

# ------------------------------------------------------------------ printed tables
P = []
P.append(f"seeds = {seeds} (n = {len(seeds)})\n")
P.append("## 1. Kill rule (i), CORRECTED — cylinder vs not, WITHIN the corner (REAL features). Chance = 0.5")
P.append("| decoder | balanced accuracy | AUROC | TPR (72 held-out cyl) | TNR (matched-seen non-cyl) |")
P.append("|---|---|---|---|---|")
for dec, nm in (("baseline_decoder", "baseline (superseded)"), ("v1_decoder", "**V1 capacity_schedule**")):
    d = g[dec]
    P.append(f"| {nm} | **{fmt(d['balanced_accuracy'])}** | {fmt(d['auroc'])} | {fmt(d['tpr'])} | {fmt(d['tnr'])} |")
P.append(f"\nper-seed balacc  baseline {g['baseline_decoder']['balanced_accuracy']['values']}"
         f"  V1 {g['v1_decoder']['balanced_accuracy']['values']}")
P.append(f"per-seed AUROC   baseline {g['baseline_decoder']['auroc']['values']}"
         f"  V1 {g['v1_decoder']['auroc']['values']}")
P.append(f"n_pos {g['v1_decoder']['n_pos']}  n_neg {g['v1_decoder']['n_neg']}")

P.append("\n## 2. Same corrected metric on the STORE READ (correlation vs random). Chance = 0.5")
P.append("| arm | decoder | balanced accuracy | AUROC |")
P.append("|---|---|---|---|")
for arm in ARMS:
    for dec in ("baseline_decoder", "v1_decoder"):
        P.append(f"| {arm} | {dec} | {fmt(c[arm][dec]['balanced_accuracy'])} | {fmt(c[arm][dec]['auroc'])} |")
for dec in ("baseline_decoder", "v1_decoder"):
    P.append(f"| **margin (corr−rand)** | {dec} | **{fmt(c['margin_corr_minus_rand'][dec]['balanced_accuracy'])}** "
             f"| {fmt(c['margin_corr_minus_rand'][dec]['auroc'])} |")

P.append("\n## 3. R0's READ re-scored. ⛔ raw multi-class top-1; NOT comparable to the balanced-accuracy tables")
P.append("| | aggregate (old dec) | aggregate (V1 dec) | shape (old dec) | shape (V1 dec) |")
P.append("|---|---|---|---|---|")
for arm in ARMS:
    P.append(f"| {arm} | {fmt(r[arm]['old_decoder']['aggregate'])} | {fmt(r[arm]['v1_decoder']['aggregate'])} "
             f"| {fmt(r[arm]['old_decoder']['per_factor']['shape'])} | {fmt(r[arm]['v1_decoder']['per_factor']['shape'])} |")
P.append(f"| **margin (corr−rand)** | **{fmt(r['margin_corr_minus_rand']['old_decoder']['aggregate'])}** "
         f"| **{fmt(r['margin_corr_minus_rand']['v1_decoder']['aggregate'])}** "
         f"| **{fmt(r['margin_corr_minus_rand']['old_decoder']['shape'])}** "
         f"| **{fmt(r['margin_corr_minus_rand']['v1_decoder']['shape'])}** |")
P.append(f"| no-store ceiling | {fmt(r['ceiling_no_store']['old_decoder'])} | {fmt(r['ceiling_no_store']['v1_decoder'])} "
         f"| {fmt(r['ceiling_no_store']['old_decoder_shape'])} | {fmt(r['ceiling_no_store']['v1_decoder_shape'])} |")
P.append(f"\nper-seed aggregate margin: old {r['margin_corr_minus_rand']['old_decoder']['aggregate']['values']}")
P.append(f"per-seed aggregate margin: V1  {r['margin_corr_minus_rand']['v1_decoder']['aggregate']['values']}")
P.append(f"per-seed shape margin:     V1  {r['margin_corr_minus_rand']['v1_decoder']['shape']['values']}")

P.append("\n## 4. Corrected leakage (balanced-accuracy transfer). Pooled")
P.append("| arm | corrected pooled leakage | corrected pooled `shape` | OLD (degenerate) pooled | OLD `shape` |")
P.append("|---|---|---|---|---|")
for arm in ARMS:
    d = lk[arm]
    P.append(f"| {arm} | {fmt(d['pooled_leakage_aggregate'])} | {fmt(d['pooled_leakage_shape'])} "
             f"| {fmt(d['old_pooled_leakage_aggregate'])} | {fmt(d['old_pooled_leakage_shape'])} |")
m = lk["margin_rand_minus_corr"]
P.append(f"| **margin (rand−corr)** | **{fmt(m['pooled_leakage_aggregate'])}** | **{fmt(m['pooled_leakage_shape'])}** "
         f"| {fmt(m['old_pooled_leakage_aggregate'])} | — |")
P.append(f"\nper-seed pooled-leakage margin (rand−corr): {m['pooled_leakage_aggregate']['values']}")

P.append("\n## 5. ⭐ NON-DEGENERACY PROOF — every (well, factor) cell, both arms, all seeds")
nd = lk["non_degeneracy_proof"]
P.append(f"| | n cells | NaN cells | min | max | cells with |leakage| > 1 |")
P.append("|---|---|---|---|---|---|")
P.append(f"| **corrected** | {nd['n_cells_total']} | **{nd['corrected']['n_nan']}** | "
         f"**{nd['corrected']['min']:+.4f}** | **{nd['corrected']['max']:+.4f}** | **{nd['corrected']['n_outside_pm1']}** |")
P.append(f"| old (as literally specified) | {nd['n_cells_total']} | {nd['old']['n_nan']} | "
         f"{nd['old']['min']:+.4f} | {nd['old']['max']:+.4f} | {nd['old']['n_outside_pm1']} |")
P.append(f"\ncorrected argmin: {nd['corrected']['argmin']}")
P.append(f"corrected argmax: {nd['corrected']['argmax']}")
P.append(f"old argmin:       {nd['old']['argmin']}")

P.append("\n## 6. Companions the corrections leave UNCHANGED (from the shipped manifests)")
P.append("| arm | block-diagonality | MIG | DCI-D | DCI-C | DCI-I |")
P.append("|---|---|---|---|---|---|")
for arm in ARMS:
    d = comp[arm]
    P.append(f"| {arm} | {fmt(d['block_diagonality'])} | {fmt(d['mig'])} | {fmt(d['dci_D'])} "
             f"| {fmt(d['dci_C'])} | {fmt(d['dci_I'])} |")

txt = "\n".join(P)
print(txt)
open(OUT / "tables.md", "w").write(txt + "\n")
