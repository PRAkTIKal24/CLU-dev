"""Aggregate the arms into the two criterion-4 verdicts + markdown tables."""
import json
from pathlib import Path

import numpy as np

O = Path("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire")

CAMELS_REFS = {  # median NSE, 447 basins, Kratzert 2019 Table 3 (verified)
    "LSTM(static) ensemble": 0.7580,
    "EA-LSTM ensemble": 0.7423,
    "LSTM single (mean of 8 seeds)": 0.731,
    "EA-LSTM single (mean of 8 seeds)": 0.714,
    "HBV upper (100 calibrated)": 0.6756,
    "mHM (basin-calibrated)": 0.6659,
    "FUSE 902": 0.6505,
    "FUSE 900": 0.6389,
    "FUSE 904": 0.6222,
    "SAC-SMA": 0.6028,
    "VIC (basin-calibrated)": 0.5513,
    "mHM (CONUS)": 0.5274,
    "HBV lower (1000 uncalibrated)": 0.4165,
    "VIC (CONUS)": 0.3070,
}
NC_REFS = {  # Arias Chao et al. Table 5, DS02 test units {11,14,15}
    "CNN hybrid": 4.14, "FNN hybrid": 4.22,
    "CNN data-driven [w,x_s]": 4.95, "FNN data-driven [w,x_s]": 7.89,
}


def load(fn):
    p = O / fn
    return [json.loads(l) for l in open(p)] if p.exists() else []


def camels():
    reg = load("arms.jsonl") + load("arms_distfix.jsonl")
    loc = load("arms_local.jsonl")
    comp = load("companions.jsonl")
    for r in reg:
        r["store"] = "regional"
    out = {}

    inb = [r for r in reg if r["in_budget"]]
    over = [r for r in reg if not r["in_budget"]]
    best_in = max(inb, key=lambda r: r["median_nse_447"]) if inb else None
    best_over = max(over, key=lambda r: r["median_nse_447"]) if over else None
    loc_in = [r for r in loc if r["in_budget"]]
    loc_over = [r for r in loc if not r["in_budget"]]
    best_loc_in = max(loc_in, key=lambda r: r["median_nse_447"]) if loc_in else None
    best_loc_over = (max(loc_over, key=lambda r: r["median_nse_447"])
                     if loc_over else None)

    cands = [r for r in (best_in, best_loc_in) if r]
    E_row = max(cands, key=lambda r: r["median_nse_447"]) if cands else None
    E = E_row["median_nse_447"]
    S = CAMELS_REFS["LSTM(static) ensemble"]
    out["E_row"] = E_row
    out["E"] = E
    out["S_primary"] = S
    out["margin_abs"] = E - S
    out["margin_rel"] = (E - S) / S
    out["dies"] = bool((E - S) >= -0.02 or (E - S) / S >= -0.02)
    out["criterion4_cleared_camels"] = not out["dies"]
    out["margins_vs_every_reference"] = {
        k: dict(S=v, margin_abs=round(E - v, 4),
                margin_rel=round((E - v) / v, 4),
                fires=bool((E - v) >= -0.02 or (E - v) / v >= -0.02))
        for k, v in CAMELS_REFS.items()}
    out["best_in_budget_regional"] = best_in
    out["best_over_budget_regional"] = best_over
    out["best_in_budget_local"] = best_loc_in
    out["best_over_budget_local"] = best_loc_over
    out["companions"] = comp

    # ladder table: max over k/weight/target/selection per (window,scaling,L)
    lad = {}
    for r in reg:
        key = (r["window"], r["scaling"], r["L"])
        lad[key] = max(lad.get(key, -9e9), r["median_nse_447"])
    out["ladder"] = {f"{w}d|{s}|L={L}": round(v, 4)
                     for (w, s, L), v in sorted(lad.items())}
    lad2 = {}
    for r in reg:
        key = (r["window"], r["scaling"], r["L"],
               "kmeans" if "kmeans" in r["tag"] else "rand")
        lad2[key] = max(lad2.get(key, -9e9), r["median_nse_447"])
    out["ladder_by_selection"] = {f"{w}d|{s}|L={L}|{sel}": round(v, 4)
                                  for (w, s, L, sel), v in sorted(lad2.items())}
    return out


def ncmapss():
    rows = load("ncmapss_arms.jsonl")
    triv = [r for r in rows if r.get("arm") in
            ("mean_RUL", "affine_cycle_index", "affine_cycle_index_clip0",
             "mean_EOL_minus_cycle")]
    store = [r for r in rows if r.get("arm") in ("knn", "traj_similarity")]
    inb = [r for r in store if r["in_budget"]]
    matched = [r for r in inb if r.get("feats") != "resid_cycle"]
    best = min(matched, key=lambda r: r["rmse"]) if matched else None
    best_any = min(inb, key=lambda r: r["rmse"]) if inb else None
    E = best["rmse"]
    out = dict(E_row=best, E_rmse=E, best_any_row=best_any,
               trivial=triv,
               best_trivial=min(triv, key=lambda r: r["rmse"]))
    out["margins_vs_every_reference"] = {
        k: dict(S=v, ratio=round(E / v, 4), rel_pct=round((E - v) / v * 100, 2),
                fires=bool(E <= v * 1.02)) for k, v in NC_REFS.items()}
    Sbest = min(NC_REFS.values())
    out["dies"] = bool(E <= Sbest * 1.02)
    out["criterion4_cleared_ncmapss"] = not out["dies"]
    bt = out["best_trivial"]["rmse"]
    out["criterion2"] = dict(
        best_trivial_rmse=bt, best_strong_rmse=Sbest,
        rel_improvement_pct=round((bt - Sbest) / bt * 100, 2),
        passes=bool((bt - Sbest) / bt > 0.10))
    return out


if __name__ == "__main__":
    res = dict(camels=camels(), ncmapss=ncmapss())
    json.dump(res, open(O / "VERDICT.json", "w"), indent=1)
    c = res["camels"]
    print("CAMELS  E =", round(c["E"], 4), "S =", c["S_primary"],
          "margin_abs =", round(c["margin_abs"], 4),
          "margin_rel =", round(c["margin_rel"], 4),
          "=> criterion4_cleared =", c["criterion4_cleared_camels"])
    print("  E row:", c["E_row"])
    n = res["ncmapss"]
    print("NCMAPSS E =", n["E_rmse"], "=> criterion4_cleared =",
          n["criterion4_cleared_ncmapss"], "| crit2:", n["criterion2"])
    print("  E row:", n["E_row"])
