"""The 63-corner paired estimand for scoreboard row 2, exactly as PREREG-0 s3/s4 fixes it.

Arithmetic only: it reads the judged outputs each arm checkpointed and produces the
table, the paired estimand against basis_head with its 95% CI over corners, the SNAP,
and one line per charter C floor. No model runs here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402

DELTA = hc.DELTA
MARGIN = 0.2 * DELTA          # PREREG-0 s4, declared in advance
CONTROL = "basis_head"        # w3-common s1: the like-for-like, FIRST
SECOND_ROW = "knn_proto"


def t_crit(df):
    """Two-sided 95% t critical value; table for small df, normal limit above."""
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
           25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    if df <= 0:
        return float("nan")
    ks = sorted(tab)
    for k in ks:
        if df <= k:
            return tab[k]
    return 1.960


def per_corner(vals, ck):
    keys = sorted(set(ck.tolist()))
    return keys, np.array([float(np.mean(vals[ck == k])) for k in keys])


def paired(arm_v, ctl_v, ck, higher_is_better):
    """Per-corner paired difference (control - arm) oriented so positive = arm better."""
    keys, a = per_corner(arm_v, ck)
    _, c = per_corner(ctl_v, ck)
    d = (a - c) if higher_is_better else (c - a)
    n = d.size
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    tc = t_crit(n - 1)
    return {
        "n_corners": int(n), "mean_diff": m, "sd_over_corners": sd, "se": float(se),
        "ci95": [m - tc * se, m + tc * se],
        "corners_favouring_arm": int((d > 0).sum()),
        "per_corner_diff": d.tolist(), "corner_keys": keys,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev512")
    a = ap.parse_args()
    d = hc.OUT / "runs" / a.tag
    man = json.load(open(d / "run_manifest.json", encoding="utf-8"))
    ck = np.array(man["corner_keys"])

    arms = {}
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        arms[f.stem] = z
    if CONTROL not in arms:
        raise SystemExit(f"the like-for-like control {CONTROL!r} is missing; "
                         "w3-common s1 makes it mandatory on every row")

    rows = {}
    for nm, z in arms.items():
        err = np.asarray(z["err"], np.float64)
        rows[nm] = {
            "HUEMAE_delta": float(err.mean() / DELTA),
            "HUEMAE_raw": float(err.mean()),
            "ROW2ACC_gated": float(np.asarray(z["row2acc"]).mean()),
            "hue_hit_ungated": float(np.asarray(z["hit"]).mean()),
            "other5_all_correct": float(np.asarray(z["ok5"]).mean()),
            "SNAPFRAC": float(np.asarray(z["snap"]).mean()),
            "mean_d_req_delta": float(np.asarray(z["d_req"]).mean() / DELTA),
            "mean_d_near_delta": float(np.asarray(z["d_near"]).mean() / DELTA),
            "mean_t": float(np.asarray(z["t"]).mean()),
            "sd_t": float(np.asarray(z["t"]).std()),
            "n_distinct_feature_vectors": int(z["n_distinct_feature_vectors"]),
            "n_distinct_judged_tuples": int(z["n_distinct_judged_tuples"]),
            "n_distinct_hue_readings": int(z["n_distinct_hue_readings"]),
            "HUEMAE_ref_delta": (float(np.asarray(z["err_ref"]).mean() / DELTA)
                                 if "err_ref" in z else None),
            "per_factor_other5": json.loads(str(z["per_factor_other5"])),
        }

    # AMENDMENT 2: the BAR is whichever fitted-basis variant is stronger on HUEMAE.
    cands = [c for c in ("basis_head", "basis_head_pca") if c in rows]
    BAR = min(cands, key=lambda c: rows[c]["HUEMAE_delta"])
    others = ", ".join(f"{c} {rows[c]['HUEMAE_delta']:.4f}" for c in cands if c != BAR)
    print("AMENDMENT 2 bar = " + BAR
          + f" (HUEMAE {rows[BAR]['HUEMAE_delta']:.4f} D vs " + others + ")")

    est = {}
    for nm, z in arms.items():
        if nm == CONTROL:
            continue
        est[nm] = {}
        if "err_ref" in z and "err_ref" in arms[CONTROL]:
            est[nm]["HUEMAE_ref"] = paired(
                np.asarray(z["err_ref"], np.float64) / DELTA,
                np.asarray(arms[CONTROL]["err_ref"], np.float64) / DELTA,
                ck, higher_is_better=False)
        est[nm].update({
            "HUEMAE": paired(np.asarray(z["err"], np.float64) / DELTA,
                             np.asarray(arms[CONTROL]["err"], np.float64) / DELTA,
                             ck, higher_is_better=False),
            "ROW2ACC": paired(np.asarray(z["row2acc"], np.float64),
                              np.asarray(arms[CONTROL]["row2acc"], np.float64),
                              ck, higher_is_better=True),
        })
        h = est[nm]["HUEMAE"]
        win = (h["mean_diff"] >= MARGIN) and (h["ci95"][0] > 0.0)
        est[nm]["verdict"] = (
            "beats basis+head beyond the declared margin" if win
            else ("A CLU mechanism that ties basis+head is decoration and is reported "
                  "as such, in those words")
        )
        est[nm]["margin_declared_in_advance"] = MARGIN / DELTA
        if nm != BAR:
            for metric, key, hib in (("HUEMAE_vs_BAR", "err", False),
                                     ("HUEMAE_ref_vs_BAR", "err_ref", False),
                                     ("ROW2ACC_vs_BAR", "row2acc", True)):
                if key in z and key in arms[BAR]:
                    sc = DELTA if key.startswith("err") else 1.0
                    est[nm][metric] = paired(np.asarray(z[key], np.float64) / sc,
                                             np.asarray(arms[BAR][key], np.float64) / sc,
                                             ck, higher_is_better=hib)

    # ---- charter C floors --------------------------------------------------
    floors = {}
    for nm in ("store_removed_span", "store_removed_discrete"):
        if nm in rows:
            floors[f"landscape_removed_collapses::{nm}"] = {
                "HUEMAE_delta": rows[nm]["HUEMAE_delta"],
                "registered_threshold_delta": 0.8,
                "collapses": rows[nm]["HUEMAE_delta"] >= 0.8,
            }
    if SECOND_ROW in rows:
        floors["beats_matched_bytes_lookup"] = {
            "knn_proto_HUEMAE_delta": rows[SECOND_ROW]["HUEMAE_delta"],
            "note": "every arm must beat the matched-bytes lookup or the store is "
                    "decorative and is reported as such (charter C.1)",
        }

    out = {
        "what": "scoreboard ROW 2 (unseen-valid, in-basin hue interpolation)",
        "scope": hc.SCOPE,
        "tag": a.tag,
        "n_requests": man["n_requests"],
        "n_corners": man["n_corners"],
        "like_for_like": CONTROL,
        "second_row": SECOND_ROW,
        "declared_margin_delta": MARGIN / DELTA,
        "table": rows,
        "paired_estimand_vs_basis_head": est,
        "floors": floors,
        "AMENDMENT_2_bar": BAR,
        "INSTRUMENT_ANCHORS": {
            "oracle_HUEMAE_delta": 0.3130,
            "perfect_lookup_HUEMAE_delta": 0.5000,
            "registered_metric_usable_span_delta": 0.1870,
            "declared_margin_delta": MARGIN / DELTA,
            "oracle_HUEMAE_ref_delta": 0.0,
            "perfect_lookup_HUEMAE_ref_delta": 0.5000,
            "SNAPFRAC_null_oracle": 0.6680,
            "SNAPFRAC_at_perfect_lookup": 1.0000,
            "WARNING": "the declared margin 0.2 D EXCEEDS the registered metric's whole "
                       "usable span 0.187 D, so no arm can win row 2 on the registered "
                       "metric. Every number in this table is DEV and decides nothing.",
        },
        "coefficient_report": man.get("coefficient_report"),
        "store_pin": man.get("store_pin"),
        "judge": man.get("judge"),
    }
    hc.jdump(out, d / "row2_table.json")

    w = max(len(k) for k in rows)
    print(f"\nROW 2 — {man['n_requests']} requests, {man['n_corners']} corners, "
          f"margin {MARGIN/DELTA:.2f}D declared in advance\n")
    print(f"{'arm':<{w}}  {'HUEMAE(D)':>9}  {'REF(D)':>7}  {'ROW2ACC':>8}  {'hit':>6}  "
          f"{'oth5':>6}  {'SNAP':>6}  {'d_req':>6}  {'d_near':>6}  {'tuples':>6}")
    order = [CONTROL, SECOND_ROW] + [k for k in rows if k not in (CONTROL, SECOND_ROW)]
    for nm in order:
        if nm not in rows:
            continue
        r = rows[nm]
        rf = r.get("HUEMAE_ref_delta")
        print(f"{nm:<{w}}  {r['HUEMAE_delta']:>9.4f}  "
              f"{(f'{rf:.4f}' if rf is not None else '   -- '):>7}  "
              f"{r['ROW2ACC_gated']:>8.4f}  "
              f"{r['hue_hit_ungated']:>6.4f}  {r['other5_all_correct']:>6.4f}  "
              f"{r['SNAPFRAC']:>6.4f}  {r['mean_d_req_delta']:>6.3f}  "
              f"{r['mean_d_near_delta']:>6.3f}  {r['n_distinct_judged_tuples']:>6d}")
    print("\npaired estimand vs basis_head (positive = arm better), 95% CI over corners:")
    for nm, e in est.items():
        h = e.get("HUEMAE_ref", e["HUEMAE"])
        print(f"  {nm:<{w}} dHUEMAE {h['mean_diff']:+.4f}D  "
              f"CI [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  "
              f"{h['corners_favouring_arm']}/{h['n_corners']} corners  -> {e['verdict'][:52]}")
    print(f"\nwrote {d}/row2_table.json")


if __name__ == "__main__":
    main()
