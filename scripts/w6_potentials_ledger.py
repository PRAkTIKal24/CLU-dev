#!/usr/bin/env python
"""w6 hub/potentials -- the RESULTS LEDGER (charter section A.6).

Every scored run, every pre-registered bar, every seed, in one table.  Writes
RESULTS-LEDGER.md.  A leg that did not run is printed as NOT RUN, never as a null.

  scripts/w6_potentials_ledger.py --out <md> --run 1 --legs <dir> [<dir> ...]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def leg_bars(obj) -> dict:
    rows = obj["rows"]
    meta = obj["meta"]
    main = [r for r in rows if r.get("grid", "main") == "main" and r["arm"] == "field"]
    est = meta["estimand"]["corners"]
    out = {"seed": meta["seed"], "condition": meta["condition"],
           "n_corners": len(est), "n_rows": meta["n_rows"],
           "field_sha": meta["field_sha"][:16],
           "bytes": meta["byte_ledger"]["deployed_bytes"],
           "bytes_ok": meta["byte_ledger"]["matches_budget"],
           "integrator": meta["integrator_check"]["max_abs_diff"],
           "B6_hessian_fd": meta.get("hessian_fd_check", {}).get("max_abs_residual"),
           "instrument_floor": meta.get("instrument_floor", {}).get("median_abs_log10")}
    if est:
        ec = np.array([c["median_err_crr"] for c in est])
        ef = np.array([c["median_err_fit1"] for c in est])
        d = ec - ef
        out.update({"B1_median_err": float(np.median(ec)),
                    "B1_pass": bool(np.median(ec) <= 0.10),
                    "B2_mean_delta": float(d.mean()),
                    "B2_corners_better": int((d < 0).sum()),
                    "B2_pass": bool(d.mean() < 0 and (d < 0).sum() >= 40)
                    if len(d) == 49 else None,
                    "fit1_C": meta["fit1_C"]})
    und = [r for r in main if r["band"] == "underdamped"
           and np.isfinite(r["n_meas"]) and not r.get("censored")]
    if len(und) >= 8:
        slopes = []
        for g in sorted({r["gamma"] for r in und}):
            pts = [(np.log10(r["mu2"]), np.log10(r["n_meas"]))
                   for r in und if r["gamma"] == g]
            if len(pts) >= 4 and len({a for a, _ in pts}) >= 3:
                a = np.array(pts)
                slopes.append(float(np.polyfit(a[:, 0], a[:, 1], 1)[0]))
        if slopes:
            out["B3_slope_median_abs"] = float(np.median(np.abs(slopes)))
            out["B3_slope_max_abs"] = float(np.max(np.abs(slopes)))
            out["B3_pass"] = bool(np.median(np.abs(slopes)) <= 0.10)
        fl = np.array([abs(np.log10(r["n_meas"] / r["n_pred_floor"])) for r in und])
        out["B3_floor_err"] = float(np.median(fl))
        out["B3_floor_pass"] = bool(np.median(fl) <= 0.10)
    vcs = [v for v in obj.get("v_curves", {}).get("field", []) if v.get("bracketed")]
    if vcs:
        a = np.array([v["argmin_over_gamma_crit"] for v in vcs])
        out["B4_argmin_ratio"] = float(np.median(a))
        out["B4_pass"] = bool(0.5 <= np.median(a) <= 2.0)
        ax = np.array([v.get("argmin_over_gamma_crit_exact", np.nan) for v in vcs])
        ax = ax[np.isfinite(ax)]
        if ax.size:
            out["B4_argmin_ratio_exact"] = float(np.median(ax))
    allv = obj.get("v_curves", {}).get("field", [])
    for side in ("over", "under"):
        ep = np.array([v[f"{side}_err_pred"] for v in allv if f"{side}_err_pred" in v])
        en = np.array([v[f"{side}_err_naive"] for v in allv if f"{side}_err_naive" in v])
        sm = np.array([v[f"{side}_slope_meas"] for v in allv if f"{side}_slope_meas" in v])
        if ep.size:
            out[f"B5_{side}_slope_meas"] = float(np.median(sm))
            out[f"B5_{side}_err_derived"] = float(ep.mean())
            out[f"B5_{side}_err_naive"] = float(en.mean())
            out[f"B5_{side}_pass"] = bool(ep.mean() < en.mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="1")
    ap.add_argument("--legs", nargs="+", required=True)
    a = ap.parse_args(argv)

    legs = []
    for d in a.legs:
        p = Path(d) / "rows.pkl"
        if not p.exists():
            legs.append({"leg": str(d), "NOT_RUN": True})
            continue
        with open(p, "rb") as fh:
            b = leg_bars(pickle.load(fh))
        b["leg"] = str(d)
        legs.append(b)

    L = []
    L.append(f"# w6 hub/potentials — RESULTS LEDGER (scored run {a.run} of 3)\n")
    L.append("Every pre-registered bar, every seed. A leg that did not run is printed "
             "**NOT RUN**, never as a null.\n")
    L.append("| bar | what it tests | " +
             " | ".join(f"seed {l.get('seed', '?')}" for l in legs) + " |")
    L.append("|---|---|" + "---|" * len(legs))

    def row(label, what, key, fmt="{:.4f}", passkey=None):
        cells = []
        for l in legs:
            if l.get("NOT_RUN"):
                cells.append("**NOT RUN**")
                continue
            v = l.get(key)
            if v is None:
                cells.append("—")
                continue
            t = fmt.format(v) if isinstance(v, float) else str(v)
            if passkey and l.get(passkey) is not None:
                t += " **PASS**" if l[passkey] else " **MISS**"
            cells.append(t)
        L.append(f"| {label} | {what} | " + " | ".join(cells) + " |")

    row("B1", "median corner \\|log10 n_meas/n_pred\\| ≤ 0.10", "B1_median_err",
        passkey="B1_pass")
    row("B2", "paired mean Δ vs FITTED one-parameter rival (< 0)", "B2_mean_delta",
        "{:+.4f}", passkey="B2_pass")
    row("B2", "corners better than the fitted rival (≥ 40/49)", "B2_corners_better",
        "{:d}")
    row("B3", "underdamped \\|d log n / d log μ²\\| ≤ 0.10 (median over γ)",
        "B3_slope_median_abs", passkey="B3_pass")
    row("B3", "…and its max over γ (printed, stricter reading)", "B3_slope_max_abs")
    row("B3", "\\|log10 n_meas / (2ln2/−ln(1−γ))\\| ≤ 0.10", "B3_floor_err",
        passkey="B3_floor_pass")
    row("B4", "V-curve argmin / γ_crit within ×2 (D16 small-h form)",
        "B4_argmin_ratio", passkey="B4_pass")
    row("B4", "…against the EXACT inversion of h*(γ)=h", "B4_argmin_ratio_exact")
    row("B5", "D17 overdamped slope measured (derived ≈ +1.14)",
        "B5_over_slope_meas", "{:+.4f}")
    row("B5", "…mean \\|meas − derived\\| (must beat naive)", "B5_over_err_derived",
        passkey="B5_over_pass")
    row("B5", "…mean \\|meas − naive +1\\|", "B5_over_err_naive")
    row("B5", "D18 underdamped slope measured (derived ≈ −1.01)",
        "B5_under_slope_meas", "{:+.4f}")
    row("B5", "…mean \\|meas − derived\\| (must beat naive)", "B5_under_err_derived",
        passkey="B5_under_pass")
    row("B5", "…mean \\|meas − naive −1\\|", "B5_under_err_naive")
    row("B6", "Hessian + D14 lift, finite difference ≤ 1e-9", "B6_hessian_fd", "{:.3e}")
    L.append("")
    L.append("| audit | " + " | ".join(f"seed {l.get('seed','?')}" for l in legs) + " |")
    L.append("|---|" + "---|" * len(legs))
    for lab, key, fmt in (("corners scored", "n_corners", "{:d}"),
                          ("rows", "n_rows", "{:d}"),
                          ("field sha", "field_sha", "{}"),
                          ("store bytes", "bytes", "{:,}"),
                          ("byte pin matched", "bytes_ok", "{}"),
                          ("integrator max\\|diff\\|", "integrator", "{:.3e}"),
                          ("instrument floor (exact quadratic)",
                           "instrument_floor", "{:.4f}"),
                          ("fitted rival C", "fit1_C", "{:.4g}")):
        cells = []
        for l in legs:
            if l.get("NOT_RUN"):
                cells.append("**NOT RUN**")
            elif l.get(key) is None:
                cells.append("—")
            else:
                cells.append(fmt.format(l[key]))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
