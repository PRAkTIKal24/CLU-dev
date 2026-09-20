#!/usr/bin/env python
"""w6 hub/potentials -- the results ledger.  Reads one or more leg directories and
prints every pre-registered bar with its number.  Renders no verdict it cannot show.

  scripts/w6_potentials_report.py <leg-dir> [<leg-dir> ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load(dirs):
    legs = []
    for d in dirs:
        p = Path(d) / "rows.pkl"
        if not p.exists():
            print(f"  !! {p} missing -- leg NOT RUN", flush=True)
            continue
        with open(p, "rb") as fh:
            legs.append((str(d), pickle.load(fh)))
    return legs


def band_table(rows):
    out = {}
    for r in rows:
        if r.get("censored") or not np.isfinite(r["n_meas"]):
            continue
        out.setdefault(r["band"], []).append(r)
    return out


def report_leg(name, obj, fh=sys.stdout):
    rows = obj["rows"]                      # scored rows only (finite, comparable)
    allr = obj.get("all_rows", rows)
    meta = obj["meta"]
    seed, cond = meta["seed"], meta["condition"]
    main = [r for r in rows if r.get("grid", "main") == "main" and r["arm"] == "field"]
    p = lambda *a: print(*a, file=fh, flush=True)  # noqa: E731

    p(f"\n{'='*78}\nLEG {name}   condition={cond} seed={seed} "
      f"corners={meta['n_corners']} rows={meta['n_rows']}")
    p(f"  field sha {meta['field_sha'][:16]}  ledger "
      f"{meta['byte_ledger']['deployed_bytes']:,} B  "
      f"matches_budget={meta['byte_ledger']['matches_budget']}")
    p(f"  integrator check max|diff| = {meta['integrator_check']['max_abs_diff']:.3e} "
      f"(PASSED={meta['integrator_check']['passed']})")
    hf = meta.get("hessian_fd_check", {})
    if hf.get("n"):
        p(f"  Hessian+D14 finite-difference max|resid| = {hf['max_abs_residual']:.3e}"
          f"   [B6]")
    tp = meta["sigma_provenance"]
    p("  TRAP CHECK well_sigma/nn_median: " + ", ".join(
        f"w{k.split('_')[1]}={v['well_sigma_over_nn_median']:.3f}" for k, v in tp.items())
      + "   (1.000 would mean the knn_density length scale was inherited)")

    # ---- B1 / B2 -----------------------------------------------------------
    est = meta["estimand"]["corners"]
    if est:
        ec = np.array([c["median_err_crr"] for c in est])
        ef = np.array([c["median_err_fit1"] for c in est])
        d = ec - ef
        p(f"\n  [B1] median over {len(est)} corners of |log10 n_meas/n_pred_CRR| "
          f"= {np.median(ec):.4f}   BAR <= 0.10   "
          f"{'PASS' if np.median(ec) <= 0.10 else 'MISS'}")
        p(f"       (factor {10**np.median(ec):.3f}x at the median corner; "
          f"IQR {np.percentile(ec,25):.4f}-{np.percentile(ec,75):.4f})")
        p(f"  [B2] paired vs FIT1 (n = C/(eps mu)^2, C fitted on these rows, "
          f"C = {meta['fit1_C']:.4g}):")
        p(f"       mean delta = {d.mean():+.4f} (BAR < 0)   "
          f"corners with delta<0: {int((d<0).sum())}/{len(d)}   BAR >= 40   "
          f"{'PASS' if (d.mean()<0 and (d<0).sum()>=40) else 'MISS'}")
        p(f"       median |log10 err|: CRR {np.median(ec):.4f}  vs  "
          f"FIT1 {np.median(ef):.4f}")
        # A SECOND one-parameter rival, because FIT1 (a pure power law) cannot express
        # the floor and would be a straw man on its own: FIT2 is a fitted CONSTANT,
        # which is the best one-parameter description of the underdamped branch.
        lg2 = np.array([np.log10(r["n_meas"]) for r in main])
        C2 = float(10.0 ** lg2.mean())
        byc2 = {}
        for r in main:
            byc2.setdefault(r["corner_id"], []).append(
                abs(np.log10(r["n_meas"] / C2)))
        e2 = np.array([np.median(v) for v in byc2.values()])
        d2 = ec - e2
        p(f"  [B2b] paired vs FIT2 (n = C constant, C = {C2:.4g}):  "
          f"mean delta = {d2.mean():+.4f}   corners better: {int((d2<0).sum())}/{len(d2)}")

        res = [r for r in main if r["n_pred_crr"] >= 10.0]
        if res:
            byc = {}
            for r in res:
                byc.setdefault(r["corner_id"], []).append(r["err_crr_log10"])
            m = np.array([np.median(v) for v in byc.values()])
            p(f"  [B1-resolved] same, restricted to cells with n_pred >= 10 "
              f"(quantisation < 0.04): median over {len(m)} corners = {np.median(m):.4f}"
              f"   (context, never the bar)")

    # ---- B3: the floor -----------------------------------------------------
    bt = band_table(main)
    p(f"\n  bands (live, main grid, field): " +
      ", ".join(f"{k}={len(v)}" for k, v in sorted(bt.items())) +
      f"   censored={sum(1 for r in allr if r.get('arm')=='field' and r.get('censored'))}")
    und = bt.get("underdamped", [])
    if len(und) >= 8:
        x = np.log10([r["mu2"] for r in und])
        y = np.log10([r["n_meas"] for r in und])
        # curvature-independence must be judged at FIXED gamma (the floor depends on gamma)
        slopes = []
        for g in sorted({r["gamma"] for r in und}):
            s = [(np.log10(r["mu2"]), np.log10(r["n_meas"])) for r in und if r["gamma"] == g]
            if len(s) >= 4 and len({a for a, _ in s}) >= 3:
                a = np.array(s)
                slopes.append(float(np.polyfit(a[:, 0], a[:, 1], 1)[0]))
        fl = np.array([abs(np.log10(r["n_meas"] / r["n_pred_floor"])) for r in und])
        if slopes:
            p(f"  [B3] underdamped d log10 n_meas / d log10 mu^2, per gamma: "
              f"median {np.median(slopes):+.4f}  max|.| {np.max(np.abs(slopes)):.4f}"
              f"   BAR <= 0.10   "
              f"{'PASS' if np.max(np.abs(slopes))<=0.10 else 'MISS'}")
        p(f"       |log10 n_meas / (2ln2/-ln(1-gamma))| median = {np.median(fl):.4f}"
          f"   BAR <= 0.10   {'PASS' if np.median(fl)<=0.10 else 'MISS'}")

    # ---- B4 / B5: the V-curve ---------------------------------------------
    vcs = obj.get("v_curves", {}).get("field", [])
    if vcs:
        br = [v for v in vcs if v.get("bracketed")]
        p(f"\n  V-curves: {len(vcs)} ({len(br)} with gamma_crit bracketed by the grid)")
        if br:
            a = np.array([v["argmin_over_gamma_crit"] for v in br])
            p(f"  [B4] measured argmin / gamma_crit (D16: gamma_crit = 2 eps mu): "
              f"median {np.median(a):.4f}  IQR {np.percentile(a,25):.3f}-"
              f"{np.percentile(a,75):.3f}   BAR within factor 2   "
              f"{'PASS' if 0.5 <= np.median(a) <= 2.0 else 'MISS'}")
        for side, lab in (("over", "D17 overdamped  "), ("under", "D18 underdamped")):
            ep = np.array([v[f"{side}_err_pred"] for v in vcs if f"{side}_err_pred" in v])
            en = np.array([v[f"{side}_err_naive"] for v in vcs if f"{side}_err_naive" in v])
            sm = np.array([v[f"{side}_slope_meas"] for v in vcs if f"{side}_slope_meas" in v])
            sp = np.array([v[f"{side}_slope_pred"] for v in vcs if f"{side}_slope_pred" in v])
            if ep.size:
                p(f"  [B5] {lab}: n={ep.size}  slope measured {np.median(sm):+.4f}  "
                  f"derived {np.median(sp):+.4f}  naive {'+1' if side=='over' else '-1'}")
                p(f"       mean |meas - derived| = {ep.mean():.4f}   "
                  f"mean |meas - naive| = {en.mean():.4f}   "
                  f"{'DERIVED WINS' if ep.mean() < en.mean() else 'NAIVE WINS'}")

    # ---- the two instruments: V2 Fig.1 says they diverge above the crossover ----
    both = [r for r in main if np.isfinite(r.get("n_meas_first", np.inf))
            and np.isfinite(r["n_meas"]) and r["n_meas_first"] > 0]
    if both:
        p("")
        for b in ("overdamped", "underdamped"):
            rs = [r for r in both if r["band"] == b]
            if len(rs) < 5:
                continue
            ratio = np.array([r["n_meas"] / r["n_meas_first"] for r in rs])
            ef = np.array([abs(np.log10(r["n_meas_first"] / r["n_pred_crr"])) for r in rs])
            ee = np.array([r["err_crr_log10"] for r in rs])
            p(f"  instruments {b:12s} n={len(rs):5d}  envelope/first-crossing median "
              f"= {np.median(ratio):8.3f}   |log10 err| envelope {np.median(ee):.4f} "
              f"vs first-crossing {np.median(ef):.4f}")
        p("       (V2 Fig. 1: the two coincide overdamped and DIVERGE above the "
          "exceptional point)")

    # ---- controls ----------------------------------------------------------
    # ⚠ READ THIS BEFORE READING THE TABLE.  The CRR is a law about the DYNAMICS, not
    # about the content: it holds of any landscape with a well-defined Hessian at a
    # critical point.  So a control is NOT expected to make the law fail, and a control
    # that matches the law is not a defect.  What the controls collapse is the CURVATURE
    # SPECTRUM -- which is the store's contribution -- and that is what this table shows
    # beside the law's accuracy.  Charter C.2 is satisfied by the spectrum column, not
    # by the error column.
    p("")
    p("  arm                     n   median|log10 CRR err|   hess_eig range     "
      "   modes  degenerate")
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm and r.get("grid", "main") == "main"]
        if not rs:
            continue
        e = np.array([r["err_crr_log10"] for r in rs])
        he = np.array([r["hess_eig_raw"] for r in rs])
        deg = bool(rs[0].get("degenerate_spectrum", False))
        nm = len({(r["well"], r["mode"]) for r in rs})
        p(f"  {arm:22s} {len(rs):5d} {np.median(e):21.4f}   "
          f"[{he.min():.4f}, {he.max():.4f}]{'':4s} {nm:5d}  {str(deg)}")
    p("       (landscape_removed: Cov_w = 0, so E = ||q-x*||^2/2 exactly and every")
    p("        direction carries the SAME trivial curvature -- the store contributes")
    p("        no structure at all.  That is the charter C.2 collapse, as a measurement.)")

    # ---- amplitude: where the law stops ------------------------------------
    aux = [r for r in rows if r.get("grid") == "aux" and r["arm"] == "field"]
    if aux:
        p("")
        for amp in sorted({r["amp"] for r in aux}):
            for al in sorted({r["alpha"] for r in aux}):
                rs = [r for r in aux if r["amp"] == amp and r["alpha"] == al]
                if len(rs) < 5:
                    continue
                e = np.array([r["err_crr_log10"] for r in rs])
                so = np.median([r["amp"] / r["sigma"] for r in rs])
                p(f"  aux amp={amp:<5g} (amp/sigma~{so:.3f}) alpha={al:<4g} "
                  f"n={len(rs):4d}  median|log10 CRR err| = {np.median(e):.4f}")
    return {"leg": name, "seed": seed, "condition": cond,
            "B1_median_err_crr": float(np.median([c["median_err_crr"] for c in est])) if est else None,
            "B2_mean_delta": float(np.mean([c["median_err_crr"] - c["median_err_fit1"]
                                            for c in est])) if est else None,
            "B2_corners_better": int(sum(1 for c in est
                                         if c["median_err_crr"] < c["median_err_fit1"])) if est else 0,
            "n_corners": len(est)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    legs = load(a.dirs)
    summ = [report_leg(n, o) for n, o in legs]
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(summ, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
