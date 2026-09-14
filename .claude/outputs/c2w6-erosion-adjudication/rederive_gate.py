#!/usr/bin/env python
"""§2 gate re-derivation + §Q1 headline estimator, from the RAW curves only.

Independent of the engineer's `aggregate()`: every scalar is recomputed from
`records[*].curve` (the per-reading series), never read from the record's
pre-computed `depth_*` fields — those are then compared digit-for-digit.

Run: /Users/user/Desktop/CHLU/.venv/bin/python \
       .claude/outputs/c2w6-erosion-adjudication/rederive_gate.py   (cwd=repo root)
"""
import glob
import json
import os

import numpy as np

AE = ".claude/outputs/c2w6-anti-erosion/"
CELLS = ["p1_off", "p1_on", "p1_on_i1_on", "w40_p1_off", "w40_p1_on",
         "resoff_p1_off", "resoff_p1_on"]


def load():
    recs = []
    for f in sorted(glob.glob(AE + "erosion_*_records.json")):
        if os.sep + "smoke" + os.sep in f:
            continue
        recs += json.load(open(f))["records"]
    return recs


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def geo(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean()))


def curve_scalars(r):
    """Recompute depth_untrained / at_200 / final / ratios from the raw curve."""
    c = r["curve"]
    steps = np.array([x["at_step"] for x in c], float)
    dep = np.array([x["depth_median"] for x in c], float)
    i200 = int(np.argmin(np.abs(steps - 200.0)))
    return {
        "n_readings": len(c),
        "step_first": steps[0], "step_last": steps[-1],
        "step_nearest_200": steps[i200],
        "d_unt": dep[0], "d_200": dep[i200], "d_fin": dep[-1],
        "r_fin_200": dep[-1] / dep[i200] if dep[i200] > 0 else float("nan"),
        "r_fin_unt": dep[-1] / dep[0] if dep[0] > 0 else float("nan"),
        "n_live_first": c[0]["n_live"], "n_live_last": c[-1]["n_live"],
    }


def main():
    recs = load()
    print("n_records =", len(recs), " cells:",
          sorted({r["cell"] for r in recs}))
    by = {(r["cell"], int(r["seed"])): r for r in recs}

    # ---------- 1. digit-for-digit check of the stored scalars ----------
    print("\n=== 1. raw-curve recomputation vs the record's stored scalars ===")
    worst = 0.0
    for k in sorted(by):
        r = by[k]
        s = curve_scalars(r)
        pairs = [("depth_untrained", s["d_unt"]), ("depth_at_200", s["d_200"]),
                 ("depth_final", s["d_fin"]),
                 ("depth_ratio_1000_over_200", s["r_fin_200"]),
                 ("depth_ratio_final_over_untrained", s["r_fin_unt"])]
        for name, mine in pairs:
            stored = float(r[name])
            rel = abs(mine - stored) / max(abs(stored), 1e-300)
            worst = max(worst, rel)
            if rel > 1e-12:
                print("  MISMATCH", k, name, mine, stored, rel)
    print("  worst relative deviation over %d records x 5 scalars: %.2e"
          % (len(by), worst))

    # readings / horizons actually present
    print("\n  cell            seeds  n_read  first..last  nearest200  nlive0->N")
    for c in CELLS:
        for sd in (0, 1, 2):
            if (c, sd) in by:
                s = curve_scalars(by[(c, sd)])
                print("  %-14s s%d    %3d    %4.0f..%-5.0f  %5.0f       %d->%d"
                      % (c, sd, s["n_readings"], s["step_first"],
                         s["step_last"], s["step_nearest_200"],
                         s["n_live_first"], s["n_live_last"]))

    # ---------- 2. E1 / E2 / E3 / K3 / K4 ----------
    print("\n=== 2. gate legs, re-derived (per seed, then arith and geo) ===")
    for c in CELLS:
        rs = [by[(c, s)] for s in (0, 1, 2) if (c, s) in by]
        if not rs:
            continue
        sc = [curve_scalars(r) for r in rs]
        f200 = np.array([x["r_fin_200"] for x in sc])
        funt = np.array([x["r_fin_unt"] for x in sc])
        dfin = np.array([x["d_fin"] for x in sc])
        bpc = np.array([float(r["bpc_live"]) for r in rs])
        print("  %-14s final/200 %s | arith %.3f +- %.3f | GEO %.3f"
              % (c, np.array2string(f200, precision=4), f200.mean(), se(f200),
                 geo(f200)))
        print("  %-14s final/unt %s | arith %.3f +- %.3f | GEO %.3f"
              % ("", np.array2string(funt, precision=4), funt.mean(), se(funt),
                 geo(funt)))
        print("  %-14s depth_fin %s | arith %.4f | GEO %.4f | bpc %.4f +- %.4f"
              % ("", np.array2string(dfin, precision=5), dfin.mean(), geo(dfin),
                 bpc.mean(), se(bpc)))

    print("\n=== 3. paired ON-OFF (bpc = K3/E3; depth = the protection claim) ===")
    for on, off in [("p1_on", "p1_off"), ("p1_on_i1_on", "p1_off"),
                    ("w40_p1_on", "w40_p1_off"), ("resoff_p1_on", "resoff_p1_off")]:
        seeds = [s for s in (0, 1, 2) if (on, s) in by and (off, s) in by]
        d = np.array([by[(on, s)]["bpc_live"] - by[(off, s)]["bpc_live"]
                      for s in seeds])
        rat = np.array([curve_scalars(by[(on, s)])["d_fin"]
                        / curve_scalars(by[(off, s)])["d_fin"] for s in seeds])
        lr = np.log(rat)
        print("  %-12s vs %-12s dbpc %+.6f +- %.6f (%.2f SE, %d/%d better) | "
              "depth ON/OFF %s geo %.3fx [%.2f,%.2f] (%d/%d >1)"
              % (on, off, d.mean(), se(d), abs(d.mean()) / se(d),
                 int((d < 0).sum()), len(d),
                 np.array2string(rat, precision=3), np.exp(lr.mean()),
                 np.exp(lr.mean() - se(lr)), np.exp(lr.mean() + se(lr)),
                 int((rat > 1).sum()), len(rat)))

    print("\n=== 4. K4 (relocation detector) legs, per cell-seed ===")
    for c in CELLS:
        for s in (0, 1, 2):
            if (c, s) not in by:
                continue
            r = by[(c, s)]
            print("  %-14s s%d |live-blank| %.3e | md_eval-live %+.5f | "
                  "none_retr-live %+.5f"
                  % (c, s, abs(float(r["bpc_live_minus_blank"])),
                     float(r["bpc_memory_deleted_minus_live"]),
                     float(r["bpc_none_minus_live"])))

    # ---------- 5. the headline estimator question (Hub Q1) ----------
    print("\n=== 5. Q1: the headline 'final/untrained' under both estimators ===")
    for c in ["p1_off", "p1_on", "w40_p1_off", "w40_p1_on", "resoff_p1_off"]:
        rs = [by[(c, s)] for s in (0, 1, 2) if (c, s) in by]
        funt = np.array([curve_scalars(r)["r_fin_unt"] for r in rs])
        lr = np.log(funt)
        print("  %-14s per-seed %s | arith %.3f +- %.3f | GEO %.4f "
              "[%.3f, %.3f] | n>1 %d/3 | log-mean/SE %.2f"
              % (c, np.array2string(funt, precision=4), funt.mean(), se(funt),
                 np.exp(lr.mean()), np.exp(lr.mean() - se(lr)),
                 np.exp(lr.mean() + se(lr)), int((funt > 1).sum()),
                 lr.mean() / se(lr)))

    # ---------- 6. compare with the engineer's aggregate() ----------
    print("\n=== 6. engineer aggregate() digit-for-digit ===")
    agg = json.load(open(AE + "erosion_aggregate.json"))
    print("  top keys:", list(agg.keys()))
    print("  n_records:", agg.get("n_records"))
    for k, v in agg.items():
        if isinstance(v, dict) and ("verdict" in v or "E1" in v):
            print("  ", k, json.dumps(v)[:900])


if __name__ == "__main__":
    main()
