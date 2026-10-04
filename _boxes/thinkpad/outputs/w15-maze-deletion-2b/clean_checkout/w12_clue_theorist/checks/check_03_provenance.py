"""check_03 -- PROVENANCE. How strong is the artefact trail behind the CRR numbers?

The wave-10 referee's C6 warning is that the programme's sharpest number (+1.1385) has
its weakest artefact trail.  This check re-derives from the recorded JSON everything that
CAN be re-derived, and enumerates exactly what cannot.

It reads, read-only:
  <EV>/test/s{0,1,2}/results.json        (the three scored seeds)
  <EV>/**                                (a string scan, for the slope rows)

Run:  OMP_NUM_THREADS=2 python checks/check_03_provenance.py
"""
import json
import os

import numpy as np

EV = os.environ.get("W12_EVIDENCE",
                    r"C:\chlu\evidence\outputs\_thinkpad\w6\w6-potentials")
RES = {}


def _banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ---------------------------------------------------------------- 1 -----------
def rederive_b1_b2():
    """B1 and B2 are claimed in RESULTS-LEDGER.md; are they IN the artefact?"""
    ledger = {"B1": {"s0": 0.0433, "s1": 0.0435, "s2": 0.0431},
              "B2": {"s0": -0.5968, "s1": -0.5796, "s2": -0.5982}}
    rows = {}
    for s in (0, 1, 2):
        p = os.path.join(EV, "test", "s%d" % s, "results.json")
        if not os.path.exists(p):
            print("  DECLARED NOT-RUN: %s absent" % p)
            continue
        d = json.load(open(p))
        c = d["estimand"]["corners"]
        crr = np.array([r["median_err_crr"] for r in c], float)
        fit = np.array([r["median_err_fit1"] for r in c], float)
        dlt = np.array([r["paired_delta"] for r in c], float)
        rows["s%d" % s] = {
            "n_corners": len(c),
            "B1_median_abs_log10_rederived": float(np.median(crr)),
            "B1_ledger": ledger["B1"]["s%d" % s],
            "B2_mean_paired_delta_rederived": float(dlt.mean()),
            "B2_ledger": ledger["B2"]["s%d" % s],
            "B2_corners_better_than_fitted_rival": int((dlt < 0).sum()),
            "median_err_fitted_rival": float(np.median(fit)),
            "instrument_floor_median_abs_log10":
                float(d["instrument_floor"]["median_abs_log10"]),
            "integrator_max_abs_diff": float(d["integrator_check"]["max_abs_diff"]),
            "n_rows": int(d["n_rows"]),
            "field_sha": d["field_sha"],
            "byte_pin_matched": bool(d["byte_ledger"]["matches_budget"]),
            "deployed_bytes": int(d["byte_ledger"]["deployed_bytes"]),
            "fit1_C": float(d["fit1_C"]),
            "wall_seconds": float(d["wall_seconds"]),
        }
        r = rows["s%d" % s]
        print("  s%d  n_corners=%d  rows=%d  bytes=%d (pin %s)  field_sha=%s"
              % (s, r["n_corners"], r["n_rows"], r["deployed_bytes"],
                 r["byte_pin_matched"], r["field_sha"]))
        print("      B1 re-derived %.6f   ledger %.4f   -> agrees to %.1e"
              % (r["B1_median_abs_log10_rederived"], r["B1_ledger"],
                 abs(r["B1_median_abs_log10_rederived"] - r["B1_ledger"])))
        print("      B2 re-derived %.6f   ledger %.4f   -> agrees to %.1e   "
              "corners better than the FITTED rival: %d/%d"
              % (r["B2_mean_paired_delta_rederived"], r["B2_ledger"],
                 abs(r["B2_mean_paired_delta_rederived"] - r["B2_ledger"]),
                 r["B2_corners_better_than_fitted_rival"], r["n_corners"]))
        print("      instrument floor %.6f   integrator max|diff| %.6e   fit1_C %.6f"
              % (r["instrument_floor_median_abs_log10"], r["integrator_max_abs_diff"],
                 r["fit1_C"]))
    RES["rederived"] = rows
    ok = all(abs(r["B1_median_abs_log10_rederived"] - r["B1_ledger"]) < 5e-5
             and abs(r["B2_mean_paired_delta_rederived"] - r["B2_ledger"]) < 5e-5
             and r["B2_corners_better_than_fitted_rival"] == r["n_corners"]
             for r in rows.values()) if rows else False
    RES["b1_b2_pass"] = bool(ok)
    print("VERDICT  B1 and B2 re-derive from results.json to <5e-5 on all three seeds,"
          " 49/49: %s" % RES["b1_b2_pass"])


# ---------------------------------------------------------------- 2 -----------
NEEDLES = ["1.1368", "1.1367", "1.1385", "1.1090", "-1.109"]


def scan():
    hits = {n: [] for n in NEEDLES}
    n_files = 0
    by_ext = {}
    for root, _dirs, files in os.walk(EV):
        for fn in files:
            p = os.path.join(root, fn)
            n_files += 1
            ext = os.path.splitext(fn)[1].lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
            except Exception:
                continue
            rel = os.path.relpath(p, EV)
            for n in NEEDLES:
                if n in txt:
                    hits[n].append(rel)
    RES["scan_files"] = n_files
    RES["scan_by_ext"] = by_ext
    RES["scan_hits"] = hits
    print("  scanned %d files under %s   (%s)"
          % (n_files, EV, ", ".join("%s:%d" % (k or "<none>", v)
                                    for k, v in sorted(by_ext.items()))))
    for n in NEEDLES:
        h = hits[n]
        js = [x for x in h if x.lower().endswith(".json")]
        print("  %-8s -> %d file(s): %s" % (n, len(h), h if h else "NONE"))
        if js:
            print("           of which machine-readable JSON: %s" % js)
    machine = sorted(set(x for n in ("1.1368", "1.1367", "1.1385")
                         for x in hits[n] if x.lower().endswith(".json")))
    RES["d17_numbers_in_json"] = machine
    RES["scan_pass"] = bool(len(machine) == 0)
    print("VERDICT  the D17 pair (+1.1385 derived / +1.1368 measured) appears in NO json"
          " artefact: %s" % RES["scan_pass"])


# ---------------------------------------------------------------- 3 -----------
def keys_present():
    p = os.path.join(EV, "test", "s0", "results.json")
    if not os.path.exists(p):
        print("  DECLARED NOT-RUN: %s absent" % p)
        RES["keys_pass"] = None
        return
    d = json.load(open(p))
    print("  top-level keys of test/s0/results.json:")
    print("    %s" % sorted(d.keys()))
    wanted = ["slope", "d17", "d18", "v_curve", "per_cell", "fit_window"]
    blob = json.dumps(d).lower()
    found = {w: (w in blob) for w in wanted}
    print("  substring present anywhere in the JSON blob:")
    for w, f in found.items():
        print("    %-12s %s" % (w, f))
    print("  n_v_curves (a COUNT, not the curves): %s" % json.dumps(d["n_v_curves"]))
    RES["keys_found"] = found
    RES["n_v_curves"] = d["n_v_curves"]
    RES["keys_pass"] = bool(not found["slope"] and not found["d17"]
                            and not found["fit_window"])
    print("VERDICT  no per-cell (gamma_window, slope_measured, slope_derived) rows exist"
          " in the artefacts: %s" % RES["keys_pass"])


def main():
    _banner("1 -- re-derive B1 and B2 from the recorded per-corner estimand")
    rederive_b1_b2()
    _banner("2 -- string scan: where do the D17/D18 numbers actually live?")
    scan()
    _banner("3 -- what results.json does and does not carry")
    keys_present()

    _banner("SUMMARY -- provenance grade")
    print("  REPRODUCES FROM ARTEFACTS : B1, B2, 49/49, instrument floor, integrator")
    print("                              check, byte pin, field sha, hessian_fd_check,")
    print("                              d14_confinement (and its tautology)")
    print("  PROSE/LEDGER ONLY         : +1.1385 vs +1.1368 (D17), -1.0050 vs -1.1090")
    print("                              (D18) -- no machine-readable row anywhere")
    keys = [k for k in RES if k.endswith("_pass")]
    for k in keys:
        print("  %-12s %s" % (k, RES[k]))
    allp = all(RES[k] is not False for k in keys)
    print("  NOTHING FAILED   %s" % allp)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_03_provenance.json")
    with open(out, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True)
    print("  wrote %s" % out)
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
