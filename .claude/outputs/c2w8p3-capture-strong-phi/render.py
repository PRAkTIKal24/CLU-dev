"""Re-derive EVERY number in the report from `results/CAPTURE-STRONG-PHI.json`.

Nothing in the report is typed by hand; this script is the only source.

    python render.py results/CAPTURE-STRONG-PHI.json
"""
import json
import sys

import numpy as np

ARM_ORDER = ["simclr", "randconv", "pca"]


def _f(x, n=4):
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{n}f}"


def main(path):
    d = json.load(open(path))
    arms = d["arms"]
    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]
    seeds = d["seeds"]

    print("=" * 78)
    print("JOINT DIAL:", json.dumps(d["joint_dial_d_atom_budget"]))
    print("SEEDS:", seeds, " ARMS:", order, " WALL(s):", _f(d["wall_s"], 0))
    print("=" * 78)

    print("\n## FIRST-10-LINES (R1): the store's WELL DEPTH, reported FIRST")
    print(f"{'arm':<10} {'depth_raw_median by seed':<44} {'INERT?':<8} n_atoms")
    for a in order:
        r = arms[a]
        dep = r["depth_raw_median_by_seed"]
        na = [c["flags"]["n_atoms"] for c in r["cells"]]
        print(f"{a:<10} {str([round(float(x), 5) for x in dep]):<44} "
              f"{str(any(r['store_inert_by_seed'])):<8} {sorted(set(na))}")

    print("\n## THE COMPLETED GATE — all four legs, per seed")
    hdr = (f"{'arm':<10} {'seed':<5} {'G-CAP':<16} {'G-DEC':<22} "
           f"{'G-DRIFT':<20} {'A1':<16} {'A2':<14} {'A3a':<20} {'A3b':<20} GATE")
    print(hdr)
    for a in order:
        r = arms[a]
        for i, s in enumerate(r["seeds"]):
            lg = r["gate"]["legs_by_seed"][i]
            g = r["gate"]["g_addr_by_seed"][i]
            cap = f"{_f(lg['G_CAP']['frac_capture_positive'],3)} {lg['G_CAP']['pass']}"
            dec = (f"{_f(lg['G_DEC']['decode'],4)}v{_f(lg['G_DEC']['chance'],4)} "
                   f"{lg['G_DEC']['pass']}")
            dr = f"{_f(lg['G_DRIFT']['ratio'],4)} {lg['G_DRIFT']['pass']}"
            a1 = f"{_f(g['A1']['correct_basin_rate'],4)} {g['A1']['pass']}"
            a2 = f"{_f(g['A2']['never_addressed_frac'],4)} {g['A2']['pass']}"
            a3a = (f"{_f(g['A3']['A3a_cue_margin'],4)}±{_f(2*g['A3']['A3a_se_paired'],4)}"
                   f" {g['A3']['A3a_pass']}")
            a3b = (f"{_f(g['A3']['A3b_stream_margin'],4)}±"
                   f"{_f(2*(g['A3']['A3b_se_pooled'] or 0),4)} {g['A3']['A3b_pass']}")
            print(f"{a:<10} {s:<5} {cap:<16} {dec:<22} {dr:<20} {a1:<16} {a2:<14} "
                  f"{a3a:<20} {a3b:<20} {g['gate_addr_pass'] and lg['G_CAP']['pass'] and lg['G_DEC']['pass'] and lg['G_DRIFT']['pass']}")
        print(f"{'':<10} -> gate_pass={r['gate']['gate_pass']} "
              f"all_four_same_seed={r['gate']['all_four_same_seed']}/{r['gate']['n_seeds']}")

    print("\n## G-ADDR headline legs (A1/A2/A3), with the 'some basin' non-leg beside them")
    print(f"{'arm':<10} {'A1 (correct-basin)':<26} {'voronoi_only':<26} "
          f"{'any_basin (NOT the leg)':<26} {'launder rate':<26}")
    for a in order:
        r = arms[a]
        gs = r["gate"]["g_addr_by_seed"]
        print(f"{a:<10} "
              f"{str([round(float(g['A1']['correct_basin_rate']),4) for g in gs]):<26} "
              f"{str([round(float(g['A1']['voronoi_only_rate']),4) for g in gs]):<26} "
              f"{str([round(float(g['A1']['any_basin_rate']),4) for g in gs]):<26} "
              f"{str([round(float(g['A3']['A3a_launder_rate']),4) for g in gs]):<26}")

    print("\n## THE CUE, as a dimensionless ratio (PREREG §4 / my PREREG §1)")
    print(f"{'arm':<10} {'kappa_q':<9} {'cue_sigma/codebook_spacing':<28} "
          f"{'cue_displacement/spacing':<26} {'codebook_spacing':<20}")
    for a in order:
        g = arms[a]["gate"]["g_addr_by_seed"][0]
        print(f"{a:<10} {_f(g['kappa_q'],2):<9} "
              f"{_f(g['cue_sigma_over_codebook_spacing'],4):<28} "
              f"{_f(g['cue_displacement_over_codebook_spacing'],4):<26} "
              f"{_f(g['codebook_spacing'],5):<20}")

    print("\n## ⭐ THE BRANCH (computed by `daylight_verdict`, never argued)")
    for a in order:
        v = arms[a]["daylight"]
        print(f"{a:<10} {v['branch']:<18} "
              f"A3a+beyond2SE={v['n_seeds_A3a_positive_beyond_2se']}/{v['n_seeds']} "
              f"A3b+beyond2SE={v['n_seeds_A3b_positive_beyond_2se']}/{v['n_seeds']}")
    print("  rule:", arms[order[0]]["daylight"]["rule"])

    print("\n## ⚠ THE D2a DIAGNOSTIC — two-sided, never a target")
    print(f"{'arm':<10} {'agreement_rate (chance)':<30} "
          f"{'median|settle-launder key|/spacing':<36} {'G-DRIFT ratio':<24}")
    for a in order:
        r = arms[a]
        ag = [round(float(x["agreement_rate"]), 4) for x in r["d2a"]["by_seed"]]
        ch = _f(r["d2a"]["by_seed"][0]["agreement_chance"], 4)
        dd = [round(float(x["median_settle_to_launder_key_over_spacing"]), 4)
              for x in r["d2a"]["by_seed"]]
        dr = [round(float(lg["G_DRIFT"]["ratio"]), 4)
              for lg in r["gate"]["legs_by_seed"]]
        print(f"{a:<10} {str(ag) + ' (' + ch + ')':<30} {str(dd):<36} {str(dr):<24}")
    for a in order:
        c = arms[a]["d2a"]["cooccurrence"]
        print(f"  {a}: best_is_also_lowest_drift = {c.get('best_is_also_lowest_drift')} "
              f"(best={c.get('best_scoring_index')}, lowest_drift={c.get('lowest_drift_index')})")

    print("\n## THE BYTE LEDGER — φ + map on EVERY arm INCLUDING the launder")
    print(f"{'arm':<10} {'enc floats':<12} {'map floats':<12} {'clu bytes':<12} "
          f"{'launder bytes':<14} {'ratio (addr only)':<19} {'ratio (with φ)':<15}")
    for a in order:
        b = arms[a]["bytes_by_seed"][0]
        print(f"{a:<10} {b['phi_param_floats']:<12} {b['map_param_floats']:<12} "
              f"{b['clu_total_bytes']:<12} {b['knn_launder_bytes']:<14} "
              f"{_f(b['ratio_clu_over_knn_launder'],1):<19} "
              f"{_f(b['ratio_clu_over_knn_launder_with_phi'],4):<15}")
        assert (b["clu_total_bytes_with_phi"] - b["clu_total_bytes"]
                == b["knn_launder_bytes_with_phi"] - b["knn_launder_bytes"]), \
            "the φ term must be the SAME number on the store row and the launder row"
    print("  matching:", arms[order[0]]["bytes_by_seed"][0]["matching"],
          "| matched_bytes:", arms[order[0]]["bytes_by_seed"][0]["matched_bytes"])

    print("\n## R2(b) LAUNDER AUDIT (asserted in code, re-read here)")
    n_ok = sum(1 for a in arms for x in arms[a]["launder_audit_by_seed"]
               if x.get("launder_reads_projected_phi")
               and x.get("bit_identical_to_store_addresses"))
    n_tot = sum(len(arms[a]["launder_audit_by_seed"]) for a in arms)
    print(f"  {n_ok}/{n_tot} cells: launder_reads_projected_phi = "
          f"bit_identical_to_store_addresses = true")
    for a in order:
        x = arms[a]["launder_audit_by_seed"][0]
        print(f"  {a}: launder_key_dim={x['launder_key_dim']} "
              f"store_address_dim={x['store_address_dim']} "
              f"phi_dim_before_map={x['phi_dim_before_map']}")

    print("\n## PAIRING EVIDENCE — the stream fingerprint, per seed, per arm")
    for a in order:
        print(f"  {a}: {arms[a]['stream_fingerprint_by_seed']}")

    print("\n## THE GEOMETRY EACH CELL MEASURED (spacing, d_safe, scale)")
    print(f"{'arm':<10} {'median_nn_task1':<34} {'d_safe':<30} {'phi_scale':<24}")
    for a in order:
        gg = arms[a]["geometry_by_seed"]
        print(f"{a:<10} {str([round(float(x['median_nn_task1']),5) for x in gg]):<34} "
              f"{str([round(float(x['d_safe']),5) for x in gg]):<30} "
              f"{str([round(float(x['phi_scale']),4) for x in gg]):<24}")

    print("\n## THE STREAM (admission / refusal — monitor #3)")
    print(f"{'arm':<10} {'n_admitted':<16} {'n_refused':<14} {'refusal_rate':<24} "
          f"{'overdig':<16} n_live")
    for a in order:
        st = arms[a]["stream_by_seed"]
        print(f"{a:<10} {str([x['n_admitted'] for x in st]):<16} "
              f"{str([x['n_refused'] for x in st]):<14} "
              f"{str([round(float(x['refusal_rate']),4) for x in st]):<24} "
              f"{str([round(float(x['overdig']),2) for x in st]):<16} "
              f"{[x['n_live_end'] for x in st]}")

    print("\n## SELF-PROBE (G-DEC's ingredient) and the store flags actually used")
    for a in order:
        sp = arms[a]["self_probe_by_seed"]
        print(f"  {a}: acq={[round(float(x['acq']),4) for x in sp]} "
              f"strict={[round(float(x['strict']),4) for x in sp]} "
              f"decode={[round(float(x['decode']),4) for x in sp]}")
    print("  store flags (seed 0, first arm):",
          json.dumps(arms[order[0]]["store_flags_by_seed"][0]))

    print("\n## WALL, per arm")
    for a in order:
        print(f"  {a}: {_f(arms[a]['wall_s'],0)} s  "
              f"(per cell: {[round(float(c['wall_s'])) for c in arms[a]['cells']]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "results/CAPTURE-STRONG-PHI.json")
