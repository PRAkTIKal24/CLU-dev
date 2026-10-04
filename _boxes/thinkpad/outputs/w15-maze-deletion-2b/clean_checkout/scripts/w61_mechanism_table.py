"""w6.1 retry -- render every mechanism number into one markdown table file.

Pure rendering: it reads the JSON/NPZ this hub's three analysis scripts wrote and
writes `MECHANISM-TABLE.md`.  Nothing is recomputed and nothing is scored.

    python scripts/w61_mechanism_table.py --mech DIR --out FILE.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEGS = ("pca_tensor", "pca_settle_zerostep", "pca_settle_f05_conv",
        "pca_settle_onestep", "pca_settle")
NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="C:/chlu/outputs/w61-compose-physics/mech")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = Path(a.mech)
    sub = [json.loads((m / f"subspace_seed{s}.json").read_text("utf-8"))
           for s in range(3)]
    lat = [json.loads((m / f"lattice_seed{s}.json").read_text("utf-8"))
           for s in range(3)]
    mar = json.loads((m / "margin_val_seed0.json").read_text("utf-8"))
    clo = json.loads((m / "close.json").read_text("utf-8"))
    L = []
    w = L.append

    w("# w6.1/compose-physics — MECHANISM TABLES")
    w("")
    w("Every number here is recomputed from the parent's already-paid artifacts, except")
    w("the judge-margin block, which is a `val`-only probe. Nothing is scored.")
    w("")
    w("## 1. The declared hypothesis: does the settled state leave `span(G_k)`?")
    w("")
    w("| seed | leg | validity | median rel. displacement | off-span fraction of the "
      "STATE (max) | off-span fraction of the DISPLACEMENT (max) | AUC(off-span -> "
      "invalid) |")
    w("|---|---|---|---|---|---|---|")
    for s in range(3):
        for leg in LEGS:
            b = sub[s]["legs"][leg]
            w(f"| {s} | `{leg}` | {b['validity']:.4f} | "
              f"{b['median_rel_displacement']:.4f} | "
              f"{b['max_off_span_fraction_of_state']:.2e} | "
              f"{b['max_off_span_fraction_of_displacement']:.2e} | "
              f"{b['auc_off_span_predicts_invalid']:.3f} |")
    w("")
    w("**Verdict: the hypothesis is false.** The maximum off-span fraction is ~2e-7 for")
    w("every leg *including the incumbent, which never moves* — it is float32 rounding")
    w("in `render_span`. The orthogonal fraction predicts invalidity at chance.")
    w("")

    w("## 2. Displacement against the answer lattice")
    w("")
    w("| seed | median state norm | rho_1 (nearest answer, any one factor) | "
      "rho_1 / norm |")
    w("|---|---|---|---|")
    for s in range(3):
        d = sub[s]
        w(f"| {s} | {d['median_state_norm']:.3f} | {d['median_rho1']:.4f} | "
          f"{d['median_rho1'] / d['median_state_norm']:.4f} |")
    w("")
    w("| seed | leg | median displacement | displacement / rho_1 | frac > 0.5 rho_1 | "
      "median tangent fraction | mean wrong factors |")
    w("|---|---|---|---|---|---|---|")
    for s in range(3):
        for leg in LEGS:
            b = sub[s]["legs"][leg]
            t = lat[s]["legs"][leg]
            w(f"| {s} | `{leg}` | {b['median_abs_displacement']:.4f} | "
              f"{b['median_displacement_over_rho1']:.4f} | "
              f"{b['frac_displacement_over_half_rho1']:.3f} | "
              f"{t['median_tangent_fraction_of_displacement']:.3f} | "
              f"{t['mean_wrong_factors']:.3f} |")
    w("")
    w("An isotropic step would place sqrt(51/308) = 0.407 of its length in the")
    w("rank-51 lattice tangent. The settle places 0.78-0.86.")
    w("")

    w("## 3. Per-factor lattice spacing and per-factor damage")
    w("")
    for s in range(3):
        sp = lat[s]["median_lattice_spacing_per_factor"]
        w(f"### seed {s}")
        w("")
        w("| factor | cardinality | rho_f | " +
          " | ".join(f"`{l}`" for l in LEGS) + " |")
        w("|---|---|---|" + "---|" * len(LEGS))
        for f in range(6):
            cells = " | ".join(
                f"{lat[s]['legs'][l]['error_rate_per_factor'][f]:.4f}" for l in LEGS)
            w(f"| {NAMES[f]} | {lat[s]['factor_cardinalities'][f]} | {sp[f]:.3f} | "
              f"{cells} |")
        w("")

    w("## 4. The judge's margin (val only), in lattice units")
    w("")
    w("P(the read-back of factor f changes) after moving s * rho_f toward the nearest")
    w("answer that differs only in factor f. 288 val requests, 15,552 judged states.")
    w("")
    w("| factor | " + " | ".join(f"s={x:g}" for x in mar["scales_in_rho_units"]) + " |")
    w("|---|" + "---|" * len(mar["scales_in_rho_units"]))
    for f in range(6):
        w(f"| {NAMES[f]} | " +
          " | ".join(f"{x:.3f}" for x in mar["margin_cdf_directed"][str(f)]) + " |")
    w("")
    w("Isotropic control: the same states moved a matched ABSOLUTE distance in a random")
    w("in-span direction.")
    w("")
    w("| factor | " + " | ".join(f"L={x:g}" for x in mar["random_control_lengths"]) +
      " |")
    w("|---|" + "---|" * len(mar["random_control_lengths"]))
    for f in range(6):
        w(f"| {NAMES[f]} | " +
          " | ".join(f"{x:.3f}" for x in mar["margin_cdf_isotropic"][str(f)]) + " |")
    w("")
    w("| | " + " | ".join(f"L={x:g}" for x in mar["random_control_lengths"]) + " |")
    w("|---|" + "---|" * len(mar["random_control_lengths"]))
    w(f"| whole-tuple validity (incumbent {mar['whole_tuple_valid_incumbent']:.4f}) | " +
      " | ".join(f"{x:.4f}" for x in mar["whole_tuple_valid_isotropic"]) + " |")
    w("")

    w("## 5. The headline, recomputed from `rows.npz`")
    w("")
    w("Row-4 novel valid tuples per corner, and the paired delta against `pca_tensor`.")
    w("")
    w("| leg | seed 0 | seed 1 | seed 2 | delta s0 | delta s1 | delta s2 |")
    w("|---|---|---|---|---|---|---|")
    for leg in LEGS:
        per = [clo["per_seed"][str(s)][leg] for s in range(3)]
        w(f"| `{leg}` | " +
          " | ".join(f"{p['novel_per_corner_mean']:.3f}" for p in per) + " | " +
          " | ".join(f"{p.get('delta_novel_per_corner', 0.0):+.3f}" for p in per) +
          " |")
    w("")
    w("Paired over all 147 corner pairs (49 corners x 3 seeds) and all 10,584 requests:")
    w("")
    w("| leg | mean delta/corner | corners worse / better / tied | sign test p | "
      "requests valid->invalid | invalid->valid | McNemar p |")
    w("|---|---|---|---|---|---|---|")
    for leg, b in clo["pooled_over_3_seeds"].items():
        w(f"| `{leg}` | {b['mean_delta_novel_per_corner']:+.3f} | "
          f"{b['corners_worse']} / {b['corners_better']} / {b['corners_tied']} | "
          f"{b['sign_test_p_over_corners']:.3g} | "
          f"{b['requests_valid_to_invalid']} | {b['requests_invalid_to_valid']} | "
          f"{b['mcnemar_p']:.3g} |")
    w("")
    ext = json.loads((m / "extras.json").read_text("utf-8"))
    evi = json.loads((m / "eviction_seed0.json").read_text("utf-8"))

    w("## 6. Can ANY leg leave span(G_k)?  (val, includes the channel-space kernel leg)")
    w("")
    w("| leg | median off-span fraction | max | median displacement | validity |")
    w("|---|---|---|---|---|")
    for leg, b in ext["A_can_any_leg_leave_the_span"].items():
        w(f"| `{leg}` | {b['median_off_span_fraction']:.3e} | "
          f"{b['max_off_span_fraction']:.3e} | {b['median_displacement']:.4f} | "
          f"{b['validity']:.4f} |")
    w("")
    w(ext["A_note"])
    w("")
    w("## 7. The alternatives, and the motion budget")
    w("")
    w(f"- standardise/un-standardise round trip, no steps taken: "
      f"**{ext['B_standardisation_round_trip_relative']:.3e}** relative")
    c = ext["C_lattice_tangent"]
    w(f"- lattice tangent: rank {c['median_rank']} of {c['n_directions']} directions "
      f"in a {c['state_dimension']}-dimensional state; an isotropic step would place "
      f"{c['isotropic_expected_tangent_fraction']:.3f} of its length there, the settle "
      f"places 0.78-0.86")
    w("")
    w("| seed | slope (validity per unit displacement/rho_1) | budget in rho_1 | "
      "budget absolute | budget as fraction of the state norm |")
    w("|---|---|---|---|---|")
    for s_, b in ext["E_motion_budget"].items():
        w(f"| {s_} | {b['slope_validity_per_unit_displacement_over_rho1']:.4f} | "
          f"{b['budget_in_rho1_units']:.3f} | {b['budget_absolute']:.4f} | "
          f"{b['budget_as_fraction_of_state_norm']:.4f} |")
    w("")
    w("## 8. Regression to seen combinations, on the ERRORS, against a local null")
    w("")
    w("| seed | leg | n wrong | mean Hamming | seen (observed) | seen (locality null) "
      "| ratio |")
    w("|---|---|---|---|---|---|---|")
    for s_, per in ext["F_regression_to_seen_combinations"].items():
        for leg, b in per.items():
            if leg == "grid_seen_fraction":
                continue
            w(f"| {s_} | `{leg}` | {b['n_wrong']} | "
              f"{b['mean_hamming_of_a_wrong_readback']:.3f} | "
              f"{b['seen_fraction_observed']:.4f} | "
              f"{b['seen_fraction_locality_matched_null']:.4f} | {b['ratio']:.3f} |")
    w("")
    w("## 9. What evicting ONE atom would cost, in the same units (seed 0)")
    w("")
    w(f"rho_1 = {evi['rho1']:.4f}, half = {evi['half_rho1']:.4f}, motion budget = "
      f"{evi['motion_budget']:.4f}")
    w("")
    w("| well | d_k | atoms | median eviction cost | cost of evicting the mean | "
      "past half rho_1 | past the budget |")
    w("|---|---|---|---|---|---|---|")
    for b in evi["wells"]:
        w(f"| {b['well']} | {b['d_k']} | {b['n_atoms']} | "
          f"{b['median_displacement_per_evicted_atom_median']:.4g} | "
          f"{b['atom_0_is_the_mean_median_displacement']:.4g} | "
          f"{b['atoms_whose_eviction_moves_the_median_request_past_half_rho1']} | "
          f"{b['atoms_past_the_motion_budget']} |")
    w("")
    for k, v in evi["all_atoms"].items():
        w(f"- `{k}` = {v:,}" if isinstance(v, int) else f"- `{k}` = {v:.4f}")
    w("")
    w("## 10. Row 3 against row 4: the metric is the validity rate, rescaled")
    w("")
    w("| seed | leg | row3 delta novel | row3 delta validity x 36 | row4 delta novel | "
      "row4 delta validity x 72 |")
    w("|---|---|---|---|---|---|")
    for s_, per in ext["D_row3_vs_row4"].items():
        for leg, b in per.items():
            w(f"| {s_} | `{leg}` | {b['row3']['delta_novel']:+.3f} | "
              f"{b['row3']['delta_validity_times_ceiling']:+.3f} | "
              f"{b['row4']['delta_novel']:+.3f} | "
              f"{b['row4']['delta_validity_times_ceiling']:+.3f} |")
    w("")
    w("## 11. The read-side byte ledger")
    w("")
    for k, v in clo["byte_price"].items():
        w(f"- `{k}` = {v:,}" if isinstance(v, int) else f"- `{k}` = {v:.4f}")
    w("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
