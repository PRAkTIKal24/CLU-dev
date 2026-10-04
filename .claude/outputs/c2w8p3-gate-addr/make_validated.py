"""Assemble GATE-ADDR-VALIDATED.json — the wave's mechanical precondition.

`gate_addr_validated` is computed MECHANICALLY from the designed controls; the
arm re-scores are carried as measurements and never enter the boolean.
"""
import glob
import json
import os
import sys

OUT = sys.argv[1]
D = os.path.dirname(OUT)
ctrl = json.load(open(os.path.join(D, "designed_controls.json")))

rescores = {}
for p in (sorted(glob.glob(os.path.join(D, "rescore", "*.json")))
          + sorted(glob.glob(os.path.join(D, "attribution", "*.json")))):
    r = json.load(open(p))
    key = os.path.splitext(os.path.basename(p))[0]
    g = r["g_addr"]
    rescores[key] = {
        "arm": r["arm"], "seed": r["seed"], "addr_scale_mult": r["addr_scale_mult"],
        "atom_width_frac_spacing": r.get("atom_width_frac_spacing"),
        "payload_scale_override": r.get("payload_scale_override"),
        "A1": g["A1"]["correct_basin_rate"], "A1_threshold": g["A1"]["threshold"],
        "A1_pass": g["A1"]["pass"],
        "A1_voronoi_only": g["A1"]["voronoi_only_rate"],
        "A1_any_basin": g["A1"]["any_basin_rate"],
        "A1_in_own_basin": g["A1"]["in_own_basin_rate"],
        "A2": g["A2"]["never_addressed_frac"], "A2_pass": g["A2"]["pass"],
        "A3a": g["A3"]["A3a_cue_margin"], "A3a_2se": 2 * g["A3"]["A3a_se_paired"],
        "A3a_pass": g["A3"]["A3a_pass"],
        "A3a_launder_rate": g["A3"]["A3a_launder_rate"],
        "A3b": g["A3"]["A3b_stream_margin"],
        "A3b_2se": (2 * g["A3"]["A3b_se_pooled"]
                    if g["A3"]["A3b_se_pooled"] is not None else None),
        "A3b_pass": g["A3"]["A3b_pass"],
        "gate_addr_pass": g["gate_addr_pass"],
        "banked_telemetry_frac_never_read": g["banked_telemetry"]["frac_never_read"],
        "banked_telemetry_n_unassigned": g["banked_telemetry"]["n_unassigned"],
        "codebook_spacing": g["codebook_spacing"],
        "spacing_ref": g["spacing_ref"],
        "cue_sigma_over_codebook_spacing": g["cue_sigma_over_codebook_spacing"],
        "pass2_gate": {k: r["pass2_gate_legs"][k]["pass"]
                       for k in ("G_CAP", "G_DEC", "G_DRIFT")},
        "own_foreign_repaired": r["own_foreign_repaired_estimator"],
        "theta_att": r["theta_att"]["theta_att"],
        "theta_att_n_non_capturing": r["theta_att"]["n_non_capturing"],
        "wall_s": r["wall_s"],
        "source": os.path.relpath(p, D),
    }

# ---- the mechanical validation -------------------------------------------
cp = ctrl["C_plus_positive"]
n2 = ctrl["N2_permutation"]
n1p = ctrl["N1prime_narrow_wells"]
s08, s125 = ctrl["S_scale_0.8"], ctrl["S_scale_1.25"]
armB = [v for v in rescores.values() if v["arm"] == "armB" and v["addr_scale_mult"] == 1.0]
armA = [v for v in rescores.values() if v["arm"] == "armA" and v["addr_scale_mult"] == 1.0]
base = {(v["arm"], v["seed"]): v for v in rescores.values()
        if v["addr_scale_mult"] == 1.0 and v["payload_scale_override"] is None}


def _pair(v, kind):
    b = base[(v["arm"], v["seed"])]
    return {
        "kind": kind, "arm": v["arm"], "seed": v["seed"],
        "scale": v["addr_scale_mult"],
        "payload_scale_override": v["payload_scale_override"],
        "A1_base": b["A1"], "A1_scaled": v["A1"], "delta_A1": v["A1"] - b["A1"],
        "A3a_base": b["A3a"], "A3a_scaled": v["A3a"],
        "delta_A3a": v["A3a"] - b["A3a"],
        "launder_base": b["A3a_launder_rate"], "launder_scaled": v["A3a_launder_rate"],
        "cue_over_codebook_base": b["cue_sigma_over_codebook_spacing"],
        "cue_over_codebook_scaled": v["cue_sigma_over_codebook_spacing"],
        "A1_base_saturated": bool(b["A1"] in (0.0, 1.0)),
    }


# (a) GENUINELY scale-covariant rescalings: the planted rigs (whose payload is
#     co-scaled with the geometry by construction) and the arm-A cell in which
#     the payload channel was explicitly co-scaled (ERRATA §3, cell S-pay).
covariant_pairs = [_pair(v, "address+payload co-scaled")
                   for v in rescores.values()
                   if v["payload_scale_override"] is not None]
# (b) ADDRESS-ONLY rescalings: the literal reading of the §4 control. The rig's
#     payload channel is ABSOLUTE, so these are not pure rescalings of the
#     geometry; reported as a finding, not as a validation check.
address_only_pairs = [_pair(v, "address-only (rig payload stays absolute)")
                      for v in rescores.values()
                      if v["addr_scale_mult"] != 1.0
                      and v["payload_scale_override"] is None]
planted_delta = [
    abs(ctrl[k]["A1"]["correct_basin_rate"] - cp["A1"]["correct_basin_rate"])
    for k in ("S_scale_0.8", "S_scale_1.25")]

checks = {
    "C_plus_positive_passes": bool(cp["gate_addr_pass"] and
                                   cp["A1"]["correct_basin_rate"] >= 0.80),
    "N1_armB_banked_config_fails": bool(len(armB) >= 3 and
                                        all(not v["gate_addr_pass"] for v in armB)),
    "N1prime_narrow_wells_fail": bool(not n1p["gate_addr_pass"]),
    "N2_planted_permutation_scores_zero_and_fails": bool(
        n2["A1"]["correct_basin_rate"] <= 0.02 and not n2["gate_addr_pass"]),
    "S_planted_scale_only_moves_A1_by_leq_0p05": bool(all(d <= 0.05
                                                         for d in planted_delta)),
    "S_real_rig_scale_covariant_rescaling_moves_A1_by_leq_0p05": bool(
        len(covariant_pairs) >= 1
        and all(abs(p["delta_A1"]) <= 0.05 for p in covariant_pairs)),
    "S_leg_machinery_is_exactly_scale_covariant": bool(
        all(abs(p["launder_scaled"] - p["launder_base"]) <= 1e-12
            # 1e-6, not 0: the phi scale and the spacing are computed in
            # float32, so the ratio agrees to ~4e-8 rather than bitwise. The
            # LAUNDER rate above is bit-identical, which is the strong statement.
            and abs(p["cue_over_codebook_scaled"]
                    - p["cue_over_codebook_base"]) <= 1e-6
            for p in address_only_pairs
            if p["arm"] == "armA")),
    "S_has_at_least_one_NON_SATURATED_real_rig_covariant_pair": bool(
        any(not p["A1_base_saturated"] for p in covariant_pairs)),
    "R3_attractor_can_move_off_the_key": bool(
        all(v["attractor_can_move"] for v in ctrl["R3_counterfactual"].values())),
}
validated = bool(all(checks.values()))

doc = {
    "artifact": "GATE-ADDR-VALIDATED.json",
    "wave": "C2W8 pass 3",
    "spoke": "c2w8p3-gate-addr (experiment-engineer)",
    "authority": ("charter §A30.1 + PREREG-C2W8-PASS3 §2; thresholds registered in "
                  ".claude/outputs/c2w8p3-gate-addr/PREREG.md §1 as amended by its "
                  "ERRATA §1, both filed before the cells they govern"),
    "gate_addr_validated": validated,
    "validation_checks": checks,
    "validation_rule": ("gate_addr_validated = AND over every check above; the arm "
                        "re-scores are MEASUREMENTS and do not enter the boolean "
                        "except through designed negative N1 (arm B must fail)"),
    "dial_declaration": {
        "dial": "none — instrument construction + one compliance counterfactual",
        "laundering_control": ("A3 IS the launder margin; every quotation states "
                               "matched-ITEMS, and the 1 253x byte ratio travels"),
        "falsifies": "a G-ADDR that cannot fail its designed negatives does not ship",
        "does_not_falsify": ("losing to the kNN launder on the metric-native cue "
                             "protocol (1-NN is Bayes-optimal there): the "
                             "metric-native-ceiling theorem, not news"),
        "no_arm_race_adjudication": ("§A30.1: the pass-2 race is VOID as a "
                                     "comparison and STAYS UNADJUDICATED here"),
    },
    "legs": {
        "A1": ("correct-basin rate: the settle resolves to the QUERIED item AND "
               "lies inside that item's MEASURED capture radius; threshold "
               "max(4 x chance, chance + 2 SE)"),
        "A2": ("never-addressed fraction: live items with zero correct cue reads; "
               "threshold <= 0.5. NOT the banked telemetry n_never_read, which is "
               "a LAUNCH-POINT coverage statistic (ERRATA §2)"),
        "A3a": ("cue launder margin vs 1-NN-in-phi on the same queries, same "
                "decision rule; pass iff >= -2 SE (paired/McNemar)"),
        "A3b": ("stream launder margin mean(read_acc - knn_acc) on held-out data; "
                "pass iff >= -2 SE (pooled binomial); declared NOT-APPLICABLE, "
                "never a null, where there is no stream"),
        "matching": "matched-ITEMS on both A3 legs; matched-bytes is NOT met",
    },
    "designed_controls": ctrl,
    "scale_control": {
        "registered": ("PREREG-C2W8-PASS3 §4 / this spoke's PREREG.md §2 row S: "
                       "identical phi, address scale x a declared constant => "
                       "|dA1| <= 0.05 (Hub Q8, prior 0.90)"),
        "covariant_pairs": covariant_pairs,
        "address_only_pairs": address_only_pairs,
        "planted_delta_A1": planted_delta,
        "rig_scale_noninvariance": (
            "⛔ DECLARED FINDING, not a leg failure. Under an ADDRESS-ONLY "
            "rescaling arm A's A1 moves (0.5000 -> 0.3750 at a=0.8, -> 0.6328 at "
            "a=1.25) because the rig's PAYLOAD channel is ABSOLUTE (|a_i| <= 0.5) "
            "while the compact support R = cutoff*s co-scales: the rescaling walks "
            "arm A across its own payload wall (its report's §5 mechanism). The "
            "SAME rescaling moves the PASS-2 legs too (self-probe acq 0.4844 -> "
            "0.3203, G-DEC 0.1484 -> 0.1094, G-DRIFT ratio 0.0071 -> 0.0273, x3.8). "
            "Co-scaling the payload restores A1 to 0.5000 and A3a to -0.3984, both "
            "to 4 dp (ERRATA §3 cell S-pay, prediction registered first). The "
            "leg's own comparator (the 1-NN launder) is bit-identical across the "
            "rescaling on arm A: 0.8984375 at both scales."),
        "hub_decision_owed": (
            "under a STRICT reading of §4 ('if rescaling moves G-ADDR it does not "
            "ship') G-ADDR does not ship — but on the same evidence neither do "
            "G-DEC and G-DRIFT, which move MORE. The Hub owns that call; both "
            "numbers are in this file."),
    },
    "arm_rescores": rescores,
    "declared_not_runs": [
        "the arm-A-vs-arm-B RACE ADJUDICATION — VOID as a comparison (§A30.1)",
        "merge / prune / restoration / any §2.7 claim cell",
        "any tier-ii, full-CLU or I2 verdict; any performance claim",
        "monitor #3's refusal-rate defect — reported, NOT fixed here",
        "the phi_dim -> addr_dim projection (wt2), the geometry precondition (wt2), "
        "the spine (wt3)",
    ],
}
with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(checks, indent=2))
print("gate_addr_validated =", validated)
print("wrote", OUT)
