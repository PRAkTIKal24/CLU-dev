"""Build `GATE-HARDENING-DONE.json` — the mechanical gate C2W11 waits on.

`gate_hardening_done` is computed as the AND over every item's `done`.
⛔ Anything that could not land is `false` with its reason, never omitted and
never quietly true.
"""
import json
import subprocess

WT = "/Users/user/Desktop/CHLU-c2w8close"
OUT = ("/Users/user/Desktop/CHLU/.claude/outputs/c2w8-close/"
       "GATE-HARDENING-DONE.json")

commits = subprocess.run(
    ["git", "-C", WT, "log", "--format=%h %s", "9e0bb25..HEAD"],
    capture_output=True, text=True).stdout.strip().splitlines()

items = {
    "i_two_sided_drift_leg_or_floor": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`chlu.core.well_lifecycle.drift_leg` + `census`: G-DRIFT is now "
            "TWO-SIDED — PASS iff floor <= median(site_drift) < ceiling, both "
            "fractions of the MEASURED codebook spacing. drift -> 0 now FAILS "
            "(D2a, table-expressible) instead of scoring perfectly; the two "
            "failure modes are named apart and `one_sided_pass2_pass` is "
            "emitted beside them."),
        "floor_rule": "0.01 x measured codebook spacing (never a bare constant)",
        "floor_derivation": (
            "registered in PREREG.md §1 BEFORE the code: the banked D2a "
            "signature (settle = same-keys kNN to ±0.0007 against a codebook "
            "spacing of order 0.14) puts an already-adjudicated "
            "table-expressible store at ~0.005 x spacing; the floor is set 2x "
            "above that measured point"),
        "designed_negative_pytest_asserted": (
            "tests/test_well_lifecycle.py::"
            "test_designed_negative_table_like_store_fails_the_two_sided_drift_leg"),
        "designed_negative_measured": {
            "planted_table_like_store_ratio": 0.0042,
            "floor": 0.01,
            "leg_verdict": "FAIL (fails_low_D2a_table_expressible)",
            "one_sided_pass2_rule_on_the_same_store": "PASS — the defect",
        },
        "tests": [
            "tests/test_well_lifecycle.py::test_designed_negative_table_like_store_fails_the_two_sided_drift_leg",
            "tests/test_well_lifecycle.py::test_the_two_sided_drift_leg_still_fails_on_the_high_side",
            "tests/test_well_lifecycle.py::test_the_drift_floor_is_a_fraction_of_a_MEASURED_spacing_not_a_constant",
        ],
    },
    "ii_A1_margin_in_se_beside_the_boolean": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`gate_addr` A1 now emits margin_in_se_vs_chance, "
            "margin_in_se_vs_threshold, n_correct, n_correct_needed and "
            "reads_to_flip; `gate_addr_verdict` emits them per seed; the "
            "experiment's per-seed line prints margin + reads-to-flip beside "
            "the boolean."),
        "why": ("`randconv` scored 31 / 31 / 29 of 128 against a 32/128 "
                "threshold — it failed by ONE read on 2 of 3 seeds"),
        "tests": [
            "tests/test_gate_addr.py::test_a1_reports_its_margin_and_reads_to_flip_beside_the_boolean",
            "tests/test_gate_addr.py::test_gate_addr_emits_the_margin_everywhere_a1_is_emitted",
        ],
    },
    "iii_full_state_coscaling_scale_guard_asserts_the_verdict": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`well_lifecycle.scale_guard` — pass condition is VERDICT "
            "STABILITY (every leg boolean identical at every declared scale); "
            "metric movement is a reported DIAGNOSTIC, never the pass "
            "condition. `LEGAL_RESCALE` + "
            "`exp_well_lifecycle.full_state_coscaled_config` implement the "
            "ratified rescale: address AND payload together (address-only is "
            "NOT a symmetry — the payload channel is absolute)."),
        "defect_repaired": (
            "the old §4 guard bounded the METRIC and not the VERDICT: a legal "
            "rescale moved A1 by +0.0469 (inside the registered 0.05 bound, so "
            "it HELD) and still flipped the leg's verdict False -> True"),
        "designed_negative_measured": {
            "banked_pair_A1": [0.24219, 0.28906],
            "abs_delta_A1": 0.04688,
            "old_metric_bound": 0.05,
            "metric_bounded_DIAGNOSTIC": True,
            "verdict_stable": False,
            "repaired_guard": "FAIL (the old guard PASSED this exact pair)",
        },
        "tests": [
            "tests/test_gate_addr.py::test_scale_guard_fails_on_the_banked_metric_bounded_verdict_flip",
            "tests/test_gate_addr.py::test_scale_guard_passes_only_when_every_leg_boolean_is_stable",
            "tests/test_gate_addr.py::test_the_legal_rescale_is_full_state_co_scaling_including_the_payload",
            "tests/test_gate_addr.py::test_scale_only_control_does_not_move_gate_addr",
        ],
    },
    "iv_covered_and_n_never_read_are_launch_point_statistics": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`CluSystem._read_diagnostics` emits `launch_covered` (the "
            "launch-point test, relabelled) AND `settle_covered` (the same "
            "half-min-separation test at the settled address); `covered` stays "
            "as a deprecated alias of `launch_covered` because monitor "
            "`settle_argmin` needs the launch-side U for Prop D1. "
            "`usage_telemetry.attach_reads` now gates the telemetry on "
            "`settle_covered`, so n_never_read / frac_never_read stop "
            "inheriting the launch-point defect. Captions corrected in the "
            "telemetry summary and in the census's G-ADDR block "
            "(`telemetry_launch_side` / `telemetry_settle_side`)."),
        "designed_negative_pytest_asserted": (
            "tests/test_well_lifecycle.py::"
            "test_designed_negative_launch_coverage_is_store_invariant_settle_is_not"),
        "designed_negative_measured": {
            "store_mutated_so_reads_land_differently": True,
            "launch_covered_before_after": "8/8 -> 8/8 (BIT-IDENTICAL)",
            "settle_covered_before_after": "8/8 -> 0/8 (MOVED)",
        },
        "tests": [
            "tests/test_well_lifecycle.py::test_designed_negative_launch_coverage_is_store_invariant_settle_is_not",
            "tests/test_well_lifecycle.py::test_never_read_telemetry_is_gated_on_the_settle_side",
        ],
    },
    "v_d_safe_population_fix": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`soft_certificate.population_median_nn` estimates the NN spacing "
            "AT THE STORE'S OWN POPULATION SIZE by subsampling the sizing "
            "pool, returning the sizing-set value beside it; "
            "`run_census_cell` sizes d_safe on it "
            "(`d_safe_population = 'store'`, default; 'sizing' restores the "
            "pre-close-out behaviour exactly)."),
        "before_after_measured": {
            "cell": ("toy regression rig (the census's own pytest stream), 3 "
                     "seeds — ⛔ a REGRESSION cell, not a science cell"),
            "spacing_ratio_population_over_sizing": [1.5666, 1.4351, 1.8426],
            "d_safe_sizing": [0.0276, 0.0407, 0.0264],
            "d_safe_store": [0.0433, 0.0584, 0.0486],
            "refusal_rate_sizing": [0.0, 0.25, 0.0],
            "refusal_rate_store": [0.25, 0.25, 0.1429],
            "direction": ("rose on 2/3 seeds, equal on 1, fell on 0 — as "
                          "registered (PREREG §3 P4/P5)"),
            "discipline": ("⚠ the refusal rate is REPORTED, never tuned to a "
                           "target"),
            "pass_3_prior_relief": ("pass 3 already ran 0.000-0.111 with 5/9 "
                                    "cells non-zero; this removes the "
                                    "arithmetic cause rather than the symptom"),
        },
        "tests": [
            "tests/test_well_lifecycle.py::test_d_safe_population_spacing_is_larger_than_the_sizing_set_spacing",
        ],
    },
    "vi_1_own_foreign_site_depth_kernel": {
        "done": True,
        "label": "DIAGNOSTIC (own/foreign is never a gate leg)",
        "what_changed": (
            "VERIFIED ON DISK, already landed at pass 3: "
            "`own_foreign_site_depth` reads the store's OWN `atom_profile` with "
            "the store's kernel/cutoff/axis_width_scale instead of hard-coding "
            "the Gaussian. The designed CROSS-KERNEL test exists and asserts "
            "that a compact (wendland) foreign contribution is EXACTLY 0 while "
            "the Gaussian form invents a tail."),
        "tests": [
            "tests/test_gate_addr.py::test_own_foreign_reads_the_compact_kernel_and_the_gaussian_form_over_reads",
            "tests/test_gate_addr.py::test_cross_kernel_over_read_factor_is_reported_not_hidden",
            "tests/test_gate_addr.py::test_numpy_atom_profile_mirrors_the_shipped_one_for_every_kernel",
            "tests/test_gate_addr.py::test_own_foreign_matches_the_legacy_gaussian_form",
        ],
    },
    "vi_2_theta_att_P_needs_n_non_capturing": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`census` now emits `P_comparability` (n_non_capturing, theta_att, "
            "theta_att_degenerate, P_comparable_across_arms + the rule) "
            "alongside every `P`, and the experiment's per-seed line prints "
            "n_non_capturing beside P. ENFORCED IN THE EMITTER: a reader can no "
            "longer obtain P without it."),
        "tests": [
            "tests/test_well_lifecycle.py::test_P_is_never_emitted_without_the_theta_att_degeneracy_qualifier",
        ],
    },
    "vi_3_errata_pass2_numbering_collision": {
        "done": True,
        "label": "housekeeping (verify only — NOT re-resolved)",
        "what_changed": (
            "VERIFIED ON DISK AND RECORDED CLOSED, exactly as the Hub's account "
            "states. `.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS2.md` "
            "carries §1, §2 (wt3's K9 block, the EARLIER one, untouched), then "
            "§5 — wt1's later block, renumbered at pass-2 integration under a "
            "dated HUB RENUMBERING banner that states the reason (wt3's §2 is "
            "cited by hash-stable references in tracked code) and the "
            "chronology. Exactly one §2 exists. ⛔ Nothing re-resolved."),
        "my_check_agrees_with_the_hub_account": True,
        "tests": [],
    },
    "vi_4_stale_x64_comment_in_test_cifar_strong_phi": {
        "done": True,
        "label": "housekeeping (verify only)",
        "what_changed": (
            "VERIFIED ON DISK: already corrected at pass 3 (rider 4c). "
            "`tests/test_cifar_strong_phi.py` lines 66-76 now state the "
            "`backbone = 'mlp'` choice is a COST choice and that the x64 dtype "
            "reason is STALE, naming the fix commit 42b781c (present in this "
            "history) and the coverage that replaced it "
            "(tests/test_cl_baselines_x64.py, present). No edit needed."),
        "tests": [],
    },
    "vi_5_census_refuses_a_non_selected_width": {
        "done": True,
        "label": "MECHANICS",
        "what_changed": (
            "`exp_well_lifecycle.UnselectedAtomWidth`: the census recovers the "
            "atom width it is about to run at as a fraction of the cell's own "
            "MEASURED key spacing and raises, loudly and by name, unless that "
            "fraction either equals the census's own un-substituted default or "
            "matches a fraction EXPLICITLY DECLARED in config "
            "(`experiment_well_lifecycle.atom_width_selection` takes absolute "
            "priority, else the arm configs' own "
            "`atom_width_frac_spacing`). The resolved source and the effective "
            "fraction ride on every cell (`atom_width_selection` block)."),
        "tests": [
            "tests/test_well_lifecycle.py::test_census_refuses_to_run_at_an_unselected_atom_width",
            "tests/test_well_lifecycle.py::test_the_width_guard_can_be_switched_off_but_is_on_by_default",
        ],
    },
    "vi_6_cue_difficulty_arm_dependence": {
        "done": True,
        "label": "MECHANICS",
        "which_fix": (
            "THE FULL FIX (declared): kappa_q is normalised on the CODEBOOK "
            "spacing — the resolution the read must actually beat — not on the "
            "~200-key sizing spacing (`gaddr_spacing_population = 'codebook'`, "
            "default; 'sizing' restores the banked behaviour)."),
        "what_changed": (
            "AND the minimum is met unconditionally: "
            "`cue_sigma_over_codebook_spacing` AND "
            "`cue_sigma_over_sizing_spacing` are emitted on EVERY cell either "
            "way, so no future cross-arm comparison can be made blind to it."),
        "measured_on_the_regression_rig": {
            "cue_sigma_over_codebook_spacing": [1.0, 1.0, 1.0],
            "cue_sigma_over_sizing_spacing": [1.5659, 6.0136, 2.116],
            "reading": ("the sizing-normalised cue difficulty varies 3.8x "
                        "across seeds on this toy rig — the same defect class "
                        "as the 0.927 / 0.875 / 0.710 spread across pass-3 arms"),
        },
        "tests": [
            "tests/test_well_lifecycle.py::test_census_cell_runs_end_to_end_and_reports_both_curves",
        ],
    },
    "acceptance_3_A3_relabelled_DIAGNOSTIC_and_out_of_the_pass_condition": {
        "done": True,
        "label": "the relabel itself",
        "what_changed": (
            "charter §A33.1 (binding program-wide) + §A34.8: `gate_addr_pass` "
            "= A1 AND A2 (the MECHANICS legs). A3, the kNN-in-phi launder "
            "margin, is labelled DIAGNOSTIC, carries `in_pass_condition: "
            "False`, and is still measured and reported two-sided. Every leg "
            "this spoke touched carries an explicit MECHANICS / DIAGNOSTIC "
            "label (A1, A2, G-DRIFT two-sided = MECHANICS; A3, own/foreign = "
            "DIAGNOSTIC)."),
        "mechanical_consequence_declared": (
            "arm B's banked configuration failed G-ADDR ONLY on A3 (measured "
            "settle-side A1 = 0.9297/0.9844/0.8750 pass 3/3, A2 = "
            "0.0625/0.0/0.125 pass 3/3) ⇒ under the repaired instrument it "
            "PASSES the two MECHANICS legs. ⛔ Stated as a mechanical "
            "consequence of a ratified rule — NOT a re-scoring into a claim, "
            "NOT an arm-race adjudication (§A30.1: the race is VOID and stays "
            "unadjudicated; arm B is claim-barred by NO_TIER_II_CLAIM "
            "regardless)."),
        "tests": [
            "tests/test_gate_addr.py::test_the_launder_margin_is_a_diagnostic_and_cannot_decide_the_verdict",
            "tests/test_gate_addr.py::test_arm_b_banked_configuration_fails_only_on_the_diagnostic_leg",
        ],
    },
}

suite = {
    "checkout": "worktree ../CHLU-c2w8close on branch c2w8-close-gate-hardening",
    "base": "main @ 9e0bb25",
    "runner": ("/Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q "
               "--no-cov (the MAIN venv, per protocol §4)"),
    "collected_at_base_9e0bb25_fresh_worktree": 1564,
    "collected_on_this_branch": 1579,
    "delta": 15,
    "delta_explained": ("15 new tests: 11 added to tests/test_well_lifecycle.py "
                        "and 6 added to tests/test_gate_addr.py, minus 2 "
                        "rewritten-in-place (N1) — net +15 collected"),
    "note": ("⚠ counts are comparable only within one checkout; both numbers "
             "above were taken with the same interpreter on the same machine, "
             "the base one in a fresh detached worktree at 9e0bb25. The task "
             "file quoted 1555 for that base; this measurement is 1564 "
             "collected — reported as measured, not reconciled by assumption."),
}

out = {
    "artifact": "GATE-HARDENING-DONE.json",
    "wave": "C2W8 CLOSE-OUT",
    "spoke": "c2w8-close-gate-hardening",
    "authority": ("charter §A31.5-§A31.6 (instrument debts), §A32.3 (gate "
                  "hardening owners), §A33.1 (MECHANICS/VALUE rule), §A34.8 "
                  "(G-ADDR barred from VALUE duty)"),
    "dial_declaration": {
        "dial": "none — instrument repair",
        "laundering_control": ("N/A, and that is the point: A3 is now a "
                               "DIAGNOSTIC column, never a pass condition "
                               "(§A33.1)"),
        "falsifies": ("a repaired leg that cannot fail its own designed "
                      "negative does not ship"),
        "not_a_verdict": ("⛔ no claim cell, no performance number, no "
                          "re-scoring of banked results into new claims, no "
                          "pass 4, no arm-race adjudication"),
    },
    "base_commit": "9e0bb25",
    "branch": "c2w8-close-gate-hardening",
    "commits": commits,
    "items": items,
    "suite": suite,
    "declared_not_runs": [
        "NO re-run of any banked arm (arm A / arm B / the spine's 9 cells) — "
        "consequences of the repairs on banked numbers are stated as mechanical "
        "consequences, never re-scored",
        "NO merge / prune / restoration verb (deferred)",
        "NO pass 4, no daylight chase, no tier-ii / full-CLU verdict",
        "NO edit to chlu/experiments/exp_capture_strong_phi.py, "
        "exp_capture_armA/B.py or emission_head.py (banked arms, read-only)",
    ],
    "gate_hardening_rule": ("mechanical AND over every item's `done`; anything "
                           "that could not land is false with its reason"),
}
out["gate_hardening_done"] = bool(all(v["done"] for v in items.values()))
out["items_done"] = {k: bool(v["done"]) for k, v in items.items()}

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"gate_hardening_done": out["gate_hardening_done"],
                  "items_done": out["items_done"],
                  "n_commits": len(commits)}, indent=1))
