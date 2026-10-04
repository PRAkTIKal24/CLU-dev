#!/usr/bin/env python
"""w13 patch 2/4 -- TABLE 7 (wave 12's scored rows 3 and 4 at K* = 32) and
TABLE 8 (the n-way ladder, n = 2..6, at K = 5).  Exact-string patch."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")

NEW = r'''
# --------------------------------------------------------------------------------------
# 8w. w13 -- WAVE 12'S ROWS.  T7..T12.  Every row states BOTH n and K.
#     ZERO scored runs here: these re-read wave 12's artefacts.
# --------------------------------------------------------------------------------------
X_ARMS = [
    ("cov_tensor", "**`cov_tensor` -- THE ARM** (single design, K = 32)", "RESULT"),
    ("landscape_removed", "`landscape_removed`", "control"),
    ("landscape_shuffled", "`landscape_shuffled`", "control"),
    ("landscape_shuffled_across_wells", "`landscape_shuffled_across_wells`", "control"),
    ("random_atoms", "`random_atoms`", "control"),
    ("mlp__as_trained", "`mlp` (524,288 B reference)", "comparator @524,288 B"),
    ("lstm__as_trained", "`lstm` (524,288 B reference)", "comparator @524,288 B"),
    ("transformer__as_trained", "`transformer` (524,288 B reference)", "comparator @524,288 B"),
    ("cont_mlp__as_trained", "`cont_mlp` (524,288 B reference)", "comparator @524,288 B"),
    ("table", "`table` -- the FLOOR, never a rival", "floor"),
]

_t7_rows = []
for _arm, _label, _cls in X_ARMS:
    _cells = {
        "row 3 - 60 pair corners": w1("x_scored1", "/per_arm/%s/row3_pair/novel_per_corner_mean" % _arm, "{:.3f}"),
        "row 4 - 49 n-way corners": w1("x_scored1", "/per_arm/%s/row4_nway/novel_per_corner_mean" % _arm, "{:.3f}"),
        "all 109 corners": w1("x_scored1", "/per_arm/%s/all/novel_per_corner_mean" % _arm, "{:.3f}"),
        "judge accuracy": w1("x_scored1", "/per_arm/%s/all/judge_accuracy_mean" % _arm, "{:.4f}"),
    }
    if _arm == "cov_tensor":
        _cells["paired delta vs the arm"] = derived(
            ["-- (this IS the arm)"], [dsrc("x_scored1", "/per_arm/cov_tensor/all/novel_per_corner_mean")],
            "the arm is not compared with itself", fmt="{}")
        _cells["CI95 of that delta"] = derived(
            ["--"], [dsrc("x_scored1", "/per_arm/cov_tensor/all/novel_per_corner_mean")], "n/a", fmt="{}")
        _cells["corners the arm wins"] = derived(
            ["--"], [dsrc("x_scored1", "/n_corners/row3_pair")], "n/a", fmt="{}")
        _cells["params @ the 524,288 B pin"] = derived(
            ["NOT A PARAMETER MODEL - see the byte ledger"],
            [dsrc("x_scored1", "/ledger_four_part/honest_total_bytes")],
            "the arm is a store; its size is the four-part ledger, not a parameter count", fmt="{}")
    else:
        _cells["paired delta vs the arm"] = w1(
            "x_scored1", "/paired_vs_cov_tensor/%s/mean_paired_delta" % _arm, "{:+.4f}")
        _cells["CI95 of that delta"] = wcell(
            [("x_scored1", "/paired_vs_cov_tensor/%s/bootstrap_ci95/0" % _arm),
             ("x_scored1", "/paired_vs_cov_tensor/%s/bootstrap_ci95/1" % _arm)],
            fmt="{:+.3f}", seeds=["lo", "hi"], stats=False)
        _cells["corners the arm wins"] = w1(
            "x_scored1", "/paired_vs_cov_tensor/%s/corners_arm_wins" % _arm, "{:.0f}")
        if _arm.endswith("__as_trained"):
            _cells["params @ the 524,288 B pin"] = w1(
                "x_scored1", "/rival_meta/%s/param_count" % _arm, "{:,.0f}")
        else:
            _cells["params @ the 524,288 B pin"] = derived(
                ["--"], [dsrc("x_scored1", "/baseline_matching/reference_column_bytes")],
                "a control or a lookup floor carries no matched parameter count", fmt="{}")
    _t7_rows.append({"label": _label, "row_class": _cls, "seeds_printed": [0], "n_seeds_printed": 1,
                     "cells": _cells})

TABLES["T7"] = {
    "title": ("TABLE 7 (MAIN) - Shapes3D value-holdout, the SINGLE DESIGN at K* = 32, TEST: "
              "rows 3 and 4. 1 scored seed, DECLARED."),
    "table_class": "RESULT",
    "placement": "MAIN TEXT",
    "environment": "Shapes3D value-holdout condition (NOT a maze: never in a table with T1 or T3)",
    "notation": {
        "K_wells": 32,
        "n_factors": ("row 3 = pair corners (2 factors pinned); row 4 = n-way corners (all six pinned). "
                      "This table varies neither: it is ONE n-set at ONE K."),
        "rule": ("T7 states K = 32 on every row; the ladder that varies n is T8 and every one of its "
                 "rows sits at K = 5."),
    },
    "pins": {"condition": "test", "K (wells)": 32, "A_requested": 63, "seed": 0,
             "metric": "distinct_valid_novel_tuples per corner",
             "estimand": "fw.paired_estimand, the CORNER as the unit",
             "margin declared in advance": 1.0,
             "feature cache": "generation 2, scope full, 480,000 rows",
             "address fit seconds": 1242.3163497447968},
    "pin_provenance": [
        dsrc("x_scored1", "/K_star"), dsrc("x_scored1", "/A_requested"), dsrc("x_scored1", "/seed"),
        dsrc("x_scored1", "/condition"), dsrc("x_scored1", "/n_corners/row3_pair"),
        dsrc("x_scored1", "/n_corners/row4_nway"), dsrc("x_scored1", "/address_fit_seconds"),
        dsrc("x_scored1", "/feature_cache/generation"), dsrc("x_scored1", "/rank_over_width_max"),
        dsrc("x_scored1", "/SCORED"),
    ],
    "seed_count": {
        "scored seed id": w1("x_scored1", "/seed", "{:.0f}"),
        "PRINTED AS A COLUMN, NOT A FOOTNOTE": "n = 1 (seed 0). Every row of T7 is a 1-seed row.",
        "why not 3": ("the crossover hub spent run 1 of its 3 `single-design` scored runs on seed 0 at "
                      "K* = 32; runs 2/3 and 3/3 (seeds 1 and 2) are NR6, wall clock."),
        "why not 5": ("seeds 3 and 4 have generation-2 caches (NR7) but no frozen decoder, no external "
                      "classifier and no fitted rivals, so no corner of them can be judged."),
        "NEVER MIX": ("T7 (1 seed) must not share a plot with T8 (3 seeds) or T1/T2/T3 (3 seeds) unless "
                      "the seed count is printed per row."),
    },
    "rows": _t7_rows,
    "rank_width": {
        "rank/width max on the scored object": w1("x_scored1", "/rank_over_width_max", "{:.4f}"),
        "wave 9's decorative threshold": 0.938,
        "reading": ("0.9688 is ABOVE the 0.938 at which wave 9's maze arm was decorative, and the four "
                    "controls still read exactly 0.000. Only the control decides; rank/width is printed "
                    "because it is the warning it is."),
    },
    "byte_ledger": {
        "rule": "FOUR PARTS, PRINTED, NEVER NETTED (w12r/MF3). Seed 0, K* = 32, TEST.",
        "parts": {
            "1. atoms": w1("x_scored1", "/ledger_four_part/p1_atom_bytes", "{:,.0f}"),
            "2. covariance + count ledger": w1("x_scored1", "/ledger_four_part/p2_covariance_plus_count_bytes", "{:,.0f}"),
            "2a. of which covariance": w1("x_scored1", "/ledger_four_part/p2_covariance_bytes", "{:,.0f}"),
            "2b. of which count": w1("x_scored1", "/ledger_four_part/p2_count_bytes", "{:,.0f}"),
            "3. address book": w1("x_scored1", "/ledger_four_part/p3_address_book_bytes", "{:,.0f}"),
            "4. coupling / hole table": w1("x_scored1", "/ledger_four_part/p4_coupling_hole_table_bytes", "{:,.0f}"),
        },
        "totals": {
            "honest total (the sum of the four)": w1("x_scored1", "/ledger_four_part/honest_total_bytes", "{:,.0f}"),
            "x the 524,288 B reference": w1("x_scored1", "/ledger_four_part/times_reference_pin", "{:.4f}"),
        },
        "netted": False,
        "pin_status": ("the object is 4.0897x the 524,288 B figure. That figure is a REFERENCE COLUMN in "
                       "this table, never a budget the object met."),
    },
    "byte_matching": {
        "WHY BOTH COLUMNS ARE PRINTED": (
            "the two matchings tell OPPOSITE stories and a reader who sees one draws the opposite "
            "conclusion. See T8's caption for the sign flip on the ladder."),
        "1x = this object's own measured total, bytes": w1(
            "x_scored1", "/ledger_four_part/honest_total_bytes", "{:,.0f}"),
        "1x = float32 parameters a 1x-matched rival would carry": w1(
            "x_cross_val", "/per_K/32/baseline_match_1x/target_params_float32", "{:,.0f}"),
        "1x column status": "DECLARED NOT-RUN - no rival was fitted at this size (NR8).",
        "524,288 B reference, bytes": w1("x_scored1", "/ledger_four_part/reference_pin_bytes", "{:,.0f}"),
        "524,288 B reference, float32 parameters": w1(
            "x_cross_val", "/per_K/32/baseline_match_1x/reference_pin_params_float32", "{:,.0f}"),
        "rivals actually fitted, params": "130,433 - 131,891, i.e. within +/-0.63 % of 131,072",
        "524,288 B column status": "MEASURED - every comparator row above is against this column.",
        "DIRECTION OF THE ERROR": (
            "AGAINST the arm. A 1x-matched rival would carry about 4.09x these parameters and would be "
            "STRONGER, so the printed comparison FLATTERS the arm and no row above is a 1x result."),
    },
    "verdict": {
        "statement": ("the control gate PASSES - all four controls read exactly 0.000 on both rows at "
                      "rank/width 0.9688 - so the row may be claimed, at 1 seed, against the 524,288 B "
                      "reference column and NOT at 1x."),
        "charter C.1 vs the prototype table floor": w1(
            "x_scored1", "/paired_vs_cov_tensor/table/mean_paired_delta", "{:+.4f}"),
        "its CI95 lower bound": w1("x_scored1", "/paired_vs_cov_tensor/table/bootstrap_ci95/0", "{:+.4f}"),
        "margin declared in advance": w1(
            "x_scored1", "/paired_vs_cov_tensor/table/margin_declared_in_advance", "{:+.1f}"),
        "provenance": [dsrc("x_scored1", "/paired_vs_cov_tensor/table/mean_paired_delta"),
                       dsrc("x_scored1", "/paired_vs_cov_tensor/landscape_removed/mean_paired_delta")],
    },
    "caveats": [
        ("`cont_mlp` scores 0.000 / 0.061 - at the floor. That is a fact about that rival's fit on this "
         "population, not a demonstration about the arm, and it is printed rather than leaned on."),
        ("The deletion legs of this same object are T9 and the refusal DISPLAY is T10. Same object, same "
         "seed, still separate tables, because a DISPLAY is not a RESULT."),
        ("The request-only property is asserted in the artefact: no feature tensor enters launch, settle "
         "or read (charter B.3), bitwise stable under feature perturbation over 64 checks."),
    ],
}

# --------------------------------------------------------------------------------------
# 8x. TABLE 8 -- THE n-WAY LADDER.  n = 2..6.  K = 5 ON EVERY ROW.  3 seeds.
# --------------------------------------------------------------------------------------
LAD_RUNGS = ["2", "3", "4", "5", "6"]
LAD_ARMS = [
    ("single_design_K5", "single design (the arm), K = 5"),
    ("pca_tensor", "pca_tensor - composition-only store (ABLATION)"),
    ("learned_1x", "learned_1x - MLP matched to the arm's OWN total"),
    ("rival_mlp", "rival_mlp @524,288 B"),
    ("rival_lstm", "rival_lstm @524,288 B"),
    ("rival_transformer", "rival_transformer @524,288 B"),
    ("rival_cont_mlp", "rival_cont_mlp @524,288 B"),
    ("rival_table", "rival_table - the FLOOR"),
    ("learned_ref_ownpool", "learned_ref_ownpool"),
]

_t8_rows = []
for _rg in LAD_RUNGS + ["shipped"]:
    _cells = {
        "K (wells)": derived([5], [dsrc("lad_json", "/K_of_the_store_arms")],
                             "every leg of this ladder sits at K = 5", fmt="{:.0f}"),
        "seeds (n)": w1("lad_json", "/n_seeds_declared", "{:.0f}"),
        "corners": w3("lad_json", "/rungs/" + _rg + "/n_corners_per_seed/{s}", "{:.0f}"),
        "ceiling / corner": w1("lad_json", "/rungs/" + _rg + "/ceiling_per_corner", "{:.0f}"),
    }
    for _arm, _lab in LAD_ARMS:
        _cells[_lab] = w3("lad_json", "/rungs/" + _rg + "/arms/" + _arm + "/per_seed/{s}/mean", "{:.4f}")
    _cells["delta vs 1x (learned_1x)"] = w3(
        "lad_json",
        "/rungs/" + _rg + "/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta",
        "{:+.4f}")
    _cells["delta vs best @524,288 B"] = w3(
        "lad_json",
        "/rungs/" + _rg + "/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta",
        "{:+.4f}")
    _cells["rank/width max (arm)"] = w3(
        "lad_json",
        "/rungs/" + _rg + "/arms/single_design_K5/per_seed/{s}/rank_width/rank_over_width_max", "{:.4f}")
    if _rg != "shipped":
        _cells["control landscape_shuffled, paired delta"] = wcell(
            [("lad_json", "/control_verdicts/" + _rg
              + "/single_design_K5/landscape_shuffled/per_seed/%d/mean_paired_delta" % i) for i in range(3)],
            fmt="{:+.4f}", seeds=[0, 1, 2])
        _cells["control random_atoms, paired delta"] = wcell(
            [("lad_json", "/control_verdicts/" + _rg
              + "/single_design_K5/random_atoms/per_seed/%d/mean_paired_delta" % i) for i in range(3)],
            fmt="{:+.4f}", seeds=[0, 1, 2])
        _lbl, _cl = "**n = " + _rg + "**", "RESULT (APPENDIX placement)"
    else:
        _lbl = "`shipped` - the value-holdout condition as it exists on disk (reproduction leg)"
        _cl = "instrument check, NOT a rung"
    _t8_rows.append({"label": _lbl, "row_class": _cl, "seeds_printed": [0, 1, 2],
                     "n_seeds_printed": 3, "cells": _cells})

TABLES["T8"] = {
    "title": ("TABLE 8 - the n-WAY LADDER on Shapes3D: n = 2..6 jointly composed factors, "
              "K = 5 WELLS on every row, 3 seeds on every row."),
    "table_class": "RESULT",
    "placement": w1("lad_json", "/placement", "{}"),
    "notation": {
        "n": "the ladder rung = the number of jointly composed factors. n = 2, 3, 4, 5, 6.",
        "K": ("5 on EVERY row of this table. The source writes the rung as `k`; this table writes the "
              "rung as n and prints K separately, because K is the WELL COUNT and the two were "
              "collided once."),
        "RULE": "T8 (K = 5) and T7 (K = 32) are two different objects and never share a table or an axis.",
    },
    "pins": {"condition": "Shapes3D value-holdout, k-way families from the w12 harness",
             "K (wells)": 5, "metric": "distinct_valid_novel_tuples per corner",
             "estimand": "paired, the corner as the unit", "margin": 1.0,
             "arms per leg": 13, "legs": "18 = (5 rungs + `shipped`) x 3 seeds",
             "scored arm-runs": 234},
    "pin_provenance": [dsrc("lad_json", "/K_of_the_store_arms"), dsrc("lad_json", "/metric"),
                       dsrc("lad_json", "/margin"), dsrc("lad_json", "/n_seeds_declared"),
                       dsrc("lad_json", "/seeds"), dsrc("lad_json", "/not_run"),
                       dsrc("lad_json", "/best_learned_rule"), dsrc("lad_json", "/K_star_source")],
    "seed_count": {
        "seeds on every row": w1("lad_json", "/n_seeds_declared", "{:.0f}"),
        "PRINTED AS A COLUMN": "`seeds (n)` is a column of this table, not a footnote.",
        "why not 5": ("seed 3 passes the generation-2 cache gate and still cannot be scored - `Judge(3)` "
                      "raises FileNotFoundError on `decoder_v1_s3.pkl`; seed 4's cache landed at "
                      "2026-09-21T15:27:15Z, after this hub had closed its seed gate. Both are NR7."),
    },
    "rows": _t8_rows,
    "byte_ledger": {
        "rule": ("FOUR PARTS, NEVER NETTED. The ladder's ledger file carries parts 1-3 measured on all 18 "
                 "fitted legs; part 4 (coupling / hole table) is NOT in that file and is a DECLARED "
                 "ABSENCE below, never a zero."),
        "parts": {
            "1. atoms (both store arms, every leg)": w1(
                "lad_bytes", "/legs/2_seed0/single_design_K5/p1_atom_bytes", "{:,.0f}"),
            "2. sufficient statistics - single design, n = 2, seeds 0/1/2": wcell(
                [("lad_bytes", "/legs/2_seed%d/single_design_K5/p2_statistics_bytes" % s) for s in range(3)],
                fmt="{:,.0f}", seeds=[0, 1, 2]),
            "2. sufficient statistics - pca_tensor (the ablation)": wcell(
                [("lad_bytes", "/legs/2_seed%d/pca_tensor/p2_statistics_bytes" % s) for s in range(3)],
                fmt="{:,.0f}", seeds=[0, 1, 2]),
            "3. address book - single design, n = 2, seeds 0/1/2": wcell(
                [("lad_bytes", "/per_arm/single_design_K5/%d/p3_address_book" % i) for i in range(3)],
                fmt="{:,.0f}", seeds=[0, 1, 2]),
            "4. coupling / hole table": {
                "status": "DECLARED ABSENT FROM THE SOURCE",
                "why": ("`BYTES.json` carries three parts per leg. The coupling/hole table is measured on "
                        "the K* object in T7 (6,610 B) and across the K-sweep in T12; it is NOT measured "
                        "on the ladder's legs and is therefore not printed as a zero here.")},
        },
        "totals": {
            "single design total - min over 18 legs": w1(
                "lad_bytes", "/summary/single_design_K5/total_bytes_min", "{:,.0f}"),
            "single design total - max over 18 legs": w1(
                "lad_bytes", "/summary/single_design_K5/total_bytes_max", "{:,.0f}"),
            "x the 524,288 B reference - min": w1(
                "lad_bytes", "/summary/single_design_K5/times_reference_pin_min", "{:.4f}"),
            "x the 524,288 B reference - max": w1(
                "lad_bytes", "/summary/single_design_K5/times_reference_pin_max", "{:.4f}"),
            "pca_tensor total - min / max": wcell(
                [("lad_bytes", "/summary/pca_tensor/total_bytes_min"),
                 ("lad_bytes", "/summary/pca_tensor/total_bytes_max")],
                fmt="{:,.0f}", seeds=["min", "max"], stats=False),
            "pca_tensor x reference - min / max": wcell(
                [("lad_bytes", "/summary/pca_tensor/times_reference_pin_min"),
                 ("lad_bytes", "/summary/pca_tensor/times_reference_pin_max")],
                fmt="{:.4f}", seeds=["min", "max"], stats=False),
        },
        "netted": False,
        "THE HEADLINE THE DRAFT OMITTED": (
            "the single design costs 23.886x-24.875x the 524,288 B reference and `pca_tensor` costs "
            "1.358x-1.372x - about 18x less - for composition scores that are IDENTICAL at every rung "
            "and every seed. The single design's bytes buy EXACT DELETION, not composition."),
    },
    "byte_matching_flip": {
        "THE CAPTION THAT MUST TRAVEL WITH THIS TABLE": (
            "THE TWO BYTE MATCHINGS TELL OPPOSITE STORIES ABOUT THE n-TREND. Against the 524,288 B rivals "
            "the arm's margin GROWS with n: +7.083..+9.650 at n = 2 rising to +14.992..+22.425 at n = 6. "
            "Against a learner matched to the arm's OWN total (1x) the margin is NEGATIVE at n = 2, 4, 5 "
            "and 6. A reader shown one column alone draws the opposite conclusion, so both columns are "
            "printed on every row of this table."),
        "delta vs best @524,288 B - n = 2, per seed": w3(
            "lad_json",
            "/rungs/2/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta", "{:+.4f}"),
        "delta vs best @524,288 B - n = 6, per seed": w3(
            "lad_json",
            "/rungs/6/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta", "{:+.4f}"),
        "delta vs 1x - n = 2, per seed": w3(
            "lad_json",
            "/rungs/2/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta", "{:+.4f}"),
        "delta vs 1x - n = 6, per seed": w3(
            "lad_json",
            "/rungs/6/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta", "{:+.4f}"),
        "delta vs 1x, ceiling-normalised, n = 2/3/4/5/6": derived(
            [-0.077315, 0.058912, -0.073380, -0.054167, -0.056404],
            [dsrc("lad_json", "/rungs/%s/arms/single_design_K5/per_seed/%d/"
                              "paired_vs_best_learned/mean_paired_delta" % (rg, s))
             for rg in LAD_RUNGS for s in range(3)]
            + [dsrc("lad_json", "/rungs/%s/ceiling_per_corner" % rg) for rg in LAD_RUNGS],
            "mean over the 3 seeds of the paired delta, divided by that rung's ceiling per corner",
            fmt="{:+.6f}",
            note=("DERIVED. Negative at n = 2, 4, 5, 6 (-0.0773 .. -0.0542) and positive only at n = 3. "
                  "Recompute from the 20 pointers in `from`.")),
        "the_1x_opponent": (
            "`learned_1x` is ONE family (an MLP) fitted at the arm's own total. A 1x-matched LSTM, "
            "transformer, continuous MLP and table are a DECLARED NOT-RUN (NR8), so the 1x column is the "
            "WEAKEST available 1x opponent and the true 1x margin can only be more negative."),
    },
    "instrument": {
        "FOR n >= 3 THIS LADDER IS A TRAINING-DATA AXIS, NOT A COMPOSITION-DEPTH AXIS": (
            "n = 3, 4, 5 and 6 probe the IDENTICAL 120 corners; only how coarsely the hold-out is carved "
            "changes, and with it the held-out row count (300,000 / 235,200 / 213,600 / 200,640). Every "
            "arm fitted on the rung's own pool tracks that count at r = -0.96 .. -0.98; every arm not "
            "fitted on it is EXACTLY constant across those four rungs. The n = 2 -> n >= 3 step is the "
            "only genuine change of probe on this ladder."),
        "corners at n = 2 (disjoint from n >= 3)": w3("lad_json", "/rungs/2/n_corners_per_seed/{s}", "{:.0f}"),
        "corners at n = 6": w3("lad_json", "/rungs/6/n_corners_per_seed/{s}", "{:.0f}"),
        "no arm has a breaking point": w1("lad_json", "/same_k_for_every_arm/statement", "{}"),
        "legs that failed": derived(["0 - the source's own `not_run` field is the empty list"],
                                    [dsrc("lad_json", "/not_run")], "stored field", fmt="{}"),
        "ceiling-normalised arm, n = 2 -> n = 6": wcell(
            [("lad_json", "/ladder_verdicts/single_design_K5/per_rung/%s/normalised_mean" % rg)
             for rg in LAD_RUNGS], fmt="{:.4f}", seeds=["n=2", "n=3", "n=4", "n=5", "n=6"], stats=False),
    },
    "verdict": {
        "statement": ("the store NEVER CLEARS THE BAR against a learner matched to its own bytes - at "
                      "n = 2, 4, 5 and 6 on every seed, and at 2 of 3 seeds at n = 3, so under the "
                      "pre-declared every-seed rule it clears at NO rung. Nothing deteriorates as n "
                      "grows; ceiling-normalised the arm goes UP, 0.8483 -> 0.8966. Both statements are "
                      "the ladder's result and neither of them is a curve."),
        "provenance": [dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/2/clears_bar"),
                       dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/6/clears_bar"),
                       dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/2/normalised_mean"),
                       dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/6/normalised_mean"),
                       dsrc("lad_json", "/same_k_for_every_arm/statement")],
    },
}

'''

ANCHOR = """# --------------------------------------------------------------------------------------
# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line."""
n = t.count(ANCHOR)
assert n == 1, "NOT-RUN anchor occurs %d times, expected 1" % n
t = t.replace(ANCHOR, NEW + ANCHOR)

SRC.write_text(t, encoding="utf-8")
print("p2 OK ->", SRC)
