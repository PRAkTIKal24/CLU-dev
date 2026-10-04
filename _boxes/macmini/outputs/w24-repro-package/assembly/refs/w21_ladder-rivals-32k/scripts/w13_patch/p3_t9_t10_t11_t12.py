#!/usr/bin/env python
"""w13 patch 3/4 -- TABLE 9 / 9b (deletion: value, conjunction, conjunction-live, item),
TABLE 10 (refusal DISPLAY), TABLE 11 (the two ablations) and TABLE 12 (the K-curve).
Exact-string patch; the anchor must occur EXACTLY once."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")

NEW = r'''
# --------------------------------------------------------------------------------------
# 8y. TABLE 9 / 9b -- DELETION.  value / conjunction / conjunction-live / item.
#     T9 = TEST, seed 0, K* = 32 (the source's own SMOKE_NOT_A_CLAIM is false).
#     T9b = VAL, seed 0, K = 5   (the source's own SMOKE_NOT_A_CLAIM is TRUE -> labelled SMOKE).
#     Two conditions and two K: two tables, never one.
# --------------------------------------------------------------------------------------
DEL_SCOPES = [
    ("value", "(a) VALUE - every fit row carrying red (`object_hue` index 0)"),
    ("conjunction", "(b) CONJUNCTION - red AND cube, AS THE TASK SPECIFIES IT -- **VACUOUS**"),
    ("conjunction_live", "(b') RED SPHERE - the DECLARED non-vacuous substitute"),
    ("item", "(c) ITEM - a random 1 % of fit rows"),
]


def _deletion_table(sid, tid, cond, kk, seed, smoke):
    rows = []
    for sc, label in DEL_SCOPES:
        b = "/scopes/" + sc
        cells = {
            "K (wells)": derived([kk], [dsrc(sid, "/K")], "stored field /K", fmt="{:.0f}"),
            "seeds (n)": derived([1], [dsrc(sid, "/seed")], "one seed, id " + str(seed), fmt="{:.0f}"),
            "rows deleted": w1(sid, b + "/n_rows_deleted", "{:,.0f}", seed=seed),
            "fraction of the fit pool": w1(sid, b + "/frac_rows_deleted", "{:.6f}", seed=seed),
            "OUR adaptation, seconds": w1(sid, b + "/our_cost_line/seconds", "{:.4f}", seed=seed),
            "OUR bytes rewritten": w1(sid, b + "/our_cost_line/bytes_rewritten", "{:,.0f}", seed=seed),
            "OUR data rows read": w1(sid, b + "/our_cost_line/data_rows_read", "{:,.0f}", seed=seed),
            "OUR refit?": w1(sid, b + "/our_cost_line/refit", "{}", seed=seed),
            "REFERENCE refit-without, seconds": w1(
                sid, b + "/exactness_store_scoped/refit_without_cost_line/seconds", "{:.3f}", seed=seed),
            "REFERENCE refit-without, data rows read": w1(
                sid, b + "/exactness_store_scoped/refit_without_cost_line/data_rows_read", "{:,.0f}", seed=seed),
            "store-scoped exactness, max abs mode dev": w1(
                sid, b + "/exactness_store_scoped/max_abs_mode_dev", "{:.3e}", seed=seed),
            "ranks agree": w1(sid, b + "/exactness_store_scoped/ranks_agree", "{}", seed=seed),
            "survivor corners": w1(sid, b + "/n_survivor_corners", "{:.0f}", seed=seed),
            "paired delta on survivors": w1(
                sid, b + "/paired_after_minus_before_on_survivors/mean_paired_delta", "{:+.4f}", seed=seed),
            "DELETED corners": w1(sid, b + "/n_deleted_corners", "{:.0f}", seed=seed),
            "principal-angle cos_min (the basis DID move)": w1(
                sid, b + "/basis_change_audit/principal_angle_cos_min", "{:.6f}", seed=seed),
        }
        if sc == "value" and cond == "test":
            cells["DELETED corners, novel/corner BEFORE"] = w1(
                sid, b + "/before_same_corners/deleted/novel_per_corner_mean", "{:.4f}", seed=seed)
            cells["DELETED corners, novel/corner AFTER"] = w1(
                sid, b + "/after/deleted/novel_per_corner_mean", "{:.4f}", seed=seed)
            cells["DELETED corners, judge acc BEFORE"] = w1(
                sid, b + "/before_same_corners/deleted/judge_accuracy_mean", "{:.4f}", seed=seed)
            cells["DELETED corners, judge acc AFTER"] = w1(
                sid, b + "/after/deleted/judge_accuracy_mean", "{:.4f}", seed=seed)
        else:
            cells["DELETED corners, novel/corner BEFORE"] = derived(
                ["DECLARED: this scope deletes no corner"], [dsrc(sid, b + "/n_deleted_corners")],
                "the source writes DECLARED_NOT_RUN in place of a number", fmt="{}")
            cells["DELETED corners, novel/corner AFTER"] = derived(
                ["DECLARED: this scope deletes no corner"], [dsrc(sid, b + "/n_deleted_corners")],
                "the source writes DECLARED_NOT_RUN in place of a number", fmt="{}")
        rows.append({"label": label, "row_class": ("RESULT" if not smoke else "SMOKE - NOT A CLAIM"),
                     "seeds_printed": [seed], "n_seeds_printed": 1, "cells": cells})

    return {
        "title": ("TABLE %s - EXACT DELETION (STORE-SCOPED) on the single design, %s condition, "
                  "K = %d, 1 seed (id %d).%s" % (tid[1:], cond.upper(), kk, seed,
                                                 "  SMOKE - NOT A CLAIM." if smoke else "")),
        "table_class": "RESULT" if not smoke else "SMOKE (the artefact labels itself so)",
        "placement": "MAIN TEXT" if not smoke else "APPENDIX",
        "SMOKE_NOT_A_CLAIM_as_the_artefact_records_it": w1(sid, "/SMOKE_NOT_A_CLAIM", "{}", seed=seed),
        "notation": {"K_wells": kk,
                     "n_factors": "not varied in this table; the n-ladder is T8",
                     "rule": "every row of T%s states K = %d and its seed count." % (tid[1:], kk)},
        "pins": {"condition": cond, "K (wells)": kk, "seed": seed,
                 "definition of exact": "STORE-SCOPED - see TABLES.json /exactness_definition",
                 "feature cache": "generation 2, scope full, 480,000 rows"},
        "pin_provenance": [dsrc(sid, "/K"), dsrc(sid, "/seed"), dsrc(sid, "/condition"),
                           dsrc(sid, "/A_requested"), dsrc(sid, "/feature_cache/generation"),
                           dsrc(sid, "/SMOKE_NOT_A_CLAIM")],
        "seed_count": {"seeds (n)": 1,
                       "PRINTED AS A COLUMN": "`seeds (n)` is a column of this table.",
                       "why not 3 or 5": "NR6 (wall clock, seeds 1-2) and NR7 (seeds 3-4 unjudgeable)."},
        "rows": rows,
        "byte_ledger": {
            "rule": "FOUR PARTS, NEVER NETTED. Measured on the object this deletion ran on.",
            "parts": {
                "1. atoms": w1(sid, "/ledger_four_part/p1_atom_bytes", "{:,.0f}", seed=seed),
                "2. covariance + count ledger": w1(
                    sid, "/ledger_four_part/p2_covariance_plus_count_bytes", "{:,.0f}", seed=seed),
                "3. address book": w1(sid, "/ledger_four_part/p3_address_book_bytes", "{:,.0f}", seed=seed),
                "4. coupling / hole table": w1(
                    sid, "/ledger_four_part/p4_coupling_hole_table_bytes", "{:,.0f}", seed=seed),
            },
            "totals": {
                "honest total": w1(sid, "/ledger_four_part/honest_total_bytes", "{:,.0f}", seed=seed),
                "x the 524,288 B reference": w1(sid, "/ledger_four_part/times_reference_pin", "{:.4f}", seed=seed),
            },
            "netted": False,
        },
        "language_rule_the_head_set": (
            "NEVER 'no changes after deletion'. The sentence is: post-deletion adaptation is LIGHT and "
            "DATA-FREE - the covariance and count are downdated by subtraction, the atoms are "
            "re-eigendecomposed from the downdated statistic and the address book is carried across the "
            "basis change, with NO row of data read - against a reference that can only delete by "
            "refitting from scratch without the deleted rows, reading 120,028-131,072 rows to do it."),
        "vacuous_row_is_printed_not_hidden": w1(
            sid, "/scopes/conjunction/vacuous_reason", "{}", seed=seed),
    }


TABLES["T9"] = _deletion_table("x_del_test", "T9", "test", 32, 0, False)
TABLES["T9b"] = _deletion_table("x_del_val", "T9b", "val", 5, 0, True)

# ---- the finding that bounds the whole deletion claim, on T9's own object ---------------
TABLES["T9"]["the_probe_that_bounds_the_claim"] = {
    "what": ("how far the RENDERED answer moved on exactly the requests whose evidence was deleted, "
             "in the store's own units. No judge, no decoder. 5,688 requests."),
    "rows": [
        {"label": "corners REQUESTING red", "row_class": "probe", "seeds_printed": [0], "n_seeds_printed": 1,
         "cells": {
             "requests": w1("x_probe", "/per_group/corners REQUESTING red/n_requests", "{:,.0f}"),
             "rel. RMS change, address book CARRIED": w1(
                 "x_probe", "/per_group/corners REQUESTING red/relative_rms_change_carried", "{:.3e}"),
             "rel. RMS change, address book NOT carried": w1(
                 "x_probe", "/per_group/corners REQUESTING red/relative_rms_change_uncarried", "{:.6f}"),
             "max abs change, carried": w1(
                 "x_probe", "/per_group/corners REQUESTING red/max_abs_change_carried", "{:.3e}"),
         }},
        {"label": "corners NOT requesting red", "row_class": "probe", "seeds_printed": [0], "n_seeds_printed": 1,
         "cells": {
             "requests": w1("x_probe", "/per_group/corners NOT requesting red/n_requests", "{:,.0f}"),
             "rel. RMS change, address book CARRIED": w1(
                 "x_probe", "/per_group/corners NOT requesting red/relative_rms_change_carried", "{:.3e}"),
             "rel. RMS change, address book NOT carried": w1(
                 "x_probe", "/per_group/corners NOT requesting red/relative_rms_change_uncarried", "{:.6f}"),
             "max abs change, carried": w1(
                 "x_probe", "/per_group/corners NOT requesting red/max_abs_change_carried", "{:.3e}"),
         }},
    ],
    "address book bytes changed by the carry": w1("x_probe", "/carry/address_book_bytes_changed", "{:,.0f}"),
    "address book bytes total": w1("x_probe", "/carry/address_book_bytes_total", "{:,.0f}"),
    "data rows touched by the carry": w1("x_probe", "/carry/data_rows_touched", "{:,.0f}"),
    "THE BOUND, STATED WITHOUT SPIN": (
        "deleting all 11,044 red rows changes the 20 red-requesting corners by EXACTLY ZERO "
        "(32.25 -> 32.25 novel valid tuples per corner, judge accuracy 0.8958 -> 0.8958). The store-scoped "
        "exactness is real - the uncarried read moves by 0.61 relative RMS - but the basis change is "
        "GENERIC, not targeted: 0.6101 on red-requesting corners against 0.6143 on the others. At "
        "K* = 32 (rank/width 0.9531-0.9688) the per-well content carries no value-specific information; "
        "the value identity lives in the address book, which the deletion CARRIES and does not refit. "
        "'The store has forgotten red' and 'the object no longer answers red' are different statements "
        "and only the first is demonstrated."),
    "the_contrast_with_wave_8": (
        "at K = 5, A = 64 (rank/width 0.156) wave 8 found the basis change CONCENTRATED: cos_min 0.1999 "
        "against 0.9976 for a size-matched random deletion. The targeting of a deletion is a function of "
        "rank/width and it disappears exactly where the byte ledger is cheapest."),
}

TABLES["T9"]["sphere_caveat"] = {
    "HEAD_RULING": w1("x_del_test", "/sphere_caveat/HEAD_RULING", "{}"),
    "source": w1("x_del_test", "/sphere_caveat/source", "{}"),
    "what_it_threatens": w1("x_del_test", "/sphere_caveat/what_it_threatens", "{}"),
    "judged-sphere excess, `landscape_removed` (degraded bound)": w1(
        "x_del_test", "/sphere_caveat/measured_upper_bound_degraded_arms/landscape_removed/judged_sphere_excess",
        "{:+.4f}"),
    "judged-sphere excess, `random_atoms` (degraded bound)": w1(
        "x_del_test", "/sphere_caveat/measured_upper_bound_degraded_arms/random_atoms/judged_sphere_excess",
        "{:+.4f}"),
    "judged-sphere excess AFTER the red-sphere deletion": w1(
        "x_del_test", "/scopes/conjunction_live/shape_report_after/judged_sphere_excess", "{:+.4f}"),
    "verdict": ("the hazard does NOT reproduce on this instrument: a degraded render here is judged CUBE, "
                "the shape held out of the split entirely, so the sphere excess is NEGATIVE for both "
                "maximally degraded arms. A degrading baseline cannot score spuriously as sphere here. "
                "Scope: seed 0, TEST corners, K* = 32, this decoder and this classifier."),
}

# --------------------------------------------------------------------------------------
# 8z. TABLE 10 -- REFUSAL DISPLAY ROWS.  A DISPLAY IS NEVER A RESULT.
# --------------------------------------------------------------------------------------
_t10_rows = [
    {"label": "**OURS - the coupling/hole table**, TEST, K* = 32, seed 0",
     "row_class": "DISPLAY - never a result", "seeds_printed": [0], "n_seeds_printed": 1,
     "cells": {
         "K (wells)": derived([32], [dsrc("x_del_test", "/K")], "stored field /K", fmt="{:.0f}"),
         "seeds (n)": derived([1], [dsrc("x_del_test", "/seed")], "one seed, id 0", fmt="{:.0f}"),
         "holes punched": w1("x_del_test", "/display_refusal/n_holes_after", "{:.0f}"),
         "bytes rewritten": w1("x_del_test", "/display_refusal/cost_line/bytes_rewritten", "{:,.0f}"),
         "data rows read": w1("x_del_test", "/display_refusal/cost_line/data_rows_read", "{:,.0f}"),
         "seconds": w1("x_del_test", "/display_refusal/cost_line/seconds", "{:.4f}"),
         "refit?": w1("x_del_test", "/display_refusal/cost_line/refit", "{}"),
         "coupling table bytes": w1("x_del_test", "/display_refusal/coupling_table_bytes", "{:,.0f}"),
         "layout asserted equal to `gen_field`'s": w1(
             "x_del_test", "/display_refusal/layout_is_the_field_store's/layout_equals_gen_field", "{}"),
         "pair slots / indices checked": wcell(
             [("x_del_test", "/display_refusal/layout_is_the_field_store's/n_pairs"),
              ("x_del_test", "/display_refusal/layout_is_the_field_store's/n_indices_checked")],
             fmt="{:,.0f}", seeds=["pairs", "checked"], stats=False),
     }},
    {"label": "OURS - the coupling/hole table, VAL, K = 5, seed 0 (SMOKE)",
     "row_class": "DISPLAY - SMOKE, never a result", "seeds_printed": [0], "n_seeds_printed": 1,
     "cells": {
         "K (wells)": derived([5], [dsrc("x_del_val", "/K")], "stored field /K", fmt="{:.0f}"),
         "seeds (n)": derived([1], [dsrc("x_del_val", "/seed")], "one seed, id 0", fmt="{:.0f}"),
         "holes punched": w1("x_del_val", "/display_refusal/n_holes_after", "{:.0f}"),
         "bytes rewritten": w1("x_del_val", "/display_refusal/cost_line/bytes_rewritten", "{:,.0f}"),
         "data rows read": w1("x_del_val", "/display_refusal/cost_line/data_rows_read", "{:,.0f}"),
         "seconds": w1("x_del_val", "/display_refusal/cost_line/seconds", "{:.4f}"),
         "refit?": w1("x_del_val", "/display_refusal/cost_line/refit", "{}"),
         "coupling table bytes": w1("x_del_val", "/display_refusal/coupling_table_bytes", "{:,.0f}"),
         "layout asserted equal to `gen_field`'s": w1(
             "x_del_val", "/display_refusal/layout_is_the_field_store's/layout_equals_gen_field", "{}"),
     }},
]

_RT = "/per_seed/{s}/records/transformer__refuse_token__%s"
for _lbl, _rec in [("VALUE deletion (red)", "value-object_hue-0"),
                   ("CONJUNCTION deletion (red AND sphere)", "conj-object_hue0-shape2")]:
    _b = _RT % _rec
    _t10_rows.append({
        "label": "COMPARATOR - refuse-token transformer, " + _lbl,
        "row_class": "DISPLAY comparator - 3 seeds, FEATURE-LEVEL, off the hub's instrument",
        "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
        "cells": {
            "K (wells)": derived(["n/a - a learned rival has no wells"],
                                 [dsrc("base_ready", "/pins/byte_pin")], "the rival is byte-matched, not partitioned",
                                 fmt="{}"),
            "seeds (n)": derived([3], [dsrc("base_ready", "/seeds_present")], "stored field", fmt="{:.0f}"),
            "requests certified impossible": w3("base_ready", _b + "/applicability/n_certified_impossible", "{:,.0f}"),
            "AUROC, impossible vs possible": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/auroc", "{:.4f}"),
            "separation (mean score gap)": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/separation", "{:.4f}"),
            "refuse rate on impossible": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/refuse_rate_on_impossible", "{:.4f}"),
            "FALSE refusal on possible": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/false_refusal_on_possible", "{:.4f}"),
            "bytes rewritten": w3("base_ready", _b + "/cost/bytes_rewritten", "{:,.0f}"),
            "data rows read": w3("base_ready", _b + "/cost/data_rows_read", "{:,.0f}"),
            "seconds": w3("base_ready", _b + "/cost/seconds", "{:.4f}"),
            "params": w3("base_ready", _b + "/param_count", "{:,.0f}"),
        }})

_t10_rows.append({
    "label": "ITEM deletion (random 1 %) - THERE IS NO REFUSAL ROW AT ALL",
    "row_class": "DECLARED NOT-APPLICABLE, measured - never a zero",
    "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
    "cells": {
        "requests built from deleted rows, tested": w3(
            "base_ready",
            "/per_seed/{s}/DECLARED_NOT_RUN/measured_not_applicable/refuse_token@item-0.01/"
            "applicability/n_deleted_requests_tested", "{:,.0f}"),
        "of those, CERTIFIED impossible": w3(
            "base_ready",
            "/per_seed/{s}/DECLARED_NOT_RUN/measured_not_applicable/refuse_token@item-0.01/"
            "applicability/n_certified_impossible", "{:,.0f}"),
        "a refusal row applies?": w3(
            "base_ready",
            "/per_seed/{s}/DECLARED_NOT_RUN/measured_not_applicable/refuse_token@item-0.01/"
            "applicability/applies", "{}"),
    }})

TABLES["T10"] = {
    "title": "TABLE 10 - REFUSAL DISPLAY ROWS.  A DISPLAY IS NEVER A RESULT.",
    "table_class": "DISPLAY",
    "placement": "MAIN TEXT, labelled DISPLAY",
    "DISPLAY_NEVER_A_RESULT": w1("x_del_test", "/display_refusal/DISPLAY_NEVER_A_RESULT", "{}"),
    "notation": {"K_wells": "32 on our TEST row, 5 on our VAL row; n/a for a byte-matched rival",
                 "rule": "our two rows are at different K and are labelled as such."},
    "pins": {"our object": "single design, Shapes3D value-holdout",
             "comparator": "refuse-token transformer, the ONLY permitted comparator for a refusal DISPLAY row",
             "comparator byte pin": 524288},
    "pin_provenance": [dsrc("base_ready", "/pins/byte_pin"), dsrc("base_ready", "/pins/refuse_threshold"),
                       dsrc("base_ready", "/seeds_present"), dsrc("base_ready", "/NOT_A_SCORED_RUN"),
                       dsrc("x_del_test", "/display_refusal/comparator_status")],
    "seed_count": {"our rows": 1, "the comparator's rows": 3,
                   "NEVER MIX": ("our 1-seed rows and the comparator's 3-seed rows are printed in one "
                                 "table with the seed count AS A COLUMN, and are NOT plotted together "
                                 "without it.")},
    "rows": _t10_rows,
    "WHAT THE TWO SIDES OF THIS TABLE ARE, AND ARE NOT": (
        "OUR side and the COMPARATOR side are NOT the same quantity and are not differenced. Ours is a "
        "cost line: 2 bytes, 0 data rows, no refit, on the object T9 deleted from. The comparator's is a "
        "SEPARATION diagnostic measured at the FEATURE level by spoke/deletion-baselines on its own "
        "16,384-row fit pool, with no judge and no decoder. The head-to-head DISPLAY leg - our refusal "
        "rate against the refuse-token transformer's render rate on the same post-deletion requests - is "
        "NR9, a DECLARED NOT-RUN: the crossover hub's wall clock ran out after the scored rows and the "
        "deletion legs."),
    "the_comparator_status_as_the_artefact_records_it": w1(
        "x_del_test", "/display_refusal/comparator_status", "{}"),
    "why_this_row_exists": w1("x_del_test", "/display_refusal/what_this_row_is_for", "{}"),
    "carried_caveat": w1("x_del_test", "/display_refusal/carried_caveat", "{}"),
    "THE POINT THE HOLES MAKE": (
        "one hole (index 920, red AND cube) is refusable although NO RED CUBE WAS EVER WRITTEN - cube is "
        "held out of the split entirely. Refusal is a SEPARATE mechanism from forgetting and does not "
        "need a deleted row to have existed. T9's probe makes that separation load-bearing rather than "
        "rhetorical: since the downdate does not stop the object answering red, the hole is the only "
        "mechanism here that can."),
    "verdict": {
        "statement": ("our side of every refusal row is MEASURED (2 bytes, 0 rows read, no refit, layout "
                      "asserted equal to the field store's over 1,322 pair slots). The head-to-head "
                      "against the refuse-token transformer is NR9. The comparator itself is measured at "
                      "3 seeds and separates at AUROC 0.8133-0.8755 with a 7.8-12.7 % FALSE refusal rate "
                      "on possible requests, which is why a DISPLAY comparison against it would have to "
                      "print that false-refusal column too."),
        "provenance": [dsrc("x_del_test", "/display_refusal/cost_line/bytes_rewritten"),
                       dsrc("x_del_test", "/display_refusal/comparator_status"),
                       dsrc("base_ready", "/per_seed/0/records/transformer__refuse_token__value-object_hue-0/"
                                          "out_of_range_diagnostic/separation/auroc")],
    },
}

# --------------------------------------------------------------------------------------
# 8aa. TABLE 11 -- THE TWO ABLATIONS.  Composition-only store, and the item store.
#      Both at the 524,288 B pin, TEST, 3 seeds -- ONE pin, so they may share a table.
# --------------------------------------------------------------------------------------
_ABL_R3 = "/arms/%s/r3_comp_2way/novel_tuples_per_corner"
_ABL_R4 = "/arms/%s/r4_comp_nway/novel_tuples_per_corner"


def _abl_cells(arm):
    return {
        "pin (store bytes)": w3s(arm, "/arms/%s/store_bytes", "{:,.0f}"),
        "address-book bytes": w3s(arm, "/arms/%s/address_bytes", "{:,.0f}"),
        "params @ the pin": w3s(arm, "/arms/%s/param_count", "{}"),
        "row 3 (2-way), 60 corners, ceiling 36": w3s(arm, _ABL_R3.replace("%s", "%s"), "{:.4f}"),
        "row 3, fraction of ceiling": w3s(arm, "/arms/%s/r3_comp_2way/frac_of_ceiling", "{:.4f}"),
        "row 4 (n-way), 49 corners, ceiling 72": w3s(arm, _ABL_R4.replace("%s", "%s"), "{:.4f}"),
        "row 4, fraction of ceiling": w3s(arm, "/arms/%s/r4_comp_nway/frac_of_ceiling", "{:.4f}"),
        "class as the artefact records it": w3s(arm, "/arms/%s/class", "{}"),
    }


_t11_rows = [
    {"label": "**`pca_tensor` - the COMPOSITION-ONLY store** (no sufficient statistics, so no exact deletion)",
     "row_class": "ABLATION - composes, cannot forget", "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("pca_tensor")},
    {"label": "**`oracle_content_items` - the ITEM store**, 64 whole written rows read by an ORACLE",
     "row_class": "ABLATION - forgets, loses composition (and it is a CEILING, not an arm)",
     "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("oracle_content_items")},
    {"label": "`rival_mlp_180k` - the learned reference these two are read against",
     "row_class": "comparator", "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("rival_mlp_180k")},
    {"label": "`oracle_content_fine` - the ABSOLUTE content ceiling at the same pin",
     "row_class": "ceiling, NOT AN ARM", "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("oracle_content_fine")},
]

TABLES["T11"] = {
    "title": ("TABLE 11 - THE TWO ABLATIONS: the composition-only store (composes, cannot forget) and "
              "the ITEM store (forgets, loses composition). w5 compose-field, TEST, 3 seeds, one pin."),
    "table_class": "ablation",
    "placement": "MAIN TEXT",
    "notation": {"K_wells": 5, "n_factors": "row 3 = 2-way, row 4 = n-way; this table does not vary n",
                 "rule": "one pin (524,288 B store bytes), one condition (TEST), 3 seeds on every row."},
    "pins": {"condition": "test", "store bytes": 524288, "K (wells)": 5,
             "metric": "distinct novel valid tuples per corner",
             "corners": "row 3: 60, ceiling 36 / row 4: 49, ceiling 72"},
    "pin_provenance": [dsrc("w5_read_s0", "/arms/pca_tensor/store_bytes"),
                       dsrc("w5_read_s0", "/corner_counts/r3_comp_2way"),
                       dsrc("w5_read_s0", "/corner_counts/r4_comp_nway"),
                       dsrc("w5_read_s0", "/corner_ceilings/r3_comp_2way"),
                       dsrc("w5_read_s0", "/corner_ceilings/r4_comp_nway"),
                       dsrc("w5_read_s0", "/primary")],
    "seed_count": {"seeds on every row": 3,
                   "PRINTED AS A COLUMN": "`seeds (n)` = 3 on every row of T11.",
                   "note": "these are wave-5 scored TEST legs and were never re-run at 5 seeds."},
    "rows": _t11_rows,
    "the_price_of_keeping_items": {
        "item store minus the learned reference, row 4, per seed": derived(
            [-21.0816, -16.5307, -18.0816],
            [dsrc(sid, "/arms/oracle_content_items/r4_comp_nway/novel_tuples_per_corner") for sid in W5]
            + [dsrc(sid, "/arms/rival_mlp_180k/r4_comp_nway/novel_tuples_per_corner") for sid in W5],
            "oracle_content_items r4 minus rival_mlp_180k r4, per seed", fmt="{:+.4f}",
            note=("DERIVED. Their 3-seed mean is -18.5646, which is the programme's '18.6 tuples/corner "
                  "price of keeping items'. Recompute from the six pointers in `from`.")),
        "the composition-only store minus the item store, row 4, per seed": derived(
            [31.2245, 25.2041, 25.8776],
            [dsrc(sid, "/arms/pca_tensor/r4_comp_nway/novel_tuples_per_corner") for sid in W5]
            + [dsrc(sid, "/arms/oracle_content_items/r4_comp_nway/novel_tuples_per_corner") for sid in W5],
            "pca_tensor r4 minus oracle_content_items r4, per seed", fmt="{:+.4f}",
            note="DERIVED. The item store is handed the TRUE FEATURE and the arm is not."),
        "reading": ("the content that WINS composition forfeits exact per-item deletion by construction, "
                    "and that forfeit is priced here at 18.56 tuples/corner on row 4 (3-seed mean). "
                    "The item store is a CEILING handed the true feature, not an arm, so the price is a "
                    "lower bound on what an item store would actually cost."),
    },
    "the_item_store_forgets_exactly_and_loses_composition": {
        "what": ("the other half of the item-store ablation, measured by w5 hub/delete on its own object: "
                 "EVICTION refuses perfectly on the deleted set and FAILS the composition bar."),
        "eviction: refusal on the deleted set": w3p("w5_delete_report",
                                                    "/across_seeds/evict_refuse_on_deleted_set/per_seed/{s}", "{:.4f}"),
        "eviction: composition mean delta (bar is +/-1.0)": w3p(
            "w5_delete_report", "/across_seeds/evict_composition_mean_delta/per_seed/{s}", "{:+.4f}"),
        "the HOLE instead: flag on the deleted set": w3p(
            "w5_delete_report", "/across_seeds/hole_flag_on_deleted_set/per_seed/{s}", "{:.4f}"),
        "the HOLE instead: composition mean delta": w3p(
            "w5_delete_report", "/across_seeds/hole_composition_mean_delta/per_seed/{s}", "{:+.4f}"),
        "row-removal seconds": w3p("w5_delete_report", "/across_seeds/row_removal_s/per_seed/{s}", "{:.4f}"),
        "retrain-from-scratch seconds": w3p(
            "w5_delete_report", "/across_seeds/retrain_from_scratch_s/per_seed/{s}", "{:.4f}"),
        "verdict": ("the operation that is exact on bytes AND free on behaviour is the HOLE, not deletion "
                    "in general: evicting `red` refuses at 1.0000 and costs -2.0899 / -0.9101 / -6.7416 "
                    "composition against a declared +/-1.0 bar, while the hole refuses at 1.0000 and "
                    "costs exactly 0.0000."),
    },
    "cross_reference": ("T8's byte ledger prices the same composition-only ablation on the LADDER's own "
                        "fitted objects: `pca_tensor` at 1.358x-1.372x the 524,288 B reference against "
                        "the single design's 23.886x-24.875x, for composition scores identical at every "
                        "rung and every seed."),
}

# --------------------------------------------------------------------------------------
# 8bb. TABLE 12 -- THE K-CURVE.  K = 5..32 WELLS.  n is not varied here.
# --------------------------------------------------------------------------------------
K_ROWS = [(0, 5), (1, 8), (2, 10), (3, 12), (4, 16), (5, 20), (6, 24), (7, 28), (8, 32)]
_t12_rows = []
for _i, _K in K_ROWS:
    _L = "/rows/%d" % _i
    _P = "/per_K/%d" % _K
    _t12_rows.append({
        "label": "**K = %d**%s" % (_K, "  (= K\\*)" if _K == 32 else ""),
        "row_class": "SMOKE (val sweep) + exact byte arithmetic",
        "seeds_printed": None, "n_seeds_printed": None,
        "cells": {
            "K (wells)": w1("x_ledger_K", _L + "/K", "{:.0f}"),
            "seeds (n) at this K": w1("x_cross_val", _P + "/n_seeds", "{:.0f}"),
            "well width min / max": wcell([("x_ledger_K", _L + "/well_width_min"),
                                           ("x_ledger_K", _L + "/well_width_max")],
                                          fmt="{:.0f}", seeds=["min", "max"], stats=False),
            "rank/width max": w1("x_ledger_K", _L + "/rank_over_width_max", "{:.4f}"),
            "val composition, mean over the seeds run at this K": w1(
                "x_cross_val", _P + "/val_composition_mean", "{:.6f}"),
            "drop vs K = 5 (the frozen rule)": w1("x_cross_val", _P + "/drop_from_K5", "{:+.6f}"),
            "1. atoms B": w1("x_ledger_K", _L + "/p1_atom_bytes", "{:,.0f}"),
            "2. covariance + count B": w1("x_ledger_K", _L + "/p2_covariance_plus_count_bytes", "{:,.0f}"),
            "3. address book B": w1("x_ledger_K", _L + "/p3_address_book_bytes", "{:,.0f}"),
            "3. address book status": w1("x_ledger_K", _L + "/p3_status", "{}"),
            "4. coupling / hole table B": w1("x_ledger_K", _L + "/p4_coupling_hole_table_bytes", "{:,.0f}"),
            "TOTAL, parts 1+2+4 ONLY (no address book)": w1(
                "x_ledger_K", _L + "/total_bytes_parts_1_2_4_only", "{:,.0f}"),
            "x reference, parts 1+2+4": w1("x_ledger_K", _L + "/times_reference_pin_parts_1_2_4", "{:.4f}"),
            "TOTAL, the HONEST FOUR-PART sum": w1("x_ledger_K", _L + "/honest_total_bytes", "{:,.0f}"),
            "x reference, four-part": w1("x_cross_val", _P + "/ledger_four_part/times_reference_pin", "{:.4f}"),
            "1x params a matched rival would carry": w1(
                "x_ledger_K", _L + "/baseline_match_1x_params_float32", "{:,.0f}"),
        }})

TABLES["T12"] = {
    "title": ("TABLE 12 - THE K-CURVE: the four-part byte ledger and val composition against K = 5..32 "
              "WELLS. n is NOT varied in this table."),
    "table_class": "SMOKE (the val sweep labels itself so) + exact arithmetic on the partition shape",
    "placement": "APPENDIX, and the figure is a main-text candidate",
    "SMOKE_NOT_A_CLAIM_as_the_artefacts_record_it": wcell(
        [("x_cross_val", "/SMOKE_NOT_A_CLAIM"), ("x_ledger_K", "/SMOKE_NOT_A_CLAIM")],
        fmt="{}", seeds=["CROSSOVER-val.json", "LEDGER-ACROSS-K.json"], stats=False),
    "notation": {"K": "the number of WELLS - this is the axis of T12",
                 "n": ("NOT VARIED. Every row of T12 reads the same 24 val corners. The n-ladder is T8 "
                       "and it sits at one K."),
                 "rule": "T12 is a K-axis; T8 is an n-axis; they are never the same plot."},
    "pins": {"condition": "val", "A_requested": "64 (63 at K = 32, where A < well width binds)",
             "rank per well": "capped strictly below the well width",
             "selection rule": "K* = max{K : v(5) - mean_seeds v(K) <= 0.02}, frozen 2026-09-21T11:10:24Z",
             "crossover rule": "min{K : drop > 0.10}"},
    "pin_provenance": [dsrc("x_cross_val", "/rule/within"), dsrc("x_cross_val", "/rule/crossover"),
                       dsrc("x_cross_val", "/rule/declared_at"), dsrc("x_cross_val", "/K_star"),
                       dsrc("x_cross_val", "/K_crossover"), dsrc("x_cross_val", "/condition"),
                       dsrc("x_ledger_K", "/reference_pin_bytes")],
    "seed_count": {
        "PRINTED AS A COLUMN": "`seeds (n) at this K` is a column: 3 at K = 5 and K = 32, 1 at every other K.",
        "NEVER MIX": ("a plot of this table MUST print n per point - the two ends carry 3 seeds and the "
                      "seven middle points carry 1. A line drawn through them without n is a lie about "
                      "the evidence."),
    },
    "rows": _t12_rows,
    "K_star": w1("x_cross_val", "/K_star", "{:.0f}"),
    "K_crossover": derived(["NONE INSIDE THE SWEPT RANGE"], [dsrc("x_cross_val", "/K_crossover")],
                           "the stored field is null", fmt="{}"),
    "K_crossover_note": w1("x_cross_val", "/K_crossover_note", "{}"),
    "control gate at K*": {
        "landscape_removed, per seed": w3("x_cross_val", "/control_gate_at_K_star/landscape_removed/per_seed/{s}",
                                          "{:.4f}"),
        "landscape_shuffled, per seed": w3("x_cross_val", "/control_gate_at_K_star/landscape_shuffled/per_seed/{s}",
                                           "{:.4f}"),
        "landscape_shuffled_across_wells, per seed": w3(
            "x_cross_val", "/control_gate_at_K_star/landscape_shuffled_across_wells/per_seed/{s}", "{:.4f}"),
        "random_atoms, per seed": w3("x_cross_val", "/control_gate_at_K_star/random_atoms/per_seed/{s}", "{:.4f}"),
    },
    "THE CAPTION THAT MUST TRAVEL WITH THIS TABLE - TWO CURVES, NOT ONE": (
        "TWO DIFFERENT TOTALS ARE PRINTED AND THEY MOVE IN DIFFERENT DIRECTIONS. "
        "(1) PARTS 1+2+4 ONLY - atoms, covariance/count and the coupling table, with NO address book - "
        "FALLS MONOTONICALLY, 7.4815x -> 2.0663x from K = 5 to K = 32, because the covariance shrinks "
        "6.0x (3,391,552 -> 560,608 B). "
        "(2) THE HONEST FOUR-PART TOTAL - the same three parts PLUS the measured address book - is "
        "U-SHAPED: 7.8545x at K = 5, a minimum of 3.9942x at K = 24, and 4.0877x at K* = 32, because the "
        "ADDRESS BOOK GROWS 5.4x (195,584 -> 1,059,824 B) and at the fine end it grows FASTER than the "
        "covariance falls. "
        "ANY SENTENCE THAT QUOTES '7.48x -> 2.07x' IS QUOTING CURVE (1) AND HAS DROPPED PART 3. The "
        "mechanism for curve (1) is not 'the covariance shrinks faster than the address book grows' - "
        "there is no address book in it. Curve (2), which does contain one, says the opposite at the "
        "fine end."),
    "the_object_never_reaches_the_retired_pin": (
        "the minimum honest four-part total over the whole sweep is 3.9942x 524,288 B, at K = 24. "
        "'Matched bytes at 524,288 B' is not available to this object at rank 64 under any K, which is "
        "why this wave's 1x pin is the object's own total and 524,288 B is a REFERENCE column."),
    "verdict": {
        "statement": ("every K in {5, 8, 10, 12, 16, 20, 24, 28, 32} satisfies the frozen rule, so "
                      "K* = 32 - the finest partition swept. The crossover threshold of 0.10 is never "
                      "reached; the largest drop anywhere is +0.012731 at K = 10. The range was NOT "
                      "extended to manufacture a crossover."),
        "provenance": [dsrc("x_cross_val", "/K_star"), dsrc("x_cross_val", "/K_star_candidates"),
                       dsrc("x_cross_val", "/K_crossover"), dsrc("x_cross_val", "/per_K/10/drop_from_K5")],
    },
}

'''

ANCHOR = """# --------------------------------------------------------------------------------------
# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line."""
n = t.count(ANCHOR)
assert n == 1, "anchor occurs %d times, expected 1" % n
t = t.replace(ANCHOR, NEW + ANCHOR)

# the two tiny helpers T11 and T11's w5 block need, added beside the other cell helpers
H_ANCHOR = """def w3(sid, tmpl, fmt="{:.4f}", seeds=(0, 1, 2)):"""
assert t.count(H_ANCHOR) == 1
t = t.replace(H_ANCHOR, """def w3s(arm, tmpl, fmt="{:.4f}"):
    \"\"\"A THREE-SEED cell across the three w5 READ-TABLE.json files, one per seed; `tmpl`
    carries one `%s` for the arm name.\"\"\"
    return wcell([(sid, tmpl % arm) for sid in W5], fmt=fmt, seeds=[0, 1, 2])


def w3p(sid, tmpl, fmt="{:.4f}", seeds=(0, 1, 2)):
    \"\"\"Alias of w3 kept separate so a per-seed pointer inside ONE json reads differently
    from a per-seed pointer across three.\"\"\"
    return w3(sid, tmpl, fmt=fmt, seeds=seeds)


""" + H_ANCHOR)

SRC.write_text(t, encoding="utf-8")
print("p3 OK ->", SRC)
