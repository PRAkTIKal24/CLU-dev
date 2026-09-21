"""w12r patch 4 -- T4: MF7 row class, MF3 four-part ledger on both legs, MF8 the field the
source spoke actually reports, MF9 both unresolvable legs, MF6 the two gate caveats,
MF4 the exactness definition attached to the table that claims exactness."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)

# ---------------- register w8-downdate/LOG.md: the four-part ledger lives there ---------
OLD_SRC = '    "downdate_val": "/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json",'
NEW_SRC = OLD_SRC + chr(10) + '    "downdate_log": "/c/chlu/outputs/w8-downdate/LOG.md",   # w12r/MF3: the four-part ledger table'
rep(OLD_SRC, NEW_SRC)


def dd_cell(tmpl, fmt="{:.4e}"):
    return ('{\n'
            '                "seeds": list(SEEDS),\n'
            f'                "values": [ptr(dd, f"{tmpl}") for s in SEEDS],\n'
            '                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],\n'
            f'                                "pointer": f"{tmpl}"}} for s in SEEDS],\n'
            f'                "format": "{fmt}",\n'
            '            }')


# ---------------- MF7: the w8 leg is an assertion battery, not a scored RESULT row -----
rep('''        "label": "w8/downdate `exp_w8_downdate` - covariance downdate == refit-without-those-rows, TEST leg",
        "row_class": "RESULT",
        "implementation": "w8 spoke/downdate-build (different code, different author)",''',
    '''        "label": "w8/downdate `exp_w8_downdate` - covariance downdate == refit-without-those-rows, TEST leg",
        "row_class": "ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 SCORED RUNS",
        "row_class_w12r_note": (
            "w12r/MF7. This row was labelled RESULT. Its own artefact says otherwise, verbatim: "
            "`/note` = 'No scored run and no claim: these are properties of the object (deviations, "
            "byte counts, row counts), not outcomes. hub/single-design scores the object.' The hub "
            "commissioned to score it - `hub/single-design` - scored 0 of 3 and is DECLARED NOT-RUN "
            "NR3, with its full budget intact. Nobody scored this leg. The artefact was NOT relabelled "
            "to match the table; the table was relabelled to match the artefact."),
        "row_class_provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/note"}],
        "scored_runs": 0,
        "implementation": "w8 spoke/downdate-build (different code, different author)",''')

# The w9 leg keeps RESULT (it IS inside scored runs S1/S2/S3) but says so explicitly.
rep('''        "label": "w9/maze `CluRead.delete_rows_exact` - delete the green target, TEST leg",
        "row_class": "RESULT",
        "implementation": "w9 hub/maze",''',
    '''        "label": "w9/maze `CluRead.delete_rows_exact` - delete the green target, TEST leg (CMAZE-2)",
        "row_class": "RESULT",
        "row_class_w12r_note": ("w12r/MF7: this leg's RESULT label is CORRECT - it sits inside the "
                                "pre-registered scored runs S1/S2/S3, 3 of 3 spent."),
        "scored_runs": 3,
        "environment": "CMAZE-2",
        "implementation": "w9 hub/maze",''')

# ---------------- MF8: print the field the source spoke's own FINAL REPORT prints -------
rep('''        "cells": {
            "max abs deviation, worst well": {
                "seeds": list(SEEDS),
                "values": [max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS],''',
    '''        "cells": {
            "max abs deviation, worst well AND worst deletion (pinned 11,024 rows + random 64) - THE SPOKE'S OWN ROW": ''' +
    dd_cell("/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev") + ''',
            "reproducibility floor for that leg (same rows downdated in two orders)": ''' +
    dd_cell("/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") + ''',
            "deviation ABOVE its own floor, i.e. resolvable at all?": derived(
                [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev")
                 > ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS],
                [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                  "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"} for s in SEEDS]
                + [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                    "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"} for s in SEEDS],
                "worst_max_abs_dev > floor_max_abs_dev, per seed", fmt="{}",
                note=("w12r/MF9: TWO of the three TEST seeds sit BELOW their own measured "
                      "reproducibility floor (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs "
                      "5.684e-14), so on those legs the residual is NOT RESOLVABLE and the claim "
                      "rests on the floor rather than on the deviation. The draft declared one.")),
            "max abs deviation over the PINNED deletion only (w12r/MF8: the narrower field the draft printed)": {
                "seeds": list(SEEDS),
                "values": [max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS],''')

rep('''                "format": "{:.3e}",
                "mean": sum(max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS) / 3,
            },''',
    '''                "format": "{:.3e}",
                "mean": sum(max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS) / 3,
                "w12r_note": ("MF8. This field covers the PINNED 11,024-row deletion only. The source "
                              "spoke's own FINAL REPORT prints the worst over the pinned deletion AND "
                              "the random-64 deletion, which is 19x larger on seed 0 and 4x larger on "
                              "seed 2. Both are printed; the spoke's own row is the one to quote."),
            },''')

# ---------------- MF6: the two gate caveats that belong in the caption ------------------
rep('''        "summary_from_artefact": {
            "worst_max_abs_dev_over_seeds": ptr(dd, "/downdate_equals_refit/worst_max_abs_dev_over_seeds"),''',
    '''        "caveats_in_the_gate_file_itself": {
            "w12r_note": "MF6. Both are leaf fields of the very file this row cites, and neither was carried.",
            "(a) admission cut is MARGINAL on every seed": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/cut_is_marginal") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/cut_is_marginal"} for s in SEEDS],
                "format": "{}",
                "what_it_means": ("a direction sits near the threshold, so a small perturbation could "
                                  "change WHICH directions are kept - the same instability class as the "
                                  "original seed-2 failure. It does not block the gate; it belongs in "
                                  "the caption of anything scored on this object."),
                "flagged_by": {"source": "single_design_log", "path": SOURCES["single_design_log"],
                               "locator": "LOG.md, 2026-09-20T22:45Z entry"},
            },
            "(b) the SUBSPACE-level assertion a1_subspace FAILS on seed 2": {
                "passed_every_seed": ptr(dd, "/a1_subspace/passed_every_seed"),
                "seeds_failing": ptr(dd, "/a1_subspace/seeds_failing"),
                "worst_in_float32_ulp": ptr(dd, "/a1_subspace/worst_in_float32_ulp"),
                "worst_floor_subspace_in_float32_ulp": ptr(dd, "/a1_subspace/worst_floor_subspace_in_float32_ulp"),
                "reading": ("the byte-level assertion `downdate_equals_refit` passes on all three seeds; "
                            "the separate subspace-level assertion `a1_subspace` on the PRE-CUT store "
                            "fails on seed 2 at 60.36 ulp - BELOW its own 66.11 ulp floor - so that "
                            "failure is a property of the CONTENT and not of the downdate, which is what "
                            "the artefact's own `reproducibility_floor.what` field states."),
                "watch_out": ("the top-level `all_three_assertions_passed = true` EXCLUDES `a1_subspace`. "
                              "A referee opening the cited JSON reads `passed: false` inside the lead "
                              "result's own artefact."),
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/a1_subspace"},
                               {"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": "/reproducibility_floor/what"}],
            },
        },
        "summary_from_artefact": {
            "worst_max_abs_dev_over_seeds": ptr(dd, "/downdate_equals_refit/worst_max_abs_dev_over_seeds"),''')

# ---------------- MF3: the four-part ledger on BOTH legs of T4 -------------------------
T4_LEDGER = '''# --- w12r/MF3: the four-part byte ledger on BOTH legs of T4 ---------------------------
# The residual below is the one number the source LOG's own four-part table does not name.
# store + covariance + count table + coupling + holes leaves exactly 16,424 B unaccounted on
# EVERY test seed.  16,424 = sum_k d_k x 8 (= 16,384 B, the running MEANS, float64, and
# sum_k d_k = 2,048 = D on every seed) + 5 x 8 (= 40 B, the per-well COUNTS).  With that, the
# ledger closes EXACTLY against honest_total_bytes on all three seeds.  The identification of
# the residual is w12r's inference from `d_k`; the residual itself is arithmetic.
_DK = [[w["d_k"] for w in ptr(dd, f"/per_seed/{s}/content_cost/per_well")] for s in SEEDS]
_COUNT_TABLE_B = 11488
_COUPLING_B, _HOLES_B = 5288, 1322
_MEANS_AND_COUNTS_B = [sum(dk) * 8 + len(dk) * 8 for dk in _DK]

T4_BYTE_LEDGER = {
    "rule": "FOUR PARTS, PRINTED SEPARATELY, NEVER NETTED INTO ONE NUMBER.",
    "w9/maze leg (CMAZE-2)": {
        "see": "T1.byte_ledger - the same object, the same fields, printed in full there",
        "1. atoms (the store)": derived([ptr(DOCS[sid], "/ledger/store_atom_floats") * 4 for sid in MAZE],
                                        [dsrc(sid, "/ledger/store_atom_floats") for sid in MAZE],
                                        "store_atom_floats x 4 (float32)"),
        "2. sufficient statistic, float64": cell(MAZE, "/ledger/statistic_bytes", DOCS, fmt="{:,.0f}"),
        "3. address book (the head)": derived([ptr(DOCS[sid], "/ledger/head_floats") * 4 for sid in MAZE],
                                              [dsrc(sid, "/ledger/head_floats") for sid in MAZE],
                                              "head_floats x 4 (float32)"),
        "4. coupling / hole table": {"status": "DECLARED NOT PRESENT in any w9/maze artefact"},
        "deployed (atoms + address book)": cell(MAZE, "/ledger/deployed_bytes", DOCS, fmt="{:,.0f}"),
        "statistic as a share of deployed": derived(
            [ptr(DOCS[sid], "/ledger/statistic_bytes") / ptr(DOCS[sid], "/ledger/deployed_bytes") for sid in MAZE],
            [dsrc(sid, "/ledger/statistic_bytes") for sid in MAZE],
            "statistic_bytes / deployed_bytes", fmt="{:.2%}"),
        "bytes the deletion's basis-change carry deploys": cell(
            MAZE, "/display/delete_green_exactness/receipt/basis_change_bytes_deployed", DOCS, fmt="{:,.0f}"),
        "pin_status": "NOT under the 524,288 B pin - this leg is matched on PARAMETERS, not on store bytes.",
    },
    "w8/downdate leg (Shapes3D, the 524,288 B pin at tolerance 0)": {
        "1. atoms - principal subspaces (THE STORE, and it matches the pin exactly)": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/store_bytes") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/store_bytes"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "2a. running covariance, float64 sum_k d_k^2": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/cov_bytes_per_seed/{s}") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/cov_bytes_per_seed/{s}"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "2b. running means + per-well counts (the residual the source table does not name)": derived(
            _MEANS_AND_COUNTS_B,
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/per_seed/{s}/content_cost/per_well"} for s in SEEDS],
            "sum_k d_k x 8 + n_wells x 8 (float64 means, float64 counts)",
            note=("16,424 B on every seed. It is exactly the residual left by honest_total_bytes minus "
                  "store minus covariance minus the count table minus coupling minus holes, so the "
                  "four-part ledger closes to the byte once it is printed.")),
        "2. covariance / count ledger, TOTAL": derived(
            [ptr(dd, f"/cov_bytes_per_seed/{s}") + _MEANS_AND_COUNTS_B[s] for s in SEEDS],
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/cov_bytes_per_seed/{s}"} for s in SEEDS],
            "cov_bytes + means_and_counts"),
        "3. request count table": locator(
            [_COUNT_TABLE_B] * 3, "downdate_log", "LOG.md:420 and :1041",
            "part 3 of the source spoke's own four-part ledger table", fmt="{:,.0f}"),
        "4. coupling + holes": locator(
            [f"{_COUPLING_B:,} + {_HOLES_B:,}"] * 3, "downdate_log", "LOG.md:421 and :1042",
            "part 4 of the source spoke's own four-part ledger table: coupling table + one byte per hole",
            fmt="{}"),
        "honest total (all four parts)": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/honest_total_bytes") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/honest_total_bytes"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "x THE 524,288 B PIN": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/times_over_budget") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/times_over_budget"} for s in SEEDS],
            "format": "{:.4f}x",
        },
        "ledger closes to the byte?": derived(
            [ptr(dd, f"/per_seed/{s}/store_bytes") + ptr(dd, f"/cov_bytes_per_seed/{s}")
             + _MEANS_AND_COUNTS_B[s] + _COUNT_TABLE_B + _COUPLING_B + _HOLES_B
             == ptr(dd, f"/per_seed/{s}/honest_total_bytes") for s in SEEDS],
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/per_seed/{s}/honest_total_bytes"} for s in SEEDS],
            "store + cov + means_and_counts + 11,488 + 5,288 + 1,322 == honest_total_bytes", fmt="{}"),
        "serialised object bytes, for comparison": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/items_kept_detail/object_bytes_full") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/items_kept_detail/object_bytes_full"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "DECLARED RESIDUAL, unexplained": derived(
            [ptr(dd, f"/per_seed/{s}/items_kept_detail/object_bytes_full")
             - ptr(dd, f"/per_seed/{s}/honest_total_bytes") for s in SEEDS],
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/per_seed/{s}/items_kept_detail/object_bytes_full"} for s in SEEDS],
            "object_bytes_full - honest_total_bytes",
            note=("exactly 16,384 B on every one of the three seeds. w12r could not attribute it to a "
                  "named field and does NOT net it away: it is reported as a declared residual. "
                  "It does not move `times_over_budget`, which the artefact computes from "
                  "honest_total_bytes.")),
        "THE HEADLINE THE DRAFT OMITTED": (
            "the object that forgets exactly is itself 24.55x / 23.84x / 23.67x the 524,288 B pin. "
            "The paper penalises the compose-and-forget object of section 5 at 7.48x; its own lead "
            "result is three times worse and printed no byte row at all."),
        "provenance_of_the_four_part_split": [
            {"source": "downdate_log", "path": SOURCES["downdate_log"],
             "locator": "LOG.md:414-422 (section 3, THE LEDGER) and LOG.md:1036-1044 (FINAL REPORT "
                        "section 3) - the source spoke's own four-part table. Parts 3 and 4 exist only "
                        "in that markdown, so they are carried as locators and flagged as not "
                        "machine-checkable by verify_tables.py."},
        ],
    },
}

TABLES["T4"] = {'''
rep('TABLES["T4"] = {', T4_LEDGER)

rep('''    "rows": t4_rows,
}''',
    '''    "rows": t4_rows,
    "byte_ledger": T4_BYTE_LEDGER,
    "exactness_definition_ref": "TABLES.json /exactness_definition (EXACT-STORE-SCOPED)",
    "exact_is_not_free": {
        "w12r_note": "MF4/MF5: exactness is claimed; cost-free erasure is not.",
        "surviving-target composition on TEST, BEFORE": cell(
            MAZE, "/display/delete_green/other_before/validity", DOCS),
        "surviving-target composition on TEST, AFTER": cell(
            MAZE, "/display/delete_green/other_after/validity", DOCS),
    },
}''')

rep('''        "Both legs are 3 seeds on generation 2 on their own TEST condition. They are reported side by side "
        "as two implementations of one claim, not as one instrument: the deviations are in different units "
        "(w9/maze reports a RELATIVE residual on the statistic and a projector residual; w8/downdate reports "
        "an ABSOLUTE max deviation per well)."''',
    '''        "EXACT HERE MEANS STORE-SCOPED (TABLES.json /exactness_definition): after deleting a set of rows the "
        "STORE CONTENT equals, to float roundoff, the content fitted from the retained rows alone, HOLDING THE "
        "FROZEN ENCODER AND DECODER FIXED. Neither is refit. It is not a claim that the renderer forgets. "
        "Both legs are 3 seeds on generation 2 on their own TEST condition. They are reported side by side "
        "as two implementations of one claim, not as one instrument: the deviations are in different units "
        "(w9/maze reports a RELATIVE residual on the statistic and a projector residual; w8/downdate reports "
        "an ABSOLUTE max deviation per well). "
        "CAVEATS THAT TRAVEL WITH THE w8 LEG (w12r/MF6): the store is built under a variance-share admission "
        "cut that is MARGINAL on every seed (`cut_is_marginal_any_seed = true`); the byte-level assertion "
        "`downdate_equals_refit` passes on all three seeds, while the separate subspace-level assertion "
        "`a1_subspace` on the PRE-CUT store FAILS on seed 2 at 60.36 ulp - below its own 66.11 ulp floor - so "
        "that failure is a property of the content, not of the downdate; and the top-level "
        "`all_three_assertions_passed = true` EXCLUDES `a1_subspace`. "
        "THE w8 LEG IS NOT A SCORED ROW (w12r/MF7): it is an assertion battery on the object at 0 scored runs; "
        "the hub commissioned to score it returned its full budget (DECLARED NOT-RUN NR3). "
        "THE BYTE LEDGER IS PRINTED BELOW AND IS NEVER NETTED (w12r/MF3): the w8 object's honest total is "
        "24.55x / 23.84x / 23.67x the 524,288 B pin."''')

P.write_text(s, encoding="utf-8")
print("patch 4 applied")
