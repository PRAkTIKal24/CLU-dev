"""w12r patch 5 -- TABLE 5: the three implementations as THREE ROWS, plus the failure-agreement
row.  Closes the '9/9 seeds' phrasing that w9-advisor, w10-advisor and w10/registry-fold
independently flagged."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)


T5 = '''
# --------------------------------------------------------------------------------------
# 6b. w12r -- TABLE 5.  THREE IMPLEMENTATIONS, THREE ROWS.
#     "9/9 seeds, two independent implementations" is wrong in three ways at once:
#     the nine legs are 3 CONFIGURATIONS x 3 seeds inside ONE implementation; the band is
#     not shared; and it was not unconditional - it FAILED first.  There are THREE
#     implementations, and the strongest form of the claim is the FAILURE AGREEMENT.
# --------------------------------------------------------------------------------------
sd = "single_design_log"
t5_rows = [
    {
        "implementation": "1. w9 hub/maze - `CluRead.delete_rows_exact` (CMAZE-2, TEST)",
        "row_class": "RESULT - 3 of 3 scored runs (S1/S2/S3)",
        "seeds": 3, "configurations": 1, "legs": 3, "scored_runs": 3,
        "configuration": "K=8, n_atoms=31 - one pinned configuration",
        "metric": "max RELATIVE residual on the sufficient statistic; max projector residual on the subspace",
        "cells": {
            "statistic max relative residual": cell(
                MAZE, "/display/delete_green_exactness/statistic_max_rel_residual", DOCS, fmt="{:.4e}"),
            "subspace max projector residual": cell(
                MAZE, "/display/delete_green_exactness/subspace_max_projector_residual", DOCS, fmt="{:.4e}"),
            "refit performed": cell(MAZE, "/display/delete_green_exactness/refit", DOCS, fmt="{}"),
            "passes over retained data": cell(
                MAZE, "/display/delete_green_exactness/passes_over_retained_data", DOCS, fmt="{:.0f}"),
        },
        "band": "7.522e-16 .. 8.826e-16 on the statistic; EXACTLY 0.000e+00 on the subspace projector",
        "note": ("this is the leg that is unambiguously in the 1e-16 band, it is the only one inside "
                 "scored runs, and it replicates from a clean checkout with ZERO numeric differences "
                 "(`/c/chlu/outputs/w9-maze/replication/REPLICATION.md`)"),
    },
    {
        "implementation": "2. w8 spoke/downdate-build - `exp_w8_downdate` (Shapes3D, TEST)",
        "row_class": "ASSERTION BATTERY (object property) - 0 SCORED RUNS",
        "seeds": 3, "configurations": 1, "legs": 3, "scored_runs": 0,
        "configuration": "factor partition K=5, A=64 - the incumbent's grain",
        "metric": "max ABSOLUTE deviation per well, downdate vs refit-without-those-rows",
        "cells": {
            "worst max abs deviation (pinned 11,024 rows + random 64)": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"}
                               for s in SEEDS],
                "format": "{:.4e}",
            },
            "its own reproducibility floor": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"}
                               for s in SEEDS],
                "format": "{:.4e}",
            },
        },
        "band": "2.842e-14 .. 1.126e-09 - FIVE ORDERS COARSER than implementation 1",
        "note": ("the worst leg (TEST seed 1, 1.126e-09) sits BELOW its own reproducibility floor of "
                 "4.282e-09, and so does seed 2 (2.842e-14 vs 5.684e-14): on TWO of three legs the "
                 "residual is not resolvable at all. Printed that way it is STRONGER, not weaker - an "
                 "unresolvable residual is the best available outcome for an exactness claim. "
                 "The two additional val legs the advisor quotes are 4.547e-13 and 1.164e-10."),
    },
    {
        "implementation": "3. w8 hub/single-design - matched-bytes candidates (Shapes3D, TEST)",
        "row_class": "ASSERTION BATTERY (object property) - 0 SCORED RUNS",
        "seeds": 3, "configurations": 3, "legs": 9, "scored_runs": 0,
        "configuration": "ward_balanced K=40 / K=44 / K=48 - THREE configurations, three seeds each",
        "metric": "max atom deviation in float32 ulp at unit scale; and the same at subspace level",
        "cells_markdown_only": {
            "atom-level ulp, worst over the 9 legs": locator(
                [0.094], sd, "LOG.md:476-481", "9/9 PASS, worst observed 0.0938 ulp", fmt="{:.4f}"),
            "subspace-level ulp, worst over the 9 legs": locator(
                [0.0251], sd, "LOG.md:495-503", "matched-bytes 9/9 PASS, worst subspace 0.0251 ulp",
                fmt="{:.4f}"),
            "absolute deviation, seed 0, K=40 / K=44 / K=48": locator(
                [5.551e-16, 1.421e-14, 1.490e-13], sd, "LOG.md:472",
                "after the sign-convention fix, test leg seed 0", fmt="{:.4e}"),
        },
        "band": "5.551e-16 .. 1.490e-13 on seed 0; worst ulp over all nine legs 0.094 (atom) / 0.0251 (subspace)",
        "note": ("THIS is the implementation the phrase '9/9' came from, and the nine are 3 "
                 "CONFIGURATIONS x 3 SEEDS, not nine seeds. It passes under BOTH the atom definition "
                 "and the subspace definition, so it does not need the definition change to survive. "
                 "Every number in this row lives in a MARKDOWN artefact and is carried as a locator: "
                 "verify_tables.py re-hashes the file but cannot re-resolve these values."),
        "machine_checked_by_verify_tables": False,
    },
]

T5_FAILURE_AGREEMENT = {
    "title": "THE STRONGEST FORM OF THE TWO-IMPLEMENTATION CLAIM IS THE FAILURE AGREEMENT, NOT THE SUCCESSES",
    "what_happened": (
        "Two implementations, different code, different authors, different sign conventions, "
        "reproduced THE SAME failure on THE SAME seed in THE SAME well: the incumbent's coarse, "
        "imbalanced `factor K=5, A=64` partition on TEST seed 2."
    ),
    "legs": {
        "w8 spoke/downdate-build, atom level": locator(
            [32.90], "downdate_log", "LOG.md:625-640 (3.921e-06 = 32.90 ulp) and LOG.md:728,794,905 (4.227e-06 = 35.46 ulp)",
            "assertion 1 FAILS on test seed 2 against a declared criterion of <= 1 float32 ulp", fmt="{:.2f}"),
        "w8 spoke/downdate-build, subspace level (a stored field)": {
            "seeds": [2], "values": [ptr(dd, "/a1_subspace/worst_in_float32_ulp")],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": "/a1_subspace/worst_in_float32_ulp"}],
            "format": "{:.5f}",
        },
        "w8 spoke/downdate-build, its own subspace floor (a stored field)": {
            "seeds": [2], "values": [ptr(dd, "/a1_subspace/worst_floor_subspace_in_float32_ulp")],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": "/a1_subspace/worst_floor_subspace_in_float32_ulp"}],
            "format": "{:.5f}",
        },
        "w8 hub/single-design, atom level": locator(
            [14.377], sd, "LOG.md:479,481", "the same seed-2 failure, reproduced independently", fmt="{:.3f}"),
        "w8 hub/single-design, subspace level": locator(
            [14.4518], sd, "LOG.md:501", "the subspace definition does NOT rescue it", fmt="{:.4f}"),
        "w8 hub/single-design, its own two-order floor": locator(
            [36.9305], sd, "LOG.md:501,506",
            "the floor EXCEEDS the observed deviation: seed 2's subspace is not a function of its data",
            fmt="{:.4f}"),
    },
    "the_citable_sentence": (
        "The second implementation reproduced the same seed-2 failure with different code and its own "
        "sign convention - 14.377 ulp against 32.90 - and both implementations' own reproducibility "
        "floors EXCEED their observed deviations there, so both independently conclude that seed 2's "
        "content is not a function of its data by any method. That is a stronger corroboration than "
        "either implementation's successes."
    ),
    "how_it_was_closed": {
        "fix": "a variance-share admission cut with NO FREE PARAMETER, declared before the re-run and not moved",
        "tau": ptr(dd, "/per_seed/0/content_cost/var_share_tol"),
        "tau_provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/content_cost/var_share_tol"} for s in SEEDS],
        "tau_derivation": "4.941056e-16 / 1.1920929e-07 = 4.144858e-09",
    },
    "what_must_never_be_printed": (
        "'1e-16..1e-14 on Shapes3D, 9/9 seeds, two independent implementations'. Wrong three ways: "
        "the nine legs are 3 configurations x 3 seeds inside ONE implementation; the band is not "
        "shared (the peer runs 2.842e-14 .. 1.126e-09, five orders coarser, its worst leg below its "
        "own 4.282e-09 reproducibility floor); and it was not unconditional - it FAILED first at "
        "32.90 ulp."
    ),
}

TABLES["T5"] = {
    "title": "Exact deletion - THE THREE IMPLEMENTATIONS AS THREE ROWS, and the failure they agree on",
    "w12r_note": ("Added by w12r spoke/artefact-mustfix. There are THREE implementations, not two, and "
                  "their bands are NOT shared. Seed count and configuration count are separate fields "
                  "on every row so that no sentence can multiply them into a seed count."),
    "instrument": ("three different codebases, three different authors, compared only on the QUESTION "
                   "(does the downdate equal the refit?), never merged into one band"),
    "pins": {
        "generation": 2,
        "condition": "TEST on all three implementations",
        "exactness_scope": "STORE-SCOPED - see TABLES.json /exactness_definition",
    },
    "pin_provenance": [
        {"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/condition"},
        {"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": "/which"},
    ],
    "caption": (
        "THE BANDS ARE NOT SHARED AND ARE NOT MERGED. Implementation 1 (w9/maze, CMAZE-2) is at "
        "7.5e-16..8.8e-16 with a projector residual of exactly 0.000e+00 and is the only one inside "
        "scored runs. Implementation 2 (w8 spoke/downdate-build) runs 2.842e-14..1.126e-09 - five "
        "orders coarser - with TWO of its three TEST legs below their own reproducibility floors. "
        "Implementation 3 (w8 hub/single-design) is 3 CONFIGURATIONS x 3 SEEDS = 9 legs, worst 0.094 "
        "atom ulp / 0.0251 subspace ulp. Nine legs are NOT nine seeds."
    ),
    "rows": t5_rows,
    "failure_agreement": T5_FAILURE_AGREEMENT,
    "seed_and_configuration_counts": SEED_AND_CONFIGURATION_COUNTS,
}

'''
rep("# --------------------------------------------------------------------------------------\n"
    "# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line.",
    T5 + "# --------------------------------------------------------------------------------------\n"
         "# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line.")

# ---------------- manifest top level + the rules ---------------------------------------
rep('''        "sources": src_manifest,
        "tables": TABLES,''',
    '''        "exactness_definition": EXACTNESS_DEFINITION,
        "environments": ENVIRONMENTS,
        "seed_and_configuration_counts": SEED_AND_CONFIGURATION_COUNTS,
        "w12r_revision": {
            "spoke": "w12r spoke/artefact-mustfix",
            "closes": ["MF3 (the byte row)", "MF4 (define exact)", "MF6 (the two gate caveats)",
                       "MF7 (the mislabelled row)", "MF8 (the field the source spoke reports)",
                       "MF9 (both unresolvable legs)", "MF10 (CMAZE-1 vs CMAZE-2)",
                       "MF11 (same-maze vs cross-maze fragment)", "MF14 (row 4's declared BLIND gap)",
                       "the '9/9 seeds' phrasing (T5)"],
            "scored_runs_spent": 0,
            "referee": "/c/chlu/evidence/outputs/w10-referee/REFEREE-v0.md",
        },
        "sources": src_manifest,
        "tables": TABLES,''')

rep('''            "CMAZE-1's n-way row is reported as a CONJUNCTION and PER ROUTE, because its ceilings are per route.",
            "Every CMAZE-1 composition row carries BLIND-SEQUENTIAL measured on its own split, not the published 0.4980.",''',
    '''            "CMAZE-2's n-way row is reported as a CONJUNCTION and PER ROUTE, because its ceilings are per route.",
            "Every CMAZE-2 composition row carries BLIND-SEQUENTIAL measured on its own split, not the published 0.4980; where no BLIND block exists (row 4) that is DECLARED, never blank.",
            "w12r/MF10: T1 is CMAZE-2 (a request layer over CMAZE-1) and T3 is CMAZE-1 proper. They are a DIFFERENT ENVIRONMENT, not merely a different budget, and they never share a table.",
            "w12r/MF3: every table that carries a store prints its FOUR-PART byte ledger - atoms, covariance/count ledger, address book, coupling/hole table - separately, NEVER netted into one number.",
            "w12r/MF4: `exact` is STORE-SCOPED and the frozen encoder and decoder are never refit; the definition is a citable field at TABLES.json /exactness_definition.",
            "w12r: never `9/9 seeds`. Seed count and configuration count are separate fields, and the three implementations are three rows (T5).",''')

P.write_text(s, encoding="utf-8")
print("patch 5 applied")
