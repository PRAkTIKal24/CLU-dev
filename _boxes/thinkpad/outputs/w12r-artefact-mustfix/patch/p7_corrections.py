"""w12r patch 7 -- record every w12r change as a CORRECTION in the manifest, so the artefact
describes its own revision and the Head can cite each one."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)


W12R = '''
# --------------------------------------------------------------------------------------
# 9b. w12r/artefact-mustfix -- the corrections THIS pass made, each against REFEREE-v0.md.
# --------------------------------------------------------------------------------------
CORRECTIONS += [
    {
        "id": "C5 (w12r/MF3)",
        "claim_as_written": "TABLE 4 (the lead result) printed EIGHT property rows on the w9 leg and "
                            "FIVE on the w8 leg, and NO BYTE ROW on either.",
        "artefact_says": "the w8 gate file records the overrun as named fields: `honest_total_bytes` "
                         "= 12,869,482 / 12,500,170 / 12,409,722 B and `times_over_budget` = 24.5466 / "
                         "23.8422 / 23.6697 against the 524,288 B pin. The w9 ledger records "
                         "`deployed_bytes` = 1,642,752 with `statistic_bytes` = 35,904 (2.19%) "
                         "declared beside it.",
        "effect": "the paper penalised the compose-and-forget object of section 5 at 7.48x the pin and "
                  "exempted its own lead result at 24.55x - three times worse - one section later. "
                  "A referee who opens the cited JSON reads `times_over_budget: 24.5466` in the file "
                  "cited for the headline.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": f"/per_seed/{s}/times_over_budget"} for s in SEEDS]
                      + [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                          "pointer": f"/per_seed/{s}/honest_total_bytes"} for s in SEEDS]
                      + [{"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": "/ledger"}],
        "action": "every table that carries a store now prints its FOUR-PART byte ledger as its own "
                  "block, never netted: T1 section 1c-quater, T2, T4. T3 carries no store and says so.",
    },
    {
        "id": "C6 (w12r/MF4)",
        "claim_as_written": "`exact` was used without a definition anywhere in the tables.",
        "artefact_says": "the frozen autoencoder's training inventory includes eval-bank items "
                         "(`ae_train_items`: eval_bank_train 543 / 529 / 552), and `refit` = False: "
                         "the encoder and decoder fitted with the deleted rows present are untouched.",
        "effect": "without the definition the claim reads as retrain-equivalence of the generative "
                  "model, which is not what was measured, and is the standard unlearning objection.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"], "pointer": "/ae_train_items"}
                       for s in SEEDS]
                      + [{"source": "maze_s0", "path": SOURCES["maze_s0"],
                          "pointer": "/display/delete_green_exactness/refit"}],
        "action": "`/exactness_definition` is a top-level citable field (STORE-SCOPED, frozen encoder "
                  "and decoder never refit); T4's caption carries it; every place the word appears is "
                  "flagged, and the two senses (exact JUDGE vs exact DELETION) are separated.",
    },
    {
        "id": "C7 (w12r/MF7)",
        "claim_as_written": "TABLE 4's w8/downdate leg was `class RESULT`.",
        "artefact_says": "`/note`: \\"No scored run and no claim: these are properties of the object "
                         "(deviations, byte counts, row counts), not outcomes. hub/single-design scores "
                         "the object.\\" The hub commissioned to score it scored 0 of 3 (DECLARED "
                         "NOT-RUN NR3, full budget intact).",
        "effect": "a RESULT label on a leg nobody scored is the one thing in this table a referee can "
                  "call inflation. It costs the paper nothing to drop: two independent implementations "
                  "is still the claim.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/note"}],
        "action": "relabelled `ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 "
                  "SCORED RUNS`. The artefact was NOT relabelled to match the table.",
    },
    {
        "id": "C8 (w12r/MF8 + MF9)",
        "claim_as_written": "T4 printed `/downdate_equals_refit/max_abs_dev_per_well_by_seed` "
                            "(2.365e-14 / 1.126e-09 / 7.105e-15) and declared ONE unresolvable leg.",
        "artefact_says": "`/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` - the field the source "
                         "spoke's own FINAL REPORT prints, worst over the pinned 11,024-row deletion AND "
                         "the random-64 deletion - is 4.547e-13 / 1.126e-09 / 2.842e-14: 19x larger on "
                         "seed 0 and 4x larger on seed 2. And TWO legs sit below their own "
                         "reproducibility floor (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs "
                         "5.684e-14), not one.",
        "effect": "a correction whose whole point was to stop an understated figure being printed was "
                  "printing a different understated figure; and the paper was hiding two thirds of the "
                  "strongest fact it has, since an unresolvable residual is the best available outcome "
                  "for an exactness claim.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"} for s in SEEDS]
                      + [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                          "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"} for s in SEEDS],
        "action": "T4 and T5 print the spoke's own row first, the narrower field beside it and labelled, "
                  "each leg's own floor, and a resolvable? row. Figure 4 plots the corrected field and "
                  "the floors.",
    },
    {
        "id": "C9 (w12r/MF10)",
        "claim_as_written": "TABLE 1 was labelled CMAZE-1, the same label as TABLE 3.",
        "artefact_says": "the w9/maze FINAL REPORT defines the scored environment as CMAZE-2, \\"a "
                         "REQUEST layer over CMAZE-1. The engine, the solver and the exact integer judge "
                         "are `w8_envscout.maze_colour`, imported unmodified\\". TABLE 3 is CMAZE-1 proper.",
        "effect": "under one label the triple 0.4980 (published CMAZE-1) / 0.5317 (CMAZE-1 at the "
                  "harness tier) / 0.9037 (CMAZE-2) is unreadable, and correction C4 - which says 0.4980 "
                  "is a property of the REQUEST LAYER, not of CMAZE-1 - contradicts itself on the page.",
        "provenance": [{"source": "maze_log", "path": SOURCES["maze_log"], "locator": "LOG.md:366-372"},
                       {"source": "maze_s0", "path": SOURCES["maze_s0"],
                        "pointer": "/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity"},
                       {"source": "harness_s0", "path": SOURCES["harness_s0"],
                        "pointer": "/columns/blind-sequential (key order from A)/validity"}],
        "action": "T1, its caption, its pins, figures 1 and 5 and every w9 maze row are relabelled "
                  "CMAZE-2; T3 is labelled CMAZE-1 PROPER; `/environments` defines both in one line "
                  "each; T3's caption no longer says `the same environment`; the two never share a table.",
    },
    {
        "id": "C10 (w12r/MF11)",
        "claim_as_written": "\\"FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere\\".",
        "artefact_says": "true of the CROSS-MAZE quantity (0.0006 / 0.0000) and false of the SAME-MAZE "
                         "quantity on w9's own split (0.0741 / 0.5105 pooled, 0.5114 as the unweighted "
                         "seed mean).",
        "effect": "the false half is also the paper's strongest exhibit: same-maze r=2 reaches 81% of "
                  "row 2's pipeline ceiling of 0.6319 where the CLU reaches 20%.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                        "pointer": "/env_columns/fragment/FRAG-r2/validity"} for s in SEEDS]
                      + [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                          "pointer": "/env_columns/fragment_cross_maze/FRAG-cross-r2/validity"} for s in SEEDS],
        "action": "T1 section 1c-bis prints both, labelled, with the charter-B.3 not-like-for-like "
                  "caveat. T3's fragment rows are labelled CMAZE-1 tier columns.",
    },
    {
        "id": "C11 (w12r) - which mean is 0.5105?",
        "claim_as_written": "the same-maze FRAGMENT-LOOKUP r=2 figure is quoted as 0.5105 by "
                            "`w9-maze/LOG.md` and by REFEREE-v0.md MF11.",
        "artefact_says": "the three per-seed validities are 0.5342 / 0.5364 / 0.4636 at n = 584 / 604 / "
                         "632. The UNWEIGHTED seed mean is 0.51143; the n-WEIGHTED (pooled) mean is "
                         "0.51044, which rounds to the quoted 0.5105.",
        "effect": "0.0010 absolute. It changes nothing in the reading, but this file's convention "
                  "everywhere else is the unweighted per-seed mean, so the two numbers must not be "
                  "used interchangeably without saying which is which.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                        "pointer": "/env_columns/fragment/FRAG-r2/validity"} for s in SEEDS]
                      + [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                          "pointer": "/env_columns/fragment/FRAG-r2/n"} for s in SEEDS],
        "action": "T1 section 1c-bis prints the three per-seed values and the unweighted mean 0.5114; "
                  "the pooled 0.5105 is named here so the prose can cite either knowingly.",
    },
    {
        "id": "C12 (w12r) - the '9/9 seeds' phrasing",
        "claim_as_written": "\\"1e-16...1e-14 on Shapes3D, 9/9 seeds, two independent implementations\\".",
        "artefact_says": "the nine legs are 3 CONFIGURATIONS (ward_balanced K=40/44/48) x 3 SEEDS inside "
                         "ONE implementation. There are THREE implementations, and their bands are not "
                         "shared: 7.5e-16..8.8e-16 (w9/maze), 2.842e-14..1.126e-09 (w8 downdate-build), "
                         "5.551e-16..1.490e-13 (w8 single-design). And it was not unconditional: it "
                         "FAILED first on TEST seed 2 at 32.90 float32 ulp.",
        "effect": "three defects in one sentence. The strongest form of the two-implementation claim is "
                  "the FAILURE AGREEMENT - the second implementation reproduced the same seed-2 failure "
                  "with different code at 14.377 ulp against 32.90.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/a1_subspace/worst_in_float32_ulp"},
                       {"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/a1_subspace/worst_floor_subspace_in_float32_ulp"},
                       {"source": "single_design_log", "path": SOURCES["single_design_log"],
                        "locator": "LOG.md:472,479,481,495-503"},
                       {"source": "downdate_log", "path": SOURCES["downdate_log"],
                        "locator": "LOG.md:625-640,728,794,905"}],
        "action": "TABLE 5 prints the three implementations as three rows with seed count and "
                  "configuration count as SEPARATE fields, and makes the failure agreement a citable "
                  "row. `/seed_and_configuration_counts` is a top-level field. The phrase is on the "
                  "never-print list.",
    },
    {
        "id": "C13 (w12r/MF14)",
        "claim_as_written": "the charter rule \\"every composition row carries BLIND-SEQUENTIAL measured "
                            "on its own split\\", applied to all four rows.",
        "artefact_says": "`/env_columns` carries exactly four keys - `unseen_pair`, `nway`, `fragment`, "
                         "`fragment_cross_maze` - on every scored seed. There is no blind block for row "
                         "4 (budget interpolation) on any seed.",
        "effect": "the rule is unsatisfiable for row 4, and neither the tables nor the draft said so.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"], "pointer": "/env_columns"}
                       for s in SEEDS],
        "action": "T1 section 1c-ter declares it NOT MEASURED - not a blank and not a zero - and the "
                  "charter rule in `rules_applied` now carries the exception.",
    },
    {
        "id": "C14 (w12r/MF6)",
        "claim_as_written": "TABLE 4 carried no caption caveats.",
        "artefact_says": "two leaf fields of the cited gate file: `cut_is_marginal` = true on ALL THREE "
                         "seeds, and `a1_subspace.passed_every_seed` = false with `seeds_failing` = "
                         "[\\"2\\"] at 60.36282 float32 ulp against its own 66.11072 ulp floor. The "
                         "top-level `all_three_assertions_passed` = true EXCLUDES `a1_subspace`.",
        "effect": "a referee opening the paper's lead-result artefact reads `passed: false` inside it "
                  "and the paper says nothing. Stated with the floor beside it, the failure is a "
                  "property of the CONTENT and not of the downdate - which is what the artefact's own "
                  "`reproducibility_floor.what` field says.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/a1_subspace"},
                       {"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/cut_is_marginal_any_seed"},
                       {"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/reproducibility_floor/what"}],
        "action": "both are printed under T4's w8 row and both are in T4's caption.",
    },
]

'''
ANCHOR = "# 9.  FIGURES -- regenerated here, from the same pointers the tables use."
rep(ANCHOR, W12R + ANCHOR)

# the fragment block should name the pooled figure too
rep('''    "the_number_the_draft_left_on_the_floor": (''',
    '''    "which_mean": (
        "the three per-seed r=2 validities are 0.5342 / 0.5364 / 0.4636 at n = 584 / 604 / 632. This "
        "file prints the UNWEIGHTED seed mean, 0.5114. The 0.5105 quoted by `w9-maze/LOG.md` and by "
        "REFEREE-v0.md is the n-WEIGHTED (pooled) mean, 0.51044. See correction C11."
    ),
    "the_number_the_draft_left_on_the_floor": (''')
rep('''    A(f"> ⭐ **{fr['the_number_the_draft_left_on_the_floor']}**")''',
    '''    A(f"> ⭐ **{fr['the_number_the_draft_left_on_the_floor']}**")
    A("")
    A(f"> ⚠ **Which mean?** {fr['which_mean']}")''')

P.write_text(s, encoding="utf-8")
print("patch 7 applied")
