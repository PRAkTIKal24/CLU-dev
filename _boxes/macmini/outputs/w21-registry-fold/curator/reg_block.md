
## ◈.15 WHAT CHANGED SINCE THE FOURTH PASS, AND WHAT THIS PASS READ

**Commissioned by** the w21 `registry-fold` brief (`~/clu/tasks/w21/registry-fold.md`, 2026-09-24; pre-authorised in full). **Base:** this file's own
pass 4 (`N366`–`N371`, `◈.12`–`◈.14`; 7,466 lines, md5 `c9e69d4e…`). ⛔ Pass-1 to pass-4 entries are **not edited**. The only changes to them are
**inserted** `⟲ PASS 5` lines. Nothing was replaced, deleted or renumbered. This block adds **`N372`–`N390`**.

⛔⛔ **The standing rule that changed (Head, 2026-09-24, all waves): pre-registrations and pre-declarations are BANNED.** Report everything, omit
nothing. ⇒ In this block:
- a PREREG bar is quoted **as the historical record of what the spoke declared**, never as authority over a measured number;
- every number a spoke withheld **because a pre-registration did not cover it** is folded **with its status and the reason it was withheld**
  (`N380`, `N390`);
- the curator writes no bar, label or prediction of its own. Where two estimands disagree, both are printed and ⛔ **no label is computed**.

**What landed, and what this pass read** (unprefixed paths are relative to `~/clu/evidence/outputs/`):
- **w20 `shapes-data-matched`** (START `2026-09-23T17:26:18Z`, END `2026-09-24T13:36:07Z`; 3 of 3 scored runs spent; target exhausted):
  `FINDINGS.md` (262 lines) and `LOG.md` in full; `AMENDMENT-1.md` and `AMENDMENT-2.md` (the Head's); `PREREG-w20-shapes-data-matched-1.md` §3–§6 and its
  appended A1–A5.4. **Numbers are taken from `SHAPES-MATCHED.json`** (`rows34.*`, `headline`, `budget`, `deletion_matched.per_scope.*`, `ladder.<n>.*`,
  `LACE_PoE_at_131072`) and `deletion_store/seed0/conjunction_live.json`. Where FINDINGS rounds, the JSON value is quoted beside it.
- **w21 `words-full-test`**: `FINDINGS.md`, `PREREG-w21-words-full-test.md`, `test/BARS-WORDS-FULL-test.json` (`HEAD_LABEL`), `repro/REPRO_COMPARE.txt`.
- **w21 `maze-t-clu-improve`**: `FINDINGS.md`, `LOG.md`, `PREREG-w21-maze-t-clu-improve.md`, `test/LABEL-BOTH-test.json`, `val_full/LABEL-BOTH-val.json`,
  `val_full/LABEL-WORDS-FULL-val.json`; `val/sweep_r{1..5}.json` **counted by the curator** (69 entries including the A1 reproduction; 57 `gate_pass`).
- **w21 `maze-t-seed4`**: `FINDINGS.md`, `AMENDMENT-1.md` (the Head's), `MAZE-T-SEED4.json`, `TABLES.md` (l.1–30, l.194–210, l.1100–1160),
  `test/BOTH_test_seed3.json` (`matched_parameters`, `rival_training`).
- **Advisor reports**: `w20-advisor-report.md` in full; `w19-advisor-report.md` §0a and the correction note (l.8–40); `w18-advisor-report.md` l.44–52.
- **Re-read for a withheld row:** `w13-seeds-and-tables/TABLES.md` l.703–722 (`learned_1x`).
- **Curator arithmetic** (the only derived numbers in this block) is produced by a script, not by hand:
  `~/clu/outputs/w21-registry-fold/curator/curator_arithmetic.py` → `curator_arithmetic.json` (cited below as **CA**). It reads evidence only.

⛔ **Rules this pass kept:** ZERO scored runs, ZERO compute beyond the CA script (JSON reads and means). No ledger written. No draft edited. The
`.tex` lock was not taken. `~/clu/evidence/` was not written. ⚠ Every `◈` tier is **CURATOR-PROPOSED**.

## ◈.16 ⭐ THE ENTRIES — `N372`–`N390`

**Reading key (w20).** "Novel" = distinct valid novel requested tuples per corner, rows 3–4 TEST (60 pair + 49 n-way = 109 corners), seeds 0 / 1 / 2.
**TENX** = 10× the optimiser steps; **P1** = the declared convergence gate (TENX − DECLARED ≤ 1.0). **Store** = K\* = 32 on 131,072 rows unless marked.

### N372 — ⭐⭐ **THE DATA-MATCHED SHAPES MARGIN IS +7.03, NOT +39…+49: IT SHRINKS ABOUT 6× AND DOES NOT COLLAPSE. Seeds +7.49 / +8.60 / +5.02; +6.54 against the rival at its best over every fit that ran (30× included); lowest CI lower bound +3.73. The draft's margin, recomputed from the same score files, is +39.7 … +49.1.** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / claims hygiene (w20 hub `shapes-data-matched`, scored runs 1/3 and 2/3, TEST, 3 seeds; ⭐ **the data-matched test M1 asked for**)

**Source:** `w20-shapes-data-matched/SHAPES-MATCHED.json` → `headline`, `rows34.P3_margin.store_K32_n131072.{vs_strongest,per_family}`,
`rows34.P3_vs_best_over_all_fits_incl_30x`, `rows34.draft_margin_recomputed_vs_16k_comparators`, `rows34.P1_convergence`; `FINDINGS.md` §0, §2–§4 (l.14–141).

- **The headline, exact** (`headline.data_matched_margin_rows34_K32`): per seed 7.486238532110092 / 8.596330275229358 / 5.018348623853211, mean
  **7.033639143730887**, verdict `SHRINKS`. Against the best over all fits incl. 30×: mean **6.54434250764526** (seed 0 +6.02 [3.83, 8.33] against
  the 30× MLP). Draft margin recomputed: min 39.71559633027523, max 49.06422018348624 (`headline.draft_margin_recomputed`).
- **The rivals** (`FINDINGS.md` §2): `mlp`, `lstm`, `transformer`, `cont_mlp` at the 131,072-parameter read-path pin, fitted on the store's **exact**
  131,072-row pool, spoke-A protocol, loss curves printed. **Every family selected lr = 3e-3 and 40 epochs — the edge of both search grids**, and the
  grids were **not** widened (PREREG A1.2). ⚠ *"the rival search space may still be truncated"* (`w20-advisor-report.md` §1).
- **P1 (convergence), per seed** (`rows34.P1_convergence`):

  | family | DECLARED | TENX | TENX − DECLARED | P1 | 30× (diagnostic) |
  |---|---|---|---|---|---|
  | mlp | 36.71 / 37.57 / 43.17 | 40.01 / 38.81 / 44.10 | +3.30 / +1.24 / +0.93 | FAIL / FAIL / PASS | 41.48 / 36.54 / 43.50 — ⚠ **s0 still +1.47 over TENX: NOT CONVERGED at 10×** |
  | lstm | 33.97 / 32.36 / 40.70 | 33.69 / 32.54 / 39.85 | −0.28 / +0.18 / −0.84 | PASS ×3 | — |
  | transformer | 25.56 / 27.79 / 38.65 | 25.30 / 27.24 / 37.84 | −0.26 / −0.55 / −0.81 | PASS ×3 | — |
  | cont_mlp | 17.92 / 16.54 / 26.84 | 28.53 / 32.94 / 42.60 | +10.6 / +16.4 / +15.8 | FAIL ×3 | 27.28 / 34.01 / 38.71 (s1 +1.07 over TENX, flagged) |

- **Other margins** (`FINDINGS.md` §3–§4): K = 5 @131,072 **+4.78**; M-16k (store refit at the rivals' sizes, against the 131k data-matched TENX MLP):
  K\* @16,384 **+6.30**, K\* @32,768 **+6.86**, K = 5 @16,384 **+4.30**, K = 5 @32,768 **+4.74**. The store beats the strongest data-matched rival on
  **67–84 of 109** corners per seed. Controls (P2): all four read 0.000 on all six store variants × 3 seeds; `landscape_shuffled_across_wells` is a
  derangement, printed on every row.
- ⚠ **A label the JSON does not bear out, in the conservative direction.** `headline.rule` says *"P1-carried strongest data-matched rival"*. At seed 2 the
  MLP **passed** P1, so the P1-carried fit is DECLARED, and the literal P1-carried margin is 5.944954128440367 (mean over seeds 7.342507645259938;
  `rows34.P3_margin.store_K32_n131072.per_family.mlp`, CA `w20_rows34_P1_carried_rule_literal`). The headline instead uses `max(DECLARED, TENX)`
  (`vs_strongest`, labelled *"strongest_rival (max of declared/tenx)"*), which gives 5.018348623853211. ⇒ **+7.03 is the smaller, safer number**; the
  label, not the value, is wrong (`N381` item 4).
- ⛔ **Scope.** The margin is ⭐ **positive on every seed with every CI above 0**, and ⛔ **its size is ≈ +7, not +39…+49.** w20's own sentence:
  *"The draft's +39…+49 is not a data-matched number; it must be replaced by ≈ +7 (rows 3–4, K\* = 32), with the P1 table beside it"* (FINDINGS l.255–256).
- **Matrix row:** `CM-W20-1`. ⟲ at `N339`.

### N373 — ⭐⭐ **M1's CAUSE WAS MISDIAGNOSED: THE RIVALS' DEFICIT WAS TRAINING SCHEDULE, NOT DATA. An MLP on the SAME 16,384 rows the draft's comparator used scores 38.6 at 20,480 steps and 5.5 at 1,280 steps. At matched steps, 8× more data buys the MLP ≈ 0.5 novel tuple per corner, and the store is equally data-insensitive (47.28 at 16,384 rows → 48.01 at 131,072).** · tier A (CURATOR-PROPOSED ◈) · benchmark design / claims hygiene (w20 hub `shapes-data-matched`, run 1/3 diagnostic columns PREREG A1.1 / A2.3; ⭐ **an entry in its own right**)

**Source:** `w20-shapes-data-matched/SHAPES-MATCHED.json` → `rows34.arms.{mlp__declared@test16k_stepmatched, mlp__declared@test32k_stepmatched,
mlp__declared@test, w12comp_mlp__as_trained, store_K32_n16384, store_K32_n131072}.all`; `FINDINGS.md` §0 (l.24–31); `LOG.md` 06:57Z (RECONCILIATION).

| MLP fit (read-path pin) | fit rows | optimiser steps | novel / corner, mean (JSON) |
|---|---|---|---|
| the draft's comparator (`w12comp_mlp__as_trained`) | 16,384 | **1,280** | **5.535168195718654** |
| `learned_ref_ownpool` (EXISTING, w12) | 32,768 | 2,560 | 11.81 |
| step-matched (A1.1) | **16,384** | 20,480 | **38.60550458715596** |
| step-matched (A2.3) | 32,768 | 20,480 | 39.06727828746177 |
| DECLARED | 131,072 | 20,480 | 39.149847094801224 |

- **The store side** (`store_K32_n16384`, `store_K32_n131072`): 47.27522935779816 → 48.006116207951074. *"So 'the store only won on data' is
  refuted for the store side. The rival side is §0: its deficit was training steps"* (FINDINGS l.139–141).
- ⭐⭐ **What this changes.** M1 (`N339`) said the rivals had an 8× **data** disadvantage. They did, and **it barely mattered**. The draft's comparator
  was trained for 1,280 steps, untuned. *"Fixing the data match alone would not have fixed the paper"* (`w20-advisor-report.md` §0).
  - The LOG recorded the suspicion **before any number** (17:26Z entry: *"at 16,384 rows x 20 epochs the rivals got 1,280 optimizer steps"*).
  - `learned_ref_ownpool` is **not contradicted**: it is the same rows at 1/8 the steps (LOG 06:57Z).
- ⚠ **Scope.** This is the MLP family only (A1.1 and A2.3 are MLP-only diagnostics that never enter P3). It is Shapes rows 3–4 only.
- ⭐ **It joins the M1 pattern** (`N339` ⟲ PASS 4: data, steps, input). Instance 1, which was filed as **data**, was in fact mostly **steps**, the same
  axis as instance 2 (w16 CMAZE-O). ⛔ The curator does not rule M1.
- **Matrix row:** `CM-W20-2`. ⟲ at `N339`.

### N374 — ⛔⛔ **M3 MEASURED AND REPLICATED: UNDER THE HEAD'S REFIT DEFINITION, DELETING `red ∧ sphere` COSTS THE SURVIVORS −6.13 / −6.43 / −6.30 NOVEL PER CORNER (the pre-declared bar was |Δ| ≤ 1.0), WITH THE 20 RED NON-SPHERE CORNERS AT −32.1 ON SEED 0. A random-1 % control moves −0.02 / −0.11 / −0.01 ⇒ CONTENT, NOT INSTABILITY. The mechanism (a), (b) is exact; the outcome (d) is not. The w15 "composition unchanged" row held only because the book was carried.** · tier A (CURATOR-PROPOSED ◈) · unlearning / claims hygiene (w20, store side: FREE replication under Head AMENDMENT-2; ⭐ **referee M3's objection, measured**)

**Source:** `w20-shapes-data-matched/SHAPES-MATCHED.json` → `deletion_matched.per_scope.{value,conjunction_live,item}.store.<seed>.{a,b,d}`;
`deletion_store/seed0/conjunction_live.json`; `FINDINGS.md` §5 (l.143–168); `AMENDMENT-2.md`; PREREG A4.1 (bars); `LOG.md` 01:22Z.

- **Definition** (Head AMENDMENT-2): forget c = (1) downdate the store, (2) downdate the **address book's statistics and refit the book from them**,
  (3) punch the hole. ⛔ *"c still composes after forgetting c"* is **RETIRED**: not computed, not printed (`deletion_matched.RETIRED`).
- **(d) survivors, after − before, paired, JSON exact:**

  | scope (rows deleted) | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | **red ∧ sphere** (5,511) | **−6.128440366972477** [−8.614678899082568, −3.8071100917431226] | **−6.431192660550459** | **−6.302752293577981** |
  | value red (11,044) | −0.056179775280898875 | 0.3258426966292135 | −0.6067415730337079 [−1.1685393258426966, −0.07865168539325842] |
  | 1 % items (1,311) | −0.01834862385321101 | −0.11009174311926606 [−0.1926605504587156, −0.027522935779816515] | −0.009174311926605505 |

  `downdate_minus_refit_without_pipeline` is **0.0 on every corner of all 9 legs** ⇒ the downdated object **is** the refit-without object.
- **The breakdown** (seed 0 only; FINDINGS l.163–165, LOG 01:22Z): the 20 corners requesting **red with other shapes** drop **−32.1** per corner
  (LOG: *"min -36, max -20"*); the 36 non-red sphere corners −1.47; the 53 others +0.51. ⚠ **This breakdown is not in any JSON on batmac** and the
  corner → requested-content map needed to re-derive it is not staged (`NEEDS-ARTEFACTS.json`). Seeds 1 and 2 have no printed breakdown.
- **(a), (b): exact within their declared bars, not "≤ 1e-8".** Max over the 9 legs (CA): (a) store mode deviation **5.383044481277466e-07** (bar 1e-5);
  (b1) statistic deviation **0** (count and first moment); (b2) rendered-feature relative deviation **7.812795388417726e-08** (bar 1e-3); max absolute
  book-W deviation 8.354888609574118e-05. Every `RESULT` is `true`. ⚠ FINDINGS l.156 (*"at 1e-8 or better"*) and the brief (*"≤1e-8"*) understate the
  largest deviations (`N381` item 1).
- ⚠ **The "control" is a little less clean than "moves nothing".** Item-scope seed 1 is −0.11009174311926606 with a CI that **excludes 0**; value-scope
  seed 2 is −0.6067415730337079 with a CI that excludes 0. Both pass |Δ| ≤ 1.0, and both are an order of magnitude below red ∧ sphere.
- ⭐⭐ **The reading, in w20's words** (FINDINGS l.165–167): *"Under the Head's definition, deleting a conjunction removes the store's ability to compose
  one of its values with other values. The w15 'composition unchanged' row held only because the book was carried."*
- **(c) DISPLAY, never a result:** value red refused 1.000 on 720 deleted-corner requests, false refusal 0.000; red ∧ sphere: *no rows 3–4 corner
  requests red ∧ sphere*, so refusal is **undefined**; item: N/A.
- **Matrix row:** `CM-W20-3`. ⟲ at `N340` and `N355`.

### N375 — ⚠⚠ **THE ALS ADDRESS BOOK HAS NO COMPACT EXACT STATISTIC: its exact statistic is 967304800 bytes (0.97 GB), i.e. the fit data itself, and the book refit costs 2,367–5,913 s. "Adaptation reads 0 pool rows" is TRUE and is NOT a size claim.** · tier A (CURATOR-PROPOSED ◈) · unlearning / claims hygiene (w20, declared before any (b) number: PREREG A4.2)

**Source:** `SHAPES-MATCHED.json` → `deletion_matched.per_scope.*.store.*.b.b1_statistic.{statistic_bytes,statistic_bytes_note}` and
`b.b2_book.book_refit_from_downdated_statistic_seconds`; `FINDINGS.md` l.158–161; PREREG A4.2; `LOG.md` (HEAD AMENDMENT-2 entry).

- `statistic_bytes_note`, verbatim: *"= the fit data itself: every Shapes3D tuple is ONE row, and the ALS book has no smaller exact statistic"*. The
  statistic is the per-row first moment of the book's 117,964 fit rows.
- **Cost:** book refit from the downdated statistic 2,367–5,913 s (FINDINGS; JSON min 2367.3, max 5912.9), *"≈ the refit-without book"*. The store
  downdate itself reads 0 pool rows in seconds.
- ⛔ **FORBIDDEN:** *"data-free deletion"*, *"reads 0 rows"* or *"no data retained"* as a **storage or privacy** claim for the book. The honest sentence is
  that the book is exact at the cost of retaining a statistic the size of its fit data.
- **Matrix row:** `CM-W20-3`.

### N376 — ⚠ **THE MATCHED RIVAL DELETION: VALUE AND ITEM SCOPES CLEAR ZERO EVERYWHERE, BUT `red ∧ sphere` IS NOT SIGNIFICANT ON 3 OF 12 LEGS, AND "THE RIVALS LOSE < 1 WHERE THE STORE LOSES 6" HOLDS FOR THE FAMILY MEANS OF THREE RIVALS, NOT PER SEED.** · tier B (CURATOR-PROPOSED ◈) · unlearning / baselines (w20 scored run 3/3, TEST, 3 seeds)

**Source:** `SHAPES-MATCHED.json` → `deletion_matched.per_scope.<scope>.rivals.<family>.<seed>.{ours_minus_rival_retrain_without, ci95,
rival_after_minus_before, rival_cost_line}`; `FINDINGS.md` §5 (l.170–185); CA `w20_rival_after_minus_before*`.

- **Store after − rival retrain-without, survivor corners:** value **+7.47 … +32.61**, item **+6.95 … +28.84**, all CIs clear of 0 (`w20-advisor-report.md` §2).
  **red ∧ sphere, not significant:** vs mlp s1 **+2.49 [−0.94, 6.02]**, vs mlp s2 **+2.60 [−0.50, 5.67]**, vs lstm s2 **+2.48 [−0.87, 5.82]**.
- ⚠ **"Rivals lose < 1 novel/corner under red ∧ sphere"** (FINDINGS l.183–184; the w20 Hub report; the brief). Per family and seed (CA), the rival's
  after − before is:
  - mlp −0.5963302752293578 / +0.9174311926605505 / **−2.9541284403669725** (family mean −0.8776758409785933);
  - lstm −0.908256880733945 / +1.3761467889908257 / −0.3577981651376147;
  - transformer −0.8348623853211009 / −0.44036697247706424 / −0.7614678899082569;
  - cont_mlp −0.22018348623853212 / +0.7614678899082569 / **−9.568807339449542** (family mean **−3.0091743119266057**).

  ⇒ True for the **family means** of mlp, lstm and transformer. ⛔ **Not true per seed** (mlp s2, cont_mlp s2) and not for cont_mlp's mean (`N381` item 2).
  The qualitative asymmetry stands: the store loses ≈ 6 on every seed, and **no rival family loses that much on its mean.**
- **Cost line** (the store's real advantage): rival retrain-without reads **120,028 / 125,561 / 129,761** rows, rewrites 521,732–527,564 B, in
  **548–2,065 s** (JSON 547.7–2064.5); the store downdate reads **0** pool rows. ⚠ The book refit (`N375`) is not free.
- ⚠ Rival retrains ran at the **DECLARED** schedule, so the mlp (s0, s1) and cont_mlp P1 flags travel with these rows. w15's 16k-rival deletion
  columns move to the **appendix only** (Head AMENDMENT-1 (ii)).
- **Matrix row:** `CM-W20-3`.

### N377 — ⚠ **THE LADDER, DATA-MATCHED (free replication): margins +5.13 (n = 2) and +10.3 … +12.6 (n = 3–6), every per-seed CI above 0. ⛔ The EXISTING ladder K = 5 arm's n = 4 and n = 5 rows are NOT CLAIMABLE: strong-form `random_atoms` reads 1.24 and 2.03 over the 1.0 bar, and two of its four controls were never run.** · tier B (CURATOR-PROPOSED ◈) · benchmark design / control integrity (w20, ladder, 3 seeds, no ledger row)

**Source:** `SHAPES-MATCHED.json` → `ladder.<n>.{P3_margin.store_K32_n131072.vs_strongest, P2_controls, integrity_rebuilt_vs_w12b, P1_convergence}`;
`FINDINGS.md` §6 (l.187–215).

| rung | K\* − strongest data-matched rival, mean | per seed | draft-style (vs spoke-A `@ref`) |
|---|---|---|---|
| n = 2 | **+5.13** | +4.63 / +6.92 / +3.85 | +12.38 / +10.50 / +9.38 |
| n = 3 | **+11.18** | +12.89 / +15.95 / +4.71 | +15.72 / +21.22 / +9.92 |
| n = 4 | **+12.55** | +14.72 / +15.80 / +7.12 | +20.81 / +24.25 / +15.21 |
| n = 5 | **+10.69** | +12.00 / +13.54 / +6.52 | +21.21 / +25.25 / +15.29 |
| n = 6 | **+10.30** | +11.72 / +11.56 / +7.62 | +21.48 / +24.68 / +15.66 |

- Lowest per-seed CI lower bound **+2.46** (n = 3, seed 2); smallest seed-level margin **+3.85**. ⭐ Rebuilt K\* = 32 ladder stores reproduce w12b per
  corner on **15/15** legs (`integrity_rebuilt_vs_w12b`: `n_corners_differ` 0 everywhere).
- ⚠ **Flags printed, not netted:** the lstm ladder spot check fails P1 at n = 2 seed 0 (20.55 → 21.85, +1.30). ⚠ Rungs n = 3–6 probe the **identical
  120 corners**: a training-data axis, not depth (`N362`).
- ⛔ **The existing ladder K = 5 arm** (w12's `DowndatableStore`, reused *"as already fitted"*): `random_atoms` per seed at n = 4 = 0.48333333333333334 /
  0.18333333333333332 / **1.2416666666666667**, at n = 5 = 0.6833333333333333 / 0.5833333333333334 / **2.033333333333333**; `landscape_removed` and
  `landscape_shuffled_across_wells` are `DECLARED_NOT_RUN_or_missing` on every rung. ⇒ **n = 4 and n = 5 fail P2 at seed 2.** Every w20-fitted ladder
  store collapses on all four controls. ⚠ **This touches the w12 / w13 K = 5 ladder rows** already in the registry (`N326`–`N336` family): the curator
  marks it and does not re-rule them (`§I8.19`).
- **Matrix row:** `CM-W20-4`.

### N378 — ⛔ **LACE AND PoE ON THE STORE'S 131,072-ROW POOL SCORE NO BETTER THAN ON 32,256: LACE 0.018 / 0.046 / 0.073, PoE 0.000 ×3 on rows 3–4. Data matching does not rescue them. ⛔⛔ SCOPE, VERBATIM: "LACE/PoE as instantiated in this feature space at their own declared budgets — not a refutation of LACE in general."** · tier B (CURATOR-PROPOSED ◈) · baselines (w20 run 1/3; Head AMENDMENT-1 (i))

**Source:** `SHAPES-MATCHED.json` → `LACE_PoE_at_131072` (`scoping_VERBATIM`, `legs.shipped.<method>.<seed>.{novel_per_corner,budget}`); `FINDINGS.md`
l.96–112 and §7 item 1; `LOG.md` 05:13Z, 07:39Z.

- **Rows 3–4 TEST** (130,560 fit + 512 val; own 36-trial selections): LACE 0.01834862385321101 / 0.045871559633027525 / 0.07339… (FINDINGS: 0.018 /
  0.046 / 0.073; LACE classifier 4,073,663 params), PoE 0.000 / 0.000 / 0.000 (3,556,665 params). w15 at 32,256 rows, printed **beside**: LACE 0.165 /
  0.073 / 0.229, PoE 0.009 / 0.000 / 0.000 (`N360`). Store on the same corners: 47.50 / 47.40 / 49.12.
- ⛔ **DECLARED NOT-RUN: LACE/PoE @131,072 on the ladder rungs** (15 legs). Measured cost: a 36-trial selection of 5,955–11,034 s plus a 50-epoch
  classifier of ~7.5 h on the val leg, ~3.5 GB each. Two rung legs were started and stopped before a number. **The w15 32,256-row rung rows remain the
  only LACE/PoE rung rows.**
- ⚠ **The w20 Hub report mis-states the comparison** (`w20-advisor-report.md` §5 item 1: *"LACE 0.0183 / 0.0459 / 0.0734 … against the store's 64.92 on
  the same corners"*). 64.92 is the **VAL** store (seed 0, 24 val corners, ceiling 72; LOG 05:13Z *"Store K32 on the same corners 64.92"*, which is the
  VAL LACE line). The rows 3–4 TEST store is 47.50 / 47.40 / 49.12 (`N381` item 3).
- **Matrix row:** `CM-W20-4`. ⟲ at `N360`.

### N379 — ⚠ **(process) w20's DEFECTS, FOUND AND FIXED DURING THE RUN, EACH WITH ITS VOID: a generation-1 dev smoke; a VOID val scoring pass (`seen_mask` marked every val tuple seen); ⭐ PER-SEED RUNG DEAD-CHANNEL MASKS read as seed 0's ⇒ seed-1/2 rung stores and scores VOIDED and rebuilt; a ladder threshold-width check; one orphan scorer. And `N350` REALISED at K = 32: the historical T7 across-wells permutation at seed 2 had 2 fixed points.** · tier B (CURATOR-PROPOSED ◈) · process / instrument integrity (w20)

**Source:** `w20-shapes-data-matched/FINDINGS.md` §8 (l.239–250) and l.104–107; `LOG.md` 17:48Z, 18:11Z, 20:48Z, 21:17Z, 00:29Z; directories `rivals/val_SMOKE`,
`scores/val_VOID_seenmask_all_seen`, `VOID_rung_deadmask_seed12_0029`.

1. **Generation-1 dev smoke** (`CHLU_GEN_FEATURES_ROOT` unset), caught by `assert_generation2`; `rivals/val_SMOKE` is dev-only, never scored.
2. **VOID val pass 1:** val corners live in `train_rows`, so every val tuple was "seen" (ceiling 0, novel 0 for every arm). Moved to
   `scores/val_VOID_seenmask_all_seen`; fixed to train rows minus val corner rows.
3. ⭐ **Rung dead-channel masks are per seed.** `load_pool('rung:k')` read seed 0's for every seed. Seed-0 rebuilds reproduced w12b ranks; seeds 1–2
   did not (LOG: *"sum|Δrank| 30-47"*). ⇒ **Seed-1/2 rung stores and their scores VOIDED** (`VOID_rung_deadmask_seed12_0029`) and rebuilt. Rows 3–4
   unaffected (T7 used one partition file for all seeds). Rival fits unaffected (no dead mask).
4. A ladder rebuild threshold-width check; rebuilt rows reproduce w12b per corner (`N377`).
5. An orphan slow-path seed-1 scorer ran beside the fast path; per-corner identical (verified at seed 0); stopped.
- ⭐ **`N350` realised:** *"the historical T7 permutation at seed 2 had 2 fixed points (N350 realised at K = 32); it still read 0.000, as does the
  derangement"* (FINDINGS l.106–107). ⇒ The T7 control was partly the identity at seed 2 **and** the store collapses under the true derangement.
- ⭐ All five are self-caught and reported as voids, not patched over. **This is the `N351` / `N363` discipline, obeyed.**
- **Matrix row:** `CM-W20-4`. ⟲ at `N350`.

### N380 — ⚠⚠ **(omit nothing) THE w20 ROWS THAT WERE WITHHELD OR NEVER RUN, EACH WITH ITS STATUS AND WHY — including `learned_1x`, the programme's highest-capacity rival, NOT PRINTED by w20; its existing w12 value at the shipped leg (48.8991 / 49.5413 / 49.4404) sits ABOVE w20's K\* = 32 store on every seed.** · tier A (CURATOR-PROPOSED ◈) · claims hygiene / baselines (w20 §7; Head omit-nothing rule, 2026-09-24)

**Source:** `w20-shapes-data-matched/FINDINGS.md` §7 (l.218–237); `LOG.md` 09:14Z, 19:04Z, 21:50Z, 02:18Z; `w20-advisor-report.md` §5 item 6 and §7
item 6; `w13-seeds-and-tables/TABLES.md` l.703–722.

| # | withheld / not run | status | why (the spoke's reason) |
|---|---|---|---|
| 1 | ⭐⭐ **`learned_1x`** (536k and ~3.2M params, matched to the store's four-part MEMORY ledger) | **NOT PRINTED**; *"kept in artefacts"* (LOG 09:14Z) | COMMON: *"rivals matched on read-path parameters; the inflated-rival row does not exist"* |
| 2 | LACE / PoE @131,072 on rungs n = 2…6 (15 legs) | DECLARED NOT-RUN (LOG 02:18Z) | measured cost (`N378`) |
| 3 | VAL `lstm` | DECLARED NOT-RUN on val (LOG 21:50Z) | fit stopped for memory; lstm measured on TEST with its own TENX |
| 4 | TENX on the ladder beyond the spot check | not run | PREREG §4: spot check at rungs 2 and 6, seed 0 |
| 5 | 30× for lstm / transformer | not run | only the two families that failed P1 |
| 6 | mlp / cont_mlp retrain-without at TENX or 30× (9 fits, ~1–5 h each) | not run | wall clock; P1 flags travel with the deletion rows |
| 7 | `landscape_removed`, `landscape_shuffled_across_wells` on the existing ladder K = 5 arm | never run | reused *"as already fitted"* (`N377`) |
| 8 | spoke A's literal 32,256-row Montero pool as a store fit pool | not used | it holds both held-out values (3,157 rows at object_hue 0.5, 7,972 at shape 0.0) |
| 9 | the 16k w12 comparators on **VAL** (mlp 5.79, lstm 4.79) | *"informational only"* (LOG 19:04Z) | they were fit on the TEST pool, which contains val-corner rows ⇒ not clean on val |
| 10 | the val margin +14.2 (against +59.1 over the 16k comparator) | printed in LOG only (19:55:47Z entry) | VAL, free; never a table row |

- ⭐⭐ **`learned_1x`, folded with its number.** w20 staged **no** `learned_1x` number on batmac (grep of `w20-shapes-data-matched/` finds it only in PREREG,
  LOG and FINDINGS). The programme's existing `learned_1x` (w12, an MLP matched to the **K = 5 arm's** own total, fitted on 32,768 rows) is in
  `w13-seeds-and-tables/TABLES.md` l.710, the `shipped` leg (= the rows 3–4 condition, 109 corners): **48.8991 / 49.5413 / 49.4404**, against that
  table's K = 5 arm 44.9541 / 46.3211 / 47.7248 (delta vs 1x -3.9450 / -3.2202 / -1.7156).
  - ⚠ **Beside w20's K\* = 32 store (47.50 / 47.40 / 49.12), `learned_1x` is higher on every seed.** This is a **curator side-by-side of two
    instruments**, unpaired, not an estimand, and ⛔ **not a margin**: `learned_1x` is memory-ledger-matched (≈ 4× the read-path pin), not read-path matched.
  - The ~3.2M-param variant (matched to K\* = 32's ledger) has **no number on batmac** (`NEEDS-ARTEFACTS.json`).
  - ⚠ **The balancing fact, from the ladder rungs:** against the same w12 `learned_1x` opponent, K\* clears the 1.0 bar on every seed of every rung,
    worst seed **3.117** (n = 2) to **11.533** (n = 3) (`w12b-nr6-and-kstar/KSTAR-LADDER.md` l.66–72; `N339`). ⇒ K\* beats `learned_1x` on the rungs
    and trails it at the shipped leg's means. No paired K\* − `learned_1x` estimand at the shipped leg exists on batmac.
  - ⭐ **Why this must be printed, not merely noted:** under the byte matching that `CM-W12-4` records, the composition verdict **changes sign**
    (*"Against a learner matched to the arm's OWN total (1x) the margin is NEGATIVE at n = 2, 4, 5 and 6"*, TABLES l.715). A reader shown only the
    read-path column draws the opposite conclusion. The w20 Hub report asks the Advisor exactly this (§7 item 6). ⛔ The curator does not rule it.
- ⛔ **None of these not-runs is a null.**
- **Matrix row:** `CM-W20-1`, `CM-W20-4`.

### N381 — ⚠ **(provenance) EIGHT PLACES WHERE THE BRIEF, w20's PROSE OR THE HUB'S REPORTS DISAGREE WITH w20's PRIMARY RECORD. The artefact wins each one. This is `N351` / `N371`'s rule ("a paraphrase is not a source") applied a third time.** · tier C (CURATOR-PROPOSED ◈) · process / provenance (w20; the w20 and w18 Hub reports; the pass-5 brief)

| # | the claim | where | what the primary record says |
|---|---|---|---|
| 1 | *"(a)/(b) are exact (≤1e-8, statistic dev 0)"*; *"at 1e-8 or better in the rendered features"* | pass-5 brief; w20 FINDINGS l.156; Hub report §2 | (a) max **5.383044481277466e-07**, (b2) rendered max **7.812795388417726e-08** (CA). Exact **within the declared bars** (1e-5, 1e-3), every `RESULT` true; not ≤ 1e-8. (b1) = 0 holds (`N374`) |
| 2 | *"the rivals lose < 1 novel/corner … while the store loses 6"* | FINDINGS l.183–184; Hub report §2; brief | mlp s2 **−2.9541284403669725**, cont_mlp s2 **−9.568807339449542**, cont_mlp mean −3.0091743119266057 (CA). True only for the family means of mlp, lstm, transformer (`N376`) |
| 3 | *"LACE 0.0183 / 0.0459 / 0.0734 … against the store's 64.92 on the same corners"* | `w20-advisor-report.md` §5 item 1 | 64.92 is the VAL store (24 val corners). Rows 3–4 TEST store: 47.50 / 47.40 / 49.12 (`N378`) |
| 4 | `headline.rule`: *"P1-carried strongest data-matched rival"* | `SHAPES-MATCHED.json` | Seed 2 uses max(DECLARED, TENX). The literal P1-carried margin is 5.944954128440367 (mean 7.342507645259938). The printed +7.03 is the **conservative** number (`N372`) |
| 5 | *"Item deletion (a random 1 %) moves nothing"* | FINDINGS l.164–165; Hub report | seed 1 **−0.11009174311926606**, CI [−0.1926605504587156, −0.027522935779816515] excludes 0. Small, inside the bar, but not nothing (`N374`) |
| 6 | the −32.1 / −1.47 / +0.51 breakdown quoted as a replicated fact | brief (*"the 20 red non-sphere corners at −32.1"*) | **seed 0 only** (FINDINGS l.163), and in no JSON on batmac (`N374`) |
| 7 | MF-2 *"CLOSED"*, citing w20's PREREG *"data-matched to 32,768 rows, NOT to 131,072"* | `w18-advisor-report.md` l.47–48 | That sentence is PREREG A2.1, about **`learned_ref_ownpool`**, not about the ladder store. w20's ladder **headline** compares the K\* = 32 store and the data-matched rivals **both** on the rung's own **131,072** rows (FINDINGS l.189). The 32,768-row store MF-2 describes is the EXISTING K = 5 arm, which `N377` finds not claimable at n = 4 / 5. The outcome is harmless for the headline; the quoted reason is not the reason |
| 8 | *"1,482 mirrored"* | `w20-advisor-report.md` header | 1481 files staged on batmac (CA `w20_staged_file_count_on_batmac`). One file is unaccounted for; ⚠ not material to any number here |

- ⛔ **FORBIDDEN in reports:** items 1, 2, 3, 5, 6 as worded. ⛔ The curator edits **no** evidence file. The fixes are flagged for the Hub (`§I8.19`).
- **Matrix row:** `CM-W20-3` (items 1, 2, 5, 6), `CM-W20-4` (item 3), `CM-W20-1` (item 4).

**Reading key (w21).** CMAZE-T, unseen n-way, `exact`, TEST unless marked. **R\*** = the best matched rival by 3-seed **mean** over every kind and both
budgets (8,000 / 80,000 steps). **The Head's label rule** (w21 PREREGs, written before the ban and quoted here as the historical record): WIN iff
Δ̄ > 0, ahead on ≥ 2 of 3 seeds, and the store gate holds; else COMPARABLE iff |Δ̄| ≤ σ; else LOSS.

### N382 — ⛔ **WORDS-FULL ON TEST: LOSS. Δ̄ −0.0175 (σ 0.0140), 0/3 seeds ahead; TEST repeats VAL (−0.0173). ⚠ Under two other defensible spreads — the CLU's per-seed sd (0.137) or R\*'s (0.151) — it would read COMPARABLE. The VAL instrument reproduces w19 bit-for-bit.** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / claims hygiene (w21 spoke `words-full-test`, scored runs 4–6/9 of target `maze-t-symmetric`)

**Source:** `w21-words-full-test/test/BARS-WORDS-FULL-test.json` → `HEAD_LABEL.rows.nway` (incl. `printed_never_the_label`); `FINDINGS.md` §0–§6;
`PREREG-w21-words-full-test.md`; `repro/REPRO_COMPARE.txt`.

- **The row:** CLU 0.724626907621842 (0.8002628120893561 / 0.5660377358490566 / 0.8075801749271136); R\* `rival: transformer` (8k)
  0.7421448824348764 (0.8252299605781866 / 0.5673854447439353 / 0.8338192419825073). Paired Δ −0.024967148488830526 / −0.0013477088948786742 /
  −0.026239067055393694, **Δ̄ −0.017517974813034298**, **σ 0.014018294086938362**. `LABEL: LOSS`, *"behind by 0.0175 > one seed's spread 0.0140"*.
- ⭐ **Previously "printed, never the label" — now folded with its status** (`N390`): σ from the CLU's own spread **0.1373909749670931** or from R\*'s
  **0.15140703330856564** would give |Δ̄| ≤ σ ⇒ **COMPARABLE**. The PREREG fixed the paired-difference σ before any TEST number; FINDINGS l.41–43 says
  that spread *"is almost entirely seed 1's shared data ceiling"*. ⛔ Both readings are printed; the curator does not pick.
- **Mechanism:** *"the shared pair block carries the composition, not the store"* (FINDINGS §0). Store gate holds (`landscape_removed` 0.000; max control
  0.2435173246859834 ≤ 0.575; derangement [1, 0]). ⭐ This **confirms on TEST** what w19 found on VAL (`N367`).
- ⭐ **Instrument check, done first:** CLU + 4 controls × 4 rows × 3 seeds on WORDS-FULL VAL reproduce w19 with **max |Δ| = 0**
  (0.8018617021276596 / 0.5688888888888889 / 0.8033946251768034). The TEST CLU and controls also equal w19's BOTH TEST bit-for-bit.
- **Other rows (TEST):** `seen` −0.032, `pair` −0.038 (learned blind 0.557 above the CLU: falsifier fires), `value` −0.287: all LOSS. Rollout validity
  0.000 on every seed for the CLU (max matched rival 0.111; 10×-float diagnostic 0.289).
- **Deletion:** store-scoped exactness passes on every seed and mode (relative ΔS 7.7e-16 … 1.1e-15; bar 1e-5); downdate == fitted-without on 100 % of
  surviving n-way rows. ⭐ PRE-deletion composition printed beside post-deletion (Advisor ruling (d)): no change except **−0.005 on seed 1, value mode**.
- **Budget:** runs 4/9, 5/9, 6/9 **SPENT → LOSS**; ledger rows appended 2026-09-23T23:34:52Z, before launch.
- **Matrix row:** `CM-W21-1`. ⟲ at `N367`.

### N383 — ⚠⚠ **CLU-REL-A10 WINS BOTH TEST (0.929 against 0.685, 3/3 seeds) — ⛔ BUT ABOUT 0.18 OF THE 0.24 MARGIN IS THE CODING: rivals given the same relational columns reach 0.862; the CLU keeps +0.066 and is ahead on only 2 of 3 seeds; ⛔⛔ AND ON WORDS-FULL VAL THE STORE'S CONTRIBUTION IS ZERO (rival + rel 0.941 against CLU 0.937). 68 candidates were tried and 57 passed the gate (its own PREREG and ledger say 69).** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / claims hygiene (w21 hub `maze-t-clu-improve`, scored runs 7–9/9, TEST, 3 seeds; VAL free)

**Source:** `w21-maze-t-clu-improve/test/LABEL-BOTH-test.json`, `val_full/LABEL-BOTH-val.json`, `val_full/LABEL-WORDS-FULL-val.json` (`rows.nway`,
`matched_parameters`, `rel_diagnostic_parameters`); `FINDINGS.md` §0–§7; `LOG.md`; `val/sweep_r{1..5}.json` (curator count); `posthoc/controls_val.json` via FINDINGS §3.

- **What changed: only the CLU.** `CLU-REL-A10` (sha256 `0d4ec5b9…`): K 2, 10 atoms, ridge 1.0; design 396 wide = `[bias | main 27 | 0.5 × pair 290 |
  eq 6 | eq × door 42 | eq × dir 30]`; **9,200** read floats (A1: 4,584). Rivals re-matched to 9,200 by the standing rule: on the 954-dim BOTH input,
  mlp width 9 (9,698), lstm width 8 (9,165), transformer width 8 (9,269).
- **BOTH TEST** (`LABEL-BOTH-test.json`): CLU **0.9285039268690677** (0.9080157687253614 / 0.8935309973045822 / 0.9839650145772595); R\* `rival@10x-steps:
  transformer` **0.6851358576873765**; margin 0.24336806918169118; per-seed margins 0.10512483574244413 / 0.3261455525606469 / 0.2988338192419825;
  3/3 ahead; store gate on means holds (max control 0.6006358339900985 ≤ 0.7785039268690677). ⇒ **WIN** by the Head's rule.
- ⛔⛔ **The printed diagnostic that bounds it** (`diag_rel_*`, "never a bar"): the same relational columns appended to every rival's frozen input.
  - BOTH TEST: best `+rel` = transformer 80k, mean **0.8621086355989208**, per seed 0.9080157687253614 / 0.8692722371967655 / 0.8090379008746356. ⇒
    About 0.18 of the 0.24 is the coding; the CLU keeps **+0.066** on the mean. Against the best `+rel` arm per seed it is **behind on seed 0** (mlp + rel
    80k 0.915 against 0.908) ⇒ ahead on 2 of 3.
  - ⛔⛔ **WORDS-FULL VAL:** `+rel` transformer 8k **0.9414055233621977** (0.9228723404255319 / 0.914074074074074 / 0.9872701555869873) against CLU
    **0.9371139657869874**. ⇒ **the store's contribution is zero there**, *"exactly as w19 found for the pair block"* (FINDINGS §5). ⚠ That `+rel`
    transformer is itself **under**-matched (residual −7.206521739130435 %), so the tie is not flattered.
- ⭐ **The mechanism, named by the spoke:** *"colour-relational request coding (an input representation), carried through the CLU's closed-form head. It
  is not a store effect"* (FINDINGS §0). RULE-B tuples go 0 → 1.000; the (c, c, c) interaction family stays **0.000 on every seed**.
- ⚠ **The store gate is weaker than A1's, printed by the spoke:** seed 0 across-wells 0.778 > that seed's CLU − 0.15 (0.758); the gate is on means.
  Post-hoc VAL (after freezing), max control / CLU climbs with atoms: A1 0.250 / 0.725 → 4 atoms 0.260 / 0.878 → 6 atoms 0.296 / 0.914 → 8 atoms
  0.361 / 0.914 → **10 atoms (frozen) 0.615 / 0.937**. The atoms-6 config *"would have had a much cleaner store gate for −0.024 of VAL n-way"*; it was
  measured on the VAL cache only (`N390`).
- ⚠ **The search, counted.** `val/sweep_r1..r5` hold 9 + 7 + 36 + 12 + 5 = 69 entries; one is the A1 reproduction ⇒ **68 candidates**, **57** with
  `gate_pass` (0 + 4 + 36 + 12 + 5). ⚠ The 57 include r4's `K 1`, which FINDINGS calls **inadmissible** (one well cannot be deranged). The PREREG (l.11)
  and the append-only ledger say *"69"*; FINDINGS l.248 corrects this and states the hashed PREREG and ledger were **not edited** (correct practice).
- **Other rows (TEST):** `pair` WIN (0.879 against 0.493; w19's blind falsifier no longer fires); ⛔ `value` **LOSS** (0.500 against 0.792: the
  never-trained colour is unreachable by colour-relational coding of seen colours); `seen` COMPARABLE. Rollout validity 0.067 / 0.000 / 0.289.
- **Budget:** runs 7, 8, 9 of 9 **SPENT → WIN** (ledger 2026-09-24T05:20:26Z / 05:29:07Z / 05:37:46Z, before launch). ⇒ Target `maze-t-symmetric` is
  now 9 of 9: 1–3 BOTH A1, 4–6 WORDS-FULL A1, 7–9 BOTH CLU-REL-A10.
- **Matrix row:** `CM-W21-2`.

### N384 — ⛔⛔ **THE A1 BOTH-TEST ROW AT SIX SEEDS: ON SEED 5 — THE EASIEST SEED (ceiling 0.939) — THE CLU LOSES TO A MATCHED RIVAL BY 0.185, traced to 0.000 on two fully train-supported class-A tuples that eight other arms, ONE OF THEM A CONTROL, score at 1.000. Seed 3's +0.262 rests on a non-converged matched MLP. Rollout validity 0.000 on all 6 seeds.** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / benchmark design (w21 spoke `maze-t-seed4`, scored runs 1–3/3 of target `maze-t-seed4`, seeds 3 / 4 / 5; no PREREG, no label)

**Source:** `w21-maze-t-seed4/MAZE-T-SEED4.json` → `classification`, `nway.BOTH-TEST.<seed>`; `FINDINGS.md` §0–§2, §8; `TABLES.md` l.1–30, l.1148,
l.1151; `AMENDMENT-1.md` (the Head's).

| | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| CLU (A1) | 0.800 | 0.566 | 0.808 | 0.674 | 0.533 | 0.682 |
| best matched rival (per seed) | 0.627 lstm-80k | 0.567 lstm-8k | 0.784 lstm-80k | 0.412 transf-8k ⚠ | 0.531 mlp-80k | **0.867 mlp-80k** |
| CLU − best (per seed) | +0.173 | −0.001 | +0.023 | +0.262 | +0.001 | **−0.185** |
| ceiling (best trained arm) | 0.830 | 0.567 | 0.808 | 0.714 | 0.535 | **0.939** |

- **Seed 5, exact** (`MAZE-T-SEED4.json`): CLU 0.6818181818181818 against `rival@10x-steps: mlp` 0.8666666666666667 (CA per-seed diff
  −0.18484848484848493). Per tuple (TABLES l.1148, l.1151): `(2,0,1,2)` n = 90 and `(2,0,3,2)` n = 80, class A (*"REACHABLE-FROM-TRIPLE"*), CLU
  **0.000**, *"8 arms incl. control: landscape_shuffled_across_wells"* at 1.000. ⇒ ⛔ **A charter-C control solves tuples the CLU misses.** FINDINGS:
  *"a CLU-specific miss on reachable data, not a hard-seed effect."*
- **Seed 3:** the matched MLP did not converge at either budget (`N387`); the 10×-floats transformer reaches 0.714 > CLU 0.674.
- **Seed 4:** CLU 0.533, best matched rival 0.531, ceiling 0.535 — within 0.004.
- **Easy / hard by measured ceiling** (`classification`): sorted ceilings 0.535 … 0.939; largest gap **0.14668862933013882** ⇒ cut
  **0.6407297594090047**; easy {0, 2, 3, 5}, hard {1, 4}. ⚠ **Sensitivity, printed:** with the matched-arms-only ceiling the same rule gives hard
  = {1, 3, 4}; seed 3 is the borderline seed. CLU means: easy 0.741, hard 0.549, original three 0.725. ⭐ **6 seeds tried, 6 kept, 0 discarded.**
- ⭐ **The cancelled screen, disclosed:** the brief's original rule would have kept a seed only if its CLU TEST n-way lay in [0.734, 0.874] — a window that
  **excludes seed 1** — and would have **rejected all three new seeds** (0.674, 0.533, 0.682). The Head's AMENDMENT-1 cancelled it before any number.
- **Other rows:** per-step 0.900–0.930; **rollout validity 0.000 on all 6 seeds** for the CLU (matched rivals ≤ 0.267); deletion store-scoped exact
  everywhere (max relative ΔS ≤ 1.94e-15), adaptation 0.09–0.88 s, rival retrain-without 71–140 s.
- **Comparable row, OBS VAL, 6 seeds:** CLU behind the best matched rival on **all 6** (−0.223 / −0.082 / −0.016 / −0.179 / −0.293 / −0.282);
  extractor whole-tuple accuracy 0.065 / 0.224 / 0.191 / 0.140 / 0.121 / 0.212.
- **Matrix row:** `CM-W21-3`.

### N385 — ⚠⚠ **(estimand) "THE 6-SEED MEAN MARGIN IS +0.046 (WAS +0.099 ON 3)" COMPARES TWO DIFFERENT ESTIMANDS. Per-seed-best: +0.065 on 3 seeds → +0.046 on 6. R\*-by-mean (the estimand the w19 relabel uses): +0.0988 on 3 → +0.128 on 6, with the CLU ahead of that R\* on 6 of 6 seeds. Both are printed; ⛔ no label is computed.** · tier A (CURATOR-PROPOSED ◈) · claims hygiene / statistics (w21 `maze-t-seed4`; the pass-5 brief; curator arithmetic)

**Source:** CA `seed4_estimands` (from `w21-maze-t-seed4/MAZE-T-SEED4.json` → `nway.BOTH-TEST.<seed>.all_arms`, `rival*` keys); `w21-maze-t-seed4/FINDINGS.md`
§1 (the +0.046 row); `w19-advisor-report.md` §0a; `w19-maze-t-symmetric/test/BARS-BOTH-test.json`.

| estimand | 3 seeds (0–2) | 6 seeds (0–5) | seeds ahead (6) |
|---|---|---|---|
| mean of per-seed (CLU − best matched rival at that seed) | 0.06514396174683464 | **0.04569812474644914** | 3 of 6 strictly (seed 5 −0.18484848484848493) |
| CLU mean − R\* by mean (`rival@10x-steps: lstm` on 3; `rival@10x-steps: transformer` on 6) | **0.0988366841188022** | **0.12802168076341014** (R\* 0.5490243919986637, CLU 0.6770460727620738) | **6 of 6** (seed 5 +0.006060606060605989); σ 0.11063018934833847 |

- ⛔ **The brief's "(was +0.099 on 3)" sets the R\*-by-mean number beside the per-seed-best one.** On like-for-like estimands, the 6-seed margin is
  +0.046 against +0.065 (per-seed-best) or +0.128 against +0.0988 (R\*-by-mean).
- ⭐ **What survives either way:** seed 5 is a real CLU miss on reachable data (`N384`), and the per-seed picture is heterogeneous (−0.185 … +0.262).
  ⚠ **What does not survive as worded:** *"the win row does NOT replicate"* is a statement about the **per-seed-best** estimand. Under the R\*-by-mean
  estimand that turned w19 BOTH TEST into a WIN (`N389`), the CLU is ahead of R\* on every one of the six seeds, by as little as 0.00606 (seed 5) and
  0.0094 (seed 1; CA 0.009433962264150941).
- ⛔ **The curator computes no label** (the seed-4 spoke printed none, by the Head's AMENDMENT-1; the ban forbids a curator bar). Which estimand the paper
  uses is the Head's (`§I8.19`).
- **Matrix row:** `CM-W21-3`.

### N386 — ⛔⛔ **CLU-REL-A10's STORE GATE FAILS ON SEED 4 VAL: the `random_atoms` control TIES THE CLU AT 0.8994252873563219, AND `landscape_removed` READS 0.2413793103448276, NOT 0. ⇒ The store is NOT load-bearing there. On seed 5 the matched 80k transformer ties the CLU (0.905).** · tier A (CURATOR-PROPOSED ◈) · memory-architecture / control integrity (w21 `maze-t-seed4`, FREE VAL leg, CLU = the peer's frozen CLU-REL-A10)

**Source:** `w21-maze-t-seed4/MAZE-T-SEED4.json` → `nway.CLU-REL-A10-BOTH-VAL.<seed>.all_arms`; `FINDINGS.md` §4; `val_cluimprove/BOTH_val_seed{3,4,5}.json`.

| seed | CLU-REL-A10 | `landscape_removed` | `landscape_shuffled` | across-wells | `random_atoms` |
|---|---|---|---|---|---|
| 3 | 0.9180555555555555 | 0.0 | 0.45555555555555555 | 0.6791666666666667 | 0.6888888888888889 |
| **4** | **0.8994252873563219** | **0.2413793103448276** | 0.21551724137931033 | 0.4540229885057471 | **0.8994252873563219** |
| 5 | 0.905160390516039 | 0.0 | 0.1101813110181311 | 0.8019525801952581 | 0.6708507670850767 |
| *0 (peer)* | *0.9029255319148937* | *0.0* | *0.44680851063829785* | *0.7819148936170213* | *0.37101063829787234* |

- ⛔⛔ **Seed 4:** a control built from random atoms reads **exactly** the CLU's score, and removing the landscape leaves 0.2413793103448276. On this seed the
  relational CLU's read does not depend on the learned store. ⚠ The seed-4 FINDINGS reports the tie and **not** the non-zero `landscape_removed`; it is
  in the JSON (`N390`).
- ⚠ **Seed 5:** matched transformer-80k 0.905 = CLU. **Seed 0:** across-wells 0.7819148936170213 (the peer's LOG already flags controls climbing with
  atom count; `N383`).
- ⭐ **What also holds:** on the two seeds hard for every A1-family arm, CLU-REL-A10 reads 0.914 (seed 1) and 0.899 (seed 4) on VAL; the zero-fit bound
  predicts it exactly where rows are unreachable even by relation (seed 1: 0.9141 = 1 − 58/675; seed 5: 0.9052 = 1 − 68/717) (`N388`).
- ⚠ **Leg caveat:** VAL, a different CLU config; not comparable cell-for-cell with the A1 TEST row.
- **Matrix row:** `CM-W21-2`, `CM-W21-3`.

### N387 — ⛔⛔ **THE MATCHED-FLOAT RULE IS NOW A MEASURED DEFECT: on the 954-dim BOTH input it gives every A1 rival 4 hidden units, and on seed 3 the matched MLP did NOT converge at either budget — final loss 0.17990881204605103 at 8,000 steps and 0.20545026659965515 at 80,000, WORSE with 10× the training. The training record shows further non-converged matched LSTMs the spoke did not flag.** · tier A (CURATOR-PROPOSED ◈) · benchmark design / baselines (w21 `maze-t-seed4`; ⭐ **`N369`'s caution, now measured**)

**Source:** `w21-maze-t-seed4/test/BOTH_test_seed3.json` → `matched_parameters`, `rival_training`; `TABLES.md` l.194–207 (rival training record; rows at l.198–206);
`FINDINGS.md` §0 (*"Printed defect"*).

- **The rule's output** (`matched_parameters`, seed 3): mlp width 4 (4693 params), lstm width 4 (4817), transformer width 4 (4869), all on
  `input_dim` 954, against the CLU's 4584 read-path floats (`clu_ledger.read_floats`).
- **Seed 3 MLP:** loss 0.17990881204605103 (8k) → 0.20545026659965515 (80k); n-way 0.003 / 0.107; seen 0.483 / 0.846. *"an optimisation/capacity
  pathology of the 4-unit matched rival, not under-training by steps"* (FINDINGS §0).
- ⚠ **From the same training table, not flagged by the spoke** (TABLES l.201, l.204): matched lstm-8k final loss 2.05e-01 (seed 0) and 3.25e-01 (seed 4);
  lstm-80k 1.37e-01 (seed 4; flagged). The transformer and the 10×-float diagnostics reach ≤ 2.76e-04 everywhere.
- ⭐⭐ **Why this is a defect of the rule, not of one fit.** w19 printed the caution (*"954 input dims leave the rivals a 4-unit hidden layer"*, `N369`).
  w21 now measures it: at 4 hidden units, 10× more steps made the MLP **worse**. ⇒ Matching **read-path floats** on a wide input can produce a rival
  that cannot fit its own training set, and a margin over it is not a margin over a trained network. It is the **fourth axis** of M1 (`N339`): data,
  steps, input, and now **width forced by the float match**.
- The w21 CLU-REL-A10 re-match (9,200 floats) gives widths 8–9 on the same input (`N383`), so it is less exposed, not immune.
- **Matrix row:** `CM-W21-3`, `CM-W17-8`.

### N388 — ⚠ **THE HARD SEEDS ARE HARD FOR A COLOUR-IDENTITY REPRESENTATION, NOT HARD DATA: CLU-REL-A10 reads 0.914 (seed 1) and 0.899 (seed 4) on VAL where every A1-family arm caps near 0.57 / 0.54. ⚠ The Head's AMENDMENT-1 had named the opposite reading as "the finding to support" before any number; the spoke measured it and reports it refuted.** · tier B (CURATOR-PROPOSED ◈) · benchmark design / process (w21 `maze-t-seed4` §2–§4, §8; zero-fit diagnosis)

**Source:** `w21-maze-t-seed4/FINDINGS.md` §2–§4, §8; `diagnosis/SEED-DIAGNOSIS-v2.json` via FINDINGS §3; `AMENDMENT-1.md`.

- **Zero-fit diagnosis (TEST):** unidentified n-way rows 403 / 742 (seed 1) and 305 / 723 (seed 4); unreachable **even by colour relation**: 79 rows on
  seed 1 (4 interaction tuples), **0** on seed 4. Every class-D tuple has a colour pair that is one of the seed's eight held-out 2-way pairs.
- **AMENDMENT-1**, verbatim: *"The finding to support, with numbers: on a hard seed every model struggles, ours included — which shows that seed's
  held-out data is inherently hard, rather than showing a CLU failure."* The spoke's §8: *"A trained arm reaching a high score on seed 1 or seed 4 from
  the same words. It happened on both hard seeds."*
- ⭐ **The curator records this as the ban working as intended:** a declared conclusion was measured and **reported against itself**. Both statements
  are printed by the spoke; *"The Head decides which is quotable."* ⛔ Under the 2026-09-24 rule, no amendment may again name a finding to support.
- **Matrix row:** `CM-W21-3`.

### N389 — ⭐⭐ **(relabel) UNDER THE HEAD'S LABEL RULE, w19's BOTH TEST BECOMES A WIN AND OBS BECOMES COMPARABLE — recorded in the Hub's w19 report §0a, which SUPERSEDES the w19 PREREG verdicts that `N366` / `N369` folded. The curator re-derived every σ from the BARS JSONs and they match. ⚠ The per-seed-best caveat travels with the WIN.** · tier A (CURATOR-PROPOSED ◈) · claims hygiene (w19 `maze-t-symmetric`, relabelled 2026-09-24; the w19 Hub report)

**Source:** `w19-advisor-report.md` §0a (l.12–30); `w19-maze-t-symmetric/val/BARS-*-val.json`, `test/BARS-BOTH-test.json` → `rows.nway.{per_seed, R_star}`;
CA `w19_relabel_check`.

| condition | Δ̄ (Hub) | σ (Hub) | σ (CA, exact) | ahead | Head's label | w19 PREREG verdict |
|---|---|---|---|---|---|---|
| WORDS-MAIN (VAL) | −0.4130 | 0.1537 | 0.15365208407334113 | 0/3 | LOSS | LOSS |
| WORDS-FULL (VAL) | −0.0173 | 0.0157 | 0.015670339259190584 | 0/3 | LOSS | LOSS |
| **OBS (VAL)** | −0.1053 | 0.1068 | 0.10677290859896575 | 0/3 | ⚠ **COMPARABLE** | LOSS |
| *OBS-X10 (diagnostic)* | −0.0137 | 0.1644 | 0.16438894585143135 | 2/3 | *COMPARABLE* | *would-be LOSS* |
| **BOTH (TEST)** | +0.0988 | 0.0751 | 0.07507017244056019 | 3/3 | ⭐⭐ **WIN** | LOSS (SB1) |

- ⚠⚠ **The caveat, in the Hub's words:** *"R\* is the arm with the best mean (lstm, 80k). ⛔ Against the per-seed best arm, seed 1 flips — the 8k lstm
  reads 0.5674 against the CLU's 0.5660 … Print both readings; do not let 'ahead 3/3' stand unqualified."*
- ⚠ **OBS becomes COMPARABLE because its spread (0.1068) exceeds its gap (0.1053)**, driven by seed 0 (CLU 0.2407 against 0.4641). The failing link
  is still READ (`N368`); a large spread is not evidence of parity. ⛔ The curator does not rule it.
- ⭐ **The same relabel note carries the correction of `N371` item 2** (*"WORDS-MAIN fails the control bar"*, `w19-advisor-report.md` l.8). ⇒ That
  pass-4 flag is **closed in the Hub report**; w19's own FINDINGS l.36–38 is still uncorrected.
- ⚠ **What the six-seed extension adds:** under this R\*-by-mean rule the CLU is ahead on all six seeds (`N385`); under the per-seed-best reading seed 5
  loses by 0.185 (`N384`). The WIN carries both.
- **Matrix row:** `CM-W19-5`. ⟲ at `N366`, `N368`, `N369`.

### N390 — ⚠ **(process, omit nothing) THE PASS-5 SWEEP FOR WITHHELD ROWS: every marker the brief lists, where it occurs in the w20/w21 record, and what is now folded. ⛔ Three PREREG files exist in this evidence (w20, w21 `words-full-test`, w21 `maze-t-clu-improve`), all written BEFORE the 2026-09-24 ban; their bars are quoted as record, never as authority.** · tier C (CURATOR-PROPOSED ◈) · process / claims hygiene (all w20 and w21 spokes; the Head's ban)

**Source:** `grep -n -i -E 'DECLARED NOT.?RUN|VOID(ED)?|UNMEASURED|never a bar|not a bar|not reported|informational only|out of brief|withdrawn|never the label'`
over the w20 / w21 FINDINGS, LOG, TABLES, PREREG and AMENDMENT files and the w19 / w20 Hub reports (curator sweep, this pass).
Numbers repeated in the table are cited at their entries: `w20-shapes-data-matched/LOG.md`, `w21-maze-t-seed4/MAZE-T-SEED4.json`,
`w21-maze-t-seed4/TABLES.md`, `w13-seeds-and-tables/TABLES.md`.

| marker | where | the withheld item | folded at |
|---|---|---|---|
| `DECLARED NOT-RUN` | w20 LOG 21:50Z, 02:18Z; w20 §7 | VAL lstm; LACE/PoE rungs; 6 more | `N380` |
| `VOID` / `VOIDED` | w20 FINDINGS §8, LOG 18:11Z, 00:29Z | val pass 1; seed-1/2 rung stores | `N379` |
| `informational only` | w20 LOG 19:04Z | 16k comparators on VAL (mlp 5.79, lstm 4.79) | `N380` item 9 |
| *not printed* (no marker word) | w20 §7 item 6, LOG 09:14Z | **`learned_1x`** | `N380` |
| `printed, never the label` | w21 words-full FINDINGS l.40, TABLES l.13 | σ alternatives ⇒ COMPARABLE | `N382` |
| `never a bar` | w21 words-full TABLES; clu-improve TABLES, FINDINGS | 10×-read-floats diagnostics; `+rel` diagnostics | `N382`, `N383` |
| `DECLARED NOT-RUN` | w21 clu-improve FINDINGS §6 | dynamics interactions, two-stage fit, store / read-verb changes; the atoms-6 cleaner-gate config (VAL cache only) | `N383` |
| `DECLARED NOT-RUN` | w21 seed4 FINDINGS §7 | brief run 2/3 (words-full, same seed); OBS TEST; BOTH VAL A1 for seeds 3–5 | `N384` |
| (in JSON, not in prose) | w21 seed4 `MAZE-T-SEED4.json` | seed-4 CLU-REL-A10 `landscape_removed` 0.2413793103448276 | `N386` |
| (in a table, not flagged) | w21 seed4 TABLES l.201 | lstm-8k losses 2.05e-01 / 3.25e-01 | `N387` |
| `not reported` | w21 words-full LOG / FINDINGS §7; w19 Hub report l.119 | smoke runs (plumbing checks) | ⚠ **left unreported, with reason:** 60-step / 150-row plumbing checks, not results |
| `UNMEASURED`, `out of brief`, `withdrawn`, `not a bar` | — | no occurrence in the w20 / w21 record | — |

- ⚠ **PREREG bars quoted in this block as record only:** w20 P1–P4 and A4.1's (d) |Δ| ≤ 1.0; w19 SB1–SB4 (superseded by `HEAD_LABEL`, as the
  w21 JSON itself states); w21 `words-full-test`'s σ choice. Where a bar and a measured number disagree, the **number** is reported and the bar is named
  as history (`N374`, `N382`).
- ⛔ The curator writes no PREREG, no bar and no label. ⛔ `NEEDS-ARTEFACTS.json` lists what batmac lacks, rather than substituting.
- **Matrix row:** `CM-W17-8` (process note), `§I8.19`.

## ◈.17 ⭐ THE FOLD-IN LEDGER, PASS 5 — every brief item, and where it now lives

| brief item | status | where |
|---|---|---|
| w20 data-matched margin +7.03 (+7.49/+8.60/+5.02), +6.54 best-of-all; draft +39.7…+49.1 ⇒ ~6× shrink, no collapse | ⭐ **FOLDED**, with the headline-label defect (conservative) | `N372`; `CM-W20-1` |
| ⭐⭐ M1's cause misdiagnosed: SCHEDULE not DATA (38.6 vs 5.5; ≈ 0.5 for 8× data; store 47.28 → 48.01) | ⭐ **FOLDED as an entry in its own right** | `N373`; `CM-W20-2`; ⟲ at `N339` |
| P1 convergence; mlp s0 NOT CONVERGED at 30×; grid-edge selections | ⭐ FOLDED | `N372` |
| ⭐⭐ M3 measured and replicated (−6.13/−6.43/−6.30; −32.1; random-1 % −0.02/−0.11/−0.01; (a)/(b) exact) | ⭐ **FOLDED**, ⚠ with "≤1e-8" corrected and −32.1 scoped to seed 0 | `N374`; `CM-W20-3`; ⟲ at `N340`, `N355` |
| ALS book: no compact exact statistic (0.97 GB); 2,367–5,913 s | ⭐ FOLDED | `N375` |
| red ∧ sphere not significant on 3 of 12 legs; "rivals lose < 1" | ⭐ FOLDED, ⚠ "< 1" scoped to family means | `N376` |
| ladder +5.13, +10.3…+12.6; 15/15; K = 5 n = 4/5 NOT CLAIMABLE | ⭐ FOLDED | `N377` |
| defects found and fixed (gen-1 smoke; VOID val; rung dead-mask VOIDED; threshold-width; orphan scorer) | ⭐ FOLDED | `N379`; ⟲ at `N350` |
| all 8 declared not-runs, incl. `learned_1x` NOT PRINTED | ⭐ **FOLDED with numbers** (w12 `learned_1x` 48.8991 / 49.5413 / 49.4404) | `N380` |
| LACE/PoE @131k | ⭐ FOLDED, ⚠ Hub-report comparison corrected | `N378`; ⟲ at `N360` |
| w21 WORDS-FULL TEST = LOSS; σ-sensitivity; VAL repro bit-identical | ⭐ FOLDED | `N382`; `CM-W21-1`; ⟲ at `N367` |
| w21 clu-improve WIN; ~0.18 of 0.24 is coding; +0.066, 2/3; zero on WORDS-FULL VAL; 68 / 57 / "69" | ⭐ FOLDED | `N383`; `CM-W21-2` |
| w21 seed4: win row does not replicate; seed 5 −0.185; class-A 0.000; cut 0.641; seed 3 inflated; rollout 0.000 | ⭐ FOLDED, ⚠ with the estimand disagreement | `N384`, `N385`; `CM-W21-3` |
| ⭐⭐ CLU-REL-A10 store-gate failure, seed 4 VAL (0.8994) | ⭐ FOLDED, plus `landscape_removed` 0.2413793103448276 | `N386` |
| ⭐ the matched-float rule as a measured defect (4 units; 0.180 → 0.205) | ⭐ FOLDED | `N387`; `CM-W17-8` |
| *(curator-added)* the w19 relabel under the Head's rule (BOTH TEST WIN, OBS COMPARABLE) | ⭐ FOLDED; supersedes the w19 verdict column | `N389`; `CM-W19-5`; ⟲ at `N366`, `N368`, `N369` |
| *(curator-added)* the hard-seed reading vs AMENDMENT-1's declared finding | ⭐ FOLDED | `N388` |
| *(curator-added)* disagreements between prose and artefact (w20) | ⭐ FOLDED | `N381` |
| ⛔⛔ the PREREG ban: sweep for withheld rows | ⭐ FOLDED | `N390` |

**⛔ WHAT DID NOT RUN AT THIS PASS**

| not run | reason |
|---|---|
| ⛔ any scored run, compute beyond reading JSONs, GPU | ZERO. The CA script only reads evidence and takes means / sds |
| ⛔ re-deriving the seed-0 −32.1 breakdown, or seeds 1–2 breakdowns | the corner → requested-content map is not staged (`NEEDS-ARTEFACTS.json`) |
| ⛔ printing the ~3.2M-param `learned_1x` | no number on batmac (`NEEDS-ARTEFACTS.json`) |
| ⛔ verifying persisted w20 / w21 objects | only manifests / JSONs are staged; objects are on the ThinkPad |
| ⛔ computing any label for the 6-seed BOTH row, or for the w20 margin | the ban; the Head's |
| ⛔ ruling M1, M3, the K = 5 ladder rows, the estimand, `learned_1x`'s placement | the Head's and the Advisor's |
| ⛔ editing any draft, `.tex`, ledger, PREREG or evidence file | standing rules; the `.tex` lock was not taken |
| ⛔ writing `~/clu/evidence/` | read-only |

⭐ **No sixth pass is required for w20 or for the four w21 units named in the brief.** Anything that lands after `2026-09-24T15:15:18Z` is a new pass.
