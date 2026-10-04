# w10 spoke/referee — LOG

## 2026-09-21T09:07:20Z — start
- Wrote START_UTC.txt.
- Read `/c/chlu/tasks/w10-spoke-referee.md`.
- GATE CHECK: `/c/chlu/outputs/w10-draft-v0/DRAFT-v0.md` exists (74813 bytes, mtime Sep 21 10:53 local). Gate PASSED.
- Evidence base confirmed at `/c/chlu/evidence/outputs/` (live). `$CHLU_CLAUDE_ROOT` NOT read and will not be read.

## 09:10Z — read DRAFT-v0.md in full (753 lines)
- Structure: §0 gate, §1 framing, §2 main-text skeleton (§§1–11), §3 appendices A–N,
  §4 claim register K1–K26, §5 table/figure slotting, §6 deleted claims, §7 never-quote list,
  §8 gaps F1–F8, §9 corrections C1–C8, §10 falsifier check, §11 what-not-done.
- Now beginning numeric cross-check against `/c/chlu/outputs/**` and `/c/chlu/evidence/**`.

## 09:25Z — cross-check pass 1: w10-seeds-and-tables + w5-compose-field
- TABLES.md T1/T2/T3/T4 read in full; SOURCE REGISTER (27 ids) read. Confirmed: **no w5, w6 or
  physics source is in the register** — draft's F5 gap is real and larger than stated (it also covers
  Table 5/the CRR).
- Verified T1 BAR1 deltas by arithmetic: row2 -0.0560, row3-conj -0.0209, row4 -0.0434 ✅ match draft.
- Verified T2 A'/A''/A and CI95 lows ✅ match draft.
- **T1 §1c has NO BLIND-SEQUENTIAL row for row 4 (budget interpolation).** Confirmed at source:
  `w9-maze/runs/scored1_seed0_test.json` `/env_columns` has only `unseen_pair`, `nway`, `fragment`,
  `fragment_cross_maze`. Row 4 is a composition row in the draft's main table with no caveat column
  in existence. Draft does not declare this.
- Dumped all 21 w5 arms x 3 seeds from `w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`.
  Reproduced +10.143/+8.673/+7.796 and +4.033/+2.483/+2.417 exactly. ✅
- **FOUND: draft §2.3 and §4 print `178,928 params = -0.01 %`, a number w5's own LOG CORRECTED at
  01:58 UTC** as a val-leg figure. Scored legs: 178,408 / 179,838 / 179,838 = -0.304 % / +0.495 % /
  +0.495 %. Arm carries MORE params than the rival on 2 of 3 seeds.
- **FOUND: arm total deployed = 524,288 + 189,344..195,064 = 713,632..719,352 B; `rival_mlp_180k`
  = 178,952 x 4 = 715,808 B.** The arm is at 99.70 % / 100.50 % / 100.50 % of the rival's bytes,
  not at 73 % as "at 524,288 B" implies.
- **FOUND: `oracle_span` and `span_head` carry `address_bytes = 1,766,400`** (9.3x the arm's book,
  3.4x the store pin) and the draft's App. E ladder omits that column.

## 09:45Z — cross-check pass 2: the deletion lead (w8-downdate + w8-single-design)
Dumped `w8-downdate/DOWNDATE-READY-test.json` in full and read `w8-downdate/LOG.md` 612-646,
840-1000, and `w8-single-design/LOG.md` 470-640. Findings:
- `honest_total_bytes` = 12,869,482 / 12,500,170 / 12,409,722 B; artefact's own
  `times_over_budget` = **24.5466 / 23.8422 / 23.6697**. The downdate spoke's own final table
  prints "24.55x / 23.84x / 23.67x". **The draft's TABLE 4 prints no byte row at all.**
- `a1_subspace.passed_every_seed = **False**`, `seeds_failing = ['2']`, worst 60.36 ulp against
  its own floor 66.11 ulp. Draft never mentions `a1_subspace`.
- `cut_is_marginal_any_seed = True` and `cut_is_marginal = True` on all three seeds.
  `w8-single-design/LOG.md` 22:45Z: "it belongs in the caption of anything I score on that object."
  Draft has no caption.
- `/note` in the same JSON: "**No scored run and no claim**: these are properties of the object...
  hub/single-design scores the object." single-design scored **0 of 3** (NR3). Draft labels this
  leg `class RESULT` / `3 / TEST` (K2).
- **Two different per-seed absolute-deviation sets inside one JSON.**
  `/downdate_equals_refit/max_abs_dev_per_well_by_seed` = 2.365e-14 / 1.126e-09 / 7.105e-15 (what
  TABLES.md T4 and the draft print) vs `/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` =
  4.547e-13 / 1.126e-09 / 2.842e-14 (what the spoke's own FINAL REPORT §2 prints, "pinned 11,024
  rows + random 64"). Factors of 19x (seed 0) and 4x (seed 2).
- Reproducibility floors AFTER the cut: 2.127e-13 / 4.282e-09 / 5.684e-14. Against the spoke's own
  A1-after values, **seeds 1 AND 2 are below their floor**, not just seed 1 as the draft says.
- tau numerator `delta_rel_worst = 4.941056e-16` is **test seed 2's** leg worst
  (`w8-downdate/LOG.md` 840-855) — the failing leg. "No free parameter" is the spoke's own phrase
  and is defensible, but the constant is calibrated on the held-out TEST legs.
- **Carry cost superseded:** `w8-downdate/LOG.md:972` "127,488 B -> **125,952 B**"; FINAL REPORT
  line 1067 prints 125,952. Draft prints 127,488 twice (§5, never-quote #6).

## 10:05Z — cross-check pass 3: w9-maze, physics EVIDENCE.md, registry fold
- w9-maze deletion numbers verified cell-by-cell from `runs/scored{1,2,3}_*_test.json`
  `/display/delete_green_exactness`: 7.5217e-16 / 8.8265e-16 / 7.8176e-16, projector 0.0, rows
  216/992, 226/975, 193/1084, refit False, passes 0, items_kept False, carry bytes 0. **All exact.**
- **Environment naming:** `w9-maze/LOG.md` §1 says the scored environment is **CMAZE-2, a REQUEST
  layer over CMAZE-1**. The registry agrees (`w10-registry-fold/FINDINGS.md` §6: "N-I8-17 ...
  recorded as a REQUEST-LAYER weakness against CMAZE-2, not as a defect in the adopted CMAZE-1").
  TABLES.md T1 and the draft both label it CMAZE-1. T3 (harness tier) IS CMAZE-1. Two environments,
  one label.
- **FRAGMENT-LOOKUP is not 0.0000 everywhere.** `w9-maze/LOG.md` §3: same-maze r=1/r=2 =
  **0.0741 / 0.5105**; cross-maze 0.0006 / 0.0000. Draft §6.2 and K16 say "0.0000 everywhere".
  0.5105 against row 2's pipeline ceiling 0.6319 is the largest number in that table.
- EVIDENCE.md §2.3: **B3 max over gamma = 59.1158 / 0.1612 / 1883.2934 against a <=0.10 bar**,
  printed by the source as "bar not moved". Absent from the draft.
- EVIDENCE.md §2.2: instrument floor median 0.0349 / p90 0.0749 / **max 0.1601** — the max exceeds
  B1's own 0.10 bar. Draft prints the median only.
- EVIDENCE.md §2.3: 192,975 rows total, of which **41,493 / 41,405 / 41,358 sit in the 49 scored
  corners** (124,256). Draft attributes all 192,975 to "49 corners x 3 seeds".
- EVIDENCE.md §2.3: the fitted rival FIT1's own median error is **0.640132 / 0.623155 / 0.641281**
  vs the CRR's 0.0433. Draft prints the delta, not the rival's level.
- **Registry `N-I8-19` (tier A) is absent from the draft**: CLU refusal on CMAZE-2 is NOVELTY
  DETECTION, not wall detection (reachable-but-unwritten 0.9781 vs walled 0.9801), false-refusal on
  grounded-possible **0.8182 against a pre-registered 0.20**.
- Registry FINDINGS §5.5 asks the Hub to spot-check `N-I8-5`'s byte ledger against
  `DOWNDATE-READY*.json` — the spot-check I ran is what surfaced 24.5x and a1_subspace=False.
  The draft's §G carries 2 of the registry's 6 open items and not this one.

## 10:35Z — cross-check pass 4: novelty STOP, advisor reports, single-design sweeps
- `verify_tables.py` RE-RUN BY ME: **PASS** — 27 sources, 27 hashes, 240 cells, 720 per-seed values
  (3 via a declared reduction), 5 figure hashes, 5 NOT-RUNs, 4 corrections. Draft App. M is sound.
- Recomputed `span_ceiling_rows` vs `pipeline_ceiling_rows` from the three scored JSONs: **max
  |diff| = 0.013761, exactly equal on 7 of 12 cells**. Draft K15 exact. (Note: the max cell is
  seed 2 unseen_pair where the SPAN ceiling 0.7018 EXCEEDS the pipeline ceiling 0.6881.)
- NOVELTY.md §A/§B counts verified: 3,038 / 254 / 759 / ~100 / ~4,050; 10 of 3,038 (0.3 %), 7 FP,
  3 real; 0 of 3,038 for frozen-decoder+tensor-head+label-index; highest atom count 4 of 6.
- **NOVELTY STOP FAILS at §1.2 of the draft.** The draft points the Head at "NOVELTY.md §3 final
  para — the survivable form is verbatim there". That sentence reads: "the whole generative object
  is a sufficient statistic, so a generative model with **a request-indexed tensor head** admits
  deletion that is exactly the refit". `request-indexed tensor head` is STOP #1 verbatim.
- Advisor numbers verified: w4 -0.0192+-0.0028 / -0.0170+-0.0024, +1.388/-12.755/+2.163 mean -3.068
  spread 14.918; w5 18.6 tuples/corner, (1.0000,1.0000), 59,392x, knn_density 0.0203;
  w6 +0.0562/+0.0337/+0.0112, t_pay(50 %) 7.992/5.994/7.992. All exact.
- **w5-advisor §6.3 (unquoted by the draft):** "'The hole is exact and free' is supported;
  '**deletion is exact and free' is NOT** -- eviction costs composition" (-2.09 / -0.91 / -6.74).
- `w8-single-design/LOG.md:139-148` ward_balanced sweep: **five** configurations fit the 524,288 B
  pin at float64 -- K=36/A=4 span 144, K=38/A=7 span 266, K=40/A=9 span 360, K=42/A=12 span 504,
  K=48/A=18 span 864. Composition was measured at **exactly one**: K=40/**A=7**, span **280**
  (23:22Z), i.e. below the sweep's own K=40 row. Line 99 also prints a **float32** matched-bytes
  point K=21/A=94, 517,952 B = 0.988x, span 336 -- excluded by argument, never measured.
- `w8-single-design/LOG.md:52,89,121`: the statistic cost is the symmetric triangle
  `sum_k d_k(d_k+1)/2` at **8 B/float**, not `sum_k d_k^2`. Draft §5 prints the latter.

## 11:05Z — REFEREE-v0.md written
- 16 MUST-FIX, 20 SHOULD-FIX, 8 NICE-TO-HAVE, plus one item promoted out of should-fix (B3's
  printed max over gamma). Simulated verdict: **REJECT as it stands; accept-with-major-revision
  reachable at 0 scored runs.**
- Also written: a "keep exactly as is" section (7 items) and a rewritten contribution paragraph.
- Six DECLARED NOT-CHECKED items with reasons (figures as images, CM-I8-* cell-by-cell, the shorts
  PDFs, CROSS-ARM.json direct, the gym LOGs, sibling bundles).

## 11:15Z — commit, bundle, verify
- Committed `projects/w10-referee/{START_UTC.txt,LOG.md,REFEREE-v0.md}` on branch `w10/referee`
  at `89db529`. (`projects/*/` is gitignored in this repo; `git add -f` used, as the sibling
  w10 spokes did.)
- `git bundle create /c/chlu/outputs/w10-referee/w10-referee.bundle w10/referee` — **that branch
  only, never --all**.
- `git bundle verify`: *"is okay … The bundle records a complete history."* `list-heads` shows
  **exactly one ref**: `89db5290… refs/heads/w10/referee`.
- Independent check: cloned the bundle to the scratchpad and diffed the deliverable.
  **Identical after CRLF normalisation** (git's checkout conversion; 50,980 B vs 50,349 B),
  content sha256 `7ef7037d1d9aaafa…` on both sides. Clone removed.
- ⛔ No background job was ever launched by this spoke; nothing is running; no box-wide kill
  was issued or was possible.

## FINAL — report to the Hub
**REFEREE-v0.md: 16 MUST-FIX, 20 SHOULD-FIX, 8 NICE-TO-HAVE + 1 promoted (B3 max over gamma).**
**Simulated verdict: REJECT as it stands; accept-with-major-revision reachable at 0 scored runs.**
The task's falsifier ("you return no must-fix items") does not fire; nine of the sixteen are in
the byte ledgers, exactly where the task said to look.
