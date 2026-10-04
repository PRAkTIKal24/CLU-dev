# w12r spoke/registry-repair — LOG

- 2026-09-21T12:38:57Z — START. Wrote START_UTC.txt. Read task file
  `/c/chlu/tasks/w12r-spoke-registry-repair.md`.
- 12:40Z — Read the evidence base. ⛔ `$CHLU_CLAUDE_ROOT` NOT opened at any point.
  Sources read: `/c/chlu/evidence/outputs/w9-advisor-report.md`, `w10-advisor-report.md`,
  `w8-advisor-report.md` (scoreboard only), `w5-advisor-report.md` §5, `w6-advisor-report.md` §§0-2,
  `w8-BAR-D-fold.md`, `w10-referee/REFEREE-v0.md` §MF15/MF16, `w10-registry-fold/FINDINGS.md`,
  `w10-physics-section-evidence/{LOG.md,EVIDENCE.md}`, `_thinkpad/w9/w9-harness/LOG.md:215-250`,
  `_thinkpad/w8/w8-encoder-loop/LOG.md`, `_thinkpad/w8/w8-env-scout/ENV.md` (via the registry quote).
- 12:41Z — BASE CHOICE, decided and mechanically checked. The live `/c/chlu/evidence/` pair is
  3794 / 643 lines (md5 `c3597411aa834304be27110dab115673` / `39be4b5f1d8f29c5058c5a0bef935243`).
  The `w10-registry-fold` pair is a STRICT SUPERSET: `head -3794 negative_results.md | diff -` against
  the live copy is EMPTY, and `head -643 claims_matrix.md | diff -` differs on LINE 1 ONLY (the title
  bump, prior title preserved verbatim as its tail). ⇒ this pass bases on the w10 copies, so the
  21 `N-I8-*` entries and 13 `CM-I8-*` rows are not dropped. ⛔ `/c/chlu/evidence/` NOT written.
- 12:52Z — DEFECT 1 EXECUTED in `negative_results.md`. `N89`(gen-pivot) → **N316**, `N90`(gen-pivot) →
  **N317**, content VERBATIM (checked with `str.endswith` on the untouched tail: both True).
  Redirects printed at: `### N89` header note, `### N90` header note, the head of
  `## ⟲ 2026-09-11 — GEN PIVOT, PHASE 1`, the file header, and the canonical REDIRECT TABLE in `⌖.1`.
  Superseded-in-place (not edited): the iter8 fold's "WHY IT IS NOT `N316`" protocol paragraph, and
  `N-I8-21`(a)'s "FLAGGED, NOT FIXED".
  ⭐ MECHANICAL ADDITIVITY CHECK: of the 3794+945 = 4739 lines of the base copy, exactly **2** are not
  present verbatim in the new file — the two renumbered label lines. Everything else is additive.
- 12:55Z — DEFECT 3 EXECUTED. `⌖.2` W5/W6 CROSSWALK (10 rows, each with the number the finding ALREADY
  holds + the path). `grep -c -i "gym-1|gym-2|illegal"` over the registry: 6 → **13**.
  ⛔ NO duplicate number allocated — declined on `N315`'s own authority; the alias scheme is written out
  as the rejected alternative so the Hub can pin numbers if it prefers.
  `N318` / `N319` / `N320` registered — three w6 negatives with NO entry anywhere
  (the `+1.1385` artefact trail · D18 INCONCLUSIVE · the tautological `d14_confinement` field).
- 13:05Z — FOLD-IN EXECUTED. `N321` (encoder-loop GPU corroboration + the unstable `alone` control),
  `N322` (BAR C.2 equal-seconds bias), `N323` (the w9 refuser + why), `N324` (3 failing tests, UNOWNED),
  `N325` (the referee's MF16 qualification of the K ≥ 36 collapse). `N-I8-5` link 3 annotated in place.
  Fold-in ledger at `⌖.5` accounts for all 10 items named in the task file.
- 13:06Z — `N324` VERIFIED, not asserted: `python -m pytest tests/test_gen_value_holdout.py -q
  --no-header -p no:cacheprovider` in the worktree at `aa91785` ⇒ **3 failed, 28 passed in 63.11s**.
  ⛔ A unit-test invocation, NOT a scored run; no SCORED-LEDGER row; `coverage.xml`/`.coverage` deleted,
  `git status` clean before and after.
- 13:15Z — DEFECT 2 EXECUTED in `claims_matrix.md`. NEW `§I8.4` with `CM-GP-1`, `CM-GP-2` (the gen-pivot
  rows — the defect itself) and `CM-R12-1` … `CM-R12-6`; NEW `§I8.5` (6 open flags). Title bumped to
  **v2.19 (PROPOSED)**, prior title preserved verbatim as its tail (14,098 → 16,270 chars).
  ⭐ MECHANICAL CHECK: lines 2–1039 byte-identical to the base copy (True). Line 1 is the only edit.
- 13:18Z — EVIDENCE READ-ONLY RE-VERIFIED at end of session. md5 of all four source files unchanged
  from arrival: `c3597411…` / `39be4b5f…` / `3b3e34f0…` / `47833db0…`.
- 13:20Z — FINDINGS.md written. 10 DECLARED OPEN ITEMS, each with its reason and its closing action.
