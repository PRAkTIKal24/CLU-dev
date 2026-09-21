# w13 spoke/registry-fold — LOG

- 2026-09-21T17:03:11Z START_UTC.txt written.
- Read /c/chlu/tasks/w13-spoke-registry-fold.md. NOTE: standing-rules block contains a
  templating glitch ("/c/chlu/outputs/w13-w13-spoke-registry-fold/", branch
  "w13/w13-spoke-registry-fold"). The job section and the invocation both say
  /c/chlu/outputs/w13-registry-fold/ and branch w13/registry-fold — using those.
- Base chosen: /c/chlu/evidence/outputs/w12r-registry-repair/ (negative_results.md 5268 lines,
  claims_matrix.md 1235 lines) per task file. Confirmed it is the superset carrying N316-N325 and
  §I8.4/§I8.5 (w12r FINDINGS.md §0).
- W8 VERIFICATION PASS (no re-fold): all five w8 items present and correct in the base copies —
  N-I8-4 (encoder-loop wash, decomposition A'/A''), N-I8-5 + the in-place w12r qualifier + N325
  (K>=36), N-I8-18 (TensorFaces/MLC-VAE STOP), N-I8-16 (MiniGrid/gym laws), N-I8-14 (Advisor's
  Mexican-hat error). No gap found; nothing re-folded.
- Read w12 evidence: w12-composition-ladder/{FINDINGS.md,LADDER.json,LADDER.md,LOG.md,BYTES.json,
  HEAD-RULING-K.md}, w12-single-design-crossover/{FINDINGS.md,LEDGER-ACROSS-K.json},
  w12-deletion-baselines/FINDINGS.md, w12-kway-ladder-harness, advisor-wave-plan-w12-w14.md:32,
  w9-/w10-advisor-report.md. $CHLU_CLAUDE_ROOT never opened.
- chlu.__file__ = C:\chlu\worktrees\CHLU-w13-registry-fold\chlu\__init__.py — asserted.
- Wrote negative_results.md: header note + 3 additive in-place pointer notes (N-I8-1, N-I8-5, N325)
  + the OMEGA block with N326-N336. 5268 -> 5732 lines. difflib: 5269 equal, 464 insert,
  ZERO replace/delete blocks.
- Wrote claims_matrix.md: title -> v2.20 (PROPOSED, prior title preserved verbatim as its tail)
  + SS-I8.6 rows CM-W12-1..10 + SS-I8.7 open flags. 1235 -> 1511 lines. lines 2..1236
  byte-identical (True); only line 1 replaced.
- Checks: 349 entry headings in the registry, ZERO duplicate N numbers anywhere; N326-N336 each
  exactly one heading, each carrying 2-8 distinct artefact path references; CM-W12-1..10 each
  exactly one heading.
- /c/chlu/evidence/ md5 re-checked at close and byte-identical to arrival (see FINDINGS.md).
- ZERO scored runs, zero compute, zero GPU, zero background jobs, zero PIDs, no box-wide kill.
