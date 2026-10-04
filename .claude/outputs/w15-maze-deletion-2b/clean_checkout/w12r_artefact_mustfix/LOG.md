# w12r spoke/artefact-mustfix — LOG

- START_UTC 2026-09-21T12:39:00Z
- Branch: w12r/artefact-mustfix ; worktree C:/chlu/worktrees/CHLU-w12r-artefact-mustfix
- ZERO compute, ZERO scored runs.

## Entries
- 12:39 Wrote START_UTC.txt, created LOG.md. Read task file /c/chlu/tasks/w12r-spoke-artefact-mustfix.md.
- 12:45 Read REFEREE-v0.md (631 lines) MF1-MF16 + SF list. Read verify_tables.py.
- 12:47 BASELINE VERIFIER (pristine /c/chlu/outputs/w10-seeds-and-tables): PASS.
  sources 27 / hashes 27 / cells 240 / values 720 (3 reductions) / figures 5 (5 hashes) /
  declared NOT-RUN 5 / corrections 4.
- 12:48 Copied w10-seeds-and-tables -> ./tables-r1 (bundle+START/END stripped). Verifier PASS in my dir,
  identical counts. This is the artefact I edit; /c/chlu/evidence and /c/chlu/outputs/w10-* untouched.
- 12:55 Confirmed build_tables.py is DETERMINISTIC: rebuilt tables-r1 into scratchpad, TABLES.md
  byte-identical except the generated_utc line; TABLES.json equal ex-timestamp; all 5 figure
  sha256 identical. => I will edit build_tables.py (the generator) rather than hand-patching the
  outputs, so the artefact stays regenerable and verify_tables.py keeps passing.
- 12:57 Byte-ledger sources located:
  * w8 leg, /c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json: store_bytes 524288 (all seeds);
    cov_bytes_per_seed 12310672/11941360/11850912; honest_total_bytes 12869482/12500170/12409722;
    times_over_budget 24.5466/23.8422/23.6697; items_kept_detail/object_bytes_full
    12885866/12516554/12426106.
  * w8 four-part ledger text: /c/chlu/outputs/w8-downdate/LOG.md:414-422 and :1036-1044 ->
    part1 store 524,288 | part2 running covariance | part3 request count table 11,488 |
    part4 coupling 5,288 + holes 1,322.
  * ARITHMETIC CHECK (mine): 524288 + cov + 11488 + 5288 + 1322 leaves a residual of exactly
    16,424 B on ALL THREE test seeds. 16,424 = sum_k d_k x 8 (= 16,384, the running MEANS) + 5x8
    (= 40, the per-well COUNTS). d_k from DOWNDATE-READY-test.json /per_seed/*/content_cost/per_well.
    With that, the ledger closes EXACTLY: 524288+(cov+16424)+11488+5288+1322 = honest_total_bytes.
    => part 2 must be labelled "running covariance + running mean + count", not covariance alone.
  * w9 leg, /c/chlu/outputs/w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json /ledger:
    store_atom_floats 7936 (=31,744 B f32), head_floats 402752 (=1,611,008 B f32),
    deployed_bytes 1,642,752, statistic_bytes 35,904 (4,488 float64). Identical on all 3 seeds.
    ⛔ NO coupling/hole field exists in the w9 maze artefacts (grep: no 'hole', no 'coupl').
- 13:05 Patch plan settled. Edits go into tables-r1/build_tables.py (the GENERATOR), never into
  TABLES.md/TABLES.json by hand, so the artefact stays regenerable and verify_tables.py (which
  does NOT import the generator) is a real check.
- 13:10-14:20 Applied patches p1..p8 (all in ./patch/, each an idempotent exact-string replacement
  script, each asserting its anchor occurs exactly once):
  p1 helpers (derived/locator/dsrc/fmt_any) + /exactness_definition + /environments +
     /seed_and_configuration_counts
  p2 T1: MF10 relabel CMAZE-2, MF3 four-part ledger, MF11 fragment same/cross split,
     MF14 declared-not-measured row 4
  p3 T2 ledger; T3 relabel CMAZE-1 PROPER, ledger NOT-APPLICABLE, caption "same environment" fixed
  p4 T4: MF7 row class, MF3 ledger both legs, MF8 corrected field, MF9 both floors, MF6 caveats
  p5 T5 (three implementations, failure agreement) + manifest top level + rules_applied
  p6 figures follow MF10/MF8; render_md for every new block
  p7 corrections C5..C14 recorded in the manifest
  p8 T6 (w5 byte/parameter ledger: MF12, MF13, SF2) + MF16 qualifier on NR3 + MF15 refusal reading
- 14:30 Also fixed, found while rendering: booleans were being averaged (`refit performed 0.0
  (False/False/False)`); `cell()` now excludes bool from the numeric mean.
- 14:49 FINAL VERIFIER (tables-r1, after all edits): PASS.
  sources 32 / hashes 32 / cells 286 / values 854 (3 reductions) / figures 5 (5 hashes) /
  NOT-RUN 5 / corrections 14.  BEFORE was 27/27/240/720/5/5/4, also PASS.
- 14:50 DETERMINISM re-checked after every edit round: rebuilt into the scratchpad, TABLES.md
  byte-identical except the `**Built**` line, TABLES.json equal ex-timestamp, all 5 figure
  sha256 identical.
- 14:52 tables-r1/README.md written; the w10 spoke's LOG/FINDINGS renamed W10-ORIGINAL-* so they
  are not mistaken for this pass's.
