# w13 spoke/seeds-and-tables — LOG

- 2026-09-21T17:03:11Z — START_UTC.txt written. Output dir /c/chlu/outputs/w13-seeds-and-tables/.
  Note: task file standing-rules block contains a template-substitution artefact
  (`/c/chlu/outputs/w13-w13-spoke-seeds-and-tables/`, branch `w13/w13-spoke-seeds-and-tables`).
  Using the job-section + invocation paths: dir `/c/chlu/outputs/w13-seeds-and-tables/`,
  branch `w13/seeds-and-tables`. DECLARED deviation, reason: the doubled prefix is a
  substitution bug; both the job header and the invocation agree on the single-prefix form.

## Survey (17:03–17:25Z)
- BEFORE baseline: `python verify_tables.py` on `/c/chlu/outputs/w12r-artefact-mustfix/tables-r1/`
  → **PASS**, sources 32/32, cells 286, per-seed values 854 (3 reductions), figures 5/5,
  declared NOT-RUN 5, corrections 14.
- Generator copied from `/c/chlu/outputs/w12r-artefact-mustfix/tables-r1/build_tables.py` (170,418 B,
  2,798 lines); `verify_tables.py` copied byte-identical (8,561 B) — it is the acceptance check and
  is NOT edited. `diff` vs `scripts/w10_verify_tables.py` = IDENTICAL.
- `chlu.__file__` = `C:\chlu\worktrees\CHLU-w13-seeds-and-tables\chlu\__init__.py` — asserted.
- Wave-12 artefacts read (structure + pointers mapped): crossover CROSSOVER-val.json,
  LEDGER-ACROSS-K.json, scored/run1_seed0_K32/scored.json, deletion/{test_seed0_K32,val_seed0_K5}/
  deletion.json, deletion/probe_test_seed0_K32.json; ladder LADDER.json + BYTES.json;
  deletion-baselines BASELINES-READY.json; cache-gen2 CACHE-READY.json + VERIFY.json.

## Build (17:25–17:40Z)
- Patch method is w12r's: `patch/p*.py` exact-string patches, each asserting its anchor occurs
  EXACTLY once. ⭐ The GENERATOR is edited; no output is touched by hand.
- p1 — 19 new sources registered (32 → 51), `NOTATION` block (n = rungs vs K = wells),
  new cell helpers `wcell` / `w1` / `w3` (wave-12 artefacts hold every seed inside ONE json).
- p2 — TABLE 7 (scored rows 3/4 at K* = 32, TEST, **1 seed**) and TABLE 8 (n-way ladder,
  n = 2…6, **K = 5**, 3 seeds, BOTH byte matchings on every row).
  verify after p2: **PASS**, 51 sources, 494 cells, 1,271 values.
- p3 — TABLE 9 (deletion, TEST K*=32) + TABLE 9b (deletion, VAL K=5, labelled SMOKE because
  `/SMOKE_NOT_A_CLAIM` is true in that artefact) + TABLE 10 (refusal DISPLAY) +
  TABLE 11 (the two ablations) + TABLE 12 (the K-curve).
  verify after p3: **PASS**, 51 sources, **872 cells**, 1,786 values.
- p4 — fig6 (ladder in n, both matchings), fig7 (K-curve, both totals, n annotated per point),
  fig8 (four-part ledger stacked across K); the generic renderer for T7…T12; NR6…NR10;
  C15…C19; manifest blocks `/notation`, `/seed_ledger`, `/gate_checks`, `/w13_fold`.
- p5 — verify FAILED once here: `MD PATH NOT IN REGISTER /c/chlu/outputs/spoke-a-rivals`.
  NR7 had cited a bare DIRECTORY, which is not a registerable source. Fixed by registering
  `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` and citing `/rivals_seeds_present`
  (= [0,1,2] for all seven families) — the field that actually carries the claim. 52 sources.
- p6 — three rendering defects found by READING the rendered md: a 1-seed cell printed as
  `516,096 (516,096)`; the ledger header hard-coded `(seed 0 / seed 1 / seed 2)` on 1-seed
  tables; a duplicated `seeds (n)` column. All three fixed in the generator.
- p7 — two figure defects found by LOOKING at the pngs: fig6 clipped its ceiling annotations
  and omitted `pca_tensor` (the ladder's strongest finding); fig8's pin annotation sat on a bar.

## Acceptance (17:50Z)
- `python verify_tables.py /c/chlu/outputs/w13-seeds-and-tables` → **PASS**
  · sources 52/52 · cells **872** · per-seed values **1,786** (3 reductions) · figures 8/8
  · declared NOT-RUN **10** · corrections **19**.
- BEFORE (`/c/chlu/outputs/w12r-artefact-mustfix/tables-r1`) → PASS · 32/32 · 286 · 854 · 5/5 · 5 · 14.
- ⛔ `verify_tables.py` never edited: `diff` against `scripts/w10_verify_tables.py` is EMPTY.
- Determinism: two consecutive builds → all 8 figure sha256 identical; TABLES.md differs only in
  the `**Built**` timestamp line; TABLES.json equal with `generated_utc` popped.

## Close (17:55Z)
- Committed to branch `w13/seeds-and-tables` as `56bf53f` (parent `aa91785 = base/iter8`):
  `scripts/w13_build_tables.py`, `scripts/w13_verify_tables.py`, `scripts/w13_patch/p1..p7`.
  Repo convention followed — w10 committed only its two scripts (`git show --stat b1068b0`).
- `git bundle create … w13/seeds-and-tables` — THAT BRANCH ONLY, never `--all`.
  `git bundle verify` → "is okay", "records a complete history".
  `git bundle list-heads` → **exactly ONE ref**: `56bf53f… refs/heads/w13/seeds-and-tables`.
  6,092,423 B.
- fig1–fig5 byte sizes are unchanged from w12r's (96,447 / 85,322 / 63,483 / 100,298 / 69,808) —
  the five carried figures were not perturbed.
- No background jobs were launched at any point; nothing was left running.
- ⛔ No box-wide kill; no other agent's session, worktree, output dir or process was touched.
