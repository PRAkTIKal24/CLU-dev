# w15/table-fixes — LOG

Slug `table-fixes`. Branch `w15/table-fixes`. Base `base/iter8 = 148f3b7`.
Worktree `/c/chlu/worktrees/CHLU-w15-table-fixes`. Output `/c/chlu/outputs/w15-table-fixes/`.
**Scored runs spent: ZERO. Compute: ZERO. No model fitted, no arm run, no background job launched.**

## 21:41Z — environment
`source /c/chlu/env_w3.sh` from inside the worktree. First assert FAILED with
`ModuleNotFoundError: No module named 'equinox'`.
⚠ **CAUSE FOUND (an environment bug other spokes will hit, reported, not fixed by me):**
`env_w3.sh` sets `PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"` with a **POSIX `:` separator**,
but the interpreter is **Windows CPython 3.11**, which splits `PYTHONPATH` on `;`. The whole value is
therefore read as ONE bogus path
(`C:\chlu\worktrees\CHLU-w15-table-fixes:\c\Users\...\site-packages`) and the wave-2 site-packages is
never on `sys.path`. `chlu` still imported only because CWD (`''`) is on `sys.path`.
**Workaround used (no file edited outside my own tree):** re-export
`PYTHONPATH="C:/chlu/worktrees/CHLU-w15-table-fixes;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"`.
Assert then passes:
`OK C:\chlu\worktrees\CHLU-w15-table-fixes\chlu\__init__.py` · `jax 0.9.0 · numpy 2.4.1 · equinox 0.13.4`.
⭐ The pinned env is the wave-2 one, as required. ⛔ No venv built, no `uv sync`.

## 21:45Z — brief located, sources identified
- The must-fix items are named in `/c/chlu/outputs/w13-referee-2/REFEREE-v1.md` §2 (MF2, MF14) and §3
  (N5, N6, N7, N8) — mirrored in-tree at `w13_referee_2/REFEREE-v1.md`.
- The file to fix is `w13_draft_revise/DRAFT-v1.md` (in-tree, folded into base/iter8 at 148f3b7).
- ⭐ **CHOICE, recorded per COMMON.md:** the in-tree `w12r_artefact_mustfix/TABLES.md` (built
  2026-09-21T12:50:40Z) carries **only T1–T6**. T7–T12, which four of my six items cite, live in
  `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md`. I cite **that** file by full path for every T7+
  number, and the in-tree copy for T1–T6, rather than silently citing "TABLES.md" and letting the
  Head resolve the wrong one. Closest to an ICLR main-track result: a citation that resolves.
- ⛔ I did NOT merge `w10/draft-v0` or `w10/referee`, and I did NOT touch `verify_tables.py`.

## 21:55Z — every number re-derived from its primary before any edit
- BLIND columns read straight out of the three scored JSONs, not out of TABLES.md:
  `/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json`
  `/env_columns/{unseen_pair,nway}/blind/{BLIND-SEQUENTIAL(key order from A),BLIND-FACTOR    (A x B local only)}/validity`
  → unseen_pair SEQ `[0.9077, 0.8905, 0.9128]` mean 0.9037 · FACTOR `[0.4564, 0.4677, 0.5046]` mean 0.4762
  → nway        SEQ `[0.9229, 0.8933, 0.8841]` mean 0.9001 · FACTOR `[0.4447, 0.4665, 0.5048]` mean 0.4720
  `/env_columns` keys = `['unseen_pair','nway','fragment','fragment_cross_maze']` — **four**, confirming
  MF14 part (ii): rows 1 and 4 have no blind block on any seed.
- Byte ratios recomputed from the stored byte counts:
  statistic 12,327,096 / 11,957,784 / 11,867,336 ÷ 524,288 = **23.5121× / 22.8077× / 22.6351×**
  honest total 12,869,482 / 12,500,170 / 12,409,722 ÷ 524,288 = **24.5466× / 23.8422× / 23.6697×** (matches T4)
  K\* honest totals 2,144,178 / 2,137,418 / 2,141,578 ÷ 524,288 = **4.0897× / 4.0768× / 4.0847×**
  K\* val pin 2,143,138 ÷ 524,288 = **4.0877×** (matches T12's K = 32 row)
  ⚠ **The referee's own MF2/N7 figure `23.5117×` is a rounding slip; the correct seed-0 value is
  `23.5121×`.** I land `23.5121×`, note the slip, and the "23.51×" of my task file is unaffected at 2 dp.

## 22:05Z — CHOICE RECORDED (COMMON.md: "where a choice is open, pick the ICLR option and record it")
1. **Edit `w13_draft_revise/DRAFT-v1.md` IN PLACE rather than forking a v2 in my output dir.**
   The file is in `base/iter8` at 148f3b7, so the branch diff is the reviewable unit and the Head
   reads one file, not two. A pre-edit copy and a unified diff are in the output dir so nothing is
   lost. ⭐ Closest to an ICLR main-track result: one draft, one history.
2. **Apply the edits with a guarded, idempotent SCRIPT (`w15_table_fixes/apply_table_fixes.py`)
   rather than by hand.** 22 exact-string replacements, each with a unique `mark` so a re-run cannot
   double-insert, each asserted to match exactly once. ⚠ The first version of the script used
   "does `old` still match?" as its applied/not-applied test; two edits PREPEND to text they keep
   verbatim, so a second run double-inserted them. Caught by running it twice, reverted with
   `git checkout --`, and fixed with mandatory marks. Recorded because it is the exact class of
   error this spoke exists to remove.
3. **Cite T7+ by full path to `/c/chlu/outputs/w13-seeds-and-tables/TABLES.md`.** The in-tree
   `w12r_artefact_mustfix/TABLES.md` stops at T6; four of my six items cite T7–T12. A citation that
   does not resolve is not a citation.
4. **Close the same defect at its remaining loci, not only where the referee pointed.** N5/N6's
   error class (a byte figure with no convention, a ratio with no seed count, an unnamed "K = 5")
   also occurred at §0.2 bullets 1 and 3, §A's MF3 row, §B's K12 and K19 rows and §G's C16 row.
   Five further edits, zero cost. ⛔ Leaving them would have re-created the must-fix next wave.
5. **Add an amendment banner at `DRAFT-v1.md:14-25`.** Not asked for. Without it the Head reads a
   file headed *"w13 spoke/draft-revise ... this spoke produced no number"* that a different spoke
   has since changed.

## 22:20Z — the six items landed
MF2 `:1340-1345` + `:232` + `:1037` · MF14(i) `:665-671` + `:673-680` + `:708-721` + `:782-791` +
`:1049` · N5 `:233-234` + `:242-250` + `:64-69` + `:592` + `:596-601` + `:1039` + `:1104` + `:1276` ·
N6 `:486-496` + `:576` + `:78` + `:1097` · N7 = MF2's clause `:1340-1345` · N8 `:1005-1016` ·
§A header `:1026` · banner `:14-25`.
`git diff --stat`: **137 insertions, 40 deletions, 1 file.** Second run of the script: 22/22
`already applied`, file unchanged. ⛔ ALL SIX LANDED — nothing DECLARED NOT-DONE within my scope.

## 22:25Z — ⛔ THE FALSIFIER IN MY TASK FILE FIRED, AND IS REPORTED
*"If moving the BLIND columns into the main table makes the maze claim look weaker … report it."*
It does. On rows 2 and 3 **BLIND-SEQUENTIAL (0.9037 / 0.9001) and BLIND-FACTOR (0.4762 / 0.4720)
both beat the CLU (0.1279 / 0.0284) and both beat the best learned baseline (0.1840 / 0.0493)** —
7.07× / 31.7× and 3.72× / 16.6× the CLU's own score. ⛔ Not softened, not moved back, not
footnoted. The caption states the consequence: a policy that never reads the request scores 0.90,
so the split does not test composition at all — the environment's own tripwire fires before the arm
is reached. ⭐ It costs the paper no result (§6.1 already claimed none under BAR 3) and buys it a
narrower, harder-to-attack sentence. Full reading in `FINDINGS.md` §3.

## 22:30Z — outputs written, branch committed and bundled
`TABLE-FIXES.md` (acceptance artefact), `FINDINGS.md`, `DRAFT-v1.BEFORE.md`, `DRAFT-v1.AFTER.md`,
`DRAFT-v1.diff`, `apply_table_fixes.py`, `PIDS.txt` (empty by design — zero background jobs, zero
kills), `LOG.md`, `START_UTC.txt`, `END_UTC.txt`.
⛔ Nothing was written outside `/c/chlu/outputs/w15-table-fixes/` and my own worktree.
⛔ `verify_tables.py` not touched. ⛔ `TABLES.md` / `TABLES.json` / `build_tables.py` not touched.
⛔ `w10/draft-v0` and `w10/referee` not merged. ⛔ No sub-agent spawned. ⛔ No scored run, no ledger
row required. 1 thread throughout.
