# w12 spoke/kway-ladder-harness — LOG

All times UTC. Worktree: `C:/chlu/worktrees/CHLU-w12-kway-ladder-harness`, branch `w12/kway-ladder-harness`.

## 11:09:29 — start
- `START_UTC.txt` written.
- Read `/c/chlu/tasks/w12-spoke-kway-ladder-harness.md` (standing rules + job).

## 11:10 — interpreter pin (standing rule)
- `python -c "import chlu; print(chlu.__file__)"` →
  `C:\chlu\worktrees\CHLU-w12-kway-ladder-harness\chlu\__init__.py` — resolves INSIDE my worktree. ✅
- Python 3.11.9.

## 11:11–11:25 — reading the existing ends (no numbers produced yet)
- `chlu/data/gen_value_holdout.py` (1749 lines): `build_pair_cells` (row-3 CELLS, `n_pair_cells=3`),
  `build_nway_cells` (row-4 CELLS, `n_nway_cells=49`), `_consecutive_runs`, `_disjoint_windows`.
- `chlu/data/gen_splits.py`: `FactorCell` (absent factor ⇒ unconstrained), `assert_cells_disjoint`,
  `build_cell_split`.
- `chlu/experiments/exp_gen_factor_wells.py`: ⭐ the CORNER layer, which is the thing the ladder
  actually scores — `nway_corners()` (row 4: the 49 cells, one corner each) and
  `pair_subcorners()` (row 3: the 3 pair CELLS **tiled** into Montero-shaped sub-corners —
  60 of them), `assert_corners_disjoint_and_heldout()`, `paired_estimand()`.
- ⇒ The distinction that decides the whole design: a **cell** is what is held out of the store's
  fit pool; a **corner** is a Montero-shaped probe box inside a cell. Row 3 = 3 cells / 60 corners,
  row 4 = 49 cells / 49 corners. The ≥49 bar is on CORNERS.

## 11:26 — PREREG written BEFORE any number (Charter A.1)
- `PREREG.md` (dated 2026-09-21) — the bar, the disjointness definition, the two end-reproduction
  assertions, and the kill conditions, all fixed before the builder was run once.

## 11:30–12:05 — the builder
- `chlu/data/gen_kway_ladder.py` (1 095 lines). Key design call, recorded in its §0: a
  **cell** (removed from the fit pool) is not a **corner** (the estimand's unit). Row 3
  clears 49 with 3 cells only because `pair_subcorners` TILES them. The ladder generalises
  the tiling: pin k, tile 6−k, same widths.
- `floorwall` is an ATOMIC coordinate (rows 3 and 4 both tie `wall_hue` to `floor_hue`), so
  it pins TWO factors. That is what forces `A4 ⊄ A5`.
- `scripts/w12_kway_ladder.py` — runner: kill check -> ends -> rungs, in that order.

## 12:06 — KILL CHECK, toy grid (offline Shapes3D-shaped fixture)
- All five rungs PASS. k=3/4/5 = 120 corners each vs bar 49.
- E1 ok (120 k=6 cells == `build_nway_cells`), E2 ok (60 k=2 corners == `pair_subcorners`).
- Crash on a console-encoding issue in one `log()` string (cp1252 vs ⛔); numbers unaffected.
  Fixed by de-emoji-ing the three console strings and setting `PYTHONIOENCODING=utf-8`.

## 12:10 — two checks added after seeing the row budgets
- `row_matching_feasibility()` — asks whether a common held-out ROW budget exists.
  Answer: **NO** (`required_floor` 45,000 > `available_ceiling` 8,640).
- `fit_pool_feasibility()` — every rung's train index vs `PINS['n_pool'] = 131,072`.
  All five clear it; tightest is k=3 with 48,928 rows of headroom. ⇒ the store is fit on
  the SAME 131,072 rows at every rung, which defuses most of the depth/fraction confound.

## 12:15 — REAL GRID (`C:\chlu\data\gen\shapes3d\3dshapes.h5`, 267,573,662 B)
- Bijection verified on 480,000 rows. Same counts as the toy grid, as designed.
- `KILL-CHECK.json` and `LADDER-READY.json` written to the output dir.

## 12:20–12:40 — tests, and one that mattered
- `tests/test_gen_kway_ladder.py` (280 lines, 35 tests) incl. 4 positive controls.
- ⚠ `test_k2_end_reproduction_fires_when_the_tiling_widths_drift` FAILED on first run —
  correctly. E2's reference side (`row3_subcorners`) was reading `cfg.widths`, the same
  config as the side under test, so the check could not fail. Fixed: `ROW3_TILE_WIDTHS` is
  now a literal, re-tied to `exp_gen_factor_wells.PINS['pair_tile_*']` by
  `assert_row3_widths_match_pins()` at emit time (status PASS in the artifact).
- Re-ran: **35 passed in 17.11 s**.

## 12:45 — regression sweep
- `test_gen_value_holdout.py + test_gen_data_splits.py + test_gen_kway_ladder.py`:
  **82 passed, 3 failed, 5 skipped**.
- The 3 failures are **PRE-EXISTING** at `aa91785` and are in `test_gen_value_holdout.py`
  (stale expectation `'row2_unseen'` vs the shipped AMENDMENT-1 `'row1_seen'`; missing
  `reg['row5_vs_row2']`). ⛔ `git diff --stat` is EMPTY — w12 modified no tracked file.
  Reported to the Hub, not patched: it is another module's test.

## 12:50 — artifacts regenerated after the E2 fix, FINDINGS written
- sha256 `LADDER-READY.json` = `4255bc63c9ecdfdb912413334cc3dd470353379a4f1506964e0169aeaf8f52c6`
- sha256 `KILL-CHECK.json`   = `8b95a950ccc08215878c4d0dca2558070f46942328620d106beac5ac6343fc76`
- sha256 `PREREG.md`         = `b2892de8df36ad8f5206cfc7acd86682301efd5c3b9363aed48c234e2e358598`

## 13:2x — commit, bundle, close
- Commit `04a3d07` "[w12] kway-ladder-harness: the k=3,4,5 composition families".
  3 files ADDED, 0 tracked files modified.
- `git bundle create … w12/kway-ladder-harness` (that branch only, ⛔ never `--all`).
  `git bundle verify` → "is okay", "The bundle records a complete history."
  `git bundle list-heads` → exactly **1** ref:
  `04a3d07e7850a419cb23d0f7b783a89d5e1d9f0b refs/heads/w12/kway-ladder-harness`.
- No `.claude/` directory created in the worktree. No detached background work.
- `FINDINGS.md` written. `END_UTC.txt` written last.
