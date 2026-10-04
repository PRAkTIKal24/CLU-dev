# w12 spoke/kway-ladder-harness — FINDINGS

**Report to the Hub.** Built 2026-09-21. Branch `w12/kway-ladder-harness`.
**Scored runs spent: 0** (charter: this spoke builds the instrument; `hub/composition-ladder`
runs it).

Every number below names the file it came from. Artifacts, all in
`/c/chlu/outputs/w12-kway-ladder-harness/`:

| file | sha256 |
|---|---|
| `PREREG.md` | `b2892de8df36ad8f5206cfc7acd86682301efd5c3b9363aed48c234e2e358598` |
| `KILL-CHECK.json` | `8b95a950ccc08215878c4d0dca2558070f46942328620d106beac5ac6343fc76` |
| `LADDER-READY.json` | `4255bc63c9ecdfdb912413334cc3dd470353379a4f1506964e0169aeaf8f52c6` |

Code, on the branch: `chlu/data/gen_kway_ladder.py` (new, 1 095 lines),
`tests/test_gen_kway_ladder.py` (new, 35 tests), `scripts/w12_kway_ladder.py` (new, runner).
⛔ **Zero tracked files were modified** — `git diff --stat` against `aa91785` is empty.

Grid: the **real** `C:\chlu\data\gen\shapes3d\3dshapes.h5` (267 573 662 B), bijection verified
on **480 000** rows (`scripts/w12_kway_ladder.py` stdout; `LADDER-READY.json:grid`).
Interpreter pin: `chlu.__file__ = C:\chlu\worktrees\CHLU-w12-kway-ladder-harness\chlu\__init__.py`
— asserted in the runner before any number.

---

## 1. ⭐ THE HEADLINE — the ladder is BUILT and every rung clears the bar

**k = 3, 4 and 5 each have 120 pairwise-disjoint corners against a bar of 49.**
No rung is a NOT-RUN. Both existing ends reproduce exactly.

| rung | pinned factors (the box) | free | cells | **corners** | bar | rows/corner | status |
|---|---|---|---|---|---|---|---|
| k=2 (row 3) | object_hue, shape | floor, wall, scale, orient | 3 | **60** | 49 | 36 | PASS — *reproduced, not rebuilt* |
| **k=3** | object_hue, shape, scale | floor, wall, orient | 12 | **120** | 49 | 72 | **PASS (NEW)** |
| **k=4** | object_hue, shape, scale, orient | floor, wall | 24 | **120** | 49 | 72 | **PASS (NEW)** |
| **k=5** | object_hue, shape, scale, floor, wall | orient | 60 | **120** | 49 | 72 | **PASS (NEW)** |
| k=6 (row 4) | all six | — | 120 | **120** | 49 | 72 | PASS — *reproduced, not rebuilt* |

Source: `LADDER-READY.json:families.k{2..6}` and `KILL-CHECK.json:rungs`.
Disjointness is **proved on rows**, not argued:
`np.unique(concat(corner_rows)).size == concat(corner_rows).size`, the same test
`exp_gen_factor_wells.assert_corners_disjoint_and_heldout` applies to rows 3–4
(`LADDER-READY.json:families.k3.disjointness_check` → `n_corner_rows: 8640`,
`pairwise_disjoint: true`; identically for k4, k5).
Every corner row is inside its rung's own hold-out —
`families.k{3,4,5}.n_corner_rows_in_holdout == 8640` and `all_corner_rows_held_out: true`.

⭐ **The rungs k=3,4,5,6 all carry exactly 120 corners of exactly 72 rows.** The paired
estimand's precision is therefore *identical* at four of the five rungs — only row 3's
existing end differs (60 corners of 36 rows), and that difference is row 3's, not w12's.

## 2. ⭐ The design decision that made this possible — cells ≠ corners

The ≥49 bar looked unreachable until the two layers were separated. A **cell** is what is
removed from the store's fit pool; a **corner** is a Montero-shaped probe box **inside** a
cell and is the unit of the estimand. Row 3 clears 49 with only **3 cells** because
`exp_gen_factor_wells.pair_subcorners` **tiles** each cell over the factors it leaves free
(3 cells × 20 tiles = 60). Row 4 is 49 cells at 1 corner each.

⇒ The ladder generalises the tiling: **pin k factors, tile the other 6 − k, same widths.**
Corners per cell fall (20 → 10 → 5 → 2 → 1) exactly as cells rise (3 → 12 → 24 → 60 → 120).

⛔ **If a future hub builds the middle from cells alone it will conclude the ladder is
impossible.** Capacity over pinned axes alone is 12 boxes at k=3 — a quarter of the bar.
Recorded in `chlu/data/gen_kway_ladder.py` §0 and in `LADDER-READY.json:kill_check`
(`n_cells_capacity` vs `corners_per_cell` vs `max_corners` are three separate fields for
this reason).

## 3. ⭐ BOTH ENDS REPRODUCE — asserted, not asserted-about

- **E1 (k=6 ≡ row 4).** 120 cells compared against `gen_value_holdout.build_nway_cells`:
  `cell_values_identical: true`, `order_identical: true`
  (`LADDER-READY.json:end_reproduction.E1_k6_vs_row4`). The shipped row 4 is the first
  **49** of those 120 and is checked separately in
  `tests/test_gen_kway_ladder.py::test_k6_reproduces_the_SHIPPED_49_nway_cells_specifically`.
- **E2 (k=2 ≡ row 3).** 60 corners compared against `exp_gen_factor_wells.pair_subcorners`:
  `corner_ids_identical: true`, `corner_row_sets_identical: true`
  (`…end_reproduction.E2_k2_vs_row3`). Corner ids are literally row 3's —
  `pair-000/f0-hh … pair-002/f8-ll` (`families.k2.corner_ids_head/tail`).
- **E2's reference side is tied to the runner's pins, not to this module's config.**
  `ROW3_TILE_WIDTHS = {hue_band 2, scale_corner 3, orientation_corner 3}` is re-checked
  against `exp_gen_factor_wells.PINS['pair_tile_*']` at emit time:
  `end_reproduction.E2_reference_widths_vs_PINS.status = PASS`.

⚠ **This last fix came out of a positive control, and it mattered.** The first E2 read the
tiling widths from its own config on *both* sides, so
`test_k2_end_reproduction_fires_when_the_tiling_widths_drift` **failed** — the check could
not fail. It now can. That test is in the suite.

## 4. ⛔ THREE INSTRUMENT FACTS THE HUB MUST HAVE BEFORE IT RUNS THIS

### 4.1 A strictly nested ladder through every k **does not exist** at this geometry
Rows 3 and 4 both **tie `wall_hue` to `floor_hue`** (`build_nway_cells` uses one `bg` for
both; `pair_subcorners` assigns one `band` to both). `floorwall` is therefore an **atomic
coordinate that pins two factors**, and from a k=4 set of four single factors the only unit
left to add lands on k=6. The pinned sets branch: `A2 ⊂ A3 ⊂ A4` and `A3 ⊂ A5 ⊂ A6`, but
**`A4 ⊄ A5`** (k=4 pins orientation and frees floor/wall; k=5 does the opposite).
`LADDER-READY.json:nesting_caveat`.

**The nested alternative is built and measured, not just named.** `LADDER_AXES_NESTED_ALT`
unties `wall_hue` at k=5 only: **120 cells → 600 pairwise-disjoint corners of 72 rows**,
hold-out 235 200 rows (`LADDER-READY.json:nested_alternative.k5`). ⛔ Its cost is on the
record in the same object: `geometry_matches_rows_3_4: false` — floor band ≠ wall band in
4 of every 5 tiles. **Hub's call; both numbers are in front of it.**

### 4.2 The ladder **cannot** be row-matched across k — this is arithmetic, not a preference
`LADDER-READY.json:row_matching_feasibility`:

| rung | min rows held out to reach 49 corners | max rows at full capacity |
|---|---|---|
| k=2 | 36 000 | 36 000 |
| k=3 | **45 000** | 108 000 |
| k=4 | 18 000 | 43 200 |
| k=5 | 9 000 | 21 600 |
| k=6 | 3 528 | **8 640** |

`required_floor = 45 000 > available_ceiling = 8 640` ⇒ **`feasible: false`**. There is no
common held-out-row budget at which every rung clears the bar. ⛔ A hub that tries to
equalise the hold-out will silently drop a rung below 49 corners.

### 4.3 …but the confound is **much smaller than it looks**, and here is the number
The fit-pool *fraction* varies 22.5 % (k=3, k-way cells only) to 1.8 % (k=6)
(`KILL-CHECK.json:rungs.*.held_out_fraction_full_sweep`). **But `exp_gen_factor_wells.fit_pool`
draws `min(PINS['n_pool'], train.size)` and every rung's train index clears
`n_pool = 131 072`** (`LADDER-READY.json:families.*.fit_pool_feasibility`):

| rung | train rows | pool drawn | headroom |
|---|---|---|---|
| k=2 | 288 000 | 131 072 | 156 928 |
| k=3 | **180 000** | 131 072 | **48 928** ← tightest |
| k=4 | 244 800 | 131 072 | 113 728 |
| k=5 | 266 400 | 131 072 | 135 328 |
| k=6 | 279 360 | 131 072 | 148 288 |

⇒ **The store is fit on the same 131 072 rows at every rung.** The confound does not act
through pool *size*; it acts through pool *coverage*. ⭐ The control that removes what
remains: run k=6 a second time with its fit pool restricted to the k=3 train index
(180 000 rows) — same depth, k=3's coverage. ⛔ w12 did not run it; it costs one scored run
and the choice is the hub's.

## 5. ⛔ NON-NEGOTIABLES — held, and tested

- `scale` and `orientation` are **never** VALUE cells. Asserted per rung in
  `tests/test_gen_kway_ladder.py::test_scale_and_orientation_never_become_a_VALUE_cell`;
  the module has no code path that could construct one. They **are** pinned inside the
  k-way box from k=3 up, exactly as in row 4.
  (`LADDER-READY.json:ineligibility`.)
- Every k-way box avoids the held-out hue `object_hue = 0.5`, the held-out shape
  `shape = 0.0`, and the three hues the `pair-*` cells reserve — the identical reservation
  rule `build_nway_cells` uses. Tested per rung
  (`test_kway_boxes_never_touch_a_held_out_or_pair_reserved_value`).
  Realised coordinate basis: `object_hue` 2 bands, `floorwall` 5 bands, `scale` 2 corners,
  `shape` 3, `orientation` 2 corners (`KILL-CHECK.json:coordinate_option_counts`).
- A rung swaps `nway-*` for `kway{k}-*` and **keeps the value and pair cells**, so rows
  1/2/5/7 stay measurable on the same split (`build_kway_condition_cells`; tested by
  `test_the_whole_rung_condition_is_disjoint_so_rows_1_2_5_7_survive`). Whole-condition
  hold-outs: k=3 300 000 rows (62.5 %), k=4 235 200 (49.0 %), k=5 213 600 (44.5 %)
  — the value cells alone are 156 000 of those and are inherited, not added by w12.
- The estimand is **inherited verbatim**: `exp_gen_factor_wells.paired_estimand`, unit =
  corner, metric `distinct_valid_novel_tuples`, margin `DECORATION_MARGIN_NOVELTY = +1.0`
  (accuracy rows `0.02`), bootstrap 10 000 @ `default_rng(20_260_919)`.
  ⛔ **No new metric, no new margin** (`LADDER-READY.json:estimand`).

## 6. Tests

`tests/test_gen_kway_ladder.py` — **35 passed** in 17.11 s
(`python -m pytest tests/test_gen_kway_ladder.py -q --no-cov`). Includes **four positive
controls** that prove the checks can fail: NOT-RUN reporting with a real count (32 corners
on a deliberately thin grid), width-drift breaking E2, duplicated corners tripping the
disjointness proof, and a truncated hold-out tripping the containment proof.

## 7. ⛔ DECLARED NOT-RUNs and one pre-existing failure

- **No scored run** — by charter. The ladder is run by `hub/composition-ladder`.
- **k=2 and k=6 were not *built*, they were *reproduced*.** Row 3's cells come from
  `build_pair_cells` verbatim; row 4's are checked against `build_nway_cells`. This spoke
  deliberately does not get a second opinion about an existing end.
- **The coverage-matched control of §4.3 is NOT-RUN** — it costs a scored run this spoke
  may not spend. Its design and its target (180 000-row fit pool at k=6) are on the record.
- **dSprites is NOT-RUN.** The builder is dataset-agnostic, but `PAIR_AXES` and the whole
  value hold-out are defined on Shapes3D's `object_hue`/`shape`; dSprites has no
  `object_hue`. No number was produced for it and none is implied.
- ⚠ **Pre-existing, NOT mine, NOT fixed:** 3 tests in `tests/test_gen_value_holdout.py` fail
  at commit `aa91785` on a clean tree —
  `test_rows_5_and_6_are_both_controlled_by_row_2`,
  `test_the_detector_fires_when_a_structural_giveaway_is_injected`,
  `test_the_grid_reach_diagnostic_separates_row_5_from_row_2`. Root cause:
  `reg["sets"]["row5_held_out"]["controls"]` is `'row1_seen'` in the shipped code (the
  AMENDMENT-1 re-pairing) while the test still expects `'row2_unseen'`, and
  `reg["row5_vs_row2"]` no longer exists. **`git diff --stat` is empty — w12 changed no
  tracked file.** Left for the owner of that module; flagging rather than silently
  patching another spoke's test.
  (82 passed, 3 failed, 5 skipped over
  `test_gen_value_holdout.py + test_gen_data_splits.py + test_gen_kway_ladder.py`.)

## 8. What the hub runs

```
python scripts/w12_kway_ladder.py --out DIR          # real 3dshapes.h5
python scripts/w12_kway_ladder.py --out DIR --toy    # offline grid, same arithmetic
```
then, per rung, `chlu.data.gen_kway_ladder.build_kway_split(grid, cfg, k)` →
`(split, cells, corners)` and feed `corners` to `exp_gen_factor_wells`'s existing
`per_corner_scores` / `paired_estimand` path unchanged. Each corner exposes `.rows(grid)`,
so it drops into `assert_corners_disjoint_and_heldout` as-is.
