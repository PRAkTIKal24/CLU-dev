# PREREG — gen-wide-split (written BEFORE the builder was run)

Author: experiment-engineer · Base `main @ fcdca43` · written 2026-09-10, before any wide split was built or measured.

## What is being predicted
The widened SECONDARY hold-out (`recomb2element_wide`) for Shapes3D. Predictions are derived by hand from the
Shapes3D factor grid (10 x 10 x 10 x 8 x 4 x 15 = 480,000) and the cell rule I am about to implement. ⛔ No code
producing these numbers has been executed yet.

## The rule I am about to implement (fixed here, not adjustable to hit a target)
Each held-out cell is **structurally the same kind as Montero's**: a conjunction over all six factors —
a 2-value band on each of `floor_hue`/`wall_hue`/`object_hue`, a 3-value corner of `scale`, exactly one `shape`,
a 3-value corner of `orientation`. Band width 2 and corner size 3 are *derived from Montero's own cell*
(|{hue>=0.8}| = 2, |{scale>=1.1}| = 3, |{orientation>20}| = 3), not chosen.

Background band `B_j = {f_j, f_j+1}` (circular on the 10-value hue ring, drawn from `default_rng(seed)`),
wall band = floor band (as in Montero's cell, where all three hues sit in one band),
object band `O_j = {f_j+s_j, f_j+s_j+1}` with **contrast step `s_j = j mod 6`**.

## P1 — realised row count per cell
**72 rows per cell, exactly** (2 x 2 x 2 x 3 x 1 x 3). Same size as Montero's cell by construction.

## P2 — pooled counts at the default `n_cells = 12`, `include_montero = True`
- additional cells: 12 x 72 = **864**
- pooled `n_test` = 864 + 72 (Montero's cell, carried as cell `montero`) = **936**
- `n_train` = 480,000 - 936 = **479,064**
- ⚠ These hold **only if no two cells overlap**. Overlap is rejected+redrawn by construction, so I predict
  **0 overlapping rows** and exactly the counts above. Any deviation is reported as realised, never patched.

## P3 — the contrast ladder (the point of the whole exercise)
Contrast label = mean over the cell's realised rows of `0.5*(d(oh,fh) + d(oh,wh))`, `d` = circular hue distance
(period 1.0, max 0.5). For a background band `{f,f+1}` and object band `{f+s,f+s+1}`, factor values are enumerated
uniformly, so the mean is `(1/4)[d(s)+d(s-1)+d(s+1)+d(s)]` in 0.1 grid steps:

| contrast step `s` | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **predicted label** | **0.05** | **0.10** | **0.20** | **0.30** | **0.40** | **0.45** |

- **Montero's cell is predicted to measure 0.05** — the bottom rung (all three hues in `{0.8, 0.9}`).
- The wide cells are predicted to **span 0.05 -> 0.45** against a theoretical max of 0.50, i.e. 90% of the range.
- ⭐ **Falsifier for acceptance item 2:** if the measured labels of the additional cells cluster within
  +/-0.05 of Montero's 0.05, the split adds rows but no contrast variation and the spoke has failed its design goal.

## P4 — the absence assertion
- `assert_holdout_disjoint` passes over all **479,064** train rows of the wide split (n_checked == n_train, not 0).
- **Positive control FIRES:** injecting one known wide-held-out row into the train index raises `HoldoutViolation`
  naming that row. ⭐ This is the criterion the task lives on.

## P5 — Montero's condition is bit-identical after the change
- `n_test == 72`, `n_train == 479,928`
- `factor_index.npz` sha256 reproduces as `5f1fce25c40cceae6de2d00fd2145c4cdc73dcc38cee7f6545cbcaea3ce78f49`
- `c_invalid.json` sha256 reproduces as `b886da239e97b19e9001a88ce04fded329dda4884d301f78c3097ea6cc66ba49`
  (both measured from the shipped artifacts at base `fcdca43`, before my edit).

## P6 — |C| for the wide condition
The existing derivation `n_per_kind = ceil(n_test/2)` gives `ceil(936/2) = 468` per kind => **|C| = 936 = |B|**.
