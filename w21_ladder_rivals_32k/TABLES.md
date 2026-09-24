# w21 spoke/ladder-rivals-32k — TABLES

Every cell is read from `LADDER-RIVALS-32K.json` (key path in each section header); that JSON names the per-corner score file or fit artefact behind every number. FREE REPLICATION: no ledger row, 0 scored runs. Seeds 0/1/2 in every cell.

## Table L3 — the ladder with THREE rival column sets

**Caption.** **Ladder rivals, three column sets — pool AND optimiser steps named for each.** Store: K\*=32, content fitted on the rung's first **32,768** rows (`n_store_train`, read from w12b `kstar_ladder/*/LEG.json` `n_fit_rows` and `partitions.json`), from a 131,072-row rung pool. **Set A** (the draft's ladder rivals, spoke-A `@ref`): **32,256** Montero combination-split rows (⚠ this pool holds the value-holdout's held-out values), 117,878–119,858 params, **5,040** steps. **Set B** (w20 data-matched): the rung's **131,072**-row pool, 130,433–131,891 params; DECLARED **20,480** steps, TENX **204,800** steps (mlp and cont_mlp carried at TENX, lstm and transformer at DECLARED). **Set C** (w21, this table): the store's own **32,768** content rows, the same four families at the same read-path float match (mlp 130,514 / lstm 130,721 / transformer 130,433 / cont_mlp 131,891 params, as set B), spoke-A selection re-run on these rows (lr 3e-3, 40 epochs on all 60 legs); DECLARED **5,120** steps, TENX **51,200** steps, and a step-matched column at set B's carried budget (mlp and cont_mlp **204,800**, lstm and transformer **20,480**). `learned_ref_ownpool`: the same **32,768** rows, 130,514 params, **2,560** steps. Margin = store − the strongest rival per seed (max over families of max(DECLARED, TENX); w20's convention), paired per corner, bootstrap CI95. Seeds = 3 in every cell. Rungs n = 3–6 probe the same 120 corners (ceiling 72); n = 2 has 60 corners (ceiling 36).

Key: `rungs.<n>.margins_K32_store_minus_rival.<set>.per_seed.<s>.delta` (CI95 in the JSON), `rungs.<n>.store.summary`, `rungs.<n>.crosscheck.X1_same_rows_same_steps.ref`.

| rung | store K\*=32 (content 32,768 rows) | **A**: K\* − strongest spoke-A `@ref` (32,256 rows · 5,040 steps) s0/s1/s2 [mean] | **B**: K\* − strongest w20 (131,072 rows · 20,480 / 204,800 steps) s0/s1/s2 [mean] | **C**: K\* − strongest w21 (32,768 rows · 5,120 / 51,200 steps) s0/s1/s2 [mean] | **C-sm**: K\* − strongest w21 step-matched to B (32,768 rows · 20,480 / 204,800 steps) [mean] | `learned_ref_ownpool` (32,768 rows · 2,560 steps) | n |
|---|---|---|---|---|---|---|---|
| n=2 | 32.64 | +12.38 / +10.50 / +9.38 [**+10.76**] | +4.63 / +6.92 / +3.85 [**+5.13**] | +6.65 / +7.37 / +4.22 [**+6.08**] | +7.57 / +6.80 / +3.25 [**+5.87**] | 7.48 | 3 |
| n=3 | 61.61 | +15.72 / +21.22 / +9.92 [**+15.62**] | +12.89 / +15.95 / +4.71 [**+11.18**] | +13.01 / +15.37 / +6.07 [**+11.48**] | +14.35 / +17.79 / +8.07 [**+13.40**] | 11.79 | 3 |
| n=4 | 66.09 | +20.81 / +24.25 / +15.21 [**+20.09**] | +14.72 / +15.80 / +7.12 [**+12.55**] | +16.14 / +16.73 / +9.13 [**+14.00**] | +20.15 / +15.02 / +8.44 [**+14.54**] | 13.28 | 3 |
| n=5 | 66.58 | +21.21 / +25.25 / +15.29 [**+20.58**] | +12.00 / +13.54 / +6.52 [**+10.69**] | +11.12 / +10.60 / +6.69 [**+9.47**] | +11.20 / +15.36 / +5.30 [**+10.62**] | 14.96 | 3 |
| n=6 | 66.60 | +21.48 / +24.68 / +15.66 [**+20.61**] | +11.72 / +11.56 / +7.62 [**+10.30**] | +11.30 / +12.49 / +7.29 [**+10.36**] | +10.88 / +11.75 / +5.27 [**+9.30**] | 15.61 | 3 |

Strongest rival per seed (the label the margin was taken against):

| rung | A | B | C | C-sm |
|---|---|---|---|---|
| n=2 | spokeA_lstm (19.42) / spokeA_mlp (22.00) / spokeA_transformer (24.25) | mlp__tenx (27.17) / mlp__tenx (25.58) / mlp__tenx (29.78) | mlp__tenx (25.15) / mlp__declared (25.13) / mlp__tenx (29.42) | mlp__steps204800 (24.23) / mlp__steps204800 (25.70) / mlp__steps204800 (30.38) |
| n=3 | spokeA_lstm (44.43) / spokeA_mlp (41.57) / spokeA_transformer (51.99) | mlp__tenx (47.26) / mlp__tenx (46.83) / mlp__tenx (57.20) | mlp__tenx (47.14) / mlp__tenx (47.42) / mlp__tenx (55.84) | mlp__steps204800 (45.80) / mlp__steps204800 (44.99) / mlp__steps204800 (53.84) |
| n=4 | spokeA_lstm (44.43) / spokeA_mlp (41.57) / spokeA_transformer (51.99) | mlp__tenx (50.52) / mlp__tenx (50.02) / mlp__tenx (60.08) | mlp__tenx (49.10) / mlp__tenx (49.09) / mlp__tenx (58.07) | mlp__steps204800 (45.09) / mlp__steps204800 (50.80) / mlp__steps204800 (58.76) |
| n=5 | spokeA_lstm (44.43) / spokeA_mlp (41.57) / spokeA_transformer (51.99) | mlp__tenx (53.64) / mlp__tenx (53.27) / mlp__tenx (60.77) | mlp__tenx (54.52) / mlp__tenx (56.22) / mlp__tenx (60.59) | mlp__steps204800 (54.44) / mlp__steps204800 (51.46) / mlp__steps204800 (61.98) |
| n=6 | spokeA_lstm (44.43) / spokeA_mlp (41.57) / spokeA_transformer (51.99) | mlp__tenx (54.18) / mlp__declared (54.69) / mlp__tenx (60.03) | mlp__tenx (54.61) / mlp__tenx (53.76) / mlp__tenx (60.36) | mlp__steps204800 (55.02) / mlp__steps204800 (54.50) / mlp__steps204800 (62.38) |

Set C, rival's best case (max over every set-C fit per seed; `setC_vs_best_of_every_fit`): n=2 +6.65 / +6.80 / +3.25 [+5.57]; n=3 +13.01 / +15.37 / +6.07 [+11.48]; n=4 +16.14 / +15.02 / +8.44 [+13.20]; n=5 +11.12 / +10.60 / +5.30 [+9.01]; n=6 +10.88 / +11.75 / +5.27 [+9.30].

Integrity: set B recomputed from w20's score files equals w20's printed ladder P3 — n=2 True, n=3 True, n=4 True, n=5 True, n=6 True (`setB_integrity_vs_w20_printed`).

## Table F — every family, every set, per rung (3-seed mean; per-seed values in the JSON)

Keys: `rungs.<n>.setA_spokeA_32256.<fam>.summary`, `rungs.<n>.setB_w20_131072.<fam>.<stage>.summary`, `rungs.<n>.setC_w21_32768.<fam>.<stage>.summary`. Column headers give POOL · STEPS.

| rung | family | A: 32,256 · 5,040 | B DECLARED: 131,072 · 20,480 | B TENX: 131,072 · 204,800 | **C DECLARED: 32,768 · 5,120** | **C TENX: 32,768 · 51,200** | **C step-matched: 32,768 · (fam)** |
|---|---|---|---|---|---|---|---|
| n=2 | mlp | 20.78 | 25.64 | 27.51 | 24.37 | 26.46 | 26.77 · 204,800 |
| n=2 | lstm | 20.83 | 23.36 | 21.85 (1 seed) | 23.21 | 24.02 | 23.54 · 20,480 |
| n=2 | transformer | 17.59 | 19.78 | 16.22 (1 seed) | 18.84 | 19.10 | 19.70 · 20,480 |
| n=2 | cont_mlp | 4.99 | 11.09 | 22.72 | 6.74 | 18.86 | 21.99 · 204,800 |
| n=3 | mlp | 42.18 | 47.30 | 50.43 | 41.65 | 50.13 | 48.21 · 204,800 |
| n=3 | lstm | 43.50 | 38.18 | NOT RUN | 36.64 | 39.81 | 40.48 · 20,480 |
| n=3 | transformer | 38.23 | 27.28 | NOT RUN | 27.30 | 29.04 | 30.73 · 20,480 |
| n=3 | cont_mlp | 9.37 | 22.13 | 35.08 | 10.80 | 30.65 | 38.93 · 204,800 |
| n=4 | mlp | 42.18 | 50.45 | 53.54 | 44.55 | 52.09 | 51.55 · 204,800 |
| n=4 | lstm | 43.50 | 45.32 | NOT RUN | 44.09 | 45.91 | 45.87 · 20,480 |
| n=4 | transformer | 38.23 | 38.10 | NOT RUN | 36.83 | 36.87 | 37.35 · 20,480 |
| n=4 | cont_mlp | 9.37 | 27.27 | 39.97 | 11.01 | 31.46 | 36.43 · 204,800 |
| n=5 | mlp | 42.18 | 54.29 | 55.89 | 48.40 | 57.11 | 55.96 · 204,800 |
| n=5 | lstm | 43.50 | 49.88 | NOT RUN | 47.64 | 48.89 | 49.39 · 20,480 |
| n=5 | transformer | 38.23 | 42.36 | NOT RUN | 42.66 | 41.62 | 41.93 · 20,480 |
| n=5 | cont_mlp | 9.37 | 30.65 | 48.10 | 16.94 | 39.15 | 48.81 · 204,800 |
| n=6 | mlp | 42.18 | 54.36 | 55.47 | 49.60 | 56.24 | 57.30 · 204,800 |
| n=6 | lstm | 43.50 | 49.82 | 45.42 (1 seed) | 48.14 | 49.96 | 50.38 · 20,480 |
| n=6 | transformer | 38.23 | 42.28 | 36.15 (1 seed) | 42.48 | 41.35 | 42.23 · 20,480 |
| n=6 | cont_mlp | 9.37 | 29.74 | 49.19 | 15.36 | 37.70 | 47.65 · 204,800 |

B TENX was run on every rung/seed only for mlp and cont_mlp; for lstm/transformer only as w20's spot check (rungs 2 and 6, seed 0) — shown with its seed count.

## Table P1 — convergence of set C (TENX − DECLARED, 10× the steps; bar 1.0)

Key: `rungs.<n>.setC_w21_32768.<fam>.P1.<s>`. Last column: step-matched − TENX.

| rung | family | DECLARED s0/s1/s2 | TENX s0/s1/s2 | TENX − DECLARED | P1 | step-matched − TENX |
|---|---|---|---|---|---|---|
| n=2 | mlp | 22.07 / 25.13 / 25.90 | 25.15 / 24.82 / 29.42 | +3.08 / -0.32 / +3.52 | FAIL / PASS / FAIL | -0.92 / +0.88 / +0.97 |
| n=2 | lstm | 22.42 / 20.67 / 26.55 | 22.58 / 22.42 / 27.05 | +0.17 / +1.75 / +0.50 | PASS / FAIL / PASS | -0.82 / +0.30 / -0.90 |
| n=2 | transformer | 13.87 / 17.50 / 25.17 | 16.85 / 16.58 / 23.87 | +2.98 / -0.92 / -1.30 | FAIL / PASS / PASS | -0.90 / +0.47 / +2.23 |
| n=2 | cont_mlp | 3.95 / 6.23 / 10.03 | 16.88 / 17.38 / 22.32 | +12.93 / +11.15 / +12.28 | FAIL / FAIL / FAIL | +0.48 / +4.55 / +4.37 |
| n=3 | mlp | 35.79 / 41.36 / 47.81 | 47.14 / 47.42 / 55.84 | +11.35 / +6.06 / +8.03 | FAIL / FAIL / FAIL | -1.34 / -2.42 / -2.00 |
| n=3 | lstm | 30.52 / 35.98 / 43.42 | 36.39 / 37.67 / 45.37 | +5.88 / +1.69 / +1.95 | FAIL / FAIL / FAIL | +0.91 / +3.31 / -2.22 |
| n=3 | transformer | 27.98 / 24.17 / 29.76 | 26.02 / 25.63 / 35.48 | -1.96 / +1.47 / +5.72 | PASS / FAIL / FAIL | +0.93 / +1.61 / +2.54 |
| n=3 | cont_mlp | 6.88 / 11.33 / 14.18 | 24.69 / 31.81 / 35.46 | +17.81 / +20.48 / +21.28 | FAIL / FAIL / FAIL | +11.78 / +11.76 / +1.28 |
| n=4 | mlp | 39.32 / 43.39 / 50.93 | 49.10 / 49.09 / 58.07 | +9.78 / +5.70 / +7.13 | FAIL / FAIL / FAIL | -4.01 / +1.71 / +0.69 |
| n=4 | lstm | 41.54 / 39.98 / 50.76 | 42.68 / 41.33 / 53.71 | +1.14 / +1.34 / +2.95 | FAIL / FAIL / FAIL | +1.11 / -0.07 / -1.16 |
| n=4 | transformer | 32.04 / 31.27 / 47.18 | 30.60 / 32.56 / 47.44 | -1.44 / +1.28 / +0.26 | PASS / FAIL / PASS | -0.03 / +0.40 / +1.07 |
| n=4 | cont_mlp | 6.10 / 8.65 / 18.28 | 20.59 / 27.65 / 46.14 | +14.49 / +19.00 / +27.86 | FAIL / FAIL / FAIL | +7.84 / +8.32 / -1.25 |
| n=5 | mlp | 43.64 / 47.91 / 53.64 | 54.52 / 56.22 / 60.59 | +10.88 / +8.31 / +6.95 | FAIL / FAIL / FAIL | -0.07 / -4.76 / +1.39 |
| n=5 | lstm | 46.10 / 44.67 / 52.17 | 45.95 / 45.62 / 55.12 | -0.15 / +0.95 / +2.95 | PASS / PASS / FAIL | +1.93 / -0.45 / +0.00 |
| n=5 | transformer | 35.08 / 39.42 / 53.47 | 36.40 / 36.44 / 52.02 | +1.32 / -2.98 / -1.45 | FAIL / PASS / PASS | -0.13 / +0.90 / +0.17 |
| n=5 | cont_mlp | 9.78 / 17.45 / 23.59 | 38.71 / 32.98 / 45.76 | +28.93 / +15.53 / +22.17 | FAIL / FAIL / FAIL | +0.12 / +18.33 / +10.54 |
| n=6 | mlp | 45.77 / 48.77 / 54.26 | 54.61 / 53.76 / 60.36 | +8.83 / +4.99 / +6.10 | FAIL / FAIL / FAIL | +0.42 / +0.74 / +2.02 |
| n=6 | lstm | 45.64 / 43.67 / 55.12 | 48.17 / 46.34 / 55.36 | +2.52 / +2.67 / +0.24 | FAIL / FAIL / PASS | -0.15 / +0.35 / +1.07 |
| n=6 | transformer | 36.87 / 37.83 / 52.74 | 35.44 / 36.19 / 52.42 | -1.42 / -1.64 / -0.33 | PASS / PASS / PASS | +0.92 / +1.30 / +0.43 |
| n=6 | cont_mlp | 9.15 / 13.01 / 23.92 | 33.28 / 34.03 / 45.79 | +24.13 / +21.02 / +21.87 | FAIL / FAIL / FAIL | +3.84 / +19.33 / +6.68 |

## Table X1 — THE CROSS-CHECK: same rows, same steps (`crosscheck.X1_same_rows_same_steps`)

`learned_ref_ownpool` (EXISTING, w12 composition-ladder AMENDMENT 1) vs this spoke's `mlp__ownpoolsched`: both mlp, width 61, 130,514 params, the rung's first 32,768 rows, 20 epochs, lr 1e-3, wd 0, batch 256 = 2,560 steps. Pre-declared bar (LOG 13:58Z): |Δ of 3-seed means| ≤ the larger across-seed SD.

| rung | ref s0/s1/s2 [mean ± sd] | mine s0/s1/s2 [mean ± sd] | Δ mine − ref | bar (max sd) | per-seed paired Δ [CI95] | verdict |
|---|---|---|---|---|---|---|
| n=2 | 5.57 / 7.58 / 9.30 [7.48 ± 1.87] | 7.37 / 5.43 / 10.83 [7.88 ± 2.74] | +0.39 | 2.74 | +1.80 [+0.67, +2.92]; -2.15 [-3.80, -0.58]; +1.53 [+0.17, +2.95] | REPRODUCES within seed spread |
| n=3 | 8.22 / 10.91 / 16.26 [11.79 ± 4.09] | 8.62 / 9.53 / 14.72 [10.96 ± 3.29] | -0.84 | 4.09 | +0.41 [-0.93, +1.73]; -1.38 [-2.64, -0.16]; -1.54 [-3.06, +0.03] | REPRODUCES within seed spread |
| n=4 | 8.54 / 12.53 / 18.77 [13.28 ± 5.15] | 11.88 / 11.88 / 22.29 [15.35 ± 6.01] | +2.07 | 6.01 | +3.34 [+1.95, +4.73]; -0.65 [-2.26, +0.97]; +3.52 [+1.58, +5.41] | REPRODUCES within seed spread |
| n=5 | 9.87 / 15.31 / 19.71 [14.96 ± 4.93] | 13.85 / 13.38 / 23.84 [17.03 ± 5.91] | +2.06 | 5.91 | +3.98 [+2.48, +5.49]; -1.93 [-3.70, -0.15]; +4.13 [+2.18, +6.04] | REPRODUCES within seed spread |
| n=6 | 10.67 / 15.12 / 21.03 [15.61 ± 5.20] | 14.53 / 13.04 / 23.88 [17.15 ± 5.87] | +1.54 | 5.87 | +3.86 [+2.22, +5.47]; -2.08 [-3.85, -0.28]; +2.84 [+0.84, +4.78] | REPRODUCES within seed spread |

Diagnostic, NOT the verdict (`crosscheck.X1_diagnostic_ref_recipe_replay`): `learned_ref_ownpool`'s own recipe replayed with its own PRNG keys and trainer (`scripts/w21_x1_replay.py`), compared per corner to the EXISTING rows file:

| rung | replay s0/s1/s2 | ref s0/s1/s2 | corners differing s0/s1/s2 | identical per corner |
|---|---|---|---|---|
| n=2 | 5.57 / 7.58 / 9.30 | 5.57 / 7.58 / 9.30 | 0/60 / 0/60 / 0/60 | True / True / True |
| n=3 | 8.22 / 10.91 / 16.26 | 8.22 / 10.91 / 16.26 | 0/120 / 0/120 / 0/120 | True / True / True |
| n=4 | 8.54 / 12.53 / 18.77 | 8.54 / 12.53 / 18.77 | 0/120 / 0/120 / 0/120 | True / True / True |
| n=5 | 9.87 / 15.31 / 19.71 | 9.87 / 15.31 / 19.71 | 0/120 / 0/120 / 0/120 | True / True / True |
| n=6 | 10.67 / 15.12 / 21.03 | 10.67 / 15.12 / 21.03 | 0/120 / 0/120 / 0/120 | True / True / True |

## Table X2 — same rows, DIFFERENT steps (expected to differ: the measured steps effect)

Key: `crosscheck.X2_same_rows_different_steps`. All columns: mlp, the same 32,768 rows.

| rung | `learned_ref_ownpool` · 2,560 steps | mine `ownpoolsched` · 2,560 | mine DECLARED · 5,120 | mine TENX · 51,200 | mine step-matched · 204,800 | w20 mlp TENX · 131,072 rows · 204,800 |
|---|---|---|---|---|---|---|
| n=2 | 7.48 | 7.88 | 24.37 | 26.46 | 26.77 | 27.51 |
| n=3 | 11.79 | 10.96 | 41.65 | 50.13 | 48.21 | 50.43 |
| n=4 | 13.28 | 15.35 | 44.55 | 52.09 | 51.55 | 53.54 |
| n=5 | 14.96 | 17.03 | 48.40 | 57.11 | 55.96 | 55.89 |
| n=6 | 15.61 | 17.15 | 49.60 | 56.24 | 57.30 | 55.47 |

