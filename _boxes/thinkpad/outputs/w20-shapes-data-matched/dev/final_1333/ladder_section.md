**Caption:** every rung's store AND data-matched rivals fit on the rung's own **131,072**-row pool (M-16k stores on its first **16,384 / 32,768** rows); the draft's ladder rivals = spoke A `@ref` (118k params, **32,256** Montero rows); `learned_ref_ownpool` = **32,768** rung rows, 2,560 steps; LACE/PoE = w15 at **32,256** rung rows (131k NOT RUN, §7). Seeds = 3 on every cell. Rungs n = 3–6 probe the IDENTICAL 120 corners (a training-data axis, not depth); ceiling 36 (n=2) / 72 (n≥3). mlp and cont_mlp carried at TENX on every rung/seed (PREREG §4: they failed P1 on rows 3–4). Source `ladder.<n>` in `SHAPES-MATCHED.json`.

| rung | store K\*=32 (131k) | K\*=32 @16,384 | K\*=32 @32,768 | strongest data-matched rival (carried) | **K\* − strongest, per seed [mean]** | draft-style: K\* − strongest spoke-A `@ref` | `learned_ref_ownpool` | LACE / PoE (w15, 32k) | table FLOOR | n |
|---|---|---|---|---|---|---|---|---|---|---|
| n=2 | 32.64 | 32.08 | 32.15 | mlp tenx (27.51) | +4.63 / +6.92 / +3.85 [**+5.13**] | +12.38 / +10.50 / +9.38 | 7.48 | 0.061 / 0.006 | 5.17 | 3 |
| n=3 | 61.61 | 62.15 | 61.97 | mlp tenx (50.43) | +12.89 / +15.95 / +4.71 [**+11.18**] | +15.72 / +21.22 / +9.92 | 11.79 | 0.156 / 0.000 | 9.53 | 3 |
| n=4 | 66.09 | 65.64 | 65.50 | mlp tenx (53.54) | +14.72 / +15.80 / +7.12 [**+12.55**] | +20.81 / +24.25 / +15.21 | 13.28 | 0.172 / 0.006 | 9.53 | 3 |
| n=5 | 66.58 | 66.26 | 66.57 | mlp tenx (55.89) | +12.00 / +13.54 / +6.52 [**+10.69**] | +21.21 / +25.25 / +15.29 | 14.96 | 0.136 / 0.003 | 9.53 | 3 |
| n=6 | 66.60 | 67.01 | 67.05 | mlp tenx (56.30) | +11.72 / +11.56 / +7.62 [**+10.30**] | +21.48 / +24.68 / +15.66 | 15.61 | 0.133 / 0.003 | 9.53 | 3 |

All K\* legs: CI95 of K\* − strongest data-matched rival lies above 0 on every seed and rung (lowest lower bound +2.46, n=2 seed 2); the smallest seed-level margin is +3.85.
