# w12 hub/composition-ladder — THE LADDER TABLE

> ⛔ **PLACEMENT — APPENDIX -- the K-sweep's ENDPOINT at K = 5. NOT main text: the main text carries ONE K, K*, and never both (Head ruling, 2026-09-21T12:38:07Z, HEAD-RULING-K.md).**

> The store arms run at **K = 5, DECLARED** (`PREREG.md` §3): K\* was unpublished when this hub started and is unpublished still — `CROSSOVER-val.json: K_star = null`. The Head re-runs the store arm at K\* as a free replication; ⛔ this hub does not wait for it and ⛔ no row here is a K\* row.

Every number below is `distinct_valid_novel_tuples` per corner, the metric and the margin the harness pre-declared (`LADDER-READY.json:estimand`). Unit = **corner**. Cells read `mean over seeds (seed0 / seed1 / seed2)`.

⛔ **Seed count and configuration count are separate numbers and are never merged.** `n_seeds = 3` (seeds [0, 1, 2]). Configurations per rung = the rung's corner count, printed in the header of each column.

## 1. The ladder — score per corner

| arm | k=2<br><sub>60 corners · ceiling 36</sub> | k=3<br><sub>120 corners · ceiling 72</sub> | k=4<br><sub>120 corners · ceiling 72</sub> | k=5<br><sub>120 corners · ceiling 72</sub> | k=6<br><sub>120 corners · ceiling 72</sub> | shipped<br><sub>109 corners (row3+row4)</sub> |
|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | 30.539 (28.633 / 31.650 / 31.333) | 59.750 (56.950 / 59.425 / 62.875) | 62.708 (60.933 / 62.233 / 64.958) | 64.719 (63.242 / 64.508 / 66.408) | 64.556 (62.692 / 63.992 / 66.983) | 46.333 (44.954 / 46.321 / 47.725) |
| **pca_tensor** (wave-5 subspace store) | 30.539 (28.633 / 31.650 / 31.333) | 59.750 (56.950 / 59.425 / 62.875) | 62.708 (60.933 / 62.233 / 64.958) | 64.719 (63.242 / 64.508 / 66.408) | 64.556 (62.692 / 63.992 / 66.983) | 46.333 (44.954 / 46.321 / 47.725) |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) | 0.003 (0.000 / 0.000 / 0.008) | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | 0.361 (0.250 / 0.083 / 0.750) | 0.464 (0.567 / 0.175 / 0.650) | 0.636 (0.483 / 0.183 / 1.242) | 1.100 (0.683 / 0.583 / 2.033) | 0.303 (0.258 / 0.158 / 0.492) | 0.578 (0.550 / 0.119 / 1.064) |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) | 0.000 (0.000 / 0.000 / 0.000) |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | 0.361 (0.250 / 0.083 / 0.750) | 0.464 (0.567 / 0.175 / 0.650) | 0.636 (0.483 / 0.183 / 1.242) | 1.100 (0.683 / 0.583 / 2.033) | 0.303 (0.258 / 0.158 / 0.492) | 0.578 (0.550 / 0.119 / 1.064) |
| MLP @ 524,288 B ref | 20.783 (17.017 / 22.000 / 23.333) | 42.175 (35.208 / 41.567 / 49.750) | 42.175 (35.208 / 41.567 / 49.750) | 42.175 (35.208 / 41.567 / 49.750) | 42.175 (35.208 / 41.567 / 49.750) | 30.416 (26.110 / 28.578 / 36.560) |
| LSTM @ 524,288 B ref | 20.833 (19.417 / 19.333 / 23.750) | 43.503 (44.433 / 39.400 / 46.675) | 43.503 (44.433 / 39.400 / 46.675) | 43.503 (44.433 / 39.400 / 46.675) | 43.503 (44.433 / 39.400 / 46.675) | 31.725 (30.028 / 28.917 / 36.229) |
| transformer @ 524,288 B ref | 17.594 (12.750 / 15.783 / 24.250) | 38.231 (28.800 / 33.900 / 51.992) | 38.231 (28.800 / 33.900 / 51.992) | 38.231 (28.800 / 33.900 / 51.992) | 38.231 (28.800 / 33.900 / 51.992) | 27.737 (20.716 / 24.982 / 37.514) |
| continuous MLP @ 524,288 B ref | 4.994 (3.883 / 4.567 / 6.533) | 9.372 (6.333 / 9.117 / 12.667) | 9.372 (6.333 / 9.117 / 12.667) | 9.372 (6.333 / 9.117 / 12.667) | 9.372 (6.333 / 9.117 / 12.667) | 8.012 (6.009 / 6.193 / 11.835) |
| table FLOOR @ 524,288 B ref | 5.167 (5.833 / 3.483 / 6.183) | 9.528 (8.433 / 7.917 / 12.233) | 9.528 (8.433 / 7.917 / 12.233) | 9.528 (8.433 / 7.917 / 12.233) | 9.528 (8.433 / 7.917 / 12.233) | 7.685 (8.229 / 5.073 / 9.752) |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | 7.483 (5.567 / 7.583 / 9.300) | 11.794 (8.217 / 10.908 / 16.258) | 13.281 (8.542 / 12.533 / 18.767) | 14.961 (9.867 / 15.308 / 19.708) | 15.606 (10.667 / 15.117 / 21.033) | 11.810 (9.183 / 11.119 / 15.128) |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | 33.322 (33.200 / 33.867 / 32.900) | 55.508 (52.133 / 59.458 / 54.933) | 67.992 (67.175 / 68.583 / 68.217) | 68.619 (68.050 / 68.758 / 69.050) | 68.617 (68.467 / 68.558 / 68.825) | 49.294 (48.899 / 49.541 / 49.440) |

⚠ **The rungs are NOT on one absolute scale.** The metric's ceiling is the corner's own row count: 36 at k=2 and 72 at k≥3. A k=2 number and a k=3 number are not comparable as absolutes; the ceiling is printed on every column for that reason.

## 2. rank / width on every store row (wave 9's saturation trap)

| arm | k=2 | k=3 | k=4 | k=5 | k=6 | shipped |
|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | 0.695 / 0.889 / 0.865 | 0.689 / 0.877 / 0.865 | 0.696 / 0.879 / 0.922 | 0.695 / 0.901 / 0.865 | 0.695 / 0.889 / 0.889 | 0.690 / 0.889 / 0.889 |
| **pca_tensor** (wave-5 subspace store) | 1.000 / 0.889 / 0.926 | 1.000 / 0.877 / 0.913 | 1.000 / 0.955 / 0.984 | 1.000 / 0.901 / 0.955 | 1.000 / 0.889 / 0.984 | 1.000 / 0.889 / 0.969 |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | 1.000 / 0.889 / 0.941 | 1.000 / 0.877 / 0.927 | 1.000 / 0.970 / 1.000 | 1.000 / 0.901 / 0.970 | 1.000 / 0.889 / 1.000 | 1.000 / 0.889 / 0.985 |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | 1.000 / 0.889 / 0.941 | 1.000 / 0.877 / 0.927 | 1.000 / 0.970 / 1.000 | 1.000 / 0.901 / 0.970 | 1.000 / 0.889 / 1.000 | 1.000 / 0.889 / 0.985 |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | 1.000 / 0.889 / 0.941 | 1.000 / 0.877 / 0.927 | 1.000 / 0.970 / 1.000 | 1.000 / 0.901 / 0.970 | 1.000 / 0.889 / 1.000 | 1.000 / 0.889 / 0.985 |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | 1.000 / 0.889 / 0.941 | 1.000 / 0.877 / 0.927 | 1.000 / 0.970 / 1.000 | 1.000 / 0.901 / 0.970 | 1.000 / 0.889 / 1.000 | 1.000 / 0.889 / 0.985 |

> wave 9 measured its maze arm DECORATIVE at `rank/d_k = 0.938`. Any cell at or above that is in the saturation regime and is flagged in FINDINGS.

## 3. ⛔ THE CONTROLS — the store is DECORATIVE at any rung where they do not collapse

| store arm | control | k=2 | k=3 | k=4 | k=5 | k=6 | shipped |
|---|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | `landscape_shuffled` | collapses<br><sub>Δ +28.63 / +31.65 / +31.33</sub> | collapses<br><sub>Δ +56.95 / +59.42 / +62.88</sub> | collapses<br><sub>Δ +60.93 / +62.23 / +64.95</sub> | collapses<br><sub>Δ +63.24 / +64.51 / +66.41</sub> | collapses<br><sub>Δ +62.69 / +63.99 / +66.98</sub> | collapses<br><sub>Δ +44.95 / +46.32 / +47.72</sub> |
| **single design** (K=5, DECLARED) | `random_atoms` | collapses<br><sub>Δ +28.38 / +31.57 / +30.58</sub> | collapses<br><sub>Δ +56.38 / +59.25 / +62.23</sub> | collapses<br><sub>Δ +60.45 / +62.05 / +63.72</sub> | collapses<br><sub>Δ +62.56 / +63.92 / +64.38</sub> | collapses<br><sub>Δ +62.43 / +63.83 / +66.49</sub> | collapses<br><sub>Δ +44.40 / +46.20 / +46.66</sub> |
| **pca_tensor** (wave-5 subspace store) | `landscape_shuffled` | collapses<br><sub>Δ +28.63 / +31.65 / +31.33</sub> | collapses<br><sub>Δ +56.95 / +59.42 / +62.88</sub> | collapses<br><sub>Δ +60.93 / +62.23 / +64.96</sub> | collapses<br><sub>Δ +63.24 / +64.51 / +66.41</sub> | collapses<br><sub>Δ +62.69 / +63.99 / +66.98</sub> | collapses<br><sub>Δ +44.95 / +46.32 / +47.72</sub> |
| **pca_tensor** (wave-5 subspace store) | `random_atoms` | collapses<br><sub>Δ +28.38 / +31.57 / +30.58</sub> | collapses<br><sub>Δ +56.38 / +59.25 / +62.23</sub> | collapses<br><sub>Δ +60.45 / +62.05 / +63.72</sub> | collapses<br><sub>Δ +62.56 / +63.92 / +64.38</sub> | collapses<br><sub>Δ +62.43 / +63.83 / +66.49</sub> | collapses<br><sub>Δ +44.40 / +46.20 / +46.66</sub> |

| store arm | k=2 | k=3 | k=4 | k=5 | k=6 | shipped |
|---|---|---|---|---|---|---|
| **single design** (K=5, DECLARED) | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung |
| **pca_tensor** (wave-5 subspace store) | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung | the store survives both charter-C.2 controls at this rung |

## 4. Paired margin against the BEST learned baseline

The rule was fixed at 2026-09-21T11:38:41Z (PREREG §5) and **widened** at 2026-09-21T11:49:46Z (AMENDMENT-3) so the opponent can only get STRONGER: per (rung, seed), the learned arm with the **largest** mean over that rung's corners, among all seven learned arms. Both readings are printed — the narrow five-rival reading is in the column marked `§5`.

| rung | best learned (all 7) — s0 / s1 / s2 | best learned (§5, 5 rivals) | Δ single design (K=5, DECLARED) | Δ pca_tensor (wave-5 subspace store) |
|---|---|---|---|---|
| k=2 | learned_1x (33.200) / learned_1x (33.867) / learned_1x (32.900) | rival_lstm (19.417) / rival_mlp (22.000) / rival_transformer (24.250) | -4.57 [-6.38, -2.88] / -2.22 [-3.37, -1.20] / -1.57 [-2.72, -0.50] | -4.57 [-6.38, -2.88] / -2.22 [-3.37, -1.20] / -1.57 [-2.72, -0.50] |
| k=3 | learned_1x (52.133) / learned_1x (59.458) / learned_1x (54.933) | rival_lstm (44.433) / rival_mlp (41.567) / rival_transformer (51.992) | +4.82 [+2.37, +7.32] / -0.03 [-1.97, +1.93] / +7.94 [+5.66, +10.34] | +4.82 [+2.37, +7.32] / -0.03 [-1.97, +1.93] / +7.94 [+5.66, +10.34] |
| k=4 | learned_1x (67.175) / learned_1x (68.583) / learned_1x (68.217) | rival_lstm (44.433) / rival_mlp (41.567) / rival_transformer (51.992) | -6.24 [-7.92, -4.62] / -6.35 [-7.84, -4.92] / -3.26 [-4.58, -1.94] | -6.24 [-7.92, -4.62] / -6.35 [-7.84, -4.92] / -3.26 [-4.58, -1.94] |
| k=5 | learned_1x (68.050) / learned_1x (68.758) / learned_1x (69.050) | rival_lstm (44.433) / rival_mlp (41.567) / rival_transformer (51.992) | -4.81 [-6.55, -3.17] / -4.25 [-5.28, -3.27] / -2.64 [-3.68, -1.62] | -4.81 [-6.55, -3.17] / -4.25 [-5.28, -3.27] / -2.64 [-3.68, -1.62] |
| k=6 | learned_1x (68.467) / learned_1x (68.558) / learned_1x (68.825) | rival_lstm (44.433) / rival_mlp (41.567) / rival_transformer (51.992) | -5.78 [-7.52, -4.09] / -4.57 [-5.76, -3.45] / -1.84 [-2.93, -0.74] | -5.78 [-7.52, -4.09] / -4.57 [-5.76, -3.45] / -1.84 [-2.93, -0.74] |
| shipped | learned_1x (48.899) / learned_1x (49.541) / learned_1x (49.440) | rival_lstm (30.028) / rival_lstm (28.917) / rival_transformer (37.514) | -3.94 [-5.06, -2.85] / -3.22 [-4.17, -2.31] / -1.72 [-2.57, -0.87] | -3.94 [-5.06, -2.85] / -3.22 [-4.17, -2.31] / -1.72 [-2.57, -0.87] |

## 4b. ⭐ THE ANSWER — deterioration and the breaking point, per arm

Both words were defined in `AMENDMENT-2.md` at 2026-09-21T11:46:30Z, before any arm number existed. `deteriorates` compares the **ceiling-normalised** score with the arm's own k=2 value and requires the same sign on every seed. `clears the bar` is the harness's own `beats_beyond_margin` against the best learned baseline, on every seed. A DECORATIVE rung claims no row and is excluded from the breaking point.

| arm | deteriorates vs its own k=2 (k=3/4/5/6) | clears the bar (k=2..6) | **breaking point** |
|---|---|---|---|
| **single design** (K=5, DECLARED) | mixed / mixed / no / no | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| **pca_tensor** (wave-5 subspace store) | mixed / mixed / no / no | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | no / no / no / no | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | mixed / mixed / no / yes | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | no / no / no / no | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | mixed / mixed / no / yes | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| MLP @ 524,288 B ref | mixed / mixed / mixed / mixed | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| LSTM @ 524,288 B ref | mixed / mixed / mixed / mixed | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| transformer @ 524,288 B ref | no / no / no / no | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| continuous MLP @ 524,288 B ref | yes / yes / yes / yes | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| table FLOOR @ 524,288 B ref | mixed / mixed / mixed / mixed | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | yes / mixed / mixed / mixed | ✗ / ✗ / ✗ / ✗ / ✗ | NEVER CLEARED THE BAR -- the arm does not clear the pre-declared +1.0 margin against the best learned baseline at k=2, so there is nothing to break |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | yes / no / no / no | ✗ / ✗ / ✗ / ✗ / ✗ | NOT APPLICABLE -- this arm IS the best learned baseline at every rung that ran (k=2, k=3, k=4, k=5, k=6), so there is no bar for it to clear; it is the bar |

> **no arm has a breaking point** — arms with a breaking point: `{}`.

## 5. ⛔ END REPRODUCTION — the non-negotiable

### k = 2 vs the shipped ROW 3 — **the end reproduces on every arm and every seed**

| arm | rung mean (s0/s1/s2) | shipped mean (s0/s1/s2) | |Δ| | within the shipped corner sd on every seed |
|---|---|---|---|---|
| **single design** (K=5, DECLARED) | 28.633 / 31.650 / 31.333 | 29.717 / 31.617 / 31.450 | 1.083 | yes |
| **pca_tensor** (wave-5 subspace store) | 28.633 / 31.650 / 31.333 | 29.717 / 31.617 / 31.450 | 1.083 | yes |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | yes |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | 0.250 / 0.083 / 0.750 | 0.233 / 0.083 / 0.933 | 0.183 | yes |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | yes |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | 0.250 / 0.083 / 0.750 | 0.233 / 0.083 / 0.933 | 0.183 | yes |
| MLP @ 524,288 B ref | 17.017 / 22.000 / 23.333 | 17.017 / 22.000 / 23.333 | 0.000 | yes |
| LSTM @ 524,288 B ref | 19.417 / 19.333 / 23.750 | 19.417 / 19.333 / 23.750 | 0.000 | yes |
| transformer @ 524,288 B ref | 12.750 / 15.783 / 24.250 | 12.750 / 15.783 / 24.250 | 0.000 | yes |
| continuous MLP @ 524,288 B ref | 3.883 / 4.567 / 6.533 | 3.883 / 4.567 / 6.533 | 0.000 | yes |
| table FLOOR @ 524,288 B ref | 5.833 / 3.483 / 6.183 | 5.833 / 3.483 / 6.183 | 0.000 | yes |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | 5.567 / 7.583 / 9.300 | 5.583 / 7.817 / 9.067 | 0.233 | yes |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | 33.200 / 33.867 / 32.900 | 32.767 / 34.233 / 33.600 | 0.700 | yes |

### k = 6 (first 49 corners) vs the shipped ROW 4 — **the end reproduces on every arm and every seed**

| arm | rung mean (s0/s1/s2) | shipped mean (s0/s1/s2) | |Δ| | within the shipped corner sd on every seed |
|---|---|---|---|---|
| **single design** (K=5, DECLARED) | 63.184 / 63.796 / 67.918 | 63.612 / 64.327 / 67.653 | 0.531 | yes |
| **pca_tensor** (wave-5 subspace store) | 63.184 / 63.796 / 67.918 | 63.612 / 64.327 / 67.653 | 0.531 | yes |
| &nbsp;&nbsp;↳ single design · control `landscape_shuffled` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | yes |
| &nbsp;&nbsp;↳ single design · control `random_atoms` | 0.204 / 0.224 / 0.388 | 0.939 / 0.163 / 1.224 | 0.837 | yes |
| &nbsp;&nbsp;↳ pca_tensor · control `landscape_shuffled` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | yes |
| &nbsp;&nbsp;↳ pca_tensor · control `random_atoms` | 0.204 / 0.224 / 0.388 | 0.939 / 0.163 / 1.224 | 0.837 | yes |
| MLP @ 524,288 B ref | 37.245 / 36.633 / 52.755 | 37.245 / 36.633 / 52.755 | 0.000 | yes |
| LSTM @ 524,288 B ref | 43.020 / 40.653 / 51.510 | 43.020 / 40.653 / 51.510 | 0.000 | yes |
| transformer @ 524,288 B ref | 30.469 / 36.245 / 53.755 | 30.469 / 36.245 / 53.755 | 0.000 | yes |
| continuous MLP @ 524,288 B ref | 8.612 / 8.184 / 18.327 | 8.612 / 8.184 / 18.327 | 0.000 | yes |
| table FLOOR @ 524,288 B ref | 11.163 / 7.020 / 14.122 | 11.163 / 7.020 / 14.122 | 0.000 | yes |
| MLP @ 524,288 B ref, **own pool** (AMENDMENT-1) | 13.878 / 14.857 / 25.143 | 13.592 / 15.163 / 22.551 | 2.592 | yes |
| MLP @ **1x** the single design's own total (AMENDMENT-1) | 68.776 / 67.694 / 69.163 | 68.653 / 68.286 / 68.837 | 0.592 | yes |

## 6. The byte ledgers — measured on the objects this hub fitted, never netted

| arm | atoms | sufficient statistics | address book | total | × 524,288 B |
|---|---|---|---|---|---|
| single_design_K5 (leg `2_seed0`) | 524,288 B | 12,284,920 B | 158,144 B | 12,967,352 B | 24.73× |
| pca_tensor (leg `2_seed0`) | 524,288 B | 0 B | 190,904 B | 715,192 B | 1.36× |

> `single_design_K5` across 18 legs: total **12,523,008 – 13,041,864 B** = **23.89× – 24.88×** the 524,288 B reference pin.

> `pca_tensor` across 18 legs: total **712,072 – 719,352 B** = **1.36× – 1.37×** the 524,288 B reference pin.

## 7. ⛔ DECLARED NOT-RUN

None — every (rung, seed, arm) cell in the matrix was scored.
