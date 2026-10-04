# gen-variance-equalisation — full tables

Seeds [0, 1, 2] · ± is population sd (ddof=0) · headline grid = SPREAD


## 0. Per-seed provenance

| seed | cache scope | cache rows | store hash | well sizes | store_train | SPREAD_train | SPREAD | CORNER_B | CORNER_MS | wall clock (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `full` | 480000 | `3e3424a1ea40…` | [303, 305, 573, 501, 366] | 32768 | 12288 | 4096 | 72 | 198 | 2446 |
| 1 | `used` | 9288 | `76827ad534ee…` | [397, 300, 307, 300, 744] | 32768 | 12288 | 4096 | 72 | 199 | 1776 |
| 2 | `used` | 9288 | `8449e6057dac…` | [283, 283, 390, 616, 476] | 32768 | 12288 | 4096 | 72 | 199 | 1546 |

## 1. ⭐ LEG 1 — variance-explained per factor, and where the signal sits

Each factor in its own headline well (`argmax_k variance_explained`). `VE` and `rank_frac_median` are **null-subtracted** (PREREG addendum 2); the raw values are in §1b.

| factor | n_val | headline well (per seed) | **VE** (share of well variance) | **rank_frac_median** | frac signal in top 10 % of axes | var frac needed for half the signal |
|---|---|---|---|---|---|---|
| `floor_hue` | [10] | [4, 3, 4] | **0.7500 ± 0.0589** | **0.00342 ± 0.00060** | 0.999 ± 0.001 | 0.4771 ± 0.0689 |
| `wall_hue` | [10] | [1, 2, 0] | **0.6804 ± 0.0250** | **0.00336 ± 0.00013** | 0.997 ± 0.000 | 0.5248 ± 0.0058 |
| `object_hue` | [10] | [3, 4, 3] | **0.3094 ± 0.0092** | **0.00331 ± 0.00053** | 0.997 ± 0.001 | 0.3390 ± 0.0256 |
| `scale` | [8] | [3, 4, 3] | **0.0236 ± 0.0025** | **0.01358 ± 0.00458** | 0.984 ± 0.006 | 0.6283 ± 0.0464 |
| `shape` | [4] | [3, 4, 3] | **0.0503 ± 0.0110** | **0.00773 ± 0.00159** | 0.995 ± 0.002 | 0.5347 ± 0.0342 |
| `orientation` | [15] | [2, 2, 2] | **0.0995 ± 0.0406** | **0.00846 ± 0.00237** | 0.981 ± 0.003 | 0.5346 ± 0.0976 |

### 1b. The raw (biased) statistic, for comparison

| factor | VE raw | rank_frac_median raw | null bias share |
|---|---|---|---|
| `floor_hue` | 0.7505 ± 0.0589 | 0.00342 ± 0.00060 | 0.000549 |
| `wall_hue` | 0.6809 ± 0.0250 | 0.00336 ± 0.00013 | 0.000549 |
| `object_hue` | 0.3100 ± 0.0092 | 0.00331 ± 0.00053 | 0.000549 |
| `scale` | 0.0240 ± 0.0025 | 0.01291 ± 0.00366 | 0.000427 |
| `shape` | 0.0505 ± 0.0110 | 0.00773 ± 0.00159 | 0.000183 |
| `orientation` | 0.1003 ± 0.0406 | 0.00846 ± 0.00237 | 0.000854 |

### 1c. Per-well VE matrix (3-seed mean)

| well | `floor_hue` | `wall_hue` | `object_hue` | `scale` | `shape` | `orientation` | d_k (per seed) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0498 | 0.6503 | 0.0398 | 0.0076 | 0.0109 | 0.0540 | [303, 397, 283] |
| 1 | 0.0520 | 0.6375 | 0.0317 | 0.0059 | 0.0101 | 0.0673 | [305, 300, 283] |
| 2 | 0.0750 | 0.4843 | 0.0746 | 0.0093 | 0.0160 | 0.0995 | [573, 307, 390] |
| 3 | 0.4220 | 0.0479 | 0.2201 | 0.0200 | 0.0385 | 0.0127 | [501, 300, 616] |
| 4 | 0.5776 | 0.0464 | 0.1379 | 0.0172 | 0.0237 | 0.0158 | [366, 744, 476] |

### 1d. ⛔ THE PREMISE VERDICT

| seed | verdict | direction test | share test | ref rank_frac_median (max) | scale rank_frac | orientation rank_frac |
|---|---|---|---|---|---|---|
| 0 | **HOLDS** | True | True | 0.00328 | 0.01996 | 0.01047 |
| 1 | **HOLDS** | True | True | 0.00333 | 0.00941 | 0.00977 |
| 2 | **DEAD** | False | True | 0.00420 | 0.01136 | 0.00513 |

### 1e. ⭐ THE SPOKE-LEVEL PREMISE VERDICT (PREREG §3, ≥ 2 of 3)

`PREREG §3: the premise HOLDS iff >= 2 of 3 seeds hold` · seeds holding **2/3** · per seed ['HOLDS', 'HOLDS', 'DEAD'] ⇒ ⭐ **HOLDS**


### 1f. ⭐ EFFECTIVE DIMENSION per well (participation ratio) — post-hoc, NOT pre-registered

`PR = (Σλ)²/Σλ²`. High-rate quantiser theory gives distortion ratio ≈ `M^(−2/PR)` for an `M`-point quantiser on a source of effective dimension `PR`. ⛔ Whitening sets `PR = d_k` exactly.

| seed | well | d_k | total variance | **PR** | `64^(−2/PR)` (unwhitened bound) | `64^(−2/d_k)` (whitened bound) |
|---|---|---|---|---|---|---|
| 0 | 0 | 303 | 68.90 | **3.12** | 0.0694 | 0.9729 |
| 0 | 1 | 305 | 86.48 | **3.46** | 0.0906 | 0.9731 |
| 0 | 2 | 573 | 118.92 | **12.54** | 0.5153 | 0.9856 |
| 0 | 3 | 501 | 68.63 | **10.50** | 0.4528 | 0.9835 |
| 0 | 4 | 366 | 113.22 | **4.07** | 0.1293 | 0.9775 |
| 1 | 0 | 397 | 105.77 | **5.62** | 0.2276 | 0.9793 |
| 1 | 1 | 300 | 70.10 | **4.01** | 0.1254 | 0.9727 |
| 1 | 2 | 307 | 82.49 | **3.37** | 0.0848 | 0.9733 |
| 1 | 3 | 300 | 102.09 | **3.86** | 0.1156 | 0.9727 |
| 1 | 4 | 744 | 104.25 | **12.00** | 0.4999 | 0.9889 |
| 2 | 0 | 283 | 81.65 | **3.46** | 0.0904 | 0.9710 |
| 2 | 1 | 283 | 63.73 | **3.85** | 0.1154 | 0.9710 |
| 2 | 2 | 390 | 83.09 | **8.45** | 0.3738 | 0.9789 |
| 2 | 3 | 616 | 91.64 | **13.63** | 0.5433 | 0.9866 |
| 2 | 4 | 476 | 171.45 | **5.64** | 0.2288 | 0.9827 |

## 2. ⭐⭐ LEG 3 — the six-factor graded profile, BEFORE and AFTER, SPREAD grid

Retention = `(acc − chance)/(acc_encoder − chance)`; the encoder point is the SAME probe for every arm.

| factor | n_val | chance | encoder | **diag_std** | **identity** | **pca_whiten** | **shipped** |
|---|---|---|---|---|---|---|---|
| `floor_hue` | [10] | 0.100 | 0.9994 ± 0.0003 | 0.990 ± 0.012 | 0.997 ± 0.002 | 0.227 ± 0.025 | 0.997 ± 0.002 |
| `wall_hue` | [10] | 0.100 | 0.9997 ± 0.0001 | 0.991 ± 0.001 | 0.995 ± 0.004 | 0.372 ± 0.035 | 0.995 ± 0.004 |
| `object_hue` | [10] | 0.100 | 0.9994 ± 0.0003 | 0.463 ± 0.057 | 0.665 ± 0.043 | 0.155 ± 0.031 | 0.665 ± 0.043 |
| `scale` | [8] | 0.125 | 0.9352 ± 0.0058 | 0.218 ± 0.013 | 0.224 ± 0.004 | 0.086 ± 0.012 | 0.224 ± 0.004 |
| `shape` | [4] | 0.250 | 0.9960 ± 0.0008 | 0.423 ± 0.029 | 0.555 ± 0.034 | 0.208 ± 0.022 | 0.555 ± 0.034 |
| `orientation` | [15] | 0.067 | 0.9915 ± 0.0018 | 0.299 ± 0.040 | 0.243 ± 0.042 | 0.124 ± 0.020 | 0.243 ± 0.042 |
| **TOTAL RETAINED** | — | — | — | **3.383 ± 0.035** | **3.678 ± 0.075** | **1.172 ± 0.076** | **3.678 ± 0.075** |
| mean retained | — | — | — | 0.564 ± 0.006 | 0.613 ± 0.012 | 0.195 ± 0.013 | 0.613 ± 0.012 |

### 2b. Raw code-probe accuracies (same table, un-normalised)

| factor | chance | code_random | **diag_std** | **identity** | **pca_whiten** | **shipped** |
|---|---|---|---|---|---|---|
| `floor_hue` | 0.100 | 0.0954 ± 0.0033 | 0.9902 ± 0.0109 | 0.9966 ± 0.0019 | 0.3041 ± 0.0228 | 0.9966 ± 0.0019 |
| `wall_hue` | 0.100 | 0.1007 ± 0.0065 | 0.9913 ± 0.0012 | 0.9948 ± 0.0032 | 0.4342 ± 0.0316 | 0.9948 ± 0.0032 |
| `object_hue` | 0.100 | 0.0989 ± 0.0037 | 0.5164 ± 0.0516 | 0.6981 ± 0.0389 | 0.2397 ± 0.0277 | 0.6981 ± 0.0389 |
| `scale` | 0.125 | 0.1199 ± 0.0051 | 0.3017 ± 0.0095 | 0.3069 ± 0.0042 | 0.1948 ± 0.0099 | 0.3069 ± 0.0042 |
| `shape` | 0.250 | 0.2533 ± 0.0059 | 0.5658 ± 0.0216 | 0.6638 ± 0.0248 | 0.4052 ± 0.0166 | 0.6638 ± 0.0248 |
| `orientation` | 0.067 | 0.0649 ± 0.0011 | 0.3427 ± 0.0369 | 0.2909 ± 0.0390 | 0.1812 ± 0.0185 | 0.2909 ± 0.0390 |

### 2c. Store MI (the free instrument): max_k MI(code_k; f)/H(f)

| factor | **diag_std** | **identity** | **pca_whiten** | **shipped** |
|---|---|---|---|---|
| `floor_hue` | 0.989 ± 0.012 | 0.995 ± 0.004 | 0.094 ± 0.011 | 0.995 ± 0.004 |
| `wall_hue` | 0.941 ± 0.022 | 0.959 ± 0.028 | 0.171 ± 0.029 | 0.959 ± 0.028 |
| `object_hue` | 0.443 ± 0.049 | 0.651 ± 0.019 | 0.070 ± 0.023 | 0.651 ± 0.019 |
| `scale` | 0.170 ± 0.016 | 0.169 ± 0.006 | 0.049 ± 0.001 | 0.169 ± 0.006 |
| `shape` | 0.251 ± 0.059 | 0.428 ± 0.071 | 0.060 ± 0.009 | 0.428 ± 0.071 |
| `orientation` | 0.364 ± 0.018 | 0.353 ± 0.026 | 0.083 ± 0.008 | 0.353 ± 0.026 |

## 3. ⭐⭐ THE TWO PRE-REGISTERED FALSIFIERS

**(a) does the probe rise? → NO**

| factor | retention `identity` | retention `pca_whiten` | Δ | seed spread | rose beyond spread? |
|---|---|---|---|---|---|
| `orientation` | 0.2425 ± 0.0424 | 0.1238 ± 0.0202 | -0.1187 | 0.0424 | **NO** |
| `scale` | 0.2245 ± 0.0036 | 0.0861 ± 0.0119 | -0.1383 | 0.0119 | **NO** |

**(b) does the distinct configuration count rise? → NO**

| count | `identity` | `pca_whiten` | Δ | seed spread |
|---|---|---|---|---|
| ⭐ request-side distinct of 72 on CORNER_B (PRIMARY) | 9.00 ± 0.82 | 5.67 ± 0.94 | -3.33 | 0.94 |
| feature-side distinct codes on 4,096 SPREAD rows (secondary) | 3638.3 ± 59.9 | 626.0 ± 126.4 | -3012.3 | — |

**Pre-registered pass (BOTH): NO**


### 3b. The SECONDARY arm `diag_std`, same adjudication (⛔ not promoted to primary)

**(a) probe rises → NO** · **(b) distinct rises → YES** · **both → NO**

| quantity | `identity` | `diag_std` | Δ | seed spread | rose? |
|---|---|---|---|---|---|
| retention `orientation` | 0.2425 ± 0.0424 | 0.2985 ± 0.0403 | +0.0560 | 0.0424 | **YES** |
| retention `scale` | 0.2245 ± 0.0036 | 0.2181 ± 0.0125 | -0.0064 | 0.0125 | **NO** |
| ⭐ distinct of 72 on CORNER_B | 9.00 ± 0.82 | 15.00 ± 2.94 | +6.00 | 2.94 | **YES** |

`TOTAL-RETAINED` 3.678 ± 0.075 → 3.383 ± 0.035; NOT the flat-total case — the total moved beyond seed spread, so the metric changed how much is retained overall, not merely its distribution.


**TOTAL-RETAINED reading:** NOT the flat-total case — the total moved beyond seed spread, so the metric changed how much is retained overall, not merely its distribution.

`total_before=3.6781` · `total_after=1.1717` · `Δ=-2.5064` · `seed spread=0.0755` · flat=False


## 4. Distinct configuration counts, every arm, every row set

| arm | request-side / 72 (CORNER_B) | request-side / SPREAD | feature-side / SPREAD (4,096) |
|---|---|---|---|
| `diag_std` | 15.00 ± 2.94 | 3331.7 ± 127.5 | 3412.0 ± 28.9 |
| `identity` | 9.00 ± 0.82 | 3572.3 ± 71.2 | 3638.3 ± 59.9 |
| `pca_whiten` | 5.67 ± 0.94 | 425.0 ± 52.2 | 626.0 ± 126.4 |
| `shipped` | 9.00 ± 0.82 | 3572.3 ± 71.2 | 3638.3 ± 59.9 |

## 5. ⭐⭐ LEG 4 — the per-well capacity bound

`M = 64` · `K = 5` · aggregate `K·log2(M)` = 30.00 bits vs `log2(480000)` = 18.87 bits ⇒ aggregate sufficient = True

| arm | wells over capacity (of 5) | per-well loads (seed 0) | carried factors (seed 0) |
|---|---|---|---|
| `diag_std` | 4.33 ± 0.47 | [150, 150, 10, 100, 100] | ['wall_hue+orientation', 'wall_hue+orientation', 'wall_hue', 'floor_hue+object_hue', 'floor_hue+object_hue'] |
| `identity` | 4.67 ± 0.47 | [1500, 150, 100, 400, 100] | ['floor_hue+wall_hue+orientation', 'wall_hue+orientation', 'wall_hue+object_hue', 'floor_hue+object_hue+shape', 'floor_hue+object_hue'] |
| `pca_whiten` | 0.00 ± 0.00 | [1, 1, 1, 1, 1] | ['—', '—', '—', '—', '—'] |
| `shipped` | 4.67 ± 0.47 | [1500, 150, 100, 400, 100] | ['floor_hue+wall_hue+orientation', 'wall_hue+orientation', 'wall_hue+object_hue', 'floor_hue+object_hue+shape', 'floor_hue+object_hue'] |

## 6. ⭐ THE BYTE LEDGER (bytes, per seed where the well sizes differ)

| arm | seed | Σ d_k² | **deployed bytes** | vs 524,288 B budget | + explicit inverse | **fit-time-only** |
|---|---|---|---|---|---|---|
| `diag_std` | 0 | 898,120 | **16,384** | 0.0312× | 24,576 | **0** |
| `diag_std` | 1 | 985,394 | **16,384** | 0.0312× | 24,576 | **0** |
| `diag_std` | 2 | 918,310 | **16,384** | 0.0312× | 24,576 | **0** |
| `identity` | 0 | 898,120 | **0** | 0.0000× | 0 | **0** |
| `identity` | 1 | 985,394 | **0** | 0.0000× | 0 | **0** |
| `identity` | 2 | 918,310 | **0** | 0.0000× | 0 | **0** |
| `pca_whiten` | 0 | 898,120 | **3,600,672** | 6.8677× | 3,608,864 | **0** |
| `pca_whiten` | 1 | 985,394 | **3,949,768** | 7.5336× | 3,957,960 | **0** |
| `pca_whiten` | 2 | 918,310 | **3,681,432** | 7.0218× | 3,689,624 | **0** |
| `shipped` | 0 | 898,120 | **0** | 0.0000× | 0 | **0** |
| `shipped` | 1 | 985,394 | **0** | 0.0000× | 0 | **0** |
| `shipped` | 2 | 918,310 | **0** | 0.0000× | 0 | **0** |

## 7. Round trip + deployment equivalence + per-well matched-compute scalar

| arm | seed | round-trip max abs | max rel | deployment-equivalence max rel Δ | per-well scalar α_k |
|---|---|---|---|---|---|
| `diag_std` | 0 | 2.384e-07 | 4.066e-08 | 0.000e+00 | ['0.8885', '0.8457', '0.6598', '0.5259', '0.8069'] |
| `diag_std` | 1 | 2.384e-07 | 3.915e-08 | 0.000e+00 | ['0.7794', '0.809', '0.8921', '0.8247', '0.5297'] |
| `diag_std` | 2 | 2.384e-07 | 3.893e-08 | 0.000e+00 | ['0.8802', '0.733', '0.7486', '0.526', '0.8663'] |
| `identity` | 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | ['1', '1', '1', '1', '1'] |
| `identity` | 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | ['1', '1', '1', '1', '1'] |
| `identity` | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | ['1', '1', '1', '1', '1'] |
| `pca_whiten` | 0 | 2.384e-07 | 4.066e-08 | 0.000e+00 | ['0.8947', '0.8458', '0.6599', '0.5282', '0.807'] |
| `pca_whiten` | 1 | 2.384e-07 | 3.915e-08 | 0.000e+00 | ['0.7795', '0.8179', '0.8922', '0.8248', '0.5322'] |
| `pca_whiten` | 2 | 2.384e-07 | 3.893e-08 | 0.000e+00 | ['0.881', '0.7387', '0.7487', '0.5275', '0.8664'] |

## 8. Reconstruction MSE in the ORIGINAL feature space, and the under-training check

| arm | seed | MSE (SPREAD) | relative MSE | MSE at k-means++ init | training improved the hard read? | store-only loss first→last |
|---|---|---|---|---|---|---|
| `diag_std` | 0 | 0.07146 | 0.3204 | 0.10592 | True | 136.4304 → 86.1025 |
| `diag_std` | 1 | 0.07278 | 0.3209 | 0.10762 | True | 132.0207 → 95.6343 |
| `diag_std` | 2 | 0.07954 | 0.3314 | 0.11977 | True | 149.3967 → 101.9539 |
| `identity` | 0 | 0.06210 | 0.2784 | 0.09734 | True | 47.4907 → 31.6470 |
| `identity` | 1 | 0.06134 | 0.2704 | 0.09879 | True | 45.0341 → 29.7418 |
| `identity` | 2 | 0.06867 | 0.2861 | 0.10935 | True | 51.4523 → 34.3546 |
| `pca_whiten` | 0 | 0.20729 | 0.9293 | 0.28698 | True | 354.7682 → 242.4967 |
| `pca_whiten` | 1 | 0.21092 | 0.9298 | 0.27635 | True | 334.8900 → 244.8330 |
| `pca_whiten` | 2 | 0.22605 | 0.9417 | 0.30489 | True | 367.6517 → 262.0528 |

## 9. Decoded reads through gen-decoder-lift's WINNER (v1_capacity_schedule)

⛔ recipe + per-seed `param_hash` asserted. n = 1024 SPREAD rows per arm. ⭐ Every read was mapped back through the inverse first.

| arm | aggregate | `floor_hue` | `wall_hue` | `object_hue` | `scale` | `shape` | `orientation` |
|---|---|---|---|---|---|---|---|
| `diag_std` | 0.5642 ± 0.0018 | 0.993 ± 0.004 | 0.978 ± 0.006 | 0.474 ± 0.062 | 0.202 ± 0.023 | 0.476 ± 0.032 | 0.263 ± 0.039 |
| `identity` | 0.6225 ± 0.0124 | 0.998 ± 0.002 | 0.989 ± 0.005 | 0.686 ± 0.030 | 0.208 ± 0.015 | 0.599 ± 0.028 | 0.255 ± 0.046 |
| `pca_whiten` | 0.1952 ± 0.0318 | 0.215 ± 0.126 | 0.201 ± 0.008 | 0.208 ± 0.044 | 0.130 ± 0.001 | 0.315 ± 0.020 | 0.102 ± 0.019 |
| `shipped` | 0.6225 ± 0.0124 | 0.998 ± 0.002 | 0.989 ± 0.005 | 0.686 ± 0.030 | 0.208 ± 0.015 | 0.599 ± 0.028 | 0.255 ± 0.046 |

## 10. ⚠ THE CORNER — reported SEPARATELY, ⛔ never as the global profile


### CORNER_B

| factor | n_val | **diag_std** | **identity** | **pca_whiten** | **shipped** |
|---|---|---|---|---|---|
| `floor_hue` | [2] | 1.000 ± 0.000 | 0.897 ± 0.145 | 0.552 ± 0.151 | 0.897 ± 0.145 |
| `wall_hue` | [2] | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.475 ± 0.045 | 1.000 ± 0.000 |
| `object_hue` | [2] | 0.552 ± 0.050 | 0.733 ± 0.196 | 0.187 ± 0.191 | 0.733 ± 0.196 |
| `scale` | [3] | 0.161 ± 0.129 | 0.241 ± 0.023 | 0.078 ± 0.047 | 0.241 ± 0.023 |
| `shape` | **1** | ⛔ NOT APPLICABLE (R31) | ⛔ NOT APPLICABLE (R31) | ⛔ NOT APPLICABLE (R31) | ⛔ NOT APPLICABLE (R31) |
| `orientation` | [3] | 0.251 ± 0.071 | 0.177 ± 0.184 | 0.320 ± 0.118 | 0.177 ± 0.184 |
| **TOTAL RETAINED** | — | **2.965 ± 0.084** | **3.047 ± 0.204** | **1.613 ± 0.253** | **3.047 ± 0.204** |

### CORNER_MS

| factor | n_val | **diag_std** | **identity** | **pca_whiten** | **shipped** |
|---|---|---|---|---|---|
| `floor_hue` | [2] | 1.000 ± 0.000 | 0.963 ± 0.053 | 0.454 ± 0.132 | 0.963 ± 0.053 |
| `wall_hue` | [2] | 0.976 ± 0.018 | 0.989 ± 0.012 | 0.314 ± 0.060 | 0.989 ± 0.012 |
| `object_hue` | [2] | 0.508 ± 0.046 | 0.722 ± 0.090 | 0.187 ± 0.141 | 0.722 ± 0.090 |
| `scale` | [3] | 0.297 ± 0.117 | 0.285 ± 0.031 | 0.123 ± 0.030 | 0.285 ± 0.031 |
| `shape` | [3] | 0.501 ± 0.121 | 0.653 ± 0.102 | 0.186 ± 0.170 | 0.653 ± 0.102 |
| `orientation` | [3] | 0.229 ± 0.036 | 0.200 ± 0.141 | 0.256 ± 0.069 | 0.200 ± 0.141 |
| **TOTAL RETAINED** | — | **3.511 ± 0.173** | **3.812 ± 0.264** | **1.520 ± 0.311** | **3.812 ± 0.264** |

## 11. ⭐ THE LAUNDERING CONTROL — knn_proto

| seed | feature sha256 identical | embedding sha256 identical | max abs Δ | verdict |
|---|---|---|---|---|
| 0 | True | True | 0.0 | **BIT-IDENTICAL** |
| 1 | True | True | 0.0 | **BIT-IDENTICAL** |
| 2 | True | True | 0.0 | **BIT-IDENTICAL** |

## 12. ⭐ WHY THE ORDER IS RULED — block diagonality under the SAME labels

⭐ The grouping was fitted on the UNWHITENED correlation structure. Per-well whitening makes every within-well correlation the identity, so that structure is destroyed — which ⛔ is exactly why R43.1 rules the grouping is fitted FIRST.

| seed | in-block |corr| fraction, unwhitened | whitened | **lift over size-matched random**, unwhitened | whitened |
|---|---|---|---|---|
| 0 | 0.4030 | 0.1751 | **1.886** | **0.819** |
| 1 | 0.4208 | 0.1745 | **1.794** | **0.744** |
| 2 | 0.4035 | 0.1808 | **1.846** | **0.827** |

(the size-matched random baseline, `in_block_pair_fraction`, is 0.2137 and is identical for both, since the labels are the same)
