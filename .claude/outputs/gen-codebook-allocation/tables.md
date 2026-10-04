# gen-codebook-allocation — tables

⛔ Generated from `summary.json`; every number is re-read, never transcribed.

## 1. ⭐ The measured, LABEL-FREE per-well effective dimension `PR`

| seed | well 0 | well 1 | well 2 | well 3 | well 4 | `d_k` |
|---|---|---|---|---|---|---|
| 0 | 3.117 | 3.463 | 12.545 | 10.498 | 4.067 | [303, 305, 573, 501, 366] |
| 1 | 5.619 | 4.007 | 3.371 | 3.856 | 11.998 | [397, 300, 307, 300, 744] |
| 2 | 3.461 | 3.852 | 8.453 | 13.635 | 5.639 | [283, 283, 390, 616, 476] |

⚠ Wells are **not** aligned across seeds (the grouping is re-fitted per seed), so a column is not a well.

## 2. ⭐⭐ THE ALLOCATIONS AND THE BYTE GATE (acceptance 2)

| seed | arm | `M_k` | store bytes | Δ vs uniform | shortfall | bits/read | `Σ M_k` |
|---|---|---|---|---|---|---|---|
| 0 | `equal_distortion` | [4, 4, 160, 70, 5] | 524,048 B | -240 B | 0.046 % | 19.77 | 243 |
| 0 | `mse_optimal` | [51, 64, 72, 53, 77] | 523,856 B | -432 B | 0.082 % | 29.84 | 317 |
| 0 | `uniform` | [64, 64, 64, 64, 64] | 524,288 B | +0 B | 0.000 % | 30.00 | 320 |
| 1 | `equal_distortion` | [11, 6, 4, 5, 164] | 523,644 B | -644 B | 0.123 % | 17.72 | 190 |
| 1 | `mse_optimal` | [83, 62, 59, 78, 51] | 524,032 B | -256 B | 0.049 % | 30.17 | 333 |
| 1 | `uniform` | [64, 64, 64, 64, 64] | 524,288 B | +0 B | 0.000 % | 30.00 | 320 |
| 2 | `equal_distortion` | [4, 4, 25, 186, 9] | 523,496 B | -792 B | 0.151 % | 19.35 | 228 |
| 2 | `mse_optimal` | [58, 54, 68, 46, 93] | 523,280 B | -1,008 B | 0.192 % | 29.76 | 319 |
| 2 | `uniform` | [64, 64, 64, 64, 64] | 524,288 B | +0 B | 0.000 % | 30.00 | 320 |

**Gate:** any arm over budget = **False** · max shortfall **0.192 %**.

## 3. ⭐⭐ THE SIX-FACTOR RETENTION LADDER + TOTAL-RETAINED (SPREAD, 3 seeds)

| factor | `n_values_realised` | chance | encoder | **`shipped`** | **`uniform`** | **`equal_distortion`** | **`mse_optimal`** |
|---|---|---|---|---|---|---|---|
| `floor_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | 0.9968 ± 0.0021 | 0.9968 ± 0.0021 | 0.7563 ± 0.0215 | 0.9964 ± 0.0057 |
| `wall_hue` | 10 | 0.100 | 0.9997 ± 0.0001 | 0.9946 ± 0.0036 | 0.9946 ± 0.0036 | 0.9541 ± 0.0117 | 0.9936 ± 0.0012 |
| `object_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | 0.6649 ± 0.0432 | 0.6649 ± 0.0432 | 0.6670 ± 0.0310 | 0.6644 ± 0.0468 |
| `scale` | 8 | 0.125 | 0.9352 ± 0.0058 | 0.2245 ± 0.0036 | 0.2245 ± 0.0036 | 0.1676 ± 0.0139 | 0.2243 ± 0.0018 |
| `shape` | 4 | 0.250 | 0.9960 ± 0.0008 | 0.5547 ± 0.0338 | 0.5547 ± 0.0338 | 0.5628 ± 0.0307 | 0.5425 ± 0.0293 |
| `orientation` | 15 | 0.067 | 0.9915 ± 0.0018 | 0.2425 ± 0.0424 | 0.2425 ± 0.0424 | 0.0645 ± 0.0230 | 0.2451 ± 0.0416 |
| ⭐⭐ **TOTAL RETAINED** | — | — | — | **3.678 ± 0.075** | **3.678 ± 0.075** | **3.172 ± 0.094** | **3.666 ± 0.040** |
| random-code control (same `M_k`) | — | — | — | -0.014 ± 0.018 | -0.014 ± 0.018 | -0.014 ± 0.008 | -0.000 ± 0.016 |
| one-hot probe width `Σ M_k`, per seed | — | — | — | [320, 320, 320] | [320, 320, 320] | [243, 190, 228] | [317, 333, 319] |

## 4. ⭐ DISTINCT CONFIGURATIONS ON THE 72 HELD-OUT REQUESTS (acceptance 4)

`CORNER_B` = **72** requests.

| arm | ⭐ request-side distinct of 72 | feature-side distinct codes / 4,096 SPREAD |
|---|---|---|
| `shipped` | **9.00 ± 0.82** | 3638.3 ± 59.9 |
| `uniform` | **9.00 ± 0.82** | 3638.3 ± 59.9 |
| `equal_distortion` | **8.00 ± 2.94** | 2499.3 ± 263.0 |
| `mse_optimal` | **14.33 ± 3.30** | 3620.7 ± 31.4 |

## 5. ⭐⭐ THE LAW — measured vs `M^(−2/PR)` (acceptance 5: the FIT, not the win)

### 5a. Variance-weighted, per arm

| arm | predicted `Σ v_k ε_k / Σ v_k` | measured relative distortion | ratio |
|---|---|---|---|
| `shipped` | — | **0.2783 ± 0.0064** | nan |
| `uniform` | 0.2533 ± 0.0217 | **0.2783 ± 0.0064** | 1.099 |
| `equal_distortion` | 0.4446 ± 0.0145 | **0.4044 ± 0.0083** | 0.910 |
| `mse_optimal` | 0.2511 ± 0.0212 | **0.2795 ± 0.0059** | 1.113 |

### 5b. The pooled log-log fit

`log ε_measured = slope · log ε_predicted + intercept`, **60 points** (arm × well × seed).

| quantity | value |
|---|---|
| slope | **0.5019** |
| multiplicative offset `exp(intercept)` | **0.5989** |
| Pearson `r` (log) | **0.9048** |
| `R²` (log) | 0.8187 |
| median ratio measured/predicted | **1.0780** |
| ratio range | [0.737, 2.728] |
| fraction of points within [0.5, 2.0]× | **0.767** |

### 5c. Per arm

| arm | slope | offset | `r` | median ratio | n |
|---|---|---|---|---|---|
| `shipped` | 0.4812 | 0.5727 | 0.8849 | 1.1814 | 15 |
| `uniform` | 0.4812 | 0.5727 | 0.8849 | 1.1814 | 15 |
| `equal_distortion` | 0.0150 | 0.4131 | 0.0056 | 0.9457 | 15 |
| `mse_optimal` | 0.5060 | 0.6009 | 0.8916 | 1.2441 | 15 |

### 5d. Every point (`uniform` arm only — the shipped configuration)

| seed | well | `PR` | `d_k` | `M_k` | predicted ε | measured ε | ratio |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.117 | 303 | 64 | 0.0694 | **0.1892** | 2.728 |
| 0 | 1 | 3.463 | 305 | 64 | 0.0906 | **0.1827** | 2.018 |
| 0 | 2 | 12.545 | 573 | 64 | 0.5153 | **0.4664** | 0.905 |
| 0 | 3 | 10.498 | 501 | 64 | 0.4528 | **0.3915** | 0.865 |
| 0 | 4 | 4.067 | 366 | 64 | 0.1293 | **0.1409** | 1.089 |
| 1 | 0 | 5.619 | 397 | 64 | 0.2276 | **0.3073** | 1.351 |
| 1 | 1 | 4.007 | 300 | 64 | 0.1254 | **0.2613** | 2.083 |
| 1 | 2 | 3.371 | 307 | 64 | 0.0848 | **0.2099** | 2.475 |
| 1 | 3 | 3.856 | 300 | 64 | 0.1156 | **0.1366** | 1.181 |
| 1 | 4 | 11.998 | 744 | 64 | 0.4999 | **0.4170** | 0.834 |
| 2 | 0 | 3.461 | 283 | 64 | 0.0904 | **0.1752** | 1.939 |
| 2 | 1 | 3.852 | 283 | 64 | 0.1154 | **0.2371** | 2.055 |
| 2 | 2 | 8.453 | 390 | 64 | 0.3738 | **0.3602** | 0.964 |
| 2 | 3 | 13.635 | 616 | 64 | 0.5433 | **0.4260** | 0.784 |
| 2 | 4 | 5.639 | 476 | 64 | 0.2288 | **0.2466** | 1.078 |

### 5e. ⚠ POST-HOC (`PREREG_ADDENDUM_1` §2) — the law's EXPONENT, within each well

⚠ POST-HOC — not pre-registered (PREREG_ADDENDUM_1 §2)

| seed | well | `PR` | `M` swept | measured ε | fitted exponent | law `−2/PR` | fitted/law |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.117 | [4, 51, 64] | 0.433, 0.204, 0.189 | **-0.2968** | -0.6416 | 0.462 |
| 0 | 1 | 3.463 | [4, 64, 64] | 0.445, 0.183, 0.185 | **-0.3186** | -0.5775 | 0.552 |
| 0 | 2 | 12.545 | [64, 72, 160] | 0.466, 0.456, 0.384 | **-0.2119** | -0.1594 | 1.329 |
| 0 | 3 | 10.498 | [53, 64, 70] | 0.404, 0.391, 0.385 | **-0.1710** | -0.1905 | 0.897 |
| 0 | 4 | 4.067 | [5, 64, 77] | 0.342, 0.141, 0.136 | **-0.3411** | -0.4918 | 0.694 |
| 1 | 0 | 5.619 | [11, 64, 83] | 0.454, 0.307, 0.290 | **-0.2211** | -0.3560 | 0.621 |
| 1 | 1 | 4.007 | [6, 62, 64] | 0.458, 0.266, 0.261 | **-0.2351** | -0.4991 | 0.471 |
| 1 | 2 | 3.371 | [4, 59, 64] | 0.472, 0.216, 0.210 | **-0.2915** | -0.5933 | 0.491 |
| 1 | 3 | 3.856 | [5, 64, 78] | 0.358, 0.137, 0.130 | **-0.3726** | -0.5187 | 0.718 |
| 1 | 4 | 11.998 | [51, 64, 164] | 0.443, 0.417, 0.339 | **-0.2260** | -0.1667 | 1.356 |
| 2 | 0 | 3.461 | [4, 58, 64] | 0.424, 0.180, 0.175 | **-0.3201** | -0.5779 | 0.554 |
| 2 | 1 | 3.852 | [4, 54, 64] | 0.495, 0.249, 0.237 | **-0.2642** | -0.5192 | 0.509 |
| 2 | 2 | 8.453 | [25, 64, 68] | 0.448, 0.360, 0.350 | **-0.2404** | -0.2366 | 1.016 |
| 2 | 3 | 13.635 | [46, 64, 186] | 0.459, 0.426, 0.343 | **-0.2080** | -0.1467 | 1.418 |
| 2 | 4 | 5.639 | [9, 64, 93] | 0.389, 0.247, 0.230 | **-0.2277** | -0.3547 | 0.642 |

**fitted exponent: -0.2631 ± 0.0555** over `PR` = 3.12 … 13.63, against a law exponent spanning **-0.6416 … -0.1467**. ratio fitted/law: median **0.642**, range [0.462, 1.418].

## 6. ⛔ THE CEILING — what NO re-allocation can buy (§0.5 consequence 2)

| target ε | modes needed per well (seed 0) | bytes | × the 524,288 B budget | feasible? |
|---|---|---|---|---|
| 0.278 | 7, 9, 3,070, 828, 14 | 8,735,228 B | 16.7× | ⛔ NO |
| 0.100 | 36, 54, 1,871,769, 177,357, 108 | 4,645,787,114 B | 8861.1× | ⛔ NO |
| 0.050 | 107, 179, 144,673,307, 6,743,888, 442 | 345,106,964,709 B | 658239.3× | ⛔ NO |

## 7. ⭐ THE VERDICT, in the pre-registered words

### `equal_distortion`

- **pre-registered pass: NO**
- ⛔ LOSS: the allocated arm is worse than the uniform control beyond seed spread.
- TOTAL-RETAINED: control 3.678 ± 0.075 vs arm 3.172 ± 0.094 · Δ = **-0.506** (seed spread 0.094)
- distinct-of-72: control 9.00 ± 0.82 vs arm 8.00 ± 2.94 · Δ = **-1.00** (spread 2.94)
- `orientation`: 0.2425 ± 0.0424 → 0.0645 ± 0.0230 · Δ = **-0.1780** · rose beyond spread = **False**
- `scale`: 0.2245 ± 0.0036 → 0.1676 ± 0.0139 · Δ = **-0.0569** · rose beyond spread = **False**
- bytes ≤ uniform every seed: **True** ([-240, -644, -792] B)

### `mse_optimal`

- **pre-registered pass: NO**
- ⭐ SPLIT, ⛔ NOT A PASS (PREREG_ADDENDUM_1): TOTAL-RETAINED is WITHIN seed spread of the uniform arm while distinct-of-72 ROSE beyond spread. ⛔ The pre-registered pass requires TOTAL to RISE, so the answer is NO — but nothing fell, so this is ⛔ not a loss. ⚠ A 72-row statistic on 3 seeds: POST-HOC, replicate before use.
- TOTAL-RETAINED: control 3.678 ± 0.075 vs arm 3.666 ± 0.040 · Δ = **-0.012** (seed spread 0.075)
- distinct-of-72: control 9.00 ± 0.82 vs arm 14.33 ± 3.30 · Δ = **+5.33** (spread 3.30)
- `orientation`: 0.2425 ± 0.0424 → 0.2451 ± 0.0416 · Δ = **+0.0025** · rose beyond spread = **False**
- `scale`: 0.2245 ± 0.0036 → 0.2243 ± 0.0018 · Δ = **-0.0002** · rose beyond spread = **False**
- bytes ≤ uniform every seed: **True** ([-432, -256, -1008] B)

## 8. The decoded column — ⛔ through the WINNER decoder (`v1_capacity_schedule`)

| arm | aggregate | `floor_hue` | `wall_hue` | `object_hue` | `scale` | `shape` | `orientation` |
|---|---|---|---|---|---|---|---|
| `shipped` | **0.6225 ± 0.0124** | 0.998 | 0.989 | 0.686 | 0.208 | 0.599 | 0.255 |
| `uniform` | **0.6225 ± 0.0124** | 0.998 | 0.989 | 0.686 | 0.208 | 0.599 | 0.255 |
| `equal_distortion` | **0.5167 ± 0.0192** | 0.568 | 0.918 | 0.705 | 0.176 | 0.625 | 0.108 |
| `mse_optimal` | **0.6142 ± 0.0045** | 0.998 | 0.988 | 0.637 | 0.231 | 0.586 | 0.244 |

## 9. ⭐ THE LAUNDERING CONTROLS

| check | result |
|---|---|
| `knn_proto` embedding, seed 0 | **BIT-IDENTICAL** (max \|Δ\| = 0.0) |
| `knn_proto` embedding, seed 1 | **BIT-IDENTICAL** (max \|Δ\| = 0.0) |
| `knn_proto` embedding, seed 2 | **BIT-IDENTICAL** (max \|Δ\| = 0.0) |
| `shipped` reference vs re-fitted `uniform`, TOTAL-RETAINED | 3.678 ± 0.075 vs 3.678 ± 0.075 |
| random-code control at `shipped`'s own `M_k`, TOTAL | -0.014 ± 0.018 |
| random-code control at `uniform`'s own `M_k`, TOTAL | -0.014 ± 0.018 |
| random-code control at `equal_distortion`'s own `M_k`, TOTAL | -0.014 ± 0.008 |
| random-code control at `mse_optimal`'s own `M_k`, TOTAL | -0.000 ± 0.016 |

## 10. Mode usage (is an over-allocated well actually used?)

| seed | arm | used / allocated modes | realised bits | allocated bits |
|---|---|---|---|---|
| 0 | `shipped` | 320 / 320 | 29.32 | 30.00 |
| 0 | `uniform` | 320 / 320 | 29.32 | 30.00 |
| 0 | `equal_distortion` | 243 / 243 | 19.28 | 19.77 |
| 0 | `mse_optimal` | 317 / 317 | 28.98 | 29.84 |
| 1 | `shipped` | 320 / 320 | 29.24 | 30.00 |
| 1 | `uniform` | 320 / 320 | 29.24 | 30.00 |
| 1 | `equal_distortion` | 190 / 190 | 16.94 | 17.72 |
| 1 | `mse_optimal` | 333 / 333 | 29.45 | 30.17 |
| 2 | `shipped` | 320 / 320 | 29.12 | 30.00 |
| 2 | `uniform` | 320 / 320 | 29.12 | 30.00 |
| 2 | `equal_distortion` | 228 / 228 | 18.59 | 19.35 |
| 2 | `mse_optimal` | 319 / 319 | 29.01 | 29.76 |

## 11. Per-seed wall clock

| seed | s |
|---|---|
| 0 | 2165 |
| 1 | 2141 |
| 2 | 2586 |
