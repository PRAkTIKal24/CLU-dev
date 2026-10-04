# gen-navigation — tables, rebuilt from disk by `tables.py`

⛔ Every row carries **both controls**. ⛔ Every per-factor row carries `n_values_realised` (R31, fourth strike).


## 1. ⭐ THE PRE-REGISTERED HEADLINE — `NAV`, and the marginal traverse rates

| pool | n qualifying pairs (per seed) | n_values_realised | **NAV** | store_snap | **linear** (control 1) | deleted_mean (control 2) | deleted_scramble |
|---|---|---|---|---|---|---|---|
| CORNER | 16/16/16 | 1, 2, 3 | **0.0000 ± 0.0000** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| GRID | 40/40/40 | 4, 8, 10, 15 | **0.0000 ± 0.0000** | 0.0083 ± 0.0118 | 0.0083 ± 0.0118 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| ALL | 56/56/56 | 1, 2, 3, 4, 8, 10, 15 | **0.0000 ± 0.0000** | 0.0060 ± 0.0084 | 0.0060 ± 0.0084 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## 2. ⭐⭐ WHY it is zero — the four clauses of `TRAVERSES`, decomposed

(fractions over qualifying pairs; a pair must satisfy **all four**)

| pool | arm | endpoints_read_ok | monotone | visits_intermediate | off_target_stability (pre-reg, vs REQUEST) | …≥0.9 | **TRAVERSES** |
|---|---|---|---|---|---|---|---|
| CORNER | store_snap | 0.000 ± 0.000 | 0.979 ± 0.029 | 0.021 ± 0.029 | 0.681 ± 0.024 | 0.000 ± 0.000 | **0.0000 ± 0.0000** |
| CORNER | linear | 0.000 ± 0.000 | 0.979 ± 0.029 | 0.021 ± 0.029 | 0.681 ± 0.024 | 0.000 ± 0.000 | **0.0000 ± 0.0000** |
| CORNER | deleted_mean | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.088 ± 0.064 | 0.000 ± 0.000 | **0.0000 ± 0.0000** |
| CORNER | deleted_scramble | 0.000 ± 0.000 | 0.938 ± 0.051 | 0.354 ± 0.193 | 0.065 ± 0.011 | 0.000 ± 0.000 | **0.0000 ± 0.0000** |
| GRID | store_snap | 0.517 ± 0.024 | 0.850 ± 0.041 | 0.683 ± 0.066 | 0.625 ± 0.020 | 0.050 ± 0.020 | **0.0083 ± 0.0118** |
| GRID | linear | 0.517 ± 0.024 | 0.833 ± 0.042 | 0.775 ± 0.054 | 0.638 ± 0.015 | 0.050 ± 0.020 | **0.0083 ± 0.0118** |
| GRID | deleted_mean | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.350 ± 0.054 | 0.108 ± 0.015 | 0.000 ± 0.000 | **0.0000 ± 0.0000** |
| GRID | deleted_scramble | 0.017 ± 0.024 | 0.350 ± 0.035 | 0.483 ± 0.012 | 0.131 ± 0.030 | 0.000 ± 0.000 | **0.0000 ± 0.0000** |

## 3. ⭐ ADDENDUM-1 DIAGNOSTICS (⛔ declared post-hoc, ⛔ never the headline)

| pool | arm | endpoint_fidelity (6 factors) | off_target_stability (vs REQUEST) | **off_target_constancy (vs step 0)** | non-target factors that move (of 5) |
|---|---|---|---|---|---|
| CORNER | store_snap | 0.587 ± 0.015 | 0.681 ± 0.024 | **0.983 ± 0.014** | 0.19 ± 0.15 |
| CORNER | linear | 0.587 ± 0.015 | 0.681 ± 0.024 | **0.978 ± 0.019** | 0.19 ± 0.15 |
| CORNER | deleted_mean | 0.142 ± 0.034 | 0.088 ± 0.064 | **1.000 ± 0.000** | 0.00 ± 0.00 |
| CORNER | deleted_scramble | 0.090 ± 0.011 | 0.065 ± 0.011 | **0.974 ± 0.007** | 0.27 ± 0.08 |
| GRID | store_snap | 0.625 ± 0.016 | 0.625 ± 0.020 | **0.861 ± 0.020** | 1.62 ± 0.13 |
| GRID | linear | 0.625 ± 0.016 | 0.638 ± 0.015 | **0.841 ± 0.027** | 1.49 ± 0.19 |
| GRID | deleted_mean | 0.105 ± 0.008 | 0.108 ± 0.015 | **1.000 ± 0.000** | 0.00 ± 0.00 |
| GRID | deleted_scramble | 0.122 ± 0.022 | 0.131 ± 0.030 | **0.663 ± 0.028** | 3.60 ± 0.28 |

## 4. By factor — ⛔ read every row WITH its `n_values_realised`

| pool | factor | n_values_realised | qualifies | pairs/seed | store_snap TRAVERSES | linear TRAVERSES | store visits_intermediate | linear visits_intermediate | identical endpoint codes | interior codes off-endpoints |
|---|---|---|---|---|---|---|---|---|---|---|
| CORNER | floor_hue | 2 | ⚠ switch-only | 8 | ⛔ n/a | ⛔ n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.33 ± 0.26 |
| CORNER | wall_hue | 2 | ⚠ switch-only | 8 | ⛔ n/a | ⛔ n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.00 ± 0.00 |
| CORNER | object_hue | 2 | ⚠ switch-only | 8 | ⛔ n/a | ⛔ n/a | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.458 ± 0.328 | 0.04 ± 0.06 |
| CORNER | scale | 3 | yes | 8 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.750 ± 0.000 | 0.04 ± 0.06 |
| CORNER | shape | 1 | ⛔ **NOT APPLICABLE** | 0 | ⛔ NOT RUN | ⛔ NOT RUN | ⛔ NOT RUN | ⛔ NOT RUN | — | — |
| CORNER | orientation | 3 | yes | 8 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.042 ± 0.059 | 0.042 ± 0.059 | 0.792 ± 0.295 | 0.00 ± 0.00 |
| GRID | floor_hue | 10 | yes | 8 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.542 ± 0.059 | 0.708 ± 0.059 | 0.000 ± 0.000 | 1.54 ± 0.33 |
| GRID | wall_hue | 10 | yes | 8 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.667 ± 0.212 | 0.750 ± 0.204 | 0.000 ± 0.000 | 2.79 ± 0.06 |
| GRID | object_hue | 10 | yes | 8 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.542 ± 0.156 | 0.708 ± 0.156 | 0.000 ± 0.000 | 0.88 ± 0.31 |
| GRID | scale | 8 | yes | 8 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.875 ± 0.102 | 0.875 ± 0.102 | 0.042 ± 0.059 | 0.50 ± 0.27 |
| GRID | shape | 4 | ⚠ switch-only | 8 | ⛔ n/a | ⛔ n/a | 0.792 ± 0.156 | 0.792 ± 0.156 | 0.042 ± 0.059 | 0.46 ± 0.16 |
| GRID | orientation | 15 | yes | 8 | 0.042 ± 0.059 | 0.042 ± 0.059 | 0.792 ± 0.118 | 0.833 ± 0.156 | 0.042 ± 0.059 | 0.46 ± 0.21 |

## 5. Smoothness — does it move smoothly or jump?

| pool | arm | mean step RMSE | jump_ratio (max/mean, 1 = even) | path_efficiency (1 = direct) | distinct codes on path | zero-length steps |
|---|---|---|---|---|---|---|
| CORNER | store_snap | 0.008 ± 0.001 | 7.022 ± 0.405 | 0.932 ± 0.029 | 1.88 ± 0.01 | 7.12 ± 0.01 |
| CORNER | linear | 0.009 ± 0.001 | 1.634 ± 0.061 | 0.853 ± 0.008 | — | 3.20 ± 0.43 |
| CORNER | deleted_mean | 0.000 ± 0.000 | — | — | 1.88 ± 0.01 | 8.00 ± 0.00 |
| CORNER | deleted_scramble | 0.028 ± 0.003 | 6.294 ± 0.449 | 0.801 ± 0.070 | 1.88 ± 0.01 | 7.12 ± 0.01 |
| GRID | store_snap | 0.032 ± 0.001 | 5.627 ± 0.173 | 0.841 ± 0.007 | 3.08 ± 0.11 | 5.92 ± 0.11 |
| GRID | linear | 0.036 ± 0.002 | 1.545 ± 0.026 | 0.747 ± 0.010 | — | 0.17 ± 0.24 |
| GRID | deleted_mean | 0.000 ± 0.000 | — | — | 3.08 ± 0.11 | 8.00 ± 0.00 |
| GRID | deleted_scramble | 0.067 ± 0.007 | 5.060 ± 0.155 | 0.665 ± 0.022 | 3.08 ± 0.11 | 5.92 ± 0.11 |

## 6. The switch-only rows (⛔ excluded from the headline by the prereg)

| pool | factor | n_values_realised | why switch-only | store_snap clean_switch | linear clean_switch | deleted_mean clean_switch |
|---|---|---|---|---|---|---|
| CORNER | floor_hue | 2 | only 2 values realised | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| CORNER | wall_hue | 2 | only 2 values realised | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| CORNER | object_hue | 2 | only 2 values realised | 0.333 ± 0.471 | 0.333 ± 0.471 | 0.000 ± 0.000 |
| GRID | shape | 4 | CATEGORICAL | 0.125 ± 0.102 | 0.125 ± 0.102 | 0.000 ± 0.000 |

## 7. ⛔ The realised channel width AT EVERY INTERMEDIATE POINT

| arm | per-point width | vs R20's 30-bit design constant | informative bits | path information (2·K·log2 M + log2(S+1)) |
|---|---|---|---|---|
| store_snap | 30.0 bits | ⭐ within | 30.0 | 63.17 bits |
| linear | 65536.0 bits | ⛔ EXCEEDS | 65536.0 | 63.17 bits |
| deleted_mean | 30.0 bits | ⭐ within | 0.0 | 63.17 bits |
| deleted_scramble | 30.0 bits | ⭐ within | 0.0 | 63.17 bits |

## 8. ⭐ SENSITIVITY — the fairest possible reading for the store arm

⛔ Post-hoc, declared. The pre-registered `TRAVERSES` conjunction is progressively relaxed to check the negative is not an artefact of one clause.

| pool | arm | (a) endpoints ∧ monotone ∧ visits ∧ stab≥0.9 (**pre-reg**) | (b) drop the stability clause | (c) (b) AND condition on endpoints reading correctly | n pairs in (c)/seed |
|---|---|---|---|---|---|
| CORNER | store_snap | 0.0000 ± 0.0000 | 0.000 ± 0.000 | — | 0.0 ± 0.0 |
| CORNER | linear | 0.0000 ± 0.0000 | 0.000 ± 0.000 | — | 0.0 ± 0.0 |
| CORNER | deleted_scramble | 0.0000 ± 0.0000 | 0.000 ± 0.000 | — | 0.0 ± 0.0 |
| GRID | store_snap | 0.0083 ± 0.0118 | 0.283 ± 0.072 | 0.550 ± 0.147 | 20.7 ± 0.9 |
| GRID | linear | 0.0083 ± 0.0118 | 0.358 ± 0.066 | 0.697 ± 0.144 | 20.7 ± 0.9 |
| GRID | deleted_scramble | 0.0000 ± 0.0000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.7 ± 0.9 |

### 8b. Head-to-head on the conditional set — ⛔ store vs the laundering control

| pool | metric | store_snap | linear | Δ (store − linear) |
|---|---|---|---|---|
| CORNER | visits_intermediate | — | — | — |
| CORNER | monotone | — | — | — |
| CORNER | off_target_constancy | — | — | — |
| GRID | visits_intermediate | 0.550 ± 0.147 | 0.712 ± 0.134 | -0.162 ± 0.027 |
| GRID | monotone | 0.726 ± 0.061 | 0.759 ± 0.105 | -0.033 ± 0.047 |
| GRID | off_target_constancy | 0.828 ± 0.030 | 0.800 ± 0.040 | 0.028 ± 0.016 |

## 9. ⭐⭐ PER-FACTOR ENDPOINT FIDELITY of the 30-bit read (read by the EXTERNAL classifier, ⛔ not the probe)

How often the decoded endpoint image shows the factor value the request asked for. ⛔ This is the **ceiling** on any traversal claim, and it is a cross-instrument replication of R31's mechanism.

| pool | factor | n_values_realised | chance | endpoint fidelity (3 seeds) | vs chance |
|---|---|---|---|---|---|
| CORNER | floor_hue | 2 | 0.500 | 0.996 ± 0.006 | above chance |
| CORNER | wall_hue | 2 | 0.500 | 1.000 ± 0.000 | above chance |
| CORNER | object_hue | 2 | 0.500 | 0.692 ± 0.224 | above chance |
| CORNER | scale | 3 | 0.333 | 0.021 ± 0.029 | ⛔ **BELOW chance** |
| CORNER | shape | 1 | nan | 0.688 ± 0.237 | ⚠ **n/a** — ONE value realised ⇒ a detection rate, not discrimination |
| CORNER | orientation | 3 | 0.333 | 0.158 ± 0.133 | ⛔ **BELOW chance** |
| GRID | floor_hue | 10 | 0.100 | 0.997 ± 0.005 | above chance |
| GRID | wall_hue | 10 | 0.100 | 0.990 ± 0.009 | above chance |
| GRID | object_hue | 10 | 0.100 | 0.701 ± 0.038 | above chance |
| GRID | scale | 8 | 0.125 | 0.229 ± 0.029 | above chance |
| GRID | shape | 4 | 0.250 | 0.583 ± 0.074 | above chance |
| GRID | orientation | 15 | 0.067 | 0.236 ± 0.081 | above chance |
