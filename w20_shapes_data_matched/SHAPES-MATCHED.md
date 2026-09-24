# SHAPES-MATCHED — the one-to-one rival tables (w20 hub/shapes-data-matched)

Every number is read from `SHAPES-MATCHED.json`, which reads only per-corner score files (`scores/…`, paths in the JSON's `sources`). Metric: distinct valid novel tuples per corner (the corner is the unit). Seeds 0/1/2; the seeds column is printed on every row. Margin = 1.0 novel tuple/corner (pre-declared).

## Table M-131k (and M-16k) — rows 3–4 TEST (60 pair + 49 n-way corners)

**Fit pools:** store 131,072 rows (its exact pool) · data-matched rivals 131,072 rows (the SAME pool) · T7's comparators 16,384 rows (= the pool's first 16,384) · `learned_ref_ownpool` 32,768 rows (= the pool's first 32,768) · M-16k stores 16,384 / 32,768 rows (nested prefixes). Rival parameters: 130,433–131,891 (the 131,072 read-path pin); the store's four-part ledger is printed separately and never matched. Ceiling = distinct requested tuples the split never holds.

| arm | role | all corners mean ± sd | per seed | row 3 (pair) | row 4 (n-way) | ceiling (all) | n seeds |
|---|---|---|---|---|---|---|---|
| store K*=32, fit 131,072 rows | THE ARM | 48.006 ±0.965 | 47.495 / 47.404 / 49.119 | 32.350 ±0.815 | 67.177 ±1.320 | 52.183 | 3 |
| store K=5, fit 131,072 rows | THE ARM (K=5) | 45.755 ±2.006 | 43.789 / 45.679 / 47.798 | 30.622 ±1.660 | 64.286 ±2.512 | 52.183 | 3 |
| store K*=32, fit 16,384 rows | M-16k arm | 47.275 ±1.230 | 46.147 / 47.092 / 48.587 | 31.900 ±1.311 | 66.102 ±1.362 | 52.183 | 3 |
| store K=5, fit 16,384 rows | M-16k arm | 45.275 ±2.464 | 43.183 / 44.651 / 47.991 | 30.128 ±2.245 | 63.823 ±2.732 | 52.183 | 3 |
| store K*=32, fit 32,768 rows | M-16k arm | 47.829 ±0.959 | 47.275 / 47.275 / 48.936 | 32.211 ±0.853 | 66.952 ±1.272 | 52.183 | 3 |
| store K=5, fit 32,768 rows | M-16k arm | 45.709 ±2.354 | 43.330 / 45.761 / 48.037 | 30.461 ±1.800 | 64.381 ±3.032 | 52.183 | 3 |
| prototype table | FLOOR, never a rival | 7.685 ±2.386 | 8.229 / 5.073 / 9.752 | 5.167 ±1.468 | 10.769 ±3.567 | 52.183 | 3 |
| MLP 130,514 p, 32,768 rows, 20 ep lr1e-3 (existing learned_ref_ownpool) | EXISTING data-matched-at-32,768 rival — NOT 131,072 | 11.810 ±3.032 | 9.183 / 11.119 / 15.128 | 7.489 ±1.765 | 17.102 ±4.784 | — | 3 |
| MLP, 16,384 rows, 20,480 steps (A1.1) | DIAGNOSTIC | 38.606 ±3.330 | 36.358 / 37.028 / 42.431 | 25.239 ±2.571 | 54.973 ±4.318 | 52.183 | 3 |
| MLP, 32,768 rows, 20,480 steps (A2.3) | DIAGNOSTIC | 39.067 ±3.108 | 37.550 / 37.009 / 42.642 | 25.733 ±2.963 | 55.395 ±3.730 | 52.183 | 3 |
| mlp, 131,072 rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | 39.150 ±3.512 | 36.706 / 37.569 / 43.174 | 25.394 ±3.266 | 55.993 ±4.045 | 52.183 | 3 |
| mlp, 131,072 rows, 10x steps | 10x DIAGNOSTIC | 40.972 ±2.775 | 40.009 / 38.807 / 44.101 | 26.417 ±1.985 | 58.796 ±3.977 | 52.183 | 3 |
| mlp, 16,384 rows, 20 ep (T7's comparator) | EXISTING 16k rival (draft) | 5.535 ±2.486 | 5.193 / 3.239 / 8.174 | 3.878 ±1.229 | 7.565 ±4.381 | 52.183 | 3 |
| lstm, 131,072 rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | 35.676 ±4.423 | 33.972 / 32.358 / 40.697 | 23.467 ±2.774 | 50.626 ±6.573 | 52.183 | 3 |
| lstm, 131,072 rows, 10x steps | 10x DIAGNOSTIC | 35.361 ±3.933 | 33.688 / 32.541 / 39.853 | 22.994 ±2.200 | 50.503 ±6.117 | 52.183 | 3 |
| lstm, 16,384 rows, 20 ep (T7's comparator) | EXISTING 16k rival (draft) | 6.266 ±2.718 | 4.761 / 4.633 / 9.404 | 3.372 ±1.341 | 9.810 ±4.415 | 52.183 | 3 |
| transformer, 131,072 rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | 30.667 ±7.004 | 25.560 / 27.789 / 38.651 | 20.050 ±5.218 | 43.667 ±9.210 | 52.183 | 3 |
| transformer, 131,072 rows, 10x steps | 10x DIAGNOSTIC | 30.128 ±6.752 | 25.303 / 27.239 / 37.844 | 18.783 ±4.086 | 44.020 ±10.018 | 52.183 | 3 |
| transformer, 16,384 rows, 20 ep (T7's comparator) | EXISTING 16k rival (draft) | 1.651 ±1.265 | 1.229 / 0.651 / 3.073 | 1.017 ±0.606 | 2.429 ±2.098 | 52.183 | 3 |
| cont_mlp, 131,072 rows, DECLARED (spoke-A protocol) | DATA-MATCHED RIVAL | 20.434 ±5.594 | 17.917 / 16.541 / 26.844 | 11.722 ±3.806 | 31.102 ±7.822 | 52.183 | 3 |
| cont_mlp, 131,072 rows, 10x steps | 10x DIAGNOSTIC | 34.688 ±7.194 | 28.532 / 32.936 / 42.596 | 22.589 ±4.709 | 49.503 ±10.414 | 52.183 | 3 |
| cont_mlp, 16,384 rows, 20 ep (T7's comparator) | EXISTING 16k rival (draft) | 0.037 ±0.016 | 0.028 / 0.028 / 0.055 | 0.006 ±0.010 | 0.075 ±0.024 | 52.183 | 3 |
| cont_mlp__declared@test30x | cont_mlp__declared@test30x | 33.333 ±5.741 | 27.284 / 34.009 / 38.706 | 21.894 ±5.279 | 47.340 ±6.559 | 52.183 | 3 |
| cont_mlp__declared@test_del_conjunction_live | cont_mlp__declared@test_del_conjunction_live | 17.425 ±0.236 | 17.697 / 17.303 / 17.275 | 9.611 ±0.675 | 26.993 ±1.337 | 52.183 | 3 |
| cont_mlp__declared@test_del_item | cont_mlp__declared@test_del_item | 22.034 ±5.622 | 19.138 / 18.450 / 28.514 | 12.628 ±3.179 | 33.551 ±8.974 | 52.183 | 3 |
| cont_mlp__declared@test_del_value | cont_mlp__declared@test_del_value | 18.086 ±2.838 | 16.303 / 21.358 / 16.596 | 9.606 ±1.677 | 28.469 ±4.645 | 52.183 | 3 |
| lstm__declared@test_del_conjunction_live | lstm__declared@test_del_conjunction_live | 35.713 ±4.021 | 33.064 / 33.734 / 40.339 | 23.233 ±2.830 | 50.993 ±5.490 | 52.183 | 3 |
| lstm__declared@test_del_item | lstm__declared@test_del_item | 35.572 ±4.609 | 33.248 / 32.587 / 40.881 | 23.433 ±3.223 | 50.435 ±6.494 | 52.183 | 3 |
| lstm__declared@test_del_value | lstm__declared@test_del_value | 31.367 ±3.153 | 30.367 / 28.835 / 34.899 | 15.800 ±1.178 | 50.429 ±5.574 | 52.183 | 3 |
| mlp__declared@test30x | mlp__declared@test30x | 40.505 ±3.578 | 41.477 / 36.541 / 43.495 | 25.006 ±4.956 | 59.483 ±2.738 | 52.183 | 3 |
| mlp__declared@test_del_conjunction_live | mlp__declared@test_del_conjunction_live | 38.272 ±2.063 | 36.110 / 38.486 / 40.220 | 25.156 ±2.413 | 54.333 ±1.875 | 52.183 | 3 |
| mlp__declared@test_del_item | mlp__declared@test_del_item | 38.514 ±3.257 | 35.881 / 37.505 / 42.156 | 25.456 ±2.843 | 54.503 ±3.797 | 52.183 | 3 |
| mlp__declared@test_del_value | mlp__declared@test_del_value | 33.557 ±2.435 | 31.945 / 32.367 / 36.358 | 16.922 ±1.542 | 53.925 ±3.530 | 52.183 | 3 |
| transformer__declared@test_del_conjunction_live | transformer__declared@test_del_conjunction_live | 29.988 ±6.968 | 24.725 / 27.349 / 37.890 | 18.711 ±4.951 | 43.796 ±9.514 | 52.183 | 3 |
| transformer__declared@test_del_item | transformer__declared@test_del_item | 30.303 ±6.642 | 25.055 / 28.083 / 37.771 | 19.478 ±4.614 | 43.558 ±9.148 | 52.183 | 3 |
| transformer__declared@test_del_value | transformer__declared@test_del_value | 26.691 ±5.323 | 22.972 / 24.312 / 32.789 | 12.506 ±1.777 | 44.061 ±9.668 | 52.183 | 3 |

### Controls (P2) — must read ≤ 1.0 on every seed

| store | landscape_removed | landscape_shuffled | across_wells (DERANGEMENT) | random_atoms | all collapse |
|---|---|---|---|---|---|
| store_K32_n131072 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | YES |
| store_K5_n131072 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | YES |
| store_K32_n16384 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | YES |
| store_K5_n16384 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | YES |
| store_K32_n32768 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | YES |
| store_K5_n32768 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | YES |

seed 0 derangement perm = `[28, 29, 25, 18, 24, 14, 15, 13, 9, 19, 7, 20, 5, 10, 3, 26, 12, 4, 0, 6, 27, 30, 11, 31, 8, 17, 22, 1, 23, 21, 2, 16]` (fixed points 0; the historical T7 permutation had 0 fixed point(s))

seed 1 derangement perm = `[2, 24, 4, 14, 10, 21, 3, 9, 30, 22, 15, 1, 13, 29, 31, 25, 7, 11, 26, 12, 28, 0, 17, 5, 27, 23, 16, 6, 20, 8, 19, 18]` (fixed points 0; the historical T7 permutation had 0 fixed point(s))

seed 2 derangement perm = `[20, 31, 17, 16, 15, 2, 8, 6, 26, 22, 9, 7, 0, 10, 25, 12, 19, 18, 29, 5, 27, 23, 1, 24, 28, 21, 30, 11, 3, 4, 14, 13]` (fixed points 0; the historical T7 permutation had 2 fixed point(s))

### P1 — convergence (DECLARED vs 10× steps), per family per seed

| family | seed | DECLARED | TENX | TENX − DECLARED | P1 | carried |
|---|---|---|---|---|---|---|
| mlp | 0 | 36.706 | 40.009 | 3.303 | UNDER-TRAINED: TENX carries | tenx |
| mlp | 1 | 37.569 | 38.807 | 1.239 | UNDER-TRAINED: TENX carries | tenx |
| mlp | 2 | 43.174 | 44.101 | 0.927 | PASS | declared |
| lstm | 0 | 33.972 | 33.688 | -0.284 | PASS | declared |
| lstm | 1 | 32.358 | 32.541 | 0.183 | PASS | declared |
| lstm | 2 | 40.697 | 39.853 | -0.844 | PASS | declared |
| transformer | 0 | 25.560 | 25.303 | -0.257 | PASS | declared |
| transformer | 1 | 27.789 | 27.239 | -0.550 | PASS | declared |
| transformer | 2 | 38.651 | 37.844 | -0.807 | PASS | declared |
| cont_mlp | 0 | 17.917 | 28.532 | 10.615 | UNDER-TRAINED: TENX carries | tenx |
| cont_mlp | 1 | 16.541 | 32.936 | 16.394 | UNDER-TRAINED: TENX carries | tenx |
| cont_mlp | 2 | 26.844 | 42.596 | 15.752 | UNDER-TRAINED: TENX carries | tenx |

### P3 — THE DATA-MATCHED MARGIN (store − strongest data-matched rival, paired per corner)

| store | seed | strongest rival | store | rival | Δ | CI95 | won/tied/lost |
|---|---|---|---|---|---|---|---|
| store_K32_n131072 | 0 | mlp__tenx@test | 47.495 | 40.009 | **+7.486** | [5.47, 9.65] | 76/11/22 |
| store_K32_n131072 | 1 | mlp__tenx@test | 47.404 | 38.807 | **+8.596** | [6.80, 10.46] | 84/9/16 |
| store_K32_n131072 | 2 | mlp__tenx@test | 49.119 | 44.101 | **+5.018** | [3.73, 6.41] | 69/27/13 |
| store_K32_n131072 | **mean** | | | | **+7.034** | | verdict **SHRINKS** (draft: +39…+49) |
| store_K5_n131072 | 0 | mlp__tenx@test | 43.789 | 40.009 | **+3.780** | [2.15, 5.48] | 62/10/37 |
| store_K5_n131072 | 1 | mlp__tenx@test | 45.679 | 38.807 | **+6.872** | [5.26, 8.55] | 84/8/17 |
| store_K5_n131072 | 2 | mlp__tenx@test | 47.798 | 44.101 | **+3.697** | [2.51, 4.96] | 63/31/15 |
| store_K5_n131072 | **mean** | | | | **+4.783** | | verdict **SHRINKS** (draft: +39…+49) |
| store_K32_n16384 | 0 | mlp__tenx@test | 46.147 | 40.009 | **+6.138** | [4.23, 8.11] | 72/14/23 |
| store_K32_n16384 | 1 | mlp__tenx@test | 47.092 | 38.807 | **+8.284** | [6.68, 9.95] | 90/8/11 |
| store_K32_n16384 | 2 | mlp__tenx@test | 48.587 | 44.101 | **+4.486** | [3.31, 5.76] | 65/30/14 |
| store_K32_n16384 | **mean** | | | | **+6.303** | | verdict **SHRINKS** (draft: +39…+49) |
| store_K5_n16384 | 0 | mlp__tenx@test | 43.183 | 40.009 | **+3.174** | [1.59, 4.79] | 63/8/38 |
| store_K5_n16384 | 1 | mlp__tenx@test | 44.651 | 38.807 | **+5.844** | [4.22, 7.55] | 84/6/19 |
| store_K5_n16384 | 2 | mlp__tenx@test | 47.991 | 44.101 | **+3.890** | [2.79, 5.10] | 69/31/9 |
| store_K5_n16384 | **mean** | | | | **+4.303** | | verdict **SHRINKS** (draft: +39…+49) |
| store_K32_n32768 | 0 | mlp__tenx@test | 47.275 | 40.009 | **+7.266** | [5.24, 9.45] | 73/12/24 |
| store_K32_n32768 | 1 | mlp__tenx@test | 47.275 | 38.807 | **+8.468** | [6.80, 10.19] | 86/11/12 |
| store_K32_n32768 | 2 | mlp__tenx@test | 48.936 | 44.101 | **+4.835** | [3.63, 6.16] | 69/29/11 |
| store_K32_n32768 | **mean** | | | | **+6.856** | | verdict **SHRINKS** (draft: +39…+49) |
| store_K5_n32768 | 0 | mlp__tenx@test | 43.330 | 40.009 | **+3.321** | [1.80, 4.92] | 64/11/34 |
| store_K5_n32768 | 1 | mlp__tenx@test | 45.761 | 38.807 | **+6.954** | [5.26, 8.74] | 82/11/16 |
| store_K5_n32768 | 2 | mlp__tenx@test | 48.037 | 44.101 | **+3.936** | [2.78, 5.19] | 68/29/12 |
| store_K5_n32768 | **mean** | | | | **+4.737** | | verdict **SHRINKS** (draft: +39…+49) |

### Margins against every rival column (paired Δ per seed)

| store | rival | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| store_K32_n131072 | mlp (data-matched, carried) | +7.486 | +8.596 | +5.945 |
| store_K32_n131072 | lstm (data-matched, carried) | +13.523 | +15.046 | +8.422 |
| store_K32_n131072 | transformer (data-matched, carried) | +21.936 | +19.615 | +10.468 |
| store_K32_n131072 | cont_mlp (data-matched, carried) | +18.963 | +14.468 | +6.523 |
| store_K32_n131072 | w12comp_mlp__as_trained | +42.303 | +44.165 | +40.945 |
| store_K32_n131072 | w12comp_lstm__as_trained | +42.734 | +42.771 | +39.716 |
| store_K32_n131072 | w12comp_transformer__as_trained | +46.266 | +46.752 | +46.046 |
| store_K32_n131072 | w12comp_cont_mlp__as_trained | +47.468 | +47.376 | +49.064 |
| store_K32_n131072 | EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | +38.312 | +36.284 | +33.991 |
| store_K32_n131072 | mlp__declared@test16k_stepmatched | +11.138 | +10.376 | +6.688 |
| store_K32_n131072 | mlp__declared@test32k_stepmatched | +9.945 | +10.394 | +6.477 |
| store_K32_n131072 | table_FLOOR | +39.266 | +42.330 | +39.367 |
| store_K5_n131072 | mlp (data-matched, carried) | +3.780 | +6.872 | +4.624 |
| store_K5_n131072 | lstm (data-matched, carried) | +9.817 | +13.321 | +7.101 |
| store_K5_n131072 | transformer (data-matched, carried) | +18.229 | +17.890 | +9.147 |
| store_K5_n131072 | cont_mlp (data-matched, carried) | +15.257 | +12.743 | +5.202 |
| store_K5_n131072 | w12comp_mlp__as_trained | +38.596 | +42.440 | +39.624 |
| store_K5_n131072 | w12comp_lstm__as_trained | +39.028 | +41.046 | +38.394 |
| store_K5_n131072 | w12comp_transformer__as_trained | +42.560 | +45.028 | +44.725 |
| store_K5_n131072 | w12comp_cont_mlp__as_trained | +43.761 | +45.651 | +47.743 |
| store_K5_n131072 | EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | +34.606 | +34.560 | +32.670 |
| store_K5_n131072 | mlp__declared@test16k_stepmatched | +7.431 | +8.651 | +5.367 |
| store_K5_n131072 | mlp__declared@test32k_stepmatched | +6.239 | +8.670 | +5.156 |
| store_K5_n131072 | table_FLOOR | +35.560 | +40.606 | +38.046 |
| store_K32_n16384 | mlp (data-matched, carried) | +6.138 | +8.284 | +5.413 |
| store_K32_n16384 | lstm (data-matched, carried) | +12.174 | +14.734 | +7.890 |
| store_K32_n16384 | transformer (data-matched, carried) | +20.587 | +19.303 | +9.936 |
| store_K32_n16384 | cont_mlp (data-matched, carried) | +17.615 | +14.156 | +5.991 |
| store_K32_n16384 | w12comp_mlp__as_trained | +40.954 | +43.853 | +40.413 |
| store_K32_n16384 | w12comp_lstm__as_trained | +41.385 | +42.459 | +39.183 |
| store_K32_n16384 | w12comp_transformer__as_trained | +44.917 | +46.440 | +45.514 |
| store_K32_n16384 | w12comp_cont_mlp__as_trained | +46.119 | +47.064 | +48.532 |
| store_K32_n16384 | EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | +36.963 | +35.972 | +33.459 |
| store_K32_n16384 | mlp__declared@test16k_stepmatched | +9.789 | +10.064 | +6.156 |
| store_K32_n16384 | mlp__declared@test32k_stepmatched | +8.596 | +10.083 | +5.945 |
| store_K32_n16384 | table_FLOOR | +37.917 | +42.018 | +38.835 |
| store_K5_n16384 | mlp (data-matched, carried) | +3.174 | +5.844 | +4.817 |
| store_K5_n16384 | lstm (data-matched, carried) | +9.211 | +12.294 | +7.294 |
| store_K5_n16384 | transformer (data-matched, carried) | +17.624 | +16.862 | +9.339 |
| store_K5_n16384 | cont_mlp (data-matched, carried) | +14.651 | +11.716 | +5.394 |
| store_K5_n16384 | w12comp_mlp__as_trained | +37.991 | +41.413 | +39.817 |
| store_K5_n16384 | w12comp_lstm__as_trained | +38.422 | +40.018 | +38.587 |
| store_K5_n16384 | w12comp_transformer__as_trained | +41.954 | +44.000 | +44.917 |
| store_K5_n16384 | w12comp_cont_mlp__as_trained | +43.156 | +44.624 | +47.936 |
| store_K5_n16384 | EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | +34.000 | +33.532 | +32.862 |
| store_K5_n16384 | mlp__declared@test16k_stepmatched | +6.826 | +7.624 | +5.560 |
| store_K5_n16384 | mlp__declared@test32k_stepmatched | +5.633 | +7.642 | +5.349 |
| store_K5_n16384 | table_FLOOR | +34.954 | +39.578 | +38.239 |
| store_K32_n32768 | mlp (data-matched, carried) | +7.266 | +8.468 | +5.761 |
| store_K32_n32768 | lstm (data-matched, carried) | +13.303 | +14.917 | +8.239 |
| store_K32_n32768 | transformer (data-matched, carried) | +21.716 | +19.486 | +10.284 |
| store_K32_n32768 | cont_mlp (data-matched, carried) | +18.743 | +14.339 | +6.339 |
| store_K32_n32768 | w12comp_mlp__as_trained | +42.083 | +44.037 | +40.761 |
| store_K32_n32768 | w12comp_lstm__as_trained | +42.514 | +42.642 | +39.532 |
| store_K32_n32768 | w12comp_transformer__as_trained | +46.046 | +46.624 | +45.862 |
| store_K32_n32768 | w12comp_cont_mlp__as_trained | +47.248 | +47.248 | +48.881 |
| store_K32_n32768 | EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | +38.092 | +36.156 | +33.807 |
| store_K32_n32768 | mlp__declared@test16k_stepmatched | +10.917 | +10.248 | +6.505 |
| store_K32_n32768 | mlp__declared@test32k_stepmatched | +9.725 | +10.266 | +6.294 |
| store_K32_n32768 | table_FLOOR | +39.046 | +42.202 | +39.183 |
| store_K5_n32768 | mlp (data-matched, carried) | +3.321 | +6.954 | +4.862 |
| store_K5_n32768 | lstm (data-matched, carried) | +9.358 | +13.404 | +7.339 |
| store_K5_n32768 | transformer (data-matched, carried) | +17.771 | +17.972 | +9.385 |
| store_K5_n32768 | cont_mlp (data-matched, carried) | +14.798 | +12.826 | +5.440 |
| store_K5_n32768 | w12comp_mlp__as_trained | +38.138 | +42.523 | +39.862 |
| store_K5_n32768 | w12comp_lstm__as_trained | +38.569 | +41.128 | +38.633 |
| store_K5_n32768 | w12comp_transformer__as_trained | +42.101 | +45.110 | +44.963 |
| store_K5_n32768 | w12comp_cont_mlp__as_trained | +43.303 | +45.734 | +47.982 |
| store_K5_n32768 | EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | +34.147 | +34.642 | +32.908 |
| store_K5_n32768 | mlp__declared@test16k_stepmatched | +6.972 | +8.734 | +5.606 |
| store_K5_n32768 | mlp__declared@test32k_stepmatched | +5.780 | +8.752 | +5.394 |
| store_K5_n32768 | table_FLOOR | +35.101 | +40.688 | +38.284 |

### P3 against the rival at its best over EVERY fit that ran (DECLARED, 10×, 30×)

| seed | best rival fit | store | rival | Δ | CI95 | won/tied/lost |
|---|---|---|---|---|---|---|
| 0 | mlp__declared@test30x | 47.495 | 41.477 | **+6.018** | [3.83, 8.33] | 67/14/28 |
| 1 | mlp__tenx@test | 47.404 | 38.807 | **+8.596** | [6.80, 10.46] | 84/9/16 |
| 2 | mlp__tenx@test | 49.119 | 44.101 | **+5.018** | [3.73, 6.41] | 69/27/13 |
| **mean** | | | | **+6.544** | | **SHRINKS** |

### The draft's margin, recomputed from the same score files (store K*=32 − 16k comparator)

| comparator (16,384 rows) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| mlp | +42.303 | +44.165 | +40.945 |
| lstm | +42.734 | +42.771 | +39.716 |
| transformer | +46.266 | +46.752 | +46.046 |
| cont_mlp | +47.468 | +47.376 | +49.064 |

## Matched deletion (Head AMENDMENT-2 definition; rows 3–4 TEST survivor corners; pool 131,072)

⛔ 'c still composes after forgetting c' -- not computed, not printed.

| scope | seed | deleted rows | (a) store max|mode dev| | (b1) statistic dev | (b1) statistic bytes | (b2) rendered rel dev | (c) DISPLAY refusal (deleted / false on survivors) | (d) survivors: before → after (Δ) | (d) downdate − refit-without |
|---|---|---|---|---|---|---|---|---|---|
| value | 0 | 11044 | 5.96e-08 | 0.0e+00 | 0.97 GB | 1.85e-09 | 1.000 / 0.000 | 50.921 → 50.865 (-0.056) | +0.000 |
| value | 1 | 11044 | 4.82e-08 | 0.0e+00 | 0.97 GB | 1.42e-09 | 1.000 / 0.000 | 50.528 → 50.854 (+0.326) | +0.000 |
| value | 2 | 11044 | 2.01e-08 | 0.0e+00 | 0.97 GB | 1.22e-09 | 1.000 / 0.000 | 52.607 → 52.000 (-0.607) | +0.000 |
| conjunction_live | 0 | 5511 | 5.38e-07 | 0.0e+00 | 0.97 GB | 3.29e-08 | — / 0.000 | 47.495 → 41.367 (-6.128) | +0.000 |
| conjunction_live | 1 | 5511 | 1.07e-08 | 0.0e+00 | 0.97 GB | 7.81e-08 | — / 0.000 | 47.404 → 40.972 (-6.431) | +0.000 |
| conjunction_live | 2 | 5511 | 6.25e-08 | 0.0e+00 | 0.97 GB | 2.03e-09 | — / 0.000 | 49.119 → 42.817 (-6.303) | +0.000 |
| item | 0 | 1311 | 2.34e-07 | 0.0e+00 | 0.97 GB | 5.83e-10 | N/A (no hole exists) | 47.495 → 47.477 (-0.018) | +0.000 |
| item | 1 | 1311 | 1.49e-08 | 0.0e+00 | 0.97 GB | 9.74e-10 | N/A (no hole exists) | 47.404 → 47.294 (-0.110) | +0.000 |
| item | 2 | 1311 | 3.16e-08 | 0.0e+00 | 0.97 GB | 6.87e-10 | N/A (no hole exists) | 49.119 → 49.110 (-0.009) | +0.000 |

| scope | seed | family | rival as-trained → retrain-without (survivors) | ours after | ours − rival retrain-without | CI95 | rival s | rival rows read | rival bytes rewritten |
|---|---|---|---|---|---|---|---|---|---|
| value | 0 | mlp | 39.607 → 39.124 | 50.865 | **+11.742** | [9.29, 14.34] | 547.7 | 120028 | 522056 |
| value | 1 | mlp | 40.449 → 39.640 | 50.854 | **+11.213** | [8.73, 13.80] | 677.4 | 120028 | 522056 |
| value | 2 | mlp | 46.315 → 44.528 | 52.000 | **+7.472** | [5.64, 9.46] | 729.8 | 120028 | 522056 |
| value | 0 | lstm | 36.685 → 37.191 | 50.865 | **+13.674** | [10.12, 17.31] | 757.5 | 120028 | 522884 |
| value | 1 | lstm | 34.831 → 35.315 | 50.854 | **+15.539** | [12.52, 18.78] | 712.4 | 120028 | 522884 |
| value | 2 | lstm | 43.303 → 42.742 | 52.000 | **+9.258** | [6.88, 11.81] | 1016.0 | 120028 | 522884 |
| value | 0 | transformer | 27.933 → 27.798 | 50.865 | **+23.067** | [20.16, 26.14] | 1298.0 | 120028 | 521732 |
| value | 1 | transformer | 30.090 → 29.775 | 50.854 | **+21.079** | [18.13, 24.18] | 1772.2 | 120028 | 521732 |
| value | 2 | transformer | 40.865 → 40.157 | 52.000 | **+11.843** | [9.11, 14.69] | 1705.3 | 120028 | 521732 |
| value | 0 | cont_mlp | 21.393 → 19.933 | 50.865 | **+30.933** | [27.69, 34.24] | 695.9 | 120028 | 527564 |
| value | 1 | cont_mlp | 20.213 → 26.157 | 50.854 | **+24.697** | [21.81, 27.69] | 760.1 | 120028 | 527564 |
| value | 2 | cont_mlp | 29.652 → 19.393 | 52.000 | **+32.607** | [29.42, 35.82] | 673.3 | 120028 | 527564 |
| conjunction_live | 0 | mlp | 36.706 → 36.110 | 41.367 | **+5.257** | [2.09, 8.49] | 814.3 | 125561 | 522056 |
| conjunction_live | 1 | mlp | 37.569 → 38.486 | 40.972 | **+2.486** | [-0.94, 6.02] | 805.6 | 125561 | 522056 |
| conjunction_live | 2 | mlp | 43.174 → 40.220 | 42.817 | **+2.596** | [-0.50, 5.67] | 746.6 | 125561 | 522056 |
| conjunction_live | 0 | lstm | 33.972 → 33.064 | 41.367 | **+8.303** | [4.31, 12.33] | 772.8 | 125561 | 522884 |
| conjunction_live | 1 | lstm | 32.358 → 33.734 | 40.972 | **+7.239** | [3.50, 11.06] | 767.5 | 125561 | 522884 |
| conjunction_live | 2 | lstm | 40.697 → 40.339 | 42.817 | **+2.477** | [-0.87, 5.82] | 1198.2 | 125561 | 522884 |
| conjunction_live | 0 | transformer | 25.560 → 24.725 | 41.367 | **+16.642** | [12.85, 20.41] | 1255.8 | 125561 | 521732 |
| conjunction_live | 1 | transformer | 27.789 → 27.349 | 40.972 | **+13.624** | [9.89, 17.42] | 1987.7 | 125561 | 521732 |
| conjunction_live | 2 | transformer | 38.651 → 37.890 | 42.817 | **+4.927** | [1.29, 8.57] | 2064.5 | 125561 | 521732 |
| conjunction_live | 0 | cont_mlp | 17.917 → 17.697 | 41.367 | **+23.670** | [20.24, 27.24] | 922.3 | 125561 | 527564 |
| conjunction_live | 1 | cont_mlp | 16.541 → 17.303 | 40.972 | **+23.670** | [20.20, 27.14] | 859.6 | 125561 | 527564 |
| conjunction_live | 2 | cont_mlp | 26.844 → 17.275 | 42.817 | **+25.541** | [22.01, 29.11] | 712.3 | 125561 | 527564 |
| item | 0 | mlp | 36.706 → 35.881 | 47.477 | **+11.596** | [9.34, 13.87] | 884.6 | 129761 | 522056 |
| item | 1 | mlp | 37.569 → 37.505 | 47.294 | **+9.789** | [7.74, 11.90] | 922.8 | 129761 | 522056 |
| item | 2 | mlp | 43.174 → 42.156 | 49.110 | **+6.954** | [5.34, 8.66] | 742.5 | 129761 | 522056 |
| item | 0 | lstm | 33.972 → 33.248 | 47.477 | **+14.229** | [11.08, 17.43] | 773.2 | 129761 | 522884 |
| item | 1 | lstm | 32.358 → 32.587 | 47.294 | **+14.706** | [12.13, 17.32] | 1093.8 | 129761 | 522884 |
| item | 2 | lstm | 40.697 → 40.881 | 49.110 | **+8.229** | [6.30, 10.18] | 1034.6 | 129761 | 522884 |
| item | 0 | transformer | 25.560 → 25.055 | 47.477 | **+22.422** | [19.72, 25.20] | 1246.0 | 129761 | 521732 |
| item | 1 | transformer | 27.789 → 28.083 | 47.294 | **+19.211** | [16.49, 21.95] | 1754.9 | 129761 | 521732 |
| item | 2 | transformer | 38.651 → 37.771 | 49.110 | **+11.339** | [8.90, 13.90] | 1562.8 | 129761 | 521732 |
| item | 0 | cont_mlp | 17.917 → 19.138 | 47.477 | **+28.339** | [25.53, 31.13] | 817.2 | 129761 | 527564 |
| item | 1 | cont_mlp | 16.541 → 18.450 | 47.294 | **+28.844** | [26.06, 31.61] | 862.4 | 129761 | 527564 |
| item | 2 | cont_mlp | 26.844 → 28.514 | 49.110 | **+20.596** | [17.63, 23.52] | 629.7 | 129761 | 527564 |

### LACE / PoE re-fit at 131,072 rows (Head AMENDMENT-1 i)

⛔ Scope, verbatim: **LACE/PoE as instantiated in this feature space at their own declared budgets -- NOT a refutation of LACE in general**. Fit rows: 130,560 fit + 512 val (w15: 32,256 + 512).

| leg | arm | per seed (novel/corner) |
|---|---|---|
| shipped | lace | s0 0.018 / s1 0.046 / s2 0.073 |
| shipped | poe | s0 0.000 / s1 0.000 / s2 0.000 |
| val | lace | s0 0.042 |
| val | poe | s0 0.000 |

### LACE / PoE

⛔ Scope, verbatim: **LACE/PoE as instantiated in this feature space at their own declared budgets -- NOT a refutation of LACE in general**.

w15 (32,256 fit rows): {"LACE": {"0": {"all": 0.1651376146788991, "n_corners": 109, "row3_pair": 0.11666666666666667, "row4_nway": 0.22448979591836735}, "1": {"all": 0.07339449541284404, "n_corners": 109, "row3_pair": 0.03333333333333333, "row4_nway": 0.12244897959183673}, "2": {"all": 0.22935779816513763, "n_corners": 109, "row3_pair": 0.26666666666666666, "row4_nway": 0.1836734693877551}}, "PoE": {"0": {"all": 0.009174311926605505, "n_corners": 109, "row3_pair": 0.0, "row4_nway": 0.02040816326530612}, "1": {"all": 0.0, "n_corners": 109, "row3_pair": 0.0, "row4_nway": 0.0}, "2": {"all": 0.0, "n_corners": 109, "row3_pair": 0.0, "row4_nway": 0.0}}}

## The ladder (free replication; rung pools 131,072 rows)

### rung n=2

| arm | mean ± sd | per seed | n seeds |
|---|---|---|---|
| EXISTING_ladder_store_K5_n131072 | 30.539 ±1.658 | 28.633 / 31.650 / 31.333 | 3 |
| EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | 7.483 ±1.869 | 5.567 / 7.583 / 9.300 | 3 |
| EXISTING_spokeA_cont_mlp@118k_32256montero | 4.994 ±1.376 | 3.883 / 4.567 / 6.533 | 3 |
| EXISTING_spokeA_lstm@118k_32256montero | 20.833 ±2.526 | 19.417 / 19.333 / 23.750 | 3 |
| EXISTING_spokeA_mlp@118k_32256montero | 20.783 ±3.329 | 17.017 / 22.000 / 23.333 | 3 |
| EXISTING_spokeA_transformer@118k_32256montero | 17.594 ±5.960 | 12.750 / 15.783 / 24.250 | 3 |
| cont_mlp__declared@rung2 | 11.094 ±0.619 | 10.383 / 11.517 / 11.383 | 3 |
| cont_mlp__tenx@rung2 | 22.717 ±1.887 | 21.333 / 21.950 / 24.867 | 3 |
| lstm__declared@rung2 | 23.361 ±3.407 | 20.550 / 22.383 / 27.150 | 3 |
| lstm__tenx@rung2 | 21.850  | 21.850 | 1 |
| mlp__declared@rung2 | 25.639 ±3.429 | 22.450 / 25.200 / 29.267 | 3 |
| mlp__tenx@rung2 | 27.511 ±2.121 | 27.167 / 25.583 / 29.783 | 3 |
| store_K32_n131072 | 32.644 ±0.925 | 31.800 / 32.500 / 33.633 | 3 |
| store_K32_n131072_REBUILT | 32.644 ±0.925 | 31.800 / 32.500 / 33.633 | 3 |
| store_K32_n16384 | 32.083 ±0.525 | 31.550 / 32.100 / 32.600 | 3 |
| store_K32_n32768 | 32.150 ±0.917 | 31.350 / 31.950 / 33.150 | 3 |
| store_K5_n16384 | 30.400 ±1.235 | 29.200 / 30.333 / 31.667 | 3 |
| store_K5_n32768 | 31.122 ±0.795 | 30.450 / 30.917 / 32.000 | 3 |
| table_FLOOR | 5.167 ±1.468 | 5.833 / 3.483 / 6.183 | 3 |
| transformer__declared@rung2 | 19.783 ±5.181 | 16.417 / 17.183 / 25.750 | 3 |
| transformer__tenx@rung2 | 16.217  | 16.217 | 1 |

store_K32_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+4.633', '+6.917', '+3.850'], mean **+5.133** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+12.383', '+10.500', '+9.383']

EXISTING_ladder_store_K5_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+1.467', '+6.067', '+1.550'], mean **+3.028** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+9.217', '+9.650', '+7.083']

store_K32_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+4.383', '+6.517', '+2.817'], mean **+4.572** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+12.133', '+10.100', '+8.350']

store_K5_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+2.033', '+4.750', '+1.883'], mean **+2.889** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+9.783', '+8.333', '+7.417']

store_K32_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+4.183', '+6.367', '+3.367'], mean **+4.639** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+11.933', '+9.950', '+8.900']

store_K5_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+3.283', '+5.333', '+2.217'], mean **+3.611** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+11.033', '+8.917', '+7.750']

### rung n=3

| arm | mean ± sd | per seed | n seeds |
|---|---|---|---|
| EXISTING_ladder_store_K5_n131072 | 59.750 ±2.976 | 56.950 / 59.425 / 62.875 | 3 |
| EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | 11.794 ±4.093 | 8.217 / 10.908 / 16.258 | 3 |
| EXISTING_spokeA_cont_mlp@118k_32256montero | 9.372 ±3.174 | 6.333 / 9.117 / 12.667 | 3 |
| EXISTING_spokeA_lstm@118k_32256montero | 43.503 ±3.726 | 44.433 / 39.400 / 46.675 | 3 |
| EXISTING_spokeA_mlp@118k_32256montero | 42.175 ±7.290 | 35.208 / 41.567 / 49.750 | 3 |
| EXISTING_spokeA_transformer@118k_32256montero | 38.231 ±12.187 | 28.800 / 33.900 / 51.992 | 3 |
| cont_mlp__declared@rung3 | 22.133 ±6.626 | 16.008 / 21.225 / 29.167 | 3 |
| cont_mlp__tenx@rung3 | 35.078 ±9.515 | 24.125 / 41.300 / 39.808 | 3 |
| lstm__declared@rung3 | 38.181 ±2.399 | 36.400 / 37.233 / 40.908 | 3 |
| mlp__declared@rung3 | 47.297 ±4.045 | 43.558 / 46.742 / 51.592 | 3 |
| mlp__tenx@rung3 | 50.431 ±5.866 | 47.258 / 46.833 / 57.200 | 3 |
| store_K32_n131072 | 61.614 ±1.341 | 60.150 / 62.783 / 61.908 | 3 |
| store_K32_n131072_REBUILT | 61.614 ±1.341 | 60.150 / 62.783 / 61.908 | 3 |
| store_K32_n16384 | 62.153 ±1.814 | 60.058 / 63.158 / 63.242 | 3 |
| store_K32_n32768 | 61.969 ±2.436 | 59.175 / 63.092 / 63.642 | 3 |
| store_K5_n16384 | 57.358 ±2.434 | 54.550 / 58.867 / 58.658 | 3 |
| store_K5_n32768 | 58.142 ±1.781 | 57.392 / 60.175 / 56.858 | 3 |
| table_FLOOR | 9.528 ±2.357 | 8.433 / 7.917 / 12.233 | 3 |
| transformer__declared@rung3 | 27.275 ±2.871 | 26.083 / 25.192 / 30.550 | 3 |

store_K32_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+12.892', '+15.950', '+4.708'], mean **+11.183** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+15.717', '+21.217', '+9.917']

EXISTING_ladder_store_K5_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+9.692', '+12.592', '+5.675'], mean **+9.319** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+12.517', '+17.858', '+10.883']

store_K32_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+12.800', '+16.325', '+6.042'], mean **+11.722** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+15.625', '+21.592', '+11.250']

store_K5_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+7.292', '+12.033', '+1.458'], mean **+6.928** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+10.117', '+17.300', '+6.667']

store_K32_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+11.917', '+16.258', '+6.442'], mean **+11.539** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+14.742', '+21.525', '+11.650']

store_K5_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+10.133', '+13.342', '-0.342'], mean **+7.711** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+12.958', '+18.608', '+4.867']

### rung n=4

| arm | mean ± sd | per seed | n seeds |
|---|---|---|---|
| EXISTING_ladder_store_K5_n131072 | 62.708 ±2.054 | 60.933 / 62.233 / 64.958 | 3 |
| EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | 13.281 ±5.153 | 8.542 / 12.533 / 18.767 | 3 |
| EXISTING_spokeA_cont_mlp@118k_32256montero | 9.372 ±3.174 | 6.333 / 9.117 / 12.667 | 3 |
| EXISTING_spokeA_lstm@118k_32256montero | 43.503 ±3.726 | 44.433 / 39.400 / 46.675 | 3 |
| EXISTING_spokeA_mlp@118k_32256montero | 42.175 ±7.290 | 35.208 / 41.567 / 49.750 | 3 |
| EXISTING_spokeA_transformer@118k_32256montero | 38.231 ±12.187 | 28.800 / 33.900 / 51.992 | 3 |
| cont_mlp__declared@rung4 | 27.275 ±7.927 | 21.000 / 24.642 / 36.183 | 3 |
| cont_mlp__tenx@rung4 | 39.975 ±8.741 | 29.883 / 45.158 / 44.883 | 3 |
| lstm__declared@rung4 | 45.322 ±7.209 | 42.492 / 39.958 / 53.517 | 3 |
| mlp__declared@rung4 | 50.450 ±6.149 | 45.408 / 48.642 / 57.300 | 3 |
| mlp__tenx@rung4 | 53.536 ±5.668 | 50.517 / 50.017 / 60.075 | 3 |
| store_K32_n131072 | 66.086 ±1.007 | 65.242 / 65.817 / 67.200 | 3 |
| store_K32_n131072_REBUILT | 66.086 ±1.007 | 65.242 / 65.817 / 67.200 | 3 |
| store_K32_n16384 | 65.639 ±1.171 | 64.725 / 65.233 / 66.958 | 3 |
| store_K32_n32768 | 65.500 ±0.581 | 64.992 / 65.375 / 66.133 | 3 |
| store_K5_n16384 | 61.906 ±2.738 | 59.208 / 61.825 / 64.683 | 3 |
| store_K5_n32768 | 62.475 ±1.121 | 61.342 / 62.500 / 63.583 | 3 |
| table_FLOOR | 9.528 ±2.357 | 8.433 / 7.917 / 12.233 | 3 |
| transformer__declared@rung4 | 38.103 ±9.314 | 31.792 / 33.717 / 48.800 | 3 |

store_K32_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+14.725', '+15.800', '+7.125'], mean **+12.550** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+20.808', '+24.250', '+15.208']

EXISTING_ladder_store_K5_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+10.417', '+12.217', '+4.883'], mean **+9.172** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+16.500', '+20.667', '+12.967']

store_K32_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+14.208', '+15.217', '+6.883'], mean **+12.103** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+20.292', '+23.667', '+14.967']

store_K5_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+8.692', '+11.808', '+4.608'], mean **+8.369** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+14.775', '+20.258', '+12.692']

store_K32_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+14.475', '+15.358', '+6.058'], mean **+11.964** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+20.558', '+23.808', '+14.142']

store_K5_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+10.825', '+12.483', '+3.508'], mean **+8.939** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+16.908', '+20.933', '+11.592']

### rung n=5

| arm | mean ± sd | per seed | n seeds |
|---|---|---|---|
| EXISTING_ladder_store_K5_n131072 | 64.719 ±1.594 | 63.242 / 64.508 / 66.408 | 3 |
| EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | 14.961 ±4.930 | 9.867 / 15.308 / 19.708 | 3 |
| EXISTING_spokeA_cont_mlp@118k_32256montero | 9.372 ±3.174 | 6.333 / 9.117 / 12.667 | 3 |
| EXISTING_spokeA_lstm@118k_32256montero | 43.503 ±3.726 | 44.433 / 39.400 / 46.675 | 3 |
| EXISTING_spokeA_mlp@118k_32256montero | 42.175 ±7.290 | 35.208 / 41.567 / 49.750 | 3 |
| EXISTING_spokeA_transformer@118k_32256montero | 38.231 ±12.187 | 28.800 / 33.900 / 51.992 | 3 |
| cont_mlp__declared@rung5 | 30.650 ±8.043 | 21.817 / 32.583 / 37.550 | 3 |
| cont_mlp__tenx@rung5 | 48.103 ±7.196 | 40.192 / 49.858 / 54.258 | 3 |
| lstm__declared@rung5 | 49.875 ±5.060 | 46.492 / 47.442 / 55.692 | 3 |
| mlp__declared@rung5 | 54.286 ±4.029 | 51.192 / 52.825 / 58.842 | 3 |
| mlp__tenx@rung5 | 55.894 ±4.223 | 53.642 / 53.275 / 60.767 | 3 |
| store_K32_n131072 | 66.581 ±0.846 | 65.642 / 66.817 / 67.283 | 3 |
| store_K32_n131072_REBUILT | 66.581 ±0.846 | 65.642 / 66.817 / 67.283 | 3 |
| store_K32_n16384 | 66.256 ±0.582 | 65.708 / 66.192 / 66.867 | 3 |
| store_K32_n32768 | 66.569 ±0.604 | 65.900 / 66.733 / 67.075 | 3 |
| store_K5_n16384 | 63.969 ±1.676 | 62.608 / 63.458 / 65.842 | 3 |
| store_K5_n32768 | 64.394 ±1.444 | 63.392 / 63.742 / 66.050 | 3 |
| table_FLOOR | 9.528 ±2.357 | 8.433 / 7.917 / 12.233 | 3 |
| transformer__declared@rung5 | 42.358 ±8.763 | 37.142 / 37.458 / 52.475 | 3 |

store_K32_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+12.000', '+13.542', '+6.517'], mean **+10.686** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+21.208', '+25.250', '+15.292']

EXISTING_ladder_store_K5_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+9.600', '+11.233', '+5.642'], mean **+8.825** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+18.808', '+22.942', '+14.417']

store_K32_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+12.067', '+12.917', '+6.100'], mean **+10.361** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+21.275', '+24.625', '+14.875']

store_K5_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+8.967', '+10.183', '+5.075'], mean **+8.075** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+18.175', '+21.892', '+13.850']

store_K32_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+12.258', '+13.458', '+6.308'], mean **+10.675** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+21.467', '+25.167', '+15.083']

store_K5_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+9.750', '+10.467', '+5.283'], mean **+8.500** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+18.958', '+22.175', '+14.058']

### rung n=6

| arm | mean ± sd | per seed | n seeds |
|---|---|---|---|
| EXISTING_ladder_store_K5_n131072 | 64.556 ±2.201 | 62.692 / 63.992 / 66.983 | 3 |
| EXISTING_learned_ref_ownpool@130514_32768rows_2560steps | 15.606 ±5.201 | 10.667 / 15.117 / 21.033 | 3 |
| EXISTING_spokeA_cont_mlp@118k_32256montero | 9.372 ±3.174 | 6.333 / 9.117 / 12.667 | 3 |
| EXISTING_spokeA_lstm@118k_32256montero | 43.503 ±3.726 | 44.433 / 39.400 / 46.675 | 3 |
| EXISTING_spokeA_mlp@118k_32256montero | 42.175 ±7.290 | 35.208 / 41.567 / 49.750 | 3 |
| EXISTING_spokeA_transformer@118k_32256montero | 38.231 ±12.187 | 28.800 / 33.900 / 51.992 | 3 |
| cont_mlp__declared@rung6 | 29.736 ±10.031 | 19.817 / 29.517 / 39.875 | 3 |
| cont_mlp__tenx@rung6 | 49.186 ±0.483 | 48.633 / 49.400 / 49.525 | 3 |
| lstm__declared@rung6 | 49.817 ±4.906 | 47.758 / 46.275 / 55.417 | 3 |
| lstm__tenx@rung6 | 45.417  | 45.417 | 1 |
| mlp__declared@rung6 | 54.361 ±4.680 | 49.525 / 54.692 / 58.867 | 3 |
| mlp__tenx@rung6 | 55.472 ±4.073 | 54.183 / 52.200 / 60.033 | 3 |
| store_K32_n131072 | 66.603 ±0.923 | 65.908 / 66.250 / 67.650 | 3 |
| store_K32_n131072_REBUILT | 66.603 ±0.923 | 65.908 / 66.250 / 67.650 | 3 |
| store_K32_n16384 | 67.006 ±0.405 | 66.583 / 67.042 / 67.392 | 3 |
| store_K32_n32768 | 67.050 ±0.946 | 66.092 / 67.075 / 67.983 | 3 |
| store_K5_n16384 | 64.147 ±1.306 | 62.967 / 63.925 / 65.550 | 3 |
| store_K5_n32768 | 63.469 ±2.570 | 61.050 / 63.192 / 66.167 | 3 |
| table_FLOOR | 9.528 ±2.357 | 8.433 / 7.917 / 12.233 | 3 |
| transformer__declared@rung6 | 42.278 ±9.242 | 36.592 / 37.300 / 52.942 | 3 |
| transformer__tenx@rung6 | 36.150  | 36.150 | 1 |

store_K32_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+11.725', '+11.558', '+7.617'], mean **+10.300** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+21.475', '+24.683', '+15.658']

EXISTING_ladder_store_K5_n131072 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+8.508', '+9.300', '+6.950'], mean **+8.253** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+18.258', '+22.425', '+14.992']

store_K32_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+12.400', '+12.350', '+7.358'], mean **+10.703** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+22.150', '+25.475', '+15.400']

store_K5_n16384 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+8.783', '+9.233', '+5.517'], mean **+7.844** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+18.533', '+22.358', '+13.558']

store_K32_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+11.908', '+12.383', '+7.950'], mean **+10.747** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+21.658', '+25.508', '+15.992']

store_K5_n32768 − strongest DATA-MATCHED rival (131,072 rung rows): per seed ['+6.867', '+8.500', '+6.133'], mean **+7.167** · draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): ['+16.617', '+21.625', '+14.175']

