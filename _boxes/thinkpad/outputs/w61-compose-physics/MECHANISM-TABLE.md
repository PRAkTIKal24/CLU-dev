# w6.1/compose-physics — MECHANISM TABLES

Every number here is recomputed from the parent's already-paid artifacts, except
the judge-margin block, which is a `val`-only probe. Nothing is scored.

## 1. The declared hypothesis: does the settled state leave `span(G_k)`?

| seed | leg | validity | median rel. displacement | off-span fraction of the STATE (max) | off-span fraction of the DISPLACEMENT (max) | AUC(off-span -> invalid) |
|---|---|---|---|---|---|---|
| 0 | `pca_tensor` | 0.8835 | 0.0000 | 1.96e-07 | 0.00e+00 | 0.526 |
| 0 | `pca_settle_zerostep` | 0.8835 | 0.0000 | 1.96e-07 | 3.86e-10 | 0.526 |
| 0 | `pca_settle_f05_conv` | 0.8761 | 0.0085 | 1.94e-07 | 4.56e-05 | 0.525 |
| 0 | `pca_settle_onestep` | 0.8455 | 0.0415 | 1.91e-07 | 9.05e-06 | 0.520 |
| 0 | `pca_settle` | 0.3285 | 0.3242 | 1.92e-07 | 1.48e-06 | 0.515 |
| 1 | `pca_tensor` | 0.8934 | 0.0000 | 1.88e-07 | 0.00e+00 | 0.553 |
| 1 | `pca_settle_zerostep` | 0.8934 | 0.0000 | 1.88e-07 | 6.77e-05 | 0.553 |
| 1 | `pca_settle_f05_conv` | 0.8863 | 0.0099 | 1.88e-07 | 3.70e-05 | 0.559 |
| 1 | `pca_settle_onestep` | 0.8407 | 0.0482 | 1.89e-07 | 7.43e-06 | 0.554 |
| 1 | `pca_settle` | 0.2418 | 0.4389 | 1.87e-07 | 1.21e-06 | 0.457 |
| 2 | `pca_tensor` | 0.9396 | 0.0000 | 1.80e-07 | 0.00e+00 | 0.467 |
| 2 | `pca_settle_zerostep` | 0.9396 | 0.0000 | 1.80e-07 | 1.04e-05 | 0.467 |
| 2 | `pca_settle_f05_conv` | 0.9342 | 0.0095 | 1.78e-07 | 3.32e-05 | 0.481 |
| 2 | `pca_settle_onestep` | 0.8985 | 0.0465 | 1.77e-07 | 6.93e-06 | 0.490 |
| 2 | `pca_settle` | 0.3971 | 0.3763 | 1.66e-07 | 1.03e-06 | 0.498 |

**Verdict: the hypothesis is false.** The maximum off-span fraction is ~2e-7 for
every leg *including the incumbent, which never moves* — it is float32 rounding
in `render_span`. The orthogonal fraction predicts invalidity at chance.

## 2. Displacement against the answer lattice

| seed | median state norm | rho_1 (nearest answer, any one factor) | rho_1 / norm |
|---|---|---|---|
| 0 | 31.299 | 1.7094 | 0.0546 |
| 1 | 32.291 | 1.8912 | 0.0586 |
| 2 | 32.788 | 1.9953 | 0.0609 |

| seed | leg | median displacement | displacement / rho_1 | frac > 0.5 rho_1 | median tangent fraction | mean wrong factors |
|---|---|---|---|---|---|---|
| 0 | `pca_tensor` | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.120 |
| 0 | `pca_settle_zerostep` | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.120 |
| 0 | `pca_settle_f05_conv` | 0.2655 | 0.1586 | 0.000 | 0.787 | 0.128 |
| 0 | `pca_settle_onestep` | 1.2993 | 0.7768 | 0.995 | 0.782 | 0.161 |
| 0 | `pca_settle` | 10.2312 | 5.8618 | 1.000 | 0.851 | 0.847 |
| 1 | `pca_tensor` | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.111 |
| 1 | `pca_settle_zerostep` | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.111 |
| 1 | `pca_settle_f05_conv` | 0.3194 | 0.1699 | 0.000 | 0.803 | 0.120 |
| 1 | `pca_settle_onestep` | 1.5584 | 0.8281 | 0.999 | 0.799 | 0.167 |
| 1 | `pca_settle` | 14.1609 | 7.5651 | 1.000 | 0.863 | 1.084 |
| 2 | `pca_tensor` | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.062 |
| 2 | `pca_settle_zerostep` | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.062 |
| 2 | `pca_settle_f05_conv` | 0.3106 | 0.1565 | 0.000 | 0.785 | 0.067 |
| 2 | `pca_settle_onestep` | 1.5193 | 0.7662 | 0.999 | 0.781 | 0.105 |
| 2 | `pca_settle` | 12.3149 | 6.1720 | 1.000 | 0.849 | 0.820 |

An isotropic step would place sqrt(51/308) = 0.407 of its length in the
rank-51 lattice tangent. The settle places 0.78-0.86.

## 3. Per-factor lattice spacing and per-factor damage

### seed 0

| factor | cardinality | rho_f | `pca_tensor` | `pca_settle_zerostep` | `pca_settle_f05_conv` | `pca_settle_onestep` | `pca_settle` |
|---|---|---|---|---|---|---|---|
| floor_hue | 10 | 7.885 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wall_hue | 10 | 9.087 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0275 |
| object_hue | 10 | 5.103 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0558 |
| scale | 8 | 2.041 | 0.0723 | 0.0723 | 0.0760 | 0.0938 | 0.5048 |
| shape | 4 | 6.989 | 0.0077 | 0.0077 | 0.0082 | 0.0116 | 0.0697 |
| orientation | 15 | 1.899 | 0.0405 | 0.0405 | 0.0442 | 0.0558 | 0.1891 |

### seed 1

| factor | cardinality | rho_f | `pca_tensor` | `pca_settle_zerostep` | `pca_settle_f05_conv` | `pca_settle_onestep` | `pca_settle` |
|---|---|---|---|---|---|---|---|
| floor_hue | 10 | 8.424 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 |
| wall_hue | 10 | 9.632 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0122 |
| object_hue | 10 | 5.413 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1814 |
| scale | 8 | 1.996 | 0.0745 | 0.0745 | 0.0819 | 0.1310 | 0.5918 |
| shape | 4 | 7.894 | 0.0122 | 0.0122 | 0.0142 | 0.0170 | 0.1310 |
| orientation | 15 | 2.172 | 0.0247 | 0.0247 | 0.0235 | 0.0193 | 0.1670 |

### seed 2

| factor | cardinality | rho_f | `pca_tensor` | `pca_settle_zerostep` | `pca_settle_f05_conv` | `pca_settle_onestep` | `pca_settle` |
|---|---|---|---|---|---|---|---|
| floor_hue | 10 | 9.023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0159 |
| wall_hue | 10 | 9.242 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| object_hue | 10 | 5.486 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2117 |
| scale | 8 | 2.119 | 0.0405 | 0.0405 | 0.0434 | 0.0694 | 0.3634 |
| shape | 4 | 8.929 | 0.0074 | 0.0074 | 0.0082 | 0.0119 | 0.1159 |
| orientation | 15 | 2.312 | 0.0139 | 0.0139 | 0.0156 | 0.0235 | 0.1128 |

## 4. The judge's margin (val only), in lattice units

P(the read-back of factor f changes) after moving s * rho_f toward the nearest
answer that differs only in factor f. 288 val requests, 15,552 judged states.

| factor | s=0.05 | s=0.1 | s=0.15 | s=0.2 | s=0.3 | s=0.4 | s=0.5 | s=0.7 | s=1 |
|---|---|---|---|---|---|---|---|---|---|
| floor_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.476 | 1.000 | 1.000 |
| wall_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.038 | 0.733 | 1.000 | 1.000 |
| object_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.010 | 0.135 | 0.431 | 0.948 | 1.000 |
| scale | 0.007 | 0.029 | 0.058 | 0.090 | 0.158 | 0.270 | 0.417 | 0.748 | 0.971 |
| shape | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 | 0.160 | 0.403 | 0.972 | 1.000 |
| orientation | 0.004 | 0.033 | 0.055 | 0.080 | 0.168 | 0.307 | 0.493 | 0.836 | 1.000 |

Isotropic control: the same states moved a matched ABSOLUTE distance in a random
in-span direction.

| factor | L=0.05 | L=0.1 | L=0.2 | L=0.5 | L=1 | L=2 | L=4 | L=7 | L=10 |
|---|---|---|---|---|---|---|---|---|---|
| floor_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| wall_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| object_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.017 |
| scale | 0.000 | 0.004 | 0.004 | 0.004 | 0.011 | 0.025 | 0.050 | 0.122 | 0.209 |
| shape | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.010 |
| orientation | 0.000 | 0.004 | 0.004 | 0.004 | 0.004 | 0.015 | 0.040 | 0.131 | 0.230 |

| | L=0.05 | L=0.1 | L=0.2 | L=0.5 | L=1 | L=2 | L=4 | L=7 | L=10 |
|---|---|---|---|---|---|---|---|---|---|
| whole-tuple validity (incumbent 0.9167) | 0.9167 | 0.9097 | 0.9132 | 0.9132 | 0.9097 | 0.8889 | 0.8438 | 0.6979 | 0.5521 |

## 5. The headline, recomputed from `rows.npz`

Row-4 novel valid tuples per corner, and the paired delta against `pca_tensor`.

| leg | seed 0 | seed 1 | seed 2 | delta s0 | delta s1 | delta s2 |
|---|---|---|---|---|---|---|
| `pca_tensor` | 63.612 | 64.327 | 67.653 | +0.000 | +0.000 | +0.000 |
| `pca_settle_zerostep` | 63.612 | 64.327 | 67.653 | +0.000 | +0.000 | +0.000 |
| `pca_settle_f05_conv` | 63.082 | 63.816 | 67.265 | -0.531 | -0.510 | -0.388 |
| `pca_settle_onestep` | 60.878 | 60.531 | 64.694 | -2.735 | -3.796 | -2.959 |
| `pca_settle` | 23.653 | 17.408 | 28.592 | -39.959 | -46.918 | -39.061 |

Paired over all 147 corner pairs (49 corners x 3 seeds) and all 10,584 requests:

| leg | mean delta/corner | corners worse / better / tied | sign test p | requests valid->invalid | invalid->valid | McNemar p |
|---|---|---|---|---|---|---|
| `pca_settle_zerostep` | +0.000 | 0 / 0 / 147 | nan | 0 | 0 | nan |
| `pca_settle_f05_conv` | -0.476 | 46 / 16 / 85 | 0.000176 | 104 | 34 | 1.92e-09 |
| `pca_settle_onestep` | -3.163 | 85 / 23 / 39 | 1.52e-09 | 602 | 137 | 2.56e-70 |
| `pca_settle` | -41.980 | 144 / 2 / 1 | 2.41e-40 | 6337 | 166 | 0 |

## 6. Can ANY leg leave span(G_k)?  (val, includes the channel-space kernel leg)

| leg | median off-span fraction | max | median displacement | validity |
|---|---|---|---|---|
| `pca_tensor` | 1.571e-07 | 1.889e-07 | 0.0000 | 0.8941 |
| `pca_settle` | 1.467e-07 | 1.719e-07 | 13.3967 | 0.3119 |
| `pca_settle_kernel` | 9.852e-08 | 1.360e-07 | 21.7777 | 0.0000 |
| `pca_settle_lookup` | 2.825e-08 | 2.826e-08 | 21.7648 | 0.0000 |

The kernel descent's force is x - sum_i p_i a_i: both terms lie in span(G_k), so span(G_k) is an INVARIANT subspace of it, exactly as it is of the coefficient-space settle. No dynamics built from these atoms can leave it.

## 7. The alternatives, and the motion budget

- standardise/un-standardise round trip, no steps taken: **1.177e-08** relative
- lattice tangent: rank 51 of 51 directions in a 308-dimensional state; an isotropic step would place 0.407 of its length there, the settle places 0.78-0.86

| seed | slope (validity per unit displacement/rho_1) | budget in rho_1 | budget absolute | budget as fraction of the state norm |
|---|---|---|---|---|
| 0 | 0.0488 | 0.285 | 0.4865 | 0.0155 |
| 1 | 0.0628 | 0.221 | 0.4184 | 0.0130 |
| 2 | 0.0529 | 0.263 | 0.5242 | 0.0160 |

## 8. Regression to seen combinations, on the ERRORS, against a local null

| seed | leg | n wrong | mean Hamming | seen (observed) | seen (locality null) | ratio |
|---|---|---|---|---|---|---|
| 0 | `pca_tensor` | 411 | 1.034 | 0.0073 | 0.5327 | 0.014 |
| 0 | `pca_settle_zerostep` | 411 | 1.034 | 0.0073 | 0.5352 | 0.014 |
| 0 | `pca_settle_f05_conv` | 437 | 1.037 | 0.0069 | 0.5356 | 0.013 |
| 0 | `pca_settle_onestep` | 545 | 1.044 | 0.0092 | 0.5369 | 0.017 |
| 0 | `pca_settle` | 2369 | 1.261 | 0.1583 | 0.5662 | 0.280 |
| 1 | `pca_tensor` | 376 | 1.045 | 0.0372 | 0.5338 | 0.070 |
| 1 | `pca_settle_zerostep` | 376 | 1.045 | 0.0372 | 0.5267 | 0.071 |
| 1 | `pca_settle_f05_conv` | 401 | 1.052 | 0.0424 | 0.5378 | 0.079 |
| 1 | `pca_settle_onestep` | 562 | 1.050 | 0.0516 | 0.5401 | 0.096 |
| 1 | `pca_settle` | 2675 | 1.429 | 0.2501 | 0.5905 | 0.423 |
| 2 | `pca_tensor` | 213 | 1.023 | 0.0000 | 0.5181 | 0.000 |
| 2 | `pca_settle_zerostep` | 213 | 1.023 | 0.0000 | 0.5237 | 0.000 |
| 2 | `pca_settle_f05_conv` | 232 | 1.022 | 0.0000 | 0.5226 | 0.000 |
| 2 | `pca_settle_onestep` | 358 | 1.034 | 0.0000 | 0.5284 | 0.000 |
| 2 | `pca_settle` | 2127 | 1.360 | 0.1683 | 0.5842 | 0.288 |

## 9. What evicting ONE atom would cost, in the same units (seed 0)

rho_1 = 1.7094, half = 0.8547, motion budget = 0.4870

| well | d_k | atoms | median eviction cost | cost of evicting the mean | past half rho_1 | past the budget |
|---|---|---|---|---|---|---|
| 0 | 777 | 64 | 0.3819 | 9.681 | 11 | 24 |
| 1 | 939 | 64 | 0.5839 | 14.73 | 17 | 36 |
| 2 | 216 | 64 | 0.2951 | 5.346 | 9 | 16 |
| 3 | 58 | 64 | 1.773e-15 | 2.206e-15 | 0 | 0 |
| 4 | 58 | 64 | 5.913e-15 | 8.847e-15 | 0 | 0 |

- `n_atoms_total` = 320
- `atoms_past_half_rho1` = 37
- `frac_past_half_rho1` = 0.1156
- `median_over_atoms` = 0.1930
- `median_in_half_rho1_units` = 0.2258
- `atoms_past_the_motion_budget` = 76
- `frac_past_the_motion_budget` = 0.2375
- `atoms_in_the_wells_whose_atoms_are_independent` = 192
- `atoms_past_budget_among_the_independent_ones` = 76

## 10. Row 3 against row 4: the metric is the validity rate, rescaled

| seed | leg | row3 delta novel | row3 delta validity x 36 | row4 delta novel | row4 delta validity x 72 |
|---|---|---|---|---|---|
| 0 | `pca_tensor` | +0.000 | +0.000 | +0.000 | +0.000 |
| 0 | `pca_settle_zerostep` | +0.000 | +0.000 | +0.000 | +0.000 |
| 0 | `pca_settle_f05_conv` | -0.033 | -0.033 | -0.531 | -0.531 |
| 0 | `pca_settle_onestep` | -1.383 | -1.383 | -2.735 | -2.735 |
| 0 | `pca_settle` | -15.800 | -15.800 | -39.959 | -39.959 |
| 1 | `pca_tensor` | +0.000 | +0.000 | +0.000 | +0.000 |
| 1 | `pca_settle_zerostep` | +0.000 | +0.000 | +0.000 | +0.000 |
| 1 | `pca_settle_f05_conv` | -0.183 | -0.183 | -0.510 | -0.510 |
| 1 | `pca_settle_onestep` | -1.617 | -1.617 | -3.796 | -3.796 |
| 1 | `pca_settle` | -25.533 | -25.533 | -46.918 | -46.918 |
| 2 | `pca_tensor` | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | `pca_settle_zerostep` | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | `pca_settle_f05_conv` | -0.033 | -0.033 | -0.388 | -0.388 |
| 2 | `pca_settle_onestep` | -0.983 | -0.983 | -2.959 | -2.959 |
| 2 | `pca_settle` | -20.350 | -20.350 | -39.061 | -39.061 |

## 11. The read-side byte ledger

- `store_bytes` = 524,288
- `address_bytes` = 189,344
- `coupling_bytes` = 78,848
- `standardisation_bytes` = 2,464
- `total_added` = 81,312
- `added_as_frac_of_store` = 0.1551
- `added_as_frac_of_store_plus_address` = 0.1139
- `added_float32_count` = 20,328

