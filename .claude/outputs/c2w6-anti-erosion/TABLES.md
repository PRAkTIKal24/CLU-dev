# c2w6-anti-erosion — TABLES (generated)

records: 21  cells: ['p1_off', 'p1_on', 'p1_on_i1_on', 'resoff_p1_off', 'resoff_p1_on', 'w40_p1_off', 'w40_p1_on']


## T1 — the erosion curve (median fitted depth, lane 0, val batch)

| cell | seed | step0 | step200 | final | final/step0 | final/step200 |
|---|---|---|---|---|---|---|
| p1_off | 0 | 0.04629 | 0.0006391 | 0.006252 | 0.135 | 9.78 |
| p1_off | 1 | 0.1619 | 0.331 | 0.299 | 1.85 | 0.903 |
| p1_off | 2 | 0.2489 | 0.0659 | 0.03496 | 0.14 | 0.53 |
| p1_on_i1_on | 0 | 0.04629 | 0.001221 | 0.01381 | 0.298 | 11.3 |
| p1_on_i1_on | 1 | 0.1619 | 0.29 | 0.08654 | 0.535 | 0.298 |
| p1_on_i1_on | 2 | 0.2489 | 0.1348 | 0.3482 | 1.4 | 2.58 |
| p1_on | 0 | 0.04629 | 0.000958 | 0.006771 | 0.146 | 7.07 |
| p1_on | 1 | 0.1619 | 0.2699 | 0.1455 | 0.899 | 0.539 |
| p1_on | 2 | 0.2489 | 0.06482 | 0.1903 | 0.765 | 2.94 |
| resoff_p1_off | 0 | 0.04629 | 0.0006457 | 0.0008708 | 0.0188 | 1.35 |
| resoff_p1_off | 1 | 0.1619 | 0.0533 | 0.01216 | 0.0751 | 0.228 |
| resoff_p1_off | 2 | 0.2489 | 0.02661 | 0.08191 | 0.329 | 3.08 |
| resoff_p1_on | 0 | 0.04629 | 0.000791 | 0.000499 | 0.0108 | 0.631 |
| resoff_p1_on | 1 | 0.1619 | 0.107 | 0.0843 | 0.521 | 0.788 |
| resoff_p1_on | 2 | 0.2489 | 0.1541 | 0.2372 | 0.953 | 1.54 |
| w40_p1_off | 0 | 0.07297 | 0.001328 | 0.0002123 | 0.00291 | 0.16 |
| w40_p1_off | 1 | 0.1359 | 0.1973 | 0.3703 | 2.73 | 1.88 |
| w40_p1_off | 2 | 0.2971 | 0.2727 | 0.2643 | 0.89 | 0.969 |
| w40_p1_on | 0 | 0.07297 | 0.02965 | 0.004569 | 0.0626 | 0.154 |
| w40_p1_on | 1 | 0.1359 | 0.4006 | 0.5614 | 4.13 | 1.4 |
| w40_p1_on | 2 | 0.2971 | 0.3269 | 0.3868 | 1.3 | 1.18 |

### T1b — seed mean ± SE per cell

| cell | n | depth final | ratio final/200 | ratio final/untrained |
|---|---|---|---|---|
| p1_off | 3 | 0.1134 ± 0.093 | 3.74 ± 3 | 0.708 ± 0.57 |
| p1_on | 3 | 0.1142 ± 0.055 | 3.51 ± 1.9 | 0.603 ± 0.23 |
| p1_on_i1_on | 3 | 0.1495 ± 0.1 | 4.73 ± 3.4 | 0.744 ± 0.33 |
| w40_p1_off | 3 | 0.2116 ± 0.11 | 1 ± 0.5 | 1.21 ± 0.8 |
| w40_p1_on | 3 | 0.3176 ± 0.16 | 0.913 ± 0.38 | 1.83 ± 1.2 |
| resoff_p1_off | 3 | 0.03164 ± 0.025 | 1.55 ± 0.83 | 0.141 ± 0.095 |
| resoff_p1_on | 3 | 0.1073 ± 0.069 | 0.986 ± 0.28 | 0.495 ± 0.27 |

## T2 — bpc, live vs blank vs memory-deleted (K4's columns)

| cell | seed | bpc live | live−blank | memdel(eval)−live | none(retrained)−live | acq / chance |
|---|---|---|---|---|---|---|
| p1_off | 0 | 3.789 | -0.00107 | 0.0778 | 0.0719 | 0.19 / 0.19 |
| p1_off | 1 | 3.708 | -0.000567 | 0.125 | 0.0911 | 0.0833 / 0.167 |
| p1_off | 2 | 3.635 | 0.000261 | 0.197 | 0.155 | 0.167 / 0.167 |
| p1_on_i1_on | 0 | 3.759 | -0.000104 | 0.129 | — | 0.167 / 0.167 |
| p1_on_i1_on | 1 | 3.702 | -0.00102 | 0.13 | — | 0.167 / 0.167 |
| p1_on_i1_on | 2 | 3.628 | -0.000372 | 0.22 | — | 0.167 / 0.167 |
| p1_on | 0 | 3.785 | -0.000314 | 0.105 | — | 0.182 / 0.182 |
| p1_on | 1 | 3.703 | -0.000847 | 0.124 | — | 0.261 / 0.174 |
| p1_on | 2 | 3.629 | -0.00195 | 0.226 | — | 0.167 / 0.167 |
| resoff_p1_off | 0 | 3.76 | 1.72e-06 | 0.113 | — | 0.136 / 0.182 |
| resoff_p1_off | 1 | 3.691 | -0.000994 | 0.15 | — | 0.167 / 0.167 |
| resoff_p1_off | 2 | 3.638 | -0.000357 | 0.202 | — | 0.208 / 0.167 |
| resoff_p1_on | 0 | 3.764 | -3.58e-05 | 0.109 | — | 0.167 / 0.167 |
| resoff_p1_on | 1 | 3.694 | -0.000764 | 0.144 | — | 0.167 / 0.167 |
| resoff_p1_on | 2 | 3.65 | -0.000211 | 0.175 | — | 0.125 / 0.167 |
| w40_p1_off | 0 | 4.328 | -0.00176 | 0.00124 | — | 0.174 / 0.174 |
| w40_p1_off | 1 | 4.316 | 0.00117 | 0.0141 | — | 0.304 / 0.174 |
| w40_p1_off | 2 | 4.278 | -0.00102 | 0.0436 | — | 0.167 / 0.167 |
| w40_p1_on | 0 | 4.328 | -0.0017 | 0.00349 | — | 0.25 / 0.2 |
| w40_p1_on | 1 | 4.316 | 0.000588 | 0.0125 | — | 0.391 / 0.174 |
| w40_p1_on | 2 | 4.295 | 5.26e-05 | 0.0233 | — | 0.304 / 0.174 |

## T3 — I1: the rewrite audit

| cell | seed | admits | occupied-target | evicting | events | pre-guard viol | post-guard viol | rate(pre) |
|---|---|---|---|---|---|---|---|---|
| p1_off | 0 | 427 | 174 | 164 | 27 | 16 | 16 | 0.593 |
| p1_off | 1 | 535 | 232 | 231 | 40 | 2 | 2 | 0.05 |
| p1_off | 2 | 515 | 233 | 196 | 70 | 3 | 3 | 0.0429 |
| p1_on_i1_on | 0 | 445 | 183 | 163 | 44 | 6 | 0 | 0.136 |
| p1_on_i1_on | 1 | 461 | 169 | 131 | 62 | 0 | 0 | 0 |
| p1_on_i1_on | 2 | 451 | 172 | 142 | 59 | 0 | 0 | 0 |
| p1_on | 0 | 404 | 141 | 119 | 37 | 1 | 1 | 0.027 |
| p1_on | 1 | 483 | 211 | 150 | 78 | 0 | 0 | 0 |
| p1_on | 2 | 473 | 194 | 165 | 53 | 0 | 0 | 0 |
| resoff_p1_off | 0 | 464 | 196 | 167 | 34 | 31 | 31 | 0.912 |
| resoff_p1_off | 1 | 502 | 201 | 186 | 43 | 23 | 23 | 0.535 |
| resoff_p1_off | 2 | 511 | 234 | 190 | 67 | 6 | 6 | 0.0896 |
| resoff_p1_on | 0 | 498 | 218 | 176 | 63 | 61 | 61 | 0.968 |
| resoff_p1_on | 1 | 515 | 214 | 201 | 58 | 0 | 0 | 0 |
| resoff_p1_on | 2 | 553 | 278 | 272 | 29 | 1 | 1 | 0.0345 |
| w40_p1_off | 0 | 78 | 0 | 0 | 0 | 0 | 0 | — |
| w40_p1_off | 1 | 177 | 70 | 39 | 34 | 0 | 0 | 0 |
| w40_p1_off | 2 | 166 | 63 | 36 | 30 | 0 | 0 | 0 |
| w40_p1_on | 0 | 125 | 14 | 19 | 1 | 0 | 0 | 0 |
| w40_p1_on | 1 | 136 | 21 | 9 | 15 | 0 | 0 | 0 |
| w40_p1_on | 2 | 161 | 49 | 41 | 19 | 0 | 0 | 0 |

### T3b — the interference audit (the measurable #9/#12 channel)

| cell | seed | events | own-leg viol vs decay law | max own residual | foreign-up rate | median rel Δforeign |
|---|---|---|---|---|---|---|
| p1_off | 0 | 1530 | 0 | 3.11e-07 | 0.541 | 20.1 |
| p1_off | 1 | 2060 | 0 | 2.69e-07 | 0.243 | -0.0281 |
| p1_off | 2 | 1963 | 0 | 3.17e-07 | 0.365 | 0.0462 |
| p1_on_i1_on | 0 | 1616 | 0 | 3.17e-07 | 0.494 | 0.649 |
| p1_on_i1_on | 1 | 1690 | 0 | 3.22e-07 | 0.316 | -0.0239 |
| p1_on_i1_on | 2 | 1643 | 0 | 3.12e-07 | 0.43 | 0.0787 |
| p1_on | 0 | 1410 | 0 | 3.32e-07 | 0.53 | 0.289 |
| p1_on | 1 | 1800 | 0 | 2.86e-07 | 0.369 | -0.0226 |
| p1_on | 2 | 1753 | 0 | 2.84e-07 | 0.41 | 0.052 |
| resoff_p1_off | 0 | 1708 | 0 | 3.29e-07 | 0.516 | 0.0998 |
| resoff_p1_off | 1 | 1895 | 0 | 3.2e-07 | 0.298 | -0.0257 |
| resoff_p1_off | 2 | 1943 | 0 | 3.2e-07 | 0.317 | 0.0452 |
| resoff_p1_on | 0 | 1878 | 0 | 3.07e-07 | 0.448 | 0.052 |
| resoff_p1_on | 1 | 1960 | 0 | 3.1e-07 | 0.308 | -0.0184 |
| resoff_p1_on | 2 | 2153 | 0 | 3.08e-07 | 0.336 | 0.0513 |
| w40_p1_off | 0 | 147 | 0 | 2.83e-07 | 0.731 | 1.1 |
| w40_p1_off | 1 | 630 | 0 | 2.44e-07 | 0.461 | 0.0776 |
| w40_p1_off | 2 | 578 | 0 | 2.5e-07 | 0.52 | 0.06 |
| w40_p1_on | 0 | 373 | 0 | 2.71e-07 | 0.629 | 0.544 |
| w40_p1_on | 1 | 425 | 0 | 2.43e-07 | 0.612 | 0.0936 |
| w40_p1_on | 2 | 553 | 0 | 2.8e-07 | 0.614 | 0.119 |

## T4 — I2: usage vs erosion (ρ, Spearman over live wells)

| cell | seed | n wells | ρ(read-selection, erosion) | ρ(LOO Δbpc, erosion) | ρ(grad, erosion) |
|---|---|---|---|---|---|
| p1_off | 0 | 6 | 0.0286 | 0.257 | 0.0286 |
| p1_off | 1 | 6 | -0.486 | 0.2 | -0.714 |
| p1_off | 2 | 6 | -0.314 | -0.257 | 0.2 |
| p1_on_i1_on | 0 | 6 | -0.486 | 0.6 | — |
| p1_on_i1_on | 1 | 6 | -0.2 | 0.371 | — |
| p1_on_i1_on | 2 | 6 | -0.771 | 0.486 | — |
| p1_on | 0 | 6 | -0.0857 | 0.143 | — |
| p1_on | 1 | 6 | -0.257 | 0.2 | — |
| p1_on | 2 | 6 | -0.257 | 0.371 | — |
| resoff_p1_off | 0 | 6 | -0.257 | -0.0286 | 0.543 |
| resoff_p1_off | 1 | 6 | 0.2 | 0.6 | 0.771 |
| resoff_p1_off | 2 | 6 | -0.0857 | 0.0286 | 0.0857 |
| resoff_p1_on | 0 | 6 | -0.0857 | 0.771 | — |
| resoff_p1_on | 1 | 6 | 0.314 | 0.943 | — |
| resoff_p1_on | 2 | 6 | -0.174 | -0.314 | — |
| w40_p1_off | 0 | 5 | 0.872 | 0.2 | 0.9 |
| w40_p1_off | 1 | 6 | 0.143 | -0.143 | 0.0857 |
| w40_p1_off | 2 | 6 | -0.771 | 0.6 | -0.714 |
| w40_p1_on | 0 | 6 | 0.371 | -0.886 | — |
| w40_p1_on | 1 | 6 | 0.714 | -0.0857 | — |
| w40_p1_on | 2 | 6 | 0.551 | 0.543 | — |

### T4b — pooled per-well rows on the partition-OFF arm

| seed | slot | erosion rate | mean read-sel | mean LOO Δbpc | mean ‖∂L/∂atoms‖ | depth first→last |
|---|---|---|---|---|---|---|
| 0 | 0 | -0.00343 | 4.17 | -0.000374 | 0.0014 | 0.0463 → 0.00847 |
| 0 | 1 | -0.00379 | 2.07 | 0.000195 | 0.000489 | 0.0462 → 0.0212 |
| 0 | 2 | -0.000689 | 3.54 | 0.000254 | 0.000903 | 0.0592 → 0.0168 |
| 0 | 3 | -0.00208 | 2.24 | 0.000127 | 0.000381 | 0 → 0.00144 |
| 0 | 4 | -0.000928 | 2.32 | -4.89e-05 | 0.00052 | 0 → 0.00403 |
| 0 | 5 | -0.000802 | 0.659 | -1.07e-05 | 0.000404 | 0 → 0.00156 |
| 1 | 0 | 0.000221 | 5.05 | -0.000871 | 0.00149 | 0.114 → 0.296 |
| 1 | 1 | 0.000269 | 1.8 | 0.000283 | 0.00112 | 0.321 → 0.0339 |
| 1 | 2 | -0.000821 | 3.78 | -0.000405 | 0.00114 | 0.162 → 0.514 |
| 1 | 3 | 0.000444 | 2 | -5.89e-06 | 0.000945 | 0.0353 → 0.514 |
| 1 | 4 | 0.0014 | 0.951 | -0.000163 | 0.000765 | 0.639 → 0.027 |
| 1 | 5 | 9.9e-06 | 1.41 | 7.59e-05 | 0.000967 | 0 → 0.302 |
| 2 | 0 | 0.00143 | 4.32 | -2.75e-05 | 0.000941 | 0.249 → 0.00167 |
| 2 | 1 | 0.000769 | 0.634 | 0.000103 | 0.00033 | 0.0688 → 0.0414 |
| 2 | 2 | -0.000419 | 1.8 | 0.000134 | 0.00136 | 0.274 → 0.288 |
| 2 | 3 | -0.000989 | 4.34 | 3.14e-05 | 0.000549 | 0 → 0.0286 |
| 2 | 4 | -0.000186 | 1.73 | 7.88e-05 | 0.000941 | 0 → 0.0263 |
| 2 | 5 | -0.00172 | 2.17 | 9.56e-05 | 0.000366 | 0 → 0.0577 |

POOLED over seeds (n=18 wells): ρ(read-sel, erosion) = -0.323, ρ(LOO Δbpc, erosion) = -0.112

## T5 — the mechanical gate (prereg §4)
```json
{
 "verdict": "FAILS_FLATTEN",
 "on": "p1_on",
 "off": "p1_off",
 "seeds": [
  0,
  1,
  2
 ],
 "n_paired_seeds": 3,
 "legs": {
  "E2_on_arm_flattens": {
   "rule": "depth(final)/depth(200) >= 0.5 on ALL seeds",
   "per_seed": [
    7.068042028561519,
    0.5392075050998889,
    2.93599365466136
   ],
   "n_met": 3,
   "passed": true
  },
  "E1_off_arm_decays": {
   "rule": "depth(final)/depth(200) <= 0.5 on >= 2/3 seeds, OR already at the collapse floor (1e-06) by step 200 (a stronger form of E1)",
   "per_seed": [
    9.781540548629886,
    0.9034927079517782,
    0.5304736434450323
   ],
   "depth_at_200_per_seed": [
    0.0006391352240705614,
    0.3309603889886801,
    0.06590218627563368
   ],
   "collapsed_by_200": [
    false,
    false,
    false
   ],
   "n_met": 0,
   "passed": false
  },
  "K3_bpc_not_worse": {
   "rule": "paired Delta bpc(ON-OFF) not worse than 2 SE AND within +-0.01 (E3's registered equivalence band)",
   "delta_per_seed": [
    -0.0035488620005539318,
    -0.004765295290106941,
    -0.006245034018130369
   ],
   "delta_mean": -0.004853063769597081,
   "delta_se": 0.0007795540106549294,
   "passed": true
  },
  "K4_not_relocated": {
   "rule": "K4 FIRES if the ON arm's |live-blank| is at the float32 floor AND the memory-deleted margin is not positive: depth would be protected while the store stayed useless, i.e. the collapse relocated into the block's other weights",
   "abs_live_minus_blank": [
    0.000313610363848138,
    0.0008472725294756955,
    0.0019505412347604612
   ],
   "memory_deleted_minus_live": [
    0.10475127897884473,
    0.12427672730527606,
    0.22585570029763113
   ],
   "fired": false,
   "passed": true
  }
 },
 "caveats": [
  "toy scale (0.16 M) \u2014 no pilot-scale claim",
  "monitor #13 / N94 demotes every w4 reading; the w40 pair is the undemoted confirmation",
  "the tier-appropriate control is the system-level swap (K4); the settle-deleted launder is inherited diagnostic only"
 ]
}
```

### the w40 gate
```json
{
 "verdict": "FAILS_FLATTEN",
 "on": "w40_p1_on",
 "off": "w40_p1_off",
 "seeds": [
  0,
  1,
  2
 ],
 "n_paired_seeds": 3,
 "legs": {
  "E2_on_arm_flattens": {
   "rule": "depth(final)/depth(200) >= 0.5 on ALL seeds",
   "per_seed": [
    0.15409223781618414,
    1.4013149772769447,
    1.1833230011468954
   ],
   "n_met": 2,
   "passed": false
  },
  "E1_off_arm_decays": {
   "rule": "depth(final)/depth(200) <= 0.5 on >= 2/3 seeds, OR already at the collapse floor (1e-06) by step 200 (a stronger form of E1)",
   "per_seed": [
    0.15986782420458387,
    1.877170568540929,
    0.9690572573235927
   ],
   "depth_at_200_per_seed": [
    0.001327750326842504,
    0.19727350814786038,
    0.2726924858583499
   ],
   "collapsed_by_200": [
    false,
    false,
    false
   ],
   "n_met": 1,
   "passed": false
  },
  "K3_bpc_not_worse": {
   "rule": "paired Delta bpc(ON-OFF) not worse than 2 SE AND within +-0.01 (E3's registered equivalence band)",
   "delta_per_seed": [
    -0.0002647672910027765,
    0.0003974519061440418,
    0.01689454372485777
   ],
   "delta_mean": 0.005675742779999678,
   "delta_se": 0.005612656962809714,
   "passed": true
  },
  "K4_not_relocated": {
   "rule": "K4 FIRES if the ON arm's |live-blank| is at the float32 floor AND the memory-deleted margin is not positive: depth would be protected while the store stayed useless, i.e. the collapse relocated into the block's other weights",
   "abs_live_minus_blank": [
    0.0017021982869147578,
    0.0005883526486014645,
    5.262669116401497e-05
   ],
   "memory_deleted_minus_live": [
    0.003486690272266557,
    0.01250692233604589,
    0.023268994687761335
   ],
   "fired": false,
   "passed": true
  }
 },
 "caveats": [
  "toy scale (0.16 M) \u2014 no pilot-scale claim",
  "monitor #13 / N94 demotes every w4 reading; the w40 pair is the undemoted confirmation",
  "the tier-appropriate control is the system-level swap (K4); the settle-deleted launder is inherited diagnostic only"
 ]
}
```

## T6 — the prereg scorecard
```json
{
 "E1": {
  "registered": "<= 0.3x on >=2/3 seeds (band (0.02, 0.5))",
  "measured_per_seed": [
   9.781540548629886,
   0.9034927079517782,
   0.5304736434450323
  ],
  "n_met_point": 0,
  "n_met_band": 0,
  "n_seeds": 3
 },
 "E2": {
  "registered": ">= 0.7x (band (0.5, 1.05)), 3/3 seeds",
  "measured_per_seed": [
   7.068042028561519,
   0.5392075050998889,
   2.93599365466136
  ],
  "n_met_point": 2,
  "n_met_band": 3,
  "n_inside_band": 1,
  "n_above_band": 2,
  "n_seeds": 3
 },
 "E3": {
  "registered": "paired |Delta bpc| <= 0.01",
  "delta_per_seed": [
   -0.0035488620005539318,
   -0.004765295290106941,
   -0.006245034018130369
  ],
  "seeds": [
   0,
   1,
   2
  ],
  "mean": -0.004853063769597081,
  "se": 0.0007795540106549294,
  "met": true
 },
 "P_residual_interaction": {
  "registered": "partition-ON final depth >= the residual-only banked 0.1321 on >=2/3 seeds; if partition-ON depth COLLAPSES BELOW the partition-OFF arm, P1 is disproved as specified",
  "on_final_per_seed": [
   0.006771428533309027,
   0.145523982937735,
   0.190315252036386
  ],
  "off_final_per_seed": [
   0.006251727110303844,
   0.2990202980721564,
   0.03495937286462861
  ],
  "n_ge_banked": 2,
  "on_below_off_seeds": 1,
  "n_seeds": 3
 },
 "I1a": {
  "registered": "10%-40% of rewrite events (band (0.02, 0.6))",
  "measured_per_seed": [
   0.5925925925925926,
   0.05,
   0.04285714285714286
  ],
  "mean": 0.2284832451499118,
  "se": 0.18206635033543123,
  "n": 3,
  "n_events_per_seed": [
   27,
   40,
   70
  ],
  "in_point": true,
  "in_band": true
 },
 "I1b": {
  "registered": "depth-reduction events = EXACTLY 0 by construction; a violation-free write is bit-identical",
  "violations_pre_guard_per_seed": [
   6,
   0,
   0
  ],
  "violations_post_guard_per_seed": [
   0,
   0,
   0
  ],
  "events_per_seed": [
   44,
   62,
   59
  ],
  "note": "the pre-guard count is how often the guard FIRED (the repairs); I1-b is the post-guard count",
  "met": true
 },
 "I2": {
  "registered": "on the partition-OFF arm, Spearman rho >= 0.5 (most-useful wells erode fastest); <= -0.3 is the registered refutation branch; |rho| < 0.3 = no usage structure",
  "rho_read_selection_per_seed": [
   0.02857142857142857,
   -0.4857142857142857,
   -0.3142857142857143
  ],
  "rho_read_selection_mean": -0.2571428571428571,
  "se": 0.15118578920369088,
  "n": 3,
  "rho_loo_delta_bpc_per_seed": [
   0.2571428571428571,
   0.2,
   -0.2571428571428571
  ],
  "rho_loo_delta_bpc_mean": 0.06666666666666667,
  "loo_se": 0.1627429284822387,
  "loo_n": 3,
  "branch": "NO_USAGE_STRUCTURE",
  "note": "provisional \u2014 the analyst adjudicates rho on the raw per-well series"
 }
}
```

## T7 — the run-3 flag block
```bash
# scripts/csf3/job_gpu_cluformer.sh — RUN 3 candidate
# ⛔ VERDICT AT EMISSION: FAILS_FLATTEN. Submitted only on the Advisor's decision.
sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,\
MEM="atom_place_radius=0.3 write_inner_steps=40 psi_payload_residual=True psi_residual_source=q_star erosion_partition=True refresh_monotonic=True",\
STORE="write_margin=0.6",\
SET="monitor_every=25 plan_workers=8" \
       -c 12 --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh

```
