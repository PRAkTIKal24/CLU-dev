seeds = [0, 1, 2] (n = 3)

## 1. Kill rule (i), CORRECTED — cylinder vs not, WITHIN the corner (REAL features). Chance = 0.5
| decoder | balanced accuracy | AUROC | TPR (72 held-out cyl) | TNR (matched-seen non-cyl) |
|---|---|---|---|---|
| baseline (superseded) | **0.7191 ± 0.0305** | 0.8806 ± 0.0113 | 0.5926 ± 0.0866 | 0.8457 ± 0.0262 |
| **V1 capacity_schedule** | **0.9414 ± 0.0179** | 0.9886 ± 0.0119 | 0.9398 ± 0.0131 | 0.9429 ± 0.0404 |

per-seed balacc  baseline [0.6843434343434344, 0.7143704634282524, 0.7586892797319933]  V1 [0.9223484848484849, 0.9364530988274706, 0.9652777777777778]
per-seed AUROC   baseline [0.8717031425364759, 0.873534338358459, 0.8965661641541038]  V1 [0.9720117845117845, 0.993858179787828, 0.9997906197654941]
n_pos [72, 72, 72]  n_neg [198, 199, 199]

## 2. Same corrected metric on the STORE READ (correlation vs random). Chance = 0.5
| arm | decoder | balanced accuracy | AUROC |
|---|---|---|---|
| correlation | baseline_decoder | 0.6349 ± 0.1004 | 0.7154 ± 0.0964 |
| correlation | v1_decoder | 0.6866 ± 0.0681 | 0.7414 ± 0.0295 |
| random | baseline_decoder | 0.5370 ± 0.0291 | 0.4896 ± 0.0945 |
| random | v1_decoder | 0.5491 ± 0.0146 | 0.5625 ± 0.0145 |
| **margin (corr−rand)** | baseline_decoder | **0.0980 ± 0.0717** | 0.2259 ± 0.0445 |
| **margin (corr−rand)** | v1_decoder | **0.1375 ± 0.0729** | 0.1788 ± 0.0338 |

## 3. R0's READ re-scored. ⛔ raw multi-class top-1; NOT comparable to the balanced-accuracy tables
| | aggregate (old dec) | aggregate (V1 dec) | shape (old dec) | shape (V1 dec) |
|---|---|---|---|---|
| correlation | 0.5997 ± 0.0019 | 0.6107 ± 0.0041 | 0.5785 ± 0.0124 | 0.5960 ± 0.0158 |
| random | 0.4492 ± 0.0008 | 0.4603 ± 0.0055 | 0.3770 ± 0.0156 | 0.3952 ± 0.0057 |
| **margin (corr−rand)** | **0.1504 ± 0.0023** | **0.1504 ± 0.0094** | **0.2015 ± 0.0064** | **0.2008 ± 0.0165** |
| no-store ceiling | 0.9507 ± 0.0129 | 0.9902 ± 0.0030 | 0.9076 ± 0.0122 | 0.9733 ± 0.0090 |

per-seed aggregate margin: old [0.14811197916666663, 0.14957682291666663, 0.15364583333333337]
per-seed aggregate margin: V1  [0.1376953125, 0.16015625000000006, 0.15332031250000006]
per-seed shape margin:     V1  [0.177734375, 0.21484375, 0.2099609375]

## 4. Corrected leakage (balanced-accuracy transfer). Pooled
| arm | corrected pooled leakage | corrected pooled `shape` | OLD (degenerate) pooled | OLD `shape` |
|---|---|---|---|---|
| correlation | 0.3317 ± 0.1406 | 0.2247 ± 0.1135 | -0.0444 ± 0.0921 | -0.2435 ± 0.5142 |
| random | 0.4654 ± 0.1819 | 0.2394 ± 0.1888 | 0.2257 ± 0.0443 | 0.7946 ± 0.1599 |
| **margin (rand−corr)** | **0.1337 ± 0.0592** | **0.0147 ± 0.0771** | 0.2701 ± 0.0518 | — |

per-seed pooled-leakage margin (rand−corr): [0.051195472463922975, 0.16244469537341288, 0.18734599534522456]

## 5. ⭐ NON-DEGENERACY PROOF — every (well, factor) cell, both arms, all seeds
| | n cells | NaN cells | min | max | cells with |leakage| > 1 |
|---|---|---|---|---|---|
| **corrected** | 180 | **0** | **-0.5534** | **+1.0000** | **0** |
| old (as literally specified) | 180 | 55 | -198.0000 | +1.0000 | 10 |

corrected argmin: {'seed': 0, 'arm': 'correlation', 'well': 0, 'factor': 'object_hue', 'corrected': -0.553447699121465, 'old': 0.022222222222222254}
corrected argmax: {'seed': 0, 'arm': 'correlation', 'well': 0, 'factor': 'floor_hue', 'corrected': 1.0, 'old': nan}
old argmin:       {'seed': 1, 'arm': 'correlation', 'well': 3, 'factor': 'shape', 'corrected': 0.4450357221991571, 'old': -198.0}

## 6. Companions the corrections leave UNCHANGED (from the shipped manifests)
| arm | block-diagonality | MIG | DCI-D | DCI-C | DCI-I |
|---|---|---|---|---|---|
| correlation | 0.4787 ± 0.0106 | 0.2026 ± 0.0355 | 0.3094 ± 0.0078 | 0.1968 ± 0.0124 | 0.5838 ± 0.0179 |
| random | 0.2223 ± 0.0095 | 0.0204 ± 0.0047 | 0.3208 ± 0.0082 | 0.0223 ± 0.0013 | 0.2992 ± 0.0052 |
