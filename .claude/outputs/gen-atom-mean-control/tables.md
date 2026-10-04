## 1. The control, at matched bytes (probe + external, 3 seeds)

| budget | bytes | atoms/well | byte match | selected (k, metric, norm, scope) | channel bits | ext in-corner | probe in-corner | distinct ATOM-CONFIGS on B | distinct factor-tuples on B |
|---|---|---|---|---|---|---|---|---|---|
| **A** | 524,288 | 64 | **True** | (5,cos,zsco,per_well); (1,cos,none,per_well); (3,cos,none,per_well) | 90.0000 ± 48.9898 | **0.6849 ± 0.0202** | 0.7088 ± 0.0443 | **24.3333 ± 8.8066** | 14.3333 ± 2.0548 |
| **B** | 4,194,304 | 512 | **True** | (1,euc,none,per_well); (3,euc,none,per_well); (3,cos,none,per_well) | 105.0000 ± 42.4264 | **0.6685 ± 0.0927** | 0.7241 ± 0.0686 | **37.3333 ± 11.6142** | 24.3333 ± 5.3125 |
| *knn_proto (regression check, UNBOUNDED channel)* | *475,136* | — | — | — | *inf* | *0.9119* | *0.8884 ± 0.0088* | — | *54.7* |
| *decoder ceiling (real features)* | — | — | — | — | — | *0.9357 ± 0.0193* | *0.9414 ± 0.0179* | — | — |

## 2. ⭐ Relative read error on `A` (rows EXCLUDED from the atom pool)

| leg | budget A (N=64) | budget B (N=512) | fitted slope d ln ε / d ln N | implied PR |
|---|---|---|---|---|
| `centroid_control/k1` | 0.2852 ± 0.0048 | 0.1695 ± 0.0021 | **-0.2501 ± 0.0050** | 8.00 |
| `oracle_feature_addressed/k1` | 0.4067 ± 0.0094 | 0.2306 ± 0.0038 | **-0.2727 ± 0.0141** | 7.35 |
| `oracle_feature_addressed/k_selected` | 0.3567 ± 0.0319 | 0.1851 ± 0.0324 | **-0.3204 ± 0.1054** | 7.09 |
| `request_addressed/k1` | 0.4618 ± 0.0112 | 0.3299 ± 0.0053 | **-0.1617 ± 0.0053** | 12.39 |
| `request_addressed/k_selected` | 0.4114 ± 0.0355 | 0.2819 ± 0.0394 | **-0.1845 ± 0.0844** | 14.55 |
| *centroid law, gen-dynamical-read PREREG §2 (RENDERED, not measured here)* | *0.2783* | *0.1468* | *−0.3076* | *6.50* |

| budget | medoid penalty (medoid k=1 / centroid k=1, same N, same Lloyd fit) | measured centroid ÷ the PREREG law |
|---|---|---|
| A | **×1.426 ± 0.016** | ×1.025 |
| B | **×1.360 ± 0.011** | ×1.155 |

## 3. ⭐ Six-factor retention ladder (probe), ⛔ every row with n_values_realised


**rows = `A`**

| factor | n_values_realised | budget A | budget B | knn_proto (unrestricted) |
|---|---|---|---|---|
| `floor_hue` | [5, 6] | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 |
| `object_hue` | [10] | 0.7871 ± 0.0752 | 0.9603 ± 0.0396 | 1.0000 |
| `orientation` | [15] | 0.1862 ± 0.0357 | 0.3164 ± 0.0633 | 0.8646 |
| `scale` | [8] | 0.1270 ± 0.0248 | 0.1908 ± 0.0097 | 0.3092 |
| `shape` | [4] | 0.4232 ± 0.0456 | 0.4766 ± 0.0299 | 0.6934 |
| `wall_hue` | [10] | 0.9993 ± 0.0009 | 1.0000 ± 0.0000 | 1.0000 |
| **aggregate** | — | **0.5871 ± 0.0221** | **0.6573 ± 0.0172** | 0.8112 ± 0.0159 |

**rows = `B`**

| factor | n_values_realised | budget A | budget B | knn_proto (unrestricted) |
|---|---|---|---|---|
| `floor_hue` | [2] | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 |
| `object_hue` | [2] | 0.8519 ± 0.0836 | 1.0000 ± 0.0000 | 1.0000 |
| `orientation` | [3] | 0.1806 ± 0.1375 | 0.3704 ± 0.1544 | 0.8796 |
| `scale` | [3] | 0.0139 ± 0.0196 | 0.0000 ± 0.0000 | 0.1667 |
| `shape` | **[1]** ⛔ NOT APPLICABLE | 0.8657 ± 0.0285 | 0.8056 ± 0.1487 | 1.0000 |
| `wall_hue` | [2] | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 |
| **aggregate** | — | **0.6520 ± 0.0212** | **0.6960 ± 0.0482** | 0.8410 ± 0.0330 |

**rows = `MS`**

| factor | n_values_realised | budget A | budget B | knn_proto (unrestricted) |
|---|---|---|---|---|
| `floor_hue` | [2] | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 |
| `object_hue` | [2] | 0.7617 ± 0.0911 | 0.9732 ± 0.0144 | 1.0000 |
| `orientation` | [3] | 0.1826 ± 0.1349 | 0.3622 ± 0.1215 | 0.8806 |
| `scale` | [3] | 0.0336 ± 0.0186 | 0.0369 ± 0.0381 | 0.4697 |
| `shape` | [3] | 0.2751 ± 0.0586 | 0.3742 ± 0.0289 | 0.7247 |
| `wall_hue` | [2] | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 |
| **aggregate** | — | **0.5422 ± 0.0430** | **0.6244 ± 0.0222** | 0.8458 ± 0.0421 |

## 4. ⚠ The declared WEAK companion (random subsample at the same bytes)

| budget | k-medoid stronger on READ ERROR (per seed) | k-medoid stronger on in-corner (per seed) | subsample in-corner |
|---|---|---|---|
| A | **[True, True, True]** | [True, True, False] | 0.6411 ± 0.0915 |
| B | **[True, True, True]** | [False, True, True] | 0.6316 ± 0.0218 |

## 5. ⚠ Row-set diagnostics — `val` is NOT exchangeable with `fit`

| row set | n | feature variance | n_values_realised per factor |
|---|---|---|---|
| `fit` | 32256 | 0.2229 | floor_hue 10, object_hue 10, orientation 15, scale 8, shape 4, wall_hue 10 |
| `val` | 512 | 0.0924 | floor_hue 1, object_hue 10, orientation 15, scale 8, shape 4, wall_hue 2 |
| `A` | 512 | 0.2006 | floor_hue 5, object_hue 10, orientation 15, scale 8, shape 4, wall_hue 10 |
| `B` | 72 | 0.0346 | floor_hue 2, object_hue 2, orientation 3, scale 3, shape 1, wall_hue 2 |
| `MS` | 198 | 0.0547 | floor_hue 2, object_hue 2, orientation 3, scale 3, shape 3, wall_hue 2 |

## 6. ⛔ knn_proto regression check (acceptance item 4)

| seed | measured probe in-corner (full precision) | published | Δ | bit-identical |
|---|---|---|---|---|
| 0 | `0.8762626262626263` | `0.8762626262626263` | **+0** | **True** |
| 1 | `0.8919597989949748` | `0.8919597989949748` | **+0** | **True** |
| 2 | `0.8969849246231156` | `0.8969849246231156` | **+0** | **True** |

## 7. Selection stability (⚠ the `val` structure, measured)

| seed | budget | selected | argmax on val half 1 | half 2 | halves agree | best−second gap | trial spread |
|---|---|---|---|---|---|---|---|
| 0 | A | k=5,cos,zsco | k=5,cos,zsco | k=1,cos,none | **False** | 0.0221 | 0.1177 |
| 0 | B | k=1,euc,none | k=1,cos,none | k=1,euc,none | **False** | 0.0026 | 0.0902 |
| 1 | A | k=1,cos,none | k=1,cos,none | k=1,cos,zsco | **False** | 0.0033 | 0.1154 |
| 1 | B | k=3,euc,none | k=3,euc,none | k=3,cos,none | **False** | 0.0007 | 0.0885 |
| 2 | A | k=3,cos,none | k=3,cos,none | k=3,cos,none | **True** | 0.0182 | 0.1539 |
| 2 | B | k=3,cos,none | k=3,euc,none | k=3,cos,none | **False** | 0.0039 | 0.1019 |

## 8. The persisted atom sets (⭐ gen-dynamical-read must LOAD these)

| seed | budget | file | sha256 | atoms/well | scope | deployed bytes |
|---|---|---|---|---|---|---|
| 0 | A | `atoms_s0_A.npz` | `e28c1f63a206fe1bff00b3db…` | 64 | per_well | 524,288 |
| 0 | B | `atoms_s0_B.npz` | `72097d361ecbf4b152a368fa…` | 512 | per_well | 4,194,304 |
| 1 | A | `atoms_s1_A.npz` | `9fabb694388a43d469a33be0…` | 64 | per_well | 524,288 |
| 1 | B | `atoms_s1_B.npz` | `7c0cf9198720cb31e8b33c5c…` | 512 | per_well | 4,194,304 |
| 2 | A | `atoms_s2_A.npz` | `1ab0bd6f2478f7728cff972e…` | 64 | per_well | 524,288 |
| 2 | B | `atoms_s2_B.npz` | `31562d805d0bc362ecb954f0…` | 512 | per_well | 4,194,304 |
