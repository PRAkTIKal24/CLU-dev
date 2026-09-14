# gen-r0-wide-refit — tables (seeds [0, 1, 2], base 194b66a)

## ⛔ THE GATE — wide-R0 vs Montero-R0 read margin (correlation − random)

| statistic | wide-R0 | Montero-R0 (shipped, V1) | Δ | Δ in pooled spreads | verdict |
|---|---|---|---|---|---|
| aggregate | +0.1410 ± 0.0097 | +0.1504 ± 0.0094 | -0.0094 | 0.70× (tol 3×) | **CONSISTENT** |
| shape | +0.1813 ± 0.0228 | +0.2008 ± 0.0165 | -0.0195 | 0.70× (tol 3×) | **CONSISTENT** |

⭐ **Overall gate verdict: CONSISTENT**

## Grouping agreement (R24 rider 1)

| quantity | per seed | mean ± sd |
|---|---|---|
| ari_wide_vs_montero | +0.1027 / +0.1099 / +0.1323 | +0.1150 ± 0.0126 |
| perm_agreement_wide_vs_montero | +0.3892 / +0.4326 / +0.4546 | +0.4255 ± 0.0272 |
| ari_chance_floor_vs_monteros_random_arm | -0.0008 / +0.0061 / -0.0021 | +0.0011 ± 0.0036 |
| ari_rowsplit_stability_same_cache | +0.4961 / +0.5030 / +0.6252 | +0.5414 ± 0.0593 |
| perm_rowsplit_stability_same_cache | +0.7290 / +0.7676 / +0.8237 | +0.7734 ± 0.0389 |
| channel identity: mean_diagonal_corr | +0.3192 / +0.3168 / +0.3643 | +0.3334 ± 0.0219 |
| channel identity: frac_diagonal_above_0.9 | +0.0389 / +0.0319 / +0.0229 | +0.0312 ± 0.0065 |
| channel identity: frac_argmax_is_diagonal | +0.0772 / +0.0849 / +0.0815 | +0.0812 ± 0.0031 |
| channel identity: mean_best_match_abs_corr | +0.7303 / +0.7195 / +0.7111 | +0.7203 ± 0.0079 |
| cross-seed ARI, Montero arm | {'0v1': 0.056542505627147396, '0v2': 0.07648368274588578, '1v2': 0.05826136587652641} | +0.0638 |
| cross-seed ARI, wide arm | {'0v1': 0.05406689170855161, '0v2': 0.06420080370201765, '1v2': 0.056112969003120204} | +0.0581 |

## R0's wide ladder coordinates

| | aggregate | `shape` |
|---|---|---|
| no-store ceiling | +0.9897 ± 0.0021 | +0.9733 ± 0.0072 |
| read, correlation | +0.5956 ± 0.0112 | +0.5671 ± 0.0438 |
| read, random | +0.4547 ± 0.0107 | +0.3857 ± 0.0211 |
| ⭐ margin | +0.1410 ± 0.0097 (14.5×) | +0.1813 ± 0.0228 (8.0×) |

### In-corner read (balanced accuracy is the gate statistic; AUROC the companion)

| arm | balacc (11 cells) | AUROC (11 cells) | balacc (all 13) |
|---|---|---|---|
| real_features | +0.9894 ± 0.0023 | +0.9993 ± 0.0005 | +0.9798 ± 0.0091 |
| correlation | +0.6899 ± 0.0292 | +0.8223 ± 0.0303 | +0.6835 ± 0.0282 |
| random | +0.5854 ± 0.0311 | +0.6903 ± 0.0511 | +0.5800 ± 0.0318 |
| ⭐ margin (corr − rand) | +0.1045 ± 0.0510 (2.0×) | +0.1320 ± 0.0606 (2.2×) | |

⛔ shared-corner pair (R23.3), never pooled:

| cell | real features | correlation | random |
|---|---|---|---|
| montero | +1.0000 ± 0.0000 | +0.7825 ± 0.0786 | +0.6002 ± 0.0720 |
| wide-000 | +0.8542 ± 0.0967 | +0.5133 ± 0.0188 | +0.5000 ± 0.0000 |

### Leakage (R23.2), MIG, DCI-C

| | pooled leakage | `shape` leakage | MIG | DCI-C |
|---|---|---|---|---|
| correlation | -0.0960 ± 0.0246 | -0.0854 ± 0.0143 | +0.1838 ± 0.0178 | +0.1630 ± 0.0061 |
| random | -0.1620 ± 0.0389 | -0.1276 ± 0.0349 | +0.0365 ± 0.0104 | +0.0330 ± 0.0079 |
| ⭐ margin | -0.0660 ± 0.0144 (4.59×, rand−corr) | -0.0422 ± 0.0271 (1.56×, rand−corr) | +0.1473 ± 0.0154 | +0.1299 ± 0.0036 |

**Non-degeneracy proof:** 180 (well × factor) cells over both arms and all seeds — **0 non-finite**, range **[-0.6645, +0.1569]**.

**Ruling-2 fallback: TRIGGERED** — `shape`-leakage margin -0.0422 ± 0.0271 = 1.56× its own seed spread (threshold 3×). `shape`-leakage is declared UNDERPOWERED, in those words.
