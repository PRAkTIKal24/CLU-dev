
### CAMELS regional ladder (best over k x weight x target)

| window | scaling | L | selection | store bytes | in budget | sub | best median NSE (447) | best k / target |
|---|---|---|---|---|---|---|---|---|
| 1d | global | 250 | random | 33,000 | yes | 1 | **0.0768** | k=5 std |
| 1d | global | 1,000 | random | 132,000 | yes | 1 | **0.1611** | k=10 raw |
| 1d | global | 5,000 | random | 660,000 | yes | 1 | **0.2109** | k=25 raw |
| 1d | global | 14,894 | random | 1,966,008 | yes | 1 | **0.2520** | k=25 raw |
| 1d | perbasin | 250 | random | 33,000 | yes | 1 | **0.0824** | k=3 std |
| 1d | perbasin | 1,000 | random | 132,000 | yes | 1 | **0.1472** | k=10 std |
| 1d | perbasin | 5,000 | random | 660,000 | yes | 1 | **0.1999** | k=25 raw |
| 1d | perbasin | 14,894 | random | 1,966,008 | yes | 1 | **0.2367** | k=25 raw |
| 1d | raw | 250 | random | 33,000 | yes | 1 | **-0.0111** | k=25 std |
| 1d | raw | 1,000 | random | 132,000 | yes | 1 | **-0.0022** | k=25 std |
| 1d | raw | 5,000 | random | 660,000 | yes | 1 | **0.0182** | k=25 std |
| 1d | raw | 14,894 | random | 1,966,008 | yes | 1 | **0.0183** | k=25 std |
| 30d | global | 250 | kmeans | 178,000 | yes | 1 | **0.1928** | k=3 raw |
| 30d | global | 250 | random | 178,000 | yes | 1 | **0.0677** | k=10 std |
| 30d | global | 500 | kmeans | 356,000 | yes | 1 | **0.1983** | k=3 raw |
| 30d | global | 500 | random | 356,000 | yes | 1 | **0.0929** | k=5 raw |
| 30d | global | 1,000 | kmeans | 712,000 | yes | 1 | **0.2407** | k=3 raw |
| 30d | global | 1,000 | random | 712,000 | yes | 1 | **0.1124** | k=10 raw |
| 30d | global | 2,000 | kmeans | 1,424,000 | yes | 1 | **0.2536** | k=3 raw |
| 30d | global | 2,000 | random | 1,424,000 | yes | 1 | **0.1159** | k=10 raw |
| 30d | global | 2,761 | kmeans | 1,965,832 | yes | 1 | **0.2559** | k=3 raw |
| 30d | global | 2,761 | random | 1,965,832 | yes | 1 | **0.1551** | k=10 raw |
| 30d | global | 5,000 | random | 3,560,000 | **NO** | 3 | **0.1397** | k=10 raw |
| 30d | perbasin | 250 | kmeans | 178,000 | yes | 1 | **0.1323** | k=3 std |
| 30d | perbasin | 250 | random | 178,000 | yes | 1 | **0.0527** | k=10 std |
| 30d | perbasin | 500 | kmeans | 356,000 | yes | 1 | **0.1743** | k=3 std |
| 30d | perbasin | 500 | random | 356,000 | yes | 1 | **0.0655** | k=10 raw |
| 30d | perbasin | 1,000 | kmeans | 712,000 | yes | 1 | **0.2217** | k=3 std |
| 30d | perbasin | 1,000 | random | 712,000 | yes | 1 | **0.0974** | k=10 raw |
| 30d | perbasin | 2,000 | kmeans | 1,424,000 | yes | 1 | **0.2393** | k=3 std |
| 30d | perbasin | 2,000 | random | 1,424,000 | yes | 1 | **0.1247** | k=25 raw |
| 30d | perbasin | 2,761 | kmeans | 1,965,832 | yes | 1 | **0.2330** | k=3 std |
| 30d | perbasin | 2,761 | random | 1,965,832 | yes | 1 | **0.1365** | k=10 raw |
| 30d | perbasin | 5,000 | random | 3,560,000 | **NO** | 3 | **0.1263** | k=10 raw |
| 30d | raw | 250 | kmeans | 178,000 | yes | 1 | **0.0386** | k=3 std |
| 30d | raw | 250 | random | 178,000 | yes | 1 | **0.0165** | k=25 std |
| 30d | raw | 500 | kmeans | 356,000 | yes | 1 | **0.0443** | k=5 std |
| 30d | raw | 500 | random | 356,000 | yes | 1 | **0.0191** | k=25 std |
| 30d | raw | 1,000 | kmeans | 712,000 | yes | 1 | **0.0429** | k=5 std |
| 30d | raw | 1,000 | random | 712,000 | yes | 1 | **0.0133** | k=25 std |
| 30d | raw | 2,000 | kmeans | 1,424,000 | yes | 1 | **0.0479** | k=5 std |
| 30d | raw | 2,000 | random | 1,424,000 | yes | 1 | **0.0243** | k=25 std |
| 30d | raw | 2,761 | kmeans | 1,965,832 | yes | 1 | **0.0485** | k=25 std |
| 30d | raw | 2,761 | random | 1,965,832 | yes | 1 | **0.0155** | k=25 std |
| 30d | raw | 5,000 | random | 3,560,000 | **NO** | 3 | **0.0127** | k=25 std |
| 365d | global | 250 | kmeans | 1,853,000 | yes | 1 | **0.0543** | k=3 std |
| 365d | global | 250 | random | 1,853,000 | yes | 1 | **0.0376** | k=25 std |
| 365d | global | 265 | kmeans | 1,964,180 | yes | 1 | **0.0561** | k=3 std |
| 365d | global | 265 | random | 1,964,180 | yes | 1 | **0.0240** | k=25 std |
| 365d | global | 500 | kmeans | 3,706,000 | **NO** | 3 | **0.0541** | k=5 std |
| 365d | global | 500 | random | 3,706,000 | **NO** | 3 | **0.0271** | k=25 std |
| 365d | global | 1,000 | kmeans | 7,412,000 | **NO** | 3 | **0.0524** | k=3 std |
| 365d | global | 1,000 | random | 7,412,000 | **NO** | 3 | **0.0137** | k=25 std |
| 365d | global | 2,000 | kmeans | 14,824,000 | **NO** | 3 | **0.0479** | k=3 std |
| 365d | global | 2,000 | random | 14,824,000 | **NO** | 3 | **0.0341** | k=25 std |
| 365d | global | 5,000 | random | 37,060,000 | **NO** | 3 | **0.0173** | k=25 std |
| 365d | perbasin | 250 | kmeans | 1,853,000 | yes | 1 | **0.0382** | k=3 std |
| 365d | perbasin | 250 | random | 1,853,000 | yes | 1 | **0.0118** | k=25 std |
| 365d | perbasin | 265 | kmeans | 1,964,180 | yes | 1 | **0.0363** | k=5 std |
| 365d | perbasin | 265 | random | 1,964,180 | yes | 1 | **0.0138** | k=25 std |
| 365d | perbasin | 500 | kmeans | 3,706,000 | **NO** | 3 | **0.0328** | k=10 std |
| 365d | perbasin | 500 | random | 3,706,000 | **NO** | 3 | **0.0013** | k=25 std |
| 365d | perbasin | 1,000 | kmeans | 7,412,000 | **NO** | 3 | **0.0303** | k=25 std |
| 365d | perbasin | 1,000 | random | 7,412,000 | **NO** | 3 | **0.0078** | k=25 std |
| 365d | perbasin | 2,000 | kmeans | 14,824,000 | **NO** | 3 | **0.0351** | k=10 std |
| 365d | perbasin | 2,000 | random | 14,824,000 | **NO** | 3 | **0.0071** | k=25 raw |
| 365d | perbasin | 5,000 | random | 37,060,000 | **NO** | 3 | **-0.0098** | k=25 std |
| 365d | raw | 250 | kmeans | 1,853,000 | yes | 1 | **0.0504** | k=3 std |
| 365d | raw | 250 | random | 1,853,000 | yes | 1 | **0.0322** | k=25 std |
| 365d | raw | 265 | kmeans | 1,964,180 | yes | 1 | **0.0509** | k=3 std |
| 365d | raw | 265 | random | 1,964,180 | yes | 1 | **0.0253** | k=25 std |
| 365d | raw | 500 | kmeans | 3,706,000 | **NO** | 3 | **0.0450** | k=3 std |
| 365d | raw | 500 | random | 3,706,000 | **NO** | 3 | **0.0111** | k=25 std |
| 365d | raw | 1,000 | kmeans | 7,412,000 | **NO** | 3 | **0.0439** | k=25 std |
| 365d | raw | 1,000 | random | 7,412,000 | **NO** | 3 | **0.0225** | k=25 std |
| 365d | raw | 2,000 | kmeans | 14,824,000 | **NO** | 3 | **0.0406** | k=25 std |
| 365d | raw | 2,000 | random | 14,824,000 | **NO** | 3 | **0.0312** | k=25 std |
| 365d | raw | 5,000 | random | 37,060,000 | **NO** | 3 | **0.0186** | k=25 std |

### CAMELS LOCAL (same-basin) arms — statics dropped

| window | scaling | L/basin | total bytes | x budget | in budget | sub | best median NSE (447) |
|---|---|---|---|---|---|---|---|
| 30d | perbasin | 5 | 1,603,620 | 0.82x | yes | 1 | **-0.0432** |
| 30d | perbasin | 250 | 80,181,000 | 40.78x | **NO** | 1 | **0.0957** |
| 30d | perbasin | 1,000 | 320,724,000 | 163.13x | **NO** | 1 | **0.1565** |
| 30d | perbasin | 3,287 | 1,054,219,788 | 536.2x | **NO** | 1 | **0.2173** |
| 30d | raw | 5 | 1,603,620 | 0.82x | yes | 1 | **-0.0476** |
| 30d | raw | 250 | 80,181,000 | 40.78x | **NO** | 1 | **0.0472** |
| 30d | raw | 1,000 | 320,724,000 | 163.13x | **NO** | 1 | **0.0532** |
| 30d | raw | 3,287 | 1,054,219,788 | 536.2x | **NO** | 1 | **0.0516** |
| 365d | perbasin | 250 | 969,606,000 | 493.17x | **NO** | 3 | **0.0449** |
| 365d | perbasin | 1,000 | 3,878,424,000 | 1972.67x | **NO** | 3 | **0.0255** |
| 365d | perbasin | 3,287 | 12,748,379,688 | 6484.16x | **NO** | 3 | **0.0159** |
| 365d | raw | 250 | 969,606,000 | 493.17x | **NO** | 3 | **0.0394** |
| 365d | raw | 1,000 | 3,878,424,000 | 1972.67x | **NO** | 3 | **0.0304** |
| 365d | raw | 3,287 | 12,748,379,688 | 6484.16x | **NO** | 3 | **0.0055** |

### CAMELS mandatory companion rows

| arm | bytes | in budget | in protocol | median NSE (447) | mean | n(NSE<=0) |
|---|---|---|---|---|---|---|
| mean_train | 2,124 | yes | yes | **-0.0073** | -0.0159 | 447 |
| mean_test | 2,124 | yes | yes | **0.0000** | 0.0000 | 447 |
| doy_clim | 777,384 | yes | yes | **0.0111** | 0.0586 | 202 |
| persistence | 4 | yes | **NO — DIFFERENT TASK** | **0.4434** | 0.4043 | 72 |

### DECLARED POST-HOC (unregistered) classical arms

| arm | window | target | bytes | in budget | median NSE (447) |
|---|---|---|---|---|---|
| ridge_per_basin | 30d | raw | 320,724 | yes | **0.4461** |
| ridge_pooled | 30d | raw | 712 | yes | **0.3480** |
| ridge_per_basin | 30d | std | 320,724 | yes | **0.4462** |
| ridge_pooled | 30d | std | 712 | yes | **0.3427** |
| ridge_per_basin | 365d | raw | 3,878,424 | **NO** | **0.1801** |
| ridge_pooled | 365d | raw | 7,412 | yes | **0.2987** |
| ridge_per_basin | 365d | std | 3,878,424 | **NO** | **0.1800** |
| ridge_pooled | 365d | std | 7,412 | yes | **0.3254** |

### N-CMAPSS DS02 — the criterion-2 rows (NOT PUBLISHED anywhere; supplied here)

| baseline | inputs | state bytes | RMSE [cycles] | s x 1e5 |
|---|---|---|---|---|
| mean_RUL | none | 4 | **19.904** | 10.566 |
| affine_cycle_index | cycle | 8 | **12.393** | 4.353 |
| affine_cycle_index_clip0 | cycle | 8 | **12.393** | 4.353 |
| mean_EOL_minus_cycle | cycle | 8 | **11.973** | 4.155 |

### N-CMAPSS DS02 — exemplar-store ladder (best over k)

| representation | L | dim | bytes | in budget | best RMSE | s x 1e5 | best k |
|---|---|---|---|---|---|---|---|
| hi14 (euclid) | 332 | 280 | 373,168 | yes | **10.362** | 3.383 | 25 |
| hi14 (euclid) | 422 | 70 | 119,848 | yes | **9.510** | 3.086 | 5 |
| hi14 (euclid) | 446 | 14 | 26,760 | yes | **8.561** | 2.817 | 5 |
| hi1d (dtw) | 332 | 20 | 27,888 | yes | **11.443** | 3.924 | 25 |
| hi1d (euclid) | 422 | 5 | 10,128 | yes | **10.663** | 3.508 | 10 |
| hi1d (euclid) | 446 | 1 | 3,568 | yes | **11.819** | 4.261 | 25 |
| resid (raw) | 250 | 14 | 15,000 | yes | **9.494** | 3.239 | 25 |
| resid (raw) | 500 | 14 | 30,000 | yes | **9.466** | 3.214 | 25 |
| resid (raw) | 1,000 | 14 | 60,000 | yes | **9.099** | 3.086 | 10 |
| resid (raw) | 2,000 | 14 | 120,000 | yes | **8.575** | 2.917 | 10 |
| resid (raw) | 5,000 | 14 | 300,000 | yes | **8.192** | 2.762 | 25 |
| resid (raw) | 32,768 | 14 | 1,966,080 | yes | **7.988** | 2.698 | 25 |
| resid_cycle (raw) | 250 | 15 | 16,000 | yes | **9.293** | 3.111 | 5 |
| resid_cycle (raw) | 500 | 15 | 32,000 | yes | **8.416** | 2.816 | 5 |
| resid_cycle (raw) | 1,000 | 15 | 64,000 | yes | **9.127** | 3.060 | 10 |
| resid_cycle (raw) | 2,000 | 15 | 128,000 | yes | **9.175** | 3.070 | 25 |
| resid_cycle (raw) | 5,000 | 15 | 320,000 | yes | **8.699** | 2.904 | 25 |
| resid_cycle (raw) | 30,720 | 15 | 1,966,080 | yes | **8.772** | 2.946 | 25 |
| w_xs (std) | 250 | 18 | 19,000 | yes | **19.965** | 11.026 | 25 |
| w_xs (std) | 500 | 18 | 38,000 | yes | **19.947** | 10.490 | 25 |
| w_xs (std) | 1,000 | 18 | 76,000 | yes | **18.683** | 9.487 | 10 |
| w_xs (std) | 2,000 | 18 | 152,000 | yes | **17.775** | 8.938 | 5 |
| w_xs (std) | 5,000 | 18 | 380,000 | yes | **16.390** | 7.794 | 5 |
| w_xs (std) | 25,869 | 18 | 1,966,044 | yes | **14.010** | 5.764 | 5 |
| w_xs_resid (std) | 250 | 32 | 33,000 | yes | **13.786** | 5.291 | 5 |
| w_xs_resid (std) | 500 | 32 | 66,000 | yes | **12.818** | 4.827 | 5 |
| w_xs_resid (std) | 1,000 | 32 | 132,000 | yes | **12.388** | 4.743 | 5 |
| w_xs_resid (std) | 2,000 | 32 | 264,000 | yes | **11.169** | 3.954 | 5 |
| w_xs_resid (std) | 5,000 | 32 | 660,000 | yes | **10.217** | 3.414 | 10 |
| w_xs_resid (std) | 14,894 | 32 | 1,966,008 | yes | **9.704** | 3.249 | 10 |
