# camels-pub-tripwire — results tables

Protocol: Li et al. 2025 PUB; 531 basins; 10 folds (RECONSTRUCTED, not published); store window 1980-10-01..1999-09-30; test 1995-10-01..1999-09-30 (1461 d); Daymet forcing; NSE unclipped, obs<0 dropped, median over 531


## Top 25 VERDICT-ADMISSIBLE arms (non-leaky, in-budget <= 1966080 B, full test window)

| # | arm | bytes | median NSE | mean | n(NSE<=0) | n basins |
|---|---|---|---|---|---|---|
| 1 | DONOR key=clim payload=concurrent Lall k=5 dist norm=pmean fp16 | 1412874 | **0.6435** | 0.1574 | 64 | 531 |
| 2 | DONOR key=clim payload=concurrent Lall k=3 dist norm=pmean fp16 | 1412874 | **0.6420** | 0.0955 | 68 | 531 |
| 3 | DONOR key=clim payload=concurrent Lall k=3 dist norm=raw fp16 | 1412874 | **0.6409** | -0.0442 | 71 | 531 |
| 4 | DONOR key=clim payload=concurrent Lall k=5 dist norm=raw fp16 | 1412874 | **0.6405** | 0.0206 | 70 | 531 |
| 5 | DONOR key=clim payload=concurrent Lall k=5 dist norm=pmean fp16 | 1412874 | **0.6405** | 0.1381 | 56 | 531 |
| 6 | DONOR key=clim payload=concurrent Lall k=5 dist norm=raw fp16 | 1412874 | **0.6390** | 0.0047 | 64 | 531 |
| 7 | DONOR key=clim payload=concurrent Lall k=5 dist norm=qmean fp16 | 1414782 | **0.6389** | -0.0313 | 75 | 531 |
| 8 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=pmean fp16 | 1397610 | **0.6371** | -0.9207 | 69 | 531 |
| 9 | DONOR key=clim payload=concurrent Lall k=3 dist norm=qmean fp16 | 1414782 | **0.6359** | -0.0311 | 80 | 531 |
| 10 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=pmean fp16 | 1397610 | **0.6357** | -0.8981 | 69 | 531 |
| 11 | DONOR key=clim payload=concurrent Lall k=3 dist norm=pmean fp16 | 1412874 | **0.6350** | 0.0367 | 64 | 531 |
| 12 | DONOR key=latlon payload=concurrent Lall k=3 dist norm=pmean fp16 | 1397610 | **0.6345** | -0.9474 | 67 | 531 |
| 13 | DONOR key=clim payload=concurrent Lall k=3 dist norm=raw fp16 | 1412874 | **0.6341** | -0.1049 | 80 | 531 |
| 14 | DONOR key=clim payload=concurrent Lall k=3 dist norm=qmean fp16 | 1414782 | **0.6335** | -0.0825 | 85 | 531 |
| 15 | DONOR key=clim payload=concurrent Lall k=3 dist norm=raw fp16 | 1412874 | **0.6325** | -0.1284 | 70 | 531 |
| 16 | DONOR key=latlon payload=concurrent Lall k=3 dist norm=pmean fp16 | 1397610 | **0.6324** | -0.9653 | 72 | 531 |
| 17 | DONOR key=clim payload=concurrent Lall k=5 dist norm=qmean fp16 | 1414782 | **0.6320** | -0.0451 | 78 | 531 |
| 18 | DONOR key=clim payload=concurrent Lall k=5 dist norm=qmean fp16 | 1414782 | **0.6310** | -0.0564 | 69 | 531 |
| 19 | DONOR key=clim payload=concurrent Lall k=3 dist norm=pmean fp16 | 1412874 | **0.6305** | 0.0601 | 76 | 531 |
| 20 | DONOR key=clim payload=concurrent Lall k=3 dist norm=qmean fp16 | 1414782 | **0.6290** | -0.1396 | 77 | 531 |
| 21 | DONOR key=clim payload=concurrent Lall k=5 dist norm=raw fp16 | 1412874 | **0.6281** | 0.0014 | 73 | 531 |
| 22 | DONOR key=clim payload=concurrent Lall k=5 dist norm=pmean fp16 | 1412874 | **0.6268** | 0.1413 | 69 | 531 |
| 23 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=raw fp16 | 1397610 | **0.6251** | -3.5212 | 80 | 531 |
| 24 | DONOR key=latlon payload=concurrent Lall k=10 dist norm=pmean fp16 | 1397610 | **0.6251** | -0.6703 | 62 | 531 |
| 25 | DONOR key=clim payload=concurrent L328_kmeans1 k=3 dist norm=pmean fp16 | 867866 | **0.6224** | 0.0893 | 69 | 531 |

## Top 10 OVER-BUDGET arms (unbounded upper bound — NOT admissible)

| # | arm | bytes | x budget | median NSE |
|---|---|---|---|---|
| 1 | DONOR key=clim payload=concurrent Lall k=5 dist norm=pmean fp32 | 2806668 | 1.43 | 0.6435 |
| 2 | DONOR key=clim payload=concurrent Lall k=3 dist norm=pmean fp32 | 2806668 | 1.43 | 0.6420 |
| 3 | DONOR key=clim payload=concurrent Lall k=3 dist norm=raw fp32 | 2806668 | 1.43 | 0.6409 |
| 4 | DONOR key=clim payload=concurrent Lall k=5 dist norm=raw fp32 | 2806668 | 1.43 | 0.6405 |
| 5 | DONOR key=clim payload=concurrent Lall k=5 dist norm=pmean fp32 | 2806668 | 1.43 | 0.6405 |
| 6 | DONOR key=clim payload=concurrent Lall k=5 dist norm=raw fp32 | 2806668 | 1.43 | 0.6390 |
| 7 | DONOR key=clim payload=concurrent Lall k=5 dist norm=qmean fp32 | 2808576 | 1.43 | 0.6389 |
| 8 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=pmean fp32 | 2791404 | 1.42 | 0.6371 |
| 9 | DONOR key=clim payload=concurrent Lall k=3 dist norm=qmean fp32 | 2808576 | 1.43 | 0.6359 |
| 10 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=pmean fp32 | 2791404 | 1.42 | 0.6357 |

## Top 10 LEAKY / out-of-protocol arms (NEVER admissible, labelled)

| # | arm | bytes | median NSE |
|---|---|---|---|
| 1 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=oracle fp16 | 1399518 | 0.6759 |
| 2 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=oracle fp32 | 2793312 | 0.6759 |
| 3 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=oracle fp16 | 1399518 | 0.6755 |
| 4 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=oracle fp32 | 2793312 | 0.6755 |
| 5 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=oracle fp16 | 1399518 | 0.6704 |
| 6 | DONOR key=latlon payload=concurrent Lall k=5 dist norm=oracle fp32 | 2793312 | 0.6704 |
| 7 | DONOR key=latlon payload=concurrent Lall k=3 dist norm=oracle fp32 | 2793312 | 0.6641 |
| 8 | DONOR key=latlon payload=concurrent Lall k=3 dist norm=oracle fp16 | 1399518 | 0.6641 |
| 9 | DONOR key=latlon payload=concurrent Lall k=3 dist norm=oracle fp32 | 2793312 | 0.6624 |
| 10 | DONOR key=latlon payload=concurrent Lall k=3 dist norm=oracle fp16 | 1399518 | 0.6624 |

## KILL RULE vs EVERY reference — E = 0.6369 (seed-mean; best single run 0.6435 = DONOR key=clim payload=concurrent Lall k=5 dist norm=pmean fp16)

| id | reference | S | status | protocol vs ours | margin_abs | margin_rel | FIRES |
|---|---|---|---|---|---|---|---|
| R1 | li2025_LSTM+dHBV_123 (FRONTIER) | 0.785833 | confirmed | exact | -0.1489 | -0.1895 | no |
| R2 | li2025_LSTM_123 | 0.768181 | confirmed | exact | -0.1313 | -0.1709 | no |
| R3 | li2025_dHBV_123 | 0.757749 | confirmed | exact | -0.1208 | -0.1595 | no |
| R4 | li2025_LSTM+dHBV_daymet (PRIMARY) | 0.74227 | confirmed | exact | -0.1054 | -0.1419 | no |
| R5 | heudorfer2026_EMB-LSTM_ALPHAEARTHplus | 0.721 | own Table 3 | 5-fold, test 1989-1999, bootstrap-median | -0.0841 | -0.1166 | no |
| R6 | li2025_dHBV_daymet | 0.706809 | confirmed | exact | -0.0699 | -0.0989 | no |
| R7 | li2025_LSTM_daymet | 0.702636 | confirmed | exact | -0.0657 | -0.0935 | no |
| R8 | heudorfer2026_baseLSTM_CAMELS27 | 0.696 | own Tables 2/3 | 5-fold, test 1989-1999, bootstrap-median | -0.0591 | -0.0849 | no |
| R9 | li2025_LSTM_maurer | 0.694156 | confirmed | exact (maurer forcing) | -0.0572 | -0.0825 | no |
| R10 | kratzert2019_LSTM_pub_ensemble | 0.6931 | Hub-recomputed, 531 rows | 12-fold, test 1995-2010, NLDAS-ext forcing | -0.0562 | -0.0811 | no |
| R11 | feng2023_dHBV(bt,gt)_daymet | 0.69 | provisional | 10-fold, test 1989-1999 | -0.0531 | -0.0769 | no |
| R12 | li2025_dHBV_nldas | 0.670636 | confirmed | exact (nldas forcing) | -0.0337 | -0.0503 | no |
| R13 | feng2023_LSTM_daymet | 0.66 | provisional | 10-fold, test 1989-1999 | -0.0231 | -0.0350 | no |
| R14 | feng2023_LSTM_maurer | 0.65 | confirmed (refs 8V5) | 10-fold, test 1989-1999 | -0.0131 | -0.0201 | **YES** |
| R15 | kratzert2019_SACSMA_calibrated | 0.6447 | Hub-recomputed, 531 rows | per-basin calibrated, test 1995-2010 | -0.0078 | -0.0121 | **YES** |
| R16 | feng2023_dHBV(bt,gt)_maurer | 0.64 | confirmed (refs 8V5) | 10-fold, test 1989-1999 | -0.0031 | -0.0048 | **YES** |
| R17 | heudorfer2026_baseLSTM_dynamiconly | 0.634 | own Table 2 | 5-fold, test 1989-1999, bootstrap-median | +0.0029 | +0.0046 | **YES** |
| R18 | qu2026_ALPHAEARTH | 0.612 | PRIMARY-CONFIRMED by this spoke (A6) | MISMATCH: 671 basins, 5-fold, test 2010-2014, bootstrap-CDF median | +0.0249 | +0.0407 | **YES** |
| R19 | kratzert2019_NWM_reanalysis | 0.5766 | Hub-recomputed, 531 rows | NWM retrospective, test 1995-2010 | +0.0603 | +0.1046 | **YES** |
| R20 | qu2026_CAMELSattributes | 0.553 | PRIMARY-CONFIRMED by this spoke (A6) | MISMATCH: 671 basins, 5-fold, test 2010-2014, bootstrap-CDF median | +0.0839 | +0.1518 | **YES** |

**dies = True** — fired against 7/20 references: R14, R15, R16, R17, R18, R19, R20
