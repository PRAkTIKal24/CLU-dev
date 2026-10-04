# camels-pur-tripwire — results tables

Protocol: Li et al. 2025 PUR; 531 basins; 7 SPATIALLY-BLOCKED rounds; region rule PINNED from mhpi/hydroDL example/PUR/trainPUR-Reg.py L54-62 (regionDivide=[[1,2],[3,6],[4,5,7],[9,10],[8,11,12,13],[14,15,16,18],[17]] over HUC-2); store window 1980-10-01..1999-09-30; test 1995-10-01..1999-09-30 (1461 d); Daymet forcing; NSE unclipped, obs<0 dropped, median over 531


## Top 25 VERDICT-ADMISSIBLE arms (pinned partition, non-leaky, in-budget <= 1966080 B/round)

| # | arm | bytes | median NSE | mean | n(NSE<=0) | n basins |
|---|---|---|---|---|---|---|
| 1 | DAY W=1 stat=none sc=global L49365_rand0 k=25 unif tgt=pmean | 1184760 | **0.2743** | -0.3731 | 112 | 531 |
| 2 | DAY W=1 stat=none sc=global L49365_rand0 k=25 dist tgt=pmean | 1184760 | **0.2743** | -0.3731 | 112 | 531 |
| 3 | DAY W=1 stat=none sc=global L81920_rand2 k=25 unif tgt=pmean | 1966080 | **0.2721** | -0.3551 | 109 | 531 |
| 4 | DAY W=1 stat=none sc=global L81920_rand2 k=25 dist tgt=pmean | 1966080 | **0.2721** | -0.3552 | 109 | 531 |
| 5 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=pmean | 1966080 | **0.2672** | -0.3870 | 121 | 531 |
| 6 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=pmean | 1966080 | **0.2672** | -0.3870 | 121 | 531 |
| 7 | DAY W=1 stat=none sc=global L81920_rand0 k=25 dist tgt=pmean | 1966080 | **0.2672** | -0.3870 | 121 | 531 |
| 8 | DAY W=1 stat=none sc=global L81920_rand0 k=25 dist tgt=pmean | 1966080 | **0.2672** | -0.3870 | 121 | 531 |
| 9 | DAY W=1 stat=none sc=global L81920_rand1 k=25 unif tgt=pmean | 1966080 | **0.2659** | -0.3787 | 105 | 531 |
| 10 | DAY W=1 stat=none sc=global L81920_rand1 k=25 dist tgt=pmean | 1966080 | **0.2659** | -0.3787 | 105 | 531 |
| 11 | DAY W=1 stat=none sc=global L49365_rand1 k=25 dist tgt=pmean | 1184760 | **0.2654** | -0.3777 | 109 | 531 |
| 12 | DAY W=1 stat=none sc=global L49365_rand1 k=25 unif tgt=pmean | 1184760 | **0.2654** | -0.3777 | 109 | 531 |
| 13 | DAY W=1 stat=none sc=global L49365_rand2 k=25 dist tgt=pmean | 1184760 | **0.2613** | -0.3799 | 112 | 531 |
| 14 | DAY W=1 stat=none sc=global L49365_rand2 k=25 unif tgt=pmean | 1184760 | **0.2613** | -0.3798 | 112 | 531 |
| 15 | DAY W=1 stat=none sc=global L20000_rand0 k=25 unif tgt=pmean | 480000 | **0.2544** | -0.4170 | 110 | 531 |
| 16 | DAY W=1 stat=none sc=global L20000_rand0 k=25 dist tgt=pmean | 480000 | **0.2543** | -0.4171 | 110 | 531 |
| 17 | DAY W=30 stat=none sc=global L3255_kmeans0 k=3 dist tgt=pmean | 1966020 | **0.2431** | -0.1930 | 61 | 531 |
| 18 | DAY W=30 stat=none sc=global L3255_kmeans0 k=3 dist tgt=pmean | 1966020 | **0.2431** | -0.1930 | 61 | 531 |
| 19 | DAY W=30 stat=none sc=global L3255_kmeans0 k=3 unif tgt=pmean | 1966020 | **0.2431** | -0.1929 | 61 | 531 |
| 20 | DAY W=30 stat=none sc=global L3255_kmeans0 k=3 unif tgt=pmean | 1966020 | **0.2431** | -0.1929 | 61 | 531 |
| 21 | DAY W=30 stat=none sc=global L3255_kmeans0 k=5 dist tgt=pmean | 1966020 | **0.2421** | -0.1394 | 59 | 531 |
| 22 | DAY W=30 stat=none sc=global L3255_kmeans0 k=5 dist tgt=pmean | 1966020 | **0.2421** | -0.1394 | 59 | 531 |
| 23 | DAY W=30 stat=none sc=global L3255_kmeans0 k=5 unif tgt=pmean | 1966020 | **0.2413** | -0.1392 | 59 | 531 |
| 24 | DAY W=30 stat=none sc=global L3255_kmeans0 k=5 unif tgt=pmean | 1966020 | **0.2413** | -0.1392 | 59 | 531 |
| 25 | DAY W=30 stat=none sc=global L3255_kmeans0 k=5 unif tgt=raw | 1966020 | **0.2402** | -2.3452 | 90 | 531 |

## Top 10 OVER-BUDGET arms (upper bound — NOT admissible)

| # | arm | bytes | x budget | median NSE |
|---|---|---|---|---|
| 1 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=5 unif tgt=pmean | 3720000 | 1.89 | 0.2658 |
| 2 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=5 dist tgt=pmean | 3720000 | 1.89 | 0.2658 |
| 3 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=5 unif tgt=raw | 3720000 | 1.89 | 0.2620 |
| 4 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=10 unif tgt=pmean | 3720000 | 1.89 | 0.2620 |
| 5 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=5 dist tgt=raw | 3720000 | 1.89 | 0.2617 |
| 6 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=10 dist tgt=pmean | 3720000 | 1.89 | 0.2616 |
| 7 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=10 dist tgt=raw | 3720000 | 1.89 | 0.2493 |
| 8 | DAY W=30 stat=ord sc=global L5000_kmeans0 k=10 unif tgt=raw | 3720000 | 1.89 | 0.2490 |
| 9 | DAY W=1 stat=none sc=global L163840_rand0 k=25 unif tgt=pmean | 3932160 | 2.00 | 0.2473 |
| 10 | DAY W=1 stat=none sc=global L163840_rand0 k=25 dist tgt=pmean | 3932160 | 2.00 | 0.2473 |

## Top 10 LEAKY / out-of-protocol arms (NEVER admissible, labelled)

| # | arm | bytes | median NSE |
|---|---|---|---|
| 1 | COMPANION LEAKY_persistence_1day | 4 | 0.4362 |
| 2 | DAY W=1 stat=none sc=global L81920_rand1 k=25 unif tgt=oracle | 1966080 | 0.2900 |
| 3 | DAY W=1 stat=none sc=global L81920_rand1 k=25 dist tgt=oracle | 1966080 | 0.2900 |
| 4 | DAY W=1 stat=none sc=global L81920_rand2 k=25 unif tgt=oracle | 1966080 | 0.2884 |
| 5 | DAY W=1 stat=none sc=global L81920_rand2 k=25 dist tgt=oracle | 1966080 | 0.2883 |
| 6 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=oracle | 1966080 | 0.2876 |
| 7 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=oracle | 1966080 | 0.2876 |
| 8 | DAY W=1 stat=none sc=global L81920_rand0 k=25 dist tgt=oracle | 1966080 | 0.2876 |
| 9 | DAY W=1 stat=none sc=global L81920_rand0 k=25 dist tgt=oracle | 1966080 | 0.2876 |
| 10 | DAY W=1 stat=none sc=global L49365_rand0 k=25 dist tgt=oracle | 1184760 | 0.2850 |

## Sensitivity partitions (lat/lon k-means 7-way blocks)

| partition | best admissible non-leaky | arm | fires? |
|---|---|---|---|
| kmeans0 | 0.2658 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=pmean | no |
| kmeans1 | 0.2683 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=pmean | no |
| kmeans2 | 0.2683 | DAY W=1 stat=none sc=global L81920_rand0 k=25 unif tgt=pmean | no |

## KILL RULE vs EVERY PUR reference — E = 0.2743

| id | reference | S | status | protocol vs ours | margin_abs | margin_rel | FIRES |
|---|---|---|---|---|---|---|---|
| P01 | li2025_LSTM+dHBV_123 (FRONTIER) | 0.700814 | confirmed + reproduced (dC -0.000644) | exact | -0.4265 | -0.6086 | no |
| P02 | li2025_LSTM+dHBV_D+N | 0.685032 | confirmed | exact (fused forcing) | -0.4107 | -0.5996 | no |
| P03 | li2025_dHBV_123 | 0.684685 | confirmed + reproduced (dC -0.000220) | exact | -0.4104 | -0.5994 | no |
| P04 | li2025_LSTM+dHBV_D+M | 0.680872 | confirmed | exact (fused forcing) | -0.4066 | -0.5971 | no |
| P05 | li2025_LSTM+dHBV_N+M | 0.679321 | confirmed | exact (fused forcing) | -0.4050 | -0.5962 | no |
| P06 | li2025_dHBV_D+N | 0.672839 | confirmed | exact (fused forcing) | -0.3985 | -0.5923 | no |
| P07 | li2025_dHBV_N+M | 0.661231 | confirmed | exact (fused forcing) | -0.3869 | -0.5852 | no |
| P08 | li2025_LSTM_123 | 0.656228 | confirmed + reproduced (dC -0.001387) | exact | -0.3819 | -0.5820 | no |
| P09 | li2025_dHBV_D+M | 0.644732 | confirmed | exact (fused forcing) | -0.3704 | -0.5746 | no |
| P10 | li2025_LSTM+dHBV_daymet (PRIMARY) | 0.644398 | confirmed + reproduced (dC -0.001242) | exact | -0.3701 | -0.5743 | no |
| P11 | li2025_LSTM_D+M | 0.636369 | confirmed | exact (fused forcing) | -0.3621 | -0.5690 | no |
| P12 | li2025_LSTM+dHBV_maurer | 0.635444 | confirmed + reproduced (dC -0.000457) | exact (maurer forcing) | -0.3612 | -0.5683 | no |
| P13 | li2025_LSTM_D+N | 0.634398 | confirmed | exact (fused forcing) | -0.3601 | -0.5676 | no |
| P14 | li2025_LSTM_N+M | 0.626939 | confirmed | exact (fused forcing) | -0.3526 | -0.5625 | no |
| P15 | li2025_dHBV_daymet | 0.622278 | confirmed + reproduced (dC -0.001353) | exact | -0.3480 | -0.5592 | no |
| P16 | li2025_LSTM+dHBV_nldas | 0.618255 | confirmed + reproduced (dC +0.000576) | exact (nldas forcing) | -0.3440 | -0.5563 | no |
| P17 | li2025_dHBV_nldas | 0.592306 | confirmed + reproduced (dC +0.000103) | exact (nldas forcing) | -0.3180 | -0.5369 | no |
| P18 | li2025_dHBV_maurer | 0.59161 | confirmed + reproduced (dC -0.000851) | exact (maurer forcing) | -0.3173 | -0.5364 | no |
| P19 | li2025_LSTM_daymet | 0.578365 | confirmed + reproduced (dC -0.001131) | exact | -0.3041 | -0.5257 | no |
| P20 | li2025_LSTM_maurer | 0.56164 | confirmed + reproduced (dC +0.001928) | exact (maurer forcing) | -0.2873 | -0.5116 | no |
| P21 | li2025_LSTM_nldas | 0.546217 | confirmed + reproduced (dC -0.002218) | exact (nldas forcing) | -0.2719 | -0.4978 | no |
| P22 | li2025_(LSTM+dHBV)seed123 | 0.705154 | confirmed (Table D4) | exact, seed-ensemble | -0.4309 | -0.6110 | no |
| P23 | li2025_(LSTM+dHBV)123+LSTMmulti | 0.68923 | confirmed (Table D4) | exact | -0.4149 | -0.6020 | no |
| P24 | li2025_(LSTM+dHBV)seed123+LSTMseedmulti | 0.701504 | confirmed (Table D4) | exact, seed-ensemble | -0.4272 | -0.6090 | no |
| P25 | li2025_LSTMseedmulti | 0.665723 | confirmed (Table D4) | exact, seed-ensemble | -0.3914 | -0.5880 | no |
| P26 | li2025_LSTMmulti | 0.623755 | confirmed (Table D4) | exact | -0.3495 | -0.5603 | no |
| P27 | feng2023_dHBV(bt,gt)_daymet_PUR | 0.62 | confirmed (own sec. 3.2 text) | 7 regions, test 1989-1999 | -0.3457 | -0.5576 | no |
| P28 | feng2023_dHBV_daymet_PUR | 0.6 | confirmed (own sec. 3.2 text) | 7 regions, test 1989-1999 | -0.3257 | -0.5428 | no |
| P29 | feng2023_dHBV(bt,gt)_maurer_PUR | 0.59 | confirmed (own sec. 3.2 text) | 7 regions, test 1989-1999 (maurer) | -0.3157 | -0.5351 | no |
| P30 | feng2023_dHBV_maurer_PUR | 0.56 | confirmed (own sec. 3.2 text) | 7 regions, test 1989-1999 (maurer) | -0.2857 | -0.5102 | no |
| P31 | feng2023_LSTM_daymet_PUR | 0.55 | confirmed (own sec. 3.2 text) | 7 regions, test 1989-1999 | -0.2757 | -0.5013 | no |
| P32 | feng2023_LSTM_maurer_PUR | 0.55 | confirmed (own sec. 3.2 text) | 7 regions, test 1989-1999 (maurer) | -0.2757 | -0.5013 | no |

**dies = False** — fired against 0/32 references: none


Retention ratio E_PUR / E_PUB = 0.2743 / 0.6369 = **0.431**
