# w13 hub/encoder-adapt — CROSS-ARM

**3 seeds (0, 1, 2) × 3 configurations = 9 arm-runs.** ⛔ That is 3 SEEDS, never “9 seeds”.

Condition `test`, K\* = 32.

## Equal steps and equal images — printed, not claimed

| seed | arm | steps | images | seconds | s/step | enc params in optimiser |
|---|---|---|---|---|---|---|
| 0 | frozen | 400 | 25600 | 2297.3 | 5.743 | 0 |
| 0 | adapter | 400 | 25600 | 2368.3 | 5.921 | 131,072 |
| 0 | anchor | 400 | 25600 | 2523.4 | 6.309 | 428,576 |
| 1 | frozen | 400 | 25600 | 2291.2 | 5.728 | 0 |
| 1 | adapter | 400 | 25600 | 2307.9 | 5.770 | 131,072 |
| 1 | anchor | 400 | 25600 | 2309.0 | 5.772 | 428,576 |
| 2 | frozen | 400 | 25600 | 2211.0 | 5.528 | 0 |
| 2 | adapter | 400 | 25600 | 2047.0 | 5.117 | 131,072 |
| 2 | anchor | 400 | 25600 | 1640.0 | 4.100 | 428,576 |

## THE MARGIN — single design − best learned baseline, per seed

| seed | arm | single design | best learned | margin | Δ vs frozen |
|---|---|---|---|---|---|
| 0 | frozen | 23.3119 | mlp_180k 22.7431 | **+0.5688** |  |
| 0 | adapter | 22.0642 | mlp_180k 20.4404 | **+1.6239** | +1.0550 |
| 0 | anchor | 25.5688 | mlp_180k 24.2385 | **+1.3303** | +0.7615 |
| 1 | frozen | 11.2018 | mlp_180k 11.2294 | **-0.0275** |  |
| 1 | adapter | 13.3486 | mlp_180k 12.9817 | **+0.3670** | +0.3945 |
| 1 | anchor | 11.1560 | mlp_180k 11.1651 | **-0.0092** | +0.0183 |
| 2 | frozen | 20.2936 | mlp_180k 19.1927 | **+1.1009** |  |
| 2 | adapter | 18.2844 | mlp_180k 17.9817 | **+0.3028** | -0.7982 |
| 2 | anchor | 20.8165 | mlp_180k 19.5138 | **+1.3028** | +0.2018 |

## Factor probe — MARGIN over random init, PER SEED, never pooled

| seed | arm | aggregate margin | exact-match margin |
|---|---|---|---|
| 0 | frozen | +0.0001 | -0.0034 |
| 0 | adapter | +0.0007 | -0.0024 |
| 0 | anchor | -0.0016 | -0.0178 |
| 1 | frozen | -0.0160 | -0.1003 |
| 1 | adapter | -0.0064 | -0.0449 |
| 1 | anchor | -0.0059 | -0.0371 |
| 2 | frozen | +0.0095 | +0.0432 |
| 2 | adapter | +0.0077 | +0.0308 |
| 2 | anchor | +0.0088 | +0.0381 |

## `anchor` — distance from the frozen encoder (REPORTED, never tuned)

| seed | feature rel-L2² | weight rel-L2² |
|---|---|---|
| 0 | 0.002146448940038681 | 0.0003244956023991108 |
| 1 | 0.002392713911831379 | 0.00024709958233870566 |
| 2 | 0.0017106134910136461 | 0.0003236146876588464 |
