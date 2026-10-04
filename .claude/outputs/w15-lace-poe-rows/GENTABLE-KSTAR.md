# GENTABLE-KSTAR — w15 spoke/lace-poe-rows

Every number below is read from the artefact named in `GENTABLE-KSTAR.json` beside it. Seed count is a column. ⛔ LACE and PoE are ADDED rows; the learned-rival columns are intact.

## PART A — the generation table (`exp_gen_cell`), scored run 3/3, generation-2 cache

| arm | n seeds | composition-B aggregate (s0 / s1 / s2) | mean ± sd | in-corner balacc mean | distinct tuples on B (mean) | deployed bytes (convention) |
|---|---|---|---|---|---|---|
| `clu` | 3 | 0.5903 / 0.6065 / 0.5880 | 0.5949 ± 0.0082 | 0.6568 | 9.0 | 524288 (the store's modes (n_modes·D·4); the add) |
| `ebm_compose` | 3 | 0.1759 / 0.1875 / 0.2269 | 0.1968 ± 0.0218 | 0.5500 | 72.0 | 14226660 (the independent per-factor energy expert) |
| `knn_proto` | 3 | 0.8588 / 0.8542 / 0.8819 | 0.8650 ± 0.0122 | 0.9165 | 56.0 | 475136 (the per-factor prototype table (R8's dec) |
| `kstar` | 3 | 0.5718 / 0.9977 / 0.6806 | 0.7500 ± 0.1807 | 0.8174 | 55.3 | 2145738 (four-part, never netted; 4.0927x ref) |
| `kstar/landscape_removed` | 3 | 0.0556 / 0.0556 / 0.2778 | 0.1296 ± 0.1048 | 0.5000 | 1.0 | 2145738 (four-part, never netted; 4.0927x ref) |
| `kstar/landscape_shuffled` | 3 | 0.1481 / 0.1389 / 0.0602 | 0.1157 ± 0.0395 | 0.4941 | 20.3 | 2145738 (four-part, never netted; 4.0927x ref) |
| `kstar/landscape_shuffled_across_wells` | 3 | 0.1065 / 0.0764 / 0.1435 | 0.1088 ± 0.0275 | 0.5417 | 13.3 | 2145738 (four-part, never netted; 4.0927x ref) |
| `kstar/random_atoms` | 3 | 0.1458 / 0.2523 / 0.1597 | 0.1860 ± 0.0473 | 0.4657 | 24.7 | 2145738 (four-part, never netted; 4.0927x ref) |
| `lace` | 3 | 0.2940 / 0.3241 / 0.2917 | 0.3032 ± 0.0148 | 0.6280 | 71.7 | None (the LACE attribute classifier's paramete) |
| `lace/ld_corrected` | 3 | 0.2662 / 0.3009 / 0.2824 | 0.2832 ± 0.0142 | 0.6581 | 71.7 | None () |
| `lace/ld_published` | 3 | 0.2685 / 0.2639 / 0.2407 | 0.2577 ± 0.0122 | 0.5740 | 55.3 | None () |
| `store_removed` | 3 | 0.1111 / 0.1389 / 0.1111 | 0.1204 ± 0.0131 | 0.5000 | 1.0 | 524288 (the store's modes (n_modes·D·4); the add) |

### Pre-declared bars (PREREG §1b)

```
{
 "S1_controls_collapse": {
  "delta_kstar_minus_control_per_seed": {
   "landscape_removed": [
    0.5162037037037036,
    0.9421296296296295,
    0.40277777777777796
   ],
   "landscape_shuffled": [
    0.42361111111111105,
    0.8587962962962963,
    0.6203703703703705
   ],
   "landscape_shuffled_across_wells": [
    0.46527777777777773,
    0.9212962962962963,
    0.5370370370370372
   ],
   "random_atoms": [
    0.4259259259259258,
    0.7453703703703702,
    0.5208333333333335
   ]
  },
  "PASS": true
 },
 "S2_vs_clu": {
  "delta_per_seed": [
   -0.0185185185185186,
   0.3912037037037036,
   0.09259259259259267
  ],
  "mean_delta": 0.15509259259259256,
  "margin": 0.02,
  "rule": "every",
  "verdict": "LOSES / MIXED"
 },
 "S3_vs_knn_proto": {
  "delta_per_seed": [
   -0.2870370370370371,
   0.1435185185185185,
   -0.20138888888888862
  ],
  "mean_delta": -0.11496913580246908,
  "margin": 0.02,
  "rule": "mean",
  "verdict": "LOSE"
 },
 "S4_vs_lace": {
  "delta_per_seed": [
   0.27777777777777773,
   0.6736111111111112,
   0.388888888888889
  ],
  "mean_delta": 0.4467592592592593,
  "margin": 0.02,
  "rule": "mean",
  "verdict": "WIN"
 },
 "S4_vs_ebm_compose": {
  "delta_per_seed": [
   0.39583333333333326,
   0.8101851851851851,
   0.45370370370370383
  ],
  "mean_delta": 0.5532407407407408,
  "margin": 0.02,
  "rule": "mean",
  "verdict": "WIN"
 }
}
```

## PART B — Shapes rows 3–4 (60 pair + 49 n-way corners)

`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.

| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |
|---|---|---|---|---|---|---|
| `kstar_store` | 3 | 47.495 | 47.404 | 49.119 | 48.006 |  |
| `kstar_store__landscape_removed` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled_across_wells` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__random_atoms` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `mlp@524288ref` | 3 | 5.193 | 3.239 | 8.174 | 5.535 |  |
| `lstm@524288ref` | 3 | 4.761 | 4.633 | 9.404 | 6.266 |  |
| `transformer@524288ref` | 3 | 1.229 | 0.651 | 3.073 | 1.651 |  |
| `cont_mlp@524288ref` | 3 | 0.028 | 0.028 | 0.055 | 0.037 |  |
| `table_FLOOR` | 3 | 8.229 | 5.073 | 9.752 | 7.685 |  |
| `LACE` | 3 | 0.165 | 0.073 | 0.229 | 0.156 | 4,073,663 clf params = 31.08x ref (63.095x with whitening); fit 32,256 rows, 50 ep, 36 trials, ode |
| `PoE` | 3 | 0.009 | 0.000 | 0.000 | 0.003 | 3,556,665 expert params = 27.135x ref; fit 32,256 rows, 20 ep, 36 trials, SGLD |

Verdict (PREREG §2a):

- **LACE**: CLU ahead — per-seed paired Δ (rival − K\*) [-47.33, -47.33, -48.89]
- **PoE**: CLU ahead — per-seed paired Δ (rival − K\*) [-47.486, -47.404, -49.119]

## PART B — ladder rung k=2

`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.

| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |
|---|---|---|---|---|---|---|
| `kstar_store` | 3 | 31.800 | 32.500 | 33.633 | 32.644 |  |
| `kstar_store__landscape_removed` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__random_atoms` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `learned_1x@kstar_total` | 3 | 25.133 | 29.383 | 30.350 | 28.289 |  |
| `mlp@524288ref` | 3 | 17.017 | 22.000 | 23.333 | 20.783 |  |
| `lstm@524288ref` | 3 | 19.417 | 19.333 | 23.750 | 20.833 |  |
| `transformer@524288ref` | 3 | 12.750 | 15.783 | 24.250 | 17.594 |  |
| `cont_mlp@524288ref` | 3 | 3.883 | 4.567 | 6.533 | 4.994 |  |
| `table_FLOOR` | 3 | 5.833 | 3.483 | 6.183 | 5.167 |  |
| `LACE` | 3 | 0.067 | 0.050 | 0.067 | 0.061 | 4,073,663 clf params = 31.08x ref (63.095x with whitening); fit 32,256 rows, 50 ep, 36 trials, ode |
| `PoE` | 3 | 0.017 | 0.000 | 0.000 | 0.006 | 3,556,665 expert params = 27.135x ref; fit 32,256 rows, 20 ep, 36 trials, SGLD |

Verdict (PREREG §2a):

- **LACE**: CLU ahead — per-seed paired Δ (rival − K\*) [-31.733, -32.45, -33.567]
- **PoE**: CLU ahead — per-seed paired Δ (rival − K\*) [-31.783, -32.5, -33.633]

## PART B — ladder rung k=3

`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.

| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |
|---|---|---|---|---|---|---|
| `kstar_store` | 3 | 60.150 | 62.783 | 61.908 | 61.614 |  |
| `kstar_store__landscape_removed` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled` | 3 | 0.008 | 0.008 | 0.000 | 0.006 |  |
| `kstar_store__random_atoms` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `learned_1x@kstar_total` | 3 | 42.275 | 49.975 | 50.375 | 47.542 |  |
| `mlp@524288ref` | 3 | 35.208 | 41.567 | 49.750 | 42.175 |  |
| `lstm@524288ref` | 3 | 44.433 | 39.400 | 46.675 | 43.503 |  |
| `transformer@524288ref` | 3 | 28.800 | 33.900 | 51.992 | 38.231 |  |
| `cont_mlp@524288ref` | 3 | 6.333 | 9.117 | 12.667 | 9.372 |  |
| `table_FLOOR` | 3 | 8.433 | 7.917 | 12.233 | 9.528 |  |
| `LACE` | 3 | 0.133 | 0.125 | 0.208 | 0.156 | 4,073,663 clf params = 31.08x ref (63.095x with whitening); fit 32,256 rows, 50 ep, 36 trials, ode |
| `PoE` | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 3,556,665 expert params = 27.135x ref; fit 32,256 rows, 20 ep, 36 trials, SGLD |

Verdict (PREREG §2a):

- **LACE**: CLU ahead — per-seed paired Δ (rival − K\*) [-60.017, -62.658, -61.7]
- **PoE**: CLU ahead — per-seed paired Δ (rival − K\*) [-60.15, -62.783, -61.908]

## PART B — ladder rung k=4

`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.

| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |
|---|---|---|---|---|---|---|
| `kstar_store` | 3 | 65.242 | 65.817 | 67.200 | 66.086 |  |
| `kstar_store__landscape_removed` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__random_atoms` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `learned_1x@kstar_total` | 3 | 52.542 | 57.075 | 58.575 | 56.064 |  |
| `mlp@524288ref` | 3 | 35.208 | 41.567 | 49.750 | 42.175 |  |
| `lstm@524288ref` | 3 | 44.433 | 39.400 | 46.675 | 43.503 |  |
| `transformer@524288ref` | 3 | 28.800 | 33.900 | 51.992 | 38.231 |  |
| `cont_mlp@524288ref` | 3 | 6.333 | 9.117 | 12.667 | 9.372 |  |
| `table_FLOOR` | 3 | 8.433 | 7.917 | 12.233 | 9.528 |  |
| `LACE` | 3 | 0.117 | 0.217 | 0.183 | 0.172 | 4,073,663 clf params = 31.08x ref (63.095x with whitening); fit 32,256 rows, 50 ep, 36 trials, ode |
| `PoE` | 3 | 0.008 | 0.000 | 0.008 | 0.006 | 3,556,665 expert params = 27.135x ref; fit 32,256 rows, 20 ep, 36 trials, SGLD |

Verdict (PREREG §2a):

- **LACE**: CLU ahead — per-seed paired Δ (rival − K\*) [-65.125, -65.6, -67.017]
- **PoE**: CLU ahead — per-seed paired Δ (rival − K\*) [-65.233, -65.817, -67.192]

## PART B — ladder rung k=5

`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.

| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |
|---|---|---|---|---|---|---|
| `kstar_store` | 3 | 65.642 | 66.817 | 67.283 | 66.581 |  |
| `kstar_store__landscape_removed` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__random_atoms` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `learned_1x@kstar_total` | 3 | 55.292 | 60.300 | 59.083 | 58.225 |  |
| `mlp@524288ref` | 3 | 35.208 | 41.567 | 49.750 | 42.175 |  |
| `lstm@524288ref` | 3 | 44.433 | 39.400 | 46.675 | 43.503 |  |
| `transformer@524288ref` | 3 | 28.800 | 33.900 | 51.992 | 38.231 |  |
| `cont_mlp@524288ref` | 3 | 6.333 | 9.117 | 12.667 | 9.372 |  |
| `table_FLOOR` | 3 | 8.433 | 7.917 | 12.233 | 9.528 |  |
| `LACE` | 3 | 0.083 | 0.142 | 0.183 | 0.136 | 4,073,663 clf params = 31.08x ref (63.095x with whitening); fit 32,256 rows, 50 ep, 36 trials, ode |
| `PoE` | 3 | 0.000 | 0.008 | 0.000 | 0.003 | 3,556,665 expert params = 27.135x ref; fit 32,256 rows, 20 ep, 36 trials, SGLD |

Verdict (PREREG §2a):

- **LACE**: CLU ahead — per-seed paired Δ (rival − K\*) [-65.558, -66.675, -67.1]
- **PoE**: CLU ahead — per-seed paired Δ (rival − K\*) [-65.642, -66.808, -67.283]

## PART B — ladder rung k=6

`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.

| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |
|---|---|---|---|---|---|---|
| `kstar_store` | 3 | 65.908 | 66.250 | 67.650 | 66.603 |  |
| `kstar_store__landscape_removed` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__landscape_shuffled` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `kstar_store__random_atoms` | 3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `learned_1x@kstar_total` | 3 | 56.533 | 60.142 | 59.200 | 58.625 |  |
| `mlp@524288ref` | 3 | 35.208 | 41.567 | 49.750 | 42.175 |  |
| `lstm@524288ref` | 3 | 44.433 | 39.400 | 46.675 | 43.503 |  |
| `transformer@524288ref` | 3 | 28.800 | 33.900 | 51.992 | 38.231 |  |
| `cont_mlp@524288ref` | 3 | 6.333 | 9.117 | 12.667 | 9.372 |  |
| `table_FLOOR` | 3 | 8.433 | 7.917 | 12.233 | 9.528 |  |
| `LACE` | 3 | 0.125 | 0.100 | 0.175 | 0.133 | 4,073,663 clf params = 31.08x ref (63.095x with whitening); fit 32,256 rows, 50 ep, 36 trials, ode |
| `PoE` | 3 | 0.000 | 0.008 | 0.000 | 0.003 | 3,556,665 expert params = 27.135x ref; fit 32,256 rows, 20 ep, 36 trials, SGLD |

Verdict (PREREG §2a):

- **LACE**: CLU ahead — per-seed paired Δ (rival − K\*) [-65.783, -66.15, -67.475]
- **PoE**: CLU ahead — per-seed paired Δ (rival − K\*) [-65.908, -66.242, -67.65]
