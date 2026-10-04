## Rows 3 and 4 — every leg, every seed  (scored 1/2)

**Labels are used ONCE at fit time to align the wells and at the address book, and nowhere else. The CONTENT is unsupervised, and so is the coupling.** The store's byte ledger, the address book's and the coupling's are printed separately and are **never netted together**. Metric = distinct valid novel tuples per corner; row 4 = 49 n-way corners (per-corner ceiling 72), row 3 = 60 pair sub-corners (ceiling 36) — ⚠ the two rows are not on one scale. Cache **generation 2**.

⭐ `pca_tensor`, every rival and every ceiling are wave 5's own scored objects, copied bit-for-bit, **not re-run** — so this table is continuous with the one wave 5 published.

| leg | class | store B | address B | params | r4 test_seed0 | r4 test_seed1 | r4 test_seed2 | r3 test_seed0 | r3 test_seed1 | r3 test_seed2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `pca_settle` | ARM (the PHYSICS read: settled state, cross-well relaxation) | 524288 | 189344 | 131072 | 23.653 | 17.408 | 28.592 | 13.917 | 6.083 | 11.100 |
| `pca_settle_onestep` | ABLATION (ONE settle step; identical beta, lambda, W) | 524288 | 189344 | 131072 | 60.878 | 60.531 | 64.694 | 28.333 | 30.000 | 30.467 |
| `pca_settle_f05_conv` | ABLATION (the gentlest dynamics: lambda = 0.05 x certified-safe, converged) | 524288 | 189344 | 131072 | 63.082 | 63.816 | 67.265 | 29.683 | 31.433 | 31.417 |
| `pca_settle_zerostep` | ABLATION (ZERO settle steps; asserted identical to pca_tensor) | 524288 | 189344 | 131072 | 63.612 | 64.327 | 67.653 | 29.717 | 31.617 | 31.450 |
| `pca_tensor` | ARM | 524288 | 189344 | 131072 | 63.612 | 64.327 | 67.653 | 29.717 | 31.617 | 31.450 |
| `rival_mlp_180k` | RIVAL | 0 | 0 | 178952 | 53.469 | 55.653 | 59.857 | 25.683 | 29.133 | 29.033 |
| `rival_table` | RIVAL | 475136 | 0 | 118784 | 11.163 | 7.020 | 14.122 | 5.833 | 3.483 | 6.183 |
| `rival_mlp` | RIVAL | 0 | 0 | 117878 | 37.245 | 36.633 | 52.755 | 17.017 | 22.000 | 23.333 |
| `rival_cont_mlp` | RIVAL | 0 | 0 | 119525 | 8.612 | 8.184 | 18.327 | 3.883 | 4.567 | 6.533 |
| `rival_transformer` | RIVAL | 0 | 0 | 119858 | 30.469 | 36.245 | 53.755 | 12.750 | 15.783 | 24.250 |
| `rival_lstm` | RIVAL | 0 | 0 | 119125 | 43.020 | 40.653 | 51.510 | 19.417 | 19.333 | 23.750 |
| `rival_lace` | **DECLARED NOT-RUN** | — | — | — | — | — | — | — | — | — | RivalRefusal: 'lace': LACE's output is a SAMPLER DRAW, not a function of the request alone: two calls on the same request return different features by construction. It is therefore not offered as a request->features callable. Its categorical head also has a closed vocabulary over the realised grid, so it cannot express an off-grid value at all - a structural limit of the method, documented at chlu.core.gen_request.DenseRequests, not one this spoke imposed. Hubs needing LACE on their own corners should ask this spoke to score those corners.
| `rival_poe_ebm` | **DECLARED NOT-RUN** | — | — | — | — | — | — | — | — | — | RivalRefusal: 'poe_ebm': The product-of-experts EBM's output is an SGLD draw from a noise initialisation, not a function of the request alone, and its experts are conditioned on a FULL factor assignment. It is not offered as a request->features callable. Hubs needing it on their own corners should ask this spoke to score those corners.
| `span_head` | ABLATION (the deployed read) | 524288 | 1766400 | — | 39.694 | 45.000 | 46.041 | 14.650 | 11.050 | 13.883 |
| `pca_tensor_sum` | ABLATION (same rank, address summed not multiplied) | 524288 | 189344 | 131072 | 12.694 | 9.673 | 19.837 | 6.267 | 5.517 | 7.600 |
| `pca_settle_lookup` | FLOOR (charter C.1/C.4, nearest atom to the settled state) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_lookup_argmax` | FLOOR (charter C.1/C.4, matched-bytes lookup) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_lookup_convex` | FLOOR (charter C.1, the mixture) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_settle_removed` | CONTROL (charter C.2, atoms zeroed) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_settle_shuffled` | CONTROL (charter C.2, atom values permuted) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_landscape_removed` | CONTROL (charter C.2, atoms zeroed) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_landscape_shuffled` | CONTROL (charter C.2, atom values permuted) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_random_refit` | CONTROL (random atoms, address book REFIT) | 524288 | 189344 | 131072 | 0.939 | 0.163 | 1.224 | 0.233 | 0.083 | 0.933 |
| `oracle_content_pool` | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | 524288 | 0 | 131072 | 64.449 | 63.878 | 68.408 | 30.717 | 32.617 | 32.950 |
| `oracle_span_field` | NOT AN ARM (ceiling, FIELD store) | 524288 | 0 | 131072 | 40.939 | 50.429 | 54.204 | 17.350 | 24.400 | 22.400 |
| `oracle_span` | NOT AN ARM (ceiling) | 524288 | 1766400 | — | 44.184 | 50.898 | 53.000 | 18.933 | 21.483 | 24.700 |
| `oracle_content_items` | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | 524288 | 0 | 131072 | 32.388 | 39.122 | 41.776 | 18.750 | 18.233 | 20.467 |
| `oracle_content_fine` | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | 524288 | 0 | 131072 | 69.082 | 68.163 | 70.735 | 33.467 | 34.367 | 34.967 |
| `oracle_true_feature` | NOT AN ARM (ceiling of EVERY store, any budget) | — | 0 | — | 69.082 | 68.163 | 70.735 | 33.467 | 34.367 | 34.967 |

## The PREREG's bars — evaluated, with both ends of the interval

Primary = `pca_settle`, margin **±1.0** novel tuple/corner, declared in `PREREG-w6-compose-physics-1.md` before the first held-out number.

`beats` requires the mean paired delta to clear **+1.0** AND the bootstrap 2.5 % quantile to clear it too, on **all three seeds**.

| bar | comparator | row | delta test_seed0 | delta test_seed1 | delta test_seed2 | 95% CI test_seed0 | 95% CI test_seed1 | 95% CI test_seed2 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B1 | `pca_settle_zerostep` | row 4 | -39.959 | -46.918 | -39.061 | [-44.449, -35.184] | [-52.490, -41.020] | [-44.531, -33.510] | **NOT met — NEGATIVE** |
| B1 | `pca_settle_zerostep` | row 3 | -15.800 | -25.533 | -20.350 | [-18.483, -13.017] | [-27.583, -23.283] | [-22.633, -17.933] | **NOT met — NEGATIVE** |
| B2 | `pca_tensor` | row 4 | -39.959 | -46.918 | -39.061 | [-44.449, -35.184] | [-52.490, -41.020] | [-44.531, -33.510] | **NOT met — NEGATIVE** |
| B2 | `pca_tensor` | row 3 | -15.800 | -25.533 | -20.350 | [-18.483, -13.017] | [-27.583, -23.283] | [-22.633, -17.933] | **NOT met — NEGATIVE** |
| B3 | `pca_settle_onestep` | row 4 | -37.224 | -43.122 | -36.102 | [-41.327, -32.999] | [-48.347, -37.571] | [-41.224, -30.959] | **NOT met — NEGATIVE** |
| B3 | `pca_settle_onestep` | row 3 | -14.417 | -23.917 | -19.367 | [-16.750, -12.083] | [-25.933, -21.767] | [-21.583, -17.050] | **NOT met — NEGATIVE** |
| C1 | `pca_settle_lookup` | row 4 | 23.653 | 17.408 | 28.592 | [19.265, 28.184] | [12.490, 22.694] | [22.938, 34.367] | **MET** |
| C1 | `pca_settle_lookup` | row 3 | 13.917 | 6.083 | 11.100 | [11.450, 16.400] | [4.383, 7.983] | [8.733, 13.683] | **MET** |

### The TIE TEST

⭐ A tie is what decides **decoration**, so it is computed from BOTH ends of the interval: |delta| < 1.0 **and** the whole 95 % interval inside ±1.0, on both rows and all three seeds.

| comparator | row | delta test_seed0 | delta test_seed1 | delta test_seed2 | 95% CI test_seed0 | 95% CI test_seed1 | 95% CI test_seed2 | tie? |
|---|---|---|---|---|---|---|---|---|
| `pca_settle_zerostep` | row 4 | -39.959 | -46.918 | -39.061 | [-44.449, -35.184] | [-52.490, -41.020] | [-44.531, -33.510] | **NOT a tie** |
| `pca_settle_zerostep` | row 3 | -15.800 | -25.533 | -20.350 | [-18.483, -13.017] | [-27.583, -23.283] | [-22.633, -17.933] | **NOT a tie** |
| `pca_tensor` | row 4 | -39.959 | -46.918 | -39.061 | [-44.449, -35.184] | [-52.490, -41.020] | [-44.531, -33.510] | **NOT a tie** |
| `pca_tensor` | row 3 | -15.800 | -25.533 | -20.350 | [-18.483, -13.017] | [-27.583, -23.283] | [-22.633, -17.933] | **NOT a tie** |
| `pca_settle_onestep` | row 4 | -37.224 | -43.122 | -36.102 | [-41.327, -32.999] | [-48.347, -37.571] | [-41.224, -30.959] | **NOT a tie** |
| `pca_settle_onestep` | row 3 | -14.417 | -23.917 | -19.367 | [-16.750, -12.083] | [-25.933, -21.767] | [-21.583, -17.050] | **NOT a tie** |

### The ladder's own tie tests against the incumbent

⭐ This is the row that decides **decoration**: does a rung where the dynamics barely move the state tie `pca_tensor`?

| rung | vs | row | delta test_seed0 | delta test_seed1 | delta test_seed2 | 95% CI test_seed0 | 95% CI test_seed1 | 95% CI test_seed2 | tie? |
|---|---|---|---|---|---|---|---|---|---|
| `pca_settle_f05_conv` | `pca_tensor` | row 4 | -0.531 | -0.510 | -0.388 | [-0.878, -0.204] | [-1.000, -0.082] | [-0.796, -0.041] | **NOT a tie** |
| `pca_settle_f05_conv` | `pca_tensor` | row 3 | -0.033 | -0.183 | -0.033 | [-0.267, 0.200] | [-0.417, 0.033] | [-0.183, 0.117] | **TIE** |
| `pca_settle_onestep` | `pca_tensor` | row 4 | -2.735 | -3.796 | -2.959 | [-4.102, -1.388] | [-5.654, -2.122] | [-4.776, -1.347] | **NOT a tie** |
| `pca_settle_onestep` | `pca_tensor` | row 3 | -1.383 | -1.617 | -0.983 | [-2.584, -0.250] | [-2.600, -0.650] | [-1.783, -0.233] | **NOT a tie** |

### The read-side byte ledger, per leg (never netted into the store)

| leg | store B | address B | coupling B | standardisation B | read-side added vs the incumbent |
|---|---|---|---|---|---|
| `pca_tensor` | 524288 | 189344 | — | — | 0 |
| `pca_settle_zerostep` | 524288 | 189344 | 78848 | 2464 | 81312 |
| `pca_settle_f05_conv` | 524288 | 189344 | 78848 | 2464 | 81312 |
| `pca_settle_onestep` | 524288 | 189344 | 78848 | 2464 | 81312 |
| `pca_settle` | 524288 | 189344 | 78848 | 2464 | 81312 |

⚠ Every settle leg pays the SAME added read-side bytes, so the ladder B1/B3 is parameter-identical and byte-identical end to end. ⛔ The store is 524,288 B on every row, tolerance 0.


## Every paired estimand against the primary

| comparison | class of comparator | delta test_seed0 | delta test_seed1 | delta test_seed2 | ci95_low test_seed0 | ci95_low test_seed1 | ci95_low test_seed2 | beats test_seed0 | beats test_seed1 | beats test_seed2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `r3_comp_2way:pca_settle_vs_oracle_content_fine` | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | -19.550 | -28.283 | -23.867 | -21.900 | -29.933 | -26.150 | False | False | False |
| `r3_comp_2way:pca_settle_vs_oracle_content_items` | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | -4.833 | -12.150 | -9.367 | -7.933 | -14.867 | -12.583 | False | False | False |
| `r3_comp_2way:pca_settle_vs_oracle_content_pool` | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | -16.800 | -26.533 | -21.850 | -19.150 | -28.333 | -24.100 | False | False | False |
| `r3_comp_2way:pca_settle_vs_oracle_span` | NOT AN ARM (ceiling) | -5.017 | -15.400 | -13.600 | -8.250 | -17.967 | -16.383 | False | False | False |
| `r3_comp_2way:pca_settle_vs_oracle_span_field` | NOT AN ARM (ceiling, FIELD store) | -3.433 | -18.317 | -11.300 | -6.367 | -20.833 | -14.433 | False | False | False |
| `r3_comp_2way:pca_settle_vs_oracle_true_feature` | NOT AN ARM (ceiling of EVERY store, any budget) | -19.550 | -28.283 | -23.867 | -21.900 | -29.933 | -26.150 | False | False | False |
| `r3_comp_2way:pca_settle_vs_pca_landscape_removed` | CONTROL (charter C.2, atoms zeroed) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_landscape_shuffled` | CONTROL (charter C.2, atom values permuted) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_lookup_argmax` | FLOOR (charter C.1/C.4, matched-bytes lookup) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_lookup_convex` | FLOOR (charter C.1, the mixture) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_random_refit` | CONTROL (random atoms, address book REFIT) | 13.683 | 6.000 | 10.167 | 11.267 | 4.300 | 7.816 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_settle_f05_conv` | ABLATION (the gentlest dynamics: lambda = 0.05 x certified-safe, converged) | -15.767 | -25.350 | -20.317 | -18.400 | -27.400 | -22.600 | False | False | False |
| `r3_comp_2way:pca_settle_vs_pca_settle_lookup` | FLOOR (charter C.1/C.4, nearest atom to the settled state) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_settle_onestep` | ABLATION (ONE settle step; identical beta, lambda, W) | -14.417 | -23.917 | -19.367 | -16.750 | -25.933 | -21.583 | False | False | False |
| `r3_comp_2way:pca_settle_vs_pca_settle_removed` | CONTROL (charter C.2, atoms zeroed) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_settle_shuffled` | CONTROL (charter C.2, atom values permuted) | 13.917 | 6.083 | 11.100 | 11.450 | 4.383 | 8.733 | True | True | True |
| `r3_comp_2way:pca_settle_vs_pca_settle_zerostep` | ABLATION (ZERO settle steps; asserted identical to pca_tensor) | -15.800 | -25.533 | -20.350 | -18.483 | -27.583 | -22.633 | False | False | False |
| `r3_comp_2way:pca_settle_vs_pca_tensor` | ARM | -15.800 | -25.533 | -20.350 | -18.483 | -27.583 | -22.633 | False | False | False |
| `r3_comp_2way:pca_settle_vs_pca_tensor_sum` | ABLATION (same rank, address summed not multiplied) | 7.650 | 0.567 | 3.500 | 4.600 | -1.800 | 0.450 | True | False | False |
| `r3_comp_2way:pca_settle_vs_rival_cont_mlp` | RIVAL | 10.033 | 1.517 | 4.567 | 7.450 | -0.583 | 1.866 | True | False | True |
| `r3_comp_2way:pca_settle_vs_rival_lstm` | RIVAL | -5.500 | -13.250 | -12.650 | -9.483 | -16.150 | -15.217 | False | False | False |
| `r3_comp_2way:pca_settle_vs_rival_mlp` | RIVAL | -3.100 | -15.917 | -12.233 | -6.933 | -18.833 | -15.417 | False | False | False |
| `r3_comp_2way:pca_settle_vs_rival_mlp_180k` | RIVAL | -11.767 | -23.050 | -17.933 | -15.133 | -25.267 | -20.533 | False | False | False |
| `r3_comp_2way:pca_settle_vs_rival_table` | RIVAL | 8.083 | 2.600 | 4.917 | 4.767 | 0.350 | 1.733 | True | False | True |
| `r3_comp_2way:pca_settle_vs_rival_transformer` | RIVAL | 1.167 | -9.700 | -13.150 | -2.233 | -12.133 | -16.183 | False | False | False |
| `r3_comp_2way:pca_settle_vs_span_head` | ABLATION (the deployed read) | -0.733 | -4.967 | -2.783 | -3.900 | -7.700 | -5.467 | False | False | False |
| `r4_comp_nway:pca_settle_vs_oracle_content_fine` | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | -45.429 | -50.755 | -42.143 | -49.735 | -56.041 | -47.755 | False | False | False |
| `r4_comp_nway:pca_settle_vs_oracle_content_items` | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | -8.735 | -21.714 | -13.184 | -14.592 | -29.449 | -21.776 | False | False | False |
| `r4_comp_nway:pca_settle_vs_oracle_content_pool` | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | -40.796 | -46.469 | -39.816 | -45.449 | -52.000 | -45.551 | False | False | False |
| `r4_comp_nway:pca_settle_vs_oracle_span` | NOT AN ARM (ceiling) | -20.531 | -33.490 | -24.408 | -27.205 | -40.347 | -30.837 | False | False | False |
| `r4_comp_nway:pca_settle_vs_oracle_span_field` | NOT AN ARM (ceiling, FIELD store) | -17.286 | -33.020 | -25.612 | -23.347 | -39.592 | -33.103 | False | False | False |
| `r4_comp_nway:pca_settle_vs_oracle_true_feature` | NOT AN ARM (ceiling of EVERY store, any budget) | -45.429 | -50.755 | -42.143 | -49.735 | -56.041 | -47.755 | False | False | False |
| `r4_comp_nway:pca_settle_vs_pca_landscape_removed` | CONTROL (charter C.2, atoms zeroed) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_landscape_shuffled` | CONTROL (charter C.2, atom values permuted) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_lookup_argmax` | FLOOR (charter C.1/C.4, matched-bytes lookup) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_lookup_convex` | FLOOR (charter C.1, the mixture) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_random_refit` | CONTROL (random atoms, address book REFIT) | 22.714 | 17.245 | 27.367 | 18.244 | 12.347 | 21.776 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_settle_f05_conv` | ABLATION (the gentlest dynamics: lambda = 0.05 x certified-safe, converged) | -39.429 | -46.408 | -38.673 | -43.796 | -51.959 | -44.102 | False | False | False |
| `r4_comp_nway:pca_settle_vs_pca_settle_lookup` | FLOOR (charter C.1/C.4, nearest atom to the settled state) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_settle_onestep` | ABLATION (ONE settle step; identical beta, lambda, W) | -37.224 | -43.122 | -36.102 | -41.327 | -48.347 | -41.224 | False | False | False |
| `r4_comp_nway:pca_settle_vs_pca_settle_removed` | CONTROL (charter C.2, atoms zeroed) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_settle_shuffled` | CONTROL (charter C.2, atom values permuted) | 23.653 | 17.408 | 28.592 | 19.265 | 12.490 | 22.938 | True | True | True |
| `r4_comp_nway:pca_settle_vs_pca_settle_zerostep` | ABLATION (ZERO settle steps; asserted identical to pca_tensor) | -39.959 | -46.918 | -39.061 | -44.449 | -52.490 | -44.531 | False | False | False |
| `r4_comp_nway:pca_settle_vs_pca_tensor` | ARM | -39.959 | -46.918 | -39.061 | -44.449 | -52.490 | -44.531 | False | False | False |
| `r4_comp_nway:pca_settle_vs_pca_tensor_sum` | ABLATION (same rank, address summed not multiplied) | 10.959 | 7.735 | 8.755 | 5.326 | 1.939 | 0.244 | True | True | False |
| `r4_comp_nway:pca_settle_vs_rival_cont_mlp` | RIVAL | 15.041 | 9.224 | 10.265 | 10.633 | 3.816 | 2.918 | True | True | True |
| `r4_comp_nway:pca_settle_vs_rival_lstm` | RIVAL | -19.367 | -23.245 | -22.918 | -26.143 | -30.205 | -30.122 | False | False | False |
| `r4_comp_nway:pca_settle_vs_rival_mlp` | RIVAL | -13.592 | -19.224 | -24.163 | -20.408 | -26.367 | -31.592 | False | False | False |
| `r4_comp_nway:pca_settle_vs_rival_mlp_180k` | RIVAL | -29.816 | -38.245 | -31.265 | -36.224 | -44.551 | -37.204 | False | False | False |
| `r4_comp_nway:pca_settle_vs_rival_table` | RIVAL | 12.490 | 10.388 | 14.469 | 6.755 | 5.143 | 6.040 | True | True | True |
| `r4_comp_nway:pca_settle_vs_rival_transformer` | RIVAL | -6.816 | -18.837 | -25.163 | -12.307 | -24.898 | -32.327 | False | False | False |
| `r4_comp_nway:pca_settle_vs_span_head` | ABLATION (the deployed read) | -16.041 | -27.592 | -17.449 | -23.633 | -35.224 | -24.653 | False | False | False |
