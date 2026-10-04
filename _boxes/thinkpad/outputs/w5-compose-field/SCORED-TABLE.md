## Rows 3 and 4 — every leg, every seed

**Labels are used ONCE at fit time to align the wells and at the address book, and nowhere else.** The store's byte ledger and the address book's are printed separately and are **never netted together**. Metric = distinct valid novel tuples per corner; row 4 = 49 n-way corners (per-corner ceiling 72), row 3 = 60 pair sub-corners (ceiling 36) — ⚠ the two rows are not on one scale. Cache **generation 2**.

| leg | class | store B | address B | params | r4 test_seed0 | r4 test_seed1 | r4 test_seed2 | r3 test_seed0 | r3 test_seed1 | r3 test_seed2 |
|---|---|---|---|---|---|---|---|---|---|---|
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
| `pca_lookup_argmax` | FLOOR (charter C.1/C.4, matched-bytes lookup) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_lookup_convex` | FLOOR (charter C.1, the mixture) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_landscape_removed` | CONTROL (charter C.2, atoms zeroed) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_landscape_shuffled` | CONTROL (charter C.2, atom values permuted) | 524288 | 189344 | 131072 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca_random_refit` | CONTROL (random atoms, address book REFIT) | 524288 | 189344 | 131072 | 0.939 | 0.163 | 1.224 | 0.233 | 0.083 | 0.933 |
| `oracle_content_pool` | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | 524288 | 0 | 131072 | 64.449 | 63.878 | 68.408 | 30.717 | 32.617 | 32.950 |
| `oracle_span_field` | NOT AN ARM (ceiling, FIELD store) | 524288 | 0 | 131072 | 40.939 | 50.429 | 54.204 | 17.350 | 24.400 | 22.400 |
| `oracle_span` | NOT AN ARM (ceiling) | 524288 | 1766400 | — | 44.184 | 50.898 | 53.000 | 18.933 | 21.483 | 24.700 |
| `oracle_content_items` | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | 524288 | 0 | 131072 | 32.388 | 39.122 | 41.776 | 18.750 | 18.233 | 20.467 |
| `oracle_content_fine` | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | 524288 | 0 | 131072 | 69.082 | 68.163 | 70.735 | 33.467 | 34.367 | 34.967 |
| `oracle_true_feature` | NOT AN ARM (ceiling of EVERY store, any budget) | — | 0 | — | 69.082 | 68.163 | 70.735 | 33.467 | 34.367 | 34.967 |

## The paired estimands — the only thing that decides a row

Primary = `pca_tensor`, margin **+1.0** novel tuple/corner, declared in the PREREG before the first held-out number. `beats` requires the mean paired delta to clear the margin AND the bootstrap 2.5 % quantile to clear it too.

| comparison | class of comparator | delta test_seed0 | delta test_seed1 | delta test_seed2 | ci95_low test_seed0 | ci95_low test_seed1 | ci95_low test_seed2 | beats test_seed0 | beats test_seed1 | beats test_seed2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `r3_comp_2way:pca_tensor_vs_oracle_content_fine` | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | -3.750 | -2.750 | -3.517 | -5.317 | -4.133 | -4.967 | False | False | False |
| `r3_comp_2way:pca_tensor_vs_oracle_content_items` | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | 10.967 | 13.383 | 10.983 | 7.967 | 10.733 | 8.500 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_oracle_content_pool` | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | -1.000 | -1.000 | -1.500 | -2.567 | -2.250 | -2.650 | False | False | False |
| `r3_comp_2way:pca_tensor_vs_oracle_span` | NOT AN ARM (ceiling) | 10.783 | 10.133 | 6.750 | 8.300 | 7.950 | 4.767 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_oracle_span_field` | NOT AN ARM (ceiling, FIELD store) | 12.367 | 7.217 | 9.050 | 10.200 | 5.250 | 7.067 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_oracle_true_feature` | NOT AN ARM (ceiling of EVERY store, any budget) | -3.750 | -2.750 | -3.517 | -5.317 | -4.133 | -4.967 | False | False | False |
| `r3_comp_2way:pca_tensor_vs_pca_landscape_removed` | CONTROL (charter C.2, atoms zeroed) | 29.717 | 31.617 | 31.450 | 28.083 | 30.233 | 29.983 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_pca_landscape_shuffled` | CONTROL (charter C.2, atom values permuted) | 29.717 | 31.617 | 31.450 | 28.083 | 30.233 | 29.983 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_pca_lookup_argmax` | FLOOR (charter C.1/C.4, matched-bytes lookup) | 29.717 | 31.617 | 31.450 | 28.083 | 30.233 | 29.983 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_pca_lookup_convex` | FLOOR (charter C.1, the mixture) | 29.717 | 31.617 | 31.450 | 28.083 | 30.233 | 29.983 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_pca_random_refit` | CONTROL (random atoms, address book REFIT) | 29.483 | 31.533 | 30.517 | 27.867 | 30.167 | 28.950 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_pca_tensor_sum` | ABLATION (same rank, address summed not multiplied) | 23.450 | 26.100 | 23.850 | 21.150 | 23.817 | 21.383 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_rival_cont_mlp` | RIVAL | 25.833 | 27.050 | 24.917 | 24.033 | 25.233 | 23.017 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_rival_lstm` | RIVAL | 10.300 | 12.283 | 7.700 | 7.683 | 9.967 | 5.983 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_rival_mlp` | RIVAL | 12.700 | 9.617 | 8.117 | 10.433 | 7.333 | 6.067 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_rival_mlp_180k` | RIVAL | 4.033 | 2.483 | 2.417 | 2.117 | 1.033 | 1.200 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_rival_table` | RIVAL | 23.883 | 28.133 | 25.267 | 21.600 | 26.117 | 22.883 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_rival_transformer` | RIVAL | 16.967 | 15.833 | 7.200 | 14.883 | 13.733 | 5.417 | True | True | True |
| `r3_comp_2way:pca_tensor_vs_span_head` | ABLATION (the deployed read) | 15.067 | 20.567 | 17.567 | 12.350 | 18.250 | 15.200 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_oracle_content_fine` | NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K) | -5.469 | -3.837 | -3.082 | -7.224 | -6.163 | -4.837 | False | False | False |
| `r4_comp_nway:pca_tensor_vs_oracle_content_items` | NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B) | 31.224 | 25.204 | 25.878 | 24.612 | 20.184 | 20.490 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_oracle_content_pool` | NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B) | -0.837 | 0.449 | -0.755 | -3.102 | -1.490 | -2.510 | False | False | False |
| `r4_comp_nway:pca_tensor_vs_oracle_span` | NOT AN ARM (ceiling) | 19.429 | 13.429 | 14.653 | 14.204 | 9.878 | 10.837 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_oracle_span_field` | NOT AN ARM (ceiling, FIELD store) | 22.673 | 13.898 | 13.449 | 18.959 | 9.633 | 8.571 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_oracle_true_feature` | NOT AN ARM (ceiling of EVERY store, any budget) | -5.469 | -3.837 | -3.082 | -7.224 | -6.163 | -4.837 | False | False | False |
| `r4_comp_nway:pca_tensor_vs_pca_landscape_removed` | CONTROL (charter C.2, atoms zeroed) | 63.612 | 64.327 | 67.653 | 61.469 | 61.612 | 65.776 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_pca_landscape_shuffled` | CONTROL (charter C.2, atom values permuted) | 63.612 | 64.327 | 67.653 | 61.469 | 61.612 | 65.776 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_pca_lookup_argmax` | FLOOR (charter C.1/C.4, matched-bytes lookup) | 63.612 | 64.327 | 67.653 | 61.469 | 61.612 | 65.776 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_pca_lookup_convex` | FLOOR (charter C.1, the mixture) | 63.612 | 64.327 | 67.653 | 61.469 | 61.612 | 65.776 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_pca_random_refit` | CONTROL (random atoms, address book REFIT) | 62.673 | 64.163 | 66.429 | 60.592 | 61.429 | 64.531 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_pca_tensor_sum` | ABLATION (same rank, address summed not multiplied) | 50.918 | 54.653 | 47.816 | 47.183 | 50.612 | 42.163 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_rival_cont_mlp` | RIVAL | 55.000 | 56.143 | 49.327 | 52.286 | 52.980 | 45.857 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_rival_lstm` | RIVAL | 20.592 | 23.673 | 16.143 | 15.510 | 19.143 | 11.959 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_rival_mlp` | RIVAL | 26.367 | 27.694 | 14.898 | 21.224 | 23.306 | 10.224 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_rival_mlp_180k` | RIVAL | 10.143 | 8.673 | 7.796 | 6.632 | 5.816 | 5.082 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_rival_table` | RIVAL | 52.449 | 57.306 | 53.531 | 48.571 | 53.857 | 48.041 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_rival_transformer` | RIVAL | 33.143 | 28.082 | 13.898 | 29.143 | 23.959 | 10.429 | True | True | True |
| `r4_comp_nway:pca_tensor_vs_span_head` | ABLATION (the deployed read) | 23.918 | 19.327 | 21.612 | 18.020 | 15.041 | 16.755 | True | True | True |
