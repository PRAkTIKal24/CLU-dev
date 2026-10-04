# w15 — T7 AT THREE SEEDS, AND THE DELETION ROWS

Rendered from `C:\chlu\outputs\w15-single-design-seeds\T7-3SEEDS.json` by `scripts/w15_findings_tables.py`. Built 2026-09-22T23:32:11Z.

**Condition TEST · K\* = 32 · n = 3 seeds [0, 1, 2] · scored runs spent by this hub: 0**

Notation: n = factors jointly composed; K = wells. Here K = 32 wells.

## T7 — rows 3 and 4 at K\* = 32 on TEST, **n = 3, seeds [0, 1, 2]**

Metric: `distinct_valid_novel_tuples` **per corner**; the corner is the unit. Row 3 = 60 tiled `pair-*` sub-corners, row 4 = 49 Montero `nway-*` corners.

| arm | role | row | seed 0 | seed 1 | seed 2 | mean ± sd | n seeds |
|---|---|---|---|---|---|---|---|
| `cov_tensor` | THE ARM | row 3 (pair) | 31.600 | 32.233 | 33.217 | 32.350 ± 0.815 | 3 |
| `cov_tensor` | THE ARM | row 4 (n-way) | 66.959 | 65.980 | 68.592 | 67.177 ± 1.320 | 3 |
| `landscape_removed` | CHARTER-C CONTROL -- must collapse | row 3 (pair) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `landscape_removed` | CHARTER-C CONTROL -- must collapse | row 4 (n-way) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `landscape_shuffled` | CHARTER-C CONTROL -- must collapse | row 3 (pair) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `landscape_shuffled` | CHARTER-C CONTROL -- must collapse | row 4 (n-way) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `landscape_shuffled_across_wells` | CHARTER-C CONTROL -- must collapse | row 3 (pair) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `landscape_shuffled_across_wells` | CHARTER-C CONTROL -- must collapse | row 4 (n-way) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `random_atoms` | CHARTER-C CONTROL -- must collapse | row 3 (pair) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `random_atoms` | CHARTER-C CONTROL -- must collapse | row 4 (n-way) | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 | 3 |
| `mlp__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 3 (pair) | 2.933 | 3.433 | 5.267 | 3.878 ± 1.229 | 3 |
| `mlp__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 4 (n-way) | 7.959 | 3.000 | 11.735 | 7.565 ± 4.381 | 3 |
| `lstm__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 3 (pair) | 2.500 | 2.700 | 4.917 | 3.372 ± 1.341 | 3 |
| `lstm__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 4 (n-way) | 7.531 | 7.000 | 14.898 | 9.810 ± 4.415 | 3 |
| `transformer__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 3 (pair) | 0.667 | 0.667 | 1.717 | 1.017 ± 0.606 | 3 |
| `transformer__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 4 (n-way) | 1.918 | 0.633 | 4.735 | 2.429 ± 2.098 | 3 |
| `cont_mlp__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 3 (pair) | 0.000 | 0.000 | 0.017 | 0.006 ± 0.010 | 3 |
| `cont_mlp__as_trained` | 524,288 B REFERENCE-column comparator; the 1x column is a DECLARED NOT-RUN | row 4 (n-way) | 0.061 | 0.061 | 0.102 | 0.075 ± 0.024 | 3 |
| `table` | PRINTED FLOOR, never a rival | row 3 (pair) | 5.833 | 3.483 | 6.183 | 5.167 ± 1.468 | 3 |
| `table` | PRINTED FLOOR, never a rival | row 4 (n-way) | 11.163 | 7.020 | 14.122 | 10.769 ± 3.567 | 3 |

## The charter-C control gate — measured at EVERY seed, never assumed

> NO STORE ROW IS CLAIMED unless landscape_removed, landscape_shuffled, landscape_shuffled_across_wells and random_atoms ALL collapse on EVERY seed. A control that collapsed at seed 0 is NOT assumed to collapse at seeds 1 and 2 -- it is measured at each.

| seed | arm (all 109 corners) | `landscape_removed` | `landscape_shuffled` | `landscape_shuffled_across_wells` | `random_atoms` | all four collapse? |
|---|---|---|---|---|---|---|
| 0 | 47.4954 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| 1 | 47.4037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| 2 | 49.1193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |

**ALL FOUR COLLAPSE ON EVERY SEED: **YES**** ⇒ store rows ARE claimable at n = 3

Paired estimand against each control, margin 1.0, per seed — the corner is the unit:

| seed | control | mean paired delta | CI95 | won / tied / lost | beats the 1.0 margin? |
|---|---|---|---|---|---|
| 0 | `landscape_removed` | 47.495 | [44.10, 51.03] | 109 / 0 / 0 | **YES** |
| 0 | `landscape_shuffled` | 47.495 | [44.10, 51.03] | 109 / 0 / 0 | **YES** |
| 0 | `landscape_shuffled_across_wells` | 47.495 | [44.10, 51.03] | 109 / 0 / 0 | **YES** |
| 0 | `random_atoms` | 47.495 | [44.10, 51.03] | 109 / 0 / 0 | **YES** |
| 1 | `landscape_removed` | 47.404 | [44.06, 50.79] | 109 / 0 / 0 | **YES** |
| 1 | `landscape_shuffled` | 47.404 | [44.06, 50.79] | 109 / 0 / 0 | **YES** |
| 1 | `landscape_shuffled_across_wells` | 47.404 | [44.06, 50.79] | 109 / 0 / 0 | **YES** |
| 1 | `random_atoms` | 47.404 | [44.06, 50.79] | 109 / 0 / 0 | **YES** |
| 2 | `landscape_removed` | 49.119 | [45.77, 52.53] | 109 / 0 / 0 | **YES** |
| 2 | `landscape_shuffled` | 49.119 | [45.77, 52.53] | 109 / 0 / 0 | **YES** |
| 2 | `landscape_shuffled_across_wells` | 49.119 | [45.77, 52.53] | 109 / 0 / 0 | **YES** |
| 2 | `random_atoms` | 49.119 | [45.77, 52.53] | 109 / 0 / 0 | **YES** |

## The four-part byte ledger, per seed — ⛔ NEVER NETTED

**Convention:** FOUR-PART TOTAL, SUMMED AND NEVER NETTED. part 1 atoms = sum_k A x d_k float32 (4 B/float); part 2 = the packed float64 second-moment triangle + first moment + count that make the deletion exact; part 3 = the fitted TensorAddress parameters; part 4 = the coupling / hole table that carries the refusal holes. `reference_pin_bytes = 524,288` is a REFERENCE COLUMN ONLY this wave -- it is NOT a budget and the object is 1x its OWN total. `times_reference_pin` is honest_total_bytes / 524,288. The MEMORY LEDGER IS PRINTED SEPARATELY AND IS NEVER MATCHED between arms.

| seed | 1. atoms | 2. covariance + count | 3. address book | 4. coupling / hole | **four-part total** | × the 524,288 B REFERENCE | n seeds |
|---|---|---|---|---|---|---|---|
| 0 | 516,096 | 560,608 | 1,060,864 | 6,610 | **2,144,178 B** | 4.0897× | 1 |
| 1 | 516,096 | 560,608 | 1,054,104 | 6,610 | **2,137,418 B** | 4.0768× | 1 |
| 2 | 516,096 | 560,608 | 1,058,264 | 6,610 | **2,141,578 B** | 4.0847× | 1 |

⛔ The memory ledger is printed here and is **never matched** between arms. The 524,288 B column is a REFERENCE only; the object is 1× its own total.

## The deletion rows — three scopes, one object, every cost line printed

**Exactness:** STORE-SCOPED: downdate(delete rows) equals the store fitted from scratch without those rows, at the same partition and the same rank. THE FROZEN ENCODER AND THE FROZEN DECODER ARE DECLARED OUTSIDE THE DELETABLE OBJECT AND ARE NOT REFIT.

**Adaptation:** post-deletion adaptation is LIGHT and DATA-FREE -- the covariance and count are downdated by subtraction, the atoms are re-eigendecomposed from the downdated statistic, the address book is carried across the basis change, and NO ROW OF DATA IS READ. It is never 'no changes'.

| scope | seed | rows deleted | survivor corners | novel/corner before → after | paired after − before | CI95 | store-scoped exactness max&nbsp;\|mode dev\| | ranks agree | OUR seconds | OUR bytes rewritten | OUR data rows read |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `value` | 0 | 11,044 | 89 | 50.921 → 50.921 | 0.0000 | [0.000, 0.000] | 5.960e-08 | **YES** | 0.8998 | 2,081,024 | **0** |
| `value` | 1 | 11,044 | 89 | 50.528 → 50.528 | 0.0000 | [0.000, 0.000] | 4.821e-08 | **YES** | 3.7503 | 2,074,368 | **0** |
| `value` | 2 | 11,044 | 89 | 52.607 → 52.607 | 0.0000 | [0.000, 0.000] | 2.011e-08 | **YES** | 1.3085 | 2,078,464 | **0** |
| `conjunction` | 0 | 0 | 109 | 47.495 → 47.495 | 0.0000 | [0.000, 0.000] | 0.000e+00 | **YES** | 0.0518 | 2,081,024 | **0** |
| `conjunction` | 1 | 0 | 109 | 47.404 → 47.404 | 0.0000 | [0.000, 0.000] | 0.000e+00 | **YES** | 0.051 | 2,074,368 | **0** |
| `conjunction` | 2 | 0 | 109 | 49.119 → 49.119 | 0.0000 | [0.000, 0.000] | 0.000e+00 | **YES** | 0.0432 | 2,078,464 | **0** |
| `conjunction_live` | 0 | 5,511 | 109 | 47.495 → 47.495 | 0.0000 | [0.000, 0.000] | 5.383e-07 | **YES** | 0.3916 | 2,081,024 | **0** |
| `conjunction_live` | 1 | 5,511 | 109 | 47.404 → 47.404 | 0.0000 | [0.000, 0.000] | 1.074e-08 | **YES** | 0.7678 | 2,074,368 | **0** |
| `conjunction_live` | 2 | 5,511 | 109 | 49.119 → 49.119 | 0.0000 | [0.000, 0.000] | 6.253e-08 | **YES** | 0.8311 | 2,078,464 | **0** |
| `item` | 0 | 1,311 | 109 | 47.495 → 47.495 | 0.0000 | [0.000, 0.000] | 2.338e-07 | **YES** | 0.1055 | 2,081,024 | **0** |
| `item` | 1 | 1,311 | 109 | 47.404 → 47.404 | 0.0000 | [0.000, 0.000] | 1.486e-08 | **YES** | 0.4182 | 2,074,368 | **0** |
| `item` | 2 | 1,311 | 109 | 49.119 → 49.119 | 0.0000 | [0.000, 0.000] | 3.162e-08 | **YES** | 0.3837 | 2,078,464 | **0** |

### The rival's own cost line on the same scope — refit-from-scratch-without

| scope | seed | arm | seconds | data rows read | bytes rewritten | refit? |
|---|---|---|---|---|---|---|
| `value` | 0 | ours (`cov_tensor`) | 0.8998 | **0** | 2,081,024 | no |
| `value` | 0 | refit-without (the store reference) | 9.259 | 120,028 | 1,065,216 | yes |
| `value` | 1 | ours (`cov_tensor`) | 3.7503 | **0** | 2,074,368 | no |
| `value` | 1 | refit-without (the store reference) | 18.595 | 120,028 | 1,065,216 | yes |
| `value` | 2 | ours (`cov_tensor`) | 1.3085 | **0** | 2,078,464 | no |
| `value` | 2 | refit-without (the store reference) | 14.43 | 120,028 | 1,065,216 | yes |
| `conjunction` | 0 | ours (`cov_tensor`) | 0.0518 | **0** | 2,081,024 | no |
| `conjunction` | 0 | refit-without (the store reference) | 9.368 | 131,072 | 1,065,216 | yes |
| `conjunction` | 1 | ours (`cov_tensor`) | 0.051 | **0** | 2,074,368 | no |
| `conjunction` | 1 | refit-without (the store reference) | 11.978 | 131,072 | 1,065,216 | yes |
| `conjunction` | 2 | ours (`cov_tensor`) | 0.0432 | **0** | 2,078,464 | no |
| `conjunction` | 2 | refit-without (the store reference) | 13.293 | 131,072 | 1,065,216 | yes |
| `conjunction_live` | 0 | ours (`cov_tensor`) | 0.3916 | **0** | 2,081,024 | no |
| `conjunction_live` | 0 | refit-without (the store reference) | 9.262 | 125,561 | 1,065,216 | yes |
| `conjunction_live` | 1 | ours (`cov_tensor`) | 0.7678 | **0** | 2,074,368 | no |
| `conjunction_live` | 1 | refit-without (the store reference) | 17.402 | 125,561 | 1,065,216 | yes |
| `conjunction_live` | 2 | ours (`cov_tensor`) | 0.8311 | **0** | 2,078,464 | no |
| `conjunction_live` | 2 | refit-without (the store reference) | 11.539 | 125,561 | 1,065,216 | yes |
| `item` | 0 | ours (`cov_tensor`) | 0.1055 | **0** | 2,081,024 | no |
| `item` | 0 | refit-without (the store reference) | 9.788 | 129,761 | 1,065,216 | yes |
| `item` | 1 | ours (`cov_tensor`) | 0.4182 | **0** | 2,074,368 | no |
| `item` | 1 | refit-without (the store reference) | 17.624 | 129,761 | 1,065,216 | yes |
| `item` | 2 | ours (`cov_tensor`) | 0.3837 | **0** | 2,078,464 | no |
| `item` | 2 | refit-without (the store reference) | 18.649 | 129,761 | 1,065,216 | yes |

## The RIVAL deletion column — a rival deletes ONLY by retrain-without

> A RIVAL DELETES ONLY BY RETRAIN-WITHOUT. Its cost line is quoted from outputs/w12-deletion-baselines/BASELINES-READY.json. Rival as-trained is printed beside it, never as a deletion mechanism.

Every figure on a row is measured on the **same survivor corners** as the arm, so the rival's as-trained → retrain-without step is a before/after on exactly the corners our before/after uses.

| scope | seed | survivors | family | rival **as-trained** (survivors) | rival **retrain-without** (survivors) | rival after − before | ours after deletion | paired ours − rival | CI95 | won / tied / lost | beats 1.0? | rival retrain seconds | rival rows read |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `value` | 0 | 89 | `mlp` | 5.551 | 4.022 | -1.528 | 50.921 | 46.899 | [43.15, 50.58] | 89 / 0 / 0 | **YES** | 8.86 | 15,032 |
| `value` | 0 | 89 | `lstm` | 5.247 | 3.708 | -1.539 | 50.921 | 47.213 | [43.57, 50.85] | 89 / 0 / 0 | **YES** | 15.99 | 15,032 |
| `value` | 0 | 89 | `transformer` | 1.494 | 0.865 | -0.629 | 50.921 | 50.056 | [46.21, 53.85] | 89 / 0 / 0 | **YES** | 34.62 | 15,032 |
| `value` | 0 | 89 | `cont_mlp` | 0.034 | 0.067 | 0.034 | 50.921 | 50.854 | [46.99, 54.69] | 89 / 0 / 0 | **YES** | 8.87 | 15,032 |
| `value` | 1 | 89 | `mlp` | 2.865 | 2.169 | -0.697 | 50.528 | 48.360 | [44.49, 52.20] | 89 / 0 / 0 | **YES** | 16.74 | 15,032 |
| `value` | 1 | 89 | `lstm` | 4.652 | 3.921 | -0.730 | 50.528 | 46.607 | [43.04, 50.11] | 89 / 0 / 0 | **YES** | 19.91 | 15,032 |
| `value` | 1 | 89 | `transformer` | 0.596 | 0.112 | -0.483 | 50.528 | 50.416 | [46.55, 54.17] | 89 / 0 / 0 | **YES** | 37.14 | 15,032 |
| `value` | 1 | 89 | `cont_mlp` | 0.034 | 0.056 | 0.022 | 50.528 | 50.472 | [46.60, 54.25] | 89 / 0 / 0 | **YES** | 13.59 | 15,032 |
| `value` | 2 | 89 | `mlp` | 8.697 | 6.697 | -2.000 | 52.607 | 45.910 | [42.33, 49.55] | 89 / 0 / 0 | **YES** | 6.35 | 15,032 |
| `value` | 2 | 89 | `lstm` | 10.663 | 8.449 | -2.213 | 52.607 | 44.157 | [40.60, 47.74] | 89 / 0 / 0 | **YES** | 11.57 | 15,032 |
| `value` | 2 | 89 | `transformer` | 3.360 | 2.281 | -1.079 | 52.607 | 50.326 | [46.64, 54.02] | 89 / 0 / 0 | **YES** | 23.24 | 15,032 |
| `value` | 2 | 89 | `cont_mlp` | 0.067 | 0.112 | 0.045 | 52.607 | 52.494 | [48.72, 56.24] | 89 / 0 / 0 | **YES** | 5.85 | 15,032 |
| `conjunction` | 0 | — | — | **DECLARED NOT-RUN** — red AND cube deletes 0 rows, and w12_deletion_baselines.partition_fit_rows REFUSES a no-op spec by construction ('a deletion that removes zero rows is not a deletion'). No comparator checkpoint exists or can exist for this scope. Reported as a declared NOT-RUN, never a zero. | | | | | | | | | |
| `conjunction` | 1 | — | — | **DECLARED NOT-RUN** — red AND cube deletes 0 rows, and w12_deletion_baselines.partition_fit_rows REFUSES a no-op spec by construction ('a deletion that removes zero rows is not a deletion'). No comparator checkpoint exists or can exist for this scope. Reported as a declared NOT-RUN, never a zero. | | | | | | | | | |
| `conjunction` | 2 | — | — | **DECLARED NOT-RUN** — red AND cube deletes 0 rows, and w12_deletion_baselines.partition_fit_rows REFUSES a no-op spec by construction ('a deletion that removes zero rows is not a deletion'). No comparator checkpoint exists or can exist for this scope. Reported as a declared NOT-RUN, never a zero. | | | | | | | | | |
| `conjunction_live` | 0 | 109 | `mlp` | 5.193 | 4.413 | -0.780 | 47.495 | 43.083 | [39.94, 46.31] | 109 / 0 / 0 | **YES** | 9.22 | 15,717 |
| `conjunction_live` | 0 | 109 | `lstm` | 4.761 | 4.055 | -0.706 | 47.495 | 43.440 | [40.33, 46.64] | 109 / 0 / 0 | **YES** | 15.66 | 15,717 |
| `conjunction_live` | 0 | 109 | `transformer` | 1.229 | 1.000 | -0.229 | 47.495 | 46.495 | [43.16, 49.95] | 109 / 0 / 0 | **YES** | 31.80 | 15,717 |
| `conjunction_live` | 0 | 109 | `cont_mlp` | 0.028 | 0.046 | 0.018 | 47.495 | 47.450 | [44.06, 50.97] | 109 / 0 / 0 | **YES** | 8.51 | 15,717 |
| `conjunction_live` | 1 | 109 | `mlp` | 3.239 | 2.890 | -0.349 | 47.404 | 44.514 | [41.01, 48.03] | 109 / 0 / 0 | **YES** | 15.02 | 15,717 |
| `conjunction_live` | 1 | 109 | `lstm` | 4.633 | 4.826 | 0.193 | 47.404 | 42.578 | [39.50, 45.70] | 109 / 0 / 0 | **YES** | 14.85 | 15,717 |
| `conjunction_live` | 1 | 109 | `transformer` | 0.651 | 0.284 | -0.367 | 47.404 | 47.119 | [43.79, 50.50] | 109 / 0 / 0 | **YES** | 40.95 | 15,717 |
| `conjunction_live` | 1 | 109 | `cont_mlp` | 0.028 | 0.037 | 0.009 | 47.404 | 47.367 | [44.02, 50.75] | 109 / 0 / 0 | **YES** | 13.72 | 15,717 |
| `conjunction_live` | 2 | 109 | `mlp` | 8.174 | 7.284 | -0.890 | 49.119 | 41.835 | [38.71, 44.96] | 109 / 0 / 0 | **YES** | 5.38 | 15,717 |
| `conjunction_live` | 2 | 109 | `lstm` | 9.404 | 7.982 | -1.422 | 49.119 | 41.138 | [38.16, 44.13] | 109 / 0 / 0 | **YES** | 11.11 | 15,717 |
| `conjunction_live` | 2 | 109 | `transformer` | 3.073 | 2.211 | -0.862 | 49.119 | 46.908 | [43.64, 50.19] | 109 / 0 / 0 | **YES** | 23.09 | 15,717 |
| `conjunction_live` | 2 | 109 | `cont_mlp` | 0.055 | 0.055 | 0.000 | 49.119 | 49.064 | [45.72, 52.48] | 109 / 0 / 0 | **YES** | 5.26 | 15,717 |
| `item` | 0 | 109 | `mlp` | 5.193 | 5.092 | -0.101 | 47.495 | 42.404 | [39.25, 45.63] | 109 / 0 / 0 | **YES** | 10.28 | 16,220 |
| `item` | 0 | 109 | `lstm` | 4.761 | 4.147 | -0.615 | 47.495 | 43.349 | [40.21, 46.55] | 109 / 0 / 0 | **YES** | 22.25 | 16,220 |
| `item` | 0 | 109 | `transformer` | 1.229 | 1.128 | -0.101 | 47.495 | 46.367 | [43.00, 49.83] | 109 / 0 / 0 | **YES** | 39.66 | 16,220 |
| `item` | 0 | 109 | `cont_mlp` | 0.028 | 0.028 | 0.000 | 47.495 | 47.468 | [44.07, 50.99] | 109 / 0 / 0 | **YES** | 10.62 | 16,220 |
| `item` | 1 | 109 | `mlp` | 3.239 | 3.211 | -0.028 | 47.404 | 44.193 | [40.64, 47.74] | 109 / 0 / 0 | **YES** | 6.75 | 16,220 |
| `item` | 1 | 109 | `lstm` | 4.633 | 4.560 | -0.073 | 47.404 | 42.844 | [39.79, 45.93] | 109 / 0 / 0 | **YES** | 12.42 | 16,220 |
| `item` | 1 | 109 | `transformer` | 0.651 | 0.560 | -0.092 | 47.404 | 46.844 | [43.49, 50.24] | 109 / 0 / 0 | **YES** | 24.00 | 16,220 |
| `item` | 1 | 109 | `cont_mlp` | 0.028 | 0.028 | 0.000 | 47.404 | 47.376 | [44.03, 50.76] | 109 / 0 / 0 | **YES** | 6.45 | 16,220 |
| `item` | 2 | 109 | `mlp` | 8.174 | 7.789 | -0.385 | 49.119 | 41.330 | [38.15, 44.54] | 109 / 0 / 0 | **YES** | 6.06 | 16,220 |
| `item` | 2 | 109 | `lstm` | 9.404 | 9.312 | -0.092 | 49.119 | 39.807 | [36.84, 42.74] | 109 / 0 / 0 | **YES** | 10.77 | 16,220 |
| `item` | 2 | 109 | `transformer` | 3.073 | 3.037 | -0.037 | 49.119 | 46.083 | [42.83, 49.33] | 109 / 0 / 0 | **YES** | 21.97 | 16,220 |
| `item` | 2 | 109 | `cont_mlp` | 0.055 | 0.055 | 0.000 | 49.119 | 49.064 | [45.72, 52.48] | 109 / 0 / 0 | **YES** | 5.76 | 16,220 |

## What did NOT run

- **`single_design_runs_2_and_3`** — DECLARED NOT-NEEDED and NOT AVAILABLE. Already spent by w12b hub/nr6-and-kstar on 2026-09-21T21:33:51Z on exactly this content; the target is exhausted at 3/3 and this hub spent 0.
- **`the_1x_comparator_column`** — DECLARED NOT-RUN, carried from AMENDMENT 1 A1.2. The comparator set is matched to param_target = 131,072 (the 524,288 B REFERENCE column), never to this object's own four-part total. A 1x-matched rival would be STRONGER, so the available comparison FLATTERS this arm and no row here is reported as a 1x result.
- **`rival_retrain_without_on_the_red_AND_cube_scope`** — DECLARED NOT-RUN and cannot exist: the scope deletes 0 rows and w12_deletion_baselines.partition_fit_rows refuses a no-op spec by construction.
- **`the_refusal_DISPLAY_comparator`** — DECLARED NOT-RUN, carried. The only permitted comparator is the refuse-token transformer; wave 9 measured a refuser of that quality at +0.047 separation with a ridge probe at AUC 0.48-0.55, i.e. chance, so the comparison is not yet informative.
- **`seeds_3_and_4`** — DECLARED NOT-RUN, carried from AMENDMENT 1 A1.1 and w12b's row: they pass the cache rule but have NO decoder and NO external classifier, so fw.Judge(seed) cannot be built. Never padded to 4 or 5 seeds.

