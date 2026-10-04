# NR6 — T7 with its seed count printed

Built by `scripts/w12b_nr6_table.py` from the scored artefacts only.

## Sources

- `K32_seed0` → `C:\chlu\outputs\w12-single-design-crossover\scored\run1_seed0_K32\scored.json`
- `K32_seed1` → `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed1_K32\scored.json`
- `K32_seed2` → `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run3_seed2_K32\scored.json`
- `K5_seed0` → `C:\chlu\outputs\w12b-nr6-and-kstar\scored\run2_seed0_K5\scored.json`
- `K5_seed1` → **DECLARED NOT-RUN: C:\chlu\outputs\w12b-nr6-and-kstar\scored\run2_seed1_K5\scored.json does not exist**
- `K5_seed2` → **DECLARED NOT-RUN: C:\chlu\outputs\w12b-nr6-and-kstar\scored\run2_seed2_K5\scored.json does not exist**

## The control gate (charter C.2) — checked BEFORE any margin

| leg | arm | `landscape_removed` | `landscape_shuffled` | `landscape_shuffled_across_wells` | `random_atoms` | all collapse? |
|---|---|---|---|---|---|---|
| `K32_seed0` | 47.4954 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| `K32_seed1` | 47.4037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| `K32_seed2` | 49.1193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |
| `K5_seed0` | 43.7890 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **YES** |

**ALL LEGS PASS: True**  ⛔ If this is False no row is claimed — that is the result.

## Every arm, every K, every seed — `distinct_valid_novel_tuples` per corner

| arm | row | `K32_seed0` | `K32_seed1` | `K32_seed2` | `K5_seed0` |
|---|---|---|---|---|---|
| `cov_tensor` | row 3 (pair) | 31.6000 | 32.2333 | 33.2167 | 28.8167 |
| `cov_tensor` | row 4 (n-way) | 66.9592 | 65.9796 | 68.5918 | 62.1224 |
| `landscape_removed` | row 3 (pair) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `landscape_removed` | row 4 (n-way) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `landscape_shuffled` | row 3 (pair) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `landscape_shuffled` | row 4 (n-way) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `landscape_shuffled_across_wells` | row 3 (pair) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `landscape_shuffled_across_wells` | row 4 (n-way) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `random_atoms` | row 3 (pair) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `random_atoms` | row 4 (n-way) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mlp__as_trained` | row 3 (pair) | 2.9333 | 3.4333 | 5.2667 | 2.9333 |
| `mlp__as_trained` | row 4 (n-way) | 7.9592 | 3.0000 | 11.7347 | 7.9592 |
| `lstm__as_trained` | row 3 (pair) | 2.5000 | 2.7000 | 4.9167 | 2.5000 |
| `lstm__as_trained` | row 4 (n-way) | 7.5306 | 7.0000 | 14.8980 | 7.5306 |
| `transformer__as_trained` | row 3 (pair) | 0.6667 | 0.6667 | 1.7167 | 0.6667 |
| `transformer__as_trained` | row 4 (n-way) | 1.9184 | 0.6327 | 4.7347 | 1.9184 |
| `cont_mlp__as_trained` | row 3 (pair) | 0.0000 | 0.0000 | 0.0167 | 0.0000 |
| `cont_mlp__as_trained` | row 4 (n-way) | 0.0612 | 0.0612 | 0.1020 | 0.0612 |
| `table` | row 3 (pair) | 5.8333 | 3.4833 | 6.1833 | 5.8333 |
| `table` | row 4 (n-way) | 11.1633 | 7.0204 | 14.1224 | 11.1633 |
| `transformer__refit_postfix` | row 3 (pair) | DECLARED NOT-RUN | 0.3000 | 1.7333 | 0.3667 |
| `transformer__refit_postfix` | row 4 (n-way) | DECLARED NOT-RUN | 1.1429 | 3.1633 | 1.1020 |
| `lstm__refit_postfix` | row 3 (pair) | DECLARED NOT-RUN | 3.0833 | 4.3667 | 2.5667 |
| `lstm__refit_postfix` | row 4 (n-way) | DECLARED NOT-RUN | 5.0204 | 8.0612 | 7.7143 |

## T7, with the count

- **K32: n = 3 seeds [0, 1, 2]** — row 3 32.350 ± 0.815, row 4 67.177 ± 1.320; per seed row 3 [31.6, 32.233, 33.217], row 4 [66.959, 65.98, 68.592]
- **K5: n = 1 seeds [0]** — row 3 28.817 ± n/a, row 4 62.122 ± n/a; per seed row 3 [28.817], row 4 [62.122]

## The paired estimand — the verdict function, per leg

| leg | comparator | mean paired delta | CI95 | corners won/lost | beats the 1.0 margin? |
|---|---|---|---|---|---|
| `K32_seed0` | `landscape_removed` | 47.495 | [44.10, 51.03] | 109/0 | **True** |
| `K32_seed0` | `landscape_shuffled` | 47.495 | [44.10, 51.03] | 109/0 | **True** |
| `K32_seed0` | `landscape_shuffled_across_wells` | 47.495 | [44.10, 51.03] | 109/0 | **True** |
| `K32_seed0` | `random_atoms` | 47.495 | [44.10, 51.03] | 109/0 | **True** |
| `K32_seed0` | `mlp__as_trained` | 42.303 | [39.17, 45.53] | 109/0 | **True** |
| `K32_seed0` | `lstm__as_trained` | 42.734 | [39.68, 45.90] | 109/0 | **True** |
| `K32_seed0` | `transformer__as_trained` | 46.266 | [42.90, 49.72] | 109/0 | **True** |
| `K32_seed0` | `cont_mlp__as_trained` | 47.468 | [44.07, 50.99] | 109/0 | **True** |
| `K32_seed0` | `table` | 39.266 | [35.86, 42.68] | 109/0 | **True** |
| `K32_seed1` | `landscape_removed` | 47.404 | [44.06, 50.79] | 109/0 | **True** |
| `K32_seed1` | `landscape_shuffled` | 47.404 | [44.06, 50.79] | 109/0 | **True** |
| `K32_seed1` | `landscape_shuffled_across_wells` | 47.404 | [44.06, 50.79] | 109/0 | **True** |
| `K32_seed1` | `random_atoms` | 47.404 | [44.06, 50.79] | 109/0 | **True** |
| `K32_seed1` | `mlp__as_trained` | 44.165 | [40.61, 47.71] | 109/0 | **True** |
| `K32_seed1` | `lstm__as_trained` | 42.771 | [39.66, 45.89] | 109/0 | **True** |
| `K32_seed1` | `transformer__as_trained` | 46.752 | [43.37, 50.17] | 109/0 | **True** |
| `K32_seed1` | `cont_mlp__as_trained` | 47.376 | [44.03, 50.76] | 109/0 | **True** |
| `K32_seed1` | `table` | 42.330 | [39.04, 45.64] | 109/0 | **True** |
| `K32_seed1` | `transformer__refit_postfix` | 46.725 | [43.43, 50.07] | 109/0 | **True** |
| `K32_seed1` | `lstm__refit_postfix` | 43.450 | [40.12, 46.80] | 109/0 | **True** |
| `K32_seed2` | `landscape_removed` | 49.119 | [45.77, 52.53] | 109/0 | **True** |
| `K32_seed2` | `landscape_shuffled` | 49.119 | [45.77, 52.53] | 109/0 | **True** |
| `K32_seed2` | `landscape_shuffled_across_wells` | 49.119 | [45.77, 52.53] | 109/0 | **True** |
| `K32_seed2` | `random_atoms` | 49.119 | [45.77, 52.53] | 109/0 | **True** |
| `K32_seed2` | `mlp__as_trained` | 40.945 | [37.72, 44.18] | 109/0 | **True** |
| `K32_seed2` | `lstm__as_trained` | 39.716 | [36.73, 42.70] | 109/0 | **True** |
| `K32_seed2` | `transformer__as_trained` | 46.046 | [42.82, 49.29] | 109/0 | **True** |
| `K32_seed2` | `cont_mlp__as_trained` | 49.064 | [45.72, 52.48] | 109/0 | **True** |
| `K32_seed2` | `table` | 39.367 | [35.77, 42.91] | 109/0 | **True** |
| `K32_seed2` | `transformer__refit_postfix` | 46.743 | [43.38, 50.07] | 109/0 | **True** |
| `K32_seed2` | `lstm__refit_postfix` | 43.092 | [39.81, 46.35] | 109/0 | **True** |
| `K5_seed0` | `landscape_removed` | 43.789 | [40.36, 47.39] | 109/0 | **True** |
| `K5_seed0` | `landscape_shuffled` | 43.789 | [40.36, 47.39] | 109/0 | **True** |
| `K5_seed0` | `landscape_shuffled_across_wells` | 43.789 | [40.36, 47.39] | 109/0 | **True** |
| `K5_seed0` | `random_atoms` | 43.789 | [40.36, 47.39] | 109/0 | **True** |
| `K5_seed0` | `mlp__as_trained` | 38.596 | [35.30, 42.00] | 109/0 | **True** |
| `K5_seed0` | `lstm__as_trained` | 39.028 | [35.89, 42.23] | 109/0 | **True** |
| `K5_seed0` | `transformer__as_trained` | 42.560 | [39.12, 46.06] | 109/0 | **True** |
| `K5_seed0` | `cont_mlp__as_trained` | 43.761 | [40.33, 47.35] | 109/0 | **True** |
| `K5_seed0` | `table` | 35.560 | [32.18, 39.03] | 109/0 | **True** |
| `K5_seed0` | `transformer__refit_postfix` | 43.092 | [39.67, 46.62] | 109/0 | **True** |
| `K5_seed0` | `lstm__refit_postfix` | 38.908 | [35.71, 42.18] | 109/0 | **True** |

## The val/test gap at the selection rule's own quantity

```
{
 "quantity": "mean over corners of the per-corner VALID FRACTION (val: `valid_mean`; test: mean of `judge_accuracy`) -- the same expression, `valid.mean()` per corner, averaged over corners",
 "WARNING": "the corner SETS differ (24 val corners vs 60 + 49 test corners), so levels are NOT comparable across conditions and are never subtracted across them. The DROP from K = 5 to K = 32 within one condition IS the quantity the frozen selection rule used, and that is what is compared.",
 "rule": "K* = max{K : v(5) - mean_seeds v(K) <= 0.02}, frozen 11:10:24Z",
 "band": 0.02,
 "val_source": "C:\\chlu\\outputs\\w12-single-design-crossover\\CROSSOVER-val.json",
 "val": {
  "K5_mean": 0.8952546296296297,
  "K32_mean": 0.9290123456790124,
  "K5_per_seed": {
   "0": 0.8917824074074074,
   "1": 0.8912037037037037,
   "2": 0.9027777777777777
  },
  "K32_per_seed": {
   "0": 0.9016203703703706,
   "1": 0.9322916666666666,
   "2": 0.953125
  },
  "drop_K5_minus_K32": -0.03375771604938271
 },
 "test": {
  "K5_per_seed": {
   "0": 0.8284913353720693
  },
  "K32_per_seed": {
   "0": 0.9012487257900103,
   "1": 0.9048165137614679,
   "2": 0.9361620795107034
  },
  "K5_mean": 0.8284913353720693,
  "K32_mean": 0.9140757730207273,
  "drop_K5_minus_K32_all_available_seeds": -0.08558443764865797,
  "paired_seeds_used_for_the_drop": [
   0
  ],
  "drop_K5_minus_K32_paired": -0.07275739041794094
 },
 "TEST_DROP_WITHIN_THE_0.02_BAND": true,
 "reading": "the selection rule's band HOLDS on TEST"
}
```
