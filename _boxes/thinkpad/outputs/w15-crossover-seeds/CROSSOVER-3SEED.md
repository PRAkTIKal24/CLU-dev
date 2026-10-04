# CROSSOVER-3SEED -- the w12 K-crossover at n = 3 seeds

**Condition `val`, the free selection condition (charter A.2). `SMOKE_NOT_A_CLAIM`. Scored runs spent by this spoke: ZERO.**

Every number below is read from a `row.json` listed in the `provenance` block of `CROSSOVER-3SEED.json` and in section 6. No number in this file lives only in prose.

**Notation.** `K` = wells. `n` in this artefact is the SEED count and it is a column on every table. (`n` = factors jointly composed is a different quantity and is not used here.)

## Byte convention

> HONEST FOUR-PART TOTAL, uncompressed raw array bytes, SUMMED AND NEVER NETTED: (1) atoms = sum_k modes_k.size x 4 B (float32, the deployed content);  
> (2) covariance + count = sum_k stat_floats_k x 8 B (float64 -- the precision actually retained;  
> counting it at 4 would be netting by dtype) + 11,488 B of count table;  
> (3) address book = TensorAddress.byte_ledger()['address_bytes'];  
> (4) coupling/hole table = 5,288 + 1,322 = 6,610 B. No entropy coding, no deduplication, no subtraction of any part from any other. The 524,288 B pin is a REFERENCE COLUMN for continuity with waves 5-8, NOT a budget in this sweep.

## 1. Composition and the four-part total vs K, n = 3 at every K

| K | n | A | well width | rank/width | val comp mean | seed sd | seed band | drop vs K=5 | 4-part total mean B | seed sd B | seed band B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3 | 64 | 409..410 | 0.1561..0.1565 | 0.8953 | 0.0065 | [0.8912, 0.9028] | +0.0000 | 4,118,034 | 0 | [4,118,034, 4,118,034] |
| 8 | 3 | 64 | 256..256 | 0.2500..0.2500 | 0.9037 | 0.0197 | [0.8837, 0.9230] | -0.0085 | 2,959,602 | 0 | [2,959,602, 2,959,602] |
| 10 | 3 | 64 | 204..205 | 0.3122..0.3137 | 0.9014 | 0.0211 | [0.8825, 0.9242] | -0.0062 | 2,606,754 | 0 | [2,606,754, 2,606,754] |
| 12 | 3 | 64 | 170..171 | 0.3743..0.3765 | 0.9082 | 0.0141 | [0.8929, 0.9207] | -0.0129 | 2,393,714 | 0 | [2,393,714, 2,393,714] |
| 16 | 3 | 64 | 128..128 | 0.5000..0.5000 | 0.9111 | 0.0203 | [0.8906, 0.9311] | -0.0158 | 2,177,330 | 0 | [2,177,330, 2,177,330] |
| 20 | 3 | 64 | 102..103 | 0.5294..0.6275 | 0.9178 | 0.0285 | [0.8877, 0.9444] | -0.0226 | 2,098,879 | 2,864 | [2,095,586, 2,100,786] |
| 24 | 3 | 64 | 85..86 | 0.4353..0.7529 | 0.9223 | 0.0235 | [0.8981, 0.9450] | -0.0270 | 2,086,677 | 7,060 | [2,080,090, 2,094,130] |
| 28 | 3 | 64 | 73..74 | 0.3425..0.8767 | 0.9267 | 0.0276 | [0.8976, 0.9525] | -0.0314 | 2,117,183 | 9,885 | [2,107,130, 2,126,890] |
| 32 | 3 | 63 | 64..64 | 0.2656..0.9844 | 0.9290 | 0.0259 | [0.9016, 0.9531] | -0.0338 | 2,140,538 | 3,640 | [2,136,378, 2,143,138] |

Seeds: [0, 1, 2]. Frozen rule (w12, declared 2026-09-21T11:10:24Z, applied here unchanged at n = 3): **K\* = 32**, **K_crossover = None** -- no K in the swept range drops more than 0.1 below K=5 at n = 3; the range is NOT extended to manufacture a crossover.

### 1b. Is the frozen rule resolvable at n = 3?

The frozen tolerance is 0.02 in composition. The table below puts the SEED sd beside it at every K. Where the seed sd is wider than the tolerance the rule cannot separate two K on three seeds, and the honest statement is that it cannot -- not that it did.

| K | n | composition seed sd | tolerance | seed sd wider than tolerance |
|---|---|---|---|---|
| 5 | 3 | 0.0065 | 0.02 | no |
| 8 | 3 | 0.0197 | 0.02 | no |
| 10 | 3 | 0.0211 | 0.02 | YES |
| 12 | 3 | 0.0141 | 0.02 | no |
| 16 | 3 | 0.0203 | 0.02 | YES |
| 20 | 3 | 0.0285 | 0.02 | YES |
| 24 | 3 | 0.0235 | 0.02 | YES |
| 28 | 3 | 0.0276 | 0.02 | YES |
| 32 | 3 | 0.0259 | 0.02 | YES |

## 2. The honest four-part ledger, per K per seed -- never netted

| K | seed | atoms B | cov+count B | address B | coupling/hole B | **honest total B** | x 524,288 B REFERENCE |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 524,288 | 3,391,552 | 195,584 | 6,610 | **4,118,034** | 7.8545 |
| 5 | 1 | 524,288 | 3,391,552 | 195,584 | 6,610 | **4,118,034** | 7.8545 |
| 5 | 2 | 524,288 | 3,391,552 | 195,584 | 6,610 | **4,118,034** | 7.8545 |
| 8 | 0 | 524,288 | 2,133,280 | 295,424 | 6,610 | **2,959,602** | 5.6450 |
| 8 | 1 | 524,288 | 2,133,280 | 295,424 | 6,610 | **2,959,602** | 5.6450 |
| 8 | 2 | 524,288 | 2,133,280 | 295,424 | 6,610 | **2,959,602** | 5.6450 |
| 10 | 0 | 524,288 | 1,713,872 | 361,984 | 6,610 | **2,606,754** | 4.9720 |
| 10 | 1 | 524,288 | 1,713,872 | 361,984 | 6,610 | **2,606,754** | 4.9720 |
| 10 | 2 | 524,288 | 1,713,872 | 361,984 | 6,610 | **2,606,754** | 4.9720 |
| 12 | 0 | 524,288 | 1,434,272 | 428,544 | 6,610 | **2,393,714** | 4.5656 |
| 12 | 1 | 524,288 | 1,434,272 | 428,544 | 6,610 | **2,393,714** | 4.5656 |
| 12 | 2 | 524,288 | 1,434,272 | 428,544 | 6,610 | **2,393,714** | 4.5656 |
| 16 | 0 | 524,288 | 1,084,768 | 561,664 | 6,610 | **2,177,330** | 4.1529 |
| 16 | 1 | 524,288 | 1,084,768 | 561,664 | 6,610 | **2,177,330** | 4.1529 |
| 16 | 2 | 524,288 | 1,084,768 | 561,664 | 6,610 | **2,177,330** | 4.1529 |
| 20 | 0 | 524,288 | 875,104 | 694,784 | 6,610 | **2,100,786** | 4.0069 |
| 20 | 1 | 524,288 | 875,104 | 689,584 | 6,610 | **2,095,586** | 3.9970 |
| 20 | 2 | 524,288 | 875,104 | 694,264 | 6,610 | **2,100,266** | 4.0059 |
| 24 | 0 | 524,288 | 735,328 | 827,904 | 6,610 | **2,094,130** | 3.9942 |
| 24 | 1 | 524,288 | 735,328 | 813,864 | 6,610 | **2,080,090** | 3.9675 |
| 24 | 2 | 524,288 | 735,328 | 819,584 | 6,610 | **2,085,810** | 3.9784 |
| 28 | 0 | 524,288 | 635,488 | 960,504 | 6,610 | **2,126,890** | 4.0567 |
| 28 | 1 | 524,288 | 635,488 | 940,744 | 6,610 | **2,107,130** | 4.0190 |
| 28 | 2 | 524,288 | 635,488 | 951,144 | 6,610 | **2,117,530** | 4.0389 |
| 32 | 0 | 516,096 | 560,608 | 1,059,824 | 6,610 | **2,143,138** | 4.0877 |
| 32 | 1 | 516,096 | 560,608 | 1,053,064 | 6,610 | **2,136,378** | 4.0748 |
| 32 | 2 | 516,096 | 560,608 | 1,058,784 | 6,610 | **2,142,098** | 4.0857 |

The four parts are printed separately and summed. No part is subtracted from any other and no part is dropped. The 524,288 B column is a REFERENCE for continuity with waves 5-8, not a budget in this sweep.

## 3. The U-shape, decided by the test declared before the numbers

**Claim under test:** the honest four-part total is U-shaped in K with its minimum near K = 24.

**Test:** LOG.md section 5 (S1-S4), written before the seed-1 and seed-2 totals existed. S1 = strictly decreasing to the minimum and strictly increasing after it. S2 = the minimum lies in [20, 24, 28]. S3 = S1 and S2 on all three seeds.

| seed | K=5 | K=8 | K=10 | K=12 | K=16 | K=20 | K=24 | K=28 | K=32 | argmin K | S1 U-shaped | S2 near K=24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4,118,034 | 2,959,602 | 2,606,754 | 2,393,714 | 2,177,330 | 2,100,786 | **2,094,130** | 2,126,890 | 2,143,138 | **24** | yes | yes |
| 1 | 4,118,034 | 2,959,602 | 2,606,754 | 2,393,714 | 2,177,330 | 2,095,586 | **2,080,090** | 2,107,130 | 2,136,378 | **24** | yes | yes |
| 2 | 4,118,034 | 2,959,602 | 2,606,754 | 2,393,714 | 2,177,330 | 2,100,266 | **2,085,810** | 2,117,530 | 2,142,098 | **24** | yes | yes |

Bold cell = that seed's minimum. Bytes are the honest four-part total under the convention named above.

- per-seed minima: `{'0': 24, '1': 24, '2': 24}`; distinct minima across seeds: `[24]`
- S1 (every seed U-shaped): **True**
- S2 (every minimum in [20, 24, 28]): **True**
- **S3 VERDICT: CONFIRMED**. S3 REFUTED is a RESULT, not a failure: the per-seed minima are then the reportable quantity and the sentence is not claimed.
- S4 resolvability at K = 24: seed band sd 7,060 B; neighbour gaps K=20 12,203 B (clears the band), K=28 30,507 B (clears the band); minimum resolvable: **True**

## 4. The charter-C controls, every K, every seed

| K | n | landscape_removed | landscape_shuffled | landscape_shuffled_across_wells | random_atoms | all four collapse (strict) |
|---|---|---|---|---|---|---|
| 5 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 8 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 10 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 12 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 16 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 20 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0139 (mean 0.000193) | 0.0000 (mean 0.000000) | NO |
| 24 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 28 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |
| 32 | 3 | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | 0.0000 (mean 0.000000) | YES |

Each cell is `worst` = max over the 24 val corners then max over the 3 seeds, with the mean over corners and seeds in brackets. A control that does not collapse means the store is DECORATIVE at that K and no row may be claimed for it. The gate above is STRICT: it asks for exactly zero on the worst of the 24 corners on the worst of the 3 seeds.

Cells that did NOT collapse under the strict gate, named rather than averaged away:

| K | seed | control | valid max over 24 corners | valid mean over 24 corners | the arm at that cell |
|---|---|---|---|---|---|
| 20 | 1 | `landscape_shuffled_across_wells` | 0.0139 | 0.000579 | 0.9213 |

`landscape_shuffled` (the SHIPPED charter C.2 definition -- atom VALUES permuted inside each well) is the control w12 declared the verdict is taken from, and it is reported beside the strict gate, never instead of it. `landscape_shuffled_across_wells` is the task file's literal wording and is weaker by construction: w12's own `scripts/w12_crossover_lib.py` records that its permutation is uniform with only the IDENTITY rejected, so about one well in K keeps its own atoms. Both are reported; neither is dropped.

| K | n | strict gate (all four) | primary C.2 control `landscape_shuffled` |
|---|---|---|---|
| 5 | 3 | PASS | collapses |
| 8 | 3 | PASS | collapses |
| 10 | 3 | PASS | collapses |
| 12 | 3 | PASS | collapses |
| 16 | 3 | PASS | collapses |
| 20 | 3 | FAIL | collapses |
| 24 | 3 | PASS | collapses |
| 28 | 3 | PASS | collapses |
| 32 | 3 | PASS | collapses |

## 5. Store-scoped exactness, worst case over seeds

Definition: the frozen encoder and decoder are declared OUTSIDE the deletable object and are NOT refit. Exactness is store-scoped -- `downdate(delete the red rows)` against a store fitted from scratch without those rows, same partition, same rank. Post-deletion adaptation is light and data-free; it is not "no changes". A rival deletes only by retrain-without, and that cost is printed in w12's own deletion artefacts, not here.

| K | n | max abs mode deviation | max subspace projector distance | ranks agree | rows refitted | data passes during deletion |
|---|---|---|---|---|---|---|
| 5 | 3 | 1.455e-11 | 1.430e-11 | True | 0 | 0 |
| 8 | 3 | 1.819e-12 | 1.370e-12 | True | 0 | 0 |
| 10 | 3 | 2.910e-11 | 1.981e-11 | True | 0 | 0 |
| 12 | 3 | 2.328e-10 | 2.126e-10 | True | 0 | 0 |
| 16 | 3 | 1.164e-10 | 1.164e-10 | True | 0 | 0 |
| 20 | 3 | 7.431e-08 | 1.976e-07 | True | 0 | 0 |
| 24 | 3 | 3.254e-08 | 5.865e-08 | True | 0 | 0 |
| 28 | 3 | 2.472e-08 | 4.667e-08 | True | 0 | 0 |
| 32 | 3 | 2.147e-07 | 2.578e-07 | True | 0 | 0 |

## 6. Provenance -- every cell, its wave and its row hash

| cell | source wave | row.json sha256 (first 16) |
|---|---|---|
| K05_seed0 | w12-single-design-crossover | `c449bee42863688a` |
| K05_seed1 | w12-single-design-crossover | `ccbde06e103efadd` |
| K05_seed2 | w12-single-design-crossover | `78e1f03e715fc8d3` |
| K08_seed0 | w12-single-design-crossover | `9a1d4bdc6f6e3ea2` |
| K08_seed1 | w15-crossover-seeds | `6694b52de077b5a4` |
| K08_seed2 | w15-crossover-seeds | `f0b4f736d0e0aaff` |
| K10_seed0 | w12-single-design-crossover | `0c67ba117c6d776b` |
| K10_seed1 | w15-crossover-seeds | `6f92bf3ab433ab45` |
| K10_seed2 | w15-crossover-seeds | `b390b57dc6e9dbcb` |
| K12_seed0 | w12-single-design-crossover | `25dff5678f70f0e9` |
| K12_seed1 | w15-crossover-seeds | `1391acce5282d0a5` |
| K12_seed2 | w15-crossover-seeds | `50a3aea51095b853` |
| K16_seed0 | w12-single-design-crossover | `7c9508106592c391` |
| K16_seed1 | w15-crossover-seeds | `269c4559996614e4` |
| K16_seed2 | w15-crossover-seeds | `10f505cf4db9848a` |
| K20_seed0 | w12-single-design-crossover | `1c61cb99732ae210` |
| K20_seed1 | w15-crossover-seeds | `49d7d01a06e683bb` |
| K20_seed2 | w15-crossover-seeds | `a997c5377c0b0c9a` |
| K24_seed0 | w12-single-design-crossover | `ca3d8563141740df` |
| K24_seed1 | w15-crossover-seeds | `e18ceefde0cecc18` |
| K24_seed2 | w15-crossover-seeds | `4d8bb04282700519` |
| K28_seed0 | w12-single-design-crossover | `9afc452dd776ff37` |
| K28_seed1 | w15-crossover-seeds | `ed0131c8db057624` |
| K28_seed2 | w15-crossover-seeds | `adfaeb8e3fb01015` |
| K32_seed0 | w12-single-design-crossover | `5ad5ae8c7ad6e42a` |
| K32_seed1 | w12-single-design-crossover | `4e72c610c0cf993b` |
| K32_seed2 | w12-single-design-crossover | `d89643467f89ade2` |

