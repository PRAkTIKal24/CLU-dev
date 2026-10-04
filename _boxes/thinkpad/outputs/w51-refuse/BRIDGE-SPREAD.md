### ⭐⭐ THE BRIDGE-SEED SPREAD -- the parent's fixed instrument beside the full one

`fixed` = bridge seed 0 at every store seed (the parent). `joint` = bridge seed = store seed: encoder, decoder, classifier and cache move WITH the seed, which is `w5 hub/delete`'s convention.
Both columns are `mean +/- spread` over store seeds 0/1/2 in THE PARENT'S OWN convention -- ⚠ its `+/-` is the **full range** `max - min` (`exp_gen_refuse_report._agg`), not a half-range and not an sd, so the interval it writes is about twice the width of the three numbers it saw. Each column's observed `[min, max]` is printed beside it so either standard can be applied. Both columns are scored by the SAME frozen `score_deletion`. ⛔ Neither column is a scored run: both re-read artifacts of rows that were already scored.

| quantity | PARENT, fixed (as published) | mine, FIXED (its config, my build) | ⭐ mine, JOINT (the wider instrument) | observed [min,max]: parent / joint | spread ratio joint:parent | ⭐⭐ joint mean vs the parent's published +/- |
|---|---|---|---|---|---|---|
| EVICTION bytes changed | 19,300 +/- 2,649 | 19,300 +/- 2,649 | 17,816 +/- 2,350 | [17,569, 20,218] / [16,764, 19,114] | x0.89 | inside both |
| EVICTION bytes changed OUTSIDE permitted ⭐ THE CLAIM | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | [0, 0] / [0, 0] | both 0 | inside both |
| EVICTION items evicted | 6.7 +/- 1 | 6.7 +/- 1 | 6.3 +/- 1 | [6, 7] / [6, 7] | x1.00 | inside both |
| EVICTION read relative change on survivors | 0.0720 +/- 0.0204 | 0.0720 +/- 0.0204 | 0.0812 +/- 0.0197 | [0.0632, 0.0836] / [0.0693, 0.0890] | x0.96 | inside both |
| NEGATIVE CONTROL bytes (1e-3 nudge to one address-book row) | 5,720 +/- 41 | 5,720 +/- 41 | 5,748 +/- 57 | [5,694, 5,735] / [5,726, 5,783] | x1.39 | ⚠ inside published, OUTSIDE observed [min,max] |
| NOVEL TUPLES post-HOLE - `field_geom` (524,288 B) | 2,317.3 +/- 312 | 2,317.3 +/- 312 | 2,294.7 +/- 540 | [2,178, 2,490] / [2,030, 2,570] | x1.73 | inside both |
| NOVEL TUPLES post-HOLE - `bitmap8` (8 B) | 2,317.3 +/- 312 | 2,317.3 +/- 312 | 2,294.7 +/- 540 | [2,178, 2,490] / [2,030, 2,570] | x1.73 | inside both |
| NOVEL TUPLES post-HOLE - `pair_presence` (166 B) | 1,458.0 +/- 222 | 1,458.0 +/- 222 | 1,490.0 +/- 382 | [1,362, 1,584] / [1,330, 1,712] | x1.72 | inside both |
| NOVEL TUPLES post-HOLE - `pair_deletion` (174 B) | 2,317.3 +/- 312 | 2,317.3 +/- 312 | 2,294.7 +/- 540 | [2,178, 2,490] / [2,030, 2,570] | x1.73 | inside both |
| NOVEL TUPLES post-EVICTION - `field_geom` (524,288 B) | 1,964.7 +/- 362 | 1,964.7 +/- 362 | 1,877.7 +/- 183 | [1,781, 2,143] / [1,787, 1,970] | x0.51 | inside both |
| NOVEL TUPLES post-EVICTION - `bitmap8` (8 B) | 1,964.7 +/- 362 | 1,964.7 +/- 362 | 1,877.7 +/- 183 | [1,781, 2,143] / [1,787, 1,970] | x0.51 | inside both |
| NOVEL TUPLES post-EVICTION - `pair_presence` (166 B) | 1,433.3 +/- 322 | 1,433.3 +/- 322 | 1,427.3 +/- 72 | [1,260, 1,582] / [1,386, 1,458] | x0.22 | inside both |
| NOVEL TUPLES post-EVICTION - `pair_deletion` (174 B) | 1,964.7 +/- 362 | 1,964.7 +/- 362 | 1,877.7 +/- 183 | [1,781, 2,143] / [1,787, 1,970] | x0.51 | inside both |

### ⭐⭐ THE PAIRED TEST -- same store seed, two instruments

⚠ The spread ratios above can look calm while every cell moves, because a range over three cells is a blunt statistic. `mine-fixed` and `mine-joint` differ in **nothing but the bridge seed**, so each store seed can be differenced against itself. ⭐ Store seed 0 must be exactly 0 in every row -- the two instruments coincide there by construction -- and that is the internal control.

| quantity | fixed (s0/s1/s2) | ⭐ joint (s0/s1/s2) | delta per seed | **max abs delta** | the parent's WHOLE published +/- | ⛔ ratio |
|---|---|---|---|---|---|---|
| EVICTION bytes changed | 17,569 / 20,113 / 20,218 | 17,569 / 16,764 / 19,114 | 0 / -3,349 / -1,104 | **3,349** | 2,649 | ⛔ **1.26x** |
| EVICTION bytes changed OUTSIDE permitted ⭐ THE CLAIM | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | **0** | 0 | - |
| EVICTION items evicted | 6 / 7 / 7 | 6 / 6 / 7 | 0 / -1 / 0 | **1** | 1 | 1.00x |
| EVICTION read relative change on survivors | 0.0693 / 0.0632 / 0.0836 | 0.0693 / 0.0890 / 0.0853 | 0.0000 / 0.0258 / 0.0017 | **0.0258** | 0.0204 | ⛔ **1.26x** |
| NEGATIVE CONTROL bytes (1e-3 nudge to one address-book row) | 5,735 / 5,730 / 5,694 | 5,735 / 5,726 / 5,783 | 0 / -4 / 89 | **89** | 41 | ⛔ **2.17x** |
| NOVEL TUPLES post-HOLE - `field_geom` (524,288 B) | 2,284 / 2,490 / 2,178 | 2,284 / 2,030 / 2,570 | 0 / -460 / 392 | **460** | 312 | ⛔ **1.47x** |
| NOVEL TUPLES post-HOLE - `bitmap8` (8 B) | 2,284 / 2,490 / 2,178 | 2,284 / 2,030 / 2,570 | 0 / -460 / 392 | **460** | 312 | ⛔ **1.47x** |
| NOVEL TUPLES post-HOLE - `pair_presence` (166 B) | 1,428 / 1,584 / 1,362 | 1,428 / 1,330 / 1,712 | 0 / -254 / 350 | **350** | 222 | ⛔ **1.58x** |
| NOVEL TUPLES post-HOLE - `pair_deletion` (174 B) | 2,284 / 2,490 / 2,178 | 2,284 / 2,030 / 2,570 | 0 / -460 / 392 | **460** | 312 | ⛔ **1.47x** |
| NOVEL TUPLES post-EVICTION - `field_geom` (524,288 B) | 1,970 / 2,143 / 1,781 | 1,970 / 1,787 / 1,876 | 0 / -356 / 95 | **356** | 362 | 0.98x |
| NOVEL TUPLES post-EVICTION - `bitmap8` (8 B) | 1,970 / 2,143 / 1,781 | 1,970 / 1,787 / 1,876 | 0 / -356 / 95 | **356** | 362 | 0.98x |
| NOVEL TUPLES post-EVICTION - `pair_presence` (166 B) | 1,458 / 1,582 / 1,260 | 1,458 / 1,386 / 1,438 | 0 / -196 / 178 | **196** | 322 | 0.61x |
| NOVEL TUPLES post-EVICTION - `pair_deletion` (174 B) | 1,970 / 2,143 / 1,781 | 1,970 / 1,787 / 1,876 | 0 / -356 / 95 | **356** | 362 | 0.98x |

⛔⛔ **On 7 of 13 quantities, swapping the instrument at a FIXED store seed moves the number by MORE than the parent's entire published `+/-`.** Its `+/-` never contained the instrument, and on these quantities the instrument is the larger term. ⭐ Stated plainly, as the target file's section 5 requires: **the parent's `+/-` figures are understated.** That is the result, not a failure.

### The fixed points -- printed so that "did not move" is MEASURED, not assumed

| quantity | parent, fixed | mine, fixed | ⭐ mine, joint |
|---|---|---|---|
| HOLE bytes changed | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| HOLE bytes changed OUTSIDE permitted | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| HOLE reject deleted conjunction, `field_geom` | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| HOLE accept never-held conjunction, `field_geom` | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| HOLE delta-reject, `bitmap8` (THE BAR) | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| HOLE accept never-held conjunction, `pair_presence` | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| EVICT reject deleted value, `field_geom` | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| EVICT reject deleted value, `bitmap8` | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| EVICT accept survivors, `field_geom` | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| HOLE rendered features bit-identical | [True, True, True] | [True, True, True] | [True, True, True] |

### ⭐ The per-seed byte table -- the three numbers, not their summary

⚠ The mean and the spread of a byte count are **provenance, not a claim**: the count is the footprint of however many stored items happen to hold the evicted value under that pool and atom draw. ⭐ The claim is the column that reads **0** on every seed in every instrument.

| tree | store seed | evict: bytes changed | evict: OUTSIDE ⭐ | items evicted | hole: bytes | hole: OUTSIDE ⭐ | negative control |
|---|---|---|---|---|---|---|---|
| parent-fixed | 0 | 17,569 | **0** | 6 | 1 | **0** | 5,735 |
| parent-fixed | 1 | 20,113 | **0** | 7 | 1 | **0** | 5,730 |
| parent-fixed | 2 | 20,218 | **0** | 7 | 1 | **0** | 5,694 |
| mine-fixed | 0 | 17,569 | **0** | 6 | 1 | **0** | 5,735 |
| mine-fixed | 1 | 20,113 | **0** | 7 | 1 | **0** | 5,730 |
| mine-fixed | 2 | 20,218 | **0** | 7 | 1 | **0** | 5,694 |
| mine-joint | 0 | 17,569 | **0** | 6 | 1 | **0** | 5,735 |
| mine-joint | 1 | 16,764 | **0** | 6 | 1 | **0** | 5,726 |
| mine-joint | 2 | 19,114 | **0** | 7 | 1 | **0** | 5,783 |

### ⭐ Reproducibility of the PINNED arm -- my build against the parent's

Same configuration, my machine, `OMP_NUM_THREADS=3`: **29/29** per-seed cells identical (byte counts, item counts, negative control and every per-seed novel-tuple count). ⛔ Kept as a separate statement rather than folded into the dial contrast, because a reproducibility failure and an instrument effect are different things and must not be allowed to look alike.

### ⭐⭐ ARE THE TWO HUBS COMPARABLE NOW? -- the joint arm against `w5 hub/delete`'s own artifacts

⛔ Read from ITS `run1/seed_*/bytes.json`, not from its LOG. After `--bridge-mode joint` the two hubs select the same pool draw, the same atom seed, the same encoder, the same decoder and the same classifier off the same store-seed index, so **its numbers are a prediction for mine**, not merely a similar quantity. The parent's fixed column is printed beside, to show what was and was not comparable before.

| store seed | quantity | parent (fixed) | ⭐ this retry (joint) | `w5 hub/delete` | joint == delete? |
|---|---|---|---|---|---|
| 0 | evict: bytes changed | 17,569 | **17,569** | 17,569 | ⭐ **YES** |
| 0 | evict: bytes OUTSIDE ⭐ | 0 | **0** | 0 | ⭐ **YES** |
| 0 | items evicted | 6 | **6** | 6 | ⭐ **YES** |
| 0 | negative control bytes | 5,735 | **5,735** | 5,735 | ⭐ **YES** |
| 1 | evict: bytes changed | 20,113 | **16,764** | 16,764 | ⭐ **YES** |
| 1 | evict: bytes OUTSIDE ⭐ | 0 | **0** | 0 | ⭐ **YES** |
| 1 | items evicted | 7 | **6** | 6 | ⭐ **YES** |
| 1 | negative control bytes | 5,730 | **5,726** | 5,726 | ⭐ **YES** |
| 2 | evict: bytes changed | 20,218 | **19,114** | 19,114 | ⭐ **YES** |
| 2 | evict: bytes OUTSIDE ⭐ | 0 | **0** | 0 | ⭐ **YES** |
| 2 | items evicted | 7 | **7** | 7 | ⭐ **YES** |
| 2 | negative control bytes | 5,694 | **5,783** | 5,783 | ⭐ **YES** |

⭐⭐ **12/12 cells agree EXACTLY.** The two hubs' seed-1 and seed-2 numbers are now **comparable**, and the reason is mechanical rather than statistical: under `joint` they are the same store, built from the same bytes, so they may share a row.

### Instrument provenance -- which encoder/decoder/classifier each seed actually read

| tree | store seed | bridge seed | features sha256 | encoder hash | decoder hash | classifier hash |
|---|---|---|---|---|---|---|
| parent-fixed | 0 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| parent-fixed | 1 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| parent-fixed | 2 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| mine-fixed | 0 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| mine-fixed | 1 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| mine-fixed | 2 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| mine-joint | 0 | 0 | `ef6359a4cd54` | `98d326028316` | `01ef91f75eea` | `c8741ac417d8` |
| mine-joint | 1 | 1 | `86b44d0f1e64` | `fc05e2c5f396` | `001011bc603a` | `aa69b436989b` |
| mine-joint | 2 | 2 | `3ab14e583c3f` | `154d49861064` | `2d7eea75add7` | `ad16c4e53273` |
