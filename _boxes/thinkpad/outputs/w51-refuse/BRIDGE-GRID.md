### ⭐⭐ THE 3x3 GRID -- (store seed) x (bridge seed), and what each direction isolates

Rows are **store seeds** (the fit-pool draw and the atom selection). Columns are **bridge seeds** -- the frozen encoder, decoder and classifier the store is built from and read back by. ⛔ Nothing about the field store, the rule, the thresholds or the metric differs between cells; only which of the three frozen instruments is in use. ⛔ No cell is a scored run.

- **column `b=0`** = the PARENT'S OWN SPREAD (it held the bridge seed at 0);
- **a row** = the INSTRUMENT effect at a fixed store -- ⭐ never measured before;
- **the diagonal** = what `--bridge-mode joint` and `w5 hub/delete` report.

**EVICTION bytes changed**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **17,569** | 17,037 | 19,210 | 2,173 |
| store 1 | 20,113 | **16,764** | 19,168 | 3,349 |
| store 2 | 20,218 | 17,000 | **19,114** | 3,218 |
| ⭐ **store range (down the column)** | 2,649 | 273 | 96 | whole grid: 3,454 |

instrument range is **1.26x** the parent's store range (3,349 vs 2,649); diagonal range 2,350.  ⛔ **The parent's ± understates this quantity.**

**EVICTION items evicted**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **6** | 6 | 7 | 1 |
| store 1 | 7 | **6** | 7 | 1 |
| store 2 | 7 | 6 | **7** | 1 |
| ⭐ **store range (down the column)** | 1 | 0 | 0 | whole grid: 1 |

instrument range is **1.00x** the parent's store range (1 vs 1); diagonal range 1.  ⭐ The parent's ± is the larger of the two effects.

**NEGATIVE CONTROL bytes**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **5,735** | 5,714 | 5,782 | 68 |
| store 1 | 5,730 | **5,726** | 5,795 | 69 |
| store 2 | 5,694 | 5,720 | **5,783** | 89 |
| ⭐ **store range (down the column)** | 41 | 12 | 13 | whole grid: 101 |

instrument range is **2.17x** the parent's store range (89 vs 41); diagonal range 57.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-HOLE - field_geom**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **2,284** | 2,287 | 2,887 | 603 |
| store 1 | 2,490 | **2,030** | 2,271 | 460 |
| store 2 | 2,178 | 2,173 | **2,570** | 397 |
| ⭐ **store range (down the column)** | 312 | 257 | 616 | whole grid: 857 |

instrument range is **1.93x** the parent's store range (603 vs 312); diagonal range 540.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-HOLE - bitmap8**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **2,284** | 2,287 | 2,887 | 603 |
| store 1 | 2,490 | **2,030** | 2,674 | 644 |
| store 2 | 2,178 | 2,173 | **2,570** | 397 |
| ⭐ **store range (down the column)** | 312 | 257 | 317 | whole grid: 857 |

instrument range is **2.06x** the parent's store range (644 vs 312); diagonal range 540.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-HOLE - pair_presence**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **1,428** | 1,552 | 1,924 | 496 |
| store 1 | 1,584 | **1,330** | 1,812 | 482 |
| store 2 | 1,362 | 1,432 | **1,712** | 350 |
| ⭐ **store range (down the column)** | 222 | 222 | 212 | whole grid: 594 |

instrument range is **2.23x** the parent's store range (496 vs 222); diagonal range 382.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-HOLE - pair_deletion**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **2,284** | 2,287 | 2,887 | 603 |
| store 1 | 2,490 | **2,030** | 2,674 | 644 |
| store 2 | 2,178 | 2,173 | **2,570** | 397 |
| ⭐ **store range (down the column)** | 312 | 257 | 317 | whole grid: 857 |

instrument range is **2.06x** the parent's store range (644 vs 312); diagonal range 540.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-EVICTION - field_geom**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **1,970** | 1,646 | 2,655 | 1,009 |
| store 1 | 2,143 | **1,787** | 1,923 | 356 |
| store 2 | 1,781 | 1,788 | **1,876** | 95 |
| ⭐ **store range (down the column)** | 362 | 142 | 779 | whole grid: 1,009 |

instrument range is **2.79x** the parent's store range (1,009 vs 362); diagonal range 183.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-EVICTION - bitmap8**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **1,970** | 1,646 | 2,655 | 1,009 |
| store 1 | 2,143 | **1,787** | 2,186 | 399 |
| store 2 | 1,781 | 1,788 | **1,876** | 95 |
| ⭐ **store range (down the column)** | 362 | 142 | 779 | whole grid: 1,009 |

instrument range is **2.79x** the parent's store range (1,009 vs 362); diagonal range 183.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-EVICTION - pair_presence**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **1,458** | 1,262 | 2,044 | 782 |
| store 1 | 1,582 | **1,386** | 1,650 | 264 |
| store 2 | 1,260 | 1,395 | **1,438** | 178 |
| ⭐ **store range (down the column)** | 322 | 133 | 606 | whole grid: 784 |

instrument range is **2.43x** the parent's store range (782 vs 322); diagonal range 72.  ⛔ **The parent's ± understates this quantity.**

**NOVEL TUPLES post-EVICTION - pair_deletion**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **1,970** | 1,646 | 2,655 | 1,009 |
| store 1 | 2,143 | **1,787** | 2,186 | 399 |
| store 2 | 1,781 | 1,788 | **1,876** | 95 |
| ⭐ **store range (down the column)** | 362 | 142 | 779 | whole grid: 1,009 |

instrument range is **2.79x** the parent's store range (1,009 vs 362); diagonal range 183.  ⛔ **The parent's ± understates this quantity.**

**EVICTION bytes OUTSIDE permitted**

| store seed \ bridge seed | b=0 | b=1 | b=2 | ⭐ instrument range (across the row) |
|---|---|---|---|---|
| store 0 | **0** | 0 | 0 | 0 |
| store 1 | 0 | **0** | 0 | 0 |
| store 2 | 0 | 0 | **0** | 0 |
| ⭐ **store range (down the column)** | 0 | 0 | 0 | whole grid: 0 |

⭐ **bounded by construction: 0 in every direction.** No instrument can move it.

### ⭐⭐ THE ACCEPT-LEG AUDIT -- the half of the parent's claim a grid can falsify

The parent's claim has two halves: it refuses the deleted conjunction at 1.0000 **and still answers 1.0000** of the requests naming a never-held one. ⛔ A refusal rule that refuses everything gets the first half free, so the second half is where a claim can die. It held on all three of the parent's cells; here it is tested on nine.

| store seed \ bridge seed | b=0 | b=1 | b=2 |
|---|---|---|---|
| store 0 (accept never-held conjunction) | **1.0000** | **1.0000** | **1.0000** |
| store 1 (accept never-held conjunction) | **1.0000** | **1.0000** | ⛔ **0.8333** |
| store 2 (accept never-held conjunction) | **1.0000** | **1.0000** | **1.0000** |

⛔⛔ **The accept leg FAILS in 1 of 9 cells:** `store1_bridge2` (accept 0.8333). ⚠ The parent's 1.0000 accept leg is **not instrument-invariant**. It is a property of the (pool, encoder) pair, not of the field store as such, and with only one instrument the parent could not have seen this.

### ⭐ Internal consistency -- the diagonal, computed twice

The grid's diagonal cells and the separately-run `--bridge-mode joint` arm are the same (store, bridge) pairs reached by two different code paths (`--bridge-seed s` with `--seed s` versus `--bridge-mode joint`). ⛔ They must agree exactly or one of the two paths is not doing what its name says.

