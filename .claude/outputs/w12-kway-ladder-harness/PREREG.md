# PRE-REGISTRATION — w12 spoke/kway-ladder-harness
**Written 2026-09-21T11:26Z, BEFORE the k-way builder was executed once.**
No capacity count, no corner count and no disjointness result existed on disk when this file
was written; the only thing read beforehand was source code (`chlu/data/gen_value_holdout.py`,
`chlu/data/gen_splits.py`, `chlu/experiments/exp_gen_factor_wells.py`). Charter A.1.

Scored runs in this spoke: **ZERO, by charter.** This spoke builds the instrument; the ladder is
run by `hub/composition-ladder`.

---

## 1. What is being built

A `kway-*` family for **k = 3, 4, 5** extending the `value_holdout` condition: **k of the six
Shapes3D factors jointly pinned to a held-out box, the other 6 − k free.** The existing ends are
**k = 2 = row 3** (`pair-*`) and **k = 6 = row 4** (`nway-*`).

**The two-layer object, fixed here so it cannot be redefined after the counts are seen:**

| layer | what it is | row 3 | row 4 |
|---|---|---|---|
| **cell** | what is removed from the store's fit pool (`FactorCell`; absent factor ⇒ unconstrained) | 3 `pair-*` cells | 49 `nway-*` cells |
| **corner** | a Montero-shaped probe box **inside** a cell — the unit of the paired estimand | 60 tiled sub-corners | 49 (cell ≡ corner) |

**The bar below is on CORNERS, not cells.**

## 2. ⭐ THE BAR — declared now, not negotiable afterwards

**B1. ≥ 49 pairwise-disjoint corners for each of k = 3, 4, 5.**
*Pairwise-disjoint* means: for the family's corner list, the concatenation of every corner's
dataset row-index array contains **zero** repeated row — `np.unique(concat).size == concat.size`,
the identical test `exp_gen_factor_wells.assert_corners_disjoint_and_heldout` applies to rows 3
and 4. Structural disjointness (a product of pairwise-disjoint coordinate option sets) is the
*argument*; the row-level test is the *proof*, and the proof is what is reported.

**B2. Every corner row lies inside the family's own hold-out**, i.e. in the test index of the
split built from that family's cells — `np.setdiff1d(corner_rows, test_rows).size == 0`.
Same test as rows 3–4.

**B3. Same estimand as rows 3–4**: `exp_gen_factor_wells.paired_estimand` — unit = **corner**,
statistic = **mean paired per-corner difference** arm − control, with paired bootstrap CI95
(10 000 resamples, `rng = default_rng(20_260_919)`) and a two-sided sign test.
Metric for rows 3–4 and therefore for every ladder rung:
`distinct_valid_novel_tuples`, **margin `DECORATION_MARGIN_NOVELTY = +1.0`**
(accuracy rows carry `DECORATION_MARGIN_ACCURACY = 0.02`; reported, not claimed).
⛔ The ladder introduces **no new metric and no new margin.**

**B4. Same corner geometry as rows 3–4** — the widths the n-way builder uses:
`hue_band = 2`, `scale_corner = 3`, `orientation_corner = 3`, `shape = 1`, and
`wall_hue` band **tied to** `floor_hue` band (both rows 3 and 4 do this).

## 3. ⭐ THE TWO END-REPRODUCTION ASSERTIONS (run before the middle is reported)

**E1 (k = 6 ≡ row 4).** `build_kway_cells(grid, cfg, k=6, pair_cells=...)` must return, cell for
cell and **in the same order**, the same `values` dicts as
`gen_value_holdout.build_nway_cells(grid, cfg, pair_cells=...)`.

**E2 (k = 2 ≡ row 3).** The k=2 corner list must equal, corner for corner and in the same order,
`exp_gen_factor_wells.pair_subcorners` on the shipped `pair-*` cells — same `corner_id`s, same row
sets.

⛔ **If E2 fails, the builder is wrong, not row 3.** E1/E2 failure ⇒ STOP and report; every ladder
point would be off-instrument.

## 4. ⛔ NON-NEGOTIABLE, restated so it cannot drift

`scale` and `orientation` remain **INELIGIBLE as VALUE hold-outs** (they are physical defaults —
`gen_value_holdout` §2). They are **eligible as pinned axes inside a k-way box**, exactly as in
row 4, and nowhere else. The held-out hue (`object_hue = 0.5`), the held-out shape (`shape = 0.0`)
and the three hues reserved by the `pair-*` cells stay excluded from every k-way box, exactly as
`build_nway_cells` excludes them.

## 5. ⛔ KILL CONDITIONS — the check is written and run BEFORE the ladder

`ladder_capacity_report()` runs first and can kill the ladder. A rung is **DECLARED NOT-RUN with
the count reached** — ⛔ never padded with overlapping corners, ⛔ never reported as zero — if:

- **K1** its maximum number of pairwise-disjoint corners is < 49;
- **K2** the row-level disjointness proof (B1) fails;
- **K3** any corner row is outside the family's hold-out (B2);
- **K4** the k=6 or k=2 end-reproduction assertion (E1/E2) fails — in which case **the whole
  ladder** is NOT-RUN, not just the rung.

## 6. Declared in advance as a REPORTED CONFOUND, not a result

A lower-k cell pins fewer factors, therefore covers **more** grid rows, therefore removes **more**
rows from the store's fit pool. The ladder therefore confounds *composition depth* with
*held-out fraction*. This spoke will report, per rung, the exact held-out row count and fraction,
and will emit each rung at the **smallest cell budget that still clears 49 corners** (the same
philosophy as `n_nway_cells = 49`, which "meets it exactly") to minimise the spread.
⛔ This spoke will not choose a correction for the hub; it reports the number and names the
control that would remove it.

## 7. Deliverables

`LADDER-READY.json` (per family: name, k, pinned axes, free axes, cell count, corner count,
disjointness check result, hold-out row budget, PASS/NOT-RUN against B1), `FINDINGS.md`, `LOG.md`,
`END_UTC.txt`, plus the builder + its tests committed to `w12/kway-ladder-harness`.
