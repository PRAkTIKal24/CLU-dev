# PREREG-w5-refuse-1 — rows 5 and 6 (and the row-1 accept leg), on the field store

**Hub** `w5 hub/refuse` · **target** `refusal` · **scored run 1 of 3**
**Written** 2026-09-20, BEFORE the run. ⛔ Not edited afterwards; amendments are appended and dated.
**Base** `base/iter5 = 6675904` · **branch** `w5/refuse` · **worktree** `/c/chlu/worktrees/CHLU-w5-refuse`
**Runner** `python -m chlu.experiments.exp_gen_refuse --stage queries --seed S --out /c/chlu/outputs/w5-refuse/runs/r1`

---

## 0. ⛔ The bar, and it is not the table

**Refusal is winnable at 8 bytes.** A per-factor value-presence bitmap plus a bracket check
scores `(1.0000, 1.0000)` on rows 1, 2, 5 and 6. That object is
`exp_gen_factor_wells.value_presence_bitmap` + `exp_gen_factor_wells.accept_rule`, frozen, and
this run **imports it verbatim** rather than reimplementing it. ⭐ Its pair is printed beside
every number this run produces. ⛔ Beating the prototype table here proves nothing and no
number from this run is offered as one.

## 1. What is measured

The four shipped query sets of the value-holdout condition, on the **test** condition,
**seeds 0, 1, 2**, cache **generation 2** (`C:/chlu/outputs/spoke-a-rivals/cache_generation2`;
the runner sets `CHLU_GEN_FEATURES_ROOT` and refuses generation 1).

| row | query set | n | what it is |
|---|---|---|---|
| `r1_seen` | `row1_seen` | 512 | ⭐ **the accept-leg** (AMENDMENT 1): on-grid, every value stored |
| `r2_unseen_valid` | `row2_unseen` | 512 | off-grid hue strictly between two stored grid-adjacent values |
| `r5_comp_refusal` | `row5_held_out` | 512 | ⭐ **on-grid, a value the store never held** |
| `r6_invalid_refusal` | `c_invalid` | 512 | `out_of_range` + `duplicate_factor` |

**Metric: the pair `(reject_invalid, accept_unseen_valid)`, never one number**, scored by the
frozen `gen_scoreboard.score_refusal_row`, whose accept-leg is **row 1** and which faults the
row to 0 unless threshold, setting fingerprint and `deployed_bytes` match between the two
halves. Scalar = the geometric mean, so refusing everything scores 0 and accepting everything
scores 0.

## 2. The refusers, all on the same requests at the same setting

| refuser | deployed bytes | rule |
|---|---|---|
| `field_geom` ⭐ PRIMARY | 524,288 (atoms) | `hole ∨ empty_basin ∨ out_of_range ∨ duplicate_factor` |
| `field_geom_gap` | 524,288 | the pinned four-mechanism rule, `+ in_gap` at `gap_tau = 4.0` |
| `bitmap8` ⭐ **THE BAR** | **8** | the frozen 8-byte rule, imported verbatim |
| `pair_presence` | 166 (packed) | the joint pair-presence table, 1,322 cells |
| `pair_deletion` | 8 + 166 | the 8-byte rule + a separate pair-deletion table (no deletions in this run) |
| `grid_syntax` | 0 | the frozen zero-byte no-model laundering rule |
| `field_landscape_removed` | 524,288 | ⛔ charter §C.2 — atoms zeroed, must collapse |
| `field_landscape_shuffled` | 524,288 | ⛔ charter §C.2 — atom coordinates permuted |
| `refuse_everything` / `accept_everything` | 0 | the degenerate guards; both must score **0** |

## 3. ⛔ Thresholds, written down HERE, before any invalid row is scored

`basin_live_tau = 2.0` and `gap_tau = 4.0`. Both were **calibrated on the `val` condition by
the w4 field-store hub**, are pinned in `exp_gen_field_store.PINS`, and existed before this hub
did. ⛔ Neither is moved. Both are in units of a **within-value spread** (`basin_sigma`,
`well_sigma`) — ⛔ **not** the spacing of stored items, which is the length scale the admission
gate died on.

`basin_live_tau` sits on a **measured plateau**: at tau ∈ {2, 4, 8} the set of basins with zero
live atoms is *exactly* the two never-held values, on all three seeds (hub LOG, probe 2). At
tau = 1.0 a retained term joins it. A parameter-free radius from basin spacing (`basin_gap/2`,
`basin_gap`) was tried and **rejected on evidence**: it empties 56 of 57 terms (probe 3).

**Dropped before scoring, on val evidence (hub LOG 2026-09-20T00:22Z):** `in_gap` is not in the
primary rule. It fires on 0.6–1.2 % of the store's own pool requests and on 0.0000 of unseen
two-way requests at this tau — pure accept-leg cost on this encoder. It is scored **beside** the
primary rule as `field_geom_gap`, never instead of it.

## 4. Expected direction — declared in advance, including against myself

1. ⭐ **I expect `field_geom` to TIE `bitmap8` at `(1.0000, 1.0000)` on rows 5 and 6.** Both
   read the same fact — which factor values the pool holds — one as 8 bytes of bitmap and one as
   live atoms in a 524,288 B landscape. **If that is what happens, the dial is falsified on these
   rows and I will say so in the target file's own words: _the 8-byte rule matches me, so there
   is no claim on rows 5 and 6._**
2. I expect `field_geom_gap` to score **below** `bitmap8` on the accept leg (≈0.988–0.994), i.e.
   the pinned rule **loses** to 8 bytes. That is a prediction against my own object.
3. `r6` is `evidential: false` in `gen_value_holdout.ROW_EVIDENCE_RANKING` and can never be
   anything else. Reported as the degenerate-arm guard, ⛔ never as evidence.
4. `pair_presence` should score **0** on row 1/row 2 acceptance (it refuses unseen conjunctions),
   hence 0 on the pair.
5. `field_landscape_removed` should refuse everything → pair `(1.0, 0.0)` → **0**.

## 5. ⛔ What would prove this run wrong / vacuous

- The threshold accepts row 2 **and** accepts rows 5/6 ⇒ vacuous, exactly as the admission gate
  was. Checked directly: row 2 acceptance and row 5 acceptance are both reported.
- `field_geom` refuses row 1 at any appreciable rate ⇒ the accept leg collapses and the pair
  goes with it.
- Any length scale traceable to inter-item spacing ⇒ void. §3 states what each is derived from.

## 6. Population footnote, binding

Every rival in `RIVALS-READY.json` is fitted on the `montero_combination` pool, which **contains
both held-out values** (3,157 rows at `object_hue` 0.5; 7,972 at `shape` 0.0). On rows 5 and 7 a
rival saw the value the field store never did. ⛔ Their mechanism can enter the row; their fit
cannot. Rival numbers carry that footnote and ⛔ are not counted as beating a value-holdout arm —
and the field store is not counted as beating them.

## 7. Labels

Labels enter **once at fit time**, to align wells and to build the term→coordinate map (the
address book), and at the interface. ⛔ Never in the landscape, the read or the write. This
sentence goes in every table caption.

## 8. Cost

Local CPU, 3 threads. No GPU. Estimated ≤ 10 min/seed.
