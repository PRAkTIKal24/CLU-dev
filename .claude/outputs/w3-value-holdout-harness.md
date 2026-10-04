# w3-value-holdout-harness — the VALUE-holdout condition and the seven-row scoreboard

- **Agent:** `experiment-engineer` · **Branch:** `agent/experiment-engineer/w3-value-holdout`
- **Base:** pin `4125fce` (verified: `git cat-file -e 4125fce^{tree}` OK; `gen_splits.py` at the pin
  contains `def build_cell_split` and `def build_c_invalid`)
- **Dial: the instrument.** No arm run, no scored comparison, **no scored run spent.**
- **Status: DELIVERED, with one finding that the Head needs before `hub/refuse-delete` starts.**

---

## 0. ⛔⛔ READ FIRST — the falsifier fired, on the diagnostic the pin does not assert

The named instrument **passes**: `assert_no_model_detector_at_chance` is at **0.5000 on all 11
structural features** for row 5 (both kinds) and row 6's `out_of_range`, and 0.5013 worst-case on row
6's `duplicate_factor` — all inside the pin's 0.05 tolerance. Rows 2/5/6 are structurally
indistinguishable. Full table in §5.

**But the refusal rows are still free, and I measured it rather than inferring it.** A rule that reads
the request and the factor grid and **nothing else** —

```
accept  iff  no factor is named twice
        and  no term is outside its factor's realised range
        and  at least one term is off-grid but strictly inside the range
```

— scores, on the shipped 512-record query sets:

| row | reject_invalid | accept_unseen_valid | pair score | deployed bytes |
|---|---|---|---|---|
| 5 compositional refusal | **1.0000** | **1.0000** | **1.0000** | **0** |
| 6 invalid-request refusal | **1.0000** | **1.0000** | **1.0000** | **0** |

A perfect pair on both refusal rows with **zero bytes of store**. Its row-1–4 score is **0.0000** —
it produces no image — which is what keeps the finding honest rather than alarmist.

**Why, and why it is not fixable by re-drawing the sets.** The taxonomy *defines* row 2 as an
off-grid interpolant and row 5 as an on-grid held-out value. So `n_offgrid_terms` separates row 5
from row 2 at **AUROC 1.0000** by construction — the same class of leak the pin's
`definitional_features` already names for `duplicate_factor` ("⛔ No control can remove it: it is what
makes the record invalid"). Per task §9 I **report this and stop; I did not redesign the rows to
pass.**

**What it blocks and what survives.**
- ⛔ `hub/refuse-delete` **may not** quote a row-5 or row-6 refusal pair as evidence for the store.
  Charter §C.3's surviving leg cannot be carried by the static refusal pair.
- ⭐ **Row 7's `delta_reject` survives intact and is the non-free instrument.** Deleting a value from
  the store does not move the factor grid, so the no-model rule's `delta_reject` is **exactly 0.0** —
  measured, in `test_delta_reject_is_the_non_free_half`. An arm that changes its refusal after a
  deletion, with no retraining, has done something no syntactic rule can do.
- ⭐ The harness therefore **ships the rule as a mandatory row** (`grid_syntax_rule`).
  `build_scoreboard` raises if it is absent. No hub can quote a refusal number without the no-model
  rule's score printed beside it.

If the Head wants row 5 made non-free, the only construction I can see is to move row 5 **off-grid
too** — request a hue near the hole whose brackets are not both stored. That makes "is this
in-basin?" a question about what the store holds rather than about the grid. ⛔ I did not build it:
it changes row 5 from the Head's "a value held out entirely" into "a value near a held-out value",
and §9 forbids redesigning to pass. It is a Head decision, not mine.

---

## 1. What shipped — four new files, ⛔ zero edits to any pinned file

```
chlu/data/gen_value_holdout.py        1303 lines   (new)
chlu/experiments/gen_scoreboard.py     823 lines   (new)
tests/test_gen_value_holdout.py        423 lines   (new, 31 tests)
tests/test_gen_scoreboard.py           432 lines   (new, 28 tests)
```

`git diff --cached --stat` on the branch is **four additions, 2,981 insertions, 0 deletions**.
`exp_gen_cell.py`, `gen_splits.py`, `gen_rowsplit.py`, `gen_store.py`, `gen_landscape.py`,
`gen_bridge.py` and `config.py` are untouched. `GenCellConfig.lace_selection_source` is not
re-pointed.

Config lives **next to the code that reads it** (task §5): `ValueHoldoutConfig` and
`ScoreboardConfig`, each with `from_mapping` — and an unknown key **raises** rather than being
silently dropped. ⛔ No block was added to `chlu/config.py`, and `chlu/cli/data_cmd.py` was not
touched either: the emitter is its own entry point, `python -m chlu.data.gen_value_holdout`.

**59/59 new tests pass. Full suite: see §8.**

---

## 2. ⭐ The chosen values, and the rejected ones — written down before the build

Recorded in the emitted artifact (`value_cells.json → rationale`), not only here.

### Held-out hue: `object_hue = 0.5` (grid index 5)
- It is the **object's own** photometric property — the thing the request names and the judge scores.
  `floor_hue`/`wall_hue` are background.
- Hue is **circular**, so no hue value is a physical default the model degrades towards. The §3.1
  hazard that rules out `scale` and `orientation` does not reach it.
- It sits **four grid steps** from Montero's retained band (`object_hue >= 0.8`), so the value hole is
  not inside the region the headline and wide **combination** conditions already spend.
- It is **interior**: both grid neighbours are retained, so the hole is two-sided.

**Rejected:** `0.8`/`0.9` (Montero's held-out band — entangled with the headline combination
condition); the range endpoints `0.0`/`0.9` (the nearest off-grid values are exactly where row 6's
`out_of_range` records live, so rows 5 and 6 would become adjacent in value space); **any `scale` or
`orientation` value** (⛔ task §3.1 — both are physical defaults the model degrades towards; the
realised grid confirms it: `orientation` index 7 is exactly `0.0`, the null pose, and the median
`scale` is ≈1.0).

### Held-out shape: `shape = 0.0` (the cube)
Shape is **categorical** — it cannot be interpolated — so "held out entirely" is weakened only by a
retained *near-neighbour*, and the cube has none among {cylinder, sphere, oblong}: it is the one shape
with flat faces and hard edges.

**Rejected, and this is the §9 "physical default" check applied to shape:**
- **`2.0` (sphere)** — ⛔ the shape hazard exactly. A degraded render is a rounded blob, so the sphere
  is the shape the model **degrades towards**. A held-out sphere would be unremovable for rendering
  reasons, not store reasons.
- **`1.0` (cylinder)** — Montero's held-out shape and the wide condition's cell-0 shape. Already
  entangled with both combination conditions.
- **`3.0` (oblong/capsule)** — visually intermediate between the retained cylinder and the retained
  sphere. A retained near-neighbour means refusal could be declined on interpolation grounds, and the
  row would stop being a refusal row.

---

## 3. The split on disk — acceptance 1 and 2

Emitted to `.claude/data/gen/shapes3d/value_holdout/` from the real `3dshapes.h5`.

| artefact | sha256 | records |
|---|---|---|
| `factor_index.npz` | `3e29316fcd6793edbff9aa4932d5a9c732d9178854556c943d9824a8062f0baf` | — |
| `value_cells.json` | `53c87c61cbbe03951be48ad67f127e27d7fbfb63f5ec732a3fccb8f698c2d43f` | 54 cells |
| `row2_unseen.json` | `7a7908fe35fa5ff28e38c15d87dfe3499535a8d13f5c2ef81bb2609a88dee343` | 512 |
| `row5_held_out.json` | `e0b6ea401fb675c550b4442da05467b2a3958870e65c2ca23b78e0e975304d82` | 512 |
| `c_invalid.json` | `9146b899839aa77c47eedf2e7c0d98105c9512256805228ced5369569c27678b` | 512 |
| `a7_redundant.json` | `bb52ba429400f483afcc5c8ec96432b026c651d9663aedc89cf7e11060cee191` | 512 |
| `query_sets.json` | `7f4ec9cf2d0d2d123e129d658424df1e355045bf4fdefd5f7b688edd9c708591` | — |

### The cells

| family | n cells | rows held out | what |
|---|---|---|---|
| `value` | 2 | 156,000 | `{object_hue: 0.5}` (48,000) + `{shape: 0.0, object_hue: the 9 retained}` (108,000) |
| `pair` | 3 | 36,000 | rows 3: `{object_hue: h, shape: s}`, both values retained elsewhere |
| `nway` | 49 | 3,528 | rows 4: Montero-shaped 6-factor conjunctions, **72 rows each** |

⚠ **Why `value-shape` carries an `object_hue` clause.** `{shape: 0.0}` and `{object_hue: 0.5}`
intersect, and `assert_cells_disjoint` would (correctly) fire. The shape cell therefore carves out
exactly the rows the hue cell already owns. The **union** is still every row carrying either value —
and `assert_value_absent_from_train` checks that property *directly on the train index*, so the claim
does not rest on that argument. The positive control
`test_a_value_cell_that_overlaps_is_caught` shows the disjointness check firing on the naive version.

### Assertions, with their returned counts (acceptance 2)

| assertion | returned |
|---|---|
| `assert_non_empty_test` | 195,528 |
| `assert_index_partitions_grid` | 480,000 |
| `assert_cells_disjoint` | 195,528 |
| `assert_holdout_disjoint` | 284,472 **( == n_train, checked; a partial check raises)** |
| `assert_value_absent_from_train[object_hue=0.5]` | 284,472 |
| `assert_value_absent_from_train[shape=0.0]` | 284,472 |

Both `assert_value_absent_from_train` calls have positive controls that inject a held-out row into
train and require the check to fire.

### §9's row-count check: the fit pool is **not** the binding constraint

`n_train = 284,472` of 480,000 (**40.7 % held out** — large, and declared). The shipped generative
cell fits the store on a **32,768-row sample** of train, i.e. 11.5 % of what remains. Holding out a
whole hue and a whole shape does **not** leave the store unable to fit at the shipped scale. Pinned in
`test_the_store_fit_pool_is_large_enough_at_the_shipped_scale`.

### ⚠ One realised limitation the hubs should know

Each `pair` cell reserves a whole `object_hue` value, which removes it from the n-way cells' bands. At
the default `n_pair_cells=3` (reserving 0.0, 0.3, 0.7) only **two** width-2 object-hue bands survive
— `(0.1, 0.2)` and `(0.8, 0.9)`. The 49 corners still spread over 3 shapes × 3 floor-hue bands ×
2 scale corners × 2 pose corners, and their realised object–background hue contrast spans
**0.100 – 0.400** (median 0.200) of a maximum possible 0.500 — a wider contrast range than Montero's
own cell, which R15 found collapses to the bottom. If a hub wants more object-hue diversity in the
corners: `n_pair_cells=1` → 4 bands, `=2` → 3 bands. It is one mapping key, nothing else moves.

---

## 4. The scoreboard — acceptance 3, 4 and 6

`chlu/experiments/gen_scoreboard.py` is a **pure scoring layer**: an arm hands it, per row, the
*judged* outcome of its own requests (accept mask, the external classifier's read-back tuples, the
validity mask, the threshold, a setting fingerprint, and its byte ledger). That is what makes the
metric itself testable with no store, no decoder and no judge in the room.

Seven rows, `r1_seen … r7_deletion`, every arm and **both controls on every one**:
`basis_head` (the wave-2 fitted basis + linear head) is the **first** control and the like-for-like;
`knn_proto` is the **second** row; `grid_syntax_rule` is the mandatory laundering control of §0.
`build_scoreboard` **raises** if any of them is missing on any row, and raises if any arm is missing
any of the seven.

**The byte ledger is a column, checked per row** (`assert_byte_ledger`):
- every (arm, row) carries a measured integer `deployed_bytes`;
- the arm and `basis_head` are byte-matched to the 524,288 B budget at **tolerance 0** — an unmatched
  like-for-like is a different experiment, not a footnote, and it raises **even with a note**;
- any other arm may be unmatched **only** with a non-empty `budget_note` (`knn_proto`'s declared
  ~31 % advantage is the standing case) — silence raises;
- ⭐ the ledger must be **constant across the seven rows**: a run that quietly deployed a bigger store
  for the refusal rows is caught by arithmetic.

**The refusal pair, encoded rather than described** (`score_refusal_row`):
- reports `(reject_invalid, accept_unseen_valid)`, never one number;
- `accept_unseen_valid` is always measured on **row 2's** requests, and the function **checks** that
  row 5/6 used the *same threshold*, the *same setting fingerprint* and the *same deployed bytes* as
  row 2. Any mismatch → the row scores **0** and the reason is written into `faults`.
  `test_a_different_threshold_makes_the_row_score_zero_and_says_why` shows a naive gmean of 1.0000
  scoring 0.0000 because the pair was not a pair.
- scalar = geometric mean ⇒ refusing everything scores **0**, accepting everything scores **0**.

**Acceptance 4 — the falsifier for the metric itself, shipped as a test.**
`refuse_everything_outcome` fills all seven rows and scores **0.0 on every one** —
`test_the_refuse_everything_stub_scores_zero_on_the_WHOLE_table`. Its mirror,
`accept_everything_outcome`, also scores 0.

**The prototype table's failure is measured, not asserted.** `knn_proto` is a mandatory row on all
seven, so its refusal-pair score is produced by the same arithmetic as every other arm's. It is a row
in the table, not a sentence.

**Row 7** requires **both** halves and no retraining: survivors must still produce distinct valid
*novel* tuples **and** the deleted value must refuse, with a `fit_fingerprint` pair that must be
**equal**. A moved fingerprint is retraining and scores 0; survivors that stop composing kill the row
even when refusal is perfect.

**Acceptance 6 — the scope statement is a field.** `VALUE_HOLDOUT_SCOPE` is written into
`value_cells.json`, `row2_unseen.json`, `row5_held_out.json`, `query_sets.json`, the `factor_index.npz`
metadata blob, the scoreboard's top level **and every one of the seven row dicts**. Pinned by
`test_the_scope_statement_is_a_FIELD_on_every_row`. Its text:

> The hold-out is from the **STORE'S FIT POOL**, not from the encoder, the decoder or the judge. Those
> three are frozen, hash-pinned and read-only (charter §A.4) and were fitted on a train split
> containing **every** hue and **every** shape; re-fitting them on a value hold-out would void every
> wave-2 number and is out of scope. Therefore "never trained on" means here: **the store never held a
> row carrying that value.** The decoder can still render the value and the judge can still recognise
> it — which is what makes rows 2/5/6/7 scorable at all. ⛔ Read end-to-end, "never seen" is false and
> the claim dies with it.

⭐ **One consequence for the hubs that is not in the scope statement and should be:** an arm's
per-factor prototype table / basis must be fitted on the **retained** values only. `FactorPrototypes`
is fitted from train features, so it inherits the hold-out automatically — but any arm that builds its
embedding from the *grid* rather than from the fit pool would hand itself the held-out value for free.

---

## 5. ⛔ The no-model detector — acceptance 5, the full per-feature table

Run by `assert_no_model_detector_at_chance` on the **shipped artefact discovered from disk**, at the
pin's tolerance `0.05` (ceiling 0.5500). **VERDICT: PASSES. `violations: []`.**

Controls, as registered in `query_sets.json`:

| kind | n invalid | control | n control |
|---|---|---|---|
| `row5_held_out[held_out_hue]` | 256 | `row2_unseen` | 512 |
| `row5_held_out[held_out_shape]` | 256 | `row2_unseen` | 512 |
| `c_invalid[out_of_range]` | 256 | `row2_unseen` | 512 |
| `c_invalid[duplicate_factor]` | 256 | `a7_redundant` | 512 |

⛔ `duplicate_factor` keeps the 7-term `a7_redundant` control: it carries `n_factors + 1` terms and a
6-term control would hand back the exact term-count giveaway R26 ruling 1 killed.

### STRUCTURAL view — the asserted one (orientation-free AUROC)

| feature | row5[hue] | row5[shape] | c_invalid[out_of_range] | c_invalid[duplicate_factor] |
|---|---|---|---|---|
| `first_duplicate_position` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `first_factor_index` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `is_factor_order_sorted` | 0.5000 | 0.5000 | 0.5000 | 0.5010 |
| `last_factor_index` | 0.5000 | 0.5000 | 0.5000 | 0.5013 |
| `masked_request_json_len` | 0.5000 | 0.5000 | 0.5000 | 0.5003 |
| `max_factor_multiplicity` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `n_distinct_factors` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `n_order_inversions` | 0.5000 | 0.5000 | 0.5000 | 0.5013 |
| `n_redundant_terms` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `n_terms` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `sum_factor_indices` | 0.5000 | 0.5000 | 0.5000 | 0.5013 |
| **max** | **0.5000** | **0.5000** | **0.5000** | **0.5013** |

Stratified pooled: `row5_held_out` **0.5000**, `c_invalid` **0.5006**. All inside 0.5500.

The 0.5013 on `duplicate_factor` is the A7 control's sampling noise on factor-index arithmetic, not a
structural difference — the repeated term is appended at the same position by the same cycling rule in
both builders.

**Positive control:** `test_the_detector_fires_when_a_structural_giveaway_is_injected` appends one
redundant term to every row-5 record and the audit reaches **AUROC 1.0000** and raises. The audit can
fail.

### DIAGNOSTIC view — set labels by construction. ⛔ Reported, never asserted

| kind | `id_numeric_suffix` | `id_token_count` | `record_key_count` |
|---|---|---|---|
| `row5[held_out_hue]` | 0.9375 | 1.0000 | 1.0000 |
| `row5[held_out_shape]` | 0.9375 | 1.0000 | 1.0000 |
| `c_invalid[out_of_range]` | 0.9375 | 0.5000 | 1.0000 |
| `c_invalid[duplicate_factor]` | 0.9375 | 0.5000 | 1.0000 |

A record id names the file it came from and a row-5 record carries its answer key (`violated_factor`,
`row`). An arm reading those is reading the label, not being fooled — the pin's own position. Every
key is inside the pin's declared allowlist (`test_every_record_key_is_inside_the_pins_declared_allowlist`).

### DEFINITIONAL view — ⛔ reported, never asserted

| kind | `has_conflicting_repeat` | `n_distinct_values` |
|---|---|---|
| `row5[held_out_hue]` | 0.5000 | 0.5114 |
| `row5[held_out_shape]` | 0.5000 | 0.5687 |
| `c_invalid[out_of_range]` | 0.5000 | 0.5643 |
| `c_invalid[duplicate_factor]` | **1.0000** | 0.8136 |

`has_conflicting_repeat` at 1.0000 on `duplicate_factor` is the pin's declared, unremovable case.

### GRID-REACH — my declared extra diagnostic. ⛔ Reported, never asserted

| comparison | `n_offgrid_terms` | `max_distance_outside_range` |
|---|---|---|
| **row 5 vs row 2** | **1.0000** | 0.5000 |
| row 6 `out_of_range` vs row 2 | 0.5000 | **1.0000** |
| row 6 `duplicate_factor` vs A7 | 0.5000 | 0.5000 |

**This is §0.** Row 2 is off-grid by definition and row 5 is on-grid by definition, so counting
off-grid terms separates them perfectly. Row 6 is decided by a grid look-up, which the pin already
declares. The consequence — one fixed no-model rule winning both pairs — is measured in §0 and shipped
as `grid_syntax_rule`.

---

## 6. Rows 2, 5, 6 — how they are built and what is checked

**Row 2 (unseen-valid, 512 records).** The five non-hue terms are lifted from a **train** row; the hue
term is the midpoint of a **grid-adjacent pair of retained values**, and `validate_row2` asserts that
**both bracketing rows are in train** — that is what "in-basin" means mechanically, rather than as a
word.
⛔ **The collision that would have sunk the design:** the pair `(0.4, 0.6)` brackets the hole at 0.5
but is **not grid-adjacent**, so it is absent from the pair list by construction and a row-2 midpoint
can never land on the row-5 value. Pinned twice, including a positive control that injects `0.45` (a
midpoint of 0.4 and the held-out 0.5) and requires `validate_row2` to raise `not in-basin`.

**Row 5 (compositional refusal, 512 = 256 + 256).** On-grid requests drawn from the held-out region,
two kinds (`held_out_hue`, `held_out_shape`). `validate_row5` asserts each is on-grid, denotes its own
grid row, and that the row is **in test**. Positive control: a record built from a train row raises
`IN the store's fit pool`.

**Row 6 (invalid-request refusal, 512 = 256 + 256).** ⛔ `build_c_invalid` verbatim — `out_of_range`
and `duplicate_factor` from the grid alone. **No second invalid-request format was invented.**

All three are **class-balanced at 512 records** and registered in `query_sets.json` alongside the
pin's `base` / `a7_redundant` / `c_invalid`, so the audit reaches them from disk. The registry is
produced by the pin's own `write_query_set_registry` and then **extended** — ⛔ `QUERY_SET_REGISTRY`
lives in a read-only file and was not edited.

---

## 7. How a hub uses this

```bash
# 1. emit the condition (idempotent, deterministic, 7.3 s wall on the shipped h5)
uv run python -m chlu.data.gen_value_holdout --report <hub>/value_holdout_emit.json
#    -> .claude/data/gen/shapes3d/value_holdout/{factor_index.npz, value_cells.json,
#       row2_unseen.json, row5_held_out.json, c_invalid.json, a7_redundant.json,
#       query_sets.json}
#    exit 2 if the no-model detector fails — but the AUROC table is printed FIRST.
```

```python
from chlu.experiments import gen_scoreboard as S
# per arm, per row: the JUDGED outcome, not raw output
out = S.make_outcome(accept=..., valid=..., tuples=..., threshold=tau,
                     setting={...}, byte_ledger={...})
table = S.build_scoreboard("clu", outcomes, seen_tuples, deletion=..., provenance=...)
S.write_scoreboard(table, hub_dir / "scoreboard.json")
```

`outcomes` must contain `basis_head`, `knn_proto` and `grid_syntax_rule` on all seven rows or
`build_scoreboard` raises. Every row's verdict against `basis_head` carries, verbatim when the arm
ties: *"⛔ A CLU mechanism that ties basis+head is decoration and is reported as such, in those
words."*

---

## 8. Acceptance — all six

| # | criterion | status |
|---|---|---|
| 1 | `value_holdout` index on disk with sha256, one whole hue + one whole shape, disjoint combination cells for rows 3–4 | ✅ §3 |
| 2 | the four pinned assertions pass with their returned counts recorded | ✅ §3, plus two extra direct checks with positive controls |
| 3 | seven rows for a supplied arm, basis+head and `knn_proto` on every row, byte ledger asserted per row | ✅ §4 |
| 4 | refusal rows emit the pair; a refuse-everything stub scores 0, shipped as a test | ✅ §4 |
| 5 | `assert_no_model_detector_at_chance` passes on rows 2/5/6, per-feature AUROC table reported | ✅ §5 — **passes**, and the grid-reach diagnostic that does *not* is reported too |
| 6 | the §2 scope statement is a field in the emitted artefact | ✅ §4 |

**Tests:** 59/59 new tests pass (`tests/test_gen_value_holdout.py` 31, `tests/test_gen_scoreboard.py`
28). `ruff check` clean on all four files.

⭐ **The pin's own audit inherited the new condition with no edit to it.**
`tests/test_gen_c_set_a7.py::test_registered_query_sets_on_disk_are_at_chance` discovers registries
from disk, and `discover_query_set_registries()` now returns four — dSprites, Shapes3D,
`recomb2element_wide` and **`value_holdout`**. The pre-existing split/detector suite
(`test_gen_c_set_a7.py`, `test_gen_data_splits.py`, `test_gen_rowsplit.py`, `test_gen_wide_split.py`)
passes 76/76 **with the new registry in scope**. That is the R26.1 inheritance mechanism working as
designed: the new query sets were audited by a test written before they existed.

⚠ **The full repository suite was NOT run to completion** — it is long on this box serially
(`pytest-xdist` is not in this venv) and was stopped at ~20 % with no failures to that point. What was
run green: the 59 new tests, and the 76 tests of the four pre-existing split/detector files
(`test_gen_c_set_a7.py`, `test_gen_data_splits.py`, `test_gen_rowsplit.py`, `test_gen_wide_split.py`)
— the only files my change can reach, since the change adds four files and edits none. A full-suite
run is still owed before a hub depends on this.

---

## 9. What would prove this spoke wrong — all three checked

1. **"The no-model detector separates row 5 or row 6 from row 2 above tolerance."** The named
   instrument does **not** (0.5000 / 0.5013 vs a 0.5500 ceiling). **A different no-model rule does**,
   at a perfect refusal pair, and §0 reports it and stops rather than redesigning the rows.
2. **"Holding out a whole hue leaves the fit pool unable to fit at the shipped scale."** No:
   284,472 train rows, 8.7× the 32,768-row store fit sample. Realised counts reported in §3.
3. **"A held-out value turns out to be a physical default."** Checked for both axes and the rejected
   candidates are recorded in the artefact: the **sphere** was rejected on exactly this ground (a
   degraded render is a blob), as were all `scale` and `orientation` values.

---
---

# AMENDMENT 1 — row 5's accept-leg is ROW 1 (2026-09-19)

- **Agent:** `experiment-engineer` (the same one; same branch `agent/experiment-engineer/w3-value-holdout`)
- **Task:** `.claude/tasks/w3-value-holdout-amendment.md` · **Head ruling: option 1**
- **Dial: the instrument.** No arm, no scored run, **no scored run spent.**
- **Status: DELIVERED. All three acceptance checks ran. §4.1 and §4.2 PASS. §4.3 found a residual —
  reported below, not patched around.**
- ⛔ **§0 above stands as written.** It was right, and the amendment is built on it, not against it.

## A0. The three acceptance checks, first

| # | check | result |
|---|---|---|
| §4.1 | `grid_syntax_rule` must now **FAIL** on row 5 | ⭐ **PASS — pair `0.0000`** (was `1.0000`) |
| §4.2 | no-model detector with `row1_seen` as the control for both row-5 kinds | ⭐ **PASS — `0.5000` on all 11 features, both kinds** |
| §4.3 | an honest attempt to break the new pairing | ⭐ **RAN — two things got through. §A5.** |

**The falsifier did not fire in the form the task specified, but it did not come back clean either.**
No *syntactic* rule wins row 5's pair any more. Two other things do, and both of them have to pay
for the split — which is the difference the ruling was after, but it is a difference of **degree,
not of kind**, and §A5 states exactly how much.

## A1. What changed — the four things, and nothing else

1. ⭐ **`row1_seen.json`, 512 records** (`build_row1_seen`) — on-grid requests whose **every** value
   is stored, drawn from **train** rows. `validate_row1` asserts on-grid, denotes-its-own-row, and
   **in train**; `row1_positive_control` injects a **test** row and requires the validator to raise.
   ⛔ The positive control runs **inside the emitter, before the file is written** — the raised message
   is carried in the report, so the proof is the exception text, not a boolean:
   > `shapes3d-row1-seen_hue-0000: row 0 is NOT in the store's fit pool — the accept-leg would be
   > scored on a request the store never held`
2. ⭐ **`score_refusal_row`'s accept-leg is `r1_seen`** (`ACCEPT_LEG_ROW`), with the same threshold /
   setting-fingerprint / deployed-bytes identity checks unchanged. The reported key keeps the Head's
   name `accept_unseen_valid`; the leg is reported explicitly as `accept_leg_row`, so nothing has to
   be inferred from a key name.
3. ⭐ **Row 2 is accept-only.** It is consumed by **no** refusal row — which also moved row 6's and
   row 7's accept-legs onto row 1, since both are refusal pairs. Its own scoring is untouched, and
   `ROW_SPEC["r2_unseen_valid"]` now says so in the artefact.
4. ⭐ **Row 6 is `evidential: false`.** `sanity_check_verdict` replaces `verdict_vs_control` for it, and
   ⛔ **omits `beats_control` entirely** rather than setting it `False` — a hub cannot read a win out
   of it *and cannot read a loss out of it either*. Verified: `"beats_control" not in verdict`.

⛔ Untouched: the split index, the cells, row 2's records, row 5's records, the scope statement, the
byte ledger, row 7, and `grid_syntax_rule` as a mandatory row on all seven.

## A2. ⭐ The structural match — asserted, not argued

`assert_row1_matches_row5` runs inside the emitter and **raises** on any mismatch:

| property | row 1 | row 5 |
|---|---|---|
| records | 512 | 512 |
| kinds | 2 (`seen_hue`, `seen_shape`) | 2 (`held_out_hue`, `held_out_shape`) |
| record-key **set** | `id, kind, reason, request, row, violated_factor` | *identical* |
| `record_key_count` | 6 | 6 |
| terms per request | 6 | 6 |
| `id_token_count` | 4 | 4 |

⚠ **Row 1 carries `violated_factor` and nothing is violated.** That key name is fixed by the pin's
`EXCLUDED_RECORD_KEYS` allowlist, and it is carried **so that `record_key_count` does not become the
new giveaway** — the task's own warning. On a row-1 record it names the **axis**: the factor carrying
the mirrored, *stored* value. `ROW1_SCHEMA` says that in those words so no reader infers a fault.

**Two choices in `build_row1_seen` are load-bearing, and both were declared in
`ROW1_SEEN_RATIONALE` before anything was measured:**

- ⭐ **One value per kind, held constant across its 256 records** — exactly as row 5 holds
  `object_hue = 0.5` (resp. `shape = 0.0`) constant. A control drawn uniformly over the retained
  values would hand a **transductive** rule a free separator: *"the file whose axis is constant is
  the refusal file."* Measured, and the match holds:

  | file | kind | constant axes |
  |---|---|---|
  | `row1_seen` | `seen_hue` | `['object_hue']` |
  | `row1_seen` | `seen_shape` | `['shape']` |
  | `row5_held_out` | `held_out_hue` | `['object_hue']` |
  | `row5_held_out` | `held_out_shape` | `['shape']` |
  | `row2_unseen` | `unseen_valid` | `[]` ← ⛔ the old accept-leg was separable this way too |

- ⭐ **The same distance from the factor's range boundary as the value it mirrors.**
  `object_hue`: held-out index 5 → `min(5, 9−5) = 4`; chosen index 4 → `min(4, 9−4) = 4`.
  `shape`: held-out index 0 → distance 0; chosen index 3 → distance 0.
  ⇒ an **unsigned** boundary rule fires identically on both legs. Confirmed in §A5: `boundary_any`
  scores `0.3354`, near the bottom of all 27 candidate choices.

## A3. ⭐ Acceptance §4.1 — `grid_syntax_rule` collapses on row 5

Scored with the shipped `score_refusal_row`, on the shipped 512-record files. The two right-hand
columns are the **original** pairing, recomputed here so the before/after is one table:

| rule (⛔ **0 bytes** of split knowledge) | `reject` | accept **(row 1)** | **PAIR** | accept *(row 2)* | *PAIR* |
|---|---|---|---|---|---|
| ⭐ **`grid_syntax_rule`** (§4, mandatory) | 1.0000 | **0.0000** | ⭐ **0.0000** | 1.0000 | *1.0000* |
| `boundary_min` (any term at grid index 0) | 0.7539 | 0.6504 | 0.7002 | 0.6895 | *0.7210* |
| `boundary_max` (any term at the last index) | 0.4902 | 0.1953 | 0.3094 | 0.4668 | *0.4784* |
| `boundary_any` (any term at **either** end) | 0.9141 | 0.1230 | 0.3354 | 0.3105 | *0.5328* |
| `interior_median` | 0.7109 | 0.5820 | 0.6433 | 0.4629 | *0.5737* |
| ⚠ `shape_is_minimum` | 0.6426 | 1.0000 | ⚠ **0.8016** | 1.0000 | *0.8016* |
| `accept_everything` (degenerate guard) | 0.0000 | 1.0000 | 0.0000 | 1.0000 | *0.0000* |
| `refuse_everything` (degenerate guard) | 1.0000 | 0.0000 | 0.0000 | 0.0000 | *0.0000* |

⭐ **`1.0000 → 0.0000`.** It rejects all-on-grid row 1, so its accept-leg collapses and the geometric
mean takes the pair to zero. **Option 1 works against the rule the ruling was aimed at.**
It collapses on row 6 identically (`0.0000`), for the same reason.

## A4. ⭐ Acceptance §4.2 — the detector, full per-feature table

`assert_no_model_detector_at_chance` on the shipped `query_sets.json`, tolerance `0.05`
(threshold `0.5500`). `n_records`: base 2048, `row1_seen` 512, `row2_unseen` 512, `row5_held_out` 512,
`c_invalid` 512, `a7_redundant` 512.

| structural feature | `duplicate_factor` vs `a7` | `out_of_range` vs **row1** | `held_out_hue` vs **row1** | `held_out_shape` vs **row1** |
|---|---|---|---|---|
| `first_duplicate_position` | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| `first_factor_index` | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| `is_factor_order_sorted` | 0.5010 | 0.5000 | **0.5000** | **0.5000** |
| `last_factor_index` | 0.5013 | 0.5000 | **0.5000** | **0.5000** |
| `masked_request_json_len` | 0.5003 | 0.5000 | **0.5000** | **0.5000** |
| `max_factor_multiplicity` | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| `n_distinct_factors` | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| `n_order_inversions` | 0.5013 | 0.5000 | **0.5000** | **0.5000** |
| `n_redundant_terms` | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| `n_terms` | 0.5000 | 0.5000 | **0.5000** | **0.5000** |
| `sum_factor_indices` | 0.5013 | 0.5000 | **0.5000** | **0.5000** |
| **pooled (stratified, directional)** | 0.5006 | — | **0.5000** | **0.5000** |

⭐ **`violations: []` — PASS.** Both row-5 kinds sit at **exactly 0.5000 on every one of the 11
features**: the two legs are structurally identical, term for term. `duplicate_factor` keeps
`a7_redundant` (7 terms; a 6-term control restores the R26.1 giveaway) and is unchanged.

**The declared extra diagnostic (`grid_reach`, ⛔ reported, never asserted):**

| comparison | `n_offgrid_terms` | `max_distance_outside_range` |
|---|---|---|
| ⭐ row 5 vs **row 1** | **0.5000** | 0.5000 |
| ⛔ row 5 vs *row 2* (retained as the finding) | ***1.0000*** | 0.5000 |
| row 6 `out_of_range` vs row 1 | 1.0000 | 1.0000 |

That `1.0000 → 0.5000` is §0's leak, closed, in the same number that measured it.

## A5. ⛔⛔ Acceptance §4.3 — what I tried to break it with, and what got through

⚠ Two things got through. Neither is the rule the ruling was aimed at, and **both must pay for the
split** — but "pays for the split" turns out to be cheap, and that is the finding.

### A5.1 ⚠ RESIDUAL — a free rule that guesses the hole's *position* scores **0.8016**

`shape_is_minimum`: *reject iff the `shape` term is the minimum shape.* It reads the request and the
grid, needs **zero** knowledge of the split, and scores `(0.6426, 1.0000) → 0.8016`.

**Why.** No train row carries `shape = 0.0`, so its accept-leg is a perfect `1.0000` **whatever row 1
is built from**. The held-out cube sits at grid **index 0** — the one grid position an adversary can
name without knowing anything.

⭐ **It is not a defect of my control, and I checked that rather than claiming it.** Re-emitting row 1
at **all 27** retained `(hue, shape)` pairs and re-scoring gives **one** distinct value:

```
shape_is_minimum_rule, over all 27 candidate row-1 choices
  distinct pair scores: [0.80161]
```

⇒ the residual is a property of **which value the condition holds out** (`shape = 0.0`, grid index 0),
which is frozen in `factor_index.npz` and not mine to move. ⛔ It is **not** removable by any choice of
accept-leg.

**⛔ I did not re-tune row 1 to suppress it, and here is the table that shows what declining cost.**
Worst free-rule score over the same 27 candidates — shipped choice marked:

| row 1 `(hue, shape)` | `grid_syntax` | `boundary_min` | `boundary_max` | `boundary_any` | `interior_median` | **worst** |
|---|---|---|---|---|---|---|
| **(0.4, 3.0)** ⭐ **SHIPPED** | 0.0000 | 0.7002 | 0.3094 | **0.3354** | 0.6433 | **0.7002** |
| (0.0, 2.0) — the minimiser | 0.0000 | **0.4793** | 0.4537 | 0.3826 | 0.3670 | **0.4793** |
| (0.9, 3.0) | 0.0000 | 0.6992 | 0.0000 | 0.0000 | 0.6302 | 0.6992 |
| *(the other 24)* | 0.0000 | 0.69–0.72 | 0.27–0.54 | 0.29–0.59 | 0.34–0.70 | 0.69–0.72 |

⛔ **I kept (0.4, 3.0) and did not take (0.0, 2.0).** Three reasons, and the first is the one that
matters:

1. ⭐ **(0.0, 2.0) scores lower only by damaging its own accept-leg.** `boundary_min`'s accept rate on
   row 1 falls from `0.650` to `0.336` — because `hue = 0.0` *is* grid index 0, so the rule refuses
   the control too. A control the rule also rejects is not a better control; row 1 is **also the
   judge-accuracy row**, and an accept-leg chosen because a rule declines it is meaningless there.
2. `hue = 0.0` is a range endpoint, which `HELD_OUT_HUE_RATIONALE` already records as *"exactly where
   `out_of_range` (row 6) lives"* — and row 1 is now **also** row 6's structural control.
3. `shape = 2.0` is the **sphere**, the shape a degraded render collapses towards (§3.1's hazard,
   recorded in `HELD_OUT_SHAPE_RATIONALE`). The worst possible choice for a leg that must *accept and
   render*.

⇒ **Tuning the control against the attack after measuring it is the contrivance §7 warns about.** The
choice was declared in `ROW1_SEEN_RATIONALE` **before** the ablation ran; the ablation is reported
**because** it did not confirm the choice on every axis.

⭐ What the declared reasoning *did* buy, confirmed: `boundary_any` (the **unsigned** boundary rule) is
`0.3354` at the shipped choice — the boundary-distance match works exactly as predicted. It is the
**signed** rule that survives, and only because the hole is at an end of the shape axis.

### A5.2 ⛔⛔ THE REAL ONE — an **8-byte store** wins row 5's pair perfectly

`value_presence_bitmap`: *reject iff any term's value is absent from the store's fit pool.*
One bit per `(factor, grid value)` — `10+10+10+8+4+15 = 57 bits = **8 bytes**`.

| arm | deployed bytes | `reject` | accept (row 1) | **PAIR** |
|---|---|---|---|---|
| ⛔ `value_presence_bitmap` | **8** | 1.0000 | 1.0000 | ⛔ **1.0000** |
| *the same bitmap on the **original** pairing* | 8 | 1.0000 | *0.0000* | *0.0000* |

⭐ **Read the second row: the amendment inverted which cheap thing wins.** The old pairing was
winnable by a **syntactic rule at 0 bytes** and *not* by a minimal store (the bitmap rejects row 2,
which is off-grid). The new pairing is winnable by a **minimal store at 8 bytes** and *not* by a
syntactic rule. ⇒ The pairing now rewards **store contents** instead of **grid syntax**, which is
exactly what the ruling was for — ⛔ but the amount of store it rewards is **8 bytes against a
524,288-byte budget**.

⛔ **What this does to row 7, which §3 asks me to rank first.** The bitmap is a *readout of the fit
pool*, so deleting a value updates it and its `delta_reject` is `1.0`, not `0.0`. `grid_syntax_rule`'s
delta is still exactly `0.0` — the grid does not move — so §3's claim holds **against syntactic
rules** and I have recorded it that way in `ROW_EVIDENCE_RANKING`. ⚠ It does **not** hold against a
minimal store. **Row 7 is the strongest instrument the programme has; it is not an unconditional one,
and the artefact should not be read as saying it is.**

⇒ ⭐ **The refusal rows can never carry a byte-efficiency claim.** Rows 5 and 7 are winnable at 8
bytes. The bitmap scores **`0.0000` on rows 1–4** — it produces no image. So the claim the scoreboard
can actually support is the **conjunction**: generate *and* refuse, at matched bytes, against
`basis_head`. The harness already prints exactly that (byte ledger constant across rows; generation
rows scored beside refusal rows), so ⛔ nothing needs changing — but a hub that quotes row 5 alone is
quoting a row an 8-byte lookup table wins.

### A5.3 What else I tried, and what it scored

| attempt | result |
|---|---|
| **"guess the hole"** — `reject iff object_hue == h or shape == s`, swept over **all 40** `(h, s)` | ⭐ exactly **1 of 40** is perfect — the true `(0.5, 0.0)`. max `1.0000`, mean `0.4418`, median `0.3572`. ⇒ an adversary who must *declare the rule in advance without the split* wins **1 time in 40** |
| **transductive constant-axis detector** | ⭐ **blind** — row 1 and row 5 each have exactly one constant axis per kind (table in §A2). It would have separated row 2 (`[]`) |
| **per-value marginals** — AUROC of every single `(factor, value)` indicator, row 5 vs row 1 | **53 of 57 at ≤ 0.55.** The only four above are the legs' own defining values: `shape==0.0` (0.8213), `hue==0.5` (0.7861), `shape==3.0` (0.7715), `hue==0.4` (0.7656) — i.e. *naming the hole or its mirror*, which is §A5.1/§A5.2 again and not a new channel |
| the four **structural** feature views, both row-5 kinds | 0.5000 on all 11 (§A4) |

## A6. Hashes — which moved, and exactly why

⭐ **The condition re-emits bit-for-bit deterministically.** Verified first: rebuilding the *original*
seven artefacts from `factor_index.npz` alone reproduced **all seven** shipped sha256s before a line
was changed. So every movement below is content, not noise.

| artefact | status | sha256 |
|---|---|---|
| `factor_index.npz` | **unchanged** | `3e29316f…0baf` |
| `c_invalid.json` | **unchanged** | `9146b899…678b` |
| `a7_redundant.json` | **unchanged** | `bb52ba42…e191` |
| `row2_unseen.json` | **unchanged** | `7a7908fe…a343` |
| ⭐ `row5_held_out.json` | **unchanged** | `e0b6ea40…4d82` ← the task's precondition hash |
| ⭐ `row1_seen.json` | **NEW** | `215104e7…7d20` |
| `value_cells.json` | **MOVED** | `da1d883e…a00b` |
| `query_sets.json` | **MOVED** | `17288e94…72bd` |

**The two that moved, keyed diff — additions and two re-points, no deletions, no changed value:**

```
value_cells.json   + /config/row1_n_per_kind        + /config/row1_seed
                   + /config/row1_seen_hue_value    + /config/row1_seen_shape_value
                   + /row1_seen_rationale           + /row_evidence_ranking

query_sets.json    + sets/row1_seen                 + /row_evidence_ranking
                   ~ sets/row5_held_out/controls  {hue,shape}: row2_unseen -> row1_seen
                   ~ sets/c_invalid/controls      out_of_range: row2_unseen -> row1_seen
```

⛔ **No cell, no split row, no held-out value, and no record of rows 2/5/6/A7 changed.** `value_cells`
moved only because `ValueHoldoutConfig` gained four row-1 keys and the metadata gained the two
declared blocks; `query_sets` moved only because the registry gained a set and two control re-points.
⛔ The seven read-only pinned files are untouched.

## A7. ⭐ §3 — the evidence ranking, in the artefact

`ROW_EVIDENCE_RANKING` is defined **once** (`chlu.data.gen_value_holdout`), folded into `ROW_SPEC` at
import, and written into **three** places so no hub re-derives it: `value_cells.json`,
`query_sets.json`, and the scoreboard table (top level **and** per row, as `evidential` /
`evidence_rank` / `evidence_note`).

| rank | row | `evidential` | why |
|---|---|---|---|
| **1** | `r7_deletion` | ✅ | deleting a value does not move the factor grid ⇒ a **syntactic** rule's `delta_reject` is exactly `0.0`. ⚠ qualified by §A5.2: a minimal store's is not |
| **2** | `r5_comp_refusal` | ✅ | evidential **under the amended pairing only** — both legs on-grid, differing only in whether the store holds the value. ⛔ Under the original pairing it was free |
| **3** | `r6_invalid_refusal` | ❌ | out-of-range is decidable from the grid alone. ⛔ Can never be non-free, at any pairing |

## A8. ⚠ §7's third bullet, said plainly in the artefact

⭐ **Row 5 is now a row `basis_head` may pass trivially — it never saw the held-out value either.**
⛔ That is **not a defect, it is the comparison**: the two arms are scored side by side at matched
bytes, and a property they *share* is not a CLU win. It is written into the scoreboard module
docstring §3 and into `ROW_SPEC["r1_seen"]`, so a hub reading only the artefact meets it.

## A9. ⛔ What I did not do

- **Tests: skipped, per the Head's standing instruction for this spoke** (only the three §4 acceptance
  checks were required, and all three ran). The four pre-existing split/detector test files were
  **not** run. ⚠ What I *can* state without running them, from the source: **three assertions now
  encode the superseded pairing** and will fail a future pass —
  `tests/test_gen_scoreboard.py:161` (`accept_unseen_valid_source == "r2_unseen_valid"`, now
  `"r1_seen"`) and `tests/test_gen_value_holdout.py:305–308` (the three `controls` entries, now
  `row1_seen`). Every call site of `score_refusal_row` / `score_deletion_row` is **positional**, so
  the signature change breaks nothing else. ⇒ **a small, bounded follow-up pass on the two owned test
  files is outstanding.**
- ⛔ I did not re-tune the row-1 values to suppress §A5.1 — see the three reasons in §A5.1.
- ⛔ I did not touch any of the seven read-only pinned files.

## A10. For the Head — the one decision this opens

**Option 1 works for what it was aimed at: no syntactic rule reaches row 5's pair any more
(`1.0000 → 0.0000`), and the two legs are structurally identical at 0.5000 on all 11 features.**
⚠ Two residuals are on the record and neither is patchable by re-drawing the accept-leg:

1. A **positional guess** at the hole (`shape_is_minimum`) scores `0.8016` at zero bytes, because the
   held-out cube sits at grid **index 0**. Removing it means holding out an **interior** shape —
   and of the two interior values, `2.0` is the sphere (the blob hazard) and `1.0` is Montero's
   held-out shape (entangled with both combination conditions). ⇒ **there may be no clean fourth
   choice on this axis**, which is worth knowing before wave 3 spends a run on it.
2. An **8-byte value-presence bitmap** wins rows 5 *and* 7 perfectly and scores `0.0000` on rows 1–4.
   ⇒ ⛔ **no refusal row can carry a byte-efficiency claim on its own.** The claim the harness can
   support is the conjunction — generate **and** refuse, at matched bytes — and it already prints it.

⛔ Neither of these says option 1 failed. Both say the refusal rows are worth **less alone** than the
seven-row table is **together**, which is an argument for quoting the table and not a row.
