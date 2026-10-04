# Task: w3-value-holdout-harness — the VALUE-holdout splits and the seven-row scoreboard

- **Agent:** `experiment-engineer` · **Branch:** `agent/experiment-engineer/w3-value-holdout`
- **Output:** `.claude/outputs/w3-value-holdout-harness.md` · **Code:** additive modules + tests
- **Base:** ⛔ branch from the pin **`4125fce`**, not from `main`, not from a peer branch.
- ⛔ **Nothing in wave 3 launches until this lands.** Four hubs are blocked on it.

## DIAL DECLARATION
**Dial: the instrument.** This spoke builds the measuring device for wave 3 and runs no arm and no
scored comparison. ⛔ It spends no scored run. The laundering control it must ship is the no-model
detector (§4). **Falsified if** the detector can separate the refusal rows from the accept row using
the request alone — then the refusal rows are free and must not ship as written.

---

## 1. The one job

**Build the value-holdout condition and the seven-row scoring table, as additive modules, and prove
both with the audits the pin already provides.**

Wave 2 held out **combinations**. Every row of the Head's taxonomy that turns on a value the model has
never stored — rows 2, 5, 6, 7 — is unmeasurable against a combination holdout. That is the gap.

## 2. ⚠ THE SCOPE STATEMENT THAT MUST TRAVEL — read before you design anything

**The holdout is from the STORE'S FIT POOL, not from the encoder, the decoder or the judge.**

Those three are frozen, hash-pinned and read-only (charter §A.4), and they were fitted on a train split
containing **every hue and every shape**. Retraining them on a value holdout would void every wave-2
number and is not in scope.

⇒ **What the taxonomy's "never trained on" means in wave 3 is: the store never held a row carrying that
value.** The decoder can still render it and the judge can still recognise it — which is what makes the
rows scorable at all.

⛔ **Write this scope statement into the scoreboard artefact itself**, as a field every hub's report
inherits, so no hub can quote a row without it. A referee who is not told this will read "never seen"
as end-to-end and the claim dies.

## 3. What to build — two additive modules, ⛔ no pinned file edited

### 3.1 `chlu/data/gen_value_holdout.py` (NEW)
The pin already has what you need — ⛔ do not reimplement it:
- `FactorCell` with a factor **absent ⇒ unconstrained** already expresses a value holdout. A cell
  constraining **only** `object_hue` to one value holds that hue out across every shape, scale and pose.
- `build_cell_split(grid, cells, ...)` makes train the **exact complement**.
- `assert_holdout_disjoint`, `assert_index_partitions_grid`, `assert_non_empty_test`, `save_index`,
  `write_json_with_checksum` are the proofs. **Run all of them.**

Emit a condition `value_holdout` with, at minimum:
- one **whole `object_hue` value** held out (the row-5 hue), and
- one **whole `shape` value** held out (the row-5 shape),
- and — because rows 3–4 still need them — **combination cells of Montero's shape** over the values that
  remain. ⛔ Value cells and combination cells must be disjoint; `assert_cells_disjoint` proves it.

⭐ **Choose the held-out hue and shape, and write down why, before you build.** ⚠ Do not pick `scale` or
`orientation` for the value holdout: wave 2 measured both as **physical defaults** the model degrades
*towards* (median `scale` index = the default object size; median `orientation` index = the null pose),
so a held-out default is unremovable for reasons that have nothing to do with the store.

### 3.2 `chlu/experiments/gen_scoreboard.py` (NEW)
The seven rows as **columns of one table**, every arm and **both controls** on every row:

| row | what is requested | scored as |
|---|---|---|
| 1 seen | a stored value-combination | judge accuracy + valid tuples |
| 2 unseen-valid | a hue **between** two stored values (in-basin interpolation) | judge accuracy + valid tuples |
| 3 compositional 2-way | seen values, unseen pair | distinct valid novel tuples |
| 4 compositional n-way | seen values, unseen n-way conjunction | distinct valid novel tuples |
| 5 compositional refusal | a value **held out entirely** | refusal pair (§4) |
| 6 invalid-request refusal | a value **outside the grid's range** | refusal pair (§4) |
| 7 deletion | post-deletion: 3–4 still pass for survivors, 5 fires for the deleted value | both, no retraining |

**Both controls, on every row, always:** ⭐ **fitted basis + linear head** (the wave-2 object) is the
**first** control and the like-for-like; `knn_proto` is the **second** row. Byte-matched, ledger asserted.

⭐ **Row 6's requests already exist:** `build_c_invalid(...)` emits `out_of_range` and `duplicate_factor`
records from the grid alone. Use it. ⛔ Do not invent a second invalid-request format.

## 4. ⭐⭐ The refusal pair — and the thing that makes it non-free
Rows 5 and 6 report **the pair `(reject_invalid, accept_unseen_valid)`**, never one number.
⛔ **An arm that refuses everything scores 0.** Encode that in the metric, do not leave it to prose:
the row's score is `0` unless `accept_unseen_valid` is measured on **row 2's** requests by the **same
threshold, at the same setting**.

⭐ **The separation that is the whole claim:** row 2 (between stored values → **accept**), row 5 (value
never stored → **refuse**), row 6 (off-grid → **refuse**). A prototype table has no entry for any of the
three and refuses all three, scoring 0 on the pair. ⇒ The table's failure must be **measured and
reported**, not asserted.

⛔⛔ **Run `assert_no_model_detector_at_chance` over the row-2 / row-5 / row-6 query sets before you
ship them**, at the tolerance the pin uses. It scores the orientation-free AUROC of every structural
feature of a request, invalid vs valid. **If a no-model detector separates them from the request alone,
the refusal rows are winnable without a store and are void.** ⚠ Report the per-feature AUROCs in your
output whether or not they pass.

## 5. Configuration — ⛔ avoid the merge trap
⛔ **Do not add a block to `chlu/config.py`.** Its same-anchor merges have cost this programme real time.
⭐ Put the config dataclass **next to the code that reads it**, with a `from_mapping` override — the
precedent is `StreamMemoryConfig` in `chlu/core/blocks.py` and `PsiSpec`, both of which say so in their
docstrings.

## 6. Files
- **You own (new):** `chlu/data/gen_value_holdout.py` · `chlu/experiments/gen_scoreboard.py` ·
  `tests/test_gen_value_holdout.py` · `tests/test_gen_scoreboard.py`
- ⛔ **Read-only, editing them voids every hub's numbers:** `chlu/experiments/exp_gen_cell.py` ·
  `chlu/data/gen_splits.py` · `chlu/data/gen_rowsplit.py` · `chlu/core/gen_store.py` ·
  `chlu/core/gen_landscape.py` · `chlu/core/gen_bridge.py` · `chlu/config.py`
- ⛔ Do not re-point `GenCellConfig.lace_selection_source`. It is a declared property of the pin.

## 7. Precondition — mechanical
`git cat-file -e 4125fce^{tree}` succeeds, and `git show 4125fce:chlu/data/gen_splits.py` contains
`def build_cell_split` and `def build_c_invalid`.

## 8. Acceptance — all six, each proven by a file on disk
1. A `value_holdout` split index on disk **with its sha256**, holding out one whole hue and one whole
   shape, plus disjoint combination cells for rows 3–4.
2. `assert_holdout_disjoint`, `assert_index_partitions_grid`, `assert_non_empty_test` and
   `assert_cells_disjoint` all pass, with their returned counts recorded.
3. `gen_scoreboard` emits all seven rows for a supplied arm, with **basis+head and `knn_proto` on every
   row**, and the byte ledger asserted per row.
4. The refusal rows emit the pair, and a **refuse-everything stub arm scores 0** — ship that stub as a
   test, it is the falsifier for the metric itself.
5. `assert_no_model_detector_at_chance` passes on rows 2/5/6, with the per-feature AUROC table reported.
6. The scope statement of §2 is a **field in the emitted artefact**, not only in prose.

## 9. ⛔ What would prove this spoke wrong
- The no-model detector separates row 5 or row 6 from row 2 above tolerance ⇒ **the refusal rows as
  specified are free. Report that and stop; do not redesign them to pass.** That result blocks
  `hub/refuse-delete` and the Head needs it immediately, not at the end.
- Holding out a whole hue leaves the store's fit pool unable to fit at the shipped scale (too few rows)
  ⇒ report the realised row counts and say so.
- A held-out value turns out to be a physical default the model degrades towards ⇒ pick again, and
  record the rejected choice.
