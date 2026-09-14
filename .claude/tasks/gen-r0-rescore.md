# Task: gen-r0-rescore — re-score R0 under the corrected metrics and the new decoder. No retraining.

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/gen-r0-rescore.md` + artefacts under `.claude/outputs/gen-r0-rescore/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R16, R19 and especially R20 are the whole brief** · `.claude/outputs/gen-store-grouping.md` **§3, §4, §6, §10** · `.claude/outputs/gen-decoder-lift.md` **§5** · `.claude/outputs/gen-feature-bridge.md` **§6**.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Base:** `main @ e4f3d71`. **Repo: read-only on `chlu/`** — you run existing entry points and write analysis. ⛔ No new production code; flag defects for the engineer rather than fixing them. A scratch analysis script under your output dir is fine and is **not** production code.
- **Precondition:** none. `chlu/core/gen_store.py`, `chlu/core/gen_bridge.py` and the R0 artefacts are all on `main`.
- **Env:** main venv, `PYTHONPATH=<repo> /Users/user/Desktop/CHLU/.venv/bin/python -m …`, `CHLU_GEN_DATA_ROOT` set.
- ⛔ **CSF3 needs 2FA and is unreachable from an agent session.** This is a local CPU task; nothing here needs the cluster.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *isolation* — R0's grouping evidence, re-expressed on metrics that are not degenerate.
- **Laundering control:** the **random grouping at the same K** stays a full arm on every number. The **matched-seen**
  set is now a *constituent of the metric itself*, not just a companion row.
- **Falsifies:** the correlation-vs-random margin **disappears** under the corrected metrics ⇒ R0's grouping evidence
  was an artefact of the degenerate normalisation. ⭐ **That is a real possible outcome and it must be reportable.**
- **Does NOT falsify:** a lower absolute level than the raw-accuracy numbers. Balanced accuracy on a 2-class problem
  has chance 0.5; ⛔ the old numbers are **not** comparable in level and must never be placed in the same column.

## The one job
Re-express R0's existing results under **R20's corrected metrics**, against the **decoder-lift winner**, with **no
retraining of anything**.

## Why nothing retrains
The store trains on cached features from the **frozen encoder**; the leakage probes never touch the decoder. Only the
**decode-and-probe** step and the **metric definitions** change. ⛔ If you find yourself retraining a store, an encoder
or a decoder, stop — you have misread the task.

## The three corrections to apply

### 1. ⭐ Kill rule (i)'s metric — discrimination within the corner
⛔ The old `shape` ratio divided a **detection rate** (held-out cell is one class, majority 1.000) by a **3-way
discrimination rate**. Replace it:
> **Positives:** the 72 held-out cylinders. **Negatives:** **matched-seen non-cylinders in the same colour corner**
> (the `A_matched` population, n≈209, restricted to `shape ≠ cylinder`).
> **Score:** binary **AUROC** *and* **balanced accuracy**, cylinder-vs-not, decoded through the frozen decoder and read
> by the probe. Report both; name which one kill rule (i) reads.
⚠ A probe that answers "cylinder" for anything in that colour corner scores 1.0 on the old metric and **0.5 on this
one**. That is the entire point.

### 2. ⭐ The ladder x-axis — balanced, never a ratio of raw accuracies
> `leakage = 1 − [balanced accuracy of the per-well shape probe on (held-out ∪ matched-seen-other-shapes), same corner]
>            ÷ [the same metric on seen corners]`
Per well, pooled, **≥3 seeds with spread**, for **both** the correlation arm and the random control.
⛔ The old definition produced per-factor values as extreme as **−12.9**. Show that the corrected one is bounded and
non-degenerate on **every** (well, factor) cell — ⭐ **report the min and max across all cells as the proof**, not as a
footnote. Report DCI/MIG/block-diagonality unchanged alongside (they were never degenerate).

### 3. The read metric against the decoder-lift winner
R0's read numbers (aggregate 0.5997, `shape` 0.5785) were decoded through the **superseded** decoder. Re-score them
through the **V1 `capacity_schedule`** decoder, which is now THE shared decoder. Report old and new side by side,
⛔ clearly labelled, and state whether the correlation-vs-random margin moves.

## ⚠ The cache trap that will bite you (from `gen-store-grouping` §10)
The bridge's feature caches are **not uniform**: seed 0 is `scope="full"` (480,000 rows); **seeds 1 and 2 are
`scope="used"` (9,288 rows) and contain NONE of the matched-seen control rows.** ⭐ Every metric above needs the
matched-seen rows, so **seeds 1 and 2 cannot be scored without regenerating their caches.**
- Regenerating a cache is a **frozen-encoder forward pass** — it retrains nothing and is permitted. Do it if needed,
  and **assert the regenerated cache reproduces seed 0's overlapping rows bit-identically** before you trust it.
- ⛔ If you cannot get 3 seeds, **report 1 seed and say so.** ⛔ Never present a 1-seed number as if it were 3.

## Acceptance
1. Kill rule (i)'s corrected metric reported (AUROC **and** balanced accuracy), correlation vs random, with spread.
2. The corrected leakage reported per well and pooled, with **min/max across all cells** as the non-degeneracy proof.
3. R0's read re-scored through the new decoder, old and new side by side, with the margin's movement stated.
4. ⭐ An explicit verdict on the falsifier: **does the correlation-vs-random margin survive the corrections?**
5. Seed count stated honestly per number; any cache regeneration justified and its bit-identity asserted.
6. Every number traces to a file under your output dir. ⛔ No number that lives only in the report.
7. ⛔ **No kill-rule verdict on the cell** — you are correcting an instrument, not running the cell.

## What would prove this spoke wrong
- ⭐ **You reported the corrected numbers in the same column as the old ones.** Balanced accuracy has chance 0.5;
  raw multi-class accuracy does not. Placing them together invites exactly the false comparison this task exists to end.
- You silently scored one seed and wrote "±" as if from three.
- The margin vanished and you looked for a third metric rather than reporting it.
