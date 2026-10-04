# Task: gen-wide-split — define and validate the widened SECONDARY hold-out, without touching Montero's

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-wide-split.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md`** (binding; **R15's photometric finding is the design input for this task**) · `.claude/outputs/gen-data-splits.md` · `.claude/outputs/gen-feature-bridge.md` **§6** · `chlu/data/gen_splits.py`, `chlu/data/shapes3d.py`.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Base:** `main @ fcdca43` · **Branch:** `agent/experiment-engineer/gen-wide-split` · **Worktree:** fresh (`../CHLU-gen-wide`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are parked, not free.
- **Precondition:** none beyond the base. ⚠ **Runs in parallel with `gen-store-grouping`** — see the config.py ownership split below.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument.* ⛔ No store, no arm, no encoder, no CLU number. Nothing scientific runs here.
- **Laundering control:** N/A (nothing measured about a model).
- **Falsifies:** N/A for a dial. Your own falsifier: the absence assertion passes on the wide split but does not fire
  on a corrupted one.

## The one job
Define a **labelled SECONDARY** held-out split — additional held-out factor combinations beyond Montero's 72 — verified
absent from training by the same assertion, emitted as a distinct named condition. **Nothing else.**

## ⛔ The two rules that make this safe
1. ⭐ **Montero's split is the headline and is not touched.** Its 72 rows, its filter, its counts and its condition name
   stay exactly as they are. The wide split is a **separate, additionally-named condition**, reported separately and
   ⛔ **never pooled with the Montero numbers.**
2. ⭐ **THE COST YOU MUST NOT HIDE — the frozen models must never have seen these rows.** The shipped encoder, decoder
   and (later) store were trained on Montero's train split, which **includes** everything you are about to hold out.
   ⛔ **You therefore cannot carve a secondary hold-out out of the existing train set and evaluate the existing frozen
   models on it — they have seen it.** The wide split is a **second full configuration**: the whole stack is re-run
   against it, later, by other spokes. Your job is to define it so that re-run is possible; ⛔ it is **not** to re-run
   anything, and ⛔ **not** to imply the existing artifacts can be scored on it.

## ⭐ Design input — use the bridge's finding, do not rediscover it
Montero's cell requires **all three hues ≥ 0.8**, which puts floor, wall and object in one colour band and collapses
object/background contrast. Measured: the `scale` deficit on held-out data (0.708) is **indistinguishable** from the
same deficit on photometrically matched *seen* data (0.703); only `shape` is genuinely compositional (0.876 → 0.500).
⭐ **So the wide split's job is not merely "more rows" — it is more rows ACROSS THE CONTRAST RANGE**, so that
composition can be separated from rendering. Design the held-out cells to **span the hue/contrast space deliberately**,
and **label each cell with its photometric contrast** so downstream analysis can condition on it.
Keep each cell **structurally the same shape as Montero's** (a conjunction over factors, one shape held out at a
scale/orientation corner) so the two splits are comparable in kind.

## What to build
1. A new condition, e.g. `recomb2element_wide`, in the existing split machinery, parameterised by the number of held-out
   cells and a seed. **Deterministic**; record the rule verbatim in the emitted metadata.
2. Per-cell metadata: the conjunction, the realised row count, and its **contrast label** (a measured quantity from the
   factor values, not a judgement).
3. Realised counts, per cell and pooled. ⛔ Report what the data yields; never adjust the rule to hit a target.
4. ⭐ **The absence assertion and its positive control**, exactly as `gen-data-splits` established: reconstruct each
   train row's factors through the grid bijection and re-evaluate the predicate; then **inject a known held-out row and
   show the assertion fires.** ⛔ An assertion never seen to fail is not evidence.
5. A C set for the wide condition, sized to match its `|B|` (the existing derivation already does this).
6. ⭐ **A `SPLIT`/`CONDITION` axis note for the launcher.** `scripts/csf3/job_gpu_gen.sh` currently carries
   DATASET/PRETEXT/STAGE/ARM/SEEDS and **no condition axis**. ⛔ **Do not edit the launcher** — it is another spoke's
   file. **State precisely what the axis needs to be** so the Hub can scope that edit.

## Files you own
`chlu/data/gen_splits.py` — **the wide-condition builder only**, additive. `chlu/data/shapes3d.py` — the condition
dispatch, additive. `tests/test_gen_wide_split.py` — new. `chlu/cli/data_cmd.py` — one added flag.
⚠ **`chlu/config.py` — shared with `gen-store-grouping`, which is running now. You own the EXISTING `GenDataConfig`
block only** (add fields inside it); ⛔ **it** appends a new dataclass *after* that block. Do not touch anything below
`GenDataConfig`. Report your exact line ranges. Rebase onto `main` before finishing and resolve **only** your own hunk.
⛔ You own nothing under `chlu/core/`, `chlu/eval/`, or `scripts/csf3/`.

## Acceptance
1. The wide condition builds deterministically; the rule is recorded verbatim; per-cell and pooled counts reported as
   **realised**.
2. Each cell carries a measured contrast label spanning the range — ⛔ not clustered in the same corner as Montero's.
3. ⭐ The absence assertion **fires** on the positive control. This is the criterion the task lives on.
4. Montero's condition is provably unchanged: its 72 count and its checksum reproduce **bit-identically** after your
   change. Assert it in a test.
5. The launcher's condition-axis requirement stated precisely, with no launcher edit made.
6. Suite green (pass count before and after, **main venv**, `uv sync --extra gen` already provides h5py);
   `git log --oneline main..agent/experiment-engineer/gen-wide-split` re-verified **from the main repo** before any
   worktree removal.

## ⛔ Cluster
CSF3 requires 2FA and **cannot be reached from an agent session.** Nothing here needs it — this is a local CPU task.

## What would prove this spoke wrong
- **You changed Montero's split.** Item 4 is the check; run it before you believe anything else.
- Your wide cells all landed in the same photometric corner as Montero's, so the split adds rows but no contrast
  variation — which is most of the reason the Head approved it.
- You implied the existing frozen encoder/decoder can be scored on these rows. **They saw them.**

## ⟲ 2026-09-10 — THIS SPOKE IS NOW MANDATORY (Head ruling R17)
It was approved as an optional secondary; it is now **required**. Reason, in one line: with `shape` as the
compositional term and its held-out ceiling at ~0.5–0.59, **72 items × 3 seeds inside a half-width band cannot
separate two arms at any confidence a reviewer will accept.** ⭐ The extra held-out combinations are what turn this
phase from an anecdote into a measurement. Everything else in this file stands unchanged — ⛔ Montero's 72 remain the
untouched headline.
