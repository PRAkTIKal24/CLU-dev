# Task: gen-r0-wide-refit — put R0 on the wide configuration, and prove the two caches are the same encoder's

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-r0-wide-refit.md` + artefacts under `.claude/outputs/gen-r0-wide-refit/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R20, R22, R23 and R24 are the brief** · `.claude/outputs/gen-store-grouping.md` **§3, §6, §8** · `.claude/outputs/gen-r0-rescore.md` **§1–§4** · `.claude/outputs/gen-bridge-wide.md` **§5** (the wide caches you consume).
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Base:** `main @ 9a4457d` · **Branch:** `agent/experiment-engineer/gen-r0-wide-refit` · **Worktree:** fresh (`../CHLU-gen-r0wide`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are parked, not free. ⚠ **Three arm spokes are running concurrently** — see the `config.py` note below.
- **Precondition:** none beyond the base. The wide feature caches exist at `scope="full"` for all 3 seeds from `gen-bridge-wide`.
- **Env:** main venv, ⛔ no `uv sync` in the worktree. ⛔ CSF3 needs 2FA and is unreachable — local CPU only.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *isolation* — R0, re-established on the configuration the ladder will actually run on.
- **Laundering control:** the **random grouping at the same K**, a full arm on every number, as in R0.
- **Falsifies:** ⭐ **wide-R0 and Montero-R0 diverge beyond seed spread on the read margin.** That is **not** a result —
  it means the two feature caches are not the same encoder's, i.e. **a plumbing fault**, and it **STOPS the ladder**.
- **Does NOT falsify:** a *level* difference between the two configurations. They hold out different rows, so their
  absolute reads may differ. ⛔ It is the **margin** (correlation − random) that must agree.

## The one job
Re-fit R0 — **grouping and store** — on the **wide** train cache, and emit R0's coordinates on the wide configuration
so R1–R5 have a comparable baseline.

## Why (Head ruling R24)
The ladder holds store config, bytes and `K` fixed across rungs, and R1–R5 run on the **wide** configuration. The
shipped R0 store was fitted on **Montero's** cache. ⭐ **A monotonicity claim on a mixed baseline is no claim at all.**

## The three riders — these are the task
### 1. ⭐ Re-derive the GROUPING on the wide train cache, not just the store weights
The correlation clustering was computed on Montero's training features; the wide training set differs by the held-out
cells. Expect near-identical channel correlations — ⭐ **but measure it, do not assume it.**
- Recompute the correlation clustering on the **wide** train cache.
- **Report agreement between the Montero grouping and the wide grouping — adjusted Rand index or equivalent**, per seed.
- ⭐ **High agreement documents a non-issue. Low agreement is a finding about grouping stability that we want BEFORE
  R1, not after.** Either way it goes in the ledger and R1 does not start until it does.
- ⛔ **`K` stays fixed** — it is a ladder constant. Load `store_plan.json` verbatim via `load_store_plan()`;
  `assert_plan_matched` must pass. ⛔ A rung that re-derives `K` is refused, and that includes you.

### 2. ⛔ Montero R0 stays as the literature link, OFF the ladder
You produce **wide-R0 only**. Montero R0 keeps its place in the compositional-results table against the published
split; ⭐ **the ladder figure carries wide-configuration rungs only. Never both on one axis.**
⛔ Do not re-run, re-fit or modify Montero R0. Read its shipped numbers for rider 3 and leave it alone.

### 3. ⭐ THE GATE — wide-R0 and Montero-R0 side by side, once
A consistency check on the **caches**, not on the science.
- Report the **read margin** (correlation − random), aggregate and `shape`, for both configurations, ≥3 seeds.
- ⛔ **If they diverge beyond seed spread, STOP before R1 and report a plumbing fault** — that would mean the two
  feature caches are not the same encoder's. ⛔ Do not rationalise a divergence as a property of the split.
- Montero's shipped values for the comparison: aggregate margin **+0.1504 ± 0.0094** (V1 decoder), `shape` margin
  **+0.2008 ± 0.0165**, 3/3 seeds positive.

## Also emit — R0's ladder coordinates on the wide configuration
- **x-axis, as ruled (R23.2):** `leakage = 1 − [balanced-accuracy transfer on (held-out ∪ matched-seen-other-shapes,
  same corner)] ÷ [the same on seen corners]`. ⛔ **Never a ratio of raw accuracies** — the old definition produced
  55/180 NaN cells and a worst case of **−198.0**. Report **min and max across every cell** as the non-degeneracy proof.
- **Companions, pre-registered not substitutes:** MIG and DCI-C.
- ⭐ **Name whether the ruling-2 fallback triggers:** if `shape` leakage separates the groupings by **less than ~3×
  seed spread**, say so plainly — `shape`-leakage is then declared **underpowered, in those words**. ⛔ That is a
  pre-registered outcome, not a failure, and ⛔ it is not yours to soften or to fix by trying another statistic.
- **y-axis:** the read metric (balanced accuracy is the gate statistic; AUROC a mandatory companion), `shape` reported
  separately from the aggregate.

## ⛔ THE DECODER TRAP
`chlu/config.py` ~line 3358: **`dec_recipe: str = "baseline"` is the SUPERSEDED decoder** (0.7191, failed the gate).
The winner is **`v1_capacity_schedule`** (0.9414). ⭐ **Select it explicitly and assert the decoder parameter hash
against `gen-decoder-lift`'s winner before recording any number.** ⛔ A run through the baseline recipe is void and
looks entirely normal in the logs.

## Shared-corner rule (R23.3)
⛔ `montero` and `wide-000` are a **two-class corner, never pooled.** The pooled headline runs on the **other 11
cells**; the corner pair is reported alongside.

## Files you own
`chlu/core/gen_store.py` — **additive only**: a wide-condition entry point and the grouping-agreement metric. ⛔ Do not
change the shipped fitting behaviour; Montero R0 must reproduce bit-identically, asserted in a test.
`tests/test_gen_r0_wide.py` — new.
⚠ **`chlu/config.py` is shared with three concurrent arm spokes.** Append your own fields at the **end of the file**,
below every existing block; ⛔ touch no other hunk. Rebase onto `main` before finishing; resolve **only** your own.
⛔ You own nothing under `chlu/eval/`, `chlu/data/`, `chlu/core/gen_bridge.py`, or `scripts/csf3/`.

## Acceptance
1. Wide grouping re-derived; **adjusted Rand agreement vs the Montero grouping reported per seed.**
2. `assert_plan_matched` passes — `K` and the byte budget are unchanged from `store_plan.json`.
3. ⭐ **The gate:** wide-R0 vs Montero-R0 read margins side by side, with an explicit verdict `CONSISTENT` /
   `PLUMBING FAULT — STOP`.
4. R0's wide-configuration ladder coordinates: corrected leakage (with min/max non-degeneracy proof), MIG, DCI-C, and
   the read metric with `shape` separate.
5. The ruling-2 fallback named explicitly as triggered or not.
6. Montero R0 reproduces bit-identically, asserted in a test.
7. Suite green (before/after, main venv); ref verified **from the main repo** before worktree removal.

## What would prove this spoke wrong
- ⭐ **You reported a wide-vs-Montero divergence as a finding.** It is a plumbing fault until proven otherwise — the
  two caches come from encoders trained on almost the same rows. Check the encoder hashes before you interpret anything.
- You re-derived `K` to make the wide grouping fit better. `K` is a ladder constant.
- You decoded through the baseline recipe.

## ⛔ `chlu/config.py` — FOUR spokes are appending to the same anchor. Read this before you rebase.
`gen-dream-sampler`, `gen-certificate`, `gen-baselines` and `gen-r0-wide-refit` are all running **now**, and all four
were told to append at the **end of `chlu/config.py`** (currently 3,930 lines, ending in `save_config`). ⭐ **You WILL
hit a conflict there when you rebase. It is expected, it is not a defect, and it is not a reason to stop.**

**How to resolve it — the only acceptable resolution:**
- ⭐ **Keep BOTH blocks**, in whatever order they arrive. Your block and the other spoke's block are independent
  dataclasses; there is no semantic conflict, only a textual one at the same anchor line.
- ⛔ **NEVER delete, rename, reorder or "tidy" another spoke's block to make the conflict go away.** That silently
  destroys a peer's work and it will not be caught until their tests fail on `main`.
- ⛔ Do not touch any hunk above the anchor. Everything from line 1 to `save_config` belongs to earlier waves.
- If the conflict is anything other than "two additions at the same end-of-file point", **stop and report** — that
  means someone edited shared state and it is not yours to adjudicate.

**Wiring:** if your dataclass needs to round-trip, wire it at the three house sites (the dataclass, `load_config`,
`save_config`) exactly as `GenDataConfig` and `GenBridgeConfig` do. ⚠ If you only add **fields to an existing
dataclass**, no new wiring is needed — they round-trip automatically.
