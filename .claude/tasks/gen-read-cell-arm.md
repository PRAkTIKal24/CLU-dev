# Task: gen-read-cell-arm — ⭐ put the dynamical read on the GENERATION TABLE. One arm. Lean on purpose.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-read-cell-arm.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/gen-dynamical-read/PREREG.md` **§2b only** · `chlu/experiments/exp_gen_cell.py` docstring
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 115d4b5` · **Branch:** `agent/experiment-engineer/gen-read-cell-arm` · **Worktree:** `../CHLU-cellarm`
- **Env:** main venv, local. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**.
- ⚠ **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**, hash asserted.
- ⚠ **JAX `split` is prefix-stable** — derive arm keys with `fold_in`. ⚠ **`--quick` clobbers explicit flags.** ⛔ No full suite.
- ⛔ **Do NOT touch `chlu/config.py`** — a peer spoke holds it. `GenDynamicalReadConfig` already exists; reuse it.
- ⚠ **Budget ≈ 1 day.**

## ⭐ DIAL DECLARATION (w24 ruling — binding on every task since w25)
- **Dial:** *compute-adaptive reads* — a read that spends settling steps, scored against reads that spend none.
- **Laundering control:** ⭐ `store_removed` on every row (the harness's) **plus** this arm's own `request_off`
  (β = 0, same launch, same chain, same key).
- **Falsifies:** the table itself — losing to the like-for-like rows beyond seed spread.
- **Does NOT falsify:** losing to `knn_proto`, an unbounded real-valued channel (metric-native-ceiling theorem).
⚠ *Added 2026-09-16. The first issue of this task omitted this block; the spoke had to reconstruct one.*

## ⭐ SCOPE RULING (Head, 2026-09-15) — read this before planning
⛔⛔ **This is deliberately LEAN. ⛔ Do NOT build five bespoke rival arms, and ⛔ do not write a new PREREG for
minor falsifiers.** The paper goal is **one row on the generation table**. Everything else is appendix.
⭐ **The rivals already exist and are already scored on this metric:**
`ARMS = ("clu", "store_removed", "knn_proto", "lace", "ebm_compose")` in `exp_gen_cell.py:155`, with the external
classifier, composition-B aggregate and `distinct_factor_tuples_B` already emitted.
⇒ ⭐⭐ **The job is WIRING, not building.**

## The one job
**Add the dynamical read as ONE new arm (`dynread`) to `exp_gen_cell.py` and score it on composition-B against the
arms already there.**

## Files you own — and nothing else
- `chlu/experiments/exp_gen_cell.py` · `tests/test_gen_cell.py` · `.claude/outputs/gen-read-cell-arm/**`
⛔ Not `chlu/config.py`, ⛔ not `chlu/core/gen_landscape.py` (frozen — you *call* it), ⛔ not `chlu/eval/rivals/**`.

## ⛔ What LEAN does NOT mean
⭐ **Lean is about SCOPE, not rigour.** These are non-negotiable because they cost nothing:
1. ⛔ **The read arm uses the SAME cell, seeds, split and external classifier as every other arm.** An arm scored on
   a different cell is not on the table.
2. ⛔ **The request-off control runs beside it** — at `request_strength = 0`, same chain, same key. It is the whole
   test of steering, because the launch clamps 0 of 5 wells and the request acts only through the energy term.
   ⭐ If request-off scores the same, the arm is settling onto a mode, not being steered, and **that is the result**.
3. ⭐ **Report `distinct_factor_tuples_B` beside the aggregate.** The codebook read renders **9 ± 1 of 72** while
   `knn_proto` renders ⛔ **56.00 ± 3.74** *(corrected 2026-09-16 — the "72/72" in the first issue of this task
   was taken from a docstring and never measured)* — the aggregate alone hides that.
4. ⛔ **`λ` comes from the coercivity certificate** (≈ 0.054), never 1.0. ⛔ The launch is the posterior **argmax**.
5. ⛔ **Chains still descending at their step budget are NOT CONVERGED** and ⛔ yield no quotable number.
   Budget A = 1000 steps, budget B = 2000, checkpointed.

## ⛔⛔ A SILENT TRUNCATION BUG IN THE PATH YOU CALL — measured 2026-09-15
⚠ **`stratified_fit_val` CLAMPS `n_val` to `pool.size − n_fit` — 512 at the shipped `n_fit`.**
⛔⛔ **A caller asking for a larger `n_val` GETS 512 ROWS AND A COMPLETELY NORMAL LOG.**
⭐ Closed inside `exp_gen_atom_mean`; ⛔ **`exp_gen_cell` calls the same function and is NOT fixed.**
**Binding here: keep the shipped `n_val = 512` — ⛔ do NOT raise it in this spoke** (that is the ladder's business,
not the scored cell's). ⭐ **But ASSERT the realised `n_val` and report it**, so a future caller cannot be silently
truncated. ⚠ ⛔ Do not fix `stratified_fit_val` itself — it is shared, and a peer wave owns that repair.

## ⛔⛔ PERSIST THE ARTEFACTS — targets 2 and 4 are scored FROM THIS RUN, not from a re-run
⭐ **Refusal (target 2) and dreaming (target 4) are scored off artefacts this run produces and then throws away
unless you keep them.** ⛔ **Losing them costs a full re-run of the scored cell.**
**Write, per arm, per seed, per budget, under `.claude/outputs/gen-read-cell-arm/runs/…`:**
1. ⭐ **`energy_traces_*.npz`** — per-chain energy **every step**, plus `accept`, `displacement`, and the
   fixed-reference energy. ⛔ Refusal's criterion is `(final energy, still-descending)`; **without the trace there is
   no refusal number.**
2. ⭐ **`settled_states_*.npz`** — the settled continuous states.
3. ⭐ **the DECODED IMAGES** and the external classifier's **factor tuple per decoded image**.
   ⛔ Dreaming's novelty test is *"a valid tuple that is not a stored atom's tuple"* — it needs the decoded tuples
   **and** the stored atoms' tuples, both from this run.
4. **`convergence_report`** per arm: still-descending fraction at **1000 and 2000**.
⛔ **These are artefacts, not results.** ⛔ **You do not score refusal and you do not score dreaming** — separate
spokes own those. **Your job is to make sure they never have to re-run the cell.**
⚠ Keep them under `.claude/`, never in a tracked file.

## Acceptance
1. `dynread` runs as a sixth arm through the unmodified cell path, 3 seeds, same split, same classifier.
2. ⭐ **The generation-table row**, reported against the arms already on it:
   `store_removed` 0.1204 ± 0.0131 · `ebm_compose` 0.1937 ± 0.0182 · `lace` **0.3117 ± 0.0153** (cell harness; ⛔ 0.3241 is the BASELINES instrument at 688,128 B) ·
   ⭐ `clu` (codebook read) **0.5949 ± 0.0082** · ⭐⭐ `knn_proto` **0.8650 ± 0.0122**.
3. ⭐ **The verdict in PREREG §2b's words, as HARDENED by `PREREG-addendum-0` (Advisor, 2026-09-15):**
   | verdict | condition |
   |---|---|
   | **WIN** | ⭐ **`Δ > m`** (m = 0.020) **and** same-sign paired difference on **3/3** seeds |
   | **TIE (favourable)** | ⭐ **`0 < Δ ≤ m`** with 3/3 same sign — ⛔ **never reported as a WIN** |
   | **TIE** | `|Δ| ≤ m` and intervals overlap · **LOSS** `Δ < −m` · **INDETERMINATE** otherwise |
   ⛔ **No other verdict vocabulary.** ⛔ A +0.001 with three agreeing signs is a **TIE**.
4. ⛔⛔ **A TIE IS ONLY AVAILABLE AGAINST `knn_proto`.** ⭐ **`store_removed`, `ebm_compose`, `lace` and `clu`
   (the codebook read this arm replaces) must each be BEATEN OUTRIGHT, beyond seed spread.**
   ⛔ **A tie against any of those is a LOSS**, and is reported as one.
   ⚠ *(The atom-mean lookup is also a like-for-like row but is a READ-ERROR measurement, not a cell arm — it is
   binding in PREREG §2a and ⛔ out of scope here. Do not attempt to add it.)*
5. The request-off control's row beside it.
6. `distinct_factor_tuples_B` and the convergence fraction for every row.
7. ⭐ **The artefacts above on disk**, path-listed in the report so the refusal and dreaming spokes can find them.
8. Targeted tests green with the data root set. ⛔ No full suite. A **cost line**.

## What would prove this spoke wrong
- ⭐⭐ **Request-off scores the same as the full arm** ⇒ the gain is not steering. **Report it; do not tune around it.**
- ⭐ **You beat `clu` but the distinct-tuple count stays ~9/72** ⇒ the aggregate moved without the read getting
  more expressive, and the number is not what it looks like.
- ⭐ **You scored the arm on a cell of your own** rather than the shipped one, so the row is not comparable and the
  whole point is lost.
- A number quoted from a NOT CONVERGED run.
