# Task: gen-atom-mean-resweep — ⭐ does the ladder mean anything? One sweep answers that and clears one stale number.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-atom-mean-resweep.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ `.claude/outputs/gen-val-split-fix.md` §5 (the 5-of-6 that motivates this) · `.claude/outputs/gen-dynamical-read/PREREG.md` §5b.0
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3eb9eb0` · **Branch:** `agent/experiment-engineer/gen-atom-mean-resweep` · **Worktree:** `../CHLU-resweep`
- **Env:** main venv, **local, CPU/`JAX_PLATFORMS=cpu` as the split-fix leg ran.** pytest with
  `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**.
- ⚠ **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**;
  select it explicitly and **assert the decoder parameter hash**. A baseline run **looks normal in the logs and is void.**
- ⚠ **JAX `split(key, n)` is prefix-stable here** — ⛔ derive arm keys with `fold_in`.
- ⚠ **`--quick` CLOBBERS explicit CLI flags** in the `exp_gen_*` mains (`apply_quick` *replaces* the seed/budget/arm/
  step lists; measured: `--quick --budget A B` ran budget A only, log entirely normal). ⛔ **Do not use `--quick`.**
- ⚠ `n_values_realised` beside every per-factor number. ⛔ **No full suite** — 16 GB box.
- ⛔ **Do NOT touch `chlu/config.py`.** Everything here is a CLI flag. A peer spoke owns that file this wave.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — a CALIBRATION and one re-measurement.** No arm, no mechanism, ⛔ no verdict on any claim.
- **Laundering control:** the sweep is over `n_val` **only**; ⛔ every other knob stays at the split-fix leg's value,
  so a change in disagreement cannot be attributed to anything else.
- **Falsifies:** ⭐ **potentially this programme's whole tuning ladder** — see below. That is the point of running it.

## Why this is worth a day
The split repair removed a **bias**. It did **not** buy **identification**. On the *exchangeable* interleaved halving
the two halves of `val` still pick **different winners in 5 of 6** cells, with best-to-second gaps of
**0.0007–0.0130** against a trial spread of **0.068–0.111**.
⇒ ⛔⛔ **A six-rung tuning ladder selected on 512 validation rows is a coin flip six times.** The pre-registered
iteration count would then measure effort, not learning.

## The one job
**Sweep `n_val` and report the smallest `n_val` at which the interleaved-half winner disagreement falls to ≤ 1 of 6
cells** — the number that decides whether the dynamical-read ladder is a measurement or theatre.

## Scope
1. **The sweep.** `n_val ∈ {512, 1024, 2048, 4096, 8192}` (extend upward only if the curve has not turned), 3 seeds,
   both budgets, on the **repaired stratified split**, using `exp_gen_atom_mean`'s existing **interleaved** halving.
   ⚠ **`n_fit + n_val` is bounded by `gen_store.n_store_train` (32,768)** — ⭐ **report the `n_fit` you traded away at
   every point**, and ⛔ stop before `n_fit` falls anywhere near the 1,024 rows where `knn_proto` fell 0.8763 → 0.8308.
   ⭐ **That trade-off IS the result** — "more val rows" is not free and the report must price it.
2. **Report, per `n_val`:** cells disagreeing (of 6) · best-to-second gap · trial spread · `n_fit` · wall clock.
3. ⭐ **The one stale number.** Re-measure the **deployable (request-addressed) exponent** on the repaired split.
   Published **−0.162 ± 0.005** is a **broken-split** value and is the last stale figure on the record.
   ⛔ **EXPONENT STUDIES ARE SUSPENDED (Head, 2026-09-15) — this is ONE cheap re-measurement so the record is clean
   into the rebuttal. ⛔ It buys no claim, adjudicates nothing, and ⛔ you draw no conclusion from it.**

## Acceptance
1. A `PREREG.md` filed **before** the sweep, predicting where the disagreement curve turns. ⛔ Filed after = void.
2. The disagreement-vs-`n_val` table, 3 seeds, with `n_fit` priced at every row.
3. ⭐ **A named recommended `n_val`**, or ⛔ **an explicit "no `n_val` in range achieves ≤ 1 of 6"** — which is a
   legitimate and important answer, ⛔ not a failure to report.
4. The re-measured deployable exponent, old → new, labelled **suspended, non-adjudicating**.
5. Targeted tests green with the data root set, passed/skipped/failed reported. ⛔ No full suite.
6. A **cost line**: hardware, wall hours.

## ⛔ FILES YOU OWN — and nothing else
- `chlu/experiments/exp_gen_atom_mean.py` · `tests/test_gen_atom_mean.py`
- `.claude/outputs/gen-atom-mean-resweep/**`
⛔ **You do NOT own `chlu/config.py`** — a peer spoke holds it this wave. Everything here is a **CLI flag**.
⛔ You do not own `chlu/data/gen_rowsplit.py` — the split is **fixed and shipped**; ⭐ you *use* it, you do not retune it.

## What would prove this spoke wrong
- ⭐ **The disagreement never falls** — the grid is a plateau and no affordable `n_val` resolves it. ⛔ Report that
  plainly; it kills the ladder as designed and that is a finding, not a failed task.
- ⭐ **You bought val rows by starving `fit` until the rivals degraded.** That manufactures identification by
  hobbling the comparators — the exact failure the split fix existed to avoid.
- You used `--quick`, or let `dec_recipe` default, and the log looked normal.
