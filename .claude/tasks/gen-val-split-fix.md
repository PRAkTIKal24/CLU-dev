# Task: gen-val-split-fix — ⛔ THE CRITICAL PATH. Every tuned configuration in this phase was selected on a broken split.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-val-split-fix.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`.claude/outputs/gen-atom-mean-control.md` §7 — it found this and quantified it** · `.claude/outputs/gen-dynamical-read/PREREG.md` §12 · **R47**
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**, ⛔ never passed alone.
- ⚠ **JAX:** `split(key, n)` is **prefix-stable** here — ⛔ derive arm keys with `fold_in`, never by lengthening a split.
- ⚠ **`dec_recipe` defaults to `"baseline"`** (the recipe that FAILED the gate). Winner **`v1_capacity_schedule`**; select explicitly and **assert the decoder parameter hash**.
- ⚠ `n_values_realised` beside every per-factor number. **JAX cold start ~20 min; 16 GB machine — ⛔ never a full suite.**
- ⚠ **`chlu/config.py`: ONE new block, APPENDED AT EOF.** ⛔ On conflict take the whole file and re-apply your block by its `BEGIN`/`END` markers, asserting **0 deletions**.
- **Base:** `main @ 5a3c32c` or the Hub's tip · **Branch:** `agent/experiment-engineer/gen-val-split-fix` · **Worktree:** `../CHLU-gen-valfix`
- ⚠ **Budget ≈ 1.5 days.** Per-unit incremental writes, skip-if-exists.

## ⭐ DIAL DECLARATION
- **Dial:** *none — this is a REPAIR and a RE-SCORE.* ⛔ No new arm, no new mechanism, no verdict on any claim.
- **Laundering control:** ⭐ **every rival is re-selected too, by the same procedure.** ⛔ Fixing only OUR split while
  the rivals keep theirs would hand us a free win — **that is the failure mode this task must not create.**
- **Falsifies:** nothing of ours directly. ⚠ **It may move published numbers, including refusal's.** That is the point.

## The defect (measured, inherited, ⛔ not introduced)
`select_baseline_rows` takes `fit = pool[:n_fit]` and `val = pool[n_fit:…]` of a **SORTED** row-index array. At
production scale `val` therefore realises **1 `floor_hue` value and 2 `wall_hue` values** against `fit`'s **10 and
10**, at **0.0924 feature variance vs 0.2229**.
⭐⭐ **Measured consequence: the two halves of `val` disagree about the winning trial on 5 of 6 (seed, budget) cells.**
⇒ ⛔ **Every configuration selected on `val` is a coin-flip dressed as a choice.**

## The one job
**Fix the split (stratified re-select), then RE-RUN EVERY CONFIGURATION SELECTION on the fixed split and RE-SCORE.**

## ⛔ Scope — all three, no exceptions (Head ruling)
1. **Ours** — every gen arm that tunes on `val`.
2. **`knn_proto`** — its configuration selection.
3. **LACE** — including its **whitening-exactness fallback, re-run on the fixed split, ONCE.**
⭐ **Never-weaken-a-competitor applies with full force:** ⛔ if any rival comes out *weaker* after re-selection,
**stop and report** — a rival degraded by our repair is our fault.
⚠ **Inherit `GenBaselinesConfig`'s 32,256 / 512 fit/val sizes**; ⛔ never your own budget (at 1,024 fit rows
`knn_proto` fell 0.8763 → 0.8308 and LACE's whitening went rank-deficient).

## ⛔ What this BLOCKS, so you know why it is the critical path
- ⛔ **Refusal's quotable status is SUSPENDED until the rivals are re-scored.**
- ⛔ **The dynamical-read scored run WAITS** — its **ceiling row** is mis-configured until this lands.

## ⭐ Report it as an INHERITED NEGATIVE
The 5-of-6 disagreement **is the evidence**; lead with it. **State plainly which previously-published numbers move
and by how much**, and ⛔ **which are unchanged** — a re-score that quietly restates everything is unreadable.

## Acceptance
1. `PREREG.md` filed **before** re-scoring, stating the stratification rule and **what you predict will move**.
2. The stratified split **asserted balanced** — realised values per factor in `fit` vs `val`, all six factors,
   `n_values_realised` on every row, **against the broken split's 1-vs-10 as the before column**.
3. **All three selection families re-run** on the fixed split; LACE's whitening fallback re-run **once**.
4. ⭐ **A MOVED/UNCHANGED table**: every affected published number, old → new, with seed spread.
5. ⭐ **No rival weaker.** If one is, ⛔ stop and report rather than proceeding.
6. Targeted tests green with the data root set. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐ **You fixed our split and not the rivals'.** That manufactures a win and is the one outcome that would be worse
  than the defect.
- ⭐ **The stratified split is balanced on the hues but not on `scale`/`orientation`** — the two factors this phase
  turns on. **Check every factor, not the ones that were obviously broken.**
- You report movement without reporting what did **not** move.
