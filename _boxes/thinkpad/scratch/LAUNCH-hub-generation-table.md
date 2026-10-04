# ADDENDUM — `hub/generation-table`  ⭐ THE HEADLINE TARGET
**Read `LAUNCH-COMMON.md` first.** Then `.claude/tasks/hub-generation-table.md` is your task.

- **Slug:** `hub-generation-table` · **Branch:** `hub/generation-table`
- **Worktree:**
  `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-generation-table -b hub/generation-table HEAD`
  ⛔ NOT `ae907466` — see COMMON §1a.
- **Core budget: `OMP_NUM_THREADS=5` / `MKL_NUM_THREADS=5` / `OPENBLAS_NUM_THREADS=5`.**

## ⛔⛔ Three things in your task file are designed to catch you. Do not skim them.
1. ⛔ **Roughly the bottom third of the table is FREE.** A near-constant output — **2.67 distinct tuples
   of 72** — scores **0.3272** and TIES `lace`. composition-B aggregate is an unweighted mean over six
   per-factor accuracies and has a high free floor.
   ⭐ **The honest dynamic range is ~0.33 → 0.865, not 0 → 0.865.** A mid-0.4 is **barely above free**,
   ⛔ not halfway home.
   ⭐⭐ **Report the distinct-tuple count beside EVERY aggregate you quote. Every one, without exception.**
2. ⛔ **`x0`'s 0.6852 is at EIGHT TIMES the bytes.** At matched bytes it scores **0.5008 and LOSES to
   `clu`**, and fails the §C floor against the atom-mean lookup (2/3 seeds, intervals overlap).
   ⛔ **Never quote 0.6852 without its byte multiple on the same line.**
3. ⚠ **`knn_proto` renders 56.00 ± 3.74 of 72, ⛔ NOT 72/72.** An earlier docstring claimed 72/72 and it
   was never measured. ⚠ And `lace` **0.3117** is the CELL harness row — ⛔ **0.3241 is the BASELINES
   instrument at 688,128 B and is NOT comparable on this table.** Both are real; they are different
   instruments.

## Your starting position is a measured negative, not a blank page
⛔ **RULED (Head): you start from "is the store earning its bytes", NOT from "beat 0.5949".**
At matched bytes **the store is currently DECORATIVE** — that is the measured starting position.
⭐ Clearing it is a **precondition, not an achievement**, and ⛔ it is not sufficient: you must also beat
the rivals and the published numbers.

⭐ **Ceiling: `knn_proto` 0.8650 ± 0.0122.** `m = 0.020`. **WIN needs Δ > m.** `0 < Δ ≤ m` with 3/3 same
sign is **TIE (favourable)**. ⛔ A tie counts **only** against the ceiling — every like-for-like row must
be beaten **outright**.

## Mechanism facts you INHERIT — ⛔ measured; do not re-derive, do not burn hours reconfirming
- ⭐⭐ **The launch beats the settled state by a mile: 0.6852 → 0.2114 after 2000 steps.** The store,
  address book, request path and decoder all work.
  ⛔ **THE DYNAMICS ARE THE DEFECT** — descending the energy *loses* accuracy, so the landscape's minima
  are not at task-correct states. ⇒ ⛔ **"more steps" makes it worse. Do not spend your 12 h there.**
- ⭐ **Steering is real and attributable:** request-off (β = 0, same chain) collapses to **1 distinct tuple
  of 72** and to 0.0000 at rung B. The request energy carries **100 %** of the distinctness — it is just
  small.
- ⭐ **Placement is NOT the lever:** learned placement nulled against a D-matched random floor
  (0.0019 against a 0.10 bar, `fraction_moved` 0.75 — a real null). The **inherited** ward/correlation
  grouping already beats that floor by 0.0975, about 50×. **The wells are already in the right place.**
- ⭐⭐ **The codebook is the largest identified loss in the programme: it costs 90.5 % of `scale` and
  100 % of `orientation`**, while the partition costs 9.5 % and 0 %.
  ⭐ That is the biggest signposted opportunity you have been handed.

## Your three falsifiers — answer all three, in these words
1. Beats `store_removed`, `ebm_compose`, `lace`, `clu` **and** the matched-bytes atom-mean lookup
   outright, beyond seed spread — and **WIN or TIE against `knn_proto`** by the rule above.
2. ⭐ **Clears the constant-output floor:** distinct factor tuples **well above 2.67/72**, reported beside
   the aggregate.
3. ⭐ **The store is load-bearing:** `store_removed` collapses your arm. ⛔ If deleting the store barely
   moves your number, you have built a decoration and ⛔ it does not count.

⭐ Because you are the headline, the pressure to report a number is highest here. COMMON §1b applies with
full force: **stopping at 12 h with an honest null is a success; a manufactured win is the one
unrecoverable failure.**
