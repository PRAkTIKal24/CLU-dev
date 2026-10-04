# Task: gen-atom-mean-control — the like-for-like control the dynamical read must beat

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-atom-mean-control.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`.claude/outputs/gen-dynamical-read/PREREG.md` §4** · `.claude/outputs/gen-cell-harness.md` (how `knn_proto` is built and scored)
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**, ⛔ never passed alone.
- ⚠ **JAX LANDMINE, measured:** `jax_threefry_partitionable=True` makes `split(key, n)` **prefix-stable** — `split(k,2)[1]` **IS** `split(k,3)[1]`. ⛔ **Derive arm keys with `fold_in`, never by lengthening a split**, or two arms are bit-identical rather than decorrelated.
- ⚠ **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**; select explicitly and **assert the decoder parameter hash**.
- ⚠ A per-factor statistic is meaningless without **`n_values_realised`**. Report it on every row.
- ⚠ **Feature caches are non-uniform** (seed 0 `scope="full"`; seeds 1–2 `scope="used"`). **JAX cold start ~20 min; 16 GB machine — ⛔ never start a full suite.**
- ⚠ **`chlu/config.py` is shared this wave: ONE new block, APPENDED AT EOF.** ⛔ On conflict do NOT merge hunk-by-hunk — take the whole file and re-apply your block by its `BEGIN`/`END` markers, asserting **0 deletions**.
- **Base:** the Hub's tip at spawn · **Branch:** `agent/experiment-engineer/gen-atom-mean-control` · **Worktree:** `../CHLU-gen-atommean`
- ⚠ **Budget ≈ 0.5 day**, 3 seeds, local.

## ⭐ DIAL DECLARATION
- **Dial:** *none — you are building a CONTROL.* ⛔ No CLU number, no verdict.
- **Laundering control:** you **are** the control. ⭐ Your falsifier is **a control that is weaker than it should be.**
- **Falsifies:** the dynamical read's claim, if it turns out a plain lookup over the same atoms does the same job.

## The one job
Build **atom-mean lookup**: `knn_proto` **restricted to the stored atoms**, at **matched bytes**, so the dynamical
read has a like-for-like rival that stores *the same points* and simply does not settle.

## ⭐ Why this control decides the claim
The dynamical read's entire case (`PREREG.md` §2) is **a better EXPONENT at matched bytes** — interpolation beating
piecewise-constant lookup **over the same stored points**. ⭐ **Without this arm, a win could be bought by storing
atoms instead of centres rather than by settling at all.**
⛔ **Matched bytes is the whole point:** `N` atoms × `D` × 4 must equal the store budget being compared.
**Both budgets: A = 524,288 B (64 atoms/well), B = 4,194,304 B (512 atoms/well).**

## ⛔ Never weaken this rival
⭐ **Standing rule, learned the hard way:** a spoke that re-fits a rival on its own row budget silently weakens it —
at 1,024 fit rows `knn_proto` fell 0.8763 → 0.8308. ⭐ **Inherit `GenBaselinesConfig`'s 32,256 / 512, ⛔ never your own.**
⛔ If any change makes this arm weaker, **stop and report.**

## Acceptance
1. Atom-mean lookup implemented at **both budgets**, byte equality **asserted**, atoms drawn from the **same** stored
   set the landscape will use.
2. Six-factor retention ladder + read error, 3 seeds, `n_values_realised` on every row.
3. **Distinct configuration count on the 72 held-out requests.**
4. `knn_proto` (full, unrestricted) reproduced **bit-identically** as the regression check.
5. Targeted tests green with the data root set. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐ **Full `knn_proto` moved** ⇒ you changed something out of scope; the comparison is void.
- You used your own fit-row budget and the arm came out weak. That would hand the dynamical read a false win.
