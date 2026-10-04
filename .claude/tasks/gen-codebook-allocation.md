# Task: gen-codebook-allocation — spend codewords where the distortion is (D6)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-codebook-allocation.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`.claude/outputs/design-notes/dynamical-read.md` §0.5 (the scaling law)** · `.claude/outputs/gen-variance-equalisation.md` (effective dimension, Leg 1) · **R45**
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**, ⛔ never passed alone.
- ⚠ **JAX LANDMINE, measured:** `jax_threefry_partitionable=True` makes `split(key, n)` **prefix-stable** — `split(k,2)[1]` **IS** `split(k,3)[1]`. ⛔ **Derive arm keys with `fold_in`, never by lengthening a split**, or two arms are bit-identical rather than decorrelated.
- ⚠ **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**; select explicitly and **assert the decoder parameter hash**.
- ⚠ A per-factor statistic is meaningless without **`n_values_realised`**. Report it on every row.
- ⚠ **Feature caches are non-uniform** (seed 0 `scope="full"`; seeds 1–2 `scope="used"`). **JAX cold start ~20 min; 16 GB machine — ⛔ never start a full suite.**
- ⚠ **`chlu/config.py` is shared this wave: ONE new block, APPENDED AT EOF.** ⛔ On conflict do NOT merge hunk-by-hunk — take the whole file and re-apply your block by its `BEGIN`/`END` markers, asserting **0 deletions**.
- **Base:** the Hub's tip at spawn · **Branch:** `agent/experiment-engineer/gen-codebook-allocation` · **Worktree:** `../CHLU-gen-alloc`
- ⚠ **Budget ≈ 0.5 day**, 3 seeds, local.

## ⭐ DIAL DECLARATION
- **Dial:** *codeword allocation across wells, at CONSTANT TOTAL BYTES.*
- **Laundering control:** ⭐ **uniform `M = 64` at the same total bytes** — the shipped arm. ⛔ Any gain must be over it.
- **Falsifies:** the scaling law's actionability. ⛔ **This is a CONTROL ARM for the dynamical read, ⛔ NOT a gate on it.**

## The one job
Allocate per-well `M_k` **by measured effective dimension** instead of uniformly, **at constant total bytes**, and
report the six-factor retention ladder and read error against the uniform arm.

## ⭐ The law you are testing
**ε ≈ M^(−2/PR)** with per-well `PR` = **3.12 … 13.63** (measured). Equal-distortion allocation sets
`M_k ∝ ε^(−PR_k/2)`. ⭐ **Constant total bytes ⇒ no byte-ledger renegotiation**, which is the whole appeal.
⛔ **It CANNOT save the high-`PR` wells** — at ε = 0.10 a `PR` = 13.63 well needs ~6.5 million codewords. **Say so;
do not present a partial fix as a fix.**

## Acceptance
1. `PREREG.md` filed **before** measuring, with the allocation rule and the predicted per-well distortions.
2. Allocation derived from **measured** per-well `PR`, ⛔ not assumed; total bytes **asserted equal** to the uniform arm.
3. Six-factor ladder + **TOTAL-RETAINED** line, allocated vs uniform, 3 seeds, `n_values_realised` on every row.
4. **Distinct configuration count on the 72 held-out requests**, both arms.
5. Measured per-well distortion against the law's prediction — ⭐ **report the fit, not just the win.**
6. Targeted tests green with the data root set. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐ **Total bytes are not exactly equal** — then it is a bigger store, not an allocation.
- The measured distortions do not follow `M^(−2/PR)` ⇒ the law does not hold per well and §0.5 needs revisiting.
