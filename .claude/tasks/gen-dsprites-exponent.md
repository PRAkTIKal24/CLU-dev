# Task: gen-dsprites-exponent — is the codebook exponent set by the FACTOR MANIFOLD?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-dsprites-exponent.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`.claude/outputs/gen-dynamical-read/PREREG.md` §11b — the prediction, filed** · `.claude/outputs/gen-codebook-allocation.md` (the Shapes3D exponent and its fit) · `.claude/outputs/gen-bridge-dsprites.md` (the dSprites bridge, ⭐ **already gated and PASSING**) · **R47**
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**, ⛔ never passed alone.
- ⚠ **JAX:** `split(key, n)` is **prefix-stable** here — ⛔ derive arm keys with `fold_in`, never by lengthening a split.
- ⚠ **`dec_recipe` defaults to `"baseline"`** (the recipe that FAILED the gate). Winner **`v1_capacity_schedule`**; select explicitly and **assert the decoder parameter hash**.
- ⚠ `n_values_realised` beside every per-factor number. **JAX cold start ~20 min; 16 GB machine — ⛔ never a full suite.**
- ⚠ **`chlu/config.py`: ONE new block, APPENDED AT EOF.** ⛔ On conflict take the whole file and re-apply your block by its `BEGIN`/`END` markers, asserting **0 deletions**.
- **Base:** `main @ 5a3c32c` or the Hub's tip · **Branch:** `agent/experiment-engineer/gen-dsprites-exponent` · **Worktree:** `../CHLU-gen-dspexp`
- ⚠ **Budget ≈ 1 day**, 3 seeds, local. ⭐ Independent of `gen-val-split-fix` — ⛔ it tunes nothing, so the broken split does not touch it.

## ⭐ DIAL DECLARATION
- **Dial:** *the dataset*, as a test of **why** the exponent is what it is. ⛔ No arm, no CLU claim, no verdict.
- **Laundering control:** ⭐ **the identical sweep and the identical fit on Shapes3D**, re-run by you — ⛔ do not
  transcribe `−0.2631` from a report; three errata on this programme came from transcribed table values.
- **Falsifies:** the factor-manifold account. ⛔ **If dSprites is not steeper, the account is REFUTED** and −0.26's
  universality needs a different explanation. ⭐ That is a result, not a failure.

## The one job
Run the **same codebook sweep and the same exponent fit** on **dSprites** and compare its exponent to Shapes3D's.

## ⭐ The pre-registered prediction (PREREG §11b — ⛔ do not edit it, it is already filed)
- ⭐ **PRIMARY, directional: the dSprites exponent is STEEPER than Shapes3D's −0.2631, non-overlapping across 3 seeds.**
- **Companions, both filed so neither can be chosen afterwards:** **−0.400** (pure factor count, `−2/5`) and
  **−0.316** (Shapes3D's ×1.267 offset carried).
⚠ **The pure factor-count model already mispredicts Shapes3D** (−0.333 predicted vs −0.2631 measured, implied
`d` = 7.60). ⭐ **Report which companion is closer; ⛔ do not declare either "the prediction".**

## ⛔ Traps specific to dSprites — every one silently produces a wrong split
1. ⭐⭐ **The colour column is DROPPED at load** (`latent_values[:, 1:]`); the index constants `shp,scl,rot,tx,ty = 0..4`
   are valid **only after** that slice. ⛔ Applying them raw **yields an empty test set**.
2. ⭐⭐ **The angle constants are SWAPPED relative to their names and the filter only works because they are**
   (`a120 = 4π/3`, `a240 = 2π/3`). ⛔ **It LOOKS like a bug. "Fixing" it inverts the held-out band.**
3. **Realised counts must be MEASURED**: 737,280 total / **6,292** test / 730,988 train. ⛔ Check before fitting anything.
4. ⛔ **The paper's prose contradicts the code. The CODE wins.**
5. ⚠ **dSprites' probe is NOT saturated** — it reads `orientation` on **real** images at **0.15**. ⛔ Shapes3D's
   "probe reads real images at 1.0000" does **not** carry across; ⛔ do not use an aggregate ceiling on this dataset.

## Acceptance
1. `PREREG.md` filed **before** measuring, restating §11b's primary + both companions verbatim.
2. Realised split counts measured and checked against 6,292.
3. ⭐ **Shapes3D re-run by you with the identical sweep and fit**, its exponent reported beside dSprites'.
4. Exponent + CI for both datasets, 3 seeds, **the directional claim answered yes/no in those words**.
5. **Which companion is closer**, ⛔ without promoting either to "the prediction".
6. Targeted tests green with the data root set. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐ **You compared your dSprites number to a TRANSCRIBED Shapes3D −0.2631** instead of re-running the fit. The two
  must come from the same code path or the comparison is meaningless.
- ⭐ Your dSprites sweep covers a different `M` range than Shapes3D's (`M ∈ [4, 186]`). **Exponents are only
  comparable over comparable ranges** — match them, and say so.
