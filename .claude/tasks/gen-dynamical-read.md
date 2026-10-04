# Task: gen-dynamical-read — BUILD the landscape over written atoms. ⛔ Build only; the scored run is gated.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-dynamical-read.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`.claude/outputs/gen-dynamical-read/PREREG.md` — BINDING, and you do not edit it** · `.claude/outputs/design-notes/dynamical-read.md` (§0.5 the scaling law, §1 dreaming's autopsy) · **R40.C, R45 in `.claude/tasks/gen-clu-phase1-rulings.md`**
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**, ⛔ never passed alone.
- ⚠ **JAX LANDMINE, measured:** `jax_threefry_partitionable=True` makes `split(key, n)` **prefix-stable** — `split(k,2)[1]` **IS** `split(k,3)[1]`. ⛔ **Derive arm keys with `fold_in`, never by lengthening a split**, or two arms are bit-identical rather than decorrelated.
- ⚠ **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**; select explicitly and **assert the decoder parameter hash**.
- ⚠ A per-factor statistic is meaningless without **`n_values_realised`**. Report it on every row.
- ⚠ **Feature caches are non-uniform** (seed 0 `scope="full"`; seeds 1–2 `scope="used"`). **JAX cold start ~20 min; 16 GB machine — ⛔ never start a full suite.**
- ⚠ **`chlu/config.py` is shared this wave: ONE new block, APPENDED AT EOF.** ⛔ On conflict do NOT merge hunk-by-hunk — take the whole file and re-apply your block by its `BEGIN`/`END` markers, asserting **0 deletions**.
- **Base:** the Hub's tip at spawn · **Branch:** `agent/experiment-engineer/gen-dynamical-read` · **Worktree:** `../CHLU-gen-dynread`
- ⚠ **Budget ≈ 2 days.** Per-unit incremental writes, skip-if-exists.

## ⭐ DIAL DECLARATION
- **Dial:** *the read* — continuous settling over stored atoms, replacing codebook copy-back.
- **Laundering control:** ⭐ **request-off on the SAME chain.** Steering-attributable gain is the falsifier — ⛔ this is
  dreaming's exact lesson: its clamp's apparent gain was taken entirely by the coupling-off control.
- **Falsifies:** the three falsifiers in `PREREG.md` §5. ⛔ **You do not render a verdict** — the scored run does.

## The one job
Implement `request → clamp + request ENERGY → launch → settle (per-well landscape over atoms + low-rank coupling) →
decode settled state`, and **smoke it locally at tiny scale.** ⛔ **No scored run, no 3-seed cell, no verdict.**

## ⛔ Non-negotiables, from the rulings
1. ⭐ **Atoms = stored TRAIN DATA POINTS. ⛔ NEVER the k-means centres** — per-value weights over centres are the old
   flat count table.
2. ⭐ **The request enters as an ENERGY TERM** — one expert per requested term over the well's atoms, **declared
   strength** — ⛔ not only as an initialisation. **Wire the request-off control on the same chain from the start.**
3. **Coupling: LOW-RANK** (or shared projection), **label-free, fitted on train features only**, rank from
   `PREREG.md` §3 (**r = 16** at budget A, **r = 128** at budget B). **Coupling-off control mandatory.**
4. ⭐ **Bypass guarantee: the request-only assertion covers launch + settle + read AS ONE PATH.** Assert no feature
   tensor enters any of the three; **show the assertion FAILING on a deliberately bypassed variant.**
   ⛔ `soft_read(store, X, τ)` takes FEATURES and voids it.
5. **1000 steps.** ⭐ **Log the energy trace EVERY step, acceptance, and per-step displacement FROM STEP 1.** A chain
   still descending at 1000 is **NOT CONVERGED**, ⛔ never a null.
6. ⛔ **The declared channel is the store's FULL WIDTH IN FLOATS** (`knn_proto` parity). ⛔ **No mention of 30 bits.**

## Acceptance
1. Tiny-scale **local smoke** of the full path end to end, into a fresh workspace. State the scale.
2. Bypass assertion passes on the real path and is **demonstrated failing** on a bypassed variant.
3. Request-off and coupling-off controls wired and exercised in the smoke.
4. Energy trace / acceptance / per-step displacement emitted; convergence fraction reported at smoke scale.
5. Byte ledger printed for both budgets and matching `PREREG.md` §3 exactly.
6. Targeted tests green with the data root set; passed/skipped/failed, every skip named. ⛔ No full suite, ⛔ no scored run.

## What would prove this spoke wrong
- ⭐ **The bypass assertion passes on a variant you deliberately broke** — then it is decoration.
- ⭐ **Request-off already gives the same trajectory at smoke scale** — say so immediately; that is dreaming again.
- You built the landscape over the k-means centres. That inherits the codebook loss and voids the arm.
