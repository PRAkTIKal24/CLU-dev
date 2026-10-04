> ⛔ **RETIRED 2026-09-14, before spawn — the premise is FALSE.** R40.E scoped this against *"the current encoder
> lacks `scale`/`orientation`"*. `gen-loss-attribution` measured the shipped frozen `recon` encoder at **`scale`
> 0.9352 ± 0.0058** and **`orientation` 0.9915 ± 0.0018** (3/3 seeds) — it **already clears R40.E's ≥0.9 bar**.
> ⭐ Advisor concurs: the candidate-8 headline swap is **MOOT**; recon already shipped. ⛔ **Do not spawn.**
> ⚠ **Not fully refuted:** R40.E's *other* clause — **swap within the family** — was never measured, and the VAE
> comparison arm is **optional, later**. If either is revived, this file is the starting point.

# Task: gen-encoder-pretrain — an encoder that KEEPS scale and orientation

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-encoder-pretrain.md` + artefacts under `.claude/outputs/gen-encoder-pretrain/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R40 (E in full, and ⛔ B: the compose path is a quantiser), R1 (reconstruction is the mandatory pretext and WHY), R18 (⛔ what a changed encoder VOIDS), R31 + R34 (the loss profile)** · `.claude/outputs/gen-loss-attribution.md` **if it exists**.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's tip at spawn · **Branch:** `agent/experiment-engineer/gen-encoder-pretrain` · **Worktree:** fresh (`../CHLU-gen-pretrain`).
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report passed / skipped / failed, ⛔ never passed alone.
- ⚠ **Budget ≈ 6 h, REDUCED SCALE, local.** ⛔ No cluster. Full scale is a later step behind the smoke. ⭐ Per-variant incremental writes, skip-if-exists.

## ⭐ DIAL DECLARATION
- **Dial:** *the encoder's pretext.* ⛔ **The store is NOT touched. The encoder stays FROZEN for phase 1** — you are
  producing a candidate cache, not changing the pipeline.
- **Laundering control:** the **current `recon` encoder**, same probe, same rows, same seeds — a full row on every table.
- **Falsifies:** ⭐ **`scale` + `orientation` probe ≥ ~0.9 on SEEN data, or swap within the family, BEFORE the store is
  touched.** ⛔ If no variant clears it, **report that and stop** — do not proceed to the store on a weak cache.

## The one job
Train **self-supervised reconstruction** encoders at reduced scale and find one whose features **retain `scale` and
`orientation`**, measured by the same probe the attribution spoke uses.

## The arms
- ⭐ **Headline: a plain AE / MAE.** Pixel reconstruction, no labels.
- **Declared comparison arm: a disentangling VAE.** Reported beside the headline, never instead of it.
- ⛔ **AUGMENTATION-CONTRASTIVE IS EXCLUDED, and the reason is not stylistic:** it trains **invariance to exactly
  `scale` and `orientation`** — the two factors we are trying to keep. ⛔ Do not add it "for completeness".

## ⭐ What is already known — do not re-derive it
- **The current encoder uses `pretext="recon"`**: the loss is `mean((recon − xb)²)` over images only, `data = (X,)`.
  ⭐ **It sees NO labels.** Hub-verified in `gen_bridge.py`. The label-taking `cls` path is the diagnostic arm and was
  **not** used for any shipped run. ⇒ ⛔ **The zero-supervision claim on the generative path is intact; you are not
  fixing a supervision leak.** You are fixing a **representation** problem.
- The loss is **global**, not corner-local: `scale` **0.229** (chance 0.125) and `orientation` **0.236** (chance
  0.067) on the spread grid, from an independent instrument.
- ⭐ **The evaluation probe still trains on labels — that is measurement, and it stays** (R1).

## ⛔ WHAT A NEW ENCODER VOIDS — state it in your report, it is not optional
A changed encoder changes the feature cache, and **every phase-1 number is computed on that cache.**
⇒ ⛔ **This is a NEW configuration BESIDE the frozen one, never a replacement.** Write to fresh paths; ⛔ overwrite,
re-fit or invalidate nothing. **List every number that must be re-measured if this cache is ever promoted** —
⭐ **refusal is RE-MEASURED, never assumed**, and ⭐ **`knn_proto` gets the same features** so the comparison stays fair.

## ⛔ Live traps
1. ⚠ **A per-factor statistic is meaningless without `n_values_realised`.** Report it beside every number.
2. ⭐ **Measure on the SPREAD grid**, with the corner reported separately. A corner-only number will mislead.
3. ⚠ **Feature caches are non-uniform** (seed 0 `scope="full"`; seeds 1–2 `scope="used"`).
4. **Reduced scale is a declared axis, not a caveat.** State rows, steps and parameter count for every variant, and
   ⛔ **never compare a reduced-scale variant against a full-scale baseline** without saying so in the same sentence.
5. **JAX cold start ~20 min. 16 GB machine** — ⛔ never start a full suite; ⛔ never hold two caches.

## Files you own
`chlu/core/gen_encoders.py` (new) · `chlu/experiments/exp_gen_pretrain.py` (new) ·
`tests/test_gen_pretrain.py` (new) · `chlu/config.py` — **ONE new block, APPENDED AT EOF.**
⛔ `chlu/core/gen_bridge.py` is **read-only** — ⭐ do **not** modify the existing encoder path; the current cache must
stay bit-identical. ⛔ Nothing under `gen_store.py`, `gen_compose.py`, `gen_dream.py`, `chlu/eval/`, `scripts/csf3/`.

## Acceptance
1. `PREREG.md` filed **before** the first probe, naming the ≥0.9 bar and the variants to be tried.
2. AE/MAE headline + VAE comparison arm, ⛔ no contrastive arm, reduced scale stated for each.
3. `scale` + `orientation` probed on **seen** data, same probe as the attribution spoke, 3 seeds, spread grid and
   corner reported separately, `n_values_realised` on every row.
4. The current `recon` encoder as a full control row.
5. **PASS/FAIL against the ≥0.9 bar stated plainly.** ⛔ On FAIL: report and stop; do not touch the store.
6. The bit-identity assertion that the existing encoder path did not change.
7. The list of numbers that would need re-measuring if this cache were promoted.
8. Targeted tests green with the data root set; passed / skipped / failed, every skip named. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐ **A variant clears the bar on the corner but not on the spread grid.** The corner is 72 items; the grid is the
  real test, and the loss we are chasing is global.
- ⭐ **You beat the baseline by training longer, not differently.** Report compute for every arm; a compute win is
  not a representation win.
- You modified the existing encoder path and the current cache moved. That voids the phase.
