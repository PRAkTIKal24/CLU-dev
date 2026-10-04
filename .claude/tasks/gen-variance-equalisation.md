# Task: gen-variance-equalisation — does k-means drop `scale` because it is LOW-VARIANCE?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-variance-equalisation.md` + artefacts under `.claude/outputs/gen-variance-equalisation/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R40.B (⛔ the compose path is a product quantiser + count table), R40.C (the dynamical read this feeds), R42 (the attribution result you are building on)** · ⭐ **`.claude/outputs/gen-loss-attribution.md` in full — it located the loss and explicitly did NOT test the cause.**
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 81ba81b` (or the Hub's tip at spawn) · **Branch:** `agent/experiment-engineer/gen-variance-equalisation` · **Worktree:** fresh (`../CHLU-gen-varianceeq`).
- **Env:** main venv. ⭐ pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report passed / skipped / failed, ⛔ never passed alone.
- ⚠ **Budget ≈ 1 day, local, 3 seeds.** ⭐ Per-seed incremental writes, skip-if-exists.

## ⭐ DIAL DECLARATION
- **Dial:** *feature scaling before the store.* ⛔ The encoder, the decoder, the partition and `K` are all UNCHANGED.
- **Laundering control:** ⭐ **`knn_proto` must be BIT-IDENTICAL.** It never sees the transform, so if its number moves
  **you have changed something you did not intend** — stop and report.
- **Falsifiers, pre-registered:** ⭐ **(a)** the `scale` + `orientation` probe **on the store code** rises above the
  unwhitened baseline **beyond seed spread**; **(b)** the **distinct configuration count rises**. ⛔ If neither moves,
  the variance hypothesis is **refuted** and that is the result.

## Why this exists — and what is NOT yet known
`gen-loss-attribution` established **where** the loss happens: the **well partition is exonerated** (it costs 9.5 % of
`scale` and **0 %** of `orientation`) and the **k-means codebook takes the rest**. The encoder is fine — it carries
`scale` **0.9352** and `orientation` **0.9915**.
⛔ **But it measured WHERE, not WHY. Its own report says the variance claim is untested.** ⭐ **This spoke tests it.**
**The hypothesis:** k-means minimises squared error, so it spends its codewords on **high-variance** directions;
if `scale` and `orientation` live in **low-variance** directions of these features, they are quantised away — not
because the information is absent, but because the objective does not value it.

## Leg 1 — ⛔ TEST THE PREMISE BEFORE YOU ACT ON IT
**Measure variance-explained per factor**, per well, on the frozen features:
- project each well's sub-vector onto its **principal axes**; report, per factor, **where its discriminative signal
  sits in the variance spectrum** (e.g. the variance rank of the directions that carry it).
- ⭐ **Report a variance-explained-per-factor row** for all six factors, 3 seeds.
⭐ **This is the row the Advisor asked for, and it decides whether Leg 2 is even worth running.**
⛔ **If `scale`/`orientation` do NOT sit in low-variance directions, the hypothesis is dead — report that and STOP.**
Do not whiten anyway to see what happens.

## Leg 2 — equalise, then re-fit and re-probe
⭐ **ORDER IS RULED AND IT MATTERS: fit the correlation grouping on UNWHITENED features FIRST, then whiten.**
⛔ Whitening before grouping would change the partition — and the partition is **already exonerated**, so you would be
moving the one part that is not broken.
- ⭐ **PRIMARY ARM: full PCA whitening WITHIN each well's own sub-space** (per-well, ⛔ not across all of `D`).
- **SECONDARY ARM: diagonal standardisation** (per-channel mean / std).
- ⭐ **Invertible, and the decoder must see its usual space:** map the read **back** before decoding, **assert the
  round-trip** (`inverse(forward(x)) ≈ x`), report the worst error. ⛔ A decoder fed whitened vectors measures the
  transform, not the store.
- ⛔ Encoder, decoder, partition and `K` all unchanged. Only the metric k-means sees.

## ⭐ Leg 3 — THE SIX-FACTOR GRADED PROFILE, BEFORE AND AFTER, WITH A TOTAL LINE
⛔ Not just `scale` and `orientation`: **all six factors, retention before and after**, plus a **TOTAL-RETAINED line**.
⭐ **The question is TRADE vs REDUCTION** — did whitening *move* retention off the hues onto `scale`/`orientation`, or
did it *add* retention overall?
Baseline ladder (from `gen-loss-attribution`; ⛔ **re-run it yourself, do not transcribe** — transcription from
rendered tables has caused three errata here):
floor_hue **0.997** · wall_hue **0.995** · object_hue **0.665** · shape **0.555** · orientation **0.243** · scale **0.225**.

## ⭐⭐ Leg 4 — THE PER-WELL CAPACITY BOUND, PRE-REGISTERED BEFORE YOU MEASURE
For each well, state **how many factor-value combinations it must resolve** against **`M = 64`** modes.
⭐ **Whitening cannot beat this bound. It can only choose which factors spend the modes.**
Hub arithmetic, to be checked and refined per well:
- aggregate `K·log₂M` = **30 bits** vs `log₂(480,000)` = **18.87 bits** ⇒ ⭐ **the code is SUFFICIENT in aggregate**
  (this is exactly why *"30 bits"* is **not** the cause and ⛔ may not be written as one);
- ⛔ **per well, `M = 64` is already exceeded by two hues alone (10×10 = 100), by hue × orientation (150), and
  overwhelmingly by all six (480,000).**
⭐ **PRE-REGISTERED READING: if the TOTAL-RETAINED line is flat and only the DISTRIBUTION moves, CAPACITY is the
binding constraint, not the metric.** ⛔ That is a first-class result, not a null — and it sends the repair to `M`
(or to a different representation), never to the metric.

## ⛔⛔ THE BYTE LEDGER — Hub-computed from the REAL well sizes
Wells are **[303, 305, 573, 501, 366]** channels (sum `D` = 2048), `M` = 64, store budget **524,288 B**.

| transform | bytes | vs the store budget |
|---|---|---|
| **diagonal standardisation** (secondary) | **16,384** | 3.1 % |
| ⭐ **per-well PCA whitening** (primary) — `Σ d_k² × 4` + means | **3,600,672** | ⛔ **6.87×** |
| *(full-`D` whitening — ⛔ not used, shown for scale)* | *16,777,216* | *32×* |

⭐⭐ **ASK THIS FIRST — IT MAY COST NOTHING.** Is the transform **fit-time only** or **deployed**?
- Whitening used **only to fit** the codebook, with centres **stored un-whitened** ⇒ **fit-time only, 0 deployed
  bytes**, and the arm is byte-free.
- If the **read** must invert it to reach the decoder ⇒ **deployed**, and it costs the table above.
⭐ **State plainly which one you built and price it accordingly.** ⛔ A deployed 6.87× transform reported as though it
were free is the easiest way for this arm to look like a win when it is a much larger model.

## ⛔ Live traps
1. **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**;
   select explicitly and assert the decoder parameter hash.
2. ⚠ **A per-factor statistic is meaningless without `n_values_realised`.** Report it on every row.
3. ⭐ **Measure on the SPREAD grid**, corner reported separately — the loss is global.
4. ⭐ **Use the same probe as `gen-loss-attribution`**, same hyper-parameters and PRNG convention, or the comparison
   to its baseline is meaningless. **Re-run its baseline yourself** rather than transcribing numbers from its report —
   ⚠ transcribing from rendered tables has caused three errata on this programme.
5. ⚠ **Feature caches are non-uniform** (seed 0 `scope="full"`; seeds 1–2 `scope="used"`).
6. **JAX cold start ~20 min. 16 GB machine** — ⛔ never start a full suite.

## Files you own
`chlu/core/gen_whiten.py` (new) · `chlu/experiments/exp_gen_variance_eq.py` (new) ·
`tests/test_gen_whiten.py` (new) · `chlu/config.py` — **ONE new block, APPENDED AT EOF.**
⛔ Read-only on `gen_store.py`, `gen_compose.py`, `gen_bridge.py`, `gen_dream.py`, `chlu/eval/`, `chlu/data/`,
`scripts/csf3/`. ⛔ **Do not modify the existing store-fitting path** — the unwhitened baseline must stay reproducible.

## Acceptance
1. `PREREG.md` filed **before** any measurement, containing both falsifiers verbatim.
2. **Leg 1 variance-explained-per-factor table**, all six factors, per well, 3 seeds. ⛔ On a dead premise: stop here.
2b. ⭐ Grouping fitted on **UNWHITENED** features, asserted; both arms run (per-well PCA primary, diagonal secondary).
2c. ⭐ **Six-factor graded profile before AND after, plus the TOTAL-RETAINED line** — trade vs reduction, in words.
2d. ⭐ **Per-well capacity bound pre-registered BEFORE measuring**; the flat-total reading stated if it applies.
2e. ⭐ **Transform declared fit-time-only or deployed**, with its byte cost.
3. Round-trip assertion for the transform, with the worst observed error.
4. **Byte ledger for the transform**, in the table, in bytes.
5. Probe on the store code, whitened vs unwhitened, **same probe**, 3 seeds, with seed spread — falsifier (a)
   answered yes/no in those words.
6. **Distinct configuration counts**, whitened vs unwhitened — falsifier (b) answered yes/no.
7. ⭐ **`knn_proto` reproduced BIT-IDENTICALLY.** ⛔ If it moved, stop and report.
8. Targeted tests green with the data root set; passed / skipped / failed, every skip named. ⛔ No full suite.

## What would prove this spoke wrong
- ⭐ **`knn_proto` moved.** The transform escaped its scope and the comparison is void.
- ⭐ **The probe rises but the distinct count does not** (or vice versa). Report both; ⛔ one without the other is not
  the pre-registered pass, and the two measure different things.
- You whitened without checking Leg 1, and now cannot say whether a null means "variance was not the cause" or
  "the premise never held".
