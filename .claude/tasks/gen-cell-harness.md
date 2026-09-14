# Task: gen-cell-harness — the generation harness end to end, and the headline table

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-cell-harness.md` + artefacts under `.claude/outputs/gen-cell-harness/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R4 + R9 (LACE, and every trap in copying its constants), R8 (request embedding, declared once), R20 (the 30-bit design constant), R26.3, R27 (all four), R29 in full** · `.claude/outputs/gen-arms-a7-wiring.md` · `.claude/outputs/gen-external-classifier.md` · `.claude/outputs/gen-read-compose.md`.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's merged tip at spawn time (the Head will name it) · **Branch:** `agent/experiment-engineer/gen-cell-harness` · **Worktree:** fresh (`../CHLU-gen-harness`).
- **Env:** main venv, ⛔ no `uv sync` in the worktree. ⛔ CSF3 unreachable (2FA) — local CPU only; emit submit lines and report every cluster item **NOT RUN**.
- ⚠ **Budget ≈ 6 h.** ⭐ Write per (arm, seed) as it completes and skip any unit whose artefact exists. ⛔ Never hold to the end.

## ⛔ PRECONDITION — mechanical, both files, content-checked. Verify BEFORE any other work.
1. `.claude/outputs/gen-external-classifier/classifier_gate.json` exists **and** its top-level `"gate"` is `"PASS"`.
2. `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` exists **and** its top-level `"status"` is `"complete"`.
3. ⭐ From `gen-arms-a7-wiring.md` §6, verify that on **both** non-exact seeds (0 and 2) **both whitening legs were
   run** — the pre-registered one and its nearest-exact `_wexact` fallback — and that the quoted number on each axis
   is **the stronger of the two**.
   ⛔ **The check is "both legs run, stronger quoted", NOT "the quoted config is exact."** R27.3's purpose is
   never-weaken-a-competitor, so where the **non-exact** leg is stronger — which it is for LACE's **composition**, the
   axis you are measuring — quoting it is **correct and required**. ⭐ **LACE's composition number stays as published
   (`ode` 0.6797 ± 0.0207); ⛔ you may NOT use R27.3 to swap it.**
   ⚠ Carry this residue into your report, unresolved: on seeds 0 and 2 the pre-registered leg still ran with the
   prior-score omission unlicensed, and R27.3's second route (the general form **with** the prior score) is **NOT
   RUN** — it needs a new drift term in `sample_ode`, a sampler change nobody has made. ⛔ Do not make it here.
⛔ If any of the three fails, **do nothing else and report blocked.** ⛔ Never "start anyway and fix later".

## ⭐ DIAL DECLARATION
- **Dial:** *composition, end to end, scored on IMAGES.*
- **Laundering control:** ⭐ the **store-removed** arm, on every row. It is not a footnote; it is the row that says whether any number came from the store.
- **Falsifies:** the headline table itself. ⭐ **Composition was RELEASED as a claim in R29 — kill rule (ii) already fired.** You are not trying to win it back. You are measuring, honestly, where we sit against three rivals on a metric nobody in this phase has used yet.

## The one job
Run `request → launch read → codes → frozen decoder → image → external classifier` on held-out combinations, across
**CLU · store-removed · knn_proto · LACE · energy-based composition**, ≥3 seeds, and produce the table.

## Non-negotiables
- ⛔ **Every arm decodes through the ONE shared frozen decoder** — assert its parameter hash for every arm, every seed,
  before any number is recorded (R9.2). ⛔ No arm substitutes its own.
- ⛔ **Every arm is scored by the SAME external classifier**, the one gated by `gen-external-classifier`.
- **Request sets are single-sourced** from `gen-data-splits` / `c_invalid.json` and are byte-identical across arms (R8).
  ⛔ Never regenerate them.
- ⛔ **CLU's mixture weights come from the REQUEST**, never from features. `soft_read(store, X, tau)` takes **FEATURES**
  and using it voids the structural no-bypass guarantee. **Assert the CLU path touches no feature tensor**, exactly as
  `gen-read-compose` did.
- **Every arm gets the same tuning-trial budget, and it is reported.**
- ⛔ **Byte ledger in the table, not in a footnote.** Store **524,288 B** · addressing head **74,240 B** · prototype
  table **475,136 B**. ⚠ The shipped kNN numbers were run at the retired **688,128 B** budget — a **~31 % advantage**
  it should not have. ⭐ That is the **conservative** direction and it is **declared, never silent.**
- **≥3 seeds on every number, with spread.** ⚠ B is 72 combinations — **say plainly how much of the spread is sampler
  noise versus arm difference.** Do not let 72 items carry a confident claim.

## ⭐ The three things this table must show that no previous table could
1. **The image-space number**, from an instrument that is not the decoder's own family.
2. ⭐ **The distinct-output count per arm, beside every accuracy.** From disk: on B the read renders **9.0 ± 0.8**
   distinct vectors of 72, while `clu_wide` and `proto` render **72/72**. ⛔ An accuracy without its distinct count
   hides the single most important fact we know about this mechanism.
3. **The store-removed row**, which on B currently reads **0.0000** on shape and **1.0** distinct vector of 72.

## ⛔ Honesty rules, restated because this is where they get bent
- ⛔ **Capability 1 — composing unseen-but-valid combinations — is NEVER presented as novel on its own.** LACE owned it
  in 2021 (R4, R14.6). If your report drifts toward "we compose", rewrite it.
- ⛔ **Store load-bearing is BY CONSTRUCTION.** It belongs in a method section, not in a headline (R29).
- **A result exists only if it is on disk.** Every number traces to a file.
- **Anything not run is reported as NOT RUN** — never as a zero, never as a failure.
- ⛔ **No kill-rule re-litigation.** (ii) fired; it is not reopened by a better metric.

## ⛔ Live traps
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate. Winner **`v1_capacity_schedule`**.
   ⭐ Select explicitly, assert the hash. A baseline run looks normal in the logs and is **void**.
2. **`include_coupling_bytes`** — correct budget is **524,288 B**, not 688,128 B.
3. ⚠ **LACE constant traps (R9.6):** shipped `n_epochs=50` (not 100, not 200); the staircase decay **never fires**;
   `dis_temp` is **not** a clean temperature, do not sweep it; **OR is non-commutative**; hard **10-operand limit**.
   ⛔ **`DO_NOT_USE_AS_A_BAR`:** LACE's published 0.94–0.98 is StyleGAN at 1024² on their splits. Only our
   re-implementation is comparable.
4. ⚠ **Feature caches are non-uniform:** seed 0 `scope="full"`, seeds 1–2 `scope="used"` with **no** matched-seen rows.
5. **JAX cold start ~20 min.** **16 GB machine** — ⛔ never start a full suite.

## Files you own
`chlu/experiments/exp_gen_cell.py` (**new** — the launcher's `STAGE=cell` already expects this exact module name and
currently aborts because it does not exist) · `chlu/core/gen_compose.py` — **read-only, do not edit** ·
`tests/test_gen_cell.py` (new) · `chlu/config.py` — **ONE new block, APPENDED AT EOF.**
⛔ Nothing under `chlu/core/gen_store.py`, `gen_bridge.py`, `gen_certificate.py`, `gen_request.py`, `gen_dream.py`,
`chlu/eval/rivals/`, `chlu/eval/gen_classifier.py`, `chlu/data/`, `scripts/csf3/`.

## Acceptance
1. All three preconditions verified on disk and the verification shown.
2. Decoder parameter hash asserted per arm per seed; external classifier identical across arms.
3. CLU path asserted to touch **no feature tensor**.
4. The table: 5 arms × ≥3 seeds, accuracy **and** distinct-output count on every row, byte ledger in-table.
5. The 31 % kNN byte advantage declared in the table, in words.
6. Submit lines emitted for a cluster re-run; every cluster item reported **NOT RUN**.
7. Per-unit incremental writes; anything not completed reported **NOT RUN**.
8. Targeted tests green. ⚠ ⛔ **Do not start a full suite.**

## What would prove this spoke wrong
- ⭐ **An arm's number moves and you cannot name which of the decoder, the classifier, or the arm moved it.** Hold two
  of the three fixed on every comparison, or the table measures nothing.
- ⭐ **A rival got weaker because of a change you made.** Regression-check the rivals against
  `gen-arms-a7-wiring.md`'s quoted numbers before you report anything.
- You reported an accuracy without its distinct-output count. That is the fact this whole wave exists to surface.
