# Task: gen-encoder-unfreeze — train the encoder with the store in the loop, without losing the bypass guarantee

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-encoder-unfreeze.md` + artefacts under `.claude/outputs/gen-encoder-unfreeze/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R1 (reconstruction is the mandatory pretext and WHY), R5 (compute + layout), R18 (⛔ what a changed encoder voids), R19.2, R20 (the 30-bit design constant), R21 + R23.6 (the cluster smoke gates everything), R29 pivot item 3** · `.claude/outputs/gen-launcher-fixes.md` §2, §3 (the submit block and the input-path gap).
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's merged tip at spawn time (the Head will name it) · **Branch:** `agent/experiment-engineer/gen-encoder-unfreeze` · **Worktree:** fresh (`../CHLU-gen-unfreeze`).
- ⚠ **Budget ≈ 5 h local.** ⭐ Write per unit, skip any unit whose artefact exists.

## ⛔ PRECONDITION — mechanical, content-checked
`.claude/outputs/gen-cluster-smoke/smoke_result.json` exists **and** its top-level `"status"` is `"ok"`.
⭐ **SATISFIED 2026-09-14** — round 3 came back green on all four runs at commit `22f9dc5` with fresh job ids, and the
`certificate` stage executed on the cluster for the first time. **You may start.**

## ⭐⭐ SCOPE CHANGE (Head, 2026-09-14) — BUILD LOCALLY, DO NOT RUN ON THE CLUSTER
⛔ **This spoke no longer produces a cluster result.** Its acceptance is a **tiny-scale LOCAL smoke** of the full
unfrozen path. ⭐ **The cluster run stays gated and is a later, separate step** — the Head clears 2FA this week.
⛔ **Do not submit anything.** Emit the submit lines and report every cluster item **NOT RUN**.
⭐ **Why:** a build that cannot be tested invites exactly the stale-run episode that cost this programme three
submission rounds. A local smoke that exercises the real path is worth more than an untested cluster launch.

## ⭐ DIAL DECLARATION
- **Dial:** *the encoder* — currently frozen; here it trains with the store in the loop.
- **Laundering control:** ⭐ **the frozen-encoder arm at matched compute.** An unfrozen encoder gets more gradient steps
  than a frozen one; ⛔ a win that is really a compute win is not a win. **Match the budget and report it.**
- **Falsifies:** the hypothesis that the commitment collapse is caused by the encoder being frozen. ⛔ **If unfreezing
  does not lift the distinct-code count on B, say so — that is a finding and it narrows the problem.**

## The one job
Build and locally smoke a training path in which the **encoder trains with the store in the loop**, keeping the
structural no-bypass guarantee intact — then emit the cluster submit lines. ⛔ **You do not run it on CSF3.**

## ⛔ THE GUARANTEE YOU MUST NOT BREAK — this is the programme's single biggest result
`read_from_codes(store, codes)` **takes no features**. The only encoder → decoder channel is **`K·log₂M = 30 bits`**,
a declared design constant (R20). ⭐ **A bypass is not merely unmeasured, it is structurally inexpressible** — and that
sentence is in the paper's method section.
⛔ **Unfreezing the encoder is exactly the change that can silently create a bypass**, because gradients now flow from
the decoder's loss back into the encoder. **You must:**
1. Keep the forward path `request → codes → read_from_codes → decoder`. ⛔ No feature tensor reaches the decoder.
2. **Assert, in code and in a test, that the decoder's input has no gradient path to the encoder's features except
   through the K integer codes.** ⭐ **Write that test FIRST and show it failing on a deliberately bypassed variant** —
   a check that cannot fail proves nothing.
3. Report the **realised channel width** for every configuration. Any widening beyond 30 bits is a **declared change to
   the claim**, not a free improvement.

## ⛔ What unfreezing VOIDS — state this in your report, it is not optional
R18: the store trains on cached features from the **frozen** encoder, and the decoder is downstream of it. ⭐ **A
changed encoder changes the feature cache and voids every downstream number in this phase.**
⇒ **This is a NEW configuration sitting BESIDE the frozen one, never a replacement.** ⛔ Write to fresh paths. ⛔ Do not
overwrite, re-fit or invalidate any existing bridge, store or decoder artefact. **Name in your report every existing
number that would have to be re-measured if this configuration were ever promoted.**

## ⭐ Pretext is not yours to choose
R1: **reconstruction is the primary pretext, mandatory.** The positioning sentence against LACE is *"their energy is
derived from label supervision; our store is learned from data"* — a factor-classification pretext collapses it.
⛔ The generative path takes **no factor labels**. The evaluation probe still trains on labels; that is measurement,
not the generative path, and it stays.

## The legs
1. **The bypass test first**, shown able to fail (above).
2. **Local smoke** of the full path at toy scale on CPU: a handful of steps, fresh workspace, artefacts written.
3. **Matched-compute frozen control** wired in as an arm, not bolted on afterwards.
4. **Distinct-code count on B** reported for both arms at smoke scale — ⭐ with its **matched-seen (MS) companion**,
   because the collapse is a property of the colour corner and not of held-out-ness.
5. **Submit lines** for CSF3, following `job_gpu_gen.sh`'s idiom exactly. ⭐ **One literal command line per
   submission** — the whole `--export` value is ONE quoted shell word and `sbatch --export` splits on **commas only**.
   ⛔ Do not build it in a variable and expand it; zsh does not word-split an expanded variable and that has submitted
   garbage on this programme before. Every line carries `GEN_GIT_COMMIT=$(git rev-parse HEAD)`.
   ⚠ Fold in `gen-launcher-fixes.md` §3.3's **input-path gap**: `push_repo.sh` excludes `.claude/`, so the payload's
   default `bridge_root` / `store_runs` **do not exist on the cluster**. Say how the artefacts get staged.
6. ⛔ **Every cluster item reported NOT RUN.** No job, no job id, no inferred number.

## ⛔ Live traps
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate. Winner **`v1_capacity_schedule`**;
   select explicitly and assert the hash.
2. **`gen_data.data_root` must fail loudly with no default** (R26.7) — a silently-skipped C set is *"declared not-run
   reported as null"* in code form.
3. **JAX cold start ~20 min. 16 GB machine** — ⛔ never start a full suite, ⛔ never hold two feature caches.
4. **Detached spokes keep running but fail their LAST write** — incremental, skip-if-exists, always.

## Files you own
`chlu/experiments/exp_gen_encoder.py` (new or extended — the launcher's `STAGE=encoder` expects this module name) ·
`chlu/core/gen_bridge.py` — ⭐ **additive only: a new unfrozen path beside the frozen one. ⛔ Do not change the frozen
path's behaviour; assert the frozen path is bit-identical before and after your change.** ·
`tests/test_gen_encoder_unfreeze.py` (new) · `chlu/config.py` — **ONE new block, APPENDED AT EOF.**
⛔ Nothing under `chlu/core/gen_store.py`, `gen_compose.py`, `gen_certificate.py`, `gen_request.py`, `gen_dream.py`,
`chlu/eval/`, `chlu/data/`. ⛔ **`scripts/csf3/job_gpu_gen.sh` is read-only to you** — if it needs a change, report it.

## Acceptance
1. Precondition verified on disk and the verification shown.
2. The bypass test exists, passes on the real path, and is **demonstrated failing** on a deliberately bypassed variant.
3. The frozen path asserted **bit-identical** before and after your change.
4. ⭐ **THE ACCEPTANCE CRITERION: a tiny-scale LOCAL smoke of the full unfrozen path completes end to end** into a
   **fresh** workspace, with nothing existing overwritten. State the scale used (rows, steps) and that it is a smoke.
5. Matched-compute frozen control reported beside the unfrozen arm, with the compute budget stated for both.
6. Distinct-code count on B **and** MS, both arms, at smoke scale.
7. Submit lines emitted, one literal line each; **every cluster item NOT RUN**.
8. The list of existing numbers this configuration would void if promoted.
9. Targeted tests green. ⚠ ⛔ **Do not start a full suite.**

## What would prove this spoke wrong
- ⭐ **Your bypass test passes on a variant you deliberately broke.** Then it is decoration and the guarantee is
  unprotected. Show it failing before you trust it passing.
- ⭐ **The unfrozen arm wins and the compute budgets were not matched.** You measured compute, not the encoder.
- You changed the frozen path. Every number in this phase rests on it.
