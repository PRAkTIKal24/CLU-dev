# Task: gen-external-classifier — an instrument that reads IMAGES, independent of the decoder

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-external-classifier.md` + artefacts under `.claude/outputs/gen-external-classifier/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R3 (the ceiling is a GATE), R15 (the probe reads real images at 1.0000 and is SATURATED), R19.1, R23.3, R29 pivot item 2** · `.claude/outputs/gen-feature-bridge.md` §the ceiling · `.claude/outputs/gen-decoder-lift.md` §the winner.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 5fee3f4` · **Branch:** `agent/experiment-engineer/gen-external-classifier` · **Worktree:** fresh (`../CHLU-gen-extclf`).
- **Env:** main venv (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`), ⛔ no `uv sync` in the worktree. ⛔ CSF3 is unreachable (2FA) — local CPU only; **emit the submit line for a cluster re-fit and report it NOT RUN.**
- ⚠ **Budget ≈ 5 h and it is the tightest in the wave.** ⭐ **Checkpoint every epoch, write per seed, skip any seed whose artefact exists.** ⛔ Never hold to the end. If you run out of budget, ship **fewer seeds, fully reported**, never a partial number presented as complete.

## ⭐ DIAL DECLARATION
- **Dial:** *the instrument* — ⛔ **no arm, no CLU number, no kill-rule verdict.** You are building and gating a measuring device.
- **Laundering control:** the device must be shown **independent of the decoder**, not merely different from it. See leg 3.
- **Falsifies:** the phase's whole ceiling stack. ⭐ **If the external classifier scores the decoder's own renders far below what the linear probe reports, every ceiling number in this phase is optimistic — and that is a finding, not a failure.**

## The one job
Build a classifier that reads factor values off a **rendered image**, trained on **real train-split images only**, and
**gate it** — then report what it says about the decoder's renders.

## Why this exists
Every composition number in the phase is currently measured by a **linear probe on the encoder's features**, and the
encoder and decoder were fitted together. R29's headline table requires
`request → read → codes → decoder → image → EXTERNAL classifier`. ⭐ **A probe that shares the decoder's lineage cannot
be the instrument that scores the decoder's output.** This is that instrument.

## The three legs
### 1. Build — real pixels, train split only
A small CNN over **raw images**, trained on the **Shapes3D train split only** (479,928 rows; subsample is allowed and
**must be reported with its size and seed**). ⛔ **It never sees a decoder render during training**, and ⛔ **it never
sees a held-out-combination image during training.** **Assert both, and report the excluded count** — R3.3's no-leak
assertion is the precedent and later "just retrain the classifier" pressure will point at exactly this leak.

### 2. ⭐ THE GATE — pre-registered, stop-and-return
On **REAL held-out-combination images**, in-corner, the classifier must reach **balanced accuracy ≥ 0.95 on `shape`**.
- Justification for the floor, stated so it is not a number pulled from air: R15 established the existing probe reads
  **REAL** images at **1.0000** on both A and B and is saturated. An instrument that cannot read real images in the
  corner is not an instrument.
- ⛔ **If the gate fails, STOP and return to the Hub. Do not proceed to leg 3 and do not tune toward a pass.**
  A failed gate means the phase needs a different instrument, which is a Head decision, not a spoke decision.

### 3. ⭐⭐ THE CONTROL THAT CAN KILL THE PHASE'S CEILINGS
Once gated, run the classifier over the **frozen decoder's renders of the A set** (seen configurations) and compare,
side by side and per seed, against the linear probe's number on the same renders. Reference rows from disk:
decoder ceiling in-corner shape balanced accuracy **0.9414 ± 0.0179**; 11-cell held-out shape **0.9894 ± 0.0021**.
- ⭐ **If the external classifier agrees with the probe, the ceilings stand and the instrument has earned its place.**
- ⛔ **If it scores the same renders materially lower, report the gap as the headline of this spoke.** Every ceiling
  in this phase would then be an artefact of measuring the decoder with its own family, and the Head must know before
  the harness runs, not after.
- ⛔ Do not adjudicate which instrument is "right" — report both, with the gap, and route it to the Hub.

## ⛔ Live traps
1. **`dec_recipe` defaults to `"baseline"`** — the decoder that FAILED the gate. The winner is
   **`v1_capacity_schedule`**. ⭐ Select it explicitly and **assert the decoder parameter hash** before rendering
   anything. A baseline run looks normal in the logs and is **void**.
2. ⚠ **Feature caches are non-uniform:** seed 0 is `scope="full"` (480,000 rows); seeds 1–2 are `scope="used"` and
   contain **none** of the matched-seen rows. You want **images**, not the cache — but the split indices come from the
   same place, so check the scope before you index.
3. **h5py is required** by `chlu/data/shapes3d.py` and reaches the venv through the `gen` extra. If it is missing, 8
   real-data tests turn into **silent skips** (`importorskip`) — nothing goes red. **Verify it imports and say so.**
4. **JAX cold start ~20 min** here. Keep the session warm; budget for it.
5. **16 GB machine, two other spokes running.** ⛔ Never start a full test suite; ⛔ never hold a full image array and a
   feature cache in memory at once.

## Files you own — all NEW
`chlu/eval/gen_classifier.py` (new) · `chlu/experiments/exp_gen_classifier.py` (new) ·
`tests/test_gen_classifier.py` (new) ·
`chlu/config.py` — ⭐ **ONE new config block, APPENDED AT END OF FILE. ⛔ Edit no existing block.**
⛔ Nothing under `chlu/core/`, `chlu/eval/rivals/`, `chlu/data/`, `scripts/csf3/`.
⚠ **`chlu/config.py` is shared with two concurrent spokes.** `gen-arms-a7-wiring` edits the existing
`GenBaselinesConfig` block; `gen-commitment-collapse` also appends at EOF. ⛔ **If you conflict there, do NOT merge
hunk-by-hunk** — take the peer's whole file, re-apply your block, assert **0 deletions**.

## Acceptance
1. `PREREG.md` filed **before** the first gate measurement, stating the ≥0.95 floor and the leg-3 comparison.
2. No-leak assertions pass and the excluded counts are reported: **0 decoder renders** and **0 held-out-combination
   images** in training, per seed.
3. Gate measured on real held-out images, 3 seeds, PASS/FAIL stated plainly. **On FAIL: stop and return.**
4. Leg-3 comparison table: external classifier vs linear probe on the **same** decoder renders, per seed, with the gap.
5. ⭐ The gate verdict written to **`.claude/outputs/gen-external-classifier/classifier_gate.json`** with a top-level
   `"gate": "PASS"` or `"gate": "FAIL"` key, written **last**. ⭐ A downstream spoke gates on that exact file and key.
6. A cluster re-fit submit line emitted and reported **NOT RUN**.
7. Per-seed incremental writes; any seed not completed reported as **NOT RUN**, never as a zero.
8. Targeted tests green. ⚠ ⛔ **Do not start a full suite** — the Hub runs one at wave close.

## What would prove this spoke wrong
- ⭐ **Your classifier agrees with the probe everywhere, including where the probe is known to be saturated.** Then it
  is a second copy of the same instrument and adds nothing. Check it against the decoder's renders, where the two
  can actually disagree — that is the only place the independence claim is testable.
- You trained on decoder renders, or on held-out combinations, and the assertions did not fire. The assertions must
  be **shown able to fire** — demonstrate that on a deliberately seeded violation, exactly as `gen-wide-split` did.
