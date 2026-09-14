# Task: gen-bridge-wide — the second configuration: can the decoder DISCRIMINATE shapes in a corner it never saw?

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-bridge-wide.md` + artefacts under `.claude/outputs/gen-bridge-wide/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all, **incl. the §5 pre-registration rule**) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R15, R16, R17, R19, R20** · `.claude/outputs/gen-wide-split.md` (your split) · `.claude/outputs/gen-decoder-lift.md` **§5 — the winning decoder RECIPE** · `.claude/outputs/gen-feature-bridge.md` **§2, §6**.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only). ⚠ **§7.23 will bite any conv module.**
- **Base:** `main @ e4f3d71` · **Branch:** `agent/experiment-engineer/gen-bridge-wide` · **Worktree:** fresh (`../CHLU-gen-bridgewide`). ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` are parked, not free.
- **Precondition:** none beyond the base — `chlu/data/shapes3d.py` carries `recomb2element_wide` and `chlu/core/gen_bridge.py` carries the decoder recipe, both on `main`.
- **Env:** main venv, no `uv sync` in the worktree (w6), `CHLU_GEN_DATA_ROOT` set. ⛔ **CSF3 needs 2FA and is unreachable from an agent session** — run locally on CPU, emit the submit line, report any cluster item as **NOT RUN**.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument, plus one gating measurement*, exactly as `gen-feature-bridge` was.
- **Laundering control:** the matched-seen population **inside each wide cell's own colour corner** — same construction
  as R16, now per cell.
- **Falsifies:** the decoder cannot **discriminate** shapes in a held-out corner ⇒ the 0.94 detection number was
  carrying more weight than it could bear, and kill rule (i)'s ceiling must be pre-registered at the measured level.
  ⭐ **A legitimate, useful outcome.**
- **Does NOT falsify:** a lower number than Montero's 0.94. ⛔ They are different quantities — detection vs
  discrimination — and must never be compared as if they were the same.

## The one job
Re-run the bridge at the **wide configuration** — encoder and decoder fitted with the wide cells **excluded** — and
report the decoder ceiling as a **discrimination** number on **multi-shape** held-out cells.

## ⭐ Why this is a run and not a re-score — read this before you plan anything
The shipped encoder and decoder trained on **Montero's** train split, which **contains the wide cells' rows**.
⛔ **They have seen them.** Scoring the existing artefacts on the wide cells would measure memorisation, not
generalisation. ⭐ **So both the encoder and the decoder are re-fitted here, with the wide cells held out**, and only
then is a ceiling measured. This is the second configuration R17 already requires; it is not extra cost.

## What to build
1. **Encoder**, `recon` pretext (R1 mandatory), trained on the **wide** train split only, then frozen. Hash it.
2. **Feature cache**, ⭐ **`cache_scope="full"`** — ⛔ **not `"used"`.** The non-uniform caches from the first bridge run
   (seed 0 full, seeds 1–2 `used` with 9,288 rows and no matched-seen rows) have now cost two spokes time. **Every seed
   gets a full cache here.**
3. **Decoder**, built from the **decoder-lift winning recipe** (`capacity_schedule`) refitted to these features. ⛔ Do
   not re-run the variant search — that question is closed. ⚠ Declare any hyperparameter you had to change because
   `D` differs, and say why.
4. **Probe**, per-factor, on **real** images. Report its accuracy on real held-out images — ⛔ if it is not ≈1.0, say so
   before any ceiling is believed.
5. ⭐ **The ceiling, as DISCRIMINATION.** For each wide cell, and pooled:
   - **balanced accuracy and AUROC** over the shapes realised in that cell, against matched-seen negatives in the same
     colour corner;
   - the multi-class `shape` accuracy alongside, **labelled as a different quantity**;
   - the same on the seen split, as the normaliser.
   ⛔ Do **not** report a bare detection rate as the headline.
6. **Per-cell by contrast.** The wide cells were built to span the photometric contrast range — ⭐ **report the ceiling
   as a function of contrast.** That is the analysis the split exists for, and it separates rendering from composition.
7. **The no-leak assertion**, per cell, with counts — same as R3.3.

## ⭐ Pre-register first
`PREREG.md` in your output dir before fitting anything: predicted discrimination ceiling (pooled and at the extremes of
the contrast range), how you derived it, and the gate verdict rule you will apply. ⛔ A hedge is worth nothing when the
number lands.

## Files you own
`chlu/experiments/exp_gen_bridge.py` — **the `--condition` passthrough only**, additive. `chlu/eval/gen_probe.py` —
**the discrimination metrics only**, additive. `tests/test_gen_bridge_wide.py` — new. `chlu/config.py` — fields inside
the **existing `GenBridgeConfig`** block only.
⛔ You own nothing under `chlu/core/gen_store.py`, `chlu/data/`, `chlu/eval/rivals/`, or `scripts/csf3/`.
⚠ **Do not change the Montero configuration's behaviour.** Assert its ceiling reproduces `gen-decoder-lift`'s numbers
after your change.

## Acceptance
1. Encoder + full-scope cache + decoder + probe at the wide configuration, each hashed and manifested, **≥3 seeds**.
2. ⭐ Discrimination ceiling (balanced accuracy **and** AUROC), pooled and per cell, with the seen-split normaliser.
3. Ceiling reported **as a function of photometric contrast** across the cells.
4. No-leak assertion per cell with counts; probe-on-real reported.
5. Montero's configuration provably unchanged — its ceiling reproduces, asserted in a test.
6. Prereg filed before fitting; outcome table for every prediction, including the falsified ones.
7. Suite green (before/after, main venv); ref verified **from the main repo** before worktree removal.

## What would prove this spoke wrong
- ⭐ **You scored the existing frozen artefacts on the wide cells.** They trained on those rows. If any number in your
  report comes from an encoder or decoder that saw a wide cell, the whole report is void — assert the exclusion first.
- You reported detection where discrimination was asked for, or put the two in one column.
- You used `cache_scope="used"` and discovered later that the matched-seen rows are missing. **Use `full`.**
