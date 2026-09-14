# PREREG — `gen-external-classifier`

**Filed BEFORE the harness that measures any of it exists.** Written 2026-09-13, by `experiment-engineer`,
on branch `agent/experiment-engineer/gen-external-classifier` off `main @ 5fee3f4`, worktree
`/Users/user/Desktop/CHLU-gen-extclf`. ⛔ At the moment of writing, `chlu/eval/gen_classifier.py`,
`chlu/experiments/exp_gen_classifier.py` and `tests/test_gen_classifier.py` **do not exist** — check their
mtimes against this file's.

---

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** *the instrument.* ⛔ No arm, no CLU number, no kill-rule verdict.
- **Laundering control:** independence of the decoder, demonstrated in leg 3 (where the two instruments can
  actually disagree), not asserted from architecture.
- **Falsifies:** the phase's ceiling stack — if the external classifier scores the decoder's own renders
  materially below what the linear probe reports, every ceiling number in phase 1 is optimistic.
- **Does NOT falsify:** the external classifier scoring *below the probe on REAL images*. That would be an
  instrument defect and is caught by the gate (leg 2), not by leg 3.

---

## 1. The instrument, as it will be built (declared before fitting)

`ExternalFactorCNN` over **raw images**, per-factor heads. Deliberately unlike the shipped `FactorProbe` on
**four** axes, so that "different instrument" is a construction fact and not a hope:

| axis | shipped `FactorProbe` | `ExternalFactorCNN` |
|---|---|---|
| training rows | the bridge's own `select_rows(...)["train"]`, **the same 8,192 rows the decoder is fit on** | a **disjoint** draw from the train split, same size (8,192), different RNG stream |
| conv stack | kernel 4, stride 2, GroupNorm, **ReLU**, channels `[32,64,128]` | kernel **5**, stride 2, GroupNorm, **SiLU**, channels `[32,64,128,128]` |
| head | flatten → one `Linear` per factor | flatten → `Linear(256)` → SiLU → one `Linear` per factor |
| key chain | `PRNGKey(seed*10_007 + 20_260_910)` split alongside the decoder's key | `PRNGKey(seed*104_729 + 20_260_913)` |

⛔ It never sees a decoder render at fit time. ⛔ It never sees a held-out-combination image at fit time.
⛔ It never sees the `eval_a` rows, nor the matched-seen negatives, nor **any** row in the Montero photometric
corner, so every row it is scored on is a row it has never seen in any form.

**Reported with the numbers (R15/M4):** subsample size, its RNG seed, and the excluded counts.

---

## 2. ⭐ THE GATE (leg 2) — pre-registered floor, stop-and-return

**Statistic.** In-corner `shape` **balanced accuracy** on **REAL** images, exactly the R20/R23 construction:
positives = the 72 real held-out-combination images (all `shape = cylinder`); negatives = the real matched-seen
non-cylinder images in the same photometric corner. Chance = 0.5. Argmax decision rule, AUROC as companion.

**Floor: 0.95.** Justification (R15): the shipped probe reads REAL images at **1.0000** on both A and B and is
saturated. An instrument that cannot read real images in the corner is not an instrument.

**Decision rule, fixed now:**
- **PASS** iff **every** seed ≥ 0.95. The mean is reported but is not the gate.
- **FAIL** otherwise ⇒ ⛔ STOP, do not run leg 3, do not tune toward a pass, return to the Hub.
- `classifier_gate.json` carries a top-level `"gate": "PASS" | "FAIL"` key, written **last**.

**Prediction P1.** PASS, with per-seed balanced accuracy in **[0.97, 1.00]**, point estimate **0.995**.
*Derivation:* the shipped probe is a smaller net fitted on an equally-sized real-image draw and reaches 1.0000
aggregate on real A and real B; `shape` is the hardest of its six heads and still saturates. Mine has one more
conv stage and a hidden layer at identical step count, so the only source of loss is the corner being out of its
training draw (≈216 corner rows exist in 479,928, so the probe's own draw contained ≈4 of them — i.e. the probe
was effectively corner-naive too, and still read 1.0000).

---

## 3. ⭐⭐ LEG 3 — the control that can kill the phase's ceilings

Both instruments read the **same rendered arrays**, produced by the **frozen `v1_capacity_schedule` decoder**
(parameter hash asserted per seed against `gen-decoder-lift/winner/winner_hashes.json`) from the **same frozen
cached features**. ⛔ Nothing is re-rendered between the two reads.

**Primary row.** In-corner `shape` balanced accuracy on **renders**: positives = renders of the 72 `eval_b`
features, negatives = renders of the matched-seen non-cylinder features (n = 198/199/199, the same negative set
R22 used: `matched_seen_rows(..., exclude=store_rows["store_train"])`).
**Declared reference (from disk, not re-measured): probe = 0.9414 ± 0.0179** (per seed 0.9223 / 0.9365 / 0.9653,
`gen-r0-rescore/rescore_seed_*.json: gate_real_features.v1_decoder.balanced_accuracy`). It will nonetheless be
**recomputed here on the same renders** and asserted against that reference, so the gap is measured within one
run rather than across two.

**Companion rows.** Per-factor top-1 and aggregate on the **A-set renders** (n = 1024), both instruments; and
each instrument's own REAL-image row on the same rows.

**Materiality rule, fixed now.** The gap `Δ = external − probe` on the primary row is **material** iff
`|mean Δ| ≥ 0.05` **and** `|mean Δ| ≥ 2 ×` the across-seed std of the per-seed Δ.

**Two competing hypotheses, both pre-registered (v5-gate precedent):**

- **H_AGREE — the ceilings stand.** `|mean Δ| < 0.05` ⇒ the external classifier confirms the probe on the
  decoder's own renders and the instrument has earned its place.
- **H_OPTIMISTIC — the ceilings are inflated.** `mean Δ ≤ −0.05` and material ⇒ ⭐ **headline of this spoke**:
  every phase-1 ceiling is an artefact of measuring the decoder with a probe of its own lineage.

**Prediction P2 (my point prediction, committed).** `mean Δ = −0.04`, i.e. external ≈ **0.90**, so I predict
**H_AGREE holds but only just** — the gap is negative and visible, and lands just inside the materiality rule.
*Derivation:* the probe's 8,192 fit images are exactly the images the decoder was trained to reconstruct, so the
render manifold is optimised toward the probe's own training distribution. That coupling is second-order (the
probe never saw the *evaluation* rows) so I do not expect a collapse, but it is one-sided: it can only flatter
the probe. I therefore predict a negative, small gap. ⛔ If Δ comes out **positive and material** that is also a
finding (the probe would be the *pessimistic* instrument) and I will report it as such.

**Prediction P3 (aggregate on A renders).** probe = 0.9893 (from disk); external predicted **0.96 ± 0.02**,
i.e. a small negative gap for the same reason. The four hue/orientation heads are at ≈1.0 for any competent
reader, so essentially all of the gap must appear in `scale` and `shape`.

**Prediction P4 — the task's own "what would prove this spoke wrong".** I predict the *non*-degenerate outcome:
`|Δ| ≥ 0.01` on renders while both instruments sit ≥ 0.98 on real images. If instead `|Δ| < 0.01` on renders AND
both are ≥ 0.99 on reals, the instrument is a second copy of the probe and **adds nothing** — I will say exactly
that, in those words.

---

## 4. Assertions that must be *shown able to fire* (`gen-wide-split` precedent)

1. **No held-out-combination image in training.** `assert_no_holdout_leak(split, clf_train_rows)`; reported as
   `n_holdout_excluded = 72`, `n_holdout_leaked = 0`, per seed.
2. **No decoder render in training.** Structural *and* verified: a random sample of the classifier's training
   tensor is re-streamed from the HDF5 file and required to be **bit-identical**. A decoder render cannot pass it.
3. **Lineage disjointness.** `|clf_train ∩ bridge_train| = 0`, `|clf_train ∩ eval_a| = 0`,
   `|clf_train ∩ corner_rows| = 0`.
Each of 1–3 is demonstrated firing on a deliberately seeded violation, in `tests/test_gen_classifier.py`.

---

## 5. Declared NOT RUN in advance
- The **CSF3 re-fit** of this classifier: the cluster is unreachable (2FA). A submit line is emitted and reported
  **NOT RUN**.
- The **wide (`recomb2element_wide`) 11-cell** comparison against the `0.9894 ± 0.0021` reference is a **stretch
  item**, attempted only if budget remains after the Montero legs land on 3 seeds. If it does not run it is
  reported **NOT RUN**, never as a zero.
