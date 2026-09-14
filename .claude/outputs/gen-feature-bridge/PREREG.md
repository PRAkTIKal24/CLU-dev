# PREREG — gen-feature-bridge (written BEFORE the ceiling harness was run)

Author: experiment-engineer · Date: 2026-09-09 · Base: `main @ 40f606d`
Branch: `agent/experiment-engineer/gen-feature-bridge`

The acceptance criterion is a **measured ratio** (`ceiling_B / ceiling_A` against a 0.8 floor), so protocol §5
requires a pre-registration. Nothing below was informed by a ceiling measurement — at the time of writing, the
only numbers I had run were a CPU step-time benchmark (0.368 s/step, batch 128) and an HDF5 read benchmark.

## 0. What is being measured

- `ceiling_X` = mean over the 6 Shapes3D factors of the probe's top-1 accuracy on images produced by
  **decoding the REAL cached encoder features** of split-`X` rows through the ONE frozen decoder.
- `A` = rows in the Montero *train* split that were **not** used to fit the encoder / decoder / probe
  (novel images, seen combinations). `B` = all 72 held-out-combination rows (novel images, unseen combinations).
- The gate: **PROCEED iff `ceiling_B >= 0.8 * ceiling_A`**, else STOP.

## 1. Point predictions (committed)

| Quantity | Predicted (point) | Predicted range | Derivation |
|---|---|---|---|
| probe accuracy on **real** A images (mean over factors) | **0.97** | 0.90 – 0.995 | Shapes3D factors are near-perfectly visually determined; the only hard heads are `orientation` (15 classes of camera azimuth, a smooth continuum) and `scale` (8 near-adjacent sizes). Hues (10 classes each) are trivially separable in RGB. |
| probe accuracy on **real** B images | **0.95** | 0.85 – 0.99 | Same instrument, but a 72-image corner of factor space; small-sample noise is ±0.03 at n=72 and the corner is at the extreme of `scale`/hue ranges, where class boundaries are one-sided (extremes are easier), so I expect it *not* to collapse. |
| `ceiling_A` | **0.90** | 0.78 – 0.96 | probe-on-real minus a decode-degradation term. A 2048-d (128×4×4) post-ReLU bottleneck reconstructs Shapes3D well; the loss is in fine orientation/scale detail, i.e. concentrated in the two hard heads. |
| `ceiling_B` | **0.84** | 0.62 – 0.94 | as `ceiling_A`, minus a compositional-rendering penalty for a feature configuration the decoder never rendered. |
| ⭐ **`ceiling_B / ceiling_A`** | **0.93** | **0.85 – 1.02** | see §2 |
| **Gate verdict** | **PROCEED** | — | 0.93 > 0.80 with the whole predicted range above the floor except its very tail |

## 2. Why I predict the ratio is high (the mechanism, stated in advance)

The shared decoder is **not** a global generator conditioned on a factor vector. It is a *local* conv renderer
inverting a **4×4×128 spatial feature map that the frozen encoder produced from the real image**. All the
information needed to render a B image is therefore already in its code; the decoder only has to invert a
locally-structured map. Conv decoders are strongly translation- and composition-generalising over local
features, so a factor *conjunction* never seen jointly is much less of an obstacle than it would be for a
latent-vector generator. Prediction: **`ratio >= 0.85`**.

## 3. The competing hypothesis this gate exists to catch (also pre-registered)

If the 4×4 code is global enough that the decoder has memorised a *combination manifold* rather than a local
feature-inversion rule, then the B corner (cylinder ∧ scale≥1.1 ∧ all three hues ≥0.8 ∧ orientation>20) is
off-manifold and rendering degrades on the factors that define the corner. Prediction under this hypothesis:
**`ceiling_B ≈ 0.55–0.70`, ratio ≈ 0.60–0.78 ⇒ STOP.**

⭐ These two are distinguishable *in advance*: H1 (local inversion) predicts the B per-factor accuracy profile
is the **same shape** as A's (worst on `orientation`); H2 (manifold memorisation) predicts the largest B−A drop
lands on the **corner-defining** factors (`object_hue`, `scale`, `shape`).

## 4. Secondary pre-registered predictions

1. **Worst factor on both A and B is `orientation`** (15 classes, smooth). Second worst: `scale`.
2. **Instrument control:** `probe_real_B / probe_real_A >= 0.90`. If this is < 0.90 the corner is hard for the
   *probe*, and the raw ceiling ratio understates the decoder; I will report both and gate on the raw ratio as
   specified, flagging the discrepancy rather than adjusting.
3. **Seed spread** of the ratio across seeds {0,1,2}: std ≤ 0.05 (B has only 72 images ⇒ ±0.03 binomial noise
   per factor per seed).
4. **`cls` diagnostic arm**: its `ceiling_A` will be **lower** than `recon`'s (a classification pretext discards
   the appearance detail a renderer needs), but its **ratio** will be similar or higher (a factor-classifying
   encoder produces a more factorised, hence more composable, code). Predicted `cls` ratio ≥ `recon` ratio − 0.05.

## 5. Falsification / stop conditions, restated

- `ceiling_B < 0.8 × ceiling_A` ⇒ **STOP**, reported as-is. ⛔ I will not retune the decoder to clear the floor.
- A low absolute accuracy on **both** A and B does **not** falsify the bridge (dial declaration); it is a weak
  probe or a hard dataset and is reported as such.
- If the no-leak assertion finds any held-out row inside the decoder's training set, **every** number above is
  void regardless of its value.

## 6. Non-negotiables fixed before measurement

- Floor = 0.8 (R3.1). Seeds = {0, 1, 2}. Aggregate = unweighted mean over the 6 factors' top-1 accuracies.
- `A` rows are disjoint from every row used in any fitting stage, drawn with the run seed.
- `B` = all 72 held-out rows, never subsampled.
- Same probe, same decoder, same protocol for A and B within a seed.
