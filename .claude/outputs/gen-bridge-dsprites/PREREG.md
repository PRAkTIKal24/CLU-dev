# PREREG — `gen-bridge-dsprites`

**Filed BEFORE any harness that measures a reported quantity has been run.**
Branch `agent/experiment-engineer/gen-bridge-dsprites`, base `main @ be27713`.
Nothing below may be edited after the first production fit starts. Amendments are appended, dated, and labelled.

---

## 0. What is being measured (stated before the numbers)

The **R3 decoder-ceiling gate on dSprites**, read on the **compositional term** (`shape`), with **R20's in-corner
discrimination metric UNCHANGED**:

- **positives** = the held-out cell's rows (`latent_classes[:, shp] == 1`, ellipse) — a **one-class** cell, exactly
  as Montero's Shapes3D cell is `shape == cylinder`;
- **negatives (matched-seen, R16's permanent companion)** = the cell's whole conjunction with the `shape` clause
  relaxed to a different, **seen** shape (square / heart), minus every fitted row and every held-out row;
- **seen normaliser** = the cell's **position** corner (`posX > 0.66 ∧ posY > 0.66`) with its own shape (ellipse) but
  **outside the cell**, i.e. scale and orientation relaxed — the dSprites transposition of `gen-bridge-wide`'s "the
  cell's hue bands + its own shape, but outside the cell". Scored against the **same** negatives.
  (dSprites has no photometric clause at all, so position is the only context clause the cell carries beyond the two
  factors that are relaxed. ⚠ This choice is declared here, before measurement, and is not revisited.)
- **decision** = argmax of the frozen probe's `shape` head on the **decoded** image equals 1;
- **gate statistic = binary balanced accuracy** ½(TPR+TNR) (R23.1). **AUROC over `p(ellipse)` is a mandatory
  companion and is NEVER the gate.**
- **Verdict rule (pre-registered, from the shipped `discrimination_verdict`):** **PASS** iff
  `balacc_heldout ≥ 0.75` absolute **AND** `balacc_heldout / balacc_seen ≥ 0.80`. ⛔ Otherwise **FAIL ⇒ STOP AND
  RETURN to the Hub** (R3.1). ⛔ No tuning toward a pass; the decoder recipe is pinned to R19.1's winner
  `v1_capacity_schedule` before any number is seen.

**Configuration pinned in advance:** `dataset=dsprites`, `pretext=recon`, `condition=recomb2element`,
`dec_recipe=v1_capacity_schedule`, `cache_scope=used`, `n_train=8192`, `n_eval_a=1024`, `enc_steps=1500`,
`probe_steps=1500`, decoder steps 1800 (cosine, warmup 0.05, `dec_width_mult=2.0`, `blocks_per_stage=2`, uniform
pixel MSE), seeds `[0, 1, 2]` in that order, caps `max_positives=6292 / max_negatives=4096 / max_seen=4096`.

---

## 1. Predictions (point value + 80 % interval), and how each is derived

| # | quantity | prediction | derivation |
|---|---|---|---|
| **P1** | **balanced accuracy, held-out (the gate statistic)** | **0.85** [0.62, 0.96] | Shapes3D/Montero/V1 reads **0.9414 ± 0.0179** on the same metric (R22/R23.5). dSprites should be *easier* on two axes — the cell admits the **largest 4 of 6 scales**, so objects are big, and the ellipse's 180° symmetry makes the 120–240° rotation band benign — and *harder* on three: the held-out cell is **6,292 rows = 0.85 %** of the grid (Shapes3D's is 72 = **0.015 %**), so the extrapolation region is **~57× larger in fraction**; the fitted draw is 8,192 / 730,988 = **1.1 %** (Shapes3D 1.7 %); and dSprites carries **no texture or colour cue at all**, so shape identity is *entirely* silhouette — which is exactly the thing uniform pixel MSE is never asked to preserve (`gen-decoder-lift` §5.3: V2's whole premise). Net: I expect a real drop below Shapes3D but not a collapse. The interval is wide because the silhouette-only argument could dominate. |
| **P2** | balanced accuracy, **seen normaliser** | **0.94** [0.84, 0.99] | Shapes3D's seen normaliser sits at 0.9774 (wide, 13 cells). dSprites' seen positives come from the same position corner at *relaxed* scale — which includes the **smallest** scales (0.5, 0.6), where a 64×64 silhouette is only a few pixels across and square/ellipse/heart genuinely converge. So a few points below Shapes3D. |
| **P3** | **ratio** (held-out / seen) | **0.90** [0.70, 1.03] | P1/P2 with correlated errors. Shapes3D wide gave 1.0025; Montero-configuration Shapes3D was not reported as a ratio on this metric. |
| **P4** | ⭐ **THE GATE VERDICT** | **PASS**, stated probability **0.60** | P3's interval puts ~20 % of mass below the 0.80 ratio floor and P1's puts ~20 % below the 0.75 absolute floor, partly overlapping. ⛔ I am explicitly *not* confident; a FAIL is a legitimate, useful result and is pre-committed to STOP. |
| **P5** | AUROC, held-out (companion) | **0.93** [0.78, 0.995] | AUROC is uniformly above balanced accuracy on this metric in every Shapes3D row (0.9886 vs 0.9414; 0.9994 vs 0.9798). |
| **P6** | bridge **aggregate** ceiling ratio `ceiling_B / ceiling_A` (context, not the gate) | **0.95** [0.82, 1.02] | The aggregate averages 5 heads; 3 of them (`orientation`, `posX`, `posY`) are read on both A and B by the same probe, so their difficulty cancels in the ratio. Shapes3D aggregate ratio was 0.8912 (baseline decoder) and ~1.0003 (V1, wide). |
| **P7** | probe on **REAL** held-out images, in-corner balanced accuracy (the instrument) | **0.99** [0.95, 1.000] | The Shapes3D probe reads real images at **1.0000** on both A and B (R15). dSprites silhouettes are cleaner still, but the smallest-scale seen positives may cost a little. |
| **P8** | ⚠ probe `orientation` accuracy on REAL images, **aggregate context** | **< 0.35** [0.05, 0.60] | 40 orientation classes over shapes with 90° (square) and 180° (ellipse) symmetry ⇒ the orientation head is **provably** unable to exceed ~1/4 (square) and ~1/2 (ellipse) of the classes. ⭐ Filed so that a low dSprites **aggregate** ceiling is not later misread as a decoder failure: ⛔ on dSprites the instrument is **NOT saturated** and the aggregate is not interpretable as it is on Shapes3D. |
| **P9** | decode MSE on held-out vs on matched-seen | ratio **1.00** [0.85, 1.25] | `gen-decoder-lift` measured 0.0027 vs 0.0031 on Shapes3D — the held-out corner is not harder to *reconstruct*, only harder to render *identifiably*. I expect the same here, and if it holds it is again evidence that any P1 shortfall is silhouette identity, not reconstruction. |
| **P10** | ⭐ **the realised split**: `n_total / n_test / n_train` | **737,280 / 6,292 / 730,988** | R11, Hub-verified. **Already measured and confirmed before this file was written** (see §3) — recorded here as an audit anchor, ⛔ not claimed as a prediction. |
| **P11** | Shapes3D bit-identity | the Shapes3D `montero_test_mask`, `select_rows`, `resolve_decoder_cfg` and default `dec_recipe` are **unchanged**, and `stream_images("shapes3d", …)` is **byte-identical** before/after my `UINT8_SCALES` edit | my diff is additive and the Shapes3D divisor stays 255.0 |
| **P12** | matched-seen negative pool | **12,584** (6,292 squares + 6,292 hearts) before fitted rows are removed | 2 shapes × 4 scales × 13 orientations × 11 posX × 11 posY. **Measured and confirmed before filing** (§3). |

## 2. ⭐ The falsifier, stated in advance
**P4 fires FAIL ⇒ the decoder cannot discriminate the held-out shape in its corner on dSprites ⇒ STOP.** In that
case nothing downstream on dSprites may run, and the correct report is the FAIL plus the diagnostic decomposition
(P7 real-image instrument, P9 decode MSE, P8 orientation) that says *which* of decoder / probe / encoder is the wall.

**Does NOT falsify:** a dSprites number below Shapes3D's 0.9414 (different dataset, different cell size, 57× larger
extrapolation region); a low **aggregate** ceiling (P8 — the orientation head is structurally capped); losing to
anything on a metric-native protocol (no arm, no CLU number appears in this spoke at all).

## 3. Measurements already taken BEFORE this file was written (declared, not hidden)
Two cheap audit checks were run first because the task requires them *before* fitting anything, and because a wrong
split would void everything downstream (traps T1–T4):
- realised split `737,280 / 6,292 / 730,988` — **matches R11 exactly**; in-cell realised values `shape` 1, `scale` 4
  (`0.7 0.8 0.9 1.0`), `orientation` 13 (120.000°–230.769°), `posX` 11, `posY` 11;
- `c_invalid.json` sha256 `10a39e713864c9f0c28f740caad49394cbda6f34f3d1d29bbc61dab9fc288666` — **matches**;
- corner/pool counts: corner 18,876; matched-seen pool 12,584; seen-normaliser pool 22,748.
⛔ **No encoder, decoder or probe has been fitted, and no discrimination number has been computed.**

## 4. Amendments
*(none yet)*
