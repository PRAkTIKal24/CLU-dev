# PREREG — gen-r0-rescore (results-analyst)

Written **before** the re-score harness existed or ran. Protocol §5 pre-registration rule:
acceptance criterion 4 is a **verdict on a measured margin**, so the margin is committed to here first.

Base: `main @ e4f3d71`. Seeds {0,1,2} (seed count per number is reported honestly; 1-seed numbers will be
labelled as such). All inputs frozen: bridge encoder/decoder/probe, `gen-decoder-lift` V1 decoder pkls,
`gen-store-grouping` store plan + labels. ⛔ Nothing retrains; the store is **re-materialised** from the
same seed/key chain and its bit-identity against the shipped `store_manifest.json` is asserted before any
new number is believed.

## How the predictions were derived

Sources (all measured, all on disk):
- `gen-decoder-lift` §5.1: `shape` top-1 on B — baseline `0.5926 ± 0.0866`, V1 `0.9398 ± 0.0131`;
  on `A_matched` — baseline `0.8492 ± 0.0269`, V1 `0.9432 ± 0.0404`.
- `gen-decoder-lift` §5.3 confusion counts on B (72 items, all cylinders): baseline cyl 36/41/51,
  V1 cyl 67/69/67.
- `gen-store-grouping` §3 read: correlation aggregate `0.5997 ± 0.0019` / `shape 0.5785 ± 0.0124`;
  random `0.4492 ± 0.0008` / `shape 0.3770 ± 0.0156`; margins `+0.1504` / `+0.2015`.
- `gen-store-grouping` §6 leakage (old, degenerate): pooled corr `−0.0444`, rand `+0.2257`,
  margin `+0.2701`; `shape` per-factor as extreme as `−12.9` on some cells.

---

## P1 — Kill rule (i), corrected: cylinder-vs-not **within the corner**, REAL features
Positives = 72 held-out cylinders (`eval_b`); negatives = matched-seen non-cylinders in the same
hue corner (n ≈ 198–199 per seed). Probe's native decision rule = argmax over the 4 shape logits.

| | predicted balanced accuracy | predicted AUROC |
|---|---|---|
| baseline (superseded) decoder | **0.75** (0.62–0.85) | **0.80** (0.68–0.90) |
| **V1 `capacity_schedule`** | **0.94** (0.86–0.98) | **0.97** (0.92–0.995) |

Derivation: positive recall ≈ the measured detection rate on B (0.593 / 0.940). Negative recall =
1 − (false-cylinder rate on `A_matched`); the baseline's 3-way `A_matched` `shape` accuracy 0.8492 and
V1's 0.9432 bound the false-cylinder rate at ≲0.15 / ≲0.06, so negative recall ≈ 0.90 / 0.95.
Balanced accuracy = mean of the two recalls ⇒ 0.75 and 0.94.

**P1-falsifier:** V1's balanced accuracy ≤ 0.8 ⇒ the 0.94 "lift" was a detection artefact and the
corrected kill-rule metric does not clear the R3.1 floor after all. ⭐ This is a live possibility and
will be reported as the headline if it happens.

## P2 — Kill rule (i)'s corrected metric on the **store read** (correlation vs random)
Same positives/negatives, but the features are replaced by the 30-bit store read before decoding.

| | predicted balanced accuracy |
|---|---|
| correlation | **0.58** (0.50–0.72) |
| random | **0.52** (0.50–0.60) |
| margin | **+0.06** (0.00–+0.18) |

Derivation: the store read is a 30-bit channel and `shape` is the factor it reads worst after
scale/orientation; within a single colour corner almost all discriminative photometric signal is
removed, so both arms are expected to sit near chance with a small correlation advantage.
⭐ **Explicitly registered as plausible:** both arms land at 0.50 ± 0.03 (pure chance). That would say the
store read carries **no** in-corner shape identity — a real negative, and it does **not** contradict P4.

## P3 — The corrected leakage x-axis (R20 balanced-accuracy transfer)
`leakage = 1 − balacc(held-out ∪ matched-seen-other-shapes, same corner) / balacc(same metric, seen corners)`

| | predicted pooled leakage |
|---|---|
| correlation | **+0.05** (−0.10 … +0.20) |
| random | **+0.25** (+0.10 … +0.45) |
| **margin (rand − corr)** | **+0.20** (+0.08 … +0.32) |

Derivation: the old pooled margin was `+0.2701`; balanced accuracy removes the one-class blow-up but the
old pooled number was dominated by the *other* five factors (which were already multi-class), so the
margin should survive at roughly its old magnitude, possibly slightly compressed.

**Non-degeneracy prediction (the proof acceptance item 2 asks for):**
⭐ **every (well, factor) cell's leakage lies in `[−1.0, +1.0]`**, and I predict the realised range is
inside **`[−0.6, +0.9]`**. Derivation: balanced accuracy ∈ [0,1] with realised-class chance ≥ 1/4 in
every cell here, and the denominator (a **seen**-corner balanced accuracy of a probe fitted on seen rows)
cannot approach 0 the way a per-cell raw accuracy could. A cell outside `[−1, +1]` falsifies the claim
that the corrected x-axis is bounded.

## P4 — R0's read, re-scored through the V1 decoder (⭐ the falsifier the task names)
The store-read feature vectors are **unchanged**; only the renderer changes.

| | old decoder (shipped) | predicted under V1 |
|---|---|---|
| correlation aggregate | 0.5997 | **0.66** (0.56–0.76) |
| random aggregate | 0.4492 | **0.50** (0.41–0.60) |
| **aggregate margin** | **+0.1504** | **+0.15** (+0.08 … +0.22) |
| correlation `shape` | 0.5785 | **0.64** (0.50–0.78) |
| random `shape` | 0.3770 | **0.42** (0.30–0.55) |
| **`shape` margin** | **+0.2015** | **+0.20** (+0.10 … +0.30) |

⭐ **Two competing hypotheses, both registered:**
- **H-A (primary, predicted):** V1 renders any feature vector more faithfully, so both arms lift by a
  similar amount and the margin is preserved in sign and roughly in size.
- **H-B (registered alternative):** V1 was fitted on **real** features and has 6.05× the capacity; a
  store read is an off-manifold composite of 5 codebook fragments, and a sharper decoder may be *more*
  brittle off-manifold. Under H-B both arms **fall** and the margin could compress.
If H-B is what happens, it is the result and it is reported as such.

**P4-falsifier (= the task's falsifier):** the correlation-vs-random aggregate margin under V1 is
**≤ 0**, or its magnitude is **smaller than the largest across-seed spread**. Either outcome means R0's
grouping evidence was an artefact of the superseded decoder / the degenerate normalisation.
⭐ **I commit in advance to reporting that outcome as the headline and not reaching for a fourth metric.**

## P5 — Provenance assertions (pass/fail, not predictions)
- The re-materialised store reproduces each shipped seed's `relative_mse` on `eval_a` with
  `abs_delta = 0.0`, and the shipped `read.aggregate` / `read.per_factor.shape` **exactly**.
- The re-derived grouping labels equal the shipped `arms[g]["labels"]` vector element-for-element.
- The old (degenerate) leakage numbers are reproduced from the manifest before the corrected ones are
  computed, so old and new are known to come from the same probes.
- Any seed whose feature cache must be extended asserts `_assert_cache_reencodes` `max_abs_delta = 0.0`.

⛔ **Column rule, committed here:** corrected balanced-accuracy numbers (chance 0.5, or 1/n_classes for
the multi-class leakage terms) are **never** tabulated in the same column as the old raw multi-class
accuracies. Every table below states its chance level in its header.
