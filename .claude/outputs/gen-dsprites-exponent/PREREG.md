# PREREG — gen-dsprites-exponent

**Filed before any codebook was fitted or any exponent measured.** Harness
`chlu/experiments/exp_gen_codebook_exponent.py` did not exist when this file was written.
⛔ No prediction below may be edited. Amendments go in a NEW file `PREREG_ADDENDUM_*.md`, dated.

---

## 0. The question
Is the codebook exponent (`ε ∝ M^a`, measured **within a well**) set by the **factor manifold** of the
dataset rather than by the well's linear dimension? Shapes3D has **6** factors; dSprites has **5**.

## 1. ⭐ THE PRE-REGISTERED PREDICTION — `gen-dynamical-read/PREREG.md` §11b, restated VERBATIM
(acceptance 1; ⛔ that file is already filed and is not edited by this spoke)

> **Hypothesis:** the codebook exponent is universal (**−0.26**) because it reflects the **factor manifold's INTRINSIC
> dimension**, ⛔ not the well's linear dimension. **Prediction: dSprites (5 factors) shows a STEEPER exponent.**
> Same sweep, same fit. ⭐ **Enters the paper only if it holds.**
>
> ### ⚠ HUB REFINEMENT — the point estimate needs care, and both candidates are pre-registered
> A pure factor-count model gives `−2/n_factors`. ⛔ **It already mispredicts Shapes3D**: predicted **−0.333** for 6
> factors against a measured **−0.2631** ⇒ **implied `d` = 7.60, an offset of ×1.267.**
> ⇒ two defensible point predictions for dSprites, **both filed now so neither can be selected afterwards**:
>
> | model | dSprites prediction |
> |---|---|
> | pure factor count (`−2/5`) — ⭐ the Advisor's stated estimate | **−0.400** |
> | Shapes3D's offset carried (`−2/(5 × 1.267)`) | **−0.316** |
>
> ⭐ **PRIMARY, DIRECTIONAL CLAIM: the dSprites exponent is STEEPER than Shapes3D's −0.2631, non-overlapping across
> 3 seeds.** The two point estimates are **companions**; ⭐ whichever lands is informative, and ⛔ neither may be
> presented as "the prediction" after the fact.
> ⛔ **If the exponent is NOT steeper, the factor-manifold account is refuted** and the universality of −0.26 needs a
> different explanation.

**Derivation of the two companions, restated so a reader need not re-derive them:** the high-rate law
`ε ≈ M^(−2/d)` with `d` the intrinsic dimension. Pure factor count sets `d = n_factors = 5 ⇒ −2/5 = −0.400`.
The carried-offset model takes Shapes3D's measured `−0.2631` ⇒ implied `d = 2/0.2631 = 7.60 = 6 × 1.267`,
and carries the **same multiplicative offset** to dSprites: `d = 5 × 1.267 = 6.335 ⇒ −2/6.335 = −0.3157 ≈ −0.316`.

## 2. ⛔ The laundering control, fixed in advance
**Shapes3D is re-run by me through the identical harness, the identical `M` grid and the identical fit.**
⛔ The number `−0.2631` is **NOT** used as the comparator for the directional claim; it is a historical reference
only. The comparator is **my own Shapes3D measurement**. If the two code paths ever diverge, the comparison is void.

## 3. The measurement, declared before it runs
- **Quantity.** Within each `(dataset, seed, well)`: OLS slope of `log ε_measured` on `log M`, where `ε` is
  `per_well_distortion(...)["per_well"][k]["relative_distortion"]` = `E‖z_k − ẑ_k‖² / E‖z_k − μ_k‖²`,
  measured on the **SPREAD** rows (held out of the store's own training pool). ⛔ Identical function, identical
  row role, identical fit form as `gen-codebook-allocation` §5d.
- **`M` grid (⭐ MATCHED ACROSS DATASETS, and matched to Shapes3D's historical range `M ∈ [4, 186]`):**
  `M ∈ {4, 8, 16, 32, 64, 128, 186}` — 7 points, endpoints exactly `4` and `186`.
  ⛔ Both datasets get the same grid; an exponent over a different range is not comparable.
- **`K = 5` wells on BOTH datasets** (Shapes3D's shipped `K`), partition **re-derived per seed** by the shipped
  `channel_correlation` + `grouping_labels(..., "correlation", ...)` procedure on that seed's own `store_train`
  features — ⛔ the same procedure on both datasets, never a loaded artefact on one and a derivation on the other.
- **Rows:** `select_store_rows` verbatim (`n_store_train = 32,768`), `SPREAD` = 4,096 rows drawn from
  `train_rows \ (store_train ∪ eval_a)` by `gen-codebook-allocation`'s own RNG stream.
- **Store recipe, identical everywhere:** `train_steps=1200`, `train_batch=128`, `train_lr=3e-3`,
  `commitment_beta=0.25`, `tau 1.0 → 0.1`, key `PRNGKey(seed·10007 + 20260911)`.
- **Seeds 0, 1, 2** on both datasets. `±` is population sd (ddof = 0) unless a row says otherwise.
- **Aggregation.** `exponent(dataset, seed)` = **mean over the 5 wells** of the per-well slope.
  `exponent(dataset)` = mean ± sd over the 3 seeds. A well-level figure (mean ± sd over all 15 `(seed, well)`
  rows) is reported beside it because that is the form `−0.2631 ± 0.0555` took.
- **CI.** Seed-level 95 % CI = `mean ± t(2, 0.975)·sd(ddof=1)/√3`, `t = 4.303`. Reported for both datasets.

## 4. ⭐ The decision rule for the PRIMARY claim, written before the numbers exist
Let `S = {exponent(shapes3d, s)}`, `Dp = {exponent(dsprites, s)}`, `s ∈ {0,1,2}`.
- **"STEEPER, non-overlapping across 3 seeds" = YES** iff `max(Dp) < min(S)` (every dSprites seed is below every
  Shapes3D seed) **and** `mean(Dp) < mean(S)`.
- **Otherwise NO** — and a NO **refutes the factor-manifold account** (§11b's own words), which is a result.
- ⛔ A NO is not rescued by a well-level or pooled statistic that happens to separate; those are reported but the
  pre-registered rule is the seed-level one above.

## 5. Point predictions (⛔ filed; each is scored HELD / FALSIFIED with its interval)

| id | quantity | prediction | 80 % interval |
|---|---|---|---|
| **E1** | ⭐ my Shapes3D exponent, 7-point sweep, seed-mean | **−0.26** | [−0.36, −0.18] |
| **E2** | ⭐ my dSprites exponent, 7-point sweep, seed-mean | **−0.33** | [−0.48, −0.22] |
| **E3** | ⭐⭐ **the PRIMARY directional answer** (§4 rule) | **YES (steeper)** — stated probability **0.55** | — |
| **E4** | which companion is closer to the measured dSprites exponent | **−0.316** (the carried offset) | — |
| **E5** | Shapes3D seed-level sd of the exponent | 0.02 | [0.002, 0.08] |
| **E6** | dSprites seed-level sd of the exponent | 0.03 | [0.002, 0.10] |
| **E7** | fraction of `(dataset, seed, well)` cells with a monotone-decreasing `ε(M)` over the 7 points | 0.95 | [0.70, 1.00] |
| **E8** | Shapes3D `ε` at `M = 64`, variance-weighted, seed 0 | 0.28 | [0.20, 0.38] |
| **E9** | dSprites `ε` at `M = 64`, variance-weighted, seed 0 | 0.30 | [0.15, 0.50] |
| **E10** | ⛔ dSprites realised split: total / test / train | **737,280 / 6,292 / 730,988** | exact |
| **E11** | my re-derived Shapes3D partition reproduces `gen-commitment-collapse`'s `labels_vec` bit-identically | **YES** on 3/3 seeds | — |
| **E12** | per-well `PR` (participation ratio), dSprites, range over wells+seeds | 3 … 14 (same band as Shapes3D) | [1.5, 40] |

**E1's derivation:** `gen-codebook-allocation` §5d measured `−0.2631 ± 0.0555` over 15 `(seed, well)` rows from a
3-point, uneven `M` sweep. A denser, matched, log-spaced sweep over the same range should recover the same slope if
the exponent is a property of the data rather than of that sweep's geometry.
**E2's derivation:** the midpoint of the two filed companions (`−0.400`, `−0.316`), widened to admit "no change"
at the top (`−0.22`) so the interval does not encode the conclusion.
**E3's probability:** the factor-manifold account is one of at least three live explanations for a near-constant
`−0.26` (the others: we are below the high-rate regime entirely, so the exponent is an artefact of the regime and
would be ≈ −0.26 on ANY dataset; or a heavy eigenvalue tail floors the distortion). 0.55 is deliberately not high.
**E4:** `−0.316` is chosen because the pure-factor-count model already mispredicted Shapes3D by the same kind of
offset; if that offset is a property of the estimator it should carry. ⛔ This is a *guess between companions*,
recorded so it cannot be selected afterwards — it is **not** "the prediction".

## 6. What this spoke does NOT do (declared scope)
⛔ No probe, no decoder, no rendered image, no retention ladder, no request side, no arm, no CLU claim, no verdict
on any dial. The decoder is **not loaded**, so `dec_recipe` cannot silently default to the failed `baseline`
recipe — declared here because the task names that trap. ⛔ No aggregate-ceiling number is quoted for dSprites
(the dSprites probe is not saturated; `gen-bridge-dsprites` §5).
⛔ The dSprites angle constants (`a120 = 4π/3`, `a240 = 2π/3`) and `montero_test_mask` are **not touched**.
