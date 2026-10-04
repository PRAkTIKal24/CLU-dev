# PREREG — gen-baselines-consolidate

**Filed 2026-09-13, before writing `.claude/outputs/gen-baselines.md`, and before any A7-controlled number
exists anywhere for the competitor arms.**

## 0. Why a prereg here
My acceptance criterion is a *report*, not a measured law, so protocol §5's pre-registration rule does not
strictly bind. I file one because **the deliverable will contain a prediction** — the direction and rough
magnitude of the A7-controlled C re-score, which I have established **cannot be computed from the artefacts**
(§ below). Protocol §5's "why this exists" paragraph is explicit that predictions and parentheticals are
exactly where wrong numbers propagate. So the prediction goes here, with its derivation, and the report
carries it as **NOT RUN + prediction**, never as a result.

## 1. What I established BEFORE predicting (mechanical, checkable)
- `git show main:chlu/experiments/exp_gen_baselines.py | grep -i a7` → **no match**. Neither
  `chlu/eval/rivals/gen_knn.py` nor `chlu/core/gen_request.py` reads `a7_redundant.json`.
  ⇒ the arms never scored an A7 request. (This is `gen-c-set-a7`'s own reconciliation item 1.)
- No per-request refusal score is persisted: the manifests carry only
  `refusal.score_summary.{A,B,C,MS}.{mean,std,n}` and the AUROCs. Grep for lists > 10 elements in a
  manifest returns only `tuning.trials` (36) and `or_not_calibration.grid` (16).
  ⇒ **no post-hoc re-score is possible from disk.** The A7 column requires a re-run of the scoring path.
- `gen-c-set-a7`'s registry assigns controls **per kind**: `duplicate_factor` → `a7_redundant`;
  `out_of_range` → `base` (the ordinary 6-term valid pool). ⇒ only the **dup half** moves under A7.

## 2. THE PREDICTIONS (point + interval + falsifier)
Quantity = AUROC(control vs C), kind = `duplicate_factor`, control = the 72 `a7_redundant` requests,
each arm at its **pre-registered selected configuration**, seeds {0,1,2}, everything else unchanged.
Contaminated reference (control = B): **kNN 0.3907 ± 0.1670**, **LACE/ode 0.9991 ± 0.0012**.

| # | quantity | point | interval | falsified if |
|---|---|---|---|---|
| Q1 | `knn` / `knn_proto` dup AUROC under A7 | **0.30** | 0.15–0.45 | outside |
| Q2 | `lace_ode` dup AUROC under A7 | **0.95** | 0.85–1.00 | outside |
| Q3 | ordering: LACE dup ≫ kNN dup survives A7 | — | — | the ordering inverts or the gap closes below 0.20 |
| Q4 | `out_of_range` AUROC is **numerically unchanged** under A7 | exact | — | any change at all |
| Q5 | pooled A7-controlled AUROC = ½(dup_A7 + oor) exactly | exact | — | any deviation > 1e-9 |
| Q6 | the no-model structural detector on (C-dup vs A7) | **0.5000** | — | > 0.55 |

**Derivations, so a hit is evidence and not a guess.**
- **Q1.** The kNN refusal score is `min_j ‖x̂ − c_j‖` on the composed prototype
  `x̂ = μ + Σ_terms (μ_{k,v} − μ)`. A **C-dup** request adds a *conflicting* same-factor deviation, so the two
  deviations partially cancel and `x̂` lands *near* an intermediate, on-manifold point ⇒ score LOW ⇒ the
  contaminated AUROC is already **below chance** (0.3907). An **A7** request adds a *duplicate of the same
  value*, so that factor's deviation is **doubled** and `x̂` is pushed **further off-manifold in the requested
  direction** ⇒ A7 scores rise above ordinary 6-term valid scores. Replacing B (lower) by A7 (higher) as the
  negative class pushes an already-below-chance AUROC **further down**. Hence < 0.3907, point 0.30.
- **Q2.** LACE's score is a summed composed energy. A7's extra term duplicates an **already-satisfied**
  constraint, so it adds ≈ one more *small* per-term energy: A7 ≈ (7/6)× a valid 6-term energy. C-dup's extra
  term is **unsatisfiable**, contributing a large irreducible residual — seed 0 measured C mean 7.13 vs B mean
  1.54 (`ld_published`), a 4.6× gap. A 1/6 inflation of the control cannot close a 4.6× gap, so the separation
  survives with a modest loss. Hence ≥ 0.85, point 0.95 (below the contaminated 0.9991, not at it).
- **Q4.** The A7 registry leaves `out_of_range`'s control at `base`; the baselines already scored it against
  6-term valid requests, and `gen-c-set-a7` measured the structural detector at exactly 0.5000 on that pairing.
  Nothing about the oor half changes. ⇒ **half the re-score is already done.**
- **Q5.** Verified to machine precision on the contaminated set (`max |pooled − ½(dup+oor)| = 1.11e-16`,
  36 = 36 C items per kind), and A7 does not change the kind counts.
- **Q6.** Measured by `gen-c-set-a7` on this exact pairing: 0.5000 on all 11 structural features.

## 3. ⛔ What I will NOT do
⛔ No kill-rule verdict (that is `gen-cell-run`'s). ⛔ No edit to `chlu/`, no branch, no worktree.
⛔ No re-run of any arm. ⛔ No A7-controlled number reported as measured — Q1–Q3 ship as **predictions
awaiting a run**, in a table headed NOT RUN.
