# PREREG — gen-bridge-wide (filed BEFORE any encoder/decoder/probe was fitted)

**Written:** 2026-09-11, before the harness `.claude/scratch/gen-bridge-wide/wide.py` existed and before any
fitting run was launched.  Base `main @ e4f3d71`.  ⛔ Nothing below is hedged: every entry is a number or a rule.

---

## 0. What is being measured (so the predictions are unambiguous)

**Configuration.** `condition = recomb2element_wide` (13 cells × 72 = 936 held-out rows, `n_train = 479,064`).
Encoder (`recon` pretext, R1), **full-scope** feature cache (480,000 rows), decoder
(`DECODER_RECIPES["v1_capacity_schedule"]`, the `gen-decoder-lift` winner, unmodified) and probe are **all re-fitted
with the wide cells excluded**, 3 seeds {0,1,2}.  ⛔ No frozen Montero artefact is scored on a wide cell.

**The metric — DISCRIMINATION, per cell `c` with shape `s_c`:**
- **positives** `P_c` = the cell's 72 held-out rows (all of shape `s_c`);
- **negatives** `N_c` = **matched-seen**: rows satisfying the cell's *whole* conjunction with the `shape` clause
  relaxed to `shape ≠ s_c` (same hue bands, same scale corner, same orientation corner) — i.e. 216 nominal rows —
  minus every row used in fitting, minus every held-out row;
- **decision** = `argmax` of the probe's `shape` head on the **decoded** image equals `s_c`;
- **balanced accuracy** = ½(TPR over `P_c` + TNR over `N_c`); **AUROC** = over softmax `p(s_c)`, `P_c` vs `N_c`.
- **seen normaliser** `S_c` = rows with the cell's hue bands and shape `s_c` but **outside** the cell (other
  scale/orientation corners), never fitted, never held out — scored against the **same** `N_c`.
  ⚠ Declared asymmetry: `S_c` cannot match the cell's scale/orientation corner (that set *is* the cell).
- **pooled** = macro-mean over the 13 cells (and, separately, the 4-way `shape` accuracy on the 936 pooled
  held-out rows — ⛔ a **different quantity**, reported in its own column, never in the balanced-accuracy column).

---

## 1. Point predictions (point value, then 80 % interval)

| # | quantity | predicted | interval | derivation |
|---|---|---|---|---|
| **P1** | pooled macro **balanced accuracy**, held-out | **0.93** | [0.86, 0.97] | V1 reads 0.9398 ± 0.0131 of Montero's held-out cylinders as cylinders ⇒ TPR≈0.94 at the *lowest* contrast in the split; matched-seen 4-way `shape` is 0.9432 ⇒ a negative is called `s_c` at most 5.7 % of the time ⇒ TNR ≥ 0.943. ½(0.94+0.95)=0.945, discounted to 0.93 because 3 of the 4 shapes were never the held-out one before. |
| **P2** | pooled macro **AUROC**, held-out | **0.97** | [0.92, 0.995] | AUROC ≥ balanced accuracy for a calibrated-ish 4-way softmax; the argmax decision throws away the margin. |
| **P3** | pooled macro balanced accuracy on the **seen** normaliser `S_c` | **0.96** | [0.91, 0.99] | `A_matched` 4-way `shape` under V1 = 0.9432; seen positives sit at higher mean contrast than Montero's corner. |
| **P4** | **ratio** held-out / seen (the number kill rule (i) would read) | **0.97** | [0.88, 1.03] | P1/P3. V1's Montero ratio vs matched-seen was 0.9964. |
| **P5** | 4-way `shape` accuracy on the pooled 936 held-out rows | **0.90** | [0.80, 0.96] | strictly harder than binary; V1 held-out detection 0.94, minus cross-shape confusion. |
| **P6** | probe accuracy on **real** held-out images | **1.0000** | [0.995, 1.0] | R15: the probe is saturated on reals; ⛔ if it is not ≈1.0 no ceiling below is believable. |
| **P7** | no-leak, per seed | 13 cells, **72/72** excluded each, **936/936** pooled, **0 leaked** | exact | the cells are disjoint (`assert_cells_disjoint`) and the fitting rows are drawn from `split.train_rows`. |
| **P8** | ⭐ Montero reproduction: decoder param hash at seed 0, recipe `v1_capacity_schedule`, Montero encoder + cache | **`01ef91f75eeaae7b…`** | bit-exact | `gen-decoder-lift` §7. Also `shape` on Montero's B = **0.9306** and probe hash `c03bfb8916d00932…`. ⛔ A mismatch means my `--condition` passthrough perturbed Montero's path and the whole report is void. |

## 2. The contrast prediction (acceptance item 3 — the analysis the split exists for)

| # | quantity | predicted | interval |
|---|---|---|---|
| **P9** | balanced accuracy at the **lowest** contrast rung (label 0.050: `montero`, `wide-000`, `wide-006`) | **0.90** | [0.80, 0.96] |
| **P10** | balanced accuracy at the **highest** rung (label 0.450: `wide-005`, `wide-011`) | **0.96** | [0.90, 0.995] |
| **P11** | OLS slope of per-cell balanced accuracy on contrast label, 13 cells × 3 seeds | **+0.15 / unit contrast** (= +0.060 across the measured 0.05→0.45 span) | [+0.00, +0.45] |
| **P12** | **direction**: the slope is positive (higher object/background hue contrast ⇒ better discrimination) | **positive**, stated at 75 % confidence | — |
| **P13** | the named hardest cell: **`wide-007`** (shape 3, contrast 0.100) finishes in the **bottom 3** of 13 by balanced accuracy | **yes** | — (rationale: a low-contrast corner *and* the silhouette most confusable with the cylinder) |

⚠ P11's interval deliberately admits zero: R16 attributed the **`scale`** deficit to photometry and the **`shape`**
deficit to composition, so a flat contrast curve for a *shape* metric is a live possibility — and it would be a
finding (rendering and composition separate cleanly), not a failure.

## 3. ⭐ The verdict rule, fixed now

> **PASS (the decoder can discriminate held-out shapes within a corner) iff**
> (a) pooled macro balanced accuracy on the held-out cells **≥ 0.75 absolute** (0.5 is chance, 0.75 is halfway),
> **and** (b) the ratio to the seen normaliser **≥ 0.80** (the standing R3.1 floor, applied to the balanced metric
> per R20).
> **Both must hold, on ≥ 2 of 3 seeds, on the pooled macro number.**

**Predicted verdict: PASS.**

**And the consequence, pre-committed:** whatever pooled value is measured, **kill rule (i)'s ceiling is registered at
that measured level** for the wide configuration (R16/R20), reported with its per-seed spread. ⛔ If the measurement
comes in below the rule above, the honest outcome is that the 0.94 detection number was carrying more weight than it
could bear — ⭐ a legitimate, useful result, and it is filed here in advance so it cannot be re-described later.

## 4. What does NOT falsify anything
- A pooled number **below Montero's 0.94**. ⛔ 0.94 is a **detection** rate on a one-class cell; these are
  **discrimination** numbers. They are different quantities and are never compared as if they were the same.
- A **wide** ceiling below the **Montero** ceiling: different configuration, different fitted rows, different cells.
- `scale`/aggregate movement: photometric (R16), not the compositional term, and not read here.
