# PREREG — the dynamical read (gen-dynamical-read)
**Hub, 2026-09-14. Filed BEFORE any implementation or measurement.**
⛔ **Head + Advisor sign this before the scored run. Nothing here is edited after the run; amendments are appended
with a date as a NEW file `PREREG-addendum-N.md`** (an in-place addendum destroys the mtime evidence the rule exists
to create).

---

## 1. What is being tested
`request → address-book clamp + REQUEST ENERGY on the addressed wells → launch particles → settle in a per-well
landscape over the WRITTEN ATOMS with a low-rank inter-well coupling → decode the SETTLED CONTINUOUS state through
store→decoder at prototype width`.

**Atoms = stored TRAIN DATA POINTS** (D2). ⛔ **Never the `k`-means centres** — per-value weights over centres are the
old flat count table wearing a different hat.

## 2. ⭐ The claim, stated as a scaling law — ⚠ **REVISION 1, 2026-09-14, BEFORE SIGNATURE**
> ⛔⛔ **REVISION 1 — the original §2 quantiser column was DERIVED from `PR` = 6.50 and is WRONG. It was never
> measured against; `gen-atom-mean-control` measured it before this document was signed, which is exactly what that
> control existed for.** Superseded numbers are struck through below, not deleted.
> **What was wrong:** the ANCHOR was right at budget A (predicted 0.2783 vs measured centroid **0.2852**, ×1.025);
> ⛔ **the EXPONENT was not.** Measured centroid slope **−0.2501 ± 0.0050** (`PR` **8.00**), ⛔ not −0.3076 (`PR` 6.50).
> At budget B the old law predicted ~~0.1468~~ against a measured **0.1695 ± 0.0021**.

Quantisation is piecewise-constant: **ε ≈ N^(−2/PR)**. Interpolation over atoms is higher order (**ε ≈ N^(−4/PR)**).
⛔ The landscape does **not** have "no `M` in its error term" — it pays **interpolation error along the factor
manifold**. ⭐ **The claim is a BETTER EXPONENT AT MATCHED BYTES**, measured, never asserted.
⭐ **At matched bytes `N_atoms = M_centres`** — both cost `count × D × 4`. The landscape gets **no extra points**.

### ⭐ THE MEASURED BASELINES (from `gen-atom-mean-control`, 3 seeds) — these replace the derived column
| leg | budget A (N=64) | budget B (N=512) | fitted slope `d ln ε / d ln N` | implied `PR` |
|---|---|---|---|---|
| `centroid_control` k=1 *(fit-time, 0 deployed bytes, ⛔ not an arm)* | 0.2852 ± 0.0048 | 0.1695 ± 0.0021 | **−0.2501 ± 0.0050** | 8.00 |
| ⭐ **`atom_mean` oracle k=1 — THE LIKE-FOR-LIKE RIVAL** | **0.4067 ± 0.0094** | **0.2306 ± 0.0038** | ⭐ **−0.2727 ± 0.0141** | 7.35 |
| `atom_mean` best `k` | **0.3567** | **0.1851** | — | — |
| deployable atom-mean (request-addressed, best `k`) | 0.4114 | 0.2819 | — | — |

### ⭐⭐ THE CONFOUND THIS CONTROL EXPOSED — and it is the reason the arm is not yet justified
**Storing atoms instead of centres does NOT by itself buy a better exponent.** It buys a **worse level (×1.43)** and a
**slightly better slope** (−0.2727 vs −0.2501, non-overlapping over 3 seeds) — ⭐ **and the two nearly cancel.**
⇒ ⛔ **A landscape that beats the centre-codebook may be winning only by storing atoms. The atom-mean row is the one
that decides the claim.**

### ⛔⛔ THE QUERY PROTOCOL IS NOW A FIRST-CLASS PRE-REGISTRATION ITEM
With a **request-addressed** query (the deployable path — R8 embedding, ⛔ no feature tensor) the measured exponent
**collapses to −0.162 ± 0.005** (`PR` 12.4): ⭐ **the R8 embedding's own error flattens the curve by 41 %.**
⇒ ⭐⭐ **An exponent measured with a FEATURE-addressed (oracle) query is NOT the deployable exponent.**
**RULED FOR THIS RUN: the headline exponent is measured on the DEPLOYABLE, request-addressed path. The oracle
exponent is reported beside it as a companion.** ⛔ Quoting the oracle exponent as the result is forbidden.

### ⛔ RE-BASED PASS CONDITION
| | budget A | budget B |
|---|---|---|
| ⭐ **the landscape must beat (atom-mean best `k`)** | **0.3567** | **0.1851** |
| its originally pre-registered prediction | 0.0642 | 0.0163 |
| ⇒ required margin | **5.6×** | **11.4×** |
| ⭐ **exponent to beat (oracle)** | — | **−0.2727 ± 0.0141** |
| ⭐ **exponent to beat (deployable, the headline)** | — | **−0.162 ± 0.005** |
⛔ ~~The old table's 3.9× / 7.8× ratios were against the CENTRE codebook and understate the bar.~~
⭐ **A win on level with a matching exponent does not support the claim. The EXPONENT is the result.**

## 3. Budgets and byte ledger (D2, D3)
| | budget A — MATCHED | budget B — DECLARED LARGER |
|---|---|---|
| store (atoms) | **524,288 B** (64 atoms/well) | **4,194,304 B** (512 atoms/well) = 8× |
| low-rank coupling `Σ_{k<l} r(d_k+d_l)·4` | `r = 16` → **524,288 B** | `r = 128` → **4,194,304 B** |
| **total** | **1,048,576 B = 2.00×** that budget | **8,388,608 B = 2.00×** that budget |
⭐ Rank is set by D3's rule *"store + coupling ≤ 2× budget"*, binding exactly at `r = 16` / `r = 128`.
⛔ Both budgets are declared HERE, before the run.

## 4. Arms and controls — every row at EVERY budget
**Falsifier-1 rows (addition 3):** `store_removed` · `codebook-argmax` · ⭐ **D6 best-codebook (matched bytes), and
RULED: it is configured with the ERROR-OPTIMAL (`mse_optimal`) allocation — the control must be at its best** ·
**atom-mean lookup (`knn_proto` restricted to the stored atoms, matched bytes)**.
**Ceiling: `knn_proto` on the same features.** **3 seeds.**
⭐ **The read must beat BOTH like-for-like controls beyond seed spread at each budget** (D2).
**Mandatory additional controls:**
- ⭐ **request-off, same chain** (addition 1) — **steering-attributable gain is the falsifier.** ⭐ This is dreaming's
  lesson: its clamp produced a gain that the coupling-off control took entirely.
- ⭐ **coupling-off** (D3), mandatory.

## 5. ⛔ Falsifiers — all three must be answered in these words
1. **Beats `store_removed` AND codebook-argmax AND D6 best-codebook AND atom-mean lookup, beyond seed spread**, at
   each budget.
2. ⭐ **REDEFINED (addition 2): ≥ 60 / 72 DISTINCT FACTOR TUPLES as read by the EXTERNAL CLASSIFIER from the DECODED
   IMAGES.** ⛔ Not distinct settled states — continuous states are trivially distinct and that number is worthless.
3. **`scale` + `orientation` probe on the settled state above the codebook read's** (baseline `scale` 0.225,
   `orientation` 0.243).

## 6. Dynamics reporting (addition 4) — binding
**Report, every run:** acceptance · **per-step displacement from step 1** · the **energy trace every step** · the
**fraction of chains still descending at 1000 steps**.
⭐ **A chain still descending at 1000 steps is reported NOT CONVERGED — ⛔ never as a null.**
⚠ Dreaming's acceptance was **0.952** — a chain that barely moved. ⛔ Without per-step displacement we cannot tell
settling from sitting still.

## 7. Channel and the bypass guarantee (D4)
- ⭐ **The declared channel is the store's FULL WIDTH IN FLOATS** — i.e. **parity with `knn_proto`**.
  ⛔ **No draft mentions 30 bits.**
- ⭐ **The bypass guarantee is the request-only assertion over launch + settle + read AS ONE PATH.** Assert no feature
  tensor enters any of the three. ⛔ `soft_read(store, X, τ)` takes FEATURES and voids it.
- **Companion:** the **empirical spread of the settled states**, reported beside the declared width.

## 8. Cells
- ⭐ **HEADLINE: full requests.** The arm's job is **RESOLUTION INSIDE THE ADDRESSED WELLS** (D1).
- **SECONDARY, pre-registered: partial requests**, own falsifier, **rival = `knn_proto` with the marginal mean for the
  unaddressed factors**. Runs **after** the headline cell reports.

## 9. Artifact paths
`.claude/outputs/gen-dynamical-read/` — `PREREG.md` (this) · `runs/<dataset>_<pretext>/budget_{A,B}/seed_*/` ·
`energy_traces_*.json` · `settled_states_*.npz` · `curve_retention_vs_bytes.json` · `tables.md`.

## 10. What would make us wrong
- ⭐ The landscape wins on **level** but its fitted **exponent matches the quantiser's** ⇒ it is buying points, not a
  better representation. **The exponent is the claim.**
- ⭐ **Request-off gives the same gain** ⇒ the chain is settling onto a mode, not being steered. **Dreaming exactly.**
- Chains still descending at 1000 steps ⇒ **NOT CONVERGED**, and no number is quotable from that run.
- Falsifier 2 measured on settled states rather than decoded factor tuples ⇒ meaningless by construction.

## 11. ⛔ REVISION 1 — THREE FURTHER CHANGES THE BUILD FORCED (2026-09-14, before signature)
`gen-dynamical-read`'s build smoke measured three things that contradict text elsewhere. **All three bind the scored run.**

1. ⛔ **COUPLING STRENGTH CANNOT BE 1.0.** Measured **λ\* ≈ 0.054** at both budgets, capped by the coercivity
   certificate. ⭐ 1.0 was **dreaming's** value and it was carried into the design note §3 by inheritance.
   **Binding: λ is set by the coercivity certificate and reported; ⛔ no text may say `coupling_strength = 1.0`.**
2. ⛔ **DREAMING'S THRESHOLDED CLAMP RULE IS DEAD ON ATOMS** — it clamps **0.00 of 5 wells**. ⭐ The "4.42 of 5"
   figure that motivated so much of §1 of the design note is a property of the **codebook** path and **does not
   transfer**. **Binding: the launch uses the ARGMAX reading, not the thresholded clamp**, and the design note's
   "address-book clamp on the addressed wells" is corrected accordingly.
   ⚠ **Note the direction:** with 0 wells clamped the request acts **only** through the energy term (addition 1),
   which makes the **request-off control the entire test of steering.** ⭐ It is now load-bearing, not a companion.
3. ⭐ **RULED (Head, 2026-09-14): BUDGET B RUNS AT 2000 STEPS WITH CHECKPOINTING, RESUMABLE. Budget A stays at
   1000.** **Report the fraction of chains still descending at BOTH 1000 and 2000.** ⛔ **Still descending at 2000 =
   NOT CONVERGED** (resume from checkpoint if so ruled), ⛔ **never a null.**
   ⭐⭐ **⛔ NO anneal or step-size change may be chosen after seeing budget B** — that would be a tuning decision made
   on the outcome. The step count is the only thing that moved, and it moved BEFORE the scored run.
   *(Original finding that forced this:)* ⛔⛔ **BUDGET B WAS NOT CONVERGED AT 1000 STEPS** — **41.7 % of chains still descending** on `CORNER_B`.
   ⭐ R40.C's "1000 steps" is validated for budget A and **NOT for budget B**.
   **Binding, and it must be settled BEFORE the scored run:** either (a) raise the step count for budget B, or
   (b) declare a step-size / annealing change — **whichever is chosen is pre-registered here first.**
   ⛔ Running budget B at 1000 steps as it stands yields **NOT CONVERGED**, and ⛔ a not-converged chain produces no
   quotable number.
   ⭐ **The Hub does not choose between (a) and (b) — Head + Advisor do, and the choice is written into this file
   before the run.** ⬜ *awaiting ruling*

## 11b. ⭐ PRE-REGISTERED COMPANION (Head ruling 3) — is the exponent set by the FACTOR MANIFOLD?
**Hypothesis:** the codebook exponent is universal (**−0.26**) because it reflects the **factor manifold's INTRINSIC
dimension**, ⛔ not the well's linear dimension. **Prediction: dSprites (5 factors) shows a STEEPER exponent.**
Same sweep, same fit. ⭐ **Enters the paper only if it holds.**

### ⚠ HUB REFINEMENT — the point estimate needs care, and both candidates are pre-registered
A pure factor-count model gives `−2/n_factors`. ⛔ **It already mispredicts Shapes3D**: predicted **−0.333** for 6
factors against a measured **−0.2631** ⇒ **implied `d` = 7.60, an offset of ×1.267.**
⇒ two defensible point predictions for dSprites, **both filed now so neither can be selected afterwards**:
| model | dSprites prediction |
|---|---|
| pure factor count (`−2/5`) — ⭐ the Advisor's stated estimate | **−0.400** |
| Shapes3D's offset carried (`−2/(5 × 1.267)`) | **−0.316** |
⭐ **PRIMARY, DIRECTIONAL CLAIM: the dSprites exponent is STEEPER than Shapes3D's −0.2631, non-overlapping across
3 seeds.** The two point estimates are **companions**; ⭐ whichever lands is informative, and ⛔ neither may be
presented as "the prediction" after the fact.
⛔ **If the exponent is NOT steeper, the factor-manifold account is refuted** and the universality of −0.26 needs a
different explanation.

## 12. ⚠ A DEFECT THIS WAVE EXPOSED THAT IS WIDER THAN THIS RUN
⛔ **`val` is NOT exchangeable with `fit`, and EVERY gen spoke selects hyper-parameters on it.**
`select_baseline_rows` takes `fit = pool[:n_fit]` and `val = pool[n_fit:…]` of a **SORTED** row-index array, so at
production scale `val` realises **1 `floor_hue` value and 2 `wall_hue` values** against `fit`'s 10 and 10, at
**0.0924 feature variance vs 0.2229**. **Measured consequence: the two halves of `val` disagree about the winning
trial on 5 of 6 (seed, budget) cells.**
⭐ **Inherited verbatim, not introduced, by `gen-atom-mean-control`** — which is why it surfaced.
⛔ **Every arm that TUNES is affected — including LACE's whitening selection and `knn_proto`'s configuration.**
**Owner: Hub — a standing §7 entry and a decision for every tuning arm. ⛔ Not fixed inside this run.**
⭐ **RULED (Head): FIX IT — a stratified re-select — and RE-RUN EVERY CONFIGURATION SELECTION on the fixed split:
ours, `knn_proto`, and LACE (its whitening-exactness fallback re-run on the fixed split, **once**). Then RE-SCORE.**
⛔⛔ **REFUSAL'S QUOTABLE STATUS IS SUSPENDED until the rivals are re-scored.**
⛔⛔ **THE SCORED DYNAMICAL-READ RUN WAITS FOR THIS — its CEILING ROW is mis-configured until then.**
**Logged as an INHERITED negative, with the 5-of-6 disagreement as the evidence.**
⇒ **Owner: `gen-val-split-fix`. This is now the critical path.**

---
## SIGN-OFF (⛔ required before the scored run)
⚠ **This document is at REVISION 1. §2 was re-based against measured baselines and §11–§12 are new.**
⛔ **Two rulings are owed before signature: §11.3 (budget-B convergence) and §12 (the `val` split).**
- [ ] **Head**
- [ ] **Advisor**
