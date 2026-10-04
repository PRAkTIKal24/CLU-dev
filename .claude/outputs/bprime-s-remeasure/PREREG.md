# PREREG — bprime-s-remeasure (results-analyst)
Written 2026-08-18, **before** any measurement harness was executed. Protocol §5 pre-registration rule
(the acceptance criterion is a measured quantity `s` and a measured ratio `s_unsubtracted / s_subtracted`).

## 0. What is being measured
`s` (effective well width) on `bprime-c6`'s banked cell family — `overload/load1x_shipped`,
`ball_radius ∈ {0.42, 0.55, 0.64, 0.80, 1.00, 1.20}`, seeds {0,1,2} — under TWO conventions:
- **(U) un-subtracted**: fit the radial profile of `V` as-is;
- **(S) subtracted**: fit the radial profile of `V(q) − α‖q‖²`, α = `confine` = 0.05.
Two estimator families are evaluated on the SAME rebuilt store:
- **E1** = `CluSystem._well_fit` (Gaussian-basis `D(1−e^{−r²/2s²})`, 8 dirs × 12 radii ∈ [0.15,1.5], grid
  search s ∈ [0.05,1.2] at 120 points) — the estimator that produced `bprime-c6`'s `s_fitted_well`.
- **E2** = `chlu.core.factored_store.effective_s` (log-linear slope of `ln profile` vs `r²`, 16 rays × 24
  radii out to `4·s_hint`) — the `orgdiv-cat-test` estimator that produced N224's 0.438 / 0.304 / 0.318.
- **E3** = the T5.5 coupling-law fit (`ln κ − ln(d/σ_q)` vs `d²`, slope `−1/2s²`) that produced
  `bprime-c6` §1.1's `s = 0.3979`, `R² = 0.9953`.

## 1. Registry-pre-registered direction (the one my task file carries, N224 / claims-matrix §0.9)
> "the correction makes `s` SMALLER and `d/s` LARGER."
Quantified from N224's own numbers (0.438 → 0.304 on the cat-test store = 1.44×):
- **P-R1**: corrected `s` at the shipped cell (R = 1.00) drops from the banked **0.3625** to
  ≈ **0.25** (0.3625 / 1.44), i.e. a drop of ≥ 20 %.
- **P-R2**: `d/s` at the shipped cell rises from the banked **3.59** (fitted ruler) to ≈ **5.2**.
- **P-R3**: `bprime-c6` §1.1's headline `s = 0.40` becomes ≈ **0.28**.

## 2. Competing hypothesis, derived from reading the shipped estimator BEFORE measuring (mine)
Derivation (code read, done before any run; all three files inspected at the exact commits):
- `chlu/core/clu_system.py::_well_fit` at `d4f56c8` (bprime-c6's base), at `be995ca` (bprime-c6's branch
  tip) and at HEAD `7fcef50` is **byte-identical** (verified by extracting the function from
  `git show <sha>:...` and comparing strings), and its line
  `conf = confine * (Σ pts² − Σ z²)` followed by `y = (vals − conf − v0)` **already implements the
  α‖q‖² subtraction analytically**, with α = `cfg.confine` = 0.05 — exactly the store's potential term
  (`chlu/core/memory_potentials.py`: `return v + self.confine * jnp.sum(q**2)`).
- `bprime-c6`'s `grad_ratio` (E3's y-values) is built from
  `‖∇V_full − ∇V_{−k}‖` (`exp_route3_attribution.py` L490–L500), a **difference of two potentials that
  share an identical, static confinement term** ⇒ α cancels **exactly**, independent of its value.
Therefore:
- **P-M1**: E1(S) reproduces the banked `s_fitted_well` **to all printed digits** on every cell
  (Δ = 0 exactly), i.e. the correction moves `s` by **0.0 %**, not −31 %.
- **P-M2**: `d/s` under the corrected convention is **unchanged** (shipped cell stays 3.59 fitted /
  4.34 atom-width), i.e. the registry's "d/s LARGER" does **not** occur.
- **P-M3**: E1(U) (subtraction switched OFF on the same store) inflates `s`; I pre-register the inflation
  factor in **[1.10, 1.80]** with a point prediction of **1.44×** (transferring N224's measured factor),
  i.e. E1(U) at the shipped cell ≈ **0.52** (band 0.40–0.65).
- **P-M4**: E3's `s_implied` is **bit-identical** under both conventions (the cancellation above), and
  §1.1's `R² = 0.9953` is **invariant** to the `s` convention because `s` is the fit's OUTPUT, not its
  input (the fit is κ vs `d²`; no `s` enters the regression).
- **P-M5** (cross-estimator, the one genuine open number): E2(S) on the c6 store will **not** equal
  E1(S); I pre-register E2(S) at the shipped cell in **[0.25, 0.40]** and E2(U)/E2(S) in **[1.2, 2.0]**.
  A materially lower E2(S) would mean `s = 0.40` is **estimator-dependent** even under the corrected
  convention — a different finding from N224's, and it would be reported as such, not as a confirmation.

## 3. How the two hypotheses are separated
Rebuild the banked cells deterministically (same seeds, same config, same commit `be995ca`; nothing
retrained beyond re-running the deterministic write) and require the rebuild to reproduce the banked
`s_fitted_well` digit-for-digit before ANY convention comparison is read. Then E1(S) vs E1(U) vs
E2(S) vs E2(U) on the identical store object.
- **Registry direction survives** iff E1(S) < banked `s` by ≥ 10 % on ≥ 2 of 3 seeds at the shipped cell.
- **Registry direction is falsified** iff E1(S) == banked `s` (Δ < 1e-6) — the correction is a no-op
  because it was already applied.
- Either way the un-subtracted inflation factor (E1(U)/E1(S), E2(U)/E2(S)) is reported as the measured
  size of the hazard N224 describes, on THIS rig.

## 4. Falsifier for my own hypothesis
If E1(S) ≠ banked `s` on any admissible cell, P-M1 is dead and the registry's flag bites `bprime-c6`
directly; the report leads with that.
