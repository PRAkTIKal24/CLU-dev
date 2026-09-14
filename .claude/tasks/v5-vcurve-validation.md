# v5-vcurve-validation — results-analyst (ME-1: the rollout-validated V-curve · ME-3: the vault on an emergent checkpoint)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 26; the quality-first posture's V5 candidates, 2026-08-19).** Read `.claude/AGENT_PROTOCOL.md`, then this file. You write `.claude/outputs/v5-vcurve-validation.md` + artifacts under `.claude/outputs/v5-vcurve-validation/`. ⛔ No model-code changes; existing harnesses only (`v5-gate`'s R3 vault harness; the t-lever/v5-gate checkpoint sets).

**DIAL DECLARATION: none — instrument validation + generalization probe on banked checkpoints. Laundering control: n/a for ME-1 (instrument cross-check); for ME-3 the vault's own scalar control (the γ-rescale) travels as in `v5-gate` §2.1. Falsifies: pre-register your expected ranges below BEFORE running; a miss is the headline, not a footnote.**

## PREREG FIRST
§1 of your report, written and saved before any run: expected rollout/Jacobian ratio range (the banked spot-check says 1.23–1.77× at γ=0.05 — state what range across the γ-grid would count as "the Jacobian instrument is validated with a stated bias" vs "the V-curve shape itself is unreliable"); for ME-3, the vault prediction on emergent checkpoints above T\* (does D̂ ∝ 1/γ_eff transfer to a register that is a soft mode rather than an exact coset?) with a numeric success/failure line.

## ME-1 — the rollout-validated V-curve (closes the MF-3 caveat with data)
Direct T=0 rollout `n₁/₂(γ)` on the dense γ-grid, designed (5 seeds) + emergent (3 seeds), same checkpoints as the banked curve; report the rollout curve beside the Jacobian curve, the per-γ ratio curve, whether the ARGMIN and the two SLOPES survive on the rollout instrument (the claims live on shape, not level), and the collapsed-variable comparison. Every number with seed spread.

## ME-3 — the vault on an emergent checkpoint (converts the paper's most quotable number from designed-only, or produces a first-class negative)
The γ_φ friction-hole + D̂ estimator on `emergent150_s{42,43,44}` at T > T\* ≈ 3e-3, per the existing `v5-gate` R3 harness; the scalar-control comparison beside it. Either outcome is a deliverable: a transferring vault (state the ratio + spread) or a clean designed-only scope confirmation (state the failure mode).

## Rules
1. Prereg precedes results in the file; both instruments' numbers side by side; no verdict language beyond the prereg's own success/failure lines — adjudication is the Advisor's/Head's.
2. Wall-clock and env in the provenance block; laptop-CPU only; suite untouched.
3. ⛔ Results are NOT wired into any draft by you — the writer folds after Advisor verification (v0.4, not v0.3).
Standard `## Proposed handover updates` + `## Flags`.
