# Task: v2-revision-2 — fold the SF-1/2/3 measurements into V2 (w9; closes the three pending slots)

- **Agent:** `paper-writer` · **Output:** report to `.claude/outputs/v2-revision-2.md`; edit `.claude/papers/v2-short/` in place (CHANGELOG v0.3).
- **Read first:** protocol · `.claude/outputs/v2-referee-experiments.md` (THE source — SF-1/2/3 all delivered) · claims matrix v1.4 (CM-4 and CM-6 amended with these exact results — use the approved wordings).

## Items
1. **SF-1 slot (§3.2/Fig 2):** wire the Mo-estimator result — Mo's OWN λ̂(T=128) tracks the budget overdamped (corr(log pred, log meas)=0.9995, meas/pred 0.86–1.03) and fails in the same ballistic direction past EP (0.30 @ δ=4, mirroring exact-gap 0.31). Predictor-substitution closed. **Correct the 44% attribution:** −15.6% max deep-overdamped (gap·T<0.1); the −44.5% max sits at the near-EP row (gap·T≈3.1). Add the overlay `sf1_mo_estimator_overlay.png` to Fig 2 (or as Fig 2b).
2. **SF-2 slot (§3.3):** RETIRE the "≈4× longer" as a compute claim per CM-4 amendment — lead with the qualitative triad; add the honest per-step line: CLU Verlet (h64) ≈6.2× LSTM / 3.1× LEM wall (14–15× FLOPs; not width-matched — state the confound); retention-per-compute inverts (23.5×/14.6× more wall).
3. **SF-3 slot (§3.5 + App):** the laws survive the cure — anchored λ=100 @3000 ep (3 seeds): vacuum intact (r*=0.917±0.007, flat μ²≈1e-15), GMOR exact to ≤1.5e-12 over 4.6 decades, retention slope −0.956 + floor 27.03, EP φ=0 below / slope 0.5165 above (bit-identical to 150 ep). Fig `sf3_anchored3000_laws.png` → appendix. This upgrades §3.5 from "the cure holds the vacuum" to "the paper's laws hold under the cure at 20× the erosion horizon."
4. Numbers verbatim from the report; flag-provenance rows into App A. Rebuild both PDFs.

**Acceptance:** all three `[pending: v2-referee-experiments]` tokens gone; PDFs build; diff-summary per item. → `v2-referee-2` (w10) for the clean-pass verdict.
