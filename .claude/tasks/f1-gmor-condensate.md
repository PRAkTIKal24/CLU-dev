# Task: f1-gmor-condensate — upgrade V2's headline from "μ²∝δ" to GMOR PROPER with a measured condensate (deep-dive F-1; Head-approved)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/f1-gmor-condensate.md` (+ figure/npz under `.claude/outputs/f1-gmor-condensate/`)
- **Read first:** protocol · **`.claude/outputs/v2-symmetry-deepdive.md` §2 (the whole ChPT section) + S1/S3 + F-1** (the exact prescription) + its scratch `.claude/scratch/v2-symmetry-deepdive/checks2.py` §A′ (the verified toy) · `chlu/core/potentials.py` (`SO2InvariantPotential`, `TiltedPotential` — the precedent wrappers) · `v2-full-runs.md` / `v2-referee-experiments.md` SF-3 (the trained SO(2) checkpoints: designed150 + anchored3000 `.pkl`).
- **Why:** the shipped `TiltedPotential` is an *angular* tilt (radius-independent) so it measures only the product μ²F²=δn² — it **cannot see the condensate Σ**. A *linear ambient* spurion resolves all three GMOR objects independently, turning V2's "a tilt produces μ²∝δ" into **GMOR proper: μ²F²=δΣ exact, with Σ=r* measured and a resonance-saturated NLO coefficient**. Probe-only — **NO retraining** (a new potential wrapper applied to existing trained checkpoints).
- **Git:** branch `agent/experiment-engineer/f1-gmor-condensate`; base local `main` (NOT origin/main — protocol §3.5).

## Items
1. **`LinearSpurionPotential(eqx.Module)`** wrapper (mirror `TiltedPotential`): V(q) = V_base(q) − δ·(u·q), a linear ambient spurion along a fixed direction u (the ChPT quark-mass term). Test: δ=0 reduces bit-exactly to V_base.
2. **The three independent measurements** on a trained SO(2) checkpoint (reuse the v2-full-runs probe harness verbatim), swept over δ∈[1e-8, 0.3]:
   - μ² = the Jacobian/Hessian spectral gap at the tilted vacuum;
   - **F² = M_ch·r*²** (coset inertia × vacuum radius²);
   - **Σ = r*(δ)** = the measured vacuum radius (the order parameter / condensate; = −∂E_vac/∂δ).
   Deliverable: the **one-identity table** μ²F² = δ·Σ (expect machine precision, per the toy's 2.2e-16) — three measurements, one exact law.
3. **The NLO coefficient (resonance saturation of the leading LEC):** measure LO-GMOR relative error (μ²_LO − μ²)/μ² and check it equals the **predicted** δ/(M_ch·μ_rad²·r*) — i.e. the leading low-energy constant is saturated by the radial (σ/Higgs) resonance μ_rad, measurable on the checkpoint. (Toy verified 0.9959.)
4. Deliverable for the V2 writer: an appendix-ready figure (μ²F² vs δΣ collapse + the NLO coefficient) + the 2–3 sentence GMOR-proper wording, with the honest note that the *angular* tilt in the main text remains the clean power-law verification (C-2) while this appendix resolves the condensate.

## Acceptance
GMOR-proper demonstrated on a TRAINED checkpoint (not just the toy): μ²F²=δΣ to ~machine precision + the resonance-saturated NLO coefficient measured. Full suite green (new wrapper + its test); flag-provenance per §5 (checkpoint id, δ grid, u direction). This lands in V2 as an appendix (v2-revision-4) — upgrades the headline for a HEP reader at zero retraining cost.
