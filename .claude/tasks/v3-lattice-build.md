# Task: v3-lattice-build — the CLU lattice: scale by mass AND size (V3 anchor, first build)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/v3-lattice-build` · **Output:** `.claude/outputs/v3-lattice-build.md`
- **Read first:** protocol · **F5 §7 (Def-4, §7.2 composition conditions — position-only coupling, uniform-γ conformality, "inter-unit communication has a mass")** · brainstorm Thread 5 (+ wave-2 update: **mass hierarchy must be DESIGNED-IN — it does not emerge**) · handover §7. Def-2 nomenclature.

**Goal:** the first working CLU-Net: a joint-Hamiltonian lattice of N units with designed inertial-mass banding and measured communication pricing. This is the V3 short's engine; keep it minimal and *measurable*.

## Build (new `chlu/core/lattice.py` + experiment)
1. **`CLULattice(eqx.Module)`:** units i=1..N with per-unit dims d_i, per-unit potentials V_i (reuse existing potential types incl. `so2_invariant`), **position-only coupling** `V_c(q_i, q_j)` on a config-declared edge list E (start: quadratic/spring coupling `κ_c·‖W_i q_i − W_j q_j‖²` with learnable small W's, plus an optional MLP coupling — flag-selected), one global Verlet step on the concatenated state (F5 §7.2 condition 2), γ uniform (condition 3; per-unit γ_i behind a flag with the conformality caveat documented). Joint H must reduce EXACTLY to N independent CHLUs at κ_c=0 — that's the correctness test.
2. **Designed mass banding:** per-unit `inertial_mass_scale` config (heavy/slow backbone units vs light/fast perception units — Thread-5 doctrine); expose `mass_vector()` per unit; verify the anisotropic causal caps v_max,i = c/√M_i per band (F5 Prop-1).
3. **Communication-pricing measurement (the first V3 result, F5 §7.2 quadratic-order claim):** 2 units with SO(2) channels + V_c invariant only under *simultaneous* rotation. Measure vs κ_c: (a) sync timescale of the relative angle (predicted ∝ κ_c^{-1/2}); (b) relative-information retention (predicted overdamped n₁/₂ ∝ 1/κ_c); (c) the diagonal (shared) channel stays an exact latch at every κ_c. Reuse `goldstone_harness` on the joint state. **This one plot — "coupling strength prices communication speed against relative-memory lifetime" — is the build's acceptance centerpiece.**
4. **Scaling smoke:** N ∈ {2, 4, 8} chain topology; verify joint symplecticity (‖JᵀΩJ−Ω‖ at γ=0 via `step_jacobian` on small N), energy behavior, and wall-clock/step scaling (informs CSF3 needs).
5. **Wormhole slot (skeleton only):** the edge list accepts non-adjacent pairs with a smooth energy gate on V_c (F5 §7.4 smooth-gate variant); one smoke test that a distant pair couples. No top-k selection logic yet.
6. **Training smoke (small):** train a 2-unit lattice on a toy composite task (e.g., each unit's channel tracks a different-frequency component of a joint signal — simplest "two timescales, two bands" demo of mass banding). Quick epochs; report whether banded beats uniform-mass at matched params (single seed = indicative only, say so).

## Tests
κ_c=0 reduction (bit-level vs independent units) · joint symplecticity · Noether for simultaneous rotation (charge conserved at γ=0 / exact decay at γ>0) · pricing-law smoke on a hand-built quadratic lattice against F5's quadratic-order prediction · causal caps per band.

**Scope guards:** NO momentum/velocity coupling (breaks separability — F5 §7.2 condition 1; if a design corner seems to need it, flag, don't implement). No γ_φ(q) field here (separate task). Reversible-BPTT is a later task — but keep rollouts scan-based/pure so it stays possible. Don't gold-plate topology configs; chain + one wormhole edge suffices.
