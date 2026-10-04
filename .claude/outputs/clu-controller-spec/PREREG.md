# PREREG — clu-controller-spec numerical checks (written BEFORE any harness ran)

Date: 2026-07-21. Agent: physics-theorist. All checks are self-contained numpy toys (no repo code), 2-D
Gaussian-atom landscapes `V = α‖q‖² − Σ A_i exp(−‖q−c_i‖²/2s²)`, damped velocity Verlet identical in form
to `chlu/core/integrators.py` (Newtonian T, M=1). Seeds fixed in-script (`default_rng(0..3)` per script).
Predictions and derivations committed here first; scripts in `.claude/scratch/clu-controller-spec/`.

## Script 1 — consistency + implicit-function differentiability (`s1_consistency.py`)
Landscape: K=4 ring R=2, s=0.35, A=1, α=0.05 (Toy-A parameters). Relax: γ=0.05, ε=0.05, N=3000.

- **P1 (relaxation landing):** queries jittered σ=0.12 around each center land in the correct basin with
  accuracy ≥ 0.99 (regime-1 replication; derived from Toy A / retrieval-demo behavior at this σ/spacing ≈ 0.04).
- **P2 (fixed-point drift law):** under a smooth global perturbation δV = a·sin(k·q+φ), the fixed-point
  displacement obeys the implicit-function first-order law δq* = −H⁻¹∇δV(q*). Predicted: relative mismatch
  ‖δq*_meas − δq*_pred‖/‖δq*_meas‖ ≤ 20% at a = 1e-2, and the mismatch scales ≈ linearly in a
  (first-order truncation ⇒ O(a) relative error). Derivation: ∇V(q*+δq*) + ∇δV = 0 ⇒ Hδq* ≈ −∇δV.
- **P3 (IFT vs finite difference):** dq*/dA_j for a *neighboring* atom amplitude matches −H⁻¹ ∂_{A_j}∇V(q*)
  to ≤ 1% relative error (Hessian well-conditioned inside a basin; λ_min ≈ A/s² ≈ 8).
- **P4 (contraction, Prop-5 replication):** finite-difference sensitivity of the *relaxation endpoint* to the
  *initial condition* q₀ (within-basin, δ=1e-4) is ≤ 1e-6 — the address-search gradient is numerically dead
  while P3's θ-gradient is O(1/λ_min) ≈ 0.1. Predicted contrast ≥ 5 orders of magnitude.

## Script 2 — non-degeneracy: gauge, injectivity, margin, confinement (`s2_nondegeneracy.py`)
Landscape: K=8 ring R=2, s=0.35, α=0.05; heterogeneous amplitudes A_i ∈ [0.7,1.3] (rng). Read: relax
γ=0.02, N=1500 from query, classify by nearest stored address. 64 queries/item.

- **P5 (assignment is gauge):** 10 random item→well permutations give identical retrieval accuracy —
  spread (max−min) ≤ 1% (T2/T3: assignment carries no performance weight once margins hold).
- **P6 (injectivity necessary):** assigning 2 items to one well makes that pair undecodable: pairwise decode
  for the merged pair ≈ chance (≤ 0.6), all other items unaffected (≥ 0.98).
- **P7 (margin law is the Gaussian-tail law):** accuracy vs angular jitter follows
  acc ≈ 2Φ(Δθ/2σ_θ)−1 (Δθ = 2π/8): predicted 0.988 at σ/spacing = 0.20, 0.955 at 0.25, 0.905 at 0.30,
  0.85 at 0.35; measured within ±0.03 of the law for σ/spacing ≤ 0.35. This makes the engineer's
  "break at σ/spacing ≈ 0.2" a *predictable admission criterion*, i.e. margin_i ≥ κσ with κ ≈ 5 for
  99%-grade retrieval.
- **P8 (confinement necessary):** giving queries tangential |p₀| with E = 1.5·h (h = measured inter-well
  saddle) collapses accuracy to ≤ 0.6; E = 0.5·h keeps ≥ 0.95 (Prop 2 in read form).
- **P9 (address error below margin is free — deadband basis):** offsetting every stored address by
  δ ∈ {0.1, 0.2, 0.3}·(half-spacing arc) changes accuracy by ≤ 1%; a cliff appears only when the offset
  crosses the separatrix (offset ≈ half-spacing). Wake-null (T2) verified in read space.

## Script 3 — rewrite stability + capacity admissibility (`s3_admission.py`)
Box landscape, s=0.35, A=1, α=0.02; sequential admission of up to 20 items, proposals uniform in [-2,2]².
Gate: admit iff min-dist ≥ d_safe = 4.4s ≈ 1.54 (else relocate to best admissible of 400 candidates, else refuse).

- **P10 (corruption bound):** incremental drift of each pre-existing fixed point per admission obeys
  ‖δq*‖ ≈ ‖∇δV_new(q*)‖/λ_min(H) within a factor 2 for gated admissions; gated drift ≤ 1e-3
  (predicted ≈ 1e-4 at d = 1.54: ∇δV ≈ A(d/s²)e^{−d²/2s²} ≈ 8e-4, λ_min ≈ 8).
- **P11 (regime-2 avoidance):** with the gate, overall selectivity stays ≥ 0.95 at every K reached
  (refusals allowed); without the gate, selectivity falls below 0.8 once any pair sits closer than ≈ 2s
  (partial merger / washboard-transport onset).

## Script 4 — deadband anti-thrashing (`s4_deadband.py`)
K=6 ring R=2; 300 "epochs"; per-epoch center jitter σ_dr = 0.02 (transient, around base) + one item's base
drifting 0.004/epoch in a fixed direction (total 1.2 > basin margin ≈ 1). Controller re-derives addresses
each epoch (Newton from stored address); policies: always-commit / deadband δ_dead = 0.3 / never-update.
Retrieval tested every 10 epochs (σ_q = 0.1 queries, relax-read).

- **P12 (deadband costs zero):** accuracy(deadband) − accuracy(always) = 0 ± 0.02 at every checkpoint.
- **P13 (thrash suppression):** update count always ≈ 6·300 = 1800; deadband ≤ 10 total
  (≈ ceil(1.2/0.3) = 4 for the drifter + ≈ 0 false triggers since σ_dr = 0.02 ≪ δ_dead); ratio ≥ 100×.
- **P14 (never-update fails):** the drifting item's retrieval falls below 0.5 by epoch 300 under
  never-update (stored address exits the migrated basin); other items stay ≥ 0.95.

Deadband width condition being tested: σ_noise(address re-derivation) ≪ δ_dead < margin − κσ_q.
