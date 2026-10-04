# PREREG — clu-memory-architecture (physics-theorist)

Written BEFORE running any of the five toy harnesses (protocol §5 pre-registration rule).
Date: 2026-07-21. Repo untouched (theory task); toys are self-contained numpy in
`.claude/scratch/clu-memory-architecture/`. All toys use the shipped Verlet map
(verified against `chlu/core/integrators.py::velocity_verlet_step`):
`p½ = p − (ε/2)∇V(q); q' = q + ε ∇_p T(p½); p' = (1−γ)(p½ − (ε/2)∇V(q'))`.

## P1 — Mass invariance theorem (Toy B, arm 1)
**Derivation.** For scalar mass `M` (isotropic), both Newtonian `T = pᵀp/2M` and relativistic
`T = c√(pᵀp/M + m₀²c²)` depend on `(p, M)` only through `p/√M`. Substituting `p = √M p̃`
gives Hamilton's equations for `(q, p̃)` with `M = 1` up to the time rescaling `t → t/√M`
(⇒ per-step: the M-trajectory at step n equals the M=1 trajectory at time `nε/√M`).
With `p₀ = 0` (⇒ `p̃₀ = 0` for every M), the continuum path image in q-space is therefore
**exactly M-independent**; only traversal speed changes.
**Predictions** (1-D double well `V = h(q²−1)², h=0.5`, `q₀=−0.5, p₀=0`, ε=0.02, N=40 000,
masses 0.25 vs 4.0):
- P1a: turning points (min q, max q) agree across masses to < 1e-3 absolute.
- P1b: dominant FFT frequency ratio `f(m=0.25)/f(m=4.0) = √(4.0/0.25) = 4.00 ± 2%`.
- P1c: same holds for the relativistic kinetic (c=5, m₀=1): turning points mass-independent
  to <1e-3; frequency ratio 4.00 ± 5% (larger tolerance: relativistic correction shifts both
  frequencies but the √M ratio is exact in the continuum, so deviation is discretization only).

## P2 — Mass gates energy shells (Toy B, arm 2)
**Derivation.** With `p₀ ≠ 0`, `E = p₀²/2M + V(q₀)`. Barrier crossing (1-D, from `q₀=−1`,
barrier `V(0)=h=0.5`, `V(q₀)=0`) iff `p₀²/2M > h` ⇒ threshold `M* = p₀²/(2h) = 0.36`
at `p₀ = 0.6`.
**Prediction.** On sweep M ∈ {0.10, 0.20, 0.30, 0.34, 0.38, 0.45, 0.60, 1.0}: every M < 0.36
crosses (max q > +0.5), every M > 0.36 stays (max q < 0); the measured threshold lies in
(0.34, 0.38). Discretization correction expected < half a grid cell.

## P3 — Gradient growth: regular is polynomial, chaotic is exponential (Toy C)
**Derivation.** Tangent map of a symplectic step is symplectic ⇒ singular values come in
`σ, 1/σ` pairs ⇒ ‖J_n‖ ≥ 1 always (no vanishing gradient at γ=0). Integrable anharmonic
motion has shear `∂ω/∂E ≠ 0` ⇒ ‖J_n‖ grows algebraically, exponent 1. Chaotic motion:
‖J_n‖ ~ e^{λ n ε} with λ the top Lyapunov exponent.
**Predictions.**
- P3a: 1-D quartic `V = q⁴/4` (q₀=1, p₀=0, ε=0.05): log-log slope of ‖J_n‖ vs n over
  n ∈ [10³, 2·10⁴] = 1.00 ± 0.15.
- P3b: Hénon–Heiles at E=0.16 (chaotic sea IC): Benettin λ ∈ [0.02, 0.15]; the growth-rate
  of log‖J_n‖ agrees with independent Benettin λ within 30%.
- P3c: extrapolated log₁₀‖J‖ at t = nε = 1000: chaotic ≥ 10; quartic ≤ 6.
  (⇒ ≥ 4 orders of magnitude separation between the regimes at equal horizon.)

## P4 — Two-item selective retrieval + interference/packing (Toy A)
Landscape: 2-D, `V = 0.05‖q‖² − Σ_k exp(−‖q−c_k‖²/2s²)`, s=0.35, `c_k` on circle R=2.
Addresses: `q₀ = c_k + N(0, 0.12²)`, `p₀=0`, M=1, ε=0.05, N=2000. Linear read: ridge
one-vs-all on 32 subsampled trajectory points (64 features). 30 train / 10 test rollouts per item.
**Derivation of packing onset.** Selectivity is guaranteed while `E = V(q₀) <` inter-well saddle.
Barrier vanishes when spacing `2πR/K ≲ 4.4s` ⇒ `K* ≈ 2πR/4.4s ≈ 8.2`.
**Predictions.**
- P4a: K=2 test accuracy = 100%, and unchanged at N=20 000 (conservative read does not decay).
- P4b: accuracy ≥ 95% for K ≤ 8.
- P4c: onset of degradation in (8, 16]: accuracy at K=16 < 95% (point prediction 50–95%);
  accuracy at K=32 ≤ 80%.

## P5 — Weak-form restructuring of a bad address (Toy D — the crux number)
Toy A landscape, K=4. Loss = ‖window-mean(q, last 25% of N=1200) − c_target‖².
18 deliberately-bad inits (3 wrong basins × 6 jitters), plain GD on (q₀, p₀, log M),
FD gradients, ≤ 4000 steps. Success = final window-mean nearest to target center and
loss < 0.09.
**Mechanism I expect (derived):** with γ=0 and p₀ trainable, GD can climb the energy shell
(raise E via p₀ until the trajectory covers several basins — window-mean moves off the wrong
center toward the origin, which is strictly closer to the target), then re-localize. The loss
path is connected though possibly rough near separatrices.
**Predictions.**
- P5a (committed): conservative-read (γ=0) plain-GD success rate ≥ 50% (9/18).
  [Honest uncertainty: this is the least-safe prediction in this file; failure would be a
  first-class finding against the weak-form learnability claim.]
- P5b: damped-read arm (γ_read = 0.02, same protocol): the read endpoint is piecewise-constant
  in the address (relaxation to basin fixed point) ⇒ FD gradients ≈ 0 inside a basin ⇒
  success ≤ 10%, and strictly below the γ=0 arm. (Prediction: contraction *destroys* address
  gradients; conservation preserves them — the inverse of the BPTT folk objection.)
- P5c (secondary, not part of the verdict): Adam on the γ=0 arm ≥ plain GD success.

## P6 — Mass multiplexing lifts the D1 gauge (Toy E)
**Derivation.** One item, one global (V, M): `log ω² = log k − log M` — only the difference is
task-visible; D1 partition freezes M (measured N7). Two items sharing ONE spring k with
per-item masses M₁, M₂: the difference `d = log M₁ − log M₂` evolves under GD as
`ḋ = −2η_m (d − d*)` — decoupled from k entirely, converging at rate `2η_m` regardless of
the fast lever. Only the *sum* stays gauged (and collapses per the D1 partition
`(η_m/η_k)/(1+η_m/η_k)`).
**Predictions** (η_k = 10 η_m = 0.05, targets a₁ = log 4, a₂ = log 0.25, 4000 GD steps):
- P6a: |d − d*| < 1e-3 at convergence (d* = a₂ − a₁ = −log 16).
- P6b: sum-displacement / required-sum-displacement = (η_m/η_k)/(1+η_m/η_k) = 1/11 ± 5% rel.
  (i.e. absolute masses still collapse to near-init — the vault of D1 — while the RATIO trains
  perfectly: per-address masses are learnable keys, a global mass is not).
