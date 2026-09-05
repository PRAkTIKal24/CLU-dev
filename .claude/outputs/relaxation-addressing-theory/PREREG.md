# PREREG — relaxation-addressing-theory (w20, physics-theorist)

Written **before** running any harness and **before** reading `.claude/outputs/learned-landscape-write-read.md`
(existence of that file verified by `ls` only; contents unread at time of writing).
Date: 2026-07-21. All toys: self-contained numpy, no repo code, no JAX.

Shared landscape (Items 3/4), identical family to w19 Toy A: 2-D,
`V(q) = 0.05‖q‖² − Σ_{k=1..4} exp(−‖q−c_k‖²/(2s²))`, s = 0.35, centers on ring R = 2 at 0°/90°/180°/270°.
Dissipative Verlet: ε = 0.05, γ = 0.02, M = 1 (Newtonian T), rest-start relaxations N = 3000–4000.

## Analytic geometry predictions (derived by hand before measuring)
- Well bottom V(c_k) ≈ −0.80 (= 0.05·4 − 1 − negligible neighbor terms).
- Inward (lowest) saddle at ≈1 unit inward of c_k, V_saddle ∈ [0.02, 0.05] (hand estimate 0.033).
- Escape barrier from well bottom **h ∈ [0.75, 0.90]** (hand estimate 0.83).
- The origin is a spurious local minimum (confinement term; Gaussians negligible there), V(0) ≈ 0,
  with its own shallow escape barrier **h_origin ∈ [0.02, 0.06]** — i.e. a "no-item" trap that is cheap to escape.
- Basin margin (distance from c_k to nearest foreign-labeled point) **margin ∈ [0.85, 1.15]** (hand estimate ≈ 1.0, inward).

## Item 1 — back-door gradient through the address definition (script `item1_backdoor.py`)
1-D double well V(q;θ) = (q²−1)² + θq; damped relaxation (ε=0.05, γ=0.02, N=4000) from q₀ in the right basin;
forward-mode tangents propagated exactly through the shipped Verlet+dissipation update.
Derivation: endpoint a*(θ) satisfies V′(a*;θ)=0 with V″(a*)>0 ⇒ implicit-function value da*/dθ = −1/V″(a*).
Unrolled differentiation of a contracting fixed-point iteration converges to the IFT value; ∂a*/∂q₀ contracts to 0 (Prop 5).
- **P-I1a:** |unrolled da_N/dθ − (−1/V″(a_N))| / |IFT| ≤ 1e-8 at N=4000.
- **P-I1b:** |da_N/dq₀| ≤ 1e-10 at N=4000 (same run).
- **P-I1c:** endpoint vs θ for a near-separatrix query (q₀=0.05) jumps by ≈ distance between minima (≈2) at the
  re-assignment θ; a*(θ) within a fixed basin varies smoothly with slope −1/V″ (checked against finite differences ≤1%).
Verdict pre-commitment: the θ-path through the relaxation is **healthy (O(1), IFT-valued)**; only the
address/query-path is dead. If P-I1a fails (unrolled ≠ IFT), the "amortize by regression, stop-grad the address"
prescription must be revised to mandatory implicit differentiation.

## Item 3 — does relaxation land where the writer wrote? (script `item3_coincidence.py`)
Write rule: a*_k = damped-relaxation endpoint of content c_k (= well center here). Read: relax query q̃ = c_k + ση (2-D Gaussian).
Coincidence is **exact at σ=0 by idempotency + determinism** (same V, same operator, converged). At σ>0 it is pure basin
geometry: coincidence(σ) = P(q̃ ∈ Basin_k) (rest-start relaxation = the labeling operator itself).
- **P-I3a:** coincidence ≥ 0.90 at σ = margin/3.
- **P-I3b:** coincidence ∈ [0.45, 0.75] at σ = margin.
- **P-I3c:** at σ = margin, ≥ 25% of failures are origin-trap captures (spurious-minimum failure mode), not wrong wells.

### Predictions for the engineer's `learned-landscape-write-read` (registered BEFORE reading their file):
- **E1:** frozen V_θ + converged relaxation + query = exact write content ⇒ coincidence ≥ 99% (failures only from
  non-convergence/step-size artifacts). **A systematic mismatch at σ→0 with frozen, converged V falsifies Item 3's
  "true by construction" claim and with it the write rule as stated.**
- **E2:** with query noise, coincidence degrades smoothly following basin geometry (no cliff before σ ~ margin/3).
- **E3:** in a *learned* MLP landscape, the dominant failure mode (> half of failures at moderate σ) is capture by
  **spurious critical points** (minima/plateaus not corresponding to any written item), not confusion between items.
- **E4:** if V_θ is trained between write and read, addresses drift by ≈ ‖H⁻¹ δ(∇V)(a*)‖ (IFT displacement law);
  coincidence fails when accumulated drift + query noise exceeds the basin margin.

## Item 4 — capture by annealing (boost + re-relax) (script `item4_retry.py`)
Queries q̃ = c_target + ση, σ = 1.0, n = 400, targets cycled over the 4 wells. Initial read: rest-start damped relaxation.
Retry ladder: ≤3 retries, kick KE ladder = [0.3h, 1.2h, 1.6h] (h = measured well barrier), re-relax after each kick;
score s = −‖q̃ − q*_settled‖; **gated acceptance** (keep new settlement iff score improves). Arms: (i) undirected
(random unit kick), (ii) query-directed (kick toward q̃), (iii) oracle (kick toward c_target; upper bound only).
Classes: initial-correct / wrong-well / origin-trap; separately **info-lost** (nearest center to q̃ ≠ target — unrecoverable
by ANY retry since the score itself prefers the wrong item).
- **P-I4a (escape pricing):** radially-inward kick escape threshold KE_c/h ∈ [1.0, 1.4] at γ=0.02 (friction tax on the climb);
  ∈ [1.0, 1.1] in a γ=0 escape arm. This is Prop 2's `M* = p₀²/2h` crossed deliberately.
- **P-I4b:** undirected ladder success on wrong-well, non-info-lost starts ∈ [0.30, 0.65].
- **P-I4c:** query-directed ladder success on wrong-well, non-info-lost starts ≥ 0.75; oracle ≥ 0.85.
- **P-I4d:** among initially-correct starts whose target center is nearest to q̃: **0 degradations** (gating guarantee).
- **P-I4e:** origin-trap recovery ≥ 0.90 with the first (0.3h) rung, query-directed (cheap escape from the spurious min).
- **P-I4f (isolation restored):** 100% of accepted final settlements have total energy < V_saddle (sub-barrier again),
  and the post-settlement tail window (last 25% of the final relaxation) lies entirely in the final basin, while
  supra-barrier transit segments visit foreign basins (>0 fraction) — i.e. isolation is violated only transiently and
  the read window can exclude it.
- **P-I4g:** info-lost class ladder success ≈ 0 (≤ 0.15) in all arms — retry recovers dynamics-misrouting, not information loss.

## Item 5 — trajectory vs endpoint information (script `item5_fiber.py`)
Two "items" at the SAME location, payload written in local curvature: V_i = (k_i/2)(q−a)², k₁=1, k₂=4, M=1.
- **P-I5a:** damped endpoints identical to ≤1e-6 (endpoint map is many-to-one: the fiber carries the payload).
- **P-I5b:** γ=0 rollout frequency ratio = √(k₂/k₁) = 2.00 within 2% (zero-crossing count).
- **P-I5c:** a linear read on trajectory windows separates the items 100/100 noisy trials; any endpoint-only read is at chance (50±10%).

## Item 6 — HiPPO-LegS per-item resolution fading (script `item6_legs.py`)
LegS ODE ċ = −A c/t + B f/t, N=64, Euler dt=0.005 from t₀=1; input = unit Gaussian bump (τ=2, at x=5) + recent unit bump at T−10.
Reconstruction via normalized Legendre on [0,T]; recovered amplitude = max of f̂ near x=5.
Derivation: Legendre local resolution ℓ(x,T) = (π/N)√(x(T−x)) (edge clustering); amplitude ~ min(1, c·τ/ℓ).
- **P-I6a (the law, primary hypothesis):** log-log slope of recovered early-bump amplitude vs T over T ∈ {300, 600, 1200, 2400}
  = **−0.5 ± 0.2** (edge-clustering derivation). **Competing hypothesis (registered so it can lose):** −1 (naive uniform-resolution
  estimate ℓ = 2T/N). Pre-commitment: primary is −0.5.
- **P-I6b:** crossover T* (amplitude drops below 0.7) at T* = (τN/π)²/x = 332; band [170, 660].
- **P-I6c:** recent bump (at T−10) recovered ≥ 0.8 at every T (recency always resolved).
- **P-I6d (exploratory, weak):** adding a strong late distractor (unit sinusoid, last 20% of window, period ≈ mid-domain
  resolution) raises early-window reconstruction RMS error by > 0.02 — finite-N budget-sharing crosstalk, despite exact
  state-level linearity/superposition.
Interpretive pre-commitment: whatever the exponent, LegS retention of a *fixed* stored item decays polynomially in total
elapsed time with NO per-item exemption mechanism (A, B fixed; measure fixed), while a CLU μ²=0 latch is age-independent —
the addressability claim will be argued as *per-item write-time retention control + exact sub-barrier isolation*, not as
"HiPPO has no notion of items".

## Item 2 (no new harness — theorem + already-measured numbers)
Prop (robustness–gradient exclusion) will be proved: if the read R is constant on B(a,ρ) (ρ-robust retrieval), then
∇_a R ≡ 0 on the interior — and ANY training signal containing the factor ∇_a(L∘R) (including end-to-end selector
training through the read) is identically zero there; epoch-averaging of zeros is zero. Escape routes that carry no
such factor: write-time regression labels, derivative-free (score-gated retry), boundary events. Numerical verification:
the engineer's γ-scan (∇ falls 7 orders while settling error saturates) + Item-1's P-I1b are the measured instances;
no new number pre-registered.
