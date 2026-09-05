# PREREG — relaxation-fiber-capacity (physics-theorist, w21)

Written **before** any harness was run. Every number below is a commitment made from
pen-and-paper derivation only. Scripts live in
`.claude/scratch/relaxation-fiber-capacity/`.

## Derivations the predictions come from (stated first, so the predictions are checkable)

**D-A (reduction to the effective force field).** Newtonian CLU with diagonal per-launch
mass `M`, landscape `V`, dissipative Verlet with per-step damping `(1-γ)` (⇒ continuous
friction rate `Γ = γ/ε` acting on `p`). Writing the *velocity* `u = M⁻¹p`:

```
q̇ = u ,        u̇ = −M⁻¹∇V(q) − Γ u ,      u(0) = M⁻¹p₀
```

⇒ the whole read (any functional of the sampled trajectory `q(·)`) depends on
`(M, V, p₀)` **only** through the pair `( G(q) := M⁻¹∇V(q) , u₀ := M⁻¹p₀ )` and `Γ`.

**D-B (the gauge group).** Hence `(M,V,p₀) → (ΛM, Ṽ, Λp₀)` with `∇Ṽ = Λ∇V` is exactly
unobservable, for any diagonal `Λ>0` for which `Λ∇V` is still a gradient field. That
condition is `(Λ_i − Λ_j)∂_i∂_j V = 0`, i.e. `Λ` must be constant on each **separable
block** of `V` near the address. ⇒ `dim(gauge) = k` = number of separable blocks;
mass has `d` parameters ⇒ **mass contributes exactly `d − k` observable dimensions.**
For `d = 1`, or for any fully separable `V`, `k = d` ⇒ **mass contributes ZERO** —
harmonic *or* anharmonic. (This contradicts the Hub's proposed resolution, so it is
pre-registered as a falsifiable claim.)

**D-C (relativistic breaking).** With `T = c√(pᵀM⁻¹p + (m₀c)²)`, `q̇ = cM⁻¹p/√(ξ+(m₀c)²)`,
`ξ = pᵀM⁻¹p`. Under `Λ = λI`, `ξ → λξ` while `M⁻¹p` is invariant ⇒ the gauge survives
only to leading (Newtonian) order and is broken at relative order
`ξ/(m₀c)² = (v/c)²/(1−(v/c)²) ≈ (v/c)²`.

**D-D (read cost).** Per-sample read noise `σ` (i.i.d., the linear-decoder resolution
model), `N` samples at spacing `ε`. Cramér–Rao for a sinusoid: `σ_ω ∝ σ/(A N^{3/2} ε)`;
for a non-phase-accumulating (amplitude-like) parameter, `σ_a ∝ σ/(A√N)`.
⇒ bits per coefficient `= log₂(range/σ_est) = c₀ + (3/2)log₂N` (frequency channel) or
`c₀ + (1/2)log₂N` (amplitude channel) ⇒ **read length is exponential in bits *per
coefficient*, polynomial-free in the *number* of coefficients.**

---

## Pre-registered predictions

| # | prediction | falsifier |
|---|---|---|
| **P1** | **Exact gauge, Newtonian.** d=1, anharmonic `V = ½kq² + βq³ + δq⁴`, γ=0.02, ε=0.05, N=2000, p₀≠0. Scaling `(M, V, p₀) → (λM, λV, λp₀)`, λ=3.7, leaves `q_n` unchanged to **max abs diff ≤ 1e-12** (float64). | any diff > 1e-10 ⇒ D-A/D-B wrong |
| **P2a** | **Anharmonicity does NOT break it.** Same as P1 with p₀=0 and β,δ ≠ 0 scaled with V: diff ≤ **1e-12**. ⇒ the Hub's "waveform separates mass from landscape" is **refuted** for scalar mass. | diff > 1e-10 |
| **P2b** | **Partial scaling IS observable.** Scaling `(M, k)` by λ=3.7 but leaving β,δ fixed: relative trajectory difference ≥ **0.10** of the orbit amplitude. (Shows the degeneracy is with the *whole* V, not with "curvature" alone.) | diff < 0.01 |
| **P3a** | **d=2 coupled harmonic:** single noise-free rest-launch rollout, least-squares fit of `q̈ = −A(q−a*)`; `A` recovered to rel err ≤ **1e-6**, and `M₁/M₂ = A₂₁/A₁₂` to rel err ≤ **1e-6**. | rel err > 1e-3 |
| **P3b** | **d=2 separable harmonic (`K₁₂=0`):** `\|A₁₂\|, \|A₂₁\| ≤ 1e-10` ⇒ mass ratio **unidentifiable** (the identifying equation is 0/0). | any off-diagonal > 1e-6 |
| **P4** | **Relativistic gauge breaking ∝ (v/c)².** Trajectory divergence between a gauge-related pair (λ=2) vs `v_max/c ∈ [0.02, 0.25]`: log–log slope = **2.0 ± 0.25**. | slope outside [1.75, 2.25] |
| **P5a** | **Frequency read cost:** `σ_ω` vs `N` log–log slope = **−1.5 ± 0.15**. | outside |
| **P5b** | **Amplitude-channel read cost:** slope = **−0.5 ± 0.10**. | outside |
| **P6** | **Resolvable jet order.** 1-D well, force `F = Σ_{r≥1} a_r q^r` with natural scaling `a_r ~ a_1/L^{r−1}`, `L=1`, orbit amplitude `A=0.4`, per-sample read noise σ/A = 1e-3, N=2000: number of coefficients recovered to <50% rel err is `r_max ∈ [5, 9]` (point estimate 7, from `(A/L)^{r−1} ≳ σ/(A√N)` ⇒ `r ≈ 1 + ln(A√N/σ)/ln(L/A)`). | outside [5,9] |
| **P7** | **Fiber bits headline (the number `clu-autoencoder` will want).** For d=2, one well, read N=1200 at ε=0.05, σ_read=1e-3 (relative to orbit amplitude), 100:1 admissible dynamic range per coefficient: **total readable fiber payload = 20–60 bits per well**, dominated by the 3 Hessian coefficients. | measured outside [20,60] |

## Composition rule — committed *before* measuring

I commit **now** to the *form* of the answer, so the verdict is not retro-fitted:

> **The fiber is ONE storing channel (the local jet of `V_θ`), not six.** `M`, `p₀` are
> address-side (reader-supplied ⇒ they *select*, they do not *store*); `γ` is a spatial
> field (shared at a location ⇒ 0 per-item bits); temperature is a noise channel.
> Therefore the honest composition is **`B_total = K_spatial × B_fiber`** (slots × payload
> per slot) — *bits* multiply, **item counts do not**: two items cannot occupy one address
> with two different jets, because a location has exactly one landscape.

If the numerics contradict any part of this I will report the contradiction, not amend the
statement.

---

## Addendum — committed **before** the Item-2/3/4 harness was run (same session, after Item 1)

**P8 (launch multiplicity).** The fiber read is a Gaussian channel: with i.i.d. read
noise σ per sampled coordinate and a Gaussian prior of width ρ_j on each jet coefficient,
the readable payload is `B = ½ Σ_i log₂(1 + s_i²)`, `s_i²` = eigenvalues of
`R^{1/2}(JᵀJ/σ²)R^{1/2}`, `J = ∂q/∂a`. Because a single rest launch explores a
**1-dimensional curve** in q-space, I predict:
- one launch in d=2 with a jet through order 4 (12 coefficients) resolves
  **≤ 8** directions at `s_i > 1`; the rest are near-null;
- bits grow **≈ linearly in the number of independent launches** until they saturate at
  the full jet dimension, and only **logarithmically in N** (P5).
Falsifier: one launch already resolves ≥ 11 of 12 directions, or bits are flat in launch
count.

**P9 (nuisance cost).** Marginalising the launch condition `(q₀,p₀)` (2d nuisance
parameters, the "query-noise-limited" regime) costs **< 30 %** of the bits, because the
nuisance directions are not aligned with the jet directions. Falsifier: > 50 % loss.

**P10 (relativistic mass channel — how big).** The (v/c)²-scale gauge breaking measured in
P4 accumulates **secularly** (it is a frequency shift), so the trajectory divergence at
fixed v/c should grow **∝ N** (log–log slope in N of **1.0 ± 0.2**) until it saturates at
O(1). ⇒ the relativistic mass channel is readable but its SNR is bought with read length.
Falsifier: slope < 0.5 (no accumulation) or > 1.5.
