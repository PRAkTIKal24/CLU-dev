# PREREG — xy-1d-control (results-analyst)

**Written BEFORE running the CLU-Langevin chain.** Commit `df5e44d`. Protocol §5 pre-registration.

## What is being predicted
The correlation function `C(r) = ⟨cos(θ₀−θ_r)⟩` of a ring of `N=16` designed SO(2) CLU units
(`MexicanHatPotential(lam=1,f=1)`, `channel_spring_coupling(κ=0.05)`, `kinetic_mode="newtonian_learned"`,
`langevin_noise="fdt"`, `use_governor=False`) sampled at temperature `T` must decay as

```
C(r) = u^r ,   u = I₁(K)/I₀(K) ,   K = J/T ,   ξ = −1/ln(u)
```

with **`J = 2κr*²`, parameter-free.** `r*` is the vacuum-ring radius of one unit = argmin U(r).
For the pure quartic Mexican hat `U = λ(r²−f²)²` with `λ=f=1`, the minimum is at **`r*=f=1`**
(no confinement term; coupling vanishes on the aligned torus), giving **`J = 2·0.05·1² = 0.10`**.

## Committed numbers (parameter-free, nothing fitted)

| T | T/J | K=J/T | u=I₁/I₀ | ξ (predicted) | 2J/T (low-T asymp) | C(r=8)=u⁸ |
|---|---|---|---|---|---|---|
| 0.050 | 0.50 | 2.0000 | 0.697775 | **2.7789** | 4.0000 | 0.0562 |
| 0.075 | 0.75 | 1.3333 | 0.552152 | **1.6837** | 2.6667 | 0.0086 |
| 0.100 | 1.00 | 1.0000 | 0.446390 | **1.2398** | 2.0000 | 0.0016 |
| 0.150 | 1.50 | 0.6667 | 0.316089 | **0.8683** | 1.3333 | 0.0001 |
| 0.200 | 2.00 | 0.5000 | 0.242500 | **0.7058** | 0.0000 | 0.0000 |

Temperatures chosen so ξ ∈ [0.7, 2.8] is resolvable in an N=16 ring (max separation r=8).

## Expected deviation (declared in advance, not a free knob)
The *pure-XY* `u` above ignores Born–Oppenheimer + thermal radial dressing. The theorist measured
the exact CLU-Gibbs marginal vs pure-XY at κ/k_r=0.00625: `⟨cosΔθ⟩` differs by **0.14% / 1.2% / 2.8%**
at T=0.02/0.05/0.10 (grows ~linearly in T). Extrapolating to my grid, I expect the *measured nearest-neighbour*
`u₁ = C(1)` to sit **≤ ~6% below** the pure-XY `u` at T=0.20, less at lower T, because dressing weakens the
effective exchange (`J₁ = 2κr*²(1−4κ/k_r)`; at κ/k_r=0.00625, `1−4κ/k_r = 0.975`, a −2.5% floor on J even at T→0).

## Acceptance / kill decision (pre-committed)
- **MATCH (⇒ dictionary established, lean GO):** fitted ξ from the CLU chain agrees with the predicted ξ to
  within the declared dressing band (≈ **within 5–8%** across the grid), AND the shape is a clean single
  exponential (log-linear C(r) vs r), AND the equilibrium-start chain shows no drift.
- **KILL (⇒ reduction failed on the real path, NO-GO on CSF3):** fitted ξ is off by a factor that cannot be
  explained by the ≤2.5%→~6% dressing (e.g. >20%), OR C(r) is not exponential, OR the chain is still drifting.

## Secondary pre-registered nulls
- **ρ_s → 0 with N** at every T>0 (no 1D stiffness): helicity modulus decreases toward 0 as N grows.
- **1D winding degrades with size:** per-site phase-slip rate ≈ constant ⇒ `τ_winding ∝ 1/N`
  (theorist's reduced-XY value at T=0.5J: per-site rate `0.00656 ± 11%`). Predict the CLU/real-path
  per-site rate is N-independent within error.
- **Broken-symmetry control** (default random-W `spring_coupling`): ξ does NOT match the parameter-free value
  (p=2 anisotropy `h₂/|J|≈1` is a relevant perturbation); expect faster/anisotropic decay or wrong ξ.
- **γ-independence at N=16:** ⟨cosΔθ⟩ identical (within MC error) across γ ∈ {0.02, 0.1, 0.4}.
