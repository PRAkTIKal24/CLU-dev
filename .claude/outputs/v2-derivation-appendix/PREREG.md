# PREREG — v2-derivation-appendix (physics-theorist), 2026-08-24

Written BEFORE running the numerical harness (`.claude/scratch/v2-derivation-appendix/check_derivations.py`).
All predictions derived symbolically from the damped velocity-Verlet 2×2 block
A = [[1−h²/2, ε], [−(1−γ)εμ²(1−h²/4), (1−γ)(1−h²/2)]] (mass-whitened mode, h=εμ),
with det A = 1−γ, tr A = (2−γ)(1−h²/2). Derivation chain: characteristic polynomial
λ² − (tr A)λ + det A = 0; discriminant Δ = (2−γ)²(1−h²/2)² − 4(1−γ).

## Predictions (committed before measurement)

| # | quantity | predicted value | how derived |
|---|---|---|---|
| P1 | underdamped envelope half-life floor at γ=0.05 (any μ in band) | n₁/₂ = 2ln2/(−ln 0.95) = **27.0268** steps (paper quotes 27.03) | complex pair ⇒ \|λ\|² = det A = 1−γ, μ-independent; (1−γ)^{n/2} = ½ |
| P2 | overdamped log n₁/₂ vs log μ² slope | **−1** asymptotically (h≪γ); direct-map fit over a finite band edge-touching grid expected ∈ [−1.00, −0.95] (paper: −0.985 on trained ckpts) | λ₊ = 1 − (2−γ)h²/(2γ) + O(h⁴) ⇒ n₁/₂ ∝ h⁻² |
| P3 | overdamped half-life closed form, spot check γ=0.05, ε=1, μ=h=0.005 | n₁/₂ = 2γln2/[(2−γ)h²] = 0.0693147/4.875e−5 = **1421.8** steps; iterated map should agree within the O(h²/γ²)≈4% expansion error (here (h/γ)²=0.01 ⇒ ≪1%) | first-order expansion of char. poly. about λ=1 |
| P4 | exceptional point (γ=0.05) | h* = √(2/(2−γ))(1−√(1−γ)) = **0.0256440**; γ/2 = 0.025 (leading order); repeated eigenvalue λ = √(1−γ) = **0.9746794**; eigenvector matrix condition number diverges as h→h* (defective) | Δ=0 root via 2−γ−2√(1−γ) = (1−√(1−γ))²; A ≠ λI since A₁₂=ε≠0 |
| P5 | phase onset above h* | φ = arg λ ∝ (h−h*)^{1/2}; log–log fit slope → **0.50** as grid→onset (paper measured 0.5165 on ckpts) | Δ ≈ −C(h−h*), C = 4(2−γ)√(1−γ)h* > 0 |
| P6 | stability limit | λ = −1 exactly at **h = 2** for every γ (checked γ ∈ {0.01,0.05,0.2,0.5}) | 1 + tr A + det A = 0 ⇔ 1−h²/2 = −1 |
| P7 | latch transport (μ=0) | q_∞ = q₀ + εp₀/(Mγ) exactly; slow eigenvalue exactly 1 | A|_{μ=0} upper-triangular, eigenvalues {1, 1−γ}; geometric sum |
| P8 | coset diffusion under FDT noise σ*² = F²Tγ(2−γ) | Var(θ_N)/(2Nε) → D_θ = εT(2−γ)/(2F²γ); at ε=0.05, T=0.1, γ=0.05, F=1: D_θ = **0.0975** (simulation within a few % at 4000 walkers × 20000 steps) | AR(1) momentum: stationary Var(p)=F²T, autocovariance ratio (1+a)/(1−a) = (2−γ)/γ with a=1−γ |

Failure criterion: any derived closed form disagreeing with the direct-map measurement beyond its stated expansion error, or with a paper constant (27.03; slope −1; h*≈γ/2; 0.5 onset), is reported as a finding about the paper per task §Numerical self-check — not papered over.
