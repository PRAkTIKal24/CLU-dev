# PREREG — v5-derivation-appendix numerical self-check
Written BEFORE running any check script (task §4). Every closed form below is verified
**against the composed elementary map** (damped Verlet ± Langevin noise, coded independently in
numpy from the step definition in `chlu/core/integrators.py`), not against the derived formula
itself. Paper = `~/Desktop/V5_PALM_Submission/paper.tex`, md5 `a5182217490642da6e62579eca576e7b`
at boot (2026-08-26 02:27:22).

Derivation basis (fixed now): per-mode 2×2 damped-Verlet map in mass-whitened coordinates,
h = εμ: `tr J = (2−γ)(1−h²/2)`, `det J = 1−γ`. All predictions below are computed by hand /
closed form from that map BEFORE the scripts run.

| id | check (against composed map / MC unless marked ALG) | pre-registered value | paper prints |
|----|----|----|----|
| R1 | det of composed one-step Jacobian, d=4, anharmonic V, γ=0.13 (finite-diff/exact) | (1−0.13)⁴ = 0.57289761; rel. err < 1e-9 | det J = (1−γ)^d |
| R1b | same with position gate φ(q′), γ=0.13 | ×(1−0.13·φ(q′))⁴, rel err < 1e-9 | (1−γφ(q′))^d |
| R2 | eigs of numerically composed quadratic-mode map vs λ± = [tr±√(tr²−4det)]/2 | max abs diff < 1e-12 | ("exactly solvable" claim) |
| R3 | underdamped identity at γ=0.002: |λ| from composed map | √0.998 = 0.9989994995; n₁/₂ = ln2/(−½ln0.998) = 692.46; asymptote 2ln2/0.002 = 693.147 | 0.998999499; 692.5; 693.1 |
| R4 | mass-independent floor at γ=0.05: n₁/₂ = ln2/(−½ln0.95) | 27.0268 | "exactly 27.03" |
| R5 | γ_crit = 2εμ from printed μ²: designed [√0.670,√1.348]·0.1; emergent s42/s43 √(5.449e-2)/√(2.029e-2)·0.1 | [0.08185, 0.11611]; 0.023343; 0.014245 | 0.082–0.116; 0.02334; 0.01424 |
| R6 | exact branch-merge γ* by bisection on composed-map discriminant vs h*(γ)=(1−√(1−γ))√(2/(2−γ)) | agree < 1e-10; γ*/(2εμ) = 1−εμ+O(ε²μ²) (→1 as εμ→0), i.e. 2εμ is the leading-order optimum | γ_crit = 2εμ |
| R7 | slope-below of ln n₁/₂ vs ln γ, exact-map eigenvalues, grid geomspace(0.002,0.5,48), window γ<γ_crit/2.5, h=γ_crit/2 per emergent seed | −1.002 ± 0.001 (analytic −(1+γ/2) over window) | I-J: −1.0023/−1.0016/−1.0022 |
| R8 | slope-above, window γ>2.5γ_crit, same grid: emergent h from R5; designed h=0.05·μ_rad, μ²_rad∈{0.670,0.771,1.190,1.093,1.348} | emergent +1.10…+1.13; designed +1.20…+1.30 (analytic local slope 1+γ/(2−γ), i.e. THE branch asymmetry) | emergent +1.1262/+1.1031/+1.1254; designed +1.23…+1.27 |
| R8b | slope-below designed, same construction | −1.004 … −1.010 | −1.006 |
| R9 | MC stationary Var(pᵢ)/(MᵢT) on ring (M=2, r*=1.3), σ*²=MTγ(2−γ), γ=0.05 | 1.000 ± 0.02 (MC) | σ*ᵢ=√(MᵢTγ(2−γ)) claim |
| R10 | MC refrigerator: Var(p)/(MT) with uniform γ_φ ∈ {0.1,0.2,0.3,0.5}, γ=0.05, absorb-only noise | 0.36249 / 0.23082 / 0.17480 / 0.12591 (±~1% MC) | same four values printed |
| R11 | MC coset diffusion D̂_θ vs εT(2−γ)/(2F²γ), F²=Mr*² (M=2, r*=1.3) | ratio 1.00 ± 0.05 | D_θ = εT(2−γ)/(2F²γ) |
| R12 | MC vault D̂(γ_φ=0)/D̂(γ_φ=0.5) at γ=0.05; coupled-bath variant (noise rescaled to γ_eff) | 110.25 ± ~10%; coupled 13.881 ± ~10%; ALG: vault_absorb/vault_coupled ≡ T/T_local = 7.9423 | 110.25; 13.88; 7.942; T_local=1.26e-4 |
| R13 | T=0 latch: ring rollout 2e5 steps, write δ=0.5, γ∈{0.002,0.5}: drift; γ=0 control: growth law | drift < 1e-10 rad; γ=0 drift LINEAR in n (ballistic, Jordan-block shear) | drift ≤4.9e-12 (checkpoints); 142.7 rad @ γ=0 (checkpoint-specific, NOT re-predicted) |
| R14 | decay∘delete = delete∘decay on a toy canonical store (priority placement, amplitude decay) | exact float equality (byte-equal) | "decay commutes with deletion" |

STOP clause: if any row disagrees with a printed number, report; do not adjust algebra or numbers.
Known in advance and NOT treated as disagreement: the paper's measured argmin 0.90×γ_crit and
measured slopes sit off the ideal-map values because they are checkpoint measurements; the paper
itself prints them as measurements (the claim is the leading-order law γ_crit = 2εμ and the ∓1
asymptotes). R6/R8 quantify exactly this.
