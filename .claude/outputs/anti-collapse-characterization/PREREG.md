# PREREG — anti-collapse-characterization numerical demo (written BEFORE running the harness)

**Date:** 2026-07-20. **Agent:** physics-theorist. **Harness:** `.claude/scratch/anti-collapse-characterization/demo.py` (pure numpy, closed-form harmonic dynamics; finite-difference gradients; no JAX, no repo code).

## The toy

Two decoupled 1-D "CLU channels", each a harmonic Hamiltonian system with learnable stiffness `k_i = exp(θ_k,i)` and learnable inertial mass `m_i = exp(θ_m,i)`; frequency `ω_i = √(k_i/m_i)`. Ground truth: `k* = (1,1)`, `M* = (4.0, 0.25)` ⇒ `ω* = (0.5, 2.0)` (the seed-sweeps 16× two-timescale setting). Init: `k = (1,1)`, `m = (0.7, 0.7) · exp(N(0,1e-3))` (uniform, the mass-spectrum-peek/CM-5 collapsed init). Trajectories from `q0=1, p0=0`, closed form `q(t)=cos(ω t)`, `p(t) = −m ω sin(ω t)`, grid `t = dt·{1..128}`, `dt=0.05`.

**Loss:** task = MSE on `q(t)` (wake sees positions only) — plus, arm-dependent, `p(t)` and/or the structural regularizer
`R_spread = λ_s (Var(log m) − s*²)²`, `Var` = population variance over the 2 channels, `s* = ½·log16 = 1.386294`, `λ_s = 1.0`.

**Optimizer:** plain GD, central finite differences (δ=1e-5), `η_k = 0.05` (fast lever, emulating the MLP potential's gradient richness), `η_m = 0.005` (slow lever, the log_mass reality), 4000 steps, 5 seeds (arm C: 10 seeds), seeds = 0..4 (C: 0..9) controlling only the init jitter.

## The physics being tested (derivations in the main report)

For a q-only loss the trajectory depends on `(k_i, m_i)` **only through `ω_i`** (with `p0=0` exactly), so `u_i = log k_i − log m_i` is the sole task-visible coordinate and the co-scaling direction `(δlog k_i = δlog m_i)` is **exactly task-null** (a gauge direction). Chain rule gives `∂L/∂log k_i = −∂L/∂log m_i` pointwise, hence under GD with lrs `(η_k, η_m)`:
`Δlog m_i = −[(η_m/η_k)/(1+η_m/η_k)]·Δu_i = −Δu_i/11` **exactly, independent of the loss landscape**, provided the run converges to `ω = ω*`.
`Δu_1 = log(0.25) − log(1/0.7) = −1.742969`, `Δu_2 = log 4 − log(1/0.7) = +1.029619`.

## Pre-registered predictions (commit-then-measure)

- **P1 (collapse, arm A: plain, q-only).** Final `Std(log m) = |Δlog m_1 − Δlog m_2|/2 = (1.742969+1.029619)/22 = 0.12603` — predicted **0.126 ± 0.02**, i.e. ≈ **9.1% of s\*** (the lever collapses to the lr-ratio floor, mirroring N7's σ_struct 2–23%); final task MSE < 1e-3 (converged); learned `k` absorbs the timescales.
- **P2 (cure, arm B: q-only + R_spread from epoch 0).** Final `|Std(log m) − s*|/s* ≤ 0.10`; final task MSE < 1e-3 and within 2× of arm A (task-null claim ⇒ no fit price at optimum); **ordering correct (m_1 > m_2) in 5/5 seeds** — mechanism: the task's early transient (channel 1 wants ω down ⇒ m_1 up; channel 2 the reverse) seeds sign(Δlog m) at rate ≈ η_m·O(0.1–1)/step, beating the Var-form regularizer's exponential amplification of the 1e-3 jitter (growth rate ≈ η_m λ_s s*²/2 ≈ 4.8e-3/step).
- **P3 (exact nullity check).** At arm B's optimum, co-shifting channel 1 by `δlog k_1 = δlog m_1 = 0.2` changes task MSE by < 1e-10 while changing `Std(log m)` by 0.1 — the regularizer moves along a task-null direction.
- **P4 (honesty arm C: warm-start at the collapsed optimum — `k_i = 0.7·ω*_i²`, `m = 0.7·exp(N(0,1e-3))` — then task+R_spread).** `Std` recovers to `s* ± 10%` but **correct ordering in only ~5/10 seeds (registered interval: 2–8 of 10)** — the spread term restores *diversity*, not *assignment*; assignment sign = sign of the random jitter once the task transient is gone. (This is my own candidate's failure mode, registered in advance; consistent with CM-5's schedule sensitivity and N8's inversion.)
- **P5 (demarcation, arm D: plain, loss on q AND p, no regularizer).** With p observed, `m_i` is task-identified (p-amplitude = m·ω), the null direction is lifted, and **no collapse occurs without any regularizer**: final `|Std(log m) − s*|/s* ≤ 0.2`. This is the generalization of the erosion demarcation law (CM-6: "erosion iff the flat direction is unconstrained by wake") to the mass lever.

**Deviation protocol:** if any arm fails to *converge in task MSE* (frequency-estimation side-lobe trapping is a known risk of GD on periodogram-like losses), I may adjust `η`, steps, or the time horizon **for all arms uniformly**, and will record the change here as a dated amendment before re-running. The registered quantities themselves (P1–P5) will not be adjusted.

---

**AMENDMENT 1 (2026-07-20, before the final registered run).** The first run confirmed the pre-flagged side-lobe risk: arms A/B/D(ch-2) trapped at wrong frequencies (ω=(1.73, 0.85), task MSE 0.84 — *not converged*, so the convergence precondition of P1/P2/P5 was unmet; results saved but void for the registered claims). Per the protocol, the task loss is changed **uniformly for all arms** to the multi-scale form `L = mean over windows T∈{16,32,64,128} of MSE(first T steps)` (short windows funnel GD into the main lobe; identical anti-symmetry `∂L/∂log k_i = −∂L/∂log m_i` still holds for any q-only loss, so P1's landscape-independent −1/11 partition derivation is untouched). A single smoke run of **arm A only** under the amended loss (before any other arm was run) converged (MSE 4.4e-23) and gave `Std(log m) = 0.12603` — consistent with P1 as registered. All P1–P5 statements stand unchanged; no other knob was touched.
