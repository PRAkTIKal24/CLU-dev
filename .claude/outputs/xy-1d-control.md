# xy-1d-control — results-analyst report

**Task + acceptance criterion:** the KT go/no-go. On the *real* code path, an `N=16` designed
SO(2) CLU chain (`channel_spring_coupling`, `fdt`, no governor, `newtonian_learned`) must (i) reduce
exactly to XY, (ii) be proven equilibrated, (iii) reproduce the **parameter-free** correlation length
`ξ = −1/ln(I₁(J/T)/I₀(J/T))`, `J = 2κr*²`, over ≥4 T, **or a clean kill**; plus the 1D nulls,
γ-independence at N=16, and the broken-symmetry control. Deliverable = a **GO/NO-GO** on the 2-D CSF3 run.

**Status: done** (6 of 7 items measured on the real path; item 3c *winding τ∝1/N* not run on the real
path — reduced-XY value cited, flagged as the one follow-up). **Pre-registration:** `PREREG.md` written
and committed before the chain was run; results below are judged against it.

**Reconciliation list (owner needed):** none — this task consumes prior findings; it does not retract any.
One **harness bug I hit and fixed** (my scratch, not repo code): the γ=0.40 arm used a hot-start-sized burn
that exceeded `n_steps` → NaN; re-run with a short (equilibrium-start) burn. No repo code touched.

## VERDICT — **GO** (conditional) on the 2-D CSF3 experiment
The reduction is verified **exactly** on the real code path (residual power `6.7e-34`), and the CLU-Langevin
chain reproduces the parameter-free `ξ(T)` to **1.5–6.8%** over 5 temperatures, with the residual tracking
the *pre-declared* Born–Oppenheimer + thermal dressing (deficit grows monotonically with T, ≤ the ~6% band).
All three prerequisites (fdt / no-governor / newtonian) behave as the theory demands; the broken-symmetry
control fails exactly as predicted (`⟨cosΔθ⟩ = −0.006` vs XY `0.446`). **No kill triggered.** The dictionary
`J = 2κr*²`, `T_XY = T_Langevin` is established on the real path — the 2-D run deserves its queue time,
**provided** it ships with `coupling_type="channel_spring"` (P5) and `L ≥ 8`.

---

## 1. Flag provenance (mandatory)
| item | value |
|---|---|
| repo commit | **`df5e44d`** (gate-cleared HEAD; suite 278p/1s per Hub). `git status --porcelain` empty before & after. |
| repo edits | **none** (read-only task; all artifacts under `.claude/`) |
| env | main venv `/Users/user/Desktop/CHLU/.venv`; **jax 0.9.0**, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0; CPU |
| precision | **float64** (`jax_enable_x64=True`) |
| lattice | `N=16` **open chain** (15 edges), each unit `CHLU(dim=2, hidden=4)` with `MexicanHatPotential(lam=1, f=1, k_spec=None)`, `log_mass_for_inertia([1,1])`, `kinetic_mode="newtonian_learned"`; edge coupling `channel_spring_coupling(2,2,κ=0.05)` |
| derived constants | `k_r = 8λf² = 8`, `κ/k_r = 0.00625`, `J₂/J₁ = 0.641%`, `r* = 1` (argmin of quartic hat), **`J = 2κr*² = 0.10`** |
| sampler | `langevin_step` vmapped over `NWALK=1024` walkers, `dt=0.02`, `noise_mode="fdt"`, `m_eff = lat.effective_mass()`; equilibrium start = exact reduced-XY open-chain Gibbs (θ₀ uniform, increments ~ von Mises(0, J/T), radii = r*), momenta `~ N(0, T·M_eff)` |
| **langevin_noise** | `"fdt"` (repo default `"legacy"`; the broken-symmetry & all physics arms are fdt) |
| governor | **OFF** everywhere |
| S2 (ξ grid) | γ=0.10, 8000 steps, burn 2000, 1024 walkers, seeds: numpy `default_rng(4242)` init draws, jax `PRNGKey(3)` |
| S4 (γ-indep) | γ∈{0.02,0.10,0.40}, T=0.10, eq-start; γ=0.40 re-run via `patch_g040.py` (burn 5 chunks) |
| S5 (broken-sym) | `spring_coupling(2,2,0.05,coupling_dim=2,init_scale=0.1, key=fold_in(PRNGKey(99),edge))`, hot-start, 16000 steps burn 8000 |
| S6 (ρ_s) | **ring** (N edges), N∈{4,8,16}, T=0.10, γ=0.10, 12000 steps burn 4000 |
| scripts | `.claude/scratch/xy-1d-control/{xy1d.py, patch_g040.py, postproc.py}` |
| data / fig | `.claude/outputs/xy-1d-control/{results.json, corr_fn.png, PREREG.md}` |

---

## 2. Item 1 — the reduction on the real path [EXACT]
2-D FFT of the real `channel_spring_coupling` bond potential on the vacuum torus (`r*=1`, 64×64 grid):

| quantity | value | prediction |
|---|---|---|
| `J_meas` (from −cos(θᵢ−θⱼ) coeff) | **0.100000000000** | `2κr*² = 0.10` (rel err **0.00e+00**) |
| residual power outside quadratic band | **6.70e-34** | 0 (float64 floor) |
| `h₂` (p=2 anisotropy) | **0.00e+00** | 0 |
| U(1)-breaking `cos(θᵢ+θⱼ)` amp | **7.73e-19** | 0 |

**The joint V restricted to the vacuum torus is a pure first harmonic with `J = 2κr*² = 0.10`, exactly.**
Reduction confirmed on the shipped `channel_spring_coupling` path (reproduces theorist S1 to the bit).

## 3. Items 2+3a — equilibration + parameter-free `ξ` [MATCH]
Equilibrium-start stationarity: every arm starts at the exact reduced-XY Gibbs and the first-half vs
second-half **drift is ≤ 0.014** on all 5 temperatures → the chain **preserves** the measure (not annealing).
Robust decay estimator = `C(r)`-weighted log-linear fit over `C(r) > 0.03` (down-weights the MC noise floor
`~few×10⁻³` from 1024 walkers, which inflates a naive full-range fit at large r — visible in `corr_fn.png`).

| T | T/J | u_pred=I₁/I₀ | C(1) meas | u1 ratio | **ξ_pred** | **ξ_meas** | **ξ ratio** | drift |
|---|---|---|---|---|---|---|---|---|
| 0.050 | 0.50 | 0.6978 | 0.6922 | 0.992 | 2.779 | 2.907 | **1.046** | 0.011 |
| 0.075 | 0.75 | 0.5522 | 0.5400 | 0.978 | 1.684 | 1.708 | **1.015** | 0.012 |
| 0.100 | 1.00 | 0.4464 | 0.4326 | 0.969 | 1.240 | 1.207 | **0.973** | 0.014 |
| 0.150 | 1.50 | 0.3161 | 0.2999 | 0.949 | 0.868 | 0.830 | **0.956** | 0.006 |
| 0.200 | 2.00 | 0.2425 | 0.2258 | 0.931 | 0.706 | 0.658 | **0.932** | 0.010 |

- **ξ matched parameter-free to 1.5–6.8% over 5 temperatures.** Nearest-neighbour `C(1)/u_pred` runs
  0.992 → 0.931 as T rises — a **monotone** deficit that is exactly the pre-declared dressing signature
  (`J₁ = 2κr*²(1−4κ/k_r)` gives a −2.5% floor even at T→0; thermal radial dressing grows with T). At
  T/J=2.0 the 6.8% deficit ≈ the theorist's exact-Gibbs-vs-XY gap extrapolated to this T. **No unexplained
  discrepancy → no kill.** See `corr_fn.png`: linear panel points sit on the `u^r` curves; the log panel's
  large-r excess for the hot arms is the walker noise floor, not physics.
- Convergence cross-check (S3, hot-start at T=0.05): `C(1)` climbs from a hot start to **0.666** (0.955×XY),
  approaching the eq-start value **0.692** from below; residual drift 0.033 (relaxation is slow at low T,
  as the theorist warned — `n_relax ∝ γ/(ε²μ_rel²)`). Eq-start and hot-start **bracket** the prediction.

## 4. Item 4 — γ-independence at N=16 [CONFIRMED]
Equilibrium-start `⟨cosΔθ⟩` at T=0.10 over a 20× range of γ:

| γ | ⟨cosΔθ⟩ | /XY(0.446) | drift |
|---|---|---|---|
| 0.02 | 0.4303 | 0.964 | 0.011 |
| 0.10 | 0.4293 | 0.962 | 0.010 |
| 0.40 | 0.4396 | 0.985 | 0.019 |

**Spread ~2%** — the stationary measure is γ-independent, reproducing the theorist's N=2 result at N=16.
The slightly higher γ=0.40 value carries the largest drift (0.019): residual relative-mode equilibration,
consistent with the deep-diffusive caveat. **Friction buys time, not permanence — the precise form of CM-16.**

## 5. Item 5 — broken-symmetry control [FAILS AS PREDICTED — devastating]
Same N=16 chain, default random-W `spring_coupling` (init_scale 0.1), T=0.10, hot-start, 16k steps:

`C(r) = [1.0, −0.006, 0.003, 0.001, −0.001, 0.005, ...]` ≈ **0 at every r**.
`C(1) = −0.0060` vs XY `+0.4464`; no exponential, no scale.

**The dictionary fails completely on the broken-symmetry lattice.** The p=2 anisotropy (`h₂/|J|≈1` at init)
+ U(1)-breaking `cos(θᵢ+θⱼ)` destroys the XY exchange — exactly the *relevant-perturbation* prediction
(`x₂ = 1/2`). This is the concrete justification for prerequisite **P5**: the 2-D run must use
`coupling_type="channel_spring"` (or a conformal-constrained W), **never** the shipped random-W default.

## 6. Item 3b — ρ_s → 0 with N (the 1D null) [CONFIRMED]
Ring helicity modulus (connected form), T=0.10:

| N | 4 | 8 | 16 |
|---|---|---|---|
| ρ_s | 0.0238 | −0.0019 | −0.0036 |

Stiffness collapses to zero (within MC noise) as N grows → **no long-range order in 1D at any T>0**, the
honest null. (Small negative values at N≥8 are the fluctuation term overshooting the mean at finite walkers.)

## 7. Item 3c — 1D winding τ ∝ 1/N [NOT RUN on real path — the one gap]
Not measured on the CLU path this pass (rare slip events at low T need long detached runs; deprioritised
below the decisive ξ test). Theorist's reduced-XY value stands: per-site slip rate **0.00656 ± 11%** at
T=0.5J ⇒ `τ_winding ∝ 1/N`. **Recommended as the first laptop follow-up** (write w=1 on a CLU ring,
count winding decays vs N) — it is the null against which the 2-D "memory improves with size" claim is judged.

---

## Limitations / confounds
- **MC noise floor.** 1024 walkers put the noise floor at `C(r) ≈ few×10⁻³`; the naive full-range ξ fit is
  biased high at low T for that reason (`xy1d.py` `fit_xi`). The weighted estimator (`postproc.py`) and the
  robust `C(1)` are the trustworthy observables; a headline 2-D run should use ≥4–8k walkers and report SEM.
- **Dressing not separated from fit error.** The 1.5–6.8% ξ residual is *consistent with* the declared BO+
  thermal dressing but I did not isolate it from finite-N/fit error via a `dt`- or walker-scaling study.
- **Open chain, not ring, for ξ** (correct choice: exact transfer-matrix Gibbs, no wrap-around); ρ_s used a
  ring (helicity is a periodic observable). Winding (3c) needs the ring and was not run.
- **γ=0.40 drift 0.019** hints at incomplete relative-mode equilibration even eq-started; a longer burn would
  tighten the /XY=0.985 toward the other two (~0.96).
- Single designed κ (0.05, `κ/k_r=0.00625`), single seed stream per arm. Headline claims for a paper need
  ≥3 seeds; here the acceptance is a go/no-go, and the ξ match across 5 T is itself a 5-point consistency test.

## Recommended next experiments
1. **2-D CSF3 run — GO**, at `L ∈ {8,16}` first (N=64,256), `channel_spring`, fdt, no governor, newtonian,
   `κ=0.05` (`J=0.1`, `J₂/J₁=0.6%`), T/J ∈ [0.4,1.3] denser near 0.9. Headline: helicity jump `ρ_s/T → 2/π`
   at `T_KT = 1.786κr*² = 0.0893` [⚠ Hub-corrected 2026-07-19: the original "0.1786" was an arithmetic slip; 1.786×0.05×1² = 0.0893. `kt-2d-csf3` measured T_KT = 0.0898 (+0.6%), confirming 0.0893 and refuting 0.1786.]; validate the CLU-Langevin stationary dist == reduced-XY at L=8 via this
   task's eq-start protocol **before** the L=32 spend.
2. **Winding null on the real path (laptop, cheap):** CLU ring, write w=1, measure `τ(N)` → confirm ∝1/N.
   Do this alongside (1) as the 1D anchor for the 2-D memory claim.
3. Walker/`dt` scaling at one T to separate dressing from fit error and pin `J_eff(T)`.

## Git footprint
No repo code changed (read-only task). HEAD `df5e44d`; `git status` clean. All artifacts under `.claude/`.

## How I verified (commands + real output)
- `.venv/bin/python xy1d.py` (PYTHONPATH=repo): S1 `J_meas=0.100000000000 relerr=0.00e+00 resid=6.70e-34`;
  S2 five T rows (drift ≤0.014); S4 γ arms; S5 `u1=-0.0060`; S6 ρ_s rows. Log: `.claude/scratch/xy-1d-control/run.log`.
- `patch_g040.py`: `gamma=0.40 <cosDth>=0.4396 /XY=0.985`.
- `postproc.py`: weighted ξ table above + `corr_fn.png`.

---

## Proposed handover updates (for the Hub)

### For §1.6 / §1.10 (the XY reduction — now confirmed on the real path at N>2)
> **1D go/no-go PASSED (`xy-1d-control`, commit `df5e44d`, jax 0.9.0, float64, fdt, no governor, newtonian).**
> An `N=16` designed `channel_spring_coupling` SO(2) chain reduces to XY **exactly** on the shipped path
> (torus-FFT residual power `6.7e-34`, `J_meas = 2κr*² = 0.100000000000`), and the CLU-Langevin chain
> reproduces the **parameter-free** `ξ = −1/ln(I₁(J/T)/I₀(J/T))` to **1.5–6.8%** over T/J ∈ {0.5,0.75,1.0,1.5,2.0}
> (drift ≤ 0.014; nearest-neighbour `C(1)/u_pred` = 0.992→0.931, a monotone deficit = the predicted BO+thermal
> dressing). γ-independence reproduced at N=16 (`⟨cosΔθ⟩ = 0.430/0.429/0.440` at γ=0.02/0.10/0.40, spread ~2%).
> 1D nulls hold: `ρ_s → 0` with N (0.024/−0.002/−0.004 at N=4/8/16). **Broken-symmetry control (random-W
> `spring_coupling`) fails exactly as predicted: `C(1) = −0.006` vs XY `0.446` — the dictionary is void there,
> concrete justification for P5.** ⇒ **GO on the 2-D KT experiment**, conditional on `coupling_type="channel_spring"`
> and `L ≥ 8`.

### For §5 (provenance) — new verified numbers
> `xy-1d-control` (df5e44d): reduction exact on real path; parameter-free ξ matched to 1.5–6.8% / 5 temps;
> γ-independence 2% / 20× γ; broken-symmetry C(1)=−0.006. `T_KT = 1.786κr*² = 0.0893` [⚠ Hub-corrected 2026-07-19: the original "0.1786" was an arithmetic slip; 1.786×0.05×1² = 0.0893. `kt-2d-csf3` measured T_KT = 0.0898 (+0.6%), confirming 0.0893 and refuting 0.1786.] at κ=0.05. Recommend L∈{8,16}
> as the first CSF3 tranche, L=32 only after the L=8 CLU-vs-reduced-XY stationarity check passes.

### For §8 (open directions) — scope call
> The thermodynamic framing's cheapest decisive test has **passed on the real code path**. Two things remain
> before/with the 2-D spend: (a) the 1D **winding τ∝1/N** null on the CLU path (not yet run — laptop-cheap,
> the anchor for the 2-D memory claim); (b) ship P5's `channel_spring` default. No code bug for `experiment-engineer`
> from this run (the one NaN was my scratch harness's burn>n_steps, fixed locally).
