# PREREG — `v5-vcurve-validation` (ME-1 rollout-validated V-curve · ME-3 vault on emergent)

**Written and saved 2026-08-19 BEFORE any harness in `.claude/scratch/v5-vcurve-validation/` was executed.**
Author: results-analyst. Commissioned by Shorts Advisor Addendum 26 (ME-1, ME-3).
Basis for the predictions: the banked `v5-gate` artifacts (`e1c_vcurve.json`, `e0_geometry.json`,
`r3main_results.json`) + the closed-form damped-map algebra below. Nothing new was measured to write this.

**DIAL DECLARATION (echoed from the task):** none — instrument validation + generalization probe on
banked checkpoints. **Laundering control:** n/a for ME-1 (instrument cross-check: the two instruments
are each other's control); for ME-3 the vault's own scalar control (scalar `γ = γ_eff = 0.525`, no
field, same `γ_eff`) travels beside every field number, exactly as in `v5-gate` §2.5.
**Falsifies:** the numbered failure lines F1–F5 (ME-1) and G1–G5 (ME-3) below.
**Does NOT falsify:** a *level* (multiplicative) disagreement between instruments that is constant in γ
— the V5 claims live on the shape (argmin + two slopes), and a constant offset is a stated bias, not a
refutation. Nor does an emergent seed whose thermal spread exceeds the escape tolerance
(`σ_θ/Δ > 1`) at the chosen T falsify ME-3; that cell is out of the diffusive window by construction.

---

## 0. What is already banked (the numbers I am predicting against)

From `v5-gate` (`e1c_vcurve.json`, γ-grid `geomspace(0.002, 0.5, 48)`, ε=0.05, δ_write=0.05 rad, T=0):

| checkpoint | μ²_soft | μ²_adiab | F² | r* | γ_crit=2εμ_soft | Jac argmin | argmin/γ_crit | Jac slope below | Jac slope above |
|---|---|---|---|---|---|---|---|---|---|
| emergent150_s42 | 5.449e-2 | 7.041e-2 | 0.9803 | 1.1894 | 0.02334 | 0.02100 | 0.8994 | −1.0022 | +1.1236 |
| emergent150_s43 | 2.029e-2 | 2.768e-2 | 0.6813 | 0.9886 | 0.01424 | 0.01288 | 0.9046 | −1.0016 | +1.1010 |
| emergent150_s44 | 5.132e-2 | 8.933e-2 | 0.7752 | 1.0748 | 0.02265 | 0.02051 | 0.9055 | −1.0022 | +1.1226 |
| designed150_s44 | −2.9e-16 | −2.9e-16 | 0.6380 | 0.9797 | 0 | — (n₁/₂=∞ ∀γ) | — | — | — |

Banked spot-check that MF-3 quotes: `n₁/₂(T=0, γ=0.05)` **Jacobian** 233.6 / 653.3 / 249.0 vs
**rollout** 190 / 370 / 150 → ratios **1.23 / 1.77 / 1.66** (the "19–43% gap"). Banked rollout argmin =
**0.00200 for all three seeds — i.e. the grid edge, no interior minimum at all.**

## 1. The mechanism I am pre-committing to (ME-1)

Read off the banked rows before running anything:

**(i) The left branch of the Jacobian V-curve is an integrator identity, not a model measurement.**
For γ < γ_crit the coset eigenpair is complex with `|λ|² = 1−γ` exactly, so
`n₁/₂ = ln2 / (−½ln(1−γ)) ≈ 2ln2/γ`, **independent of μ² and of the checkpoint**. Banked rows confirm
this numerically: at γ=0.002, s42 and s43 print the *same* `λ_ret = 0.998999499` and the *same*
`n_half_jac = 692.5`, and `2ln2/0.002 = 693.1`. ⇒ the −1 slope carries **zero model information**;
all model information on the V-curve lives in (a) the argmin location γ_crit=2εμ and (b) the +1 branch.
This is the sharp form of the referee's A1 "linearization tautology" attack, and the rollout is the
only instrument that can answer it: **does the full nonlinear model actually shed amplitude at rate
γ/2 per step below γ_crit?**

**(ii) The banked 19–43% "gap" is a constant-in-γ AMPLITUDE offset, not a rate disagreement.**
The banked rollout instrument is `first n with |θ(n)| ≤ δ/2`. If the ring write of δ deposits only a
fraction `c ≤ 1` of the amplitude into the slow coset branch (the emergent coset direction has angular
overlap 0.697/0.816/0.886 with the softest Hessian mode, so `c<1` is forced), then
`θ(n) ≈ cδ·|λ|ⁿ` and the crossing happens at `n_roll = ln(2c)/Γ`, giving
`n_jac/n_roll = ln2/ln(2c)` — **a constant, γ-independent factor.** Banked check (never used to
build this prediction beyond reading it): s42 ratios at γ = 0.0537/0.0858/0.1373/0.2197/0.3515 =
1.24/1.32/1.34/1.35/1.36 (→ c≈0.83); s43 = 1.78/1.82/1.83/1.83/1.83 (→ c≈0.73). Constant, and ordered
like the overlaps. ⇒ **the gap is a level bias with a known cause; the rate is predicted to agree.**

**(iii) The banked rollout argmin (0.002 = grid edge) is the C-9 first-crossing artifact.**
Below γ_crit the trajectory *oscillates*; the first crossing of δ/2 occurs at the first quarter-ish
period, `n ≈ arccos(1/2c)/ω` with `ω ≈ εμ`, which is **independent of γ**. So the naive rollout's left
branch is flat, its minimum is pushed to the grid edge, and it cannot reproduce the V. Fix: measure
the **envelope decay rate**, not a first crossing.

### ME-1 instruments (defined now, in advance)
- **I-J** (banked): `n₁/₂ = ln2 / (−ln|λ_ret|)` from the one-step Jacobian at (q*, 0).
- **I-R1**: rollout, **first** crossing of `|θ| ≤ δ/2` (the banked instrument; expected to carry the artifact).
- **I-R2**: rollout, **last** crossing — `min{n : max_{m≥n}|θ(m)| ≤ δ/2}` (suffix-max envelope). Removes
  the quarter-period artifact; retains the amplitude offset; adds an additive ≤ ½-period bias.
- **I-R3 (PRIMARY)**: rollout, **envelope rate fit** — least squares of `log(suffix-max |θ|)` vs n over
  the late window where the envelope ∈ [0.3δ, 1e-6 rad]; `n₁/₂^roll = ln2/Γ_roll`. Immune to both the
  amplitude offset and the half-period bias. This is the apples-to-apples partner of I-J.
- **I-R3-early**: the same fit over the **early** window (envelope ∈ [0.9δ, 0.3δ]) at large write
  amplitude δ = 0.5 rad — the anharmonicity/finite-amplitude probe (answers SF-11/A1).

---

## 2. ME-1 pre-registered predictions

**P1 — rate agreement (the headline).** For γ ≥ 2γ_crit (unambiguously overdamped branch), on all 3
emergent seeds at δ=0.05: `Γ_jac/Γ_roll ∈ [0.92, 1.08]` at ≥ 90% of grid points, seed-mean in
[0.95, 1.05].
**F1 (falsifier):** seed-mean outside [0.80, 1.25], or systematic γ-dependence of the ratio with
|d log(ratio)/d log γ| > 0.10 ⇒ **the Jacobian instrument does not measure the true decay rate ⇒ the
V-curve shape is unreliable.**

**P2 — the offset is constant and explained.** For I-R1/I-R2 above 2γ_crit the ratio `n_jac/n_roll` has
coefficient of variation over the γ-grid **< 8%** per seed, per-seed mean ∈ [1.15, 2.10], and the
implied slow-branch amplitude fraction `c = ½exp(ln2/ratio)` lands in [0.60, 0.95] and is **ordered
across the three seeds like the banked angular overlaps (0.697 < 0.816 < 0.886)**.
**F2:** CV > 20% (⇒ the discrepancy is γ-dependent and cannot be quoted as one stated bias).

**P3 — the ARGMIN survives on the rollout instrument.** Using I-R3 (and I-R2):
`argmin_γ n₁/₂^roll / γ_crit ∈ [0.75, 1.05]` on 3/3 emergent seeds (banked Jacobian: 0.902 ± 0.003).
**F3:** any seed outside [0.60, 1.20], or no interior minimum on I-R3 ⇒ shape not reproduced.

**P4 — the two SLOPES survive on the rollout instrument.** On I-R3:
slope below the minimum ∈ **[−1.20, −0.85]** (banked −1.0020 ± 0.0003); slope above ∈ **[+0.90, +1.35]**
(banked +1.116 ± 0.011). Additionally I pre-register that **I-R1's below-slope is ≈ 0**
(|slope| < 0.35) — the artifact, quantified.
**F4:** a sign change on I-R3 on either branch, or |slope| deviating from the banked value by > 40%.

**P5 — designed control (5 seeds, `designed150_s{42,43,44,45,46}`).** The rollout instrument must not
manufacture decay at the μ→0 corner: at every γ on the grid and every δ ∈ {0.05, 0.2, 0.5},
`max_n |θ(n) − δ| ≤ 1e-6 rad` over ≥ 20 000 steps, i.e. `n₁/₂^roll = ∞`, matching I-J's `|λ|=1`.
**F5:** any designed seed/γ with a finite rollout half-life below 1e6 steps ⇒ the rollout instrument
(not the model) is the source of decay, and the whole ME-1 comparison is void.

**P6 — finite amplitude (the A1 answer).** At δ = 0.2 and δ = 0.5 rad the emergent argmin/γ_crit stays
inside [0.75, 1.05] and both slopes inside P4's bands; and the **early-window** rate at δ=0.5 agrees
with the Jacobian rate to within 35% (`Γ_jac/Γ_roll^early ∈ [0.65, 1.35]`). If a seed's δ=0.5 write
clears its washboard barrier the trajectory settles in a neighbouring minimum (`|θ_final| > 1 rad`) —
**this is recorded as a capacity fact, not a P6 failure**, and that cell is excluded from the fits.

### Verdict rule, stated in advance
- **"Jacobian instrument validated with a stated bias"** iff **P1 ∧ P3 ∧ P4 ∧ P5** hold. The stated
  bias to be quoted in the paper is then the measured per-seed threshold-instrument offset from P2.
- **"The V-curve shape itself is unreliable"** iff **F1 ∨ F3 ∨ F4** fires.
- **F5 firing voids ME-1** (instrument fault) — report as blocked, not as a physics result.

---

## 3. ME-3 pre-registered predictions (the vault on an emergent checkpoint)

Checkpoints `emergent150_s{42,43,44}`; `T ∈ {4e-3, 8e-3}` (both **> T\* ≈ 3e-3**, per `v5-gate` §3.5);
scalar γ=0.05; uniform `γ_φ ∈ {0, 0.1, 0.2, 0.3, 0.5}` for V/D; localized compact hole (radius 1.0,
width 0.25) for the FPT arm; scalar control γ = 0.525 with **no field** beside every field number.
Geometry check done in advance from banked `e1c_vcurve.json`: r* = 1.1894/0.9886/1.0748 ⇒ the
`|Δθ| ≤ 0.5` exit chord is 0.59/0.49/0.53 < 1.0 (inside the hole) and the outside arm sits at
2r* = 2.38/1.98/2.15 > 1.25 (outside the gate + width) on all three seeds — the `v5-gate` hole geometry
transfers unchanged.

**The physical question:** on designed, the register is an *exact coset* (a flat direction, free
diffusion, D̂ is the whole story). On emergent it is a **soft mode with a restoring force**
(μ²_adiab = 2.8e-2 … 8.9e-2) — a *bounded* OU coordinate. Does `D̂ ∝ γ_eff^{-2}` (the absorb-only
refrigerator law) still transfer?

**My prior, stated:** the absorb-only law is a property of the **integrator's momentum update**
(`p ← (1−γ_φ)(1−γ)p + σ(γ)ξ`), not of the potential, so Q1/Q2 should transfer *exactly*, while the
*consequence* for the register is **stronger** than on designed, because the refrigerator also shrinks
the stationary spread of a mode that has a restoring force (Q3) — something a flat coset cannot show.

**Q1 — the refrigerator transfers (stage V).** `Var(p_i)/(M_i T)` at γ_φ=0.5, mean over channel dims,
3 seeds, both T: **0.1259 ± 0.006** (absorb prediction 0.125908; coupled prediction 1.0).
**G1:** measured outside [0.110, 0.145] ⇒ does not transfer.

**Q2 — the D̂ vault transfers (stage D).** In a lag window where the mass correction to the massless
absorb prediction is < 5% (enforced by choosing the lag, and verified against an **exact
linear-response MSD** computed numerically from the checkpoint's own Jacobian + noise covariance),
`D̂(γ_φ=0)/D̂(γ_φ=0.5) ∈ [95, 120]×` (point prediction `(γ_eff/γ)² = 110.25×`); the whole γ_φ ladder
follows `(γ_eff/γ)²` with `MSD_obs/MSD_pred_absorb = 1.00 ± 0.06` over all 15 (γ_φ × seed) cells.
**G2:** vault outside **[80, 140]×** with the mass correction demonstrably < 5% in both arms
⇒ the vault is designed-only. (If the mass correction cannot be pushed < 5% in both arms, that is
reported as *instrument-limited*, not as a transfer failure.)

**Q3 — the NEW emergent-only statement: the hole shrinks the register's equilibrium spread.**
Because the hole cools the local bath by 7.9423×, a mode with a restoring force must equilibrate to a
narrower Boltzmann distribution: `σ_θ(in)/σ_θ(out) = √(T_local/T) = 1/√7.9423 = 0.35484`, independent
of μ², F², T and of the seed. Predicted measured: **0.355 ± 0.020** (3 seeds), and in absolute terms
`σ_θ² = T_local/(F²μ²_adiab)`.
**G3:** ratio outside [0.30, 0.42] ⇒ the refrigerator does not reach the configurational distribution.
*(This is the claim that has no designed analogue — a flat coset has no stationary spread.)*

**Q4 — the scalar control (laundering control, mandatory).** Scalar γ=0.525, no field:
`Var(p)/(M T) = 1.00 ± 0.05`, `σ_θ(scalar)/σ_θ(γ=0.05) = 1.00 ± 0.06` (a scalar γ does **not** cool),
and D̂ vault vs γ=0.05 = **13.9× predicted, 13.3 ± 1.0× expected measured**. Field/scalar ratio
**7.94 predicted, ∈ [6.5, 9.5] expected**.
**G4:** field/scalar < 4 or > 14 ⇒ the "a field is not just stronger friction" statement does not
transfer to emergent.

**Q5 — the FPT consequence (directional, run only if Q1–Q4 land).** Because the hole cools the local
bath 7.94×, it drops `σ_θ/Δ` by 2.82× — pushing the *inside* arm from the diffusive regime
(`σ_θ/Δ ≈ 0.5–0.9` at T=4e-3) into the mass-protected regime (`σ_θ/Δ ≈ 0.17–0.32`), where escape is
Kramers-like and **exponentially** suppressed. So on an emergent (massive) register the FPT vault must
be **much larger than the designed 87–110×**: I pre-register `n₁/₂(in)/n₁/₂(out) > 300×`, most likely
fully right-censored at a 2e5-step cap (reported then as a **lower bound**).
**G5:** measured FPT vault **< 110×** ⇒ the emergent register is *not* better protected than the
designed one, and the "vault" framing does not transfer to soft modes.

### Verdict rule, stated in advance
- **"The vault transfers to an emergent register"** iff **Q1 ∧ Q2 ∧ Q4** hold (Q3, Q5 are the bonus
  emergent-only statements; their failure is reported but does not by itself revoke transfer).
- **"Designed-only scope confirmation"** iff **G1 ∨ G2 ∨ G4** fires; the report must then name the
  failure mode (bath vs configurational vs field-vs-scalar) explicitly.

---

## 4. Provenance of this prereg
Repo HEAD at writing: `7fcef50`. Env: main venv `/Users/user/Desktop/CHLU/.venv`, jax 0.9.0,
equinox 0.13.4, float64 (`jax_enable_x64`). Checkpoints: `.claude/scratch/v2-full-runs/runs/<tag>/models/exp_d_chlu.pkl`.
Non-default flags in force for every run below: `langevin_noise="fdt"`, `retie(model)` on,
ε(dt)=0.05, Δ=0.5 rad, `FrictionField(gate="compact", gamma_max=0.9, width=0.25, trainable=False)`,
no temperature field, `tilt_delta=0`, `spurion_delta=0`.
