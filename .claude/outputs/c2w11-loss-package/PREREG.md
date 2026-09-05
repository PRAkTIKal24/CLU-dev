# PREREG — c2w11-loss-package (physics-theorist)

Filed **2026-08-10, before any script in `.claude/scratch/c2w11-loss-package/` existed or was run.**
Protocol §5 pre-registration rule. This task's acceptance criterion is a **formalization**, not a
performance number — but Deliverable 2 asks me to *derive* `ζ`, a critical boundary, a separation
condition and a stopping criterion **in the program's shipped constants**, and those are laws with
numbers. Every number below is derived **by hand, symbolically, before running numpy**; the scripts
exist only to confirm the algebra. Anything the scripts contradict is reported as a **derivation
error of mine**, not quietly corrected.

## Shipped constants assumed (from the task file + `bprime-theory` §0 provenance)
`α = 0.05` (`confine`) ⇒ vacuum curvature floor `2α = 0.10` · `s ≈ 0.32` · `dt = 0.05` ·
two-phase read `(γ,N) = (0.05, 400) → (0.02, 800)` · per-step damping is the SHIPPED
`p ← (1−γ)p` (a *per-step multiplier*, not a `−γ p dt` force), `chlu/core/integrators.py:77`.

## P1 — the continuum reduction of the shipped damping
The per-step multiplier `(1−γ)` is a continuous drag `−Γ_c p` with
`Γ_c = −ln(1−γ)/dt`. **Predicted: `Γ_c(0.05) = 1.025866`, `Γ_c(0.02) = 0.404054`.**

## P2 — the damping ratio
`m q̈ = −λ q − Γ_c m q̇` ⇒ `ζ = Γ_c √(m/λ) / 2`, i.e. **`ζ ∝ √m`** at fixed `Γ_c, λ`.
Predicted, `m = 1`:
| mode | `λ` | `ζ` at γ=0.05 | `ζ` at γ=0.02 |
|---|---|---|---|
| vacuum / floor mode | `2α = 0.10` | **1.6222** (over) | **0.63886** (under) |
| written well, `A=1, s=0.32` | `A/s² + 2α = 9.8656` | **0.16330** (under) | **0.06432** (under) |

## P3 — the under/over-damped boundary in shipped constants
Continuum: `λ_crit = Γ_c² m / 4` ⇒ **`0.263099 m` (γ=0.05)**, **`0.0408151 m` (γ=0.02)**.
Discrete (exact map): critical `κ ≡ dt²λ/m` at `κ_crit = 2[1 − 2√(1−γ)/(2−γ)]` ⇒
**`κ_crit = 6.5744e-4` ⇒ `λ_crit = 0.262976 m`** (γ=0.05) and
**`κ_crit = 1.02041e-4` ⇒ `λ_crit = 0.0408163 m`** (γ=0.02).
**Predicted agreement discrete-vs-continuum: < 0.05 % relative.**

## P4 — two exact algebraic identities of the shipped map (linearized)
(i) **`det A = (1−γ)` exactly**, independent of `λ, m, dt`.
(ii) **In the underdamped regime `|ρ| = √(1−γ)` exactly**, independent of `λ, m, dt`
⇒ **mass has NO first-order effect on the contraction modulus when `λ > λ_crit`; it moves only the
phase.** Predicted verification residual ≤ 1e-14.

## P5 — retention under the SHIPPED two-phase budget
`retention = ρ₊(γ₁)^400 · ρ₊(γ₂)^800`.
- stiff well mode (`λ=9.8656`, both phases underdamped): **1.0854e-8** (= `e^{−18.34}`, reproduces
  `bprime-theory` T3.1's `C = 18.34`).
- floor mode (`λ = 2α = 0.10`, phase 1 **over**damped `ρ₊ = 0.994639`, phase 2 underdamped):
  **3.605e-5**. Ratio soft/stiff ≈ **3.32e3**.
⇒ **predicted verdict: the shipped read erases within-well launch information on EVERY direction the
confinement permits, including the softest one.**

## P6 — `τ_max` in steps
`τ_max = Γ/2α` with `Γ = Γ_c m` ⇒ `m=1`: **10.2587 t.u. = 205.2 steps** (γ=0.05),
**4.0405 t.u. = 80.8 steps** (γ=0.02). Shipped budget ≈ **11.85 τ_max** total.

## P7 — flat-floor stopping (`λ = 0` exactly)
`v̇ = −Γ_c v` ⇒ the particle **stops at finite range** `Δ = v₀/Γ_c = p₀/(m Γ_c)`; with a mass
**tensor**, `Δ_i = p_{0,i}/(m_i Γ_c)`. Predicted verification: discrete map's total displacement
matches `p₀/(mΓ_c)` to < 1 % for `N Γ_c dt ≫ 1`.

## P8 — the (b)-term's designed side-effect: depth heterogeneity moves the basin boundary
Using `PREREG-TierII` §7's O2 offset `δ = ln(A_i/A_j)/(d/s² − 4/d)` at the registered operating
point `d/s = 2.7`, `s = 0.32` ⇒ `d = 0.864`, denominator `= 3.80787`:
- **at depth ratio 3× (the registered minimum heterogeneity): `δ = 0.288507`, `δ/d = 0.33392`.**
- **the shallow well's basin is annihilated (`δ = d/2`) at depth ratio `e^{1.645} = 5.181`.**
Predicted numerical check on a 1-D two-Gaussian-plus-confinement potential: measured separatrix
offset within **±15 %** of `δ` at ratio 3 (O2 is a leading-order result), and the shallow minimum
ceases to exist at a measured ratio in **[4.0, 7.0]**.

## P9 — the A31.2 geometry check (the zero-compute discriminator)
`σ_q/spacing = 0.32` is quoted without saying whether `σ_q` is a **norm** (`E‖q−key‖`) or a
**per-coordinate** std. Predicted 1-NN accuracy over `K` keys in `R^d`, isotropic Gaussian cue noise:
- **norm reading** (`E‖δ‖ = 0.32 · spacing`), `d = 12, K = 16`: **≥ 0.9999** (the along-axis
  component is `≈ 0.32/√12 = 0.0924` of spacing ⇒ a `> 5σ` event is needed to flip).
- **per-coordinate reading** (`σ = 0.32 · spacing` per axis ⇒ `E‖δ‖ ≈ 1.11 · spacing`): **< 0.30**.
⇒ **Predicted finding: the two readings differ by more than the entire span between the banked
launder (0.79–0.90) and the store (≈0.50), so "comfortable geometry" is NOT established by the 0.32
number alone, and which reading is meant is a 1-engineer-hour, zero-compute check that must be run
before any A31.2 mechanism is funded.** I predict the norm reading is the intended one and that it is
**inconsistent with a banked 1-NN launder of 0.79–0.90 on the same population** — i.e. one of the two
banked numbers is measured on a different object.

## Declared in advance
- Every toy below is **1-D / 2-D / low-D designed wells with a single width `s`**. The standing
  `bprime-theory` §9.2 bracket applies: **transfer to a learned multi-atom store is BRACKETED, NOT
  MEASURED.** No number here is a paper number, a claim cell, or a verdict.
- If a script disagrees with a prediction above, the report states the prediction, the measurement,
  and which is wrong — the prediction is not edited.
