# PREREG — mass-visible-objective (w20)

Written **before** the training harness (`arms.py`) was run. Item-1 zero-gradient
check was already done analytically+numerically (it is an exact identity, not a
predicted ratio, so it needs no prereg); everything below is a commitment.

Baseline = `dt-units-split` @ `0eec592`, FD001, seed 42, 150 epochs, dt=0.125,
data_dt=1.0: common:differential **3.28:1**, `M_max/M_min` **1.3489**
(init 1.2656), mass-gradient shares predict_mse 99.84% / E_contrast 0.00% /
E_reg 0.16%.

## P0 — Item 1 reproduction
Re-running the baseline arm on my branch reproduces the above to ≲5% (my code is
additive with no-op defaults, so I predict **exact** agreement on `M_max/M_min`
and the gradient shares). **`E_contrast` mass gradient remains exactly 0.**

## P1 — candidate (a), mass-perturbed negatives (`neg_momentum_scale=σ`)
**Derivation.** `H = K(p) + V(q)`, and the mass enters only `K`. With negatives
`(q+δq, p)` the kinetic terms cancel identically in `⟨H_data⟩−⟨H_neg⟩`, hence the
zero gradient. With `δp ~ N(0,σ²)` and `K = ½Σᵢpᵢ²/Mᵢ`:

  `E[E_contrast] = −½σ² Σᵢ Mᵢ⁻¹`  ⟹  `∂/∂Mᵢ = +½σ² Mᵢ⁻²`

Two committed consequences:
1. **The gradient scales as σ².** *(Already measured pre-prereg: 24.64× and
   4.12× against predicted 25× and 4× — reported as confirmation, not as a
   prereg pass.)*
2. **The pressure is anti-uniform, not common-mode.** `∂/∂Mᵢ ∝ Mᵢ⁻²` is larger
   for smaller `Mᵢ`, so descent shrinks light channels faster than heavy ones
   and the spectrum **spreads**. This is the key structural prediction.

**Predictions.** At σ=0.5: `E_contrast` share of the mass gradient goes
0.00% → **>10%**; common:differential falls **below 3.28:1**; `M_max/M_min`
rises **above 1.3489** and above its 1.2656 init. I predict a *modest* effect,
**`M_max/M_min` in 1.4–2.5** — the term is one of three and predict_mse holds
99.8% of the mass gradient at baseline.

## P2 — candidate (b), zero-mean (`*_zeromean`)
The common mode is gauge-fixed **by construction**, so "common:differential → 0"
is trivially true and is **not** evidence. The non-trivial prediction is whether
the differential part *grows*: **`M_max/M_min` > 1.3489**.
**Ablation committed:** `softplus_zeromean` with `energy_reg=0` — if the ratio
gain survives, the gain is real; if it vanishes, the "fix" was only re-routing
`E_reg`'s common-mode pressure into the differential direction (the failure mode
the task names). I predict the gain **partially survives** (>1.30).

## P3 — candidate (c), `exp` reparameterization
For `n=14` iid Gaussian `log_mass` of std `s`, `E[max−min] ≈ 3.4 s`, so
`ratio_exp ≈ exp(3.4 s)` whereas softplus compresses toward 1 in its linear
regime. At the baseline final `s = 0.1395` this predicts
**`ratio_exp ≈ exp(0.474) = 1.61`** vs softplus's 1.3489, i.e. `exp` buys
**≈1.2×** more ratio at *identical* `log_mass` statistics. Committed:
**`exp` > `softplus` on `M_max/M_min` at matched `std(log M)`.**

## P4 — Item 3, timescale-vs-mass (**the real acceptance criterion**)
For motion in a locally quadratic potential of curvature `kᵢ`, the response
frequency is `ωᵢ = √(kᵢ/Mᵢ)`, so the relaxation timescale is `τᵢ ∝ √Mᵢ`.

**Committed exponent: slope of `log τ` vs `log M` = +0.5.**

Tolerance: I will call it confirmed if the fitted slope is in **[0.35, 0.65]**
with `R² > 0.5`. A slope ≈0 means the spectrum is cosmetic — masses differ but
the dynamics does not resolve information at different rates — which would be a
**negative result for the whole access-key programme** and must be reported as
such. Caveat committed in advance: `kᵢ` is itself learned and need not be
mass-independent, so a slope below 0.5 does not by itself falsify the mechanism;
I will report the measured `k`–`M` correlation alongside.

## P5 — combined arm
`exp_zeromean` + `neg_momentum_scale=0.5` is predicted to be the best arm on
`M_max/M_min`, and I predict it **exceeds 2.0**.

## What would count as failure
- `M_max/M_min` ≤ init (1.2656) for every candidate ⇒ the objective still cannot
  see the spectrum; report as such.
- Ratio gains that vanish under the `energy_reg=0` ablation ⇒ re-routed
  common-mode pressure, not a fix.
- Timescale slope ≈ 0 ⇒ spread spectrum is not a functional one.
