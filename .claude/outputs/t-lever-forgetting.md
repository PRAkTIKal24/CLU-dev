# t-lever-forgetting — results-analyst report

**Task + acceptance criterion:** decide whether TEMPERATURE (not friction) is the forgetting lever for flat-direction (coset) memory on *trained* CHLU checkpoints, by confirming F-5's three predictions — (a) the latch survives any γ at T=0, (b) `D_θ = εT(2−γ)/(2F²γ)` and `n₁/₂ ∝ F²γ_c/T`, (c) `∂n₁/₂/∂γ > 0`, the opposite sign to massive-mode forgetting — and, on that basis, recommend **V5 short** vs **V2 appendix**.

**Status: done.** All three predictions confirmed on trained models. Two of them are *stronger* than predicted; one required correcting the deep-dive's framing (the massive mode is non-monotone in γ, see §4). One shipped-code bug found and quantified (§6). Repo read-only — no tracked file touched.

---

## 0. Headline

> **On a trained designed-SO(2) checkpoint, friction cannot erase the coset register at any strength (drift < 5e-12 rad over 200k steps at every γ ∈ [0.002, 0.5], 5 seeds). At T > 0 the register decays diffusively with `D_θ = εT(2−γ)/(2F²γ)` — verified to 1.0068 ± 0.0219 over 25 (γ,T) cells — so `n₁/₂ ∝ γ/T`. Raising friction 4× (0.05 → 0.2) *lengthens* the coset half-life by 3.77 ± 0.23× (5/5 seeds). Temperature is the forgetting lever; friction is an information *preserver*.**

And a sharper unification the deep-dive did not predict:

> The massive mode's `n₁/₂(γ)` is **non-monotone**, with a minimum at critical damping `γ_crit = 2εμ`: `∂n₁/₂/∂γ < 0` only for `γ < γ_crit` (slope −1.006), and `∂n₁/₂/∂γ > 0` above it. **The flat mode is exactly the `μ → 0` corner where `γ_crit → 0`, so the coset is permanently overdamped — that is *why* its γ-dependence has the opposite sign.** The "sign flip" is one curve, not two laws.

---

## 1. Setup — provenance, configs, seeds, commands

### 1.1 Flag-provenance table (mandatory, §5)

| item | value |
|---|---|
| repo commit at run time | `27f232f` → **`9bc2cf7`** (HEAD moved mid-session; concurrent `experiment-engineer` F-1 spurion work) |
| code paths I depend on | `mass_vector`, `effective_inertia`, `effective_mass`, `T`, `H`, `stochastic_step` (chlu_unit), all of `integrators.py`, `SO2InvariantPotential` — **verified byte-identical across `27f232f..9bc2cf7`** (additive `spurion_delta` / `LinearSpurionPotential` only) |
| reproducibility check | headline D-cell re-run at `9bc2cf7`: `D_hat=1.519788e-03`, ratio `0.9945±0.0193` — **bit-identical** to the pre-move run |
| checkpoints | `.claude/scratch/v2-full-runs/runs/designed150_s{42,43,44,45,46}/models/exp_d_chlu.pkl` |
| checkpoint training config | `exp_d` designed SO(2): `dim=4`, `hidden_dim=64`, `train_epochs=150`, `potential_type="so2_invariant"`, `kinetic_energy_mode="newtonian_learned"`, `tie_channel_mass=True`, `sleep_mode="on"`, `lyapunov_penalty="max"`, `anchor_data_energy_lambda=0.0`, `confinement α=0.05` |
| **langevin_noise** | **`"fdt"` everywhere** (`noise_mode="fdt"`). ⚠ repo default is `"legacy"` — **none of these laws hold under legacy noise** (T not in energy units) |
| friction_field / temperature_field | `none` (no field) |
| tilt / spurion | `tilt_delta=0`, `spurion_delta=0` (unbroken, exact Goldstone) |
| dt (ε) | `0.05` throughout |
| γ grid | `{0.005,0.01,0.02,0.05,0.1,0.2}` (S1); `{0.01,0.02,0.05,0.1,0.2}` (S2/S3); dense `geomspace(0.002,0.5,22)` (S4b) |
| T grid | `{0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` |
| precision | **float64** (`jax_enable_x64`), weights cast f32→f64; training was f32 |
| seeds | model seeds 42–46; PRNG keys derived per (seed, T, γ) — recorded in each script |
| **checkpoint retie** | **all quantitative runs use `common.retie(model)`** — folds `tie_channel_mass` into `log_mass`. `H` is **bit-identical** (verified: `0.5520809844448227` both). This is required to work around the FDT bug in §6. Pytree-level only; **no repo edit**. |
| env | jax 0.9.0, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0, main venv (`/Users/user/Desktop/CHLU/.venv`) |

### 1.2 Vacuum geometry (`s0_geometry.json`, retied models)

| seed | r* | M_ch | **F² = M_ch r\*²** | μ²_ang | μ²_rad | γ_crit=2εμ_rad | \|∇V(q*)\| |
|---|---|---|---|---|---|---|---|
| 42 | 0.966993 | 0.683714 | 0.639324 | 1.60e-15 | 0.670302 | 0.0819 | 6.7e-09 |
| 43 | 0.966263 | 0.684413 | 0.639011 | 1.52e-15 | 0.770891 | 0.0878 | 3.7e-09 |
| 44 | 0.979706 | 0.664706 | 0.638001 | −2.92e-16 | 1.190122 | 0.1091 | 5.1e-09 |
| 45 | 0.959136 | 0.660379 | 0.607511 | 2.94e-16 | 1.092699 | 0.1045 | 2.6e-12 |
| 46 | 0.990693 | 0.687486 | 0.674749 | 1.21e-16 | 1.347820 | 0.1161 | 3.7e-14 |

`μ²_ang ≈ 1e-15` = exact flatness (architectural: `SO2InvariantPotential` is `f(r²)+αr²`). **F² spans only 0.608–0.675 (11%)** — see confound C4.

### 1.3 Commands (all from `.claude/scratch/t-lever-forgetting/`, `PYTHONPATH=/Users/user/Desktop/CHLU`, `.venv/bin/python`)

```
s0_setup.py                                                     # geometry
s1_t0_latch.py                                                  # (a) latch @ T=0 + transport law
s1b_amplitude.py                                                # transport-law deficit ∝ p0²?
s2_dlaw.py   --seeds 44 --tag s2                                # (b) D_θ law, 5γ × 5T
s3_halflife.py --seeds 44 --tag s3                              # (c) n₁/₂ grid  [headline]
s3_halflife.py --seeds 42 43 44 45 46 --gammas 0.05 0.1 0.2 \
               --temps 1e-3 1e-2 --tag s3seeds                  # seed variance
s4_massive.py                                                   # massive contrast (superseded parts)
s4b_massive_exact.py                                            # exact Jacobian |λ| for both modes
s4c2_two_by_two.py                                              # T-independence + equipartition
s5_fdt_bug.py ; s5b_fdt_bug_direct.py                           # the shipped FDT bug
s6_toy_control.py                                               # ideal-AR(1)-coset control
s7_figures.py
```

Artifacts → `.claude/outputs/t-lever-forgetting/` (5 PNG, 13 JSON, 4 NPZ). Scripts → `.claude/scratch/t-lever-forgetting/`.

---

## 2. Prediction (a) — the latch survives ANY γ at T = 0 (F5 Cor-13) ✅ **confirmed, exactly**

**Protocol.** Write a coset value with the CLU's native charge impulse `p = a·Xq*` (`|p|=p₀=0.1`, purely tangential), let it settle for `n=60/γ` steps, then **hold for 200,000 further steps** at T=0 and measure the extra angle accrued.

`s1_t0_latch.json` — 5 seeds × 6 γ:

| quantity | result |
|---|---|
| **latch drift over 200k steps** | **max 4.925e-12 rad**, median 9.47e-13 rad (30/30 cells) |
| final momentum \|p\| | ≤ 9.9e-16 (fully drained) |
| radial drift \|r−r*\| | ≤ 4.7e-15 |
| γ=0 control (no friction) | θ advances **142.7 rad / 20k steps**, and 2× that in 2× the steps — ballistic, **no latch** |
| transport law `Δθ = εp₀/(M_ch r* γ)` | ratio **0.9921 ± 0.0061** (30 cells) |

**Exact operator statement (`s4b_jacobian.json`).** The slow eigenvalue of the model's *actual* one-step Jacobian at `(q*, 0)`, restricted to the coset tangent `(Xq*, 0)`:

> `| |λ_flat| − 1 | ≤ 1.7e-14` for **every** γ in `geomspace(0.002, 0.5, 22)`, all 5 seeds (float64 eps = 2.2e-16).

So `n₁/₂(flat, T=0) = ∞` for all γ. Friction has **zero** contractive action on the flat direction — Cor-13, verified on trained weights, not a toy. (Fig 4b.)

**Sub-finding (new): the 0.8% transport-law deficit is a finite-amplitude *write* artifact, not a law failure.** The deficit at fixed p₀ shrinks with γ (1.95% at γ=0.005 → 0.05% at γ=0.2). `s1b_amplitude.json` resolves it: at every γ,

`d log(1 − ratio) / d log p₀ = 1.907, 1.920, 1.981, 1.939` (γ = 0.005, 0.01, 0.05, 0.2) — **∝ p₀², i.e. ∝ v²: centrifugal.**

The tangential write pushes the walker outward to `r_eq > r*`; since `θ` advances as `εL/(M r²)` with `L` the (frictionally-decaying) angular momentum, a larger `r` under-rotates. The measured outward excursion matches `δr = v²/(r* μ_rad²)` when the momentum decay time `1/γ` exceeds the radial period `2π/(εμ_rad) = 115 steps` (γ=0.005: obs 4.4e-4 vs pred 3.0e-4), and collapses when it does not (γ=0.2: obs 9.8e-6 vs pred 3.0e-4 — the radius has no time to respond). At `p₀ = 0.0125` the ratio is **0.99980**. **The F5 transport law is exact in linear response.**

---

## 3. Prediction (b) — the coset-diffusion law ✅ **confirmed to 0.7%**

**Estimator.** Block increments (blocks of `L = max(500, 20/γ)` steps, `L·γ ≥ 20` so blocks are independent — measured lag-1 correlation of squared increments `|c₁| ≤ 0.030`), compared against the **exact finite-N AR(1) partial-sum variance**

`MSD_θ(N) = (ε²T/F²)·[ N(2−γ)/γ − 2(1−γ)(1−(1−γ)^N)/γ² ] → 2 D_θ ε N`.

512 walkers × 10 blocks = 5120 samples/cell (rel. SEM ≈ 2.0%). Walkers start at `q*` with `p ~ N(0, √(M_dyn T))` (Maxwell–Boltzmann, no momentum transient) after a `max(3000, 10/γ)`-step burn-in.

`s2_dlaw_cells.json`, seed 44, 5 γ × 5 T (Fig 1):

| statistic | value | prediction |
|---|---|---|
| `D̂_θ / D_θ^pred` over 25 cells | **1.0068 ± 0.0219** (min 0.9644, max 1.0484) | 1 |
| `d log D / d log T` (per γ) | **+0.9981, +0.9970, +0.9920, +1.0095, +1.0063** | +1 |
| `d log D / d log γ`, raw (per T) | −1.0458, −1.0109, −1.0302, −1.0324, −1.0136 | ≈ −1 |
| `d log D / d log γ`, ÷ (2−γ) | **−1.0144, −0.9795, −0.9988, −1.0010, −0.9822** | **−1 exactly** |

The `(2−γ)` discrete factor is *resolved*: dividing it out moves the exponent from −1.03 to −1.00. Spans 2 decades in T and 1.3 in γ.

**Free bonus — equipartition (`s2b_equipartition_from_rstd.json`, `s4c2_equipartition.json`).** `RMS(r−r*)` vs `√(T/(M_ch μ_rad²))`:

- **1.0023 ± 0.0100** over the 20 harmonic cells (T ≤ 3e-3);
- **1.047 ± 0.016** at T = 1e-2 (anharmonic — the highest-T column is outside linear response);
- co-moving re-measurement (independent code path): **1.0044 ± 0.0056** (T ≤ 1e-3).

So the massive mode has a **bounded** stationary law; the flat mode has **none** (Var(θ) grows linearly forever).

---

## 4. Prediction (c) — the SIGN FLIP ✅ **confirmed** (and the correct framing is richer)

### 4.1 Definition

`n₁/₂` := **median** first-passage step at which a latched coset value has drifted by `|Δθ| ≥ Δ = 0.5` rad (i.e. half the ensemble has lost the value). For a driftless diffusion the exact 2-sided exit statistics give

- mean FPT `= 0.500000 · Δ²/(D_θ ε)`  ← this is the deep-dive's `n₁/₂ ≈ Δ²/(2D_θε)`
- median FPT `= 0.378748 · Δ²/(D_θ ε)` ← calibrated from `S(τ)=(4/π)Σ(−1)^k/(2k+1)e^{−(2k+1)²π²τ/4}`, solving `S=½`.

Substituting the D-law: **`n₁/₂ = 0.378748 · Δ² F² γ / (ε² T (2−γ))`.**

### 4.2 The result (`s3_halflife_cells.json`, `s3seeds_halflife_cells.json`; Fig 2)

**`∂n₁/₂/∂γ > 0` in 10/10 seed×temperature conditions.** Log-slopes `d log n₁/₂ / d log γ` over γ ∈ {0.05, 0.1, 0.2}, 5 seeds:

| T | slope (mean ± sd over 5 seeds) | all > 0 ? | predicted |
|---|---|---|---|
| 1e-3 | **+0.9552 ± 0.0422** | ✅ 5/5 | +1 |
| 1e-2 | +0.6841 ± 0.0480 | ✅ 5/5 | +1 (biased low, see §5) |

**Effect size**, γ: 0.05 → 0.2 at T=1e-3: `n₁/₂` grows **3.77 ± 0.23 ×** (per-seed 3.55, 3.64, 3.61, 3.89, 4.15; ideal-coset toy predicts 3.61 — see §5).

**`∂n₁/₂/∂T < 0`** (seed 44, per-γ): `d log n₁/₂ / d log T` = −0.818, −0.872, −0.922, −0.988 at γ = 0.02, 0.05, 0.1, 0.2; restricted to deep-diffusive cells: **−0.956, −0.979** (γ = 0.1, 0.2). Predicted −1.

Together with §2 (`T=0 ⇒ n₁/₂ = ∞`): **temperature is the forgetting lever; friction fights it.**

### 4.3 The massive-mode contrast — and a correction to the deep-dive's framing

⚠ **The deep-dive (and my task file) states the massive mode simply has `∂n₁/₂/∂γ < 0`. That is only true below critical damping.** Measured exactly from the model's one-step Jacobian on a dense γ grid (`s4b_jacobian.json`, Fig 4a):

| seed | γ_crit = 2εμ_rad | γ of measured minimum | slope, γ < γ_crit/2 | slope, γ > 2γ_crit |
|---|---|---|---|---|
| 42 | 0.0819 | 0.0759 | **−1.0056** | +1.2302 |
| 43 | 0.0878 | 0.0759 | **−1.0056** | +1.2359 |
| 44 | 0.1091 | 0.0961 | **−1.0068** | +1.2666 |
| 45 | 0.1045 | 0.0961 | **−1.0068** | +1.2625 |
| 46 | 0.1161 | 0.0961 | **−1.0068** | +1.2732 |

So the massive mode's `n₁/₂(γ)` is a **V**: `≈ 2ln2/γ` when underdamped (slope −1.006, matching the budget table), a minimum at critical damping `γ_crit = 2εμ`, and `≈ ln2·γ/(ε²μ²)` (slope → +1) when overdamped. The V2 budget table's regime (γ = 0.05, μ_rad ≈ 0.82–1.35 ⇒ γ_crit = 0.082–0.116) sits **on the underdamped branch**, so the reported `∂n/∂γ < 0` is correct there — but it is not a universal property of massive modes.

**The unification.** A flat mode is `μ = 0`, hence `γ_crit = 2εμ = 0`: **the coset is overdamped at every γ > 0.** That is exactly why its `n₁/₂ ∝ γ^{+1}` — it is the `μ→0` limit of the *overdamped* branch, on which more friction always means slower dynamics and longer memory. The "sign flip" is one damping-optimum curve evaluated at two values of μ, not two unrelated laws. (This also retro-explains the repo's existing critical-damping knob `friction_field_c1_lambda`, whose docstring already calls `γ_k → 2·dt·μ(c_k)` "the critical-damping forgetting optimum" — same γ_crit.)

### 4.4 The 2×2 "which knob erases what" table (all four entries measured)

| | `∂(forgetting rate)/∂γ` | `∂(forgetting rate)/∂T` |
|---|---|---|
| **massive (underdamped)** | **> 0** — rate = gap ≈ γ/2; slope of `n₁/₂` = −1.006 | **≈ 0** — rate across T∈{0,1e-4,1e-3}: **max/min ≤ 1.013**, 9/9 (seed,γ) cells (`s4c2_rates.json`); T only sets the *floor* `Var(r−r*) = T/(M_ch μ²)`, verified to 1.0044 ± 0.0056 |
| **flat / coset** | **< 0** — rate ∝ 1/γ; slope of `n₁/₂` = **+0.955 ± 0.042** | **> 0** — rate ∝ T; slope of `n₁/₂` = −0.956…−0.979 |

Also measured: at T>0 the flat mode's **ensemble-mean** coset angle never decays (`|⟨Δθ⟩| / SEM = 0.09–1.45`, i.e. consistent with zero, while `RMS(Δθ)` is 22–348× larger; `s4_flat_mean_rms.json`). **Friction + temperature *diffuse* the register; they never *restore* it.** A massive mode, by contrast, is pulled back to `d = 0`.

*(Caveat on the massive-rate/Jacobian-gap agreement: `rate_obs/gap` = 0.9998, 0.9983, 1.0085 when `γ/γ_crit ≲ 0.25`, but degrades to 0.58–1.19 as `γ → γ_crit`, because at the branch point the response is `(A+Bn)λⁿ`, not a single exponential, and a log-linear envelope fit is biased. The Jacobian `|λ|` is exact and is what §4.3 quotes; the CRN fit is only used for the T-independence claim, where it is a *ratio across T at fixed γ* and the bias cancels.)*

---

## 5. Is the absolute law right? An ideal-coset control (this matters)

The raw `n₁/₂` obs/pred ratio over 20 cells is **1.257 ± 0.261** — not 1. The excess is *structured*: it grows as the diffusive quality `q = n_fp·γ` falls. Two hypotheses: (i) the trained model's coset is not an ideal flat direction; (ii) the first-passage **estimator** is biased at finite thermal persistence.

I simulated the deep-dive's exact flat-mode map (`q_{n+1}=q_n+(ε/M)p_n`, `p_{n+1}=(1−γ)p_n+σ*ξ`, `σ*²=MTγ(2−γ)`) with `scipy.signal.lfilter` (4096 walkers, no JAX) and measured the *same* median FPT (`s6_toy_control.json`, Fig 3):

| | result |
|---|---|
| **`n₁/₂`(trained) / `n₁/₂`(ideal AR(1) coset toy)**, 20 cells | **1.0020 ± 0.0495** (min 0.904, max 1.120) |
| toy / pure-diffusion formula − 1 | `= 3.099·(ℓ_θ/Δ) − 0.0059` (intercept ≈ 0) |
| trained / pure-diffusion formula, deep-diffusive cells (`ℓ_θ/Δ < 0.05`, n=9) | **1.054 ± 0.057** |
| deep-diffusive exponents | `d log n₁/₂/d log γ` = +1.093, +0.947 · `d log n₁/₂/d log T` = −0.956, −0.979 |

with `ℓ_θ/Δ = ε√(T/M)/(γ r* Δ)` the thermal persistence length in units of the tolerance.

**Verdict: hypothesis (ii).** The trained model's coset behaves as the *exact* ideal flat coset, cell by cell, to 0.2 ± 5.0%. The entire excess is the boundary-layer correction of a *persistent* walker exiting an interval — the toy reproduces it with intercept 0 — and it vanishes as `ℓ_θ/Δ → 0`. **The pure-diffusion formula `n₁/₂ = 0.378748·Δ²F²γ/(ε²T(2−γ))` is the `ℓ_θ/Δ → 0` limit, accurate to 5% once `ℓ_θ/Δ < 0.05`.** This also explains why the raw γ-slope at T=1e-2 (+0.684) is shallower than at T=1e-3 (+0.955): the low-γ end of the high-T row sits at `ℓ_θ/Δ = 0.25`, where the estimator inflates `n₁/₂` by ~1.8×.

This is a genuine addition to the deep-dive: **the F-5 law needs the qualifier `Δ ≫ ℓ_θ`**, and `ℓ_θ` is the physical resolution limit of a coset register at temperature T.

---

## 6. CODE BUG (for `experiment-engineer`) — shipped FDT noise violates Gibbs on tied-mass checkpoints

**`chlu/core/chlu_unit.py`:** `H`/`T` use `mass_vector()`, which applies the `tie_channel_mass` log-space mean. `effective_mass()` (line ~296) returns the **raw `softplus(self.log_mass)` — untied**, and also omits the `+1e-6` that `H` inverts. `stochastic_step` builds the FDT noise scale from `effective_mass()`:

```python
m_eff = self.effective_mass() if noise_mode == "fdt" else None   # UNTIED
...
noise_scale = sqrt(m_eff * temperature * gamma * (2 - gamma))    # integrators.py:166
```

**Consequence.** On any `tie_channel_mass=True` checkpoint the noise is injected with a different inertia than the dynamics invert, so `Var(p_i) = effective_mass()_i · T` while Maxwell–Boltzmann for `H` demands `effective_inertia()_i · T`. Each channel coordinate equilibrates at its own temperature `T_eff,i = T · effective_mass()_i / effective_inertia()_i`; **the stationary law is not `exp(−H/T)`** and there is no Gibbs invariant.

**Measured** (`s5b_fdt_bug_direct.json`, 5 seeds, γ=0.05, T=1e-3, 2048 walkers × 40 samples; Fig 5) — channel temperature ratio `T_eff,0/T_eff,1` (Gibbs demands exactly 1):

| arm | seed 42 | 43 | 44 | 45 | 46 | max \|dev\| |
|---|---|---|---|---|---|---|
| **shipped (untied)** | 0.91584 | 0.96911 | 0.99058 | 0.95074 | 1.01903 | **8.42 %** |
| *predicted* `M_noise,0/M_noise,1` | 0.89246 | 0.97332 | 1.00030 | 0.94202 | 1.03002 | |
| **retied** (control) | 1.01304 | 0.99216 | 0.99033 | 0.99961 | 0.99468 | 1.30 % |

and directly, seed 42: `Var(p)_channel = [6.577e-4, 7.182e-4]` vs `effective_mass()·T = [6.456e-4, 7.234e-4]` (ratio 1.019, 0.993 ± 0.005) vs Gibbs `M_dyn·T = [6.837e-4, 6.837e-4]`. The observed anisotropy is slightly *smaller* than predicted because the radial force exchanges energy between `p₀` and `p₁` as the walker rotates, partially re-equilibrating the channel.

**Fix (one line, engineer's call):** make `effective_mass()` delegate to `effective_inertia()` (or have `stochastic_step` call `effective_inertia()`). This changes `noise_mode="fdt"` behaviour **only on `tie_channel_mass=True` checkpoints**; `"legacy"` and untied models are unaffected. My workaround (`common.retie`) folds the tie into `log_mass` at the pytree level, leaving `H` bit-identical — **all quantitative results above use it**, so they are unaffected by the bug.

**Negative result worth recording:** I first tried to see this bug in `D_θ(θ₀)` (`s5_fdt_bug.json`). It is a *bad* instrument — the walker's coset angle wanders by O(1) rad within a measurement block, averaging `⟨cos²θ⟩` and shrinking the predicted +5.8% anisotropy to ~+1.4%, below MC error. Measured paired difference: +0.9% at θ₀=0, −0.8% at θ₀=π/2 (signs correct, magnitude washed out). The momentum-variance test is the right instrument.

---

## 7. Retro-explaining the `γ_φ` −24% negative (task item 2)

`fit-gap-anatomy` N12/N13: on the circle-vacuum task, the learned friction field `γ_φ(q)` (K=2) **recovered −24%** of the twin fit gap, while a global `γ=0.05` recovered 92%.

**First-principles explanation, now measured, not asserted:**

1. **Friction cannot delete latched coset content — at all.** `|λ_flat| = 1` to 1.7e-14 at every γ ∈ [0.002, 0.5] (§2). A `γ_φ(q)` hole multiplies `p` by `(1−γ_φ(q))`; on a flat direction that only drains the momentum of a *write in progress*. Once `p = 0` the register is frozen: `Δθ < 5e-12` rad over 200k steps. A friction field applied to a Goldstone register is a **no-op on stored content**.
2. **Worse: friction is an information *preserver*.** With any global `T > 0` present, `D_θ ∝ 1/γ` (§3), so a `γ_φ` hole *increases* the enclosed register's half-life: `n₁/₂ ∝ γ_eff`. Raising `γ_φ` to 0.5 inside a hole (`γ_eff = 1−(1−0.05)(1−0.5) = 0.525` vs 0.05) would **lengthen** the enclosed coset half-life by `(0.525/1.475)/(0.05/1.95) ≈ 13.9×`. **A friction hole is a memory vault, not a shredder.**
3. **What little the `γ_φ` rung *did* do to the register came from an unintended symmetry break, not from dissipation.** The fit-gap ladder reports the flat `μ²` rising `5.2e-3 → 2.1e-2` (≈4×) and the latch drift moving `0.778 → 0.706` on that rung. Lifting `μ²` is *explicit breaking* (it turns the Goldstone into a pseudo-Goldstone with a finite `T=0` lifetime `n₁/₂ ∝ 1/δ`), which is a **potential**-space effect. The friction field's training perturbed `V`; the friction itself did nothing to the stored value.

So the −24% is not a tuning failure and not noise: **`γ_φ` is provably the wrong operator for the job.** F-6 is confirmed as a *theorem-backed* design conclusion. (Caveat: I did **not** re-run the fit-gap ladder; that rung also retrained with the sleep phase on, so the exact −24% remains a non-matched delta, as `fit-gap-anatomy` itself flags. The *sign and mechanism* are what this report establishes.)

---

## 8. Part 2 — ENGINEER SPEC: a localized temperature field `T_φ(q)` (do NOT build unless Head greenlights V5)

**Goal.** The `T`-analogue of `chlu/core/friction_field.py`: a learned, position-gated hot region that **deletes latched coset content locally**, where a `γ_φ(q)` hole provably cannot (§7). Thread-1's original Hawking framing; `integrators.py:130-133` already names this the "S2 study hook" ("localized bath / Hawking re-emission").

### 8.1 The physics the implementation must satisfy

Let `γ_eff(q) = 1 − (1−γ)(1−γ_φ(q))` be the **total** per-step damping actually applied at `q`, and `T_tot(q) = T + T_φ(q)`. The discrete-FDT sub-step `p' = (1−γ_eff)p + σξ` has stationary `Var = σ²/(γ_eff(2−γ_eff))`, so local Maxwell–Boltzmann requires

```
sigma_i(q) = sqrt( M_eff_i * T_tot(q) * gamma_eff(q) * (2 - gamma_eff(q)) )
```

⚠ **The noise must be built from the same total damping that was applied at that step.** The current `langevin_step` applies scalar `γ`, then `γ_field`, then noise scaled by scalar `γ` only — deliberately (absorb-only sink). For a *temperature* field we want a localized **bath**, so `γ_φ` must be coupled into the noise scale. Keep the absorb-only path reachable behind a flag (`temperature_field_couple_gamma_phi: bool = True`).

Then, inside a hot region, the flat direction obeys (from §3, with local values):

```
D_theta(q) = eps * T_tot(q) * (2 - gamma_eff) / (2 * F^2 * gamma_eff)
n_1/2      = 0.378748 * Delta^2 * F^2 * gamma_eff / (eps^2 * T_tot (2 - gamma_eff))
```

i.e. **erasure rate `∝ T_φ / (γ_c F²)`** as F-6 states — and note the `1/γ_eff`: **a `T_φ` hot spot should NOT be co-located with a `γ_φ` hole**, or the friction will partially protect the very content the heat is meant to destroy. This is a non-obvious, testable design rule that falls straight out of §3.

### 8.2 Exact changes

1. **`chlu/core/temperature_field.py`** (new) — `TemperatureField(eqx.Module)` mirroring `FrictionField`:
   - params: `centers (K,dim)`, `log_radii (K,)`, `raw_strengths (K,)`, static `width w`, `gate ∈ {"sigmoid","compact"}`, static cap `t_max`.
   - `__call__(q) -> T_phi >= 0`, `T_phi = t_max * sum_k softplus_gate(...)`, capped, **`>= 0` strictly** (a negative temperature is not a cooling knob — it is an instability).
   - Mirror `friction_field_gate="compact"` (exact hard cutoff) — the sigmoid tail-leakage retention gap seen in `gamma-field-build` S1 will bite here too, and worse: a leaking `T_φ` tail heats the *whole* ring and destroys every register, not just the targeted one.
2. **`chlu/core/chlu_unit.py`** — add `temperature_field: Optional[eqx.Module]` field + `__init__` kwarg (default `None`); `getattr(self, "temperature_field", None)` guard everywhere (handover §7.13 pattern, as `friction_field` already does). Pass through in `stochastic_step`.
3. **`chlu/core/integrators.py:langevin_step`** — new kwarg `temperature_field=None`. Compute `gamma_eff` and `T_tot` at `q_next`; build `noise_scale` per the formula above. `T_φ ≡ 0` **must be bit-identical** to today (with the §6 fix applied).
4. **`chlu/config.py`** — `temperature_field: str = "none"|"fixed"|"learned"`, `temperature_field_k`, `_t_max`, `_width`, `_gate`, `_init_radius/_strength/_center_scale`, `_fixed_centers`, and **`temperature_field_lr: Optional[float] = 1e-2`** (two-timescale — the `gamma-field-build` lesson that q-space-adjacent parameters cannot move at the base Adam lr).
5. **Training** (mirror the `γ_φ` contrastive scheme): wake pushes `T_φ(q_data)` **down** (protect data), sleep pushes `T_φ(q_hallu)` **up** (damn hallucinations), gated by the existing `friction_field_hallu_gate="energy"` logic.
6. **Blocking prerequisite:** fix §6 first. Any `T_φ` calibration on a tied-mass checkpoint is otherwise off by up to 8.4% per coordinate, and there is no Gibbs measure to calibrate *against*.

### 8.3 Acceptance tests (I will run these; harnesses already exist in this task's scratch)

| # | test | prediction |
|---|---|---|
| **T1** | `T_φ ≡ 0` | bit-identical to current `langevin_step` |
| **T2** | constant `T_φ = c`, measure `Var(p_i)` (`s5b` harness) | `= M_eff_i (T + c)` exactly |
| **T3** | hot region covering coset angles `θ∈[a,b]` on the designed ring; write a latch **inside** | decays with `n₁/₂ = 0.378748 Δ²F²γ_eff/(ε²T_φ(2−γ_eff))` (`s3` harness) |
| **T4** | same, latch written **outside** the hot region, `T=0` globally | preserved: drift `< 1e-11` rad / 200k steps |
| **T5** | **the money control** — replace the `T_φ` hot region with a `γ_φ` hole of *any* strength, `T=0` | latch preserved **both** inside and outside → **0% erasure** (§2) |
| **T6** | as T5 but with global `T = 1e-3` | the enclosed latch decays **~13.9× SLOWER** than outside (`n₁/₂ ∝ γ_eff`). **A friction hole is an information vault.** |

T5+T6 are cheap (hours), need no retraining, and are the sharpest falsifiables in the whole programme: they turn the `γ_φ` −24% negative into a *predicted, quantitative* result.

---

## 9. Limitations & confounds (honest)

- **C1 — one architecture.** Everything is `designed150` (dim=4, hidden 64, 150 ep, exact `SO2InvariantPotential`). The **emergent (MLP) checkpoints were not tested.** Their flat direction is only *approximately* flat (`fit-gap-anatomy` reports `μ² ≈ 5.2e-3` for the circle-vacuum CLU), so there the latch already has a finite `T=0` lifetime and the pure sign flip may be masked by a pseudo-Goldstone relaxation channel. **This is the single biggest generalization risk and the top next experiment.**
- **C2 — retie.** Headline numbers use retied checkpoints (§6). On the shipped path the channel temperature is anisotropic by ≤ 8.4%, which perturbs `D_θ` by a few % but not the exponents.
- **C3 — estimator bias.** `n₁/₂` carries an `O(ℓ_θ/Δ)` boundary-layer bias, fully characterized and controlled against an exact toy (§5). Raw ratios (1.257 ± 0.261) should never be quoted without it; the deep-diffusive figure is **1.054 ± 0.057**.
- **C4 — the `F²` scaling is NOT independently established here.** All 5 seeds have `F² ∈ [0.608, 0.675]` (11% spread), comparable to the ~7% per-cell statistical error on `n₁/₂`. `F²` enters only through the absolute prediction (which matched to 5.4% in deep cells). To test `n₁/₂ ∝ F²` properly, retrain with a designed `r*` sweep (e.g. vary the confinement `α`) → **recommended experiment R2**.
- **C5 — near-critical-damping fits.** `rate_obs/gap` for the massive mode degrades to 0.58–1.19 as `γ → γ_crit` (the response is `(A+Bn)λⁿ`, not a single exponential). §4.3's numbers come from the exact Jacobian `|λ|`, which has no such issue.
- **C6 — T = 1e-2 is anharmonic** (equipartition ratio 1.047–1.112). The top-T column of every sweep is outside linear response; exponents restricted to `T ≤ 3e-3` are the trustworthy ones.
- **C7 — `langevin_noise="fdt"` throughout; the repo default is `"legacy"`.** Under legacy noise `σ = √(2γTdt)` uniformly, so `T` is not in energy units and each mode equilibrates at its own temperature — **none of these laws hold**. Any paper text must state the flag.
- **C8 — the Arrhenius / pseudo-Goldstone hop channel (`n_hop ∝ e^{2δr*/T}`), the third leg of F-5, was NOT tested on trained models.** The deep-dive verified it only in the toy (ratio 1.074). It needs `tilt_delta > 0` checkpoints.
- **C9 — `s4c` (my first 2×2 attempt) was wrong** and is superseded by `s4c2`: I projected `q − q*` on the *fixed* Cartesian radial eigenvector at `q*`, which, once the coset angle has diffused by O(1) rad, picks up the *angular* excursion (`Var(d_rad)` saturated at 20–300× `T/μ²`). The radial mode must be read co-moving (`r = |q_ch|`, `p_r = (q_ch·p_ch)/r`). `s4c_*.json` are retained on disk but **must not be used**. Likewise `s4_massive_vsT.json` (part [B]) returned *negative* decay rates — its fit ran into the ensemble-mean noise floor; superseded by `s4c2_rates.json`.

---

## 10. Deliverable — the scope recommendation

**Recommendation: V5-worthy — *conditional* on Part 2 (T5/T6 at minimum) landing. If the Head will not fund the `T_φ` build, this is a strong two-figure V2 appendix, not a short.**

**The case for V5 (a coherent "physics of forgetting" short, sibling to V2):**
1. A counterintuitive, quotable headline verified on trained weights, 5 seeds, 10/10 conditions: **friction remembers, temperature forgets.** Effect size 3.77 ± 0.23× (not marginal).
2. A quantitative law confirmed to **0.7%** over 25 cells and 2 decades of T (Fig 1), with the discrete `(2−γ)` factor resolved.
3. An **exact** structural statement: `|λ_flat| = 1` to 1.7e-14 at every γ — Cor-13 as an operator identity on trained weights (Fig 4b).
4. A **unification the deep-dive did not have**: the massive mode's `n₁/₂(γ)` is non-monotone with a minimum at `γ_crit = 2εμ`; the coset is the `μ→0` corner where `γ_crit → 0`, hence permanently overdamped. One curve, two regimes. It also subsumes the repo's existing `friction_field_c1_lambda` critical-damping knob.
5. A **new physical resolution limit** (§5): the coset register's tolerance must exceed the thermal persistence length `ℓ_θ = ε√(T/M)/(γ r*)`. This is not in F-5.
6. It **explains an existing negative** (`γ_φ` −24%) from first principles, and converts it into a *predicted* falsifiable (T5/T6: a friction hole is a 13.9× memory vault).
7. It delivers a **design change** (`T_φ(q)`, §8) with a spec, plus a **design rule** (never co-locate `T_φ` with `γ_φ`).
8. It found a **real shipped bug** (§6) with 5-seed evidence.

**The honest case against (why it is conditional):**
- **Everything is one architecture** (C1). The emergent-potential case — where CHLU's flat directions actually *come from* — is untested, and it is precisely the case where the clean sign flip could blur into a pseudo-Goldstone relaxation. A "Forgetting" short that only exhibits a *designed* exact symmetry is answerable with "you verified the Einstein relation on a symmetry you put in by hand."
- **The mechanism is unbuilt.** `T_φ(q)` is a spec, not a result. The short's punchline ("here is the operator that forgets") currently has no experiment behind it.
- **No task-level payoff.** Nobody has shown `T_φ` improves fit, generation, or continual learning. V2 ("Memory") had the register; "Forgetting" currently has a law and an absence.
- At bottom, §3 is the fluctuation–dissipation theorem. The genuinely new content is (4), (5), (6), (7) — strong, but they lean on the `T_φ` build to become a *story* rather than a *note*.

**Decision rule for the Head.** Fund the `T_φ` build + T3/T5/T6 (cheap: no retraining, harnesses exist) + one emergent-checkpoint replication of §4.2 (C1). If T5/T6 land as predicted — friction hole preserves, temperature hole erases, on the same checkpoint, same figure — **that single figure is the V5 short**, and it is a sibling to V2 in exactly the way the task hopes. If they do not, fold Fig 2 + Fig 4 into a V2 appendix titled "the T = 0 face of the budget cube" and move on.

---

## 11. Recommended next experiments (priority order)

| id | experiment | cost | why |
|---|---|---|---|
| **R1** | Replicate §4.2 (sign flip) on **emergent (MLP) `emergent150_s{42,43,44}`** checkpoints, which have a near-flat but not exact direction. Predict: the `T=0` latch now decays with `n₁/₂ = ln2/gap(μ²_ang, ε, γ)` (pseudo-Goldstone), so at small T the `∂n/∂γ` sign flips back to negative below a crossover `T* ≈ ε²μ²_ang F²/(...)`. **Measuring `T*` is a new falsifiable.** | small (harnesses exist) | **kills or confirms C1, the main risk to V5** |
| **R2** | `F²` scaling: retrain designed SO(2) at 3–4 confinements `α` to spread `r*` (hence `F²`) by ≥ 3×; test `n₁/₂ ∝ F²`. | medium (retrain) | closes C4 — the only untested factor in the F-5 law |
| **R3** | T5/T6 (`γ_φ` hole preserves; §8.3) — **no new code needed**, `FrictionField` already exists. | hours | the sharpest cheap falsifiable in the programme; makes §7 a *prediction*, not a post-hoc |
| **R4** | Build `T_φ(q)` per §8; run T1–T4. | medium | the V5 mechanism |
| **R5** | Arrhenius channel (C8) on `tilt_delta > 0` checkpoints: `n_hop ∝ e^{2δr*/T}`. Completes the `(μ, γ, T)` budget cube. | small | third leg of F-5, currently toy-only |
| **R6** | Fix §6, then re-verify `Var(p_i) = M_eff,i T` on all tied checkpoints; audit whether any *shipped result* used `fdt` + `tie_channel_mass` (I believe none did — `fdt` is not the default). | small | correctness |

---

## Git footprint

**None.** No tracked file was created, modified, or deleted. Repo read-only, as the task requires. `git status --short` clean throughout. All artifacts live under `.claude/` (gitignored):

- scripts → `.claude/scratch/t-lever-forgetting/{common,kernels,s0_setup,s1_t0_latch,s1b_amplitude,s2_dlaw,s3_halflife,s4_massive,s4b_massive_exact,s4c_two_by_two,s4c2_two_by_two,s5_fdt_bug,s5b_fdt_bug_direct,s6_toy_control,s7_figures}.py`
- results → `.claude/outputs/t-lever-forgetting/` (`fig1_dlaw.png`, `fig2_signflip.png`, `fig3_toy_control.png`, `fig4_massive_vs_flat.png`, `fig5_fdt_bug.png`; `s0_geometry.json`, `s1_t0_latch.json`, `s1b_amplitude.json`, `s2_dlaw_cells.json`, `s2_exponents_s44.json`, `s2b_equipartition_from_rstd.json`, `s3_halflife_cells.json`, `s3seeds_halflife_cells.json`, `s4b_jacobian.json`, `s4c2_rates.json`, `s4c2_equipartition.json`, `s5b_fdt_bug_direct.json`, `s6_toy_control.json`; `s3_fpt.npz`, `s4b_curves.npz`, `s4c2_curves.npz`, `s2_curves`-family)
- ⚠ **do not use** `s4c_rates.json`, `s4c_equipartition.json`, `s4_massive_vsT.json` (superseded, see C9). `s5_fdt_bug.json` is a documented negative (wrong instrument, §6).

---

## Proposed handover updates (for the Hub)

### For §1.6 / a new §1.9 "Forgetting: the (μ, γ, T) budget cube"

> **F-5/F-6 confirmed on trained models** (`t-lever-forgetting`, 5 designed-SO(2) seeds, `langevin_noise="fdt"`, ε=0.05, retied checkpoints, commit `9bc2cf7`):
> - **(a) Cor-13 exact.** `| |λ_flat| − 1 | ≤ 1.7e-14` at every γ ∈ [0.002, 0.5]; latch drift ≤ **4.9e-12 rad over 200k steps**, 30/30 (seed,γ) cells. Friction cannot erase a Goldstone register. The γ=0 control drifts 142.7 rad/20k steps (no latch).
> - **(b) `D_θ = εT(2−γ)/(2F²γ)` verified to `1.0068 ± 0.0219`** over 25 (γ,T) cells; `d log D/d log T = +0.992…+1.010`; `d log D/d log γ` after dividing the discrete `(2−γ)` factor `= −0.980…−1.014`.
> - **(c) SIGN FLIP: `∂n₁/₂/∂γ > 0`, 10/10 seed×T conditions.** `d log n₁/₂/d log γ = +0.955 ± 0.042` (T=1e-3, 5 seeds). γ: 0.05→0.2 lengthens the coset half-life **3.77 ± 0.23×**. `d log n₁/₂/d log T = −0.956…−0.979`.
> - **Absolute law:** `n₁/₂(median) = 0.378748·Δ²F²γ/(ε²T(2−γ))` (the `0.5` prefactor is the *mean* FPT). Verified to **1.054 ± 0.057** once `ℓ_θ/Δ < 0.05`.
> - **New qualifier (not in the deep-dive):** the law requires `Δ ≫ ℓ_θ = ε√(T/M_ch)/(γ r*)`, the thermal persistence length — the physical resolution limit of a coset register at temperature T. Below that the walker is ballistic and `n₁/₂` inflates by `≈(1+ℓ_θ/Δ)²`. Controlled against an exact AR(1)-coset toy: **`n₁/₂(trained)/n₁/₂(toy) = 1.0020 ± 0.0495`** over 20 cells — the trained coset *is* the ideal flat coset.

### CORRECTION to the deep-dive §5 / F-5 framing (please fold in)

> The claim "more friction ⇒ *faster* forgetting for a massive mode" is **true only below critical damping.** Measured exactly (one-step Jacobian, dense γ, 5 seeds): massive `n₁/₂(γ)` is **non-monotone**, minimum at `γ_crit = 2εμ` (obs 0.076–0.096 vs pred 0.082–0.116); slope **−1.006** for `γ < γ_crit/2`, **+1.23…+1.27** for `γ > 2γ_crit`. **The flat mode is the `μ→0` corner where `γ_crit → 0`, so the coset is permanently overdamped — that is *why* `n₁/₂ ∝ γ^{+1}`.** The sign flip is one damping-optimum curve at two values of μ, which also subsumes the repo's `friction_field_c1_lambda` "critical-damping forgetting optimum". The V2 budget table (γ=0.05, μ_rad 0.82–1.35) does sit on the underdamped branch, so its reported sign is correct.

> **2×2 knob table (all four entries measured).** Massive: `∂rate/∂γ > 0` (underdamped), `∂rate/∂T ≈ 0` (max/min ≤ **1.013** across T∈{0,1e-4,1e-3}, 9/9 cells — T sets the *floor* `Var(r−r*) = T/(M_ch μ²)`, verified `1.0044 ± 0.0056`). Flat: `∂rate/∂γ < 0`, `∂rate/∂T > 0`. At T>0 the flat mode's ensemble-*mean* coset angle never decays (|⟨Δθ⟩|/SEM = 0.09–1.45) while its RMS grows 22–348× larger: **friction+temperature diffuse the register, they never restore it.**

### For §5 (provenance) and §7 (known issues) — **new bug, please log**

> **BUG (found by `t-lever-forgetting`, 2026-07-09).** `CHLU.effective_mass()` returns raw `softplus(log_mass)` — it does **not** apply `tie_channel_mass` (which `mass_vector()`, and hence `H`/`T`, does) and omits the `+1e-6` that `H` inverts. `stochastic_step` builds the `noise_mode="fdt"` scale from it, so on any `tie_channel_mass=True` checkpoint the Langevin noise uses a different inertia than the dynamics: `Var(p_i) = effective_mass()_i·T` instead of the Gibbs-required `effective_inertia()_i·T`. Channel coords equilibrate at different temperatures — **no Gibbs invariant.** Measured, 5 seeds, γ=0.05, T=1e-3: `T_eff,0/T_eff,1` = 0.916, 0.969, 0.991, 0.951, 1.019 (max deviation **8.4%**), tracking the predicted `M_noise,0/M_noise,1`; a retied control gives 1.013, 0.992, 0.990, 1.000, 0.995. Fix: `effective_mass()` should delegate to `effective_inertia()`. Affects **only** `fdt` + tied checkpoints (`legacy` is the repo default, so no shipped result is believed contaminated — worth an audit). Assign: `experiment-engineer`. Evidence: `.claude/outputs/t-lever-forgetting/s5b_fdt_bug_direct.json`, `fig5_fdt_bug.png`.

> **Provenance note:** HEAD moved `27f232f → 9bc2cf7` mid-session (concurrent `experiment-engineer` F-1 spurion work). All code paths used here were verified byte-identical across that range, and the headline D-cell reproduces bit-identically at `9bc2cf7` (`D_hat = 1.519788e-03`).

### For §8 (open directions) — scope call

> **"Forgetting" = V5-worthy CONDITIONAL on the `T_φ` build.** Strong physics core (sign flip + critical-damping unification + `γ_φ` negative explained from first principles + the `ℓ_θ` resolution limit), one design change (`T_φ(q)`), one design rule (never co-locate `T_φ` with `γ_φ` — friction *protects* the content the heat should destroy). **But:** single architecture (designed, exact SO(2)); mechanism unbuilt; no task payoff. **Gate on R1 (emergent-checkpoint replication) + R3 (the free `γ_φ`-hole-preserves falsifiable, T5/T6).** If R3 lands — *"a friction hole is a 13.9× memory vault; a temperature hole is a shredder"*, same checkpoint, one figure — that is the V5 short. Otherwise: V2 appendix, Fig 2 + Fig 4, titled "the T=0 face of the budget cube".

> **`fit-gap-anatomy` N12/N13 (γ_φ −24%) now has a first-principles explanation** — not a tuning failure: (i) friction has *zero* action on flat-direction content (`|λ_flat|=1` to 1e-14); (ii) with T>0 it *lengthens* coset memory (`D∝1/γ`); (iii) the small latch-drift change (0.778→0.706) on that rung tracks the flat `μ²` rising 5.2e-3→2.1e-2, i.e. an unintended **explicit symmetry break** in `V`, not dissipation. `γ_φ` is provably the wrong operator; the trash-region programme needs `T_φ(q)`.
