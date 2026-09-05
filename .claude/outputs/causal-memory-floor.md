# causal-memory-floor — results-analyst report

**Task + acceptance criterion:** measure (and *attack*) the F-10 causal memory-lifetime floor in both kinetic modes; reproduce D-saturation vs linear growth on the coded path; verify the F-11 companding law across decades with the γ-conditional aliasing protection stated as conditional; probe O7's large-ζ failure mode. Every relativistic cell carries `T/(m₀c²)` and the R8 non-Gibbs caveat.

**Status: done.** All acceptance items met. **The headline result is a falsification-with-repair:** R7's floor **as written (`n ≥ Δ/(ε·θ̇_max)`) is FALSE on a curved coset** — I break it in 332/787 adversarial cells, by up to **3.14×** — but the *assumption-free* bound it was trying to express is a **Cartesian light-cone bound**, which is never violated (0/787) and is **exactly attained** (`n = ⌈d_min(Δ)/(ε v_max)⌉` for every Δ ≤ π). Two further corrections: my own proposed `γ_c/γ` refinement to R6 is **refuted**; and R3's transport law is a **stiff-ring** law that fails at **1% by ζ₀ ≈ 0.19** on the actual trained ring (not 0.2% at ζ ≤ 3). Two new closed forms derived and verified: the relativistic **diffusion saturation constant** (arcsine law) and its **dimensional** generalization. Repo read-only.

---

## 0. Headline

> **Relativistic mode buys a *pathwise*, assumption-free memory guarantee: `|q_{n+1} − q_n| < ε·c/√M_min` at every step, for any noise, temperature, or adversary** — verified to 1 ulp at `|p| = 10¹⁸`, and saturated (`max |dq|/bound = 0.999999516` over 70 thermal cells, `1 − 1.6e-15` under an `A = 10⁷` adversary). Newtonian mode has no such bound: its per-step displacement grows as `√T` (measured `0.034 → 302` over `T ∈ [10⁻², 10⁶]`) and a single step erases the register.
>
> **The floor is real, but it lives in the ambient metric, not in the coset angle.** On the trained SO(2) ring an adversary that *dives toward the origin* trades radius for angular speed: `θ̇ = v/r ≤ v_max/r`, and `r` is the adversary's to choose. The correct floor is
> `n ≥ d_min(Δ)/(ε·v_max)`, `d_min(Δ) = r*·sin Δ` (Δ ≤ π/2), `= r*` (Δ > π/2)
> — i.e. **the erasure cost saturates at `1/(ε·θ̇_max)` and stops growing with Δ.** R7's `Δ/(ε·θ̇_max)` is recovered only under the extra hypothesis `r ≥ r*`, and only for `Δ ≤ 1 rad` is it the optimal-attack cost.

And the ML-facing statement (§9) is unaffected and in fact *strengthened*: bounded forgetting under unbounded noise injection, with an explicit constant.

---

## 1. Setup

### 1.1 Flag-provenance table (mandatory, protocol §5)

| item | value |
|---|---|
| repo commit | **`d6f8bac`** (`[experiment-engineer] fix FDT noise inertia: effective_mass() -> effective_inertia()`); `git status --short` clean before, during and after |
| repo edits | **none** (read-only, as the task requires) |
| checkpoints | `.claude/scratch/v2-full-runs/runs/designed150_s{42,43,44,45,46}/models/exp_d_chlu.pkl` |
| checkpoint training config | `exp_d` designed SO(2): `dim=4`, `hidden_dim=64`, `train_epochs=150`, `potential_type="so2_invariant"`, **`kinetic_energy_mode="newtonian_learned"`**, `tie_channel_mass=True`, `sleep_mode="on"`, `lyapunov_penalty="max"`, `anchor_data_energy_lambda=0.0`, confinement `α=0.05` (per `t-lever-forgetting` §1.1) |
| **arm construction** | `cmf.make_arm`: fresh `CHLU(kinetic_mode=…)` + `eqx.tree_at` graft of the trained `(log_mass, potential_net)`. `kinetic_mode` is an eqx **static** field so it cannot be `tree_at`-ed. **`V(q)` is bit-identical across arms: max abs diff = `0.0` exactly**, 64 random `q`, 5 seeds (`s0_setup.json`) |
| **M_eff matched at rest — MEASURED, not assumed** | `M_meas,i := lim_{p→0} p_i /(∂T/∂p_i)`. rel/newt max rel. dev **≤ 8.44e-15** (5 seeds × 4 coords). `m₀ = 1` ⇒ `M_eff^rel = m₀M = M = M_eff^newt` (`effective_inertia`, `chlu_unit.py:196`) |
| `rest_mass` m₀ | `1.0` (code default) |
| `c` | `1.0` (code default) ⇒ **`T/(m₀c²) = T` numerically in every table below** |
| `langevin_noise` | **`"fdt"`** (`noise_mode="fdt"`) for s1/s2. ⚠ repo default is `"legacy"`. **s3/s3b/s4/s4b are `T = 0` — no noise model at all.** |
| **R8 / CM-17 caveat** | In relativistic mode the coded O-step `p←(1−γ)p+σξ` is a **linear OU recursion** ⇒ Gaussian stationary law; relativistic Gibbs demands **Maxwell–Jüttner**. **No σ gives it a Gibbs invariant.** ⇒ **No relativistic `T` in this report is an equilibrium temperature.** Every quantitative claim here is either (a) **pathwise** (the floor, the certificate) — needs no invariant measure — or (b) a statement about **the coded chain's own stationary law** `p ~ N(0, M_eff T)`, which I use as the walker initialization and which is exactly what the sampler realizes. Ensembles initialized at `N(0, M_eff T)` therefore have **no momentum transient in either arm**. |
| friction γ | `0.05` (s1, s2, s4, s4b); `{0.05,0.02,0.01,0.005,0.002}` (s4 Part B); **`0.0`** (s3, s3b — worst case for the defender) |
| dt (ε) | `0.05` throughout |
| Δ (erasure tolerance) | `0.5` rad (s1, s2 — matches `t-lever-forgetting`); swept `{0.5, 1.0, π/2, 2.0, 3.0, π, 2π}` (s3) |
| T grid | s1: `{1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1e3,1e4,1e5,1e6}` (8 decades); s2: `{1e-2…1e6}` (9 pts) |
| adversary amplitude A | `{1e-1,1,3,10,1e2,1e3,1e5,1e7}` |
| ensembles | s1: 512 walkers × 5 seeds × 2 arms × 14 T = 70 cells/arm; s2: **16384** walkers; s3: deterministic (1 trajectory/cell), 840 rel cells |
| precision | **float64** (`jax_enable_x64=True`, set in `cmf.py` before `jax.numpy` binds); weights cast f32→f64; training was f32 |
| `retie` | applied (folds `tie_channel_mass` into `log_mass`; leaves `mass_vector()`/`H` bit-identical). **No longer load-bearing at `d6f8bac`** — the `effective_mass` bug is fixed — kept so the arm graft cannot reintroduce a tie mismatch |
| seeds | model seeds 42–46; PRNG keys derived per (seed, arm, T) and recorded in each script |
| env | jax **0.9.0**, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0, matplotlib 3.10.8, main venv `/Users/user/Desktop/CHLU/.venv` |

### 1.2 Vacuum geometry & causal constants (`s0_setup.json`, relativistic arm)

| seed | r\* | M_ch (inertial) | μ²_ang | μ_rad | `v_max = c/√M_ch` | `θ̇_max = v_max/r*` | `ε·θ̇_max` | `n_floor(Δ=0.5)` |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.966993 | 0.683715 | −3.2e-15 | 0.8187 | 1.209379 | 1.250660 | 0.062533 | 7.996 |
| 43 | 0.966263 | 0.684414 | +1.5e-15 | 0.8780 | 1.208762 | 1.250966 | 0.062548 | 7.994 |
| 44 | 0.979706 | 0.664707 | −1.8e-15 | 1.0910 | 1.226549 | 1.251955 | 0.062598 | 7.988 |
| 45 | 0.959136 | 0.660380 | 0.0 | 1.0453 | 1.230561 | 1.282988 | 0.064149 | 7.794 |
| 46 | 0.990693 | 0.687487 | +1.2e-16 | 1.1610 | 1.206057 | 1.217386 | 0.060869 | 8.214 |

`μ²_ang ≈ 1e-15` = exact flatness (architectural). **`M_eff` measured = `M_eff` declared to 8.4e-15 in both arms.** Measured `v_max` per coordinate = `c/√M_i` to `< 1e-16` relative (relativistic); Newtonian `|∂T/∂p|` at `p = 10⁹` is `1.46e9` — **unbounded**.

### 1.3 Commands

All from `.claude/scratch/causal-memory-floor/`, `PYTHONPATH=/Users/user/Desktop/CHLU`, `/Users/user/Desktop/CHLU/.venv/bin/python`:

```
s0_setup.py             # geometry, arm construction, MEASURED rest-inertia match, v_max
s1_thermal.py           # F-10 thermal sweep on the trained ring (8 decades of T)
s2_gutter.py            # D-saturation on a STRICTLY FLAT direction (G1 d=1, G2 d=4)
s3_adversary.py         # THE ATTACK: tangent / chord / compress impulse trains, T=0
s3b_attack_graded.py    # certificate at the source; rho-graded compression attack
s4_companding.py        # F-11 A: exact flat law | B: aliasing p_crit(gamma) | C: curved ring (O7)
s4b_o7_stiffness.py     # O7 resolved: R3 error vs radial stiffness mu_rad
s5_dim_saturation.py    # the dimensional saturation law D_sat(d)
s6_figures.py
```
Artifacts → `.claude/outputs/causal-memory-floor/` (5 PNG, 8 JSON, 1 NPZ). Scripts → `.claude/scratch/causal-memory-floor/`. JAX was warm (~4 s import); total compute ≈ 22 min.

---

## 2. F-10 Part 1 — the thermal floor on the trained ring ✅ (`s1`, Fig 1)

`γ=0.05`, `fdt`, `Δ=0.5`, 512 walkers × 5 seeds. `n_erase` = **median** first-passage of `|Δθ| ≥ 0.5`; `n_min` = minimum over all 2560 walkers.

| `T` = `T/(m₀c²)` | newt `n_med` | newt `n_min` | newt `max|dq|` | rel `n_med` | rel `n_min` | rel `max|dq|` |
|---|---|---|---|---|---|---|
| 1e-2 | 214.7 | 29 | 0.0377 | 226.7 | 26 | 0.0321 |
| 1e-1 | 38.1 | 7 | 0.100 | 42.9 | 12 | 0.0522 |
| 1 | 11.4 | 3 | 0.320 | 22.7 | 9 | 0.0635 |
| 10 | 4.0 | **1** | 0.983 | 19.3 | 8 | 0.0651 |
| 1e2 | 1.8 | **1** | 3.236 | 19.6 | 8 | 0.0655 |
| 1e3 | **1.0** | **1** | 10.23 | 19.0 | 8 | 0.0655 |
| 1e4 | 1.0 | 1 | 32.49 | 19.0 | 8 | 0.0655 |
| 1e5 | 1.0 | 1 | 97.37 | 19.1 | 8 | 0.0655 |
| 1e6 | 1.0 | 1 | **302.1** | 18.8 | **8** | 0.0655 |

- **Floor respected in 70/70 relativistic cells** at every T, every seed (`n_min ≥ ⌊Δ/(ε θ̇_max)⌋ = 7`; observed `n_min = 8` = the exact integer floor since `Δ/(εθ̇_max) = 7.996`).
- **Relativistic erasure time SATURATES**: `n_med = 18.98 ± 0.84` for all `T ≥ 10³` (i.e. 2.4× the floor). Newtonian → **1 step** and cannot go lower (discretization, not physics).
- **Pathwise certificate**: `max_n |q_{n+1} − q_n| / (ε c/√M_min) = 0.999999516` (max over all 70 relativistic cells). Newtonian `max|dq|` grows `∝ √T` without bound.
- ⚠ **The angular-diffusion coefficient `D_θ` on this ring is NOT the coset diffusivity for `T ≳ 0.1`**: `RMS(r−r*) = √(T/(M_ch μ_rad²))` reaches `r*` already at `T ≈ 0.4`, and the ring dissolves (min radius over the run drops to `4e-3` at `T = 0.1`). This is precisely why the D-saturation claim needs a strictly flat control (§3), and it is a real limitation of testing R7's *diffusive corollary* on a trained checkpoint.

---

## 3. R7's diffusive corollary — D saturates ✅, and I derive the constant (`s2`, `s5`, Fig 2)

The deep-dive's `D_rel: 0.0093 → 0.671` is a **strictly-flat-direction** statement. Reproduced on the coded path with two matched-inertia geometries (`M=1`, `c=m₀=1`, `γ=ε=0.05`, 16384 walkers, `N_MSD=500`):
**G1** = `dim=1, V≡0` (Verlet kicks vanish identically ⇒ `p_{n+1}=(1−γ)p_n+σξ` **exactly**); **G2** = `dim=4`, flat in `q₀`, three stiff transverse DOF (`k=100`).

### 3.1 G1 — measured (`s2_gutter_cells.json`)

| `T` = `T/(m₀c²)` | `D_newt` | `D_rel` | `D_rel/D_newt` |
|---|---|---|---|
| 1e-2 | 9.451e-3 | 9.281e-3 | **0.982** |
| 1e-1 | 9.335e-2 | 7.381e-2 | 0.791 |
| 1 | 9.232e-1 | 3.069e-1 | 0.332 |
| 1e1 | 9.665e0 | 5.423e-1 | 5.61e-2 |
| 1e2 | 9.417e1 | 6.360e-1 | 6.75e-3 |
| 1e3 | 9.503e2 | 6.459e-1 | **6.80e-4** |
| 1e6 | 9.439e5 | 6.596e-1 | **6.99e-7** |

**`D_newt` linear over 8 decades** (ratio to `εT(2−γ)/(2Mγ)` finite-N: `1.000 ± 0.02`). **`D_rel` saturates at ≈ 0.65.** Deep-dive's `0.0093 → 0.671` and `0.971 → 7.0e-4` are **reproduced** (mine: `0.00928 → 0.646` and `0.982 → 6.80e-4` at the same `T ∈ [0.01,1000]`).

### 3.2 New closed form — the **arcsine law**

The deep-dive had only the number. On a strictly flat direction `v(p) = v_max·tanh ζ`, `sinh ζ = p/(√M m₀c)`, and `p ~ N(0, MT)` is the coded chain's exact stationary law. As `T → ∞` the velocity retains only the **sign** of `p`, so the AR(1) autocorrelation `ρ_k = (1−γ)^k` is replaced by its Gaussian **sign**-correlation:

```
ρ_k → (2/π)·arcsin((1−γ)^k)
D_sat = (ε·v_max²/2)·[ 1 + (4/π)·Σ_{k≥1} arcsin((1−γ)^k) ]        [NEW, exact]
```

At `ε=γ=0.05, v_max=1`: `D_sat(N→∞) = 0.677788`; truncated at the simulated window `N=500`: **`0.653034`**. Measured `D_rel(T=1e5) = 0.653238 ± 0.0069` → **ratio 1.0003**. (The deep-dive's `0.671` corresponds to `N ≈ 2000`.)

⚠ **Methodological note (a real trap):** the finite-`T` theory curve requires `E[u(g₁)u(g₂)]` for a bivariate normal, and **Gauss–Hermite quadrature fails for `τ = T/(m₀c²) ≫ 1`** — the integrand has a boundary layer of width `1/√τ` at `g=0` that GH-120 nodes (spacing ~0.1 near 0) cannot resolve. Symptom: `ρ₁(GH-120) = 0.8338` at `τ=1e6` vs the exact `0.797835`, and `D_pred` becomes **non-monotone in T**. The sign limit is analytic and was used for `τ ≥ 10³`; every cell in `s2_gutter_cells.json` carries a `theory_method` tag.

### 3.3 New: the **dimensional** saturation law (`s5`, found while running s2)

In relativistic mode the transverse momenta enter the **shared square root** and eat the causal budget: `v₀ → v_max·ĝ₀` with `ĝ` uniform on `S^{d−1}`, so `Var(v₀) = v_max²/d`. There is **no Newtonian analogue** (`v₀ = p₀/M` is blind to `p_{j≠0}`).

```
D_sat(d) = (ε·v_max²/(2d))·[ 1 + 2·Σ_k ρ_k^{(d)} ],   ρ_k^{(d)} = Corr(ĝ₀(n), ĝ₀(n+k))
```

| d | `D_sat(N=500)` predicted | measured | ratio | naive `D₁/d` |
|---|---|---|---|---|
| 1 | 0.653034 (exact arcsine) | 0.653238 ± 0.0069 | **1.0003** | 0.653034 |
| 4 | 0.214291 (MC, 2e6) | 0.212723 ± 0.0023 | **0.9927** | 0.163259 ✗ |

Newtonian `D` is **identical in G1 and G2** (`9.37e-3` both, all T). **Relativistic flat-direction diffusivity falls with latent width; Newtonian is dimension-blind.** The naive `1/d` is wrong by 30% — the direction vector decorrelates more slowly than a sign. *This is a new, untested-elsewhere prediction: relativistic registers get more robust as the latent gets wider.*

---

## 4. F-10 Part 2 — **THE ATTACK** (`s3`, `s3b`, Fig 3)

Threat model: at the start of every step the adversary **overwrites** `p ← A·û(q)` (equivalently injects an unbounded impulse). `T = 0`, `γ = 0` — the adversary is the only forcing, and no noise-model/Gibbs question arises. Everything else is the shipped `model.step`.

### 4.1 The two floors

| bound | statement | status |
|---|---|---|
| **ANGULAR** (R7 as written) | `n ≥ Δ/(ε·θ̇_max)`, `θ̇_max = c/(√M_ch r*)` | **BROKEN** — silently assumes `r ≥ r*` |
| **CARTESIAN** (the true light cone) | `\|q_n − q_0\| ≤ n·ε·c/√M_min` ⇒ `n ≥ d_min(Δ)/(ε·v_max)` | **holds, and is exactly attained** |

with `d_min(Δ) = dist(q*, {ray at angle θ₀+Δ}) = r*·sin Δ` for `Δ ≤ π/2`, `= r*` for `Δ > π/2` (infimum, approached by grazing the origin).

The mechanism is elementary once seen: `θ̇ = v_t/r ≤ v_max/r`, and **`r` is the adversary's to choose.** A straight chord from `q*` passes *inside* the vacuum circle, so `θ̇ > θ̇_max` along it.

### 4.2 Result: 332/787 angular-floor violations, **0/787** Cartesian (`A = 10⁷`, all 5 seeds)

| Δ (rad) | ang-floor violations | min `n` observed | `⌈d_min/(ε v_max)⌉` | angular floor | overstatement |
|---|---|---|---|---|---|
| 0.5 | 0/120 | 8 | 8 (7.67) | 8.00 | 1.04× (< 1 step — invisible) |
| 1.0 | 30/120 | **14** | 14 (13.46) | 15.99 | 1.14× |
| π/2 | 70/120 | **16** | 16 (15.99) | 25.12 | 1.57× |
| 2.0 | 70/120 | **16** | 16 (15.99) | 31.99 | 2.00× |
| 3.0 | 70/120 | **16** | 16 (15.99) | 47.98 | 3.00× |
| π | 70/120 | **16** | 16 (15.99) | 50.24 | **3.14×** |
| 2π | 22/67 | 21 | 16 (15.99) | 100.48 | 4.79× |

- **Tightest ratio `n_erase / floor_cartesian = 1.0005`.** The Cartesian floor is not merely valid, it is *achieved*.
- The `Δ/sin Δ` correction is exact: at `Δ=1.0`, `⌈13.456⌉ = 14` = measured; at `Δ=π/2`, `⌈15.992⌉ = 16` = measured.
- **The erasure cost stops growing with Δ above π/2** (`16` steps for `Δ = π/2, 2, 3, π` alike). Wanting the register "more wrong" is free once you are allowed to pass near the origin.
- At `Δ = 0.5` (the tolerance every prior CHLU experiment used) the violation is 4.3% — **smaller than one integer step**. *That is why nobody saw this.*

### 4.3 It is not an origin singularity (`s3b`, Fig 3b)

Graded dive radius `ρ`, `A=10⁵`: `n_erase(ρ)` tracks the continuum cost `[(r*−ρ) + ρΔ]/(ε v_max)` to ≲1 step, and

```
∂n/∂ρ ∝ (Δ − 1)   ⇒   compression HELPS iff Δ > 1 rad
```

Measured (seed 42): `Δ=0.5`: `9 → 16` as `ρ: r* → 0.01r*` (compression **hurts**). `Δ=1.0`: `17,17,17,18,17,17,17,16,16,16` — **flat**, the predicted degenerate crossover. `Δ=2.0`: `33 → 16`. `Δ=π`: `51 → 16`. The crossover sits exactly at **Δ = 1 rad**, with no free parameters.

### 4.4 The certificate, measured at the source (`s3b`)

`s3` logged 35/840 cells where `max|q_{n+1}−q_n|` exceeded `ε c/√M_min` by `≤ 1.55e-15` relative — **all at `A = 10⁷`**. This is the float64 rounding of the *subtraction* `q_next − q` (two `O(1)` numbers, `ulp(1)=2.2e-16`, `|dq| ≈ 0.06` ⇒ `~1.8e-15` relative), **not a dynamical violation.** Evaluating the bound at its source:

| seed | `max |∂T/∂p| / v_max`, random `p` over 7 decades (2e5 samples) | at `p = 10¹⁸` on the min-mass axis |
|---|---|---|
| 42 | 0.99998… | 1.00000000000000022 |
| 43 | 0.99997599828484451 | 1.00000000000000022 |
| 44 | 0.99998787122543364 | 1.00000000000000022 |
| 45 | 0.99999988040519705 | 1.00000000000000000 |
| 46 | 0.99994202390765485 | 0.99999999999999978 |

`|∂T/∂p| ≤ v_max` to **1 ulp**. The inequality is strict in exact arithmetic (`|tanh ζ| < 1`) and saturates in float64. **Zero true violations.**

---

## 5. F-11 — the rapidity register (`s4`, Fig 4)

### 5.1 Part A: R3 is **EXACT** on a strictly flat direction

`dim=1`, `V≡0`, `M=0.683715` (the trained `M_ch`), `γ=ε=0.05`, 26 values of `p₀` over 5 decades, `4000` steps:

```
Δx = ε·v_max·Σ_{n≥0} tanh ζₙ ,   sinh ζₙ = (1−γ)ⁿ sinh ζ₀ ,   sinh ζ₀ = p₀/(√M m₀c)
```
**max relative error `4.22e-15`** (relativistic); Newtonian control `Δx = εp₀/(Mγ)`, max rel err `4.11e-15`. Machine precision, both arms, all `p₀`.

**Companding, on our trained `M_ch`:**

| `p₀` | `ζ₀` | rel `Δx` | newt `Δx` | compression | local slope `d log Δx/d log p₀` |
|---|---|---|---|---|---|
| 0.01 | 0.012 | 0.014626 | 0.014626 | 1.00× | 0.9999 |
| 1 | 1.022 | 1.2282 | 1.4626 | 1.2× | 0.745 |
| 10 | 3.188 | 3.7879 | 14.626 | 3.9× | 0.312 |
| 100 | 5.488 | 6.5005 | 146.26 | 22.5× | 0.182 |
| 500 | 6.897 | 8.3978 | 731.30 | 87.1× | — |
| 1000 | 7.791 | 9.2149 | 1462.60 | **158.7×** | **0.132** |

Slope → `1/ζ₀ = 0.128` — **transport is logarithmic in the impulse.** (Task's toy numbers `Δθ=7.678` vs `625.0` at `M=0.8`; ours differ only through `M_ch=0.6837`, same structure.)

### 5.2 Part B: aliasing protection is **conditional on γ** — and my own correction is REFUTED

`p_crit` := write impulse whose flat-direction transport equals `2π`. Newtonian `p_crit^N = 2π M_ch r* γ/ε`; relativistic by Brent bisection on the exact R3 sum. Seed-42 geometry (`ε·θ̇_max = 0.062533`), `ζ* = 2πγ_c/(ε θ̇_max)`, `γ_c = −ln(1−γ)`:

| γ | ζ\* | `p_crit^N` | `p_crit^rel` | **measured ratio** | `sinh ζ*/ζ*` (R6) | `(sinh ζ*/ζ*)(γ_c/γ)` (mine) |
|---|---|---|---|---|---|---|
| 0.05 | 5.1538 | 4.1541 | 69.749 | **16.790** | 16.792 ✅ | 17.227 ✗ (+2.6%) |
| 0.02 | 2.0299 | 1.6616 | 3.0623 | **1.8429** | 1.8430 ✅ | 1.8616 ✗ |
| 0.01 | 1.0098 | 0.8308 | 0.9794 | **1.1788** | 1.1788 ✅ | 1.1848 ✗ |
| 0.005 | 0.5036 | 0.4154 | 0.4332 | **1.0428** | 1.0428 ✅ | 1.0454 |
| 0.002 | 0.2012 | 0.1662 | 0.1673 | **1.0068** | 1.0068 ✅ | 1.0078 |

- **R6's `sinh(ζ*)/ζ*` is confirmed to ≤ 1.1e-4 relative** against exact bisection, across a 25× range of γ.
- **I proposed a `γ_c/γ` discrete correction and it is wrong** (2.6% at γ=0.05). The ultra-relativistic sum `Σ tanh ζₙ = ζ₀/γ_c + C + o(1)` carries an `O(1)` offset `C` that cancels it. **Reported rather than dropped.**
- **γ-conditionality confirmed, including a cell where the protection is ABSENT**: `16.8× → 1.007×` as `γ: 0.05 → 0.002`. At `γ ≤ 0.005` the protection is `≤ 4.3%` — *nil*. X4 stands.
- ⚠ **Our factor is `16.8×`, not the deep-dive's `48.9×`.** Their toy had `c/(√M r*) = 1` exactly ⇒ `ε θ̇_max = 0.05`; the trained ring has `θ̇_max = 1.2507` ⇒ `ε θ̇_max = 0.0625`, and `sinh(ζ*)/ζ*` is exponentially sensitive to it. **The protection factor is geometry-dependent and ~3× smaller on the actual checkpoint.** Any paper text must quote the geometry.

### 5.3 Part C + O7: on the **curved trained ring** the law fails early, and aliasing is **unreachable**

Same tangential write on `designed150`, 26 `p₀` × 5 seeds × 2 arms, 4000 steps (`s4_companding.json`, Fig 4c):

| `ζ₀` at which `|Δθ_sim/Δθ_R3 − 1|` first exceeds | 0.2% | 1% | 10% | 50% |
|---|---|---|---|---|
| relativistic, 5 seeds | **0.076–0.122** | **0.190–0.194** | 0.702–0.714 | 2.27–2.74 |

So on the trained ring R3 is already **1% wrong by `ζ₀ ≈ 0.19`** and **65% wrong at `ζ₀ = 3.19`** — not "0.2% at ζ ≤ 3". (The deep-dive's 0.2% is its **R1 `F_Q` form-factor** check, a different observable; **R3's transport on a curved coset had never been measured.** O7 was right to be open.)

**Two facts nobody has looked at (task item 8):**

1. **A hard write does NOT dislodge the latch.** `0/130` cells dislodged, **both arms**; `max |r_final/r* − 1| = 6.4e-15` (rel), `4.3e-15` (newt) over all `p₀ ≤ 1000`. Friction returns the state to the vacuum manifold to machine precision. *The companding register's natural failure mode is not dislodgement.*
2. **It is transport saturation.** `Δθ(p₀→∞)` **saturates at `1.4841 ± 0.0027` rad (rel) and `1.5701 ± 0.0002` rad (newt ≈ π/2)** — the hard write ejects the state outward, `θ̇ = v/r → 0`, and the angle converges. (`π/2` is exact for a Newtonian free particle launched tangentially from radius `r*`.) **The `2π` aliasing threshold is therefore UNREACHABLE by a tangential write on this ring, in either arm.** R6's protection factor describes a threshold the geometry forbids you from reaching.
3. **What the governor *actually* buys on the curved ring is transient containment.** At `p₀=1000` the write-induced excursion is `r_max/r* = 7.96 ± 0.29` (rel) vs **`994 ± 226` (newt)** — a **125× smaller blast radius**. This is the memory-side analogue of the paper's Exp-II velocity-saturation figure, and it is the honest F-11 headline for a *curved* register.

### 5.4 O7 resolved: R3 is a **stiff-ring** law (`s4b`, Fig 5)

Exact `MexicanHatPotential` ring (`r*=1`, `M=1`, `θ̇_max=1`), sweeping the radial stiffness `μ_rad` (Verlet needs `ε μ < 2` ⇒ `μ < 40`):

| `μ_rad` | `μ/θ̇_max` | rel err @ζ₀=0.5 | @ζ₀=3.0 | `r_max/r* − 1` @ζ₀=3 |
|---|---|---|---|---|
| 1.0 | 1.0 | 4.99e-2 | 3.84e-1 | 0.920 |
| 4.0 | 4.0 | 8.86e-3 | 1.25e-1 | 0.317 |
| 8.0 | 8.0 | 2.46e-3 | 6.04e-2 | 0.142 |
| 16.0 | 16.0 | 6.13e-4 | 2.20e-2 | 0.052 |
| 32.0 | 32.0 | **1.33e-4** | 6.78e-3 | 0.016 |

- **In linear response (`ζ₀ = 0.5`) the error is `∝ μ_rad^{-2}`** exactly as the centrifugal mechanism predicts (`δr = v²/(r*μ_rad²)`): consecutive-doubling error ratios `2.91, 3.60, 4.01` → 4.
- At `ζ₀ = 3` the excursion is **non-perturbative** and the error falls only as `μ_rad^{-1.7}`; `0.2% at ζ₀=3` would need `μ_rad ≳ 60`, which **violates Verlet stability at ε=0.05**. So *no* Verlet-stable ring reproduces the deep-dive's 0.2%-at-ζ≤3 for the **transport** observable.
- **Scope statement:** R3's transport law requires `θ̇_max ≪ μ_rad`. The trained `designed150` ring has `θ̇_max/μ_rad ∈ [1.05, 1.53]` — **O(1)** — so the closed form is *inapplicable there*, exactly as measured.
- *Hypothesis (not verified):* the residual excess of `r_max` over `v²/(r*μ²)` at large ζ (13× at μ=16) is the Lorentz enhancement of the transverse inertia (`γ_L = cosh ζ = 10.07` at ζ=3). Flagged as a conjecture, not a result.

---

## 6. Figures

| file | content |
|---|---|
| `.claude/outputs/causal-memory-floor/fig1_floor_vs_T.png` | (a) `n_erase` vs T, trained ring, both arms + floor; (b) pathwise certificate: `max|dq|` bounded (rel) vs `∝√T` (newt) |
| `.claude/outputs/causal-memory-floor/fig2_D_saturation.png` | `D` vs T on the strictly flat direction: `D_rel` saturates at the derived `D_sat`, `D_newt` linear; (b) `d=4` and the dimensional law |
| `.claude/outputs/causal-memory-floor/fig3_attack.png` | **the money figure**: `n_erase` vs Δ for 3 adversaries, angular floor (broken) vs Cartesian floor (attained); ρ-graded attack with the `Δ=1 rad` crossover |
| `.claude/outputs/causal-memory-floor/fig4_companding.png` | R3 exact on the flat direction; aliasing protection vs γ (incl. absent cell + my refuted correction); curved-ring transport saturating below `π/2 ≪ 2π` |
| `.claude/outputs/causal-memory-floor/fig5_o7_stiffness.png` | R3 error vs `μ_rad` (stiff-ring law, trained band shaded); write-induced excursion `994×` vs `7.96×` |

JSON: `s0_setup.json`, `s1_thermal_cells.json`, `s2_gutter_cells.json`, `s3_adversary.json`, `s3b_graded.json`, `s4_companding.json`, `s4b_o7_stiffness.json`, `s5_dim_saturation.json`; NPZ: `s1_msd_acc.npz`.

---

## 7. Limitations & confounds (honest)

- **C1 — one architecture.** All trained-checkpoint work is `designed150` (dim=4, exact `SO2InvariantPotential`, 150 ep). **Emergent (MLP) checkpoints untested.** Their coset is only approximately flat, so the write and the attack both acquire a pseudo-Goldstone channel. Same top risk `t-lever-forgetting` flagged (its C1).
- **C2 — the relativistic arm is a KINETIC-MODE SWAP on a checkpoint TRAINED NEWTONIAN.** `V_θ` and `log_mass` were optimized under `newtonian_learned`. The rest inertia and the vacuum geometry are identical by construction (§1.1–1.2), so the *comparison at rest* is fair, but a model **trained** relativistic could learn a different `M_ch`/`r*` and hence a different `θ̇_max`. **All F-10/F-11 numbers here are "governor swapped in at inference", not "governor trained in".** This is the single biggest confound for any claim about what the relativistic governor buys *in a trained system*. (It is also exactly the regime V2 would be in if it adopted the governor post hoc.)
- **C3 — the floor is per-step, so it is `ε`-dependent.** `n ≥ Δ/(ε θ̇_max)` counts *steps*, not time; the physical bound `|Δq|/Δt ≤ v_max` is `ε`-free. Halving `ε` doubles the step floor and buys nothing physically. Any abstract claim must say "steps at fixed ε" or convert to time.
- **C4 — matched at rest ≠ matched in flight.** `M_eff = m₀M` holds at `p≈0` (verified to 8e-15). At finite rapidity the relativistic coset inertia is `γ_L`-enhanced (`F_Q² = F_Q²(0) cosh ζ`, R1), so the two arms are *not* dynamically matched once the register is being written or driven hot. Stated, not controlled.
- **C5 — `D` on the trained ring is not a coset diffusivity for `T ≳ 0.1`** (the ring dissolves; §2). All D-saturation claims come from the **strictly flat** controls (G1/G2), not from trained weights. The deep-dive's `D` numbers are likewise flat-direction numbers.
- **C6 — the adversary is unbounded in impulse *and* in energy.** It can drive the state to `r ≈ 0` against `V`. Under a **bounded-energy** or bounded-`|Δp|` threat model the compression attack is priced and the angular floor is (partly) restored. **This is the most important untested variant** and the obvious next experiment (R2 below). What survives *any* threat model is the Cartesian bound.
- **C7 — R8/CM-17.** The relativistic `fdt` chain has no Gibbs invariant. I never claim one: §2/§3 initialize and describe the **coded chain's own** stationary momentum law `N(0, M_eff T)`, and §4–§5 are `T=0`. But it does mean **`T` in §2–§3 is a noise-amplitude knob, not a thermodynamic temperature**, and `D_newt = εT(2−γ)/(2Mγ)` is the Gibbs-correct law only in the Newtonian arm.
- **C8 — `Δθ` accumulation near the origin.** The `compress` adversary passes within `5.6e-19` of `r=0`, where the polar angle is undefined; per-step `|dθ|` can approach π. The `Δ ≤ π` results use the **wrapped readout** (well-defined) and agree with the unwrapped accumulator; only the `Δ=2π` row depends on unwrapping through the singularity and should be read as indicative.
- **C9 — GH quadrature failure** (§3.2). Any downstream reuse of a finite-`τ` relativistic `D` curve must not use Gauss–Hermite; `theory_method` is tagged per cell.
- **C10 — single trajectory per adversarial cell.** s3/s3b are deterministic (`T=0`), so there is no ensemble variance — but also no seed-averaging beyond the 5 model seeds. The floor statements are per-trajectory identities, so this is appropriate; the *policies* are hand-designed, not optimized, so `min n` is an **upper bound on the optimal attack** (it happens to meet the lower bound, which is why we can say "attained").

---

## 8. Recommended next experiments

| id | experiment | cost | why |
|---|---|---|---|
| **R1** | **Bounded-impulse / bounded-energy adversary.** Cap `|Δp| ≤ B` (or the injected energy per step) and re-run s3's Δ-sweep. Predict a crossover: for small `B` the compression dive is unaffordable and the angular floor is restored; extract `B*(Δ)`. | small (harness exists) | **closes C6, the main scope hole.** Turns "the floor is Cartesian" into a *quantitative* threat-model boundary — this is what a security-flavored ML claim needs. |
| **R2** | **Train the relativistic arm.** Retrain `exp_d` designed SO(2) with `kinetic_energy_mode="relativistic"` (5 seeds) and re-measure `θ̇_max`, the floor, and the write law. | medium (retrain) | **closes C2**, the biggest confound. Only this can say what the governor buys *in a trained system* rather than swapped in. |
| **R3** | **Emergent (MLP) checkpoints** — replicate §2 + §4 on `emergent150_s{42,43,44}`. Predict: pseudo-Goldstone `μ²_ang > 0` gives a finite `T=0` lifetime that competes with the causal floor; a crossover `T*` where the floor stops binding. | small | closes C1; same top risk as t-lever's R1 — do them together. |
| **R4** | **Dimensional saturation on a trained lattice.** `D_sat(d)` (§3.3) predicts relativistic registers get more robust with latent width. Test on a V3 CLU-lattice at `d ∈ {2,4,8,16}`. | medium | a **falsifiable, ML-relevant** prediction that no one has looked at; directly relevant to "does the governor scale". |
| **R5** | **Retention–bandwidth (F-12)** now has the tools: `θ̇_max` measured per seed (§1.2), `n₁/₂` harness from t-lever. Test `n₁/₂·θ̇_max² = const` across V3 mass bands. | small | the last unmeasured leg of §7bis. |
| **R6** | **`ε`-sweep of the floor** (C3): confirm `n_floor ∝ 1/ε` and `t_floor = Δ r*/v_max` is `ε`-invariant. | trivial | pre-empts an obvious reviewer objection. |

---

## 9. Deliverable — the guarantee, in ML terms

> **Bounded forgetting under unbounded noise injection.** A CHLU running the relativistic kinetic governor satisfies a hard, per-step displacement bound in latent space — `‖q_{n+1} − q_n‖ < ε·c/√(u ᵀM u)` along any unit direction `u` — that follows from the integrator identity `‖∇_p T‖ < c/√M_min` alone. It therefore holds **pathwise**: for any noise process, any temperature, any adversary with unbounded impulse budget, and with no assumption of stationarity, detailed balance, or a Gibbs invariant (which the coded relativistic Langevin provably does not have). Consequently, corrupting a stored value by a latent distance `d` requires at least `d/(ε·c/√M)` steps, and the erasure rate of a flat (Goldstone) memory register saturates: its diffusion coefficient is capped at `D_sat = (ε v_max²/2)[1 + (4/π)Σ_k arcsin((1−γ)^k)]` no matter how hot the noise, whereas the Newtonian unit's `D` grows linearly in the noise power without bound (measured: `D_rel/D_newt` falls from `0.982` to `7.0e-7` as `T` sweeps `10⁻²→10⁶`, and a single Newtonian step erases the register at `T ≳ 10`, where the relativistic unit still needs `≥ 8`). Two caveats are load-bearing and must travel with the claim. First, the bound lives in the **ambient latent metric, not in the register's coordinate**: on a curved memory manifold an adversary can trade radius for angular speed, so the angular guarantee `n ≥ Δ/(ε·θ̇_max)` holds only for tolerances `Δ ≤ 1 rad` and only while the state stays on the vacuum manifold; the assumption-free statement is the light-cone one, whose cost saturates at `r*/(ε v_max)` and does **not** grow with `Δ`. Second, the guarantee is a **causality bound, not an equilibrium one** — it says how fast a value *can* be destroyed, never that it *will* be preserved; temperature remains the only eraser of a flat direction (CM-16), and the governor's contribution is to put a finite, computable speed limit on that eraser.

---

## Git footprint

**None.** No tracked file created, modified, or deleted. `git status --short` clean throughout; HEAD `d6f8bac` unchanged. All artifacts under `.claude/` (gitignored):
- scripts → `.claude/scratch/causal-memory-floor/{cmf,s0_setup,s1_thermal,s2_gutter,s3_adversary,s3b_attack_graded,s4_companding,s4b_o7_stiffness,s5_dim_saturation,s6_figures}.py`
- results → `.claude/outputs/causal-memory-floor/` (5 PNG, 8 JSON, 1 NPZ)

**No new repo bug found.** The `effective_mass()` FDT bug reported by `t-lever-forgetting` §6 is **fixed at `d6f8bac`** and verified: `effective_mass()` now delegates to `effective_inertia()`, `M_eff_measured == M_eff_declared` to 8.4e-15 on tied checkpoints in both kinetic modes. `common.retie` is no longer load-bearing.

---

## Proposed handover updates (for the Hub)

### For §1.6 / §1.9 — **new claim CM-18 (F-10), and a CORRECTION to §7bis R7**

> **CM-18 — The causal memory floor is a LIGHT-CONE bound, not an angular one (w14, `causal-memory-floor`).** In `relativistic` mode the coded position update obeys `‖q_{n+1}−q_n‖ < ε·c/√(uᵀMu)` for every direction `u`, **pathwise, for any noise/temperature/adversary and with no invariant measure** (verified: `|∂T/∂p|/v_max ≤ 1.0` to **1 ulp** at `|p|=10¹⁸`; `max|dq|/bound = 0.999999516` over 70 thermal cells, `T/(m₀c²) ∈ [10⁻²,10⁶]`). Newtonian mode has **no** bound (`max|dq| ∝ √T`: `0.034 → 302`; a single step erases at `T ≳ 10`). **⚠ CORRECTION to §7bis R7:** the published form `n ≥ Δ/(ε·θ̇_max)` **silently assumes `r ≥ r*`** and is **FALSE on a curved coset** — `θ̇ = v/r` and `r` is the adversary's to choose. Measured on trained `designed150` (5 seeds, `T=0`, `A=10⁷`): **332/787 adversarial cells violate the angular floor**, by `Δ/sin Δ` for `Δ ≤ π/2` and by `Δ` above it (**3.14× at Δ=π**). **0/787 violate the Cartesian floor**, which is **exactly attained**: `n_min = ⌈d_min(Δ)/(ε v_max)⌉` for every `Δ ≤ π`, tightest ratio **1.0005**, with `d_min(Δ) = r*sin Δ` (`Δ≤π/2`), `= r*` (`Δ>π/2`). **Erasure cost SATURATES at `1/(ε θ̇_max)` and does not grow with Δ.** Compression helps iff `Δ > 1 rad` (crossover measured exactly at 1 rad, ρ-graded attack, no free parameters). At `Δ=0.5` (the tolerance all prior CHLU work used) the violation is 4.3% — **below one integer step**, which is why it was never seen. **Scope:** adversary unbounded in impulse *and* energy; a bounded-energy threat model is untested (→ R1). Provenance: commit `d6f8bac`, `dt=0.05`, `γ=0` (s3) / `0.05` (s1), `m₀=c=1`, kinetic-mode swap on `newtonian_learned`-trained checkpoints. |
> **thermal side:** floor respected **70/70** cells; relativistic `n_erase` **saturates at 18.98 ± 0.84** steps for all `T ≥ 10³` while Newtonian → **1**; the minimum over 2560 walkers hits **exactly the integer floor (8)**, i.e. *high-T thermal noise saturates the causal floor.*

> **R7 diffusive corollary — REPRODUCED on the coded path, and the constant is now derived.** On a strictly flat direction (`d=1`, `V≡0`, `M=c=m₀=1`, `ε=γ=0.05`, 16384 walkers): `D_newt` linear over **8 decades** (ratio to `εT(2−γ)/(2Mγ)`: `1.000 ± 0.02`); **`D_rel` saturates**, `0.00928 → 0.6596`; `D_rel/D_newt = 0.982 → 6.99e-7` (deep-dive: `0.0093 → 0.671`, `0.971 → 7.0e-4` — **confirmed**). **NEW closed form (arcsine law):** at saturation the velocity retains only `sign(p)`, so the AR(1) autocorrelation `(1−γ)^k` is replaced by its Gaussian sign-correlation `(2/π)arcsin((1−γ)^k)`, giving `D_sat = (ε v_max²/2)[1 + (4/π)Σ_{k≥1} arcsin((1−γ)^k)] = 0.677788` (`N→∞`), `0.653034` at the simulated `N=500`; **measured `0.653238 ± 0.0069`, ratio 1.0003.** **NEW dimensional law:** transverse momenta enter the shared square root, so `Var(v₀) = v_max²/d` and `D_sat(d=4) = 0.214291` predicted vs **`0.212723 ± 0.0023` measured** (ratio 0.9927); the naive `D₁/d` is 30% wrong. **Newtonian `D` is dimension-blind (identical at d=1 and d=4).** ⇒ *relativistic registers get MORE robust as the latent widens* — a new, falsifiable, ML-relevant prediction (→ R4).

### For §1.6 — **F-11 verdict (three qualifiers, all load-bearing)**

> **F-11 confirmed where it is exact, and scoped where it is not.** (a) **R3 is EXACT on a strictly flat direction**: `Δx = ε v_max Σ tanh ζₙ` to **4.22e-15** over 5 decades of `p₀` (Newtonian control `4.11e-15`). Transport is **logarithmic**: local slope `d log Δx/d log p₀ → 0.132 ≈ 1/ζ₀` at `p₀=1000`, where the relativistic latch stores `9.21` and the Newtonian `1462.6` (**159× companding**). (b) **R6's `sinh(ζ*)/ζ*` confirmed to ≤1.1e-4** against exact bisection, and **conditional on γ exactly as X4 says**: `16.79× (γ=0.05) → 1.007× (γ=0.002)`, **protection ABSENT at γ ≤ 0.005**. ⚠ **The factor is geometry-dependent: `16.8×` on the trained ring, not the deep-dive's `48.9×`** (their toy had `c/(√M r*)=1`; ours is `1.2507`). Quote the geometry or don't quote the number. (c) ⚠ **A `γ_c/γ` correction to R6 that I derived is REFUTED by the exact bisection** (+2.6% at γ=0.05); the `O(1)` offset in `Σ tanh ζₙ = ζ₀/γ_c + C` cancels it. Reported rather than dropped.

### For §1.6 / open questions — **O7 CLOSED (and it bites)**

> **O7 resolved: R3's transport law is a STIFF-RING law, `θ̇_max ≪ μ_rad`.** On an exact MexicanHat ring the error is `∝ μ_rad^{-2}` in linear response (doubling ratios `2.91, 3.60, 4.01 → 4`) and `∝ μ_rad^{-1.7}` at `ζ₀=3`. **The trained `designed150` ring has `θ̇_max/μ_rad ∈ [1.05, 1.53]` — O(1) — so the closed form is inapplicable there:** measured error exceeds **1% by `ζ₀ ≈ 0.19`** and **65% at `ζ₀ = 3.19`** (5 seeds). The deep-dive's "0.2% at ζ≤3" refers to its **R1 `F_Q` form-factor** check, a different observable; `0.2%` on the *transport* observable at `ζ=3` would require `μ_rad ≳ 60`, which **violates Verlet stability (`εμ<2`) at ε=0.05**. **Two facts on the curved register, previously unlooked-at:** (i) **a hard write does NOT dislodge the latch** — `0/130` cells, both arms, `max|r_final/r*−1| = 6.4e-15`; friction restores the orbit exactly. (ii) **The failure mode is transport saturation, not aliasing**: `Δθ(p₀→∞) → 1.484 ± 0.003` rad (rel) and `1.5701 ± 0.0002 ≈ π/2` (newt) — **the `2π` aliasing threshold is unreachable by a tangential write on this ring in either arm**, so R6's protection factor guards a threshold the geometry already forbids. **What the governor actually buys on a curved register is transient containment:** write-induced excursion `r_max/r* = 7.96 ± 0.29` (rel) vs **`994 ± 226` (newt)** at `p₀=1000` — a **125× smaller blast radius**, the memory-side analogue of the paper's Exp-II velocity-saturation figure.

### For §5 (provenance) / §7 (known issues)

> **`t-lever-forgetting`'s FDT bug is FIXED and verified** at `d6f8bac` (`effective_mass() → effective_inertia()`): `M_eff_measured == M_eff_declared` to **8.4e-15** on `tie_channel_mass=True` checkpoints, in **both** kinetic modes. `common.retie` is no longer load-bearing for correctness. **No new repo bug found by this task.**

> **New harness note.** `kinetic_mode` is an eqx **static** field and cannot be `eqx.tree_at`-ed. To A/B kinetic modes on one trained checkpoint, construct a fresh `CHLU` with the target statics and graft `(log_mass, potential_net)` (`cmf.make_arm`). Verified potential-preserving: `max |V_rel(q) − V_newt(q)| = 0.0` exactly. **Numerical trap:** Gauss–Hermite quadrature of relativistic velocity correlators **fails for `T/(m₀c²) ≫ 1`** (boundary layer `1/√τ`); use the analytic sign limit above `τ ≈ 10³` (symptom: non-monotone `D(T)`).

### For §8 (open directions) — scope call

> **F-10 is the strongest physics→ML claim in the program, and it is now *safe to state* only in its Cartesian form.** It is the one place where a relativistic term buys a guarantee that (i) holds against an adversary, (ii) needs no invariant measure — hence survives CM-17's Gibbs no-go untouched — and (iii) has a Newtonian control that provably has no analogue. **Recommend it as the physics-flagship (Nature-MI) headline and the ICLR-long's safety claim, with the angular-floor correction folded in up front** — stating the broken version and repairing it is *stronger* than stating the repaired one, and it pre-empts exactly the reviewer who draws the chord. **Two gates before it can carry a paper:** **R1** (bounded-energy adversary — is the angular floor recoverable under a realistic threat model?) and **R2** (train the relativistic arm — everything here is a governor *swapped in at inference* on a Newtonian-trained checkpoint, confound C2). **Do not** put F-11's aliasing protection in an abstract: it is γ-conditional (absent at γ≤0.005), geometry-dependent (16.8× not 48.9×), and on the trained ring it guards an unreachable threshold. F-11's defensible payoff is the **exact logarithmic companding law** (machine precision) plus the **125× transient containment**.
