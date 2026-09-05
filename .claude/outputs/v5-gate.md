# v5-gate — results-analyst report

**Task + acceptance criterion:** run the two cheap gates named in `t-lever-forgetting` §10 — **R1** (replicate the §4.2 sign flip on the *emergent* MLP checkpoints; kill or confirm confound C1) and **R3** (the free falsifiable: does a `γ_φ` friction hole act as a memory *vault* at T>0?) — and return the **V5 GO / V2-APPENDIX** call with evidence. Predictions registered before measuring. Repo read-only. Do **not** build `T_φ(q)`.

**Status: done.** Both gates executed on trained checkpoints. Predictions were pre-registered in `.claude/outputs/v5-gate/PREREG.md` **before** any harness ran.

---

## 0. Headline — **V5 GO**

> **R3 lands, harder than `t-lever` predicted.** A `γ_φ` friction hole is a **110× memory vault** — not the 13.9× §7 estimated — because the *shipped* `FrictionField` is **absorb-only**: it damps momentum but does **not** rescale the FDT noise. So a hole is simultaneously a **brake and a refrigerator**: `T_local = 1.259e-4` inside vs `1e-3` outside (measured `Var(p_i)/(M_iT) = 0.12600 ± 0.00031`, predicted `0.12591`), and `D_θ ∝ γ_eff^{-2}`, not `γ_eff^{-1}`. Measured vault `107.77 ± 4.78×` (D̂-based, 3 seeds) vs pre-registered `110.25×`. The coupled-bath hypothesis (`13.88×`) is **rejected by a factor 8.11 ± 0.37** against its own dedicated control (scalar `γ = 0.525`, no field, same `γ_eff`, measured vault `13.28 ± 0.12×`). At `T=0` the hole erases **nothing**: latch drift `≤ 1.75e-12` rad over 200k steps inside *and* outside, `||λ_flat|−1| ≤ 2.0e-15` at every γ **even at the hole edge where `|∇γ_φ| = 3.0`**.

> **R1 splits the CM-16 claim cleanly in two, and that split *is* the emergent-arm result.**
> - **The `T=0` face (CM-16 a / Cor-13) does NOT generalize.** On emergent (MLP) nets the coset is a pseudo-Goldstone: `1−|λ_coset| = 1.1e-3 … 3.0e-3` at γ≈0.05 (designed: `≤ 1.1e-15`) — **~12 orders of magnitude**. Any written coset value `δ ∈ {0.1, 0.3, 0.5}` rad relaxes **completely** back to the washboard minimum (`|retained| ≤ 2.1e-3`); the designed net retains `δ` exactly. **An emergent CHLU has no continuous coset register at all** — only 2–3 discrete washboard minima.
> - **The unification (the deepest CM-16 claim) DOES generalize, exactly.** The emergent `T=0` `n₁/₂(γ)` is the predicted **V-curve**: minimum at `0.902 ± 0.003 × γ_crit(=2εμ_soft)`, log-slope **−1.0020 ± 0.0003** below and **+1.116 ± 0.011** above, 3/3 seeds. The designed checkpoint is the `μ→0` corner of the *same* curve (`n₁/₂ = ∞` everywhere).
> - **The `T>0` face (CM-16 b,c) generalizes above a crossover `T*`.** `∂n₁/₂/∂γ > 0` in **10/10** emergent conditions. Bias-corrected `n₁/₂^obs/n₁/₂^Goldstone` (deep-diffusive cells): `2.75, 3.40` at `T=2e-3` → `0.94–1.25` at `T ≥ 4e-3`. **Measured `T* ≈ 3e-3`**, vs predicted `2.72e-3 … 3.66e-3`. Above `T*` the *emergent* coset obeys the *exact-Goldstone* diffusion law to ±25%.
> - **A confound the task's framing missed, now closed:** the task predicted the γ-slope "flips back to negative below `T*`". **It does not, and cannot,** for the CM-16 observable (a register written *at* the vacuum): the pseudo-Goldstone restoring force *protects* it, so both channels give `∂n₁/₂/∂γ > 0` above `γ_crit`. The negative slope exists — but for a register written *off* the vacuum, where it is exactly the V-curve's `γ < γ_crit` branch (slope −1.0020). **Two observables, two signs.**

> **Critically:** a matched designed-vs-emergent control on the *same* (γ,T) grid shows the raw exponents are **not** discriminating — designed gives `d log n₁/₂/d log T = −0.53, −0.60, −1.04` and `d log n₁/₂/d log γ = +0.78, +0.63, +0.55`, i.e. **the same shallow slopes as emergent**. Those deviations are CM-16(d)'s `ℓ_θ/Δ` boundary-layer bias, present in *both* families. Any emergent-vs-designed claim quoted from raw exponents would have been an artifact. This is the single most important methodological finding here.

---

## 1. Flag-provenance (mandatory, protocol §5)

| item | value |
|---|---|
| repo commit at start | `9bc2cf7`, **working tree dirty**: uncommitted `chlu/core/chlu_unit.py` (concurrent `experiment-engineer`) |
| repo commit at end | **`d6f8bac`** `[experiment-engineer] fix FDT noise inertia: effective_mass() -> effective_inertia()` — i.e. the §6 bug `t-lever-forgetting` reported was fixed **mid-session**; the fix was present in the working tree for **all** of my runs |
| invariance to that fix | **all runs use `common.retie(model)`**, under which `effective_mass() ≡ effective_inertia()` regardless of the fix. Verified numerically per checkpoint (`e0_geometry.json`): `max\|eff_mass − eff_inertia\| = 0.000e+00` (retied) on all 4 checkpoints. ⇒ every number below is invariant to `9bc2cf7` vs `d6f8bac`. |
| checkpoints (R3) | `.claude/scratch/v2-full-runs/runs/designed150_s{42,43,44}/models/exp_d_chlu.pkl` (`potential_type="so2_invariant"`, `dim=4`, `hidden=64`, 150 ep, `kinetic="newtonian_learned"`, `tie_channel_mass=True`) |
| checkpoints (R1) | `emergent150_s{42,43,44}` (`potential_type="mlp"`, otherwise identical) + `designed150_s44` as matched control |
| **langevin_noise** | **`"fdt"` everywhere.** Repo default is `"legacy"`, under which **none** of these laws hold (CM-16 mandatory flag) |
| **retie** | **on, all quantitative runs** (pytree-level `log_mass[0]=log_mass[1]=mean`; `H` bit-identical). No repo edit. |
| friction_field | `FrictionField(k=1, gamma_max=0.9, width=0.25, gate="compact", trainable=False)`; `init_strength = γ_φ`; `init_radius = 50.0` (uniform arm) or `1.0` (localized arm), centred on the ring point `Ring(0)` |
| temperature_field | none (not built — out of scope, per task) |
| tilt / spurion | `tilt_delta = 0`, `spurion_delta = 0` |
| dt (ε) | `0.05` throughout |
| Δ (coset tolerance) | **`0.5` rad** — every `n₁/₂` below is `Δ = 0.5`; **never quote `n₁/₂` without `Δ` and `ℓ_θ/Δ`** |
| γ grids | R3: scalar `γ = 0.05` (+ scalar control `0.525`); `γ_φ ∈ {0, 0.1, 0.2, 0.3, 0.5}`. R1 Jacobian: `geomspace(0.002, 0.5, 48)`. R1 T-sweep: `{0.01, 0.05, 0.2}` |
| T grids | R3: `T = 1e-3` (and `T = 0` for T5). R1: `{2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2}` |
| ensembles | V: 2048 walkers × 40 samples · D: 512 walkers × 10 blocks · T6: 256 walkers, cap 400k · E2: 96 walkers, cap 150k |
| precision | **float64** (`jax_enable_x64`); weights cast f32→f64; training was f32 |
| seeds | model seeds as listed; PRNG keys derived per (seed, T, γ, γ_φ) — formulas in each script |
| env | jax 0.9.0, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0, **main venv** `/Users/user/Desktop/CHLU/.venv` (no worktree sync — w6 lesson) |

### 1.1 Commands (all from `.claude/scratch/v5-gate/`, `PYTHONPATH=/Users/user/Desktop/CHLU`, `.venv/bin/python`)

```
e0_geometry.py                                                            # geometry + ring profile   (50 s)
r3_vault.py --seeds 42 --stages F V S --tag r3smoke                       # smoke
r3_vault.py --seeds 42 43 44 --stages F V D S T5 --tag r3main             # R3 core     (728 s)
r3_vault.py --seeds 42 43 44 --stages T6 --t6-gphi 0.5 --t6-max 400000 \
            --tag r3t6_g50                                                # R3 direct FPT (~17 min)
e1_jacobian.py                                                            # R1 T=0, superseded argmin (54 s)
e1b_coset.py                                                              # ⚠ WRONG — see C-9        (18 s)
e1c_vcurve.py                                                             # R1 T=0 V-curve  [FINAL]  (73 s)
e2_halflife.py --tags emergent150_s42 emergent150_s44 --gammas 0.01 0.05 0.2 \
   --temps 2e-3 4e-3 8e-3 1.6e-2 3.2e-2 --n-walk 96 --cap 150000 --tag e2emg   (864 s)
e2_halflife.py --tags designed150_s44 --gammas 0.01 0.05 0.2 \
   --temps 2e-3 4e-3 8e-3 --n-walk 96 --cap 60000 --tag e2des             # matched control (212 s)
figs.py
```

Artifacts → `.claude/outputs/v5-gate/` (4 PNG, 7 JSON, 4 NPZ, `PREREG.md`). Scripts → `.claude/scratch/v5-gate/`.

### 1.2 Harness cross-check
My FPT harness reproduces `t-lever-forgetting`'s matched cell (designed s42, γ=0.05, T=1e-3, Δ=0.5): **mine `n₁/₂ = 1690` (ratio to pred 1.361)** vs **theirs `1600` (ratio 1.288)** — a 5.6% difference, inside their quoted ~7% per-cell statistical error. Different PRNG keys, same 256 walkers.

---

## 2. R3 — the friction hole. **PRE-REGISTERED, then measured.**

### 2.1 The correction that made R3 sharp (registered in `PREREG.md` §R3.0, before measuring)

`t-lever` §7/§8.3-T6 predicts a **13.9×** vault from `n₁/₂ ∝ γ_eff/(2−γ_eff)`. **That assumes the hole is a locally thermalized bath.** The shipped code is explicitly not:

> `chlu/core/integrators.py`: *"the field friction is deliberately NOT coupled to the noise scale — a pure sink (absorb-only). Coupling it (localized bath / 'Hawking re-emission') is the S2 study hook."*

So `p ← (1−γ_φ)(1−γ)p + σ(γ)ξ` with `σ² = M T γ(2−γ)` built from the **scalar** γ. With `a = 1−γ_eff`, `γ_eff = 1−(1−γ)(1−γ_φ)`:

```
Var(p)_stat = σ²/(γ_eff(2−γ_eff))                       ⇒  T_local = T·γ(2−γ)/(γ_eff(2−γ_eff))
D_θ = (ε/2M²r*²)·Var(p)·(1+a)/(1−a) = ε T γ(2−γ)/(2 F² γ_eff²)      ← absorb-only  (D ∝ γ_eff⁻²)
D_θ = ε T (2−γ_eff)/(2 F² γ_eff)                                     ← coupled bath (D ∝ γ_eff⁻¹)
```

⇒ **`n₁/₂ ∝ γ_eff²`**, vault `= (γ_eff/γ)² = 110.25×` at `γ_φ=0.5`. The two hypotheses differ by **7.942×** in both `n₁/₂` and `Var(p)` — decisive, not a fit.

### 2.2 Field sanity (`r3main_results.json:F`)
Compact gate, hole radius 1.0, width 0.25, centred on the ring at `θ=0` (`r*=0.967`):

| θ (rad) | dist to centre | γ_φ | γ_eff |
|---|---|---|---|
| 0.00 | 0.0000 | **0.500000** | 0.525 |
| 0.50 | 0.4785 | **0.500000** | 0.525 |
| 0.75 | 0.7084 | **0.500000** | 0.525 |
| 0.936 | 0.8724 | 0.257724 | 0.294837 |
| 1.20 | 1.0920 | **0.000000** | 0.050 |
| π | 1.9340 | **0.000000** | 0.050 |

The whole `|Δθ| ≤ 0.5` exit region sits at `γ_φ = 0.5` **exactly**; the outside arm (`θ₀=π`) at `γ_φ = 0` **exactly**. No sigmoid tail leakage (the `gamma-field-build` S1 gap is closed by `gate="compact"`).

### 2.3 The refrigerator (`stage V`; 3 seeds × 5 γ_φ; 2048 walkers × 40 samples)

`Var(p_i)/(M_i T)` — **the coupled hypothesis predicts exactly 1.0 at every γ_φ.**

| γ_φ | γ_eff | measured (3 seeds) | absorb pred | obs/pred | coupled pred | obs/pred |
|---|---|---|---|---|---|---|
| 0.0 | 0.050 | 0.99856 ± 0.00178 | 1.00000 | 0.9986 | 1.0 | 0.9986 |
| 0.1 | 0.145 | 0.36171 ± 0.00129 | 0.36249 | 0.9978 | 1.0 | **0.3617** |
| 0.2 | 0.240 | 0.23044 ± 0.00072 | 0.23082 | 0.9983 | 1.0 | **0.2304** |
| 0.3 | 0.335 | 0.17513 ± 0.00071 | 0.17480 | 1.0019 | 1.0 | **0.1751** |
| 0.5 | 0.525 | **0.12600 ± 0.00031** | **0.12591** | **1.0007** | 1.0 | **0.1260** |

**`T_local = 1.2600e-4` inside the hole vs `1e-3` outside — a 7.94× refrigerator.** Absorb-only confirmed to 0.2%; coupled bath rejected by 7.9×.

### 2.4 The diffusion law (`stage D`; block increments, 512 walkers × 10 blocks)

`MSD_obs / MSD_pred` (the `MSD` predictor is the exact finite-`L` AR(1) partial-sum variance, generalized to `a=1−γ_eff`, `σ²=MTγ(2−γ)`):

| γ_φ | / absorb pred | / coupled pred |
|---|---|---|
| 0.0 | 1.0144 ± 0.0177 | 1.0144 ± 0.0177 |
| 0.1 | 0.9866 ± 0.0042 | 0.3576 ± 0.0015 |
| 0.2 | 1.0104 ± 0.0054 | 0.2332 ± 0.0013 |
| 0.3 | 0.9928 ± 0.0140 | 0.1735 ± 0.0024 |
| 0.5 | 1.0013 ± 0.0349 | 0.1261 ± 0.0044 |
| **all 15 cells** | **1.0011 ± 0.0215** (min 0.9713, max 1.0502) | — |

**D̂-based vault `D_θ(out)/D_θ(in)`:**

| γ_φ | measured (3 seeds) | pred absorb `(γ_eff/γ)²` | pred coupled |
|---|---|---|---|
| 0.1 | 8.42 ± 0.12× | 8.41× | 3.05× |
| 0.2 | 22.39 ± 0.37× | 23.04× | 5.32× |
| 0.3 | 44.31 ± 1.19× | 44.89× | 7.85× |
| **0.5** | **107.77 ± 4.78×** | **110.25×** | 13.88× |

### 2.5 The discriminator: same `γ_eff`, no field (`stage S`)

| seed | `D̂(γ=0.525)` | pred | ratio | vault vs γ=0.05 | `Var(p)/(M T)` |
|---|---|---|---|---|---|
| 42 | 1.1115e-4 | 1.0986e-4 | 1.0142 | 13.21× | 1.0328 |
| 43 | 1.0977e-4 | 1.0992e-4 | 1.0011 | 13.45× | 1.0466 |
| 44 | 1.0824e-4 | 1.1009e-4 | 0.9856 | 13.19× | 0.9880 |

Scalar friction at the **same** `γ_eff = 0.525` gives a **13.28 ± 0.12×** vault (pred 13.88) and leaves `Var(p) = M_i T` intact. The **field** gives **107.77 ± 4.78×**. Ratio **8.11 ± 0.37** vs predicted **7.942**.

> **A position-gated friction field is not "locally stronger friction". It is locally stronger friction *plus* a local refrigerator.** That is a physical statement about the shipped operator, and it is what makes the hole a 110× vault instead of a 14× one.

### 2.6 T5 — `T=0`: the hole erases nothing (`stage T5`, 3 seeds)

| arm | field | latch drift / 200k steps | `θ_settle` | `\|p_f\|` |
|---|---|---|---|---|
| inside | γ_φ=0.5 | **8.3e-14 … 1.5e-13 rad** | +0.014400 / +0.014400 / +0.014624 | ≤ 4.9e-16 |
| inside | none | 5.8e-13 … 9.5e-13 | +0.150387 / +0.150378 / +0.152796 | ≤ 3.7e-16 |
| outside | γ_φ=0.5 | 4.0e-13 … 1.75e-12 | +0.150387 / +0.150378 / +0.152796 | ≤ 4.0e-16 |
| outside | none | 4.0e-13 … 1.75e-12 | (identical to above) | ≤ 4.0e-16 |

**Exact operator statement.** With the field ON, over 27 (seed × point × γ) cells, `γ ∈ {0.002, 0.05, 0.5}`, points = {hole centre, **hole edge**, outside}:

> `||λ|_max − 1| ≤ 2.0e-15` — including **at the hole edge, where `|∇γ_φ| = 3.0`.**

(The `∇γ_φ` term in the Jacobian multiplies `p* = 0` at the latched vacuum, so it drops out identically — as registered.) ⇒ **0% erasure, at any γ_φ, anywhere.**

**New sub-finding (not in `t-lever`): the hole attenuates the WRITE, not the stored value.** `θ_settle(inside, field)/θ_settle(outside)` = 0.09576 / 0.09576 / 0.09572, vs the transport-law prediction `γ/γ_eff = 0.095238` → **ratio 1.0054 ± 0.0002**. A `γ_φ` hole is a **write attenuator**: it divides the angle a given charge impulse writes by `γ_eff/γ`, while leaving already-latched content bit-frozen. That is a *second*, independent reason the `γ_φ` rung of the fit-gap ladder recovered −24%.

### 2.7 T6 — `T=1e-3`: the direct FPT vault (`r3t6_g50_results.json`, 256 walkers, cap 400k)

| seed | outside `n₁/₂` (cens) | inside `n₁/₂` (cens) | raw vault | inside obs/pred(absorb) | inside obs/pred(coupled) |
|---|---|---|---|---|---|
| 42 | 1690 (0.000) | 141880 (0.082) | 83.95× | 1.036 | 8.23 |
| 43 | 1715 (0.000) | 147480 (0.105) | 85.99× | 1.078 | 8.23 |
| 44 | 1560 (0.000) | 141880 (0.113) | 90.95× | 1.038 | 8.25 |
| **mean** | | | **86.97 ± 2.94×** | **1.051 ± 0.019** | **8.24** |

**Why raw FPT gives 87× where D̂ gives 108×:** CM-16(d). The *outside* arm sits at `ℓ_θ/Δ = 0.0791` — the boundary-layer regime, where `n₁/₂` is inflated (obs/pred = 1.361, 1.382, 1.259; `t-lever`'s own toy predicts `1 + 3.099·ℓ_θ/Δ = 1.245`, and their matched cell measured 1.288). The *inside* arm sits at `ℓ_θ/Δ = 0.0027` — essentially unbiased, and it lands on the absorb prediction at **1.051 ± 0.019**. **`D̂` is the unbiased instrument; the 107.77 ± 4.78× number is the one to quote.** The raw 87× is the honest number for a figure whose y-axis is a measured first-passage time.

**R3 verdict: LANDED, and the pre-registered `t-lever` §7 number was wrong by 7.94× in the direction that makes the result stronger.**

---

## 3. R1 — the emergent (MLP) generalization gate

### 3.1 Geometry: an emergent "flat direction" is not flat, and is barely soft (`e0_geometry.json`, `e1c_vcurve.json`)

| checkpoint | Hessian `μ²` (ascending) | `μ²_soft` | angular overlap of the soft mode | coset `μ²_stiff` / `μ²_adiab` | **stiffest/coset ratio** | washboard minima | ring ripple (barrier) |
|---|---|---|---|---|---|---|---|
| emergent s42 | 0.0545, 0.0824, 0.1208, 0.1611 | 5.449e-2 | 0.697 | 0.0784 / 0.0704 | **2.29** | 2 | 2.294e-2 |
| emergent s43 | 0.0203, 0.0690, 0.1040, 0.1367 | 2.029e-2 | 0.816 | 0.0487 / 0.0277 | **4.94** | 2 | 6.813e-3 |
| emergent s44 | 0.0513, 0.0660, 0.1067, 0.1546 | 5.132e-2 | 0.886 | 0.0976 / 0.0893 | **1.73** | 3 | 3.571e-2 |
| **designed s44** | **−2.9e-16**, 0.1137, 0.1848, 1.1901 | −2.9e-16 | 1.000 | −2.9e-16 | **∞** | 122 (numerically flat) | **1.94e-16** |

`μ²_stiff = (aᵀHa)/M_ch`, `μ²_adiab = 1/((H⁻¹)_aa M_ch)` at the settled vacuum, `a = Xq*/|Xq*|`. The ring-profile machinery validates on designed (ripple 1.9e-16, `μ²_ring = 1.7e-12`).

> The emergent coset direction is only **1.7–4.9× softer than the *stiffest* direction in the whole spectrum.** It is not a near-Goldstone; it is a middle-of-the-spectrum massive mode.

### 3.2 Cor-13 (CM-16 a) **FAILS on emergent** (`e1c_vcurve.json`, Jacobian at `(q*,0)`, γ grid `geomspace(0.002,0.5,48)`)

| checkpoint | `1−\|λ_coset\|` at γ≈0.048 | `min(1−\|λ_coset\|)` over the whole grid |
|---|---|---|
| emergent s42 | **2.963e-3** | 2.044e-4 |
| emergent s43 | **1.060e-3** | 7.609e-5 |
| emergent s44 | **2.780e-3** | 1.925e-4 |
| **designed s44** | **−6.7e-16** (i.e. 1.0 exactly) | **0.000e+00** |

CM-16(a) reports `||λ_flat|−1| ≤ 1.7e-14` on designed. On emergent the deficit is `~1e-3` — **~12 orders of magnitude larger.** `n₁/₂(T=0, γ=0.05) = 233.6 / 653.3 / 249.0` steps (Jacobian; rollout-measured 190 / 370 / 150).

### 3.3 The register itself is gone (`e1_jacobian.json:decay`; Fig 3)

Write a coset value `δ`, release from rest, `T=0`, γ=0.05, 20k steps. Retained value `|θ(N)|`:

| written δ | emergent s42 | s43 | s44 | designed s44 |
|---|---|---|---|---|
| 0.1 rad | 0.0000 | 0.0000 | 0.0000 | **0.1000** |
| 0.3 rad | 0.0000 | 0.0000 | 0.0000 | **0.3000** |
| 0.5 rad | 0.0000 | 0.0000 | 0.0000 | **0.5000** |

(Emergent max retained over all γ ∈ {0.005, 0.0266, 0.05, 0.2}: `2.1e-3`.)

> **An emergent CHLU stores nothing on its coset.** Its "register" has capacity `log₂(2–3 minima) ≈ 1–1.6 bits`, not a continuum. The designed net stores an arbitrary real value indefinitely.

### 3.4 …but the CM-16 **unification** generalizes exactly (Fig 2a)

`n₁/₂(γ) = ln2/gap`, coset-tracked eigenvalue (`λ_ret = max{|λ_j| : coset overlap ≥ 0.30}`):

| checkpoint | `γ_crit = 2εμ_soft` | measured `argmin_γ n₁/₂` | ratio | slope below min | slope above min |
|---|---|---|---|---|---|
| emergent s42 | 0.02334 | 0.02100 | 0.8994 | **−1.0022** | **+1.1236** |
| emergent s43 | 0.01424 | 0.01288 | 0.9046 | **−1.0016** | **+1.1010** |
| emergent s44 | 0.02265 | 0.02051 | 0.9055 | **−1.0022** | **+1.1226** |
| designed s44 | 0 | — (`n₁/₂ = ∞` ∀γ) | — | — | — |

`argmin/γ_crit = 0.902 ± 0.003` (3/3). The 10% deficit is the known discrete-map correction (`t-lever` §4.3 measured 0.83–0.93 on the massive mode). Predicted `n₁/₂` from `μ²_soft` reproduces the measured Jacobian `n₁/₂` **to printed precision at every γ on the grid** (see `e1c_vcurve.json`).

> **One damping-optimum curve, two checkpoints, eleven orders of magnitude in `μ²` (1.7e-12 → 7e-2).** The designed coset *is* the `μ→0` corner. CM-16's unification is now an emergent-arm result, not a designed-only one.

### 3.5 The `T>0` face: sign flip replicates — but the raw exponents are an artifact

`n₁/₂` = median FPT to `|Δθ| ≥ 0.5`, walkers started at `q*`, FDT, 96 walkers, cap 150k (60k designed). **0/45 cells censored.**

**Raw exponents (`e2emg_halflife.json`, `e2des_halflife.json`):**

| | `d log n₁/₂ / d log T` (γ=0.01 / 0.05 / 0.2) | `d log n₁/₂ / d log γ` (T=2e-3 / 4e-3 / 8e-3) |
|---|---|---|
| emergent s42 | −0.910 / −1.104 / −1.333 | +0.770 / +0.608 / +0.602 |
| emergent s44 | −1.073 / −1.161 / −1.327 | +0.574 / +0.592 / +0.580 |
| **designed s44 (matched control)** | **−0.534 / −0.599 / −1.044** | **+0.777 / +0.633 / +0.546** |

`∂n₁/₂/∂γ > 0` in **10/10** emergent (seed × T) conditions — **the sign flip replicates.** But the *magnitudes* (+0.35…+0.77, well below CM-16's designed +0.955 ± 0.042) are **not** an emergent effect: the designed control on the same grid gives the same shallow values. Both families are contaminated by CM-16(d)'s `ℓ_θ/Δ` boundary-layer bias, which is large here because this grid reaches `T = 3.2e-2` and `γ = 0.01` (`ℓ_θ/Δ` up to **2.03**). CM-16's `+0.955 ± 0.042` was measured at `T ≤ 1e-2` with `γ ∈ {0.05,0.1,0.2}`.

**Bias-corrected, deep-diffusive cells only (`ℓ_θ/Δ < 0.06`, i.e. γ=0.2), dividing by the `t-lever` toy bias `1 + 3.099 ℓ_θ/Δ`:** (Fig 4b)

| | `T` | `σ_θ/Δ = √(T/(F²μ²))/Δ` | bias-corrected `n₁/₂^obs / n₁/₂^Goldstone` |
|---|---|---|---|
| designed s44 | 2e-3, 4e-3, 8e-3 | ∞ (μ²=0) | **1.045 ± 0.095** (1.172, 0.942, 1.022) |
| emergent s42 | 2e-3 | 0.39 | **2.746** |
| emergent s44 | 2e-3 | 0.45 | **3.395** |
| emergent s42 | 4e-3 | 0.55 | 0.995 |
| emergent s44 | 4e-3 | 0.63 | 1.250 |
| emergent s42 | 8e-3 | 0.77 | 0.944 |
| emergent s44 | 8e-3 | 0.90 | 1.075 |

**The crossover `T*` is measured, and it is where predicted.**

| checkpoint | `T*` predicted (relaxation = diffusion, γ=0.2) | `T*` predicted (`σ_θ = Δ/2`) | `T*` **measured** |
|---|---|---|---|
| emergent s42 | 3.659e-3 | 3.339e-3 | between 2e-3 and 4e-3 |
| emergent s44 | 2.724e-3 | 2.486e-3 | between 2e-3 and 4e-3 |

> Below `T*` the pseudo-Goldstone mass **protects** the register (2.7–3.4× longer than the exact-Goldstone law). Above `T*` the emergent coset obeys the **exact-Goldstone diffusion law to ±25%** — the mass is thermally invisible. **`T* ≈ 3e-3`, comfortably below the washboard barrier (2.29e-2, 3.57e-2).**

### 3.6 The confound the task's prediction contained (falsified, cleanly)

The task predicted: *"at small T the `∂n₁/₂/∂γ` sign flips **back to negative** below a crossover `T*`."* **This is false, and it is false for a structural reason.** Two different registers:

| register | destroyed by | `∂n₁/₂/∂γ` | measured |
|---|---|---|---|
| written **at** the washboard minimum (the CM-16 observable) | thermal diffusion out to `Δ` | **> 0** (both channels; relaxation *protects*) | +0.35 … +0.77, **10/10** |
| written **off** the minimum (an arbitrary value) | `T=0` relaxation home | **< 0 for γ < γ_crit**, > 0 above (the V-curve) | **−1.0020 ± 0.0003**, 3/3 |

Both signs exist on the emergent checkpoint. They belong to different observables. A "sign flip vs γ" figure that does not name the initial condition is meaningless.

**R1 verdict: C1 is CLOSED, with a split answer.** The T=0 structural claim is designed-only; the unification and the T>0 law generalize. That split is a *better* emergent-arm result than a flat confirmation would have been.

---

## 4. The V5 call

**Pre-registered decision rule** (`PREREG.md`, written before measuring): *GO iff R3 lands (vault ≥ 10×, T5 drift < 1e-11) **and** R1 exhibits a measurable `T*` with a diffusive branch above it (`∂n₁/₂/∂T ≤ −0.5` at some `T < barrier`, with `∂n₁/₂/∂γ > 0` there).*

| condition | result |
|---|---|
| R3 vault ≥ 10× | ✅ **107.77 ± 4.78×** (D̂), 86.97 ± 2.94× (raw FPT) |
| R3 T5 drift < 1e-11 rad | ✅ **≤ 1.75e-12 rad** / 200k steps, 12/12 cells |
| R1 measurable `T*` below the barrier | ✅ **`T* ≈ 3e-3`** vs barrier 2.29e-2 / 3.57e-2 |
| R1 `∂n₁/₂/∂T ≤ −0.5` there | ✅ γ=0.2, `T ∈ [4e-3, 8e-3]`: **−1.02** |
| R1 `∂n₁/₂/∂γ > 0` there | ✅ **10/10** conditions |

# ⇒ **V5 GO.**

R1 gives an emergent-arm result (three, in fact: Cor-13's failure, the unification's survival, the `T*` crossover). R3 lands the vault, 8× harder than `t-lever` predicted, with a mechanism (`the hole is a refrigerator`) that nobody had.

### 4.1 The one figure the short is built around

**`fig1_friction_hole_vault.png`, panels (b)+(c)** — *"Same `γ_eff`, 8× different `D_θ`"* + *"the hole is a 110× memory vault; at `T=0` it erases nothing."* Once R4 lands, add the `T_φ` panel and the figure becomes the 2×2:

| | `T = 0` | `T > 0` |
|---|---|---|
| **`γ_φ` hole** | 0% erasure (`\|λ\|=1` to 2e-15) | **110× vault** (brake + refrigerator) |
| **`T_φ` hole** | *(R4)* erasure ∝ `T_φ/(γ_c F²)` | *(R4)* shredder |

That is the short: **"friction preserves, temperature shreds — and the operator you reach for first does the opposite of what you want."**

### 4.2 Is the `T_φ(q)` spec (`t-lever` §8) still exactly right? **Three corrections required.**

1. **§8.3 T6's "≈13.9× SLOWER" is wrong for the shipped code.** Replace with **`(γ_eff/γ)² = 110.25×`** (measured `107.77 ± 4.78×` by D̂; `86.97 ± 2.94×` by raw FPT at `ℓ_θ/Δ = 0.079`). The 13.9× number is what a *coupled-bath* field would give, and it is now the measured value of the **scalar-γ control** (13.28 ± 0.12×).
2. **§8.1's `D_theta(q)` / `n_1/2` formulas are correct only under `temperature_field_couple_gamma_phi=True`.** The spec must carry **both** branches, and **T2 must be split** into T2-coupled (`Var(p_i) = M_i(T+c)`) and T2-absorb (`Var(p_i) = M_i(T+c)·γ(2−γ)/(γ_eff(2−γ_eff))`). Otherwise the acceptance test will "fail" against correct code.
3. **§8.1's design rule "do NOT co-locate `T_φ` with `γ_φ`" is understated by 7.94×.** Under the shipped absorb-only path, co-location suppresses the local temperature *and* raises the damping: erasure is diluted by `(γ_eff/γ)²`, not `γ_eff/(2−γ_eff)`. With `γ_φ = 0.5`, a co-located `T_φ` hot spot is **110× weaker**, not 14× weaker. **The rule should be stated as a hard constraint, not a caution.**
4. **§8.2 item 6 (blocking prerequisite) is DISCHARGED:** the §6 FDT bug is fixed and committed as **`d6f8bac`**. I re-verified on all four checkpoints that `effective_mass() ≡ effective_inertia()`.

Everything else in §8 (the `TemperatureField` module shape, `gate="compact"`, `T_φ ≥ 0` strictly, two-timescale `temperature_field_lr`, the contrastive wake/sleep training scheme, `getattr` guards) stands unchanged and is **confirmed by R3's mechanics** (the compact gate gave γ_φ = 0.5 and 0.0 *exactly*, with zero tail leakage, at the arc positions that matter).

---

## 5. Limitations & confounds (honest; C-9 negatives written)

- **L1 — `t-lever` C1 is closed only for `dim=4`, `hidden=64`, 150-epoch, `newtonian_learned` MLP checkpoints.** Three emergent seeds. No high-dim, no `deep_mlp`, no relativistic kinetic mode.
- **L2 — `emergent150_s43` is geometrically atypical** (`μ²_soft = 0.0203`, angular overlap 0.816, barrier 6.8e-3 — 3–5× shallower than s42/s44). It was used for the Jacobian V-curve (where it agrees perfectly) but **excluded from the T-sweep**; its `T*` (9.4e-4) is close to the lowest T I probed.
- **L3 — the raw `T>0` exponents on this grid are estimator-dominated** and must never be quoted alone. `ℓ_θ/Δ` reaches **2.03** at (γ=0.01, T=3.2e-2). Only the bias-corrected, `ℓ_θ/Δ < 0.06` cells (γ=0.2) support a physics claim. The designed matched control is what proves this.
- **L4 — `T ≥ 8e-3` is deeply anharmonic on emergent** (`σ_θ` comparable to the inter-minimum spacing; the measured `σ_θ` blows up to 1.5–9 rad, i.e. walkers circulate). CM-16's C6 already restricts trustworthy cells to `T ≤ 3e-3`. My `T*` determination straddles `T ∈ [2e-3, 4e-3]`, which is *inside* the trustworthy window — but it rests on **2 cells per seed**, not a fitted crossover. **A finer T grid (`T ∈ [1.5e-3, 6e-3]`, 6 points) would turn "`T* ≈ 3e-3`" into a fitted number with error bars.** Recommended (R1b below).
- **L5 — the T6 raw vault (87×) is not the law's number (110×).** The outside arm is boundary-layer-biased; I did not re-run it with a larger Δ or a colder outside. The D̂-based vault (107.77 ± 4.78×) is the unbiased one, and the inside arm independently confirms the absolute law (obs/pred = 1.051 ± 0.019).
- **L6 — the T6 inside arm has 8–11% right-censoring** at cap 400k. The median is unbiased while censoring < 0.5 (order statistic), so this is fine, but the mean FPT is not reported for those cells.
- **L7 — R3 used only `designed150` checkpoints.** The vault law is a property of the *integrator + field*, not of the potential, so it should hold identically on emergent — **but I did not measure it there.** Cheap follow-up.
- **L8 — no `T_φ` was built**, per the task. The "shredder" half of the 2×2 is still a prediction.
- **L9 — single PRNG stream per cell.** Seed variance is across *model* seeds (3), not across noise realizations. The D̂ estimator's own rel. SEM is ≈2%, consistent with the observed spreads.

### C-9 negatives / superseded artifacts — **do not use**
- **`e1b_coset.py` / `e1b_coset.json` — WRONG.** Selecting the coset eigenpair by `argmax(coset overlap)` picks arbitrarily between the **slow and fast branches of an overdamped pair** (they share the same q-direction). It reported `argmin_γ n₁/₂` = 0.42–0.50 (nonsense) and `|λ|` as low as 0.698. Superseded by **`e1c_vcurve.py`**, which takes `max|λ|` *among* coset-overlapping pairs. Retained on disk as a documented negative.
- **`e1_jacobian.json:argmin_gamma_jac`** tracks `max|λ|` over **all** modes. It happens to coincide with the softest mode's `γ_crit` (ratio 0.80–0.91) but that is a coincidence of this spectrum. Use `e1c`.
- **`e1_jacobian.json:decay[*].n_half_decay` and `e1c:n_half_rollout` at `γ < γ_crit`** are **first-crossing artifacts**, not envelope decay: an underdamped write crosses `δ/2` within a quarter-period (80–110 steps at γ=0.002), independent of the true gap (692 steps). This is exactly the F5 Appendix-N artifact `goldstone_harness` warns about. **The Jacobian `|λ|` is the ground truth**; the rollout is only quoted for `γ > γ_crit`.
- **`e0_geometry.json` `mu_sq_ring` / `F_sq_ring` / `ring_theta_min` for `emergent150_s43` are unreliable.** The BFGS ring-profile continuation slid into a *different basin* (`r: 0.989 → 0.548`), so its `F²_ring = 0.209` is not the vacuum's `F² = 0.681`. The local curvatures (`μ²_stiff`, `μ²_adiab` in `e1c_vcurve.json`) are basin-correct and are what §3 uses. s42/s44 ring profiles are fine (`r_min` matches `r*` to 3e-4).
- **`e2*_halflife.json:n_mean_pred_ou`** compares an analytic **mean** FPT to an observed **median**. Not a valid comparison; the column is diagnostic only. Ignore it.
- **`e2*_halflife.json:sigma_theta_obs`** is the std of the *accumulated* coset angle over early chunks — contaminated by drift and by walkers that have already escaped. Use the analytic `σ_θ = √(T/(F²μ²))`. (This is why `sigma_theta_obs` reaches 8.9 rad.)

---

## 6. Recommended next experiments

| id | experiment | cost | why |
|---|---|---|---|
| **R4** | **Build `T_φ(q)` per `t-lever` §8 with my three corrections (§4.2 above)**; run T1–T4 + the co-location test (`T_φ` inside a `γ_φ` hole ⇒ erasure diluted 110×). | medium | **the V5 mechanism**; now funded |
| **R1b** | Fit `T*` properly: emergent s42/s44, **γ = 0.2 only**, `T ∈ geomspace(1.5e-3, 6e-3, 6)`, 256 walkers. Turns "`T* ≈ 3e-3`" into `T* = _ ± _`, testable against `T* = 0.378748 Δ² F² μ² /((2−γ) ln2)`. | **small** (≈15 min) | the emergent arm's headline falsifiable currently rests on 2 cells/seed (L4) |
| **R3b** | Repeat R3's `V`/`D` stages on **`emergent150_*`** (the vault law should be potential-independent). | small | closes L7; makes the vault figure architecture-agnostic |
| **R2** | (unchanged from `t-lever`) `F²` scaling: retrain designed SO(2) at 3–4 confinements α to spread `r*` by ≥3×; test `n₁/₂ ∝ F²`. | medium | closes CM-16's C4 — still the only untested factor in the F-5 law |
| **R5** | Arrhenius channel on `tilt_delta > 0` checkpoints. **Now more attractive:** the emergent washboard (barrier 6.8e-3 … 3.6e-2, 2–3 minima) is a *naturally occurring* tilt. Measure `n_hop ∝ e^{2δr*/T}` on emergent directly, no retraining. | small | third leg of F-5, currently toy-only; and it quantifies the emergent register's true (discrete) capacity |
| **R7** | **Capacity claim:** measure the emergent coset register's bit capacity directly (`log₂` of distinguishable washboard minima × retention). Fig 3 says 1–1.6 bits vs designed = continuum. | small | this is the sharpest "why designed symmetry matters" statement in the programme |

---

## Git footprint

**None.** No tracked file created, modified, or deleted. `git status --short` clean at the end. Repo read-only as required.

- Two commits landed **from another agent** during my session: `9bc2cf7 → 64af0e7?no → d6f8bac` (HEAD moved `9bc2cf7 → d6f8bac`). The uncommitted `chlu_unit.py` edit I found at start was `experiment-engineer`'s `effective_mass() → effective_inertia()` fix; it was committed as `d6f8bac` mid-session. **I did not touch it.** All my runs used `retie()`, which makes results invariant to that change (verified: `max|eff_mass − eff_inertia| = 0` on all 4 checkpoints).
- One transient stray file: `e2emg.log` was briefly written to the repo root by a mis-`cd`'d command (untracked, not gitignored); **deleted immediately**, `git status` clean. Flagging it since it touched the shared checkout.
- Scripts → `.claude/scratch/v5-gate/{ecommon,e0_geometry,e1_jacobian,e1b_coset,e1c_vcurve,e2_halflife,r3_vault,figs}.py` (reusing `.claude/scratch/t-lever-forgetting/{common,kernels}.py` verbatim, via `sys.path`).
- Results → `.claude/outputs/v5-gate/`: `PREREG.md`; `fig1_friction_hole_vault.png`, `fig2_emergent_vcurve.png`, `fig3_register_capacity.png`, `fig4_emergent_Tstar.png`; `e0_geometry.json`, `e1_jacobian.json`, `e1b_coset.json` (⚠ superseded), `e1c_vcurve.json`, `e2emg_halflife.json`, `e2des_halflife.json`, `r3main_results.json`, `r3t6_g50_results.json`, `r3_prereg_numbers.json`; `e0_ring_*.npz` (4).

### For `experiment-engineer`
1. **Not a bug, but undocumented physics.** `langevin_step`'s absorb-only `gamma_field` means a friction hole **locally cools** the bath: `T_local = T·γ(2−γ)/(γ_eff(2−γ_eff))` and `D_θ ∝ γ_eff⁻²`. The docstring calls it "a pure sink"; it is a sink **and a refrigerator**. Measured to 0.2% (§2.3–2.4). Please document — this is load-bearing for the `T_φ` spec.
2. **Real ergonomics bug.** Checkpoints saved before `friction_field` was added to `CHLU` unpickle **without the attribute**, so `eqx.tree_at(lambda m: m.friction_field, ...)` raises `AttributeError: 'CHLU' object has no attribute 'friction_field'`. Reads are guarded (`getattr`), writes are not. I worked around it with `object.__setattr__` on a `copy.copy`. Suggest a supported `chlu.core.chlu_unit.with_friction_field(model, field)` helper (or backfill the attribute on load in `chlu/utils/checkpoints.py`). Affects anyone attaching a field to an existing checkpoint — i.e. all of R4.
3. §6 FDT fix confirmed good at `d6f8bac`: `effective_mass() ≡ effective_inertia()` on `designed150_s{42,43,44}` and `emergent150_s{42,43,44}`.

---

## Proposed handover updates (for the Hub)

### §1.6 / §1.9 "Forgetting" — **CM-16 must be split into CM-16a (designed-only) and CM-16b (universal)**

> **CM-16 REFINED by `v5-gate` (2026-07-09/10; `d6f8bac`; `langevin_noise="fdt"`; retie; ε=0.05; Δ=0.5; float64).**
> - **(a) Cor-13 is DESIGNED-ONLY.** On `emergent150_s{42,43,44}` (`potential_type="mlp"`) the coset is a pseudo-Goldstone: `1−|λ_coset| = 1.06e-3 … 2.96e-3` at γ≈0.048 (designed: `≤ 1.1e-15`). **~12 orders of magnitude.** A written coset value `δ ∈ {0.1,0.3,0.5}` rad relaxes **completely** home (`|retained| ≤ 2.1e-3`); the designed net retains `δ` exactly. **An emergent CHLU has no continuous coset register — only 2–3 discrete washboard minima (ripple 6.8e-3 … 3.6e-2), i.e. ~1–1.6 bits.** The emergent coset is only **1.7–4.9× softer than the stiffest Hessian mode** (`μ²_soft` = 2.0e-2 … 5.4e-2). *"CHLU's flat directions" do not emerge; they must be designed in.*
> - **(c-unification) GENERALIZES EXACTLY.** Emergent `T=0` `n₁/₂(γ)` is the V-curve: `argmin_γ / (2εμ_soft) = 0.902 ± 0.003` (3/3 seeds), slope **−1.0020 ± 0.0003** below, **+1.116 ± 0.011** above. One damping-optimum curve spanning `μ² ∈ [1.7e-12, 7e-2]`. **This is now the strongest, most portable claim in CM-16.**
> - **(b,c) GENERALIZE above a crossover `T*`.** `∂n₁/₂/∂γ > 0` in 10/10 emergent conditions. Bias-corrected `n₁/₂^obs/n₁/₂^Goldstone` (γ=0.2, `ℓ_θ/Δ<0.06`): **2.75, 3.40** at `T=2e-3` (`σ_θ/Δ = 0.39, 0.45`) → **0.94–1.25** at `T ≥ 4e-3`. **`T* ≈ 3e-3`**, vs predicted `T* = 0.378748 Δ²F²μ²/((2−γ)ln2)` = 3.66e-3 (s42) / 2.72e-3 (s44). Above `T*` the emergent coset obeys the exact-Goldstone law to ±25%.
> - **⚠ METHOD WARNING (fold into C3/(d)):** on a matched designed control run at the same (γ,T) grid, the raw exponents are `d log n₁/₂/d log T = −0.53, −0.60, −1.04` and `d log n₁/₂/d log γ = +0.78, +0.63, +0.55` — **indistinguishable from emergent.** The shallow slopes are the `ℓ_θ/Δ` boundary layer, not physics. **No emergent-vs-designed claim may be quoted from raw exponents.** `ℓ_θ/Δ` reaches 2.03 on that grid.
> - **The "sign flips back to negative below `T*`" conjecture is FALSE** and structurally cannot hold: for a register written **at** the vacuum (the CM-16 observable) the pseudo-Goldstone force *protects* it, so `∂n₁/₂/∂γ > 0` in both channels. The negative slope belongs to a register written **off** the vacuum — where it is exactly the V-curve's `γ < γ_crit` branch (−1.0020). **Two observables, two signs. Always name the initial condition.**

### §1.6 / §5 — **new CM entry: the friction hole is a vault AND a refrigerator (R3 landed)**

> **CM-17 (new; `v5-gate`, 3 designed seeds, pre-registered):** `t-lever` §7's "`γ_φ` hole ⇒ 13.9× vault" is **wrong for the shipped code, by 7.94×.** `chlu/core/integrators.py` applies the field **absorb-only** (`p ← (1−γ_φ)(1−γ)p + σ(γ)ξ`, noise built from the **scalar** γ). Hence
> `Var(p) = M T γ(2−γ)/(γ_eff(2−γ_eff))`, `T_local = T·γ(2−γ)/(γ_eff(2−γ_eff))`, **`D_θ = εTγ(2−γ)/(2F²γ_eff²)`**, so **`n₁/₂ ∝ γ_eff²`** and the vault is **`(γ_eff/γ)²`**.
> Measured (γ=0.05, γ_φ=0.5 ⇒ γ_eff=0.525, T=1e-3, Δ=0.5):
> - `Var(p_i)/(M_iT) = 0.12600 ± 0.00031` vs absorb `0.12591`, vs coupled `1.0` → **the hole is a 7.94× refrigerator**, `T_local = 1.26e-4`.
> - `MSD/pred_absorb = 1.0011 ± 0.0215` over 15 cells (5 γ_φ × 3 seeds); `/pred_coupled = 0.1261 ± 0.0044` at γ_φ=0.5.
> - **Vault (D̂-based) = 107.77 ± 4.78×** vs pre-registered **110.25×**. Vault by direct FPT = 86.97 ± 2.94× (outside arm biased, `ℓ_θ/Δ=0.079`); the **inside** arm hits the absolute law at **obs/pred = 1.051 ± 0.019**.
> - **Discriminator:** scalar `γ = 0.525` (same `γ_eff`, no field) gives only **13.28 ± 0.12×** and `Var(p) = M_iT` intact. Field/scalar = **8.11 ± 0.37** vs predicted 7.942. *A friction field is not "locally stronger friction".*
> - **T5 (T=0): 0% erasure.** Latch drift `≤ 1.75e-12` rad / 200k steps, inside **and** outside, 12/12 cells; `||λ|_max−1| ≤ 2.0e-15` over 27 cells **including at the hole edge where `|∇γ_φ| = 3.0`** (the `∇γ_φ` term multiplies `p*=0`).
> - **New:** the hole **attenuates the write**, not the stored value: `θ_settle(in)/θ_settle(out) = 0.09576` vs `γ/γ_eff = 0.095238` (**ratio 1.0054 ± 0.0002**). A second, independent reason the `γ_φ` rung of the fit-gap ladder recovered −24%.

### §8 (open directions) — **the scope call**

> **"Forgetting" = V5 GO.** Both pre-registered gates passed. R1 delivered *three* emergent-arm results (Cor-13's failure; the unification's exact survival; a measured `T* ≈ 3e-3`). R3 delivered the vault at 110×, not 13.9×, with a new mechanism (local refrigeration) and a free bonus (write attenuation).
> **The short's anchor figure:** `fig1_friction_hole_vault.png` (b)+(c) — *"same `γ_eff`, 8× different `D_θ`"* + *"110× vault; at `T=0`, zero erasure"* — completed by the `T_φ` panel once R4 lands, giving the 2×2 `{γ_φ, T_φ} × {T=0, T>0}`.
> **Honest framing the short must adopt:** the headline "friction remembers, temperature forgets" is a statement about **designed** symmetry at `T=0`, and about **any** near-flat direction at `T > T*`. The emergent case is the *interesting* one, not the embarrassing one: it shows the coset register is an **architectural** object, and it hands us a naturally-occurring washboard (a free `tilt_delta`) for the R5 Arrhenius leg.

### §8 — **`T_φ(q)` spec (`t-lever` §8) needs 3 edits before R4 is funded**
> 1. **§8.3 T6: "≈13.9× SLOWER" → `(γ_eff/γ)² = 110.25×`** (measured 107.77 ± 4.78×). The 13.9× figure is now the measured value of the **scalar-γ control**.
> 2. **§8.1: the `D_θ(q)` / `n₁/₂` formulas hold only under `temperature_field_couple_gamma_phi=True`.** Carry both branches; **split T2** into T2-coupled (`Var = M_i(T+c)`) and T2-absorb (`Var = M_i(T+c)·γ(2−γ)/(γ_eff(2−γ_eff))`), else a correct implementation fails its own acceptance test.
> 3. **§8.1's "never co-locate `T_φ` with `γ_φ`" is understated by 7.94×** — under the shipped path co-location dilutes erasure by `(γ_eff/γ)²` (110×, not 14×). Promote from caution to **hard constraint**.
> 4. **§8.2 item 6 (blocking prerequisite) is DISCHARGED** — the §6 FDT bug is fixed and committed (`d6f8bac`); re-verified on all 6 checkpoints.
> Everything else in §8 stands, and `gate="compact"` is *confirmed* by R3 (γ_φ = 0.5 and 0.0 exactly, zero tail leakage, at the arc positions that matter).

### §7 (known issues) — one new engineering item
> **Checkpoints predating the `friction_field` field cannot be given one.** They unpickle without the attribute (reads are `getattr`-guarded; writes are not), so `eqx.tree_at(lambda m: m.friction_field, ...)` raises `AttributeError`. Workaround used in `v5-gate`: `object.__setattr__` on a `copy.copy`. **Blocks R4 for every existing checkpoint.** Assign `experiment-engineer`: add `with_friction_field(model, field)` or backfill on load in `chlu/utils/checkpoints.py`.
