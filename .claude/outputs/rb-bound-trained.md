# rb-bound-trained — results-analyst report

**Task + acceptance criterion:** F-12 — test the deep-dive's causal retention–bandwidth bound
`n₁/₂·θ̇_max² = 2γln2·m₀c²/((2−γ)ε²δΣ)` on **trained** designed-SO(2) checkpoints, using mass banding
as the `M`-lever at fixed `δΣ`; match the `(εμ/γ)²` residual; exhibit the Newtonian null; handle CM-17.

**Status: done.** All five acceptance clauses met. Predictions were frozen in
`.claude/outputs/rb-bound-trained/PREREG.md` **before** any half-life or bandwidth was measured.
**The law holds on trained models**, and two things sharpen it (§4, §5). Repo read-only; no tracked file touched.

---

## 0. Headline

> **The product is conserved across a 16× inertial-mass band on trained checkpoints, to the
> pre-registered residual.** `n₁/₂` moves **16.6×** (342.4 → 5690.3 steps) and `θ̇_max` moves **4.0×**
> (1.4102 → 0.3526) in opposite directions; the product `P = n₁/₂·θ̇_max²` moves **1.0386×**
> (pre-registered **1.0388×**), 5/5 seeds, and every band's `P/P₀` matches its pre-registered value to
> **≤ 4×10⁻¹¹**. If `M` bought both, `P` would span 16×; it spans 3.9 %.

> **The Newtonian null is exhibited, and it is sharper than "undefined".** In `newtonian_learned`,
> `θ̇(p₀) = p₀/(M_ch r*)` **exactly** (10 digits, `p₀` up to 10⁴): no supremum, so `sup_p P = ∞` for every
> band. And even at *fixed finite* write impulse the Newtonian product is **not conserved** — it spreads
> **15.40× across the band at every `p₀`**, while the relativistic product's spread collapses
> `15.40× → 1.0386×` as the write saturates the causal cap. **The governor creates the conservation law.**

> **Two sharpenings, both registered in advance or forced by the data.**
> **(1)** The residual's true leading coefficient is **`((2−γ)/2)²·(εμ/γ)²`**, not bare `(εμ/γ)²` — the
> deep-dive's label is the **γ→0 limit**. A γ-sweep separates them decisively: measured `d/x² =
> 0.8101 / 0.9030 / 0.9529` at `γ = 0.2 / 0.1 / 0.05` vs `((2−γ)/2)² = 0.8100 / 0.9025 / 0.9506`
> (rel. dev 1.2e-4 / 5.6e-4 / 2.4e-3), where bare `x²` predicts **1.000 at every γ** (19 % wrong at γ=0.2).
> **The deep-dive's own analytic table already contains this** (its deficits `3.95/1.94/0.96/0.48 %` sit on
> the exact map `3.948/1.933/0.957/0.476`, not on `x² = 4.00/2.00/1.00/0.50`).
> **(2) A scope qualifier R5 does not state:** the invariance is a **saturated-write** statement. Below the
> cap (`p₀ ≪ √M_ch·m₀c`) the relativistic arm reproduces the Newtonian `P ∝ 1/M` (spread 15.40×). The
> conserved quantity is built from the **bandwidth capacity**, not the bandwidth you use.

> **`M` allocates; `m₀c²` funds.** `M_ch` and `m₀` have *identical* effect on retention (both enter only
> as `M_eff = m₀M_ch`), but `θ̇_max = c/(√M_ch·r*)` is **independent of `m₀`**. Measured: `M_ch` over 16×
> leaves `P` flat; `m₀,c` scale `P` as **`m₀c²`** to 5 digits (`P/P_ref` vs `m₀c²`: 0.12243/0.125,
> 4.00000/4.000, 8.07964/8.000 — deviations are exactly the `x²` residual, which moves with `m₀`).

> **CM-17 handled by construction, and then *measured*.** Every `n₁/₂` here is a **T = 0 deterministic
> relaxation**, so `T/(m₀c²) = 0` in all 25 relativistic cells, `ℓ_θ ≡ 0`, and CM-16(d)'s boundary-layer
> bias is absent by construction. To justify that choice rather than assert it, I measured on the trained
> checkpoint that (i) the coded relativistic FDT chain's momentum marginal is **Gaussian at every**
> `T/(m₀c²)` (`Var/(M_eff T) = 1.0034–1.0043`, excess kurtosis `≈ −0.003`) where Gibbs demands
> Maxwell–Jüttner (`1.036 / 1.328 / 5.312 / 40.04`) — a **39.9× variance gap** at `T/(m₀c²) = 8`; and
> (ii) the coset diffusion constant that calibrates any thermal FPT instrument is wrong by up to **34×**
> (`D_meas/D_law = 0.971 / 0.669 / 0.184 / 0.029`), because `q̇ = ∇_pT` saturates. **This discharges the
> deep-dive's open O8** ("trained-model consequence CONJECTURED") — CM-17 is now a *measured* trained-model
> fact, not just a free-particle proof.

**The one ML sentence (task item 6):** *the mass spectrum cannot buy retention and write-bandwidth at the
same time — `M` reallocates the causal budget between them and the product is fixed; only raising the
budget itself (`m₀c²`) or lowering the forgetting drive (`δΣ`) buys both.*

---

## 1. Flag-provenance (mandatory, protocol §5)

| item | value |
|---|---|
| repo commit | **`d6f8bac`** (`[experiment-engineer] fix FDT noise inertia: effective_mass() -> effective_inertia()`); `git status --short` **clean** at start and end |
| repo edits | **none** (read-only). All scripts under `.claude/scratch/rb-bound-trained/`, artifacts under `.claude/outputs/rb-bound-trained/` |
| checkpoints | `.claude/scratch/v2-full-runs/runs/designed150_s{42,43,44,45,46}/models/exp_d_chlu.pkl` — `potential_type="so2_invariant"`, `dim=4`, `hidden=64`, 150 ep, trained `kinetic_mode="newtonian_learned"`, `tie_channel_mass=True`, `c=1.0`, `rest_mass=1.0` |
| **retie** | **on, all runs** (pytree-level `log_mass[0]=log_mass[1]=mean`; `H` bit-identical). Verified redundant here: these checkpoints already ship `log_mass[0]==log_mass[1]`, and at `d6f8bac` `effective_mass() ≡ effective_inertia()` |
| **kinetic_mode** | swapped **at probe time** to `relativistic` / `newtonian_learned` (`object.__setattr__` on a shallow copy — these are `eqx` **static** fields; see §8 for `experiment-engineer`). `V_θ` untouched. **Not retrained relativistically** (limitation L2) |
| **M-lever (band)** | `M_ch ∈ {0.5, 1, 2, 4, 8}` (16× span), set **exactly** via `goldstone_harness.log_mass_for_inertia` so `softplus(log_mass)+1e-6 = M_ch`. **Channel coords (0,1) only**; spectator masses untouched (`SO2InvariantPotential` is channel/spectator-decoupled). "Native" `M_ch` = 0.660–0.687 |
| **spurion** | `LinearSpurionPotential(V, δ, u)`, **`spurion_delta = δ = 0.02`**, `spurion_angle = 0` (`u = e₀`). NOT the shipped angular tilt (CM-15: the tilt normalizes `Σ` away) |
| ε (dt) | **0.05** everywhere |
| γ | **0.05** (main grid); γ-sweep `{0.02, 0.05, 0.1, 0.2}` |
| m₀, c | main grid `m₀ = c = 1`; budget arm `m₀ ∈ {0.5,1,2} × c ∈ {0.5,1,2}` |
| **temperature** | **T = 0** for every `n₁/₂` and every `θ̇` in §2–§5. ⇒ **`T/(m₀c²) = 0` in all 25 relativistic cells.** `langevin_noise` **not used** in the main result |
| CM-17 arm only (§6) | `langevin_noise="fdt"`, `T = 2e-4`, `c = √(T/ratio)` for `T/(m₀c²) ∈ {0.01, 0.1, 1, 8}`, 1024 walkers × (600 burn + 1500 samp); diffusion arm `δ=0`, 512 walkers × 5000 steps, 3 seeds |
| **Δ (read tolerance)** | **`Δ_read = θ₀/2 = 0.025 rad`** (write `θ₀ = 0.05 rad`) for the rollout instrument. **`ℓ_θ = ε√(T/M_ch)/(γr*) = 0` exactly (T=0) ⇒ `ℓ_θ/Δ = 0`** in every cell. The Jacobian instrument is Δ-free |
| overdamping | all bands: `h = εμ ∈ [0.0025, 0.0100]`, `h/h*(0.05) ∈ [0.097, 0.393]` ⇒ **register/overdamped band**; `γ_crit = 2εμ ∈ [0.00497, 0.02014] ≪ γ = 0.05`. `x = εμ/γ ≤ 0.1997` ⇒ **inside R5's derived deep-overdamped band** |
| precision | **float64** (`jax_enable_x64`); f32 weights cast to f64 |
| seeds | model seeds `{42,43,44,45,46}` (5) for §2–§5; `{42,43,44}` (3) for §6 and the rollout diagnostics |
| env | jax **0.9.0**, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0, **main venv** `/Users/user/Desktop/CHLU/.venv` (no worktree sync — w6 lesson) |

### 1.1 Commands (all from `.claude/scratch/rb-bound-trained/`)
```
PYTHONPATH=/Users/user/Desktop/CHLU /Users/user/Desktop/CHLU/.venv/bin/python p0_prereg.py             #  6 s  -> PREREG numbers (frozen)
PYTHONPATH=... python m1_bands.py                # 5 seeds x 5 bands x 2 kinetic modes      (~4 min)
PYTHONPATH=... python m2_null_gamma_budget.py    # null + gamma-sweep + m0/c + diagnostics  (~9 min)
PYTHONPATH=... python m3_cm17_instrument.py      # CM-17 momentum marginal + coset diffusion (~22 min)
PYTHONPATH=... python figs.py && python figs4.py
```
Artifacts → `.claude/outputs/rb-bound-trained/`: `PREREG.md`, `prereg_numbers.json`, `m1_bands.json`,
`m2_null_gamma_budget.json`, `m3_cm17_instrument.json`, `fig1_rb_bound.png`, `fig2_newtonian_null.png`,
`fig3_gamma_and_budget.png`, `fig4_cm17_instrument.png`.

---

## 2. The confound: `δΣ` held fixed, `Σ` measured independently (task item 4)

`Σ` is the condensate `r*(δ)`. It is a property of `V` alone (`∇V = 0` is `M`-free), but per the task I
**measure** it per band rather than assume it, by four routes:

| route | what it is | result |
|---|---|---|
| Newton on `dW/dr = δ` | 1-D root on the `u`-axis of the base potential | residual `≤ 3.1e-16` |
| BFGS polish of the **spurioned, banded** model | structure-agnostic; `M` enters the `settle` pre-pass | `‖∇V‖ ≤ 2.6e-13` |
| Hellmann–Feynman `u·q*` | — | `|Σ_HF − Σ_geom| = 0` (exact: `q* = r*u + q_spec`) |
| `−dE_vac/dδ` (central FD, h=1e-6) | **genuinely independent** thermodynamic definition | rel. dev **8.5e-12 … 4.6e-11** |

| seed | `r*(0)` | **`Σ = r*(δ)`** | `δΣ` | `P₀ = 2γln2·m₀c²/((2−γ)ε²δΣ)` |
|---|---|---|---|---|
| 42 | 0.966992651236 | 1.009325043591 | 2.018650e-2 | 704.352 |
| 43 | 0.966262586053 | 1.002961887550 | 2.005924e-2 | 708.821 |
| 44 | 0.979706476404 | 1.004480959307 | 2.008962e-2 | 707.749 |
| 45 | 0.959136140888 | 0.985754563348 | 1.971509e-2 | 721.194 |
| 46 | 0.990693318180 | 1.011977152986 | 2.023954e-2 | 702.506 |

The condensate **runs** with `δ` by +2.1…+4.4 % (`r*(0) → r*(0.02)`) — exactly the effect the shipped
*angular* tilt cannot see (CM-15). All quantities below use `Σ = r*(δ)`, never `r*(0)`.

**`M`-independence of `Σ`, measured.** In `m1_bands.py` the BFGS was seeded at the Newton vacuum, so its
`Σ` spread across bands is `0.0` **by construction of the starting point** — an honest caveat, not evidence.
The real test starts all bands from a common **off-vacuum** point `q = (0.9, 0.35, 0, 0)`:

| band / mode | `settle` end-point after 4000 steps | `r*` after BFGS |
|---|---|---|
| `M=0.5`, relativistic | `(1.00929, 0.00854, …)` | 1.0093250435850976 |
| `M=0.5`, newtonian | `(1.00929, 0.00854, …)` | 1.0093250435850989 |
| `M=8`, relativistic | `(0.96410, 0.29176, …)` — **has not even relaxed the angle** | 1.0093250435363985 |
| `M=8`, newtonian | `(0.96410, 0.29176, …)` | 1.0093250412755503 |
| Newton reference | — | 1.0093250435908556 |

The *dynamics* differ wildly across bands; the *vacuum* agrees to **≤ 2.3e-9** (BFGS `gtol` limited).
**`δΣ` is held fixed while `M` varies over 16×.** ✅

**GMOR on every band (PR-5).** `μ²_H·F² = δΣ` with `μ²_H` the autodiff angular Hessian eigenvalue of
`W = M_eff^{-1/2}·HessV·M_eff^{-1/2}` and `F² = m₀M_ch·r*(δ)²`: **max abs dev `2.88e-16`, max rel dev
`1.42e-14`** over all 50 cells. Angular-mode overlap `= 1.000000` everywhere. So `μ` is *not* a free
parameter — it is fixed by the independently measured `Σ` and the imposed `M`.

---

## 3. PR-1/PR-2 — the product is conserved across a 16× mass band

**Instrument.** `n₁/₂` = ground-truth coset eigenvalue of the **full `2·dim = 8` step Jacobian** at `(q*, 0)`
(v5-gate `e1c` method: among eigenpairs with coset overlap ≥ 0.30, take `max|λ|`); `n₁/₂ = ln2/(−ln|λ|)`.
Coset overlap `= 1.000000` in all 50 cells. **`Δ` = n/a (Δ-free instrument)**, **`ℓ_θ/Δ = 0`** (T=0).

`θ̇_max` = `c/(√M_ch·r*(δ))`, cross-checked empirically as `sup_p (∇_pT·ê_θ)/r*` (§4).

**5 seeds, relativistic, γ = ε = 0.05, δ = 0.02** (mean ± std over seeds):

| `M_ch` | `μ` | `x = εμ/γ` | `n₁/₂` (steps) | `θ̇_max` | `P = n₁/₂·θ̇_max²` | **`P/P₀` measured** | `P/P₀` **pre-registered** | deficit `d` | `d/x²` |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.19972 | 0.19972 | 342.42 ± 3.26 | 1.41024 | 680.94 ± 6.01 | **0.960524 ± 0.000380** | 0.960524 | 3.9476 % | 0.98968 |
| 1.0 | 0.14122 | 0.14122 | 699.20 ± 6.52 | 0.99719 | 695.22 ± 6.29 | **0.980667 ± 0.000182** | 0.980667 | 1.9333 % | 0.96936 |
| 2.0 | 0.09986 | 0.09986 | 1412.32 ± 13.03 | 0.70512 | 702.14 ± 6.42 | **0.990429 ± 0.000089** | 0.990429 | 0.9571 % | 0.95981 |
| 4.0 | 0.07061 | 0.07061 | 2838.35 ± 26.07 | 0.49860 | 705.55 ± 6.48 | **0.995238 ± 0.000044** | 0.995238 | 0.4762 % | 0.95517 |
| 8.0 | 0.04993 | 0.04993 | 5690.31 ± 52.13 | 0.35256 | 707.24 ± 6.51 | **0.997624 ± 0.000022** | 0.997624 | 0.2376 % | 0.95289 |

- `n₁/₂` spans **16.62×**; `θ̇_max` spans **4.00×**; **`P` spans 1.0386×.**
- Per-seed `P/P₀` spread: **1.038354 / 1.038619 / 1.038555 / 1.039354 / 1.038245** (s42…s46) — pre-registered **1.0388**. **5/5 seeds.**
- `max |P/P₀ − pre-registered|` over all 25 cells = **4.0e-11**.
- Counterfactual in Fig 1b: if `M` bought both, `P ∝ 1/M` → 1391 → 87 across the band.

**Instrument agreement.** Full 8×8 Jacobian vs the exact 2×2 mode map from `μ²_H`
(`tr = (2−γ)(1−h²/2)`, `det = 1−γ`): `|n_Jac/n_2×2 − 1| ≤ 3.9e-11` over all cells. **The single-mode
reduction (F5 Prop-2) is exact at `p≈0` on the trained SO(2) potential** — no radial–angular or spectator
contamination. This is the one place the trained model could have broken the law, and it does not.

**PR-6 — retention is kinetic-mode blind.** `n₁/₂(relativistic) = n₁/₂(newtonian_learned)` to
`≤ 1.0e-11` relative at every band (both linearize to `M_eff = m₀M`, and `m₀=1`). *Retention alone contains
no bound; the entire content of R5 lives in `θ̇_max`.* This also certifies that a Newtonian-run retention
measurement would have been admissible under CM-17 — I did not need that escape hatch because T = 0.

**Rollout instrument (Δ = 0.025 rad, ℓ_θ/Δ = 0), and its bias.** A T=0 nonlinear rollout from
`(θ₀, p=0)` overestimates `n₁/₂` by an **additive start-up lag** of `20.41 ± 0.30` steps across all 25 cells
(`1/γ = 20`): momentum must build before the angle moves. Reproduced by the analytic 2×2 map released from
rest (predicted lag 20.07–21.23). **It is additive, not multiplicative**, so it inflates `P/P₀` by 6.1 % at
`M=0.5` and 0.36 % at `M=8` — i.e. it would *fake* a small `M`-dependence. The Jacobian instrument is used
for every headline number. Amplitude linearity: lag = 20.02 / 20.40 / 21.77 / 36.49 steps (band mean) at
`θ₀ = 0.02 / 0.05 / 0.1 / 0.3` — **`θ₀ ≤ 0.05` is the linear window**; at `θ₀ = 0.3` the coset pendulum
(`V = const − δr*cosθ`) is visibly anharmonic (lag up to 61.9 steps at `M=8`).

📊 `fig1_rb_bound.png`

---

## 4. PR-7 — the Newtonian null (half the result)

`θ̇(p₀)` measured exactly as `(∇_pT·ê_θ)/r*` at tangential momentum `p₀`, swept over 6 decades.

| arm | `θ̇(p₀)` | supremum |
|---|---|---|
| `relativistic` | `→ θ̇_max` : `θ̇/θ̇_max = 0.7071 / 0.9950 / 0.99995 / 0.999999995` at `p₀ = 1 / 10 / 100 / 10⁴` | **`c/(√M_ch r*)`, finite** |
| `newtonian_learned` | `= p₀/(M_ch r*)` **to 10 significant digits** at every `p₀` up to 10⁴ | **none — `sup_p θ̇ = ∞`** |

Hence `sup_p P = n₁/₂·sup_pθ̇² = ∞` in the Newtonian arm, for every band: **there is no bound to be
invariant.** But the stronger, quantitative statement is at *fixed finite* write impulse:

`P(p₀) := n₁/₂·θ̇(p₀)²`, band spread `max_M/min_M`:

| `p₀` | 0.01 | 10 | 10⁴ |
|---|---|---|---|
| **relativistic** | 15.4020 | 1.0512 | **1.0386** |
| **newtonian_learned** | 15.4049 | 15.4049 | 15.4049 |

The Newtonian product is **not conserved at any write amplitude** (spread ≈ 16×, i.e. `P ∝ 1/M`, the
`n₁/₂∝M · θ̇²∝1/M²` scaling; 15.40 rather than 16 because `n₁/₂` carries the `x²` residual). The
relativistic product **is** conserved — but only once the write saturates the cap.

> **New scope qualifier (not in R5).** The bound is a statement about **bandwidth *capacity*, not
> bandwidth used.** In the weak-write corner `p₀ ≪ √M_ch·m₀c` the relativistic arm is Newtonian and shows
> the same 15.4× spread. The conserved quantity is `n₁/₂·θ̇_max²`; `n₁/₂·θ̇(p₀)²` is conserved only as
> `p₀ → ∞`. Any ML reading of R5 must say *capacity*.

📊 `fig2_newtonian_null.png`

---

## 5. PR-3/PR-4 — the residual is the exact map; the deep-dive's `(εμ/γ)²` is its γ→0 limit

Expanding the exact 2×2 damped-Verlet mode map (`u ≡ 1−λ_slow`, `b = γ + c`, `c = (2−γ)h²/2`, `h = εμ`):

```
1 − n_exact/n_od  =  ((2−γ)/2)² · (εμ/γ)²  +  O(x⁴)          [derived, then registered as PR-3]
```

so the deep-dive's `(εμ/γ)²` is the **`γ→0` limit**; at γ = 0.05 the coefficient is `((2−γ)/2)² = 0.950625`.
Registered in `PREREG.md` §PR-3 **before** measuring. A γ-sweep separates the two (bare `x²` ⇒ `d/x² ≡ 1`):

`d/x²`, 5-seed mean (`*` = `x > 0.25`, **outside** R5's derived deep-overdamped band, excluded):

| γ | `((2−γ)/2)²` | M=0.5 | M=1 | M=2 | M=4 | M=8 | deepest cell rel. dev vs `((2−γ)/2)²` | vs bare `x²` |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.980100 | 1.71206* | 1.14292* | 1.04863 | 1.01201 | 0.99554 | 1.6e-2 (x still 0.125) | 4.5e-3 |
| 0.05 | **0.950625** | 0.98968 | 0.96936 | 0.95981 | 0.95517 | **0.95289** | **2.4e-3** | **4.7e-2** |
| 0.10 | **0.902500** | 0.91076 | 0.90659 | 0.90454 | 0.90352 | **0.90301** | **5.6e-4** | **9.7e-2** |
| 0.20 | **0.810000** | 0.81163 | 0.81081 | 0.81041 | 0.81020 | **0.81010** | **1.2e-4** | **1.9e-1** |

Convergence is controlled by `x`, not `γ` (the `O(x⁴)` term); at the deepest cell of each γ the coefficient
is recovered to `1.2e-4 … 2.4e-3`, while bare `x²` is wrong by **19 % at γ = 0.2**. The γ = 0.02 / M ≤ 1
cells sit at `x = 0.353 / 0.250` — I flag them as excursions and exclude them (R5 is derived for `εμ ≪ γ`;
its underdamped counterpart is theorist's open O6 and was **not** entered: `γ_crit = 2εμ ≤ 0.0201 < γ` in
every cell reported).

**This is not a discrepancy with the deep-dive's numerics — it is a relabelling of them.** Its analytic-map
deficits `3.95 / 1.94 / 0.96 / 0.48 %` lie on the exact map (`3.948 / 1.933 / 0.957 / 0.476`, rel. dev
≤ 4e-3) and **not** on `x² = 4.00 / 2.00 / 1.00 / 0.50` (rel. dev 1.3–4.4 %). The label, not the data, was
the γ→0 limit.

**PR-8 — `M` allocates, `m₀c²` funds** (seed-mean, `M_ch = 1`, 5 seeds):

| `m₀` | `c` | `n₁/₂` | `θ̇_max` | `P/P_ref` | `m₀c²` | `P/P₀` |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 342.419 | 0.49860 | 0.12243 | 0.125 | 0.960524 |
| 0.5 | 2.0 | 342.419 | 1.99439 | 1.95892 | 2.000 | 0.960524 |
| 1.0 | 0.5 | 699.199 | 0.49860 | 0.25000 | 0.250 | 0.980667 |
| 1.0 | 2.0 | 699.199 | 1.99439 | 4.00000 | 4.000 | 0.980667 |
| 2.0 | 1.0 | 1412.317 | 0.99719 | 2.01991 | 2.000 | 0.990429 |
| 2.0 | 2.0 | 1412.317 | 1.99439 | 8.07964 | 8.000 | 0.990429 |

`n₁/₂` is **exactly `c`-independent** (699.199 at all three `c`); `θ̇_max` is **exactly `∝ c`**; `P/P₀` is
**exactly `c`-invariant**. Deviations of `P/P_ref` from `m₀c²` are *precisely* the `x²` residual, which
moves with `m₀` because `μ² = δΣ/(m₀M_ch r*²)`: `0.12243/0.125 = 0.9794 = 0.960524/0.980667`. **Raising `m₀`
buys retention at zero bandwidth cost — `M_ch` cannot.**

📊 `fig3_gamma_and_budget.png`

---

## 6. CM-17, handled explicitly — and discharged on a trained checkpoint (task item 5)

**What I did:** every `n₁/₂` and `θ̇` in §2–§5 is **T = 0 deterministic**. `T/(m₀c²) = 0` in all 25
relativistic cells. `ℓ_θ = ε√(T/M_ch)/(γr*) = 0` ⇒ `ℓ_θ/Δ = 0`. No equilibrium temperature, no Gibbs
measure, no diffusion constant enters any headline number. `θ̇_max` is measured pathwise/analytically.

**Why that was necessary (measured, not asserted).** `designed150_s{42,43,44}`, `M_ch=1`, spurioned,
`langevin_noise="fdt"`, `T = 2e-4`, `c` set so `T/(m₀c²)` = ratio; 1024 walkers × 1500 samples after 600 burn:

**(i) The momentum marginal.** Gibbs demands `π(p) ∝ exp(−T(p)/T)` = Maxwell–Jüttner. The coded O-step
`p ← (1−γ)p + σξ` is a linear OU recursion ⇒ exactly Gaussian.

| `T/(m₀c²)` | `c` | coded chain `Var(p_θ)/(M_eff T)` | coded exc. kurtosis | **Gibbs (MJ, dim=4) demands** | MJ kurtosis |
|---|---|---|---|---|---|
| 0.01 | 0.14142 | 1.00409 ± 0.00158 | −0.0031 | 1.03561 | −0.0153 |
| 0.10 | 0.04472 | 1.00381 ± 0.00161 | −0.0026 | 1.32806 | 0.2782 |
| 1.00 | 0.01414 | 1.00343 ± 0.00177 | −0.0029 | **5.31157** | 1.0261 |
| 8.00 | 0.00500 | 1.00426 ± 0.00196 | −0.0039 | **40.04147** | 1.1975 |

Newtonian control (Gibbs momentum law **is** Gaussian): `1.00414 ± 0.00158`, kurtosis `−0.0033`,
**bit-identical at all four `c`** (as it must be — `c` does not enter Newtonian dynamics). The residual
`+0.4 %` is the `O(ε²)` shadow/finite-sample floor.

> **In the relativistic arm, `Var/(M_eff T) = 1.000` is not a success — it is the defect.** Same number,
> opposite meaning. My MJ quadrature reproduces CM-17's own dim=1 table exactly
> (`1.0150 / 1.1534 / 2.6995 / 16.282`; kurtosis `0.030 / 0.295 / 1.857 / 2.907`), validating the reference;
> the table above is the dim=4 marginal appropriate to these checkpoints.

**(ii) The diffusion constant that calibrates any thermal FPT instrument.** `D_θ = εT(2−γ)/(2F²γ)` is
derived from `q̇ = p/M_eff`. Relativistically `q̇ = ∇_pT` **saturates**. Measured on the `δ = 0` (exactly
flat) trained coset:

| `T/(m₀c²)` | relativistic `D_meas/D_law` | newtonian `D_meas/D_law` |
|---|---|---|
| 0.01 | 0.97070 | 1.02717 |
| 0.10 | 0.66881 | 1.02717 |
| 1.00 | 0.18363 | 1.02717 |
| 8.00 | **0.02903** | 1.02717 |

Newtonian is flat and `c`-independent (1.02717 at all four `c`) — Gibbs' position marginal is `c`-free.
Relativistic collapses by **34×**. A thermal `n₁/₂ = Δ²/(2D_θε)` instrument in the relativistic arm is
therefore mis-normalized by up to 34×, **and would have been mis-normalized *differently in each band*,**
since `M_ch` sets `θ̇_max` and hence how relativistic the thermal motion is. **It would have manufactured
an apparent violation of the very law under test.** This is the trap the task flagged, and it is real.

**⇒ Discharge of the deep-dive's open O8.** CM-17's "trained-model consequence CONJECTURED" is now
**measured**: on trained designed-SO(2) checkpoints the coded relativistic FDT chain samples a Gaussian
momentum law where Gibbs demands Maxwell–Jüttner (39.9× variance gap at `T/(m₀c²)=8`; Gibbs already demands 32 % more variance at 0.1),
and its coset diffusion is not the Newtonian `D_θ` law. **Note also that even `T/(m₀c²) = 0.01` — nominally
"benign" — biases `D_θ` by 5.5 %** relative to the Newtonian control.

📊 `fig4_cm17_instrument.png`

---

## 7. Limitations & confounds (honest)

- **L1 — one architecture.** Designed *exact* SO(2), `dim=4`, `hidden=64`, 150 ep, 5 seeds. Per v5-gate /
  CM-16(a), **emergent (MLP) CHLUs have no continuous coset register at all** (2–3 washboard minima,
  `1−|λ_coset| ≈ 1e-3`), so F-12 as stated is **not even well-posed** on the emergent arm — `n₁/₂` there is
  a pseudo-Goldstone relaxation with its own `μ²_soft`, not `δΣ/F²`. The bound is an **architectural**
  statement, exactly as the register is. *Do not generalize this to "CHLU" unqualified.*
- **L2 — probe-only relativity.** The checkpoints were **trained** `newtonian_learned`; I swapped
  `kinetic_mode` at probe time. `V_θ`, `Σ`, `F²`, `μ²` are untouched by that swap (the vacuum and the Hessian
  are kinetic-mode-blind up to `M_eff = m₀M`), so the *law* is tested correctly — but **no model was trained
  relativistically**, and training under a causal cap could move `V_θ` (hence `Σ`). Untested.
- **L3 — `M` is designed, not learned.** The band is imposed exactly (`log_mass_for_inertia`), not induced
  by V3's `mass_lr_mult` recipe. This tests *the law*, not *whether training discovers the allocation*. A
  natural follow-up (§9 F-12b).
- **L4 — the law's content is a composition of three exact facts,** not a statistical fit: (a) GMOR
  `μ²F² = δΣ` (1.4e-14), (b) full-Jacobian ≡ exact 2×2 map (3.9e-11), (c) `∇V = 0` is `M`-free (2.3e-9).
  That is *why* it is a conservation law rather than a regression — but it also means seed variance enters
  only through `r*(δ)` (`P` std 6.0–6.5 across seeds, ~0.9 %), and the `P/P₀` error bars (≤ 3.8e-4) are
  **not** independent replications of a noisy measurement. State it that way; don't oversell the ±.
- **L5 — single `δ`.** `δ = 0.02` only. `P₀ ∝ 1/(δΣ)` is untested across `δ`; `Σ(δ)` runs by +4.4 %, so a
  `δ`-sweep is a genuine (cheap) test of the `1/δΣ` leg. Not run.
- **L6 — the rollout instrument is biased** by an additive `≈1/γ` start-up lag (measured `20.41 ± 0.30`
  steps) and by pendulum anharmonicity at `θ₀ ≥ 0.1`. All headline numbers use the Jacobian. Anyone quoting
  a *first-passage* `n₁/₂` figure must subtract the lag or say it's there.
- **L7 — derived-band discipline.** All headline cells have `x = εμ/γ ≤ 0.1997` and `γ_crit ≤ 0.0201 ≪ γ`.
  The two γ = 0.02 cells with `x > 0.25` are reported and **excluded**. The underdamped regime (theorist's
  open **O6**) was **not entered**; R5 has no derivation there and I make no claim about it.
- **L8 — `θ̇_max` uses the *bare* `M_ch` (+1e-6), not `M_eff`.** `v_max = c/√M` is `m₀`-free (code:
  `M_inv = 1/(softplus(log_mass)+1e-6)` inside the square root; `m₀` enters only the rest-energy term). This
  is the asymmetry that makes PR-8 work, and it is a **code fact**, verified to 10 digits in the Newtonian arm.
- **L9 — CM-17 arm `T` and `c` are coupled.** I varied `T/(m₀c²)` by lowering `c` at fixed `T = 2e-4`
  (rather than raising `T`, which would leave the harmonic window). Lowering `c` also slows the dynamics;
  the momentum marginal equilibrates on `1/γ = 20` steps regardless, and the diffusion arm was run to 5000
  steps with the MSD fit on the last 3/4. Not a confound for the *shape* claim (Gaussian vs MJ), which is
  amplitude-free (kurtosis).
- **L10 — no `n₁/₂` in this report is thermal.** Every `Δ` / `ℓ_θ/Δ` pair quoted is `(0.025 rad, 0)`.
  The CM-16(d) boundary-layer bias `1 + 3.099·ℓ_θ/Δ` is therefore exactly 1 here — **by construction, not by
  luck.** The v5-gate §3.5 contamination cannot occur in this design.

---

## 8. For `experiment-engineer` (no bug in the physics; two ergonomics items)

1. **Static-field mutation has no supported helper.** `kinetic_mode`, `c`, `rest_mass` are
   `eqx.field(static=True)`, so `eqx.tree_at` cannot touch them; every probe agent is forced into
   `object.__setattr__` on a `copy.copy` (same family as v5-gate's `friction_field` `AttributeError`).
   Suggest `chlu.core.chlu_unit.with_kinetic(model, mode=None, c=None, m0=None)` (and the already-requested
   `with_friction_field`). This bites **any** kinetic-mode ablation, which §8 of the handover explicitly wants.
2. **`goldstone_harness.log_mass_for_inertia` is the right band-setter and is under-advertised.** It gives
   `softplus(log_mass)+1e-6 == inertia` exactly, which is what makes `θ̇_max = c/√M_ch·r*` exact to 10
   digits. Worth cross-referencing from `chlu/data/band_selection.py` (which normalizes to unit geomean and
   does *not* pin absolute `M`).
3. **Verified good at `d6f8bac`:** `effective_mass() ≡ effective_inertia()` on all five `designed150_*`;
   `spectrum_probe` correctly returns `M_eff = rest_mass·(M+1e-6)` in relativistic mode. **No defect found.**

---

## 9. Recommended next experiments

| id | experiment | cost | why |
|---|---|---|---|
| **F-12a** | **`δ`-sweep**: `δ ∈ geomspace(2e-3, 0.1, 5)` × 3 bands, same instrument. Test `P₀ ∝ 1/(δΣ)` with the *running* `Σ(δ)`. | **tiny** (~3 min; all Jacobian) | the only leg of R5 untested here; `Σ` runs 4.4 %, so `Σ(0)` vs `Σ(δ)` is discriminable |
| **F-12b** | **Learned allocation**: induce the band with `mass_lr_mult ≈ 10` (CM-5 safe default) instead of imposing it; ask whether SGD lands on a *particular* point of the conserved line, and whether the task's required bandwidth selects it. | medium | turns the conservation law into a **training** statement: "what does the optimizer spend the budget on?" This is the ML payoff |
| **F-12c** | **Retrain relativistically** at 2–3 `c` and re-measure. Does the causal cap move `V_θ` (hence `Σ`)? | medium | closes L2, the only structural gap |
| **F-10** | Now strongly motivated by §6(ii): the causal memory-lifetime floor. My `D_rel/D_newt` collapse (`0.971 → 0.029` over `T/(m₀c²) ∈ [0.01, 8]`) **is** R7's saturation, measured on a trained checkpoint for the first time. F-10 is one script away. | small | the ML-measurable benefit that earns the governor a place in the memory story (P1) |
| **F-9** | CM-17's Exp-C consequence. §6 shows the defect is already **32 % in `Var(p)` at `T/(m₀c²) = 0.1`** and 5.3× at 1.0 — and `experiment_c` runs at exactly 1.0. The `c=5` fix (`finalA`) is one config line. | tiny | now has trained-checkpoint evidence behind it, not just a free-particle proof |

---

## Git footprint

**None.** No tracked file created, modified, or deleted. `git status --short` clean at start (`d6f8bac`)
and at end. No branch, no commit, no worktree. Scripts → `.claude/scratch/rb-bound-trained/`
(`rbcommon.py`, `p0_prereg.py`, `m1_bands.py`, `m2_null_gamma_budget.py`, `m3_cm17_instrument.py`,
`figs.py`, `figs4.py`, `m3.log`); artifacts → `.claude/outputs/rb-bound-trained/`.

---

## Proposed handover updates (for the Hub)

### §1.6 / new claims-matrix entry — **CM-18: the retention–bandwidth conservation law, on trained models**

> **CM-18 (new; `rb-bound-trained`, w14; pre-registered).** On trained designed-SO(2) checkpoints
> (`designed150_s{42..46}`, `d6f8bac`, f64, `retie`, linear spurion `δ=0.02`, `ε=γ=0.05`, `m₀=c=1`,
> **T=0 deterministic instrument**), the causal retention–bandwidth bound
> `n₁/₂·θ̇_max² = 2γln2·m₀c²/((2−γ)ε²δΣ)` **holds**. Across a **16× inertial-mass band**
> (`M_ch ∈ {0.5,1,2,4,8}`) `n₁/₂` moves **16.62×** and `θ̇_max` **4.00×** in opposite directions while the
> product moves **1.0386×** (pre-registered 1.0388×), 5/5 seeds; every band's `P/P₀` matches its
> pre-registered value to **≤ 4e-11**. `Σ = r*(δ)` is measured four ways (Newton / BFGS / Hellmann–Feynman /
> `−dE_vac/dδ`, agreeing to ≤4.6e-11) and is `M`-independent to **≤2.3e-9** from a common off-vacuum start.
> GMOR `μ²F² = δΣ` holds on every band (max rel dev **1.42e-14**). The full 8×8 step Jacobian's coset
> eigenvalue equals the exact 2×2 mode map to **3.9e-11** ⇒ F5 Prop-2's single-mode reduction is exact at
> `p≈0` on the trained potential.
> **The Newtonian null is exhibited and is stronger than "undefined":** `θ̇(p₀) = p₀/(M_ch r*)` to 10 digits
> (no supremum ⇒ `sup_p P = ∞`), **and** at *fixed* `p₀` the Newtonian product spreads **15.40× across the
> band at every `p₀`**, whereas the relativistic product's spread collapses `15.40× → 1.0386×` as the write
> saturates the cap. **The causal governor creates the conservation law.**
> **`M` allocates, `m₀c²` funds:** `M_ch` and `m₀` act identically on retention (both only via
> `M_eff = m₀M_ch`) but `θ̇_max = c/(√M_ch r*)` is `m₀`-free ⇒ `M_ch` over 16× leaves `P` flat while
> `(m₀,c)` scale `P` as `m₀c²` to 5 digits. `n₁/₂` is exactly `c`-independent; `P/P₀` exactly `c`-invariant.
> **Scope (mandatory):** designed **exact** SO(2) only — emergent MLP checkpoints have **no continuous coset
> register** (CM-16a), so F-12 is not well-posed there; probe-only relativity (trained `newtonian_learned`,
> `kinetic_mode` swapped at probe); `M` imposed, not learned; single `δ`; deep-overdamped band `x = εμ/γ ≤
> 0.1997`, `γ_crit ≤ 0.0201 ≪ γ` (underdamped = theorist's open O6, **not entered**);
> **`n₁/₂` is a T=0 deterministic relaxation ⇒ `Δ = 0.025 rad`, `ℓ_θ/Δ = 0` exactly.**

### §1.6 / F5 note / deep-dive §7bis R5 — **the residual's coefficient must be corrected**

> **R5's "residual = `(εμ/γ)²`" is the `γ→0` limit.** The exact leading deficit of the 2×2 damped-Verlet map is
> ```
> 1 − n_exact/n_od = ((2−γ)/2)²·(εμ/γ)² + O((εμ/γ)⁴)
> ```
> Registered in `PREREG.md` §PR-3 before measurement, then confirmed by a γ-sweep on the trained checkpoints:
> measured `d/x²` at the deepest cell = **0.81010 / 0.90301 / 0.95289** for `γ = 0.2 / 0.1 / 0.05` vs
> `((2−γ)/2)² = 0.8100 / 0.9025 / 0.9506` (rel. dev 1.2e-4 / 5.6e-4 / 2.4e-3), where bare `x²` predicts
> **1.000 at every γ** and is wrong by **19 % at γ = 0.2**.
> **The deep-dive's own analytic table already agrees with the corrected form** — its deficits
> `3.95/1.94/0.96/0.48 %` sit on the exact map (`3.948/1.933/0.957/0.476`) and *not* on
> `x² = 4.00/2.00/1.00/0.50`. **This is a relabelling, not a retraction**: no number in §7bis R5 changes,
> only the closed form attached to them. At γ = 0.05 the two differ by 5 %, which is why it went unnoticed.
> **Action:** correct the wording at the R5 site and anywhere the F5 note inherits "the deep-overdamped
> correction `(εμ/γ)²`" → "`((2−γ)/2)²(εμ/γ)²`, i.e. `(εμ/γ)²` as `γ→0`". Assign to `f5-corrigendum-3` /
> theorist.

### §1.6 — **a scope qualifier R5 needs before it enters any paper**

> The bound is a statement about **bandwidth capacity, not bandwidth used.** `n₁/₂·θ̇_max²` is invariant;
> `n₁/₂·θ̇(p₀)²` is invariant **only for `p₀ ≫ √M_ch·m₀c`** (measured: band spread `15.40× → 1.0386×` as
> `p₀: 0.01 → 10⁴`). Below the cap the relativistic arm is Newtonian and `P ∝ 1/M`. Any ML sentence must say
> *capacity*: **"the mass spectrum cannot buy retention and write-bandwidth at the same time; only the causal
> budget `m₀c²` (or a smaller forgetting drive `δΣ`) can."**

### §1.6 / CM-17 — **O8 DISCHARGED: the relativistic Gibbs defect is now a measured trained-model fact**

> CM-17's trained-model consequence was tagged **CONJECTURED (O8)**. Measured here on
> `designed150_s{42,43,44}` (`fdt`, `T=2e-4`, 1024 walkers): the coded relativistic chain's momentum marginal
> is **Gaussian at every `T/(m₀c²)`** — `Var(p_θ)/(M_eff T) = 1.0034…1.0043`, excess kurtosis `≈ −0.003` —
> where Gibbs demands Maxwell–Jüttner: **1.036 / 1.328 / 5.312 / 40.04** (dim-4 marginal) at
> `T/(m₀c²) = 0.01 / 0.1 / 1 / 8`. A **39.9× variance gap** at 8; **32 % already at 0.1**. Newtonian control
> `1.00414 ± 0.00158`, bit-identical at all four `c`. (My MJ quadrature reproduces CM-17's dim-1 table
> exactly: `1.0150/1.1534/2.6995/16.282`, kurtosis `0.030/0.295/1.857/2.907`.)
> **Second, independent consequence:** the coset diffusion constant is not the Newtonian law —
> `D_meas/D_law = 0.971 / 0.669 / 0.184 / 0.029` (relativistic) vs a flat `1.02717` (Newtonian, `c`-free).
> ⇒ **Any thermal first-passage `n₁/₂` instrument is mis-normalized by up to 34× in the relativistic arm,
> and mis-normalized *band-dependently*** (since `M_ch` sets how relativistic the thermal motion is) — it
> would have manufactured an apparent violation of F-12. **F-12 avoided it by measuring `n₁/₂` at `T=0`.**
> This also hands **F-10** (causal memory-lifetime floor) its first trained-checkpoint evidence: `D_rel`
> saturates exactly as R7 predicts.

### §5 (provenance) / §7 (issues)
- New artifacts: `.claude/outputs/rb-bound-trained/` (PREREG + 3 JSON + 4 PNG). Scripts in
  `.claude/scratch/rb-bound-trained/`. **No tracked code touched**; base `d6f8bac` unmoved.
- **§7 ergonomics (for `experiment-engineer`):** `kinetic_mode` / `c` / `rest_mass` are `eqx` **static**
  fields ⇒ `eqx.tree_at` cannot mutate them; every probe agent must use `object.__setattr__` on a
  `copy.copy`. Request `with_kinetic(model, mode=, c=, m0=)` alongside the already-requested
  `with_friction_field`. This blocks *every* kinetic-mode ablation in §8's backlog.
- **No physics defect found** at `d6f8bac`: `effective_mass() ≡ effective_inertia()` on all 5 designed
  checkpoints; `spectrum_probe` returns `M_eff = m₀(M+1e-6)` correctly in relativistic mode.
