# v5-vcurve-validation — results-analyst report

**Task + acceptance criterion:** ME-1 — close the referee's MF-3 caveat with data: a direct `T=0`
**rollout** `n₁/₂(γ)` on the banked dense γ-grid (designed 5 seeds + emergent 3 seeds) beside the
Jacobian curve, with the per-γ ratio curve, and a verdict on whether the **argmin** and the **two
slopes** survive on the rollout instrument. ME-3 — the `γ_φ` friction-hole vault + `D̂` estimator on
`emergent150_s{42,43,44}` at `T > T* ≈ 3e-3`, with the scalar-γ laundering control beside it.
**Prereg-first; both instruments' numbers side by side; no verdict language beyond the prereg's own
success/failure lines.**

**Status: done.** All predictions were written to
`.claude/outputs/v5-vcurve-validation/PREREG.md` **before any harness ran** (P1–P6/F1–F5 for ME-1,
Q1–Q5/G1–G5 for ME-3, plus the two verdict rules).

> ### ⚠ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — needs an owner at review)
> 1. **MF-3's "19–43% Jacobian/rollout disagreement" is not a rate disagreement.** It is a
>    **constant-in-γ amplitude offset** on a *threshold* instrument (`d log(ratio)/d log γ = −0.0034 …
>    +0.0001`, F1's threshold 0.10). The rates agree to **0.5 ± 0.3%** (asymptotic window). The MF-3
>    fix sentence the writer is about to insert must say this, not "the instruments disagree by 19–43%".
> 2. **MF-1's instrument-floor problem has a much better replacement number.** The direct rollout gives
>    a *model-side* bound on the designed coset: over 5 seeds × 48 γ × 3 write amplitudes × 32 000 steps,
>    `max|θ(n) − δ| = 9.40e-13 rad` ⇒ `Γ ≤ 7.31e-17` per step ⇒ **`n₁/₂ ≥ 9.5e15` steps**. This is
>    quotable where "μ² = 1.7e-12" (a ring-profile instrument floor) is not.
> 3. **`v5-gate`'s R3 `T6` design does not transfer to emergent unmodified**: on an MLP checkpoint the
>    "outside" arm `θ=π` is **not a vacuum** (`|∇V| = 0.042/0.105/0.041` vs `0.0000` at `θ=0`). Any
>    emergent FPT number quoted against a `θ=π` baseline is confounded. A same-site control is required
>    and is supplied here.
> 4. **The `t-lever`/`v5-gate` `σ_θ` diagnostic is non-stationary on emergent above `T*`** — the scalar
>    control that must return 1.000 returns **0.459 ± 0.118**. Do not quote an emergent `σ_θ` ratio.

**DIAL DECLARATION (echoed).** Dial: **none — instrument validation + generalization probe on banked
checkpoints.** Laundering control: n/a for ME-1 (the two instruments are each other's control); for
ME-3 the scalar-γ control (`γ = γ_eff = 0.525`, no field) travels beside every field number, as in
`v5-gate` §2.5. Falsifies: the pre-registered lines F1–F5 / G1–G5. Does **not** falsify: a level
(multiplicative) offset between instruments that is constant in γ; nor an emergent cell whose thermal
spread exceeds the escape tolerance, which is out of the diffusive window by construction.

---

## 0. Headline

> **ME-1 — the V-curve survives the rollout instrument, and MF-3's caveat is the wrong shape.** On the
> full nonlinear `T=0` rollout of the trained model, the **argmin** reproduces the Jacobian's to
> `0.9001 ± 0.0052` vs `0.9032 ± 0.0027` in units of `γ_crit = 2εμ_soft` (3/3 seeds), and the **two
> slopes** reproduce to `−0.983 ± 0.013` / `+1.117 ± 0.010` vs the Jacobian's `−1.0020 ± 0.0003` /
> `+1.1182 ± 0.0107`. The decay **rates** agree to `Γ_jac/Γ_roll = 0.995 ± 0.003` in the asymptotic
> window. The banked "19–43% gap" is entirely a **constant amplitude offset** on the *first-crossing*
> instrument (`n_jac/n_R1 = 1.331 / 1.798 / 1.820`, coefficient of variation **2.2–3.7%** over 21–25
> γ-points, `d log(ratio)/d log γ ≈ 0`). The shape claims live on the rate; the offset lives on the
> threshold. **P1(marginal)·P2(main clause)·P3·P4·P5·P6(main clause) hold; no falsifier F1–F5 fires.**
>
> **And the referee's A1 "linearization tautology" is answered with a number, in both directions.** The
> left branch of the Jacobian V-curve *is* an integrator identity — `|λ| = √(1−γ)` exactly, so
> `n₁/₂ = 692.5` at `γ=0.002` on s42 **and** s43 despite a 2.7× difference in `μ²`. But it was not
> guaranteed that the *full nonlinear model* would shed amplitude at that rate: it does, at finite write
> amplitude up to `δ = 0.5` rad (argmin `0.897–0.917 γ_crit`, slopes unchanged to <2%). The falsifiable
> content lives in the argmin and the `+1` branch, and both survive.

> **ME-3 — the vault's two *laws* transfer to an emergent register exactly; the vault's *contrast
> number* does not, and the reason is the control arm, not the field.**
> - **Q1 (the refrigerator) transfers exactly.** Over 24 field cells (3 seeds × 2 T × 4 `γ_φ`),
>   `Var(p_i)/(M_i T)` / absorb prediction = **0.9998 ± 0.0019**. At `γ_φ=0.5`: `T_local/T = 0.12570`,
>   a **7.955× refrigerator** (predicted 7.942×). The coupled-bath hypothesis predicts 1.0 and is
>   measured at **0.2235** — rejected on an emergent checkpoint too.
> - **Q2 (the `D̂ ∝ γ_eff^{-2}` law) transfers.** On cells whose register is bounded,
>   `D̂/D_absorb = 1.016 … 1.103` (per-`γ_φ` means, spread ≤ 0.08); `D̂/D_coupled = 0.131 ± 0.007` at
>   `γ_φ=0.5`. **Law-referenced vault `106.1 ± 5.0×`** vs the prediction 110.25× and `v5-gate`'s banked
>   designed `107.77 ± 4.78×`.
> - **G4 fires.** The field/scalar contrast measures **23.39 ± 10.06** on emergent (pre-registered band
>   [6.5, 9.5]; falsifier > 14) where the **identical estimator on designed gives `8.03 ± 0.80`**
>   (banked `8.11 ± 0.37`). **The failure mode is named:** the scalar-γ control arm is itself
>   anharmonically delocalised on emergent (`D̂/D_flat = 1.18 … 5.23`), while the field arm sits on the
>   law (1.02–1.15). By the prereg's own verdict rule this is a **designed-only scope confirmation for
>   the *contrast number*** — while Q1 and Q2, the two statements the number is derived from, transfer.
> - **G3 fires, and its own control says why.** `σ_θ(in)/σ_θ(out) = 0.143 ± 0.052` vs the predicted
>   0.355; but the scalar control, which must return 1.000, returns **0.459 ± 0.118** ⇒ the outside arm
>   is not a stationary bounded register at `T = 4e-3`. **The statement that does hold, and has no
>   designed analogue:** the hole **confines** the emergent register — hop fraction (`|θ| > 1` rad)
>   `5.5% / 43.0% / 2.4%` outside → **`0.0000 / 0.0000 / 0.0000`** inside, with the scalar control at
>   the *same* `γ_eff` still hopping (`0.73% / 10.2% / 0.26%`).
> - **Q5 is 2/3.** Same-site FPT vault: **`> 1379×` (s42), `35.5×` (s43), `> 1290×` (s44)**.
>   Pre-registered `> 300×` holds on s42/s44 as censoring-limited lower bounds; **G5 fires on s43**
>   (35.5× < 110×) — the geometrically atypical seed `v5-gate` already flagged (L2).

---

## 1. Flag-provenance (mandatory, protocol §5)

| item | value |
|---|---|
| repo commit (start **and** end) | **`7fcef50`** `[experiment-engineer] tests: TTT inner-loop stability, the D5 passthrough, resume-accept`. `git status --short` clean throughout; **no tracked file touched.** |
| checkpoints | `.claude/scratch/v2-full-runs/runs/{emergent150_s42,43,44 · designed150_s42,43,44,45,46}/models/exp_d_chlu.pkl` — the **same** checkpoints as `v5-gate`'s banked curve |
| model config (all) | `dim=4`, `hidden=64`, 150 epochs, `kinetic="newtonian_learned"`, `tie_channel_mass=True`; `potential_type="mlp"` (emergent) / `"so2_invariant"` (designed) |
| **langevin_noise** | **`"fdt"`** everywhere (repo default is `"legacy"`, under which none of these laws hold). ME-1 is `T=0` ⇒ deterministic `model.step`, no noise path. |
| **retie** | **on** for every load (`ecommon.load_run` → `retie(to_x64(load_model(...)))`); pytree-level only, no repo edit |
| ε (dt) | **0.05** throughout |
| Δ (escape tolerance) | **0.5 rad** for every `n₁/₂` FPT number |
| γ grid (ME-1) | `np.geomspace(0.002, 0.5, 48)` — bit-identical to `e1c_vcurve.py` |
| write amplitudes δ (ME-1) | **0.05, 0.2, 0.5 rad**; release from rest (`p=0`) at the ring-rotated vacuum |
| rollout length (ME-1) | `n_chunks=8000`, stride ∈ {1,2,4,8,16,32,64,128} chosen so `n_total ≥ max(20 000, 30·n₁/₂^jac)`; cap 1.024e6 steps |
| coset eigenpair rule (ME-1) | `λ_ret = max{|λ_j| : coset overlap ≥ 0.30}` — identical to `e1c_vcurve.py` (the `e1b` C-9 negative is *not* reused) |
| friction field (ME-3) | `FrictionField(k=1, gamma_max=0.9, width=0.25, gate="compact", trainable=False)`; `init_strength = γ_φ`; `init_radius = 50.0` (uniform, stages V/D/X/S) or `1.0` centred on `Ring(0)` (localized, FPT) |
| scalar γ (ME-3) | 0.05; `γ_φ ∈ {0, 0.1, 0.2, 0.3, 0.5}`; scalar control **γ = 0.525 with no field** |
| temperatures (ME-3) | **`T = 4e-3` and `8e-3`** (both `> T* ≈ 3e-3`); designed cross-check at `T = 1e-3` |
| temperature field | **none built** (out of scope) |
| tilt / spurion | `tilt_delta = 0`, `spurion_delta = 0` |
| ensembles (ME-3) | V: 2048 walkers × 40 samples · D: 1024 walkers × 4000 lags · X: 1024 walkers, burn 20k–80k · FPT: 256 walkers, caps 2e5 (outside/scalar) and 1e6 (inside) |
| precision | **float64** (`jax_enable_x64` set in `common.py` before `jnp` binds); weights cast f32→f64; training was f32 |
| seeds | model seeds as listed; PRNG keys derived per (seed, T, γ, γ_φ) — formulas inline in each script |
| env | **main venv** `/Users/user/Desktop/CHLU/.venv` (no worktree sync, w6 lesson): **jax 0.9.0**, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0, matplotlib present; macOS laptop CPU only |
| wall clock | ME-1 emergent 87 s ×2 · ME-1 designed 130 s · ME-3 designed cross-check 452 s · ME-3 emergent V/D/S 924 s · stage X 341 s · FPT 317 s + 1750 s · same-site control 252 s ⇒ **≈ 73 min total compute**. JAX import 11 s (session was warm; **not** the 20-min cold start). |

### 1.1 Commands (cwd `.claude/scratch/v5-vcurve-validation/`, `PYTHONPATH=/Users/user/Desktop/CHLU:/Users/user/Desktop/CHLU/.claude/scratch/v5-gate`, `/Users/user/Desktop/CHLU/.venv/bin/python`)

```
m1_rollout.py --tags emergent150_s42 emergent150_s43 emergent150_s44 \
              --deltas 0.05 0.2 0.5 --out m1_emergent.json                       #  87 s
m1_rollout.py --tags designed150_s42 designed150_s43 designed150_s44 \
              designed150_s45 designed150_s46 --deltas 0.05 0.2 0.5 \
              --out m1_designed.json                                             # 130 s
a1_analyse.py                                                                    # tables 1-5 + fig
m3_vault_emergent.py --seeds 42 43 44 --temps 1e-3 --stages D S \
              --tagpref designed150_s --out m3_designed_crosscheck.json           # 452 s (INSTRUMENT CROSS-CHECK)
m3_vault_emergent.py --seeds 42 43 44 --temps 4e-3 8e-3 --stages V D S \
              --out m3_emergent_VDS.json                                         # 924 s
m3_vault_emergent.py --seeds 42 43 44 --temps 4e-3 --stages X \
              --out m3_emergent_X.json                                           # 341 s
m3_fpt.py --seeds 42 43 44 --temps 4e-3 --cap-in 1000000 --cap-out 200000 \
              --n-walk 256 --out m3_fpt_emergent.json                            # 1750 s
m3_fpt_samesite.py                                                               # 252 s (CONFOUND CONTROL)
a3_analyse.py ; a3_fig.py
```

Artifacts → `.claude/outputs/v5-vcurve-validation/`: `PREREG.md`, `m1_emergent.json`,
`m1_designed.json`, `m3_emergent_VDS.json`, `m3_emergent_X.json`, `m3_fpt_emergent.json`,
`m3_fpt_samesite.json`, `m3_designed_crosscheck.json`, `fig_me1_vcurve_rollout.png`,
`fig_me3_vault_emergent.png`. Scripts + logs → `.claude/scratch/v5-vcurve-validation/`.

### 1.2 Instrument cross-check against the banked run (protocol: never trust a new harness)
| quantity | this report's estimator, on **designed**, `T=1e-3` | `v5-gate` banked |
|---|---|---|
| `D̂` field vault `D(γ_φ=0)/D(γ_φ=0.5)` | **112.58 ± 1.09×** (3 seeds) | 107.77 ± 4.78× |
| `D̂` scalar control vault `D(0.05)/D(0.525)` | **14.16 ± 1.38×** | 13.28 ± 0.12× |
| field/scalar | **8.03 ± 0.80** | 8.11 ± 0.37 |
| `D̂/D_absorb` over all `γ_φ` | **0.9963 ± 0.0126 / 0.9860 ± 0.0073 / 1.0296 ± 0.0287** (s42/43/44) | 1.0011 ± 0.0215 |
| Jacobian `argmin/γ_crit`, emergent | **0.8994 / 0.9046 / 0.9055** | 0.8994 / 0.9046 / 0.9055 (bit-identical) |

---

# ME-1 — the rollout-validated V-curve

## 2. Instruments (defined in `PREREG.md` §1, before measuring)
- **I-J** (banked): `n₁/₂ = ln2 / (−ln|λ_ret|)` from the one-step Jacobian at `(q*, 0)`.
- **I-R1**: rollout, **first** crossing of `|θ| ≤ δ/2` — *the banked instrument, and the one MF-3 quotes.*
- **I-R2**: rollout, **last** crossing (`min{n : max_{m≥n}|θ(m)| ≤ δ/2}`, suffix-max envelope).
- **I-R3 (primary)**: rollout, **envelope-rate fit** — least squares of `log(suffix-max |θ|)` vs `n`
  over the late window `env ∈ [max(1e-6, 30·floor), 0.3δ]`; `n₁/₂ = ln2/Γ_roll`. Fit `R² ≥ 0.9955`
  (median 0.9996) on every emergent cell.
- **I-R3-early**: the same fit over `env ∈ [0.3δ, 0.9δ]` — the finite-amplitude / anharmonicity probe.

## 3. Results

### 3.1 Table 1 — the shape claims on all four instruments (δ = 0.05 rad, 48 γ)
Slopes are on **fixed windows keyed to the Jacobian's `γ_crit`** (`γ < γ_crit/2.5`, `γ > 2.5γ_crit`),
so the four instruments are compared on identical γ-windows.

| checkpoint | γ_crit | **argmin/γ_crit** I-J / I-R1 / I-R2 / **I-R3** | **slope below** I-J / I-R1 / I-R2 / **I-R3** | **slope above** I-J / I-R1 / I-R2 / **I-R3** |
|---|---|---|---|---|
| emergent150_s42 | 0.02334 | 0.8994 / 0.0857 / 0.2068 / **0.8928** | −1.0023 / +0.0725 / −1.0556 / **−0.9651** | +1.1262 / +1.1027 / +1.1027 / **+1.1255** |
| emergent150_s43 | 0.01424 | 0.9046 / 0.1404 / 0.1404 / **0.9047** | −1.0016 / +0.0688 / +0.0688 / **−0.9977** | +1.1031 / +1.0889 / +1.0889 / **+1.1030** |
| emergent150_s44 | 0.02265 | 0.9055 / 0.0883 / 0.3031 / **0.9028** | −1.0022 / +0.0455 / −0.8854 / **−0.9854** | +1.1254 / +1.0977 / +1.0977 / **+1.1231** |
| **mean ± sd** | | **0.9032±0.0027** / 0.1048±0.0252 / 0.2168±0.0668 / **0.9001±0.0052** | **−1.0020±0.0003** / +0.0623±0.0120 / −0.6240±0.4948 / **−0.9827±0.0134** | **+1.1182±0.0107** / +1.0964±0.0057 / +1.0964±0.0057 / **+1.1172±0.0101** |

- **P3 (argmin) HOLDS.** Rollout-rate argmin `0.9001 ± 0.0052 γ_crit` vs Jacobian `0.9032 ± 0.0027`
  — a **0.35% difference**, inside the pre-registered [0.75, 1.05] by a wide margin, 3/3 seeds.
- **P4 (slopes) HOLDS.** Both branches inside their bands ([−1.20, −0.85], [+0.90, +1.35]) 3/3.
- **The pre-registered artifact is confirmed exactly.** I-R1's below-slope was pre-registered as
  "≈ 0, |slope| < 0.35": measured **+0.0623 ± 0.0120**. Its argmin sits at the grid edge
  (`0.105 ± 0.025 γ_crit`) — the C-9 first-crossing artifact, now quantified rather than asserted.
- **I-R2 is not a usable instrument either** (argmin `0.217 ± 0.067 γ_crit`, below-slope
  `−0.62 ± 0.49`): the last-crossing definition removes the quarter-period artifact but carries an
  additive ≤ ½-period bias that diverges as the oscillation slows toward critical damping. Recorded as
  a negative; **use I-R3**.

### 3.2 Table 2 — the ratio curve (γ ≥ 2γ_crit, i.e. the unambiguously overdamped branch)

| checkpoint | δ | n(γ) | `Γ_jac/Γ_R3` (full window) | (asymptotic window) | `d log(ratio)/d log γ` | `n_jac/n_R1` | CV | measured `c` (fit intercept/δ) | early-window `Γ_jac/Γ` |
|---|---|---|---|---|---|---|---|---|---|
| s42 | 0.05 | 21 | 0.9689 ± 0.0008 | **0.9935** | −0.00099 | **1.3307** | 2.9% | 0.638 | 0.754 |
| s42 | 0.2 | 21 | 0.9723 ± 0.0007 | 0.9951 | −0.00081 | 1.2732 | 2.9% | 0.650 | 0.778 |
| s42 | 0.5 | 21 | 0.9725 ± 0.0006 | 0.9952 | −0.00072 | 1.1251 | 2.7% | 0.726 | 0.848 |
| s43 | 0.05 | 25 | 0.9982 ± 0.0000 | **1.0000** | −0.00005 | **1.7981** | 2.2% | 0.644 | 0.698 |
| s43 | 0.2 | 25 | 0.9952 ± 0.0001 | 0.9999 | +0.00004 | 1.9155 | 2.4% | 0.560 | 0.622 |
| s43 | 0.5 | 25 | 0.9904 ± 0.0001 | 0.9999 | +0.00013 | 2.0304 | 3.0% | 0.439 | 0.520 |
| s44 | 0.05 | 21 | 0.9216 ± 0.0027 | **0.9948** | −0.00340 | **1.8204** | 3.4% | 0.327 | 0.539 |
| s44 | 0.2 | 21 | 0.9337 ± 0.0022 | 0.9977 | −0.00269 | 1.8475 | 3.2% | 0.288 | 0.527 |
| s44 | 0.5 | 21 | 0.9340 ± 0.0017 | 0.9983 | −0.00207 | 1.7888 | 3.7% | 0.255 | 0.522 |

- **P1 (rate agreement): holds, with one marginal miss that the falsifier does not touch.** On the
  **asymptotic** window (the deepest half of the fit range in log-amplitude) the three seeds give
  `0.9935 / 1.0000 / 0.9948` at δ=0.05 — **`Γ_jac/Γ_roll = 0.995 ± 0.003`.** On the **full** window
  s44 reads `0.9216 [0.9196, 0.9296]`, i.e. its minimum sits `0.0004` below the pre-registered per-point
  band [0.92, 1.08], and its seed mean is below the [0.95, 1.05] band. **Reported as a miss of the
  literal P1 wording.** The cause is measured, not inferred: the full window includes the early decade
  where a second, faster mode is still present (early-window ratio 0.539 for s44), and the
  window-split diagnostic converges monotonically to 1 as the fit moves into the asymptote.
  **F1 does not fire** on either of its two conditions: seed means 0.92–1.00 ⊂ [0.80, 1.25], and
  `|d log(ratio)/d log γ| ≤ 0.0034` vs the 0.10 threshold.
- **P2 (constant offset): main clause holds, explanatory clause fails.** CV of `n_jac/n_R1` over the
  γ-grid is **2.2–3.7% (F2 threshold 20%)**; per-seed means 1.13–2.03 (s42 at δ=0.5 reads 1.125, just
  under the pre-registered [1.15, 2.10]). **But the single-slow-mode amplitude model I pre-registered
  is wrong**: the measured amplitude fraction `c` (envelope-fit intercept / δ) is 0.638/0.644/0.327,
  which does *not* order like the banked angular overlaps (0.697 < 0.816 < 0.886), and for s44
  `c < 0.5` makes the predicted `ln2/ln(2c)` undefined. The offset is real, constant, and quotable;
  its *microscopic* explanation is a multi-mode early transient, not a single-mode projection.
- **`n_jac/n_R2 = n_jac/n_R1` identically above `2γ_crit`** (the motion is monotone there, so first
  crossing = last crossing) — a consistency check that the two threshold instruments are the same
  instrument on the overdamped branch, and differ only below `γ_crit`.

### 3.3 Table 3 — finite write amplitude (P6, and the answer to referee A1/SF-11)

| checkpoint | δ (rad) | argmin/γ_crit (I-R3) | slope below | slope above | `\|θ_final\|` |
|---|---|---|---|---|---|
| s42 | 0.05 / 0.2 / 0.5 | 0.8928 / 0.9007 / **0.8968** | −0.9651 / −0.9767 / **−0.9833** | +1.1255 / +1.1257 / **+1.1257** | ≤ 2.9e-10 |
| s43 | 0.05 / 0.2 / 0.5 | 0.9047 / 0.9045 / **0.9174** | −0.9977 / −1.0000 / **−0.9919** | +1.1030 / +1.1031 / **+1.1031** | ≤ 4.1e-11 |
| s44 | 0.05 / 0.2 / 0.5 | 0.9028 / 0.9037 / **0.9070** | −0.9854 / −0.9932 / **−0.9920** | +1.1231 / +1.1236 / **+1.1240** | ≤ 6.5e-10 |

**P6's main clause holds:** at a write amplitude 10× larger (0.5 rad = the same Δ used for every FPT
number in V5), the argmin moves by at most **+1.4%** and the slopes by at most **1.9%**. Every δ=0.5
write relaxed **all the way home** (`|θ_final| ≤ 6.5e-10`), i.e. no seed's 0.5-rad write cleared its
washboard barrier. **P6's early-window clause fails**: `Γ_jac/Γ_roll^early = 0.52 … 0.85` (band was
[0.65, 1.35]) — the first half-decade of decay runs up to 1.9× faster than linear response.
**This is mode mixing, not anharmonicity**, and the amplitude sweep proves it: for s44 the early ratio
is 0.539 / 0.527 / 0.522 across a 10× amplitude range, i.e. **amplitude-independent**.

### 3.4 Table 4 — the designed control (P5): the rollout instrument does not manufacture decay
5 seeds × 48 γ × 3 δ × 32 000 steps, `T = 0`:

| statistic | worst case over all 720 cells |
|---|---|
| `max_n \|θ(n) − δ\|` | **9.400e-13 rad** |
| implied `Γ = max\|Δθ\|/(δ·N)` | **≤ 7.309e-17 per step** |
| implied `n₁/₂` | **≥ 9.483e15 steps** |
| Hessian `μ²_soft` on the 5 designed seeds | 1.60e-15, 1.52e-15, −2.92e-16, 2.94e-16, 1.21e-16 |

**P5 holds by ~6 orders of margin** (bound was 1e-6 rad); **F5 does not fire**. This is also the
replacement number for MF-1 (reconciliation item 2): a *rollout-side*, model-side statement about the
designed register that does not depend on any spectral instrument's floor.

### 3.5 Table 5 — the collapsed-variable comparison
`x = γ/γ_crit`, `y = n₁/₂·γ_crit`, interpolated onto a common `x ∈ [0.15, 20]` grid; `μ²_soft` spans
`2.029e-2 … 5.449e-2` across the three emergent seeds.

| instrument | seed-to-seed spread of `y(x)` | `y_min` per seed | continuum prediction |
|---|---|---|---|
| **I-J** (Jacobian) | max **11.09%**, median **1.15%** | 1.579 / 1.527 / 1.499 | `2√2 ln2 = 1.961` at `x = 0.707` |
| **I-R3** (rollout rate) | max **10.51%**, median **5.67%** | 1.631 / 1.535 / 1.662 | idem |

The three emergent curves collapse onto one curve to ~1% (median) on the Jacobian and ~6% on the
rollout, on **both** instruments, with the residual concentrated near the minimum where the parabolic
argmin is least well conditioned. The measured `y_min ≈ 1.5–1.66` sits **22–24% below** the continuum
`2√2 ln2 = 1.961` and the measured `x_min ≈ 0.90` sits **27% above** the continuum `0.707` — both are
the known discrete-map corrections (`t-lever` §4.3 measured `x_min = 0.83–0.93` on the massive mode).
**Figure:** `.claude/outputs/v5-vcurve-validation/fig_me1_vcurve_rollout.png`, panels (a) curves,
(b) per-γ ratio, (c) collapse.

### 3.6 The finding the referee's A1 was reaching for, stated with numbers
The Jacobian V-curve's **left branch is an exact integrator identity and carries zero model
information**: for `γ < γ_crit` the coset eigenpair is complex with `|λ|² = 1−γ`, so
`n₁/₂ = ln2/(−½ln(1−γ))`, *independent of `μ²` and of the checkpoint*. Evidence: at `γ = 0.002`,
emergent s42 (`μ² = 5.449e-2`) and s43 (`μ² = 2.029e-2`) both print `λ_ret = 0.998999499` and
`n₁/₂ = 692.5`, and `2ln2/0.002 = 693.1`. **All the model-dependent content of the V-curve is in the
argmin location (`γ_crit = 2εμ`) and the `+1` branch.** What ME-1 adds is that the *nonlinear model*
actually obeys that identity — which was not guaranteed and is the falsifiable part.

---

# ME-3 — the vault on an emergent checkpoint (`T > T*`)

## 4. Design note: why the `v5-gate` D̂ estimator had to be rebuilt (and how it was validated)
On designed, the coset is exactly flat, so `MSD_θ(n)` grows linearly forever and a fixed block
`L = 500` is a valid diffusion estimator. On emergent the coset is a **bounded** mode
(`μ²_adiab = 2.77e-2 … 8.93e-2`), so `MSD` saturates and the banked estimator is biased low by up to
55%. Two estimators are used here and both are reported:
- **`D̂_lin`** — slope of `MSD(n)` over `[n_lo, n_hi]`, `n_lo = 4/γ_eff` (4 momentum correlation times),
  `n_hi` = the largest lag whose **local exponent `d log MSD/d log n` is still ≥ 0.90**; usable only if
  `n_hi ≥ 2 n_lo`. Both bounds are printed for every cell.
- **`D̂_ou`** (primary on emergent, **window-free**) — 2-parameter fit `MSD(n) = A(1−e^{−n/τ})` over
  `n ≥ n_lo`; `D̂ = A/(2ετ)`, `σ_θ = √(A/2)`. Exact for an OU coordinate; degenerates gracefully to the
  linear estimator as `τ→∞` (the flat-coset limit).

**Both were cross-validated on `designed150_s{42,43,44}` at `T=1e-3` against `v5-gate`'s banked
numbers before being used on emergent** — see §1.2 (112.58 ± 1.09× vs 107.77 ± 4.78×; scalar
14.16 ± 1.38× vs 13.28 ± 0.12×; field/scalar 8.03 ± 0.80 vs 8.11 ± 0.37).

## 5. Q1 — the refrigerator transfers exactly (stage V, `emergent150_s{42,43,44}`)

`Var(p_i)/(M_i T)`, channel dims, 2048 walkers × 40 samples, 3 seeds:

| T | γ_φ | γ_eff | absorb pred | coupled pred | **measured (3 seeds)** | obs/absorb | obs/coupled |
|---|---|---|---|---|---|---|---|
| 4e-3 | 0.0 | 0.050 | 1.00000 | 1.0 | 0.99815 ± 0.00257 | 0.9981 | 0.9981 |
| 4e-3 | 0.1 | 0.145 | 0.36249 | 1.0 | 0.36326 ± 0.00067 | 1.0021 | **0.3633** |
| 4e-3 | 0.2 | 0.240 | 0.23082 | 1.0 | 0.23062 ± 0.00080 | 0.9991 | **0.2306** |
| 4e-3 | 0.3 | 0.335 | 0.17480 | 1.0 | 0.17473 ± 0.00046 | 0.9996 | **0.1747** |
| 4e-3 | 0.5 | 0.525 | 0.12591 | 1.0 | **0.12592 ± 0.00055** | **1.0001** | **0.1259** |
| 8e-3 | 0.5 | 0.525 | 0.12591 | 1.0 | **0.12548 ± 0.00032** | 0.9966 | 0.1255 |

**All 24 field cells (γ_φ>0, both T, 3 seeds): `obs/absorb = 0.9998 ± 0.0019`.** At `γ_φ=0.5`,
`T_local/T = 0.12570` ⇒ **7.955× refrigerator** (predicted 7.942×). **Q1 holds; G1 does not fire.**
The coupled-bath hypothesis (which predicts 1.0 at every `γ_φ`) is rejected on emergent by the same
7.9× as on designed.

## 6. Q2 — the `D̂ ∝ γ_eff^{-2}` law transfers (stage D)

`D̂_ou` vs the absorb-only prediction, restricted to cells whose register is **bounded**
(`σ_θ^ou < 2σ_θ^Boltzmann` — the criterion and the per-cell values are in `m3_emergent_VDS.json`):

| γ_φ | n cells | `D̂_ou / D_absorb` | `D̂_ou / D_coupled` |
|---|---|---|---|
| 0.0 | 1 | 1.0480 | 1.0480 |
| 0.1 | 4 | **1.1026 ± 0.0799** | 0.3997 ± 0.0290 |
| 0.2 | 5 | **1.0158 ± 0.0519** | 0.2345 ± 0.0120 |
| 0.3 | 5 | **1.0552 ± 0.0496** | 0.1845 ± 0.0087 |
| 0.5 | 6 | **1.0418 ± 0.0519** | **0.1312 ± 0.0065** |

**Vault ratios (6 cells = 3 seeds × 2 T):**

| definition | measured | prediction | banked designed (`v5-gate`) |
|---|---|---|---|
| **law-referenced** `D_absorb(γ_φ=0) / D̂(γ_φ=0.5)` | **106.1 ± 5.0×** | 110.25× | 107.77 ± 4.78× |
| measured/measured `D̂(0)/D̂(0.5)` | 297.8 ± 196.8× (median 234.4×) | 110.25× | — |

**Q2 holds on the law-referenced measure** (`[95, 120]` pre-registered; measured 106.1 ± 5.0). The
measured/measured ratio is 2.7× *larger* and highly variable **because the outside arm is not a bounded
register at these temperatures**, not because the hole is stronger than the law: 5 of the 6 `γ_φ=0`
cells fail the bounded test (`σ_θ^ou = 0.42 … 885 rad` vs Boltzmann 0.24 … 0.65). Every `γ_φ ≥ 0.2`
cell passes it. **G2 does not fire** on the law-referenced measure, which is the one whose mass bias is
controlled by construction.

## 7. Q4 — the scalar-γ laundering control: **G4 fires**, and the failure mode is the control

| T | seed | scalar `Var(p)/MT` | `D̂_ou(scalar 0.525)` | `D̂_ou/D_flat-prediction` | field/scalar |
|---|---|---|---|---|---|
| 4e-3 | 42 | 0.99353 | 3.3725e-04 | **1.1768** | 9.14 |
| 4e-3 | 43 | 0.95791 | 1.5389e-03 | **3.7319** | 25.75 |
| 4e-3 | 44 | 1.00183 | 1.2244e-03 | **3.3782** | 26.96 |
| 8e-3 | 42 | 0.96666 | 1.6048e-03 | **2.7999** | 21.70 |
| 8e-3 | 43 | 1.01762 | 4.3121e-03 | **5.2284** | 41.31 |
| 8e-3 | 44 | 1.01974 | 1.4865e-03 | **2.0506** | 15.48 |

- **The half of Q4 that transfers:** a scalar γ does **not** cool —
  `Var(p)/(M T) = 0.99288 ± 0.02354` at `γ = 0.525` with no field (both hypotheses predict 1.0). The
  *mechanism* statement ("a friction field is not locally stronger friction; it is locally stronger
  friction **plus** a local refrigerator") therefore stands on emergent: same `γ_eff`, momentum
  variance 1.0 vs 0.126.
- **The half that does not:** **field/scalar = 23.39 ± 10.06** (pre-registered [6.5, 9.5]; G4's
  falsifier > 14). **G4 FIRES.** The named failure mode: the *control arm itself* is off its own law by
  `D̂/D_flat = 1.18 … 5.23`, i.e. at `γ = 0.525` with no cooling the emergent coset is still
  anharmonically delocalised (hop fractions 0.73% / 10.2% / 0.26%, §8), while the *field* arm sits on
  the absorb law at 1.02–1.15. **The identical estimator on designed returns 8.03 ± 0.80.** ⇒ the
  contrast *number* is a designed-only quantity; the contrast *direction* transfers.

## 8. Q3 — the stationary spread: **G3 fires**, and the hole-confines-the-register statement replaces it

`T = 4e-3`, 1024 walkers, burn 20 000–80 000 steps (≥ 15 relaxation times), IQR estimator
(`σ = IQR/1.349`, insensitive to hoppers); "hop" = `|θ| > 1` rad:

| seed | arm | γ_eff | `σ_θ` (sd) | `σ_θ` (IQR) | **hop fraction** | Boltzmann @ `T_local` | obs/pred |
|---|---|---|---|---|---|---|---|
| 42 | no hole, γ=0.05 | 0.050 | 0.68374 | 0.40795 | **5.50%** | 0.24073 | 1.6946 |
| 42 | scalar 0.525 (control) | 0.525 | 0.26888 | 0.20034 | **0.73%** | 0.24073 | 0.8322 |
| 42 | **γ_φ hole** | 0.525 | 0.07423 | **0.06442** | **0.0000** | 0.08542 | 0.7542 |
| 43 | no hole | 0.050 | 2.57066 | 1.17601 | **42.97%** | 0.46059 | 2.5533 |
| 43 | scalar 0.525 | 0.525 | 1.19816 | 0.35335 | **10.20%** | 0.46059 | 0.7672 |
| 43 | **γ_φ hole** | 0.525 | 0.09889 | **0.08628** | **0.0000** | 0.16343 | 0.5279 |
| 44 | no hole | 0.050 | 0.49338 | 0.36922 | **2.36%** | 0.24034 | 1.5363 |
| 44 | scalar 0.525 | 0.525 | 0.26177 | 0.21574 | **0.26%** | 0.24034 | 0.8977 |
| 44 | **γ_φ hole** | 0.525 | 0.08168 | **0.07279** | **0.0000** | 0.08528 | 0.8535 |

- `σ_θ(in)/σ_θ(out) = 0.1428 ± 0.0516` vs the pre-registered `0.3548 ± 0.020` ⇒ **G3 FIRES.**
- **Why, stated by the control, not by me:** the scalar control must return exactly 1.000 (an
  equilibrium spread cannot depend on γ). It returns **0.4586 ± 0.1181.** ⇒ the outside arm has not
  reached a stationary distribution and is not one; at `γ = 0.05` the emergent coset at `T = 4e-3` is a
  **hopping rotor** (2.4–43% of sampled states beyond 1 rad), so the pre-registered ratio was measured
  against a non-existent reference. The Q3 comparison is **not measurable on these checkpoints at
  `T > T*`.**
- **The statement that is measurable, and has no designed analogue:** with the hole on, the hop
  fraction is **0.0000 on 3/3 seeds** and the spread sits at `0.53–0.85` of its Boltzmann value at
  `T_local`; with the *same* `γ_eff` and no cooling it is 0.26–10.2%. **The refrigerator converts a
  hopping rotor into a bounded register**, and the laundering control shows friction alone does not.

## 9. Q5 — the direct FPT vault, with the confound control ME-3 needed

**The confound (new; not present on designed):** the emergent potential has **no rotational symmetry**,
so `v5-gate`'s "outside" arm at `θ = π` is not a vacuum:

| seed | `\|∇V\|` at `θ=0` | `\|∇V\|` at `θ=π` | `V(π) − V(0)` | `μ²_adiab` at 0 | at π |
|---|---|---|---|---|---|
| 42 | 0.0000 | 0.0420 | +0.03163 | 7.041e-2 | 7.278e-2 |
| 43 | 0.0000 | 0.1049 | +0.06003 | 2.767e-2 | 3.018e-2 |
| 44 | 0.0000 | 0.0414 | +0.04315 | 8.933e-2 | **4.154e-2** |

**FPT to `|Δθ| ≥ 0.5`, `T = 4e-3`, γ = 0.05, 256 walkers** (localized compact hole radius 1.0 centred on
`Ring(0)`; `γ_φ` verified at 0.500000 at the hole centre **and at the exit edge**, 0.000000 at `θ=π`):

| seed | `θ=π`, no hole | `θ=0`, **no hole** (same-site control) | `θ=0`, **hole** | censored | **same-site vault** | scalar control (`γ=0.525`, θ=π) |
|---|---|---|---|---|---|---|
| 42 | 1245 [930, 2203] | **725 [590, 880]** | **> 1e6** | 86.7% | **> 1379×** | 3995 → 3.21× |
| 43 | 295 [230, 365] | **255 [225, 295]** | **9050 [5950, 12556]** | 0.0% | **35.5×** | 2370 → 8.03× |
| 44 | 15585 [9315, 21200] | **775 [680, 950]** | **> 1e6** | 93.4% | **> 1290×** | 4280 → 0.27× |

- **Q5 holds on 2/3 seeds** as censoring-limited lower bounds (`> 1379×`, `> 1290×`; pre-registered
  `> 300×` and "most likely fully censored" — it was, at 86.7% and 93.4% at a 1e6-step cap).
- **G5 fires on s43** (35.5× < 110×). s43 is the seed `v5-gate` already flagged as geometrically
  atypical (its L2): softest `μ²`, shallowest barrier (6.8e-3), and at `T=4e-3` its coset is
  `σ_θ/Δ = 0.92` outside and still `0.33` inside — it never leaves the diffusive regime, so it gets no
  Kramers enhancement.
- **The `θ=π` baseline is unusable on emergent** and the same-site control is what should be quoted:
  s44's `θ=π` number (15585) is 20× its own `θ=0` number (775), and its scalar control reads **0.27×**
  (raising γ 10.5× makes escape *faster*) — a physical impossibility for a stationary register, and the
  signature of launching walkers on a slope.

**Figure:** `.claude/outputs/v5-vcurve-validation/fig_me3_vault_emergent.png` — (a) Q1, (b) Q2 with the
designed overlay, (c) Q3 confinement + hop fractions, (d) Q5 same-site FPT.

## 10. Pre-registered verdict rules, applied mechanically
| rule | outcome |
|---|---|
| ME-1: *"Jacobian instrument validated with a stated bias"* iff **P1 ∧ P3 ∧ P4 ∧ P5** | **P3 ✅ P4 ✅ P5 ✅; P1 ✅ on its asymptotic form and its falsifier F1, ⚠ marginal miss (0.9216 vs 0.92) on the literal full-window wording for s44.** No falsifier F1–F5 fires. |
| ME-1: *"the V-curve shape itself is unreliable"* iff **F1 ∨ F3 ∨ F4** | **does not fire.** |
| ME-1 stated bias to carry with every quote | threshold instruments read **1.13–2.03× short** (per-seed constant, CV 2.2–3.7%); rates agree to **0.5 ± 0.3%**. |
| ME-3: *"the vault transfers"* iff **Q1 ∧ Q2 ∧ Q4** | **Q1 ✅ Q2 ✅ Q4 ✗ (G4 fired).** |
| ME-3: *"designed-only scope confirmation"* iff **G1 ∨ G2 ∨ G4** | **fires, via G4 only.** Named failure mode (required by the rule): **the scalar-γ laundering control is not measurable on an emergent register at `T > T*`** — its own `D̂` sits 1.18–5.23× above the flat-coset law because a scalar γ does not confine the mode. The two laws the number is built from (Q1 bath, Q2 diffusion) both transfer. |
| ME-3 bonus statements | **G3 fires** (with its own control explaining why); **Q5 holds 2/3, G5 fires on s43.** |

---

## 11. Limitations & confounds (honest)
- **L1 — ME-1 is a `T=0`, `p=0` release experiment**, as the banked curve is. It validates the
  *instrument*, not the thermal (`T>0`) claims, which remain on the `v5-gate` FPT footing.
- **L2 — the collapse spans only 2.7× in `μ²`** (2.03e-2 … 5.45e-2 across the 3 emergent seeds). The
  "eleven decades" span in the draft is **one measured decade of emergent variation attached to the
  designed `μ→0` corner**, exactly as the referee wrote; ME-1 does not change that, it only shows the
  corner is real to `n₁/₂ ≥ 9.5e15` steps by direct rollout.
- **L3 — the P1 marginal miss on s44 is a fit-window effect, and I chose the windows.** The
  asymptotic-window numbers (0.9935 / 1.0000 / 0.9948) are the ones I would quote; the full-window
  numbers are also printed so the choice is auditable. `m1_emergent.json` carries both plus the
  per-γ split.
- **L4 — ME-3's `T=8e-3` cells are largely out of regime.** 5 of 6 `γ_φ=0` cells and all of s43's
  low-`γ_φ` cells are delocalised (`σ_θ` up to 885 rad — the walker circulates). Only cells passing the
  printed bounded test support a law claim. `v5-gate`'s own C6 already restricts trustworthy cells to
  `T ≤ 3e-3`; every `T > T*` measurement is therefore squeezed between `T*` and anharmonicity.
- **L5 — `emergent150_s43` remains the atypical seed** and it is the one that fires G5. n=3 emergent
  seeds is not enough to say whether 35.5× or >1300× is typical.
- **L6 — the FPT inside arms are 86.7% / 93.4% censored at a 1e6-step cap**, so `>1379×` and `>1290×`
  are lower bounds, not measurements. Closing them needs ~1e7 steps/walker.
- **L7 — the localized hole is a ball of radius 1.0 in the full 4-D space**, so radial/spectator
  excursions of order 1.0 (≈ 4.5 `σ_r` at `T=4e-3`) leave it. This is a plausible contributor to s43's
  low vault and is not separately measured here.
- **L8 — single PRNG stream per cell.** All quoted spreads are across *model* seeds (3 emergent, 5
  designed), not across noise realizations.
- **L9 — no `T_φ` was built** (out of scope), so the `{γ_φ, T_φ} × {T=0, T>0}` 2×2 remains half
  prediction.

## 12. Recommended next experiments
| id | experiment | cost | why |
|---|---|---|---|
| **N1** | **Widen the V-curve collapse**: run `m1_rollout.py` on `emergent_s{42,43,44}` and `brokeniso150_s{42,43,44}` (already on disk, no training) to spread `μ²` beyond 2.7×. | **~5 min** | the collapse is the paper's portable claim and currently rests on 2.7× in `μ²`; this is free. |
| **N2** | Close the FPT censoring: s42/s44 inside arm at cap 1e7, 128 walkers, stride 500. | ~40 min | turns `> 1379×` into a number; the 2/3-seed Q5 result is currently a bound. |
| **N3** | Repeat ME-3 at **`T = 3.2e-3`** (just above `T*`, below the delocalisation onset) with the same-site FPT design, 5 emergent seeds if more exist. | ~30 min | the only regime where both arms are bounded; would let Q3/Q4 be measured rather than voided. |
| **N4** | Same-site FPT control on **designed** (`θ=0` no-field vs hole) to confirm `v5-gate`'s `θ=π` baseline was harmless there (it should be, by SO(2) symmetry — but it was never checked). | ~10 min | closes reconciliation item 3 for the banked designed numbers too. |
| **N5** | ME-2 (the parked minus-the-physics twin, G2 control) on the *rollout* instrument now that it is validated. | medium | the referee's next-candidate; the instrument question is now settled. |

## Git footprint
**None.** No tracked file created, modified, or deleted; `git status --short` clean at start and end;
HEAD unchanged at `7fcef50` throughout. Scripts →
`.claude/scratch/v5-vcurve-validation/{m1_rollout,a1_analyse,m3_vault_emergent,m3_fpt,m3_fpt_samesite,a3_analyse,a3_fig}.py`
(reusing `.claude/scratch/v5-gate/ecommon.py` and `.claude/scratch/t-lever-forgetting/{common,kernels}.py`
verbatim via `PYTHONPATH`). One superseded artifact was **deleted** before filing: `m3_smoke.json` (a
smoke run whose stage-D estimator is the broken pre-rebuild version, §4) — it is not in the output dir
and must not be resurrected from the scratch logs.

---

## Proposed handover updates (for the Hub)

### §1.6 — new CM sub-entry: the V-curve is instrument-independent (ME-1)
> **CM-16(c-unification) is now ROLLOUT-VALIDATED (`v5-vcurve-validation`, 2026-08-19; `7fcef50`;
> `langevin_noise="fdt"` n/a (T=0); retie; ε=0.05; δ_write ∈ {0.05,0.2,0.5} rad; γ-grid
> geomspace(0.002,0.5,48); float64).** Direct `T=0` rollout of the trained model, envelope-rate
> instrument, 3 emergent seeds: **argmin `0.9001 ± 0.0052 γ_crit`** vs the Jacobian's
> `0.9032 ± 0.0027`; **slopes `−0.9827 ± 0.0134` / `+1.1172 ± 0.0101`** vs `−1.0020 ± 0.0003` /
> `+1.1182 ± 0.0107`; decay **rates** agree to **`Γ_jac/Γ_roll = 0.995 ± 0.003`** in the asymptotic
> window, with **`|d log(ratio)/d log γ| ≤ 0.0034`**. Shape survives to a **10× larger write amplitude
> (δ = 0.5 rad)**: argmin moves ≤ 1.4%, slopes ≤ 1.9%. **Collapse:** `(γ/γ_crit, n₁/₂γ_crit)` collapses
> the 3 seeds to **1.15% median (I-J) / 5.67% median (I-R3)**; `y_min ≈ 1.50–1.66` vs continuum
> `2√2 ln2 = 1.961`, `x_min ≈ 0.90` vs continuum 0.707 (the known discrete-map correction).
> **Designed control, 5 seeds × 48 γ × 3 δ × 32 000 steps: `max|θ−δ| = 9.40e-13` rad ⇒ `n₁/₂ ≥ 9.5e15`
> steps** — the rollout instrument manufactures no decay at the `μ→0` corner.

### §1.6 / §5 — **MF-3's premise must be corrected before it is written into V5**
> **The "19–43% Jacobian/rollout disagreement" (`v5-gate` §3.2, → V5 referee MF-3) is a constant
> amplitude offset on a *threshold* instrument, not a rate disagreement.** `n_jac/n_R1` =
> **1.331 / 1.798 / 1.820** (s42/43/44) with **coefficient of variation 2.2–3.7%** over 21–25 γ-points
> and `d log(ratio)/d log γ ≈ 0`. Cause: the ring write deposits only a fraction of its amplitude in the
> slow branch, so a first-crossing of `δ/2` fires early by a γ-independent factor; the *rates* agree to
> 0.5 ± 0.3%. **The MF-3 sentence should read "a constant per-checkpoint offset of 1.13–2.03× on the
> threshold instrument, with the decay rates agreeing to 0.5%", not "the instruments disagree by
> 19–43%".** ⚠ The banked `e1c:n_half_rollout` argmin (0.002 = grid edge, 3/3 seeds) is the **C-9
> first-crossing artifact quantified**: its below-`γ_crit` slope is `+0.0623 ± 0.0120` where the true
> slope is `−1`. **Never quote a rollout `n₁/₂` from a first- or last-crossing threshold below `γ_crit`.**

### §1.6 / §5 — **a replacement number for MF-1's instrument floor**
> The designed coset's `T=0` lifetime now has a **rollout-side, model-side bound** that needs no
> spectral instrument: over 5 designed seeds × 48 γ × 3 write amplitudes × 32 000 steps,
> `max|θ(n) − δ| = 9.400e-13` rad ⇒ `Γ ≤ 7.31e-17`/step ⇒ **`n₁/₂ ≥ 9.5e15` steps**. Quote this where
> "μ² = 1.7e-12" (a ring-profile *instrument floor*, per MF-1) currently appears.

### §1.6 / §5 — **CM-17 (the vault) splits like CM-16 did**
> **CM-17 REFINED by `v5-vcurve-validation` (3 emergent seeds, `T ∈ {4e-3, 8e-3}` both `> T* ≈ 3e-3`;
> pre-registered; estimator cross-validated on designed at `T=1e-3` → `112.58 ± 1.09×` vs the banked
> `107.77 ± 4.78×`).**
> - **(a) The bath law TRANSFERS EXACTLY.** `Var(p_i)/(M_i T)` / absorb prediction = **0.9998 ± 0.0019**
>   over 24 emergent field cells; `T_local/T = 0.12570` ⇒ **7.955× refrigerator** (pred 7.942×). The
>   coupled-bath hypothesis is rejected on emergent by the same 7.9×.
> - **(b) The `D̂ ∝ γ_eff^{-2}` law TRANSFERS.** On bounded cells `D̂/D_absorb = 1.016 … 1.103`
>   (per-`γ_φ`); `D̂/D_coupled = 0.1312 ± 0.0065` at `γ_φ=0.5`. **Law-referenced vault
>   `106.1 ± 5.0×`** (pred 110.25×, banked designed 107.77 ± 4.78×).
> - **(c) ⛔ The FIELD/SCALAR CONTRAST NUMBER DOES NOT TRANSFER.** Measured **23.39 ± 10.06** on
>   emergent vs **8.03 ± 0.80** from the identical estimator on designed (banked 8.11 ± 0.37).
>   **Failure mode: the scalar-γ control arm is itself off its own law by 1.18–5.23×** — at `γ = 0.525`
>   with no cooling the emergent coset is still anharmonically delocalised. The *direction* ("a field is
>   not locally stronger friction; it is friction **plus** a refrigerator") transfers, evidenced by
>   `Var(p)/(M T) = 0.99288 ± 0.02354` for the scalar control vs 0.126 for the field at the same
>   `γ_eff`. **Quote "8.11 ± 0.37" as designed-only.**
> - **(d) NEW, emergent-only, no designed analogue: the hole CONFINES the register.** Hop fraction
>   (`|θ| > 1` rad, `T=4e-3`): **5.5% / 43.0% / 2.4%** with no hole → **0.0000 / 0.0000 / 0.0000**
>   inside the hole, with the **scalar control at the same `γ_eff` still hopping (0.73% / 10.2% /
>   0.26%)**. ⚠ The pre-registered `σ_θ(in)/σ_θ(out) = 0.355` **failed (0.1428 ± 0.0516)** because the
>   outside arm is a non-stationary hopping rotor — the scalar control, which must return 1.000, returns
>   **0.4586 ± 0.1181**. **Do not quote an emergent `σ_θ` ratio.**
> - **(e) FPT vault, same-site: `> 1379×` (s42), `35.5×` (s43), `> 1290×` (s44)** at `T=4e-3`, Δ=0.5,
>   caps 1e6 (86.7% / 0% / 93.4% censored). 2/3 seeds far exceed the designed 87–110×; **s43 (the
>   `v5-gate`-L2 atypical seed) comes in below it at 35.5×**.

### §7 (known issues) — **a design bug in the banked T6 harness, for emergent use**
> **`v5-gate`'s R3/T6 "outside" arm at `θ = π` is not a vacuum on an emergent checkpoint.** Measured
> `|∇V| = 0.0420 / 0.1049 / 0.0414` at `θ=π` vs `0.0000` at `θ=0`, with `V(π) − V(0) = +0.032 … +0.060`
> and `μ²_adiab` differing by up to **2.15×** between the two sites (s44). Consequence: s44's `θ=π`
> FPT is 20× its own `θ=0` FPT, and its scalar control reads **0.27×** (raising γ 10.5× apparently
> *speeding* escape) — impossible for a stationary register. **Any emergent FPT must use a same-site
> (`θ=0`, no-field) baseline.** Harmless on designed by SO(2) symmetry, but **never checked there** —
> see N4.

### §8 — **funded, cheap follow-ups**
> **N1 (≈5 min, no training):** widen the V-curve collapse using `emergent_s{42,43,44}` /
> `brokeniso150_*`, already on disk — the collapse currently spans only 2.7× in `μ²` and is V5's most
> portable claim. **N3 (≈30 min):** repeat ME-3 at `T = 3.2e-3`, the only window where both arms are
> bounded, so Q3/Q4 become measurable instead of void. **N4 (≈10 min):** same-site FPT control on
> designed, to certify the banked `θ=π` baseline.

## Flags
- **For `experiment-engineer` (no bug in `chlu/`, but two harness-level items):**
  1. **The `AttributeError` on `friction_field` for pre-field checkpoints is still live** (`v5-gate`
     reported it; I hit it again and used the same `object.__setattr__` on a `copy.copy` workaround).
     A supported `with_friction_field(model, field)` helper, or a backfill on load in
     `chlu/utils/checkpoints.py`, would remove the workaround from three separate scratch harnesses.
  2. **`v5-gate`'s stage-D block estimator (`L = max(500, 20/γ_eff)`) is only valid for an exactly flat
     coset.** On a bounded mode it is biased low by up to 55% at `γ = 0.05`. If any of that machinery is
     ever promoted into `chlu/utils/metrics.py`, it must carry the local-exponent gate and/or the OU fit
     used here (`m3_vault_emergent.py::_msd_analyse`), both cross-validated against the banked designed
     numbers in §1.2.
- **For the Hub:** the four reconciliation items at the top of this report each need an owner. Items 1
  and 2 are **writer** work (they change what MF-1 and MF-3's fix sentences say, and the task file
  states results fold at v0.4, not v0.3). Items 3 and 4 are **registry/negatives** work — they are
  never-quote candidates in the same class as the `n₁/₂`-without-`Δ` rule.
- **No verdict language used beyond the prereg's own success/failure lines** (task rule 1). Adjudication
  of "does the vault transfer" — given Q1 ✅, Q2 ✅, G4 fired with a named control-side failure mode — is
  the Advisor's / Head's, not mine.
