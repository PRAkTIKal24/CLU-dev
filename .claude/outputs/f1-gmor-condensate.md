# f1-gmor-condensate — experiment-engineer report

**Task + acceptance criterion:** ship a `LinearSpurionPotential` (linear ambient ChPT spurion) and use it to demonstrate **GMOR proper** — `μ²F² = δΣ` to machine precision with an independently *measured* condensate Σ, plus the resonance-saturated NLO coefficient — on **trained** SO(2) checkpoints, probe-only, no retraining; full suite green.

**Status: done.** All four items delivered. GMOR verified on 8 trained checkpoints × 10 δ: **max absolute deviation `1.33e-15`** (f64 machine floor), relative deviation `≤ 2.7e-14` wherever the probe is not roundoff-limited. NLO/LEC ratio → **`0.99999607`** (mean, δ≤1e-5), with `ratio = 1 − 1.05·x` corrections. Suite: **200 passed, 1 skipped**.

---

## What I did

1. **`LinearSpurionPotential(eqx.Module)`** in `chlu/core/potentials.py`: `V(q) = V_base(q) − δ·(u·q)`. Mirrors `TiltedPotential` (static `spurion_delta`, composable over any trained checkpoint via `eqx.tree_at`). Added `channel_spurion_direction(dim, angle)` next to `so2_generator` to build `u` inside the channel plane (zero spectator support ⇒ the exact channel/spectator decoupling of `SO2InvariantPotential` survives the breaking). Unlike the tilt, the gradient is globally smooth (no `atan2` singularity at the origin).
2. **Config-wired** (not hardcoded): `ExperimentDConfig.spurion_{delta,angle}` → `run_experiment_d` → `CHLU.__init__`, mirroring the `tilt_delta`/`tilt_n` chain exactly. **Defaults `0.0` ⇒ no wrapper constructed, every existing checkpoint/experiment bit-compatible.**
3. **Probe** (`.claude/scratch/f1-gmor-condensate/gmor_condensate.py`) reusing the `v2-full-runs` `probe_common` harness verbatim (x64, `polish_vacuum`, `spectrum_probe`, `angular/radial_mode_index`). Three *independent* measurements per (checkpoint, δ).
4. **Contrast + figure + analysis** (`analyze_and_figure.py`): the shipped angular tilt applied to the *same* checkpoints, to substantiate "the tilt cannot see Σ".

### The vacuum solve (why it reaches machine precision)
`V_base` of `SO2InvariantPotential` is exactly channel-invariant and channel/spectator-decoupled, so with `u` in the channel plane the tilted vacuum is exactly `q*(δ) = r*(δ)·u + q_spec` with `q_spec` **δ-independent**. I solve the 1D root `u·∇V_base(r u + q_spec) = δ` by Newton (residual `≤ 4.4e-16`) and **cross-check structure-agnostically** with a BFGS polish of the full spurioned potential: `max |r*_BFGS − r*_Newton| = 0.0` (bit-identical), BFGS `|∇V| ≤ 2.6e-13`.

---

## How I verified

```
.venv/bin/python -m pytest -q --no-cov                    → 200 passed, 1 skipped, 186.79s
.venv/bin/python -m pytest tests/test_goldstone.py -k spurion → 4 passed
.venv/bin/python -m ruff check chlu/ tests/               → All checks passed!
PYTHONPATH=… .venv/bin/python .claude/scratch/f1-gmor-condensate/gmor_condensate.py
PYTHONPATH=… .claude/scratch/f1-gmor-condensate/analyze_and_figure.py
```
*(Note: `ruff format --check chlu/core/chlu_unit.py` reports "would reformat" — **verified pre-existing on `main`** via `git stash`; I did not reformat, per §3.3.)*

---

## Findings

### 1. The one-identity table — `μ²F² = δ·Σ`, three measurements, one exact law
8 trained checkpoints (5 designed150 + 3 anchored3000), `u = e₀`, mean over checkpoints:

| δ | Σ = r* (mean) | μ²F² (mean) | max rel dev | max **abs** dev |
|---|---|---|---|---|
| 1e-8 | 0.95186937111604 | 9.5186937286e-09 | 1.10e-08 | **1.01e-16** |
| 1e-7 | 0.95186948174437 | 9.5186948223e-08 | 1.87e-09 | **1.85e-16** |
| 1e-6 | 0.95187058802597 | 9.5187058802e-07 | 1.87e-10 | **1.86e-16** |
| 1e-5 | 0.95188165067351 | 9.5188165068e-06 | 2.18e-11 | **2.16e-16** |
| 1e-4 | 0.95199226031843 | 9.5199226032e-05 | 1.34e-12 | **1.33e-16** |
| 1e-3 | 0.95309668422658 | 9.5309668423e-04 | 2.27e-13 | **2.09e-16** |
| 1e-2 | 0.96398369127653 | 9.6398369128e-03 | 2.69e-14 | **2.69e-16** |
| 3e-2 | 0.98732513581811 | 2.9619754075e-02 | 9.66e-15 | **2.71e-16** |
| 1e-1 | 1.06435252346294 | 1.0643525235e-01 | 1.55e-15 | **1.53e-16** |
| 3e-1 | 1.60757655208590 | 4.8227296563e-01 | 1.55e-15 | **1.33e-15** |

**`max |μ²F² − δΣ| = 1.33e-15` over all 80 (checkpoint, δ) pairs** — the f64 floor, flat in δ.

> **⚠ Honest precision statement (important for the paper — do not quote "2.2e-16 relative").**
> The *relative* deviation grows as `ε/δ` at small δ. This is **not** a law failure but the roundoff floor of the **autodiff Hessian**: the angular curvature is `K_ang = δ/r*`, reconstructed as a difference of `O(‖K‖)` terms, so its absolute error is `~ε‖K‖` and its relative error `~ε/δ`. I proved this directly (`precision_floor.py`): on the Mexican hat, an **analytic cancellation-free** Hessian gives relative deviation **`1.1e-16 – 2.2e-16` at every δ from 1e-8 to 0.3** (reproducing the toy's `2.2e-16`), while the autodiff Hessian on the *same* potential gives `2.28e-8 → 3.31e-10 → 6.44e-13 → 2.68e-14 → 4.22e-15` for `δ = 1e-8 … 0.3` — exactly `∝1/δ`. The absolute deviation is flat at `≤1.6e-15` throughout, always under the `ε·‖K‖·F²` floor.
> **Correct claim: GMOR holds to machine precision in absolute terms at every δ; relative exactness (≤2.7e-14) is demonstrated for δ ≥ 1e-2, and the small-δ residual is a measured `ε/δ` probe artifact, not physics.**

### 2. Σ is genuinely *measured*, three ways
- `Σ_geom = r*(δ)` (vacuum radius / order parameter)
- `Σ_HF = u·q*(δ)` (Hellmann–Feynman): **`max |Σ_HF − Σ_geom| = 0.0`** (exact)
- `Σ_FD = −dE_vac/dδ` (central finite difference, h=1e-6, re-solving the vacuum at δ±h — the *envelope-theorem* definition, structurally independent of the Hessian): **`max |Σ_FD/Σ_geom − 1| = 9.2e-11`** (FD-limited)

So `Σ = −∂E_vac/∂δ = r*` is confirmed on trained nets, and `μ²` (autodiff Hessian) and `F² = M_ch r*²` are separate instruments. **Three independent measurements, one identity.**

### 3. The shipped angular tilt provably cannot see Σ (the reason F-1 exists)
Applying `TiltedPotential(δ, n=1)` to the *same* checkpoints at δ ∈ {1e-4, 1e-2, 0.3}:

- **`max |r*_tilt − f| = 2.22e-15`** over all (ckpt, δ) — the condensate does not move *at all*.
- **`max |μ²F²/(δn²) − 1| = 1.03e-11`** — the tilt measures only the **product**; "Σ" degenerates to the pure number `n²`.
- Under the linear spurion, by contrast, **Σ runs by +16.1% … +210.1%** across the δ grid (per-checkpoint: a3000 s42/s43/s44 = +19.8/+17.4/+16.1%; d150 s42…s46 = +210.1/+169.3/+35.9/+34.4/+41.1%).

### 4. NLO — the leading LEC is saturated by the radial (σ/Higgs) resonance
Predicted `(μ²_LO − μ²)/μ² = x ≡ δ/(M_ch·μ_rad²·f)`, with `μ²_LO ≡ δΣ(0)/F²(0)`.

- **`ratio = 0.99999607` (mean over 8 ckpts, δ ≤ 1e-5), max |dev| = 2.20e-05.**
- From the **measured** `μ²` (not the geometric shortcut), δ∈[1e-4,1e-2]: ratio ∈ **[0.97917, 0.99992]** (cf. the toy's `0.9959`).
- Corrections are first-order in the expansion parameter: **`d(ratio)/dx = −1.0504`** at small x ⇒ `ratio = 1 − O(x)`, exactly as ChPT demands of a *leading* LEC.
- **Breakdown is visible and expected:** at δ=0.3 the softest-radial seed `designed150_s42` (`μ_rad² = 0.670`) has `x = 0.68` — not small — and `Σ` has run +210%. The expansion is uncontrolled there; do not quote δ=0.3 for the NLO claim. (This is the same "softest seed anharmonicity" the deep-dive §2.4 predicted.)

### 5. Direction independence (free consistency check)
`u` at angles {0.0, 0.7, −2.3} on `designed150_s42`, δ=1e-2: **μ² relative spread `6.21e-15`** — as required by exact channel invariance of `V_base`.

---

## Flag provenance (§5, mandatory)

| item | value |
|---|---|
| **branch / commits** | `agent/experiment-engineer/f1-gmor-condensate` @ `9bc2cf7` (base local `main` `27f232f`) |
| **checkpoints (probe-only, NO retraining)** | `designed150_s{42,43,44,45,46}` (`.claude/scratch/v2-full-runs/runs/<tag>/models/exp_d_chlu.pkl`) · `anchored3000_s{42,43,44}` (`.claude/scratch/v2-referee-experiments/anchored3000/anchored_l100_s{seed}_ep3000.pkl`) |
| **training config (inherited, unchanged)** | designed150: `potential_type=so2_invariant`, `kinetic=newtonian_learned`, `tie_channel_mass=True`, `tilt_delta=0`, `sleep_mode=on`, 150 ep · anchored3000: same + anchor `λ=100`, 3000 ep |
| **seeds** | 42–46 (designed), 42–44 (anchored); probe is deterministic (no PRNG draw) |
| **spurion** | `LinearSpurionPotential`, `u = channel_spurion_direction(4, 0.0) = e₀`; direction check also at angles 0.7, −2.3 |
| **δ grid** | `[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]` (task's `[1e-8, 0.3]`) |
| **precision** | `jax_enable_x64=True` (set in `probe_common`); f32-trained weights cast to f64 via `to_x64`; JAX **0.9.0**, equinox 0.13.4, main venv (`/Users/user/Desktop/CHLU/.venv`), no `uv sync` in a worktree |
| **vacuum solve** | Newton on `u·∇V_base = δ`, residual `≤4.4e-16`; BFGS cross-check `gtol=1e-12`, `|r*_BFGS − r*_Newton| = 0.0`, `|∇V|_BFGS ≤ 2.6e-13` |
| **grad_norm at probe point** | `≤ 6.65e-09` (full-vector; dominated by the **spectator** BFGS residual, which is exactly decoupled from the channel Hessian block — channel residual is `4.4e-16`) |
| **Σ_FD step** | `h = 1e-6`, central difference, δ ≥ 1e-4 only |
| **not swept** | dt, γ, kinetic mode — this probe touches only `V` and its Hessian (no rollouts, no integrator) |

**Per-checkpoint constants** (untilted vacuum):

| ckpt | f = Σ(0) | M_ch | F²(0) | μ_rad² | flat μ² |
|---|---|---|---|---|---|
| anchored3000_s42 | 0.925407523 | 0.32827 | 0.28112 | 4.02208 | −7.61e-16 |
| anchored3000_s43 | 0.908626794 | 0.23096 | 0.19068 | 6.31577 | −1.26e-15 |
| anchored3000_s44 | 0.918129380 | 0.38180 | 0.32184 | 4.48295 | 2.22e-15 |
| designed150_s42 | 0.966992651 | 0.68371 | 0.63932 | 0.67030 | 1.60e-15 |
| designed150_s43 | 0.966262586 | 0.68441 | 0.63901 | 0.77089 | 1.52e-15 |
| designed150_s44 | 0.979706476 | 0.66471 | 0.63800 | 1.19012 | −2.92e-16 |
| designed150_s45 | 0.959136141 | 0.66038 | 0.60751 | 1.09270 | 2.94e-16 |
| designed150_s46 | 0.990693318 | 0.68749 | 0.67475 | 1.34782 | 1.21e-16 |

---

## Deliverable for the V2 writer

**Figure:** `.claude/outputs/f1-gmor-condensate/fig_gmor_condensate.png` (4 panels: (a) the `μ²F² = δΣ` collapse over 8 decades; (b) absolute deviation pinned at the f64 floor vs the `ε/δ` relative roundoff line; (c) the running condensate `Σ(δ)/Σ(0)` against the angular tilt's flat `Σ ≡ f`; (d) the LEC ratio → 1 as `x → 0`).
**Data:** `gmor_condensate.npz` (80 rows × 27 fields), `angular_tilt_contrast.npz`, `gmor_condensate.json`, `analysis_summary.json`.

**Proposed appendix wording (2–3 sentences), with the honest main-text caveat:**

> Replacing the angular tilt `δ cos nθ` with a **linear ambient spurion** `V → V − δ(u·q)` — the exact analogue of the ChPT quark-mass term — lets the vacuum radius run with `δ`, so the three GMOR objects can be measured *independently* on a trained checkpoint: the spectral gap `μ²`, the decay constant `F² = M_ch r*²`, and the condensate `Σ = r*(δ) = −∂E_vac/∂δ`. Across eight trained SO(2) checkpoints and eight decades of `δ ∈ [10⁻⁸, 0.3]` we find `μ²F² = δΣ` satisfied to the double-precision floor (`max |μ²F² − δΣ| = 1.3×10⁻¹⁵`; the residual *relative* error at small `δ` is the `ε/δ` roundoff of the autodiff Hessian, not a violation — a cancellation-free Hessian returns `2.2×10⁻¹⁶` at every `δ`). The leading low-energy constant is moreover **saturated by the radial (σ) resonance**: the LO-GMOR relative error equals `δ/(M_ch μ_rad² f)` with measured ratio `0.999996` for `δ ≤ 10⁻⁵` and `O(x)` corrections thereafter, so CLU realises ChPT's *resonance saturation of LECs* exactly rather than phenomenologically.
>
> *Honest note (belongs in the appendix, not the main text):* the **angular** tilt of §3.1 remains the clean power-law verification (C-2) — it is radius-independent, hence leaves the vacuum radius fixed to `2.2×10⁻¹⁵` and measures only the product `μ²F² = δn²`. That is the right instrument for verifying `μ² ∝ δ`; it simply **cannot resolve the condensate**. This appendix resolves it.

---

## Git footprint

- **Branch:** `agent/experiment-engineer/f1-gmor-condensate` (base: local `main` @ `27f232f`; rebase onto `main` = no-op, base unmoved). **Not pushed, no PR** (per §3.6).
- **Commits (3, atomic, tagged):**
  - `2aca35c` add LinearSpurionPotential (linear ambient ChPT spurion)
  - `64af0e7` wire spurion_delta/spurion_angle through CHLU + Exp-D config
  - `9bc2cf7` test condensate-resolving GMOR (mu^2 F^2 = delta Sigma)
- **Files touched (5, +248/−2):** `chlu/core/potentials.py` (+71: new wrapper + `channel_spurion_direction`), `chlu/core/chlu_unit.py` (+28: 2 ctor kwargs + guarded wrap), `chlu/config.py` (+7: 2 `ExperimentDConfig` fields), `chlu/experiments/exp_d_goldstone.py` (+7/−2: pass-through + metrics), `tests/test_goldstone.py` (+137: 4 tests).
- **No shared code refactored.** `ruff format` on `chlu_unit.py` deliberately NOT run (pre-existing diff on `main`; would have swept unrelated hunks).
- **Tests:** `200 passed, 1 skipped` (was 196 passed + 1 skipped before my 4 tests). `ruff check` clean.
- **Conflicts:** none. Working tree clean; no other agent's work present in the checkout.

---

## Open questions / follow-ups / risks

1. **The `ε/δ` probe floor is a general property of every `μ²` measurement in V2**, including the *published* angular-tilt sweeps (`gmor_ratio = 1.00000 ± 1e-12` in SF-3). Those quote relative agreement at δ ≥ 1e-4 where the floor is ~1e-12 — consistent, but the Hub should know the small-δ end of any future μ² sweep is roundoff-limited, and that quoting "machine precision relative" for δ ≲ 1e-6 would be wrong. **Not a defect in prior results**; a measurement-resolution note.
2. **δ=0.3 is outside the chiral expansion** for the soft-radial designed150 seeds (`x` up to 0.68, Σ running +210%). If the V2 appendix plots the NLO panel, cap it at `x < 0.25` (as my panel (d) does) or state `x` explicitly.
3. `LinearSpurionPotential.direction` is an **array leaf** (not static), so `eqx.partition(·, eqx.is_array)` would place `u` in the trainable partition. Harmless for this probe-only use and consistent with `IntraWormholePotential`'s leaf fields, but **do not train with a spurion attached** without freezing `u` (`δ` itself is static, so it is safe).
4. Not done (out of scope, cheap follow-ups): the **relativistic** kinetic mode (deep-dive O5 — `F²` should acquire a boost correction, "running decay constant"); the **emergent** MLP arm (its `δ_eff` self-breaking would add to the applied `δ`, so GMOR should hold with `δ → δ + δ_eff` — a sharp test of F-8).

## Proposed handover updates (for the Hub)

- **§7 (Known Issues) — add a measurement-resolution note (non-blocking):** the autodiff-Hessian `μ²` probe has relative precision floored at `~ε/δ` (absolute error `~ε‖Hess V‖`). Verified: analytic cancellation-free Hessian gives `2.2e-16` relative at all δ; autodiff gives `2.3e-8 @ δ=1e-8 → 4.2e-15 @ δ=0.3`, exactly `∝1/δ`. Consequence: GMOR-type identities should be quoted as **absolute** deviations at small δ. Does not invalidate any prior result (all prior sweeps are δ ≥ 1e-4).
- **§8 / deep-dive F-1 — CLOSE as done.** GMOR proper is demonstrated on trained checkpoints (not just the toy): `max |μ²F² − δΣ| = 1.33e-15` over 8 checkpoints × 10 δ; `Σ = −∂E_vac/∂δ = r*` measured three ways (exact / `9.2e-11` FD); the leading LEC is resonance-saturated (`ratio = 0.999996`, `1 − 1.05x` corrections). Lands in **v2-revision-4** as an appendix. Zero retraining cost. The deep-dive's **S1 and S3 are now confirmed on trained models**, discharging its own honest flag **O3** ("everything here is on toy Mexican-hat potentials, not trained checkpoints") for those two claims.
- **§3 (Config) — new knobs, defaults preserve behavior:** `ExperimentDConfig.spurion_delta = 0.0`, `spurion_angle = 0.0` → `CHLU(spurion_delta=…, spurion_angle=…)`. When `δ=0` no wrapper is constructed (bit-compatible). Sibling of the existing `tilt_delta`/`tilt_n`. No CLI flag added (config.yaml override only) — say the word if a `--spurion-delta` hook is wanted.
- **§2 (Architecture) — `potentials.py` now ships `LinearSpurionPotential` + `channel_spurion_direction`**, alongside `TiltedPotential`. Rule of thumb for the docs: **angular tilt = verify the power law; linear spurion = resolve the condensate.**
- **Nomenclature (deep-dive 7.16 corrigendum) is now measurable:** `F² = M_ch·r*²` and `Σ = r*` are separately instrumented in `gmor_condensate.npz`, so the F5 "decay constant = orbit radius" fix has data behind it.
