# v2-so2-build — experiment-engineer report

Task + acceptance criterion: build the SO(2) Goldstone-memory apparatus (SO(2)-equivariant potential + tied channel inertial masses + GMOR tilt + spectrum/perturb-track/Noether harness + SO(2)-degenerate training experiment), validated as pytest smoke checks against F5 §3.3–§3.4 / Appendix-N exact quadratic predictions, plus one quick learned end-to-end run with reported numbers.
Status: **done**

## What I did
- Worked in a dedicated worktree `../CHLU-v2-so2-build` (branch `agent/experiment-engineer/v2-so2-build` off `main`@d2d2401) — **necessary**: fix-pack-2 was actively committing in the main checkout mid-session (observed commit `bffceac` + uncommitted `integrators.py` edits appear there).
- **Core** (`chlu/core/potentials.py`, `chlu_unit.py`):
  - `SO2InvariantPotential`: V = f_θ(r²) + 0.05·r² + PotentialMLP(q_spec), channel = coords (0,1). Exactly invariant by construction (radial MLP fed the polynomial invariant **r², not r** — every smooth SO(2)-invariant is smooth in r²; r would allow a conical cusp at the origin that breaks Hessian probes). Coercive (F5 Prop-10 A1 holds architecturally, unlike deep/conv).
  - `TiltedPotential`: composable δ·cos(nθ) explicit-breaking wrapper (F5 §3.3c GMOR probe) — probe any trained checkpoint at any δ **without retraining** (avoids the static-field-replacement problem of baking δ into the module).
  - CHLU: `potential_type="so2_invariant"` dispatch; `tie_channel_mass` static flag (kinetic isotropy per F5 §4.1) implemented as **log-space mean at use time** (constraint-by-reparameterization; robust to whatever Adam does to raw entries); `mass_vector()`/`effective_inertia()` accessors exposing the exact rest inertia the dynamics use **including the 1e-6 epsilon in H** (per kinetic mode: 1 / M+1e-6 / m₀(M+1e-6)); `H()` routed through `mass_vector()` — bit-identical for untied models. `getattr` guard for pre-field checkpoints (§7.13 pattern), with a regression test.
- **Harness** (`chlu/experiments/goldstone_harness.py`, reusable, analyst-facing): `spectrum_probe` (W = M_eff^{-1/2}∇²V M_eff^{-1/2} → μ²_k + canonical eigvecs + |∇V| settledness), `settle`, `rollout_from` (prepends initial state so indices = map applications, F5 n-counting), `perturb_and_track` (position/momentum kicks along canonical eigendirections; per-mode canonical coords d, pc and **envelope amplitude** √(d²+(pc/μ)²) — raw |d| first-crossing measures phase, not retention), `half_life_first_crossing`, `fit_decay_rate`, `exact_mode_eigenvalues`/`h_star`/`classify_mode`/`predicted_half_life` (F5 §3.4 band table + exact-λ half-lives), `noether_charge`, `coset_angle` (unwrapped), `latch_prediction`, `step_jacobian`; hand-built `QuadraticPotential`/`MexicanHatPotential` + `clu_with_potential`/`log_mass_for_inertia` builders (inverse-softplus so `effective_inertia()==requested` exactly).
- **Experiment** (`exp_d_goldstone.py` + `ExperimentDConfig` + `chlu/data/circle_vacuum.py` + CLI `chlu exp-d` + `plot_goldstone_summary` + `results/exp_d_metrics.npz`): dataset = constant states on a circle (simplest SO(2)-degenerate vacuum; documented choice), standard wake–sleep training (**epochs/window/dt passed explicitly to `train_chlu` — immune to the §7.10 quick-mode trap**), then settle → spectrum → perturb → latch → Noether. `--quick`, `--potential-type {so2_invariant,mlp}`, `--broken-isotropy`, `--tilt-delta/--tilt-n` flags on both the CLI and a documented `python -m chlu.experiments.exp_d_goldstone` entry. `all-experiments` left as A/B/C (paper trio) on purpose.
- **Tests** (`tests/test_goldstone.py`, 12 tests): the five mandated F5 smoke checks (a)–(e) + GMOR spectrum + invariance/tie/grads/dataset/roundtrip/checkpoint-compat. Module enables **JAX x64** at import (F5 App-N numbers are float64); full suite verified unaffected.

## How I verified (commands + observed numbers)
- `uv run pytest -q --no-cov` (worktree): **30 passed in 29.4 s** (18 pre-existing + 12 new; x64 global flag harmless — pre-existing tests are tolerance-based).
- `uv run ruff check chlu/ tests/` → All checks passed; new files `ruff format`-clean (did NOT reformat pre-existing files: base `chlu_unit.py` was already format-dirty — out of scope).
- Harness-vs-F5 Appendix-N (measured through the real CHLU/`integrators.py` path, jax x64):

| Check | F5 App-N | **Measured (this harness)** |
|---|---|---|
| (a) latch error \|q_N − (q0+εp0/(Mγ))\| | 1.0e-15 | **2.2e-16**; frozen 2000→4000 exactly (0.0); curved companion 6.7e-46; momentum law (1−γ)ⁿ to 8.3e-17 |
| (b) overdamped n_1/2 (μ²=0.04/0.01, γ=0.2, ε=0.05) | 1544 / 6165, ratio 3.993 | **1544.0 / 6165.0 (bit-for-bit), ratio 3.9929**; exact-λ preds 1537.2/6158.2 |
| (c) underdamped \|λ\|=√(1−γ) (m=1 and m=0.25) | exact to 1e-9 | **1.1e-16 both masses**; envelope rate fit −0.02575 vs −0.02565 (0.4%); first-crossing 37 (position kick) — see phase-artifact finding below |
| (d) Q decay (1−γ)ⁿ | 9.3e-16 | **4.6e-16** (γ=0.05, 1000 steps, Mexican hat) |
| (e) isotropy: equal vs unequal channel M | 3.0e-14 vs 2.6 | **1.1e-14 vs 0.29** (2e4 steps; ratio ~3e13 — O(1) drift confirmed) |
| (f) GMOR spectrum μ² = {δn²/(Mf²), 8λf²/M} | — | **[0.02136752, 6.20307692] vs predicted [0.021367521367521, 6.203076923076923]** (exact to print), \|∇V(q*)\|=2e-18; settle → r*=1.200000000000 (f=1.2) |

- **Quick learned end-to-end run** (config: seed 42, 150 epochs, otherwise `ExperimentDConfig` defaults — dim 4, hidden 64, newtonian_learned, so2_invariant, tie_channel_mass=True, δ=0, n_points 256, seq_len 65, R=1.0, dt 0.05, probe γ=0.05×4000 steps, kick 0.1; TrainingConfig defaults lr 1e-3, sleep_freq 5, sleep_steps 500, sleep_temperature 0.5, persistent_sleep_buffer False; float32):
  - Final wake loss 1.93e-4; settled vacuum r* = 0.9670 (data R = 1.0; the 0.05r² confinement biases the learned radial minimum inward at quick epochs — expected), |∇V(q*)| = 5.3e-6.
  - **Spectrum (learned model): μ² = [8.7e-07, 0.136, 0.206, 0.670]** → a near-flat direction along the designed orbit, **5 orders of magnitude below the next mode** (architecturally protected here; the emergent-`mlp` variant is one config switch). Bands: [register(μ²≈0 residual; pred n_1/2 = 1.6e7 steps!), register (88.7), register (50.4), working_memory (27.0)].
  - Measured: flat-mode n_1/2 = **inf over 4000 steps**; stiffest measured 44 vs pred 27 (phase ripple ±12 for h=0.041 + anharmonicity of learned V + f32); stiff envelope rate fit −0.02503 vs −0.02565 (**2.4%** — the law holds on a learned potential).
  - **Latch on the learned model:** momentum write d∞ measured 0.09935 vs predicted 0.1000 (0.65%; residual = orbit curvature vs linear prediction + settle residual); **freeze drift over last 2000 steps = 0.0 (bit-frozen)**; coset angle plateau exact (drift 0.0).
  - **Noether decay on the learned model:** max |Q_n − (1−γ)ⁿQ₀|/|Q₀| = **1.19e-7 = float32 eps** — tie_channel_mass + invariant V deliver the exact law at working precision.
  - M_eff = [0.6837, 0.6837, 0.5820, 0.7136] — channel masses exactly tied as enforced.
  - Artifacts: `.claude/scratch/v2-so2-build/{plots/exp_d_goldstone_summary.png, results/exp_d_metrics.npz, models/exp_d_chlu.pkl}` (4-panel figure: spectral gap, flat-mode retention pinned at 1.0, Q on the exact law, coset-angle plateau).
- CLI/entries: `chlu exp-d --help` (via `main()`; console-script path still blocked by §7.12 env bug) and `python -m chlu.experiments.exp_d_goldstone --help` both parse; config YAML roundtrip preserves `experiment_d`.

## Findings (beyond the build)
1. **F5 App-N's underdamped first-crossing "23–26 vs 27" is kick-phase-dependent** (new precision on their documented artifact): friction only bites momentum, so log-amplitude ripples with amplitude γ/(2h) in log E ⇒ crossing jitter ≈ ±(γ/2h)/|ln√(1−γ)| steps (±10 at γ=0.05, h=0.05). A position kick lags to 37, a momentum kick to 40, F5's phase gave 23–26 — all within the ripple band around 27.03; the exact statements (rate ½ln(1−γ), |λ|=√(1−γ)) hold to 0.4% / 1e-16. Test asserts the law tightly and the crossing within the ripple window (which still excludes the overdamped law's 14.2).
2. The **budget-table machinery quantifies even numerical flatness**: the learned flat mode's residual μ² = 8.7e-7 (settle residual + f32) still prices out at n_1/2 ≈ 1.6e7 steps — useful language for V2 ("retention floor set by measurement residual, not dynamics").
3. fix-pack-2 had NOT merged at build time ⇒ ran with **legacy flags**: degenerate Lyapunov penalty (≡0 in wake) and legacy Langevin sleep noise. When fix-pack-2 merges (its new `lyapunov_penalty` default "max"), retraining exp-d will shift numbers slightly — my run above is the legacy-flag baseline.

## Git footprint
- Branch: `agent/experiment-engineer/v2-so2-build` (worktree `../CHLU-v2-so2-build`, removed after completion; branch left for review; **not pushed**). Rebased on `main` (unmoved, d2d2401): up to date, no conflicts.
- Commits: `c811232` (SO(2) potential + tilt wrapper + tied inertial mass), `5b0619b` (harness + F5 smoke tests), `b0e40a7` (Exp D: config/data/CLI/plots/metrics), `5e9f1e0` (return-key rename + pre-field checkpoint-compat test).
- Files: `chlu/core/potentials.py`, `chlu/core/chlu_unit.py`, `chlu/config.py`, `chlu/data/circle_vacuum.py`(new), `chlu/data/__init__.py`, `chlu/experiments/goldstone_harness.py`(new), `chlu/experiments/exp_d_goldstone.py`(new), `chlu/experiments/__init__.py`, `chlu/cli/experiment_cmd.py`, `chlu/utils/plotting.py`, `tests/test_goldstone.py`(new). +1735/−11.
- Shared-file overlap risk vs fix-pack-2: `chlu/config.py` (they add TrainingConfig fields; I add ExperimentDConfig — different hunks, trivial merge) and possibly `experiment_cmd.py` (unlikely per their scope). No overlap on core files they touch (`regularization.py`, `integrators.py`, `mnist.py`, exp_a/b).

## Open questions / follow-ups / risks
1. **Analyst follow-ups now unblocked** (full-scale runs + figures, per task): (i) GMOR δ-sweep on the saved checkpoint via `TiltedPotential` wrap (no retraining) → measured n_1/2 ∝ δ^{-1} with saturation; (ii) emergent-symmetry variant (`--potential-type mlp`) — does a near-flat direction emerge without the architectural guarantee?; (iii) broken-isotropy training (`--broken-isotropy`) — does learned M isotropize on symmetric data (F5 §4.1 falsifiable)?; (iv) γ- and seed-sweeps; longer training to close the r*→R gap.
2. Stiff-mode first-crossing on learned V (44 vs 27) deserves the anharmonicity decomposition (smaller kick → closer to linear prediction) — one-liner for the analyst.
3. x64 note: `tests/test_goldstone.py` flips the process-global x64 flag at collection; verified harmless for the current suite, but future f32-bit-exactness tests would need isolation (pytest-forked or env guard).
4. `spectrum_probe` is exact only at p≈0 (F5 Prop-2 relativistic mode-coupling) — relativistic hot-state spectra out of scope here.
5. Worktree removed; branch intact. Rebase again if fix-pack-2 merges before this branch (config.py adjacency).

## Proposed handover updates (for the Hub)
- **§2/§3:** new experiment **D** (`chlu exp-d`, `exp_d_goldstone.py`, `ExperimentDConfig` defaults: dim 4, so2_invariant, newtonian_learned, tie_channel_mass=True, tilt δ=0/n=1, circle R=1.0×256 pts, probe γ=0.05) + `chlu/experiments/goldstone_harness.py` (reusable measurement instruments incl. `spectrum_probe` — this is the Hessian-at-attractor probe F5 update-12 asked for; mass-spectrum-peek can drive it on existing checkpoints) + `chlu/data/circle_vacuum.py` + new potential types `so2_invariant` and `TiltedPotential` + CHLU `tie_channel_mass`/`mass_vector()`/`effective_inertia()`.
- **§7.10 note:** exp-d is NOT affected (passes `epochs=` explicitly); A/B remain affected until fix-pack-2 merges.
- **F5 App-N caveat sharpened:** the underdamped first-crossing artifact window is kick-phase-dependent, ±(γ/2h)/|ln√(1−γ)| steps around 2ln2/(−ln(1−γ)) (evidence above) — worth a one-line annotation next to F5's "known diagnostic artifacts".
- **V2 status:** harness reproduces F5 App-N to 1e-16-grade (n_1/2 = 1544/6165 bit-for-bit); first learned CLU shows the designed near-flat orbit direction (μ² gap 5 orders), an exact latch (bit-frozen coset angle), and the exact Noether decay law at f32 precision — V2's decisive-experiment apparatus is ready for full-scale analyst runs.
