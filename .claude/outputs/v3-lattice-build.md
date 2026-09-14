# v3-lattice-build — experiment-engineer report

Task + acceptance criterion: build the first CLU-Net (`chlu/core/lattice.py` + experiment) — joint-Hamiltonian N-unit lattice, position-only coupling, designed inertial-mass banding; acceptance centerpiece = the "coupling strength prices communication speed against relative-memory lifetime" plot, with κ_c=0 bit-level reduction / joint symplecticity / Noether / pricing-law / causal-cap tests.

Status: **done**

## What I did
- **`chlu/core/lattice.py` (new):** `CLULattice(eqx.Module)` — units = tuple of `CHLU`s (per-unit dims d_i, any existing potential type incl. `so2_invariant`), one joint H over the concatenated state, one global Verlet step. F5 §7.2 conditions enforced *structurally*: couplings only ever receive `(q_i, q_j)` (momentum coupling impossible by construction — scope guard satisfied); uniform scalar γ = conformal default; per-unit γ_i behind `gamma_vector()` with the conformality caveat documented (det J = Π(1−γ_i)^{d_i}, Prop-4 pairing lost). Couplings: `SpringCoupling` κ_c·‖W_i q_i − W_j q_j‖² (learnable W, **static κ_c** so the optimizer can't move the swept knob), `MLPCoupling` (flag `coupling_type="mlp"`), `GatedCoupling` = wormhole skeleton with a smooth sigmoid energy gate on V_c (F5 §7.4 smooth variant; bounded energy; no top-k). Banding: `scale_inertial_mass()` — *exact* softplus-space rescaling (softplus(new)=scale·softplus(old)); `causal_caps()` = c/√M_i (F5 Prop-1); banding an identity-kinetic unit raises loudly. `build_lattice()` = config-level factory (chain default, wormhole edges).
- **Duck-typing decision:** lattice exposes the CHLU surface (`H, T, potential_net, effective_inertia, effective_mass, mass_vector, step, stochastic_step, __call__`) ⇒ **the entire `goldstone_harness` and `train_chlu` work on the joint state verbatim** — zero new measurement code. Rollouts scan-based/pure (reversible-BPTT preserved).
- **`chlu/core/chlu_unit.py` (surgical):** extracted `CHLU.T(p)`; `H` delegates (op-for-op identical; bit-level test). Needed for separable T_net = ΣT_i without duplicating kinetic physics.
- **`chlu/config.py`:** `ExperimentLatticeConfig` (all knobs; no magic numbers in the experiment body) wired into `CHLUConfig`/`load_config`/`save_config`. **Plus an out-of-scope-but-necessary 1-hunk fix, own commit** — see Finding 0.
- **`chlu/data/two_timescale_orbits.py` (new):** reference-oscillator composite signal — per-unit channel circles at (ω=0.5, ω=2.0) with p = M dq/dt and **shared stiffness k_i = M_iω_i² = 1**: the two timescales come from the inertial-mass band alone (F5 §5 role 3). Seeded phases only.
- **`chlu/experiments/exp_lattice.py` (new)** + `plot_lattice_pricing` (plotting.py, appended) + `chlu exp-lattice` CLI hook (`--quick`, `--skip-training`; quick-epochs wiring added to `_get_config_and_paths`) + export in `experiments/__init__.py`. Metrics → `results/exp_lattice_metrics.npz`.
- **`tests/test_lattice.py` (new, 12 tests)** — the task's five acceptance tests + banding/wormhole/gradient-flow/validation extras. x64 at import (test_goldstone convention).

## How I verified (real commands, real numbers)
Env: worktree `../CHLU-v3-lattice-build` (gamma-field agent runs in parallel — §3.2 isolation), main venv + `PYTHONPATH=<worktree>` (verified `chlu.__file__` resolves to worktree; bypasses the §7.12 uv-CLI issue).
- `python -m pytest tests/test_lattice.py` → **12 passed** (21.7s). Full suite `pytest tests/` → **103 passed, 1 skipped** (main baseline was 91+1; +12 mine). `ruff check chlu/ tests/test_lattice.py` → clean; `ruff format` applied to my 4 new files only (chlu_unit/config are pre-existing format-dirty on main — left alone to avoid diff noise).
- `python -m chlu.experiments.exp_lattice --quick --seed 42` and full `--seed 42` → both end-to-end green; CLI parser wiring verified by parsing `['exp-lattice','--quick','--skip-training','--seed','7']`.
- Config fix verified: before = `is_dataclass(ExperimentV1GateConfig)=False`, `difficulty_levels` a raw `dataclasses.Field`; after = True + YAML save/load round-trip clean.

## Findings / results

**0. BUG FOUND ON MAIN (fixed here, commit `b2ce79f`): `ExperimentV1GateConfig` lost its `@dataclass` decorator in the w2 union-merge** (`config.py` line 199 glued directly onto ExperimentDConfig's last field). Consequences on current main: `field(default_factory=...)` attrs are raw `Field` objects, and `load_config`/`save_config` crash (`.__dataclass_fields__`/`asdict`) ⇒ **every `--project` path and exp-v1-gate are broken on main**. ⚠ The v1-pivot agent is working on top of this. My fix is a 1-hunk own-commit; cherry-pickable.

**1. Communication pricing (acceptance centerpiece) — F5 §7.2 confirmed on the working lattice** (designed 2-unit Mexican-hat SO(2) pair + channel spring; f=M=λ=1, γ_probe=0.2, dt=0.05, seed 42, f32):

| κ_c | μ_rel² (pred 4κ/M) | sync steps (pred) | n₁/₂ (pred) | latch freeze | Noether err |
|---|---|---|---|---|---|
| 0.003 | 0.012000 (0.012000) | 290 (287) | 5137 (5131) | 0.0e+00 | 1.0e-07 |
| 0.01 | 0.040000 (0.040000) | 159 (157) | 1544 (1537) | 0.0e+00 | 1.2e-07 |
| 0.03 | 0.120000 (0.120000) | 92 (91) | 517 (510) | 0.0e+00 | 1.0e-07 |
| 0.1 | 0.400000 (0.400000) | 51 (50) | 158 (151) | 0.0e+00 | 6.2e-08 |
| 0.3 | 1.200000 (1.200000) | 29 (29) | 55 (48) | 0.0e+00 | 4.9e-08 |

Fitted log-log slopes: **sync ∝ κ^−0.499 (pred −1/2), n₁/₂ ∝ κ^−0.986 (pred −1)**. Shared channel **bit-frozen latch at every κ** (drift exactly 0.0 over the probe's last half). μ_rel² = 4κ_c/M to 6 decimals. Cross-check: n₁/₂=1544 at μ²=0.04, γ=0.2 reproduces the F5 App-N anchor bit-for-bit. Plot + npz: `.claude/outputs/v3-lattice-build/` (also emitted by the experiment to project results/).
- Deviations, both understood: (i) n₁/₂ at κ=0.3 is +15% over prediction (55 vs 48) — first-crossing kick-phase ripple growing as h→h*(γ) (F5 App-N known artifact; the exact-eigenvalue tests at small h agree to <2%); (ii) sync ~+1% high — pendulum anharmonicity at Δ₀=0.4 (∝Δ₀²/16, documented in the module).

**2. Tests encode the physics contracts:** κ_c=0 reduction is **exactly bit-level** (heterogeneous dims 2+3, mixed newtonian_learned+relativistic units, γ∈{0,0.05}, both empty-edges and κ=0-spring) — `jnp.array_equal` passes, no tolerance. Joint symplecticity <1e-12 (x64) at N=2,4. Per-unit-γ det law exact <1e-12. Joint Noether: conserved <1e-10 at γ=0, decay law <1e-9 at γ>0, **and the single-unit charge is NOT conserved at κ>0 (drift >1e-3) — communication is literally charge flow between units**. Causal caps: v saturates at c/√M_i per band to 1e-9, heavy band strictly slower.
- Two test-tolerance corrections during verification (both physics, not code bugs): (a) shared-channel latch *prediction* differs from the linear formula by the curved-vacuum projection bias ≈|d∞|³/6 (measured 4.3e-7 vs cubic scale 3.3e-7) — freeze stays machine-exact, prediction gets the documented cubic tolerance (machine-exact linear-latch test already lives in test_goldstone); (b) the MLP coupling's **final bias has legitimately zero gradient** (constant in H = gauge, no force) — test now requires weight-matrix gradients, not every leaf.

**3. Scaling smoke** (chain, mlp potentials, f32, laptop): N=2/4/8 → symplectic err 5.4e-8/4.9e-8/4.4e-8 (flat in N; f32 floor), energy drift 4.5e-4/2.8e-5/5.9e-5 over 2000 steps, **233k/125k/35k steps/s**. Sub-linear-ish cost growth to D=16; fine for laptop V3 prototyping, CSF3 only needed at much larger D/N or training sweeps.

**4. Wormhole skeleton:** gated edge (0,3) on a 4-chain — aligned: moving unit 0 changes the force on unit 3 by 3.7e-3 (V_wh=1.3e-4, gate open); far: V_wh=1.2e-9 (smooth closure; 5 orders suppression). Everything stays C¹/conformally-symplectic — no energy ledger needed at this stage.

**5. Training smoke — banded beats uniform (SINGLE SEED = INDICATIVE ONLY):** identical architecture/params (2382)/key/data/budget, only log_mass init differs (banded = ×(4.0, 0.25) exact scaling of the same random init → ≈(2.5, 0.17); uniform ≈(0.63, 0.68)). 300 epochs: banded final wake loss **0.331** vs uniform **1.714**; eval rollout MSE (8 held-out trajs, 255 steps) **1.362 vs 2.516**. Quick 60-epoch: 0.917/2.971 vs 2.625/4.097 — banded ahead at both budgets, consistent with "banding = learnability prior, not capacity" (F5 §5 honest deflation). This is Thread-5 falsifiable (iii) *evidence*, not a claim — needs a seed sweep (analyst task).

## Git footprint
- Branch: `agent/experiment-engineer/v3-lattice-build` (worktree, off main @ `dbeb2c2`; rebase = no-op, main unmoved; worktree removed after completion; NOT pushed, no PR).
- Commits: `3f1c4a3` CHLU.T extraction · `b2ce79f` @dataclass fix · `c124103` CLULattice core · `9c39b12` data generator · `c1e34be` exp-lattice + plot + CLI + config · `79dcac9` tests.
- Files: new `chlu/core/lattice.py`, `chlu/data/two_timescale_orbits.py`, `chlu/experiments/exp_lattice.py`, `tests/test_lattice.py`; edited `chlu/core/chlu_unit.py` (T extraction only), `chlu/config.py` (decorator fix + ExperimentLatticeConfig + wiring), `chlu/utils/plotting.py` (appended `plot_lattice_pricing`), `chlu/cli/experiment_cmd.py` (exp-lattice parser/cmd + one quick-mode hunk in `_get_config_and_paths`), `chlu/experiments/__init__.py` (export).
- Commands: full suite **103 passed / 1 skipped**; ruff clean; experiment run full + quick (outputs in `.claude/outputs/v3-lattice-build/`).

## Open questions / follow-ups / risks
1. **Merge-order note for the Hub:** commit `b2ce79f` (config fix) will textually conflict with any other w3 branch touching `config.py` near ExperimentV1GateConfig — trivial union resolution; and v1-pivot should be told main's exp-v1-gate config is currently broken.
2. Pricing ran on *designed* potentials (attribution-clean by design). Trained-lattice pricing (does a **learned** V_c also price communication by its curvature?) = natural analyst/engineer follow-up; the harness needs zero changes for it.
3. Training smoke is 1 seed. Falsifiable (iii) needs a seed sweep + budget curve (wake-loss vs epoch crossover) — analyst-shaped, laptop-scale.
4. Per-unit γ implemented for `step`/`__call__` only; `stochastic_step` stays scalar-γ (heterogeneous γ + FDT noise needs its own spec — flagged, not implemented).
5. Experiment runs default f32: |μ_sym²| reads ~1e-7 (f32 Hessian floor; <1e-10 in x64 tests) and Noether err ~1e-7 (f32 eps). Enable x64 for paper-grade figures.
6. No momentum coupling anywhere; no corner appeared that needed it.

## Proposed handover updates (for the Hub)
- **§7 (Known Issues) — add & mark resolved-on-branch:** "w2 union-merge artifact: `@dataclass` missing on `ExperimentV1GateConfig` ⇒ `load_config`/`save_config` (all `--project` paths) and exp-v1-gate broken on main. Fixed on `agent/experiment-engineer/v3-lattice-build` @ `b2ce79f` (1-hunk, cherry-pickable)." Consider a smoke test that round-trips the default config (would have caught this and the paren artifact).
- **§2 (Architecture):** add `core/lattice.py` (CLULattice + Spring/MLP/Gated couplings + build_lattice + scale_inertial_mass), `data/two_timescale_orbits.py`, `experiments/exp_lattice.py`; note `CHLU.T(p)` now exists and `H = T + V` delegates.
- **§3 (CLI/config):** `chlu exp-lattice [--project] [--seed] [--quick] [--skip-training]`; new `experiment_lattice` config group (defaults: spring coupling, κ sweep [0.003…0.3], probe γ=0.2, banded scales [4.0, 0.25], 300 train epochs; quick=60).
- **§8/V3:** the F5 §7.2 pricing claim is now *measured* on a working lattice (slopes −0.499/−0.986, μ_rel²=4κ/M exact, shared latch bit-frozen) — V3's first result exists; banded-vs-uniform single-seed evidence logged for Thread-5 falsifiable (iii).
