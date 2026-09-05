# Task: fix-pack-7 — the two defects w14 created, + the exact relativistic thermostat (w15, engineer)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/fix-pack-7.md`
- **Read first:** protocol (**§3.5 rebase onto local `main`**; **§5 now carries the new pre-registration rule**) · `.claude/claims_matrix.md` **v1.9 CM-17** (rewritten — read it, my v1.8 row was wrong) · `.claude/outputs/f5-corrigendum-2.md` §2 (the `d·Θ` law) and §3 (the **latent-mass thermostat**, with its InvGauss spec) · `.claude/outputs/relativistic-gibbs-expc.md` §6 (the `sleep_friction` no-op) · your own `.claude/outputs/fix-pack-6.md`.
- **Git:** branch `agent/experiment-engineer/fix-pack-7` off local `main` (`df5e44d`).

## Context — w14 created these two defects itself
`fix-pack-6` and `f5-corrigendum-2` ran **in parallel and could not see each other**. The engineer shipped a guard-rail keyed on `Θ := T/(m₀c²)`; the theorist proved, in the same wave, that the control parameter is **`d·Θ`**. So the shipped warning and helper are **under-conservative at high `d` — exactly where the defect bites.** At Exp-C (`d=784`), `Θ = 1` looks benign; `d·Θ = 784` is catastrophic (`Var_MJ/(M_eff T) = 785×`, `KL ≈ 3.24e6` nats).

## Item 1 — `d·Θ`, not `Θ`
1. `CHLU.thermal_causal_ratio(temperature)` currently returns `Θ`. Either return `d·Θ` or (preferred, less surprising) **keep `Θ` and add `CHLU.gibbs_defect_parameter(temperature) → d·Θ`**, with the docstring stating plainly that **`d·Θ` is the quantity that governs the defect and `Θ` alone is not**. Ultra-relativistically `Var_MJ/(M_eff·T) → (d+1)·Θ`.
2. `RelativisticGibbsWarning` must report **`d·Θ`** (and may mention `Θ`). Its message should say the free mitigation needs `c ≳ √(dT/m₀)` — ≈ 28 at Exp-C's `T=1`, **not** `c=5`.
3. Test: at `d=784, m₀=c=1, T=1` the reported defect parameter is `784`, and the warning names it.

## Item 2 — `training.sleep_friction` is a silent no-op on the generative path
`chlu/training/train_generative.py:102-103` reads `config.experiment_c.friction` where it should read `config.training.sleep_friction`; `sleep_temperature` **is** read from `config.training`. Consequences already established: **Exp-C training is stochastic in its sleep phase** (γ=0.3, T=0.5, legacy) — `langevin_noise` is live during Exp-C *training*, not just at dream time — and **N19's scope line ("`sleep_temperature` is a no-op whenever `sleep_friction=0`") does not apply to Exp-C.**

⚠ **Do not silently change Exp-C's training dynamics.** Every `mnist*` checkpoint was trained under the current behaviour. Honour `training.sleep_friction` **only when explicitly set**, falling back to `experiment_c.friction` otherwise (and say so in the docstring), **or** delete `sleep_friction` from the schema if it is dead everywhere else — your call, but **state which and why, and prove the default path is bit-identical.** This is the program's fourth silent knob; the point is to stop it being silent, not to change history.

## Item 3 — implement the exact relativistic thermostat (CM-17 F2) behind a flag
Maxwell–Jüttner is a **Gaussian scale mixture**: `p|s ~ N(0, M/(2s))`, `s|p ~ InvGauss(mean = c²/(2T·T(p)), shape = c²/(2T²))`. Drawing `s|p` and then running the **same linear Gaussian O-step** with variance `M/(2s)` preserves MJ **exactly**. Physically: *the exact relativistic FDT noise is the coded Gaussian noise with a randomized inertia equal to the relativistic mass `m₀γ_Lorentz`.* Prop-9's `σ*` is exactly its `dΘ→0` limit.

- Add `langevin_noise="fdt_relativistic"` (name your call) implementing this. One inverse-Gaussian draw per step, closed-form, jax-able. **Do not change any default.**
- The theorist's verification target: the `O(1)` bias `−0.311 / −0.536 / −0.727` collapses to `+0.0006 / +0.0011 / +0.0011`, i.e. the Newtonian `O(ε²)` shadow floor (`+0.00014`). **Reproduce this.** Per §5, **pre-register** these targets in `PREREG.md` before running.
- It **dominates** an exact MJ momentum refresh (it keeps momentum persistence, so `q` mixes faster). Note that in the docstring.
- Guard: the `RelativisticGibbsWarning` must **not** fire for this mode.

## Item 4 — carry-over
`CLULattice.stochastic_step` must route the new mode correctly (it builds `m_eff` from `effective_mass()`); add the lattice test.

**Acceptance:** `d·Θ` exposed and reported by the warning; `sleep_friction` no longer silent with the default path proven bit-identical; the latent-mass thermostat implemented behind a flag and **verified against the pre-registered bias targets**; defaults byte-for-byte unchanged; suite green (≥278). Flag-provenance + PREREG per §5.
