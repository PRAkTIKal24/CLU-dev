# Task: fix-pack-5 — the FDT/tie_channel_mass Gibbs bug + audit (w13, engineer, small)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/fix-pack-5.md`
- **Read first:** protocol (**§3.5: rebase onto local `main`, NOT origin/main**) · **`.claude/outputs/t-lever-forgetting.md` §6** (the bug, with 5-seed evidence + the exact fix) · `chlu/core/chlu_unit.py` (`mass_vector`, `effective_inertia`, `effective_mass`), `chlu/core/integrators.py:~166` (the noise scale).
- **Git:** branch `agent/experiment-engineer/fix-pack-5` off local `main`.

## The bug (measured, not hypothesized)
`CHLU.effective_mass()` returns raw `softplus(log_mass)` — it does **not** apply `tie_channel_mass` (which `mass_vector()`, and hence `H`/`T`, does) and omits the `+1e-6` that `H` inverts. `stochastic_step` builds the `noise_mode="fdt"` noise scale from it, so on any `tie_channel_mass=True` checkpoint the Langevin noise uses a **different inertia than the dynamics invert**: `Var(p_i) = effective_mass()_i·T` instead of the Gibbs-required `effective_inertia()_i·T`. Channels equilibrate at different temperatures ⇒ **no Gibbs invariant**. Measured T_eff ratio deviates up to **8.4%** (5 seeds, γ=0.05, T=1e-3), tracking the predicted `M_noise,0/M_noise,1`; a retied control gives ≤1.3%.

## Items
1. **Fix:** make `effective_mass()` delegate to `effective_inertia()` (or have `stochastic_step` call `effective_inertia()` directly). Affects **only** `noise_mode="fdt"` + `tie_channel_mass=True`; `legacy` and untied models must remain **bit-identical** (assert this in a test).
2. **Regression test:** on a tied checkpoint under `fdt`, assert `Var(p_i) ≈ effective_inertia()_i · T` (equal channel temperatures, ratio → 1 within MC error). Reuse the analyst's instrument — the **momentum-variance test** (`s5b_fdt_bug_direct.py`); note the analyst's documented negative: `D_θ(θ₀)` is the WRONG instrument (the coset angle wanders within a block and washes out the anisotropy).
3. **Audit (the important part):** grep every shipped/reported result for `langevin_noise="fdt"` **AND** `tie_channel_mass=True` co-occurring. The analyst believes **none** are contaminated (`legacy` is the repo default), but confirm and state it explicitly — this determines whether any published/claimed number needs an asterisk. Report the audit list either way.
4. Bit-compat: defaults unchanged; full suite green (expect ≥200).

**Acceptance:** bug fixed, `legacy`/untied paths bit-identical, regression test pins Gibbs on tied+fdt, audit answered in writing. Flag-provenance per §5.
