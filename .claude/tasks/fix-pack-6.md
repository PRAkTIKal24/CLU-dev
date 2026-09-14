# Task: fix-pack-6 — the ⛔ blocking FDT NaN + two latent tie bugs + honest `fdt` docs (w14, engineer)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/fix-pack-6.md`
- **Read first:** protocol (**§3.5: rebase onto local `main`, NOT `origin/main`** — origin is frozen at `40c2f31`) · **`.claude/outputs/xy-lattice-theory.md` §5(i-b)** (the blocking bug, with the verified fix) · **`.claude/outputs/v2-symmetry-deepdive.md` §7bis R8 / §10 X6** (the relativistic no-go) · your own `.claude/outputs/fix-pack-5.md` **§7 "NEW latent issue"** · `.claude/claims_matrix.md` **CM-17**.
- **Git:** branch `agent/experiment-engineer/fix-pack-6` off local `main` (now at `d6f8bac`).
- **Parallel safety:** a sibling task `lattice-xy-prereqs` runs concurrently and owns **`chlu/core/lattice.py`**. **Do not touch `lattice.py`.** Your files: `integrators.py`, `transforms.py`, `chlu_unit.py`, `config.py`, `train.py`, `train_generative.py`. Use a worktree per §3.2.

## Item 1 — ⛔ BLOCKING: the `fdt` noise has a NaN gradient at γ=0
`chlu/core/integrators.py:~170`:
```python
noise_scale = jnp.sqrt(jnp.maximum(0.0, m_eff * temperature * gamma * (2.0 - gamma)))
```
At `γ=0` this is `sqrt(0)`, whose derivative w.r.t. the **learnable** `log_mass` inside `m_eff` is `∞·0 = NaN`. `legacy` is immune only because its sqrt argument contains no parameter.

**Trigger — all repo defaults except the first:** `langevin_noise="fdt"` **and** `sleep_temperature=0.5` (default ⇒ sleep uses `stochastic_step`) **and** `sleep_friction=0.0` (**default**) **and** a `log_mass`-carrying kinetic mode (`newtonian_learned`/`relativistic`). Since `epoch % sleep_frequency == 0` always fires at epoch 0, **`train_chlu` NaNs every parameter on the first sleep step.** Measured: `losses[0]=145.99` finite, `losses[1:]=NaN`. Reproduced on a plain `CHLU` **and** on `CLULattice`; **still live at `d6f8bac`**.

**Consequence: no FDT-correct model can currently be trained at repo defaults.** This blocks the Thread-10 KT program, the `T_φ(q)` build, and every Gibbs-sampling claim. **It is the highest-priority item in the program right now.**

**Fix (theorist-verified: bit-identical for `arg>0`, gradient exactly `0.0` at `arg==0`):**
```python
arg  = m_eff * temperature * gamma * (2.0 - gamma)
safe = jnp.where(arg > 0.0, arg, 1.0)
noise_scale = jnp.where(arg > 0.0, jnp.sqrt(safe), 0.0)
```
Test: `jax.grad` w.r.t. `log_mass` is finite at `γ ∈ {0, 1e-12, 0.05}` in all three kinetic modes; `legacy` unchanged; and an end-to-end `train_chlu` smoke with `langevin_noise="fdt"` at defaults produces **finite** losses for ≥3 epochs (this is the test that would have caught it).

## Item 2 — the *other* `effective_mass` (your own §7 finding)
`chlu/core/transforms.py:35` `effective_mass(model)` is a **separate free function** carrying the same tie bug: raw `softplus(log_mass)`, no `tie_channel_mass`, no `+1e-6`. It feeds `mass_weighted_squeeze` (`exp_v1_gate.py:520`, `exp_paid_access.py`). No live contamination (both consumers build untied models, no Langevin) — but **V1's boost/squeeze work is exposed the moment it touches a tied checkpoint.** Fix: delegate to `model.effective_inertia()`. Strictly bit-identical for untied models; assert that in a test.

## Item 3 — make the `fdt` flag tell the truth (CM-17)
R8 proves the `fdt` mode's promise — *"exact discrete fluctuation-dissipation; temperatures in energy units"* (`config.py:94-97`) — holds **only in the Newtonian kinetic modes.** The coded O-step `p←(1−γ)p+σξ` is a linear OU recursion (Gaussian stationary law); the relativistic Gibbs marginal is Maxwell–Jüttner. **No σ fixes it.** Root cause, stated crisply by `xy-lattice-theory` §5(v): *the Gibbs-preserving underdamped Langevin damps the **velocity** `∇_pT`; the code damps `p`.* For Newtonian these coincide (`Γ = γM`); for relativistic `∇_pT ∝ p/T(p)` and they do not.

1. Scope the config comment and the docstrings that repeat it (`train.py:78`, `train_generative.py:73`, `chlu_unit.py:372/504`, `integrators.py:109-127`). Give the control parameter `T/(m₀c²)` and the free mitigation (raise `c` or `m₀` until `T ≪ m₀c²`).
2. **Guard-rail:** on `noise_mode="fdt"` **and** `kinetic_mode="relativistic"`, emit a **warning** naming the call's `T/(m₀c²)`. **Warn, do not raise** — Exp-C must keep running and `relativistic-gibbs-expc` needs exactly this cell. Test that it fires there and **not** on Newtonian+fdt or any `legacy` path.
3. Expose `CHLU.thermal_causal_ratio(temperature)` → `T/(m₀c²)`. It is the single number governing the defect; analysts must be able to report it without recomputing.

## Item 4 — document the `fix-pack-5` propagation
`CLULattice.effective_mass()` and both `twins.py` wrappers delegate to `CHLU.effective_mass()`, so `fix-pack-5` reached the lattice and twin `fdt` paths too. Add a test pinning `CLULattice.effective_mass() == CLULattice.effective_inertia()`. (Comment only in `twins.py`; **`lattice.py` belongs to the sibling task** — put the lattice assertion in your *test file*, not in `lattice.py`.)

## ⚠ DO NOT "FIX" — a design decision that is load-bearing
`FrictionField` is **absorb-only**: `p ← (1−γ_φ(q))(1−γ)p` while the noise scale still uses the **scalar γ only**. This looks like an oversight. **It is not, and it must not be "corrected" to a locally-thermalized bath.** `v5-gate` measured that the absorb-only form makes a friction hole a **107.77 ± 4.78× memory vault** (a brake *and* a refrigerator, `T_local = 1.26e-4` vs `1e-3` outside), where the coupled-bath form gives only `13.28 ± 0.12×` — the coupled-bath hypothesis was **rejected by a factor 8.11 ± 0.37** against its own dedicated control. Add a comment at the noise-scale site recording this, citing `v5-gate` §R3, so no future agent "tidies" it away.

## Acceptance
NaN gradient gone with an end-to-end `train_chlu`+`fdt` smoke; `transforms.effective_mass` delegated and bit-identity asserted for untied; `fdt` docs true in every kinetic mode; relativistic+fdt warns with its own `T/(m₀c²)`; absorb-only comment landed; **defaults unchanged**; full suite green (expect ≥217 + your new tests). Flag-provenance per §5. **Report the `train_chlu`+`fdt` smoke output explicitly** — it is the evidence the blocker is actually cleared.
