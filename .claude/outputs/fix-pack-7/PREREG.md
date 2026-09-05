# PREREG — fix-pack-7 Item 3 (exact relativistic thermostat `fdt_relativistic`)

Written BEFORE running the JAX harness that measures the bias (protocol §5 pre-registration rule).
Commit: to be made on `agent/experiment-engineer/fix-pack-7` (base df5e44d).

## What is measured
A 1-D harmonic well `V(q)=½ k q²` (k=1) evolved by the *coded* `langevin_step`
in `kinetic_mode="relativistic"` (m₀=1, c=1, mass M=1), γ=0.1, dt(ε)=0.05.
Observable: the equilibrium position-variance bias

    B := Var(q) / (T/k) − 1

measured for three noise modes at Θ := T/(m₀c²) ∈ {0.5, 2, 8} (T = 0.5, 2, 8).
Long chains, many walkers, burn-in discarded, single thread seed per cell.

## Derivation of the predictions
Source: `f5-corrigendum-2.md` §3 + verification rows `fixes (II)/(IV)` and
`nogo_chain (b)`. In relativistic mode the coded additive-Gaussian O-step has a
Gaussian invariant momentum law (Prop-9′), so it under-samples the heavy
Maxwell–Jüttner tails → `Var(q)` is biased **low** (B < 0). The latent-mass
thermostat (`fdt_relativistic`) draws `s|p ~ InvGauss` and runs the *same* O-step
with variance `M/(2s)`, preserving MJ exactly, so B collapses to the Newtonian
`O(ε²)` shadow floor.

## Pre-registered predictions

| Θ (=T)  | `fdt` (biased, relativistic) B | `fdt_relativistic` B (predicted) |
|---------|-------------------------------:|---------------------------------:|
| 0.5     | ≈ −0.31                        | ≈ +0.0006 (\|B\| < 0.01)          |
| 2       | ≈ −0.54                        | ≈ +0.0011 (\|B\| < 0.01)          |
| 8       | ≈ −0.73                        | ≈ +0.0011 (\|B\| < 0.01)          |

Newtonian `fdt` control (any T): B ≈ +0.00014 (the shadow floor; flat in T).

**Falsification / pass criteria (registered):**
1. **Primary:** `fdt_relativistic` reduces |B| by **≥ 30×** vs `fdt` at every Θ,
   landing at |B| < 0.01 with **positive** sign (the shadow floor is +, not −).
2. The biased `fdt` arm reproduces the theorist's −0.31/−0.54/−0.73 to within
   finite-sample error (±0.03), confirming the harness matches the coded map.
3. Newtonian `fdt` control stays at |B| ≲ 1e-3.

Finite-MC tolerance: with ~1e4–1e5 walkers the standard error on B is O(1e-3),
so exact reproduction of +0.0011 is not required; the registered claim is the
≥30× collapse and the sign flip from strongly-negative to ~0⁺.
