# PREREG — `phi-particle-head` (C2W2, experiment-engineer)

Written **before any measured run** (protocol §5 pre-registration rule). Base local `main @ 233fd9e`,
branch `agent/experiment-engineer/phi-particle-head`, worktree `../CHLU-phipart`, main venv reused.

Calibration anchors inherited from `trainability-spike`: implicit-vs-FD **5.11e-10** (float64 toy, bar
1e-5); truncation `= ρ^k` within factor 1.05; the φ-gradient ratio **2.42e6** (trajectory 6.421e-3 vs
unroll 2.654e-9); wall-clock **0.500 s/step** point / **8.550 s/step** trajectory against a 30 s budget.

---

## 1. D3 — does the mass gauge dissolve under a trajectory read? (charter §A2.2)

**Object.** Real store (`exp_clu_system` `S0_baseline`, `sep/σ_q ≈ 6.8`, `kinetic_mode
newtonian_learned` so `M` is live in `T`), matched-parameter ψ pair from `matched_pair` (identical
initial parameters, only `input_mode` differs), particle head emitting `(q₀, log_mass, friction)`
per query, loss = MSE(ψ, payload).

**H0 (charter §A2.2 holds).** The settled point is `M`- and `γ`-independent (Prop Q1.1:
`Fix(T_θ) = {(q,0): ∇V = 0}` contains neither `M` nor `γ`), the trajectory is not.

| # | quantity | registered prediction |
|---|---|---|
| P1 | `‖∂L/∂log_mass‖`, point ψ, **implicit** `q*` (mass as the model's `log_mass` leaf) | **exactly 0.0** (bitwise; the implicit VJP is `−VJP_θ[∇V]`, and `∇V` contains no `M`) |
| P2 | `‖∂L/∂log_mass‖`, point ψ, **full-unroll** backprop, per-particle head | **> 0 but ≤ 1e-6 × (P4)** — geometric death `ρ₁^400 ρ₂^800 ≈ 3.5e-5 × 3.1e-4 ≈ 1e-8`, floored by float32 round-off |
| P3 | `‖∂L/∂friction‖`, point ψ, full unroll | same band as P2 |
| P4 | `‖∂L/∂log_mass‖`, **trajectory** ψ, full unroll | **nonzero, point estimate 1e-2, ≥ 1e-4** |
| P5 | `‖∂L/∂friction‖`, trajectory ψ, full unroll | **nonzero, point estimate 1e-1, ≥ 1e-4** (γ multiplies every step's momentum; strictly stronger than mass) |
| P6 | ratio `P4/P2` (the mass-gauge dissolution ratio) | **point estimate 1e6, FLOOR 1e3** |
| P7 | FD cross-check, trajectory arm, central FD on a scalar offset added to `log_mass`, `h = 1e-2`, float32 | rel. err **≤ 5e-2** |
| P8 | FD cross-check on the float64 `GaussianWellsPotential` toy, `h = 1e-4` | rel. err **≤ 1e-5** (spike bar) |
| P9 | wall-clock per gradcheck arm (batch 32) | ≤ **30 s**; falsifier at 300 s |

⛔ **Falsifier (headline if it fires).** If **P4 < 1e-10** (i.e. the trajectory arm's mass gradient is
numerically zero too), charter **§A2.2 is REFUTED** and *"mass as selector"* stays dead. Declared
numerically-zero threshold: `< 1e-10` absolute **or** `< 1e-10 ×` the same arm's `‖∂L/∂ψ‖`.
**Does NOT falsify:** P2/P3 being small-but-nonzero (that is the geometric-death law); P4 landing an
order off its point estimate.

⚠ **Truncation direction is load-bearing (spike R-2).** Every particle/φ gradient in this task is taken
at **`retain=None` (full backprop)**. Tail truncation enters through a `stop_gradient` and would make
`‖∂L/∂φ‖` and `‖∂L/∂(log_mass, friction)‖` **exactly 0** at every finite `k`, which would be a
measurement artefact, not a refutation. Registered here so a zero cannot be mis-read.

## 2. D1/D2 — φ plumbing, particle head, byte ledger

| # | quantity | registered prediction |
|---|---|---|
| P10 | `phi_bytes` = 4 B × learnable scalars (float32), all arms | identity **0 B**; pca (4→4 + mean) **80 B**; mlp(4→5, h32, d2) **325 params = 1300 B**; mlp **+ particle head** (outputs 5+5+1=11) **523 params = 2092 B**; cnn/gru: measured, predicted 10³–10⁵ B |
| P11 | identical-φ invariant | enforceable in code: a mismatched φ across arms **raises** (`PhiMismatchError`), never warns. Predicted **enforceable** — no arm structurally needs its own φ |
| P12 | default-off | with `particle_head=False` the read is **bit-identical** to the shipped read (γ ≡ `gamma_address`/`gamma_read`, M ≡ the model's global mass): max abs traj diff **exactly 0.0** |
| P13 | friction band | declared clamp `γ ∈ [0.02, 0.20]` (0.4×–4× the shipped `gamma_address = 0.05`). Monitor #1 trips at `ρ_conv > 1e-6`; doctrine R2 measured the shipped read at **4.3e-7** (2.3× inside). Predicted: `ρ_conv` at the band's **low** end (γ=0.02) **exceeds 1e-6** (i.e. the low end of the declared band is monitor-#1-hot and must be reported), at the shipped γ it does not |

## 3. D4 — monitor repairs, re-scored trip counts

| # | quantity | registered prediction |
|---|---|---|
| P14 | **#6 dead-band**: how many of the 58 first-ever trips survive `slope_loss < −eps and slope_acq ≤ 0`, `eps = 1e-9 × scale` | **29 survive of 58** (band 28–30). Derivation: 58 loud-trip lines in `memory-gym-v0/run_full.log` carry `value = slope_acq`; **30** have `|slope_acq| < 1e-12` (values 0.0, −2.97e-17, −5.93e-17, −1.48e-16) and **28** have `|slope_acq| ≥ 5.9e-4`. The one exact `0.0` may have a genuine `slope_loss`, which is why the point estimate is 29, matching gym R2's independent count |
| P15 | repairs **#1, #9, #2** | predicted **ALREADY LANDED** in `monitors.py` at `233fd9e` (ρ_conv/δ predicate with corr demoted; Δ_ret effect size with corr direction-only; `U`, `ρ_ex`, INAPPLICABLE at `U < 0.01`) ⇒ **zero trip-state change** from them |
| P16 | repair **#10 tier (a)** | new code (access-counting proxy + startup failure). Not wired into the frozen harness ⇒ **zero trip-state change** on the anchor (monitor #10 stays inapplicable there: no `knob_reads` in `ctx.extras`) |
| P17 | repair **#7** | `gauge_orbit_residual` is already trajectory-wise; adding the `kinetic_mode` parameterisation is a **pytest scope**, not a runtime trip ⇒ **zero trip-state change**. Predicted residuals: newtonian_identity/newtonian_learned **≤ 1e-6**, relativistic **O(1/c²) ≈ 1e-2–1e-1 at c=1** (a SCOPE, not a pass) |
| P18 | **D4 acceptance diff** on the C2W1 anchor `overload/load1x_shipped@s{0,1,2}` (478× cell) vs `.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json` | **exactly one** monitor's state may change — `objective_divergence` (#6) — and only in the direction TRIP → clear. Predicted: `@s0` and `@s2` were `['vacuous_gate','objective_divergence']`, `@s1` was `['vacuous_gate']`; predicted post-repair **all three = `['vacuous_gate']`** *iff* those two trips are ε-artefacts, else unchanged. **Every other monitor bit-identical.** Any other change is a regression, to be reported as such and not rationalised |

## 4. Compute budget
≤ 3 h of measured runs (hard stop 6 h). Anchor cells cost 15–19 s each (C2W1 log) × 3 seeds; the
gradcheck is a handful of grad evaluations at batch 32.

## 5. Scoring
Every row above is scored ✅ / ⛔ / — in the report's PREREG scorecard, with the measured value.
