# PREREG — trainability-spike (experiment-engineer, C2W1)

**Written before any measured run.** Repo base: local `main @ 4160cf7`, worktree `../CHLU-trainspike`,
branch `agent/experiment-engineer/trainability-spike`. Env: **main venv** `/Users/user/Desktop/CHLU/.venv`
(no worktree `uv sync`; w6 hazard avoided), JAX version to be reported in the flag-provenance table.

Sources of the registered constants: `.claude/outputs/trainability-spike-theory.md` §7 (requests 1–3, 7),
§Q4.2 (truncation depth table), §Q1.3 (the identity being checked).

---

## 0. Dial declaration (echoed)
- **Dial:** pillar 1 — expressive latents (point / trajectory / manifold) + the trainability infrastructure.
- **Laundering control:** the point-vs-trajectory ablation *is* the internal launder (settled-point read =
  "trajectory deleted"); plus implicit-vs-truncated-unroll (substitute-gradient control); plus the
  harness-native blank / same-keys / settle-deleted controls and the **trajectory launder**
  (`eval/dividend.py::trajectory_launder`, full vs `q0_only` vs `endpoints`) which becomes MANDATORY the
  moment ψ can see the address block.
- **Falsifies:** (i) implicit gradients disagree with truncated unroll beyond the tolerance registered in
  §1; (ii) wall-clock per training step > **10× the budget declared in §3**.
- **Does NOT falsify:** a flat point-vs-trajectory ablation *provided Stage 0 passed*; needing a ridge;
  a v0 learned ψ weaker than the handcrafted read.

---

## 1. Registered gradcheck tolerances (from the theorist, §7 request 3)

Object under test: `∂L/∂θ` where `L = ½‖q*(θ) − target‖²`, `q*` = settled point of the **shipped damped
velocity-Verlet map** (3 substeps then `p ← (1−γ)p`), `θ` = a parameter of `V_θ`.

| check | depth / setting | registered PASS bar | source |
|---|---|---|---|
| implicit vs **truncated unroll**, `k = 180`, settle residual ≤ 1e-4 | γ=0.05, dt=0.05 | **rel. err ≤ 3e-2** | theory §7.3 row 1 |
| implicit vs **truncated unroll**, `k = 270`, settle residual ≤ 1e-5 | γ=0.05, dt=0.05 | **rel. err ≤ 3e-3** ← primary | theory §7.3 row 2 (recommended default) |
| implicit vs **truncated unroll**, `k = 449`, settle residual ≤ 1e-7 | γ=0.05, dt=0.05 | **rel. err ≤ 3e-5** | theory §7.3 row 3 |
| implicit vs **finite differences** (re-settled, central, float64) | γ=0.05, dt=0.05 | **rel. err ≤ 1e-5** | C1 prior art (1e-5 / 1e-7 / 3.1e-7); theory reached 1.3e-8 |
| `(γ, dt)`-independence of the implicit answer | γ∈{0.02,0.05,0.1,0.3}, dt∈{0.02,0.05,0.1} | **spread ≤ 1e-8** (theory: 5.5e-14 in float64 numpy) | theory §Q1.3 |
| `∂p*/∂θ = 0` and `‖p*‖ ≈ 0` at the fixed point | — | `‖p*‖ ≤ 1e-10` (float64) | theory §Q1.1 |

⚠ Registered **explicitly NOT** expected: 1e-5 at k=270 (theory says unreachable). A disagreement at
`‖∇V(q_N)‖ > 1e-3` or `λ_min < λ_ridge` is **reported as a measurement of the Q3.5 triple, not a failure.**

**Ridge (theory §7 request 2, zero free parameters):**
`λ_ridge = 2γ m ln(1/tol) / (N_settle (2−γ) dt²)` → **0.354** at (γ=0.05, dt=0.05, N=400, tol=1e-3, m=1).
Reported as a flag on every number; **alarm (never silently proceed) if `λ_ridge > 0.1·median(λ)`**.
Gradchecks are run at **λ_ridge = 0** (healthy toy) so the ridge is not hiding an error; the ridged value
is reported alongside.

## 2. Registered truncation depth (theory §Q4.2 + §7 request 7)
- Phase 1 (`γ_address = 0.05`, 400 steps): retain **k = 270** steps of backprop; `ρ = 0.97479`,
  `ρ^270 = 1.0e-3` (0.1 % of the gradient discarded).
- Phase 2 (`γ_read = 0.02` as shipped — **not** the theory's `γ_read = 0` case): `ρ = √0.98 = 0.98995`,
  `k*(ε=1e-3) = 684` ≥ the shipped 800 steps × 0.855, so **retain the phase-2 window in full**.
- Truncation boundary = **the phase-1/phase-2 seam**, per §7 request 7.
- Anything earlier than the seam uses the **implicit** gradient.

## 3. Declared wall-clock usability budget
Machine: this Mac, main venv, CPU. Configuration: `addr_dim 4`, `payload_dim 1`, `dim 5`,
`n_atoms 2048`, batch **32 queries**, shipped read budget **400 + 800 = 1200** Verlet steps,
ψ = learned (DeepSets or attention, ≈10–30 k params), Adam.

> **DECLARED BUDGET: ≤ 30 s per training step**, measured as the median of ≥10 steps **after** JIT
> warm-up (compile time reported separately, and explicitly excluded from the budget — a one-off).
> **FALSIFIER: > 300 s / step** (10× the budget).

Rationale for 30 s: a 200-step ψ/φ pilot must fit in ≈100 min, which is one uninterrupted session slot
on this machine given the ~20 min JAX cold start. Anything slower is not "usable infrastructure".

## 4. Predicted sign and magnitude — committed numbers

### 4a. STAGE 0 (the blocking axis-liveness gate)
Probes: multinomial-logistic (linear, ridge-regularised) and kNN(k=5), 5-fold CV over held-out query
draws, on the **healthy** S0 geometry (`sep/σ_q = 6.83`), 8 live items, `σ_q = 0.15`.
Feature sets, all at **matched capacity** (trajectory PCA-reduced to the endpoint feature dimension):
`q0_only` · `endpoints = [q0, q_addr, q*, p*]` · `full = strided trajectory`.
Targets: **T1** winner identity · **T2** competitor (2nd-nearest) identity · **T3** competitor payload
(regression R²) · **T4** ambiguity level `t` (regression R²).

Two competing hypotheses, both registered:

| | **H_A — axis dead (monitor #10 stands)** | **H_B — ambiguity hypothesis (charter §2.1(c))** |
|---|---|---|
| basis | `traj_stride` moves the read 2.97e-4 noise units; S6 decode bit-identical to S5 | a trajectory passing near competing wells encodes a distribution over answers; a point cannot |
| T2 gain `full − endpoints`, ambiguous (`t ≥ 0.35`) | **≤ +0.02** (inside 3σ) | **≥ +0.10**, point estimate **+0.15** |
| T2 gain, unambiguous (`t ≤ 0.10`) | ≤ +0.02 | **≤ +0.03** (the trajectory has *no reason* to help here) |
| T3 (competitor payload) ΔR², ambiguous | ≤ +0.02 | **≥ +0.10** |

> **GATE RULE (binding):** Stage 0 **PASSES** iff at least one (probe, target) cell shows
> `full − endpoints ≥ +0.10` **and** ≥ 3σ, in **some** cell of the swept regime
> (ambiguity `t ∈ {0, .1, .2, .3, .4, .45}` × `traj_stride ∈ {1,2,4,8,16,32}` ×
> `γ_read ∈ {0.005, 0.01, 0.02, 0.04, 0.08}` × phase-1-only / phase-2-only / both).
> If **no** cell qualifies → **STOP, do not run Part B's ablation**, and report
> *"pillar 1's trajectory channel is not measurably live on this harness at v0; here is the range swept."*
> ⭐ Prediction registered: **Stage 0 PASSES on T2/T3 in the ambiguous band and FAILS in the
> unambiguous band.** (If it passes in the *unambiguous* band too, suspect a q0 leak and re-launder.)

### 4b. PART B — the point-vs-trajectory ablation (only if Stage 0 passes)
Matched ψ family, **matched parameter count** (identical architecture; only the input set differs),
matched bytes, matched φ, matched seeds (3 seeds: 0, 1, 2).

- **Registered point prediction: `decode(trajectory ψ) − decode(point ψ) = +0.06`**,
  registered range **[0.00, +0.15]**, in the **ambiguous** regime (`t ≥ 0.35`).
- **Registered: `= 0.00 ± 0.02` in the unambiguous regime** (the standard read task the harness ran).
- **Stride sweep curve prediction: monotone non-increasing gain in stride** —
  gain(stride 1) ≥ gain(2) ≥ gain(4) ≥ gain(8) ≥ gain(16) ≥ gain(32), with
  gain(32) ≤ ½ · gain(1). (Quote the curve, not the endpoint.)
- **Trajectory launder prediction:** with a learned ψ that sees the address block, the blank-store control
  **rises above its chance bar** (N68 configuration; blanks scored 0.992–1.000 there). Registered:
  blank-store decode ≥ **0.5** for the raw-trajectory ψ, and **≤ chance + 3se** for the
  store-relative (`traj − q0`) ψ. If the raw-ψ blank does NOT rise, the launder is reported as
  a refutation of the doctrine's I-2 prediction on this harness.

### 4c. The φ-gradient structural prediction (new, registered here)
The theory (§Q1.3) gives `∂q*/∂θ = −H⁻¹∂_θ∇V` and `∂p*/∂θ = 0`; the fixed-point set does not contain
`q0`, so **`∂q*/∂q0 = 0` almost everywhere** (basin identity is piecewise constant). Therefore:

> **Registered prediction:** with the implicit settle, the gradient reaching φ through the
> **settled-point** read is **exactly 0** (float noise: ≤1e-12 relative), while through the
> **trajectory** read it is O(1). Registered ratio
> `‖∂L/∂φ‖_trajectory / ‖∂L/∂φ‖_settled-point ≥ 1e3`, point estimate **≥ 1e6**.
> Under a *truncated unroll* instead of the implicit solve the point-arm ratio is set by
> `ρ_1^{400} ρ_2^{800} = 0.97479^400 · 0.98995^800 = 3.6e-5 · 3.1e-4 ≈ 1.1e-8`
> — so the unrolled point arm is *numerically* dead too, and the two computations should agree
> on that verdict. **If this survives, it is a structural pillar-1 result: the trajectory read is what
> makes φ trainable at all.** If refuted (point-arm φ gradient is O(1)), my implicit implementation
> or the theory's Q1.1 is wrong, and that is the finding.

## 5. Compute order (anything unreached is **NOT RUN**, never a null)
1. **A1–A2** — `implicit_grad.py` + gradcheck vs truncated unroll + finite differences (toy, float64;
   no harness, no store, fast).
2. **A5 / acceptance half 1** — end-to-end `query → φ → settle → ψ → loss` gradient + wall-clock
   measurement against the §3 budget (+ the §4c φ-gradient prediction).
3. **STAGE 0** — axis liveness on the healthy S0 store (blocking gate).
4. **B1–B2** — `psi_readout.py` (DeepSets + attention) and the ablation + stride curve
   **only if Stage 0 passes**, incl. the trajectory launder and the blank control.
5. **A3–A4** — conditioning telemetry (`λ_min`, ridge, the Q3.5 triple) wired as a trainer health check
   by *consuming* `monitors.py`, never editing it.
6. **B3** — the "what a trajectory carries that a point cannot" probe (internal, v0).

## 6. Scope declarations made in advance
- Seeds: gradchecks are deterministic (seedless / seed 0). Stage 0 and Part B run **seeds 0, 1, 2**;
  any single-seed number is labelled as such and is not a paper number.
- The store is built by the shipped harness write path (`CluSystem.write_stream`) at the **S0** stage
  config, `ball_radius = 1.0` — never S4's collapsed geometry.
- `CluSystem.read()` mixes numpy into its diagnostics and is therefore **not traceable end-to-end**.
  I will implement a differentiable two-phase read in my own module and **verify it reproduces
  `CluSystem.read`'s trajectory and `q*` to float32 round-off** rather than editing the frozen API.
  If that check fails I report it and request the hook from the Hub.
- No edits to `clu_system.py`, `monitors.py`, `clu_controller.py`, `dividend.py`, `exp_clu_system.py`,
  `config.py`, `cli/experiment_cmd.py`, `integrators.py`, or any C1W27 file.
