# trainability-spike — experiment-engineer report

Task + acceptance criterion: build implicit/DEQ gradients through the shipped settle and a learned trajectory-ψ, and show **gradients flow end-to-end `query → φ → settle → ψ → loss` at usable wall-clock** + **the point-vs-trajectory ablation runs** — with the Hub's **Stage 0 axis-liveness gate blocking Part B**.
Status: **done, with the headline being a pre-registered NEGATIVE.** Part A ✅ (every registered gradcheck bar met, several by 3–4 orders). Acceptance half 1 ✅ (**0.500 s/step** point arm, **8.55 s/step** trajectory arm vs a declared 30 s budget; falsifier 300 s not fired). **⛔ STAGE 0 FAILED its registered gate on 3/3 seeds (0 qualifying cells of 588 examined per seed)** — so per the C2W1 amendment the ablation is **NOT filed as pillar 1's first datum**; it was run only as instrument evidence + to execute the never-before-run **trajectory launder**, and is labelled `gate_passed=False` in every artifact.

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R-1 — the theorist's `k* = 269` governs an ENDPOINT loss, not a TRAJECTORY loss, and the difference is 3 orders.** For `L(q_N)` I confirm Q4.2 exactly (rel. err **9.74e-4** at k=270 vs the predicted 1e-3). For `L(ψ(traj))` — the actual C2 read-out — the θ-gradient relative error at k=270 is **0.680**, and it does **not** follow `ρ^k` at any depth (0.695 / 0.690 / 0.685 / 0.679 / 0.680 / 0 at k = 0/50/100/180/270/400). A read-out that pools the *whole* window loses `O(1)`, not `O(ε)`, to truncation. *(Owner: theorist + every site quoting "unroll beyond ~270 steps is numerically worthless" without saying "of `∂q_N/∂θ`".)*
> **R-2 — theory §7 request 7's truncation recipe makes `φ` untrainable, and by an exact 0, not a small number.** Tail truncation enters the retained window through a `stop_gradient`, so `‖∂L/∂φ‖` is **exactly 0.0** at k = 0, 50, 100, 180, **270**, and only becomes nonzero (6.42e-3) at full backprop. Truncation *direction* is load-bearing and the theory does not name it. *(Owner: theorist §7 request 7; any C2 trainer built on it.)*
> **R-3 — monitor #10's `traj_stride` "dead axis" is STRUCTURAL, not a tuning artifact, and the handover wording should change.** `settled_point_psi` never touches the buffer: movement **exactly 0.000** noise units at every stride, 3 seeds. `tail_mean_psi` moves 4.5e-4–2.5e-2. The correct statement is not "the knob is inert" but **"no shipped ψ consumes the buffer"** — which is a different bug with a different fix. *(Owner: curator/Hub; handover §7 R3 from `full-clu-harness`.)*
> **R-4 — `full-clu-harness`'s PREREG refutation #4 is RESOLVED, and it is REFUTED, not confirmed.** The trajectory launder in `chlu/eval/dividend.py` has now been run for the first time (18 stride × seed cells). With a learned DeepSets ψ that **can** see the address block, `psi(q0_only)` = **0.129 against a chance of 0.125** — *at chance* — and the blank store scores **0.148** (leak fires in **3/18** cells, all marginal, all on seed 2). The harness's prediction ("the `q₀ = φ(x)` leak becomes live the moment your learned ψ can see the address block", N68 blanks 0.992–1.000) and **my own PREREG §4b prediction of blank ≥ 0.5** are both **refuted**. Hypothesis for why, untested: **pooled set read-outs dilute `q₀` to 1 of 150 points**; an *attention* ψ that can select the first point should leak where DeepSets does not. *(Owner: Hub → whoever holds doctrine I-2; and the untested attention arm is a one-flag follow-up.)*

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** **pillar 1 — expressive latents (point / trajectory / manifold)** + the trainability infrastructure every later C2 wave depends on.
- **Laundering control:** the point-vs-trajectory ablation is itself the internal launder (settled-point read = "trajectory deleted"); implicit gradients checked against truncated unroll (substitute-gradient control) on **both** a controlled toy and the real learned store; the **trajectory launder** (`full` / `q0_only` / `endpoints`) + **blank-store control** on every learned-ψ number; and — added after Stage 0's first pass diagnosed the registered baseline — a **blank-store probe control** (Stage 0b), which is the decisive one.
- **Falsifies:** (i) implicit ≠ truncated unroll beyond the registered tolerance → **did NOT fire**; (ii) wall-clock > 10× budget → **did NOT fire**.
- **Does NOT falsify:** a flat ablation *provided Stage 0 passed*. **Stage 0 did not pass, so no ablation number in this report is a pillar-1 result and none is offered as one.**

---

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/trainability-spike` @ `da214f7`, `7d2082f`, `d0dddf2` (base local `main @ 4160cf7`) |
| worktree | `../CHLU-trainspike`, **main venv reused** (`/Users/user/Desktop/CHLU/.venv`, no `uv sync` — w6 hazard avoided) |
| env | **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, scikit-learn 1.8.0, numpy 2.4.1, CPU |
| artifacts | `.claude/outputs/trainability-spike/` — `PREREG.md`, `exp_trajectory_read_{a,e2e,stage0,stage0b,b}_seed{0,1,2}.json`, `exp_trajectory_read_seed{0,1,2}.png` (4 panels: axis liveness · Stage-0 gain-vs-stride · the ablation stride curve · the truncation study) |
| commands | `PYTHONPATH=. python -u -m chlu.experiments.exp_trajectory_read --part {a,e2e,stage0,stage0b,b} --seed N [--n-rep 6] [--force-b]` |
| **seeds** | Part A: deterministic (seedless toy). e2e: seed 0 (single seed, declared). **Stage 0 / Stage 0b / Part B: seeds 0, 1, 2.** |
| store (all harness runs) | `exp_clu_system`'s **S0_baseline** stage config, `addr_dim 4`, `payload_dim 1`, `dim 5`, `ball_radius 1.0`, `capacity 8`, `n_atoms 2048`, `confine 0.05`, `atom_init_scale 1.0`, masked write 300 steps Adam(3e-3) — **8 live items, `sep/σ_q` = 6.830 / 6.761 / 6.915** (seeds 0/1/2). ⚠ Never S4's 3.07. |
| read | `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, 400 + 800 steps, `traj_stride` swept, `kinetic_mode newtonian_learned` (M = I), `σ_q 0.15`, no anneal, no retry |
| ridge | **OFF (`ridge = 0.0`) for every gradcheck**, so nothing is hidden. `theory_ridge(0.05, 0.05, 400, 1e-3, 1) = 0.35424` (theorist: 0.354) reported alongside; `theory_ridge` at the phase-2 read (γ=0.02, N=800) = **0.0698**, `ridge_alarm = False` (0.0698 vs median λ 4.34) |
| truncation | Part A endpoint gradcheck k ∈ {180, 270, 449} of N=1500; e2e truncation study k ∈ {0, 50, 100, 180, 270, 400} of the 400-step address phase; phase 2 always full |
| langevin / temperature | **N/A** — deterministic, T = 0, `p₀ = 0` |
| wake–sleep / lyapunov | **N/A** — neither `train.py` nor `train_generative.py` is used; ψ/φ are fitted with plain Adam (charter §2.4) |
| dtype | Part A **float64** (enabled inside the function, never at import); everything on the harness **float32** (the harness's own precision) |
| N94 maturity | ψ fits run **2000** Adam steps; e2e loop **20** steps after a warm-up compile; store writes 300 steps/item |

---

## 1. What I did

1. **`chlu/core/implicit_grad.py`** — a custom-VJP settle for the **shipped** damped velocity-Verlet map. Forward = `CHLU`'s own rollout (the integrator is **not** forked). Backward = the theorist's §7 request 1: solve the `d×d` system `(Hess V(q*) + λI) w = g`, then `θ̄ = −VJP_θ[∇V(·,q*)](w)`; `∂p*/∂θ` skipped (exactly 0); **zero cotangent for `q₀`/`p₀`**. Ridge defaults **off** and is a reported flag; `theory_ridge` is the zero-free-parameter value; `ridge_alarm` is the theorist's guard. Also `truncated_rollout` (with the `return_endpoint` hook the harness's stride layout needs), `settle_telemetry` (the **Q3.5 triple**: residual · λ_min · basin identity, from one `d×d` eigendecomposition), `coset_transport` (the Q2.2 closed form), and `GaussianWellsPotential` — the controlled toy with a known answer.
2. **`chlu/core/psi_readout.py`** — learned ψ over strided trajectory points: **DeepSets** and **attention pooling**, both accepting `settled_point` and `endpoints` as **degenerate cases at identical parameters** (`matched_pair` asserts it and a test checks the parameters are *bit-identical*, not merely equinumerous). Plus the doctrine-I-2 `store_relative` representation and a `LearnedPhi` that launches on the payload-zero manifold.
3. **`chlu/experiments/exp_trajectory_read.py`** — five parts (`a`, `e2e`, `stage0`, `stage0b`, `b`) + plot. Contains `differentiable_read`, a **traceable** re-implementation of `CluSystem.read`'s two phases (the frozen API mixes numpy into its diagnostics and cannot be differentiated), **tested to agree with it to float32 round-off**.
4. **Ran the Hub's Stage 0 gate on 3 seeds over 588 cells each**, then — because the registered baseline turned out to be gameable — added and ran **Stage 0b**, the blank-store probe control.
5. **65 → +37 tests**; no frozen/C1W27 file touched; **no CLI hook added** (`experiment_cmd.py` is `memory-gym-v0`'s this wave).

---

## 2. PART A — implicit/DEQ gradients (every registered bar met)

Controlled toy: 4 Gaussian wells on the unit ring, `s = 0.35`, `α = 0.05`, `amp = [1.0, 0.9, 1.1, 0.95]`, launch `c₀ + (0.2, −0.15)`, `L = ½‖q*‖²`, differentiated w.r.t. the 4 amplitudes. float64, N = 1500, `dt = 0.05`, `γ = 0.05`, **ridge 0**.

**Settle quality:** `‖p*‖ = 4.61e-16`, `‖∇V(q*)‖ = 4.86e-16`, `λ_min = 8.2139`, `cond(H) = 1.0012`, basin 0, `d_nearest = 0.0127`. (Prop Q1.1 confirmed: the fixed point has *exactly* zero momentum.)

| check | registered bar | **measured** | verdict |
|---|---|---|---|
| implicit vs **re-settled central finite differences** | ≤ **1e-5** | **5.112e-10** | ✅ **4.3 orders inside**; better than the theorist's own 1.3e-8 |
| implicit vs truncated unroll, **k = 180** | ≤ 3e-2 (expect 1e-2) | **1.0006e-2** | ✅ point prediction hit to **0.06 %** |
| implicit vs truncated unroll, **k = 270** ← primary | ≤ **3e-3** (expect 1e-3) | **9.736e-4** | ✅ point prediction hit to **2.6 %** |
| implicit vs truncated unroll, **k = 449** | ≤ 3e-5 (expect 1e-5) | **9.502e-6** | ✅ point prediction hit to **5.0 %** |
| implicit vs **full** backprop (k = N = 1500) | — | **1.938e-14** | ✅ machine precision |
| `(γ, dt)`-independence of the implicit answer | spread ≤ 1e-8 | **2.504e-11** | ✅ (γ ∈ {0.02,0.05,0.1,0.3} × dt ∈ {0.02,0.05,0.1}) |
| ridge bias at `λ_ridge = 0.35424` | theorist: "≈4.1 %" | **4.134 %** | ✅ |

⭐ **Q4.2 is confirmed as a law, not just a bound**: the measured truncation error is `ρ^k` to within a factor 1.05 at three depths spanning 4 orders (`ρ = √(1−γ) = 0.97468`).

**One registered check initially failed, and the failure is a confirmation.** With the theorist's underdamped-only settle budget `N = 2ln(1/tol)/γ`, the `(γ,dt)` spread was **1.286**, not ≤1e-8. Localised to the cells the theory itself predicts: at `(γ=0.3, dt=0.02)` the well mode is **overdamped** (`λ_crit = γ²m/(2(2−γ)dt²) = 66.2 > λ = 8.21`), so `ρ = 0.99069` and 185 steps is **16× too short** (residual 0.4196, `‖p*‖ = 0.0201` — the settle simply had not arrived). Using the **two-branch `ρ`** of Q4.2 the spread is **2.50e-11**. Reported as a measurement, not a bug: *`N_settle ≈ 2ln(1/tol)/γ` is `dt`-free only while the mode is underdamped.*

**The `ρ^N` geometric-death law, verified on the same object** (`‖∂‖q*‖²/∂q₀‖` by full unroll):

| N | 50 | 100 | 200 | 400 | 800 | 1500 |
|---|---|---|---|---|---|---|
| measured | 6.03e-1 | 1.36e-1 | 7.21e-3 | 4.01e-5 | 1.222e-9 | **3.99e-17** |
| `ρ^N` | 2.77e-1 | 7.69e-2 | 5.92e-3 | 3.51e-5 | 1.229e-9 | 1.96e-17 |

and by the **implicit** path the same quantity is **exactly 0.0** — which is the point of §3.

---

## 3. ⭐ ACCEPTANCE HALF 1 — gradients flow `query → φ → settle → ψ → loss`

Real learned store (S0, 8 live, `sep/σ_q = 6.83`, 57 472 B), batch 32, shipped 1200-step read, DeepSets ψ (**3553 params, identical in both arms**), `LearnedPhi` (325 params), plain Adam(1e-3), 20 measured steps after one warm-up compile.

| arm | compile | **median s/step** | budget (PREREG §3) | falsifier | loss (20 steps) |
|---|---|---|---|---|---|
| settled-point ψ (implicit settle) | 1.0 s | **0.500** | 30 s → **60× inside** | 300 s, not fired | 0.4812 → **0.0381** |
| trajectory ψ (full backprop) | 8.9 s | **8.550** | 30 s → **3.5× inside** | 300 s, not fired | 0.4961 → **0.0478** |

✅ **Acceptance half 1 MET.** The trajectory arm costs **17.1×** the point arm — that is the real price of the trajectory read and it should be quoted whenever the trajectory is proposed.

### 3.1 ⭐ PREREG §4c CONFIRMED — the settled-point read cannot train its own `φ`

| arm | `‖∂L/∂φ‖` | `‖∂L/∂ψ‖` | `‖∂L/∂θ‖` |
|---|---|---|---|
| settled_point, **implicit** | **0.0 exactly** | 0.834248 | 0.0337211 |
| settled_point, full unroll | **2.654e-09** | 0.834248 | 0.0337211 |
| trajectory | **6.421e-03** | 0.831923 | 0.0346242 |
| endpoints | 6.568e-03 | 0.769439 | 0.0171095 |

- Registered: ratio ≥ 1e3, point estimate ≥ 1e6. **Measured 6.421e-3 / 2.654e-9 = 2.42e6** (and ∞ against the implicit path's exact zero). ✅
- ⭐ **This is a structural pillar-1 argument that does not depend on any benchmark:** `Fix(T_θ)` contains no `q₀`, so `∂q*/∂q₀ = 0` a.e.; **a settled-point read-out sends no gradient to its read-in, and the trajectory read is the only channel that does.** It is the exact, non-numerical form of N61.
- **Substitute-gradient control on the REAL store, not just the toy:** implicit and full-unroll `‖∂L/∂θ‖` agree to **6 significant figures** (0.0337211 both).

### 3.2 ⭐ The truncation study — R-1 and R-2

Retain the last `k` of the 400-step address phase; ψ = trajectory (pools the whole buffer).

| retain `k` | 0 | 50 | 100 | 180 | **270** | 400 (full) |
|---|---|---|---|---|---|---|
| `‖∂L/∂φ‖` | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | 6.421e-3 |
| θ-grad rel. err vs full | 0.695 | 0.690 | 0.685 | 0.679 | **0.680** | 0 |
| theory `ρ^k` | 1.0 | 0.277 | 7.69e-2 | 9.89e-3 | **9.83e-4** | 3.51e-5 |

**Neither row follows `ρ^k`.** The θ row is flat at ≈0.68 — because a ψ that pools the whole window depends on the early points *directly*, and truncation deletes that dependence outright rather than attenuating it. The φ row is an exact zero for the same reason. Contrast with §2, where the *endpoint* loss follows `ρ^k` to 5 %. → R-1, R-2.

### 3.3 A3/A4 — the Q3.5 conditioning triple, wired as a trainer health check

Computed by **consuming** `monitors.py` semantics, never editing it; one `d×d` eigendecomposition per read.

| leg | value (batch 32, seed 0) |
|---|---|
| (i) residual `‖∇V(q*)‖` | median **1.73e-7**, max 4.40e-7 |
| (ii) `λ_min(H)` | median **3.383**, min 2.760; `λ_median` 4.336; `cond` 1.321; **0 negative modes** |
| (iii) basin identity | **0.844** correct |
| ridge | `λ_ridge(γ=0.02, N=800) = 0.0698`, **`ridge_alarm = False`** (0.0698 < 0.1 × 4.336) |

The three legs disagree exactly as the theorist predicted they would: the residual says "converged" (1e-7) and `λ_min` says "well-conditioned" (3.38) while **basin identity says 15.6 % of reads are in the wrong well.** Legs (i) and (ii) alone would have reported a healthy read. **Monitor #11 supplies one leg; the union is the health check.**

---

# 4. ⛔ STAGE 0 — THE HEADLINE. The trajectory channel is not measurably live at v0.

**Design.** Healthy S0 geometry (`sep/σ_q` = 6.83 / 6.76 / 6.92 on seeds 0/1/2, 8 live items — never S4's 3.07). Queries graded by **ambiguity**: `q = (1−t)c_i + t c_j + N(0, σ_q)`, `t ∈ {0, .1, .2, .3, .4, .45}`, competitor `j` drawn **uniformly among the other live items** (not `i`'s nearest neighbour — otherwise `j` is a deterministic function of `i` and "predict the competitor" collapses into "predict the winner"). **2016 queries/seed.** Probes: multinomial-logistic (L2, C=1) and kNN(5) for identity, RidgeCV and kNN for payload; 5-fold CV, **any dimensionality reduction fit inside the fold** (no transductive leak). Feature sets: `q0_only` · `q_star_only` (the classical 26-wave read) · `endpoints = [q0, q_addr, q*, p*]` (the **fair point baseline** — it holds the query *and* the settled point, so any trajectory gain is attributable to the intermediate points) · `full_pca` (the buffer projected to `dim(endpoints)` = 20, **capacity-matched**) · `full_raw`.
**Regimes swept (12 per seed):** `γ_read ∈ {0, 0.005, 0.02, 0.08, 0.2}` — ⭐ **including `γ_read = 0`, where theory Q1.1b says Liouville forbids an attractor so there is no settled point at all and the trajectory is all there is** — × `address_steps ∈ {50, 100, 400}` × `read_steps ∈ {50, 200, 800, 1600}` × `traj_stride ∈ {1, 2, 4, 8, 16, 32}` × window ∈ {phase1, phase2, both} × 4 targets × 4 probes × 3 ambiguity bands. **588 scored cells per seed, 1764 total.**

## 4.1 The gate

| seed | registered gate (`full_pca − endpoints ≥ +0.10` **and** ≥ 3σ, ambiguous band, competitor targets) | cells examined | wall |
|---|---|---|---|
| 0 | ⛔ **FAIL** — 0 qualifying | 588 | 629 s |
| 1 | ⛔ **FAIL** — 0 qualifying | 588 | 552 s |
| 2 | ⛔ **FAIL** — 0 qualifying | 588 | 614 s |

**Best cell anywhere in the swept range, per baseline** (ambiguous band, competitor targets):

| seed | feature | baseline | max gain | se | z | cell |
|---|---|---|---|---|---|---|
| 0 | full_pca | endpoints | +0.1008 | 0.0357 | 2.82 | γ=0.2, phase1, st 8, competitor_payload, knn_reg |
| 1 | full_pca | endpoints | +0.0580 | 0.0192 | 3.02 | γ=0.02, addr 50, phase1, st 8, competitor_id, knn |
| 2 | full_pca | endpoints | +0.0809 | 0.0425 | 1.90 | γ=0.08, phase1, st 8, competitor_payload, knn_reg |
| 0 | full_raw | endpoints | +0.1303 | 0.0357 | 3.65 | γ=0.2, phase1, st 8, competitor_payload, knn_reg |
| 1 | full_raw | endpoints | +0.0699 | 0.0192 | 3.64 | γ=0.02, addr 50, phase1, st 8, competitor_id, knn |
| 2 | full_raw | endpoints | +0.0640 | 0.0212 | 3.02 | γ=0.08, phase1, st 8, competitor_id, knn |

**Cross-seed replication (a post-hoc STRICTER criterion, per the standing multi-seed rule):** of the 196 cells present in all three seeds, **0 qualify on all three seeds** for `full_pca − endpoints`, `full_raw − endpoints`, or `full_pca − q0_only`. The seed-0 maxima are in *different cells* from the seed-1 and seed-2 maxima. **Three cells replicate for `full_raw − q0_only`** — and §4.3 shows what they are.

## 4.2 Axis liveness in monitor #10's own noise units (this travels with every number above)

| ψ | stride 1 | 2 | 4 | 8 | 16 | 32 | 3σ bar |
|---|---|---|---|---|---|---|---|
| `settled_point_psi` (shipped, S0–S5) | **0** | **0** | **0** | **0** | **0** | **0** | 3.0 |
| `tail_mean_psi` (shipped, S6) seed 0 | 0 | 5.6e-4 | 5.6e-4 | 7.8e-4 | 4.5e-4 | 2.0e-3 | 3.0 |
| `tail_mean_psi` seed 1 | 0 | 4.9e-4 | 4.9e-4 | 6.1e-4 | 4.9e-4 | 1.9e-3 | 3.0 |
| `tail_mean_psi` seed 2 | 0 | 8.0e-4 | 2.1e-3 | 4.4e-3 | 1.7e-3 | 2.5e-2 | 3.0 |

⭐ **The `settled_point_psi` row is exactly 0.000 at every stride and every seed — not "small", zero — because that ψ never reads the buffer.** The harness's monitor #10 trip (2.97e-4) is reproduced independently and re-diagnosed: **the knob is not inert, nothing consumes it.** → R-3.

## 4.3 ⭐ STAGE 0b — the blank-store probe control kills the one replicating effect

The only cross-seed-replicating effect is `full_raw` beating **`q0_only`**. But the trajectory *contains* `q₀` and the settle is a **nonlinear map of it**, so a 1500-dimensional trajectory is among other things a random-feature expansion of the query — and a linear probe on a feature expansion beats a linear probe on the raw query *whether or not anything is stored*. Control: the identical probe on trajectories from a **store with nothing written into it** (the launder discipline applied to the instrument itself).

`competitor_id`, linear probe, `full_raw − q0_only`, **ambiguous** band, stride 8:

| seed | γ_read | **live store** | **blank store** | excess (live − blank) | blank share of the effect |
|---|---|---|---|---|---|
| 0 | 0.02 | +0.0625 | +0.0193 | +0.0432 | 31 % |
| 1 | 0.02 | +0.1235 | +0.0506 | +0.0729 | **41 %** |
| 2 | 0.02 | +0.1116 | +0.0699 | +0.0417 | **63 %** |
| 0 | 0.005 | +0.1057 | +0.0387 | +0.0670 | 37 % |
| 1 | 0.005 | +0.1607 | +0.0551 | +0.1057 | 34 % |
| 2 | 0.005 | +0.1205 | +0.0580 | +0.0625 | 48 % |
| 0 | **0.0** | +0.1071 | +0.0402 | +0.0670 | 38 % |
| 1 | **0.0** | +0.1592 | +0.0580 | +0.1012 | 36 % |
| 2 | **0.0** | +0.1190 | +0.0521 | +0.0670 | 44 % |

**Three things this says.**
1. **31–63 % of the only replicating "trajectory gain" is reproduced by a store containing nothing.** It is a feature expansion of the query, not retrieved content.
2. **The residual excess is measured only against `q0_only` — a 5-dimensional baseline that the 1500-dimensional `full_raw` outranks by 300× in capacity.** Against the capacity-matched `endpoints`, the same cells are **negative** (§4.4). There is no baseline that is simultaneously fair and beaten.
3. ⭐ **The numbers at `γ_read = 0` are indistinguishable from those at `γ_read = 0.02`** (+0.107/+0.159/+0.119 vs +0.063/+0.124/+0.112). At `γ_read = 0` **no fixed point exists at all** (Liouville, theory Q1.1b) — if the trajectory carried anything a settled point cannot, this is the regime where it would show. It does not move.

## 4.4 The stride curve (quote the curve, not the endpoint) and the conservative read

`competitor_id`, linear, ambiguous band, reference regime, gain over the capacity-matched `endpoints` baseline:

| seed | st 1 | st 2 | st 4 | st 8 | st 16 | st 32 |
|---|---|---|---|---|---|---|
| 0 | −0.058 | −0.058 | −0.060 | −0.058 | −0.058 | −0.061 |
| 1 | −0.174 | −0.171 | −0.171 | −0.167 | −0.176 | −0.171 |
| 2 | −0.155 | −0.153 | −0.153 | −0.146 | −0.146 | −0.135 |

**Flat and negative at every stride, on every seed.** The curve has no structure to quote: 32× more trajectory resolution buys nothing.

At `γ_read = 0` (the conservative read), ambiguous band, all 12 (seed × target × probe) cells: the gain over `endpoints` is **negative in 12 of 12** (−0.030 to −0.199).

## 4.5 The instrument positive control — the honest limitation

Registered before running: the trajectory provably contains `q₀` and `q_star_only` provably does not, so `full − q_star_only` must be large or the probe is broken. `winner_id`, band=all, stride 8:

| seed | q0_only | q_star_only | endpoints | full_pca | full_raw |
|---|---|---|---|---|---|
| 0 | 0.8309 | 0.6473 | 0.8274 | 0.6815 | 0.7773 |
| 1 | 0.8378 | 0.6632 | 0.8423 | 0.7307 | 0.7902 |
| 2 | 0.8438 | 0.6776 | 0.8363 | 0.7287 | 0.7817 |

The control **fires but weakly** (`full_pca − q_star_only` = +0.034/+0.068/+0.051): the capacity-matched PCA-20 projection scores **below** the 5-dimensional `q0_only`, i.e. it discards information the trajectory demonstrably contains. **So the registered `full_pca` feature is under-powered, and I say so rather than quietly swapping baselines.** That is why §4.1 reports `full_raw` (no capacity match, all the information) alongside — and `full_raw` fails the cross-seed criterion against `endpoints` too. The negative therefore does not rest on the weak feature.

## 4.6 ⛔ Stage 0 verdict

> **Pillar 1's trajectory channel is not measurably live on this harness at v0.** Range swept: `γ_read ∈ [0, 0.2]` **including the conservative `γ_read = 0` where no fixed point exists**, `address_steps ∈ {50, 100, 400}`, `read_steps ∈ {50, 200, 800, 1600}`, `traj_stride ∈ {1…32}`, phase-1 / phase-2 / both windows, 4 targets, 4 probe families, 3 ambiguity bands, **588 cells × 3 seeds = 1764 scored cells**. The registered gate qualifies **0 cells on every seed**; no cell qualifies on all three seeds against any capacity-matched baseline; and the single replicating effect against the *unmatched* `q0_only` baseline is **31–63 % reproduced by a store with nothing in it**.
>
> ⇒ **Part B's ablation is NOT filed as pillar 1's first datum.** Per the C2W1 amendment, this negative *is* the finding, and it is a monitor-#10 result — which is exactly what monitor #10 exists to produce.

---

## 5. PART B — run, but **`gate_passed = False` in every artifact**

⛔ **Read this before any number below.** Stage 0 did not pass, so **none of §5 is a pillar-1 datum** and I do not offer it as one. It is here for exactly three reasons the amendment does not gate: (a) the acceptance criterion asks that the ablation machinery *run*; (b) the **trajectory launder** in `chlu/eval/dividend.py` had **never been executed** and running it was explicitly assigned to me; (c) it independently corroborates the Stage-0 null with a *learned* read-out rather than a probe. `gate_passed: false` is written into `exp_trajectory_read_b_seed{0,1,2}.json`.

**Setup.** Same store, same φ (identity), same bytes, same seeds, same read; **DeepSets ψ at 3553 parameters in every arm** (asserted at runtime and tested to be bit-identical at init); 2000 Adam(3e-3) steps on a fixed read; 70/30 train/test over 1344 ambiguity-graded queries; decode = nearest stored payload.

### 5.1 The ablation (3 seeds, mean ± se across seeds)

`winner_payload` — the standard read task:

| stride | point ψ (s0 s1 s2) | endpoints ψ | trajectory ψ | **traj − point** |
|---|---|---|---|---|
| 1 | 0.649 0.554 0.473 | 0.631 0.597 0.579 | 0.535 0.552 0.512 | **−0.026 ± 0.046** |
| 2 | ″ | ″ | 0.577 0.488 0.557 | **−0.018 ± 0.051** |
| 4 | ″ | ″ | 0.582 0.478 0.535 | **−0.027 ± 0.045** |
| 8 | ″ | ″ | 0.562 0.532 0.517 | **−0.022 ± 0.038** |
| 16 | ″ | ″ | 0.562 0.485 0.495 | **−0.045 ± 0.034** |
| 32 | ″ | ″ | 0.569 0.465 0.535 | **−0.036 ± 0.049** |

**Flat and slightly negative across the whole stride curve; the per-seed sign flips** (seeds 0 and 1 negative at all 6 strides, seed 2 positive at all 6). Chance 0.125. ⭐ **A single-seed run here would have produced a confident ±0.09 "result" of either sign** — this is the multi-seed rule earning its keep.

`competitor_payload` (the pillar-1-native "distribution over answers" target): every arm sits at **0.114–0.168 against a chance of 0.125**. The ablation is `+0.004` to `+0.028`, i.e. a difference between two read-outs that are both at chance. Not a result.

**Ambiguity-resolved** (`winner_payload`, stride 8, n ≈ 130/band/seed): unambiguous **+0.016 ± 0.055**, ambiguous **−0.110 ± 0.051**. The trajectory read is, if anything, **worse** exactly where charter §2.1(c) predicts it should be better.

**Internal dividend** (`dividend()`, stride 8, seed 0, `metric = decode(trajectory ψ) − decode(settled-point ψ)`): **−0.0867**; controls `endpoints_psi 0.6312`, `chance 0.125`; bytes `full 57 472 / launder 160 = 359.2×`, `matched = False`.

### 5.2 ⭐ THE TRAJECTORY LAUNDER — first execution, and it **refutes** the doctrine's prediction

`trajectory_launder(ψ_trajectory, traj, state)`, 18 (stride × seed) cells:

| quantity | mean | range |
|---|---|---|
| `full` — ψ on the real trajectory | **0.5298** | 0.465–0.582 |
| `q0_only` — ψ on a buffer of nothing but `q₀` | **0.1293** | 0.032–0.213 |
| `endpoints` — ψ on `{q₀, q*}` only | 0.1976 | 0.109–0.272 |
| `blank_store` — the same learned ψ reading a store with nothing in it | **0.1484** | 0.087–0.270 |
| bar (`chance + 3 se`) | 0.190 / 0.204 | — |

- ⭐ **`q0_only` = 0.129, i.e. AT CHANCE (0.125).** The learned trajectory ψ is **not** a classifier on `φ(x)`. The N68 leak did not materialise.
- **`blank_leak` fired in 3 of 18 cells — all three on seed 2, at strides 2, 4, 16, marginally (0.208–0.270 vs a 0.204 bar).** Zero of 12 on seeds 0 and 1.
- ⛔ **My PREREG §4b predicted blank ≥ 0.5 for a raw-trajectory learned ψ. REFUTED: 0.148 mean.** So is `full-clu-harness`'s PREREG refutation #4, which predicted the leak would go live "the moment your learned ψ can see the address block". It can see it, and it does not use it.
- **Mechanism (offered as a hypothesis, not a measurement):** a *pooled set* read-out dilutes `q₀` to 1 of 150 points under mean/max pooling. The prediction that follows and that I did **not** test: an **attention** ψ, which can select the first point, should leak where DeepSets does not. `AttentionPsi` is implemented and one flag away (`--family attention`). → §9.

### 5.3 What §5 is allowed to be quoted as

Only this: *"with a learned, matched-parameter, matched-byte DeepSets ψ trained for 2000 steps on the shipped read, the trajectory arm does not beat the settled-point arm on any stride, on any of 3 seeds, with a cross-seed mean of −0.02 to −0.05; and the trajectory launder shows the learned ψ is not reading `φ(x)`."* It is **corroboration of §4's null with a different instrument**, and it carries §4.2's axis-liveness numbers wherever it goes.



---

## 6. PREREG scorecard

`PREREG.md` was written before any measured run.

| # | prediction | outcome |
|---|---|---|
| §1 | implicit vs FD ≤ 1e-5 | ✅ **5.11e-10** |
| §1 | implicit vs unroll ≤ 3e-2 / 3e-3 / 3e-5 at k = 180/270/449 | ✅ **1.00e-2 / 9.74e-4 / 9.50e-6** — the *point* predictions (1e-2/1e-3/1e-5) hit to 0.06 % / 2.6 % / 5.0 % |
| §1 | `(γ,dt)` spread ≤ 1e-8 | ◐ **first attempt 1.286 — FAILED with the underdamped-only step budget**, then **2.50e-11** with Q4.2's two-branch `ρ`. The failure localises exactly where the theory says it must (overdamped `(γ=0.3, dt=0.02)`), so it is a confirmation of Q4.2, reported as such, not a silent fix |
| §1 | `‖p*‖ ≤ 1e-10` | ✅ 4.61e-16 |
| §1 | ridge bias "≈4.1 %" | ✅ **4.134 %** |
| §3 | ≤ 30 s / training step; falsifier > 300 s | ✅ **0.500 s** (point) / **8.550 s** (trajectory) |
| **§4a** | **Stage 0 PASSES on T2/T3 in the ambiguous band (gain ≥ +0.10 at ≥3σ), FAILS in the unambiguous band** | ⛔ **REFUTED — H_B rejected, H_A (axis dead) upheld.** 0 qualifying cells of 588, on 3/3 seeds; 0 cells replicate across seeds against any capacity-matched baseline |
| §4a | H_A: gain ≤ +0.02 in both bands | ◐ **partially** — the *capacity-matched* gain is ≤ +0.10 everywhere and ≤0 at the reference cell, but raw-feature gains reach +0.13 against `q0_only`; §4.3's blank control shows 31–63 % of that is a store-free feature expansion. H_A's *conclusion* is upheld, its *bound* was optimistic |
| §4a | "if it passes in the unambiguous band too, suspect a `q0` leak" | ✅ **this fired and was correct** — the seed-0 best cell (+0.193) scored **+0.183 in the unambiguous band**, which is what exposed it as a kNN-baseline artifact |
| **§4c** | `‖∂L/∂φ‖_traj / ‖∂L/∂φ‖_point ≥ 1e3`, point estimate ≥ 1e6 | ✅ **2.42e6** (and exactly ∞ against the implicit path's true zero) |
| §4c | the unrolled point arm is numerically dead too (predicted scale 1.1e-8) | ✅ **2.65e-9** — same order, prediction stands |
| §4b | ablation `= +0.06` (range [0, +0.15]) in the ambiguous regime | **NOT SCORED — the gate closed.** Scoring it would be exactly the laundering the amendment forbids. §5's numbers are reported as instrument evidence only |
| §4b | stride-sweep gain monotone non-increasing | **NOT SCORED** (same reason); the Stage-0 stride curve is flat and negative (§4.4) |
| §4b | blank-store leak with a raw-trajectory learned ψ (**≥ 0.5**), none with `store_relative` | ⛔ **REFUTED — blank = 0.148 (mean of 18 cells), `psi(q0_only)` = 0.129 at a chance of 0.125.** The learned DeepSets ψ does not read `φ(x)` at all. `full-clu-harness`'s own refutation-#4 prediction falls with it. → R-4 |

**Two pre-registered predictions of mine failed and both failures were informative** — the `(γ,dt)` budget (which confirmed the theory's overdamped rider) and the Stage-0 gate itself (which is the report's headline). *A pre-registered prediction that survives is evidence; one that fails is a finding.*

## 7. Test suite

**`PYTHONPATH=. python -m pytest -q` on the branch (worktree, main venv, JAX 0.9.0): `862 passed, 31 warnings in 1027.82s (17:07)` — zero failures.** The pre-existing suite was 825 (`full-clu-harness`'s final count) and I added 37, so **825 + 37 = 862 exactly: nothing pre-existing broke.**

`ruff check chlu/ tests/`: **clean** (all checks passed) at every commit.

Both new test modules carry the repo's x64/float32 isolation fixture (handover §7.2): `test_implicit_grad.py` turns x64 **on** inside an autouse fixture (the 1e-5 FD bar is below the float32 noise floor) and restores it; `test_psi_readout.py` pins **float32** because it compares against the harness's own float32 read. Verified in the full-suite ordering above, not only in isolation.

New tests: **37** — `tests/test_implicit_grad.py` (19) · `tests/test_psi_readout.py` (18). They encode the registered bars, so a regression in the implicit gradient fails CI rather than a report.

## 8. Git footprint

- **Branch** `agent/experiment-engineer/trainability-spike`, **worktree `../CHLU-trainspike`**, base local `main @ 4160cf7` (did not move under me; rebase was a no-op).
- **Commits:** `da214f7` implicit/DEQ gradients through the shipped dissipative settle · `7d2082f` learned ψ over the strided trajectory + the pilot runner + 37 tests · `d0dddf2` Stage 0b blank-store probe control, truncation study, plots.
- **Files touched — all NEW, zero read-only violations** (checked file-by-file against the task's list): `chlu/core/implicit_grad.py` · `chlu/core/psi_readout.py` · `chlu/experiments/exp_trajectory_read.py` · `tests/test_implicit_grad.py` · `tests/test_psi_readout.py`.
- **NOT touched:** `clu_system.py`, `monitors.py`, `clu_controller.py`, `eval/dividend.py`, `exp_clu_system.py`, `memory_gym.py`, `exp_memory_gym.py`, **`chlu/cli/experiment_cmd.py` (no CLI hook added — `memory-gym-v0` owns it this wave)**, `chlu/config.py`, **`chlu/core/integrators.py` (the shipped Verlet step is used through `CHLU`'s own rollout; it is neither edited nor forked)**, and every other C1W27 file.
- ψ / implicit-grad config lives in `SettleSpec` and `PsiSpec` **inside my own modules**, per the file-ownership rule.
- **Not pushed, not merged.** Branch left for Hub review.

## 9. Open questions / follow-ups / risks

1. **The negative is scoped to *this* harness at v0, and I would not generalise it.** `K = 8`, `d = 4`, `m = 1`, one query law, one store family, a *frozen* store (ψ was fitted on reads from a landscape that was never trained to make its trajectory informative). The honest claim is "the shipped read's trajectory carries no measurable content beyond its endpoint **on this instrument**", not "trajectories cannot carry content".
2. ⭐ **The most promising untested lever, and it is one line of the charter I could not reach:** nothing in the *write* objective ever asked the trajectory to encode anything. `train_memory_landscape` shapes `V` so that the **settled point** is right. A trajectory read cannot beat a point read on a landscape optimised for the point. **The next honest test of pillar 1 is a write objective with a trajectory term**, not a better ψ. This reframes the negative from "pillar 1 is dead" to "pillar 1 has never been *written* for".
3. **A probe null is a probe null.** I used linear + kNN at fixed capacity. A high-capacity nonlinear probe might find content — but it would then have to beat the same blank-store control, which is the test that actually bit here.
4. **The 17.1× wall-clock cost of the trajectory arm** is measured with full backprop over 1200 steps. R-1/R-2 say truncation cannot buy it back for a whole-window ψ. A ψ that pools only the *late* window would be truncatable — untested.
5. **Single-seed e2e.** §3's wall-clock and gradient-norm table is seed 0 only (it is an infrastructure measurement, not a performance claim). Stage 0 / 0b / Part B are 3 seeds.
6. **`differentiable_read` duplicates the phase-2 rollout** when `implicit_q_star=True` (once inside `implicit_settle`'s forward, once for the buffer). ~1.5× avoidable cost on the trajectory arm; not optimised because it never approached the budget.
7. **Risk to flag:** if a later wave wants `∂L/∂φ` through a settled-point read, **there is no gradient to get** — not a small one, none. Any C2 design that assumes φ trains through the settle is wrong by theorem, and the fix is architectural (read the trajectory, or supervise `q₀` directly), not numerical.

## 10. Proposed handover updates (for the Hub)

**§2/§3 (architecture + config).**
- New modules `chlu/core/implicit_grad.py` (custom-VJP settle, `SettleSpec`, `theory_ridge`/`ridge_alarm`, `truncated_rollout`, `settle_telemetry`, `coset_transport`) and `chlu/core/psi_readout.py` (`PsiSpec`, `DeepSetsPsi`, `AttentionPsi`, `LearnedPhi`, `matched_pair`), runner `chlu/experiments/exp_trajectory_read.py`. **Config deliberately NOT in `chlu/config.py`** (C2W1 ownership rule); `SettleSpec`/`PsiSpec` carry it. **No CLI hook** — run by module invocation.
- **Gradients now flow `query → φ → settle → ψ → loss` on the full harness**, at **0.500 s/step** (settled-point ψ, implicit settle) and **8.550 s/step** (trajectory ψ, full backprop), batch 32, shipped 1200-step read.

**§1 (physics / formalism) — add:**
- ⭐ **`∂q*/∂q₀ = 0` exactly, so a settled-point read-out sends NO gradient to its read-in.** Measured: implicit **0.0**, full unroll **2.65e-9**, trajectory read **6.42e-3** (ratio **2.42e6**). This is the exact form of N61 and it is a *structural* argument for a trajectory read that needs no benchmark.
- The theorist's Q1/Q4.2 package is **confirmed on the shipped code**: implicit vs re-settled FD **5.11e-10**; truncation error `= ρ^k` to within 5 % at k = 180/270/449; `(γ,dt)` spread **2.50e-11**; ridge bias **4.134 %** at `λ_ridge = 0.35424`.

**§7 (known issues / live) — add:**
- ⚠ **R-1: `k* = 269` is the truncation depth for `∂q_N/∂θ`, NOT for `∂ψ(traj)/∂θ`.** Measured θ-gradient error at k=270 for a whole-window ψ: **0.680**, flat in `k`. Quoting `k*` for a trajectory read-out is wrong by 3 orders.
- ⚠ **R-2: tail truncation zeroes `∂L/∂φ` exactly.** Theory §7 request 7's recipe is correct for θ and makes φ untrainable.
- ⚠ **R-3: re-scope the monitor-#10 `traj_stride` finding.** `settled_point_psi` moves **exactly 0.000** noise units at every stride (3 seeds × 6 strides): **no shipped ψ consumes the buffer.** "The knob is inert" is the wrong diagnosis.
- ⚠ **The Q3.5 triple's legs genuinely disagree in production:** residual 1.7e-7 ✓, `λ_min` 3.38 ✓, **basin identity 0.844 ✗** on the same batch. A two-leg health check would have passed a read that is wrong 15.6 % of the time.

**§8/§10 (record) — add:**
- ⛔ ⭐ **Pillar 1's trajectory channel is NOT measurably live on the C2W1 harness at v0** — 1764 scored probe cells over 3 seeds, `γ_read ∈ [0, 0.2]` (including the conservative `γ_read = 0` where no fixed point exists), read lengths 50–1600, strides 1–32, ambiguity-graded queries; **0 cells pass the pre-registered gate on any seed**, and the only cross-seed-replicating effect is **31–63 % reproduced by a store with nothing in it**. This is the program's first honest datum on its highest-novelty pillar, and it is negative.
- ⭐ **The reframe that follows, and the recommended next task:** *nothing in the write objective has ever asked the trajectory to carry anything.* `train_memory_landscape` optimises the **settled point**. Pillar 1 has not been falsified; it has never been **written** for. The cheap decisive next experiment is a write objective with a trajectory term, not a better ψ.
