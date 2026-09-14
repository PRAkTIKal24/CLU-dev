# phi-particle-head — experiment-engineer report

Task + acceptance criterion: build the strong-φ read-in that **parametrizes the particle** (`q₀` + per-particle mass + friction, default-off), gradcheck it, decide charter **§A2.2** (does the mass gauge dissolve under a trajectory read?), and land the six already-specified monitor repairs with a **before/after trip-state diff on the C2W1 anchor**.
Status: **done.** §A2.2 is **SUPPORTED** on 3/3 seeds (not refuted). D4 acceptance **met exactly**: across the *whole* C2W1 gym (28 cells, 112 monitor-#6 readings) **only monitor #6's trips changed** — 58 → 27 — and every other monitor's trip count is **bit-identical**.

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R-1 — `memory-gym-v0` R2's "29 of 58" is wrong; the exact count is 31 of 58.** R2 was inferred from the loud log, which prints only `slope_acq`. Re-running all 28 cells with both slopes recorded: **31** trips die at `|slope_write_loss| ≤ 5.19e-17` and **27** survive with `|slope_write_loss| ≥ 8.13e-7`. *(Owner: curator — `memory-gym-v0` R2 and any site quoting "29"; and monitor #6's row: "TRIPS ×58" → **"×58 pre-repair, ×27 after the C2W2 dead-band"**.)*
> **R-2 — doctrine I-7's "the gauge is Newtonian-only" is too weak: it is `newtonian_learned`-**only**.** Parameterising the #7 gauge by `kinetic_mode` (the repair itself) measures residuals **0.2505 / 2.52e-7 / 0.0274** for identity / learned / relativistic. Under `newtonian_identity` `T = ½p²` ignores `M`, so `(M,V,p₀)→(λM,λV,λp₀)` rescales `V` and `p₀` with *nothing* to compensate them — it is **not a gauge orbit at all**, not "exact". N76 ("mass stores nothing") still holds there, trivially. *(Owner: theorist + curator; `monitors.GAUGE_SCOPE` already carries the corrected text.)*
> **R-3 — 19 cell-level rows of the C2W1 `monitor_table` change** (`objective_divergence: TRIP → clear`), listed in §5.3. There is no curator this wave; §5.3 is the only corrected record until C2W3.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial / pillar:** **none — instrument + infrastructure.** No performance claim, no leaderboard entry, no dividend. Everything here is consumed by C2W2's race and C2W3.
- **Laundering control:** the identical-φ invariant is **enforced in code** (`PhiMismatchError`, raises never warns) and exercised in both directions; the matched-parameter ψ pair (`matched_pair`, bit-identical parameters) is the internal control for the D3 measurement; the point arm is the "trajectory deleted" substitute for the trajectory arm.
- **Falsifies:** trajectory-arm mass gradient numerically zero (**did NOT fire**); the identical-φ invariant unenforceable (**did NOT fire**); a repair changing a C2W1 verdict in an invalidating direction (**did NOT fire** — the only changes are #6 TRIP→clear, i.e. false trips removed).
- **Does NOT falsify:** a strong φ costing bytes (declared, §3.1); small-but-nonzero particle gradients at the point arm (measured 1e-8, FD-confirmed real).

---

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/phi-particle-head` @ `376208a`, `42332d7`, `81420bd`, `804e7f4`, `a7d55d9` (base local `main @ 233fd9e`) |
| worktree | `../CHLU-phipart`, **main venv reused** (`/Users/user/Desktop/CHLU/.venv`, no `uv sync` — w6 hazard avoided) |
| env | **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, numpy 2.4.1, CPU (identical to `trainability-spike`) |
| commands | `PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part {phi,grad,monitors,gym-rescore} [--seed N] [--batch 16]` — **no CLI hook added** (`experiment_cmd.py` is `traj-write-objective`'s) |
| artifacts | `.claude/outputs/phi-particle-head/` — `PREREG.md`, `exp_phi_particle_{phi,grad_seed0/1/2,monitors,gym-rescore}*.json`, `run_gym_rescore.log` |
| seeds | φ/monitors: 0 (+ anchor seeds 0,1,2); **D3 gradcheck: seeds 0, 1, 2**; gym re-score: the C2W1 plan's own 28 (family, arm, seed) cells |
| store (D3 + φ) | `exp_clu_system` **S0_baseline**, `addr_dim 4`, `payload_dim 1`, `dim 5`, `capacity 8`, `n_atoms 2048`, `atom_init_scale 1.0`, masked write 300 steps Adam(3e-3), **8 live items, `sep/σ_q` = 6.830** |
| read | `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, 400 + 800 steps, `traj_stride 8`, **`kinetic_mode newtonian_learned`** (M live in `T` — the mass measurement is vacuous under `newtonian_identity`), `σ_q 0.15`, no anneal, no retry |
| gradients | **`retain=None` (full backprop) everywhere** — spike R-2: tail truncation enters through a `stop_gradient` and would give an artefactual exact 0. Ridge **OFF (0.0)** on every implicit solve |
| dtype | store/φ/ψ **float32** (the harness's precision); the FD cross-checks additionally run in **float64** (store upcast; dynamics otherwise identical) and on the float64 toy |
| φ | `PhiSpec(family=mlp, hidden 32, depth 2, residual, particle_head=True)`, `log_mass_center 0.5413 (= softplus⁻¹1)`, `log_mass_span 1.0`, **friction band [0.02, 0.20], init 0.05** |
| ψ | `deepsets`, hidden 32, depth 2, stride 1, `matched_pair` — **3553 params both arms, bit-identical at init** |
| batch / wall | D3 batch **16**; measured runs total **≈17 min** (φ 38 s · grad 118/99/92 s · anchor 55 s · gym re-score 598 s) against a ≤3 h budget |
| monitors | `ObjectiveDivergenceMonitor(window=3, **eps_rel=1e-9**)`; all other bands at their doctrine defaults |
| langevin / temperature / wake–sleep | **N/A** — deterministic reads, `T = 0`, `p₀ = 0`; neither `train.py` nor `train_generative.py` is used |

---

## 1. PREREG scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | point ψ, **implicit** `q*`: `‖∂L/∂log_mass‖` **exactly 0.0** | **0.0 bitwise**, 3/3 seeds (`np.all(g.log_mass == 0)`) | ✅ |
| P2 | point ψ, full unroll: `> 0` but `≤ 1e-6 ×` P4 | **8.73e-9 / 3.98e-8 / 1.09e-8** = **5.0e-6 / 3.4e-6 / 5.8e-6 ×** P4 | ⛔ **MISS ×5** (nonzero as predicted, but 5× larger than the registered bound) |
| P3 | point ψ friction, same band as P2 | **6.52e-8 / 1.62e-7 / 6.73e-8** | ✅ |
| P4 | **trajectory** ψ mass, nonzero, pt est 1e-2, ≥ 1e-4 | **1.74e-3 / 1.17e-2 / 1.88e-3** | ✅ |
| P5 | trajectory ψ friction, pt est 1e-1, ≥ 1e-4 | **2.38e-2 / 4.14e-2 / 3.29e-2** | ✅ |
| P6 | **ratio P4/P2**: pt est 1e6, **floor 1e3** | **1.998e5 / 2.949e5 / 1.726e5** (friction: 3.64e5 / 2.56e5 / 4.88e5) | ✅ (floor met by 200×; point estimate 5× high) |
| P7 | FD cross-check, trajectory arm, **float32**, rel err ≤ 5e-2 | **0.656** (mass), 0.289 (friction) | ⛔ **MISS** — diagnosed, see §2.3; resolved by an unregistered float64 re-run at **1.05e-7** |
| P8 | float64 toy FD, rel err ≤ 1e-5 | **2.79e-9** (mass), **1.80e-6** (friction) | ✅ (3.6 orders inside) |
| P9 | wall-clock ≤ 30 s/arm; falsifier 300 s | **5.95 s** point / **4.76 s** trajectory (seed 0) | ✅ |
| P10 | φ byte costs: identity 0 B; pca 80 B; mlp 1300 B; mlp+head 2092 B; cnn/gru 10³–10⁵ B | **0 / 64 / 5524 / 6316 / 25 260 / 15 020 B** | ◐ identity ✅ exact, cnn+gru ✅ in range; **pca/mlp MISS** — I predicted a 2-layer *direct* trunk, the shipped one is `in→hidden→hidden` + head (measured table in §3.1) |
| P11 | identical-φ invariant enforceable in code | **enforced**: raises `PhiMismatchError` on a rebuilt arm, accepts the shared instance | ✅ |
| P12 | default-off **bit-identical** to the shipped read | **0.0 exactly** vs `differentiable_read` (traj and `q*`) | ✅ |
| P13 | the band's **low** end is monitor-#1 hot; shipped γ is not | `ρ_conv` **8.11e-4** (γ=0.02) · **2.02e-5** (0.03) · **4.45e-7** (0.04) · **2.21e-7** (0.05) · 5.00e-7 (0.10) · 4.66e-7 (0.20); edge 1e-6 | ✅ (crossing between 0.03 and 0.04) |
| P14 | **29 of 58** #6 trips survive the dead-band (band 28–30) | **27 survive, 31 killed** | ⛔ **MISS by 1** — and the derivation's source (gym R2's "29") is itself wrong: see R-1 |
| P15 | repairs #1/#9/#2 already landed ⇒ zero trip-state change | already landed (verified in code + 6 tests); **zero change**: gym-wide totals identical | ✅ |
| P16 | repair #10 tier (a) ⇒ zero trip-state change on the anchor | `dead_axis` **inapplicable** on every anchor cell (no `knob_reads` in the frozen harness) | ✅ |
| P17 | repair #7 ⇒ zero trip-state change; identity/learned ≤ 1e-6, relativistic O(1/c²) | zero trip-state change ✅; residuals **0.2505 / 2.52e-7 / 0.0274** ⇒ **identity prediction REFUTED** (see R-2) | ◐ |
| P18 | anchor diff: **exactly one** monitor may change (#6), TRIP→clear only, all else bit-identical | **exactly that**, and it holds gym-wide over 28 cells | ✅ |

**Score: 12 ✅ · 2 ◐ · 4 ⛔.** Both ⛔ numerics (P2, P7) are float32-resolution stories with a float64 resolution; P14's miss is a **correction to a C2W1 registry line**, not a failure of the repair; P17's ◐ is a **finding** (R-2).

---

## 2. ⭐ D3 — the mass gauge DOES dissolve under a trajectory read (§A2.2 SUPPORTED)

### 2.1 The headline table (real store, matched-parameter ψ, full backprop)

| arm | path | `‖∂L/∂log_mass‖` | `‖∂L/∂friction‖` | `‖∂L/∂ψ‖` |
|---|---|---|---|---|
| settled-point ψ | **implicit** `q*` (mass as the model's own `log_mass`) | **0.0 — exactly, bitwise** | n/a (γ is static in `SettleSpec`) | 0.815 |
| settled-point ψ | full unroll, per-particle head | **8.73e-9** | **6.52e-8** | 0.815 |
| **trajectory ψ** | full unroll, per-particle head | **1.74e-3** | **2.38e-2** | 0.770 |
| **ratio traj / point** | | **1.998e5** | **3.643e5** | — |

Seeds 1 / 2 reproduce it: mass ratio **2.949e5 / 1.726e5**, friction ratio **2.559e5 / 4.884e5**, verdict identical. Declared numerically-zero threshold `1e-10` (absolute or relative to the arm's own ψ-gradient) — the trajectory arm is **7 orders above it**.

> ⛔ **The falsifier did NOT fire. Charter §A2.2 is SUPPORTED: "mass as selector" is live for the first time** — a trajectory read-out sends O(1e-3) gradient to a per-query mass and O(1e-2) to a per-query friction, while a settled-point read-out sends 1e-8 (unroll) or **exactly 0** (implicit).

**Why the point arm's zero is exact, not lucky.** The implicit VJP is `θ̄ = −VJP_θ[∇V(·,q*)](w)`, and `∇V` contains neither `M` nor `γ`; so the mass cotangent is structurally zero (Prop Q1.1: `Fix(T_θ) = {(q,0) : ∇V = 0}`). The 8.7e-9 of the unrolled point arm is the **geometric-death remnant** of `ρ₁^400 ρ₂^800 ≈ 1e-8`, and it is *real*, not round-off: the float64 FD reproduces it to **1.03e-5 relative** (§2.3). The same quantity through the model's global `log_mass` reads **0.0 (implicit) vs 9.56e-9 (unroll)** — the two paths' disagreement *is* the truncation remnant.

### 2.2 The controlled toy (float64), same statement with a clean floor

| loss | `‖∂L/∂log_mass‖` | `‖∂L/∂γ‖` | FD rel err (mass) |
|---|---|---|---|
| **endpoint** `½‖q_N‖²`, N=1500 | **1.26e-16** (machine zero) | 3.04e-15 | — (FD is 0.0: below float64 resolution) |
| **trajectory** (strided, N=1500) | **5.57e-4** | 1.27e-2 | **2.79e-9** |
| ratio | **4.42e12** | | |

### 2.3 The gradcheck, and an honest float32 story

| check | bar | float32 (shipped precision) | float64 (same store, upcast) |
|---|---|---|---|
| trajectory arm, AD vs central FD (mass) | ≤5e-2 (registered) | **0.656** ⛔ | **1.05e-7** at h=1e-4 (0.045 at h=1e-3, 0.654 at h=1e-2) |
| trajectory arm, AD vs FD (friction) | — | 0.289 | **1.35e-7** |
| point arm, AD vs FD (mass) | — | unresolvable | **1.03e-5** |

Two effects, separated by the h-sweep: (i) the trajectory loss is **strongly curved in `log M`** — central-FD error falls as `h²` (0.654 → 0.045 → 1.05e-7 over h = 1e-2 → 1e-3 → 1e-4), so the registered `h = 1e-2` was simply too coarse, not the AD wrong; (ii) in float32 the FD **noise floor** is `ε·L/2h ≈ 3e-6`, which is **400× the point arm's true gradient** — FD cannot resolve the point arm at the store's own precision at all. Registered bar missed; the claim survives at 1e-7 in float64. *(The float64 real-store cross-check was added after the float32 pass and is therefore **unregistered** — declared as such.)*

### 2.4 Wall clock
5.95 s (point) / 4.76 s (trajectory) per full grad at batch 16 — inside the 30 s budget, falsifier (300 s) not fired. `--part grad` end-to-end: 118 / 99 / 92 s per seed including compiles, the float64 re-run and the toy.

---

## 3. D1/D2 — the φ interface, the particle head, the joint dial

### 3.1 Byte ledger — `phi_id` / `phi_bytes` on every arm (§A4.3), measured
`phi_bytes = 4 B × learnable scalars` (float32 = the store's own precision). At `in_dim=4 → dim=5, addr_dim=4, hidden=32, depth=2`:

| variant | family | head | params | **phi_bytes** | `phi_id` |
|---|---|---|---|---|---|
| identity | identity | off | 0 | **0** | `8d8680aeac1ba324` |
| pca (frozen 4×4 projection) | pca | off | 16 | **64** | `19defc43cfad00bf` |
| mlp | mlp | off | 1381 | **5 524** | `c3206495f5980868` |
| **mlp + particle head** | mlp | **on** | 1579 | **6 316** | `92097e32d9323750` |
| cnn (`ConvTrunk`, 1×8×8, ch (8,16)) | cnn | on | 6315 | **25 260** | `d07f57d58c865dfc` |
| gru (T=4, F=2) | gru | on | 3755 | **15 020** | `9b8c6e4360c246cd` |

The widened head costs **+198 params / +792 B** on the MLP trunk (outputs `dim` → `2·dim+1`), and it is **paid for in the ledger on every arm** — a launder reading the same φ pays the same price. Every row also carries `d = 4`, `n_atoms = 2048`, `bytes_per_well = (d+m)·4 = 20`.

### 3.2 ⭐ The identical-φ invariant, enforced in code (not in prose)
`phi_fingerprint` hashes **architecture *and* parameter bytes**; `assert_identical_phi` / `SharedPhi.check` **raise `PhiMismatchError`**. Measured: four arms (`clu`, `baseline_knn`, `trajectory_launder`, `blank_store`) drawing from one `SharedPhi` all carry `phi_id = cb309ba7bb6781d7`; handing one arm a re-built φ of the *same architecture* raises (`ae55198573f02426 != cb309ba7bb6781d7`). **No arm structurally needed its own φ** — the urgent falsifier did not fire, so C2W2's race can assert the invariant before emitting any number.

### 3.3 The particle head, and "default-off" made structural
Head emits `(q₀, log_mass, friction)`; `M = softplus(log_mass_center + span·tanh(·))` (positive by construction, the repo's convention) and `γ = lo + (hi−lo)·σ(· + b₀)` (inside the declared band by construction, `= friction_init` at init). **With the head off the launch emits `log_mass = friction = None`**, so a reader falls back to the model's own `M` and the config's `γ` — measured **max |Δtraj| = 0.0 exactly** against `exp_trajectory_read.differentiable_read`. (Pinning a *live* head to the nominal values instead gives 8.2e-7 — round-off scale, because `softplus(centre) = 1` is not bit-equal to the store's own inertia. That is why default-off is `None`, not a nominal value.)

At init the head is a small perturbation of the shipped read: `std(log_mass) = 0.0121`, `std(γ) = 1.1e-4`, `mean(γ) = 0.05042`.

### 3.4 The declared friction band vs monitor #1 (where we sit)
Band **[0.02, 0.20]** (0.4×–4× the shipped `γ_address = 0.05`). Monitor #1 trips at `ρ_conv > 1e-6`; doctrine R2 measured the shipped read at 4.3e-7.

| γ | 0.02 | 0.03 | 0.04 | **0.05 (shipped)** | 0.10 | 0.20 |
|---|---|---|---|---|---|---|
| `ρ_conv` | 8.11e-4 | 2.02e-5 | 4.45e-7 | **2.21e-7** | 5.00e-7 | 4.66e-7 |
| trips #1 | ⛔ | ⛔ | clear | clear | clear | clear |

⚠ **The low end of the declared band is monitor-#1 hot on this harness** (crossing between 0.03 and 0.04). Reported, not hidden: the band is harness-specific, and a C2W2/C2W3 consumer running this store should set `friction_lo = 0.04` (or accept a #1 trip as an instrument-validity flag). Our shipped point sits **4.5× inside** the edge, consistent with doctrine R2's 2.3×.

### 3.5 One strong encoder, wired end to end and smoke-run
`family="cnn"` → `phi_encoders.ConvTrunk` (imported, **not forked**) → particle head → `particle_read` → DeepSets ψ → MSE. Batch 8: loss **0.5080 → 0.4873** in one Adam step, `‖∂L/∂φ‖ = 0.202`, `‖∂L/∂ψ‖ = 1.186`, 8.7 s compile+grad, 3.2 s/step. **Gradient reaches φ.** Not trained to a benchmark number, as instructed.

### 3.6 The `(d, atom-budget)` joint dial — DECLARED, not swept
`joint_dial(d, K)` returns `d`, `n_atoms`, `n_atoms_required`, `co_scaling_ok`, `k_learned_designed = 2^d`, `reach_sigma_scale = √d`, and the law
`n_atoms ≥ max(atoms_per_item·K, 384, round(512·√2^d))`, **asserted** by `assert_joint_dial` (raises) and pinned by a test to `CluSystemConfig.n_atoms` for **every d in 1…8**. At the store used here: `d=4, n_atoms = 2048 = required`, `atoms_per_item 256`, `bytes_per_well = 20 ∝ (d+m)`. No `d`-sweep was run (task §3). ⛔ `K_learned(8)` is quoted **lower-bounded**, never bracketed.

---

## 4. D4 — the monitor repairs

| repair | source | status |
|---|---|---|
| **#1** ρ_conv/δ trip, `corr` demoted to a diagnostic | I-3, R3 | ✅ **already landed at `233fd9e`** — verified in code and by 3 tests (a healthy `corr = 0.977` does not trip; an unconverged settle and a non-moving `q*` do) |
| **#9** effect size `Δ_ret ≤ 0.10`, corr direction-only | I-4, R4 | ✅ **already landed** — verified (perfect rank correlation + tiny effect ⇒ no trip) |
| **#2** report `U`, `ρ_ex = D/U`, **INAPPLICABLE** when `U < 0.01` | I-6 | ✅ **already landed and confirmed so on the anchor** (`settle_argmin: inapplicable` on every anchor cell) — "verify it is already so; if yes, say so and move on" |
| **#6** dead-band | gym R2 | ⭐ **LANDED** (this wave) — §5 |
| **#10 tier (a)** O(1) access-counting config proxy | I-8 | ⭐ **LANDED** — `ConfigAccessProxy` counts every attribute read; `assert_knobs_live` raises `DeadKnobError` at startup on a declared-but-never-read field; `knob_extras()` feeds monitor #10 and now sets `knob_tier_a_implemented: True`. Not wired into the frozen harness (read-only to me) ⇒ zero trip-state change, as predicted |
| **#7** whole trajectory + parameterised by `kinetic_mode` | I-7 | ⭐ **LANDED** — `gauge_orbit_residual` was already trajectory-wise; `GAUGE_SCOPE` + a parameterised pytest now cover all three modes, **and the identity mode's result contradicts the doctrine's wording (R-2)**. A second test shows the endpoint comparison is vacuous: on a *non*-gauge mass perturbation the trajectory residual is >100× the endpoint residual |

**#6's implementation.** `slope_loss < −eps and slope_acq ≤ 0` with `eps = eps_rel · max|loss|` over the window, `eps_rel = 1e-9` (relative to the objective's own magnitude, so it survives any rescaling of the loss). `eps_rel = 0` **reproduces the pre-repair predicate exactly**, and every reading now carries `tripped_pre_repair`, which is what makes the diff below exact rather than a second stochastic run.

⚠ Monitors remain **guards, never losses**: no monitor quantity enters any objective anywhere in this branch.

---

## 5. ⭐ D4 ACCEPTANCE — the before/after trip diff (the citable table)

### 5.1 The anchor (`overload/load1x_shipped`, the 478× cell), 3 seeds
| cell | C2W1 trips (artifact) | now | #6 trips pre → post | other monitors |
|---|---|---|---|---|
| `@s0` | `['vacuous_gate', 'objective_divergence']` | **`['vacuous_gate']`** | 4 → 1 | **bit-identical** |
| `@s1` | `['vacuous_gate']` | **`['vacuous_gate']`** | 2 → 1 | **bit-identical** |
| `@s2` | `['vacuous_gate', 'objective_divergence']` | **`['vacuous_gate']`** | 3 → 0 | **bit-identical** |

Every changed trip maps one-to-one to **repair #6**, and the killed readings are visible: `@s0` kills three readings at `slope_write_loss = −5.795e-20` and keeps one at **−4.99e-4**; `@s2` kills three at `−9.06e-22`; `@s1` kills one at `−5.66e-22` and keeps one at `−8.13e-7`.

### 5.2 ⭐ Gym-wide (all 28 C2W1 cells re-run, 598 s) — the strongest form of the acceptance
| monitor | trips **pre-repair** | trips **post-repair** |
|---|---|---|
| overdamping (#1) | 142 | **142** |
| settle_argmin (#2) | 9 | **9** |
| vacuous_gate (#3) | 190 | **190** |
| addressing (#5) | 138 | **138** |
| **objective_divergence (#6)** | **58** | **27** |
| certificates (#8) | 128 | **128** |
| lifetimes (#9) | 128 | **128** |
| dead_axis (#10) | 0 | **0** |
| reach (#11) | 7 | **7** |
| starvation (#12) | 170 | **170** |

- **`n_trips_pre_repair = 58` reproduces C2W1's 58 exactly** on 112 applicable readings — i.e. the whole gym re-ran deterministically under JAX 0.9.0 and the repairs.
- **`n_new_trips = 0`.** A dead-band can only remove trips; it did.
- **`monitors_changed_other_than_6 = []`.** No unexplained trip-state change anywhere ⇒ no regression.
- **The separation is 10 orders wide**, so `eps_rel` is not a tuned threshold: killed trips have `|slope_write_loss| ≤ **5.19e-17**` (max), survivors `≥ **8.13e-7**` (min); the applied `eps` ranged 2.85e-15…2.55e-10 — *any* eps in that decade-wide gap gives the identical answer. (5.19e-17 is exactly the `−5.2e-17` gym R2 quoted.)

### 5.3 The corrected `monitor_table` rows (no curator this wave — **cite this table**)
19 cell-level rows change, **all** `objective_divergence: TRIP → clear`:

`overload/load1x_ref8@s0` + `/annealed` · `overload/load1x_ref3@s0` + `/annealed` · `overload/load1x_shipped@s0` + `/annealed` · `overload/load1x_shipped@s2` + `/annealed` · `manifold/base@s0` + `/annealed` · `manifold/base@s2` + `/annealed` · `aggregate/base@s0/annealed` · `aggregate/base@s2/annealed` · `recency/base@s0/annealed` · `recency/base@s1/annealed` · `recency/base@s2/annealed` · `aggregate/tight@s0/annealed` · `aggregate/tight@s1/annealed`.

Per-cell #6 counts pre → post (only cells with a change shown): `load1x@s0` 2→1 · `load1x@s1` 2→1 · `load1x@s2` 1→0 · `load1x_ref8@s0` 3→0 · `load1x_ref3@s0` 3→0 · `load1x_shipped@s0` 4→1 · `load1x_shipped@s1` 2→1 · `load1x_shipped@s2` 3→0 · `aggregate/base@s0` 1→0 · `aggregate/base@s2` 1→0 · `recency/base@s0` 1→0 · `recency/base@s1` 2→1 · `recency/base@s2` 1→0 · `manifold/base@s0` 4→1 · `manifold/base@s1` 2→1 · `manifold/base@s2` 3→0 · `aggregate/tight@s0` 4→3 · `aggregate/tight@s1` 4→3 · `manifold/ridge@s0` 2→1. Unchanged: `ref8@s0` 1→1 · `ref16@s0` 2→2 · `overload/base@s0` 2→2 · `overload/base@s2` 3→3 · `aggregate/base@s1` 1→1 · `aggregate/tight@s2` 2→2 · `reach_free@s0` 2→2.

⚠ **Monitor #6 remains APPLICABLE and TRIPPING (27×) after the repair** — it is not disabled, and the gym's own clean configuration (`overload/load1x_shipped`) now trips **only #3**, not #3 and #6.

---

## 6. How I verified (commands + observed output)

```
PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part phi          # 38.4 s
PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part grad --seed {0,1,2} --batch 16
                                                                               # 118 / 99 / 92 s
PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part monitors --seeds 0 1 2   # 55 s
PYTHONPATH=. python -u -m chlu.experiments.exp_phi_particle --part gym-rescore  # 597.8 s, 28 cells
PYTHONPATH=. python -m pytest tests/test_phi_particle.py tests/test_monitors.py -q   # 69 passed
PYTHONPATH=. python -m pytest tests/ -q --no-cov              # 923 passed, 0 failed, 995.5 s
ruff check chlu/ tests/                                       # All checks passed
```
Test additions: **`tests/test_phi_particle.py` (22 new)** + **`tests/test_monitors.py` (+9)**. Full-suite status is in §8 (Open items).

Three test failures were **real findings**, not test noise, and each changed the code or the claim: (i) the toy's endpoint mass gradient is 1.09e-8 at N=800 and only machine-zero at N=1500 (the geometric-death law, so the test uses the spike's N); (ii) a traced `γ` is not bit-identical to a Python-float `γ` (float32 last bits — the Python path is untouched, and the test now says so); (iii) the #7 gauge **fails** under `newtonian_identity` (R-2).

---

## 7. Findings, in the order they matter

1. ⭐ **Charter §A2.2 is SUPPORTED, first measurement ever.** Trajectory reads make the mass and friction of the particle trainable: ratio **1.7e5–2.9e5** (mass) and **2.6e5–4.9e5** (friction) against the settled-point arm, 3/3 seeds, with the point arm **exactly 0** through the implicit path. "Mass as selector" is live. C2W3's factored store can now treat `(q₀, M, γ)` as the address.
2. **The point arm's residual is real, not noise** — 8.7e-9, confirmed by float64 FD to 1.03e-5 relative. It is the `ρ₁^400 ρ₂^800 ≈ 1e-8` geometric-death remnant of the unroll, and it vanishes exactly under the implicit path. Anyone measuring "the settled point does see the mass a little" is measuring truncation depth, not physics.
3. **Friction is a stronger channel than mass** (ratio 2.4e-2 vs 1.7e-3, ~14×): γ multiplies every step's momentum, mass only rescales the timescale. If C2W3 wants one particle attribute, it should take friction first.
4. **31 of the gym's 58 monitor-#6 trips were an ε-artefact** (not 29), with a 10-order gap between artefact and signal. The gym's headline monitor row needs the corrected count (R-1).
5. **The #7 mass gauge is `newtonian_learned`-only** (R-2) — a sharpening of doctrine I-7 that only appeared because the repair required parameterising it by `kinetic_mode`.
6. **The declared friction band's low end is monitor-#1 hot** on the S0 store (§3.4). This is the doctrine's "harness-specific band" made concrete: a particle head allowed to shrink γ freely will drive the store into an overdamping trip, and monitor #1 will say so — which is the system working.
7. **No arm structurally needs its own φ**: the identical-φ invariant is enforceable and enforced. C2W2's race can assert it before any cell runs.

---

## 8. Open questions / follow-ups / risks
- **Full suite: `923 passed, 0 failed` in 995.5 s** (`pytest tests/ -q --no-cov`, log copied to `.claude/outputs/phi-particle-head/pytest_full.log`). No regression from the two shared surfaces I touched (`truncated_rollout`'s signature — additive, default path preserved; monitor #6's predicate — covered by the gym-wide diff).
- The particle head is **default-off and untrained**. Nothing here claims it *helps*; D3 only shows the gradient exists. Training it is C2W3's, and it must run with the trajectory launder (`full`/`q0_only`/`endpoints`) because a head that emits `q₀` can leak the address.
- **`friction_lo = 0.02` should probably become 0.04** for anything running the S0 store (§3.4). Left at the declared value so the reported measurement stands; a consumer must set it.
- Phase-2 friction is derived as `γ·(γ_read/γ_address)` to preserve the shipped ratio. An independent per-phase friction is a strictly larger head and was not built.
- The gym re-score instruments `exp_memory_gym.build_system` **at runtime** (the module is read-only to me). If the gym gains a second observing system, `max(readings)` selection must be revisited.

## Git footprint
Branch **`agent/experiment-engineer/phi-particle-head`** (worktree `../CHLU-phipart`, base local `main @ 233fd9e`), 5 commits, left unmerged, never pushed:

| commit | files |
|---|---|
| `376208a` | `chlu/core/psi_readout.py` (+554: `PhiSpec`, `ParticlePhi`, `SharedPhi`, `phi_ledger`, `joint_dial`) |
| `42332d7` | `chlu/core/implicit_grad.py` (+35: traced `gamma`, `mass_override`) |
| `81420bd` | `chlu/core/monitors.py` (+199: #6 dead-band, `ConfigAccessProxy`, `GAUGE_SCOPE`) |
| `804e7f4` | `tests/test_phi_particle.py` (new), `tests/test_monitors.py` (+9 tests) |
| `a7d55d9` | `chlu/experiments/exp_phi_particle.py` (new) |

No read-only file was edited: `clu_system.py`, `config.py`, `memory_gym.py`, `exp_memory_gym.py`, `experiment_cmd.py`, `dividend.py`, `exp_trajectory_read.py` are untouched (`git diff --stat main..HEAD` covers exactly the five files above). No CLI hook added. No conflicts.

---

## Proposed handover updates (for the Hub)
1. **§7 / registry — `memory-gym-v0` R2 is off by two**: "29 of 58" → **"31 of 58 are ε-artefacts; 27 survive the dead-band"** (exact, both slopes measured, all 28 cells). Monitor #6's row: "TRIPS ×58" → **"×58 pre-repair → ×27 after the C2W2 dead-band; 0 new trips; all other monitors bit-identical"**.
2. **Doctrine I-7 wording**: "the gauge is Newtonian-only" → **"the gauge is `newtonian_learned`-only"**; under `newtonian_identity` `(M,V,p₀)→(λM,λV,λp₀)` is not a gauge orbit at all (residual **0.2505** vs **2.52e-7** learned, **0.0274** relativistic at c=1). `monitors.GAUGE_SCOPE` carries the corrected text.
3. **Charter §A2.2 status**: promote from *asserted* to **measured and SUPPORTED** — trajectory/point gradient ratio **1.7e5–2.9e5** (mass), **2.6e5–4.9e5** (friction), 3 seeds, `exp_phi_particle --part grad`.
4. **Monitor #10 tier (a) is no longer open**: `knob_tier_a_implemented` can be `true` where the proxy is used (`ConfigAccessProxy` + `assert_knobs_live`). It is **not** wired into the frozen harness — someone owning `clu_system.py`/`memory_gym.py` must wrap the config there for it to fire in a run.
5. **New known-issue candidate**: the φ friction band's low end (γ ≤ 0.03) trips monitor #1 on the S0 store; `PhiSpec.friction_lo` defaults to 0.02 and should be raised to 0.04 by any consumer of that store.
6. **New config surface** (not in `chlu/config.py`, by the file-ownership rule): `PhiSpec` in `chlu/core/psi_readout.py`, override via a project YAML through `PhiSpec.from_mapping`. Worth a line in §3 next to `PsiSpec`.
7. **Never-quote addition**: do not quote monitor #6's "58 trips" without "pre-repair" — post-repair it is 27.
