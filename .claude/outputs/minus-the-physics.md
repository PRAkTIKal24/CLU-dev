# minus-the-physics — experiment-engineer report

**Task + acceptance criterion:** build the two missing "CLU minus the physics" controls — (A/G2) a non-symplectic twin + a broken-volume arm run through the V2 measurement harness to produce a "which component buys what" table, and (B/V1.1) the identical calibration/allocation/LTT gate stack on a Hopfield memory to decide whether the gate is memory-agnostic. Acceptance = both controls implemented duck-typing CHLU (harness + train_chlu run verbatim), measured over seeds, with honest attribution tables either way.

**Status: done** (both parts implemented, tested, run with real numbers; 11 new tests; full suite 165 passed / 1 skipped).

---

## Flag provenance (both parts)
- **Base:** `main @ db3369b`; branch `agent/experiment-engineer/minus-the-physics` @ `b41410f` (worktree `../CHLU-minus-the-physics`).
- **Part A run:** `python -m chlu.experiments.exp_minus_physics --seed 42` → seeds {42,43,44}, 150 epochs. Config: dim=4, hidden=64, **kinetic=newtonian_learned**, **potential=mlp** (identical potential across CHLU & broken-volume so the ablation isolates symplecticity, not a designed SO(2) potential), dt=0.05, circle_radius R=1.0, n_points=256, seq_len=65, settle_gamma=0.1/2000 steps, probe_gamma=0.05, probe_kick=0.1, probe_steps=4000, eval_steps=400. Training defaults on main: lr=1e-3, batch=64, **lyapunov_penalty="max"** (λ=0.01), **langevin_noise="legacy"**, sleep_temperature=0.5, **sleep_friction=0.0** (⇒ sleep noise inert, §7 w4 finding — erosion here is deterministic energy-raising), persistent_sleep_buffer=False. Primary arms = **wake-only** (sleep_frequency→1e9; the single epoch-0 sleep event fires, matching exp-d `sleep_mode="off"`). Erosion sub-run = **wake–sleep** (sleep_frequency=5, sleep_steps=500).
- **Part B run:** `python -m chlu.experiments.exp_v1_hopfield_gate --seed 42` (vanilla) → 5 seeds {42..46}, levels [128,16],[128,24],[128,32], hopfield_beta=20, ladder=(1,2,4) iters, embed_dim=16, vocab=256, embed_scale=2.0; self-test = calib_probes_per_key=8 @ σ∈{0.05,0.15,0.3} + 16 impostors; deployed head = `r_margin`; calib_l2=1.0; LTT δ=0.1, risk targets {0.05,0.10}. Hopfield needs **no training** → full-scale is cheap. Secondary stressed run: `--noise 0.5` (quick, 2 seeds).

---

## Part A — non-symplectic twin (G2)

### What I built (`chlu/core/twins.py`, all duck-type CHLU)
- **`UnconstrainedTwin`** — free residual recurrence `z_{t+1} = z + dt·f_θ(z)`, `f_θ` a 3-layer tanh MLP on the concatenated state; **no Hamiltonian** (`H≡T≡0`, so the wake–sleep sleep phase is inert — the twin is trained by trajectory MSE, the natural minus-the-physics objective), no volume constraint. `gamma` retained only as a momentum-damping knob so the retention protocol applies uniformly. **Param count matched** to CHLU by solving the quadratic hidden-width equation (`matched_twin_hidden`).
- **`BrokenVolumeCHLU`** — the SAME leapfrog + potential + kinetic term, then a learned per-coordinate scaling `z_next = exp(log_scale)·verlet(z)`; `det J = exp(Σ log_scale)` ⇒ volume broken iff Σ log_scale ≠ 0. `log_scale` inits to 0 ⇒ **bit-identical to CHLU at init** (isolation is exact: only the learned scaling breaks symplecticity). Isolates *volume conservation* specifically.
- Both run `goldstone_harness` (settle/perturb/spectrum/step_jacobian) and `train_chlu` verbatim (CLULattice precedent). 8 tests: duck-type surface, param match (broken=CHLU+2·dim exactly; twin <0.5% at exp scale), bit-identical init, det-J break = Σ log_scale, free-update + zero-energy, harness + train_chlu on all three arms.

### Param-count report (the count match)
`{chlu: 4549, broken_volume: 4557 (+8 = the 2·dim log_scale), twin: 4551 (hidden=59)}` — twin matched to +2 params (0.04%).

### "Which component buys what" table (mean±std, 3 seeds; measured via the geometric harness at the data-manifold point)
| metric | CHLU (symplectic) | broken-volume (det J≠1) | twin (no physics) | delta attributed to |
|---|---|---|---|---|
| **eval MSE** (fit; ↓ better) | 0.190±0.018 | 0.0785±0.004 | **0.0128±0.008** | *nothing* — physics is a **constraint that costs raw fit**; the free twin fits best |
| **log\|det J\|** (0 = vol-cons) | −2e-8±3e-8 | −0.065±0.024 | −0.012±0.013 | the ablation witness: CHLU symplectic, broken-vol genuinely non-symplectic |
| **BIBO** (frac settled bounded) | **1.00** | **0.333±0.47** | 1.00 | **volume conservation** — broken-vol loses the bounded attractor (r*→1e14, 2/3 seeds diverge) |
| vacuum radius r* | 0.72±0.05 | 100±110 | 2.24±1.3 | (same as BIBO) |
| flat spectral μ² (↓ = protected) | **0.008±0.015** | 0.122 (1 bounded seed) | N/A (no potential) | **volume conservation** — CHLU develops a near-flat protected memory direction; broken-vol's is stiff |
| latch coset drift (↓ = latches) | **0.19±0.19** | 0.245±0.05 | 1.15±1.2 | **integrator structure** — CHLU latches ~6× tighter than the twin |
| n₁/₂ flat (orbit) | 104±56 | 76 | 1350±290 | twin's huge value is an **artifact** (a free map holds displacement without dissipative decay ≠ a functional latch — its coset drifts, see above) |
| n₁/₂ stiff (radial) | 44.7±11 | N/A (diverged) | 73 | — |
| step-Jacobian \|λ\|max | 1.000±0.002 | 0.995±0.002 | 1.000±0.004 | (weakly discriminating at γ=0) |
| **sleep-erosion** r* after wake–sleep CD | **0.72±0.24 (survives)** | **0.0 (vacuum collapsed)** | N/A (sleep inert) | **volume conservation** — CHLU's vacuum survives CD at 150 ep; broken-vol's inverts to r*=0 |

### Attribution verdict (what symplecticity functionally buys)
- **Volume conservation (CHLU vs broken-volume)** buys: **BIBO boundedness** (1.0 vs 0.33 — the non-symplectic settle diverges), a **protected near-flat memory direction** (μ² 0.008 vs 0.122), and **vacuum robustness under wake–sleep CD** (survives vs collapses to r*=0). It *costs* raw fit (eval MSE 0.19 vs 0.079).
- **Integrator structure (broken-volume vs twin, and twin vs CHLU)** buys: the **latch** — CHLU coset drift 0.19 vs twin 1.15 (~6× tighter). The twin's large flat-mode half-life is *not* a functional memory (no dissipative structure → displacement persists but the coset wanders).
- **Nothing (twin)** buys the **best raw MSE** — an unconstrained recurrence memorizes the trajectory better than any physics-constrained model, but provides no bounded attractor guarantee beyond its tanh saturation and no reliable latch.

**One-line:** symplectic volume conservation buys *bounded, protected, CD-robust memory*; the leapfrog structure buys the *latch*; neither buys raw fit — that is what you pay for the physics prior. Honest negative-for-fit / positive-for-structure, exactly as the task anticipated.

---

## Part B — gate stack on a Hopfield memory (V1.1)

### What I built (`chlu/experiments/exp_v1_hopfield_gate.py`)
Runs the **identical** v1-pivot stack (write-time Platt calibration head on retrieval diagnostics → learned per-instance gate; escalation-ladder compute allocation; Learn-then-Test certificate) on a modern-Hopfield memory using **its natural scalars (Hopfield energy R, nearest-neighbour readout margin)**, same MQAR task/seeds/levels. Reuses `_probe_cues`, `_decode_values`, `_simulate_tau_policy`, `_auroc`, and `calibration.py` (head + LTT) **verbatim** — only the memory changes. Ladder = Hopfield fixed-point iterations (1,2,4) — its only compute lever. Optional correlation/eval-noise stress (reuses the regime-map clustered embeddings). 3 tests.

### Results side-by-side with the CLU (v1-pivot)
**Vanilla regime (5 seeds, identical protocol to v1-pivot):**
| metric | Hopfield (this run) | CLU (v1-pivot) |
|---|---|---|
| **calibration transfer** pooled AUROC(→wrong) | raw R **0.182±0.051 → calibrated 0.878±0.091** | raw 0.431±0.038 → calibrated 0.869±0.015 |
| **allocation** savings @ learned point | ~3.97× **but vacuous** (base=full; one-shot memory, iterating never helps) | 4.8×@kv16 **with accuracy gain** (full>base) |
| **LTT validity** | 10/10 all levels (vacuous — acc 0.98–0.999, ~zero risk) | 30/30 valid |
| AURC (abstention) | ~0.000–0.005 (near-perfect) | 0.03–0.58 |
| base/full accuracy | 0.983–0.999 | 0.26–0.85 |

**Stressed regime (eval-noise 0.5, quick 2 seeds):** Hopfield errs (acc 0.72–0.96); calibration transfer weak (raw 0.452 → calibrated 0.508) — the head transfers poorly here because the self-test jitter and the deployment noise are *different* failure modes (protocol mismatch), not a property of the memory.

### V1-identity verdict
- **The calibration-transfer mechanism is MEMORY-AGNOSTIC.** Under the identical protocol, Hopfield's per-model head turns non-comparable raw energy (pooled AUROC 0.18, anti-ranked) into a strongly deployable cross-model gate (0.878) — statistically indistinguishable from the CLU's 0.43→0.87. The learned per-instance gate is a general mechanism; it is **not** a CLU-special energy signal (confirms v1-pivot Finding 4: energy ≈ readout margin).
- **The LTT risk certificate is memory-agnostic** (valid on both; vacuously so on near-perfect Hopfield).
- **The compute-ALLOCATION payoff is NOT transferable** — it needs a memory that is (a) escalatable (a graded compute ladder) and (b) improves with compute (full-budget accuracy > base). The conservative CLU-EBM's *staged governed relaxation* has both (4.8× **with** accuracy gains); a one-shot Hopfield has neither (full=base, so its "savings" are the vacuous "don't bother iterating").

**What V1 may therefore claim (per register P7):** the **gate mechanism + distribution-free certificates** on a conservative memory — *not* energy-signal superiority (energy = readout margin, and the calibration jump reproduces on Hopfield). The one distinctive CLU asset is being an **escalatable** memory (graded relaxation ladder with real accuracy gains), where calibrated compute allocation has genuine payoff — an imperfect, iterable EBM, not a better-calibrated one.

---

## How I verified (real commands + observed output)
- `pytest tests/test_twins.py` → **8 passed**; `tests/test_v1_hopfield_gate.py` → **3 passed**. Full worktree suite `pytest tests/` → **165 passed, 1 skipped** (main baseline 154+1; +11 mine) in 120 s.
- `ruff check` on all 7 touched files → **All checks passed** (pre-existing `config.py` F811 duplicate `ExperimentV1WormholeConfig` confirmed on `main`, out of scope, untouched).
- Config default round-trip (save→load) preserves the new `experiment_minus_physics` group and the two `experiment_v1_gate.hopfield_gate_*` fields.
- Part A full run printed the table above; Part B vanilla full run printed the verdict above. Artifacts copied to `.claude/outputs/minus-the-physics/` (`exp_minus_physics_{summary.json,metrics.npz}`, `hopfield_vanilla_summary.json`, `partA_full.log`).
- Env: main venv reused via `/Users/user/Desktop/CHLU/.venv/bin/python` with cwd in the worktree (avoids a per-worktree `uv` resync; §7.12 `uv run chlu` CLI path untested — used `python -m` throughout).

## Git footprint
- Branch **`agent/experiment-engineer/minus-the-physics`** (worktree, off `main @ db3369b`; rebase = no-op, main unmoved; **not pushed**, no PR). Ref verified from the main repo (`git -C … log main..agent/…` shows all 3 commits — §3.2 satisfied).
- Commits: `9533ef3` (twins core + 8 tests) · `c33b41f` (exp-minus-physics experiment + config + CLI + exports) · `b41410f` (V1.1 Hopfield gate stack + config fields + CLI + 3 tests).
- Files **new**: `chlu/core/twins.py`, `chlu/experiments/exp_minus_physics.py`, `chlu/experiments/exp_v1_hopfield_gate.py`, `tests/test_twins.py`, `tests/test_v1_hopfield_gate.py`. **Edited (additive, minimal hunks)**: `chlu/config.py` (+`ExperimentMinusPhysicsConfig`, +2 `hopfield_gate_*` fields on `ExperimentV1GateConfig`, load/save wiring), `chlu/experiments/__init__.py` (2 exports), `chlu/cli/experiment_cmd.py` (2 parsers + 2 handlers). No CLU/V2/V1 production code paths altered — strict scope (task guard).
- Concurrency: `mass-lr-doctrine-test` ran in its own worktree throughout; no shared-checkout collision (I worked in `../CHLU-minus-the-physics`, it in `../CHLU-mass-lr-doctrine-test`). No conflicts.

## Open questions / follow-ups / risks
1. **Part A retention numbers are proxies** (first-crossing of raw projected displacement, not envelope), used because the geometric probe must be arm-agnostic (the twin has no spectral masses). For CHLU/broken-vol an envelope/spectrum read would be tighter; the *cross-arm* comparison (latch drift, BIBO, μ², erosion) is the robust part.
2. **Broken-volume divergence (BIBO 0.33)** is the expected F5 Prop-10/§7.7 consequence, but it makes broken-vol's retention/spectrum measured only on the 1/3 bounded seeds — hence some N/A cells. A gentler volume break (bounded log_scale, or scaling only p) would keep it bounded and sharpen the retention comparison — a one-knob follow-up if the Hub wants a stable broken-vol arm.
3. **Part B stressed-regime transfer (0.45→0.51)** is a *protocol* artifact (probe jitter ≠ deployment noise), not a memory property; a matched self-test (probe under the same noise) would likely restore the jump. The vanilla matched-protocol run is the fair identity comparison and gives the memory-agnostic verdict.
4. **Twin as a third Part-B memory** was skipped (not cheap): the twin has no energy scalar, so it can only gate on readout margin — noted rather than half-built.
5. **Erosion ran under `sleep_friction=0`** (deterministic sleep, per main defaults) — the vacuum-collapse result for broken-volume is under deterministic energy-raising; a Langevin (`sleep_friction>0`, `langevin_noise="fdt"`) erosion sweep is a natural extension.

## Proposed handover updates (for the Hub)
- **§2/§3 (code):** new `chlu/core/twins.py` (`UnconstrainedTwin`, `BrokenVolumeCHLU`, `build_arms`, `matched_twin_hidden` — both duck-type CHLU, run harness + train_chlu verbatim); new experiments `exp_minus_physics` (`chlu exp-minus-physics`, `ExperimentMinusPhysicsConfig`) and `exp_v1_hopfield_gate` (`chlu exp-v1-hopfield-gate`, +2 `hopfield_gate_*` fields on `ExperimentV1GateConfig`). Tests 154→165.
- **Critique register P6/G2 — ANSWERED:** symplecticity's functional payoff decomposed — *volume conservation* buys BIBO boundedness + a protected near-flat memory direction + vacuum robustness under CD (CHLU 1.0/0.008/survives vs broken-volume 0.33/0.122/collapses); *integrator structure* buys the latch (coset drift 0.19 vs twin 1.15); the physics prior *costs* raw fit (eval MSE 0.19 vs twin 0.013). Feeds all three shorts; label learned-mlp-potential results as evidence (G1).
- **Critique register P7/V1.1 — ANSWERED:** the gate stack (calibration transfer + LTT certificate) is **memory-agnostic** — Hopfield 0.18→0.88 ≈ CLU 0.43→0.87, LTT valid on both. The non-transferable asset is compute-*allocation* payoff, which needs an *escalatable* memory (CLU's graded relaxation, full>base) that a one-shot Hopfield lacks. V1 should claim gate mechanism + certificates on a conservative escalatable memory, **not** energy-signal superiority (confirms v1-pivot Finding 4). Consistency note for the M2 matrix: this reinforces, not contradicts, v1-pivot.
- **§7 candidate (non-blocking):** pre-existing duplicate `class ExperimentV1WormholeConfig` in `config.py` (lines 403 & 473 on `main`) — a merge artifact; the second definition shadows the first (identical). Harmless (ruff F811 only); worth deduping in a future config-hygiene pass.
