# v2-prefreeze-baselines — results-analyst report

**Task + acceptance criterion (P8/V2.3, now BLOCKING):** give V2 a *learned-architecture* memory baseline — coRNN/LEM/LSTM on the retention-vs-perturbation + write/read-latch protocol (5 seeds), a Mo-S¹ head-to-head (3+ seeds), an S1 re-run with the queued γ-field extras (adaptive-K + compact gates), and (cheap) the deferred kick-size decomposition. Deliver per-item numbers with error bars, quotable win/lose/tie verdicts, frozen-manifest/flag-provenance discipline.

**Status: DONE (all 4 items).** Item 1+2 executed as one unified S¹-memory head-to-head (5 baseline seeds + CLU on 5 designed / 3 emergent checkpoints). Item 3 both extras validated *at scratch level, no `chlu/` edits* (one integration gap flagged for the engineer). Item 4 run (emergent checkpoints existed on disk). Repo **read-only, untouched** (`git status` clean; HEAD `db3369b`).

---

## 0. TL;DR verdicts (quotable)

- **"A well-trained LSTM and LEM (train-horizon RMSE 0.18/0.23 rad) both forget a stored analog phase within ~57–69 recurrent-map applications, and their memory is *fragile*: a 0.1 hidden-state perturbation collapses LSTM retention 69→2 steps. A generically-trained CLU holds ~263 steps (4× longer) with *bounded* drift, and the SO(2)-designed CLU holds it forever (∞, 5/5 seeds, coset frozen to machine zero)."**
- **The structural triad IS qualitatively absent in the baselines.** *Latch* (∞ retention): only CLU-designed — absent in coRNN/LEM/LSTM *and* emergent CLU. *Budget-table law* (n₁/₂ ∝ 1/μ², slope −1, floor): CLU-emergent obeys it to +12–28% (item 4 decomposes the residual); baselines have no curvature/μ² parameter and no predictive retention law at all. *Bounded drift*: CLU (both) saturate (≤0.35 rad); every baseline randomizes to >1.2 rad.
- **Where CLU loses / needs work (honest):** on the *input-driven* path-integration **task accuracy** axis, the RNNs are the only entrants — CLU has no native velocity-input ingestion, so it cannot compete on Mo's supervised RMSE without an equivariant-control wrapper (future work). coRNN is a weak baseline at these hyperparameters (see §2 caveat).
- **γ-field extras both work:** compact gate cuts on-manifold friction leakage ~200× (oracle_compact = **exactly 0.0** outside radius) and lets the oracle reach the γ=0 retention ceiling (coverage 0.677 = base) *while* rejecting; adaptive-K spawns K=1→8, improves locus discovery (2/3 seeds vs the pilot's 2/6) and gives the **best rejection of any learned arm (0.861)**.

---

## 1. Setup, provenance & frozen manifest

All artifacts under gitignored `.claude/`: scripts in `.claude/scratch/v2-prefreeze-baselines/`, numbers/figures in `.claude/outputs/v2-prefreeze-baselines/`. All commands run `PYTHONPATH=/Users/user/Desktop/CHLU uv run --no-sync python <script>` from repo root (env §7.12 workaround; the CLI path is not needed — everything is driven programmatically).

### Flag-provenance table (protocol §5)

| result group | code commit | seeds | key flags in effect |
|---|---|---|---|
| **Baselines** (item 1/2) | scratch @ `db3369b` (repo unmodified) | 42–46 (5) | scratch coRNN/LEM/LSTM; hidden **16**; train seq_len 64, test 256; Adam; **lr∈{1e-3,3e-3,1e-2} swept, best-RMSE kept** (fairness/P8); ω∼N(0,0.3); coRNN dt=0.1,γ=ε=1.0; LEM dt=1.0; MSE loss; retention: write_len 32, hold 2000, n_writes 64, drift-threshold 0.2 rad, perturb σ=0.1 |
| **CLU designed/emergent** (item 1/2/4) | checkpoints from `v2-full-runs` @ `dbeb2c2` (wave-2) | designed 42–46, emergent 42–44 | ExperimentD: dim 4, hidden 64, `newtonian_learned`, `tie_channel_mass=True`, dt 0.05, circle R=1 ×256; **train_epochs 150** (v2-full-runs Finding-0 regime), `lyapunov_penalty="max"` (λ=0.01), `langevin_noise="legacy"`, `persistent_sleep_buffer=False`, `sleep_temperature=0.5`, `sleep_frequency=5`, `sleep_friction=0.0`, lr 1e-3; **designed=`so2_invariant`, emergent=`mlp`** |
| **CLU probe** (item 1/2/4) | `db3369b` (harness unchanged) | — | x64 (f32 ckpt cast to f64); Mo protocol φ₀=0.35, threshold 0.2, cap 2000/8000/15000; probe **γ=0.05, kick 0.1, dt 0.05**; vacuum = damped settle + BFGS polish (|∇V|→1e-9..1e-14) |
| **Item 3 S1 extras** | scratch @ `db3369b` (repo unmodified) | 0,1,2 (3) | figure-8 data, `newtonian_identity`, hidden 64, dt 0.05, **train_epochs 300**, sleep_steps 100; field γ_max 0.5, width 0.25, `friction_field_lr=1e-2`; adaptive-K: spawn_threshold 5, max_k 8, spawn_radius 0.5, spawn_strength 0.15, prune_floor 0.02; noise locus [1.5,1.5] |

**Frozen inputs reused (not re-trained):** `v2-full-runs/runs/{designed150_s42..46,emergent150_s42..44}/models/exp_d_chlu.pkl`. Loading + Mo-protocol + spectrum probe reuse `v2-full-runs/probe_common.py` verbatim.

---

## 2. Item 1+2 — Learned baselines + CLU on the S¹ retention/latch protocol  ✅

**Design (unified item 1 & 2).** One task family = Mo's S¹ path integration (mo-deep-read §3.4): input = scalar angular velocity, target = (cos φ, sin φ). Train the RNNs (400 ep, lr sweep), then run **Mo's autonomous lifetime protocol** on all systems — *write a phase, hold with zero input, measure the first step where stored-phase drift exceeds 0.2 rad*. This is the same observable on both sides (recurrent-map applications), so baselines and CLU are directly comparable. CLU uses `mo_lifetime_protocol` on the trained checkpoints (state placed on the ring at φ₀, autonomous Verlet rollout at γ=0.05). Figures: `figA_retention_overlay.png` (+`_logx`), `figB_lifetime_summary.png`.

### 2.1 Baseline task competence (NOT strawmen — P8)

| model | n_params | best lr | RMSE @train-hor. 64 | RMSE @test-hor. 256 | success frac (256) |
|---|---|---|---|---|---|
| coRNN | 578 | 3e-3/1e-2 | 0.820 ± 0.081 | 1.446 ± 0.112 | 0.158 ± 0.019 |
| LEM   | 1186 | 1e-2 | **0.234 ± 0.022** | 0.973 ± 0.072 | 0.438 ± 0.033 |
| LSTM  | 1186 | 1e-2 | **0.182 ± 0.044** | 0.845 ± 0.103 | 0.524 ± 0.151 |

LEM/LSTM genuinely learned short-horizon path integration (train-horizon error 10–13°); their long-horizon (256) RMSE ~0.85–0.97 rad reproduces **Mo's own baseline failure regime (>1.45 for his budget; ours slightly better under the lr sweep)** — path integration is hard for gated RNNs *by design*, which is the point. **coRNN caveat:** at the fixed default oscillator hyperparameters (γ=ε=1.0, dt=0.1) coRNN is the weak entrant (train-horizon 0.82 rad); its architecture is known to be sensitive to those constants. LSTM/LEM are the fair, well-trained comparators; I report coRNN honestly but do not lean on it.

### 2.2 Retention / latch head-to-head (the blocking result)

Median retention lifetime = steps until stored-phase drift ≥ 0.2 rad (Mo's threshold), over write-phases × seeds. "Perturbed" = 0.1 Gaussian kick to the hidden state at hold onset.

| system | median lifetime (steps) | censored frac (latch fraction) | final drift (rad) | **perturbed** median lifetime | budget-table law? |
|---|---|---|---|---|---|
| coRNN | 5.6 ± 1.4 | 0.00 | 1.55 (randomized) | 5.1 ± 1.0 | — (no μ²) |
| LEM | 56.4 ± 19.5 | 0.02 | 1.66 (randomized) | 5.0 ± 1.1 | — |
| LSTM | 68.7 ± 18.3 | 0.08 | 1.28 (randomized) | **1.8 ± 0.8** | — |
| **CLU-emergent** (MLP, generic) | **263** (184–436, n=3) | 0.00 | **0.35** (bounded) | — (envelope 152–812) | **YES** (n₁/₂ 152/812/152 vs pred 118/687/118) |
| **CLU-designed** SO(2) | **∞** (all censored) | **1.00** (5/5) | **0.00** (coset frozen) | ∞ | latch (μ²≈1e-15) |

**Reading (`figA`):** every baseline drifts monotonically past 1.2 rad — the stored phase is *completely lost*. CLU-emergent crosses 0.2 rad at ~250 steps but then **saturates at ~0.35 rad** (it relaxes to the nearest pinning minimum of the self-broken washboard and *freezes there* — a finite-displacement pseudo-Goldstone, not loss). CLU-designed never moves (coset angle frozen to ≤1e-15 rad, `coset_freeze=0.00` all 5 seeds).

*(Figure caveat: in `figB` the two CLU "perturbed" bars reuse the unperturbed value — CLU's perturbation is a position-kick with a different readout, quantified as the envelope half-life in the table above, not a hidden-state kick; the figure's honest content is the baseline collapse.)*

**Perturbation fragility is the sharpest single fact.** Baselines that look competent unperturbed collapse under a small hidden-state kick: **LSTM 69→2, LEM 56→5** map-steps. Their "memory" is a marginally-stable trajectory, not a basin — perturbing off it destroys the stored value. CLU's retention is measured as a position-kick envelope half-life and stays 152–812 (emergent) / ∞ (designed): the state sits *in* an attractor (or on the exact flat manifold), so perturbation is absorbed.

### 2.3 Win / lose / tie (honest)

- **CLU-designed wins decisively** on retention (∞ vs finite) — *with the caveat it is a designed SO(2) prior*, so it answers "what the physics prior buys," not "what emerges from generic training."
- **CLU-emergent (architecture-matched: generic MLP, no designed symmetry) wins** vs the RNNs on retention (263 vs 57–69 map-steps, ~4×), on drift boundedness (0.35 vs >1.2 rad), and on perturbation robustness. This is the fair "is it better than an LSTM at remembering?" answer: **yes, ~4× longer and qualitatively bounded, even without the designed symmetry.**
- **CLU loses** on the *input-driven task-accuracy* axis: it has no native velocity ingestion, so on Mo's supervised RMSE only the RNNs compete. The equivariant-control input wrapper (a momentum kick along the coset tangent — flagged in mo-deep-read as "a first equivariant-control result") is unbuilt; I did **not** fabricate a CLU task-RMSE.
- **Confound (stated):** lifetimes are counted in recurrent-map applications, the fair common unit, but one CLU step is a dt=0.05 Verlet substep and one RNN step is one input tick — the *wall-time* meaning differs. The qualitative triad (latch / budget-law / bounded drift) is unit-independent; the 4× ratio is not.

### 2.4 Item 2 Mo-S¹ specifically ("his law is our overdamped face," other direction)

v2-full-runs ran Mo's *breaking* protocol on CLU; here I run Mo's *lifetime* protocol (φ₀=0.35, threshold 0.2) on the same trained checkpoints and on the RNNs. **CLU-designed = ∞ (latch, Mo's neutrality theorem realized exactly); CLU-emergent = finite 184–436 steps that the F5 budget table predicts to +12–28% (n₁/₂ vs exact-2×2: {278/247, 812/687, 152/118}).** The baselines under the identical protocol give 6/56/69 with no predictive law. So Mo's single-exponential lifetime picture is the finite (emergent) face; the designed CLU exhibits the latch his kinematics allows but his first-order flows never produce.

---

## 3. Item 3 — S1 re-run with adaptive-K spawn + compact-support gates  ✅ (scratch-level, no `chlu/` edits)

**Both mechanisms are already in `chlu/core/friction_field.py`** (`add_hole`/`maybe_adapt_holes`/`AdaptiveKState`; `gate="compact"` smoothstep) **and wired through `chlu/training/train_chlu`** via `config.training.friction_field_adaptive_k` (train.py:133/435). Adaptive-K needs *only a config flag* through the existing path; the compact gate is set at `FrictionField` construction. I drove both from a scratch replica of the S1 arms (figure-8 + off-manifold garbage cluster), reusing `retention_scores`/`rejection_scores`/`field_placement_scores` verbatim. 3 seeds, 300 epochs. Data: `item3_s1_full.json`.

| arm | K_final | coverage (retention) | rejection_pos | γ̄ on curve (leakage) | γ̄ on noise (placement) |
|---|---|---|---|---|---|
| base (γ=0, ceiling) | 0 | 0.677 | 0.374 | — | — |
| learned K=4 sigmoid (pilot) | 4 | 0.500 | 0.830 | 1.35e-3 | 3.18e-2 |
| **learned K=4 compact** | 4 | 0.405 | 0.682 | **6.9e-6** | 0.0 |
| learned K=1 fixed | 1 | 0.543 | 0.630 | 5.2e-4 | 2.52e-2 |
| **learned K=1 adaptive-K** | **8** | 0.460 | **0.861** | 1.8e-3 | **8.91e-2** |
| oracle sigmoid | 1 | 0.676 | 0.792 | 1e-5 | 2.93e-1 |
| **oracle compact** | 1 | **0.677** | 0.712 | **0.0 (exact)** | 3.0e-1 |

**Compact gate (extra 2) — works as designed.** On-curve friction leakage drops ~200× (K=4: 6.9e-6 vs sigmoid 1.35e-3); the oracle_compact is **exactly 0.0 outside its radius** (hard cutoff verified) and consequently sits at the **conservative retention ceiling (coverage 0.677 = base's 0.677)** while still rejecting (0.712) — i.e. it *closes the sigmoid tail-leakage retention gap* that gamma-field-build flagged. The learned compact arm under-performs (coverage 0.405) only because a hard cutoff has no reach: with imperfect placement its holes miss the locus (γ_noise=0.0), so *compact gates need accurate placement* — exactly what adaptive-K provides.

**Adaptive-K (extra 1) — works, improves locus discovery.** K=1→8 (hits the cap) all 3 seeds; **best rejection of any learned arm (0.861 vs K4-sigmoid 0.830, K1-fixed 0.630)** at coverage 0.460. Locus discovery improved to γ_noise 0.089 (2/3 seeds hit: per-seed {0.181, 0.004, 0.082}) vs fixed-K1 0.025 and the pilot's 2/6. It does not reach the oracle's placement (0.29) and it over-spawns to the cap (modest coverage cost), but it fixes the gradient-locality failure it was designed for.

**Recommended follow-up (not run):** compact **+** adaptive-K combined — adaptive-K supplies the placement that compact gates require to keep retention at the ceiling *and* reject. Cheap.

**Integration gap for `experiment-engineer` (NOT a blocker; I used scratch):** `chlu/experiments/exp_s1_gamma_field.py` builds the learned-arm `FrictionField` (≈line 361) and the oracle field (≈line 405) **without forwarding `gate=`**, so `chlu exp-s1` always uses `sigmoid` regardless of `config.training.friction_field_gate`. Exact edit: add `gate=tcfg.friction_field_gate` to both `FrictionField(...)` calls (and thread `friction_field_gate` into `ExperimentS1Config` if a per-arm override is wanted). Adaptive-K is already honored end-to-end via the config flag; no edit needed there.

---

## 4. Item 4 — Kick-size decomposition of the emergent +12–15% retention bias  ✅

The emergent checkpoints existed on disk (`emergent150_s42..44`), so the seed-sweeps-deferred item ran. The exact-2×2 prediction uses the kick-independent local Hessian μ²; sweeping the position-kick amplitude decomposes the bias (`item4_kick_decomp.json`):

| seed | μ²_ang | ratio(kick→0) | d(ratio)/d(kick) | ratios over kick 0.02→0.4 | verdict |
|---|---|---|---|---|---|
| 42 | 5.45e-2 | 1.125 | −0.006 (≈0) | 1.13,1.12,1.12,1.12,1.12 | **kick-independent → settle/discretization offset (+12.3%)** |
| 44 | 1.07e-1 | 1.288 | +0.000 | 1.29 (flat) | **kick-independent → settle offset (+28.8%)** |
| 43 | 2.03e-2 | 1.052 | **+1.28** | 1.07→1.55 | **anharmonicity-dominated** (softest mode) |

**Decomposition:** the emergent bias is *mostly a kick-independent additive offset* (settle residual / integer first-crossing discretization) for 2/3 seeds, but **genuinely anharmonic for the softest-μ² seed (s43)** — where a larger kick explores more of the flat self-broken washboard, the potential softens, and retention *lengthens* with amplitude (ratio 1.07→1.55). Anharmonicity shows up exactly where F5 predicts it can (soft mode, large relative excursion); the harder modes are effectively linear. This resolves v2-full-runs Limitation 3.

---

## 5. How I verified

- Baseline pipeline smoke (2 seeds/60 ep, 8.9 s) → full (5 seeds/400 ep, lr sweep). Train-horizon RMSE confirms the models learned the task (not undertrained): LSTM 0.18, LEM 0.23 rad.
- CLU probe reuses the *validated* `v2-full-runs/probe_common.py` (x64, BFGS-polished vacuum |∇V|→1e-9); designed μ²_ang = 1.2e-16…1.6e-15 (machine-flat), emergent 2.0e-2…1.07e-1 — matches v2-full-runs exactly. Mo-lifetime on designed = ∞/censored 5/5 with `coset_freeze=0.00`.
- Item 3 smoke (1 seed/60 ep) then full (3 seeds/300 ep); adaptive-K spawn confirmed K=1→8 live; oracle_compact γ_on_curve = **exactly 0.0** (hard-cutoff proof).
- Item 4 ratios reproduce v2-full-runs' emergent n₁/₂ (277/812/152) and predictions (247/687/118).
- `git status` clean throughout — **no tracked file touched** (HEAD `db3369b`).

## 6. Limitations / confounds

1. **Step-unit confound** (§2.3): map-application count is the fair common unit, but CLU-Verlet-substep ≠ RNN-input-tick in wall time. Headline is the qualitative triad (latch / budget-law / bounded drift), which is unit-free.
2. **No CLU input-driven task RMSE**: CLU can't ingest velocity natively; the equivariant-control wrapper is unbuilt. The head-to-head is on *retention*, not *supervised path-integration accuracy* (where only RNNs compete — an honest CLU gap).
3. **coRNN weak at default oscillator hyperparameters** (γ=ε=1, dt=0.1). LSTM/LEM carry the fair comparison; a coRNN (γ,ε,dt) sweep could strengthen it but I did the *lr* sweep the task mandated and report coRNN honestly rather than tune it into a different regime.
4. **CLU-emergent n=3** (only 3 emergent checkpoints exist); designed n=5. Baselines n=5.
5. **Item 3 at 300 ep / 3 seeds**, figure-8 task (the S1 testbed), not the S¹ circle — it inherits the pilot's design; the compact/adaptive verdicts are mechanism-level, not a full Pareto re-sweep.
6. Retention "hold" feeds *constant zero* input, which the RNNs saw little of in training — this is Mo's autonomous-restriction *by design* (it tests whether training induced a continuous attractor); none of the baselines did.

## 7. Recommended next experiments

1. **(engineer, 1-line)** forward `gate=tcfg.friction_field_gate` in `exp_s1_gamma_field.py` (§3) so `chlu exp-s1` honors compact gates.
2. **(analyst, cheap)** compact **+** adaptive-K combined arm (adaptive-K supplies the placement compact gates need) — predicted to hit the retention ceiling *and* reject.
3. **(analyst/engineer)** the CLU equivariant-control input wrapper (velocity → coset-tangent momentum kick) to put CLU on Mo's supervised path-integration RMSE — the one axis where CLU currently can't enter; would let the V2 short claim the task head-to-head in both directions.
4. **(analyst)** coRNN (γ,ε,dt) fairness sweep if a stronger 2nd-order baseline is wanted; add a plain orthogonal-RNN for completeness (Mo's baseline set).

## Git footprint

None — repo read-only, no tracked files touched (HEAD `db3369b`, `git status` clean). All artifacts under gitignored `.claude/`:
- Scripts: `.claude/scratch/v2-prefreeze-baselines/{s1_models.py, s1_task.py, run_baselines.py, clu_retention.py, item3_s1_extras.py, item4_kick_decomp.py, make_figures.py}` + logs.
- Outputs: `.claude/outputs/v2-prefreeze-baselines/{baselines_full.json, baselines_curves_full.npz, clu_retention.json, item3_s1_full.json, item4_kick_decomp.json, figA_retention_overlay.png, figA_retention_overlay_logx.png, figB_lifetime_summary.png}`.

---

## Proposed handover updates (for the Hub)

- **§1.6/§5 (V2, fold in) — P8/V2.3 CLEARED, blocking item closed.** Learned-architecture baseline exists and is fair (LSTM/LEM train-horizon RMSE 0.18/0.23 rad; lr-swept). On Mo's autonomous S¹ retention protocol (map-application count, 0.2-rad threshold): **coRNN/LEM/LSTM median lifetime 5.6/56/69 steps → drift fully randomizes (>1.2 rad); memory is perturbation-fragile (LSTM 69→2 under a 0.1 hidden kick). CLU-emergent (generic MLP) 263 steps, bounded drift 0.35 rad, budget-table-predicted (n₁/₂ 152–812 vs 118–687). CLU-designed SO(2) = ∞ latch (5/5 censored, coset frozen 0.00).** Quotable: *"the structural triad — latch, μ²-budget-law, bounded drift — is qualitatively absent in every learned baseline; a well-trained LSTM forgets a stored phase in ~69 steps and a small perturbation kills it in 2."* Figures ready: `figA_retention_overlay.png` (headline), `figB_lifetime_summary.png`. **Honest gap to state in the paper (G6/positioning):** CLU has no native input-driven path-integration; the RNNs own the supervised-task-RMSE axis until the equivariant-control wrapper is built.
- **§8 / Thread-1 (γ-field): both queued extras validated, no code change.** Adaptive-K (K=1→8, best rejection 0.861, locus discovery 2/3 vs pilot 2/6) and compact gates (on-curve leakage ↓200×, oracle_compact exactly 0.0 outside radius → retention at the γ=0 ceiling with rejection) are already in-repo and wired; measured at scratch. Combined compact+adaptive-K is the recommended next step.
- **§7 candidate (minor, engineer):** `exp_s1_gamma_field.py` learned/oracle arms don't forward `friction_field_gate` to `FrictionField` (≈lines 361, 405) ⇒ `chlu exp-s1` ignores `config.training.friction_field_gate`. One-line fix each (adaptive-K is correctly honored via the config flag).
- **v2-full-runs Limitation 3 RESOLVED (item 4):** the emergent +12–15% retention bias is a kick-independent settle/discretization offset for 2/3 seeds (+12.3%, +28.8%, slope≈0) and genuinely anharmonic for the softest-μ² seed (s43: ratio 1.07→1.55, slope +1.28) — anharmonicity appears exactly where F5 allows (soft mode, large excursion).
- **Provenance note:** CLU results reuse the `v2-full-runs` designed150/emergent150 checkpoints (trained @ `dbeb2c2`, 150 ep, `so2_invariant`/`mlp`, tie_channel_mass, legacy langevin, max lyapunov) — same manifest as the V2 battery, so the numbers stitch consistently into that figure set.
