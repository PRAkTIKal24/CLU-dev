# generative-studies — results-analyst report

Task + acceptance criterion: (A) test the FDT-imbalance conjecture (F5 Prop-9 / §7.9) directly on Exp-C checkpoints + via legacy-vs-fdt generation comparison with verdict; (B) settle §7.4 PCD-vs-CD on the dynamics path with matched seeds — with numbers.
Status: **done** (both studies complete; Study A verdict is split: *mechanism confirmed exactly, imbalance-attribution refuted*; Study B is a characterized null: *the switch is functionally inert in the dynamics path*).

## What I did
- **Study A.1:** constant-T Langevin chains (production `stochastic_rollout`, no code edits) on 3 healthy Exp-C checkpoints (mnist, mnistFF, mnistFFF; mnistF skipped per task) × {legacy, fdt, fdt scale-matched} noise (+1 dt-doubling control), 64 chains × 2000 steps, per-mode `Var(p_i)` vs `M_eff,i` from `model.effective_mass()`.
- **Study A.2:** paired dream batches from mnistFFF (paper-like config) under 4 conditions × 3 noise seeds × 64 dreams, scored with a recorded sklearn classifier; pooled + per-seed mode statistics; paired per-sample flip analysis.
- **Study B:** `training.persistent_sleep_buffer ∈ {False (CD), True (PCD)}` on the Exp-B dynamics path, 3 matched seeds × 2 arms (paired data/init/train keys), 500 epochs; + a 6-run supplement at `sleep_friction=0.2` where the distinction could have been live.
- All artifacts: scripts in `.claude/scratch/generative-studies/`, raw results in `.../{a1_results,a2_results,b_results,b2_results}/`, figures in `.claude/outputs/generative-studies/`.

## How I verified (commands + environment)
- Env: Python 3.11.13, jax 0.9.0 (CPU), equinox 0.13.4, sklearn 1.8.0; repo `main` @ `dbeb2c2` (post wave-2 merge), working tree clean, **no tracked files touched**. JAX warm import 3.6 s.
- Commands (cwd = repo root, all `uv run --no-sync python`):
  `prep_mnist.py` → `smoke_rollout.py` → `study_a1.py` → `plot_a1.py` → `train_classifier.py` (OMP_NUM_THREADS=2) → `study_a2_dreams.py` → `study_a2_score.py` → `paired_flips.py` → `study_b.py` → `study_b2.py` → `plot_b.py`.
- Smoke-level validation before full runs: legacy pooled Var(p)=0.02943 vs Prop-9 exact 0.02941; fdt 0.31009 vs M̄·T=0.3102 (mnistFFF, γ=0.3, dt=0.05, T=0.5). No NaNs anywhere in any run.
- Stationarity of the measured quantity: split-half Var(p_i) median ratios 0.999–1.001 in all 10 A.1 conditions; per-chunk pooled Var(p) flat to <0.5% over 2000 steps.

---

## Study A — the FDT-imbalance conjecture (§7.9, F5 Prop-9)

### A.1 Per-mode test on checkpoints — **mechanism confirmed exactly**

Protocol: γ=0.3, dt=0.05 (mnistFFF's dream params, uniform probe across checkpoints), T=0.5; chains init at 64 real MNIST images (`rng(0)` choice) + p~0.1·N(0,1); burn-in 500 steps, 96 000 samples/mode; keys: rollout base `PRNGKey(1000+cond_idx)` fold_in(chunk, chain). `fdt_matched` = fdt at T·κ, κ = 2dt/((2−γ)·M̄_eff) (κ = 0.0895/0.0911/0.0948 for mnist/FF/FFF). Full numbers: `a1_results/summary.json`; figures `a1_Teff_vs_mass.png`, `a1_varp_slopes.png`, `a1_nonequilibrium.png`.

| condition (all 3 ckpts agree) | slope log Var(pᵢ) vs log M_eff,i | ρ_Spearman(T_eff, 1/M_eff) | T_eff CV | T_eff p95/p5 | Var(p) mean vs exact prediction |
|---|---|---|---|---|---|
| legacy @ T=0.5 | **0.002 / −0.001 / 0.001 (±0.004)** | **0.994–0.995** | 0.075–0.077 | **1.275–1.282** | 0.02941–0.02943 vs 2dtT/(2−γ)=**0.029412** (≤0.06% off) |
| fdt @ T=0.5 | **0.999–1.001 (±0.004)** | −0.007…0.012 | 0.007–0.008 | 1.025–1.027 | median Var(pᵢ)/(M_eff,i·T) = **1.0000–1.0004** |
| fdt @ T·κ (matched) | 1.001–1.003 | ≈0 | 0.008 | 1.026 | mean Var(p)=0.02941–0.02943 = legacy's (scale match verified) |
| legacy dt=0.10 (mnistFFF) | 0.002 | 0.995 | 0.077 | 1.278 | T_eff mean **0.0954 = 2× the dt=0.05 value 0.0477** |

- **Prop-9 holds exactly on real trained models, not just the harmonic toy:** legacy noise gives every mode the *same* momentum variance regardless of inertial mass (slope 0 = mass-blind), so per-mode temperature T_eff,i = Var(pᵢ)/M_eff,i tracks 1/M_eff,i at ρ=0.994. The `fdt` flag restores a single temperature (slope 1, T_eff flat at 0.5000±0.004 CV): the momentum marginal becomes Gibbs-consistent — despite the conv potential and relativistic kinetic coupling, which do *not* wash out or restore anything measurable.
- **Temperatures are not in energy units** (Prop-9 consequence 1) verified live: doubling dt exactly doubles T_eff; and nominal T=0.5 legacy is really T_eff≈0.045–0.048 (≈11× colder). κ≈0.09 is the conversion factor at these settings.
- **BUT the violation's dynamic range is small:** because the learned inertial-mass spectrum is narrow (max/min≈1.5, mass-spectrum-peek), the per-mode temperature spread is only CV≈7.5%, hottest/coldest ≈1.28×. The "each mode its own temperature" pathology is real but weak at current checkpoints.
- **New side-finding — no positional equilibrium under either mode:** the momentum marginal is stationary, but mean H falls monotonically the whole 2000 steps (mnistFFF legacy −115→−231) and chains exit the trained cube (max|q| →3.66 at dt=0.05; →7.64 at dt=0.1). Constant-T sampling on these checkpoints has no confining stationary state in q — direct §7.7 consequence (Conv potential architecturally non-coercive; training-time clamp isn't in the unit). All A.1 statements are therefore about the (stationary) momentum marginal; that is exactly what Prop-9 predicts and what the conjecture needed.

### A.2 Generation comparison (mnistFFF, paper-like dream protocol) — **imbalance-attribution refuted**

Protocol: exp_c "Annealed Thermal" faithfully in-script (γ=0.3, dt=0.05, T 1.0→0.01 exponential over 1000 steps, init = centroid(first 10k imgs)+0.5·N(0,1), p 0.1·N(0,1)); 64 dreams × seeds {0,1,2}; **same init states and same key stream across conditions** (paired chains). Decode = exp_c's pca-None path `tanh(3q)`. Scorer: sklearn MLPClassifier((128,), max_iter=40, random_state=0) on canonical 60k/10k split in [−1,1], **test acc 0.9767** (`mnist_classifier.joblib`). Results: `a2_results/score_summary.json`; figures `a2_digit_freqs.png`, `a2_sample_grids.png`.

| condition | pooled digit hist [0..9] (n=192) | entropy (bits) | TV vs uniform | f(3,5,8,9) | mean maxprob | pairwise-L2 diversity | mean final H |
|---|---|---|---|---|---|---|---|
| legacy | [0,2,3,0,**66,68**,4,0,11,**38**] | 2.04 | 0.596 | 0.609 | 0.753 | 9.68 | −254.4 |
| fdt_matched (same heat, uniform allocation) | [0,2,3,0,**68,68**,4,0,11,**36**] | 2.03 | 0.596 | 0.599 | 0.753 | 9.67 | −254.6 |
| fdt_nominal (same nominal T ⇒ 10.5× hotter) | [0,2,2,3,41,12,4,0,**82,46**] | 2.09 | 0.580 | 0.745 | 0.801 | 13.06 | −167.6 |
| det_annealed (T=0 control) | [0,0,1,0,7,**182**,0,0,0,2] | 0.35 | 0.848 | 0.958 | 0.932 | 5.92 | −384.8 |

Head-to-head tests:
- **legacy vs fdt_matched: χ²=0.08, p=1.00** (pooled); per-seed histograms nearly identical; paired per-sample analysis: **2/192 classification flips (1.0%)**, mean paired-image L2 **0.10** (vs ≈9.7 typical inter-image distance). *Correcting the per-mode FDT violation at matched effective temperature changes essentially nothing about what gets generated.*
- legacy vs fdt_nominal: χ²=103.2, p<1e-4; **109/192 flips (57%)**, paired L2 5.49 — but this is the *global scale* (nominal T semantics change ~11×), not the anisotropy; distribution shifts 5→8 and stays strongly imbalanced (2.09 bits vs uniform 3.32).
- det_annealed reproduces the paper's deterministic mode collapse (95% '5', 0.35 bits) — '5' is this checkpoint's deepest accessible basin; stochastic conditions spread over the *adjacent* basins {4,5,9,(8)}.
- Note: this checkpoint's over-represented set is {4,5,9}+8, vs the paper's reported {3,5,8,9} — same "loopy digit" family, checkpoint-dependent membership (paper figures came from a different run/subsample; also §7.11/§7.13: those subsets are unreconstructable).

### A.3 Verdict on the conjecture (task item 3)

**Mixed, decisively resolved in both halves:**
1. **The physics of Prop-9 is exactly right on trained models** — legacy sampling has no single temperature; per-mode T_eff,i ∝ 1/M_eff,i at ρ=0.994, magnitudes to <0.1% of the closed-form prediction; the `fdt` fix restores a flat temperature to within 0.8% CV.
2. **The attribution "⇒ candidate driver of the 3/5/8/9 mode imbalance" is refuted at current checkpoints:** the scale-matched FDT correction leaves the generated-mode distribution statistically identical (χ²=0.08, 2/192 flips). The imbalance is a property of the **learned energy landscape** (basin depths/volumes + annealed relaxation from the centroid), not of the sampler's broken Gibbs invariant. Root cause of the smallness: the mass spectrum is too narrow (max/min≈1.5 ⇒ temperature anisotropy ≤±13% around mean) — connects directly to mass-spectrum-peek's "latent tendency, not a hierarchy". *Conditional:* if future training induces a genuine mass hierarchy (V3 designed banding, mass-aware objectives), the legacy sampler would grow a real per-mode temperature hierarchy — the fdt flag then stops being cosmetic. No retrain-with-fdt was needed (results unambiguous); retraining *with* fdt noise asks a different question (does the learned landscape change?) — listed under follow-ups.
3. What *does* matter practically today is **temperature calibration**: identical nominal schedules mean ~11× different heat between modes (and T_eff ∝ dt under legacy). Any cross-run temperature comparison, annealing-schedule transfer, or dt change under legacy noise silently rescales the sampler.

---

## Study B — PCD vs CD on the dynamics path (§7.4)

Protocol: Exp-B setting (CHLU dim=1, hidden 64, newtonian_learned, c=1, m₀=1; sine data `generate_sine_waves` 100×1000, dt=0.05, 80/20 split), seeds {42,43,44}, `train_chlu` 500 epochs, arms {CD=False, PCD=True} otherwise identical **and paired** (same data key, model init, train key ⇒ identical wake windows and buffer-sampling indices; only buffer *contents* can differ). Config recorded per run: `lyapunov_penalty="max"` (post-wave-2 default, live), `langevin_noise="legacy"`, `sleep_temperature=0.5`, `sleep_friction=0.0`, sleep_steps=500, sleep_frequency=5, buffer 10000, batch 64, lr 1e-3, clamp 1000/ramp 0.5. Downstream: governed_rollout (0.99·target_E, sens 0.95) MSE at σ∈{0,0.1,0.5,1.0} on 20 held-out waves, paired noise keys. Results: `b_results/summary.json` (+`b2_results/` supplement), figure `b_pcd_vs_cd.png`.

| seed | arm | wake loss (last-100 mean) | target_E | M learned | MSE σ=0 / 0.1 / 0.5 / 1.0 |
|---|---|---|---|---|---|
| 42 | CD | 33.2647 | 4.6501 | 0.7062 | 12.299 / 12.357 / 12.684 / 13.647 |
| 42 | PCD | 33.2643 | 4.6500 | 0.7062 | 12.299 / 12.357 / 12.684 / 13.647 |
| 43 | CD | 28.0528 | 3.3346 | 0.7392 | 11.886 / 11.988 / 12.456 / 13.334 |
| 43 | PCD | 28.0522 | 3.3345 | 0.7392 | 11.885 / 11.987 / 12.455 / 13.333 |
| 44 | CD | 32.5154 | 3.8073 | 0.7223 | 12.526 / 12.524 / 13.111 / 13.560 |
| 44 | PCD | 32.5149 | 3.8070 | 0.7223 | 12.526 / 12.524 / 13.110 / 13.560 |

- **Null at ~1e-5 relative, consistent across all 3 seeds and all readouts** (Δloss −0.0004…−0.0006; ΔtargetE ≤2e-4; ΔM ≤5e-6; ΔMSE ≤1e-3 at every σ). Per-epoch relative loss divergence between arms grows from ~1e-7 to only ~1e-4 by epoch 500 (fig, middle panel) — chaotic micro-divergence, no systematic effect.
- **Mechanism (why it's null, not just "small"):** at dynamics defaults `sleep_friction=0` ⇒ (a) Langevin σ=√(2γTdt)=0 — `sleep_temperature=0.5` injects **zero** noise under *both* noise modes (σ∝√γ) — and (b) γ=0 evolution is conservative, so the persistent buffer performs an **isoenergetic walk**: negatives' energies are frozen at their N(0,1)-init distribution forever in both arms (verified: E[H(random)] vs E[H(random evolved 500 steps)] differ by 0.002 ≈ 0.03% integrator drift). PCD has nothing to persist that CD doesn't already have.
- **Supplement (`sleep_friction=0.2`, 6 more runs):** arms become identical to *all printed digits* — with damping time 1/γ=5 steps ≪ k=500 sleep steps, one sleep event fully mixes the negative chain to its quasi-stationary distribution from any start (`evolve(evolve(x)) ≈ evolve(x)`), so persistence is again inert. Also: γ 0→0.2 shifted target_E by only ~0.03% (4.6501→4.6486) — the whole sleep phase is a weak perturbation to dynamics training relative to the clamped wake MSE.
- **§7.4 decision input:** persistence changes **nothing the paper cares about** in the dynamics path — the code-vs-Algorithm-1 discrepancy has no empirical consequence for Exp-A/B-class results; keeping default `False` is safe, and historical results need no asterisk. The only regime where PCD can matter is **k_steps shorter than the negative-chain mixing time** (the classic RBM regime) — the dynamics trainer never enters it (k=500 with either γ=0 no-mixing or γ>0 over-mixing). `train_generative` (k=100, per-epoch write-back, reinit, live noise) is where persistence plausibly matters and has always been persistent.

---

## Limitations / confounds
1. **A.1 probes the momentum marginal only** — the honest Gibbs test in q is impossible on these checkpoints (no confining stationary state; H drifts down, chains leave the cube). The conjecture as stated is about per-mode temperatures, so this is the right observable, but "fdt ⇒ samples the Gibbs measure" remains unverifiable end-to-end here (also F5 Open-3: relativistic momentum marginal is non-Gaussian; flatness of Var(pᵢ)/M_eff,i is the isotropy-exact test used).
2. **A.2 uses one checkpoint (mnistFFF) for generation** — mode-set membership is checkpoint-dependent (cf. paper's 3/5/8/9). The *anisotropy-null* is expected to generalize (A.1 shows the same narrow T_eff spread on all three checkpoints), but was demonstrated on one.
3. Samples are low-confidence digits (mean maxprob 0.75; speckly — see `a2_sample_grids.png`); classifier-based mode assignment on such images is noisier than on real MNIST. Paired-chain design + per-seed consistency (3 seeds, near-identical hists) mitigates.
4. Study B: n=3 seeds, 500 epochs (not 1000), and the post-wave-2 `lyapunov_penalty="max"` default (live regularizer) — the null is internally controlled but is a statement about *this* trainer configuration; k_steps was not swept (flagged as the only live-PCD regime).
5. mnistFFF was trained under legacy sleep noise; A.2 sampling from it with fdt is a train/sample mismatch by design (task-intended). The "landscape, not sampler" conclusion is about *sampling-time* effects only.

## Code gaps flagged for experiment-engineer (not fixed here, per protocol)
1. (Known, confirmed relevant) `exp_c_dreaming.run_dream_batch` calls `stochastic_rollout` without `noise_mode` — dream phase can't use `fdt` from the CLI; my A.2 had to bypass exp_c. One-kwarg-per-call fix.
2. (Known) `train_generative` negative-chain scales hardcoded for [−1,1] data (q_noise U(−1,1), clamp ±1) — unchanged, still true.
3. **NEW (§7.12 regression evidence):** the `zzz_chlu_dev.pth` shim itself was found `UF_HIDDEN`-flagged at session start (along with all managed .pth files, incl. `_editable_impl_chlu.pth` freshly rewritten Jul 6 12:21) — fix-pack-2's durability assumption ("uv never rewrites unmanaged files ⇒ shim never re-flagged") is violated: whatever applies the flag hits *existing* unmanaged files too, not just fresh editable-install writes. `chflags nohidden .venv/lib/python3.11/site-packages/*.pth` healed it; recommend `make fix-env` (or at least the chflags line) as a **per-session** preamble in every task file until the flag-applier is identified.
4. Minor: `train_chlu` records only wake losses (sleep-loss history is computed but discarded) — cheap observability win for future sleep-phase studies.

## Recommended next experiments
1. **Temperature-calibration note > sampler fix:** since the anisotropy is inert but the units are wrong by ~11× (and ∝dt), the practical deliverable from §7.9 is a calibration table/wrapper (nominal→energy-units via κ) for any Exp-C follow-up or paper revision; `fdt` should be default for *new* projects (correct units, no downside observed), with legacy kept for checkpoint reproduction.
2. **Landscape-based imbalance study** (the mechanism left standing): per-digit basin statistics on exp_c checkpoints — V at class centroids/exemplars, basin volumes via relaxation-catchment from perturbed class exemplars → predicts the {4,5,9,8} frequencies? Directly tests "imbalance = landscape geometry".
3. **Re-test the FDT-imbalance link on a model with a real mass hierarchy** (once V3 designed banding / mass-aware objectives exist) — the conjecture's conditional survives; current refutation is at max/min≈1.5.
4. **fdt-trained Exp-C run** (laptop, ~200 quick epochs): does training under correct per-mode noise change the learned landscape/mass pattern (mass-peek's ink-light/border-heavy) — different question from A.2, now cheap to ask.
5. Study-B closure for the record: a k_steps sweep (k∈{10,50,100} at γ∈{0.05,0.2}) would map where PCD starts to matter in the dynamics path, if anyone ever wants to defend Algorithm 1 wording; low priority given the null.

Git footprint: none (no tracked files touched; analysis scripts/results/figures all under `.claude/`). Untracked env side-effect: re-ran `chflags nohidden` on `.venv/.../*.pth` (see gap 3).

## Proposed handover updates (for the Hub)
- **§7.9 → conjecture RESOLVED (2026-07-06, this report):** Prop-9's mechanism confirmed *exactly* on all 3 healthy Exp-C checkpoints (legacy: slope(logVar(pᵢ),logM_eff,i)=0.00±0.004, T_eff,i∝1/M_eff,i at ρ=0.994, Var(p)=2dtT/(2−γ) to ≤0.06%, T_eff∝dt verified; fdt: slope 1.00, single temperature to 0.8% CV). **But the MNIST mode-imbalance attribution is refuted:** scale-matched fdt sampling leaves the generated digit distribution statistically identical to legacy (χ²=0.08 p=1.0; 2/192 paired class flips; paired-image L2 0.10 vs 9.7 inter-image) — the imbalance is landscape geometry (det-annealed control: 95% collapse to '5'), and the per-mode anisotropy is too weak at current mass spreads (max/min≈1.5 ⇒ T_eff spread 1.28×). Practical residue: **nominal temperatures are ~11× hotter than legacy effective units (κ≈0.09 at γ=0.3, dt=0.05)** — calibration, not correction, is what Exp-C reproduction needs. Conditional retained: a future mass *hierarchy* re-arms the conjecture.
- **§7.4 → CLOSED (empirically):** `persistent_sleep_buffer` is functionally inert in the dynamics path — 3 matched seeds × 2 arms: all readouts (wake loss, target_energy, learned M, governed noise-rejection MSE at 4 σ's) agree to ≤1e-3 absolute / ~1e-5 relative; mechanism proven from the runs: γ_sleep=0 ⇒ zero Langevin noise (σ∝√γ) + isoenergetic conservative negatives (E drift 0.03%/500 steps); γ_sleep=0.2 supplement ⇒ one 500-step sleep event fully mixes ⇒ CD≡PCD again. Default False is safe; historical Exp-A/B results need no correction; only k_steps ≪ mixing time could ever distinguish them.
- **§1.6 Exp-III note:** deterministic-annealed mode collapse reproduced quantitatively on mnistFFF (182/192 → '5', 0.35 bits); stochastic legacy pipeline gives entropy 2.04 bits over modes {4,5,9,8} (n=192, 3 seeds) — paper's {3,5,8,9} set is checkpoint-dependent.
- **§7.12 update:** durable-shim assumption broken — `zzz_chlu_dev.pth` itself found UF_HIDDEN'd on 2026-07-06 session start; per-session `chflags nohidden .venv/.../*.pth` (or `make fix-env`) required until the flag-applier is identified. (Also: this session's `.pth`s incl. a fresh `_editable_impl_chlu.pth` write from another agent's `uv sync` were all flagged.)
- **§6/§9 ops:** mid-task model switches can mount a different tool profile (this thread temporarily lost Bash/Write after a `/model` change and had to pause) — spawning guidance should note: keep one model per spoke thread, or expect to re-grant tools.
- New reusable assets: `mnist_cache.npz` (70k×784, [−1,1], canonical order) and `mnist_classifier.joblib` (MLP 97.67%) under `.claude/scratch/generative-studies/` for any future Exp-C scoring.
