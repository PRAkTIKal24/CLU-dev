# v2-referee-experiments — results-analyst report

**Task + acceptance criterion:** run the three small experiments the referee's SHOULD-FIXes need (SF-1 Mo's own λ̂(T=128) estimator across all regimes; SF-2 per-step FLOP/wall ratio CLU-vs-LSTM-vs-LEM; SF-3 GMOR + EP onset re-verified on an anchored 3000-ep checkpoint). Deliver numbers + which draft section each feeds. Repo read-only.
**Status:** **done** — all three. SF-1 turned out **already-run** (extract+plot only). SF-2 and SF-3 executed fresh, laptop-CPU, reproducible.

**Provenance (shared).** Repo `main` @ **`37dc664`**, clean tree, read-only (no tracked files touched). Env: main venv `/Users/user/Desktop/CHLU/.venv` (JAX warm ≈2 s import — no cold-start hit this session). All artifacts under `.claude/outputs/v2-referee-experiments/` (npz/json/png) and `.claude/scratch/v2-referee-experiments/` (scripts, 3 anchored `.pkl`). SF-1 source data: `.claude/outputs/v2-full-runs/gmor_sweep.npz` (battery-2, commit `dbeb2c2`, 150-ep checkpoints).

---

## SF-1 — Mo's own λ̂(T=128) estimator across all regimes  → feeds §3.2 / Fig 2

**The sweep was already run.** `v2-full-runs/gmor_sweep.npz` already carries `lam_hat_128` for **all 70 rows (5 seeds × 14 δ)** — computed by `probe_common.group_tangent_exponent(T=128)`, Mo's finite-horizon group-tangent exponent λ̂(T)=(1/T)ln‖AᵀΞ‖/‖Ξ‖ at the settled tilted vacuum, Ξ=(Xq*,0). No retraining or re-running needed; this is the extraction/plot job the referee's item-3 anticipated. (Sign note: the stored `lam_hat_128` is the log-decay exponent = **−(per-step rate)**; the rate magnitude used below is `−lam_hat_128`.)

**Flag-provenance (SF-1)**

| field | value |
|---|---|
| source | `v2-full-runs/gmor_sweep.npz`, commit `dbeb2c2` |
| checkpoints | designed (so2_invariant, tied), **train_epochs=150** (Finding-0 regime) |
| seeds | 42–46 (5) |
| probe | x64; dt=0.05, γ=0.05, kick 0.1, tilt n=1; Mo protocol φ₀=0.35, thr 0.2, cap 15000 |
| non-defaults | lyapunov_penalty="max"(λ=0.01), langevin_noise="legacy", persistent_sleep_buffer=False, sleep_temperature=0.5, sleep_frequency=5 |
| estimator | λ̂(T=128), group-tangent seed Ξ=(Xq*,0) |

**Lifetime predictor from Mo's OWN estimator vs from the exact Jacobian gap vs measurement** (mean over seeds; lifetime predictor = 0.847298/rate):

| δ | regime | measured | pred (exact gap) | **pred (Mo λ̂128)** | meas/pred_gap | **meas/pred_λ̂** |
|---|---|---|---|---|---|---|
| 1e-3 | over | 11232.8 | 11102.1 | 13098.5 | 1.012 | **0.858** |
| 3e-3 | over | 3747.0 | 3689.6 | 4357.1 | 1.016 | 0.860 |
| 1e-2 | over | 1126.8 | 1095.1 | 1297.5 | 1.029 | 0.868 |
| 3e-2 | over | 378.8 | 353.3 | 423.1 | 1.072 | 0.895 |
| 6e-2 | over | 192.8 | 167.0 | 204.0 | 1.155 | 0.945 |
| 0.1 | over | 119.4 | 91.0 | 115.7 | 1.313 | **1.032** |
| 0.17 | ≈EP | 76.0 | 34.7 | 59.5 | 2.202 | 1.278 |
| 0.6 | under | 31.0 | 33.0 | 29.3 | 0.938 | 1.060 |
| 2 | under | 15.2 | 33.0 | 35.7 | 0.460 | 0.426 |
| 4 | under | 10.2 | 33.0 | 33.6 | 0.309 | **0.304** |

(δ≤3e-4 censored — same pattern as Mo's own censored ε=1e-4 row.)

**Result — the predictor-substitution gap (referee SF-1) is closed.** Using **Mo's actual estimator** as the lifetime predictor:
- **Overdamped:** `corr(log pred_λ̂, log meas) = 0.9995` (even tighter than the exact-gap 0.9987); meas/pred ratio 0.86–1.03 brackets 1. Mo's own estimator tracks the retention budget in his regime.
- **Underdamped (past EP):** the estimator-based prediction fails in the **same calculable ballistic direction** — meas/pred_λ̂ falls to 0.30 at δ=4, mirroring the exact-gap 0.31.
- **So "Mo's law is the overdamped face" survives on Mo's own estimator, not a substituted predictor.** Both predictors agree overdamped and both break identically past h*.

**Estimator-vs-exact-gap deviation (correcting a draft number).** The finite-horizon λ̂(T=128) carries a systematic transient bias vs the asymptotic gap: **≈−15% in deep overdamped** (gap·T≪1, transient-dominated at a fixed point), growing to a **per-row max of −44.5%** — and that 44% maximum sits at the **near-EP row δ=0.17 (gap·T≈3.1)**, *not* at gap·T≲0.1 as §3.2 currently states. Deep-overdamped (gap·T<0.1) max is only −15.6%. **Recommend the draft attribute the 44% to the near-EP transient, not to gap·T≲0.1.** Crucially, despite the 15–44% rate bias, the *lifetime* prediction still tracks measurement (the bias lengthens predicted lifetime, pulling meas/pred from 1.01–1.16 down to 0.86–1.03) — the containment claim is robust to it.

**Deliverable:** `sf1_mo_estimator_overlay.png` (top: three lifetime curves vs δ, shaded past-EP band; bottom: meas/pred ratio for both predictors). Extract: `mo_estimator_extract.npz`, `mo_estimator_table.json`.

---

## SF-2 — per-step FLOP & wall-time: CLU-Verlet vs LSTM vs LEM  → feeds §3.3

One CLU dissipative-Verlet step (KDK: **two** ∇_qV backprops through a dim-4→64→64→1 tanh MLP, newtonian_learned, γ=0.05, dt=0.05) vs one LSTM cell vs one LEM cell (both in=1, hidden=16, out=2 — the exact prefreeze-baseline retention models).

**Flag-provenance (SF-2):** commit `37dc664`; f32; CLU = `build_exp_d(42, potential_type="mlp")` (exp-d emergent arch, the 263-step model); FLOPs via XLA `cost_analysis()`; wall = median of 7 reps of a `lax.scan` of 2×10⁵ steps (dispatch-amortized). Seeds don't affect FLOP/timing.

| unit | params | **FLOPs/step** | wall ns/step (scan-amortized) |
|---|---|---|---|
| CLU Verlet (h64, dim4, KDK) | 4549 | **36148** | 1554 |
| LSTM cell (h16) | 1186 | 2512 | 252 |
| LEM cell (h16) | 1186 | 2400 | 500 |

**Per-step ratios:** CLU/LSTM = **14.4× FLOPs, 6.2× wall**; CLU/LEM = **15.1× FLOPs, 3.1× wall**.
(The naïve single-call timing in `sf2_timing.json` is dispatch-bound — all three hit a ~5 µs Python-dispatch floor — so it is **not** informative; the scan-amortized `sf2_walltime_scan.json` numbers above are the compute-bound truth. Report the scan numbers.)

**Compute-normalized retention (retire the "4×").** The headline "263 vs 69 map-steps ≈ 4× longer retention" is in *map-steps*. Normalizing by per-step cost (retention_steps × cost_per_step):

| normalization | CLU vs LSTM | CLU vs LEM |
|---|---|---|
| raw map-steps (263 vs 69 / 56) | 3.8× *longer* | 4.7× *longer* |
| **FLOP-normalized** | **54.8× more compute** | **70.7× more compute** |
| **wall-normalized** | **23.5× more wall** | **14.6× more wall** |

**The 4× does NOT survive compute normalization — it inverts.** Per unit of retention, CLU spends ≈15–24× the wall time (≈55–71× the FLOPs) of the RNN it "beats." **Recommendation: retire the quantitative "≈4× longer" as a compute claim; lead §3.3 with the qualitative structural triad (latch / μ²-budget-law / bounded drift), which is architecture- and compute-independent and is the stronger claim.** If a compute sentence is wanted, state it honestly: *"per-step, one CLU Verlet update costs ≈6× an LSTM / ≈3× a LEM cell in wall time (≈14× in FLOPs); the 4×-longer retention is a map-step statement that does not survive compute normalization."*
**Confound (must flag):** the comparison is *not* width-matched — CLU is hidden-64 (2 grad evals/step) vs baselines hidden-16. A width-matched CLU would narrow the gap, but the sign (CLU costlier/step) is robust because the Verlet step requires two backprops through the potential. The retention numbers (263/69/56) are themselves from those specific configs, so the compute penalty is the honest apples-to-apples for *those* results.

---

## SF-3 — GMOR tilt-sweep + EP onset on an ANCHORED 3000-ep checkpoint  → feeds §3.5 / §3.1–3.2 survival claim

Re-verify the headline laws (§3.1 GMOR μ²∝δ, §3.2 retention, C3 EP √(h−h*)) on checkpoints trained to **3000 ep WITH the anchor λ=100** (fix-pack-4's bulletproof setting), i.e. **past the sleep-erosion horizon** that destroys the vacuum at default 1000-ep (v2-full-runs Finding 0).

**Flag-provenance (SF-3)**

| field | value |
|---|---|
| commit | `37dc664` |
| training | so2_invariant, tie_channel_mass=True, **anchor_lambda=100** (anchor_target=mean V(ring), anchor_data=ring q), **train_epochs=3000** |
| seeds | 42, 43, 44 (3) |
| non-defaults | sleep_frequency=5, sleep_steps=500, persistent_sleep_buffer=False, sleep_friction=0.0, lyapunov_penalty="max", langevin_noise="legacy", dt=0.05, newtonian_learned |
| driver | `sleep-erosion-study/driver.py::train_checkpointed` (validated train_chlu replica) + anchor term |
| wall | 132 s/model (f32, CPU) |
| probe | x64 (`v2-full-runs/probe_common`); dt=0.05, γ=0.05, kick 0.1, tilt n=1; grid δ∈{1e-4…4}, EP factors 0.7–4×δ* |
| checkpoints | `.claude/scratch/v2-referee-experiments/anchored3000/anchored_l100_s{42,43,44}_ep3000.pkl` |

**Vacuum intact at 3000 ep (anchor holds):** ring_depth (V₀−V_ring) = **+0.106 / +0.101 / +0.124** (positive ⇒ data ring is a well, not a max), r* = **0.925/0.909/0.918 (mean 0.917±0.007)**, flat-mode μ² = **−7.6e-16 / −1.3e-15 / +2.2e-15 (machine-flat)**, angular-mode overlap 1.0000. (r* matches `anchor-robustness` 0.911±0.016; seed-42 wake loss 0.06102 reproduces anchor-robustness `chlu_l100_s42` to 5 digits — cross-run reproducibility check passes.)

**GMOR μ²∝δ law — EXACT at 3000 ep, 3/3 seeds:**
`μ²_meas / [δn²/(M_ch r*²)] = 1.00000 ± ≤1.5e-12` at every δ across **4.6 decades** (δ 1e-4→4). The GMOR spectral-mass law is untouched by 3000 ep of anchored wake–sleep training.

**Retention law survives:**
- overdamped power-law slope **−0.956** (F5 predicts −1; cf. 150-ep −0.985 — marginally softer but the law holds);
- saturation at the **mass-independent floor 27.03** in underdamped (n₁/₂ 24–31 straddling 27.03 within the kick-phase ripple);
- flat/latch mode preserved (μ²≈1e-15 ⇒ ∞ retention on the coset).

**EP onset survives — identical to 150 ep:**
- **below EP: φ = 0 exactly** (max freq_jac = 0.000e+00, all seeds);
- **above EP: log-log slope φ vs (h−h*) = 0.5165** (C3 predicts 0.5) — **bit-identical to the 150-ep 0.5165**.

**Conclusion:** every headline law (GMOR exactness, retention slope/floor, latch, EP √-onset) **survives past the erosion horizon under the shipped anchor cure**. §3.1/§3.2 laws — verified at 150 ep for pre-erosion cleanliness — are confirmed to still hold at 3000 ep with λ=100. SF-3's "do your laws survive your own cure at long horizon?" is answered **yes**.

**Deliverable:** `sf3_anchored3000_laws.png` (GMOR retention + EP onset), `sf3_gmor_sweep.npz`, `sf3_ep_sweep.npz`, `sf3_per_seed.json`, `sf3_train_manifest.json`.

---

## Limitations / confounds
1. **SF-1** rests on the 150-ep battery (its own provenance regime); the estimator is a *Jacobian-derived* λ̂ (Aᵀ applied to the group tangent), faithful to Mo's finite-horizon definition but not a full trajectory-fit escape run of his exp19 systems (which are non-public). The overdamped correlation and the past-EP failure direction are robust regardless.
2. **SF-2** is single-core CPU, XLA, batch-1; GPU/batched throughput could shift wall ratios (not FLOPs). The width mismatch (h64 vs h16) is the dominant confound — flagged above. Direction of the conclusion (CLU costlier/step, 4× does not survive) is robust to both.
3. **SF-3** overdamped slope −0.956 is slightly softer than the 150-ep −0.985; consistent with mild anharmonic drift of the longer-trained V, not a law failure (GMOR ratio is still exactly 1). 3 seeds (task-specified); r*/M_ch seed-invariant so tiny GMOR error bars, as at 150 ep.
4. No divergences/NaNs/OOM encountered in any run.

## Recommended next experiments (cheap)
- SF-2 fairness: a **width-matched** CLU (h16 potential) vs LSTM/LEM per-step ratio to isolate the "2 grad evals" cost from the width cost — would let the paper state a width-controlled per-step factor.
- SF-3 could add seeds 45/46 to match the 150-ep 5-seed battery if the Hub wants error bars identical across §3.1 and §3.5 (currently 3 vs 5).

## Git footprint
None — repo read-only, no tracked files touched (HEAD `37dc664`, clean). All artifacts under gitignored `.claude/`.

## Open questions / follow-ups / risks
1. Draft §3.2 currently pins the 44% λ̂ deviation to "gap·T≲0.1"; my extraction puts the 44% max at the **near-EP** row (gap·T≈3.1) and only −15.6% at gap·T<0.1. The writer should correct the attribution (numbers in SF-1 above).
2. SF-2's compute-normalized inversion is a genuine *weakening* of a current draft sentence — the Hub should decide between (a) retiring the 4× and leading qualitative, or (b) stating the honest per-step factor. Analyst recommendation: (a).

---

## Proposed handover updates (for the Hub)

- **§1.6/§5 (V2 results, fold in):**
  - **SF-1 (Mo estimator, §3.2/Fig 2):** Mo's OWN λ̂(T=128) as lifetime predictor tracks the retention budget overdamped (corr(log pred,log meas)=**0.9995**, meas/pred 0.86–1.03) and fails in the same ballistic direction past EP (**0.30 @ δ=4**, mirroring exact-gap 0.31). The predictor-substitution gap is closed — "Mo's law is the overdamped face" now rests on Mo's own estimator. **Correct the 44% attribution:** λ̂(T=128) deviates from the asymptotic gap by −15.6% max in deep overdamped (gap·T<0.1) and **−44.5% at the near-EP row (gap·T≈3.1)** — the 44% is a near-EP transient, not a gap·T≲0.1 effect. Overlay: `sf1_mo_estimator_overlay.png`.
  - **SF-2 (§3.3, retire the 4×):** per-step CLU Verlet (h64) = **36148 FLOPs / 1554 ns**; LSTM cell (h16) = 2512 / 252; LEM cell (h16) = 2400 / 500. Per-step CLU/LSTM = **14.4× FLOPs, 6.2× wall**; CLU/LEM = 15.1× / 3.1×. **The "263 vs 69 ≈4× longer retention" inverts under compute normalization: 23.5× more wall (54.8× FLOPs) vs LSTM, 14.6× wall (70.7× FLOPs) vs LEM.** Recommend leading §3.3 with the qualitative triad; if a compute line is kept, state the honest per-step ≈6×(LSTM)/≈3×(LEM) wall factor. Confound: not width-matched (h64 vs h16).
  - **SF-3 (§3.5, laws survive the cure):** on anchored λ=100 **3000-ep** checkpoints (3 seeds), vacuum intact (r*=0.917±0.007, flat μ²≈1e-15, ring_depth +0.10–0.12); **GMOR μ²/δ ratio = 1.00000±1e-12 over 4.6 decades**; overdamped retention slope **−0.956** (pred −1) with floor 27.03; **EP φ=0 below, slope 0.5165 above (pred 0.5, bit-identical to 150 ep)**. Headline laws survive past the erosion horizon under the shipped anchor. Fig: `sf3_anchored3000_laws.png`.
- **No code bugs** for experiment-engineer this round (probe_common `mu_floor` and Noether-r* guards were already flagged by v2-full-runs; not re-hit here).
