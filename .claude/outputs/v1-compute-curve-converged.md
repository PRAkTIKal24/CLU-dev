# v1-compute-curve-converged — results-analyst report

**Task + acceptance criterion:** re-run `exp_v1_calibration` at converged training (`train_epochs=2000`) on `(128,32) (256,64) (384,96)` × 5 seeds and re-render the gated-compute-allocation figure, with every panel's gate accuracy ≥ 0.8 and the learned operating points reproducing the banked regime numbers.
**Status: done.** All three panels ≥ 0.8 (**0.9969 / 0.9938 / 0.9052**). No STOP condition triggered.

**⚠ DOWNSTREAM RECONCILIATION LIST (owner needed — protocol §5 corollary), 3 items:**
1. **The shipped figure is a 500-epoch run, not 400-epoch.** The task file, and therefore whatever it inherited from, says 400. `.claude/outputs/v1-pivot/full/run_log.txt` contains `Training generative model for 500 epochs` **90/90 times and `...400 epochs` 0 times** (both polarities checked); `config.py:358` `experiment_v1_gate.train_epochs = 500`. Every site describing `figs/fig_compute_allocation.png` as "400-epoch" needs correcting.
2. **§4.1's "6.2×" for (384,96) is not reproduced by this harness** — it is **5.54 ± 0.20** here vs **6.09 ± 0.85** banked (0.64σ, not material — see §3). If §4.1 prints 6.2× it is quoting the regime harness, and that provenance should be named.
3. **The converged figure is three flat lines.** Any caption asserting a visible accuracy-vs-compute *curve* must change; see §5.

---

## DIAL DECLARATION (echoed per protocol §7)
- **Dial:** compute-adaptive reads.
- **Laundering control:** naive `margin-gated` and `raw-R-gated` arms plotted on the same axes — both present in every panel, neither dropped.
- **Falsifies:** no step reduction at converged accuracy, or a naive arm dominating the learned gate at equal compute. **Neither occurred.**
- **Does NOT falsify:** losing to Hopfield/an oracle on a metric-native protocol. (Hopfield *does* beat the gate at `(384,96)`: 0.9615 vs 0.9052. Metric-native-ceiling theorem, not news; and Hopfield is cut from V1 figures by Head ruling.)

---

## 0. Pre-flight: the re-run was necessary (verified, not assumed)
```
python -c "import json,glob;print(sorted({k for f in glob.glob('.claude/scratch/regime-remap-2000ep/runs/*.json') for k,v in json.load(open(f)).items() if isinstance(v,list)}))"
→ []          (over 198 files)
```
Zero list-valued keys across all 198 banked cells ⇒ no `score_stages`/`correct_stages`/`cost` arrays ⇒ a curve is genuinely unrecoverable from the banked converged data. Re-run confirmed as the only route.

**Epoch clamp verified NOT in force.** `quick=False` was passed explicitly; the `train_epochs = min(train_epochs, 120)` clamps at `exp_v1_calibration.py:408` and `:1262` sit inside `if quick:` blocks. Run log line 9 onwards: `Training generative model for 2000 epochs...`; header confirms `ladder: 300 + 3 x 900 steps | gate features=r_margin | p_exit=0.5 | clamp_key=True`.

## 1. Setup — commands, seeds, artifacts
```
# driver (writes provenance.json, calls run_experiment_v1_calibration verbatim)
PYTHONPATH=/Users/user/Desktop/CHLU .venv/bin/python -u \
  .claude/scratch/v1-compute-curve-converged/driver.py \
  /Users/user/Desktop/CHLU/.claude/outputs/v1-compute-curve-converged/run2000
# render (shipped code path; only the RES + output-path lines differ — 3-line diff shown in report)
PYTHONPATH=/Users/user/Desktop/CHLU .venv/bin/python .claude/scratch/v1-compute-curve-converged/render_2000.py
PYTHONPATH=/Users/user/Desktop/CHLU .venv/bin/python .claude/scratch/v1-compute-curve-converged/tap.py
```
**Wall time: 822.8 s (13.7 min)** for 15 cell-seeds / 40 episode trainings — far under the 1–3 h budget. Per-cell 41–91 s. No divergence, no NaN in any accuracy, no OOM.

| artifact | path |
|---|---|
| **canonical figure (no Hopfield)** | `.claude/outputs/v1-compute-curve-converged/fig_compute_allocation_2000ep.png` |
| presentation variant, shared y-axis | `.claude/outputs/v1-compute-curve-converged/fig_compute_allocation_2000ep_sharedy.png` |
| with-Hopfield repro (NOT for V1) | `.claude/outputs/v1-compute-curve-converged/REPRO_with_hopfield_2000ep.png` |
| data tap (2000 ep) | `.claude/outputs/v1-compute-curve-converged/data_tap.json` |
| data tap (shipped 500 ep, for §4) | `.claude/outputs/v1-compute-curve-converged/data_tap_SHIPPED500ep.json` |
| summary + arrays | `.../run2000/results/exp_v1_calibration_{summary.json,metrics.npz}` |
| run log (timestamped) | `.../run2000/run_log.txt` |
| machine-readable config dump | `.../run2000/provenance.json` |

### Flag-provenance table (protocol §5)
| field | value |
|---|---|
| commit (HEAD, unmodified) | `7fcef50` |
| JAX | **0.9.0** (main venv reused; no worktree sync) |
| seeds | 42, 43, 44, 45, 46 (`base_seed=42`, `calib_n_seeds=5`) |
| **`train_epochs`** | **2000** ⟵ *non-default (default 500)* |
| **`calib_difficulty_levels`** | **`[[128,32],[256,64],[384,96]]`** ⟵ *non-default* |
| **`use_pretrained`** | **True** ⟵ *non-default; crash insurance only. `models_dir` was empty at launch, so nothing was loaded — every cell trained from scratch this run (log shows 40 × "Training generative model for 2000 epochs", 0 × "loaded").* |
| `calib_n_policy_taus` | 25 (default, unchanged) |
| `calib_features` | `r_margin` (default) |
| `calib_p_exit` | 0.5 (default) |
| ladder | `relax_steps=300` + `calib_n_stages=3` × `calib_stage_steps=900` ⇒ cost stages `[300, 1200, 2100, 3000]` |
| `calib_l2` / `calib_fit_all_stages` | 1.0 / True |
| `calib_min_trials_per_level` / `max_episodes_per_level` | 128 / 8 ⇒ n_eps = 4 / 2 / 2 at kv 32 / 64 / 96 |
| probes | 8/key @ σ=[0.05,0.15,0.3] + 16 impostors |
| `vocab_size` / `embed_dim` / `embed_scale` / `hidden_dim` | 256 / 16 (⇒ CLU dim 32) / 2.0 / default |
| `kinetic_energy_mode` / `potential_type` / `clamp_key` | `relativistic` / `mlp` / True |
| `dt`, `governor_sensitivity`, train lr/bs/k_steps/friction/temp/noise | all default (see `run2000/provenance.json`) |
| everything else | stock `get_default_config()` |

---

## 2. Acceptance criterion 1 — every panel ≥ 0.8 ✅

| cell | gate acc (mean ± std, 5 seeds) | per-seed | gate cost (steps) | always-full acc | always-small acc | fidelity |
|---|---|---|---|---|---|---|
| N=128, kv=32 | **0.99687 ± 0.00383** | 1.000, 0.9922, 0.9922, 1.000, 1.000 | 308.4 ± 10.3 | 0.99687 | 0.99687 | 1.000 |
| N=256, kv=64 | **0.99375 ± 0.00313** | 0.9922 ×4, 1.000 | 308.4 ± 10.3 | 0.99375 | 0.99375 | 1.000 |
| N=384, kv=96 | **0.90521 ± 0.00607** | 0.9010, 0.8958, 0.9115, 0.9115, 0.9062 | 541.9 ± 19.3 | 0.90208 ± 0.00896 | 0.90104 | 0.975 |

All ≥ 0.8. (Errors in absolute terms: 0.4/128, 0.8/128, 18.2/192.)

## 3. Acceptance criterion 2 — agreement with the banked regime numbers ✅ (no material disagreement)

| cell | banked regime @2000ep (ne=5, s=0.0) | this run (exp_v1_calibration @2000ep) | Δ |
|---|---|---|---|
| (128,32) gate acc | 0.9963 ± 0.0031 | 0.99687 ± 0.00383 | +0.0006 (0.18σ) |
| (128,32) savings | 9.893 ± 0.213 | 9.737 ± 0.322 | −0.16 (0.73σ) |
| (256,64) gate acc | 0.9913 ± 0.0036 | 0.99375 ± 0.00313 | +0.0025 (0.68σ) |
| (256,64) savings | 9.540 ± 0.202 | 9.737 ± 0.322 | +0.20 (0.98σ) |
| (384,96) gate acc | 0.9088 ± 0.0247 | 0.90521 ± 0.00607 | −0.0036 (0.15σ) |
| (384,96) savings | **6.088 ± 0.848** | **5.543 ± 0.199** | −0.545 (**0.64σ**) |

Savings = `cost[-1]/gate_cost` = `3000/gate_cost`, matching `exp_v1_calibration.py:1170`; I report **mean-of-per-seed-ratios** (the banked field's own definition). Every deviation is < 1σ of the banked per-seed spread ⇒ **no STOP.** The banked (384,96) savings is itself noisy (per-seed 4.83 … 7.05).

⚠ **Confound worth naming:** these are *related but not identical* harnesses, so exact agreement was never expected. `_regime_cell` uses `_clustered_embeddings(rho=0, n_clusters=8)` and 5 episodes/cell; `exp_v1_calibration` uses `make_token_embeddings` and n_eps = ⌈128/kv⌉ (4/2/2). Treating this as "two independent runs of the same configuration" would be slightly too strong — it is two independent harnesses at the same (N, kv, epochs, seeds).

## 4. ⭐ THE §4 QUESTION: naive-vs-learned at convergence — **the 500-ep inversion is gone**

At 500 ep the naive margin arm beat the learned gate. At 2000 ep it does not — it **ties exactly** at the two easy cells and **loses on cost-to-peak** at the hard one.

**Shipped 500-epoch run** (recomputed with the identical tap, `data_tap_SHIPPED500ep.json`):

| cell | learned peak | margin peak | learned−margin (mean over 24 matched-compute pts) | grid pts learned ahead |
|---|---|---|---|---|
| (128,16) | 0.89876 @652 | **0.91680 @652** | **−0.00677** | 10/24, 0 tied |
| (128,24) | 0.55130 @1474 | **0.55751 @1709** | −0.00640 | 6/24, 0 tied |
| (128,32) | 0.29782 @1591 | **0.30516 @1944** | −0.00221 | 7/24, 0 tied |

(Reproduces the task's stated 0.9168 vs 0.8988 at the kv16 knee exactly.)

**This 2000-epoch run:**

| cell | learned peak | margin peak | raw-R peak | learned−margin mean | learned−rawR mean |
|---|---|---|---|---|---|
| (128,32) | 0.99687 @**300** | 0.99687 @300 | 0.99687 @300 | **0.00000** (24/24 exactly tied) | 0.00000 (24/24 tied) |
| (256,64) | 0.99375 @**300** | 0.99375 @300 | 0.99375 @300 | **0.00000** (24/24 exactly tied) | 0.00000 (24/24 tied) |
| (384,96) | **0.90521 @417** | 0.90521 @**652** | 0.90365 @1474 | **+0.00015** (8 ahead, 12 tied, 4 behind) | **+0.00097** (14 ahead, 7 tied, 3 behind) |

**Reading (honest):** the learned gate never loses at convergence. At (384,96) it and the margin arm reach the **same peak accuracy 0.90521**, but the learned gate gets there at **417 steps vs 652** — a **1.56× cheaper knee** — and it strictly dominates raw-R (higher peak, ahead on 14/24 matched-compute points, behind on 3). At kv32/kv64 all three arms are **bit-identical**, for the reason in §5. So: better than the 500-ep figure for the paper, but the win is a *cost-to-peak* win at one cell, not an accuracy win anywhere.

## 5. ⛔ The finding the Head needs before this ships: **at convergence the curve is flat**

Directly from the arrays (not the figure):

| cell | stage-wise accuracy at costs [300, 1200, 2100, 3000] | queries whose correctness changes anywhere on the ladder (per seed) |
|---|---|---|
| (128,32) | `[0.99688, 0.99688, 0.99688, 0.99688]` | **0, 0, 0, 0, 0** of 128 |
| (256,64) | `[0.99375, 0.99375, 0.99375, 0.99375]` | **0, 0, 0, 0, 0** of 128 |
| (384,96) | `[0.90104, 0.90312, 0.90208, 0.90208]` | 6, 3, 3, 1, 4 of 192 (≤3.1%) |

At kv32 and kv64, **not one query out of 128, in any of 5 seeds, changes its answer between 300 and 3000 Verlet steps.** The read is fully settled at stage 0. This is the *strongest possible* form of the sanctioned claim — **10× fewer steps at literally identical accuracy** — and simultaneously the reason the figure has no curve to draw: panels 1 and 2 are dead-flat lines and matplotlib autoscales onto the 4th decimal (panel 1's y-axis spans 0.993–1.001), which reads as noise. Hence the shared-y variant.

**Caption fences honoured:** the sanctioned form **"same accuracy at ~1/10th the steps"** is exactly what the data show; **"more compute buys more accuracy" is not supported anywhere** — always-full minus always-small is **+0.000 / +0.000 / +0.00104** (the last being 0.12σ of its own 0.00896 std). **Non-monotonicity, reported as required:** the (384,96) learned-gate curve rises 0.90104 → 0.90521 (peak @417) then falls back to 0.90312 @3000; the margin arm peaks at 652 and also falls. Amplitude ≤0.004, i.e. under one query out of 192 — noise, not structure, but it is non-monotonic and should not be described as a rising curve.

## 6. Data tap — the numbers a caption may quote

| panel | learned-gate peak acc | cost at peak | always-full acc | always-full cost | **step-reduction at peak** | learned operating point (p_exit=0.5) | always-small acc @300 | Hopfield (V1: not plotted) |
|---|---|---|---|---|---|---|---|---|
| N=128, kv=32 | **0.9969** | **300.0** | 0.9969 | 3000 | **10.0×** | 0.99687 ± 0.00383 @ 308.4 ± 10.3 | 0.99687 | 0.9922 |
| N=256, kv=64 | **0.9938** | **300.0** | 0.9938 | 3000 | **10.0×** | 0.99375 ± 0.00313 @ 308.4 ± 10.3 | 0.99375 | 0.9750 |
| N=384, kv=96 | **0.9052** | **417.4** | 0.9021 | 3000 | **7.19×** | 0.90521 ± 0.00607 @ 541.9 ± 19.3 | 0.90104 | 0.9615 |

Two distinct ratios exist and must not be conflated: **7.19×** is `3000 / cost-at-swept-peak`; **5.54×** is `3000 / learned-operating-point cost` (the p_exit=0.5 deployed gate, the number comparable to the banked 6.09×). For kv32/kv64 both are ~10×.

## 7. Limitations / confounds
- **n = 5 seeds, one base seed family (42–46);** identical to the shipped figure, so comparable, but the (384,96) savings spread (banked 4.83–7.05) shows a single seed is meaningless here.
- **`use_pretrained=True`** was set as restart insurance. It was inert this run (0 loads, 40 trainings) but is a non-default flag and is recorded as such.
- **kv32/kv64 are saturated at stage 0**, so they carry *no information about gating quality* — any policy, including a coin flip over exit thresholds, scores identically. Only (384,96) is a real test of the gate, and it is a weak one (18 errors).
- **Harness mismatch** with the banked regime table (§3) — different embedding generator and episode count.
- `pooledAUROC = nan` for 4 of 15 cells (128/32 seeds 42,45,46; 256/64 seed 46) because those cells have **zero wrong answers**, so wrong/right AUROC is undefined. Not a bug; worth knowing that any pooled-AUROC aggregate over this grid is computed on a subset.
- Convergence to 2000 epochs was **assumed from the banked regime sweep**, not re-verified here; I did not run 4000 ep to confirm the plateau.

## 8. Recommended next experiments
1. **Pick a harder third cell so the figure has a curve.** The whole grid is saturated at 2000 ep. A cell with converged accuracy in the 0.80–0.90 band *and* meaningful stage-to-stage movement would give panel 3's structure to all three panels. `(512,128)` (needs `vocab_size ≥ 512` per the `kv < vocab/2` guard) is the natural next capacity step; the banked sweep should be mined for its 2000-ep accuracy first.
2. **Or change the x-axis story:** since accuracy is compute-invariant, the honest figure may be *accuracy vs. capacity at fixed 300 steps* with an always-full reference, rather than accuracy vs. compute.
3. **Add an eval-noise or correlation stress axis** at these same cells — stress is what un-saturates them, and `regime_stress_grid` already banked 0.5/0.9 correlation and 0.3/0.6 eval-noise cells at 2000 ep with the scalar metrics; re-running `exp_v1_calibration` under stress would produce curves with genuine structure at ≥0.8 accuracy.
4. **Verify 2000 ep is actually the plateau** for (384,96) by taking the banked 4000-ep cells and, if they differ, re-running the calibration there.

## Git footprint
**None.** No tracked file was created, modified, or deleted; `git status --porcelain` is empty at `7fcef50`. All artifacts are under gitignored `.claude/`. No branch created (research-only spoke, protocol §3).

## Open questions / risks
- Should the canonical figure be the autoscaled one (faithful to the shipped code path, but panels 1–2 zoom onto 4th-decimal noise) or the shared-y variant (readable, same data, one-line difference)? **My recommendation: shared-y**, with the flatness stated in the caption as the result.
- **No PNG was copied into `.claude/NIPSsubmission/v1-ttcl/figs/`** — per the task, that happens on Hub acceptance only.
- No code bug found in `chlu/`. The two `quick` clamps behaved exactly as documented.

---

## Proposed handover updates (for the Hub)

**§1.6 / experiments — add:**
- `v1-compute-curve-converged` (2026-08-29, commit `7fcef50`, JAX 0.9.0, seeds 42–46, `train_epochs=2000`, levels `(128,32) (256,64) (384,96)`, 822.8 s): converged re-run of `exp_v1_calibration`. Gate accuracy **0.9969 / 0.9938 / 0.9052**, step reduction **10.0× / 10.0× / 7.19×** at the swept peak and **9.74× / 9.74× / 5.54×** at the deployed p_exit=0.5 operating point. All panels ≥ 0.8 ⇒ acceptance criterion 1 met.
- **The 500-ep naive-arm inversion does not survive convergence.** At 500 ep margin-gated beat the learned gate by 0.0068 (kv16 knee: 0.9168 vs 0.8988). At 2000 ep the arms are **exactly tied at kv32/kv64 (24/24 grid points)** and at kv96 they share peak accuracy 0.90521 while the learned gate reaches it at **417 vs 652 steps (1.56× cheaper)**; learned beats raw-R on 14/24 points. The laundering control is on the figure and does not dominate ⇒ **dial claim (compute-adaptive reads) not falsified.**
- **⛔ New constraint on the figure:** at 2000 ep, **0 of 128 queries in 0/5 seeds change correctness between 300 and 3000 steps** at kv32 and kv64 (kv96: ≤3.1%). The compute curve is flat. "Same accuracy at ~1/Nth the steps" is confirmed in its strongest form; **any caption promising a visible accuracy-vs-compute curve must be rewritten.** always-full − always-small = +0.000 / +0.000 / +0.00104 (0.12σ) ⇒ "more compute buys more accuracy" remains unsupported, as fenced.

**§5 / provenance — two corrections:**
- **The shipped `figs/fig_compute_allocation.png` is a *500-epoch* run, not 400-epoch.** Evidence: `.claude/outputs/v1-pivot/full/run_log.txt` has `Training generative model for 500 epochs` 90 times and `...400 epochs` 0 times; `config.py:358` sets `experiment_v1_gate.train_epochs = 500`. Propagate the correction wherever "400-epoch" appears.
- **§4.1's 6.2× for (384,96) is a regime-harness number.** `exp_v1_calibration` at the same (N, kv, epochs, seeds) gives **5.54 ± 0.20** vs banked **6.09 ± 0.85** (0.64σ — not material, no STOP). If both numbers appear in the paper, name the harness; the two differ in embedding generator (`_clustered_embeddings` vs `make_token_embeddings`) and episodes/cell (5 vs 2).

**§8 / next steps — add:** the V1 grid is **saturated** at converged training; the interesting figure now needs either a harder cell (`(512,128)`, requires `vocab_size ≥ 512`) or a stress axis (correlation 0.5/0.9, eval-noise 0.3/0.6 — already banked as scalars at 2000 ep), or a re-framed x-axis (accuracy vs capacity at fixed 300 steps).

**For `experiment-engineer`:** no bug to file. One nicety — `_summarize` emits `nan` pooled AUROC for zero-error cells (4/15 here); a guard or an explicit `n_cells_with_errors` field would stop downstream aggregates silently averaging over a subset.
