# clu-anomaly-scorer — experiment-engineer report
Task + acceptance criterion: build the CLU→eval-harness bridge (`CHLUScorer(BaselineScorer)` + compliant factory + `chlu eval` + CSF3 job + laptop smoke) so the G7b torus-CLU voraus-AD flagship can be scored at all; acceptance = ABC implemented, both `energy`/`residual` and `predict` arms + ROC/AUROC alongside VUS-PR in an `EvalRunResult`, `job_gpu_eval.sh` exists, laptop smoke on a real dataset gives finite CLU numbers alongside baselines, hybrid hook exists but untuned, settling-time marker placed, defaults unchanged elsewhere, suite green.

**Status: done.**

**Owner note (protocol §5 corollary):** no downstream reconciliation list. One proposed CORE addition is flagged for a later pack (see Open questions) — it is a proposal, not a live contradiction.

**No PREREG.md:** the acceptance criterion is an infrastructure/finiteness bar, not a measured ratio/exponent/slope/law, so the pre-registration rule does not apply. The smoke numbers below are reported as diagnostics with a full flag-provenance table, not as a claim.

## What I did
- **`chlu/eval/clu_scorer.py` (new):** `CHLUScorer(BaselineScorer)` — the first CLU implementing the harness `fit(train_windows)/score(windows)` contract. One trained CLU is read by parallel score arms (`_SharedCLUFit` trains once, all arms reuse it → "the comparison is one experiment", per Head 2026-07-19):
  - `energy` — mean `H(q,p)` over a window's states with finite-difference momentum (EBM reading).
  - `residual` — relaxation residual `R0 = mean ‖∇V(q_relaxed)‖²` after a short damped rollout from evenly-spaced window anchors ("fails to settle into a basin").
  - `predict` — multi-step CLU-rollout prediction MSE over the window (conventional TSAD; fairest head-to-head vs PCA-recon).
  - `hybrid` — UNTUNED equal-weight z-score combination of energy+predict, using train-set stats (hook only; documented as final-iteration, not tuned in pass 1).
  - Training = simplified Hamiltonian Contrastive Divergence (§1.4): wake predict-MSE (makes `predict` meaningful) + denoising-EBM contrastive energy `mean H(data) − mean H(data+noise)` + `energy_reg·(⟨H²⟩)` magnitude regularizer (mirrors `train_generative`'s 0.005 term; makes `energy`/`residual` meaningful). `eqx.filter_jit`/`filter_value_and_grad`, Optax Adam, explicit PRNG threading.
  - Respects the scaler contract: consumes the already-`StandardScaler`-scaled flattened windows the harness passes; infers channel count `C = width // window_size`.
- **`make_clu_scorers(config, clu_config, modes)` (in `clu_scorer.py`):** returns `{**make_default_baselines(config), "clu_<mode>": CHLUScorer(...)}` — keeps the mandatory four statistical baselines (harness rejects a factory without them) and adds one CLU arm per mode. Default modes = `("energy","residual","predict")` (both mandatory first-pass arms + residual variant).
- **`chlu/eval/config.py` (extended, NOT `chlu/config.py`):** `CLUScorerConfig` (all scorer knobs, explicit, smoke-fast defaults), `CLULatticeConfig` (G7b torus-coset hook), `CLU_SCORE_MODES`, `CLU_DEFAULT_SCORE_MODES`. **Parked settling-time marker placed** in the `CLU_SCORE_MODES` docstring (`# FUTURE: settling-time score (handover 2026-07-19)`).
- **`chlu/eval/harness.py` (minimal, backward-compatible):** optional `raw_scores: dict|None=None` collector threaded through `evaluate_dataset`/`_run_cross_unit`/`_run_per_unit_prefix`; when a dict is passed it is filled with pooled `{method:{"scores","labels"}}` so ROC/PR curves can be re-plotted without rerun. Default `None` = no behaviour change (all 34 existing eval tests still pass).
- **`chlu/cli/eval_cmd.py` (new) + registration:** `chlu eval --dataset {voraus,skab,tep,smd} [--score-mode …] [--seed] [--limit] [--out] [--window] [--lattice …] [--download] [--quick] …`. Calls `evaluate_dataset`, writes `eval_<ds>.npz` + `eval_<ds>.md` + `eval_<ds>_raw.npz` + `eval_<ds>_roc.npz` (per-arm fpr/tpr + AUROC/AUPR), prints per-arm primary-metric + AUROC.
- **`scripts/csf3/job_gpu_eval.sh` (new):** mirrors `job_gpu_single.sh` (A100 gpuA, `module purge`, `env.sh`, JAX-GPU preflight assert) but invokes `chlu eval` (`DATASET/SCORE_MODE/SEED/OUT/EXTRA_ARGS`; results under `~/scratch`). Header documents the array+`afterany` seed-sweep pattern (Head's `sample_script_csf3.sh` pattern, not its HEPA specifics).
- **`tests/test_eval_clu_scorer.py` (new):** 14 tests — ABC compliance, factory keeps baselines + adds arms, shared-model reuse (no retrain), per-arm finite/non-constant scores, energy separates gross anomalies, before-fit guard, non-divisible-width guard, single-unit→CHLU, lattice-hook→CLULattice of matching dim, exact-tiling guard, harness end-to-end + raw-score collection + npz round-trip.
- **Torus-coset hook (flag, not default):** `CLULatticeConfig` → `_build_model` builds a `CLULattice` via `build_lattice` (tiles C channels into `unit_dim`-channel units; `coupling_type="auto"`→`channel_spring` for `so2_invariant`; chain or torus topology). **Wired:** the scorer *accepts* a lattice config and fits/scores on the joint state. **NOT wired (next task `g7b-torus-voraus`):** the literal joint-angle→`so2_invariant`-coset mapping and non-divisible channel layouts — the hook requires an exact `unit_dim` tiling and raises otherwise.

## How I verified (real output)
- `ruff check` on all 8 touched files → **All checks passed!**
- New tests: `pytest tests/test_eval_clu_scorer.py` → **14 passed in 74.97s** (clu_scorer.py 95% covered).
- Existing eval tests (regression on the harness raw-score change): `pytest tests/test_eval_{baselines,metrics,splits}.py` → **34 passed**.
- **Full suite:** `pytest -q` → **299 passed, 7 warnings in 279s.** (Suite green.)
- **End-to-end laptop smoke on REAL data** (SKAB, downloaded live, GPL — not vendored): `chlu eval --dataset skab --download --quick --limit 8 --score-mode default --metrics-mode fast` → ran to completion, wrote npz + md + raw npz + roc npz; verified `eval_skab_roc.npz`/`eval_skab_raw.npz` load and shapes align (e.g. `clu_energy` 8055 pooled scores/labels; per-arm fpr/tpr present).
- `chlu eval --help` renders the full parser (dataset/score-mode/lattice flags).

### SKAB smoke — per-arm numbers (both arms, as the Head asked)
Per-unit nanmean over 8 test units (harness `EvalRunResult`), primary=VUS-PR; and pooled-across-timesteps AUROC/AUPR from the ROC npz:

| method | VUS-PR (per-unit) | AUROC (per-unit) | AUROC (pooled) | AUPR (pooled) |
|---|---|---|---|---|
| pca_recon | 0.5531 | 0.5863 | 0.5365 | 0.4484 |
| iforest | 0.5945 | 0.6096 | 0.6024 | 0.3775 |
| lof | 0.6099 | 0.6418 | 0.5837 | 0.4038 |
| knn | 0.8329 | 0.8725 | 0.6466 | 0.5447 |
| **clu_energy** | 0.3894 | 0.3829 | 0.4009 | 0.3568 |
| **clu_residual** | 0.4278 | 0.4775 | 0.5056 | 0.4089 |
| **clu_predict** | 0.3818 | 0.4010 | 0.4208 | 0.3474 |

**Reading (honest, CM-3-compliant):** the bridge produces **finite, in-range** CLU scores alongside the baselines — the acceptance evidence. On this *tiny --quick* config (20 epochs, window 32, ≤800 fit windows) the CLU arms are **below the statistical baselines** and `clu_energy`/`clu_predict` sit slightly *below* random AUROC (0.38–0.42), while `clu_residual` is ~random-to-weak (0.48/0.51). This is expected for a smoke — it is **not** a claim about CLU quality (that is `voraus-baseline-floors` + the real CSF run). I make **no** superiority claim (CM-3). The per-unit vs pooled AUROC gap (e.g. knn 0.87 vs 0.65) is because the harness reports per-unit-averaged AUROC while the ROC npz pools all timesteps — both are emitted so the Head can choose the operating-characteristic view.

## Flag-provenance (SKAB smoke)
- **Commit:** `5c45a87` (branch tip) · **base** local `main` `df5e44d`.
- **Env (reused main venv per §4):** jax 0.9.0, equinox 0.13.4, optax 0.2.6, Python 3.x; CPU (laptop).
- **Seed:** 42 (harness + CLU init/train/subsample).
- **Harness flags:** dataset=skab (`--download`), `--limit 8` (1 train unit = `anomaly-free`, 8 test units), window=32 (`--quick`), stride=1, train_stride=1, metrics_mode=fast, metrics_sliding_window=32, max_train_windows=2000 (`--quick`), 2000 train windows used.
- **CLU flags (`--quick` overrides in *bold*):** kinetic_mode=newtonian_learned (default), potential_type=mlp, hidden=64, dt=0.05, gamma=0.1, rest_mass=1.0, c=1.0, tie_channel_mass=False, **epochs=20**, lr=1e-3, batch_size=64, **max_fit_windows=800**, predict_horizon=16, relax_steps=32, residual_anchors=8, predict_weight=1.0, energy_weight=1.0, neg_noise_scale=0.5, energy_reg=0.005, momentum_init=finite_diff, lattice=None.
- Artifacts: `/tmp/clu_smoke_skab/eval_skab.{npz,md}`, `eval_skab_raw.npz`, `eval_skab_roc.npz` (scratch; not committed).

## Git footprint
- **Branch:** `agent/experiment-engineer/clu-anomaly-scorer` (off local `main` `df5e44d`; rebased onto `main` = up-to-date; **not pushed, left for review**). Worked in a dedicated **worktree** per §3.2 (concurrent `fix-pack-7` was live in its own worktree — no collision); verified branch ref from the main repo before removing the worktree.
- **Commits (4, atomic):**
  - `9e6f8c5` add CLU anomaly scorer + harness raw-score hook (`chlu/eval/{config,clu_scorer,harness}.py`)
  - `b59e478` add `chlu eval` CLI command (`chlu/cli/eval_cmd.py`, `chlu/cli/__init__.py`, `chlu/chlu.py`)
  - `fddad9c` add CSF3 GPU eval job (`scripts/csf3/job_gpu_eval.sh`)
  - `5c45a87` tests for CLU anomaly scorer + raw-score hook (`tests/test_eval_clu_scorer.py`)
- **Files:** 8 changed, +1108/−5. All in-scope (`chlu/eval/**`, `chlu/cli/**`, `scripts/csf3/**`, `tests/**`, plus the 2-line `chlu/chlu.py` CLI hook). **`chlu/core/**` and `chlu/config.py` untouched** (fix-pack-7's territory).

## Open questions / follow-ups / risks
1. **CLU score quality is unaddressed by design.** The smoke shows a working bridge, not a good detector. Next steps own this: `voraus-baseline-floors` (baselines) + a real CSF `chlu eval` run with a full-length CLU config (more epochs, window 100, larger `max_fit_windows`) to see if any arm is competitive. The `energy`/`predict` sign and the residual definition may need revisiting once real numbers land.
2. **Score-mode is the experiment (Head 2026-07-19):** the machinery to compare arms in one run exists; picking `energy` vs `predict` vs `hybrid` is a data call from the real run, not made here. `hybrid` is deliberately untuned.
3. **Proposed CORE addition (later pack, NOT done here per parallel-safety):** a lattice-side relaxation-residual helper (e.g. `CLULattice.settle`/a residual accessor) would let the residual arm avoid recomputing `jax.grad(V)` per anchor; I computed it from the public API (`H`, `potential_net`, `__call__`) instead, to stay file-disjoint from fix-pack-7. Flagging for a future `chlu/core` pack.
4. **G7b literal mapping remains for `g7b-torus-voraus`:** the hook builds a torus/chain lattice tiling C channels into `unit_dim`-channel units, but the literal joint-angle→coset wiring (and non-divisible/odd channel layouts, and mapping voraus's ~130 signals to `n` U(1) registers) is that task's job. The hook raises on non-exact tilings by design so the next task cannot silently mis-map.
5. **Perf:** `residual` relaxes `residual_anchors` (default 8) anchors × `relax_steps` per window with a per-anchor `jax.grad(V)` — fine on GPU/CSF3 and for `--limit` smokes; for full point-labelled datasets (per-timestep, stride 1) it is the heaviest arm. Capped via `residual_anchors`.

## Proposed handover updates (for the Hub)
- **§2/§3 (architecture/CLI):** new `chlu eval` command and `chlu/eval/clu_scorer.py` (`CHLUScorer`, `make_clu_scorers`) — the CLU→harness bridge (G7b prerequisite). New scorer config in `chlu/eval/config.py` (`CLUScorerConfig`, `CLULatticeConfig`), NOT `chlu/config.py`. New CSF3 job `scripts/csf3/job_gpu_eval.sh`.
- **Harness API:** `evaluate_dataset(..., raw_scores=dict)` now optionally emits pooled per-method raw score/label arrays (ROC-replottable); default off, back-compatible.
- **Suite count:** full suite now **299 passed** (was ~200) — new `tests/test_eval_clu_scorer.py` (14 tests).
- **Score-mode-is-the-experiment (Head 2026-07-19)** is now buildable: `chlu eval --score-mode default` runs energy+residual+predict as parallel arms reading one CLU; `--score-mode all` adds the untuned `hybrid`; settling-time is marked as parked future work in `chlu/eval/config.py`.
- **Next task ready:** `g7b-torus-voraus` can pass a `CLULatticeConfig` to `make_clu_scorers`/`chlu eval --lattice`; the literal joint-angle→`so2_invariant`-coset mapping is unbuilt and flagged.
