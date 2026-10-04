# v1-hopfield-stress — experiment-engineer report

Task + acceptance criterion: chart the CLU-gate-vs-Hopfield **regime map** — extend `exp_v1_calibration` with stress knobs and produce a figure showing where Hopfield dominates / is comparable / CLU-gate wins (accuracy, calibration, or abstention), + compute-allocation curves in stressed regimes + an honest framing paragraph. "Hopfield wins everywhere reachable" is a valid, reportable outcome.

Status: **done** (code + 22-green related tests + full laptop-scale pilot on both stress axes + a targeted CLU-strong-regime probe; regime map figures + summaries written; numbers below).

**One-line verdict: no crossover is reachable at laptop scale — Hopfield is dominant in all 26 cells across 4 sweeps. Both stress axes (correlated keys, noisy cues) DO break Hopfield (acc 1.0 → ~0.47–0.67 at max stress), but the CLU energy-memory degrades *faster* (cue-noise-fragile basins; storage collapses under correlated patterns), so the gap never closes; and the v1-pivot compute-savings advantage (7× here) collapses to ~1× once the task is hard everywhere. The driver of the gap is CLU memory fragility, not Hopfield robustness — so the crossover is gated by making the CLU memory competitive, which the map isolates as the prerequisite.**

## What I did
- Extended `exp_v1_calibration` (reused the episode helpers `_probe_cues`/`_ladder_records`/`_hopfield_confidences`/`_fit_episode_heads` verbatim; original driver untouched) with a self-contained regime-map driver `run_v1_hopfield_regime_map` that sweeps **capacity (kv) × a stress axis** over N seeds and classifies every cell.
- **Two fair stress mechanics** (CLU and Hopfield see the *identical* stressed inputs each cell):
  - `correlation` — `_clustered_embeddings(rho, n_clusters)`: pulls key/value embeddings toward shared centroids while preserving marginal norm (`emb = √(1−ρ²)·iid + ρ·centroid`); same-cluster cosine → ρ². The classic reduced-separation Hopfield failure mode. Changes stored content ⇒ the memory is retrained per ρ.
  - `eval_noise` — Gaussian σ added to the deployment query cue only (memory written from clean patterns; degrades retrieval, not storage).
- **Per cell**: CLU learned-gate accuracy (τ fixed at write time, `p_exit`) + compute savings, CLU full-budget calibrated abstention AURC, vs Hopfield accuracy + **Platt-calibrated logit-margin** AURC (same probe-fit head as v1-pivot — fair), coverage@risk for both, storage fidelity. Classifier `_classify_regime` → `hopfield_dominant` / `comparable` / `clu_gate_advantage` on mean Δacc and ΔAURC vs a `regime_comparable_margin` band.
- **Deliverable figure** `plot_v1_regime_map` (4 panels: classification map annotated with Δacc; Δacc heatmap; ΔAURC heatmap; accuracy-vs-stress lines with per-point compute-savings annotations).
- New `regime_*` knobs in `ExperimentV1GateConfig` (all defaults laptop-scale; the pre-existing `calib_*`/main-calibration behavior is unchanged). CLI hook `chlu exp-v1-regime [--project|--seed|--quick]`; export in `experiments/__init__`.
- Two early-validation guards after two MQAR-constraint bugs surfaced in the pilot (see Findings): `N ≥ 3·kv` and `kv < vocab_size/2`, failing fast with a clear message.
- Tests: `_clustered_embeddings` correlation monotonicity + norm preservation; `_classify_regime` three-way logic incl. the mixed→comparable case.

## How I verified
- `uv run ruff check` on all touched files: **clean**.
- Config YAML round-trip of the new `regime_*` fields: **OK**.
- Related suites `test_v1_calibration.py` (8, +2 mine) + `test_config.py` + `test_calibration.py`: **22 passed** (13 s).
- Full suite: **137 passed, 1 skipped, 1 failed**. The single failure is `test_lattice.py::test_kappa_zero_reduction_bitlevel` — a **bit-level `array_equal`** test on lattice code I do not touch (not in my import graph). **Confirmed environment-only**: it *passes in the main repo's warm venv* and only fails in my freshly-`uv sync`'d worktree venv (XLA-build float-reproducibility artifact). Not a regression from this branch.
- **End-to-end pilots** (real training, `quick=False`), models under `.claude/scratch/v1-hopfield-stress/`:
  - `corr` (correlation axis) 3 caps × 3 ρ × 2 seeds — **wall 170 s**.
  - `noise` (eval-noise axis) 3 caps × 3 σ × 2 seeds — **wall 156 s**.
  - `kv16_*` (CLU's strong regime, 1 cap × 4 stress × 2 axes × 3 seeds) — ~4 min.
  - Total evidence-base compute ≈ **9 min on CPU**; no cell near the ~1 h CSF3 threshold. Figures: `{corr,noise,kv16_eval_noise,kv16_correlation}/plots/exp_v1_regime_map.png`; per-run `results/exp_v1_regime_summary.json` + `.npz`.

## Findings / results

### Main grid (kv 32→96), mean over seeds — `acc(CLU gate)` / `acc(Hopfield)`; all cells `hopfield_dominant`
**Correlation ρ:**
| cap | ρ=0.0 | ρ=0.5 | ρ=0.9 | CLU fidelity |
|---|---|---|---|---|
| N128/kv32 | 0.344 / 0.969 | 0.328 / 0.984 | 0.203 / **0.797** | 0.69–0.84 |
| N256/kv64 | 0.047 / 0.977 | 0.023 / 0.961 | 0.016 / **0.617** | ~0.36 |
| N384/kv96 | 0.036 / 0.964 | 0.021 / 0.958 | 0.016 / **0.557** | 0.30–0.47 |

**Eval noise σ:**
| cap | σ=0.0 | σ=0.3 | σ=0.6 |
|---|---|---|---|
| N128/kv32 | 0.344 / 0.969 | 0.312 / 0.953 | 0.188 / **0.750** |
| N256/kv64 | 0.047 / 0.977 | 0.031 / 0.875 | 0.047 / **0.539** |
| N384/kv96 | 0.036 / 0.964 | 0.021 / 0.844 | 0.031 / **0.469** |

### CLU-strong-regime probe (kv=16, fidelity ≈1.0 at ρ/σ=0) — 3 seeds
| stress | CLU gate | Hopfield | Δacc | CLU savings | CLU fid |
|---|---|---|---|---|---|
| σ=0.0 | 0.917 | 1.000 | −0.083 | **7.0×** | 1.00 |
| σ=0.3 | 0.812 | 0.938 | −0.125 | 4.3× | 1.00 |
| σ=0.6 | 0.500 | 0.792 | −0.292 | 1.9× | 1.00 |
| σ=0.9 | 0.208 | 0.583 | −0.375 | 1.3× | 1.00 |
| ρ=0.0 | 0.917 | 1.000 | −0.083 | 7.0× | 1.00 |
| ρ=0.5 | 0.875 | 1.000 | −0.125 | 5.2× | 0.98 |
| ρ=0.8 | 0.500 | 0.979 | −0.479 | 1.4× | 0.83 |
| ρ=0.95 | 0.104 | 0.667 | −0.562 | 1.0× | **0.44** |

### Read
1. **No crossover anywhere reachable at laptop scale** — 26/26 cells `hopfield_dominant`, on **both** accuracy and calibrated-abstention AURC. Even where CLU is a *perfect* store (kv16, fid 1.0) and stress is maximal, Hopfield's accuracy stays above CLU's.
2. **We DID find Hopfield's failure modes** — correlated keys and noisy cues each drive Hopfield from ~1.0 down to ~0.47–0.67 (noise slightly stronger at high kv: 0.469 vs 0.557 at kv96). So the "Hopfield errs" half of the trade is real and locatable.
3. **The gap does not close because CLU degrades *faster*.** Under eval-noise both fall smoothly but Hopfield leads throughout (Δacc widens −0.08 → −0.38). Under correlation Hopfield is *robust until ρ≈0.8* then drops, while CLU's storage collapses earlier (fidelity 1.0 → 0.44 by ρ=0.95). Hopfield degrades **more gracefully on both axes**, and its Platt-calibrated margin keeps AURC ≤0.22 at max stress while CLU's calibrated-gate AURC rises to 0.6–0.82 — CLU's *confidence signal* decays faster too.
4. **The compute-allocation advantage does NOT hold when the task is hard everywhere** (directly answers the task question): savings are 7×/5× only at low stress (σ≤0.3, ρ≤0.5); they collapse to ~1× as stress rises, because a uniformly-hard workload has no confidently-easy subset to skimp on. The v1-pivot "4.8×" is a *low-difficulty* phenomenon.
5. **The binding constraint is CLU memory strength, not Hopfield's robustness.** Storage fidelity collapses past kv≈32 (0.84→0.36) and under high correlation — i.e. the CLU is out-of-capacity before stress is even applied at kv≥64. A crossover requires first making the CLU competitive at these capacities (bigger embed/hidden, longer/better write objective, or a mass/curvature-aware store), *then* re-mapping. The regime map cleanly isolates this as the prerequisite.

### Honest framing paragraph (draft for the short)
> We charted the CLU-gate-vs-Hopfield trade across capacity (kv 16–96), correlated-key, and noisy-cue stress. Both stresses degrade modern Hopfield as expected — correlated patterns and noisy queries drop its recall from ~1.0 to ~0.5 — confirming its classical failure modes are reachable at laptop scale. Yet across every cell we tested Hopfield remained dominant on both accuracy and calibrated abstention: the conservative energy-memory degrades at least as fast as Hopfield (its retrieval basins are cue-noise-fragile and its storage saturates under correlated content), so the gap never inverts. The learned energy gate's compute-allocation win (up to 7× fewer steps at matched accuracy) is a **low-difficulty** phenomenon — it collapses to ~1× once the workload is uniformly hard, because there is no easy subset to ration. Practical guidance: **for associative recall at these scales, use Hopfield; the CLU energy-gate is a compute-rationing tool for a memory that is already accurate on an easy majority, not a way to beat Hopfield where Hopfield struggles.** The crossover the map fails to reach is gated by CLU memory competitiveness at high capacity — the concrete precondition any future "when to build with CLU" claim must first establish.

## Git footprint
- Worktree `../CHLU-v1-hopfield-stress`, branch **`agent/experiment-engineer/v1-hopfield-stress`** off `main @ b1782b0`. Rebase onto `main` = up-to-date (no-op). NOT pushed.
- Commits (4): `72376d3` regime driver + plotting + CLI + exports · `52a6b25` tests · `fe8bef7` MQAR `N≥3·kv` validation + default fix · `47fc168` `kv<vocab_size/2` validation + default caps kv≤96.
- Files: **edited** `chlu/config.py` (regime_* knobs in `ExperimentV1GateConfig`), `chlu/experiments/exp_v1_calibration.py` (+regime machinery), `chlu/utils/plotting.py` (+`plot_v1_regime_map` at EOF), `chlu/cli/experiment_cmd.py` (+parser/cmd/import), `chlu/experiments/__init__.py` (+export), `tests/test_v1_calibration.py` (+2 tests).
- **⚠ Overlap flag for the Hub:** the concurrent `agent/experiment-engineer/fix-pack-3` branch has uncommitted edits to `chlu/config.py` (and `exp_d_goldstone.py`) in the main working tree. My `config.py` change is an additive block at the end of `ExperimentV1GateConfig`; expect a merge overlap there. Both are pure additions — a union should resolve cleanly, but verify the `@dataclass` decorator survives (the recurring w2/w3 union-merge trap).
- Commands: `uv sync` (fresh worktree venv), `uv run ruff check`, `uv run pytest` (22 targeted green; full 137p/1s/1f env-only lattice bit-flake), 4 pilot runs. No conflicts.
- Scratch (gitignored, under `.claude/scratch/v1-hopfield-stress/`): `pilot.py`, `pilot.log`, `kv16.log`, and `{corr,noise,kv16_eval_noise,kv16_correlation}/` with figures/summaries/npz.
- Ops: hit §7.12 UF_HIDDEN `.pth` bug (`No module named 'chlu'`) on a `nohup` relaunch; worked around with `chflags nohidden … && PYTHONPATH=$PWD uv run --no-sync`. Bites background/CLI launches from a fresh worktree venv.

## Open questions / follow-ups / risks
1. **Crossover is CLU-capacity-gated, not stress-gated.** The single highest-value follow-up: make the CLU memory competitive at kv≥64 (raise `embed_dim`/`hidden_dim`/`train_epochs`, or a better write objective) so fidelity >0.9, then re-map — that is the only way a `clu_gate_advantage` cell can appear. Recommend the Hub decide whether V1 pursues this or accepts the honest "Hopfield-dominant everywhere; CLU-gate = compute rationer" framing.
2. **Memory-budget parity (task axis d) not implemented.** Hopfield stores O(kv·d) explicitly; CLU compresses into fixed θ. A matched-parameter comparison (e.g. Hopfield with a *subsampled/compressed* pattern set vs CLU at equal bytes) could be the one axis where CLU's fixed-θ compression is structurally favored — but it needs a principled parity definition. Flag for Hub scoping.
3. **CLU gate accuracy < storage fidelity everywhere** (e.g. kv32: gate 0.34 vs fid 0.84): cue-clamped relaxation retrieves far worse than the memory stores. The retrieval dynamics (clamp + governed relax) — not storage — are a large part of the accuracy deficit; worth a separate diagnostic before scaling.
4. Pilots are 2–3 seeds / 1 episode-per-cell (laptop budget); error bars exist but are wide. A publication grid should use ≥5 seeds and more episodes at the chosen band; still laptop-feasible (~5 min per axis-triplet).
5. The env-only lattice bit-level test failure (see verification) should be reproduced/dismissed by the Hub on the warm venv; it is a fragile `array_equal` test, not my change.

## Proposed handover updates (for the Hub)
- **§8/roadmap V1 (regime map, Head decision 1b):** at laptop scale the CLU-gate-vs-Hopfield trade has **no reachable crossover** — Hopfield dominant in 26/26 cells across capacity(kv16–96) × correlation(ρ≤0.95) × eval-noise(σ≤0.9). Both stresses break Hopfield (1.0→~0.5) but the CLU energy-memory degrades faster (noise-fragile basins; storage collapses under correlated patterns / past kv≈32) so the gap never inverts; Hopfield also stays better-calibrated (AURC ≤0.22 vs CLU 0.6–0.82 at max stress). **The v1-pivot compute-allocation win is low-difficulty-only** (7×@easy → ~1× when hard everywhere). The crossover is gated by CLU memory competitiveness, not by finding Hopfield's failure mode. Suggested short framing = *compute-rationing of an already-accurate conservative memory*, Hopfield dominance stated plainly (aligns with v1-pivot's read). Draft framing paragraph in this report.
- **§2/§3 (code):** new experiment `run_v1_hopfield_regime_map` + `regime_*` config block in `experiment_v1_gate` group + CLI `chlu exp-v1-regime [--quick]` + `plot_v1_regime_map`; stress knobs = `_clustered_embeddings` (correlated keys) and deployment cue-noise. Defaults preserve all prior behavior. Early guards added: MQAR `N≥3·kv` and `kv<vocab_size/2` (both were latent traps for any high-kv config).
- **§7 (merge watch):** my `chlu/config.py` addition overlaps `fix-pack-3`'s uncommitted `config.py` edit — union-merge, re-verify the `@dataclass` decorator (recurring trap).
