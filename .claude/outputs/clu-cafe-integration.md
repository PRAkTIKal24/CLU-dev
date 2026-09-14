# clu-cafe-integration — experiment-engineer report

Task + acceptance criterion: Register CLU as a `cafe-bench` model reusing `clu_scorer`'s training, and report a first real FD001 Event-Prediction h-AUROC vs HEPA's 0.918.
Status: **done** (integration + first numbers + levers), with **three load-bearing corrections to the task's premises** — see the reconciliation list.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (read first, per protocol §5).**
> 1. **CAFE's HEPA wrapper does NOT override `event_predict`** — it is `encode()`-only through the default **CoxPH** probe (`hepa_model.py` verbatim: *"all downstream tasks use the default linear probes defined in BaseModel"*). This **reverses** `scout-cafe-hepa` rec #3 and task item 1's implication that CLU should override. CLU therefore ships `encode()`-only for Event. Any doc saying "override `event_predict`, HEPA sets the precedent" must be corrected.
> 2. **The scout's proposed headline figure (per-horizon AUROC vs Δt) is not model-discriminating under the default probe.** Measured: a CoxPH risk score induces a **bit-identical sample ranking at all 125 horizons** (rank corr 1.0000, identical permutation). Per-horizon AUROC varies *only* because the labels change. The "graceful long-horizon decay" claim needs an overridden horizon-conditioned head to be testable at all.
> 3. **Task item 4 ("baselines come free") is false as written.** The CAFE checkout registers **only `hepa`, `moment`, `units`**. There is **no DeepSVDD, LSTM-AE, DeepHit or CoxPH model wrapper** in the repo; `hepa` additionally needs a checkpoint from a sibling repo (`hepa-sd/HEPA-SP`) that is not present, so **HEPA cannot be re-run locally** — 0.918 is a README-only number. Peer comparisons must be run by us or dropped.
> 4. **The 0.918-vs-0.81 discrepancy is NOT an aggregation artifact** (scout §3.4 candidate (a) is eliminated): CAFE's `evaluate_event` computes the **mean of per-horizon AUROCs**, exactly the paper's definition. Candidate (d) (placeholder) gains weight: **the repo contains no `leaderboard.json` and no `results/`** — 0.918 exists only in README prose.

---

## Answer first

CLU is registered and runs on the real harness. **FD001 h-AUROC = 0.6554 ± 0.0017** (3 seeds, default config) and **0.7168 ± 0.0015** (3 seeds, best single lever). Against HEPA that is a gap of **−0.263** (vs README 0.918) or **−0.155** (vs paper 0.81) at default, narrowing to **−0.201 / −0.093** tuned.

The uncomfortable, decision-relevant part: **a 56-dimensional raw-summary-statistics encoder scores 0.7486 through the identical probe — better than every CLU arm** — and **concatenating CLU's features onto it makes it worse (0.7420)**. On FD001, CLU's physics is currently contributing *no* information beyond the raw window. Nearly all of CLU's score comes from `q*` (the settled point, a nonlinear smoother of the sensors); the genuinely physical scalars (energy, ‖∇V‖, relaxation residual, rollout MSE) score **0.5887 alone** — near chance.

---

## What I did

- Cloned CAFE to `~/cafe-bench` (kept **outside** the repo; not vendored). Read the real `base.py` / `pipeline.py` / `event.py` / `cmapss.py` / `hepa_model.py` — this closed the scout's blocking gap (the README's `...` override signatures).
- **`chlu/eval/cafe_model.py`** (new): `CLUCafeMixin.encode((N,T,C)) -> (N,D)` reusing `clu_scorer._SharedCLUFit` (unchanged) as the training path. Two registered identities: `clu` (all default probes) and `clu_valley` (valley-aware `anomaly_score` override). `chlu` never hard-depends on `cafe_bench`: the mixins are composed with the harness `BaseModel` inside `register()`.
- **`chlu/eval/config.py`**: added `CLUCafeEncodeConfig` + `CAFE_FEATURE_GROUPS` / `CAFE_ANOMALY_MODES`. Encode-side `relax_gamma`/`relax_steps` overrides so CAFE can fix its damping **without touching the shared anomaly-scorer defaults** (which are live on the voraus path).
- **`scripts/cafe/run_clu_cafe.py`** (new): the entry point; `--quick`, `--subsample`, full config provenance.
- **`scripts/csf3/job_gpu_cafe.sh`** (new): CSF3 wrapper to current conventions (`-n 1 -c 8`, separate `-e`, `logs/`, `$CLU_MAIL`, GPU + `lifelines`/`cafe_bench` preflight, download-once note, per-(dataset,seed) sweep recipe).
- **`tests/test_eval_cafe_model.py`** (new, 13 tests): contract tests against a local stand-in for `BaseModel`, so the suite runs with **no CAFE checkout**.

## How I verified

```
uv/.venv pytest tests/test_eval_cafe_model.py -q   -> 13 passed
full suite                                          -> 341 passed, 0 failed (exit 0)
                                                       (= 328 baseline + my 13)
ruff check <all touched files>                      -> All checks passed
bash -n scripts/csf3/job_gpu_cafe.sh                -> OK
runner reproducibility after post-smoke edits       -> --quick 0.6693, bit-identical to pre-edit
```
My own test caught a real bug pre-commit: `feature_groups=("basin_coords",)` alone hit `jnp.stack([])`. Fixed + regression-tested.

### Data provenance (needed — CAFE's own path is broken)
CAFE's `scripts/download_all.py` C-MAPSS route is **dead**: both `ti.arc.nasa.gov/c/6/` and `data.nasa.gov/download/ff5v-kuh6/...` return **404** (verified; network to NASA/HF/GitHub is otherwise fine). I used the repo's own HF fallback source (`LucasThil/nasa_turbofan_degradation_FD001`) and converted to CAFE's txt format. Sanity: **20631 train rows / 100 units, 13096 test rows / 100 units** = canonical FD001. Loader yields **X_train (17731,30,14), X_test (10196,30,14)**.

> ⚠ **C-MAPSS label caveat (benchmark-design, affects the absolute number for every model equally).** `cmapss.py` sets `e=1` for all windows and `t = max_cycle - i`, i.e. **cycles remaining in the recording**, ignoring the true-RUL file. The official test sequences are truncated *before* failure, so this is not true RUL: e.g. test unit 1 has 31 recorded cycles with **true RUL 112 at its last cycle**, which CAFE labels `t=0`. Comparisons within CAFE stay internally fair; the number is **not** comparable to any externally-published C-MAPSS RUL result, and this is a live candidate for the 0.918-vs-0.81 gap.

---

## Findings / results

### Headline (all through CAFE's own loader → probe → `evaluate_event`)

| arm | h-AUROC | vs HEPA 0.918 | vs HEPA-paper 0.81 |
|---|---|---|---|
| **CLU `clu`, default config** (3 seeds) | **0.6554 ± 0.0017** | **−0.263** | −0.155 |
| **CLU `clu`, relax_budget=1.6** (3 seeds) | **0.7168 ± 0.0015** | **−0.201** | −0.093 |
| CLU, `basin_coords` only, budget 1.6 (s42) | 0.7230 | −0.195 | −0.087 |
| *reference:* raw_stats 56-d, same probe | **0.7486** | −0.169 | −0.061 |
| *reference:* raw_last 14-d | 0.7203 | | |
| *reference:* raw_mean 14-d | 0.7068 | | |
| HEPA (CAFE README, **not reproducible here**) | 0.918 | — | |
| HEPA (paper Table 1) | 0.81 ± .03 | | — |

Per-seed: default 0.6540 / 0.6577 / 0.6544; tuned 0.7158 / 0.7189 / 0.7156.

### Where CLU's signal actually lives (univariate h-AUROC, default config)
`q_star_6` 0.613 · `gradV_mean` 0.605 · `q_star_7` 0.601 · `V_trend` 0.596 · … then a floor: `relax_residual` **0.517**, `energy_mean` **0.518**, `energy_trend` **0.509**, `predict_mse` **0.510**, `energy_last` **0.501**, `K_last` **0.501**.
**The basin-exit scalars the task asked for are at chance.** The signal is in the settled *coordinates*, not the energetics.

### Ablations (real probe, train-first ordering)
| config | h-AUROC |
|---|---|
| ALL features, budget 0.16 (default) | 0.6540 |
| ALL features, budget 1.60 | **0.7158** |
| **PHYS scalars only** (no `q*`), budget 1.60 | **0.5887** |
| `basin_coords` only, budget 1.60 | **0.7230** |
| ALL, budget 1.60, **epochs 150→600** | 0.7147 |
| raw_stats ⊕ CLU-ALL | 0.7420 |
| raw_stats ⊕ CLU-PHYS | 0.7423 |
| raw_stats alone | **0.7486** |

Two hard reads: **(a) dropping the physics scalars *improves* CLU** (0.7230 > 0.7158) — they are currently probe noise; **(b) CLU is not additive with raw stats** (0.7486 → 0.742) — it contributes nothing a linear probe can use.

### ⭐ Mechanism finding: the relaxation budget, and a measured single-basin collapse
The damping budget `γ·steps·dt` — not γ or steps alone — controls settling. The inherited default is **0.16**, which damps only **15%** of the initial velocity: `q*` is a *free-streaming continuation*, not a settled point. Measured directly: `q*` cross-sample spread is **2.55× larger** than `q_last` at the default (the opposite of relaxation).

| budget γT | `q*` spread | `relax_residual` univariate | h-AUROC |
|---|---|---|---|
| 0.16 (default) | 1.636 | 0.5172 | 0.6540 |
| **1.60** | 0.754 | 0.5464 | **0.7158** |
| 12.8 | 0.842 | 0.5455 | 0.7075 |
| 64.0 | **0.000** | 0.5000 | **CoxPH ConvergenceError (singular)** |

At γT=64 **every window settles onto the same point — cross-sample std exactly 0.000**, the embedding goes rank-deficient and the probe fails. This is the **anti-collapse phenomenon made quantitative on a real benchmark**: the learned potential has effectively **one basin**, so complete relaxation is information-destroying, and the useful regime is a narrow window of *incomplete* relaxation. This is direct empirical input for `anti-collapse-characterization` (and it is a *structural/dynamical* collapse — mode-band, not representation — which is exactly the narrowing the scout said that thread needs).

### Top-3 levers to close the gap (ranked by measured evidence)
1. **Relaxation budget γ·steps·dt** — the only large confirmed lever: **+0.061** (0.6554 → 0.7168). Optimum near 1.6; ≥60 is catastrophic. *Cheap, already exposed on the CLI.*
2. **Encode-feature choice** — **+0.007** by *deleting* the physics scalars (`basin_coords` only, 0.7230). Equivalently: the energetics need to be made informative before they are worth including.
3. **The training objective — NOT training length.** Epochs 150→600 is a **null lever** (0.7158 → 0.7147, within seed noise), which **falsifies "training length" as a top-3 lever** from the task file. The HCD objective is *degradation-agnostic*: it learns "is this a plausible C-MAPSS window", never "how far along is this engine", so energy has no reason to be monotone in wear. Closing a 0.20 gap needs either (a) anti-collapse / multi-basin structure so basin identity means something, or (b) horizon-conditioning the representation (HEPA's actual mechanism). Everything else is second-order.

**Honest CM-3 framing:** the headline is not "CLU trails HEPA" (expected, fine) — it is **"CLU currently trails a 56-d raw-statistics baseline, and its physics features are at chance."** The encoder is not yet extracting degradation structure on this dataset. Reporting the tuned 0.7168 without the 0.7486 raw-stats reference next to it would be misleading.

---

## Git footprint
- Branch **`agent/experiment-engineer/clu-cafe-integration`** (off local `main` @ `a5978f6`), rebased onto `main` (no-op, base unmoved). **Not pushed.** Working tree clean.
- Commits: **`5043362`** (register CLU as a cafe-bench model), **`b3ef665`** (expose relaxation-budget lever on the runner CLI).
- Files: `chlu/eval/cafe_model.py` (new), `chlu/eval/config.py` (**additive only** — new dataclass + 2 constants; no existing default changed), `scripts/cafe/run_clu_cafe.py` (new), `scripts/csf3/job_gpu_cafe.sh` (new), `tests/test_eval_cafe_model.py` (new). **`clu_scorer.py` untouched.**
- Not formatted with `ruff format`: the repo is not ruff-format-clean (`chlu_unit.py`, `clu_scorer.py`, `harness.py` all fail `--check`), so reformatting would have been out-of-scope churn. `ruff check` passes.

### ⚠ Environment change I made (flag for the Hub)
`lifelines` (backs CAFE's default CoxPH event probe) was **missing** from `.venv`. Installing it **downgraded pandas 3.0.3 → 2.3.3** in the venv. The full suite is **341 green** after the downgrade, so nothing broke, but `uv.lock` is untouched and this is not reproducible from the lock — someone should decide whether `lifelines` becomes a `cafe` extra in `pyproject.toml`. (Also note `uv pip install` without `--python .venv/bin/python` silently targets the *miniconda* env on this machine.)

### Flag provenance
| item | value |
|---|---|
| commit | `b3ef665` (results identical at `5043362`) |
| seeds | 42, 43, 44 |
| dataset | `cmapss_fd001`, CAFE loader, window 30, C=14, horizons 1…125 |
| model | `clu`, `encode()`-only, **default CoxPH probe** (`penalizer=0.1`) |
| CLU config | `kinetic_mode=newtonian_learned`, `potential_type=mlp`, `hidden=64`, `dt=0.05`, `gamma=0.1`, `epochs=150`, `lr=1e-3`, `batch_size=64`, `max_fit_windows=4000`, `predict_horizon=16`, `relax_steps=32`, `neg_noise_scale=0.5`, `energy_reg=0.005`, `momentum_init=finite_diff`, no lattice |
| encode config | all 7 feature groups, `standardize=True`, `batch_size=512`; **default arm** `relax_budget=0.16` (inherited); **tuned arm** `relax_gamma=0.5, relax_steps=64` → `relax_budget=1.60` |
| embedding | D=30 (16 scalars + 14 `q*` coords) |
| env | JAX **0.9.0**, equinox 0.13.4, main `.venv` (no worktree sync), lifelines 0.30.3, pandas 2.3.3 (downgraded), CPU |
| CAFE | `~/cafe-bench` @ `dc3dbd0` |

No PREREG was written: the acceptance criterion is a *first measurement of an existing quantity*, not a predicted ratio/exponent/law. The lever ablations were exploratory and are reported as such.

---

## Open questions / follow-ups / risks
1. **The 0.918 target is still unverified and now looks shakier** (no `leaderboard.json`, no `results/`, aggregation matches the paper so it cannot explain the gap, HEPA not locally runnable). The Head is a HEPA co-author — one question settles it. **Do not lock an ICLR headline target until then.**
2. **FD002/FD004 not yet run.** The scout argues these multi-regime cells are the most winnable *and* the most mechanism-diagnostic (multi-regime = multi-basin). Given the measured single-basin collapse, that is now a **sharp, falsifiable prediction**: if CLU's potential really is one basin, it should do *badly* on FD002/FD004 too until anti-collapse lands. Worth pre-registering. (Data: only FD001 was downloaded; the other three need the same HF conversion.)
3. **Anomaly track is wired but unmeasured** — `clu_valley` is implemented and tested, but no CAFE anomaly dataset was downloaded. `anomaly_mode ∈ {valley, valley_predict, energy, predict}`.
4. **Fit-order footgun:** `encode` lazily fits on first call. The harness always encodes train first (verified in all three probes), but *analysis scripts* can silently train on test — this contaminated one of my own diagnostics before I caught it. Mitigated with an explicit `fit()` + a regression test, but reviewers of any future analysis script should check ordering.
5. Not done: PhysioNet event sets, classification track, the `event_predict` override variant (deliberately — see reconciliation #1).

## Proposed handover updates (for the Hub)
1. **Correct the override guidance everywhere** (§2026-07-20 PIVOT, `scout-cafe-hepa` rec #3, `clu-cafe-integration` item 1): CAFE's HEPA is **`encode()`-only through the default CoxPH probe**. CLU matches it. An `event_predict` override is a *separate, non-comparable* leaderboard identity.
2. **Retire the "per-horizon AUROC vs Δt" headline-figure proposal** (scout update #5) *as stated* — under the default probe the ranking is provably horizon-invariant (measured identical at h=1 vs h=125). It only becomes a real figure with an overridden horizon-conditioned head.
3. **First CLU-vs-HEPA number, for the record:** FD001 `h-AUROC 0.6554 ± 0.0017` (default) / `0.7168 ± 0.0015` (relax_budget=1.6), vs HEPA README 0.918 / paper 0.81 — **and vs a 56-d raw-statistics baseline at 0.7486 that beats both CLU arms.** The raw-stats reference should travel with the number.
4. **New matrix-worthy finding (anti-collapse thread):** on real C-MAPSS the learned potential is effectively **single-basin** — full relaxation drives the cross-sample spread of the settled point to **exactly 0.000** and makes the linear probe singular. Useful behaviour lives in a narrow band of *incomplete* relaxation (γT≈1.6). Hand this to `anti-collapse-characterization` as measured evidence.
5. **"Baselines come free" is wrong** — CAFE registers only `hepa`/`moment`/`units`, and `hepa` needs an absent checkpoint. Any peer comparison must be run by us. Update the backlog item that deprecated "add parametric baselines to chlu/eval".
6. **CAFE's C-MAPSS download path is dead** (NASA 404 ×2); use the HF route. And **CAFE's C-MAPSS test labels are cycles-remaining-in-recording, not true RUL** — flag before any external comparison.
7. **`lifelines` dependency decision owed** (see env note): make it a `cafe` extra or leave venv-local. Note the pandas 3.0.3→2.3.3 downgrade; suite stayed 341-green.
