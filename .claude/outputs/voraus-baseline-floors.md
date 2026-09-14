# voraus-baseline-floors — results-analyst report
Task + acceptance criterion: pre-register + measure the statistical-baseline floors (VUS-PR/AUROC, no point-adjust, unit-level, per-category) on voraus-AD & TEP-Rieth under the binding F2 harness, size the CSF3 envelope for the CLU run. CSF-prep (full floors run on CSF3).
Status: **partial (CSF-prep complete; laptop de-risk of voraus episode-mode; full floors pending CSF3)**

**Downstream reconciliation list (owner needed — see §First-line flag):**
1. **Metric label fix:** task/handover call voraus per-category "VUS-PR"; voraus is *episode*-labelled ⇒ the correct primary is **episode AUC-ROC** (harness `PRIMARY_METRIC["episode"]="AUC-ROC"`). Every downstream "voraus VUS-PR" mention must say AUROC.
2. **CSF3 env gap:** `scripts/csf3/setup_env_job.sh` syncs `--extra cuda` only; the loaders need `--extra eval` (pandas/pyarrow/pyreadr). Must be fixed before the CLU-on-voraus/TEP run — flagged for `experiment-engineer`.

## What I did
- Read protocol (§5 pre-reg + flag-provenance), handover, f2-eval-harness report, harness/baselines/loader/metrics code, CSF3 runbook.
- **Characterized voraus-AD & TEP-Rieth directly from the parquet/RData** (isolated `uv --with pandas/pyreadr --no-project` env — **no `chlu` import, no 20-min JAX cold start**) to ground the PREREG and the CSF3 sizing in real counts.
- **Pre-registered** predicted rankings, AUROC/VUS-PR bands, per-category structure, and the compute envelope in `PREREG.md` **before** any harness run.
- Wrote two fit-once/score-all drivers (`voraus_floor.py`, `tep_floor.py`) using only public `chlu.eval`/`chlu.data` APIs — faithfully replicate the harness code path and add the **per-category / per-fault** maps the harness does not expose.
- Ran a **laptop de-risk pre-smoke** of the (never-before-run-on-real-data) voraus **episode-mode** path (reduced channels/stride to fit 16 GB RAM).
- Wrote CSF3 jobscripts + run README (`csf3/`) with the measured resource envelope.

## Setup (configs / seeds / commands)
- Repo `main` @ commit **`df5e44d`** (wave-14 integration), tree clean. Env: main `.venv` after `uv sync --extra eval` → **jax 0.9.0 (unchanged, = CSF3 pin), pandas 3.0.3, pyarrow 24.0.0, sklearn 1.8.0**. macOS, 16 GB RAM. `chlu` not pip-installed in `.venv`; ran drivers with `PYTHONPATH=<repo> .venv/bin/python`.
- Data: local verified copies under `.claude/scratch/f2-eval-harness/data/` — voraus `voraus-ad-dataset-100hz.parquet` sha256 `c90ab1c7…` (1.04 GiB); TEP fault-free RData (faulty-testing NOT local → CSF3).
- Metadata characterization: `uv run --with pandas --with pyarrow --no-project python /tmp/voraus_meta.py`; `uv run --with pyreadr --no-project python /tmp/tep_meta.py`.
- Pre-smoke command:
  `PYTHONPATH=<repo> .venv/bin/python .claude/scratch/voraus-baseline-floors/voraus_floor.py --root .../voraus_ad --columns-limit 24 --train-stride 30 --max-train-windows 30000 --test-stride 5 --out .../presmoke` (seed 42, window 100).

## Data characterization (measured, JAX-free)
**voraus-AD 100 Hz** (137 cols = 7 meta + **130 machine signals**; 2 321 690 rows):
- 2122 episodes; train (setting=PRE_A=72) = **948 normal**; test = **1174** (**755 anomalous + 419 normal**, cat 12).
- Per-category **test** episode counts: {AXIS_FRICTION:144, AXIS_WEIGHT:156, COLLISION_FOAM:72, COLLISION_CABLE:48, COLLISION_CARTON:22, MISS_CAN:11, LOSE_CAN:74, CAN_WEIGHT:80, ENTANGLED:10, INVALID_POSITION:12, MOTOR_COMMUTATION:89, WOBBLING_STATION:37}; NORMAL pool 419.
- Episode length min/median/max = **986 / 1096 / 1164** (~11 s @100 Hz). Meta-col read 1.65 s.

**TEP-Rieth** (52 signals; 55 cols):
- fault-free-training = **500 runs × 500 samples** (all normal → the train set); fault-free-testing = 500 × 960 (all normal → **degenerate**, harness must NaN); faulty-testing (CSF3) = **20 faults × 500 runs × 960 = 10 000 faulty test units**; anomalous fraction per faulty run = 800/960 = **0.83** (onset sample 160).
- pyreadr read: fault-free-training 1.0 s / fault-free-testing 2.5 s (fault-free only ≈ 320 MB RAM).

## Resource / memory envelope for CSF3 (the CLU-run sizing)
| dataset | test units | train windows (chosen) | flat dim | peak RAM (pred.) | cores | wall (pred.) |
|---|---|---|---|---|---|---|
| voraus-AD | 1174 episodes | ~95 k @ train_stride=10 | 100×130 = 13 000 | ~20 GB (PCA U≈10 GB) | `-n 16` | **2–5 h** (test_stride=5) |
| TEP-Rieth | 10 000 faulty (+500 degenerate) | ~100 k @ train_stride=2 | 100×52 = 5 200 | ~8–12 GB (pyreadr 4 GB) | `-n 24` | 1–6 h |
- **Binding constraint (voraus):** train_stride=1 concat = ~943 k × 13 000 float32 = **~49 GB peak** before the 100 k subsample → OOM. `train_stride=10` fixes it (~95 k windows, ~5 GB). PCA full-SVD is then the RAM bottleneck (~10 GB U). CPU-only; run on `multicore`, not `gpuA`.
- **Wall-time driver (measured, voraus):** KNN + LOF **scoring** dominate — 179 s *each* for 1174 episodes at only 24 ch / test_stride=5. Full run scales ×~5.4 (130 ch) ⇒ recommend **test_stride=5** (episode mean-reduce is near-insensitive to it — measured — for ~5× speed, ~lossless) and `-t 12:00:00`. At test_stride=1 budget ×5 again.
- TEP: 10 000 test units × KNN/LOF is the wall-time driver (the budget stressor the task flags). `--metrics-mode fast` to bound per-unit cost.
- CSF3 recipe, jobscripts, and the **env-extra gap fix** are in `.claude/outputs/voraus-baseline-floors/csf3/`.

## Pre-registered predictions (full text in `PREREG.md`)
- voraus overall AUROC **0.60–0.85**; ranking **knn≈lof≥pca>iforest** (competing hyp: at 13 000-dim, **pca could lead** — a finding, not a bug). Per-category: **collisions(2/3/4) & commutation(10) high (0.8–0.98) > weight/friction > can-handling/position/wobble**. Shuffled-label control ≈0.50.
- TEP overall VUS-PR **0.55–0.85**; **faults 3/9/15 near-undetectable** (VUS-PR ≈ base rate); ranking knn≈lof≥pca>iforest. Fault-free-testing must NaN (degenerate control).
- Neither dataset is a designated negative control (SMD was); both are positive benchmarks CLU should do *well* on.

## Findings / results
### voraus-AD laptop de-risk pre-smoke (PIPELINE VALIDATION ONLY — reduced config, NOT the floor)
Config: **24-channel subset** (first 24 signals), window=100, train_stride=30 (→30 000 train windows, dim=2400), test_stride=5, seed=42, episode_reduce=mean, all 1174 test episodes. JAX import here was ~2–3 min (warm disk), **not** the 20-min worst case. Fits `voraus_floor.{json,md}` under `.claude/outputs/voraus-baseline-floors/presmoke/`.

**Overall (episode AUROC primary):**
| method | AUC-ROC | AUC-PR | shuffled-label control (AUROC) |
|---|---|---|---|
| pca_recon | 0.528 | 0.687 | 0.514 |
| iforest | 0.628 | 0.769 | 0.489 |
| lof | 0.749 | 0.853 | 0.515 |
| knn | **0.772** | 0.872 | 0.496 |

**Per-category AUROC (this category vs the 419-episode normal pool; best = knn/lof):**
| category | n_anom | knn | lof | iforest | pca |
|---|---|---|---|---|---|
| AXIS_FRICTION | 144 | **0.992** | 0.976 | 0.814 | 0.340 |
| ENTANGLED | 10 | **0.991** | 0.961 | 0.884 | 0.502 |
| CAN_WEIGHT | 80 | **0.874** | 0.803 | 0.631 | 0.392 |
| COLLISION_FOAM | 72 | **0.847** | 0.825 | 0.769 | 0.426 |
| MISS_CAN | 11 | 0.754 | 0.601 | **0.846** | 0.775 |
| MOTOR_COMMUTATION | 89 | 0.708 | **0.719** | 0.642 | 0.650 |
| COLLISION_CABLE | 48 | **0.726** | 0.712 | 0.603 | 0.565 |
| INVALID_POSITION | 12 | 0.718 | **0.741** | 0.484 | 0.597 |
| WOBBLING_STATION | 37 | 0.621 | 0.606 | 0.575 | **0.689** |
| AXIS_WEIGHT | 156 | **0.686** | 0.672 | 0.509 | 0.677 |
| COLLISION_CARTON | 22 | 0.580 | 0.575 | 0.520 | **0.618** |
| LOSE_CAN | 74 | 0.563 | 0.519 | 0.394 | 0.503 |

**Timing (24 ch):** fit pca 5.4 s / iforest 1.0 s / lof 15.0 s / knn 0.02 s; **score (1174 episodes) lof 179 s, knn 179 s** (dominant), iforest 5.4 s, pca 1.6 s. Total ~6.5 min compute.

**Interpretation (vs PREREG):**
- **Pipeline de-risked:** the voraus **episode-mode harness path + per-category driver ran end-to-end on real data for the first time** — no NaN/crash, all metrics finite. The **shuffled-label negative control lands at 0.49–0.51 for all four baselines** (P1-neg PREREG ✓ — metric wiring unbiased).
- **Ranking:** knn(0.772) > lof(0.749) > iforest(0.628) > pca(0.528). PREREG had knn≈lof top (✓) but predicted pca≥iforest; **pca_recon is the *weakest* here** and even sub-chance on several categories (AXIS_FRICTION 0.340, CAN_WEIGHT 0.392, COLLISION_FOAM 0.426) — the anomalies fall *inside* the retained-variance subspace, so recon error is *lower* than normal. This is a genuine PCA-recon weakness on this subset (control passes, so not a bug), and the reason the SKAB pca prior (0.580) does **not** transfer.
- **Per-category structure:** friction/entangled/can-weight/collision-foam are the most separable (knn 0.85–0.99); LOSE_CAN, COLLISION_CARTON, WOBBLING the hardest (knn 0.56–0.62) — broadly the "geometric excursions easier than subtle control faults" story, though **AXIS_FRICTION topping the list and MOTOR_COMMUTATION only moderate are subset artefacts** (the first-24-channel projection likely omits the specific per-axis current channels that make commutation obvious). The full 130-channel CSF3 floor should shift several categories **up**.
- **No surprise-trigger fired:** no category < 0.45 for the best baseline; best-baseline overall 0.772 sits inside the predicted 0.60–0.85 band even on a crippled 24-channel view. Loader/split/label look healthy.

### Full floors
Pending CSF3 (voraus + TEP). Drivers + jobscripts ready; PREREG bands committed.

## Literature sanity check
voraus-AD paper (Brockmann et al., IEEE T-RO 2024, arXiv:2311.04765) headline method **MVT-Flow** reports high detection AUROC (~0.9 mean). Our statistical-baseline floor is expected **below the flow, above chance** (0.6–0.85). A precise per-category cross-check needs their results Table — flagged as a follow-up; a best-baseline overall AUROC outside ~[0.6,0.9] (esp. <0.6 or ≈1.0) would indicate a loader/label/split bug and must be investigated, not reported.

## Limitations / confounds
- Laptop pre-smoke uses a **24-channel subset + train_stride 30 + test_stride 5** to fit 16 GB — numbers are indicative of *pipeline correctness*, not the floor. The real floor (all 130 channels, train_stride 10, test_stride 1) runs on CSF3.
- Per-category AUROC uses **this-category vs the 419-episode normal pool** (not vs all-other-anomalies) — matches "detect anomaly of type k against normal". Small-n categories (MISS_CAN 11, ENTANGLED 10, INVALID_POSITION 12) have wide CIs.
- StandardScaler + flattened windows follow the harness exactly; window=100 (1 s @100 Hz) is a deliberate, recorded choice — a window-sensitivity pass is cheap follow-up.

## Flag-provenance (this report)
| item | commit | seed | window | train_stride | test_stride | max_train_win | channels | metrics_mode |
|---|---|---|---|---|---|---|---|---|
| data characterization | df5e44d | — | — | — | — | — | 130/52 | — |
| voraus pre-smoke | df5e44d | 42 | 100 | 30 | 5 | 30 000 | 24 (subset) | episode AUROC/AUPR |
| voraus CSF3 (planned) | df5e44d | 42 | 100 | 10 | 5 | 100 000 | 130 (all) | episode AUROC/AUPR |
| TEP CSF3 (planned) | df5e44d | 42 | 100 | 2 | 1 | 100 000 | 52 (all) | fast (VUS-PR/ROC/AUC-PR/ROC) |

## Open questions / follow-ups / risks
- **CSF3 env must add `--extra eval`** (flag for experiment-engineer) — loaders fail otherwise.
- If voraus PCA full-SVD OOMs at 95 k×13 000, drop `--max-train-windows` to 40 k or `--window` to 50 (dim 6 500) — report which.
- Confirm `import chlu` works under the CSF3 `.venv` (locally needed `PYTHONPATH`).
- Per-category literature cross-check against the voraus paper Table is outstanding.
- **No git footprint** (all artifacts under `.claude/`, no tracked code touched). Observed a **pre-existing uncommitted change `M tests/test_langevin_fdt.py`** present at session start — **not mine**, left untouched per protocol §3.2; flagging so the Hub knows another agent's work may be parked in the checkout.
- Observed JAX import cost here was ~2–3 min (warm disk), not the 20-min worst case — cold-start budget may be over-stated when the OS file cache is warm.

## Proposed handover updates (for the Hub)
- **§1.6 / §5 (experiments & provenance) — voraus-AD is EPISODE-labelled → primary metric is episode AUC-ROC, not VUS-PR.** Correct any "voraus VUS-PR" wording in tasks/paper. Per-category = this-category-vs-normal-pool AUROC.
- **§1.6 baseline floors — voraus-AD data facts (measured, JAX-free):** 948 normal train (PRE_A) / 1174 test (755 anom + 419 normal); 130 signals; 12 anomaly categories with test counts {friction144, weight156, coll_foam72, coll_cable48, coll_carton22, miss_can11, lose_can74, can_weight80, entangled10, invalid_pos12, commutation89, wobbling37}; episodes ~1096 samples (~11 s @100 Hz).
- **§1.6 — voraus laptop pre-smoke floor (INDICATIVE, 24-ch subset, NOT the headline floor):** episode AUROC knn 0.772 / lof 0.749 / iforest 0.628 / pca 0.528; shuffled control ≈0.50 (passes). **PCA-recon is weak/sub-chance on several categories** on voraus — the SKAB pca prior does not transfer. Full 130-ch floor pending CSF3.
- **§1.6 — TEP-Rieth data facts:** train = 500 fault-free-training runs; test = 10 000 faulty-testing runs (+500 degenerate fault-free-testing → NaN); 52 ch; anomaly base rate 0.83/run. Faulty-training (471 MB) not needed.
- **§8 / runbook — CSF3 env gap:** `setup_env_job.sh` must sync `--extra eval` (pandas/pyarrow/pyreadr) or industrial loaders fail. Adding it keeps jax pinned at 0.9.0. **Flag for `experiment-engineer`.** Also confirm `import chlu` under the CSF3 `.venv` (locally needed `PYTHONPATH=<repo>`).
- **§8 — CSF3 sizing for the CLU-vs-baseline run:** voraus `-p multicore -n16 -t12:00:00`, train_stride=10 (avoids ~49 GB OOM), test_stride=5 (KNN/LOF scoring is the wall driver); TEP `-n24 -t1-0`. Jobscripts + README at `.claude/outputs/voraus-baseline-floors/csf3/`.
- **Reusable drivers** `voraus_floor.py` / `tep_floor.py` (per-category/per-fault maps via public `chlu.eval` API) at `.claude/scratch/voraus-baseline-floors/` — rsync with the repo for CSF3.
- **These floors are the reference line for every CLU-vs-baseline voraus/TEP claim** in the ICLR long. The voraus headline floor (best statistical baseline, full 130 ch) is the number CLU must beat/Pareto-match; per-category map tells us where the "geometric prior" should help most (LOSE_CAN, COLLISION_CARTON, WOBBLING are the hard categories where baselines are weakest — knn 0.56–0.62 even in the subset).
