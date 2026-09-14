# f2-eval-harness — experiment-engineer report

Task + acceptance criterion: build the dataset-agnostic industrial eval harness (loaders voraus-AD/SKAB/TEP-Rieth/SMD-via-TSB-AD + TSB-AD-wrapped metrics with VUS-PR primary & point-adjust banned + 4 mandatory statistical baselines + leakage-safe unit splits); accepted when all 4 baselines run end-to-end on SKAB with real VUS-PR numbers + pytest for splits/metrics incl. a VUS-PR check against TSB-AD's own output.
Status: **done** (incl. stretch MIMII skeleton; SMD negative-control harness run bonus — see below)

## What I did
- **New packages** (all new files; no shared `chlu/` file touched):
  - `chlu/eval/` — `metrics.py` (TSB-AD wrapper: point + episode modes, forbidden-metric guard), `splits.py` (unit_split / cross_condition_split / assert_no_unit_leakage), `baselines.py` (PCA-recon, IForest, LOF, KNN + explicit windowing & overlap-average score alignment), `harness.py` (`evaluate_dataset` → `EvalRunResult` with per-unit metric tensor, `results/eval_<dataset>.npz` writer, markdown table emitter, `load_eval_npz`), `config.py` (`EvalConfig`/`WindowConfig`, self-contained — deliberately NOT in `chlu/config.py` to avoid colliding with concurrent fix-pack-2), `_tsb_vendor/` (pinned TSB-AD v1.5 evaluation code, see Deviation).
  - `chlu/data/industrial/` — `base.py` (`IndustrialDataset` ABC + `UnitRecord` + checksum download helpers), `skab.py`, `voraus_ad.py`, `tep_rieth.py`, `smd_tsb.py`, `mimii.py` (stretch skeleton), `__init__.py` (lazy registry + `get_dataset`).
- **pyproject**: `[project.optional-dependencies] eval = [pandas>=2.0, pyarrow>=15.0, pyreadr>=0.5.0]` (loaders only — metrics need NO extra) + `[tool.ruff] extend-exclude` for the vendor dir. `chlu/data/__init__.py` untouched (subpackage import works without it).
- **Tests**: 41 new (`test_eval_metrics.py`, `test_eval_splits.py`, `test_eval_baselines.py`, `test_industrial_loaders.py`) — all offline/synthetic-fixture; loader tests `importorskip` pandas so base envs stay green.
- **Verification runs on real data** (downloads + runs on this laptop, artifacts in `.claude/scratch/f2-eval-harness/`).

## ⚠ Deviation from task letter (flag for Hub): TSB-AD is VENDORED, not pip-installed
The task says "wrap the Apache-2.0 TSB-AD harness". The `TSB-AD` PyPI distribution pins **`numpy<2.0`** and requires torch/transformers/tslearn/stumpy. Our uv lock resolves **numpy 2.x** for the JAX stack, and uv builds ONE universal resolution including extras ⇒ adding TSB-AD to pyproject (even as an extra) would force a **project-wide numpy downgrade** and perturb every concurrent agent's env. The evaluation subtree itself needs only numpy+sklearn+stdlib, so I vendored exactly that subtree, **pinned & checksum-verified** (sdist sha256 `52e474cd…`, v1.5), byte-verbatim except documented modifications (`chlu/eval/_tsb_vendor/README.md`; Apache-2.0 LICENSE retained). VUS-PR is still 100% TSB-AD's own code — not reimplemented. **Proof of equivalence:** `test_vendored_matches_upstream_tsb_ad_reference` pins wrapper output against the REAL `TSB-AD==1.5` package (run in an isolated `uv run --with TSB-AD --no-project` env on bit-identical pure-python-LCG input): all 8 metrics agree to <1e-10.

**Point-adjust ban implemented physically:** `metric_PointF1PA`/`_adjust_predicts`/`metric_new`/`PA-F1` are excised from the vendored source (a test asserts the strings don't exist in the tree); `assert_metric_allowed()` rejects every PA alias at the wrapper. Kept `metric_EventF1PA` — despite the upstream name it is event-wise recall (one credit/event) × point precision = the legitimate "Event-based-F1" from the TSB-AD paper, not point-adjust. **Bonus evidence for the ban:** on the reference case upstream reports **PA-F1 = 1.0 (perfect!) where AUC-ROC = 0.727** — the Kim-et-al inflation pathology reproduced in one line.

## How I verified (commands + real numbers)
- `uv run pytest -q` (worktree): **59 passed** (18 pre-existing + 41 new), 3 warnings (sklearn UndefinedMetricWarning from upstream's own threshold sweep — present with real TSB-AD too). `uv run ruff check .`: **All checks passed** (vendor dir excluded). `ruff format` applied to my files.
- Post-extra lock check: numpy stays 2.x (`uv.lock` is **gitignored** — no cross-agent lock surface; worktree resolved numpy 2.4.6/pandas 3.0.3/pyarrow 24.0.0/pyreadr 0.5.6).
- **SKAB end-to-end (acceptance run)** — `SKAB(root=fresh, download=True)` exercised the real pinned fetch (commit `b2c0d46c`, zip sha256 `45ac11b4…`, 5.2 MB, 1.3 s), 35 units (1 train / 34 test), 9306 train windows (size=100, stride=1), full metric suite, seed 42; **eval wall time 77.7 s** (fit 0.003–0.57 s; score 16.5–20.6 s/method incl. per-unit VUS):

| method | **VUS-PR** | VUS-ROC | AUC-PR | AUC-ROC | Standard-F1 | Event-based-F1 | R-based-F1 | Affiliation-F |
|---|---|---|---|---|---|---|---|---|
| pca_recon | 0.580 ± 0.236 | 0.673 ± 0.224 | 0.520 ± 0.221 | 0.626 ± 0.228 | 0.664 ± 0.135 | 0.804 ± 0.183 | 0.614 ± 0.159 | 0.865 ± 0.086 |
| iforest | 0.522 ± 0.258 | 0.613 ± 0.257 | 0.483 ± 0.251 | 0.573 ± 0.266 | 0.640 ± 0.154 | 0.750 ± 0.197 | 0.601 ± 0.168 | 0.834 ± 0.101 |
| lof | 0.868 ± 0.182 | 0.912 ± 0.122 | 0.847 ± 0.203 | 0.897 ± 0.143 | 0.852 ± 0.140 | 0.958 ± 0.108 | 0.858 ± 0.145 | 0.963 ± 0.062 |
| knn | **0.873 ± 0.184** | 0.916 ± 0.123 | 0.853 ± 0.205 | 0.902 ± 0.144 | 0.861 ± 0.144 | 0.958 ± 0.107 | 0.867 ± 0.149 | 0.964 ± 0.063 |

  (all n=34; npz at `.claude/scratch/f2-eval-harness/results/eval_skab.npz`, table at `…/skab_table.md`)
  **Leaderboard context (not comparable, per protocol):** SKAB repo leaderboard best F1≈0.78 (Conv-AE) under a global-fixed-threshold pointwise protocol; our Standard-F1 is per-unit threshold-*optimal* (TSB-AD convention) so it is structurally optimistic (knn 0.861). The ordering knn≈lof ≫ pca ≫ iforest and the high absolute values are consistent with TSB-AD/Quo-Vadis "simple baselines are strong" + SKAB's long anomaly segments (~35% contamination/unit). These numbers are the **baseline floor every CLU table must include and beat/Pareto-match**.
- **voraus-AD loader smoke (real 1.04 GiB parquet):** download 2 min-ish in background (`curl`), sha256 recorded `c90ab1c7…` and pinned in the loader. Index (reads only 4 meta columns): **0.2 s** → 2122 episodes: **948 train (setting=PRE_A) / 1174 test (755 anomalous)**; all 13 categories present. Column-subset preload + 2 episode loads: 0.4 s; episode shapes ≈ (1069–1096, C) @100 Hz. Real schema confirmed: 137 columns = 7 meta + **130 machine signals** (21/axis × 6 + 4 robot-level) — matches paper; NB signal names differ from my first guess (`motor_iq_*` not `motor_current_*`), loader takes explicit `columns=`.
- **TEP-Rieth loader smoke (real Dataverse files):** md5-verified downloads FaultFree_Training 23.5 MB/1.9 s (`ec126484…`), FaultFree_Testing 45.1 MB/17.6 s (`38ad9810…`); ids+md5+sizes for all 4 files pinned in the loader from the Dataverse API listing. Unit table (subset faults=(0,), runs_per_fault=25): 3.5 s, 50 units; shapes exactly (500, 52) train / (960, 52) test; fault-free labels all-zero as designed. **Faulty files (471/798 MB, ~4 GB RAM via pyreadr) deferred to CSF3 — `fetch(keys=…)` is wired.**
- **SMD negative control (bonus, full 22-unit harness run):** TSB-AD-M.zip 540 MB downloaded in 39 s, sha256 pinned `7de86ac2…`; `fetch()` verified the checksum + selectively extracted the 22 `*_SMD_*` members; per-unit-prefix harness, fast mode (threshold-independent metrics), window 100/stride 1/train_stride 2, seed 42, 39361 train windows total, **wall time 248.7 s**:

| method | **VUS-PR** | VUS-ROC | AUC-PR | AUC-ROC |
|---|---|---|---|---|
| pca_recon | 0.465 ± 0.254 | 0.859 ± 0.157 | 0.449 ± 0.275 | 0.880 ± 0.157 |
| iforest | 0.273 ± 0.215 | 0.713 ± 0.193 | 0.369 ± 0.223 | 0.779 ± 0.192 |
| lof | 0.428 ± 0.265 | 0.829 ± 0.169 | 0.418 ± 0.288 | 0.857 ± 0.170 |
| knn | 0.459 ± 0.247 | 0.845 ± 0.169 | 0.441 ± 0.283 | 0.874 ± 0.166 |

  (all n=22; npz `…/results/eval_smd_tsb.npz`) — PCA-recon ranking first matches TSB-AD's own multivariate finding; this is the negative-control floor CLU is *not supposed to beat* by physics (P1 falsifiability).

## Findings / design decisions the Hub should know
1. **Two label kinds, two protocols** are first-class: voraus-AD is *episode*-labelled (no per-timestep GT ⇒ VUS undefined; episode AUROC primary, matching the voraus paper), SMD-via-TSB uses *per_unit_prefix* (fit on each series' own normal prefix — TSB-AD's protocol). Everything else is point-labelled cross-unit. `PRIMARY_METRIC = {"point": "VUS-PR", "episode": "AUC-ROC"}`.
2. **Degenerate ground truth → NaN, never crash:** fault-free TEP testing runs are all-normal; wrapper returns NaN rows with a warning, aggregation is nanmean + explicit valid-n. Honest per-unit reporting survives.
3. **Windowing is explicit config** (`WindowConfig(size, stride, train_stride)`), recorded in the npz + markdown footer. VUS `sliding_window` is a separate explicit knob (TSB-AD default 100). I did NOT port TSB-AD's `find_length_rank` auto-period estimator (needs statsmodels; window choice should be a deliberate per-dataset decision — record it).
4. **Train-window hygiene:** cross-unit training windows overlapping any labelled-anomalous timestep are dropped (semi-supervised normal-only); seeded subsample guard `max_train_windows` for KNN/LOF memory.
5. **License hygiene:** SKAB (GPL-3.0) never vendored — fetch()-to-user-root only; voraus CC BY-NC-SA noted in docstring; TEP CC0; MIMII CC BY-SA skeleton has no auto-fetch. Vendored TSB-AD is Apache-2.0 with license + provenance retained.
6. **Downloads registry (URL → sha256/md5):**
   - SKAB zip `https://github.com/waico/SKAB/archive/b2c0d46c2971dcbfe71e26087b6d231998bb91c2.zip` → sha256 `45ac11b460e495ba2c1301c3f8e871b688b5ecfa6cd0770b4225594fe45efc80` (5.2 MB)
   - voraus 100 Hz `https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet` → sha256 `c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704` (1,115,942,833 B)
   - TEP via `https://dataverse.harvard.edu/api/access/datafile/<id>`: 3031241 md5 `ec126484534331f85001d8c4ebce6d17` (23.5 MB) · 3031240 `38ad9810fc871026157086ae2c2f0ee9` (45.1 MB) · 3031242 `c5f594d54c47e620ff877feb58407fda` (471 MB) · 3031243 `556bdb64c83021bc0c5f92e427753565` (798 MB)
   - TSB-AD-M `https://www.thedatum.org/datasets/TSB-AD-M.zip` → sha256 `7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d` (540,383,983 B)
   - TSB-AD sdist `https://files.pythonhosted.org/packages/51/c8/…/tsb_ad-1.5.tar.gz` → sha256 `52e474cda6aeb3c2f8f6b3a45e58b11b5b7b55a1510bb5c6f6a15b9053f7b0da`
   - Data root default: `$CHLU_DATA_ROOT` or `~/.cache/chlu/datasets/<name>`; this session's copies live in `.claude/scratch/f2-eval-harness/data/`.
7. **CLU integration path (for V3/ICLR):** implement a `BaselineScorer` producing CLU energy-residual window scores and pass it via `scorer_factory` — the factory *enforces* that the four statistical baselines stay in every run, so the binding rule survives future use.

## Git footprint
- Branch **`agent/experiment-engineer/f2-eval-harness`** (base = local `main` @ `d2d2401`; rebased/up-to-date vs origin/main `40c2f31` — local main is ahead by the unpushed first-fixes merge). Built in a dedicated worktree because the main checkout is parked on concurrent `fix-pack-2`'s branch; worktree removed after commit per protocol §3.2 (reproduce: `git worktree add ../CHLU-f2 agent/experiment-engineer/f2-eval-harness && cd ../CHLU-f2 && uv sync --extra eval`). Main checkout left exactly as found (on `agent/experiment-engineer/fix-pack-2`, clean).
- Commits: `d5c2b70` (vendor TSB-AD v1.5, PA excised) · `9c45748` (metric wrapper + splits + tests) · `10dba0e` (loaders + baselines + harness + eval extra + tests).
- Files: NEW `chlu/eval/**` (6 modules + `_tsb_vendor/**`), NEW `chlu/data/industrial/**` (7 modules), NEW 4 test files; MODIFIED `pyproject.toml` only (optional `eval` extra + ruff vendor exclude). Not pushed; no PR; left for review.
- Commands run: `uv sync --extra eval`, `uv run pytest -q` (59 passed), `uv run ruff check .` (clean), real-data runs via `uv run python` (scripts preserved in `.claude/scratch/f2-eval-harness/*.py`, logs `*.log`).

## Open questions / follow-ups / risks
1. **Vendoring deviation** (above) needs Hub sign-off; if the Hub prefers the pip package, options are a separate eval venv or waiting for TSB-AD numpy-2 support — both worse operationally, evidence of equivalence is in-tree.
2. VUS `sliding_window=100` on ~1 Hz SKAB is a defensible default but unstudied — a window-sensitivity pass (also baseline window size) is cheap analyst work.
3. voraus-AD full-episode harness run (all 130 channels, 948-train-episode fit) is CSF3-scale; laptop smoke covered index + subset loads only. Same for TEP faulty files (RAM) and MIMII (fetch).
4. SKAB "promised v1.0 (300+ files)" never landed upstream (still 34 test runs at the pinned 2024-08 commit) — fine for us, pin gives reproducibility.
5. pandas 3.0.3 resolved in the fresh worktree env (uv.lock is gitignored ⇒ unpinned envs drift by machine). Loaders pass on pandas 3; if other agents' older envs use pandas 2.x the `>=2.0` bound covers them, but env reproducibility across machines is an open repo-level gap.
6. The stray `[tool.setuptools]` block in pyproject is dead (build backend is hatchling) — left untouched (out of scope).

## Proposed handover updates (for the Hub)
- **§2/§6 (architecture/works):** add `chlu/eval/` (F2 harness: TSB-AD-vendored metrics [VUS-PR primary, PA physically excised], 4 mandatory statistical baselines, unit-level splits, `evaluate_dataset` → npz + markdown) and `chlu/data/industrial/` (voraus-AD/SKAB/TEP-Rieth/SMD-TSB loaders + MIMII skeleton; fetch-or-point-at-path with pinned checksums). Install loaders' deps via `uv sync --extra eval`. Tests now **59 passing**.
- **§3 (config):** eval-harness config lives in `chlu/eval/config.py` (`EvalConfig`/`WindowConfig`), intentionally outside `chlu/config.py` (fix-pack-2 collision avoidance). Data root: `$CHLU_DATA_ROOT` or `~/.cache/chlu/datasets`.
- **F2 status:** baseline floors established — SKAB (VUS-PR: knn 0.873, lof 0.868, pca 0.580, iforest 0.522; n=34 units) and SMD negative control (pca 0.465, knn 0.459, lof 0.428, iforest 0.273; n=22 units); both window 100, seed 42, npz + markdown in `.claude/scratch/f2-eval-harness/results/`. voraus/TEP-faulty/MIMII full runs = CSF3 items for the runbook.
- **New reusable fact:** upstream TSB-AD PA-F1 = **1.0** on a case with AUC-ROC 0.727 (our pinned reference input) — concrete, citable-in-appendix demonstration of point-adjust inflation.
- **Ops:** `TSB-AD` PyPI pins numpy<2.0 — never add it to pyproject; the vendored copy is the sanctioned path (regeneration script in `.claude/scratch/f2-eval-harness/vendor_tsb.py`).
