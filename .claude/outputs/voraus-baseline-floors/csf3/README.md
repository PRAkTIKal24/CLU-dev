# CSF3 run recipe — voraus-AD & TEP-Rieth baseline floors

**These floors need NO GPU and NO CLU scorer** — only the 4 statistical baselines
(sklearn/numpy, CPU). Run on the **`multicore`** partition (AMD Genoa, 7-day max),
not `gpuA`. Drivers live in `.claude/scratch/voraus-baseline-floors/` (gitignored,
so they must be rsync'd up with the repo — see step 2).

## ⚠ CRITICAL env gap (must fix before submit)
`scripts/csf3/setup_env_job.sh` runs `uv sync --frozen --extra cuda` — it does
**NOT** install the loaders' deps (pandas/pyarrow/pyreadr). The industrial loaders
will `ModuleNotFoundError` without them. **Fix:** build the env with both extras:
```
uv sync --frozen --extra cuda --extra eval --python 3.11
```
Do this either by editing the setup job's sync line, or once interactively on a
compute node. Also `uv lock` on the laptop must have resolved the `eval` extra
(pyproject already declares it) before `push_repo.sh`, or `--frozen` fails.
Confirmed locally: adding `--extra eval` keeps **jax pinned at 0.9.0** (pandas 3.0.3,
pyarrow 24.0.0, sklearn 1.8.0) — additive, no JAX bump.

Also: `chlu` is not pip-installed by `uv sync` on a bare python; the drivers rely on
`env.sh` activating `.venv` (uv path injection) OR run with
`PYTHONPATH=$CLU_REPO` — the jobscripts source `env.sh`, so `.venv` must be active.
(Locally I had to add `PYTHONPATH=<repo>` to `.venv/bin/python`; verify `import chlu`
works under the CSF3 env before the real submit.)

## 1. Data staging
- **voraus-AD** (1.04 GiB parquet, sha256 `c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704`):
  either (a) rsync the already-verified local copy
  `.claude/scratch/f2-eval-harness/data/voraus_ad/voraus-ad-dataset-100hz.parquet`
  → `~/scratch/chlu_data/voraus_ad/`, or (b) let `job_voraus_floor.sh`'s commented
  `download=True` line fetch it on the compute node (has internet). rsync preferred
  (avoids a 1 GB re-download; sha is re-verified on load either way).
- **TEP-Rieth**: `job_tep_floor.sh` downloads the 3 needed RData on the compute node
  (md5-verified): fault_free_training (24 MB), fault_free_testing (47 MB),
  **faulty_testing (798 MB)**. Faulty-training (471 MB) is NOT downloaded — canonical
  train = fault-free-training only.

## 2. Submit
```bash
# LAPTOP: uv lock (fold eval+cuda), push repo (rsync carries the gitignored driver + uv.lock)
uv lock && export CSF3_USER=<user> && scripts/csf3/push_repo.sh
# ...also rsync the driver dir (it's under .claude, gitignored):
rsync -av .claude/scratch/voraus-baseline-floors/ $CSF3_USER@csf3...:~/scratch/CHLU/.claude/scratch/voraus-baseline-floors/
# (and copy the two jobscripts from .claude/outputs/.../csf3/ to ~/scratch/CHLU/)

# CSF3: build env WITH eval extra (see gap above), then:
cd ~/scratch/CHLU
sbatch .claude/outputs/voraus-baseline-floors/csf3/job_voraus_floor.sh   # -p multicore -n16 -t6:00:00
sbatch .claude/outputs/voraus-baseline-floors/csf3/job_tep_floor.sh      # -p multicore -n24 -t1-0
squeue --me
```

## 3. Resource envelope (measured priors → CSF3 sizing)
| dataset | test units | train windows (chosen) | flat dim | peak RAM (predicted) | cores req | wall (predicted) |
|---|---|---|---|---|---|---|
| voraus-AD | 1174 episodes | ~95 k (train_stride=10) | 100×130=13 000 | ~20 GB (PCA full-SVD U≈10 GB) | `-n 16` | 20 min – 2 h |
| TEP-Rieth | **10 000 faulty** (+500 degenerate) | ~100 k (train_stride=2) | 100×52=5 200 | ~8–12 GB (pyreadr 4 GB + windows) | `-n 24` | 1 – 6 h |

- **Why train_stride matters (voraus):** at train_stride=1 the concat of ~943 k
  windows × 13 000 float32 = **~49 GB** peak *before* the 100 k subsample cap →
  OOM risk. `train_stride=10` → ~95 k windows → ~5 GB. **Set it.**
- **PCA-recon is the voraus memory/time bottleneck** (full economy SVD on
  95 k × 13 000). If it OOMs, lower `--max-train-windows` (e.g. 40 000) or
  `--window` (e.g. 50 → dim 6 500). Report which.
- TEP KNN/LOF scoring over 10 000 units is the wall-time driver; `--metrics-mode fast`
  (threshold-independent metrics only) keeps per-unit cost down.

## 4. Expected outputs (pull with sync_project.sh)
- voraus: `projects/voraus_floor/results/{voraus_floor.json, voraus_floor.md}` —
  overall episode AUROC/AUPR per baseline + **per-category AUROC map** + shuffled control.
- TEP: `projects/tep_floor/results/{tep_floor.json, tep_table.md, eval_tep_rieth.npz}` —
  overall VUS-PR/AUROC per baseline + **per-fault VUS-PR map** (faults 3/9/15 predicted low).
Record: job IDs, `sacct -j <id> --format=Elapsed,MaxRSS`, and any OOM/NaN.
