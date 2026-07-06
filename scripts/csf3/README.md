# CLU on CSF3 (Manchester) — runbook

Execution path for CLU/CHLU experiments on the University of Manchester
**CSF3** cluster (target: **NVIDIA A100 80GB**, partition `gpuA`). Prepared
2026-07-06 from the **official CSF3 docs** (all cluster facts cited below);
the agent that wrote this had **no CSF3 credentials**, so anything that can
only run on the cluster is marked **UNTESTED-BY-AGENT** with the exact
command to run.

Status legend:
- **[DOCS]** — verified against the official CSF3 documentation (URL + page
  "last modified" date given).
- **[LOCAL]** — executed and verified on the dev laptop.
- **[UNTESTED-BY-AGENT]** — prepared, pattern-based; the Head runs it first.

---

## 1. Cluster facts (all [DOCS], ri.itservices.manchester.ac.uk)

| Fact | Value | Source page (last modified) |
|---|---|---|
| Scheduler | **Slurm** (`sbatch`/`squeue`/`scancel`/`srun`; migrated from SGE, an "SGE to Slurm Reference" exists) | `/csf3/batch-slurm/` (2026-03-30) |
| Login | `ssh <uom-username>@csf3.itservices.manchester.ac.uk` (IT username of form `mabcxyz1`, not email); off-campus ⇒ GlobalProtect VPN | `/csf3/getting-started/connecting/` (2026-03-03) |
| **Internet** | **Login nodes: campus-only (no GitHub/PyPI!). Compute nodes (batch & interactive jobs): full off-campus access.** Env builds & dataset downloads must run inside jobs | `/csf3/getting-started/connecting/` (2026-03-03) |
| A100 partition | `-p gpuA`: 19 nodes × 4× A100 SXM4 **80GB** (sm_80), 48 AMD Milan cores + 512GB/node, driver **580.126.09** (CUDA 13.0), **free at point of use, ≤4 GPUs concurrently**, `-G N` required, ≤12 host cores/GPU (10.4GB RAM/core), batch max `-t 4-0`, interactive max 1 day | `/csf3/batch-slurm/gpu-jobs-slurm/` (2026-06-11), `/csf3/batch-slurm/partitions/` (2026-04-15) |
| Other partitions | `serial` (1-core Intel, 7-0), `multicore` (AMD Genoa 2–168 cores, 7-0), `interactive` (6h), `gpuL` (L40S 48GB), `gpuA40GB`/`gpuH` (restricted). V100s removed Oct 2025 | `/csf3/batch-slurm/partitions/` (2026-04-15) |
| Wallclock | **No default — `-t` is mandatory**, sbatch rejects jobs without it. Format `d-h` recommended (`1-0` = 1 day) | `/csf3/batch-slurm/timelimits-slurm/` (2026-01-12) |
| Filesystems | `$HOME` = `/mnt/iusers01/<group>/<user>`: small, backed up, keep ≤50GB, don't run jobs there. `~/scratch → /scratch/<user>`: fast, **no backup, 3-month auto-cleanup**, run jobs here. `$TMPDIR`: 1.6TB node-local NVMe, wiped at job end | `/csf3/filesystems/home-scratch-rds/` (2026-03-17) |
| Jobscript shape | `#!/bin/bash --login`, `#SBATCH` directives, `module purge` recommended, submit from scratch | `/csf3/batch-slurm/partitions/` (2026-04-15) |
| Job arrays | `#SBATCH -a 0-N` + `${SLURM_ARRAY_TASK_ID}` (max task id 250000) | `/csf3/batch-slurm/job-arrays-slurm/` |
| Interactive GPU | `srun -p gpuA -G 1 -n 1 -t 1-0 --pty bash` (24h limit) | `/csf3/batch-slurm/gpu-jobs-slurm/` (2026-06-11) |
| GPU monitoring | `squeue` → `ssh <node>` (allowed where you have a job) → `module load tools/bintools/nvitop; nvitop` or `nvidia-smi` | `/csf3/batch-slurm/gpu-jobs-slurm/` (2026-06-11) |
| CUDA modules | `libs/cuda/12.8.1` newest; **not needed for us** (see §3) | `/csf3/batch-slurm/gpu-jobs-slurm/` (2026-06-11) |
| Conda | `module load apps/binapps/conda/miniforge3/25.9.1` (envs default to `~/.conda` — watch home quota; `-p` to relocate) | `/csf3/software/applications/conda/` (2026-04-23) |
| File transfer | rsync/scp over ssh to the login host, port 22 | `/csf3/filesystems/file-transfer/` (2024-10-31) |

**Slurm GPU-job env vars** [DOCS]: `$CUDA_VISIBLE_DEVICES` (IDs start at 0),
`$SLURM_GPUS` (count). GPUs run in DEFAULT compute mode but **Slurm gives
your job exclusive use of its assigned GPUs** — no other job shares them.

---

## 2. Files in this directory

| File | Runs where | Purpose |
|---|---|---|
| `env.sh` | cluster (sourced by jobs) | venv activation + headless env vars (`MPLBACKEND=Agg`, `SCIKIT_LEARN_DATA`, JAX compile cache, `PYTHONUNBUFFERED`) |
| `setup_env_job.sh` | cluster (sbatch, `serial`) | one-time env build **on a compute node** (internet): installs uv, `uv sync --frozen --extra cuda` |
| `job_cpu_smoke.sh` | cluster (sbatch, `serial`) | ~5-min CPU smoke: CLI, Agg plot, `pytest tests/test_core.py tests/test_data.py` |
| `job_gpu_single.sh` | cluster (sbatch, `gpuA`) | one experiment on one A100; GPU preflight assert; `--export` knobs `EXP/PROJECT/SEED/EXTRA_ARGS` |
| `job_gpu_array_seeds.sh` | cluster (sbatch, `gpuA`) | seed-sweep array; one A100 + one `projects/<prefix>_s<seed>` per task |
| `push_repo.sh` | **laptop** | rsync repo **+ gitignored `uv.lock`** → `~/scratch/CHLU` |
| `sync_project.sh` | **laptop** | `pull|push` of `projects/<name>/` artifacts (plots/results/models) |
| `requirements-csf3.txt` | either | pinned pip fallback, generated from `uv.lock` (see §3.3) |

---

## 3. Environment recipe

### 3.1 Why pip CUDA wheels, not the cluster CUDA module
The `cuda` extra added to `pyproject.toml` is `jax[cuda12]`
(Linux-only marker), which resolves to `jax-cuda12-plugin[with-cuda]` —
**self-contained wheels bundling CUDA + cuDNN**. They require only the node's
NVIDIA driver: CUDA-12 wheels need ≥525.60.13, and the A100 nodes run
**580.126.09** [DOCS] — comfortably compatible (a `jax[cuda13]` extra also
exists upstream if ever needed; driver 580 supports CUDA 13.0). Therefore
**do not `module load libs/cuda/...`** — an LD_LIBRARY_PATH CUDA can shadow
the bundled libs and break jax.

Locked pins (laptop parity, Python 3.11): `jax==0.9.0`, `jaxlib==0.9.0`,
`jax-cuda12-plugin==0.9.0`, `equinox==0.13.4`, `optax==0.2.6`,
`diffrax==0.7.0` [LOCAL, from `uv.lock`].

### 3.2 Primary path — uv (recommended)
```bash
# LAPTOP, once after this branch merges: fold the cuda extra into the
# canonical lock (additive; verified to keep jax at 0.9.0):
uv lock

# LAPTOP (VPN/campus): ship code + lockfile (uv.lock is gitignored — rsync is
# the ONLY way it reaches the cluster; a git clone would resolve fresh):
export CSF3_USER=<uom-username>
scripts/csf3/push_repo.sh

# CSF3 LOGIN NODE: build env on a compute node (login nodes have no PyPI):
ssh $CSF3_USER@csf3.itservices.manchester.ac.uk
cd ~/scratch/CHLU && sbatch scripts/csf3/setup_env_job.sh     # [UNTESTED-BY-AGENT]
```
`setup_env_job.sh` installs uv (`~/.local/bin`), puts the uv cache and
managed CPython 3.11 on scratch (CUDA wheels ≈ 6GB — too big for home),
runs `uv sync --frozen --extra cuda --python 3.11` (exact laptop pins), and
ends with a CPU-side import/jit check. Jobs then activate `.venv` directly —
**no network in the job critical path**.

On macOS the extra is a marker no-op: `uv sync --extra cuda` verified
[LOCAL] to install **0** cuda/nvidia packages there.

### 3.3 Fallback path — Miniforge conda + pinned pip
Inside an interactive/batch job (internet), [UNTESTED-BY-AGENT]:
```bash
module purge
module load apps/binapps/conda/miniforge3/25.9.1        # [DOCS]
conda create -p ~/scratch/envs/clu python=3.11 -y       # -p: keep off home quota
conda activate ~/scratch/envs/clu
cd ~/scratch/CHLU
pip install -r scripts/csf3/requirements-csf3.txt       # pinned, from uv.lock
pip install -e . --no-deps
```
Then `conda activate` before sourcing `env.sh` in jobscripts (env.sh detects
a pre-activated env and skips `.venv`). Regenerate the pin file after any
dependency change:
`uv export --frozen --extra cuda --no-hashes --no-emit-project --no-annotate -o scripts/csf3/requirements-csf3.txt`.
Caveat: the file's markers are exact for Linux and/or Python 3.11 — use
Python 3.11 (as everywhere in this runbook).

### 3.4 JAX persistent compilation cache & memory
`env.sh` exports `JAX_COMPILATION_CACHE_DIR=~/scratch/.jax_cache` +
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1`: XLA executables persist
across processes/jobs — array-sweep tasks after the first skip compilation.
Verified [LOCAL] on jax 0.9.0: cache dir populated with `*-cache` entries;
a second process logs `jax._src.lru_cache: Cache hit for key: ...`. GPU memory: the job owns its A100
(§1), so JAX's default ~75% preallocation is correct for batch jobs;
`XLA_PYTHON_CLIENT_MEM_FRACTION` / `XLA_PYTHON_CLIENT_PREALLOCATE=false`
are documented in `env.sh` for co-located processes / interactive nvitop.

---

## 4. Running experiments

```bash
# single run (all knobs via --export; wallclock via -t, REQUIRED, max 4-0):
cd ~/scratch/CHLU
sbatch --export=ALL,EXP=exp-b,PROJECT=csf3_b1,SEED=42,EXTRA_ARGS= -t 1-0 \
       scripts/csf3/job_gpu_single.sh

# 5-seed sweep (seeds 0..4  ->  projects/csf3_b_s0 ... csf3_b_s4):
sbatch --export=ALL,EXP=exp-b,PROJECT_PREFIX=csf3_b,SEED_BASE=0,EXTRA_ARGS= \
       -t 1-0 -a 0-4 scripts/csf3/job_gpu_array_seeds.sh

# monitor / cancel:
squeue --me            # ST=PD pending, R running
scancel <jobid>
# live GPU view: squeue -> ssh <node> -> module load tools/bintools/nvitop && nvitop

# artifacts back to the laptop (LAPTOP shell):
scripts/csf3/sync_project.sh pull csf3_b1
scripts/csf3/sync_project.sh pull 'csf3_b_s*'
```

Experiment CLI knobs (from `chlu/cli/experiment_cmd.py`): `exp-a|exp-b|exp-c|
all-experiments`, `--project`, `--seed`, `--quick`; exp-c also `--init-mode
{random,centroid}` and `--centroid-noise-scale F`. Pass extras through
`EXTRA_ARGS`.

### Known caveats baked into the templates
- **`--quick` is not wired for CHLU dynamics training** (handover §7.10):
  for exp-a/b it still trains `training.epochs=1000`; only baselines/exp-c
  shorten. Budget wallclock accordingly until fix-pack-2 merges.
- **First `exp-c` run downloads MNIST** via `fetch_openml` — works because
  compute nodes have internet [DOCS]; cached under `$SCIKIT_LEARN_DATA`
  (repo-local on scratch) for later runs.
- `projects/` is created **relative to $PWD** (`chlu/project.py`) — `env.sh`
  always `cd`s to `$CLU_REPO`, so artifacts land in
  `~/scratch/CHLU/projects/<name>/`.
- **Scratch is not storage**: no backup + 3-month auto-cleanup [DOCS]. Pull
  results to the laptop promptly (`sync_project.sh pull`).
- Runtimes on A100 are **UNKNOWN until first run** — templates default to
  `-t 4:00:00` for smokes; override per run. A too-small `-t` gets the job
  SIGTERM'd (30s grace, then SIGKILL) [DOCS].
- $TMPDIR NVMe staging is *not* used: current datasets are tiny (MNIST
  55MB, generated figure-8/sine). Revisit for F3 industrial datasets.

---

## 5. Smoke-test checklist for the Head (ordered; all [UNTESTED-BY-AGENT])

0. **Prereqs (laptop):** on campus network or GlobalProtect VPN; a CSF3
   account (UoM IT username).
1. **Refresh lock + push code:** in the repo:
   `uv lock && export CSF3_USER=<username> && scripts/csf3/push_repo.sh`
   → expect an rsync file list ending `done. uv.lock included: yes`
   (a warning about the lock lacking the cuda extra means `uv lock` is
   still needed).
2. **Build env (≈5–15 min queued + built):**
   `ssh $CSF3_USER@csf3.itservices.manchester.ac.uk`, then
   `cd ~/scratch/CHLU && sbatch scripts/csf3/setup_env_job.sh`.
   Watch `squeue --me`; then `cat slurm-<jobid>.out` → expect
   `python OK; jax 0.9.0 | chlu 0.2.4`, all five `pkg==version` lines
   (no `MISSING`), `jit smoke: 8.0`, `[setup] DONE`.
3. **CPU smoke (~5 min run):** `sbatch scripts/csf3/job_cpu_smoke.sh` →
   output ends `=== CPU smoke PASSED ===` with the pytest line reporting
   **0 failures** (18 passed on main as of 2026-07-06).
4. **A100 smoke:** `sbatch scripts/csf3/job_gpu_single.sh` (defaults:
   exp-a, project `csf3_smoke_a`, seed 42, `--quick`, `-t 4:00:00`).
   Expect: `nvidia-smi` shows one A100-SXM4-80GB; preflight prints
   `backend: gpu` + `CudaDevice(id=0)`; run completes with
   `✓ Experiment A completed`; `projects/csf3_smoke_a/{plots,results,models}`
   listed non-empty (plots `.png`, `results/exp_a_metrics.npz`, model
   `.pkl`s). **Note the wallclock actually used** (slurm `sacct -j <id>
   --format=Elapsed`) — it calibrates `-t` for real runs (§7.10 means this
   smoke trains the full 1000-epoch dynamics loop).
5. **Pull artifacts (laptop):** `scripts/csf3/sync_project.sh pull csf3_smoke_a`
   → `projects/csf3_smoke_a/` appears locally with the same tree.
6. *(Optional)* interactive session for debugging:
   `srun -p gpuA -G 1 -n 4 -t 0-1 --pty bash`, then
   `source scripts/csf3/env.sh` and poke around (24h cap, don't idle-hold
   GPUs — etiquette section of the GPU docs).

Report back: job IDs, elapsed times, and any deviation from the expected
outputs above — those calibrate the wallclock defaults in the templates.

---

## 6. What was verified locally before hand-off ([LOCAL], 2026-07-06)

- `bash -n` clean on all seven shell scripts; executable bits set.
- `uv lock` with the new `cuda` extra is **additive** (jax stays 0.9.0;
  plugin + nvidia-* wheels added under Linux-only resolution).
- macOS `uv sync --extra cuda --dry-run`: 80 packages, **zero** cuda/nvidia.
- `MPLBACKEND=Agg` + `import chlu.utils.plotting` + savefig: works headless.
- `SCIKIT_LEARN_DATA` env var redirects `sklearn.datasets.get_data_home()`.
- JAX persistent compilation cache: second process gets a cache hit.
