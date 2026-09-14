# csf3-runbook — experiment-engineer report

Task + acceptance criterion: Prepare a copy-paste CSF3 (2×A100) execution path — runbook + JAX/CUDA env recipe + Slurm batch/array/rsync templates under `scripts/csf3/` + minimal headless fixes — cluster facts verified from official docs, everything credential-gated marked UNTESTED with exact commands for the Head.

Status: **done** (all laptop-verifiable parts executed; cluster-side steps are prepared + labeled UNTESTED-BY-AGENT with an ordered smoke checklist).

## What I did
- **Verified cluster facts from the official CSF3 docs over the web** (curl from this machine; laptop has internet even though I have no CSF3 credentials). Headline: **CSF3 now runs Slurm** (page "Running Jobs – The Batch System (Slurm)", ri.itservices.manchester.ac.uk/csf3/batch-slurm/, mod. 2026-03-30; an "SGE to Slurm Reference" documents the qsub→sbatch migration). All facts + URLs + page-modification dates tabulated in `scripts/csf3/README.md` §1.
- **Built the full `scripts/csf3/` kit** (9 files, branch `agent/experiment-engineer/csf3-runbook`): README runbook · `env.sh` (shared job env) · `setup_env_job.sh` (env build **as a batch job** — see key discovery below) · `job_cpu_smoke.sh` · `job_gpu_single.sh` · `job_gpu_array_seeds.sh` (seed sweeps, per-seed projects) · `push_repo.sh` + `sync_project.sh` (rsync helpers) · `requirements-csf3.txt` (pinned pip fallback exported from uv.lock).
- **Repo touch-up (minimal, zero `chlu/` edits):** added `[project.optional-dependencies] cuda = ["jax[cuda12]; sys_platform == 'linux'"]` to `pyproject.toml`. All other headless needs solved by env vars in `env.sh` (no code changes → no collision surface with fix-pack-2, which is concurrently editing `chlu/` — I worked in a dedicated worktree per protocol §3.2 and removed it after committing).

## Key cluster facts (all cited in README §1)
1. **Scheduler = Slurm.** `sbatch/squeue/scancel/srun`; jobscripts `#!/bin/bash --login` + `#SBATCH`; **wallclock `-t` is mandatory (no default — sbatch rejects without it)**; GPU batch max `4-0`.
2. **A100 target = `-p gpuA`**: 19 nodes × 4× A100 SXM4 **80GB** (sm_80), AMD Milan hosts, **free at point of use, ≤4 GPUs concurrently per user** (2026-02-20 policy), `-G N` required, ≤12 host cores/GPU @10.4GB RAM/core. `gpuA40GB` exists but is restricted; V100s retired Oct 2025. Driver **580.126.09 (CUDA 13.0)**.
3. **Login nodes have NO off-campus internet; compute nodes DO** (official connecting page). ⇒ env builds, uv installer, CPython download, and the first MNIST `fetch_openml` all must run *inside jobs* — `setup_env_job.sh` is a batch job for exactly this reason. Off-campus laptop access needs GlobalProtect VPN; host `csf3.itservices.manchester.ac.uk`, port 22.
4. **Filesystems:** home small/backed-up (keep ≤50GB, don't run jobs there); `~/scratch` fast, **no backup + 3-month auto-cleanup** — repo lives at `~/scratch/CHLU`, artifacts must be pulled to the laptop promptly; `$TMPDIR` = 1.6TB node-local NVMe (not used yet — our datasets are tiny; flagged for F3).
5. Slurm gives a job **exclusive use of its assigned GPUs** ⇒ JAX default preallocation is correct for batch; `XLA_PYTHON_CLIENT_MEM_FRACTION`/`PREALLOCATE=false` documented in env.sh for the two exception cases.

## Env recipe (decisions + rationale)
- **CUDA via self-contained pip wheels, NOT the cluster CUDA module:** `jax[cuda12]` → `jax-cuda12-plugin[with-cuda]` bundles CUDA+cuDNN; only the node driver matters (580.126.09 ≫ required 525.60.13). README warns *against* `module load libs/cuda` (LD_LIBRARY_PATH shadowing breaks jax). `jax[cuda13]` also exists upstream as an option (driver supports 13.0).
- **Primary path = uv:** `push_repo.sh` (laptop) → `sbatch setup_env_job.sh` → installs uv, caches on scratch (CUDA wheels ≈6GB > home budget), `uv sync --frozen --extra cuda --python 3.11` → jobs activate `.venv` directly (**no network/resolver in the job path**). Python 3.11 pinned = laptop parity (jax 0.9.0 branch of the lock).
- **Critical wrinkle handled: `uv.lock` is gitignored** ⇒ env parity travels ONLY via rsync (`push_repo.sh` carries it; a git clone would resolve fresh — also GitHub is unreachable from login nodes anyway). Both `push_repo.sh` and `setup_env_job.sh` guard against a lock predating the cuda extra (grep for `jax-cuda12`; setup job runs an additive `uv lock` on-node if needed).
- **Fallback path = Miniforge conda + pinned pip:** `module load apps/binapps/conda/miniforge3/25.9.1` (per official conda page) + `pip install -r scripts/csf3/requirements-csf3.txt` + `pip install -e . --no-deps`. The requirements file is exported from the lock (regeneration command in README §3.3) and includes dev deps (pytest) so smokes work in either env. `env.sh` detects a pre-activated conda env and skips `.venv`.
- **JAX persistent compilation cache** wired in env.sh (`JAX_COMPILATION_CACHE_DIR=~/scratch/.jax_cache`, min-compile-time 1s) — directly attacks our cold-start pain; array-sweep tasks after the first reuse compiled executables.
- **Headless env (no code changes):** `MPLBACKEND=Agg` (plotting imports pyplot at module import — env var is early enough), `SCIKIT_LEARN_DATA` (mnist.py calls `fetch_openml` without `data_home` ⇒ env var honored), `PYTHONUNBUFFERED=1` (live slurm-*.out).

## How I verified (all executed; observed outputs)
- `bash -n` on all 7 shell scripts: clean (×7 "bash -n OK"); executable bits set on the 6 executables.
- **Agg headless:** `MPLBACKEND=Agg .venv/bin/python` importing `chlu.utils.plotting` + savefig → `Agg OK; ... wrote .../agg_smoke.png 17331 bytes`.
- **sklearn cache redirect:** `SCIKIT_LEARN_DATA=/tmp/clu_skl_test` → `get_data_home()` = `/tmp/clu_skl_test` (assert passed).
- **JAX persistent cache (jax 0.9.0):** run 1 populated `/tmp/clu_jaxcache_test` with `jit__lambda-…-cache` entries; run 2 logged `DEBUG:jax._src.lru_cache: Cache hit for key: 'jit_convert_element_type-…'`.
- **cuda extra:** `uv lock` after adding it is **additive** (jax stays 0.9.0/0.6.2; adds jax-cuda12-plugin/pjrt 0.9.0 + nvidia-\*-cu12 wheels, e.g. nvidia-cudnn-cu12 9.24.0.43). macOS `uv sync --extra cuda --dry-run`: **80 packages, 0 cuda/nvidia** (marker no-op verified twice). Exact cluster command `uv sync --frozen --extra cuda --python 3.11 --dry-run` against the updated lock: **exit 0**.
- Export check: `requirements-csf3.txt` (109 lines) carries `jax==0.9.0 ; python_full_version >= '3.11'`, plugin pins, equinox 0.13.4, optax 0.2.6, diffrax 0.7.0, pytest 8.4.2.

## UNTESTED-BY-AGENT (Head's ordered smoke checklist — full version in README §5)
0. VPN/campus + CSF3 account. 1. `uv lock && export CSF3_USER=<uom-username> && scripts/csf3/push_repo.sh` (expect `uv.lock included: yes`). 2. ssh in; `cd ~/scratch/CHLU && sbatch scripts/csf3/setup_env_job.sh` → slurm-*.out ends `python OK; jax 0.9.0 | chlu 0.2.4` + 5 pkg pins + `jit smoke: 8.0` + `[setup] DONE`. 3. `sbatch scripts/csf3/job_cpu_smoke.sh` → `=== CPU smoke PASSED ===`, pytest 0 failures. 4. `sbatch scripts/csf3/job_gpu_single.sh` (defaults exp-a/--quick/seed 42/`-t 4:00:00`) → nvidia-smi shows A100-SXM4-80GB, preflight `backend: gpu`, `✓ Experiment A completed`, non-empty `projects/csf3_smoke_a/{plots,results,models}`; **record `sacct -j <id> --format=Elapsed`** to calibrate `-t`. 5. Laptop: `scripts/csf3/sync_project.sh pull csf3_smoke_a`.
- Honest unknowns: A100 runtimes for our trainings (templates default `-t 4:00:00`, flagged); whether uv's managed-CPython download works behind the campus egress (if it fails: conda fallback documented); conda module version availability drift.

## Design notes for Hub review
- GPU jobs **fail fast** if JAX lands on CPU (preflight `assert jax.default_backend()=="gpu"`) — prevents silently burning 4-day GPU allocations on CPU training.
- Seed-sweep array: one A100 + one `projects/<prefix>_s<seed>` per task (no artifact collisions; `chlu project create … || true` for idempotence); compilation cache shared across tasks.
- Templates parametrize via `sbatch --export=ALL,EXP=…,PROJECT=…,SEED=…,EXTRA_ARGS=…` — one template serves all experiments; §7.10 caveat (--quick doesn't shorten CHLU dynamics training) is written into the template comments and the wallclock guidance.
- `projects/` is cwd-relative (`chlu/project.py`) ⇒ `env.sh` always `cd $CLU_REPO`; artifacts land on scratch by construction.
- §7.12 (UF_HIDDEN .pth) is a macOS-only pathology — the Linux venv-activate path is unaffected (and jobs don't use `uv run` at all).

## Git footprint
- Branch: **`agent/experiment-engineer/csf3-runbook`** (base `main` @ d2d2401), built in a dedicated worktree (now removed; branch persists). NOT pushed, no PR (per task).
- Commits: **f6cb0ab** `[experiment-engineer] add cuda extra (jax[cuda12], Linux-marked) for GPU installs` (pyproject.toml, +7) · **c36ae55** `[experiment-engineer] add CSF3 Slurm runbook, batch templates, rsync helpers` (9 new files under scripts/csf3/, +745).
- Files touched: `pyproject.toml` (7-line extra only) + new `scripts/csf3/{README.md, env.sh, setup_env_job.sh, job_cpu_smoke.sh, job_gpu_single.sh, job_gpu_array_seeds.sh, push_repo.sh, sync_project.sh, requirements-csf3.txt}`. **Zero `chlu/**` edits** (deliberate — fix-pack-2 concurrency).
- Rebase: `git rebase main` → already up to date. No conflicts. Note: `git worktree list` shows three more live agent worktrees (f2-eval-harness, v1-l0-gate, v2-so2-build) and fix-pack-2 advancing in the main checkout — parallel discipline held.

## Open questions / follow-ups / risks
1. **Merge-order dependency:** after merging, the Head must run `uv lock` once on the laptop before the first `push_repo.sh` (guards exist in both scripts, but doing it up front avoids one wasted queue cycle).
2. `pytest` addopts force `--cov` (pyproject) — fine in both env paths (pytest-cov pinned in the export), but coverage.xml will be written in the cluster repo; harmless.
3. When fix-pack-2 lands (--quick wiring, seeded mnist subsample), the wallclock guidance in `job_gpu_single.sh` comments should be relaxed — one-line follow-up.
4. AWS H100 path (roadmap D3 secondary) not covered here — same env recipe applies (pip CUDA wheels), only the scheduler wrapper differs; separate task if needed.
5. If CSF3's free-at-point-of-use policy changes (4-GPU limit, Feb-2026 wording), array-sweep throughput assumptions change; re-check the gpu-jobs page then.

## Proposed handover updates (for the Hub)
- **§3 (CLI/config):** pyproject now has an optional extra `cuda = ["jax[cuda12]; sys_platform == 'linux'"]` — no-op on macOS; cluster installs use `uv sync --frozen --extra cuda`.
- **§6 (env facts) add:** CSF3 execution path exists at `scripts/csf3/` (branch csf3-runbook): CSF3 = **Slurm**, A100-80GB = `-p gpuA -G N` (free tier ≤4 GPUs, max `-t 4-0`, no default wallclock); **login nodes have no off-campus internet — compute nodes do** (env builds + dataset downloads run as jobs); repo lives at `~/scratch/CHLU` on-cluster; `uv.lock` is gitignored ⇒ laptop↔cluster env parity flows via `scripts/csf3/push_repo.sh` (rsync), never via git clone.
- **§7.12 scope note:** UF_HIDDEN `.pth` bug is macOS-only; the Linux cluster path (direct venv activation, no `uv run` at job time) is unaffected.
- **§7.10 cross-ref:** cluster templates encode the "--quick not wired for dynamics training" caveat in their wallclock guidance; update `scripts/csf3/job_gpu_single.sh` comment when fix-pack-2 merges.
- JAX persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`) is verified working on jax 0.9.0 — worth adopting on the laptop too to blunt the §6 cold-start pain.
