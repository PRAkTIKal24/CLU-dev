# scripts/csf3/env.sh — shared runtime environment for CLU jobs on CSF3 (Slurm).
#
# Source this from every jobscript AFTER `module purge`:
#     source "$CLU_REPO/scripts/csf3/env.sh"
# It is a *sourced* file, not an executable — do not `sbatch` it.
#
# Everything here is env-var-only on purpose: no chlu/ code changes are needed
# for headless cluster runs (matplotlib backend, sklearn cache, JAX caches).

# ---------------------------------------------------------------------------
# Repo location on the cluster. CSF3 policy: run jobs from ~/scratch (fast,
# no quota), NOT from home. `projects/` is created relative to $PWD by
# chlu.project.ProjectManager, so we cd into the repo before running anything.
# Override by exporting CLU_REPO before sourcing.
# ---------------------------------------------------------------------------
export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
cd "$CLU_REPO" || { echo "FATAL: CLU_REPO=$CLU_REPO not found"; exit 1; }

# ---------------------------------------------------------------------------
# Python env: the uv-built venv inside the repo (see setup_env_job.sh).
# We activate the venv directly instead of `uv run` so that job runtime never
# touches the network (CSF3 login nodes have no off-campus access; compute
# nodes do, but we don't want a resolver in the job's critical path).
# NOTE: we do NOT `module load libs/cuda/...` — jax is installed with
# self-contained pip CUDA wheels (jax[cuda12] / with-cuda), which bundle
# CUDA + cuDNN and only require the node's NVIDIA driver. Mixing in a CUDA
# module via LD_LIBRARY_PATH can shadow the bundled libs and break jax.
# ---------------------------------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
    echo "[env.sh] using pre-activated env: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
elif [ -f "$CLU_REPO/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CLU_REPO/.venv/bin/activate"
else
    echo "FATAL: no venv at $CLU_REPO/.venv — run setup_env_job.sh first"
    echo "       (or conda-activate the fallback env before sourcing env.sh)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Headless plotting: force the non-interactive Agg backend (no $DISPLAY on
# compute nodes). chlu.utils.plotting imports pyplot at module import time,
# so this must be set before any chlu import. Env var only — no code change.
# ---------------------------------------------------------------------------
export MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# scikit-learn dataset cache (MNIST via fetch_openml in chlu/data/mnist.py,
# which calls fetch_openml without data_home => honours SCIKIT_LEARN_DATA).
# Kept inside the repo on scratch: ~55MB, re-downloadable from a compute node
# (which has off-campus access) if the scratch cleaner removes it.
# ---------------------------------------------------------------------------
export SCIKIT_LEARN_DATA="${SCIKIT_LEARN_DATA:-$CLU_REPO/.sklearn_data}"
mkdir -p "$SCIKIT_LEARN_DATA"

# ---------------------------------------------------------------------------
# JAX persistent compilation cache — big win for repeated jobs (sweeps):
# XLA executables are cached on disk and reused across processes/jobs.
# min_compile_time_secs=1 caches anything that took >=1s to compile.
# ---------------------------------------------------------------------------
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/scratch/.jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# ---------------------------------------------------------------------------
# GPU memory. Slurm gives this job exclusive use of its assigned GPUs
# (docs: "Slurm will prevent other jobs from accessing the GPUs assigned to
# your job"), so JAX's default preallocation (~75% of GPU RAM) is fine and
# fastest for batch jobs — leave it alone.
# Uncomment ONLY for special cases:
#   - two of our own processes sharing one A100:
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
#   - interactive debugging where you want nvitop to show true usage:
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# ---------------------------------------------------------------------------

# Stream python output into slurm-*.out as it happens (rich/tqdm progress).
export PYTHONUNBUFFERED=1

echo "[env.sh] repo=$CLU_REPO python=$(python --version 2>&1) venv=$VIRTUAL_ENV"
echo "[env.sh] MPLBACKEND=$MPLBACKEND JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"

# ---------------------------------------------------------------------------
# ⭐ BEGIN gen-wells-route-b — THE ARTEFACT ROOT (owner: experiment-engineer)
#
# ⛔ The inherited defect this closes.  Neither this file nor any jobscript knew about
# the gen artefact root, so every submit had to carry five CHLU_* variables by hand in
# `--export` — and an omission fails LATE, inside the loader, AFTER the scheduler has
# already handed the job its GPU.  On a queue that costs a resubmit and a wait, and it
# will happen.  ⭐ A job now INHERITS the root instead of being handed it.
#
# ⭐ Every variable keeps the `${VAR:-default}` form, so an explicit `--export` still
# wins.  ⛔ No path is hardcoded: everything hangs off $CLU_REPO.
#
# The layout is `gen-cluster-stage-artefacts`' own, not a guess — its STAGED.json
# records `artefact_root` with `data/gen/...` and `outputs/...` beneath it.
# ---------------------------------------------------------------------------
export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CLU_REPO/artefacts}"
export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"

# ⭐ Assert the roots exist AT JOB START and die loudly with the MISSING NAME.
# ⛔ A job must never reach the GPU only to discover it cannot open an artefact.
# ⚠ Opt out with CHLU_SKIP_ARTEFACT_CHECK=1 for jobs that genuinely need no artefacts
# (the CPU smoke, the env build) — ⛔ never for a training job.
if [ "${CHLU_SKIP_ARTEFACT_CHECK:-0}" != "1" ]; then
    _chlu_missing=""
    for _v in CHLU_CLAUDE_ROOT CHLU_GEN_DATA_ROOT CHLU_GEN_BRIDGE_ROOT \
              CHLU_GEN_STORE_RUNS CHLU_GEN_DECLIFT_DIR; do
        eval "_p=\${$_v}"
        [ -e "$_p" ] || _chlu_missing="$_chlu_missing\n    $_v -> $_p"
    done
    if [ -n "$_chlu_missing" ]; then
        # shellcheck disable=SC2059
        printf "FATAL [env.sh] gen artefact root(s) MISSING, by name:$_chlu_missing\n"
        echo   "       Stage them first (scripts/csf3/stage_artefacts.sh, owned by"
        echo   "       gen-cluster-stage-artefacts) or override via sbatch --export."
        echo   "       ⛔ Failing here, at job start, is deliberate: the alternative is"
        echo   "       discovering this inside the loader after the GPU was allocated."
        exit 1
    fi
    unset _v _p _chlu_missing
fi
echo "[env.sh] CHLU_CLAUDE_ROOT=$CHLU_CLAUDE_ROOT (gen artefact roots present)"
# --- END gen-wells-route-b ---------------------------------------------------
