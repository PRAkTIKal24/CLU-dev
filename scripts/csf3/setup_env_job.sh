#!/bin/bash --login
# scripts/csf3/setup_env_job.sh — one-time environment build for CLU on CSF3.
#
# WHY A BATCH JOB: CSF3 login nodes cannot reach off-campus sites; compute
# nodes can (official docs, "Access to External Websites and Repositories
# from the CSF"). All downloads (uv installer, CPython, wheels) therefore
# happen inside this job, not on the login node.
#
# PREREQ: the repo (INCLUDING uv.lock — it is gitignored, so it only arrives
# via rsync, see push_repo.sh) is at $CLU_REPO (default ~/scratch/CHLU).
#
# SUBMIT (from the repo root on the CSF3 login node):
#     cd ~/scratch/CHLU && sbatch scripts/csf3/setup_env_job.sh
# Re-running is safe: uv sync is idempotent and repairs a partial venv.
#
#SBATCH -p serial            # 1-core CPU partition (Intel), 7-0 max (no -n/-c/-G)
#SBATCH -t 2:00:00           # REQUIRED (no default); generous for ~6GB of CUDA wheels
#SBATCH --job-name=clu-env
#SBATCH -o logs/%x-%j.out         # stdout -> logs/ (dir must exist)
#SBATCH -e logs/%x-%j.err         # stderr -> logs/ (separate stream)
#SBATCH --mail-type=END,FAIL      # mail on end/fail; set the address at submit:
#                                 #   sbatch --mail-user=$CLU_MAIL ... (no addr in-repo)

module purge
set -eo pipefail

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
cd "$CLU_REPO"

# --- uv caches on scratch (CUDA wheel cache is multi-GB; home quota is small).
# CAVEAT: scratch files unused for 3 months can be auto-deleted -> a stale env
# may lose its interpreter. Cheap fix: re-run this job to rebuild.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/scratch/.uv_pythons}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

# --- 1. Install uv (standalone, ~/.local/bin) if missing -------------------
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 2. Build the venv from the lockfile -----------------------------------
# Python 3.11 to match the laptop dev env (=> jax 0.9.0 branch of the lock).
# --frozen: install EXACTLY the rsynced uv.lock (bit-parity with the laptop
# resolution). If the lock is missing (e.g., bare git clone), we resolve
# fresh and warn loudly — versions may then drift from the laptop.
if [ -f uv.lock ]; then
    if ! grep -q "jax-cuda12" uv.lock; then
        # Lock predates the cuda extra: refresh it here (compute node has
        # internet). uv lock is additive — existing pins are preserved.
        echo "[setup] uv.lock lacks the cuda extra — running additive uv lock"
        uv lock
    fi
    echo "[setup] uv sync --frozen from uv.lock (cuda + eval extras)"
    # --extra eval carries the industrial-loader deps (pandas/pyarrow/pyreadr);
    # without it voraus/TEP cannot even read their data on a compute node
    # (voraus-baseline-floors blocker 1). The eval extra pulls pure-Python /
    # already-pinned wheels only — it does NOT bump jax off the 0.9.0 pin
    # (verify with the sanity check below).
    uv sync --frozen --extra cuda --extra eval --python 3.11
else
    echo "[setup] WARNING: no uv.lock found — resolving fresh (env may drift"
    echo "[setup] from the laptop). Prefer push_repo.sh which carries the lock."
    uv sync --extra cuda --extra eval --python 3.11
fi

# --- 3. CPU-side sanity check (this is a non-GPU node: force cpu backend; ---
# ---    the GPU backend is exercised by job_gpu_single.sh / smoke test #3) --
# shellcheck disable=SC1091
source .venv/bin/activate
JAX_PLATFORMS=cpu python - <<'PY'
import importlib.metadata as md
import jax, chlu  # noqa: F401  (import = editable install works)
print("python OK; jax", jax.__version__, "| chlu", md.version("chlu"))
print("cpu devices:", jax.devices())
# The eval extra must NOT bump jax off the CSF3 pin (0.9.0). Fail loudly if it did.
assert jax.__version__.startswith("0.9."), (
    f"jax pin drifted to {jax.__version__} (expected 0.9.x) — the eval extra "
    "must not upgrade jax; check uv.lock resolution."
)
# The industrial loaders need the eval extra (pandas/pyarrow; pyreadr for TEP).
for pkg in ("pandas", "pyarrow", "pyreadr"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg} MISSING — eval extra did not install (voraus/TEP will not load)!")
for pkg in ("jax-cuda12-plugin", "jax-cuda12-pjrt", "equinox", "optax", "diffrax"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg} MISSING — CUDA extra did not install!")
import jax.numpy as jnp
print("jit smoke:", jax.jit(lambda x: (x * 2).sum())(jnp.ones(4)))
PY

echo "[setup] DONE. Next: sbatch scripts/csf3/job_cpu_smoke.sh"
