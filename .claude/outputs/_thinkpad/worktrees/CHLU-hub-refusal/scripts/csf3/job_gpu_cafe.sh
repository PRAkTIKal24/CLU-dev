#!/bin/bash --login
# scripts/csf3/job_gpu_cafe.sh — one CLU x CAFE-benchmark run on a single A100.
#
# Registers CLU as a `cafe-bench` model (chlu.eval.cafe_model) and runs one
# (dataset, model) evaluation through the CAFE harness, writing the harness's
# own results JSON (metrics + per-horizon AUROC + our config provenance).
#
# Parametrized via --export at submit time (defaults below), e.g.:
#   cd ~/scratch/CHLU
#   sbatch --mail-user=$CLU_MAIL scripts/csf3/job_gpu_cafe.sh                # FD001, defaults
#   sbatch --mail-user=$CLU_MAIL --export=ALL,DATASET=cmapss_fd002,SEED=43 \
#          scripts/csf3/job_gpu_cafe.sh
#
# The whole Event-Prediction task is only 6 datasets, so the full sweep is
# cheap — one job per (dataset, seed):
#   for D in cmapss_fd001 cmapss_fd002 cmapss_fd003 cmapss_fd004; do
#     for S in 42 43 44; do
#       sbatch --mail-user=$CLU_MAIL \
#              --export=ALL,DATASET=$D,SEED=$S,OUT=$HOME/scratch/clu_cafe/${D}_s${S} \
#              -t 2:00:00 scripts/csf3/job_gpu_cafe.sh
#     done
#   done
# then aggregate with `cafe-bench leaderboard --results-dir ...`.
#
# ⚠ CAFE IS A SEPARATE REPO — it is deliberately NOT vendored into CHLU.
#   Clone it ONCE on the cluster (login node has git+internet):
#     git clone git@github.com:Forgis-Labs/CAFE.git $HOME/scratch/cafe-bench
#   and fetch the datasets ONCE, serially, before launching any array
#   (same download-once discipline as job_gpu_eval.sh — N parallel jobs racing
#   a shared cache on the networked home FS is what killed the first flagship
#   launch). C-MAPSS is the exception that needs care: CAFE's own
#   scripts/download_all.py C-MAPSS path is DEAD (the NASA ti.arc.nasa.gov and
#   data.nasa.gov URLs both 404 as of 2026-07-20). Use the HuggingFace route:
#     python $CAFE_REPO/scripts/download_from_hf.py --datasets cmapss --root $CAFE_DATA
#
# Partition facts (CSF3 gpu-jobs page, mod. 2026-06-11): gpuA = A100 80GB, free
# at point of use, <=4 GPUs concurrently per user, batch wallclock max 4-0.
#
#SBATCH -p gpuA              # A100 (80GB) partition
#SBATCH -G 1                 # 1 GPU
#SBATCH -n 1                 # 1 task ...
#SBATCH -c 8                 # ... with 8 cores (<=12/GPU on CSF3; ~10.4GB RAM/core)
#SBATCH -t 2:00:00           # OVERRIDE PER RUN (sbatch -t ...). Max 4-0.
#SBATCH --job-name=clu-cafe
#SBATCH -o logs/%x-%j.out         # stdout -> logs/ (dir must exist)
#SBATCH -e logs/%x-%j.err         # stderr -> logs/ (separate stream)
#SBATCH --mail-type=END,FAIL      # mail on end/fail; set the address at submit:
#                                 #   sbatch --mail-user=$CLU_MAIL ... (no addr in-repo)

module purge                 # jax[cuda12] pip wheels bundle CUDA/cuDNN; only
                             # the node NVIDIA driver is required.
set -eo pipefail

# ---- knobs (override with sbatch --export=ALL,VAR=...) ---------------------
DATASET="${DATASET:-cmapss_fd001}"        # cmapss_fd001..4 | physionet_*
MODEL="${MODEL:-clu}"                     # clu | clu_valley
SEED="${SEED:-42}"
OUT="${OUT:-$HOME/scratch/clu_cafe/${DATASET}_s${SEED}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"              # e.g. --epochs 400 --relax-gamma 0.5

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
export CAFE_REPO="${CAFE_REPO:-$HOME/scratch/cafe-bench}"
export CAFE_DATA="${CAFE_DATA:-$HOME/scratch/cafe-data}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

if [ ! -d "$CAFE_REPO/cafe_bench" ]; then
  echo "ERROR: no CAFE checkout at CAFE_REPO=$CAFE_REPO" >&2
  echo "  git clone git@github.com:Forgis-Labs/CAFE.git $CAFE_REPO" >&2
  exit 2
fi
export PYTHONPATH="$CAFE_REPO:$CLU_REPO:${PYTHONPATH:-}"

echo "=== node/GPU provenance ==="
hostname
nvidia-smi || true
echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CAFE_REPO=$CAFE_REPO ($(git -C "$CAFE_REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "CLU_REPO=$CLU_REPO  ($(git -C "$CLU_REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"

echo "=== JAX GPU preflight (fail fast rather than score on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY

echo "=== CAFE preflight (lifelines backs the default CoxPH event probe) ==="
python - <<'PY'
import importlib, sys
missing = [m for m in ("lifelines", "cafe_bench") if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"missing required module(s): {missing}  (uv pip install lifelines)")
print("cafe_bench + lifelines OK")
PY

mkdir -p "$OUT"
echo "=== run: CLU x CAFE ${DATASET} model=${MODEL} seed=${SEED} ==="
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
python "$CLU_REPO/scripts/cafe/run_clu_cafe.py" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --data-root "$CAFE_DATA" \
    --results-dir "$OUT" \
    --seed "$SEED" $EXTRA_ARGS

echo "=== artifacts ==="
ls -R "$OUT" 2>/dev/null || true
echo "Pull to laptop with: rsync -avz csf3:${OUT}/ ./cafe_${DATASET}_s${SEED}/"
