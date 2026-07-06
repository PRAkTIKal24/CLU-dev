#!/bin/bash --login
# scripts/csf3/job_gpu_single.sh — one CLU experiment on a single A100 (80GB).
#
# Parametrized via --export at submit time (defaults below), e.g.:
#   cd ~/scratch/CHLU
#   sbatch scripts/csf3/job_gpu_single.sh                                # exp-a quick smoke
#   sbatch --export=ALL,EXP=exp-c,PROJECT=csf3_c1,SEED=42,EXTRA_ARGS= \
#          -t 1-0 scripts/csf3/job_gpu_single.sh                         # real exp-c run
#
# Partition facts (official CSF3 docs, gpu-jobs page, mod. 2026-06-11):
#   gpuA = A100 80GB, free at point of use, <=4 GPUs concurrently per user,
#   <=12 host cores per GPU (10.4GB RAM/core), batch wallclock max 4-0.
#
#SBATCH -p gpuA              # A100 (80GB) partition
#SBATCH -G 1                 # 1 GPU
#SBATCH -n 8                 # host cores (RAM scales with this: ~10.4GB/core)
#SBATCH -t 4:00:00           # OVERRIDE PER RUN (sbatch -t ...). Max 4-0.
#SBATCH --job-name=clu-gpu

module purge                 # no CUDA module needed: jax[cuda12] pip wheels
                             # bundle CUDA/cuDNN; the node driver (580.126.09)
                             # is all that's required.
set -eo pipefail

# ---- knobs (override with sbatch --export=ALL,VAR=...) ---------------------
EXP="${EXP:-exp-a}"                       # exp-a | exp-b | exp-c | all-experiments
PROJECT="${PROJECT:-csf3_smoke_a}"        # chlu project name (dir under projects/)
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:---quick}"       # default = quick smoke; set EXTRA_ARGS= for full
# NOTE (handover §7.10): --quick currently does NOT shorten the CHLU dynamics
# training loop for exp-a/b (train_chlu reads training.epochs=1000); it only
# shortens baselines/exp-c. Budget wallclock for a full dynamics train until
# fix-pack-2 lands.

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

echo "=== node/GPU provenance ==="
hostname
nvidia-smi || true
echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "=== JAX GPU preflight (fail fast rather than train on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY

echo "=== run: chlu ${EXP} --project ${PROJECT} --seed ${SEED} ${EXTRA_ARGS} ==="
# Idempotent project creation (create fails politely if it already exists).
chlu project create "$PROJECT" --description "CSF3 job ${SLURM_JOB_ID:-?} ${EXP} seed=${SEED}" || true
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
chlu "$EXP" --project "$PROJECT" --seed "$SEED" $EXTRA_ARGS

echo "=== artifacts ==="
ls -R "projects/$PROJECT/plots" "projects/$PROJECT/results" "projects/$PROJECT/models" 2>/dev/null || true
echo "Pull to laptop with: scripts/csf3/sync_project.sh pull $PROJECT"
