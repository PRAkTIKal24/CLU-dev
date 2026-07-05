#!/bin/bash --login
# scripts/csf3/job_gpu_array_seeds.sh — seed sweep as a Slurm job array,
# one A100 per task, one chlu project per seed (no artifact collisions).
#
# SUBMIT (5 seeds 0..4, exp-b full run, 1 day each):
#   cd ~/scratch/CHLU
#   sbatch --export=ALL,EXP=exp-b,PROJECT_PREFIX=csf3_b,SEED_BASE=0,EXTRA_ARGS= \
#          -t 1-0 scripts/csf3/job_gpu_array_seeds.sh
# Change the sweep width with:  sbatch -a 0-9 ... (task ids feed the seeds).
#
# Free-at-point-of-use limit is 4 concurrent A100s (docs, 2026-02-20 update):
# wider arrays simply queue — submit as many tasks as you like.
# Array syntax per official job-arrays page: #SBATCH -a, ${SLURM_ARRAY_TASK_ID}.
#
#SBATCH -p gpuA              # A100 (80GB) partition
#SBATCH -G 1                 # 1 GPU per array task
#SBATCH -n 8                 # host cores per task
#SBATCH -t 4:00:00           # OVERRIDE PER RUN. Max 4-0.
#SBATCH -a 0-4               # tasks 0..4 -> seeds SEED_BASE+0..4
#SBATCH --job-name=clu-sweep

module purge
set -eo pipefail

EXP="${EXP:-exp-a}"
PROJECT_PREFIX="${PROJECT_PREFIX:-csf3_sweep}"
SEED_BASE="${SEED_BASE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"              # e.g. --quick (see §7.10 caveat in job_gpu_single.sh)

SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID))
PROJECT="${PROJECT_PREFIX}_s${SEED}"

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

echo "=== array task ${SLURM_ARRAY_TASK_ID}: ${EXP} seed=${SEED} project=${PROJECT} ==="
nvidia-smi -L || true

python - <<'PY'
import jax
assert jax.default_backend() == "gpu", "JAX did not initialise the A100"
print("devices:", jax.devices())
PY

chlu project create "$PROJECT" --description "CSF3 sweep ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} ${EXP} seed=${SEED}" || true
# shellcheck disable=SC2086
chlu "$EXP" --project "$PROJECT" --seed "$SEED" $EXTRA_ARGS

echo "=== task ${SLURM_ARRAY_TASK_ID} done -> projects/${PROJECT} ==="
