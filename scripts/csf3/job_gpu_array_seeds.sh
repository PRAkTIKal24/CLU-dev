#!/bin/bash --login
# scripts/csf3/job_gpu_array_seeds.sh — seed sweep as a Slurm job array,
# one A100 per task, one chlu project per seed (no artifact collisions).
#
# SUBMIT (5 seeds 0..4, exp-b full run, 1 day each):
#   cd ~/scratch/CHLU
#   sbatch --export=ALL,EXP=exp-b,PROJECT_PREFIX=csf3_b,SEED_BASE=0,EXTRA_ARGS= \
#          -t 1-0 scripts/csf3/job_gpu_array_seeds.sh
# Change the sweep width / throttle with:  sbatch -a 0-9%4 ... (overrides the
# directive below; task ids feed the seeds, %N caps concurrency).
#
# Free-at-point-of-use limit is 4 concurrent A100s (docs, 2026-02-20 update):
# the `%4` throttle below keeps the array within that cap (override with
# `sbatch -a 0-N%M`). Array syntax per the official job-arrays page.
#
#SBATCH -p gpuA              # A100 (80GB) partition
#SBATCH -G 1                 # 1 GPU per array task
#SBATCH -n 1                 # 1 task ...
#SBATCH -c 8                 # ... with 8 cores per array task (<=12/GPU)
#SBATCH -t 4:00:00           # OVERRIDE PER RUN. Max 4-0.
#SBATCH -a 0-4%4             # tasks 0..4 -> seeds SEED_BASE+0..4; %4 = <=4 concurrent
#                            #   (CSF3 4-GPU/user cap; override: sbatch -a 0-N%M)
#SBATCH --job-name=clu-sweep
#SBATCH -o logs/%x-%A_%a.out      # per-task stdout -> logs/ (dir must exist)
#SBATCH -e logs/%x-%A_%a.err      # per-task stderr -> logs/ (separate stream)
#SBATCH --mail-type=END,FAIL      # mail on end/fail; set the address at submit:
#                                 #   sbatch --mail-user=$CLU_MAIL ... (no addr in-repo)

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
