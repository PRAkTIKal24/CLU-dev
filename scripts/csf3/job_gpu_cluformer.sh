#!/bin/bash --login
# scripts/csf3/job_gpu_cluformer.sh — the TIER-III PILOT at 26-47 M on one A100.
#
# ⚠ STATUS: NOT SUBMITTED BY THE AUTHORING AGENT. `csf3.itservices.manchester.ac.uk`
#   does not resolve from the laptop without GlobalProtect VPN, which the agent
#   cannot establish (`ssh csf3` -> "Could not resolve hostname"). Everything
#   below is laptop-verified (`bash -n`, and the identical Python entry point is
#   exercised at toy scale) and is labelled UNTESTED-ON-CLUSTER.
#
# Submit (Head):
#   cd ~/scratch/CHLU && mkdir -p logs
#   # 0. ONE-OFF, SERIAL: stage enwik8 (compute nodes have internet, login nodes
#   #    do NOT). Never let N array tasks race the download.
#   sbatch --job-name=clu-stage -t 0:30:00 scripts/csf3/job_gpu_cluformer.sh   # STAGE_ONLY=1 default OFF -> see below
#   sbatch --export=ALL,STAGE_ONLY=1 -p serial -t 0:30:00 scripts/csf3/job_gpu_cluformer.sh
#   # 1. then the three seeds, <=4 concurrent (free-tier policy)
#   sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4 --mail-user=$CLU_MAIL \
#          -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
#   # 2. pull the artifacts
#   rsync -av csf3:~/scratch/CHLU/.claude/outputs/cluformer-pilot/ ./.claude/outputs/cluformer-pilot/
#
# BUDGET DECLARED BEFORE SUBMISSION (task section 0.2): <= 108 A100-hours total
#   = 3 arms x 3 seeds x <= 12 h. ⛔ If one arm x seed will not finish inside
#   `-t 12:00:00`, STOP and report — do not resubmit at a larger wallclock.
#   Cut order if exceeded: (1) the anytime curve D5, (2) the TTT arm at pilot
#   scale, (3) n_layers 12 -> 8 with d_model 512 -> 640. NEVER the seed count,
#   the swap control, the dyn-eval column, or a monitor.
#
# Partition facts (official CSF3 docs, gpu-jobs page): gpuA = A100 80GB, free at
# point of use, <=4 GPUs concurrently per user, <=12 host cores/GPU, max -t 4-0.
#
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 12:00:00               # OVERRIDE PER RUN. Max 4-0.
#SBATCH --job-name=clu-tier3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END,FAIL      # address at submit: sbatch --mail-user=$CLU_MAIL

module purge                      # jax[cuda12] wheels bundle CUDA/cuDNN
set -eo pipefail

STAGE="${STAGE:-pilot}"           # pilot (26-47 M) | toy
STG="${STG:-s4}"                  # s1 | s2 | s3 | s4
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-}"                # empty => the config's own step count
ARMS="${ARMS:-}"                  # empty => all five arms
OUT="${OUT:-.claude/outputs/cluformer-pilot}"
STAGE_ONLY="${STAGE_ONLY:-0}"     # 1 => fetch enwik8 and exit (run this FIRST)

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

# ---- step 0: download-once (serial), then exit ----------------------------
if [ "$STAGE_ONLY" = "1" ]; then
  echo "=== staging enwik8 (SERIAL, download-once) ==="
  python -c "from chlu.data.enwik8 import stage_enwik8; print('staged', stage_enwik8())"
  echo "=== stage-only DONE ==="
  exit 0
fi

echo "=== node/GPU provenance ==="
hostname
nvidia-smi || true
echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "=== JAX GPU preflight (fail fast rather than burn 12 h on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY

echo "=== enwik8 cache preflight (must already be staged by STAGE_ONLY=1) ==="
python - <<'PY'
from chlu.data.enwik8 import stage_enwik8
print("enwik8:", stage_enwik8(download=False))
PY

EXTRA=""
[ -n "$STEPS" ] && EXTRA="$EXTRA --steps $STEPS"
[ -n "$ARMS" ] && EXTRA="$EXTRA --arms $ARMS"

echo "=== tier-iii pilot: scale=$STAGE stage=$STG seeds=$SEEDS ==="
# shellcheck disable=SC2086
python -u -m chlu.experiments.exp_cluformer_pilot \
    --scale "$STAGE" --stage "$STG" --seeds $SEEDS --out "$OUT" $EXTRA

echo "=== artifacts ==="
ls -l "$OUT" || true
echo "Pull to laptop with:"
echo "  rsync -av csf3:$CLU_REPO/$OUT/ ./.claude/outputs/cluformer-pilot/"
