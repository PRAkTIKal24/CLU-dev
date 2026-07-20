#!/bin/bash --login
# scripts/csf3/job_gpu_eval.sh — one `chlu eval` run on a single A100 (80GB).
#
# The CLU->harness bridge on the cluster: scores a real anomaly dataset with
# the CLU arms (energy/residual + predict) alongside the four mandatory
# statistical baselines, writing the EvalRunResult npz + markdown + raw/ROC
# npz under ~/scratch.
#
# Parametrized via --export at submit time (defaults below), e.g.:
#   cd ~/scratch/CHLU
#   sbatch scripts/csf3/job_gpu_eval.sh                                   # voraus smoke
#   sbatch --export=ALL,DATASET=voraus,SCORE_MODE=default,SEED=42,EXTRA_ARGS='--download' \
#          -t 1-0 scripts/csf3/job_gpu_eval.sh                            # full voraus run
#
# G7b FLAGSHIP — literal joint-angle->so2-coset torus map (voraus T^6=U(1)^6):
#   the 6 joint angles are cos/sin-embedded (VorausTorusAD) and each feeds one
#   so2_invariant coset unit coupled on the arm's kinematic chain (ring=1-D
#   torus, channel_spring, U(1)-preserving). Both score arms, 3 seeds.
#
#   ⚠ DOWNLOAD-ONCE (do NOT put --download on the parallel jobs). The dataset
#   cache is a SHARED path on the networked home FS; N parallel `--download`
#   jobs racing it is what killed 5/6 of the first flagship launch. Instead
#   fetch + verify ONCE, serially, on an internet-capable compute node, THEN
#   launch the eval array with NO --download:
#     # (a) one-time serial fetch (setup_env_job.sh's node has internet too):
#     sbatch --export=ALL,DATASET=voraus,EXTRA_ARGS='--fetch-only' \
#            -t 0:30:00 scripts/csf3/job_gpu_eval.sh
#     # (b) then the parallel eval jobs — note: NO --download below:
#     for S in 42 43 44; do
#       sbatch --export=ALL,DATASET=voraus,SCORE_MODE=default,SEED=$S,\
# OUT=$HOME/scratch/clu_eval/voraus_torus_s$S,\
# EXTRA_ARGS='--lattice --lattice-layout literal --lattice-topology ring --window 100 --train-stride 10 --stride 5 --metrics-mode fast --max-train-windows 100000' \
#              -t 12:00:00 scripts/csf3/job_gpu_eval.sh
#     done
#   (`download_file` is now concurrency-safe — unique-temp + atomic-rename —
#   so a stray --download can no longer corrupt the cache; the fetch-once
#   pattern above still avoids N redundant ~1 GB pulls.)
#   TOPOLOGY-MATCH CONTROL (pre-registered falsifier): identical, add
#   --lattice-shuffle-angles --lattice-shuffle-seed $S, OUT=..._shuf_s$S.
#   Size (voraus-baseline-floors): ~2.5GB data (torus-embed nets +6 ch),
#   train_stride=10 avoids the ~49GB OOM; KNN/LOF scoring is the wall driver
#   (test_stride=5, ~near-lossless for episode mean-reduce). gpuA -c8 has
#   ~83GB RAM — ample. Episode AUC-ROC is primary (voraus is episode-labelled).
#
# Seed sweeps: mirror job_gpu_array_seeds.sh (#SBATCH -a 0-4, SEED from the
# task id); aggregate with a dependent `sbatch --dependency=afterany:<jobid>`
# collector, following the Head's sample_script_csf3.sh array+afterany pattern
# (NOT its HEPA specifics).
#
# Partition facts (CSF3 gpu-jobs page, mod. 2026-06-11): gpuA = A100 80GB, free
# at point of use, <=4 GPUs concurrently per user, batch wallclock max 4-0.
#
#SBATCH -p gpuA              # A100 (80GB) partition
#SBATCH -G 1                 # 1 GPU
#SBATCH -n 1                 # 1 task ...
#SBATCH -c 8                 # ... with 8 cores (<=12/GPU on CSF3; ~10.4GB RAM/core)
#SBATCH -t 4:00:00           # OVERRIDE PER RUN (sbatch -t ...). Max 4-0.
#SBATCH --job-name=clu-eval
#SBATCH -o logs/%x-%j.out         # stdout -> logs/ (dir must exist)
#SBATCH -e logs/%x-%j.err         # stderr -> logs/ (separate stream)
#SBATCH --mail-type=END,FAIL      # mail on end/fail; set the address at submit:
#                                 #   sbatch --mail-user=$CLU_MAIL ... (no addr in-repo)

module purge                 # jax[cuda12] pip wheels bundle CUDA/cuDNN; only
                             # the node NVIDIA driver is required.
set -eo pipefail

# ---- knobs (override with sbatch --export=ALL,VAR=...) ---------------------
DATASET="${DATASET:-voraus}"              # voraus | skab | tep | smd
SCORE_MODE="${SCORE_MODE:-default}"       # default | all | energy | residual | predict | hybrid
SEED="${SEED:-42}"
OUT="${OUT:-$HOME/scratch/clu_eval/${DATASET}_s${SEED}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"              # e.g. --download --limit 50 --lattice

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

echo "=== node/GPU provenance ==="
hostname
nvidia-smi || true
echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "=== JAX GPU preflight (fail fast rather than score on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY

mkdir -p "$OUT"
echo "=== run: chlu eval --dataset ${DATASET} --score-mode ${SCORE_MODE} --seed ${SEED} --out ${OUT} ${EXTRA_ARGS} ==="
# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
chlu eval --dataset "$DATASET" --score-mode "$SCORE_MODE" --seed "$SEED" --out "$OUT" $EXTRA_ARGS

echo "=== artifacts ==="
ls -R "$OUT" 2>/dev/null || true
echo "Pull to laptop with: rsync -avz csf3:${OUT}/ ./eval_${DATASET}_s${SEED}/"
