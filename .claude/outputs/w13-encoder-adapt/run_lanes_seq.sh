#!/usr/bin/env bash
set -uo pipefail
W13=/c/chlu/outputs/w13-encoder-adapt
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd /c/chlu/worktrees/CHLU-w13-encoder-adapt
for J in "$@"; do
  S="${J%%:*}"; A="${J##*:}"
  bash scripts/w13_arm_pipeline.sh "C:/chlu/outputs/w13-encoder-adapt/run" "$S" "$A" \
       "mlp,mlp_180k,cont_mlp,transformer,lstm" > "$W13/logs/lane_s${S}_${A}.log" 2>&1
  echo "[lanes] $(date -u +%H:%M:%SZ) seed=$S arm=$A exit=$?"
done
echo "[lanes] WORKER DONE $(date -u +%H:%M:%SZ)"
