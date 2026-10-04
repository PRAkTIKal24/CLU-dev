#!/usr/bin/env bash
# One SEQUENTIAL scored-leg worker: runs the (seed, arm) pairs given to it, one at a time.
set -uo pipefail
W13=/c/chlu/outputs/w13-encoder-adapt
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd /c/chlu/worktrees/CHLU-w13-encoder-adapt
for J in "$@"; do
  S="${J%%:*}"; A="${J##*:}"
  if [ -f "$W13/score/test_seed${S}_${A}_K32/score.json" ]; then
    echo "[scored] seed=$S arm=$A already on disk -- SKIPPED"; continue
  fi
  echo "[scored] $(date -u +%H:%M:%SZ) START seed=$S arm=$A"
  python -u scripts/w13_score_leg.py --arm "$A" --seed "$S" --condition test --run-n 1 \
      --run-base "C:/chlu/outputs/w13-encoder-adapt/run" \
      --out "C:/chlu/outputs/w13-encoder-adapt/score" \
      > "$W13/logs/score_s${S}_${A}.log" 2>&1
  echo "[scored] $(date -u +%H:%M:%SZ) END   seed=$S arm=$A exit=$?"
done
echo "[scored] WORKER DONE $(date -u +%H:%M:%SZ)"
