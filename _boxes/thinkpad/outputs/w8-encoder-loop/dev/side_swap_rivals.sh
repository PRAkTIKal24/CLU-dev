#!/usr/bin/env bash
# The loop_swap refit baselines, run NOW in parallel with the val gate. Fitting a
# baseline on TRAINING features touches no held-out corner, so it is not part of the
# scored run and does not need the val gate to clear first. Idempotent: the score phase
# reaching this stage later is a no-op.
set -uo pipefail
cd /c/chlu/worktrees/CHLU-w8-encoder-loop
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8
export CHLU_GEN_CLASSIFIER_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-external-classifier
RUN=/c/chlu/outputs/w8-encoder-loop/run
M=chlu.experiments.exp_w8_encoder_loop
for s in "$@"; do
  echo "[swaprivals $(date -u +%H:%M:%SZ)] rivals loop_swap seed $s"
  python -u -m "$M" --stage rivals --seed "$s" --arm loop_swap --out "$RUN" --families mlp_180k \
    || echo "[swaprivals] s$s FAILED"
done
echo "[swaprivals $(date -u +%H:%M:%SZ)] done"
