#!/usr/bin/env bash
# SIDE TRACK, one process, 3 threads: `alone` is finished on every seed, so its cache and
# its refit baseline no longer need to wait behind loop_swap and alone_wc. Both stages are
# idempotent (each skips when its artefact exists), so the main pipeline reaching them
# later is a no-op rather than a conflict. This moves ~2 h off the critical path at the
# cost of ONE extra process.
set -uo pipefail
cd /c/chlu/worktrees/CHLU-w8-encoder-loop
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8
export CHLU_GEN_CLASSIFIER_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-external-classifier
RUN=/c/chlu/outputs/w8-encoder-loop/run
M=chlu.experiments.exp_w8_encoder_loop
for s in 0 1 2; do
  echo "[side $(date -u +%H:%M:%SZ)] cache alone seed $s"
  python -u -m "$M" --stage cache --seed "$s" --arm alone --out "$RUN" || echo "[side] cache s$s FAILED"
  echo "[side $(date -u +%H:%M:%SZ)] rivals alone seed $s"
  python -u -m "$M" --stage rivals --seed "$s" --arm alone --out "$RUN" --families mlp_180k \
    || echo "[side] rivals s$s FAILED"
done
echo "[side $(date -u +%H:%M:%SZ)] done"
