#!/usr/bin/env bash
# w15 spoke/bar-d-score -- one seed of the GPU leg's CPU-side stages, resumable.
#   scripts/w15_bar_d_pipeline.sh <seed> <phase: prep|val|test> <run-base> <score-base>
# prep : cache (alone, loop_nosw, loop_swap) + rivals mlp_180k (alone, loop_swap)
# val  : score on val   (free; the PREREG's V1-V3 gate reads it)
# test : score on test  (SCORED; refused without w15's PREREG + ledger row)
# The GPU pins are passed to every stage so the log records them; the stages here do not
# train, so they change nothing -- the scored arms' own arm.json is what BAR-D.json quotes.
set -uo pipefail
SEED="${1:?seed}"; PHASE="${2:?phase}"; RUN="${3:?run base}"; SCORE="${4:?score base}"
: "${THREADS:=3}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS" JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
M=chlu.experiments.exp_w8_encoder_loop
GPU_PINS=(--joint-steps 4000 --n-train 32768 --dec-steps 1800 --warm-steps 4000)
FULL="pca_tensor rival_mlp_180k pca_landscape_removed pca_landscape_shuffled pca_random_refit pca_lookup_argmax"
say() { echo "[w15 s$SEED $(date -u +%H:%M:%SZ)] $*"; }
FAILED=()
run() { say "---- $*"; "$@" || { say "!!!! FAILED: $*"; FAILED+=("$*"); return 1; }; }
if [ "$PHASE" = prep ]; then
  for arm in alone loop_swap loop_nosw; do
    run python -u -m $M --stage cache --seed "$SEED" --arm "$arm" --out "$RUN" "${GPU_PINS[@]}"
  done
  for arm in alone loop_swap; do
    run python -u -m $M --stage rivals --seed "$SEED" --arm "$arm" --out "$RUN" --families mlp_180k "${GPU_PINS[@]}"
  done
elif [ "$PHASE" = val ]; then
  for arm in alone loop_swap; do
    run bash "$(dirname "$0")/w15_bar_d_score.sh" val "$SEED" "$arm" "$RUN" "$SCORE" pca_tensor rival_mlp_180k pca_landscape_removed
  done
elif [ "$PHASE" = test ]; then
  for arm in alone loop_swap; do
    run bash "$(dirname "$0")/w15_bar_d_score.sh" test "$SEED" "$arm" "$RUN" "$SCORE" $FULL
  done
  run bash "$(dirname "$0")/w15_bar_d_score.sh" test "$SEED" loop_nosw "$RUN" "$SCORE" pca_tensor
fi
[ "${#FAILED[@]}" -gt 0 ] && { say "done WITH FAILURES: ${FAILED[*]}"; exit 3; }
say "done: $PHASE clean"
