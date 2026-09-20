#!/usr/bin/env bash
# w8 hub/encoder-loop -- THE FULL PER-SEED PIPELINE, CPU, resumable at every stage.
#
#   scripts/w8_encoder_loop_pipeline.sh <seed> <condition> <run-base> <score-base>
#
# One process = one seed.  Three of these run side by side, three threads each.
#
# ORDER IS LOAD-BEARING, twice:
#   * `alone_wc` trains LAST because it is the WALL-CLOCK match for `loop_swap` and
#     refuses to guess a step count before the treatment has run.  It needs `alone`'s
#     seconds/step too, so `alone` must precede it.
#   * `cache` precedes `rivals` and `score` because both read the cache.
#
# WHAT GETS THE FULL PIPELINE AND WHAT DOES NOT, declared here rather than discovered:
#   TRAIN + PROBE : alone, loop_nosw, loop_swap, alone_wc   (the probe needs no cache)
#   CACHE + RIVALS + SCORE : alone, loop_nosw, loop_swap
#   `alone_wc` is the COMPUTE control and the equal-seconds question is decided by its
#   training seconds and its probe margin, neither of which needs a scoring leg.  That
#   is a declared scope decision, not an omission, and the report says so.
set -uo pipefail

SEED="${1:?seed}"
COND="${2:?condition: val|test}"
RUN="${3:?run base}"
SCORE="${4:?score base}"

: "${THREADS:=3}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT

M=chlu.experiments.exp_w8_encoder_loop
TRAIN_ARMS=(alone loop_nosw loop_swap alone_wc)
SCORE_ARMS=(alone loop_nosw loop_swap)
: "${SCORE_ROWS:=pca_tensor rival_mlp_180k}"

say() { echo "[pipe s$SEED $(date -u +%H:%M:%SZ)] $*"; }
FAILED=()
run() {  # a failure is RECORDED LOUDLY and the pipeline continues; never swallowed
  say "---- $*"
  if ! "$@"; then
    say "!!!! FAILED: $*"
    FAILED+=("$*")
    return 1
  fi
}

say "=== warm start ==="
run python -u -m "$M" --stage warm --seed "$SEED" --out "$RUN" || exit 1

for arm in "${TRAIN_ARMS[@]}"; do
  say "=== train $arm ==="
  run python -u -m "$M" --stage train --seed "$SEED" --arm "$arm" --out "$RUN"
done

for arm in "${TRAIN_ARMS[@]}"; do
  say "=== probe $arm (a MARGIN over the random-init floor, never an absolute) ==="
  run python -u -m "$M" --stage probe --seed "$SEED" --arm "$arm" --out "$RUN"
done

for arm in "${SCORE_ARMS[@]}"; do
  say "=== cache $arm (480,000 rows, one CPU device, per-block sha256) ==="
  run python -u -m "$M" --stage cache --seed "$SEED" --arm "$arm" --out "$RUN" || continue
  say "=== rivals $arm (refit on THIS arm's features, w4_learners verbatim) ==="
  run python -u -m "$M" --stage rivals --seed "$SEED" --arm "$arm" --out "$RUN"
  say "=== score $arm on $COND (rows 3 and 4) ==="
  run bash "$(dirname "$0")/w8_encoder_loop_score.sh" \
      "$COND" "$SEED" "$arm" "$RUN" "$SCORE" $SCORE_ROWS
done

python -u -m "$M" --stage report --out "$RUN" || true

if [ "${#FAILED[@]}" -gt 0 ]; then
  say "done WITH FAILURES (${#FAILED[@]}): ${FAILED[*]}"
  exit 3
fi
say "done: seed $SEED clean"
