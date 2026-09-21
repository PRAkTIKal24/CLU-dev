#!/usr/bin/env bash
# w13 hub/encoder-adapt -- the whole pipeline for ONE (seed, arm), resumable at every stage.
#
#   scripts/w13_arm_pipeline.sh <out> <seed> <arm> [joint_steps] [dec_steps]
#
# Every stage is idempotent: a completed artefact short-circuits, so a killed run is
# restarted by re-issuing the identical command.  The stages, in order:
#
#   train   the arm's adaptation from the SHARED warm start + its own render decoder
#   cache   a full-scope 480,000-row GENERATION-2 feature cache from THIS arm's encoder
#   probe   the factor probe, reported as a MARGIN over a random-init encoder
#   rivals  the learned baselines REFIT on this arm's own features (protocol verbatim)
#
# The scored leg is a separate script (`w13_score_leg.py`) because it is the one that
# spends a run.
set -uo pipefail

OUT="${1:?out dir}"
SEED="${2:?seed}"
ARM="${3:?arm}"
JS="${4:-}"
DS="${5:-}"
LOGD="$(dirname "$OUT")/logs"
mkdir -p "$LOGD"

EXTRA=()
[ -n "$JS" ] && EXTRA+=(--joint-steps "$JS")
[ -n "$DS" ] && EXTRA+=(--dec-steps "$DS")

export PYTHONIOENCODING=utf-8

for STAGE in train cache probe rivals; do
  L="$LOGD/${ARM}_s${SEED}_${STAGE}.log"
  echo "[pipeline] $(date -u +%H:%M:%SZ) seed=$SEED arm=$ARM stage=$STAGE -> $L"
  python -u -m chlu.experiments.exp_w13_encoder_adapt \
      --stage "$STAGE" --seed "$SEED" --arm "$ARM" --out "$OUT" "${EXTRA[@]}" \
      >> "$L" 2>&1
  RC=$?
  echo "[pipeline] $(date -u +%H:%M:%SZ) seed=$SEED arm=$ARM stage=$STAGE exit=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "[pipeline] STOPPING this lane: $STAGE failed (see $L)"
    exit "$RC"
  fi
done
echo "[pipeline] $(date -u +%H:%M:%SZ) seed=$SEED arm=$ARM COMPLETE"
