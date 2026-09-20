#!/usr/bin/env bash
# w8 hub/encoder-loop -- THE ROWS-3/4 SCORING LEG, one (encoder arm, seed) at a time.
#
#   scripts/w8_encoder_loop_score.sh <condition> <seed> <arm> <run-base> <out-base> [score-arm ...]
#
# ⭐ THE WHOLE POINT OF THIS SCRIPT IS THE FOUR ENV HOOKS.  It runs the UNMODIFIED w5 leg
# (`exp_w5_compose_field` -> `exp_gen_read` -> `exp_gen_factor_wells`) and changes nothing
# in it; it only points the harness's own documented redirections at ONE encoder arm:
#
#   CHLU_GEN_BRIDGE_ROOT_OVERRIDE -> that arm's feature cache   (so `pca_tensor` is built
#                                    from that arm's features, and so is every rival row)
#   CHLU_GEN_DECLIFT_DIR          -> that arm's own refit decoder (an unfrozen encoder's
#                                    features are NOT the frozen decoder's input space)
#   CHLU_RIVALS_ROOT              -> that arm's REFIT learned baselines (spoke A's are
#                                    fitted in the FROZEN encoder's feature space and
#                                    would be meaningless here; they are never touched)
#   CHLU_GEN_CLASSIFIER_ROOT      -> ⭐ THE FROZEN CLASSIFIER. ⛔ THIS ONE DOES NOT MOVE.
#                                    It reads IMAGES, so it is the one instrument that
#                                    survives a change of feature space, and it is what
#                                    makes the arms comparable at all.
#
# ⛔ `test` is a SCORED RUN under charter A.2 and the underlying leg refuses it without a
# PREREG file and a ledger row already on disk.  That refusal is inherited deliberately.
set -euo pipefail

COND="${1:?condition: val|test}"
SEED="${2:?seed}"
ARM="${3:?encoder arm}"
RUN="${4:?run base (the --out of exp_w8_encoder_loop)}"
OUT="${5:?score out base}"
shift 5 || true

: "${THREADS:=3}"
: "${MAX_SECONDS:=2700}"
ARMDIR="$RUN/seed_${SEED}/${ARM}"

for need in "$ARMDIR/cache/shapes3d_recon/seed_${SEED}/features.npy" \
            "$ARMDIR/declift/winner_hashes.json" ; do
  [ -f "$need" ] || { echo "[score] REFUSING: $need is missing -- run the cache stage first" >&2; exit 2; }
done

export CHLU_GEN_BRIDGE_ROOT_OVERRIDE="$ARMDIR/cache"
export CHLU_GEN_DECLIFT_DIR="$ARMDIR/declift"
export CHLU_RIVALS_ROOT="$ARMDIR/rivals"
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT
export MAX_SECONDS THREADS

echo "[score] cond=$COND seed=$SEED encoder-arm=$ARM"
echo "[score]   features  -> $CHLU_GEN_BRIDGE_ROOT_OVERRIDE"
echo "[score]   decoder   -> $CHLU_GEN_DECLIFT_DIR   (this arm's own refit; NOT the gated winner)"
echo "[score]   rivals    -> $CHLU_RIVALS_ROOT       (refit on THIS arm's features)"
echo "[score]   judge     -> $CHLU_GEN_CLASSIFIER_ROOT   *** INVARIANT, does not move ***"

SCORE_ARMS=("$@")
if [ "${#SCORE_ARMS[@]}" -eq 0 ]; then
  # THE ARM, then THE BAR, then the floors and controls that can retire it, then the rest
  # of the learned baselines.  Order matters only for what lands first if the leg is
  # killed; every stage is resumable at the arm.
  SCORE_ARMS=(pca_tensor rival_mlp_180k
              pca_lookup_argmax pca_landscape_removed
              rival_cont_mlp rival_transformer rival_lstm)
fi

export PRIMARY=pca_tensor
export MARGIN=1.0
exec bash "$(dirname "$0")/w5_compose_field_leg.sh" \
     "$COND" "$SEED" "$OUT/${ARM}" "${SCORE_ARMS[@]}"
