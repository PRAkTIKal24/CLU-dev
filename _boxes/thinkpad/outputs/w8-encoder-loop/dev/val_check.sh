#!/usr/bin/env bash
# THE DECLARED val FLOOR-EFFECT CHECK (PREREG section 3), fired the moment seed 0's
# loop_swap cache exists. It answers ONE question, on the SELECTION condition, before
# any held-out number is allowed to exist:
#
#   is pca_tensor's row-4 number NON-DEGENERATE under the shortened `w8_v1_short`
#   renderer, or has cutting the decoder collapsed every arm to zero and made the
#   whole comparison vacuous?
#
# A degenerate number here is reported as "the comparison cannot be made at this
# compute" -- never quietly, and never by carrying on to spend a scored run.
set -uo pipefail
RUN=/c/chlu/outputs/w8-encoder-loop/run
SCORE=/c/chlu/outputs/w8-encoder-loop/valscore
CACHE="$RUN/seed_0/loop_swap/cache/shapes3d_recon/seed_0/features.npy"
DECL="$RUN/seed_0/loop_swap/declift/winner_hashes.json"
echo "[val] waiting for seed 0 loop_swap cache ..."
for _ in $(seq 1 720); do
  [ -f "$CACHE" ] && [ -f "$DECL" ] && [ -f "$RUN/seed_0/loop_swap/cache.json" ] && break
  sleep 30
done
[ -f "$RUN/seed_0/loop_swap/cache.json" ] || { echo "[val] cache never appeared -- NOT RUN"; exit 2; }
echo "[val] cache present at $(date -u +%H:%M:%SZ); running the val leg (pca_tensor only)"
export MAX_SECONDS=2700 THREADS=2
exec bash scripts/w8_encoder_loop_score.sh val 0 loop_swap "$RUN" "$SCORE" pca_tensor
