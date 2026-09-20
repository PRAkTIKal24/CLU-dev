#!/usr/bin/env bash
# w6 hub/potentials -- THE ONE LAUNCHER for the scored run.  Seeds sequentially.
#   scripts/w6_potentials_scored.sh <out-base> <prereg> [aux_corners]
set -euo pipefail
OUT="${1:?out base}"
PREREG="${2:?prereg file}"
AUX="${3:-4}"

export CHLU_GEN_BRIDGE_ROOT="C:\chlu\outputs\spoke-a-rivals\cache_generation2"
export CHLU_GEN_CLASSIFIER_ROOT="C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.claude\outputs\gen-external-classifier"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

for S in 0 1 2; do
  echo "=== SEED $S  $(date -u +%H:%M:%S) ==="
  python -u scripts/w6_potentials_leg.py \
    --condition test --seed "$S" --out "$OUT/s$S" \
    --corners nway --aux-corners "$AUX" \
    --arms field,landscape_removed,landscape_shuffled \
    --prereg "$PREREG"
done
echo "=== ALL SEEDS DONE $(date -u +%H:%M:%S) ==="
