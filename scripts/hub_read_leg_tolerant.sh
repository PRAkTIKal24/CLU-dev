#!/usr/bin/env bash
# hub/read -- THE ONE LAUNCHER (w4-common section 6).
#
# One leg = one (condition, seed).  Every stage is resumable: each writes one artifact
# and skips when it is already complete, so a kill costs one arm's corner and not the
# leg.  Memory and per-arm wall clock are capped on every call.
#
#   scripts/hub_read_leg.sh <condition> <seed> <out-base> [arm ...]
#
# CACHE GENERATION.  ⛔ Every SCORED run uses generation 2.  Pass the generation-2 root
# in CHLU_GEN_BRIDGE_ROOT_OVERRIDE (spoke A publishes it as cache_gen2_root inside
# RIVALS-READY.json); the leg refuses to run a `test` condition without it, because a
# scored number on generation 1 cannot be quoted beside a generation-2 rival.
set -euo pipefail

COND="${1:?condition: val|test}"
SEED="${2:?seed}"
OUT="${3:?out base}"
shift 3 || true

: "${MAX_SECONDS:=2700}"        # per-arm wall clock cap, honoured between corners
: "${THREADS:=3}"               # this box gives the hub 3 threads
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export XLA_FLAGS="--xla_force_host_platform_device_count=1 ${XLA_FLAGS:-}"
export JAX_PLATFORMS=cpu

# The external classifier's location is DERIVED from the cache path by
# exp_gen_factor_wells.classifier_dir, so moving the cache to generation 2 would send the
# judge looking for a classifier that is not there.  Pin it explicitly; the file, and its
# asserted hash, are unchanged.
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT
echo "[leg] classifier root -> $CHLU_GEN_CLASSIFIER_ROOT"

if [ -n "${CHLU_GEN_BRIDGE_ROOT_OVERRIDE:-}" ]; then
  export CHLU_GEN_BRIDGE_ROOT="$CHLU_GEN_BRIDGE_ROOT_OVERRIDE"
  echo "[leg] cache root -> $CHLU_GEN_BRIDGE_ROOT (generation 2)"
elif [ "$COND" = "test" ]; then
  echo "[leg] REFUSING: condition=test with no CHLU_GEN_BRIDGE_ROOT_OVERRIDE." >&2
  echo "[leg] Every scored run is on cache generation 2 (w4-common section 2)." >&2
  exit 2
else
  echo "[leg] cache root -> $CHLU_GEN_BRIDGE_ROOT (generation 1; DEV only)"
fi

ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  ARMS=(span_head tensor_addr tensor_addr_sum pvm_span pvm_table oracle_span \
        landscape_removed landscape_shuffled lookup_argmax lookup_convex \
        corr_linear knn_proto)
fi

LEG="$OUT/${COND}_seed${SEED}"
mkdir -p "$LEG"
echo "[leg] condition=$COND seed=$SEED leg=$LEG arms=${ARMS[*]}"

python -u -m chlu.experiments.exp_gen_read --stage pins \
  --out "$OUT" --seed "$SEED" --condition "$COND" | tail -1
python -u -m chlu.experiments.exp_gen_read --stage prepare \
  --out "$OUT" --seed "$SEED" --condition "$COND"

for arm in "${ARMS[@]}"; do
  echo "[leg] ---- $arm ----"
  # One arm's failure must not cost the leg: every stage is resumable, the failure
  # is loud in the log, and an arm with no rows.npz does not appear on the table.
  if ! python -u -m chlu.experiments.exp_gen_read --stage fit \
       --out "$OUT" --seed "$SEED" --condition "$COND" --arm "$arm"; then
    echo "[leg] ARM FAILED (fit): $arm"
  fi
  if ! python -u -m chlu.experiments.exp_gen_read --stage run \
       --out "$OUT" --seed "$SEED" --condition "$COND" --arm "$arm" \
       --max-seconds "$MAX_SECONDS"; then
    echo "[leg] ARM FAILED (run): $arm"
  fi
done
# the table: every arm classed, the paired estimand under the PREREG's margin
python -u -m chlu.experiments.exp_gen_read --stage table   --out "$OUT" --seed "$SEED" --condition "$COND"   --primary "${PRIMARY:-tensor_addr}" --margin "${MARGIN:-1.0}"

echo "[leg] done: $LEG"
