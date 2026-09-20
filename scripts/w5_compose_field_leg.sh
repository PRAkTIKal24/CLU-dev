#!/usr/bin/env bash
# w5 hub/compose-field -- THE ONE LAUNCHER (w5-common section 5).
#
# One leg = one (condition, seed).  Every stage is resumable: each writes one artifact
# and skips when it is already on disk, so a kill costs one arm's corner and not the leg.
# Memory and per-arm wall clock are capped on every call.
#
#   scripts/w5_compose_field_leg.sh <condition> <seed> <out-base> [arm ...]
#
# CACHE GENERATION.  Every SCORED run is on generation 2 (w5-common section 2).  The
# generation-2 root is published by spoke A as cache_gen2_root inside RIVALS-READY.json;
# the leg REFUSES a `test` condition without it, because a scored number on generation 1
# cannot be quoted beside a generation-2 rival.
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
# exp_gen_factor_wells.classifier_dir, so moving the cache to generation 2 would send
# the judge looking for a classifier that is not there.  Pin it explicitly; the FILE,
# and the hash Judge asserts against it, are unchanged.
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT
echo "[leg] classifier root -> $CHLU_GEN_CLASSIFIER_ROOT"

if [ -n "${CHLU_GEN_BRIDGE_ROOT_OVERRIDE:-}" ]; then
  export CHLU_GEN_BRIDGE_ROOT="$CHLU_GEN_BRIDGE_ROOT_OVERRIDE"
  echo "[leg] cache root -> $CHLU_GEN_BRIDGE_ROOT (generation 2)"
elif [ "$COND" = "test" ]; then
  echo "[leg] REFUSING: condition=test with no CHLU_GEN_BRIDGE_ROOT_OVERRIDE." >&2
  echo "[leg] Every scored run is on cache generation 2 (w5-common section 2)." >&2
  exit 2
else
  echo "[leg] cache root -> $CHLU_GEN_BRIDGE_ROOT (generation 1; DEV only)"
fi

# The held-out condition is a SCORED RUN under charter section A.2.  It does not start
# without a pre-registration file and a row in the ledger already on disk.
if [ "$COND" = "test" ]; then
  : "${PREREG:?condition=test is a SCORED RUN: set PREREG to the pre-registration file}"
  [ -f "$PREREG" ] || { echo "[leg] REFUSING: PREREG $PREREG does not exist" >&2; exit 2; }
  grep -q "w5/compose-field" /c/chlu/SCORED-LEDGER.md || {
    echo "[leg] REFUSING: no /c/chlu/SCORED-LEDGER.md row for w5/compose-field" >&2
    exit 2; }
  echo "[leg] scored run authorised by $PREREG + the ledger row"
fi

M=chlu.experiments.exp_w5_compose_field

ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  # THE GATE first, then the bar, then everything that shares the table for free.
  ARMS=(oracle_span_field oracle_span rival_mlp_180k field_lookup_argmax \
        rival_table rival_mlp rival_cont_mlp rival_transformer rival_lstm \
        rival_lace rival_poe_ebm \
        span_head lookup_argmax landscape_removed landscape_shuffled)
fi

LEG="$OUT/${COND}_seed${SEED}"
mkdir -p "$LEG"
echo "[leg] condition=$COND seed=$SEED leg=$LEG arms=${ARMS[*]}"

python -u -m "$M" --stage pins \
  --out "$OUT" --seed "$SEED" --condition "$COND" | tail -3
python -u -m "$M" --stage prepare \
  --out "$OUT" --seed "$SEED" --condition "$COND"

for arm in "${ARMS[@]}"; do
  echo "[leg] ---- $arm ----"
  python -u -m "$M" --stage fit \
    --out "$OUT" --seed "$SEED" --condition "$COND" --arm "$arm"
  python -u -m "$M" --stage run \
    --out "$OUT" --seed "$SEED" --condition "$COND" --arm "$arm" \
    --max-seconds "$MAX_SECONDS"
done

python -u -m "$M" --stage table --out "$OUT" --seed "$SEED" --condition "$COND" \
  --primary "${PRIMARY:-oracle_span_field}" --margin "${MARGIN:-1.0}"

echo "[leg] done: $LEG"
