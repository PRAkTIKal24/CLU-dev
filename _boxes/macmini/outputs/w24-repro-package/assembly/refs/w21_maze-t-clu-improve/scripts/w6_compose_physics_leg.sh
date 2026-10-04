#!/usr/bin/env bash
# w6 hub/compose-physics -- THE ONE LAUNCHER (w6-common section 5).
#
# One leg = one (condition, seed).  Every stage is resumable: each writes one artifact
# and skips when it is already on disk, so a kill costs one arm's corner and not the leg.
#
#   scripts/w6_compose_physics_leg.sh <condition> <seed> <out-base> [arm ...]
#
# CACHE GENERATION.  Every SCORED run is on generation 2 (w6-common section 1).  The
# leg REFUSES a `test` condition without the generation-2 root, because a scored number
# on generation 1 cannot be quoted beside a generation-2 rival (w6-common section 3.8).
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
  echo "[leg] Every scored run is on cache generation 2 (w6-common section 1)." >&2
  exit 2
else
  echo "[leg] cache root -> $CHLU_GEN_BRIDGE_ROOT (generation 1; DEV/val only)"
fi

# The held-out condition is a SCORED RUN under charter section A.2.  It does not start
# without a pre-registration file and a row in the ledger already on disk.
if [ "$COND" = "test" ]; then
  : "${PREREG:?condition=test is a SCORED RUN: set PREREG to the pre-registration file}"
  [ -f "$PREREG" ] || { echo "[leg] REFUSING: PREREG $PREREG does not exist" >&2; exit 2; }
  grep -Eq "w6[ /]hub/compose-physics|w6/compose-physics" /c/chlu/SCORED-LEDGER.md || {
    echo "[leg] REFUSING: no /c/chlu/SCORED-LEDGER.md row for w6 hub/compose-physics" >&2
    exit 2; }
  echo "[leg] scored run authorised by $PREREG + the ledger row"
fi

M=chlu.experiments.exp_w6_compose_physics

ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  ARMS=(pca_settle_zerostep pca_settle_onestep pca_settle pca_tensor)
fi

LEG="$OUT/${COND}_seed${SEED}"
mkdir -p "$LEG"
echo "[leg] condition=$COND seed=$SEED leg=$LEG arms=${ARMS[*]}"

python -u -m "$M" --stage pins \
  --out "$OUT" --seed "$SEED" --condition "$COND" | tail -3
python -u -m "$M" --stage prepare \
  --out "$OUT" --seed "$SEED" --condition "$COND"

# One arm's failure must not cost the others.  `set -e` aborted two scored legs in w5 on
# a single control's crash; the arms are independent and every stage is resumable, so a
# failure is RECORDED LOUDLY and the leg continues.  It is never swallowed: the arm gets
# a FAILED.json, the leg prints it, and the leg exits non-zero so a failure cannot look
# like a clean run.
FAILED=()
for arm in "${ARMS[@]}"; do
  echo "[leg] ---- $arm ----"
  ok=1
  python -u -m "$M" --stage fit \
    --out "$OUT" --seed "$SEED" --condition "$COND" --arm "$arm" || ok=0
  if [ "$ok" = 1 ]; then
    python -u -m "$M" --stage run \
      --out "$OUT" --seed "$SEED" --condition "$COND" --arm "$arm" \
      --max-seconds "$MAX_SECONDS" || ok=0
  fi
  if [ "$ok" = 0 ]; then
    echo "[leg] !!!! ARM FAILED: $arm -- continuing; see FAILED.json" >&2
    mkdir -p "$LEG/arms/$arm"
    printf '{"arm":"%s","failed":true,"seed":%s,"condition":"%s","note":"the stage raised; this is a FAILURE, not a declared NOT-RUN, and it is never a null"}\n' \
      "$arm" "$SEED" "$COND" > "$LEG/arms/$arm/FAILED.json"
    FAILED+=("$arm")
  fi
done

python -u -m "$M" --stage table --out "$OUT" --seed "$SEED" --condition "$COND" \
  --primary "${PRIMARY:-pca_settle}" --margin "${MARGIN:-1.0}"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[leg] done WITH FAILURES: $LEG -- failed arms: ${FAILED[*]}" >&2
  echo "[leg] done: $LEG"
  exit 3
fi
echo "[leg] done: $LEG"
