#!/usr/bin/env bash
# w8 hub/encoder-loop -- THE ONE LAUNCHER (w8-common section 5: one launcher per hub).
#
#   scripts/w8_encoder_loop_leg.sh <seed> <out-base> [arm ...]
#
# One leg = one SEED, every arm in sequence.  Every stage is resumable: each writes one
# artefact and skips when it is already on disk, so a kill costs one stage and not the leg.
#
# The arm order is not cosmetic.  `loop_swap` runs BEFORE `alone_wc`, because `alone_wc`
# is the wall-clock match and cannot know its step count until the treatment has run;
# it refuses rather than guesses.  `alone` runs first because it is the control the
# target names and a leg that dies early should still have left the control on disk.
set -euo pipefail

SEED="${1:?seed}"
OUT="${2:?out base}"
shift 2 || true

: "${THREADS:=3}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
# MEASURED: Windows' default stdout codec is cp1252 and the frozen harness prints a
# star in gen_bridge._announce_encode_precision. Redirected to a file that raises
# UnicodeEncodeError and kills the stage AFTER its training has finished -- which is the
# worst possible place. Forced to UTF-8 rather than by editing a frozen file.
export PYTHONIOENCODING=utf-8
export XLA_FLAGS="--xla_force_host_platform_device_count=1 ${XLA_FLAGS:-}"

# The external classifier's location is DERIVED from the bridge path.  This hub moves the
# FEATURES per arm and must therefore pin the judge explicitly; the FILE, and the hash it
# is asserted against, are unchanged.  It is THE invariant instrument of this hub.
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT
echo "[leg] seed=$SEED out=$OUT threads=$THREADS"
echo "[leg] judge (INVARIANT) -> $CHLU_GEN_CLASSIFIER_ROOT"

M=chlu.experiments.exp_w8_encoder_loop
ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  ARMS=(alone frozen loop_nosw loop_swap alone_wc)
fi

python -u -m "$M" --stage warm --seed "$SEED" --out "$OUT"

# One arm's failure must not cost the others.  Every stage is resumable, so a failure is
# RECORDED LOUDLY and the leg continues; the leg exits non-zero at the end so a failure
# can never look like a clean run.
FAILED=()
for arm in "${ARMS[@]}"; do
  echo "[leg] ================ $arm ================"
  ok=1
  for stage in train cache probe; do
    python -u -m "$M" --stage "$stage" --seed "$SEED" --arm "$arm" --out "$OUT" || { ok=0; break; }
  done
  if [ "$ok" = 0 ]; then
    echo "[leg] !!!! ARM FAILED: $arm -- continuing; see FAILED.json" >&2
    mkdir -p "$OUT/seed_${SEED}/${arm}"
    printf '{"arm":"%s","seed":%s,"failed":true,"note":"the stage raised; this is a FAILURE, not a declared NOT-RUN, and it is never a null"}\n' \
      "$arm" "$SEED" > "$OUT/seed_${SEED}/${arm}/FAILED.json"
    FAILED+=("$arm")
  fi
done

python -u -m "$M" --stage report --out "$OUT" || true

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[leg] done WITH FAILURES: seed $SEED -- failed arms: ${FAILED[*]}" >&2
  exit 3
fi
echo "[leg] done: seed $SEED"
