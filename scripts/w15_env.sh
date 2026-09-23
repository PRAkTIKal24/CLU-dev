#!/usr/bin/env bash
# w15 coverage-control: env_w3.sh + the w12 ladder's pins + a Windows-form PYTHONPATH
# (env_w3.sh's mixed "C:/..:/c/.." form is not converted by git-bash, so equinox is lost).
source /c/chlu/env_w3.sh
export PYTHONPATH="$(cygpath -w "$PWD");C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"
export CHLU_GEN_FEATURES_ROOT="C:/chlu/outputs/spoke-a-rivals/cache_generation2"
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT
: "${THREADS:=1}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=1" PYTHONIOENCODING=utf-8 PYTHONUTF8=1
