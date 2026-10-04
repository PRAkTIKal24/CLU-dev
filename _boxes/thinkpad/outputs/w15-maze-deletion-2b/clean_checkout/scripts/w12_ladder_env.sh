#!/usr/bin/env bash
# w12 hub/composition-ladder -- the environment every leg runs under.
#
# Two variables are NOT in this box's ambient environment and both are load-bearing:
#   CHLU_GEN_FEATURES_ROOT  unset => exp_gen_factor_wells.features_root() falls back to
#                           the frozen bridge = cache GENERATION 1, which w5-common 3.8
#                           forbids for any number.  Pinned here to spoke A's gen-2 tree.
#   CHLU_GEN_CLASSIFIER_ROOT the external classifier's location is DERIVED from the
#                           bridge path, so redirecting the features would send the judge
#                           looking for a classifier that is not there.  Pinned to the
#                           same FILE the shipped legs use; the hash Judge asserts against
#                           it is unchanged.
export CHLU_GEN_FEATURES_ROOT="C:/chlu/outputs/spoke-a-rivals/cache_generation2"
: "${CHLU_GEN_CLASSIFIER_ROOT:=$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")}"
export CHLU_GEN_CLASSIFIER_ROOT

: "${THREADS:=3}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1 ${XLA_FLAGS:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
