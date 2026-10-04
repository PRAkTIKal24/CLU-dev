# ladder legs: env.sh + the w12 ladder's gen-2 pins (scripts/w12_ladder_env.sh, verbatim values)
source /c/chlu/outputs/w15-lace-poe-rows/env.sh
export CHLU_GEN_FEATURES_ROOT="C:/chlu/outputs/spoke-a-rivals/cache_generation2"
export CHLU_GEN_CLASSIFIER_ROOT="$(python -c "import os,pathlib;print(pathlib.Path(os.environ['CHLU_GEN_BRIDGE_ROOT']).parent.parent / 'gen-external-classifier')")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
