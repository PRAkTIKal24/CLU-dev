# hub/generation-table-2 shell preamble. Source this in every shell.
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
cd /c/Users/PratikJawahar/chlu/chlu-bundle-v2
. ./env_hub.sh
export CHLU_WT=/c/Users/PratikJawahar/chlu/CHLU-hub-generation-table-2
export CHLU_OUT="$CHLU_HUB_OUTPUTS/hub-generation-table"
export CHLU_OUT2="$CHLU_OUT/iter2"
export PYTHONUTF8=1
export PYTHONPATH="$CHLU_WT"
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd "$CHLU_WT"
