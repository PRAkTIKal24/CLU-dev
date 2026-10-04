# hub/refusal-2 (RELAUNCH) — source me at the top of every shell.
cd /c/Users/PratikJawahar/chlu/chlu-bundle-v2
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
. ./env_hub.sh
export PYTHONUTF8=1
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export HUB=hub-refusal
export OUT="$CHLU_HUB_OUTPUTS/$HUB"
export WT=/c/Users/PratikJawahar/chlu/CHLU-hub-refusal-2
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
