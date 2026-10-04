# hub/refusal-2 — source this at the top of EVERY unit. ⭐ Code = worktree, data = bundle pin.
cd /c/Users/PratikJawahar/chlu/chlu-bundle-v2
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
. ./env_hub.sh
export PYTHONUTF8=1
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
# ⚠ 4 hubs x 4 threads = 16 cores. ⛔ Never exceed 4.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export HUB_WT=/c/Users/PratikJawahar/chlu/CHLU-hub-refusal-2
export HUB_OUT="$CHLU_HUB_OUTPUTS/hub-refusal-2"
