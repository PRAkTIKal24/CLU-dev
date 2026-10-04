# source me at the top of every shell
cd /c/Users/PratikJawahar/chlu/chlu-bundle-v2
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
. ./env_hub.sh >/dev/null
export PYTHONUTF8=1
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
export HUB="$CHLU_HUB_OUTPUTS/hub-deletion"
export WT=/c/Users/PratikJawahar/chlu/CHLU-hub-deletion-2
# Windows-native paths for python.exe (msys /c/... is not understood by native python)
export WROOT="$(cygpath -w "$CHLU_ROOT")"
export WHUB="$(cygpath -w "$CHLU_HUB_OUTPUTS/hub-deletion")"
export WWT="$(cygpath -w "$WT")"
export WOUT="$(cygpath -w "$CHLU_HUB_OUTPUTS")"
