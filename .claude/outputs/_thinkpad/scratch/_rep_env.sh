unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
cd "$CHLU_ROOT" && . ./env_hub.sh
export PYTHONUTF8=1
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WT=/c/Users/PratikJawahar/chlu/CHLU-hub-composition-2
export HUBENV=/c/Users/PratikJawahar/chlu/hub-composition-env
export PYTHONPATH="$WT:$CHLU_ROOT:$HUBENV/pylibs"
export CHLU_GEN_DATA_ROOT="$HUBENV/data/gen"
export REP="$CHLU_CLAUDE_ROOT/outputs/hub-composition/iter2r/replication"
mkdir -p "$REP"
