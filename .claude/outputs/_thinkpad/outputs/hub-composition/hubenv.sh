# source this at the top of every shell. Self-contained.
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
cd "$CHLU_ROOT"
. ./env_hub.sh
export PYTHONUTF8=1
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export WT=/c/Users/PratikJawahar/chlu/CHLU-hub-composition
export OUT="$CHLU_HUB_OUTPUTS/hub-composition"
# --- hub/composition private additions (in-scope; nothing shared is mutated) ---
# ⚠ MOVED 2026-09-17T00:12Z out of $OUT so the return pack carries RESULTS ONLY.
#   These are ENVIRONMENT, not results: a pip --target dir and a public dataset.
#   ⛔ Nothing was deleted; both were moved intact (h5 sha256 re-verified after the move).
export HUBENV=/c/Users/PratikJawahar/chlu/hub-composition-env
export PYTHONPATH="$CHLU_ROOT:$HUBENV/pylibs"      # h5py 3.16.0, --no-deps, numpy untouched
export CHLU_GEN_DATA_ROOT="$HUBENV/data/gen"       # hardlinks to bundled files + downloaded 3dshapes.h5
