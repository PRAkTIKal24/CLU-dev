# hub/composition-2 (iteration 2) â source this at the top of every shell. Self-contained.
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
cd "$CHLU_ROOT"
. ./env_hub.sh
export PYTHONUTF8=1
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
# LAUNCH-COMMON Â§3: FOUR hubs x 4 threads = 16 cores. No hub exceeds its 4.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export WT=/c/Users/PratikJawahar/chlu/CHLU-hub-composition-2
export OUT="$CHLU_HUB_OUTPUTS/hub-composition/iter2"
# --- inherited ENVIRONMENT from the predecessor hub (not results; already on disk) ---
export HUBENV=/c/Users/PratikJawahar/chlu/hub-composition-env
# ⛐ the WORKTREE first: base/iter2 = the pin tree + the reference-read files + the gate
# patch. `chlu.core.gen_factored_read` exists only there. The rest of the tree is byte-identical
# to the bundle (verified by `diff -rq`), so nothing else changes identity.
export PYTHONPATH="$WT:$CHLU_ROOT:$HUBENV/pylibs"
export CHLU_GEN_DATA_ROOT="$HUBENV/data/gen"
