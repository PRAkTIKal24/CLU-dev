# WAVE 4 env. Source from INSIDE your worktree:  cd <worktree> && source /c/chlu/env_w4.sh
for v in $(env | grep -o '^CHLU_[A-Z_]*'); do unset "$v"; done
export CHLU_ROOT="$PWD"
_H=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
export CHLU_CLAUDE_ROOT="$_H/.claude"
# ⚠ GENERATION 1 caches. Spoke A builds generation 2 and publishes its root in
#   /c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json -> "cache_gen2_root".
#   DEV may use gen-1; ⛔ every SCORED run uses gen-2 and never mixes the two.
export CHLU_GEN_BRIDGE_ROOT="$_H/.claude/outputs/gen-feature-bridge/runs"
export CHLU_GEN_WIDE_BRIDGE_ROOT="$_H/.claude/outputs/gen-bridge-wide/runs"
export CHLU_GEN_STORE_RUNS="$_H/.claude/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="$_H/.claude/outputs/gen-decoder-lift/winner"
export CHLU_GEN_DATA_ROOT=/c/chlu/data/gen
export CHLU_HUB_OUTPUTS=/c/chlu/outputs
export CHLU_RIVALS_READY=/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json
export PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"   # ⛔ CHLU_ROOT FIRST — the import trap
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
unset _H
