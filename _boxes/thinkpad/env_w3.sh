# WAVE 3 environment. Source from INSIDE your worktree:  cd <worktree> && source /c/chlu/env_w3.sh
# Reads the frozen instrument in place (read-only). Every write goes to the dual-ACL tree.
for v in $(env | grep -o '^CHLU_[A-Z_]*'); do unset "$v"; done   # a stale CHLU_* silently overrides
export CHLU_ROOT="$PWD"
_H=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
export CHLU_CLAUDE_ROOT="$_H/.claude"
export CHLU_GEN_BRIDGE_ROOT="$_H/.claude/outputs/gen-feature-bridge/runs"
export CHLU_GEN_WIDE_BRIDGE_ROOT="$_H/.claude/outputs/gen-bridge-wide/runs"
export CHLU_GEN_STORE_RUNS="$_H/.claude/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="$_H/.claude/outputs/gen-decoder-lift/winner"
export CHLU_GEN_DATA_ROOT=/c/chlu/data/gen     # WRITABLE: h5 + montero + value_holdout
export CHLU_HUB_OUTPUTS=/c/chlu/outputs        # WRITABLE: the ONLY place a hub writes
# ⛔ THE IMPORT TRAP: $CHLU_ROOT must precede site-packages, which carries an editable
#    _editable_impl_chlu.pth pointing at the Head's own checkout. Verify chlu.__file__.
export PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset _H
