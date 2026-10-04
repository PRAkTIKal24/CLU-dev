# env_hub.sh — source this on the far side. ⭐ $CHLU_ROOT is the single source of truth.
#   cd <bundle> && source ./env_hub.sh && ./scripts/verify_hub_workspace.sh
# ⚠ If CHLU_ROOT is unset it defaults to THIS file's directory, so the bundle is
#   correct wherever it is extracted. ⛔ Not one path here names the build machine.
_ENV_HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export CHLU_ROOT="${CHLU_ROOT:-$_ENV_HUB_DIR}"
export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CHLU_ROOT/.claude}"
export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
export CHLU_GEN_WIDE_BRIDGE_ROOT="${CHLU_GEN_WIDE_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-bridge-wide/runs}"
export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"
# ⭐ The one path a hub may write (agent profile §0.3), derived — ⛔ never hardcoded.
export CHLU_HUB_OUTPUTS="${CHLU_HUB_OUTPUTS:-$CHLU_CLAUDE_ROOT/outputs}"
# ⛔⛔ THE IMPORT TRAP. `python -c` puts the CWD at sys.path[0], AHEAD of PYTHONPATH.
# Run from anywhere but $CHLU_ROOT and you silently import some other checkout's chlu.
# verify_hub_workspace.sh asserts chlu.__file__ for exactly this reason.
export PYTHONPATH="$CHLU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset _ENV_HUB_DIR
