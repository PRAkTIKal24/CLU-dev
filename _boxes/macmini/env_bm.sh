# batmac environment for CHLU spokes. Mirrors forgis-14's env_w3.sh contract.
# Source from INSIDE a worktree so CHLU_ROOT is that worktree.
for v in $(env | grep -o '^CHLU_[A-Z_]*'); do unset "$v"; done
export CHLU_ROOT="$PWD"
_S=/Users/batmac/clu/stage/outputs
export CHLU_EVIDENCE=/Users/batmac/clu/evidence/outputs
export CHLU_HUB_OUTPUTS=/Users/batmac/clu/outputs
# --- data + frozen instrument (staged from forgis-14; dataset sha256 verified
#     against the runbook pin 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785)
export CHLU_GEN_DATA_ROOT=/Users/batmac/clu/data/gen
export CHLU_GEN_DECLIFT_DIR="$_S/gen-decoder-lift/winner"
export CHLU_GEN_FEATURES_ROOT="$_S/spoke-a-rivals/cache_generation2"
# ⚠ PARTIAL: only seed_0's features.npy is staged. A spoke needing seeds 1-2 must
#    declare it (NEEDS-ARTEFACTS.json) rather than silently substituting.
export CHLU_GEN_CLASSIFIER_ROOT="$_S/gen-external-classifier"
export PATH="/Users/batmac/clu/workspace/.venv/bin:/Library/TeX/texbin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"
export PYTHONPATH="$CHLU_ROOT"
# 10 cores; leave headroom for the tmux server that holds this machine's keychain
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
unset _S
