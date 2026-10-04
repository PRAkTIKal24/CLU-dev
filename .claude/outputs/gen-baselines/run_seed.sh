#!/bin/zsh
# gen-baselines full run, one seed per invocation. ⛔ Main venv (protocol §4); no uv sync in the worktree.
set -e
SEED=$1
export PYTHONPATH=/Users/user/Desktop/CHLU-gen-baselines
export CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen
export CHLU_GEN_BRIDGE_ROOT=/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs
export CHLU_GEN_DECLIFT_DIR=/Users/user/Desktop/CHLU/.claude/outputs/gen-decoder-lift/winner
export CHLU_GEN_STORE_RUNS=/Users/user/Desktop/CHLU/.claude/outputs/gen-store-grouping/runs
# ⛔ cwd MUST be the worktree: the main repo carries an editable-install .pth for
# `chlu` that otherwise shadows PYTHONPATH and silently runs the WRONG tree.
cd /Users/user/Desktop/CHLU-gen-baselines
OUT=/Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/runs/shapes3d_recon/seed_${SEED}
mkdir -p $OUT
echo "=== seed ${SEED} START $(date -Iseconds) ==="
/Users/user/Desktop/CHLU/.venv/bin/python -u -m chlu.experiments.exp_gen_baselines \
  --dataset shapes3d --pretext recon --stage baselines --arm all --seed ${SEED} --out $OUT
echo "=== seed ${SEED} END $(date -Iseconds) ==="
