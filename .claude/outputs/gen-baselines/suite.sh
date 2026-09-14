#!/bin/zsh
# $1 = worktree path, $2 = label. ⛔ Main venv (protocol §4); ⛔ no uv sync in a worktree (w6).
cd $1
echo "=== suite $2 START $(date -Iseconds) HEAD=$(git rev-parse --short HEAD) ==="
PYTHONPATH=$1 /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q -p no:cacheprovider --no-cov
echo "=== suite $2 END $(date -Iseconds) HEAD=$(git rev-parse --short HEAD) ==="
