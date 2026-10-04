#!/bin/zsh
# All three seeds, sequential, incremental. ⛔ Main venv; ⛔ no uv sync in the worktree.
for s in 0 1 2; do
  /Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/run_seed.sh $s \
    > /Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/seed${s}.log 2>&1
  echo "seed $s exit=$? at $(date -Iseconds)" >> /Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/progress.txt
done
echo "ALL SEEDS DONE $(date -Iseconds)" >> /Users/user/Desktop/CHLU/.claude/outputs/gen-baselines/progress.txt
