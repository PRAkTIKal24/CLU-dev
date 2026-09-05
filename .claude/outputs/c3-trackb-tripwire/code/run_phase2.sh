#!/bin/bash
# Phase 2: the 1-day perbasin arm (re-run after the predict() chunking patch),
# the LOCAL (same-basin) arms, the deep dive and the shuffled-order null.
set -u
cd /Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire
P=./.venv/bin/python
O=/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire
while pgrep -f run_regional.sh > /dev/null; do sleep 30; done
echo "regional finished, phase2 starts $(date)"

$P camels_tripwire.py --window 1 --scaling perbasin --Ls 250,1000,5000,14894 \
   --seeds 0,1,2 --kmeans 0 --sub 1 --tag inbudget --out $O/arms.jsonl

for SC in perbasin raw; do
  $P camels_local.py --mode local --window 30 --scaling $SC \
     --Ls 5,250,1000,3287 --seeds 0 --out $O/arms_local.jsonl
  $P camels_local.py --mode local --window 365 --scaling $SC \
     --Ls 250,1000,3287 --seeds 0 --sub 3 --out $O/arms_local.jsonl
done
echo "ALL PHASE2 DONE $(date)"
