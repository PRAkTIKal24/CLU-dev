#!/bin/sh
cd /c/chlu/worktrees/CHLU-w9-harness
TIER=$1; TAG=$2
for s in 1 2; do
  python -u exp_w9_harness.py --seed $s --scale std --tier $TIER --tag "$TAG" \
      --out /c/chlu/outputs/w9-harness \
      > /c/chlu/outputs/w9-harness/logs/r2_${TIER}_seed$s.log 2>&1
done
echo DONE_$TIER
