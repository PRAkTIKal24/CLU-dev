#!/bin/sh
cd /c/chlu/worktrees/CHLU-w9-harness
for s in 0 1 2; do
  python -u exp_w9_harness.py --seed $s --scale std --out /c/chlu/outputs/w9-harness \
      > /c/chlu/outputs/w9-harness/logs/seed$s.log 2>&1
  python publish_ready.py --out /c/chlu/outputs/w9-harness > /dev/null 2>&1
done
echo ALLDONE
