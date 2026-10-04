#!/bin/zsh
# wait for leg H, then run A (atom-mean) and C (cell harness) in sequence.
P=/Users/user/Desktop/CHLU/.claude/outputs/gen-val-split-fix/runs/progress.txt
until grep -q "STAGE H DONE" $P 2>/dev/null; do sleep 30; done
/Users/user/Desktop/CHLU/.claude/outputs/gen-val-split-fix/run_rescore.sh A
/Users/user/Desktop/CHLU/.claude/outputs/gen-val-split-fix/run_rescore.sh C
echo "CHAIN DONE" | tee -a $P
