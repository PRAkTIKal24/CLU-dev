#!/bin/sh
cd /c/chlu/worktrees/CHLU-w9-harness
export OMP_NUM_THREADS=1
python -u exp_w9_refuseprobe.py --seeds 0 --tiers cmaze1,cmaze1-easy --out /c/chlu/outputs/w9-harness
python -u exp_w9_probe.py --seeds 0 --out /c/chlu/outputs/w9-harness
echo PROBESDONE
