#!/bin/sh
cd /c/chlu/worktrees/CHLU-w9-harness
export OMP_NUM_THREADS=1
echo "=== BEFORE the fix: single run, as the seed sweep used ==="
python -u exp_w9_probe.py --seeds 0,1,2 --tiers cmaze1,cmaze1-easy --restarts 1 \
    --name encoder_probe_1restart --out /c/chlu/outputs/w9-harness
echo
echo "=== AFTER the fix: up to 4 restarts, keep the best RECONSTRUCTION loss ==="
python -u exp_w9_probe.py --seeds 0,1,2 --tiers cmaze1,cmaze1-easy --restarts 4 \
    --accept-bce 0.02 --name encoder_probe_restarts --out /c/chlu/outputs/w9-harness
echo ENCSTABDONE
