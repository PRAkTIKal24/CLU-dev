#!/bin/sh
cd /c/chlu/worktrees/CHLU-w9-maze
export OMP_NUM_THREADS=3
P=C:/chlu/outputs/w9-maze/PREREG-w9-maze
L=/c/chlu/outputs/w9-maze/logs
# seed 1: encoder is already training in ae_seed1.log; wait for its cache to appear
while [ ! -f "C:/chlu/outputs/w9-maze/runs/ae/ae_seed1_steps12000.eqx" ]; do sleep 20; done
python -u scripts/w9_maze_dev.py --seed 1 --which test --ae-steps 12000 --n-atoms 31 --K 8 \
  --ridge 0.01 --lowrank-atoms 8 --tag scored2 --display --prereg $P-2.md > $L/scored2.log 2>&1
python -u scripts/w9_maze_ae_cache.py seed=2 ae_steps=12000 > $L/ae_seed2.log 2>&1
python -u scripts/w9_maze_dev.py --seed 2 --which test --ae-steps 12000 --n-atoms 31 --K 8 \
  --ridge 0.01 --lowrank-atoms 8 --tag scored3 --display --prereg $P-3.md > $L/scored3.log 2>&1
echo CHAIN_DONE
