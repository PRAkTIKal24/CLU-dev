#!/bin/sh
cd /c/chlu/worktrees/CHLU-w9-harness
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for s in 0 1 2; do
  python -u exp_w9_harness.py --seed $s --scale std --tier cmaze1-easy --tag _easy \
      --out /c/chlu/outputs/w9-harness > /c/chlu/outputs/w9-harness/logs/easy_seed$s.log 2>&1
done
echo EASYDONE
