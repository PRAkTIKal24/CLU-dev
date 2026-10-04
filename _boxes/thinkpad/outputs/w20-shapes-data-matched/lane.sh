#!/bin/bash
# usage: lane.sh <lane-name> <cmdfile>   -- runs each line of cmdfile sequentially
source /c/chlu/outputs/w20-shapes-data-matched/env.sh
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
O=/c/chlu/outputs/w20-shapes-data-matched
mkdir -p $O/logs
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "[$(date -u +%H:%M:%SZ)] START $line" >> $O/logs/lane_$1.log
  bash -c "$line" >> $O/logs/lane_$1.log 2>&1
  echo "[$(date -u +%H:%M:%SZ)] END rc=$? $line" >> $O/logs/lane_$1.log
done < "$2"
echo "[$(date -u +%H:%M:%SZ)] LANE DONE" >> $O/logs/lane_$1.log
