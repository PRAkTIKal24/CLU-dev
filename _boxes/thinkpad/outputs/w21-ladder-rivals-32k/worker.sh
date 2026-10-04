#!/bin/bash
# worker.sh <name> <jobfile>: pull jobs from <jobfile> in order; claim each atomically via mkdir; run it.
# A job line is "<jobid> <command...>". Waits for >=5 GB free before each job.
O=/c/chlu/outputs/w21-ladder-rivals-32k
source $O/env.sh
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
mkdir -p $O/claims $O/logs
while IFS= read -r line; do
  [ -z "$line" ] && continue
  id=${line%% *}; cmd=${line#* }
  mkdir $O/claims/$id 2>/dev/null || continue
  until [ "$(powershell -NoProfile -c '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)')" -ge 5 ]; do sleep 60; done
  echo "[$(date -u +%H:%M:%SZ)] $1 START $id" >> $O/logs/worker_$1.log
  bash -c "$cmd" > $O/logs/job_$id.log 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%SZ)] $1 END rc=$rc $id" >> $O/logs/worker_$1.log
  echo "$rc" > $O/claims/$id/rc
done < "$2"
echo "[$(date -u +%H:%M:%SZ)] $1 QUEUE DONE" >> $O/logs/worker_$1.log
