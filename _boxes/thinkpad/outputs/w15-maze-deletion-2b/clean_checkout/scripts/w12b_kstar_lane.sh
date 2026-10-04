#!/usr/bin/env bash
# w12b hub/nr6-and-kstar -- ONE LANE of the K* ladder replication.
#
# A lane is a SEQUENTIAL list of "<rung>:<seed>:<K>" jobs run at ONE thread.  Three lanes
# = the 3 threads the task file allows.  The lane records its own PID and never touches
# any other process: ⛔ `hub/encoder-adapt` is TRAINING on this box and there is no
# box-wide kill anywhere in this hub.
#
# A leg that has already written every `rows_*.json` it was asked for is skipped by the
# runner itself, so a lane is restartable and re-running it costs nothing.
set -u
LOGDIR=/c/chlu/outputs/w12b-nr6-and-kstar/logs
mkdir -p "$LOGDIR"
lane="$1"; shift
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for job in "$@"; do
  rung="${job%%:*}"; rest="${job#*:}"; seed="${rest%%:*}"; K="${rest##*:}"
  tag="kstar_${rung}_s${seed}_K${K}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane=$lane START $tag" >> "$LOGDIR/KSTAR-PROGRESS.txt"
  python -u scripts/w12b_ladder_kstar.py --rung "$rung" --seed "$seed" --K "$K" \
      > "$LOGDIR/$tag.log" 2>&1
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane=$lane END   $tag rc=$rc" >> "$LOGDIR/KSTAR-PROGRESS.txt"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lane=$lane LANE COMPLETE" >> "$LOGDIR/KSTAR-PROGRESS.txt"
