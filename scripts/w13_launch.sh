#!/usr/bin/env bash
# w13 hub/encoder-adapt -- launch the arm pipelines, N lanes at a time, PIDs recorded.
#
#   scripts/w13_launch.sh <out> <lanes> [joint_steps] [dec_steps] [seeds] [arms]
#
# Every lane is ONE (seed, arm) running train -> cache -> probe -> rivals.  Each lane gets
# ONE thread (OMP/MKL/OpenBLAS/NumExpr = 1) so that <lanes> lanes use <lanes> threads of
# this shared box -- this hub's allowance is 4.
#
# ⛔ Every PID is appended to MY_PIDS.txt at launch.  Nothing in this hub ever issues a
# box-wide kill: other agents are running here.
set -uo pipefail

OUT="${1:?out dir}"
LANES="${2:-3}"
JS="${3:-}"
DS="${4:-}"
SEEDS="${5:-0 1 2}"
ARMS="${6:-frozen adapter anchor}"

W13=/c/chlu/outputs/w13-encoder-adapt
mkdir -p "$W13/logs"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

JOBS=()
for S in $SEEDS; do for A in $ARMS; do JOBS+=("$S:$A"); done; done
echo "[launch] ${#JOBS[@]} lanes total, $LANES at a time, joint_steps='${JS:-PIN}' dec_steps='${DS:-PIN}'"

running=0
for J in "${JOBS[@]}"; do
  S="${J%%:*}"; A="${J##*:}"
  while [ "$(jobs -rp | wc -l)" -ge "$LANES" ]; do sleep 10; done
  bash "$(dirname "$0")/w13_arm_pipeline.sh" "$OUT" "$S" "$A" "$JS" "$DS" \
      > "$W13/logs/lane_s${S}_${A}.log" 2>&1 &
  P=$!
  echo "lane_s${S}_${A} PID $P" | tee -a "$W13/MY_PIDS.txt"
  running=$((running + 1))
done
wait
echo "[launch] ALL LANES DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
