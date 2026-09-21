#!/usr/bin/env bash
# w12 hub/composition-ladder -- one LANE.  `scripts/w12_ladder_lane.sh <lane-name> <job ...>`
# A job is "<rung>:<seed>".  Every leg is resumable (each arm writes its own rows_*.json and
# is skipped when present), so a lane that dies costs one arm, not the lane.
# A leg's failure is RECORDED and the lane CONTINUES; the lane exits non-zero at the end so
# a failure can never look like a clean run.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/w12_ladder_env.sh

LANE="${1:?lane name}"; shift
OUT=C:/chlu/outputs/w12-composition-ladder
LOGD=/c/chlu/outputs/w12-composition-ladder/logs
mkdir -p "$LOGD"

FAILED=()
for job in "$@"; do
  rung="${job%%:*}"; seed="${job##*:}"
  echo "[$LANE] ==== rung=$rung seed=$seed $(date -u +%H:%M:%SZ) ===="
  python -u scripts/w12_ladder_leg.py --rung "$rung" --seed "$seed" --out "$OUT/runs" \
      >> "$LOGD/${rung}_seed${seed}.log" 2>&1 || FAILED+=("$job")
  echo "[$LANE] ---- done rung=$rung seed=$seed rc=$? $(date -u +%H:%M:%SZ) ----"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[$LANE] FINISHED WITH FAILURES: ${FAILED[*]}" >&2
  exit 3
fi
echo "[$LANE] finished clean"
