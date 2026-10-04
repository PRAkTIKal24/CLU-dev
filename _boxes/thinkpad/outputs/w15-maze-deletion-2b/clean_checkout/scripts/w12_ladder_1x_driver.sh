#!/usr/bin/env bash
# w12 hub/composition-ladder -- the AMENDMENT-1 column, one leg at a time.
#
# Waits for each ladder leg to write its LEG.json, then fits and scores that leg's two
# byte-matched learners on the leg's OWN pool.  ONE process at a time, so it adds a single
# competitor to the three lanes rather than twelve.
#
#   scripts/w12_ladder_1x_driver.sh <rung>:<seed> ...
#
# A leg that never appears is skipped with a DECLARED NOT-RUN line rather than blocking the
# queue; a leg whose 1x fit fails is recorded and the driver CONTINUES.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/w12_ladder_env.sh
export THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT=C:/chlu/outputs/w12-composition-ladder
RUNS="$OUT/runs"
LOGD=/c/chlu/outputs/w12-composition-ladder/logs
: "${MAX_WAIT:=9000}"        # per leg, seconds

FAILED=()
for job in "$@"; do
  rung="${job%%:*}"; seed="${job##*:}"
  leg="/c/chlu/outputs/w12-composition-ladder/runs/${rung}_seed${seed}"
  waited=0
  while [ ! -f "$leg/LEG.json" ]; do
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      echo "[1x] DECLARED NOT-RUN ${rung}:${seed} -- the ladder leg did not finish within ${MAX_WAIT}s"
      break
    fi
    sleep 20; waited=$((waited + 20))
  done
  if [ ! -f "$leg/LEG.json" ]; then FAILED+=("$job(leg-absent)"); continue; fi
  echo "[1x] ==== ${rung}:${seed} $(date -u +%H:%M:%SZ) ===="
  python -u scripts/w12_ladder_1x.py --rung "$rung" --seed "$seed" --out "$RUNS" \
      >> "$LOGD/onex_${rung}_seed${seed}.log" 2>&1 || FAILED+=("$job")
  echo "[1x] ---- done ${rung}:${seed} $(date -u +%H:%M:%SZ) ----"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[1x] FINISHED WITH FAILURES: ${FAILED[*]}" >&2
  exit 3
fi
echo "[1x] finished clean"
