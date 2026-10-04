#!/usr/bin/env bash
# ⭐⭐ THE BLOCKING GATE, ENFORCED IN THE DRIVER AND NOT BY MY GOOD INTENTIONS.
#
#   1. wait for run_det.sh to finish both full-scale builds
#   2. compare them -> DETERMINISM.json
#   3. ⛔ if the verdict is not PASS, STOP. Nothing downstream runs. That is the
#      falsifier (§5.5) and it is wired in, so a tired operator cannot skip it.
#   4. only on PASS, run the ordered DEV cell.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
. "$H/hubenv.sh"

echo "[gate] waiting for run_det.sh ... $(date -u +%H:%M:%SZ)"
for _ in $(seq 1 480); do
  grep -q "ALL DONE" "$HUB/logs/run_det.out" 2>/dev/null && break
  sleep 15
done
if ! grep -q "ALL DONE" "$HUB/logs/run_det.out" 2>/dev/null; then
  echo "[gate] ⛔ run_det.sh did not finish within the wait window. STOP."; exit 2
fi

echo "[gate] comparing det_A vs det_B ... $(date -u +%H:%M:%SZ)"
"$PY" "$(cygpath -m "$HUB")/compare_det.py" "$(cygpath -m "$HUB")/runs" det_A det_B \
    > "$HUB/DETERMINISM.json" 2> "$HUB/logs/determinism.err"
RC=$?
cat "$HUB/logs/determinism.err"
if [ "$RC" -ne 0 ]; then
  echo "[gate] ⛔⛔ DETERMINISM FAILED (rc=$RC). Nothing downstream is meaningful."
  echo "[gate] ⛔ Per PREREG-01 §1 / falsifier §5.5: STOP and report both digests and the batching."
  echo "[gate] ⛔ The DEV cell is NOT launched."
  exit 1
fi
echo "[gate] ⭐ DETERMINISM PASS at real scale. Releasing the DEV cell. $(date -u +%H:%M:%SZ)"
exec bash "$H/run_cell.sh" "$H/plan_ordered.jsonl"
