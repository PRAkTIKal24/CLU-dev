#!/usr/bin/env bash
# 1) wait for det_B   2) THE DETERMINISM GATE (det_A vs det_B)   3) only then, the cell.
# ⛔ The gate is wired into the driver, not left to my judgement: if it FAILS the cell
# never starts and a STOP marker is written.
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-deletion/hubenv.sh
until [ -f "$HUB/runs/det_B/build.json" ]; do sleep 20; done
echo "[gate] det_B landed $(date -u +%H:%M:%SZ)"
"$PY" "$HUB/compare_det.py" "$(cygpath -w "$HUB/runs")" det_A det_B > "$HUB/DETERMINISM.json" 2> "$HUB/logs/gate.err"
rc=$?
cat "$HUB/logs/gate.err"
if [ $rc -ne 0 ]; then
  echo "[gate] ⛔ DETERMINISM FAILED — the cell will NOT start." | tee "$HUB/STOP-DETERMINISM-FAILED.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ >> "$HUB/STOP-DETERMINISM-FAILED.txt"
  exit 1
fi
echo "[gate] ⭐ DETERMINISM PASS at $(date -u +%H:%M:%SZ) — starting the cell"
exec bash "$HUB/run_cell.sh" "$HUB/plan_merged.jsonl"
