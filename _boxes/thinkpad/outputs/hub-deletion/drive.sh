#!/usr/bin/env bash
# wait for det_B, then run the whole cell ledger sequentially
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-deletion/hubenv.sh
until [ -f "$HUB/runs/det_B/build.json" ]; do sleep 20; done
echo "det_B landed $(date -u +%H:%M:%SZ) — starting cell ledger"
exec bash "$WT/scripts/hub_deletion/run_cell.sh"
