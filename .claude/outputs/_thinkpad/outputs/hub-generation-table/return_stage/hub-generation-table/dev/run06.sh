#!/usr/bin/env bash
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-generation-table/hubenv.sh
cd "$CHLU_WT"
"$PY" "$CHLU_OUT/dev/exp06_launch.py" 0 16,32,64,128,256 full,16,7 > "$CHLU_OUT/dev/exp06_seed0.log" 2>&1
echo "exit=$?" >> "$CHLU_OUT/dev/exp06_seed0.log"
