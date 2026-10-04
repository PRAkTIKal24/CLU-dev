#!/usr/bin/env bash
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-generation-table/hubenv.sh
for i in $(seq 1 500); do grep -q "^exit=" "$CHLU_OUT/dev/exp09_seed0.log" 2>/dev/null && break; sleep 10; done
cd "$CHLU_WT"
"$PY" "$CHLU_OUT/dev/exp10_shrink.py" 0 16,32,64,128 > "$CHLU_OUT/dev/exp10_seed0.log" 2>&1
echo "exit=$?" >> "$CHLU_OUT/dev/exp10_seed0.log"
