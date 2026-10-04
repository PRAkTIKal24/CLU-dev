#!/usr/bin/env bash
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-generation-table/hubenv.sh
for i in $(seq 1 180); do
  grep -q "^exit=" "$CHLU_OUT/dev/exp03_seed0.log" 2>/dev/null && break
  sleep 10
done
cd "$CHLU_WT"
"$PY" "$CHLU_OUT/dev/exp05_rivals_rank.py" 0 32 > "$CHLU_OUT/dev/exp05_seed0.log" 2>&1
echo "exit=$?" >> "$CHLU_OUT/dev/exp05_seed0.log"
