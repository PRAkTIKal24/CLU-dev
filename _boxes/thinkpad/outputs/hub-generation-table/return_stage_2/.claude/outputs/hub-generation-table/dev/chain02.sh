#!/usr/bin/env bash
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-generation-table/hubenv.sh
# wait for exp01 to finish (its json appears) before starting, to respect the 5-thread budget
for i in $(seq 1 240); do
  [ -f "$CHLU_OUT/dev/exp01_seed0.json" ] && break
  sleep 10
done
cd "$CHLU_WT"
"$PY" "$CHLU_OUT/dev/exp02_ksweep.py" 0 5,10,20,40,64 > "$CHLU_OUT/dev/exp02_seed0.log" 2>&1
echo "exp02 exit=$?" >> "$CHLU_OUT/dev/exp02_seed0.log"
