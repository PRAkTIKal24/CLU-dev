#!/usr/bin/env bash
# Hold the foreground for N seconds, then print a compact status of the whole hub.
# Exists so a wait costs one tool call and one short block of output, not a poll loop.
S=${1:-110}
python -c "
import time
t0=time.time()
while time.time()-t0 < $S: time.sleep(5)
" 2>/dev/null
D=/c/chlu/outputs/w12-single-design-crossover
echo "UTC $(date -u +%H:%M:%SZ)"
echo "rows: $(ls $D/val_sweep 2>/dev/null | tr '\n' ' ')"
for f in A B C; do
  printf 'sweep_%s: %s\n' "$f" "$(grep -hE '=====|valid |EXACTNESS' $D/logs/sweep_s0_$f.log 2>/dev/null | tail -1 | sed 's/^ *//')"
done
echo "gate BASELINES-READY: $([ -f /c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json ] && echo YES || echo no)"
echo "gate CACHE-READY:     $([ -f /c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json ] && echo YES || echo no)"
