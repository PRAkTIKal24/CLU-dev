#!/usr/bin/env bash
# w12 ROUND 2 -- seeds 1 and 2 at the DECISION-RELEVANT K only, split across three
# processes.  The K list is read from CROSSOVER-val.json (K=5, K*, and the crossover K),
# so this script selects nothing: it executes the frozen rule's output.
#
# Every (K, seed) cell NOT launched here is a DECLARED NOT-RUN whose reason is the wall
# clock, and the realised n is printed on every row of the report.
set -u
cd "$(dirname "$0")/.."
OUTD=/c/chlu/outputs/w12-single-design-crossover
KS=$(python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("C:/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json")
               .read_text(encoding="utf-8"))
ks = [5]
for key in ("K_star", "K_crossover"):
    v = d.get(key)
    if v and v not in ks:
        ks.append(int(v))
print(" ".join(str(k) for k in sorted(ks)))
PY
)
echo "round 2 K list from the frozen rule: $KS"
ARGS=""
for k in $KS; do ARGS="$ARGS --K $k"; done
bash scripts/w12_launch.sh sweep_s1_r2 scripts/w12_val_sweep.py --condition val --seed 1 $ARGS
bash scripts/w12_launch.sh sweep_s2_r2 scripts/w12_val_sweep.py --condition val --seed 2 $ARGS
echo "launched; PIDs appended to $OUTD/logs/PIDS.txt"
