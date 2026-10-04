#!/bin/bash
# w21 throttled queue: <=3 of OUR heavy python processes, >=5 GB free before each launch.
# TEST jobs: ledger row APPENDED (>>) immediately before launch.
source /c/chlu/outputs/w21-maze-t-clu-improve/env.sh || exit 1
export PYTHONUTF8=1
OUT=/c/chlu/outputs/w21-maze-t-clu-improve; O=$(cygpath -m $OUT)
CFG=$O/prereg/CLU-REL-A10.json
mkdir -p $OUT/val_full $OUT/test
JOBS="BOTH:val:0 BOTH:val:1 BOTH:val:2 BOTH:test:0 BOTH:test:1 BOTH:test:2 WORDS-FULL:val:0 WORDS-FULL:val:1 WORDS-FULL:val:2"
declare -a PIDS=()
free_gb() { powershell -NoProfile -c '$o=Get-CimInstance Win32_OperatingSystem; [int]($o.FreePhysicalMemory/1MB)' | tr -d '\r'; }
alive() { local n=0; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && n=$((n+1)); done; echo $n; }
for j in $JOBS; do
  IFS=: read cond leg seed <<< "$j"
  while true; do
    if [ $(alive) -lt 3 ] && [ $(free_gb) -ge 5 ]; then break; fi
    sleep 60
  done
  if [ "$leg" = test ]; then
    run=$((7 + seed)); d=$OUT/test
    echo "| $(date -u +%Y-%m-%dT%H:%M:%SZ) | w21 hub/maze-t-clu-improve | **maze-t-symmetric** | **$run/9** | outputs/w21-maze-t-clu-improve/PREREG-w21-maze-t-clu-improve.md (sha256 eaa0f69b…) | CMAZE-T, condition **BOTH** (rivals obs ⊕ 318 design, frozen recipe, re-matched to 9,200 read floats by the standing rule; CLU true words), CLU = **CLU-REL-A10** (prereg/CLU-REL-A10.json sha256 0d4ec5b9…: K 2, 10 atoms, ridge 1, [bias|main|0.5·pair|colour-eq(6)|eq×door|eq×dir], 9,200 read floats), TEST leg seed $seed, branch w21/maze-t-clu-improve @ e7fdb9b. VAL gate PASSED: nway 0.9029/0.9141/0.9943 (mean 0.9371) vs A1 0.8019/0.5689/0.8034 (69 candidates in LOG.md). Label = Head's WIN/COMPARABLE/LOSS vs best matched rival over both budgets. Deletion + '+rel' diagnostic ON. Row written BEFORE the run. |" >> /c/chlu/SCORED-LEDGER.md
  else
    d=$OUT/val_full
  fi
  python /c/chlu/worktrees/CHLU-w21-maze-t-clu-improve/scripts/w19_symmetric.py $seed $(cygpath -m $d) --cond $cond --leg $leg --rel_diag 1 --clu_cfg $CFG > $OUT/logs/${leg}_${cond}_seed${seed}.log 2>&1 &
  p=$!; PIDS+=($p)
  echo "$(date -u +%FT%TZ) launched $j msys_pid=$p winpid=$(cat /proc/$p/winpid 2>/dev/null) free_gb=$(free_gb)" | tee -a $OUT/PIDS.txt >> $OUT/LOG.md
  sleep 90
done
wait
echo "$(date -u +%FT%TZ) queue finished" >> $OUT/LOG.md
