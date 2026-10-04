#!/bin/bash
# FREE VAL leg: peer CLU-REL-A10 (vendored e7fdb9b runner), BOTH val seeds 3,4,5. Starts after queue 1 is done.
D=/c/chlu/outputs/w21-maze-t-seed4; O=C:/chlu/outputs/w21-maze-t-seed4
cd /c/chlu/worktrees/CHLU-w21-maze-t-seed4; source $D/env.sh
export PYTHONPATH="$PYTHONPATH;C:/chlu/worktrees/CHLU-w21-maze-t-seed4/scripts"
V=peer_w21_clu_improve_e7fdb9b
mine() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like '*symmetric.py*w21-maze-t-seed4*' }).Count" | tr -d '\r'; }
freegb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r'; }
until grep -q QUEUE-DONE $D/logs/queue.log; do sleep 60; done
for s in 3 4 5; do
  while true; do m=$(mine); f=$(freegb); [ "$m" -lt 3 ] && [ "$f" -ge 5 ] && break; sleep 60; done
  ( python $V/w19_symmetric.py $s $O/val_cluimprove --cond BOTH --leg val --rel_diag 1 --clu_cfg $V/CLU-REL-A10.json > $D/logs/val_cluimprove_BOTH_seed$s.log 2>&1; echo "EXIT $?" >> $D/logs/val_cluimprove_BOTH_seed$s.log ) &
  sleep 5
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like \"*w19_symmetric.py $s *val_cluimprove*\" } | % { '{0} {1}' -f \$_.ProcessId, \$_.CommandLine }" | tr -d '\r' >> $D/PIDS.txt
  echo "- $(date -u +%FT%TZ) QUEUE2 launched FREE VAL CLU-REL-A10 BOTH val seed $s (free ${f}GB, mine $m)" >> $D/LOG.md
done
echo QUEUE2-DONE
