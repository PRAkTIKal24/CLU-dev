#!/bin/bash
# w21 maze-t-seed4 queue: <=3 heavy python of MINE, >=5 GB free before each launch.
D=/c/chlu/outputs/w21-maze-t-seed4; O=C:/chlu/outputs/w21-maze-t-seed4
source $D/env.sh
mine() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like '*w21_symmetric.py*w21-maze-t-seed4*' }).Count" | tr -d '\r'; }
freegb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r'; }
wait_slot() { while true; do m=$(mine); f=$(freegb); if [ "$m" -lt 3 ] && [ "$f" -ge 5 ]; then echo "$(date -u +%FT%TZ) slot: mine=$m free=${f}GB"; return; fi; sleep 60; done; }
launch() { # seed cond leg
  local s=$1 c=$2 l=$3
  ( python scripts/w21_symmetric.py $s $O/$l --cond $c --leg $l > $D/logs/${l}_${c}_seed$s.log 2>&1; echo "EXIT $?" >> $D/logs/${l}_${c}_seed$s.log ) &
  sleep 5
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -like \"*w21_symmetric.py $s *--cond $c --leg $l*\" } | % { '{0} {1}' -f \$_.ProcessId, \$_.CommandLine }" | tr -d '\r' >> $D/PIDS.txt
  echo "- $(date -u +%FT%TZ) QUEUE launched $c $l seed $s (free ${f}GB, mine before launch $m)" >> $D/LOG.md
}
wait_slot
T=$(date -u +%FT%TZ)
echo "| $T | w21 spoke/maze-t-seed4 | **maze-t-seed4** | **3/3** | outputs/w21-maze-t-seed4/AMENDMENT-1.md (Head; NO PREREG — pre-registration banned 2026-09-24) | CMAZE-T, condition **BOTH** (words-plus-observation: rivals obs ⊕ 318 design, CLU the true words), store config **A1 UNCHANGED** (latent 128, K 2, 6 atoms, ridge 1.0), **TEST leg seed 5** (new seed; seeds 3/4/5 all run and kept, no screening). All rivals 8k + 80k, 10x-read-floats diagnostic, 4 controls (across-wells derangement asserted), learned blind arms, floors, deletion, per-step, rollout; objects persisted with hashes. Runner scripts/w21_symmetric.py (sha256 68497f65…) at branch w21/maze-t-seed4 @ 1c4e304 (base w19 0455961). Spend record only: no expected outcome, no bar. Row written BEFORE the run. |" >> /c/chlu/SCORED-LEDGER.md
echo "- $T LEDGER row maze-t-seed4 3/3 (seed 5) written BEFORE launch." >> $D/LOG.md
launch 5 BOTH test
wait_slot; launch 4 OBS val
wait_slot; launch 5 OBS val
echo QUEUE-DONE
