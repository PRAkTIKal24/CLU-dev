#!/usr/bin/env bash
# w15/lace-poe-rows: lock-based ladder worker.  usage: w15_lace_poe_queue.sh WORKER rung:seed ...
# A leg is claimed by an atomic mkdir of ladder/<leg>/.claimed; a claimed leg is skipped.
O=/c/chlu/outputs/w15-lace-poe-rows
source $O/env_ladder.sh
w=$1; shift
for rs in "$@"; do
  r=${rs%%:*}; s=${rs##*:}; d=$O/ladder/${r}_seed${s}
  mkdir -p $d
  if ! mkdir $d/.claimed 2>/dev/null; then continue; fi
  echo "$(date -u +%FT%TZ) worker $w claims $r seed $s" >> $O/logs/chains.log
  python -u scripts/w15_lace_poe_ladder.py --rung $r --seed $s --out $O/ladder > $O/logs/ladder_${r}_seed${s}.log 2>&1
  echo "$(date -u +%FT%TZ) worker $w end $r seed $s rc=$?" >> $O/logs/chains.log
done
echo "$(date -u +%FT%TZ) worker $w DONE" >> $O/logs/chains.log
