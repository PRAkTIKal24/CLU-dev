#!/usr/bin/env bash
# w15/lace-poe-rows: run ladder legs sequentially.  usage: w15_lace_poe_chain.sh NAME rung:seed ...
O=/c/chlu/outputs/w15-lace-poe-rows
source $O/env_ladder.sh
name=$1; shift
for rs in "$@"; do
  r=${rs%%:*}; s=${rs##*:}
  echo "$(date -u +%FT%TZ) chain $name start $r seed $s" >> $O/logs/chains.log
  python -u scripts/w15_lace_poe_ladder.py --rung $r --seed $s --out $O/ladder > $O/logs/ladder_${r}_seed${s}.log 2>&1
  echo "$(date -u +%FT%TZ) chain $name end $r seed $s rc=$?" >> $O/logs/chains.log
done
echo "$(date -u +%FT%TZ) chain $name DONE" >> $O/logs/chains.log
