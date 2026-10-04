#!/bin/bash
source /c/chlu/outputs/w20-shapes-data-matched/env.sh
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
O=/c/chlu/outputs/w20-shapes-data-matched; S=/c/chlu/worktrees/CHLU-w20-shapes-data-matched/scripts
while true; do
  missing=0
  for s in 0 1 2; do arms=""
    for sc in value conjunction_live item; do for fam in mlp lstm transformer cont_mlp; do
      [ -f $O/scores/rows34/seed$s/${fam}__declared@test_del_$sc.json ] && continue
      missing=$((missing+1))
      [ -f $O/rivals/test_del_$sc/seed$s/$fam/declared.json ] && arms="$arms,rival:test_del_$sc:${fam}__declared"
    done; done
    [ -n "$arms" ] && python $S/w20_score.py --corners rows34 --run-n 3 --seed $s --arms "${arms#,}"
  done
  echo "[$(date -u +%H:%M:%SZ)] missing=$missing"; [ $missing -eq 0 ] && break; sleep 300
done; echo "DELSCORE DONE"
