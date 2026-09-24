#!/bin/bash
# autoscore.sh: every 5 min, score on rows 3-4 (run 1/3) any w20 rival checkpoint / K=5 store that
# exists but has no score file yet. Exits when every expected item is scored.
source /c/chlu/outputs/w20-shapes-data-matched/env.sh
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
O=/c/chlu/outputs/w20-shapes-data-matched; S=/c/chlu/worktrees/CHLU-w20-shapes-data-matched/scripts
while true; do
  missing=0
  for s in 0 1 2; do
    arms=""
    for tag in test test16k_stepmatched test32k_stepmatched test30x; do
      for fam in mlp lstm transformer cont_mlp; do
        for st in declared tenx; do
          [ "$tag" != test ] && { [ "$st" != declared ] || { [ "$fam" != mlp ] && { [ "$tag" != test30x ] || [ "$fam" != cont_mlp ]; }; }; } && continue
          nm=${fam}__${st}
          if [ -f $O/scores/rows34/seed$s/${nm}@${tag}.json ]; then continue; fi
          missing=$((missing+1))
          [ -f $O/rivals/$tag/ckpt/seed_$s/$nm.rival.eqx ] && [ -f $O/rivals/$tag/seed$s/$fam/$st.json ] && arms="$arms,rival:$tag:$nm"
        done
      done
    done
    if [ $s -gt 0 ] && [ ! -f $O/scores/rows34/seed$s/store_K5_n131072.json ]; then
      missing=$((missing+1))
      [ -f $O/stores/test/seed$s/K5_n131072/info.json ] && arms="$arms,store:C:/chlu/outputs/w20-shapes-data-matched/stores/test/seed$s/K5_n131072/arm.pkl:store_K5_n131072"
    fi
    if [ -n "$arms" ]; then
      echo "[$(date -u +%H:%M:%SZ)] scoring seed $s: ${arms#,}"
      python $S/w20_score.py --corners rows34 --run-n 1 --seed $s --arms "${arms#,}"
    fi
  done
  echo "[$(date -u +%H:%M:%SZ)] missing=$missing"
  [ $missing -eq 0 ] && break
  sleep 300
done
echo "AUTOSCORE DONE"
