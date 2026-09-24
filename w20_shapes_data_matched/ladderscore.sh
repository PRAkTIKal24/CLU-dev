#!/bin/bash
# ladderscore.sh: every 5 min, score on each rung any w20 ladder rival / store that exists but is unscored (FREE replication).
source /c/chlu/outputs/w20-shapes-data-matched/env.sh
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
O=/c/chlu/outputs/w20-shapes-data-matched; S=/c/chlu/worktrees/CHLU-w20-shapes-data-matched/scripts
Om=C:/chlu/outputs/w20-shapes-data-matched
while true; do
  missing=0
  for k in ${RUNGS_ENV:-2 3 4 5 6}; do for s in 0 1 2; do
    arms=""; ctl=""
    for fam in mlp lstm transformer cont_mlp; do for st in declared tenx; do
      [ "$st" = tenx ] && [ "$fam" != mlp ] && [ "$fam" != cont_mlp ] && { [ $s != 0 ] || { [ $k != 2 ] && [ $k != 6 ]; }; } && continue
      nm=${fam}__${st}
      [ -f $O/scores/rung$k/seed$s/${nm}@rung$k.json ] && continue
      missing=$((missing+1))
      [ -f $O/rivals/rung$k/seed$s/$fam/$st.json ] && arms="$arms,rival:rung$k:$nm"
    done; done
    for sd in K32_n16384 K5_n16384 K32_n32768 K5_n32768; do
      [ -f $O/scores/rung$k/seed$s/store_${sd}__random_atoms.json ] && continue
      missing=$((missing+1))
      [ -f $O/stores/rung$k/seed$s/$sd/info.json ] && arms="$arms,store:$Om/stores/rung$k/seed$s/$sd/arm.pkl:store_$sd"
    done
    if [ -n "$arms" ]; then echo "[$(date -u +%H:%M:%SZ)] rung $k seed $s: ${arms#,}"; python $S/w20_score.py --corners rung:$k --seed $s --arms "${arms#,}"; fi
    if [ ! -f $O/scores/rung$k/seed$s/store_K32_n131072__landscape_shuffled_across_wells.json ]; then
      missing=$((missing+1))
      if [ -f $O/stores/rung$k/seed$s/K32_n131072/info.json ]; then
        python $S/w20_score.py --corners rung:$k --seed $s --arms "store:$Om/stores/rung$k/seed$s/K32_n131072/arm.pkl:store_K32_n131072_REBUILT" --only-controls landscape_shuffled_across_wells
        python $S/w20_score.py --corners rung:$k --seed $s --no-controls --arms "store:$Om/stores/rung$k/seed$s/K32_n131072/arm.pkl:store_K32_n131072_REBUILT"
        cp $O/scores/rung$k/seed$s/store_K32_n131072_REBUILT__landscape_shuffled_across_wells.json $O/scores/rung$k/seed$s/store_K32_n131072__landscape_shuffled_across_wells.json
      fi
    fi
  done; done
  echo "[$(date -u +%H:%M:%SZ)] missing=$missing"
  [ $missing -eq 0 ] && break
  sleep 300
done
echo "LADDERSCORE DONE"
