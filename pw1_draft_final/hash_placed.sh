#!/usr/bin/env bash
# usage: hash_placed.sh OUTDIR TAG  -- sha256 of the 26 placed figure files in both folders
set -euo pipefail
S="$1"; TAG="$2"
STEMS="fig_task_shapes fig_task_cmazet rendered/rc_pw1_shapes_per_model rendered/rc_pw1_shapes_baseline_wins fig_ladder fig_crossover fig_deletion fig_encoder_adapt fig_maze_conditions fig_cmazet_families fig_cmazet_per_arm fig_cmazet_rel_per_arm fig_cmazet_door_then_tint"
FILES=(); for s in $STEMS; do FILES+=("$s.pdf" "$s.png"); done
for F in paper-visuals w17-paper/figures; do
  n=$(echo $F | tr / _)
  (cd /Users/batmac/clu/outputs/$F && shasum -a 256 "${FILES[@]}") > "$S/PLACED-$TAG-$n.sha256"
  echo "$n: $(wc -l < "$S/PLACED-$TAG-$n.sha256") placed files hashed"
done
