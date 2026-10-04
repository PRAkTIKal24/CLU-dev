#!/bin/zsh
# gen-atom-mean-resweep — the n_val CALIBRATION sweep.
# ⭐ Per-(point, seed) resumable: a unit whose atom_manifest.json exists is SKIPPED.
# ⛔ NO --quick anywhere (apply_quick REPLACES n_val/tune_n; the CLI now refuses the pair).
# ⭐ tune_n tracks n_val, or the selection still sees 256 rows and the curve is flat
#   for a PLUMBING reason.
# ⚠ --allow-knn-proto-move only where n_fit was moved ON PURPOSE (n_val > 512): the
#   rival is re-fitted on fewer rows so it MUST move, and that move is the PRICE.
set -u
WT=/Users/user/Desktop/CHLU-resweep
REPO=/Users/user/Desktop/CHLU
OUT=$REPO/.claude/outputs/gen-atom-mean-resweep/runs
PY=$REPO/.venv/bin/python
export PYTHONPATH=$WT
export CHLU_CLAUDE_ROOT=$REPO/.claude
export CHLU_GEN_DATA_ROOT=$REPO/.claude/data/gen
export CHLU_GEN_BRIDGE_ROOT=$REPO/.claude/outputs/gen-feature-bridge/runs
export CHLU_GEN_DECLIFT_DIR=$REPO/.claude/outputs/gen-decoder-lift/winner
export CHLU_GEN_STORE_RUNS=$REPO/.claude/outputs/gen-store-grouping/runs
export JAX_PLATFORMS=cpu
cd $WT
mkdir -p $OUT
echo "HEAD $(git rev-parse HEAD) @ $(date +%H:%M:%S)" | tee -a $OUT/progress.txt

run_point () {   # label n_val tune_n seed extra...
  lab=$1; nv=$2; tn=$3; s=$4; shift 4
  d=$OUT/$lab
  if [ -f "$d/seed_$s/atom_manifest.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $lab seed $s (manifest exists)" | tee -a $OUT/progress.txt
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $lab seed $s (n_val=$nv tune_n=$tn)" | tee -a $OUT/progress.txt
  t0=$(date +%s)
  $PY -u -m chlu.experiments.exp_gen_atom_mean --dataset shapes3d --pretext recon \
      --stage atoms --seed $s --out $d --n-val $nv --tune-n $tn "$@" \
      > $OUT/${lab}_seed$s.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   $lab seed $s rc=$rc ($(( $(date +%s) - t0 ))s)" | tee -a $OUT/progress.txt
  return 0        # ⛔ a failed unit is NOT RUN and must not abort the rest of the sweep
}

POINT=${1:-all}

# ⭐ THE ANCHOR — the shipped pair, i.e. gen-val-split-fix leg A exactly.
# ⛔ NO waiver: n_fit is 32,256 here, so knn_proto MUST come back bit-identical.
if [ "$POINT" = "all" ] || [ "$POINT" = "anchor" ]; then
  for s in 0 1 2; do run_point nval0512_tune0256 512 256 $s; done
fi

# ⭐ n_val = 512 with the selection rendering ALL of it (a different cell to the anchor).
if [ "$POINT" = "all" ] || [ "$POINT" = "512" ]; then
  for s in 0 1 2; do run_point nval0512_tune0512 512 512 $s; done
fi

# ⚠ from here n_fit comes down arithmetically (32,768 - n_val) and the rival is re-fitted.
for nv in 1024 2048 4096 8192; do
  if [ "$POINT" = "all" ] || [ "$POINT" = "$nv" ]; then
    lab=$(printf "nval%04d_tune%04d" $nv $nv)
    for s in 0 1 2; do run_point $lab $nv $nv $s --allow-knn-proto-move; done
  fi
done

echo "[$(date +%H:%M:%S)] POINT $POINT DONE" | tee -a $OUT/progress.txt
