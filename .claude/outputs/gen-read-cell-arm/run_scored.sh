#!/bin/zsh
# gen-read-cell-arm — the SCORED run: `dynread` as a sixth arm on the REPAIRED-split cell.
#
# ⭐ The cell is NOT re-made.  gen-val-split-fix's repaired-split manifests are COPIED
#   in first, so the five rival rows on this table are the published ones bit-for-bit
#   and the only thing this run adds is the arm.  ⛔ A cell of our own would make the
#   row incomparable, which is the whole point.
# ⭐ Per-row resumable: a row already in the manifest is SKIPPED, so a kill -9 costs
#   at most the rung in flight.
# ⛔ No --quick anywhere: --quick clobbers explicit flags and a quick number is never reported.
set -u
WT=/Users/user/Desktop/CHLU-cellarm
REPO=/Users/user/Desktop/CHLU
OUT=$REPO/.claude/outputs/gen-read-cell-arm/runs
SRC=$REPO/.claude/outputs/gen-val-split-fix/runs/cell
PY=$REPO/.venv/bin/python
export PYTHONPATH=$WT
export CHLU_CLAUDE_ROOT=$REPO/.claude
export CHLU_GEN_DATA_ROOT=$REPO/.claude/data/gen
export CHLU_GEN_BRIDGE_ROOT=$REPO/.claude/outputs/gen-feature-bridge/runs
export CHLU_GEN_DECLIFT_DIR=$REPO/.claude/outputs/gen-decoder-lift/winner
export CHLU_GEN_STORE_RUNS=$REPO/.claude/outputs/gen-store-grouping/runs
export CHLU_GEN_CELL_ARTEFACTS=$OUT/artefacts
export JAX_PLATFORMS=cpu
cd $WT
mkdir -p $OUT
echo "HEAD $(git rev-parse HEAD)  $(date +%F_%T)" | tee -a $OUT/progress.txt

for s in 0 1 2; do
  d=$OUT/cell/seed_$s/seed_$s
  mkdir -p $d
  if [ ! -f "$d/cell_manifest.json" ]; then
    cp $SRC/seed_$s/seed_$s/cell_manifest.json $d/cell_manifest.json
    echo "[$(date +%H:%M:%S)] SEEDED seed $s from the repaired-split cell" | tee -a $OUT/progress.txt
  fi
done

for s in 0 1 2; do
  echo "[$(date +%H:%M:%S)] START seed $s" | tee -a $OUT/progress.txt
  $PY -u -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon \
      --stage cell --seed $s --arm dynread --out $OUT/cell/seed_$s \
      --config $REPO/.claude/outputs/gen-val-split-fix/cell_rescore.yaml \
      > $OUT/dynread_seed$s.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   seed $s rc=$rc" | tee -a $OUT/progress.txt
done
echo "[$(date +%H:%M:%S)] ALL SEEDS DONE" | tee -a $OUT/progress.txt
