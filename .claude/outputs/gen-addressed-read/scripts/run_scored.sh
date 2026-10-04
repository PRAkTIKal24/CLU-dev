#!/bin/zsh
# gen-addressed-read — THE SCORED RUN: `x0`, the ADDRESSED READ, as an arm.
#
# ⭐ The cell is NOT re-made. `gen-read-cell-arm`'s manifests are COPIED in first, so
#   every rival row (clu, store_removed, knn_proto, lace, ebm_compose) AND the dynread
#   rows on this table are the published ones bit-for-bit, and the only thing this run
#   adds is the arm + its charter §C floors. ⛔ A cell of our own would make the row
#   incomparable, which is the whole point.
# ⭐ Per-ROW resumable: a row already in the manifest is SKIPPED.
# ⛔ No --quick anywhere. ⛔ CPU only (charter §D: this is a floor measurement).
set -u
WT=/Users/user/Desktop/CHLU-addrread
REPO=/Users/user/Desktop/CHLU
OUT=$REPO/.claude/outputs/gen-addressed-read/runs
SRC=$REPO/.claude/outputs/gen-read-cell-arm/runs/cell
PY=$REPO/.venv/bin/python
export PYTHONPATH=$WT
export CHLU_CLAUDE_ROOT=$REPO/.claude
export CHLU_GEN_DATA_ROOT=$REPO/.claude/data/gen
export CHLU_GEN_BRIDGE_ROOT=$REPO/.claude/outputs/gen-feature-bridge/runs
export CHLU_GEN_DECLIFT_DIR=$REPO/.claude/outputs/gen-decoder-lift/winner
export CHLU_GEN_STORE_RUNS=$REPO/.claude/outputs/gen-store-grouping/runs
export CHLU_GEN_X0_ARTEFACTS=$OUT/artefacts
export JAX_PLATFORMS=cpu
cd $WT
mkdir -p $OUT/cell
echo "HEAD $(git rev-parse HEAD)  $(date +%F_%T)" | tee -a $OUT/progress.txt

for s in 0 1 2; do
  d=$OUT/cell/seed_$s
  mkdir -p $d
  if [ ! -f "$d/cell_manifest.json" ]; then
    cp $SRC/seed_$s/seed_$s/cell_manifest.json $d/cell_manifest.json
    echo "[$(date +%H:%M:%S)] SEEDED seed $s from gen-read-cell-arm's cell" | tee -a $OUT/progress.txt
  fi
done

for s in 0 1 2; do
  echo "[$(date +%H:%M:%S)] START seed $s" | tee -a $OUT/progress.txt
  $PY -u -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon \
      --stage cell --seed $s --arm x0 --out $OUT/cell \
      --config $REPO/.claude/outputs/gen-val-split-fix/cell_rescore.yaml \
      > $OUT/x0_seed$s.log 2>&1
  rc=$?   # ⛔ captured BEFORE any command substitution, which would reset it
  echo "[$(date +%H:%M:%S)] END   seed $s rc=$rc" | tee -a $OUT/progress.txt
  if [ $rc -ne 0 ]; then
    echo "⛔ seed $s FAILED — see $OUT/x0_seed$s.log" | tee -a $OUT/progress.txt
    tail -5 $OUT/x0_seed$s.log | tee -a $OUT/progress.txt
  fi
done

echo "[$(date +%H:%M:%S)] AGGREGATE" | tee -a $OUT/progress.txt
$PY -u -m chlu.experiments.exp_gen_cell --dataset shapes3d --pretext recon \
    --stage aggregate --seed 0 --seed 1 --seed 2 --out $OUT/cell \
    --config $REPO/.claude/outputs/gen-val-split-fix/cell_rescore.yaml \
    > $OUT/aggregate.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] ALL DONE rc=$rc" | tee -a $OUT/progress.txt
