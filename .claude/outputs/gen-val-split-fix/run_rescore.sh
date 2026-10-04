#!/bin/zsh
# gen-val-split-fix — the RE-SCORE on the repaired (stratified) fit/val split.
# ⭐ Per-unit, resumable: a unit whose manifest exists is SKIPPED.
# ⛔ Every family runs through the SAME repaired split: ours AND both rivals.
#   H  = baselines, historical 688,128 B, --arm all  -> LACE's published leg + the
#        R27.3 whitening-exactness fallback + kNN historical
#   D  = baselines, deployed 524,288 B, --arm knn    -> knn_proto's published reference
#   C  = the cell harness (ebm_compose selection + the CLU rows + inherited LACE)
#   A  = the atom-mean control (our tuned arm, 2 budgets)
set -u
WT=/Users/user/Desktop/CHLU-gen-valfix
REPO=/Users/user/Desktop/CHLU
OUT=$REPO/.claude/outputs/gen-val-split-fix/runs
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
echo "HEAD $(git rev-parse HEAD)" | tee -a $OUT/progress.txt

run_one () {  # leg module manifest outdir seed extra...
  leg=$1; mod=$2; man=$3; d=$4; s=$5; shift 5
  if [ -f "$d/seed_$s/$man" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $leg seed $s (manifest exists)" | tee -a $OUT/progress.txt
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $leg seed $s" | tee -a $OUT/progress.txt
  $PY -u -m $mod --dataset shapes3d --pretext recon --seed $s \
      --out $d/seed_$s "$@" > $OUT/${leg}_seed$s.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   $leg seed $s rc=$rc" | tee -a $OUT/progress.txt
  return $rc
}

STAGE=${1:-all}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "D" ]; then
for s in 0 1 2; do
  run_one D chlu.experiments.exp_gen_baselines baselines_manifest.json \
          $OUT/deployed_524288 $s --stage baselines --arm knn
done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "H" ]; then
for s in 0 1 2; do
  run_one H chlu.experiments.exp_gen_baselines baselines_manifest.json \
          $OUT/historical_688128 $s --stage baselines --arm all --byte-budget historical
done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "C" ]; then
for s in 0 1 2; do
  run_one C chlu.experiments.exp_gen_cell cell_manifest.json \
          $OUT/cell $s --stage cell --config $REPO/.claude/outputs/gen-val-split-fix/cell_rescore.yaml
done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "A" ]; then
for s in 0 1 2; do
  run_one A chlu.experiments.exp_gen_atom_mean atom_manifest.json \
          $OUT/atom_mean $s --stage atoms
done
fi

echo "[$(date +%H:%M:%S)] STAGE $STAGE DONE" | tee -a $OUT/progress.txt
