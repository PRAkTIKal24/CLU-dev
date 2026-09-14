#!/bin/zsh
# gen-arms-a7-wiring — the two refusal legs.
#  H = the HISTORICAL byte budget (688,128 B, --byte-budget historical): reproduces the
#      published configuration exactly, so `out_of_range` is a bit-identity regression
#      and the A7-controlled dup column is directly comparable to the published one.
#  D = the DEPLOYED budget (524,288 B, the new default), kNN only: LACE never reads the
#      byte target (source-verified: `byte_target` is consumed only by run_knn_arm).
# ⭐ per-seed, resumable: a seed whose manifest exists is SKIPPED.
set -u
WT=/Users/user/Desktop/CHLU-gen-a7wire
REPO=/Users/user/Desktop/CHLU
OUT=$REPO/.claude/outputs/gen-arms-a7-wiring/runs
PY=$REPO/.venv/bin/python
export PYTHONPATH=$WT
export CHLU_GEN_DATA_ROOT=$REPO/.claude/data/gen
export CHLU_GEN_BRIDGE_ROOT=$REPO/.claude/outputs/gen-feature-bridge/runs
export CHLU_GEN_DECLIFT_DIR=$REPO/.claude/outputs/gen-decoder-lift/winner
export CHLU_GEN_STORE_RUNS=$REPO/.claude/outputs/gen-store-grouping/runs
cd $WT
echo "HEAD $(git rev-parse HEAD)" | tee -a $OUT/progress.txt

run_one () {  # leg outdir seed extra...
  leg=$1; d=$2; s=$3; shift 3
  if [ -f "$d/seed_$s/baselines_manifest.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $leg seed $s (manifest exists)" | tee -a $OUT/progress.txt
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $leg seed $s" | tee -a $OUT/progress.txt
  $PY -u -m chlu.experiments.exp_gen_baselines --dataset shapes3d --pretext recon \
      --stage baselines --seed $s --out $d/seed_$s "$@" > $OUT/${leg}_seed$s.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   $leg seed $s rc=$rc" | tee -a $OUT/progress.txt
  return $rc
}

mkdir -p $OUT
for s in 0 1 2; do run_one H $OUT/historical_688128 $s --arm all --byte-budget historical; done
for s in 0 1 2; do run_one D $OUT/deployed_524288  $s --arm knn; done
echo "[$(date +%H:%M:%S)] ALL LEGS DONE" | tee -a $OUT/progress.txt
