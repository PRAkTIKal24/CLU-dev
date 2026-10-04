#!/usr/bin/env bash
CODE=C:/chlu/outputs/hub-unseen-valid/w4/code
RUNS=/c/chlu/outputs/hub-unseen-valid/w4/runs
until [ -f $RUNS/score_run1_AGG.json ]; do sleep 30; done
echo "=== scored run finished; filling all seven rows ==="
for S in 0 1 2; do
  OMP_NUM_THREADS=3 python -u $CODE/rows_all.py $S factor_aligned_balanced,knn_proto,basis_head,landscape_removed,landscape_shuffled 5400
done
echo "ALL ROWS DONE"
