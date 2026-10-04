#!/usr/bin/env bash
CODE=C:/chlu/outputs/hub-unseen-valid/w4/code
RUNS=/c/chlu/outputs/hub-unseen-valid/w4/runs
until [ -f $RUNS/score_rep1_s0.json ]; do sleep 30; done
echo "=== replication done; running the distinct-output diagnostic ==="
OMP_NUM_THREADS=2 python -u $CODE/j_diversity.py 0
