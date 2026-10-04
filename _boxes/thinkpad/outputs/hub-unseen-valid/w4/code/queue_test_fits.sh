#!/usr/bin/env bash
CODE=C:/chlu/outputs/hub-unseen-valid/w4/code
# wait for the gen-2 span job to release the threads
while pgrep -f "j_span_full.py" >/dev/null 2>&1; do sleep 30; done
for S in 0 1 2; do
  echo "=== test fit seed $S ==="
  OMP_NUM_THREADS=3 python -u $CODE/fit_test.py $S
done
echo "ALL TEST FITS DONE"
