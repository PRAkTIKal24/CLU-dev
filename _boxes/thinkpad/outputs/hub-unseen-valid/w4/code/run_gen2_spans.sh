#!/usr/bin/env bash
CODE=C:/chlu/outputs/hub-unseen-valid/w4/code
for S in 1 2 0; do
  while true; do
    if python -c "
import sys; sys.path.insert(0,'$CODE')
import w4_common as C
sys.exit(0 if C.gen2_ready($S) else 1)"; then break; fi
    sleep 60
  done
  echo "=== gen2 span seed $S ==="
  OMP_NUM_THREADS=3 python -u $CODE/j_span_full.py --n 4096 --seed $S --batch 128 --gen2
done
