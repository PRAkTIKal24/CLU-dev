#!/bin/bash
# CAMELS regional exemplar-store ladder, per PREREG.md §2/§3.
set -u
cd /Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire
P=./.venv/bin/python
OUT=/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire/arms.jsonl
for SC in perbasin global raw; do
  # ---- 30-day variant (at-budget L=2761) ----
  $P camels_tripwire.py --window 30 --scaling $SC --Ls 250,500,1000,2000,2761 \
     --seeds 0,1,2 --kmeans 1 --sub 1 --tag inbudget --out $OUT
  $P camels_tripwire.py --window 30 --scaling $SC --Ls 5000 \
     --seeds 0 --kmeans 0 --sub 3 --tag overbudget --out $OUT
  # ---- 1-day variant (declared extra; at-budget L=14894) ----
  $P camels_tripwire.py --window 1 --scaling $SC --Ls 250,1000,5000,14894 \
     --seeds 0,1,2 --kmeans 0 --sub 1 --tag inbudget --out $OUT
  # ---- 365-day variant (at-budget L=265) ----
  $P camels_tripwire.py --window 365 --scaling $SC --Ls 250,265 \
     --seeds 0,1,2 --kmeans 1 --sub 1 --tag inbudget --out $OUT
  $P camels_tripwire.py --window 365 --scaling $SC --Ls 500,1000,2000 \
     --seeds 0 --kmeans 1 --sub 3 --tag overbudget --out $OUT
  $P camels_tripwire.py --window 365 --scaling $SC --Ls 5000 \
     --seeds 0 --kmeans 0 --sub 3 --tag overbudget --out $OUT
  echo "=== scaling $SC done $(date) ==="
done
echo "ALL REGIONAL DONE $(date)"
