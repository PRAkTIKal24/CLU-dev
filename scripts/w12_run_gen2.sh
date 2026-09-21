#!/usr/bin/env bash
# w12: determinism falsifier, then the full-scope generation-2 build, per seed.
set -u
OUT=/c/chlu/outputs/w12-cache-gen2-seeds-3-4
export PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /c/chlu/worktrees/CHLU-w12-cache-gen2-seeds-3-4
for S in "$@"; do
  echo "=== gen2 seed $S start $(date -u +%H:%M:%S) ==="
  python -u scripts/w12_cache_gen2_seeds_3_4.py --stage det    --seeds "$S" >> "$OUT/gen2_seed_$S.log" 2>&1
  rc=$?; echo "  det rc=$rc"
  if [ $rc -ne 0 ]; then echo "STOP: determinism failed on seed $S"; continue; fi
  python -u scripts/w12_cache_gen2_seeds_3_4.py --stage cache2 --seeds "$S" >> "$OUT/gen2_seed_$S.log" 2>&1
  echo "  cache2 rc=$? $(date -u +%H:%M:%S)"
  python -u scripts/w12_cache_gen2_seeds_3_4.py --stage gate   --seeds 3 4 >> "$OUT/gen2_seed_$S.log" 2>&1
  echo "  gate rc=$? $(date -u +%H:%M:%S)"
done
