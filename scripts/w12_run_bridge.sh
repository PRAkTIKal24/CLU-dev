#!/usr/bin/env bash
# w12: build the seed-3 and seed-4 bridges with the SHIPPED exp_gen_bridge.
set -u
OUT=/c/chlu/outputs/w12-cache-gen2-seeds-3-4
export PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /c/chlu/worktrees/CHLU-w12-cache-gen2-seeds-3-4
for S in "$@"; do
  D="$OUT/bridge/shapes3d_recon/seed_$S"
  if [ -f "$D/bridge_manifest.json" ]; then echo "seed $S bridge already complete"; continue; fi
  echo "=== bridge seed $S start $(date -u +%H:%M:%S) ==="
  python -u -m chlu.experiments.exp_gen_bridge \
      --dataset shapes3d --pretext recon --stage all --seed "$S" \
      --cache-scope used --no-plots --out "$D" >> "$OUT/bridge_seed_$S.log" 2>&1
  echo "=== bridge seed $S rc=$? end $(date -u +%H:%M:%S) ==="
done
