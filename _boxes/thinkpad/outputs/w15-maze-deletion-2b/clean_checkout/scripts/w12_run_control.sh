#!/usr/bin/env bash
# w12 CONTROL: refit seed 2's bridge ENCODER on THIS box with the identical shipped
# command, so the report can say - measured, not assumed - whether an encoder trained
# here reproduces the frozen generation-1 encoder trained on the production box.
# Compares param_hash against $CHLU_GEN_BRIDGE_ROOT/shapes3d_recon/seed_2.
set -u
OUT=/c/chlu/outputs/w12-cache-gen2-seeds-3-4
export PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /c/chlu/worktrees/CHLU-w12-cache-gen2-seeds-3-4
D="$OUT/control/seed_2_encoder_rebuild"
if [ ! -f "$D/encoder_manifest.json" ]; then
  echo "=== control: refit seed 2 encoder, start $(date -u +%H:%M:%S) ==="
  python -u -m chlu.experiments.exp_gen_bridge \
      --dataset shapes3d --pretext recon --stage encoder --seed 2 \
      --cache-scope used --no-plots --out "$D" >> "$OUT/control_seed_2_encoder.log" 2>&1
  echo "=== control rc=$? end $(date -u +%H:%M:%S) ==="
fi
