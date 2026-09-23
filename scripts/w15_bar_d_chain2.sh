#!/usr/bin/env bash
# w15: per-seed chain 2 -- after the side track's rivals(alone) has finished, fit rivals
# (loop_swap) and score it on val. Serialises rivals so no two fits ever share a checkpoint.
set -uo pipefail
S="${1:?seed}"; RUN=C:/chlu/outputs/w15-bar-d-score/gpu-run; SC=C:/chlu/outputs/w15-bar-d-score/score
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export JAX_PLATFORMS=cpu PYTHONIOENCODING=utf-8 XLA_FLAGS="--xla_force_host_platform_device_count=1"
ok() { python -c "import json,sys;j=json.load(open(sys.argv[1]));sys.exit(0 if j.get('status')=='ok' and 'mlp_180k' in j.get('families_done',[]) else 1)" "$1" 2>/dev/null; }
until ok "$RUN/seed_$S/alone/rivals.json"; do sleep 30; done
echo "[chain2 s$S $(date -u +%H:%M:%SZ)] rivals(alone) done -> rivals(loop_swap)"
python -u -m chlu.experiments.exp_w8_encoder_loop --stage rivals --seed "$S" --arm loop_swap --out "$RUN" \
  --families mlp_180k --joint-steps 4000 --n-train 32768 --dec-steps 1800 --warm-steps 4000 || exit 3
bash "$(dirname "$0")/w15_bar_d_score.sh" val "$S" loop_swap "$RUN" "$SC" pca_tensor rival_mlp_180k pca_landscape_removed || exit 3
echo "[chain2 s$S $(date -u +%H:%M:%SZ)] done"
