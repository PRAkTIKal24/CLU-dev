#!/usr/bin/env bash
# w5.1/refuse -- the bridge-seed spread driver.
# NO SCORED RUN: every stage here re-measures the SPREAD of an already-scored quantity
# under a wider instrument. Both arms are run by ME, in ONE environment, so that the only
# thing differing between the two columns is which bridge seed the store seed selects.
set -u
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
# Windows stdout defaults to cp1252 and the runner logs a star. Without this the process
# dies on a LOG LINE, not on a number. (w5.1 LOG 2026-09-20T01:35Z)
export PYTHONIOENCODING=utf-8
# THREE THREADS, and it is a NUMBER-BEARING setting, not a speed knob. MEASURED (w5.1 LOG
# 2026-09-20T02:05Z): at OMP_NUM_THREADS=1 my seed-0 store hashes 617e322125fb6754...; at
# 3 it hashes 697d60dfa3e4b70c6822e04af1abeb3b -- BIT-IDENTICAL to the parent's and to
# w5 hub/delete's. The difference is float32 ULP noise in coupling_u / coupling_v only
# (30/10,576 and 1,441/245,760 elements, max abs 1e-7), because build_field takes a LAPACK
# np.linalg.svd whose reduction order depends on the BLAS thread count. Every other
# segment -- atoms, basins, term map, coupling weights, hole flags -- is bit-identical at
# either setting. 3 threads each means the stages run ONE AT A TIME.
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_PLATFORMS=cpu
RT=/c/chlu/outputs/w51-refuse/runs
L=$RT/logs
P=/c/chlu/outputs/w51-refuse/MY_PIDS.txt
mkdir -p "$L"
cd /c/chlu/worktrees/CHLU-w51-refuse
echo "# w5.1/refuse PIDs -- kill ONLY these. driver=$$ (started $(date -u +%FT%TZ))" > "$P"

stage () {  # stage <mode> <seed> <stagename>
  local m=$1 s=$2 st=$3
  python -m chlu.experiments.exp_gen_refuse --stage "$st" --seed "$s" \
      --bridge-mode "$m" --out "$RT/$m" >"$L/${st}_${m}_s$s.log" 2>&1 &
  local pid=$!
  echo "$pid ${st}_${m}_s$s" >> "$P"
  wait $pid
  echo "$st $m s$s exit=$?"
  tail -3 "$L/${st}_${m}_s$s.log"
}

echo "=== stage seed0check (both modes, one store build each) ==="
python -m chlu.experiments.exp_gen_refuse --stage seed0check --out "$RT" \
    >"$L/seed0check.log" 2>&1
echo "seed0check exit=$?"
tail -4 "$L/seed0check.log"

echo "=== stage fit, BOTH modes, seeds 0/1/2 ==="
for m in pinned joint; do for s in 0 1 2; do stage "$m" "$s" fit; done; done

echo "=== stage deletion, BOTH modes, seeds 0/1/2 -- SEQUENTIAL at 3 threads ==="
for m in pinned joint; do for s in 0 1 2; do stage "$m" "$s" deletion; done; done

echo "=== DRIVER DONE ==="
