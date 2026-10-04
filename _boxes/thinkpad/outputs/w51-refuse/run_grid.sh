#!/usr/bin/env bash
# w5.1/refuse -- the 3x3 (store seed) x (bridge seed) GRID.
# NO SCORED RUN. Every cell re-reads rows the parent already paid for.
#
# Why a grid and not just two spreads. `pinned` and `joint` each give one three-seed
# number, but the joint spread CONFOUNDS the store draw with the instrument and cannot say
# which is doing the work. The grid separates them:
#   down column bridge=0    -> the STORE effect at a fixed instrument = the parent's own
#                              spread, which is exactly what it said it measured;
#   across a row, fixed s   -> the INSTRUMENT effect alone, which neither hub has measured;
#   the diagonal            -> what `joint` and `w5 hub/delete` report.
#
# runs/pinned already holds the bridge=0 column, so this script fills bridge=1 and
# bridge=2. Stages run ONE AT A TIME at 3 threads, for the reason in run_joint.sh.
set -u
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_PLATFORMS=cpu
RT=/c/chlu/outputs/w51-refuse/runs
L=$RT/logs
P=/c/chlu/outputs/w51-refuse/MY_PIDS.txt
mkdir -p "$L"
cd /c/chlu/worktrees/CHLU-w51-refuse
echo "# grid driver=$$ (started $(date -u +%FT%TZ))" >> "$P"

cell () {  # cell <bridge> <storeseed> <stagename>
  local b=$1 s=$2 st=$3
  python -m chlu.experiments.exp_gen_refuse --stage "$st" --seed "$s" \
      --bridge-seed "$b" --out "$RT/bridge$b" >"$L/${st}_b${b}_s$s.log" 2>&1 &
  local pid=$!
  echo "$pid ${st}_b${b}_s$s" >> "$P"
  wait $pid
  echo "$st bridge=$b store=$s exit=$?"
  tail -3 "$L/${st}_b${b}_s$s.log"
}

for b in 1 2; do
  # the partition is thread- and bridge-independent (verified: bit-identical to the
  # parent's on all 11 arrays and the JSON), so it is copied rather than rebuilt.
  mkdir -p "$RT/bridge$b"
  [ -d "$RT/bridge$b/partitions" ] || cp -r "$RT/pinned/partitions" "$RT/bridge$b/partitions"
  for s in 0 1 2; do cell "$b" "$s" fit; done
  for s in 0 1 2; do cell "$b" "$s" deletion; done
done

echo "=== GRID DONE ==="
