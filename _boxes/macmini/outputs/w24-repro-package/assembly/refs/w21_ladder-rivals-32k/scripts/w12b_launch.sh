#!/usr/bin/env bash
# w12b hub/nr6-and-kstar launcher.  ONE thread per lane, 3 lanes = the 3 threads the
# task file allows.  OWN PID recorded; ⛔ never a box-wide kill -- only the PIDs in
# this hub's own PIDS.txt are ever this hub's to stop, and `hub/encoder-adapt` is
# TRAINING on this box.
#
# CHLU_GEN_FEATURES_ROOT is set HERE and nowhere else: unset, features_root() falls
# back to the frozen bridge = cache GENERATION 1, which w5-common §3.8 forbids.
set -u
LOGDIR=/c/chlu/outputs/w12b-nr6-and-kstar/logs
mkdir -p "$LOGDIR"
tag="$1"; shift
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python -u "$@" > "$LOGDIR/$tag.log" 2>&1 &
pid=$!
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$tag pid=$pid cmd=$*" >> "$LOGDIR/PIDS.txt"
echo "$pid"
