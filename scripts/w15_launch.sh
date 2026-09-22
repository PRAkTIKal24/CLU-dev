#!/usr/bin/env bash
# w15 spoke/crossover-seeds launcher.  One lane, own log, OWN PID recorded.
# NEVER a box-wide kill -- only the PIDs in PIDS.txt are this spoke's to stop.
#
# CHLU_GEN_FEATURES_ROOT is set HERE and nowhere else: it is not in this box's ambient
# environment, and a run that forgets it silently reads the frozen bridge (cache
# GENERATION 1), which w5-common 3.8 forbids for any number.
#
# PYTHONPATH is rebuilt with ';' and Windows-form paths: /c/chlu/env_w3.sh joins its two
# entries with a POSIX ':' and native python then sees ONE broken path, so equinox/jax
# vanish and chlu fails to import.
set -u
OUTD=/c/chlu/outputs/w15-crossover-seeds
LOGDIR=$OUTD/logs
tag="$1"; shift
export CHLU_ROOT=/c/chlu/worktrees/CHLU-w15-crossover-seeds
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export PYTHONPATH="C:/chlu/worktrees/CHLU-w15-crossover-seeds;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages"
# 4 lanes on a 4-core box -> one BLAS thread per lane; oversubscribed BLAS is slower
# than serial BLAS, and the address-book fit is the whole cost of a leg.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd /c/chlu/worktrees/CHLU-w15-crossover-seeds
python -u "$@" > "$LOGDIR/$tag.log" 2>&1 &
pid=$!
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$tag pid=$pid cmd=$*" >> "$OUTD/PIDS.txt"
echo "$pid"
