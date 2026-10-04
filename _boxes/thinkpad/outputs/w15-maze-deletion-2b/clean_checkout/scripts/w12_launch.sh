#!/usr/bin/env bash
# w12 launcher: 3 threads, unbuffered, own log, OWN PID recorded.
# Never a box-wide kill -- only the PIDs in logs/PIDS.txt are ever this hub's to stop.
#
# CHLU_GEN_FEATURES_ROOT IS SET HERE AND NOWHERE ELSE.  It is NOT in this box's
# ambient environment, so a run that forgets it silently reads the frozen bridge --
# cache GENERATION 1, which w5-common section 3.8 forbids for any number, and which at
# bridge seeds 1 and 2 holds only 9,288 rows against the pool's 131,072 indices.
set -u
LOGDIR=/c/chlu/outputs/w12-single-design-crossover/logs
tag="$1"; shift
export CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export PYTHONUNBUFFERED=1
python -u "$@" > "$LOGDIR/$tag.log" 2>&1 &
pid=$!
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$tag bashpid=$pid cmd=$*" >> "$LOGDIR/PIDS.txt"
echo "$pid"
