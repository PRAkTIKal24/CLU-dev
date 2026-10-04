#!/bin/bash
# scripts/cloud/shutdown.sh -- GCE `shutdown-script` for a CHLU job VM.
#
# GCE runs this on a Spot PRE-EMPTION (and on any ordered stop), with roughly
# 30 seconds before the VM is powered off. `docker stop` sends SIGTERM to the
# payload, whose handler force-checkpoints and exits; we then push whatever is
# on disk one last time.
#
# This is the SECOND of two pre-emption paths. The first lives inside the
# payload (a thread long-polling instance/preempted) and usually wins the race.
# They are redundant on purpose: 30 seconds is not enough time to discover that
# your only mechanism did not fire.
set -uo pipefail
LOG=/var/log/chlu-job.log
{
  echo "=== SHUTDOWN/PRE-EMPTION NOTICE $(date -u +%FT%TZ) ==="
  md() { curl -s -f -H 'Metadata-Flavor: Google' \
          "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null; }
  echo "preempted=$(md preempted)"
  timeout 25 docker stop -t 20 chlu-job 2>/dev/null || docker kill chlu-job 2>/dev/null
  JOB="$(md attributes/chlu-job)"; REMOTE="$(md attributes/chlu-remote)"
  if [ -n "${REMOTE:-}" ]; then
    echo preempted > /var/chlu/out/PREEMPTED
    gcloud storage rsync -r /var/chlu/out "$REMOTE/$JOB/out" -q
    gcloud storage cp "$LOG" "$REMOTE/$JOB/out/job.log" -q
  fi
  echo "=== shutdown handler done $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1
