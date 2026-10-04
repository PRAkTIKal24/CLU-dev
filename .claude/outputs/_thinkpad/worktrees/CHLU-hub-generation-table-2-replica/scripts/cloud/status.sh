#!/usr/bin/env bash
# scripts/cloud/status.sh <job> -- where is it, what has it done, what has it cost.
# Reads the bucket, not the VM, so it keeps working after a pre-emption.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
JOB="${1:?usage: status.sh <job>}"
REMOTE="gs://${CHLU_BUCKET}/jobs/${JOB}"
echo "== ${JOB}"
# by name across zones: launch.sh may have fallen back on a stockout
VMLINE="$(gcloud compute instances list --project="$CHLU_PROJECT" --filter="name=chlu-${JOB}" \
  --format='value[separator="  "](name,zone.basename(),status,scheduling.provisioningModel,creationTimestamp)' 2>/dev/null)"
[ -n "$VMLINE" ] && echo "   vm: $VMLINE" \
  || echo "   vm: gone (finished, deleted, or pre-empted with termination-action=DELETE)"
echo "-- progress.json"
gcloud storage cat "$REMOTE/out/progress.json" 2>/dev/null || echo "   (none yet)"
echo "-- checkpoints"
gcloud storage ls "$REMOTE/ckpt/" 2>/dev/null || echo "   (none yet)"
