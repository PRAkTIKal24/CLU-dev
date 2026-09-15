#!/usr/bin/env bash
# scripts/cloud/fetch_results.sh <job> [dest] -- pull a job's outputs to the laptop.
#
# Default destination is .claude/outputs/<slug>/<job>/ because a result that is
# not synced back does not exist. The job already pushes to GCS on a timer; this
# is the last hop.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
REPO="$(cd "$HERE/../.." && pwd)"
JOB="${1:?usage: fetch_results.sh <job> [dest]}"
DEST="${2:-$REPO/.claude/outputs/gen-cloud-enablement/$JOB}"
mkdir -p "$DEST"
gcloud storage rsync -r "gs://${CHLU_BUCKET}/jobs/${JOB}/out" "$DEST" --project="$CHLU_PROJECT"
echo "== ${JOB} -> ${DEST}"
ls -la "$DEST"
