#!/usr/bin/env bash
# scripts/cloud/fetch_results.sh <job> [dest] -- pull a job's outputs to the laptop.
#
# Default destination is .claude/outputs/cloud/<job>/ -- under .claude/ because
# protocol §2 says every agent artefact lives there, and generic because a spoke
# writing into another spoke's slug directory is how reports get mixed up. Pass a
# dest (or set CHLU_FETCH_DEST) to put it in your own slug's directory.
#
# The job already pushes to GCS on a timer; this is only the last hop. A result
# that is not synced back does not exist.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
REPO="$(cd "$HERE/../.." && pwd)"
JOB="${1:?usage: fetch_results.sh <job> [dest]}"
DEST="${2:-${CHLU_FETCH_DEST:-$REPO/.claude/outputs/cloud}/$JOB}"
mkdir -p "$DEST"
gcloud storage rsync -r "gs://${CHLU_BUCKET}/jobs/${JOB}/out" "$DEST" --project="$CHLU_PROJECT"
echo "== ${JOB} -> ${DEST}"
ls -la "$DEST"
