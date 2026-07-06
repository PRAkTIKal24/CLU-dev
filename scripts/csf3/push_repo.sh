#!/usr/bin/env bash
# scripts/csf3/push_repo.sh — sync the CLU repo laptop -> CSF3 scratch.
# RUN ON THE LAPTOP (on-campus network or GlobalProtect VPN required).
#
#   export CSF3_USER=<uom-it-username>        # e.g. mabcxyz1, NOT the email
#   scripts/csf3/push_repo.sh                 # code + uv.lock -> ~/scratch/CHLU
#   PUSH_DELETE=1 scripts/csf3/push_repo.sh   # also delete remote strays
#
# WHY RSYNC, NOT GIT: (a) uv.lock is gitignored, but it is REQUIRED on the
# cluster for env parity (uv sync --frozen) — rsync carries it; (b) CSF3
# login nodes cannot reach GitHub (off-campus), so a clone would itself need
# a batch job. rsync goes laptop->login node over ssh, which always works.
#
# Remote transfer facts per official CSF3 file-transfer page (mod. 2024-10-31):
# host csf3.itservices.manchester.ac.uk, port 22, ssh-based tools.

set -euo pipefail

CSF3_HOST="${CSF3_HOST:-csf3.itservices.manchester.ac.uk}"
CSF3_USER="${CSF3_USER:?set CSF3_USER=<uom-it-username> (form mabcxyz1)}"
REMOTE_DIR="${REMOTE_DIR:-scratch/CHLU}"   # relative to $HOME on CSF3; ~/scratch is a standard symlink
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

DELETE_FLAG=()
if [ "${PUSH_DELETE:-0}" = "1" ]; then
    DELETE_FLAG=(--delete)   # excluded paths below are protected from deletion
fi

# Guard: the cluster build runs `uv sync --frozen --extra cuda`, which needs
# the cuda extra IN the lock. A lock predating the extra must be refreshed
# (additive — existing pins are preserved) before pushing.
if [ -f "$LOCAL_REPO/uv.lock" ] && ! grep -q "jax-cuda12" "$LOCAL_REPO/uv.lock"; then
    echo "WARNING: uv.lock predates the cuda extra — run 'uv lock' in the repo"
    echo "         first, or the cluster's frozen sync will refuse the lock."
fi

echo "push: $LOCAL_REPO/ -> $CSF3_USER@$CSF3_HOST:$REMOTE_DIR/"
rsync -avz --progress "${DELETE_FLAG[@]}" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.claude/' \
    --exclude 'docs/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '*.egg-info/' \
    --exclude '.coverage' \
    --exclude 'coverage.xml' \
    --exclude '.DS_Store' \
    --exclude 'projects/*/' \
    --exclude '.sklearn_data/' \
    --exclude 'slurm-*.out' \
    "$LOCAL_REPO/" "$CSF3_USER@$CSF3_HOST:$REMOTE_DIR/"

echo "done. uv.lock included: $(ls "$LOCAL_REPO/uv.lock" >/dev/null 2>&1 && echo yes || echo 'NO — cluster env will resolve fresh!')"
