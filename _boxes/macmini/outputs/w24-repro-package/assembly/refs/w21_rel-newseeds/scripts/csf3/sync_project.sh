#!/usr/bin/env bash
# scripts/csf3/sync_project.sh — move projects/<name>/ artifacts (plots,
# results/*.npz metrics, models/*.pkl checkpoints, config) between the laptop
# and CSF3. RUN ON THE LAPTOP (campus network or GlobalProtect VPN).
#
#   export CSF3_USER=<uom-it-username>
#   scripts/csf3/sync_project.sh pull csf3_b_s0        # cluster -> laptop
#   scripts/csf3/sync_project.sh push finalA           # laptop  -> cluster
#   scripts/csf3/sync_project.sh pull 'csf3_b_s*'      # glob: pull a whole sweep
#
# projects/*/ is gitignored by design — artifact movement is rsync-only.
# PULL PROMPTLY: CSF3 scratch has NO backup and a 3-month auto-cleanup
# (official filesystems docs, mod. 2026-03-17).

set -euo pipefail

usage() { echo "usage: [CSF3_USER=...] $0 pull|push <project-name-or-glob>"; exit 1; }
[ $# -ge 2 ] || usage

DIRECTION="$1"; NAME="$2"
CSF3_HOST="${CSF3_HOST:-csf3.itservices.manchester.ac.uk}"
CSF3_USER="${CSF3_USER:?set CSF3_USER=<uom-it-username> (form mabcxyz1)}"
REMOTE_DIR="${REMOTE_DIR:-scratch/CHLU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

RSYNC_OPTS=(-avz --progress)
# No --delete: syncs are additive for safety; clean up manually if needed.

case "$DIRECTION" in
    pull)
        mkdir -p "$LOCAL_REPO/projects"
        # trailing-slash-free source dir => rsync recreates projects/<name>/ locally
        rsync "${RSYNC_OPTS[@]}" \
            "$CSF3_USER@$CSF3_HOST:$REMOTE_DIR/projects/$NAME" \
            "$LOCAL_REPO/projects/"
        echo "pulled -> $LOCAL_REPO/projects/$NAME"
        ;;
    push)
        [ -d "$LOCAL_REPO/projects/$NAME" ] || { echo "no local projects/$NAME"; exit 1; }
        rsync "${RSYNC_OPTS[@]}" \
            "$LOCAL_REPO/projects/$NAME" \
            "$CSF3_USER@$CSF3_HOST:$REMOTE_DIR/projects/"
        echo "pushed -> $REMOTE_DIR/projects/$NAME"
        ;;
    *) usage ;;
esac
