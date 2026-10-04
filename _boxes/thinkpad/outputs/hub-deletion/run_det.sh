#!/usr/bin/env bash
# STEP 1 driver: two independent full-scale builds from the SAME live set.
# Resumable: build_unit.py skips a tag whose build.json exists.
set -u
. "$(dirname "$0")/hubenv.sh"
cd "$CHLU_ROOT"
mkdir -p "$HUB/logs" "$HUB/runs"
for TAG in det_A det_B; do
  if [ -f "$HUB/runs/$TAG/build.json" ]; then echo "SKIP $TAG"; continue; fi
  if [ "$TAG" = det_A ]; then VERIFY_FLAG=""; else VERIFY_FLAG="--no-verify-cache"; fi
  echo "=== $TAG start $(date -u +%H:%M:%SZ) ==="
  "$PY" "$WWT\scripts\hub_deletion\build_unit.py" \
      --tag "$TAG" --spec '{"kind":"full"}' --out "$WHUB\runs" \
      --seed 0 $VERIFY_FLAG \
      > "$HUB/logs/$TAG.log" 2>&1
  echo "=== $TAG rc=$? end $(date -u +%H:%M:%SZ) ==="
done
echo "ALL DONE $(date -u +%H:%M:%SZ)"
