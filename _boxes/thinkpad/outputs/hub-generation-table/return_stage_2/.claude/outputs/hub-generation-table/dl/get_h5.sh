#!/usr/bin/env bash
# Resumable unit: fetch 3dshapes.h5, verify sha256, publish atomically.
# ⛔ Never overwrites an existing published file (three hubs share the data root).
set -u
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-generation-table/hubenv.sh
SHA=0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785
DEST="$CHLU_GEN_DATA_ROOT/shapes3d/3dshapes.h5"
TMP="$CHLU_OUT/dl/3dshapes.h5.part"
LEDGER="$CHLU_OUT/dl/ledger.txt"
echo "$(date -u +%FT%TZ) start" >> "$LEDGER"
if [ -f "$DEST" ]; then
  echo "$(date -u +%FT%TZ) DEST already present, verifying" >> "$LEDGER"
else
  curl -L --retry 5 --retry-delay 5 -C - -o "$TMP" https://storage.googleapis.com/3d-shapes/3dshapes.h5 \
      >> "$LEDGER" 2>&1
  GOT=$(sha256sum "$TMP" | cut -d' ' -f1)
  echo "$(date -u +%FT%TZ) downloaded sha=$GOT" >> "$LEDGER"
  if [ "$GOT" != "$SHA" ]; then
    echo "$(date -u +%FT%TZ) ⛔ SHA MISMATCH expected $SHA" >> "$LEDGER"; exit 1
  fi
  if [ -f "$DEST" ]; then
    echo "$(date -u +%FT%TZ) a peer published first; leaving theirs in place" >> "$LEDGER"
  else
    mv "$TMP" "$DEST"
    echo "$(date -u +%FT%TZ) published -> $DEST" >> "$LEDGER"
  fi
fi
GOT2=$(sha256sum "$DEST" | cut -d' ' -f1)
echo "$(date -u +%FT%TZ) DEST sha=$GOT2 match=$([ "$GOT2" = "$SHA" ] && echo YES || echo NO)" >> "$LEDGER"
echo "$(date -u +%FT%TZ) done" >> "$LEDGER"
