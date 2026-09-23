#!/usr/bin/env bash
# w15 spoke/bar-d-score -- the GPU leg's rows-3/4 scoring leg, under THIS spoke's own
# authorisation.  It adds exactly one thing to scripts/w8_encoder_loop_score.sh: the
# `test` guard is checked against w15's OWN PREREG and w15's OWN ledger row, because a
# scored run is never authorised by another hub's row (charter A.2).  The w8 script's
# own guard still runs after this one; it is an extra check, never the authorising one.
#
#   scripts/w15_bar_d_score.sh <condition> <seed> <arm> <run-base> <out-base> [score-arm ...]
set -euo pipefail
COND="${1:?condition: val|test}"
export PREREG="${PREREG:-/c/chlu/outputs/w15-bar-d-score/PREREG-w15-bar-d.md}"
if [ "$COND" = "test" ]; then
  [ -f "$PREREG" ] || { echo "[w15] REFUSING: no PREREG at $PREREG" >&2; exit 2; }
  grep -Eq "w15[ /]spoke/bar-d-score|w15/bar-d-score" /c/chlu/SCORED-LEDGER.md || {
    echo "[w15] REFUSING: no /c/chlu/SCORED-LEDGER.md row for w15 spoke/bar-d-score." >&2
    exit 2; }
  echo "[w15] scored run authorised by $PREREG + w15's OWN ledger row"
fi
exec bash "$(dirname "$0")/w8_encoder_loop_score.sh" "$@"
