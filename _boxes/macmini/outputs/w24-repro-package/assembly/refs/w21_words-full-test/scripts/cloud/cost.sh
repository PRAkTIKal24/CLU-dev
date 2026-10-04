#!/usr/bin/env bash
# scripts/cloud/cost.sh <job> <card> <spot:0|1> [t0_epoch] -- print the cost line.
#
# Billed hours are taken from the VM's own lifetime where GCE still knows it
# (`creationTimestamp` -> now) and otherwise from the wall clock the launcher
# observed, which is an over-estimate by the few seconds of API latency. The
# rate is a public LIST price, so this is an estimate: it is here to keep a
# running job's spend visible, not to replace the invoice.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
JOB="${1:?job}"; CARD="${2:-l4}"; SPOT="${3:-0}"; T0="${4:-}"
PY="${PYTHON:-python3}"
VM="chlu-${JOB}"
RATE="$(chlu_price "$CARD" "$SPOT")"

# Look the VM up by name across the project: launch.sh falls back to another
# zone on a stockout, so the configured zone is not necessarily where it ran.
CREATED="$(gcloud compute instances list --project="$CHLU_PROJECT" \
           --filter="name=${VM}" --format='value(creationTimestamp)' 2>/dev/null | head -1)"
NOW="$(date -u +%s)"
if [ -n "$CREATED" ]; then
  START="$("$PY" -c "import datetime,sys;print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))" "$CREATED")"
  SRC="live VM uptime"
elif [ -n "$T0" ]; then
  START="$T0"; SRC="launcher wall clock (VM already deleted)"
else
  echo "== cost: no timing source for job ${JOB}"; exit 0
fi
SECS=$(( NOW - START ))
"$PY" - "$JOB" "$CARD" "$SPOT" "$SECS" "$RATE" "$SRC" "$CHLU_PRICES_AS_OF" <<'PYEOF'
import sys
job, card, spot, secs, rate, src, asof = sys.argv[1:8]
h = int(secs) / 3600.0
usd = h * float(rate)
mode = "spot" if spot == "1" else "on-demand"
print("== COST")
print(f"   job {job}: {card} ({mode})  {h:.3f} h @ ${float(rate):.3f}/h  =  ${usd:.2f}")
print(f"   basis: {src}; list price as of {asof}")
PYEOF
