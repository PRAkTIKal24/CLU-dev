#!/usr/bin/env bash
# scripts/cloud/run_until_done.sh -- drive one CHLU job to completion across
# pre-emptions, with NO HUMAN IN THE LOOP.
#
# ⭐ WHY THIS EXISTS, measured. `gen-cloud-enablement` proved resume works: a
# destroyed VM loses a checkpoint interval, not a run. What it did not ship is
# anything that RE-INVOKES the job. Its `det-ctl` needed THREE launches to
# finish 600 steps and a human drove all three, and GCE pre-empted two of this
# project's spot L4s unprompted inside one short acceptance window. ⛔ A
# fallback that needs a human to notice and relaunch is not a fallback.
#
#   scripts/cloud/run_until_done.sh --job pilot1 --card l4 --spot \
#       --exp cluformer --units '0 1' --exp-args '--stage s1 --scale toy' \
#       --max-attempts 6 --max-usd 3.00
#
# ⛔ NEVER AN UNBOUNDED LOOP. Two independent hard caps, both refusing by name:
#   * --max-attempts   how many launches, total. A stockout counts.
#   * --max-usd        estimated total spend. Checked against spend SO FAR
#                      PLUS the ceiling of the attempt about to be launched, so
#                      the cap is never breached by the attempt that tests it.
# A relaunch loop that retries forever spends real money for nothing and is
# worse than failing: the failure at least gets looked at.
#
# The completion sentinel is `out/result.json`, written by startup.sh only when
# every unit is banked in the journal. Absence of it is the relaunch condition;
# nothing else is.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
PY="${PYTHON:-python3}"

JOB="" CARD=l4 SPOT=0
MAX_ATTEMPTS=6 MAX_USD=5.00 MAX_HOURS=2 BACKOFF=60 BACKOFF_MAX=900
POLL_VM=20 VM_GONE_WAIT=300
LAUNCH_ARGS=()

die() { echo "FATAL: $*" >&2; exit 1; }
say() { echo "[supervisor $(date -u +%FT%TZ)] $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB="$2"; LAUNCH_ARGS+=(--job "$2"); shift 2 ;;
    --card) CARD="$2"; LAUNCH_ARGS+=(--card "$2"); shift 2 ;;
    --spot) SPOT=1; LAUNCH_ARGS+=(--spot); shift ;;
    --on-demand) SPOT=0; LAUNCH_ARGS+=(--on-demand); shift ;;
    --max-hours) MAX_HOURS="$2"; LAUNCH_ARGS+=(--max-hours "$2"); shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --max-usd) MAX_USD="$2"; shift 2 ;;
    --backoff-s) BACKOFF="$2"; shift 2 ;;
    --backoff-max-s) BACKOFF_MAX="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    # everything else is launch.sh's business (--exp, --units, --exp-args,
    # --artefacts, --zone, --env, --deterministic, ...). ⛔ Not re-declared
    # here: two copies of one flag list is how the two drift apart.
    *) LAUNCH_ARGS+=("$1"); shift ;;
  esac
done
[ -n "$JOB" ] || die "--job is required"
case "$MAX_ATTEMPTS" in ''|*[!0-9]*) die "--max-attempts '$MAX_ATTEMPTS' is not a positive integer" ;; esac
[ "$MAX_ATTEMPTS" -ge 1 ] || die "--max-attempts must be >= 1"
"$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$MAX_USD" \
  || die "--max-usd '$MAX_USD' must be a positive number"

REMOTE="gs://${CHLU_BUCKET}/jobs/${JOB}"
RATE="$(chlu_price "$CARD" "$SPOT")"
[ "$RATE" != "0" ] || die "no price for card '$CARD' in $CHLU_CARDS_JSON -- refusing to run
       an unbounded-cost loop against a card whose rate is unknown."
STATE_DIR="${CHLU_SUPERVISOR_DIR:-$HERE/../../.claude/outputs/gen-cloud-runner/supervisor}"
mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/${JOB}.json"

# --- seams, so the caps are testable without spending anything ----------------
# The acceptance for a cap is that it REFUSES. Proving that against real GCP
# would mean deliberately burning the budget it exists to protect, so both the
# launcher and the completion probe are overridable and the tests drive them.
LAUNCH_CMD="${CHLU_LAUNCH_CMD:-$HERE/launch.sh}"
is_done() {
  if [ -n "${CHLU_DONE_PROBE:-}" ]; then "$CHLU_DONE_PROBE" "$JOB"; return $?; fi
  gcloud storage cat "$REMOTE/out/result.json" --project="$CHLU_PROJECT" >/dev/null 2>&1
}
vm_state() {
  if [ -n "${CHLU_VM_PROBE:-}" ]; then "$CHLU_VM_PROBE" "$JOB"; return 0; fi
  gcloud compute instances list --project="$CHLU_PROJECT" --filter="name=chlu-${JOB}" \
      --format='value(status)' 2>/dev/null | head -1
}
nap() { [ -n "${CHLU_NO_SLEEP:-}" ] || sleep "$1"; }

# --- state --------------------------------------------------------------------
ATTEMPTS=0 SPENT=0 STOCKOUTS=0 PREEMPTIONS=0 CUR_BACKOFF="$BACKOFF"
write_state() {   # $1 = status, $2 = note
  "$PY" - "$STATE" "$JOB" "$CARD" "$SPOT" "$ATTEMPTS" "$MAX_ATTEMPTS" "$SPENT" "$MAX_USD" \
         "$STOCKOUTS" "$PREEMPTIONS" "$RATE" "$1" "${2:-}" <<'PYST'
import json, os, sys, time
(p, job, card, spot, att, maxatt, spent, maxusd, stock, preempt, rate, status, note) = sys.argv[1:14]
rec = {"job": job, "card": card, "spot": spot == "1", "attempts_used": int(att),
       "max_attempts": int(maxatt), "usd_estimated": round(float(spent), 4),
       "max_usd": float(maxusd), "stockouts": int(stock), "preemptions_or_failures": int(preempt),
       "usd_per_hour": float(rate), "status": status, "note": note,
       "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tmp = p + ".tmp"
with open(tmp, "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True)
os.replace(tmp, p)
print("[supervisor] " + json.dumps(rec, sort_keys=True))
PYST
}

# --- the loop -----------------------------------------------------------------
say "job=$JOB card=$CARD $([ "$SPOT" = 1 ] && echo spot || echo on-demand) rate=\$$RATE/h"
say "caps: <= $MAX_ATTEMPTS attempts AND <= \$$MAX_USD estimated. Sentinel: $REMOTE/out/result.json"

if is_done; then
  say "result.json already present -- nothing to do."
  write_state done "already complete before the first attempt"; exit 0
fi

FINAL=incomplete
while :; do
  # 1. attempt cap -- checked BEFORE launching, so the cap is never exceeded.
  if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
    say "⛔ REFUSING to relaunch: attempt cap reached ($ATTEMPTS of $MAX_ATTEMPTS used)."
    say "   The job is NOT complete. Raise --max-attempts deliberately, or look at why"
    say "   $MAX_ATTEMPTS attempts did not finish it. ⛔ An unbounded retry loop spends"
    say "   real money for nothing and is worse than stopping here."
    FINAL=attempt_cap; break
  fi

  # 2. spend cap -- spend SO FAR plus the CEILING of the attempt about to start.
  #    Checking only the spend so far would let the cap be breached by exactly
  #    the attempt that tests it.
  PROJECTED="$("$PY" -c "print(round(float('$SPENT') + float('$RATE')*float('$MAX_HOURS'), 4))")"
  if "$PY" -c "import sys; sys.exit(0 if float('$PROJECTED') > float('$MAX_USD') else 1)"; then
    say "⛔ REFUSING to relaunch: spend cap would be exceeded."
    say "   spent so far \$$SPENT + next attempt ceiling (${MAX_HOURS}h x \$$RATE/h) = \$$PROJECTED > cap \$$MAX_USD"
    say "   The job is NOT complete. Raise --max-usd deliberately."
    FINAL=spend_cap; break
  fi

  # 3. do not race a live VM of our own. A previous attempt that is still
  #    RUNNING is not a reason to launch a second one; `instances create` would
  #    fail on the name anyway, and two VMs sharing one journal prefix is worse.
  waited=0
  while :; do
    st="$(vm_state)"
    [ -z "$st" ] && break
    [ "$waited" -ge "$VM_GONE_WAIT" ] && break
    say "chlu-${JOB} still exists (status ${st}); waiting for it to go away (${waited}s)"
    nap "$POLL_VM"; waited=$((waited + POLL_VM))
  done

  ATTEMPTS=$((ATTEMPTS + 1))
  say "=== attempt $ATTEMPTS/$MAX_ATTEMPTS (spent so far \$$SPENT of \$$MAX_USD)"
  write_state running "attempt $ATTEMPTS starting"
  T0="$(date -u +%s)"
  OUTF="$STATE_DIR/${JOB}.attempt${ATTEMPTS}.log"
  "$LAUNCH_CMD" "${LAUNCH_ARGS[@]}" --wait 2>&1 | tee "$OUTF"
  LRC="${PIPESTATUS[0]}"
  T1="$(date -u +%s)"
  ELAPSED=$((T1 - T0))

  # 4. ⛔ A QUOTA REFUSAL IS NOT TRANSIENT. Stock comes back; a quota metric of
  #    zero does not, and relaunching against it burns the attempt budget to
  #    reprint the same refusal. Stop, and hand the Head the actionable line the
  #    preflight already wrote.
  if grep -qE "REFUSING TO LAUNCH|quota preflight refused this launch" "$OUTF"; then
    say "⛔ STOPPING: the quota preflight refused this card/region, and that does not"
    say "   change by retrying. See $OUTF for the metric and the limit."
    FINAL=quota_refused; break
  fi

  # 5. price this attempt. ⭐ The basis is WHETHER A VM WAS EVER CREATED, not the
  #    launcher's wall clock: a stockout, a tag-pinned image and every other
  #    refusal all create nothing, and charging them would make the spend cap
  #    refuse far too early -- for spend that never happened.
  CREATED=0; grep -q "^== created chlu-" "$OUTF" && CREATED=1
  if [ "$CREATED" = "0" ]; then
    COST=0
    if grep -qiE "no zone in .* had a free|stock/availability, not quota" "$OUTF"; then
      STOCKOUTS=$((STOCKOUTS + 1))
      say "attempt $ATTEMPTS: STOCKOUT after ${ELAPSED}s -- no VM created, \$0.00"
    else
      say "attempt $ATTEMPTS: the launcher created NO VM after ${ELAPSED}s -- \$0.00. See $OUTF."
    fi
    if is_done; then FINAL=done; break; fi
    say "backing off ${CUR_BACKOFF}s before the next attempt (nothing was created, so nothing was spent)"
    write_state backoff "no VM created; sleeping ${CUR_BACKOFF}s"
    nap "$CUR_BACKOFF"
    CUR_BACKOFF=$(( CUR_BACKOFF * 2 )); [ "$CUR_BACKOFF" -gt "$BACKOFF_MAX" ] && CUR_BACKOFF="$BACKOFF_MAX"
    continue
  fi

  COST="$("$PY" -c "print(round(${ELAPSED}/3600.0*float('$RATE'), 4))")"
  SPENT="$("$PY" -c "print(round(float('$SPENT') + float('$COST'), 4))")"
  CUR_BACKOFF="$BACKOFF"   # a real VM ran: the region is not stocked out

  if is_done; then
    say "✅ attempt $ATTEMPTS: result.json present. ${ELAPSED}s, ~\$$COST."
    FINAL=done; break
  fi

  PREEMPTIONS=$((PREEMPTIONS + 1))
  say "attempt $ATTEMPTS ended WITHOUT result.json (launcher rc=$LRC, ${ELAPSED}s, ~\$$COST)."
  say "   Pre-emption or a failed unit. Completed units are banked in the journal;"
  say "   the next attempt resumes and re-runs only the unit that was in flight."
  write_state relaunching "attempt $ATTEMPTS incomplete; relaunching"
done

write_state "$FINAL" "attempts=$ATTEMPTS stockouts=$STOCKOUTS incomplete_attempts=$PREEMPTIONS"
say "=== SUPERVISOR FINISHED: $FINAL"
say "    attempts used: $ATTEMPTS of $MAX_ATTEMPTS   (stockouts $STOCKOUTS, incomplete $PREEMPTIONS)"
say "    estimated spend: \$$SPENT of \$$MAX_USD   ($CHLU_PRICES_AS_OF)"
say "    state: $STATE"
[ "$FINAL" = "done" ] && exit 0
[ "$FINAL" = "quota_refused" ] && exit 3
[ "$FINAL" = "attempt_cap" ] && exit 4
[ "$FINAL" = "spend_cap" ] && exit 5
exit 1
