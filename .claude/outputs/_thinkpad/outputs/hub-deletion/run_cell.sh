#!/usr/bin/env bash
# Resumable driver: for each unit in the plan, build -> eval -> rival.
# "a kill costs ONE UNIT, never the run": every stage skips if its artefact exists.
#   usage: run_cell.sh <plan.jsonl> [extra args passed to eval/rival, e.g. --with-b]
# NOTE the environment is sourced ONCE here and inherited by every unit, so every
# build in the campaign sees the identical thread counts and the identical backend.
# That is deliberate: det_A is the pre-deletion control, and changing the numeric
# environment part-way through would make it incomparable to the deleted builds.
set -u
. "$(dirname "$0")/hubenv.sh"
cd "$CHLU_ROOT"
PLAN="$1"; shift || true
EXTRA="${*:-}"
mkdir -p "$HUB/logs" "$HUB/runs"
S="$(cygpath -m "$WT")/scripts/hub_deletion"
O="$(cygpath -m "$CHLU_HUB_OUTPUTS/hub-deletion")"
TGT="@$O/targets.json"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  TAG=$(printf '%s' "$line" | "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["tag"])')
  SPEC=$(printf '%s' "$line" | "$PY" -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read())["spec"]))')
  echo "=== $TAG  start $(date -u +%H:%M:%SZ) ==="
  if [ ! -f "$HUB/runs/$TAG/build.json" ]; then
    "$PY" "$S/build_unit.py" --tag "$TAG" --spec "$SPEC" --out "$O/runs" \
        --seed 0 --no-verify-cache > "$HUB/logs/$TAG.build.log" 2>&1 \
      || { echo "BUILD FAIL $TAG"; tail -5 "$HUB/logs/$TAG.build.log"; continue; }
  fi
  if [ ! -f "$HUB/runs/$TAG/eval.json" ]; then
    "$PY" "$S/eval_unit.py" --tag "$TAG" --out "$O/runs" --seed 0 --targets "$TGT" $EXTRA \
        > "$HUB/logs/$TAG.eval.log" 2>&1 \
      || { echo "EVAL FAIL $TAG"; tail -8 "$HUB/logs/$TAG.eval.log"; }
  fi
  if [ ! -f "$HUB/runs/$TAG/rival.json" ]; then
    "$PY" "$S/rival_unit.py" --tag "$TAG" --out "$O/runs" --seed 0 --targets "$TGT" $EXTRA \
        > "$HUB/logs/$TAG.rival.log" 2>&1 \
      || { echo "RIVAL FAIL $TAG"; tail -8 "$HUB/logs/$TAG.rival.log"; }
  fi
  echo "--- $TAG done $(date -u +%H:%M:%SZ)" \
       "build=$([ -f "$HUB/runs/$TAG/build.json" ] && echo ok || echo MISSING)" \
       "eval=$([ -f "$HUB/runs/$TAG/eval.json" ] && echo ok || echo MISSING)" \
       "rival=$([ -f "$HUB/runs/$TAG/rival.json" ] && echo ok || echo MISSING)"
done < "$PLAN"
echo "PLAN COMPLETE $(date -u +%H:%M:%SZ)"
