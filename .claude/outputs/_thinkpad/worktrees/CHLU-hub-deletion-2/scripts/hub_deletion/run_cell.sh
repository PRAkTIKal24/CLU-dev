#!/usr/bin/env bash
# The deletion cell, as a LEDGER of resumable units.
#   "a kill costs ONE UNIT, never the run"   (scripts/csf3/job_cpu_gen.sh)
#
# Each unit is build -> eval -> rival, and each of those three skips itself if its
# own output file already exists.  So re-running this script after any kind of death
# resumes at the exact unit that died, and costs at most one rebuild (~9.5 min).
#
# Units run STRICTLY SEQUENTIALLY: this hub owns 4 of 16 cores and shares the box
# with three others.  Two builds at 4 threads each would be 8.  Do not parallelise.
set -u   # ⛔ NOT -e: one failed unit must not abandon the other fourteen.

HUBDIR="${CHLU_HUB_OUTPUTS}/hub-deletion"
RUNS="$(cygpath -w "${HUBDIR}/runs")"
LOGS="${HUBDIR}/logs"
LEDGER="${HUBDIR}/LEDGER.tsv"
TARGETS="@$(cygpath -w "${HUBDIR}/targets.json")"
PY="${CHLU_ROOT}/.venv/Scripts/python.exe"
cd "$(dirname "$0")/../.." || exit 1
mkdir -p "$LOGS"
[ -f "$LEDGER" ] || printf 'utc\ttag\tstage\tstatus\tseconds\tnote\n' > "$LEDGER"

note() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" "$5" >> "$LEDGER"; }

unit() {  # unit <tag> <spec-json>
  local tag="$1" spec="$2" t0 rc
  for stage in build eval rival; do
    case "$stage" in
      build) out="build.json";  cmd=( "$PY" scripts/hub_deletion/build_unit.py --tag "$tag" --spec "$spec" --out "$RUNS" --no-verify-cache ) ;;
      eval)  out="eval.json";   cmd=( "$PY" scripts/hub_deletion/eval_unit.py  --tag "$tag" --out "$RUNS" --targets "$TARGETS" ) ;;
      rival) out="rival.json";  cmd=( "$PY" scripts/hub_deletion/rival_unit.py --tag "$tag" --out "$RUNS" --targets "$TARGETS" ) ;;
    esac
    if [ -f "${HUBDIR}/runs/${tag}/${out}" ]; then note "$tag" "$stage" SKIP 0 "already present"; continue; fi
    t0=$(date +%s)
    "${cmd[@]}" >> "${LOGS}/${tag}.log" 2>&1; rc=$?
    if [ $rc -eq 0 ]; then note "$tag" "$stage" OK   "$(( $(date +%s) - t0 ))" ""
    else                 note "$tag" "$stage" FAIL "$(( $(date +%s) - t0 ))" "rc=$rc see logs/${tag}.log"; break; fi
  done
}

# ---- 1. THE BLOCKING GATE: a second independent full-pool build ---------------
# det_B is already built (it doubles as `base`, the pre-deletion store).
unit det_C '{"kind":"full"}'
unit det_B '{"kind":"full"}'          # build SKIPs; this fills in its eval + rival

# ---- 2. the six deletions, one median-index value per factor (PREREG §3) ------
unit del_f0v5 '{"kind":"delete","spec":[[0,5]]}'
unit del_f1v5 '{"kind":"delete","spec":[[1,5]]}'
unit del_f2v5 '{"kind":"delete","spec":[[2,5]]}'
unit del_f3v4 '{"kind":"delete","spec":[[3,4]]}'
unit del_f4v2 '{"kind":"delete","spec":[[4,2]]}'
unit del_f5v7 '{"kind":"delete","spec":[[5,7]]}'

# ---- 3. the PLACEBOS: the churn floor, size-matched, semantically aligned to
#         nothing.  ⭐ Without these, "collateral" cannot be told apart from "a
#         rebuild moves numbers".
unit plc_n3269 '{"kind":"placebo","n":3269,"seed":0}'
unit plc_n3224 '{"kind":"placebo","n":3224,"seed":1}'
unit plc_n3207 '{"kind":"placebo","n":3207,"seed":2}'
unit plc_n4095 '{"kind":"placebo","n":4095,"seed":3}'
unit plc_n8234 '{"kind":"placebo","n":8234,"seed":4}'
unit plc_n2142 '{"kind":"placebo","n":2142,"seed":5}'

# ---- 4. the CONJUNCTION (the gs charter's re-chartered target) and the
#         PATH-INDEPENDENCE pair (PREREG §4c).
unit del_conj_f2v5_f4v2 '{"kind":"delete","spec":[[2,5],[4,2]]}'
unit plc_n852           '{"kind":"placebo","n":852,"seed":6}'
# two different code paths, one live set: their digests must be equal.
unit del_seq_f0v5_f1v5   '{"kind":"sequence","specs":[[[0,5]],[[1,5]]]}'
unit del_union_f0v5_f1v5 '{"kind":"union","specs":[[[0,5]],[[1,5]]]}'

printf '\nLEDGER:\n' ; cat "$LEDGER"
