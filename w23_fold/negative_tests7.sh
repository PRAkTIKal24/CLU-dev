#!/usr/bin/env bash
# w23 fold negative tests for the register's pass-7 cells. STRICTER than pass 5's harness, after pass 5's own defect
# (a test that "passed" by crashing on a missing out dir): a test counts as CAUGHT only if
#   (0) the UNMUTATED copy builds rc=0 in this same harness (else every catch is void),
#   (1) the mutated build exits exactly 1, (2) with no Traceback, and (3) the failure output NAMES the expected target.
# The harness also runs a self-test: a crash (missing out dir) MUST be reported NOT CAUGHT.
# usage: negative_tests7.sh <staged register dir> <scratch dir>
set -u; R="$1"; S="$2"; EV=/Users/batmac/clu/evidence/outputs; H="$(cd "$(dirname "$0")" && pwd)"; fails=0
mkdir -p "$S"
judge() { # name rc buildlog expect
  if [ "$2" -eq 1 ] && ! grep -q Traceback "$3" && grep -qF -- "$4" "$3"; then echo "CAUGHT: $1 :: $(grep -m1 -F -- "$4" "$3" | cut -c1-150)"; return 0
  else echo "NOT CAUGHT (rc=$2, traceback=$(grep -c Traceback "$3"), names-target=$(grep -cF -- "$4" "$3")): $1"; return 1; fi
}
# (0) baseline
d="$S/base"; mkdir -p "$d/out"; cp -R "$R" "$d/reg"
(cd "$d/reg" && python3 build_tables.py "$EV" "$d/out" > "$d/build.txt" 2>&1); rc=$?
if [ $rc -ne 0 ]; then echo "BASELINE FAILS (rc=$rc): every test below is VOID"; exit 99; fi
echo "baseline: unmutated copy builds rc=0 :: $(tail -1 "$d/build.txt" | cut -c1-110)"
run() { # name expect mutate-args...
  local name="$1" expect="$2"; shift 2; local d="$S/neg-$name"
  mkdir -p "$d/out"; cp -R "$R" "$d/reg"
  (cd "$d/reg" && python3 "$H/mutate.py" "$@") || { echo "MUTATION FAILED TO APPLY: $name"; fails=$((fails+1)); return; }
  (cd "$d/reg" && python3 build_tables.py "$EV" "$d/out" > "$d/build.txt" 2>&1); local rc=$?
  judge "$name" $rc "$d/build.txt" "$expect" || fails=$((fails+1))
}
run p7_draft_cell_v2     "CELL FAIL p7_v_500:"  cells_pass7.py "'p7_v_500': ('D5V', 718, '52.607')" "'p7_v_500': ('D5V', 718, '52.608')"
run p7_draft_cell_style  "CELL FAIL p7_s_7:"    cells_pass7.py "'p7_s_7': ('D5S', 116, '+6.54')" "'p7_s_7': ('D5S', 117, '+6.54')"
run p7_source_cell       "CELL FAIL p7_f_5:"    cells_pass7.py "'p7_f_5': ('S7N1', 16, '+5.02')" "'p7_f_5': ('S7N1', 16, '+5.03')"
run p7_json_text_cell    "CELL FAIL p7_x_3:"    cells_pass7.py "0.6407297594090047'" "0.6407297594090048'"
run p7_draft_edit        "CELL FAIL p7_v_500:"  draft_pass5/DRAFT-v2.pass5-6cecd36a.tex "52.607" "52.617"
run p7_unknown_token     "UNKNOWN TOKEN p7_v_999999" --append TABLES.template.md "| x | {{p7_v_999999}} | y |"
run p7_bare_number       "UNSOURCED NUMBER"     --append TABLES.template.md "| an unsourced 27.14 in section 17 |"
# harness self-test: a crash must NOT count as a catch
d="$S/selftest"; mkdir -p "$d"; cp -R "$R" "$d/reg"
(cd "$d/reg" && python3 build_tables.py "$EV" "$d/NO_SUCH_DIR" > "$d/build.txt" 2>&1); rc=$?
if judge selftest_crash_on_missing_dir $rc "$d/build.txt" "CELL FAIL" >/dev/null; then echo "HARNESS DEFECT: a crash was counted as a catch"; fails=$((fails+1))
else echo "harness self-test OK: an unmutated build that crashes (rc=$rc, Traceback) is reported NOT CAUGHT"; fi
echo "negative tests not caught: $fails"
exit $fails
