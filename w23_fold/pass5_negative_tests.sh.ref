#!/usr/bin/env bash
# PASS 5 negative tests: each mutation of a scratch copy of the register must make build_tables.py exit 1.
# usage: negative_tests.sh <register dir> <scratch dir>
set -u; R="$1"; S="$2"; EV=/Users/batmac/clu/evidence/outputs; H="$(cd "$(dirname "$0")" && pwd)"; fails=0
mkdir -p "$S"
run() { # name, then the mutate.py arguments (run inside the copy)
  local name="$1"; shift; local d="$S/neg-$name"
  rm -rf "$d" 2>/dev/null; mkdir -p "$d/out"; cp -R "$R" "$d/reg"; cp -R "$H" "$d/w22_seeds5"
  (cd "$d/reg" && python3 "$H/mutate.py" "$@") || { echo "MUTATION FAILED TO APPLY: $name"; fails=$((fails+1)); return; }
  (cd "$d/reg" && python3 build_tables.py "$EV" "$d/out" > "$d/build.txt" 2>&1); rc=$?
  if [ $rc -eq 1 ] && ! grep -q Traceback "$d/build.txt"; then echo "exit 1 (caught): $name :: $(grep -m1 -E 'FAIL|UNSOURCED|UNKNOWN|POINTER|BAD|UNRENDERED' "$d/build.txt" | cut -c1-160)"
  else echo "NOT CAUGHT (rc=$rc): $name"; fails=$((fails+1)); fi
}
run t7_cell            cells_pass4.py "'p4_c0_b': ('SD', 88, '2,081,024')" "'p4_c0_b': ('SD', 88, '2,081,025')"
run pass4_draft_cell   cells_pass4.py "'d4_index': ('D4V', 2859," "'d4_index': ('D4V', 2858,"
run w15_draft_cell     cells_pass5.py "'D5V', 564, '60.150'" "'D5V', 564, '60.151'"
run w15_findings_cell  cells_pass5.py "('CCF', 59, '60.150')" "('CCF', 58, '60.150')"
run w15_json_old_value cells_pass5.py "'/PART_B_rows_3_4/arms/PoE/budget_per_seed/0/x_store_reference', '27.135'" "'/PART_B_rows_3_4/arms/PoE/budget_per_seed/0/x_store_reference', '27.13'"
run w20_json_headline  cells_pass5.py "'/headline/data_matched_margin_rows34_K32/mean', '7.03'" "'/headline/data_matched_margin_rows34_K32/mean', '7.04'"
run tie4_equality      cells_pass5.py "'/ROWS/nway/control: random_atoms'" "'/ROWS/nway/control: landscape_shuffled_across_wells'"
run tie5_equality      cells_pass5.py "'/ROWS/nway/rival@10x-steps: transformer'" "'/ROWS/nway/rival@10x-steps: mlp'"
run orphan_value_token --append TABLES.template.md "orphan {{p5_w15d_1:v}}"
run json_value_only    --append TABLES.template.md "json cell without pointer {{p5_w20h_1:v}} {{@JW20:1}}"
run bare_number        --append TABLES.template.md "an unsourced 27.14 here"
run historical_tamper  --pass2-tamper draft_pass3/pass2-manifest.json
echo "negative tests not caught: $fails"
exit $fails
