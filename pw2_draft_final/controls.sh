#!/usr/bin/env bash
# usage: controls.sh G   -- positive controls for every closing-audit scan, on planted copies under G/controls/ (never the live drafts)
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$H"); G="$1"; C="$G/controls"; mkdir -p "$C"
V="$G/v2/DRAFT-v2.tex"; S="$G/style/DRAFT-v2-style.tex"; P="$C/plant"; mkdir -p "$P"
res() { if [ "$2" = yes ]; then echo "CONTROL $1: CAUGHT"; else echo "CONTROL $1: **MISSED**"; fi; }
# 1 compile metrics + rendered ?? : an undefined \ref and an undefined \citep planted in a scratch copy of the v2 folder
rsync -a --exclude '*.pdf' --exclude '*.aux' --exclude '*.log' --exclude '*.bbl' --exclude '*.blg' --exclude '*.out' --exclude 'compile-*' "$G/v2/" "$C/build-v2/"
rsync -a "$G/v2/figures/" "$C/build-v2/figures/"
python3 - "$C/build-v2/DRAFT-v2.tex" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read(); a='We name three properties and use them throughout.'
assert s.count(a)==1; s=s.replace(a, a+' PLANTED Sec.~\\ref{sec:planted-nowhere} and~\\citep{planted_nokey_2099}.'); open(p,'w').write(s)
PY
M=$(bash "$R/pw2_draft_sections/compile_folder.sh" "$C/build-v2" DRAFT-v2); echo "  $M"
uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$C/build-v2/DRAFT-v2.pdf" "$C/build-v2/DRAFT-v2.pdf.txt"
q=$(python3 "$H/scans.py" defects "$C/build-v2/DRAFT-v2.tex" "$C/build-v2/DRAFT-v2.pdf.txt" | head -1); echo "  $q"
echo "$M" | grep -q "ucit=[1-9]" && echo "$M" | grep -q "uref=[1-9]" && echo "$q" | grep -q "(undefined reference): [1-9]" && echo "$q" | grep -q "(undefined citation): [1-9]" && r=yes || r=no; res "compile: planted undefined ref + cite -> ucit/uref > 0 and ?? rendered" $r
# 2 style diff-check: one number changed in the style copy
python3 - "$S" "$P/style-num.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='margin of 1.0 tuple'; assert s.count(a)==1; open(sys.argv[2],'w').write(s.replace(a,'margin of 1.5 tuple'))
PY
(cd "$R/pw1_draft_sweeps" && python3 style_diff_check.py "$V" "$P/style-num.tex" > "$C/style-num.md"); [ $? -ne 0 ] && r=yes || r=no; res "style diff-check: one number changed in the style twin -> exit 1" $r
# 3 provenance compare: a planted unsourced number line
python3 - "$V" "$P/v2-prov.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='We name three properties and use them throughout.'; assert s.count(a)==1; open(sys.argv[2],'w').write(s.replace(a, a+' It scored 123.456 once.'))
PY
(cd "$R/pw1_draft_sweeps" && python3 check_provenance.py "$P/v2-prov.tex" > "$C/prov-plant.txt"); python3 "$H/prov_compare.py" "$G/checks/prov-pw1-DRAFT-v2.txt" "$C/prov-plant.txt" > "$C/prov-plant-compare.md"; [ $? -ne 0 ] && r=yes || r=no; res "provenance: planted unsourced 123.456 -> REGRESSED, exit 1" $r
# 4 numbers: a planted number with no rule -> UNCLASSIFIED
python3 - "$V" "$P/v2-num.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='We name three properties and use them throughout.'; assert s.count(a)==1; open(sys.argv[2],'w').write(s.replace(a, a+' The margin was 7.486 here.'))
PY
python3 "$H/numclass.py" "$P/v2-num.tex" > "$C/numclass-plant.md"; e=$?; grep -q "7.486\` | UNCLASSIFIED" "$C/numclass-plant.md" && [ $e -ne 0 ] && r=yes || r=no; res "numbers: planted 7.486 in editable prose -> UNCLASSIFIED, exit 1" $r
# 5 rounded: a planted wrong rounding rule
python3 "$H/rounded_check.py" "$V" --plant > "$C/rounded-plant.md"; [ $? -ne 0 ] && grep -q "PLANTED.*FAIL" "$C/rounded-plant.md" && r=yes || r=no; res "rounded: 7.034 claimed to round to 8 -> FAIL, exit 1" $r
# 6 Head-held: one added space in the Results first paragraph
python3 - "$V" "$P/v2-hh.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='is the scored result.'; assert s.count(a)==1; open(sys.argv[2],'w').write(s.replace(a,'is the scored  result.'))
PY
mkdir -p "$P/hh"; mv "$P/v2-hh.tex" "$P/hh/DRAFT-v2.tex"
(cd "$R/pw2_draft_sections" && python3 headheld.py "$P/hh/DRAFT-v2.tex" /Users/batmac/clu/outputs/pw2-draft-leftovers/HEAD-HELD.md > "$C/hh-plant.txt"); e1=$?
cp "$C/hh-plant.txt" "$C/hh-plant-now.txt"
python3 "$H/scans.py" headheld "$P/hh/DRAFT-v2.tex" /Users/batmac/clu/outputs/pw2-draft-leftovers/snap-start/v2/DRAFT-v2.tex /Users/batmac/clu/outputs/pw2-draft-leftovers/HEAD-HELD.md "$C/hh-plant-now.txt" > "$C/hh-plant-raw.md"; e2=$?
[ $e1 -eq 1 ] && [ $e2 -eq 1 ] && grep -q "HEAD-HELD CHECK: FAIL" "$C/hh-plant.txt" && grep -q "RAW CHECK: FAIL" "$C/hh-plant-raw.md" && r=yes || r=no; res "Head-held: one added space in the Results first paragraph -> headheld.py FAIL and raw check FAIL" $r
# 7 TODO / PLACEHOLDER: one planted \PLACEHOLDER
python3 - "$V" "$P/v2-ph.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='We name three properties and use them throughout.'; assert s.count(a)==1; open(sys.argv[2],'w').write(s.replace(a, a+' \\PLACEHOLDER{planted row}'))
PY
python3 "$H/scans.py" todo "$P/v2-ph.tex" > "$C/todo-plant.md"; grep -q "PLACEHOLDER uses: 1" "$C/todo-plant.md" && grep -q "planted row" "$C/todo-plant.md" && r=yes || r=no; res "TODO/PLACEHOLDER: planted \\PLACEHOLDER -> listed, count 1" $r
# 8 citation scope: jawahar2026chlu cited for the retention law
python3 - "$V" "$P/v2-cite.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='We name three properties and use them throughout.'; assert s.count(a)==1
open(sys.argv[2],'w').write(s.replace(a, a+' The retention law is due to~\\citep{jawahar2026chlu}.'))
PY
python3 "$H/scans.py" cite "$P/v2-cite.tex" > "$C/cite-plant.md"; [ $? -ne 0 ] && grep -q "VIOLATION: disallowed" "$C/cite-plant.md" && r=yes || r=no; res "citation scope: jawahar2026chlu cited for the retention law -> VIOLATION, exit 1" $r
# 9 tells: pw2_draft_tells/controls.py (planted tells raise every detector)
python3 "$R/pw2_draft_tells/controls.py" "$V" "$C/tells" > "$C/tells-controls.txt" 2>&1; grep -q "ALL CONTROLS PASS" "$C/tells-controls.txt" && r=yes || r=no; res "tells: planted tells raise all seven detectors; the brief's seven examples match" $r
# 10 defects: a planted absolute path in a comment and a planted doubled word
python3 - "$V" "$P/v2-def.tex" <<'PY'
import sys; s=open(sys.argv[1]).read(); a='We name three properties and use them throughout.'; assert s.count(a)==1
open(sys.argv[2],'w').write(s.replace(a, a+' the the planted % /Users/someone/x'))
PY
python3 "$H/scans.py" defects "$P/v2-def.tex" "$G/checks/DRAFT-v2.pdf.txt" > "$C/defects-plant.md"; grep -q "code and comments): 1" "$C/defects-plant.md" && grep -q "tables): 1" "$C/defects-plant.md" && r=yes || r=no; res "defects: planted absolute path + doubled word in the source -> both listed" $r
# 11 bib hash: a one-byte change to a copy changes the sha256
cp "$G/v2/refs_clu_iclr.bib" "$P/bib.bib"; printf ' ' >> "$P/bib.bib"; [ "$(shasum -a 256 "$P/bib.bib" | cut -c1-64)" != b2cf715f25e020acf02038d5df4a2fd0ce816fb59e0aa3cfb101335684a71bd3 ] && r=yes || r=no; res "bib: one appended byte -> sha256 differs from b2cf715f..." $r
