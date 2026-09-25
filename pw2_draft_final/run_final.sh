#!/usr/bin/env bash
# usage: run_final.sh OUTDIR   -- the whole PW-2 closing audit on a clean copy of the live folders (read-only on the live drafts)
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$H"); O="$1"; LIVE=~/clu/pw2/live
G="$O/final-$(date -u +%Y%m%dT%H%M%SZ)"; echo "$G" > "$O/CURRENT_STAGE"; mkdir -p "$G/snap" "$G/checks" "$G/numbers" "$G/tells"
for f in v2 style; do
  rsync -a --exclude '*.aux' --exclude '*.log' --exclude '*.bbl' --exclude '*.blg' --exclude '*.out' --exclude '*.pdf' --exclude 'compile-*' "$LIVE/$f/" "$G/$f/"
  rsync -a "$LIVE/$f/figures/" "$G/$f/figures/"; done
shasum -a 256 "$LIVE"/v2/DRAFT-v2.{tex,pdf} "$LIVE"/style/DRAFT-v2-style.{tex,pdf} "$LIVE"/*/refs_clu_iclr.bib "$LIVE"/*/figures/*.pdf > "$G/snap/SHA256-live-start.txt"
# 1 compile
bash "$R/pw2_draft_sections/compile_folder.sh" "$G/v2" DRAFT-v2; bash "$R/pw2_draft_sections/compile_folder.sh" "$G/style" DRAFT-v2-style
for p in v2/DRAFT-v2 style/DRAFT-v2-style; do b=$(basename $p)
  uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$G/$p.pdf" "$G/checks/$b.pdf.txt"
  uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$LIVE/$p.pdf" "$G/checks/$b.live.pdf.txt"
  cmp -s "$G/checks/$b.live.pdf.txt" "$G/checks/$b.pdf.txt" && echo "$b: installed PDF text == fresh build text (all pages)" || echo "$b: INSTALLED PDF TEXT DIFFERS"; done > "$G/checks/LIVE-PDF-VS-BUILD.txt"
# 2 style diff-check; 3 provenance now and PW-1 final
(cd "$R/pw1_draft_sweeps" && python3 style_diff_check.py "$G/v2/DRAFT-v2.tex" "$G/style/DRAFT-v2-style.tex" > "$G/checks/STYLE-DIFF-CHECK.md")
PW1=~/clu/outputs/pw1-draft-citations/stage-145204/post-install
for d in DRAFT-v2 DRAFT-v2-style; do f=v2; [ $d = DRAFT-v2-style ] && f=style
  (cd "$R/pw1_draft_sweeps" && python3 check_provenance.py "$G/$f/$d.tex" > "$G/checks/prov-now-$d.txt" && python3 check_provenance.py "$PW1/$d.tex" > "$G/checks/prov-pw1-$d.txt")
done
for d in DRAFT-v2 DRAFT-v2-style; do echo "## $d"; python3 "$H/prov_compare.py" "$G/checks/prov-pw1-$d.txt" "$G/checks/prov-now-$d.txt"; done > "$G/checks/PROVENANCE-VS-PW1.md"
# 4-5 figure/table index and page markers
bash "$R/pw2_draft_sections/measure.sh" "$G" after
# 6 numbers; 7 tells; 8 head-held; 9 TODO; 10 cite; defects
SN=~/clu/outputs/pw2-draft-leftovers/snap-start; HH=~/clu/outputs/pw2-draft-leftovers/HEAD-HELD.md
for p in v2/DRAFT-v2 style/DRAFT-v2-style; do b=$(basename $p)
  python3 "$H/rounded_check.py" "$G/$p.tex" > "$G/numbers/ROUNDED-$b.md"
  python3 "$H/numclass.py" "$G/$p.tex" --json "$G/numbers/$b-class.json" > "$G/numbers/$b-class.md"
  python3 "$H/compare_numbers.py" "$G/numbers/$b-class.json" ~/clu/outputs/pw2-draft-leftovers/NUMBERS-LEFT.md "$b.tex" > "$G/numbers/COMPARE-$b.md"
  (cd "$R/pw2_draft_sections" && python3 headheld.py "$G/$p.tex" "$HH" > "$G/checks/HEADHELD-$b.txt")
  python3 "$H/scans.py" headheld "$G/$p.tex" "$SN/$p.tex" "$HH" "$G/checks/HEADHELD-$b.txt" > "$G/checks/HEADHELD-RAW-$b.md"
  python3 "$H/scans.py" todo "$G/$p.tex" "$G/checks/$b.pdf.txt" > "$G/checks/TODO-$b.md"
  python3 "$H/scans.py" cite "$G/$p.tex" > "$G/checks/CITE-$b.md"
  python3 "$H/scans.py" defects "$G/$p.tex" "$G/checks/$b.pdf.txt" > "$G/checks/DEFECTS-$b.md"
done
python3 "$H/tells_now.py" "$G/v2/DRAFT-v2.tex" "$G/style/DRAFT-v2-style.tex" ~/clu/outputs/pw2-draft-tells/TELLS-COUNT.md > "$G/tells/TELLS-NOW.md"
# 11 bib
{ echo "| file | sha256 | 56 entries |"; echo "|---|---|---|"
  for f in v2 style; do h=$(shasum -a 256 "$LIVE/$f/refs_clu_iclr.bib" | cut -c1-64); n=$(grep -c '^@' "$LIVE/$f/refs_clu_iclr.bib")
    echo "| \`live/$f/refs_clu_iclr.bib\` | \`$h\` $( [ $h = b2cf715f25e020acf02038d5df4a2fd0ce816fb59e0aa3cfb101335684a71bd3 ] && echo '= b2cf715f… **unchanged**' || echo '**CHANGED**') | $n |"; done; } > "$G/checks/BIB.txt"
# controls
bash "$H/controls.sh" "$G" > "$G/controls-run.txt" 2>&1; mkdir -p "$G/controls"; grep -E "^CONTROL|^  " "$G/controls-run.txt" > "$G/controls/CONTROLS.txt"
echo "stage: $G"
