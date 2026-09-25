#!/usr/bin/env bash
# PW-1 draft-final: the closing audit on the LIVE tree (jobs 1-6), writing only under OUTDIR.
# usage: bash run_final.sh OUTDIR [--inplace]
#   --inplace additionally compiles in ~/clu/outputs/w17-paper itself (writes .aux/.bbl/.blg/.log/.out/.pdf there).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$HERE")"
O="$1"; INPLACE="${2:-}"; P=/Users/batmac/clu/outputs/w17-paper; PV=/Users/batmac/clu/outputs/paper-visuals
export PATH="/Library/TeX/texbin:/Users/batmac/clu/workspace/.venv/bin:$PATH"
mkdir -p "$O"
EXC=(--exclude='/DRAFT-v2*.aux' --exclude='/DRAFT-v2*.bbl' --exclude='/DRAFT-v2*.blg' --exclude='/DRAFT-v2*.log'
     --exclude='/DRAFT-v2*.out' --exclude='/DRAFT-v2*.pdf' --exclude='/compile-*.log')
# job 1a: scratch compile from a clean copy of the w17-paper tree
rsync -a "${EXC[@]}" "$P/" "$O/scratch/"
bash "$HERE/compile_tree.sh" "$O/scratch" > "$O/COMPILE-scratch.txt"
# job 1b: in place
if [ "$INPLACE" = --inplace ]; then
  bash "$HERE/compile_tree.sh" "$P" > "$O/COMPILE-inplace.txt"
  mv "$P/compile-DRAFT-v2.log" "$O/compile-inplace-DRAFT-v2.log"; mv "$P/compile-DRAFT-v2-style.log" "$O/compile-inplace-DRAFT-v2-style.log"
fi
for d in DRAFT-v2 DRAFT-v2-style; do
  uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$O/scratch/$d.pdf" "$O/scratch-$d.pdf.txt" > /dev/null 2>&1
  uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$P/$d.pdf" "$O/inplace-$d.pdf.txt" > /dev/null 2>&1
  for e in aux bbl out; do cmp -s "$P/$d.$e" "$O/scratch/$d.$e" && echo "$d.$e in-place == scratch" || echo "$d.$e DIFFERS"; done
  cmp -s "$O/scratch-$d.pdf.txt" "$O/inplace-$d.pdf.txt" && echo "$d PDF text in-place == scratch" || echo "$d PDF text DIFFERS"
  grep '^Overfull' "$P/$d.log" > "$O/OVERFULL-$d.txt"; echo "$d overfull boxes: $(wc -l < "$O/OVERFULL-$d.txt" | tr -d ' ')"
done > "$O/JOB1-compare.txt"
# style diff-check
python3 "$R/pw1_draft_sweeps/style_diff_check.py" "$P/DRAFT-v2.tex" "$P/DRAFT-v2-style.tex" > "$O/STYLE-DIFF-CHECK.md"
for d in DRAFT-v2 DRAFT-v2-style; do
  python3 "$R/pw1_draft_sweeps/check_provenance.py" "$P/$d.tex" > "$O/PROV-$d.txt"
  python3 "$R/pw1_draft_wording2/check_derived.py" "$P/$d.tex" "$O/PROV-$d.txt" > "$O/PROV-DERIVED-$d.txt"
done
# job 2: figure/table index
python3 "$HERE/fig_table_index.py" "$P" "$O/FIGTAB-INDEX.json" > "$O/FIGTAB-INDEX.txt"
# job 3: page markers on a scratch copy (A: \typeout{\thepage}; C: shipped page + y inside the heading)
rsync -a "${EXC[@]}" "$P/" "$O/pages/"
for d in DRAFT-v2 DRAFT-v2-style; do
  mv "$O/pages/$d.tex" "$O/pages/$d.orig.tex.txt"
  python3 "$HERE/page_markers_c.py" "$O/pages/$d.orig.tex.txt" "$O/pages/$d.tex" > "$O/MARKS-$d.json"
done
bash "$HERE/compile_tree.sh" "$O/pages" > "$O/COMPILE-pages.txt"
for d in DRAFT-v2 DRAFT-v2-style; do
  diff <(grep newlabel "$O/scratch/$d.aux") <(grep newlabel "$O/pages/$d.aux") > /dev/null \
    && echo "$d: marker build, every \\newlabel identical to the unmarked build" || echo "$d: MARKERS MOVED A LABEL"
  python3 "$HERE/pages.py" "$O/MARKS-$d.json" "$O/pages" "$d" "$O/scratch/$d.aux"
  cp "$O/pages/PAGES-$d.json" "$O/"
done > "$O/JOB3-pages.txt"
# job 4
for d in DRAFT-v2 DRAFT-v2-style; do python3 "$HERE/ph_list.py" "$P/$d.tex" "$P/$d.aux" "$O/PH-$d.json"; done > "$O/JOB4-ph.txt"
# job 5
for d in DRAFT-v2 DRAFT-v2-style; do
  python3 "$HERE/residual_audit.py" "$P/$d.tex" "$O/inplace-$d.pdf.txt" "$O/AUDIT-$d.json" > "$O/AUDIT-$d.txt"
done
# job 6: figure folders -- placed stems in both folders, equal hashes; README regenerated in a copy == installed README
python3 "$HERE/folder_audit.py" "$P" "$PV" "$O" > "$O/JOB6-folders.txt"
echo "run_final done: $O"
