#!/usr/bin/env bash
# usage: compile_check.sh <dir holding DRAFT-v2.tex and DRAFT-v2-style.tex> <build dir (new)> [figures dir]
# Lays the build out like w17-paper (iclr2027/, figures/, references.bib), runs
# pdflatex -> bibtex -> pdflatex x3 on each draft, and prints one metrics row per draft.
set -uo pipefail
SRC="$1"; B="$2"; P=/Users/batmac/clu/outputs/w17-paper; FIG="${3:-$P/figures}"
mkdir -p "$B"
cp "$SRC/DRAFT-v2.tex" "$SRC/DRAFT-v2-style.tex" "$B/"
cp -R "$P/iclr2027" "$B/iclr2027"
cp "$P/references.bib" "$B/"
ln -s "$FIG" "$B/figures"
cd "$B"
export BIBINPUTS="$B:$P:" BSTINPUTS="$B/iclr2027:$P/iclr2027:"
for d in DRAFT-v2 DRAFT-v2-style; do
  { pdflatex -interaction=nonstopmode "$d.tex"; e1=$?
    bibtex "$d"; eb=$?
    pdflatex -interaction=nonstopmode "$d.tex"; pdflatex -interaction=nonstopmode "$d.tex"
    pdflatex -interaction=nonstopmode "$d.tex"; e4=$?
    echo "EXITS pdflatex1=$e1 bibtex=$eb pdflatex_last=$e4"; } > "compile-$d.log" 2>&1
  L="$d.log"
  ucit=$(grep -c "Citation .* undefined" "$L"); uref=$(grep -c "Reference .* undefined" "$L")
  err=$(grep -c "^!" "$L"); ovf=$(grep -c "Overfull" "$L"); bib=$(grep -c "\\\\bibitem" "$d.bbl" 2>/dev/null)
  pages=$(grep -o "Output written on .* (\([0-9]*\) pages" "$L" | grep -o "([0-9]* pages" | tr -dc 0-9)
  echo "$d | ucit=$ucit uref=$uref errors=$err overfull=$ovf bibitems=$bib pages=$pages | $(tail -1 compile-$d.log)"
done
