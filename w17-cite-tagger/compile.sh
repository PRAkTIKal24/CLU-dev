#!/bin/bash
# compile.sh <dir> <stem> <log>: pdflatex -> bibtex -> pdflatex x3 (third pass settles page refs after the bibliography lands), every step logged with exit code
cd "$1" || exit 2
{
echo "### $(date -u +%FT%TZ) host=$(hostname) cwd=$PWD"
pdflatex --version | head -1; bibtex --version | head -1
for step in "pdflatex -interaction=nonstopmode -halt-on-error $2.tex" "bibtex $2" \
            "pdflatex -interaction=nonstopmode -halt-on-error $2.tex" \
            "pdflatex -interaction=nonstopmode -halt-on-error $2.tex" \
            "pdflatex -interaction=nonstopmode -halt-on-error $2.tex"; do
  echo "### RUN: $step"; $step; echo "### EXIT: $?"
done
echo "### ---- $2.blg ----"; cat "$2.blg"
echo "### ---- warnings in final $2.log ----"
grep -n -E "Citation .* undefined|There were undefined|Reference .* undefined|Label\(s\) may have changed|multiply defined|Error" "$2.log" || echo "(no undefined citations/references, no errors)"
} > "$3" 2>&1
grep "### EXIT" "$3" | tr '\n' ' '; echo "<- $2"
