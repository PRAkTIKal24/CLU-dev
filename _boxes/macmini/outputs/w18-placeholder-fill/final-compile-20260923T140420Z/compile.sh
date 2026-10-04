#!/bin/bash
# compile one draft: pdflatex, bibtex, pdflatex x2; full transcript to compile-<stem>.log
stem=$1; L=compile-$stem.log
{ echo "### $(date -u +%FT%TZ) host=$(hostname -s) cwd=$PWD"; pdflatex --version | head -1; bibtex --version | head -1;
  for step in "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex" "bibtex $stem" "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex" "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex"; do
    echo "### RUN: $step"; $step; echo "### EXIT: $?"; done; } > $L 2>&1
grep "### EXIT" $L | tr '\n' ' '; echo; grep -c "^!" $L | sed 's/^/errors(!): /'; grep -c "Warning: Reference\|undefined" $stem.log | sed 's/^/undefined-ref lines: /'; grep -o "Output written on .*" $stem.log
