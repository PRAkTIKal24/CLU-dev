#!/bin/bash
# w18 placeholder-fill pass 2: compile one draft in the current dir.
# pdflatex -> bibtex (BIBINPUTS/BSTINPUTS = w17-paper/) -> pdflatex x3; transcript to compile-<stem>.log
P=/Users/batmac/clu/outputs/w17-paper
export TEXINPUTS=.:$P/: BIBINPUTS=$P/ BSTINPUTS=$P/
stem=$1; L=compile-$stem.log
{ echo "### $(date -u +%FT%TZ) host=$(hostname -s) cwd=$PWD TEXINPUTS=$TEXINPUTS BIBINPUTS=$BIBINPUTS BSTINPUTS=$BSTINPUTS"
  pdflatex --version | head -1; bibtex --version | head -1
  for step in "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex" "bibtex $stem" \
              "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex" \
              "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex" \
              "pdflatex -interaction=nonstopmode -halt-on-error $stem.tex"; do
    echo "### RUN: $step"; $step; echo "### EXIT: $?"; done; } > $L 2>&1
echo "$stem: $(grep '### EXIT' $L | tr '\n' ' ')"
echo "  bibtex errors: $(grep -c 'error message' $stem.blg; grep -m1 "I couldn't open" $stem.blg)"
echo "  errors(!) in final pass: $(awk '/### RUN/{n++} n==5' $L | grep -c '^!')"
echo "  undefined citations (final .log): $(grep -c 'Citation .* undefined' $stem.log)"
echo "  undefined references (final .log): $(grep -c 'Reference .* undefined' $stem.log)"
echo "  'There were undefined' (final .log): $(grep -c 'There were undefined' $stem.log)"
echo "  labels may have changed: $(grep -c 'Label(s) may have changed' $stem.log)"
echo "  overfull: $(grep -c 'Overfull' $stem.log)"
echo "  $(grep -o 'Output written on .*' $stem.log)"
