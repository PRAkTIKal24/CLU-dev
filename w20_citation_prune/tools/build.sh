#!/bin/zsh
# build.sh <texdir> <builddir> : pdflatex -> bibtex -> pdflatex x3 for both drafts; prints counts from logs.
set -u
T=$1; D=$2; P=/Users/batmac/clu/outputs/w17-paper
mkdir -p $D; cp $T/DRAFT-v2.tex $T/DRAFT-v2-style.tex $T/references.bib $D/
ln -sfn $P/figures $D/figures; ln -sfn $P/iclr2027 $D/iclr2027
cd $D
export BIBINPUTS=$D:$P:$P/iclr2027: BSTINPUTS=$D:$P:$P/iclr2027:
for f in DRAFT-v2 DRAFT-v2-style; do
  L=compile-$f.log; : > $L
  pdflatex -interaction=nonstopmode -halt-on-error $f.tex >> $L 2>&1; e1=$?
  bibtex $f >> $L 2>&1; eb=$?
  for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error $f.tex >> $L 2>&1; done; e2=$?
  undef_cit=$(grep -c "Citation .* undefined" $f.log)
  undef_ref=$(grep -c "Reference .* undefined" $f.log)
  terr=$(grep -c '^!' $f.log)
  bibitems=$(grep -c '\\bibitem' $f.bbl)
  blgw=$(grep -ciE 'warning|error' $f.blg)
  pages=$(grep -oE 'Output written on [^ ]+ \(([0-9]+) pages' $f.log | grep -oE '[0-9]+ pages')
  echo "$f | pdflatex1 exit $e1 | bibtex exit $eb | last pdflatex exit $e2 | $pages | bibitems $bibitems | blg warn/err $blgw | UNDEFINED CITATIONS $undef_cit | undefined refs $undef_ref | TeX ! errors $terr"
done
