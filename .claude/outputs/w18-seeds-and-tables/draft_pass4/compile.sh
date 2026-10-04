#!/bin/bash
# usage: compile.sh TEXFILE LABEL  -- compiles a copy in a scratch build dir laid out like w17-paper/
set -u
P=/Users/batmac/clu/outputs/w17-paper
S=/private/tmp/claude-501/-Users-batmac-clu-worktrees-CHLU-w18-seeds-and-tables/42a585c7-6859-47af-b109-7e9fcbb6f121/scratchpad/pass4
B=$S/build-$2; rm -rf $B; mkdir -p $B
cp -R $P/iclr2027 $P/figures $B/; cp $P/references.bib $B/; cp $P/*.tex $B/ 2>/dev/null; cp $P/../w17-paper/iclr2027/math_commands.tex $B/ 2>/dev/null
cp "$1" $B/doc.tex
cd $B
export BIBINPUTS=$P: BSTINPUTS=$P:$P/iclr2027:
pdflatex -interaction=nonstopmode doc.tex >/dev/null 2>&1; e1=$?
bibtex doc >bibtex.out 2>&1; eb=$?
for i in 1 2 3; do pdflatex -interaction=nonstopmode doc.tex >/dev/null 2>&1; e=$?; done
undef_c=$(grep -c "Citation .* undefined" doc.log)
undef_r=$(grep -c "Reference .* undefined" doc.log)
pages=$(grep -o 'Output written on doc.pdf ([0-9]* pages' doc.log | grep -o '[0-9]* pages')
echo "$2: pdflatex1=$e1 bibtex=$eb pdflatex_last=$e pages=[$pages] undefined_citations=$undef_c undefined_refs=$undef_r bibtex_warn=$(grep -c -i warning bibtex.out) overfull=$(grep -c Overfull doc.log) warnings=$(grep -c -i 'warning' doc.log)"
