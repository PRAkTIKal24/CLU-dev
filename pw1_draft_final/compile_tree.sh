#!/usr/bin/env bash
# usage: compile_tree.sh TREE  -- builds DRAFT-v2 and DRAFT-v2-style inside TREE (a w17-paper-layout dir:
# iclr2027/, references.bib, figures/); pdflatex -> bibtex -> pdflatex until no "Rerun" (max 5); one metrics row per draft.
set -uo pipefail
cd "$1"
for d in DRAFT-v2 DRAFT-v2-style; do
  { pdflatex -interaction=nonstopmode "$d.tex"; e1=$?
    bibtex "$d"; eb=$?
    n=1; ex=""
    # from a clean tree the label pass needs one more run than the w17 recipe; run until LaTeX stops asking (max 5)
    while [ $n -lt 6 ]; do pdflatex -interaction=nonstopmode "$d.tex"; ex="$ex $?"; n=$((n+1))
      grep -q "Rerun to get" "$d.log" || break; done
    echo "EXITS pdflatex1=$e1 bibtex=$eb later_pdflatex=[$ex ] passes=$n"; } > "compile-$d.log" 2>&1
  L="$d.log"
  ucit=$(grep -c "Citation .* undefined" "$L"); uref=$(grep -c "Reference .* undefined" "$L")
  err=$(grep -c "^!" "$L"); ovf=$(grep -c "^Overfull" "$L"); unf=$(grep -c "^Underfull" "$L")
  rerun=$(grep -c "Rerun to get" "$L"); bib=$(grep -c "\\\\bibitem" "$d.bbl" 2>/dev/null)
  bwarn=$(grep -c "^Warning--" "$d.blg" 2>/dev/null)
  pages=$(grep -o "Output written on .* (\([0-9]*\) pages" "$L" | grep -o "([0-9]* pages" | tr -dc 0-9)
  echo "$d | ucit=$ucit uref=$uref errors=$err overfull=$ovf underfull=$unf rerun=$rerun bibitems=$bib bibtexwarn=$bwarn pages=$pages | $(tail -1 compile-$d.log)"
done
