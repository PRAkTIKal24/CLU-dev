#!/usr/bin/env bash
# usage: compile_folder.sh FOLDER STEM -- pdflatex, bibtex, pdflatex until no "Rerun" (max 5), built FROM the folder; one metrics row.
set -uo pipefail
cd "$1"; d="$2"
{ pdflatex -interaction=nonstopmode "$d.tex"; e1=$?
  bibtex "$d"; eb=$?
  n=1; ex=""
  while [ $n -lt 6 ]; do pdflatex -interaction=nonstopmode "$d.tex"; ex="$ex $?"; n=$((n+1))
    grep -q "Rerun to get" "$d.log" || break; done
  echo "EXITS pdflatex1=$e1 bibtex=$eb later_pdflatex=[$ex ] passes=$n"; } > "compile-$d.log" 2>&1
L="$d.log"
ucit=$(grep -c "Citation .* undefined" "$L"); uref=$(grep -c "Reference .* undefined" "$L")
err=$(grep -c "^!" "$L"); ovf=$(grep -c "^Overfull" "$L"); unf=$(grep -c "^Underfull" "$L")
rerun=$(grep -c "Rerun to get" "$L"); bib=$(grep -c "\\\\bibitem" "$d.bbl" 2>/dev/null)
mlab=$(grep -c "multiply defined" "$L")
pages=$(grep -o "Output written on .* (\([0-9]*\) pages" "$L" | grep -o "([0-9]* pages" | tr -dc 0-9)
echo "$d | ucit=$ucit uref=$uref errors=$err multiply_defined=$mlab overfull=$ovf underfull=$unf rerun=$rerun bibitems=$bib pages=$pages | $(tail -1 compile-$d.log)"
