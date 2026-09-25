#!/usr/bin/env bash
# usage: neutral_builds.sh STAGE  -- control builds N0 (from V0) and N2 (from V2) with every citation command and the
# bibliography printing nothing (neutralise.py), then the rendered word diff N0 -> N2 per draft.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$HERE")"; S="$1"
for p in "V0 N0" "V2 N2"; do
  set -- $p
  rsync -a --exclude '*.aux' --exclude '*.bbl' --exclude '*.blg' --exclude '*.log' --exclude '*.out' \
        --exclude 'DRAFT-v2*.pdf' --exclude '*.pdf.txt' --exclude 'COMPILE.txt' "$S/$1/" "$S/$2/"
  for d in DRAFT-v2 DRAFT-v2-style; do python3 "$HERE/neutralise.py" "$S/$1/$d.tex" "$S/$2/$d.tex"; done
  bash "$R/pw1_draft_final/compile_tree.sh" "$S/$2" > "$S/$2/COMPILE.txt" 2>&1; cat "$S/$2/COMPILE.txt"
  for d in DRAFT-v2 DRAFT-v2-style; do
    uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$S/$2/$d.pdf" "$S/$2/$d.pdf.txt" > /dev/null 2>&1; done
done
for d in DRAFT-v2 DRAFT-v2-style; do
  python3 "$R/pw1_draft_template/worddiff.py" "$S/N0/$d.pdf.txt" "$S/N2/$d.pdf.txt" > "$S/WORDDIFF-neutral-$d.txt"
  echo "$d neutral word diff exit $?"
done
