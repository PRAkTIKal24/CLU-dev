#!/usr/bin/env bash
# usage: measure.sh G SIDE   (SIDE = before | after; after = G/v2, G/style) -- fig/table index + exact page markers
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$H"); G="$1"; SIDE="$2"
if [ "$SIDE" = before ]; then B="$G/before"; else B="$G"; fi
M="$G/measure-$SIDE"; mkdir -p "$M/both"
cp "$B/v2/DRAFT-v2.tex" "$B/v2/DRAFT-v2.aux" "$B/style/DRAFT-v2-style.tex" "$B/style/DRAFT-v2-style.aux" "$M/both/"
python3 "$R/pw1_draft_final/fig_table_index.py" "$M/both" "$M/FIGTAB-INDEX.json" > "$M/FIGTAB-INDEX.txt"
for pair in "v2 DRAFT-v2" "style DRAFT-v2-style"; do
  f=${pair%% *}; d=${pair##* }
  rsync -a --exclude '*.pdf' --exclude '*.aux' --exclude '*.log' --exclude '*.bbl' --exclude '*.blg' --exclude '*.out' "$B/$f/" "$M/pages-$f/"
  rsync -a "$B/$f/figures/" "$M/pages-$f/figures/"
  mv "$M/pages-$f/$d.tex" "$M/pages-$f/$d.orig.tex.txt"
  python3 "$H/page_markers_c.py" "$M/pages-$f/$d.orig.tex.txt" "$M/pages-$f/$d.tex" > "$M/MARKS-$d.json"
  "$H/compile_folder.sh" "$M/pages-$f" "$d"
  diff <(grep newlabel "$B/$f/$d.aux") <(grep newlabel "$M/pages-$f/$d.aux") > /dev/null \
    && echo "$d: marker build, every \\newlabel identical to the unmarked build" || echo "$d: MARKERS MOVED A LABEL"
  python3 "$R/pw1_draft_final/pages.py" "$M/MARKS-$d.json" "$M/pages-$f" "$d" "$B/$f/$d.aux"
  cp "$M/pages-$f/PAGES-$d.json" "$M/"
done
