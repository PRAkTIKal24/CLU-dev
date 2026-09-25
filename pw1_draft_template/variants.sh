#!/usr/bin/env bash
# usage: bash variants.sh OUTDIR
# Builds, in fresh copies of the w17-paper tree, one variant per row of the table below, for both drafts:
#   V0      the live drafts unchanged (the "before" of every comparison, built in scratch)
#   Vh      template fancyhdr/natbib only
#   Vl      list keys only
#   Vhl     both (the candidate install)
#   L<n>    one list-key removal at source line n alone (per-override page effect)
# Every variant: compile_tree metrics, PDF text, label pages (.aux \newlabel), and where the AI-use statement starts.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$HERE")"
O="$1"; P=/Users/batmac/clu/outputs/w17-paper
export PATH="/Library/TeX/texbin:/Users/batmac/clu/workspace/.venv/bin:$PATH"
mkdir -p "$O"
EXC=(--exclude='/DRAFT-v2*.aux' --exclude='/DRAFT-v2*.bbl' --exclude='/DRAFT-v2*.blg' --exclude='/DRAFT-v2*.log'
     --exclude='/DRAFT-v2*.out' --exclude='/DRAFT-v2*.pdf' --exclude='/compile-*.log')
build() {  # name  only-arg-for-v2  only-arg-for-style   (only-arg "none" = copy unchanged)
  local v="$1"; local T="$O/$v"; rsync -a "${EXC[@]}" "$P/" "$T/"
  for pair in "DRAFT-v2:$2" "DRAFT-v2-style:$3"; do
    d="${pair%%:*}"; a="${pair#*:}"
    if [ "$a" != none ]; then mv "$T/$d.tex" "$T/$d.orig.tex.txt"
      python3 "$HERE/apply_template.py" "$T/$d.orig.tex.txt" "$T/$d.tex" --only "$a" > "$T/EDITS-$d.txt"; fi
  done
  bash "$R/pw1_draft_final/compile_tree.sh" "$T" > "$T/COMPILE.txt"
  for d in DRAFT-v2 DRAFT-v2-style; do
    uv run -q --with pypdf python "$R/pw1_draft_cuts/pdftext.py" "$T/$d.pdf" "$T/$d.pdf.txt" > /dev/null 2>&1
    grep newlabel "$T/$d.aux" > "$T/$d.labels.txt"
  done
  echo "== $v"; cat "$T/COMPILE.txt"
}
"${@:2}"
