#!/bin/bash
# build.sh <dir>  -- 3-pass pdflatex build, reports pages/overfull/underfull
set -e
D="$1"
export PATH="/Library/TeX/texbin:$PATH"
cd "$D"
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode submission.tex >/dev/null 2>&1 || true
done
echo "pages: $(/opt/homebrew/bin/mutool info submission.pdf 2>/dev/null | grep -i 'Pages:' | head -1)"
echo "overfull: $(grep -c 'Overfull' submission.log || true)  underfull: $(grep -c 'Underfull' submission.log || true)  undefined: $(grep -c 'undefined' submission.log || true)"
