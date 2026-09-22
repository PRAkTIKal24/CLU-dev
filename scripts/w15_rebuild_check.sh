#!/usr/bin/env bash
# w15 spoke/crossover-seeds -- THE ACCEPTANCE CHECK: byte-identical on rebuild.
#
# The acceptance criterion names four artefacts and adds "byte-identical on rebuild".
# This script hashes them, rebuilds all four from the same per-(K, seed) `row.json`
# files, hashes them again, and fails loudly on any difference.  It writes ONLY inside
# this spoke's output dir and never touches w12's tree.
set -eu
OUTD=/c/chlu/outputs/w15-crossover-seeds
cd /c/chlu/worktrees/CHLU-w15-crossover-seeds
source "$OUTD/env.sh"

FILES="CROSSOVER-3SEED.json CROSSOVER-3SEED.md fig_crossover_3seed.pdf fig_crossover_3seed.png"

echo "--- pass 1 ---"
python scripts/w15_crossover_report.py >/dev/null
python scripts/w15_crossover_figure.py >/dev/null
for f in $FILES; do sha256sum "$OUTD/$f"; done > "$OUTD/REBUILD-pass1.sha256"

echo "--- pass 2 (rebuilt from the same rows) ---"
python scripts/w15_crossover_report.py >/dev/null
python scripts/w15_crossover_figure.py >/dev/null
for f in $FILES; do sha256sum "$OUTD/$f"; done > "$OUTD/REBUILD-pass2.sha256"

if diff -u "$OUTD/REBUILD-pass1.sha256" "$OUTD/REBUILD-pass2.sha256"; then
  echo "BYTE-IDENTICAL ON REBUILD: PASS"
  cp "$OUTD/REBUILD-pass1.sha256" "$OUTD/ARTEFACTS.sha256"
  exit 0
fi
echo "BYTE-IDENTICAL ON REBUILD: FAIL"
exit 1
