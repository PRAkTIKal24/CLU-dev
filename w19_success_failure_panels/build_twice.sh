#!/bin/bash
# Build every w19 panel set twice in separate processes, compare SHA-256, install only if equal.
set -euo pipefail
cd "$(dirname "$0")/.."
source /Users/batmac/clu/outputs/w19-success-failure-panels/run_env.sh
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
B=$W19_SF_OUT/builds/$STAMP
mkdir -p $B/b1 $B/b2
python -m w19_success_failure_panels.build $B/b1 > $B/b1.log 2>&1
python -m w19_success_failure_panels.build $B/b2 > $B/b2.log 2>&1
(cd $B/b1 && shasum -a 256 * | sort -k2) > $B/SHA256-w19-build1.txt
(cd $B/b2 && shasum -a 256 * | sort -k2) > $B/SHA256-w19-build2.txt
if ! cmp -s $B/SHA256-w19-build1.txt $B/SHA256-w19-build2.txt; then
  diff $B/SHA256-w19-build1.txt $B/SHA256-w19-build2.txt; echo "REBUILD MISMATCH -- not installed"; exit 1
fi
DST=/Users/batmac/clu/outputs/w17-paper/figures/rendered
cp $B/b1/* $DST/
cp $B/SHA256-w19-build1.txt $B/SHA256-w19-build2.txt $DST/
echo "BYTE-IDENTICAL; installed $(ls $B/b1 | wc -l) files from $B/b1 to $DST"
