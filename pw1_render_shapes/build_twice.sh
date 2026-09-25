#!/bin/bash
# Build the PW-1 Shapes figures twice in separate processes into <OUT>/build/{a,b}; compare
# SHA-256 manifests; exit 1 on any difference.  Nothing is installed (a later spoke places them).
set -euo pipefail
cd "$(dirname "$0")/.."
source pw1_render_shapes/run_env.sh
B=$PW1_RS_OUT/build
for x in a b; do
  if [ -e $B/$x ]; then mv $B/$x $PW1_RS_OUT/trial/superseded-build-$x-$(date -u +%Y%m%dT%H%M%SZ); fi
  mkdir -p $B/$x
  python -m pw1_render_shapes.build $B/$x > $B/build-$x.log 2>&1
  python -m pw1_render_shapes.provenance $B/$x
done
(cd $B/a && shasum -a 256 * | sort -k2) > $B/SHA256-build1.txt
(cd $B/b && shasum -a 256 * | sort -k2) > $B/SHA256-build2.txt
cp $B/SHA256-build1.txt $B/a/; cp $B/SHA256-build2.txt $B/b/
if ! cmp -s $B/SHA256-build1.txt $B/SHA256-build2.txt; then
  diff $B/SHA256-build1.txt $B/SHA256-build2.txt || true; echo "REBUILD MISMATCH"; exit 1
fi
cp $B/a/REQUESTS-pw1.json $B/a/PROVENANCE-render.json $PW1_RS_OUT/
echo "BYTE-IDENTICAL: $(wc -l < $B/SHA256-build1.txt) files"
