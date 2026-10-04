#!/usr/bin/env bash
# Build the figures twice into scratch dirs, prove byte-identity, then install into the
# shared draft's figures/ dir (never touching figures/rendered/, which another spoke owns).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/Users/batmac/clu/outputs/w17-paper/figures}"
TMP="${TMPDIR:-/tmp}/w17-plots-build.$$"
mkdir -p "$TMP/a" "$TMP/b"
python "$HERE/make_figures.py" --out "$TMP/a"
python "$HERE/make_figures.py" --out "$TMP/b"
(cd "$TMP/a" && shasum -a 256 * | sort -k2) > "$TMP/sha_a.txt"
(cd "$TMP/b" && shasum -a 256 * | sort -k2) > "$TMP/sha_b.txt"
if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
mkdir -p "$DEST"
cp "$TMP"/a/fig_*.pdf "$TMP"/a/fig_*.png "$TMP/a/PROVENANCE.json" "$DEST/"
cp "$HERE/make_figures.py" "$HERE/build.sh" "$DEST/"
cp "$TMP/sha_a.txt" "$DEST/SHA256-build1.txt"
cp "$TMP/sha_b.txt" "$DEST/SHA256-build2.txt"
rm -rf "$TMP"
