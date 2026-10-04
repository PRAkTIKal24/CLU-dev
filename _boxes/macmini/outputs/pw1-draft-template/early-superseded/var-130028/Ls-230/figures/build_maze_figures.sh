#!/usr/bin/env bash
# Build the maze figures twice, prove byte-identity, then install ONLY fig_maze_* into the shared
# figures/ dir (never figures/rendered/, never the non-maze figures). The replaced files are kept
# in $BACKUP first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/Users/batmac/clu/outputs/w17-paper/figures}"
BACKUP="${2:?backup dir}"
TMP="${TMPDIR:-/tmp}/w18-maze-figs.$$"
mkdir -p "$TMP/a" "$TMP/b" "$BACKUP"
python "$HERE/make_maze_figures.py" --out "$TMP/a"
python "$HERE/make_maze_figures.py" --out "$TMP/b"
(cd "$TMP/a" && shasum -a 256 * | sort -k2) > "$TMP/sha_a.txt"
(cd "$TMP/b" && shasum -a 256 * | sort -k2) > "$TMP/sha_b.txt"
if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
for f in fig_maze_readverb.pdf fig_maze_readverb.png fig_maze_requestlayer.pdf fig_maze_requestlayer.png; do
  [ -e "$DEST/$f" ] && cp -p "$DEST/$f" "$BACKUP/$f"
done
cp "$TMP"/a/fig_maze_*.pdf "$TMP"/a/fig_maze_*.png "$TMP/a/PROVENANCE-maze-w18.json" "$DEST/"
cp "$HERE/make_maze_figures.py" "$HERE/w17_make_figures.py" "$HERE/build_maze_figures.sh" "$DEST/"
cp "$TMP/sha_a.txt" "$DEST/SHA256-maze-w18-build1.txt"
cp "$TMP/sha_b.txt" "$DEST/SHA256-maze-w18-build2.txt"
cat "$TMP/sha_a.txt"
