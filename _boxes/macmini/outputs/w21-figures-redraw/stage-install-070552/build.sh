#!/usr/bin/env bash
# Build the figures twice into scratch dirs, prove byte-identity, then install into the
# shared draft's figures/ dir (never touching figures/rendered/, which another spoke owns).
#
# w21 figures-redraw extends this build (it does not replace it):
#   1. make_figures.py      -- the w17 figures, now with NO in-image banner (tag_placeholder is a no-op)
#   2. make_figures_w21.py  -- redraws fig_crossover / fig_ladder / fig_deletion over step 1's files,
#                              adds fig_maze_conditions, and (Part C) fig_cmazet_* when the verified
#                              per-item extraction data/cmazet_items.json is present
#   3. PROVENANCE.json      -- step 1's entries, overridden per stem by step 2's (merge, never drop)
# Rows / rival sets / panel sets are DATA: data/maze_conditions.json, data/ladder_rival_sets.json,
# data/cmazet_panels.json. Adding one is an edit there plus a rebuild, not a code change.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/Users/batmac/clu/outputs/w17-paper/figures}"
TMP="${TMPDIR:-/tmp}/w21-figures-build.$$"
PANELS=()
[ -f "$HERE/data/cmazet_items.json" ] && PANELS=(--panels "$HERE/data/cmazet_items.json")
mkdir -p "$TMP/a" "$TMP/b"
for d in a b; do
  python "$HERE/make_figures.py" --out "$TMP/$d"
  python "$HERE/make_figures_w21.py" --out "$TMP/$d" ${PANELS[@]+"${PANELS[@]}"} --merge "$TMP/$d/PROVENANCE.json"
done
(cd "$TMP/a" && shasum -a 256 * | sort -k2) > "$TMP/sha_a.txt"
(cd "$TMP/b" && shasum -a 256 * | sort -k2) > "$TMP/sha_b.txt"
if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
mkdir -p "$DEST" "$DEST/data"
cp "$TMP"/a/fig_*.pdf "$TMP"/a/fig_*.png "$TMP/a/PROVENANCE.json" "$DEST/"
cp "$HERE/make_figures.py" "$HERE/make_figures_w21.py" "$HERE/validate_palette.py" "$HERE/build.sh" "$DEST/"
[ -f "$HERE/cmazet_panels.py" ] && cp "$HERE/cmazet_panels.py" "$DEST/"
cp "$HERE"/data/*.json "$DEST/data/"
cp "$TMP/sha_a.txt" "$DEST/SHA256-w21-build1.txt"
cp "$TMP/sha_b.txt" "$DEST/SHA256-w21-build2.txt"
cat "$TMP/sha_a.txt"
rm -rf "$TMP"
