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
# w22 figures-redraw pass 2: step 2 is make_figures_w22.py (it imports make_figures_w21.py as a
# library and adds fig_headline_steps / fig_maze_rel_tuples; maze_six.py reads the six-seed CMAZE-T
# artefacts). A second destination (paper-visuals) receives the same bytes.
# Rows / rival sets / panel sets are DATA: data/maze_conditions.json, data/ladder_rival_sets.json,
# data/cmazet_panels.json. Adding one is an edit there plus a rebuild, not a code change.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/Users/batmac/clu/outputs/w17-paper/figures}"
DEST2="${2:-/Users/batmac/clu/outputs/paper-visuals}"
TMP="${TMPDIR:-/tmp}/w21-figures-build.$$"
PANELS=()
[ -f "$HERE/data/cmazet_items.json" ] && PANELS=(--panels "$HERE/data/cmazet_items.json")
mkdir -p "$TMP/a" "$TMP/b"
for d in a b; do
  python "$HERE/make_figures.py" --out "$TMP/$d"
  python "$HERE/make_figures_w22.py" --out "$TMP/$d" ${PANELS[@]+"${PANELS[@]}"} --merge "$TMP/$d/PROVENANCE.json"
done
(cd "$TMP/a" && shasum -a 256 * | sort -k2) > "$TMP/sha_a.txt"
(cd "$TMP/b" && shasum -a 256 * | sort -k2) > "$TMP/sha_b.txt"
if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
for D in "$DEST" "$DEST2"; do
  mkdir -p "$D" "$D/data"
  cp "$TMP"/a/fig_*.pdf "$TMP"/a/fig_*.png "$TMP/a/PROVENANCE.json" "$D/"
  cp "$HERE/make_figures.py" "$HERE/make_figures_w21.py" "$HERE/make_figures_w22.py" \
     "$HERE/maze_six.py" "$HERE/validate_palette.py" "$HERE/build.sh" "$D/"
  [ -f "$HERE/cmazet_panels.py" ] && cp "$HERE/cmazet_panels.py" "$D/"
  cp "$HERE"/data/*.json "$D/data/"
  cp "$TMP/sha_a.txt" "$D/SHA256-w22-build1.txt"
  cp "$TMP/sha_b.txt" "$D/SHA256-w22-build2.txt"
  # every file this build produced, re-hashed at its destination (other spokes' files are not ours)
  (cd "$D" && awk '{print $2}' "$TMP/sha_a.txt" | xargs shasum -a 256 | sort -k2) > "$TMP/sha_inst.txt"
  if diff "$TMP/sha_a.txt" "$TMP/sha_inst.txt"; then echo "INSTALLED BYTE-IDENTICAL: $D"
  else echo "INSTALL DIFFERS: $D"; exit 1; fi
done
cat "$TMP/sha_a.txt"
rm -rf "$TMP"
