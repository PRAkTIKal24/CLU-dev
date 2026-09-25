#!/usr/bin/env bash
# Build the figures twice into scratch dirs, prove byte-identity, then install into two
# destinations (the draft's figures/ dir and paper-visuals/; never figures/rendered/).
#
# PW-1 item 4: this script runs in either layout (figroot.py):
#   repo   -- <CHLU repo>/w21_figures/build.sh; the maze engine is the repo;
#   folder -- IN PLACE in outputs/paper-visuals/ (or w17-paper/figures/, or any copy of them):
#             every script sits next to this file and the maze engine is the folder's engine/.
#   usage:  bash build.sh [dest1] [dest2]
#           defaults: dest1 = ~/clu/outputs/w17-paper/figures, dest2 = ~/clu/outputs/paper-visuals
#   External inputs (read-only): the evidence tree $CHLU_EVIDENCE (default ~/clu/evidence/outputs)
#   and the project venv's python (numpy, matplotlib 3.10.8, jax/equinox for the engine's imports).
#
# Steps, per build (a and b):
#   1. make_figures.py      -- the w17 figures (fig_encoder_adapt; the others are overwritten by 2)
#   2. make_figures_w22.py  -- fig_crossover, fig_ladder, fig_maze_conditions, fig_deletion and the
#                              fig_cmazet_* panels (data/cmazet_items*.json, data/cmazet_panels.json)
#   3. make_task_figures.py --only cmazet -- fig_task_cmazet (fig_task_shapes is never built here),
#      then verify_glyph.py (task-figure agent == result-panel agent) and the provenance merge.
# fig_headline_steps is no longer built (no slot; retired-pw1/, PW-1 item 4).
# Rows / baseline sets / panel sets are DATA: data/maze_conditions.json, data/ladder_rival_sets.json,
# data/cmazet_panels.json. Adding one is an edit there plus a rebuild, not a code change.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/Users/batmac/clu/outputs/w17-paper/figures}"
DEST2="${2:-/Users/batmac/clu/outputs/paper-visuals}"
PY="${PYTHON:-python}"
LAYOUT="$("$PY" "$HERE/figroot.py" layout)"
CODE="$("$PY" "$HERE/figroot.py" code)"
if [ "$LAYOUT" = repo ]; then
  REPO="$(dirname "$HERE")"
  TASKFIG="$REPO/w19_task_figures/make_task_figures.py"
  VERIFY="$REPO/w19_task_figures/verify_glyph.py"
else
  TASKFIG="$HERE/make_task_figures.py"
  VERIFY="$HERE/verify_glyph.py"
fi
MERGE="$HERE/tools/merge_task_prov.py"
# the engine root first, so no other checkout on sys.path (e.g. the venv's editable chlu) is used
export PYTHONPATH="$HERE:$CODE/scripts:$CODE"
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$HERE/figroot.py" check
# the assets dir is read only by fig_task_shapes, which this build never draws (--only cmazet)
ASSETS="${W19_TASK_ASSETS:-/Users/batmac/clu/outputs/w19-task-figures/assets}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/w21-figures-build.XXXXXX")"
PANELS=()
[ -f "$HERE/data/cmazet_items.json" ] && PANELS=(--panels "$HERE/data/cmazet_items.json")
mkdir -p "$TMP/a" "$TMP/b"
for d in a b; do
  "$PY" "$HERE/make_figures.py" --out "$TMP/$d"
  "$PY" "$HERE/make_figures_w22.py" --out "$TMP/$d" ${PANELS[@]+"${PANELS[@]}"} --merge "$TMP/$d/PROVENANCE.json"
  # w22 figures3 (AMENDMENT-1): fig_task_cmazet, drawn with the panels' own draw_state; the glyph
  # identity gate (task figure agent == result-panel agent) must pass before anything is installed
  "$PY" "$TASKFIG" --only cmazet --assets "$ASSETS" --out "$TMP/task_$d"
  mv "$TMP/task_$d/fig_task_cmazet.pdf" "$TMP/task_$d/fig_task_cmazet.png" "$TMP/$d/"
  "$PY" "$VERIFY" "$TMP/$d" "$TMP/$d" "$TMP/$d/GLYPH-IDENTITY.json" > /dev/null
  "$PY" "$MERGE" "$TMP/$d/PROVENANCE.json" \
      "$TMP/task_$d/PROVENANCE-task-figures.json" "$TMP/$d/GLYPH-IDENTITY.json"
  rm -r "$TMP/task_$d"
done
# manifests sorted in the C locale, so a fresh copy under any locale writes the same bytes
(cd "$TMP/a" && shasum -a 256 * | LC_ALL=C sort -k2) > "$TMP/sha_a.txt"
(cd "$TMP/b" && shasum -a 256 * | LC_ALL=C sort -k2) > "$TMP/sha_b.txt"
if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
# the code a folder needs to rebuild itself (flat; figroot.py resolves the layout)
SCRIPTS=("$HERE/make_figures.py" "$HERE/make_figures_w21.py" "$HERE/make_figures_w22.py"
         "$HERE/maze_six.py" "$HERE/validate_palette.py" "$HERE/build.sh" "$HERE/figroot.py"
         "$HERE/cmazet_panels.py" "$HERE/cmazet_door_search.py" "$HERE/cmazet_extract_rel.py"
         "$HERE/cmazet_ambiguity.py" "$TASKFIG" "$VERIFY")
for D in "$DEST" "$DEST2"; do
  mkdir -p "$D" "$D/data" "$D/tools"
  cp "$TMP"/a/fig_*.pdf "$TMP"/a/fig_*.png "$TMP/a/PROVENANCE.json" "$TMP/a/GLYPH-IDENTITY.json" "$D/"
  if [ "$(cd "$D" && pwd -P)" != "$(cd "$HERE" && pwd -P)" ]; then   # in place: code is already here
    cp "${SCRIPTS[@]}" "$D/"
    cp "$MERGE" "$HERE/tools/vendor_engine.py" "$D/tools/"
    cp "$HERE"/data/*.json "$D/data/"
    "$PY" "$HERE/tools/vendor_engine.py" "$CODE" "$D"
  fi
  cp "$TMP/sha_a.txt" "$D/SHA256-w22-build1.txt"
  cp "$TMP/sha_b.txt" "$D/SHA256-w22-build2.txt"
  # every file this build produced, re-hashed at its destination (other spokes' files are not ours)
  (cd "$D" && awk '{print $2}' "$TMP/sha_a.txt" | xargs shasum -a 256 | LC_ALL=C sort -k2) > "$TMP/sha_inst.txt"
  if diff "$TMP/sha_a.txt" "$TMP/sha_inst.txt"; then echo "INSTALLED BYTE-IDENTICAL: $D"
  else echo "INSTALL DIFFERS: $D"; exit 1; fi
done
cat "$TMP/sha_a.txt"
rm -rf "$TMP"
