#!/usr/bin/env bash
# Install the figure build's CODE (never a figure) from a CHLU repo into a flat figure folder, so
# that `bash build.sh` runs in place there: the same file set build.sh copies on install, plus the
# engine snapshot (tools/vendor_engine.py). Prints the sha256 of every file it wrote.
# usage: bash pw1_visuals_tooling/install_scripts.sh <CHLU repo root> <folder>
set -euo pipefail
REPO="$(cd "$1" && pwd)"; D="$2"
W="$REPO/w21_figures"; T="$REPO/w19_task_figures"
FILES=("$W/make_figures.py" "$W/make_figures_w21.py" "$W/make_figures_w22.py" "$W/maze_six.py"
       "$W/validate_palette.py" "$W/build.sh" "$W/figroot.py" "$W/cmazet_panels.py"
       "$W/cmazet_door_search.py" "$W/cmazet_extract_rel.py" "$W/cmazet_ambiguity.py"
       "$T/make_task_figures.py" "$T/verify_glyph.py")
mkdir -p "$D/tools" "$D/data"
cp "${FILES[@]}" "$D/"
cp "$W/tools/merge_task_prov.py" "$W/tools/vendor_engine.py" "$D/tools/"
cp "$W"/data/*.json "$D/data/"
python "$W/tools/vendor_engine.py" "$REPO" "$D"
(cd "$D" && for f in "${FILES[@]}"; do shasum -a 256 "$(basename "$f")"; done
 shasum -a 256 tools/merge_task_prov.py tools/vendor_engine.py data/*.json engine/ENGINE.json)
