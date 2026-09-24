#!/usr/bin/env bash
# Build the two task figures twice into fresh scratch dirs, prove byte-identity, then install
# ONLY fig_task_shapes.{pdf,png} and fig_task_cmazet.{pdf,png} into the shared draft's figures/
# dir (the only files this spoke owns there).  Provenance and the two SHA lists go to PROOF.
# Never touches figures/rendered/ or any other figure.
# usage: bash w19_task_figures/build.sh <assets_dir> <proof_dir> [dest]   (from the worktree root,
#        after `source /Users/batmac/clu/env_bm.sh`)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="${1:?assets dir}"
PROOF="${2:?proof dir}"
DEST="${3:-/Users/batmac/clu/outputs/w17-paper/figures}"
TMP="${TMPDIR:-/tmp}/w19-task-figures.$$"
mkdir -p "$TMP/a" "$TMP/b"
python -m w19_task_figures.make_task_figures --assets "$ASSETS" --out "$TMP/a"
python -m w19_task_figures.make_task_figures --assets "$ASSETS" --out "$TMP/b"
(cd "$TMP/a" && shasum -a 256 * | sort -k2) > "$TMP/sha_a.txt"
(cd "$TMP/b" && shasum -a 256 * | sort -k2) > "$TMP/sha_b.txt"
cat "$TMP/sha_a.txt"
if diff "$TMP/sha_a.txt" "$TMP/sha_b.txt"; then echo "REBUILD: BYTE-IDENTICAL"; else echo "REBUILD: DIFFERS"; exit 1; fi
mkdir -p "$DEST"
cp "$TMP"/a/fig_task_shapes.pdf "$TMP"/a/fig_task_shapes.png "$TMP"/a/fig_task_cmazet.pdf \
   "$TMP"/a/fig_task_cmazet.png "$DEST/"
mkdir -p "$PROOF"
cp "$TMP/a/PROVENANCE-task-figures.json" "$PROOF/"
cp "$TMP/sha_a.txt" "$PROOF/SHA256-build1.txt"
cp "$TMP/sha_b.txt" "$PROOF/SHA256-build2.txt"
rm -rf "$TMP"
echo "installed into $DEST"
