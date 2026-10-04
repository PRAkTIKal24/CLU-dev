#!/usr/bin/env bash
# w6 hub/potentials -- REPLICATION (charter section A.5).
#
# Re-runs the hub's configuration from its PREREG file on a CLEAN CHECKOUT of the hub's
# branch, taken from the bundle, in a directory that shares nothing with the worktree
# except the read-only cache and the PREREG.  Unreplicated = not a result.
#
#   scripts/w6_potentials_replicate.sh <bundle> <clean-dir> <out-dir> <prereg> [seed] [corners]
set -euo pipefail
BUNDLE="${1:?bundle path}"
CLEAN="${2:?clean checkout dir}"
OUT="${3:?output dir}"
PREREG="${4:?prereg file}"
SEED="${5:-0}"
NCORNERS="${6:-12}"

rm -rf "$CLEAN"
git clone -q --branch w6/potentials "$BUNDLE" "$CLEAN"
cd "$CLEAN"
echo "[repl] clean checkout at $CLEAN"
echo "[repl] HEAD = $(git rev-parse HEAD)"
echo "[repl] tree clean: $(git status --porcelain | wc -l) modified files"

# ⛔ The clean checkout must be the thing that runs: CHLU_ROOT and PYTHONPATH point HERE,
# not at the worktree.  The assertion below is the whole point of the exercise.
# PYTHONPATH must carry a WINDOWS path.  Python on Windows cannot use an MSYS
# "/c/..." path, and the failure is silent until an import: running `python -c`
# from inside the checkout still works (the CWD is on sys.path) while running a
# script in scripts/ does not (sys.path[0] is the SCRIPT directory).  Convert.
CLEAN_WIN="$(cygpath -w "$CLEAN")"
export CHLU_ROOT="$CLEAN_WIN"
export PYTHONPATH="$CLEAN_WIN;C:\Users\PratikJawahar\chlu\chlu-bundle-v2\.venv\Lib\site-packages"
export CHLU_GEN_BRIDGE_ROOT="C:\\chlu\\outputs\\spoke-a-rivals\\cache_generation2"
export CHLU_GEN_CLASSIFIER_ROOT="C:\\Users\\PratikJawahar\\chlu\\chlu-bundle-v2\\.claude\\outputs\\gen-external-classifier"
export CHLU_HUB_OUTPUTS="C:/chlu/outputs"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

# Run the assertion the SAME way the leg runs -- from a file in scripts/, so sys.path[0]
# is that directory and the CWD cannot mask a broken PYTHONPATH.
cat > "$CLEAN/scripts/_repl_assert.py" <<'PY'
import chlu, pathlib, os, sys
here = pathlib.Path(os.environ['CHLU_ROOT']).resolve()
got = pathlib.Path(chlu.__file__).resolve()
assert here in got.parents, f'chlu resolved to {got}, NOT inside the clean checkout {here}'
print(f'[repl] chlu resolves inside the clean checkout: {got}')
PY
python "$CLEAN/scripts/_repl_assert.py"
rm -f "$CLEAN/scripts/_repl_assert.py"

python -u scripts/w6_potentials_leg.py \
  --condition test --seed "$SEED" --out "$OUT" \
  --corners nway --max-corners "$NCORNERS" --aux-corners 0 \
  --arms field --prereg "$PREREG"

echo "[repl] done -- compare $OUT/results.json against the primary leg's"
