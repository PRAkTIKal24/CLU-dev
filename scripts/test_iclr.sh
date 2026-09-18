#!/usr/bin/env bash
# scripts/test_iclr.sh — run ONLY the ICLR test set (tests/ICLR.txt).
# ⭐ Deselection, ⛔ not deletion. After ICLR: `pytest -q tests/` and retire this.
# ⚠ Always with CHLU_GEN_DATA_ROOT set, and always report passed/skipped/failed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mapfile -t FILES < <(grep -vE '^\s*#|^\s*$' "$HERE/tests/ICLR.txt" | sed "s|^|$HERE/tests/|")
echo "[iclr-tests] ${#FILES[@]} files (of $(ls "$HERE"/tests/*.py | wc -l | tr -d ' ') present)"
exec "${PYTHON:-python}" -m pytest -q -rs "${FILES[@]}" "$@"
