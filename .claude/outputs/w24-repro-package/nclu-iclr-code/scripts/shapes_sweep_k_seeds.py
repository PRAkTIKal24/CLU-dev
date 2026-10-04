"""shapes sweep k seeds."""

from __future__ import annotations
from nclu.paths import run_dir
import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(str(run_dir("shapes3d/k_sweep_seeds/val_sweep")))


def _assert_own_checkout():
    """The imported package must be the one this script belongs to."""
    import nclu

    cf = pathlib.Path(nclu.__file__).resolve()
    root = pathlib.Path(__file__).resolve().parents[1]
    assert root in cf.parents, f"the package resolves outside {root}: {cf}"
    print(f"OK package={cf}", flush=True)


def main(argv=None) -> int:
    _assert_own_checkout()
    import shapes_sweep_k as VS

    VS.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"rows -> {OUT}", flush=True)
    return VS.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
