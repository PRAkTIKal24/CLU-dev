"""check_00 -- environment assertion. MUST pass before any number below is believed.

The shared site-packages carries an editable .pth to the Head's checkout.  If that .pth
wins, `chlu` resolves outside this worktree and every number produced here is void.
Run:  OMP_NUM_THREADS=2 python checks/check_00_env.py
"""
import os
import sys

import numpy as np

WORKTREE = os.environ.get("W12_WORKTREE", r"C:\chlu\worktrees\CHLU-w12-clue-theorist")


def main() -> int:
    import chlu
    f = os.path.abspath(chlu.__file__)
    root = os.path.abspath(WORKTREE)
    ok = os.path.normcase(f).startswith(os.path.normcase(root))
    print(f"chlu.__file__      = {f}")
    print(f"expected worktree  = {root}")
    print(f"INSIDE_WORKTREE    = {ok}")
    print(f"python             = {sys.version.split()[0]}")
    print(f"numpy              = {np.__version__}")
    try:
        import jax
        print(f"jax                = {jax.__version__}")
    except Exception as e:                                    # pragma: no cover
        print(f"jax                = UNAVAILABLE ({e})")
    assert ok, "chlu resolved OUTSIDE this worktree -- numbers are void"
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
