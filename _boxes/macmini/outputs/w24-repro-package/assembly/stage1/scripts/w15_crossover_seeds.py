"""w15 spoke/crossover-seeds -- THE 14 MISSING LEGS OF THE w12 CROSSOVER.

w12's val sweep ran K in {5, 32} at seeds 0, 1, 2 and K in {8, 10, 12, 16, 20, 24, 28}
at seed 0 ALONE.  Every interior K therefore carried n = 1 and no seed band, and the
sentence the wave wants to say -- "the four-part total is U-shaped in K with its minimum
near K = 24" -- rested on a single draw of the feature cache, the partition and the
address-book fit.  This script runs the 14 missing (K, seed) cells: K in
{8, 10, 12, 16, 20, 24, 28} at seeds 1 and 2.

THE INSTRUMENT IS NOT REIMPLEMENTED.  `scripts/w12_val_sweep.py` is imported and its
`main()` is called verbatim; the ONLY thing this file changes is the directory the rows
are written to, because w12's artefacts are read-only to this spoke.  Same partition
(`ward_balanced_labels`), same rank rule (A = min(64, min_k d_k - 1)), same read
(`TensorAddress`, `R.READ_PINS`, 90/10, `pinned_basis`), same 24-corner val metric, same
four charter-C controls, same store-scoped exactness check.  A leg produced here is
therefore comparable line by line with the seed-0 leg w12 produced at the same K.

The val condition is the free selection condition (charter A.2).  Every number is
`SMOKE_NOT_A_CLAIM`.  ZERO scored runs are spent by this spoke.
"""
from __future__ import annotations
import os
import pathlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("C:/chlu/outputs/w15-crossover-seeds/val_sweep")
def _assert_own_checkout():
    """The imported package must be the one this script belongs to."""
    import chlu
    cf = pathlib.Path(chlu.__file__).resolve()
    root = pathlib.Path(__file__).resolve().parents[1]
    assert root in cf.parents, f"the package resolves outside {root}: {cf}"
    print(f"OK package={cf}", flush=True)


def main(argv=None) -> int:
    _assert_own_checkout()
    import w12_val_sweep as VS
    # The one and only divergence from w12: where the rows land.
    VS.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"rows -> {OUT}", flush=True)
    return VS.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
