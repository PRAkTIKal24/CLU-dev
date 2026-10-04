"""PREREG-0 s4: extend the row-2 request set with the PINNED builder at the PINNED seed,
and ASSERT the first 512 records are byte-identical to the frozen file before using it.

If the assertion fails the extension is discarded and the hub falls back to the frozen
512, and says so. Nothing about the builder, the seed or the split is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hub_common as hc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()

    from chlu.data.gen_splits import GenSplit
    from chlu.data.gen_value_holdout import ValueHoldoutConfig, build_row2_unseen

    grid, _, _ = hc.grid_from_h5()
    z = np.load(hc.VH / "factor_index.npz", allow_pickle=True)
    train_rows = np.asarray(z["train_rows"], np.int64)
    test_rows = np.asarray(z["test_rows"], np.int64)
    in_test = np.zeros(grid.n_rows, bool)
    in_test[test_rows] = True

    split = GenSplit(
        dataset="shapes3d", condition="value_holdout",
        variant="value_cells_plus_combination_cells", grid=grid,
        train_rows=train_rows, test_rows=test_rows,
        test_predicate=lambda v: in_test[grid.rows_of(
            np.stack([np.searchsorted(grid.values[k], v[:, k]) for k in range(6)], 1))],
    )

    frozen = json.load(open(hc.VH / "row2_unseen.json", encoding="utf-8"))
    cfg = ValueHoldoutConfig(row2_n_requests=int(a.n), row2_seed=int(frozen["seed"]))
    built = build_row2_unseen(grid, split, cfg)

    # the assertion PREREG-0 s4 requires, on the RECORDS themselves
    fa = json.dumps(frozen["records"], sort_keys=True)
    fb = json.dumps(built["records"][:len(frozen["records"])], sort_keys=True)
    identical = fa == fb
    print(f"built {built['n_requests']} requests at seed {cfg.row2_seed}")
    print(f"first {len(frozen['records'])} records byte-identical to the frozen file: "
          f"{identical}")
    if not identical:
        n_diff = sum(1 for x, y in zip(frozen["records"], built["records"])
                     if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True))
        print(f"  {n_diff} of {len(frozen['records'])} records differ -> "
              "EXTENSION DISCARDED, falling back to the frozen 512 (PREREG-0 s4)")
        return 1

    built["superset_assertion"] = {
        "frozen_file": str(hc.VH / "row2_unseen.json"),
        "n_frozen": len(frozen["records"]),
        "first_n_byte_identical": True,
        "note": "the pinned builder's RNG stream is fixed by one leading permutation "
                "call, so n_want only sets the break; asserted, not assumed",
    }
    out = hc.OUT / "runs" / f"row2_unseen_n{a.n}.json"
    hc.jdump(built, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
