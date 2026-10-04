"""w12/kway-ladder-harness — run the KILL CHECK, then the ends, then emit LADDER-READY.

⛔ Order is load-bearing and is the order the PREREG declares:
  1. the kill check  (``ladder_capacity_report``)  — can stop the ladder
  2. E1 / E2         (the k=6 / k=2 end reproductions)
  3. the rungs       (built, split, row-level disjointness proved)

⛔ No scored run.  This script builds the instrument only.

Usage::

    python scripts/w12_kway_ladder.py --out DIR [--toy]

``--toy`` uses the Shapes3D-shaped offline grid from ``tests/test_gen_value_holdout.py``
(same cardinalities, same 0.1-spaced hue ring) instead of the 268 MB ``3dshapes.h5``,
so the arithmetic can be checked without the file.  ⛔ The shipped numbers are the
non-``--toy`` ones and every artifact records which grid it used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from chlu.data import gen_kway_ladder as K
from chlu.data.gen_splits import FactorGrid

TOY_NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")


def toy_grid() -> FactorGrid:
    """The offline Shapes3D-shaped grid (tests/test_gen_value_holdout.py fixture)."""
    values = (
        np.arange(10) / 10.0,
        np.arange(10) / 10.0,
        np.arange(10) / 10.0,
        np.linspace(0.75, 1.25, 8),
        np.arange(4, dtype=float),
        np.linspace(-30.0, 30.0, 15),
    )
    return FactorGrid(names=TOY_NAMES, values=values, dataset="toy3d")


def real_grid() -> FactorGrid:
    from chlu.data import shapes3d as s3d

    labels = s3d.load_labels(s3d.shapes3d_root())
    grid = s3d.build_grid(labels)
    n = grid.verify_bijection(labels)
    print(f"  grid bijection verified on {n:,} rows of the real 3dshapes.h5")
    return grid


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--toy", action="store_true")
    args = p.parse_args(argv)

    import chlu

    print(f"chlu.__file__ = {chlu.__file__}")
    assert "CHLU-w12-kway-ladder-harness" in chlu.__file__, (
        "⛔ chlu does NOT resolve inside this worktree — every number would be void"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("grid:")
    grid = toy_grid() if args.toy else real_grid()
    print(f"  {grid.dataset}  {dict(zip(grid.names, grid.cardinalities))}  "
          f"n_rows={grid.n_rows:,}")

    cfg = K.KwayLadderConfig()
    suffix = "-toy" if args.toy else ""

    # ---- 1. THE KILL CHECK, standalone and on disk BEFORE anything is built ----
    cap = K.ladder_capacity_report(grid, cfg)
    kill_path = out / f"KILL-CHECK{suffix}.json"
    kill_path.write_text(json.dumps(cap, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nKILL CHECK -> {kill_path}")
    print(f"  coordinate options: {cap['coordinate_option_counts']}")
    for k, r in cap["rungs"].items():
        print(
            f"  k={k}  pinned={r['pinned_factors']}\n"
            f"        boxes={r['n_cells_capacity']} x tiles={r['corners_per_cell']}"
            f" = {r['max_corners']} corners   bar={r['bar']}   {r['status']}"
            f"   rows/cell={r['rows_per_cell']:,}"
            f"   held-out(full sweep)={r['held_out_rows_full_sweep']:,}"
            f" ({r['held_out_fraction_full_sweep']:.4%})"
        )
    dead = [k for k, r in cap["rungs"].items() if r["status"] == "NOT_RUN"]
    if dead:
        print(f"  ⛔ NOT-RUN rungs: {dead}")

    # ---- 2 + 3. ends and rungs, written by the module ----
    print()
    payload = K.emit_ladder_ready(
        grid, out, cfg, filename=f"LADDER-READY{suffix}.json"
    )
    print(f"\nelapsed {time.time() - t0:.1f}s")
    print(json.dumps(
        {kk: {"status": v.get("status"), "n_cells": v.get("n_cells"),
              "n_corners": v.get("n_corners"),
              "held_out_fraction_total": v.get("held_out_fraction_total")}
         for kk, v in payload["families"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
