"""w12 hub/composition-ladder -- INSTRUMENT CHECK, run BEFORE any arm.

Answers, on disk, the two non-negotiables:
  * does the k=2 rung reproduce ROW 3 (corner for corner, and on the same split)?
  * does the k=6 rung reproduce ROW 4 (cell for cell, and on the same split)?
No features are read; this is index arithmetic only.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

import chlu
assert "CHLU-w12-composition-ladder" in chlu.__file__, chlu.__file__

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.data import gen_kway_ladder as L


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rep = {"chlu_file": chlu.__file__}
    grid, z = fw.build_grid()
    cond = fw.load_condition()
    cfg = L.KwayLadderConfig()

    shipped_train = np.asarray(cond["train_rows"], np.int64)
    shipped_test = np.asarray(cond["test_rows"], np.int64)
    rep["shipped_condition"] = {
        "n_train": int(shipped_train.size), "n_test": int(shipped_test.size),
        "scope": cond["scope"],
        "n_cells": len(cond["cells"]["cells"]),
        "families": {f: sum(1 for c in cond["cells"]["cells"] if c["meta"]["family"] == f)
                     for f in sorted({c["meta"]["family"] for c in cond["cells"]["cells"]})},
    }

    # row 3 and row 4 corners as the SHIPPED condition defines them
    r3 = fw.pair_subcorners(cond)
    r4 = fw.nway_corners(cond)
    rep["shipped_row3"] = {"n_corners": len(r3),
                           "rows_per_corner": sorted({int(c["rows"].size) for c in r3}),
                           "ids_head": [c["corner_id"] for c in r3[:3]],
                           "ids_tail": [c["corner_id"] for c in r3[-2:]]}
    rep["shipped_row4"] = {"n_corners": len(r4),
                           "rows_per_corner": sorted({int(c["rows"].size) for c in r4}),
                           "ids_head": [c["corner_id"] for c in r4[:3]],
                           "ids_tail": [c["corner_id"] for c in r4[-2:]]}

    rungs = {}
    for k in (2, 3, 4, 5, 6):
        split, cells, corners = L.build_kway_split(grid, cfg, k)
        tr = np.asarray(split.train_rows, np.int64)
        te = np.asarray(split.test_rows, np.int64)
        crows = [np.asarray(c.rows(grid), np.int64) for c in corners]
        cat = np.concatenate(crows)
        d = {
            "k": k,
            "n_cells_total": len(cells),
            "n_corners": len(corners),
            "rows_per_corner": sorted({int(r.size) for r in crows}),
            "n_train": int(tr.size), "n_test": int(te.size),
            "corner_rows_total": int(cat.size),
            "pairwise_disjoint": bool(np.unique(cat).size == cat.size),
            "all_corner_rows_in_test": bool(np.setdiff1d(cat, te).size == 0),
            "corner_ids_head": [c.cell_id for c in corners[:3]],
            "corner_ids_tail": [c.cell_id for c in corners[-2:]],
            # how this rung's split relates to the SHIPPED one
            "train_equals_shipped": bool(tr.size == shipped_train.size
                                         and np.array_equal(tr, shipped_train)),
            "test_equals_shipped": bool(te.size == shipped_test.size
                                        and np.array_equal(te, shipped_test)),
            "train_shared_with_shipped": int(np.intersect1d(tr, shipped_train).size),
        }
        rungs[f"k{k}"] = d
        print(f"[rung k={k}] cells={d['n_cells_total']} corners={d['n_corners']} "
              f"rows/corner={d['rows_per_corner']} train={d['n_train']} test={d['n_test']} "
              f"disjoint={d['pairwise_disjoint']} in_test={d['all_corner_rows_in_test']}",
              flush=True)
    rep["rungs"] = rungs

    # ---- E2: k=2 corners vs the SHIPPED row 3 corners, row set for row set ----
    _, _, c2 = L.build_kway_split(grid, cfg, 2)
    ids2 = [c.cell_id for c in c2]
    ids3 = [c["corner_id"] for c in r3]
    same_ids = ids2 == ids3
    same_rows = same_ids and all(
        np.array_equal(np.asarray(a.rows(grid), np.int64), np.asarray(b["rows"], np.int64))
        for a, b in zip(c2, r3))
    rep["E2_k2_vs_shipped_row3"] = {
        "corner_ids_identical": bool(same_ids),
        "corner_row_sets_identical": bool(same_rows),
        "n_compared": len(ids3),
        "reference": "exp_gen_factor_wells.pair_subcorners on the SHIPPED value_holdout condition",
    }
    # the harness's own assertions, re-run here rather than quoted
    rep["harness_assert_k2_matches_row3"] = L.assert_k2_matches_row3(grid, cfg)
    rep["harness_assert_k6_matches_row4"] = L.assert_k6_matches_row4(grid, cfg)
    rep["harness_assert_row3_widths_match_pins"] = L.assert_row3_widths_match_pins()

    # ---- E1': k=6 corners vs the SHIPPED row 4 corners (the shipped 49) ----
    _, _, c6 = L.build_kway_split(grid, cfg, 6)
    ids6 = [c.cell_id for c in c6]
    ids4 = [c["corner_id"] for c in r4]
    pref = ids6[:len(ids4)] == ids4
    pref_rows = pref and all(
        np.array_equal(np.asarray(a.rows(grid), np.int64), np.asarray(b["rows"], np.int64))
        for a, b in zip(c6[:len(ids4)], r4))
    rep["E1_k6_vs_shipped_row4"] = {
        "shipped_row4_is_prefix_of_k6_ids": bool(pref),
        "shipped_row4_is_prefix_of_k6_rows": bool(pref_rows),
        "n_shipped_row4": len(ids4), "n_k6": len(ids6),
        "reference": "exp_gen_factor_wells.nway_corners on the SHIPPED value_holdout condition",
    }

    p = out / "INSTRUMENT.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[sys.argv.index("--out") + 1])))
