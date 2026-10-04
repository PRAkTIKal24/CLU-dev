"""shapes ladder lib."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nclu
from pathlib import Path as _P

if _P(__file__).resolve().parents[1] not in _P(nclu.__file__).resolve().parents:
    raise AssertionError("the package resolves outside this checkout: %s" % nclu.__file__)
from nclu.data import gen_kway_ladder as L
from nclu.experiments import factor_wells as fw
from nclu.experiments import exp_gen_read as R


def make_condition(rung: str) -> dict:
    """The rung's condition, in ``load_condition``'s own shape."""
    base = fw.load_condition()
    if rung == "shipped":
        base["rung"] = "shipped"
        base["corners"] = fw.pair_subcorners(base) + fw.nway_corners(base)
        return base
    k = int(rung)
    grid = base["grid"]
    cfg = L.KwayLadderConfig()
    split, cells, corners = L.build_kway_split(grid, cfg, k)
    cell_json = [
        {
            "cell_id": c.cell_id,
            "values": {kk: [float(x) for x in vv] for kk, vv in c.values.items()},
            "meta": c.meta,
            "n_rows_realised": int(c.rows(grid).size),
        }
        for c in cells
    ]
    return {
        "grid": grid,
        "train_rows": np.asarray(split.train_rows, np.int64),
        "test_rows": np.asarray(split.test_rows, np.int64),
        "cells": {"cells": cell_json, "config": base["cells"]["config"]},
        "query_sets": base["query_sets"],
        "scope": base["scope"],
        "rung": "k%d" % k,
        "corners": [
            {
                "corner_id": c.cell_id,
                "family": c.meta.get("family", "kway%d" % k),
                "rows": np.asarray(c.rows(grid), np.int64),
                "meta": c.meta,
            }
            for c in corners
        ],
    }


_COND = {}


def _patched_load_condition():
    return _COND["cond"]


def score_arm(obj: dict, cond: dict, judge, bitmap, seen_mask) -> list:
    grid = cond["grid"]
    corners = cond["corners"]
    vals_cat, offs = ([], [])
    p = 0
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        vals_cat.append(vals)
        offs.append((p, p + vals.shape[0]))
        p += vals.shape[0]
    V = np.concatenate(vals_cat)
    feats, oor = R.arm_features(obj, V, grid)
    acc = fw.accept_rule(V, oor, np.zeros(V.shape[0], bool), bitmap, grid)
    tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
    valid = fw.judge_valid(V, tup, grid) & acc
    return fw.per_corner_scores(corners, V, tup, valid, acc, offs, grid, seen_mask)
