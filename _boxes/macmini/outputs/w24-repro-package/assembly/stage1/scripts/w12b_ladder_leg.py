"""w12 hub/composition-ladder -- ONE LEG = one (rung, seed).

The k-way families are the HARNESS's (``chlu.data.gen_kway_ladder``).  This runner
builds no family of its own: it swaps the condition under
``exp_gen_factor_wells.load_condition`` and drives the SHIPPED arm/judge path unchanged.

Rungs:  k = 2, 3, 4, 5, 6  (the ladder) and ``shipped`` (the value-hold-out condition on
disk, which carries row 3 and row 4 as they are published -- the reproduction reference).

Arms:
  store   single_design_K5             the w8 DowndatableStore.  K* UNPUBLISHED => K=5, DECLARED.
          pca_tensor                   the wave-5 subspace store.
          <store>__landscape_shuffled  charter C.2, POST-fit: atoms permuted, book kept.
          <store>__random_atoms        charter C.2 strong form: random atoms, book REFIT.
  learned rival_{mlp,lstm,transformer,cont_mlp,table}  @ the 524,288 B REFERENCE pin.

Every arm is scored on the rung's own corners with the harness's pre-declared estimand:
``exp_gen_factor_wells.per_corner_scores`` -> ``distinct_valid_novel_tuples``.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chlu
from pathlib import Path as _P
if _P(__file__).resolve().parents[1] not in _P(chlu.__file__).resolve().parents:
    raise AssertionError("the package resolves outside this checkout: %s" % chlu.__file__)
from chlu.data import gen_kway_ladder as L
from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
def make_condition(rung: str) -> dict:
    """The rung's condition, in ``load_condition``'s own shape."""
    base = fw.load_condition()          # the shipped value-hold-out condition
    if rung == "shipped":
        base["rung"] = "shipped"
        base["corners"] = fw.pair_subcorners(base) + fw.nway_corners(base)
        return base
    k = int(rung)
    grid = base["grid"]
    cfg = L.KwayLadderConfig()
    split, cells, corners = L.build_kway_split(grid, cfg, k)
    cell_json = [{"cell_id": c.cell_id,
                  "values": {kk: [float(x) for x in vv] for kk, vv in c.values.items()},
                  "meta": c.meta,
                  "n_rows_realised": int(c.rows(grid).size)} for c in cells]
    return {
        "grid": grid,
        "train_rows": np.asarray(split.train_rows, np.int64),
        "test_rows": np.asarray(split.test_rows, np.int64),
        "cells": {"cells": cell_json, "config": base["cells"]["config"]},
        "query_sets": base["query_sets"],
        "scope": base["scope"],
        "rung": "k%d" % k,
        "corners": [{"corner_id": c.cell_id,
                     "family": c.meta.get("family", "kway%d" % k),
                     "rows": np.asarray(c.rows(grid), np.int64),
                     "meta": c.meta} for c in corners],
    }


_COND = {}
def _patched_load_condition():
    return _COND["cond"]


def score_arm(obj: dict, cond: dict, judge, bitmap, seen_mask) -> list:
    """Every corner of the rung, through the shipped acceptance / validity / scoring path.

    ⭐ The corners are answered in ONE call rather than one call each. MEASURED
    (`logs/batch_check.log`): `exp_gen_read.arm_features` returns `IDENTICAL: True,
    max|diff| 0.0` batched vs per-corner, and the judge's tuples are `IDENTICAL: True`
    at the pinned `eval_batch = 256` vs 72-row fragments. ⛔ Nothing is approximated:
    the reads and the judge are row-wise, the offsets slice the same rows back into the
    same corners, and `per_corner_scores` is the shipped function, unchanged.
    """
    grid = cond["grid"]
    corners = cond["corners"]
    vals_cat, offs = [], []
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


