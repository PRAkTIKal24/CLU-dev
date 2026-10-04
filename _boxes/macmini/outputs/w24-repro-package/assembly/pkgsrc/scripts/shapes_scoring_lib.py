"""shapes scoring lib."""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes_nclu_lib import log


def score_corners_full(obj, corners, grid, judge, bitmap, tag):
    from nclu.experiments import factor_wells as fw
    from nclu.experiments import exp_gen_read as R

    done, t0 = ({}, time.time())
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        feats, oor = R.arm_features(obj, vals, grid)
        acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
        tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
        val = fw.judge_valid(vals, tup, grid) & acc
        done[c["corner_id"]] = {
            "accept": acc,
            "valid": val,
            "tuples": np.where(acc[:, None], tup, -1),
            "n": int(vals.shape[0]),
            "family": c["family"],
        }
    log(f"    {tag:38s} {len(done)} corners in {time.time() - t0:.0f}s")
    return done
