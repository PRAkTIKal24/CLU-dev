"""Per-request verdict matrix over EVERY held-out rows-3-4 TEST request (109 corners), one seed
at a time, for the six columns, through that seed's ONE hash-asserted frozen decoder.

GATE: for every corner and every column, the rendered n_valid, n_accepted and
distinct_valid_novel_tuples must equal the w20 score file (scores/rows34/seed<N>/<arm>.json).
Any mismatch stops the scan and nothing is written.

    python -m pw1_render_shapes.scan 0 [1 2]    # -> <OUT>/scan/verdicts-seed<N>.json
"""
from __future__ import annotations
import numpy as np
from . import common as C
def context(seed: int):
    C.setenv()
    from chlu.experiments import exp_gen_factor_wells as fw
    from w17_rendered_compositions.same_decoder import OneDecoder
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = seed
    cond = fw.load_condition()
    grid = cond["grid"]
    judge = fw.Judge(seed, log=print)
    od = OneDecoder.from_judge(judge, seed, C.WINNER_HASHES)
    objs, meta = C.load_columns(seed)
    return {"fw": fw, "cond": cond, "grid": grid, "judge": judge, "od": od, "objs": objs,
            "meta": meta, "bitmap": fw.value_presence_bitmap(grid, cond["train_rows"]),
            "corners": fw.pair_subcorners(cond) + fw.nway_corners(cond), "seed": seed}


def read_rows(ctx, rows, cols=C.ORDER, arm_tag=None):
    """rows -> (vals, {col: {imgs, stamps, valid, tuples, accept}})."""
    from chlu.experiments import exp_gen_read as R
    fw, grid, od = ctx["fw"], ctx["grid"], ctx["od"]
    vals = fw._request_values_from_rows(grid, np.asarray(rows, np.int64))
    res = {}
    for col in cols:
        obj = ctx["objs"][col]
        feats, oor = R.arm_features(obj, vals, grid)
        acc = fw.accept_rule(vals, oor, np.zeros(vals.shape[0], bool), ctx["bitmap"], grid)
        imgs, stamps = od.render(feats, arm=arm_tag or col)
        valid, tup = od.verdict(vals, acc, imgs, grid)
        res[col] = {"imgs": imgs, "stamps": stamps, "valid": np.asarray(valid, bool),
                    "tuples": np.asarray(tup, np.int64), "accept": np.asarray(acc, bool)}
    return vals, res


