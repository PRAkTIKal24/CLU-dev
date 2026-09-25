"""Per-request verdict matrix over EVERY held-out rows-3-4 TEST request (109 corners), one seed
at a time, for the six columns, through that seed's ONE hash-asserted frozen decoder.

GATE: for every corner and every column, the rendered n_valid, n_accepted and
distinct_valid_novel_tuples must equal the w20 score file (scores/rows34/seed<N>/<arm>.json).
Any mismatch stops the scan and nothing is written.

    python -m pw1_render_shapes.scan 0 [1 2]    # -> <OUT>/scan/verdicts-seed<N>.json
"""
from __future__ import annotations

import json
import sys

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


def scan(seed: int) -> dict:
    ctx = context(seed)
    fw, grid, cond = ctx["fw"], ctx["grid"], ctx["cond"]
    stages = ctx["meta"]["carried"]
    seen = fw.seen_tuple_mask(grid, cond["train_rows"])
    want = {}
    for col in C.ORDER:
        p = C.score_file(seed, col, stages)
        want[col] = ({c["corner_id"]: c for c in json.loads(p.read_text(encoding="utf-8"))["per_corner"]}, str(p))
    reqs, mism, n_checked = [], [], 0
    for c in ctx["corners"]:
        rows = np.asarray(c["rows"], np.int64)
        _vals, res = read_rows(ctx, rows)
        fidx = grid.factor_indices_of(rows)
        for col in C.ORDER:
            r = res[col]
            got = fw.per_corner_scores([c], None, r["tuples"], r["valid"], r["accept"],
                                       [(0, rows.size)], grid, seen)[0]
            w = want[col][0][c["corner_id"]]
            for k in ("n_requests", "n_accepted", "n_valid", "distinct_valid_novel_tuples"):
                n_checked += 1
                if int(got[k]) != int(w[k]):
                    mism.append([c["corner_id"], col, k, int(got[k]), int(w[k])])
        for i in range(rows.size):
            reqs.append([c["corner_id"], c["family"], int(rows[i]), i,
                         [int(x) for x in fidx[i]],
                         "".join("1" if res[col]["valid"][i] else "0" for col in C.ORDER),
                         [[int(x) for x in res[col]["tuples"][i]] for col in C.ORDER]])
        if mism:
            raise SystemExit(f"REPRODUCTION FAILED seed {seed}: {mism[:5]} -- nothing written")
    return {"schema": "pw1-render-shapes/scan/1", "seed": seed, "columns": list(C.ORDER),
            "carried": stages, "objects": ctx["meta"]["objects"],
            "decoder_param_hash": ctx["od"].hash, "classifier_param_hash": ctx["od"].classifier_hash,
            "reproduction": {"against": {col: want[col][1] for col in C.ORDER},
                             "fields": ["n_requests", "n_accepted", "n_valid",
                                        "distinct_valid_novel_tuples"],
                             "n_comparisons": n_checked, "mismatches": 0},
            "factor_names": list(grid.names),
            "requests_columns": ["corner_id", "family", "dataset_row", "index_in_corner",
                                 "requested_factor_indices", "valid_bits(columns order)",
                                 "judge_readback_per_column(columns order)"],
            "n_corners": len(ctx["corners"]), "n_requests": len(reqs), "requests": reqs}


if __name__ == "__main__":
    d = C.OUT / "scan"
    d.mkdir(parents=True, exist_ok=True)
    print(C.verify_manifest(), flush=True)
    for s in [int(x) for x in sys.argv[1:]] or [0]:
        res = scan(s)
        (d / f"verdicts-seed{s}.json").write_text(json.dumps(res, indent=0), encoding="utf-8")
        print(f"seed {s}: {res['n_corners']} corners, {res['n_requests']} requests, "
              f"{res['reproduction']['n_comparisons']} comparisons, 0 mismatches", flush=True)
