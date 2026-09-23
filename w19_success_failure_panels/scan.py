"""Per-request verdict matrix over EVERY held-out T7 test corner at K* = 32, seeds 0, 1, 2.

For each request: which of the six columns (CLU, 4 rivals, lookup floor) the judge marks
valid, through that seed's ONE hash-asserted frozen decoder.  Every corner's per-arm
n_valid must reproduce the scored.json it came from, or the scan stops.

    python -m w19_success_failure_panels.scan            # writes <SPOKE_OUT>/scan/verdicts.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .common import SPOKE_OUT, w17


def scan(seeds=(0, 1, 2)) -> dict:
    RR = w17()
    RR._setenv(True)
    from w17_rendered_compositions import selection as S
    from chlu.experiments import exp_gen_factor_wells as fw
    order = [a for a, _, _ in S.ARMS]
    out = {"schema": "w19-success-failure-panels/scan/1", "arms": order, "K_star": 32,
           "sources": {str(s): str(S.EVIDENCE / S.SCORED[s]) for s in seeds}, "seeds": {}}
    for seed in seeds:
        fw.PINS["condition"] = "test"
        fw.PINS["seed"] = seed
        cond = fw.load_condition()
        grid = cond["grid"]
        bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
        judge = fw.Judge(seed, log=print)
        od = RR.OneDecoder.from_judge(judge, seed, S.EVIDENCE / S.WINNER_HASHES)
        objs, _ = RR._load_arms(seed, grid, None)
        pc = S.per_corner(seed)
        corners = fw.pair_subcorners(cond) + fw.nway_corners(cond)
        reqs, mism = [], []
        for c in corners:
            _vals, res = RR._read_corner(objs, od, c, grid, bitmap)
            V = np.stack([res[a]["valid"] for a in order], 1)
            got = {a: int(V[:, j].sum()) for j, a in enumerate(order)}
            want = {a: int(pc[a][c["corner_id"]]["n_valid"]) for a in order}
            if got != want:
                mism.append([c["corner_id"], got, want])
            for i in range(V.shape[0]):
                reqs.append([c["corner_id"], c["family"], int(c["rows"][i]), i,
                             "".join("1" if x else "0" for x in V[i])])
        if mism:
            raise SystemExit(f"seed {seed}: {len(mism)} corner(s) do not reproduce scored.json: {mism[:3]}")
        out["seeds"][str(seed)] = {"decoder_param_hash": od.hash,
                                   "classifier_param_hash": od.classifier_hash,
                                   "n_corners": len(corners), "n_requests": len(reqs),
                                   "reproduction_mismatches": 0,
                                   "requests_columns": ["corner_id", "family", "dataset_row",
                                                        "index_in_corner", "valid_bits(arms order)"],
                                   "requests": reqs}
        print(f"seed {seed}: {len(corners)} corners, {len(reqs)} requests, reproduced", flush=True)
    return out


if __name__ == "__main__":
    d = SPOKE_OUT / "scan"
    d.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in sys.argv[1:]) or (0, 1, 2)
    res = scan(seeds)
    (d / "verdicts.json").write_text(json.dumps(res, indent=0), encoding="utf-8")
    print("wrote", d / "verdicts.json")
