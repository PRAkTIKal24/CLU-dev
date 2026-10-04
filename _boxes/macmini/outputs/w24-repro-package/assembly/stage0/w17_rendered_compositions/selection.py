"""Which corners each panel set draws from -- decided from the SCORED per-corner verdicts only.

Nothing here is hand-picked.  Each set has a rule, the rule is applied to the per-corner
`n_valid` counts the T7 scored leg wrote (K* = 32), and the output names the file every
count came from.  The rules:

  set 1  CLU composes, rivals fail
         corners where the CLU's n_valid == n_requests AND every rival and the lookup
         floor have n_valid == 0.  Every request in such a corner is, by the scored counts
         alone, a CLU success and a failure for every other arm.  One 2-way (pair) corner
         -- preferring a large-scale, high-orientation sub-corner ("-hh", the task's
         "large, tilted") -- and one n-way (full six-factor box) corner.
  set 2  ALL arms succeed
         corners where EVERY arm, the continuous MLP included, has n_valid > 0.  That is
         necessary, not sufficient: whether one request is valid for every arm at once is
         only decidable on the rendered requests, so `render` decides it and DECLARES the
         set if no such held-out request exists.
  set 3  ALL arms fail
         corners where every rival and the floor have n_valid == 0 and the CLU has
         n_valid < n_requests, lowest CLU fraction first.  Every CLU failure in such a
         corner is an all-arm failure, so the set is guaranteed non-empty by the counts.

Seed 0 is the rendered seed: it is the seed whose CLU object (`run1_seed0_K32/arm.pkl`) the
deletion rows were run on, and one seed is what a picture can show.  Rates over n = 3 seeds
are the T7 table's job, not this figure's.
"""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path("/Users/batmac/clu/evidence/outputs")

#: The T7 scored legs at K* = 32, n = 3 seeds (w12b-nr6-and-kstar/NR6-T7.md "Sources").
SCORED = {
    0: "w12-single-design-crossover/scored/run1_seed0_K32/scored.json",
    1: "w12b-nr6-and-kstar/scored/run3_seed1_K32/scored.json",
    2: "w12b-nr6-and-kstar/scored/run3_seed2_K32/scored.json",
}
DELETION = "w12-single-design-crossover/deletion/test_seed0_K32/deletion.json"
WINNER_HASHES = "gen-decoder-lift/winner/winner_hashes.json"

#: Column order and the names/colours the `plots` spoke uses on every figure
#: (figures/FIGURES.md "One arm, one name, one colour").
ARMS = (
    ("cov_tensor", "CLU", "#2a78d6"),
    ("mlp__as_trained", "MLP", "#1baf7a"),
    ("lstm__as_trained", "LSTM", "#eda100"),
    ("transformer__as_trained", "Transformer", "#e87ba4"),
    ("cont_mlp__as_trained", "Continuous\nMLP", "#008300"),
    ("table", "Lookup table\n(floor)", "#6b6a66"),
)
CLU = "cov_tensor"
OTHERS = tuple(a for a, _, _ in ARMS if a != CLU)
RENDER_SEED = 0
REQUESTS_PER_CORNER = 2


def per_corner(seed: int) -> dict:
    d = json.loads((EVIDENCE / SCORED[seed]).read_text(encoding="utf-8"))
    return {a: {c["corner_id"]: c for c in d["per_arm"][a]["per_corner"]}
            for a, _, _ in ARMS}


def _counts(pc, cid):
    return {a: [int(pc[a][cid]["n_valid"]), int(pc[a][cid]["n_requests"])]
            for a, _, _ in ARMS}


def select(seed: int = RENDER_SEED) -> dict:
    pc = per_corner(seed)
    ids = list(pc[CLU])
    fam = {cid: pc[CLU][cid]["family"] for cid in ids}

    def full(cid):
        return pc[CLU][cid]["n_valid"] == pc[CLU][cid]["n_requests"]

    def others_zero(cid):
        return all(pc[a][cid]["n_valid"] == 0 for a in OTHERS)

    s1 = [cid for cid in ids if full(cid) and others_zero(cid)]
    s1_pair = sorted([c for c in s1 if fam[c] == "pair"],
                     key=lambda c: (not c.endswith("-hh"), c))
    s1_nway = [c for c in s1 if fam[c] == "nway"]
    set1 = s1_pair[:1] + s1_nway[:1]

    set2 = {}
    for s in SCORED:
        pcs = per_corner(s)
        set2[s] = [cid for cid in pcs[CLU]
                   if all(pcs[a][cid]["n_valid"] > 0 for a, _, _ in ARMS)]

    s3 = sorted([cid for cid in ids if others_zero(cid) and not full(cid)],
                key=lambda c: (pc[CLU][c]["n_valid"] / pc[CLU][c]["n_requests"], c))
    s3_pair = [c for c in s3 if fam[c] == "pair"]
    s3_nway = [c for c in s3 if fam[c] == "nway"]
    set3 = s3_pair[:1] + s3_nway[:1] if s3_nway else s3_pair[:2]

    return {
        "schema": "w17-rendered-compositions/selection/1",
        "render_seed": seed,
        "K_star": 32,
        "sources": {"scored_per_seed": {str(k): str(EVIDENCE / v) for k, v in SCORED.items()},
                    "deletion": str(EVIDENCE / DELETION),
                    "winner_hashes": str(EVIDENCE / WINNER_HASHES)},
        "arms": [{"key": a, "label": l, "colour": c} for a, l, c in ARMS],
        "set1_clu_composes_rivals_fail": {
            "rule": "CLU n_valid == n_requests AND every rival + floor n_valid == 0 (seed %d); "
                    "first '-hh' pair sub-corner, first n-way corner" % seed,
            "all_qualifying_corners": s1,
            "corners": set1,
            "counts_n_valid_of_n": {c: _counts(pc, c) for c in set1},
            "verdict_by_construction": {"CLU": "valid", "every other arm": "invalid"},
        },
        "set2_all_arms_succeed": {
            "rule": "every arm incl. the continuous MLP n_valid > 0 in the SAME seed "
                    "(necessary, not sufficient); request-level co-validity decided at render",
            "candidate_corners_per_seed": {str(k): v for k, v in set2.items()},
            "counts_n_valid_of_n": {str(s): {c: _counts(per_corner(s), c) for c in v}
                                    for s, v in set2.items()},
            "render_rule": "render scans EVERY held-out corner (pair first, then n-way, "
                           "scorer order) for requests valid in all six columns; if there "
                           "are none, the all-six set is DECLARED NOT BUILDABLE with the "
                           "count and the figure shows the nearest pattern (every column "
                           "except the continuous MLP), one request per corner, first 4",
        },
        "set3_all_arms_fail": {
            "rule": "every rival + floor n_valid == 0 AND CLU n_valid < n_requests (seed %d), "
                    "lowest CLU fraction first" % seed,
            "all_qualifying_corners": s3,
            "corners": set3,
            "counts_n_valid_of_n": {c: _counts(pc, c) for c in set3},
            "verdict_by_construction": {"every arm incl. CLU": "invalid"},
        },
        "requests_per_corner": REQUESTS_PER_CORNER,
        "within_corner_rule": "the corner's requests in the scorer's own order "
                              "(fw._request_values_from_rows over the corner's rows); the "
                              "first REQUESTS_PER_CORNER that meet the set's verdict pattern",
    }


if __name__ == "__main__":
    print(json.dumps(select(), indent=2))
