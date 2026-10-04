"""The fixed-seed DIVERSITY draw.  Applied to the seed-0 scan only; never to an image.

RULE (per figure, rows filled in their fixed order)
  1. Qualifying pool of a row = every held-out seed-0 request whose valid bits match the row's
     pattern (e.g. "MLP valid"), excluding requests already drawn for this figure.
  2. Shape target.  S = the shapes present among the figure's qualifying requests.  The target
     order is rng.permutation(S minus capsule) followed by capsule, cycled over the rows; so
     capsule is never drawn more often than any other shape.  If the target shape has no
     qualifying request for this row, the target is the present shape with the fewest draws
     so far (ties: earlier in the target order).
  3. Hue spread.  Among the pool at the target shape, score each request by how many of its
     object, floor and wall hue values are already used by this figure's earlier rows (0-3,
     each hue factor counted against its own used set); keep the minimum-score requests.
  4. Draw one uniformly with rng.integers.
rng = numpy.random.default_rng(SELECT_SEED); one generator per figure, seeded identically.
"""
from __future__ import annotations

import numpy as np

from . import common as C

SELECT_SEED = 20260925
F_FLOOR, F_WALL, F_OBJ, F_SCALE, F_SHAPE, F_ORIENT = range(6)
CAPSULE = 3


def _draw(requests, patterns):
    """patterns: list of (row_tag, predicate(bits)->bool).  -> list of drawn dicts."""
    rng = np.random.default_rng(SELECT_SEED)
    quals = [[r for r in requests if pred(r[5])] for _tag, pred in patterns]
    present = sorted({r[4][F_SHAPE] for q in quals for r in q})
    non_caps = [s for s in present if s != CAPSULE]
    order = [int(s) for s in rng.permutation(non_caps)] + ([CAPSULE] if CAPSULE in present else [])
    used = {F_OBJ: set(), F_FLOOR: set(), F_WALL: set()}
    n_shape = {s: 0 for s in order}
    taken, out = set(), []
    for i, ((tag, _pred), q) in enumerate(zip(patterns, quals)):
        pool = [r for r in q if r[2] not in taken]
        if not pool:
            out.append({"row": tag, "DECLARED": "no qualifying held-out seed-0 request",
                        "pool_size": 0})
            continue
        target = order[i % len(order)]
        at = [r for r in pool if r[4][F_SHAPE] == target]
        how = "cycled target"
        if not at:
            avail = [s for s in order if any(r[4][F_SHAPE] == s for r in pool)]
            target = min(avail, key=lambda s: (n_shape[s], order.index(s)))
            at = [r for r in pool if r[4][F_SHAPE] == target]
            how = "fallback: fewest-drawn shape with a qualifying request"
        score = [sum(r[4][f] in used[f] for f in used) for r in at]
        m = min(score)
        best = [r for r, s in zip(at, score) if s == m]
        r = best[int(rng.integers(len(best)))]
        taken.add(r[2])
        n_shape[target] += 1
        for f in used:
            used[f].add(r[4][f])
        out.append({"row": tag, "corner_id": r[0], "family": r[1], "dataset_row": r[2],
                    "index_in_corner": r[3], "requested_factor_indices": r[4],
                    "scan_bits": r[5], "pool_size": len(pool), "shape_target": target,
                    "shape_rule": how, "n_at_target_shape": len(at),
                    "hue_reuse_score": int(m), "n_tied_at_min_score": len(best)})
    return {"shape_order": order, "shapes_present": present, "rows": out}


def per_model(scan) -> dict:
    cols = scan["columns"]
    pats = []
    for col in (C.CLU,) + C.BASELINES:
        j = cols.index(col)
        pats.append((f"{col}:valid", lambda b, j=j: b[j] == "1"))
        pats.append((f"{col}:invalid", lambda b, j=j: b[j] == "0"))
    return _draw(scan["requests"], pats)


def baseline_wins(scan) -> dict:
    cols = scan["columns"]
    ci = cols.index(C.CLU)
    pats = [(f"{col}:valid&n_clu:invalid",
             lambda b, j=cols.index(col): b[j] == "1" and b[ci] == "0") for col in C.BASELINES]
    return _draw(scan["requests"], pats)


def counts(scan) -> dict:
    cols, R = scan["columns"], scan["requests"]
    ci = cols.index(C.CLU)
    shape = lambda r: C.SHAPE_NAMES[r[4][F_SHAPE]]  # noqa: E731

    def by_shape(rs):
        out = {}
        for r in rs:
            out[shape(r)] = out.get(shape(r), 0) + 1
        return dict(sorted(out.items()))

    per = {}
    for j, col in enumerate(cols):
        v = [r for r in R if r[5][j] == "1"]
        f = [r for r in R if r[5][j] == "0"]
        per[col] = {"valid": len(v), "valid_by_shape": by_shape(v),
                    "invalid": len(f), "invalid_by_shape": by_shape(f)}
    bw, cw = {}, {}
    for col in C.BASELINES + (C.TABLE,):
        j = cols.index(col)
        q = [r for r in R if r[5][j] == "1" and r[5][ci] == "0"]
        w = [r for r in R if r[5][j] == "0" and r[5][ci] == "1"]
        bw[col] = {"n": len(q), "by_shape": by_shape(q), "by_family": {
            fam: sum(1 for r in q if r[1] == fam) for fam in ("pair", "nway")}}
        cw[col] = {"n": len(w), "by_shape": by_shape(w)}
    anyb = [r for r in R if r[5][ci] == "0" and any(r[5][cols.index(c)] == "1" for c in C.BASELINES)]
    allb = [r for r in R if r[5][ci] == "1" and all(r[5][cols.index(c)] == "0" for c in C.BASELINES)]
    return {"seed": scan["seed"], "n_heldout_requests": len(R),
            "n_heldout_by_shape": by_shape(R), "per_column": per,
            "baseline_valid_and_n_clu_invalid": bw,
            "any_baseline_valid_and_n_clu_invalid": {"n": len(anyb), "by_shape": by_shape(anyb)},
            "n_clu_valid_and_baseline_invalid": cw,
            "n_clu_valid_and_every_baseline_invalid": {"n": len(allb), "by_shape": by_shape(allb)},
            "note": "the lookup table is the floor, counted for completeness, never a baseline"}
