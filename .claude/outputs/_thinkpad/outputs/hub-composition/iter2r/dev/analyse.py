"""Paired R1 statistics over the rivals ledger.  ⭐ Read-only; it measures nothing new.

    python analyse.py [tag] [--half sel|val|all] [--arms a,b,c]

R1 (Head, 2026-09-17), applied on the TUPLE column:
    tie  := the paired 95 % CI on Δ(arm − rival) includes 0, OR |Δ| ≤ m
    win  := Δ > 0 with the CI excluding 0
    loss := Δ < −m with the CI excluding 0
m = 0.020 is stated on the composition-B aggregate, a 0–1 metric. One aggregate unit is
72 tuples on this column, so the same margin is m × 72 = 1.44 tuples. ⭐ Both the
aggregate verdict and the tuple verdict are printed, so ⛔ nobody has to take the
rescaling on trust.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

import comp
import rivals

RIVAL = "knn_proto"


def load(tag=None, half="all"):
    rows = rivals.ledger_rows()
    if tag:
        rows = [r for r in rows if r.get("tag") == tag]
    if half == "sel":
        rows = [r for r in rows if r["corner_idx"] in comp.SELECTION_IDX]
    elif half == "val":
        rows = [r for r in rows if r["corner_idx"] in comp.VALIDATION_IDX]
    # last write wins per (tag, corner, arm)
    by = {}
    for r in rows:
        by[(r["tag"], r["corner_idx"], r["arm"])] = r
    return list(by.values())


def table(rows, column="tuples_valid_novel"):
    d = defaultdict(dict)
    for r in rows:
        d[r["arm"]][r["corner_idx"]] = r["headline"][column]
    return d


def report(tag=None, half="all", arms=None, column="tuples_valid_novel"):
    # ⚠ m = 0.020 is stated in AGGREGATE units. On the tuple column one aggregate unit is
    # 72 tuples, so the margin must be rescaled; on the aggregate column it must NOT be.
    # (Fixed 11:45Z — the first print of the aggregate column applied scale=72 to it and
    #  therefore called every row a tie. ⛔ The tuple column, which is the headline, was
    #  always scaled correctly. Corrected here, not papered over.)
    scale = 72.0 if column == "tuples_valid_novel" else 1.0
    rows = load(tag, half)
    if not rows:
        print("no rows")
        return {}
    T = table(rows, column)
    if RIVAL not in T:
        print(f"⛔ no {RIVAL} rows — there is no bar to test against")
        return {}
    arms = arms or [a for a in T if a != RIVAL]
    common_all = set.intersection(*[set(T[a]) for a in list(T)]) if T else set()
    print(f"tag={tag} half={half} column={column}")
    print(f"corners with EVERY arm present: n = {len(common_all)}  {sorted(common_all)}")
    byt = {r["arm"]: r["bytes"] for r in rows}
    out = {"tag": tag, "half": half, "column": column,
           "n_corners_complete": len(common_all),
           "corners": sorted(int(x) for x in common_all), "arms": {}}
    hdr = (f"{'arm':24s} {'bytes':>9s} {'n':>3s} {'mean':>7s} {'sd':>6s} "
           f"{'Δ vs knn':>9s} {'CI95 low':>9s} {'CI95 high':>9s} {'W/T/L':>9s} {'R1':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for a in [RIVAL] + sorted(arms):
        ks = sorted(set(T[a]) & set(T[RIVAL]))
        if not ks:
            continue
        x = np.array([T[a][k] for k in ks], float)
        y = np.array([T[RIVAL][k] for k in ks], float)
        if a == RIVAL:
            print(f"{a:24s} {byt.get(a, 0):9,d} {len(ks):3d} {x.mean():7.2f} "
                  f"{x.std(ddof=1):6.2f} {'—':>9s} {'—':>9s} {'—':>9s} "
                  f"{'(rival)':>9s} {'—':>6s}")
            out["arms"][a] = {"n": len(ks), "mean": float(x.mean()),
                              "sd": float(x.std(ddof=1)), "bytes": byt.get(a)}
            continue
        p = comp.paired(x, y)
        v = comp.r1_verdict(p, m=0.020, scale=scale)
        losses = int((np.asarray(x) - np.asarray(y) < 0).sum())
        print(f"{a:24s} {byt.get(a, 0):9,d} {len(ks):3d} {x.mean():7.2f} "
              f"{x.std(ddof=1):6.2f} {p['delta']:9.3f} {p['ci95'][0]:9.3f} "
              f"{p['ci95'][1]:9.3f} {p['wins']:3d}/{p['ties']}/{losses:<3d} {v:>6s}")
        out["arms"][a] = {"n": len(ks), "mean": float(x.mean()),
                          "sd": float(x.std(ddof=1)), "bytes": byt.get(a),
                          "paired_vs_knn_proto": p, "R1_verdict_tuple_column": v,
                          "R1_verdict_aggregate_units":
                              comp.r1_verdict({**p, "delta": p["delta"] / 72.0,
                                               "ci95": [p["ci95"][0] / 72.0,
                                                        p["ci95"][1] / 72.0]},
                                              m=0.020, scale=1.0),
                          "per_corner": {int(k): float(T[a][k]) for k in ks}}
    print(f"\nm = 0.020 aggregate units; the margin applied on this column is "
          f"{0.020 * scale:.4f}.")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = args[0] if args and not args[0].startswith("--") else None
    half = "all"
    col = "tuples_valid_novel"
    for i, a in enumerate(args):
        if a == "--half":
            half = args[i + 1]
        if a == "--column":
            col = args[i + 1]
    res = report(tag, half, column=col)
    if res:
        p = rivals.OUT / "dev" / f"analysis_{tag or 'all'}_{half}_{col}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"-> {p}")
