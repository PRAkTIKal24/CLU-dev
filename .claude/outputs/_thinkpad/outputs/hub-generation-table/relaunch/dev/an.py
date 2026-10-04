"""RELAUNCH - paired analysis of the DEV ledger.  Read-only.

Every row prints its distinct-tuple count beside its aggregate (the task's standing rule)
and its total byte ledger.  Deltas are PAIRED per corner against ``knn_proto`` measured
on that same corner, with a 95 % CI on the paired difference (R1's tie rule).

usage:  an.py [table ...]            (default: every table in the ledger)
        an.py --set select|confirm|all
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

OUT3 = os.environ["CHLU_OUT3"].replace("/c/", "C:/", 1)
LEDGERS = sorted(__import__("glob").glob(os.path.join(OUT3, "runs", "r1_ledger*.jsonl")))

M_TIE = 0.020          # R1's tie band
FLOOR_STRUCTURAL = 0.5278


def load():
    rows = []
    for path in LEDGERS:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return [r for r in rows if "error" not in r]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = "all"
    for a in sys.argv[1:]:
        if a.startswith("--set"):
            which = a.split("=", 1)[1] if "=" in a else "all"
    rows = load()
    by = defaultdict(dict)                       # corner_tag -> variant -> row
    order = {}
    for r in rows:
        by[r["corner_tag"]][r["variant"]] = r
        order[r["corner_tag"]] = r["corner_index"]

    # the admitted-corner order defines select (first 6) / confirm (the rest)
    tags = sorted(by, key=lambda t: order[t])
    admitted_rank = {t: i for i, t in enumerate(tags)}
    if which == "select":
        tags = [t for t in tags if admitted_rank[t] < 6]
    elif which == "confirm":
        tags = [t for t in tags if admitted_rank[t] >= 6]

    variants = []
    for t in tags:
        for v in by[t]:
            if v not in variants:
                variants.append(v)
    if args:
        keep = set(args) | {"knn_proto"}
        variants = [v for v in variants if v in keep or
                    any(v.startswith(a) for a in args)]

    print("ledgers: " + ", ".join(os.path.basename(x) for x in LEDGERS))
    print(f"corners in scope: {len(tags)}  (set={which})")
    hdr = (f"{'variant':20s} {'n':>3s} {'aggregate':>10s} {'sd':>7s} {'tuples':>7s} "
           f"{'bytes':>8s} {'D vs knn':>9s} {'SE':>7s} {'95% CI':>18s} {'wins':>7s} "
           f"{'verdict':>8s}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for v in variants:
        ts = [t for t in tags if v in by[t] and "knn_proto" in by[t]]
        if not ts:
            continue
        a = np.array([by[t][v]["aggregate"] for t in ts])
        k = np.array([by[t]["knn_proto"]["aggregate"] for t in ts])
        tup = np.array([by[t][v]["distinct_factor_tuples"] for t in ts])
        b = int(by[ts[0]][v].get("total_bytes", 0))
        d = a - k
        n = len(ts)
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        if v == "knn_proto":
            verd = "-"
        elif d.mean() > 0 and lo > 0:
            verd = "WIN"
        elif (lo <= 0 <= hi) or abs(d.mean()) <= M_TIE:
            verd = "tie"
        else:
            verd = "LOSS"
        print(f"{v:20s} {n:3d} {a.mean():10.4f} {a.std(ddof=1) if n>1 else 0:7.4f} "
              f"{tup.mean():7.1f} {b:8d} {d.mean():+9.4f} {se:7.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {int((a>k).sum()):3d}/{n:<3d} {verd:>8s}")
        out[v] = {"n": n, "aggregate": float(a.mean()), "sd": float(a.std(ddof=1)) if n > 1 else 0.0,
                  "tuples": float(tup.mean()), "total_bytes": b,
                  "delta_vs_knn": float(d.mean()), "se": se, "ci95": [float(lo), float(hi)],
                  "wins": int((a > k).sum()), "verdict": verd,
                  "per_corner": {t: float(by[t][v]["aggregate"]) for t in ts}}
    bc = [by[t].get("knn_proto", {}).get("best_constant") for t in tags]
    bc = [x for x in bc if x is not None]
    if bc:
        print(f"\nbest-constant floor on these corners: {np.mean(bc):.4f} "
              f"(the structural eval_b floor is {FLOOR_STRUCTURAL})")
    print("R1: win = D>0 with CI excluding 0; tie = CI contains 0 OR |D| <= 0.020")

    # ------------------------------------------------------------------ #
    # PAIRED variant-vs-REFERENCE.  This is the statistic that can SELECT a
    # lever: two variants share nearly all of the corner-to-corner variance
    # (paired sd vs knn ~ 0.075; vs each other ~ 0.01), so the comparison
    # that decides a mechanism question must be variant-vs-variant, never
    # variant-vs-knn.  The last column is the n a two-sided 95% test would
    # need to resolve the observed effect -- the honest power statement.
    # ------------------------------------------------------------------ #
    ref = None
    for a in sys.argv[1:]:
        if a.startswith("--ref="):
            ref = a.split("=", 1)[1]
    if ref and ref in out:
        print("")
        print(f"PAIRED against reference variant '{ref}':")
        h2 = (f"{'variant':20s} {'n':>3s} {'D vs ref':>9s} {'pairedSD':>9s} {'SE':>7s} "
              f"{'95% CI':>18s} {'wins':>7s} {'n for 95%':>10s}")
        print(h2)
        print("-" * len(h2))
        for v in variants:
            if v == ref:
                continue
            ts = [t for t in tags if v in by[t] and ref in by[t]]
            if len(ts) < 2:
                continue
            d = np.array([by[t][v]["aggregate"] - by[t][ref]["aggregate"] for t in ts])
            n = len(ts)
            sd = float(d.std(ddof=1))
            se = sd / np.sqrt(n)
            lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
            need = int(np.ceil((1.96 * sd / max(abs(d.mean()), 1e-9)) ** 2)) if d.mean() else 10 ** 9
            print(f"{v:20s} {n:3d} {d.mean():+9.4f} {sd:9.4f} {se:7.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {int((d>0).sum()):3d}/{n:<3d} {min(need, 99999):10d}")
    # the paired sd against knn -- what any WIN claim has to beat
    kd = {}
    for v in variants:
        if v == "knn_proto":
            continue
        ts = [t for t in tags if v in by[t] and "knn_proto" in by[t]]
        if len(ts) > 1:
            d = np.array([by[t][v]["aggregate"] - by[t]["knn_proto"]["aggregate"] for t in ts])
            kd[v] = float(d.std(ddof=1))
    if kd:
        sd = float(np.median(list(kd.values())))
        print("")
        print(f"median PAIRED sd vs knn_proto on these corners: {sd:.4f}")
        for n in (6, 24, 43, 49):
            print(f"   at n={n:2d}: SE={sd/np.sqrt(n):.4f}  =>  a WIN (CI excluding 0) "
                  f"needs D > {1.96*sd/np.sqrt(n):+.4f}")
    json.dump(out, open(os.path.join(OUT3, "runs", f"an_{which}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
