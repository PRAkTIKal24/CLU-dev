"""The exhaustive structural sweep: does the speed limit actually bend the route?

This is the 2-D analogue of the parent's decisive 1-D result

    "25,261,855 solvable configurations. Configurations made more expensive by the wall: 0."

which proved an obstacle can never bend a path on a line. The two questions asked here are
the two that a 2-D gym has to answer before the wave-7 question can even be posed:

  S1  does the OBSTACLE bend the path?  For every configuration, the exact optimal cost
      with the obstacle field against the exact optimal cost with it deleted.
  S2  does the SPEED LIMIT bend the ROUTE?  For every configuration that can cross either
      way, the cheaper of (tunnel-only, detour-only) at each V, and whether that choice
      is the same at every V. A configuration whose route flips is one where "compose the
      speed limit with the obstacle to find a different route" is a real instruction.

Exhaustive, not sampled, in (start cell x start velocity): one backward BFS gives the exact
cost from every state at once, so each (layout, V, goal) triple contributes the whole
(400 cells x velocities) product. Goals are subsampled; starts and velocities are not.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

from chlu.gym2 import grid2d as E2, layouts as LY

OUT = os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs") + "/w51-gym2"
GOALS_PER_LAYOUT = 40
V_LIST = (1, 2, 3, 4)
_LAYOUTS = None


def _cost_slice(lid, V, gx, gy, extra=()):
    """(400, 9) exact costs for every start cell and every |v| <= 1 start velocity."""
    d = E2._dist_field(lid, V, gx, gy, extra)
    if d is None:
        return None
    nv = 2 * V + 1
    d = d.reshape(E2.G * E2.G, nv, nv)
    lo, hi = V - 1, V + 2
    return d[:, lo:hi, lo:hi].reshape(E2.G * E2.G, 9)


_FREE_LID = None


def work(args):
    lid, seed = args
    rng = np.random.default_rng(seed)
    lay = E2.get_layout(lid)
    meta = lay.meta
    free_cells = np.nonzero(~lay.blocked.reshape(-1))[0]
    goals = rng.choice(free_cells, size=min(GOALS_PER_LAYOUT, free_cells.size),
                       replace=False)

    n_conf = 0
    n_solv = 0
    n_obst_costlier = 0
    n_obst_infeasible = 0          # solvable without the obstacle, unsolvable with it
    n_route_pairs = 0              # configurations where BOTH routes are feasible
    n_route_flip = 0               # ... and the cheaper one is not the same at every V
    flip_by_pair = {}
    route_hist = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}   # [tunnel, detour]
    tie = 0

    for gflat in goals:
        gx, gy = int(gflat) // E2.G, int(gflat) % E2.G
        per_V = {}
        for V in V_LIST:
            c_all = _cost_slice(lid, V, gx, gy)
            if c_all is None:
                continue
            c_free = _cost_slice(_FREE_LID, V, gx, gy)
            ct = _cost_slice(lid, V, gx, gy, meta["block_top"])
            cd = _cost_slice(lid, V, gx, gy, meta["block_tunnel"])
            per_V[V] = (c_all, c_free, ct, cd)

            solv = c_all >= 0
            freeok = (c_free is not None) and (c_free >= 0)
            n_conf += solv.size
            n_solv += int(solv.sum())
            if c_free is not None:
                n_obst_costlier += int((solv & freeok & (c_all > c_free)).sum())
                n_obst_infeasible += int((~solv & freeok).sum())
            if ct is not None and cd is not None:
                both = (ct >= 0) & (cd >= 0)
                route_hist[V][0] += int((both & (ct < cd)).sum())
                route_hist[V][1] += int((both & (cd < ct)).sum())
                tie += int((both & (ct == cd)).sum())

        Vs = [V for V in V_LIST if V in per_V]
        if len(Vs) >= 2:
            both_all = None
            choice = []
            for V in Vs:
                _, _, ct, cd = per_V[V]
                if ct is None or cd is None:
                    choice = []
                    break
                b = (ct >= 0) & (cd >= 0)
                both_all = b if both_all is None else (both_all & b)
                choice.append(np.where(ct <= cd, 0, 1))
            if choice:
                ch = np.stack(choice, 0)
                flip = both_all & (ch.min(0) != ch.max(0))
                n_route_pairs += int(both_all.sum())
                n_route_flip += int(flip.sum())
                for i in range(len(Vs) - 1):
                    k = f"V{Vs[i]}->V{Vs[i+1]}"
                    m = both_all & (choice[i] != choice[i + 1])
                    flip_by_pair[k] = flip_by_pair.get(k, 0) + int(m.sum())

    return dict(lid=lid, n_conf=n_conf, n_solv=n_solv,
                n_obst_costlier=n_obst_costlier, n_obst_infeasible=n_obst_infeasible,
                n_route_pairs=n_route_pairs, n_route_flip=n_route_flip,
                flip_by_pair=flip_by_pair, route_hist=route_hist, tie=tie,
                tunnel_len=meta["tunnel_len"])


def main():
    t0 = time.time()
    with open(f"{OUT}/splits.pkl", "rb") as f:
        d = pickle.load(f)
    entries = list(d["layouts"])
    # an empty grid, appended once, is the obstacle-deleted reference for S1
    entries.append((np.zeros((E2.G, E2.G), dtype=bool), {"purpose": "obstacle_free"}))
    free_lid = len(entries) - 1

    lids = sorted(set(d["train_lids"]) | set(d["novel_lids"]))
    print(f"structural sweep over {len(lids)} layouts x {len(V_LIST)} V x "
          f"{GOALS_PER_LAYOUT} goals, exhaustive in (start cell x start velocity)",
          flush=True)

    global _FREE_LID
    _FREE_LID = free_lid

    with Pool(2, initializer=_init_worker, initargs=(entries, free_lid)) as p:
        res = p.map(work, [(l, 1000 + l) for l in lids], chunksize=4)

    agg = dict(n_conf=0, n_solv=0, n_obst_costlier=0, n_obst_infeasible=0,
               n_route_pairs=0, n_route_flip=0, tie=0)
    flip_by_pair, route_hist = {}, {v: [0, 0] for v in V_LIST}
    by_bucket = {}
    for r in res:
        for k in agg:
            agg[k] += r[k]
        for k, v in r["flip_by_pair"].items():
            flip_by_pair[k] = flip_by_pair.get(k, 0) + v
        for v in V_LIST:
            route_hist[v][0] += r["route_hist"][v][0]
            route_hist[v][1] += r["route_hist"][v][1]
        b = LY.t_bucket(r["tunnel_len"])
        e = by_bucket.setdefault(b, [0, 0])
        e[0] += r["n_route_pairs"]
        e[1] += r["n_route_flip"]

    out = {
        "goals_per_layout": GOALS_PER_LAYOUT, "n_layouts": len(lids),
        "V_list": list(V_LIST),
        "configurations_examined": agg["n_conf"],
        "solvable": agg["n_solv"],
        "S1_obstacle_raises_cost": agg["n_obst_costlier"],
        "S1_obstacle_raises_cost_frac_of_solvable":
            agg["n_obst_costlier"] / max(agg["n_solv"], 1),
        "S1_obstacle_makes_infeasible": agg["n_obst_infeasible"],
        "S2_route_choice_posed": agg["n_route_pairs"],
        "S2_route_flips_with_V": agg["n_route_flip"],
        "S2_route_flip_frac": agg["n_route_flip"] / max(agg["n_route_pairs"], 1),
        "S2_flip_by_adjacent_V": flip_by_pair,
        "S2_route_hist_by_V": {str(k): {"tunnel": v[0], "detour": v[1]}
                                for k, v in route_hist.items()},
        "S2_ties": agg["tie"],
        "S2_flip_frac_by_tunnel_bucket": {
            str(k): {"posed": v[0], "flips": v[1], "frac": v[1] / max(v[0], 1)}
            for k, v in sorted(by_bucket.items())},
        "seconds": time.time() - t0,
    }
    with open(f"{OUT}/structure.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


def _init_worker(entries, free_lid):
    global _FREE_LID
    E2.reset_registry(entries)
    _FREE_LID = free_lid


if __name__ == "__main__":
    main()
