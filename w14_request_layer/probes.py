"""The three diagnostic probes the repair rests on.  Each prints; nothing is fitted.

  python -m w14_request_layer.probes breakdown   > artefacts/anchor_breakdown.txt
  python -m w14_request_layer.probes density     > artefacts/probe_density.txt
  python -m w14_request_layer.probes direction   > artefacts/probe_direction.txt

None of these is a scored run.  They are measurements of the ENVIRONMENT and of the
REQUEST LAYER's acceptance rule, made before any object is fitted to anything.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict

import numpy as np

from w8_envscout.maze_colour import Instance, _reachable_without_keys, generate, solve
from w8_envscout.screen import key_order_of, plan, score_route
from w9_maze.task import _free_cells, door_free_components
from w14_request_layer.task2 import probe

PER = 3


def breakdown(n=400, seed0=0):
    """Why BLIND-SEQUENTIAL fails on CMAZE-1: it is the key ORDER, and nothing else."""
    insts, s = [], seed0
    while len(insts) < n:
        i = generate(s)
        s += 1
        if i is not None:
            insts.append(i)
    by_k = defaultdict(lambda: [0, 0])
    order_same = Counter()
    by_edges = defaultdict(lambda: [0, 0])
    for inst in insts:
        _L, r = solve(inst)
        k = len(key_order_of(inst, r))
        _la, ra = solve(inst, rule_b=False)
        same = key_order_of(inst, ra) == key_order_of(inst, r)
        ok, _sc, _rs = score_route(inst, plan(inst, "sequential"))
        by_k[k][0] += bool(ok)
        by_k[k][1] += 1
        order_same[(bool(same), bool(ok))] += 1
        comps = door_free_components(inst.grid, inst.door)
        comp_of = {c: n_ for n_, comp in enumerate(comps) for c in comp}
        cs = comp_of.get(inst.start, -1)
        nd = 0
        H, W = inst.grid.shape
        for (r_, c_) in map(tuple, np.argwhere(inst.door > 0)):
            touch = set()
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nr, nc = int(r_) + dr, int(c_) + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) in comp_of:
                    touch.add(comp_of[(nr, nc)])
            if cs in touch and len(touch) > 1:
                nd += 1
        by_edges[nd][0] += bool(ok)
        by_edges[nd][1] += 1
    print("CMAZE-1 proper, generate() at its own defaults, n=%d, seeds %d.." % (n, seed0))
    print()
    print("BLIND-SEQUENTIAL validity by KEY COUNT of the A+B optimum:")
    for k in sorted(by_k):
        print("   keys=%d : %3d/%3d = %.4f" % (k, by_k[k][0], by_k[k][1],
                                               by_k[k][0] / by_k[k][1]))
    print()
    print("Does the RULE-A key order equal the A+B key order, and is BLIND-SEQUENTIAL valid?")
    for same in (True, False):
        tot = order_same[(same, True)] + order_same[(same, False)]
        print("   order %-8s n=%3d  valid %3d = %.4f"
              % ("SAME" if same else "DIFFERENT", tot, order_same[(same, True)],
                 order_same[(same, True)] / max(1, tot)))
    print()
    print("BLIND-SEQUENTIAL validity by number of DOORS incident on the start's component:")
    for k in sorted(by_edges):
        print("   ndoors=%d : %3d/%3d = %.4f" % (k, by_edges[k][0], by_edges[k][1],
                                                 by_edges[k][0] / by_edges[k][1]))


def density(seeds=range(1_000_000, 1_000_012), ncand=24, rngseed=0):
    """Is '>= 2 keys' a property of the PAIR or of the MAZE?  (It is the maze.)"""
    rng = np.random.default_rng(rngseed)
    print("Random cell pairs inside one maze; %d candidate cells per maze." % ncand)
    print("seed        ordered pairs   door-gated+feasible      of those, >= 2 keys")
    for s in seeds:
        base = generate(s)
        if base is None:
            continue
        free = _free_cells(base)
        cand = [free[i] for i in rng.permutation(len(free))[:ncand]]
        good = feas = tot = 0
        for u in cand:
            for v in cand:
                if u == v:
                    continue
                tot += 1
                inst = Instance(0, base.grid, base.tint, base.door, base.key_at, u, v,
                                base.n_colours, 10 ** 9, 0, 0)
                if _reachable_without_keys(inst):
                    continue
                _L, r = solve(inst)
                if r is None:
                    continue
                if solve(inst, rule_b=False)[1] is None:
                    continue
                feas += 1
                good += len(key_order_of(inst, r)) >= 2
        print("%d   %5d           %5d (%.2f)            %5d (%.2f of feasible)"
              % (s, tot, feas, feas / tot, good, good / max(1, feas)))
    print()
    print("Reading: the column on the right is 0.00 or 1.00 and nothing between -- the")
    print("key count is a property of the DOOR CUT, so it is a property of the MAZE.")


def direction(seeds=range(1_000_000, 1_000_060)):
    """Is it directional?  (Yes -- and the reverse direction is not merely 1-key, it is
    infeasible, because the cut door's key lives on one side of the cut.)"""
    print("6 agents in one door-free component, 6 targets in the other; 18 cross pairs.")
    print("seed       direction   feasible/18   >= 2 keys/18")
    hits = seen = 0
    for s in seeds:
        rng = np.random.default_rng(s * 7919 + 13)
        base = generate(s)
        if base is None:
            continue
        comps = door_free_components(base.grid, base.door)
        freeset = set(_free_cells(base))
        usable = [u for u in ([c for c in comp if c in freeset] for comp in comps)
                  if len(u) >= 2 * PER]
        if len(usable) < 2:
            continue
        seen += 1
        A, B = usable[0], usable[1]
        for (src, dst, lab) in ((A, B, "0->1"), (B, A, "1->0")):
            agents = tuple(src[k] for k in rng.permutation(len(src))[:2 * PER])
            targets = tuple(dst[k] for k in rng.permutation(len(dst))[:2 * PER])
            tint = base.tint.copy()
            for c in list(agents) + list(targets):
                tint[c] = 0
            st = {"grid": base.grid, "tint": tint, "door": base.door,
                  "key_at": base.key_at, "agents": agents, "targets": targets,
                  "n_colours": base.n_colours}
            cross = [(i, j) for i in range(2 * PER) for j in range(2 * PER)
                     if (i < PER) != (j < PER)]
            n2 = sum(probe(st, i, j, 2)[0] for (i, j) in cross)
            nf = sum(probe(st, i, j, 0)[0] for (i, j) in cross)
            if n2 or nf != 18:
                print("%d   %s        %2d            %2d" % (s, lab, nf, n2))
            if n2 >= 12:
                hits += 1
    print()
    print("%d of %d mazes with two usable components have a direction at >= 12/18 keys>=2."
          % (hits, seen))


if __name__ == "__main__":
    t0 = time.time()
    {"breakdown": breakdown, "density": density, "direction": direction}[sys.argv[1]]()
    print()
    print("(%s, %.0fs)" % (sys.argv[1], time.time() - t0))
