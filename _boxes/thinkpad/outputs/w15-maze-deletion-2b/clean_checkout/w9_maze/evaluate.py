"""Rendering a read into an answer, judging it exactly, and the refusal certificate.

RESULT rows never refuse: the arm emits, the exact judge grades, and a failure is a
zero.  That keeps the two row classes apart -- a RESULT row measures the read, and
nothing else.

The certificate (DISPLAY rows only)
-----------------------------------
A request-only object is entitled to consult exactly one world: the one its own read
imagines.  So the certificate decodes the scene the read produced -- floor, tints,
doors, keys, the agent cell, the target cell -- and runs the SAME parameterless
executor on THAT world.  If the imagined rollout does not reach the imagined target
within the requested budget, the object refuses.

There is no threshold in it, nothing is fitted to refusals, and no impossible example
was ever written.  Its only comparator is a supervised refuser that HAS seen them.
"""

from __future__ import annotations

import numpy as np

from w8_envscout.maze_colour import FLOOR, judge
from w9_maze.render import N_MAZE_CH, execute, execute_world


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def imagined_world(scene_logits):
    """Decode one scene into a world the executor can be run on."""
    p = _sig(scene_logits)
    grid = (p[:, :, 0] > 0.5).astype(np.uint8)
    H, W = grid.shape
    tint = np.zeros((H, W), np.int8)
    door = np.zeros((H, W), np.int8)
    tc = p[:, :, 1:5]
    dc = p[:, :, 5:9]
    ti = np.argmax(tc, axis=-1)
    di = np.argmax(dc, axis=-1)
    tint[tc.max(axis=-1) > 0.5] = (ti + 1)[tc.max(axis=-1) > 0.5]
    door[dc.max(axis=-1) > 0.5] = (di + 1)[dc.max(axis=-1) > 0.5]
    key_at = []
    for c in range(4):
        k = p[:, :, 9 + c]
        key_at.append(tuple(map(int, np.unravel_index(int(np.argmax(k)), k.shape))))
    start = tuple(map(int, np.unravel_index(int(np.argmax(p[:, :, 13])), (H, W))))
    goal = tuple(map(int, np.unravel_index(int(np.argmax(p[:, :, 14])), (H, W))))
    if grid[start] == 0:
        grid[start] = FLOOR
    if grid[goal] == 0:
        grid[goal] = FLOOR
    return dict(grid=grid, tint=tint, door=door, key_at=tuple(key_at),
                start=start, goal=goal)


def certificate(plan_cls, scene_logits, budget, forbid=()):
    """(refuse, imagined_steps).  Request-only: the true maze is never consulted."""
    w = imagined_world(scene_logits)
    route = execute_world(plan_cls, w["grid"], w["tint"], w["door"], w["key_at"],
                          w["start"], w["goal"], int(budget), 4, forbid)
    # replay on the imagined world to see whether the imagined target is reached
    pos = w["start"]
    from w8_envscout.maze_colour import CARRIED, MOVES, UNTAKEN, USED, _pocket
    key_cell = {w["key_at"][c]: c + 1 for c in range(4)}
    ks = [UNTAKEN] * 4
    reached = False
    for a in route:
        if a == "P":
            kc = key_cell.get(pos, 0)
            if kc and ks[kc - 1] == UNTAKEN and _pocket(tuple(ks)) == 0:
                ks[kc - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        npos = (pos[0] + dr, pos[1] + dc)
        dcol = int(w["door"][npos])
        if dcol and ks[dcol - 1] == CARRIED:
            ks[dcol - 1] = USED
        pos = npos
        if pos == w["goal"]:
            reached = True
            break
    ok = reached and len(route) <= int(budget)
    return (not ok), len(route)


def answers(bank, rows, plans, scenes=None, use_certificate=False, forbid=()):
    """Execute every read and judge it exactly.

    ``rows`` is a list of ``(request, m, i, j)``; ``plans[n]`` is that request's
    decoded plan map.  Returns one record per row -- nothing is aggregated here, so
    every breakout (rule B bends / does not bend, budget word, maze) is available to
    the caller without a second pass.
    """
    out = []
    for n, (req, m, i, j) in enumerate(rows):
        site = bank[m]
        refused, imagined_steps = (False, -1)
        if use_certificate:
            refused, imagined_steps = certificate(plans[n], scenes[n], req.b, forbid)
        route = "" if refused else execute(plans[n], site, i, j, req.b, forbid)
        inst = site.instance(i, j, req.b)
        ok, reason, _ = judge(inst, route)
        #: a walled pair has no optimum; nothing can be valid for it, and the judge
        #: agrees, so it is scored 0 and carried with opt = -1 to stay visible.
        walled = (i, j) not in site.opt
        opt = site.opt.get((i, j), -1)
        out.append({
            "m": m, "i": i, "j": j, "b": int(req.b),
            "feasible": bool((not walled) and opt <= req.b),
            "walled": bool(walled),
            "bends": bool((not walled) and site.bends(i, j)),
            "opt": int(opt), "opt_a": int(site.opt_a.get((i, j), -1)),
            "refused": bool(refused), "imagined_steps": int(imagined_steps),
            "valid": bool(ok and not refused),
            "score": float(opt / len(route)) if (ok and route and not refused) else 0.0,
            "len": len(route), "reason": "refused" if refused else reason,
        })
    return out


def aggregate(recs, mask=None):
    r = [x for x in recs if (mask is None or mask(x))]
    if not r:
        return {"n": 0, "validity": float("nan"), "score": float("nan"),
                "refusal": float("nan")}
    return {"n": len(r),
            "validity": float(np.mean([x["valid"] for x in r])),
            "score": float(np.mean([x["score"] for x in r])),
            "refusal": float(np.mean([x["refused"] for x in r]))}


def nway_aggregate(recs):
    """Row 3 scores the CONJUNCTION: two agents, one target, both routes valid."""
    by = {}
    for x in recs:
        by.setdefault((x["m"], x["j"], x["b"]), []).append(x)
    both, sc, n = [], [], 0
    for _k, g in by.items():
        if len(g) < 2:
            continue
        n += 1
        both.append(all(y["valid"] for y in g))
        sc.append(float(np.mean([y["score"] for y in g])) if all(
            y["valid"] for y in g) else 0.0)
    return {"n": n, "validity": float(np.mean(both)) if n else float("nan"),
            "score": float(np.mean(sc)) if n else float("nan"),
            "refusal": float(np.mean([x["refused"] for x in recs]))}
