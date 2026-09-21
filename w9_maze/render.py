"""The ITEM: a maze image with the answer drawn on it, and the shared executor.

What is written, and why it is an image
---------------------------------------
Wave-9 rule 1 binds every arm to one shared encoder trained by reconstruction on maze
images and one shared action head.  So the object the store holds has to BE an image.
The item for a request ``(maze, agent, target)`` is the maze, plus the answer drawn as
a **plan map**: for every (cell, phase) the action the optimal route takes there.

    phase p = 2 * (keys used so far) + (carrying a key)      -> 10 phases at k=4
    class   = 0 none, 1..5 = N, E, S, W, P                   -> 6 classes

A plan map, not an action sequence, is what makes the shared head a fixed, parameterless
executor: read the class at the cell you are standing on in the phase you are in, take
that action, repeat.  The dynamics that move you are the environment's own, and the
judge is CMAZE-1's, unchanged.

The phase axis is not decoration.  A route walks to a key, picks it up, walks to the
door and opens it; it therefore re-enters cells it has already visited, and what it
should do there depends on what it is carrying.  ``plan_fidelity`` measures exactly how
often (cell, phase) is a sufficient index -- it is the CEILING on every arm, it is
measured before any arm is built, and it is printed with every result.

Channels (C = 75 at k = 4)
    0        floor
    1..4     tint colour one-hot
    5..8     door colour one-hot
    9..12    key home colour one-hot
    13       the requested agent's cell
    14       the requested target's cell
    15..74   the plan map, 10 phases x 6 classes
"""

from __future__ import annotations

import numpy as np

from w8_envscout.maze_colour import CARRIED, FLOOR, MOVES, UNTAKEN, USED, _pocket, judge

ACTION_CHARS = "NESWP"
N_CLASS = 6                      # 0 = no action, then N, E, S, W, P
N_PHASE = 10                     # 2 * (keys used) + carrying, at n_colours = 4
N_MAZE_CH = 15
N_PLAN_CH = N_PHASE * N_CLASS
N_CH = N_MAZE_CH + N_PLAN_CH


def phase_index(ks) -> int:
    """``2 * (keys used) + (carrying)``, clipped into the phase axis."""
    used = sum(1 for v in ks if v == USED)
    carry = 1 if _pocket(tuple(ks)) else 0
    return min(2 * used + carry, N_PHASE - 1)


def maze_channels(site, i: int, j: int) -> np.ndarray:
    """``(H, W, 15)`` -- the scene: the maze, and which agent and which target."""
    H, W = site.grid.shape
    out = np.zeros((H, W, N_MAZE_CH), np.float32)
    out[:, :, 0] = (site.grid == FLOOR)
    for c in range(1, site.n_colours + 1):
        out[:, :, c] = (site.tint == c)
        out[:, :, 4 + c] = (site.door == c)
        out[site.key_at[c - 1][0], site.key_at[c - 1][1], 8 + c] = 1.0
    out[site.agents[i][0], site.agents[i][1], 13] = 1.0
    out[site.targets[j][0], site.targets[j][1], 14] = 1.0
    return out


def plan_classes(site, i: int, j: int, route: str) -> np.ndarray:
    """``(H, W, N_PHASE)`` int8 class map drawn along one route.

    Returns the map and, separately, how many (cell, phase) slots the route asked to
    hold two different actions -- a collision is a loss of the representation, not of
    the arm, so it is counted here and reported, never absorbed.
    """
    H, W = site.grid.shape
    cls = np.zeros((H, W, N_PHASE), np.int8)
    key_cell = {site.key_at[c]: c + 1 for c in range(site.n_colours)}
    ks = [UNTAKEN] * site.n_colours
    pos = site.agents[i]
    collisions = 0
    for a in route:
        p = phase_index(ks)
        want = ACTION_CHARS.index(a) + 1
        have = int(cls[pos[0], pos[1], p])
        if have and have != want:
            collisions += 1
        else:
            cls[pos[0], pos[1], p] = want
        if a == "P":
            ks[key_cell[pos] - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        npos = (pos[0] + dr, pos[1] + dc)
        dcol = int(site.door[npos])
        if dcol and ks[dcol - 1] == CARRIED:
            ks[dcol - 1] = USED
        pos = npos
    return cls, collisions


def item_tensor(site, i: int, j: int, route: str) -> np.ndarray:
    """``(H, W, 75)`` float32 -- the maze image with the answer drawn on it."""
    H, W = site.grid.shape
    cls, _ = plan_classes(site, i, j, route)
    out = np.zeros((H, W, N_CH), np.float32)
    out[:, :, :N_MAZE_CH] = maze_channels(site, i, j)
    plan = np.zeros((H, W, N_PHASE, N_CLASS), np.float32)
    ii, jj, pp = np.meshgrid(np.arange(H), np.arange(W), np.arange(N_PHASE),
                             indexing="ij")
    plan[ii, jj, pp, cls] = 1.0
    out[:, :, N_MAZE_CH:] = plan.reshape((H, W, N_PLAN_CH))
    return out


def plan_from_tensor(x: np.ndarray) -> np.ndarray:
    """``(H, W, 75)`` -> ``(H, W, N_PHASE)`` argmax class map."""
    H, W = x.shape[:2]
    plan = np.asarray(x[:, :, N_MAZE_CH:], np.float32).reshape(
        (H, W, N_PHASE, N_CLASS))
    return np.argmax(plan, axis=-1).astype(np.int8)


# --------------------------------------------------------------- the executor
def execute_world(cls: np.ndarray, grid, tint, door, key_at, start, goal, budget,
                  n_colours=4, forbid=()):
    """THE SHARED ACTION HEAD.  Parameterless: read the class, take the action.

    It is identical for every arm -- the CLU, the lookup floor, the MLP, the LSTM and
    the transformer all emit the same object (a plan map) and it is executed the same
    way.  Nothing here is learned and nothing here is tuned, so a margin between two
    arms is a margin on what they READ.

    The world is passed in rather than taken from a Site, because the refusal
    certificate runs this same executor on the world the READ IMAGINES -- which is the
    only world a request-only object is entitled to consult.

    ``forbid`` is the set of action characters the interface no longer carries; it is
    how "delete move-left" is applied, and it is empty everywhere else.

    The rollout stops on 'no action', on a deleted action, on an illegal move, or at
    the budget; the route it has emitted by then is handed to the judge as it stands.
    """
    H, W = np.asarray(grid).shape
    key_cell = {tuple(key_at[c]): c + 1 for c in range(n_colours)
                if key_at[c] is not None}
    ks = [UNTAKEN] * n_colours
    pos = tuple(start)
    goal = tuple(goal) if goal is not None else None
    out = []
    for _ in range(int(budget)):
        k = int(cls[pos[0], pos[1], phase_index(ks)])
        if k == 0:
            break
        a = ACTION_CHARS[k - 1]
        if a in forbid:
            break
        out.append(a)
        if a == "P":
            kc = key_cell.get(pos, 0)
            if kc == 0 or ks[kc - 1] != UNTAKEN or _pocket(tuple(ks)) != 0:
                break
            ks[kc - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] != FLOOR:
            break
        dcol = int(door[nr, nc])
        if dcol and ks[dcol - 1] != USED:
            if ks[dcol - 1] != CARRIED:
                break
            ks[dcol - 1] = USED
        tcol = int(tint[nr, nc])
        if tcol and _pocket(tuple(ks)) == tcol:
            break
        pos = (nr, nc)
        if goal is not None and pos == goal:
            break
    return "".join(out)


def execute(cls: np.ndarray, site, i: int, j: int, budget: int, forbid=()) -> str:
    """The executor on the TRUE world -- what the exact judge then grades."""
    return execute_world(cls, site.grid, site.tint, site.door, site.key_at,
                         site.agents[i], site.targets[j], budget,
                         site.n_colours, forbid)


def score_route(site, i, j, budget, route):
    """(valid, score, reason) under the EXACT judge.  score = opt / len, else 0."""
    inst = site.instance(i, j, budget)
    ok, reason, _ = judge(inst, route)
    sc = (site.opt[(i, j)] / len(route)) if (ok and route) else 0.0
    return bool(ok), float(sc), reason


def plan_fidelity(site, i, j, budget=None):
    """The CEILING: draw the exact optimum, execute the drawing, judge it."""
    route = site.route[(i, j)]
    b = int(budget if budget is not None else site.opt[(i, j)])
    cls, coll = plan_classes(site, i, j, route)
    emitted = execute(cls, site, i, j, b)
    ok, sc, reason = score_route(site, i, j, b, emitted)
    return ok, sc, coll, reason
