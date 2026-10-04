"""judge maze."""

from __future__ import annotations
from nclu.cmaze.maze_colour import WALL

_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


def judge_ref(inst, route):
    """A second replay, written from the RULES, not from `judge`'s code.

    RULE A  entering a door cell of colour c requires the pocket to hold key c;
            the key is consumed and the door is open from then on.
    RULE B  entering a cell tinted colour c is forbidden while the pocket holds c.
    Plus: a pickup needs an empty pocket and an as-yet-untaken key on the current
    cell, the goal must be entered, and the route may not exceed the budget.

    Returns (valid, reason, steps) using the same reason vocabulary as `judge`,
    colour suffixes included, so the agreement check is on the colour too.
    """
    if route is None:
        return (False, "no-route", 0)
    if int(len(route)) > int(inst.budget):
        return (False, "over-budget", int(len(route)))
    grid, tint, door = (inst.grid, inst.tint, inst.door)
    H, W = (int(grid.shape[0]), int(grid.shape[1]))
    homes = {inst.key_at[i]: i + 1 for i in range(int(inst.n_colours))}
    carried = 0
    used = set()
    taken = set()
    r, c = (int(inst.start[0]), int(inst.start[1]))
    for t in range(len(route)):
        a = route[t]
        if a == "P":
            col = homes.get((r, c), 0)
            if col == 0:
                return (False, "pickup-on-empty-cell", t)
            if col in taken:
                return (False, "key-not-there", t)
            if carried != 0:
                return (False, "pocket-full", t)
            taken.add(col)
            carried = col
            continue
        if a not in _DELTA:
            return (False, "bad-action", t)
        dr, dc = _DELTA[a]
        nr, nc = (r + dr, c + dc)
        if nr < 0 or nr >= H or nc < 0 or (nc >= W) or (int(grid[nr, nc]) == WALL):
            return (False, "into-wall", t)
        dcol = int(door[nr, nc])
        if dcol != 0 and dcol not in used:
            if carried != dcol:
                return (False, "locked-door-c%d" % dcol, t)
            used.add(dcol)
            carried = 0
        tcol = int(tint[nr, nc])
        if tcol != 0 and carried == tcol:
            return (False, "forbidden-tint-c%d" % tcol, t)
        r, c = (nr, nc)
        if (r, c) == (int(inst.goal[0]), int(inst.goal[1])):
            return (True, "goal", t + 1)
    return (False, "no-goal", int(len(route)))


def reason_class(reason):
    return reason.split("@")[0].split("(")[0]
