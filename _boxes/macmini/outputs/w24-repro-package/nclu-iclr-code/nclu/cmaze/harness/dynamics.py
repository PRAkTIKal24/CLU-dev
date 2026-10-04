"""The stepper the rollouts use, and the proof that it is the judge's twin.

An arm has to be told where it is before it can pick its next action, so a rollout
needs a transition function.  That is a second piece of code that knows the rules,
and a second piece of code that knows the rules is exactly how a harness ends up
grading itself against something other than the judge.

Two things keep that from happening:
  1. the stepper NEVER produces a score -- only the exact judge does;
  2. `audit_stepper` replays random routes through the stepper and checks that the
     stepper's own accept/reject and stop index match `JUDGE` exactly, on every
     route.  If they ever diverged, the rollout would be feeding arms a world the
     judge does not grade, and the audit would say so.

Everything here is integers.
"""

from __future__ import annotations
from nclu.cmaze.maze_colour import CARRIED, FLOOR, MOVES, UNTAKEN, USED, _pocket


class State:
    __slots__ = ("pos", "ks", "t")

    def __init__(self, pos, ks, t=0):
        self.pos = pos
        self.ks = ks
        self.t = t

    def copy(self):
        return State(self.pos, self.ks, self.t)

    @property
    def pocket(self):
        return _pocket(self.ks)


def initial_state(inst):
    return State(inst.start, tuple([UNTAKEN] * int(inst.n_colours)), 0)


def key_homes(inst):
    return {inst.key_at[i]: i + 1 for i in range(int(inst.n_colours))}


def step(inst, st, action, homes=None):
    homes = homes if homes is not None else key_homes(inst)
    ks = list(st.ks)
    if action == "P":
        col = homes.get(st.pos, 0)
        if col == 0 or ks[col - 1] != UNTAKEN or _pocket(st.ks) != 0:
            return (st, False, False)
        ks[col - 1] = CARRIED
        return (State(st.pos, tuple(ks), st.t + 1), True, False)
    if action not in MOVES:
        return (st, False, False)
    dr, dc = MOVES[action]
    nr, nc = (st.pos[0] + dr, st.pos[1] + dc)
    H, W = inst.grid.shape
    if not (0 <= nr < H and 0 <= nc < W) or inst.grid[nr, nc] != FLOOR:
        return (st, False, False)
    dcol = int(inst.door[nr, nc])
    if dcol != 0 and ks[dcol - 1] != USED:
        if ks[dcol - 1] != CARRIED:
            return (st, False, False)
        ks[dcol - 1] = USED
    tcol = int(inst.tint[nr, nc])
    if tcol != 0 and _pocket(tuple(ks)) == tcol:
        return (st, False, False)
    nxt = State((nr, nc), tuple(ks), st.t + 1)
    return (nxt, True, (nr, nc) == inst.goal)
