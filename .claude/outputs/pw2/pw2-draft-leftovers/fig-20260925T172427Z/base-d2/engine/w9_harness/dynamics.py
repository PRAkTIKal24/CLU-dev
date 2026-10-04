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

import numpy as np

from w8_envscout.maze_colour import CARRIED, FLOOR, MOVES, UNTAKEN, USED, _pocket
from w9_harness.judge_maze import JUDGE, ACTIONS, reason_class


class State:
    """(cell, keystate) -- the whole of the agent's situation.  Small ints only."""

    __slots__ = ("pos", "ks", "t")

    def __init__(self, pos, ks, t=0):
        self.pos = pos
        self.ks = ks                     # tuple of UNTAKEN/CARRIED/USED, length k
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
    """Apply one action.  Returns (new_state, legal, at_goal).

    Illegal actions do not move the agent; the caller ends the episode.  The legality
    conditions are RULES A+B, the pickup preconditions and the grid -- the same
    conditions the judge checks, and `audit_stepper` is what holds them together.
    """
    homes = homes if homes is not None else key_homes(inst)
    ks = list(st.ks)
    if action == "P":
        col = homes.get(st.pos, 0)
        if col == 0 or ks[col - 1] != UNTAKEN or _pocket(st.ks) != 0:
            return st, False, False
        ks[col - 1] = CARRIED
        return State(st.pos, tuple(ks), st.t + 1), True, False
    if action not in MOVES:
        return st, False, False
    dr, dc = MOVES[action]
    nr, nc = st.pos[0] + dr, st.pos[1] + dc
    H, W = inst.grid.shape
    if not (0 <= nr < H and 0 <= nc < W) or inst.grid[nr, nc] != FLOOR:
        return st, False, False
    dcol = int(inst.door[nr, nc])
    if dcol != 0 and ks[dcol - 1] != USED:
        if ks[dcol - 1] != CARRIED:
            return st, False, False                       # RULE A
        ks[dcol - 1] = USED                               # key consumed
    tcol = int(inst.tint[nr, nc])
    if tcol != 0 and _pocket(tuple(ks)) == tcol:
        return st, False, False                           # RULE B
    nxt = State((nr, nc), tuple(ks), st.t + 1)
    return nxt, True, (nr, nc) == inst.goal


def replay(inst, route):
    """Run a whole route through the stepper.  Returns (states, legal_prefix_len,
    reached_goal).  `states[i]` is the state BEFORE action i."""
    homes = key_homes(inst)
    st = initial_state(inst)
    states = []
    for i, a in enumerate(route):
        states.append(st)
        st, legal, goal = step(inst, st, a, homes)
        if not legal:
            return states, i, False
        if goal:
            return states, i + 1, True
    return states, len(route), False


def audit_stepper(insts, n_routes=10, verbose=True):
    """The stepper's verdict must equal the exact judge's, route for route."""
    rng = np.random.default_rng(11)
    bad = 0
    total = 0
    for inst in insts:
        routes = ["".join(ACTIONS[int(rng.integers(5))]
                          for _ in range(int(rng.integers(1, int(inst.budget) + 3))))
                  for _ in range(n_routes)]
        for route in routes:
            total += 1
            _, plen, goal = replay(inst, route)
            # what the stepper says about THIS route, expressed as the judge would:
            step_ok = bool(goal and plen == len(route) and len(route) <= int(inst.budget))
            jok, jwhy, jn = JUDGE(inst, route)
            if step_ok != jok:
                bad += 1
                continue
            if not jok and reason_class(jwhy) not in ("over-budget", "no-goal") \
                    and plen != jn:
                bad += 1                          # disagreed on WHERE it went wrong
    if verbose:
        print("STEPPER AUDIT: %d routes, %d disagreements with the exact judge -> %s"
              % (total, bad, "PASS" if bad == 0 else "FAIL"))
    return {"routes": total, "disagreements": bad, "pass": bad == 0}
