"""GYM-1 environment: one agent on a line with a speed limit and an uncrossable wall.

Design note (why momentum). Without velocity, "move toward the goal by at most V" is
optimal and the wall never bends a single *solvable* path -- it would only ever decide
solvability, i.e. it would be decorative. With velocity the agent can overshoot, so
braking distance makes the wall load-bearing for solvable pairs too.

Everything here is integer arithmetic. Every validity predicate is exact: there is no
tolerance anywhere in this file, by construction, because an approximate validity check
would make every downstream refusal claim approximate.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Fixed geometry of the gym.
# ---------------------------------------------------------------------------

L = 128        # cells of the line, x in [0, L-1]
A_MAX = 1      # acceleration limit: the action alphabet is a in [-A_MAX, +A_MAX]
V_MAX_MAX = 6  # the largest speed limit any episode can carry

# A_MAX < V is what gives the agent a braking DISTANCE of V*(V-1)/2 cells, and that is
# what makes the wall load-bearing rather than decorative: an agent at speed cannot stop
# dead, so it has to plan its approach. (With A_MAX >= V the agent can always halt in one
# step and the obstacle only ever decides solvability -- measured, see LOG.md.)
#
# The alphabet is still WIDER than the rules permit: repeated +1 drives |v| past any V,
# and driving at speed into the end of the line puts x' off it. A gym whose interface
# makes violations inexpressible cannot measure whether a model obeys a rule.
ACTIONS = tuple(range(-A_MAX, A_MAX + 1))
N_ACTIONS = len(ACTIONS)

FAULT_SPEED = "speed_limit_violation"
FAULT_WALL = "wall_crossing"
FAULT_TELEPORT = "teleport"
FAULT_NAMES = (FAULT_SPEED, FAULT_WALL, FAULT_TELEPORT)

NO_WALL = -1


class Rules(NamedTuple):
    """The per-episode rules. These are VISIBLE to every model trained here."""

    V: int    # speed limit: |v'| <= V
    w0: int   # wall left edge (ww == 0 means "no wall")
    ww: int   # wall width in cells; 0 means no wall

    @property
    def wall_hi(self) -> int:
        """Inclusive right edge. Meaningless when ww == 0."""
        return self.w0 + self.ww - 1


class Episode(NamedTuple):
    """One task instance."""

    V: int
    w0: int
    ww: int
    s: int     # start cell
    v0: int    # start velocity
    g: int     # goal cell; the agent must arrive AT REST and stay

    @property
    def rules(self) -> Rules:
        return Rules(self.V, self.w0, self.ww)

    def key(self) -> tuple:
        return (self.V, self.w0, self.ww, self.s, self.v0, self.g)


# ---------------------------------------------------------------------------
# Exact predicates.
# ---------------------------------------------------------------------------

def is_free_cell(x: int, w0: int, ww: int) -> bool:
    """True iff cell x exists on the line and is not wall."""
    if not (0 <= x < L):
        return False
    if ww <= 0:
        return True
    return not (w0 <= x <= w0 + ww - 1)


def segment_hits_wall(x: int, x2: int, w0: int, ww: int) -> bool:
    """Does the CLOSED swept segment [min(x,x2), max(x,x2)] meet the wall?

    Swept-segment rather than endpoint containment is exactly what makes the wall
    uncrossable: with momentum a single step can jump clean over a narrow wall, and
    that jump is a crossing.
    """
    if ww <= 0:
        return False
    lo, hi = (x, x2) if x <= x2 else (x2, x)
    return lo <= w0 + ww - 1 and hi >= w0


def step_faults(x: int, v: int, a: int, V: int, w0: int, ww: int):
    """Apply acceleration `a` and return (x2, v2, faults).

    `faults` is a dict of the three exact rule predicates. They are reported
    INDEPENDENTLY -- one step can break more than one rule -- and each is an exact
    integer test with no tolerance.
    """
    v2 = v + a
    x2 = x + v2
    faults = {
        FAULT_SPEED: abs(v2) > V,
        FAULT_WALL: segment_hits_wall(x, x2, w0, ww),
        FAULT_TELEPORT: not (0 <= x2 < L),
    }
    return x2, v2, faults


def step(x: int, v: int, a: int, V: int, w0: int, ww: int):
    """(x2, v2, is_valid)."""
    x2, v2, faults = step_faults(x, v, a, V, w0, ww)
    return x2, v2, not any(faults.values())


# ---------------------------------------------------------------------------
# Ground truth: exhaustive BFS over the (x, v) transition graph.
#
# The graph has L * (2V+1) <= 1408 states, so "optimal" here means provably optimal,
# not "what a heuristic teacher happened to produce".
# ---------------------------------------------------------------------------

class _Graph(NamedTuple):
    V: int
    n_states: int
    succ: np.ndarray    # (n_states, N_ACTIONS) int32, -1 where the transition is invalid
    alive: np.ndarray   # (n_states,) bool -- free cell and |v| <= V
    edges_from: np.ndarray
    edges_to: np.ndarray


def _sidx(x: int, v: int, V: int) -> int:
    return x * (2 * V + 1) + (v + V)


@lru_cache(maxsize=256)
def _graph(V: int, w0: int, ww: int) -> _Graph:
    """Build the valid-transition graph for one rule setting. Cached per rule setting."""
    nv = 2 * V + 1
    n = L * nv
    succ = np.full((n, N_ACTIONS), -1, dtype=np.int32)
    alive = np.zeros(n, dtype=bool)

    free = np.array([is_free_cell(x, w0, ww) for x in range(L)], dtype=bool)
    for x in range(L):
        if not free[x]:
            continue
        for v in range(-V, V + 1):
            i = _sidx(x, v, V)
            alive[i] = True
            for ai, a in enumerate(ACTIONS):
                v2 = v + a
                if abs(v2) > V:
                    continue
                x2 = x + v2
                if not (0 <= x2 < L):
                    continue
                if not free[x2]:
                    continue
                if segment_hits_wall(x, x2, w0, ww):
                    continue
                succ[i, ai] = _sidx(x2, v2, V)

    ef, ea = np.nonzero(succ >= 0)
    return _Graph(
        V=V,
        n_states=n,
        succ=succ,
        alive=alive,
        edges_from=ef.astype(np.int32),
        edges_to=succ[ef, ea].astype(np.int32),
    )


@lru_cache(maxsize=8192)
def _dist_to_goal(V: int, w0: int, ww: int, g: int):
    """Backward BFS: exact number of steps from every (x, v) to the parked goal (g, 0).

    Returns an int32 array with -1 for "cannot reach", or None if the parked goal state
    is not itself a legal state.
    """
    if not is_free_cell(g, w0, ww):
        return None
    G = _graph(V, w0, ww)
    gi = _sidx(g, 0, V)
    if not G.alive[gi]:
        return None

    dist = np.full(G.n_states, -1, dtype=np.int32)
    dist[gi] = 0
    frontier = np.zeros(G.n_states, dtype=bool)
    frontier[gi] = True
    level = 0
    ef, et = G.edges_from, G.edges_to
    while frontier.any():
        level += 1
        cand = ef[frontier[et]]
        cand = cand[dist[cand] < 0]
        if cand.size == 0:
            break
        frontier = np.zeros(G.n_states, dtype=bool)
        dist[cand] = level
        frontier[cand] = True
    return dist


def plan(ep: Episode):
    """Provably optimal action sequence taking (s, v0) to (g, 0), or None if unsolvable.

    Ties break deterministically by action-alphabet order, so an episode's ground-truth
    path is a function of the episode alone.
    """
    V, w0, ww = ep.V, ep.w0, ep.ww
    if not is_free_cell(ep.s, w0, ww) or abs(ep.v0) > V:
        return None
    dist = _dist_to_goal(V, w0, ww, ep.g)
    if dist is None:
        return None
    G = _graph(V, w0, ww)
    cur = _sidx(ep.s, ep.v0, V)
    if dist[cur] < 0:
        return None

    actions = []
    while dist[cur] > 0:
        target = dist[cur] - 1
        nxt = G.succ[cur]
        chosen = -1
        for ai in range(N_ACTIONS):
            j = int(nxt[ai])
            if j >= 0 and dist[j] == target:
                chosen = ai
                break
        assert chosen >= 0, "BFS distance field is inconsistent"
        actions.append(ACTIONS[chosen])
        cur = int(nxt[chosen])
    return actions


def positions_of(ep: Episode, actions: Sequence[int]):
    """Position sequence produced by `actions` from the episode start (no legality check)."""
    x, v = ep.s, ep.v0
    xs = [x]
    for a in actions:
        v = v + a
        x = x + v
        xs.append(x)
    return xs


def _forward_reachable(V: int, w0: int, ww: int, s: int, v0: int):
    """Boolean mask of every state reachable by VALID transitions from (s, v0)."""
    G = _graph(V, w0, ww)
    i0 = _sidx(s, v0, V)
    seen = np.zeros(G.n_states, dtype=bool)
    if not (0 <= i0 < G.n_states) or not G.alive[i0]:
        return seen
    seen[i0] = True
    frontier = seen.copy()
    ef, et = G.edges_from, G.edges_to
    while frontier.any():
        cand = et[frontier[ef]]
        cand = cand[~seen[cand]]
        if cand.size == 0:
            break
        frontier = np.zeros(G.n_states, dtype=bool)
        seen[cand] = True
        frontier[cand] = True
    return seen


def can_park(V: int, w0: int, ww: int, s: int, v0: int) -> bool:
    """Can the agent come to rest anywhere at all without ever breaking a rule?

    A state with `v = 0` can be held forever (action 0 is always legal there), so this is
    exactly "a fault-free infinite behaviour exists from (s, v0)". Episodes where this is
    False are doomed -- no behaviour is valid, so they can measure nothing about a model.
    """
    if abs(v0) > V or not is_free_cell(s, w0, ww):
        return False
    seen = _forward_reachable(V, w0, ww, s, v0)
    nv = 2 * V + 1
    return bool(seen.reshape(L, nv)[:, V].any())   # column V is v == 0


def wall_is_uncrossable(V: int, w0: int, ww: int) -> bool:
    """Exhaustive proof, for this rule setting, that no valid path crosses the wall.

    Flood-fills the valid-transition graph from every legal state strictly left of the
    wall and checks nothing right of the wall is ever reached. This is a statement about
    the graph itself, not a sampling argument.
    """
    if ww <= 0:
        return False
    G = _graph(V, w0, ww)
    nv = 2 * V + 1
    seen = np.zeros(G.n_states, dtype=bool)
    for x in range(0, w0):
        for v in range(-V, V + 1):
            i = _sidx(x, v, V)
            if G.alive[i]:
                seen[i] = True
    if not seen.any():
        return True
    frontier = seen.copy()
    ef, et = G.edges_from, G.edges_to
    while frontier.any():
        cand = et[frontier[ef]]
        cand = cand[~seen[cand]]
        if cand.size == 0:
            break
        frontier = np.zeros(G.n_states, dtype=bool)
        seen[cand] = True
        frontier[cand] = True
    xs = np.nonzero(seen)[0] // nv
    return bool(xs.max(initial=-1) < w0)


# ---------------------------------------------------------------------------
# Rollout. No masking, no clamping, no projection: validity is MEASURED, never imposed.
# ---------------------------------------------------------------------------

class RolloutResult(NamedTuple):
    xs: list               # positions visited, xs[0] == s
    vs: list               # velocities, vs[0] == v0
    acts: list             # actions actually emitted
    n_speed: int           # counts over the whole horizon (continuing mode)
    n_wall: int
    n_teleport: int
    n_steps: int           # steps actually taken
    first_fault_step: int  # 1-based index of the first fault, or -1 if none
    first_fault_kind: str  # "" if none
    all_steps_valid: bool
    reached_goal: bool     # x == g and v == 0 at some step, every PRECEDING step valid
    reach_step: int        # 1-based step at which that happened, or -1
    held_at_goal: bool     # reached and still parked at the goal at the horizon
    n_resets: int          # times the continuing mode put the agent back on the line


def rollout(ep: Episode, policy, horizon: int, continue_after_fault: bool = True) -> RolloutResult:
    """Free-running rollout of `policy` in the real environment.

    `policy(t, x, v, ep) -> a` is called with the TRUE state and whatever action it
    returns is applied verbatim. There is no legality mask, no clamp of v to the speed
    limit, and no projection of x back onto the line while the trajectory is still
    clean. A guardrail here would make every validity number vacuous.

    `continue_after_fault=True` rolls on to the horizon so that every violation of every
    kind is counted. To keep the state representable after a teleport the agent is put
    back on the line (x clipped, v zeroed) and the reset is counted; the teleport has
    already been recorded and the trajectory is already marked invalid, so no metric is
    softened by this.
    """
    V, w0, ww, g = ep.V, ep.w0, ep.ww, ep.g
    x, v = ep.s, ep.v0
    xs, vs, acts = [x], [v], []
    n_speed = n_wall = n_tel = n_resets = 0
    first_fault_step, first_fault_kind = -1, ""
    clean = True
    reached, reach_step = False, -1
    t = 0

    for t in range(1, horizon + 1):
        a = int(policy(t - 1, x, v, ep))
        acts.append(a)
        x2, v2, f = step_faults(x, v, a, V, w0, ww)
        n_speed += int(f[FAULT_SPEED])
        n_wall += int(f[FAULT_WALL])
        n_tel += int(f[FAULT_TELEPORT])
        if any(f.values()):
            if first_fault_step < 0:
                first_fault_step = t
                for k in FAULT_NAMES:   # fixed attribution order; each is also reported alone
                    if f[k]:
                        first_fault_kind = k
                        break
            clean = False
            if not continue_after_fault:
                xs.append(x2)
                vs.append(v2)
                break
            if f[FAULT_TELEPORT]:
                x2 = min(max(x2, 0), L - 1)
                v2 = 0
                n_resets += 1
        x, v = x2, v2
        xs.append(x)
        vs.append(v)
        if clean and not reached and x == g and v == 0:
            reached, reach_step = True, t

    held = bool(reached and clean and xs[-1] == g and vs[-1] == 0)
    return RolloutResult(
        xs=xs, vs=vs, acts=acts,
        n_speed=int(n_speed), n_wall=int(n_wall), n_teleport=int(n_tel), n_steps=t,
        first_fault_step=first_fault_step, first_fault_kind=first_fault_kind,
        all_steps_valid=clean, reached_goal=bool(reached), reach_step=reach_step,
        held_at_goal=held, n_resets=int(n_resets),
    )


def replay_policy(actions: Sequence[int], pad: int = 0):
    """A fixed action sequence as a policy, padded with `pad` past its end.

    The lookup tripwire uses this: a retrieved trajectory is REPLAYED in the real
    environment and scored by exactly the same metrics as a neural model.
    """
    seq = [int(a) for a in actions]

    def _p(t, x, v, ep):
        return seq[t] if t < len(seq) else pad

    return _p
