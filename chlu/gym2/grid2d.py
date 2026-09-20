"""GYM-2 environment: one agent with momentum on a 2-D grid, a speed limit and obstacles.

Why 2-D (w5/gym1 FINAL REPORT section 6.1). On a line an obstacle can never bend a valid
path -- proved there over 25,261,855 solvable configurations -- so composing a speed limit
with an obstacle is not expressible in 1-D at all. In 2-D a route has a *shape*, and the
shape can depend on the speed limit: a short twisty passage can only be crawled, while a
long open detour can be cruised, so the cheaper route flips as V grows. That flip is the
only thing that makes "compose the speed limit with the obstacle to find a different
route" a real question, and this file exists to make it exactly measurable.

Everything here is integer or exact-rational arithmetic. Every validity predicate has ZERO
tolerance, by construction, because an approximate validity check would make every
downstream refusal claim approximate. In particular the swept-cell test is an exact
segment/axis-aligned-box intersection computed once with `fractions.Fraction`, not a
sampled line trace.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Fixed geometry.
# ---------------------------------------------------------------------------

G = 20          # the grid is G x G cells; x, y in [0, G)
A_MAX = 1       # acceleration limit per axis
V_MAX_MAX = 4   # the largest speed limit any episode can carry

#: 9 actions: (ax, ay) with ax, ay in {-1, 0, +1}.
ACTIONS = tuple((ax, ay) for ax in range(-A_MAX, A_MAX + 1)
                for ay in range(-A_MAX, A_MAX + 1))
N_ACTIONS = len(ACTIONS)

FAULT_SPEED = "speed_limit_violation"
FAULT_OBSTACLE = "obstacle_collision"
FAULT_OUT = "out_of_bounds"
FAULT_NAMES = (FAULT_SPEED, FAULT_OBSTACLE, FAULT_OUT)

# A_MAX < V is what gives the agent a braking DISTANCE, and that is what makes the
# obstacle load-bearing rather than decorative. The action alphabet is deliberately WIDER
# than the rules permit: repeated (+1,+1) drives |v| past any V and drives the agent off
# the grid and through obstacles. A gym whose interface makes violations inexpressible
# cannot measure whether a model obeys a rule.


# ---------------------------------------------------------------------------
# Exact swept-cell geometry, computed once.
# ---------------------------------------------------------------------------

def _seg_box_intersects(px, py, qx, qy, bx0, by0, bx1, by1) -> bool:
    """Exact: does the CLOSED segment P->Q meet the CLOSED box [bx0,bx1]x[by0,by1]?

    Slab method in exact rational arithmetic. All inputs are integers (doubled
    coordinates), all comparisons are exact. No epsilon appears anywhere.
    """
    t0, t1 = Fraction(0), Fraction(1)
    for p, q, lo, hi in ((px, qx, bx0, bx1), (py, qy, by0, by1)):
        d = q - p
        if d == 0:
            if p < lo or p > hi:
                return False
            continue
        a = Fraction(lo - p, d)
        b = Fraction(hi - p, d)
        if a > b:
            a, b = b, a
        if a > t0:
            t0 = a
        if b < t1:
            t1 = b
        if t0 > t1:
            return False
    return True


def _sweep_offsets(dx: int, dy: int):
    """Cell offsets (relative to the start cell) swept by a move of (dx, dy).

    Cell (i, j) is the closed unit square [i, i+1] x [j, j+1]; the agent sits at its
    centre. A move is the closed segment between the two centres, and a cell is SWEPT iff
    that segment meets its closed square. Touching a corner counts.

    Consequence, stated rather than hidden: a pure diagonal step grazes the corners of the
    two orthogonal neighbours, so a diagonal move needs all four cells free. That is the
    standard "no corner cutting" rule and it is exact -- it is not a safety margin.
    """
    px, py = 1, 1                      # doubled centre of cell (0, 0)
    qx, qy = 2 * dx + 1, 2 * dy + 1    # doubled centre of cell (dx, dy)
    out = []
    lo_i, hi_i = min(0, dx), max(0, dx)
    lo_j, hi_j = min(0, dy), max(0, dy)
    for i in range(lo_i, hi_i + 1):
        for j in range(lo_j, hi_j + 1):
            if _seg_box_intersects(px, py, qx, qy, 2 * i, 2 * j, 2 * i + 2, 2 * j + 2):
                out.append((i, j))
    return tuple(out)


#: SWEEP[(dx, dy)] for every reachable per-step displacement. Built once, exactly.
SWEEP = {(dx, dy): _sweep_offsets(dx, dy)
         for dx in range(-V_MAX_MAX, V_MAX_MAX + 1)
         for dy in range(-V_MAX_MAX, V_MAX_MAX + 1)}


# ---------------------------------------------------------------------------
# Layouts.
# ---------------------------------------------------------------------------

class Layout(NamedTuple):
    """An obstacle field. `blocked` is a (G, G) bool array indexed [x, y]."""

    lid: int
    blocked: np.ndarray
    meta: dict            # generator parameters; carries the tunnel cell set and barrier band

    def free(self) -> np.ndarray:
        return ~self.blocked


class Episode(NamedTuple):
    """One task instance. `lid` indexes into the layout registry."""

    lid: int
    V: int
    sx: int
    sy: int
    vx0: int
    vy0: int
    gx: int
    gy: int

    def key(self) -> tuple:
        return (self.lid, self.V, self.sx, self.sy, self.vx0, self.vy0, self.gx, self.gy)


# The registry is module-level so that an Episode stays a small tuple of ints (cheap to
# hash, cheap to pickle) while the layouts themselves are shared.
_REGISTRY: list = []


def register_layout(blocked: np.ndarray, meta: dict) -> int:
    lid = len(_REGISTRY)
    _REGISTRY.append(Layout(lid, blocked.astype(bool), meta))
    return lid


def get_layout(lid: int) -> Layout:
    return _REGISTRY[lid]


def n_layouts() -> int:
    return len(_REGISTRY)


def reset_registry(entries=None):
    """Replace the registry (used when re-loading a saved split)."""
    global _REGISTRY
    _REGISTRY = []
    if entries:
        for blocked, meta in entries:
            register_layout(blocked, meta)
    _move_ok_cache.clear()
    _graph.cache_clear()
    _dist_field.cache_clear()


# ---------------------------------------------------------------------------
# Exact predicates.
# ---------------------------------------------------------------------------

def is_free_cell(lid: int, x: int, y: int) -> bool:
    if not (0 <= x < G and 0 <= y < G):
        return False
    return not bool(get_layout(lid).blocked[x, y])


def move_faults(lid: int, x: int, y: int, dx: int, dy: int):
    """(hits_obstacle, out_of_bounds) for a single displacement from (x, y).

    Out of bounds is checked on EVERY swept cell, not just the endpoint: leaving the grid
    mid-step and coming back would otherwise go unrecorded.
    """
    off = SWEEP.get((dx, dy))
    if off is None:                     # |v| beyond the precomputed range
        off = _sweep_offsets(dx, dy)
    blocked = get_layout(lid).blocked
    oob = False
    hit = False
    for oi, oj in off:
        cx, cy = x + oi, y + oj
        if not (0 <= cx < G and 0 <= cy < G):
            oob = True
        elif blocked[cx, cy]:
            hit = True
    return hit, oob


def step_faults(lid: int, x: int, y: int, vx: int, vy: int, a, V: int):
    """Apply acceleration `a = (ax, ay)`; return (x2, y2, vx2, vy2, faults).

    The three rule predicates are reported INDEPENDENTLY -- one step can break more than
    one -- and each is an exact integer test with no tolerance.
    """
    ax, ay = a
    vx2, vy2 = vx + ax, vy + ay
    x2, y2 = x + vx2, y + vy2
    hit, oob = move_faults(lid, x, y, vx2, vy2)
    faults = {
        FAULT_SPEED: abs(vx2) > V or abs(vy2) > V,
        FAULT_OBSTACLE: hit,
        FAULT_OUT: oob,
    }
    return x2, y2, vx2, vy2, faults


def step(lid, x, y, vx, vy, a, V):
    x2, y2, vx2, vy2, f = step_faults(lid, x, y, vx, vy, a, V)
    return x2, y2, vx2, vy2, not any(f.values())


# ---------------------------------------------------------------------------
# Vectorised legality of a displacement, per layout. Shared across speed limits.
# ---------------------------------------------------------------------------

_move_ok_cache: dict = {}


def move_ok_from_free(free: np.ndarray) -> np.ndarray:
    """(2Vm+1, 2Vm+1, G, G) bool: can the agent at (x, y) displace by (dx, dy) legally?

    "Legally" here means every swept cell exists and is free. The speed limit is NOT
    folded in -- it depends on V and is applied separately -- so one table serves every V.
    """
    Vm = V_MAX_MAX
    n = 2 * Vm + 1
    out = np.zeros((n, n, G, G), dtype=bool)
    for dx in range(-Vm, Vm + 1):
        for dy in range(-Vm, Vm + 1):
            ok = np.ones((G, G), dtype=bool)
            for oi, oj in SWEEP[(dx, dy)]:
                sh = np.zeros((G, G), dtype=bool)
                xs0, xs1 = max(0, -oi), min(G, G - oi)
                ys0, ys1 = max(0, -oj), min(G, G - oj)
                if xs0 < xs1 and ys0 < ys1:
                    sh[xs0:xs1, ys0:ys1] = free[xs0 + oi:xs1 + oi, ys0 + oj:ys1 + oj]
                ok &= sh
            out[dx + Vm, dy + Vm] = ok
    return out


def move_ok_table(lid: int) -> np.ndarray:
    t = _move_ok_cache.get(lid)
    if t is None:
        t = move_ok_from_free(get_layout(lid).free())
        _move_ok_cache[lid] = t
    return t


# ---------------------------------------------------------------------------
# Ground truth: exhaustive backward BFS over the (x, y, vx, vy) transition graph.
# ---------------------------------------------------------------------------

class _Graph(NamedTuple):
    V: int
    nv: int
    n_states: int
    succ: np.ndarray       # (n_states, 9) int32, -1 where the transition is invalid
    alive: np.ndarray      # (n_states,) bool
    edges_from: np.ndarray
    edges_to: np.ndarray


def sidx(V: int, x, y, vx, vy):
    nv = 2 * V + 1
    return ((x * G + y) * nv + (vx + V)) * nv + (vy + V)


@lru_cache(maxsize=96)
def _graph(lid: int, V: int, block_extra: tuple = ()) -> _Graph:
    """Valid-transition graph for one (layout, speed limit).

    `block_extra` is a tuple of extra blocked cells, used only by the route-restricted
    planners (O3 and the structural sweep). It participates in the cache key.
    """
    nv = 2 * V + 1
    nvv = nv * nv
    n = G * G * nvv

    if block_extra:
        blocked = get_layout(lid).blocked.copy()
        for cx, cy in block_extra:
            blocked[cx, cy] = True
        free = ~blocked
        mo = move_ok_from_free(free)      # not registered: derived graphs stay ephemeral
    else:
        mo = move_ok_table(lid)
        free = get_layout(lid).free()

    Vm = V_MAX_MAX
    xs, ys, vxs, vys = np.meshgrid(np.arange(G), np.arange(G),
                                   np.arange(-V, V + 1), np.arange(-V, V + 1),
                                   indexing="ij")
    xs, ys, vxs, vys = xs.ravel(), ys.ravel(), vxs.ravel(), vys.ravel()
    alive = free[xs, ys]

    succ = np.full((n, N_ACTIONS), -1, dtype=np.int32)
    for ai, (ax, ay) in enumerate(ACTIONS):
        vx2 = vxs + ax
        vy2 = vys + ay
        ok = (np.abs(vx2) <= V) & (np.abs(vy2) <= V) & alive
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        legal = mo[vx2[idx] + Vm, vy2[idx] + Vm, xs[idx], ys[idx]]
        idx = idx[legal]
        if idx.size == 0:
            continue
        x2 = xs[idx] + vx2[idx]
        y2 = ys[idx] + vy2[idx]
        succ[idx, ai] = (((x2 * G + y2) * nv + (vx2[idx] + V)) * nv + (vy2[idx] + V))

    ef, ea = np.nonzero(succ >= 0)
    return _Graph(V=V, nv=nv, n_states=n, succ=succ, alive=alive,
                  edges_from=ef.astype(np.int32),
                  edges_to=succ[ef, ea].astype(np.int32))


@lru_cache(maxsize=2048)
def _dist_field(lid: int, V: int, gx: int, gy: int, block_extra: tuple = ()):
    """Backward BFS: exact steps from every (x, y, vx, vy) to the parked goal.

    Returns an int32 array (-1 = unreachable), or None if the parked goal is not itself a
    legal state. Because ground truth is a complete BFS over the whole state graph,
    "optimal" here is provable, not "what a heuristic teacher happened to produce".
    """
    Gr = _graph(lid, V, block_extra)
    gi = sidx(V, gx, gy, 0, 0)
    if not (0 <= gi < Gr.n_states) or not Gr.alive[gi]:
        return None
    dist = np.full(Gr.n_states, -1, dtype=np.int32)
    dist[gi] = 0
    frontier = np.zeros(Gr.n_states, dtype=bool)
    frontier[gi] = True
    ef, et = Gr.edges_from, Gr.edges_to
    level = 0
    while frontier.any():
        level += 1
        cand = ef[frontier[et]]
        cand = cand[dist[cand] < 0]
        if cand.size == 0:
            break
        frontier = np.zeros(Gr.n_states, dtype=bool)
        dist[cand] = level
        frontier[cand] = True
    return dist


def plan(ep: Episode, block_extra: tuple = ()):
    """Provably optimal action sequence taking (s, v0) to (g, (0,0)), or None.

    Ties break deterministically by action-alphabet order, so an episode's ground-truth
    path is a function of the episode alone.
    """
    V = ep.V
    if not is_free_cell(ep.lid, ep.sx, ep.sy):
        return None
    if abs(ep.vx0) > V or abs(ep.vy0) > V:
        return None
    dist = _dist_field(ep.lid, V, ep.gx, ep.gy, block_extra)
    if dist is None:
        return None
    Gr = _graph(ep.lid, V, block_extra)
    cur = sidx(V, ep.sx, ep.sy, ep.vx0, ep.vy0)
    if dist[cur] < 0:
        return None
    acts = []
    while dist[cur] > 0:
        target = dist[cur] - 1
        nxt = Gr.succ[cur]
        chosen = -1
        for ai in range(N_ACTIONS):
            j = int(nxt[ai])
            if j >= 0 and dist[j] == target:
                chosen = ai
                break
        assert chosen >= 0, "BFS distance field is inconsistent"
        acts.append(ACTIONS[chosen])
        cur = int(nxt[chosen])
    return acts


def opt_cost(ep: Episode, block_extra: tuple = ()) -> int:
    """Exact optimal number of steps, or -1 if unreachable."""
    dist = _dist_field(ep.lid, ep.V, ep.gx, ep.gy, block_extra)
    if dist is None:
        return -1
    return int(dist[sidx(ep.V, ep.sx, ep.sy, ep.vx0, ep.vy0)])


def positions_of(ep: Episode, actions: Sequence):
    """Position sequence produced by `actions` (no legality check)."""
    x, y, vx, vy = ep.sx, ep.sy, ep.vx0, ep.vy0
    out = [(x, y)]
    for ax, ay in actions:
        vx += ax
        vy += ay
        x += vx
        y += vy
        out.append((x, y))
    return out


def _forward_reachable(lid: int, V: int, sx, sy, vx0, vy0):
    Gr = _graph(lid, V)
    i0 = sidx(V, sx, sy, vx0, vy0)
    seen = np.zeros(Gr.n_states, dtype=bool)
    if not (0 <= i0 < Gr.n_states) or not Gr.alive[i0]:
        return seen
    seen[i0] = True
    frontier = seen.copy()
    ef, et = Gr.edges_from, Gr.edges_to
    while frontier.any():
        cand = et[frontier[ef]]
        cand = cand[~seen[cand]]
        if cand.size == 0:
            break
        frontier = np.zeros(Gr.n_states, dtype=bool)
        seen[cand] = True
        frontier[cand] = True
    return seen


def can_park(lid: int, V: int, sx, sy, vx0, vy0) -> bool:
    """Does a fault-free INFINITE behaviour exist from this state?

    A state with v == (0, 0) can be held forever, so this is exactly that question.
    Episodes where it is False are doomed and can measure nothing about a model.
    """
    if abs(vx0) > V or abs(vy0) > V or not is_free_cell(lid, sx, sy):
        return False
    seen = _forward_reachable(lid, V, sx, sy, vx0, vy0)
    nv = 2 * V + 1
    return bool(seen.reshape(G * G, nv, nv)[:, V, V].any())


# ---------------------------------------------------------------------------
# Rollout. No masking, no clamping, no projection: validity is MEASURED, never imposed.
# ---------------------------------------------------------------------------

class RolloutResult(NamedTuple):
    xs: list
    ys: list
    vxs: list
    vys: list
    acts: list
    n_speed: int
    n_obstacle: int
    n_out: int
    n_steps: int
    first_fault_step: int
    first_fault_kind: str
    all_steps_valid: bool
    reached_goal: bool
    reach_step: int
    held_at_goal: bool
    n_resets: int


def rollout(ep: Episode, policy, horizon: int, continue_after_fault: bool = True):
    """Free-running rollout of `policy` in the real environment.

    `policy(t, x, y, vx, vy, ep) -> (ax, ay)` is called with the TRUE state and whatever
    it returns is applied verbatim. There is no legality mask, no clamp of v to the speed
    limit, and no projection of the agent back onto the grid while the trajectory is still
    clean. A guardrail here would make every validity number vacuous.

    `continue_after_fault=True` rolls on to the horizon so every violation of every kind is
    counted. To keep the state representable after leaving the grid the agent is put back
    (position clipped, velocity zeroed) and the reset is counted; the excursion has already
    been recorded and the trajectory is already marked invalid, so no metric is softened.
    """
    lid, V, gx, gy = ep.lid, ep.V, ep.gx, ep.gy
    x, y, vx, vy = ep.sx, ep.sy, ep.vx0, ep.vy0
    xs, ys, vxs, vys, acts = [x], [y], [vx], [vy], []
    n_speed = n_obs = n_out = n_resets = 0
    first_fault_step, first_fault_kind = -1, ""
    clean = True
    reached, reach_step = False, -1
    t = 0

    for t in range(1, horizon + 1):
        a = policy(t - 1, x, y, vx, vy, ep)
        ax, ay = int(a[0]), int(a[1])
        acts.append((ax, ay))
        x2, y2, vx2, vy2, f = step_faults(lid, x, y, vx, vy, (ax, ay), V)
        n_speed += int(f[FAULT_SPEED])
        n_obs += int(f[FAULT_OBSTACLE])
        n_out += int(f[FAULT_OUT])
        if any(f.values()):
            if first_fault_step < 0:
                first_fault_step = t
                for k in FAULT_NAMES:
                    if f[k]:
                        first_fault_kind = k
                        break
            clean = False
            if not continue_after_fault:
                xs.append(x2); ys.append(y2); vxs.append(vx2); vys.append(vy2)
                break
            if f[FAULT_OUT] or not (0 <= x2 < G and 0 <= y2 < G):
                x2 = min(max(x2, 0), G - 1)
                y2 = min(max(y2, 0), G - 1)
                vx2 = vy2 = 0
                n_resets += 1
        x, y, vx, vy = x2, y2, vx2, vy2
        xs.append(x); ys.append(y); vxs.append(vx); vys.append(vy)
        if clean and not reached and x == gx and y == gy and vx == 0 and vy == 0:
            reached, reach_step = True, t

    held = bool(reached and clean and xs[-1] == gx and ys[-1] == gy
                and vxs[-1] == 0 and vys[-1] == 0)
    return RolloutResult(
        xs=xs, ys=ys, vxs=vxs, vys=vys, acts=acts,
        n_speed=int(n_speed), n_obstacle=int(n_obs), n_out=int(n_out), n_steps=t,
        first_fault_step=first_fault_step, first_fault_kind=first_fault_kind,
        all_steps_valid=clean, reached_goal=bool(reached), reach_step=reach_step,
        held_at_goal=held, n_resets=int(n_resets),
    )


def replay_policy(actions: Sequence, pad=(0, 0)):
    """A fixed action sequence as a policy, padded past its end.

    The lookup tripwire uses this: a retrieved trajectory is REPLAYED in the real
    environment and scored by exactly the same metrics as a neural model.
    """
    seq = [(int(a[0]), int(a[1])) for a in actions]

    def _p(t, x, y, vx, vy, ep):
        return seq[t] if t < len(seq) else pad

    return _p


def verify_unmasked_rollout():
    """Prove the environment applies illegal actions verbatim instead of repairing them.

    Inherited from w5/gym1 `train.verify_unmasked_rollout`, ported to 2-D. Drive a
    deliberately illegal policy -- constant (+1, +1), which must break the speed limit and
    then run off the grid -- and check that violations are recorded, that the recorded
    states equal raw integer arithmetic up to the first fault, and that the velocity is
    allowed to exceed the speed limit rather than being capped.
    """
    blocked = np.zeros((G, G), dtype=bool)
    blocked[10, 10] = True
    lid = register_layout(blocked, {"purpose": "verify_unmasked_rollout"})
    ep = Episode(lid=lid, V=1, sx=2, sy=2, vx0=0, vy0=0, gx=5, gy=5)
    r = rollout(ep, lambda t, x, y, vx, vy, e: (1, 1), horizon=12,
                continue_after_fault=False)
    x, y, vx, vy = ep.sx, ep.sy, 0, 0
    tx, ty = [x], [y]
    for _ in range(len(r.acts)):
        vx += 1
        vy += 1
        x += vx
        y += vy
        tx.append(x)
        ty.append(y)
    maxv = max(max(abs(u) for u in r.vxs), max(abs(u) for u in r.vys))
    return {
        "speed_violations_recorded": r.n_speed > 0,
        "out_of_bounds_recorded": r.n_out > 0,
        "max_abs_velocity_reached": int(maxv),
        "speed_limit": ep.V,
        "velocity_exceeded_limit": bool(maxv > ep.V),
        "states_match_raw_arithmetic": (r.xs == tx and r.ys == ty),
        "first_fault_step": r.first_fault_step,
        "first_fault_kind": r.first_fault_kind,
        "ok": bool(r.n_speed > 0 and maxv > ep.V and r.xs == tx and r.ys == ty),
    }
