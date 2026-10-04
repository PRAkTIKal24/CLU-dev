"""maze colour."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

WALL, FLOOR = (0, 1)
UNTAKEN, CARRIED, USED = (0, 1, 2)
MOVES = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
MOVE_ORDER = "NESW"


@dataclass(frozen=True)
class Instance:
    """One episode.  Everything is small ints; nothing here is approximate."""

    seed: int
    grid: np.ndarray
    tint: np.ndarray
    door: np.ndarray
    key_at: tuple
    start: tuple
    goal: tuple
    n_colours: int
    budget: int
    opt_len: int
    opt_len_a_only: int

    def _replace_tint(self, tint):
        """A view of this instance with a different tint array (adversarial placement)."""
        return Instance(
            self.seed,
            self.grid,
            tint,
            self.door,
            self.key_at,
            self.start,
            self.goal,
            self.n_colours,
            self.budget,
            self.opt_len,
            self.opt_len_a_only,
        )


def _pocket(ks):
    """Colour held (1-based), or 0 for an empty pocket."""
    for i, v in enumerate(ks):
        if v == CARRIED:
            return i + 1
    return 0


def _carve_maze(h, w, rng):
    """Randomised-DFS perfect maze on a (2h+1) x (2w+1) grid."""
    H, W = (2 * h + 1, 2 * w + 1)
    g = np.zeros((H, W), dtype=np.uint8)
    seen = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    stack = [start]
    seen[start] = True
    g[2 * start[0] + 1, 2 * start[1] + 1] = FLOOR
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in MOVES.values():
            nr, nc = (r + dr, c + dc)
            if 0 <= nr < h and 0 <= nc < w and (not seen[nr, nc]):
                nbrs.append((nr, nc, dr, dc))
        if not nbrs:
            stack.pop()
            continue
        nr, nc, dr, dc = nbrs[int(rng.integers(len(nbrs)))]
        g[2 * r + 1 + dr, 2 * c + 1 + dc] = FLOOR
        g[2 * nr + 1, 2 * nc + 1] = FLOOR
        seen[nr, nc] = True
        stack.append((nr, nc))
    return g


def _braid(g, rng, frac):
    """Knock out a fraction of the interior walls, in place, to create loops.

    A perfect maze has exactly one route between any two cells, which would make
    "find a DIFFERENT route" vacuous.  Braiding is what gives route choice.
    """
    H, W = g.shape
    cands = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if g[r, c] != WALL:
                continue
            if (r % 2 == 1) != (c % 2 == 1):
                a = g[r - 1, c] == FLOOR and g[r + 1, c] == FLOOR
                b = g[r, c - 1] == FLOOR and g[r, c + 1] == FLOOR
                if a or b:
                    cands.append((r, c))
    rng.shuffle(cands)
    for r, c in cands[: int(frac * len(cands))]:
        g[r, c] = FLOOR


def _floor_cells(g):
    return [tuple(map(int, p)) for p in np.argwhere(g == FLOOR)]


def solve(
    inst, rule_a=True, rule_b=True, colour_blind=False, tint_local_only=False, key_order=None
):
    g, tint, door = (inst.grid, inst.tint, inst.door)
    H, W = g.shape
    k = inst.n_colours
    key_cell = {inst.key_at[i]: i + 1 for i in range(k)}
    near_door = None
    if tint_local_only:
        near_door = np.zeros(door.shape, dtype=bool)
        for r, c in np.argwhere(door > 0):
            for dr, dc in MOVES.values():
                if 0 <= r + dr < H and 0 <= c + dc < W:
                    near_door[r + dr, c + dc] = True
            near_door[r, c] = True
    start = (inst.start, tuple([UNTAKEN] * k))
    prev = {}
    seen = {start}
    q = deque([start])
    goal_state = None
    while q:
        cur = q.popleft()
        (r, c), ks = cur
        if (r, c) == inst.goal:
            goal_state = cur
            break
        pk = _pocket(ks)
        if (r, c) in key_cell:
            kc = key_cell[r, c]
            ordered_ok = True
            if key_order is not None:
                taken = sum((1 for v in ks if v != UNTAKEN))
                ordered_ok = taken < len(key_order) and key_order[taken] == kc
            if ks[kc - 1] == UNTAKEN and pk == 0 and ordered_ok:
                nks = list(ks)
                nks[kc - 1] = CARRIED
                nxt = ((r, c), tuple(nks))
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = (cur, "P")
                    q.append(nxt)
        for a in MOVE_ORDER:
            dr, dc = MOVES[a]
            nr, nc = (r + dr, c + dc)
            if not (0 <= nr < H and 0 <= nc < W) or g[nr, nc] == WALL:
                continue
            nks = list(ks)
            dcol = int(door[nr, nc])
            if dcol and rule_a:
                if ks[dcol - 1] == USED:
                    pass
                elif colour_blind:
                    if pk == 0:
                        continue
                    nks[pk - 1] = USED
                elif pk == dcol:
                    nks[dcol - 1] = USED
                else:
                    continue
            npk = _pocket(tuple(nks))
            tcol = int(tint[nr, nc])
            if tcol and rule_b and (npk != 0):
                if not tint_local_only or near_door[nr, nc]:
                    if colour_blind or npk == tcol:
                        continue
            nxt = ((nr, nc), tuple(nks))
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = (cur, a)
                q.append(nxt)
    if goal_state is None:
        return (None, None)
    route = []
    cur = goal_state
    while cur != start:
        cur, a = prev[cur]
        route.append(a)
    return (len(route), "".join(reversed(route)))


def _poison(inst, tint, target_kwargs, rng, max_poison):
    """Tint cells so that ONE named blind planner's route becomes ILLEGAL.

    Runs the blind planner named by `target_kwargs`, replays its route, and for every
    cell it enters while carrying key c, sets that cell's tint to c -- which is exactly
    the condition RULE B refuses.  Each addition is kept only if the instance stays
    feasible under RULES A+B, so the oracle is never destroyed.

    This is the alternative to the acceptance filter: instead of DISCARDING instances
    where the two rules happen not to interact, it CONSTRUCTS the interaction in every
    instance.  Nothing is thrown away, so the distribution itself carries the
    composition rather than a rejection step.  Returns the number of cells poisoned.
    """
    L, route = solve(inst._replace_tint(tint), **target_kwargs)
    if route is None:
        return 0
    key_cell = {inst.key_at[i]: i + 1 for i in range(inst.n_colours)}
    protected = {inst.start, inst.goal} | set(inst.key_at)
    cands, pos, ks = ([], inst.start, [UNTAKEN] * inst.n_colours)
    for a in route:
        if a == "P":
            kc = key_cell.get(pos, 0)
            if kc:
                ks[kc - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        npos = (pos[0] + dr, pos[1] + dc)
        dcol = int(inst.door[npos])
        if dcol and ks[dcol - 1] == CARRIED:
            ks[dcol - 1] = USED
        else:
            pk = _pocket(tuple(ks))
            if pk and npos not in protected and (tint[npos] == 0) and (dcol == 0):
                cands.append((npos, pk))
        pos = npos
    rng.shuffle(cands)
    used = 0
    for (r, c), colour in cands:
        if used >= max_poison:
            break
        if tint[r, c] != 0:
            continue
        tint[r, c] = colour
        if solve(inst._replace_tint(tint))[0] is None:
            tint[r, c] = 0
        else:
            used += 1
    return used


def generate(
    seed,
    h=7,
    w=7,
    n_colours=4,
    braid=0.4,
    p_tint=0.22,
    slack=0.1,
    max_tries=400,
    require_bend=False,
    adversarial=2,
    max_poison=8,
):
    """Return an accepted Instance, or None if max_tries is exhausted.

    THE DEFAULTS ARE THE ADOPTED CONFIGURATION: braid=0.40, adversarial=2,
    require_bend=False.  The interaction between the two rules lives in the
    DISTRIBUTION (adversarial tint placement, see `_poison`), not in a rejection
    filter, so no instance is discarded for failing to interact.  Measured at these
    defaults, n=500, k=4: composition-blind 0.4980, fragment-lookup 0.0000, oracle
    1.0000.  With uniform tinting instead, the same unfiltered generator gives a
    composition-blind 0.8940-0.9100 -- that is the number this default exists to avoid.

    Unconditional acceptance conditions, both statements about the INSTANCE:
      * the goal is unreachable with no key at all (the doors are load-bearing);
      * the instance is feasible under RULES A+B.

    `require_bend=True` additionally keeps only instances where
    opt_len(A+B) > opt_len(A alone) -- the research question turned into a filter.
    It is OFF by default: adversarial placement already makes rule B bend the route in
    ~78% of instances without discarding the rest.

    `adversarial=R` runs R rounds of poisoning: round 0 targets the rule-A-only
    planner, later rounds the factored planner.  It needs route multiplicity to work,
    which is why braid defaults to 0.40 -- at braid 0.12 the key->door corridor is
    usually unique, so ~8 of 9 poison candidates revert to preserve feasibility.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        g = _carve_maze(h, w, rng)
        _braid(g, rng, braid)
        cells = _floor_cells(g)
        if len(cells) < 6 * n_colours:
            continue
        pick = rng.permutation(len(cells))
        chosen = [cells[i] for i in pick[: 2 + 2 * n_colours]]
        start, goal = (chosen[0], chosen[1])
        doors = chosen[2 : 2 + n_colours]
        keys = chosen[2 + n_colours :]
        if start == goal:
            continue
        door = np.zeros(g.shape, dtype=np.int8)
        for i, (r, c) in enumerate(doors):
            door[r, c] = i + 1
        tint = np.zeros(g.shape, dtype=np.int8)
        for r, c in cells:
            if rng.random() < p_tint:
                tint[r, c] = int(rng.integers(1, n_colours + 1))
        for r, c in [start, goal] + doors + keys:
            tint[r, c] = 0
        inst = Instance(seed, g, tint, door, tuple(keys), start, goal, n_colours, 10**9, 0, 0)
        if _reachable_without_keys(inst):
            continue
        if solve(inst)[0] is None:
            continue
        for r_i in range(adversarial):
            target = dict(rule_b=False) if r_i == 0 else dict(tint_local_only=True)
            _poison(inst, tint, target, rng, max_poison)
        lj, _ = solve(inst, rule_a=True, rule_b=True)
        if lj is None:
            continue
        la, _ = solve(inst, rule_a=True, rule_b=False)
        if la is None:
            continue
        if require_bend and la >= lj:
            continue
        budget = lj + int(np.ceil(slack * lj))
        return Instance(seed, g, tint, door, tuple(keys), start, goal, n_colours, budget, lj, la)
    return None


def _reachable_without_keys(inst):
    g, door = (inst.grid, inst.door)
    H, W = g.shape
    q = deque([inst.start])
    seen = {inst.start}
    while q:
        r, c = q.popleft()
        if (r, c) == inst.goal:
            return True
        for dr, dc in MOVES.values():
            nr, nc = (r + dr, c + dc)
            if (
                0 <= nr < H
                and 0 <= nc < W
                and (g[nr, nc] == FLOOR)
                and (door[nr, nc] == 0)
                and ((nr, nc) not in seen)
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return False
