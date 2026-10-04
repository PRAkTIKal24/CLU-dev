"""CMAZE-1 -- colour-keyed maze with an exact judge.

Reference implementation for the wave-8 env-scout spoke.  Built because nothing
off the shelf satisfied the adoption law (see ENV.md): every shipped door-key
environment enforces key/door colour matching in its engine but never makes it
load-bearing in its task generator.

Design, one line per requirement:
  (a) random maze per episode  -- randomised-DFS perfect maze + braiding, fresh per seed;
  (b) legality composing across agent x colour x door --
        RULE A  a door cell of colour c is IMPASSABLE unless the pocket holds key c;
        RULE B  a floor cell tinted colour c is IMPASSABLE while the pocket holds key c;
      both render a route ILLEGAL, not merely longer, and both are indexed by the
      pocket, so the legal-move relation is a function of (cell, pocket, door state);
  (c) budget -- an integer step budget B; a route longer than B is rejected;
  (d) exact validity -- the judge is integer replay.  No float, no tolerance, no metric.

The pocket holds at most one key and a key is CONSUMED by the door it opens.  That
single choice is what makes rule B bite: while you carry key c you are locked out of
every c-tinted cell and you cannot put the key down, so the order in which you commit
to colours changes which routes exist at all.

State is (cell, keystate) with keystate[c] in {UNTAKEN, CARRIED, USED} and at most one
CARRIED.  A door is open iff its key is USED, so the door mask is not separate state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- constants
WALL, FLOOR = 0, 1
UNTAKEN, CARRIED, USED = 0, 1, 2
MOVES = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
MOVE_ORDER = "NESW"
ACTIONS = "NESWP"  # P = pick up the key on the current cell


@dataclass(frozen=True)
class Instance:
    """One episode.  Everything is small ints; nothing here is approximate."""

    seed: int
    grid: np.ndarray          # (H, W) uint8, WALL/FLOOR
    tint: np.ndarray          # (H, W) int8, 0 = untinted, else colour 1..k
    door: np.ndarray          # (H, W) int8, 0 = not a door, else colour 1..k
    key_at: tuple             # key_at[c-1] = (r, c) home cell of key colour c
    start: tuple
    goal: tuple
    n_colours: int
    budget: int
    opt_len: int              # exact optimum under RULES A+B
    opt_len_a_only: int       # exact optimum under RULE A alone (rule B dropped)

    def _replace_tint(self, tint):
        """A view of this instance with a different tint array (adversarial placement)."""
        return Instance(self.seed, self.grid, tint, self.door, self.key_at, self.start,
                        self.goal, self.n_colours, self.budget, self.opt_len,
                        self.opt_len_a_only)


# ---------------------------------------------------------------- key state
def _keystates(k):
    """All keystate tuples with at most one CARRIED entry."""
    out = []

    def rec(prefix):
        if len(prefix) == k:
            if sum(1 for v in prefix if v == CARRIED) <= 1:
                out.append(tuple(prefix))
            return
        for v in (UNTAKEN, CARRIED, USED):
            rec(prefix + [v])

    rec([])
    return out


def _pocket(ks):
    """Colour held (1-based), or 0 for an empty pocket."""
    for i, v in enumerate(ks):
        if v == CARRIED:
            return i + 1
    return 0


# ---------------------------------------------------------------- generation
def _carve_maze(h, w, rng):
    """Randomised-DFS perfect maze on a (2h+1) x (2w+1) grid."""
    H, W = 2 * h + 1, 2 * w + 1
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
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
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
            if (r % 2 == 1) != (c % 2 == 1):  # a wall between two cell centres
                a = (g[r - 1, c] == FLOOR and g[r + 1, c] == FLOOR)
                b = (g[r, c - 1] == FLOOR and g[r, c + 1] == FLOOR)
                if a or b:
                    cands.append((r, c))
    rng.shuffle(cands)
    for r, c in cands[: int(frac * len(cands))]:
        g[r, c] = FLOOR


def _floor_cells(g):
    return [tuple(map(int, p)) for p in np.argwhere(g == FLOOR)]


# ---------------------------------------------------------------- solver
def solve(inst, rule_a=True, rule_b=True, colour_blind=False, tint_local_only=False,
          key_order=None):
    """Exact BFS over (cell, keystate).  Returns (length, route) or (None, None).

    The flags are how a *blind* planner is expressed: a planner blind to part of
    the composition is exactly a planner that searches a relaxed legality
    relation.  The judge is never relaxed.

      rule_a=False        doors are not there at all (walk through any door)
      rule_b=False        tints are ignored
      colour_blind=True   any key opens any door and a tint blocks any carried key
      tint_local_only     rule B is applied only on cells orthogonally adjacent to a
                          door -- the classic "the rules only interact locally" factoring
      key_order           if given, keys may only be picked up in this colour order;
                          this is how a planner that has committed to an ordering
                          (blind to the fact that rule B changes which orderings
                          survive) is expressed
    """
    g, tint, door = inst.grid, inst.tint, inst.door
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
        # pick up the key on this cell
        if (r, c) in key_cell:
            kc = key_cell[(r, c)]
            ordered_ok = True
            if key_order is not None:
                taken = sum(1 for v in ks if v != UNTAKEN)
                ordered_ok = taken < len(key_order) and key_order[taken] == kc
            if ks[kc - 1] == UNTAKEN and pk == 0 and ordered_ok:
                nks = list(ks)
                nks[kc - 1] = CARRIED
                nxt = ((r, c), tuple(nks))
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = (cur, "P")
                    q.append(nxt)
        # moves
        for a in MOVE_ORDER:
            dr, dc = MOVES[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or g[nr, nc] == WALL:
                continue
            nks = list(ks)
            dcol = int(door[nr, nc])
            if dcol and rule_a:
                if ks[dcol - 1] == USED:
                    pass                                  # already open
                elif colour_blind:
                    if pk == 0:
                        continue
                    nks[pk - 1] = USED                    # any key opens any door
                elif pk == dcol:
                    nks[dcol - 1] = USED
                else:
                    continue
            npk = _pocket(tuple(nks))
            tcol = int(tint[nr, nc])
            if tcol and rule_b and npk != 0:
                if (not tint_local_only) or near_door[nr, nc]:
                    if colour_blind or npk == tcol:
                        continue
            nxt = ((nr, nc), tuple(nks))
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = (cur, a)
                q.append(nxt)
    if goal_state is None:
        return None, None
    route = []
    cur = goal_state
    while cur != start:
        cur, a = prev[cur]
        route.append(a)
    return len(route), "".join(reversed(route))


# ---------------------------------------------------------------- exact judge
def judge(inst, route):
    """Exact.  Integer replay of RULES A+B.  Returns (valid, reason, steps).

    There is no tolerance and no metric anywhere in this function; every comparison
    is between small integers.  A route is valid iff every action is legal, the goal
    is reached, and the length is within the budget.
    """
    if route is None:
        return False, "no-route", 0
    if len(route) > inst.budget:
        return False, "over-budget(%d>%d)" % (len(route), inst.budget), len(route)
    g, tint, door = inst.grid, inst.tint, inst.door
    H, W = g.shape
    k = inst.n_colours
    key_cell = {inst.key_at[i]: i + 1 for i in range(k)}
    ks = [UNTAKEN] * k
    pos = inst.start
    for t, a in enumerate(route):
        if a == "P":
            kc = key_cell.get(pos, 0)
            if kc == 0:
                return False, "pickup-on-empty-cell@%d" % t, t
            if ks[kc - 1] != UNTAKEN:
                return False, "key-not-there@%d" % t, t
            if _pocket(tuple(ks)) != 0:
                return False, "pocket-full@%d" % t, t
            ks[kc - 1] = CARRIED
            continue
        if a not in MOVES:
            return False, "bad-action(%s)@%d" % (a, t), t
        dr, dc = MOVES[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < H and 0 <= nc < W) or g[nr, nc] == WALL:
            return False, "into-wall@%d" % t, t
        dcol = int(door[nr, nc])
        if dcol and ks[dcol - 1] != USED:
            if ks[dcol - 1] != CARRIED:
                return False, "locked-door-c%d@%d" % (dcol, t), t          # RULE A
            ks[dcol - 1] = USED
        pk = _pocket(tuple(ks))
        tcol = int(tint[nr, nc])
        if tcol and pk == tcol:
            return False, "forbidden-tint-c%d@%d" % (tcol, t), t            # RULE B
        pos = (nr, nc)
        if pos == inst.goal:
            return True, "goal", t + 1
    return False, "no-goal", len(route)



# ---------------------------------------------------------------- adversarial tints
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
    cands, pos, ks = [], inst.start, [UNTAKEN] * inst.n_colours
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
            if pk and npos not in protected and tint[npos] == 0 and dcol == 0:
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
            tint[r, c] = 0                      # would destroy the oracle -- revert
        else:
            used += 1
    return used


# ---------------------------------------------------------------- generator
def generate(seed, h=7, w=7, n_colours=4, braid=0.40, p_tint=0.22,
             slack=0.10, max_tries=400, require_bend=False, adversarial=2,
             max_poison=8):
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
        start, goal = chosen[0], chosen[1]
        doors = chosen[2: 2 + n_colours]
        keys = chosen[2 + n_colours:]
        if start == goal:
            continue
        door = np.zeros(g.shape, dtype=np.int8)
        for i, (r, c) in enumerate(doors):
            door[r, c] = i + 1
        tint = np.zeros(g.shape, dtype=np.int8)
        for (r, c) in cells:
            if rng.random() < p_tint:
                tint[r, c] = int(rng.integers(1, n_colours + 1))
        for (r, c) in [start, goal] + doors + keys:
            tint[r, c] = 0
        inst = Instance(seed, g, tint, door, tuple(keys), start, goal,
                        n_colours, 10 ** 9, 0, 0)
        if _reachable_without_keys(inst):
            continue                      # doors are not load-bearing
        if solve(inst)[0] is None:
            continue                      # infeasible before poisoning
        for r_i in range(adversarial):
            target = dict(rule_b=False) if r_i == 0 else dict(tint_local_only=True)
            _poison(inst, tint, target, rng, max_poison)
        lj, _ = solve(inst, rule_a=True, rule_b=True)
        if lj is None:
            continue                      # infeasible
        la, _ = solve(inst, rule_a=True, rule_b=False)
        if la is None:
            continue
        if require_bend and la >= lj:
            continue                      # rule B did not bend the route
        budget = lj + int(np.ceil(slack * lj))
        return Instance(seed, g, tint, door, tuple(keys), start, goal,
                        n_colours, budget, lj, la)
    return None


def _reachable_without_keys(inst):
    g, door = inst.grid, inst.door
    H, W = g.shape
    q = deque([inst.start])
    seen = {inst.start}
    while q:
        r, c = q.popleft()
        if (r, c) == inst.goal:
            return True
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == FLOOR \
                    and door[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False
