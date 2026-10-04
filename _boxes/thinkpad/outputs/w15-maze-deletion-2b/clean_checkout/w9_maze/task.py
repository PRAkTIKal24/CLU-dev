"""CMAZE-2 -- the REQUEST layer over CMAZE-1.

Why a layer at all
------------------
``w8_envscout.maze_colour`` generates ONE (start, goal) pair per maze.  The wave-9
RESULT rows are about a *request*: ``(maze, agent, target, budget)``, with rows for an
unseen agent-target pair on a seen maze and for two agents to one target.  That needs
several named starts and several named targets **inside one maze**, which the wave-8
generator does not emit.  Everything else is used unchanged and nothing in the engine
or the judge is touched: ``generate``, ``solve`` and ``judge`` are imported, not copied.

What is preserved from the ADOPTED configuration
------------------------------------------------
* the generator defaults (h=w=7, n_colours=4, braid=0.40, p_tint=0.22, adversarial=2,
  no acceptance filter) -- ``generate()`` is called with its own defaults;
* the two adoption conditions, now required PER PAIR rather than for one pair: the
  target is unreachable with no key at all (the doors are load-bearing), and the pair is
  feasible under RULES A+B;
* the adversarial tint placement, generalised: a tint is added only if EVERY kept pair
  stays feasible, so the interaction is constructed for all pairs at once rather than
  by discarding instances.

The environment's published columns (blind-sequential 0.4980, fragment-lookup
0.0000/0.0000, rule B bends 81.6 %) were measured on the ONE-PAIR distribution.  They
are not assumed to carry: ``w9_maze.envcols`` re-measures all three on THIS
distribution, and both sets of numbers are printed on every row.

The budget is an ABSOLUTE integer step budget drawn from a grid of words, so a budget
BETWEEN two trained words is a real number the certificate can use, and a request whose
optimum exceeds its budget is simply *not written* -- no impossible example is ever seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from w8_envscout.maze_colour import (FLOOR, MOVES, UNTAKEN, Instance, _pocket,
                                     _reachable_without_keys, generate, solve)

#: The target words.  A target is a named, coloured goal cell; the word is what the
#: address book grounds and what "delete the green target" deletes.
TARGET_WORDS = ("red", "green", "blue", "yellow", "cyan", "magenta")
#: The agent words.  An agent is a named start cell inside a maze.
AGENT_WORDS = ("a0", "a1", "a2", "a3", "a4", "a5")
#: The budget words: ABSOLUTE step budgets.  Trained values.
BUDGET_GRID = (30, 45, 60)
#: Strictly between two trained words -- this is what row 4 asks for.
BUDGET_INTERP = (38, 53)


@dataclass
class Site:
    """One maze with several named agents and several named targets."""

    seed: int
    grid: np.ndarray
    tint: np.ndarray
    door: np.ndarray
    key_at: tuple
    agents: tuple            # agents[i] = cell of agent word i
    targets: tuple           # targets[j] = cell of target word j
    n_colours: int
    pairs: tuple             # kept (i, j) pairs
    opt: dict                # (i, j) -> optimal length under RULES A+B
    opt_a: dict              # (i, j) -> optimal length under RULE A alone
    route: dict              # (i, j) -> the exact optimal route string
    n_poisoned: int = 0
    #: a target word whose cell has been sealed behind walls, if any.  Its pairs are
    #: IMPOSSIBLE, are never written, and are the second refusal condition the target
    #: file names ("the target is walled").
    walled_target: int = -1
    walled_pairs: tuple = ()

    def instance(self, i: int, j: int, budget: int) -> Instance:
        """The CMAZE-1 Instance for one (agent, target, budget) -- judged unchanged."""
        #: a WALLED pair has no optimum -- there is no route -- so the instance
        #: carries 0 and the judge simply never accepts anything for it.
        return Instance(self.seed, self.grid, self.tint, self.door, self.key_at,
                        self.agents[i], self.targets[j], self.n_colours, int(budget),
                        int(self.opt.get((i, j), 0)), int(self.opt_a.get((i, j), 0)))

    def bends(self, i: int, j: int) -> bool:
        """Does RULE B bend this pair's route?  The row breakout the task file demands."""
        return self.opt[(i, j)] > self.opt_a[(i, j)]


def _free_cells(inst: Instance) -> list:
    """Floor cells that are neither a door nor a key home."""
    bad = set(inst.key_at) | {tuple(map(int, p)) for p in np.argwhere(inst.door > 0)}
    return [tuple(map(int, c)) for c in np.argwhere(inst.grid == FLOOR)
            if tuple(map(int, c)) not in bad]


def door_free_components(grid, door):
    """Connected components of the floor with every door cell removed.

    This is the structure the adoption condition "the doors are load-bearing" is
    about: a pair whose agent and target sit in DIFFERENT components cannot be walked
    without opening a door, for every such pair at once.  Choosing the agent and
    target words across two components is how CMAZE-2 keeps that condition for all
    pairs instead of the single pair ``generate()`` guarantees it for.
    """
    from collections import deque

    H, W = grid.shape
    seen, out = set(), []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != FLOOR or door[r, c] != 0 or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cur = [(r, c)]
            while q:
                y, x = q.popleft()
                for dr, dc in MOVES.values():
                    ny, nx = y + dr, x + dc
                    if (0 <= ny < H and 0 <= nx < W and grid[ny, nx] == FLOOR
                            and door[ny, nx] == 0 and (ny, nx) not in seen):
                        seen.add((ny, nx))
                        q.append((ny, nx))
                        cur.append((ny, nx))
            out.append(cur)
    return sorted(out, key=len, reverse=True)


def _probe(st, i, j):
    """(feasible, optimal length A+B, route, optimal length under A alone)."""
    inst = Instance(0, st["grid"], st["tint"], st["door"], st["key_at"],
                    st["agents"][i], st["targets"][j], st["n_colours"], 10 ** 9, 0, 0)
    L, r = solve(inst)
    if L is None:
        return False, None, None, None
    La, _ = solve(inst, rule_b=False)
    if La is None:
        return False, None, None, None
    if _reachable_without_keys(inst):
        return False, None, None, None          # doors not load-bearing for this pair
    return True, L, r, La


def _poison_pairs(st, pairs, rng, target_kwargs, max_poison):
    """Adversarial tinting, generalised to many pairs.

    For each pair the named blind planner's route is replayed and every cell it enters
    while carrying key c is a candidate for tint c -- exactly RULE B's refusal
    condition.  A candidate is kept only if EVERY kept pair stays feasible under RULES
    A+B, so the interaction is constructed in the distribution and nothing is discarded.
    """
    key_cell = {st["key_at"][i]: i + 1 for i in range(st["n_colours"])}
    protected = set(st["key_at"]) | set(st["agents"]) | set(st["targets"])
    tint = st["tint"]
    used = 0
    order = list(pairs)
    rng.shuffle(order)
    for (i, j) in order:
        inst = Instance(0, st["grid"], tint, st["door"], st["key_at"],
                        st["agents"][i], st["targets"][j], st["n_colours"],
                        10 ** 9, 0, 0)
        _, route = solve(inst, **target_kwargs)
        if route is None:
            continue
        cands, pos, ks = [], st["agents"][i], [UNTAKEN] * st["n_colours"]
        for a in route:
            if a == "P":
                kc = key_cell.get(pos, 0)
                if kc:
                    ks[kc - 1] = 1
                continue
            dr, dc = MOVES[a]
            npos = (pos[0] + dr, pos[1] + dc)
            dcol = int(st["door"][npos])
            if dcol and ks[dcol - 1] == 1:
                ks[dcol - 1] = 2
            else:
                pk = _pocket(tuple(ks))
                if pk and npos not in protected and tint[npos] == 0 and dcol == 0:
                    cands.append((npos, pk))
            pos = npos
        rng.shuffle(cands)
        added = 0
        for (r, c), colour in cands:
            if added >= max_poison:
                break
            if tint[r, c] != 0:
                continue
            tint[r, c] = colour
            if all(_probe(st, a, b)[0] for (a, b) in pairs):
                added += 1
                used += 1
            else:
                tint[r, c] = 0
    return used


def _seal(grid, cell):
    """Wall a target in: every floor neighbour of the cell becomes WALL.

    The cell itself stays floor, so the target is still THERE and still visible in the
    picture -- it simply cannot be reached.  That is what makes the refusal a statement
    about the world rather than about a missing word.
    """
    g = grid.copy()
    H, W = g.shape
    for dr, dc in MOVES.values():
        r, c = cell[0] + dr, cell[1] + dc
        if 0 <= r < H and 0 <= c < W and g[r, c] == FLOOR:
            g[r, c] = 0
    return g


def build_site(seed: int, per_side: int = 2, min_pairs: int = 8,
               adversarial: int = 2, max_poison: int = 6, max_tries: int = 12,
               wall_target: int = -1):
    """One Site, or None.  ``generate()`` is called with ITS OWN DEFAULTS.

    ``per_side`` agents and ``per_side`` targets are placed in EACH of the two largest
    door-free components, and only the CROSS pairs are kept -- so every kept pair is
    door-gated (the adoption condition), and the kept set is symmetric in direction.
    With ``per_side=2`` that is 8 pairs of the 16 combinations.
    """
    rng = np.random.default_rng(seed * 7919 + 13)
    base = generate(seed)                      # the adopted configuration, untouched
    if base is None:
        return None
    n_agents = n_targets = 2 * per_side
    comps = door_free_components(base.grid, base.door)
    freeset = set(_free_cells(base))
    usable = [[c for c in comp if c in freeset] for comp in comps]
    usable = [u for u in usable if len(u) >= 2 * per_side]
    if len(usable) < 2:
        return None
    side0, side1 = usable[0], usable[1]
    #: agent words a0..a{p-1} and target words t0..t{p-1} live in side 0; the rest in
    #: side 1.  side[i] is therefore a property of the WORD, constant across mazes.
    for _ in range(max_tries):
        p0 = rng.permutation(len(side0))[: 2 * per_side]
        p1 = rng.permutation(len(side1))[: 2 * per_side]
        c0 = [side0[k] for k in p0]
        c1 = [side1[k] for k in p1]
        agents = tuple(c0[:per_side] + c1[:per_side])
        targets = tuple(c0[per_side:] + c1[per_side:])
        if len(set(agents) | set(targets)) != n_agents + n_targets:
            continue
        tint = base.tint.copy()
        for c in list(agents) + list(targets):
            tint[c] = 0                         # as generate() does for start and goal
        st = {"grid": base.grid, "tint": tint, "door": base.door, "key_at": base.key_at,
              "agents": agents, "targets": targets, "n_colours": base.n_colours}
        cross = [(i, j) for i in range(n_agents) for j in range(n_targets)
                 if (i < per_side) != (j < per_side)]
        walled, walled_pairs = -1, ()
        if wall_target >= 0:
            #: try each target word in a rotated order and seal the first one that can
            #: be sealed without costing any OTHER pair -- sealing a corridor cell is
            #: often impossible, so insisting on one particular word would leave the
            #: walled condition almost empty.
            for off in range(n_targets):
                cand = (int(wall_target) + off) % n_targets
                g2 = _seal(base.grid, targets[cand])
                st2 = dict(st, grid=g2)
                others = [(i, j) for (i, j) in cross if j != cand]
                if all(_probe(st2, i, j)[0] for (i, j) in others) and not any(
                        _probe(st2, i, cand)[0] for i in range(n_agents)):
                    st = st2
                    walled = cand
                    walled_pairs = tuple((i, j) for (i, j) in cross if j == cand)
                    cross = others
                    break
        kept = [(i, j) for (i, j) in cross if _probe(st, i, j)[0]]
        if len(kept) < min_pairs:
            continue
        npo = 0
        for r_i in range(adversarial):
            tgt = dict(rule_b=False) if r_i == 0 else dict(tint_local_only=True)
            npo += _poison_pairs(st, kept, rng, tgt, max_poison)
        kept = [(i, j) for (i, j) in kept if _probe(st, i, j)[0]]
        if len(kept) < min_pairs:
            continue
        opt, opt_a, route = {}, {}, {}
        for (i, j) in kept:
            _ok, L, r, La = _probe(st, i, j)
            opt[(i, j)], opt_a[(i, j)], route[(i, j)] = L, La, r
        return Site(seed, st["grid"], st["tint"], base.door, base.key_at, agents,
                    targets, base.n_colours, tuple(kept), opt, opt_a, route, npo,
                    walled, walled_pairs)
    return None


def build_bank(seed0: int, n_sites: int, **kw):
    """``n_sites`` Sites from consecutive generator seeds."""
    out, s = [], int(seed0)
    while len(out) < n_sites:
        st = build_site(s, **kw)
        s += 1
        if st is not None:
            out.append(st)
    return out


# ------------------------------------------------------------------ requests
@dataclass(frozen=True)
class Request:
    """A request is four words.  No feature, no cell, no image."""

    m: int          # maze word
    a: int          # agent word
    t: int          # target word
    b: int          # budget word, an ABSOLUTE integer step budget

    def values(self):
        return (float(self.m), float(self.a), float(self.t), float(self.b))


def enumerate_requests(bank, budgets=BUDGET_GRID):
    """Every (maze, agent, target, budget), split by whether the optimum fits.

    A request that does not fit is never written -- it is the refusal material, and the
    fact that the object has never seen one is the whole DISPLAY line.
    """
    feas, infeas = [], []
    for m, site in enumerate(bank):
        for (i, j) in site.pairs:
            for b in budgets:
                r = Request(m, i, j, int(b))
                (feas if site.opt[(i, j)] <= b else infeas).append(r)
    return feas, infeas
