"""CMAZE-2b -- the REPAIRED request layer.  One acceptance rule is added; nothing else moves.

The defect, in one measurement
------------------------------
``w9_maze.task.build_site`` keeps a pair iff it is feasible under RULES A+B and is not
reachable with no key at all.  On the adopted generator that admits pairs whose optimal
route picks up **exactly one key**: 1242 of 1449 routes (198 two, 9 three), so there is
no key ORDER for RULE B to trap, and BLIND-SEQUENTIAL reads 0.9037-0.9130 instead of the
published 0.4980.

Measured here on CMAZE-1 proper (``artefacts/anchor_breakdown.txt``) the mechanism is
exact.  BLIND-SEQUENTIAL commits to the key order the RULE-A-only planner chose and then
re-plans under RULES A+B with that order fixed; it is valid on **201 of the 201**
instances where the rule-A key order equals the A+B key order and on **3 of the 199**
where it does not.  Its validity by key count of the A+B optimum is

    1 key 0.6532 (194/297) | 2 keys 0.1351 (10/74) | 3 keys 0.0000 (0/26) | 4 keys 0.0000 (0/3)

With one key ``key_order=(c,)`` admits exactly one pickup of exactly colour c, which is
what the oracle does too, so BLIND-SEQUENTIAL *is* the oracle on those routes.

The repair
----------
ONE line is added to the request layer's acceptance test:

    a pair is kept iff its RULES A+B optimal route picks up at least ``MIN_KEYS`` keys.

and one structural consequence follows from it, which is a measurement, not a choice.
The >= 2-key property is **all-or-nothing per maze and DIRECTIONAL**
(``artefacts/probe_direction.txt``): of 40 mazes with two usable door-free components,
13 have a direction in which **18 of 18** cross pairs need >= 2 keys, and in every one of
those the REVERSE direction is **0 of 18 feasible** -- the key that opens the cut door
lives on one side of it, so from the far side the target cannot be reached at all.
w9's layer demanded pairs in BOTH directions (``min_pairs=12`` of 18 cross pairs drawn
symmetrically), and that demand is exactly what forced the cut to be single-key.
So CMAZE-2b places all agent words in the SOURCE component and all target words in the
DESTINATION one.

  DECLARED: the kept set is therefore no longer symmetric in direction.  w9's
  ``splits`` docstring wanted "neither held-out set is a single direction"; here every
  pair runs the same way across the cut, so direction is a CONSTANT of the split rather
  than a balanced factor.  It cannot confound a held-out comparison because it does not
  vary, but it is a difference from CMAZE-2 and it is stated rather than hidden.

The engine, the solver and the exact integer judge in ``w8_envscout.maze_colour`` are
imported unmodified, exactly as CMAZE-2 imports them.  ``solve`` computes the optimum and
``w8_envscout.screen.key_order_of`` reads the key order off that route; neither is
touched, and no tolerance is introduced anywhere -- a key count is an integer.
"""

from __future__ import annotations

import numpy as np

from w8_envscout.maze_colour import Instance, _reachable_without_keys, generate, solve
from w8_envscout.screen import key_order_of
from w9_maze.task import (Site, _free_cells, _poison_pairs, _seal,
                          door_free_components)

#: The repair.  2 is the smallest number of keys for which a key ORDER exists at all.
MIN_KEYS = 2


def probe(st, i, j, min_keys=MIN_KEYS):
    """``(ok, L, route, La, key_order)``.  ``ok`` also requires >= ``min_keys`` keys.

    w9's three conditions, in w9's sense, plus the repair.  They are evaluated
    cheapest-first only so a rejected pair costs one BFS instead of three; the accepted
    set does not depend on the order they are checked in.
    """
    inst = Instance(0, st["grid"], st["tint"], st["door"], st["key_at"],
                    st["agents"][i], st["targets"][j], st["n_colours"], 10 ** 9, 0, 0)
    if _reachable_without_keys(inst):
        return False, None, None, None, ()      # doors not load-bearing for this pair
    L, r = solve(inst)
    if L is None:
        return False, None, None, None, ()
    ko = key_order_of(inst, r)
    if len(ko) < min_keys:
        return False, L, r, None, ko            # <-- THE REPAIR
    La, _ = solve(inst, rule_b=False)
    if La is None:
        return False, None, None, None, ()
    return True, L, r, La, ko


def _try_placement(base, rng, src, dst, per_side, min_pairs, min_keys, adversarial,
                   max_poison, wall_target, seed):
    """One placement of 2*per_side agents in ``src`` and 2*per_side targets in ``dst``."""
    n_agents = n_targets = 2 * per_side
    p0 = rng.permutation(len(src))[:n_agents]
    p1 = rng.permutation(len(dst))[:n_targets]
    agents = tuple(src[k] for k in p0)
    targets = tuple(dst[k] for k in p1)
    if len(set(agents) | set(targets)) != n_agents + n_targets:
        return None
    tint = base.tint.copy()
    for c in list(agents) + list(targets):
        tint[c] = 0                             # as generate() does for start and goal
    st = {"grid": base.grid, "tint": tint, "door": base.door, "key_at": base.key_at,
          "agents": agents, "targets": targets, "n_colours": base.n_colours}
    cross = [(i, j) for i in range(n_agents) for j in range(n_targets)
             if (i < per_side) != (j < per_side)]
    #: THE SCREEN RUNS BEFORE ANY SEALING.  The >= min_keys condition must already hold
    #: on the UNSEALED maze, so the walls that create the refusal condition cannot be
    #: what creates the multi-key structure.  It is also what makes the build
    #: affordable: a single-key cut is rejected after 18 BFS calls rather than ~126.
    if sum(probe(st, i, j, min_keys)[0] for (i, j) in cross) < min_pairs:
        return None
    walled, walled_pairs = -1, ()
    if wall_target >= 0:
        for off in range(n_targets):
            cand = (int(wall_target) + off) % n_targets
            g2 = _seal(base.grid, targets[cand])
            st2 = dict(st, grid=g2)
            others = [(i, j) for (i, j) in cross if j != cand]
            if all(probe(st2, i, j, min_keys)[0] for (i, j) in others) and not any(
                    probe(st2, i, cand, 0)[0] for i in range(n_agents)):
                st = st2
                walled = cand
                walled_pairs = tuple((i, j) for (i, j) in cross if j == cand)
                cross = others
                break
    kept = [(i, j) for (i, j) in cross if probe(st, i, j, min_keys)[0]]
    if len(kept) < min_pairs:
        return None
    npo = 0
    for r_i in range(adversarial):
        tgt = dict(rule_b=False) if r_i == 0 else dict(tint_local_only=True)
        npo += _poison_pairs(st, kept, rng, tgt, max_poison)
    kept = [(i, j) for (i, j) in kept if probe(st, i, j, min_keys)[0]]
    if len(kept) < min_pairs:
        return None
    opt, opt_a, route, keys = {}, {}, {}, {}
    for (i, j) in kept:
        _ok, L, r, La, ko = probe(st, i, j, min_keys)
        opt[(i, j)], opt_a[(i, j)], route[(i, j)], keys[(i, j)] = L, La, r, ko
    site = Site(seed, st["grid"], st["tint"], base.door, base.key_at, agents,
                targets, base.n_colours, tuple(kept), opt, opt_a, route, npo,
                walled, walled_pairs)
    #: the key order of every kept route, carried ON the site so the histogram is a
    #: property of the artefact rather than something recomputed at report time.
    site.key_order = keys
    return site


def build_site_v2(seed: int, per_side: int = 3, min_pairs: int = 12,
                  adversarial: int = 2, max_poison: int = 6, max_tries: int = 12,
                  wall_target: int = -1, min_keys: int = MIN_KEYS):
    """One CMAZE-2b Site, or None.  ``generate()`` is called with ITS OWN DEFAULTS.

    Both orientations of the door cut are tried, because which way round the multi-key
    direction runs is a property of where ``generate`` happened to drop the keys.
    """
    rng = np.random.default_rng(seed * 7919 + 13)
    base = generate(seed)                      # the adopted configuration, untouched
    if base is None:
        return None
    comps = door_free_components(base.grid, base.door)
    freeset = set(_free_cells(base))
    usable = [u for u in ([c for c in comp if c in freeset] for comp in comps)
              if len(u) >= 2 * per_side]
    if len(usable) < 2:
        return None
    for (a, b) in ((0, 1), (1, 0)):
        for _ in range(max_tries):
            site = _try_placement(base, rng, usable[a], usable[b], per_side, min_pairs,
                                  min_keys, adversarial, max_poison, wall_target, seed)
            if site is not None:
                return site
    return None


def build_bank_v2(seed0: int, n_sites: int, **kw):
    out, s = [], int(seed0)
    while len(out) < n_sites:
        st = build_site_v2(s, **kw)
        s += 1
        if st is not None:
            out.append(st)
    return out


def site_key_orders(site):
    """``{(i, j): key colours in order}`` for every kept pair, off the solver's route."""
    ko = getattr(site, "key_order", None)
    if ko is not None:
        return ko
    out = {}
    for (i, j) in site.pairs:
        inst = site.instance(i, j, 10 ** 9)
        out[(i, j)] = key_order_of(inst, site.route[(i, j)])
    return out


def key_count_histogram(bank):
    """``{n_keys: n_routes}`` over every kept pair of every site -- the acceptance number."""
    hist = {}
    for site in bank:
        for (i, j), ko in site_key_orders(site).items():
            hist[len(ko)] = hist.get(len(ko), 0) + 1
    return dict(sorted(hist.items()))
