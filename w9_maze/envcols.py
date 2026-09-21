"""The environment's own columns, re-measured on THIS hub's split.

The adopted CMAZE-1 numbers (blind-sequential 0.4980, fragment-lookup 0.0000/0.0000,
rule B bends 81.6 % of routes) were measured on the one-pair distribution.  CMAZE-2
changes where the agents and the targets are, so those numbers are not assumed to
carry -- they are re-measured here on exactly the requests the rows are scored on, and
both sets are printed.

Two of them are load-bearing for this hub's claims:

* BLIND-SEQUENTIAL is the caveat that travels with every composition claim.  It is
  printed beside every RESULT row, and every row is additionally broken out by whether
  rule B bends that route, because a win only where blind-sequential also wins is a win
  on the easy half of the environment.
* FRAGMENT-LOOKUP is the tripwire that killed GYM-1 and GYM-2.  If it rises above
  0.0000 on this split, the split leaks and the hub says so.

The planners are ``w8_envscout.screen``'s, unmodified: a blind planner is a planner
that searches a RELAXED legality relation and is graded by the UNRELAXED exact judge.
"""

from __future__ import annotations

import numpy as np

from w8_envscout.screen import (PLANNERS, build_table, canon_features, frag_rollout,
                                plan, random_walk, replay_states, score_route)


def blind_columns(bank, rows, planners=None):
    """Every blind planner on the rows this hub scores."""
    out = {}
    names = PLANNERS if planners is None else planners
    for name, spec in names.items():
        okc, scs, nb_ok, nb_n, b_ok, b_n = 0, 0.0, 0, 0, 0, 0
        for (req, m, i, j) in rows:
            inst = bank[m].instance(i, j, req.b)
            route = plan(inst, spec)
            ok, sc, _r = score_route(inst, route)
            okc += bool(ok)
            scs += sc
            if bank[m].bends(i, j):
                b_n += 1
                b_ok += bool(ok)
            else:
                nb_n += 1
                nb_ok += bool(ok)
        n = max(1, len(rows))
        out[name] = {"validity": okc / n, "score": scs / n, "n": len(rows),
                     "validity_bend": (b_ok / b_n) if b_n else float("nan"),
                     "validity_nobend": (nb_ok / nb_n) if nb_n else float("nan")}
    return out


def random_column(bank, rows, seed=12345):
    rng = np.random.default_rng(seed)
    okc, scs = 0, 0.0
    for (req, m, i, j) in rows:
        inst = bank[m].instance(i, j, req.b)
        ok, sc, _ = score_route(inst, random_walk(inst, rng))
        okc += bool(ok)
        scs += sc
    n = max(1, len(rows))
    return {"validity": okc / n, "score": scs / n, "n": len(rows)}


def fragment_columns(bank, train_rows, eval_rows, radii=(1, 2), seed=12345):
    """The retrieval floor: local colour-canonicalised signatures from TRAIN routes."""
    rng = np.random.default_rng(seed)
    tr = []
    for (req, m, i, j) in train_rows:
        inst = bank[m].instance(i, j, req.b)
        tr.append((inst, bank[m].route[(i, j)]))
    out = {}
    for radius in radii:
        table = build_table(tr, radius)
        hit = tot = 0
        for (req, m, i, j) in eval_rows:
            inst = bank[m].instance(i, j, req.b)
            for pos, ks, _a in replay_states(inst, bank[m].route[(i, j)]):
                tot += 1
                hit += canon_features(inst, pos, ks, radius) in table
        okc, scs = 0, 0.0
        for (req, m, i, j) in eval_rows:
            inst = bank[m].instance(i, j, req.b)
            ok, sc, _ = score_route(inst, frag_rollout(inst, table, radius, rng))
            okc += bool(ok)
            scs += sc
        n = max(1, len(eval_rows))
        out["FRAG-r%d" % radius] = {"validity": okc / n, "score": scs / n,
                                    "signatures": len(table),
                                    "state_coverage": hit / max(1, tot), "n": n}
    return out


def fragment_cross_maze(bank, pool_sites, eval_rows, radii=(1, 2), seed=12345,
                        max_pool=120):
    """The CMAZE-1 comparison: the fragment table built from OTHER MAZES ONLY.

    CMAZE-1 measured fragment-lookup at 0.0000 because its evaluation mazes were
    unseen.  This hub's rows are unseen pairs on a SEEN maze, so its same-maze
    fragment table is a far stronger floor (measured 0.4586 at r=2).  Both are
    reported: this function is the like-for-like with the published 0.0000, and
    ``fragment_columns`` is the floor that actually binds this hub's rows.
    """
    rng = np.random.default_rng(seed)
    tr = []
    for st in pool_sites[:max_pool]:
        for (i, j) in st.pairs:
            b = st.opt[(i, j)] + int(np.ceil(0.1 * st.opt[(i, j)]))
            tr.append((st.instance(i, j, b), st.route[(i, j)]))
    out = {}
    for radius in radii:
        table = build_table(tr, radius)
        okc, scs = 0, 0.0
        for (req, m, i, j) in eval_rows:
            inst = bank[m].instance(i, j, req.b)
            ok, sc, _ = score_route(inst, frag_rollout(inst, table, radius, rng))
            okc += bool(ok)
            scs += sc
        n = max(1, len(eval_rows))
        out["FRAG-cross-r%d" % radius] = {"validity": okc / n, "score": scs / n,
                                          "signatures": len(table),
                                          "routes": len(tr), "n": n}
    return out


def bend_stats(bank, rows):
    o = np.array([bank[m].opt[(i, j)] for (_r, m, i, j) in rows], float)
    a = np.array([bank[m].opt_a[(i, j)] for (_r, m, i, j) in rows], float)
    return {"n": len(rows), "bend_fraction": float((o > a).mean()),
            "detour_steps": float((o - a).mean()), "opt_mean": float(o.mean()),
            "opt_a_mean": float(a.mean())}
