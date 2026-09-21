"""The table floor, and the fragment-lookup REPORTED COLUMN.

TABLE FLOOR (`table`) -- the memorisation floor.  A dictionary from a raw, discrete,
uncanonicalised local state to the most frequent oracle action.  It is deliberately
the one arm that does NOT consume the encoder: its job is to say how far pure
memorisation of the training distribution gets, which is the number every learned
arm has to beat before it has demonstrated anything.  Its coverage on the evaluation
set is printed, not hidden.

TABLE-ENC (`table_enc`) -- the same lookup policy keyed on the SHARED ENCODER's
features, sign-quantised.  This is the honest check on whether the encoder is even
usable by a lookup policy, and it is reported beside the raw floor rather than
substituted for it.

FRAGMENT-LOOKUP -- env-scout's definition, reused unchanged from
`w8_envscout.screen`: colours recoded relative to the pocket, which is the most
generous featurisation a retrieval policy can be given.  It is a REPORTED COLUMN.
It is never an abort condition: a high fragment-lookup number is a fact about the
distribution to be printed, not a reason to stop.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from w8_envscout.maze_colour import FLOOR, UNTAKEN, USED
from w8_envscout.screen import build_table, canon_features, frag_rollout
from w9_harness.dynamics import initial_state, key_homes, replay, step
from w9_harness.features import enc_key
from w9_harness.judge_maze import ACTIONS


# ------------------------------------------------------------- raw state key
def raw_key(inst, st, r=1):
    """Uncanonicalised local code patch + pocket + goal-direction sign.

    Codes: 0 off-grid/wall, 1 floor, 2..1+k tint colour, 2+k..1+2k closed door,
    1+2k+1 open door, 2+2k+1.. untaken key colour.  No colour canonicalisation --
    that is what separates this floor from fragment lookup.
    """
    g, tint, door = inst.grid, inst.tint, inst.door
    H, W = g.shape
    k = int(inst.n_colours)
    homes = key_homes(inst)
    patch = []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            rr, cc = st.pos[0] + dr, st.pos[1] + dc
            if not (0 <= rr < H and 0 <= cc < W) or g[rr, cc] != FLOOR:
                patch.append(0)
                continue
            dcol = int(door[rr, cc])
            if dcol:
                patch.append(2 + 2 * k if st.ks[dcol - 1] == USED else 1 + k + dcol)
                continue
            kc = homes.get((rr, cc), 0)
            if kc and st.ks[kc - 1] == UNTAKEN:
                patch.append(3 + 2 * k + kc)
                continue
            tcol = int(tint[rr, cc])
            patch.append(1 + tcol if tcol else 1)
    return (tuple(patch), st.pocket,
            int(np.sign(inst.goal[0] - st.pos[0])), int(np.sign(inst.goal[1] - st.pos[1])))


def build(train_insts, train_routes, keyfn):
    tab = defaultdict(Counter)
    for inst, route in zip(train_insts, train_routes):
        states, _, _ = replay(inst, route)
        for st, a in zip(states, route):
            tab[keyfn(inst, st)][a] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


def table_rollout(insts, table, keyfn, rng):
    """Greedy lookup rollout under the true dynamics.  Misses fall back to random."""
    routes, misses, total = [], 0, 0
    for inst in insts:
        homes = key_homes(inst)
        st = initial_state(inst)
        out = []
        for _ in range(int(inst.budget)):
            total += 1
            a = table.get(keyfn(inst, st))
            if a is None:
                misses += 1
                a = ACTIONS[int(rng.integers(4))]
            out.append(a)
            nxt, legal, goal = step(inst, st, a, homes)
            if not legal or goal:
                break
            st = nxt
        routes.append("".join(out))
    return routes, {"coverage": 1.0 - misses / max(total, 1), "size": len(table)}


def make_raw_key():
    return lambda inst, st: raw_key(inst, st)


def make_enc_key(feats_by_id):
    return lambda inst, st: enc_key(feats_by_id[id(inst)], st)


# ---------------------------------------------------- fragment-lookup column
def fragment_column(train_insts, train_routes, eval_insts, radius, rng):
    """env-scout's fragment lookup, verbatim, scored here by the exact judge."""
    table = build_table(list(zip(train_insts, train_routes)), radius)
    routes = [frag_rollout(i, table, radius, rng) for i in eval_insts]
    hit = tot = 0
    for inst, route in zip(train_insts[:200], train_routes[:200]):
        states, _, _ = replay(inst, route)
        for st in states:
            tot += 1
            hit += canon_features(inst, st.pos, st.ks, radius) in table
    return routes, {"signatures": len(table), "train_coverage": hit / max(tot, 1)}
