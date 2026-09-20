"""Screening harness for CMAZE-1.

Produces the two numbers the wave-8 task asks for -- the composition-blind score
and the fragment-lookup score -- plus the baselines needed to read them honestly.

A "composition-blind" planner is defined here as a planner that searches a RELAXED
legality relation and is then graded by the UNRELAXED exact judge.  That is a
mechanical definition, not a hand-written heuristic, so it cannot be dismissed as a
straw man: each blind planner is OPTIMAL within the view of the world it is given.

Two score columns are reported:
  validity   fraction of episodes whose emitted route the exact judge accepts
             (legal at every step, goal reached, length <= budget)
  score      mean of  (valid ? opt_len / len(route) : 0)  -- in [0, 1], 1.0 for an
             optimal valid route.  This is the column to compare with GYM-2's
             composition-blind 0.9367-1.0000.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict

import numpy as np

from w8_envscout.maze_colour import (CARRIED, FLOOR, MOVES, MOVE_ORDER, UNTAKEN,
                                     USED, _pocket, generate, judge, solve)

# name -> kwargs for solve()
PLANNERS = {
    "ORACLE          (full composition)": dict(),
    "BLIND-A         (rule B dropped)": dict(rule_b=False),
    "BLIND-B         (rule A dropped)": dict(rule_a=False),
    "BLIND-COLOUR    (colours collapsed)": dict(colour_blind=True),
    "BLIND-FACTOR    (A x B local only)": dict(tint_local_only=True),
    "BLIND-SEQUENTIAL(key order from A)": "sequential",
}


def score_route(inst, route):
    ok, reason, _ = judge(inst, route)
    sc = (inst.opt_len / len(route)) if (ok and route) else 0.0
    return ok, sc, reason


def key_order_of(inst, route):
    """The sequence of key colours a route picks up, in order."""
    key_cell = {inst.key_at[i]: i + 1 for i in range(inst.n_colours)}
    pos, order = inst.start, []
    ks = [UNTAKEN] * inst.n_colours
    for a in route:
        if a == "P":
            kc = key_cell.get(pos, 0)
            if kc:
                order.append(kc)
                ks[kc - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        npos = (pos[0] + dr, pos[1] + dc)
        dcol = int(inst.door[npos])
        if dcol and ks[dcol - 1] == CARRIED:
            ks[dcol - 1] = USED
        pos = npos
    return tuple(order)


def plan(inst, spec):
    """Run one planner and return its route (or None)."""
    if spec == "sequential":
        la, ra = solve(inst, rule_b=False)          # commit to rule A's key order
        if ra is None:
            return None
        return solve(inst, key_order=key_order_of(inst, ra))[1]
    return solve(inst, **spec)[1]


# ---------------------------------------------------------------- fragments
def canon_features(inst, pos, ks, radius=2):
    """Colour-canonicalised local signature.

    Colours are recoded RELATIVE TO THE POCKET, so a fragment learnt on a red key
    transfers to a blue one.  This is the most generous reasonable featurisation for
    a lookup policy and is what gives fragment lookup its best shot.
    Codes: 0 wall, 1 plain floor, 2 tint == pocket colour, 3 tint != pocket,
           4 door == pocket colour, 5 door != pocket, 6 an untaken key.
    """
    g, tint, door = inst.grid, inst.tint, inst.door
    H, W = g.shape
    pk = _pocket(ks)
    key_cell = {inst.key_at[i]: i + 1 for i in range(inst.n_colours)}
    patch = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = pos[0] + dr, pos[1] + dc
            if not (0 <= r < H and 0 <= c < W) or g[r, c] != FLOOR:
                patch.append(0)
                continue
            dcol = int(door[r, c])
            if dcol:
                patch.append(1 if ks[dcol - 1] == USED else (4 if dcol == pk else 5))
                continue
            kc = key_cell.get((r, c), 0)
            if kc and ks[kc - 1] == UNTAKEN:
                patch.append(6)
                continue
            tcol = int(tint[r, c])
            if tcol:
                patch.append(2 if tcol == pk else 3)
                continue
            patch.append(1)
    gr = int(np.sign(inst.goal[0] - pos[0]))
    gc = int(np.sign(inst.goal[1] - pos[1]))
    return (tuple(patch), 1 if pk else 0, gr, gc)


def replay_states(inst, route):
    """Yield (pos, keystate, action) before each action of a legal route."""
    key_cell = {inst.key_at[i]: i + 1 for i in range(inst.n_colours)}
    ks = [UNTAKEN] * inst.n_colours
    pos = inst.start
    for a in route:
        yield pos, tuple(ks), a
        if a == "P":
            ks[key_cell[pos] - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        npos = (pos[0] + dr, pos[1] + dc)
        dcol = int(inst.door[npos])
        if dcol and ks[dcol - 1] == CARRIED:
            ks[dcol - 1] = USED
        pos = npos


def build_table(train_insts, radius):
    tab = defaultdict(Counter)
    for inst, route in train_insts:
        for pos, ks, a in replay_states(inst, route):
            tab[canon_features(inst, pos, ks, radius)][a] += 1
    return {f: cnt.most_common(1)[0][0] for f, cnt in tab.items()}


def frag_rollout(inst, table, radius, rng):
    """Greedy fragment-lookup rollout under the true dynamics."""
    key_cell = {inst.key_at[i]: i + 1 for i in range(inst.n_colours)}
    ks = [UNTAKEN] * inst.n_colours
    pos, out = inst.start, []
    H, W = inst.grid.shape
    for _ in range(inst.budget):
        a = table.get(canon_features(inst, pos, tuple(ks), radius))
        if a is None:
            a = MOVE_ORDER[int(rng.integers(4))]
        out.append(a)
        if a == "P":
            kc = key_cell.get(pos, 0)
            if kc == 0 or ks[kc - 1] != UNTAKEN or _pocket(tuple(ks)) != 0:
                break
            ks[kc - 1] = CARRIED
            continue
        dr, dc = MOVES[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < H and 0 <= nc < W) or inst.grid[nr, nc] != FLOOR:
            break
        dcol = int(inst.door[nr, nc])
        if dcol and ks[dcol - 1] != USED:
            if ks[dcol - 1] != CARRIED:
                break
            ks[dcol - 1] = USED
        tcol = int(inst.tint[nr, nc])
        if tcol and _pocket(tuple(ks)) == tcol:
            break
        pos = (nr, nc)
        if pos == inst.goal:
            break
    return "".join(out)


def random_walk(inst, rng):
    return "".join(MOVE_ORDER[int(rng.integers(4))] if rng.random() < 0.9 else "P"
                   for _ in range(inst.budget))


# ---------------------------------------------------------------- one sweep
def make_set(n, seed0, slack, require_bend, genkw):
    out, s, tried = [], seed0, 0
    while len(out) < n:
        inst = generate(s, slack=slack, require_bend=require_bend, **genkw)
        s += 1
        tried += 1
        if inst is not None:
            out.append(inst)
    return out, tried


def sweep(label, n_eval, n_train, slack, require_bend, seed0, genkw, rng):
    t0 = time.time()
    ev, tried = make_set(n_eval, seed0, slack, require_bend, genkw)
    opt = np.array([i.opt_len for i in ev], dtype=float)
    optA = np.array([i.opt_len_a_only for i in ev], dtype=float)

    print("\n" + "=" * 78)
    print("%s   (n_eval=%d, slack=%.0f%%, acceptance filter 'rule B bends the route': %s)"
          % (label, n_eval, 100 * slack, "ON" if require_bend else "OFF"))
    print("=" * 78)
    print("  seed acceptance rate      : %.3f  (%d seeds for %d instances)"
          % (n_eval / tried, tried, n_eval))
    print("  optimal length under A+B  : mean %6.1f   median %3d   max %3d"
          % (opt.mean(), int(np.median(opt)), int(opt.max())))
    print("  optimal length under A    : mean %6.1f" % optA.mean())
    print("  detour forced by rule B   : mean %+.1f steps; rule B bends the route in %.1f%% of episodes"
          % ((opt - optA).mean(), 100.0 * (opt > optA).mean()))

    print("\n  %-36s %9s %9s   %s" % ("planner", "validity", "score", "top failure mode"))
    rows = {}
    for name, spec in PLANNERS.items():
        okc, scs, reasons = 0, 0.0, Counter()
        for inst in ev:
            route = plan(inst, spec)
            ok, sc, reason = score_route(inst, route)
            okc += ok
            scs += sc
            if not ok:
                reasons[reason.split("@")[0]] += 1
        rows[name] = (okc / n_eval, scs / n_eval)
        top = reasons.most_common(1)[0] if reasons else ("-", 0)
        print("  %-36s %9.4f %9.4f   %s x%d"
              % (name, okc / n_eval, scs / n_eval, top[0], top[1]))

    okc, scs = 0, 0.0
    for inst in ev:
        ok, sc, _ = score_route(inst, random_walk(inst, rng))
        okc += ok
        scs += sc
    rows["RANDOM"] = (okc / n_eval, scs / n_eval)
    print("  %-36s %9.4f %9.4f" % ("RANDOM          (uniform walk)", okc / n_eval, scs / n_eval))

    # ---- fragment lookup
    tr, _ = make_set(n_train, seed0 + 10 ** 7, slack, require_bend, genkw)
    tr = [(i, solve(i)[1]) for i in tr]
    ev_opt = [(i, solve(i)[1]) for i in ev]
    for radius in (1, 2):
        table = build_table(tr, radius)
        hit = tot = 0
        for inst, route in ev_opt:
            for pos, ks, _a in replay_states(inst, route):
                tot += 1
                hit += canon_features(inst, pos, ks, radius) in table
        okc, scs = 0, 0.0
        for inst in ev:
            ok, sc, _ = score_route(inst, frag_rollout(inst, table, radius, rng))
            okc += ok
            scs += sc
        rows["FRAG-r%d" % radius] = (okc / n_eval, scs / n_eval)
        print("  %-36s %9.4f %9.4f   %d signatures from %d routes, coverage %.4f"
              % ("FRAGMENT-LOOKUP r=%d  (retrieval)" % radius, okc / n_eval, scs / n_eval,
                 len(table), n_train, hit / tot))
    print("  [%s took %.1fs]" % (label, time.time() - t0))
    return rows


def main(n_eval=500, n_train=2000, seed0=1_000_000, **genkw):
    rng = np.random.default_rng(12345)
    print("CMAZE-1 screening -- exact judge, no tolerance anywhere")
    out = {}
    out["filtered/slack10"] = sweep("A. ADOPTION CONFIGURATION", n_eval, n_train,
                                    0.10, True, seed0, genkw, rng)
    out["unfiltered/slack10"] = sweep("B. UNFILTERED GENERATOR (no 'bends the route' filter)",
                                      n_eval, n_train, 0.10, False, seed0 + 2 * 10 ** 7,
                                      genkw, rng)
    out["filtered/slack50"] = sweep("C. LOOSE BUDGET (+50% slack) -- how much the budget matters",
                                    n_eval, n_train, 0.50, True, seed0 + 4 * 10 ** 7,
                                    genkw, rng)
    return out


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    main(**kw)
