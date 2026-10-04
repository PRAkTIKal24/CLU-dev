"""Instance sets, oracle trajectories, and the impossible examples.

Two populations, kept apart on purpose:

  POSSIBLE    `make_instances` -- the adopted CMAZE-1 configuration, unchanged from
              env-scout's defaults.  Every instance here has an exact oracle route.
              This is the ONLY population the MLP, LSTM, transformer and table arms
              ever see, in training or in evaluation.

  IMPOSSIBLE  `make_impossible` -- instances for which NO route is accepted by the
              exact judge, for one of two exactly-certified reasons:
                budget    the exact BFS optimum exists but is longer than the budget
                          (impossible because the optimum is optimal);
                legality  BFS over (cell, keystate) finds no route at all.
              Only the supervised refuser is trained on these.  `assert_unseen`
              is the check that keeps our line honest: no impossible instance may
              appear in any non-refuser arm's data.
"""

from __future__ import annotations

import numpy as np

from w8_envscout.maze_colour import Instance, generate, solve
from w9_harness.dynamics import initial_state, replay, step, key_homes
from w9_harness.features import global_summary, obs
from w9_harness.judge_maze import ACTIONS, JUDGE


# --------------------------------------------------------------- possible set
def make_instances(n, seed0, **genkw):
    """n accepted instances and the seeds they came from.  Deterministic in seed0."""
    out, seeds, s = [], [], seed0
    while len(out) < n:
        inst = generate(s, **genkw)
        if inst is not None:
            out.append(inst)
            seeds.append(s)
        s += 1
    return out, seeds


def oracle_routes(insts):
    return [solve(i)[1] for i in insts]


# ------------------------------------------------------------- impossible set
def _with(inst, tint=None, budget=None):
    return Instance(inst.seed, inst.grid,
                    inst.tint if tint is None else tint, inst.door, inst.key_at,
                    inst.start, inst.goal, inst.n_colours,
                    inst.budget if budget is None else budget,
                    inst.opt_len, inst.opt_len_a_only)


def _impossible_by_budget(inst):
    """Budget one below the exact optimum.  No route can be shorter than optimal."""
    if inst.opt_len < 2:
        return None
    cand = _with(inst, budget=inst.opt_len - 1)
    L, _ = solve(cand)
    if L is None or L <= cand.budget:
        return None
    return cand


def _impossible_by_legality(inst, rng, max_add=60):
    """Tint cells until BFS over (cell, keystate) finds nothing at all."""
    tint = inst.tint.copy()
    k = int(inst.n_colours)
    protected = {inst.start, inst.goal} | set(inst.key_at)
    cells = [tuple(map(int, p)) for p in np.argwhere(inst.grid == 1)]
    cells = [c for c in cells if c not in protected and int(inst.door[c]) == 0]
    rng.shuffle(cells)
    for cell in cells[:max_add]:
        if tint[cell] != 0:
            continue
        tint[cell] = int(rng.integers(1, k + 1))
        cand = _with(inst, tint=tint)
        if solve(cand)[0] is None:
            return cand
    return None


def make_impossible(n, seed0, rng_seed=0, **genkw):
    """n instances the exact judge can never accept, half by budget, half by legality."""
    rng = np.random.default_rng(rng_seed)
    out, fams, s = [], [], seed0
    while len(out) < n:
        base = generate(s, **genkw)
        s += 1
        if base is None:
            continue
        want = "budget" if len(out) % 2 == 0 else "legality"
        cand = (_impossible_by_budget(base) if want == "budget"
                else _impossible_by_legality(base, rng))
        if cand is None:
            other = (_impossible_by_legality(base, rng) if want == "budget"
                     else _impossible_by_budget(base))
            if other is None:
                continue
            cand, want = other, ("legality" if want == "budget" else "budget")
        out.append(cand)
        fams.append(want)
    return out, fams


def certify_impossible(insts, fams):
    """Exact certification -- no sampling, no search over routes.  Returns failures."""
    bad = []
    for inst, fam in zip(insts, fams):
        L, route = solve(inst)
        if fam == "legality":
            if L is not None:
                bad.append((inst.seed, "legality instance is solvable", L))
        else:
            if L is None:
                bad.append((inst.seed, "budget instance is unsolvable", None))
            elif L <= int(inst.budget):
                bad.append((inst.seed, "optimum fits the budget", L))
            elif JUDGE(inst, route)[0]:
                bad.append((inst.seed, "judge accepted the optimum", L))
    return bad


def assert_unseen(possible_seeds, impossible_insts):
    """Our line: no impossible example was ever seen by a non-refuser arm."""
    imp = {int(i.seed) for i in impossible_insts}
    overlap = imp & {int(s) for s in possible_seeds}
    return {"impossible_in_arm_data": len(overlap), "clean": len(overlap) == 0}


# -------------------------------------------------------------- trajectories
def build_sequences(insts, feats, routes, obs_dim, tmax=None, n_out_pad=0):
    """(X, Y, M) padded step-level supervision from oracle routes.

    X[i, t] is `features.obs` at the state BEFORE action t -- the same vector for
    every arm.  Y[i, t] is the oracle action index.  M is the validity mask.
    """
    lens = [len(r) for r in routes]
    T = tmax if tmax is not None else max(lens)
    truncated = sum(1 for L in lens if L > T)
    N = len(insts)
    X = np.zeros((N, T, obs_dim), dtype=np.float32)
    Y = np.zeros((N, T), dtype=np.int32)
    M = np.zeros((N, T), dtype=np.float32)
    for i, (inst, S, route) in enumerate(zip(insts, feats, routes)):
        gs = global_summary(S)
        states, _, _ = replay(inst, route)
        for t, (st, a) in enumerate(zip(states, route)):
            if t >= T:
                break
            X[i, t] = obs(inst, S, st, gs)
            Y[i, t] = ACTIONS.index(a)
            M[i, t] = 1.0
    return X, Y, M, {"T": T, "n_truncated_routes": truncated,
                     "steps_supervised": int(M.sum()),
                     "steps_available": int(sum(lens))}


def refuse_sequences(insts, feats, obs_dim, T, refuse_index, weight=1.0):
    """Impossible instances: the target is REFUSE, from the first step on.

    The agent is walked along a legal random prefix so the refusal is not trivially
    tied to the start cell -- a refuser that only knows how to refuse at t=0 would
    be refusing the picture, not the problem.

    `weight` scales the loss mask.  An impossible instance contributes ~4 labelled
    steps against a possible instance's ~20-40, so without it REFUSE is a few
    percent of the objective and the refuser learns to never use the token.  The
    driver sets it so the two classes carry equal total weight, and says so.
    """
    rng = np.random.default_rng(3)
    N = len(insts)
    X = np.zeros((N, T, obs_dim), dtype=np.float32)
    Y = np.full((N, T), refuse_index, dtype=np.int32)
    M = np.zeros((N, T), dtype=np.float32)
    for i, (inst, S) in enumerate(zip(insts, feats)):
        gs = global_summary(S)
        homes = key_homes(inst)
        st = initial_state(inst)
        steps = min(int(rng.integers(1, 9)), T)
        for t in range(steps):
            X[i, t] = obs(inst, S, st, gs)
            M[i, t] = weight
            for a in rng.permutation(list(ACTIONS)):
                nxt, legal, goal = step(inst, st, str(a), homes)
                if legal and not goal:
                    st = nxt
                    break
            else:
                break
    return X, Y, M
