"""Rollout under the true dynamics, scored by the exact judge and nothing else.

The rollout is lockstep and batched: all evaluation instances advance one step at a
time, the whole padded observation history is re-encoded each step, and each arm
reads the logits at the current position.  Causal masking (transformer) and the
prefix-only recurrence (LSTM) make reading position t from a full-length padded
forward pass identical to feeding the prefix, so no arm sees its own future.

An episode ends the moment an action is illegal, the goal is entered, the budget is
exhausted, or (refuser only) REFUSE is emitted.  The emitted string is then handed
to `JUDGE`, which is the only thing in this file that decides anything.
"""

from __future__ import annotations

import time
from collections import Counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from w8_envscout.maze_colour import Instance
from w9_harness.dynamics import initial_state, key_homes, replay, step
from w9_harness.features import global_summary, obs
from w9_harness.judge_maze import ACTIONS, JUDGE, REFUSE, reason_class


def _budget_lifted(inst):
    """The same instance with the budget removed.  Used only for the DIAGNOSTIC
    'did it reach the goal legally at all', which is the exact judge applied to a
    declared variant -- never the scored column."""
    return Instance(inst.seed, inst.grid, inst.tint, inst.door, inst.key_at,
                    inst.start, inst.goal, inst.n_colours, 10 ** 9,
                    inst.opt_len, inst.opt_len_a_only)


@eqx.filter_jit
def _forward(policy, X):
    """Module-level so the jit cache is keyed on the policy's structure, not on a
    fresh lambda per rollout -- otherwise every rollout pays a full recompile."""
    return jax.vmap(policy)(X)


def rollout(policy, insts, feats, obs_dim, n_out, refuse_index=None, tmax=None,
            verbose=False):
    """Greedy rollout.  Returns (routes, refused, steps_taken)."""
    t0 = time.time()
    N = len(insts)
    T = tmax if tmax is not None else max(int(i.budget) for i in insts)
    H = np.zeros((N, T, obs_dim), dtype=np.float32)
    homes = [key_homes(i) for i in insts]
    gsums = [global_summary(S) for S in feats]
    states = [initial_state(i) for i in insts]
    live = np.ones(N, dtype=bool)
    routes = [[] for _ in range(N)]
    refused = np.zeros(N, dtype=bool)
    for t in range(T):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        for i in idx:
            H[i, t] = obs(insts[i], feats[i], states[i], gsums[i])
        logits = np.asarray(_forward(policy, jnp.asarray(H)))[:, t, :]
        acts = logits.argmax(-1)
        for i in idx:
            a = int(acts[i])
            if refuse_index is not None and a == refuse_index:
                refused[i] = True
                live[i] = False
                continue
            ch = ACTIONS[a]
            routes[i].append(ch)
            nxt, legal, goal = step(insts[i], states[i], ch, homes[i])
            if not legal or goal or len(routes[i]) >= int(insts[i].budget):
                live[i] = False
            states[i] = nxt
    out = ["".join(r) for r in routes]
    if verbose:
        print("    rollout %d instances, T=%d, %.1fs" % (N, T, time.time() - t0))
    return out, refused, [len(r) for r in out]


def score(insts, routes, refused=None, label=""):
    """The exact judge, and only the exact judge.

      validity  fraction of episodes whose route the judge accepts
      score     mean of (valid ? opt_len / len(route) : 0), the env-scout column
    """
    ok_n, sc, reasons = 0, 0.0, Counter()
    goal_lifted, legal_frac = 0, []
    for i, (inst, route) in enumerate(zip(insts, routes)):
        if refused is not None and refused[i]:
            reasons["refused"] += 1
            continue
        ok, why, _ = JUDGE(inst, route)
        if ok:
            ok_n += 1
            sc += inst.opt_len / len(route)
        else:
            reasons[reason_class(why)] += 1
        goal_lifted += JUDGE(_budget_lifted(inst), route)[0]
        _, plen, _ = replay(inst, route)
        legal_frac.append(min(plen, inst.opt_len) / max(inst.opt_len, 1))
    n = len(insts)
    return {"label": label, "n": n, "validity": ok_n / n, "score": sc / n,
            "goal_rate_budget_lifted": goal_lifted / n,
            "mean_legal_prefix_over_opt": float(np.mean(legal_frac)) if legal_frac else 0.0,
            "n_refused": int(refused.sum()) if refused is not None else 0,
            "refusal_rate": float(refused.mean()) if refused is not None else 0.0,
            "failure_modes": dict(reasons.most_common(6))}


HEADER = ("  %-38s %8s %8s %7s %7s   %s"
          % ("arm / column", "validity", "score", "goal*", "legal*", "top failure modes"))
NOTE = ("   * goal  = reached the goal legally with the budget lifted "
        "(exact judge, declared variant);  legal = mean legal prefix / "
        "optimal length.  Both are DIAGNOSTICS -- the scored column is "
        "validity.")


def print_row(res):
    print("  %-38s %8.4f %8.4f %7.4f %7.4f   %s"
          % (res["label"][:38], res["validity"], res["score"],
             res["goal_rate_budget_lifted"], res["mean_legal_prefix_over_opt"],
             ", ".join("%s x%d" % kv for kv in list(res["failure_modes"].items())[:3])))
