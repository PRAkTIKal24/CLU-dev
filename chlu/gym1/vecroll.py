"""Batched, exact rollout and metrics.

This is the same arithmetic as `line1d.rollout`, vectorised over episodes so that a
neural policy can be rolled out on thousands of episodes at once. It is not an
approximation of the scalar environment: `check_against_scalar` asserts the two agree
episode-for-episode on every reported field, and the experiment runs that check before
trusting any number.

Nothing here masks, clamps or projects an action. Validity is measured, never imposed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from chlu.gym1.line1d import L, Episode, replay_policy, rollout


class VecMetrics(NamedTuple):
    n: int
    horizon: int
    # exact validity counts, summed over all steps of all episodes
    n_speed: int
    n_wall: int
    n_teleport: int
    n_steps_total: int
    # episode-level rates
    frac_any_speed: float
    frac_any_wall: float
    frac_any_teleport: float
    all_steps_valid: float
    reached_goal: float
    held_at_goal: float
    valid_and_reached: float
    # persistence
    survival: np.ndarray        # survival[t] = frac of episodes fault-free through step t+1
    first_fault_hist: dict
    first_fault_kind: dict
    at_goal_frac: np.ndarray    # at_goal_frac[t] = frac parked on the goal at step t+1


def pack(eps):
    a = lambda f: np.array([f(e) for e in eps], dtype=np.int64)
    return dict(V=a(lambda e: e.V), w0=a(lambda e: e.w0), ww=a(lambda e: e.ww),
                s=a(lambda e: e.s), v0=a(lambda e: e.v0), g=a(lambda e: e.g))


def vec_rollout(eps, policy_fn, horizon: int):
    """`policy_fn(t, x, v, P) -> int array of actions`, one per episode.

    `x` and `v` are the TRUE states. Whatever comes back is applied verbatim.
    """
    P = pack(eps)
    n = len(eps)
    V, w0, ww, g = P["V"], P["w0"], P["ww"], P["g"]
    x, v = P["s"].copy(), P["v0"].copy()

    n_speed = np.zeros(n, dtype=np.int64)
    n_wall = np.zeros(n, dtype=np.int64)
    n_tel = np.zeros(n, dtype=np.int64)
    clean = np.ones(n, dtype=bool)
    first_step = np.full(n, -1, dtype=np.int64)
    first_kind = np.zeros(n, dtype=np.int64)     # 1 speed, 2 wall, 3 teleport
    reached = np.zeros(n, dtype=bool)
    survival = np.zeros(horizon, dtype=np.float64)
    at_goal = np.zeros(horizon, dtype=np.float64)
    wall_hi = w0 + ww - 1

    for t in range(horizon):
        a = np.asarray(policy_fn(t, x, v, P), dtype=np.int64)
        v2 = v + a
        x2 = x + v2
        lo = np.minimum(x, x2)
        hi = np.maximum(x, x2)
        f_speed = np.abs(v2) > V
        f_wall = (ww > 0) & (lo <= wall_hi) & (hi >= w0)
        f_tel = (x2 < 0) | (x2 >= L)
        any_f = f_speed | f_wall | f_tel

        n_speed += f_speed
        n_wall += f_wall
        n_tel += f_tel

        newly = any_f & (first_step < 0)
        first_step = np.where(newly, t + 1, first_step)
        kind = np.where(f_speed, 1, np.where(f_wall, 2, 3))   # fixed attribution order
        first_kind = np.where(newly, kind, first_kind)
        clean &= ~any_f

        # continue-after-fault: put a teleported agent back on the line so the rest of the
        # horizon is still counted. The teleport is already recorded and the episode is
        # already marked invalid, so no metric is softened.
        x2 = np.where(f_tel, np.clip(x2, 0, L - 1), x2)
        v2 = np.where(f_tel, 0, v2)
        x, v = x2, v2

        parked = (x == g) & (v == 0)
        reached |= clean & parked
        survival[t] = clean.mean()
        at_goal[t] = (clean & parked).mean()

    held = clean & (x == g) & (v == 0) & reached
    fh, fk = {}, {}
    for s_ in np.unique(first_step):
        fh[int(s_)] = int((first_step == s_).sum())
    for k_ in np.unique(first_kind[first_step > 0]):
        fk[{1: "speed_limit_violation", 2: "wall_crossing", 3: "teleport"}[int(k_)]] = \
            int((first_kind[first_step > 0] == k_).sum())

    return VecMetrics(
        n=n, horizon=horizon,
        n_speed=int(n_speed.sum()), n_wall=int(n_wall.sum()), n_teleport=int(n_tel.sum()),
        n_steps_total=int(n * horizon),
        frac_any_speed=float((n_speed > 0).mean()),
        frac_any_wall=float((n_wall > 0).mean()),
        frac_any_teleport=float((n_tel > 0).mean()),
        all_steps_valid=float(clean.mean()),
        reached_goal=float(reached.mean()),
        held_at_goal=float(held.mean()),
        valid_and_reached=float((clean & reached).mean()),
        survival=survival, first_fault_hist=fh, first_fault_kind=fk,
        at_goal_frac=at_goal,
    )


def check_against_scalar(eps, action_seqs, horizon: int) -> dict:
    """Assert the batched environment equals the scalar one, field by field.

    Run before any reported number depends on `vec_rollout`.
    """
    def pol(t, x, v, P):
        return np.array([seq[t] if t < len(seq) else 0 for seq in action_seqs],
                        dtype=np.int64)

    vm = vec_rollout(eps, pol, horizon)
    agree = {"n": len(eps), "mismatches": []}
    sc_valid = sc_reach = sc_held = 0
    sc_sp = sc_wa = sc_te = 0
    for ep, seq in zip(eps, action_seqs):
        r = rollout(ep, replay_policy(seq), horizon=horizon, continue_after_fault=True)
        sc_valid += r.all_steps_valid
        sc_reach += r.reached_goal
        sc_held += r.held_at_goal
        sc_sp += r.n_speed
        sc_wa += r.n_wall
        sc_te += r.n_teleport
    n = len(eps)
    pairs = [("all_steps_valid", vm.all_steps_valid, sc_valid / n),
             ("reached_goal", vm.reached_goal, sc_reach / n),
             ("held_at_goal", vm.held_at_goal, sc_held / n),
             ("n_speed", vm.n_speed, sc_sp),
             ("n_wall", vm.n_wall, sc_wa),
             ("n_teleport", vm.n_teleport, sc_te)]
    for name, a, b in pairs:
        if abs(float(a) - float(b)) > 1e-12:
            agree["mismatches"].append({"field": name, "vec": float(a), "scalar": float(b)})
    agree["ok"] = not agree["mismatches"]
    return agree
