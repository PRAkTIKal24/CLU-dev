"""Batched, exact rollout and metrics for GYM-2.

This is the same arithmetic as `grid2d.rollout`, vectorised over episodes so a neural
policy can be rolled out on thousands of episodes at once. It is not an approximation of
the scalar environment: `check_against_scalar` asserts the two agree episode-for-episode
on every reported field, and the experiments run that check before trusting any number.

The swept-cell test is done exactly, by grouping the batch on the step displacement and
applying that displacement's exact offset list -- including displacements larger than any
legal speed, which an unmasked policy can and does produce. Nothing here masks, clamps or
projects an action. Validity is measured, never imposed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from chlu.gym2.grid2d import (
    G, SWEEP, Episode, _sweep_offsets, get_layout, n_layouts, replay_policy, rollout,
)
from chlu.gym2.models import V_EMB as V_EMB_CLIP


class VecMetrics(NamedTuple):
    n: int
    horizon: int
    n_speed: int
    n_obstacle: int
    n_out: int
    n_steps_total: int
    frac_any_speed: float
    frac_any_obstacle: float
    frac_any_out: float
    all_steps_valid: float
    reached_goal: float
    held_at_goal: float
    valid_and_reached: float
    survival: np.ndarray
    first_fault_hist: dict
    first_fault_kind: dict
    at_goal_frac: np.ndarray
    max_abs_velocity: int
    frac_v_past_embed_clip: float
    clean_flags: np.ndarray        # per-episode: fault-free through the whole horizon
    reached_flags: np.ndarray      # per-episode: parked on the goal with a clean prefix


def blocked_stack():
    """(n_layouts, G, G) bool. Built once so the batched step is a single gather."""
    return np.stack([get_layout(i).blocked for i in range(n_layouts())], 0)


def pack(eps):
    a = lambda f: np.array([f(e) for e in eps], dtype=np.int64)
    return dict(lid=a(lambda e: e.lid), V=a(lambda e: e.V),
                sx=a(lambda e: e.sx), sy=a(lambda e: e.sy),
                vx0=a(lambda e: e.vx0), vy0=a(lambda e: e.vy0),
                gx=a(lambda e: e.gx), gy=a(lambda e: e.gy))


def _sweep_faults(BL, lid, x, y, dx, dy):
    """Exact (hit_obstacle, out_of_bounds) for a whole batch, grouped by displacement."""
    n = x.shape[0]
    hit = np.zeros(n, dtype=bool)
    oob = np.zeros(n, dtype=bool)
    key = (dx.astype(np.int64) << np.int64(20)) + dy.astype(np.int64)
    for k in np.unique(key):
        m = key == k
        d = (int(dx[m][0]), int(dy[m][0]))
        off = SWEEP.get(d)
        if off is None:
            off = _sweep_offsets(*d)
        xs, ys, ls = x[m], y[m], lid[m]
        h = np.zeros(xs.shape[0], dtype=bool)
        o = np.zeros(xs.shape[0], dtype=bool)
        for oi, oj in off:
            cx, cy = xs + oi, ys + oj
            inb = (cx >= 0) & (cx < G) & (cy >= 0) & (cy < G)
            o |= ~inb
            h |= inb & BL[ls, np.clip(cx, 0, G - 1), np.clip(cy, 0, G - 1)]
        hit[m] = h
        oob[m] = o
    return hit, oob


def vec_rollout(eps, policy_fn, horizon: int, BL=None):
    """`policy_fn(t, x, y, vx, vy, P) -> (n, 2) int array`. Applied verbatim."""
    P = pack(eps)
    n = len(eps)
    if BL is None:
        BL = blocked_stack()
    lid, V, gx, gy = P["lid"], P["V"], P["gx"], P["gy"]
    x, y = P["sx"].copy(), P["sy"].copy()
    vx, vy = P["vx0"].copy(), P["vy0"].copy()

    n_speed = np.zeros(n, dtype=np.int64)
    n_obs = np.zeros(n, dtype=np.int64)
    n_out = np.zeros(n, dtype=np.int64)
    clean = np.ones(n, dtype=bool)
    first_step = np.full(n, -1, dtype=np.int64)
    first_kind = np.zeros(n, dtype=np.int64)     # 1 speed, 2 obstacle, 3 out
    reached = np.zeros(n, dtype=bool)
    survival = np.zeros(horizon, dtype=np.float64)
    at_goal = np.zeros(horizon, dtype=np.float64)
    max_abs_v = np.maximum(np.abs(vx), np.abs(vy))

    for t in range(horizon):
        a = np.asarray(policy_fn(t, x, y, vx, vy, P), dtype=np.int64)
        vx2 = vx + a[:, 0]
        vy2 = vy + a[:, 1]
        x2, y2 = x + vx2, y + vy2
        f_speed = (np.abs(vx2) > V) | (np.abs(vy2) > V)
        f_obs, f_out = _sweep_faults(BL, lid, x, y, vx2, vy2)
        any_f = f_speed | f_obs | f_out

        n_speed += f_speed
        n_obs += f_obs
        n_out += f_out
        newly = any_f & (first_step < 0)
        first_step = np.where(newly, t + 1, first_step)
        kind = np.where(f_speed, 1, np.where(f_obs, 2, 3))   # fixed attribution order
        first_kind = np.where(newly, kind, first_kind)
        clean &= ~any_f

        off_grid = f_out | (x2 < 0) | (x2 >= G) | (y2 < 0) | (y2 >= G)
        x2 = np.where(off_grid, np.clip(x2, 0, G - 1), x2)
        y2 = np.where(off_grid, np.clip(y2, 0, G - 1), y2)
        vx2 = np.where(off_grid, 0, vx2)
        vy2 = np.where(off_grid, 0, vy2)
        x, y, vx, vy = x2, y2, vx2, vy2
        max_abs_v = np.maximum(max_abs_v, np.maximum(np.abs(vx), np.abs(vy)))

        parked = (x == gx) & (y == gy) & (vx == 0) & (vy == 0)
        reached |= clean & parked
        survival[t] = clean.mean()
        at_goal[t] = (clean & parked).mean()

    held = clean & (x == gx) & (y == gy) & (vx == 0) & (vy == 0) & reached
    fh = {int(s): int((first_step == s).sum()) for s in np.unique(first_step)}
    names = {1: "speed_limit_violation", 2: "obstacle_collision", 3: "out_of_bounds"}
    fk = {names[int(k)]: int((first_kind[first_step > 0] == k).sum())
          for k in np.unique(first_kind[first_step > 0])}

    return VecMetrics(
        n=n, horizon=horizon,
        n_speed=int(n_speed.sum()), n_obstacle=int(n_obs.sum()), n_out=int(n_out.sum()),
        n_steps_total=int(n * horizon),
        frac_any_speed=float((n_speed > 0).mean()),
        frac_any_obstacle=float((n_obs > 0).mean()),
        frac_any_out=float((n_out > 0).mean()),
        all_steps_valid=float(clean.mean()),
        reached_goal=float(reached.mean()),
        held_at_goal=float(held.mean()),
        valid_and_reached=float((clean & reached).mean()),
        survival=survival, first_fault_hist=fh, first_fault_kind=fk,
        at_goal_frac=at_goal,
        max_abs_velocity=int(max_abs_v.max()),
        frac_v_past_embed_clip=float((max_abs_v > V_EMB_CLIP).mean()),
        clean_flags=clean.copy(), reached_flags=reached.copy(),
    )


def check_against_scalar(eps, action_seqs, horizon: int) -> dict:
    """Assert the batched environment equals the scalar one, field by field."""
    def pol(t, x, y, vx, vy, P):
        return np.array([seq[t] if t < len(seq) else (0, 0) for seq in action_seqs],
                        dtype=np.int64)

    vm = vec_rollout(eps, pol, horizon)
    sc_valid = sc_reach = sc_held = sc_sp = sc_ob = sc_ou = 0
    for ep, seq in zip(eps, action_seqs):
        r = rollout(ep, replay_policy(seq), horizon=horizon, continue_after_fault=True)
        sc_valid += r.all_steps_valid
        sc_reach += r.reached_goal
        sc_held += r.held_at_goal
        sc_sp += r.n_speed
        sc_ob += r.n_obstacle
        sc_ou += r.n_out
    n = len(eps)
    pairs = [("all_steps_valid", vm.all_steps_valid, sc_valid / n),
             ("reached_goal", vm.reached_goal, sc_reach / n),
             ("held_at_goal", vm.held_at_goal, sc_held / n),
             ("n_speed", vm.n_speed, sc_sp),
             ("n_obstacle", vm.n_obstacle, sc_ob),
             ("n_out", vm.n_out, sc_ou)]
    agree = {"n": n, "mismatches": []}
    for name, a, b in pairs:
        if abs(float(a) - float(b)) > 1e-12:
            agree["mismatches"].append({"field": name, "vec": float(a), "scalar": float(b)})
    agree["ok"] = not agree["mismatches"]
    return agree
