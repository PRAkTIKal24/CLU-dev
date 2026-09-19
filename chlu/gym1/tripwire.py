"""The matched-bytes trajectory-lookup TRIPWIRE.

The question this answers, and it has to be answered before the gym is used for anything:

    if a table of stored trajectories, given the same number of bytes as the neural
    baselines have parameters, already solves the held-out pairs, then the held-out pairs
    are not testing composition and this gym is not an instrument.

Four strategies, all scored by REPLAYING what they retrieve in the real environment and
measuring exactly the same validity metrics as a neural model. The headline is the best
of the four, because a weak tripwire is a weak instrument.

  T1  exact key lookup
  T2  nearest neighbour on the absolute key, replay
  T3  nearest neighbour on the RELATIVE key (V, ww, v0, g-s, s-w0), replay.
      The gym is translation-invariant away from the ends of the line, so this key is
      close to a sufficient statistic; the backoff also prefers a LOWER stored speed
      limit (a plan built under a tighter limit is automatically legal under a looser
      one) and a WIDER stored wall (conservative), which is the strongest honest
      retrieval policy I can give it.
  T4  fragment stitching: shortest-path search over macro-actions harvested from the
      training trajectories, reported as a CURVE over the maximum number of fragments
      allowed. K=1 is whole-trajectory replay; large K is re-planning out of stored
      pieces. The curve is what tells you how much recombination the gym costs.

Byte accounting is explicit and reported, including how much of the budget was actually
needed -- if the budget turns out to be slack, the tripwire is not byte-limited at all
and the result is the strongest possible form of the test.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from chlu.gym1.line1d import (
    ACTIONS, L, Episode, replay_policy, rollout, segment_hits_wall,
)

KEY_BYTES_ABS = 6      # V, w0, ww, s, v0, g -- one byte each
KEY_BYTES_REL = 5      # V, ww, v0, g-s, s-w0
LEN_BYTES = 1


def action_bytes(n: int) -> int:
    """Actions live in {-1, 0, 1}: two bits each, packed."""
    return math.ceil(n / 4)


def entry_bytes_abs(traj_len: int) -> int:
    return KEY_BYTES_ABS + LEN_BYTES + action_bytes(traj_len)


def entry_bytes_rel(traj_len: int) -> int:
    return KEY_BYTES_REL + LEN_BYTES + action_bytes(traj_len)


# ---------------------------------------------------------------------------

class TripwireScore(NamedTuple):
    name: str
    n: int
    valid_and_reached: float   # THE number: fault-free AND parked on the goal
    reached: float
    all_valid: float
    held_at_goal: float
    hit_rate: float            # fraction of queries the table answered at all
    bytes_used: int
    bytes_budget: int
    detail: dict


def _score(name, eps, act_seqs, horizon, budget, bytes_used, detail=None):
    n = len(eps)
    vr = rc = av = hg = hit = 0
    for ep, acts in zip(eps, act_seqs):
        if acts is None:
            continue
        hit += 1
        r = rollout(ep, replay_policy(acts), horizon=horizon, continue_after_fault=True)
        rc += r.reached_goal
        av += r.all_steps_valid
        hg += r.held_at_goal
        vr += (r.all_steps_valid and r.reached_goal)
    n1 = max(n, 1)
    return TripwireScore(name, n, vr / n1, rc / n1, av / n1, hg / n1, hit / n1,
                         bytes_used, budget, detail or {})


# ---------------------------------------------------------------------------
# T1 / T2 / T3 -- stored whole trajectories
# ---------------------------------------------------------------------------

def _fit_entries(records, budget, cost_fn):
    """Greedily store training trajectories until the byte budget is exhausted.

    Cheapest-first, so the table holds as many trajectories as the budget allows.
    """
    order = sorted(range(len(records)), key=lambda i: cost_fn(len(records[i].actions)))
    kept, used = [], 0
    for i in order:
        c = cost_fn(len(records[i].actions))
        if used + c > budget:
            break
        kept.append(i)
        used += c
    return kept, used


def t1_exact(train, held, horizon, budget):
    kept, used = _fit_entries(train, budget, entry_bytes_abs)
    table = {train[i].ep.key(): train[i].actions for i in kept}
    outs = [table.get(r.ep.key()) for r in held]
    return _score("T1_exact_key", [r.ep for r in held], outs, horizon, budget, used,
                  {"entries": len(table)})


def _nn_build_abs(train, kept):
    arr = np.array([[train[i].ep.V, train[i].ep.ww, train[i].ep.w0,
                     train[i].ep.s, train[i].ep.v0, train[i].ep.g] for i in kept],
                   dtype=np.int32)
    return arr, [train[i].actions for i in kept]


def t2_nn_absolute(train, held, horizon, budget):
    kept, used = _fit_entries(train, budget, entry_bytes_abs)
    arr, acts = _nn_build_abs(train, kept)
    # Weights make the rule fields dominate: a trajectory from a different rule setting
    # is a worse answer than one from a different place.
    W = np.array([40, 20, 4, 1, 8, 1], dtype=np.int32)
    outs = []
    for r in held:
        q = np.array([r.ep.V, r.ep.ww, r.ep.w0, r.ep.s, r.ep.v0, r.ep.g], dtype=np.int32)
        d = np.abs(arr - q) @ W
        outs.append(acts[int(np.argmin(d))])
    return _score("T2_nn_absolute", [r.ep for r in held], outs, horizon, budget, used,
                  {"entries": len(kept)})


def t3_nn_relative(train, held, horizon, budget):
    """Nearest neighbour on the relative key, with a conservative backoff.

    `V` is compared one-sidedly: a stored trajectory built under a TIGHTER speed limit is
    automatically legal under a looser one, so undershooting V is cheap and overshooting
    is expensive. Likewise a stored trajectory that respected a WIDER wall is safe under
    a narrower one.
    """
    kept, used = _fit_entries(train, budget, entry_bytes_rel)
    arr = np.array([[train[i].ep.V, train[i].ep.ww, train[i].ep.v0,
                     train[i].ep.g - train[i].ep.s, train[i].ep.s - train[i].ep.w0]
                    for i in kept], dtype=np.int32)
    acts = [train[i].actions for i in kept]
    outs = []
    for r in held:
        q = np.array([r.ep.V, r.ep.ww, r.ep.v0, r.ep.g - r.ep.s, r.ep.s - r.ep.w0],
                     dtype=np.int32)
        dV = arr[:, 0] - q[0]
        cV = np.where(dV > 0, 60 * dV, -6 * dV)      # stored V above query V: expensive
        dW = arr[:, 1] - q[1]
        cW = np.where(dW < 0, 25 * (-dW), 3 * dW)    # stored wall narrower: expensive
        cost = (cV + cW
                + 30 * np.abs(arr[:, 2] - q[2])      # initial velocity must match
                + 12 * np.abs(arr[:, 3] - q[3])      # displacement must match to park
                + 1 * np.abs(arr[:, 4] - q[4]))
        outs.append(acts[int(np.argmin(cost))])
    return _score("T3_nn_relative", [r.ep for r in held], outs, horizon, budget, used,
                  {"entries": len(kept)})


# ---------------------------------------------------------------------------
# T4 -- fragment stitching
# ---------------------------------------------------------------------------

class _Frags(NamedTuple):
    v_start: np.ndarray
    v_end: np.ndarray
    dx: np.ndarray
    minoff: np.ndarray
    maxoff: np.ndarray
    maxabsv: np.ndarray
    length: np.ndarray
    seqs: list
    bytes_used: int


def harvest_fragments(train, min_frag=3, max_frag=16, budget=None) -> _Frags:
    """Every contiguous piece of every training trajectory, deduplicated by BEHAVIOUR.

    A fragment's effect is completely described by its starting velocity and its
    per-step velocity profile, so two different action strings with the same profile are
    the same macro-action and only one needs storing. Dominance pruning then keeps, for
    each (v_start, v_end, dx, length), only the piece with the tightest excursion --
    anything wider is strictly worse at avoiding a wall and can never help.
    """
    best = {}
    for rec in train:
        acts = rec.actions
        v_at = [rec.ep.v0]
        x_at = [0]
        v = rec.ep.v0
        x = 0
        for a in acts:
            v += a
            x += v
            v_at.append(v)
            x_at.append(x)
        n = len(acts)
        for i in range(n):
            vs = v_at[i]
            for ell in range(min_frag, min(max_frag, n - i) + 1):
                j = i + ell
                sub = acts[i:j]
                xs = [x_at[t] - x_at[i] for t in range(i, j + 1)]
                vs_prof = v_at[i:j + 1]
                sig = (vs, tuple(vs_prof[1:]))
                key = (vs, v_at[j], xs[-1], ell)
                cand = (max(xs), -min(xs), ell, sub, max(abs(u) for u in vs_prof[1:]),
                        min(xs))
                cur = best.get(key)
                if cur is None or cand[:3] < cur[:3]:
                    best[key] = cand

    items = list(best.items())
    # byte cost: v_start + v_end + packed actions
    def cost(ell):
        return 2 + action_bytes(ell)

    items.sort(key=lambda kv: cost(kv[0][3]))
    used = 0
    v_start, v_end, dx, minoff, maxoff, maxabsv, length, seqs = [], [], [], [], [], [], [], []
    for (vs, ve, d, ell), (mx, negmn, _e, sub, mav, mn) in items:
        c = cost(ell)
        if budget is not None and used + c > budget:
            break
        used += c
        v_start.append(vs)
        v_end.append(ve)
        dx.append(d)
        minoff.append(mn)
        maxoff.append(mx)
        maxabsv.append(mav)
        length.append(ell)
        seqs.append(sub)
    A = lambda z: np.array(z, dtype=np.int32)
    return _Frags(A(v_start), A(v_end), A(dx), A(minoff), A(maxoff), A(maxabsv),
                  A(length), seqs, used)


def _group_by_v(F: _Frags):
    """Fragments bucketed by starting velocity, as contiguous numpy views."""
    groups = {}
    for v in range(-(2 * 8), 2 * 8 + 1):
        idx = np.nonzero(F.v_start == v)[0]
        if idx.size:
            groups[v] = (idx, F.v_end[idx], F.dx[idx], F.minoff[idx], F.maxoff[idx],
                         F.maxabsv[idx], F.length[idx])
    return groups


def _expand(ep: Episode, groups, cost_t, reach, horizon):
    """One layer of the fragment search, vectorised over (source position x fragment)."""
    V, w0, ww = ep.V, ep.w0, ep.ww
    nv = 2 * V + 1
    new_cost = cost_t.copy()
    for v in range(-V, V + 1):
        grp = groups.get(v)
        if grp is None:
            continue
        src = np.nonzero(reach[:, v + V])[0]
        if src.size == 0:
            continue
        idx, ve, dx, mn, mx, mav, ln = grp
        keep = mav <= V
        if not keep.any():
            continue
        ve, dx, mn, mx, ln = ve[keep], dx[keep], mn[keep], mx[keep], ln[keep]
        sc = cost_t[src, v + V]

        lo = src[:, None] + mn[None, :]
        hi = src[:, None] + mx[None, :]
        tot = sc[:, None] + ln[None, :]
        ok = (lo >= 0) & (hi <= L - 1) & (tot <= horizon)
        if ww > 0:
            ok &= ~((lo <= w0 + ww - 1) & (hi >= w0))
        if not ok.any():
            continue
        dst = src[:, None] + dx[None, :]
        tgt = dst * nv + (ve[None, :] + V)
        np.minimum.at(new_cost.reshape(-1), tgt[ok], tot[ok])
    return new_cost


def _stitch_one(ep: Episode, F: _Frags, K: int, horizon: int, groups):
    """Fewest-fragment plan from (s, v0) to (g, 0), or None.

    Validity of a whole fragment is an O(1) test: the union of the swept segments along a
    fragment is exactly [x + minoff, x + maxoff], because consecutive positions are joined
    by their own segment. So a fragment is legal at x iff that interval stays on the line
    and misses the wall, and iff its velocity profile never exceeds V.

    The plan is then REPLAYED in the real environment by the caller, so a mistake in this
    fast test would show up as a search that claims success and a rollout that faults.
    """
    V, g = ep.V, ep.g
    nv = 2 * V + 1
    if not (0 <= ep.s < L) or abs(ep.v0) > V:
        return None
    BIG = 1 << 29
    cost_t = np.full((L, nv), BIG, dtype=np.int32)
    cost_t[ep.s, ep.v0 + V] = 0
    layers = [cost_t]
    for _ in range(K):
        reach = layers[-1] < BIG
        nxt = _expand(ep, groups, layers[-1], reach, horizon)
        layers.append(nxt)
        if nxt[g, V] < BIG:
            break
    if layers[-1][g, V] >= BIG:
        return None

    # Reconstruct backwards: find, at each layer, a fragment and predecessor whose cost
    # exactly accounts for this state's cost.
    pieces = []
    x, v, c = g, 0, int(layers[-1][g, V])
    for k in range(len(layers) - 1, 0, -1):
        prev = layers[k - 1]
        if prev[x, v + V] == c:               # state was already reached earlier
            continue
        found = False
        for vs in range(-V, V + 1):
            grp = groups.get(vs)
            if grp is None:
                continue
            idx, ve, dx, mn, mx, mav, ln = grp
            sel = np.nonzero((ve == v) & (mav <= V))[0]
            if sel.size == 0:
                continue
            px = x - dx[sel]
            good = (px >= 0) & (px < L)
            if not good.any():
                continue
            sel, px = sel[good], px[good]
            pc = prev[px, vs + V]
            hit = np.nonzero(pc + ln[sel] == c)[0]
            if hit.size == 0:
                continue
            j = int(hit[0])
            fid = int(idx[sel[j]])
            pieces.append(F.seqs[fid])
            x, v, c = int(px[j]), vs, int(pc[j])
            found = True
            break
        if not found:
            return None
        if x == ep.s and v == ep.v0 and c == 0:
            break
    if not (x == ep.s and v == ep.v0):
        return None
    seq = []
    for piece in reversed(pieces):
        seq.extend(piece)
    return seq or None


def prune_dominated(F: _Frags) -> _Frags:
    """Keep only Pareto-optimal fragments per (v_start, v_end, dx).

    A fragment that is longer AND has a wider excursion than another with the same
    endpoints and displacement can never help: it costs more steps and is more likely to
    hit the wall. Dropping those shrinks the search without weakening the tripwire.
    """
    by = {}
    for i in range(len(F.seqs)):
        by.setdefault((int(F.v_start[i]), int(F.v_end[i]), int(F.dx[i])), []).append(i)
    keep = []
    for _, ids in by.items():
        cand = sorted(ids, key=lambda i: (int(F.length[i]), int(F.maxoff[i]), -int(F.minoff[i])))
        best_mx, best_mn = None, None
        for i in cand:
            mx, mn = int(F.maxoff[i]), int(F.minoff[i])
            if best_mx is not None and mx >= best_mx and mn <= best_mn:
                continue
            keep.append(i)
            best_mx = mx if best_mx is None else min(best_mx, mx)
            best_mn = mn if best_mn is None else max(best_mn, mn)
    keep = np.array(sorted(keep), dtype=np.int64)
    return _Frags(F.v_start[keep], F.v_end[keep], F.dx[keep], F.minoff[keep],
                  F.maxoff[keep], F.maxabsv[keep], F.length[keep],
                  [F.seqs[int(i)] for i in keep],
                  sum(2 + action_bytes(int(F.length[i])) for i in keep))


def t4_stitch(train, held, horizon, budget, ks=(1, 2, 3, 4, 6, 8), min_frag=3,
              harvest_from=None, sample=None, seed=0):
    """Fragment-stitching tripwire, reported as a curve over the fragment allowance K."""
    src = train if harvest_from is None else train[:harvest_from]
    F0 = harvest_fragments(src, min_frag=min_frag, budget=budget)
    F = prune_dominated(F0)
    groups = _group_by_v(F)
    if sample is not None and sample < len(held):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(held), size=sample, replace=False)
        held = [held[int(i)] for i in sel]
    out = []
    for K in ks:
        seqs = [_stitch_one(r.ep, F, K, horizon, groups) for r in held]
        out.append(_score(f"T4_stitch_K{K}", [r.ep for r in held], seqs, horizon, budget,
                          F.bytes_used,
                          {"fragments_before_prune": len(F0.seqs),
                           "fragments": len(F.seqs), "min_frag": min_frag, "K": K,
                           "harvested_from_episodes": len(src),
                           "held_subsample": len(held)}))
    return out


def run_all(train, held, horizon, budget, ks=(1, 2, 3, 4, 6, 8),
            harvest_from=30000, t4_sample=None):
    scores = [t1_exact(train, held, horizon, budget),
              t2_nn_absolute(train, held, horizon, budget),
              t3_nn_relative(train, held, horizon, budget)]
    scores += t4_stitch(train, held, horizon, budget, ks=ks,
                        harvest_from=harvest_from, sample=t4_sample)
    return scores
