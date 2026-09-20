"""GYM-2 episodes, the compositional split, and the exact novelty checks.

The held-out split has to make paths GENUINELY NEW, not recombined from seen ones. That is
not asserted here, it is measured, and measured against the corpus a stitching lookup table
actually attacks.

  N1   the held-out episode key was never seen in training;
  N2a  the held-out optimal POSITION sequence is not a contiguous substring of any training
       position sequence (absolute -- the parent's check);
  N2b  the held-out ACTION sequence is not a contiguous substring of any training one;
  N2c  the held-out BEHAVIOUR string is not a contiguous substring of any training one.

The behaviour string is the fix for the weakness in the parent's N3, which tiled against
*absolute* positions while its own T4 tripwire harvested *translation-invariant* macro
actions. A fragment's entire effect -- displacement, velocity profile, swept footprint --
is determined by its start velocity plus its action string, so the right object is

    B = v0 a0 v1 a1 ... v_{T-1} a_{T-1} v_T

over disjoint character ranges for velocities and actions. A translation-invariant fragment
covering steps [i, j) is exactly the substring `B[2i : 2j+1]`, and "does the table already
hold it?" is an exact substring test. N3-rel tiles the held-out behaviour string out of
training behaviour substrings; N3-abs does the parent's absolute-position version, so the
two are directly comparable and the gap between them is reportable.

The compositional axis is (speed limit V) x (tunnel-length bucket) -- see `layouts.HELD_VT`.
Every V and every bucket appears in training, never in the held-out pairings. A held-out
episode therefore asks the agent to choose between crawling through a tunnel and detouring
over the block for a (speed limit, tunnel length) combination it was never supervised on,
which is the wave-7 question stated as a split.
"""

from __future__ import annotations

import random
from typing import NamedTuple, Optional

import numpy as np

from chlu.gym2 import layouts as LY
from chlu.gym2.grid2d import (
    ACTIONS, G, V_MAX_MAX, Episode, can_park, get_layout, is_free_cell, opt_cost, plan,
    positions_of, replay_policy, rollout, sidx, _dist_field,
)

MIN_PATH = 10
MAX_PATH = 36
HORIZON = 48          # main metric horizon (> MAX_PATH, so parking must be held)
HORIZON_LONG = 96     # persistence probe: 2x the horizon
MIN_FRAG = 3          # shortest fragment a stitching table is allowed to reuse (parent's)

_AIDX = {a: i for i, a in enumerate(ACTIONS)}

# disjoint character ranges so a behaviour string can never be mis-parsed
_C_POS = 0x100        # positions: _C_POS + x*G + y
_C_VEL = 0x800        # velocities: _C_VEL + (vx+Vm)*(2Vm+1) + (vy+Vm)
_C_ACT = 0x1000       # actions: _C_ACT + action index
_C_SEP = 0x2000


def pos_str(positions) -> str:
    return "".join(chr(_C_POS + x * G + y) for x, y in positions)


def act_str(actions) -> str:
    return "".join(chr(_C_ACT + _AIDX[(int(a[0]), int(a[1]))]) for a in actions)


def vel_char(vx, vy) -> str:
    Vm = V_MAX_MAX
    return chr(_C_VEL + (int(vx) + Vm) * (2 * Vm + 1) + (int(vy) + Vm))


def behaviour_str(vx0, vy0, actions) -> str:
    """v0 a0 v1 a1 ... vT -- the translation-invariant object a stitching table stores."""
    out = [vel_char(vx0, vy0)]
    vx, vy = int(vx0), int(vy0)
    for a in actions:
        out.append(chr(_C_ACT + _AIDX[(int(a[0]), int(a[1]))]))
        vx += int(a[0])
        vy += int(a[1])
        out.append(vel_char(vx, vy))
    return "".join(out)


class EpisodeRecord(NamedTuple):
    ep: Episode
    actions: tuple
    positions: tuple
    cost: int
    cost_tunnel: int      # optimal cost restricted to the tunnel (detour blocked), -1 = none
    cost_detour: int      # optimal cost restricted to the detour (tunnel blocked), -1 = none
    route: str            # "tunnel" / "detour" / "none" (no crossing needed)
    tunnel_len: int
    crossing: bool


def _route_costs(ep: Episode):
    meta = get_layout(ep.lid).meta
    ct = opt_cost(ep, block_extra=meta["block_top"])
    cd = opt_cost(ep, block_extra=meta["block_tunnel"])
    return ct, cd


def _side(lid, x):
    m = get_layout(lid).meta
    if x < m["cx"]:
        return -1
    if x >= m["cx"] + m["tw"]:
        return +1
    return 0


def make_record(ep: Episode, with_routes: bool = True) -> Optional[EpisodeRecord]:
    acts = plan(ep)
    if acts is None or not (MIN_PATH <= len(acts) <= MAX_PATH):
        return None
    meta = get_layout(ep.lid).meta
    crossing = _side(ep.lid, ep.sx) * _side(ep.lid, ep.gx) < 0
    ct = cd = -1
    route = "none"
    if with_routes and crossing:
        ct, cd = _route_costs(ep)
        if ct >= 0 and (cd < 0 or ct <= cd):
            route = "tunnel"
        elif cd >= 0:
            route = "detour"
    return EpisodeRecord(
        ep=ep, actions=tuple(acts), positions=tuple(positions_of(ep, acts)),
        cost=len(acts), cost_tunnel=ct, cost_detour=cd, route=route,
        tunnel_len=meta["tunnel_len"], crossing=bool(crossing),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_split(layout_specs, n_target: int, seed: int, held: bool,
                 crossing_fraction: float = 0.8, with_routes: bool = True,
                 goals_per_lv: int = 10, starts_per_goal: int = 24):
    """Draw episodes. `held=True` keeps only held-out (V, tunnel-bucket) pairings.

    Work is allocated over (layout, V) PAIRS with a per-pair quota, so the split spreads
    across every layout and every admissible speed limit instead of exhausting the first
    few layouts it happens to visit. Within a pair, generation is grouped by goal so the
    transition graph and the backward BFS distance field are each computed once and reused
    for many starts -- that is what lets ground truth stay exhaustive (a complete BFS per
    goal) and still cheap. `crossing_fraction` of episodes put start and goal on opposite
    sides of the block, so the route decision is actually posed.
    """
    rng = random.Random(seed)
    pairs = [(sp, V) for sp in layout_specs for V in LY.V_CHOICES
             if LY.is_held_vt(V, sp.tunnel_len) == held]
    if not pairs:
        return []
    rng.shuffle(pairs)
    out, seen = [], set()
    for rnd in range(6):
        if len(out) >= n_target:
            break
        quota = max(1, -(-(n_target - len(out)) // len(pairs)))
        for sp, V in pairs:
            if len(out) >= n_target:
                break
            got = 0
            lay = get_layout(sp.lid)
            free_xy = np.argwhere(~lay.blocked)
            for _ in range(goals_per_lv):
                if got >= quota or len(out) >= n_target:
                    break
                gx, gy = (int(u) for u in free_xy[rng.randrange(len(free_xy))])
                if _dist_field(sp.lid, V, gx, gy) is None:
                    continue
                want_cross = rng.random() < crossing_fraction
                gside = _side(sp.lid, gx)
                for _ in range(starts_per_goal):
                    if got >= quota or len(out) >= n_target:
                        break
                    sx, sy = (int(u) for u in free_xy[rng.randrange(len(free_xy))])
                    if want_cross and (gside == 0 or _side(sp.lid, sx) != -gside):
                        continue
                    if (sx, sy) == (gx, gy):
                        continue
                    ep = Episode(sp.lid, V, sx, sy,
                                 rng.randint(-V, V), rng.randint(-V, V), gx, gy)
                    if ep.key() in seen:
                        continue
                    rec = make_record(ep, with_routes=with_routes)
                    if rec is None:
                        continue
                    seen.add(ep.key())
                    out.append(rec)
                    got += 1
    return out


def sample_refusal(layout_specs, n_target: int, seed: int, n_sealed: int = 40):
    """Episodes where the goal is UNREACHABLE but a fault-free behaviour still exists.

    Inherited from the parent: it is the one place a rule carries information that goal
    reaching cannot, and it is what proves the collision metric is able to fire rather
    than being structurally unable to. Ordinary GYM-2 layouts are fully connected by
    construction, so these are built on SEALED variants (both crossings closed), where the
    obstacle map is the only thing that says the goal cannot be reached. Episodes with no
    valid behaviour at all are excluded -- nothing can be measured on them.
    """
    rng = random.Random(seed)
    picks = rng.sample(range(len(layout_specs)), min(n_sealed, len(layout_specs)))
    sealed = [(LY.make_sealed_layout(layout_specs[i].lid), layout_specs[i]) for i in picks]
    out, seen = [], set()
    tries = 0
    while len(out) < n_target and tries < n_target * 60:
        tries += 1
        slid, sp = sealed[rng.randrange(len(sealed))]
        V = LY.V_CHOICES[rng.randrange(len(LY.V_CHOICES))]
        lay = get_layout(slid)
        free_xy = np.argwhere(~lay.blocked)
        sx, sy = (int(u) for u in free_xy[rng.randrange(len(free_xy))])
        gx, gy = (int(u) for u in free_xy[rng.randrange(len(free_xy))])
        if (sx < sp.cx) == (gx < sp.cx):
            continue                       # same component: the goal would be reachable
        vx0, vy0 = rng.randint(-V, V), rng.randint(-V, V)
        ep = Episode(slid, V, sx, sy, vx0, vy0, gx, gy)
        if ep.key() in seen:
            continue
        if plan(ep) is not None:
            continue                       # must be genuinely unreachable
        if not can_park(slid, V, sx, sy, vx0, vy0):
            continue                       # doomed: measures nothing
        seen.add(ep.key())
        out.append(ep)
    return out


# ---------------------------------------------------------------------------
# Exact novelty machinery.
# ---------------------------------------------------------------------------

class Corpus:
    """All training sequences joined by a separator character.

    Membership is Python's exact substring search, so `contains` is an exact statement
    about the whole training set -- no index approximation and no candidate cap.
    """

    def __init__(self, seqs):
        self.blob = chr(_C_SEP).join(seqs)
        self._exact = set(seqs)

    def contains(self, s: str) -> bool:
        return s in self.blob

    def contains_exact_whole(self, s: str) -> bool:
        return s in self._exact

    def max_match_from(self, s: str, i: int, cap: int, stride: int = 1,
                       extra: int = 0) -> int:
        """Longest `ell` (in units of `stride`) with `s[i*stride : i*stride+ell*stride+extra]`
        present. Occurrence is monotone in `ell`, so the binary search is exact."""
        n_units = (len(s) - extra - i * stride) // stride
        lo, hi = 0, min(cap, n_units)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.contains(s[i * stride: i * stride + mid * stride + extra]):
                lo = mid
            else:
                hi = mid - 1
        return lo


def fragment_cover(s: str, corpus: Corpus, n_units: int, min_frag: int,
                   stride: int = 1, extra: int = 0) -> Optional[int]:
    """Fewest contiguous corpus fragments (each >= `min_frag` units) tiling `s` exactly.

    Exact DP over unit positions. `stride`/`extra` let the same DP run over an absolute
    position string (stride 1, extra 0 -- but tiling a path of `n` steps means `n+1`
    positions, handled by the caller) and over a behaviour string (stride 2, extra 1: a
    fragment spanning steps [i, j) is `B[2i : 2j+1]`).

    Returns None when the sequence cannot be tiled at all under the minimum-fragment rule.
    This is the number a stitching lookup table would have to achieve, so it is reported
    rather than assumed.
    """
    INF = float("inf")
    dp = [INF] * (n_units + 1)
    dp[n_units] = 0
    for i in range(n_units - 1, -1, -1):
        best = corpus.max_match_from(s, i, n_units - i, stride, extra)
        if best < min_frag:
            continue
        for ell in range(min_frag, best + 1):
            j = i + ell
            if j != n_units and n_units - j < min_frag:
                continue
            if dp[j] + 1 < dp[i]:
                dp[i] = dp[j] + 1
    return None if dp[0] == INF else int(dp[0])


def novelty_report(held, train, min_frag: int = MIN_FRAG, limit: Optional[int] = None,
                   coarse_min_frag: Optional[int] = None):
    """N1/N2a/N2b/N2c and N3 both ways. Exact throughout."""
    train_keys = {r.ep.key() for r in train}
    pos_corpus = Corpus([pos_str(r.positions) for r in train])
    act_corpus = Corpus([act_str(r.actions) for r in train])
    beh_corpus = Corpus([behaviour_str(r.ep.vx0, r.ep.vy0, r.actions) for r in train])

    sub = held if limit is None else held[:limit]
    n1 = n2a = n2b = n2c = 0
    cov_abs, cov_rel = [], []
    unc_abs = unc_rel = 0
    coarse_ok = 0
    lens = []
    for r in sub:
        ps = pos_str(r.positions)
        as_ = act_str(r.actions)
        bs = behaviour_str(r.ep.vx0, r.ep.vy0, r.actions)
        T = len(r.actions)
        lens.append(T)
        n1 += int(r.ep.key() not in train_keys)
        n2a += int(not pos_corpus.contains(ps))
        n2b += int(not act_corpus.contains(as_))
        n2c += int(not beh_corpus.contains(bs))

        ca = fragment_cover(ps, pos_corpus, len(r.positions), min_frag, 1, 0)
        if ca is None:
            unc_abs += 1
        else:
            cov_abs.append(ca)
        cr = fragment_cover(bs, beh_corpus, T, min_frag, 2, 1)
        if cr is None:
            unc_rel += 1
        else:
            cov_rel.append(cr)
        if coarse_min_frag:
            c2 = fragment_cover(bs, beh_corpus, T, coarse_min_frag, 2, 1)
            coarse_ok += int(c2 is not None)

    n = max(len(sub), 1)
    hist = lambda cs: {int(k): int(v) for k, v in
                       sorted({c: cs.count(c) for c in set(cs)}.items())}
    med = lambda cs: float(np.median(cs)) if cs else None
    return {
        "n_checked": len(sub),
        "n_train": len(train),
        "path_len_mean": float(np.mean(lens)) if lens else None,
        "path_len_median": float(np.median(lens)) if lens else None,
        "N1_key_novel_frac": n1 / n,
        "N2a_position_path_novel_frac": n2a / n,
        "N2b_action_seq_novel_frac": n2b / n,
        "N2c_behaviour_novel_frac": n2c / n,
        "min_frag": min_frag,
        "N3abs_mean": float(np.mean(cov_abs)) if cov_abs else None,
        "N3abs_median": med(cov_abs),
        "N3abs_hist": hist(cov_abs),
        "N3abs_uncoverable_frac": unc_abs / n,
        "N3rel_mean": float(np.mean(cov_rel)) if cov_rel else None,
        "N3rel_median": med(cov_rel),
        "N3rel_hist": hist(cov_rel),
        "N3rel_uncoverable_frac": unc_rel / n,
        "coarse_min_frag": coarse_min_frag,
        "coarse_coverable_frac": (coarse_ok / n) if coarse_min_frag else None,
    }
