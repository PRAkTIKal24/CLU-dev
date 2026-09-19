"""GYM-1 episodes, the held-out split, and the exact novelty checks.

The held-out split has to make paths GENUINELY NEW, not recombined from seen ones.
"Genuinely new" is not asserted here, it is measured, by three nested checks:

  N1  the held-out episode key was never seen in training;
  N2  the held-out optimal POSITION SEQUENCE is not a contiguous subsequence of any
      training position sequence;
  N3  the minimum number of contiguous training fragments (each of length >= MIN_FRAG)
      needed to tile the held-out position sequence exactly -- an exact DP. N3 is the
      anti-recombination number: it upper-bounds what any stitching lookup table can do.

N2/N3 run against a byte corpus of every training path, so "is a contiguous subsequence
of some training path" is an exact substring test, not a sample-based estimate.

The compositional axis is (speed limit V) x (displacement magnitude bucket). See LOG.md
2026-09-20T00:04Z/00:06Z: an exhaustive sweep showed an obstacle on a LINE can never bend
a solvable path (0 of 25,261,855 solvable configurations cost more with the wall than
without), so a wall-based held-out axis would have been built on nothing. The wall stays
in every episode -- validity metrics need a wall a bad model can cross -- and its actual
contribution is measured by an ablation rather than assumed.
"""

from __future__ import annotations

import random
from typing import NamedTuple, Optional, Sequence

from chlu.gym1.line1d import (
    A_MAX, L, Episode, can_park, is_free_cell, plan, positions_of, replay_policy, rollout,
)

# --- configuration (see LOG.md: pre-registration + two logged deviations) -----

V_CHOICES = (2, 3, 4, 5, 6)
WW_CHOICES = (1, 2, 3, 4)
W0_CHOICES = (28, 36, 44, 52, 60, 68, 76, 84)

#: Displacement-magnitude buckets, |g - s|, as (lo, hi) inclusive.
D_BUCKETS = ((3, 10), (11, 20), (21, 32), (33, 48), (49, 70))

#: THE compositional split. Every speed limit and every displacement bucket appears in
#: training -- never in these pairings. A held-out episode therefore asks for a cruise
#: phase held at a speed for a length that was never supervised together.
HELD_VD = ((2, 4), (3, 2), (4, 0), (5, 3), (6, 1))

#: Kept from the pre-registration so the wall's (ir)relevance can be MEASURED.
HELD_COMBOS = ((2, 4), (3, 2), (4, 1), (5, 3), (6, 2))

MIN_PATH = 3
MAX_PATH = 40
HORIZON = 48         # main metric horizon (> MAX_PATH, so parking must be held)
HORIZON_LONG = 96    # persistence probe: 2x the horizon, 2.4x the longest path supervised
MIN_FRAG = 3         # shortest fragment a stitching table is allowed to reuse

SEP = 255            # byte that cannot occur in a position (0..127) or a shifted action


def d_bucket(d: int) -> int:
    for i, (lo, hi) in enumerate(D_BUCKETS):
        if lo <= d <= hi:
            return i
    return -1


def is_held_vd(V: int, d: int) -> bool:
    return (V, d_bucket(d)) in set(HELD_VD)


def train_combos():
    held = set(HELD_COMBOS)
    return [(v, w) for v in V_CHOICES for w in WW_CHOICES if (v, w) not in held]


# ---------------------------------------------------------------------------
# Whether the obstacle did anything. Reported, not assumed.
# ---------------------------------------------------------------------------

def wall_free_plan(ep: Episode):
    """Optimal plan for the same episode with the wall removed."""
    return plan(Episode(ep.V, 0, 0, ep.s, ep.v0, ep.g))


def wall_free_plan_fails(ep: Episode) -> bool:
    """Does *this particular* wall-free optimal plan fail once the wall is put back?

    NOTE this is a statement about the tie-broken representative, not about the
    obstacle: by Finding 2 the optimal COST is never raised by the wall, so whenever
    this is True there exists an equally short wall-respecting plan. Kept because the
    wall ablation needs it.
    """
    pf = wall_free_plan(ep)
    if pf is None:
        return False
    r = rollout(ep, replay_policy(pf), horizon=len(pf), continue_after_fault=True)
    return (not r.all_steps_valid) or (not r.reached_goal)


def wall_raises_cost(ep: Episode) -> bool:
    """Does the obstacle make this solvable episode strictly more expensive?

    Finding 2 says this is False everywhere; it is recomputed per-episode in the
    ablation so the claim is checked on the actual split rather than inherited.
    """
    pw = plan(ep)
    pf = wall_free_plan(ep)
    if pw is None or pf is None:
        return False
    return len(pw) > len(pf)


class EpisodeRecord(NamedTuple):
    ep: Episode
    actions: tuple      # provably optimal action sequence
    positions: tuple    # positions along it, len == len(actions) + 1
    wf_fails: bool      # the tie-broken wall-free plan faults in the walled env
    wall_costs: bool    # the obstacle strictly raises the optimal cost

    @property
    def d(self) -> int:
        return abs(self.ep.g - self.ep.s)


def make_record(ep: Episode) -> Optional[EpisodeRecord]:
    acts = plan(ep)
    if acts is None or not (MIN_PATH <= len(acts) <= MAX_PATH):
        return None
    return EpisodeRecord(
        ep=ep,
        actions=tuple(acts),
        positions=tuple(positions_of(ep, acts)),
        wf_fails=wall_free_plan_fails(ep),
        wall_costs=wall_raises_cost(ep),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sides(w0: int, ww: int):
    """The two free components of the line, as (lo, hi, wall_dir) inclusive ranges."""
    out = []
    if w0 - 1 >= 1:
        out.append((0, w0 - 1, -1))        # wall lies to the RIGHT of this side
    if w0 + ww <= L - 2:
        out.append((w0 + ww, L - 1, +1))   # wall lies to the LEFT of this side
    return out


def sample_split(
    n_target: int,
    seed: int,
    held: bool,
    near_wall_fraction: float = 0.5,
    max_tries_factor: int = 80,
):
    """Draw episodes. `held=True` keeps only held-out (V, displacement-bucket) pairings.

    Generation is grouped by (rule setting, goal) so the BFS distance-field cache in
    `line1d` stays hot: that is what lets ground truth stay exhaustive and still cheap.
    A fraction of starts is placed within braking range of the wall, so that the wall is
    genuinely crossable by a model that ignores it -- otherwise the wall-crossing metric
    would be unable to fire and would prove nothing.
    """
    rng = random.Random(seed)
    out, seen = [], set()
    tries, max_tries = 0, n_target * max_tries_factor

    while len(out) < n_target and tries < max_tries:
        V = V_CHOICES[rng.randrange(len(V_CHOICES))]
        ww = WW_CHOICES[rng.randrange(len(WW_CHOICES))]
        w0 = W0_CHOICES[rng.randrange(len(W0_CHOICES))]
        sides = _sides(w0, ww)
        if not sides:
            continue
        lo, hi, wall_dir = sides[rng.randrange(len(sides))]
        g = rng.randint(lo, hi)

        for _ in range(12):               # reuse this (V, w0, ww, g): one BFS, many episodes
            if len(out) >= n_target:
                break
            tries += 1
            near_wall = rng.random() < near_wall_fraction
            if near_wall:
                band = min(hi - lo, 2 * V + 1)
                if wall_dir < 0:
                    s = rng.randint(max(lo, hi - band), hi)
                    v0 = rng.randint(0, V)          # aimed at the wall
                else:
                    s = rng.randint(lo, min(hi, lo + band))
                    v0 = -rng.randint(0, V)
            else:
                s = rng.randint(lo, hi)
                v0 = rng.randint(-V, V)
            if s == g or not is_free_cell(s, w0, ww):
                continue
            if is_held_vd(V, abs(g - s)) != held:
                continue
            ep = Episode(V=V, w0=w0, ww=ww, s=s, v0=v0, g=g)
            if ep.key() in seen:
                continue
            rec = make_record(ep)
            if rec is None:
                continue
            seen.add(ep.key())
            out.append(rec)
    return out


def sample_refusal(n_target: int, seed: int):
    """Episodes where the goal is UNREACHABLE but a fault-free behaviour still exists.

    This is the only place the obstacle carries information in 1-D (Finding 2), so it is
    where an obstacle-reading model can be distinguished from one that ignores the wall.
    Episodes from which the agent cannot avoid the wall at all are excluded: nothing can
    be measured on an episode with no valid behaviour.
    """
    rng = random.Random(seed)
    out, seen = [], set()
    tries = 0
    while len(out) < n_target and tries < n_target * 200:
        tries += 1
        V = V_CHOICES[rng.randrange(len(V_CHOICES))]
        ww = WW_CHOICES[rng.randrange(len(WW_CHOICES))]
        w0 = W0_CHOICES[rng.randrange(len(W0_CHOICES))]
        if w0 - 1 < 1 or w0 + ww > L - 2:
            continue
        s = rng.randint(0, w0 - 1)
        g = rng.randint(w0 + ww, L - 1)
        v0 = rng.randint(-V, V)
        if rng.random() < 0.5:
            s, g = g, s
        ep = Episode(V=V, w0=w0, ww=ww, s=s, v0=v0, g=g)
        if ep.key() in seen or not is_free_cell(s, w0, ww):
            continue
        if plan(ep) is not None:          # must be genuinely unreachable
            continue
        if not can_park(ep.V, ep.w0, ep.ww, ep.s, ep.v0):
            continue                      # doomed: no valid behaviour exists, measures nothing
        seen.add(ep.key())
        out.append(ep)
    return out


# ---------------------------------------------------------------------------
# Exact novelty machinery: a byte corpus of every training path.
# ---------------------------------------------------------------------------

class Corpus:
    """All training sequences concatenated with a separator byte.

    Membership is Python's exact substring search, so `contains` is an exact statement
    about the whole training set -- no index approximation, no candidate cap.
    """

    def __init__(self, seqs, shift: int = 0):
        self.shift = shift
        parts = [bytes(int(u) + shift for u in s) for s in seqs]
        self.blob = bytes([SEP]).join(parts)
        self._exact = set(parts)

    def contains(self, seq) -> bool:
        return bytes(int(u) + self.shift for u in seq) in self.blob

    def contains_exact_whole(self, seq) -> bool:
        return bytes(int(u) + self.shift for u in seq) in self._exact

    def max_match_from(self, seq, i: int, cap: int) -> int:
        """Longest ell such that seq[i:i+ell] occurs in the corpus (ell <= cap).

        Occurrence is monotone in ell, so the binary search is exact.
        """
        if i >= len(seq):
            return 0
        lo, hi = 0, min(cap, len(seq) - i)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.contains(seq[i:i + mid]):
                lo = mid
            else:
                hi = mid - 1
        return lo


def fragment_cover(seq, corpus: Corpus, min_frag: int = MIN_FRAG) -> Optional[int]:
    """N3: fewest contiguous corpus fragments (each >= min_frag) that tile `seq` exactly.

    Returns None if `seq` cannot be tiled at all under the minimum-fragment rule. Exact
    DP over positions -- this is the number a stitching lookup table would have to
    achieve, so it is reported rather than assumed.
    """
    n = len(seq)
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[n] = 0
    for i in range(n - 1, -1, -1):
        best_len = corpus.max_match_from(seq, i, n - i)
        if best_len < min_frag:
            continue
        for ell in range(min_frag, best_len + 1):
            j = i + ell
            if j != n and n - j < min_frag:   # no leftover tail shorter than min_frag
                continue
            if dp[j] + 1 < dp[i]:
                dp[i] = dp[j] + 1
    return None if dp[0] == INF else int(dp[0])


def novelty_report(held, train, min_frag: int = MIN_FRAG, limit: Optional[int] = None):
    """Run N1/N2/N3 over a held-out split against a training split. Exact."""
    train_keys = {r.ep.key() for r in train}
    pos_corpus = Corpus([r.positions for r in train], shift=0)
    act_corpus = Corpus([r.actions for r in train], shift=A_MAX)   # -1..1 -> 0..2

    sub = held if limit is None else held[:limit]
    n1 = n2 = n2_act = 0
    covers, uncoverable = [], 0
    for r in sub:
        n1 += int(r.ep.key() not in train_keys)
        n2 += int(not pos_corpus.contains(r.positions))
        n2_act += int(not act_corpus.contains(r.actions))
        c = fragment_cover(r.positions, pos_corpus, min_frag)
        if c is None:
            uncoverable += 1
        else:
            covers.append(c)

    n = max(len(sub), 1)
    hist = {}
    for c in covers:
        hist[c] = hist.get(c, 0) + 1
    return {
        "n_checked": len(sub),
        "n_train": len(train),
        "N1_key_novel_frac": n1 / n,
        "N2_position_path_novel_frac": n2 / n,
        "N2_action_seq_novel_frac": n2_act / n,
        "N3_min_fragments_mean": (sum(covers) / len(covers)) if covers else None,
        "N3_min_fragments_hist": dict(sorted(hist.items())),
        "N3_uncoverable_frac": uncoverable / n,
        "min_frag": min_frag,
    }
