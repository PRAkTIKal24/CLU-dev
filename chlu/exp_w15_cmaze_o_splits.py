"""CMAZE-O's hold-outs: VALUE, unseen 2-way, unseen n-way -- and the disjointness proof.

The four rows
-------------
``seen``          training requests, at the words they were written under.
``unseen_2way``   requests carrying a PAIR of words ``(f_i = v_i, f_j = v_j)`` that never
                  co-occurs in training, while each word alone is heavily trained.
``unseen_nway``   requests carrying a 4-TUPLE of words over
                  ``(layout, agent_zone, key_colour, tint_colour)`` that never occurs in
                  training, while every 3-way sub-tuple of it does.  **n = 4** factors
                  jointly composed in this row; the vocabulary's own n is 9.
``value_holdout`` requests carrying colour word 5, which appears in **no** training
                  request in **any** of the four colour slots -- the task file's literal
                  "one colour never seen".
``value_slot_holdout``  requests whose AGENT colour is word 4, which appears in training
                  on keys, on doors and on tinted floors but **never on an agent**.

Why BOTH value rows ship
------------------------
``value_holdout`` is the strict reading and it is the right instrument check: a colour
that was never once rendered cannot be rendered, and an arm that scores above zero on it
is reading something it should not have.  But it cannot separate two arms, because no
arm can pass it.  ``value_slot_holdout`` is the reading that can: the colour word IS
grounded, the slot IS grounded, and only their combination is new.  It is a real
composition question **only because the item tensor factors colour out of the slot**
(see ``exp_w15_cmaze_o_env``'s channel layout); with a per-slot colour block it would be
structurally zero and would say nothing about any arm.

Why the rows are SAMPLED CONDITIONALLY
---------------------------------------
A uniform pool cannot supply an n-way row: the 4-tuple ``(layout, agent_zone,
key_colour, tint_colour)`` has 1,728 combinations, so a uniform draw lands on a
particular held-out one about once in 1,728.  So each held-out row is drawn with its
defining words FIXED and the remaining words uniform.  The train pool is drawn uniformly
and then filtered against every hold-out pattern, which is what makes the disjointness a
property of the CONSTRUCTION rather than of a check run afterwards.

What "proven by construction" means here, concretely
-----------------------------------------------------
:func:`disjointness_report` prints, and :mod:`chlu.exp_w15_cmaze_o_readiness` asserts:

1. the number of training requests matching each hold-out pattern -- must be 0;
2. the number of exact 9-word duplicates between each held-out row and train -- must be 0;
3. the TRAIN SUPPORT of every word and every ``(n-1)``-way sub-tuple of each held-out
   pattern -- must be > 0, otherwise the row is not a composition hold-out at all, it is
   just an unknown word;
4. the train COVERAGE of the n-way factor set (what fraction of its 1,728 combinations
   train does see).  Coverage near 1 is what makes the held-out tuples special rather
   than merely absent by sampling accident.

Only ADMISSIBLE requests are ever put in any list, so no impossible request is ever
shown to a non-refuser arm -- see ``exp_w15_cmaze_o_env.admissible``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chlu.exp_w15_cmaze_o_env import (N_COLOURS, N_ZONES, ORequest,
                                      VALUE_HOLDOUT_COLOUR, admissible)

#: the trained colour words (the VALUE hold-out colour is excluded from all of them)
TRAINED_COLOURS = tuple(range(1, VALUE_HOLDOUT_COLOUR))
#: the colour word that is trained on keys, doors and tints but NEVER on an agent
AGENT_SLOT_HOLDOUT_COLOUR = 4
#: the colour words an AGENT may carry in training
TRAINED_AGENT_COLOURS = tuple(c for c in TRAINED_COLOURS
                              if c != AGENT_SLOT_HOLDOUT_COLOUR)

#: the factor pairs the 2-way hold-out is drawn from, by factor index into FACTORS
TWO_WAY_PAIRS = ((0, 5), (0, 7), (5, 8), (2, 3), (1, 4), (6, 7))

#: the factor set the n-way hold-out is drawn from.  n = 4.
NWAY_FACTORS = (0, 5, 2, 4)          # layout, agent_zone, key_colour, tint_colour
NWAY_N = len(NWAY_FACTORS)


def _cardinality(f: int, n_layouts: int) -> int:
    if f == 0:
        return n_layouts
    if f == 1:
        return len(TRAINED_AGENT_COLOURS)
    if f in (2, 3, 4):
        return len(TRAINED_COLOURS)
    return N_ZONES


def _values(f: int, n_layouts: int):
    if f == 0:
        return list(range(n_layouts))
    if f == 1:
        return list(TRAINED_AGENT_COLOURS)
    if f in (2, 3, 4):
        return list(TRAINED_COLOURS)
    return list(range(N_ZONES))


def _draw(rng, n_layouts) -> ORequest:
    ca = int(TRAINED_AGENT_COLOURS[int(rng.integers(len(TRAINED_AGENT_COLOURS)))])
    c = [int(TRAINED_COLOURS[int(rng.integers(len(TRAINED_COLOURS)))])
         for _ in range(3)]
    z = [int(rng.integers(N_ZONES)) for _ in range(4)]
    return ORequest(int(rng.integers(n_layouts)), ca, c[0], c[1], c[2],
                    z[0], z[1], z[2], z[3])


def _with(req: ORequest, assign: dict) -> ORequest:
    w = list(req.words())
    for f, v in assign.items():
        w[f] = int(v)
    return ORequest(*w)


@dataclass
class Holdouts:
    value_colour: int = VALUE_HOLDOUT_COLOUR
    two_way: tuple = ()          # ((f_i, v_i, f_j, v_j), ...)
    n_way: tuple = ()            # ((v_f0, v_f1, v_f2, v_f3), ...) over NWAY_FACTORS

    def matches_two_way(self, req: ORequest) -> bool:
        w = req.words()
        return any(w[fi] == vi and w[fj] == vj for (fi, vi, fj, vj) in self.two_way)

    def matches_n_way(self, req: ORequest) -> bool:
        w = req.words()
        return any(tuple(w[f] for f in NWAY_FACTORS) == t for t in self.n_way)

    def carries_value(self, req: ORequest) -> bool:
        w = req.words()
        return any(w[f] == self.value_colour for f in (1, 2, 3, 4))

    def carries_agent_slot_holdout(self, req: ORequest) -> bool:
        return int(req.ca) == AGENT_SLOT_HOLDOUT_COLOUR

    def is_train_eligible(self, req: ORequest) -> bool:
        return not (self.carries_value(req) or self.carries_agent_slot_holdout(req)
                    or self.matches_two_way(req) or self.matches_n_way(req))


def choose_holdouts(rng, n_layouts: int, n_two: int = 6, n_n: int = 8) -> Holdouts:
    """Pick the held-out patterns BEFORE any request is drawn."""
    two = []
    for (fi, fj) in TWO_WAY_PAIRS[:n_two]:
        vi = int(rng.choice(_values(fi, n_layouts)))
        vj = int(rng.choice(_values(fj, n_layouts)))
        two.append((fi, vi, fj, vj))
    nway = set()
    while len(nway) < n_n:
        nway.add(tuple(int(rng.choice(_values(f, n_layouts))) for f in NWAY_FACTORS))
    return Holdouts(VALUE_HOLDOUT_COLOUR, tuple(two), tuple(sorted(nway)))


@dataclass
class Pool:
    bank: list = field(default_factory=list)
    hold: Holdouts = None
    train: list = field(default_factory=list)
    rows: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def _sample_until(rng, bank, hold, n_want, gen, accept, max_draws, want_lb=None,
                  seen=None):
    """Rejection sampler over requests: draws, tests admissibility, dedupes.

    ``want_lb=True`` keeps only requests whose door is LOAD-BEARING, ``False`` only
    those whose door is not, ``None`` keeps either.  That is the stratum machinery:
    the load-bearing door is a 5 %-of-uniform event and would otherwise have no n.
    """
    seen = set() if seen is None else seen
    out, draws = [], 0
    while len(out) < n_want and draws < max_draws:
        draws += 1
        r = gen(rng)
        w = r.words()
        if w in seen:
            continue
        if not accept(r):
            continue
        ok, info = admissible(bank[r.layout], r)
        if not ok:
            continue
        lb = bool(info["door_load_bearing"])
        if want_lb is not None and lb != want_lb:
            continue
        seen.add(w)
        out.append((r, lb))
    return out, draws


def _stratified(rng, bank, hold, n_want, gen, accept, frac_lb, seen=None,
                budget=800):
    """``n_want`` admissible requests, ``frac_lb`` of them with a load-bearing door."""
    seen = set() if seen is None else seen
    n_lb = int(round(frac_lb * n_want))
    a, d1 = _sample_until(rng, bank, hold, n_lb, gen, accept, budget * max(1, n_lb),
                          want_lb=True, seen=seen)
    b, d2 = _sample_until(rng, bank, hold, n_want - len(a), gen, accept,
                          budget * max(1, n_want), want_lb=None, seen=seen)
    return a + b, int(d1 + d2)


def build_pool(bank, seed: int, n_train: int = 8000, n_row: int = 400,
               n_seen: int = 600, load_bearing_frac: float = 0.25,
               n_two: int = 6, n_n: int = 8, verbose: bool = True) -> Pool:
    """The whole split for one seed: hold-outs, a train pool, and the four rows."""
    n_layouts = len(bank)
    rng = np.random.default_rng(10_000 + int(seed))
    hold = choose_holdouts(rng, n_layouts, n_two, n_n)

    # ---- train pool, stratified so the load-bearing door is a reportable breakout
    gen = lambda r: _draw(r, n_layouts)
    train_all, d1 = _stratified(rng, bank, hold, n_train, gen,
                                hold.is_train_eligible, load_bearing_frac)
    d2 = 0
    seen_words = {r.words() for (r, _f) in train_all}
    train = [r for (r, _f) in train_all]
    train_lb = {r.words(): f for (r, f) in train_all}

    # ---- PRUNE any hold-out pattern train cannot ground
    #
    # A hold-out is only a COMPOSITION hold-out if its parts are trained.  A pattern
    # whose word, or whose (n-1)-way sub-tuple, happens to have no training support is
    # not a composition question at all -- it is an unknown word -- so it is dropped
    # here, BEFORE any evaluation request is drawn, and the drop is reported.  Train is
    # already filtered against it, which only makes the disjointness stronger.
    Wtr = np.array([r.words() for r in train], np.int64)
    kept_two, dropped_two = [], []
    for (fi, vi, fj, vj) in hold.two_way:
        if (Wtr[:, fi] == vi).sum() > 0 and (Wtr[:, fj] == vj).sum() > 0:
            kept_two.append((fi, vi, fj, vj))
        else:
            dropped_two.append((fi, vi, fj, vj))
    cols = [Wtr[:, f] for f in NWAY_FACTORS]
    kept_n, dropped_n = [], []
    for t in hold.n_way:
        ok = True
        for drop in range(NWAY_N):
            keep_i = [i for i in range(NWAY_N) if i != drop]
            if int(np.all(np.stack([cols[i] == t[i] for i in keep_i], 1), 1).sum()) == 0:
                ok = False
                break
        (kept_n if ok else dropped_n).append(t)
    hold = Holdouts(hold.value_colour, tuple(kept_two), tuple(kept_n))

    # ---- the held-out rows, drawn CONDITIONALLY on their defining words
    rows = {}
    rows["seen"] = list(rng.permutation(np.arange(len(train)))[:n_seen])
    rows["seen"] = [train[int(i)] for i in rows["seen"]]

    def not_train(r):
        return r.words() not in seen_words

    # 2-way
    out2 = []
    per = max(1, n_row // max(1, len(hold.two_way)))
    for (fi, vi, fj, vj) in hold.two_way:
        g = lambda rr, a={fi: vi, fj: vj}: _with(_draw(rr, n_layouts), a)
        acc = lambda r: (not hold.carries_value(r)) and (not hold.matches_n_way(r)) \
            and not_train(r)
        got, _d = _stratified(rng, bank, hold, per, g, acc, load_bearing_frac)
        out2 += got
    rows["unseen_2way"] = [r for (r, _f) in out2]
    lb_2way = {r.words(): f for (r, f) in out2}

    # n-way
    outn = []
    pern = max(1, n_row // max(1, len(hold.n_way)))
    for t in hold.n_way:
        a = {f: v for f, v in zip(NWAY_FACTORS, t)}
        g = lambda rr, a=a: _with(_draw(rr, n_layouts), a)
        acc = lambda r: (not hold.carries_value(r)) and (not hold.matches_two_way(r)) \
            and not_train(r)
        got, _d = _stratified(rng, bank, hold, pern, g, acc, load_bearing_frac)
        outn += got
    rows["unseen_nway"] = [r for (r, _f) in outn]
    lb_n = {r.words(): f for (r, f) in outn}

    # value hold-out: exactly one colour slot carries the unseen colour
    def gv(rr):
        r = _draw(rr, n_layouts)
        slot = int(rr.integers(4)) + 1
        return _with(r, {slot: VALUE_HOLDOUT_COLOUR})

    accv = lambda r: (not hold.matches_two_way(r)) and (not hold.matches_n_way(r)) \
        and (not hold.carries_agent_slot_holdout(r)) and not_train(r)
    outv, _d = _stratified(rng, bank, hold, n_row, gv, accv, load_bearing_frac)
    rows["value_holdout"] = [r for (r, _f) in outv]
    lb_v = {r.words(): f for (r, f) in outv}

    # the SLOT value hold-out: agent colour 4, which training never puts on an agent
    def gs(rr):
        return _with(_draw(rr, n_layouts), {1: AGENT_SLOT_HOLDOUT_COLOUR})

    accs = lambda r: (not hold.matches_two_way(r)) and (not hold.matches_n_way(r)) \
        and (not hold.carries_value(r)) and not_train(r)
    outs, _d = _stratified(rng, bank, hold, n_row, gs, accs, load_bearing_frac)
    rows["value_slot_holdout"] = [r for (r, _f) in outs]
    lb_s = {r.words(): f for (r, f) in outs}

    lb_map = dict(train_lb)
    lb_map.update(lb_2way)
    lb_map.update(lb_n)
    lb_map.update(lb_v)
    lb_map.update(lb_s)

    stats = {
        "holdout_patterns_dropped_for_no_train_support": {
            "two_way": [[int(x) for x in d] for d in dropped_two],
            "n_way": [[int(x) for x in d] for d in dropped_n],
            "two_way_kept": len(kept_two), "n_way_kept": len(kept_n),
            "why": ("a pattern whose word or (n-1)-way sub-tuple has no training "
                    "support is an UNKNOWN WORD, not a composition hold-out")},
        "n_layouts": n_layouts,
        "n_train": len(train),
        "train_draws_stratum": int(d1), "train_draws_free": int(d2),
        "train_load_bearing_n": int(sum(1 for r in train if lb_map[r.words()])),
        "train_load_bearing_frac": float(np.mean([lb_map[r.words()] for r in train])),
        "row_sizes": {k: len(v) for k, v in rows.items()},
        "row_load_bearing_frac": {
            k: (float(np.mean([lb_map[r.words()] for r in v])) if v else float("nan"))
            for k, v in rows.items()},
    }
    if verbose:
        print("  pool: train %d (lb %.3f)  rows %s"
              % (len(train), stats["train_load_bearing_frac"], stats["row_sizes"]),
              flush=True)
    return Pool(bank=bank, hold=hold, train=train, rows=rows, stats=stats)


# ------------------------------------------------------------------ the proof
def disjointness_report(pool: Pool) -> dict:
    """Hold-out disjointness from train, PROVEN BY CONSTRUCTION and printed."""
    hold = pool.hold
    tr_words = {r.words() for r in pool.train}
    W = np.array([r.words() for r in pool.train], np.int64)

    out = {"n_train": len(pool.train), "duplicates_with_train": {}, "checks": {}}
    for k, rs in pool.rows.items():
        if k == "seen":
            continue
        out["duplicates_with_train"][k] = int(
            sum(1 for r in rs if r.words() in tr_words))

    # 1. VALUE
    v = hold.value_colour
    out["checks"]["value_holdout"] = {
        "colour": int(v),
        "train_requests_carrying_it": int(
            sum(int(((W[:, f] == v).sum())) for f in (1, 2, 3, 4))),
        "train_support_of_each_other_colour": {
            int(c): int(sum(int((W[:, f] == c).sum()) for f in (1, 2, 3, 4)))
            for c in TRAINED_COLOURS},
        "row_requests": len(pool.rows.get("value_holdout", [])),
    }
    out["checks"]["value_slot_holdout"] = {
        "colour": int(AGENT_SLOT_HOLDOUT_COLOUR),
        "train_agents_carrying_it": int((W[:, 1] == AGENT_SLOT_HOLDOUT_COLOUR).sum()),
        "train_support_in_other_slots": {
            "key_colour": int((W[:, 2] == AGENT_SLOT_HOLDOUT_COLOUR).sum()),
            "door_colour": int((W[:, 3] == AGENT_SLOT_HOLDOUT_COLOUR).sum()),
            "tint_colour": int((W[:, 4] == AGENT_SLOT_HOLDOUT_COLOUR).sum())},
        "train_support_of_agent_slot": {
            int(c): int((W[:, 1] == c).sum()) for c in TRAINED_AGENT_COLOURS},
        "row_requests": len(pool.rows.get("value_slot_holdout", [])),
    }

    # 2. 2-way
    two = []
    for (fi, vi, fj, vj) in hold.two_way:
        joint = int(((W[:, fi] == vi) & (W[:, fj] == vj)).sum())
        two.append({
            "pattern": [int(fi), int(vi), int(fj), int(vj)],
            "train_joint_count": joint,
            "train_marginal_i": int((W[:, fi] == vi).sum()),
            "train_marginal_j": int((W[:, fj] == vj).sum()),
        })
    out["checks"]["two_way"] = two

    # 3. n-way, plus the coverage that makes the row meaningful
    cols = [W[:, f] for f in NWAY_FACTORS]
    seen_tuples = set(map(tuple, np.stack(cols, axis=1).tolist()))
    n_layouts = pool.stats["n_layouts"]
    card = 1
    for f in NWAY_FACTORS:
        card *= _cardinality(f, n_layouts)
    nway = []
    for t in hold.n_way:
        joint = int(np.all(np.stack([c == v for c, v in zip(cols, t)], 1), 1).sum())
        subs = {}
        for drop in range(NWAY_N):
            keep = [i for i in range(NWAY_N) if i != drop]
            cnt = int(np.all(np.stack([cols[i] == t[i] for i in keep], 1), 1).sum())
            subs["drop_%s" % NWAY_FACTORS[drop]] = cnt
        nway.append({"tuple": [int(x) for x in t], "train_joint_count": joint,
                     "train_count_of_each_3way_subtuple": subs})
    out["checks"]["n_way"] = {
        "factors": list(NWAY_FACTORS), "n": NWAY_N,
        "factor_set_cardinality": int(card),
        "train_coverage_of_factor_set": float(len(seen_tuples) / card),
        "held_out": nway,
    }

    fails = []
    if out["checks"]["value_holdout"]["train_requests_carrying_it"] != 0:
        fails.append("value colour appears in train")
    vs = out["checks"]["value_slot_holdout"]
    if vs["train_agents_carrying_it"] != 0:
        fails.append("agent-slot hold-out colour appears on a training agent")
    if min(vs["train_support_in_other_slots"].values()) == 0:
        fails.append("agent-slot hold-out colour is ungrounded in the other slots")
    for d in two:
        if d["train_joint_count"] != 0:
            fails.append("2-way pattern %s appears in train" % d["pattern"])
        if d["train_marginal_i"] == 0 or d["train_marginal_j"] == 0:
            fails.append("2-way pattern %s has an ungrounded word" % d["pattern"])
    for d in nway:
        if d["train_joint_count"] != 0:
            fails.append("n-way tuple %s appears in train" % d["tuple"])
        if min(d["train_count_of_each_3way_subtuple"].values()) == 0:
            fails.append("n-way tuple %s has an ungrounded 3-way sub-tuple" % d["tuple"])
    for k, n in out["duplicates_with_train"].items():
        if n != 0:
            fails.append("row %s shares %d exact requests with train" % (k, n))
    out["pass"] = len(fails) == 0
    out["failures"] = fails
    return out
