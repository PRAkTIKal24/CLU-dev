"""The split, and the four RESULT rows it defines.

Every maze holds two DIRECTION BLOCKS: agents on side 0 to targets on side 1, and
agents on side 1 to targets on side 0.  One block supplies this maze's VAL holdouts,
the other its TEST holdouts; which is which alternates with the maze index, so neither
held-out set is a single direction.

Inside a block, the holdouts are chosen so the rows mean what the task file says:

ROW 2  ``unseen_pair``  one pair (a, t) is withheld while BOTH of its words remain
       in training for this maze through other pairs.  It is a new legal route, and
       nothing about it is new except the combination.
ROW 3  ``nway``  two pairs (a, t*) and (a', t*) sharing one target are withheld
       together, while (a'', t*) stays in training, so the target word is still
       grounded in this maze.  The row scores the CONJUNCTION: two agents, one target,
       both routes valid.  Its per-route score is reported beside it.
ROW 1  ``seen``  the training pairs themselves, at the trained budget words.
ROW 4  ``budget_interp``  the row-2 and row-3 requests re-asked at a budget word
       strictly between two trained ones (38 and 53), which no arm has ever seen.

A pair is only ever a training item at a budget word its optimum fits: an impossible
example is never written.  That is what the refusal DISPLAY row rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from w9_maze.task import BUDGET_GRID, BUDGET_INTERP, Request


@dataclass
class Split:
    train_pairs: dict = field(default_factory=dict)      # m -> [(i, j), ...]
    unseen_pair: dict = field(default_factory=dict)      # m -> [(i, j)]      val/test
    nway: dict = field(default_factory=dict)             # m -> [(i, j), (i2, j)]
    #: every pair of ONE target word, withheld entirely: the (maze, target)
    #: combination is ungrounded, exactly as the walled target's is, but the target is
    #: REACHABLE.  It is the control that stops a refusal on the walled target from
    #: being read as a refusal when it is only novelty detection.
    mt_unseen: dict = field(default_factory=dict)
    per_side: int = 3

    def all_train(self):
        return [(m, p) for m, ps in self.train_pairs.items() for p in ps]


def _block_of(i, j, per_side):
    return 0 if i < per_side else 1


def make_split(bank, seed: int, per_side: int = 3):
    """VAL and TEST splits over the SAME training set.

    The training set is shared: holding a pair out of one held-out set does not put it
    back into training for the other.  Both held-out sets are therefore measured
    against one fitted object, which is what makes 'selection on val only' meaningful.
    """
    rng = np.random.default_rng(1000 + seed)
    val, test = Split(per_side=per_side), Split(per_side=per_side)
    train = {}
    for m, site in enumerate(bank):
        pairs = list(site.pairs)
        blocks = {0: [p for p in pairs if _block_of(p[0], p[1], per_side) == 0],
                  1: [p for p in pairs if _block_of(p[0], p[1], per_side) == 1]}
        which = {0: (val if m % 2 == 0 else test), 1: (test if m % 2 == 0 else val)}
        held = set()
        # -- the matched novelty control: one whole target word, withheld, reachable
        by_t_all = {}
        for (i, j) in pairs:
            by_t_all.setdefault(j, []).append(i)
        cand_u = [t for t, ags in by_t_all.items() if len(ags) >= 2
                  and len({q[1] for q in pairs
                           if _block_of(q[0], q[1], per_side)
                           == _block_of(ags[0], t, per_side)}) >= 3]
        if cand_u:
            tu = int(rng.choice(sorted(cand_u)))
            mt = [(int(i), tu) for i in sorted(by_t_all[tu])]
            val.mt_unseen[m] = mt
            test.mt_unseen[m] = mt
            held |= set(mt)
            blocks = {b: [p for p in v if p not in held] for b, v in blocks.items()}
        for b in (0, 1):
            ps = blocks[b]
            dest = which[b]
            by_t = {}
            for (i, j) in ps:
                by_t.setdefault(j, []).append(i)
            # -- row 3: a target reached by >= 3 agents; withhold two of them
            cand = [t for t, ags in by_t.items() if len(ags) >= 3]
            if cand:
                t3 = int(rng.choice(cand))
                ags = sorted(by_t[t3])
                rng.shuffle(ags)
                a1, a2 = int(ags[0]), int(ags[1])
                dest.nway[m] = [(a1, t3), (a2, t3)]
                held |= {(a1, t3), (a2, t3)}
            else:
                t3 = None
            # -- row 2: one more pair whose agent word and target word both survive
            for (i, j) in rng.permutation(np.array(ps, dtype=int)).tolist():
                i, j = int(i), int(j)
                if (i, j) in held or j == t3:
                    continue
                rest = [q for q in ps if q not in held and q != (i, j)]
                if any(q[0] == i for q in rest) and any(q[1] == j for q in rest):
                    dest.unseen_pair[m] = [(i, j)]
                    held.add((i, j))
                    break
        train[m] = [p for p in pairs if p not in held]
    val.train_pairs = train
    test.train_pairs = dict(train)
    return val, test


# ------------------------------------------------------------------ requests
def requests_for(bank, pairs_by_maze, budgets=BUDGET_GRID, feasible_only=True):
    """Expand ``m -> [(i, j)]`` into requests at the given budget words."""
    out = []
    for m, ps in pairs_by_maze.items():
        for (i, j) in ps:
            for b in budgets:
                if feasible_only and bank[m].opt[(i, j)] > b:
                    continue
                out.append((Request(m, i, j, int(b)), m, i, j))
    return out


def train_items(bank, split, budgets=BUDGET_GRID):
    """The written items: one request per (maze, pair, feasible budget word).

    The ITEM (the picture with the route drawn on it) does not depend on the budget;
    the REQUEST does, so a pair that fits two budget words is written twice, under two
    different addresses.  That is what makes the budget a word of the address book
    rather than a decoration.
    """
    return requests_for(bank, split.train_pairs, budgets, feasible_only=True)


def walled_requests(bank, budgets=BUDGET_GRID):
    """Requests for a target that is SEALED BEHIND WALLS.  Impossible, never written."""
    out = []
    for m, site in enumerate(bank):
        for (i, j) in site.walled_pairs:
            for b in budgets:
                out.append((Request(m, i, j, int(b)), m, i, j))
    return out


def row_requests(bank, split, row, budgets=BUDGET_GRID):
    if row == "mt_unseen":
        return requests_for(bank, split.mt_unseen, budgets, True)
    if row == "seen":
        return requests_for(bank, split.train_pairs, budgets, True)
    if row == "unseen_pair":
        return requests_for(bank, split.unseen_pair, budgets, True)
    if row == "nway":
        return requests_for(bank, split.nway, budgets, True)
    if row == "budget_interp":
        held = {m: (split.unseen_pair.get(m, []) + split.nway.get(m, []))
                for m in split.train_pairs}
        return requests_for(bank, held, BUDGET_INTERP, True)
    raise KeyError(row)


def infeasible_requests(bank, split, budgets=BUDGET_GRID):
    """The refusal material: held-out pairs whose optimum does NOT fit the budget.

    Never written, never seen, and the object is asked for them anyway.
    """
    held = {m: (split.unseen_pair.get(m, []) + split.nway.get(m, []))
            for m in split.train_pairs}
    out = []
    for m, ps in held.items():
        for (i, j) in ps:
            for b in budgets:
                if bank[m].opt[(i, j)] > b:
                    out.append((Request(m, i, j, int(b)), m, i, j))
    return out
