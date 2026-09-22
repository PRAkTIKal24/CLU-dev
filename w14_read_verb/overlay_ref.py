"""THE GO/NO-GO: can a FIXED plan-space overlay express a CMAZE-2b route at all?

Wave 9 measured, on CMAZE-2, that overlaying a held-out route's same-agent and
same-target TRAINING plans scores 1.0000 -- optimal on every pair, all three seeds --
while averaging the identical halves in latent space scores 0.13.  It shipped that
result with its own caveat, and the caveat is this wave's precondition:

  "that the overlay is perfect is itself a statement that CMAZE-2's routes decompose
   additively, which is the same property that leaves BLIND-SEQUENTIAL at 0.9037 here.
   In an environment whose routes need two or more keys the overlay would not be
   perfect"
  -- outputs/_thinkpad/w9/w9-maze/LOG.md:339-341

CMAZE-2b is that environment: 100 % of its routes pick up >= 2 keys.  So the overlay's
perfection is not inherited, it is re-measured here, on the repaired layer, BEFORE any
arm exists and with no scored budget spent.

⛔ This is a REFERENCE, not an arm.  It keeps whole written training items and is not
matched to any byte count.  Its only job is to answer the task file's stopping rule --
"a fixed merge cannot express the task without per-instance tuning => stop" -- and to
re-measure wave 9's two premises (plan coverage, factorisation) on the new layer.

    python -m w14_read_verb.overlay_ref <bank.pkl> <seed> <out.json> [budget_grid]
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np

from w9_maze import splits
from w9_maze.render import N_PHASE, execute, plan_classes, score_route
from w9_maze.task import Request
from w14_read_verb.merge import MERGE_RULES, fold_overlay, overlay


def _plan(site, i, j):
    cls, coll = plan_classes(site, i, j, site.route[(i, j)])
    return cls, coll


def _slots(cls):
    """The set of (cell, phase, action) triples a plan map asserts."""
    nz = np.argwhere(np.asarray(cls) != 0)
    return {(int(r), int(c), int(p), int(cls[r, c, p])) for r, c, p in nz}


def halves(site, train_pairs, i, j):
    """The same-agent and same-target TRAINING plan maps, folded in sorted order."""
    same_a = [p for p in train_pairs if p[0] == i and p != (i, j)]
    same_t = [p for p in train_pairs if p[1] == j and p != (i, j)]
    pa = fold_overlay([_plan(site, *p)[0] for p in sorted(same_a)]) if same_a else None
    pt = fold_overlay([_plan(site, *p)[0] for p in sorted(same_t)]) if same_t else None
    z = np.zeros(site.grid.shape + (N_PHASE,), np.int8)
    return (pa if pa is not None else z, pt if pt is not None else z,
            len(same_a), len(same_t))


def coverage(site, train_pairs, i, j):
    """Wave 9's premises (i) and (ii), re-measured on THIS layer.

    (i) plan coverage: the fraction of the held-out route's own (cell, phase, action)
        triples that already appear in the union of the SAME MAZE's training plans.
    (ii) the factorisation: the same fraction against the same-agent plans, the
        same-target plans, and their union.
    """
    tgt = _slots(_plan(site, i, j)[0])
    if not tgt:
        return None
    allp = {q for q in train_pairs if q != (i, j)}
    u_all = set().union(*[_slots(_plan(site, *q)[0]) for q in sorted(allp)]) if allp else set()
    sa = [q for q in allp if q[0] == i]
    st = [q for q in allp if q[1] == j]
    u_a = set().union(*[_slots(_plan(site, *q)[0]) for q in sorted(sa)]) if sa else set()
    u_t = set().union(*[_slots(_plan(site, *q)[0]) for q in sorted(st)]) if st else set()
    n = float(len(tgt))
    return {"plan_coverage": len(tgt & u_all) / n,
            "same_agent": len(tgt & u_a) / n,
            "same_target": len(tgt & u_t) / n,
            "union": len(tgt & (u_a | u_t)) / n,
            "n_slots": int(n)}


def run(bank, seed, budgets, which="val"):
    t0 = time.time()
    val, test = splits.make_split(bank, seed)
    split = val if which == "val" else test
    rows = {r: splits.row_requests(bank, split, r, budgets) if r != "budget_interp"
            else splits.row_requests(bank, split, r)
            for r in ("unseen_pair", "nway")}
    out = {"seed": int(seed), "which": which, "budgets": list(budgets),
           "n_mazes": len(bank)}
    for row, rq in rows.items():
        cov, sc = [], {k: [] for k in MERGE_RULES}
        ok = {k: 0 for k in MERGE_RULES}
        rnd = []
        rng = np.random.default_rng(seed + 4242)
        for (req, m, i, j) in rq:
            site = bank[m]
            tp = list(split.train_pairs[m])
            c = coverage(site, tp, i, j)
            if c is not None:
                cov.append(c)
            pa, pt, na, nt = halves(site, tp, i, j)
            for rule in MERGE_RULES:
                cls = overlay(pa, pt, rule)
                route = execute(cls, site, i, j, req.b)
                v, s, _r = score_route(site, i, j, req.b, route)
                ok[rule] += bool(v)
                sc[rule].append(s)
            #: the control wave 9 ran: the target half kept, the agent half replaced by
            #: a RANDOM training plan of the same maze.
            if tp:
                q = tp[int(rng.integers(0, len(tp)))]
                cls = overlay(_plan(site, *q)[0], pt, "target_wins")
                route = execute(cls, site, i, j, req.b)
                v, s, _r = score_route(site, i, j, req.b, route)
                rnd.append((bool(v), s))
        n = max(1, len(rq))
        out[row] = {
            "n": len(rq),
            "validity": {k: ok[k] / n for k in MERGE_RULES},
            "score": {k: float(np.mean(sc[k])) if sc[k] else 0.0 for k in MERGE_RULES},
            "random_agent_half": {
                "validity": float(np.mean([a for a, _ in rnd])) if rnd else 0.0,
                "score": float(np.mean([s for _, s in rnd])) if rnd else 0.0},
            "coverage": {k: float(np.mean([c[k] for c in cov])) if cov else 0.0
                         for k in ("plan_coverage", "same_agent", "same_target",
                                   "union")},
            "coverage_complete_fraction": (
                float(np.mean([c["union"] >= 0.999 for c in cov])) if cov else 0.0),
            "coverage_ge_95_fraction": (
                float(np.mean([c["plan_coverage"] >= 0.95 for c in cov])) if cov else 0.0),
            "n_covered": len(cov),
        }
    out["seconds"] = time.time() - t0
    return out


def main(bank_path, seed, out_path, grid=None):
    with open(bank_path, "rb") as fh:
        bank = pickle.load(fh)["sites"]
    budgets = (tuple(int(x) for x in grid.split(",")) if grid
               else __import__("w9_maze.task", fromlist=["BUDGET_GRID"]).BUDGET_GRID)
    res = {"bank": bank_path, "grid": list(budgets)}
    for which in ("val",):
        res[which] = run(bank, int(seed), budgets, which)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:4000], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else None)
