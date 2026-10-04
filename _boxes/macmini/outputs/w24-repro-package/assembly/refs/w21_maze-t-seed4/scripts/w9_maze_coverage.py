"""What is the 0.13-0.19 band on the unseen-pair row actually made of?

Every arm on the scored legs -- the CLU, the lookup, the shuffled control and the
learned baselines -- lands in a narrow band on the unseen-pair row.  A band that
narrow across objects that different is a property of the ROWS, not of the objects,
and this script asks which held-out pairs are in it.

PLAN COVERAGE of a held-out pair = the fraction of its optimal route's
(cell, phase, action) triples that already appear in the union of the TRAINING routes
of the SAME maze.  Coverage 1.0 means the answer is already written somewhere in that
maze's training plans and an arm only has to retrieve and stitch; coverage near 0
means the route has to be composed.

If validity is concentrated at high coverage, the band is retrieval, and the rows
measure composition only in their low-coverage half.  That is reported either way.

VAL/TEST note: this reads the TEST leg of a scored run that has already been reported.
It computes no new arm and moves no bar; it is an analysis of a number already taken.
"""

from __future__ import annotations

import pickle
import sys

import equinox as eqx
import jax
import numpy as np

from w9_maze import arms as A
from w9_maze import evaluate, splits
from w9_maze.encoder import AE, encode_all
from w9_maze.pipeline import Renderer, build_items
from w9_maze.render import plan_classes
from w9_maze.task import BUDGET_GRID


def triples(site, i, j):
    """The (cell, phase, action) triples a route occupies."""
    cls, _ = plan_classes(site, i, j, site.route[(i, j)])
    idx = np.argwhere(cls > 0)
    return {(int(r), int(c), int(p), int(cls[r, c, p])) for r, c, p in idx}


def main(seed=0, ae_steps=12000):
    with open("C:/chlu/outputs/w9-maze/banks2/bank_seed%d.pkl" % seed, "rb") as fh:
        bank = pickle.load(fh)["sites"]
    _val, test = splits.make_split(bank, seed)
    train_rows = splits.train_items(bank, test)
    rows = splits.row_requests(bank, test, "unseen_pair")

    # plan coverage, per held-out pair, against that maze's training routes
    train_by_maze = {}
    for (_r, m, i, j) in train_rows:
        train_by_maze.setdefault(m, set()).update(triples(bank[m], i, j))
    cov = []
    for (_r, m, i, j) in rows:
        t = triples(bank[m], i, j)
        seen = train_by_maze.get(m, set())
        cov.append(len(t & seen) / max(1, len(t)))
    cov = np.asarray(cov)

    # the arms, rebuilt from the cached encoder (no training of any baseline here)
    Xtr, itr, _ = build_items(bank, train_rows)
    ae = eqx.tree_deserialise_leaves(
        "C:/chlu/outputs/w9-maze/runs/ae/ae_seed%d_steps%d.eqx" % (seed, ae_steps),
        AE(jax.random.PRNGKey(seed), 75))
    Zraw = encode_all(ae, Xtr)[itr]
    mu, sd = Zraw.mean(0), Zraw.std(0) + 1e-6
    Ztrain = ((Zraw - mu) / sd).astype(np.float32)
    render = Renderer(ae, mu, sd)
    reqs_train = [r for (r, _m, _i, _j) in train_rows]
    reqs = [r for (r, _m, _i, _j) in rows]
    grids = A.make_grids(len(bank), 6, 6, BUDGET_GRID)
    clu = A.CluRead(Ztrain, reqs_train, grids, K=8, n_atoms=31, ridge=0.01, seed=seed)
    look = A.LookupRead(Ztrain, reqs_train, budget_floats=clu.ledger()["read_floats"],
                        seed=seed)

    Xr, ir, _ = build_items(bank, rows)
    Zt = encode_all(ae, Xr)[ir]                      # the TRUE latent -> the ceiling

    out = {}
    for name, Z in (("clu", clu.read(reqs)), ("lookup_matched", look.read(reqs)),
                    ("pipeline_ceiling", ((Zt - mu) / sd).astype(np.float32))):
        recs = evaluate.answers(bank, rows, *render.plans(Z))
        out[name] = np.array([r["valid"] for r in recs], float)

    print("unseen-pair rows: %d   plan coverage mean %.3f  median %.3f"
          % (len(rows), cov.mean(), float(np.median(cov))))
    print("  coverage distribution: <0.5 %d | 0.5-0.8 %d | 0.8-0.95 %d | >=0.95 %d"
          % (int((cov < 0.5).sum()), int(((cov >= 0.5) & (cov < 0.8)).sum()),
             int(((cov >= 0.8) & (cov < 0.95)).sum()), int((cov >= 0.95).sum())))
    print()
    print("  %-18s %8s %8s %8s %8s" % ("bucket", "n", "clu", "lookup", "ceiling"))
    edges = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    for lo, hi in edges:
        m = (cov >= lo) & (cov < hi)
        if not m.any():
            continue
        print("  coverage %4.2f-%4.2f %8d %8.4f %8.4f %8.4f"
              % (lo, min(hi, 1.0), int(m.sum()), out["clu"][m].mean(),
                 out["lookup_matched"][m].mean(), out["pipeline_ceiling"][m].mean()))
    print()
    for name in ("clu", "lookup_matched", "pipeline_ceiling"):
        v = out[name]
        if v.sum() > 0:
            print("  %-18s overall %.4f   mean coverage of the pairs it gets RIGHT "
                  "%.3f   of those it gets wrong %.3f"
                  % (name, v.mean(), cov[v > 0].mean(),
                     cov[v == 0].mean() if (v == 0).any() else float("nan")))


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    main(**{k: int(v) for k, v in kw.items()})
