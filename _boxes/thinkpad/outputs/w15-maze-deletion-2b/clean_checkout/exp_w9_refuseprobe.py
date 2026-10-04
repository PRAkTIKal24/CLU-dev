"""Is the refusal decision even present in the shared observation?

The supervised refuser is the only comparator for DISPLAY rows, and it separates
possible from impossible by very little.  There are two ways that can happen:

  (a) the plumbing is wrong -- the fact is in the observation and the refuser
      cannot get at it;
  (b) the problem is hard -- deciding "is this maze solvable within B steps" is
      close to solving it, so the fact is not readably present at t=0 at all.

This script distinguishes them.  It takes the SAME shared observation every arm
consumes, at t=0, and fits the strongest cheap classifier there is any excuse for:
ridge regression, fit on one half, scored by AUC on the other.  Two contrasts:

  possible vs impossible-by-BUDGET    -- differs only in the budget number
  possible vs impossible-by-LEGALITY  -- differs in the tint layout, so the maze
                                         image itself is different

It also reports the AUC of the budget feature alone, which is the honest floor: if
that number were high, the refuser would be reading a giveaway rather than the maze.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from exp_w9_harness import FEAT_DIM, SCALES, TIERS
from w9_harness import data, dynamics, encoder, features, render


def auc(scores, y):
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def t0_obs(insts, feats):
    return np.stack([features.obs(i, S, dynamics.initial_state(i))
                     for i, S in zip(insts, feats)])


def ridge_auc(Xp, Xn, ridge=1.0, seed=0):
    """Fit on the first half of each class, score AUC on the second."""
    rng = np.random.default_rng(seed)
    Xp, Xn = Xp[rng.permutation(len(Xp))], Xn[rng.permutation(len(Xn))]
    hp, hn = len(Xp) // 2, len(Xn) // 2
    Xtr = np.concatenate([Xp[:hp], Xn[:hn]])
    ytr = np.concatenate([np.ones(hp), np.zeros(hn)])
    Xte = np.concatenate([Xp[hp:], Xn[hn:]])
    yte = np.concatenate([np.ones(len(Xp) - hp), np.zeros(len(Xn) - hn)]).astype(bool)
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    A = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return auc(Xte @ w, yte), len(Xtr), len(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scale", default="std")
    ap.add_argument("--tiers", default="cmaze1,cmaze1-easy")
    ap.add_argument("--out", default="/c/chlu/outputs/w9-harness")
    a = ap.parse_args()
    cfg = SCALES[a.scale]
    allout = {}
    for tier in a.tiers.split(","):
        gen = TIERS[tier]
        for seed in [int(x) for x in a.seeds.split(",")]:
            base = 1_000_000 + 5_000_000 * seed
            tr, _ = data.make_instances(cfg["n_train"], base, **gen)
            ev, _ = data.make_instances(cfg["n_eval"], base + 2_000_000, **gen)
            imp, fam = data.make_impossible(cfg["n_imp_eval"], base + 4_500_000,
                                            rng_seed=100 + seed, **gen)
            enc, erep = encoder.train(render.render_batch(tr), feat_dim=FEAT_DIM,
                                      steps=cfg["enc_steps"], seed=seed,
                                      restarts=4, accept_bce=0.02, verbose=False)
            Fp = encoder.features(enc, render.render_batch(ev))
            Fi = encoder.features(enc, render.render_batch(imp))
            Xp = t0_obs(ev, Fp)
            Xi = t0_obs(imp, Fi)
            fam = np.array(fam)
            res = {}
            for name, sel in (("all impossible", np.ones(len(imp), bool)),
                              ("impossible-by-budget", fam == "budget"),
                              ("impossible-by-legality", fam == "legality")):
                if sel.sum() < 10:
                    continue
                s, ntr, nte = ridge_auc(Xi[sel], Xp, seed=seed)
                res[name] = {"ridge_auc": s, "n_train": ntr, "n_test": nte}
            # the budget scalar on its own -- the giveaway floor
            bi = np.array([int(i.budget) for i in imp], float)
            bp = np.array([int(i.budget) for i in ev], float)
            res["budget scalar alone"] = {
                "auc": auc(np.concatenate([bi, bp]),
                           np.concatenate([np.ones(len(bi)), np.zeros(len(bp))]).astype(bool))}
            allout["%s/seed%d" % (tier, seed)] = res
            print("%s seed %d  (encoder BCE %.5f -- the fixed, restart-selected "
                  "encoder, so a low AUC here is about the PROBLEM, not a bad "
                  "encoder):" % (tier, seed, erep["final_bce"]))
            for k, v in res.items():
                print("   %-26s %s" % (k, {kk: round(vv, 4) if isinstance(vv, float)
                                           else vv for kk, vv in v.items()}))
    path = os.path.join(a.out, "refusal_separability.json")
    with open(path, "w") as f:
        json.dump(allout, f, indent=1)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
