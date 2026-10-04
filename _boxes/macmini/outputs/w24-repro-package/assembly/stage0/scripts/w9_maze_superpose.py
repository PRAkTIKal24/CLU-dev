"""Where exactly does the composition break: the decomposition, or the superposition?

Three facts are already measured on the scored legs:

1. every (cell, phase, action) triple of a held-out route already appears in the
   TRAINING routes of the same maze -- plan coverage 1.000;
2. the union of the same-AGENT and same-TARGET training plans covers 99.7 % of a
   held-out route's triples and is complete for 99.1 % of pairs, so the factorisation
   the CLU's address book assumes -- (maze x agent) + (maze x target) -- really does
   carry the answer;
3. and the CLU still reads 0.128 against a pipeline ceiling of 0.632.

So the information is present and the factorisation is right.  This script asks what
happens between them, by running the same composition at two different places:

  PLAN-SPACE UNION      overlay the two half-plans as plan maps and execute the
                        result.  This is the decomposition done in the space where
                        the answer lives.
  LATENT SUPERPOSITION  average the two half-plans' LATENTS, decode, execute.  This
                        is the operation the CLU's read actually performs -- its read
                        is linear in the request design, so composing two addresses
                        composes their latents, not their plans.

The gap between those two numbers is the cost of composing in the latent rather than
in the plan.  Nothing here is an arm and nothing here moves a bar: it is an analysis
of scored legs already reported.
"""

from __future__ import annotations

import pickle
import sys

import equinox as eqx
import jax
import numpy as np

from w9_maze import evaluate, splits
from w9_maze.encoder import AE, encode_all
from w9_maze.pipeline import Renderer, build_items
from w9_maze.render import N_PHASE, plan_classes


def main(seed=0, ae_steps=12000):
    with open("C:/chlu/outputs/w9-maze/banks2/bank_seed%d.pkl" % seed, "rb") as fh:
        bank = pickle.load(fh)["sites"]
    _val, test = splits.make_split(bank, seed)
    train_rows = splits.train_items(bank, test)
    rows = splits.row_requests(bank, test, "unseen_pair")

    tr_pairs = {}
    for (_r, m, i, j) in train_rows:
        tr_pairs.setdefault(m, set()).add((i, j))

    # pick, for each held-out pair, one same-agent and one same-target training pair
    chosen, kept_rows = [], []
    for (rq, m, i, j) in rows:
        same_a = sorted(p for p in tr_pairs.get(m, ()) if p[0] == i)
        same_t = sorted(p for p in tr_pairs.get(m, ()) if p[1] == j)
        if not same_a or not same_t:
            continue
        chosen.append((m, i, j, same_a[0], same_t[0]))
        kept_rows.append((rq, m, i, j))
    print("held-out pairs with both halves available: %d of %d"
          % (len(kept_rows), len(rows)))

    # -- plan-space union ---------------------------------------------------
    union_cls = []
    for (m, i, j, pa, pt) in chosen:
        site = bank[m]
        ca, _ = plan_classes(site, pa[0], pa[1], site.route[pa])
        ct, _ = plan_classes(site, pt[0], pt[1], site.route[pt])
        #: the same-target half wins where the two disagree -- the second leg of the
        #: journey is the one that has to be right at the end, and a fixed rule is
        #: used so nothing is chosen per instance.
        u = np.where(ct > 0, ct, ca)
        union_cls.append(u)
    union_cls = np.stack(union_cls)
    recs_u = evaluate.answers(bank, kept_rows, union_cls,
                              [None] * len(kept_rows))
    agg_u = evaluate.aggregate(recs_u)

    # -- latent superposition ----------------------------------------------
    Xtr, itr, uniq = build_items(bank, train_rows)
    ae = eqx.tree_deserialise_leaves(
        "C:/chlu/outputs/w9-maze/runs/ae/ae_seed%d_steps%d.eqx" % (seed, ae_steps),
        AE(jax.random.PRNGKey(seed), 75))
    Ztr_u = encode_all(ae, Xtr)
    Zraw = Ztr_u[itr]
    mu, sd = Zraw.mean(0), Zraw.std(0) + 1e-6
    render = Renderer(ae, mu, sd)
    zs = {}
    for (m, i, j), n in uniq.items():
        zs[(m, i, j)] = (Ztr_u[n] - mu) / sd
    Zmix = np.stack([0.5 * (zs[(m, pa[0], pa[1])] + zs[(m, pt[0], pt[1])])
                     for (m, i, j, pa, pt) in chosen]).astype(np.float32)
    recs_m = evaluate.answers(bank, kept_rows, *render.plans(Zmix))
    agg_m = evaluate.aggregate(recs_m)

    # -- and the two halves alone, decoded, as a floor ----------------------
    Za = np.stack([zs[(m, pa[0], pa[1])] for (m, i, j, pa, pt) in chosen]).astype(np.float32)
    Zt = np.stack([zs[(m, pt[0], pt[1])] for (m, i, j, pa, pt) in chosen]).astype(np.float32)
    agg_a = evaluate.aggregate(evaluate.answers(bank, kept_rows, *render.plans(Za)))
    agg_t = evaluate.aggregate(evaluate.answers(bank, kept_rows, *render.plans(Zt)))

    print()
    print("  %-34s %8s %8s" % ("composition performed in ...", "validity", "score"))
    print("  %-34s %8.4f %8.4f" % ("PLAN space (overlay the halves)",
                                   agg_u["validity"], agg_u["score"]))
    print("  %-34s %8.4f %8.4f" % ("LATENT space (average the halves)",
                                   agg_m["validity"], agg_m["score"]))
    print("  %-34s %8.4f %8.4f" % ("neither: the same-agent half alone",
                                   agg_a["validity"], agg_a["score"]))
    print("  %-34s %8.4f %8.4f" % ("neither: the same-target half alone",
                                   agg_t["validity"], agg_t["score"]))
    return agg_u, agg_m


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    main(**{k: int(v) for k, v in kw.items()})
