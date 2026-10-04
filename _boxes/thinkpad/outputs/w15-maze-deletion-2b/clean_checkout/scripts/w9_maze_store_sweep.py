"""Give the store its best configuration, on VAL only.

DEV leg 2 measured the CLU at 0.0000 on the unseen-pair row and, in the same table,
the reason: the STORE-SPAN CEILING -- a true latent projected onto the deployed atom
span, decoded and executed -- was 0.0337.  With K=8 wells of 32 channels and 8 atoms
each the store spans 64 of 256 latent dimensions, and maze-route latents are not
concentrated in 64 dimensions the way Shapes3D's factorial latents are.

So before any conclusion about transfer, the store is given every configuration the
charter allows.  Note the identity that makes this a real trade-off rather than a
free parameter:

    span fraction = K * n_atoms / D = n_atoms / d_k = THE SATURATION RATIO

A store cannot span more of the latent space without moving toward saturation, where
``rank == d_k``, the projection is the identity and the CONTENT carries no
information (the wave-8 module's SATURATION TRAP, which refuses to build there).  So
every row of this sweep carries ``rank/d_k`` beside its score, and a configuration
that only works near saturation is reported as decorative, not as a win.

Selection here is on VAL.  The test split is never touched.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import equinox as eqx
import jax
import numpy as np

from w9_maze import arms as A
from w9_maze import evaluate, splits
from w9_maze.encoder import AE, encode_all, train_ae
from w9_maze.pipeline import Renderer, build_items
from w9_maze.task import BUDGET_GRID

OUT = "C:/chlu/outputs/w9-maze/runs"
BANKS = "C:/chlu/outputs/w9-maze/banks2"
POOL = "C:/chlu/outputs/w9-maze/pool"


def load_all(seed, ae_steps):
    import glob
    with open(os.path.join(BANKS, "bank_seed%d.pkl" % seed), "rb") as fh:
        bank = pickle.load(fh)["sites"]
    pool = []
    for p in sorted(glob.glob(os.path.join(POOL, "pool_shard*.pkl"))):
        with open(p, "rb") as fh:
            pool.extend(pickle.load(fh)["sites"])
    val, _test = splits.make_split(bank, seed)
    train_rows = splits.train_items(bank, val)
    Xtr, itr, _ = build_items(bank, train_rows)
    pool_rows = [(None, m, i, j) for m, st in enumerate(pool) for (i, j) in st.pairs]
    Xp, _i, _u = build_items(pool, pool_rows)
    X_ae = np.concatenate([Xtr, Xp], axis=0)
    os.makedirs(os.path.join(OUT, "ae"), exist_ok=True)
    cache = os.path.join(OUT, "ae", "ae_seed%d_steps%d.eqx" % (seed, ae_steps))
    if os.path.exists(cache):
        ae = eqx.tree_deserialise_leaves(cache,
                                         AE(jax.random.PRNGKey(seed), 75))
        print("encoder loaded:", cache, flush=True)
    else:
        print("training encoder (%d items, %d steps)" % (X_ae.shape[0], ae_steps),
              flush=True)
        ae, _h = train_ae(X_ae, jax.random.PRNGKey(seed), steps=ae_steps, verbose=True)
        eqx.tree_serialise_leaves(cache, ae)
        print("encoder cached:", cache, flush=True)
    return bank, val, train_rows, Xtr, itr, ae


def main(seed=0, ae_steps=12000):
    t00 = time.time()
    bank, val, train_rows, Xtr, itr, ae = load_all(seed, ae_steps)
    M = len(bank)
    rows_up = splits.row_requests(bank, val, "unseen_pair")
    rows_seen = train_rows
    Ztr_all = encode_all(ae, Xtr)
    Zraw = Ztr_all[itr]
    mu, sd = Zraw.mean(0), Zraw.std(0) + 1e-6
    Ztrain = ((Zraw - mu) / sd).astype(np.float32)
    render = Renderer(ae, mu, sd)
    Xup, iup, _ = build_items(bank, rows_up)
    Zup = ((encode_all(ae, Xup)[iup] - mu) / sd).astype(np.float32)
    reqs_train = [r for (r, _m, _i, _j) in train_rows]
    reqs_up = [r for (r, _m, _i, _j) in rows_up]
    grids = A.make_grids(M, 6, 6, BUDGET_GRID)

    def span_project(store, Z):
        Zp = np.zeros_like(Z)
        for ch, Mk in zip(store.well_channels, store.modes):
            U = np.linalg.qr(np.asarray(Mk, np.float64).T)[0]
            Zp[:, ch] = np.asarray((Z[:, ch] @ U) @ U.T, np.float32)
        return Zp

    print("\n%-6s %-6s %-7s %9s %9s %9s %9s %10s"
          % ("K", "atoms", "ridge", "rank/d_k", "span:seen", "span:unsn",
             "clu:seen", "clu:unseen"), flush=True)
    results = []
    for K in (8, 16, 32):
        d_k = 256 // K
        for n_atoms in (4, 8, d_k // 2, int(0.75 * d_k), d_k - 1):
            if n_atoms < 2 or n_atoms >= d_k:
                continue
            store_only = A.CluRead(Ztrain, reqs_train, grids, K=K, n_atoms=n_atoms,
                                   ridge=1.0, seed=seed)
            sp_seen = evaluate.aggregate(evaluate.answers(
                bank, rows_seen, *render.plans(span_project(store_only.store, Ztrain))))
            sp_up = evaluate.aggregate(evaluate.answers(
                bank, rows_up, *render.plans(span_project(store_only.store, Zup))))
            for ridge in (0.01, 1.0, 100.0):
                clu = store_only
                if ridge != 1.0:
                    clu = A.CluRead(Ztrain, reqs_train, grids, K=K, n_atoms=n_atoms,
                                    ridge=ridge, seed=seed)
                a_seen = evaluate.aggregate(evaluate.answers(
                    bank, rows_seen, *render.plans(clu.read(reqs_train))))
                a_up = evaluate.aggregate(evaluate.answers(
                    bank, rows_up, *render.plans(clu.read(reqs_up))))
                led = clu.ledger()
                rec = {"K": K, "n_atoms": n_atoms, "ridge": ridge,
                       "rank_over_dk": float(np.mean(led["rank_over_width"])),
                       "span_seen": sp_seen["validity"], "span_unseen": sp_up["validity"],
                       "clu_seen": a_seen["validity"], "clu_unseen": a_up["validity"],
                       "read_floats": led["read_floats"],
                       "statistic_bytes": led["statistic_bytes"]}
                results.append(rec)
                print("%-6d %-6d %-7.2f %9.3f %9.4f %9.4f %9.4f %10.4f   reads %d"
                      % (K, n_atoms, ridge, rec["rank_over_dk"], rec["span_seen"],
                         rec["span_unseen"], rec["clu_seen"], rec["clu_unseen"],
                         rec["read_floats"]), flush=True)
    with open(os.path.join(OUT, "store_sweep_seed%d.json" % seed), "w") as fh:
        json.dump(results, fh, indent=1)
    print("\n[%.0fs]  wrote store_sweep_seed%d.json" % (time.time() - t00, seed))


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    main(**{k: int(v) for k, v in kw.items()})
