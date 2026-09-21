"""How much encoder does the pipeline need?

The shared encoder is a CEILING on every arm: whatever an arm reads is rendered by
this decoder and executed, so if ``encode -> decode -> execute -> exact judge`` is not
near 1.0 on the training items, no row below it means anything.  This script trains
one encoder and measures that ceiling at checkpoints, so the training length is chosen
against the ceiling and not against a guess.

It is a DEV instrument: it touches the training items only.
"""

from __future__ import annotations

import pickle
import sys
import time

import equinox as eqx
import jax
import numpy as np

from w9_maze import evaluate, splits
from w9_maze.encoder import AE, decode_plans, encode_all
from w9_maze.encoder import _step as ae_step
from w9_maze.pipeline import build_items
import optax


def main(seed=0, n_sites=96, total=16000, ckpt=2000, latent=None):
    with open("C:/chlu/outputs/w9-maze/banks/bank_seed%d.pkl" % seed, "rb") as fh:
        bank = pickle.load(fh)["sites"][:n_sites]
    val, _test = splits.make_split(bank, seed)
    train_rows = splits.train_items(bank, val)
    X, idx, _ = build_items(bank, train_rows)
    print("sites %d  train requests %d  unique items %d"
          % (len(bank), len(train_rows), X.shape[0]), flush=True)

    model = AE(jax.random.PRNGKey(seed), int(X.shape[-1]))
    print("AE params %d"
          % sum(v.size for v in jax.tree.leaves(eqx.filter(model, eqx.is_array))))
    sched = optax.warmup_cosine_decay_schedule(1e-4, 1e-3, int(0.05 * total), total,
                                               1e-5)
    opt = optax.adam(sched)
    state = opt.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for t in range(total):
        b = rng.integers(0, X.shape[0], size=64)
        model, state, loss = ae_step(model, state, jax.numpy.asarray(X[b]), opt)
        if (t + 1) % ckpt == 0:
            Z = encode_all(model, X)
            plans, scenes = decode_plans(model, Z)
            recs = evaluate.answers(bank, train_rows, plans[idx], scenes[idx])
            agg = evaluate.aggregate(recs)
            print("  step %6d  loss %.5f  CEILING validity %.4f score %.4f  [%.0fs]"
                  % (t + 1, float(loss), agg["validity"], agg["score"],
                     time.time() - t0), flush=True)


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    main(**{k: int(v) for k, v in kw.items()})
