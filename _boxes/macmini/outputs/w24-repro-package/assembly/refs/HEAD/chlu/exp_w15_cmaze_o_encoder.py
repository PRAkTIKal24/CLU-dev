"""The ONE shared, FROZEN encoder and decoder for every CMAZE-O arm.

Wave-9 rule 1, carried over: one shared encoder, every arm consumes its latent space,
one shared renderer, one exact judge.  The autoencoder is trained ONCE per seed, by
reconstruction on the TRAIN items only -- no request, no word, no label ever enters it --
and is then **frozen**.  Every arm (the CLU, every learned rival, the lookup floor and
every charter-C control) predicts a point in this latent space and is rendered by this
same decoder into a grid state which ``exp_w15_cmaze_o_env.judge_state`` grades.

That is what makes a margin between two arms a margin **on the read**.

The encoder and decoder are declared OUTSIDE the deletable object
-----------------------------------------------------------------
COMMON.md's deletion language: store-scoped exactness is a statement about the store.
The encoder and the decoder are shared, frozen, and are **not** part of what deletion
touches; they are named here so that they are never quietly counted as deleted.

THE LOSS IS SHAPED BY THE JUDGE, NOT BY THE PICTURE
----------------------------------------------------
The judge has **no tolerance**: it reads the decoder's output through
``decode_state``, which takes an **argmax over the 225 cells** for each object and an
**argmax over the 5 colours** at that object's cell.  So every piece of the loss is the
cross-entropy of exactly the argmax ``decode_state`` will take:

* **floor** -- independent indicators, plain binary cross-entropy;
* **tinted-floor mask** -- independent indicators, binary cross-entropy with positives
  upweighted by :data:`POS_WEIGHT` (about 20 of 225 cells are tinted);
* **door / key / agent / goal** -- exactly one cell of 225, so softmax cross-entropy
  **over the cell axis**;
* **the shared colour field** -- softmax cross-entropy over the 5 colour channels,
  **masked to the cells that carry a colour** (the tint cells and three object cells),
  which is the only place ``decode_state`` ever reads it.

This is measured, not asserted.  The autoencoder's own ceiling -- encode the TRUE
picture, decode it, judge it -- moved as follows
(``$CHLU_HUB_OUTPUTS/w15-cmaze-o-harness/artefacts/ae_loss_probe.json``):

======================================  ==============  =============  ============
loss on the object / colour channels    object cells    object colour  exact_objects
======================================  ==============  =============  ============
per-cell BCE everywhere                 --              --             0.0025
spatial softmax, colours still BCE      1.0000          ~0.36          0.0200
spatial softmax + masked colour softmax see FINDINGS.md see FINDINGS   see FINDINGS
======================================  ==============  =============  ============

The loss is a property of the SHARED encoder, identical for every arm, fixed before any
arm exists, and never tuned against a held-out number.

The CEILING this module sets is measured, not assumed: ``encode -> decode -> judge`` on
the items themselves is the best any arm on this pipeline can do, and it is printed
beside every row as the ``AE-CEILING`` arm.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.exp_w15_cmaze_o_env import (CH_AGENT, CH_COL, CH_DOOR, CH_FLOOR,
                                      CH_GOAL, CH_KEY, CH_TINT, H, N_CH,
                                      N_COLOURS, W)

#: Upweight of a positive in every channel but the floor.  Fixed before any arm exists.
POS_WEIGHT = 40.0
LATENT_D = 128


class Encoder(eqx.Module):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, key, in_ch, d):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.c1 = eqx.nn.Conv2d(in_ch, 48, 3, stride=1, padding=1, key=k1)
        self.c2 = eqx.nn.Conv2d(48, 72, 3, stride=2, padding=1, key=k2)
        self.c3 = eqx.nn.Conv2d(72, 96, 3, stride=2, padding=1, key=k3)
        self.fc = eqx.nn.Linear(96 * 4 * 4, d, key=k4)

    def __call__(self, x):                       # x: (C, 15, 15)
        h = jax.nn.gelu(self.c1(x))
        h = jax.nn.gelu(self.c2(h))
        h = jax.nn.gelu(self.c3(h))
        return self.fc(h.reshape(-1))


class Decoder(eqx.Module):
    """Latent -> logits.  One projection to the full 15x15 grid, then local cleanup.

    The same shape-of-decoder argument as ``w9_maze.encoder.Decoder``: the grid is
    15x15 and the job is to render a SPECIFIC picture from a D-vector, so the
    upsampling is one projection and the convolutions clean it up locally.
    """

    fc: eqx.nn.Linear
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    out: eqx.nn.Conv2d

    def __init__(self, key, out_ch, d):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.fc = eqx.nn.Linear(d, 64 * H * W, key=k1)
        self.c1 = eqx.nn.Conv2d(64, 96, 3, stride=1, padding=1, key=k2)
        self.c2 = eqx.nn.Conv2d(96, 96, 3, stride=1, padding=1, key=k3)
        self.out = eqx.nn.Conv2d(96, out_ch, 1, key=k4)

    def __call__(self, z):
        h = jax.nn.gelu(self.fc(z)).reshape((64, H, W))
        h = jax.nn.gelu(self.c1(h))
        h = jax.nn.gelu(self.c2(h))
        return self.out(h)                        # logits (C, 15, 15)


class AE(eqx.Module):
    enc: Encoder
    dec: Decoder

    def __init__(self, key, ch=N_CH, d=LATENT_D):
        ke, kd = jax.random.split(key)
        self.enc = Encoder(ke, ch, d)
        self.dec = Decoder(kd, ch, d)


#: the channels read as ONE CELL OF 225 -- a categorical position
OBJ_CH = (CH_DOOR, CH_KEY, CH_AGENT, CH_GOAL)


def recon_loss(model, xb):
    """xb: (B, 15, 15, C) float32 targets in {0, 1}."""
    obj_idx = jnp.asarray(OBJ_CH)

    def one(x):
        logits = model.dec(model.enc(jnp.transpose(x, (2, 0, 1))))
        lg = jnp.transpose(logits, (1, 2, 0))              # (15, 15, C)
        # floor -- plain BCE; it is ~44 % positive and must not be upweighted
        floor = optax.sigmoid_binary_cross_entropy(
            lg[:, :, CH_FLOOR], x[:, :, CH_FLOOR]).mean()
        # tinted-floor mask -- sparse, so positives are upweighted
        t_ce = optax.sigmoid_binary_cross_entropy(lg[:, :, CH_TINT], x[:, :, CH_TINT])
        t_w = jnp.where(x[:, :, CH_TINT] > 0.5, POS_WEIGHT, 1.0)
        tint = (t_ce * t_w).sum() / t_w.sum()
        # the four objects -- one cell of 225 each
        o_lg = lg[:, :, obj_idx].reshape((H * W, len(OBJ_CH))).T
        o_t = x[:, :, obj_idx].reshape((H * W, len(OBJ_CH))).T
        pos = optax.softmax_cross_entropy(o_lg, o_t).mean()
        # the shared colour field -- softmax over colours, only where a colour lives
        c_lg = lg[:, :, CH_COL:CH_COL + N_COLOURS]
        c_t = x[:, :, CH_COL:CH_COL + N_COLOURS]
        c_mask = c_t.sum(axis=-1)
        c_ce = optax.softmax_cross_entropy(c_lg, c_t)
        colour = (c_ce * c_mask).sum() / jnp.maximum(c_mask.sum(), 1.0)
        return floor + tint + pos + colour

    return jnp.mean(jax.vmap(one)(xb))


@eqx.filter_jit
def _step(model, opt_state, xb, optimiser):
    loss, grads = eqx.filter_value_and_grad(recon_loss)(model, xb)
    updates, opt_state = optimiser.update(grads, opt_state,
                                          eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), opt_state, loss


def train_ae(X_train, key, steps=4000, batch=64, lr=1e-3, log_every=500,
             verbose=True, d=LATENT_D):
    """Train the shared encoder by reconstruction on the TRAIN items only.

    ``X_train`` is ``(N, 15, 15, C)`` uint8; it is cast per batch so the pool stays
    at one byte per channel in memory.
    """
    model = AE(key, int(X_train.shape[-1]), d)
    sched = optax.warmup_cosine_decay_schedule(1e-4, lr, int(0.05 * steps), steps, 1e-5)
    optimiser = optax.adam(sched)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(0)
    n = int(X_train.shape[0])
    hist = []
    for t in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        xb = jnp.asarray(np.asarray(X_train[idx], np.float32))
        model, opt_state, loss = _step(model, opt_state, xb, optimiser)
        if (t + 1) % log_every == 0 or t == 0:
            hist.append((t + 1, float(loss)))
            if verbose:
                print("    ae step %5d  loss %.5f" % (t + 1, float(loss)), flush=True)
    return model, hist


@eqx.filter_jit
def _encode_batch(model, xb):
    return jax.vmap(lambda x: model.enc(jnp.transpose(x, (2, 0, 1))))(xb)


@eqx.filter_jit
def _decode_batch(model, zb):
    return jax.vmap(model.dec)(zb)


def encode_all(model, X, batch=256):
    out = []
    for s in range(0, int(X.shape[0]), batch):
        xb = jnp.asarray(np.asarray(X[s:s + batch], np.float32))
        out.append(np.asarray(_encode_batch(model, xb)))
    return np.concatenate(out, axis=0).astype(np.float32)


def decode_all(model, Z, batch=256):
    """Latents -> ``(N, 15, 15, C)`` logits, the object the state decoder reads."""
    out = []
    for s in range(0, int(Z.shape[0]), batch):
        lg = np.asarray(_decode_batch(model, jnp.asarray(Z[s:s + batch])))
        out.append(np.transpose(lg, (0, 2, 3, 1)))
    return np.concatenate(out, axis=0).astype(np.float32)


def ae_param_count(model) -> dict:
    enc = int(sum(v.size for v in jax.tree.leaves(eqx.filter(model.enc, eqx.is_array))))
    dec = int(sum(v.size for v in jax.tree.leaves(eqx.filter(model.dec, eqx.is_array))))
    return {"encoder_params": enc, "decoder_params": dec, "total": enc + dec,
            "declared": ("SHARED and FROZEN; identical for every arm; declared OUTSIDE "
                         "the deletable object and never counted as deleted")}
