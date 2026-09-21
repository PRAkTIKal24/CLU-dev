"""The ONE shared encoder, trained by reconstruction on maze images.

Wave-9 rule 1: one shared encoder, every arm consumes its features, one shared action
head, one exact judge.  This module is that encoder and its decoder.  It is trained
ONCE per seed, on the TRAIN items only, by reconstruction -- no label, no request, no
route supervision beyond the picture itself -- and is then frozen.  Every arm, the CLU
and every learned baseline alike, predicts a point in this encoder's latent space and
is rendered by this decoder into a plan map that the parameterless executor runs.

The reconstruction loss has two parts because the image has two kinds of channel:
* the scene (floor, tints, doors, keys, the agent cell, the target cell) is a set of
  independent indicators -- binary cross-entropy;
* the plan is a categorical map: exactly one of 6 classes per (cell, phase) -- softmax
  cross-entropy over the class axis.

Slots that carry an action are ~2 % of the plan map (about 42 of 2,250), so the
positive slots are upweighted by :data:`POS_WEIGHT`.  That weight is a property of the
SHARED encoder, identical for every arm, fixed before any arm exists, and never tuned
against a held-out number.

The ceiling this module sets is measured, not assumed: ``encode -> decode -> execute ->
exact judge`` on the training items is the best any arm on this pipeline can do, and it
is printed beside every row.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from w9_maze.render import N_CLASS, N_MAZE_CH, N_PHASE, N_PLAN_CH

#: Upweight of the plan slots that carry an action.  Fixed before any arm is built.
POS_WEIGHT = 20.0
LATENT_D = 256


class Encoder(eqx.Module):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, key, in_ch, d):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.c1 = eqx.nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, key=k1)
        self.c2 = eqx.nn.Conv2d(64, 96, 3, stride=2, padding=1, key=k2)
        self.c3 = eqx.nn.Conv2d(96, 128, 3, stride=2, padding=1, key=k3)
        self.fc = eqx.nn.Linear(128 * 4 * 4, d, key=k4)

    def __call__(self, x):                       # x: (C, 15, 15)
        h = jax.nn.gelu(self.c1(x))
        h = jax.nn.gelu(self.c2(h))              # (96, 8, 8)
        h = jax.nn.gelu(self.c3(h))              # (128, 4, 4)
        return self.fc(h.reshape(-1))


class Decoder(eqx.Module):
    """Latent -> logits.

    A full transposed-convolution stack at 128 channels costs ~78 MMAC per sample and
    measured 0.27 s per 64-item step on this box, which would have put one encoder at
    over an hour.  The spatial grid is 15x15 and the decoder's job is to render a
    SPECIFIC picture from a 256-vector, so the upsampling is done in one projection
    and the convolutions are left to clean it up locally.  Same job, ~3x the speed.
    """

    fc: eqx.nn.Linear
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    out: eqx.nn.Conv2d

    def __init__(self, key, out_ch, d):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.fc = eqx.nn.Linear(d, 64 * 15 * 15, key=k1)
        self.c1 = eqx.nn.Conv2d(64, 96, 3, stride=1, padding=1, key=k2)
        self.c2 = eqx.nn.Conv2d(96, 96, 3, stride=1, padding=1, key=k3)
        self.out = eqx.nn.Conv2d(96, out_ch, 1, key=k4)

    def __call__(self, z):
        h = jax.nn.gelu(self.fc(z)).reshape((64, 15, 15))
        h = jax.nn.gelu(self.c1(h))
        h = jax.nn.gelu(self.c2(h))
        return self.out(h)                        # logits (C, 15, 15)


class AE(eqx.Module):
    enc: Encoder
    dec: Decoder

    def __init__(self, key, ch, d=LATENT_D):
        ke, kd = jax.random.split(key)
        self.enc = Encoder(ke, ch, d)
        self.dec = Decoder(kd, ch, d)


def _split_logits(logits):
    """(C, 15, 15) logits -> (scene (15,15,15), plan (15,15,PHASE,CLASS))."""
    x = jnp.transpose(logits, (1, 2, 0))
    return x[:, :, :N_MAZE_CH], x[:, :, N_MAZE_CH:].reshape(
        (15, 15, N_PHASE, N_CLASS))


def recon_loss(model, xb):
    """xb: (B, 15, 15, C) float32 targets."""
    def one(x):
        logits = model.dec(model.enc(jnp.transpose(x, (2, 0, 1))))
        s_log, p_log = _split_logits(logits)
        s_tgt = x[:, :, :N_MAZE_CH]
        scene = optax.sigmoid_binary_cross_entropy(s_log, s_tgt).mean()
        p_tgt = x[:, :, N_MAZE_CH:].reshape((15, 15, N_PHASE, N_CLASS))
        cls = jnp.argmax(p_tgt, axis=-1)
        ce = optax.softmax_cross_entropy(p_log, p_tgt)
        w = jnp.where(cls > 0, POS_WEIGHT, 1.0)
        plan = (ce * w).sum() / w.sum()
        return scene + plan
    return jnp.mean(jax.vmap(one)(xb))


@eqx.filter_jit
def _step(model, opt_state, xb, optimiser):
    loss, grads = eqx.filter_value_and_grad(recon_loss)(model, xb)
    updates, opt_state = optimiser.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), opt_state, loss


def train_ae(X_train, key, steps=6000, batch=64, lr=1e-3, log_every=500, verbose=True):
    """Train the shared encoder by reconstruction on the TRAIN items only."""
    ch = int(X_train.shape[-1])
    model = AE(key, ch)
    sched = optax.warmup_cosine_decay_schedule(1e-4, lr, int(0.05 * steps), steps, 1e-5)
    optimiser = optax.adam(sched)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(0)
    n = int(X_train.shape[0])
    hist = []
    for t in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        xb = jnp.asarray(X_train[idx])
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


def encode_all(model, X, batch=128):
    out = []
    for s in range(0, int(X.shape[0]), batch):
        out.append(np.asarray(_encode_batch(model, jnp.asarray(X[s:s + batch]))))
    return np.concatenate(out, axis=0).astype(np.float32)


def decode_plans(model, Z, batch=128):
    """Latents -> (N, 15, 15, N_PHASE) argmax plan class maps, plus scene logits."""
    plans, scenes = [], []
    for s in range(0, int(Z.shape[0]), batch):
        lg = np.asarray(_decode_batch(model, jnp.asarray(Z[s:s + batch])))
        x = np.transpose(lg, (0, 2, 3, 1))
        scenes.append(x[:, :, :, :N_MAZE_CH])
        p = x[:, :, :, N_MAZE_CH:].reshape((-1, 15, 15, N_PHASE, N_CLASS))
        plans.append(np.argmax(p, axis=-1).astype(np.int8))
    return np.concatenate(plans, 0), np.concatenate(scenes, 0)
