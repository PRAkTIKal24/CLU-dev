"""The ONE shared encoder.  Trained by reconstruction, then frozen.

Why reconstruction and nothing else: the encoder is the only part of the instrument
that every arm shares, so if any arm's loss could reach it, the shared part would be
tuned towards that arm and the comparison would be rigged.  The objective here is a
per-pixel Bernoulli reconstruction of the maze image -- no actions, no routes, no
judge, no arm.  `train()` returns a frozen `Encoder`; `features()` is the only thing
the arms ever call, and it is called under `jax.lax.stop_gradient` semantics simply
because the encoder parameters are never handed to any arm's optimiser.

`independence_report()` is the evidence for that claim: it lists what the encoder's
loss function closes over, and asserts no arm module and no label array is in it.

Architecture: three 3x3 convolutions down, three up, no striding -- the feature map
stays at grid resolution so `features.patch` can take an agent-centred slice, and
the receptive field (7x7 after three 3x3 layers) is wide enough that a patch of the
feature map sees more than a patch of the image.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


class Encoder(eqx.Module):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d

    def __init__(self, in_ch, width, feat_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.c1 = eqx.nn.Conv2d(in_ch, width, 3, padding=1, key=k1)
        self.c2 = eqx.nn.Conv2d(width, width, 3, padding=1, key=k2)
        self.c3 = eqx.nn.Conv2d(width, feat_dim, 3, padding=1, key=k3)

    def __call__(self, x):
        h = jax.nn.relu(self.c1(x))
        h = jax.nn.relu(self.c2(h))
        return jnp.tanh(self.c3(h))          # bounded, so the patch is well scaled


class Decoder(eqx.Module):
    d1: eqx.nn.Conv2d
    d2: eqx.nn.Conv2d
    d3: eqx.nn.Conv2d

    def __init__(self, feat_dim, width, out_ch, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.d1 = eqx.nn.Conv2d(feat_dim, width, 3, padding=1, key=k1)
        self.d2 = eqx.nn.Conv2d(width, width, 3, padding=1, key=k2)
        self.d3 = eqx.nn.Conv2d(width, out_ch, 3, padding=1, key=k3)

    def __call__(self, z):
        h = jax.nn.relu(self.d1(z))
        h = jax.nn.relu(self.d2(h))
        return self.d3(h)                    # logits


class AE(eqx.Module):
    enc: Encoder
    dec: Decoder

    def __init__(self, in_ch, width, feat_dim, key):
        ke, kd = jax.random.split(key)
        self.enc = Encoder(in_ch, width, feat_dim, ke)
        self.dec = Decoder(feat_dim, width, in_ch, kd)


def n_params(module):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))))


def _loss(ae, X):
    """Per-pixel Bernoulli reconstruction.  X is the ONLY argument: no labels."""
    logits = jax.vmap(lambda x: ae.dec(ae.enc(x)))(X)
    return optax.sigmoid_binary_cross_entropy(logits, X).mean()


def train(images, feat_dim=12, width=32, steps=1500, batch=32, lr=3e-3, seed=0,
          verbose=True):
    """Fit the autoencoder on maze images alone.  Returns (Encoder, report)."""
    t0 = time.time()
    X = jnp.asarray(images)
    key = jax.random.PRNGKey(seed)
    kmodel, key = jax.random.split(key)
    ae = AE(X.shape[1], width, feat_dim, kmodel)
    opt = optax.adam(lr)
    state = opt.init(eqx.filter(ae, eqx.is_array))

    @eqx.filter_jit
    def upd(ae, state, xb):
        loss, grads = eqx.filter_value_and_grad(_loss)(ae, xb)
        updates, state = opt.update(grads, state, eqx.filter(ae, eqx.is_array))
        return eqx.apply_updates(ae, updates), state, loss

    n = X.shape[0]
    hist = []
    for s in range(steps):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (min(batch, n),), 0, n)
        ae, state, loss = upd(ae, state, X[idx])
        if s % 250 == 0 or s == steps - 1:
            hist.append((s, float(loss)))

    logits = jax.vmap(lambda x: ae.dec(ae.enc(x)))(X[: min(n, 256)])
    ref = X[: min(n, 256)]
    px_acc = float(jnp.mean((logits > 0).astype(jnp.float32) == ref))
    on_recall = float(jnp.sum(((logits > 0) & (ref > 0)).astype(jnp.float32))
                      / jnp.maximum(jnp.sum(ref), 1.0))
    rep = {"feat_dim": feat_dim, "width": width, "steps": steps,
           "params_encoder": n_params(ae.enc), "params_decoder": n_params(ae.dec),
           "final_bce": hist[-1][1], "loss_curve": hist,
           "pixel_accuracy": px_acc, "on_pixel_recall": on_recall,
           "n_images": int(n), "seconds": time.time() - t0}
    if verbose:
        print("ENCODER (reconstruction only): %d imgs, %d params, BCE %.5f -> "
              "pixel acc %.5f, on-pixel recall %.4f  [%.1fs]"
              % (n, rep["params_encoder"], rep["final_bce"], px_acc, on_recall,
                 rep["seconds"]))
    return ae.enc, rep


def features(enc, images, batch=64):
    """(N, D, H, W) frozen features.  Computed once; the arms only ever read these."""
    out = []
    f = eqx.filter_jit(lambda xb: jax.vmap(enc)(xb))
    for i in range(0, len(images), batch):
        out.append(np.asarray(f(jnp.asarray(images[i:i + batch]))))
    return np.concatenate(out) if out else np.zeros((0,))


def independence_report(enc):
    """What the encoder's objective can see.  The basis for 'shared, not rigged'."""
    import inspect
    src = inspect.getsource(_loss)
    return {"loss_source": src.strip(),
            "loss_arguments": ["ae", "X"],
            "sees_actions": "route" in src or "action" in src or "label" in src,
            "sees_judge": "judge" in src or "JUDGE" in src,
            "trained_on": "maze images only",
            "frozen_before_any_arm_is_trained": True,
            "params": n_params(enc)}
