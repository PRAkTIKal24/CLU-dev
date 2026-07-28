"""⭐ CL-capable read-in ``φ``: small **convolutional** encoders (w26 `cl-encoder`).

**Why this module exists.** The w25 continual-learning entry is a *null* on
Split-CIFAR-10 (CLU 0.149, below LwF 0.162) and the diagnosis was unambiguous: the
failure is the **address space**, not the stream discipline — ``kNN-in-φ`` over the
same PCA-32-of-raw-pixels features caps at **0.21**, so *no* store built on those
addresses can be competitive, while the store's own retrieval on them is near perfect.
PCA (and the 1-hidden-layer MLP autoencoder that shares its objective) measure
*pixel-space* variance, which on CIFAR is dominated by colour and low spatial
frequency, and a 1-NN read-out in that space is translation-sensitive: a shifted
object is far in pixel space. Convolution + spatial pooling removes exactly that
deficiency; a contrastive objective additionally *optimises* the invariances a
nearest-address read-out needs.

**Three arms, all unsupervised (PREREG_CL_PHI §3: no label, no retrieval loss, no
store gradient), all frozen after fitting, none ever trained through the store
(the w20 law):**

* ``randconv`` — the untrained conv trunk. Sees **no data at all**; the honest control
  that says how much of any gain is the architecture rather than the fitting.
* ``convae``  — the trunk + a mirror decoder, **reconstruction MSE** (the objective
  ``PREREG_CL_PHI`` explicitly licenses).
* ``simclr``  — the trunk + a projection head, **NT-Xent** over two augmented views
  (crop / flip / colour-jitter / grayscale). The projection head is discarded and
  ``φ`` reads the trunk, as in the original recipe.

In every arm the trunk's pooled feature map ``h`` is reduced to ``phi_dim`` by a
**PCA head fit on the same fit pool** (optionally whitened), so ``phi_dim`` keeps
meaning exactly what it meant for the PCA/AE arms — the store's address dimension —
and every number stays quotable with its ``phi_dim`` (binding since w24).

Everything here is additive: existing ``pca``/``ae`` callers are bit-identical.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

#: Arms served by this module (dispatched from ``exp_phi_read_in.build_read_in``).
ENCODER_ARMS = ("randconv", "convae", "simclr")

#: Defaults for every knob, used when a config group has not declared it. The
#: canonical home of these knobs is ``ExperimentClEntryConfig`` /
#: ``ExperimentPhiStreamConfig`` in ``chlu/config.py``; this table only keeps older
#: config groups (w23's ``ExperimentPhiReadInConfig``) working unchanged.
ENCODER_DEFAULTS = {
    "enc_channels": (32, 64, 128),
    "enc_pool": 2,             # side of the final average-pooled map (h = C·pool²)
    "enc_groups": 8,           # GroupNorm groups (0 ⇒ no normalisation)
    "enc_steps": 1500,
    "enc_batch": 128,          # images per step (simclr sees 2× that many views)
    "enc_lr": 1e-3,
    "enc_temperature": 0.5,    # NT-Xent τ
    "enc_proj_dim": 64,        # projection head width (discarded after fitting)
    "enc_head": "pca_whiten",  # h → φ: pca | pca_whiten | none
    "enc_l2_normalize": False,  # L2-normalise φ (⚠ changes the store geometry)
    "enc_aug_crop_pad": 4,
    "enc_aug_zoom_p": 0.5,
    "enc_aug_zoom_size": 20,
    "enc_aug_flip_p": 0.5,
    "enc_aug_color_p": 0.8,
    "enc_aug_color_strength": 0.4,
    "enc_aug_gray_p": 0.2,
}


def _p(cfg, name):
    """Read knob ``name`` from ``cfg`` (config-driven), else the documented default."""
    return getattr(cfg, name, ENCODER_DEFAULTS[name])


def image_shape(dataset: str, dim: int):
    """``(C, H, W)`` for a flattened image row. CIFAR-10's python format is
    channel-major (3, 32, 32); MNIST is (1, 28, 28)."""
    if dataset == "cifar10" or dim == 3072:
        return (3, 32, 32)
    if dataset == "mnist" or dim == 784:
        return (1, 28, 28)
    side = int(round(dim**0.5))
    if side * side != dim:
        raise ValueError(f"cannot infer an image shape for dim={dim} ({dataset!r})")
    return (1, side, side)


# ---------------------------------------------------------------------------
# The trunk
# ---------------------------------------------------------------------------


class ConvTrunk(eqx.Module):
    """A small conv trunk: ``[conv → norm → relu] × len(channels)`` with a 2× max-pool
    after each block, then average-pool to ``pool × pool`` and flatten.

    GroupNorm (not BatchNorm) because it is stateless — no ``eqx`` state threading, and
    identical at fit and read time, which matters for a **frozen** ``φ``.
    """

    convs: list
    norms: list
    pool: int
    h_dim: int

    def __init__(self, in_ch, channels, pool, groups, key):
        keys = jax.random.split(key, len(channels))
        convs, norms, c_in = [], [], in_ch
        for c_out, k in zip(channels, keys, strict=False):
            convs.append(eqx.nn.Conv2d(c_in, c_out, 3, padding=1, key=k))
            norms.append(
                eqx.nn.GroupNorm(min(groups, c_out), c_out) if groups > 0 else None
            )
            c_in = c_out
        self.convs = convs
        self.norms = norms
        self.pool = int(pool)
        self.h_dim = int(channels[-1] * pool * pool)

    def __call__(self, x):
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.relu(x)
            x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = eqx.nn.AdaptiveAvgPool2d(self.pool)(x)
        return x.reshape(-1)


class _ProjHead(eqx.Module):
    """SimCLR projection head ``g`` — trained, then **discarded** (φ reads the trunk)."""

    mlp: eqx.nn.MLP

    def __init__(self, h_dim, out_dim, key):
        self.mlp = eqx.nn.MLP(h_dim, out_dim, h_dim, 1, activation=jax.nn.relu, key=key)

    def __call__(self, h):
        return self.mlp(h)


class _Decoder(eqx.Module):
    """Mirror decoder for the ``convae`` arm: ``h → 4×4 map → ×2 upsample + conv → image``."""

    lift: eqx.nn.Linear
    convs: list
    out_shape: tuple
    c0: int

    def __init__(self, h_dim, channels, out_shape, key):
        keys = jax.random.split(key, len(channels) + 1)
        self.c0 = int(channels[-1])
        self.lift = eqx.nn.Linear(h_dim, self.c0 * 16, key=keys[0])
        chans = list(channels[::-1]) + [out_shape[0]]
        self.convs = [
            eqx.nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, key=keys[i + 1])
            for i in range(len(chans) - 1)
        ]
        self.out_shape = tuple(out_shape)

    def __call__(self, h):
        x = self.lift(h).reshape(self.c0, 4, 4)
        for i, conv in enumerate(self.convs):
            side = min(4 * 2 ** (i + 1), self.out_shape[1])
            x = jax.image.resize(x, (x.shape[0], side, side), method="bilinear")
            x = conv(x)
            if i < len(self.convs) - 1:
                x = jax.nn.relu(x)
        x = jax.image.resize(x, self.out_shape, method="bilinear")
        return jax.nn.sigmoid(x)


# ---------------------------------------------------------------------------
# Augmentations (SimCLR) — traceable, key-threaded, fixed shapes
# ---------------------------------------------------------------------------


def augment(x, key, cfg):
    """One augmented view of ``x`` (C,H,W) in [0,1]. Crop (translate + zoom), flip,
    colour jitter, grayscale — the crop/colour pair the SimCLR ablations identify as
    the load-bearing one. All branches are ``jnp.where``/``lax`` so the whole thing
    stays traceable (repo convention, cf. ``integrators.langevin_step``)."""
    C, H, W = x.shape
    k_off, k_flip, k_zoom, k_zoff, k_col, k_b, k_c, k_s, k_gray = jax.random.split(key, 9)

    pad = int(_p(cfg, "enc_aug_crop_pad"))
    if pad > 0:
        xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        oy, ox = jax.random.randint(k_off, (2,), 0, 2 * pad + 1)
        x = jax.lax.dynamic_slice(xp, (0, oy, ox), (C, H, W))

    x = jnp.where(jax.random.bernoulli(k_flip, _p(cfg, "enc_aug_flip_p")),
                  x[:, :, ::-1], x)

    zs = int(_p(cfg, "enc_aug_zoom_size"))
    if 0 < zs < H:
        oy, ox = jax.random.randint(k_zoff, (2,), 0, H - zs + 1)
        crop = jax.lax.dynamic_slice(x, (0, oy, ox), (C, zs, zs))
        zoomed = jax.image.resize(crop, (C, H, W), method="bilinear")
        x = jnp.where(jax.random.bernoulli(k_zoom, _p(cfg, "enc_aug_zoom_p")), zoomed, x)

    s = float(_p(cfg, "enc_aug_color_strength"))
    do_col = jax.random.bernoulli(k_col, _p(cfg, "enc_aug_color_p"))
    bright = jax.random.uniform(k_b, (), minval=1 - s, maxval=1 + s)
    contr = jax.random.uniform(k_c, (), minval=1 - s, maxval=1 + s)
    xc = x * bright
    xc = xc.mean() + (xc - xc.mean()) * contr
    if C == 3:
        sat = jax.random.uniform(k_s, (), minval=1 - s, maxval=1 + s)
        gray = xc.mean(axis=0, keepdims=True)
        xc = gray + (xc - gray) * sat
    x = jnp.where(do_col, xc, x)

    if C == 3:
        gray = jnp.repeat(x.mean(axis=0, keepdims=True), 3, axis=0)
        x = jnp.where(jax.random.bernoulli(k_gray, _p(cfg, "enc_aug_gray_p")), gray, x)
    return jnp.clip(x, 0.0, 1.0)


def nt_xent(z, tau):
    """NT-Xent over ``z`` = (2N, d) with view pairs ``(i, i+N)`` (Chen et al. 2020)."""
    z = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    n2 = z.shape[0]
    n = n2 // 2
    sim = (z @ z.T) / tau
    sim = sim - jnp.eye(n2) * 1e9  # mask self-similarity
    pos = jnp.concatenate([jnp.arange(n, n2), jnp.arange(0, n)])
    return float_mean_ce(sim, pos)


def float_mean_ce(logits, targets):
    logp = jax.nn.log_softmax(logits, axis=1)
    return -jnp.mean(logp[jnp.arange(logits.shape[0]), targets])


# ---------------------------------------------------------------------------
# h → φ head (PCA, optionally whitened) — keeps ``phi_dim`` meaningful
# ---------------------------------------------------------------------------


class _PCAHead:
    """PCA-k of the trunk features, fit on the SAME (unsupervised) fit pool.

    Not :class:`chlu.experiments.exp_phi_read_in.PCAReadIn` because that one is fit on
    *images* and cannot whiten; whitening equalises the trunk's feature scales, which a
    1-NN read-out in the address space is directly sensitive to.
    """

    def __init__(self, H, k, whiten):
        H = np.asarray(H, np.float64)
        self.mean = H.mean(axis=0)
        Hc = H - self.mean
        _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        k = int(min(k, Vt.shape[0]))
        self.components = Vt[:k].astype(np.float32)
        scale = S[:k] / np.sqrt(max(len(H) - 1, 1))
        self.scale = (1.0 / np.maximum(scale, 1e-6)).astype(np.float32) if whiten \
            else np.ones(k, np.float32)
        self.k = k

    def __call__(self, H):
        H = np.asarray(H, np.float32)
        return (H - self.mean.astype(np.float32)) @ self.components.T * self.scale


# ---------------------------------------------------------------------------
# The read-in
# ---------------------------------------------------------------------------


class ConvEncoderReadIn:
    """A frozen convolutional read-in ``φ(x) = head(trunk(x))``.

    ``objective ∈ {"none", "recon", "simclr"}`` selects the arm. Fitting only ever sees
    ``fit_pool`` (the regime's own pool — task-1 classes only for the PRIMARY arm), never
    a stored item, never a label, never the store.
    """

    def __init__(self, fit_pool, dataset, dim, phi_dim, cfg, seed, objective):
        self.shape = image_shape(dataset, dim)
        self.objective = objective
        self.chunk = 512
        key = jax.random.PRNGKey(seed + 20260728)
        key, tk = jax.random.split(key)
        trunk = ConvTrunk(self.shape[0], tuple(_p(cfg, "enc_channels")),
                          int(_p(cfg, "enc_pool")), int(_p(cfg, "enc_groups")), tk)
        X = jnp.asarray(np.asarray(fit_pool, np.float32).reshape((-1,) + self.shape))
        self.n_fit = int(X.shape[0])
        self.steps_run = 0
        self.loss_first = None
        self.loss_final = None
        if objective == "simclr":
            trunk, key = self._fit_simclr(trunk, X, cfg, key)
        elif objective == "recon":
            trunk, key = self._fit_recon(trunk, X, cfg, key)
        self._trunk = eqx.filter_jit(jax.vmap(trunk))
        self.h_dim = int(trunk.h_dim)
        H = self._features(np.asarray(fit_pool, np.float32))
        head = str(_p(cfg, "enc_head"))
        self.head = None if head == "none" else _PCAHead(
            H, phi_dim, whiten=(head == "pca_whiten")
        )
        self.k = int(phi_dim if self.head is None else self.head.k)
        self.l2_normalize = bool(_p(cfg, "enc_l2_normalize"))

    # -- fitting ---------------------------------------------------------
    def _fit_simclr(self, trunk, X, cfg, key):
        steps, bs = int(_p(cfg, "enc_steps")), int(_p(cfg, "enc_batch"))
        bs = min(bs, int(X.shape[0]))
        tau = float(_p(cfg, "enc_temperature"))
        key, hk = jax.random.split(key)
        model = (trunk, _ProjHead(trunk.h_dim, int(_p(cfg, "enc_proj_dim")), hk))
        opt = optax.adam(float(_p(cfg, "enc_lr")))
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_value_and_grad
        def loss_fn(m, xb, ak):
            tr, head = m
            k1, k2 = jax.random.split(ak)
            n = xb.shape[0]
            v1 = jax.vmap(lambda x, k: augment(x, k, cfg))(xb, jax.random.split(k1, n))
            v2 = jax.vmap(lambda x, k: augment(x, k, cfg))(xb, jax.random.split(k2, n))
            h = jax.vmap(tr)(jnp.concatenate([v1, v2]))
            return nt_xent(jax.vmap(head)(h), tau)

        @eqx.filter_jit
        def step(m, opt_state, xb, ak):
            loss, grads = loss_fn(m, xb, ak)
            updates, opt_state = opt.update(grads, opt_state, m)
            return eqx.apply_updates(m, updates), opt_state, loss

        for i in range(steps):
            key, sk, ak = jax.random.split(key, 3)
            idx = jax.random.choice(sk, X.shape[0], (bs,), replace=False)
            model, opt_state, loss = step(model, opt_state, X[idx], ak)
            self.loss_final = float(loss)
            if i == 0:
                self.loss_first = float(loss)
            self.steps_run = i + 1
        return model[0], key

    def _fit_recon(self, trunk, X, cfg, key):
        steps, bs = int(_p(cfg, "enc_steps")), int(_p(cfg, "enc_batch"))
        bs = min(bs, int(X.shape[0]))
        key, dk = jax.random.split(key)
        model = (trunk, _Decoder(trunk.h_dim, tuple(_p(cfg, "enc_channels")),
                                 self.shape, dk))
        opt = optax.adam(float(_p(cfg, "enc_lr")))
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_value_and_grad
        def loss_fn(m, xb):
            tr, dec = m
            recon = jax.vmap(lambda x: dec(tr(x)))(xb)
            return jnp.mean((recon - xb) ** 2)

        @eqx.filter_jit
        def step(m, opt_state, xb):
            loss, grads = loss_fn(m, xb)
            updates, opt_state = opt.update(grads, opt_state, m)
            return eqx.apply_updates(m, updates), opt_state, loss

        for i in range(steps):
            key, sk = jax.random.split(key)
            idx = jax.random.choice(sk, X.shape[0], (bs,), replace=False)
            model, opt_state, loss = step(model, opt_state, X[idx])
            self.loss_final = float(loss)
            if i == 0:
                self.loss_first = float(loss)
            self.steps_run = i + 1
        return model[0], key

    # -- reading ---------------------------------------------------------
    def _features(self, X):
        X = np.asarray(X, np.float32).reshape((-1,) + self.shape)
        out = [np.asarray(self._trunk(jnp.asarray(X[i:i + self.chunk])))
               for i in range(0, len(X), self.chunk)]
        return np.concatenate(out) if out else np.zeros((0, self.h_dim), np.float32)

    def __call__(self, X):
        H = self._features(X)
        F = H if self.head is None else self.head(H)
        if self.l2_normalize:
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        return jnp.asarray(np.asarray(F, np.float32))

    def provenance(self, arm):
        return {
            "arm": arm, "k": self.k, "h_dim": self.h_dim,
            "objective": self.objective, "steps": self.steps_run,
            "loss_first": self.loss_first, "loss_final": self.loss_final,
            "n_fit": self.n_fit, "frozen": True,
        }


def build_encoder_read_in(arm, dataset, store_pool, fit_pool, cfg, seed):
    """Dispatch for the conv arms (called from ``exp_phi_read_in.build_read_in``)."""
    objective = {"randconv": "none", "convae": "recon", "simclr": "simclr"}[arm]
    dim = int(np.asarray(store_pool).shape[1])
    phi = ConvEncoderReadIn(fit_pool, dataset, dim, int(cfg.phi_dim), cfg, seed,
                            objective)
    return phi, phi.provenance(arm)
