"""The frozen encoder / feature cache / **one shared frozen decoder** bridge.

This module is the *instrument* the whole generative phase-1 cell reads through.
Nothing here is evidence for the CLU primitive: there is no store, no well, no
arm and no composition claim.  What it provides is

1. a small conv **encoder** trained on the Montero *train* split only, with a
   declared pretext (``recon`` — the primary, R1 — or ``cls``, the diagnostic),
   then **frozen** and hashed;
2. a **feature cache** of the encoder's penultimate feature maps, written once
   with a checksum and a manifest, read by every downstream arm and never
   recomputed per arm;
3. **THE decoder** — one conv renderer from penultimate-feature space back to
   pixels, fit on real ``(feature, image)`` pairs from the **train split only**,
   frozen, hashed, and shared by every arm forever.  ⛔ No arm may substitute
   its own; :func:`assert_decoder_hash` is how a downstream spoke checks.

⭐ **The design constraint on the feature dimension.**  ``D = C·H·W`` of the
penultimate map is *not* a free choice: the LACE baseline is only valid in
whitened coordinates and needs a ``D × D`` eigendecomposition, which is only
feasible up to ``D ≲ 2e4``.  The shipped default is ``128 × 4 × 4 = 2048``.
:func:`whitening_feasibility` reports the verdict that goes in the manifest.

⛔ **Images are never materialised.**  ``open_images()`` hands back a lazy h5py
handle over 1.7 GB (Shapes3D); :func:`stream_images` walks it in *contiguous*
blocks.  The files are gzip-chunked as ``(15000, 4, 4, 1)``, so scattered fancy
indexing is ~40× slower than a block scan — measured, not assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.gen_splits import GenSplit, assert_holdout_disjoint, file_sha256

PRETEXTS = ("recon", "cls")
#: (channels, height, width) of one image, per dataset.
IMAGE_SHAPES = {"shapes3d": (3, 64, 64), "dsprites": (1, 64, 64)}
#: h5py dataset key holding the images, per dataset.
IMAGE_KEYS = {"shapes3d": "images", "dsprites": "imgs"}
#: Above this ``D`` a full ``D × D`` whitening becomes infeasible and the LACE
#: arm would fall back to channel-only whitening — a declared deviation that
#: weakens the strongest competitor arm (``gen-lace-spec``).
WHITENING_D_LIMIT = 20_000


# ---------------------------------------------------------------------------
# Dataset plumbing (⛔ never re-derive a split or a loader here)
# ---------------------------------------------------------------------------


def dataset_module(dataset: str):
    """Return the loader module for ``dataset`` (``chlu.data.{shapes3d,dsprites}``)."""
    if dataset == "shapes3d":
        from ..data import shapes3d as mod
    elif dataset == "dsprites":
        from ..data import dsprites as mod
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected shapes3d|dsprites)")
    return mod


def image_shape(dataset: str) -> tuple[int, int, int]:
    try:
        return IMAGE_SHAPES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {dataset!r}") from exc


def stream_images(
    dataset: str,
    rows: np.ndarray,
    root: str | Path | None = None,
    block: int = 8192,
    handle=None,
) -> np.ndarray:
    """Read ``rows`` as float32 ``(N, C, H, W)`` in [0, 1] by contiguous blocks.

    ⛔ Never does ``f[key][:]``.  Blocks that contain no requested row are
    skipped entirely; a block that does is read once and gathered from.

    Args:
        dataset: ``shapes3d`` | ``dsprites``.
        rows: row indices into the exhaustive factor grid (any order).
        root: dataset root (``None`` -> ``$CHLU_GEN_DATA_ROOT``).
        block: contiguous rows per read.
        handle: optional already-open h5py File (kept open by the caller).
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        c, h, w = image_shape(dataset)
        return np.empty((0, c, h, w), dtype=np.float32)
    mod = dataset_module(dataset)
    key = IMAGE_KEYS[dataset]
    order = np.argsort(rows, kind="stable")
    srows = rows[order]
    own = handle is None
    f = mod.open_images(root) if own else handle
    try:
        data = f[key]
        n = int(data.shape[0])
        # shape read from the FILE, not from a constant: a loader that changed
        # resolution must fail loudly downstream, not be silently reshaped here.
        rest = tuple(int(x) for x in data.shape[1:])
        c, h, w = (1, rest[0], rest[1]) if len(rest) == 2 else (rest[2], rest[0], rest[1])
        out = np.empty((rows.size, c, h, w), dtype=np.float32)
        if int(srows[-1]) >= n:
            raise IndexError(f"row {int(srows[-1])} is outside {dataset} ({n} rows)")
        start = 0
        for lo in range(0, n, block):
            if start >= srows.size:
                break
            hi = min(lo + block, n)
            end = start + int(np.searchsorted(srows[start:], hi, side="left"))
            if end == start:
                continue
            chunk = np.asarray(data[lo:hi])
            sel = srows[start:end] - lo
            imgs = chunk[sel]
            if imgs.ndim == 3:  # (n, H, W) -> (n, 1, H, W)
                imgs = imgs[:, None, :, :]
            else:  # (n, H, W, C) -> (n, C, H, W)
                imgs = np.transpose(imgs, (0, 3, 1, 2))
            scale = 255.0 if imgs.dtype == np.uint8 else 1.0
            out[order[start:end]] = imgs.astype(np.float32) / scale
            start = end
        if start != srows.size:
            raise RuntimeError(
                f"{dataset}: streamed {start} of {srows.size} requested rows"
            )
    finally:
        if own:
            f.close()
    return out


# ---------------------------------------------------------------------------
# Hashing / provenance
# ---------------------------------------------------------------------------


def param_hash(model) -> str:
    """sha256 over an ``eqx.Module``'s array leaves (shape + dtype + bytes).

    Deterministic across processes: leaves are taken in ``jax.tree`` order,
    which is the module's own field order, and every leaf contributes its shape
    and dtype as well as its bytes, so two differently-shaped models with the
    same byte count cannot collide.
    """
    params, _ = eqx.partition(model, eqx.is_array)
    hasher = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        arr = np.asarray(leaf)
        hasher.update(str(arr.shape).encode())
        hasher.update(str(arr.dtype).encode())
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def param_count(model) -> int:
    """Number of float parameters carried by ``model``."""
    params, _ = eqx.partition(model, eqx.is_array)
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def assert_decoder_hash(decoder, expected: str) -> str:
    """⭐ Downstream guard: an arm that silently retrained the decoder fails here.

    Every arm renders through the ONE decoder this module froze.  Call this with
    the hash from ``bridge_manifest.json`` before scoring anything.
    """
    got = param_hash(decoder)
    if got != expected:
        raise AssertionError(
            "decoder parameter hash mismatch — this is NOT the shared frozen "
            f"decoder (expected {expected}, got {got}). Every arm must decode "
            "through the same renderer; re-training it invalidates the cell."
        )
    return got


def whitening_feasibility(d: int, n: int) -> dict:
    """LACE whitening verdict at penultimate dimension ``d`` and ``n`` train features."""
    return {
        "D": int(d),
        "N_train_features": int(n),
        "limit": WHITENING_D_LIMIT,
        "full_whitening_feasible": bool(d <= WHITENING_D_LIMIT),
        "shrinkage_mandatory": bool(d > n),
        # ⭐ Head ruling (2026-09-09): our architecture choices never weaken a
        # competitor. Either pick a width LACE can whiten NATIVELY (option 1) or
        # keep a wider encoder and hand LACE a DECLARED PCA projection, reported
        # both ways (option 2). ⛔ Channel-only whitening is never silent.
        "lace_option": (
            "1 — feature width chosen so LACE whitens natively (full D x D "
            "shrinkage covariance over the cached train features)"
            if d <= WHITENING_D_LIMIT
            else "2 — wider encoder kept; LACE gets a DECLARED PCA projection and "
            "the arm is reported both ways"
        ),
        "note": (
            f"D={d} <= {WHITENING_D_LIMIT}: a full D x D eigendecomposition is "
            "feasible, so the LACE arm runs in fully whitened coordinates and is "
            "NOT pushed onto channel-only whitening."
            if d <= WHITENING_D_LIMIT
            else f"D={d} > {WHITENING_D_LIMIT}: full whitening is infeasible; the "
            "LACE arm would fall back to channel-only whitening. HEAD CALL."
        ),
    }


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class GenEncoder(eqx.Module):
    """Small conv encoder: ``[conv(k, stride) -> GroupNorm -> relu] x len(channels)``.

    The **penultimate feature map** is the output of the last block (post-ReLU),
    i.e. what LACE would have to whiten.  GroupNorm (not BatchNorm) because it is
    stateless and identical at fit and read time — which is what "frozen" has to
    mean.
    """

    convs: list
    norms: list
    in_shape: tuple
    feat_shape: tuple

    def __init__(self, in_shape, channels, kernel, stride, groups, key):
        keys = jax.random.split(key, len(channels))
        convs, norms = [], []
        c_in = int(in_shape[0])
        pad = max((int(kernel) - int(stride)) // 2, 0)
        for c_out, k in zip(channels, keys, strict=True):
            convs.append(
                eqx.nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=pad, key=k)
            )
            norms.append(
                eqx.nn.GroupNorm(min(groups, c_out), c_out) if groups > 0 else None
            )
            c_in = c_out
        self.convs = convs
        self.norms = norms
        self.in_shape = tuple(int(x) for x in in_shape)
        # measured, not assumed: a kernel/stride/padding triple that does not
        # halve exactly would otherwise give a wrong D in the manifest.
        self.feat_shape = tuple(
            int(x)
            for x in jax.eval_shape(
                self.__call__, jax.ShapeDtypeStruct(self.in_shape, self.param_dtype)
            ).shape
        )

    @property
    def param_dtype(self):
        """⚠ §7.23: a module-scoped ``jax_enable_x64`` fixture elsewhere in the
        suite makes eqx build float64 weights; a float32 input then crashes the
        conv. Every module here casts its input rather than assuming float32."""
        return self.convs[0].weight.dtype

    def __call__(self, x):
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.relu(x)
        return x

    @property
    def feature_dim(self) -> int:
        """``D = C·H·W`` of the penultimate map — the LACE design constraint."""
        c, h, w = self.feat_shape
        return int(c * h * w)


def _n_upsamples(feat_shape, out_shape) -> int:
    """Number of x2 steps from the feature map to the image (raises if inexact)."""
    n_up, side = 0, int(feat_shape[1])
    while side < int(out_shape[1]):
        side *= 2
        n_up += 1
    if side != int(out_shape[1]):
        raise ValueError(
            f"feature side {feat_shape[1]} does not reach {out_shape[1]} by x2 steps"
        )
    return n_up


class GenDecoder(eqx.Module):
    """Conv renderer: penultimate feature map -> pixels in [0, 1].

    ``[bilinear x2 -> conv3 -> GroupNorm -> relu] x n`` then a final conv +
    sigmoid, with ``n`` fixed by the encoder's spatial reduction so the output
    lands exactly on the image size.
    """

    convs: list
    norms: list
    feat_shape: tuple
    out_shape: tuple

    def __init__(self, feat_shape, out_shape, channels, groups, key, blocks_per_stage=1):
        n_up = _n_upsamples(feat_shape, out_shape)
        b = max(1, int(blocks_per_stage))
        chans = list(channels[:n_up])
        while len(chans) < n_up:
            chans.append(chans[-1] if chans else feat_shape[0])
        # ``blocks_per_stage`` repeats each stage's channel width; b == 1 leaves
        # the key split, the conv order and every constructor argument
        # bit-identical to the shipped decoder (asserted in the tests).
        chans = [c for c in chans for _ in range(b)] + [out_shape[0]]
        keys = jax.random.split(key, len(chans))
        convs, norms, c_in = [], [], int(feat_shape[0])
        for i, (c_out, k) in enumerate(zip(chans, keys, strict=True)):
            convs.append(eqx.nn.Conv2d(c_in, c_out, 3, padding=1, key=k))
            last = i == len(chans) - 1
            norms.append(
                eqx.nn.GroupNorm(min(groups, c_out), c_out)
                if (groups > 0 and not last)
                else None
            )
            c_in = c_out
        self.convs = convs
        self.norms = norms
        self.feat_shape = tuple(feat_shape)
        self.out_shape = tuple(out_shape)

    @property
    def blocks_per_stage(self) -> int:
        """Derived, never stored — an added pytree field would break the shipped
        ``decoder.pkl`` (``eqx.partition`` + pickle round-trips the module itself)."""
        return (len(self.convs) - 1) // _n_upsamples(self.feat_shape, self.out_shape)

    def __call__(self, z):
        x = jnp.asarray(z, self.convs[0].weight.dtype)
        b = self.blocks_per_stage
        n_conv_up = len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            if i < n_conv_up and i % b == 0:
                x = jax.image.resize(
                    x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear"
                )
            x = conv(x)
            if norm is not None:
                x = norm(x)
            if i < len(self.convs) - 1:
                x = jax.nn.relu(x)
        return jax.nn.sigmoid(x)


class FactorHeads(eqx.Module):
    """One linear head per factor over a flattened feature map (logits per factor)."""

    linears: list

    def __init__(self, in_dim, cardinalities, key):
        keys = jax.random.split(key, len(cardinalities))
        self.linears = [
            eqx.nn.Linear(in_dim, int(c), key=k)
            for c, k in zip(cardinalities, keys, strict=True)
        ]

    def __call__(self, z):
        flat = jnp.asarray(z, self.linears[0].weight.dtype).reshape(-1)
        return [lin(flat) for lin in self.linears]


class FactorProbe(eqx.Module):
    """⭐ The measurement instrument: a per-factor classifier over **images**.

    Trained on **real** train-split images with factor labels.  R1's "no label
    supervision" governs the generative path; the evaluation probe is
    measurement and keeps its labels — stripping them would destroy the
    instrument, not protect the claim.

    Architecturally independent of :class:`GenEncoder` so that a probe result is
    not a restatement of the encoder's own inductive bias.
    """

    convs: list
    norms: list
    heads: FactorHeads

    def __init__(self, in_shape, channels, groups, cardinalities, key):
        keys = jax.random.split(key, len(channels) + 1)
        convs, norms = [], []
        c_in, h, w = in_shape
        for c_out, k in zip(channels, keys[:-1], strict=True):
            convs.append(eqx.nn.Conv2d(c_in, c_out, 4, stride=2, padding=1, key=k))
            norms.append(
                eqx.nn.GroupNorm(min(groups, c_out), c_out) if groups > 0 else None
            )
            c_in = c_out
            h, w = h // 2, w // 2
        self.convs = convs
        self.norms = norms
        self.heads = FactorHeads(int(c_in * h * w), cardinalities, keys[-1])

    def __call__(self, x):
        x = jnp.asarray(x, self.convs[0].weight.dtype)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x)
            if norm is not None:
                x = norm(x)
            x = jax.nn.relu(x)
        return self.heads(x)


# ---------------------------------------------------------------------------
# Training (Optax Adam, explicit PRNGKey threading — repo convention)
# ---------------------------------------------------------------------------


def model_dtype(model):
    """Parameter dtype of ``model`` — under ``jax_enable_x64`` eqx builds float64
    weights and a float32 input would crash the conv (phi_encoders precedent)."""
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.result_type(leaves[0]) if leaves else jnp.float32


def _sgd_loop(
    model,
    loss_fn,
    data,
    steps,
    batch,
    lr,
    key,
    log: Optional[Callable] = None,
    schedule=None,
):
    """Minibatch Adam loop.  ``loss_fn(model, *batch_arrays) -> scalar``.

    ``schedule=None`` (the default everywhere except an explicitly non-constant
    decoder recipe) keeps the shipped constant-LR ``optax.adam(lr)`` **exactly**.
    """
    opt = optax.adam(lr if schedule is None else schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    n = int(data[0].shape[0])
    batch = min(int(batch), n)

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(m, s, *arrays):
        loss, grads = grad_fn(m, *arrays)
        updates, s = opt.update(grads, s, m)
        return eqx.apply_updates(m, updates), s, loss

    history = []
    for i in range(int(steps)):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        model, opt_state, loss = step(model, opt_state, *[a[idx] for a in data])
        loss = float(loss)
        history.append(loss)
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    step {i + 1}/{steps}  loss={loss:.5f}")
    return model, history, key


def _xent(logits_list, labels):
    """Mean over factors of the per-factor cross-entropy (labels: (B, n_factors))."""
    total = 0.0
    for j, logits in enumerate(logits_list):
        logp = jax.nn.log_softmax(logits, axis=-1)
        total = total - jnp.mean(
            jnp.take_along_axis(logp, labels[:, j][:, None], axis=-1)
        )
    return total / len(logits_list)


def train_encoder(
    images: np.ndarray,
    pretext: str,
    cfg,
    key,
    labels: Optional[np.ndarray] = None,
    cardinalities: Optional[Sequence[int]] = None,
    log: Optional[Callable] = None,
):
    """Fit the encoder on train-split images only; return ``(encoder, aux, history)``.

    ``pretext="recon"`` (R1 primary) trains against a **throwaway** decoder that
    is discarded — the shared decoder is fit separately against the *frozen*
    encoder, so it is exactly the renderer used at evaluation.
    ``pretext="cls"`` is the diagnostic arm and trains linear per-factor heads.
    """
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}, got {pretext!r}")
    in_shape = tuple(images.shape[1:])
    key, ek, ak = jax.random.split(key, 3)
    encoder = GenEncoder(
        in_shape,
        tuple(cfg.enc_channels),
        int(cfg.enc_kernel),
        int(cfg.enc_stride),
        int(cfg.enc_groups),
        ek,
    )
    dtype = model_dtype(encoder)
    X = jnp.asarray(images, dtype)

    if pretext == "recon":
        aux = GenDecoder(
            encoder.feat_shape, in_shape, tuple(cfg.dec_channels), int(cfg.dec_groups), ak
        )

        def loss_fn(m, xb):
            enc, dec = m
            recon = jax.vmap(lambda x: dec(enc(x)))(xb)
            return jnp.mean((recon - xb) ** 2)

        data = (X,)
    else:
        if labels is None or cardinalities is None:
            raise ValueError("pretext='cls' needs labels and cardinalities")
        aux = FactorHeads(encoder.feature_dim, cardinalities, ak)
        Y = jnp.asarray(labels, jnp.int32)

        def loss_fn(m, xb, yb):
            enc, heads = m
            logits = jax.vmap(lambda x: heads(enc(x)))(xb)
            return _xent(logits, yb)

        data = (X, Y)

    model, history, key = _sgd_loop(
        (encoder, aux), loss_fn, data, cfg.enc_steps, cfg.enc_batch, cfg.enc_lr, key, log
    )
    return model[0], model[1], history


# ---------------------------------------------------------------------------
# ⭐ THE DECODER RECIPE (gen-decoder-lift)
#
# A decoder is a **recipe** — architecture + schedule + loss, refit from scratch
# against whatever feature cache it is handed — ⛔ not one fitted ``.pkl`` that
# every rung reuses.  A later factorisation rung retrains the encoder, so its
# feature space differs and its decoder must be refit and scored against *its
# own* ceiling.  ⛔ Nothing below is a function of the feature width ``D``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderRecipe:
    """Resolved decoder hyperparameters (a config, not a fitted artifact)."""

    name: str
    channels: tuple
    groups: int
    blocks_per_stage: int
    steps: int
    batch: int
    lr: float
    loss: str
    edge_weight: float
    edge_blur: int
    grad_weight: float
    lr_schedule: str
    warmup_frac: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "channels": [int(c) for c in self.channels],
            "groups": int(self.groups),
            "blocks_per_stage": int(self.blocks_per_stage),
            "steps": int(self.steps),
            "batch": int(self.batch),
            "lr": float(self.lr),
            "loss": self.loss,
            "edge_weight": float(self.edge_weight),
            "edge_blur": int(self.edge_blur),
            "grad_weight": float(self.grad_weight),
            "lr_schedule": self.lr_schedule,
            "warmup_frac": float(self.warmup_frac),
        }


#: Named presets.  A recipe overrides only the ``GenBridgeConfig`` fields it
#: names; ``"baseline"`` names none, so it reproduces the shipped decoder
#: bit-identically (asserted in ``tests/test_gen_decoder_lift.py``).
DECODER_RECIPES: dict = {
    "baseline": {},
    # ⭐ V1 — capacity + schedule.  Filed as the *null control* ("is the renderer
    # merely too small / undertrained?") and it WON: on Shapes3D/recon, 3 seeds,
    # against the frozen encoder and the untouched feature cache, held-out
    # `shape` goes 0.5926 -> 0.9398 and the R16 ratio goes 0.653 -> 0.966
    # (random-A) / 0.698 -> 0.996 (matched-seen), clearing the 0.8 floor on
    # every seed and both normalisers.  Decode MSE 0.0030 -> 0.0011.
    # ⛔ The shipped decoder was under-resourced, not compositionally blocked.
    # Cost: 1,516,227 params (6.05x) and ~16x the baseline CPU training time.
    "v1_capacity_schedule": {
        "dec_width_mult": 2.0,
        "dec_blocks_per_stage": 2,
        # 1800, not the 4800 first drafted: hand-counted at 6.69x the baseline
        # per-step MAC count, 4800 steps is ~25 h of CPU for three seeds, outside
        # this task's bound. Cut before any V1 run and recorded in PREREG.md §6;
        # the training-loss curve is reported so "undertrained" stays checkable.
        "dec_steps": 1800,
        "dec_lr_schedule": "cosine",
        "dec_warmup_frac": 0.05,
    },
    # ⛔ V2 — the objective.  Architecture and schedule identical to the shipped
    # baseline so the comparison isolates the loss.  It LOST: it renders the
    # silhouettes it has SEEN much better (matched-seen `shape` 0.849 -> 0.974)
    # and turns the held-out one into a coin flip (0.139 / 0.847 / 0.583 across
    # seeds, mean 0.523 vs the baseline's 0.593).  Kept, reported, and ⛔ NOT
    # recommended; a sharper silhouette objective buys confidence, not identity.
    "v2_object_weighted": {
        "dec_loss": "object_weighted",
        "dec_edge_weight": 8.0,
        "dec_edge_blur": 5,
        "dec_grad_weight": 1.0,
    },
}


def resolve_decoder_cfg(cfg, recipe: Optional[str] = None) -> DecoderRecipe:
    """Fold ``DECODER_RECIPES[name]`` onto ``cfg``'s ``dec_*`` knobs.

    The knobs remain the source of truth; a recipe is a named set of overrides,
    so ``dec_recipe="baseline"`` (the default) changes nothing.
    """
    name = str(recipe if recipe is not None else getattr(cfg, "dec_recipe", "baseline"))
    if name not in DECODER_RECIPES:
        raise ValueError(
            f"unknown decoder recipe {name!r} (have {sorted(DECODER_RECIPES)})"
        )
    over = DECODER_RECIPES[name]

    def g(field, default=None):
        return over.get(field, getattr(cfg, field, default))

    width = float(g("dec_width_mult", 1.0))
    chans = tuple(max(1, int(round(float(c) * width))) for c in g("dec_channels"))
    return DecoderRecipe(
        name=name,
        channels=chans,
        groups=int(g("dec_groups")),
        blocks_per_stage=int(g("dec_blocks_per_stage", 1)),
        steps=int(g("dec_steps")),
        batch=int(g("dec_batch")),
        lr=float(g("dec_lr")),
        loss=str(g("dec_loss", "mse")),
        edge_weight=float(g("dec_edge_weight", 8.0)),
        edge_blur=int(g("dec_edge_blur", 5)),
        grad_weight=float(g("dec_grad_weight", 1.0)),
        lr_schedule=str(g("dec_lr_schedule", "constant")),
        warmup_frac=float(g("dec_warmup_frac", 0.05)),
    )


def make_decoder(feat_shape, out_shape, recipe: DecoderRecipe, key):
    """Build (do not fit) the recipe's decoder for an arbitrary feature shape."""
    return GenDecoder(
        feat_shape,
        out_shape,
        recipe.channels,
        recipe.groups,
        key,
        blocks_per_stage=recipe.blocks_per_stage,
    )


def _fdiff(x, axis: int):
    """Forward difference along ``axis``, zero-padded so the shape is preserved."""
    d = jnp.diff(x, axis=axis)
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, 1)
    return jnp.pad(d, pad)


def edge_weight_map(images, alpha: float, blur: int):
    """Per-pixel weights ``w = 1 + alpha * s`` from the **target image only**.

    ``s`` is a box-blurred luminance-gradient magnitude renormalised to mean 1
    *per image*, so a low-contrast image (the Montero hue corner is exactly
    that) still gets its object boundary weighted — the weighting is about
    *where* the error is, not how bright the image is.
    ⛔ No factor label is involved: this is a function of the pixels (R1).
    """
    y = jnp.mean(images, axis=1)  # (B, H, W)
    g = jnp.abs(_fdiff(y, -2)) + jnp.abs(_fdiff(y, -1))
    k = max(1, int(blur))
    if k > 1:
        g = jax.lax.reduce_window(
            g, 0.0, jax.lax.add, (1, k, k), (1, 1, 1), "SAME"
        ) / float(k * k)
    g = g / (jnp.mean(g, axis=(-2, -1), keepdims=True) + 1e-8)
    return 1.0 + float(alpha) * g


def decoder_loss_fn(recipe: DecoderRecipe) -> Callable:
    """``loss(decoder, z_batch, x_batch) -> scalar`` for the recipe's objective.

    ⚠ ``"mse"`` is the shipped uniform pixel MSE and is bit-identical to it.
    ``"object_weighted"`` exists because uniform pixel MSE is **never asked to
    preserve silhouette identity**: the object is a small pixel fraction, so a
    wrong-but-right-sized blob is nearly free (measured: decode MSE 0.0027 on
    the held-out corner vs 0.0031 on seen data — the held-out corner is *not*
    harder to reconstruct, only harder to render *identifiably*).
    """
    if recipe.loss == "mse":

        def loss_fn(dec, zb, xb):
            return jnp.mean((jax.vmap(dec)(zb) - xb) ** 2)

        return loss_fn
    if recipe.loss == "object_weighted":
        alpha, blur, lam = recipe.edge_weight, recipe.edge_blur, recipe.grad_weight

        def loss_fn(dec, zb, xb):
            recon = jax.vmap(dec)(zb)
            w = jax.lax.stop_gradient(edge_weight_map(xb, alpha, blur))  # (B, H, W)
            se = jnp.sum((recon - xb) ** 2, axis=1)  # sum over channels
            num = jnp.sum(w * se, axis=(-2, -1))
            den = float(xb.shape[1]) * jnp.sum(w, axis=(-2, -1))
            l_w = jnp.mean(num / den)
            l_g = jnp.mean((_fdiff(recon, -2) - _fdiff(xb, -2)) ** 2) + jnp.mean(
                (_fdiff(recon, -1) - _fdiff(xb, -1)) ** 2
            )
            return l_w + lam * l_g

        return loss_fn
    raise ValueError(f"unknown dec_loss {recipe.loss!r} (have 'mse', 'object_weighted')")


def decoder_schedule(recipe: DecoderRecipe):
    """Optax LR schedule for the recipe, or ``None`` for the shipped constant LR."""
    if recipe.lr_schedule == "constant":
        return None
    if recipe.lr_schedule == "cosine":
        warm = int(max(0, round(recipe.warmup_frac * recipe.steps)))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=recipe.lr,
            warmup_steps=warm,
            decay_steps=max(1, recipe.steps),
            end_value=0.0,
        )
    raise ValueError(
        f"unknown dec_lr_schedule {recipe.lr_schedule!r} (have 'constant', 'cosine')"
    )


def train_decoder_from_features(
    feat_shape,
    features: np.ndarray,
    images: np.ndarray,
    cfg,
    key,
    log: Optional[Callable] = None,
    recipe: Optional[str] = None,
):
    """⭐ The recipe applied to **any** feature cache — the refit entry point.

    A rung whose encoder was retrained hands its own ``(features, images)`` here
    and gets a decoder fit by the same recipe, to be scored against its own
    ceiling.  ⛔ The encoder is not touched and no cache is rewritten.
    """
    rec = resolve_decoder_cfg(cfg, recipe)
    in_shape = tuple(images.shape[1:])
    key, dk = jax.random.split(key)
    decoder = make_decoder(feat_shape, in_shape, rec, dk)
    dtype = model_dtype(decoder)
    Z = jnp.asarray(features, dtype)
    X = jnp.asarray(images, dtype)
    decoder, history, key = _sgd_loop(
        decoder,
        decoder_loss_fn(rec),
        (Z, X),
        rec.steps,
        rec.batch,
        rec.lr,
        key,
        log,
        schedule=decoder_schedule(rec),
    )
    return decoder, history, rec


def train_decoder(encoder, images: np.ndarray, cfg, key, log: Optional[Callable] = None):
    """⭐ THE shared decoder: real ``(feature, image)`` pairs, train split only.

    The encoder is **frozen** here (features are precomputed and detached), so
    the renderer cannot co-adapt with it and is exactly the map every arm uses.
    """
    feats = encode_images(encoder, images, int(cfg.eval_batch))
    decoder, history, _ = train_decoder_from_features(
        encoder.feat_shape, feats, images, cfg, key, log=log
    )
    return decoder, history


def train_probe(
    images: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    cfg,
    key,
    log: Optional[Callable] = None,
):
    """Fit the per-factor probe on **real** images + factor labels (the R1 exemption)."""
    in_shape = tuple(images.shape[1:])
    key, pk = jax.random.split(key)
    probe = FactorProbe(
        in_shape, tuple(cfg.probe_channels), int(cfg.probe_groups), cardinalities, pk
    )
    dtype = model_dtype(probe)
    X = jnp.asarray(images, dtype)
    Y = jnp.asarray(labels, jnp.int32)

    def loss_fn(m, xb, yb):
        return _xent(jax.vmap(m)(xb), yb)

    probe, history, key = _sgd_loop(
        probe, loss_fn, (X, Y), cfg.probe_steps, cfg.probe_batch, cfg.probe_lr, key, log
    )
    return probe, history


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


def encode_images(encoder, images: np.ndarray, batch: int = 512) -> np.ndarray:
    """Frozen forward pass -> ``(N, C, H, W)`` float32 penultimate maps."""
    dtype = model_dtype(encoder)
    fwd = eqx.filter_jit(jax.vmap(encoder))
    out = []
    n = int(images.shape[0])
    for i in range(0, n, batch):
        chunk = jnp.asarray(images[i : i + batch], dtype)
        out.append(np.asarray(fwd(chunk), dtype=np.float32))
    if not out:
        return np.zeros((0,) + encoder.feat_shape, np.float32)
    return np.concatenate(out)


def build_feature_cache(
    encoder,
    dataset: str,
    rows: np.ndarray,
    out_dir: str | Path,
    cfg,
    root: str | Path | None = None,
    scope: str = "full",
    log: Optional[Callable] = None,
) -> dict:
    """Encode ``rows`` once and write ``features.npy`` + ``feature_cache.json``.

    Everything downstream reads this cache; ⛔ it is never recomputed per arm.
    Written in blocks so neither the images nor the features are ever fully in
    RAM at once.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    dtype = np.dtype(cfg.cache_dtype)
    feat_shape = encoder.feat_shape
    feat_path = out_dir / "features.npy"
    rows_path = out_dir / "feature_rows.npy"

    arr = np.lib.format.open_memmap(
        feat_path, mode="w+", dtype=dtype, shape=(rows_sorted.size,) + feat_shape
    )
    mod = dataset_module(dataset)
    block = int(cfg.cache_block)
    handle = mod.open_images(root)
    try:
        for i in range(0, rows_sorted.size, block):
            chunk_rows = rows_sorted[i : i + block]
            imgs = stream_images(dataset, chunk_rows, root, block=block, handle=handle)
            arr[i : i + chunk_rows.size] = encode_images(
                encoder, imgs, int(cfg.eval_batch)
            ).astype(dtype)
            if log is not None:
                log(f"    cached {min(i + block, rows_sorted.size)}/{rows_sorted.size}")
    finally:
        handle.close()
    arr.flush()
    del arr
    np.save(rows_path, rows_sorted)

    manifest = {
        "schema": "chlu-gen-feature-cache/1",
        "dataset": dataset,
        "scope": scope,
        "n_rows": int(rows_sorted.size),
        "feat_shape": [int(x) for x in feat_shape],
        "D": int(np.prod(feat_shape)),
        "dtype": str(dtype),
        "encoder_param_hash": param_hash(encoder),
        "features_file": feat_path.name,
        "features_sha256": file_sha256(feat_path),
        "features_bytes": int(feat_path.stat().st_size),
        "rows_file": rows_path.name,
        "rows_sha256": file_sha256(rows_path),
    }
    with open(out_dir / "feature_cache.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def load_feature_cache(out_dir: str | Path, verify: bool = True):
    """Return ``(rows, features_memmap, manifest)`` for a cache written above."""
    out_dir = Path(out_dir)
    with open(out_dir / "feature_cache.json") as fh:
        manifest = json.load(fh)
    feat_path = out_dir / manifest["features_file"]
    if verify:
        got = file_sha256(feat_path)
        if got != manifest["features_sha256"]:
            raise AssertionError(
                f"feature cache checksum mismatch at {feat_path}: "
                f"{got} != {manifest['features_sha256']}"
            )
    rows = np.load(out_dir / manifest["rows_file"])
    features = np.load(feat_path, mmap_mode="r")
    return rows, features, manifest


def features_for_rows(cache_rows: np.ndarray, features, rows: np.ndarray) -> np.ndarray:
    """Gather cached features for ``rows`` (raises if any row is not cached)."""
    rows = np.asarray(rows, dtype=np.int64)
    pos = np.searchsorted(cache_rows, rows)
    if np.any(pos >= cache_rows.size) or np.any(cache_rows[np.minimum(pos, cache_rows.size - 1)] != rows):
        missing = rows[
            (pos >= cache_rows.size)
            | (cache_rows[np.minimum(pos, cache_rows.size - 1)] != rows)
        ]
        raise KeyError(
            f"{missing.size} requested row(s) are not in the feature cache "
            f"(first: {int(missing[0])}) — the cache scope is too narrow"
        )
    return np.asarray(features[pos], dtype=np.float32)


# ---------------------------------------------------------------------------
# R3.3 — the no-leak assertion
# ---------------------------------------------------------------------------


def assert_no_holdout_leak(split: GenSplit, used_rows: np.ndarray) -> dict:
    """⭐ R3.3: assert the decoder's training rows contain **no** held-out combination.

    Re-evaluates the dataset's own test predicate on factor values reconstructed
    from each used row through the grid bijection — it does *not* re-read the
    mask that produced the split, so a corrupted training index cannot smuggle a
    held-out combination past it.  Raises ``HoldoutViolation`` naming the row.

    Returns a dict with the number of rows checked and the number of images
    **excluded** from training because they are held-out combinations.
    """
    used = np.asarray(used_rows, dtype=np.int64)
    n_checked = assert_holdout_disjoint(split.with_train_rows(used))
    excluded = np.setdiff1d(split.test_rows, used, assume_unique=False)
    return {
        "n_train_rows_used": int(used.size),
        "n_checked": int(n_checked),
        "n_holdout_excluded": int(excluded.size),
        "n_holdout_total": int(split.test_rows.size),
        "n_holdout_leaked": int(split.test_rows.size - excluded.size),
    }
