"""⭐ Candidate self-supervised **reconstruction** encoders for the generative phase.

⛔ **What this module is not.**  It does not touch
:mod:`chlu.core.gen_bridge` — the shipped ``pretext="recon"`` path is imported
and called, never edited, so the frozen feature cache stays bit-identical.  The
arms here produce a **candidate** encoder BESIDE the frozen one; promoting one is
a separate, Head-level decision that re-measures every phase-1 number.

**Why it exists (R40.E).**  ``gen-navigation`` measured, on the *spread* grid and
with an independent instrument, that the read renders the right ``scale`` only
0.229 of the time (chance 0.125) and the right ``orientation`` 0.236 (chance
0.067).  R40.A.2's pre-registered reading splits that into two worlds: if the
**encoder features** already carry those factors, the store drops them (world A);
if they do not, the store is innocent (world B).  This module supplies the arms
for the second half of that question — *can a reconstruction pretext be made to
keep ``scale`` and ``orientation`` at all?*

⛔ **Augmentation-contrastive is excluded and the reason is not stylistic:** it
trains invariance to exactly ``scale`` and ``orientation``, the two factors we
are trying to keep.

**The arms.**  ``recon_control`` is the laundering control and is the *shipped*
code path verbatim.  ``ae_plain`` is the same objective run through this module's
loop — a plumbing null whose gap to ``recon_control`` is the batch-sequence
nuisance and nothing else.  ``recon_control_2x`` is the **compute** control: a
variant that only wins by training longer is not a representation win.
``ae_deep`` / ``mae`` / ``ae_hires`` are the AE/MAE headline family and ``vae``
is the declared comparison arm, reported beside the headline, never instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .gen_bridge import (
    FactorHeads,
    GenDecoder,
    GenEncoder,
    model_dtype,
    param_count,
    param_hash,
    train_encoder,
)

#: Every arm this module knows how to build.  ⭐ ``recon_control`` first: it is the
#: control row that appears on every table, not an afterthought.
PRETRAIN_VARIANTS = (
    "recon_control",
    "random_init",
    "ae_plain",
    "recon_control_2x",
    "ae_deep",
    "mae",
    "ae_hires",
    "vae",
)

#: Arms that are the *shipped* encoder path (``gen_bridge.train_encoder``) and must
#: stay bit-identical to it for a given key/config.
SHIPPED_PATH_VARIANTS = ("recon_control", "recon_control_2x")

#: Probe families.  ⭐ ``linear`` is the headline instrument (the same head family
#: as the bridge's ``cls`` diagnostic); ``mlp`` is a capacity check reported beside
#: it — ⛔ a linear FAIL with an MLP PASS is "present but not linearly readable",
#: which is a different finding and never counts as clearing the bar.
PROBE_KINDS = ("linear", "mlp")


class VariantError(ValueError):
    """Unknown or mis-specified pretraining variant."""


# ---------------------------------------------------------------------------
# The variant table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    """Everything that distinguishes one arm from the control, and nothing else.

    ⛔ ``steps_mult`` is 1 for every arm except the **declared** compute control:
    a representation claim may not be bought with extra optimisation steps.
    """

    name: str
    family: str  # "recon" (shipped path) | "ae" | "mae" | "vae"
    enc_channels: tuple
    dec_channels: tuple
    dec_blocks_per_stage: int = 1
    steps_mult: int = 1
    mask_ratio: float = 0.0
    mask_patch: int = 8
    loss_on_masked_only: bool = True
    beta: float = 0.0
    note: str = ""
    is_control: bool = False
    is_comparison_arm: bool = False


def _scaled(channels: Sequence[int], mult: float) -> tuple:
    return tuple(max(1, int(round(float(c) * float(mult)))) for c in channels)


def variant_spec(name: str, cfg) -> VariantSpec:
    """Resolve one arm against ``cfg`` (``config.gen_pretrain``)."""
    if name not in PRETRAIN_VARIANTS:
        raise VariantError(f"variant must be one of {PRETRAIN_VARIANTS}, got {name!r}")
    enc = tuple(int(c) for c in cfg.enc_channels)
    dec = tuple(int(c) for c in cfg.dec_channels)
    if name == "random_init":
        # ⭐ The TRIVIAL SUBSTITUTE.  An untrained conv stack already produces a
        # random projection of the image; whatever a probe reads off it is the
        # floor that *any* pretext must beat before "the encoder learned it" is a
        # sentence anyone may write.  0 steps, so ⛔ it can never win on compute.
        return VariantSpec(name, "random", enc, dec, steps_mult=0, is_control=True,
                           note="LAUNDERING FLOOR: random init, zero training steps")
    if name == "recon_control":
        return VariantSpec(name, "recon", enc, dec, is_control=True,
                           note="the shipped pretext='recon' path, called verbatim")
    if name == "recon_control_2x":
        return VariantSpec(name, "recon", enc, dec, steps_mult=int(cfg.compute_control_mult),
                           is_control=True,
                           note="COMPUTE control: the shipped path at N x steps")
    if name == "ae_plain":
        return VariantSpec(name, "ae", enc, dec, is_control=True,
                           note="plumbing null: the control objective through THIS loop")
    if name == "ae_deep":
        return VariantSpec(name, "ae", enc, _scaled(dec, cfg.deep_width_mult),
                           dec_blocks_per_stage=int(cfg.deep_blocks_per_stage),
                           note="reconstruction PRESSURE: wider, 2-blocks-per-stage decoder")
    if name == "mae":
        return VariantSpec(name, "mae", enc, dec, mask_ratio=float(cfg.mae_mask_ratio),
                           mask_patch=int(cfg.mae_patch),
                           loss_on_masked_only=bool(cfg.mae_loss_on_masked),
                           note="masked autoencoding: patch-masked input, loss on masked pixels")
    if name == "ae_hires":
        return VariantSpec(name, "ae", tuple(int(c) for c in cfg.hires_channels), dec,
                           note="one fewer downsample: features stay 8x8 (D=8192)")
    if name == "vae":
        return VariantSpec(name, "vae", enc, dec, beta=float(cfg.vae_beta),
                           is_comparison_arm=True,
                           note="DECLARED COMPARISON ARM: mse + beta*KL, features = mu")
    raise VariantError(name)  # pragma: no cover - exhaustive above


# ---------------------------------------------------------------------------
# The VAE arm's encoder wrapper
# ---------------------------------------------------------------------------


class VAEEncoder(eqx.Module):
    """``x -> mu(conv_trunk(x))``: the deterministic read-time encoder of the VAE arm.

    ⚠ The cached feature is ``mu`` — **pre-activation**, so unlike every other arm
    it is not non-negative.  That is inherent to the arm and is reported, not
    hidden: a probe sees a differently-shaped feature space, which is exactly what
    "a different pretext" means.
    """

    trunk: GenEncoder
    mu: eqx.nn.Conv2d
    logvar: eqx.nn.Conv2d
    in_shape: tuple
    feat_shape: tuple

    def __init__(self, trunk: GenEncoder, key):
        km, kl = jax.random.split(key)
        c = int(trunk.feat_shape[0])
        self.trunk = trunk
        self.mu = eqx.nn.Conv2d(c, c, 1, key=km)
        self.logvar = eqx.nn.Conv2d(c, c, 1, key=kl)
        self.in_shape = tuple(trunk.in_shape)
        self.feat_shape = tuple(trunk.feat_shape)

    def __call__(self, x):
        return self.mu(self.trunk(x))

    def moments(self, x):
        h = self.trunk(x)
        return self.mu(h), self.logvar(h)

    @property
    def feature_dim(self) -> int:
        c, h, w = self.feat_shape
        return int(c * h * w)


# ---------------------------------------------------------------------------
# The loop (key-threaded — MAE needs a fresh mask and the VAE a fresh epsilon)
# ---------------------------------------------------------------------------


def _keyed_sgd_loop(model, loss_fn, data, steps, batch, lr, key,
                    log: Optional[Callable] = None):
    """Minibatch Adam, ``loss_fn(model, key, *batch_arrays) -> scalar``.

    ⭐ Deliberately a *separate* loop from ``gen_bridge._sgd_loop`` rather than an
    edit to it: the shipped loop is on the frozen path and must not move.  The
    cost is that the batch **sequence** differs from the shipped loop's, which is
    why ``ae_plain`` exists — it runs the control's objective here, so the
    nuisance is measured instead of argued about.
    """
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    n = int(data[0].shape[0])
    batch = min(int(batch), n)
    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(m, s, k, *arrays):
        loss, grads = grad_fn(m, k, *arrays)
        updates, s = opt.update(grads, s, m)
        return eqx.apply_updates(m, updates), s, loss

    history = []
    for i in range(int(steps)):
        key, sk, lk = jax.random.split(key, 3)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        model, opt_state, loss = step(model, opt_state, lk, *[a[idx] for a in data])
        history.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, int(steps) // 5) == 0):
            log(f"    step {i + 1}/{steps}  loss={history[-1]:.5f}")
    return model, history, key


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def patch_mask(key, shape, patch: int, ratio: float, dtype=jnp.float32):
    """``(B, 1, H, W)`` Bernoulli **patch** mask, 1 = masked.

    ⚠ Bernoulli, not top-k: the masked *fraction* is ``ratio`` in expectation and
    varies per image.  Stated rather than hidden — a fixed-count mask would be a
    different (and slightly stronger) pretext.
    """
    b, _, h, w = shape
    p = max(1, int(patch))
    ph, pw = (h + p - 1) // p, (w + p - 1) // p
    m = jax.random.bernoulli(key, float(ratio), (b, 1, ph, pw)).astype(dtype)
    m = jnp.repeat(jnp.repeat(m, p, axis=2), p, axis=3)
    return m[:, :, :h, :w]


def masked_mse(recon, x, mask, on_masked_only: bool = True):
    """MSE restricted to the masked pixels (MAE) or over all of them."""
    se = (recon - x) ** 2
    if not on_masked_only:
        return jnp.mean(se)
    w = jnp.broadcast_to(mask, se.shape)
    return jnp.sum(se * w) / (jnp.sum(w) + 1e-8)


def gaussian_kl(mu, logvar):
    """Mean-per-element KL of ``N(mu, exp(logvar))`` from ``N(0, 1)``."""
    return jnp.mean(0.5 * (jnp.exp(logvar) + mu**2 - 1.0 - logvar))


# ---------------------------------------------------------------------------
# Training one arm
# ---------------------------------------------------------------------------


def train_variant(
    images: np.ndarray,
    variant: str,
    cfg,
    key,
    bridge_cfg=None,
    log: Optional[Callable] = None,
):
    """Fit one arm on train-split images only.  Returns ``(encoder, aux, history, meta)``.

    ``encoder`` is whatever a frozen read would call: ``GenEncoder`` for every arm
    except ``vae``, where it is the deterministic ``VAEEncoder`` (``mu``).  The
    pretext decoder/head is **discarded** exactly as on the shipped path; its hash
    is kept only so the pretext is auditable.
    """
    spec = variant_spec(variant, cfg)
    steps = int(cfg.enc_steps) * int(spec.steps_mult)
    in_shape = tuple(int(x) for x in images.shape[1:])

    if spec.family == "random":
        key, ek, ak = jax.random.split(key, 3)  # same split arity as train_encoder
        encoder = GenEncoder(in_shape, spec.enc_channels, int(cfg.enc_kernel),
                             int(cfg.enc_stride), int(cfg.enc_groups), ek)
        aux = GenDecoder(encoder.feat_shape, in_shape, spec.dec_channels,
                         int(cfg.dec_groups), ak, spec.dec_blocks_per_stage)
        meta = {"objective": "NONE (untrained)", "path": "gen_encoders.train_variant"}
        return encoder, aux, [], _meta(spec, encoder, aux, [], 0, meta, cfg)

    if spec.family == "recon":
        # ⭐ THE SHIPPED PATH, called verbatim.  ⛔ Do not reimplement it here: the
        # control row's only job is to be the thing already in the repo.
        if bridge_cfg is None:
            raise VariantError("the shipped-path arms need the gen_bridge config")
        # ⛔ The control row's whole job is to be a like-for-like baseline.  The
        # shipped path reads its architecture from ``GenBridgeConfig``, so if the
        # two configs have drifted apart the "control" is silently a different
        # encoder — fail loudly rather than ship a mismatched table.
        mismatch = {
            k: (a, b)
            for k, a, b in (
                ("enc_channels", tuple(spec.enc_channels), tuple(int(c) for c in bridge_cfg.enc_channels)),
                ("enc_kernel", int(cfg.enc_kernel), int(bridge_cfg.enc_kernel)),
                ("enc_stride", int(cfg.enc_stride), int(bridge_cfg.enc_stride)),
                ("enc_groups", int(cfg.enc_groups), int(bridge_cfg.enc_groups)),
                ("enc_batch", int(cfg.enc_batch), int(bridge_cfg.enc_batch)),
                ("enc_lr", float(cfg.enc_lr), float(bridge_cfg.enc_lr)),
            )
            if a != b
        }
        if mismatch:
            raise VariantError(
                f"⛔ the {variant!r} control would not be like-for-like: "
                f"gen_pretrain vs gen_bridge differ on {mismatch}"
            )
        saved = int(bridge_cfg.enc_steps)
        try:
            bridge_cfg.enc_steps = steps
            encoder, aux, hist = train_encoder(images, "recon", bridge_cfg, key, log=log)
        finally:
            bridge_cfg.enc_steps = saved
        meta = {"objective": "mean((dec(enc(x)) - x)**2)", "path": "gen_bridge.train_encoder"}
        return encoder, aux, hist, _meta(spec, encoder, aux, hist, steps, meta, cfg)

    key, ek, ak = jax.random.split(key, 3)  # same split arity as train_encoder
    trunk = GenEncoder(in_shape, spec.enc_channels, int(cfg.enc_kernel),
                       int(cfg.enc_stride), int(cfg.enc_groups), ek)
    dtype = model_dtype(trunk)
    X = jnp.asarray(images, dtype)

    if spec.family == "vae":
        key, vk = jax.random.split(key)
        encoder = VAEEncoder(trunk, vk)
        dec = GenDecoder(encoder.feat_shape, in_shape, spec.dec_channels,
                         int(cfg.dec_groups), ak, spec.dec_blocks_per_stage)
        beta = float(spec.beta)

        def loss_fn(m, k, xb):
            enc, d = m
            mu, logvar = jax.vmap(enc.moments)(xb)
            eps = jax.random.normal(k, mu.shape, mu.dtype)
            z = mu + jnp.exp(0.5 * logvar) * eps
            recon = jax.vmap(d)(z)
            return jnp.mean((recon - xb) ** 2) + beta * gaussian_kl(mu, logvar)

        obj = f"mean((dec(z)-x)**2) + {beta} * KL,  z = mu + sigma*eps"
    else:
        encoder = trunk
        dec = GenDecoder(encoder.feat_shape, in_shape, spec.dec_channels,
                         int(cfg.dec_groups), ak, spec.dec_blocks_per_stage)
        if spec.family == "mae":
            ratio, patch = float(spec.mask_ratio), int(spec.mask_patch)
            only = bool(spec.loss_on_masked_only)

            def loss_fn(m, k, xb):
                enc, d = m
                mask = patch_mask(k, xb.shape, patch, ratio, xb.dtype)
                recon = jax.vmap(lambda x: d(enc(x)))(xb * (1.0 - mask))
                return masked_mse(recon, xb, mask, only)

            obj = (f"masked MSE, patch={patch}, ratio={ratio}, "
                   f"{'masked pixels only' if only else 'all pixels'}")
        else:

            def loss_fn(m, k, xb):
                enc, d = m
                recon = jax.vmap(lambda x: d(enc(x)))(xb)
                return jnp.mean((recon - xb) ** 2)

            obj = "mean((dec(enc(x)) - x)**2)"

    model, hist, key = _keyed_sgd_loop(
        (encoder, dec), loss_fn, (X,), steps, int(cfg.enc_batch), float(cfg.enc_lr), key, log
    )
    encoder, aux = model
    return encoder, aux, hist, _meta(spec, encoder, aux, hist, steps,
                                     {"objective": obj, "path": "gen_encoders.train_variant"}, cfg)


def _meta(spec: VariantSpec, encoder, aux, hist, steps, extra: dict, cfg) -> dict:
    return {
        "variant": spec.name,
        "family": spec.family,
        "note": spec.note,
        "is_control": bool(spec.is_control),
        "is_comparison_arm": bool(spec.is_comparison_arm),
        "enc_channels": list(spec.enc_channels),
        "dec_channels": list(spec.dec_channels),
        "dec_blocks_per_stage": int(spec.dec_blocks_per_stage),
        "steps": int(steps),
        "steps_mult": int(spec.steps_mult),
        "batch": int(cfg.enc_batch),
        "lr": float(cfg.enc_lr),
        "encoder_param_hash": param_hash(encoder),
        "encoder_param_count": param_count(encoder),
        "pretext_head_param_hash": param_hash(aux),
        "pretext_head_param_count": param_count(aux),
        "pretext_head_discarded": True,
        "feat_shape": [int(x) for x in encoder.feat_shape],
        "D": int(encoder.feature_dim),
        "loss_first": float(hist[0]) if hist else None,
        "loss_final": float(hist[-1]) if hist else None,
        "loss_final_mean_last_50": float(np.mean(hist[-50:])) if hist else None,
        **extra,
    }


# ---------------------------------------------------------------------------
# ⭐ The probe — the SAME head family and the SAME scoring code as the phase
# ---------------------------------------------------------------------------


class FeatureProbe(eqx.Module):
    """Per-factor classifier over a **flattened feature vector**.

    ``linear``: :class:`~chlu.core.gen_bridge.FactorHeads` directly — one
    ``eqx.nn.Linear`` per factor, i.e. the same head family as the bridge's ``cls``
    diagnostic and the external classifier's output stage.  ``mlp``: one SiLU
    hidden layer first, as a capacity check.

    ⛔ Standardisation is **not** a field here.  z-scoring lives outside the module
    (:func:`standardize`) so the statistics cannot be dragged into the optimiser by
    ``eqx.partition(..., is_array)``.
    """

    hidden: Optional[eqx.nn.Linear]
    heads: FactorHeads

    def __init__(self, in_dim, cardinalities, key, hidden: int = 0):
        kh, kf = jax.random.split(key)
        if int(hidden) > 0:
            self.hidden = eqx.nn.Linear(int(in_dim), int(hidden), key=kh)
            self.heads = FactorHeads(int(hidden), cardinalities, kf)
        else:
            self.hidden = None
            self.heads = FactorHeads(int(in_dim), cardinalities, kf)

    def __call__(self, z):
        w = (self.hidden.weight if self.hidden is not None else self.heads.linears[0].weight)
        flat = jnp.asarray(z, w.dtype).reshape(-1)
        if self.hidden is not None:
            flat = jax.nn.silu(self.hidden(flat))
        return self.heads(flat)


def standardize_stats(features: np.ndarray, eps: float = 1e-6) -> dict:
    """Per-dimension mean/std of the **probe-fit** rows only (flattened)."""
    F = np.asarray(features, np.float32).reshape(int(features.shape[0]), -1)
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    return {"mean": mean, "std": np.maximum(std, float(eps))}


def standardize(features: np.ndarray, stats: dict) -> np.ndarray:
    """Apply fit-row statistics.  ⛔ Never re-fit on the evaluation rows."""
    F = np.asarray(features, np.float32).reshape(int(features.shape[0]), -1)
    return ((F - stats["mean"]) / stats["std"]).astype(np.float32)


def fit_feature_probe(features: np.ndarray, labels: np.ndarray, cardinalities,
                      cfg, key, kind: str = "linear", log: Optional[Callable] = None):
    """Fit one probe on standardised features.  Identical for every arm and seed."""
    if kind not in PROBE_KINDS:
        raise VariantError(f"probe kind must be one of {PROBE_KINDS}, got {kind!r}")
    key, pk = jax.random.split(key)
    hidden = int(cfg.probe_hidden) if kind == "mlp" else 0
    probe = FeatureProbe(int(features.shape[1]), cardinalities, pk, hidden)
    dtype = model_dtype(probe)
    X = jnp.asarray(features, dtype)
    Y = jnp.asarray(labels, jnp.int32)

    from .gen_bridge import _sgd_loop, _xent  # the shipped loop + the shipped loss

    def loss_fn(m, xb, yb):
        return _xent(jax.vmap(m)(xb), yb)

    probe, history, _ = _sgd_loop(
        probe, loss_fn, (X, Y), int(cfg.probe_steps), int(cfg.probe_batch),
        float(cfg.probe_lr), key, log
    )
    return probe, history


# ---------------------------------------------------------------------------
# Reporting helpers — ⚠ trap 1: a per-factor statistic is meaningless without
# ``n_values_realised``, so it is computed here and travels with every number.
# ---------------------------------------------------------------------------


def realised_values(labels: np.ndarray, factor_names) -> dict:
    """``{factor: {n_values_realised, chance, majority_rate, values}}`` for a pool."""
    Y = np.asarray(labels, np.int64)
    out = {}
    for j, name in enumerate(factor_names):
        vals, counts = np.unique(Y[:, j], return_counts=True)
        n = int(vals.size)
        out[str(name)] = {
            "n_values_realised": n,
            "chance": float(1.0 / n) if n else float("nan"),
            "majority_rate": float(counts.max() / counts.sum()) if n else float("nan"),
            "values": [int(v) for v in vals],
        }
    return out


def bar_verdict(per_factor: dict, factors: Sequence[str], bar: float) -> dict:
    """⭐ The pre-registered bar: **every** named factor must reach ``bar``."""
    got = {str(f): float(per_factor[str(f)]) for f in factors}
    passed = bool(got) and all(v >= float(bar) for v in got.values())
    return {
        "bar": float(bar),
        "factors": [str(f) for f in factors],
        "values": got,
        "verdict": "PASS" if passed else "FAIL",
        "margin": {k: float(v - float(bar)) for k, v in got.items()},
    }
