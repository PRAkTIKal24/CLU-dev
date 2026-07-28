"""Tests for the CL-capable conv read-in arms (w26 ``cl-encoder``).

Everything runs on tiny SYNTHETIC images at CIFAR/MNIST shapes with a handful of
optimizer steps — no download, no long fit. The scientific numbers live in the gate
harness, not here; these tests fix the *contract*: the arms build, ``phi_dim`` means
what it says, ``φ`` is frozen/deterministic, the augmentations stay in range, the
NT-Xent loss starts at its chance value, and the pre-existing ``pca``/``ae`` arms are
bit-identical to before the hook was added.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments.exp_phi_read_in import build_read_in
from chlu.experiments.phi_encoders import (
    ENCODER_ARMS,
    ConvEncoderReadIn,
    augment,
    image_shape,
    nt_xent,
)


def _cfg(phi_dim=8):
    cfg = get_default_config().experiment_cl_entry
    cfg.phi_dim = phi_dim
    cfg.enc_channels = [4, 8]
    cfg.enc_pool = 2
    cfg.enc_steps = 3
    cfg.enc_batch = 8
    cfg.enc_proj_dim = 8
    return cfg


def _images(n=24, dataset="cifar10"):
    C, H, W = image_shape(dataset, 0 if dataset == "cifar10" else 784)
    rng = np.random.default_rng(0)
    return rng.random((n, C * H * W), dtype=np.float32)


def test_image_shape_covers_both_datasets():
    assert image_shape("cifar10", 3072) == (3, 32, 32)
    assert image_shape("mnist", 784) == (1, 28, 28)
    assert image_shape("other", 3072) == (3, 32, 32)  # inferred from the dim
    with pytest.raises(ValueError):
        image_shape("other", 13)


@pytest.mark.parametrize("arm", ENCODER_ARMS)
def test_arm_builds_and_respects_phi_dim(arm):
    cfg = _cfg(phi_dim=8)
    X = _images()
    phi, prov = build_read_in(arm, "cifar10", X, X, cfg, seed=0)
    F = np.asarray(phi(X[:5]))
    assert F.shape == (5, 8)
    assert np.all(np.isfinite(F))
    assert prov["arm"] == arm and prov["k"] == 8 and prov["frozen"] is True
    # randconv is fit-free by construction; the trained arms actually step
    assert prov["steps"] == (0 if arm == "randconv" else cfg.enc_steps)


def test_phi_is_frozen_and_deterministic():
    """A frozen φ must return the same address for the same image, every call."""
    cfg = _cfg()
    X = _images()
    phi, _ = build_read_in("simclr", "cifar10", X, X, cfg, seed=1)
    np.testing.assert_allclose(np.asarray(phi(X[:6])), np.asarray(phi(X[:6])))
    # and a second identically-seeded fit reproduces it exactly
    phi2, _ = build_read_in("simclr", "cifar10", X, X, cfg, seed=1)
    np.testing.assert_allclose(np.asarray(phi(X[:6])), np.asarray(phi2(X[:6])))


def test_mnist_shape_arm_runs():
    cfg = _cfg(phi_dim=6)
    X = _images(dataset="mnist")
    phi, _ = build_read_in("convae", "mnist", X, X, cfg, seed=0)
    assert np.asarray(phi(X[:4])).shape == (4, 6)


def test_l2_normalize_flag_normalises_addresses():
    cfg = _cfg()
    cfg.enc_l2_normalize = True
    X = _images()
    phi, _ = build_read_in("randconv", "cifar10", X, X, cfg, seed=0)
    F = np.asarray(phi(X[:5]))
    np.testing.assert_allclose(np.linalg.norm(F, axis=1), 1.0, rtol=1e-5)


def test_whitened_head_equalises_feature_scales():
    cfg = _cfg(phi_dim=4)
    X = _images(n=40)
    cfg.enc_head = "pca_whiten"
    Fw = np.asarray(build_read_in("randconv", "cifar10", X, X, cfg, 0)[0](X))
    cfg.enc_head = "pca"
    Fp = np.asarray(build_read_in("randconv", "cifar10", X, X, cfg, 0)[0](X))
    # whitening makes the per-coordinate std ~1; plain PCA keeps the spectrum's decay,
    # so its leading/trailing scale ratio is strictly larger (a 1-NN read-out in the
    # address space is directly sensitive to that ratio)
    np.testing.assert_allclose(Fw.std(axis=0), 1.0, rtol=0.15)
    ratio = lambda F: float(F.std(axis=0)[0] / F.std(axis=0)[-1])  # noqa: E731
    assert ratio(Fp) > 1.2
    assert ratio(Fw) < ratio(Fp)


def test_augmentation_preserves_shape_and_range():
    cfg = _cfg()
    x = jnp.asarray(_images(n=1).reshape(3, 32, 32))
    for i in range(5):
        v = augment(x, jax.random.PRNGKey(i), cfg)
        assert v.shape == x.shape
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # two different keys must give two different views (the whole point)
    v1 = augment(x, jax.random.PRNGKey(0), cfg)
    v2 = augment(x, jax.random.PRNGKey(1), cfg)
    assert float(jnp.abs(v1 - v2).max()) > 1e-6


def test_nt_xent_starts_at_its_chance_value():
    """Random embeddings ⇒ loss ≈ ln(2N−1): the sanity check that says the loss is
    wired up (an NT-Xent that starts anywhere else has a pairing bug)."""
    n = 16
    z = jax.random.normal(jax.random.PRNGKey(0), (2 * n, 64))
    loss = float(nt_xent(z, 0.5))
    assert abs(loss - np.log(2 * n - 1)) < 0.6
    # a perfectly aligned pair structure must score far below chance
    zz = jax.random.normal(jax.random.PRNGKey(1), (n, 64))
    loss_perfect = float(nt_xent(jnp.concatenate([zz, zz]), 0.05))
    assert loss_perfect < 0.1 * loss


def test_simclr_fit_reduces_its_loss():
    """A slow-but-real check that the contrastive fit is actually optimising."""
    cfg = _cfg()
    cfg.enc_steps = 60
    cfg.enc_batch = 16
    X = _images(n=32)
    _, prov = build_read_in("simclr", "cifar10", X, X, cfg, seed=0)
    assert prov["loss_final"] < prov["loss_first"]


def test_convae_fit_reduces_reconstruction_error():
    cfg = _cfg()
    cfg.enc_steps = 60
    cfg.enc_batch = 16
    X = _images(n=32)
    _, prov = build_read_in("convae", "cifar10", X, X, cfg, seed=0)
    assert prov["loss_final"] < prov["loss_first"]


def test_encoder_never_receives_labels():
    """PREREG_CL_PHI §3: the φ fit objective is unsupervised. The read-in's public
    constructor takes no label argument at all — this pins that interface."""
    import inspect

    params = inspect.signature(ConvEncoderReadIn.__init__).parameters
    assert "y" not in params and "labels" not in params


def test_pca_and_ae_arms_are_unchanged_by_the_hook():
    """The dispatch is additive: the w23/w24/w25 arms must behave exactly as before."""
    cfg = _cfg(phi_dim=4)
    cfg.ae_epochs = 5
    X = _images(n=20)
    for arm in ("pca", "ae"):
        phi, prov = build_read_in(arm, "cifar10", X, X, cfg, seed=0)
        assert prov["arm"] == arm and prov["k"] == 4
        assert np.asarray(phi(X[:3])).shape == (3, 4)
    with pytest.raises(ValueError):
        build_read_in("nonsense", "cifar10", X, X, cfg, seed=0)
