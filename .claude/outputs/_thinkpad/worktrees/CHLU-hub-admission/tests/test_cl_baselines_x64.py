"""⭐ **Item 3a — `cl_baselines.ConvNet` under `jax_enable_x64`.**

**The gap this closes.** ``ConvNet`` builds its conv weights at the **ambient**
JAX dtype, while ``build_cl_stream`` always supplies **float32** images. Under
``jax_enable_x64`` (which several repo test modules turn on **at module import**,
so it is globally ON in a full-suite run) ``lax.conv_general_dilated`` — which is
dtype-**strict** — raised

    lax.conv_general_dilated requires arguments to have the same dtypes,
    got float32, float64.

⇒ the whole ``backbone="cnn"`` path was **untestable inside the full suite**, on a
path that produces potentially quotable numbers. No shipped result is affected
(every real run is x64-off), but this is the **§7.23 / N211 hazard class** and it
had bitten three times before this one. The fix promotes the input to the
parameter dtype; **at x64-off that is a no-op**, and this file asserts that
**bit-identically** against the pre-fix arithmetic rather than merely asserting
"it still runs".

⚠ x64 is toggled by a **function-scoped fixture that restores the previous
value** — it never leaks into another module (N211's own remedy).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.experiments.cl_baselines import ConvNet, _train_task, make_net


@pytest.fixture
def x64_on():
    """x64 ON for one test, restored afterwards (never module-scoped: N211)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def x64_off():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


class _Cfg:
    """The three `cl_baselines` fields the CNN path reads, at toy size."""

    backbone = "cnn"
    mlp_width = 8
    mlp_depth = 1
    cnn_channels = [4, 4, 4]
    baseline_iters = 2
    baseline_batch = 4
    baseline_lr = 1e-3


def _net(key=0, channels=(4, 4, 4), width=8):
    return ConvNet((3, 32, 32), 4, width, jax.random.PRNGKey(key), channels=channels)


def _images(n, seed=0):
    """Exactly what `build_cl_stream` hands the baselines: **float32**, flat."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3 * 32 * 32)).astype(np.float32)


def _prefix_features(model, x):
    """``ConvNet.features`` **as it was before the fix** — the bit-identity oracle."""
    h = x.reshape(model.shape)
    for conv in model.conv:
        h = jax.nn.relu(conv(h))
        h = eqx.nn.MaxPool2d(2, 2)(h)
    return jax.nn.relu(model.lin(h.reshape(-1)))


# ---------------------------------------------------------------------------
# 1. the regression itself: the CNN path runs under x64
# ---------------------------------------------------------------------------
def test_convnet_forward_runs_under_x64_on_float32_images(x64_on):
    """⭐ The bug, pinned: float32 images through float64 conv weights."""
    m = _net()
    assert m.conv[0].weight.dtype == jnp.float64      # weights follow the ambient dtype
    x = jnp.asarray(_images(3))
    assert x.dtype == jnp.float32                     # the stream's dtype, unchanged
    out = jax.vmap(m)(x)
    assert out.shape == (3, 4)
    assert np.all(np.isfinite(np.asarray(out)))
    assert out.dtype == jnp.float64                   # promoted to the parameter dtype


def test_make_net_cnn_backbone_trains_one_step_under_x64(x64_on):
    """The path end-to-end (forward + grads + adam), not just the forward."""
    cfg = _Cfg()
    m = make_net(cfg, 3 * 32 * 32, 4, jax.random.PRNGKey(1))
    assert isinstance(m, ConvNet)
    X = _images(8, seed=1)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    mask = np.ones((4,), dtype=bool)
    m2, _, loss, grads = _train_task(m, X, y, mask, cfg, jax.random.PRNGKey(2))
    assert np.isfinite(loss)
    w0, w2 = np.asarray(m.conv[0].weight), np.asarray(m2.conv[0].weight)
    assert not np.array_equal(w0, w2)                 # it actually took a step
    assert np.all(np.isfinite(np.asarray(grads.conv[0].weight)))


# ---------------------------------------------------------------------------
# 2. ⭐ the OFF path is bit-identical to the pre-fix arithmetic
# ---------------------------------------------------------------------------
def test_x64_off_is_bit_identical_to_the_prefix_convnet(x64_off):
    """⭐ K6-style: at x64-off the promotion is a no-op, **bit-for-bit**."""
    m = _net()
    assert m.conv[0].weight.dtype == jnp.float32
    for x in jnp.asarray(_images(4, seed=3)):
        got = np.asarray(m.features(x))
        want = np.asarray(_prefix_features(m, x))
        assert got.dtype == want.dtype == np.float32
        assert np.array_equal(got, want)              # exact, not allclose
        assert np.array_equal(np.asarray(m(x)),
                              np.asarray(m.head(_prefix_features(m, x))))


def test_x64_off_parameter_count_and_shapes_are_unchanged(x64_off):
    """The fix touches no parameter (the C2 'OFF is parameter-count-identical' habit)."""
    m = _net()
    leaves = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))
    assert sum(int(np.asarray(p).size) for p in leaves) == (
        # 3 convs (3->4, 4->4, 4->4, 3x3 + bias) + lin(4*4*4 -> 8) + head(8 -> 4)
        (4 * 3 * 9 + 4) + (4 * 4 * 9 + 4) + (4 * 4 * 9 + 4) + (64 * 8 + 8) + (8 * 4 + 4)
    )


# ---------------------------------------------------------------------------
# 3. the promotion is the parameter's dtype, not a hard-coded one
# ---------------------------------------------------------------------------
def test_a_float64_input_is_not_downcast_under_x64(x64_on):
    """Promotion follows the WEIGHTS, so an already-float64 input is untouched."""
    m = _net()
    x32 = jnp.asarray(_images(1, seed=5)[0])
    x64 = x32.astype(jnp.float64)
    assert np.array_equal(np.asarray(m(x32)), np.asarray(m(x64)))
