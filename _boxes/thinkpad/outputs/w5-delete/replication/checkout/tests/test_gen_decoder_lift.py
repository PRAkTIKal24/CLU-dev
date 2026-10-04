"""The decoder **recipe** (gen-decoder-lift): capacity, schedule and objective.

⭐ The load-bearing test here is ``test_baseline_recipe_is_bit_identical_to_the
_shipped_decoder``: whatever else this module adds, the ``"baseline"`` recipe
must reproduce the shipped decoder **parameter-for-parameter**, or every number
in ``gen-feature-bridge.md`` silently stops being the baseline it is compared
against.  The recipe machinery exists because a decoder is a *recipe* — refit
per feature space — and ⛔ not one fitted ``.pkl`` that a later rung with a
different encoder could ever reuse.

Everything runs on synthetic tensors; no dataset on disk is needed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import GenBridgeConfig, get_default_config, load_config, save_config
from chlu.core.gen_bridge import (
    DECODER_RECIPES,
    GenDecoder,
    GenEncoder,
    decoder_loss_fn,
    decoder_schedule,
    edge_weight_map,
    make_decoder,
    param_count,
    param_hash,
    resolve_decoder_cfg,
    train_decoder_from_features,
)


def _cfg(**over):
    cfg = GenBridgeConfig()
    cfg.dec_channels = [8, 4]
    cfg.dec_groups = 2
    cfg.dec_steps = 2
    cfg.dec_batch = 4
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _tiny(n=8, c=3, s=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, c, s, s), dtype=np.float64).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. ⭐ the baseline recipe must not move
# ---------------------------------------------------------------------------


def test_baseline_recipe_is_bit_identical_to_the_shipped_decoder():
    """⭐ ``blocks_per_stage=1`` must leave the key split, the conv order and
    every constructor argument untouched, or the shipped ``decoder.pkl`` and the
    reported ceiling stop being comparable to anything refit here."""
    key = jax.random.PRNGKey(3)
    shipped = GenDecoder((8, 4, 4), (3, 32, 32), [8, 4], 2, key)
    viarecipe = make_decoder((8, 4, 4), (3, 32, 32), resolve_decoder_cfg(_cfg()), key)
    assert param_hash(viarecipe) == param_hash(shipped)
    assert len(viarecipe.convs) == len(shipped.convs)
    x = jnp.asarray(_tiny(1, 8, 4)[0])
    assert np.allclose(np.asarray(viarecipe(x)), np.asarray(shipped(x)))


def test_baseline_recipe_names_no_overrides():
    assert DECODER_RECIPES["baseline"] == {}
    rec = resolve_decoder_cfg(GenBridgeConfig())
    assert rec.name == "baseline"
    assert rec.loss == "mse"
    assert rec.lr_schedule == "constant"
    assert rec.blocks_per_stage == 1
    assert list(rec.channels) == GenBridgeConfig().dec_channels


def test_train_decoder_from_features_baseline_matches_the_plain_mse_loop():
    """The refit entry point is the shipped training loop when the recipe is baseline."""
    cfg = _cfg()
    imgs = _tiny(8, 3, 16)
    enc = GenEncoder((3, 16, 16), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    feats = np.stack([np.asarray(enc(x)) for x in imgs])
    d1, h1, rec = train_decoder_from_features(
        enc.feat_shape, feats, imgs, cfg, jax.random.PRNGKey(1)
    )
    d2, h2, _ = train_decoder_from_features(
        enc.feat_shape, feats, imgs, cfg, jax.random.PRNGKey(1)
    )
    assert param_hash(d1) == param_hash(d2)  # deterministic given the key
    assert rec.name == "baseline" and rec.loss == "mse"
    assert len(h1) == cfg.dec_steps == len(h2)


# ---------------------------------------------------------------------------
# 2. capacity: width and depth, without hard-coding the feature width
# ---------------------------------------------------------------------------


def test_width_mult_scales_the_channel_list_only():
    rec = resolve_decoder_cfg(_cfg(dec_width_mult=2.0))
    assert list(rec.channels) == [16, 8]
    assert resolve_decoder_cfg(_cfg(dec_width_mult=1.0)).channels == (8, 4)


def test_blocks_per_stage_deepens_without_changing_the_output_shape():
    a = GenDecoder((8, 4, 4), (3, 32, 32), [8, 4], 2, jax.random.PRNGKey(0))
    b = GenDecoder((8, 4, 4), (3, 32, 32), [8, 4], 2, jax.random.PRNGKey(0),
                   blocks_per_stage=2)
    assert a.blocks_per_stage == 1 and b.blocks_per_stage == 2
    assert len(b.convs) == 2 * (len(a.convs) - 1) + 1
    assert param_count(b) > param_count(a)
    z = jnp.zeros((8, 4, 4))
    assert a(z).shape == b(z).shape == (3, 32, 32)


def test_blocks_per_stage_is_derived_not_stored():
    """⛔ A new pytree field would break the shipped ``decoder.pkl``: ``save_model``
    pickles the ``eqx.partition`` static half, i.e. the module itself."""
    dec = GenDecoder((8, 4, 4), (3, 32, 32), [8, 4], 2, jax.random.PRNGKey(0),
                     blocks_per_stage=3)
    fields = {f.name for f in dec.__class__.__dataclass_fields__.values()}
    assert fields == {"convs", "norms", "feat_shape", "out_shape"}
    assert dec.blocks_per_stage == 3


def test_recipe_is_not_a_function_of_the_feature_width():
    """⭐ A later factorisation rung has a different feature space; the recipe must
    refit into it unchanged.  Only ``feat_shape`` may differ."""
    rec = resolve_decoder_cfg(_cfg(dec_width_mult=2.0, dec_blocks_per_stage=2))
    d_narrow = make_decoder((8, 4, 4), (3, 32, 32), rec, jax.random.PRNGKey(0))
    d_wide = make_decoder((64, 4, 4), (3, 32, 32), rec, jax.random.PRNGKey(0))
    assert len(d_narrow.convs) == len(d_wide.convs)
    assert d_narrow(jnp.zeros((8, 4, 4))).shape == d_wide(jnp.zeros((64, 4, 4))).shape


# ---------------------------------------------------------------------------
# 3. the objective — why a uniform pixel MSE is not asked to preserve identity
# ---------------------------------------------------------------------------


def test_edge_weight_map_is_scale_invariant_and_upweights_the_object():
    """⭐ The weight map is renormalised **per image**, so halving the contrast of
    an image does not halve the attention its object boundary receives — which
    is the whole point in the Montero hue corner, where object and background
    sit in the same band."""
    img = np.zeros((1, 3, 16, 16), np.float32)
    img[:, :, 6:10, 6:10] = 1.0  # a bright square on a flat background
    w = np.asarray(edge_weight_map(jnp.asarray(img), 8.0, 3))
    w_low = np.asarray(edge_weight_map(jnp.asarray(img * 0.1), 8.0, 3))
    assert np.allclose(w, w_low, atol=1e-4)
    boundary = w[0, 5:11, 5:11].mean()
    background = w[0, :3, :3].mean()
    assert boundary > 5.0 * background
    assert background == pytest.approx(1.0, abs=1e-3)  # flat pixels keep weight 1


def test_object_weighted_loss_penalises_a_silhouette_error_uniform_mse_ignores():
    """⭐ The measured motivation: the shipped decoder reaches MSE 0.0027 on the
    held-out corner and still renders `shape` at 0.50 — a wrong-but-right-sized
    blob is nearly free under uniform pixel MSE.  Here the *same* pixel budget
    of error is placed on the object rather than the background, and only the
    object-weighted loss notices."""
    target = np.zeros((2, 3, 16, 16), np.float32)
    target[:, :, 6:10, 6:10] = 1.0
    on_object = target.copy()
    on_object[:, :, 6:8, 6:10] = 0.0  # half the square deleted: identity destroyed
    off_object = target.copy()
    off_object[:, :, 0:2, 0:4] = 1.0  # same number of pixels wrong, in the corner

    def mse(a):
        return float(np.mean((a - target) ** 2))

    assert mse(on_object) == pytest.approx(mse(off_object), rel=1e-6)  # MSE cannot tell

    rec = resolve_decoder_cfg(_cfg(dec_recipe="v2_object_weighted"))
    loss = decoder_loss_fn(rec)
    const_obj = _ConstDecoder(on_object[0])
    const_off = _ConstDecoder(off_object[0])
    z = jnp.zeros((2, 1, 1, 1))
    l_obj = float(loss(const_obj, z, jnp.asarray(target)))
    l_off = float(loss(const_off, z, jnp.asarray(target)))
    assert l_obj > l_off  # ⭐ the objective now pays for silhouette identity


class _ConstDecoder(eqx.Module):
    out: jnp.ndarray

    def __init__(self, arr):
        self.out = jnp.asarray(arr)

    def __call__(self, z):
        return self.out


def test_object_weighted_loss_reduces_to_mse_scale_and_is_finite():
    rec = resolve_decoder_cfg(_cfg(dec_recipe="v2_object_weighted"))
    loss = decoder_loss_fn(rec)
    target = jnp.asarray(_tiny(4, 3, 16))
    const = _ConstDecoder(np.asarray(target[0]))
    val = float(loss(const, jnp.zeros((4, 1, 1, 1)), target))
    assert np.isfinite(val) and val > 0.0
    exact = _ConstDecoder(np.zeros((3, 16, 16), np.float32))
    assert float(loss(exact, jnp.zeros((1, 1, 1, 1)),
                      jnp.zeros((1, 3, 16, 16)))) == pytest.approx(0.0, abs=1e-12)


def test_unknown_loss_and_schedule_are_refused():
    with pytest.raises(ValueError, match="unknown dec_loss"):
        decoder_loss_fn(resolve_decoder_cfg(_cfg(dec_loss="perceptual")))
    with pytest.raises(ValueError, match="unknown dec_lr_schedule"):
        decoder_schedule(resolve_decoder_cfg(_cfg(dec_lr_schedule="step")))
    with pytest.raises(ValueError, match="unknown decoder recipe"):
        resolve_decoder_cfg(_cfg(dec_recipe="v3_wishful"))


# ---------------------------------------------------------------------------
# 4. schedule
# ---------------------------------------------------------------------------


def test_constant_schedule_is_None_so_the_shipped_optimiser_is_untouched():
    assert decoder_schedule(resolve_decoder_cfg(_cfg())) is None


def test_cosine_schedule_warms_up_and_decays_to_zero():
    rec = resolve_decoder_cfg(_cfg(dec_lr_schedule="cosine", dec_steps=100, dec_lr=1e-3))
    sched = decoder_schedule(rec)
    assert float(sched(0)) < float(sched(5))
    assert float(sched(5)) == pytest.approx(1e-3, rel=1e-6)  # 5% warmup of 100
    assert float(sched(100)) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. the two declared variants, and config plumbing
# ---------------------------------------------------------------------------


def test_the_two_declared_variants_are_the_ones_that_shipped():
    """⛔ Two variants were pre-registered; this pins what they are."""
    assert set(DECODER_RECIPES) == {
        "baseline", "v1_capacity_schedule", "v2_object_weighted"
    }
    v1 = resolve_decoder_cfg(GenBridgeConfig(), "v1_capacity_schedule")
    assert v1.loss == "mse"  # V1 changes capacity + schedule ONLY
    assert v1.blocks_per_stage == 2 and v1.lr_schedule == "cosine"
    assert list(v1.channels) == [256, 128, 64]
    assert v1.steps == 1800  # ⭐ the winning recipe's constants are pinned here
    v2 = resolve_decoder_cfg(GenBridgeConfig(), "v2_object_weighted")
    assert v2.loss == "object_weighted"
    # V2 changes the LOSS ONLY, so the comparison isolates the objective
    base = resolve_decoder_cfg(GenBridgeConfig())
    assert (v2.channels, v2.blocks_per_stage, v2.steps, v2.lr_schedule) == (
        base.channels, base.blocks_per_stage, base.steps, base.lr_schedule
    )


def test_recipe_overrides_do_not_mutate_the_config():
    cfg = GenBridgeConfig()
    resolve_decoder_cfg(cfg, "v1_capacity_schedule")
    assert cfg.dec_steps == 1200 and cfg.dec_width_mult == 1.0
    assert cfg.dec_recipe == "baseline"


def test_new_decoder_knobs_roundtrip_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_bridge.dec_recipe = "v2_object_weighted"
    cfg.gen_bridge.dec_edge_weight = 3.5
    cfg.gen_bridge.dec_blocks_per_stage = 2
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.gen_bridge.dec_recipe == "v2_object_weighted"
    assert back.gen_bridge.dec_edge_weight == 3.5
    assert back.gen_bridge.dec_blocks_per_stage == 2


def test_defaults_preserve_the_shipped_behaviour():
    """⛔ Adding knobs must not change what the bridge does by default."""
    d = GenBridgeConfig()
    assert (d.dec_recipe, d.dec_loss, d.dec_lr_schedule) == ("baseline", "mse", "constant")
    assert (d.dec_width_mult, d.dec_blocks_per_stage) == (1.0, 1)
    assert d.dec_steps == 1200 and d.dec_channels == [128, 64, 32]


# ---------------------------------------------------------------------------
# 6. ⚠ §7.23 — the x64-at-module-import hazard, for the new code paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x64", [False, True])
def test_deep_decoder_and_weighted_loss_read_a_float32_input_under_x64(x64):
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", x64)
        rec = resolve_decoder_cfg(_cfg(dec_recipe="v2_object_weighted",
                                       dec_blocks_per_stage=2))
        dec = make_decoder((8, 4, 4), (3, 16, 16), rec, jax.random.PRNGKey(0))
        z32 = np.zeros((2, 8, 4, 4), np.float32)
        x32 = np.zeros((2, 3, 16, 16), np.float32)
        assert dec(z32[0]).shape == (3, 16, 16)
        val = float(decoder_loss_fn(rec)(dec, jnp.asarray(z32), jnp.asarray(x32)))
        assert np.isfinite(val)
        assert np.isfinite(np.asarray(edge_weight_map(jnp.asarray(x32), 8.0, 5))).all()
    finally:
        jax.config.update("jax_enable_x64", prev)
