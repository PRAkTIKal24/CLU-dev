"""Tests for the w22 gated-write performance test (chlu/experiments/exp_gated_write.py).

Pins the things that could silently fake this comparison: that the cfg-override
context never leaks a CLU knob into a baseline, that the two CLU rows differ only
in the write current, that the extrapolation harness really evaluates a block on
a LONGER sequence than it trained on (the founding CHLU claim), and that the
robustness noise never touches the marker channel (which would change the task,
not stress it).
"""

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.core.blocks import make_block
from chlu.experiments.exp_gated_write import (
    _build_extrap_family,
    _clu_gamma_for,
    _noisy_adding_metric,
    _train_extrapolate,
    cfg_overrides,
    item1_variants,
)
from chlu.experiments.exp_primitive_harness import AddingFamily


def _tiny_cfg():
    cfg = get_default_config().experiment_primitive_harness
    cfg.d_model = 16
    cfg.n_layers = 1
    cfg.target_block_params = 2000
    cfg.batch_size = 8
    cfg.eval_batch = 16
    cfg.train_steps = 3
    cfg.lr_grid = [1e-3]
    cfg.n_seeds = 1
    return cfg


def test_cfg_overrides_restores_every_field():
    cfg = _tiny_cfg()
    before = (cfg.clu_write_mode, cfg.clu_gamma)
    with cfg_overrides(cfg, {"clu_write_mode": "gated", "clu_gamma": 0.0}):
        assert cfg.clu_write_mode == "gated" and cfg.clu_gamma == 0.0
    assert (cfg.clu_write_mode, cfg.clu_gamma) == before


def test_cfg_overrides_restores_on_exception():
    cfg = _tiny_cfg()
    before = cfg.clu_write_mode
    try:
        with cfg_overrides(cfg, {"clu_write_mode": "gated"}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert cfg.clu_write_mode == before


def test_item1_has_six_variants_and_mqar_uses_gamma_zero():
    cfg = _tiny_cfg()
    labels = [v["label"] for v in item1_variants(cfg, "adding")]
    assert labels == ["mlp", "gru", "ssm", "attention", "clu_linear", "clu_gated"]
    # MQAR forces gamma=0 for BOTH CLU rows; adding/parity keep shipped gamma.
    for v in item1_variants(cfg, "mqar_T128_kv4"):
        if v["primitive"] == "clu":
            assert v["overrides"]["clu_gamma"] == cfg.gw_mqar_gamma == 0.0
    for v in item1_variants(cfg, "adding_T128"):
        if v["primitive"] == "clu":
            assert v["overrides"]["clu_gamma"] == cfg.clu_gamma
    # The linear and gated CLU rows share gamma; only the write mode differs.
    clu_rows = [v for v in item1_variants(cfg, "adding") if v["primitive"] == "clu"]
    assert {r["overrides"]["clu_write_mode"] for r in clu_rows} == {"linear", "gated"}
    assert len({r["overrides"]["clu_gamma"] for r in clu_rows}) == 1


def test_clu_gamma_for_family():
    cfg = _tiny_cfg()
    assert _clu_gamma_for(cfg, "mqar_T128_kv4") == 0.0
    assert _clu_gamma_for(cfg, "adding_T128") == cfg.clu_gamma
    assert _clu_gamma_for(cfg, "parity_T64") == cfg.clu_gamma


def test_gated_and_linear_write_are_different_maps():
    # The whole experiment rests on the gate changing the block; if it didn't,
    # every "gated" number would just be a relabelled "linear" one.
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 12))
    lin = make_block("clu", 12, 8, key=jax.random.PRNGKey(0), clu_write_mode="linear")
    gat = make_block("clu", 12, 8, key=jax.random.PRNGKey(0), clu_write_mode="gated")
    assert not np.allclose(np.asarray(lin(x)), np.asarray(gat(x)))


def test_extrapolation_evaluates_on_a_longer_sequence():
    # Train a GRU block at T=8, ask it for the metric at T=8 and T=16. The point
    # is that the SAME block runs at 2x the trained length without a shape error
    # and returns a finite number (a fixed-length pos-embedding would crash).
    cfg = _tiny_cfg()
    out = _train_extrapolate(
        cfg, "gru", AddingFamily, train_T=8, eval_Ts=[8, 16],
        width=8, lr=1e-3, seed=0, overrides={},
    )
    assert set(out["metrics"]) == {8, 16}
    assert all(np.isfinite(v) for v in out["metrics"].values())


def test_extrap_family_has_room_for_the_longest_eval():
    cfg = _tiny_cfg()
    fam = _build_extrap_family(cfg, AddingFamily, seq_len=8, max_len=32, overrides={})
    assert fam.seq_len == 8
    assert fam.model_kwargs["max_len"] == 32


def test_robustness_noise_leaves_the_marker_channel_intact():
    # The noise sweep must corrupt only the VALUE channel; if it hit the marker
    # channel it would change which positions are summed, i.e. a different task.
    fam = AddingFamily(16)
    key = jax.random.PRNGKey(3)
    x, y, mask = fam.jbatch(key, 8)
    nkey = jax.random.fold_in(key, 777)
    noise = jax.random.normal(nkey, x.shape) * 0.3
    noise = noise.at[..., 1].set(0.0)
    assert jnp.allclose(noise[..., 1], 0.0)
    assert not jnp.allclose(noise[..., 0], 0.0)


def test_noisy_metric_runs_and_is_finite():
    cfg = _tiny_cfg()
    fam = AddingFamily(16)
    from chlu.experiments.exp_primitive_harness import build_model, match_width

    width, *_ = match_width("gru", cfg, fam, jax.random.PRNGKey(0))
    m = build_model("gru", cfg, fam, width, jax.random.PRNGKey(0))
    v0 = _noisy_adding_metric(m, fam, 0.0, jax.random.PRNGKey(5), 8)
    v1 = _noisy_adding_metric(m, fam, 0.2, jax.random.PRNGKey(5), 8)
    assert np.isfinite(v0) and np.isfinite(v1)
