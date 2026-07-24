"""Tests for the RETRY-COMPUTE study (w23) — the accuracy-vs-compute curve.

Everything runs on tiny SYNTHETIC patterns so the suite stays fast and has no
dataset dependency (the MNIST arm is exercised by the CLI). The float32 fixture
matches the reported numerics and dodges the repo-wide x64 test-isolation hazard
(handover §7.2); the experiment code itself is x64-safe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments.exp_hopfield_capacity import (
    _median_nn_distance,
    build_clu_memory,
    noise_query,
)
from chlu.experiments.exp_retry_compute import (
    _confidence_and_nn,
    _dt_of,
    _ensemble_ladder,
    _feedforward_ladder,
    _hopfield_ladder,
    _majority,
    _retry_ladder,
    _settle,
    run_cell,
)


@pytest.fixture
def float32_dynamics():
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _patterns(M, D, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray((rng.random((M, D)) > 0.5).astype(np.float32))


def _quick_cfg(M_pool=32):
    cfg = get_default_config().experiment_retry_compute
    cfg.retry_ladder = [0, 1, 2]
    cfg.conf_thresholds = [0.97, 1.0]
    cfg.main_threshold = 0.99
    cfg.retry_step_frac = 0.25
    cfg.clu_steps = 30
    cfg.rollout_chunk = 32
    cfg.n_data_pool = M_pool
    return cfg


def test_confidence_is_label_free_and_bounded():
    P = _patterns(6, 32, seed=1)
    reads = np.asarray(P)  # perfect reads -> confidence 1, nn == truth
    cos, nn = _confidence_and_nn(reads, P)
    np.testing.assert_allclose(cos, 1.0, atol=1e-4)
    np.testing.assert_array_equal(nn, np.arange(6))


def test_majority_vote():
    votes = np.array([[0, 1, 2], [0, 1, 5], [3, 1, 2]])  # (R=3, Nq=3)
    maj = _majority(votes)
    np.testing.assert_array_equal(maj, [0, 1, 2])


def test_settle_is_finite_and_shape_stable(float32_dynamics):
    P = _patterns(5, 24, seed=2)
    cfg = _quick_cfg()
    s = cfg.clu_s_frac * _median_nn_distance(P)
    model = build_clu_memory(P, s, cfg)
    Q = np.asarray(noise_query(P, 0.3, jax.random.PRNGKey(0)))
    P0 = np.zeros_like(Q)
    out = _settle(model, Q, P0, cfg.clu_steps, _dt_of(cfg, s), cfg.clu_gamma, 3, 32)
    assert out.shape == Q.shape
    assert np.all(np.isfinite(out))


def test_ladders_return_all_rungs_and_monotone_compute(float32_dynamics):
    P = _patterns(8, 32, seed=3)
    cfg = _quick_cfg()
    s = cfg.clu_s_frac * _median_nn_distance(P)
    dt = _dt_of(cfg, s)
    model = build_clu_memory(P, s, cfg)
    Q = noise_query(P, 0.3, jax.random.PRNGKey(1))
    true = np.arange(8)
    reads0 = _settle(model, np.asarray(Q), np.zeros_like(np.asarray(Q)),
                     cfg.clu_steps, dt, cfg.clu_gamma, 3, 32)
    rng = np.random.default_rng(0)

    gated = _retry_ladder(model, Q, reads0, P, true, cfg, dt, 0.99, "gated", rng)
    ung = _retry_ladder(model, Q, reads0, P, true, cfg, dt, 0.99, "ungated", rng)
    kick = _retry_ladder(model, Q, reads0, P, true, cfg, dt, 0.99, "kick", rng)
    ens = _ensemble_ladder(model, Q, reads0, P, true, cfg, dt, rng)
    ff = _feedforward_ladder(Q, P, true, cfg, rng)
    hop = _hopfield_ladder(Q, P, true, cfg)

    for ladder in (gated, ung, kick, ens, ff, hop):
        assert set(ladder.keys()) == set(cfg.retry_ladder)
        accs = [ladder[k][0] for k in sorted(ladder)]
        comps = [ladder[k][1] for k in sorted(ladder)]
        assert all(0.0 <= a <= 1.0 for a in accs)
        # compute is non-decreasing along the ladder
        assert all(comps[i] <= comps[i + 1] + 1e-9 for i in range(len(comps) - 1))

    # the gate is adaptive: gated spends <= ungated at the top of the ladder
    top = max(cfg.retry_ladder)
    assert gated[top][1] <= ung[top][1] + 1e-9
    # ungated pays the full (k+1)x budget
    np.testing.assert_allclose(ung[top][1], top + 1, atol=1e-6)


@pytest.mark.parametrize("query_type,level", [("mask", 0.5), ("noise", 0.3)])
def test_run_cell_has_six_lines_and_threshold_sweep(
    float32_dynamics, query_type, level
):
    P = _patterns(16, 32, seed=4)
    cfg = _quick_cfg()
    cell = run_cell(cfg, P, query_type, level, seed=0)
    assert set(cell["lines"].keys()) == {
        "clu_gated", "ungated_all", "ensemble",
        "random_kick", "feedforward_nn", "hopfield_ksteps",
    }
    assert set(cell["threshold_sweep"].keys()) == {"0.97", "1.00"}
    assert cell["query_type"] == query_type
    assert 0.0 <= cell["first_pass_acc"] <= 1.0


def test_config_round_trip_includes_retry_compute(tmp_path):
    from chlu.config import get_default_config, load_config, save_config

    cfg = get_default_config()
    cfg.experiment_retry_compute.retry_ladder = [0, 1, 3]
    cfg.experiment_retry_compute.main_threshold = 0.42
    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.experiment_retry_compute.retry_ladder == [0, 1, 3]
    assert loaded.experiment_retry_compute.main_threshold == 0.42
