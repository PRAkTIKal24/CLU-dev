"""The dt-units split on the EVAL/CAFE path (w20).

``cfg.dt`` was read for two physically distinct quantities: the DATA sampling
interval (finite-difference momentum ``p = Delta-q / dt``) and the Verlet
INTEGRATOR step. On cycle-indexed C-MAPSS the true sampling interval is 1 cycle,
but the shipped default was 0.05, which inflated every momentum 20x and every
kinetic energy 400x -- and that made ``energy_reg`` 99.2% of the loss
(``clu-latent-io-audit``, w19).

These tests pin the split:

  * ``data_dt`` carries the physical unit and ONLY scales the momentum;
  * ``dt`` carries the numerical unit and ONLY sets integrator resolution;
  * ``substeps``/``dt_eff`` keep a rollout on the data's physical time grid, so
    ``dt`` can be refined WITHOUT silently changing what a horizon means --
    the trap that makes this more than a one-line fix;
  * ``data_dt == dt`` reproduces the pre-split (conflated) behaviour exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.eval.clu_scorer import _build_model, _SharedCLUFit, rollout_on_data_grid
from chlu.eval.config import CLUCafeEncodeConfig, CLUScorerConfig

L, C = 12, 4


def _windows(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, L * C)).astype(np.float32)


def _model(cfg=None, c=C, seed=0):
    cfg = cfg or CLUScorerConfig(hidden=8)
    return _build_model(cfg, c, jax.random.PRNGKey(seed))


# ── config surface ────────────────────────────────────────────────────────
def test_data_dt_defaults_to_one_cycle():
    """C-MAPSS is cycle-indexed: one frame is one cycle."""
    assert CLUScorerConfig().data_dt == 1.0


def test_substeps_and_dt_eff_consistent():
    cfg = CLUScorerConfig(dt=0.05, data_dt=1.0)
    assert cfg.substeps == 20
    assert cfg.dt_eff == pytest.approx(0.05)
    # substeps integrator steps span exactly one data interval
    assert cfg.substeps * cfg.dt_eff == pytest.approx(cfg.data_dt)


def test_dt_eff_snaps_to_the_data_grid():
    """A dt that does not divide data_dt is snapped, not left to drift."""
    cfg = CLUScorerConfig(dt=0.3, data_dt=1.0)
    assert cfg.substeps == 3
    assert cfg.dt_eff == pytest.approx(1.0 / 3.0)
    assert cfg.substeps * cfg.dt_eff == pytest.approx(1.0)


def test_equal_dt_and_data_dt_is_the_single_step_path():
    for v in (0.05, 1.0):
        cfg = CLUScorerConfig(dt=v, data_dt=v)
        assert cfg.substeps == 1
        assert cfg.dt_eff == pytest.approx(v)


def test_integrator_step_may_not_exceed_the_data_interval():
    with pytest.raises(ValueError, match="must not exceed"):
        CLUScorerConfig(dt=2.0, data_dt=1.0)


@pytest.mark.parametrize("kw", [{"dt": 0.0}, {"data_dt": 0.0}, {"data_dt": -1.0}])
def test_non_positive_steps_rejected(kw):
    with pytest.raises(ValueError):
        CLUScorerConfig(**kw)


# ── the momentum is a DATA-interval quantity ──────────────────────────────
def test_momentum_scales_with_data_dt_only():
    """p = Delta-q / data_dt -- refining the INTEGRATOR must not touch it."""
    q0 = jnp.zeros((3, C))
    q1 = jnp.ones((3, C))

    p_coarse = _SharedCLUFit(
        CLUScorerConfig(dt=1.0, data_dt=1.0), window_size=L
    )._momentum(q0, q1)
    p_fine = _SharedCLUFit(
        CLUScorerConfig(dt=0.05, data_dt=1.0), window_size=L
    )._momentum(q0, q1)
    np.testing.assert_allclose(np.asarray(p_coarse), np.asarray(p_fine))
    np.testing.assert_allclose(np.asarray(p_coarse), 1.0)

    # ...whereas the physical interval does scale it (this is the w19 bug: at
    # data_dt=0.05 the momentum is 20x too large on cycle-indexed data).
    p_bug = _SharedCLUFit(
        CLUScorerConfig(dt=0.05, data_dt=0.05), window_size=L
    )._momentum(q0, q1)
    np.testing.assert_allclose(np.asarray(p_bug), 20.0)


def test_kinetic_energy_inflation_is_the_square_of_the_interval_error():
    """The 400x that made energy_reg 99.2% of the loss."""
    m = _model()
    q = jnp.zeros((C,))
    dq = jnp.full((C,), 0.1)
    fit = lambda ddt: _SharedCLUFit(  # noqa: E731
        CLUScorerConfig(dt=min(ddt, 1.0), data_dt=ddt), window_size=L
    )._momentum(q, q + dq)
    k = lambda p: float(m.H(q, p) - jnp.sum(m.potential_net(q)))  # noqa: E731
    assert k(fit(0.05)) / k(fit(1.0)) == pytest.approx(400.0, rel=1e-3)


# ── the rollout stays on the data's physical time grid ────────────────────
def test_rollout_on_data_grid_returns_one_row_per_data_frame():
    m = _model()
    q0, p0 = jnp.zeros((C,)), jnp.ones((C,))
    traj = rollout_on_data_grid(m, q0, p0, n_samples=5, dt=0.25, substeps=4)
    assert traj.shape[0] == 5


def test_substepping_preserves_physical_time_not_step_count():
    """The trap: a finer dt must predict the SAME physical horizon.

    A naive ``model(q,p,hz,dt)`` with dt refined 8x would cover 1/8 the physical
    time while still being compared against data hz frames ahead. Substepping is
    what stops that, so the two rollouts must agree to integrator accuracy.
    """
    m = _model()
    q0, p0 = jnp.zeros((C,)), jnp.ones((C,))
    coarse = np.asarray(rollout_on_data_grid(m, q0, p0, 4, dt=1.0, substeps=1))
    fine = np.asarray(rollout_on_data_grid(m, q0, p0, 4, dt=0.125, substeps=8))
    naive = np.asarray(m(q0, p0, 4, 0.125, 0.0))  # the bug: 4 steps of 0.125
    assert coarse.shape == fine.shape == naive.shape

    err_substepped = np.abs(coarse - fine).max()
    err_naive = np.abs(coarse - naive).max()
    # substepping lands on the same physical times (O(dt^2) apart); the naive
    # rollout covers 1/8 the physical time and is an order of magnitude off.
    assert err_substepped < 0.1, err_substepped
    assert err_naive > 10 * err_substepped, (err_naive, err_substepped)


def test_refining_the_integrator_converges():
    """Halving dt at fixed physical horizon must reduce the discrepancy."""
    m = _model()
    q0, p0 = jnp.zeros((C,)), jnp.ones((C,))
    ref = np.asarray(rollout_on_data_grid(m, q0, p0, 4, dt=1 / 64, substeps=64))
    errs = [
        np.abs(np.asarray(rollout_on_data_grid(m, q0, p0, 4, dt=1.0 / s, substeps=s)) - ref).max()
        for s in (1, 2, 4, 8)
    ]
    assert errs == sorted(errs, reverse=True), errs


# ── end-to-end: the split is reachable and backward-compatible ────────────
def _fit(**kw):
    cfg = CLUScorerConfig(
        hidden=8, epochs=3, batch_size=16, max_fit_windows=64,
        predict_horizon=4, relax_steps=3, seed=0, **kw
    )
    f = _SharedCLUFit(cfg, window_size=L)
    f.ensure_fit(_windows())
    return f


def test_conflated_config_is_bit_identical_to_the_legacy_path():
    """data_dt == dt reproduces pre-split behaviour, so w19 stays reproducible."""
    a = _fit(dt=0.05, data_dt=0.05)
    b = _fit(dt=0.05, data_dt=0.05)
    np.testing.assert_array_equal(
        a.score(_windows(16, seed=1), "energy"),
        b.score(_windows(16, seed=1), "energy"),
    )


def test_split_changes_the_energy_scale_but_stays_finite():
    legacy = _fit(dt=0.05, data_dt=0.05).score(_windows(16, seed=1), "energy")
    split = _fit(dt=1.0, data_dt=1.0).score(_windows(16, seed=1), "energy")
    assert np.all(np.isfinite(legacy)) and np.all(np.isfinite(split))
    # correcting a 20x momentum error must lower the energy scale substantially
    assert np.mean(np.abs(split)) < np.mean(np.abs(legacy))


def test_relax_budget_uses_the_integrator_step():
    """gamma*steps*dt_eff -- relax_steps counts INTEGRATOR steps."""
    enc = CLUCafeEncodeConfig()
    cfg = CLUScorerConfig(dt=1.0, data_dt=1.0, gamma=0.1, relax_steps=32)
    assert enc.relax_budget(cfg) == pytest.approx(3.2)
    # refining the integrator at fixed step COUNT shortens the damping budget
    cfg_fine = CLUScorerConfig(dt=0.05, data_dt=1.0, gamma=0.1, relax_steps=32)
    assert enc.relax_budget(cfg_fine) == pytest.approx(0.16)
