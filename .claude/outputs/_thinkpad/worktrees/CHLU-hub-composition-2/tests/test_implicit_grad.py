"""Tests for ``chlu/core/implicit_grad.py`` — the implicit/DEQ settle gradient.

The numeric bars here are the ones registered in
``.claude/outputs/trainability-spike/PREREG.md`` §1, taken from
``trainability-spike-theory`` §7 request 3 and §Q4.2.

⚠ **x64 discipline.** The gradcheck needs float64 (the registered bar is 1e-5
against re-settled finite differences, which is below the float32 noise floor),
so this module turns x64 on **inside an autouse fixture** and restores the
previous value afterwards — never at module import, which is the repo-wide
isolation hazard documented in ``tests/test_clu_system.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.implicit_grad import (
    GaussianWellsPotential,
    SettleSpec,
    coset_transport,
    implicit_grad,
    implicit_settle,
    ridge_alarm,
    settle_forward,
    settle_telemetry,
    theory_ridge,
    toy_model,
    truncated_rollout,
    unroll_grad,
)


@pytest.fixture(autouse=True)
def x64():
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


CENTERS = np.stack([np.cos(2 * np.pi * np.arange(4) / 4),
                    np.sin(2 * np.pi * np.arange(4) / 4)], axis=1)
AMP = np.array([1.0, 0.9, 1.1, 0.95])


def _toy():
    return toy_model(CENTERS, jnp.asarray(AMP), s=0.35, alpha=0.05)


def _q0():
    return jnp.asarray(CENTERS[0] + np.array([0.2, -0.15]))


def _loss(q):
    return 0.5 * jnp.sum(q**2)


def _amp(tree):
    return np.asarray(tree.potential_net.amp)


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


# --------------------------------------------------------------------------
# the theorist's constants
# --------------------------------------------------------------------------
def test_theory_ridge_reproduces_the_shipped_value():
    """lambda_ridge = 2 gamma m ln(1/tol) / (N (2-gamma) dt^2) = 0.354 shipped."""
    assert theory_ridge(0.05, 0.05, 400, 1e-3, 1.0) == pytest.approx(0.354, abs=5e-4)


def test_ridge_alarm_fires_only_above_a_tenth_of_the_median():
    assert not ridge_alarm(0.354, [8.24])  # 0.354 < 0.824
    assert ridge_alarm(1.94, [8.24])  # the theorist's gamma=0.1, N=150 example


def test_coset_transport_is_the_closed_form_geometric_sum():
    # theta_inf - theta_0 = dt L0 / (m r*^2 gamma); sum_{n>=0}(1-gamma)^n = 1/gamma
    got = coset_transport(L0=0.2, r_star=1.0, gamma=0.05, dt=0.05, mass=1.0)
    assert got == pytest.approx(0.05 * 0.2 / (1.0 * 1.0 * 0.05))


# --------------------------------------------------------------------------
# Q1.1: the fixed point of the shipped dissipative map
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gamma", [0.02, 0.05, 0.1, 0.3])
def test_settle_lands_on_a_critical_point_with_exactly_zero_momentum(gamma):
    """Fix(T) = {(q, 0) : grad V(q) = 0} — theory Prop Q1.1."""
    model = _toy()
    # two-branch step budget (an underdamped-only budget is 16x short at
    # gamma=0.3, dt=0.02 -- the overdamped rider of Q4.2)
    spec = SettleSpec(steps=4000, dt=0.05, gamma=gamma)
    q_star, p_star = settle_forward(model, _q0(), jnp.zeros(2), spec)
    assert float(jnp.linalg.norm(p_star)) < 1e-10
    tele = settle_telemetry(model, q_star[None, :], centers=CENTERS)
    assert float(tele["residual"][0]) < 1e-10
    assert float(tele["lambda_min"][0]) > 0.0  # a genuine minimum


# --------------------------------------------------------------------------
# A1-A2: THE GRADCHECK (the registered bars)
# --------------------------------------------------------------------------
def test_implicit_gradient_matches_resettled_finite_differences():
    """Registered bar: rel. err <= 1e-5 (PREREG §1). Theory reached 1.3e-8."""
    model = _toy()
    spec = SettleSpec(steps=1500, dt=0.05, gamma=0.05)
    g_imp = _amp(implicit_grad(model, _q0(), jnp.zeros(2), spec, _loss))

    h = 1e-5
    g_fd = np.zeros(4)
    for i in range(4):
        ap, am = AMP.copy(), AMP.copy()
        ap[i] += h
        am[i] -= h
        qp, _ = settle_forward(toy_model(CENTERS, jnp.asarray(ap), s=0.35, alpha=0.05),
                               _q0(), jnp.zeros(2), spec)
        qm, _ = settle_forward(toy_model(CENTERS, jnp.asarray(am), s=0.35, alpha=0.05),
                               _q0(), jnp.zeros(2), spec)
        g_fd[i] = float((_loss(qp) - _loss(qm)) / (2 * h))
    assert _relerr(g_imp, g_fd) < 1e-5


@pytest.mark.parametrize("k,bar", [(180, 3e-2), (270, 3e-3), (449, 3e-5)])
def test_implicit_matches_truncated_unroll_at_the_registered_depths(k, bar):
    """The theorist's depth table (§7 request 3), registered before running.

    The disagreement is ``rho^k`` with ``rho = sqrt(1-gamma) = 0.97479``, i.e.
    1e-2 / 1e-3 / 1e-5 at k = 180 / 270 / 449. The bars are 3x those.
    """
    model = _toy()
    spec = SettleSpec(steps=1500, dt=0.05, gamma=0.05)
    g_imp = _amp(implicit_grad(model, _q0(), jnp.zeros(2), spec, _loss))
    g_k = _amp(unroll_grad(model, _q0(), jnp.zeros(2), spec, _loss, retain=k))
    err = _relerr(g_k, g_imp)
    assert err < bar
    # ...and it is genuinely rho^k, not merely "small": within a factor 3 of it.
    rho_k = float(np.sqrt(1 - 0.05) ** k)
    assert 0.3 * rho_k < err < 3.0 * rho_k


def test_implicit_answer_is_gamma_and_dt_independent():
    """Q1.3: the fixed-point set contains no gamma, dt or M, so neither does the
    answer. Registered spread bar 1e-8 (theory measured 5.5e-14 in numpy)."""
    model = _toy()
    grads = []
    for gamma in (0.02, 0.05, 0.1):
        for dt in (0.02, 0.05, 0.1):
            spec = SettleSpec(steps=4000, dt=dt, gamma=gamma)
            grads.append(_amp(implicit_grad(model, _q0(), jnp.zeros(2), spec, _loss)))
    G = np.array(grads)
    spread = np.max(np.abs(G - G.mean(0))) / np.max(np.abs(G))
    assert spread < 1e-8


def test_the_ridge_costs_the_predicted_bias_and_is_off_by_default():
    """Theory: the shipped ridge costs ~4.1 % on a healthy well mode. And it must
    never be silently enabled."""
    assert SettleSpec().ridge == 0.0
    assert SettleSpec().as_flags()["ridge_enabled"] is False
    model = _toy()
    spec = SettleSpec(steps=1500, dt=0.05, gamma=0.05)
    g0 = _amp(implicit_grad(model, _q0(), jnp.zeros(2), spec, _loss))
    lam = theory_ridge(0.05, 0.05, 400, 1e-3, 1.0)
    import dataclasses

    g1 = _amp(implicit_grad(model, _q0(), jnp.zeros(2),
                            dataclasses.replace(spec, ridge=lam), _loss))
    assert _relerr(g1, g0) == pytest.approx(0.041, abs=0.01)


# --------------------------------------------------------------------------
# the structural claim: d q*/d q0 = 0
# --------------------------------------------------------------------------
def test_the_implicit_settle_sends_exactly_zero_gradient_to_the_launch_point():
    """``Fix(T)`` does not contain ``q0``, so ``d q*/d q0 = 0`` a.e. — the exact
    form of N61 ("gradient search for an address is dead"). This is what makes a
    settled-point read-out unable to train its own ``phi``."""
    model = _toy()
    spec = SettleSpec(steps=1500, dt=0.05, gamma=0.05)

    def f(z):
        return jnp.sum(implicit_settle(model, z, jnp.zeros(2), spec) ** 2)

    assert float(jnp.linalg.norm(jax.grad(f)(_q0()))) == 0.0

    # and the unrolled version agrees, at the theory's rho^N: (1-0.05)^(1500/2)
    def g(z):
        qs, _ = settle_forward(model, z, jnp.zeros(2), spec)
        return jnp.sum(qs**2)

    assert float(jnp.linalg.norm(jax.grad(g)(_q0()))) < 1e-14


# --------------------------------------------------------------------------
# truncated_rollout mechanics
# --------------------------------------------------------------------------
def test_truncated_rollout_matches_the_harness_stride_layout():
    model = _toy()
    q, p = _q0(), jnp.zeros(2)
    full = model(q, p, 400, 0.05, 0.05)
    for stride in (1, 2, 8, 16):
        got = truncated_rollout(model, q, p, 400, 0.05, 0.05, retain=None,
                                stride=stride)
        assert got.shape == full[::stride].shape
        assert np.allclose(np.asarray(got), np.asarray(full[::stride]))
        # ...and the same buffer is produced when the window is split at a seam
        split = truncated_rollout(model, q, p, 400, 0.05, 0.05, retain=137,
                                  stride=stride)
        assert np.allclose(np.asarray(split), np.asarray(full[::stride]))


def test_truncated_rollout_returns_the_true_endpoint_not_the_last_strided_point():
    model = _toy()
    q, p = _q0(), jnp.zeros(2)
    full = model(q, p, 400, 0.05, 0.05)
    traj, q_end, p_end = truncated_rollout(model, q, p, 400, 0.05, 0.05,
                                           retain=270, stride=8,
                                           return_endpoint=True)
    assert np.allclose(np.asarray(q_end), np.asarray(full[-1, :2]))
    assert np.allclose(np.asarray(p_end), np.asarray(full[-1, 2:]))
    # 400 steps strided by 8 stops at index 392, so the endpoint is NOT in traj
    assert not np.allclose(np.asarray(traj[-1, :2]), np.asarray(q_end))


def test_tail_truncation_severs_the_launch_point_entirely():
    """⚠ Truncation DIRECTION matters. Retaining the last k steps is right for
    ``theta`` and fatal for ``phi``: the retained window is entered through a
    ``stop_gradient``, so ``d/d q0`` is **exactly** 0, not ``rho^k``."""
    model = _toy()
    p = jnp.zeros(2)

    def f(z, retain):
        tr = truncated_rollout(model, z, p, 400, 0.05, 0.05, retain=retain, stride=8)
        return jnp.sum(tr**2)

    assert float(jnp.linalg.norm(jax.grad(f)(_q0(), 270))) == 0.0
    assert float(jnp.linalg.norm(jax.grad(f)(_q0(), None))) > 1e-6


def test_settle_telemetry_reports_the_full_q35_triple():
    model = _toy()
    spec = SettleSpec(steps=1500, dt=0.05, gamma=0.05)
    q_star, _ = settle_forward(model, _q0(), jnp.zeros(2), spec)
    t = settle_telemetry(model, q_star[None, :], centers=CENTERS, ridge=0.354,
                         d_capture=0.5)
    for leg in ("residual", "lambda_min", "basin"):
        assert leg in t
    assert t["basin"][0] == 0  # launched next to well 0 and stayed there
    assert bool(t["basin_ok"][0])
    assert t["n_negative_modes"][0] == 0


def test_gaussian_wells_potential_is_the_analytic_toy():
    V = GaussianWellsPotential(CENTERS, jnp.asarray(AMP), s=0.35, alpha=0.05)
    q = jnp.asarray(CENTERS[0])
    # at a well centre the well's own contribution is -amp
    assert float(V(q)) < -0.5
    # and the confinement term is alpha*|q|^2 at large radius
    far = jnp.asarray([10.0, 10.0])
    assert float(V(far)) == pytest.approx(0.05 * 200.0, rel=1e-6)
