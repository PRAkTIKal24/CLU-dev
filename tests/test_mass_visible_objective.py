"""Can the objective SEE the mass spectrum? (w20 mass-visible-objective)

w19 found, and the w20 dt-units fix did NOT change, a structural defect: the
EBM contrastive term has **exactly zero** gradient w.r.t. ``log_mass``. The
negatives perturb ``q`` only, so with ``H = K(p) + V(q)`` the kinetic terms
cancel identically in ``<H_data> - <H_neg>``. If masses are meant to be the
access keys, the representational half of the objective is blind to them.

These tests pin:
  * the defect itself (so a future refactor cannot silently "fix" or worsen it
    without a test changing status);
  * ``neg_momentum_scale`` as the source-level repair, including its
    analytically predicted ``sigma^2`` gradient scaling;
  * the ``mass_parameterization`` vocabulary -- the zero-mean gauge fix and the
    escape from softplus's linear regime;
  * that every new knob is OFF by default and bit-compatible.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.chlu_unit import CHLU
from chlu.eval.clu_scorer import (
    _collect_log_mass,
    _collect_mass_vector,
    _mass_ratio,
    _SharedCLUFit,
)
from chlu.eval.config import CLUScorerConfig

L, C = 12, 4


def _windows(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, L * C)).astype(np.float32)


def _fit(**kw):
    cfg = CLUScorerConfig(
        hidden=8, epochs=4, batch_size=16, max_fit_windows=64,
        predict_horizon=4, relax_steps=3, seed=0, **kw
    )
    f = _SharedCLUFit(cfg, window_size=L)
    f.ensure_fit(_windows())
    return f


# ── defaults are off / bit-compatible ────────────────────────────────────
def test_new_knobs_default_to_historical_behaviour():
    cfg = CLUScorerConfig()
    assert cfg.neg_momentum_scale == 0.0
    assert cfg.mass_parameterization == "softplus"


def test_default_arm_is_bit_identical_to_explicit_defaults():
    """The knobs must not perturb the RNG stream when they are off."""
    a = _fit()
    b = _fit(neg_momentum_scale=0.0, mass_parameterization="softplus")
    np.testing.assert_array_equal(
        _collect_log_mass(a.model), _collect_log_mass(b.model)
    )


@pytest.mark.parametrize("bad", [-0.1, -1.0])
def test_negative_neg_momentum_scale_rejected(bad):
    with pytest.raises(ValueError, match="neg_momentum_scale"):
        CLUScorerConfig(neg_momentum_scale=bad)


def test_unknown_mass_parameterization_rejected():
    with pytest.raises(ValueError, match="mass_parameterization"):
        CLUScorerConfig(mass_parameterization="quadratic")
    with pytest.raises(ValueError, match="mass_parameterization"):
        CHLU(dim=3, mass_parameterization="quadratic", key=jax.random.PRNGKey(0))


# ── the defect, and its repair ───────────────────────────────────────────
def _contrast_grad(neg_p, dim=6, n=128):
    """RMS |d E_contrast / d log_mass| for momentum-perturbation scale neg_p."""
    model = CHLU(
        dim=dim, hidden=16, kinetic_mode="newtonian_learned",
        key=jax.random.PRNGKey(0),
    )
    qs = jax.random.normal(jax.random.PRNGKey(7), (n, dim))
    ps = jax.random.normal(jax.random.PRNGKey(8), (n, dim))

    def contrast(m):
        h_data = jax.vmap(m.H)(qs, ps)
        nkey = jax.random.PRNGKey(3)
        if neg_p > 0.0:
            qk, pk = jax.random.split(nkey)
            noise = 0.5 * jax.random.normal(qk, qs.shape)
            ps_neg = ps + neg_p * jax.random.normal(pk, ps.shape)
        else:
            noise = 0.5 * jax.random.normal(nkey, qs.shape)
            ps_neg = ps
        return jnp.mean(h_data) - jnp.mean(jax.vmap(m.H)(qs + noise, ps_neg))

    g = np.asarray(eqx.filter_grad(contrast)(model).log_mass)
    return float(np.sqrt(np.mean(g**2)))


def test_contrastive_mass_gradient_is_EXACTLY_zero_without_momentum_noise():
    """The w19 defect. Exactly zero -- an identity, not a small number."""
    assert _contrast_grad(0.0) == 0.0


def test_momentum_perturbed_negatives_break_the_kinetic_cancellation():
    assert _contrast_grad(0.5) > 1e-3


def _mean_contrast_grad(neg_p, dim=6, n=128, n_keys=64):
    """RMS of the NOISE-AVERAGED mass gradient.

    The sigma^2 law is a statement about E[E_contrast]. A single noise draw
    also carries the cross-term -sum_i p_i d_i / M_i, which is O(sigma) with
    random sign and vanishes only in expectation -- so it must be averaged out
    before the law is testable (see the companion test below).
    """
    model = CHLU(
        dim=dim, hidden=16, kinetic_mode="newtonian_learned",
        key=jax.random.PRNGKey(0),
    )
    qs = jax.random.normal(jax.random.PRNGKey(7), (n, dim))
    ps = jax.random.normal(jax.random.PRNGKey(8), (n, dim))

    def contrast(m, k):
        h_data = jax.vmap(m.H)(qs, ps)
        qk, pk = jax.random.split(jax.random.PRNGKey(k))
        noise = 0.5 * jax.random.normal(qk, qs.shape)
        ps_neg = ps + neg_p * jax.random.normal(pk, ps.shape)
        return jnp.mean(h_data) - jnp.mean(jax.vmap(m.H)(qs + noise, ps_neg))

    tot = np.zeros(dim)
    for k in range(n_keys):
        tot += np.asarray(eqx.filter_grad(contrast)(model, k).log_mass)
    return float(np.sqrt(np.mean((tot / n_keys) ** 2)))


def test_expected_contrastive_mass_gradient_scales_as_sigma_squared():
    """E[E_contrast] = -0.5*sigma^2*sum_i 1/M_i, so d/dM_i = +0.5*sigma^2/M_i^2
    and the expected gradient is QUADRATIC in the perturbation scale."""
    g1, g2 = _mean_contrast_grad(0.25), _mean_contrast_grad(0.5)
    assert g2 / g1 == pytest.approx(4.0, rel=0.05)


def test_single_draw_gradient_is_dominated_by_the_cross_term_at_small_sigma():
    """Practical consequence for choosing sigma: for a SINGLE noise draw the
    O(sigma) cross-term dominates at small sigma, so the per-batch mass
    gradient grows ~linearly there and only approaches the sigma^2 law once
    sigma is large. Pins the crossover so it is not mistaken for a bug."""
    small = _contrast_grad(0.2) / _contrast_grad(0.1)
    large = _contrast_grad(1.0) / _contrast_grad(0.5)
    assert small < 3.0, "small-sigma regime should be sub-quadratic"
    assert large == pytest.approx(4.0, rel=0.1), "large-sigma should be quadratic"
    assert small < large


# ── mass parameterizations ───────────────────────────────────────────────
def test_softplus_parameterization_is_unchanged():
    key = jax.random.PRNGKey(1)
    a = CHLU(dim=5, hidden=8, key=key)
    b = CHLU(dim=5, hidden=8, mass_parameterization="softplus", key=key)
    np.testing.assert_array_equal(a.mass_vector(), b.mass_vector())
    np.testing.assert_allclose(
        np.asarray(a.mass_vector()),
        np.asarray(jax.nn.softplus(a.log_mass)),
        rtol=1e-6,
    )


def test_zeromean_gauge_fixes_the_overall_scale():
    """A pure common-mode shift of log_mass must not change M at all -- that
    is what makes the common mode a gauge direction rather than a lever."""
    key = jax.random.PRNGKey(2)
    m = CHLU(dim=5, hidden=8, mass_parameterization="exp_zeromean", key=key)
    shifted = eqx.tree_at(lambda t: t.log_mass, m, m.log_mass + 3.7)
    np.testing.assert_allclose(
        np.asarray(m.mass_vector()), np.asarray(shifted.mass_vector()), rtol=1e-5
    )


def test_exp_zeromean_pins_geometric_mean_to_one():
    m = CHLU(dim=6, hidden=8, mass_parameterization="exp_zeromean",
             key=jax.random.PRNGKey(3))
    mv = np.asarray(m.mass_vector())
    assert float(np.exp(np.mean(np.log(mv)))) == pytest.approx(1.0, rel=1e-5)


def test_exp_buys_more_ratio_than_softplus_at_identical_log_mass():
    """The softplus trap: softplus is linear for x >> 0, so a log-scale spread
    stops buying exponential dynamic range. exp does not have that regime."""
    key = jax.random.PRNGKey(4)
    sp = CHLU(dim=8, hidden=8, mass_parameterization="softplus", key=key)
    ex = CHLU(dim=8, hidden=8, mass_parameterization="exp", key=key)
    np.testing.assert_array_equal(sp.log_mass, ex.log_mass)  # same init
    assert _mass_ratio(np.asarray(ex.mass_vector())) > _mass_ratio(
        np.asarray(sp.mass_vector())
    )


def test_softplus_trap_worsens_with_common_mode_drift():
    """Pushing log_mass up (what energy_reg does) COMPRESSES the softplus
    spectrum toward 1 while leaving the exp spectrum invariant."""
    key = jax.random.PRNGKey(5)
    sp = CHLU(dim=8, hidden=8, mass_parameterization="softplus", key=key)
    drifted = eqx.tree_at(lambda t: t.log_mass, sp, sp.log_mass + 3.5)
    r0 = _mass_ratio(np.asarray(sp.mass_vector()))
    r1 = _mass_ratio(np.asarray(drifted.mass_vector()))
    assert r1 < r0, "common-mode drift should compress the softplus spectrum"

    ex = CHLU(dim=8, hidden=8, mass_parameterization="exp", key=key)
    ex_drift = eqx.tree_at(lambda t: t.log_mass, ex, ex.log_mass + 3.5)
    assert _mass_ratio(np.asarray(ex_drift.mass_vector())) == pytest.approx(
        _mass_ratio(np.asarray(ex.mass_vector())), rel=1e-4
    ), "exp ratio must be invariant to a common-mode shift"


# ── diagnostics plumbing ─────────────────────────────────────────────────
def test_mass_ratio_diagnostics_are_reported():
    f = _fit()
    md = f.mass_diagnostics
    for k in ("mass_ratio_init", "mass_ratio_final", "neg_momentum_scale",
              "mass_parameterization"):
        assert k in md
    assert md["mass_ratio_init"] > 1.0


def test_collect_mass_vector_applies_the_parameterization():
    f = _fit(mass_parameterization="exp_zeromean")
    mv = _collect_mass_vector(f.model)
    assert float(np.exp(np.mean(np.log(mv)))) == pytest.approx(1.0, rel=1e-4)


def test_mass_ratio_of_degenerate_spectrum_is_one():
    assert _mass_ratio(np.ones(5)) == pytest.approx(1.0)


# ── per-launch mass override (Prop 6 / OQ-B) ─────────────────────────────
# As shipped, the mass is a GLOBAL model parameter: every rollout shares it,
# so it cannot be an address component at all. These pin the minimal core
# change that makes mass a per-launch attribute the caller supplies.
def _unit(dim=4):
    return CHLU(dim=dim, hidden=16, kinetic_mode="newtonian_learned",
                key=jax.random.PRNGKey(0))


def _q0p0(dim=4):
    return (
        jnp.linspace(-0.3, 0.5, dim),
        jnp.linspace(0.05, -0.1, dim),
    )


def test_mass_override_default_is_bit_identical():
    m = _unit()
    q0, p0 = _q0p0()
    np.testing.assert_array_equal(
        np.asarray(m(q0, p0, 20, 0.1, 0.0)),
        np.asarray(m(q0, p0, 20, 0.1, 0.0, None)),
    )


def test_mass_override_with_global_mass_reproduces_global_rollout():
    m = _unit()
    q0, p0 = _q0p0()
    np.testing.assert_allclose(
        np.asarray(m(q0, p0, 20, 0.1, 0.0)),
        np.asarray(m(q0, p0, 20, 0.1, 0.0, m.mass_vector())),
        rtol=1e-6, atol=1e-7,
    )


def test_mass_override_changes_H_and_T():
    m = _unit()
    q0, p0 = _q0p0()
    heavy = m.mass_vector() * 4.0
    assert float(m.T(p0, heavy)) < float(m.T(p0))
    assert float(m.H(q0, p0, heavy)) < float(m.H(q0, p0))


def test_heavier_launch_mass_slows_the_rollout():
    """The access-key property: mass addresses TIMESCALE. A 4x heavier launch
    must move less far in the same number of steps (tau ~ sqrt(M))."""
    m = _unit()
    q0, p0 = _q0p0()
    base = np.asarray(m(q0, p0, 20, 0.1, 0.0))[-1, :4]
    heavy = np.asarray(m(q0, p0, 20, 0.1, 0.0, m.mass_vector() * 4.0))[-1, :4]
    d_base = np.linalg.norm(base - np.asarray(q0))
    d_heavy = np.linalg.norm(heavy - np.asarray(q0))
    assert d_heavy < d_base


def test_mass_override_is_differentiable_as_an_address_component():
    """The override must carry gradient, or it cannot be a LEARNED key."""
    m = _unit()
    q0, p0 = _q0p0()

    def final_disp(mass):
        traj = m(q0, p0, 10, 0.1, 0.0, mass)
        return jnp.sum(traj[-1, :4] ** 2)

    g = np.asarray(jax.grad(final_disp)(m.mass_vector()))
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) > 0.0
