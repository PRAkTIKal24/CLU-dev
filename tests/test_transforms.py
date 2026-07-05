"""Tests for the symplectic squeeze transforms (F5 Def-6, §5.4)."""

import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.core.transforms import (
    effective_mass,
    mass_weighted_squeeze,
    squeeze,
    squeeze_matrix,
    symplectic_form,
)


def test_squeeze_symplectic_and_unit_det():
    """S^T Omega S = Omega and det S = 1 to 1e-12 (raw + mass-weighted)."""
    with jax.enable_x64():
        key = jax.random.PRNGKey(0)
        dim = 6
        omega = symplectic_form(dim)
        for i, zeta in enumerate(
            [0.37, -1.2, jax.random.normal(key, (dim,), dtype=jnp.float64)]
        ):
            k_m = jax.random.fold_in(key, i)
            m_eff = jnp.exp(
                jax.random.normal(k_m, (dim,), dtype=jnp.float64)
            )  # spread masses
            for m in [None, m_eff]:
                S = squeeze_matrix(zeta, dim, m_eff=m)
                err = jnp.max(jnp.abs(S.T @ omega @ S - omega))
                det = jnp.linalg.det(S)
                assert err <= 1e-12, f"symplecticity violated: {err}"
                assert jnp.abs(det - 1.0) <= 1e-12, f"det != 1: {det}"


def test_mass_weighted_position_response():
    """dq_i'/dzeta at zeta=0 equals p_i / M_eff_i on random states."""
    with jax.enable_x64():
        key = jax.random.PRNGKey(1)
        dim = 8
        kq, kp, km = jax.random.split(key, 3)
        q = jax.random.normal(kq, (dim,), dtype=jnp.float64)
        p = jax.random.normal(kp, (dim,), dtype=jnp.float64)
        m_eff = jnp.exp(jax.random.normal(km, (dim,), dtype=jnp.float64))

        def q_of_zeta(zeta):
            q_new, _ = mass_weighted_squeeze(q, p, zeta, m_eff)
            return q_new

        dq_dzeta = jax.jacfwd(q_of_zeta)(0.0)
        assert jnp.max(jnp.abs(dq_dzeta - p / m_eff)) <= 1e-10


def test_matrix_matches_functional_form():
    """The dense matrix and the state-space function agree."""
    with jax.enable_x64():
        key = jax.random.PRNGKey(2)
        dim = 5
        kq, kp, km = jax.random.split(key, 3)
        q = jax.random.normal(kq, (dim,), dtype=jnp.float64)
        p = jax.random.normal(kp, (dim,), dtype=jnp.float64)
        m_eff = jnp.exp(jax.random.normal(km, (dim,), dtype=jnp.float64))
        zeta = 0.61

        S = squeeze_matrix(zeta, dim, m_eff=m_eff)
        z_new = S @ jnp.concatenate([q, p])
        q_new, p_new = mass_weighted_squeeze(q, p, zeta, m_eff)
        assert jnp.max(jnp.abs(z_new[:dim] - q_new)) <= 1e-12
        assert jnp.max(jnp.abs(z_new[dim:] - p_new)) <= 1e-12

        S_raw = squeeze_matrix(zeta, dim)
        z_raw = S_raw @ jnp.concatenate([q, p])
        q_raw, p_raw = squeeze(q, p, zeta)
        assert jnp.max(jnp.abs(z_raw[:dim] - q_raw)) <= 1e-12
        assert jnp.max(jnp.abs(z_raw[dim:] - p_raw)) <= 1e-12


def test_effective_mass_modes():
    """effective_mass mirrors the kinematics coded in CHLU.H for all modes."""
    key = jax.random.PRNGKey(3)
    for mode in ["newtonian_identity", "newtonian_learned", "relativistic"]:
        model = CHLU(dim=4, hidden=16, rest_mass=2.0, kinetic_mode=mode, key=key)
        m_eff = effective_mass(model)
        assert m_eff.shape == (4,)
        assert jnp.all(m_eff > 0)
        softplus_m = jax.nn.softplus(model.log_mass) + 1e-6
        if mode == "newtonian_identity":
            assert jnp.allclose(m_eff, jnp.ones(4))
        elif mode == "newtonian_learned":
            assert jnp.allclose(m_eff, softplus_m)
        else:
            assert jnp.allclose(m_eff, 2.0 * softplus_m)


def test_raw_squeeze_is_mass_blind():
    """Raw and mass-weighted squeezes agree iff M = I."""
    key = jax.random.PRNGKey(4)
    kq, kp = jax.random.split(key)
    q = jax.random.normal(kq, (4,))
    p = jax.random.normal(kp, (4,))
    ones = jnp.ones(4)
    q_r, p_r = squeeze(q, p, 0.3)
    q_w, p_w = mass_weighted_squeeze(q, p, 0.3, ones)
    assert jnp.allclose(q_r, q_w) and jnp.allclose(p_r, p_w)

    m = jnp.array([0.5, 1.0, 2.0, 4.0])
    q_w2, p_w2 = mass_weighted_squeeze(q, p, 0.3, m)
    assert not jnp.allclose(q_r, q_w2)
