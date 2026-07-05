"""Tests for the discrete-FDT Langevin noise fix (handover §7.9, F5 Prop-9).

The historical noise scale sqrt(2*gamma*T*dt) gives a stationary momentum
variance of 2*dt*T/(2-gamma) per coordinate — mass-independent and != the
Maxwell-Boltzmann M_eff_i * T. The "fdt" mode uses the exact discrete-FDT
per-mode scale sigma_i* = sqrt(M_eff_i * T * gamma * (2-gamma)), whose
stationary momentum variance is M_eff_i * T (exactly, for a harmonic mode).
"""

import jax
import jax.numpy as jnp
import pytest

from chlu.core.chlu_unit import CHLU
from chlu.core.integrators import langevin_step

# Harmonic toy: two coordinates with distinct inertial masses
_M = jnp.array([0.5, 2.0])
_K_SPRING = 1.0
_GAMMA = 0.3
_TEMPERATURE = 0.5
_DT = 0.1


def _stationary_p_var(
    noise_mode: str,
    n_chains: int = 256,
    n_steps: int = 600,
    n_burn: int = 200,
    seed: int = 0,
) -> jnp.ndarray:
    """Empirical stationary Var(p_i) of the Langevin chain on the harmonic toy."""
    dim = _M.shape[0]

    def H_fn(q, p):
        # T(p) = 0.5 * p^T M^-1 p (inertial mass M), V(q) = 0.5 * k * |q|^2
        return 0.5 * jnp.sum(p * p / _M) + 0.5 * _K_SPRING * jnp.sum(q * q)

    m_eff = _M if noise_mode == "fdt" else None

    @jax.jit
    def run(q0, p0, keys):
        def one_chain(q, p, key):
            def step_fn(carry, _):
                q_c, p_c, key_c = carry
                q_n, p_n, key_n = langevin_step(
                    H_fn, q_c, p_c, _DT, _GAMMA, _TEMPERATURE, key_c,
                    noise_mode=noise_mode, m_eff=m_eff,
                )
                return (q_n, p_n, key_n), p_n

            _, ps = jax.lax.scan(step_fn, (q, p, key), None, length=n_steps)
            return ps  # (n_steps, dim)

        return jax.vmap(one_chain)(q0, p0, keys)  # (n_chains, n_steps, dim)

    kq, kp, kk = jax.random.split(jax.random.PRNGKey(seed), 3)
    q0 = jax.random.normal(kq, (n_chains, dim)) * 0.1
    p0 = jax.random.normal(kp, (n_chains, dim)) * 0.1
    keys = jax.random.split(kk, n_chains)

    ps = run(q0, p0, keys)
    samples = ps[:, n_burn:, :].reshape(-1, dim)
    return jnp.var(samples, axis=0)


def test_fdt_noise_satisfies_maxwell_boltzmann():
    """"fdt": stationary Var(p_i) ≈ M_eff_i * T per mode (5% tolerance)."""
    var_p = _stationary_p_var("fdt")
    expected = _M * _TEMPERATURE  # Maxwell-Boltzmann: [0.25, 1.0]
    assert jnp.allclose(var_p, expected, rtol=0.05), (
        f"Var(p)={var_p} vs Maxwell-Boltzmann {expected}"
    )


def test_legacy_noise_reproduces_fdt_mismatch():
    """"legacy": Var(p_i) ≈ 2*T*dt/(2-gamma), mass-independent != M_eff_i*T."""
    var_p = _stationary_p_var("legacy")
    predicted = 2.0 * _TEMPERATURE * _DT / (2.0 - _GAMMA)  # ≈ 0.0588, both modes
    maxwell_boltzmann = _M * _TEMPERATURE

    # Reproduces the F5 Prop-9 prediction...
    assert jnp.allclose(var_p, predicted, rtol=0.05), (
        f"Var(p)={var_p} vs predicted legacy variance {predicted}"
    )
    # ...which is far from the Maxwell-Boltzmann target for every mode.
    assert jnp.all(jnp.abs(var_p - maxwell_boltzmann) > 0.5 * maxwell_boltzmann)


def test_effective_mass_per_kinetic_mode():
    """CHLU.effective_mass follows the F5 §2.1 table (I / M / m0*M)."""
    key = jax.random.PRNGKey(0)

    identity = CHLU(dim=3, hidden=8, kinetic_mode="newtonian_identity", key=key)
    assert jnp.allclose(identity.effective_mass(), jnp.ones(3))

    learned = CHLU(dim=3, hidden=8, kinetic_mode="newtonian_learned", key=key)
    assert jnp.allclose(
        learned.effective_mass(), jax.nn.softplus(learned.log_mass)
    )

    relativistic = CHLU(
        dim=3, hidden=8, rest_mass=2.0, kinetic_mode="relativistic", key=key
    )
    assert jnp.allclose(
        relativistic.effective_mass(),
        2.0 * jax.nn.softplus(relativistic.log_mass),
    )


def test_stochastic_step_and_rollout_accept_noise_mode():
    """Model-level wiring: fdt runs, differs from legacy at the same key."""
    model = CHLU(
        dim=2, hidden=8, kinetic_mode="newtonian_learned", key=jax.random.PRNGKey(1)
    )
    q, p = jnp.array([0.5, -0.5]), jnp.array([0.1, 0.2])
    key = jax.random.PRNGKey(2)

    q_leg, p_leg, _ = model.stochastic_step((q, p), 0.05, 0.2, 1.0, key)
    q_fdt, p_fdt, _ = model.stochastic_step(
        (q, p), 0.05, 0.2, 1.0, key, noise_mode="fdt"
    )
    assert jnp.all(jnp.isfinite(p_leg)) and jnp.all(jnp.isfinite(p_fdt))
    # Same key, different noise scale -> different momenta (positions match:
    # the noise is applied after the position update)
    assert not jnp.allclose(p_leg, p_fdt)
    assert jnp.allclose(q_leg, q_fdt)

    traj = model.stochastic_rollout(
        q, p, steps=10, dt=0.05, gamma=0.2, temperature=1.0,
        key=key, noise_mode="fdt",
    )
    assert traj.shape == (10, 4)
    assert jnp.all(jnp.isfinite(traj))


def test_langevin_step_validates_arguments():
    def H_fn(q, p):
        return 0.5 * jnp.sum(p * p) + 0.5 * jnp.sum(q * q)

    q = p = jnp.zeros(2)
    key = jax.random.PRNGKey(0)

    with pytest.raises(ValueError, match="requires m_eff"):
        langevin_step(H_fn, q, p, 0.05, 0.2, 1.0, key, noise_mode="fdt")

    with pytest.raises(ValueError, match="Unknown noise_mode"):
        langevin_step(H_fn, q, p, 0.05, 0.2, 1.0, key, noise_mode="bogus")
