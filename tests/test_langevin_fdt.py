"""Tests for the discrete-FDT Langevin noise fix (handover §7.9, F5 Prop-9).

The historical noise scale sqrt(2*gamma*T*dt) gives a stationary momentum
variance of 2*dt*T/(2-gamma) per coordinate — mass-independent and != the
Maxwell-Boltzmann M_eff_i * T. The "fdt" mode uses the exact discrete-FDT
per-mode scale sigma_i* = sqrt(M_eff_i * T * gamma * (2-gamma)), whose
stationary momentum variance is M_eff_i * T (exactly, for a harmonic mode).
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chlu.core import transforms
from chlu.core.chlu_unit import CHLU, RelativisticGibbsWarning
from chlu.core.integrators import langevin_step

# Harmonic toy: two coordinates with distinct inertial masses
_M = jnp.array([0.5, 2.0])
_K_SPRING = 1.0
_GAMMA = 0.3
_TEMPERATURE = 0.5
_DT = 0.1


def _pre_fix_effective_mass(model):
    """Verbatim pre-2026-07-09 ``CHLU.effective_mass``: raw softplus(log_mass).

    Bit-compat oracle. It ignored ``tie_channel_mass`` and omitted the ``+1e-6``
    that ``H`` inverts, so the fdt noise used a different inertia than the
    dynamics. Kept here to pin the exact blast radius of the fix.
    """
    M = jax.nn.softplus(model.log_mass)
    if model.kinetic_mode == "newtonian_identity":
        return jnp.ones(model.dim)
    if model.kinetic_mode == "newtonian_learned":
        return M
    return model.rest_mass * M


def _pre_fix_stochastic_step(model, state, dt, gamma, temperature, key, noise_mode):
    """Pre-fix ``CHLU.stochastic_step`` (``langevin_step`` itself is unchanged)."""
    q, p = state
    m_eff = _pre_fix_effective_mass(model) if noise_mode == "fdt" else None
    return langevin_step(
        model.H,
        q,
        p,
        dt,
        gamma,
        temperature,
        key,
        noise_mode=noise_mode,
        m_eff=m_eff,
        gamma_field=getattr(model, "friction_field", None),
    )


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
                    H_fn,
                    q_c,
                    p_c,
                    _DT,
                    _GAMMA,
                    _TEMPERATURE,
                    key_c,
                    noise_mode=noise_mode,
                    m_eff=m_eff,
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
    """ "fdt": stationary Var(p_i) ≈ M_eff_i * T per mode (5% tolerance)."""
    var_p = _stationary_p_var("fdt")
    expected = _M * _TEMPERATURE  # Maxwell-Boltzmann: [0.25, 1.0]
    assert jnp.allclose(var_p, expected, rtol=0.05), (
        f"Var(p)={var_p} vs Maxwell-Boltzmann {expected}"
    )


def test_legacy_noise_reproduces_fdt_mismatch():
    """ "legacy": Var(p_i) ≈ 2*T*dt/(2-gamma), mass-independent != M_eff_i*T."""
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
    """CHLU.effective_mass follows the F5 §2.1 table (I / M+eps / m0*(M+eps)).

    The ``+1e-6`` is the epsilon ``H`` actually inverts (``M_inv = 1/(M+1e-6)``),
    so the fdt noise inertia matches the dynamics inertia exactly.
    """
    key = jax.random.PRNGKey(0)
    eps = 1e-6

    identity = CHLU(dim=3, hidden=8, kinetic_mode="newtonian_identity", key=key)
    assert jnp.allclose(identity.effective_mass(), jnp.ones(3))

    learned = CHLU(dim=3, hidden=8, kinetic_mode="newtonian_learned", key=key)
    assert jnp.allclose(
        learned.effective_mass(), jax.nn.softplus(learned.log_mass) + eps
    )

    relativistic = CHLU(
        dim=3, hidden=8, rest_mass=2.0, kinetic_mode="relativistic", key=key
    )
    assert jnp.allclose(
        relativistic.effective_mass(),
        2.0 * (jax.nn.softplus(relativistic.log_mass) + eps),
    )


@pytest.mark.parametrize(
    "kinetic_mode", ["newtonian_identity", "newtonian_learned", "relativistic"]
)
@pytest.mark.parametrize("tied", [False, True])
def test_effective_mass_is_exactly_effective_inertia(kinetic_mode, tied):
    """The fdt noise inertia IS the inertia the dynamics invert (bit-exact).

    Regression for the tie_channel_mass/fdt Gibbs bug: ``effective_mass`` used
    to return raw ``softplus(log_mass)`` while ``H``/``T`` use ``mass_vector``
    (which applies the channel tie) + 1e-6.
    """
    model = CHLU(
        dim=3,
        hidden=8,
        kinetic_mode=kinetic_mode,
        rest_mass=2.0,
        tie_channel_mass=tied,
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_at(lambda m: m.log_mass, model, jnp.array([-1.0, 1.5, 0.2]))
    assert jnp.array_equal(model.effective_mass(), model.effective_inertia())


def test_effective_mass_applies_channel_tie():
    """On a tied model the fdt noise sees EQUAL channel inertias (the fix)."""
    model = CHLU(
        dim=3,
        hidden=8,
        kinetic_mode="newtonian_learned",
        tie_channel_mass=True,
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_at(lambda m: m.log_mass, model, jnp.array([-1.0, 1.5, 0.2]))

    m_eff = model.effective_mass()
    # channel coords (0, 1) are tied ...
    assert m_eff[0] == m_eff[1]
    assert jnp.allclose(m_eff[:2], model.mass_vector()[:2] + 1e-6)
    # ... and the pre-fix implementation did NOT tie them (the bug)
    assert not jnp.allclose(_pre_fix_effective_mass(model)[:2], m_eff[:2])
    # the untied coordinate is untouched by the fix (up to the 1e-6 epsilon)
    assert jnp.allclose(_pre_fix_effective_mass(model)[2], m_eff[2], atol=2e-6)


@pytest.mark.parametrize(
    "kinetic_mode", ["newtonian_identity", "newtonian_learned", "relativistic"]
)
@pytest.mark.parametrize("tied", [False, True])
def test_legacy_noise_bit_identical_after_fix(kinetic_mode, tied):
    """BIT-COMPAT: noise_mode="legacy" never reads m_eff -> unchanged, exactly.

    Holds for every kinetic mode, tied and untied.
    """
    model = CHLU(
        dim=3,
        hidden=8,
        kinetic_mode=kinetic_mode,
        rest_mass=2.0,
        tie_channel_mass=tied,
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_at(lambda m: m.log_mass, model, jnp.array([-1.0, 1.5, 0.2]))

    q, p = jnp.array([0.5, -0.5, 0.3]), jnp.array([0.1, 0.2, -0.15])
    key = jax.random.PRNGKey(2)

    q_new, p_new, _ = model.stochastic_step((q, p), 0.05, 0.2, 1.0, key)
    q_old, p_old, _ = _pre_fix_stochastic_step(
        model, (q, p), 0.05, 0.2, 1.0, key, "legacy"
    )
    assert jnp.array_equal(q_new, q_old)
    assert jnp.array_equal(p_new, p_old)


@pytest.mark.parametrize("tied", [False, True])
def test_fdt_untied_identity_bit_identical_after_fix(tied):
    """BIT-COMPAT: fdt + newtonian_identity is unchanged (M_eff = 1 either way)."""
    model = CHLU(
        dim=3,
        hidden=8,
        kinetic_mode="newtonian_identity",
        tie_channel_mass=tied,
        key=jax.random.PRNGKey(0),
    )
    q, p = jnp.array([0.5, -0.5, 0.3]), jnp.array([0.1, 0.2, -0.15])
    key = jax.random.PRNGKey(2)

    _, p_new, _ = model.stochastic_step((q, p), 0.05, 0.2, 1.0, key, noise_mode="fdt")
    _, p_old, _ = _pre_fix_stochastic_step(model, (q, p), 0.05, 0.2, 1.0, key, "fdt")
    assert jnp.array_equal(p_new, p_old)


def test_fdt_untied_learned_changes_only_by_epsilon():
    """Scope: fdt + UNTIED learned/relativistic shifts only by the +1e-6 epsilon.

    Not bit-identical (the pre-fix noise inertia omitted the epsilon that ``H``
    inverts), but the relative change is O(1e-6) -- the second half of the same
    bug. Pinned so the blast radius stays documented.
    """
    for kinetic_mode in ("newtonian_learned", "relativistic"):
        model = CHLU(
            dim=3,
            hidden=8,
            kinetic_mode=kinetic_mode,
            rest_mass=2.0,
            tie_channel_mass=False,
            key=jax.random.PRNGKey(0),
        )
        model = eqx.tree_at(lambda m: m.log_mass, model, jnp.array([-1.0, 1.5, 0.2]))
        new, old = model.effective_mass(), _pre_fix_effective_mass(model)
        rel = jnp.max(jnp.abs(new - old) / jnp.abs(old))
        assert not jnp.array_equal(new, old)
        assert rel < 1e-5, f"{kinetic_mode}: rel change {rel} exceeds the epsilon"


def test_fdt_tied_channel_restores_gibbs():
    """THE REGRESSION: on a tied model under fdt, Var(p_i) = effective_inertia_i * T.

    Momentum-variance instrument (the analyst's ``s5b_fdt_bug_direct``). For a
    separable H the momentum marginal of exp(-H/T) is exactly Gaussian with
    Var(p_i) = M_eff_i * T, independent of V, so the channel temperature ratio
    T_eff,0/T_eff,1 must be 1.

    (Documented negative: D_theta(theta_0) is the WRONG instrument -- the coset
    angle wanders within a measurement block and washes out the anisotropy.)
    """
    dim, dt, gamma, temperature = 3, 0.02, 0.3, 0.5
    n_chains, n_steps, n_burn = 512, 1500, 500

    model = CHLU(
        dim=dim,
        hidden=8,
        kinetic_mode="newtonian_learned",
        tie_channel_mass=True,
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_at(lambda m: m.log_mass, model, jnp.array([-1.0, 1.5, 0.2]))
    m_dyn = model.effective_inertia()

    def _var_p(m_eff, seed=0):
        @jax.jit
        def run(q0, p0, keys):
            def one_chain(q, p, key):
                def step_fn(carry, _):
                    q_c, p_c, key_c = carry
                    q_n, p_n, key_n = langevin_step(
                        model.H,
                        q_c,
                        p_c,
                        dt,
                        gamma,
                        temperature,
                        key_c,
                        noise_mode="fdt",
                        m_eff=m_eff,
                    )
                    return (q_n, p_n, key_n), p_n

                _, ps = jax.lax.scan(step_fn, (q, p, key), None, length=n_steps)
                return ps

            return jax.vmap(one_chain)(q0, p0, keys)

        kq, kp, kk = jax.random.split(jax.random.PRNGKey(seed), 3)
        q0 = jax.random.normal(kq, (n_chains, dim)) * 0.1
        p0 = jax.random.normal(kp, (n_chains, dim)) * 0.1
        ps = run(q0, p0, jax.random.split(kk, n_chains))
        return jnp.var(ps[:, n_burn:, :].reshape(-1, dim), axis=0)

    # FIXED path: Maxwell-Boltzmann per coordinate, equal channel temperatures.
    var_fixed = _var_p(model.effective_mass())
    assert jnp.allclose(var_fixed, m_dyn * temperature, rtol=0.05), (
        f"Var(p)={var_fixed} vs Maxwell-Boltzmann {m_dyn * temperature}"
    )
    t_eff = var_fixed / m_dyn
    assert abs(float(t_eff[0] / t_eff[1]) - 1.0) < 0.02

    # PRE-FIX path: channels equilibrate at different temperatures, and the
    # ratio tracks the (untied) noise-inertia ratio -> no Gibbs invariant.
    m_noise_old = _pre_fix_effective_mass(model)
    var_bug = _var_p(m_noise_old)
    t_eff_bug = var_bug / m_dyn
    ratio_bug = float(t_eff_bug[0] / t_eff_bug[1])
    predicted = float(m_noise_old[0] / m_noise_old[1])
    assert abs(ratio_bug - 1.0) > 0.1, "pre-fix bug should break equipartition"
    assert abs(ratio_bug - predicted) / predicted < 0.05, (
        f"bug ratio {ratio_bug} should track M_noise,0/M_noise,1 {predicted}"
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
        q,
        p,
        steps=10,
        dt=0.05,
        gamma=0.2,
        temperature=1.0,
        key=key,
        noise_mode="fdt",
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


# ---------------------------------------------------------------------------
# fix-pack-6 item 1: the sqrt(0) NaN-gradient blocker at gamma == 0
# ---------------------------------------------------------------------------


def _naive_noise_scale(m_eff, temperature, gamma):
    """Verbatim pre-fix-pack-6 fdt scale: sqrt(max(0, arg)).

    Bit-compat oracle AND the bug: d/d(m_eff) is inf*0 = NaN at arg == 0.
    """
    return jnp.sqrt(jnp.maximum(0.0, m_eff * temperature * gamma * (2.0 - gamma)))


def _safe_noise_scale(m_eff, temperature, gamma):
    """The shipped double-where safe sqrt (chlu/core/integrators.py)."""
    arg = m_eff * temperature * gamma * (2.0 - gamma)
    return jnp.where(arg > 0.0, jnp.sqrt(jnp.where(arg > 0.0, arg, 1.0)), 0.0)


@pytest.mark.parametrize("gamma", [0.0, 1e-12, 1e-8, 0.05, 0.2, 1.0, 2.0])
def test_safe_sqrt_is_bit_identical_to_naive(gamma):
    """The safe sqrt changes no value anywhere -- only the gradient at arg==0."""
    naive = _naive_noise_scale(0.7321, 0.5, gamma)
    safe = _safe_noise_scale(0.7321, 0.5, gamma)
    assert jnp.array_equal(naive, safe), f"value changed at gamma={gamma}"


def test_naive_sqrt_has_nan_gradient_at_zero_gamma():
    """Pins the ROOT CAUSE: sqrt(0) has an infinite derivative."""
    g_naive = jax.grad(lambda m: _naive_noise_scale(m, 0.5, 0.0))(0.7321)
    assert jnp.isnan(g_naive), "oracle should reproduce the pre-fix NaN"

    g_safe = jax.grad(lambda m: _safe_noise_scale(m, 0.5, 0.0))(0.7321)
    assert g_safe == 0.0, f"safe sqrt must have exactly zero gradient, got {g_safe}"


@pytest.mark.parametrize(
    "kinetic_mode", ["newtonian_identity", "newtonian_learned", "relativistic"]
)
@pytest.mark.parametrize("gamma", [0.0, 1e-12, 0.05])
@pytest.mark.parametrize("noise_mode", ["legacy", "fdt"])
def test_stochastic_step_gradient_finite_wrt_log_mass(kinetic_mode, gamma, noise_mode):
    """d(loss)/d(log_mass) through one Langevin step is finite in all 18 cells.

    Pre-fix, (fdt, gamma=0) x (newtonian_learned, relativistic) returned NaN,
    because m_eff -- which carries the learnable log_mass -- sat inside a
    sqrt() evaluated at exactly 0. This NaN'd every parameter on the first
    sleep step of train_chlu at repo defaults (sleep_friction = 0.0).
    """
    model = CHLU(dim=3, hidden=8, kinetic_mode=kinetic_mode, key=jax.random.PRNGKey(0))
    q0 = jnp.array([0.3, -0.2, 0.1])
    p0 = jnp.array([0.1, 0.2, -0.3])
    key = jax.random.PRNGKey(1)

    def loss(m):
        q, p, _ = m.stochastic_step(
            (q0, p0),
            dt=0.05,
            gamma=gamma,
            temperature=0.5,
            key=key,
            noise_mode=noise_mode,
        )
        return jnp.sum(p**2) + jnp.sum(q**2)

    with warnings.catch_warnings():  # relativistic+fdt warns by design (CM-17)
        warnings.simplefilter("ignore", RelativisticGibbsWarning)
        grads = eqx.filter_grad(loss)(model)

    assert jnp.all(jnp.isfinite(grads.log_mass)), (
        f"NaN/inf grad: {kinetic_mode}/{noise_mode}/gamma={gamma}: {grads.log_mass}"
    )


@pytest.mark.parametrize(
    "kinetic_mode", ["newtonian_identity", "newtonian_learned", "relativistic"]
)
def test_train_chlu_fdt_at_repo_defaults_stays_finite(kinetic_mode):
    """END-TO-END regression: the test that would have caught the blocker.

    ``langevin_noise="fdt"`` is the ONLY non-default here: sleep_temperature
    (0.5) and sleep_friction (0.0) come from TrainingConfig, and
    ``epoch % sleep_frequency == 0`` fires at epoch 0, so the first sleep step
    used to NaN every parameter (measured: losses[0] finite, losses[1:] NaN).
    """
    from chlu.config import get_default_config
    from chlu.training.train import train_chlu

    cfg = get_default_config()
    assert cfg.training.sleep_friction == 0.0  # the trigger, still a default
    assert cfg.training.sleep_temperature > 0.0  # sleep uses stochastic_step

    key = jax.random.PRNGKey(42)
    mkey, tkey = jax.random.split(key)
    model = CHLU(dim=2, hidden=16, kinetic_mode=kinetic_mode, key=mkey)

    t = jnp.linspace(0.0, 2.0 * jnp.pi, 80)
    data = jnp.stack(
        [jnp.sin(t), jnp.sin(t) * jnp.cos(t), jnp.cos(t), jnp.cos(2.0 * t)], axis=-1
    )

    with warnings.catch_warnings():  # relativistic+fdt warns by design (CM-17)
        warnings.simplefilter("ignore", RelativisticGibbsWarning)
        trained, losses, _ = train_chlu(
            model,
            data,
            tkey,
            config=cfg,
            epochs=3,
            sleep_steps=5,
            window_size=16,
            batch_size=8,
            langevin_noise="fdt",
        )

    losses = jnp.asarray(losses)
    assert losses.shape[0] == 3
    assert jnp.all(jnp.isfinite(losses)), f"non-finite losses: {losses}"

    leaves = jax.tree_util.tree_leaves(eqx.filter(trained, eqx.is_inexact_array))
    n_nan = sum(int(jnp.sum(jnp.isnan(leaf))) for leaf in leaves)
    assert n_nan == 0, f"{n_nan} NaN parameters after training"


# ---------------------------------------------------------------------------
# fix-pack-6 item 3: the CM-17 relativistic-Gibbs guard-rail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rest_mass,c,temperature,expected",
    [
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 5.0, 1.0, 0.04),
        (1.0, 1.0, 8.0, 8.0),
        (1.0, 0.5, 2.0, 8.0),  # (c=0.5,T=2) === (c=1,T=8): the ratio is the only knob
        (2.0, 2.0, 4.0, 0.5),
    ],
)
def test_thermal_causal_ratio(rest_mass, c, temperature, expected):
    """T/(m0 c^2) -- the single number governing the relativistic Gibbs defect."""
    model = CHLU(
        dim=2,
        hidden=4,
        kinetic_mode="relativistic",
        rest_mass=rest_mass,
        c=c,
        key=jax.random.PRNGKey(0),
    )
    assert float(model.thermal_causal_ratio(temperature)) == pytest.approx(expected)


def test_relativistic_fdt_warns_and_names_the_ratio():
    """Guard-rail fires on relativistic+fdt and reports this call's T/(m0 c^2)."""
    model = CHLU(
        dim=2,
        hidden=4,
        kinetic_mode="relativistic",
        rest_mass=1.0,
        c=5.0,
        key=jax.random.PRNGKey(0),
    )
    q, p, key = jnp.zeros(2), jnp.zeros(2), jax.random.PRNGKey(0)

    with pytest.warns(RelativisticGibbsWarning, match=r"T/\(m0\*c\^2\) = 0.04"):
        model.stochastic_step(
            (q, p), dt=0.05, gamma=0.1, temperature=1.0, key=key, noise_mode="fdt"
        )


def test_relativistic_fdt_warns_from_stochastic_rollout():
    """The Exp-C cell: annealing schedule -> warn on the hottest T (traced inside scan)."""
    model = CHLU(
        dim=2, hidden=4, kinetic_mode="relativistic", key=jax.random.PRNGKey(0)
    )
    schedule = jnp.linspace(1.0, 0.01, 5)
    with pytest.warns(RelativisticGibbsWarning, match=r"T/\(m0\*c\^2\) = 1"):
        traj = model.stochastic_rollout(
            jnp.zeros(2),
            jnp.zeros(2),
            steps=5,
            dt=0.05,
            gamma=0.1,
            temperature=schedule,
            key=jax.random.PRNGKey(1),
            noise_mode="fdt",
        )
    assert jnp.all(jnp.isfinite(traj))


@pytest.mark.parametrize(
    "kinetic_mode,noise_mode,temperature",
    [
        ("newtonian_identity", "fdt", 0.5),
        ("newtonian_learned", "fdt", 0.5),  # fdt IS exactly Gibbs here -- no warning
        ("newtonian_identity", "legacy", 0.5),
        ("newtonian_learned", "legacy", 0.5),
        ("relativistic", "legacy", 0.5),  # legacy: broken, but not by THIS defect
        ("relativistic", "fdt", 0.0),  # T=0: no noise, no sampler, no claim
    ],
)
def test_no_relativistic_gibbs_warning_elsewhere(kinetic_mode, noise_mode, temperature):
    """The guard-rail must not cry wolf on Newtonian+fdt or any legacy path."""
    model = CHLU(dim=2, hidden=4, kinetic_mode=kinetic_mode, key=jax.random.PRNGKey(0))
    q, p, key = jnp.zeros(2), jnp.zeros(2), jax.random.PRNGKey(0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RelativisticGibbsWarning)
        model.stochastic_step(
            (q, p),
            dt=0.05,
            gamma=0.1,
            temperature=temperature,
            key=key,
            noise_mode=noise_mode,
        )


# ---------------------------------------------------------------------------
# fix-pack-6 items 2 & 4: effective_mass delegation reached every consumer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kinetic_mode", ["newtonian_identity", "newtonian_learned", "relativistic"]
)
def test_transforms_effective_mass_bit_identical_for_untied(kinetic_mode):
    """Item 2: delegating transforms.effective_mass is a no-op on untied models.

    Both the old inline spelling and effective_inertia() carry the +1e-6, so
    for tie_channel_mass=False this is *strictly* bit-identical -- unlike the
    CHLU.effective_mass fix of fix-pack-5.
    """
    model = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode=kinetic_mode,
        tie_channel_mass=False,
        rest_mass=1.3,
        key=jax.random.PRNGKey(0),
    )
    old = _old_inline_transforms_effective_mass(model)
    assert jnp.array_equal(transforms.effective_mass(model), old)


@pytest.mark.parametrize("kinetic_mode", ["newtonian_learned", "relativistic"])
def test_transforms_effective_mass_applies_channel_tie(kinetic_mode):
    """The bug: on a tied model the old inline spelling ignored the tie."""
    model = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode=kinetic_mode,
        tie_channel_mass=True,
        rest_mass=1.3,
        key=jax.random.PRNGKey(0),
    )
    m_eff = transforms.effective_mass(model)
    assert jnp.array_equal(m_eff, model.effective_inertia())
    assert float(m_eff[0]) == pytest.approx(float(m_eff[1]))  # channel tied
    old = _old_inline_transforms_effective_mass(model)
    assert not jnp.allclose(m_eff, old)  # and the old spelling did NOT tie


def _old_inline_transforms_effective_mass(model):
    """Verbatim pre-fix-pack-6 ``chlu.core.transforms.effective_mass`` body."""
    if model.kinetic_mode == "newtonian_identity":
        return jnp.ones(model.dim)
    mass = jax.nn.softplus(model.log_mass) + 1e-6
    if model.kinetic_mode == "newtonian_learned":
        return mass
    return model.rest_mass * mass


@pytest.mark.parametrize("tied", [False, True])
def test_lattice_effective_mass_is_effective_inertia(tied):
    """Item 4: fix-pack-5's delegation propagated to CLULattice for free.

    ``CLULattice.effective_mass()`` concatenates per-unit ``CHLU.effective_mass()``,
    which is now an exact alias of ``effective_inertia()``. This pins the
    lattice's fdt noise scale to the inertia its dynamics invert. (Assertion
    lives here, not in lattice.py -- that file is owned elsewhere.)
    """
    from chlu.core.lattice import CLULattice

    units = tuple(
        CHLU(
            dim=4,
            hidden=8,
            kinetic_mode="newtonian_learned",
            tie_channel_mass=tied,
            key=jax.random.PRNGKey(i),
        )
        for i in range(2)
    )
    lattice = CLULattice(units=units)
    assert jnp.array_equal(lattice.effective_mass(), lattice.effective_inertia())


def test_twin_effective_mass_is_effective_inertia():
    """Item 4: the twin wrappers agree with effective_inertia, so they inherit the fix.

    ``BrokenVolumeCHLU`` delegates to the wrapped ``CHLU.effective_mass()``
    (fix-pack-5's alias), so it is tie-aware. ``UnconstrainedTwin`` wraps no
    CHLU at all -- its inertia is identically 1 -- so it was never exposed.
    """
    from chlu.core.twins import BrokenVolumeCHLU, UnconstrainedTwin

    base = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode="newtonian_learned",
        tie_channel_mass=True,
        key=jax.random.PRNGKey(0),
    )
    bv = BrokenVolumeCHLU(base)
    assert jnp.array_equal(bv.effective_mass(), bv.effective_inertia())
    assert jnp.array_equal(bv.effective_mass(), base.effective_inertia())
    # ... and the tie is genuinely applied through the wrapper
    assert float(bv.effective_mass()[0]) == pytest.approx(float(bv.effective_mass()[1]))

    twin = UnconstrainedTwin(4, hidden=16, key=jax.random.PRNGKey(3))
    assert jnp.array_equal(twin.effective_mass(), twin.effective_inertia())
