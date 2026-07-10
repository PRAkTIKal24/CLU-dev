"""Symplectic phase-space transforms for CHLU.

Implements the squeeze ("boost") operators of the F5 formalism note
(Def-6, §5.4, §7.5):

Raw squeeze, per conjugate pair i with rapidity zeta_i:

    q_i' = q_i * cosh(zeta_i) + p_i * sinh(zeta_i)
    p_i' = q_i * sinh(zeta_i) + p_i * cosh(zeta_i)

These are the non-compact (hyperbolic) directions of Sp(2d):
S^T Omega S = Omega and det S = 1 exactly, so applying a squeeze between
Verlet relaxations keeps the composed inference trajectory in the
(conformally-)symplectic class (F5 Prop-12 retry certificate).

Mass-weighted squeeze  S^(M) = N^{-1} S N  with
N = diag(M_eff^{1/2}, M_eff^{-1/2}):

    q_i' = q_i * cosh(zeta_i) + (p_i / M_eff_i) * sinh(zeta_i)
    p_i' = M_eff_i * q_i * sinh(zeta_i) + p_i * cosh(zeta_i)

Position response at zeta=0 is dq_i'/dzeta = p_i / M_eff_i: a single global
rapidity reframes light (inertial-mass) coordinates strongly and heavy ones
barely (F5 §5.4). Raw squeezes are mass-blind; per F5 the mass-weighted
version is the one to use (the raw one is kept only as a comparison flag).

Nomenclature (F5 Def-2): M_eff here is the *inertial* mass (kinetic term),
not the spectral mass mu.
"""

import jax.numpy as jnp


def effective_mass(model) -> jnp.ndarray:
    """
    Per-coordinate effective inertial mass M_eff of a CHLU at rest (p ~ 0).

    Thin delegate to ``model.effective_inertia()`` — the inertia the dynamics
    actually invert — so that the squeeze's position response
    ``dq/dzeta = p / M_eff`` matches the unit's true velocity response
    ``dq/dt = grad_p T`` at small p:

    - ``newtonian_identity``: M_eff = 1
    - ``newtonian_learned``:  M_eff = mass_vector() + 1e-6
    - ``relativistic``:       M_eff = rest_mass * (mass_vector() + 1e-6)

    where ``mass_vector() = softplus(log_mass)`` **with ``tie_channel_mass``
    applied** (channel coords (0, 1) share their log-space mean).

    HISTORY (bug fix, 2026-07-10): this free function used to inline
    ``softplus(model.log_mass) + 1e-6``, which ignored ``tie_channel_mass`` —
    while ``mass_vector``, hence ``H``/``T``, applies it. Its promise to match
    the unit's true velocity response was therefore **false on a tied model**,
    and ``mass_weighted_squeeze`` would reframe a tied checkpoint with the
    wrong inertia (a live trap for V1 boost/squeeze work). This is the
    free-function twin of the ``CHLU.effective_mass`` bug fixed in fix-pack-5.
    No shipped result was contaminated (both consumers — ``exp_v1_gate``,
    ``exp_paid_access`` — build untied models), and delegation is
    **bit-identical for untied models**: both spellings already carried +1e-6.

    Args:
        model: CHLU instance — or anything exposing ``effective_inertia()``
            (CLULattice, the twins).

    Returns:
        Array of shape (dim,) with strictly positive effective masses.
    """
    return model.effective_inertia()


def squeeze(q: jnp.ndarray, p: jnp.ndarray, zeta) -> tuple:
    """
    Raw (mass-blind) symplectic squeeze S_zeta on state (q, p).

    NOTE: per F5 §5.4 the raw squeeze is mass-blind — use
    ``mass_weighted_squeeze`` in the escalation cascade; this version exists
    for the explicit raw-vs-weighted comparison.

    Args:
        q: Position (dim,)
        p: Momentum (dim,)
        zeta: Rapidity — scalar (global boost) or per-coordinate (dim,)

    Returns:
        (q', p') transformed state.
    """
    ch = jnp.cosh(zeta)
    sh = jnp.sinh(zeta)
    q_new = q * ch + p * sh
    p_new = q * sh + p * ch
    return q_new, p_new


def mass_weighted_squeeze(
    q: jnp.ndarray, p: jnp.ndarray, zeta, m_eff: jnp.ndarray
) -> tuple:
    """
    Mass-weighted squeeze S^(M)_zeta = N^{-1} S_zeta N,
    N = diag(M_eff^{1/2}, M_eff^{-1/2})  (F5 §5.4).

    Symplectic (conjugation by the symplectic N preserves S's symplecticity),
    det = 1, and dq_i'/dzeta|_0 = p_i / M_eff_i: light coordinates reframe
    strongly, heavy ones barely.

    Args:
        q: Position (dim,)
        p: Momentum (dim,)
        zeta: Rapidity — scalar (global boost, the L0 mechanism) or (dim,)
        m_eff: Per-coordinate effective inertial mass (dim,), e.g. from
               ``effective_mass(model)``

    Returns:
        (q', p') transformed state.
    """
    ch = jnp.cosh(zeta)
    sh = jnp.sinh(zeta)
    q_new = q * ch + (p / m_eff) * sh
    p_new = (m_eff * q) * sh + p * ch
    return q_new, p_new


def squeeze_matrix(zeta, dim: int, m_eff: jnp.ndarray = None) -> jnp.ndarray:
    """
    Dense (2*dim, 2*dim) matrix of the squeeze in [q; p] coordinates.

    For verification/tests (symplecticity, determinant); the state-space
    functions above are the ones to use in dynamics code.

    Args:
        zeta: Rapidity — scalar or (dim,)
        dim: Latent dimension d
        m_eff: If given, the mass-weighted matrix N^{-1} S N; else raw S.

    Returns:
        (2*dim, 2*dim) matrix.
    """
    zeta_vec = jnp.broadcast_to(jnp.asarray(zeta), (dim,))
    ch = jnp.diag(jnp.cosh(zeta_vec))
    sh = jnp.diag(jnp.sinh(zeta_vec))
    if m_eff is None:
        top = jnp.concatenate([ch, sh], axis=1)
        bot = jnp.concatenate([sh, ch], axis=1)
    else:
        m = jnp.asarray(m_eff)
        top = jnp.concatenate([ch, sh @ jnp.diag(1.0 / m)], axis=1)
        bot = jnp.concatenate([sh @ jnp.diag(m), ch], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def symplectic_form(dim: int) -> jnp.ndarray:
    """Canonical symplectic form Omega = [[0, I], [-I, 0]] in [q; p] coords."""
    eye = jnp.eye(dim)
    zero = jnp.zeros((dim, dim))
    top = jnp.concatenate([zero, eye], axis=1)
    bot = jnp.concatenate([-eye, zero], axis=1)
    return jnp.concatenate([top, bot], axis=0)
