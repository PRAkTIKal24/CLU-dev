"""Symplectic integrators for Hamiltonian dynamics."""

import jax
import jax.numpy as jnp


def velocity_verlet_step(
    H_fn, q: jnp.ndarray, p: jnp.ndarray, dt: float, gamma: float = 0.0
) -> tuple:
    """
    Velocity Verlet (Leapfrog) symplectic integrator.

    This integrator preserves phase space volume (det(Jacobian) = 1)
    and approximately conserves energy over long trajectories.

    Algorithm:
        1. p_half = p - 0.5 * dt * ∂H/∂q(q, p)
        2. q_next = q + dt * ∂H/∂p(q, p_half)
        3. p_next = p_half - 0.5 * dt * ∂H/∂q(q_next, p_half)

    Args:
        H_fn: Hamiltonian function H(q, p) -> scalar
        q: Position (dim,)
        p: Momentum (dim,)
        dt: Time step
        gamma: Friction coefficient (default: 0.0, no friction)

    Returns:
        (q_next, p_next): Updated state
    """
    # Compute gradients of Hamiltonian
    # ∂H/∂q and ∂H/∂p
    grad_H_q = jax.grad(H_fn, argnums=0)
    grad_H_p = jax.grad(H_fn, argnums=1)

    # Half-step momentum update
    p_half = p - 0.5 * dt * grad_H_q(q, p)

    # Full-step position update using half-step momentum
    q_next = q + dt * grad_H_p(q, p_half)

    # Half-step momentum update to complete the step
    p_next = p_half - 0.5 * dt * grad_H_q(q_next, p_half)

    # Apply friction if gamma > 0
    p_next = (1.0 - gamma) * p_next

    return q_next, p_next


def langevin_step(
    H_fn,
    q: jnp.ndarray,
    p: jnp.ndarray,
    dt: float,
    gamma: float,
    temperature: float,
    key: jax.random.PRNGKey,
    noise_mode: str = "legacy",
    m_eff: jnp.ndarray = None,
) -> tuple:
    """
    Velocity Verlet integrator with Langevin thermal noise.

    Extends the deterministic Velocity Verlet algorithm with temperature-scaled
    Gaussian noise following the fluctuation-dissipation theorem. This allows the
    system to explore energy landscapes rather than deterministically settling into
    the nearest minimum.

    Physical interpretation:
        - At temperature=0: Identical to velocity_verlet_step with friction
        - At temperature>0: Particles undergo Brownian motion, can escape local minima

    Algorithm:
        1. p_half = p - 0.5 * dt * ∂H/∂q(q, p)
        2. q_next = q + dt * ∂H/∂p(q, p_half)
        3. p_next = p_half - 0.5 * dt * ∂H/∂q(q_next, p_half)
        4. p_next = (1 - gamma) * p_next  [friction]
        5. p_next += sigma * N(0,1)  [thermal noise; sigma set by noise_mode]

    Noise modes (F5 Prop-9; handover §7.9):
        - "legacy": sigma = sqrt(2 * gamma * T * dt), uniform over coordinates.
          Violates the *discrete* fluctuation-dissipation theorem: the
          stationary momentum variance is 2*dt*T/(2-gamma) per coordinate
          instead of Maxwell-Boltzmann's M_eff_i * T, i.e. temperatures are
          not in energy units (dt and inertial mass get absorbed) and, with a
          learned non-uniform inertial mass M, each mode equilibrates at its
          own temperature (no Gibbs invariant). Default, for backward
          compatibility with existing checkpoints/annealing schedules.
        - "fdt": exact discrete-FDT noise. The damping+noise sub-step
          p' = (1-gamma)*p + sigma*xi has stationary variance
          sigma^2 / (gamma*(2-gamma)); matching Maxwell-Boltzmann
          Var(p_i) = M_eff_i * T requires the per-mode scale
              sigma_i* = sqrt(M_eff_i * T * gamma * (2 - gamma)).
          Requires ``m_eff`` (per-coordinate *inertial* mass at p≈0; see
          ``CHLU.effective_mass`` — not the spectral mass of a potential mode).

    Args:
        H_fn: Hamiltonian function H(q, p) -> scalar
        q: Position (dim,)
        p: Momentum (dim,)
        dt: Time step
        gamma: Friction coefficient (must be > 0 for temperature to have effect)
        temperature: Temperature parameter (0 = deterministic, >0 = stochastic)
        key: JAX random key for noise generation
        noise_mode: "legacy" (historical scale) or "fdt" (exact discrete FDT)
        m_eff: Per-coordinate inertial mass M_eff, shape (dim,) or scalar.
               Required for noise_mode="fdt"; ignored for "legacy".

    Returns:
        (q_next, p_next, new_key): Updated state and split key for future use
    """
    # Compute gradients of Hamiltonian
    grad_H_q = jax.grad(H_fn, argnums=0)
    grad_H_p = jax.grad(H_fn, argnums=1)

    # Half-step momentum update
    p_half = p - 0.5 * dt * grad_H_q(q, p)

    # Full-step position update using half-step momentum
    q_next = q + dt * grad_H_p(q, p_half)

    # Half-step momentum update to complete the step
    p_next = p_half - 0.5 * dt * grad_H_q(q_next, p_half)

    # Apply friction
    p_next = (1.0 - gamma) * p_next

    # Add Langevin thermal noise
    # (Here temperature already includes Boltzmann constant k.)
    # `noise_mode` is a static Python string, so this branch resolves at trace
    # time and stays jit-compatible.
    key, subkey = jax.random.split(key)
    if noise_mode == "legacy":
        # Historical scale sqrt(2 * gamma * T * dt) — see docstring caveat.
        noise_scale = jnp.sqrt(jnp.maximum(0.0, 2.0 * gamma * temperature * dt))
    elif noise_mode == "fdt":
        # Exact discrete-FDT per-mode scale sigma_i* (F5 Prop-9).
        if m_eff is None:
            raise ValueError(
                "noise_mode='fdt' requires m_eff (per-coordinate inertial "
                "mass at p≈0; see CHLU.effective_mass)."
            )
        noise_scale = jnp.sqrt(
            jnp.maximum(0.0, m_eff * temperature * gamma * (2.0 - gamma))
        )
    else:
        raise ValueError(
            f"Unknown noise_mode: {noise_mode!r}. Must be 'legacy' or 'fdt'."
        )
    # Always split key and compute noise, but use jnp.where to conditionally apply.
    # This ensures the function is traceable in JAX (no Python conditionals on traced values)
    noise = jax.random.normal(subkey, p.shape) * noise_scale
    # Apply noise only if temperature > 0 (using jnp.where for traceability)
    p_next = jnp.where(temperature > 0.0, p_next + noise, p_next)

    return q_next, p_next, key


def get_temperature_schedule(
    start: float, end: float, steps: int, schedule_type: str = "exponential"
) -> jnp.ndarray:
    """
    Generate a temperature annealing schedule.

    Creates an array of temperature values that decrease from start to end over
    the specified number of steps. Used for simulated annealing where the system
    starts hot (high exploration) and cools down (converges to minima).

    Args:
        start: Initial temperature (high for exploration)
        end: Final temperature (low for convergence)
        steps: Number of temperature values to generate
        schedule_type: Type of decay schedule
            - "exponential": T(t) = start * (end/start)^(t/steps)
                            Cools slowly at first, faster near the end
            - "linear": T(t) = start - (start-end) * (t/steps)
                       Constant cooling rate

    Returns:
        Temperature array of shape (steps,)

    Example:
        >>> temps = get_temperature_schedule(1.0, 0.01, 1000, "exponential")
        >>> temps[0]   # 1.0
        >>> temps[500] # ~0.1
        >>> temps[-1]  # 0.01
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if start <= 0 or end <= 0:
        raise ValueError(f"Temperatures must be positive, got start={start}, end={end}")

    t = jnp.linspace(0, 1, steps)

    if schedule_type == "exponential":
        # Exponential decay: T(t) = start * (end/start)^t
        # This gives slower cooling initially, faster later
        return start * jnp.power(end / start, t)

    elif schedule_type == "linear":
        # Linear decay: T(t) = start - (start - end) * t
        return start - (start - end) * t

    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}. Must be 'exponential' or 'linear'."
        )
