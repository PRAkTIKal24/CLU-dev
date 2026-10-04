"""Symplectic integrators for Hamiltonian dynamics."""

import jax
import jax.numpy as jnp


def _sample_inverse_gaussian(key, mean, shape_lambda):
    """Draw one scalar from InverseGaussian(mean=``mean``, shape=``shape_lambda``).

    Michael, Schucany & Haas (1976) exact transform — closed-form, jax-able,
    two uniforms/normals per draw. Used by the ``fdt_relativistic`` thermostat
    to sample the latent scale ``s|p`` of the Maxwell–Juttner Gaussian scale
    mixture (f5-corrigendum-2 §3). ``mean`` and ``shape_lambda`` are positive
    scalars.
    """
    k1, k2 = jax.random.split(key)
    mu = mean
    lam = shape_lambda
    nu = jax.random.normal(k1)
    y = nu * nu
    x = mu + (mu * mu * y) / (2.0 * lam) - (mu / (2.0 * lam)) * jnp.sqrt(
        4.0 * mu * lam * y + mu * mu * y * y
    )
    u = jax.random.uniform(k2)
    return jnp.where(u <= mu / (mu + x), x, (mu * mu) / x)


def velocity_verlet_step(
    H_fn, q: jnp.ndarray, p: jnp.ndarray, dt: float, gamma: float = 0.0,
    gamma_field=None,
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
        gamma_field: Optional position-gated friction field gamma_phi(q)
            (callable q -> scalar in [0, 1), e.g. ``FrictionField``). When
            present, damping picks up the extra factor
            (1 - gamma_phi(q_next)) evaluated at the POST-step position
            (F5 Def-5); composes multiplicatively with the scalar gamma
            (gamma=0 reduces exactly to Def-5). Prop-11: contributes exactly
            (1 - gamma_phi(q_next))^dim to det(Jacobian) — volume destroyed
            only inside the horizons. Default None = historical scalar-gamma
            path, bit-compatible.

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

    # Position-gated friction field (trash regions), F5 Def-5: evaluate at
    # q_{n+1}, after the Verlet substeps. Python-level branch => the default
    # None path traces identically to the historical integrator.
    # [S2 re-emission hook: local Hawking noise would inject here, with a
    #  scale tied to gamma_field(q_next) — out of scope for this build.]
    if gamma_field is not None:
        p_next = (1.0 - gamma_field(q_next)) * p_next

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
    rest_mass=None,
    c=None,
    group_sizes=None,
    gamma_field=None,
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
        - "fdt": exact discrete-FDT noise **in the Newtonian kinetic modes
          only** (CM-17; see the kinetic-mode caveat below). The damping+noise
          sub-step p' = (1-gamma)*p + sigma*xi has stationary variance
          sigma^2 / (gamma*(2-gamma)); matching Maxwell-Boltzmann
          Var(p_i) = M_eff_i * T requires the per-mode scale
              sigma_i* = sqrt(M_eff_i * T * gamma * (2 - gamma)).
          Requires ``m_eff`` (per-coordinate *inertial* mass at p≈0; see
          ``CHLU.effective_mass`` — not the spectral mass of a potential mode).
          Then temperatures ARE in energy units and the stationary law is the
          Gibbs measure exp(-H/T).

    ⚠ Kinetic-mode caveat (CM-17 v1.9; f5-corrigendum-2 Prop-9', proven):
        "fdt" gives a Gibbs invariant **only for kinetic_mode in
        {newtonian_identity, newtonian_learned}**. The O-step coded above,
        p <- (1-gamma)*p + sigma*xi, is an autonomous *linear* OU recursion,
        so its stationary momentum law is exactly *Gaussian*. In
        ``relativistic`` mode the Gibbs momentum marginal is Maxwell-Juttner,
        which is not Gaussian — hence **no sigma whatsoever makes the coded
        relativistic Langevin sample Gibbs.** Root cause: the Gibbs-preserving
        underdamped Langevin damps the *velocity* grad_p T; this code damps
        *p*. For Newtonian T these coincide (Gamma = gamma*M); for
        T(p) = c*sqrt(p^T M^-1 p + (m0 c)^2), grad_p T ∝ p / T(p) and they do
        not.

        **CONTROL PARAMETER IS d*Theta, NOT Theta** (f5-corrigendum-2 §2).
        T(p) shares ONE square root over all d coordinates, so the defect is
        set by the TOTAL kinetic energy: Var_MJ/(M_eff*T) = 1 + (d+2)Theta/2
        -> (d+1)*Theta ultra-relativistically, with Theta := T/(m0 c^2).
        The d=1 table (Var_MJ/(M_eff*T) = 1.015 / 1.153 / 2.70 / 16.28,
        KL = 7.4e-5 / 6.8e-3 / 0.384 / 6.31 nats at Theta = 0.01/0.1/1/8) is
        **d=1 ONLY** — do NOT quote it against Exp-C, which runs at d=784,
        where Theta=1 gives d*Theta=784 and Var_MJ/(M_eff*T)=785x
        (KL ~ 3.24e6 nats). See ``CHLU.gibbs_defect_parameter`` (d*Theta) and
        ``CHLU.thermal_causal_ratio`` (Theta).

        **Mitigations (fix ranking, replaces the refuted "raise c is free"):**
        - The EXACT cheap fix is ``noise_mode="fdt_relativistic"`` below (the
          latent-mass thermostat) — preserves MJ exactly, one InvGauss draw
          per step.
        - Raising ``c``/``rest_mass`` only shrinks d*Theta approximately and is
          NOT free at large d: to reach d*Theta < 1 at Exp-C (T=1) needs
          c >~ sqrt(d*T/m0) ~= 28, NOT c=5 (c=5 leaves d*Theta = 31.4). It also
          moves the chain off the velocity-saturation plateau (making the
          residual error dynamically visible) and empirically degrades Exp-C
          generation — so it is NOT a paper fix (CM-17 v1.9). `finalA` (c=5)
          is at d*Theta = 31.4, i.e. still ultra-relativistic — NOT "benign".
        - Metropolis adjustment is the only route to exact Gibbs in H.
        ``CHLU.stochastic_step`` warns (does not raise) on relativistic+fdt,
        naming d*Theta.

    "fdt_relativistic" — the exact relativistic thermostat (f5-corrigendum-2
        §3, F2). Maxwell-Juttner is a Gaussian scale mixture:
            p | s ~ N(0, M/(2s)),
            s | p ~ InvGauss(mean = c^2/(2 T T(p)), shape = c^2/(2 T^2)).
        Draw ONE scalar s|p per relativistic unit (H is kinetic-separable, so
        MJ factorizes per unit) and run the *same* linear Gaussian O-step with
        variance M/(2s) — this preserves the MJ momentum marginal **exactly**
        (Prop-9's sigma* is its d*Theta -> 0 limit). Physically: the exact
        relativistic FDT noise is the coded Gaussian noise with a *randomized
        inertia* equal to the relativistic mass m0*gamma_Lorentz. Requires
        ``m_eff``, ``rest_mass`` and ``c``; ``group_sizes`` (per-unit dims) is
        used by the lattice so each relativistic unit gets its own s. Because
        it keeps momentum persistence it **dominates** an exact Maxwell-Juttner
        momentum refresh (which decorrelates p every step and slows q-mixing).
        No ``RelativisticGibbsWarning`` is emitted for this mode.

    Args:
        H_fn: Hamiltonian function H(q, p) -> scalar
        q: Position (dim,)
        p: Momentum (dim,)
        dt: Time step
        gamma: Friction coefficient (must be > 0 for temperature to have effect)
        temperature: Temperature parameter (0 = deterministic, >0 = stochastic)
        key: JAX random key for noise generation
        noise_mode: "legacy" (historical scale), "fdt" (exact discrete FDT,
               Newtonian only), or "fdt_relativistic" (exact latent-mass
               thermostat for relativistic mode; see the caveat above).
        m_eff: Per-coordinate inertial mass M_eff, shape (dim,) or scalar.
               Required for noise_mode="fdt" and "fdt_relativistic"; ignored
               for "legacy".
        rest_mass: Rest mass m0 (scalar, or per-coordinate array for a lattice
               of heterogeneous units). Required for "fdt_relativistic".
        c: Speed of causality (scalar, or per-coordinate array). Required for
               "fdt_relativistic".
        group_sizes: Optional static tuple of per-unit dims (sum == dim). For
               "fdt_relativistic" each group gets its own latent scale s (H is
               kinetic-separable, so MJ factorizes per unit). None (default) =
               one group over the whole vector (the single-CHLU case).
        gamma_field: Optional position-gated friction field gamma_phi(q)
               (F5 Def-5; see velocity_verlet_step). Applied at q_next after
               the scalar friction and BEFORE the thermal noise. NOTE: the
               field friction is deliberately NOT coupled to the noise scale —
               a pure sink (absorb-only). Coupling it (localized bath /
               "Hawking re-emission") is the S2 study hook.

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

    # Momentum ENTERING the O-step (post-Verlet, PRE-friction): the latent
    # scale s|p of the fdt_relativistic thermostat is drawn from this p.
    p_verlet = p_next

    # Apply friction
    p_next = (1.0 - gamma) * p_next

    # Position-gated friction field (F5 Def-5): after scalar friction,
    # before the thermal noise. Absorb-only (S2 re-emission hook point).
    if gamma_field is not None:
        p_next = (1.0 - gamma_field(q_next)) * p_next

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
        # DESIGN NOTE (do not "fix"): the scale uses the *scalar* gamma only,
        # never gamma_field(q_next). The shipped FrictionField is deliberately
        # ABSORB-ONLY (a pure sink), so a friction hole is simultaneously a
        # brake and a *refrigerator* (T_local = 1.26e-4 vs 1e-3 outside).
        # v5-gate §R3 measured this: absorb-only makes a hole a 107.77 ± 4.78x
        # memory vault, where the "correct-looking" locally-thermalized
        # (coupled-bath) form gives only 13.28 ± 0.12x — the coupled-bath
        # hypothesis was REJECTED by a factor 8.11 ± 0.37 against its own
        # dedicated control. Coupling the noise to gamma_field would destroy
        # the effect. The S2 "Hawking re-emission" study is the place to
        # explore the coupled bath, behind its own flag.
        #
        # Safe sqrt (double-where): the naive sqrt(max(0, arg)) has an
        # infinite derivative at arg == 0, and m_eff carries the LEARNABLE
        # log_mass — so at gamma == 0 (the repo default sleep_friction) the
        # gradient d sigma / d log_mass is inf * 0 = NaN, which poisons every
        # parameter on the first sleep step. "legacy" is immune only because
        # its sqrt argument (2*gamma*T*dt) contains no parameter. The
        # double-where is bit-identical for arg > 0 and gives exactly 0 value
        # AND 0 gradient at arg == 0. (xy-lattice-theory §5(i-b).)
        arg = m_eff * temperature * gamma * (2.0 - gamma)
        safe_arg = jnp.where(arg > 0.0, arg, 1.0)
        noise_scale = jnp.where(arg > 0.0, jnp.sqrt(safe_arg), 0.0)
    elif noise_mode == "fdt_relativistic":
        # Exact latent-mass thermostat for relativistic mode (f5-corrigendum-2
        # §3). MJ is a Gaussian scale mixture: p|s ~ N(0, M/(2s)),
        # s|p ~ InvGauss(mean = c/(2 T sqrt(u+(m0 c)^2)), shape = c^2/(2 T^2)),
        # u = p^T M^-1 p. Drawing s|p and running the same OU O-step with
        # stationary variance M/(2s) preserves the MJ momentum marginal exactly.
        if m_eff is None or rest_mass is None or c is None:
            raise ValueError(
                "noise_mode='fdt_relativistic' requires m_eff, rest_mass and c "
                "(see CHLU.stochastic_step / CLULattice.stochastic_step)."
            )
        # Per-coordinate BARE inertia the dynamics invert: M+1e-6 = m_eff/m0.
        rm_arr = jnp.broadcast_to(jnp.asarray(rest_mass, p.dtype), p.shape)
        c_arr = jnp.broadcast_to(jnp.asarray(c, p.dtype), p.shape)
        m_inertia = m_eff / rm_arr
        # T=0 is masked out at the very end; keep beta finite so no NaN leaks
        # into the gradient (double-where discipline, cf. the "fdt" branch).
        t_safe = jnp.where(temperature > 0.0, temperature, 1.0)
        # Group structure: one shared latent scale s per relativistic unit.
        if group_sizes is None:
            bounds = ((0, p.shape[0]),)
        else:
            bounds = []
            _start = 0
            for _g in group_sizes:
                bounds.append((_start, _start + int(_g)))
                _start += int(_g)
            bounds = tuple(bounds)
        arg = jnp.zeros_like(p)
        for a, b in bounds:
            m_g = m_inertia[a:b]
            p_g = p_verlet[a:b]
            u_g = jnp.sum(p_g * p_g / m_g)  # p^T M^-1 p over this unit
            rm_g = rm_arr[a]
            c_g = c_arr[a]
            beta_g = c_g / t_safe
            tp_over_c = jnp.sqrt(u_g + (rm_g * c_g) ** 2)  # = T(p)/c
            mean_s = beta_g / (2.0 * tp_over_c)
            lam_s = beta_g * beta_g / 2.0
            key, s_key = jax.random.split(key)
            s_g = _sample_inverse_gaussian(s_key, mean_s, lam_s)  # scalar
            # OU noise variance targeting N(0, M/(2s)):
            #   sigma^2 = (1-(1-gamma)^2) * M/(2s) = gamma(2-gamma) * M/(2s)
            arg = arg.at[a:b].set(gamma * (2.0 - gamma) * m_g / (2.0 * s_g))
        # Safe sqrt (double-where): 0 value AND 0 gradient at gamma == 0, where
        # m_g carries the learnable log_mass (cf. the "fdt" branch NaN note).
        safe_arg = jnp.where(arg > 0.0, arg, 1.0)
        noise_scale = jnp.where(arg > 0.0, jnp.sqrt(safe_arg), 0.0)
    else:
        raise ValueError(
            f"Unknown noise_mode: {noise_mode!r}. Must be 'legacy', 'fdt' or "
            f"'fdt_relativistic'."
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
