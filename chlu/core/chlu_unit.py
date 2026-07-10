"""Causal Hamiltonian Learning Unit (CHLU) - Core Implementation."""

import warnings
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.integrators import velocity_verlet_step, langevin_step
from chlu.core.potentials import (
    PotentialMLP,
    DeepPotentialMLP,
    ConvPotential,
    LinearSpurionPotential,
    SO2InvariantPotential,
    TiltedPotential,
    channel_spurion_direction,
)


class RelativisticGibbsWarning(UserWarning):
    """Emitted when ``noise_mode="fdt"`` is used with ``relativistic`` kinetics.

    The coded O-step is a linear OU recursion (Gaussian stationary law) while
    the relativistic Gibbs momentum marginal is Maxwell-Juttner: **no sigma
    gives the coded relativistic Langevin a Gibbs invariant** (CM-17;
    v2-symmetry-deepdive §7bis R8). Warn, never raise — Exp-C runs in exactly
    this cell by design, and quantifying the defect there is an open task.
    """


class CHLU(eqx.Module):
    """
    Causal Hamiltonian Learning Unit.

    A dynamical system grounded in symplectic mechanics with a relativistic
    Hamiltonian that ensures energy stability and causal bounds.

    Hamiltonian:
        H(q, p) = sqrt(p^T M p + m^2) + V(q)

    where:
        - M: learnable positive-definite mass matrix (diagonal)
        - m: rest mass constant
        - V(q): learnable potential function (MLP)
    """

    potential_net: eqx.Module  # PotentialMLP, DeepPotentialMLP, ConvPotential, SO2InvariantPotential (opt. TiltedPotential-wrapped)
    log_mass: jnp.ndarray  # Log-parameterized for positivity
    rest_mass: float = eqx.field(static=True)
    c: float = eqx.field(static=True)  # Speed of causality
    dim: int = eqx.field(static=True)
    kinetic_mode: str = eqx.field(
        static=True
    )  # "newtonian_identity", "newtonian_learned", "relativistic"
    potential_type: str = eqx.field(static=True)  # "mlp", "deep_mlp", "conv", "so2_invariant"
    # Kinetic isotropy on the SO(2) channel (coords 0, 1): tie the channel's
    # inertial masses so the kinetic term cannot explicitly break the channel
    # symmetry (F5 §4.1 — multiplet members share a common inertial mass).
    tie_channel_mass: bool = eqx.field(static=True)
    # Optional position-gated friction field gamma_phi(q) (trash regions,
    # F5 Def-5/Prop-11; chlu/core/friction_field.py). None (default) keeps the
    # historical scalar-gamma damping path bit-compatible. Access via
    # getattr(self, "friction_field", None) — checkpoints saved before this
    # field existed unpickle without it (handover §7.13 pattern).
    friction_field: Optional[eqx.Module]

    def __init__(
        self,
        dim: int,
        hidden: int = 32,
        rest_mass: float = 1.0,
        c: float = 1.0,
        kinetic_mode: str = "newtonian_identity",
        potential_type: str = "mlp",
        tie_channel_mass: bool = False,
        tilt_delta: float = 0.0,
        tilt_n: int = 1,
        spurion_delta: float = 0.0,
        spurion_angle: float = 0.0,
        friction_field: Optional[eqx.Module] = None,
        key: jax.random.PRNGKey = None,
    ):
        """
        Initialize CHLU.

        Args:
            dim: Dimensionality of position/momentum space
            hidden: Hidden units in potential network (default: 32)
            rest_mass: Rest mass constant m (default: 1.0)
            c: Speed of causality (default: 1.0)
            kinetic_mode: Kinetic energy calculation mode (default: "newtonian_identity")
                         Options: "newtonian_identity", "newtonian_learned", "relativistic"
            potential_type: Potential network architecture (default: "mlp")
                           Options: "mlp" (standard), "deep_mlp" (high-capacity),
                           "conv" (convolutional), "so2_invariant" (SO(2)-equivariant
                           channel over coords (0, 1) — F5 §4)
            tie_channel_mass: If True, tie the inertial masses of coords (0, 1)
                           (kinetic isotropy for the SO(2) channel, F5 §4.1).
                           False (default) preserves independent per-coordinate
                           masses — the "broken isotropy" switch.
            tilt_delta: Explicit SO(2)-breaking amplitude delta for an additive
                           delta*cos(n*theta) tilt on the channel (F5 §3.3c GMOR
                           probe). 0.0 (default) = no tilt (no wrapper added).
            tilt_n: Harmonic n of the tilt (default 1). Ignored if tilt_delta == 0.
            spurion_delta: Explicit SO(2)-breaking amplitude delta for a LINEAR
                           AMBIENT spurion -delta*(u.q) along the channel
                           direction u (the ChPT quark-mass term; condensate-
                           resolving GMOR probe). 0.0 (default) = no spurion
                           (no wrapper added). Unlike ``tilt_delta`` the vacuum
                           radius r* runs with delta, so mu^2, F^2 = M_ch*r*^2
                           and Sigma = r* are resolved independently and
                           mu^2 F^2 = delta*Sigma holds exactly.
            spurion_angle: Angle (rad) of u inside the channel plane
                           (u = (cos, sin, 0...)). Ignored if spurion_delta == 0.
                           Physically irrelevant for a channel-invariant V_base.
            friction_field: Optional position-gated friction field gamma_phi(q)
                           (``chlu.core.friction_field.FrictionField``; trash
                           regions, F5 Def-5). None (default) = scalar-gamma
                           damping only, bit-compatible with prior behavior.
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        k1, k2 = jax.random.split(key, 2)

        self.dim = dim
        self.rest_mass = rest_mass
        self.c = c
        self.kinetic_mode = kinetic_mode
        self.potential_type = potential_type
        if tie_channel_mass and dim < 2:
            raise ValueError(
                f"tie_channel_mass requires dim >= 2 (channel = coords (0, 1)), got dim={dim}"
            )
        self.tie_channel_mass = tie_channel_mass

        # Initialize potential network based on potential_type
        if potential_type == "conv":
            # Convolutional network for spatial structure (MNIST images)
            if dim != 784:
                raise ValueError(
                    f"ConvPotential requires dim=784 (28x28 images), got dim={dim}"
                )
            self.potential_net = ConvPotential(key=k1)
        elif potential_type == "deep_mlp":
            # High-capacity MLP for high-dimensional data
            self.potential_net = DeepPotentialMLP(dim, hidden, key=k1)
        elif potential_type == "mlp":
            # Standard MLP for low-dimensional dynamics
            self.potential_net = PotentialMLP(dim, hidden, key=k1)
        elif potential_type == "so2_invariant":
            # Exactly SO(2)-invariant potential on channel (0, 1) (F5 §4)
            self.potential_net = SO2InvariantPotential(dim, hidden, key=k1)
        else:
            raise ValueError(
                f"Unknown potential_type: {potential_type}. "
                f"Must be 'mlp', 'deep_mlp', 'conv', or 'so2_invariant'."
            )

        # Controlled explicit symmetry breaking (GMOR probe, F5 §3.3c):
        # compose the potential with an additive delta*cos(n*theta) tilt.
        if tilt_delta != 0.0:
            self.potential_net = TiltedPotential(self.potential_net, tilt_delta, tilt_n)

        # Condensate-resolving explicit breaking (GMOR proper): compose with a
        # linear ambient spurion -delta*(u.q) along the channel direction u.
        if spurion_delta != 0.0:
            if dim < 2:
                raise ValueError(
                    f"spurion_delta requires dim >= 2 (channel = coords (0, 1)), got dim={dim}"
                )
            self.potential_net = LinearSpurionPotential(
                self.potential_net,
                spurion_delta,
                channel_spurion_direction(dim, spurion_angle),
            )

        # Initialize log mass (use log for positive-definiteness via softplus)
        self.log_mass = jax.random.normal(k2, (dim,)) * 0.1

        # Optional trash-region friction field (F5 Def-5); None = no field.
        self.friction_field = friction_field

    def mass_vector(self) -> jnp.ndarray:
        """
        Per-coordinate inertial mass M = softplus(log_mass), with optional
        channel tying.

        If ``tie_channel_mass`` the channel entries (0, 1) are replaced by
        their log-space mean *at use time* (constraint by reparameterization),
        so the dynamics see exactly equal channel inertial masses regardless
        of what gradient descent does to the raw entries (F5 §4.1 kinetic
        isotropy). ``getattr`` guards checkpoints saved before this field
        existed (see handover §7.13).

        Note (F5 Def-2): this is the *inertial* mass M — do not conflate with
        the spectral mass mu (eigenvalues of M_eff^{-1} Hess V).
        """
        log_m = self.log_mass
        if getattr(self, "tie_channel_mass", False):
            tied = 0.5 * (log_m[0] + log_m[1])
            log_m = log_m.at[0].set(tied).at[1].set(tied)
        return jax.nn.softplus(log_m)

    def effective_inertia(self) -> jnp.ndarray:
        """
        Exact per-coordinate rest inertia M_eff used by this unit's dynamics
        (the inertia at p ≈ 0; F5 §2.1 table):

            newtonian_identity: 1
            newtonian_learned:  M + 1e-6
            relativistic:       rest_mass * (M + 1e-6)

        The 1e-6 is the numerical-stability epsilon inside ``H`` (the dynamics
        invert M + 1e-6, not M), included here so spectrum probes and latch
        predictions are exactly consistent with the integrated dynamics.
        """
        if self.kinetic_mode == "newtonian_identity":
            return jnp.ones(self.dim)
        M = self.mass_vector() + 1e-6
        if self.kinetic_mode == "newtonian_learned":
            return M
        elif self.kinetic_mode == "relativistic":
            return self.rest_mass * M
        else:
            raise ValueError(f"Unknown kinetic mode: {self.kinetic_mode}.")

    def T(self, p: jnp.ndarray) -> float:
        """
        Kinetic energy T(p) alone, per the configured kinetic_mode:

        - "newtonian_identity": T = 0.5 * p^2 (identity mass, classic)
        - "newtonian_learned": T = 0.5 * p^T M^-1 p (learned mass, classic)
        - "relativistic": T = sqrt(p^T M^-1 p + m^2) (learned mass, relativistic)

        Exposed separately from ``H`` so a lattice of units can assemble a
        SEPARABLE joint Hamiltonian T_net = sum_i T_i(p_i) (F5 §7.2 condition
        1) without duplicating the kinetic physics. ``H`` delegates here —
        the op sequence is identical to the historical in-``H`` computation.

        Args:
            p: Momentum (dim,)

        Returns:
            Kinetic energy (scalar)
        """
        # Compute inertial-mass vector (always prepared, used if needed).
        # mass_vector() == softplus(log_mass), with optional channel tying
        # (bit-identical to the historical computation when untied).
        M = self.mass_vector()  # Ensure positive-definite
        M_inv = 1.0 / (M + 1e-6)  # Inverse inertial mass with numerical stability

        # Select kinetic energy calculation based on mode
        if self.kinetic_mode == "newtonian_identity":
            # Classic T = 0.5 * p^2 (identity mass)
            # Best for: Lemniscate/Figure-8 to preserve geometric properties
            return 0.5 * jnp.sum(p * p)

        elif self.kinetic_mode == "newtonian_learned":
            # T = 0.5 * p^T M^-1 p (learned diagonal mass)
            # Best for: Systems with varying inertia across dimensions
            return 0.5 * jnp.sum((p * p) * M_inv)

        elif self.kinetic_mode == "relativistic":
            # T = c(sqrt(p^T M^-1 p + (mc)^2)) (relativistic with learned mass)
            # Best for: High-dimensional systems, bounded velocities, noise robustness
            p_norm_squared = jnp.sum((p * p) * M_inv)

            # Compute rest energy term
            rest_energy = (self.rest_mass * self.c) ** 2

            return self.c * jnp.sqrt(p_norm_squared + rest_energy)

        else:
            raise ValueError(
                f"Unknown kinetic mode: {self.kinetic_mode}. "
                f"Must be 'newtonian_identity', 'newtonian_learned', or 'relativistic'."
            )

    def H(self, q: jnp.ndarray, p: jnp.ndarray) -> float:
        """
        Compute the Hamiltonian with selectable kinetic energy mode.

        H(q, p) = T(p) + V(q)

        Where T(p) depends on kinetic_mode (see ``T``):
        - "newtonian_identity": T = 0.5 * p^2 (identity mass, classic)
        - "newtonian_learned": T = 0.5 * p^T M^-1 p (learned mass, classic)
        - "relativistic": T = sqrt(p^T M^-1 p + m^2) (learned mass, relativistic)

        Args:
            q: Position (dim,)
            p: Momentum (dim,)

        Returns:
            Total energy (scalar)
        """
        kinetic = self.T(p)

        # Potential energy (always computed the same way)
        potential = self.potential_net(q)

        return kinetic + potential

    def effective_mass(self) -> jnp.ndarray:
        """
        Per-coordinate *inertial* mass M_eff at p ≈ 0 (F5 Def-2 / §2.1 table).

        Exact alias of ``effective_inertia`` — the inertia the dynamics
        actually invert. Kept as a separate name because the discrete-FDT
        Langevin noise scale is documented against it (F5 Prop-9):

            sigma_i* = sqrt(M_eff_i * T * gamma * (2 - gamma)).

        This is the kinetic-term inertia (velocity = p / M_eff near rest) —
        not to be confused with the *spectral* mass mu of a potential mode.

        HISTORY (bug fix, 2026-07-09): this method used to return the raw
        ``softplus(log_mass)``, which (a) ignored ``tie_channel_mass`` — while
        ``mass_vector``, and hence ``H``/``T``, applies it — and (b) omitted the
        ``+1e-6`` that ``H`` inverts. ``stochastic_step`` builds the
        ``noise_mode="fdt"`` scale from here, so on a ``tie_channel_mass=True``
        checkpoint the noise was injected with a *different* inertia than the
        dynamics invert: each channel coordinate then equilibrated at its own
        temperature ``T_eff,i = T * M_noise,i / M_dyn,i`` and the stationary law
        was **not** ``exp(-H/T)`` — no Gibbs invariant (measured: channel
        temperature ratio off by up to 8.4%). Delegating here restores
        Maxwell-Boltzmann ``Var(p_i) = effective_inertia()_i * T``.

        Returns:
            Effective inertial mass vector of shape (dim,)
        """
        return self.effective_inertia()

    def thermal_causal_ratio(self, temperature):
        """
        The dimensionless ratio ``T / (m0 c^2)`` — thermal energy over rest
        energy — which **alone** governs the relativistic Gibbs defect (CM-17).

        In ``relativistic`` mode the exact Gibbs momentum marginal is
        Maxwell-Juttner while the coded Langevin O-step
        ``p <- (1-gamma) p + sigma xi`` is a linear OU recursion with a
        Gaussian stationary law. No sigma reconciles them
        (v2-symmetry-deepdive §7bis R8). The size of the mismatch depends on
        ``T``, ``rest_mass`` and ``c`` only through this ratio — verified
        exactly: ``(c=1, T=8)`` and ``(c=0.5, T=2)`` give bit-identical
        observables (-0.7290074).

        Reference scale (theorist's tables):

        =================  ======================  ==================
        T / (m0 c^2)       Var_MJ / (M_eff * T)    KL(MJ || Gauss)
        =================  ======================  ==================
        0.01               1.015                   7.4e-5 nats
        0.1                1.153                   6.8e-3 nats
        1.0                2.70                    0.384 nats
        8.0                16.28                   6.31 nats
        =================  ======================  ==================

        The non-relativistic limit is ``T << m0 c^2`` (ratio -> 0), where the
        Maxwell-Juttner law tends to the Gaussian and ``fdt`` becomes exact.
        The free mitigation is to raise ``c`` or ``rest_mass``. Note
        ``experiment_c`` (the published Exp III) runs at ratio 1.0; the
        paper-run project `finalA` used ``c=5`` => ratio 0.04, benign.

        This number is meaningful for *any* kinetic mode (rest_mass and c are
        always defined), but only *governs* anything in ``relativistic`` mode:
        the Newtonian modes have no m0 c^2 scale in their kinetic term and
        ``fdt`` is exactly Gibbs there.

        Args:
            temperature: Temperature T in energy units (scalar or array).

        Returns:
            T / (rest_mass * c**2), same shape as ``temperature``.
        """
        return temperature / (self.rest_mass * self.c**2)

    def _warn_if_relativistic_fdt(self, temperature, noise_mode: str) -> None:
        """Guard-rail for CM-17: warn (never raise) on relativistic + fdt.

        Fires only when noise is actually injected (T > 0) — at T=0 there is no
        sampler and hence no Gibbs claim to violate. Silently skips when
        ``temperature`` is a JAX tracer (e.g. the per-step temperature scanned
        inside ``stochastic_rollout``): the concrete value is unavailable, and
        the rollout already warns up front with the concrete schedule.

        NOTE: the concreteness probe goes through **numpy**, not ``jnp``.
        Inside a ``jit``/``filter_jit`` trace ``float(jnp.max(jnp.asarray(x)))``
        raises ``ConcretizationTypeError`` even for a genuine Python float, so a
        jnp-based probe would silently swallow the warning on exactly the paths
        that matter (``train_chlu``/``train_generative`` close over a concrete
        ``sleep_temperature`` and call ``stochastic_step`` under
        ``eqx.filter_jit``). ``np.asarray`` keeps concrete values concrete and
        raises ``TracerArrayConversionError`` on true tracers — the semantics we
        want. Warnings are emitted at trace time, so once per compilation.
        """
        if noise_mode != "fdt" or self.kinetic_mode != "relativistic":
            return
        try:
            t_max = float(np.max(np.asarray(temperature)))
        except Exception:  # traced temperature: no concrete value to report
            return
        if t_max <= 0.0:
            return
        ratio = t_max / (self.rest_mass * self.c**2)
        warnings.warn(
            f"noise_mode='fdt' with kinetic_mode='relativistic' does NOT sample "
            f"Gibbs: the coded O-step p<-(1-gamma)p+sigma*xi is a linear OU "
            f"recursion (Gaussian stationary law), while the relativistic Gibbs "
            f"momentum marginal is Maxwell-Juttner. No sigma fixes this "
            f"(CM-17 / v2-symmetry-deepdive R8). This call has "
            f"T/(m0*c^2) = {ratio:.4g} (T={t_max:.4g}, rest_mass="
            f"{self.rest_mass:.4g}, c={self.c:.4g}); the defect vanishes as this "
            f"ratio -> 0. Free mitigation: raise c or rest_mass until "
            f"T << m0*c^2. See CHLU.thermal_causal_ratio.",
            RelativisticGibbsWarning,
            stacklevel=3,
        )

    def step(self, state: tuple, dt: float, gamma: float = 0.0) -> tuple:
        """
        Single time step using Velocity Verlet integrator.

        If this unit carries a ``friction_field`` (trash regions, F5 Def-5),
        the damping picks up the position-gated factor
        (1 - gamma_phi(q_next)), composed multiplicatively with the scalar
        gamma. Without a field the path is bit-compatible with the historical
        integrator.

        Args:
            state: (q, p) tuple
            dt: Time step

        Returns:
            (q_next, p_next): Updated state
        """
        q, p = state
        return velocity_verlet_step(
            self.H, q, p, dt, gamma,
            gamma_field=getattr(self, "friction_field", None),
        )

    def stochastic_step(
        self,
        state: tuple,
        dt: float,
        gamma: float,
        temperature: float,
        key: jax.random.PRNGKey,
        noise_mode: str = "legacy",
    ) -> tuple:
        """
        Single stochastic time step using Langevin dynamics.

        Adds temperature-scaled Gaussian noise to enable exploration of the
        energy landscape. The system can escape local minima and discover
        multiple modes in the distribution.

        Args:
            state: (q, p) tuple
            dt: Time step
            gamma: Friction coefficient (required for temperature to have effect)
            temperature: Temperature parameter (0 = deterministic, >0 = stochastic)
            key: JAX random key for reproducible noise generation
            noise_mode: "legacy" (historical sqrt(2*gamma*T*dt) scale, default)
                        or "fdt" (per-mode scale sigma_i* using this unit's
                        inertial mass M_eff; F5 Prop-9). "fdt" is the exact
                        discrete-FDT noise — i.e. temperatures in energy units,
                        stationary law exp(-H/T) — **only in the Newtonian
                        kinetic modes.** In ``relativistic`` mode no sigma
                        yields a Gibbs invariant (CM-17); a
                        ``RelativisticGibbsWarning`` is emitted, naming this
                        call's ``T/(m0 c^2)``. See ``thermal_causal_ratio``.

        Returns:
            (q_next, p_next, new_key): Updated state and split key
        """
        q, p = state
        self._warn_if_relativistic_fdt(temperature, noise_mode)
        m_eff = self.effective_mass() if noise_mode == "fdt" else None
        return langevin_step(
            self.H, q, p, dt, gamma, temperature, key,
            noise_mode=noise_mode, m_eff=m_eff,
            gamma_field=getattr(self, "friction_field", None),
        )

    def __call__(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        gamma: float = 0.0,
    ) -> jnp.ndarray:
        """
        Unroll trajectory using jax.lax.scan for efficiency.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """

        def scan_fn(state, _):
            q, p = state
            q_next, p_next = self.step((q, p), dt, gamma)
            # Concatenate q and p for output
            output = jnp.concatenate([q_next, p_next])
            return (q_next, p_next), output

        # Run scan
        _, trajectory = jax.lax.scan(scan_fn, (q0, p0), None, length=steps)

        return trajectory

    def governed_rollout(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        target_energy: float,
        sensitivity: float = 1.0,
    ) -> jnp.ndarray:
        """
        Unroll trajectory with energy-based governor (active limit cycle control).

        The governor dynamically adjusts friction based on energy error:
        - If current_energy > target_energy (noisy): Apply positive friction (brake)
        - If current_energy < target_energy (damped): Coast (maintain energy)

        This creates a Van der Pol-like limit cycle attractor at the target energy.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size
            target_energy: Target Hamiltonian energy (learned from training data)
            sensitivity: Governor sensitivity (default: 1.0). Controls correction speed.

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """

        def scan_fn(state, _):
            q, p = state

            # Compute current energy
            current_energy = self.H(q, p)

            # Energy error: positive if above target (noise), negative if below (damped)
            energy_error = current_energy - target_energy

            # Symmetric control: tanh clamps to [-1, 1] for stability
            # Positive error → positive gamma (friction/brake)
            # Negative error → zero gamma (frictionless coasting)
            gamma = sensitivity * jnp.tanh(jnp.maximum(0, energy_error))

            # Step with dynamic gamma
            q_next, p_next = self.step((q, p), dt, gamma)

            # Concatenate q and p for output
            output = jnp.concatenate([q_next, p_next])
            return (q_next, p_next), output

        # Run scan for steps-1, then prepend initial condition
        # This ensures output length = steps and includes (q0, p0) as first point
        _, trajectory = jax.lax.scan(scan_fn, (q0, p0), None, length=steps - 1)

        # Prepend initial condition to match LSTM/NODE behavior
        initial_state = jnp.concatenate([q0, p0])[None, :]
        return jnp.concatenate([initial_state, trajectory], axis=0)

    def stochastic_rollout(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        gamma: float,
        temperature: float | jnp.ndarray,
        key: jax.random.PRNGKey,
        noise_mode: str = "legacy",
    ) -> jnp.ndarray:
        """
        Unroll stochastic trajectory using Langevin dynamics.

        This method uses jax.lax.scan for efficiency while properly threading
        the random key through each step to ensure reproducible stochastic behavior.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size
            gamma: Friction coefficient
            temperature: Temperature parameter for thermal noise.
                        Can be scalar (constant) or array of shape (steps,) for annealing.
            key: JAX random key for reproducible stochastic evolution
            noise_mode: "legacy" (historical scale, default) or "fdt"
                        (per-mode discrete-FDT scale; see F5 Prop-9). "fdt" is
                        exactly Gibbs **only in the Newtonian kinetic modes**;
                        in ``relativistic`` mode it is not, for any sigma
                        (CM-17) — a ``RelativisticGibbsWarning`` is emitted,
                        naming the hottest ``T/(m0 c^2)`` of the schedule.

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """
        # CM-17 guard-rail, raised here (not only in stochastic_step) because
        # the per-step temperature is a tracer inside the scan below.
        self._warn_if_relativistic_fdt(temperature, noise_mode)

        # Convert temperature to array schedule
        # If scalar, broadcast to constant schedule
        temp_schedule = jnp.atleast_1d(temperature)
        if temp_schedule.shape[0] == 1:
            temp_schedule = jnp.repeat(temp_schedule, steps)
        elif temp_schedule.shape[0] != steps:
            raise ValueError(f"Temperature schedule length {temp_schedule.shape[0]} must match steps {steps}")

        def scan_fn(carry, temp_t):
            q, p, key_state = carry
            q_next, p_next, new_key = self.stochastic_step(
                (q, p), dt, gamma, temp_t, key_state, noise_mode=noise_mode
            )
            # Concatenate q and p for output
            output = jnp.concatenate([q_next, p_next])
            return (q_next, p_next, new_key), output

        # Run scan with key threading, iterating over temperature schedule
        _, trajectory = jax.lax.scan(scan_fn, (q0, p0, key), temp_schedule)

        return trajectory
