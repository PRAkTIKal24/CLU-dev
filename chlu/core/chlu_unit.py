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


#: Accepted values of ``CHLU.mass_parameterization`` (see ``mass_vector``).
_MASS_PARAMETERIZATIONS = frozenset(
    {"softplus", "exp", "softplus_zeromean", "exp_zeromean"}
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
    # How log_mass maps to the inertial mass M (w20 mass-visible-objective).
    # "softplus" (default) is the historical map and is bit-compatible.
    # The other three exist because softplus is LINEAR for x >> 0, so once a
    # common-mode drift pushes log_mass into that regime a log-scale spread
    # stops buying exponential dynamic range (w19: std(log M)=0.56 bought a
    # mass ratio of only 1.28). "*_zeromean" additionally centres log_mass at
    # use time, which makes the overall mass SCALE an exact gauge (theorist
    # Prop 6: only ratios are physical) and so denies any common-mode-only
    # pressure — notably energy_reg — a direction to express itself in.
    mass_parameterization: str = eqx.field(static=True)
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
        mass_parameterization: str = "softplus",
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
            mass_parameterization: map from ``log_mass`` to the inertial mass M
                           (default "softplus" = historical, bit-compatible).
                           Options:
                             "softplus"          M = softplus(x)
                             "exp"               M = exp(x)  — a log-scale
                                 spread buys EXPONENTIAL range at any offset,
                                 escaping softplus's linear regime (w20 (c)).
                             "softplus_zeromean" M = softplus(x - mean(x))
                             "exp_zeromean"      M = exp(x - mean(x)), i.e. the
                                 geometric mean of M is pinned to exactly 1.
                           The "_zeromean" variants gauge-fix the overall mass
                           SCALE at use time (constraint by reparameterization,
                           same trick as ``tie_channel_mass``), so common-mode
                           gradient pressure is projected out and only the
                           physically meaningful RATIOS can move (Prop 6).
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
        if mass_parameterization not in _MASS_PARAMETERIZATIONS:
            raise ValueError(
                f"unknown mass_parameterization {mass_parameterization!r}; "
                f"expected one of {sorted(_MASS_PARAMETERIZATIONS)}"
            )
        self.mass_parameterization = mass_parameterization

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
        # getattr guards checkpoints pickled before this field existed, which
        # must keep decoding as the historical softplus map (handover §7.13).
        mode = getattr(self, "mass_parameterization", "softplus")
        if mode.endswith("_zeromean"):
            # Centre at USE time, so the common mode is a gauge direction the
            # gradient cannot move (it is projected out of every mass gradient)
            # rather than a soft penalty that merely competes with energy_reg.
            log_m = log_m - jnp.mean(log_m)
        if mode.startswith("exp"):
            return jnp.exp(log_m)
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

    def T(self, p: jnp.ndarray, mass_override: Optional[jnp.ndarray] = None) -> float:
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
            mass_override: optional per-launch inertial mass vector (dim,)
                replacing the global ``mass_vector()`` for THIS evaluation
                (Prop 6 per-address masses). None (default) = global mass.

        Returns:
            Kinetic energy (scalar)
        """
        # Compute inertial-mass vector (always prepared, used if needed).
        # mass_vector() == softplus(log_mass), with optional channel tying
        # (bit-identical to the historical computation when untied).
        #
        # mass_override (w20, theorist Prop 6 / OQ-B) makes the mass a
        # PER-LAUNCH attribute rather than a global model parameter: the
        # address selector supplies the M for this rollout. Without it the
        # mass cannot be an address component at all, because every rollout
        # shares one M. None (default) = the trainable global mass = the
        # historical path, bit-for-bit.
        M = self.mass_vector() if mass_override is None else mass_override
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

    def H(
        self,
        q: jnp.ndarray,
        p: jnp.ndarray,
        mass_override: Optional[jnp.ndarray] = None,
    ) -> float:
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
            mass_override: optional per-launch inertial mass vector (dim,),
                forwarded to ``T`` (Prop 6 per-address masses). None
                (default) = the global trainable mass, bit-compatible.

        Returns:
            Total energy (scalar)
        """
        kinetic = self.T(p, mass_override)

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
        The dimensionless ratio ``Theta := T / (m0 c^2)`` — thermal energy over
        rest energy.

        ⚠ **Theta ALONE does NOT govern the relativistic Gibbs defect** — the
        control parameter is ``d*Theta`` (see ``gibbs_defect_parameter``). This
        method is kept because ``Theta`` is the natural per-mode scale and the
        ``d=1`` reference table below is stated in it; but at ``d>1`` the defect
        is ``d`` times larger, and Exp-C runs at ``d=784`` (CM-17 v1.9).

        In ``relativistic`` mode the exact Gibbs momentum marginal is
        Maxwell-Juttner while the coded Langevin O-step
        ``p <- (1-gamma) p + sigma xi`` is a linear OU recursion with a
        Gaussian stationary law. No sigma reconciles them (f5-corrigendum-2
        Prop-9'). At FIXED d the mismatch depends on ``T``, ``rest_mass`` and
        ``c`` only through ``Theta`` — verified exactly: ``(c=1, T=8)`` and
        ``(c=0.5, T=2)`` give bit-identical observables.

        Reference scale (**d=1 ONLY** — do NOT quote against a d>1 experiment):

        =================  ======================  ==================
        Theta = T/(m0 c^2) Var_MJ / (M_eff * T)    KL(MJ || Gauss)
        =================  ======================  ==================
        0.01               1.015                   7.4e-5 nats
        0.1                1.153                   6.8e-3 nats
        1.0                2.70                    0.384 nats
        8.0                16.28                   6.31 nats
        =================  ======================  ==================

        The non-relativistic limit is ``d*Theta << 1`` (NOT ``Theta << 1``),
        where Maxwell-Juttner tends to the Gaussian and ``fdt`` becomes exact.
        Exp-C runs at ``Theta = 1.0`` but ``d*Theta = 784``; the paper-run
        project `finalA` used ``c=5`` => ``Theta = 0.04`` but ``d*Theta = 31.4``
        — still ultra-relativistic, **NOT benign**. The exact fix is
        ``noise_mode="fdt_relativistic"`` (see ``gibbs_defect_parameter``).

        This number is meaningful for *any* kinetic mode (rest_mass and c are
        always defined), but only *governs* anything in ``relativistic`` mode.

        Args:
            temperature: Temperature T in energy units (scalar or array).

        Returns:
            T / (rest_mass * c**2), same shape as ``temperature``.
        """
        return temperature / (self.rest_mass * self.c**2)

    def gibbs_defect_parameter(self, temperature):
        """
        ``d*Theta`` — the quantity that **actually governs** the relativistic
        Gibbs defect (``Theta`` alone does NOT; CM-17 v1.9, f5-corrigendum-2
        §2).

        The coded relativistic kinetic term shares a SINGLE square root over
        all ``d`` coordinates (``T(p) = c*sqrt(sum p^2/M + (m0 c)^2)``), so the
        equilibrium's "relativistic-ness" is set by the TOTAL kinetic energy:
        ``<T_kin>/(m0 c^2) ~= d*Theta/2``. The system is non-relativistic (and
        ``fdt`` is exactly Gibbs) iff ``d*Theta << 1`` — **not** ``Theta << 1``.

        The momentum-variance defect grows with d:
            Var_MJ/(M_eff*T) = 1 + (d+2)*Theta/2 + O((d*Theta)^2)
                             -> (d+1)*Theta        (ultra-relativistic)

        So a ``d=1`` reference table (``thermal_causal_ratio``) badly
        under-states the defect at large d. At Exp-C (``d=784``, ``Theta=1``):
        ``d*Theta = 784`` and ``Var_MJ/(M_eff*T) = 785x`` (KL ~ 3.24e6 nats).
        To reach ``d*Theta < 1`` at ``T=1`` needs ``c >~ sqrt(d*T/m0) ~= 28``,
        NOT ``c=5``. The exact fix that works at any d is the latent-mass
        thermostat, ``noise_mode="fdt_relativistic"``.

        Args:
            temperature: Temperature T in energy units (scalar or array).

        Returns:
            ``dim * T / (rest_mass * c**2)`` (= ``dim * thermal_causal_ratio``).
        """
        return self.dim * self.thermal_causal_ratio(temperature)

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
        ratio = t_max / (self.rest_mass * self.c**2)  # Theta
        d_theta = self.dim * ratio  # the actual control parameter
        c_needed = float(np.sqrt(self.dim * t_max / self.rest_mass))
        warnings.warn(
            f"noise_mode='fdt' with kinetic_mode='relativistic' does NOT sample "
            f"Gibbs: the coded O-step p<-(1-gamma)p+sigma*xi is a linear OU "
            f"recursion (Gaussian stationary law), while the relativistic Gibbs "
            f"momentum marginal is Maxwell-Juttner. No sigma fixes this "
            f"(CM-17 v1.9 / f5-corrigendum-2 Prop-9'). The defect is governed by "
            f"d*Theta = {d_theta:.4g} (d={self.dim}, "
            f"Theta = T/(m0*c^2) = {ratio:.4g}, "
            f"T={t_max:.4g}, rest_mass={self.rest_mass:.4g}, c={self.c:.4g}); "
            f"Theta ALONE does not govern it, and Var_MJ/(M_eff*T) ~ (d+1)*Theta "
            f"ultra-relativistically. The EXACT fix is "
            f"noise_mode='fdt_relativistic' (latent-mass thermostat). Raising c "
            f"is NOT free at large d: reaching d*Theta<1 needs "
            f"c >~ sqrt(d*T/m0) = {c_needed:.4g} (NOT c=5). See "
            f"CHLU.gibbs_defect_parameter (d*Theta) / thermal_causal_ratio.",
            RelativisticGibbsWarning,
            stacklevel=3,
        )

    def step(
        self,
        state: tuple,
        dt: float,
        gamma: float = 0.0,
        mass_override: Optional[jnp.ndarray] = None,
    ) -> tuple:
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
        # ``mass_override is None`` is a PYTHON-level branch on a static
        # argument (not a traced value), so passing self.H unwrapped keeps the
        # default path bit-identical rather than merely numerically equal.
        H_fn = (
            self.H
            if mass_override is None
            else (lambda qq, pp: self.H(qq, pp, mass_override))
        )
        return velocity_verlet_step(
            H_fn, q, p, dt, gamma,
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
            noise_mode: "legacy" (historical sqrt(2*gamma*T*dt) scale, default),
                        "fdt" (per-mode scale sigma_i* using this unit's
                        inertial mass M_eff; F5 Prop-9), or "fdt_relativistic"
                        (the exact latent-mass thermostat). "fdt" is the exact
                        discrete-FDT noise — temperatures in energy units,
                        stationary law exp(-H/T) — **only in the Newtonian
                        kinetic modes.** In ``relativistic`` mode no sigma
                        yields a Gibbs invariant (CM-17), governed by
                        ``d*Theta`` (see ``gibbs_defect_parameter``); a
                        ``RelativisticGibbsWarning`` is emitted naming it.
                        "fdt_relativistic" preserves the Maxwell-Juttner
                        momentum marginal EXACTLY (one InvGauss draw/step) and
                        does NOT warn.

        Returns:
            (q_next, p_next, new_key): Updated state and split key
        """
        q, p = state
        self._warn_if_relativistic_fdt(temperature, noise_mode)
        m_eff = (
            self.effective_mass()
            if noise_mode in ("fdt", "fdt_relativistic")
            else None
        )
        rest_mass = self.rest_mass if noise_mode == "fdt_relativistic" else None
        c = self.c if noise_mode == "fdt_relativistic" else None
        return langevin_step(
            self.H, q, p, dt, gamma, temperature, key,
            noise_mode=noise_mode, m_eff=m_eff, rest_mass=rest_mass, c=c,
            gamma_field=getattr(self, "friction_field", None),
        )

    def __call__(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        gamma: float = 0.0,
        mass_override: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Unroll trajectory using jax.lax.scan for efficiency.

        Args:
            q0: Initial position (dim,)
            p0: Initial momentum (dim,)
            steps: Number of time steps
            dt: Time step size
            gamma: Friction coefficient
            mass_override: optional per-launch inertial mass (dim,). This is
                what makes the mass an ADDRESS component (theorist Prop 6 /
                OQ-B): the caller — an address selector — chooses the mass
                for this retrieval, so different addresses read the same
                landscape at different timescales (tau ~ sqrt(M)). None
                (default) uses the global trainable mass and is bit-identical
                to the historical rollout.

        Returns:
            Trajectory of shape (steps, 2*dim) where each row is [q, p]
        """

        def scan_fn(state, _):
            q, p = state
            q_next, p_next = self.step((q, p), dt, gamma, mass_override)
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
