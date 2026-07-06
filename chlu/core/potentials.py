"""Learnable potential energy functions for CHLU."""

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class PotentialMLP(eqx.Module):
    """
    V(q) - Learnable potential energy function.

    A simple MLP that maps position q to scalar potential energy.
    Architecture: Linear(dim→hidden) → tanh → Linear(hidden→hidden) → tanh → Linear(hidden→1)
    """

    layers: list

    def __init__(self, dim: int, hidden: int = 32, key: jax.random.PRNGKey = None):
        """
        Initialize the potential network.

        Args:
            dim: Dimensionality of position space
            hidden: Number of hidden units (default: 32)
            key: JAX random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, 3)

        self.layers = [
            eqx.nn.Linear(dim, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, 1, key=keys[2]),
        ]

    def __call__(self, q: jnp.ndarray) -> float:
        """
        Compute potential energy V(q).

        Args:
            q: Position vector (dim,)

        Returns:
            Scalar potential energy
        """
        x = q
        # First layer + activation
        x = self.layers[0](x)
        x = jnp.tanh(x)

        # Second layer + activation
        x = self.layers[1](x)
        x = jnp.tanh(x)

        # Output layer (scalar)
        x = self.layers[2](x)

        # Neural potential
        v_n = jnp.squeeze(x)

        # # Neural potential
        # v_n = 0.5 * jnp.sum(h**2)  # This is bounded (max 0.5 * dim)

        # Global confinement potential
        # This ensures V(q) -> inf as q -> inf
        v_g = 0.05 * jnp.sum(q**2)

        return v_n + v_g


class DeepPotentialMLP(eqx.Module):
    """
    High-Capacity Potential for 784-dim MNIST.
    Architecture: 784 -> 1024 -> 1024 -> 1024 -> 1
    Activation: Swish (SiLU) for better gradient flow.
    """

    layers: list

    def __init__(self, dim: int, hidden: int = 1024, key: jax.random.PRNGKey = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # 3 Hidden Layers (Depth = Sharpness)
        self.layers = [
            eqx.nn.Linear(dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, hidden, key=k3),
            eqx.nn.Linear(hidden, 1, key=k4),  # Output scalar Energy
        ]

    def __call__(self, q: jnp.ndarray) -> float:
        x = q

        # Swish Activation (x * sigmoid(x))
        # Much better for Physics/EBMs than tanh or ReLU
        x = jnn.swish(self.layers[0](x))
        x = jnn.swish(self.layers[1](x))
        x = jnn.swish(self.layers[2](x))

        # Final projection to Scalar Energy
        x = self.layers[3](x)
        v_n = jnp.squeeze(x)

        # CRITICAL CHANGE: NO GLOBAL CONFINEMENT (v_g)
        # We rely on jnp.clip(q, -1, 1) in the step function for boundaries.
        # We rely on L2 Regularization in the loss for stability.

        return v_n


class ConvPotential(eqx.Module):
    """
    Convolutional Potential for MNIST.
    Learns 'Local Physics' (edges, strokes) instead of 'Global Pixels'.
    Architecture: Conv layers to detect edges -> strokes -> curves -> digits -> scalar energy
    """

    layers: list

    def __init__(self, key: jax.random.PRNGKey):
        """
        Initialize convolutional potential network.

        Args:
            key: JAX random key for initialization
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.layers = [
            # Layer 1: Detect Edges (Strokes)
            # Input: 1 channel (Greyscale), Output: 16 Features
            eqx.nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1, key=k1),
            # Layer 2: Assemble Strokes into Curves
            eqx.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, key=k2),
            # Layer 3: Assemble Curves into Digits
            eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, key=k3),
            # Layer 4: Global Energy Assessment
            # Flatten -> Linear -> Scalar Energy
            eqx.nn.Linear(64 * 3 * 3, 1, key=k4),
        ]

    def __call__(self, q: jnp.ndarray) -> float:
        """
        Compute potential energy V(q) from image pixels.

        Args:
            q: Flattened image vector (784,)

        Returns:
            Scalar potential energy
        """
        # Reshape flat 784 -> Image (1, 28, 28)
        x = q.reshape(1, 28, 28)

        # Conv Operations with Swish (Smooth Physics)
        x = jnn.swish(self.layers[0](x))  # -> 14x14x16
        x = jnn.swish(self.layers[1](x))  # -> 7x7x32
        x = jnn.swish(self.layers[2](x))  # -> 4x4x64

        # Flatten and Project to Energy
        x = x.ravel()
        E = self.layers[3](x)

        # CRITICAL: Scale down by 100.0 to keep energies in reasonable range.
        # The ConvPotential sums outputs of thousands of neurons.
        # Without this scaling, energy magnitudes explode (e.g., -8000)
        # and temperature/noise parameters become ineffective.
        return jnp.squeeze(E) / 100.0


class SO2InvariantPotential(eqx.Module):
    """
    Exactly SO(2)-invariant potential over a designated channel pair (F5 §4).

    V(q) = f_theta(r^2) + alpha * r^2 + g_theta(q_spec)

    The channel is coordinates (0, 1) by convention; r^2 = q0^2 + q1^2 is the
    generating polynomial invariant of SO(2) on the channel plane. The learned
    radial profile f_theta is a small MLP fed r^2 rather than r: every smooth
    SO(2)-invariant function of the channel is a smooth function of r^2,
    whereas an MLP on r could express a conical cusp at the origin (breaking
    the Hessian-based spectrum probes). Spectator coordinates (2..dim-1) are
    governed by a standard ``PotentialMLP`` (which carries its own confinement
    term), additively — so channel and spectators do not mix in V.

    The alpha * r^2 confinement on the channel keeps the total potential
    coercive (F5 Prop-10 assumption A1 holds architecturally) and is itself
    SO(2)-invariant, so it cannot lift the angular flat direction: a flat
    (Goldstone) direction along a vacuum circle is exact by construction and
    protected under discretization (F5 Prop-8).

    Explicit symmetry breaking is deliberately NOT part of this module —
    compose with ``TiltedPotential`` (the F5 §3.3c GMOR probe).
    """

    radial_layers: list
    spectator_net: PotentialMLP | None
    dim: int = eqx.field(static=True)
    confinement: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden: int = 32,
        confinement: float = 0.05,
        key: jax.random.PRNGKey = None,
    ):
        """
        Args:
            dim: Total dimensionality (>= 2). Channel = coords (0, 1),
                 spectators = coords (2..dim-1).
            hidden: Hidden units for both the radial MLP and the spectator MLP.
            confinement: alpha for the invariant alpha * r^2 channel
                 confinement (default 0.05, matching PotentialMLP).
            key: JAX random key.
        """
        if dim < 2:
            raise ValueError(
                f"SO2InvariantPotential requires dim >= 2 (one channel pair), got dim={dim}"
            )
        if key is None:
            key = jax.random.PRNGKey(0)

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.dim = dim
        self.confinement = confinement
        # Learned radial profile f_theta(r^2): 1 -> hidden -> hidden -> 1 (tanh)
        self.radial_layers = [
            eqx.nn.Linear(1, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, 1, key=k3),
        ]
        # Non-symmetric spectator dims: standard MLP (has its own confinement)
        self.spectator_net = (
            PotentialMLP(dim - 2, hidden, key=k4) if dim > 2 else None
        )

    def __call__(self, q: jnp.ndarray) -> float:
        """Compute V(q). Exactly invariant under rotations of (q0, q1)."""
        q_ch = q[:2]
        r2 = jnp.sum(q_ch * q_ch)

        x = jnp.tanh(self.radial_layers[0](r2[None]))
        x = jnp.tanh(self.radial_layers[1](x))
        v = jnp.squeeze(self.radial_layers[2](x))

        # Invariant channel confinement (coercivity, F5 Prop-10 A1)
        v = v + self.confinement * r2

        if self.spectator_net is not None:
            v = v + self.spectator_net(q[2:])

        return v


class TiltedPotential(eqx.Module):
    """
    Controlled explicit SO(2) breaking: V(q) + delta * cos(n * theta).

    The F5 §3.3c GMOR probe — an additive tilt along the channel angle
    theta = atan2(q1, q0), lifting an exact Goldstone mode to a
    pseudo-Goldstone with spectral mass mu^2 = delta * n^2 / (M_eff * f^2)
    on a vacuum circle of radius f. ``delta`` and ``n`` are fixed probe
    parameters (static, not learned).

    Composable over any base potential (learned or hand-built), so the same
    trained checkpoint can be probed at several breaking strengths without
    retraining.

    Caveat: the gradient of atan2 is singular at the channel origin
    (|grad| ~ delta*n/r). The probe is intended for dynamics on/near a vacuum
    circle r = f > 0; do not train from states passing through the origin
    with delta != 0.
    """

    base: eqx.Module
    tilt_delta: float = eqx.field(static=True)
    tilt_n: int = eqx.field(static=True)

    def __call__(self, q: jnp.ndarray) -> float:
        theta = jnp.arctan2(q[1], q[0])
        return self.base(q) + self.tilt_delta * jnp.cos(self.tilt_n * theta)
