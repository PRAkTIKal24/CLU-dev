"""Hand-designed potentials for the addressable-memory (write/address/read) demo.

**Everything in this module is DESIGNED, not learned.** It exists to answer the
stage-1 question of the Head's architectural vision (handover 2026-07-21): does
the *physics* support selective storage and retrieval at all, before any encoder,
address selector or read-out head is trained? Nothing here should ever be cited
as an emergent capability (N46 precedent: designed structure worked, emergent
structure never formed).

Two landscapes:

``RingRegisterPotential``
    A ring of ``K`` angular wells (item sites) carrying a **payload
    coordinate**. The payload is the anti-decoration guard: the retrieved value
    ``a_k`` is written into ``V`` only, and every query is launched with
    ``q2(0) = 0``, so a read-out that recovers ``a_k`` cannot have got it from
    the address.

``ThreeModePotential``
    The three write modes of the vision realized side by side in one landscape:
    a permanent latch (exactly flat SO(2) angle, ``mu^2 == 0``), a decaying item
    written at the *same locus* (the radial excursion, ``mu_rad^2 = 8*lam*f^2``),
    and an uncorrelated item at a fresh location (a broken-symmetry pair).
"""

import equinox as eqx
import jax.numpy as jnp


def _safe_theta(q0: jnp.ndarray, q1: jnp.ndarray, eps: float):
    """Channel angle plus a radial envelope that kills the r=0 singularity.

    ``arctan2`` has gradient ``(-q1, q0)/r^2``, which blows up at the origin.
    Every angular term below is multiplied by ``env = r^2/(r^2+eps)``, which
    vanishes quadratically at the origin and is ~1 on the vacuum ring, so the
    composed potential is smooth everywhere while the physics on the ring
    (where all trajectories live) is unchanged to O(eps/f^2).

    ⚠ The envelope alone is NOT enough. At exactly ``q = 0`` the ``arctan2``
    gradient is NaN, and ``NaN * 0 = NaN`` — the standard JAX
    multiply-the-singularity-away trap — so ``grad V`` came back NaN at the
    origin even though ``V`` was finite (caught by
    ``test_ring_potential_finite_and_differentiable_at_origin``). The argument
    must be masked with ``jnp.where`` BEFORE it reaches ``arctan2`` so the
    singular branch is never evaluated.
    """
    r2 = q0 * q0 + q1 * q1
    at_origin = r2 <= 0.0
    q0s = jnp.where(at_origin, 1.0, q0)
    q1s = jnp.where(at_origin, 0.0, q1)
    theta = jnp.arctan2(q1s, q0s)
    env = r2 / (r2 + eps)
    return theta, env, r2


class RingRegisterPotential(eqx.Module):
    """K-item addressable register on a ring, with a payload read-out channel.

    .. code-block:: text

        V(q) = lam * (r^2 - f^2)^2                     # ring vacuum, r^2 = q0^2 + q1^2
             + b * (1 - cos(K * theta)) * env          # K angular wells = K item sites
             + 0.5 * kappa * (q2 - s(theta))^2         # payload channel
        s(theta) = sum_k a_k * exp((cos(theta - theta_k) - 1) / w) * env

    - **Item k** = the well at ``theta_k = 2*pi*k/K`` together with its stored
      payload ``a_k``.
    - **Address** = ``(m, q0, p0)``; the payload coordinate is always launched
      at ``q2(0) = 0``.
    - **Read** = the tail of the rollout. A payload-only read (``q2`` alone) is
      blind to the address plane, which is what makes the demo non-decorative.

    The payload bump width ``w`` is held FIXED as ``K`` grows, so neighbouring
    bumps overlap more and more — that overlap is the interference mechanism the
    item-count sweep measures (it is a designed, stated property, not an
    accident).

    Set ``payloads`` to zeros for the **blank control**: identical dynamics and a
    live read-out channel, but nothing stored. A working loop must read out at
    chance there.
    """

    payloads: jnp.ndarray  # (K,) stored values a_k
    theta_k: jnp.ndarray  # (K,) item site angles
    lam: float = eqx.field(static=True)
    f: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    K: int = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    bump_width: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    n_spectator: int = eqx.field(static=True)
    spectator_k: float = eqx.field(static=True)

    def __init__(
        self,
        payloads,
        lam: float = 1.0,
        f: float = 1.0,
        b: float = 0.05,
        kappa: float = 1.0,
        bump_width: float = 0.05,
        eps: float = 1e-3,
        n_spectator: int = 0,
        spectator_k: float = 1.0,
    ):
        """
        Args:
            payloads: (K,) stored payload values a_k. Designed, typically a
                NON-monotone permutation of a grid so the payload is not a
                smooth function of the address angle.
            lam: quartic ring stiffness; radial spectral mass mu_rad^2 = 8*lam*f^2.
            f: vacuum radius.
            b: angular barrier height between item sites.
            kappa: payload spring constant.
            bump_width: von-Mises-like payload bump width w (angular sigma ~ sqrt(w)).
            eps: radial-envelope regularizer (see ``_safe_theta``).
            n_spectator: number of extra harmonic spectator coordinates (>= 3).
            spectator_k: spring constant for those spectators.
        """
        self.payloads = jnp.asarray(payloads, dtype=jnp.float32)
        self.K = int(self.payloads.shape[0])
        self.theta_k = jnp.arange(self.K, dtype=jnp.float32) * (2.0 * jnp.pi / self.K)
        self.lam = lam
        self.f = f
        self.b = b
        self.kappa = kappa
        self.bump_width = bump_width
        self.eps = eps
        self.n_spectator = n_spectator
        self.spectator_k = spectator_k

    def payload_profile(self, theta):
        """s(theta) — the designed payload written around the ring (no envelope)."""
        bumps = jnp.exp(
            (jnp.cos(theta[..., None] - self.theta_k) - 1.0) / self.bump_width
        )
        return jnp.sum(self.payloads * bumps, axis=-1)

    def __call__(self, q: jnp.ndarray) -> float:
        theta, env, r2 = _safe_theta(q[0], q[1], self.eps)

        v = self.lam * (r2 - self.f**2) ** 2
        v = v + self.b * (1.0 - jnp.cos(self.K * theta)) * env

        s = self.payload_profile(theta) * env
        v = v + 0.5 * self.kappa * (q[2] - s) ** 2

        if self.n_spectator > 0:
            v = v + 0.5 * self.spectator_k * jnp.sum(q[3:] ** 2)
        return v


class ThreeModePotential(eqx.Module):
    """The vision's three write modes side by side in one designed landscape.

    .. code-block:: text

        V(q) = lam * (r^2 - f^2)^2          # channel (q0,q1): PERMANENT angle (mu^2 == 0)
                                            #                  + DECAYING radius (mu_rad^2 = 8*lam*f^2)
             + beta * (q2^2 - d^2)^2        # fresh site: UNCORRELATED broken-symmetry pair +/- d
             + 0.5 * k_perp * q3^2          # transverse confinement at the fresh site

    - **(a) permanent** — the channel angle ``theta``. Exactly flat by designed
      SO(2) invariance: ``grad V`` is radial on the channel, so every Verlet kick
      is torque-free and ``theta`` cannot drift. ``mu^2 == 0`` identically.
    - **(b) decaying, written NEAR (a)** — a radial excursion ``delta_r`` at the
      *same* latent locus. Finite ``mu_rad^2 = 8*lam*f^2`` gives it a half-life
      under friction, while its neighbour (a) is untouched (a purely radial write
      carries zero angular momentum).
    - **(c) uncorrelated, fresh location** — the sign of ``q2`` at a
      symmetry-broken pair of vacua, additively separable from the channel.

    Write locality here is **designed** (additive separability + exact symmetry),
    not emergent. An MLP potential has no locality by default; this is the
    structured potential the task allows, and the report says so.
    """

    lam: float = eqx.field(static=True)
    f: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    d: float = eqx.field(static=True)
    k_perp: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(
        self,
        lam: float = 1.0,
        f: float = 1.0,
        beta: float = 1.0,
        d: float = 1.0,
        k_perp: float = 1.0,
        dim: int = 4,
    ):
        if dim < 4:
            raise ValueError(f"ThreeModePotential requires dim >= 4, got dim={dim}")
        self.lam = lam
        self.f = f
        self.beta = beta
        self.d = d
        self.k_perp = k_perp
        self.dim = dim

    def __call__(self, q: jnp.ndarray) -> float:
        r2 = q[0] ** 2 + q[1] ** 2
        v = self.lam * (r2 - self.f**2) ** 2
        v = v + self.beta * (q[2] ** 2 - self.d**2) ** 2
        v = v + 0.5 * self.k_perp * jnp.sum(q[3:] ** 2)
        return v


def designed_payloads(K: int, seed: int = 0, lo: float = -1.0, hi: float = 1.0):
    """A DESIGNED, deliberately non-monotone set of K payload values.

    A fixed permutation of an even grid on ``[lo, hi]``. Non-monotone in the
    site index so the payload is not a smooth (hence not a trivially
    linearly-decodable) function of the address angle.
    """
    import numpy as np

    grid = np.linspace(lo, hi, K)
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.permutation(grid), dtype=jnp.float32)
