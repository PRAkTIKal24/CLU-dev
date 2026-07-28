"""Hand-designed potentials for the addressable-memory (write/address/read) demo.

**Everything in this module is DESIGNED, not learned.** It exists to answer the
stage-1 question of the Head's architectural vision (handover 2026-07-21): does
the *physics* support selective storage and retrieval at all, before any encoder,
address selector or read-out head is trained? Nothing here should ever be cited
as an emergent capability (N46 precedent: designed structure worked, emergent
structure never formed).

Three landscapes:

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

``BallRegisterPotential``
    The **d-dimensional** generalization of the ring: K Gaussian item wells packed
    into a flat-bottomed ball of radius ``R`` in ``d`` address dimensions, plus the
    same payload channel. The ring's 8-item ceiling is an *angular*-resolution
    limit of a 1-D address manifold; this class exists to measure whether capacity
    is exponential in ``d`` (the packing bound ``(1 + 2R/w)^d``) or whether the
    handful-of-items ceiling is a property of the primitive.
``DesignFreedomPotential`` (w20 — the one LEARNED family in this module)
    The same memory landscape rebuilt as ``designed part + LEARNED part``, with a
    switch (``rung``) controlling how much of the structure is designed in. This
    is the vehicle for the w20 question "does the w19 loop survive a learned
    landscape, and how much design does it need?". Its learned part is trained by
    ``chlu.training.train_memory``. It is the ONLY thing here that is not purely
    hand-built, and its docstring says exactly which terms are designed at each
    rung so no result from it can be mistaken for emergence.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.core.potentials import PotentialMLP


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


class BallRegisterPotential(eqx.Module):
    """K-item addressable register in a **d-dimensional** address ball.

    The `d`-dimensional generalization of ``RingRegisterPotential``. The ring is a
    1-D address manifold embedded in 2-D, so its capacity is set by an *angular*
    resolution limit (w19 measured ``K_max ~ 0.2 * 2*pi / sigma_theta`` = 8 items).
    That number is a property of the ring, NOT of CLU — this class exists to find
    out which.

    .. code-block:: text

        V(q) = c_conf * relu(||x||^2 - R^2)^2               # flat inside the ball, walls outside
             - b * sum_k exp(-||x - c_k||^2 / (2 w^2))      # K item wells
             + 0.5 * kappa * (y - s(x))^2                   # payload channel
        s(x) = sum_k a_k * exp(-||x - c_k||^2 / (2 w^2))

    with ``x = q[:d]`` the **address plane** and ``y = q[d]`` the **payload
    channel**. Any spectator coordinates ``q[d+1:]`` get a harmonic well.

    - **Item k** = the well at ``c_k`` together with its stored payload ``a_k``.
    - **Address** = ``(m, q0, p0)``; the payload coordinate is always launched at
      ``y(0) = 0``, so a payload-only read is structurally blind to the address.
      This is the w19 anti-decoration guard, carried over verbatim.
    - **Blank control** = the same object with ``payloads`` all zero: identical
      dynamics, live read-out channel, nothing stored. A working loop must read
      out at chance there. **A cell without a passing blank control is not a
      measurement.**

    Geometry notes (these are the whole point of the class):

    - The confinement is **flat inside the ball** (``relu`` of ``||x||^2 - R^2``),
      so the only structure inside the address region is the wells themselves.
      A quartic/harmonic confinement would add a `d`-dependent restoring force
      and confound the dimension scaling with a geometry change.
    - The well width ``w`` is held FIXED as ``K`` grows (as ``bump_width`` is on
      the ring), so neighbouring wells overlap more and more — that overlap is
      the interference mechanism, and ``w`` vs the site separation is exactly the
      packing-bound question ``(1 + 2R/w)^d``.
    """

    payloads: jnp.ndarray  # (K, m) stored values a_k (m payload channels)
    centers: jnp.ndarray  # (K, d) item site locations
    d: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    R: float = eqx.field(static=True)
    w: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    c_conf: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    spectator_k: float = eqx.field(static=True)

    def __init__(
        self,
        payloads,
        centers,
        R: float = 1.0,
        w: float = 0.15,
        b: float = 1.0,
        kappa: float = 1.0,
        c_conf: float = 10.0,
        dim: Optional[int] = None,
        spectator_k: float = 1.0,
    ):
        """
        Args:
            payloads: (K,) stored payload values a_k (designed, non-monotone), or
                **(K, m)** for the w26 multi-channel read-out code — the payload then
                occupies channels ``q[d:d+m]`` and the default latent dim is ``d+m``.
                A 1-D ``payloads`` is the shipped single-channel geometry,
                bit-identical.
            centers: (K, d) item site locations inside the ball of radius R.
            R: address-ball radius (the "region" of the packing bound).
            w: Gaussian well width (the "w" of the packing bound ``(1+2R/w)^d``).
            b: well depth.
            kappa: payload spring constant.
            c_conf: stiffness of the outside-the-ball confining wall.
            dim: total latent dim; defaults to ``d + 1`` (address + payload).
            spectator_k: spring constant for any coordinates beyond ``d + 1``.
        """
        pay = jnp.asarray(payloads, dtype=jnp.float32)
        self.payloads = pay[:, None] if pay.ndim == 1 else pay
        self.centers = jnp.asarray(centers, dtype=jnp.float32)
        self.K = int(self.centers.shape[0])
        self.d = int(self.centers.shape[1])
        self.m = int(self.payloads.shape[1])
        self.dim = int(self.d + self.m) if dim is None else int(dim)
        if self.dim < self.d + self.m:
            raise ValueError(
                f"dim={self.dim} too small for a {self.d}-D address plane plus "
                f"{self.m} payload channel(s) (need dim >= d + m)"
            )
        if self.payloads.shape[0] != self.K:
            raise ValueError(
                f"payloads has {self.payloads.shape[0]} entries but there are "
                f"{self.K} centers"
            )
        self.R = R
        self.w = w
        self.b = b
        self.kappa = kappa
        self.c_conf = c_conf
        self.spectator_k = spectator_k

    def bumps(self, x: jnp.ndarray) -> jnp.ndarray:
        """(K,) Gaussian activations of each item site at address ``x``."""
        d2 = jnp.sum((x[None, :] - self.centers) ** 2, axis=-1)
        return jnp.exp(-d2 / (2.0 * self.w**2))

    def payload_profile(self, x: jnp.ndarray):
        """s(x) — the designed payload written across the address ball.

        Scalar for the shipped single-channel register (bit-identical to w19-w25),
        ``(m,)`` for the w26 multi-channel code.
        """
        prof = jnp.sum(self.payloads * self.bumps(x)[:, None], axis=0)
        return prof[0] if self.m == 1 else prof

    def __call__(self, q: jnp.ndarray) -> float:
        x = q[: self.d]
        y = q[self.d : self.d + self.m]

        # Confining wall: identically zero inside the ball (relu), so the only
        # structure the particle sees in the address region is the item wells.
        excess = jnp.maximum(jnp.sum(x * x) - self.R**2, 0.0)
        v = self.c_conf * excess**2

        bumps = self.bumps(x)
        v = v - self.b * jnp.sum(bumps)
        prof = jnp.sum(self.payloads * bumps[:, None], axis=0)  # (m,)
        v = v + 0.5 * self.kappa * jnp.sum((y - prof) ** 2)

        if self.dim > self.d + self.m:
            v = v + 0.5 * self.spectator_k * jnp.sum(q[self.d + self.m :] ** 2)
        return v


def designed_sites(d: int, K: int, R: float = 1.0, seed: int = 0, pool: int = 20000):
    """K item-site locations in the `d`-ball of radius R, by FARTHEST-POINT sampling.

    Farthest-point (maximin) sampling from a uniform candidate pool maximizes the
    achieved minimum pairwise separation, so this is the **best packing this design
    can do** at a given K — the capacity it yields is an upper envelope, not a
    typical draw. Deterministic given ``seed``.

    The volume argument predicts the achieved separation
    ``Delta(d, K) ~ 2 * R * K**(-1/d)``; ``site_separation`` measures what was
    actually achieved so the prediction can be checked rather than assumed.

    Returns:
        (K, d) float32 array of site locations.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_pool = max(pool, 40 * K)
    # Uniform in the d-ball: normalized Gaussian direction * U^(1/d) radius.
    g = rng.normal(size=(n_pool, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    rad = R * rng.random(n_pool) ** (1.0 / d)
    cand = g * rad[:, None]

    idx = [0]
    d2 = np.sum((cand - cand[0]) ** 2, axis=1)
    for _ in range(1, K):
        nxt = int(np.argmax(d2))
        idx.append(nxt)
        d2 = np.minimum(d2, np.sum((cand - cand[nxt]) ** 2, axis=1))
    return jnp.asarray(cand[idx], dtype=jnp.float32)


def site_separation(centers) -> float:
    """Achieved minimum pairwise separation of a site set (the packing measurement)."""
    import numpy as np

    c = np.asarray(centers)
    if c.shape[0] < 2:
        return float("inf")
    d2 = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def _atom_group_owner(n_atoms: int, n_groups: int):
    """Owner group index per atom row, matching :meth:`AtomDictionaryPotential.group_rows`.

    Contiguous blocks with boundaries ``round(g*n/G)`` — the vectorized inverse of
    that partition (numpy, so it is a trace-time constant). Mirrors
    ``chlu.training.train_memory._atom_owner``; kept local to avoid a core->training
    import.
    """
    import numpy as np

    bounds = np.array(
        [round(g * n_atoms / n_groups) for g in range(n_groups + 1)], dtype=int
    )
    rows = np.arange(n_atoms)
    return np.clip(np.searchsorted(bounds, rows, side="right") - 1, 0, n_groups - 1)


def _uniform_ball(key: jax.random.PRNGKey, n: int, dim: int) -> jnp.ndarray:
    """``(n, dim)`` points drawn uniformly in the unit ``dim``-ball.

    Normalized Gaussian direction times ``U^(1/dim)`` radius — the same construction
    :func:`designed_sites` uses for its candidate pool, so a localized atom init and
    a designed site set are drawn from the same geometry.
    """
    k_g, k_r = jax.random.split(key, 2)
    g = jax.random.normal(k_g, (n, dim))
    g = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    rad = jax.random.uniform(k_r, (n,)) ** (1.0 / max(dim, 1))
    return g * rad[:, None]


class RBFAtoms(eqx.Module):
    """A learned dictionary of LOCALIZED wells: ``V = -sum_j d_j exp(-|q-c_j|^2/2s_j^2)``.

    The point of this family is that locality is **imposed by the basis** (each
    atom has compact-ish support) while *where* the atoms sit, how deep and how
    wide they are, is learned. It is the rung that isolates "designed locality,
    learned placement" from "designed placement".

    Depths are ``softplus``-positive so an atom can only ever dig a well, never
    build a hill — a designed sign constraint, stated because it is structure.
    """

    centers: jnp.ndarray  # (n_atoms, dim)
    log_width: jnp.ndarray  # (n_atoms,)
    depth_raw: jnp.ndarray  # (n_atoms,) -> softplus -> depth
    confine: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        n_atoms: int,
        key: jax.random.PRNGKey,
        init_scale: float = 1.0,
        init_width: float = 0.3,
        confine: float = 0.05,
    ):
        k_c, k_d = jax.random.split(key, 2)
        self.centers = jax.random.normal(k_c, (n_atoms, dim)) * init_scale
        self.log_width = jnp.full((n_atoms,), jnp.log(init_width))
        self.depth_raw = jax.random.normal(k_d, (n_atoms,)) * 0.1
        self.confine = confine

    def __call__(self, q: jnp.ndarray) -> float:
        s = jnp.exp(self.log_width)
        d2 = jnp.sum((q[None, :] - self.centers) ** 2, axis=-1)
        depth = jax.nn.softplus(self.depth_raw)
        v = -jnp.sum(depth * jnp.exp(-d2 / (2.0 * s**2 + 1e-9)))
        return v + self.confine * jnp.sum(q**2)


class AtomDictionaryPotential(eqx.Module):
    """The theorist's MVC-0 substrate (clu-controller-spec §4): a LOCAL-support
    dictionary ``V = alpha*|q|^2 + sum_j (-A_j exp(-|q-c_j|^2 / 2 s_j^2))`` whose
    amplitudes, centers and widths are all learned.

    Two things distinguish it from :class:`RBFAtoms`, and both are load-bearing
    for w21 — neither is cosmetic:

    1. **It starts FLAT, without a vanishing amplitude gradient.** The amplitude
       is ``A_j = amp_j**2`` (non-negative and smooth), initialised at
       ``sqrt(depth_init)`` with ``depth_init = 1e-4``: at init ``V ~
       alpha*|q|^2`` and the *writer digs the wells*.
       ⚠ Both halves of that sentence are load-bearing and were **measured**, not
       assumed. ``RBFAtoms`` initialises ``depth_raw ~ N(0, 0.1)`` => every atom
       already has depth ``softplus(0) = 0.69``, so a large dictionary starts
       rugged on exactly the length scale the relaxation must traverse. But
       simply pushing a ``softplus`` parameterisation down to a flat start
       (``depth_raw = -8``) is *worse*: ``d softplus/dx = sigmoid(-8) = 3.4e-4``,
       so with Adam the raw amplitude can move at most ``lr * steps = 1.8`` and
       the deepest well reachable in a 600-step write is ``softplus(-6.2) =
       2e-3`` — the write silently no-ops (measured: strict 0.062 at K=4, write
       loss stuck at 0.12 vs 0.003 for the MLP). The squared parameterisation has
       an O(1) gradient at small amplitude and reaches depth ``~1.8**2`` in the
       same budget.
    2. **It carries a GROUP STRUCTURE** (``n_groups``): atom rows are partitioned
       into ``n_groups`` contiguous blocks, one per item slot. Together with
       :func:`atom_write_mask_fn` this makes a write *local in parameter space* —
       writing item ``i`` leaves every other item's atoms **bit-identical**. That
       is the concrete form of the theorist's C3 claim ("dictionary/atom writes
       are C3-local by construction") and the H-SUPP arm of w21.

    ``A_j = amp_j**2 >= 0``: an atom can only ever dig a well, never build a
    hill. That is a designed sign constraint and is stated as structure.
    """

    centers: jnp.ndarray  # (n_atoms, dim)
    log_width: jnp.ndarray  # (n_atoms,)
    amp: jnp.ndarray  # (n_atoms,) -> amp**2 -> depth
    confine: float = eqx.field(static=True)
    n_groups: int = eqx.field(static=True)
    # ⭐ w26: optional PER-AXIS width multiplier, a read-time knob. Axis ``i``'s
    # effective width is ``axis_width_scale[i] * s_j``, i.e. each atom is convolved
    # with an anisotropic Gaussian. ``None`` (default) is the isotropic shipped
    # potential, bit-identical. Held as a tuple of PYTHON floats, so it is inert
    # under ``eqx.is_inexact_array`` and can never be picked up by the write's
    # ``trainable_filter``; it is a read-time knob only.
    axis_width_scale: Optional[tuple] = None

    def __init__(
        self,
        dim: int,
        n_atoms: int,
        key: jax.random.PRNGKey,
        init_scale: float = 1.0,
        init_width: float = 0.3,
        confine: float = 0.05,
        depth_init: float = 1e-4,
        n_groups: int = 1,
        group_centers=None,
        local_radius: float = 0.0,
        axis_width_scale=None,
    ):
        """
        Args:
            group_centers: optional ``(n_groups, L)`` localization targets, ``L <=
                dim`` — group ``j``'s atoms are initialised in a ball of radius
                ``local_radius`` around ``group_centers[j]`` **in the leading ``L``
                coordinates only**; coordinates ``L:`` keep the scattered
                ``N(0, init_scale)`` init. Together with ``local_radius > 0`` this
                is the **N98 fix** (see below). ``None`` (default) => the historical
                scatter, bit-identical.
            local_radius: radius of that ball; ``0.0`` (default) disables the
                localization entirely, so the default construction is unchanged
                bit-for-bit (the localized draw uses a *folded* key, so even the
                RNG stream of the default path is untouched).

        ⚠ **N98 (w24 ``lattice-capacity-theory`` §1.4, measured).** With the shipped
        ``init_scale=1.0`` EVERY group's atoms are scattered over the whole ball, so
        group ``j`` owns atoms sitting inside item ``i``'s well from step 0. A masked
        write for item ``i`` must then compensate for foreign atoms it is not allowed
        to move, and masked-write != N-independent-optimizers by **1.4% in amplitude
        — three orders above the 1.5e-5 site tail** (the theorist's own prediction P3
        failed here). Localizing group ``j``'s atoms near site ``j`` restores W3 to
        the site tail and makes Prop L2 exact in practice.

        ⚠ **Why only the LEADING ``L`` coordinates (N46 fairness).** In the d-ball
        harness the payload channel is coordinate ``d`` and its value is what the
        write must *learn*; the address sites are already supplied to the write
        objective. Localizing the ADDRESS axes therefore hands the writer nothing it
        did not already have, while localizing the payload axis would hand it the
        answer — and would also destroy the measured basin-reach property that makes
        ``init_scale=1.0`` load-bearing (a well initialised at payload 0 cannot reach
        ``|a_i| = 1``). Callers pass address-only ``group_centers`` for this reason.
        """
        centers = jax.random.normal(key, (n_atoms, dim)) * init_scale
        if group_centers is not None and float(local_radius) > 0.0:
            gc = jnp.asarray(group_centers, dtype=centers.dtype)
            if gc.ndim != 2:
                raise ValueError(f"group_centers must be 2-D, got shape {gc.shape}")
            n_g = max(1, int(n_groups))
            if int(gc.shape[0]) != n_g:
                raise ValueError(
                    f"group_centers has {gc.shape[0]} rows for {n_g} groups"
                )
            local_dims = int(gc.shape[1])
            if local_dims > dim:
                raise ValueError(
                    f"group_centers width {local_dims} exceeds dim {dim}"
                )
            owner = jnp.asarray(_atom_group_owner(n_atoms, n_g))
            k_ball = jax.random.fold_in(key, 1)  # default path's stream untouched
            offs = _uniform_ball(k_ball, n_atoms, local_dims) * float(local_radius)
            centers = centers.at[:, :local_dims].set(gc[owner] + offs)
        self.centers = centers
        self.log_width = jnp.full((n_atoms,), jnp.log(init_width))
        self.amp = jnp.full((n_atoms,), float(depth_init) ** 0.5)
        self.confine = confine
        self.n_groups = max(1, int(n_groups))
        self.axis_width_scale = (
            None if axis_width_scale is None
            else tuple(float(x) for x in axis_width_scale)
        )

    @property
    def n_atoms(self) -> int:
        return int(self.centers.shape[0])

    def group_rows(self, group: int) -> jnp.ndarray:
        """Boolean row mask (n_atoms,) selecting the atoms owned by ``group``.

        Blocks are contiguous and as equal as possible; a group index outside
        ``[0, n_groups)`` selects nothing (used by the blank/degenerate cases).
        """
        import numpy as np

        n = self.n_atoms
        mask = np.zeros((n,), dtype=bool)
        if 0 <= group < self.n_groups:
            lo = round(group * n / self.n_groups)
            hi = round((group + 1) * n / self.n_groups)
            mask[lo:hi] = True
        return jnp.asarray(mask)

    def __call__(self, q: jnp.ndarray) -> float:
        s = jnp.exp(self.log_width)
        diff = q[None, :] - self.centers
        if self.axis_width_scale is not None:
            diff = diff / jnp.asarray(self.axis_width_scale, dtype=diff.dtype)[None, :]
        d2 = jnp.sum(diff**2, axis=-1)
        depth = self.amp**2
        v = -jnp.sum(depth * jnp.exp(-d2 / (2.0 * s**2 + 1e-9)))
        return v + self.confine * jnp.sum(q**2)


class HopfieldPotential(eqx.Module):
    """⭐ The **modern-Hopfield / attention** potential (Ramsauer et al. 2020):

    ``V(q) = -(1/beta) * logsumexp_i(beta * <W q, k_i> + b_i) + alpha * |q|^2``.

    This is the w21 discriminator arm. Three facts make it the right object:

    * With ``alpha = 1/2`` and ``W = I`` this is **exactly** the modern-Hopfield
      energy, and ``grad V = 0`` reads ``q = sum_i softmax(beta <q,k_i>)_i k_i``
      — i.e. the stationarity condition *is* one step of attention over the
      memory codebook. The arm is therefore a genuine in-framework test of
      "attention-as-memory vs atoms-as-memory", not an MLP wearing a hat.
    * Its **parameter support is exponentially local in the inner-product
      metric**: ``dV/dk_i = -softmax_i * q``, so a memory at inner-product gap
      ``d`` from ``q`` influences ``V(q)`` by ``~exp(-beta d)``. Attention is
      therefore NOT automatically "more global than an MLP" — its support is a
      *tunable* function of ``beta``, which an MLP has no analogue of. w21
      measures this decay rather than assuming it.
    * Capacity is held fixed while ``beta`` varies, which separates capacity from
      support with a single knob.

    ``beta`` is static (a hyperparameter of the class, as in Ramsauer), so the
    learned parameter count is exactly ``n_mem * (d_head + 1)``.
    """

    keys_: jnp.ndarray  # (n_mem, d_head)
    bias: jnp.ndarray  # (n_mem,)
    proj: Optional[jnp.ndarray]  # (d_head, dim) or None => identity (d_head=dim)
    beta: float = eqx.field(static=True)
    confine: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        n_mem: int,
        key: jax.random.PRNGKey,
        beta: float = 8.0,
        confine: float = 0.5,
        d_head: int = 0,
        key_init: float = 0.1,
    ):
        d_head = dim if d_head in (0, None) else int(d_head)
        k_k, k_p = jax.random.split(key, 2)
        self.keys_ = jax.random.normal(k_k, (n_mem, d_head)) * key_init
        self.bias = jnp.zeros((n_mem,))
        # d_head == dim => the pure Ramsauer form (no projection, W = I). A
        # projection is only built when a different head width is asked for,
        # because it would otherwise break the exact modern-Hopfield equivalence
        # (and add d_head*dim parameters to the matched count).
        if d_head == dim:
            self.proj = None
        else:
            self.proj = jax.random.normal(k_p, (d_head, dim)) / jnp.sqrt(dim)
        self.beta = float(beta)
        self.confine = float(confine)

    def __call__(self, q: jnp.ndarray) -> float:
        u = q if self.proj is None else self.proj @ q
        s = self.beta * (self.keys_ @ u) + self.bias
        return -jax.nn.logsumexp(s) / self.beta + self.confine * jnp.sum(q**2)


class AttentionPotential(eqx.Module):
    """Single-head **cross-attention** from the state to a learned memory
    codebook, read out as a scalar energy:

    ``V(q) = sum_i softmax(beta * <W q, k_i>)_i * v_i + alpha * |q|^2``.

    Complements :class:`HopfieldPotential`. The Hopfield form is the *energy*
    whose stationarity is the attention update; this one is the *architecture*
    (learned query projection ``W``, keys ``k``, values ``v``, softmax mixing)
    with a scalar value head. Both are run so that a negative transformer result
    cannot be dismissed as "you did not build a real attention layer": the two
    differ in whether the value head is tied to the key geometry (Hopfield) or
    free (here).
    """

    proj: jnp.ndarray  # (d_head, dim)
    keys_: jnp.ndarray  # (n_mem, d_head)
    values: jnp.ndarray  # (n_mem,)
    beta: float = eqx.field(static=True)
    confine: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        n_mem: int,
        key: jax.random.PRNGKey,
        beta: float = 1.0,
        confine: float = 0.05,
        d_head: int = 8,
        key_init: float = 0.5,
        value_init: float = 0.1,
    ):
        k_p, k_k, k_v = jax.random.split(key, 3)
        self.proj = jax.random.normal(k_p, (d_head, dim)) / jnp.sqrt(dim)
        self.keys_ = jax.random.normal(k_k, (n_mem, d_head)) * key_init
        self.values = jax.random.normal(k_v, (n_mem,)) * value_init
        self.beta = float(beta)
        self.confine = float(confine)

    def __call__(self, q: jnp.ndarray) -> float:
        u = self.proj @ q
        a = jax.nn.softmax(self.beta * (self.keys_ @ u))
        return jnp.sum(a * self.values) + self.confine * jnp.sum(q**2)


def atom_write_mask_fn(row_mask: jnp.ndarray):
    """Build an optax-update mask that freezes every atom row outside ``row_mask``.

    This is what makes an atom write **local in parameter space**: the returned
    callable multiplies the optimizer's updates for ``centers``/``log_width``/
    ``amp`` by the row mask, so atoms belonging to other items come out of
    the write **bit-identical** (masking the *updates*, not the gradients, is
    required — ``optax.adamw``'s decoupled weight decay would otherwise still
    shrink the frozen rows).

    Intended use: ``train_memory_landscape(..., update_mask_fn=atom_write_mask_fn(m))``
    where the potential is a :class:`DesignFreedomPotential` whose ``.learned`` is
    an :class:`AtomDictionaryPotential`.
    """
    m = jnp.asarray(row_mask, dtype=jnp.float32)

    def apply(updates):
        atoms = updates.learned
        return eqx.tree_at(
            lambda u: [u.learned.centers, u.learned.log_width, u.learned.amp],
            updates,
            replace=[
                atoms.centers * m[:, None],
                atoms.log_width * m,
                atoms.amp * m,
            ],
        )

    return apply


#: Learned potential families available to :class:`DesignFreedomPotential`. The
#: RUNG says how much structure is designed in; the FAMILY says what function
#: class the learned remainder is drawn from. w20 swept the rung at family
#: ``mlp``; w21 sweeps the family at fixed rung.
LEARNED_FAMILIES = ("mlp", "atoms", "hopfield", "attn")


#: Design-freedom ladder, least free -> most free. The integer is the "freedom"
#: coordinate of the fidelity-vs-design-freedom curve.
DESIGN_RUNGS = (
    "designed",  # 0: w19 hand-built landscape, zero learned parameters
    "skeleton_residual",  # 1: w19 landscape + a small learned residual
    "sites_learned_payload",  # 2: designed ring + K angular wells; payload learned
    "local_rbf",  # 3: learned localized atoms (locality designed, placement learned)
    "free_mlp",  # 4: free MLP + coercivity only
)


class DesignFreedomPotential(eqx.Module):
    """``V = V_designed(q) + scale * V_learned(q)`` with a design-freedom switch.

    The five rungs (``DESIGN_RUNGS``), and **exactly** what is designed in each:

    ==========================  ========================================  ===================
    rung                        designed terms                            learned terms
    ==========================  ========================================  ===================
    ``designed``                ring vacuum + K angular wells +           (none)
                                payload spring carrying the true a_k
    ``skeleton_residual``       same as ``designed``                      MLP x residual_scale
    ``sites_learned_payload``   ring vacuum + K angular wells             MLP
                                (address geometry only, NO payload)
    ``local_rbf``               coercivity only                           RBF atoms
    ``free_mlp``                coercivity only                           ``learned_family``
    ==========================  ========================================  ===================

    **Two orthogonal axes (w21).** The *rung* says how much structure is designed
    in; ``learned_family`` (one of :data:`LEARNED_FAMILIES`) says which function
    class the learned remainder is drawn from. w20 swept the rung at family
    ``mlp`` and concluded "the loop needs essentially all of the designed
    structure"; w21 asks whether that conclusion is a property of the *function
    class* by sweeping the family. ``learned_family`` defaults to ``"mlp"``, so
    every w20 result is reproduced bit-for-bit by the default path.
    ``local_rbf`` pins the family to ``RBFAtoms`` (it IS the family statement of
    that rung) and ignores ``learned_family``.

    ⚠ **Two honesty notes that must travel with every number this produces.**

    1. Even ``free_mlp`` is not structure-free: ``PotentialMLP`` carries a
       ``0.05*|q|^2`` confinement term, which is required for the unit to be
       coercive at all (F5 Prop-10 — Deep/Conv omit it and are architecturally
       non-coercive). "Free" means free *potential family*, not free structure.
    2. In **every** rung the WRITER supplies the target sites ``c_i`` to the
       training objective. The landscape is learned; the *placement of the items*
       is chosen. Nothing here tests whether item sites emerge on their own, and
       no result from this module may be quoted as if it did (N46/D3).

    The designed part is held FIXED during training; only ``learned`` is
    optimized (see ``chlu.training.train_memory.trainable_filter``).
    """

    designed: Optional[RingRegisterPotential]
    learned: Optional[eqx.Module]
    rung: str = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)
    confine: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    learned_family: str = eqx.field(static=True, default="mlp")

    def __init__(
        self,
        rung: str,
        dim: int,
        payloads,
        key: jax.random.PRNGKey,
        lam: float = 1.0,
        f: float = 1.0,
        barrier: float = 0.2,
        payload_kappa: float = 1.0,
        bump_width: float = 0.05,
        hidden: int = 64,
        n_atoms: int = 24,
        residual_scale: float = 0.1,
        confine: float = 0.05,
        rbf_init_width: float = 0.3,
        learned_family: str = "mlp",
        n_mem: int = 1120,
        beta: float = 8.0,
        d_head: int = 8,
        hopfield_confine: float = 0.5,
        key_init: float = 0.1,
        atom_depth_init: float = -8.0,
        atom_groups: int = 1,
        atom_init_scale: float = 1.0,
        atom_group_centers=None,
        atom_local_radius: float = 0.0,
    ):
        """
        Args:
            rung: one of ``DESIGN_RUNGS``.
            dim: latent dim (>= 3: address plane q0,q1 + payload channel q2).
            payloads: (K,) stored values. Used by the DESIGNED payload spring
                (``designed``/``skeleton_residual``) and, for every rung, by the
                training objective as the write target.
            key: PRNG key for the learned part.
            residual_scale: multiplier on the learned residual at rung 1. Small
                by design: the rung asks "does adding learned slop break a
                working designed landscape?", not "can learning replace it?".
            learned_family: which function class the LEARNED part is drawn from
                (:data:`LEARNED_FAMILIES`). ``"mlp"`` reproduces w20 exactly.
                ``n_mem``/``beta``/``d_head``/``hopfield_confine``/``key_init``
                size the attention families; ``n_atoms``/``atom_depth_init``/
                ``atom_groups``/``atom_init_scale`` size the atom dictionary.
            atom_group_centers / atom_local_radius: the **N98 localized atom init**
                (w25), forwarded verbatim to :class:`AtomDictionaryPotential`.
                ``atom_local_radius = 0.0`` (default) is the historical scatter,
                bit-identical.
        """
        if rung not in DESIGN_RUNGS:
            raise ValueError(f"rung must be one of {DESIGN_RUNGS}, got {rung!r}")
        if learned_family not in LEARNED_FAMILIES:
            raise ValueError(
                f"learned_family must be one of {LEARNED_FAMILIES}, "
                f"got {learned_family!r}"
            )
        if dim < 3:
            raise ValueError(f"DesignFreedomPotential requires dim >= 3, got {dim}")
        self.rung = rung
        self.dim = dim
        self.residual_scale = residual_scale
        self.confine = confine
        # `local_rbf` IS a family statement, so it records its own family name
        # rather than silently accepting a contradictory `learned_family`.
        if rung == "designed":
            self.learned_family = "none"
        elif rung == "local_rbf":
            self.learned_family = "rbf_atoms"
        else:
            self.learned_family = learned_family

        pay = jnp.asarray(payloads, dtype=jnp.float32)
        if rung in ("designed", "skeleton_residual"):
            designed_pay = pay
            kappa = payload_kappa
        elif rung == "sites_learned_payload":
            # Address geometry designed, payload channel NOT: kappa=0 removes the
            # payload spring entirely, so the payload well must be learned.
            designed_pay = jnp.zeros_like(pay)
            kappa = 0.0
        else:
            designed_pay = None
            kappa = 0.0

        if designed_pay is not None:
            self.designed = RingRegisterPotential(
                designed_pay,
                lam=lam,
                f=f,
                b=barrier,
                kappa=kappa,
                bump_width=bump_width,
                n_spectator=dim - 3,
            )
        else:
            self.designed = None

        if rung == "designed":
            self.learned = None
        elif rung == "local_rbf":
            self.learned = RBFAtoms(
                dim, n_atoms, key, init_width=rbf_init_width, confine=confine
            )
        elif learned_family == "mlp":
            self.learned = PotentialMLP(dim, hidden=hidden, key=key)
        elif learned_family == "atoms":
            self.learned = AtomDictionaryPotential(
                dim,
                n_atoms,
                key,
                init_scale=atom_init_scale,
                init_width=rbf_init_width,
                confine=confine,
                depth_init=atom_depth_init,
                n_groups=atom_groups,
                group_centers=atom_group_centers,
                local_radius=atom_local_radius,
            )
        elif learned_family == "hopfield":
            self.learned = HopfieldPotential(
                dim,
                n_mem,
                key,
                beta=beta,
                confine=hopfield_confine,
                d_head=0,  # pure Ramsauer form: W = I
                key_init=key_init,
            )
        else:  # "attn"
            self.learned = AttentionPotential(
                dim,
                n_mem,
                key,
                beta=beta,
                confine=confine,
                d_head=d_head,
                key_init=key_init,
            )

    @property
    def design_freedom(self) -> int:
        """Position on the design-freedom ladder (0 = fully designed)."""
        return DESIGN_RUNGS.index(self.rung)

    def __call__(self, q: jnp.ndarray) -> float:
        v = 0.0
        if self.designed is not None:
            v = v + self.designed(q)
        if self.learned is not None:
            scale = self.residual_scale if self.rung == "skeleton_residual" else 1.0
            v = v + scale * self.learned(q)
        return v


def ring_sites(K: int, f: float = 1.0, dim: int = 3, payloads=None) -> jnp.ndarray:
    """The K write targets ``z_i = (f*cos th_i, f*sin th_i, a_i, 0...)``.

    Identical geometry to the w19 designed landscape, so the design-freedom sweep
    varies ONLY the potential family, not where the items live.
    """
    th = jnp.arange(K, dtype=jnp.float32) * (2.0 * jnp.pi / K)
    z = jnp.zeros((K, dim))
    z = z.at[:, 0].set(f * jnp.cos(th)).at[:, 1].set(f * jnp.sin(th))
    if payloads is not None:
        z = z.at[:, 2].set(jnp.asarray(payloads, dtype=jnp.float32))
    return z


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


class AtomStorePotential(eqx.Module):
    """MVC-0's designed store: a **dictionary of localized atoms** with payloads.

    .. code-block:: text

        V(q) = alpha * |q_addr|^2                                # coercivity
             - sum_i m_i * A_i * exp(-|q_addr - c_i|^2 / 2 s^2)  # K item wells
             + 0.5 * kappa * (q_pay - S(q_addr))^2               # payload channel
             + 0.5 * k_spec * |q_spec|^2                         # spectators
        S(q_addr) = sum_i m_i * a_i * exp(-|q_addr - c_i|^2 / 2 s_pay^2)

    ``q_addr = q[:2]``, ``q_pay = q[2]``, ``q_spec = q[3:]`` -- identical index
    convention to :class:`RingRegisterPotential`, so the same queries, the same
    anti-decoration guard (``q2(0) = 0``) and the same reads apply.

    **Why this class exists.** The w19/w20 ring pins K items at K *evenly spaced*
    angles determined by K itself, so "write one more item" is not an operation
    it supports: the whole landscape changes. Sequential writing needs a
    landscape whose sites are free, and MVC-0 §4 specifies exactly this object
    (``AtomStorePotential``) as the designed store, because an atom write is
    **C3-local by construction**: the gradient it contributes at a stored minimum
    at distance ``d`` carries ``exp(-d^2 / 2 s^2)``, which is ``6.3e-5`` at the
    admission radius ``d_safe = 4.4 s``.

    Writes are functional and mask-based: a fixed ``capacity`` of slots, an
    ``active`` 0/1 mask, and :meth:`with_item` returns a NEW module with the next
    slot filled. Nothing is trained here -- this is the designed arm.
    """

    centers: jnp.ndarray  # (capacity, addr_dim) address-space sites
    payloads: jnp.ndarray  # (capacity,) stored values a_i
    amps: jnp.ndarray  # (capacity,) well depths A_i
    active: jnp.ndarray  # (capacity,) 0/1 slot mask m_i
    alpha: float = eqx.field(static=True)
    s: float = eqx.field(static=True)
    s_pay: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    spectator_k: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int = 3,
        capacity: int = 16,
        alpha: float = 0.02,
        s: float = 0.35,
        s_pay: Optional[float] = None,
        kappa: float = 1.0,
        spectator_k: float = 1.0,
        addr_dim: int = 2,
    ):
        """An EMPTY store. Items are added with :meth:`with_item`.

        Args:
            dim: latent dim (>= ``addr_dim + 1``: address block q[:addr_dim] +
                payload channel q[addr_dim]).
            capacity: number of slots (fixed, so the object is a static PyTree).
            alpha: address-space confinement (coercivity; F5 Prop-10).
            s: atom width -- the ONLY length scale the admission radius
                ``d_safe = 4.4 s`` is expressed in.
            s_pay: payload-bump width (defaults to ``s``).
            kappa: payload spring constant.
            addr_dim: dimension of the address block (default **2**, the w23
                address plane -- unchanged behaviour). The w25 continual-learning
                entry addresses the store by ``φ(x) ∈ R^{phi_dim}``, so the store
                must support an address block of arbitrary dimension; nothing else
                about the object changes.
        """
        if addr_dim < 1:
            raise ValueError(f"addr_dim must be >= 1, got {addr_dim}")
        if dim < addr_dim + 1:
            raise ValueError(
                f"AtomStorePotential requires dim >= addr_dim + 1 "
                f"({addr_dim + 1}), got {dim}"
            )
        self.dim = dim
        self.addr_dim = int(addr_dim)
        self.centers = jnp.zeros((capacity, int(addr_dim)))
        self.payloads = jnp.zeros((capacity,))
        self.amps = jnp.zeros((capacity,))
        self.active = jnp.zeros((capacity,))
        self.alpha = alpha
        self.s = s
        self.s_pay = s if s_pay is None else s_pay
        self.kappa = kappa
        self.spectator_k = spectator_k

    @property
    def capacity(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_stored(self) -> int:
        return int(jnp.sum(self.active))

    def sites(self, dim: Optional[int] = None) -> jnp.ndarray:
        """(n_stored, dim) full stored states ``z_i = (c_i, a_i, 0...)``."""
        d = self.dim if dim is None else dim
        a = self.addr_dim
        idx = jnp.nonzero(self.active, size=self.capacity, fill_value=-1)[0]
        idx = idx[idx >= 0]
        z = jnp.zeros((idx.shape[0], d))
        z = z.at[:, :a].set(self.centers[idx])
        z = z.at[:, a].set(self.payloads[idx])
        return z

    def with_item(self, center, payload: float, amp: float = 1.0):
        """Append one item. Returns a NEW potential (writes are functional).

        Raises ``RuntimeError`` when the dictionary is full -- an overflow is a
        capacity alarm the controller must see, never a silent overwrite.
        """
        slot = int(jnp.argmin(self.active))
        if float(self.active[slot]) != 0.0:
            raise RuntimeError(f"AtomStorePotential is full ({self.capacity})")
        c = jnp.asarray(center, dtype=jnp.float32).reshape(-1)[: self.addr_dim]
        new = eqx.tree_at(lambda t: t.centers, self, self.centers.at[slot].set(c))
        new = eqx.tree_at(
            lambda t: t.payloads, new, new.payloads.at[slot].set(float(payload))
        )
        new = eqx.tree_at(lambda t: t.amps, new, new.amps.at[slot].set(float(amp)))
        return eqx.tree_at(lambda t: t.active, new, new.active.at[slot].set(1.0))

    def evict(self, slot: int):
        """Clear one slot (``active -> 0``, ``amp -> 0``). Returns a NEW potential.

        Eviction is the controller's budget verb (MVC-0 §3.C / C5): a full store
        makes room by *removing* an item, never by silently overwriting (which
        :meth:`with_item` forbids). A zero-amplitude, inactive slot contributes
        nothing to ``V`` and its address is free to be re-used by a later write.
        """
        new = eqx.tree_at(lambda t: t.active, self, self.active.at[slot].set(0.0))
        new = eqx.tree_at(lambda t: t.amps, new, new.amps.at[slot].set(0.0))
        return new

    def with_amps(self, amps):
        """Return a NEW potential with per-slot amplitudes replaced.

        This is the physical form of **scheduled decay** (leaky wells, w22): a
        controller tick multiplies each non-permanent well's amplitude by
        ``exp(-leak)``, so an item's basin shallows over time and, once its depth
        falls below the confinement, stops holding a query — per-item forgetting
        by construction, with permanent (``leak == 0``) wells left untouched.
        """
        a = jnp.asarray(amps, dtype=self.amps.dtype).reshape(-1)
        if a.shape[0] != self.amps.shape[0]:
            raise ValueError(f"amps must have length {self.amps.shape[0]}, got {a.shape[0]}")
        return eqx.tree_at(lambda t: t.amps, self, a)

    def __call__(self, q: jnp.ndarray) -> float:
        a = self.addr_dim
        addr = q[:a]
        d2 = jnp.sum((addr[None, :] - self.centers) ** 2, axis=-1)
        w_addr = self.active * jnp.exp(-d2 / (2.0 * self.s**2))
        w_pay = self.active * jnp.exp(-d2 / (2.0 * self.s_pay**2))

        v = self.alpha * jnp.sum(addr**2)
        v = v - jnp.sum(self.amps * w_addr)
        s_of_q = jnp.sum(self.payloads * w_pay)
        v = v + 0.5 * self.kappa * (q[a] - s_of_q) ** 2
        if self.dim > a + 1:
            v = v + 0.5 * self.spectator_k * jnp.sum(q[a + 1 :] ** 2)
        return v


class GaussianMemoryPotential(eqx.Module):
    """A **D-dimensional dense associative memory** as a designed CLU landscape.

    .. code-block:: text

        V(q) = 0.5 * alpha * |q|^2  -  b * sum_i exp(-|q - xi_i|^2 / (2 s^2))

    Each stored pattern ``xi_i`` (e.g. an image flattened to ``D`` pixels) is a
    Gaussian well of depth ``b`` and width ``s``; the tiny ``alpha`` term is the
    coercivity floor (F5 Prop-10). Retrieval is the **damped velocity-Verlet
    rollout of a CLU wired to this potential**: launch from the (masked/noisy)
    query, let the particle settle into the nearest well, read the settled state.

    This is the direct-pattern analogue of :class:`BallRegisterPotential`/
    :class:`AtomStorePotential` (the task-named designed registers): there the
    address plane is low-D and the pattern is a separate payload; here the pattern
    IS the address, so no payload channel is needed and the settled position is
    the retrieved memory. It is a *localized* dense associative memory (Gaussian
    interaction, cf. Krotov-Hopfield 2016) — distinct from the modern-Hopfield
    ``logsumexp`` inner-product energy (:class:`HopfieldPotential`), whose
    stationarity is one softmax-attention step. Nothing here is learned: the
    centers ARE the stored patterns, ``s`` is set by a fixed data-driven rule.

    ``s`` is intentionally the ONLY resolution knob (like ``beta`` in a
    modern-Hopfield net); it is not tuned per load, so the capacity curve is not
    flattered. The well curvature at a center is ``b/s^2 * I`` (a stiff quadratic
    basin), so a settled particle sits at the argmin well to leading order — which
    is why a Gaussian CLU register is expected to track the nearest-neighbour
    floor and the sparse-Hopfield line, not the near-uniform dense-softmax line.
    """

    centers: jnp.ndarray  # (M, D) stored patterns
    amps: jnp.ndarray  # (M,) per-well depth multipliers (all-ones by default)
    s: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, centers, s: float, b: float = 1.0, alpha: float = 1e-3,
                 amps=None):
        """Args:
            amps: optional ``(M,)`` per-well depth multipliers. ``None`` (the
                default) means all-ones — **exactly** the uniform-depth store of
                w22–w24. Per-well amplitudes exist so a store under **scheduled
                per-item retention** (w25) can be read: a well whose amplitude has
                decayed no longer holds a settling particle, which is what makes
                the retention schedule physical rather than bookkeeping.
        """
        self.centers = jnp.asarray(centers, dtype=jnp.float32)
        m = int(self.centers.shape[0])
        self.amps = (
            jnp.ones((m,), dtype=jnp.float32)
            if amps is None
            else jnp.asarray(amps, dtype=jnp.float32).reshape(-1)
        )
        if self.amps.shape[0] != m:
            raise ValueError(f"amps must have length {m}, got {self.amps.shape[0]}")
        self.s = float(s)
        self.b = float(b)
        self.alpha = float(alpha)

    @property
    def n_stored(self) -> int:
        return int(self.centers.shape[0])

    def __call__(self, q: jnp.ndarray) -> float:
        d2 = jnp.sum((q[None, :] - self.centers) ** 2, axis=-1)
        wells = self.amps * jnp.exp(-d2 / (2.0 * self.s**2))
        return 0.5 * self.alpha * jnp.sum(q * q) - self.b * jnp.sum(wells)
