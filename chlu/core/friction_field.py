"""Position-gated friction field gamma_phi(q) — the learned trash region.

Implements F5 §7.3 (Thread-1 mechanism made real):

    Def-5 (friction field): replace the scalar per-step friction gamma by
    gamma_phi(q) in [0, gamma_max), applied as

        (q, p) -> (q, (1 - gamma_phi(q_{n+1})) * p)

    after the Verlet substeps (evaluated at the POST-step position q_{n+1}).

    Prop-11 (position-gated volume contraction, exact): the damping Jacobian
    is block-triangular, so regardless of grad gamma_phi,

        det D(step) = (1 - gamma_phi(q_{n+1}))^d.

    Phase-space volume is destroyed exactly and only where gamma_phi > 0 —
    the "event horizon" is the superlevel set of gamma_phi; outside it the
    dynamics are exactly conservative. (tests/test_friction_field.py checks
    the determinant numerically.)

The field is K sigmoidal "holes" (learnable centers c_k, radii r_k via
softplus, strengths gamma_k = gamma_max * sigmoid(s_k)), combined
additive-saturating (noisy-OR in units of gamma_max):

    g_k(q)       = sigmoid((r_k - ||q - c_k||) / w)          # horizon gate
    gamma_phi(q) = gamma_max * (1 - prod_k (1 - (gamma_k / gamma_max) g_k(q)))

For K = 1 this reduces exactly to the Thread-1 form
gamma_1 * sigmoid((r_1 - ||q - c_1||) / w). Strictly gamma_phi < gamma_max < 1,
so the damping factor never vanishes (the step map stays invertible).

Training (Thread-1 round-2, one contrastive-divergence signal, two fields):
V_theta learns what things look like; gamma_phi learns what deserves to die —
wake pushes gamma_phi(q_data) down (protection), sleep pushes
gamma_phi(q_hallucination) up. The terms live in ``chlu.training.train`` behind
``training.friction_field_*`` config weights.

C1 spec (mo-deep-read §5, derived from F5 §3.3): retention is minimized at
critical damping gamma* ~= 2 * eps * mu, so an optimally-forgetting hole
targets gamma_k ~= 2 * dt * mu(c_k) (local spectral mass at its center).
``c1_regularizer`` optionally nudges strengths toward that optimum (default
OFF — measure, don't force; the S1 experiment *reports* the comparison).

Nomenclature (F5 Def-2): inertial mass M (kinetic diagonal) vs spectral mass
mu = sqrt(eig(M_eff^{-1} Hess V)). Never "mass" unqualified.

Composability / scope notes:
    - This module is independent of CHLU and of the lattice: any owner holding
      a field and calling it on a position vector can use it (a CLU-lattice
      unit can carry its own field later).
    - S2 (Hawking re-emission) is NOT implemented here by design. The re-emission
      hook attaches at the integrator application point (see the marked line in
      ``chlu.core.integrators``): local re-emission = noise with scale a function
      of gamma_phi(q_{n+1}) injected right after the field damping; global
      re-emission = routing the captured momentum norm into the sleep-phase
      temperature budget. The field API already exposes everything a hook needs
      (``__call__`` and ``hole_params``).
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def _inv_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: y such that softplus(y) == x (x > 0)."""
    return jnp.log(jnp.expm1(x))


def _logit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of sigmoid, with clipping away from {0, 1} for finiteness."""
    x = jnp.clip(x, 1e-4, 1.0 - 1e-4)
    return jnp.log(x / (1.0 - x))


class FrictionField(eqx.Module):
    """K learnable friction holes ("trash regions"), gamma_phi: R^dim -> [0, gamma_max).

    Parameters (all shape-(K,...) arrays, learnable unless ``trainable=False``):
        centers:         hole centers c_k, (K, dim)
        log_radii:       r_k = softplus(log_radii), (K,)
        strength_logits: gamma_k = gamma_max * sigmoid(strength_logits), (K,)

    Static:
        gamma_max: strict upper bound on gamma_phi (must be in (0, 1))
        width:     sigmoid horizon width w (smaller = harder horizon)
        trainable: if False, parameters are stop_gradient-ed at use time
                   (the "fixed" / hand-placed control variant)
    """

    centers: jnp.ndarray
    log_radii: jnp.ndarray
    strength_logits: jnp.ndarray
    gamma_max: float = eqx.field(static=True)
    width: float = eqx.field(static=True)
    trainable: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        k: int = 1,
        gamma_max: float = 0.5,
        width: float = 0.25,
        centers: Optional[jnp.ndarray] = None,
        init_radius: float = 1.0,
        init_strength: float = 0.25,
        init_center_scale: float = 1.0,
        trainable: bool = True,
        key: jax.random.PRNGKey = None,
    ):
        """
        Args:
            dim: position-space dimension.
            k: number of holes K (ignored if ``centers`` is given explicitly).
            gamma_max: strict cap on gamma_phi, in (0, 1).
            width: sigmoid horizon width w.
            centers: optional explicit (K, dim) centers (hand placement — the
                "fixed"/oracle variant, or a designed init). If None, centers
                are drawn ~ N(0, init_center_scale^2) using ``key``.
            init_radius: initial hole radius (exact, via inverse softplus).
            init_strength: initial hole strength gamma_k (exact, via inverse
                sigmoid; clipped into (0, gamma_max)).
            trainable: False freezes all parameters (stop_gradient at use).
            key: PRNG key for random center init (required if centers is None).
        """
        if not 0.0 < gamma_max < 1.0:
            raise ValueError(f"gamma_max must be in (0, 1), got {gamma_max}")
        if width <= 0.0:
            raise ValueError(f"width must be positive, got {width}")

        if centers is not None:
            centers = jnp.asarray(centers, dtype=jnp.result_type(float))
            if centers.ndim != 2 or centers.shape[1] != dim:
                raise ValueError(
                    f"centers must have shape (K, dim={dim}), got {centers.shape}"
                )
            k = centers.shape[0]
            self.centers = centers
        else:
            if key is None:
                key = jax.random.PRNGKey(0)
            self.centers = init_center_scale * jax.random.normal(key, (k, dim))

        self.log_radii = jnp.full((k,), _inv_softplus(jnp.asarray(init_radius)))
        self.strength_logits = jnp.full(
            (k,), _logit(jnp.asarray(init_strength / gamma_max))
        )
        self.gamma_max = gamma_max
        self.width = width
        self.trainable = trainable

    @property
    def k(self) -> int:
        """Number of holes K."""
        return self.centers.shape[0]

    def hole_params(self) -> tuple:
        """(centers (K, dim), radii (K,), strengths (K,)) in physical units.

        Applies the positivity/bound transforms; stop_gradients everything
        when ``trainable=False`` (the fixed-variant freeze).
        """
        c, lr, sl = self.centers, self.log_radii, self.strength_logits
        if not self.trainable:
            c = jax.lax.stop_gradient(c)
            lr = jax.lax.stop_gradient(lr)
            sl = jax.lax.stop_gradient(sl)
        radii = jax.nn.softplus(lr)
        strengths = self.gamma_max * jax.nn.sigmoid(sl)
        return c, radii, strengths

    def __call__(self, q: jnp.ndarray) -> jnp.ndarray:
        """gamma_phi(q): scalar friction in [0, gamma_max) at position q (dim,)."""
        c, radii, strengths = self.hole_params()
        # Safe distance (grad of ||.|| at 0 is NaN; the epsilon keeps
        # evaluation AT a hole center differentiable).
        dist = jnp.sqrt(jnp.sum((q[None, :] - c) ** 2, axis=-1) + 1e-12)
        gate = jax.nn.sigmoid((radii - dist) / self.width)  # (K,)
        u = (strengths / self.gamma_max) * gate  # per-hole fraction of gamma_max
        return self.gamma_max * (1.0 - jnp.prod(1.0 - u))


def build_friction_field(
    training_config, dim: int, key: jax.random.PRNGKey = None
) -> Optional[FrictionField]:
    """Construct the field named by ``config.training.friction_field``.

    Modes:
        "none"    -> None (default; scalar-gamma behavior, bit-compatible)
        "fixed"   -> hand-placed frozen holes at
                     ``friction_field_fixed_centers`` (or the origin), with
                     ``friction_field_fixed_radius`` / ``_fixed_strength``
        "learned" -> K = ``friction_field_k`` trainable holes, centers
                     ~ N(0, friction_field_init_center_scale^2)
    """
    mode = training_config.friction_field
    if mode == "none":
        return None
    if mode == "fixed":
        fixed = training_config.friction_field_fixed_centers
        centers = (
            jnp.asarray(fixed, dtype=jnp.result_type(float))
            if fixed is not None
            else jnp.zeros((training_config.friction_field_k, dim))
        )
        return FrictionField(
            dim,
            gamma_max=training_config.friction_field_gamma_max,
            width=training_config.friction_field_width,
            centers=centers,
            init_radius=training_config.friction_field_fixed_radius,
            init_strength=training_config.friction_field_fixed_strength,
            trainable=False,
        )
    if mode == "learned":
        return FrictionField(
            dim,
            k=training_config.friction_field_k,
            gamma_max=training_config.friction_field_gamma_max,
            width=training_config.friction_field_width,
            init_radius=training_config.friction_field_init_radius,
            init_strength=training_config.friction_field_init_strength,
            init_center_scale=training_config.friction_field_init_center_scale,
            trainable=True,
            key=key,
        )
    raise ValueError(
        f"Unknown friction_field mode: {mode!r}. Must be 'none', 'fixed', or 'learned'."
    )


def spectral_masses_at(model, q: jnp.ndarray) -> jnp.ndarray:
    """Local spectral masses mu_k(q) = sqrt(clip(eig(W), 0)) of the canonical
    Hessian W = M_eff^{-1/2} Hess V(q) M_eff^{-1/2} (F5 §3.1, Def-2).

    Negative-curvature (unstable) directions are clipped to 0 — they have no
    oscillation frequency and hence no critical-damping optimum.

    Mirrors ``chlu.experiments.goldstone_harness.spectrum_probe`` (which core
    must not import — layering); the harness remains the reporting instrument.
    """
    K = jax.hessian(model.potential_net)(q)
    s = 1.0 / jnp.sqrt(model.effective_inertia())
    W = (K * s[:, None]) * s[None, :]
    W = 0.5 * (W + W.T)  # symmetrize fp noise
    mu_sq = jnp.linalg.eigvalsh(W)
    return jnp.sqrt(jnp.clip(mu_sq, 0.0, None))


def c1_regularizer(model, dt: float) -> jnp.ndarray:
    """Optional C1 nudge: mean_k (gamma_k - 2*dt*mu_bar(c_k))^2.

    mu_bar(c_k) is the mean local spectral mass at hole center c_k. The target
    is stop_gradient-ed so the term ONLY moves the hole strengths gamma_k
    toward the critical-damping optimum 2*eps*mu (mo-deep-read C1) — it must
    not warp V's curvature (or drag the centers) to meet gamma. Behind
    ``training.friction_field_c1_lambda`` (default 0.0 = off: measure, don't
    force).
    """
    field = model.friction_field
    centers, _, strengths = field.hole_params()

    def target(center):
        return 2.0 * dt * jnp.mean(spectral_masses_at(model, center))

    t = jax.lax.stop_gradient(jax.vmap(target)(centers))
    return jnp.mean((strengths - t) ** 2)
