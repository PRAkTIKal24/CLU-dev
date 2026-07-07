"""'CLU minus the physics' controls (critique G2 / P6).

Two identical-capacity, duck-typed CHLU stand-ins that let us decompose *what
symplecticity functionally buys*. Both expose the CHLU surface
(``H``, ``T``, ``potential_net``, ``effective_inertia``, ``effective_mass``,
``mass_vector``, ``step``, ``stochastic_step``, ``__call__``, ``dim``,
``kinetic_mode``, ``potential_type``), so the ``goldstone_harness`` instruments
and ``train.train_chlu`` run on them verbatim (``CLULattice`` precedent).

Three arms, decomposing the physics prior:
  - **CHLU**                 — symplectic Verlet + volume conservation (det J = 1).
  - **BrokenVolumeCHLU**     — the SAME leapfrog + potential + kinetic term, plus
                               a learned per-coordinate scaling that breaks
                               det J = 1. Isolates *volume conservation* (only).
  - **UnconstrainedTwin**    — a free residual recurrence z_{t+1} = z + dt*f_theta(z),
                               no Hamiltonian, no volume constraint. Removes the
                               *whole* physics (integrator structure + volume).

Attribution (F5-style ablation): CHLU vs BrokenVolume isolates the delta from
volume conservation; BrokenVolume vs UnconstrainedTwin isolates the delta from
the leapfrog integrator structure itself.

Nomenclature (F5 Def-2): inertial mass M vs spectral mass mu — never "mass"
unqualified. The twin has no Hamiltonian and hence no spectral masses.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.core.integrators import velocity_verlet_step
from chlu.utils.metrics import count_params


# ---------------------------------------------------------------------------
# Arm 2: broken-volume twin (isolates volume conservation)
# ---------------------------------------------------------------------------


class BrokenVolumeCHLU(eqx.Module):
    """The CHLU leapfrog with a learned volume-violating per-coordinate scale.

    One dissipative-Verlet step (identical H, potential, kinetic term, dt,
    gamma to the wrapped CHLU) followed by a learned diagonal scaling of the
    full state:

        (q', p') = verlet(q, p);  z_next = exp(log_scale) * concat(q', p').

    ``det J = det(verlet) * prod_i exp(log_scale_i) = exp(sum log_scale)``, so
    phase-space volume is conserved iff ``sum(log_scale) == 0``. ``log_scale``
    is initialized to zeros, so at init the arm is **bit-identical** to the
    wrapped CHLU (the isolation is exact: only the learned scaling can break
    symplecticity). Everything else — architecture, potential, kinetic mode,
    integrator — is the CHLU's, so this arm isolates *volume conservation*
    specifically, as opposed to the twin which removes the whole physics.

    The scaling is a genuine non-symplectic deformation of the same map: it
    is applied *outside* the Hamiltonian flow, so no shadow Hamiltonian /
    modified-equation guarantee survives it.
    """

    base: CHLU
    log_scale: jnp.ndarray  # (2*dim,) log per-coordinate scaling of concat[q, p]

    def __init__(self, base: CHLU):
        self.base = base
        self.log_scale = jnp.zeros(2 * base.dim)

    # ----- duck-typed static-ish surface (delegates to the wrapped CHLU) -----
    @property
    def dim(self) -> int:
        return self.base.dim

    @property
    def kinetic_mode(self) -> str:
        return self.base.kinetic_mode

    @property
    def potential_type(self) -> str:
        return self.base.potential_type

    @property
    def potential_net(self):
        return self.base.potential_net

    def H(self, q, p):
        return self.base.H(q, p)

    def T(self, p):
        return self.base.T(p)

    def mass_vector(self):
        return self.base.mass_vector()

    def effective_inertia(self):
        return self.base.effective_inertia()

    def effective_mass(self):
        return self.base.effective_mass()

    # ------------------------------- dynamics --------------------------------
    def _apply_scale(self, q, p):
        s = jnp.exp(self.log_scale)
        d = q.shape[0]
        return q * s[:d], p * s[d:]

    def volume_log_jac(self) -> float:
        """log det J contributed by the scaling (0 => volume conserved)."""
        return float(jnp.sum(self.log_scale))

    def step(self, state: tuple, dt: float, gamma: float = 0.0) -> tuple:
        q, p = state
        q1, p1 = velocity_verlet_step(self.base.H, q, p, dt, gamma)
        return self._apply_scale(q1, p1)

    def stochastic_step(
        self, state, dt, gamma, temperature, key, noise_mode="legacy"
    ):
        q, p, new_key = self.base.stochastic_step(
            state, dt, gamma, temperature, key, noise_mode=noise_mode
        )
        q, p = self._apply_scale(q, p)
        return q, p, new_key

    def __call__(self, q0, p0, steps, dt, gamma=0.0):
        def scan_fn(state, _):
            q, p = self.step(state, dt, gamma)
            return (q, p), jnp.concatenate([q, p])

        _, traj = jax.lax.scan(scan_fn, (q0, p0), None, length=steps)
        return traj


# ---------------------------------------------------------------------------
# Arm 3: unconstrained twin (removes the whole physics)
# ---------------------------------------------------------------------------


class UnconstrainedTwin(eqx.Module):
    """Free residual recurrence z_{t+1} = z_t + dt * f_theta(z_t).

    A generic learned map on the concatenated state z = concat(q, p) with the
    SAME (q, p) dimensions and (approximately) matched parameter count as a
    CHLU — but no Hamiltonian, no symplectic structure, no volume constraint.
    This is the "minus the physics" control.

    ``gamma`` retains a role as a momentum-damping knob (p_next *= (1-gamma))
    so the retention / erosion measurement protocol applies uniformly across
    arms; it is the ONLY piece of shared physics vocabulary, documented as the
    minimal common friction handle. ``H``/``T`` are identically zero (no energy
    function), which makes the wake-sleep *sleep* phase inert by construction —
    the twin is trained by trajectory MSE (wake), the natural minus-the-physics
    objective.
    """

    layers: list
    dim: int = eqx.field(static=True)
    kinetic_mode: str = eqx.field(static=True)
    potential_type: str = eqx.field(static=True)

    def __init__(self, dim: int, hidden: int = 59, key: Optional[jax.Array] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        state_dim = 2 * dim
        self.layers = [
            eqx.nn.Linear(state_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, state_dim, key=k3),
        ]
        self.dim = dim
        self.kinetic_mode = "none"
        self.potential_type = "twin"

    def f(self, z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(self.layers[0](z))
        x = jnp.tanh(self.layers[1](x))
        return self.layers[2](x)

    # ----- duck-typed surface (no physics: energies are identically zero) ----
    @property
    def potential_net(self):
        return lambda q: 0.0 * jnp.sum(q)

    def H(self, q, p):
        # No energy function; kept so train_chlu's sleep phase is a no-op.
        return 0.0 * jnp.sum(q) + 0.0 * jnp.sum(p)

    def T(self, p):
        return 0.0 * jnp.sum(p)

    def mass_vector(self):
        return jnp.ones(self.dim)

    def effective_inertia(self):
        return jnp.ones(self.dim)

    def effective_mass(self):
        return jnp.ones(self.dim)

    # ------------------------------- dynamics --------------------------------
    def step(self, state: tuple, dt: float, gamma: float = 0.0) -> tuple:
        q, p = state
        z = jnp.concatenate([q, p])
        z = z + dt * self.f(z)
        d = self.dim
        q_next, p_next = z[:d], z[d:]
        p_next = (1.0 - gamma) * p_next  # minimal shared friction knob
        return q_next, p_next

    def stochastic_step(
        self, state, dt, gamma, temperature, key, noise_mode="legacy"
    ):
        q, p = self.step(state, dt, gamma)
        key, subkey = jax.random.split(key)
        scale = jnp.sqrt(jnp.maximum(0.0, 2.0 * gamma * temperature * dt))
        noise = jax.random.normal(subkey, p.shape) * scale
        p = jnp.where(temperature > 0.0, p + noise, p)
        return q, p, key

    def __call__(self, q0, p0, steps, dt, gamma=0.0):
        def scan_fn(state, _):
            q, p = self.step(state, dt, gamma)
            return (q, p), jnp.concatenate([q, p])

        _, traj = jax.lax.scan(scan_fn, (q0, p0), None, length=steps)
        return traj


# ---------------------------------------------------------------------------
# Parameter-count matching
# ---------------------------------------------------------------------------


def _twin_param_count(dim: int, hidden: int) -> int:
    """Analytic param count of an UnconstrainedTwin(dim, hidden) MLP."""
    sd = 2 * dim
    return (sd * hidden + hidden) + (hidden * hidden + hidden) + (hidden * sd + sd)


def matched_twin_hidden(target_params: int, dim: int) -> tuple[int, int]:
    """Hidden width whose twin param count is closest to ``target_params``.

    Solves h^2 + h(4d + 2) + 2d = target for h and rounds to the nearer of
    floor/ceil by exact count. Returns (hidden, param_count).
    """
    sd = 2 * dim
    # h^2 + (2*sd + 2) h + (sd - target) = 0
    b = 2 * sd + 2
    c = sd - target_params
    disc = b * b - 4 * c
    h0 = (-b + disc**0.5) / 2.0
    cands = {max(1, int(h0)), max(1, int(h0) + 1), max(1, int(round(h0)))}
    best = min(cands, key=lambda h: abs(_twin_param_count(dim, h) - target_params))
    return best, _twin_param_count(dim, best)


def build_arms(key, dim, hidden, kinetic_mode, potential_type,
               rest_mass=1.0, c=1.0, tie_channel_mass=False):
    """Build the three matched-capacity arms sharing an init key stream.

    Returns:
        dict with keys 'chlu', 'broken_volume', 'twin' and a 'params' dict of
        their parameter counts (the count-match report).
    """
    k_clu, k_twin = jax.random.split(key)
    chlu = CHLU(
        dim=dim, hidden=hidden, rest_mass=rest_mass, c=c,
        kinetic_mode=kinetic_mode, potential_type=potential_type,
        tie_channel_mass=tie_channel_mass, key=k_clu,
    )
    broken = BrokenVolumeCHLU(chlu)
    n_clu = count_params(chlu)
    h_twin, _ = matched_twin_hidden(n_clu, dim)
    twin = UnconstrainedTwin(dim, hidden=h_twin, key=k_twin)
    return {
        "chlu": chlu,
        "broken_volume": broken,
        "twin": twin,
        "twin_hidden": h_twin,
        "params": {
            "chlu": count_params(chlu),
            "broken_volume": count_params(broken),
            "twin": count_params(twin),
        },
    }
