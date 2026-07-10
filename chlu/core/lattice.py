"""CLU lattice: N CHLU units under ONE joint Hamiltonian (CLU-Net, F5 §7).

F5 Def-4 (CLU lattice): units i = 1..N with states (q_i, p_i) in R^{2 d_i} on
a coupling graph E, evolved under the single joint Hamiltonian

    H_net(q, p) = sum_i [ T_i(p_i) + V_i(q_i) ] + sum_{(i,j) in E} V_c(q_i, q_j)

by the SAME dissipative-Verlet map as a single unit, applied to the
concatenated state ("depth" = rollout time, "width" = N, "architecture" = E).

Composition conditions (F5 §7.2) and how this module enforces them:
  1. **Kinetic separability** — T_net = sum_i T_i(p_i). Couplings here are
     POSITION-ONLY *by construction*: coupling modules are called with
     (q_i, q_j) and never see momenta. Momentum/velocity coupling (magnetic,
     boost-conditioned, ...) would make H non-separable and cost the explicit
     integrator its exact symplecticity — do NOT add it here; flag the
     proposal instead (task scope guard).
  2. **One global step** — a single ``velocity_verlet_step`` on the
     concatenated state; kicks use the full joint force
     grad_{q_i}(V_i + sum_j V_c). No multirate/per-unit stepping.
  3. **Damping** — uniform scalar gamma preserves global conformal
     symplecticity with factor (1 - gamma) (F5 Prop-3/4). Per-unit gamma_i
     (via ``gamma_vector``, the flagged heterogeneous path) keeps the exact
     volume law det J = prod_i (1 - gamma_i)^{d_i} but LOSES the global
     singular-value pairing of F5 Prop-4 — state depth-stability claims
     accordingly.

**Exact reduction:** with an empty edge list (or kappa_c = 0) the joint step
reduces bit-level to N independent CHLU steps — the correctness anchor
(tests/test_lattice.py::test_kappa_zero_reduction_bitlevel).

**Designed inertial-mass banding** (Thread-5 doctrine; wave-2 evidence says
the hierarchy must be designed-in, it does not emerge): per-unit
``inertial_mass_scale`` rescales each unit's inertial mass exactly
(softplus-space, see ``scale_inertial_mass``); heavy/slow backbone units vs
light/fast perception units. Per-band anisotropic causal caps
v_max,i = c / sqrt(M_i) follow (F5 Prop-1).

Harness compatibility: ``CLULattice`` exposes the same duck-typed surface as
``CHLU`` (``H``, ``T``, ``potential_net``, ``effective_inertia``,
``effective_mass``, ``mass_vector``, ``step``, ``stochastic_step``,
``__call__``), so every instrument in
``chlu.experiments.goldstone_harness`` (spectrum_probe, perturb_and_track,
step_jacobian, settle, ...) and the wake–sleep trainer
(``chlu.training.train.train_chlu``) work on the joint state unchanged.
Rollouts are ``lax.scan``-based and pure (reversible-BPTT stays possible).

Nomenclature (F5 Def-2, binding): **inertial mass M** = kinetic-term diagonal
(larger M = slower, lower causal cap c/sqrt(M_i)); **spectral mass mu** =
sqrt(eigenvalue of M_eff^{-1} Hess V) (larger mu = faster, shorter memory).
Never "mass" unqualified.
"""

import warnings
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.core.chlu_unit import CHLU
from chlu.core.integrators import langevin_step, velocity_verlet_step


# ---------------------------------------------------------------------------
# Position-only couplings V_c(q_i, q_j)  (F5 §7.2 condition 1 by construction)
# ---------------------------------------------------------------------------


class SpringCoupling(eqx.Module):
    """
    Quadratic (spring) coupling  V_c(q_i, q_j) = kappa * ||W_i q_i - W_j q_j||^2.

    W_i (k, d_i) and W_j (k, d_j) are learnable projections into a shared
    k-dimensional coupling space; ``kappa`` is the static coupling-strength
    knob (config-declared, swept in experiments — not a trained parameter, so
    the optimizer cannot silently move the swept variable).

    The coupling curvature is what prices communication (F5 §7.2): at a
    synchronized point the relative mode acquires spectral mass
    mu_rel^2 ∝ kappa / M_eff — sync timescale ∝ kappa^{-1/2},
    relative-information retention ∝ 1/kappa (overdamped).

    With W_i = W_j = a common channel selector, V_c is invariant under
    *simultaneous* rotations of both channels but not under relative ones —
    the F5 "inter-unit communication has a mass" geometry
    (see ``channel_spring_coupling``).
    """

    W_i: jnp.ndarray  # (k, d_i)
    W_j: jnp.ndarray  # (k, d_j)
    kappa: float = eqx.field(static=True)

    def __call__(self, q_i: jnp.ndarray, q_j: jnp.ndarray) -> jnp.ndarray:
        u = self.W_i @ q_i - self.W_j @ q_j
        return self.kappa * jnp.sum(u * u)


class MLPCoupling(eqx.Module):
    """
    Learned nonlinear position-only coupling
    V_c(q_i, q_j) = kappa * f_theta(concat(q_i, q_j)), f_theta a small tanh MLP.

    Flag-selected alternative to ``SpringCoupling`` (coupling_type="mlp").
    Positions only — momenta never enter (F5 §7.2 condition 1). Note a
    generic f_theta has no symmetry; equivariant couplings are a designed
    choice, not an emergent one.
    """

    layers: list
    kappa: float = eqx.field(static=True)

    def __init__(
        self,
        d_i: int,
        d_j: int,
        kappa: float,
        hidden: int = 16,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        self.kappa = float(kappa)
        self.layers = [
            eqx.nn.Linear(d_i + d_j, hidden, key=k1),
            eqx.nn.Linear(hidden, 1, key=k2),
        ]

    def __call__(self, q_i: jnp.ndarray, q_j: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([q_i, q_j])
        x = jnp.tanh(self.layers[0](x))
        return self.kappa * jnp.squeeze(self.layers[1](x))


class GatedCoupling(eqx.Module):
    """
    Wormhole slot (skeleton): a base coupling wrapped in a SMOOTH energy gate
    (F5 §7.4, smooth-gate variant — everything in §7.2 keeps holding).

    **The gate is the annealed average of a discrete Ising bond variable**
    (``xy-lattice-theory`` §6.1). Put sigma in {0, 1} on the edge with edge
    Hamiltonian H_e = sigma * (v - t) at gate-temperature w, v = base(q_i, q_j).
    Summing sigma out exactly gives

        <sigma> = sigmoid((t - v) / w)                 <- the gate
        F(v)    = -w * softplus((t - v) / w)           <- its FREE ENERGY
        dF/dv   = <sigma>                              <- the annealed mean force

    ``energy_mode`` selects what ``__call__`` contributes to the joint V:

    - ``"free_energy"`` (**default**) returns ``F(v)``. Its gradient is exactly
      the annealed mean force ``<sigma> * grad v``: always attractive, force
      fraction bounded in [0, 1], ``F`` monotone increasing in ``v`` and
      bounded in ``[-w*ln(1 + e^{t/w}), 0)``. This is the physically-correct
      annealed potential of the Ising gate.
    - ``"mean_energy"`` (**legacy**, the pre-2026-07-10 shipped behavior)
      returns the mean ENERGY ``<sigma> * v``. ⚠ That is NOT a free energy: its
      force ``<sigma> - (v/w) <sigma> (1 - <sigma>)`` **changes sign** at finite
      ``v`` *inside* the nominally-open region (v = 0.8020 for the defaults
      t=1.0, w=0.25, reaching -0.718), so ``V_wh`` is non-monotone and the
      wormhole **repels its own endpoints** over a finite range of separations;
      on an XY bond it can flip the effective exchange antiferromagnetic.
      Retained for backwards-compatibility only (nothing is silently deleted) —
      do not use it for new physics.

    The gate value is a smooth function of state *through the potential*, so
    H stays C^1 and exactly (conformally) symplectic — no piecewise-defined
    Hamiltonian, no energy jumps to ledger (those appear only with hard top-k
    selection, which is deliberately NOT implemented here).

    Behavior (both modes): when the endpoint states are close in the base's
    coupling space (v << threshold) the gate is open (≈1) and the wormhole
    transmits force; at large separation (v >> threshold) the gate closes
    exponentially and the distant pair decouples. The wormhole energy is
    bounded — the "energy cost" of the non-local edge is capped by
    construction.

    No top-k / selection logic here (task scope guard) — the edge list just
    accepts non-adjacent pairs.
    """

    base: eqx.Module
    threshold: float = eqx.field(static=True)
    width: float = eqx.field(static=True)
    energy_mode: str = eqx.field(static=True, default="free_energy")

    def __call__(self, q_i: jnp.ndarray, q_j: jnp.ndarray) -> jnp.ndarray:
        v = self.base(q_i, q_j)
        drive = (self.threshold - v) / self.width
        if self.energy_mode == "free_energy":
            return -self.width * jax.nn.softplus(drive)
        elif self.energy_mode == "mean_energy":
            return jax.nn.sigmoid(drive) * v
        raise ValueError(
            f"Unknown gate energy_mode: {self.energy_mode!r}. Must be "
            "'free_energy' (annealed free energy, default) or 'mean_energy' "
            "(legacy; non-monotone force)."
        )

    def occupancy(self, q_i: jnp.ndarray, q_j: jnp.ndarray) -> jnp.ndarray:
        """The annealed Ising occupancy <sigma> = sigmoid((t - v)/w) in (0, 1).

        In ``free_energy`` mode this is exactly ``dV_wh/dv`` — the fraction of
        the ungated force the wormhole transmits.
        """
        v = self.base(q_i, q_j)
        return jax.nn.sigmoid((self.threshold - v) / self.width)


def spring_coupling(
    d_i: int,
    d_j: int,
    kappa: float,
    coupling_dim: int = 2,
    init_scale: float = 0.1,
    key: Optional[jax.random.PRNGKey] = None,
    init_mode: str = "random",
) -> SpringCoupling:
    """
    Spring coupling with LEARNABLE projections W.

    ``init_mode``:

    - ``"random"`` (default, legacy): ``W ~ N(0, init_scale^2)``. ⚠ A generic
      random ``W`` **breaks the lattice's global U(1)** (``xy-lattice-theory``
      §2.5, Prop-2/Cor): writing ``A = W[:, :2]``, the reduced bond potential
      carries a ``p=2`` single-site anisotropy ``h_2``, a Dzyaloshinskii–Moriya
      phase, and a ``cos(theta_i + theta_j)`` term. Measured at init:
      ``P(J < 0) = 0.52`` (the exchange is antiferromagnetic half the time),
      median ``h_2/|J| = 1.00``, median U(1)-breaking/``|J| = 1.48``. Since a
      ``p=2`` anisotropy has scaling dimension ``x_2 = 1/2 < 2`` at the KT
      fixed point it is a **relevant** perturbation [José–Kadanoff–Kirkpatrick–
      Nelson 1977] and destroys any 2-D memory phase.
    - ``"conformal"``: ``W_i = W_j = 1_k`` (identity on the first
      ``coupling_dim`` coordinates) at init, **still fully trainable**. Exactly
      the ``channel_spring_coupling`` geometry at step 0, so the reduction is
      exactly XY there (``h_2 = U(1)-break = 0``). Training on U(1)-symmetric
      data keeps it there (``h_2/|J| <= 0.017`` after 400 epochs) and recovers
      ``J/J_true = 1.03–1.07`` at equal-or-better wake loss — **the symmetry is
      free**; the objective never wanted it broken.

    ``init_scale`` is ignored in ``"conformal"`` mode.
    """
    if init_mode not in ("random", "conformal"):
        raise ValueError(
            f"Unknown spring init_mode: {init_mode!r}. Must be 'random' or 'conformal'."
        )
    if init_mode == "conformal":
        if coupling_dim > min(d_i, d_j):
            raise ValueError(
                f"conformal init needs coupling_dim ({coupling_dim}) <= "
                f"min(d_i, d_j) = {min(d_i, d_j)}"
            )
        # eye(k, d) = identity embedding of the first k coords (the channel)
        return SpringCoupling(
            W_i=jnp.eye(coupling_dim, d_i),
            W_j=jnp.eye(coupling_dim, d_j),
            kappa=float(kappa),
        )
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    W_i = init_scale * jax.random.normal(k1, (coupling_dim, d_i))
    W_j = init_scale * jax.random.normal(k2, (coupling_dim, d_j))
    return SpringCoupling(W_i=W_i, W_j=W_j, kappa=float(kappa))


def channel_spring_coupling(
    d_i: int,
    d_j: int,
    kappa: float,
    channel: tuple = (0, 1),
) -> SpringCoupling:
    """
    Spring coupling with FIXED identity-on-channel projections:
    V_c = kappa * ||q_i[channel] - q_j[channel]||^2.

    Invariant under simultaneous SO(2) rotations of both channels, NOT under
    relative rotation — the designed instrument for the communication-pricing
    measurement (F5 §7.2). At a synchronized vacuum with channel inertial
    mass M per unit, the joint channel spectrum is exactly
    {0 (shared latch), 4*kappa/M (relative), radial, radial + 4*kappa/M}.

    **This is the U(1)-preserving coupling** (``xy-lattice-theory`` Prop-1,
    prerequisite P5). With ``so2_invariant`` units, restricting the joint V to
    the product of vacuum rings (radius r*) gives EXACTLY

        V = sum_<ij> 2 kappa r*^2 (1 - cos(theta_i - theta_j)),   J = 2 kappa r*^2

    a pure first harmonic — no ``cos 2 dtheta``, no anisotropy, no U(1)
    breaking (verified to 2.2e-16; all other harmonics <= 7.4e-18). Prefer this
    (or ``spring_coupling(..., init_mode="conformal")``) over the random-W
    ``spring_coupling`` for any SO(2) lattice.
    """
    rows = []
    for c in channel:
        if not (0 <= c < min(d_i, d_j)):
            raise ValueError(f"channel coord {c} out of range for dims ({d_i}, {d_j})")
        rows.append(c)
    W_i = jnp.zeros((len(rows), d_i))
    W_j = jnp.zeros((len(rows), d_j))
    for r, c in enumerate(rows):
        W_i = W_i.at[r, c].set(1.0)
        W_j = W_j.at[r, c].set(1.0)
    return SpringCoupling(W_i=W_i, W_j=W_j, kappa=float(kappa))


# ---------------------------------------------------------------------------
# Designed inertial-mass banding (Thread-5; F5 §5)
# ---------------------------------------------------------------------------


def scale_inertial_mass(unit: CHLU, scale: float) -> CHLU:
    """
    Rescale a unit's inertial mass EXACTLY in softplus space:
    softplus(new_log_mass) = scale * softplus(old_log_mass).

    (A naive log_mass shift would not scale M exactly — softplus is not an
    exponential.) Requires a kinetic mode that reads log_mass; banding a
    ``newtonian_identity`` unit would be silently ignored, so it raises.
    """
    if scale <= 0.0:
        raise ValueError(f"inertial_mass_scale must be > 0, got {scale}")
    if unit.kinetic_mode == "newtonian_identity":
        raise ValueError(
            "inertial-mass banding requires a kinetic mode that reads log_mass "
            "('newtonian_learned' or 'relativistic'); 'newtonian_identity' "
            "would silently ignore the designed band."
        )
    target = scale * jax.nn.softplus(unit.log_mass)  # softplus target
    new_log_mass = jnp.log(jnp.expm1(target))  # inverse softplus
    return eqx.tree_at(lambda u: u.log_mass, unit, replace=new_log_mass)


# ---------------------------------------------------------------------------
# The lattice
# ---------------------------------------------------------------------------


class CLULattice(eqx.Module):
    """
    Joint-Hamiltonian lattice of CHLU units (F5 Def-4). See module docstring
    for the composition conditions; see ``build_lattice`` for the
    config-friendly constructor.

    State layout: concatenated blocks — q = [q_1; ...; q_N] (total dim D =
    sum_i d_i), p likewise; trajectories are (steps, 2*D) rows [q, p], exactly
    like a single CHLU of dimension D.
    """

    units: tuple  # tuple[CHLU, ...]
    couplings: tuple  # tuple[coupling module, ...], aligned with ``edges``
    edges: tuple = eqx.field(static=True)  # tuple[(i, j), ...]
    unit_dims: tuple = eqx.field(static=True)
    offsets: tuple = eqx.field(static=True)  # q-block start offsets, len N+1
    dim: int = eqx.field(static=True)  # total D

    def __init__(
        self,
        units: Sequence[CHLU],
        edges: Sequence[tuple] = (),
        couplings: Sequence[eqx.Module] = (),
    ):
        """
        Args:
            units: the CHLU units (each owns its V_i, log_mass, kinetic mode).
            edges: (i, j) unit-index pairs; any pair is legal (non-adjacent
                pairs = wormhole slots), no self-loops.
            couplings: position-only coupling modules, one per edge, called as
                coupling(q_i, q_j).
        """
        units = tuple(units)
        edges = tuple((int(i), int(j)) for (i, j) in edges)
        couplings = tuple(couplings)
        if len(units) < 1:
            raise ValueError("CLULattice requires at least one unit")
        if len(edges) != len(couplings):
            raise ValueError(
                f"edges ({len(edges)}) and couplings ({len(couplings)}) must align"
            )
        n = len(units)
        for i, j in edges:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"edge ({i}, {j}) out of range for {n} units")
            if i == j:
                raise ValueError(f"self-coupling edge ({i}, {j}) is not allowed")

        self.units = units
        self.edges = edges
        self.couplings = couplings
        dims = tuple(int(u.dim) for u in units)
        self.unit_dims = dims
        offs = [0]
        for d in dims:
            offs.append(offs[-1] + d)
        self.offsets = tuple(offs)
        self.dim = offs[-1]

    # ------------------------------------------------------------- structure

    @property
    def n_units(self) -> int:
        return len(self.units)

    def unit_slice(self, i: int) -> slice:
        """Slice of unit i's block in a concatenated q (or p) vector."""
        return slice(self.offsets[i], self.offsets[i + 1])

    def split(self, x: jnp.ndarray) -> tuple:
        """Split a concatenated (D,) vector into per-unit blocks."""
        return tuple(
            x[self.offsets[i] : self.offsets[i + 1]] for i in range(len(self.units))
        )

    # --------------------------------------------------------------- physics

    def T(self, p: jnp.ndarray) -> jnp.ndarray:
        """Separable joint kinetic energy T_net(p) = sum_i T_i(p_i) (F5 §7.2 cond 1)."""
        blocks = self.split(p)
        total = self.units[0].T(blocks[0])
        for unit, p_i in zip(self.units[1:], blocks[1:], strict=True):
            total = total + unit.T(p_i)
        return total

    def V(self, q: jnp.ndarray) -> jnp.ndarray:
        """Joint potential sum_i V_i(q_i) + sum_(i,j) V_c(q_i, q_j) — positions only."""
        blocks = self.split(q)
        total = self.units[0].potential_net(blocks[0])
        for unit, q_i in zip(self.units[1:], blocks[1:], strict=True):
            total = total + unit.potential_net(q_i)
        for (i, j), coupling in zip(self.edges, self.couplings, strict=True):
            total = total + coupling(blocks[i], blocks[j])
        return total

    @property
    def potential_net(self):
        """Duck-type the CHLU surface: the joint potential as a callable
        q -> scalar, so goldstone_harness.spectrum_probe et al. work on the
        joint state unchanged."""
        return self.V

    def H(self, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """The ONE joint Hamiltonian H_net = T_net(p) + V_net(q) (F5 Def-4)."""
        return self.T(p) + self.V(q)

    # ------------------------------------------------------ masses & budgets

    def mass_vector(self) -> jnp.ndarray:
        """Concatenated per-coordinate inertial mass M (F5 Def-2)."""
        return jnp.concatenate([u.mass_vector() for u in self.units])

    def unit_mass_vectors(self) -> tuple:
        """Per-unit inertial-mass vectors (the designed banding, unit by unit)."""
        return tuple(u.mass_vector() for u in self.units)

    def effective_inertia(self) -> jnp.ndarray:
        """Concatenated exact rest inertia M_eff (dynamics-consistent; feeds
        goldstone_harness.spectrum_probe on the joint state)."""
        return jnp.concatenate([u.effective_inertia() for u in self.units])

    def effective_mass(self) -> jnp.ndarray:
        """Concatenated M_eff at p ≈ 0 (CHLU.effective_mass semantics; used by
        the discrete-FDT Langevin noise scale)."""
        return jnp.concatenate([u.effective_mass() for u in self.units])

    def causal_caps(self) -> jnp.ndarray:
        """
        Per-coordinate anisotropic causal velocity caps v_max,i = c / sqrt(M_i)
        (F5 Prop-1) on the concatenated state. Only binding for relativistic
        units; Newtonian units have no cap (returns +inf for those blocks).
        """
        caps = []
        for u in self.units:
            if u.kinetic_mode == "relativistic":
                caps.append(u.c / jnp.sqrt(u.mass_vector() + 1e-6))
            else:
                caps.append(jnp.full(u.dim, jnp.inf))
        return jnp.concatenate(caps)

    # ------------------------------------------------------------- gamma API

    def gamma_vector(self, per_unit_gammas: Sequence[float]) -> jnp.ndarray:
        """
        Expand per-unit friction constants gamma_i to a per-coordinate (D,)
        vector for ``step``/``__call__`` — the FLAGGED heterogeneous-damping
        path (F5 §7.2 condition 3): the exact volume law
        det J = prod_i (1 - gamma_i)^{d_i} survives, but global conformal
        symplecticity and the Prop-4 singular-value pairing are LOST unless
        all gamma_i are equal. Prefer a uniform scalar gamma.
        """
        gammas = list(per_unit_gammas)
        if len(gammas) != len(self.units):
            raise ValueError(
                f"got {len(gammas)} per-unit gammas for {len(self.units)} units"
            )
        return jnp.concatenate(
            [jnp.full(d, float(g)) for g, d in zip(gammas, self.unit_dims, strict=True)]
        )

    # ------------------------------------------------------------- dynamics

    def step(self, state: tuple, dt: float, gamma=0.0) -> tuple:
        """
        One global dissipative-Verlet step on the concatenated state (F5 §7.2
        condition 2). ``gamma`` is a scalar (uniform, conformal) or a (D,)
        per-coordinate vector from ``gamma_vector`` (heterogeneous, flagged).
        """
        q, p = state
        return velocity_verlet_step(self.H, q, p, dt, gamma)

    def stochastic_step(
        self,
        state: tuple,
        dt: float,
        gamma: float,
        temperature: float,
        key: jax.random.PRNGKey,
        noise_mode: str = "legacy",
    ) -> tuple:
        """One joint Langevin step (scalar gamma only; mirrors CHLU.stochastic_step)."""
        q, p = state
        m_eff = self.effective_mass() if noise_mode == "fdt" else None
        return langevin_step(
            self.H,
            q,
            p,
            dt,
            gamma,
            temperature,
            key,
            noise_mode=noise_mode,
            m_eff=m_eff,
        )

    def __call__(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        steps: int,
        dt: float,
        gamma=0.0,
    ) -> jnp.ndarray:
        """
        Deterministic joint rollout via lax.scan (pure — reversible-BPTT
        compatible). Returns (steps, 2*D) post-step states, exactly the
        CHLU.__call__ contract.
        """

        def scan_fn(state, _):
            q, p = state
            q_next, p_next = self.step((q, p), dt, gamma)
            return (q_next, p_next), jnp.concatenate([q_next, p_next])

        _, trajectory = jax.lax.scan(scan_fn, (q0, p0), None, length=steps)
        return trajectory


# ---------------------------------------------------------------------------
# Config-friendly builder
# ---------------------------------------------------------------------------


def chain_edges(n_units: int) -> tuple:
    """Chain topology (0,1), (1,2), ..., (N-2, N-1)."""
    return tuple((i, i + 1) for i in range(n_units - 1))


def build_lattice(
    key: jax.random.PRNGKey,
    unit_dims: Sequence[int],
    hidden: int = 32,
    potential_type: str = "mlp",
    kinetic_mode: str = "newtonian_learned",
    mass_scales: Optional[Sequence[float]] = None,
    edges: Optional[Sequence[tuple]] = None,
    coupling_type: str = "auto",
    kappa_c: float = 0.05,
    coupling_dim: int = 2,
    coupling_hidden: int = 16,
    proj_init_scale: float = 0.1,
    proj_init_mode: str = "random",
    wormhole_edges: Sequence[tuple] = (),
    wormhole_gate_threshold: float = 1.0,
    wormhole_gate_width: float = 0.25,
    gate_energy_mode: str = "free_energy",
    rest_mass: float = 1.0,
    c: float = 1.0,
    tie_channel_mass: bool = False,
) -> CLULattice:
    """
    Build a CLU lattice from config-level knobs.

    Args:
        key: PRNG key (split across units and couplings).
        unit_dims: per-unit dims d_i (len = N).
        hidden: hidden width of each unit's potential net.
        potential_type: per-unit potential ("mlp", "deep_mlp", "so2_invariant").
        kinetic_mode: shared kinetic mode for all units.
        mass_scales: optional per-unit inertial_mass_scale — the DESIGNED mass
            banding (heavy/slow backbone vs light/fast perception; Thread-5 +
            wave-2: the hierarchy must be designed-in). Exact softplus-space
            scaling; requires a log_mass-reading kinetic mode.
        edges: coupling edge list; default = chain. Non-adjacent pairs are
            legal here too (ungated); ``wormhole_edges`` adds GATED non-local
            pairs (F5 §7.4 smooth gate).
        coupling_type: one of

            - ``"auto"`` (default) — ``"channel_spring"`` for ``so2_invariant``
              units, ``"spring"`` for every other potential type. This is the
              theorist's design rule (``xy-lattice-theory`` P5): an SO(2)
              lattice must be coupled U(1)-symmetrically or its ``p=2``
              anisotropy (a RELEVANT perturbation) destroys the KT phase.
            - ``"spring"`` — quadratic, learnable W (see ``proj_init_mode``).
            - ``"channel_spring"`` — quadratic, FIXED identity-on-channel W.
              Exact XY reduction ``V = sum 2 kappa r*^2 (1 - cos dtheta)``,
              ``J = 2 kappa r*^2``. Not trainable (no coupling parameters).
            - ``"mlp"`` — learned nonlinear position-only coupling.

        kappa_c: coupling strength (static knob; kappa_c = 0 with spring
            coupling reduces the dynamics exactly to independent units).
        coupling_dim / coupling_hidden / proj_init_scale: coupling shapes.
            ``coupling_dim`` also selects the channel ``(0, ..., coupling_dim-1)``
            for ``"channel_spring"``.
        proj_init_mode: ``"random"`` (legacy, U(1)-breaking) or ``"conformal"``
            (``W = 1_k`` at init, still trainable) — see ``spring_coupling``.
            Only used by ``coupling_type="spring"``.
        wormhole_edges: distant (i, j) pairs to couple through a smooth
            energy gate (skeleton — no top-k selection logic).
        gate_energy_mode: ``"free_energy"`` (default, correct annealed
            potential) or ``"mean_energy"`` (legacy) — see ``GatedCoupling``.
        rest_mass, c, tie_channel_mass: forwarded to each CHLU unit.

    Returns:
        CLULattice.
    """
    unit_dims = tuple(int(d) for d in unit_dims)
    n = len(unit_dims)
    if edges is None:
        edges = chain_edges(n)
    edges = tuple((int(i), int(j)) for (i, j) in edges)
    wormhole_edges = tuple((int(i), int(j)) for (i, j) in wormhole_edges)

    if mass_scales is not None and len(mass_scales) != n:
        raise ValueError(f"mass_scales has {len(mass_scales)} entries for {n} units")
    if gate_energy_mode not in ("free_energy", "mean_energy"):
        raise ValueError(
            f"Unknown gate_energy_mode: {gate_energy_mode!r}. "
            "Must be 'free_energy' or 'mean_energy'."
        )

    # Design rule (xy-lattice-theory P5): U(1)-preserving coupling for SO(2)
    # units. "auto" preserves today's behavior for every non-so2 potential.
    if coupling_type == "auto":
        coupling_type = (
            "channel_spring" if potential_type == "so2_invariant" else "spring"
        )
    elif (
        coupling_type == "spring"
        and potential_type == "so2_invariant"
        and proj_init_mode == "random"
    ):
        warnings.warn(
            "Building an so2_invariant lattice with a random-W spring_coupling: "
            "the learnable projections break the lattice's global U(1). At init "
            "P(J < 0) = 0.52 and median h_2/|J| = 1.00; after 400 epochs on "
            "U(1)-symmetric data J/J_true ~ 0.02 while h_2/|J| = 0.6-2.1. The "
            "p=2 anisotropy is a RELEVANT perturbation at the KT fixed point "
            "(x_2 = 1/2) and destroys any 2-D memory phase. Use "
            "coupling_type='channel_spring' (or 'auto'), or "
            "proj_init_mode='conformal'.",
            UserWarning,
            stacklevel=2,
        )

    n_couplings = len(edges) + len(wormhole_edges)
    keys = jax.random.split(key, n + max(n_couplings, 1))
    unit_keys, coupling_keys = keys[:n], keys[n:]

    units = []
    for i, d in enumerate(unit_dims):
        unit = CHLU(
            dim=d,
            hidden=hidden,
            rest_mass=rest_mass,
            c=c,
            kinetic_mode=kinetic_mode,
            potential_type=potential_type,
            tie_channel_mass=tie_channel_mass,
            key=unit_keys[i],
        )
        if mass_scales is not None:
            unit = scale_inertial_mass(unit, float(mass_scales[i]))
        units.append(unit)

    def make_coupling(i, j, k):
        if coupling_type == "spring":
            return spring_coupling(
                unit_dims[i],
                unit_dims[j],
                kappa_c,
                coupling_dim=coupling_dim,
                init_scale=proj_init_scale,
                key=k,
                init_mode=proj_init_mode,
            )
        elif coupling_type == "channel_spring":
            return channel_spring_coupling(
                unit_dims[i],
                unit_dims[j],
                kappa_c,
                channel=tuple(range(coupling_dim)),
            )
        elif coupling_type == "mlp":
            return MLPCoupling(
                unit_dims[i], unit_dims[j], kappa_c, hidden=coupling_hidden, key=k
            )
        else:
            raise ValueError(
                f"Unknown coupling_type: {coupling_type}. Must be 'auto', "
                "'spring', 'channel_spring' or 'mlp'."
            )

    couplings = [
        make_coupling(i, j, coupling_keys[e]) for e, (i, j) in enumerate(edges)
    ]
    for w, (i, j) in enumerate(wormhole_edges):
        base = make_coupling(i, j, coupling_keys[len(edges) + w])
        couplings.append(
            GatedCoupling(
                base=base,
                threshold=float(wormhole_gate_threshold),
                width=float(wormhole_gate_width),
                energy_mode=gate_energy_mode,
            )
        )

    return CLULattice(units=units, edges=edges + wormhole_edges, couplings=couplings)
