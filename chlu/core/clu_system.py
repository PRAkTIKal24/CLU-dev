"""The full-CLU synthesis harness: every lever live at once, in one object.

**What "full CLU" means here** (intervention §4 — the config that must be
*simultaneously* live, and the point of C2W1):

* **Write** — items held in a **learned ``V_theta``**, not explicit per-item
  arrays; derived addressing; an admission policy; per-item lifetimes; a
  local/masked write; **permitted basin interaction**.
* **Read** — learned ``phi`` in; two-phase relaxation; mass as selector;
  **trajectory *and* settled point available to the read-out**; confidence-gated
  retry.
* **Structure** — the causal limit as a real constraint; flat directions / trash
  regions / wormholes wired as hooks, deployed only when a monitor fires.
* **Control** — controller v0 over the designed verb set.

⭐ **The one API decision that makes pillar 1 testable at all:** :meth:`read`
returns the **trajectory**, not only the settled point — a strided buffer plus
``q*`` — so a trajectory read-out is a *configuration*, not a rewrite. In 26
waves every experiment used settled points only.

**API FREEZE (C2W1).** ``memory-gym-v0`` and ``trainability-spike`` branch off
this surface: :class:`CluSystemConfig`, :class:`CluSystem`, :class:`ReadResult`,
:class:`ReadState`, :class:`LearnedVStore`, :func:`settled_point_psi`.

**RACE+SEAM FREEZE (C2W2).** Two additive seams, both default-off and both
asserted bit-identical-when-off in ``tests/test_clu_system.py``, so the wave's
other two engineer branches build on this file without editing it:

* **(a) the write-objective passthrough** — ``CluSystem(write_objective=...)`` /
  ``build_system(write_objective=...)``, normalised by
  :func:`normalize_write_objective`, forwarded verbatim to
  :func:`~chlu.training.train_memory.train_memory_landscape`. This is how the
  C2W2 Route-1 coefficients reach the write **without changing a single**
  :class:`CluSystemConfig` **semantic**.
* **(b) the store-potential factory hook** —
  ``CluSystemConfig.store_potential_factory`` (an import path, resolved by
  :func:`resolve_store_potential_factory`) + ``store_potential_kwargs``, so a
  brand-new store family (``ssb-shell-atoms``'s shell atoms) is registered from
  config alone.
* **(c) the store-WRITE-MASK factory hook** (C2W3 rider A) —
  ``CluSystemConfig.store_write_mask_factory`` (+ ``store_write_mask_kwargs``),
  resolved by :func:`resolve_store_write_mask_factory`. Seam (b) without it was
  half a seam: a new family could supply its potential but **not** its update
  mask, so any leaf outside ``learned.{centers,log_width,amp}`` was written
  unmasked and **C3 locality broke silently**.

**Hook for ``trainability-spike``:** :class:`CluSystem` takes
``psi: Callable[[Trajectory, ReadState], Array]`` and the settle exposes its
**fixed-point residual** (:meth:`CluSystem.fixed_point_residual`), so an
implicit/DEQ gradient can attach at the settled point. A handcrafted psi
(settled-point linear read, :func:`settled_point_psi`) is the v0 default; the
learned psi is `trainability-spike`'s to build, not this task's.

⚠ **Config lives here, not in ``chlu/config.py``** (C2W1 file-ownership rule:
C1W27 owns two blocks of that file this wave and it is the single most likely
cross-campaign merge conflict). :class:`CluSystemConfig` is a plain dataclass
with a ``from_mapping`` override path, so the harness stays config-driven
without touching the shared config module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.clu_controller import (
    CluControllerV0,
    ControllerPolicy,
    derived_d_safe,
)
from chlu.core.controller import Controller
from chlu.core.memory_potentials import (
    DesignFreedomPotential,
    atom_write_mask_fn,
    designed_sites,
)
from chlu.core.monitors import (
    MonitorContext,
    default_registry,
    erf_margin_accuracy,
    saddle_reach_threshold,
)
from chlu.core.potentials import PotentialMLP  # noqa: F401  (documented family)
from chlu.experiments.goldstone_harness import clu_with_potential
from chlu.training.train_memory import train_memory_landscape

#: A strided read trajectory: ``(n_points, 2 * dim)`` = ``[q | p]`` per point.
Trajectory = jnp.ndarray


@dataclass
class CluSystemConfig:
    """Every knob of the harness, with its known-productive-band default.

    Staged activation (charter §3.1): each lever starts in the band 26 waves
    measured, and is freed **one at a time inside the running full system**.
    The ``stage_*`` flags below are what the experiment sweeps; the defaults are
    stage 0 (all levers in-band, nothing freed).
    """

    # -- geometry ---------------------------------------------------------
    addr_dim: int = 4  # d: the address ball (w26 unclamped d=4)
    payload_dim: int = 1  # m: payload channels (m>1 = the multi-channel code)
    n_spectator: int = 0
    ball_radius: float = 1.0

    # -- the learned V_theta store (the w22-w26 KNOWN PRODUCTIVE BAND) ----
    # Every default below is the shipped `ExperimentDesignedMechanismConfig`
    # value, so stage 0 starts where 26 waves measured the levers, not at a
    # guess (charter §3.1). Deviations are declared in the report.
    capacity: int = 8
    atoms_per_item: int = 32  # shipped: n_atoms = atoms_per_item * K
    min_atoms: int = 384  # hard floor
    min_atoms_base: int = 512  # dimension-aware floor: base * c**d
    min_atoms_c: float = 1.4142135623730951  # sqrt(2), empirically anchored (w23)
    atom_width: float = 0.3  # atom_init_width (NOT the designed well width 0.15)
    atom_depth_init: float = 1e-4  # flat start; the writer digs the wells
    atom_init_scale: float = 1.0  # LOAD-BEARING: 0.5 caps strict at 0.5 (basin reach)
    atom_local_radius: float = 0.0  # N98 localized init; 0.0 = historical scatter
    confine: float = 0.05  # learned_confine (coercivity alpha)

    # -- C2W8 PASS 2, ARM A: how far each atom's influence REACHES ----------
    # ⭐ Pass 1's diagnosis: C3 locality HOLDS in parameter space (own-leg
    # violation rate 0.000) and FAILS in function space (78-84 % of writes raise
    # the foreign contribution; foreign exceeds own on 45 of 48 wells). A write
    # touches only its own atom block, but **atoms have TAILS**. `atom_kernel`
    # selects the influence profile (:func:`chlu.core.memory_potentials.atom_profile`):
    # `"wendland"` / `"truncated_gaussian"` are COMPACT — identically zero beyond
    # `R = atom_kernel_cutoff * s` — which is the K2 compact-gate lesson applied
    # one level down, to the atoms.
    # ⛔ `"gaussian"` (the default) is the shipped kernel: **bit-identical AND
    # parameter-count-identical** to `main @ 80d7d4b` (K6; asserted in
    # `tests/test_compact_atoms.py`). Both new fields are static/parameter-free.
    atom_kernel: str = "gaussian"
    atom_kernel_cutoff: float = 2.5  # support radius in units of the atom width s
    # ⭐ The companion lever a compact kernel needs to be runnable at all: at the
    # shipped scattered init the nearest of ALL atoms to a site is 0.738 at
    # addr_dim = 8 (`ERRATA-C2W8.md` §3), so a co-scaled compact atom feels
    # nothing and the write gradient is exactly zero. This re-draws the admitted
    # slot's atom CENTERS into a ball around the item's own address **at
    # admission time** — the N98 localized init moved from build time (where a
    # phi-addressed stream cannot use it) to the moment the address is known.
    # ⛔ It initialises atom parameters; it does NOT pin, snap or regularize the
    # attractor toward phi(item) — the write still learns depth/width/center, the
    # settled point stays free and basins stay free to interact.
    # ⛔ `False` (the default) => the shipped admission path, bit-identical.
    atom_site_local_init: bool = False
    atom_site_local_radius: float = 0.0  # 0.0 => use `atom_width`

    # -- the write ---------------------------------------------------------
    write_steps: int = 300
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    write_n_perturb: int = 32
    # ⚠ Masked/local writes are REQUIRED by "full CLU" (intervention §4) and are
    # the only form a stream supports, but they are NOT the learned arm's best
    # single-shot fidelity write: w22 measured atoms global 1.000 vs local 0.859
    # at K=4. That gap is a declared property of this configuration, not a bug.
    masked_write: bool = True  # local in parameter space (C3-local)

    # -- the read ----------------------------------------------------------
    dt: float = 0.05
    # Shipped two-phase band: phase 2 REQUIRES dissipation (payload err 0.57 at
    # gamma_read=0 vs ~1e-6 at 0.02 — the payload is launched at 0 and must
    # dissipate up to a_i).
    gamma_address: float = 0.05
    gamma_read: float = 0.02
    address_steps: int = 400
    read_steps: int = 800
    traj_stride: int = 8  # the strided trajectory buffer
    kinetic_mode: str = "newtonian_learned"
    query_sigma: float = 0.15  # sigma_q (shipped query jitter)

    # -- control -----------------------------------------------------------
    d_safe_kappa_prime: float = 2.576  # 99% point of the corrected margin law
    # ⚠ An explicit admission radius, used ONLY to take the gate deliberately out
    # of its designed band so basins are allowed to interact (intervention §3.3).
    # When set, the merge certificate `2 s_max + kappa' sigma_q <= d_safe` is
    # KNOWINGLY violated and monitor #8 is expected to say so — that is the
    # measurement, not a bug. None = the derived (in-band) radius.
    d_safe_override: Optional[float] = None
    budget: Optional[int] = None  # live-item budget (None => capacity)
    leak: float = 0.0  # per-tick decay of a non-permanent item
    amp_floor: float = 0.05
    retry_tau: float = 0.5
    retry_max_rounds: int = 0
    anneal_payload_mult: float = 1.0
    anneal_stages: int = 1

    # -- staged activation flags ------------------------------------------
    stage_lifetimes: bool = False
    stage_admission: bool = False
    stage_capacity_pressure: bool = False
    stage_deletion: bool = False
    stage_basin_interaction: bool = False
    stage_retry: bool = False
    stage_trajectory_read: bool = False

    # -- C2W2 SEAM (b): the store-potential factory hook -------------------
    # ⭐ An import path so a NEW store family can be registered from ANOTHER
    # module without editing this file (C2W2 file-ownership: `ssb-shell-atoms`
    # registers the shell-atom family through this hook and edits nothing it does
    # not own). Format ``"pkg.module:factory"`` or ``"pkg.module.factory"``; the
    # callable is invoked as ``factory(cfg=cfg, key=key, **store_potential_kwargs)``
    # and must return an ``eqx.Module`` with the ``DesignFreedomPotential``
    # surface :class:`LearnedVStore` uses: a ``.learned`` subtree exposing
    # ``centers``, ``amp``, ``log_width``, ``n_groups`` and ``group_rows(slot)``.
    # ⛔ ``None`` (the default) => the shipped ``DesignFreedomPotential`` path,
    # **bit-identical** (asserted in ``tests/test_clu_system.py``).
    store_potential_factory: Optional[str] = None
    store_potential_kwargs: Dict[str, Any] = field(default_factory=dict)

    # -- C2W3 RIDER A: the matching WRITE-MASK hook ------------------------
    # ⭐ ``store_potential_factory`` lets a new store family supply its own
    # potential; without a matching hook it could **not** supply its own update
    # mask, so the write fell back to :func:`atom_write_mask_fn`, which masks
    # exactly ``learned.{centers,log_width,amp}``. Any leaf a new family carries
    # outside those three is left **unmasked** — it is updated by every write, so
    # writing item ``j`` moves a parameter item ``i``'s read depends on and
    # **C3 locality breaks** (``tests/test_clu_system.py`` asserts both halves:
    # the unmasked leaf breaks it, the family's own mask restores it).
    #
    # Resolved by the same ``pkg.module:attr`` mechanism as
    # :func:`resolve_store_potential_factory`, and invoked as::
    #
    #     factory(cfg=cfg, store=store, slot=slot, default_mask_fn=default,
    #             **store_write_mask_kwargs)  ->  (updates -> updates) | None
    #
    # ``default_mask_fn`` is the shipped row mask (or ``None`` when
    # ``masked_write`` is off), so a family can *compose* with it rather than
    # replace it. ⛔ ``None`` (the default) => the shipped write path,
    # **bit-identical** (asserted in ``tests/test_clu_system.py``).
    store_write_mask_factory: Optional[str] = None
    store_write_mask_kwargs: Dict[str, Any] = field(default_factory=dict)

    # -- C2W3 RIDER B: the merge certificate as a MONITORED SOFT CONSTRAINT --
    # ⭐ SC-1…SC-7 (`doctrine-repairs.md` §4.4, Head ruling §A9.8). The shipped
    # harness sets `d_safe := 2 s_max + kappa' sigma_q`, so **the admission
    # radius IS the certificate radius** and the two "mutually exclusive" bands
    # are one object. With this ON, `d_safe = zeta * sep_expected` (SC-1) is an
    # independent declared radius, `R_cert` is still computed and reported but is
    # no longer the gate, and a violation of the budget `B` **TRIPS monitor #3
    # rather than refusing a write** (SC-3) — a soft constraint that refuses is a
    # hard constraint with extra steps.
    # ⛔ `False` (the default) => the shipped harness, **bit-identical**
    # (blocking regression test in `tests/test_soft_certificate.py`).
    # ⚠ The price is carried, unsoftened: `rho_ex` up to 6.3x at a `lambda_min`
    # cost of 2.2-6.0x, and the dividend in that region stays ~0. It is a
    # PRECONDITION, not a result.
    soft_certificate: bool = False
    soft_certificate_kwargs: Dict[str, Any] = field(default_factory=dict)

    # -- C2W8: the TRASH REGION gamma_phi(q) — its genuine FIRST USE ---------
    # ⭐ `chlu/core/friction_field.py` was built in C1 and is referenced NOWHERE
    # in this file: the trash region has never been wired into the CLU's settle.
    # This is the plumbing (the field itself is unchanged — Prop-11 gives its
    # exact volume contraction, and the CHLU unit already composes
    # `1 - (1 - gamma)(1 - gamma_phi(q))` inside `velocity_verlet_step`).
    #
    # ⛔ `gamma_phi = False` (the default) => `model()` attaches NO field, so the
    # read path is **bit-identical AND parameter-count-identical** to the
    # pre-C2W8 path (the P1 / psires precedent; both halves asserted in
    # `tests/test_well_lifecycle.py::test_gamma_phi_off_is_*`).
    #
    # ⚠ The default gate is **"compact"**, not the field's own "sigmoid"
    # default: the compact smoothstep is identically 0 beyond `r_k`, so a hole
    # placed far from every well leaves every read bit-identical *exactly*
    # rather than to 1e-30. A trash region that leaks a tail everywhere is not a
    # trash region, it is a global friction change (K2 designed negative (b)).
    #
    # ⚠ **gamma_phi holes are BYTES** and are on the ledger (:meth:`n_bytes`):
    # `K x (dim centers + radius + strength)`. A trash region off the ledger is
    # a hidden capacity increase (the §A9.6 ledger-drift collapse mode).
    gamma_phi: bool = False
    gamma_phi_gate: str = "compact"  # compact => EXACT zero outside r_k
    gamma_phi_max: float = 0.5
    gamma_phi_width: float = 0.05
    gamma_phi_radius: float = 0.2
    gamma_phi_strength: float = 0.25
    gamma_phi_trainable: bool = False  # the trash region is PLACED, not fitted

    # -- harness -----------------------------------------------------------
    seed: int = 0
    n_query_per_item: int = 8  # shipped harness uses 32; 8 declared for cost
    payload_tol: float = 0.1
    quick: bool = False

    @classmethod
    def from_mapping(cls, overrides: Optional[dict] = None) -> "CluSystemConfig":
        """Build from a YAML/JSON mapping, ignoring unknown keys.

        Mirrors ``chlu.config.load_config``'s tolerance so an old project file
        does not crash a new schema. This is the config-driven override path for
        ``projects/<name>/config/config.yaml`` under a ``clu_system:`` block.
        """
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in dict(overrides or {}).items() if k in known}
        return cls(**kw)

    def as_flag_table(self) -> Dict[str, Any]:
        """Every non-default flag in effect — the flag-provenance table."""
        base = CluSystemConfig()
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v != getattr(base, f.name):
                out[f.name] = v
        return out

    @property
    def dim(self) -> int:
        return int(self.addr_dim + self.payload_dim + self.n_spectator)

    def soft_cert_config(self):
        """The :class:`~chlu.core.soft_certificate.SoftCertificateConfig` in force.

        Lives in ``soft_certificate.py`` (the C2-owned module that uses it) and
        is assembled here so ``chlu/config.py`` is untouched.
        """
        from chlu.core.soft_certificate import SoftCertificateConfig

        return SoftCertificateConfig(enabled=bool(self.soft_certificate),
                                     **dict(self.soft_certificate_kwargs or {}))

    @property
    def n_atoms(self) -> int:
        """``max(atoms_per_item*K, min_atoms, base*c**d)`` — the w23 dimension-aware
        atom floor. Scaling the budget with K only starves the write at high ``d``
        (the fraction of atoms landing near any site decays ~geometrically per
        added dimension), and a starved cell reads as a capacity result when it is
        an optimizer artefact."""
        geo = round(self.min_atoms_base * self.min_atoms_c ** self.addr_dim)
        n = max(self.atoms_per_item * self.capacity, self.min_atoms, int(geo))
        # keep it a multiple of the group count so every item owns equal blocks
        return int(self.capacity * int(np.ceil(n / self.capacity)))


class ReadState(eqx.Module):
    """The state a read exposes to ``psi`` alongside the trajectory.

    ``q0`` launch (= ``phi(x)``), ``q_addr`` the phase-1 settled address,
    ``q_star``/``p_star`` the final settled point.
    """

    q0: jnp.ndarray
    p0: jnp.ndarray
    q_addr: jnp.ndarray
    p_addr: jnp.ndarray
    q_star: jnp.ndarray
    p_star: jnp.ndarray


@dataclass
class ReadResult:
    """What :meth:`CluSystem.read` returns. **The trajectory is first-class.**

    Attributes:
        value: ``psi(traj, state)`` — the read-out.
        traj: ``(B, n_points, 2*dim)`` strided trajectory buffer (phases 1+2).
        phase: ``(n_points,)`` 1/2 marker per trajectory point.
        state: :class:`ReadState` (``q0``, ``q_addr``, ``q_star``, momenta).
        confidence: label-free per-query confidence used by the retry gate —
            the **settle residual**, never the post-settle energy (N97).
        residual: ``|grad V(q*)|`` per query (also the DEQ fixed-point residual).
        rho_conv: ``|grad V(q*)| / |grad V(q0)|`` per query (monitor #1).
        n_steps: total Verlet steps actually spent (the compute-dial axis).
        retries: how many retry rounds fired.
        diagnostics: everything the monitors read.
    """

    value: jnp.ndarray
    traj: Trajectory
    phase: np.ndarray
    state: ReadState
    confidence: np.ndarray
    residual: np.ndarray
    rho_conv: np.ndarray
    n_steps: int
    retries: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteReport:
    """The result of :meth:`CluSystem.write_stream`."""

    admitted: List[int] = field(default_factory=list)
    refused: List[int] = field(default_factory=list)
    evicted: List[int] = field(default_factory=list)
    deleted: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    readings: List[Any] = field(default_factory=list)
    log: List[dict] = field(default_factory=list)


@dataclass
class ConsolidationReport:
    """The result of :meth:`CluSystem.consolidate` — the offline maintenance
    phase (charter §2.4 repositions wake-sleep as consolidation: re-packing,
    decay enforcement, gate re-calibration, certificate re-check)."""

    certificates: Dict[str, Any] = field(default_factory=dict)
    readings: List[Any] = field(default_factory=list)
    self_probe: Dict[str, Any] = field(default_factory=dict)
    n_moves: int = 0


class LearnedVStore(eqx.Module):
    """Items held in a **learned ``V_theta``**, with a codebook of addresses.

    This is the post-w26 unclamping and the whole point of the task: the store is
    a learned :class:`~chlu.core.memory_potentials.AtomDictionaryPotential` whose
    atoms are grouped one block per item slot, so a write is **local in parameter
    space** (C3-local by construction) and a per-item lifetime is a physical
    shallowing of that item's own atoms.

    **What is and is not an array** (this distinction is the task):

    * the **payload** ``a_i`` lives **only** in the landscape. The read must
      recover it from ``V_theta``; nothing in the read path may consult a stored
      value. The codebook's payload column exists for **eval/launder/monitors
      only** and is asserted never-read-by-``read`` in ``tests/test_clu_system.py``.
    * the **derived address** ``c_i`` IS retained per live item
      (`controller-doctrine` I-1): without it the same-keys launder cannot be
      constructed and **monitor #2 has no runtime form at all**. It is also the
      exact object the launder is given, so retaining it is conservative — it
      *strengthens* the control we are measured against.
    """

    V: eqx.Module
    dim: int = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    atoms_per_item: int = eqx.field(static=True)

    def __init__(self, cfg: CluSystemConfig, key):
        self.dim = int(cfg.dim)
        self.addr_dim = int(cfg.addr_dim)
        self.payload_dim = int(cfg.payload_dim)
        self.capacity = int(cfg.capacity)
        self.atoms_per_item = int(cfg.n_atoms // max(cfg.capacity, 1))
        # -- C2W2 SEAM (b): an externally-registered store family --------------
        if cfg.store_potential_factory:
            factory = resolve_store_potential_factory(cfg.store_potential_factory)
            self.V = factory(cfg=cfg, key=key, **dict(cfg.store_potential_kwargs or {}))
            return
        # rung "free_mlp" + family "atoms" => designed part is None, so the
        # landscape is ENTIRELY learned (coercivity only). One atom group per
        # item slot makes the write local in parameter space.
        self.V = DesignFreedomPotential(
            rung="free_mlp",
            dim=self.dim,
            payloads=jnp.zeros((self.capacity,)),
            key=key,
            learned_family="atoms",
            n_atoms=int(cfg.n_atoms),
            rbf_init_width=float(cfg.atom_width),
            confine=float(cfg.confine),
            atom_depth_init=float(cfg.atom_depth_init),
            atom_groups=self.capacity,
            atom_init_scale=float(cfg.atom_init_scale),
            atom_kernel=str(cfg.atom_kernel),
            atom_kernel_cutoff=float(cfg.atom_kernel_cutoff),
        )

    # -- introspection -----------------------------------------------------
    @property
    def atoms(self):
        return self.V.learned

    def group_rows(self, slot: int) -> jnp.ndarray:
        """Boolean atom-row mask owned by ``slot`` (the masked-write support)."""
        return self.V.learned.group_rows(int(slot))

    def group_stats(self, slot: int, center) -> Tuple[float, float]:
        """``(D_i, s_i)`` of the item's own wells, read off the learned atoms.

        ``D_i`` is the depth its own atoms contribute **at its recorded site**
        and ``s_i`` the depth-weighted width — the ``(D, s)`` pair the saddle
        reach criterion needs, taken from the landscape rather than fitted.
        """
        m = np.asarray(self.group_rows(slot), dtype=bool)
        A = np.asarray(self.V.learned.amp, dtype=float)[m] ** 2
        s = np.exp(np.asarray(self.V.learned.log_width, dtype=float)[m])
        c = np.asarray(self.V.learned.centers, dtype=float)[m]
        z = np.zeros((self.dim,), dtype=float)
        z[: self.addr_dim] = np.asarray(center, dtype=float)[: self.addr_dim]
        d2 = np.sum((c - z[None, :]) ** 2, axis=-1)
        w = A * np.exp(-d2 / (2.0 * s**2 + 1e-12))
        D = float(np.sum(w))
        s_eff = float(np.sum(w * s) / max(np.sum(w), 1e-12)) if D > 0 else float(np.mean(s))
        return D, s_eff

    def scale_group_amplitude(self, slot: int, factor: float) -> "LearnedVStore":
        """Multiply that item's atom **depths** by ``factor`` (per-item decay).

        ``A_j = amp_j^2``, so the amplitude parameter is scaled by
        ``sqrt(factor)``. This is the physical form of a lifetime: the item's own
        wells shallow, nothing else in ``V_theta`` moves.
        """
        m = jnp.asarray(self.group_rows(slot), dtype=jnp.float32)
        scale = jnp.where(m > 0, jnp.sqrt(jnp.asarray(float(factor))), 1.0)
        V = eqx.tree_at(lambda t: t.learned.amp, self.V, self.V.learned.amp * scale)
        return eqx.tree_at(lambda t: t.V, self, V)

    def reinit_group(self, slot: int, key, cfg: "CluSystemConfig") -> "LearnedVStore":
        """Eviction/deletion: **re-draw** the freed group from the init distribution.

        ⚠ Measured, and it is a mechanism, not a detail. Zeroing the freed rows
        (the obvious "leave no trace") is wrong twice over:

        1. **It starves the next item in that slot.** Atoms at the origin cannot
           reach a well whose payload is ``|a_i| ~ 1`` from the payload-zero launch
           manifold — the ``atom_init_scale`` lesson (0.5 caps strict at 0.500 at
           d=2 K=4 regardless of atom count). Measured here: every item written
           into a recycled slot came out with fitted depth ``D = 0.00`` and was
           unretrievable, dragging self-probe acquisition to 0.33.
        2. **It is a membership leak.** A zeroed row is *distinguishable* from a
           never-used row (which holds a scattered draw at ``amp = sqrt(1e-4)``),
           so "this slot once held something" survives the eviction. Re-drawing
           from the same distribution a fresh slot uses makes an evicted slot
           statistically indistinguishable from a never-written one.
        """
        m = np.asarray(self.group_rows(slot), dtype=bool)
        n = int(m.sum())
        if n == 0:
            return self
        k_c, k_a = jax.random.split(key, 2)
        rows = jnp.asarray(np.nonzero(m)[0])
        centers = jax.random.normal(k_c, (n, self.dim)) * float(cfg.atom_init_scale)
        new_centers = self.V.learned.centers.at[rows].set(centers)
        new_amp = self.V.learned.amp.at[rows].set(float(cfg.atom_depth_init) ** 0.5)
        new_w = self.V.learned.log_width.at[rows].set(float(np.log(cfg.atom_width)))
        V = eqx.tree_at(
            lambda t: [t.learned.centers, t.learned.amp, t.learned.log_width],
            self.V, replace=[new_centers, new_amp, new_w],
        )
        return eqx.tree_at(lambda t: t.V, self, V)

    def n_bytes(self) -> int:
        """Bytes of learned state (float32) — the matched-bytes denominator."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self.V, eqx.is_inexact_array))
        return int(sum(int(np.asarray(x).size) for x in leaves) * 4)


class CluSystem:
    """Assemble a full CLU from a store, a ``phi``, a ``psi``, a controller and a
    monitor registry, and run write/read streams on it.

    Args:
        store: a :class:`LearnedVStore`.
        phi: read-in ``x -> q0`` (``(B, ...) -> (B, dim)``). ``None`` => identity
            on an already-embedded query.
        psi: read-out ``(traj, state) -> value``. Defaults to
            :func:`settled_point_psi` — the handcrafted v0 read. A learned psi is
            `trainability-spike`'s.
        controller: :class:`~chlu.core.clu_controller.CluControllerV0`.
        registry: :class:`~chlu.core.monitors.MonitorRegistry`.
        config: :class:`CluSystemConfig`.
        write_objective: ⭐ **C2W2 SEAM (a)** — an *objective spec* handed
            straight to :func:`~chlu.training.train_memory.train_memory_landscape`
            on every write, **without changing a single**
            :class:`CluSystemConfig` **semantic**. See
            :func:`normalize_write_objective`. ``None`` (the default) => the
            shipped write, bit-identical.
    """

    def __init__(
        self,
        store: LearnedVStore,
        phi: Optional[Callable] = None,
        psi: Optional[Callable[[Trajectory, ReadState], jnp.ndarray]] = None,
        controller=None,
        registry=None,
        config: Optional[CluSystemConfig] = None,
        write_objective: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = config or CluSystemConfig()
        self.write_objective = normalize_write_objective(write_objective)
        self.store = store
        self.phi = phi
        self.psi = psi or settled_point_psi(store.addr_dim, store.payload_dim)
        self.registry = registry if registry is not None else default_registry()
        self.controller = controller if controller is not None else make_controller(
            self.cfg, self.registry
        )
        # eval-only bookkeeping (NEVER read by ``read``)
        self._payloads: Dict[int, np.ndarray] = {}
        self._born: Dict[int, int] = {}
        self._t = 0
        self._write_log: List[dict] = []
        self._losses: List[float] = []
        #: endpoint-only write loss per write (C2W2 gate ruling (i)); equals
        #: ``_losses`` exactly when no Route-1 coefficient is set.
        self._endpoint_losses: List[float] = []
        self._prev_store: Optional[LearnedVStore] = None
        self._c3_ratio = float("nan")
        self._oldest_drop = float("nan")
        #: per-pair (B, delta, lambda_min) of the last write — monitor #3's
        #: replacement leg (C2W3 D3) and SC-4(ii) read this.
        self._c3_pairs: List[dict] = []
        #: SC-2: one `deficit_rel` per admitted write, accumulated over the run.
        self._cert_deficits: List[float] = []
        self._knob_reads: Dict[str, int] = {}
        self._evict_key = jax.random.PRNGKey(self.cfg.seed + 5150)
        self._wall0 = time.time()
        #: ⭐ C2W8: the trash region. ``None`` unless ``cfg.gamma_phi`` — and
        #: ``None`` is what makes OFF bit-identical AND parameter-count-identical
        #: (no leaf is added to the read model at all). Holes are placed by
        #: :meth:`trash_route`, never fitted by a loss.
        self.trash: Optional[eqx.Module] = None
        if self.cfg.gamma_phi:
            from chlu.core.friction_field import FrictionField

            # ⚠ float32, unconditionally: the read casts `q0` to float32 (see
            # :meth:`read`), and under `jax_enable_x64` a float64 field would
            # promote `p` inside the damping and break the scan carry's dtype
            # ("carry input float32[d] vs output float64[d]"). Found by §7.23's
            # ordering hazard, pinned in `tests/test_well_lifecycle.py`.
            self.trash = _as_f32(FrictionField(
                dim=self.store.dim, centers=jnp.zeros((0, self.store.dim)),
                gamma_max=float(self.cfg.gamma_phi_max),
                width=float(self.cfg.gamma_phi_width),
                gate=str(self.cfg.gamma_phi_gate),
                trainable=bool(self.cfg.gamma_phi_trainable),
            ))

    # -- model -------------------------------------------------------------
    def model(self, store: Optional[LearnedVStore] = None,
              payload_width_mult: float = 1.0):
        """The CLU wired to the learned landscape (optionally width-annealed)."""
        st = store or self.store
        V = st.V
        if payload_width_mult != 1.0:
            axis = [1.0] * st.dim
            for j in range(st.addr_dim, st.addr_dim + st.payload_dim):
                axis[j] = float(payload_width_mult)
            V = eqx.tree_at(
                lambda t: t.learned,
                V,
                eqx.tree_at(lambda a: a.axis_width_scale, V.learned, tuple(axis),
                            is_leaf=lambda x: x is None),
            )
        m = clu_with_potential(
            V, dim=st.dim, kinetic_mode=self.cfg.kinetic_mode,
            inertia=jnp.ones(st.dim),
        )
        # ⭐ C2W8: the trash region's FIRST USE. Attached only when it exists —
        # even an EMPTY field is not bit-identical to no field (the integrator
        # composes `1 - (1 - gamma)(1 - gamma_phi)`, and `1 - (1 - g)*1.0 != g`
        # in floating point), which is exactly why OFF means *no attachment*.
        if getattr(self, "trash", None) is not None:
            m = eqx.tree_at(lambda t: t.friction_field, m, self.trash,
                            is_leaf=lambda x: x is None)
        return m

    # -- the three public operations --------------------------------------
    def write_stream(self, items: Sequence[dict], key=None,
                     interleave_read_every: int = 0) -> WriteReport:
        """Stream writes (and optionally interleaved reads) through the system.

        Each item is a dict ``{"item_id", "x" or "address", "payload",
        "permanent", "leak", "delete"}``. Every write goes through the
        controller's designed verbs (admit -> place, with eviction under budget
        pressure), then a **masked** learned write into ``V_theta``. Monitors run
        after every write.
        """
        key = jax.random.PRNGKey(self.cfg.seed) if key is None else key
        rep = WriteReport()
        for item in items:
            key, k_w = jax.random.split(key)
            self._t += 1
            if item.get("delete"):
                self._delete(int(item["item_id"]), rep)
                continue
            iid = int(item["item_id"])
            payload = np.atleast_1d(np.asarray(item["payload"], dtype=float))
            address = np.asarray(
                item["address"] if "address" in item else self._embed(item["x"]),
                dtype=float,
            ).reshape(-1)[: self.store.addr_dim]
            reach = self._reach_margin_for(address, payload)
            res = self.controller.admit(
                iid, address, float(payload[0]),
                utility=float(item.get("utility", 1.0)),
                reach_margin=reach if self.cfg.stage_admission else None,
                permanent=bool(item.get("permanent", False)),
                leak=item.get("leak", self.cfg.leak if self.cfg.stage_lifetimes else 0.0),
            )
            row = dict(res.detail.get("row", {}))
            row.update({"t": self._t, "item_id": iid, "verb": "admit",
                        "applied": res.applied, "guard": res.guard,
                        "decision": row.get("decision", res.reason),
                        "reach_margin": reach,
                        "gate_margin": row.get("d_min_proposed", float("nan"))})
            if not res.applied:
                rep.refused.append(iid)
                self._write_log.append(row)
                rep.log.append(row)
                continue
            for vr in self.controller.log[-4:]:
                if vr.verb == "evict" and vr.applied and vr.t == self.controller.t:
                    ev_id = int(vr.detail.get("item_id", -1))
                    if ev_id >= 0 and ev_id not in rep.evicted:
                        rep.evicted.append(ev_id)
                        self._payloads.pop(ev_id, None)
            slot = self._slot_of(iid)
            self._payloads[iid] = payload
            self._born[iid] = self._t
            # --- the LEARNED write: masked, into V_theta ---
            self._prev_store = self.store
            pre = self._relaxed_sites()
            # ⭐ C2W8 pass-2 ARM A: site-local atom init, at the only moment a
            # phi-addressed stream knows its address. Touches ONLY this slot's
            # rows (C3-local in parameter space, like the masked write itself),
            # and is counted INSIDE the measured C3 drift because `pre` is taken
            # above it. ⛔ OFF (the default) leaves the key stream untouched too:
            # the key is FOLDED from `k_w`, never split off it.
            if self.cfg.atom_site_local_init:
                self._localize_slot_atoms(slot, address, payload,
                                          jax.random.fold_in(k_w, int(slot)))
            loss = self._write_item(slot, address, payload, k_w)
            self._losses.append(loss)
            rep.losses.append(loss)
            self._c3_ratio, self._oldest_drop = self._c3_check(pre)
            rep.admitted.append(iid)
            row["write_loss"] = loss
            row["post_write_drift"] = self._last_drift
            # ⭐ SC-2: every admitted write logs its certificate margin and its
            # relative deficit. Reporting only — the write is never refused for
            # it (SC-3: exceeding the budget TRIPS #3).
            row.update(self._cert_margin_now())
            self._write_log.append(row)
            rep.log.append(row)
            if self.cfg.stage_lifetimes:
                self.controller.decay(1)
                self._sync_decay()
            if interleave_read_every and (len(rep.admitted) % interleave_read_every == 0):
                rep.readings.extend(self.observe(stage=f"write:t{self._t}"))
        return rep

    def read(self, x, *, steps: Optional[int] = None,
             schedule: Optional[Sequence[float]] = None,
             allow_retry: Optional[bool] = None,
             collect_trajectory: bool = True) -> ReadResult:
        """Two-phase relaxation read. ⭐ **Returns the trajectory, not only ``q*``.**

        Phase 1 relaxes the launch point ``q0 = phi(x)`` under ``gamma_address``
        (address acquisition — the address of an item is *derived*, it is
        wherever the query relaxes to). Phase 2 rolls out under ``gamma_read``,
        which is where the value is read. Both phases are recorded into a
        **strided** buffer, so ``psi`` may use the trajectory, the settled point,
        or both, and point-vs-trajectory is an *internal ablation*.

        ``schedule`` is the annealed read (``anneal`` verb): a per-stage payload
        width multiplier. Its designed guard — the schedule must return to the
        stored landscape before the value is read — is enforced by the controller.
        """
        cfg = self.cfg
        q0 = jnp.asarray(self._embed(x), dtype=jnp.float32)
        if q0.ndim == 1:
            q0 = q0[None, :]
        q0 = q0.at[:, cfg.addr_dim: cfg.addr_dim + cfg.payload_dim].set(0.0)
        p0 = jnp.zeros_like(q0)
        n_addr = int(steps or cfg.address_steps)
        sched = list(schedule) if schedule is not None else list(
            getattr(self.controller, "schedule", (1.0,))
        )
        if abs(float(sched[-1]) - 1.0) > 1e-9:
            raise ValueError(
                "read schedule must return to the stored landscape (last multiplier 1.0)"
            )

        # --- phase 1: address acquisition (annealed if a schedule is set) ---
        q, p = q0, p0
        traj_parts, phase_parts = [], []
        per_stage = max(1, n_addr // max(1, len(sched)))
        for mult in sched:
            m = self.model(payload_width_mult=float(mult))
            tr = _rollout(m, q, p, per_stage, cfg.dt, cfg.gamma_address)
            tr_s = tr[:, :: cfg.traj_stride, :]
            if collect_trajectory:
                traj_parts.append(tr_s)
                phase_parts.append(np.ones((tr_s.shape[1],), dtype=int))
            q, p = tr[:, -1, : q.shape[1]], tr[:, -1, q.shape[1]:]
        q_addr, p_addr = q, p

        # --- phase 2: the read rollout (where the value is read) ---
        m0 = self.model()
        tr2 = _rollout(m0, q_addr, p_addr, int(cfg.read_steps), cfg.dt, cfg.gamma_read)
        tr2_s = tr2[:, :: cfg.traj_stride, :]
        if collect_trajectory:
            traj_parts.append(tr2_s)
            phase_parts.append(2 * np.ones((tr2_s.shape[1],), dtype=int))
        q_star, p_star = tr2[:, -1, : q0.shape[1]], tr2[:, -1, q0.shape[1]:]

        traj = jnp.concatenate(traj_parts, axis=1) if collect_trajectory else tr2_s
        phase = np.concatenate(phase_parts) if collect_trajectory else np.full(
            (tr2_s.shape[1],), 2, dtype=int
        )
        state = ReadState(q0=q0, p0=p0, q_addr=q_addr, p_addr=p_addr,
                          q_star=q_star, p_star=p_star)

        g0 = np.asarray(_grad_norms(m0, q0))
        gs = np.asarray(_grad_norms(m0, q_star))
        rho = gs / np.maximum(np.median(g0), 1e-12)
        conf = np.exp(-np.clip(rho, 0.0, 50.0))
        n_steps = per_stage * len(sched) + int(cfg.read_steps)

        retries = 0
        do_retry = cfg.stage_retry if allow_retry is None else bool(allow_retry)
        if do_retry:
            for r in range(int(self.controller.policy.retry_max_rounds)):
                vr = self.controller.retry(float(np.min(conf)), round_index=r)
                if not vr.applied:
                    break
                retries += 1
                tr3 = _rollout(m0, q_star, p_star, int(cfg.read_steps),
                               cfg.dt, cfg.gamma_read)
                tr3_s = tr3[:, :: cfg.traj_stride, :]
                traj = jnp.concatenate([traj, tr3_s], axis=1)
                phase = np.concatenate([phase, 2 * np.ones((tr3_s.shape[1],), dtype=int)])
                q_star, p_star = tr3[:, -1, : q0.shape[1]], tr3[:, -1, q0.shape[1]:]
                state = ReadState(q0=q0, p0=p0, q_addr=q_addr, p_addr=p_addr,
                                  q_star=q_star, p_star=p_star)
                gs = np.asarray(_grad_norms(m0, q_star))
                rho = gs / np.maximum(np.median(g0), 1e-12)
                conf = np.exp(-np.clip(rho, 0.0, 50.0))
                n_steps += int(cfg.read_steps)

        value = self.psi(traj, state)
        diag = self._read_diagnostics(q0, q_addr, q_star, g0, gs, rho)
        return ReadResult(value=value, traj=traj, phase=phase, state=state,
                          confidence=conf, residual=gs, rho_conv=rho,
                          n_steps=n_steps, retries=retries, diagnostics=diag)

    def place_pass(self) -> int:
        """Re-derive every live address by **relaxation** and re-place it.

        Monitor #8-N3's restoring verb: the recorded site is where the writer was
        *told* to put the item; the address the read actually lands on is where
        the dissipative dynamics take it. Committing the latter is the derived
        address, and it is committed only when ``lambda_min(H) > 0`` — never by a
        critical-point solver (a Newton re-derivation once wrote a saddle into the
        codebook).
        """
        ids, centers, pays = self.codebook()
        if len(ids) == 0:
            return 0
        q0 = np.zeros((len(ids), self.store.dim), dtype=np.float32)
        q0[:, : self.store.addr_dim] = centers
        res = self.read(q0)
        q_star = np.asarray(res.state.q_star)
        moved = 0
        for i, iid in enumerate(ids):
            site = q_star[i, : self.store.addr_dim]
            lam = self._lambda_min_at(q_star[i: i + 1])
            if lam <= 0:
                continue
            if np.linalg.norm(site - centers[i]) < 1e-6:
                continue
            if self.controller.place(int(iid), site, lambda_min=lam).applied:
                moved += 1
        return moved

    def consolidate(self, key=None, place_pass: bool = False) -> ConsolidationReport:
        """Offline maintenance: re-pack, enforce decay, re-check certificates,
        run the label-free **self-probe** pass (the store re-reads its own
        written items), and re-calibrate the gate.

        This is where wake-sleep is repositioned (charter §2.4): consolidation,
        not the trainer.
        """
        n_moves = self.place_pass() if place_pass else 0
        probe = self.self_probe(key)
        certs = self.certificates()
        reads = probe.get("read").diagnostics if probe.get("read") is not None else None
        readings = self.observe(stage="consolidate", self_probe=probe,
                                certificates=certs, reads=reads)
        return ConsolidationReport(certificates=certs, readings=readings,
                                   self_probe=probe, n_moves=n_moves)

    # -- monitors ----------------------------------------------------------
    def observe(self, stage: str, self_probe: Optional[dict] = None,
                certificates: Optional[dict] = None,
                blank: Optional[dict] = None, reads: Optional[dict] = None,
                extras: Optional[dict] = None) -> List[Any]:
        """Run the monitor registry against the system's current state."""
        ids, centers, pays = self.codebook()
        sep = _min_separation(centers)
        s_max = self._s_max()
        ex = {
            "sep": sep,
            "kinetic_mode": self.cfg.kinetic_mode,
            "write_steps": self.cfg.write_steps,
            "wall_clock_s": time.time() - self._wall0,
            "utilisation": self._utilisation(),
            "fairness": self._fairness(),
            "c3_ratio": self._c3_ratio,
            # ⭐ C2W3 D3: the per-pair record monitor #3's replacement leg reads.
            # Present ALWAYS (the leg replacement is a repair, not a flag); the
            # soft-certificate block below is what is default-off.
            "c3_pairs": list(self._c3_pairs),
            "oldest_retention_drop": self._oldest_drop,
            "min_sep_minus_2s": (sep - 2.0 * s_max) if np.isfinite(sep) else float("nan"),
            # ⚠ Monitor #10 tier (a) — the O(1) access-counting config proxy — is
            # NOT implemented in v0; only tier (b) (the semantic sweep) runs. The
            # declared-knob list is therefore the swept set, and tier (a) is
            # reported as a gap rather than silently passing.
            "knob_reads": self._knob_reads or None,
            "knobs_declared": sorted(self._knob_reads) if self._knob_reads else None,
            "knob_tier_a_implemented": False,
        }
        ex.update(self._reach_extras())
        # ⭐ C2W3 rider B: SC-1…SC-6, default-OFF. When off, nothing is added and
        # the monitors see exactly the shipped context.
        if self.cfg.soft_certificate:
            ex["soft_certificate"] = self.soft_certificate_state()
        if certificates is not None:
            ex["certificates"] = certificates
        if extras:
            ex.update(extras)
        if ex.get("knob_sweep") and not ex.get("knobs_declared"):
            # tier (b) is live: every swept dial is "declared active this run"
            ex["knobs_declared"] = sorted(ex["knob_sweep"])
            ex["knob_reads"] = {k: 1 for k in ex["knob_sweep"]}
        ctx = MonitorContext(stage=stage, t=self._t, system=self, reads=reads,
                             self_probe=self_probe, blank=blank,
                             write_log=self._write_log, controller=self.controller,
                             extras=ex)
        return self.registry.observe(ctx)

    # -- hooks -------------------------------------------------------------
    def fixed_point_residual(self, q, p, *, gamma: Optional[float] = None):
        """``(q, p) - Step_gamma(q, p)`` — the settle's fixed-point residual.

        The attachment point for an implicit/DEQ gradient
        (`trainability-spike`): a settle is a fixed point, so one differentiates
        through the equilibrium rather than the unroll.
        """
        m = self.model()
        g = self.cfg.gamma_address if gamma is None else float(gamma)
        q = jnp.asarray(q)
        p = jnp.asarray(p)
        one = m(q, p, 1, self.cfg.dt, g)  # (1, 2*dim)
        d = q.shape[-1]
        return jnp.concatenate([q - one[-1, :d], p - one[-1, d:]], axis=-1)

    def self_probe(self, key=None) -> Dict[str, Any]:
        """Label-free diagnostic pass: the store re-reads what it wrote.

        Feeds monitors #5 (acquisition), #6 (objective divergence), #9
        (lifetimes) and #12 (starvation) without ever consulting a task label.
        """
        ids, centers, pays = self.codebook()
        if len(ids) == 0:
            return {"acq": float("nan"), "n_probed": 0}
        key = jax.random.PRNGKey(self.cfg.seed + 7919) if key is None else key
        n_per = int(self.cfg.n_query_per_item)
        labels = np.repeat(np.arange(len(ids)), n_per)
        jitter = np.asarray(
            jax.random.normal(key, (len(labels), self.store.addr_dim))
        ) * float(self.cfg.query_sigma)
        q0 = np.zeros((len(labels), self.store.dim), dtype=np.float32)
        q0[:, : self.store.addr_dim] = centers[labels] + jitter
        res = self.read(q0)
        val = np.asarray(res.value).reshape(len(labels), -1)
        addr = np.asarray(res.state.q_addr)[:, : self.store.addr_dim]
        basin = _assign(addr, centers)
        ok = basin == labels
        err = np.linalg.norm(val - pays[labels], axis=-1)
        strict = ok & (err < float(self.cfg.payload_tol))
        # decode: which stored payload is the read value closest to
        dec = np.argmin(np.linalg.norm(val[:, None, :] - pays[None, :, :], axis=-1), axis=1)
        retention = np.array([float(np.mean(strict[labels == i])) for i in range(len(ids))])
        return {
            "acq": float(np.mean(ok)),
            "strict": float(np.mean(strict)),
            "decode": float(np.mean(dec == labels)),
            "chance": 1.0 / max(len(ids), 1),
            "n_probed": int(len(labels)),
            "retention": retention,
            "payload_abs": np.abs(pays).max(axis=1),
            "write_loss": (self._losses[-1] if self._losses else float("nan")),
            "delta_read_basin_conditioned": (
                float(np.median(err[ok])) if np.any(ok) else float("nan")
            ),
            "values": val,
            "labels": labels,
            "assign_settle": basin,
            "q0": q0,
            "rho_conv": np.asarray(res.rho_conv),
            "read": res,
        }

    def certificates(self) -> Dict[str, Any]:
        """The C1-C5 / N1-N4 certificate pass (monitor #8's input)."""
        ids, centers, pays = self.codebook()
        out: Dict[str, Any] = {}
        if len(ids) == 0:
            return out
        sep = _min_separation(centers)
        out["injective"] = bool(sep > 1e-6)
        out["sep"] = sep
        out["sep_over_sigma_q"] = sep / max(float(self.cfg.query_sigma), 1e-12)
        out["erf_accuracy"] = erf_margin_accuracy(sep / 2.0, float(self.cfg.query_sigma))
        out["lambda_min"] = self._lambda_min(centers, pays)
        if len(ids) > 1:
            d = np.linalg.norm(pays[:, None, :] - pays[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            out["payload_gap"] = float(np.min(d))
        else:
            out["payload_gap"] = float("inf")
        probe = self.self_probe()
        out["delta_read_basin_conditioned"] = probe.get(
            "delta_read_basin_conditioned", float("nan")
        )
        return out

    def codebook(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(ids, addresses, payloads)`` over live items — **eval only**.

        The read path never calls this. It exists so the same-keys launder can be
        constructed (doctrine I-1) and so monitors have ground truth about what
        the store believes it holds.
        """
        recs = sorted(self.controller.allocator.records.values(), key=lambda r: r.item_id)
        ids = np.array([r.item_id for r in recs], dtype=int)
        centers = (np.stack([np.asarray(r.center, dtype=float) for r in recs])
                   if recs else np.zeros((0, self.store.addr_dim)))
        pays = (np.stack([self._payloads[r.item_id] for r in recs])
                if recs else np.zeros((0, self.store.payload_dim)))
        return ids, centers, pays

    def n_bytes(self) -> int:
        """Matched-bytes accounting for the full system.

        ⚠ **The trash region's holes are bytes and are counted here** (§3.5): a
        `gamma_phi` field off the ledger is a hidden capacity increase.
        """
        ids, centers, _ = self.codebook()
        return int(self.store.n_bytes() + centers.size * 4 + self.trash_bytes())

    def trash_bytes(self) -> int:
        """Bytes of the trash region: ``K x (dim centers + radius + strength)``."""
        if getattr(self, "trash", None) is None:
            return 0
        return int(self.trash.k * (self.store.dim + 2) * 4)

    def trash_route(self, center, *, radius: Optional[float] = None,
                    strength: Optional[float] = None) -> int:
        """Place one trash hole at ``center`` — the routing verb, not a fit.

        ⛔ Requires ``cfg.gamma_phi``: with the flag off there is no field, and
        silently building one would make OFF non-identical to the pre-build path.
        Returns the new hole count ``K``.
        """
        from chlu.core.friction_field import add_hole

        if self.trash is None:
            raise RuntimeError(
                "trash_route requires cfg.gamma_phi=True; with the flag off the "
                "read path carries no field at all (that is what makes OFF "
                "bit-identical to the pre-C2W8 path)"
            )
        z = np.zeros((self.store.dim,), dtype=float)
        c = np.asarray(center, dtype=float).reshape(-1)
        z[: min(c.size, self.store.dim)] = c[: self.store.dim]
        self.trash = _as_f32(add_hole(
            self.trash, jnp.asarray(z, dtype=self.trash.centers.dtype),
            float(self.cfg.gamma_phi_radius if radius is None else radius),
            float(self.cfg.gamma_phi_strength if strength is None else strength),
        ))
        return int(self.trash.k)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _embed(self, x):
        x = jnp.asarray(x, dtype=jnp.float32)
        if self.phi is None:
            if x.shape[-1] == self.store.dim:
                return x
            pad = jnp.zeros(x.shape[:-1] + (self.store.dim - x.shape[-1],))
            return jnp.concatenate([x, pad], axis=-1)
        return self.phi(x)

    def _slot_of(self, item_id: int) -> int:
        for r in self.controller.allocator.records.values():
            if r.item_id == int(item_id):
                return int(r.slot)
        raise KeyError(item_id)

    def _localize_slot_atoms(self, slot: int, address, payload, key) -> None:
        """ARM A wiring: re-draw ``slot``'s atom centers around the item's site.

        One call into :func:`~chlu.core.memory_potentials.localize_group_atoms`,
        which carries the mechanism and the compliance note. The payload channels
        are included in the target point, so an atom can reach the payload well
        it must dig; the ball radius defaults to the atom width.
        """
        from chlu.core.memory_potentials import localize_group_atoms

        z = np.zeros((self.store.dim,), dtype=float)
        z[: self.store.addr_dim] = np.asarray(address, dtype=float)[: self.store.addr_dim]
        z[self.store.addr_dim: self.store.addr_dim + self.store.payload_dim] = payload
        r = float(self.cfg.atom_site_local_radius or self.cfg.atom_width)
        learned = localize_group_atoms(
            self.store.V.learned, np.asarray(self.store.group_rows(int(slot))),
            z, r, key)
        V = eqx.tree_at(lambda t: t.learned, self.store.V, learned)
        self.store = eqx.tree_at(lambda s: s.V, self.store, V)

    def _write_item(self, slot: int, address, payload, key) -> float:
        cfg = self.cfg
        z = np.zeros((1, self.store.dim), dtype=np.float32)
        z[0, : self.store.addr_dim] = address[: self.store.addr_dim]
        z[0, self.store.addr_dim: self.store.addr_dim + self.store.payload_dim] = payload
        ids, centers, pays = self.codebook()
        crowd = np.zeros((len(ids), self.store.dim), dtype=np.float32)
        crowd[:, : self.store.addr_dim] = centers
        crowd[:, self.store.addr_dim: self.store.addr_dim + self.store.payload_dim] = pays
        mask_fn = (atom_write_mask_fn(self.store.group_rows(slot))
                   if cfg.masked_write else None)
        # -- C2W3 RIDER A: the store family's OWN write mask ------------------
        # Additive: absent factory => `mask_fn` is the shipped row mask, untouched.
        if cfg.store_write_mask_factory:
            mask_fn = resolve_store_write_mask_factory(cfg.store_write_mask_factory)(
                cfg=cfg, store=self.store, slot=int(slot), default_mask_fn=mask_fn,
                **dict(cfg.store_write_mask_kwargs or {}),
            )
        loss_kwargs = dict(
            n_perturb=int(cfg.write_n_perturb),
            sigma_addr=float(cfg.write_sigma_addr),
            sigma_pay=float(cfg.write_sigma_pay),
            margin=float(cfg.write_margin),
            barrier=float(cfg.write_barrier),
            payload_index=int(self.store.addr_dim),
            barrier_pairs="nn",
            crowd_targets=jnp.asarray(crowd) if len(ids) > 1 else None,
        )
        train_kwargs = dict(
            steps=int(cfg.write_steps), lr=float(cfg.write_lr),
            weight_decay=float(cfg.write_weight_decay),
        )
        # -- C2W2 SEAM (a): the write-objective passthrough -------------------
        # Additive only: an empty/absent spec leaves both dicts untouched, so the
        # shipped write is bit-identical (asserted in tests/test_clu_system.py).
        loss_kwargs.update(self.write_objective.get("loss_kwargs", {}))
        train_kwargs.update(self.write_objective.get("train_kwargs", {}))
        V, hist = train_memory_landscape(
            self.store.V, jnp.asarray(z), key,
            loss_kwargs=loss_kwargs,
            update_mask_fn=mask_fn,
            **train_kwargs,
        )
        self.store = eqx.tree_at(lambda s: s.V, self.store, V)
        # ⭐ C2W2: the ENDPOINT-ONLY write loss, re-evaluated on the written V with
        # the Route-1 coefficients OFF. Without it a Route-1 arm looks unconverged
        # merely for carrying its own extra term (its recorded total includes
        # `lambda * L_term`, which is not required to reach zero), and the gate's
        # convergence ruling would exclude exactly the arms under test. One extra
        # loss evaluation per write.
        try:
            from chlu.training.train_memory import write_loss as _wl

            base_kwargs = {k: v for k, v in loss_kwargs.items()
                           if k not in ("lambda_traj", "lambda_path",
                                        "traj_kwargs", "path_kwargs")}
            self._endpoint_losses.append(
                float(_wl(V, jnp.asarray(z), key, **base_kwargs)))
        except Exception:  # never let a diagnostic break a write
            self._endpoint_losses.append(float("nan"))
        return float(hist[-1]) if hist else float("nan")

    def _relaxed_sites(self) -> Optional[np.ndarray]:
        ids, centers, _ = self.codebook()
        if len(ids) == 0:
            return None
        q0 = np.zeros((len(ids), self.store.dim), dtype=np.float32)
        q0[:, : self.store.addr_dim] = centers
        res = self.read(q0)
        return np.asarray(res.state.q_star)

    def _c3_check(self, pre: Optional[np.ndarray]):
        """Measured vs C3-predicted drift of previously-stored fixed points."""
        self._last_drift = float("nan")
        if pre is None or self._prev_store is None or pre.shape[0] < 2:
            return float("nan"), float("nan")
        post = self._relaxed_sites()
        if post is None or post.shape[0] != pre.shape[0]:
            return float("nan"), float("nan")
        measured = np.linalg.norm(post[:-1] - pre[:-1], axis=-1)  # exclude the new item
        self._last_drift = float(np.max(measured)) if measured.size else float("nan")
        m_old = self.model(store=self._prev_store)
        m_new = self.model()
        g_old = np.asarray(_grad_vecs(m_old, jnp.asarray(pre[:-1])))
        g_new = np.asarray(_grad_vecs(m_new, jnp.asarray(pre[:-1])))
        dV = np.linalg.norm(g_new - g_old, axis=-1)
        lam_i = self._lambda_min_per_point(pre[:-1])
        lam = max(float(np.min(lam_i)) if lam_i.size else float("nan"), 1e-9)
        predicted = dV / lam
        ratio = float(np.median(measured / np.maximum(predicted, 1e-12)))
        # ⭐ C2W3 rider B / D3: the PER-PAIR record monitor #3's replacement leg
        # needs (`soft_certificate.c3_calibration`). Zero extra cost — every term
        # is already computed above — and ⛔ **no `max(lambda, 1e-9)` clamp**
        # (fix S3): the clamp sends `B -> inf` and `rho_C3 -> 0` at any
        # non-minimum, i.e. it reports a perfect certificate precisely where
        # there is none. A non-minimum pair is DISQUALIFIED, not rescued.
        self._c3_pairs = [
            {"B": float(dv / li) if li > 0 else float("nan"),
             "delta": float(md), "lambda_min": float(li), "grad_dV": float(dv)}
            for dv, md, li in zip(dV, measured, lam_i, strict=True)
        ]
        # oldest-item retention proxy: the drift of the OLDEST stored fixed point
        drop = float(measured[0] / max(_min_separation(self.codebook()[1]), 1e-9)) \
            if measured.size else float("nan")
        return ratio, drop

    def _cert_margin_now(self) -> Dict[str, float]:
        """⭐ **SC-2** at the current state: ``cert_margin`` / ``deficit_rel``.

        ``R_cert`` is computed **always** (it is a reported quantity under SC-1,
        not a gate), so the two radii can be compared even in shipped runs; the
        deficit is accumulated for SC-3's budget.
        """
        from chlu.core.soft_certificate import cert_margin, cert_radius

        _, centers, _ = self.codebook()
        sep = _min_separation(centers)
        if not np.isfinite(sep):
            return {}
        r = cert_radius(self._s_max(), float(self.cfg.query_sigma),
                        float(self.cfg.d_safe_kappa_prime))
        m = cert_margin(sep, r)
        self._cert_deficits.append(float(m["deficit_rel"]))
        return {"cert_margin": m["cert_margin"], "deficit_rel": m["deficit_rel"],
                "R_cert": m["R_cert"]}

    def soft_certificate_state(self, measure_capture: Optional[bool] = None
                               ) -> Dict[str, Any]:
        """⭐ **SC-1…SC-6** evaluated on the current store (the artifact's record).

        ``measure_capture`` defaults to "yes if the soft certificate is on and
        ``capture_dirs > 0``": SC-6's basin measurement is the one part of the
        spec that costs anything (32-direction bisection at **one** site), and a
        run that does not measure it reports SC-6 as **INAPPLICABLE, never
        passed**.
        """
        from chlu.core.soft_certificate import (
            expected_separation,
            soft_certificate_report,
        )

        sc = self.cfg.soft_cert_config()
        ids, centers, pays = self.codebook()
        sep = _min_separation(centers)
        sep_exp = (float(sc.sep_expected) if sc.sep_expected is not None
                   else expected_separation(designed_sites(
                       int(self.cfg.addr_dim), int(self.cfg.capacity),
                       R=float(self.cfg.ball_radius), seed=int(self.cfg.seed))))
        z = np.zeros((len(ids), self.store.dim), dtype=np.float32)
        if len(ids):
            z[:, : self.store.addr_dim] = centers
            z[:, self.store.addr_dim: self.store.addr_dim + self.store.payload_dim] = pays
        lam_i = self._lambda_min_per_point(z) if len(ids) else np.zeros((0,))
        capture = None
        want = (bool(sc.enabled and sc.capture_dirs > 0) if measure_capture is None
                else bool(measure_capture))
        if want and len(ids):
            from chlu.core.soft_certificate import capture_radius

            worst = int(np.argmin(lam_i))
            capture = capture_radius(
                self._relax_points, z[worst], n_dirs=int(sc.capture_dirs),
                r_hi=float(sep if np.isfinite(sep) else 1.0),
                steps=int(sc.capture_bisect_steps),
                tol=float(self.cfg.query_sigma), seed=int(self.cfg.seed))
        return soft_certificate_report(
            sc, sep_expected=sep_exp, sep_after=sep, s_max=self._s_max(),
            sigma_q=float(self.cfg.query_sigma),
            kappa_prime=float(self.cfg.d_safe_kappa_prime),
            deficits=(self._cert_deficits or None), c3_pairs=self._c3_pairs,
            lambda_mins=lam_i, capture=capture)

    def _relax_points(self, pts) -> np.ndarray:
        """Relax ``(n, dim)`` states under the CURRENT landscape (SC-6's basin test)."""
        q = jnp.asarray(np.atleast_2d(np.asarray(pts, dtype=np.float32)))
        p = jnp.zeros_like(q)
        tr = _rollout(self.model(), q, p, int(self.cfg.read_steps), self.cfg.dt,
                      self.cfg.gamma_read)
        return np.asarray(tr[:, -1, : q.shape[1]])

    def _lambda_min_per_point(self, points) -> np.ndarray:
        """``lambda_min`` **per site** — the quantity SC-4(i)/SC-6 and the C3
        calibration leg need. The shipped scalar is its minimum, so this is a
        refactor at zero cost, not a new computation."""
        m = self.model()
        V = m.potential_net
        return np.asarray(
            [float(np.linalg.eigvalsh(
                np.asarray(jax.hessian(lambda q: V(q))(jnp.asarray(z)))).min())
             for z in np.asarray(points)], dtype=float)

    def _lambda_min_at(self, points) -> float:
        lam = self._lambda_min_per_point(points)
        return float(np.min(lam)) if lam.size else float("nan")

    def _lambda_min(self, centers, pays) -> float:
        if len(centers) == 0:
            return float("nan")
        z = np.zeros((len(centers), self.store.dim), dtype=np.float32)
        z[:, : self.store.addr_dim] = centers
        z[:, self.store.addr_dim: self.store.addr_dim + self.store.payload_dim] = pays
        return self._lambda_min_at(z)

    def _well_fit(self, z: np.ndarray, n_dirs: int = 8, seed: int = 0
                  ) -> Tuple[float, float]:
        """Fit ``(D_i, s_i)`` of the well at ``z`` **on the real learned V**.

        The saddle reach criterion (monitor #11) needs the well's depth and width,
        and reading them off the atom parameters underestimates both whenever an
        item's atoms are displaced to reach a large payload excursion (measured:
        an ``|a| = 1`` item's own-atom sum at its site is 0.017 while its actual
        well is 0.46 deep). So fit the isotropic profile

            V(z + r u) - alpha*(|z+ru|^2 - |z|^2) - V(z)  ~  D (1 - e^{-r^2/2s^2})

        over ``n_dirs`` random directions and a radius ladder — the doctrine's
        "64 dirs x 40 radii" cost row, at a cheaper but honest resolution.
        """
        V = self.store.V
        z = np.asarray(z, dtype=np.float32)
        rng = np.random.default_rng(int(seed))
        u = rng.normal(size=(int(n_dirs), self.store.dim))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        radii = np.linspace(0.15, 1.5, 12)
        pts = (z[None, None, :] + radii[None, :, None] * u[:, None, :]).reshape(-1, self.store.dim)
        vals = np.asarray(jax.vmap(V)(jnp.asarray(pts, dtype=jnp.float32)))
        v0 = float(V(jnp.asarray(z)))
        conf = float(self.cfg.confine) * (np.sum(pts**2, axis=1) - float(np.sum(z**2)))
        y = (vals - conf - v0).reshape(int(n_dirs), radii.size).mean(axis=0)
        best = (float("nan"), float("nan"), np.inf)
        for s_try in np.linspace(0.05, 1.2, 120):
            basis = 1.0 - np.exp(-(radii**2) / (2.0 * s_try**2))
            denom = float(np.sum(basis * basis))
            if denom <= 0:
                continue
            D = float(np.sum(basis * y) / denom)
            resid = float(np.sum((y - D * basis) ** 2))
            if resid < best[2]:
                best = (D, float(s_try), resid)
        D, s_fit, _ = best
        return max(D, 0.0), s_fit

    def well_fits(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(D, s)`` per live item, fitted on the learned landscape."""
        ids, centers, pays = self.codebook()
        Ds, ss = [], []
        for c, a in zip(centers, pays, strict=True):
            z = np.zeros((self.store.dim,), dtype=np.float32)
            z[: self.store.addr_dim] = c
            z[self.store.addr_dim: self.store.addr_dim + self.store.payload_dim] = a
            D, sv = self._well_fit(z, seed=self.cfg.seed)
            Ds.append(D)
            ss.append(sv)
        return np.asarray(Ds), np.asarray(ss)

    def _s_max(self) -> float:
        return float(np.max(np.exp(np.asarray(self.store.atoms.log_width, dtype=float))))

    def _utilisation(self) -> float:
        cfg = self.cfg
        d_safe = derived_d_safe(self._s_max(), cfg.query_sigma, cfg.d_safe_kappa_prime)
        n_pack = max((2.0 * cfg.ball_radius / d_safe + 1.0) ** cfg.addr_dim, 1.0)
        return float(self.controller.allocator.n_live / n_pack)

    def _fairness(self) -> float:
        ids, centers, _ = self.codebook()
        if len(ids) < 2:
            return float("nan")
        depths, _ = self.well_fits()
        return float(np.min(depths) / max(np.max(depths), 1e-12))

    def _reach_extras(self) -> dict:
        ids, centers, pays = self.codebook()
        if len(ids) == 0:
            return {}
        margins, a_us, bad = [], [], []
        Ds, ss = self.well_fits()
        for iid, c, a, D, s in zip(ids, centers, pays, Ds, ss, strict=True):
            a_u = saddle_reach_threshold(D, s, float(self.cfg.confine),
                                         float(np.linalg.norm(c)))
            a_us.append(a_u)
            margin = float(a_u - np.max(np.abs(a)))
            margins.append(margin)
            if margin <= 0:
                bad.append(int(iid))
        return {"reach_margins": margins, "a_U": a_us, "unreachable_ids": bad}

    def _reach_margin_for(self, address, payload) -> float:
        """Write-time reach margin, using the store's typical (D, s)."""
        ids, centers, _ = self.codebook()
        if len(ids) == 0:
            # nothing written yet: the store's own init width and a unit well are
            # the only honest prior, and the margin is re-checked at consolidation.
            D, s = 1.0, float(self.cfg.atom_width)
        else:
            Ds, ss = self.well_fits()
            D, s = float(np.median(Ds)), float(np.median(ss))
        a_u = saddle_reach_threshold(max(D, 1e-6), s, float(self.cfg.confine),
                                     float(np.linalg.norm(address)))
        return float(a_u - np.max(np.abs(payload)))

    def _sync_decay(self) -> None:
        """Apply the controller's decay factors to the LEARNED landscape."""
        last = [r for r in self.controller.log if r.verb == "decay"]
        if not last:
            return
        det = last[-1].detail
        for iid, factor in det.get("factors", {}).items():
            try:
                slot = self._slot_of(int(iid))
            except KeyError:
                continue
            if abs(float(factor) - 1.0) > 1e-12:
                self.store = self.store.scale_group_amplitude(slot, float(factor))

    def _delete(self, item_id: int, rep: WriteReport) -> None:
        try:
            self._slot_of(item_id)
        except KeyError:
            return
        res = self.controller.evict(
            item_id, reason="delete", trips=self.controller.policy.evict_persistence_W
        )
        if res.applied:
            pass  # store_apply("evict") already re-drew the freed group
            self._payloads.pop(item_id, None)
            rep.deleted.append(item_id)
        row = {"t": self._t, "item_id": item_id, "verb": "evict",
               "decision": "delete" if res.applied else "delete_refused",
               "applied": res.applied, "guard": res.guard}
        self._write_log.append(row)
        rep.log.append(row)

    def _read_diagnostics(self, q0, q_addr, q_star, g0, gs, rho) -> dict:
        ids, centers, pays = self.codebook()
        q0n = np.asarray(q0)[:, : self.store.addr_dim]
        addr = np.asarray(q_addr)[:, : self.store.addr_dim]
        out = {
            "grad_norm_q0": g0,
            "grad_norm_qstar": gs,
            "rho_conv": rho,
            "displacement": np.linalg.norm(np.asarray(q_star) - np.asarray(q0), axis=-1),
            "corr_q0_qstar": _corr(np.asarray(q0), np.asarray(q_star)),
        }
        if len(ids) > 0:
            a_settle = _assign(addr, centers)
            a_argmin = _assign(q0n, centers)
            r_i = 0.5 * _min_separation(centers)
            cov = np.min(np.linalg.norm(q0n[:, None, :] - centers[None, :, :], axis=-1),
                         axis=1) <= r_i
            out.update({"assign_settle": a_settle, "assign_argmin": a_argmin,
                        "covered": cov})
        return out

    # -- the store-side sink the controller's verbs reach ------------------
    def store_apply(self, verb: str, payload: dict) -> None:
        if verb == "evict":
            self._evict_key, k = jax.random.split(self._evict_key)
            self.store = self.store.reinit_group(int(payload["slot"]), k, self.cfg)
        elif verb == "expand":
            self.cfg.ball_radius = float(self.cfg.ball_radius) * float(payload["factor"])


# --------------------------------------------------------------------------
# C2W2 seams (frozen public surface — the other C2W2 branches build on these)
# --------------------------------------------------------------------------
#: The only two top-level keys a write-objective spec may carry.
WRITE_OBJECTIVE_KEYS = ("loss_kwargs", "train_kwargs")


def normalize_write_objective(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """⭐ **C2W2 SEAM (a).** Validate/normalise a write-objective spec.

    A spec is a mapping with at most two keys:

    ``loss_kwargs``
        extra keyword arguments forwarded to
        :func:`~chlu.training.train_memory.write_loss` — this is where the C2W2
        Route-1 coefficients (``lambda_traj``, ``lambda_path``, both defaulting
        to ``0.0``) are handed in **without touching**
        :class:`CluSystemConfig` **semantics**.
    ``train_kwargs``
        optimizer-level overrides for
        :func:`~chlu.training.train_memory.train_memory_landscape` (``steps``,
        ``lr``, ``weight_decay``) — the D5 longer-write arm rides here.

    ``None`` or ``{}`` returns ``{}`` and the shipped write runs bit-identical.
    An unknown top-level key raises rather than being silently dropped: a
    mis-spelled coefficient that is quietly ignored would report as *"the term
    is inert"*, which is exactly the finding the gate must not fabricate.
    """
    if not spec:
        return {}
    spec = dict(spec)
    unknown = sorted(set(spec) - set(WRITE_OBJECTIVE_KEYS))
    if unknown:
        raise ValueError(
            f"unknown write-objective key(s) {unknown}; a spec carries only "
            f"{list(WRITE_OBJECTIVE_KEYS)} (a silently-dropped coefficient would "
            "report as an inert term)"
        )
    return {k: dict(spec[k]) for k in WRITE_OBJECTIVE_KEYS if spec.get(k)}


def resolve_store_potential_factory(path: str) -> Callable:
    """⭐ **C2W2 SEAM (b).** Resolve ``"pkg.module:attr"`` / ``"pkg.module.attr"``.

    Lets a store family living in a module this file has never heard of be wired
    into :class:`LearnedVStore` from config alone (``ssb-shell-atoms``'s shell
    atoms are registered exactly this way). The resolved object is called as
    ``factory(cfg=cfg, key=key, **cfg.store_potential_kwargs)``.
    """
    import importlib

    p = str(path)
    if ":" in p:
        mod_name, attr = p.split(":", 1)
    elif "." in p:
        mod_name, attr = p.rsplit(".", 1)
    else:
        raise ValueError(
            f"store_potential_factory {path!r} must be 'pkg.module:attr' "
            "or 'pkg.module.attr'"
        )
    obj = getattr(importlib.import_module(mod_name), attr)
    if not callable(obj):
        raise TypeError(f"store_potential_factory {path!r} resolved to a non-callable")
    return obj


def resolve_store_write_mask_factory(path: str) -> Callable:
    """⭐ **C2W3 RIDER A.** Resolve ``"pkg.module:attr"`` / ``"pkg.module.attr"``.

    The write-mask twin of :func:`resolve_store_potential_factory`: a store family
    registered through that seam can now also register **its own update mask**,
    from config alone, without editing a line of this file. The resolved object is
    called as ``factory(cfg=cfg, store=store, slot=slot,
    default_mask_fn=default, **cfg.store_write_mask_kwargs)`` and must return an
    ``updates -> updates`` callable (or ``None`` for "no masking at all").

    ⛔ Without it a family whose learned leaves are not exactly
    ``learned.{centers,log_width,amp}`` writes those leaves **unmasked**, and C3
    locality — *"writing item j leaves every other item bit-identical"* — is lost
    silently. That is why this is a blocker for any future store family
    (``route3-stage2``'s slotted store included).
    """
    import importlib

    p = str(path)
    if ":" in p:
        mod_name, attr = p.split(":", 1)
    elif "." in p:
        mod_name, attr = p.rsplit(".", 1)
    else:
        raise ValueError(
            f"store_write_mask_factory {path!r} must be 'pkg.module:attr' "
            "or 'pkg.module.attr'"
        )
    obj = getattr(importlib.import_module(mod_name), attr)
    if not callable(obj):
        raise TypeError(
            f"store_write_mask_factory {path!r} resolved to a non-callable")
    return obj


# --------------------------------------------------------------------------
# construction helpers
# --------------------------------------------------------------------------
def make_controller(cfg: CluSystemConfig, registry=None,
                    policy: Optional[ControllerPolicy] = None) -> CluControllerV0:
    """Controller v0 over a C1 MVC-0 allocator sized for ``cfg``.

    The allocator's designed ``AtomStorePotential`` is used **only** as the
    address codebook (admission geometry + records); the energy landscape the
    read runs on is the learned ``V_theta``.
    """
    from chlu.core.memory_potentials import AtomStorePotential

    shadow = AtomStorePotential(
        dim=cfg.addr_dim + 1, capacity=int(cfg.capacity), s=float(cfg.atom_width),
        addr_dim=int(cfg.addr_dim),
    )
    d_safe = (float(cfg.d_safe_override) if cfg.d_safe_override is not None
              else derived_d_safe(cfg.atom_width, cfg.query_sigma,
                                  cfg.d_safe_kappa_prime))
    # -- C2W3 RIDER B / ⭐ SC-1: break the identification ---------------------
    # `d_safe := 2 s_max + kappa' sigma_q` makes the admission radius and the
    # certificate radius ONE OBJECT. With the soft certificate on, the admission
    # radius is declared independently as `zeta * sep_expected` (the harness's own
    # S4 convention) and `R_cert` becomes a reported quantity, not the gate.
    # ⛔ This is also what retires `d_safe_override` — the override WAS the soft
    # certificate, undeclared — so an explicit override still wins and is
    # reported as the legacy path.
    if cfg.soft_certificate and cfg.d_safe_override is None:
        from chlu.core.soft_certificate import expected_separation, soft_d_safe

        sc = cfg.soft_cert_config()
        sep_exp = (float(sc.sep_expected) if sc.sep_expected is not None
                   else expected_separation(designed_sites(
                       int(cfg.addr_dim), int(cfg.capacity), R=float(cfg.ball_radius),
                       seed=int(cfg.seed))))
        d_safe = soft_d_safe(sep_exp, sc.zeta)
    alloc = Controller(
        shadow, d_safe=d_safe, budget=int(cfg.capacity), amp=1.0,
        leak=float(cfg.leak), amp_floor=float(cfg.amp_floor),
        evict_policy="depth",  # set-function, never LRU
        allow_relocation=False,  # the address IS the content (q = phi(x))
    )
    pol = policy or ControllerPolicy(
        decay_leak=float(cfg.leak),
        retry_confidence_tau=float(cfg.retry_tau),
        retry_max_rounds=int(cfg.retry_max_rounds),
        anneal_payload_mult=float(cfg.anneal_payload_mult),
        anneal_stages=int(cfg.anneal_stages),
    )
    budget = int(cfg.budget if cfg.budget is not None else cfg.capacity)
    return CluControllerV0(alloc, policy=pol, registry=registry, budget=budget)


def build_system(cfg: CluSystemConfig, key=None, phi=None, psi=None,
                 loud: bool = True,
                 write_objective: Optional[Dict[str, Any]] = None) -> CluSystem:
    """Assemble a full CLU (store + controller + monitors) from a config.

    ``write_objective`` is the C2W2 seam-(a) passthrough (see
    :func:`normalize_write_objective`); ``None`` => the shipped write.
    """
    key = jax.random.PRNGKey(cfg.seed) if key is None else key
    store = LearnedVStore(cfg, key)
    registry = default_registry(loud=loud)
    controller = make_controller(cfg, registry)
    sys_ = CluSystem(store, phi=phi, psi=psi, controller=controller,
                     registry=registry, config=cfg,
                     write_objective=write_objective)
    controller.store_apply = sys_.store_apply
    return sys_


# --------------------------------------------------------------------------
# psi: the v0 handcrafted read-out (a learned psi is `trainability-spike`'s)
# --------------------------------------------------------------------------
def settled_point_psi(addr_dim: int, payload_dim: int = 1) -> Callable:
    """The v0 default read-out: the payload channels of the **settled point**.

    Deliberately the *weakest* read in the space, because it is the one 26 waves
    used: point-vs-trajectory then becomes an ablation against a known baseline
    rather than a comparison against a new one.
    """

    def psi(traj: Trajectory, state: ReadState) -> jnp.ndarray:
        return state.q_star[..., addr_dim: addr_dim + payload_dim]

    psi.representation = "settled_point"
    return psi


def tail_mean_psi(addr_dim: int, payload_dim: int = 1,
                  tail_frac: float = 0.5) -> Callable:
    """A trajectory read-out: mean payload channel over the phase-2 tail.

    Still handcrafted (no parameters), but it *uses the trajectory*, so the
    trajectory buffer has a consumer in v0 and monitor #4's trajectory-launder
    requirement is exercised before a learned psi lands.
    """

    def psi(traj: Trajectory, state: ReadState) -> jnp.ndarray:
        n = traj.shape[1]
        i0 = int(max(0, n - max(1, int(tail_frac * n))))
        seg = traj[:, i0:, addr_dim: addr_dim + payload_dim]
        return jnp.mean(seg, axis=1)

    psi.representation = "trajectory_tail"
    return psi


def store_relative_trajectory(traj: Trajectory, state: ReadState,
                              mask_q0: bool = False) -> Trajectory:
    """``traj - q0`` (and optionally drop the ``q0`` point) — doctrine I-2.

    The trajectory **contains ``q0 = phi(x)``**, so a psi over the raw buffer has
    direct access to the query embedding and a blank-store psi read is exactly "a
    classifier on ``phi(x)``" (N68's 1e-4 leak, at 100% strength). Any trajectory
    read must be reported alongside its store-relative and ``q0``-only laundered
    forms.
    """
    d = state.q0.shape[-1]
    ref = jnp.concatenate([state.q0, jnp.zeros_like(state.p0)], axis=-1)
    out = traj - ref[:, None, :]
    if mask_q0:
        out = out[:, 1:, :]
    _ = d
    return out


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def _as_f32(tree):
    """Cast every inexact leaf to float32 — the dtype the read path runs in.

    The trash field is the only object attached to the read model from outside
    the store, so it is the only one that can disagree with ``q0``'s dtype; under
    ``jax_enable_x64`` that disagreement is a ``lax.scan`` carry-dtype error, not
    a rounding difference.
    """
    return jax.tree_util.tree_map(
        lambda x: (jnp.asarray(x, dtype=jnp.float32) if eqx.is_inexact_array(x) else x),
        tree,
    )


@eqx.filter_jit
def _rollout(model, q, p, steps: int, dt: float, gamma: float):
    return jax.vmap(lambda a, b: model(a, b, steps, dt, gamma))(q, p)


@eqx.filter_jit
def _grad_vecs(model, q):
    V = model.potential_net
    return jax.vmap(jax.grad(lambda z: V(z)))(q)


def _grad_norms(model, q):
    return jnp.linalg.norm(_grad_vecs(model, jnp.asarray(q)), axis=-1)


def _assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.asarray(points)[:, None, :] - centers[None, :, :], axis=-1)
    return np.argmin(d, axis=1)


def _min_separation(centers: np.ndarray) -> float:
    c = np.asarray(centers, dtype=float)
    if c.shape[0] < 2:
        return float("inf")
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


__all__ = [
    "Trajectory", "CluSystemConfig", "ReadState", "ReadResult", "WriteReport",
    "ConsolidationReport", "LearnedVStore", "CluSystem", "build_system",
    "make_controller", "settled_point_psi", "tail_mean_psi",
    "store_relative_trajectory",
    # C2W2 seams
    "WRITE_OBJECTIVE_KEYS", "normalize_write_objective",
    "resolve_store_potential_factory",
    "resolve_store_write_mask_factory",
]
