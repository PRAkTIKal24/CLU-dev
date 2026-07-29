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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

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

    # -- the learned V_theta store ---------------------------------------
    capacity: int = 12
    atoms_per_item: int = 8
    atom_width: float = 0.15
    atom_depth_init: float = 1e-4
    atom_init_scale: float = 1.0
    atom_local_radius: float = 0.0  # N98 localized init; 0.0 = historical scatter
    confine: float = 0.05

    # -- the write ---------------------------------------------------------
    write_steps: int = 300
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    write_n_perturb: int = 16
    masked_write: bool = True  # local in parameter space (C3-local)

    # -- the read ----------------------------------------------------------
    dt: float = 0.05
    gamma_address: float = 0.2  # doctrine row 1: measured optimum ~0.2
    gamma_read: float = 0.2
    address_steps: int = 400
    read_steps: int = 200
    traj_stride: int = 8  # the strided trajectory buffer
    kinetic_mode: str = "newtonian_learned"
    query_sigma: float = 0.24  # sigma_q, the doctrine grid's in-band value

    # -- control -----------------------------------------------------------
    d_safe_kappa_prime: float = 2.576  # 99% point of the corrected margin law
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

    # -- harness -----------------------------------------------------------
    seed: int = 0
    n_query_per_item: int = 8
    quick: bool = False

    @classmethod
    def from_mapping(cls, overrides: Optional[dict] = None) -> "CluSystemConfig":
        """Build from a YAML/JSON mapping, ignoring unknown keys.

        Mirrors ``chlu.config.load_config``'s tolerance so an old project file
        does not crash a new schema. This is the config-driven override path for
        ``projects/<name>/config/config.yaml`` under a ``clu_system:`` block.
        """
        raise NotImplementedError

    def as_flag_table(self) -> Dict[str, Any]:
        """Every non-default flag in effect — the flag-provenance table."""
        raise NotImplementedError


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
      value. ``payload_of`` exists for **eval/launder/monitors only** and is
      asserted never-read-by-``read`` in ``tests/test_clu_system.py``.
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
        raise NotImplementedError

    def group_rows(self, slot: int) -> jnp.ndarray:
        """Boolean atom-row mask owned by ``slot`` (the masked-write support)."""
        raise NotImplementedError

    def scale_group_amplitude(self, slot: int, factor: float) -> "LearnedVStore":
        """Multiply that item's atom **depths** by ``factor`` (decay/eviction).

        ``A_j = amp_j^2``, so the amplitude parameter is scaled by
        ``sqrt(factor)``; ``factor = 0`` is a physical eviction (the item's wells
        vanish from ``V_theta`` and its rows are zeroed).
        """
        raise NotImplementedError

    def n_bytes(self) -> int:
        """Bytes of learned state (float32) — the matched-bytes denominator."""
        raise NotImplementedError


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
    """

    def __init__(
        self,
        store: LearnedVStore,
        phi: Optional[Callable] = None,
        psi: Optional[Callable[[Trajectory, ReadState], jnp.ndarray]] = None,
        controller=None,
        registry=None,
        config: Optional[CluSystemConfig] = None,
    ):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def consolidate(self, key=None) -> ConsolidationReport:
        """Offline maintenance: re-pack, enforce decay, re-check certificates,
        run the label-free **self-probe** pass (the store re-reads its own
        written items), and re-calibrate the gate.

        This is where wake-sleep is repositioned (charter §2.4): consolidation,
        not the trainer.
        """
        raise NotImplementedError

    # -- hooks -------------------------------------------------------------
    def fixed_point_residual(self, q, p, *, gamma: Optional[float] = None):
        """``(q, p) - Step_gamma(q, p)`` — the settle's fixed-point residual.

        The attachment point for an implicit/DEQ gradient
        (`trainability-spike`): a settle is a fixed point, so one differentiates
        through the equilibrium rather than the unroll.
        """
        raise NotImplementedError

    def self_probe(self, key=None) -> Dict[str, Any]:
        """Label-free diagnostic pass: the store re-reads what it wrote.

        Feeds monitors #5 (acquisition), #6 (objective divergence), #9
        (lifetimes) and #12 (starvation) without ever consulting a task label.
        """
        raise NotImplementedError

    def codebook(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(ids, addresses, payloads)`` over live items — **eval only**.

        The read path never calls this. It exists so the same-keys launder can be
        constructed (doctrine I-1) and so monitors have ground truth about what
        the store believes it holds.
        """
        raise NotImplementedError

    def n_bytes(self) -> int:
        """Matched-bytes accounting for the full system."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# psi: the v0 handcrafted read-out (a learned psi is `trainability-spike`'s)
# --------------------------------------------------------------------------
def settled_point_psi(addr_dim: int, payload_dim: int = 1) -> Callable:
    """The v0 default read-out: the payload channels of the **settled point**.

    Deliberately the *weakest* read in the space, because it is the one 26 waves
    used: point-vs-trajectory then becomes an ablation against a known baseline
    rather than a comparison against a new one.
    """
    raise NotImplementedError


def tail_mean_psi(addr_dim: int, payload_dim: int = 1,
                  tail_frac: float = 0.5) -> Callable:
    """A trajectory read-out: mean payload channel over the phase-2 tail.

    Still handcrafted (no parameters), but it *uses the trajectory*, so the
    trajectory buffer has a consumer in v0 and monitor #4's trajectory-launder
    requirement is exercised before a learned psi lands.
    """
    raise NotImplementedError


def store_relative_trajectory(traj: Trajectory, state: ReadState,
                              mask_q0: bool = False) -> Trajectory:
    """``traj - q0`` (and optionally drop the ``q0`` point) — doctrine I-2.

    The trajectory **contains ``q0 = phi(x)``**, so a psi over the raw buffer has
    direct access to the query embedding and a blank-store psi read is exactly "a
    classifier on ``phi(x)``" (N68's 1e-4 leak, at 100% strength). Any trajectory
    read must be reported alongside its store-relative and ``q0``-only laundered
    forms.
    """
    raise NotImplementedError


__all__ = [
    "Trajectory", "CluSystemConfig", "ReadState", "ReadResult", "WriteReport",
    "ConsolidationReport", "LearnedVStore", "CluSystem",
    "settled_point_psi", "tail_mean_psi", "store_relative_trajectory",
]
