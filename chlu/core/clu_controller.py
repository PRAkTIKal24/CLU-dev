"""Controller v0 — the designed verb set, with designed guards and free params.

**Designed action space, learned policy** (charter §3.2). The verbs and their
guards are *designed*; when and how hard to pull them is left free for a policy
to set. v0 ships a hand-set policy: the free parameters exist, are declared, and
are box-constrained, but nothing is learned yet.

The formal split (`controller-doctrine` §6, P4):

    the policy proposes ``u = pi_Theta(obs)``; the controller executes
    ``Pi_G(u)`` — the **projection** of ``u`` onto the designed feasible set.
    **Guards are CONSTRAINTS, never PENALTIES.** ``G`` appears in no loss.

Why exactly this form: if a guard were a penalty the policy could trade it off
against reward, and with enough reward pressure it will — that is w20 ("free
learning erases design") reproduced one level up. Under projection an infeasible
action is not discouraged, it is **unreachable**.

**The verb set.** The task's designed set is {admit, place, evict, decay, route,
retry, stop}. `controller-doctrine` §4 proves it **incomplete** and adds two,
each measured-load-bearing:

* ``expand(factor)`` — grow/shard the address space. N91 measured that with a
  fixed address space the whole controller is capped at ``N_pack/K`` (per-offered
  0.081, last of seven) and that sizing the space to the load takes the same
  controller to 0.669, beating all four primitives. **The binding constraint was
  the address space and there was no verb for it.**
* ``anneal(schedule)`` — set the read schedule (widths, gamma, step budget).
  ``retry`` is the degenerate case (same schedule, more steps). The annealed read
  is the measured mechanism that unclamped ``K_learned(4)`` 16 -> 32 at zero
  extra bytes/dims/steps (N109).

**And one verb that must NOT exist.** Monitor #2 (settle -> arg-min, the
dividend) has **no restoring verb**: it escalates. A controller able to act on
the dividend would learn to act by suppressing the settle — which is exactly how
w26's same-keys result happened.

This module **wraps** :class:`chlu.core.controller.Controller` (C1's MVC-0
allocator: admission, canonical placement, eviction, decay bookkeeping) and adds
the C2 verb surface on top. It edits nothing in C1's file (C1W27 owns it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


#: The designed verb set: the task's seven plus the doctrine's two.
VERBS = (
    "admit", "place", "evict", "decay", "route", "retry", "stop",
    "anneal", "expand",
)


class GuardViolation(RuntimeError):
    """Raised when a caller asks for an action outside the designed feasible set.

    This is *not* how a policy is meant to be constrained — a policy's proposal
    is projected (:meth:`CluControllerV0.project`), never rejected. The exception
    exists for direct/programmatic misuse, so a guard can never be bypassed by
    calling the verb method itself.
    """


@dataclass
class VerbResult:
    """The outcome of one verb call (a row of the controller log)."""

    verb: str
    applied: bool
    reason: str = ""
    guard: str = ""  # which designed guard decided this
    detail: Dict[str, Any] = field(default_factory=dict)
    t: int = 0


@dataclass
class ControllerPolicy:
    """The **free** parameters — "when" and "how hard". Each is box-constrained
    to a band whose endpoints are designed (`controller-doctrine` §6 table).

    v0 sets them by hand. A learned policy replaces the values, never the bands.
    """

    admit_priority_threshold: float = 0.0  # offer utility below which we abstain
    evict_persistence_W: int = 1  # consecutive trips required before evicting
    decay_leak: float = 0.0  # per-tick leak for a non-permanent item
    retry_confidence_tau: float = 0.5  # below this confidence, retry
    retry_max_rounds: int = 1
    anneal_payload_mult: float = 1.0  # read-time payload-width multiplier
    anneal_stages: int = 1
    expand_growth: float = 1.0  # address-space growth factor per expand
    route_hop_budget: int = 0


@dataclass
class ControllerBands:
    """The **designed** endpoints the policy is projected onto."""

    evict_W: Tuple[int, int] = (1, 8)
    retry_rounds: Tuple[int, int] = (0, 4)
    retry_tau: Tuple[float, float] = (0.0, 1.0)
    anneal_mult: Tuple[float, float] = (1.0, 8.0)
    anneal_stages: Tuple[int, int] = (1, 16)
    expand_growth: Tuple[float, float] = (1.0, 4.0)
    leak: Tuple[float, float] = (0.0, 1.0)
    hop_budget: Tuple[int, int] = (0, 4)


class CluControllerV0:
    """The seven designed verbs (+2), each with a guard it may not violate.

    Args:
        allocator: a :class:`chlu.core.controller.Controller` — C1's MVC-0
            admission/placement/eviction/decay machinery, used **as the address
            allocator and codebook**. The energy landscape itself is the learned
            ``V_theta`` held by :class:`chlu.core.clu_system.LearnedVStore`; the
            allocator's own designed store is never read by the CLU read path
            (it is the codebook, and it is exactly what the same-keys launder
            gets to use).
        policy: the free parameters (v0: hand-set).
        bands: the designed endpoints the policy is projected onto.
        registry: the monitor registry, consulted for class-I trips before any
            memory-mutating verb (doctrine §5, consequence 1).
        d_safe: admission radius. ⚠ ``4.4 s`` is **not** self-consistent at
            working widths (measured 1.625 vs a store spacing of 1.050 — the
            store fails its own gate); use
            :func:`derived_d_safe` = ``2 s_max + kappa' sigma_q``.
    """

    def __init__(
        self,
        allocator,
        policy: Optional[ControllerPolicy] = None,
        bands: Optional[ControllerBands] = None,
        registry=None,
        store_apply: Optional[Callable[[str, dict], None]] = None,
    ):
        raise NotImplementedError

    # -- the projection Pi_G ---------------------------------------------
    def project(self, policy: ControllerPolicy) -> ControllerPolicy:
        """``Pi_G`` — clip a proposed policy onto the designed feasible set.

        No ``Theta`` maps to an infeasible action; that is the whole mechanism.
        """
        raise NotImplementedError

    # -- the verbs --------------------------------------------------------
    def admit(self, item_id: int, address, payload: float, *,
              utility: float = 1.0, reach_margin: Optional[float] = None,
              permanent: bool = False) -> VerbResult:
        """**admit** — offer an item to the store.

        Designed guard (may never be violated): never admit a site that violates
        the **merge certificate** ``2 s_max + kappa' sigma_q <= sep`` or the
        **reach certificate** ``|a_i| < a_U`` (monitor #11). Free: the utility
        threshold at which an item is offered, and the refusal rate under budget
        pressure.
        """
        raise NotImplementedError

    def place(self, item_id: int, address) -> VerbResult:
        """**place** — commit the derived address.

        Designed guard: placement must be a **set function of the retained set**
        (canonical/PGCP), so ``delete = set-minus`` holds below capacity / under
        set-function eviction; and the committed candidate must satisfy
        ``lambda_min(H) > 0``. Re-derivation is by ``gamma > 0`` **relaxation**,
        **never** a critical-point solver (a Newton re-derivation once wrote a
        *saddle* into the codebook and the deadband preserved it for 150 epochs).
        Free: the priority function — provided it depends only on the key, never
        on arrival order.
        """
        raise NotImplementedError

    def evict(self, item_id: int, *, reason: str = "budget",
              trips: int = 0) -> VerbResult:
        """**evict** — irreversible; the maximal element of the trigger order.

        Designed guard: requires ``W`` consecutive trips + hysteresis + **no
        class-I monitor tripped in this step**; eviction must itself be a
        set-function policy (priority/attribute), **never LRU** (LRU is query
        history, so an LRU store is not order-independent and the exact
        store-level deletion claim does not cover it). Free: the eviction
        score's weights and ``W``.
        """
        raise NotImplementedError

    def decay(self, ticks: int = 1) -> VerbResult:
        """**decay** — the amplitude law is exactly ``A <- A e^{-leak}``, and
        decay **commutes with delete** (PGCP Thm 4). Free: per-item ``leak_i``
        (equivalently: the user sets a half-life and the store solves for it).

        ⚠ Under a learned ``V_theta`` the decay is applied to the *item's own
        atom rows*, so it is a physical shallowing of that item's wells — not
        bookkeeping.
        """
        raise NotImplementedError

    def route(self, query, *, signal: str = "address") -> VerbResult:
        """**route** — shard/wormhole selection.

        Designed guard: route on **address geometry only**. Post-settle energy is
        **not** a routing/confidence signal (N97) and may not be wired to one —
        ``signal`` is checked against an allow-list. Free: the selection policy
        and the hop budget.
        """
        raise NotImplementedError

    def retry(self, confidence: float, *, round_index: int = 0) -> VerbResult:
        """**retry** — the compute dial: re-settle with more steps.

        Designed guard: may not exceed the declared compute budget; ladder
        energies must stay **sub-barrier** (``E < h_i``); may not consume ground
        truth. Free: the confidence threshold ``tau`` and the ladder depth.
        """
        raise NotImplementedError

    def stop(self, reason: str) -> VerbResult:
        """**stop** — always available, fires unconditionally on any class-I
        monitor trip. Not a learnable decision.
        """
        raise NotImplementedError

    def anneal(self, schedule: Sequence[float]) -> VerbResult:
        """**anneal** (doctrine addition) — set the read schedule.

        Designed guard: the schedule **must return to the stored landscape before
        the value is read**. The ``static`` control (a schedule that does not
        return) reaches basin 0.9993 and reads the **wrong value** — "wider
        wells" is a different and failing result. Free: the schedule shape within
        designed bounds.
        """
        raise NotImplementedError

    def expand(self, factor: float) -> VerbResult:
        """**expand** (doctrine addition) — grow/shard the address space.

        Designed guard: may never *shrink* the space while items are live, and
        must preserve the placement rule's set-function property. Free: when to
        expand and the growth factor.
        """
        raise NotImplementedError

    # -- introspection ----------------------------------------------------
    @property
    def log(self) -> List[VerbResult]:
        """Every verb call, applied or refused (the reported artifact)."""
        raise NotImplementedError

    def guard_fire_counts(self) -> Dict[str, int]:
        """How often each designed guard actually fired — M14's input.

        A guard with count 0 on a canary stream constructed to require it is a
        guard that has become arithmetically vacuous (N74, learned).
        """
        raise NotImplementedError


def derived_d_safe(s_max: float, sigma_q: float, kappa_prime: float = 2.576) -> float:
    """``d_safe = 2 s_max + kappa' sigma_q`` — the merge+margin admission radius.

    Replaces ``d_safe = 4.4 s``, which is **not self-consistent**: at the well
    widths a working store needs, ``4.4 s`` exceeds the store's own site spacing
    (measured 1.625 vs 1.050), so the store fails its own gate — N74's vacuity
    with the sign flipped (`controller-doctrine` R5/I-13).

    ``kappa' = 2.576`` is the 99% point of the corrected margin law
    ``acc ~ erf(margin / sqrt(2) sigma)``.
    """
    raise NotImplementedError


__all__ = [
    "VERBS", "GuardViolation", "VerbResult", "ControllerPolicy",
    "ControllerBands", "CluControllerV0", "derived_d_safe",
]
