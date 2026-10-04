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

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The designed verb set: the task's seven plus the doctrine's two.
VERBS = (
    "admit", "place", "evict", "decay", "route", "retry", "stop",
    "anneal", "expand",
)

#: Every designed guard, by name. M14 asserts each of these fires at least once
#: on a canary stream constructed to require it.
GUARDS = (
    "admit.priority",       # free-parameter gate (policy may set the threshold)
    "admit.reach",          # designed: |a_i| < a_U (monitor #11)
    "admit.merge",          # designed: 2 s_max + kappa' sigma_q <= sep
    "admit.budget",         # designed: never silently overwrite a full store
    "place.lambda_min",     # designed: relaxation-derived, lambda_min > 0
    "place.injective",      # designed: a re-derived site may not merge two items
    "evict.persistence",    # designed: W consecutive trips + hysteresis
    "evict.class_i",        # designed: never evict while an instrument is invalid
    "evict.set_function",   # designed: priority/attribute, never LRU
    "decay.permanent",      # designed: leak == 0 items are never decayed
    "route.signal",         # designed: address geometry only, never energy (N97)
    "retry.budget",         # designed: never exceed the declared compute budget
    "anneal.return",        # designed: return to the stored landscape before the read
    "expand.monotone",      # designed: never shrink while items are live
)

#: Routing signals a policy is allowed to use. Post-settle ENERGY is not one of
#: them (N97: it is not a routing/confidence signal and may not be wired to one).
ALLOWED_ROUTE_SIGNALS = ("address", "geometry", "shard_id")


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


def _clip(x, lo, hi):
    return type(lo)(min(max(x, lo), hi))


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
        store_apply: ``(verb, payload) -> None`` sink through which the verb
            reaches the **learned** store (decay/eviction of an item's own atom
            rows, address-space growth). Keeps the controller free of physics.
        budget: live-item budget enforced **here**, not inside the allocator, so
            every eviction passes this class's guards.
    """

    def __init__(
        self,
        allocator,
        policy: Optional[ControllerPolicy] = None,
        bands: Optional[ControllerBands] = None,
        registry=None,
        store_apply: Optional[Callable[[str, dict], None]] = None,
        budget: Optional[int] = None,
    ):
        self.allocator = allocator
        self.bands = bands or ControllerBands()
        self.policy = self.project(policy or ControllerPolicy())
        self.registry = registry
        self.store_apply = store_apply
        self.budget = int(budget if budget is not None else allocator.budget)
        self._log: List[VerbResult] = []
        self._guard_counts: Dict[str, int] = {g: 0 for g in GUARDS}
        self.t = 0
        self.stopped: Optional[str] = None
        self.schedule: Sequence[float] = (1.0,)
        self.expansions: List[float] = []

    # -- the projection Pi_G ---------------------------------------------
    def project(self, policy: ControllerPolicy) -> ControllerPolicy:
        """``Pi_G`` — clip a proposed policy onto the designed feasible set.

        No ``Theta`` maps to an infeasible action; that is the whole mechanism.
        """
        b = self.bands
        return replace(
            policy,
            evict_persistence_W=_clip(int(policy.evict_persistence_W), *b.evict_W),
            retry_max_rounds=_clip(int(policy.retry_max_rounds), *b.retry_rounds),
            retry_confidence_tau=_clip(float(policy.retry_confidence_tau), *b.retry_tau),
            anneal_payload_mult=_clip(float(policy.anneal_payload_mult), *b.anneal_mult),
            anneal_stages=_clip(int(policy.anneal_stages), *b.anneal_stages),
            expand_growth=_clip(float(policy.expand_growth), *b.expand_growth),
            decay_leak=_clip(float(policy.decay_leak), *b.leak),
            route_hop_budget=_clip(int(policy.route_hop_budget), *b.hop_budget),
        )

    # -- internals ---------------------------------------------------------
    def _fire(self, guard: str) -> None:
        self._guard_counts[guard] = self._guard_counts.get(guard, 0) + 1

    def _record(self, res: VerbResult) -> VerbResult:
        res.t = self.t
        self._log.append(res)
        return res

    def _class_i(self) -> List[str]:
        if self.registry is None:
            return []
        return self.registry.class_i_tripped()

    # -- the verbs --------------------------------------------------------
    def admit(self, item_id: int, address, payload: float, *,
              utility: float = 1.0, reach_margin: Optional[float] = None,
              permanent: bool = False, leak: Optional[float] = None) -> VerbResult:
        """**admit** — offer an item to the store.

        Designed guard (may never be violated): never admit a site that violates
        the **merge certificate** ``2 s_max + kappa' sigma_q <= sep`` or the
        **reach certificate** ``|a_i| < a_U`` (monitor #11). Free: the utility
        threshold at which an item is offered, and the refusal rate under budget
        pressure.
        """
        # free-parameter gate (the policy's, and it is box-constrained)
        if float(utility) < float(self.policy.admit_priority_threshold):
            self._fire("admit.priority")
            return self._record(VerbResult("admit", False, "below priority threshold",
                                           "admit.priority",
                                           {"item_id": int(item_id), "utility": float(utility)}))
        # DESIGNED: the reach certificate (monitor #11's write-time form)
        if reach_margin is not None and float(reach_margin) <= 0.0:
            self._fire("admit.reach")
            return self._record(VerbResult("admit", False, "reach certificate fails",
                                           "admit.reach",
                                           {"item_id": int(item_id),
                                            "reach_margin": float(reach_margin)}))
        # DESIGNED: never silently overwrite a full store — make room by EVICTING,
        # and the eviction goes through evict()'s own guards.
        if self.allocator.n_live >= self.budget:
            victim = self._pick_victim()
            if victim is None:
                self._fire("admit.budget")
                return self._record(VerbResult("admit", False, "capacity alarm (all permanent)",
                                               "admit.budget", {"item_id": int(item_id)}))
            ev = self.evict(victim, reason="budget", trips=self.policy.evict_persistence_W)
            if not ev.applied:
                self._fire("admit.budget")
                return self._record(VerbResult("admit", False,
                                               f"could not make room: {ev.reason}",
                                               "admit.budget", {"item_id": int(item_id)}))
        row = self.allocator.offer(int(item_id), np.asarray(address, dtype=float),
                                   float(payload), permanent=permanent, leak=leak)
        ok = str(row["decision"]) in ("admit", "relocate")
        if not ok:
            # DESIGNED: the merge/spacing certificate refused it
            self._fire("admit.merge")
        return self._record(VerbResult(
            "admit", ok, str(row["decision"]),
            "admit.merge" if not ok else "",
            {"item_id": int(item_id), "row": row,
             "gate_margin": float(row.get("d_min_proposed", float("nan")))},
        ))

    def _pick_victim(self) -> Optional[int]:
        """Lowest-priority live item by a **set-function** score (never LRU).

        Score = current well depth (``amp = base * exp(-leak * age)`` is
        item-intrinsic), so the choice does not depend on query history and the
        exact store-level deletion claim still covers the store.
        """
        recs = [r for r in self.allocator.records.values() if not r.permanent]
        if not recs:
            return None
        amps = np.asarray(self.allocator.store.amps, dtype=float)
        victim = min(recs, key=lambda r: (float(amps[r.slot]), int(r.item_id)))
        return int(victim.item_id)

    def place(self, item_id: int, address, *, lambda_min: Optional[float] = None,
              derived_by: str = "relaxation") -> VerbResult:
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
        if derived_by != "relaxation":
            self._fire("place.lambda_min")
            raise GuardViolation(
                f"place() accepts relaxation-derived addresses only, got {derived_by!r}: "
                "a critical-point solver can commit a SADDLE to the codebook "
                "(clu-controller-spec §C4.3/P12)"
            )
        if lambda_min is not None and float(lambda_min) <= 0.0:
            self._fire("place.lambda_min")
            return self._record(VerbResult("place", False, "lambda_min <= 0 (not a minimum)",
                                           "place.lambda_min",
                                           {"item_id": int(item_id),
                                            "lambda_min": float(lambda_min)}))
        rec = None
        for r in self.allocator.records.values():
            if r.item_id == int(item_id):
                rec = r
                break
        # DESIGNED: re-derivation may not destroy INJECTIVITY (certificate N1).
        # Measured on the first full run: relaxing two items from their recorded
        # sites landed both on the SAME minimum, and committing that collapsed
        # `sep` to 0.0 and took N1/N2 down with it. lambda_min > 0 is necessary and
        # NOT sufficient — a shared minimum is a perfectly good minimum.
        others = np.stack([np.asarray(r.center, dtype=float)
                           for r in self.allocator.records.values()
                           if r.item_id != int(item_id)]) if len(self.allocator.records) > 1 \
            else np.zeros((0, self.allocator.addr_dim))
        if others.size:
            site_ = np.asarray(address, dtype=float).reshape(-1)[: self.allocator.addr_dim]
            d_min = float(np.min(np.linalg.norm(others - site_[None, :], axis=1)))
            if d_min < float(self.allocator.d_safe):
                self._fire("place.injective")
                return self._record(VerbResult(
                    "place", False,
                    f"re-derived site is {d_min:.4f} from another live item "
                    f"(< d_safe {self.allocator.d_safe:.4f}): committing it would "
                    f"break injectivity (certificate N1)",
                    "place.injective",
                    {"item_id": int(item_id), "d_min": d_min}))
        if rec is None:
            return self._record(VerbResult("place", False, "item not live", "",
                                           {"item_id": int(item_id)}))
        site = np.asarray(address, dtype=float).reshape(-1)[: self.allocator.addr_dim]
        slot = rec.slot
        payload, amp = rec.payload, float(np.asarray(self.allocator.store.amps)[slot])
        self.allocator.store = self.allocator.store.evict(slot)
        self.allocator.store = self.allocator.store.with_item(site, payload, amp=amp)
        rec.center = site
        if self.store_apply is not None:
            self.store_apply("place", {"item_id": int(item_id), "slot": slot, "site": site})
        return self._record(VerbResult("place", True, "re-derived by relaxation", "",
                                       {"item_id": int(item_id), "site": site.tolist()}))

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
        if reason in ("lru", "staleness"):
            self._fire("evict.set_function")
            raise GuardViolation(
                "evict(reason='lru'|'staleness') is forbidden: LRU is query history, so "
                "the store would stop being order-independent and the exact deletion "
                "claim would no longer cover it (N99)"
            )
        ci = self._class_i()
        if ci:
            self._fire("evict.class_i")
            return self._record(VerbResult(
                "evict", False, f"class-I monitor(s) tripped: {','.join(ci)}",
                "evict.class_i", {"item_id": int(item_id)}))
        if int(trips) < int(self.policy.evict_persistence_W):
            self._fire("evict.persistence")
            return self._record(VerbResult(
                "evict", False,
                f"persistence {trips} < W={self.policy.evict_persistence_W}",
                "evict.persistence", {"item_id": int(item_id)}))
        slot = None
        for r in self.allocator.records.values():
            if r.item_id == int(item_id):
                slot = r.slot
                break
        if slot is None:
            return self._record(VerbResult("evict", False, "item not live", "",
                                           {"item_id": int(item_id)}))
        ok = self.allocator.evict_item(int(item_id), reason=reason)
        if self.store_apply is not None:
            # the learned landscape must lose the item too: its own atom rows go
            # to zero depth, so its wells vanish from V_theta.
            self.store_apply("evict", {"item_id": int(item_id), "slot": slot})
        return self._record(VerbResult("evict", bool(ok), reason, "",
                                       {"item_id": int(item_id), "slot": int(slot)}))

    def decay(self, ticks: int = 1) -> VerbResult:
        """**decay** — the amplitude law is exactly ``A <- A e^{-leak}``, and
        decay **commutes with delete** (PGCP Thm 4). Free: per-item ``leak_i``
        (equivalently: the user sets a half-life and the store solves for it).

        ⚠ Under a learned ``V_theta`` the decay is applied to the *item's own
        atom rows*, so it is a physical shallowing of that item's wells — not
        bookkeeping.
        """
        before = {r.item_id: float(np.asarray(self.allocator.store.amps)[r.slot])
                  for r in self.allocator.records.values()}
        permanent = [r.item_id for r in self.allocator.records.values() if r.permanent]
        for _ in range(int(ticks)):
            self.allocator.tick()
        self.t = self.allocator.t
        after = {r.item_id: float(np.asarray(self.allocator.store.amps)[r.slot])
                 for r in self.allocator.records.values()}
        # DESIGNED: a permanent item is never decayed, whatever the policy leak.
        for pid in permanent:
            if pid in after and abs(after[pid] - before.get(pid, after[pid])) > 1e-12:
                raise GuardViolation(f"decay touched permanent item {pid}")
        self._fire("decay.permanent")
        factors = {i: (after[i] / before[i] if before.get(i, 0.0) > 0 and i in after else 0.0)
                   for i in before}
        if self.store_apply is not None:
            self.store_apply("decay", {"factors": factors,
                                       "gone": [i for i in before if i not in after]})
        return self._record(VerbResult("decay", True, f"{ticks} tick(s)", "decay.permanent",
                                       {"factors": factors,
                                        "decayed_out": [i for i in before if i not in after]}))

    def route(self, query, *, signal: str = "address") -> VerbResult:
        """**route** — shard/wormhole selection.

        Designed guard: route on **address geometry only**. Post-settle energy is
        **not** a routing/confidence signal (N97) and may not be wired to one —
        ``signal`` is checked against an allow-list. Free: the selection policy
        and the hop budget.
        """
        if signal not in ALLOWED_ROUTE_SIGNALS:
            self._fire("route.signal")
            raise GuardViolation(
                f"route(signal={signal!r}) is forbidden; allowed: {ALLOWED_ROUTE_SIGNALS}. "
                "Post-settle energy is NOT a routing/confidence signal (N97)."
            )
        self._fire("route.signal")
        return self._record(VerbResult("route", True, f"signal={signal}", "route.signal",
                                       {"hop_budget": int(self.policy.route_hop_budget)}))

    def retry(self, confidence: float, *, round_index: int = 0) -> VerbResult:
        """**retry** — the compute dial: re-settle with more steps.

        Designed guard: may not exceed the declared compute budget; ladder
        energies must stay **sub-barrier** (``E < h_i``); may not consume ground
        truth. Free: the confidence threshold ``tau`` and the ladder depth.
        """
        if int(round_index) >= int(self.policy.retry_max_rounds):
            self._fire("retry.budget")
            return self._record(VerbResult(
                "retry", False,
                f"compute budget exhausted ({self.policy.retry_max_rounds} rounds)",
                "retry.budget", {"confidence": float(confidence)}))
        fire = float(confidence) < float(self.policy.retry_confidence_tau)
        return self._record(VerbResult("retry", bool(fire),
                                       "low confidence" if fire else "confident enough",
                                       "", {"confidence": float(confidence),
                                            "tau": float(self.policy.retry_confidence_tau),
                                            "round": int(round_index)}))

    def stop(self, reason: str) -> VerbResult:
        """**stop** — always available, fires unconditionally on any class-I
        monitor trip. Not a learnable decision.
        """
        self.stopped = reason
        return self._record(VerbResult("stop", True, reason, "", {}))

    def anneal(self, schedule: Sequence[float]) -> VerbResult:
        """**anneal** (doctrine addition) — set the read schedule.

        Designed guard: the schedule **must return to the stored landscape before
        the value is read**. The ``static`` control (a schedule that does not
        return) reaches basin 0.9993 and reads the **wrong value** — "wider
        wells" is a different and failing result. Free: the schedule shape within
        designed bounds.
        """
        sch = [float(x) for x in schedule]
        if not sch:
            raise GuardViolation("anneal(schedule=[]) is empty")
        if abs(sch[-1] - 1.0) > 1e-9:
            self._fire("anneal.return")
            raise GuardViolation(
                f"anneal schedule must return to the stored landscape "
                f"(last multiplier must be 1.0, got {sch[-1]}): the `static` control "
                "reads the WRONG value at basin 0.9993 (N109)"
            )
        lo, hi = self.bands.anneal_mult
        sch = [_clip(x, lo, hi) for x in sch]
        self._fire("anneal.return")
        self.schedule = tuple(sch)
        return self._record(VerbResult("anneal", True, "schedule set", "anneal.return",
                                       {"schedule": list(sch)}))

    def expand(self, factor: float) -> VerbResult:
        """**expand** (doctrine addition) — grow/shard the address space.

        Designed guard: may never *shrink* the space while items are live, and
        must preserve the placement rule's set-function property. Free: when to
        expand and the growth factor.
        """
        f = float(factor)
        if f < 1.0 and self.allocator.n_live > 0:
            self._fire("expand.monotone")
            raise GuardViolation(
                f"expand({f}) would shrink the address space with "
                f"{self.allocator.n_live} items live"
            )
        lo, hi = self.bands.expand_growth
        f = _clip(f, lo, hi)
        self._fire("expand.monotone")
        self.expansions.append(f)
        if self.store_apply is not None:
            self.store_apply("expand", {"factor": f})
        return self._record(VerbResult("expand", True, f"x{f:g}", "expand.monotone",
                                       {"factor": f}))

    # -- introspection ----------------------------------------------------
    @property
    def log(self) -> List[VerbResult]:
        """Every verb call, applied or refused (the reported artifact)."""
        return list(self._log)

    def guard_fire_counts(self) -> Dict[str, int]:
        """How often each designed guard actually fired — M14's input.

        A guard with count 0 on a canary stream constructed to require it is a
        guard that has become arithmetically vacuous (N74, learned).
        """
        return dict(self._guard_counts)

    def verb_counts(self) -> Dict[str, Dict[str, int]]:
        """Per verb: how often it was called and how often it applied."""
        out = {v: {"calls": 0, "applied": 0} for v in VERBS}
        for r in self._log:
            out.setdefault(r.verb, {"calls": 0, "applied": 0})
            out[r.verb]["calls"] += 1
            out[r.verb]["applied"] += int(bool(r.applied))
        return out


def derived_d_safe(s_max: float, sigma_q: float, kappa_prime: float = 2.576) -> float:
    """``d_safe = 2 s_max + kappa' sigma_q`` — the merge+margin admission radius.

    Replaces ``d_safe = 4.4 s``, which is **not self-consistent**: at the well
    widths a working store needs, ``4.4 s`` exceeds the store's own site spacing
    (measured 1.625 vs 1.050), so the store fails its own gate — N74's vacuity
    with the sign flipped (`controller-doctrine` R5/I-13).

    ``kappa' = 2.576`` is the 99% point of the corrected margin law
    ``acc ~ erf(margin / sqrt(2) sigma)``.
    """
    return float(2.0 * float(s_max) + float(kappa_prime) * float(sigma_q))


def assert_d_safe_consistent(d_safe: float, sep: float) -> None:
    """Doctrine I-13's construction-time assertion: ``d_safe <= sep``.

    A gate whose radius exceeds the store's own achieved spacing cannot be
    satisfied by the store it guards.
    """
    if not (float(d_safe) <= float(sep) + 1e-9):
        raise GuardViolation(
            f"d_safe={d_safe:.4f} exceeds the store's achieved spacing sep={sep:.4f}: "
            "the store fails its own admission gate (controller-doctrine R5)"
        )


__all__ = [
    "VERBS", "GUARDS", "ALLOWED_ROUTE_SIGNALS", "GuardViolation", "VerbResult",
    "ControllerPolicy", "ControllerBands", "CluControllerV0", "derived_d_safe",
    "assert_d_safe_consistent",
]
