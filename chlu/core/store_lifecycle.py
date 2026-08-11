"""⭐ **C2W10 — the THREE-STATE store lifecycle** (charter Add.12 §A34.3).

The states are **PROTECTED <-> ACTIVE -> TRASH**:

``PROTECTED``
    no decay (``leak = 0``, the allocator's existing permanent flag).
``ACTIVE``
    the designed decay applies.
``TRASH``
    routed to ``gamma_phi(q)`` through :meth:`chlu.core.clu_system.CluSystem.trash_route`
    (C2W8's K2 region — C2W10 is its first experimental use ON).

⛔ **Demotion is PROTECTED -> ACTIVE, NEVER to trash.** Trash is the
never-useful / spurious route only.
⛔ **Depth never enters the usefulness criterion** (§A28.3(ii): erosion drives
depth to zero, and depth != usefulness). The registered proxy is item-id-keyed
``read_hits``, aggregated per chunk and per stream, and nothing else.
⛔ **The optimizer's erosion is churn, not curation** (§A28.3(iii)): nothing here
reads a gradient, and no lifecycle quantity is ever a loss term. The
protected-fraction bound fails **loudly at runtime** through a named monitor row.

**Build order.** The designed negatives of L1-L5 landed RED against a stub in the
preceding commit (doctrine §A12) and are unchanged by this one.

**What lives where.** The verbs are decided here and applied through two narrow
seams: :meth:`chlu.core.controller.Controller.set_permanence` (promotion/
demotion) and a caller-supplied ``trash_route`` callable (the trash route). This
module owns no physics: it never touches ``V_theta`` except through the store's
own published methods.

⚠ ``refresh_monotonic`` also exists as a **memory-cell** flag at
``chlu/core/blocks.py:683`` (C2W6, ships OFF). That file is frozen CSF3
territory; the guard here is the **store-level** implementation of the same
rule, and :func:`replay_rewrite_events` exists so the two can be validated
against each other on the *same* recorded rewrite events (L5-b) rather than
silently diverging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

#: the three lifecycle states
PROTECTED = "PROTECTED"
ACTIVE = "ACTIVE"
TRASH = "TRASH"
STATES = (PROTECTED, ACTIVE, TRASH)

#: the two readings of §A20.6's trash criterion. Both ship; the default is
#: declared here and asserted to differ in ``tests/test_store_lifecycle.py``.
TRASH_CRITERIA = ("last_k_streams", "since_first_seen")

#: the monitor row this module adds (severity registered in
#: :data:`chlu.core.monitors.SEVERITY`).
PROTECTED_SATURATION = "protected_saturation"


@dataclass
class LifecycleParams:
    """Every lifecycle knob, with the shipped default **OFF** (L7).

    ⚠ ``d_dwell > window`` is a **derived** constraint, not a taste (PREREG §P2):
    a single burst of ``h_hi`` hits in chunk ``c`` holds the trailing-window test
    for exactly ``window`` consecutive chunks and no more, so the hysteresis
    binds — and L1's burst negative can fire at all — iff ``d_dwell > window``.
    With ``d_dwell <= window`` the designed negative is *arithmetically incapable
    of failing*, which is the vacuous-guard defect class. Hence
    :meth:`_assert_hysteresis_binds`, called at construction of
    :class:`StoreLifecycle`.
    """

    #: the master switch. ``False`` (shipped) => no verb ever fires (L7).
    lifecycle: bool = False
    promote: bool = True
    demote: bool = True
    trash: bool = True

    # -- L1 promotion / L2 demotion --
    h_hi: int = 2          # trailing-window hits qualifying for promotion
    h_lo: int = 1          # below this the well counts as unused
    window: int = 2        # trailing window length, in chunks
    d_dwell: int = 3       # chunks the >= h_hi condition must be SUSTAINED
    d_demote: int = 2      # chunks below h_lo before PROTECTED -> ACTIVE

    # -- L3 trash --
    k_streams: int = 3     # stream boundaries the criterion looks back over
    trash_criterion: str = "last_k_streams"
    censoring_guard: bool = True   # never-useful-YET != never-useful

    # -- L4 the protected fraction --
    f_max: float = 0.25    # Hub default, NOT re-derived (PREREG §P3)

    # -- the designed decay a demoted well is re-exposed to --
    leak: float = 0.02

    # -- L5 I1 refresh-monotonicity (store level; ships OFF, like blocks.py) --
    refresh_monotonic: bool = False
    #: amplitude-gain cap, in ``blocks.py``'s units (depth gain is its square).
    refresh_max_gain: float = 4.0

    def __post_init__(self):
        if str(self.trash_criterion) not in TRASH_CRITERIA:
            raise ValueError(
                f"trash_criterion must be one of {TRASH_CRITERIA}, "
                f"got {self.trash_criterion!r}"
            )

    def _assert_hysteresis_binds(self) -> "LifecycleParams":
        if int(self.d_dwell) <= int(self.window):
            raise ValueError(
                f"d_dwell={self.d_dwell} <= window={self.window}: a single burst "
                f"satisfies the trailing-window test for exactly `window` chunks, so "
                f"the hysteresis does not bind and L1's burst negative cannot fail. "
                f"A guard that cannot be shown to fail is not a guard (§A12)."
            )
        return self

    @classmethod
    def from_config(cls, group) -> "LifecycleParams":
        """Build from a config group, taking only the fields that exist on both."""
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: getattr(group, k) for k in names if hasattr(group, k)})

    def as_flag_table(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ==========================================================================
# the pure decision layer — no state, no store, trivially falsifiable
# ==========================================================================
def _window_hits(hits_by_chunk: Sequence[int], chunk: int, window: int) -> int:
    lo = max(0, int(chunk) - int(window) + 1)
    return int(sum(int(h) for h in hits_by_chunk[lo: int(chunk) + 1]))


def promotion_dwell(hits_by_chunk: Sequence[int], chunk: int,
                    params: LifecycleParams) -> int:
    """Consecutive chunks ending at ``chunk`` whose trailing window is ``>= h_hi``."""
    d = 0
    c = int(chunk)
    while c >= 0 and _window_hits(hits_by_chunk, c, params.window) >= int(params.h_hi):
        d += 1
        c -= 1
    return d


def should_promote(hits_by_chunk: Sequence[int], chunk: int,
                   params: LifecycleParams) -> bool:
    """**L1**: ACTIVE -> PROTECTED on *sustained* usage (hysteresis)."""
    if not params.promote:
        return False
    return promotion_dwell(hits_by_chunk, chunk, params) >= int(params.d_dwell)


def demotion_dwell(hits_by_chunk: Sequence[int], chunk: int,
                   params: LifecycleParams) -> int:
    """Consecutive chunks ending at ``chunk`` whose **own** hits are ``< h_lo``.

    ⚠ **Deliberately asymmetric with :func:`promotion_dwell`, and the asymmetry
    is forced by the registered statements.** L1 is scored on the *trailing
    window* (so a burst cannot promote); L2 must fire *"within ``d_demote``
    chunks of usage falling below ``h_lo``"*, and a trailing window of length
    ``W`` carries the last pre-abandonment hits forward for ``W-1`` further
    chunks, delaying demotion to ``W - 1 + d_demote``. Scoring demotion on the
    chunk's own hits is what makes "within ``d_demote``" true as written.
    Promotion is sticky, demotion is prompt — which is also the only version in
    which the rich-get-richer negative bites.
    """
    d = 0
    c = int(chunk)
    while c >= 0 and int(hits_by_chunk[c]) < int(params.h_lo):
        d += 1
        c -= 1
    return d


def should_demote(hits_by_chunk: Sequence[int], chunk: int,
                  params: LifecycleParams) -> bool:
    """**L2**: PROTECTED -> ACTIVE within ``d_demote`` chunks of usage falling
    below ``h_lo``. ⛔ Never to trash."""
    if not params.demote:
        return False
    return demotion_dwell(hits_by_chunk, chunk, params) >= int(params.d_demote)


def should_trash(hits_by_stream: Mapping[int, int], first_seen_stream: int,
                 stream: int, params: LifecycleParams):
    """**L3** (§A20.6): is this well never-useful over ``k`` stream boundaries?

    Returns ``(decision, reason)``. ``hits_by_stream`` is the item-id-keyed
    read-hit count **aggregated per stream**; ``stream`` is the index of the
    stream that just ended.

    ⛔ **Depth is not an argument and never will be** (§A28.3(ii)); the signature
    is asserted depth-free in ``tests/test_store_lifecycle.py``.

    **The two readings, both implemented (the ambiguity is made mechanical, not
    resolved in prose).** The registered wording is *"never useful since first
    appearance over k stream boundaries"*, but the registered designed negative
    (a) is *"useful in stream 1 only => trashed at k"* — an item useful in stream
    1 is not never-useful-since-first-appearance, so the two are only jointly
    satisfiable under a trailing-window reading:

    ``last_k_streams`` (default)
        zero hits in **each of the last ``k`` streams**. Satisfies (a), (b) and
        (c); the never-useful-at-all well is the special case whose whole history
        is zero, and it is trashed the moment its age reaches ``k``.
    ``since_first_seen``
        zero hits in **every** stream since first appearance — the literal
        wording, under which designed negative (a) does **not** fire.
    """
    k = int(params.k_streams)
    s = int(stream)
    age = s - int(first_seen_stream)
    if params.censoring_guard and age < k:
        return False, (f"censoring guard: age {age} < k = {k} stream boundaries "
                       f"(never-useful-YET is not never-useful)")
    if str(params.trash_criterion) == "since_first_seen":
        window = range(int(first_seen_stream), s + 1)
    else:
        window = range(max(int(first_seen_stream), s - k + 1), s + 1)
    total = int(sum(int(hits_by_stream.get(t, 0)) for t in window))
    if total > 0:
        return False, (f"useful: {total} read hits over streams "
                       f"[{min(window)}..{max(window)}]")
    return True, (f"never useful over streams [{min(window)}..{max(window)}] "
                  f"(criterion={params.trash_criterion}, k={k}, age={age})")


# ==========================================================================
# L5 / L5-b — I1 refresh-monotonicity at the STORE level
# ==========================================================================
def refresh_factor(depth_before: float, depth_after: float,
                   params: Optional[LifecycleParams] = None,
                   max_gain: Optional[float] = None) -> float:
    """The I1 refresh factor in ``blocks.py``'s **amplitude** units.

    ``f = clip(sqrt(d_before / d_after), 1, refresh_max_gain)`` on a violating
    rewrite (``d_after < d_before``) and **exactly 1.0** otherwise — so a
    violation-free write is bit-identical to the unguarded one (I1-b).

    ⚠ Depth scales as the square of this factor. At the store level the factor is
    applied through :meth:`LearnedVStore.scale_group_amplitude`, whose argument is
    a **depth** factor, hence ``f**2`` there and ``f`` here. Stating the units is
    the whole point of L5-b: the two implementations must not silently diverge.
    """
    gain = float(params.refresh_max_gain if max_gain is None and params is not None
                 else (4.0 if max_gain is None else max_gain))
    db, da = float(depth_before), float(depth_after)
    if not (da < db):
        return 1.0
    if params is not None and not params.refresh_monotonic:
        return 1.0
    return float(min(math.sqrt(db / max(da, 1e-30)), gain))


def replay_rewrite_events(events: Sequence[Mapping[str, Any]],
                          params: LifecycleParams,
                          tol: float = 1e-6) -> Dict[str, Any]:
    """Replay recorded rewrite events through the **store-level** guard.

    This is L5-b's instrument: C2W6's ``p1_off`` / ``p1_on_i1_on`` cells recorded
    every rewrite event's ``depth_before`` / ``depth_after`` / ``refresh_factor``
    / ``violation`` through ``blocks.py``'s own code path, so the two
    implementations can be compared **on the same events** instead of on two
    different runs. ``n_flag_mismatch`` is E1, ``max_factor_dev`` is E2 and
    ``n_violations_post`` is E3 (PREREG §P1).
    """
    n = 0
    pre = post = mism = 0
    factors: List[float] = []
    guarded: List[float] = []
    dev = 0.0
    for e in events:
        if float(e.get("rewrite", 1.0)) <= 0.5:
            continue
        n += 1
        db = float(e["depth_before"])
        da = float(e["depth_after"])
        viol = bool(da < db)
        pre += int(viol)
        if "violation" in e:
            mism += int(bool(float(e["violation"]) > 0.5) != viol)
        f = refresh_factor(db, da, params)
        factors.append(f)
        if "refresh_factor" in e:
            theirs = float(e["refresh_factor"])
            dev = max(dev, abs(f - theirs) / max(abs(theirs), 1e-30))
        dg = da * f * f
        guarded.append(dg)
        post += int(dg < db * (1.0 - tol))
    return {
        "n_events": n,
        "n_violations_pre": pre,
        "n_violations_post": post,
        "rate_pre": (pre / n) if n else float("nan"),
        "rate_post": (post / n) if n else float("nan"),
        "n_flag_mismatch": mism,
        "max_factor_dev": dev,
        "factors": factors,
        "depth_guarded": guarded,
        "guard": bool(params.refresh_monotonic),
    }


def guarded_rewrite(system, item_id: int, address, payload, key,
                    params: LifecycleParams) -> Dict[str, Any]:
    """One **rewrite** of a live item, with the I1 guard applied at store level.

    A rewrite is a write into a well that already exists — which the admission
    gate refuses by construction (the proposal sits at distance 0 from itself),
    so it must be issued into the item's own slot. The depth is read off the
    store's own ``group_stats`` (the item's own atoms at its recorded site), the
    write runs through the system's masked learned write, and on a violating
    event the group's **depth** is scaled by ``f**2``.

    ⛔ ``chlu/core/clu_system.py`` is not edited by this wave: the write and the
    scaling are the system's own published operations, called from here.
    """
    slot = int(system._slot_of(int(item_id)))
    rec = None
    for r in system.controller.allocator.records.values():
        if int(r.item_id) == int(item_id):
            rec = r
            break
    if rec is None:  # pragma: no cover - _slot_of would already have raised
        raise KeyError(item_id)
    z = np.zeros((system.store.dim,), dtype=float)
    z[: system.store.addr_dim] = np.asarray(rec.center, dtype=float)[: system.store.addr_dim]
    d_before, _ = system.store.group_stats(slot, z[: system.store.addr_dim])
    loss = system._write_item(slot, np.asarray(address, dtype=float),
                              np.atleast_1d(np.asarray(payload, dtype=float)), key)
    d_after, _ = system.store.group_stats(slot, z[: system.store.addr_dim])
    viol = bool(d_after < d_before)
    f = refresh_factor(d_before, d_after, params)
    if params.refresh_monotonic and viol and f > 1.0:
        system.store = system.store.scale_group_amplitude(slot, float(f * f))
    d_guarded, _ = system.store.group_stats(slot, z[: system.store.addr_dim])
    return {
        "item_id": int(item_id), "slot": slot, "rewrite": 1.0,
        "depth_before": float(d_before), "depth_after": float(d_after),
        "depth_guarded": float(d_guarded), "refresh_factor": float(f),
        "violation": float(viol), "write_loss": float(loss),
        "guard": bool(params.refresh_monotonic),
    }


# ==========================================================================
# L6 — netting (Add.9 §A27.1: a BUILD REQUIREMENT, not an option)
# ==========================================================================
def cumulative_decay(controller) -> Dict[int, float]:
    """Cumulative **designed** decay per item, from the controller's own log.

    Thin, deliberate wrapper over
    :func:`chlu.core.well_lifecycle.designed_decay_factors` — that file belongs
    to a concurrent spoke and is **imported read-only, never reimplemented**
    (un-netted curves overstated recovery by up to 34 % on C2W6 seed 0 and by
    14-20 % on C2W8's census).
    """
    from chlu.core.well_lifecycle import designed_decay_factors

    return designed_decay_factors(controller)


def net_depth(raw: float, cum_factor: float) -> float:
    """Undo the designed decay: ``depth_netted = depth_raw / prod(factors)``.

    ⚠ ``cum_factor == 1.0`` returns ``raw`` **unchanged and bitwise** (leak = 0
    must not move a single bit — the L6 assertion is on the bit pattern, not on
    ``allclose``).
    """
    f = float(cum_factor)
    if f == 1.0:
        return float(raw)
    return float(raw) / max(f, 1e-300)


# ==========================================================================
# the stateful lifecycle
# ==========================================================================
@dataclass
class _Item:
    item_id: int
    state: str = ACTIVE
    first_seen_chunk: int = 0
    first_seen_stream: int = 0
    hits_by_chunk: List[int] = field(default_factory=list)
    hits_by_stream: Dict[int, int] = field(default_factory=dict)
    center: Optional[np.ndarray] = None
    depth_chunks: List[int] = field(default_factory=list)
    depth_raw: List[float] = field(default_factory=list)
    depth_netted: List[float] = field(default_factory=list)
    cum_factor: List[float] = field(default_factory=list)


class StoreLifecycle:
    """The three-state lifecycle over one store, driven at chunk granularity.

    Args:
        params: :class:`LifecycleParams` (``lifecycle=False`` => inert).
        budget: the **well budget** the protected fraction is a fraction of.
        controller: a :class:`chlu.core.controller.Controller` (the allocator).
            Promotion/demotion reach it through
            :meth:`~chlu.core.controller.Controller.set_permanence`; ``None``
            keeps the lifecycle a pure bookkeeping object (the unit-test path).
        trash_route: ``(item_id, center) -> None`` sink for L3. In the rig this
            is ``lambda iid, c: system.trash_route(c)``; ``None`` records the
            decision and reports the route as not taken.
    """

    def __init__(self, params: LifecycleParams, budget: int,
                 controller=None, trash_route: Optional[Callable] = None):
        self.params = params._assert_hysteresis_binds()
        self.budget = int(budget)
        self.controller = controller
        self._trash_route = trash_route
        self.items: Dict[int, _Item] = {}
        self.events: List[Dict[str, Any]] = []
        self.chunk = -1
        self.stream = 0
        self.n_promote_refused = 0
        self.stats = {"promote": 0, "demote": 0, "trash": 0, "promote_refused": 0,
                      "trash_route_missing": 0}

    # -- geometry of the bound ------------------------------------------
    @property
    def protected_cap(self) -> int:
        """``floor(f_max * budget)`` — evaluated BEFORE each promotion (L4)."""
        return int(math.floor(float(self.params.f_max) * self.budget))

    @property
    def n_protected(self) -> int:
        return sum(1 for it in self.items.values() if it.state == PROTECTED)

    def protected_fraction(self) -> float:
        return self.n_protected / max(self.budget, 1)

    def saturated(self) -> bool:
        return self.n_protected >= self.protected_cap

    # -- bookkeeping ------------------------------------------------------
    def note_admitted(self, item_id: int, *, chunk: int, stream: int,
                      center=None) -> None:
        iid = int(item_id)
        it = self.items.get(iid)
        if it is None:
            it = _Item(item_id=iid, first_seen_chunk=int(chunk),
                       first_seen_stream=int(stream))
            self.items[iid] = it
        if center is not None:
            it.center = np.asarray(center, dtype=float)

    def state(self, item_id: int) -> str:
        it = self.items.get(int(item_id))
        return ACTIVE if it is None else it.state

    def set_state(self, item_id: int, state: str) -> None:
        """Set a state directly (planting a designed negative's initial condition).

        ⚠ This is the **test/planting** seam, not a verb: it records no event and
        applies no policy. The verbs are :meth:`observe_chunk` and
        :meth:`end_stream`.
        """
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {state!r}")
        self.items[int(item_id)].state = state
        self._apply_permanence(int(item_id), state)

    def first_seen_stream(self, item_id: int) -> int:
        return int(self.items[int(item_id)].first_seen_stream)

    def hits_by_stream(self, item_id: int) -> Dict[int, int]:
        return dict(self.items[int(item_id)].hits_by_stream)

    def note_depth(self, item_id: int, *, chunk: int, depth_raw: float,
                   cum_factor: float) -> None:
        """Record one measurement point of an item's depth curve, **both forms**.

        Add.9 §A27.1 is a build requirement: there is no path through this class
        that stores a raw depth without its netted twin.
        """
        it = self.items[int(item_id)]
        it.depth_chunks.append(int(chunk))
        it.depth_raw.append(float(depth_raw))
        it.cum_factor.append(float(cum_factor))
        it.depth_netted.append(net_depth(float(depth_raw), float(cum_factor)))

    def depth_curves(self) -> Dict[int, Dict[str, List[float]]]:
        return {
            iid: {"chunk": list(it.depth_chunks), "depth_raw": list(it.depth_raw),
                  "depth_netted": list(it.depth_netted),
                  "cum_decay_factor": list(it.cum_factor)}
            for iid, it in self.items.items() if it.depth_chunks
        }

    # -- the verbs --------------------------------------------------------
    def observe_chunk(self, chunk: int, hits: Mapping[int, int],
                      stream: int = 0) -> List[Dict[str, Any]]:
        """Fold one chunk's read hits in, then run **promotion** and **demotion**.

        ``hits`` is item-id-keyed (never slot-keyed: a recycled slot is a
        different item). Items absent from ``hits`` scored zero this chunk.
        """
        self.chunk = int(chunk)
        self.stream = int(stream)
        for iid in dict(hits):
            self.note_admitted(int(iid), chunk=int(chunk), stream=int(stream))
        for iid, it in self.items.items():
            while len(it.hits_by_chunk) <= int(chunk):
                it.hits_by_chunk.append(0)
            n = int(dict(hits).get(iid, 0))
            it.hits_by_chunk[int(chunk)] += n
            if n:
                it.hits_by_stream[int(stream)] = it.hits_by_stream.get(int(stream), 0) + n
            it.hits_by_stream.setdefault(int(stream), it.hits_by_stream.get(int(stream), 0))
        out: List[Dict[str, Any]] = []
        if not self.params.lifecycle:
            return out
        for iid in sorted(self.items):
            it = self.items[iid]
            if it.state == TRASH:
                continue
            if it.state == ACTIVE and should_promote(it.hits_by_chunk, chunk, self.params):
                out.append(self._promote(iid))
            elif it.state == PROTECTED and should_demote(it.hits_by_chunk, chunk,
                                                         self.params):
                out.append(self._demote(iid))
        return [e for e in out if e is not None]

    def end_stream(self, stream: int) -> List[Dict[str, Any]]:
        """Run the **L3 trash sweep** at a stream boundary.

        ⛔ PROTECTED wells are out of the sweep's scope: trash is the never-useful
        route and a protected well is by definition one the store was told to
        keep. ⛔ A demotion never routes here (§A34.3).
        """
        out: List[Dict[str, Any]] = []
        if not (self.params.lifecycle and self.params.trash):
            return out
        for iid in sorted(self.items):
            it = self.items[iid]
            if it.state != ACTIVE:
                continue
            ok, why = should_trash(it.hits_by_stream, it.first_seen_stream,
                                   int(stream), self.params)
            if ok:
                out.append(self._trash(iid, why))
        return out

    # -- verb internals ---------------------------------------------------
    def _apply_permanence(self, item_id: int, state: str) -> None:
        if self.controller is None:
            return
        setter = getattr(self.controller, "set_permanence", None)
        if setter is None:  # pragma: no cover - the hook ships with this wave
            return
        if state == PROTECTED:
            setter(int(item_id), True, 0.0)
        elif state == ACTIVE:
            setter(int(item_id), False, float(self.params.leak))

    def _event(self, verb: str, item_id: int, frm: str, to: str, reason: str,
               **detail) -> Dict[str, Any]:
        row = {"verb": verb, "item_id": int(item_id), "chunk": int(self.chunk),
               "stream": int(self.stream), "from": frm, "to": to,
               "reason": reason, "n_protected": self.n_protected,
               "protected_cap": self.protected_cap, **detail}
        self.events.append(row)
        self.stats[verb] = self.stats.get(verb, 0) + 1
        return row

    def _promote(self, item_id: int) -> Optional[Dict[str, Any]]:
        # ⭐ L4: the bound is checked BEFORE the promotion, and a breach REFUSES
        # rather than partially protecting. It fails loudly (a named monitor row),
        # never as a loss term — the anti-collapse doctrine.
        if self.saturated():
            self.n_promote_refused += 1
            return self._event(
                "promote_refused", item_id, ACTIVE, ACTIVE,
                f"protected fraction bound f_max={self.params.f_max} reached "
                f"({self.n_protected}/{self.budget}); promotion refused",
                monitor=PROTECTED_SATURATION)
        it = self.items[item_id]
        it.state = PROTECTED
        self._apply_permanence(item_id, PROTECTED)
        return self._event("promote", item_id, ACTIVE, PROTECTED,
                           f"sustained usage: dwell >= d_dwell={self.params.d_dwell} "
                           f"at h_hi={self.params.h_hi} over window={self.params.window}")

    def _demote(self, item_id: int) -> Dict[str, Any]:
        it = self.items[item_id]
        it.state = ACTIVE          # ⛔ NEVER to trash (§A34.3)
        self._apply_permanence(item_id, ACTIVE)
        return self._event("demote", item_id, PROTECTED, ACTIVE,
                           f"usage below h_lo={self.params.h_lo} for "
                           f"d_demote={self.params.d_demote} chunks; re-exposed to "
                           f"the designed decay (leak={self.params.leak})")

    def _trash(self, item_id: int, why: str) -> Dict[str, Any]:
        it = self.items[item_id]
        routed = False
        if self._trash_route is not None:
            self._trash_route(int(item_id), it.center)
            routed = True
        else:
            self.stats["trash_route_missing"] += 1
        it.state = TRASH
        return self._event("trash", item_id, ACTIVE, TRASH, why, routed=routed)

    # -- reporting ---------------------------------------------------------
    def monitor_state(self) -> Dict[str, Any]:
        return {
            "name": PROTECTED_SATURATION,
            "tripped": bool(self.saturated() and self.n_promote_refused > 0),
            "saturated": bool(self.saturated()),
            "n_protected": self.n_protected,
            "protected_cap": self.protected_cap,
            "protected_fraction": self.protected_fraction(),
            "f_max": float(self.params.f_max),
            "n_promote_refused": int(self.n_promote_refused),
            "budget": int(self.budget),
        }

    def summary(self) -> Dict[str, Any]:
        states = {s: [iid for iid, it in self.items.items() if it.state == s]
                  for s in STATES}
        return {
            "n_items": len(self.items),
            "states": {s: len(v) for s, v in states.items()},
            "state_ids": states,
            "counts": dict(self.stats),
            "monitor": self.monitor_state(),
            "params": self.params.as_flag_table(),
            "hits_by_stream": {iid: dict(it.hits_by_stream)
                               for iid, it in self.items.items()},
            "first_seen_stream": {iid: int(it.first_seen_stream)
                                  for iid, it in self.items.items()},
        }


# ==========================================================================
# the monitor row (L4)
# ==========================================================================
class ProtectedSaturationMonitor:
    """⭐ The new named monitor row ``protected_saturation`` (L4).

    ⚠ **It fails loudly at runtime; it is not a loss term** — the anti-collapse
    doctrine. It trips when the protected population has reached
    ``floor(f_max * budget)`` *and* at least one promotion has been refused
    because of it, so "the bound is exactly saturated and nothing was ever turned
    away" is reported as **not tripped** rather than as a false alarm.

    ⚠ It is registered on the shared :class:`chlu.core.monitors.MonitorRegistry`
    through its public :meth:`~chlu.core.monitors.MonitorRegistry.register`
    method; the only edit this wave makes to ``monitors.py`` is one line in
    :data:`~chlu.core.monitors.SEVERITY` giving the row its trigger-ordering
    class.
    """

    name = PROTECTED_SATURATION
    mode = 8  # lifetimes/eviction economics (the row #8 family)

    def __init__(self, lifecycle: StoreLifecycle):
        self.lifecycle = lifecycle

    def observe(self, ctx):
        from chlu.core.monitors import MonitorReading

        st = self.lifecycle.monitor_state()
        return MonitorReading(
            name=self.name, mode=self.mode,
            value=float(st["protected_fraction"]),
            band=(f"protected fraction <= f_max = {st['f_max']:.3g} of the well "
                  f"budget ({st['protected_cap']}/{st['budget']} wells)"),
            tripped=bool(st["tripped"]), severity_class="II",
            lever="promotion (L1)", verb="refuse promotion",
            detail=st,
        )


__all__ = [
    "PROTECTED", "ACTIVE", "TRASH", "STATES", "TRASH_CRITERIA",
    "PROTECTED_SATURATION",
    "LifecycleParams", "StoreLifecycle", "ProtectedSaturationMonitor",
    "promotion_dwell", "demotion_dwell", "should_promote", "should_demote",
    "should_trash", "refresh_factor", "guarded_rewrite", "replay_rewrite_events",
    "cumulative_decay", "net_depth",
]
