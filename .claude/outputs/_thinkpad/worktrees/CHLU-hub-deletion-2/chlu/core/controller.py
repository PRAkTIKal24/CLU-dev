"""MVC-0 controller: the minimum viable, HAND-CODED write policy on a designed store.

This is the primitive-vision verbs — *decide, add, trash, evict* — made to exist,
with **no learning anywhere in the loop** (Head ruling, w20: differentiability is
not a first-paper requirement; the debt is stated in ``clu-controller-spec`` §4,
not hidden). The controller is a *certifier + allocator*, not an optimizer.

It wraps an :class:`~chlu.core.memory_potentials.AtomStorePotential` (the designed
store whose atom writes are C3-local by construction) and mechanises the three
decision rules of the controller spec (§3):

``admission`` (C5-A1/A2)
    Novelty test against existing wells using the ``d_safe`` packing geometry
    (:func:`chlu.core.admission.admit_site`, refuse-and-relocate). N74's lesson —
    choose the address geometry so admission is *decidable*: on the w20 ring the
    gate was arithmetically vacuous (spacing 1.4142 ≥ d_safe 1.10); on a disk of
    proposals with ``d_safe = 4.4·s`` it fires.

``placement`` (C1)
    The derived address is where the write's locality holds — the admitted or
    relocated site itself. The atom is written there; the writer records where it
    wrote (nothing is searched — Prop 5's dead cross-basin gradient is routed
    around, not re-funded).

``eviction / decay`` (C5 budget + §3.C trash)
    A budget policy: the store holds at most ``budget`` live items. When full, a
    new admissible item **evicts** by staleness (least-recently-used) or by
    current depth, unless the victim is flagged permanent (a full all-permanent
    store raises a *capacity alarm*, never a silent overwrite). Independently,
    **scheduled decay** shallows every leaky well by ``exp(-leak)`` per
    :meth:`tick`; a well below ``amp_floor`` self-evicts. Permanent wells
    (``leak == 0``) never decay — the per-item retention machinery of w22
    (permanent + leaky wells in one store).

What the controller CANNOT do (``clu-controller-spec`` §5, stated up front): it
cannot manufacture a τ=∞ item on an unconstrained *learned* V (permanence is a
designed flat coset, here ``leak == 0``); it cannot beat the packing bound (on a
fixed address space per-offered retention is capped at ``N_pack / K``); and its
spacing certificate is meaningless for a **global-support** learned write (N75) —
so this controller is exercised on the *designed* store only.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from chlu.core.admission import admit_site, min_separation
from chlu.core.memory_potentials import AtomStorePotential
from chlu.core.placement import CanonicalPlacer


@dataclass
class ItemRecord:
    """One codebook entry (MVC-0 §4 table, trimmed to what the store needs)."""

    item_id: int
    slot: int
    center: np.ndarray  # (2,) address-plane site actually written
    payload: float
    base_amp: float
    leak: float  # decay rate per tick; 0.0 == permanent
    permanent: bool
    born: int  # tick at write
    last_used: int  # tick of last touch (staleness)


class Controller:
    """Hand-coded MVC-0 controller over an :class:`AtomStorePotential`.

    Args:
        store: an (empty or partial) designed store; ``store.capacity`` must be
            ``>= budget``.
        d_safe: admission radius (``= d_safe_mult * store.s`` at the call site).
        budget: maximum number of simultaneously-live items. Defaults to the
            store capacity (eviction then fires only on a genuinely full store).
        amp: well depth for a freshly written item.
        leak: default per-tick decay rate for a non-permanent item (0 => the
            store never forgets on its own; decay is opt-in).
        amp_floor: a leaky well is evicted once its depth drops below this.
        evict_policy: ``"staleness"`` (LRU) or ``"depth"`` (shallowest first).
        n_candidates: relocation candidates drawn per admission attempt.
        allow_relocation: if ``False``, a proposal that fails the spacing gate is
            **refused outright** — relocation is never attempted. Default ``True``
            (the w23 behaviour). ⚠ The w25 continual-learning entry sets this
            ``False``: there the address IS the content (``φ(x)``), so moving an
            item to a free site would store it under an address no query can
            reach. Refusal is the only legal admission outcome when the address is
            derived from the item.
        peer_addresses_fn: optional ``() -> (m, addr_dim)`` callable returning
            addresses held by OTHER shards. When given, the admission spacing test
            runs against the **union** of this controller's addresses and its
            peers' — i.e. the controller becomes one node of a *global address
            allocator* (:class:`chlu.core.shard_store.ShardedRegistry`, w25 build
            item 3). This is the only global object in a sharded store, and it is
            a **registry, not an optimizer**: no gradient and no optimizer state
            crosses a shard boundary, only the list of where things were written.
            ``None`` (default) => single-store behaviour, unchanged.
        placement: ``"relocate"`` (default, **unchanged w23 behaviour**) or
            ``"canonical"``. Under ``"canonical"`` the allocator is
            :class:`~chlu.core.placement.CanonicalPlacer` (PGCP): items live on a hex
            lattice of spacing ``d_safe`` inside a disk of radius ``lattice_radius``,
            each taking the first cell of its own probe order not claimed by a
            higher-priority item. Placement is then a **set function** of the live
            records — no arrival order, no relocation draws — which is what makes
            :meth:`delete` exact (Theorem 2) and closes the allocator-trace membership
            channel that refuse-and-relocate leaks at ``AUC 0.99985``. Admission becomes
            "did it get a cell"; the spacing gate is a lattice *invariant* (Theorem 3),
            not a per-write test.
        waitlist: ⭐ **the P2 waitlist (w27)**, canonical placement only. When an offer
            cannot be seated (the lattice is full of higher-priority keys) the record is
            kept in a side dict instead of being forgotten, and the *first later op that
            frees a reachable cell re-seats the highest-priority waiting key*. This is what
            extends exact deletion from "below capacity" to **any load**: without it, a
            background item refused while the target was resident does not counterfactually
            return when the target is deleted, and the post-delete store is NOT the store
            that never held the target (measured at 8 offers into a 7-cell lattice:
            ``AUC(n_live) = 1.000``, byte-equality 0/3072). ``False`` restores the w26
            rung-P1 behaviour. Ignored under ``placement="relocate"``.
        lattice_radius: address-disk radius the canonical lattice is clipped to
            (**required** when ``placement="canonical"``, ignored otherwise). The hard
            admission capacity is the resulting cell count, not the packing bound —
            ``chlu.core.placement.n_cells_for`` reports it, and the sizing rule
            ``R(K)·1.05`` gives ``n_cells >= K`` for K in {16,32,64,128}.

    The controller owns its store: every method returns nothing and mutates
    ``self.store`` (a new frozen PyTree each time) and ``self.records``.
    """

    def __init__(
        self,
        store: AtomStorePotential,
        d_safe: float,
        budget: Optional[int] = None,
        amp: float = 1.0,
        leak: float = 0.0,
        amp_floor: float = 0.05,
        evict_policy: str = "staleness",
        n_candidates: int = 400,
        allow_relocation: bool = True,
        peer_addresses_fn: Optional[Callable] = None,
        placement: str = "relocate",
        lattice_radius: Optional[float] = None,
        waitlist: bool = True,
    ):
        if evict_policy not in ("staleness", "depth"):
            raise ValueError(f"evict_policy must be staleness|depth, got {evict_policy}")
        if placement not in ("relocate", "canonical"):
            raise ValueError(f"placement must be relocate|canonical, got {placement}")
        self.placement = placement
        self.waitlist = bool(waitlist) and placement == "canonical"
        self.placer: Optional[CanonicalPlacer] = None
        if placement == "canonical":
            # --- the guards the deletion claim's scope requires (theorist §4c) ---
            if evict_policy == "staleness":
                raise ValueError(
                    "placement='canonical' forbids evict_policy='staleness': LRU is "
                    "intrinsically history-dependent (last_used is query history), so a "
                    "store that can LRU-evict is NOT order-independent and the exact "
                    "store-level deletion claim does not cover it. Use "
                    "evict_policy='depth' (amp = base*exp(-leak*age) is item-intrinsic)."
                )
            if lattice_radius is None:
                raise ValueError(
                    "placement='canonical' requires lattice_radius (the address-disk "
                    "radius the canonical hex lattice is clipped to)"
                )
            if not allow_relocation:
                raise ValueError(
                    "placement='canonical' is incompatible with allow_relocation=False: "
                    "canonical placement quantizes every address onto the lattice (up to "
                    "the covering radius d_safe/sqrt(3)), so an address-IS-content store "
                    "(q = phi(x)) must not use it until that quantization cost is measured."
                )
            if peer_addresses_fn is not None:
                raise ValueError(
                    "placement='canonical' does not support peer_addresses_fn: a sharded "
                    "canonical allocator needs one global lattice, which is not built."
                )
        self.store = store
        self.addr_dim = int(getattr(store, "addr_dim", 2))
        self.allow_relocation = bool(allow_relocation)
        self.d_safe = float(d_safe)
        self.budget = int(store.capacity if budget is None else budget)
        if self.budget > store.capacity:
            raise ValueError(f"budget {self.budget} exceeds capacity {store.capacity}")
        self.amp = float(amp)
        self.leak = float(leak)
        self.amp_floor = float(amp_floor)
        self.evict_policy = evict_policy
        self.n_candidates = int(n_candidates)
        self.peer_addresses_fn = peer_addresses_fn
        if placement == "canonical":
            if self.addr_dim != 2:
                raise ValueError(
                    f"placement='canonical' needs a 2-D address plane (the lattice is "
                    f"planar), got addr_dim={self.addr_dim}"
                )
            self.placer = CanonicalPlacer(
                float(lattice_radius), self.d_safe, waitlist=self.waitlist
            )

        self.records: Dict[int, ItemRecord] = {}  # slot -> record (live only)
        # P2: item_id -> record of an offer the lattice could not seat. It is NOT in the
        # store (it contributes nothing to V) but it is still in the *offered* set, so a
        # freed cell re-seats it by priority. Empty unless placement="canonical" +
        # waitlist=True. `waiting_amps` carries each waiting record's depth through the
        # SAME per-tick arithmetic the live wells get (Theorem 4: amplitudes factorize per
        # record), so a re-seated item lands bit-identically on the depth the history that
        # never refused it would have produced.
        self.waiting: Dict[int, ItemRecord] = {}
        self.waiting_amps: Dict[int, float] = {}
        #: ⭐ C2W8 B2: item-id-keyed read counter, written by :meth:`touch` and
        #: never cleared on eviction. Decision-free here (the prune verb reads it
        #: through :mod:`chlu.experiments.usage_telemetry`, not the allocator).
        self.read_hits: Dict[int, int] = {}
        self.t = 0
        self.stats = {
            "offered": 0,
            "admitted": 0,
            "relocated": 0,
            "refused_spacing": 0,
            "refused_full": 0,
            "evicted": 0,
            "decayed_out": 0,
            "deleted": 0,
            "moves": 0,
            "waitlisted": 0,
            "reseated": 0,
        }
        self.log: List[dict] = []

    # -- introspection --------------------------------------------------------
    @property
    def n_live(self) -> int:
        return len(self.records)

    @property
    def n_waiting(self) -> int:
        """Offers the lattice could not seat, still in the offered set (P2 waitlist)."""
        return len(self.waiting)

    def live_slots(self) -> List[int]:
        return sorted(self.records)

    def stored_addresses(self) -> np.ndarray:
        """(n_live, addr_dim) address-space sites currently live."""
        if not self.records:
            return np.zeros((0, self.addr_dim))
        return np.stack([self.records[s].center for s in self.live_slots()])

    def live_items(self):
        """(ids, centers, payloads) over the currently-live items, id-sorted."""
        recs = sorted(self.records.values(), key=lambda r: r.item_id)
        ids = [r.item_id for r in recs]
        centers = (
            np.stack([r.center for r in recs])
            if recs
            else np.zeros((0, self.addr_dim))
        )
        pays = np.array([r.payload for r in recs], dtype=float)
        return ids, centers, pays

    def live_amps(self) -> np.ndarray:
        """(n_live,) current well depths, id-sorted (matches :meth:`live_items`)."""
        recs = sorted(self.records.values(), key=lambda r: r.item_id)
        amps = np.asarray(self.store.amps, dtype=float)
        return np.array([amps[r.slot] for r in recs], dtype=float)

    def evict_item(self, item_id: int, reason: str = "policy") -> bool:
        """Public eviction verb: remove one live item by id. Returns whether it
        was found. The w25 entry uses this for its class-balanced budget policy
        (evict the least-recently-used item of the most-represented class), which
        is a *policy* on top of the controller, not a change to its rules."""
        for slot, r in list(self.records.items()):
            if r.item_id == item_id:
                self._evict(slot, reason="budget" if reason == "policy" else reason)
                return True
        return False

    # -- the three decisions --------------------------------------------------
    def offer(
        self,
        item_id: int,
        q_new,
        payload: float,
        key=None,
        proposer: Optional[Callable] = None,
        permanent: bool = False,
        leak: Optional[float] = None,
    ) -> dict:
        """Offer one item to the store. Runs admission -> (evict) -> placement.

        Returns a decision dict; the item is committed iff
        ``decision in {"admit", "relocate"}`` and no capacity alarm fired.
        A refusal is a *correct* controller output, reported and never retried.
        """
        if self.placement == "canonical":
            return self._offer_canonical(item_id, q_new, payload, permanent, leak)
        self.stats["offered"] += 1
        q2 = np.asarray(q_new, dtype=float).reshape(-1)[: self.addr_dim]
        stored = self.stored_addresses()
        if self.peer_addresses_fn is not None:
            # GLOBAL allocation: the spacing test is run against the UNION of every
            # shard's live addresses (w25 build item 3). Peers are *read only* —
            # this controller never writes into, nor optimizes, another shard.
            peers = np.asarray(self.peer_addresses_fn(), dtype=float).reshape(
                -1, self.addr_dim
            )
            if peers.size:
                stored = np.concatenate([stored, peers], axis=0)

        # 1. ADMISSION — spacing gate with refuse-and-relocate (C5-A1/A2)
        if self.allow_relocation:
            dec = admit_site(
                q2, stored, self.d_safe, key=key, proposer=proposer,
                n_candidates=self.n_candidates,
            )
        else:
            dec = self._admit_no_relocate(q2, stored)
        if dec["decision"] == "refuse":
            self.stats["refused_spacing"] += 1
            row = self._row(item_id, "refuse_spacing", dec, None)
            self.log.append(row)
            return row
        site = np.asarray(dec["site"], dtype=float)[: self.addr_dim]

        # 2. BUDGET / EVICTION — make room if full (C5 budget, §3.C trash)
        evicted_id = None
        if self.n_live >= self.budget:
            victim = self._pick_victim()
            if victim is None:
                # every live item is permanent: a capacity alarm, NOT a silent
                # overwrite. The controller declines and reports.
                self.stats["refused_full"] += 1
                row = self._row(item_id, "refuse_full", dec, None)
                self.log.append(row)
                return row
            evicted_id = self.records[victim].item_id
            self._evict(victim, reason="budget")

        # 3. PLACEMENT — write the atom at the derived (admitted) address (C1)
        self.store = self.store.with_item(site, float(payload), amp=self.amp)
        slot = self._last_slot()
        # permanent <=> leak 0 (a flat coset; clu-controller-spec Prop C-N): a
        # permanent item is NEVER decayed, whatever the controller's default leak.
        item_leak = 0.0 if permanent else (self.leak if leak is None else float(leak))
        self.records[slot] = ItemRecord(
            item_id=int(item_id), slot=slot, center=site, payload=float(payload),
            base_amp=self.amp, leak=item_leak,
            permanent=bool(permanent), born=self.t, last_used=self.t,
        )
        self.stats["admitted"] += 1
        if dec["decision"] == "relocate":
            self.stats["relocated"] += 1
        row = self._row(item_id, dec["decision"], dec, slot, evicted_id=evicted_id)
        self.log.append(row)
        return row

    def set_permanence(self, item_id: int, permanent: bool,
                       leak: Optional[float] = None) -> bool:
        """⭐ **C2W10 (L1/L2): the promotion/demotion hook.** Returns whether the
        item was found.

        The three-state lifecycle's only reach into the allocator: PROTECTED is
        the existing permanent flag (``leak == 0``, the flat coset of
        ``clu-controller-spec`` Prop C-N) and demotion re-exposes the item to the
        designed decay by clearing it. Permanence and ``leak == 0`` are kept in
        lockstep here exactly as :meth:`offer` establishes them, so a promoted
        item is skipped by :meth:`tick` and by :meth:`_pick_victim` for the same
        reason a born-permanent one is.

        ⛔ This is a **setter, not a policy**: which item to promote or demote is
        decided in :mod:`chlu.core.store_lifecycle`, from item-id-keyed read
        hits, and never from depth (§A28.3(ii)). The LRU/staleness semantics are
        untouched — ``last_used`` is not written here — so the pytest-pinned
        eviction behaviour is unchanged for any item the lifecycle never touches.
        """
        for r in self.records.values():
            if r.item_id == int(item_id):
                r.permanent = bool(permanent)
                r.leak = 0.0 if permanent else (
                    self.leak if leak is None else float(leak))
                return True
        return False

    def touch(self, item_id: int) -> None:
        """Mark an item as used *now* (staleness clock for LRU eviction).

        ⭐ **C2W8 (B2, usage telemetry).** The touch path is also the *read*
        counter: every touch increments :attr:`read_hits` ``[item_id]``, which is
        keyed by **item id, never by slot** (slot != well: a recycled slot is a
        different item) and **survives eviction**, so "this item was never read
        since it first appeared" stays computable after the well is gone. The
        counter is pure bookkeeping — no decision in this class reads it — so the
        staleness/LRU behaviour is bit-identical to the pre-C2W8 path.
        """
        self.read_hits[int(item_id)] = self.read_hits.get(int(item_id), 0) + 1
        for r in self.records.values():
            if r.item_id == item_id:
                r.last_used = self.t
                return

    def tick(self) -> None:
        """Advance one time step: shallow every leaky well; self-evict spent ones.

        Permanent (``leak == 0``) wells are untouched. This is the scheduled-decay
        half of the budget policy — the per-item retention machinery of w22, run
        as a physical amplitude decay rather than a bookkeeping delete.
        """
        self.t += 1
        # a waitlisted leaky record ages on its own clock (Theorem 4): once the depth it
        # WOULD have had falls below the floor, the never-refused history had already
        # self-evicted it, so it leaves the offered set too.
        for item_id, r in list(self.waiting.items()):
            if r.leak <= 0.0:
                continue
            a = self.waiting_amps[item_id] * float(np.exp(-r.leak))   # same arithmetic
            if a < self.amp_floor:                                    # as the live wells
                self.waiting.pop(item_id)
                self.waiting_amps.pop(item_id)
                self.placer.delete(item_id)
                self.stats["decayed_out"] += 1
            else:
                self.waiting_amps[item_id] = self._amp_cast(a)
        if not self.records:
            return
        amps = np.asarray(self.store.amps, dtype=float).copy()
        spent = []
        for slot, r in self.records.items():
            if r.leak > 0.0:
                amps[slot] *= float(np.exp(-r.leak))
                if amps[slot] < self.amp_floor:
                    spent.append(slot)
        self.store = self.store.with_amps(amps)
        # evict by ITEM ID, not by slot: under canonical placement a removal re-packs the
        # slots (the fix-up cascade), so a pre-computed slot list goes stale after the
        # first eviction. Under "relocate" slots are stable and this is a no-op change.
        for item_id in [self.records[s].item_id for s in spent]:
            r = self._record_for_id(item_id)
            if r is not None:
                self._evict(r.slot, reason="decay")
                self.stats["decayed_out"] += 1

    def _admit_no_relocate(self, q_new, stored) -> dict:
        """Spacing gate WITHOUT relocation (``allow_relocation=False``).

        Same decision rule as :func:`chlu.core.admission.admit_site`'s first test,
        with the same return schema, but a failed proposal is refused rather than
        moved — the only legal behaviour when the address is derived from the item
        itself (``q = φ(x)``) rather than allocated by the controller.
        """
        d_min = min_separation(q_new, stored)
        ok = d_min >= self.d_safe
        return {
            "decision": "admit" if ok else "refuse",
            "site": np.asarray(q_new, dtype=float) if ok else None,
            "d_min_proposed": d_min,
            "d_min_written": d_min if ok else float("nan"),
            "n_candidates_examined": 0,
        }

    # -- canonical placement (PGCP) ------------------------------------------
    def _record_for_id(self, item_id: int) -> Optional[ItemRecord]:
        for r in self.records.values():
            if r.item_id == int(item_id):
                return r
        return None

    def _amp_cast(self, x: float) -> float:
        """Round a waitlisted amplitude through the STORE's dtype.

        A live well's depth is rounded to the store array's dtype on every
        :meth:`tick` (``with_amps`` casts); a waiting record must follow the *same*
        rounding path or a re-seated item lands one ULP away from the never-refused
        history and byte-identity fails. Reading the dtype from the store (rather than
        hard-coding float32) keeps that true under ``jax_enable_x64``.
        """
        return float(np.asarray(x, dtype=np.asarray(self.store.amps).dtype))

    def _empty_store(self) -> AtomStorePotential:
        s = self.store
        return AtomStorePotential(
            dim=s.dim, capacity=s.capacity, alpha=s.alpha, s=s.s, s_pay=s.s_pay,
            kappa=s.kappa, spectator_k=s.spectator_k, addr_dim=s.addr_dim,
            payload_gate=s.payload_gate, payload_g0=s.payload_g0,
            payload_eps=s.payload_eps,
        )

    def _canonical_sync(
        self, new_record: Optional[ItemRecord] = None, new_amp: Optional[float] = None
    ) -> None:
        """Rewrite the store so its slots follow the placer's canonical layout.

        This is what makes ``Store(S)`` a *bit-identical* set function and not merely a
        set of atoms: the slot order is the canonical (descending-priority) order, so two
        histories reaching the same live set produce byte-equal ``centers/payloads/amps/
        active`` arrays (theorist H7, verified on this very PyTree). It is O(n)
        ``with_item`` calls per op — the cheap in-place slot move is an optimization, not
        a requirement, at MVC-0 sizes.

        Amplitudes travel with the *record*, not the slot (Theorem 4: decay factorizes
        per item), so decay and placement commute across a rebuild. ``new_record`` is the
        one record of an insert that has no slot yet; it enters at depth ``new_amp``.
        """
        amps = np.asarray(self.store.amps, dtype=float)
        recs = list(self.records.values())
        amp_by_id = {r.item_id: float(amps[r.slot]) for r in recs}
        if new_record is not None:
            recs.append(new_record)
            amp_by_id[new_record.item_id] = float(new_amp)
        if self.waitlist:
            # ⭐ P2, both directions. A record whose key lost its cell to a
            # higher-priority key goes to the waitlist (offered, not stored); a waiting
            # record whose key just took a freed cell re-enters at the amplitude the
            # never-refused history would have given it — amplitudes are item-intrinsic
            # (Theorem 4: base_amp * exp(-leak * age)), so this is still a set function.
            placed = set(self.placer.placed_keys())
            for r in list(recs):
                if r.item_id not in placed:
                    recs.remove(r)
                    self.waiting[r.item_id] = r
                    self.waiting_amps[r.item_id] = amp_by_id.pop(
                        r.item_id, self._amp_cast(self.amp))
                    self.stats["waitlisted"] += 1
            for key in placed:
                if key in self.waiting:
                    recs.append(self.waiting.pop(key))
                    amp_by_id[key] = self.waiting_amps.pop(key)
                    self.stats["reseated"] += 1
        by_id = {r.item_id: r for r in recs}
        store = self._empty_store()
        new_records: Dict[int, ItemRecord] = {}
        for slot, (key, center) in enumerate(self.placer.layout()):
            r = by_id[int(key)]
            store = store.with_item(center, r.payload, amp=amp_by_id[r.item_id])
            r.slot = slot
            r.center = np.asarray(center, dtype=float)
            new_records[slot] = r
        self.store = store
        self.records = new_records

    def _offer_canonical(self, item_id, q_new, payload, permanent, leak) -> dict:
        """``offer`` under ``placement="canonical"``: admission == "it got a cell".

        No spacing test is run per write — the lattice guarantees ``>= d_safe`` by
        construction (Theorem 3). The item's offered address is its *anchor*: it takes the
        nearest cell of the lattice that no higher-priority live item has claimed, and
        higher-priority items are never disturbed. A write can displace lower-priority
        items (the cascade), and on a full lattice the lowest-priority key loses its cell
        — priority eviction, a set-function policy (theorist §4b). ⭐ Under
        ``waitlist=True`` (w27, rung P2) a key that loses its cell — or never gets one —
        is **waitlisted, not forgotten**: it stays in the offered set and is re-seated by
        the first op that frees a reachable cell, which is what makes deletion exact under
        overflow as well as below capacity.
        """
        self.stats["offered"] += 1
        if int(item_id) in self.waiting:
            raise ValueError(
                f"item_id {item_id} is already offered (waitlisted); canonical placement "
                f"keys the lattice by item_id, so ids must be unique among offered items"
            )
        if self._record_for_id(item_id) is not None:
            raise ValueError(
                f"item_id {item_id} is already live; canonical placement keys the "
                f"lattice by item_id, so ids must be unique among live items"
            )
        q2 = np.asarray(q_new, dtype=float).reshape(-1)[: self.addr_dim]
        stored = self.stored_addresses()
        d0 = min_separation(q2, stored)

        # 1. BUDGET — make room first, so the lattice sees the true live set
        evicted_id = None
        if self.n_live >= self.budget:
            victim = self._pick_victim()
            if victim is None:
                self.stats["refused_full"] += 1
                row = self._row(item_id, "refuse_full", self._dec(d0), None)
                self.log.append(row)
                return row
            evicted_id = self.records[victim].item_id
            self._evict(victim, reason="budget")

        # 2. PLACEMENT == ADMISSION (Theorems 1/3)
        placed = self.placer.insert(item_id, q2)
        moves = list(self.placer.moves_last_op)
        dropped = [int(k) for k in self.placer.dropped_last_op if int(k) != int(item_id)]
        if not self.waitlist:
            for key in dropped:  # priority eviction under overflow (rung P1: forgotten)
                r = self._record_for_id(key)
                if r is not None:
                    self.records.pop(r.slot, None)
                    self.stats["evicted"] += 1
        item_leak = 0.0 if permanent else (self.leak if leak is None else float(leak))
        rec = ItemRecord(
            item_id=int(item_id), slot=-1, center=q2.copy(), payload=float(payload),
            base_amp=self.amp, leak=item_leak, permanent=bool(permanent),
            born=self.t, last_used=self.t,
        )
        if not placed:
            self.stats["refused_spacing"] += 1
            if self.waitlist:
                # P2: offered but unseated — remembered, so a later delete re-seats it
                self.waiting[int(item_id)] = rec
                self.waiting_amps[int(item_id)] = self._amp_cast(self.amp)
                self.stats["waitlisted"] += 1
            if moves or dropped or self.waitlist:
                self._canonical_sync()
            row = self._row(item_id, "refuse_spacing", self._dec(d0), None)
            row["waitlisted"] = bool(self.waitlist)
            self.log.append(row)
            return row

        rec.center = self.placer.center_of(item_id)
        self._canonical_sync(new_record=rec, new_amp=self.amp)

        self.stats["admitted"] += 1
        self.stats["moves"] += len(moves)
        probe = int(
            np.where(self.placer.probe_order(item_id) == self.placer.cell_of(item_id))[0][0]
        )
        if probe > 0:
            self.stats["relocated"] += 1
        others = np.stack(
            [r.center for r in self.records.values() if r.item_id != int(item_id)]
        ) if self.n_live > 1 else np.zeros((0, self.addr_dim))
        dec = self._dec(
            d0, written=True, probe=probe,
            d_written=min_separation(rec.center, others),
            quant=float(np.linalg.norm(rec.center - q2[:2])),
        )
        row = self._row(item_id, "admit" if probe == 0 else "relocate", dec, rec.slot,
                        evicted_id=evicted_id)
        row["moves"] = len(moves)
        self.log.append(row)
        return row

    def delete(self, item_id: int) -> dict:
        """⭐ **Exact store-level deletion (scoped)** — Theorem 2. Canonical placement only.

        Removes the item and restores canonical placement over the survivors, so the
        resulting store is **bit-identical** to the store that holds exactly the remaining
        records and never held this one — including each survivor's scheduled decay and
        permanence (Theorem 4: deletion and decay commute). Any interleaving of writes and
        deletes reaching the same live set yields the same store.

        The price, stated: deleting an item legitimately *moves* lower-priority survivors
        (mean ~2.8 moves/delete at full lattice load, 0.2 at half load) — to exactly where
        the never-written store would have placed them. Reads must therefore use the
        record's current ``center``, which the controller updates here.

        ⛔ Scope: the **store** only, and only under set-function eviction. With
        ``waitlist=True`` (default) the claim no longer needs the "below capacity"
        qualifier: a key refused because the lattice was full **does** return when
        something is deleted, so the post-delete store equals the never-held-it store at
        any load. With ``waitlist=False`` the below-capacity qualifier is load-bearing.
        This is not unlearning: the encoder and any learned-landscape residue are separate
        channels. No ``(eps, delta)`` claim.

        Deleting a **waitlisted** item (offered, never seated) is legal and is a no-op on
        the store — which is exactly the counterfactual: the store never held it.

        Returns a log row with the number of survivor moves; raises ``KeyError`` if the
        item is neither live nor waiting.
        """
        if self.placement != "canonical":
            raise ValueError(
                "Controller.delete requires placement='canonical'. Under "
                "refuse-and-relocate, placement is history-dependent: removing an item "
                "does NOT reproduce the store that never held it (the allocator trace "
                "alone identifies membership at AUC 0.99985), so there is no exact "
                "deletion verb to offer. Use evict_item() for the non-exact removal."
            )
        r = self._record_for_id(item_id)
        waiting = int(item_id) in self.waiting
        if r is None and not waiting:
            raise KeyError(f"item_id {item_id} is neither live nor waiting")
        if waiting:
            # offered but never seated: drop it from the offered set. The placer's greedy
            # over the lower-priority suffix cannot change (the key held no cell), so the
            # store is untouched — which IS the never-offered counterfactual.
            self.waiting.pop(int(item_id))
            self.waiting_amps.pop(int(item_id), None)
            self.placer.delete(int(item_id))
            self._canonical_sync()
        else:
            self._evict(r.slot, reason="delete")
        n_moves = len(self.placer.moves_last_op)
        self.stats["deleted"] += 1
        row = {
            "t": self.t, "item_id": int(item_id), "decision": "delete",
            "slot": None, "moves": n_moves, "n_live": self.n_live,
            "was_waiting": bool(waiting),
        }
        self.log.append(row)
        return row

    def _dec(self, d0, written: bool = False, probe: int = 0,
             d_written: float = float("nan"), quant: float = 0.0) -> dict:
        """Decision record in :func:`admit_site`'s schema, for the canonical path."""
        return {
            "decision": "admit" if written else "refuse",
            "site": None,
            "d_min_proposed": d0,
            "d_min_written": d_written,
            "n_candidates_examined": probe,
            "probe_index": probe,
            "quantization": quant,
        }

    # -- eviction internals ---------------------------------------------------
    def _pick_victim(self) -> Optional[int]:
        cand = [s for s, r in self.records.items() if not r.permanent]
        if not cand:
            return None
        if self.evict_policy == "staleness":
            return min(cand, key=lambda s: (self.records[s].last_used, self.records[s].born))
        # "depth": evict the currently-shallowest well
        amps = np.asarray(self.store.amps, dtype=float)
        return min(cand, key=lambda s: amps[s])

    def _evict(self, slot: int, reason: str) -> None:
        if self.placement == "canonical":
            # under canonical placement removal must go through the placer, or the
            # invariant ("slots ARE the canonical layout") breaks: the survivors' cells
            # and slots both change (Theorem 2's fix-up cascade).
            r = self.records.pop(slot, None)
            if r is not None:
                self.placer.delete(r.item_id)
                self.stats["moves"] += len(self.placer.moves_last_op)
                self._canonical_sync()
        else:
            self.store = self.store.evict(slot)
            self.records.pop(slot, None)
        if reason == "budget":
            self.stats["evicted"] += 1

    def _last_slot(self) -> int:
        """Slot :meth:`AtomStorePotential.with_item` just filled (highest active,
        lowest free index — argmin over the *pre-write* mask)."""
        active = np.asarray(self.store.active)
        used = set(self.records)
        free = [i for i in range(self.store.capacity) if active[i] == 1.0 and i not in used]
        return int(free[0]) if free else int(np.argmax(active))

    def _row(self, item_id, decision, dec, slot, evicted_id=None) -> dict:
        return {
            "t": self.t,
            "item_id": int(item_id),
            "decision": decision,
            "slot": (None if slot is None else int(slot)),
            "d_min_proposed": float(dec["d_min_proposed"]),
            "d_min_written": float(dec["d_min_written"]),
            "n_candidates_examined": int(dec["n_candidates_examined"]),
            "evicted_item": (None if evicted_id is None else int(evicted_id)),
            "n_live": self.n_live,
        }


# ---------------------------------------------------------------------------
# Geometry helpers (the packing bound the admission fraction is checked against)
# ---------------------------------------------------------------------------
def packing_bound_disk(radius: float, d_safe: float) -> float:
    """Hex/farthest-point packing bound for a disk (N74's form).

    ``N_pack = π R² / (√3/2 · d_safe²)`` — the number of points that fit in a disk
    of radius ``R`` with pairwise separation ``>= d_safe``. N74 measured
    ``6.0 ± 0.9`` against this bound's ``6.12`` at ``R=2, d_safe=1.54``.
    """
    return float(np.pi * radius**2 / ((np.sqrt(3.0) / 2.0) * d_safe**2))


def radius_for_capacity(k: int, d_safe: float) -> float:
    """Disk radius whose packing bound is ``>= k`` (invert :func:`packing_bound_disk`).

    ``R = d_safe · sqrt((√3/2) · k / π) ≈ 0.808 · d_safe · sqrt(k)``. Used by the
    "sized" geometry arm: give the address space enough room for the load and the
    controller no longer has to abstain (theorist A4 — count never binds).
    """
    return float(d_safe * np.sqrt((np.sqrt(3.0) / 2.0) * k / np.pi))
