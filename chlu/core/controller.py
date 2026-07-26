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
    ):
        if evict_policy not in ("staleness", "depth"):
            raise ValueError(f"evict_policy must be staleness|depth, got {evict_policy}")
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

        self.records: Dict[int, ItemRecord] = {}  # slot -> record (live only)
        self.t = 0
        self.stats = {
            "offered": 0,
            "admitted": 0,
            "relocated": 0,
            "refused_spacing": 0,
            "refused_full": 0,
            "evicted": 0,
            "decayed_out": 0,
        }
        self.log: List[dict] = []

    # -- introspection --------------------------------------------------------
    @property
    def n_live(self) -> int:
        return len(self.records)

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
        self.stats["offered"] += 1
        q2 = np.asarray(q_new, dtype=float).reshape(-1)[: self.addr_dim]
        stored = self.stored_addresses()

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

    def touch(self, item_id: int) -> None:
        """Mark an item as used *now* (staleness clock for LRU eviction)."""
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
        for slot in spent:
            self._evict(slot, reason="decay")
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
