"""PGCP — priority-greedy canonical placement (w26; theorist ``order-independent-placement``).

The shipped MVC-0 allocator (``refuse-and-relocate``, :mod:`chlu.core.admission`) is
**history-dependent**: where an item lands depends on which items were offered before it
and on the relocation draws, so removing an item cannot reproduce the store that never
held it. The measured price of that is not a technicality — post-eviction, the allocator
trace alone identifies membership at ``AUC 0.99985`` (``mia-decay-measurement`` §2).

This module implements the alternative rule, proven in ``order-independent-placement`` §2:

``Store(S)`` **is a set function.** Fix the geometry ``G`` (address disk of radius ``R``, a
hex lattice ``Λ`` of cells spaced ``d_safe`` apart, anchored at the origin). Give every
item a deterministic priority ``prio(κ)`` (splitmix64 of its key) and a probe order
``π(κ)`` = all cells sorted by ``(|c − g(κ)|, cell index)``, where ``g(κ)`` is the item's
*anchor* — its offered address if it has one (content-addressed store), else a
deterministic hash point. Canonical placement processes the live keys in **descending
priority**; each takes the first cell of its own probe order not already taken by a
higher-priority key. Nothing in that definition mentions arrival order.

Consequences (Theorems 1–4 of the theorist report, ported here as executable code):

* **T1** ``pos_S`` depends only on ``S`` and ``G`` — writes commute.
* **T2** deletion is exact: remove the atom and re-run the greedy over the keys of *lower*
  priority (the bounded fix-up cascade); the result equals ``Store(S∖{i})`` bit for bit.
  Keys of higher priority provably never move (suffix-stability lemma).
* **T3** the spacing certificate is free: distinct lattice cells are ``>= d_safe`` apart by
  construction, so the admission gate is a lattice *invariant*, not a per-write test.
* **T4** decay/permanence commute with deletion (amplitudes factorize per record), so this
  module never needs to know about amplitudes at all.

⚠ **Scope.** Exactness is claimed **below capacity** (``|S| <= n_cells``) or under
set-function eviction (priority / item-intrinsic attribute). Under overflow this
implementation drops the lowest-priority key *and forgets it* (rung P1): a key refused
because the store was full does not counterfactually return when something is deleted.
Recency (LRU) eviction is intrinsically historical and is excluded — the controller
hard-errors on that combination.

Pure numpy (float64), no JAX: this is bookkeeping, not physics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

M64 = (1 << 64) - 1
C_G1, C_G2, C_PRIO = 0xA5A5A5A5DEADBEEF, 0x0123456789ABCDEF, 0xFEDCBA9876543210


# ---------------------------------------------------------------------------
# deterministic per-key data (splitmix64 — no RNG state, no history)
# ---------------------------------------------------------------------------
def splitmix64(x: int) -> int:
    """SplitMix64 finalizer — the deterministic bit-mixer the rule is built on."""
    x = (int(x) + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) & M64


def u01(x: int) -> float:
    """Deterministic uniform-in-[0,1) draw from an integer seed."""
    return splitmix64(x) / 2.0**64


def hash_point(key: int, radius: float) -> np.ndarray:
    """``g(κ)`` — deterministic uniform-in-disk anchor for a key with no address."""
    r = radius * np.sqrt(u01(int(key) ^ C_G1))
    th = 2.0 * np.pi * u01(int(key) ^ C_G2)
    return np.array([r * np.cos(th), r * np.sin(th)], dtype=np.float64)


def prio(key: int) -> int:
    """``prio(κ)`` — 64-bit priority; ties are broken by the key itself."""
    return splitmix64(int(key) ^ C_PRIO)


def hex_cells(radius: float, d: float) -> np.ndarray:
    """Hex lattice of spacing ``d`` anchored at the origin, kept iff ``|c| <= radius``.

    Returned in a canonical (lexicographic) order, so the cell *index* is itself a
    function of the geometry alone.
    """
    n = int(np.ceil(radius / d)) + 2
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            c = np.array([d * i + 0.5 * d * j, (np.sqrt(3.0) / 2.0) * d * j])
            if np.hypot(c[0], c[1]) <= radius + 1e-9:
                pts.append(c)
    pts = np.array(pts, dtype=np.float64)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def n_cells_for(radius: float, d_safe: float) -> int:
    """Number of lattice cells (= the hard admission capacity) of a disk geometry."""
    return int(len(hex_cells(radius, d_safe)))


def radius_for_cells(k: int, d_safe: float, max_mult: float = 2.0) -> float:
    """Smallest radius (on a 1 %-step ladder) whose lattice holds ``>= k`` cells.

    The packing *bound* ``radius_for_capacity`` inverts is an area count and overshoots on
    the boundary: at ``K=64`` it gives 61 cells, at ``K=8`` only 7. The theorist's sizing
    rule is "multiply by 1.05" — true for K in {16,32,64,128} (19/37/73/139) but **not**
    for small K, where the lattice is discrete (K=8 needs the next ring, mult ~1.35). This
    helper is the honest version: grow until the cells actually exist.
    """
    r0 = float(d_safe * np.sqrt((np.sqrt(3.0) / 2.0) * k / np.pi))
    mult = 1.0
    while mult <= max_mult:
        r = r0 * mult
        if n_cells_for(r, d_safe) >= k:
            return r
        mult *= 1.01
    raise ValueError(f"no radius <= {max_mult}*R({k}) holds {k} cells at d_safe={d_safe}")


# ---------------------------------------------------------------------------
# the placer
# ---------------------------------------------------------------------------
class CanonicalPlacer:
    """Incrementally-maintained canonical placement over a fixed lattice.

    The placer owns **only** the assignment ``key -> cell index``; payloads, amplitudes,
    decay and the potential itself stay with the store/controller (T4: they factorize).

    Args:
        radius: address-disk radius the lattice is clipped to.
        d_safe: lattice spacing == the admission radius (T3).
        anchor_dim: dimension of an anchor point; only the first 2 coordinates index the
            lattice (the store's address plane).

    Every mutating verb leaves the object in the *canonical* configuration for its current
    key set, and records the displacement moves it had to apply in :attr:`moves_last_op`
    (``(key, old_cell, new_cell)``) — the delete-time churn the theorist priced at ~2.84
    moves/delete at full load.
    """

    def __init__(self, radius: float, d_safe: float, anchor_dim: int = 2):
        self.radius = float(radius)
        self.d_safe = float(d_safe)
        self.anchor_dim = int(anchor_dim)
        self.cells = hex_cells(self.radius, self.d_safe)
        self.n_cells = int(len(self.cells))
        self.anchors: Dict[int, np.ndarray] = {}
        self.pos: Dict[int, Optional[int]] = {}
        self._probe: Dict[int, np.ndarray] = {}
        self.moves_last_op: List[Tuple[int, Optional[int], int]] = []
        self.dropped_last_op: List[int] = []

    # -- introspection --------------------------------------------------------
    @property
    def keys(self) -> List[int]:
        """Placed keys, in canonical (descending-priority) order."""
        return self._sorted_keys()

    def cell_of(self, key: int) -> Optional[int]:
        return self.pos.get(int(key))

    def center_of(self, key: int) -> np.ndarray:
        """The lattice site a placed key occupies (2,)."""
        return self.cells[self.pos[int(key)]].copy()

    def min_spacing(self) -> float:
        """Achieved min pairwise spacing — a lattice invariant, ``>= d_safe`` (T3)."""
        idx = [self.pos[k] for k in self._sorted_keys()]
        if len(idx) < 2:
            return float("inf")
        c = self.cells[np.asarray(idx)]
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        return float(d[np.triu_indices(len(c), 1)].min())

    # -- deterministic per-key data ------------------------------------------
    def probe_order(self, key: int) -> np.ndarray:
        """``π(κ)`` — every cell, sorted by distance to the key's anchor then by index."""
        key = int(key)
        if key not in self._probe:
            g = self.anchors[key]
            d = np.hypot(self.cells[:, 0] - g[0], self.cells[:, 1] - g[1])
            self._probe[key] = np.lexsort((np.arange(self.n_cells), d))
        return self._probe[key]

    def _sorted_keys(self) -> List[int]:
        return sorted(self.pos, key=lambda k: (-prio(k), k))

    # -- the canonical greedy over a priority suffix --------------------------
    def _replace_from(self, pivot_prio: int):
        """Re-run the greedy for every key with ``prio <= pivot_prio``.

        Keys of strictly higher priority never reference lower ones, so their cells are
        provably unchanged (suffix-stability lemma) — this is the *bounded* fix-up
        cascade, not a global rebuild.
        """
        moves: List[Tuple[int, Optional[int], int]] = []
        dropped: List[int] = []
        occupied = set()
        for k in self._sorted_keys():
            if prio(k) > pivot_prio and self.pos[k] is not None:
                occupied.add(self.pos[k])
        for k in self._sorted_keys():
            if prio(k) > pivot_prio:
                continue
            newpos = None
            for c in self.probe_order(k):
                if int(c) not in occupied:
                    newpos = int(c)
                    break
            if newpos is None:
                dropped.append(k)
                continue
            old = self.pos.get(k)
            if old != newpos:
                moves.append((k, old, newpos))
            self.pos[k] = newpos
            occupied.add(newpos)
        for k in dropped:
            self.pos.pop(k, None)
            self.anchors.pop(k, None)
            self._probe.pop(k, None)
        self.moves_last_op = [m for m in moves if m[1] is not None]
        self.dropped_last_op = dropped
        return self.moves_last_op, dropped

    # -- the verbs ------------------------------------------------------------
    def insert(self, key: int, anchor=None) -> bool:
        """Offer a key (with its address ``anchor``). Returns whether it is placed.

        ``anchor=None`` uses the deterministic :func:`hash_point` of the key — the
        content-free rung used by the theorist's harness.
        """
        key = int(key)
        if key in self.pos:
            raise KeyError(f"key {key} already placed")
        g = (
            hash_point(key, self.radius)
            if anchor is None
            else np.asarray(anchor, dtype=np.float64).reshape(-1)[:2]
        )
        self.anchors[key] = g
        self._probe.pop(key, None)
        self.pos[key] = None
        self._replace_from(prio(key))
        return key in self.pos

    def delete(self, key: int) -> List[Tuple[int, Optional[int], int]]:
        """Exact deletion (**T2**): drop the key, then restore canonical placement.

        Returns the displacement moves applied to the *survivors* — deleting ``i``
        legitimately relocates lower-priority items, to exactly where the store that never
        held ``i`` would have put them.
        """
        key = int(key)
        if key not in self.pos:
            raise KeyError(f"key {key} is not placed")
        p = prio(key)
        self.pos.pop(key)
        self.anchors.pop(key, None)
        self._probe.pop(key, None)
        moves, _ = self._replace_from(p)
        return moves

    # -- the bit-identity object ---------------------------------------------
    def layout(self) -> List[Tuple[int, np.ndarray]]:
        """``[(key, center)]`` in canonical priority order — the store's slot layout."""
        return [(k, self.cells[self.pos[k]].copy()) for k in self._sorted_keys()]

    def centers(self) -> np.ndarray:
        """(n_placed, 2) occupied sites in canonical priority order."""
        ks = self._sorted_keys()
        if not ks:
            return np.zeros((0, 2))
        return np.stack([self.cells[self.pos[k]] for k in ks])


def canonical_layout(radius: float, d_safe: float, items) -> List[Tuple[int, np.ndarray]]:
    """``Store(S)`` built fresh from a set of ``(key, anchor)`` pairs.

    The reference object every incremental history is compared against: feeding the same
    set in any order (or reaching it by any write/delete interleaving) must give this.
    """
    pl = CanonicalPlacer(radius, d_safe)
    for key, anchor in sorted(items, key=lambda it: (-prio(it[0]), int(it[0]))):
        pl.insert(key, anchor)
    return pl.layout()
