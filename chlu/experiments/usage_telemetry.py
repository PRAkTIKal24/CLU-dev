"""⭐ **C2W8 build requirement B2** — usage telemetry at I2 grade.

*"Truly useless" must be **computable**, or the prune verb is a guess wearing the
word "decision"* (`PREREG-C2W8.md` §2 B2). This module is that instrument, and
nothing more: it answers **"which live wells were never read?"**, which needs no
correlation and no effect size.

Three prerequisites, all inherited from the C2W6 erosion adjudication and all
binding here:

* **item-id key, never a slot key** — slot != well (same-slot site drift is
  0.32-0.67x the between-slot spread at place radius 0.30), so a slot-keyed
  counter silently merges an evicted item with its successor. The counter lives
  on :class:`~chlu.core.controller.Controller` keyed by ``item_id`` and survives
  eviction.
* **one registered primary proxy, decided in the prereg and not on results** —
  :data:`PRIMARY_PROXY` = ``read_hits(i)``: the number of stream reads whose
  settled point is assigned to well ``i``'s basin (``_assign`` against the live
  codebook). Computable online at O(1) per read, independent of depth
  (mechanic 2), and exactly the quantity the ``gamma_phi`` criterion names.
* **the LOO probe is reported only beside its ICC(1,1)** and, when
  ``ICC <= 0``, is labelled :data:`UNDEFINED` with **no number quoted from it**.
  C2W6 measured ICC negative on 3/3 seeds => attenuation ceiling 0.000 =>
  ``rho(LOO)`` was a *non-measurement*, not a null. :func:`icc_1_1` and
  :func:`loo_loss_contribution` exist so that this wave cannot repeat it.

⛔ **Depth is not a usage proxy and does not enter ``U``** (prereg §3.2).
⛔ **No I2 verdict is computed here.** The I2 correlation test (does usefulness
predict erosion?) is a declared NOT-RUN deferred to C2W10 (prereg §9); this file
builds the instrument that makes "never read" computable and stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: The registered primary usage proxy (prereg §3.2). Fixed before any cell ran.
PRIMARY_PROXY = "read_hits"

#: The label a quantity carries when its ICC(1,1) is <= 0 — never "0.0", never a
#: null: an attenuation ceiling of 0 means the quantity was not measured at all.
UNDEFINED = "UNDEFINED"


@dataclass
class UsageTelemetry:
    """Item-id-keyed usage record ``U`` accumulated over a stream.

    Attributes:
        read_hits: ``item_id -> `` number of reads assigned to that item's basin.
        first_seen: ``item_id -> `` stream tick at which the item was admitted
            ("since first appearance", the Head's criterion).
        last_read: ``item_id -> `` stream tick of the most recent hit
            (``-1`` = never).
        n_read_events: total reads observed (the denominator).
        n_unassigned: reads that landed in no live basin at all.
    """

    read_hits: Dict[int, int] = field(default_factory=dict)
    first_seen: Dict[int, int] = field(default_factory=dict)
    last_read: Dict[int, int] = field(default_factory=dict)
    n_read_events: int = 0
    n_unassigned: int = 0
    proxy: str = PRIMARY_PROXY

    # -- accumulation ------------------------------------------------------
    def note_admitted(self, item_id: int, t: int) -> None:
        """Record an admission (``first_seen``); leaves ``read_hits`` at 0."""
        iid = int(item_id)
        self.first_seen.setdefault(iid, int(t))
        self.read_hits.setdefault(iid, 0)
        self.last_read.setdefault(iid, -1)

    def note_read(self, item_id: int, t: int, controller=None) -> None:
        """Record one read hit on ``item_id``.

        ⭐ Routed through :meth:`chlu.core.controller.Controller.touch` when a
        controller is given — the prereg's "the existing ``Controller.touch``
        path extended to record reads" — so the store's own counter and this
        object cannot drift apart.
        """
        iid = int(item_id)
        if controller is not None:
            controller.touch(iid)
        self.read_hits[iid] = self.read_hits.get(iid, 0) + 1
        self.last_read[iid] = int(t)
        self.first_seen.setdefault(iid, int(t))

    def observe_basins(self, item_ids: Sequence[int], basins: Sequence[int],
                       t: int, controller=None,
                       covered: Optional[Sequence[bool]] = None) -> int:
        """Fold one read batch in: ``basins[k]`` indexes ``item_ids``.

        ``covered[k] = False`` marks a read that landed in no basin (it is
        counted in ``n_unassigned`` and credited to nobody — an uncovered read is
        not evidence that the nearest well was useful).
        """
        ids = np.asarray(item_ids, dtype=int)
        bs = np.asarray(basins, dtype=int)
        cov = np.ones(bs.shape, dtype=bool) if covered is None else np.asarray(covered, bool)
        n = 0
        for b, c in zip(bs, cov, strict=True):
            self.n_read_events += 1
            if (not bool(c)) or b < 0 or b >= ids.size:
                self.n_unassigned += 1
                continue
            self.note_read(int(ids[b]), t, controller=controller)
            n += 1
        return n

    # -- the decision input ------------------------------------------------
    def hits(self, item_id: int) -> int:
        return int(self.read_hits.get(int(item_id), 0))

    def never_read(self, item_id: int) -> bool:
        """``read_hits(i) == 0`` — "never useful since first appearance"."""
        return self.hits(item_id) == 0

    def summary(self, live_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """The report block. ``live_ids`` restricts it to the live population."""
        ids = (sorted(self.read_hits) if live_ids is None
               else [int(i) for i in live_ids])
        h = np.asarray([self.hits(i) for i in ids], dtype=float)
        return {
            "proxy": self.proxy,
            "key": "item_id",
            "n_items": int(len(ids)),
            "n_read_events": int(self.n_read_events),
            "n_unassigned": int(self.n_unassigned),
            "n_never_read": int(np.sum(h == 0)) if h.size else 0,
            "frac_never_read": float(np.mean(h == 0)) if h.size else float("nan"),
            "hits_mean": float(np.mean(h)) if h.size else float("nan"),
            "hits_median": float(np.median(h)) if h.size else float("nan"),
            "hits_max": float(np.max(h)) if h.size else float("nan"),
            "hits_by_item": {int(i): self.hits(i) for i in ids},
        }


def attach_reads(system, telemetry: UsageTelemetry, read_result, t: int) -> int:
    """Fold a :class:`~chlu.core.clu_system.ReadResult` into ``U``.

    The assignment is the read's own ``assign_settle`` diagnostic — the settled
    address matched against the **live codebook** — so the telemetry uses exactly
    the basin the system itself resolved, not a re-derivation.
    """
    ids, _, _ = system.codebook()
    diag = getattr(read_result, "diagnostics", {}) or {}
    basins = diag.get("assign_settle")
    if basins is None or len(ids) == 0:
        return 0
    return telemetry.observe_basins(ids, basins, t,
                                    controller=system.controller.allocator,
                                    covered=diag.get("covered"))


# --------------------------------------------------------------------------
# the SECONDARY, reported-not-deciding leg: leave-one-out + its ICC
# --------------------------------------------------------------------------
def icc_1_1(values: np.ndarray) -> float:
    """ICC(1,1) of an ``(n_targets, k_repeats)`` table (one-way random effects).

    ``ICC = (MSB - MSW) / (MSB + (k-1) MSW)``. A value ``<= 0`` means the
    between-target variance is not resolved above the within-target noise: the
    attenuation ceiling is 0 and **any correlation computed on this quantity is
    undefined, not null** (C2W6, 3/3 seeds).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return float("nan")
    n, k = x.shape
    grand = float(np.mean(x))
    row = np.mean(x, axis=1)
    msb = float(k * np.sum((row - grand) ** 2) / (n - 1))
    msw = float(np.sum((x - row[:, None]) ** 2) / (n * (k - 1)))
    denom = msb + (k - 1) * msw
    if abs(denom) < 1e-300:
        return float("nan")
    return float((msb - msw) / denom)


def loo_loss_contribution(system, item_ids: Sequence[int], *,
                          repeats: int = 2, seed: int = 0) -> Dict[str, Any]:
    """SECONDARY proxy: leave-one-out self-probe damage, **with** its ICC(1,1).

    For each live item ``i`` the item's own atom group is muted (amplitudes
    scaled to ~0, the store's own per-item lifetime mechanism at its limit) and
    the label-free self-probe is re-run on the **remaining** items; the
    contribution is the drop in their strict retention. Repeated ``repeats``
    times with different probe jitter so ICC(1,1) is computable.

    ⛔ The returned ``values`` are **reported only beside** ``icc``; when
    ``icc <= 0`` the caller must label the quantity :data:`UNDEFINED` and quote
    no number from it. This function never decides anything.
    """
    import jax

    ids = [int(i) for i in item_ids]
    base_store = system.store
    out = np.full((len(ids), int(repeats)), np.nan, dtype=float)
    try:
        for r in range(int(repeats)):
            key = jax.random.PRNGKey(int(seed) + 977 * (r + 1))
            base = system.self_probe(key)
            base_ret = np.asarray(base.get("retention", []), dtype=float)
            live_ids, _, _ = system.codebook()
            pos = {int(v): j for j, v in enumerate(live_ids)}
            for a, iid in enumerate(ids):
                if iid not in pos:
                    continue
                try:
                    slot = system._slot_of(iid)
                except KeyError:
                    continue
                system.store = base_store.scale_group_amplitude(slot, 1e-12)
                probe = system.self_probe(key)
                ret = np.asarray(probe.get("retention", []), dtype=float)
                system.store = base_store
                mask = np.ones(ret.shape, dtype=bool)
                mask[pos[iid]] = False
                if ret.size == base_ret.size and mask.any():
                    out[a, r] = float(np.mean(base_ret[mask]) - np.mean(ret[mask]))
    finally:
        system.store = base_store
    icc = icc_1_1(out[np.all(np.isfinite(out), axis=1)])
    usable = bool(np.isfinite(icc) and icc > 0)
    return {
        "proxy": "loss_contribution_loo",
        "role": "SECONDARY — reported, never a decision input (prereg §3.2)",
        "icc_1_1": float(icc),
        "status": ("usable" if usable else UNDEFINED),
        "values": (out.tolist() if usable else None),
        "note": ("ICC(1,1) <= 0 => attenuation ceiling 0.000 => the quantity is "
                 "UNDEFINED, not a null; no number is quoted from it"
                 if not usable else
                 "ICC(1,1) > 0: values may be reported, still never a decision input"),
        "item_ids": ids,
        "repeats": int(repeats),
    }


__all__ = [
    "PRIMARY_PROXY", "UNDEFINED", "UsageTelemetry", "attach_reads",
    "icc_1_1", "loo_loss_contribution",
]
