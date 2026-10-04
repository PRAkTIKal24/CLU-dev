"""⭐ **The store-attribution curve** — Route 3 stage 1's instrument (charter §A10).

C2W2 asked the *write* to put information into the trajectory and it was a monotone
cost (dividend −0.0278 → −0.2639 over a three-decade ``lambda_traj`` grid). The
Advisor's ruling on that null is **not** "buy a longer rollout" but a change of
variable (§A8.1, verbatim):

    The read-length requirement was an artifact of the payload convention, not
    intrinsic. Payload channels pinned to 0 at launch must *climb*, so information
    lives late. **If the latent is the visited state itself, the store acts from
    the first step**: acceleration is ``-M^-1 grad V`` and **``grad V`` IS the
    store** — and **momentum ``p_t ~ -int grad V dt`` at small ``t`` is almost
    pure store**, while **position at small ``t`` is almost pure query**. => the
    design-deciding quantity is **store-attribution over time**, not rollout
    length.

This module measures exactly that, on the *existing* merged rig — no new store, no
new objective (that is stage 2's job, and it only exists if the §A9.4 bar clears).

**Four objects per slot, and they are deliberately four different things.**

``full``
    the family's own answer channel read out of slot ``t`` of the real read.
``launder`` (**the per-slot settle-deleted launder**)
    the identical read on the **store-deleted** system — same launch, same
    integrator, same slot, same instrument. ⚠ Declared mapping: the shipped
    :func:`chlu.eval.dividend.settle_deleted_launder` is a *settled-point* table
    object and has **no slot index**; the per-slot instantiation of "delete the
    settle" is "delete the store that creates the settle", which leaves the launch
    and the dynamics intact. ``full - launder`` is the **store-attributable**
    part, which is the quantity §A9.4 names.
``floor`` (**the launch-noise floor**)
    the same discriminability on the store-deleted arm under an independently
    re-drawn launch perturbation of the same law — *"perturb the launch within its
    own cloud and measure the slot's discriminability under that noise alone"*.
    It is subtracted **in addition** to the launder, so the bar is conservative by
    construction; the un-floored dividend is reported beside it.
``table`` (⛔ **the §A9.5 per-slot matched-bytes table launder — a KILL-CONDITION,
not a control**)
    ``K`` time-indexed rows (one per live item, the item's mean slot content),
    keyed by the query's nearest stored key, evaluated **leave-one-query-out**.
    Intervention §8.2, instantiated: *"if it reproduces the slotted read, Route 3
    has degenerated into K time-indexed lookup tables and FAILS REGARDLESS OF
    DIVIDEND."*

**The instrument is a scale-free rank correlation and it carries no fitted
parameters and no bytes.** That is a deliberate design choice made before the
first run (see ``PREREG.md`` §1.2): at small ``t`` the momentum channel is
``O(t)`` small in magnitude but — if §A8.1 is right — *proportional* to the stored
payload, so any magnitude-based decode would confound "small" with
"uninformative". Identical instrument on both channels, all four arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SLOT_GRID",
    "THIRDPARTY_SLOT_GRID",
    "SlotScore",
    "ThirdPartyRow",
    "spearman",
    "discriminability",
    "slot_channel",
    "slot_index_table",
    "attribution_curve",
    "identity_decode",
    "address_block_curve",
    "slot_block",
    "per_slot_table_launder",
    "table_row_assignment",
    "table_third_party_delta",
    "slot_deltas",
    "third_party_curve",
    "coupling_law_fit",
    "jacobian_curves",
    "apply_a94_bar",
    "a95_verdict",
]

#: ⛔ PRE-REGISTERED slot grid (``PREREG.md`` §2), declared before any run.
#: Concatenated-buffer point indices: phase 1 = 0..49 (``address_steps=400`` at
#: ``traj_stride=8``), phase 2 = 50..149 (``read_steps=800``). Point ``j`` of a
#: phase is integrator step ``8j+1``. Log-dense at the small-``t`` end, because
#: that is where §A8.1 makes its prediction.
SLOT_GRID: Tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 49,
                              54, 62, 74, 99, 124, 149)

#: ⛔ PRE-REGISTERED slot grid for the **third-party** probe (C2W4 `bprime-c6`,
#: theorist code request **C7**): buffer slots -> integrator steps
#: ``{1,9,17,25,33,49,65,97,129,161,193,233}`` at ``traj_stride = 8``, i.e. the
#: whole grid lies inside C7's ``t in [1, 240]`` steps. Beyond ``t ~ 240`` the
#: within-basin variance is ``< 1e-3 sigma_q`` and by ``t = 1200`` it is
#: ``4.7e-9 sigma_q`` — slots out there measure nothing.
THIRDPARTY_SLOT_GRID: Tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 29)

CHANNELS = ("q", "p")


# --------------------------------------------------------------------------
# the scale-free instrument
# --------------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — the only thing Spearman needs."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    # average tied ranks
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i + 1
        while j < xs.size and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = float(np.mean(ranks[order[i:j]]))
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Spearman rank correlation. ``nan`` if either side is constant."""
    ra, rb = _rank(a), _rank(b)
    sa, sb = float(np.std(ra)), float(np.std(rb))
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def identity_decode(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out nearest-prototype **item-identity** accuracy of a slot.

    ⚠ **POST-HOC SECONDARY DIAGNOSTIC, declared as such** (added after the
    pre-registered curve was measured; it does **not** enter the §A9.4 bar). It
    exists to adjudicate the half of §A8.1 the answer-channel instrument cannot
    see: *"position at small t is almost pure query"*. The answer channel of
    ``q0`` is identically zero by construction (``CluSystem.read`` zeroes the
    payload block), so the claim can only be tested on the **address block**,
    where ``q0`` *is* the query — and there the right question is identity, not
    payload order.
    """
    v = np.asarray(vectors, dtype=float)
    lab = np.asarray(labels).ravel()
    uniq = np.unique(lab)
    if uniq.size < 2 or v.shape[0] != lab.size:
        return float("nan")
    sums = {k: v[lab == k].sum(axis=0) for k in uniq}
    counts = {k: int(np.sum(lab == k)) for k in uniq}
    hits = 0
    for i in range(v.shape[0]):
        protos, keys = [], []
        for k in uniq:
            n = counts[k] - (1 if lab[i] == k else 0)
            if n <= 0:
                continue
            s = sums[k] - (v[i] if lab[i] == k else 0.0)
            protos.append(s / n)
            keys.append(k)
        if not protos:
            continue
        d = np.linalg.norm(np.stack(protos) - v[i][None, :], axis=-1)
        hits += int(keys[int(np.argmin(d))] == lab[i])
    return float(hits / v.shape[0])


def slot_block(res, slot: int, channel: str, lo: int, hi: int) -> np.ndarray:
    """A contiguous block of one slot's state: ``(B, hi-lo)``."""
    traj = np.asarray(res.traj)
    dim = traj.shape[-1] // 2
    off = 0 if channel == "q" else dim
    return traj[:, int(slot), off + int(lo): off + int(hi)]


def address_block_curve(res_full, res_launder, res_floor, labels: np.ndarray,
                        addr_dim: int, *, slots: Sequence[int] = SLOT_GRID
                        ) -> List[dict]:
    """⚠ POST-HOC secondary: identity-decode of the **address block** per slot.

    Same four arms, same slots, different question. If §A8.1's *"position at
    small t is almost pure query"* holds, ``full`` and ``launder`` coincide at
    small ``t`` here (both are the query) and separate only once the store has
    had time to move the particle.
    """
    out: List[dict] = []
    n = int(np.asarray(res_full.traj).shape[1])
    for j in slots:
        if j >= n:
            continue
        for ch in CHANNELS:
            a = identity_decode(slot_block(res_full, j, ch, 0, addr_dim), labels)
            b = identity_decode(slot_block(res_launder, j, ch, 0, addr_dim), labels)
            c = identity_decode(slot_block(res_floor, j, ch, 0, addr_dim), labels)
            out.append({"slot": int(j), "channel": ch, "block": "address",
                        "full": _j(a), "launder": _j(b), "floor": _j(c),
                        "dividend": _j(a - b)})
    return out


def discriminability(values: np.ndarray, target: np.ndarray) -> float:
    """``|Spearman rho|`` — the per-slot discriminability ``D``.

    Scale-free by construction (see the module docstring): a channel that is
    ``1e-6`` in magnitude but perfectly ordered with the target scores 1.0.
    """
    r = spearman(values, target)
    return float(abs(r)) if np.isfinite(r) else float("nan")


# --------------------------------------------------------------------------
# slots
# --------------------------------------------------------------------------
def slot_index_table(res, slots: Sequence[int] = SLOT_GRID,
                     traj_stride: int = 8, dt: float = 0.05,
                     address_steps: int = 400) -> List[dict]:
    """``slot -> (phase, integrator step, time)`` provenance for every slot.

    Retry rounds append phase-2 points; slots are indexed off the phase-1/phase-2
    boundary recorded in ``res.phase``, and any appended retry points are **not**
    added to the grid (``PREREG.md`` §2).
    """
    phase = np.asarray(res.phase)
    n = int(np.asarray(res.traj).shape[1])
    n1 = int(np.sum(phase == 1))
    out = []
    for j in slots:
        if j >= n:
            continue
        ph = int(phase[j])
        k = j if ph == 1 else j - n1
        step = traj_stride * k + 1 + (address_steps if ph == 2 else 0)
        out.append({"slot": int(j), "phase": ph, "step": int(step),
                    "t": float(step * dt)})
    return out


def slot_channel(res, slot: int, channel: str, index: int) -> np.ndarray:
    """The scalar answer channel of one slot: ``(B,)``.

    ``res.traj`` is ``(B, n_points, 2*dim)`` = ``[q | p]`` per point, so ``q`` and
    ``p`` slots are the two halves — **scored separately**, per §A9.4.
    """
    traj = np.asarray(res.traj)
    dim = traj.shape[-1] // 2
    off = 0 if channel == "q" else dim
    return traj[:, int(slot), off + int(index)]


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------
@dataclass
class SlotScore:
    """One ``(slot, channel)`` row of the attribution curve, one seed."""

    slot: int
    channel: str
    phase: int
    step: int
    t: float
    full: float = float("nan")
    launder: float = float("nan")
    floor: float = float("nan")
    table: float = float("nan")
    signed_rho_full: float = float("nan")

    @property
    def dividend(self) -> float:
        """``full - launder`` — the store-attributable part (un-floored)."""
        return float(self.full - self.launder)

    @property
    def margin(self) -> float:
        """⭐ The §A9.4 quantity: ``(full - launder) - launch-noise floor``."""
        return float(self.dividend - self.floor)

    @property
    def table_margin(self) -> float:
        """⛔ §A9.5: ``read - table``. ``<= 0`` means K time-indexed rows
        reproduce (or beat) the slotted read."""
        return float(self.full - self.table)

    def as_dict(self) -> dict:
        return {"slot": self.slot, "channel": self.channel, "phase": self.phase,
                "step": self.step, "t": self.t, "full": _j(self.full),
                "launder": _j(self.launder), "floor": _j(self.floor),
                "table": _j(self.table), "signed_rho_full": _j(self.signed_rho_full),
                "dividend": _j(self.dividend), "margin": _j(self.margin),
                "table_margin": _j(self.table_margin)}


def attribution_curve(res_full, res_launder, res_floor, target: np.ndarray,
                      channel_index: int, *, slots: Sequence[int] = SLOT_GRID,
                      keys: Optional[np.ndarray] = None,
                      centers: Optional[np.ndarray] = None,
                      traj_stride: int = 8, dt: float = 0.05,
                      address_steps: int = 400) -> List[SlotScore]:
    """The per-slot curve for **both** channels, one seed, one family.

    ``res_full`` real store · ``res_launder`` store-deleted, same launch ·
    ``res_floor`` store-deleted, independently re-drawn launch. ``keys``/
    ``centers`` (optional) switch on the ⛔ §A9.5 per-slot table launder.
    """
    prov = {r["slot"]: r for r in slot_index_table(
        res_full, slots, traj_stride=traj_stride, dt=dt,
        address_steps=address_steps)}
    tgt = np.asarray(target, dtype=float).ravel()
    rows: List[SlotScore] = []
    for j, meta in prov.items():
        for ch in CHANNELS:
            v_full = slot_channel(res_full, j, ch, channel_index)
            row = SlotScore(slot=j, channel=ch, phase=meta["phase"],
                            step=meta["step"], t=meta["t"])
            row.full = discriminability(v_full, tgt)
            row.signed_rho_full = spearman(v_full, tgt)
            row.launder = discriminability(
                slot_channel(res_launder, j, ch, channel_index), tgt)
            row.floor = discriminability(
                slot_channel(res_floor, j, ch, channel_index), tgt)
            if keys is not None and centers is not None:
                row.table = per_slot_table_launder(v_full, tgt, keys, centers)
            rows.append(row)
    return rows


def per_slot_table_launder(values: np.ndarray, target: np.ndarray,
                           keys: np.ndarray, centers: np.ndarray) -> float:
    """⛔ **§A9.5, the kill-condition.** ``D`` of ``K`` time-indexed rows.

    Row ``k`` = the mean slot content of the queries whose **nearest stored key**
    is item ``k``; the prediction for query ``i`` is row ``k(i)`` computed
    **leave-one-out** (query ``i`` never contributes to its own row), so the table
    is not fitted on the point it is scored at. Bytes: ``K x 4 B`` per slot —
    strictly *cheaper* than the store (>= 478x on ``overload``), which is what
    makes reproduction fatal rather than merely unimpressive.
    """
    v = np.asarray(values, dtype=float).ravel()
    d = np.linalg.norm(np.asarray(keys)[:, None, :] - np.asarray(centers)[None, :, :],
                       axis=-1)
    assign = np.argmin(d, axis=1)
    pred = np.full(v.shape, np.nan, dtype=float)
    for k in np.unique(assign):
        m = assign == k
        n = int(np.sum(m))
        if n <= 1:
            pred[m] = np.nan  # a singleton row cannot be evaluated leave-one-out
            continue
        s = float(np.sum(v[m]))
        pred[m] = (s - v[m]) / (n - 1)
    ok = np.isfinite(pred)
    if int(np.sum(ok)) < 3:
        return float("nan")
    return discriminability(pred[ok], np.asarray(target, dtype=float).ravel()[ok])


# --------------------------------------------------------------------------
# ⭐⭐ C6 — THIRD-PARTY STORE ATTRIBUTION (`bprime-theory` T5.4, charter §A14.1)
#
# > **Prop T5.4 (the shared-index bottleneck).** A per-slot matched-bytes table
# > computes ``y = psi(x, r_{i(x),1..S})`` with ``i(x)`` ONE query-dependent row
# > selection, so for fixed ``x`` its dependence on any **non-selected** row is
# > ``d y / d(row) = 0`` **exactly**. A CLU slot obeys ``qdd = -M^-1 grad V`` with
# > ``grad V`` a sum over **ALL** wells, so deleting a stored item the query did
# > not select moves every slot.
#
# ⛔ **This is protocol evidence for the audit paper (TIER i), not a revival.**
# §A9.5's kill stands and is scoped by §A14.1 to inference-read claims. The
# table's third-party ``Delta`` is **exactly 0 BY CONSTRUCTION** — it is computed
# here rather than assumed, and it is the *definition of the contrast*, never a
# win. The CLU's side is **exponentially suppressed by our own admission gate**
# (`exp(-(d/s)^2/2)`; T5.5), which is the sentence this instrument exists to
# measure.
# --------------------------------------------------------------------------
def table_row_assignment(keys: np.ndarray, centers: np.ndarray, *,
                         drop: Optional[int] = None) -> np.ndarray:
    """Which table row each query selects — ``argmin`` over the stored keys.

    ``drop`` removes one row from the table first (the *deletion*), and the
    returned indices are into the ORIGINAL row set so the two assignments are
    directly comparable.
    """
    c = np.asarray(centers, dtype=float)
    rows = [k for k in range(c.shape[0]) if drop is None or k != int(drop)]
    if not rows:
        return np.full((np.asarray(keys).shape[0],), -1, dtype=int)
    d = np.linalg.norm(np.asarray(keys, dtype=float)[:, None, :] - c[rows][None, :, :],
                       axis=-1)
    return np.asarray(rows, dtype=int)[np.argmin(d, axis=1)]


def _table_predictions(values: np.ndarray, keys: np.ndarray, centers: np.ndarray,
                       *, drop: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-query-out per-slot table predictions, ``(pred, base_assignment)``.

    Identical object to :func:`per_slot_table_launder`'s (row ``k`` = the mean
    slot content of the queries whose nearest stored key is ``k``, evaluated
    leave-one-out), with the option of **deleting a row**.

    ⚠ **The row CONTENT is the table's bytes and is never re-fitted.** Deleting
    row ``k`` removes that row from the *selection* set; the surviving rows keep
    exactly the content the full table gave them. (Re-fitting the survivors on
    the orphaned queries would make the table's third-party ``Delta`` nonzero by
    an artefact of the estimator, not by any dependence on the deleted row — and
    Prop T5.4 is a statement about a fixed table.)
    """
    v = np.asarray(values, dtype=float).ravel()
    base = table_row_assignment(keys, centers)           # defines the row content
    sel = table_row_assignment(keys, centers, drop=drop)  # who each query reads
    sums = {int(k): float(np.sum(v[base == k])) for k in np.unique(base)}
    counts = {int(k): int(np.sum(base == k)) for k in np.unique(base)}
    pred = np.full(v.shape, np.nan, dtype=float)
    for i in range(v.size):
        k = int(sel[i])
        if k < 0 or k not in counts:
            continue
        n, s = counts[k], sums[k]
        if int(base[i]) == k:            # leave-one-out on its own row
            if n > 1:
                pred[i] = (s - v[i]) / (n - 1)
        elif n > 0:                      # an orphan reads a row it never joined
            pred[i] = s / n
    return pred, base


def table_third_party_delta(values: np.ndarray, keys: np.ndarray,
                            centers: np.ndarray, *, drop: int) -> Dict[str, Any]:
    """⭐ **The table's third-party Delta — COMPUTED, and it is exactly 0.**

    Deletes row ``drop`` from the §A9.5 per-slot matched-bytes table and measures
    how the prediction changes **for the queries that did not select that row**.

    ⭐ **`max_abs_delta` is `0.0` exactly — by construction (Prop T5.4)**, because
    a row-selecting table never reads a row it did not select. Reported as the
    measured zero it is; it is the contrast's definition, not a result.
    """
    pred_full, assign_full = _table_predictions(values, keys, centers)
    pred_drop, _ = _table_predictions(values, keys, centers, drop=int(drop))
    non_sel = (assign_full != int(drop))
    ok = non_sel & np.isfinite(pred_full) & np.isfinite(pred_drop)
    n = int(np.sum(ok))
    d = np.abs(pred_full[ok] - pred_drop[ok]) if n else np.zeros((0,), dtype=float)
    return {"dropped_row": int(drop), "n_non_selecting_queries": n,
            "max_abs_delta": float(np.max(d)) if n else float("nan"),
            "mean_abs_delta": float(np.mean(d)) if n else float("nan"),
            "exactly_zero": bool(n > 0 and np.all(d == 0.0)),
            "why": "0 by construction (Prop T5.4): a row-selecting table never "
                   "reads a row it did not select"}


def slot_deltas(res_a, res_b, *, slots: Sequence[int] = THIRDPARTY_SLOT_GRID,
                dim: Optional[int] = None) -> Dict[str, np.ndarray]:
    """``||s_t(a) - s_t(b)||`` per query, per slot, per channel: ``(n_slots, B)``.

    The whole latent is used (address + payload + spectator) because "the change
    in each slot's content" is a statement about the slot, not about one channel
    of it. ``q`` and ``p`` are returned separately (C7: record ``p_t``, not only
    ``q_t``).
    """
    ta = np.asarray(res_a.traj, dtype=float)
    tb = np.asarray(res_b.traj, dtype=float)
    d = int(dim or ta.shape[-1] // 2)
    js = [int(j) for j in slots if j < ta.shape[1] and j < tb.shape[1]]
    out = {}
    for ch, off in (("q", 0), ("p", d)):
        out[ch] = np.stack([np.linalg.norm(ta[:, j, off: off + d]
                                           - tb[:, j, off: off + d], axis=-1)
                            for j in js], axis=0)
    out["slots"] = np.asarray(js, dtype=int)
    return out


@dataclass
class ThirdPartyRow:
    """One ``(slot, channel)`` row of the third-party attribution curve, one seed.

    All ``delta_*`` are in units of ``sigma_q`` (the query law's own scale), so
    the numbers are comparable across the ``d/s`` sweep.
    """

    slot: int
    channel: str
    step: int
    t: float
    delta_third: float = float("nan")   # delete the SECOND-nearest item
    delta_third_se: float = float("nan")
    delta_own: float = float("nan")     # delete the query's OWN (selected) item
    delta_blank: float = float("nan")   # the blank-store delete control
    delta_launch: float = float("nan")  # the launch-noise floor
    delta_table: float = float("nan")   # ⛔ exactly 0 by construction (T5.4)
    coupling: float = float("nan")      # ⭐ MEDIAN_i delta_third / delta_own
    coupling_se: float = float("nan")
    coupling_mean: float = float("nan")
    n: int = 0

    @property
    def margin_blank(self) -> float:
        """``delta_third - blank-store delete control`` — the bar of task §4."""
        return float(self.delta_third - self.delta_blank)

    def as_dict(self) -> dict:
        return {"slot": self.slot, "channel": self.channel, "step": self.step,
                "t": self.t, "delta_third": _j(self.delta_third),
                "delta_third_se": _j(self.delta_third_se),
                "delta_own": _j(self.delta_own), "delta_blank": _j(self.delta_blank),
                "delta_launch": _j(self.delta_launch),
                "delta_table": _j(self.delta_table), "coupling": _j(self.coupling),
                "coupling_se": _j(self.coupling_se),
                "coupling_mean": _j(self.coupling_mean), "n": int(self.n),
                "margin_blank": _j(self.margin_blank)}


def third_party_curve(deltas_by_item: Dict[int, Dict[str, np.ndarray]],
                      sel: np.ndarray, third: np.ndarray, *,
                      blank: Optional[Dict[str, np.ndarray]] = None,
                      launch: Optional[Dict[str, np.ndarray]] = None,
                      table_delta: float = 0.0,
                      sigma_q: float = 1.0,
                      slot_meta: Optional[Dict[int, dict]] = None
                      ) -> List[ThirdPartyRow]:
    """⭐ The curve: per-slot ``Delta`` from deleting a **non-selected** item.

    ``deltas_by_item[k]`` is :func:`slot_deltas` between the full read and the
    read with item ``k`` deleted. ``sel[i]``/``third[i]`` are the query's nearest
    and second-nearest stored keys — ``sel`` is exactly the row a per-slot table
    selects, so ``third`` is exactly a row it provably never reads (T5.4).

    ⚠ ``coupling`` is the **median** per-query ratio, with the mean reported
    beside it: the ratio's denominator is a per-query gradient magnitude, so its
    mean is dominated by the queries whose own-well gradient is near zero. The
    estimator was fixed to the median on plumbing (``--quick``) runs, **before**
    the registered sweep.
    """
    any_item = next(iter(deltas_by_item.values()))
    js = [int(x) for x in any_item["slots"]]
    rows: List[ThirdPartyRow] = []
    sel = np.asarray(sel, dtype=int)
    third = np.asarray(third, dtype=int)
    for ch in CHANNELS:
        stack = {k: np.asarray(v[ch], dtype=float) for k, v in deltas_by_item.items()}
        for si, j in enumerate(js):
            d_third = np.asarray([stack[int(third[i])][si, i] for i in range(sel.size)])
            d_own = np.asarray([stack[int(sel[i])][si, i] for i in range(sel.size)])
            ok = np.isfinite(d_third) & np.isfinite(d_own) & (d_own > 0)
            ratio = d_third[ok] / d_own[ok]
            meta = (slot_meta or {}).get(j, {})
            n = int(np.sum(ok))
            rows.append(ThirdPartyRow(
                slot=j, channel=ch, step=int(meta.get("step", -1)),
                t=float(meta.get("t", float("nan"))),
                delta_third=float(np.mean(d_third) / sigma_q),
                delta_third_se=float(np.std(d_third, ddof=1)
                                     / np.sqrt(max(d_third.size, 1)) / sigma_q)
                if d_third.size > 1 else float("nan"),
                delta_own=float(np.mean(d_own) / sigma_q),
                delta_blank=(float(np.mean(blank[ch][si])) / sigma_q
                             if blank is not None else float("nan")),
                delta_launch=(float(np.mean(launch[ch][si])) / sigma_q
                              if launch is not None else float("nan")),
                delta_table=float(table_delta),
                coupling=float(np.median(ratio)) if n else float("nan"),
                coupling_se=(float(1.2533 * np.std(ratio, ddof=1) / np.sqrt(n))
                             if n > 1 else float("nan")),
                coupling_mean=float(np.mean(ratio)) if n else float("nan"),
                n=n))
    return rows


def coupling_law_fit(d_over_s: Sequence[float], coupling: Sequence[float], *,
                     sigma_q: float, d: Sequence[float]) -> Dict[str, Any]:
    """⭐ Fit T5.5's law and return the **implied well width**.

    T5.5's exchange rate is exactly

        ``kappa(d) = (d/sigma_q) * exp(-(d^2 - sigma_q^2) / (2 s^2))``

    (reproduced to the digit at its own ``sigma_q = 0.15, s = 0.35``), so
    ``ln kappa - ln(d/sigma_q)`` is **linear in ``d^2``** with slope
    ``-1/(2 s^2)``: the fit returns an *independent third estimate* of the
    learned well width, beside ``atom_init_width`` and ``CluSystem.well_fits``.
    """
    dd = np.asarray(d, dtype=float)
    k = np.asarray(coupling, dtype=float)
    ok = np.isfinite(dd) & np.isfinite(k) & (k > 0)
    if int(np.sum(ok)) < 3:
        return {"n": int(np.sum(ok)), "s_implied": float("nan"),
                "slope": float("nan"), "r2": float("nan"), "decades": float("nan")}
    y = np.log(k[ok]) - np.log(dd[ok] / float(sigma_q))
    x = dd[ok] ** 2 - float(sigma_q) ** 2
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    slope = float(coef[0])
    s_imp = float(np.sqrt(-1.0 / (2.0 * slope))) if slope < 0 else float("nan")
    return {"n": int(np.sum(ok)), "slope": slope, "intercept": float(coef[1]),
            "s_implied": s_imp,
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "decades": float(np.log10(np.max(k[ok]) / np.min(k[ok]))),
            "d_over_s_span": [float(np.min(np.asarray(d_over_s, dtype=float))),
                              float(np.max(np.asarray(d_over_s, dtype=float)))]}


# --------------------------------------------------------------------------
# ⭐ §A8.2 — the flow-map Jacobian, measured while we are in there
# --------------------------------------------------------------------------
def jacobian_curves(res_full, res_perturbed, delta: np.ndarray,
                    labels: np.ndarray, *, slots: Sequence[int] = SLOT_GRID,
                    dim: Optional[int] = None) -> Dict[str, Any]:
    """⭐ *"Encoder controls whether trajectories diverge or coincide"* = supervising
    the flow map's Jacobian (§A8.2): **contractive within an item's launch cloud**
    (noise robustness) and **separated across items** (capacity).

    * ``contraction[t]`` = ``mean_i ||s_t(q0_i + delta_i) - s_t(q0_i)|| / ||delta_i||``
      — the empirical operator gain of the flow map along launch-cloud directions.
      **< 1 = contractive.**
    * ``separation[t]`` = between-item prototype spread at ``t``, normalised by its
      value at the launch (``> 1`` = items pulled apart).
    * ``fisher[t]`` = between-item spread / within-item spread — the unitless
      capacity-relevant form (no normalisation needed).

    Both channels, because §A9.4 scores them separately.
    """
    tf = np.asarray(res_full.traj)
    tp = np.asarray(res_perturbed.traj)
    d = int(dim or tf.shape[-1] // 2)
    dn = np.linalg.norm(np.asarray(delta), axis=-1)
    lab = np.asarray(labels).ravel()
    out: Dict[str, Any] = {"slots": [], "channels": {}}
    for ch, off in (("q", 0), ("p", d)):
        con, sep, fis = [], [], []
        for j in slots:
            if j >= tf.shape[1]:
                continue
            a = tf[:, j, off: off + d]
            b = tp[:, j, off: off + d]
            gain = np.linalg.norm(b - a, axis=-1) / np.maximum(dn, 1e-12)
            con.append(float(np.mean(gain)))
            protos = np.stack([a[lab == k].mean(axis=0) for k in np.unique(lab)])
            pd = np.linalg.norm(protos[:, None, :] - protos[None, :, :], axis=-1)
            iu = np.triu_indices(protos.shape[0], 1)
            between = float(np.mean(pd[iu])) if iu[0].size else float("nan")
            within = float(np.mean([np.mean(np.linalg.norm(
                a[lab == k] - a[lab == k].mean(axis=0), axis=-1))
                for k in np.unique(lab)]))
            sep.append(between)
            fis.append(float(between / max(within, 1e-12)))
        sep0 = sep[0] if sep and np.isfinite(sep[0]) and sep[0] > 0 else float("nan")
        out["channels"][ch] = {
            "contraction": con,
            "separation_raw": sep,
            "separation_normalised": [float(s / sep0) for s in sep],
            "fisher": fis,
        }
    out["slots"] = [int(j) for j in slots if j < tf.shape[1]]
    return out


# --------------------------------------------------------------------------
# ⭐ §A9.4 — the bar, applied ARITHMETICALLY (nothing here interprets anything)
# --------------------------------------------------------------------------
def apply_a94_bar(per_seed: Dict[int, Sequence[SlotScore]], *,
                  family: str, admissible_seeds: Sequence[int],
                  min_seeds: int = 3) -> Dict[str, Any]:
    """⭐ The pre-registered stage-2 unlock bar, computed and **not interpreted**.

    > *Stage 2 unlocks iff the per-slot store-attributable discriminability
    > (full − settle-deleted launder, per slot t) clears the launch-noise floor
    > BEYOND 2 SE, at >= 3 seeds, on >= 1 family, at ANY t — q-slots and p-slots
    > scored separately (a live p-channel unlocks even with a dead q-channel).*

    ``margin = (full - launder) - floor`` per seed; ``mean - 2*SE > 0`` with
    ``SE = sd/sqrt(n)``, **sample sd (ddof=1)**, ``n = 3``.
    """
    seeds = [s for s in per_seed if s in set(admissible_seeds)]
    rows: List[dict] = []
    keyed: Dict[Tuple[int, str], Dict[int, SlotScore]] = {}
    for s in seeds:
        for r in per_seed[s]:
            keyed.setdefault((r.slot, r.channel), {})[s] = r
    for (slot, ch), by_seed in sorted(keyed.items()):
        m = np.asarray([by_seed[s].margin for s in sorted(by_seed)], dtype=float)
        d = np.asarray([by_seed[s].dividend for s in sorted(by_seed)], dtype=float)
        f = np.asarray([by_seed[s].full for s in sorted(by_seed)], dtype=float)
        ok = np.isfinite(m)
        n = int(np.sum(ok))
        mean = float(np.mean(m[ok])) if n else float("nan")
        sd = float(np.std(m[ok], ddof=1)) if n > 1 else float("nan")
        se = float(sd / np.sqrt(n)) if n > 1 else float("nan")
        clears = bool(n >= min_seeds and np.isfinite(se) and (mean - 2.0 * se) > 0.0)
        rows.append({
            "family": family, "channel": ch, "slot": int(slot),
            "phase": int(by_seed[sorted(by_seed)[0]].phase),
            "t": float(by_seed[sorted(by_seed)[0]].t),
            "n_seeds": n, "full_mean": float(np.nanmean(f)) if n else float("nan"),
            "dividend_mean": float(np.nanmean(d)) if n else float("nan"),
            "margin_mean": mean, "margin_sd": sd, "margin_se": se,
            "lower_2se": float(mean - 2.0 * se) if np.isfinite(se) else float("nan"),
            "clears": clears,
            "per_seed_margin": {int(s): _j(by_seed[s].margin) for s in sorted(by_seed)},
        })
    clearing = [r for r in rows if r["clears"]]
    return {
        "family": family,
        "rule": ("(full - settle_deleted_launder) - launch_noise_floor, per slot; "
                 "mean - 2*SE > 0; SE = sd/sqrt(n); sample sd ddof=1; n >= 3 seeds; "
                 "q and p scored separately (a live p-channel unlocks alone)"),
        "n_admissible_seeds": len(seeds),
        "admissible_seeds": sorted(int(s) for s in seeds),
        "rows": rows,
        "clearing_set": [{k: r[k] for k in ("channel", "slot", "t", "margin_mean",
                                            "margin_se", "lower_2se")}
                         for r in clearing],
        "unlock": bool(clearing) and len(seeds) >= min_seeds,
    }


def a95_verdict(per_seed: Dict[int, Sequence[SlotScore]], *,
                reproduces_tol: float = 0.05, beat_tol: float = 0.10,
                admissible_seeds: Optional[Sequence[int]] = None,
                clearing_slots: Optional[Sequence[Tuple[str, int]]] = None
                ) -> Dict[str, Any]:
    """⛔ **The §A9.5 kill-condition, computed.** *Does a per-slot matched-bytes
    table reproduce the slotted read?*

    ``reproduces`` = ``|read - table| <= reproduces_tol`` (or the table is better);
    ``read_beats`` = ``read - table > beat_tol``. The verdict is reported per
    channel and — because §A9.5 overrides §A9.4 — restricted to the slots that
    actually cleared the unlock bar as well.
    """
    seeds = sorted(per_seed) if admissible_seeds is None else sorted(
        s for s in per_seed if s in set(admissible_seeds))
    keyed: Dict[Tuple[int, str], List[float]] = {}
    for s in seeds:
        for r in per_seed[s]:
            if np.isfinite(r.table_margin):
                keyed.setdefault((r.slot, r.channel), []).append(r.table_margin)
    rows = []
    for (slot, ch), vals in sorted(keyed.items()):
        m = float(np.mean(vals))
        rows.append({"slot": int(slot), "channel": ch, "table_margin_mean": m,
                     "table_margin_sd": (float(np.std(vals, ddof=1))
                                         if len(vals) > 1 else float("nan")),
                     "n_seeds": len(vals),
                     "reproduces": bool(m <= reproduces_tol),
                     "read_beats": bool(m > beat_tol)})
    out: Dict[str, Any] = {"rows": rows, "reproduces_tol": reproduces_tol,
                           "beat_tol": beat_tol}
    for ch in CHANNELS:
        sub = [r for r in rows if r["channel"] == ch]
        n = len(sub)
        out[ch] = {
            "n_slots": n,
            "frac_reproduced": (float(np.mean([r["reproduces"] for r in sub]))
                                if n else float("nan")),
            "n_read_beats": int(sum(r["read_beats"] for r in sub)),
            "worst_read_margin": (float(max(r["table_margin_mean"] for r in sub))
                                  if n else float("nan")),
        }
    if clearing_slots is not None:
        cs = {(ch, int(sl)) for ch, sl in clearing_slots}
        sub = [r for r in rows if (r["channel"], r["slot"]) in cs]
        out["on_clearing_slots"] = {
            "n": len(sub),
            "n_reproduced": int(sum(r["reproduces"] for r in sub)),
            "n_read_beats": int(sum(r["read_beats"] for r in sub)),
            "rows": sub,
        }
        out["fires"] = bool(sub) and all(r["reproduces"] for r in sub)
    else:
        out["fires"] = bool(rows) and all(r["reproduces"] for r in rows)
    out["verdict"] = (
        "⛔ FIRES — a per-slot matched-bytes table reproduces the slotted read; "
        "Route 3 has degenerated into K time-indexed lookup tables and FAILS "
        "REGARDLESS OF DIVIDEND (intervention §8.2)"
        if out["fires"] else
        "does not fire — the slotted read is not reproduced by K time-indexed rows")
    return out


def _j(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


@dataclass
class CurveBundle:
    """Everything one ``(family, seed)`` cell contributes, JSON-ready."""

    family: str
    seed: int
    admissible: bool
    reason: str = ""
    rows: List[SlotScore] = field(default_factory=list)
    jacobian: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"family": self.family, "seed": int(self.seed),
                "admissible": bool(self.admissible), "reason": self.reason,
                "rows": [r.as_dict() for r in self.rows],
                "jacobian": self.jacobian, "flags": self.flags}
