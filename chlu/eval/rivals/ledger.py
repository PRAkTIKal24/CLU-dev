"""⭐ The **two-sided byte ledger** B′ prices every memory family with (D3.3).

One protocol, applied the same way to every family — that is the whole content of
the audit paper. Two axes, always reported together and never merged
(``rival-recon`` F2a; MAD's *iso-state* and Sparse Delta Memory's *isoParam +
state/param ratio* are the field's own precedents for doing so):

``F1 — parameters``
    every float that is **shared across sequences/streams**: the projections
    ``theta_{K,Q,V,O}``, the gates, and — by the rule below — the memory's
    **initialisation**.
``F2 — state``
    every float **mutated during the stream** and nothing else.

⭐ **The learned-initial-state rule** (`PREREG-Bprime.md` §4.1, and one of B′'s own
contributions):

    the initialisation ``W_0`` / ``S_0`` / ``V_theta(init)`` is **PARAMETERS**;
    only the **per-stream deviation** is **STATE**. Both are declared.

Counting the init as state **inflates**; counting the deviation as parameters
**launders**. ⛔ It is applied to the **CLU in the same table** — see
:func:`clu_two_sided_ledger`, which *measures* which ``V_theta`` leaves the masked
write actually moved rather than assuming all of them did — so no referee can say
we scored ourselves generously.

The matched-byte table a weight-valued memory is audited against
(:func:`matched_table_rows`, **P5**'s construction) is the byte-equal table of the
memory's own ``(theta_K x_t, theta_V x_t)`` pairs: ``n_rows = floor(state_floats /
(d_k + d_v))``. ⭐ Unlike the CLU — whose byte floor is
``ratio = [A(D+2) + d]/(d+m) >= 2.20x`` (`bprime-theory` T1, corrected; ⛔ the
published *"verified to 1e-9 in all 28 cells"* is **24/28**) — a rival's state
**can** be matched to its own table exactly, and measuring that asymmetry is part
of the audit's result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

#: float32 everywhere in this harness; declared, not assumed.
DTYPE_BYTES = 4

#: TTT's mini-batch, the paper's own value (Sun et al. §2.4: *"we chose b = 16 for
#: all experiments in this paper"*). The in-flight buffer is part of TTT's state
#: ledger (`PREREG-Bprime.md` §2).
TTT_MINI_BATCH = 16


# --------------------------------------------------------------------------
# the ledger object
# --------------------------------------------------------------------------
@dataclass
class TwoSidedLedger:
    """One arm's F1/F2 ledger, with the learned-initial-state rule applied.

    ``param_breakdown`` / ``state_breakdown`` are float **counts** (not bytes), so
    the arithmetic in a report is checkable by eye against the papers' own
    formulas.
    """

    arm: str
    param_floats: int
    state_floats: int
    param_breakdown: Dict[str, int] = field(default_factory=dict)
    state_breakdown: Dict[str, int] = field(default_factory=dict)
    state_convention: str = ""
    note: str = ""

    @property
    def param_bytes(self) -> int:
        return int(self.param_floats) * DTYPE_BYTES

    @property
    def state_bytes(self) -> int:
        return int(self.state_floats) * DTYPE_BYTES

    @property
    def state_over_param(self) -> float:
        """The ratio Sparse Delta Memory's Table 1 reports (⛔ whose *values* are
        quarantined — two extractions disagree — but whose *axis* is precedent)."""
        return float(self.state_floats) / max(float(self.param_floats), 1.0)

    def check(self) -> "TwoSidedLedger":
        """Structural self-consistency: the breakdowns must sum to the totals."""
        for name, total, bd in (("param", self.param_floats, self.param_breakdown),
                                ("state", self.state_floats, self.state_breakdown)):
            s = int(sum(int(v) for v in bd.values()))
            if bd and s != int(total):
                raise LedgerError(
                    f"{self.arm}: {name} breakdown sums to {s} but total is {total}")
        return self

    def as_dict(self) -> dict:
        return {"arm": self.arm,
                "param_floats": int(self.param_floats),
                "state_floats": int(self.state_floats),
                "param_bytes": self.param_bytes, "state_bytes": self.state_bytes,
                "state_over_param": self.state_over_param,
                "param_breakdown": dict(self.param_breakdown),
                "state_breakdown": dict(self.state_breakdown),
                "state_convention": self.state_convention, "note": self.note}


class LedgerError(AssertionError):
    """A byte ledger that does not add up. Raised, never warned."""


# --------------------------------------------------------------------------
# the matched-byte table (P5's construction)
# --------------------------------------------------------------------------
def matched_table_rows(state_floats: int, d_k: int, d_v: int) -> int:
    """How many ``(key, value)`` rows fit in **exactly** the memory's state bytes.

    This is **P5**: *"weight-valued memories → a byte-equal table of the
    ``(theta_K x, theta_V x)`` pairs"*. Rows are taken from the write stream in
    order; if the budget exceeds the stream the table is **lossless** and that is
    reported (``table_is_lossless``), because a lossless table is the strongest
    possible control and makes the audit's question sharp rather than convenient.
    """
    return int(max(0, int(state_floats) // max(int(d_k) + int(d_v), 1)))


def table_ledger(n_rows: int, d_k: int, d_v: int, *, arm: str = "table_launder",
                 param_floats: int = 0,
                 param_breakdown: Optional[Dict[str, int]] = None) -> TwoSidedLedger:
    """The launder's own ledger: the table is **state**, the shared projections are
    **parameters** (identical to the arm it launders, so they cancel — which is
    exactly why they must both be printed)."""
    st = int(n_rows) * (int(d_k) + int(d_v))
    return TwoSidedLedger(
        arm=arm, param_floats=int(param_floats), state_floats=st,
        param_breakdown=dict(param_breakdown or {}),
        state_breakdown={"table_keys": int(n_rows) * int(d_k),
                         "table_values": int(n_rows) * int(d_v)},
        state_convention=f"n_rows*(d_k+d_v) = {n_rows}*({d_k}+{d_v})",
    ).check()


def head_width_for_budget(kind: str, budget_floats: int, *,
                          buffer_tokens: int = TTT_MINI_BATCH) -> int:
    """⭐ The **iso-state sizing rule**, filed in the PREREG before any run.

    Return the largest head width whose *state* fits in ``budget_floats``:

    * ``"ttt_linear"``  ``d^2 + b*d``      (``d_head^2`` + the in-flight buffer)
    * ``"ttt_mlp"``     ``8 d^2 + b*d``    (two layers, 4x hidden)
    * ``"delta"``       ``d^2``            (``n_head * d_k * d_v`` at ``n_head=1``)

    ⛔ Never re-tuned after seeing a score: the budget is the CLU's own banked
    ``aggregate@base`` figure and the rule is arithmetic.
    """
    budget = int(budget_floats)
    b = int(buffer_tokens)
    d = 1
    while True:
        nxt = d + 1
        if kind == "ttt_linear":
            cost = nxt * nxt + b * nxt
        elif kind == "ttt_mlp":
            cost = 8 * nxt * nxt + b * nxt
        elif kind == "delta":
            cost = nxt * nxt
        else:
            raise ValueError(f"unknown ledger kind {kind!r}")
        if cost > budget:
            return int(d)
        d = nxt


# --------------------------------------------------------------------------
# the CLU's own ledger, under the SAME rule (the audit's sharpest edge)
# --------------------------------------------------------------------------
def clu_two_sided_ledger(v_before, v_after, n_codebook_rows: int, addr_dim: int,
                         *, atol: float = 0.0) -> TwoSidedLedger:
    """Apply the learned-initial-state rule to the **CLU**, by measurement.

    ``V_theta``'s *initialisation* is parameters; the **floats the masked write
    actually moved** are state. Because the write is group-masked (one atom group
    per item slot), only written slots' rows move — so this is measured by diffing
    the store's potential before and after the stream rather than assuming the
    whole store is state (which would inflate our own state budget and flatter the
    rivals' iso-state sizing).

    The live-address codebook is written at test time and is therefore **state**.
    """
    import equinox as eqx
    import jax

    def _leaves(v):
        return [np.asarray(x) for x in
                jax.tree_util.tree_leaves(eqx.filter(v, eqx.is_inexact_array))]

    a, b = _leaves(v_before), _leaves(v_after)
    if len(a) != len(b):
        raise LedgerError("V_theta leaf structure changed across the write stream")
    total = int(sum(x.size for x in a))
    moved = 0
    per_leaf: Dict[str, int] = {}
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        d = int(np.sum(np.abs(x - y) > atol))
        moved += d
        per_leaf[f"V_theta_leaf{i}{tuple(x.shape)}"] = d
    code = int(n_codebook_rows) * int(addr_dim)
    st = {k: v for k, v in per_leaf.items() if v}
    st["codebook_addresses"] = code
    return TwoSidedLedger(
        arm="clu", param_floats=total, state_floats=moved + code,
        param_breakdown={"V_theta_init": total},
        state_breakdown=st,
        state_convention=("floats moved by the masked write (measured leaf-diff) "
                          "+ K*addr_dim live codebook"),
        note=("learned-initial-state rule applied to the CLU in the same table as "
              "the rivals: V_theta's INIT is parameters, only the write-time "
              "deviation is state"),
    ).check()


# --------------------------------------------------------------------------
# the identical-phi invariant — enforced in code, raised not warned
# --------------------------------------------------------------------------
def phi_row(q0: np.ndarray, keys: np.ndarray) -> Dict[str, Any]:
    """``phi_id`` / ``phi_bytes`` for the gym's identity-launch read-in.

    Every arm — CLU, rival, table launder, +0 B substitute, null, blank — consumes
    the **same** ``(q0, keys)``, so the row is a content hash of them and
    ``phi_bytes = 0`` (the launch has no learnable parameters). ⚠ A rival's
    ``theta_{K,Q,V}`` are **not** ``phi``: they are the *memory arm's* parameters
    and are counted in F1 on the rival's read **and on its own table launder**,
    identically, which is what makes the two commensurable.
    """
    h = hashlib.sha256()
    for arr in (np.asarray(q0, dtype=np.float64), np.asarray(keys, dtype=np.float64)):
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return {"phi_family": "identity_launch", "phi_id": h.hexdigest()[:16],
            "phi_params": 0, "phi_bytes": 0}


def assert_identical_phi(rows: Dict[str, Dict[str, Any]]) -> str:
    """Every arm's ``phi`` row must be identical — **raise**, never warn."""
    from chlu.core.psi_readout import PhiMismatchError

    ids = {k: (v["phi_id"], v["phi_bytes"]) for k, v in rows.items()}
    uniq = sorted(set(ids.values()))
    if len(uniq) != 1:
        raise PhiMismatchError(
            "identical-phi invariant VIOLATED across B′ rival arms: "
            + "; ".join(f"{k}: {v}" for k, v in sorted(ids.items())))
    return uniq[0][0]


__all__ = [
    "DTYPE_BYTES", "TTT_MINI_BATCH", "TwoSidedLedger", "LedgerError",
    "matched_table_rows", "table_ledger", "head_width_for_budget",
    "clu_two_sided_ledger", "phi_row", "assert_identical_phi",
]
