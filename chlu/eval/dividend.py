"""The dynamics dividend and its harness-native controls (charter §2.1).

    **dynamics dividend = (full CLU) - (its own settle-deleted / matched-bytes
    launder)**, on the same harness, the same bytes and the same ``phi``.

It is measured on **every** full-system experiment from C2W1 onward, and it is
the sole KPI of the memory gym. The paper exists when the dividend is positive,
robust and multi-seed on an admissible task; if no such task exists at this
weight class, **that is the program's most important finding** and is reported as
such — it is the null hypothesis of the whole program.

⭐ **A dividend of ~0 or negative at v0 is the expected, honest starting line and
falsifies nothing.** A *positive* v0 dividend is suspicious and goes through all
three controls plus a seed re-run before it is written down.

**Why the sign is structurally predictable** (`controller-doctrine` Prop D1/D2):
under the store's own margin certificate the settle and the same-keys arg-min
launder **agree on every query inside a certified ball**, so the disagreement
mass obeys ``D <= U`` and ``D = 0 => dividend <= 0``. For equal-depth,
equal-width, symmetrically-placed wells ``D = 0`` *exactly* — which is why w26's
same-keys launder beat CLU 6/6 structurally rather than accidentally. The
dividend lives in ``B_i \\ Vor_i``, and that set is created by **geometric
heterogeneity**.

**The three harness-native controls** (not optional add-ons):

``settle_deleted_launder``
    The settle deleted and nothing else changed: arg-min over the store's own
    admitted keys ``c_i``, returning the stored payload. This is w26's same-keys
    launder promoted to a permanent harness fixture, and it is monitor #2.
``same_keys_null``
    The same keys, content destroyed (payloads permuted). Whatever survives is
    address structure, not content — the "did the score come from the store at
    all" control.
``blank_store_control``
    The identical system with **nothing stored**. Anything it scores above the
    empirical chance rate is a leak (N68: blanks 0.992-1.000).

Plus, on the doctrine's request (I-12), a **second launder**: the best
*shared-metric* arg-min, so a future positive dividend cannot be dismissed as
"you beat a weaker metric" — Prop D3's mechanism is metric-shaped.

⚠ **Co-owned file (C2W1).** ``full-clu-harness`` lands these signatures and the
system-side implementations; ``memory-gym-v0`` lands the gym-side callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class ByteAccount:
    """Matched-bytes accounting. Everything is float32 unless stated.

    ``full`` counts the learned ``V_theta`` **plus** the retained address
    codebook (the codebook is not free and is not hidden); ``launder`` counts the
    key table plus the payload table it is allowed to keep.
    """

    full_bytes: int
    launder_bytes: int
    breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """``full / launder``. A dividend claimed at ratio > 1 is not matched."""
        return float(self.full_bytes) / max(float(self.launder_bytes), 1.0)

    def matched(self, tol: float = 0.05) -> bool:
        """Are the two within ``tol`` relative bytes."""
        return abs(self.ratio - 1.0) <= float(tol)

    def as_dict(self) -> dict:
        return {"full_bytes": int(self.full_bytes),
                "launder_bytes": int(self.launder_bytes),
                "ratio": self.ratio, "matched": self.matched(),
                "breakdown": dict(self.breakdown)}


@dataclass
class DividendReport:
    """One dividend measurement, with every control that must travel with it."""

    metric: str
    full: float
    launder: float
    dividend: float
    se: float = float("nan")
    n_seeds: int = 1
    controls: Dict[str, float] = field(default_factory=dict)
    bytes: Optional[ByteAccount] = None
    flags: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return {"metric": self.metric, "full": self.full, "launder": self.launder,
                "dividend": self.dividend, "se": self.se, "n_seeds": self.n_seeds,
                "controls": dict(self.controls),
                "bytes": (self.bytes.as_dict() if self.bytes else None),
                "flags": dict(self.flags), "notes": self.notes}

    def to_markdown(self) -> str:
        """The reported artifact, controls included — never the bare number."""
        lines = [
            f"**dividend ({self.metric}) = {self.dividend:+.4f}** "
            f"= full {self.full:.4f} - settle-deleted launder {self.launder:.4f}"
            + (f"  (se {self.se:.4f}, {self.n_seeds} seeds)"
               if np.isfinite(self.se) else f"  ({self.n_seeds} seed)"),
            "",
            "| control | value |",
            "|---|---|",
        ]
        for k, v in self.controls.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        if self.bytes is not None:
            lines.append(
                f"| bytes full / launder | {self.bytes.full_bytes} / "
                f"{self.bytes.launder_bytes} (ratio {self.bytes.ratio:.1f}x, "
                f"matched={self.bytes.matched()}) |"
            )
        if self.notes:
            lines += ["", self.notes]
        return "\n".join(lines)


def dividend(full: float, launder: float, *, metric: str = "accuracy",
             se: float = float("nan"), controls: Optional[dict] = None,
             bytes_account: Optional[ByteAccount] = None,
             flags: Optional[dict] = None) -> DividendReport:
    """``(full CLU) - (its own settle-deleted / matched-bytes launder)``.

    Higher-is-better metrics only; pass an already-sign-corrected value for
    error-like metrics and say so in ``metric``.
    """
    return DividendReport(
        metric=metric, full=float(full), launder=float(launder),
        dividend=float(full) - float(launder), se=float(se),
        controls=dict(controls or {}), bytes=bytes_account, flags=dict(flags or {}),
    )


def _argmin_assign(keys: np.ndarray, queries: np.ndarray,
                   metric_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    q = np.asarray(queries, dtype=float)
    k = np.asarray(keys, dtype=float)
    diff = q[:, None, :] - k[None, :, :]
    if metric_matrix is None:
        d2 = np.sum(diff * diff, axis=-1)
    else:
        M = np.asarray(metric_matrix, dtype=float)
        d2 = np.einsum("qkd,de,qke->qk", diff, M, diff)
    return np.argmin(d2, axis=1)


def settle_deleted_launder(keys: np.ndarray, payloads: np.ndarray,
                           queries: np.ndarray, *,
                           metric: str = "value") -> np.ndarray:
    """**The settle deleted.** Arg-min over the store's own keys -> stored value.

    Same bytes, same ``phi``, same admitted set — the *only* difference is that
    the dynamics have been removed. This is the control w26 measured beating CLU
    6/6 on both axes, and it is the one the charter binds the dividend to.

    Returns the retrieved payloads (``metric="value"``) or the assignment
    indices (``metric="assign"``).
    """
    idx = _argmin_assign(keys, queries)
    return idx if metric == "assign" else np.asarray(payloads, dtype=float)[idx]


def shared_metric_launder(keys: np.ndarray, payloads: np.ndarray,
                          queries: np.ndarray, metric_matrix: np.ndarray
                          ) -> np.ndarray:
    """Arg-min under a single shared ``O(d^2)`` Mahalanobis metric (doctrine I-12).

    The strongest launder that costs only a *shared* metric. A per-item-covariance
    launder is stronger still but is **not matched-bytes** (+``K d(d+1)/2``
    floats) and must be reported as such.
    """
    idx = _argmin_assign(keys, queries, metric_matrix)
    return np.asarray(payloads, dtype=float)[idx]


def same_keys_null(keys: np.ndarray, payloads: np.ndarray, queries: np.ndarray,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Same keys, **content destroyed** (payloads permuted).

    Isolates "how much of the score is address structure". A system that scores
    the same here as with the true payloads is reading its own addresses.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    pays = np.asarray(payloads, dtype=float)
    perm = rng.permutation(pays.shape[0])
    return settle_deleted_launder(keys, pays[perm], queries)


def blank_store_control(read_fn: Callable, queries, *,
                        chance: Optional[float] = None,
                        labels: Optional[np.ndarray] = None,
                        metric: str = "decode",
                        representation: str = "settled_point") -> Dict[str, float]:
    """Read an identically-configured store with **nothing in it**.

    ``chance`` defaults to the empirical marginal (not ``1/K``: a skewed label
    distribution makes ``1/K`` the wrong bar). Returns the score and its
    ``chance + 3 se`` bar — monitor #4's input.
    """
    pred = np.asarray(read_fn(queries))
    if labels is None:
        raise ValueError("blank_store_control needs labels to score a decode")
    labels = np.asarray(labels)
    score = float(np.mean(pred == labels))
    if chance is None:
        _, counts = np.unique(labels, return_counts=True)
        chance = float(np.max(counts) / labels.size)  # the empirical marginal
    n = max(labels.size, 1)
    se = float(np.sqrt(max(chance * (1.0 - chance), 1e-12) / n))
    return {"score": score, "chance": float(chance), "se": se,
            "bar": float(chance + 3.0 * se), "metric": metric,
            "representation": representation}


def trajectory_launder(psi: Callable, traj, state) -> Dict[str, float]:
    """Doctrine I-2: ``psi(traj)`` vs ``psi(q0)`` vs ``psi(q0, q*)``.

    The trajectory **contains ``q0 = phi(x)``**. Without this three-way split the
    trajectory pillar's first datum is uninterpretable: a psi over the raw buffer
    could be a classifier on the query embedding and nothing else.

    Returns the three read-outs as arrays under keys ``full``, ``q0_only`` and
    ``endpoints``; scoring them is the caller's job (the gym does it).
    """
    import jax.numpy as jnp

    full = psi(traj, state)
    q0_pt = jnp.concatenate([state.q0, state.p0], axis=-1)[:, None, :]
    q0_only = psi(jnp.broadcast_to(q0_pt, traj.shape), state)
    ends = jnp.concatenate(
        [q0_pt, jnp.concatenate([state.q_star, state.p_star], axis=-1)[:, None, :]],
        axis=1,
    )
    endpoints = psi(ends, state)
    return {"full": full, "q0_only": q0_only, "endpoints": endpoints}


def count_bytes(tree: Any) -> int:
    """Bytes of every inexact array in a PyTree (float32 => 4 B/param)."""
    import equinox as eqx
    import jax

    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return int(sum(int(np.asarray(x).size) for x in leaves) * 4)


def byte_account(system, keys: np.ndarray, payloads: np.ndarray) -> ByteAccount:
    """Byte accounting for a :class:`~chlu.core.clu_system.CluSystem` and the
    launder table it is measured against."""
    keys = np.asarray(keys)
    pays = np.asarray(payloads)
    v_bytes = int(system.store.n_bytes())
    code_bytes = int(keys.size * 4)
    launder = int((keys.size + pays.size) * 4)
    return ByteAccount(
        full_bytes=v_bytes + code_bytes, launder_bytes=launder,
        breakdown={"V_theta": v_bytes, "codebook_addresses": code_bytes,
                   "launder_keys": int(keys.size * 4),
                   "launder_payloads": int(pays.size * 4)},
    )


__all__ = [
    "ByteAccount", "DividendReport", "dividend", "settle_deleted_launder",
    "shared_metric_launder", "same_keys_null", "blank_store_control",
    "trajectory_launder", "count_bytes", "byte_account",
]
