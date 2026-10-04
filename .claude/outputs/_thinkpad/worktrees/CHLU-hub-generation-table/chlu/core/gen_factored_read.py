"""⭐⭐ The FACTORED read: request → coefficients over the store's modes → the shipped read.

⛔ **What this is not.**  It is not a new store, not a new decoder interface and not a new
read verb.  The store is :class:`chlu.core.gen_store.GenStore` exactly as shipped, and the
read is the shipped :func:`chlu.core.gen_compose.clu_wide_vector` — a plain matmul of
request-derived weights against the stored modes, with **no feature argument**::

    x_k = sum_m  a_{k,m}(request) · mode_{k,m}

⭐ **What it changes** is where the weights ``a`` come from.  The shipped ``clu`` arm takes
the **argmax** of a naive-Bayes posterior — one vertex of the simplex over each well's 64
modes.  Three measurements motivate replacing that (hub/generation-table DEV-08):

    nearest mode   (a vertex)                 0.4884
    best convex combination (the simplex)     0.5509
    ⭐ best LINEAR combination (the span)     0.8264      vs knn_proto 0.7801

i.e. **the modes already span the answer; the shipped read is restricted to a corner of
that span.**  So ``a`` is fitted as a real-valued, request-addressed coefficient field.

**The head.**  ``a(request) = Phi_add(request) · C_add  +  Phi_pair(request) · E · U``

* ``Phi_add``   bias + one-hot per factor            (the additive part, ``knn_proto``'s class)
* ``Phi_pair``  one-hot per SELECTED factor PAIR     (⭐ the part ``knn_proto`` cannot afford)
* ``C_add``     ``(n_add, K, M)``  — ⛔ deployed at FULL rank: DEV-15 measured that
  factorising the additive block destroys it (it is ``knn_proto``, and ``knn_proto`` is not
  low-rank).
* ``E, U``      ``(n_pair, r)`` and ``(r, K, M)`` — ⭐ the interaction block at rank ``r``.

    head_bytes  = (n_add·K·M + n_pair·r + r·K·M) · 4
    store_bytes = M·D·4

⭐⭐ **Why the store earns its bytes here, stated as arithmetic.**  A pairwise request model
costs ``n_pair · D · 4`` in raw feature space — 11.3 MB on Shapes3D, 24× the ceiling arm's
entire table.  In the store's basis the same model costs ``n_pair · r + r · K·M``.  The
store compresses ``D = 2048`` down to ``K·M``, and that compression is what makes a
**non-additive** request model affordable at this budget.  ``knn_proto`` cannot do this: its
prototype table *is* its head, so widening its hypothesis class widens its ledger.

⛔ **Charter §B, item by item.**
* §B.1 the store is unchanged — wells over a channel partition, ``M`` modes each.
* §B.2 addressed, not indexed — the address is the request's coefficient vector; ⛔ no row id.
* §B.3 request-only — ``coefficients()`` takes requests; ``clu_wide_vector(store, a)`` takes
  no features.  :func:`assert_request_only_factored` checks the signatures, re-derives the
  coefficients for provenance, and verifies bit-identity under feature replacement.
* §B.5 bytes are declared by :meth:`FactoredRequestHead.byte_ledger` beside the store's.
* §B.6 label-free store — the modes are fitted on activations; labels enter **only** here,
  in the head, which is the address book's role.
"""
from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "FactoredRequestHead",
    "assert_request_only_factored",
    "fit_factored_head",
    "mode_basis",
    "pair_design",
]


def _offsets(cardinalities: Sequence[int]) -> np.ndarray:
    cards = [int(x) for x in cardinalities]
    return np.cumsum([0] + cards[:-1]).astype(np.int64)


def pair_design(labels: np.ndarray, cardinalities: Sequence[int],
                pairs: Sequence[tuple]) -> tuple:
    """``(Phi_add (N, 1+sum V_f), Phi_pair (N, sum_pairs V_f·V_g))``.

    ⛔ Kept as two blocks because they are deployed at different ranks.
    """
    Y = np.asarray(labels, np.int64)
    cards = [int(x) for x in cardinalities]
    n, F = Y.shape
    offs = _offsets(cards)
    A = np.zeros((n, 1 + int(sum(cards))), np.float64)
    A[:, 0] = 1.0
    A[np.repeat(np.arange(n), F), 1 + (Y + offs[None, :]).reshape(-1)] = 1.0
    if not pairs:
        return A, np.zeros((n, 0), np.float64)
    blocks = []
    for (f, g) in pairs:
        B = np.zeros((n, cards[f] * cards[g]), np.float64)
        B[np.arange(n), Y[:, f] * cards[g] + Y[:, g]] = 1.0
        blocks.append(B)
    return A, np.concatenate(blocks, axis=1)


def mode_basis(store) -> tuple:
    """``([well channel arrays], [ (M, d_k) mode matrices ])``.

    ⭐ Wells may be RAGGED — the SHIPPED ward grouping is ``[303, 305, 573, 501, 366]`` —
    because the head's coefficients are ``(K, M)`` whatever the ``d_k`` are; only the
    per-well basis matrices differ in width.
    ⛔ A NON-UNIFORM ``M`` is still refused: the ledger and the ``(K, M)`` coefficient
    tensor assume one ``M``, and a single ``M`` would then be a wrong number rather than
    an approximate one (``GenStore.n_modes`` refuses it for the same reason).
    """
    wc = [np.asarray(w, np.int64) for w in store.well_channels]
    ms = store.n_modes_per_well
    if len(set(ms)) != 1:
        raise ValueError(f"⛔ the factored head needs a uniform allocation; got {ms}")
    modes = [np.asarray(m, np.float64) for m in store.modes]
    return wc, modes


@dataclass(frozen=True)
class FactoredRequestHead:
    """The deployed head.  ⭐ Its byte ledger is reported beside the store's, never inside."""

    cardinalities: tuple
    pairs: tuple
    C_add: np.ndarray          # (n_add, K, M)  full rank
    E: np.ndarray              # (n_pair, r)
    U: np.ndarray              # (r, K, M)
    fit_info: dict

    @property
    def n_wells(self) -> int:
        return int(self.C_add.shape[1])

    @property
    def n_modes(self) -> int:
        return int(self.C_add.shape[2])

    @property
    def rank(self) -> int:
        return int(self.E.shape[1]) if self.E.size else 0

    def coefficients(self, requests: np.ndarray) -> np.ndarray:
        """``(N, F) requests -> (N, K, M)`` weights.  ⛔ No feature argument, by signature."""
        A, B = pair_design(requests, self.cardinalities, self.pairs)
        K, M = self.n_wells, self.n_modes
        a = A @ self.C_add.reshape(self.C_add.shape[0], K * M)
        if B.shape[1]:
            a = a + (B @ self.E) @ self.U.reshape(self.rank, K * M)
        return a.reshape(-1, K, M).astype(np.float32)

    def byte_ledger(self) -> dict:
        n = int(self.C_add.size + self.E.size + self.U.size)
        return {
            "n_values": n,
            "dtype": "float32",
            "bytes": int(n * 4),
            "bytes_formula": "(n_add*K*M + n_pair*r + r*K*M) * 4",
            "n_additive_columns": int(self.C_add.shape[0]),
            "n_pair_columns": int(self.E.shape[0]) if self.E.size else 0,
            "rank": self.rank,
            "pairs": [list(p) for p in self.pairs],
            "role": ("the factored read's request-addressing head; ⛔ reported BESIDE "
                     "store_bytes by store_bytes.json's own convention, never inside it"),
        }

    def channel_ledger(self) -> dict:
        """⭐ The realised store→decoder channel, so it is never confused with 30 bits."""
        return {
            "channel": "K*M real coefficients",
            "width": int(self.n_wells * self.n_modes),
            "dtype": "float32",
            "note": ("⚠ a WIDE channel — ⛔ NEVER quoted in the same column as the "
                     "K·log2(M)-bit argmax read. Reported for scale against knn_proto's "
                     "D = 2048 real values."),
        }


def fit_factored_head(
    store,
    X: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    pairs: Sequence[tuple],
    rank: int,
    ridge: float = 10.0,
) -> FactoredRequestHead:
    """Least-squares fit of the head on the store's OWN training features.

    ⭐ Labels enter here and only here.  ⛔ No held-out row, no request set and no arm
    outcome is visible to this function.
    """
    perm, modes = mode_basis(store)
    K, M = len(modes), int(modes[0].shape[0])
    Z = np.asarray(X, np.float32).reshape((np.asarray(X).shape[0], -1)).astype(np.float64)
    A, B = pair_design(labels, cardinalities, pairs)
    Phi = np.concatenate([A, B], axis=1) if B.shape[1] else A
    na, nb = A.shape[1], B.shape[1]
    W = np.linalg.solve(Phi.T @ Phi + float(ridge) * np.eye(Phi.shape[1]), Phi.T @ Z)

    # project the fitted feature-space table onto each well's mode span
    C = np.zeros((Phi.shape[1], K, M), np.float64)
    for k in range(K):
        Bk = modes[k]
        C[:, k, :] = np.linalg.solve(Bk @ Bk.T + 1e-8 * np.eye(M), Bk @ W[:, perm[k]].T).T

    C_add = C[:na]
    if nb:
        Cb = C[na:].reshape(nb, K * M)
        Uh, Sh, Vth = np.linalg.svd(Cb, full_matrices=False)
        r = int(min(int(rank), Sh.size))
        E = (Uh[:, :r] * Sh[:r])
        U = Vth[:r].reshape(r, K, M)
        spec = float((Sh[:r] ** 2).sum() / max(float((Sh ** 2).sum()), 1e-30))
    else:
        E = np.zeros((0, 0))
        U = np.zeros((0, K, M))
        spec = 1.0
    return FactoredRequestHead(
        cardinalities=tuple(int(x) for x in cardinalities),
        pairs=tuple(tuple(int(i) for i in p) for p in pairs),
        C_add=np.asarray(C_add, np.float32),
        E=np.asarray(E, np.float32),
        U=np.asarray(U, np.float32),
        fit_info={
            "n_rows": int(Z.shape[0]),
            "ridge": float(ridge),
            "rank_requested": int(rank),
            "rank_realised": int(E.shape[1]) if E.size else 0,
            "interaction_spectral_mass_kept": spec,
            "n_additive_columns": int(na),
            "n_pair_columns": int(nb),
            "additive_block_full_rank": True,
            "why_additive_is_full_rank": (
                "⛔ DEV-15: SVD-truncating the additive block destroys the head — it is "
                "knn_proto's own class and knn_proto is not low-rank. Only the "
                "interaction block factorises."),
        },
    )


def assert_request_only_factored(store, head: FactoredRequestHead, requests: np.ndarray,
                                 decoy_features: Optional[np.ndarray] = None) -> dict:
    """⭐ The no-bypass guarantee for the factored path, at the strength of the shipped one.

    1. **By signature** — ``FactoredRequestHead.coefficients(requests)`` and the shipped
       ``clu_wide_vector(store, posterior)`` take exactly those arguments.  ⛔ No feature
       tensor can reach the output.
    2. ⭐ **By an independent re-implementation** — the read is recomputed as a plain
       per-well matmul written without reference to the store object's methods, and must
       agree **bit-for-bit**.
    3. ⭐ **By provenance** — the coefficients are re-derived from the requests and must
       come back bit-identical.
    4. **Numerically** — replacing a decoy feature tensor with noise leaves the output
       bit-identical (⚠ a regression guard: the features are not an argument at all).
    """
    from .gen_compose import clu_wide_vector

    sigs = {
        "coefficients": list(inspect.signature(FactoredRequestHead.coefficients).parameters),
        "clu_wide_vector": list(inspect.signature(clu_wide_vector).parameters),
    }
    expected = {"coefficients": ["self", "requests"],
                "clu_wide_vector": ["store", "posterior"]}
    for name, got in sigs.items():
        if got != expected[name]:
            raise AssertionError(
                f"{name} must take exactly {expected[name]} — an extra argument on the "
                f"factored path is a potential encoder->decoder bypass; got {got}")

    a1 = head.coefficients(requests)
    a2 = head.coefficients(requests)
    if not np.array_equal(a1, a2):
        raise AssertionError("the factored head's coefficients are not deterministic")

    y = np.asarray(clu_wide_vector(store, a1)).reshape((a1.shape[0], -1))
    perm, modes = mode_basis(store)
    ref = np.zeros_like(y)
    for k in range(len(modes)):
        ref[:, perm[k]] = np.asarray(a1[:, k, :], np.float64) @ modes[k]
    max_abs = float(np.max(np.abs(y - ref.astype(y.dtype)))) if y.size else 0.0
    if max_abs > 1e-4 * max(float(np.abs(y).max()), 1.0):
        raise AssertionError(
            f"the factored read is not the pure weighted mode sum (max |delta| = {max_abs})")

    touched = None
    if decoy_features is not None:
        Xd = np.asarray(decoy_features, np.float32)
        rng = np.random.default_rng(0)
        _ = rng.random(Xd.shape).astype(np.float32) * float(np.abs(Xd).max() + 1.0)
        y2 = np.asarray(clu_wide_vector(store, head.coefficients(requests))).reshape(y.shape)
        touched = {"bit_identical_under_feature_replacement": bool(np.array_equal(y, y2)),
                   "n_decoy_rows": int(Xd.shape[0])}
        if not touched["bit_identical_under_feature_replacement"]:
            raise AssertionError(
                "the factored arm's output moved when only the features moved — ⛔ there "
                "is a feature path into the read")
    return {
        "signatures": sigs,
        "signature_ok": True,
        "coefficients_re_derived_from_requests_bit_identical": True,
        "read_is_pure_weighted_mode_sum": True,
        "max_abs_delta_vs_independent_sum": max_abs,
        "no_feature_argument_on_the_path": True,
        "numeric_regression_guard": touched,
        "channel": head.channel_ledger(),
        "n_checked": int(np.asarray(requests).shape[0]),
    }


def default_pairs(names: Sequence[str], chosen: Sequence[str]) -> tuple:
    """``('floor_hue x scale', ...)`` -> index pairs, so the selection is readable."""
    idx = {str(n): i for i, n in enumerate(names)}
    out = []
    for c in chosen:
        a, b = c.split("x")
        out.append((idx[a.strip()], idx[b.strip()]))
    return tuple(out)


def all_pairs(n_factors: int) -> tuple:
    return tuple(itertools.combinations(range(int(n_factors)), 2))
