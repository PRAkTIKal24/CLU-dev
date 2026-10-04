"""hub/store-content — the two mandatory controls.

⭐ ``basis_head`` is the LIKE-FOR-LIKE (w3-hub-common §1): the wave-2 object, a
fitted basis over frozen encoder features plus a **request-conditioned linear
regression head**, at matched bytes.  ``knn_proto`` is the second row.
⛔ Neither is optional and ``build_scoreboard`` refuses a table that is missing either.

basis_head, spelled out
-----------------------
*The basis.*  Per well ``k``, a dictionary ``A_k`` of ``M`` vectors of width ``d_k``
over the **penultimate** map — ``M·D·4 = 524,288 B``, the same ledger as the store.
⭐ The atoms are the same stored train rows the CLU arm uses.  That is not a shortcut:
w3-hub-common §1 records that on this object **medoid atoms tie fitted atoms exactly**,
so a k-means refit would move the ledger's contents without moving its number, and
holding them identical makes the comparison turn on the HEAD and the READ — which is
exactly what §1 says the wave-2 finding was about.

*The head.*  ``c(request) = φ(request) · W_k`` where ``φ`` is the concatenated
one-hot of the request's factor values (``Σ_f V_f = 57`` for Shapes3D).  ``W_k`` is
fitted by ridge in two stages: each train row's coordinates in the dictionary are
found by ridge projection onto ``A_k``, then those coefficients are ridge-regressed
on ``φ``.  ⛔ No simplex, no non-negativity — w3-hub-common §4.5 measured the store's
own coefficients at ~47 % negative and 0.00 % probability vectors.

*The read.*  ``x_k = c(request) · A_k`` — the signed linear combination.  ⭐ This is
the read verb w3-hub-common §4.4 calls the largest measured effect in the programme,
so the control gets it and the comparison is not a read-verb comparison in disguise.

*Its head's bytes.*  ``K · 57 · M · 4``.  At M=64, K=5 that is 72,960 B against the
CLU address book's ``(K·N + 57·K·N)·4 = 74,240 B`` — within 1.8 %.  Both are excluded
from ``store_bytes`` by the standing convention and both are reported anyway.
"""
from __future__ import annotations

import numpy as np

from chlu.core.gen_store import (
    channel_correlation, correlation_grouping, dead_channels,
    well_channels_from_labels,
)
from chlu.core.gen_landscape import select_atom_rows, store_byte_ledger

from content import BUDGET_BYTES

K_WELLS = 5
#: ⭐ PINNED IN ADVANCE (w3-hub-common §3: pin the store fit COMPLETELY).
RIDGE_PROJ = 1e-2      # ridge on the dictionary projection
RIDGE_HEAD = 1e-1      # ridge on the request -> coefficient regression
ATOM_SELECTION = "kmeanspp"


def onehot(requests: np.ndarray, cardinalities) -> np.ndarray:
    """(n, F) factor-index requests -> (n, Σ_f V_f) one-hot design matrix.

    ⛔ A ``-1`` (factor not named) contributes an all-zero block rather than a row of
    the identity, so an unnamed factor is *absent* from the regression instead of
    silently meaning "value 0".
    """
    R = np.asarray(requests, np.int64)
    cards = [int(c) for c in cardinalities]
    out = np.zeros((R.shape[0], sum(cards)), np.float32)
    off = 0
    for f, c in enumerate(cards):
        ok = R[:, f] >= 0
        out[np.flatnonzero(ok), off + R[ok, f]] = 1.0
        off += c
    return out


def _ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """``argmin_W ||XW - Y||² + lam·||W||²`` via the normal equations (float64)."""
    Xd = np.asarray(X, np.float64)
    G = Xd.T @ Xd
    G[np.diag_indices_from(G)] += float(lam) * max(1.0, float(np.trace(G)) / max(G.shape[0], 1))
    return np.linalg.solve(G, Xd.T @ np.asarray(Y, np.float64)).astype(np.float32)


class BasisHead:
    """The wave-2 object: a fitted basis + a request-conditioned linear head."""

    def __init__(self, Z: np.ndarray, requests: np.ndarray, cardinalities,
                 n_atoms: int, seed: int = 0, log=print):
        Z = np.asarray(Z, np.float32)
        self.D = int(Z.shape[1])
        self.M = int(n_atoms)
        self.cards = [int(c) for c in cardinalities]

        corr = channel_correlation(Z)
        lab, ginfo = correlation_grouping(corr, K_WELLS, method="auto",
                                          dead=dead_channels(Z), return_info=True)
        self.well_channels = [np.asarray(w, np.int64)
                              for w in well_channels_from_labels(np.asarray(lab, np.int64),
                                                                 K_WELLS)]
        self.grouping = {"well_sizes": [int(w.size) for w in self.well_channels],
                         "K": K_WELLS, "method_used": ginfo.get("method_used"),
                         "n_dead_channels": int(ginfo.get("n_dead_channels", -1))}

        rows = select_atom_rows(Z, self.M, ATOM_SELECTION, int(seed))
        self.atom_rows = np.asarray(rows, np.int64)
        self.A = [np.asarray(Z[np.ix_(rows, w)], np.float32) for w in self.well_channels]

        Phi = onehot(requests, self.cards)
        self.W = []
        for k, w in enumerate(self.well_channels):
            Ak = np.asarray(self.A[k], np.float64)            # (M, d_k)
            Zk = np.asarray(Z[:, w], np.float64)              # (n, d_k)
            # stage 1: each row's coefficients in the dictionary (ridge projection)
            C = _ridge(Ak.T, Zk.T, RIDGE_PROJ).T              # (n, M)
            # stage 2: request -> coefficients
            self.W.append(_ridge(Phi, C, RIDGE_HEAD))         # (57, M)
        led = store_byte_ledger(self.M, self.grouping["well_sizes"])
        self.ledger = {**led, "deployed_bytes": int(led["store_bytes"]),
                       "budget_bytes": BUDGET_BYTES,
                       "what": f"basis_head: {self.M} basis vectors x {self.D} float32"}
        if int(led["store_bytes"]) != BUDGET_BYTES:
            raise AssertionError(
                f"basis_head ledger {led['store_bytes']:,} != {BUDGET_BYTES:,}")
        self.head_bytes = int(sum(np.asarray(x).size for x in self.W) * 4)
        log(f"  [basis_head] M={self.M} D={self.D} wells {self.grouping['well_sizes']} "
            f"store {led['store_bytes']:,} B  head {self.head_bytes:,} B")

    def read(self, requests: np.ndarray) -> np.ndarray:
        Phi = onehot(requests, self.cards)
        x = np.zeros((Phi.shape[0], self.D), np.float32)
        for k, w in enumerate(self.well_channels):
            x[:, w] = (Phi @ self.W[k]) @ self.A[k]
        return x

    def coefficient_report(self, requests: np.ndarray) -> dict:
        """⭐ The §4.5 check: is the head using the basis as a SIGNED span?"""
        Phi = onehot(requests, self.cards)
        neg, l1, simplex = [], [], []
        for k in range(len(self.A)):
            C = Phi @ self.W[k]
            neg.append(float((C < 0).mean()))
            l1.append(float(np.abs(C).sum(axis=1).mean()))
            simplex.append(float(((C >= 0).all(axis=1)
                                  & (np.abs(C.sum(axis=1) - 1.0) < 1e-3)).mean()))
        return {"frac_negative": float(np.mean(neg)),
                "mean_l1_per_well": float(np.mean(l1)),
                "frac_probability_vectors": float(np.mean(simplex))}


class KnnProto:
    """The second control: per-(factor, value) prototypes, composed additively.

    ⛔ ``FactorPrototypes`` is the pin's own object, called rather than
    re-implemented, so this row is arithmetically the ``knn_proto`` every other
    table carries.  Its ledger does NOT match the budget and that is declared, not
    hidden — the scoreboard requires a ``budget_note`` for exactly this case.
    """

    def __init__(self, grid, features: np.ndarray, rows: np.ndarray, log=print):
        from chlu.core.gen_request import FactorPrototypes

        self.grid = grid
        self.proto = FactorPrototypes.fit(grid, features, rows)
        b = int(self.proto.n_bytes)
        self.ledger = {
            "deployed_bytes": b, "budget_bytes": BUDGET_BYTES,
            "what": "knn_proto: (Sigma_f V_f + 1) prototypes x D float32",
            "budget_note": (
                f"DECLARED MISMATCH: {b:,} B against the {BUDGET_BYTES:,} B budget "
                f"({b / BUDGET_BYTES:.3f}x). The prototype table is this arm's whole "
                "deployed object and it is not byte-matchable at tolerance 0 without "
                "changing what the arm IS. It is the SECOND control for this reason; "
                "basis_head is the like-for-like."),
            **self.proto.byte_ledger(),
        }
        log(f"  [knn_proto] {self.proto.n_prototypes} prototypes x {self.proto.dim} "
            f"= {b:,} B ({b / BUDGET_BYTES:.3f}x budget)")

    def read(self, requests: np.ndarray) -> np.ndarray:
        """(n, F) factor-index requests -> (n, D), via the declared additive rule."""
        R = np.asarray(requests, np.int64)
        out = np.repeat(self.proto.mu_global[None, :].astype(np.float32), R.shape[0], 0)
        for f in range(R.shape[1]):
            ok = R[:, f] >= 0
            if not ok.any():
                continue
            P = np.asarray(self.proto.protos[f], np.float32)
            out[ok] += P[R[ok, f]] - self.proto.mu_global[None, :]
        return out
