"""The interaction block, at a cost that does NOT scale with the number of atoms.

THE BYTE PROBLEM THIS SOLVES.  The factored read's head is

    a(request) = Phi_add . C_add  +  (Phi_pair . E) . U
    head bytes = (n_add . A  +  n_pair . r  +  r . A) . 4        A = sum_k M_k

The ``r . A`` term is the problem: every unit of interaction rank costs ``A`` floats, so
at the atom counts that actually work (A ~ 800-1400, measured: the arm's score is
monotone increasing in A) a single rank unit costs more than three atoms.  The budget
forces a choice between SPAN and INTERACTION, and the measured frontier says span wins -
which leaves the arm a lossy factorisation of ``knn_proto``'s own additive model class,
able to tie it but structurally unable to beat it.

THE OBSERVATION.  ``U`` is an ``(r, A)`` basis in ATOM-coefficient space, and it is
stored purely so the interaction can name directions there.  But the head ALREADY stores
an ``(n_add, A)`` basis in exactly that space: ``C_add``, the additive block.  Its 52 rows
are the atom-space directions that the six factors move along - which is precisely where
a factor-pair interaction lives.

    => take  U := C_add  and store NOTHING for it.

    a(request) = Phi_add . C_add  +  (Phi_pair . E) . C_add  =  (Phi_add + Phi_pair . E) . C_add

    head bytes = (n_add . A  +  n_pair . n_add) . 4       <- INDEPENDENT OF A

The interaction becomes a ``(n_pair, n_add)`` remap of the pair design onto the additive
design's own columns.  At ``n_pair = 160, n_add = 52`` that is 33,280 B whatever ``A`` is,
against 3,968 B *per rank unit* at ``A = 832``.  The freed bytes go back into atoms.

WHAT IT GIVES UP, stated plainly.  The interaction can no longer point anywhere in the
A-dimensional coefficient space - only inside the ``n_add``-dimensional subspace the
additive block already spans.  That is a real restriction and it is the reason this is a
measurement and not an obvious improvement.  ``ridge_pair`` still regularises ``E``.

  - Labels enter here and ONLY here (charter B.6, the address book).  The dictionary is
    fitted label-free elsewhere; nothing in this file touches the landscape or the write.
  - ``coefficients`` takes requests and nothing else, by signature (charter B.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gen_dict_fit import _pair_blocks, reduced_additive_design

__all__ = ["SharedBasisHead", "fit_shared_basis_head", "shared_basis_head_bytes"]


def shared_basis_head_bytes(n_add: int, n_pair: int, n_atoms: int) -> int:
    """``(n_add*A + n_pair*n_add) * 4``.  The second term does not involve ``A``."""
    return int((int(n_add) * int(n_atoms) + int(n_pair) * int(n_add)) * 4)


@dataclass(frozen=True)
class SharedBasisHead:
    """``C_add`` (n_add, A) and ``E`` (n_pair, n_add).  There is no ``U`` to store."""

    cardinalities: tuple
    pairs: tuple
    modes_per_well: tuple
    reduced: bool
    C_add: np.ndarray            # (n_add, A)
    E: np.ndarray                # (n_pair, n_add)
    fit_info: dict

    @property
    def n_atoms(self) -> int:
        return int(self.C_add.shape[1])

    @property
    def rank(self) -> int:
        """The interaction's realised rank -- at most ``n_add``, and free in bytes."""
        return int(self.E.shape[1]) if self.E.size else 0

    def coefficients(self, requests: np.ndarray) -> list:
        """``(N, F) requests -> [ (N, M_k) ]``.  No feature argument, by signature."""
        A = reduced_additive_design(requests, self.cardinalities, self.reduced)
        if self.E.size:
            B = _pair_blocks(requests, self.cardinalities, self.pairs)
            if B.shape[1]:
                A = A + B @ self.E
        a = A @ self.C_add
        out, s = [], 0
        for M in self.modes_per_well:
            out.append(np.asarray(a[:, s:s + int(M)], np.float32))
            s += int(M)
        return out

    def byte_ledger(self) -> dict:
        n = int(self.C_add.size + self.E.size)
        return {
            "n_values": n, "dtype": "float32", "bytes": int(n * 4),
            "bytes_formula": "(n_add*A + n_pair*n_add) * 4,  A = sum_k M_k "
                             "-- the interaction term does NOT scale with A",
            "n_additive_columns": int(self.C_add.shape[0]),
            "n_pair_columns": int(self.E.shape[0]) if self.E.size else 0,
            "rank": self.rank, "n_atoms_A": self.n_atoms,
            "modes_per_well": [int(x) for x in self.modes_per_well],
            "additive_design": ("reduced (one reference level dropped per factor)"
                                if self.reduced else "full one-hot"),
            "pairs": [list(p) for p in self.pairs],
            "interaction_basis": "SHARED with the additive block (U := C_add, stored once)",
            "role": "the factored read's request-addressing head; reported BESIDE "
                    "store_bytes, never inside it",
        }

    def channel_ledger(self) -> dict:
        return {"channel": "sum_k M_k real coefficients", "width": self.n_atoms,
                "dtype": "float32",
                "note": "a WIDE channel - never quoted in the 30-bit column."}


def fit_shared_basis_head(
    dicts: Sequence[np.ndarray],
    perm: Sequence[np.ndarray],
    X: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    pairs: Sequence[tuple],
    ridge_add: float = 10.0,
    ridge_pair: float = 100.0,
    reduced: bool = True,
) -> SharedBasisHead:
    """Fit ``C_add`` and the shared-basis interaction remap ``E``.

    Step 1 is the same ridge regression of features on the full design that
    ``fit_factored_head_v2`` does, and the same per-well projection onto the dictionary,
    giving ``C_add`` (n_add, A) and ``C_pair`` (n_pair, A).

    Step 2 is the only new arithmetic: solve, in the FEATURE-space metric so the error
    minimised is the error that is decoded,

        min_E  || (C_pair - E . C_add) L ||_F      with  L L^T = blockdiag_k(B_k B_k^T)

    i.e. ``E = C_pair L (C_add L)^+``.  Whitening by ``L`` matters here for the same
    reason it does in a rank truncation: two coefficient directions of equal norm do not
    cost the same number of features.

    ``rank`` is not a parameter: the interaction is exactly ``n_add``-dimensional and
    costs the same either way, so there is nothing to truncate and nothing to choose.
    """
    Z = np.asarray(X, np.float64).reshape((np.asarray(X).shape[0], -1))
    Adesign = reduced_additive_design(labels, cardinalities, reduced)
    Bdesign = _pair_blocks(labels, cardinalities, pairs)
    na, nb = Adesign.shape[1], Bdesign.shape[1]
    Phi = np.concatenate([Adesign, Bdesign], 1) if nb else Adesign
    lam = (np.concatenate([np.full(na, float(ridge_add)), np.full(nb, float(ridge_pair))])
           if nb else np.full(na, float(ridge_add)))
    W = np.linalg.solve(Phi.T @ Phi + np.diag(lam), Phi.T @ Z)        # (n_cols, D)

    Ms = [int(np.asarray(d).shape[0]) for d in dicts]
    Atot = int(sum(Ms))
    C = np.zeros((Phi.shape[1], Atot), np.float64)
    L = np.zeros((Atot, Atot), np.float64)
    s = 0
    for k, d in enumerate(dicts):
        Bk = np.asarray(d, np.float64)
        Gk = Bk @ Bk.T
        M = Ms[k]
        C[:, s:s + M] = np.linalg.solve(Gk + 1e-8 * np.eye(M), Bk @ W[:, perm[k]].T).T
        ev, V = np.linalg.eigh(Gk + 1e-10 * np.eye(M))
        L[s:s + M, s:s + M] = (V * np.sqrt(np.maximum(ev, 0.0))[None, :]) @ V.T
        s += M

    C_add = C[:na]
    if nb:
        C_pair = C[na:]
        # E = C_pair L (C_add L)^+  -- least squares in the feature-space metric
        E, *_ = np.linalg.lstsq(np.asarray((C_add @ L).T, np.float64),
                                np.asarray((C_pair @ L).T, np.float64), rcond=None)
        E = np.asarray(E.T, np.float64)                              # (n_pair, n_add)
        resid = float(np.linalg.norm((C_pair - E @ C_add) @ L))
        total = float(np.linalg.norm(C_pair @ L))
        kept = 1.0 - (resid / max(total, 1e-30)) ** 2
    else:
        E = np.zeros((0, 0))
        kept = 1.0

    return SharedBasisHead(
        cardinalities=tuple(int(x) for x in cardinalities),
        pairs=tuple(tuple(int(i) for i in p) for p in pairs),
        modes_per_well=tuple(Ms), reduced=bool(reduced),
        C_add=np.asarray(C_add, np.float32), E=np.asarray(E, np.float32),
        fit_info={
            "n_rows": int(Z.shape[0]), "ridge_add": float(ridge_add),
            "ridge_pair": float(ridge_pair),
            "rank_requested": int(na), "rank_realised": int(E.shape[1]) if E.size else 0,
            "interaction_spectral_mass_kept": float(kept),
            "metric_correct_truncation": True,
            "n_additive_columns": int(na), "n_pair_columns": int(nb),
            "reduced_additive_design": bool(reduced),
            "additive_block_full_rank": True,
            "interaction_basis": "shared with the additive block (U := C_add)",
        },
    )
