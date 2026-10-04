"""⭐⭐ Fitting the store's dictionary **for the read verb that is actually used**.

⛔ **The defect this module addresses, stated as a mismatch.**
``gen_store.train_store`` fits the modes by a *nearest-mode* (vector-quantisation)
objective: it minimises ``sum_i min_m ||x_i - mode_m||^2``.  That is the right objective
for the SHIPPED read, which takes an **argmax** and emits one mode.  ⭐ But
``hub/generation-table`` DEV-08 measured that the argmax read is the largest single loss
in the programme, and the read that replaced it is the **linear span** of the modes
(``gen_compose.clu_wide_vector``).  The dictionary is therefore being fitted for a verb
the read no longer uses::

    fitted for :  x ~= mode_{m*}                  (one vertex)
    read as    :  x ~= sum_m a_m . mode_m         (the span)

=> **fit the dictionary for the span.**

⛔ **Why the answer is NOT "use PCA".**  Unconstrained, the span-optimal ``M``-dimensional
dictionary *is* the principal subspace, and the Head has ruled (**R4**, 2026-09-17) that
*"the PCA-affine dictionary is NOT adopted as a store ... a principal direction cannot be
decoded or deleted, and the deletion target needs atoms."*  That ruling is respected
here: ``pca_affine`` remains a **control only**.

⭐⭐ **What this module uses instead: NON-NEGATIVE atoms (NMF).**  The bridge's features
are post-ReLU - measured on seed 0's cache, ``min = 0.0`` with **51 % of entries exactly
zero**.  A non-negative dictionary is therefore the natural *parts* decomposition of this
representation, and it has exactly the properties charter §B and R4 demand and a
principal direction lacks:

* every atom is itself a **valid feature vector** (non-negative, in the data's own
  orthant) => it can be pushed through the frozen decoder and **looked at** - decodable;
* atoms are **parts, not a rotation** => removing one removes identifiable content -
  **deletable**, which is what §B.4's exact deletion needs;
* the NMF objective is ``min ||X - A W||^2`` over a *linear combination* of atoms => it
  is the **read verb's own objective**, unlike k-means;
* ⛔ it is label-free: it sees train activations and nothing else (§B.6).

⚠ The read still allows **negative** coefficients; only the ATOMS are constrained.  The
constraint is what keeps them atoms.  Its price - NMF spans strictly less than PCA at the
same ``M`` - is the price of being charter-legal, and it is reported, not hidden.

---

**The second thing this module fixes: the rank truncation was in the wrong metric.**
``gen_factored_read.fit_factored_head`` truncates the interaction block by an SVD of the
**coefficient** matrix ``C_b``.  That minimises coefficient-space Frobenius error - but
the deployed error lives in **feature** space, and the map from coefficients to features
is the (non-orthonormal) mode matrix ``B_k``.  Two coefficient directions of equal norm
do not cost the same number of features.  :func:`fit_factored_head_v2` truncates in the
**feature-space metric** (whitening by each well's Gram ``B_k B_k^T``), which is the same
byte ledger and a strictly better ``r``-rank approximation of the thing that is decoded.

**And the third: the additive design carried 6 redundant columns.**  ``bias + one-hot per
factor`` is rank ``1 + sum_f (V_f - 1) = 52`` on a 58-column design, so ``6 . A`` floats
of the head were structurally unidentifiable.  ``reduced=True`` drops one reference level
per factor - ⭐ an exactly equivalent parameterisation at 52 columns - and hands the
freed bytes to the interaction rank.  ⛔ This is a re-parameterisation, not a resolution
cut: DEV-20 measured that *reducing the resolution* of the additive block (a smooth
basis) destroys it.  Dropping a redundant level does not.

⛔ Nothing here edits a pinned harness file.  The store is ``gen_store.GenStore``; the
read is the shipped ``gen_compose.clu_wide_vector``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "DICTIONARY_METHODS",
    "RaggedHead",
    "allocate_atoms",
    "build_store_from_dicts",
    "fit_factored_head_v2",
    "head_bytes",
    "reduced_additive_design",
    "well_dictionary",
]

#: ⭐ ``kmeans``/``nmf`` are ARMS (learned, decodable, deletable).  Everything else is a
#: CONTROL: the three ``atoms_*`` are charter §C.1 (a dictionary of real stored rows),
#: ``pca_affine`` is R4's control-only row, ``random`` is the D-matched null and
#: ``mean`` is §C.2.
DICTIONARY_METHODS = (
    "kmeans", "nmf",
    "atoms_medoid", "atoms_kpp", "atoms_random",
    "pca_affine", "random", "mean",
)


# ---------------------------------------------------------------------------
# 1. the dictionaries
# ---------------------------------------------------------------------------

def _assign_chunked(Z: np.ndarray, cen: np.ndarray, block: int = 8192) -> np.ndarray:
    out = np.empty(Z.shape[0], np.int64)
    cn = (cen ** 2).sum(1)
    for s in range(0, Z.shape[0], block):
        e = min(s + block, Z.shape[0])
        out[s:e] = np.argmin(cn[None, :] - 2.0 * (Z[s:e] @ cen.T), 1)
    return out


def _kmeans(Z: np.ndarray, M: int, seed: int, n_iter: int = 25) -> np.ndarray:
    """k-means++ init + Lloyd.  ⭐ The *vertex* objective - what the shipped store fits."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    M = int(min(M, n))
    cen = np.empty((M, Z.shape[1]), np.float64)
    cen[0] = Z[rng.integers(n)]
    d2 = ((Z - cen[0]) ** 2).sum(1)
    for j in range(1, M):
        tot = float(d2.sum())
        p = (d2 / tot) if tot > 0 else None
        cen[j] = Z[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((Z - cen[j]) ** 2).sum(1))
    for _ in range(int(n_iter)):
        lab = _assign_chunked(Z, cen)
        for j in range(M):
            m = lab == j
            if m.any():
                cen[j] = Z[m].mean(0)
    return cen


def _nmf(Z: np.ndarray, M: int, seed: int, n_iter: int = 150, eps: float = 1e-9) -> np.ndarray:
    """Multiplicative-update NMF, ``Z ~= A W`` with ``A, W >= 0``; returns ``W (M, d)``.

    ⭐ The SPAN objective, with the atoms constrained to the data's own orthant.  ⛔ Not
    a rotation of the data: every row of ``W`` is a non-negative feature vector.
    Initialised from k-means centroids (clipped non-negative) so it is a strict
    refinement of the shipped dictionary and never a worse starting point.
    """
    rng = np.random.default_rng(seed + 7919)
    Zp = np.maximum(np.asarray(Z, np.float64), 0.0)
    M = int(min(M, Zp.shape[0]))
    W = np.maximum(_kmeans(Zp, M, seed, n_iter=10), 0.0) + eps
    A = np.maximum(rng.random((Zp.shape[0], M)) * 0.1, eps)
    for _ in range(int(n_iter)):
        WT = W.T
        A *= (Zp @ WT) / np.maximum(A @ (W @ WT), eps)
        W *= (A.T @ Zp) / np.maximum((A.T @ A) @ W, eps)
        s = np.linalg.norm(W, axis=1, keepdims=True)
        W /= np.maximum(s, eps)
        A *= s.T
    return W


def _atoms_medoid(Z: np.ndarray, M: int, seed: int) -> np.ndarray:
    """⭐ §C.1, the SHARP form: the real row nearest each learned centre.

    Differs from the arm in exactly one thing - whether an entry is the fitted centre of
    a cluster or the stored row closest to it.
    """
    cen = _kmeans(Z, M, seed)
    out = np.empty_like(cen)
    zn = (Z ** 2).sum(1)
    for j in range(cen.shape[0]):
        out[j] = Z[int(np.argmin(zn - 2.0 * (Z @ cen[j])))]
    return out


def _atoms_kpp(Z: np.ndarray, M: int, seed: int) -> np.ndarray:
    """§C.1, the ADVERSARIAL form: ``M`` maximally spread real rows (farthest-point)."""
    rng = np.random.default_rng(seed + 101)
    n = Z.shape[0]
    M = int(min(M, n))
    idx = [int(rng.integers(n))]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(1)
    for _ in range(1, M):
        j = int(np.argmax(d2))
        idx.append(j)
        d2 = np.minimum(d2, ((Z - Z[j]) ** 2).sum(1))
    return Z[np.asarray(idx)].copy()


def _pca_affine(Z: np.ndarray, M: int) -> np.ndarray:
    """⛔ R4 CONTROL ONLY - mean + top ``M-1`` centred principal directions.

    ⚠ It is the FAIR PCA control: a centred-only basis cannot represent the well mean and
    is unfairly weak (hub/generation-table DEV-28 retracted a claim over exactly that).
    """
    mu = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    return np.concatenate([mu[None, :], Vt[: max(int(M) - 1, 0)]], 0)


def well_dictionary(Z: np.ndarray, M: int, method: str, seed: int) -> np.ndarray:
    """``(n, d) train activations -> (M, d)`` dictionary.  ⛔ Sees no label, ever."""
    Z = np.asarray(Z, np.float64)
    M = int(M)
    if M <= 0:
        return np.zeros((0, Z.shape[1]), np.float64)
    if method == "kmeans":
        return _kmeans(Z, M, seed)
    if method == "nmf":
        return _nmf(Z, M, seed)
    if method == "atoms_medoid":
        return _atoms_medoid(Z, M, seed)
    if method == "atoms_kpp":
        return _atoms_kpp(Z, M, seed)
    if method == "atoms_random":
        rng = np.random.default_rng(seed + 202)
        return Z[rng.choice(Z.shape[0], size=min(M, Z.shape[0]), replace=False)].copy()
    if method == "pca_affine":
        return _pca_affine(Z, M)
    if method == "random":
        rng = np.random.default_rng(seed + 303)
        sc = float(np.sqrt((Z ** 2).sum(1).mean() / max(Z.shape[1], 1)))
        return rng.normal(0.0, sc, size=(M, Z.shape[1]))
    if method == "mean":
        return np.repeat(Z.mean(0)[None, :], M, 0)
    raise ValueError(f"unknown dictionary method {method!r}")


# ---------------------------------------------------------------------------
# 2. the allocation - ⭐ free at fixed bytes, and nobody had spent it
# ---------------------------------------------------------------------------

def allocate_atoms(Zs: Sequence[np.ndarray], total_atoms: int, *,
                   method: str = "waterfill", floor: int = 2) -> np.ndarray:
    """How many atoms each well gets, at a FIXED total ``sum_k M_k``.

    ⛔ The shipped arm gives every well the same ``M``.  Wells are not equally hard:
    with a correlation partition some are nearly rank-1 and some are not, and a uniform
    allocation overspends on the easy ones.  Both cost models are linear in
    ``sum_k M_k`` (the head is ``n_add . sum_k M_k``; with equal ``d_k`` the store is
    ``(D/K) . sum_k M_k``), so **redistributing atoms is free**.

    ``waterfill`` adds each next atom to the well whose leading UNUSED eigenvalue is
    largest - the greedy optimum for total residual variance under a per-well subspace
    budget.  ⛔ Label-free: the input is the train-activation spectrum per well.
    """
    K = len(Zs)
    total = int(total_atoms)
    floor = int(floor)
    if method == "uniform":
        base = total // K
        M = np.full(K, base, np.int64)
        M[: total - base * K] += 1
        return M
    if method != "waterfill":
        raise ValueError(f"unknown allocation method {method!r}")
    if total < floor * K:
        raise ValueError(f"total_atoms {total} below the floor {floor}*{K}")
    specs = []
    for Z in Zs:
        Zc = np.asarray(Z, np.float64)
        mu = Zc.mean(0)
        s = np.linalg.svd(Zc - mu, compute_uv=False)
        specs.append((s ** 2) / max(Zc.shape[0], 1))
    M = np.full(K, floor, np.int64)
    dmax = np.array([int(np.asarray(Z).shape[1]) for Z in Zs])
    for _ in range(total - floor * K):
        gains = np.array([
            specs[k][M[k] - 1] if (M[k] - 1) < specs[k].size and M[k] < dmax[k] else -1.0
            for k in range(K)
        ])
        k = int(np.argmax(gains))
        if gains[k] < 0.0:
            cand = np.flatnonzero(M < dmax)
            if cand.size == 0:
                break
            k = int(cand[np.argmin(M[cand])])
        M[k] += 1
    return M


def build_store_from_dicts(well_channels, dicts, feat_shape):
    """A :class:`~chlu.core.gen_store.GenStore` holding exactly ``dicts``.

    ⭐ ``GenStore`` already supports a per-well allocation (``n_modes_per_well``); this
    just fills it, so the shipped ``store_bytes`` ledger and the shipped
    ``clu_wide_vector`` read apply unchanged.
    """
    import jax

    from .gen_store import GenStore
    return GenStore(well_channels, int(max(1, len(dicts[0]))), feat_shape,
                    jax.random.PRNGKey(0),
                    init=[np.asarray(d, np.float32) for d in dicts])


# ---------------------------------------------------------------------------
# 3. the head - ragged-aware, metric-correct, reduced design
# ---------------------------------------------------------------------------

def reduced_additive_design(labels: np.ndarray, cardinalities: Sequence[int],
                            reduced: bool = True) -> np.ndarray:
    """``bias + one-hot per factor``, optionally with one reference level dropped.

    ⭐ The full design has rank ``1 + sum_f (V_f - 1)``; the extra ``F`` columns are
    structurally unidentifiable and cost ``F . A`` floats of head for nothing.
    ⛔ This is a re-parameterisation of the SAME hypothesis class, not a resolution cut.
    """
    Y = np.asarray(labels, np.int64)
    cards = [int(x) for x in cardinalities]
    n, F = Y.shape
    if not reduced:
        offs = np.cumsum([0] + cards[:-1]).astype(np.int64)
        A = np.zeros((n, 1 + int(sum(cards))), np.float64)
        A[:, 0] = 1.0
        A[np.repeat(np.arange(n), F), 1 + (Y + offs[None, :]).reshape(-1)] = 1.0
        return A
    widths = [c - 1 for c in cards]
    offs = np.cumsum([0] + widths[:-1]).astype(np.int64)
    A = np.zeros((n, 1 + int(sum(widths))), np.float64)
    A[:, 0] = 1.0
    for f in range(F):
        nz = np.flatnonzero(Y[:, f] > 0)
        A[nz, 1 + offs[f] + (Y[nz, f] - 1)] = 1.0
    return A


def _pair_blocks(labels: np.ndarray, cardinalities: Sequence[int],
                 pairs: Sequence[tuple]) -> np.ndarray:
    Y = np.asarray(labels, np.int64)
    cards = [int(x) for x in cardinalities]
    if not pairs:
        return np.zeros((Y.shape[0], 0), np.float64)
    blocks = []
    for (f, g) in pairs:
        B = np.zeros((Y.shape[0], cards[f] * cards[g]), np.float64)
        B[np.arange(Y.shape[0]), Y[:, f] * cards[g] + Y[:, g]] = 1.0
        blocks.append(B)
    return np.concatenate(blocks, 1)


@dataclass(frozen=True)
class RaggedHead:
    """The deployed head over a possibly NON-UNIFORM allocation ``(M_0 ... M_{K-1})``.

    ⭐ Coefficients are carried as one flat ``(N, A)`` block with ``A = sum_k M_k`` and
    split per well only at the read, so the byte ledger is exact for a ragged store.
    """

    cardinalities: tuple
    pairs: tuple
    modes_per_well: tuple
    reduced: bool
    C_add: np.ndarray          # (n_add, A)   full rank
    E: np.ndarray              # (n_pair, r)
    U: np.ndarray              # (r, A)
    fit_info: dict

    @property
    def rank(self) -> int:
        return int(self.E.shape[1]) if self.E.size else 0

    @property
    def n_atoms(self) -> int:
        return int(self.C_add.shape[1])

    def coefficients(self, requests: np.ndarray) -> list:
        """``(N, F) requests -> [ (N, M_k) ]``.  ⛔ No feature argument, by signature."""
        A = reduced_additive_design(requests, self.cardinalities, self.reduced)
        B = _pair_blocks(requests, self.cardinalities, self.pairs)
        a = A @ self.C_add
        if B.shape[1] and self.E.size:
            a = a + (B @ self.E) @ self.U
        out, s = [], 0
        for M in self.modes_per_well:
            out.append(np.asarray(a[:, s:s + int(M)], np.float32))
            s += int(M)
        return out

    def byte_ledger(self) -> dict:
        n = int(self.C_add.size + self.E.size + self.U.size)
        return {
            "n_values": n, "dtype": "float32", "bytes": int(n * 4),
            "bytes_formula": "(n_add*A + n_pair*r + r*A) * 4,  A = sum_k M_k",
            "n_additive_columns": int(self.C_add.shape[0]),
            "n_pair_columns": int(self.E.shape[0]) if self.E.size else 0,
            "rank": self.rank, "n_atoms_A": self.n_atoms,
            "modes_per_well": [int(x) for x in self.modes_per_well],
            "additive_design": ("reduced (one reference level dropped per factor)"
                                if self.reduced else "full one-hot"),
            "pairs": [list(p) for p in self.pairs],
            "role": ("the factored read's request-addressing head; reported BESIDE "
                     "store_bytes, never inside it"),
        }

    def channel_ledger(self) -> dict:
        return {"channel": "sum_k M_k real coefficients", "width": self.n_atoms,
                "dtype": "float32",
                "note": "⚠ a WIDE channel - ⛔ never quoted in the 30-bit column."}


def fit_factored_head_v2(
    dicts: Sequence[np.ndarray],
    perm: Sequence[np.ndarray],
    X: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    pairs: Sequence[tuple],
    rank: int,
    ridge_add: float = 10.0,
    ridge_pair: float = 100.0,
    reduced: bool = True,
    metric_correct: bool = True,
) -> RaggedHead:
    """Fit the head.  ⭐ Labels enter here and ONLY here (§B.6, the address book).

    Three differences from :func:`~chlu.core.gen_factored_read.fit_factored_head`, each
    of which is free in bytes:

    1. **Separate ridges** for the additive and interaction blocks.  They are different
       estimation problems - 52 dense columns against ~1.3 k sparse ones - and DEV-12
       measured the raw pairwise model's optimum at ridge 100 while the additive block's
       is 10.  One shared ``lambda`` cannot be right for both.
    2. ⭐ **Metric-correct rank truncation.**  The interaction block is truncated in the
       FEATURE-space metric (whitened by each well's Gram ``B_k B_k^T``), not in the
       coefficient metric.  Same bytes, a strictly better ``r``-rank approximation of
       the quantity that is decoded.
    3. **Reduced additive design** (52 columns, not 58), freeing ``6.A`` floats.

    ⛔ No held-out row, no request set and no arm outcome is visible here.
    """
    Z = np.asarray(X, np.float64).reshape((np.asarray(X).shape[0], -1))
    A = reduced_additive_design(labels, cardinalities, reduced)
    B = _pair_blocks(labels, cardinalities, pairs)
    na, nb = A.shape[1], B.shape[1]
    Phi = np.concatenate([A, B], 1) if nb else A
    lam = (np.concatenate([np.full(na, float(ridge_add)), np.full(nb, float(ridge_pair))])
           if nb else np.full(na, float(ridge_add)))
    W = np.linalg.solve(Phi.T @ Phi + np.diag(lam), Phi.T @ Z)   # (n_cols, D)

    Ms = [int(np.asarray(d).shape[0]) for d in dicts]
    Atot = int(sum(Ms))
    C = np.zeros((Phi.shape[1], Atot), np.float64)
    L_blocks, s = [], 0
    for k, d in enumerate(dicts):
        Bk = np.asarray(d, np.float64)                   # (M_k, d_k)
        Gk = Bk @ Bk.T
        M = Ms[k]
        C[:, s:s + M] = np.linalg.solve(Gk + 1e-8 * np.eye(M), Bk @ W[:, perm[k]].T).T
        ev, V = np.linalg.eigh(Gk + 1e-10 * np.eye(M))
        L_blocks.append((V * np.sqrt(np.maximum(ev, 0.0))[None, :]) @ V.T)
        s += M

    C_add = C[:na]
    if nb and int(rank) > 0:
        Cb = C[na:]
        if metric_correct:
            Lb = np.zeros((Atot, Atot), np.float64)
            s = 0
            for k, M in enumerate(Ms):
                Lb[s:s + M, s:s + M] = L_blocks[k]
                s += M
            Uh, Sh, Vth = np.linalg.svd(Cb @ Lb, full_matrices=False)
            r = int(min(int(rank), Sh.size))
            E = Uh[:, :r] * Sh[:r]
            U = np.linalg.solve(Lb.T, Vth[:r].T).T       # un-whiten: E @ U ~= Cb
        else:
            Uh, Sh, Vth = np.linalg.svd(Cb, full_matrices=False)
            r = int(min(int(rank), Sh.size))
            E = Uh[:, :r] * Sh[:r]
            U = Vth[:r]
        spec = float((Sh[:r] ** 2).sum() / max(float((Sh ** 2).sum()), 1e-30))
    else:
        E = np.zeros((0, 0))
        U = np.zeros((0, Atot))
        spec = 1.0

    return RaggedHead(
        cardinalities=tuple(int(x) for x in cardinalities),
        pairs=tuple(tuple(int(i) for i in p) for p in pairs),
        modes_per_well=tuple(Ms), reduced=bool(reduced),
        C_add=np.asarray(C_add, np.float32),
        E=np.asarray(E, np.float32), U=np.asarray(U, np.float32),
        fit_info={
            "n_rows": int(Z.shape[0]), "ridge_add": float(ridge_add),
            "ridge_pair": float(ridge_pair), "rank_requested": int(rank),
            "rank_realised": int(E.shape[1]) if E.size else 0,
            "interaction_spectral_mass_kept": spec,
            "metric_correct_truncation": bool(metric_correct),
            "n_additive_columns": int(na), "n_pair_columns": int(nb),
            "reduced_additive_design": bool(reduced),
            "additive_block_full_rank": True,
        },
    )


def head_bytes(n_add: int, n_pair: int, rank: int, n_atoms: int) -> int:
    """``(n_add*A + n_pair*r + r*A) * 4`` - the deployed head, in bytes."""
    return int((int(n_add) * int(n_atoms) + int(n_pair) * int(rank)
                + int(rank) * int(n_atoms)) * 4)
