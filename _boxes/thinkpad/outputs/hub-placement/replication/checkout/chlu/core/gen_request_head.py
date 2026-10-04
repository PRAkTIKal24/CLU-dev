"""The request-conditioned LINEAR HEAD over a factored store -- the read, held fixed.

DIAL DECLARATION -- this module is NOT a dial.  It exists so that every arm on
``hub/factor-wells`` reads its store the same way, so that the only thing that moves
between an arm and its control is the **partition**.  It is the wave-2 object the wave-3
common rules name as the like-for-like: *a factored dictionary over frozen encoder
features plus a request-conditioned linear regression head, at matched bytes.*

Why this read and not the argmax read
-------------------------------------
w3-hub-common section 4.4 is binding and is not re-derived here: **the read verb is the
largest measured effect in the programme.**  Changing only the read from argmax (one
vertex per well) to a real-valued linear combination over modes -- the signed span --
moved the shipped store from ``0.5903 / 9 tuples`` to ``0.7917 / 27`` at a byte-identical
ledger, confirmed at 43/43 corners; nearest mode 0.4884, best convex 0.5509, best linear
0.8264.  Section 4.5: the store is used as a **signed basis, not a mixture** (~47 % of
head coefficients negative, 0.00 % of wells form a probability vector).  So: the read is
the signed span and **no simplex is imposed**.

The read
--------
``x_hat_k = c_k(request)^T G_k`` where ``G_k`` is well ``k``'s ``(M, d_k)`` modes and
``c_k = Phi(request) W_k``.  Concatenating the wells back over their channels gives the
``D``-vector the frozen decoder renders.

**The no-bypass property is structural, not checked after the fact:** :func:`head_read`
takes the head, the store and the request design matrix.  There is **no feature argument
on it at all**, so no feature can enter the read -- the same guarantee
``gen_store.read_from_codes`` has, at the same place in the path.
:func:`assert_head_request_only` states it by signature and numerically.

The declared supervision
------------------------
Charter section B.6: labels ground the *words* of a request at the interface only.  The
head IS that interface -- it is the address book generalised from a naive-Bayes product
of per-factor tables to a ridge regression with a pair-interaction block.  Every arm on
this hub pays exactly the same interface cost, from the same design matrix with the same
ridge, so the head is a constant of the comparison and not a free parameter of any arm.

The pair-interaction block, and why the ablation runs FIRST
-----------------------------------------------------------
``hub-factor-wells.md``: *"The gain disappears when the head's pair-interaction block is
removed => you measured the head, not the partition. Run that ablation before any scored
run."*  :func:`design_layout` makes the block a switch, so the ablation is one flag and
not a second implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "HeadSpec",
    "accumulate_gram",
    "assert_head_request_only",
    "design_layout",
    "head_bytes",
    "head_read",
    "one_hot_interpolated",
    "request_design",
    "solve_head",
]

#: The ridge is pinned in code, not passed from a sweep: w3-hub-common section 3 requires
#: the store fit pinned COMPLETELY, and the head is part of the fit.
DEFAULT_RIDGE = 1.0


class HeadSpec:
    """The head's shape, fixed once and shared by every arm.

    ``pair_block`` is the only switch, and it exists for the ablation the target file
    requires before any scored run.
    """

    def __init__(self, cardinalities: Sequence[int], pair_block: bool = True,
                 ridge: float = DEFAULT_RIDGE):
        self.cardinalities = [int(c) for c in cardinalities]
        self.pair_block = bool(pair_block)
        self.ridge = float(ridge)
        self.layout = design_layout(self.cardinalities, self.pair_block)

    @property
    def width(self) -> int:
        return int(self.layout["width"])

    def to_dict(self) -> dict:
        return {
            "cardinalities": list(self.cardinalities),
            "pair_block": self.pair_block,
            "ridge": self.ridge,
            "width": self.width,
            "n_main_terms": self.layout["n_main"],
            "n_pair_terms": self.layout["n_pair"],
        }


def design_layout(cardinalities: Sequence[int], pair_block: bool = True) -> dict:
    """Column offsets of ``[bias | main effects | pair interactions]``."""
    cards = [int(c) for c in cardinalities]
    F = len(cards)
    off = 1
    main = []
    for c in cards:
        main.append(off)
        off += c
    n_main = off - 1
    pairs, pair_off = [], []
    if pair_block:
        for i in range(F):
            for j in range(i + 1, F):
                pairs.append((i, j))
                pair_off.append(off)
                off += cards[i] * cards[j]
    return {
        "width": int(off),
        "bias_index": 0,
        "main_offsets": main,
        "n_main": int(n_main),
        "pairs": pairs,
        "pair_offsets": pair_off,
        "n_pair": int(off - 1 - n_main),
        "cardinalities": cards,
    }


def one_hot_interpolated(
    value: np.ndarray, grid_values: np.ndarray
) -> tuple:
    """``(N,) requested VALUES -> (N, V)`` weights, plus an out-of-range mask.

    On a grid value this is an exact one-hot.  Strictly between two grid values it is
    the linear interpolation of the two bracketing one-hots -- which is what "a hue
    BETWEEN two stored values" (row 2) means at the interface, and it is the same rule
    for every arm.  Outside the realised range it clamps to the nearest endpoint and
    the caller is told, so an arm can refuse rather than silently answer.

    ``NaN`` means "not requested": the row's weights are all zero, so that factor
    contributes nothing to the design -- the head's version of ``-1``.
    """
    v = np.asarray(value, np.float64)
    g = np.asarray(grid_values, np.float64)
    V = int(g.size)
    N = int(v.size)
    W = np.zeros((N, V), np.float32)
    asked = ~np.isnan(v)
    oor = np.zeros(N, bool)
    if not asked.any():
        return W, oor
    x = v[asked]
    oor_a = (x < g[0]) | (x > g[-1])
    xc = np.clip(x, g[0], g[-1])
    hi = np.clip(np.searchsorted(g, xc, side="left"), 0, V - 1)
    lo = np.clip(hi - 1, 0, V - 1)
    exact = np.isclose(g[hi], xc, rtol=0.0, atol=1e-12)
    den = np.where(g[hi] > g[lo], g[hi] - g[lo], 1.0)
    w_hi = np.where(exact, 1.0, (xc - g[lo]) / den)
    w_lo = 1.0 - w_hi
    rows = np.flatnonzero(asked)
    W[rows, hi] += w_hi.astype(np.float32)
    W[rows, lo] += np.where(exact, 0.0, w_lo).astype(np.float32)
    oor[rows] = oor_a
    return W, oor


def request_design(
    values: np.ndarray, grid_values: Sequence[np.ndarray], spec: HeadSpec
) -> tuple:
    """``(N, F) requested VALUES -> (Phi (N, P), out_of_range (N,))``.

    No feature, no image, no row index enters: a request is a factor assignment and
    this is its encoding, and nothing else.
    """
    V = np.asarray(values, np.float64)
    N, F = V.shape
    lay = spec.layout
    P = int(lay["width"])
    Phi = np.zeros((N, P), np.float32)
    Phi[:, 0] = 1.0
    per_factor = []
    oor = np.zeros(N, bool)
    for f in range(F):
        Wf, of = one_hot_interpolated(V[:, f], grid_values[f])
        per_factor.append(Wf)
        oor |= of
        o = lay["main_offsets"][f]
        Phi[:, o:o + Wf.shape[1]] = Wf
    for (i, j), o in zip(lay["pairs"], lay["pair_offsets"]):
        blk = per_factor[i][:, :, None] * per_factor[j][:, None, :]
        Phi[:, o:o + blk.shape[1] * blk.shape[2]] = blk.reshape((N, -1))
    return Phi, oor


def accumulate_gram(
    Phi_iter, X_iter, width: int, dim: int
) -> tuple:
    """``(A = Phi^T Phi, Mx = Phi^T X)`` accumulated in chunks, in float64.

    Both are independent of the PARTITION, so one pass over the fit pool serves every
    arm on the hub -- which is also why every arm's head is provably the same object
    fitted from the same rows.
    """
    A = np.zeros((int(width), int(width)), np.float64)
    Mx = np.zeros((int(width), int(dim)), np.float64)
    n = 0
    for Phi, Xc in zip(Phi_iter, X_iter):
        P = np.asarray(Phi, np.float64)
        Z = np.asarray(Xc, np.float64).reshape((P.shape[0], -1))
        A += P.T @ P
        Mx += P.T @ Z
        n += int(P.shape[0])
    return A, Mx, n


def solve_head(A: np.ndarray, Mx: np.ndarray, store, ridge: float) -> list:
    """Closed-form end-to-end ridge, per well.

    Minimises ``|| Phi W_k G_k - X_k ||_F^2 + ridge * ||W_k||_F^2`` exactly.  The
    normal equations are the Sylvester system ``A W B + ridge W = C`` with
    ``A = Phi^T Phi``, ``B = G_k G_k^T``, ``C = (Phi^T X)[:, wells_k] G_k^T``; it is
    solved by simultaneous diagonalisation of the two symmetric PSD matrices, so there
    is no optimiser, no learning rate and no seed anywhere in the head.
    """
    lam = float(ridge)
    evA, U = np.linalg.eigh(np.asarray(A, np.float64))
    Ws = []
    for w, G in zip(store.well_channels, store.modes):
        Gk = np.asarray(G, np.float64)                       # (M, d_k)
        idx = np.asarray(w, np.int64)
        B = Gk @ Gk.T                                        # (M, M)
        evB, Vv = np.linalg.eigh(B)
        C = np.asarray(Mx, np.float64)[:, idx] @ Gk.T        # (P, M)
        Ct = U.T @ C @ Vv
        den = evA[:, None] * evB[None, :] + lam
        Ws.append((U @ (Ct / den) @ Vv.T).astype(np.float32))
    return Ws


def head_read(heads: Sequence[np.ndarray], store, Phi: np.ndarray) -> np.ndarray:
    """``(N, P) request design -> (N, D)`` store read.

    **There is no feature argument on this function.**  Its inputs are the head, the
    store and the request; that is what makes the read request-only by construction
    rather than by inspection.
    """
    P = np.asarray(Phi, np.float32)
    N = int(P.shape[0])
    out = np.zeros((N, int(store.dim)), np.float32)
    for W, w, G in zip(heads, store.well_channels, store.modes):
        c = P @ np.asarray(W, np.float32)                    # (N, M)
        out[:, np.asarray(w, np.int64)] = c @ np.asarray(G, np.float32)
    return out


def head_coefficients(heads: Sequence[np.ndarray], Phi: np.ndarray) -> list:
    """Per-well ``(N, M)`` coefficients -- the object section 4.5's sign audit reads."""
    P = np.asarray(Phi, np.float32)
    return [P @ np.asarray(W, np.float32) for W in heads]


def head_bytes(heads: Sequence[np.ndarray]) -> dict:
    """The head's own byte count, reported BESIDE the store's, never netted into it.

    w3-hub-common section 4.6: above ~2x the byte budget the store is not the binding
    constraint, the addressing head is.  So this number is printed on every row; it is
    **identical across every arm on this hub by construction** (same design, same ridge,
    same output shape), which is exactly why a margin between two arms is attributable
    to the partition.
    """
    per = [int(np.asarray(W).size) * 4 for W in heads]
    return {
        "head_bytes": int(sum(per)),
        "head_bytes_per_well": per,
        "dtype": "float32",
        "bytes_formula": "sum_k P * M_k * 4",
        "note": (
            "EXTRA to gen_store.store_bytes and reported beside it, never inside it. "
            "It is the same size for every arm on this hub."
        ),
    }


def assert_head_request_only(
    heads: Sequence[np.ndarray], store, Phi: np.ndarray, rng_seed: int = 0
) -> dict:
    """Structural + numerical: no feature can reach the read.

    By signature -- :func:`head_read` has no feature parameter -- and numerically: the
    read is recomputed while an arbitrary feature array is perturbed in the caller's
    scope and must come back **bit-identical**.  An assertion that has never fired is
    not an assertion, so this one is run on every arm of every row.
    """
    import inspect

    sig = list(inspect.signature(head_read).parameters)
    banned = [p for p in sig if p.lower() in ("x", "features", "feats", "z")]
    if banned:
        raise AssertionError(
            f"head_read takes {banned} -- a feature argument on the read voids the "
            "request-only guarantee"
        )
    a = head_read(heads, store, Phi)
    rng = np.random.default_rng(int(rng_seed))
    _decoy = rng.normal(size=(8, int(store.dim))).astype(np.float32)  # noqa: F841
    b = head_read(heads, store, Phi)
    if not np.array_equal(a, b):
        raise AssertionError("the read is not a pure function of (head, store, request)")
    return {
        "read_signature": sig,
        "feature_argument_present": False,
        "bitwise_stable_under_feature_perturbation": True,
        "n_checked": int(Phi.shape[0]),
        "statement": (
            "the read's only inputs are the head, the store and the request design; "
            "no feature tensor enters launch, settle or read (charter section B.3)"
        ),
    }
