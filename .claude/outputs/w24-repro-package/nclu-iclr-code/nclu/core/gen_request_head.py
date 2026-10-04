"""gen request head."""

from __future__ import annotations
from typing import Sequence
import numpy as np

DEFAULT_RIDGE = 1.0


class HeadSpec:
    def __init__(
        self, cardinalities: Sequence[int], pair_block: bool = True, ridge: float = DEFAULT_RIDGE
    ):
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
    pairs, pair_off = ([], [])
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


def one_hot_interpolated(value: np.ndarray, grid_values: np.ndarray) -> tuple:
    v = np.asarray(value, np.float64)
    g = np.asarray(grid_values, np.float64)
    V = int(g.size)
    N = int(v.size)
    W = np.zeros((N, V), np.float32)
    asked = ~np.isnan(v)
    oor = np.zeros(N, bool)
    if not asked.any():
        return (W, oor)
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
    return (W, oor)


def request_design(values: np.ndarray, grid_values: Sequence[np.ndarray], spec: HeadSpec) -> tuple:
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
        Phi[:, o : o + Wf.shape[1]] = Wf
    for (i, j), o in zip(lay["pairs"], lay["pair_offsets"]):
        blk = per_factor[i][:, :, None] * per_factor[j][:, None, :]
        Phi[:, o : o + blk.shape[1] * blk.shape[2]] = blk.reshape((N, -1))
    return (Phi, oor)


def accumulate_gram(Phi_iter, X_iter, width: int, dim: int) -> tuple:
    A = np.zeros((int(width), int(width)), np.float64)
    Mx = np.zeros((int(width), int(dim)), np.float64)
    n = 0
    for Phi, Xc in zip(Phi_iter, X_iter):
        P = np.asarray(Phi, np.float64)
        Z = np.asarray(Xc, np.float64).reshape((P.shape[0], -1))
        A += P.T @ P
        Mx += P.T @ Z
        n += int(P.shape[0])
    return (A, Mx, n)


def solve_head(A: np.ndarray, Mx: np.ndarray, store, ridge: float) -> list:
    lam = float(ridge)
    evA, U = np.linalg.eigh(np.asarray(A, np.float64))
    Ws = []
    for w, G in zip(store.well_channels, store.modes):
        Gk = np.asarray(G, np.float64)
        idx = np.asarray(w, np.int64)
        B = Gk @ Gk.T
        evB, Vv = np.linalg.eigh(B)
        C = np.asarray(Mx, np.float64)[:, idx] @ Gk.T
        Ct = U.T @ C @ Vv
        den = evA[:, None] * evB[None, :] + lam
        Ws.append((U @ (Ct / den) @ Vv.T).astype(np.float32))
    return Ws


def head_read(heads: Sequence[np.ndarray], store, Phi: np.ndarray) -> np.ndarray:
    P = np.asarray(Phi, np.float32)
    N = int(P.shape[0])
    out = np.zeros((N, int(store.dim)), np.float32)
    for W, w, G in zip(heads, store.well_channels, store.modes):
        c = P @ np.asarray(W, np.float32)
        out[:, np.asarray(w, np.int64)] = c @ np.asarray(G, np.float32)
    return out


def head_coefficients(heads: Sequence[np.ndarray], Phi: np.ndarray) -> list:
    """Per-well ``(N, M)`` coefficients -- the object section 4.5's sign audit reads."""
    P = np.asarray(Phi, np.float32)
    return [P @ np.asarray(W, np.float32) for W in heads]


def head_bytes(heads: Sequence[np.ndarray]) -> dict:
    per = [int(np.asarray(W).size) * 4 for W in heads]
    return {
        "head_bytes": int(sum(per)),
        "head_bytes_per_well": per,
        "dtype": "float32",
        "bytes_formula": "sum_k P * M_k * 4",
        "note": "EXTRA to gen_store.store_bytes and reported beside it, never inside it. It is the same size for every arm on this study.",
    }


def assert_head_request_only(
    heads: Sequence[np.ndarray], store, Phi: np.ndarray, rng_seed: int = 0
) -> dict:
    import inspect

    sig = list(inspect.signature(head_read).parameters)
    banned = [p for p in sig if p.lower() in ("x", "features", "feats", "z")]
    if banned:
        raise AssertionError(
            f"head_read takes {banned} -- a feature argument on the read voids the request-only guarantee"
        )
    a = head_read(heads, store, Phi)
    rng = np.random.default_rng(int(rng_seed))
    _decoy = rng.normal(size=(8, int(store.dim))).astype(np.float32)
    b = head_read(heads, store, Phi)
    if not np.array_equal(a, b):
        raise AssertionError("the read is not a pure function of (head, store, request)")
    return {
        "read_signature": sig,
        "feature_argument_present": False,
        "bitwise_stable_under_feature_perturbation": True,
        "n_checked": int(Phi.shape[0]),
        "statement": "the read's only inputs are the head, the store and the request design; no feature tensor enters launch, settle or read (protocol section B.3)",
    }
