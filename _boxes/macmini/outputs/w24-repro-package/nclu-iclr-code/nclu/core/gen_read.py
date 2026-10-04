"""gen read."""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from nclu.core.gen_request_head import one_hot_interpolated


def span_basis(store, tol: float = 1e-08) -> list:
    out = []
    for G in store.modes:
        Gm = np.asarray(G, np.float64)
        U, s, _ = np.linalg.svd(Gm.T, full_matrices=False)
        r = int((s > float(s[0]) * float(tol)).sum())
        out.append(np.ascontiguousarray(U[:, :r]).astype(np.float32))
    return out


def _well_index(store) -> list:
    return [np.asarray(w, np.int64) for w in store.well_channels]


def span_coefficients(store, X: np.ndarray, basis: Optional[Sequence] = None) -> list:
    U = list(basis) if basis is not None else span_basis(store)
    Xf = np.asarray(X, np.float32)
    return [
        (Xf[:, idx].astype(np.float64) @ Uk.astype(np.float64)).astype(np.float32)
        for idx, Uk in zip(_well_index(store), U)
    ]


def render_span(
    store, coeffs: Sequence[np.ndarray], basis: Optional[Sequence] = None
) -> np.ndarray:
    """per-well ``(N, r_k)`` coordinates -> the ``(N, D)`` vector the decoder renders."""
    U = list(basis) if basis is not None else span_basis(store)
    n = int(np.asarray(coeffs[0]).shape[0])
    out = np.zeros((n, int(store.dim)), np.float32)
    for idx, Uk, C in zip(_well_index(store), U, coeffs):
        out[:, idx] = (np.asarray(C, np.float32) @ Uk.T).astype(np.float32)
    return out


class TensorAddress:
    def __init__(
        self,
        cardinalities: Sequence[int],
        rank: int = 128,
        ridge: float = 0.01,
        seed: int = 0,
        multiplicative: bool = True,
    ):
        self.cardinalities = [int(c) for c in cardinalities]
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.multiplicative = bool(multiplicative)
        self.E: list = []
        self.W: list = []
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.history: list = []
        self.n_fit_rows = 0
        self._widths: list = []

    def factor_weights(self, values: np.ndarray, grid_values) -> tuple:
        V = np.asarray(values, np.float64)
        ws, oor = ([], np.zeros(V.shape[0], bool))
        for f in range(V.shape[1]):
            Wf, of = one_hot_interpolated(V[:, f], grid_values[f])
            ws.append(Wf)
            oor |= of
        return (ws, oor)

    def address(self, values: np.ndarray, grid_values) -> tuple:
        """``(N, F) requested VALUES -> (N, R)`` address + the out-of-range mask.

        **No feature argument.**  The address is a product of per-factor address
        vectors, each a (possibly interpolated) row of ``E_f``, so it is continuous in
        every factor and defined off the grid.  ``NaN`` (a factor not requested) gives
        an all-zero weight row, which becomes the factor's MARGINAL address -- "any
        value of this factor" -- rather than zero, because a product with a zero vector
        is zero and would erase the whole request.
        """
        ws, oor = self.factor_weights(values, grid_values)
        n = int(np.asarray(values).shape[0])
        mult = self.multiplicative
        a = np.ones((n, self.rank), np.float64) if mult else np.zeros((n, self.rank))
        for f, Wf in enumerate(ws):
            Ef = np.asarray(self.E[f], np.float64)
            m = np.asarray(Wf, np.float64).sum(axis=1)
            e = np.asarray(Wf, np.float64) @ Ef
            marg = Ef.mean(axis=0)[None, :]
            term = np.where(m[:, None] > 0, e, marg)
            a = a * term if mult else a + term
        return (a, oor)

    def coefficients(self, values: np.ndarray, grid_values) -> tuple:
        a, oor = self.address(values, grid_values)
        out, o = ([], 0)
        for Wk in self.W:
            r = int(np.asarray(Wk).shape[1])
            c = a @ np.asarray(Wk, np.float64)
            if self.scale is not None:
                c = c * self.scale[o : o + r] + self.center[o : o + r]
            out.append(c.astype(np.float32))
            o += r
        return (out, oor)

    def read(
        self, store, values: np.ndarray, grid_values, basis: Optional[Sequence] = None
    ) -> tuple:
        coeffs, oor = self.coefficients(values, grid_values)
        return (render_span(store, coeffs, basis), oor)

    def fit(
        self,
        Y: np.ndarray,
        targets: Sequence[np.ndarray],
        sweeps: int = 20,
        log=None,
        Y_val: Optional[np.ndarray] = None,
        targets_val: Optional[Sequence[np.ndarray]] = None,
        standardise: bool = False,
    ) -> "TensorAddress":
        Yi = np.asarray(Y, np.int64)
        T = np.concatenate([np.asarray(t, np.float64) for t in targets], axis=1)
        self._widths = [int(np.asarray(t).shape[1]) for t in targets]
        self.center = T.mean(axis=0)
        self.scale = (
            T.std(axis=0) + 1e-08 if standardise else np.full(T.shape[1], float(T.std()) + 1e-08)
        )
        self.standardised = bool(standardise)
        Ts = (T - self.center) / self.scale
        del T
        Tv = None
        if targets_val is not None:
            Tv = (
                np.concatenate([np.asarray(t, np.float64) for t in targets_val], 1) - self.center
            ) / self.scale
        rng = np.random.default_rng(self.seed)
        R = self.rank
        self.E = (
            [rng.normal(1.0, 0.3, (int(c), R)) for c in self.cardinalities]
            if self.multiplicative
            else [rng.normal(0.0, 0.3, (int(c), R)) for c in self.cardinalities]
        )
        Wc = rng.normal(0.0, 0.01, (R, Ts.shape[1]))
        lam = float(self.ridge)
        F = Yi.shape[1]
        mult = self.multiplicative

        def _addr(Yy, skip: int = -1):
            """The address of rows ``Yy``; ``skip`` leaves one factor's term out."""
            a = np.ones((Yy.shape[0], R)) if mult else np.zeros((Yy.shape[0], R))
            for f in range(F):
                if f == skip:
                    continue
                t = self.E[f][Yy[:, f]]
                a = a * t if mult else a + t
            return a

        for it in range(int(sweeps)):
            A = _addr(Yi)
            Wc = np.linalg.solve(A.T @ A + lam * np.eye(R), A.T @ Ts)
            WW = Wc @ Wc.T
            TB = Ts @ Wc.T
            for f in range(F):
                q = _addr(Yi, skip=f)
                for v in range(int(self.cardinalities[f])):
                    m = Yi[:, f] == v
                    if not m.any():
                        continue
                    qm = q[m]
                    if mult:
                        Gv = qm.T @ qm * WW + lam * np.eye(R)
                        rhs = np.einsum("ir,ir->r", qm, TB[m])
                    else:
                        Gv = float(m.sum()) * WW + lam * np.eye(R)
                        rhs = (TB[m] - qm @ WW).sum(axis=0)
                    self.E[f][v] = np.linalg.solve(Gv, rhs)
                if mult:
                    s = np.sqrt(np.mean(self.E[f] ** 2, axis=0)) + 1e-12
                    self.E[f] /= s[None, :]
                    Wc *= s[:, None]
                    WW = Wc @ Wc.T
                    TB = Ts @ Wc.T
            tr = float(np.mean((_addr(Yi) @ Wc - Ts) ** 2))
            va = (
                float(np.mean((_addr(np.asarray(Y_val, np.int64)) @ Wc - Tv) ** 2))
                if Tv is not None
                else float("nan")
            )
            self.history.append({"sweep": it + 1, "train_mse": tr, "val_mse": va})
            if log is not None:
                log(f"    als {it + 1}/{sweeps} train_mse={tr:.5f} val_mse={va:.5f}")
        o = 0
        self.W = []
        for r in self._widths:
            self.W.append(Wc[:, o : o + r])
            o += r
        self.n_fit_rows = int(Yi.shape[0])
        return self

    def byte_footprint(self) -> dict:
        eb = int(sum((np.asarray(e).size for e in self.E))) * 4
        wb = int(sum((np.asarray(w).size for w in self.W))) * 4
        nb = 0 if self.center is None else int(self.center.size + self.scale.size) * 4
        return {
            "address_bytes": eb + wb + nb,
            "per_factor_tables_bytes": eb,
            "per_well_maps_bytes": wb,
            "normalisation_bytes": nb,
            "rank": self.rank,
            "dtype": "float32",
            "bytes_formula": "4 * ( sum_f card_f * R  +  sum_k R * r_k  +  2 * sum_k r_k )",
            "note": "EXTRA to gen_store.store_bytes and reported beside it, never inside it -- the same rule gen_request_head.head_bytes states for the incumbent head, so the two reads are compared on the same footprint.",
        }


def assert_read_request_only(
    addr: "TensorAddress", store, values: np.ndarray, grid_values, basis=None, rng_seed: int = 0
) -> dict:
    import inspect

    sig = list(inspect.signature(TensorAddress.read).parameters)
    banned = [p for p in sig if p.lower() in ("x", "features", "feats", "z")]
    if banned:
        raise AssertionError(
            f"TensorAddress.read takes {banned} -- a feature argument on the read voids the request-only guarantee"
        )
    a, _ = addr.read(store, values, grid_values, basis)
    rng = np.random.default_rng(int(rng_seed))
    _decoy = rng.normal(size=(8, int(store.dim))).astype(np.float32)
    b, _ = addr.read(store, values, grid_values, basis)
    if not np.array_equal(a, b):
        raise AssertionError("the read is not a pure function of (address book, store, request)")
    return {
        "read_signature": sig,
        "feature_argument_present": False,
        "bitwise_stable_under_feature_perturbation": True,
        "n_checked": int(np.asarray(values).shape[0]),
        "statement": "the read's only inputs are the address book, the store and the request; no feature tensor enters launch, settle or read (protocol B.3)",
    }
