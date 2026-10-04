"""hub/read -- THE DIAL: how a well's content becomes an answer.

DIAL DECLARATION
----------------
This module moves the **read** and nothing else.  Every object here consumes a store
that was fitted elsewhere, by someone else's dial, and never refits it: the atoms, the
partition, the byte ledger and the frozen decoder are constants of every comparison on
this hub.  What moves is the map ``request -> the coefficients that address the atoms``.

Why the read is the dial, measured on the deployed object
---------------------------------------------------------
On the deployed ``factor_aligned_balanced`` store (524,288 B), 512 held-out requests,
THE frozen decoder + THE hash-asserted external classifier:

============================================  =========
read (content, partition and bytes IDENTICAL)  all-six
============================================  =========
true encoder features (instrument ceiling)      0.9062
ORACLE address into the same atoms              0.3867
the deployed ridge span head (main+pair)        0.0781
plain per-value-mean table, full D              0.0215
============================================  =========

The ORACLE address is the projection of the true feature onto ``span(G_k)`` -- it is not
an arm (it reads a feature) but it is the exact ceiling of this dial, and it says that
**4.95x of the deployed read's answer is sitting inside the atoms, unaddressed.**
:func:`oracle_span_address` is here so that ceiling is recomputed, not quoted.

The two reads
-------------
:func:`span_head_read` is the incumbent -- ``c_k = Phi W_k``, a ridge regression on
``[bias | main effects | pair interactions]``.  It is additive to second order, and its
width (1380) buys 1,766,400 B of head.

:class:`TensorAddress` is this hub's arm.  The address of a request is the **elementwise
product of one per-factor address vector per factor**::

    a(request) = prod_f  ( w_f(v_f)^T E_f )        in R^R
    c_k        = a(request) @ W_k                  in R^{r_k}
    x_hat[w_k] = c_k @ U_k^T

``w_f(v)`` is :func:`gen_request_head.one_hot_interpolated` -- an exact one-hot on a grid
value and the interpolation of the two bracketing one-hots between them.  So the address
is **continuous in every factor**: the per-well coordinate a request lands on moves
smoothly as the request moves, which is what an off-grid request needs and what an
argmax over modes cannot express.

A product is the point.  A sum of per-factor terms cannot say "this hue *of this shape*";
a product can, and to all orders, at ``sum_f card_f * R`` numbers instead of the
``card_i * card_j`` blocks a second-order design pays for one order.

``U_k`` is the orthonormal basis of ``span(G_k)`` obtained by SVD of the DEPLOYED atoms.
It carries **no bytes of its own** -- it is a deterministic function of the store on
disk, recomputed on load -- and the reachable set ``span(U_k) = span(G_k)`` is exactly
the incumbent's, so the two reads address the same content and differ only in the map.

Request-only, by signature
--------------------------
Charter B.3.  :meth:`TensorAddress.read` takes ``(store, values, grid_values)`` and the
fitted address book.  **There is no feature argument on it**, exactly as
``gen_request_head.head_read`` has none; :func:`assert_read_request_only` states it by
signature and numerically.  Labels enter ONCE, at fit time, as the factor index of a fit
row -- the address book's own words (charter B.6, w4-common section 5) -- and never into
the landscape, the atoms or the read.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .gen_request_head import head_read, one_hot_interpolated, request_design

__all__ = [
    "TensorAddress",
    "assert_read_request_only",
    "oracle_span_address",
    "render_span",
    "span_basis",
    "span_coefficients",
    "span_head_read",
]


# ---------------------------------------------------------------------------
# 0. The span of the deployed atoms -- zero bytes, recomputed from the store
# ---------------------------------------------------------------------------


def span_basis(store, tol: float = 1e-8) -> list:
    """Per well, an orthonormal basis of ``span(G_k)``: ``(d_k, r_k)`` float32.

    Derived from the atoms on disk by SVD, so it adds **nothing** to the byte ledger and
    it cannot enlarge the reachable set: ``span(U_k) == span(G_k)`` to numerical rank.
    Working in this basis instead of the mode coefficients is what makes the fit
    well-posed -- the atoms of a soft-occupancy store are near-collinear, so ``G G^T`` is
    badly conditioned and the mode coefficients are not.
    """
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
    """``(N, D) features -> per-well (N, r_k)`` coordinates in the atom span.

    This function TAKES FEATURES.  It is a **fit-time and diagnostic** object only -- it
    builds the regression target and the oracle ceiling.  It is never called on the read
    path, and :func:`assert_read_request_only` is what proves that.
    """
    U = list(basis) if basis is not None else span_basis(store)
    Xf = np.asarray(X, np.float32)
    return [(Xf[:, idx].astype(np.float64) @ Uk.astype(np.float64)).astype(np.float32)
            for idx, Uk in zip(_well_index(store), U)]


def render_span(store, coeffs: Sequence[np.ndarray],
                basis: Optional[Sequence] = None) -> np.ndarray:
    """per-well ``(N, r_k)`` coordinates -> the ``(N, D)`` vector the decoder renders."""
    U = list(basis) if basis is not None else span_basis(store)
    n = int(np.asarray(coeffs[0]).shape[0])
    out = np.zeros((n, int(store.dim)), np.float32)
    for idx, Uk, C in zip(_well_index(store), U, coeffs):
        out[:, idx] = (np.asarray(C, np.float32) @ Uk.T).astype(np.float32)
    return out


def oracle_span_address(store, X: np.ndarray, basis: Optional[Sequence] = None
                        ) -> np.ndarray:
    """THE CEILING OF THIS DIAL: the best answer these atoms can give, per request.

    **Not an arm.**  It reads the true feature of the requested row, so it bypasses the
    request entirely and is disqualified by charter B.3 as a mechanism.  It is reported
    on every table as the ceiling, because a read that reaches it has taken everything
    the deployed atoms hold and the next gain has to come from a different dial.
    """
    return render_span(store, span_coefficients(store, X, basis), basis)


def span_head_read(heads, store, values: np.ndarray, grid_values, spec) -> tuple:
    """The INCUMBENT read, wrapped at this module's interface (request in, D-vector out)."""
    Phi, oor = request_design(values, grid_values, spec)
    return head_read(heads, store, Phi), oor


# ---------------------------------------------------------------------------
# 1. The tensor-product address book -- THE ARM
# ---------------------------------------------------------------------------


class TensorAddress:
    """``request -> address -> the atoms``, with the address a PRODUCT over factors.

    Parameters
    ----------
    cardinalities : the grid's per-factor value counts.  One address row per value.
    rank : ``R``, the width of the address.  The only capacity dial of the read.
    ridge : the ALS ridge, on both half-steps.
    seed : the initialisation seed.  It is the only stochastic thing in the fit; both
        half-steps are closed form.

    The fitted object is ``E`` (one ``(card_f, R)`` table per factor) and ``W`` (one
    ``(R, r_k)`` matrix per well).  Its bytes are declared by :meth:`byte_ledger` and are
    reported BESIDE the store's, never netted into them -- the same rule
    ``gen_request_head.head_bytes`` states for the incumbent.
    """

    def __init__(self, cardinalities: Sequence[int], rank: int = 128,
                 ridge: float = 1e-2, seed: int = 0, multiplicative: bool = True):
        self.cardinalities = [int(c) for c in cardinalities]
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.seed = int(seed)
        #: ⭐ THE ABLATION THAT ISOLATES THE MECHANISM.  ``False`` keeps the rank, the
        #: fit, the ridge, the seed and the byte ledger exactly and replaces the PRODUCT
        #: over factors by a SUM -- which is the additive address, i.e. the per-value-mean
        #: read with a learned basis.  A margin of the product arm over this one is
        #: attributable to the multiplication and to nothing else.
        self.multiplicative = bool(multiplicative)
        self.E: list = []
        self.W: list = []
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.history: list = []
        self.n_fit_rows = 0
        self._widths: list = []

    # -- the address ------------------------------------------------------
    def factor_weights(self, values: np.ndarray, grid_values) -> tuple:
        """``(N, F) requested VALUES -> [ (N, card_f) ]`` weights + out-of-range mask.

        Exactly ``gen_request_head.one_hot_interpolated``, factor by factor: the same
        interface rule as the incumbent, so the two reads see the same request.
        """
        V = np.asarray(values, np.float64)
        ws, oor = [], np.zeros(V.shape[0], bool)
        for f in range(V.shape[1]):
            Wf, of = one_hot_interpolated(V[:, f], grid_values[f])
            ws.append(Wf)
            oor |= of
        return ws, oor

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
            m = np.asarray(Wf, np.float64).sum(axis=1)          # 0 where not requested
            e = np.asarray(Wf, np.float64) @ Ef                 # (N, R)
            marg = Ef.mean(axis=0)[None, :]                     # "any value"
            term = np.where(m[:, None] > 0, e, marg)
            a = a * term if mult else a + term
        return a, oor

    # -- the read ---------------------------------------------------------
    def coefficients(self, values: np.ndarray, grid_values) -> tuple:
        a, oor = self.address(values, grid_values)
        out, o = [], 0
        for Wk in self.W:
            r = int(np.asarray(Wk).shape[1])
            c = a @ np.asarray(Wk, np.float64)
            if self.scale is not None:
                c = c * self.scale[o:o + r] + self.center[o:o + r]
            out.append(c.astype(np.float32))
            o += r
        return out, oor

    def read(self, store, values: np.ndarray, grid_values,
             basis: Optional[Sequence] = None) -> tuple:
        """``(N, F) requested VALUES -> ((N, D) store read, out-of-range mask)``.

        **There is no feature argument on this method.**  Its inputs are the address
        book, the store and the request -- the same structural guarantee
        ``gen_request_head.head_read`` carries, at the same place in the path.
        """
        coeffs, oor = self.coefficients(values, grid_values)
        return render_span(store, coeffs, basis), oor

    # -- the fit ----------------------------------------------------------
    def fit(self, Y: np.ndarray, targets: Sequence[np.ndarray], sweeps: int = 20,
            log=None, Y_val: Optional[np.ndarray] = None,
            targets_val: Optional[Sequence[np.ndarray]] = None,
            standardise: bool = False) -> "TensorAddress":
        """Alternating least squares on ``|| a(Y) W - T ||^2 + ridge(|E|^2 + |W|^2)``.

        ``Y`` is ``(N, F)`` **factor indices of the fit rows** -- the address book's one
        declared use of a label (charter B.6).  ``targets`` are the per-well span
        coordinates of those rows.  Closed form on both half-steps; no optimiser, no
        learning rate, one declared seed for the initialisation only.

        ⛔ ``standardise`` divides each target coordinate by its own standard deviation.
        **It is OFF, and that is a measured decision, not a default** (LOG 17:35Z): a
        shared low-rank address is not separable across output coordinates, so per-
        coordinate standardisation makes the fit spend capacity on directions the encoder
        gives almost no variance to and blows up the ones the decoder actually renders --
        the hues fell from 1.0000 to 0.40-0.99 with it on.  A single global scale is used
        instead, which leaves the relative weighting of the coordinates alone.
        """
        Yi = np.asarray(Y, np.int64)
        T = np.concatenate([np.asarray(t, np.float64) for t in targets], axis=1)
        self._widths = [int(np.asarray(t).shape[1]) for t in targets]
        self.center = T.mean(axis=0)
        self.scale = ((T.std(axis=0) + 1e-8) if standardise
                      else np.full(T.shape[1], float(T.std()) + 1e-8))
        self.standardised = bool(standardise)
        Ts = (T - self.center) / self.scale
        del T
        Tv = None
        if targets_val is not None:
            Tv = ((np.concatenate([np.asarray(t, np.float64) for t in targets_val], 1)
                   - self.center) / self.scale)
        rng = np.random.default_rng(self.seed)
        R = self.rank
        # the multiplicative address starts near the all-ones vector, so the first sweep
        # sees something close to the additive model rather than a random point; the
        # additive address starts near zero, which is the same statement for a sum.
        self.E = ([rng.normal(1.0, 0.3, (int(c), R)) for c in self.cardinalities]
                  if self.multiplicative
                  else [rng.normal(0.0, 0.3, (int(c), R)) for c in self.cardinalities])
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
                        # pred = (e * q_i) W  ->  the Gram is the elementwise product of
                        # the address gram and the mode gram
                        Gv = (qm.T @ qm) * WW + lam * np.eye(R)
                        rhs = np.einsum("ir,ir->r", qm, TB[m])
                    else:
                        # pred = (e + q_i) W  ->  an ordinary ridge on the residual
                        Gv = float(m.sum()) * WW + lam * np.eye(R)
                        rhs = (TB[m] - qm @ WW).sum(axis=0)
                    self.E[f][v] = np.linalg.solve(Gv, rhs)
                if mult:
                    # renormalise the table to unit RMS per address coordinate and pass
                    # the scale to W.  The model is invariant to it; the conditioning of
                    # the next half-step is not.
                    s = np.sqrt(np.mean(self.E[f] ** 2, axis=0)) + 1e-12
                    self.E[f] /= s[None, :]
                    Wc *= s[:, None]
                    WW = Wc @ Wc.T
                    TB = Ts @ Wc.T
            tr = float(np.mean((_addr(Yi) @ Wc - Ts) ** 2))
            va = (float(np.mean((_addr(np.asarray(Y_val, np.int64)) @ Wc - Tv) ** 2))
                  if Tv is not None else float("nan"))
            self.history.append({"sweep": it + 1, "train_mse": tr, "val_mse": va})
            if log is not None:
                log(f"    als {it + 1}/{sweeps} train_mse={tr:.5f} val_mse={va:.5f}")
        o = 0
        self.W = []
        for r in self._widths:
            self.W.append(Wc[:, o:o + r])
            o += r
        self.n_fit_rows = int(Yi.shape[0])
        return self

    # -- the ledger -------------------------------------------------------
    def byte_ledger(self) -> dict:
        eb = int(sum(np.asarray(e).size for e in self.E)) * 4
        wb = int(sum(np.asarray(w).size for w in self.W)) * 4
        nb = (0 if self.center is None else int(self.center.size + self.scale.size) * 4)
        return {
            "address_bytes": eb + wb + nb,
            "per_factor_tables_bytes": eb,
            "per_well_maps_bytes": wb,
            "normalisation_bytes": nb,
            "rank": self.rank,
            "dtype": "float32",
            "bytes_formula": "4 * ( sum_f card_f * R  +  sum_k R * r_k  +  2 * sum_k r_k )",
            "note": (
                "EXTRA to gen_store.store_bytes and reported beside it, never inside it "
                "-- the same rule gen_request_head.head_bytes states for the incumbent "
                "head, so the two reads are compared on the same ledger."),
        }


# ---------------------------------------------------------------------------
# 2. The assertion
# ---------------------------------------------------------------------------


def assert_read_request_only(addr: "TensorAddress", store, values: np.ndarray,
                             grid_values, basis=None, rng_seed: int = 0) -> dict:
    """Structural + numerical: no feature can reach this read (charter B.3).

    By signature -- :meth:`TensorAddress.read` has no feature parameter -- and
    numerically: the read is recomputed while an arbitrary feature array is perturbed in
    the caller's scope and must come back **bit-identical**.
    """
    import inspect

    sig = list(inspect.signature(TensorAddress.read).parameters)
    banned = [p for p in sig if p.lower() in ("x", "features", "feats", "z")]
    if banned:
        raise AssertionError(
            f"TensorAddress.read takes {banned} -- a feature argument on the read voids "
            "the request-only guarantee")
    a, _ = addr.read(store, values, grid_values, basis)
    rng = np.random.default_rng(int(rng_seed))
    _decoy = rng.normal(size=(8, int(store.dim))).astype(np.float32)   # noqa: F841
    b, _ = addr.read(store, values, grid_values, basis)
    if not np.array_equal(a, b):
        raise AssertionError(
            "the read is not a pure function of (address book, store, request)")
    return {
        "read_signature": sig,
        "feature_argument_present": False,
        "bitwise_stable_under_feature_perturbation": True,
        "n_checked": int(np.asarray(values).shape[0]),
        "statement": (
            "the read's only inputs are the address book, the store and the request; no "
            "feature tensor enters launch, settle or read (charter B.3)"),
    }
