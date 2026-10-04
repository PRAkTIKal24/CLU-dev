"""⭐ The request → mode addressing head, **fitted discriminatively**.

⛔ **What this module does NOT change.**  The forward pass is *the same function* as
:func:`chlu.core.gen_dream.address_requests`::

    score[n, k, m] = b[k, m] + sum_f  W[offset_f + r[n, f], k, m]

and the parameter tensor has *exactly* the shape
:class:`chlu.core.gen_dream.AddressBook` already carries — ``(K, M)`` plus one
``(V_f, K, M)`` block per factor.  ⭐ So the addressing head's byte ledger,
``gen_compose.address_book_bytes``' closed form ``(K*M + sum_f V_f*K*M)*4``, is
**unchanged to the byte**.  ⛔ Nothing here widens the channel, and nothing here is a
new byte claim.

⭐ **What it changes.**  How those numbers are *estimated*.
:func:`chlu.core.gen_dream.fit_address_book` sets them by **naive-Bayes counting** —
each factor's block is fitted **alone**, from the marginal ``P(mode | factor_f = v)``.
Summing those blocks at read time is a log-linear pool of per-factor marginals, which
assumes the factors are conditionally independent given the mode.  ⛔ They are not, and
a *composition request is a conjunction* — the one object such a pool cannot represent
without double-counting the evidence the factors share.

This module fits the **same** parameters by maximum conditional likelihood
(multinomial logistic regression, one well at a time), so shared evidence is divided
out rather than added up.  ⭐ Naive Bayes is a particular closed-form setting of these
very weights, so this estimator starts from it and can only improve on the objective;
``steps=0`` returns it bit-identically.

⛔ **Charter §B.6 (label-free representation) is respected exactly as the shipped head
respects it.**  Labels enter here and *only* here — the addressing head is the interface
where the words of a request are grounded.  The landscape (the modes), the write and the
read never see a label: the fitting targets are ``address_codes(store, X_store)``, which
are the *store's own* codes for its own train features, and the read is still
``gen_store.read_from_codes(store, codes)`` — signature ``(store, codes)``, so
``gen_store.assert_no_bypass`` passes unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "DiscriminativeAddressBook",
    "fit_discriminative_address_book",
    "address_requests_logit",
    "as_address_book",
]


@dataclass(frozen=True)
class DiscriminativeAddressBook:
    """``log_prior (K, M)`` + one ``log_cond (V_f, K, M)`` block per factor.

    ⭐ Deliberately the same two fields, the same shapes and the same dtypes as
    :class:`chlu.core.gen_dream.AddressBook`, so it can be handed to
    :func:`chlu.core.gen_compose.address_book_bytes` and to any consumer of the shipped
    book without a shim.
    """

    log_prior: np.ndarray
    log_cond: tuple
    fit_info: dict

    @property
    def n_wells(self) -> int:
        return int(self.log_prior.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.log_prior.shape[1])


def as_address_book(book):
    """Re-wrap as the shipped :class:`~chlu.core.gen_dream.AddressBook`.

    ⭐ So the head can be passed to ``request_posterior`` / ``address_requests`` /
    ``address_book_bytes`` verbatim — ⛔ no second read path, no second byte formula.
    """
    from .gen_dream import AddressBook

    return AddressBook(
        np.asarray(book.log_prior, np.float32),
        [np.asarray(c, np.float32) for c in book.log_cond],
    )


def _design(labels: np.ndarray, cardinalities: Sequence[int]) -> tuple:
    """Active one-hot column per (row, factor), plus the per-factor offsets."""
    Y = np.asarray(labels, np.int64)
    cards = [int(c) for c in cardinalities]
    offs = np.cumsum([0] + cards[:-1]).astype(np.int64)
    cols = (Y + offs[None, :]).astype(np.int64)
    return cols, offs, int(sum(cards))


def fit_discriminative_address_book(
    codes: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    *,
    l2: float = 1e-3,
    steps: int = 400,
    lr: float = 0.5,
    soft_targets: Optional[Sequence[np.ndarray]] = None,
    init_from_naive_bayes: bool = True,
    smoothing: float = 1.0,
    log=None,
) -> DiscriminativeAddressBook:
    """Fit ``(b, W)`` by maximum conditional likelihood, one well at a time.

    Args:
        codes: ``(n, K)`` the store's OWN codes for its OWN train features.
        labels: ``(n, F)`` the train rows' factor values.  ⭐ The only labels in the
            mechanism, and they stop here.
        cardinalities: ``V_f`` per factor.
        l2: ridge on ``W`` (⛔ not on the bias).  Selected on DEV, ⛔ never on a
            held-out set.
        soft_targets: optional ``[ (n, M) ]`` per-well soft assignments to distil from
            instead of the hard code (``gen_store.soft_weights``).  ⭐ Strictly more
            information about the store's own geometry, at zero byte cost.
        init_from_naive_bayes: start from the shipped estimator, so ``steps=0`` is
            bit-identical to it and the fit can only improve the objective.

    Returns:
        A :class:`DiscriminativeAddressBook` with the shipped shapes.
    """
    C = np.asarray(codes, np.int64)
    Y = np.asarray(labels, np.int64)
    n, K = C.shape
    M = int(C.max()) + 1 if C.size else 1
    if soft_targets is not None:
        M = max(M, int(np.asarray(soft_targets[0]).shape[1]))
    cols, offs, V = _design(Y, cardinalities)
    cards = [int(c) for c in cardinalities]
    F = len(cards)

    if init_from_naive_bayes:
        from .gen_dream import fit_address_book

        nb = fit_address_book(C, Y, cards, float(smoothing))
        b = np.asarray(nb.log_prior, np.float64).copy()
        W = np.concatenate(
            [np.asarray(lc, np.float64) - b[None, :, :] for lc in nb.log_cond], axis=0
        )
    else:
        b = np.zeros((K, M), np.float64)
        W = np.zeros((V, K, M), np.float64)

    # ⭐ All K wells are fitted at once, full-batch, in float32 BLAS.
    # The design is the SAME one-hot block matrix ``Xd (n x sum_f V_f)`` for every well,
    # so the forward pass is one GEMM ``Xd @ W`` and the gradient one GEMM ``Xd.T @ G``.
    # ``sum_f V_f`` is 57 on Shapes3D, so a DENSE GEMM beats both the sparse product and
    # the ``np.add.at`` gather this replaced (479 s -> seconds at n = 32,766, K = 8).
    # ⛔ Row-chunked with accumulation, so the peak array is ``(chunk, K*M)`` and a
    # K = 256 head does not materialise a 2 GB temporary.  Full-batch and deterministic:
    # ⛔ no minibatch sampling, so there is no extra seed to thread.
    Xd = np.zeros((n, V), np.float32)
    Xd[np.repeat(np.arange(n), F), cols.reshape(-1)] = 1.0
    Wf = np.ascontiguousarray(W.reshape(V, K * M), np.float32)
    bf = np.ascontiguousarray(b.reshape(1, K * M), np.float32)
    KM = K * M
    chunk = max(256, int(2 ** 22 // max(KM, 1)))

    T = None
    if soft_targets is not None:
        T = np.zeros((n, K, M), np.float32)
        for k in range(K):
            Tk = np.asarray(soft_targets[k], np.float32)
            T[:, k, : Tk.shape[1]] = Tk

    mW = np.zeros_like(Wf)
    vW = np.zeros_like(Wf)
    mb = np.zeros_like(bf)
    vb = np.zeros_like(bf)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    ar_k = np.arange(K)
    last_per_well = [None] * K
    for t in range(1, int(steps) + 1):
        gW = np.zeros_like(Wf)
        gb = np.zeros_like(bf)
        nll = np.zeros(K, np.float64)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            Xc = Xd[i:j]
            Z = (Xc @ Wf + bf).reshape(j - i, K, M)
            Z -= Z.max(axis=2, keepdims=True)
            np.exp(Z, out=Z)
            Z /= Z.sum(axis=2, keepdims=True)
            idx = (np.arange(j - i)[:, None], ar_k[None, :], C[i:j])
            if T is None:
                nll += -np.log(np.maximum(Z[idx], 1e-30)).sum(axis=0)
                Z[idx] -= 1.0
            else:
                Tc = T[i:j]
                nll += -(Tc * np.log(np.maximum(Z, 1e-30))).sum(axis=2).sum(axis=0)
                Z -= Tc
            G = Z.reshape(j - i, KM)
            gW += Xc.T @ G
            gb += G.sum(axis=0, keepdims=True)
        gW /= n
        gb /= n
        nll /= n
        gW += np.float32(l2) * Wf
        for arr, g, m_, v_ in ((Wf, gW, mW, vW), (bf, gb, mb, vb)):
            m_ *= beta1
            m_ += (1 - beta1) * g
            v_ *= beta2
            v_ += (1 - beta2) * (g * g)
            arr -= np.float32(lr) * (m_ / (1 - beta1 ** t)) / (
                np.sqrt(v_ / (1 - beta2 ** t)) + eps
            )
        last_per_well = nll.tolist()
        if log is not None and (t == 1 or t % max(1, int(steps) // 4) == 0):
            log(f"    step {t}/{steps} mean nll={float(np.mean(last_per_well)):.5f}")
    W = Wf.astype(np.float64).reshape(V, K, M)
    b = bf.astype(np.float64).reshape(K, M)
    hist = list(last_per_well)

    log_cond = []
    for f in range(F):
        blk = W[offs[f]: offs[f] + cards[f], :, :] + b[None, :, :]
        log_cond.append(np.asarray(blk, np.float32))
    return DiscriminativeAddressBook(
        log_prior=np.asarray(b, np.float32),
        log_cond=tuple(log_cond),
        fit_info={
            "estimator": "multinomial logistic regression, per well",
            "n_rows": int(n),
            "K": int(K),
            "M": int(M),
            "l2": float(l2),
            "steps": int(steps),
            "lr": float(lr),
            "soft_targets": bool(soft_targets is not None),
            "init_from_naive_bayes": bool(init_from_naive_bayes),
            "final_nll_per_well": [None if x is None else float(x) for x in hist],
            "parameter_shape_note": (
                "⭐ identical to chlu.core.gen_dream.AddressBook — "
                "(K*M + sum_f V_f*K*M)*4 bytes, unchanged"
            ),
        },
    )


def address_requests_logit(book, requests: np.ndarray) -> np.ndarray:
    """``(N, F) -> (N, K, M)`` log-scores.  ⭐ Delegates to the SHIPPED forward pass."""
    from .gen_dream import address_requests

    return address_requests(as_address_book(book), requests)
