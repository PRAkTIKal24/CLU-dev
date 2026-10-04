"""gen swap probe."""

from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np

PAIR_RNG_SEED = 20260919


def make_pairs(
    Y: np.ndarray,
    n_pairs: int,
    rng_seed: int = PAIR_RNG_SEED,
    require_all_differ: bool = True,
    max_tries: int = 2000000,
    vary: Optional[Sequence[int]] = None,
    fix: Optional[Sequence[int]] = None,
) -> np.ndarray:
    Yl = np.asarray(Y, np.int64)
    F = int(Yl.shape[1])
    fixl = [int(x) for x in fix or []]
    varyl = [int(x) for x in vary] if vary is not None else [f for f in range(F) if f not in fixl]
    rng = np.random.default_rng(int(rng_seed))
    n = int(Yl.shape[0])
    out = []
    tries = 0
    while len(out) < int(n_pairs) and tries < int(max_tries):
        a, b = (int(x) for x in rng.integers(0, n, 2))
        tries += 1
        if a == b:
            continue
        if fixl and (not bool((Yl[a, fixl] == Yl[b, fixl]).all())):
            continue
        if require_all_differ and varyl and (not bool((Yl[a, varyl] != Yl[b, varyl]).all())):
            continue
        out.append((a, b))
    if len(out) < int(n_pairs):
        raise RuntimeError(
            f"only {len(out)} of {n_pairs} pairs with vary={varyl} fix={fixl} in {tries} tries"
        )
    return np.asarray(out, np.int64)


def swap_read(
    judge, Xi: np.ndarray, Xj: np.ndarray, channels: np.ndarray, batch: int = 256
) -> np.ndarray:
    """``x_i`` with ``channels`` taken from ``x_j`` -> the judge's tuple.  ``(n, F)``."""
    ch = np.asarray(channels, np.int64)
    Xs = np.asarray(Xi, np.float32).copy()
    Xs[:, ch] = np.asarray(Xj, np.float32)[:, ch]
    return judge.read_back(Xs, int(batch))


def transfer_matrix(
    judge,
    Xi,
    Xj,
    Yi,
    Yj,
    labels: np.ndarray,
    k: int,
    batch: int = 256,
    log: Optional[Callable] = None,
) -> dict:
    """Per-well swap outcomes for one partition.  ``T[w, f] = P(factor f follows j)``.

    Returns ``T`` (follow-j), ``S`` (follow-i, i.e. *stayed*) and the raw tuples, so a
    caller can score any composite it likes without re-decoding.
    """
    lab = np.asarray(labels, np.int64)
    K = int(k)
    Yi_ = np.asarray(Yi, np.int64)
    F = int(Yi_.shape[1])
    T = np.zeros((K, F), np.float64)
    S = np.zeros((K, F), np.float64)
    tuples = np.zeros((K,) + tuple(Yi_.shape), np.int64)
    for w in range(K):
        ch = np.flatnonzero(lab == w)
        if ch.size == 0:
            raise ValueError(
                f"well {w} is empty -- a degenerate partition is reported, never silently merged"
            )
        t = swap_read(judge, Xi, Xj, ch, batch)
        tuples[w] = t
        T[w] = (t == np.asarray(Yj, np.int64)).mean(axis=0)
        S[w] = (t == Yi_).mean(axis=0)
        if log is not None:
            log(f"    well {w} ({ch.size:4d} ch): follow-j " + " ".join((f"{x:.3f}" for x in T[w])))
    return {"T": T, "S": S, "tuples": tuples, "well_sizes": np.bincount(lab, minlength=K).tolist()}


def derive_ownership(T: np.ndarray, vary: Optional[Sequence[int]] = None) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    Tm = np.asarray(T, np.float64)
    own = np.full(Tm.shape[0], -1, np.int64)
    if vary is None:
        rows, cols = linear_sum_assignment(-Tm)
        own[rows] = cols
        return own
    v = np.asarray(list(vary), np.int64)
    rows, cols = linear_sum_assignment(-Tm[:, v])
    own[rows] = v[cols]
    return own


def identity_ceiling(judge, Xi, Yi, batch: int = 256) -> dict:
    """The probe's own ceiling: no swap at all.  ``exact`` cannot exceed this.

    On real encoder features the judge is not perfect (``scale`` 0.9375, ``shape``
    0.9668 -- measured), so a partition that moved exactly the right factor and nothing
    else would still not score 1.0.  Every ``exact`` is reported against this number.
    """
    t = judge.read_back(np.asarray(Xi, np.float32), int(batch))
    Yl = np.asarray(Yi, np.int64)
    return {
        "exact": float((t == Yl).all(axis=1).mean()),
        "per_factor_tuple": (t == Yl).mean(axis=0).tolist(),
        "note": "no swap; the judge's own accuracy on real encoder features",
    }
