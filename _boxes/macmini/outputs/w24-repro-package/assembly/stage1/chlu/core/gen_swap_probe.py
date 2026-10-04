"""hub/placement -- the SWAP PROBE: a judge-scored loss for WHERE the wells are.

DIAL DECLARATION.  The only object this module touches is the label vector
``labels: (D,) -> {0..K-1}`` that ``gen_store.well_channels_from_labels`` consumes.  It
does not touch the read verb, the atom count, the byte ledger (``n_modes*D*4`` is
invariant under every partition of ``D`` channels into ``K`` wells), the decoder, the
classifier or the content.

WHAT IT MEASURES.  Take two fit-pool images ``i`` and ``j`` whose factor tuples differ in
**every** factor.  Replace well ``w``'s channels of ``x_i`` with well ``w``'s channels of
``x_j``, leave every other channel alone, push the result through THE frozen decoder and
THE hash-asserted external classifier, and ask the judge what came out.  If the partition
is factor-modular, exactly the factors that well ``w`` carries follow ``j`` and every
other factor stays at ``i``.  That composite tuple is the **intended** answer and the
judge scores it, so the loss is a validity rate and not a proxy.

WHY IT IS NOT RECONSTRUCTION.  w4-common section 4.3: placement learned from
reconstruction alone was a NULL.  Reconstruction asks whether the channels can be
rebuilt; this asks whether a well can be **moved independently**, which is the property
rows 3 and 4 actually consume.  Nothing here optimises an MSE.

THE CONTROL IS PART OF THE MEASUREMENT.  A random partition with the SAME well-size
histogram is measured on the SAME pinned pairs at every call site, because a swap of 400
channels is a large intervention and the reader is owed the size-matched floor.

MEASURED, and it is why the probe works at all (``exp_gen_placement`` stage ``curve``):
at 400 channels a top-relevance set transfers its factor at up to 1.000 while a
size-matched RANDOM set transfers <= 0.086.  Transfer is alignment, not budget.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np
PAIR_RNG_SEED = 20_260_919
def make_pairs(Y: np.ndarray, n_pairs: int, rng_seed: int = PAIR_RNG_SEED,
               require_all_differ: bool = True, max_tries: int = 2_000_000,
               vary: Optional[Sequence[int]] = None,
               fix: Optional[Sequence[int]] = None) -> np.ndarray:
    """``(n_pairs, 2)`` index pairs into ``Y``, drawn to match a request distribution.

    Default (``vary``/``fix`` unset): the tuples differ in EVERY factor, so every factor's
    transfer is readable in one pass.  That is how this hub's first screen was run.

    MEASURED, AND IT COST THIS HUB ITS BAR (2026-09-19).  The symmetric draw treats the six
    factors alike; **the scoreboard does not.**  Every row-3 and row-4 corner holds ``shape``
    CONSTANT and varies ``scale`` across three values, so a probe that rewards moving
    ``shape`` selects a partition that spends its scarce fifth well on a factor the estimand
    never exercises -- which is exactly what happened: the selected arm gained +11.958
    novel tuples/corner on the 24 val corners and only **+1.388 (p = 0.135)** on the 49
    scored ones, failing a pre-registered bar of +1.0 with the CI95 low above +1.0.

    ⭐ ``vary`` / ``fix`` exist so a surrogate can be drawn from the SAME request
    distribution the estimand uses: ``fix=[shape_index]`` makes both images share a shape,
    and ``vary`` restricts which factors must differ.  A hub screening a dial against a
    corner-based estimand should pass them.
    """
    Yl = np.asarray(Y, np.int64)
    F = int(Yl.shape[1])
    fixl = [int(x) for x in (fix or [])]
    varyl = ([int(x) for x in vary] if vary is not None
             else [f for f in range(F) if f not in fixl])
    rng = np.random.default_rng(int(rng_seed))
    n = int(Yl.shape[0])
    out = []
    tries = 0
    while len(out) < int(n_pairs) and tries < int(max_tries):
        a, b = (int(x) for x in rng.integers(0, n, 2))
        tries += 1
        if a == b:
            continue
        if fixl and not bool((Yl[a, fixl] == Yl[b, fixl]).all()):
            continue
        if require_all_differ and varyl and not bool(
                (Yl[a, varyl] != Yl[b, varyl]).all()):
            continue
        out.append((a, b))
    if len(out) < int(n_pairs):
        raise RuntimeError(
            f"only {len(out)} of {n_pairs} pairs with vary={varyl} fix={fixl} "
            f"in {tries} tries")
    return np.asarray(out, np.int64)


def swap_read(judge, Xi: np.ndarray, Xj: np.ndarray, channels: np.ndarray,
              batch: int = 256) -> np.ndarray:
    """``x_i`` with ``channels`` taken from ``x_j`` -> the judge's tuple.  ``(n, F)``."""
    ch = np.asarray(channels, np.int64)
    Xs = np.asarray(Xi, np.float32).copy()
    Xs[:, ch] = np.asarray(Xj, np.float32)[:, ch]
    return judge.read_back(Xs, int(batch))


def transfer_matrix(judge, Xi, Xj, Yi, Yj, labels: np.ndarray, k: int,
                    batch: int = 256, log: Optional[Callable] = None) -> dict:
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
            raise ValueError(f"well {w} is empty -- a degenerate partition is reported, "
                             "never silently merged")
        t = swap_read(judge, Xi, Xj, ch, batch)
        tuples[w] = t
        T[w] = (t == np.asarray(Yj, np.int64)).mean(axis=0)
        S[w] = (t == Yi_).mean(axis=0)
        if log is not None:
            log(f"    well {w} ({ch.size:4d} ch): follow-j "
                + " ".join(f"{x:.3f}" for x in T[w]))
    return {"T": T, "S": S, "tuples": tuples,
            "well_sizes": np.bincount(lab, minlength=K).tolist()}


def derive_ownership(T: np.ndarray, vary: Optional[Sequence[int]] = None) -> np.ndarray:
    """``own[w]`` = the factor well ``w`` carries, by max-weight assignment on ``T``.

    ``K <= F`` here, so at least one factor is UNOWNED by construction -- with K=5 wells
    and 6 Shapes3D factors exactly one factor has no well of its own.  That is a property
    of the pinned ledger (K=5), not a choice this module makes, and it is reported.
    Ownership is derived from the MEASURED transfer, not asserted from provenance, so the
    score is defined for any partition including the random floors.
    """
    from scipy.optimize import linear_sum_assignment
    Tm = np.asarray(T, np.float64)
    own = np.full(Tm.shape[0], -1, np.int64)
    if vary is None:
        rows, cols = linear_sum_assignment(-Tm)
        own[rows] = cols
        return own
    # ⛔ MEASURED BUG, FIXED 2026-09-19 BEFORE ITS NUMBERS WERE REPORTED.  When a factor is
    # HELD CONSTANT across the pair (``make_pairs(fix=...)``), ``follow-j`` and ``follow-i``
    # are the SAME event for it, so ``T[w, f]`` is just the classifier's accuracy on that
    # factor -- large for EVERY well.  A Hungarian assignment over all factors then happily
    # hands some well "ownership" of the fixed factor for free, which corrupts the whole
    # decomposition.  Ownership and leakage are therefore restricted to the factors that
    # actually vary; the fixed ones can only ever be part of the "must stay" requirement,
    # which ``exact`` already enforces.
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
    return {"exact": float((t == Yl).all(axis=1).mean()),
            "per_factor_tuple": (t == Yl).mean(axis=0).tolist(),
            "note": "no swap; the judge's own accuracy on real encoder features"}


