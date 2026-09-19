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

from typing import Callable, Optional

import numpy as np

#: The pinned RNG stream for the pair draw.  ONE pair set serves every candidate
#: partition, so every comparison in this hub is PAIRED on the same (i, j) content.
PAIR_RNG_SEED = 20_260_919

#: Well-scale swap.  E1 measured that transfer switches on at ~400 channels and that
#: 1984/5 = 397 is the balanced partition's own well size, so the informative scale and
#: the deployment scale coincide.  Declared, not tuned.
WELL_SCALE_NOTE = "intervention scale = the well itself; no free size parameter"


def make_pairs(Y: np.ndarray, n_pairs: int, rng_seed: int = PAIR_RNG_SEED,
               require_all_differ: bool = True, max_tries: int = 2_000_000) -> np.ndarray:
    """``(n_pairs, 2)`` index pairs into ``Y`` whose tuples differ in EVERY factor.

    Every factor must differ or the swap is uninformative for that factor: ``follow-j``
    and ``follow-i`` would be the same event and the rate would be unreadable.
    """
    Yl = np.asarray(Y, np.int64)
    rng = np.random.default_rng(int(rng_seed))
    n = int(Yl.shape[0])
    out = []
    tries = 0
    while len(out) < int(n_pairs) and tries < int(max_tries):
        a, b = (int(x) for x in rng.integers(0, n, 2))
        tries += 1
        if a == b:
            continue
        if require_all_differ and not bool((Yl[a] != Yl[b]).all()):
            continue
        out.append((a, b))
    if len(out) < int(n_pairs):
        raise RuntimeError(
            f"only {len(out)} of {n_pairs} all-factors-differ pairs in {tries} tries")
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


def derive_ownership(T: np.ndarray) -> np.ndarray:
    """``own[w]`` = the factor well ``w`` carries, by max-weight assignment on ``T``.

    ``K <= F`` here, so at least one factor is UNOWNED by construction -- with K=5 wells
    and 6 Shapes3D factors exactly one factor has no well of its own.  That is a property
    of the pinned ledger (K=5), not a choice this module makes, and it is reported.
    Ownership is derived from the MEASURED transfer, not asserted from provenance, so the
    score is defined for any partition including the random floors.
    """
    from scipy.optimize import linear_sum_assignment
    Tm = np.asarray(T, np.float64)
    rows, cols = linear_sum_assignment(-Tm)
    own = np.full(Tm.shape[0], -1, np.int64)
    own[rows] = cols
    return own


def score_partition(judge, Xi, Xj, Yi, Yj, labels: np.ndarray, k: int,
                    batch: int = 256, log: Optional[Callable] = None,
                    own: Optional[np.ndarray] = None) -> dict:
    """The swap loss, in full, with every component a reader needs to check it.

    ``exact`` is the headline: the fraction of (well, pair) swaps whose decoded image the
    judge reads back as **exactly** the intended composite tuple -- ``y_i`` with well
    ``w``'s owned factor replaced by ``y_j``'s.  It is a judge-scored validity rate, so
    it is on the same footing as the scoreboard's own rows.

    ``control`` / ``leak`` decompose it: ``control`` is how much of the owned factor
    actually moves and ``leak`` is how much of everything else moves with it.  A
    partition can buy ``control`` with ``leak`` and ``exact`` will not pay for it.

    ``own`` may be passed to FIX the ownership (e.g. from the initialisation's
    provenance) instead of deriving it from this partition's own transfer; a refinement
    loop must fix it, or the objective moves under the optimiser.
    """
    tm = transfer_matrix(judge, Xi, Xj, Yi, Yj, labels, k, batch=batch, log=log)
    T, S = tm["T"], tm["S"]
    derived = derive_ownership(T)
    own_used = derived if own is None else np.asarray(own, np.int64)
    K, F = T.shape
    Yi_ = np.asarray(Yi, np.int64)
    Yj_ = np.asarray(Yj, np.int64)
    exact_w = np.zeros(K, np.float64)
    perfac_w = np.zeros(K, np.float64)
    for w in range(K):
        intended = Yi_.copy()
        f = int(own_used[w])
        if f >= 0:
            intended[:, f] = Yj_[:, f]
        hit = tm["tuples"][w] == intended
        exact_w[w] = float(hit.all(axis=1).mean())
        perfac_w[w] = float(hit.mean())
    control = float(np.mean([T[w, own_used[w]] for w in range(K) if own_used[w] >= 0]))
    off = [(w, f) for w in range(K) for f in range(F) if f != own_used[w]]
    leak = float(np.mean([T[w, f] for w, f in off]))
    return {
        "exact": float(exact_w.mean()),
        "per_factor": float(perfac_w.mean()),
        "control": control,
        "leak": leak,
        "control_minus_leak": control - leak,
        "own": own_used.tolist(),
        "own_derived": derived.tolist(),
        "own_was_fixed": own is not None,
        "exact_per_well": exact_w.tolist(),
        "T": T.tolist(), "S": S.tolist(),
        "owned_transfer": [float(T[w, own_used[w]]) if own_used[w] >= 0 else None
                           for w in range(K)],
        "well_sizes": tm["well_sizes"],
        "n_pairs": int(Yi_.shape[0]),
        "n_decodes": int(K * Yi_.shape[0]),
    }


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


def size_matched_random(labels: np.ndarray, seed: int) -> np.ndarray:
    """A random relabelling that holds the well-size histogram EXACTLY.

    Identical in spirit to ``gen_store.random_grouping``: the floor differs from the arm
    only in which coordinate sits in which well.
    """
    lab = np.asarray(labels, np.int64)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(lab.size)
    out = np.empty_like(lab)
    out[perm] = lab
    return out
