"""FACTOR-ALIGNED WELLS -- the partition, and nothing else.

DIAL DECLARATION -- *how the store is ORGANISED*.  Every other stage is held frozen:
the encoder, the feature cache, the decoder, the judge, the mode count, the byte ledger,
the read verb and the addressing head.  The **only** thing this module produces is the
``D -> K`` ``labels`` vector that :func:`chlu.core.gen_store.well_channels_from_labels`
turns into ``well_channels``.  An arm built on it differs from the shipped ward arm in
**the label vector and nothing else** -- which is the discipline the shipped store plan's
own note already imposes ("only the grouping (the `labels` field) is allowed to change
between rungs").

THE LABEL PATH, declared here because it is the thing that can void the claim.
Charter section B.6: labels ground the *words* of a request at the interface only -- the
address book -- never in the landscape, the read or the write.  This module uses labels in
exactly one place: :func:`channel_factor_relevance` measures ``I(channel_d ; factor_f)`` on
the **store's own fit pool**, which is the same quantity the address book is fitted from
and the same rows it is fitted on.  No held-out row, no cell, no arm outcome and no scored
quantity enters it -- :func:`label_path_audit` records that as a machine-checkable
statement and :func:`assert_relevance_is_train_only` fires if the row set handed in
intersects the hold-out.

Learned placement is a MEASURED NULL (w3-hub-common section 4.1) and nothing here learns a
rotation: the assignment is an argmax of a measured mutual information, closed form, no
optimiser, no seed of its own beyond the binning subsample.

The floor is not optional.  ``gen_store.random_grouping`` permutes the label vector
produced here, holding the well-size histogram exactly, so the only thing that varies
between an arm and its floor is which coordinate sits in which well.  Every partition
number is a **margin over that floor**, never an absolute.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: The merge rules, declared in code so none can be swapped in after a result.
#: Shapes3D realises SIX factors and the pinned plan realises FIVE wells; a rule that
#: closes that gap is a free parameter unless it is named in advance.
MERGE_RULES = ("min_relevance_mass",)

#: The declared default: merge the TWO groups carrying the least relevance mass.
#: Train-only, deterministic, and it spends the scarce well on the factors that carry
#: the most measured information rather than on a tie-break.
DEFAULT_MERGE_RULE = "min_relevance_mass"

__all__ = [
    "DEFAULT_MERGE_RULE",
    "MERGE_RULES",
    "assert_relevance_is_train_only",
    "channel_factor_relevance",
    "factor_aligned_balanced_labels",
    "factor_aligned_labels",
    "label_path_audit",
    "quantile_bin_edges",
    "well_rank_headroom",
]


def assert_relevance_is_train_only(
    fit_rows: np.ndarray, holdout_rows: np.ndarray
) -> dict:
    """The partition may not see a held-out row.  Fires, never warns.

    This is the assertion behind the target file's "audit your own label path": a
    partition fitted against held-out outcomes is the whole claim smuggled in.
    """
    fit = np.asarray(fit_rows, np.int64)
    held = np.asarray(holdout_rows, np.int64)
    inter = np.intersect1d(fit, held, assume_unique=False)
    if inter.size:
        raise AssertionError(
            f"the partition's fit pool intersects the hold-out in {inter.size} rows "
            f"(e.g. {inter[:8].tolist()}) -- the partition has seen a held-out row and "
            "every number built on it is void"
        )
    return {
        "n_fit_rows": int(fit.size),
        "n_holdout_rows": int(held.size),
        "intersection": 0,
        "statement": (
            "the partition's fit pool and the hold-out are disjoint; the partition "
            "saw no held-out row, no cell and no arm outcome"
        ),
    }


def quantile_bin_edges(
    X: np.ndarray, n_bins: int, seed: int, n_subsample: int = 65_536
) -> np.ndarray:
    """``(D, n_bins - 1)`` interior quantile edges per channel.

    Drawn from a seeded subsample of the fit pool so the edges are reproducible from
    ``(pool, seed, n_subsample, n_bins)`` alone, which is what "pin the store fit
    completely" means for this step.
    """
    Xf = np.asarray(X)
    n = int(Xf.shape[0])
    m = int(min(int(n_subsample), n))
    rng = np.random.default_rng(int(seed) * 1_000_003 + 20_260_918)
    idx = np.sort(rng.choice(n, size=m, replace=False)) if m < n else np.arange(n)
    S = np.asarray(Xf[idx], np.float32).reshape((m, -1))
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    return np.quantile(S, qs, axis=0).T.astype(np.float32)  # (D, n_bins-1)


def channel_factor_relevance(
    X: np.ndarray,
    Y: np.ndarray,
    cardinalities: Sequence[int],
    n_bins: int = 8,
    seed: int = 0,
    chunk: int = 8192,
) -> dict:
    """``R[d, f] = I(bin(channel_d) ; factor_f)`` in bits, on the fit pool.

    This is the SAME quantity the address book is built from (``gen_dream``'s
    ``fit_address_book`` fits ``log P(code_k | f = v)`` on train codes and train
    labels) -- read one level lower, at the channel rather than at the mode.  It is
    the address book's own relevance, and it is where labels are allowed to be.

    Nothing here is optimised.  Histograms, closed form, no gradient, no rotation.
    """
    Xf = np.asarray(X)
    n = int(Xf.shape[0])
    D = int(np.prod(Xf.shape[1:]))
    Yl = np.asarray(Y, np.int64)
    cards = [int(c) for c in cardinalities]
    F = len(cards)
    B = int(n_bins)
    edges = quantile_bin_edges(Xf, B, seed)
    joint = [np.zeros((D, B, c), np.int64) for c in cards]
    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        Z = np.asarray(Xf[s:e], np.float32).reshape((e - s, D))
        # per-channel digitize, vectorised: count how many edges each value exceeds
        bins = np.zeros((e - s, D), np.int16)
        for b in range(B - 1):
            bins += (Z >= edges[None, :, b]).astype(np.int16)
        base = (np.arange(D, dtype=np.int64) * B)[None, :] + bins.astype(np.int64)
        for f in range(F):
            c = cards[f]
            idx = (base * c + Yl[s:e, f][:, None]).ravel()
            joint[f] += np.bincount(idx, minlength=D * B * c).reshape((D, B, c))
    R = np.zeros((D, F), np.float64)
    for f in range(F):
        J = joint[f].astype(np.float64) / float(n)
        pd_ = J.sum(axis=2, keepdims=True)
        pf = J.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = J * np.log2(J / (pd_ * pf))
        R[:, f] = np.nansum(np.where(J > 0, t, 0.0), axis=(1, 2))
    R = np.maximum(R, 0.0)
    return {
        "mi_bits": R,
        "n_bins": B,
        "n_rows": n,
        "seed": int(seed),
        "binned_channel_entropy_ceiling_bits": float(np.log2(B)),
        "note": (
            "I(bin(channel); factor) on the fit pool. Train-only: no held-out row, "
            "no cell, no arm outcome. This is the address book's own relevance, read "
            "at the channel instead of at the mode."
        ),
    }


def factor_aligned_labels(
    relevance: np.ndarray,
    k: int,
    dead: Optional[np.ndarray] = None,
    merge_rule: str = DEFAULT_MERGE_RULE,
    factor_names: Optional[Sequence[str]] = None,
) -> tuple:
    """``(labels, info)`` -- each live channel to the factor it carries most.

    **Wells carry a factor; modes are its values.**  A channel joins well ``f`` iff
    ``f = argmax_f R[d, f]``; the factor groups are then merged down to exactly ``k``
    by ``merge_rule`` so that ``K``, the byte ledger (``M*D*4``) and the read channel
    width (``K*log2 M``) are **identical to the shipped plan** and the label vector is
    the only thing that has moved.

    Standing convention (``gen_store``): no well is ever named semantically in an
    artifact.  The factor a group came from is recorded in ``info["provenance"]`` as
    book-keeping for the audit; wells stay integer-indexed everywhere else.
    """
    if merge_rule not in MERGE_RULES:
        raise ValueError(f"merge_rule must be one of {MERGE_RULES}, got {merge_rule!r}")
    R = np.asarray(relevance, np.float64)
    D, F = R.shape
    K = int(k)
    if K > F:
        raise ValueError(
            f"k={K} exceeds the {F} factors -- a factor-aligned partition cannot "
            "make more wells than there are factors to align them to"
        )
    dead = np.zeros(D, bool) if dead is None else np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    grp = np.argmax(R[live], axis=1)              # the factor each channel carries
    mass = np.zeros(F, np.float64)
    for f in range(F):
        sel = live[grp == f]
        mass[f] = float(R[sel, f].sum()) if sel.size else 0.0
    # the declared merge: fold the least-relevant groups together until K remain.
    order = list(np.argsort(mass, kind="stable"))  # ascending mass
    merged_into: dict = {}
    active = list(range(F))
    while len(active) > K:
        victim = next(int(f) for f in order if int(f) in active)
        active.remove(victim)
        host = next(int(f) for f in reversed(order) if int(f) in active)
        merged_into[victim] = host

    def _host(f: int) -> int:
        seen: set = set()
        while f in merged_into and f not in seen:
            seen.add(f)
            f = merged_into[f]
        return f

    resolved = np.array([_host(int(f)) for f in grp], np.int64)
    uniq = sorted(set(int(x) for x in resolved))
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.zeros(D, np.int64)
    labels[live] = np.array([remap[int(x)] for x in resolved], np.int64)
    counts = np.bincount(labels[live], minlength=K).astype(np.int64)
    if counts.size != K or (counts == 0).any():
        raise ValueError(
            f"the factor-aligned partition realised {int((counts > 0).sum())} "
            f"non-empty wells, not {K} -- an empty well is a degenerate partition and "
            f"is reported, never silently merged (sizes {counts.tolist()})"
        )
    # dead coordinates into the smallest well -- information-free, and it keeps a
    # whole well from being spent on constants.  Identical rule to correlation_grouping.
    for c in np.flatnonzero(dead):
        j = int(np.argmin(counts))
        labels[c] = j
        counts[j] += 1
    prov = {}
    for u, i in remap.items():
        members = sorted([int(f) for f in range(F) if _host(f) == u])
        prov[str(i)] = {
            "source_factor_indices": members,
            "source_factors": (
                [str(factor_names[f]) for f in members] if factor_names else None
            ),
            "relevance_mass_bits": float(sum(mass[f] for f in members)),
        }
    info = {
        "method": "factor_aligned_argmax_relevance",
        "merge_rule": merge_rule,
        "k": K,
        "n_factors": F,
        "relevance_mass_bits_per_factor": [float(x) for x in mass],
        "merged_into": {str(a): int(b) for a, b in merged_into.items()},
        "provenance": prov,
        "n_dead_channels": int(dead.sum()),
        "n_live_channels": int(live.size),
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
        "label_path": (
            "labels entered ONCE, through I(channel; factor) on the store's own fit "
            "pool -- the address book's relevance. No held-out row, no cell, no arm "
            "outcome, no learned rotation."
        ),
    }
    return labels, info


def factor_aligned_balanced_labels(
    relevance: np.ndarray,
    k: int,
    dead: Optional[np.ndarray] = None,
    merge_rule: str = DEFAULT_MERGE_RULE,
    factor_names: Optional[Sequence[str]] = None,
) -> tuple:
    """``(labels, info)`` -- factor-aligned, then rebalanced to EQUAL wells.

    Why it exists, declared before any arm was fitted: on the frozen Shapes3D encoder
    the argmax partition is measured to be extremely unequal (4 channels top-ranked by
    ``scale``, 12 by ``shape``, against 935 by ``wall_hue``), so two of its five wells
    fall below ``M`` and **spend modes they cannot use** (section 4.10).  The target file
    sanctions a balanced variant explicitly: *"a balanced variant cutting the dendrogram
    leaf order gives equal wells at any K. Both keep the correlation structure and
    neither sees a label."*  This is the same idea applied to the relevance ordering
    rather than to the dendrogram leaf order.

    The rule, in full, so it is not a free parameter: every live channel is ranked by its
    **relevance margin** ``max_f R[d, f] - second_f R[d, f]`` descending -- the channels
    most unambiguously about one factor go first -- and each is placed in its argmax
    group while that group still has capacity ``ceil/floor(D/k)``, otherwise in its
    best-ranked group that does.  Deterministic, train-only, no optimiser, no seed.
    """
    R = np.asarray(relevance, np.float64)
    D, F = R.shape
    K = int(k)
    dead = np.zeros(D, bool) if dead is None else np.asarray(dead, bool)
    # the same merge as the argmax variant, so the two differ ONLY in the balancing
    _, base_info = factor_aligned_labels(
        R, K, dead=dead, merge_rule=merge_rule, factor_names=factor_names)
    groups = {int(i): set(v["source_factor_indices"])
              for i, v in ((int(a), b) for a, b in base_info["provenance"].items())}
    # per-channel score of each MERGED group = the best relevance it contributes
    S = np.zeros((D, K), np.float64)
    for g, members in groups.items():
        S[:, g] = R[:, sorted(members)].max(axis=1)
    live = np.flatnonzero(~dead)
    srt = np.sort(S[live], axis=1)
    margin = srt[:, -1] - (srt[:, -2] if K > 1 else 0.0)
    order = live[np.argsort(-margin, kind="stable")]
    base = D // K
    cap = np.full(K, base, np.int64)
    cap[: D - base * K] += 1                      # the remainder to the low-index wells
    labels = np.full(D, -1, np.int64)
    counts = np.zeros(K, np.int64)
    for d in order:
        pref = np.argsort(-S[d], kind="stable")
        for g in pref:
            if counts[g] < cap[g]:
                labels[d] = int(g)
                counts[g] += 1
                break
    # dead channels fill whatever capacity is left, lowest-occupancy well first
    for c in np.flatnonzero(dead):
        j = int(np.argmin(counts - cap))
        labels[c] = j
        counts[j] += 1
    if (labels < 0).any():
        raise ValueError("the balanced assignment left channels unplaced")
    prov = {
        str(g): {
            "source_factor_indices": sorted(int(x) for x in members),
            "source_factors": (
                [str(factor_names[f]) for f in sorted(members)] if factor_names else None
            ),
        }
        for g, members in groups.items()
    }
    info = {
        "method": "factor_aligned_argmax_relevance_balanced",
        "merge_rule": merge_rule,
        "balance_rule": (
            "channels ranked by relevance MARGIN (best minus second-best) descending, "
            "each placed in its best-scoring group with capacity floor/ceil(D/k); dead "
            "channels fill the remaining capacity, lowest-occupancy well first"
        ),
        "k": K,
        "n_factors": F,
        "capacity_per_well": [int(x) for x in cap],
        "provenance": prov,
        "n_dead_channels": int(dead.sum()),
        "n_live_channels": int(live.size),
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
        "unbalanced_well_sizes": base_info["well_sizes"],
        "label_path": base_info["label_path"],
    }
    return labels, info


def well_rank_headroom(well_sizes: Sequence[int], n_modes: int) -> dict:
    """w3-hub-common section 4.10: modes beyond a well's channel dimension buy nothing.

    A well with ``d_k < M`` cannot use its ``M`` modes -- its read span is capped at
    ``d_k``.  Reported, never repaired silently: a partition that starves a well is a
    property of the partition and is part of the result.
    """
    ds = [int(x) for x in well_sizes]
    M = int(n_modes)
    starved = [i for i, d in enumerate(ds) if d < M]
    return {
        "n_modes": M,
        "well_sizes": ds,
        "max_read_rank_per_well": [int(min(d, M)) for d in ds],
        "total_read_rank": int(sum(min(d, M) for d in ds)),
        "rank_if_no_well_starved": int(M * len(ds)),
        "starved_wells": starved,
        "n_starved_wells": len(starved),
        "note": (
            "a well with d_k < M spends modes it cannot use; the arm's reachable "
            "subspace is sum_k min(d_k, M), not K*M"
        ),
    }


def label_path_audit(
    relevance_info: dict, partition_info: dict, disjointness: dict
) -> dict:
    """The one artifact a referee reads to decide whether the claim is void."""
    return {
        "question": (
            "did a label reach the partition anywhere other than the address book's "
            "own relevance, measured on the store's own fit pool?"
        ),
        "answer": "no",
        "the_one_label_use": {
            "what": "I(bin(channel_d); factor_f), mutual information in bits",
            "rows": "the store's fit pool only",
            "n_rows": relevance_info.get("n_rows"),
            "n_bins": relevance_info.get("n_bins"),
            "seed": relevance_info.get("seed"),
        },
        "disjointness_assertion": disjointness,
        "what_did_NOT_enter": [
            "any held-out row",
            "any cell membership",
            "any arm's score",
            "any judged outcome",
            "any learned rotation (w3-hub-common 4.1: a measured null)",
        ],
        "partition_method": partition_info.get("method"),
        "merge_rule": partition_info.get("merge_rule"),
        "landscape_read_write": (
            "the labels do NOT enter the landscape, the read or the write: the modes "
            "are fitted by gen_store.train_store on features alone, and the read takes "
            "the request and the store and no feature argument at all"
        ),
    }
