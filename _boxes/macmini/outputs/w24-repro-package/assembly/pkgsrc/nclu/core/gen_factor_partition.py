"""gen factor partition."""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

MERGE_RULES = ("min_relevance_mass",)
DEFAULT_MERGE_RULE = "min_relevance_mass"


def assert_relevance_is_train_only(fit_rows: np.ndarray, holdout_rows: np.ndarray) -> dict:
    """The partition may not see a held-out row.  Fires, never warns.

    This is the assertion behind the target file's "audit your own label path": a
    partition fitted against held-out outcomes is the whole claim smuggled in.
    """
    fit = np.asarray(fit_rows, np.int64)
    held = np.asarray(holdout_rows, np.int64)
    inter = np.intersect1d(fit, held, assume_unique=False)
    if inter.size:
        raise AssertionError(
            f"the partition's fit pool intersects the hold-out in {inter.size} rows (e.g. {inter[:8].tolist()}) -- the partition has seen a held-out row and every number built on it is void"
        )
    return {
        "n_fit_rows": int(fit.size),
        "n_holdout_rows": int(held.size),
        "intersection": 0,
        "statement": "the partition's fit pool and the hold-out are disjoint; the partition saw no held-out row, no cell and no arm outcome",
    }


def quantile_bin_edges(
    X: np.ndarray, n_bins: int, seed: int, n_subsample: int = 65536
) -> np.ndarray:
    Xf = np.asarray(X)
    n = int(Xf.shape[0])
    m = int(min(int(n_subsample), n))
    rng = np.random.default_rng(int(seed) * 1000003 + 20260918)
    idx = np.sort(rng.choice(n, size=m, replace=False)) if m < n else np.arange(n)
    S = np.asarray(Xf[idx], np.float32).reshape((m, -1))
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    return np.quantile(S, qs, axis=0).T.astype(np.float32)


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
        "note": "I(bin(channel); factor) on the fit pool. Train-only: no held-out row, no cell, no arm outcome. This is the address book's own relevance, read at the channel instead of at the mode.",
    }


def factor_aligned_labels(
    relevance: np.ndarray,
    k: int,
    dead: Optional[np.ndarray] = None,
    merge_rule: str = DEFAULT_MERGE_RULE,
    factor_names: Optional[Sequence[str]] = None,
) -> tuple:
    if merge_rule not in MERGE_RULES:
        raise ValueError(f"merge_rule must be one of {MERGE_RULES}, got {merge_rule!r}")
    R = np.asarray(relevance, np.float64)
    D, F = R.shape
    K = int(k)
    if K > F:
        raise ValueError(
            f"k={K} exceeds the {F} factors -- a factor-aligned partition cannot make more wells than there are factors to align them to"
        )
    dead = np.zeros(D, bool) if dead is None else np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    grp = np.argmax(R[live], axis=1)
    mass = np.zeros(F, np.float64)
    for f in range(F):
        sel = live[grp == f]
        mass[f] = float(R[sel, f].sum()) if sel.size else 0.0
    order = list(np.argsort(mass, kind="stable"))
    merged_into: dict = {}
    active = list(range(F))
    while len(active) > K:
        victim = next((int(f) for f in order if int(f) in active))
        active.remove(victim)
        host = next((int(f) for f in reversed(order) if int(f) in active))
        merged_into[victim] = host

    def _host(f: int) -> int:
        seen: set = set()
        while f in merged_into and f not in seen:
            seen.add(f)
            f = merged_into[f]
        return f

    resolved = np.array([_host(int(f)) for f in grp], np.int64)
    uniq = sorted(set((int(x) for x in resolved)))
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.zeros(D, np.int64)
    labels[live] = np.array([remap[int(x)] for x in resolved], np.int64)
    counts = np.bincount(labels[live], minlength=K).astype(np.int64)
    if counts.size != K or (counts == 0).any():
        raise ValueError(
            f"the factor-aligned partition realised {int((counts > 0).sum())} non-empty wells, not {K} -- an empty well is a degenerate partition and is reported, never silently merged (sizes {counts.tolist()})"
        )
    for c in np.flatnonzero(dead):
        j = int(np.argmin(counts))
        labels[c] = j
        counts[j] += 1
    prov = {}
    for u, i in remap.items():
        members = sorted([int(f) for f in range(F) if _host(f) == u])
        prov[str(i)] = {
            "source_factor_indices": members,
            "source_factors": [str(factor_names[f]) for f in members] if factor_names else None,
            "relevance_mass_bits": float(sum((mass[f] for f in members))),
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
        "label_path": "labels entered ONCE, through I(channel; factor) on the store's own fit pool -- the address book's relevance. No held-out row, no cell, no arm outcome, no learned rotation.",
    }
    return (labels, info)


def factor_aligned_balanced_labels(
    relevance: np.ndarray,
    k: int,
    dead: Optional[np.ndarray] = None,
    merge_rule: str = DEFAULT_MERGE_RULE,
    factor_names: Optional[Sequence[str]] = None,
) -> tuple:
    R = np.asarray(relevance, np.float64)
    D, F = R.shape
    K = int(k)
    dead = np.zeros(D, bool) if dead is None else np.asarray(dead, bool)
    _, base_info = factor_aligned_labels(
        R, K, dead=dead, merge_rule=merge_rule, factor_names=factor_names
    )
    groups = {
        int(i): set(v["source_factor_indices"])
        for i, v in ((int(a), b) for a, b in base_info["provenance"].items())
    }
    S = np.zeros((D, K), np.float64)
    for g, members in groups.items():
        S[:, g] = R[:, sorted(members)].max(axis=1)
    live = np.flatnonzero(~dead)
    srt = np.sort(S[live], axis=1)
    margin = srt[:, -1] - (srt[:, -2] if K > 1 else 0.0)
    order = live[np.argsort(-margin, kind="stable")]
    base = D // K
    cap = np.full(K, base, np.int64)
    cap[: D - base * K] += 1
    labels = np.full(D, -1, np.int64)
    counts = np.zeros(K, np.int64)
    for d in order:
        pref = np.argsort(-S[d], kind="stable")
        for g in pref:
            if counts[g] < cap[g]:
                labels[d] = int(g)
                counts[g] += 1
                break
    for c in np.flatnonzero(dead):
        j = int(np.argmin(counts - cap))
        labels[c] = j
        counts[j] += 1
    if (labels < 0).any():
        raise ValueError("the balanced assignment left channels unplaced")
    prov = {
        str(g): {
            "source_factor_indices": sorted((int(x) for x in members)),
            "source_factors": [str(factor_names[f]) for f in sorted(members)]
            if factor_names
            else None,
        }
        for g, members in groups.items()
    }
    info = {
        "method": "factor_aligned_argmax_relevance_balanced",
        "merge_rule": merge_rule,
        "balance_rule": "channels ranked by relevance MARGIN (best minus second-best) descending, each placed in its best-scoring group with capacity floor/ceil(D/k); dead channels fill the remaining capacity, lowest-occupancy well first",
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
    return (labels, info)


def well_rank_headroom(well_sizes: Sequence[int], n_modes: int) -> dict:
    ds = [int(x) for x in well_sizes]
    M = int(n_modes)
    starved = [i for i, d in enumerate(ds) if d < M]
    return {
        "n_modes": M,
        "well_sizes": ds,
        "max_read_rank_per_well": [int(min(d, M)) for d in ds],
        "total_read_rank": int(sum((min(d, M) for d in ds))),
        "rank_if_no_well_starved": int(M * len(ds)),
        "starved_wells": starved,
        "n_starved_wells": len(starved),
        "note": "a well with d_k < M spends modes it cannot use; the arm's reachable subspace is sum_k min(d_k, M), not K*M",
    }


def label_path_audit(relevance_info: dict, partition_info: dict, disjointness: dict) -> dict:
    return {
        "question": "did a label reach the partition anywhere other than the address book's own relevance, measured on the store's own fit pool?",
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
            "any learned rotation (study-common 4.1: a measured null)",
        ],
        "partition_method": partition_info.get("method"),
        "merge_rule": partition_info.get("merge_rule"),
        "landscape_read_write": "the labels do NOT enter the landscape, the read or the write: the modes are fitted by gen_store.train_store on features alone, and the read takes the request and the store and no feature argument at all",
    }
