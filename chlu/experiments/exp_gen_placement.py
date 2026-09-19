"""hub/placement -- WHERE THE WELLS ARE.  Candidate partitions and the swap-loss screen.

TARGET (w4 task file): **beat the label-MI partition -- which is an ABLATION baseline,
w4-common section 1 -- while HOLDING row 4 against the full rival set.**  Both halves.

THE DIAL, and nothing else moves.  Every candidate here is a label vector
``(2048,) -> {0..4}``.  ``K = 5``, ``n_modes = 64``, ``store_bytes = 64*2048*4 = 524,288``
are invariant under every one of them, so a placement change cannot buy itself bytes,
atoms, read width or head shape.  The read verb, the ridge, the fit pool, the decoder and
the classifier are the ones ``exp_gen_factor_wells`` pins.

THE BASELINE TO BEAT is ``mi_balanced`` -- ``gen_factor_partition.factor_aligned_balanced_labels``,
the label-MI partition, wave 3's winning arm.  Beating it is an **ABLATION** win and is
labelled so.  The RESULT half is row 4 against the rivals and is scored separately.

WHAT IS NEW HERE, in one line each:
  * ``gen_swap_probe`` -- a judge-scored loss on the dial: swap one well between two
    images, decode, and let the judge score the intended composite tuple.
  * the **normalised-relevance** family -- the label-MI partition ranks channels by raw
    bits, and bits are dominated by hue (measured: 41% of channels top-rank ``floor_hue``
    and 46% ``wall_hue``, against 0.6% ``shape`` and 0.2% ``scale``).  Rank/z/share
    normalisation asks which factor a channel is *relatively* most about.
  * the **independence** family (task item c) -- a between-well independence penalty plus
    a within-well variance floor.
  * the **two-stage refinement** (task items a+b) -- MI init, then greedy moves scored by
    the swap loss.

Runnable, every stage resumable and wall-clock capped::

    python -m chlu.experiments.exp_gen_placement --stage curve   --out DIR
    python -m chlu.experiments.exp_gen_placement --stage screen  --out DIR
    python -m chlu.experiments.exp_gen_placement --stage refine  --out DIR --init mi_balanced
    python -m chlu.experiments.exp_gen_placement --stage export  --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import exp_gen_factor_wells as W

#: The screen's pair count.  5 wells x N pairs decodes per candidate; at the measured
#: 161 img/s a 512-pair screen is ~16 s per candidate.  Declared, not tuned.
SCREEN_PAIRS = 512

#: The refinement's pair count -- smaller, because a refinement step re-measures only the
#: two wells a move touches.  The WINNER is always re-scored at SCREEN_PAIRS.
REFINE_PAIRS = 256

#: Every normalisation of the relevance matrix that this hub tries.  All deterministic,
#: all train-only, none of them a fitted parameter.
NORMALISATIONS = ("raw", "rank", "zscore", "share", "sqrt")

#: ``candidates.npz`` carries the inputs beside the label vectors so one file reproduces
#: the whole construction.  These keys are NOT candidates and are excluded everywhere a
#: candidate list is derived from the archive.
NON_LABEL_KEYS = frozenset({"mi_bits", "dead", "pool", "var"})


def labels_from_npz(z) -> dict:
    """Only the label vectors, in the archive's own order."""
    return {k: np.asarray(z[k], np.int64) for k in z.files
            if k not in NON_LABEL_KEYS}


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# 1. The relevance matrix and its normalisations
# ---------------------------------------------------------------------------


def normalise_relevance(R: np.ndarray, how: str, dead: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    """Re-scale ``R[d, f]`` ACROSS CHANNELS, per factor, before any assignment.

    The label-MI partition compares raw bits between factors.  That is a category error
    for placement: ``I(x_d; wall_hue)`` and ``I(x_d; scale)`` are not on the same scale
    because the factors do not carry the same entropy through this encoder, so an argmax
    over raw bits sends almost every channel to a hue well (measured: 87% of channels
    top-rank one of the two wall/floor hues).  Each normalisation below asks the same
    question in a different currency -- *which factor is this channel RELATIVELY most
    about* -- and none of them looks at a held-out row or an arm outcome.

    ``raw``    the label-MI partition's own currency, bits.  The baseline.
    ``rank``   per-factor rank of the channel, in [0, 1].  Scale-free, outlier-free.
    ``zscore`` per-factor standardisation.  Scale-free, keeps the gap structure.
    ``share``  ``R[d,f] / sum_f R[d,f]`` -- the channel's own budget, not the factor's.
    ``sqrt``   a soft compression between ``raw`` and ``rank``.
    """
    Rm = np.asarray(R, np.float64).copy()
    D, F = Rm.shape
    dead = np.zeros(D, bool) if dead is None else np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    if how == "raw":
        return Rm
    if how == "share":
        s = Rm.sum(axis=1, keepdims=True)
        return np.divide(Rm, np.where(s > 0, s, 1.0))
    if how == "sqrt":
        out = np.sqrt(np.maximum(Rm, 0.0))
        mx = out[live].max(axis=0, keepdims=True)
        return np.divide(out, np.where(mx > 0, mx, 1.0))
    out = np.zeros_like(Rm)
    for f in range(F):
        col = Rm[live, f]
        if how == "rank":
            o = np.argsort(np.argsort(col, kind="stable"), kind="stable")
            out[live, f] = o / max(1, col.size - 1)
        elif how == "zscore":
            sd = float(col.std())
            out[live, f] = (col - float(col.mean())) / (sd if sd > 0 else 1.0)
        else:
            raise ValueError(f"unknown normalisation {how!r}; want {NORMALISATIONS}")
    return out


def balanced_assign(S: np.ndarray, k: int, dead: Optional[np.ndarray] = None,
                    groups: Optional[dict] = None) -> np.ndarray:
    """Capacity-constrained assignment on a per-(channel, well) score ``S[d, w]``.

    ⭐ This is ``factor_aligned_balanced_labels``' OWN rule, lifted out verbatim so that
    every candidate in this hub is built by the identical construction and the ONLY thing
    that differs between them is the score matrix ``S``.  Channels are ranked by their
    margin (best minus second-best well score) descending; each takes its best well with
    capacity ``floor/ceil(D/k)``; dead channels fill what is left, lowest-occupancy first.
    Deterministic, train-only, no optimiser, no seed.
    """
    Sm = np.asarray(S, np.float64)
    D, K = Sm.shape
    if int(k) != K:
        raise ValueError(f"score matrix has {K} wells, k={k}")
    dead = np.zeros(D, bool) if dead is None else np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    srt = np.sort(Sm[live], axis=1)
    margin = srt[:, -1] - (srt[:, -2] if K > 1 else 0.0)
    order = live[np.argsort(-margin, kind="stable")]
    base = D // K
    cap = np.full(K, base, np.int64)
    cap[: D - base * K] += 1
    labels = np.full(D, -1, np.int64)
    counts = np.zeros(K, np.int64)
    for d in order:
        for g in np.argsort(-Sm[d], kind="stable"):
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
    return labels


def merged_group_scores(R: np.ndarray, k: int, dead: np.ndarray, merge_rule: str,
                        factor_names) -> tuple:
    """``(S[d, w], groups)`` -- the K merged factor groups, from the SHIPPED merge.

    The merge that folds ``F`` factors down to ``K`` wells is ``factor_aligned_labels``'
    own ``min_relevance_mass`` rule and is held FIXED across every candidate, so no
    candidate can win by choosing a different merge.  Only the score inside it moves.
    """
    from ..core.gen_factor_partition import factor_aligned_labels
    _, base_info = factor_aligned_labels(
        np.asarray(R, np.float64), int(k), dead=dead, merge_rule=merge_rule,
        factor_names=factor_names)
    groups = {int(i): sorted(int(x) for x in v["source_factor_indices"])
              for i, v in base_info["provenance"].items()}
    return groups, base_info


def score_from_relevance(R: np.ndarray, groups: dict, k: int) -> np.ndarray:
    """``S[d, w] = max_{f in group w} R[d, f]`` -- the shipped per-well score."""
    Rm = np.asarray(R, np.float64)
    S = np.zeros((Rm.shape[0], int(k)), np.float64)
    for g, members in groups.items():
        S[:, int(g)] = Rm[:, members].max(axis=1)
    return S


# ---------------------------------------------------------------------------
# 2. Task item (c): between-well independence + a within-well variance floor
# ---------------------------------------------------------------------------


def independence_scores(R: np.ndarray, corr: np.ndarray, groups: dict, k: int,
                        dead: np.ndarray, lam: float, var: np.ndarray,
                        var_floor_q: float = 0.25, norm: str = "rank") -> np.ndarray:
    """Task item (c), as a score matrix so it goes through the SAME assignment.

    Two terms, both train-only and both computed from the fit pool alone:

    **between-well independence.**  A well is read as a block: the head writes all of its
    channels together.  If channel ``d`` is strongly correlated with the channels already
    committed to well ``w'`` then putting ``d`` in a different well ``w`` makes the two
    wells move together, and two wells that move together cannot be composed
    independently.  So ``d``'s score for well ``w`` is penalised by its mean absolute
    correlation with every OTHER well's committed channels.  Implemented as one pass over
    a fixed reference partition's blocks (the MI argmax blocks), so the penalty is a
    property of the features and not of the optimiser's own trajectory.

    **within-well variance floor.**  A well whose channels barely move carries no content
    and spends 64 modes on a constant.  Channels below the ``var_floor_q`` quantile of
    live variance are barred from being the thing that fills a well: their score is
    flattened, so the margin rule places them last and they land wherever capacity is
    left, exactly as dead channels do.
    """
    Rn = normalise_relevance(R, norm, dead)
    S = score_from_relevance(Rn, groups, k)
    live = np.flatnonzero(~np.asarray(dead, bool))
    C = np.abs(np.asarray(corr, np.float64))
    # reference blocks: the argmax-of-normalised-relevance groups, fixed before any move
    ref = np.argmax(S, axis=1)
    pen = np.zeros_like(S)
    for w in range(int(k)):
        others = [c for c in live if ref[c] != w]
        if not others:
            continue
        pen[:, w] = C[:, others].mean(axis=1)
    S = S - float(lam) * pen
    v = np.asarray(var, np.float64)
    thr = float(np.quantile(v[live], float(var_floor_q))) if live.size else 0.0
    weak = v < thr
    S[weak] = S[weak].mean(axis=1, keepdims=True)
    return S


# ---------------------------------------------------------------------------
# 3. The candidate set
# ---------------------------------------------------------------------------


def build_candidates(out_dir: Path, log: Callable = print) -> dict:
    """Every candidate label vector this hub screens.  Resumable.

    The three inherited partitions (``mi_argmax``, ``mi_balanced``, ``ward``,
    ``ward_balanced``) are rebuilt HERE from the same inputs rather than copied, so the
    screen compares candidates that all came out of one code path on one fit pool.
    """
    art = out_dir / "candidates.npz"
    meta_p = out_dir / "candidates.json"
    if art.exists() and meta_p.exists():
        log("  candidates: already on disk, skipping")
        with open(meta_p, encoding="utf-8") as fh:
            return {"labels": labels_from_npz(np.load(art)), "info": json.load(fh)}

    from ..core.gen_factor_partition import (
        assert_relevance_is_train_only, channel_factor_relevance,
        factor_aligned_balanced_labels, factor_aligned_labels,
    )
    from ..core.gen_store import (
        adjusted_rand_index, block_diagonality, channel_correlation, correlation_grouping,
        dead_channels, random_grouping,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = W.load_condition()
    grid = cond["grid"]
    K = int(W.PINS["K"])
    D = int(W.PINS["D"])
    seed = int(W.PINS["seed"])
    F = W.features_mmap(seed)
    excl = W.val_corner_rows(cond) if W.PINS["condition"] == "val" else None
    pool = W.fit_pool(cond["train_rows"], int(W.PINS["n_pool"]),
                      int(W.PINS["pool_rng_seed"]), exclude=excl)
    held = (np.union1d(cond["test_rows"], excl) if excl is not None
            else cond["test_rows"])
    disj = assert_relevance_is_train_only(pool, held)
    log(f"  fit pool {pool.size} rows, {disj['intersection']} shared with the hold-out")
    X = W.gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    dead = dead_channels(X[: int(W.PINS["n_corr"])],
                         eps=float(W.PINS["dead_channel_eps"]))
    t = _now()
    rel = channel_factor_relevance(X, Y, cards, n_bins=int(W.PINS["relevance_n_bins"]),
                                   seed=seed)
    R = rel["mi_bits"]
    log(f"  relevance {R.shape} in {_now() - t:.1f}s; dead {int(dead.sum())}")
    corr = channel_correlation(X[: int(W.PINS["n_corr"])])
    var = X.var(axis=0)
    del X

    merge = str(W.PINS["merge_rule"])
    groups, base_info = merged_group_scores(R, K, dead, merge, list(grid.names))
    labels: dict = {}
    notes: dict = {}

    lab_mi, info_mi = factor_aligned_labels(R, K, dead=dead, merge_rule=merge,
                                            factor_names=list(grid.names))
    lab_bal, info_bal = factor_aligned_balanced_labels(
        R, K, dead=dead, merge_rule=merge, factor_names=list(grid.names))
    labels["mi_argmax"] = lab_mi
    labels["mi_balanced"] = lab_bal
    notes["mi_argmax"] = "factor_aligned_labels -- the label-MI partition, argmax of bits"
    notes["mi_balanced"] = ("factor_aligned_balanced_labels -- THE BASELINE TO BEAT "
                            "(wave 3's arm); label-MI, rebalanced to equal wells")

    lab_ward, info_ward = correlation_grouping(
        corr, K, method=str(W.PINS["grouping_method"]), dead=dead,
        min_well_frac=float(W.PINS["min_well_frac"]), return_info=True)
    labels["ward"] = lab_ward
    notes["ward"] = "the shipped correlation grouping -- basis_head's partition"

    from scipy.cluster.hierarchy import leaves_list, linkage
    live_idx = np.flatnonzero(~dead)
    Zl = linkage(np.abs(corr[np.ix_(live_idx, live_idx)]), method="ward")
    order = live_idx[leaves_list(Zl)]
    lab_wb = np.zeros(D, np.int64)
    base = order.size // K
    capw = np.full(K, base, np.int64)
    capw[: order.size - base * K] += 1
    pos = 0
    for j, c in enumerate(capw):
        lab_wb[order[pos:pos + int(c)]] = j
        pos += int(c)
    cnt = np.bincount(lab_wb[live_idx], minlength=K).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(cnt))
        lab_wb[c] = j
        cnt[j] += 1
    labels["ward_balanced"] = lab_wb
    notes["ward_balanced"] = ("the CONFOUND arm: correlation order, equal wells, no "
                             "label -- separates alignment from balance")

    # -- the normalised-relevance family -----------------------------------
    for how in NORMALISATIONS:
        if how == "raw":
            continue           # raw + balanced_assign IS mi_balanced, by construction
        Rn = normalise_relevance(R, how, dead)
        S = score_from_relevance(Rn, groups, K)
        labels[f"norm_{how}"] = balanced_assign(S, K, dead)
        notes[f"norm_{how}"] = (f"the shipped merge and the shipped balanced assignment, "
                                f"on {how}-normalised relevance instead of raw bits")

    # -- task item (c): independence + variance floor ------------------------
    for lam in (0.5, 1.0, 2.0):
        S = independence_scores(R, corr, groups, K, dead, lam, var)
        labels[f"indep_lam{lam:g}"] = balanced_assign(S, K, dead)
        notes[f"indep_lam{lam:g}"] = (
            f"task item (c): rank-normalised relevance minus {lam:g} x mean |corr| with "
            "every other well's block, plus a within-well variance floor at q=0.25")

    # -- the size-matched laundering floors ---------------------------------
    for nm in ("mi_balanced", "ward", "ward_balanced"):
        labels[f"random_{nm}"] = random_grouping(labels[nm], seed)
        notes[f"random_{nm}"] = (f"laundering floor for {nm}: the well-size histogram "
                                 "held EXACTLY, only which channel sits where is permuted")

    info = {
        "condition": W.PINS["condition"],
        "K": K, "D": D, "merge_rule": merge,
        "fit_pool": {"n": int(pool.size), "rng_seed": int(W.PINS["pool_rng_seed"]),
                     "first8": [int(x) for x in pool[:8]]},
        "disjointness": disj,
        "n_dead": int(dead.sum()),
        "relevance": {k: v for k, v in rel.items() if k != "mi_bits"},
        "relevance_top_factor_share": {
            str(grid.names[f]): float((np.argmax(R, axis=1) == f).mean())
            for f in range(len(cards))},
        "merged_groups": {str(g): [str(grid.names[f]) for f in m]
                          for g, m in groups.items()},
        "notes": notes,
        "well_sizes": {k: np.bincount(v, minlength=K).tolist() for k, v in labels.items()},
        "block_diagonality": {k: block_diagonality(corr, v) for k, v in labels.items()},
        "ARI_vs_mi_balanced": {
            k: float(adjusted_rand_index(v, labels["mi_balanced"]))
            for k, v in labels.items()},
        "label_path": (
            "labels enter ONCE, through I(bin(channel); factor) on the store's own fit "
            "pool, and at the address book. w4-common section 5: declared in every "
            "caption. No held-out row, no cell, no arm outcome."),
        "byte_ledger_invariance": {
            "store_bytes": int(W.PINS["n_modes"]) * D * 4,
            "budget_bytes": int(W.PINS["store_bytes"]),
            "note": "n_modes*D*4 does not depend on the partition -- every candidate "
                    "deploys the identical ledger",
        },
    }
    np.savez_compressed(art, mi_bits=R, dead=dead, pool=pool, var=var, **labels)
    W._jdump(info, meta_p)
    log(f"  candidates: {len(labels)} written -> {art}")
    return {"labels": labels, "info": info}


# ---------------------------------------------------------------------------
# 4. The screen -- the swap loss over every candidate, on ONE pinned pair set
# ---------------------------------------------------------------------------


def _probe_inputs(out_dir: Path, n_pairs: int, log: Callable = print):
    from ..core import gen_swap_probe as SP
    cond = W.load_condition()
    grid = cond["grid"]
    z = np.load(out_dir / "candidates.npz")
    pool = np.asarray(z["pool"], np.int64)
    Y = grid.factor_indices_of(pool)
    pairs = SP.make_pairs(Y, int(n_pairs))
    F = W.features_mmap(int(W.PINS["seed"]))
    Xi = W.gather(F, pool[pairs[:, 0]])
    Xj = W.gather(F, pool[pairs[:, 1]])
    log(f"  probe: {pairs.shape[0]} pinned pairs, all 6 factors differ in each")
    return grid, Xi, Xj, Y[pairs[:, 0]], Y[pairs[:, 1]], pairs


def screen(out_dir: Path, n_pairs: int = SCREEN_PAIRS, max_seconds: float = 5400.0,
           only: Optional[list] = None, log: Callable = print) -> dict:
    """Score every candidate with the swap loss.  Resumable per candidate."""
    from ..core import gen_swap_probe as SP
    cands = build_candidates(out_dir, log=log)
    art = out_dir / "screen.json"
    res = {}
    if art.exists():
        with open(art, encoding="utf-8") as fh:
            res = json.load(fh)
        log(f"  screen: resuming with {len(res.get('candidates', {}))} already scored")
    grid, Xi, Xj, Yi, Yj, pairs = _probe_inputs(out_dir, n_pairs, log=log)
    judge = W._JUDGE[0] if W._JUDGE else W.Judge(int(W.PINS["seed"]), log=log)
    if not W._JUDGE:
        W._JUDGE.append(judge)
    res.setdefault("candidates", {})
    if "ceiling" not in res:
        res["ceiling"] = SP.identity_ceiling(judge, Xi, Yi, int(W.PINS["eval_batch"]))
        log(f"  ceiling (no swap): exact={res['ceiling']['exact']:.4f}")
    res["meta"] = {"n_pairs": int(pairs.shape[0]), "condition": W.PINS["condition"],
                   "pair_rng_seed": SP.PAIR_RNG_SEED, "judge": judge.info,
                   "factor_names": list(grid.names)}
    names = list(cands["labels"]) if only is None else list(only)
    t0 = _now()
    for nm in names:
        if nm in res["candidates"]:
            continue
        if _now() - t0 > float(max_seconds):
            log("  screen: wall-clock cap hit -- state saved, rerun to continue")
            break
        s = SP.score_partition(judge, Xi, Xj, Yi, Yj,
                               np.asarray(cands["labels"][nm], np.int64),
                               int(W.PINS["K"]), int(W.PINS["eval_batch"]))
        s["candidate"] = nm
        s["note"] = cands["info"]["notes"].get(nm, "")
        res["candidates"][nm] = s
        log(f"  {nm:22s} exact={s['exact']:.4f} control={s['control']:.4f} "
            f"leak={s['leak']:.4f} own={s['own']}")
        W._jdump(res, art)
    W._jdump(res, art)
    return res


# ---------------------------------------------------------------------------
# 4b. THE ENCODER CEILING -- the task file's own question, made a measurement
#
# "A placement method cannot invent structure the encoder does not carry, so report what
#  fraction of your gain is placement and what is the ceiling."
#
# Three numbers bracket every placement result on this hub, all on the SAME pinned pairs:
#   FLOOR    the size-matched random partition -- what zero placement information buys.
#   REACHED  the candidate.
#   RELAXED  the RELAXED REFERENCE: each well is allowed the best 410 channels for its own
#            factor by relevance, with the disjointness constraint DROPPED, so a channel may
#            serve several wells at once.
#            CORRECTED 2026-09-19, before this number was reported: this is a RELAXED
#            REFERENCE and NOT a proven upper bound on `exact`.  Dropping disjointness does
#            relax the problem, but the channels are ordered by RELEVANCE, which maximises
#            the owned factor's TRANSFER and not `exact`; `exact` also pays for leakage onto
#            the other five factors.  A partition that transfers its factor slightly less
#            and leaks much less can therefore score a higher `exact` than this reference.
#            It bounds CONTROL, not `exact`, and it is reported in those words.
#   JUDGE    identity_ceiling -- no swap at all.  Nothing can beat the judge's own accuracy.
#
# The per-factor curve is the part that actually answers the question: if this encoder
# cannot move `scale` at well scale no matter which channels are chosen, then a placement
# that stops spending a well on `scale` is exploiting an encoder asymmetry rather than
# inventing structure, and the report must say so in those words.
# ---------------------------------------------------------------------------


def transfer_curve(out_dir: Path, sizes=(32, 64, 128, 256, 400, 800), n_pairs: int = 128,
                   norm: str = "rank", log: Callable = print) -> dict:
    """Per-factor swap transfer against channel-set SIZE, with a size-matched RANDOM row.

    This is the hub's first measurement and the licence for everything after it: if a
    size-matched random set transferred as much as a relevance-ranked one, the swap loss
    would be measuring the size of the intervention and not the dial.
    """
    from ..core import gen_swap_probe as SP
    art = out_dir / "curve.json"
    if art.exists():
        with open(art, encoding="utf-8") as fh:
            return json.load(fh)
    build_candidates(out_dir, log=log)
    z = np.load(out_dir / "candidates.npz")
    R = np.asarray(z["mi_bits"], np.float64)
    dead = np.asarray(z["dead"], bool)
    live = np.flatnonzero(~dead)
    grid, Xi, Xj, Yi, Yj, _ = _probe_inputs(out_dir, n_pairs, log=log)
    judge = W._JUDGE[0] if W._JUDGE else W.Judge(int(W.PINS["seed"]), log=log)
    if not W._JUDGE:
        W._JUDGE.append(judge)
    Rn = normalise_relevance(R, norm, dead)
    names = list(grid.names)
    rows = []
    for f, nm in enumerate(names):
        order = live[np.argsort(-Rn[live, f], kind="stable")]
        for sz in sizes:
            t = SP.swap_read(judge, Xi, Xj, order[:int(sz)], int(W.PINS["eval_batch"]))
            fj = (t == np.asarray(Yj)).mean(axis=0)
            rows.append({"set": nm, "kind": f"top_{norm}", "size": int(sz),
                         "follow_j": fj.tolist(), "own_transfer": float(fj[f])})
            log(f"    top_{norm}[{nm:11s}] s={sz:4d} own={fj[f]:.3f} "
                f"leak_max={float(np.max(np.delete(fj, f))):.3f}")
    rng = np.random.default_rng(SP.PAIR_RNG_SEED + 7)
    for sz in sizes:
        t = SP.swap_read(judge, Xi, Xj, rng.choice(live, int(sz), replace=False),
                         int(W.PINS["eval_batch"]))
        fj = (t == np.asarray(Yj)).mean(axis=0)
        rows.append({"set": "RANDOM", "kind": "size_matched_control", "size": int(sz),
                     "follow_j": fj.tolist(), "own_transfer": None})
        log(f"    RANDOM            s={sz:4d} max_follow_j={float(fj.max()):.3f}")
    out = {"rows": rows, "factor_names": names, "n_pairs": int(n_pairs),
           "normalisation": norm, "sizes": [int(x) for x in sizes],
           "per_factor_encoder_ceiling_at_any_size": {
               nm: max(r["own_transfer"] for r in rows if r["set"] == nm)
               for nm in names},
           "per_factor_encoder_ceiling_at_well_scale": {
               nm: next(r["own_transfer"] for r in rows
                        if r["set"] == nm and r["size"] == 400)
               for nm in names} if 400 in sizes else None,
           "feature_cache": W.feature_cache_manifest(int(W.PINS["seed"])),
           "note": ("the RANDOM row is the size-matched control: transfer that survives it "
                    "is alignment, transfer that does not is intervention size")}
    W._jdump(out, art)
    return out


def encoder_ceiling(out_dir: Path, n_pairs: int = SCREEN_PAIRS, norm: str = "rank",
                    log: Callable = print) -> dict:
    """FLOOR / REACHED / CEILING / JUDGE for the placement dial, on the pinned pairs.

    The relaxed reference drops disjointness: well ``w`` takes the best ``cap`` channels for
    its own factor by normalised relevance, and the same channel may be taken by several
    wells.  It is NOT an achievable configuration -- a channel cannot really sit in two
    wells -- so it is never reported as an arm.

    ⚠ CORRECTED BEFORE REPORTING: it is also **not a proven upper bound on** ``exact``.
    The relaxation is real, but the channel ordering is by RELEVANCE, which targets the
    owned factor's transfer, whereas ``exact`` additionally pays for leakage onto the other
    five factors.  So it bounds ``control`` and is a REFERENCE for ``exact``.  The honest
    encoder-ceiling statement is the per-factor one in ``transfer_curve``: among
    relevance-ranked sets, no set of this encoder moves ``scale`` past 0.172 at well scale.
    """
    from ..core import gen_swap_probe as SP
    art = out_dir / "ceiling.json"
    if art.exists():
        with open(art, encoding="utf-8") as fh:
            return json.load(fh)
    cands = build_candidates(out_dir, log=log)
    z = np.load(out_dir / "candidates.npz")
    R = np.asarray(z["mi_bits"], np.float64)
    dead = np.asarray(z["dead"], bool)
    live = np.flatnonzero(~dead)
    K = int(W.PINS["K"])
    D = int(W.PINS["D"])
    cap = D // K
    grid, Xi, Xj, Yi, Yj, _ = _probe_inputs(out_dir, n_pairs, log=log)
    judge = W._JUDGE[0] if W._JUDGE else W.Judge(int(W.PINS["seed"]), log=log)
    if not W._JUDGE:
        W._JUDGE.append(judge)
    Yi_ = np.asarray(Yi, np.int64)
    Yj_ = np.asarray(Yj, np.int64)
    Rn = normalise_relevance(R, norm, dead)
    names = list(grid.names)
    # the relaxed oracle, per FACTOR: every factor gets its own best cap channels
    per_factor = {}
    for f, nm in enumerate(names):
        order = live[np.argsort(-Rn[live, f], kind="stable")][:cap]
        t = SP.swap_read(judge, Xi, Xj, order, int(W.PINS["eval_batch"]))
        intended = Yi_.copy()
        intended[:, f] = Yj_[:, f]
        per_factor[nm] = {
            "exact_if_this_well_were_free": float((t == intended).all(axis=1).mean()),
            "own_transfer": float((t[:, f] == Yj_[:, f]).mean()),
            "n_channels": int(order.size),
        }
        log(f"    relaxed oracle well for {nm:11s}: exact="
            f"{per_factor[nm]['exact_if_this_well_were_free']:.4f} "
            f"transfer={per_factor[nm]['own_transfer']:.4f}")
    # a K=5 partition owns 5 of the 6 factors; the ceiling takes the best 5
    ex = sorted(((v["exact_if_this_well_were_free"], nm)
                 for nm, v in per_factor.items()), reverse=True)
    ceiling = float(np.mean([x for x, _ in ex[:K]]))
    ceiling_owned = [nm for _, nm in ex[:K]]
    judge_ceiling = SP.identity_ceiling(judge, Xi, Yi_, int(W.PINS["eval_batch"]))
    out = {
        "relaxed_oracle_per_factor": per_factor,
        "relaxed_reference_exact": ceiling,
        "relaxed_oracle_ceiling_exact": ceiling,   # kept: attribution reads this key
        "relaxed_oracle_owned_factors": ceiling_owned,
        "relaxed_oracle_unowned_factor": [nm for nm in names if nm not in ceiling_owned],
        "judge_ceiling_exact": judge_ceiling["exact"],
        "judge_per_factor_on_real_features": judge_ceiling["per_factor_tuple"],
        "cap_per_well": int(cap), "K": K, "n_pairs": int(n_pairs),
        "normalisation": norm,
        "feature_cache": W.feature_cache_manifest(int(W.PINS["seed"])),
        "what_this_bounds": (
            "CONTROL, not exact. Disjointness is dropped, which relaxes the problem, but "
            "the channels are ordered by RELEVANCE, which targets the owned factor's "
            "transfer and not the exact composite; exact also pays for leakage onto the "
            "other five factors, so a partition that leaks less can exceed this number. "
            "Reported as a RELAXED REFERENCE, never as a proven upper bound on exact."),
        "not_achievable_because": (
            "a channel cannot sit in two wells at once; this is never an arm"),
    }
    W._jdump(out, art)
    return out


def attribution(out_dir: Path, baseline: str = "mi_balanced",
                reached: Optional[str] = None, log: Callable = print) -> dict:
    """PLACEMENT versus ENCODER CEILING, as a fraction, from artifacts already on disk."""
    with open(out_dir / "screen.json", encoding="utf-8") as fh:
        sc = json.load(fh)
    ce = encoder_ceiling(out_dir, log=log)
    C = sc["candidates"]
    if reached is None:
        reached = max((k for k in C if not k.startswith("random_")),
                      key=lambda k: C[k]["exact"])
    floor_key = "random_" + baseline if "random_" + baseline in C else None
    floor = C[floor_key]["exact"] if floor_key else None
    b, a = C[baseline]["exact"], C[reached]["exact"]
    cap = ce["relaxed_oracle_ceiling_exact"]      # the RELAXED REFERENCE (see above)
    # ⭐ What the relaxed configuration DOES bound is CONTROL: every well is handed the best
    # relevance-ranked channels for its own factor with disjointness dropped, and control is
    # exactly the owned factor's transfer.  So control/control_bound is the defensible
    # "how much of what this encoder allows did placement actually take" number, and it is
    # the one the report quotes.
    pf = ce["relaxed_oracle_per_factor"]
    owned = ce["relaxed_oracle_owned_factors"]
    control_bound = float(np.mean([pf[nm]["own_transfer"] for nm in owned]))
    cb = {
        "control_bound_relaxed": control_bound,
        "control_bound_owned_factors": owned,
        "baseline_control": C[baseline]["control"],
        "reached_control": C[reached]["control"],
        "floor_control": (C[floor_key]["control"] if floor_key else None),
        "fraction_of_control_bound_taken_by_baseline":
            C[baseline]["control"] / control_bound if control_bound > 0 else None,
        "fraction_of_control_bound_taken_by_reached":
            C[reached]["control"] / control_bound if control_bound > 0 else None,
        "why_this_is_the_defensible_fraction": (
            "the relaxed configuration hands every well the best relevance-ranked channels "
            "for its own factor with disjointness dropped, and CONTROL is exactly that "
            "factor's transfer -- so it bounds control even though it does not bound exact"),
    }
    out = {
        "floor_size_matched_random": floor, "floor_candidate": floor_key,
        "baseline": baseline, "baseline_exact": b,
        "reached": reached, "reached_exact": a,
        "relaxed_reference": cap,
        "judge_ceiling": ce["judge_ceiling_exact"],
        "gain_over_baseline": a - b,
        "gain_over_floor": (a - floor) if floor is not None else None,
        "baseline_gain_over_floor": (b - floor) if floor is not None else None,
        "reached_exceeds_relaxed_reference": bool(a > cap),
        "reading_if_it_exceeds": (
            "the arm bought its exact by LEAKING LESS, not by transferring more: its control "
            "sits below the relaxed control bound while its exact sits above the relaxed "
            "configuration's exact. Decontamination, not transfer."),
        "control": cb,
        "leak": {"baseline": C[baseline]["leak"], "reached": C[reached]["leak"],
                 "floor": (C[floor_key]["leak"] if floor_key else None)},
        "reading": (
            "the numerator is what PLACEMENT bought on a fixed encoder. The denominator is "
            "a RELAXED REFERENCE, not a proven bound on exact: if reached EXCEEDS it, that "
            "is not a contradiction -- it means the arm bought its exact by leaking less "
            "rather than by transferring more, and the per-factor transfer curve is then "
            "the encoder-ceiling statement to quote."),
    }
    W._jdump(out, out_dir / "attribution.json")
    log("  attribution: " + json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in out.items() if k != "reading"}))
    return out


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True,
                   choices=("candidates", "screen", "curve", "ceiling",
                            "attribution", "refine", "export"))
    p.add_argument("--out", required=True)
    p.add_argument("--condition", choices=("val", "test"), default=None)
    p.add_argument("--pairs", type=int, default=None)
    p.add_argument("--only", default=None, help="comma-separated candidate names")
    p.add_argument("--init", default="mi_balanced")
    p.add_argument("--max-seconds", type=float, default=5400.0)
    return p


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.condition:
        W.PINS["condition"] = a.condition
    logf = open(out / f"{a.stage}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"stage={a.stage} out={out} condition={W.PINS['condition']}")
    only = a.only.split(",") if a.only else None
    if a.stage == "candidates":
        build_candidates(out, log=log)
    elif a.stage == "screen":
        screen(out, n_pairs=int(a.pairs or SCREEN_PAIRS),
               max_seconds=float(a.max_seconds), only=only, log=log)
    elif a.stage == "curve":
        transfer_curve(out, n_pairs=int(a.pairs or 128), log=log)
    elif a.stage == "ceiling":
        encoder_ceiling(out, n_pairs=int(a.pairs or SCREEN_PAIRS), log=log)
    elif a.stage == "attribution":
        attribution(out, reached=(only[0] if only else None), log=log)
    else:
        from . import exp_gen_placement_refine as RF
        RF.dispatch(a, out, log)
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
