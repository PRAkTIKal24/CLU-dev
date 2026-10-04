"""hub/composition-2 — THE HEADLINE COLUMN and the fast head-search.

SELECTION ONLY. Nothing here is a scored number and `eval_b` / `eval_a` never enter
(the guard is `dev.dev_setup`, inherited). The scored table is `exp_gen_cell` and only it.

THE HEADLINE (R2): distinct valid novel tuples, counted by the INHERITED instrument
`valid_novel.valid_novel_report` -- not re-defined here.
    VALID(i) := the external classifier reads render i as the tuple that was REQUESTED,
                on all six factors (== gen_probe exact_match per item).
    NOVEL(t) := t is not the factor tuple of any row the store was fitted on.
On a Montero-SHAPED corner held out of the pool, all 72 targets are distinct and none is a
pool tuple, so distinct_valid_novel = distinct_valid = 72 * exact_match. That identity is
ASSERTED per corner by `headline()`, never assumed.

THE FAST HEAD SEARCH. `fit_factored_head` (pinned) solves
(Phi^T Phi + ridge I) W = Phi^T Z and projects W onto each well's mode span. For a search
over PAIR SUBSETS those two Gram statistics are SUBMATRICES of the all-pairs ones, so they
are computed ONCE per corner and every subset is a small solve.
The fast path is used to SEARCH. Anything REPORTED is re-fitted through the pinned
`chlu.core.gen_factored_read.fit_factored_head`, and the two are asserted to agree.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np

import ctx  # noqa: F401  (inherited)
import dev  # noqa: F401  (inherited)
from valid_novel import valid_novel_report

OUT = Path(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1))

# ---------------------------------------------------------------------------
# The corner universe. Identical construction to the inherited DEV-24 list, so the 49
# admitted corners are the SAME 49 objects and the inherited numbers stay comparable.
# ---------------------------------------------------------------------------
CORNERS_RAW = [
    dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=False, shape_value=3.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=False, shape_value=1.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=True, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.2, scale_hi=True, shape_value=1.0, orient_hi=True),
    dict(hue_lo=0.4, scale_hi=True, shape_value=3.0, orient_hi=False),
    dict(hue_lo=0.6, scale_hi=False, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.2, scale_hi=False, shape_value=0.0, orient_hi=False),
    dict(hue_lo=0.4, scale_hi=False, shape_value=2.0, orient_hi=True),
    dict(hue_lo=0.6, scale_hi=True, shape_value=3.0, orient_hi=True),
] + [dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
     for h in (0.0, 0.2, 0.4, 0.6) for sh in (0.0, 1.0, 2.0, 3.0)
     for sc in (True, False) for o in (True, False)]


def corner_list():
    seen, out = set(), []
    for cd in CORNERS_RAW:
        k = tuple(sorted(cd.items()))
        if k not in seen:
            seen.add(k)
            out.append(cd)
    return out


CORNERS = corner_list()

# DECLARED AT BOOT, BEFORE ANY ITERATION-2 MEASUREMENT EXISTED.
# Even candidate index = SELECTION (mechanism may be chosen on it).
# Odd candidate index  = VALIDATION (looked at only to CONFIRM a choice already made).
# The split is on the CANDIDATE index, so a corner refused by the held-out guard cannot
# move another corner between the halves.
SELECTION_IDX = [i for i in range(len(CORNERS)) if i % 2 == 0]
VALIDATION_IDX = [i for i in range(len(CORNERS)) if i % 2 == 1]


# ---------------------------------------------------------------------------
# The headline
# ---------------------------------------------------------------------------
def headline(c, feats, lab_dev, pool_tuples, *, batch=256) -> dict:
    """One render -> the R2 headline + the aggregate + the whole ladder beneath both."""
    from chlu.eval.gen_probe import predict_factors
    from chlu.experiments.exp_gen_cell import render, score_images

    imgs = render(c["decoder"], np.asarray(feats, np.float32), c["feat_shape"], int(batch))
    preds = predict_factors(c["external"], imgs, int(batch))
    sc = score_images(c["external"], imgs, lab_dev, c["names"], int(batch))
    vn = valid_novel_report(preds, lab_dev, pool_tuples, c["names"])
    n = int(vn["n"])
    # THE IDENTITY, ASSERTED -- never assumed.
    if vn["n_targets_distinct"] != n or vn["n_targets_that_are_novel"] != n:
        raise AssertionError(
            f"this corner is not a clean Montero-shaped cell: {vn['n_targets_distinct']} "
            f"distinct targets and {vn['n_targets_that_are_novel']} novel of {n} -- the "
            "identity distinct_valid_novel = n * exact_match does NOT hold here")
    if vn["distinct_valid_novel"] != vn["n_valid_items"]:
        raise AssertionError(
            f"distinct_valid_novel {vn['distinct_valid_novel']} != n_valid_items "
            f"{vn['n_valid_items']} -- the identity broke; report it, do not paper over it")
    return {
        "tuples_valid_novel": int(vn["distinct_valid_novel"]),   # R2's column
        "aggregate": float(sc["aggregate"]),                     # generation-table's column
        "distinct": int(vn["distinct"]),
        "exact_match": float(sc["exact_match"]),
        "distinct_valid": int(vn["distinct_valid"]),
        "distinct_novel": int(vn["distinct_novel"]),
        "per_factor": {k: float(v) for k, v in sc["per_factor"].items()},
        "per_factor_n_values": {k: int(v["n_values_realised_in_targets"])
                                for k, v in vn["per_factor"].items()},
        "n": n,
    }


def pool_tuple_array(grid, pool, factor_labels) -> np.ndarray:
    """The factor tuples the store was fitted on -- the NOVEL test's reference set."""
    return np.unique(np.asarray(factor_labels(grid, pool), np.int64), axis=0)


# ---------------------------------------------------------------------------
# Sufficient statistics for the head, over the FULL pair design
# ---------------------------------------------------------------------------
def all_pairs(F: int):
    return list(itertools.combinations(range(int(F)), 2))


def pair_slices(cards, pairs):
    out, off = {}, 0
    for (f, g) in pairs:
        w = int(cards[f]) * int(cards[g])
        out[(f, g)] = slice(off, off + w)
        off += w
    return out, off


def design_columns(labels, cards, pairs):
    """Active column index per row, per block. One-hot, so a column index is enough."""
    Y = np.asarray(labels, np.int64)
    n = Y.shape[0]
    offs = np.cumsum([0] + [int(x) for x in cards[:-1]]).astype(np.int64)
    cols = [np.zeros(n, np.int64)]
    for f in range(len(cards)):
        cols.append(1 + offs[f] + Y[:, f])
    n_add = 1 + int(sum(int(x) for x in cards))
    sl_map, n_pair = pair_slices(cards, pairs)
    for (f, g) in pairs:
        sl = sl_map[(f, g)]
        cols.append(n_add + sl.start + Y[:, f] * int(cards[g]) + Y[:, g])
    return np.stack(cols, axis=1), n_add, n_pair


def sufficient_stats(Z, labels, cards, pairs):
    """(G = Phi^T Phi, H = Phi^T Z) over the FULL design, float64.

    Phi is one-hot per block, so both are group sums -- no dense (n, P) is built.
    """
    import scipy.sparse as sp

    C, n_add, n_pair = design_columns(labels, cards, pairs)
    n, nb = C.shape
    P = int(n_add + n_pair)
    rows = np.repeat(np.arange(n, dtype=np.int64), nb)
    Phi = sp.csr_matrix((np.ones(n * nb), (rows, C.reshape(-1))), shape=(n, P))
    G = np.asarray((Phi.T @ Phi).todense(), np.float64)
    H = np.asarray(Phi.T @ np.asarray(Z, np.float64), np.float64)
    return G, H, n_add, n_pair


def subset_index(cards, pairs_all, pairs_use, n_add):
    """Column indices of [additive block] + [the chosen pair blocks], in `pairs_use` order."""
    sl_map, _ = pair_slices(cards, pairs_all)
    idx = [np.arange(n_add, dtype=np.int64)]
    for p in pairs_use:
        sl = sl_map[tuple(p)]
        idx.append(np.arange(n_add + sl.start, n_add + sl.stop, dtype=np.int64))
    return np.concatenate(idx)


def solve_W(G, H, idx, ridge: float):
    Gi = G[np.ix_(idx, idx)]
    return np.linalg.solve(Gi + float(ridge) * np.eye(Gi.shape[0]), H[idx])


def project_to_modes(W, perm, modes):
    """C[:, k, :] = least-squares coefficients of W[:, well k] on well k's modes.

    Line-for-line the same arithmetic as `fit_factored_head`'s projection loop.
    """
    K, M, _ = modes.shape
    C = np.zeros((W.shape[0], K, M), np.float64)
    for k in range(K):
        Bk = np.asarray(modes[k], np.float64)
        C[:, k, :] = np.linalg.solve(Bk @ Bk.T + 1e-8 * np.eye(M),
                                     Bk @ np.asarray(W[:, perm[k]], np.float64).T).T
    return C


def truncate(C, n_add, rank):
    """Rank-r truncation of the interaction block, exactly as `fit_factored_head` does."""
    K, M = int(C.shape[1]), int(C.shape[2])
    Cb = C[n_add:].reshape(-1, K * M)
    if Cb.shape[0] and int(rank) > 0:
        Uh, Sh, Vth = np.linalg.svd(Cb, full_matrices=False)
        r = int(min(int(rank), Sh.size))
        return C[:n_add], Uh[:, :r] * Sh[:r], Vth[:r]
    return C[:n_add], np.zeros((Cb.shape[0], 0)), np.zeros((0, K * M))


def coefficients(C_add, E, U, labels_req, cards, pairs, K, M):
    """(N, F) requests -> (N, K, M) weights, from the truncated blocks."""
    Cols, n_add, _ = design_columns(labels_req, cards, pairs)
    n = Cols.shape[0]
    a = np.zeros((n, K * M), np.float64)
    Cadd = np.asarray(C_add, np.float64).reshape(n_add, K * M)
    for j in range(1 + len(cards)):
        a += Cadd[Cols[:, j]]
    if np.asarray(E).size:
        B = np.zeros((n, int(np.asarray(E).shape[0])), np.float64)
        for j in range(1 + len(cards), Cols.shape[1]):
            B[np.arange(n), Cols[:, j] - n_add] = 1.0
        a += (B @ np.asarray(E, np.float64)) @ np.asarray(U, np.float64)
    return a.reshape(n, K, M).astype(np.float32)


# ---------------------------------------------------------------------------
# Byte ledgers
# ---------------------------------------------------------------------------
def head_bytes(cards, pairs, K, M, rank) -> int:
    n_add = 1 + int(sum(int(x) for x in cards))
    _, n_pair = pair_slices(cards, pairs)
    r = int(rank) if n_pair else 0
    return int((n_add * int(K) * int(M) + n_pair * r + r * int(K) * int(M)) * 4)


def store_bytes_of(M, D) -> int:
    return int(int(M) * int(D) * 4)


def total_bytes(cards, pairs, K, M, rank, D) -> int:
    return store_bytes_of(M, D) + head_bytes(cards, pairs, K, M, rank)


def knn_proto_bytes(cards, D) -> int:
    """knn_proto's deployed ledger: (sum_f V_f + 1) * D * 4."""
    return int((int(sum(int(x) for x in cards)) + 1) * int(D) * 4)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------
def paired(x, y):
    """Paired mean difference, SE, 95% CI (t), wins. n = len(x)."""
    from scipy import stats

    d = np.asarray(x, float) - np.asarray(y, float)
    n = d.size
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return {"n": int(n), "delta": m, "se": se,
            "ci95": [m - t * se, m + t * se],
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "t": (m / se if se else float("nan"))}


def r1_verdict(p, m=0.020, scale=1.0):
    """R1 applied. `scale` converts the column into the units m is stated in.

    m = 0.020 is stated on the composition-B aggregate (a 0-1 metric). On the tuple
    column, one aggregate unit is `scale` tuples, so the same margin is m * scale.
    """
    mm = float(m) * float(scale)
    lo, hi = p["ci95"]
    if lo <= 0.0 <= hi:
        return "tie"
    if abs(p["delta"]) <= mm:
        return "tie"
    return "win" if p["delta"] > 0 else "loss"
