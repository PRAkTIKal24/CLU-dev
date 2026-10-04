"""hub/generation-table-2 — the iteration-2 arm, its variants and its controls.

⛔ DEV ONLY.  Nothing here ever touches ``eval_a`` / ``eval_b``.

One function, :func:`run_variant`, builds a store + head at a named configuration and
returns its decoded score.  Every lever is a named flag so a variant is a data row, not
a code path, and the byte ledger is recomputed from the LIVE objects on every call.
"""
from __future__ import annotations

import numpy as np

import ctx  # noqa: F401
import dev
import fast

from chlu.core.gen_compose import clu_wide_vector
from chlu.core.gen_dict_fit import (allocate_atoms, build_store_from_dicts,
                                    fit_factored_head_v2, head_bytes, well_dictionary)
from chlu.core.gen_store import store_bytes

#: ⭐ knn_proto's deployed prototype table — the byte budget every arm matches.
KNN_BYTES = 475_136


def budget_rank(K: int, A: int, n_add: int, n_pair: int, d_per_well: int,
                budget: int = KNN_BYTES) -> int:
    """The largest interaction rank that still fits under ``budget``.  ⭐ Arithmetic, not a choice."""
    store = int(d_per_well) * int(A) * 4
    add = int(n_add) * int(A) * 4
    left = int(budget) - store - add
    per_r = (int(n_pair) + int(A)) * 4
    return max(int(left // per_r), 0)


def well_views(Z: np.ndarray, perm: np.ndarray) -> list:
    return [np.asarray(Z[:, perm[k]], np.float64) for k in range(perm.shape[0])]


def build_dicts(views: list, Ms: np.ndarray, method: str, seed: int) -> list:
    return [well_dictionary(views[k], int(Ms[k]), method, int(seed) * 1000 + k)
            for k in range(len(views))]


def run_variant(c, s, Zs, lab_store, perm, req, lab_dev, *, method: str, K: int, A: int,
                rank: int, pairs, ridge_add: float, ridge_pair: float, reduced: bool,
                metric_correct: bool, alloc: str, seed: int, n_add: int, n_pair: int,
                views=None, dicts=None, want_store=False):
    """One (dictionary, allocation, head) configuration -> its decoded, classified score."""
    if views is None:
        views = well_views(Zs, perm)
    Ms = allocate_atoms(views, int(A), method=alloc) if alloc != "uniform" else \
        allocate_atoms(views, int(A), method="uniform")
    if dicts is None:
        dicts = build_dicts(views, Ms, method, seed)
    st = build_store_from_dicts([tuple(int(i) for i in perm[k]) for k in range(int(K))],
                                dicts, (Zs.shape[1],))
    head = fit_factored_head_v2(dicts, perm, Zs, lab_store, c["cards"], pairs, int(rank),
                                ridge_add=float(ridge_add), ridge_pair=float(ridge_pair),
                                reduced=bool(reduced), metric_correct=bool(metric_correct))
    feats = np.asarray(clu_wide_vector(st, head.coefficients(req)), np.float32)
    out = dev.score_features(c, feats, lab_dev)
    sb = store_bytes(st)
    hb = head.byte_ledger()
    out["store_bytes"] = int(sb["store_bytes"])
    out["head_bytes"] = int(hb["bytes"])
    out["total_bytes"] = int(sb["store_bytes"] + hb["bytes"])
    out["modes_per_well"] = [int(x) for x in Ms]
    out["A"] = int(sum(int(x) for x in Ms))
    out["rank_realised"] = int(head.fit_info["rank_realised"])
    out["n_add"] = int(n_add)
    out["n_pair"] = int(n_pair)
    if want_store:
        return out, st, head, dicts
    return out
