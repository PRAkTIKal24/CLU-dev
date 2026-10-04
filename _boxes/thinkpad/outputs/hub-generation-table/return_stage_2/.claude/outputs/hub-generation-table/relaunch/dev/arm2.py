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

# RELAUNCH FIX: `head.coefficients` returns a LIST of (N, M_k); the pinned wide read takes a
# rectangular (N, K, M). `gen_wide_read.wide_vector` is a pass-through to the SHIPPED function
# when the allocation is uniform (tested bit-identical, tests/test_gen_wide_read.py) and only
# writes the scatter out for a ragged one (DEV only, declared on every row).
from chlu.core.gen_wide_read import wide_vector
from chlu.core.gen_dict_fit import (allocate_atoms, build_store_from_dicts,
                                    fit_factored_head_v2, head_bytes, well_dictionary)
from chlu.core.gen_shared_basis_head import (fit_shared_basis_head,
                                             shared_basis_head_bytes)
from chlu.core.gen_store import store_bytes

#: ⭐ knn_proto's deployed prototype table — the byte budget every arm matches.
KNN_BYTES = 475_136


def budget_atoms_shared(K: int, n_add: int, n_pair: int, d_per_well: int,
                        budget: int = 475_136) -> int:
    """Largest A the SHARED-BASIS head affords.  The interaction costs n_pair*n_add*4
    whatever A is, so everything else goes into atoms.  Arithmetic, not a choice."""
    left = int(budget) - int(n_pair) * int(n_add) * 4
    per_atom = (int(d_per_well) + int(n_add)) * 4
    return max(int(left // per_atom), 0)


def budget_rank(K: int, A: int, n_add: int, n_pair: int, d_per_well: int,
                budget: int = KNN_BYTES) -> int:
    """The largest interaction rank that still fits under ``budget``.  ⭐ Arithmetic, not a choice."""
    store = int(d_per_well) * int(A) * 4
    add = int(n_add) * int(A) * 4
    left = int(budget) - store - add
    per_r = (int(n_pair) + int(A)) * 4
    return max(int(left // per_r), 0)


def well_views(Z: np.ndarray, perm) -> list:
    """``perm`` may be a (K, d) array OR a ragged list of index arrays (the SHIPPED
    correlation grouping gives unequal wells)."""
    K = perm.shape[0] if hasattr(perm, "shape") else len(perm)
    return [np.asarray(Z[:, np.asarray(perm[k], np.int64)], np.float64) for k in range(K)]


def build_dicts(views: list, Ms: np.ndarray, method: str, seed: int) -> list:
    return [well_dictionary(views[k], int(Ms[k]), method, int(seed) * 1000 + k)
            for k in range(len(views))]


def run_dense_reference(c, Zs, lab_store, req, lab_dev, *, pairs, n_add, n_pair,
                        ridge_add, ridge_pair, reduced=True, interaction=True):
    """THE REFERENCE ROW THAT IS NOT AN ARM: no store at all.

    The deployed additive map of any factored read is ``Phi_add . C_add . B``, a
    ``(n_add, D)`` matrix.  This row stores that matrix DIRECTLY -- 52 dense
    D-dimensional prototypes -- plus the same shared-basis pair remap, and pays

        (n_add*D + n_pair*n_add) * 4  =  (52*2048 + 220*52)*4  =  471,744 B  <=  475,136

    so it fits UNDER knn_proto's own budget with no factorisation loss whatever.

    It answers one question the generation table has never had an answer to:
    IS knn_proto BEATABLE AT MATCHED BYTES AT ALL, or is it the ceiling?

    It is NOT an arm and is never reported as one: there is no store, no partition, no
    landscape, nothing to delete atom-wise -- it is precisely the "glorified lookup
    table" the programme's goal statement rules out as a deliverable.  It is a BOUND.
    """
    from chlu.core.gen_dict_fit import _pair_blocks, reduced_additive_design

    Z = np.asarray(Zs, np.float64)
    Ad = reduced_additive_design(lab_store, c["cards"], reduced)
    Bd = _pair_blocks(lab_store, c["cards"], pairs) if interaction else         np.zeros((Z.shape[0], 0))
    Phi = np.concatenate([Ad, Bd], 1) if Bd.shape[1] else Ad
    lam = np.concatenate([np.full(Ad.shape[1], float(ridge_add)),
                          np.full(Bd.shape[1], float(ridge_pair))])
    W = np.linalg.solve(Phi.T @ Phi + np.diag(lam), Phi.T @ Z)        # (n_cols, D)
    W_add = W[: Ad.shape[1]]
    Ar = reduced_additive_design(req, c["cards"], reduced)
    if Bd.shape[1]:
        E, *_ = np.linalg.lstsq(W_add.T, W[Ad.shape[1]:].T, rcond=None)
        Br = _pair_blocks(req, c["cards"], pairs)
        Ar = Ar + Br @ np.asarray(E.T, np.float64)
    feats = np.asarray((Ar @ W_add).reshape((-1,) + tuple(c["feat_shape"])), np.float32)
    out = dev.score_features(c, feats, lab_dev)
    nb = int(W_add.size + (Bd.shape[1] * Ad.shape[1] if Bd.shape[1] else 0))
    out["store_bytes"] = 0
    out["head_bytes"] = int(nb * 4)
    out["total_bytes"] = int(nb * 4)
    out["A"] = 0
    out["rank_realised"] = int(Ad.shape[1]) if Bd.shape[1] else 0
    out["n_add"] = int(n_add)
    out["n_pair"] = int(n_pair) if interaction else 0
    out["head_kind"] = "dense_reference_NOT_AN_ARM"
    out["modes_per_well"] = []
    out["uniform_allocation"] = True
    out["read_verb"] = "NO STORE - a dense prototype table. A BOUND, never an arm."
    return out


def run_variant(c, s, Zs, lab_store, perm, req, lab_dev, *, method: str, K: int, A: int,
                rank: int, pairs, ridge_add: float, ridge_pair: float, reduced: bool,
                metric_correct: bool, alloc: str, seed: int, n_add: int, n_pair: int,
                views=None, dicts=None, want_store=False, head_kind: str = "factored"):
    """One (dictionary, allocation, head) configuration -> its decoded, classified score."""
    if views is None:
        views = well_views(Zs, perm)
    Ms = allocate_atoms(views, int(A), method=alloc) if alloc != "uniform" else \
        allocate_atoms(views, int(A), method="uniform")
    if dicts is None:
        dicts = build_dicts(views, Ms, method, seed)
    st = build_store_from_dicts(
        [tuple(int(i) for i in np.asarray(perm[k])) for k in range(int(K))],
        dicts, (Zs.shape[1],))
    if str(head_kind) == "shared":
        head = fit_shared_basis_head(dicts, perm, Zs, lab_store, c["cards"], pairs,
                                     ridge_add=float(ridge_add),
                                     ridge_pair=float(ridge_pair), reduced=bool(reduced))
    else:
        head = fit_factored_head_v2(dicts, perm, Zs, lab_store, c["cards"], pairs,
                                    int(rank), ridge_add=float(ridge_add),
                                    ridge_pair=float(ridge_pair), reduced=bool(reduced),
                                    metric_correct=bool(metric_correct))
    feats = np.asarray(wide_vector(st, head.coefficients(req)), np.float32)
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
    out["head_kind"] = str(head_kind)
    out["uniform_allocation"] = bool(sb["uniform_allocation"])
    out["read_verb"] = ("shipped gen_compose.clu_wide_vector"
                        if sb["uniform_allocation"] else
                        "ragged equivalent (DEV only) - NOT the shipped call")
    if want_store:
        return out, st, head, dicts
    return out
