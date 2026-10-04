"""⭐⭐ THE SINGLE-DESIGN STORE: the object that is meant to COMPOSE *and* to FORGET.

What this module is, in one sentence
------------------------------------
It is ``pca_tensor``'s content -- per-well leading principal subspaces, unsupervised, at
the pinned 524,288 B -- refitted as a function of **sufficient statistics** instead of a
one-shot SVD, so that ``delete(rows)`` is a covariance downdate plus a
re-eigendecomposition plus a count downdate, and the result **equals the store fitted
without those rows**.

Why the object exists
---------------------
⭐ Composition is won by ``pca_tensor`` (:class:`chlu.experiments.exp_w5_compose_field.
PCASpanStore`): per-well leading principal subspaces, label-free, 524,288 B, beating
``mlp_180k`` on rows 3 and 4 on all three seeds.

⛔ But **a principal direction is not a written item.**  Charter B.4's "deletion is an
exact operation on atoms" does not hold of it: there is no atom to evict, because every
atom is a function of every row.  ``build_pca_content``'s own docstring declares this as
a cost of the content.

⛔ Meanwhile the object that *does* forget exactly -- the field store's
``coupling_hole``, one byte, nothing outside it, no refit -- **loses composition when it
evicts atoms**, because its content is whole written items and an evicted item is
content that is simply gone.

⭐⭐ This design is the attempt to have both, and the whole attempt rests on one
observation: the leading principal subspace of a set of rows is a function of the set's
**second moment**, and a second moment is a *sum over rows*.  A sum can be un-summed.
So the store keeps no row and still forgets a row exactly::

    S_k = sum_i x_ik x_ik^T      t_k = sum_i x_ik      n = |rows|

    delete(rows_del):   S_k -= sum_{i in del} x_ik x_ik^T
                        t_k -= sum_{i in del} x_ik
                        n   -= |rows_del|
                        modes_k = eig(S_k - t_k t_k^T / n)

⛔ **The covariance is not free.**  It costs ``d_k^2`` per well and it is declared in the
ledger BESIDE the store, never netted into the 524,288 B and never netted against the
address book.  :meth:`DowndatableStore.covariance_ledger` prints it separately and
:meth:`DowndatableStore.honest_ledger` prints the sum, over budget and labelled as such.

Four parts, one object
----------------------
1. **the principal subspaces** -- ``A x d_k`` float32 per well, ``A * D * 4`` total, the
   primary pin at tolerance 0.  This is the content the decoder renders.
2. **the running covariance** -- ``S_k`` (``d_k^2`` float64) and ``t_k`` (``d_k``), the
   sufficient statistics that make the downdate exact.  ⛔ Declared beside, never inside.
3. **the request count table** -- ``term_n`` per ``(factor, value)`` and ``pair_n`` per
   term pair: what the store has been asked for and what it holds evidence of.
4. **the coupling table with holes** -- ``coupling`` (the smoothed PMI, recomputed from
   the counts, so it downdates exactly with them) and ``coupling_hole``, w5's mechanism:
   ONE byte per pair, in a **different array** from the value, so "unseen" and "deleted"
   can never be confused (``gen_field.punch_hole``, w5-delete's result: 1 byte changed,
   0 bytes outside, 0.0000 composition cost).

⚠ TWO PROPERTIES OF THE DEFINITION, declared here because they are not tolerances that
buy a pass -- they are what makes "the store fitted without those rows" a well-posed
phrase at all.
------------------------------------------------------------------------------------
⭐ **The rank cut.**  A direction whose eigenvalue is numerically zero is not a principal
direction: it carries no variance, and in a null (or degenerate) eigenspace the
eigenvector is **not a function of the data** -- LAPACK returns an arbitrary rotation, and
two refits of the very same rows on two builds would disagree.  MEASURED on the pinned
partition: wells 3 and 4 hold 48 and 19 **dead channels** out of 60, so their covariance
has effective rank 16 and 41, and asking for ``min(A-1, d_k) = 60`` directions makes the
store write 44 and 19 arbitrary ones.  Before the cut the downdate/refit deviation in
those wells was **1.0**; after it, 5.06e-12 and 1.14e-13.  So the store keeps
``min(A-1, d_k, #{j : lambda_j > eig_rel_tol * lambda_max})`` directions and writes the
rest as **zero rows** -- which is exactly the convention ``PCASpanStore`` already states
for a well narrower than ``A``: "they cost the same bytes and buy no span".
⛔ The incumbent has the identical non-uniqueness and simply never had to be a function
of its fit set twice.  ⭐ The cut is reported with its **margin** on both sides
(:meth:`rank_report`) so a reader can see whether it was marginal; on the pinned
partition the smallest kept eigenvalue is >= 2.3e3 x the threshold and the largest
dropped one <= 3.6e-4 x it.

⭐ **The sign convention.**  An eigenvector's sign is free, so each direction is signed to
make its largest-magnitude component positive (lowest index breaks a tie).  Without it a
downdate and a refit could differ by exactly ``2 * |v|`` on a direction they agree about.

⛔ WHAT THIS OBJECT DOES NOT CLAIM
----------------------------------
``delete(X_del, ...)`` is **handed** the rows to forget; it does not look them up.  The
store never held them and does not hold them after.  ⛔ That is an assumption about the
*deleter*, and it is declared, not hidden: the party asking for a deletion presents what
it wants deleted.  What is asserted here is the thing that is actually in question --
that the delete touches **only** those rows and never the retained corpus
(:data:`ROWS_CONSUMED` and :func:`no_data_pass`), and that no row survives anywhere in
the object (:func:`retention_audit`).
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

#: ⭐ The rank cut, declared.  See the module docstring: this is a well-posedness pin,
#: not a tolerance chosen to make a comparison pass, and :meth:`DowndatableStore.
#: rank_report` prints the margin on both sides of it so that claim is checkable.
EIG_REL_TOL = 1e-12

#: ⭐ The mean's tolerance in :func:`pinned_span_basis`, declared for the same
#: well-posedness reason as :data:`EIG_REL_TOL` and with its margin reported beside it.
#: MEASURED on the pinned partition: the mean's component orthogonal to the directions
#: has relative norm 0.69-0.80 in wells 0-2 and 2.3e-8 / 4.9e-8 in wells 3-4 -- in the
#: rank-deficient wells the mean lies INSIDE the directions' span and what is left over
#: is numerical noise, whose direction is exactly the kind of arbitrary null direction
#: the rank cut exists to keep out of the basis. 1e-4 sits four orders clear of both.
MU_PERP_REL_TOL = 1e-4

#: ⛔⭐ THE VARIANCE-SHARE CUT, declared 2026-09-20T18:45Z BEFORE the re-run that
#: tests it, at the hub's direction after the subspace call was withdrawn.
#:
#:     tau = delta_rel_worst / ulp_float32 = 4.941056e-16 / 1.1920929e-07
#:         = 4.144858e-09
#:
#: ``delta_rel_worst`` is MEASURED, not assumed: the largest relative perturbation of
#: the centred scatter ``C`` under a change that must not matter -- accumulating the
#: same rows in a different order -- over all wells of all four legs (val seed 0, test
#: seeds 0/1/2), with no deletion and no refit. It is the numerical noise floor of the
#: accumulate step and it is what the reproducibility floor is made of.
#: ``ulp_float32`` is the finest angular distinction a float32 store can represent and
#: is therefore the target precision.
#:
#: ⭐ Their quotient is the variance share below which a direction's eigenvalue sits
#: at or under the level the accumulation can resolve to that target, so its
#: orientation is not a function of the data and it is not content. There is NO FREE
#: PARAMETER: one measured quantity, one target fixed by the storage format.
#:
#: ⛔ NOT ROUNDED, and the reason runs opposite to intuition: rounding 4.1449e-09 up
#: to a clean 1e-8 makes the cut STRICTER, which drops more marginal directions and is
#: therefore MORE likely to make the failing seed pass. Rounding up is conservative
#: about content and anti-conservative about the verdict. The exact measured quotient
#: leaves no room to have chosen.
#:
#: ⛔ ``EIG_REL_TOL`` is NOT retuned -- it stays at 1e-12 and both cuts apply;
#: whichever binds, binds.
VARIANCE_SHARE_TOL = 4.144858e-09

#: ⭐ The count table's smoothing, declared.  The coupling value is a function of the
#: counts alone, which is what makes the coupling table downdate exactly with them.
COUPLING_ALPHA = 1.0

#: ⛔⭐ THE TRIPWIRE.  Every feature row that this module consumes, ever, is counted
#: here -- and :func:`_row_moments` is the ONLY place in the module that touches one.
#: "No pass over the data" is then not prose: it is
#: ``ROWS_CONSUMED["n"]`` before and after a :meth:`DowndatableStore.delete`, which must
#: differ by exactly the number of rows the caller asked to delete and by nothing else,
#: at every corpus size.
ROWS_CONSUMED = {"n": 0, "calls": 0}


def reset_rows_consumed() -> None:
    ROWS_CONSUMED["n"] = 0
    ROWS_CONSUMED["calls"] = 0


# ---------------------------------------------------------------------------
# 1. The only function that consumes a feature row
# ---------------------------------------------------------------------------


def _row_moments(X: np.ndarray, channels: np.ndarray) -> tuple:
    """``(rows, channels) -> (S, t, n)``: the well's second moment, first moment, count.

    ⛔ THE ONLY function in this module that reads a feature row, which is what makes
    :data:`ROWS_CONSUMED` a complete account rather than a sample of one.

    Accumulated in **float64**.  MEASURED, and it is why float32 is not an option: the
    residual ``|(S_all - S_del) - S_keep|`` runs to 5e-10 absolute on the pinned
    partition, which a float32 accumulator (eps ~ 1e-7 relative on entries of order
    1e+9) would bury entirely.  ⭐ Note the products themselves are exact: a float32
    times a float32 has at most 48 significand bits and lands in a float64 without
    rounding, so the only error in the whole downdate is the *summation* order.
    """
    ch = np.asarray(channels, np.int64)
    M = np.asarray(np.asarray(X)[:, ch], np.float64)
    ROWS_CONSUMED["n"] += int(M.shape[0])
    ROWS_CONSUMED["calls"] += 1
    return M.T @ M, M.sum(axis=0), int(M.shape[0])


# ---------------------------------------------------------------------------
# 2. The store's content as a PURE FUNCTION of the sufficient statistics
# ---------------------------------------------------------------------------


def _canonical_sign(V: np.ndarray) -> np.ndarray:
    """Sign each row so its largest-magnitude entry is positive; ties by lowest index."""
    if V.size == 0:
        return V
    idx = np.argmax(np.abs(V), axis=1)
    s = np.sign(V[np.arange(V.shape[0]), idx])
    s[s == 0] = 1.0
    return V * s[:, None]


def modes_from_moments(S: np.ndarray, t: np.ndarray, n: int, n_atoms: int,
                       eig_rel_tol: float = EIG_REL_TOL,
                       var_share_tol: float = None) -> tuple:
    """``(S, t, n) -> (A x d_k float32 atom block, report)``.  ⭐ A PURE FUNCTION.

    ⛔ It takes no row, no row index and no corpus.  Its whole input is the three
    sufficient statistics, which is the entire reason the downdate can be exact: the
    downdated statistics and the without-rows statistics go through **this same
    function**, so any difference between the two stores is a difference in the
    statistics and nothing else.

    The block is ``[mu ; V_1 .. V_r ; 0 .. 0]`` exactly as ``PCASpanStore`` writes it --
    the pool mean as the first atom (so the subspace is affine-capable, as a store that
    writes items is: an item may be the mean item), then the kept principal directions,
    then zero rows for the atoms the well is too narrow or too rank-deficient to use.
    """
    d = int(np.asarray(t).size)
    A = int(n_atoms)
    if int(n) <= 0:
        raise ValueError("a well with no rows has no mean and no principal direction")
    Sm = np.asarray(S, np.float64)
    tv = np.asarray(t, np.float64)
    mu = tv / float(n)
    C = Sm - np.outer(tv, tv) / float(n)
    C = 0.5 * (C + C.T)                       # exactly symmetric on BOTH paths
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order].T                          # rows are directions
    lmax = float(max(float(w[0]), 0.0))
    n_above = int((w > float(eig_rel_tol) * lmax).sum())
    # ⛔⭐ THE DECLARED VARIANCE-SHARE CUT. See VARIANCE_SHARE_TOL: a direction whose
    # share of its well's variance is below tau has an eigenvalue at or under what the
    # accumulate step resolves to one float32 ulp, so it is not content. Applied
    # ALONGSIDE eig_rel_tol, which is unchanged; whichever binds, binds.
    vst = VARIANCE_SHARE_TOL if var_share_tol is None else float(var_share_tol)
    tot_pos = float(np.maximum(w, 0.0).sum())
    share = (np.maximum(w, 0.0) / tot_pos) if tot_pos > 0 else np.zeros_like(w)
    n_share = int((share >= vst).sum())
    keep_no_share = int(min(A - 1, d, n_above))
    keep = int(min(A - 1, d, n_above, n_share))
    B = np.vstack([mu[None, :], _canonical_sign(V[:keep])]).astype(np.float32)
    if B.shape[0] < A:
        B = np.vstack([B, np.zeros((A - B.shape[0], d), np.float32)])
    tot = float(np.maximum(w, 0.0).sum())
    thr = float(eig_rel_tol) * lmax
    # ⛔⭐ THE PREDICTOR, computed at FIT TIME from the statistics alone -- no
    # deletion, no second fit. Classical perturbation theory: an eigenvector rotates by
    # ~ ||dC|| / gap, and the downdate's residual on C is at machine epsilon. So the
    # precision to which this well's atom BYTES are determined is eps / gap, and the
    # object can state it in advance instead of discovering it in a deletion.
    #
    # ⚠ THIS IS THE QUANTITY I PINNED WRONG AT 16:52Z. I pinned the eigenvalue
    # MAGNITUDE and argued well-posedness from its margins. What actually determines
    # whether an eigenvector is a function of the data is its GAP to its neighbours: a
    # direction far above the magnitude threshold can be nearly degenerate with the one
    # beside it, and is then no better determined than a null direction. MEASURED on
    # the test leg: well 3's smallest kept gap is 1.36e-03 at seed 0 and 4.95e-11 at
    # seed 2, and seed 2's downdate deviates by 3.92e-06 -- 33 float32 ulp -- against a
    # predicted 4.45e-06. The val leg's ~1e-3 gap is seven orders safer, which is
    # exactly how one leg passing hid it.
    eps64 = float(np.finfo(np.float64).eps)
    gap_kept = float("inf")
    gap_any = float("inf")
    if keep > 1:
        gap_kept = float(np.min(np.abs(np.diff(w[:keep]))))
    if keep > 0 and w.size > 1:
        for j in range(keep):
            nb = []
            if j > 0:
                nb.append(abs(w[j] - w[j - 1]))
            if j + 1 < w.size:
                nb.append(abs(w[j] - w[j + 1]))
            if nb:
                gap_any = min(gap_any, float(min(nb)))
    # relative to lambda_max, which is the scale the perturbation is measured against
    rel_gap_kept = gap_kept / lmax if lmax > 0 else float("inf")
    rel_gap_any = gap_any / lmax if lmax > 0 else float("inf")
    sens_kept = eps64 / rel_gap_kept if rel_gap_kept > 0 else float("inf")
    sens_any = eps64 / rel_gap_any if rel_gap_any > 0 else float("inf")
    f32_ulp = float(np.spacing(np.float32(1.0)))
    report = {
        "d_k": d, "n_rows": int(n), "rank_kept": keep,
        "rank_limited_by": ("atom budget A-1" if keep == A - 1 else
                            ("well width d_k" if keep == d else "eigenvalue tolerance")),
        "n_zero_rows": int(A - 1 - keep),
        "eig_threshold": thr,
        # ⭐ the margin on BOTH sides of the cut: the claim "the cut is not marginal"
        # is only worth anything if the number that would falsify it is printed too.
        "smallest_kept_over_threshold": (float(w[keep - 1] / thr)
                                         if keep > 0 and thr > 0 else float("inf")),
        "largest_dropped_over_threshold": (float(w[keep] / thr)
                                           if keep < w.size and thr > 0 else 0.0),
        "lambda_max": lmax,
        "lambda_smallest_kept": float(w[keep - 1]) if keep > 0 else float("nan"),
        "lambda_largest_dropped": float(w[keep]) if keep < w.size else float("nan"),
        "variance_explained": (float(np.maximum(w[:keep], 0.0).sum() / tot)
                               if tot > 0 else 1.0),
        # ⚠ hub/single-design's saturation trap: when A-1 >= d_k the atom block spans
        # the WHOLE well, the projection is the identity and the CONTENT carries no
        # information -- the address book does all the work.  Carried per well so a
        # saturated build is visible instead of inferred.
        "rank_over_d": float(keep) / float(d),
        "saturated": bool(A - 1 >= d),
        # -- the declared variance-share cut and what it costs -----------------
        "var_share_tol": vst,
        "rank_kept_without_share_cut": keep_no_share,
        "n_directions_dropped_by_share_cut": int(max(keep_no_share - keep, 0)),
        "share_cut_binds": bool(keep < keep_no_share),
        "variance_share_lost_to_share_cut": float(
            share[keep:keep_no_share].sum()) if keep < keep_no_share else 0.0,
        "smallest_kept_variance_share": float(share[keep - 1]) if keep > 0 else None,
        "largest_dropped_variance_share": (float(share[keep])
                                           if keep < share.size else 0.0),
        # -- the fit-time well-posedness declaration -------------------------
        "min_rel_gap_among_kept": rel_gap_kept,
        "min_rel_gap_to_any_neighbour": rel_gap_any,
        "predicted_byte_sensitivity": sens_any,
        "predicted_byte_sensitivity_kept_only": sens_kept,
        "predicted_sensitivity_in_float32_ulp": sens_any / f32_ulp,
        # ⭐ the claim the object is entitled to make about ITSELF, before any
        # deletion: are this well's atom bytes determined more finely than a float32
        # store can represent a unit direction?
        "bytes_determined_below_one_float32_ulp": bool(sens_any < f32_ulp),
        "sensitivity_rule": ("eps64 / (min relative eigenvalue gap); an eigenvector "
                             "rotates by ~||dC||/gap and the downdate residual on C "
                             "is at machine epsilon"),
    }
    return np.ascontiguousarray(B), report


# ---------------------------------------------------------------------------
# 3. The count table and the coupling table
# ---------------------------------------------------------------------------


def term_offsets(cardinalities: Sequence[int]) -> np.ndarray:
    """Start of each factor's value block in the flat term index (gen_field's layout)."""
    c = [int(x) for x in cardinalities]
    return np.concatenate([[0], np.cumsum(c)]).astype(np.int64)


def coupling_layout(cardinalities: Sequence[int]) -> dict:
    """⭐ ``gen_field.coupling_layout`` verbatim -- the SAME index space as w5's hole.

    Re-derived here rather than imported so this object carries no dependency on a
    ``FieldStore`` byte image it does not deploy; the layout is asserted equal to
    ``gen_field``'s in :func:`assert_layout_matches_field_store`.
    """
    cards = [int(x) for x in cardinalities]
    pairs, offs, off = [], [], 0
    for f in range(len(cards)):
        for g in range(f + 1, len(cards)):
            pairs.append((f, g))
            offs.append(off)
            off += cards[f] * cards[g]
    return {"pairs": pairs, "offsets": offs, "size": int(off), "cardinalities": cards}


def coupling_index(cardinalities: Sequence[int], f: int, v: int, g: int, u: int) -> int:
    """Flat index of ``(f, v) x (g, u)``; argument order-free, as gen_field's is."""
    if int(f) == int(g):
        raise ValueError("a factor does not couple to itself")
    if int(f) > int(g):
        f, v, g, u = g, u, f, v
    lay = coupling_layout(cardinalities)
    j = lay["pairs"].index((int(f), int(g)))
    return int(lay["offsets"][j] + int(v) * lay["cardinalities"][int(g)] + int(u))


def assert_layout_matches_field_store(cardinalities: Sequence[int]) -> dict:
    """⛔ The hole's index space here IS the field store's, asserted not assumed."""
    from .gen_field import coupling_index as gf_index
    from .gen_field import coupling_layout as gf_layout

    mine, theirs = coupling_layout(cardinalities), gf_layout(cardinalities)
    if (mine["pairs"] != theirs["pairs"] or mine["offsets"] != theirs["offsets"]
            or mine["size"] != theirs["size"]):
        raise AssertionError(
            "the coupling layout here is not gen_field's -- a hole punched in this "
            "object would not be the hole w5 measured")
    cards = [int(x) for x in cardinalities]
    n_checked = 0
    for f in range(len(cards)):
        for g in range(f + 1, len(cards)):
            for v in (0, cards[f] - 1):
                for u in (0, cards[g] - 1):
                    if coupling_index(cards, f, v, g, u) != gf_index(cards, f, v, g, u):
                        raise AssertionError(
                            f"coupling_index disagrees with gen_field at "
                            f"({f},{v})x({g},{u})")
                    n_checked += 1
    return {"layout_equals_gen_field": True, "n_pairs": int(mine["size"]),
            "n_indices_checked": int(n_checked)}


def _value_terms(values: np.ndarray, grid_values: Sequence[np.ndarray]) -> np.ndarray:
    """``(N, F) float values -> (N, F) int term indices`` on the grid, nearest value."""
    V = np.asarray(values, np.float64)
    out = np.empty(V.shape, np.int64)
    toff = term_offsets([len(g) for g in grid_values])
    for f, g in enumerate(grid_values):
        gv = np.asarray(g, np.float64)
        out[:, f] = int(toff[f]) + np.argmin(np.abs(V[:, f][:, None] - gv[None, :]), 1)
    return out


# ---------------------------------------------------------------------------
# 4. The store
# ---------------------------------------------------------------------------


class DowndatableStore:
    """⭐⭐ The single-design store.  Four parts, one object, and it forgets exactly.

    Presents the same span interface every read on this programme already speaks --
    ``modes``, ``well_channels``, ``dim``, ``K``, ``byte_ledger()`` -- so
    ``gen_read.span_basis`` / ``render_span`` / ``TensorAddress`` read it unchanged and
    no rival's or control's code path moves.
    """

    def __init__(self, well_channels, dim: int, n_atoms: int,
                 cardinalities: Sequence[int], eig_rel_tol: float = EIG_REL_TOL,
                 coupling_alpha: float = COUPLING_ALPHA,
                 var_share_tol: float = VARIANCE_SHARE_TOL):
        self.well_channels = [np.ascontiguousarray(np.asarray(c, np.int64))
                              for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(n_atoms)
        self.cardinalities = [int(c) for c in cardinalities]
        self.eig_rel_tol = float(eig_rel_tol)
        #: the declared variance-share cut; see :data:`VARIANCE_SHARE_TOL`
        self.var_share_tol = float(var_share_tol)
        self.coupling_alpha = float(coupling_alpha)

        d = [int(c.size) for c in self.well_channels]
        # -- part 2: the sufficient statistics, float64, declared BESIDE the store ----
        self.S = [np.zeros((dk, dk), np.float64) for dk in d]
        self.t = [np.zeros(dk, np.float64) for dk in d]
        self.n = np.zeros(len(d), np.int64)
        # -- part 3: the request count table -----------------------------------------
        self.n_terms = int(sum(self.cardinalities))
        self.term_n = np.zeros(self.n_terms, np.int64)
        self.request_n = np.zeros(self.n_terms, np.int64)
        lay = coupling_layout(self.cardinalities)
        self.n_pairs = int(lay["size"])
        self.pair_n = np.zeros(self.n_pairs, np.int64)
        # -- part 4: the coupling table WITH HOLES -----------------------------------
        self.coupling = np.zeros(self.n_pairs, np.float32)
        self.coupling_hole = np.zeros(self.n_pairs, np.uint8)
        # -- part 1: the content ------------------------------------------------------
        self.modes = [np.zeros((self.n_atoms, dk), np.float32) for dk in d]
        self.rank_reports: list = [{} for _ in d]
        self.history: list = []

    # -- the span interface ---------------------------------------------------
    @property
    def K(self) -> int:
        return len(self.modes)

    @property
    def well_sizes(self) -> list:
        return [int(c.size) for c in self.well_channels]

    # -- fit ------------------------------------------------------------------
    def absorb(self, X: np.ndarray, values: Optional[np.ndarray] = None,
               grid_values: Optional[Sequence[np.ndarray]] = None) -> dict:
        """Add rows to the statistics.  ⭐ Additive, so the fit is itself streaming."""
        for k, ch in enumerate(self.well_channels):
            S, t, n = _row_moments(X, ch)
            self.S[k] += S
            self.t[k] += t
            self.n[k] += n
        if values is not None:
            self._count(values, grid_values, sign=+1)
        return {"op": "absorb", "n_rows": int(np.asarray(X).shape[0])}

    def _count(self, values, grid_values, sign: int) -> None:
        """The count table, ``+1`` on absorb and ``-1`` on delete.  Counts, not rows."""
        T = _value_terms(values, grid_values)
        lay = coupling_layout(self.cardinalities)
        toff = term_offsets(self.cardinalities)
        for f in range(T.shape[1]):
            np.add.at(self.term_n, T[:, f], int(sign))
        for j, (f, g) in enumerate(lay["pairs"]):
            idx = (int(lay["offsets"][j])
                   + (T[:, f] - int(toff[f])) * self.cardinalities[g]
                   + (T[:, g] - int(toff[g])))
            np.add.at(self.pair_n, idx, int(sign))
        if (self.term_n < 0).any() or (self.pair_n < 0).any():
            raise AssertionError(
                "a count went negative -- rows were deleted that were never absorbed")

    def note_requests(self, values, grid_values) -> dict:
        """The REQUEST half of the count table: what the store has been asked for.

        ⛔ Separate from ``term_n`` (what it holds evidence of) on purpose, and for the
        same reason w5 kept ``coupling_w`` and ``coupling_hole`` in different arrays: a
        thing that was asked for and a thing that was written are different facts, and
        an object that stores them in one array cannot tell them apart afterwards.
        """
        T = _value_terms(values, grid_values)
        for f in range(T.shape[1]):
            np.add.at(self.request_n, T[:, f], 1)
        return {"op": "note_requests", "n": int(np.asarray(values).shape[0])}

    # -- the content and the coupling, both recomputed from the statistics -----
    def refresh(self) -> dict:
        """Re-derive EVERY derived array from the statistics.  ⭐ No row is involved."""
        for k in range(self.K):
            self.modes[k], self.rank_reports[k] = modes_from_moments(
                self.S[k], self.t[k], int(self.n[k]), self.n_atoms, self.eig_rel_tol,
                var_share_tol=getattr(self, "var_share_tol", VARIANCE_SHARE_TOL))
        self._refresh_coupling()
        return {"op": "refresh", "ranks": [r["rank_kept"] for r in self.rank_reports]}

    def _refresh_coupling(self) -> None:
        """PMI with add-alpha smoothing -- a function of the COUNTS and nothing else.

        ⭐ That is the point: the coupling table downdates *exactly*, by construction,
        because a count downdates exactly.  ⛔ ``coupling_hole`` is not touched here; it
        is a different array and a deletion is not evidence.
        """
        a = self.coupling_alpha
        lay = coupling_layout(self.cardinalities)
        toff = term_offsets(self.cardinalities)
        N = float(max(int(self.n.max()) if self.n.size else 0, 1))
        for j, (f, g) in enumerate(lay["pairs"]):
            cf, cg = self.cardinalities[f], self.cardinalities[g]
            o = int(lay["offsets"][j])
            blk = self.pair_n[o:o + cf * cg].reshape(cf, cg).astype(np.float64)
            nf = self.term_n[int(toff[f]):int(toff[f]) + cf].astype(np.float64)
            ng = self.term_n[int(toff[g]):int(toff[g]) + cg].astype(np.float64)
            denom = N + a * cf * cg
            pj = (blk + a) / denom
            pf = (nf + a * cg) / denom
            pg = (ng + a * cf) / denom
            self.coupling[o:o + cf * cg] = np.log(
                pj / np.maximum(pf[:, None] * pg[None, :], 1e-300)).ravel().astype(
                    np.float32)

    # -- THE DELETION ---------------------------------------------------------
    def delete(self, X_del: np.ndarray, values_del: Optional[np.ndarray] = None,
               grid_values: Optional[Sequence[np.ndarray]] = None) -> dict:
        """⭐⭐ ``delete(rows)`` = covariance downdate + re-eigendecomposition + count downdate.

        ⛔ **No pass over the data.**  The retained corpus is never opened: the only rows
        this function reads are the ``X_del`` it was handed, and :data:`ROWS_CONSUMED`
        proves it -- the counter moves by exactly ``len(X_del)`` whatever the corpus size.

        ⛔ **No row is kept.**  Nothing about ``X_del`` survives the call except its
        contribution *subtracted out* of three aggregates; :func:`retention_audit`
        proves the object holds no per-row array at all.
        """
        before = int(ROWS_CONSUMED["n"])
        m = int(np.asarray(X_del).shape[0])
        if m <= 0:
            raise ValueError("delete() was handed no rows")
        for k, ch in enumerate(self.well_channels):
            S, t, n = _row_moments(X_del, ch)
            if n >= int(self.n[k]):
                raise AssertionError(
                    f"well {k}: deleting {n} of {int(self.n[k])} rows leaves no mean")
            self.S[k] -= S                      # covariance downdate
            self.t[k] -= t
            self.n[k] -= n                      # count downdate
        if values_del is not None:
            self._count(values_del, grid_values, sign=-1)   # count downdate, table half
        self.refresh()                          # re-eigendecomposition
        touched = int(ROWS_CONSUMED["n"]) - before
        if touched != m * self.K:
            raise AssertionError(
                f"delete() consumed {touched} row-reads for {m} rows over {self.K} "
                "wells -- something other than the deleted rows was read")
        rec = {"op": "delete", "n_rows_deleted": m,
               "rows_consumed_by_this_call": touched,
               "rows_consumed_per_well": m,
               "n_rows_remaining": int(self.n[0]),
               "ranks_after": [r["rank_kept"] for r in self.rank_reports],
               "refit": False,
               "what": ("covariance downdate + re-eigendecomposition + count downdate; "
                        "the retained corpus was not opened")}
        self.history.append(rec)
        return rec

    def punch_hole(self, f: int, v: int, g: int, u: int) -> dict:
        """⭐ w5's mechanism, unchanged: ONE byte, and the coupling VALUE stays put.

        The marginals ``(f,v)`` and ``(g,u)`` stay answerable and an unseen pair still
        gets its smoothed prior -- what changes is one byte of a different array saying
        *this conjunction was deleted*.  w5-delete measured the cost of this at 1 byte
        changed, 0 bytes outside, 0.0000 composition cost.
        """
        idx = coupling_index(self.cardinalities, f, v, g, u)
        before = float(self.coupling[idx])
        self.coupling_hole[idx] = 1
        rec = {"op": "punch_hole", "pair": [[int(f), int(v)], [int(g), int(u)]],
               "coupling_index": int(idx), "permitted_bytes": 1,
               "coupling_value_unchanged": bool(float(self.coupling[idx]) == before),
               "n_holes_after": int(self.coupling_hole.sum()), "refit": False}
        self.history.append(rec)
        return rec

    # -- ledgers: the store, the covariance, and the honest sum ----------------
    def byte_ledger(self, budget_bytes: int = 524_288) -> dict:
        """⭐ PART 1 ONLY -- the deployed content, at the primary pin, tolerance 0."""
        b = int(sum(M.size for M in self.modes) * 4)
        return {
            "deployed_bytes": b, "budget_bytes": int(budget_bytes),
            "matches_budget": bool(b == int(budget_bytes)),
            "param_count": b // 4,
            "what": (f"{self.n_atoms} atoms x {self.dim} float32 -- the per-well "
                     "leading principal subspace, the SAME block pca_tensor deploys"),
            "variance_explained_per_well": [round(float(r.get("variance_explained", 0.0)), 6)
                                            for r in self.rank_reports],
            "rank_kept_per_well": [int(r.get("rank_kept", 0)) for r in self.rank_reports],
            "rank_over_d_per_well": [round(float(r.get("rank_over_d", 0.0)), 4)
                                     for r in self.rank_reports],
            "any_well_saturated": bool(any(r.get("saturated") for r in self.rank_reports)),
            "note": ("⛔ the running covariance is NOT in this number; it is declared "
                     "separately in covariance_ledger() and never netted into it"),
        }

    def covariance_ledger(self) -> dict:
        """⛔ PART 2 -- the price of exact deletion, printed and never hidden."""
        d = self.well_sizes
        sq = int(sum(x * x for x in d))
        tri = int(sum(x * (x + 1) // 2 for x in d))
        return {
            "covariance_bytes": int(sq * 8),
            "covariance_bytes_per_well": [int(x * x * 8) for x in d],
            "d_k_squared_per_well": [int(x * x) for x in d],
            "sum_d_k_squared": sq,
            "dtype": "float64",
            "first_moment_bytes": int(sum(d) * 8),
            "count_bytes": int(self.n.size * 8),
            "bytes_formula": "8 * sum_k d_k^2  (+ 8 * sum_k d_k for the first moment)",
            "total_statistics_bytes": int(sq * 8 + sum(d) * 8 + self.n.size * 8),
            # the cheaper realisable variants, so the reader can price the design down
            # without recomputing anything -- still declared, still never netted.
            "if_symmetric_triangle_float64": int(tri * 8),
            "if_symmetric_triangle_float32": int(tri * 4),
            "if_full_square_float32": int(sq * 4),
            "why_float64": ("MEASURED: the downdate residual on S runs to 5e-10 "
                            "absolute on entries of order 1e+9; a float32 accumulator "
                            "(eps ~ 1e-7 relative) would bury the exactness this "
                            "object exists to have"),
            "note": ("⛔ EXTRA to the store's 524,288 B and reported BESIDE it, never "
                     "netted into it and never netted against the address book -- the "
                     "same rule w5 stated for address_bytes and w6 for coupling_bytes"),
        }

    def count_ledger(self) -> dict:
        """PART 3 -- the request count table."""
        return {
            "term_count_bytes": int(self.term_n.nbytes),
            "request_count_bytes": int(self.request_n.nbytes),
            "pair_count_bytes": int(self.pair_n.nbytes),
            "n_terms": int(self.n_terms), "n_pairs": int(self.n_pairs),
            "dtype": "int64",
            "total_count_bytes": int(self.term_n.nbytes + self.request_n.nbytes
                                     + self.pair_n.nbytes),
            "what": ("term_n = rows held carrying each (factor, value); request_n = "
                     "times each value was ASKED for; pair_n = rows held carrying each "
                     "conjunction. ⛔ Held apart on purpose: asked-for and written-down "
                     "are different facts."),
        }

    def coupling_ledger(self) -> dict:
        """PART 4 -- the coupling table and its holes."""
        return {
            "coupling_bytes": int(self.coupling.nbytes),
            "coupling_hole_bytes": int(self.coupling_hole.nbytes),
            "bytes_per_hole": 1,
            "n_pairs": int(self.n_pairs),
            "n_holes": int(self.coupling_hole.sum()),
            "value_and_hole_in_different_arrays": True,
            "what": ("coupling = add-alpha PMI, a function of the COUNTS alone, so it "
                     "downdates exactly with them; coupling_hole = w5's mechanism, ONE "
                     "byte per pair, 0 bytes outside, and the value is left untouched "
                     "so an unseen pair still gets its prior"),
        }

    def honest_ledger(self, budget_bytes: int = 524_288) -> dict:
        """⛔⭐ The sum of all four parts, against the pin, labelled over or under.

        ⛔ Task section 3: print the store bytes and the covariance bytes separately,
        both declared.  This is the *sum*, printed so nobody has to add it up and so
        nothing can be quietly netted; it does not replace the two separate numbers.
        """
        st = self.byte_ledger(budget_bytes)
        cv = self.covariance_ledger()
        ct = self.count_ledger()
        cp = self.coupling_ledger()
        tot = int(st["deployed_bytes"] + cv["total_statistics_bytes"]
                  + ct["total_count_bytes"] + cp["coupling_bytes"]
                  + cp["coupling_hole_bytes"])
        cheap = int(st["deployed_bytes"] + cv["if_symmetric_triangle_float64"]
                    + cv["first_moment_bytes"] + cv["count_bytes"]
                    + ct["total_count_bytes"] + cp["coupling_bytes"]
                    + cp["coupling_hole_bytes"])
        return {
            "store_bytes": int(st["deployed_bytes"]),
            "covariance_bytes": int(cv["covariance_bytes"]),
            "count_bytes": int(ct["total_count_bytes"]),
            "coupling_bytes": int(cp["coupling_bytes"] + cp["coupling_hole_bytes"]),
            "honest_total_bytes": tot,
            "honest_total_bytes_symmetric_triangle": cheap,
            "budget_bytes": int(budget_bytes),
            "times_over_budget": round(tot / float(budget_bytes), 4),
            "times_over_budget_symmetric_triangle": round(cheap / float(budget_bytes), 4),
            "over_budget": bool(tot > int(budget_bytes)),
            "verdict": ("⛔ THE HONEST LEDGER IS OVER THE PIN. The content alone is at "
                        "524,288 B exactly, but exact deletion is paid for in "
                        "sufficient statistics and they are not free. Printed, not "
                        "netted (task section 3)."),
        }

    def rank_report(self) -> dict:
        """The rank cut, per well, with the margin on both sides of the threshold."""
        rr = [dict(r) for r in self.rank_reports]
        # ⚠ MEASURED BUG, fixed here: marginality is only a question in the wells where
        # the TOLERANCE is what binds. In a well limited by the atom budget (A-1 < the
        # number of directions with real variance) the first dropped eigenvalue is
        # supposed to be large -- direction 64 is dropped because there are only 64
        # atoms, not because it is numerically null. Scoring those wells against the
        # threshold reported "cut_is_marginal = True" on a cut whose true margins are
        # 2e3 and 5e-5. Only tolerance-limited wells are counted.
        tol_lim = [r for r in rr if r["rank_limited_by"] == "eigenvalue tolerance"]
        lo = [r["smallest_kept_over_threshold"] for r in tol_lim
              if np.isfinite(r["smallest_kept_over_threshold"])]
        hi = [r["largest_dropped_over_threshold"] for r in tol_lim]
        return {
            "eig_rel_tol": self.eig_rel_tol, "per_well": rr,
            "n_wells_limited_by_tolerance": len(tol_lim),
            "n_wells_limited_by_atom_budget": sum(
                1 for r in rr if r["rank_limited_by"] == "atom budget A-1"),
            "min_smallest_kept_over_threshold": float(min(lo)) if lo else float("inf"),
            "max_largest_dropped_over_threshold": float(max(hi)) if hi else 0.0,
            "cut_is_marginal": bool((min(lo) if lo else np.inf) < 10.0
                                    or (max(hi) if hi else 0.0) > 0.1),
            "marginality_scored_over": ("only the wells where the eigenvalue tolerance "
                                        "is what binds the rank; a well limited by the "
                                        "atom budget drops a large eigenvalue by "
                                        "design and says nothing about the cut"),
            "what": ("a direction below the threshold is written as a ZERO row: it "
                     "costs the same bytes and buys no span, which is PCASpanStore's "
                     "own stated convention for an atom a well cannot use"),
        }

    def content_cost_report(self) -> dict:
        """⛔ What the declared variance-share cut REMOVED from the store.

        ⭐ A remedy that stabilises the object by deleting its content is not free, and
        this is the invoice: directions dropped per well, and the share of each well's
        variance that went with them.
        """
        rr = [dict(r) for r in self.rank_reports]
        return {
            "var_share_tol": getattr(self, "var_share_tol", VARIANCE_SHARE_TOL),
            "per_well": [{
                "well": i, "d_k": r["d_k"],
                "rank_kept": r["rank_kept"],
                "rank_kept_without_share_cut": r.get("rank_kept_without_share_cut"),
                "n_directions_dropped_by_share_cut": r.get(
                    "n_directions_dropped_by_share_cut"),
                "variance_share_lost_to_share_cut": r.get(
                    "variance_share_lost_to_share_cut"),
                "smallest_kept_variance_share": r.get(
                    "smallest_kept_variance_share"),
                "largest_dropped_variance_share": r.get(
                    "largest_dropped_variance_share"),
                "variance_explained": r.get("variance_explained"),
            } for i, r in enumerate(rr)],
            "total_directions_dropped": int(sum(
                r.get("n_directions_dropped_by_share_cut", 0) for r in rr)),
            "worst_variance_share_lost_in_a_well": float(max(
                [r.get("variance_share_lost_to_share_cut", 0.0) for r in rr] or [0.0])),
            "wells_where_share_cut_binds": [
                i for i, r in enumerate(rr) if r.get("share_cut_binds")],
            "store_bytes_unchanged": ("the dropped directions are written as ZERO "
                                      "rows, so the byte pin is untouched -- the cost "
                                      "is content, not bytes"),
        }

    def well_posedness_report(self) -> dict:
        """⛔⭐ To what precision are this store's OWN BYTES determined? Fit time.

        ⭐ The point of computing this from the statistics alone is that the object
        declares its precision **before** anyone deletes anything. The failure it exists
        to prevent -- test seed 2, deviation 3.92e-06 = 33 float32 ulp -- was invisible
        on the val leg and would have been quoted as exactness across seeds.

        ⚠ A well that reports ``bytes_determined_below_one_float32_ulp = False`` is
        not broken arithmetic: its statistics still downdate exactly. It is a well whose
        *eigenvectors* are not a function of its data, so neither the downdate nor a
        refit from scratch has a unique answer for those bytes.
        """
        rr = [dict(r) for r in self.rank_reports]
        f32 = float(np.spacing(np.float32(1.0)))
        sens = [r.get("predicted_byte_sensitivity", float("nan")) for r in rr]
        worst = max(sens) if sens else float("nan")
        bad = [i for i, r in enumerate(rr)
               if not r.get("bytes_determined_below_one_float32_ulp", True)]
        return {
            "float32_ulp_at_unit_scale": f32,
            "per_well": [{
                "well": i, "d_k": r["d_k"], "rank_kept": r["rank_kept"],
                "rank_limited_by": r["rank_limited_by"],
                "min_rel_gap_among_kept": r.get("min_rel_gap_among_kept"),
                "min_rel_gap_to_any_neighbour": r.get("min_rel_gap_to_any_neighbour"),
                "predicted_byte_sensitivity": r.get("predicted_byte_sensitivity"),
                "predicted_sensitivity_in_float32_ulp": r.get(
                    "predicted_sensitivity_in_float32_ulp"),
                "bytes_determined_below_one_float32_ulp": r.get(
                    "bytes_determined_below_one_float32_ulp"),
            } for i, r in enumerate(rr)],
            "worst_predicted_byte_sensitivity": worst,
            "worst_predicted_in_float32_ulp": worst / f32 if f32 > 0 else None,
            "wells_not_determined_at_float32": bad,
            "store_bytes_well_posed": bool(not bad),
            "what": ("computed from the sufficient statistics at FIT time -- no "
                     "deletion and no second fit. ⛔ A well listed in "
                     "wells_not_determined_at_float32 has atom bytes that are not a "
                     "function of its data, so no deletion procedure and no refit can "
                     "agree on them; that is a property of the CONTENT, not of the "
                     "downdate."),
        }

    def content_sha(self) -> str:
        h = hashlib.sha256()
        for M in self.modes:
            h.update(np.ascontiguousarray(np.asarray(M, np.float32)).tobytes())
        return h.hexdigest()

    # -- the inventory the audits run over ------------------------------------
    def state_arrays(self) -> list:
        """EVERY array this object holds: ``(name, array, declared shape formula)``.

        ⛔ Exhaustive by construction -- :func:`retention_audit` cross-checks it against
        ``vars(self)``, so an array added later and forgotten here fails the audit
        instead of slipping past it.
        """
        out = []
        for k in range(self.K):
            dk = self.well_sizes[k]
            out.append((f"modes[{k}]", self.modes[k], f"(A={self.n_atoms}, d_k={dk})"))
            out.append((f"S[{k}]", self.S[k], f"(d_k={dk}, d_k={dk})"))
            out.append((f"t[{k}]", self.t[k], f"(d_k={dk},)"))
            out.append((f"well_channels[{k}]", self.well_channels[k], f"(d_k={dk},)"))
        out.append(("n", self.n, f"(K={self.K},)"))
        out.append(("term_n", self.term_n, f"(T={self.n_terms},)"))
        out.append(("request_n", self.request_n, f"(T={self.n_terms},)"))
        out.append(("pair_n", self.pair_n, f"(NC={self.n_pairs},)"))
        out.append(("coupling", self.coupling, f"(NC={self.n_pairs},)"))
        out.append(("coupling_hole", self.coupling_hole, f"(NC={self.n_pairs},)"))
        return out


# ---------------------------------------------------------------------------
# 5. The three assertions' machinery
# ---------------------------------------------------------------------------


class _Tripwire:
    """An object that raises on ANY use.  It is what the corpus is replaced by.

    ⛔ Not a mock that returns zeros -- a mock that returns zeros lets a delete that
    secretly reads the corpus finish and merely get the wrong answer.  This one turns
    "the delete read the corpus" into an exception at the instant it happens.
    """

    def __init__(self, what: str):
        self._what = str(what)

    def _boom(self, *_a, **_k):
        raise AssertionError(
            f"delete() touched {self._what} -- this object claims NO PASS OVER THE DATA "
            "and it just made one")

    __getitem__ = __getattr__ = __call__ = __len__ = __iter__ = __array__ = _boom


def no_data_pass(fn, *args, modules=(), **kwargs) -> dict:
    """⛔⭐ ASSERTION 2, MECHANICALLY: run ``fn`` with the corpus **unreachable**.

    Two independent proofs, because either alone has a hole:

    * **the tripwire** -- every named route to the feature corpus
      (``features_mmap``, ``gather``, ``np.load``, ``np.memmap``) is replaced by an
      object that raises.  A delete that reads the corpus by any of those routes dies
      inside the call rather than returning a number nobody can check.
    * **the counter** -- :data:`ROWS_CONSUMED` is read before and after.  This catches
      the route the tripwire does not name: a corpus the caller already had in memory
      and handed in.  A tripwire cannot see that; a row counter can.

    ⭐ The counter is the load-bearing one, and it is checked against the corpus size by
    the caller: what makes "no pass over the data" true is that the number is
    ``m`` rows at **every** corpus size, not merely that it is small.
    """
    import builtins  # noqa: F401  (kept so the patch list reads as a complete account)

    patched = []
    try:
        for mod in modules:
            for name in ("features_mmap", "gather"):
                if hasattr(mod, name):
                    patched.append((mod, name, getattr(mod, name)))
                    setattr(mod, name, _Tripwire(f"{mod.__name__}.{name}"))
        for name in ("load", "memmap", "fromfile"):
            patched.append((np, name, getattr(np, name)))
            setattr(np, name, _Tripwire(f"numpy.{name}"))
        before = int(ROWS_CONSUMED["n"])
        result = fn(*args, **kwargs)
        after = int(ROWS_CONSUMED["n"])
    finally:
        for mod, name, orig in patched:
            setattr(mod, name, orig)
    return {"result": result, "rows_consumed": int(after - before),
            "tripwires_installed": [f"{m.__name__}.{n}" for m, n, _ in patched],
            "n_tripwires": len(patched),
            "corpus_was_unreachable_during_the_call": True}


def retention_audit(store: DowndatableStore, n_rows_fitted: int) -> dict:
    """⛔⭐ ASSERTION 3, MECHANICALLY: **no row is retained anywhere in the object.**

    Three checks, and the third is the one that cannot be argued with:

    1. **the inventory is complete** -- every ndarray reachable from ``vars(store)`` is
       named in :meth:`DowndatableStore.state_arrays`.  An array nobody declared is a
       place a row could hide.
    2. **no shape depends on the corpus** -- every array's shape is a function of
       ``(A, d_k, K, T, NC)`` only.  Not one axis is ``n_rows``.
    3. **the object's size is independent of the corpus** -- checked by the caller by
       fitting at two corpus sizes and comparing byte counts. An object that keeps items
       grows with the corpus; this one does not.

    ⚠ Declared honestly, because it is the one way the argument could fail: the
    covariance of ``n <= d_k`` rows is rank-deficient in a way that can expose the rows
    themselves.  The audit therefore ASSERTS ``n > d_k`` per well and reports the margin.
    """
    declared = {name: arr for name, arr, _ in store.state_arrays()}
    seen, undeclared = set(), []

    def _walk(obj, path):
        if isinstance(obj, np.ndarray):
            if id(obj) not in {id(a) for a in declared.values()}:
                undeclared.append((path, tuple(obj.shape)))
            seen.add(id(obj))
        elif isinstance(obj, (list, tuple)):
            for i, o in enumerate(obj):
                _walk(o, f"{path}[{i}]")
        elif isinstance(obj, dict):
            for kk, vv in obj.items():
                _walk(vv, f"{path}[{kk!r}]")

    for attr, val in vars(store).items():
        _walk(val, attr)
    if undeclared:
        raise AssertionError(
            f"retention audit: {len(undeclared)} array(s) reachable from the store are "
            f"not in state_arrays() -- {undeclared[:4]} -- an undeclared array is a "
            "place a row could be hiding")

    allowed = {store.n_atoms, store.K, store.n_terms, store.n_pairs}
    allowed |= set(store.well_sizes)
    rows = []
    for name, arr, formula in store.state_arrays():
        a = np.asarray(arr)
        bad = [int(x) for x in a.shape if int(x) not in allowed]
        rows.append({"name": name, "shape": list(a.shape), "dtype": str(a.dtype),
                     "bytes": int(a.nbytes), "declared_shape": formula,
                     "axes_not_explained": bad})
        if bad:
            raise AssertionError(
                f"retention audit: {name} has shape {a.shape}, and axis/axes {bad} are "
                f"not explained by (A, d_k, K, T, NC) -- this array may hold rows")
        if int(n_rows_fitted) in {int(x) for x in a.shape}:
            raise AssertionError(
                f"retention audit: {name} has an axis of length n_rows "
                f"({n_rows_fitted}) -- this object is keeping items")

    margins = [int(nk) - dk for nk, dk in zip(store.n, store.well_sizes)]
    if min(margins) <= 0:
        raise AssertionError(
            f"retention audit: a well has n <= d_k (margins {margins}). A covariance "
            "of fewer rows than channels is rank-deficient in a way that can expose "
            "the rows themselves, so the no-items claim does not hold there")
    return {
        "items_kept": False,
        "n_arrays": len(rows), "arrays": rows,
        "total_object_bytes": int(sum(r["bytes"] for r in rows)),
        "n_rows_fitted": int(n_rows_fitted),
        "no_array_has_an_n_rows_axis": True,
        "every_shape_explained_by": "(A, d_k, K, T, NC)",
        "inventory_complete_vs_vars": True,
        "n_over_d_margin_per_well": [int(m) for m in margins],
        "min_n_over_d_margin": int(min(margins)),
        "what": ("every array in the object, named, shaped and explained; not one axis "
                 "is the corpus size, so the object cannot be holding a row"),
    }


def compare_stores(a: DowndatableStore, b: DowndatableStore) -> dict:
    """⭐⭐ ASSERTION 1: is ``a`` the same store as ``b``?  Max abs deviation PER WELL.

    Reports, per well: the max absolute deviation on the deployed float32 atom block,
    that deviation in units of a **float32 ulp** at the block's own scale (the number
    that decides whether "to float precision" is met, since a float32 store cannot
    represent a smaller difference than one ulp), whether the blocks are bit-identical,
    and -- because a deviation in a near-degenerate eigenspace would otherwise be
    mistaken for an inexact downdate -- the **subspace** deviation, the operator norm of
    the difference of the two orthogonal projectors, which is invariant to any rotation
    within a degenerate eigenspace.
    """
    if a.K != b.K:
        raise AssertionError(f"{a.K} wells vs {b.K}: not the same store shape")
    per_well, ok = [], True
    for k in range(a.K):
        A = np.asarray(a.modes[k], np.float32)
        B = np.asarray(b.modes[k], np.float32)
        if A.shape != B.shape:
            raise AssertionError(f"well {k}: {A.shape} vs {B.shape}")
        dif = np.abs(A.astype(np.float64) - B.astype(np.float64))
        dev = float(dif.max()) if dif.size else 0.0
        scale = float(np.max(np.abs(B))) if B.size else 1.0
        # ⚠ WHICH UNIT TO MEASURE "float precision" IN -- both obvious answers are
        # wrong, and both are reported so the choice is visible rather than convenient.
        #   * at the BLOCK's maximum: nearly free. The mean atom has norm ~18, whose
        #     float32 ulp is ~2e-6, and every deviation here is 1e-10 or smaller.
        #   * ENTRYWISE at each entry's own magnitude: meaningless in the other
        #     direction. An atom component that is ~0 has a denormal ulp of 1e-45, so
        #     a 1e-10 difference on a near-zero component scores ~3e7 ulp and the
        #     statistic is decided entirely by components that carry no direction.
        # ⭐ The unit that fits the object: every direction row is a UNIT VECTOR, so
        # the precision that matters is absolute precision at unit scale -- one float32
        # ulp at 1.0 is 1.19e-7, and that is the finest distinction a float32 store can
        # draw between two directions. The mean row has norm ~18 and a coarser ulp, so
        # scoring it at unit scale too is conservative.
        mag = np.maximum(np.abs(A.astype(np.float64)), np.abs(B.astype(np.float64)))
        ulp_e = np.spacing(np.maximum(mag, np.finfo(np.float32).tiny).astype(np.float32))
        ulp_entry = float(np.max(dif / np.asarray(ulp_e, np.float64))) if dif.size else 0.0
        ulp_unit = float(np.spacing(np.float32(1.0)))
        ulp_dist = dev / ulp_unit
        ulp = float(np.spacing(np.float32(max(scale, np.finfo(np.float32).tiny))))
        # subspace-invariant deviation: ||P_A - P_B||_2 over the direction rows
        ra = int(a.rank_reports[k].get("rank_kept", 0))
        rb = int(b.rank_reports[k].get("rank_kept", 0))
        if ra == rb and ra > 0:
            Qa = np.linalg.qr(A[1:1 + ra].T.astype(np.float64))[0]
            Qb = np.linalg.qr(B[1:1 + rb].T.astype(np.float64))[0]
            proj = float(np.linalg.norm(Qa @ Qa.T - Qb @ Qb.T, 2))
        else:
            proj = float("nan")
        row = {
            "well": k, "d_k": int(a.well_sizes[k]),
            "max_abs_dev": dev,
            # THE criterion: the deviation in float32 ulp AT UNIT SCALE, the finest
            # distinction a float32 store can draw between two unit directions.
            "max_dev_in_float32_ulp_at_unit_scale": ulp_dist,
            "max_rel_dev_to_block_scale": (dev / scale) if scale > 0 else 0.0,
            # both rejected alternatives, reported so the choice above is checkable
            "max_entrywise_float32_ulp_distance": ulp_entry,
            "max_abs_dev_over_ulp_at_block_scale": (dev / ulp) if ulp > 0 else float("inf"),
            "bit_identical": bool(np.array_equal(A, B)),
            "atom_block_scale": scale,
            "float32_ulp_at_scale": ulp,
            "rank_kept_a": ra, "rank_kept_b": rb,
            "ranks_agree": bool(ra == rb),
            "subspace_projector_dev": proj,
            "n_rows_a": int(a.n[k]), "n_rows_b": int(b.n[k]),
            "counts_agree": bool(int(a.n[k]) == int(b.n[k])),
        }
        ok = ok and row["ranks_agree"] and row["counts_agree"]
        per_well.append(row)
    devs = [r["max_abs_dev"] for r in per_well]
    ulps = [r["max_dev_in_float32_ulp_at_unit_scale"] for r in per_well]
    return {
        "max_abs_dev_per_well": devs,
        "max_abs_dev": float(max(devs)) if devs else 0.0,
        "max_dev_in_float32_ulp_at_unit_scale_per_well": ulps,
        "max_dev_in_float32_ulp": float(max(ulps)) if ulps else 0.0,
        "max_entrywise_float32_ulp_distance_per_well": [
            r["max_entrywise_float32_ulp_distance"] for r in per_well],
        "n_wells_bit_identical": int(sum(r["bit_identical"] for r in per_well)),
        "ranks_agree_in_every_well": bool(ok),
        "counts_equal": bool(all(r["counts_agree"] for r in per_well)),
        "term_counts_equal": bool(np.array_equal(a.term_n, b.term_n)),
        "pair_counts_equal": bool(np.array_equal(a.pair_n, b.pair_n)),
        "coupling_max_abs_dev": float(np.max(np.abs(
            a.coupling.astype(np.float64) - b.coupling.astype(np.float64)))),
        "per_well": per_well,
    }


# ---------------------------------------------------------------------------
# 6. The gauge: why an exactly-forgetting store can still read differently
# ---------------------------------------------------------------------------


def pinned_span_basis(store, tol: float = 1e-8,
                      mu_rel_tol: float = None) -> list:
    """⭐⭐ An orthonormal basis of each well's span that is a CONTINUOUS function of
    the atoms -- the fix for a defect that exact deletion exposes and nothing else does.

    ⛔ THE DEFECT, and it is downstream of the store rather than in it.
    ``gen_read.span_basis`` re-derives ``U_k`` by taking the SVD of the atom block on
    load.  On *this* content the atom block is ``[mu ; v_1 .. v_r]`` with the ``v_j``
    **orthonormal**, so the block's singular values are ``1`` repeated ``r`` times plus
    one larger value from the mean -- a **maximally degenerate spectrum**.  Inside a
    degenerate singular subspace the SVD's rotation is arbitrary, so a perturbation of
    the atoms of order 1e-10 can rotate ``U_k`` by order 1.  An address book fitted
    against the pre-deletion ``U_k`` is then stale against the post-deletion one **even
    though the two stores are the same store to 1e-10**.

    ⚠ ``pca_tensor`` has exactly the same gauge freedom -- it deploys unit eigenvectors
    too -- and never pays for it, because its deletion verb
    (``exp_gen_read.delete_value_from_address``) is declared in its own docstring as "an
    operation on the INTERFACE, not on atoms" and so never moves an atom.  ⛔ The
    incumbent evades this by not having the property.  An object whose deletion really
    is a charter B.4 operation on atoms cannot evade it and has to pin the gauge.

    ⭐ THE FIX, and it costs zero bytes exactly as ``span_basis`` does: do not ask an
    SVD where the basis should point -- the store already knows.  The direction rows are
    already orthonormal and already ordered by eigenvalue, so take them as they are, and
    append the component of the mean orthogonal to them.  The result spans exactly what
    ``span_basis`` spans, is orthonormal, involves no degenerate decomposition, and
    moves continuously with the atoms: a 1e-10 change in the store is a 1e-10 change in
    the basis.
    """
    mrt = MU_PERP_REL_TOL if mu_rel_tol is None else float(mu_rel_tol)
    out = []
    for G in store.modes:
        M = np.asarray(G, np.float64)
        mu, dirs = M[0], M[1:]
        keep = dirs[np.linalg.norm(dirs, axis=1) > tol]          # drop the zero rows
        U = keep.T if keep.size else np.zeros((M.shape[1], 0), np.float64)
        r = mu - (U @ (U.T @ mu)) if U.shape[1] else mu
        nr = float(np.linalg.norm(r))
        nmu = max(float(np.linalg.norm(mu)), 1e-300)
        # ⛔ The mean joins the basis only if it is really outside the directions'
        # span. In a rank-deficient well it is not (measured: 2.3e-8 relative), and
        # admitting the leftover would put an arbitrary null direction back into the
        # basis -- undoing, at the interface, exactly what the rank cut does at the
        # store. See MU_PERP_REL_TOL.
        if nr / nmu > mrt:
            U = np.hstack([U, (r / nr)[:, None]])
        out.append(np.ascontiguousarray(U.astype(np.float32)))
    return out


def pinned_basis_report(store, mu_rel_tol: float = None) -> dict:
    """The mean-admission decision per well, with the margin on both sides of it."""
    mrt = MU_PERP_REL_TOL if mu_rel_tol is None else float(mu_rel_tol)
    rows = []
    for k, G in enumerate(store.modes):
        M = np.asarray(G, np.float64)
        mu, dirs = M[0], M[1:]
        keep = dirs[np.linalg.norm(dirs, axis=1) > 1e-8]
        U = keep.T if keep.size else np.zeros((M.shape[1], 0), np.float64)
        r = mu - (U @ (U.T @ mu)) if U.shape[1] else mu
        nmu = max(float(np.linalg.norm(mu)), 1e-300)
        rel = float(np.linalg.norm(r)) / nmu
        rows.append({"well": k, "n_directions": int(keep.shape[0]),
                     "mu_norm": nmu, "mu_perp_relative_norm": rel,
                     "mean_admitted": bool(rel > mrt),
                     "margin_over_threshold": float(rel / mrt)})
    return {"mu_perp_rel_tol": mrt, "per_well": rows,
            "span_ranks": [r["n_directions"] + int(r["mean_admitted"]) for r in rows],
            "closest_margin_above": min([r["margin_over_threshold"] for r in rows
                                         if r["mean_admitted"]] or [float("inf")]),
            "closest_margin_below": max([r["margin_over_threshold"] for r in rows
                                         if not r["mean_admitted"]] or [0.0])}


def gauge_report(store_a, store_b, coeffs, basis_fn) -> dict:
    """⛔ Push the SAME coefficients through two stores that are the SAME store.

    Any difference in the rendered vector is therefore **pure gauge** -- it cannot be an
    effect of the deletion, because the two stores agree to 1e-10.  This is the
    measurement that stops a gauge artefact being reported as a composition cost.
    """
    from .gen_read import render_span

    Ua, Ub = basis_fn(store_a), basis_fn(store_b)
    ra = [int(np.asarray(U).shape[1]) for U in Ua]
    rb = [int(np.asarray(U).shape[1]) for U in Ub]
    if ra != rb:
        return {"basis": basis_fn.__name__, "ranks_a": ra, "ranks_b": rb,
                "comparable": False,
                "note": "the two bases do not even have the same rank"}
    Xa = render_span(store_a, [np.asarray(c, np.float32)[:, :r] for c, r in
                               zip(coeffs, ra)], Ua)
    Xb = render_span(store_b, [np.asarray(c, np.float32)[:, :r] for c, r in
                               zip(coeffs, rb)], Ub)
    d = np.abs(Xa.astype(np.float64) - Xb.astype(np.float64))
    scale = float(np.max(np.abs(Xb))) if Xb.size else 1.0
    return {
        "basis": basis_fn.__name__, "comparable": True, "span_ranks": ra,
        "max_abs_render_dev": float(d.max()),
        "median_row_l2_dev": float(np.median(np.linalg.norm(Xa - Xb, axis=1))),
        "render_scale": scale,
        "relative_to_render_scale": float(d.max() / max(scale, 1e-12)),
        "what": ("the same coefficients through two stores that agree to 1e-10; any "
                 "deviation here is the BASIS, not the deletion"),
    }


def direction_correspondence(modes_pre, modes_post, tol: float = 1e-8) -> dict:
    """⭐ Which post-deletion direction is which pre-deletion direction?

    ⛔ THE DEFECT THIS MEASURES, raised by ``hub/single-design``: the store orders its
    directions by **eigenvalue**, and a deletion changes the eigenvalues, so two
    directions whose eigenvalues cross **swap slots**.  An address book fitted before
    the deletion then has slot ``j`` multiplying a direction that is no longer direction
    ``j`` -- and the two are *orthogonal*, so the error in that slot is O(1), not small,
    even though the pair is nearly degenerate and the SPAN barely moves.

    ⭐ The matching is greedy on descending ``|overlap|`` rather than a per-row argmax,
    because an argmax can assign two pre-directions to the same post-direction and then
    the "permutation" is not one.

    ⚠ This is a MEASUREMENT, and the permutation it returns is a real (small) cost if
    it is used: ``sum_k r_k`` indices, ~2 B each.  Declared, not free.
    """
    out = []
    for k, (Gp, Gq) in enumerate(zip(modes_pre, modes_post)):
        P = np.asarray(Gp, np.float64)[1:]
        Q = np.asarray(Gq, np.float64)[1:]
        live_p = np.flatnonzero(np.linalg.norm(P, axis=1) > tol)
        live_q = np.flatnonzero(np.linalg.norm(Q, axis=1) > tol)
        M = np.abs(P[live_p] @ Q[live_q].T)
        perm = np.full(live_p.size, -1, np.int64)
        cos = np.zeros(live_p.size, np.float64)
        taken = np.zeros(live_q.size, bool)
        order = np.dstack(np.unravel_index(np.argsort(M, axis=None)[::-1], M.shape))[0]
        for i, j in order:
            if perm[i] < 0 and not taken[j]:
                perm[i], cos[i], taken[j] = int(j), float(M[i, j]), True
                if (perm >= 0).all():
                    break
        shift = np.abs(perm - np.arange(perm.size))
        out.append({
            "well": k, "r": int(live_p.size),
            "permutation": [int(x) for x in perm],
            "n_rank_order_changed": int((shift > 0).sum()),
            "frac_rank_order_changed": float((shift > 0).mean()) if shift.size else 0.0,
            "max_shift": int(shift.max()) if shift.size else 0,
            "min_matched_cosine": float(cos.min()) if cos.size else 1.0,
            "median_matched_cosine": float(np.median(cos)) if cos.size else 1.0,
            "is_identity": bool((shift == 0).all()),
        })
    return {
        "per_well": out,
        "any_reordering": bool(any(not w["is_identity"] for w in out)),
        "total_slots_reordered": int(sum(w["n_rank_order_changed"] for w in out)),
        "total_slots": int(sum(w["r"] for w in out)),
        "permutation_bytes_if_carried": int(sum(w["r"] for w in out) * 2),
        "what": ("the store orders directions by eigenvalue; a deletion moves the "
                 "eigenvalues, so slots swap. The swapped pairs are nearly degenerate "
                 "so the SPAN barely moves, but the two directions are orthogonal, so "
                 "a book slot pointed at the wrong one is O(1) wrong in that slot."),
    }


def pinned_span_basis_matched(store, modes_pre, tol: float = 1e-8,
                              mu_rel_tol: float = None) -> tuple:
    """:func:`pinned_span_basis`, with the directions put back in ``modes_pre``'s order.

    ⭐ This is what makes "delete, then read with the SAME address book" a fair test:
    slot ``j`` of the book keeps pointing at the direction it was fitted against.
    ⛔ It is not free -- see ``permutation_bytes_if_carried`` -- and it needs the
    pre-deletion directions at delete time.  It needs **no data pass** and it does not
    change one byte of the address book, which is the property that matters for a
    charter B.4 claim: the interface is not refitted, only re-indexed.
    """
    corr = direction_correspondence(modes_pre, store.modes, tol=tol)
    reordered = []
    for G, w in zip(store.modes, corr["per_well"]):
        M = np.asarray(G, np.float32).copy()
        dirs = M[1:]
        live = np.flatnonzero(np.linalg.norm(np.asarray(dirs, np.float64), axis=1) > tol)
        perm = np.asarray(w["permutation"], np.int64)
        block = dirs[live].copy()
        M[1:1 + live.size] = block[perm]        # pre-order j <- post-direction perm[j]
        reordered.append(M)

    class _Reordered:
        modes = reordered
        well_channels = store.well_channels
        dim = store.dim

    return pinned_span_basis(_Reordered, tol=tol, mu_rel_tol=mu_rel_tol), corr


def carry_address_book(addr, U_pre, U_post):
    """⭐⭐ Bring the address book into correspondence with the downdated store.

    ``hub/single-design``'s mechanism, adopted after it measured that my permutation
    match removes only 63% of the error: the residual is rotation **within**
    near-degenerate subspaces, which a slot swap cannot express.  The full linear
    correspondence can::

        M_k = U0_k^T U1_k        W'_k = W_k M_k

    ⭐ Why this is the right answer and not a fudge: the read becomes
    ``a W M U1^T = (a W U0^T) U0 U1^T``... more precisely ``c M U1^T`` is the
    pre-deletion ambient answer **projected onto the post-deletion span**.  The
    component of the old answer lying outside the new span -- which is exactly the
    deleted content -- is projected out, and nothing else moves.  So the carry does not
    mask the deletion; it is what the deletion *means* at the interface.

    ⛔ THE COST, declared. No pass over the data: ``M_k`` is formed from two bases
    already in hand at delete time, ``O(r_k^2 d_k)``.  But the book's ``W`` **is
    rewritten**, so unlike :func:`pinned_span_basis` this is not byte-identical at the
    interface -- ``sum_k R r_k`` floats change value (not count). For a charter B.4
    argument that is weaker than "nothing changed" and stronger than "refitted".
    """
    import copy

    new = copy.deepcopy(addr)
    Ms = []
    for k, (U0, U1) in enumerate(zip(U_pre, U_post)):
        A0 = np.asarray(U0, np.float64)
        A1 = np.asarray(U1, np.float64)
        if A0.shape[1] != np.asarray(addr.W[k]).shape[1]:
            raise AssertionError(
                f"well {k}: the book emits {np.asarray(addr.W[k]).shape[1]} "
                f"coefficients but the pre-deletion basis has {A0.shape[1]} columns")
        Ms.append(A0.T @ A1)
    # c = (a W) * s + m  ->  c M = (a W) diag(s) M + m M, exactly
    o = 0
    for k, M in enumerate(Ms):
        Wk = np.asarray(new.W[k], np.float64)
        r_in = Wk.shape[1]
        if getattr(new, "scale", None) is not None:
            s = np.asarray(new.scale[o:o + r_in], np.float64)
            Wk = Wk * s[None, :]
        new.W[k] = np.ascontiguousarray((Wk @ M).astype(np.asarray(addr.W[k]).dtype))
        o += r_in
    if getattr(new, "scale", None) is not None:
        cen, o = [], 0
        for k, M in enumerate(Ms):
            r_in = np.asarray(addr.W[k]).shape[1]
            cen.append(np.asarray(new.center[o:o + r_in], np.float64) @ M)
            o += r_in
        new.center = np.concatenate(cen).astype(np.asarray(addr.center).dtype)
        new.scale = np.ones_like(new.center)
    return new, {
        "op": "carry_address_book",
        "n_wells": len(Ms),
        "correspondence_min_singular_value": [
            float(np.linalg.svd(M, compute_uv=False).min()) for M in Ms],
        "correspondence_max_abs_off_identity": [
            float(np.abs(M - np.eye(M.shape[0])).max()) for M in Ms],
        "address_book_bytes_changed": int(
            sum(np.asarray(w).size for w in addr.W) * 4),
        "address_book_bytes_total": int(addr.byte_ledger().get("address_bytes", 0)),
        "data_rows_touched": 0,
        "refit": False,
    }

# ---------------------------------------------------------------------------
# 7. THE OBJECT IS THE SUBSPACE (hub design call, Head-cleared, 2026-09-20)
# ---------------------------------------------------------------------------


def deployed_span(G, tol: float = 1e-8) -> np.ndarray:
    """An orthonormal basis of what the atom block actually spans, in float64.

    ⛔ This is the object the read sees: ``gen_read.render_span`` reads through a
    basis of ``span(G)``, so ``span(G)`` -- not the particular rows written into ``G``
    -- is what the store deploys. Zero rows are dropped because they span nothing.
    """
    M = np.asarray(G, np.float64)
    live = M[np.linalg.norm(M, axis=1) > tol]
    if live.size == 0:
        return np.zeros((M.shape[1], 0), np.float64)
    # ⚠ SVD, NOT QR. Un-pivoted Householder QR is not rank revealing: on a block
    # whose mean lies nearly inside the directions' span (measured: relative
    # perpendicular norm 2.3e-08 in the rank-deficient wells) the small entry of
    # diag(R) is not in a known position, so truncating to the FIRST r columns of Q
    # discards an arbitrary direction. MEASURED consequence of getting this wrong:
    # two BIT-IDENTICAL stores reported a principal angle of 0.97 rad. The byte-level
    # contrast printed beside it is what exposed that -- an absurd number next to a
    # known-zero one.
    U, sv, _ = np.linalg.svd(live.T, full_matrices=False)
    r = int((sv > tol * float(sv[0])).sum()) if sv.size else 0
    return np.ascontiguousarray(U[:, :r])


def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """``||P_A - P_B||_2`` -- the projector distance, which is ``sin(theta_max)``.

    ⚠ NOT ``arccos`` of the singular values of ``A^T B``, which is the textbook
    formula and is useless here: ``arccos`` is ill-conditioned at 1, with an error
    floor of ``sqrt(eps) ~ 1.5e-08`` rad. MEASURED: comparing a store with ITSELF --
    where the answer is exactly 0 -- that route returned 4.47e-08 rad, which is the
    same order as the real signal on the passing legs and would have made them
    indistinguishable from noise. The projector difference is exact at 0 and stable
    everywhere.
    """
    if A.shape[1] == 0 and B.shape[1] == 0:
        return 0.0
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 1.0
    return float(np.linalg.norm(A @ A.T - B @ B.T, 2))


def compare_subspaces(a, b) -> dict:
    """⭐⭐ A1 UNDER THE SUBSPACE DEFINITION: is the deployed SPAN the same span?

    ⛔ The hub's design call, Head-cleared: the stored object is the **subspace**, not
    the particular eigenvector basis written into the atom rows. Inside a degenerate or
    near-degenerate eigenspace the individual eigenvectors are not a function of the
    data -- two runs of the same computation disagree about them -- but the SPAN they
    sweep may still be perfectly determined. This measures the span.

    ⚠ It is NOT a softer metric chosen to make a failure go away, and it must not be
    read as one: a rotation *inside* the kept set preserves the span and this reports
    ~0, but mixing between a KEPT direction and a DROPPED one moves the span and this
    reports it at full size. If the rank boundary itself is unstable, the subspace
    definition does not rescue anything and this number says so.

    Reported per well: the largest principal angle between the deployed spans, its sine
    (= the 2-norm of the difference of the orthogonal projectors), the same for the
    directions alone with the mean excluded, and the mean's own deviation -- the mean is
    ``t/n``, a determined quantity, and is compared entrywise rather than as a span.
    """
    if a.K != b.K:
        raise AssertionError(f"{a.K} wells vs {b.K}: not the same store shape")
    f32 = float(np.spacing(np.float32(1.0)))
    per_well, ok = [], True
    for k in range(a.K):
        GA = np.asarray(a.modes[k], np.float32)
        GB = np.asarray(b.modes[k], np.float32)
        QA, QB = deployed_span(GA), deployed_span(GB)
        pd = subspace_distance(QA, QB)
        max_ang = float(np.arcsin(min(pd, 1.0)))
        # directions only, mean excluded
        DA, DB = deployed_span(GA[1:]), deployed_span(GB[1:])
        pdd = subspace_distance(DA, DB)
        max_angd = float(np.arcsin(min(pdd, 1.0)))
        mu_dev = float(np.max(np.abs(GA[0].astype(np.float64)
                                     - GB[0].astype(np.float64))))
        mu_norm = float(np.linalg.norm(GB[0].astype(np.float64)))
        row = {
            "well": k, "d_k": int(a.well_sizes[k]),
            "span_rank_a": int(QA.shape[1]), "span_rank_b": int(QB.shape[1]),
            "span_ranks_agree": bool(QA.shape[1] == QB.shape[1]),
            "max_principal_angle_rad": max_ang,
            "projector_2norm": pd,
            "max_principal_angle_in_float32_ulp": max_ang / f32,
            "max_principal_angle_directions_only_rad": max_angd,
            "mean_max_abs_dev": mu_dev,
            "mean_norm": mu_norm,
            "mean_rel_dev": mu_dev / max(mu_norm, 1e-300),
            "subspace_equal_below_one_float32_ulp": bool(max_ang <= f32),
        }
        ok = ok and row["span_ranks_agree"]
        per_well.append(row)
    angs = [r["max_principal_angle_rad"] for r in per_well]
    mus = [r["mean_max_abs_dev"] for r in per_well]
    return {
        "definition": ("the stored object is the SUBSPACE the atom block spans, plus "
                       "the mean; the particular eigenvector basis is not the object"),
        "float32_ulp_at_unit_scale": f32,
        "max_principal_angle_rad_per_well": angs,
        "worst_max_principal_angle_rad": float(max(angs)) if angs else 0.0,
        "worst_in_float32_ulp": (float(max(angs)) / f32) if angs else 0.0,
        "mean_max_abs_dev_per_well": mus,
        "worst_mean_max_abs_dev": float(max(mus)) if mus else 0.0,
        "span_ranks_agree_every_well": bool(ok),
        "n_wells_span_equal_below_one_ulp": int(
            sum(r["subspace_equal_below_one_float32_ulp"] for r in per_well)),
        "all_wells_span_equal_below_one_ulp": bool(
            all(r["subspace_equal_below_one_float32_ulp"] for r in per_well)),
        "per_well": per_well,
    }
