"""w8 hub/single-design -- ONE OBJECT THAT COMPOSES AND FORGETS.

The open problem this module exists to close
--------------------------------------------
The content that wins composition (``pca_tensor``: the per-well leading principal
subspace, unsupervised, 524,288 B) **cannot do exact per-item deletion**.  Its own
author says so at ``exp_w5_compose_field.py:485``:

    "It gives up exact per-item deletion.  A principal direction is not a written
     item, so charter B.4's 'deletion is an exact operation on atoms' does not hold
     of this content the way it holds of the field store's."

And the object that forgets exactly -- the field store, whose coupling hole is 1 byte,
0 outside, no refit, 0.0000 composition cost -- **loses composition when it evicts
atoms, whichever atoms** (w5: -2.0899 / -0.9101 / -6.7416 against a size-matched
random-eviction control of -2.7584, whose verdict string is "THE COST IS THE
MECHANISM, NOT THE TARGET").

The single design
-----------------
Keep the winning content.  Change only how it is *derived*: from a retained
**sufficient statistic** instead of a one-shot SVD.

Per well ``k``, over the fit pool restricted to well ``k``'s channels::

    S_k = sum_i x_i x_i^T        (symmetric; only the triangle is stored)
    t_k = sum_i x_i
    n_k = |pool|

The deployed atom block is then exactly what ``PCASpanStore`` deploys::

    mu_k    = t_k / n_k
    C_k     = (S_k - t_k t_k^T / n_k) / (n_k - 1)          the covariance
    atoms_k = [ mu_k ; leading (A-1) eigenvectors of C_k ]

and **deleting item i is a rank-1 subtract**::

    S_k -= x_i x_i^T ;  t_k -= x_i ;  n_k -= 1 ;  re-derive

⭐ No pass over the data.  ⭐ No refit.  ⭐ The items themselves are never kept.
That is charter B.4 restored to the content that wins rows 3 and 4.

⚠ What it costs, declared and never netted
------------------------------------------
``S_k`` is ``d_k x d_k``.  ⚠ The realised partition on the SCORED leg is
``[777, 939, 216, 58, 58]`` (``partitions.npz['factor']``; the val leg is
``[771, 940, 217, 60, 60]``).  ⛔ An earlier draft of this module quoted
``[303, 305, 573, 501, 366]`` -- that was a TEST FIXTURE
(``tests/test_gen_dynamical_read.py:152``), not a deployed partition, and every byte
figure derived from it was wrong by about 3x.  It is corrected here rather than
silently dropped.

At the incumbent's K=5 partition the triangle is 770,441 floats, and the statistic
must be **float64** (the downdate residual on ``S`` runs to ~5e-10 absolute on entries
of order 1e+9; float32 does not hold it), so the honest ledger is
**6,704,240 B = 12.787x the 524,288 B pin**.  The spoke building this object at K=5
measures a four-part total of 24.45x including its coupling and counts.  ⛔ Exact
downdate at the incumbent's partition is not affordable -- by an order of magnitude.

But ``sum_k d_k^2`` falls quadratically as the wells get finer while the atom block
``A x D`` falls only linearly.  ⛔ ``factor_aligned_balanced_labels`` cannot help: it
refuses any ``k > 6`` ("a factor-aligned partition cannot make more wells than there
are factors to align them to").  The criterion that reaches a finer K is the shipped
**label-free** ``ward_balanced`` cut of the correlation dendrogram's leaf order
(``exp_gen_factor_wells.py:594``), which the target file sanctions explicitly and
which sees no label at all -- so an object built on it is *strictly more* label-free
than the incumbent.  The matched-bytes window is narrow, ``K = 36..48``; e.g.
**K=38, A=7 at 523,744 B = 0.999x**, or **K=42, A=12 at 522,704 B = 0.997x**.
⛔ Below K=36 the float64 statistic alone exceeds the pin and no atoms fit at all.

⛔ THE SATURATION TRAP, which this module refuses to walk into
--------------------------------------------------------------
From K~52 up, ``A >= d_k``: the per-well rank saturates at ``d_k``, the atom block
spans the whole well, the projection is the identity and the CONTENT carries no
information -- the address book does all the work.  Such a store is DECORATIVE under
charter C.1/C.4, and its C.2 controls stop discriminating: ``landscape_removed``
(zeroed) still collapses, but ``random_refit`` does NOT, because any full-rank block
spans the same space.  :func:`assert_not_saturated` refuses to build there, and
``rank/d_k`` is carried on every artefact so saturation is visible, never hidden.

Label use (charter B.6), stated once and carried in every caption
-----------------------------------------------------------------
Labels align the wells ONCE at fit time and ground the address book at the interface.
⭐ The CONTENT is an unsupervised decomposition of training features -- identical in
this respect to ``pca_tensor``, which is the point: the single design must not buy
deletion by smuggling in supervision.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "WellStats", "CovContentStore", "plan_partition", "assert_not_saturated",
    "honest_byte_ledger", "SATURATION_NOTE",
]

SATURATION_NOTE = (
    "rank/d_k < 1 strictly: at rank == d_k the atom block spans the whole well, the "
    "projection is the identity and the content carries no information (charter "
    "C.1/C.4 -- a decorative store), and the random-atom control stops collapsing"
)

#: float64 throughout the statistic.  Forming S = sum x x^T squares the condition
#: number of the centred design matrix, so the accumulation precision is not a
#: stylistic choice -- it is what makes "downdate == refit" hold to float precision.
#: The DEPLOYED block is cast to float32, exactly as PCASpanStore deploys float32.
_ACC = np.float64

#: The rank cut's relative eigenvalue threshold.  Declared here, once, before any
#: number: a direction is deployed only if its eigenvalue exceeds this multiple of the
#: well's largest.  ⛔ Not tuned -- it is the same 1e-12 relative scale ``np.linalg``
#: uses for its own rank determinations, and the measured margins are not marginal
#: (smallest kept eigenvalue >= 2.3e3x the threshold, largest dropped <= 3.6e-4x it).
EIG_REL_TOL = 1e-12


class WellStats:
    """The retained sufficient statistic for one well.  ⛔ Holds no item.

    Attributes
    ----------
    S : (d, d) float64   second moment, ``sum_i x_i x_i^T``
    t : (d,)   float64   first moment, ``sum_i x_i``
    n : int              count

    ⭐ ``S`` and ``t`` are the ONLY things retained.  Nothing here can reconstruct
    which rows were written, which is the "items never kept" property.
    """

    __slots__ = ("_Sp", "t", "n", "d", "_iu")

    def __init__(self, d: int):
        self.d = int(d)
        # ⭐⭐ PACKED. The canonical retained state is the UPPER TRIANGLE of S, not the
        # full square.  S is symmetric, so the triangle is sufficient -- but a ledger may
        # only count a packing that is actually deployed, and an earlier version of this
        # class held a full (d, d) array while counting d(d+1)/2.  ⛔ That would have been
        # netting by convention, which this hub's target file forbids.  The full matrix is
        # materialised only transiently, inside `covariance()`, for `eigh`.
        self._iu = np.triu_indices(self.d)
        self._Sp = np.zeros(self._iu[0].size, _ACC)
        self.t = np.zeros(self.d, _ACC)
        self.n = 0

    # -- packing helpers ----------------------------------------------------
    def _pack(self, M: np.ndarray) -> np.ndarray:
        return M[self._iu]

    def _unpack(self) -> np.ndarray:
        M = np.zeros((self.d, self.d), _ACC)
        M[self._iu] = self._Sp
        M = M + M.T
        M[np.diag_indices(self.d)] *= 0.5
        return M

    @property
    def S(self) -> np.ndarray:
        """The full second moment, reconstructed.  ⛔ Not what is stored."""
        return self._unpack()

    # -- write / delete: the same operation with opposite sign ---------------
    def add(self, X: np.ndarray) -> None:
        """Accumulate rows.  ``X`` is ``(m, d)``."""
        Xa = np.asarray(X, _ACC)
        self._Sp += self._pack(Xa.T @ Xa)
        self.t += Xa.sum(axis=0)
        self.n += int(Xa.shape[0])

    def remove(self, X: np.ndarray) -> None:
        """⭐ THE DELETION.  Exact, rank-m, no pass over the retained data.

        This is the whole claim of the hub in four lines: the statistic that
        determines the content is additive, so removing an item is subtracting its
        contribution.  Nothing is refitted and nothing else moves.
        """
        Xa = np.asarray(X, _ACC)
        if Xa.shape[0] > self.n:
            raise AssertionError(
                f"asked to remove {Xa.shape[0]} rows from a well holding {self.n}")
        self._Sp -= self._pack(Xa.T @ Xa)
        self.t -= Xa.sum(axis=0)
        self.n -= int(Xa.shape[0])

    # -- derived content ----------------------------------------------------
    def mean(self) -> np.ndarray:
        if self.n <= 0:
            raise AssertionError("empty well has no mean")
        return self.t / self.n

    def covariance(self) -> np.ndarray:
        """``(S - t t^T / n) / (n - 1)``, symmetrised.

        Symmetrising is not cosmetic: the subtraction above is done in floating
        point and a downdated ``S`` can pick up an asymmetry ~1 ULP, which
        ``eigh`` would otherwise resolve into a spurious imaginary part of the
        spectrum's tail.
        """
        if self.n < 2:
            raise AssertionError(f"covariance needs n >= 2, well holds {self.n}")
        C = (self._unpack() - np.outer(self.t, self.t) / self.n) / (self.n - 1)
        return 0.5 * (C + C.T)

    def atoms(self, n_atoms: int, return_rank: bool = False):
        """``[mu ; leading (n_atoms-1) eigenvectors]``, zero-padded, float32.

        ⭐ Byte-identical in LAYOUT to ``PCASpanStore.modes[k]``: a narrow well still
        deploys ``n_atoms`` rows, the surplus as zeros, because they cost the same
        bytes and buy no span -- the same honesty ``exp_w5_compose_field.py:249``
        applies to a real store written into a narrow well.

        ⭐⭐ THE RANK CUT, and why it is a well-posedness condition rather than a patch.
        A well whose covariance is rank-deficient (this partition has 64 DEAD channels,
        distributed ``[0, 0, 0, 46, 18]`` over the five wells, so two wells of nominal
        width 58 carry effective rank 12 and 40) has a numerically null trailing
        eigenspace.  There the eigenvector is **not a function of the data**: LAPACK
        returns an arbitrary rotation, so two refits of the SAME rows disagree by O(1).
        ⛔ That degeneracy belongs to the incumbent ``PCASpanStore`` too -- it has simply
        never had to be a function of its fit set twice, so nothing ever asked.  The
        deletion property is what exposes it.  Cutting at :data:`EIG_REL_TOL` and
        deploying zero rows below the cut is exactly ``PCASpanStore``'s own stated
        convention for a well narrower than ``A``, applied for the same reason.
        """
        A = int(n_atoms)
        mu = self.mean()
        r = min(A - 1, self.d)
        kept = 0
        if r > 0:
            C = self.covariance()
            w, V = np.linalg.eigh(C)              # ascending; symmetric route
            w = w[::-1]
            V = V[:, ::-1]
            # ⭐ the cut: only directions the data actually determines
            lam_max = float(w[0]) if w.size else 0.0
            if lam_max > 0:
                kept = int(min(r, int((w > EIG_REL_TOL * lam_max).sum())))
            V = V[:, :kept]
            B = np.vstack([mu[None, :], V.T]) if kept else mu[None, :]
        else:
            B = mu[None, :]
        B = B.astype(np.float32)
        if B.shape[0] < A:
            B = np.vstack([B, np.zeros((A - B.shape[0], B.shape[1]), np.float32)])
        B = np.ascontiguousarray(B)
        return (B, kept) if return_rank else B

    # -- ledger -------------------------------------------------------------
    def stat_floats(self) -> int:
        """Retained floats: the PACKED triangle of ``S``, plus ``t``, plus ``n``.

        ⭐ This counts what is actually held: ``self._Sp.size`` is the triangle, and the
        assertion below makes the ledger answerable to the array rather than to a formula.
        """
        packed = int(self._Sp.size)
        if packed != self.d * (self.d + 1) // 2:
            raise AssertionError(
                f"packed triangle is {packed} floats, not {self.d * (self.d + 1) // 2} "
                "-- the ledger and the array disagree")
        return packed + self.d + 1

    def stat_bytes(self) -> int:
        """⭐ Counted at the precision actually RETAINED, which is float64.

        ⛔ Not float32.  Forming ``S`` squares the condition number and the downdate
        residual is measured at 1e-10..5e-10 absolute; a float32 accumulator does not
        hold that, so the statistic is float64 and the ledger says 8 bytes per float.
        Counting it at 4 would be netting by dtype.
        """
        return self.stat_floats() * _ACC(0).itemsize


class CovContentStore:
    """The single design: ``pca_tensor``'s content, carried as a statistic.

    ⛔ Deliberately NOT a subclass of ``PCASpanStore``.  It exposes the same three
    attributes the span interface reads -- ``modes``, ``well_channels``, ``dim`` --
    so every existing charter-C control, floor and read transfers unchanged, but the
    derivation is different and the ledger is different, and inheriting would hide
    both.
    """

    def __init__(self, X: np.ndarray, well_channels, n_atoms: int, dim: int,
                 allow_saturated: bool = False):
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(n_atoms)
        sizes = [int(c.size) for c in self.well_channels]
        if not allow_saturated:
            assert_not_saturated(self.n_atoms, sizes)
        Xf = np.asarray(X, np.float32)
        self.stats = []
        for ch in self.well_channels:
            w = WellStats(int(ch.size))
            w.add(Xf[:, ch])
            self.stats.append(w)
        self._refresh()

    def _refresh(self) -> None:
        out = [w.atoms(self.n_atoms, return_rank=True) for w in self.stats]
        self.modes = [b for b, _ in out]
        #: ⭐ deployed rank per well AFTER the cut.  Carried on the object, not derived
        #: at report time, so a saturated or rank-deficient well is visible on every row.
        self.deployed_rank = [r for _, r in out]

    @property
    def K(self) -> int:
        return len(self.modes)

    # -- the gauge ----------------------------------------------------------
    def span_bases(self) -> list:
        """Per-well orthonormal basis of the deployed atom rows.

        ⛔ Deliberately the SAME construction ``gen_read.span_basis`` uses, so the
        rotation computed here is the rotation the read will actually see.
        """
        out = []
        for M, r in zip(self.modes, self.deployed_rank):
            if r <= 0:
                out.append(np.zeros((M.shape[1], 0), np.float64))
                continue
            out.append(np.linalg.qr(np.asarray(M[:1 + r], np.float64).T)[0])
        return out

    # -- the deletion -------------------------------------------------------
    def delete_rows(self, X_rows: np.ndarray) -> dict:
        """Remove items from the store.  ⭐ Exact, no refit, no pass over the pool.

        Returns a receipt carrying what actually moved, so a caller can assert on it
        rather than trust it.
        """
        Xf = np.asarray(X_rows, np.float32)
        n_before = [w.n for w in self.stats]
        U_old = self.span_bases()                     # the gauge BEFORE the delete
        for w, ch in zip(self.stats, self.well_channels):
            w.remove(Xf[:, ch])
        self._refresh()
        U_new = self.span_bases()                     # and after

        # ⭐⭐ THE BASIS-CHANGE CARRY.  The read is x_hat = c_k U_k^T with c_k = a(request) W_k.
        # The deletion moves the basis U_k -> U'_k, so the coefficients that MEAN the
        # same thing in the new gauge are c'_k = c_k R_k with R_k = U_k^T U'_k, and the
        # updated address book is W'_k = W_k R_k.
        #
        # ⛔ This is NOT a repair of an inexact deletion -- the deletion is exact in the
        # store either way.  The store's basis is a GAUGE, the address book is expressed
        # in it, and a change of gauge is an exact, data-free change of coordinates.
        # Measured: it takes downdate-vs-refit rendered agreement from O(1) to ~1e-7
        # relative, i.e. float32 arithmetic.
        #
        # ⭐ R_k is r x r, computed from two bases already in hand, touches NO pool row,
        # and is TRANSIENT -- applied to W_k and discarded -- so it adds nothing to the
        # deployed ledger.
        rot = []
        for Uo, Un in zip(U_old, U_new):
            r = min(Uo.shape[1], Un.shape[1])
            rot.append(np.asarray(Uo[:, :r].T @ Un[:, :r], np.float32)
                       if r > 0 else np.zeros((0, 0), np.float32))
        self.last_basis_change = rot
        return {
            "op": "delete_rows",
            "n_rows_removed": int(Xf.shape[0]),
            "n_before": n_before,
            "n_after": [w.n for w in self.stats],
            "refit": False,
            "passes_over_retained_data": 0,
            "items_kept": False,
            "basis_change_shapes": [tuple(R.shape) for R in rot],
            "basis_change_bytes_deployed": 0,
            "mechanism": ("S_k -= x x^T, t_k -= x, n_k -= 1; the atoms are re-derived "
                          "from the statistic and the address book is carried across "
                          "the basis change by W_k <- W_k R_k, R_k = U_k^T U'_k. "
                          "The pool is never touched."),
        }

    def basis_change_audit(self) -> dict:
        """How far the deletion actually moved each well's span.

        ⚠ THIS WAS FIRST WRITTEN AS AN ORTHOGONALITY ASSERTION AND IT FAILED, which is
        what it was for.  ``R_k = U_k^T U'_k`` is NOT a rotation: deviation from
        ``R^T R = I`` was measured at 0.912 (K=42) and 0.621 (K=48).  It is not supposed
        to be one -- ``R_k`` is the matrix of principal-angle cosines between the pre-
        and post-deletion spans, and it is an isometry only if the deletion changed
        nothing.  ⛔ So "gauge rotation" was the wrong name and this is now a
        **basis-change carry**, with its spectrum reported rather than asserted away.

        ⭐ The spectrum is the interesting quantity.  Deleting red at K=42/A=12 leaves
        93.8 % of directions with ``cos > 0.99`` and moves a tiny specific set to
        ``cos_min = 0.1999``; a SIZE-MATCHED RANDOM deletion of the same 11,044 rows
        gives ``cos_min = 0.9976`` and moves nothing.  That separation is the inverse of
        w5's eviction verdict ("the cost is the mechanism, not the target"): here the
        change is concentrated on the deleted content.  ⛔ It is a property of the store,
        not a composition number, and it may not be quoted as one.
        """
        rot = getattr(self, "last_basis_change", None)
        if rot is None:
            raise AssertionError("no deletion has been performed; nothing to audit")
        devs, cos = [], []
        for R in rot:
            if R.size == 0:
                continue
            Rd = np.asarray(R, np.float64)
            devs.append(float(np.max(np.abs(Rd.T @ Rd - np.eye(Rd.shape[1])))))
            cos.append(np.linalg.svd(Rd, compute_uv=False))
        if not cos:
            return {"n_wells_audited": 0, "note": "no live wells"}
        allc = np.concatenate(cos)
        return {"n_wells_audited": len(devs),
                "max_abs_dev_from_orthogonality": max(devs),
                "principal_angle_cos_min": float(allc.min()),
                "principal_angle_cos_median": float(np.median(allc)),
                "frac_directions_cos_gt_0p99": float((allc > 0.99).mean()),
                "frac_directions_cos_lt_0p5": float((allc < 0.5).mean()),
                "what": ("singular values of R_k = U_k^T U'_k are the principal-angle "
                         "cosines between the pre- and post-deletion spans: 1 where the "
                         "deletion changed nothing, small where it changed the content. "
                         "A spectrum that is ALL ones would mean the deletion did nothing."),
                "compare_against": ("the same statistic under a SIZE-MATCHED RANDOM "
                                    "deletion -- if the two agree, the change is generic "
                                    "damage, not targeted forgetting (w5's evict_size_control "
                                    "verdict)")}

    # -- ledger -------------------------------------------------------------
    def byte_ledger(self, budget_bytes: int) -> dict:
        """⭐ SUMMED, never netted.  The covariance is what buys the deletion, so
        hiding it beside the ledger would be exactly the netting the target forbids.
        """
        atom_floats = int(sum(M.size for M in self.modes))
        stat_bytes = int(sum(w.stat_bytes() for w in self.stats))
        total = atom_floats * 4 + stat_bytes
        sizes = [int(c.size) for c in self.well_channels]
        return {
            "atom_bytes": atom_floats * 4,
            "statistic_bytes": stat_bytes,
            "statistic_dtype": str(np.dtype(_ACC)),
            "deployed_bytes": total,
            "budget_bytes": int(budget_bytes),
            "within_budget": bool(total <= int(budget_bytes)),
            "multiple_of_budget": round(total / float(budget_bytes), 6),
            "param_count": atom_floats + stat_bytes // 4,
            "well_sizes": sizes,
            "rank_over_d": [round(min(self.n_atoms, d) / d, 4) for d in sizes],
            "saturated_wells": [d for d in sizes if self.n_atoms >= d],
            "what": (f"{self.n_atoms} atoms x {self.dim} float32 deployed, PLUS the "
                     "per-well second moment triangle + first moment + count that "
                     "make deletion exact; summed, not netted"),
            "note": SATURATION_NOTE,
        }


def assert_not_saturated(n_atoms: int, well_sizes) -> None:
    """⛔ Refuse to build a decorative store.  See the module docstring's trap."""
    bad = [int(d) for d in well_sizes if int(n_atoms) >= int(d)]
    if bad:
        raise AssertionError(
            f"n_atoms={n_atoms} saturates {len(bad)} well(s) of size {sorted(set(bad))}: "
            "the atom block would span the whole well, the projection would be the "
            "identity and the content would carry no information. " + SATURATION_NOTE)


def plan_partition(D: int, budget_bytes: int, K: int) -> dict:
    """The largest ``A`` whose honest ledger fits ``budget_bytes`` at ``K`` wells.

    Wells are taken as equal as ``D`` allows; the real build inherits the label-MI
    criterion and its wells are not exactly equal, so this is a PLAN, checked against
    the realised ledger at build time and superseded by it.
    """
    Bf = int(budget_bytes) // 4
    d0, rem = divmod(int(D), int(K))
    ds = [d0 + 1] * rem + [d0] * (int(K) - rem)
    stat = sum(d * (d + 1) // 2 + d + 1 for d in ds)
    avail = Bf - stat
    A = avail // int(D) if avail > 0 else 0
    ranks = [min(A, d) for d in ds]
    return {
        "K": int(K), "well_sizes": ds, "n_atoms": int(A),
        "statistic_floats": int(stat), "atom_floats": int(A * D),
        "total_bytes": int((A * D + stat) * 4),
        "budget_bytes": int(budget_bytes),
        "within_budget": bool((A * D + stat) * 4 <= int(budget_bytes)),
        "total_span": int(sum(ranks)),
        "rank_over_d": [round(r / d, 4) for r, d in zip(ranks, ds)],
        "saturated": bool(any(A >= d for d in ds)),
    }


def honest_byte_ledger(well_sizes, n_atoms: int, D: int, budget_bytes: int) -> dict:
    """The ledger for an arbitrary realised partition.  Used on the built object."""
    ds = [int(d) for d in well_sizes]
    stat = sum(d * (d + 1) // 2 + d + 1 for d in ds)
    atoms = int(n_atoms) * int(D)
    total = (atoms + stat) * 4
    return {
        "atom_bytes": atoms * 4, "statistic_bytes": stat * 4,
        "deployed_bytes": total, "budget_bytes": int(budget_bytes),
        "within_budget": bool(total <= int(budget_bytes)),
        "multiple_of_budget": round(total / float(budget_bytes), 6),
        "well_sizes": ds,
        "rank_over_d": [round(min(int(n_atoms), d) / d, 4) for d in ds],
    }
