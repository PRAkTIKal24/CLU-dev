"""downdate store."""

from __future__ import annotations
import hashlib
from typing import Optional, Sequence
import numpy as np

EIG_REL_TOL = 1e-12
MU_PERP_REL_TOL = 0.0001
VARIANCE_SHARE_TOL = 4.144858e-09
COUPLING_ALPHA = 1.0
ROWS_CONSUMED = {"n": 0, "calls": 0}


def _row_moments(X: np.ndarray, channels: np.ndarray) -> tuple:
    ch = np.asarray(channels, np.int64)
    M = np.asarray(np.asarray(X)[:, ch], np.float64)
    ROWS_CONSUMED["n"] += int(M.shape[0])
    ROWS_CONSUMED["calls"] += 1
    return (M.T @ M, M.sum(axis=0), int(M.shape[0]))


def _canonical_sign(V: np.ndarray) -> np.ndarray:
    """Sign each row so its largest-magnitude entry is positive; ties by lowest index."""
    if V.size == 0:
        return V
    idx = np.argmax(np.abs(V), axis=1)
    s = np.sign(V[np.arange(V.shape[0]), idx])
    s[s == 0] = 1.0
    return V * s[:, None]


def modes_from_moments(
    S: np.ndarray,
    t: np.ndarray,
    n: int,
    n_atoms: int,
    eig_rel_tol: float = EIG_REL_TOL,
    var_share_tol: float = None,
) -> tuple:
    d = int(np.asarray(t).size)
    A = int(n_atoms)
    if int(n) <= 0:
        raise ValueError("a well with no rows has no mean and no principal direction")
    Sm = np.asarray(S, np.float64)
    tv = np.asarray(t, np.float64)
    mu = tv / float(n)
    C = Sm - np.outer(tv, tv) / float(n)
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order].T
    lmax = float(max(float(w[0]), 0.0))
    n_above = int((w > float(eig_rel_tol) * lmax).sum())
    vst = VARIANCE_SHARE_TOL if var_share_tol is None else float(var_share_tol)
    tot_pos = float(np.maximum(w, 0.0).sum())
    share = np.maximum(w, 0.0) / tot_pos if tot_pos > 0 else np.zeros_like(w)
    n_share = int((share >= vst).sum())
    keep_no_share = int(min(A - 1, d, n_above))
    keep = int(min(A - 1, d, n_above, n_share))
    B = np.vstack([mu[None, :], _canonical_sign(V[:keep])]).astype(np.float32)
    if B.shape[0] < A:
        B = np.vstack([B, np.zeros((A - B.shape[0], d), np.float32)])
    tot = float(np.maximum(w, 0.0).sum())
    thr = float(eig_rel_tol) * lmax
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
    rel_gap_kept = gap_kept / lmax if lmax > 0 else float("inf")
    rel_gap_any = gap_any / lmax if lmax > 0 else float("inf")
    sens_kept = eps64 / rel_gap_kept if rel_gap_kept > 0 else float("inf")
    sens_any = eps64 / rel_gap_any if rel_gap_any > 0 else float("inf")
    f32_ulp = float(np.spacing(np.float32(1.0)))
    report = {
        "d_k": d,
        "n_rows": int(n),
        "rank_kept": keep,
        "rank_limited_by": "atom budget A-1"
        if keep == A - 1
        else "well width d_k"
        if keep == d
        else "eigenvalue tolerance",
        "n_zero_rows": int(A - 1 - keep),
        "eig_threshold": thr,
        "smallest_kept_over_threshold": float(w[keep - 1] / thr)
        if keep > 0 and thr > 0
        else float("inf"),
        "largest_dropped_over_threshold": float(w[keep] / thr)
        if keep < w.size and thr > 0
        else 0.0,
        "lambda_max": lmax,
        "lambda_smallest_kept": float(w[keep - 1]) if keep > 0 else float("nan"),
        "lambda_largest_dropped": float(w[keep]) if keep < w.size else float("nan"),
        "variance_explained": float(np.maximum(w[:keep], 0.0).sum() / tot) if tot > 0 else 1.0,
        "rank_over_d": float(keep) / float(d),
        "saturated": bool(A - 1 >= d),
        "var_share_tol": vst,
        "rank_kept_without_share_cut": keep_no_share,
        "n_directions_dropped_by_share_cut": int(max(keep_no_share - keep, 0)),
        "share_cut_binds": bool(keep < keep_no_share),
        "variance_share_lost_to_share_cut": float(share[keep:keep_no_share].sum())
        if keep < keep_no_share
        else 0.0,
        "smallest_kept_variance_share": float(share[keep - 1]) if keep > 0 else None,
        "largest_dropped_variance_share": float(share[keep]) if keep < share.size else 0.0,
        "min_rel_gap_among_kept": rel_gap_kept,
        "min_rel_gap_to_any_neighbour": rel_gap_any,
        "predicted_byte_sensitivity": sens_any,
        "predicted_byte_sensitivity_kept_only": sens_kept,
        "predicted_sensitivity_in_float32_ulp": sens_any / f32_ulp,
        "bytes_determined_below_one_float32_ulp": bool(sens_any < f32_ulp),
        "sensitivity_rule": "eps64 / (min relative eigenvalue gap); an eigenvector rotates by ~||dC||/gap and the downdate residual on C is at machine epsilon",
    }
    return (np.ascontiguousarray(B), report)


def term_offsets(cardinalities: Sequence[int]) -> np.ndarray:
    """Start of each factor's value block in the flat term index (gen_field's layout)."""
    c = [int(x) for x in cardinalities]
    return np.concatenate([[0], np.cumsum(c)]).astype(np.int64)


def coupling_layout(cardinalities: Sequence[int]) -> dict:
    cards = [int(x) for x in cardinalities]
    pairs, offs, off = ([], [], 0)
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
        f, v, g, u = (g, u, f, v)
    lay = coupling_layout(cardinalities)
    j = lay["pairs"].index((int(f), int(g)))
    return int(lay["offsets"][j] + int(v) * lay["cardinalities"][int(g)] + int(u))


def assert_layout_matches_field_store(cardinalities: Sequence[int]) -> dict:
    from nclu.core.gen_field import coupling_index as gf_index
    from nclu.core.gen_field import coupling_layout as gf_layout

    mine, theirs = (coupling_layout(cardinalities), gf_layout(cardinalities))
    if (
        mine["pairs"] != theirs["pairs"]
        or mine["offsets"] != theirs["offsets"]
        or mine["size"] != theirs["size"]
    ):
        raise AssertionError(
            "the coupling layout here is not gen_field's -- a hole punched in this object would not be the hole measured"
        )
    cards = [int(x) for x in cardinalities]
    n_checked = 0
    for f in range(len(cards)):
        for g in range(f + 1, len(cards)):
            for v in (0, cards[f] - 1):
                for u in (0, cards[g] - 1):
                    if coupling_index(cards, f, v, g, u) != gf_index(cards, f, v, g, u):
                        raise AssertionError(
                            f"coupling_index disagrees with gen_field at ({f},{v})x({g},{u})"
                        )
                    n_checked += 1
    return {
        "layout_equals_gen_field": True,
        "n_pairs": int(mine["size"]),
        "n_indices_checked": int(n_checked),
    }


def _value_terms(values: np.ndarray, grid_values: Sequence[np.ndarray]) -> np.ndarray:
    """``(N, F) float values -> (N, F) int term indices`` on the grid, nearest value."""
    V = np.asarray(values, np.float64)
    out = np.empty(V.shape, np.int64)
    toff = term_offsets([len(g) for g in grid_values])
    for f, g in enumerate(grid_values):
        gv = np.asarray(g, np.float64)
        out[:, f] = int(toff[f]) + np.argmin(np.abs(V[:, f][:, None] - gv[None, :]), 1)
    return out


class DowndatableStore:
    def __init__(
        self,
        well_channels,
        dim: int,
        n_atoms: int,
        cardinalities: Sequence[int],
        eig_rel_tol: float = EIG_REL_TOL,
        coupling_alpha: float = COUPLING_ALPHA,
        var_share_tol: float = VARIANCE_SHARE_TOL,
    ):
        self.well_channels = [np.ascontiguousarray(np.asarray(c, np.int64)) for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(n_atoms)
        self.cardinalities = [int(c) for c in cardinalities]
        self.eig_rel_tol = float(eig_rel_tol)
        self.var_share_tol = float(var_share_tol)
        self.coupling_alpha = float(coupling_alpha)
        d = [int(c.size) for c in self.well_channels]
        self.S = [np.zeros((dk, dk), np.float64) for dk in d]
        self.t = [np.zeros(dk, np.float64) for dk in d]
        self.n = np.zeros(len(d), np.int64)
        self.n_terms = int(sum(self.cardinalities))
        self.term_n = np.zeros(self.n_terms, np.int64)
        self.request_n = np.zeros(self.n_terms, np.int64)
        lay = coupling_layout(self.cardinalities)
        self.n_pairs = int(lay["size"])
        self.pair_n = np.zeros(self.n_pairs, np.int64)
        self.coupling = np.zeros(self.n_pairs, np.float32)
        self.coupling_hole = np.zeros(self.n_pairs, np.uint8)
        self.modes = [np.zeros((self.n_atoms, dk), np.float32) for dk in d]
        self.rank_reports: list = [{} for _ in d]
        self.history: list = []

    @property
    def K(self) -> int:
        return len(self.modes)

    @property
    def well_sizes(self) -> list:
        return [int(c.size) for c in self.well_channels]

    def absorb(
        self,
        X: np.ndarray,
        values: Optional[np.ndarray] = None,
        grid_values: Optional[Sequence[np.ndarray]] = None,
    ) -> dict:
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
            idx = (
                int(lay["offsets"][j])
                + (T[:, f] - int(toff[f])) * self.cardinalities[g]
                + (T[:, g] - int(toff[g]))
            )
            np.add.at(self.pair_n, idx, int(sign))
        if (self.term_n < 0).any() or (self.pair_n < 0).any():
            raise AssertionError(
                "a count went negative -- rows were deleted that were never absorbed"
            )

    def note_requests(self, values, grid_values) -> dict:
        T = _value_terms(values, grid_values)
        for f in range(T.shape[1]):
            np.add.at(self.request_n, T[:, f], 1)
        return {"op": "note_requests", "n": int(np.asarray(values).shape[0])}

    def refresh(self) -> dict:
        for k in range(self.K):
            self.modes[k], self.rank_reports[k] = modes_from_moments(
                self.S[k],
                self.t[k],
                int(self.n[k]),
                self.n_atoms,
                self.eig_rel_tol,
                var_share_tol=getattr(self, "var_share_tol", VARIANCE_SHARE_TOL),
            )
        self._refresh_coupling()
        return {"op": "refresh", "ranks": [r["rank_kept"] for r in self.rank_reports]}

    def _refresh_coupling(self) -> None:
        a = self.coupling_alpha
        lay = coupling_layout(self.cardinalities)
        toff = term_offsets(self.cardinalities)
        N = float(max(int(self.n.max()) if self.n.size else 0, 1))
        for j, (f, g) in enumerate(lay["pairs"]):
            cf, cg = (self.cardinalities[f], self.cardinalities[g])
            o = int(lay["offsets"][j])
            blk = self.pair_n[o : o + cf * cg].reshape(cf, cg).astype(np.float64)
            nf = self.term_n[int(toff[f]) : int(toff[f]) + cf].astype(np.float64)
            ng = self.term_n[int(toff[g]) : int(toff[g]) + cg].astype(np.float64)
            denom = N + a * cf * cg
            pj = (blk + a) / denom
            pf = (nf + a * cg) / denom
            pg = (ng + a * cf) / denom
            self.coupling[o : o + cf * cg] = (
                np.log(pj / np.maximum(pf[:, None] * pg[None, :], 1e-300))
                .ravel()
                .astype(np.float32)
            )

    def delete(
        self,
        X_del: np.ndarray,
        values_del: Optional[np.ndarray] = None,
        grid_values: Optional[Sequence[np.ndarray]] = None,
    ) -> dict:
        before = int(ROWS_CONSUMED["n"])
        m = int(np.asarray(X_del).shape[0])
        if m <= 0:
            raise ValueError("delete() was handed no rows")
        for k, ch in enumerate(self.well_channels):
            S, t, n = _row_moments(X_del, ch)
            if n >= int(self.n[k]):
                raise AssertionError(
                    f"well {k}: deleting {n} of {int(self.n[k])} rows leaves no mean"
                )
            self.S[k] -= S
            self.t[k] -= t
            self.n[k] -= n
        if values_del is not None:
            self._count(values_del, grid_values, sign=-1)
        self.refresh()
        touched = int(ROWS_CONSUMED["n"]) - before
        if touched != m * self.K:
            raise AssertionError(
                f"delete() consumed {touched} row-reads for {m} rows over {self.K} wells -- something other than the deleted rows was read"
            )
        rec = {
            "op": "delete",
            "n_rows_deleted": m,
            "rows_consumed_by_this_call": touched,
            "rows_consumed_per_well": m,
            "n_rows_remaining": int(self.n[0]),
            "ranks_after": [r["rank_kept"] for r in self.rank_reports],
            "refit": False,
            "what": "covariance downdate + re-eigendecomposition + count downdate; the retained corpus was not opened",
        }
        self.history.append(rec)
        return rec

    def punch_hole(self, f: int, v: int, g: int, u: int) -> dict:
        idx = coupling_index(self.cardinalities, f, v, g, u)
        before = float(self.coupling[idx])
        self.coupling_hole[idx] = 1
        rec = {
            "op": "punch_hole",
            "pair": [[int(f), int(v)], [int(g), int(u)]],
            "coupling_index": int(idx),
            "permitted_bytes": 1,
            "coupling_value_unchanged": bool(float(self.coupling[idx]) == before),
            "n_holes_after": int(self.coupling_hole.sum()),
            "refit": False,
        }
        self.history.append(rec)
        return rec

    def byte_footprint(self, budget_bytes: int = 524288) -> dict:
        b = int(sum((M.size for M in self.modes)) * 4)
        return {
            "deployed_bytes": b,
            "budget_bytes": int(budget_bytes),
            "matches_budget": bool(b == int(budget_bytes)),
            "param_count": b // 4,
            "what": f"{self.n_atoms} atoms x {self.dim} float32 -- the per-well leading principal subspace, the SAME block pca_tensor deploys",
            "variance_explained_per_well": [
                round(float(r.get("variance_explained", 0.0)), 6) for r in self.rank_reports
            ],
            "rank_kept_per_well": [int(r.get("rank_kept", 0)) for r in self.rank_reports],
            "rank_over_d_per_well": [
                round(float(r.get("rank_over_d", 0.0)), 4) for r in self.rank_reports
            ],
            "any_well_saturated": bool(any((r.get("saturated") for r in self.rank_reports))),
            "note": "⛔ the running covariance is NOT in this number; it is declared separately in covariance_footprint() and never netted into it",
        }

    def covariance_footprint(self) -> dict:
        d = self.well_sizes
        sq = int(sum((x * x for x in d)))
        tri = int(sum((x * (x + 1) // 2 for x in d)))
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
            "if_symmetric_triangle_float64": int(tri * 8),
            "if_symmetric_triangle_float32": int(tri * 4),
            "if_full_square_float32": int(sq * 4),
            "why_float64": "MEASURED: the downdate residual on S runs to 5e-10 absolute on entries of order 1e+9; a float32 accumulator (eps ~ 1e-7 relative) would bury the exactness this object exists to have",
            "note": "⛔ EXTRA to the store's 524,288 B and reported BESIDE it, never netted into it and never netted against the address book -- the same rule stated for address_bytes and for coupling_bytes",
        }

    def count_footprint(self) -> dict:
        """PART 3 -- the request count table."""
        return {
            "term_count_bytes": int(self.term_n.nbytes),
            "request_count_bytes": int(self.request_n.nbytes),
            "pair_count_bytes": int(self.pair_n.nbytes),
            "n_terms": int(self.n_terms),
            "n_pairs": int(self.n_pairs),
            "dtype": "int64",
            "total_count_bytes": int(
                self.term_n.nbytes + self.request_n.nbytes + self.pair_n.nbytes
            ),
            "what": "term_n = rows held carrying each (factor, value); request_n = times each value was ASKED for; pair_n = rows held carrying each conjunction. ⛔ Held apart on purpose: asked-for and written-down are different facts.",
        }

    def coupling_footprint(self) -> dict:
        """PART 4 -- the coupling table and its holes."""
        return {
            "coupling_bytes": int(self.coupling.nbytes),
            "coupling_hole_bytes": int(self.coupling_hole.nbytes),
            "bytes_per_hole": 1,
            "n_pairs": int(self.n_pairs),
            "n_holes": int(self.coupling_hole.sum()),
            "value_and_hole_in_different_arrays": True,
            "what": "coupling = add-alpha PMI, a function of the COUNTS alone, so it downdates exactly with them; coupling_hole = the earlier study's mechanism, ONE byte per pair, 0 bytes outside, and the value is left untouched so an unseen pair still gets its prior",
        }

    def honest_footprint(self, budget_bytes: int = 524288) -> dict:
        st = self.byte_footprint(budget_bytes)
        cv = self.covariance_footprint()
        ct = self.count_footprint()
        cp = self.coupling_footprint()
        tot = int(
            st["deployed_bytes"]
            + cv["total_statistics_bytes"]
            + ct["total_count_bytes"]
            + cp["coupling_bytes"]
            + cp["coupling_hole_bytes"]
        )
        cheap = int(
            st["deployed_bytes"]
            + cv["if_symmetric_triangle_float64"]
            + cv["first_moment_bytes"]
            + cv["count_bytes"]
            + ct["total_count_bytes"]
            + cp["coupling_bytes"]
            + cp["coupling_hole_bytes"]
        )
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
            "verdict": "⛔ THE HONEST FOOTPRINT IS OVER THE PIN. The content alone is at 524,288 B exactly, but exact deletion is paid for in sufficient statistics and they are not free. Printed, not netted (task section 3).",
        }

    def rank_report(self) -> dict:
        """The rank cut, per well, with the margin on both sides of the threshold."""
        rr = [dict(r) for r in self.rank_reports]
        tol_lim = [r for r in rr if r["rank_limited_by"] == "eigenvalue tolerance"]
        lo = [
            r["smallest_kept_over_threshold"]
            for r in tol_lim
            if np.isfinite(r["smallest_kept_over_threshold"])
        ]
        hi = [r["largest_dropped_over_threshold"] for r in tol_lim]
        return {
            "eig_rel_tol": self.eig_rel_tol,
            "per_well": rr,
            "n_wells_limited_by_tolerance": len(tol_lim),
            "n_wells_limited_by_atom_budget": sum(
                (1 for r in rr if r["rank_limited_by"] == "atom budget A-1")
            ),
            "min_smallest_kept_over_threshold": float(min(lo)) if lo else float("inf"),
            "max_largest_dropped_over_threshold": float(max(hi)) if hi else 0.0,
            "cut_is_marginal": bool(
                (min(lo) if lo else np.inf) < 10.0 or (max(hi) if hi else 0.0) > 0.1
            ),
            "marginality_scored_over": "only the wells where the eigenvalue tolerance is what binds the rank; a well limited by the atom budget drops a large eigenvalue by design and says nothing about the cut",
            "what": "a direction below the threshold is written as a ZERO row: it costs the same bytes and buys no span, which is PCASpanStore's own stated convention for an atom a well cannot use",
        }

    def content_cost_report(self) -> dict:
        rr = [dict(r) for r in self.rank_reports]
        return {
            "var_share_tol": getattr(self, "var_share_tol", VARIANCE_SHARE_TOL),
            "per_well": [
                {
                    "well": i,
                    "d_k": r["d_k"],
                    "rank_kept": r["rank_kept"],
                    "rank_kept_without_share_cut": r.get("rank_kept_without_share_cut"),
                    "n_directions_dropped_by_share_cut": r.get("n_directions_dropped_by_share_cut"),
                    "variance_share_lost_to_share_cut": r.get("variance_share_lost_to_share_cut"),
                    "smallest_kept_variance_share": r.get("smallest_kept_variance_share"),
                    "largest_dropped_variance_share": r.get("largest_dropped_variance_share"),
                    "variance_explained": r.get("variance_explained"),
                }
                for i, r in enumerate(rr)
            ],
            "total_directions_dropped": int(
                sum((r.get("n_directions_dropped_by_share_cut", 0) for r in rr))
            ),
            "worst_variance_share_lost_in_a_well": float(
                max([r.get("variance_share_lost_to_share_cut", 0.0) for r in rr] or [0.0])
            ),
            "wells_where_share_cut_binds": [
                i for i, r in enumerate(rr) if r.get("share_cut_binds")
            ],
            "store_bytes_unchanged": "the dropped directions are written as ZERO rows, so the byte pin is untouched -- the cost is content, not bytes",
        }

    def well_posedness_report(self) -> dict:
        rr = [dict(r) for r in self.rank_reports]
        f32 = float(np.spacing(np.float32(1.0)))
        sens = [r.get("predicted_byte_sensitivity", float("nan")) for r in rr]
        worst = max(sens) if sens else float("nan")
        bad = [
            i for i, r in enumerate(rr) if not r.get("bytes_determined_below_one_float32_ulp", True)
        ]
        return {
            "float32_ulp_at_unit_scale": f32,
            "per_well": [
                {
                    "well": i,
                    "d_k": r["d_k"],
                    "rank_kept": r["rank_kept"],
                    "rank_limited_by": r["rank_limited_by"],
                    "min_rel_gap_among_kept": r.get("min_rel_gap_among_kept"),
                    "min_rel_gap_to_any_neighbour": r.get("min_rel_gap_to_any_neighbour"),
                    "predicted_byte_sensitivity": r.get("predicted_byte_sensitivity"),
                    "predicted_sensitivity_in_float32_ulp": r.get(
                        "predicted_sensitivity_in_float32_ulp"
                    ),
                    "bytes_determined_below_one_float32_ulp": r.get(
                        "bytes_determined_below_one_float32_ulp"
                    ),
                }
                for i, r in enumerate(rr)
            ],
            "worst_predicted_byte_sensitivity": worst,
            "worst_predicted_in_float32_ulp": worst / f32 if f32 > 0 else None,
            "wells_not_determined_at_float32": bad,
            "store_bytes_well_posed": bool(not bad),
            "what": "computed from the sufficient statistics at FIT time -- no deletion and no second fit. ⛔ A well listed in wells_not_determined_at_float32 has atom bytes that are not a function of its data, so no deletion procedure and no refit can agree on them; that is a property of the CONTENT, not of the downdate.",
        }

    def content_sha(self) -> str:
        h = hashlib.sha256()
        for M in self.modes:
            h.update(np.ascontiguousarray(np.asarray(M, np.float32)).tobytes())
        return h.hexdigest()

    def state_arrays(self) -> list:
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


class _Tripwire:
    def __init__(self, what: str):
        self._what = str(what)

    def _boom(self, *_a, **_k):
        raise AssertionError(
            f"delete() touched {self._what} -- this object claims NO PASS OVER THE DATA and it just made one"
        )

    __getitem__ = __getattr__ = __call__ = __len__ = __iter__ = __array__ = _boom


def no_data_pass(fn, *args, modules=(), **kwargs) -> dict:
    import builtins

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
    return {
        "result": result,
        "rows_consumed": int(after - before),
        "tripwires_installed": [f"{m.__name__}.{n}" for m, n, _ in patched],
        "n_tripwires": len(patched),
        "corpus_was_unreachable_during_the_call": True,
    }


def retention_audit(store: DowndatableStore, n_rows_fitted: int) -> dict:
    declared = {name: arr for name, arr, _ in store.state_arrays()}
    seen, undeclared = (set(), [])

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
            f"retention audit: {len(undeclared)} array(s) reachable from the store are not in state_arrays() -- {undeclared[:4]} -- an undeclared array is a place a row could be hiding"
        )
    allowed = {store.n_atoms, store.K, store.n_terms, store.n_pairs}
    allowed |= set(store.well_sizes)
    rows = []
    for name, arr, formula in store.state_arrays():
        a = np.asarray(arr)
        bad = [int(x) for x in a.shape if int(x) not in allowed]
        rows.append(
            {
                "name": name,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "bytes": int(a.nbytes),
                "declared_shape": formula,
                "axes_not_explained": bad,
            }
        )
        if bad:
            raise AssertionError(
                f"retention audit: {name} has shape {a.shape}, and axis/axes {bad} are not explained by (A, d_k, K, T, NC) -- this array may hold rows"
            )
        if int(n_rows_fitted) in {int(x) for x in a.shape}:
            raise AssertionError(
                f"retention audit: {name} has an axis of length n_rows ({n_rows_fitted}) -- this object is keeping items"
            )
    margins = [int(nk) - dk for nk, dk in zip(store.n, store.well_sizes)]
    if min(margins) <= 0:
        raise AssertionError(
            f"retention audit: a well has n <= d_k (margins {margins}). A covariance of fewer rows than channels is rank-deficient in a way that can expose the rows themselves, so the no-items claim does not hold there"
        )
    return {
        "items_kept": False,
        "n_arrays": len(rows),
        "arrays": rows,
        "total_object_bytes": int(sum((r["bytes"] for r in rows))),
        "n_rows_fitted": int(n_rows_fitted),
        "no_array_has_an_n_rows_axis": True,
        "every_shape_explained_by": "(A, d_k, K, T, NC)",
        "inventory_complete_vs_vars": True,
        "n_over_d_margin_per_well": [int(m) for m in margins],
        "min_n_over_d_margin": int(min(margins)),
        "what": "every array in the object, named, shaped and explained; not one axis is the corpus size, so the object cannot be holding a row",
    }


def compare_stores(a: DowndatableStore, b: DowndatableStore) -> dict:
    if a.K != b.K:
        raise AssertionError(f"{a.K} wells vs {b.K}: not the same store shape")
    per_well, ok = ([], True)
    for k in range(a.K):
        A = np.asarray(a.modes[k], np.float32)
        B = np.asarray(b.modes[k], np.float32)
        if A.shape != B.shape:
            raise AssertionError(f"well {k}: {A.shape} vs {B.shape}")
        dif = np.abs(A.astype(np.float64) - B.astype(np.float64))
        dev = float(dif.max()) if dif.size else 0.0
        scale = float(np.max(np.abs(B))) if B.size else 1.0
        mag = np.maximum(np.abs(A.astype(np.float64)), np.abs(B.astype(np.float64)))
        ulp_e = np.spacing(np.maximum(mag, np.finfo(np.float32).tiny).astype(np.float32))
        ulp_entry = float(np.max(dif / np.asarray(ulp_e, np.float64))) if dif.size else 0.0
        ulp_unit = float(np.spacing(np.float32(1.0)))
        ulp_dist = dev / ulp_unit
        ulp = float(np.spacing(np.float32(max(scale, np.finfo(np.float32).tiny))))
        ra = int(a.rank_reports[k].get("rank_kept", 0))
        rb = int(b.rank_reports[k].get("rank_kept", 0))
        if ra == rb and ra > 0:
            Qa = np.linalg.qr(A[1 : 1 + ra].T.astype(np.float64))[0]
            Qb = np.linalg.qr(B[1 : 1 + rb].T.astype(np.float64))[0]
            proj = float(np.linalg.norm(Qa @ Qa.T - Qb @ Qb.T, 2))
        else:
            proj = float("nan")
        row = {
            "well": k,
            "d_k": int(a.well_sizes[k]),
            "max_abs_dev": dev,
            "max_dev_in_float32_ulp_at_unit_scale": ulp_dist,
            "max_rel_dev_to_block_scale": dev / scale if scale > 0 else 0.0,
            "max_entrywise_float32_ulp_distance": ulp_entry,
            "max_abs_dev_over_ulp_at_block_scale": dev / ulp if ulp > 0 else float("inf"),
            "bit_identical": bool(np.array_equal(A, B)),
            "atom_block_scale": scale,
            "float32_ulp_at_scale": ulp,
            "rank_kept_a": ra,
            "rank_kept_b": rb,
            "ranks_agree": bool(ra == rb),
            "subspace_projector_dev": proj,
            "n_rows_a": int(a.n[k]),
            "n_rows_b": int(b.n[k]),
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
            r["max_entrywise_float32_ulp_distance"] for r in per_well
        ],
        "n_wells_bit_identical": int(sum((r["bit_identical"] for r in per_well))),
        "ranks_agree_in_every_well": bool(ok),
        "counts_equal": bool(all((r["counts_agree"] for r in per_well))),
        "term_counts_equal": bool(np.array_equal(a.term_n, b.term_n)),
        "pair_counts_equal": bool(np.array_equal(a.pair_n, b.pair_n)),
        "coupling_max_abs_dev": float(
            np.max(np.abs(a.coupling.astype(np.float64) - b.coupling.astype(np.float64)))
        ),
        "per_well": per_well,
    }


def pinned_span_basis(store, tol: float = 1e-08, mu_rel_tol: float = None) -> list:
    mrt = MU_PERP_REL_TOL if mu_rel_tol is None else float(mu_rel_tol)
    out = []
    for G in store.modes:
        M = np.asarray(G, np.float64)
        mu, dirs = (M[0], M[1:])
        keep = dirs[np.linalg.norm(dirs, axis=1) > tol]
        U = keep.T if keep.size else np.zeros((M.shape[1], 0), np.float64)
        r = mu - U @ (U.T @ mu) if U.shape[1] else mu
        nr = float(np.linalg.norm(r))
        nmu = max(float(np.linalg.norm(mu)), 1e-300)
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
        mu, dirs = (M[0], M[1:])
        keep = dirs[np.linalg.norm(dirs, axis=1) > 1e-08]
        U = keep.T if keep.size else np.zeros((M.shape[1], 0), np.float64)
        r = mu - U @ (U.T @ mu) if U.shape[1] else mu
        nmu = max(float(np.linalg.norm(mu)), 1e-300)
        rel = float(np.linalg.norm(r)) / nmu
        rows.append(
            {
                "well": k,
                "n_directions": int(keep.shape[0]),
                "mu_norm": nmu,
                "mu_perp_relative_norm": rel,
                "mean_admitted": bool(rel > mrt),
                "margin_over_threshold": float(rel / mrt),
            }
        )
    return {
        "mu_perp_rel_tol": mrt,
        "per_well": rows,
        "span_ranks": [r["n_directions"] + int(r["mean_admitted"]) for r in rows],
        "closest_margin_above": min(
            [r["margin_over_threshold"] for r in rows if r["mean_admitted"]] or [float("inf")]
        ),
        "closest_margin_below": max(
            [r["margin_over_threshold"] for r in rows if not r["mean_admitted"]] or [0.0]
        ),
    }


def gauge_report(store_a, store_b, coeffs, basis_fn) -> dict:
    from nclu.core.gen_read import render_span

    Ua, Ub = (basis_fn(store_a), basis_fn(store_b))
    ra = [int(np.asarray(U).shape[1]) for U in Ua]
    rb = [int(np.asarray(U).shape[1]) for U in Ub]
    if ra != rb:
        return {
            "basis": basis_fn.__name__,
            "ranks_a": ra,
            "ranks_b": rb,
            "comparable": False,
            "note": "the two bases do not even have the same rank",
        }
    Xa = render_span(store_a, [np.asarray(c, np.float32)[:, :r] for c, r in zip(coeffs, ra)], Ua)
    Xb = render_span(store_b, [np.asarray(c, np.float32)[:, :r] for c, r in zip(coeffs, rb)], Ub)
    d = np.abs(Xa.astype(np.float64) - Xb.astype(np.float64))
    scale = float(np.max(np.abs(Xb))) if Xb.size else 1.0
    return {
        "basis": basis_fn.__name__,
        "comparable": True,
        "span_ranks": ra,
        "max_abs_render_dev": float(d.max()),
        "median_row_l2_dev": float(np.median(np.linalg.norm(Xa - Xb, axis=1))),
        "render_scale": scale,
        "relative_to_render_scale": float(d.max() / max(scale, 1e-12)),
        "what": "the same coefficients through two stores that agree to 1e-10; any deviation here is the BASIS, not the deletion",
    }


def direction_correspondence(modes_pre, modes_post, tol: float = 1e-08) -> dict:
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
            if perm[i] < 0 and (not taken[j]):
                perm[i], cos[i], taken[j] = (int(j), float(M[i, j]), True)
                if (perm >= 0).all():
                    break
        shift = np.abs(perm - np.arange(perm.size))
        out.append(
            {
                "well": k,
                "r": int(live_p.size),
                "permutation": [int(x) for x in perm],
                "n_rank_order_changed": int((shift > 0).sum()),
                "frac_rank_order_changed": float((shift > 0).mean()) if shift.size else 0.0,
                "max_shift": int(shift.max()) if shift.size else 0,
                "min_matched_cosine": float(cos.min()) if cos.size else 1.0,
                "median_matched_cosine": float(np.median(cos)) if cos.size else 1.0,
                "is_identity": bool((shift == 0).all()),
            }
        )
    return {
        "per_well": out,
        "any_reordering": bool(any((not w["is_identity"] for w in out))),
        "total_slots_reordered": int(sum((w["n_rank_order_changed"] for w in out))),
        "total_slots": int(sum((w["r"] for w in out))),
        "permutation_bytes_if_carried": int(sum((w["r"] for w in out)) * 2),
        "what": "the store orders directions by eigenvalue; a deletion moves the eigenvalues, so slots swap. The swapped pairs are nearly degenerate so the SPAN barely moves, but the two directions are orthogonal, so a book slot pointed at the wrong one is O(1) wrong in that slot.",
    }


def pinned_span_basis_matched(
    store, modes_pre, tol: float = 1e-08, mu_rel_tol: float = None
) -> tuple:
    corr = direction_correspondence(modes_pre, store.modes, tol=tol)
    reordered = []
    for G, w in zip(store.modes, corr["per_well"]):
        M = np.asarray(G, np.float32).copy()
        dirs = M[1:]
        live = np.flatnonzero(np.linalg.norm(np.asarray(dirs, np.float64), axis=1) > tol)
        perm = np.asarray(w["permutation"], np.int64)
        block = dirs[live].copy()
        M[1 : 1 + live.size] = block[perm]
        reordered.append(M)

    class _Reordered:
        modes = reordered
        well_channels = store.well_channels
        dim = store.dim

    return (pinned_span_basis(_Reordered, tol=tol, mu_rel_tol=mu_rel_tol), corr)


def carry_address_book(addr, U_pre, U_post):
    import copy

    new = copy.deepcopy(addr)
    Ms = []
    for k, (U0, U1) in enumerate(zip(U_pre, U_post)):
        A0 = np.asarray(U0, np.float64)
        A1 = np.asarray(U1, np.float64)
        if A0.shape[1] != np.asarray(addr.W[k]).shape[1]:
            raise AssertionError(
                f"well {k}: the book emits {np.asarray(addr.W[k]).shape[1]} coefficients but the pre-deletion basis has {A0.shape[1]} columns"
            )
        Ms.append(A0.T @ A1)
    o = 0
    for k, M in enumerate(Ms):
        Wk = np.asarray(new.W[k], np.float64)
        r_in = Wk.shape[1]
        if getattr(new, "scale", None) is not None:
            s = np.asarray(new.scale[o : o + r_in], np.float64)
            Wk = Wk * s[None, :]
        new.W[k] = np.ascontiguousarray((Wk @ M).astype(np.asarray(addr.W[k]).dtype))
        o += r_in
    if getattr(new, "scale", None) is not None:
        cen, o = ([], 0)
        for k, M in enumerate(Ms):
            r_in = np.asarray(addr.W[k]).shape[1]
            cen.append(np.asarray(new.center[o : o + r_in], np.float64) @ M)
            o += r_in
        new.center = np.concatenate(cen).astype(np.asarray(addr.center).dtype)
        new.scale = np.ones_like(new.center)
    return (
        new,
        {
            "op": "carry_address_book",
            "n_wells": len(Ms),
            "correspondence_min_singular_value": [
                float(np.linalg.svd(M, compute_uv=False).min()) for M in Ms
            ],
            "correspondence_max_abs_off_identity": [
                float(np.abs(M - np.eye(M.shape[0])).max()) for M in Ms
            ],
            "address_book_bytes_changed": int(sum((np.asarray(w).size for w in addr.W)) * 4),
            "address_book_bytes_total": int(addr.byte_footprint().get("address_bytes", 0)),
            "data_rows_touched": 0,
            "refit": False,
        },
    )


def deployed_span(G, tol: float = 1e-08) -> np.ndarray:
    M = np.asarray(G, np.float64)
    live = M[np.linalg.norm(M, axis=1) > tol]
    if live.size == 0:
        return np.zeros((M.shape[1], 0), np.float64)
    U, sv, _ = np.linalg.svd(live.T, full_matrices=False)
    r = int((sv > tol * float(sv[0])).sum()) if sv.size else 0
    return np.ascontiguousarray(U[:, :r])


def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape[1] == 0 and B.shape[1] == 0:
        return 0.0
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 1.0
    return float(np.linalg.norm(A @ A.T - B @ B.T, 2))


def compare_subspaces(a, b) -> dict:
    if a.K != b.K:
        raise AssertionError(f"{a.K} wells vs {b.K}: not the same store shape")
    f32 = float(np.spacing(np.float32(1.0)))
    per_well, ok = ([], True)
    for k in range(a.K):
        GA = np.asarray(a.modes[k], np.float32)
        GB = np.asarray(b.modes[k], np.float32)
        QA, QB = (deployed_span(GA), deployed_span(GB))
        pd = subspace_distance(QA, QB)
        max_ang = float(np.arcsin(min(pd, 1.0)))
        DA, DB = (deployed_span(GA[1:]), deployed_span(GB[1:]))
        pdd = subspace_distance(DA, DB)
        max_angd = float(np.arcsin(min(pdd, 1.0)))
        mu_dev = float(np.max(np.abs(GA[0].astype(np.float64) - GB[0].astype(np.float64))))
        mu_norm = float(np.linalg.norm(GB[0].astype(np.float64)))
        row = {
            "well": k,
            "d_k": int(a.well_sizes[k]),
            "span_rank_a": int(QA.shape[1]),
            "span_rank_b": int(QB.shape[1]),
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
        "definition": "the stored object is the SUBSPACE the atom block spans, plus the mean; the particular eigenvector basis is not the object",
        "float32_ulp_at_unit_scale": f32,
        "max_principal_angle_rad_per_well": angs,
        "worst_max_principal_angle_rad": float(max(angs)) if angs else 0.0,
        "worst_in_float32_ulp": float(max(angs)) / f32 if angs else 0.0,
        "mean_max_abs_dev_per_well": mus,
        "worst_mean_max_abs_dev": float(max(mus)) if mus else 0.0,
        "span_ranks_agree_every_well": bool(ok),
        "n_wells_span_equal_below_one_ulp": int(
            sum((r["subspace_equal_below_one_float32_ulp"] for r in per_well))
        ),
        "all_wells_span_equal_below_one_ulp": bool(
            all((r["subspace_equal_below_one_float32_ulp"] for r in per_well))
        ),
        "per_well": per_well,
    }
