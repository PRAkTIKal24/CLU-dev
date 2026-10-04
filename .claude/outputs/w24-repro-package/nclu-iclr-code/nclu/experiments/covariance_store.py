"""covariance store."""

from __future__ import annotations
import numpy as np

SATURATION_NOTE = "rank/d_k < 1 strictly: at rank == d_k the atom block spans the whole well, the projection is the identity and the content carries no information (protocol C.1/C.4 -- a decorative store), and the random-atom control stops collapsing"
_ACC = np.float64
EIG_REL_TOL = 1e-12


class WellStats:
    __slots__ = ("_Sp", "t", "n", "d", "_iu")

    def __init__(self, d: int):
        self.d = int(d)
        self._iu = np.triu_indices(self.d)
        self._Sp = np.zeros(self._iu[0].size, _ACC)
        self.t = np.zeros(self.d, _ACC)
        self.n = 0

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
        return self._unpack()

    def add(self, X: np.ndarray) -> None:
        """Accumulate rows.  ``X`` is ``(m, d)``."""
        Xa = np.asarray(X, _ACC)
        self._Sp += self._pack(Xa.T @ Xa)
        self.t += Xa.sum(axis=0)
        self.n += int(Xa.shape[0])

    def remove(self, X: np.ndarray) -> None:
        Xa = np.asarray(X, _ACC)
        if Xa.shape[0] > self.n:
            raise AssertionError(f"asked to remove {Xa.shape[0]} rows from a well holding {self.n}")
        self._Sp -= self._pack(Xa.T @ Xa)
        self.t -= Xa.sum(axis=0)
        self.n -= int(Xa.shape[0])

    def mean(self) -> np.ndarray:
        if self.n <= 0:
            raise AssertionError("empty well has no mean")
        return self.t / self.n

    def covariance(self) -> np.ndarray:
        if self.n < 2:
            raise AssertionError(f"covariance needs n >= 2, well holds {self.n}")
        C = (self._unpack() - np.outer(self.t, self.t) / self.n) / (self.n - 1)
        return 0.5 * (C + C.T)

    def atoms(self, n_atoms: int, return_rank: bool = False):
        A = int(n_atoms)
        mu = self.mean()
        r = min(A - 1, self.d)
        kept = 0
        if r > 0:
            C = self.covariance()
            w, V = np.linalg.eigh(C)
            w = w[::-1]
            V = V[:, ::-1]
            lam_max = float(w[0]) if w.size else 0.0
            if lam_max > 0:
                kept = int(min(r, int((w > EIG_REL_TOL * lam_max).sum())))
            V = V[:, :kept]
            if kept:
                Vr = V.T
                idx = np.argmax(np.abs(Vr), axis=1)
                sg = np.sign(Vr[np.arange(Vr.shape[0]), idx])
                sg[sg == 0] = 1.0
                Vr = Vr * sg[:, None]
                B = np.vstack([mu[None, :], Vr])
            else:
                B = mu[None, :]
        else:
            B = mu[None, :]
        B = B.astype(np.float32)
        if B.shape[0] < A:
            B = np.vstack([B, np.zeros((A - B.shape[0], B.shape[1]), np.float32)])
        B = np.ascontiguousarray(B)
        return (B, kept) if return_rank else B

    def stat_floats(self) -> int:
        packed = int(self._Sp.size)
        if packed != self.d * (self.d + 1) // 2:
            raise AssertionError(
                f"packed triangle is {packed} floats, not {self.d * (self.d + 1) // 2} -- the footprint and the array disagree"
            )
        return packed + self.d + 1

    def stat_bytes(self) -> int:
        return self.stat_floats() * _ACC(0).itemsize


class CovContentStore:
    def __init__(
        self, X: np.ndarray, well_channels, n_atoms: int, dim: int, allow_saturated: bool = False
    ):
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
        self.deployed_rank = [r for _, r in out]

    @property
    def K(self) -> int:
        return len(self.modes)

    def span_bases(self) -> list:
        out = []
        for M, r in zip(self.modes, self.deployed_rank):
            if r <= 0:
                out.append(np.zeros((M.shape[1], 0), np.float64))
                continue
            out.append(np.linalg.qr(np.asarray(M[: 1 + r], np.float64).T)[0])
        return out

    def delete_rows(self, X_rows: np.ndarray) -> dict:
        Xf = np.asarray(X_rows, np.float32)
        n_before = [w.n for w in self.stats]
        U_old = self.span_bases()
        for w, ch in zip(self.stats, self.well_channels):
            w.remove(Xf[:, ch])
        self._refresh()
        U_new = self.span_bases()
        rot = []
        for Uo, Un in zip(U_old, U_new):
            r = min(Uo.shape[1], Un.shape[1])
            rot.append(
                np.asarray(Uo[:, :r].T @ Un[:, :r], np.float32)
                if r > 0
                else np.zeros((0, 0), np.float32)
            )
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
            "mechanism": "S_k -= x x^T, t_k -= x, n_k -= 1; the atoms are re-derived from the statistic and the address book is carried across the basis change by W_k <- W_k R_k, R_k = U_k^T U'_k. The pool is never touched.",
        }

    def basis_change_audit(self) -> dict:
        rot = getattr(self, "last_basis_change", None)
        if rot is None:
            raise AssertionError("no deletion has been performed; nothing to audit")
        devs, cos = ([], [])
        for R in rot:
            if R.size == 0:
                continue
            Rd = np.asarray(R, np.float64)
            devs.append(float(np.max(np.abs(Rd.T @ Rd - np.eye(Rd.shape[1])))))
            cos.append(np.linalg.svd(Rd, compute_uv=False))
        if not cos:
            return {"n_wells_audited": 0, "note": "no live wells"}
        allc = np.concatenate(cos)
        return {
            "n_wells_audited": len(devs),
            "max_abs_dev_from_orthogonality": max(devs),
            "principal_angle_cos_min": float(allc.min()),
            "principal_angle_cos_median": float(np.median(allc)),
            "frac_directions_cos_gt_0p99": float((allc > 0.99).mean()),
            "frac_directions_cos_lt_0p5": float((allc < 0.5).mean()),
            "what": "singular values of R_k = U_k^T U'_k are the principal-angle cosines between the pre- and post-deletion spans: 1 where the deletion changed nothing, small where it changed the content. A spectrum that is ALL ones would mean the deletion did nothing.",
            "compare_against": "the same statistic under a SIZE-MATCHED RANDOM deletion -- if the two agree, the change is generic damage, not targeted forgetting (the earlier study's evict_size_control verdict)",
        }

    def byte_footprint(self, budget_bytes: int) -> dict:
        atom_floats = int(sum((M.size for M in self.modes)))
        stat_bytes = int(sum((w.stat_bytes() for w in self.stats)))
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
            "what": f"{self.n_atoms} atoms x {self.dim} float32 deployed, PLUS the per-well second moment triangle + first moment + count that make deletion exact; summed, not netted",
            "note": SATURATION_NOTE,
        }


def assert_not_saturated(n_atoms: int, well_sizes) -> None:
    bad = [int(d) for d in well_sizes if int(n_atoms) >= int(d)]
    if bad:
        raise AssertionError(
            f"n_atoms={n_atoms} saturates {len(bad)} well(s) of size {sorted(set(bad))}: the atom block would span the whole well, the projection would be the identity and the content would carry no information. "
            + SATURATION_NOTE
        )


def plan_partition(D: int, budget_bytes: int, K: int) -> dict:
    Bf = int(budget_bytes) // 4
    d0, rem = divmod(int(D), int(K))
    ds = [d0 + 1] * rem + [d0] * (int(K) - rem)
    stat = sum((d * (d + 1) // 2 + d + 1 for d in ds))
    avail = Bf - stat
    A = avail // int(D) if avail > 0 else 0
    ranks = [min(A, d) for d in ds]
    return {
        "K": int(K),
        "well_sizes": ds,
        "n_atoms": int(A),
        "statistic_floats": int(stat),
        "atom_floats": int(A * D),
        "total_bytes": int((A * D + stat) * 4),
        "budget_bytes": int(budget_bytes),
        "within_budget": bool((A * D + stat) * 4 <= int(budget_bytes)),
        "total_span": int(sum(ranks)),
        "rank_over_d": [round(r / d, 4) for r, d in zip(ranks, ds)],
        "saturated": bool(any((A >= d for d in ds))),
    }


def honest_byte_footprint(well_sizes, n_atoms: int, D: int, budget_bytes: int) -> dict:
    ds = [int(d) for d in well_sizes]
    stat = sum((d * (d + 1) // 2 + d + 1 for d in ds))
    atoms = int(n_atoms) * int(D)
    total = (atoms + stat) * 4
    return {
        "atom_bytes": atoms * 4,
        "statistic_bytes": stat * 4,
        "deployed_bytes": total,
        "budget_bytes": int(budget_bytes),
        "within_budget": bool(total <= int(budget_bytes)),
        "multiple_of_budget": round(total / float(budget_bytes), 6),
        "well_sizes": ds,
        "rank_over_d": [round(min(int(n_atoms), d) / d, 4) for d in ds],
    }
