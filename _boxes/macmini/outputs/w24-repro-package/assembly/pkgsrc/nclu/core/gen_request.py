"""gen request."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

Embedding = np.ndarray


def requests_from_rows(grid, rows: np.ndarray) -> list[dict]:
    """The A / B request list: one complete, satisfiable assignment per row.

    A **valid** request names each factor exactly once at a realised grid value,
    so it is exactly the factor tuple of a dataset row — which is what makes the
    probe's read of the arm's render a *composition* score rather than a guess.
    """
    rows = np.asarray(rows, dtype=np.int64)
    vals = grid.values_of(rows)
    idx = grid.factor_indices_of(rows)
    out = []
    for i, row in enumerate(rows.tolist()):
        out.append(
            {
                "id": f"row-{int(row)}",
                "kind": "valid",
                "row": int(row),
                "request": [
                    {"factor": grid.names[k], "factor_index": int(k), "value": float(vals[i, k])}
                    for k in range(grid.n_factors)
                ],
                "class_indices": [int(x) for x in idx[i]],
            }
        )
    return out


def _extrapolating_index(values: np.ndarray, v: float) -> tuple[int, int, float]:
    n = int(values.size)
    if n == 1:
        return (0, 0, 0.0)
    hit = np.flatnonzero(values == v)
    if hit.size:
        return (int(hit[0]), int(hit[0]), 0.0)
    if v < values[0]:
        i, j = (0, 1)
    elif v > values[-1]:
        i, j = (n - 2, n - 1)
    else:
        j = int(np.searchsorted(values, v))
        i = j - 1
    denom = float(values[j] - values[i])
    w = 0.0 if denom == 0.0 else float((v - values[i]) / denom)
    return (i, j, w)


@dataclass
class FactorPrototypes:
    names: tuple
    values: tuple
    protos: tuple
    mu_global: np.ndarray
    n_rows_fit: int
    n_per_value: tuple

    @classmethod
    def fit(
        cls, grid, features: np.ndarray, rows: np.ndarray, min_per_value: int = 1
    ) -> "FactorPrototypes":
        X = np.asarray(features, np.float32)
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        fi = grid.factor_indices_of(np.asarray(rows, np.int64))
        if fi.shape[0] != X.shape[0]:
            raise ValueError(f"{fi.shape[0]} rows for {X.shape[0]} feature vectors")
        mu = X.mean(axis=0)
        protos, counts = ([], [])
        for k in range(grid.n_factors):
            c_k = int(grid.cardinalities[k])
            P = np.zeros((c_k, X.shape[1]), np.float32)
            n_k = np.zeros(c_k, np.int64)
            for c in range(c_k):
                m = fi[:, k] == c
                n_k[c] = int(m.sum())
                P[c] = X[m].mean(axis=0) if n_k[c] >= int(min_per_value) else mu
            protos.append(P)
            counts.append(tuple((int(x) for x in n_k)))
        return cls(
            names=tuple(grid.names),
            values=tuple((np.asarray(v, np.float64) for v in grid.values)),
            protos=tuple(protos),
            mu_global=mu,
            n_rows_fit=int(X.shape[0]),
            n_per_value=tuple(counts),
        )

    @property
    def dim(self) -> int:
        return int(self.mu_global.shape[0])

    @property
    def n_prototypes(self) -> int:
        return int(sum((p.shape[0] for p in self.protos)))

    @property
    def n_bytes(self) -> int:
        """Bytes of the prototype table **as deployed** (float32), + the global mean."""
        return int(
            sum((p.size * p.dtype.itemsize for p in self.protos))
            + self.mu_global.size * self.mu_global.dtype.itemsize
        )

    def byte_footprint(self) -> dict:
        return {
            "n_prototypes": self.n_prototypes,
            "dim": self.dim,
            "dtype": "float32",
            "bytes_formula": "(sum_k C_k + 1) * D * 4",
            "bytes": self.n_bytes,
            "role": "request embedding == the kNN arm's ADDRESSING HEAD; the store's own byte count excludes the CLU addressing head by the same convention",
        }

    def prototype(self, factor_index: int, value: float) -> np.ndarray:
        """One term's prototype, interpolating/extrapolating off-grid values."""
        k = int(factor_index)
        i, j, w = _extrapolating_index(self.values[k], float(value))
        P = self.protos[k]
        if w == 0.0:
            return P[i]
        return P[i] + np.float32(w) * (P[j] - P[i])

    def embed(self, request: Sequence[dict]) -> Embedding:
        out = self.mu_global.astype(np.float32).copy()
        for term in request:
            out += self.prototype(int(term["factor_index"]), float(term["value"])) - self.mu_global
        return out

    def embed_many(self, requests: Sequence[dict]) -> np.ndarray:
        """``(N, D)`` embeddings for a list of request dicts (or raw term lists)."""
        if not requests:
            return np.zeros((0, self.dim), np.float32)
        out = np.empty((len(requests), self.dim), np.float32)
        for i, req in enumerate(requests):
            terms = req["request"] if isinstance(req, dict) else req
            out[i] = self.embed(terms)
        return out

    def offgrid_report(self, requests: Sequence[dict]) -> dict:
        n_terms = n_off = 0
        max_outside = 0.0
        for req in requests:
            for term in req["request"] if isinstance(req, dict) else req:
                k = int(term["factor_index"])
                v = float(term["value"])
                n_terms += 1
                vals = self.values[k]
                if not np.any(vals == v):
                    n_off += 1
                    if v < vals[0]:
                        max_outside = max(max_outside, float(vals[0] - v))
                    elif v > vals[-1]:
                        max_outside = max(max_outside, float(v - vals[-1]))
        return {
            "n_terms": int(n_terms),
            "n_offgrid_terms": int(n_off),
            "max_distance_outside_range": float(max_outside),
        }


@dataclass(frozen=True)
class DenseRequests:
    cls_targets: np.ndarray
    val_targets: np.ndarray
    extra_factor: np.ndarray
    extra_cls: np.ndarray
    extra_val: np.ndarray
    n_factors: int

    @property
    def n(self) -> int:
        return int(self.cls_targets.shape[0])


def normalise_value(values: np.ndarray, v: float) -> float:
    """Min-max normalise ``v`` against a factor's realised grid range."""
    lo, hi = (float(values[0]), float(values[-1]))
    span = hi - lo
    return float((v - lo) / span) if span != 0.0 else 0.0


def dense_request_arrays(grid, requests: Sequence[dict]) -> DenseRequests:
    n_f = int(grid.n_factors)
    n = len(requests)
    cls_t = np.zeros((n, n_f), np.int32)
    val_t = np.zeros((n, n_f), np.float32)
    ex_f = np.full(n, -1, np.int32)
    ex_c = np.zeros(n, np.int32)
    ex_v = np.zeros(n, np.float32)
    for i, req in enumerate(requests):
        terms = req["request"] if isinstance(req, dict) else req
        seen = set()
        for term in terms:
            k = int(term["factor_index"])
            v = float(term["value"])
            vals = np.asarray(grid.values[k], np.float64)
            c = int(np.argmin(np.abs(vals - v)))
            nv = normalise_value(vals, v)
            if k in seen:
                if ex_f[i] != -1:
                    raise ValueError(
                        f"request {req.get('id')!r} has more than one duplicate term; the dense form carries exactly one extra slot"
                    )
                ex_f[i], ex_c[i], ex_v[i] = (k, c, nv)
            else:
                seen.add(k)
                cls_t[i, k], val_t[i, k] = (c, nv)
    return DenseRequests(cls_t, val_t, ex_f, ex_c, ex_v, n_f)
