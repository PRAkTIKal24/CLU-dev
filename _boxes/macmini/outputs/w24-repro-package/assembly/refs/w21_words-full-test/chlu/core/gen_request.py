"""⭐ R8 — the **request embedding**, declared once and applied identically by every
baseline arm.

⛔ **Why this module exists at all.**  A refusal benchmark asks an arm to score a
request that *has no image*: "floor_hue at 0.8 **and** at 0.0" cannot be rendered,
so it cannot be encoded.  Every arm therefore needs a way to turn a **factor-space
request** into something its machinery can consume, and R8 (Head, 2026-09-09) says
that mapping is **declared once, up front, and never chosen by the arm that is
about to be scored** — an arm that picks its own request embedding is setting its
own exam.

What lives here:

* :class:`FactorPrototypes` — the **kNN arm's** embedding: per-(factor, value)
  prototype features computed from the **cached train features**, composed
  additively in deviation space.
* :func:`requests_from_rows` / :func:`requests_from_records` — the A/B and C
  request lists, built from ``chlu.data.gen_splits`` and ``c_invalid.json``
  respectively.  ⛔ Neither is ever *regenerated* here; A and B come from the
  pinned split and C is read from disk byte-for-byte.
* :func:`dense_request_arrays` — the dense ``(cls, value, extra)`` form the
  **LACE arm** consumes.  Both arms therefore read the *same* request objects.

⛔ The **CLU** arm's addressing head is ``gen-store-grouping``'s and is neither
reimplemented nor imported here (R8: "neither reimplements the other's").

⭐ **The one judgement call, declared rather than buried.**  An ``out_of_range``
request names a value that is *not on the grid*, so it has no prototype.  It is
obtained by **linear extrapolation in value space** from the two nearest grid
prototypes.  ⛔ It is **not** clipped to the nearest grid value: clipping would
silently convert an unsatisfiable request into a satisfiable one and hand the kNN
arm a free win on C.  A ``duplicate_factor`` request names one factor twice and
**both terms enter the composition** — that is what "requested simultaneously"
means — so the embedding is pulled by two mutually inconsistent deviations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: float32 ``np.ndarray`` of shape ``(D,)``.
Embedding = np.ndarray


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


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
                    {
                        "factor": grid.names[k],
                        "factor_index": int(k),
                        "value": float(vals[i, k]),
                    }
                    for k in range(grid.n_factors)
                ],
                "class_indices": [int(x) for x in idx[i]],
            }
        )
    return out


def requests_from_records(records: Sequence[dict]) -> list[dict]:
    """The C request list, taken **verbatim** from ``c_invalid.json``.

    ⛔ Nothing is re-sampled, re-ordered or re-expressed: the records are the
    single source (R8) and are byte-identical across arms.
    """
    out = []
    for rec in records:
        out.append(
            {
                "id": rec["id"],
                "kind": rec["kind"],
                "row": None,
                "request": [dict(t) for t in rec["request"]],
                "violated_factor": rec.get("violated_factor"),
            }
        )
    return out


def load_c_requests(path: str | Path) -> list[dict]:
    """Read ``c_invalid.json`` from disk and return its requests.  ⛔ Never rebuilds it."""
    with open(path) as fh:
        payload = json.load(fh)
    return requests_from_records(payload["records"])


def requests_from_valid_records(records: Sequence[dict]) -> list[dict]:
    """A **valid control** set (e.g. ``a7_redundant.json``), taken verbatim.

    ⭐ Same verbatim contract as :func:`requests_from_records`; the difference is
    that a control record is *satisfiable* and therefore carries the ``row`` it was
    built from, which the caller uses to assert the control never spends a held-out
    row.  ⛔ Nothing is re-sampled, re-ordered or re-expressed.
    """
    out = []
    for rec in records:
        out.append(
            {
                "id": rec["id"],
                "kind": rec.get("kind", "valid"),
                "row": (int(rec["row"]) if rec.get("row") is not None else None),
                "request": [dict(t) for t in rec["request"]],
                "redundant_factor": rec.get("redundant_factor"),
            }
        )
    return out


def deduplicate_terms(requests: Sequence[dict], suffix: str = "-dedup") -> list[dict]:
    """⭐ **DECLARED SENSITIVITY, ⛔ never the registered control.**

    The same requests with every *repeated* factor term after the first removed.
    Its only job is to isolate one arithmetic fact about R8's declared embedding
    rule: a ``valid_redundant`` (A7) request names one factor **twice at the same
    value**, and "both terms enter the composition" means its deviation is added
    **twice** — so an A7 request sits off the valid manifold in the same direction a
    ``duplicate_factor`` request does, by construction rather than by accident.
    ⛔ That is a property of the DECLARED rule, applied identically to the control
    and to C, so it is not a defect to be quietly patched; but a Hub reading an
    A7-controlled AUROC must be able to see how much of it is this doubling, which
    is what this set measures.
    """
    out = []
    for req in requests:
        seen, terms = set(), []
        for t in req["request"]:
            k = int(t["factor_index"])
            if k in seen:
                continue
            seen.add(k)
            terms.append(dict(t))
        out.append({**req, "id": f"{req['id']}{suffix}", "request": terms})
    return out


def load_query_set_registry(path: str | Path) -> dict:
    """Read ``query_sets.json`` — the per-kind declaration of C's structural controls.

    ⭐ **Why an arm reads this at all** (R26.1, standing rule).  A C kind whose
    records are structurally distinguishable from their control is scoring the
    control, not the arm: all 36 ``duplicate_factor`` records carry **7** terms
    against A/B's **6**, so a bare term counter scores AUROC **1.0000** on that
    half.  ``a7_redundant`` is the valid set built to erase exactly that giveaway,
    and the registry — ⛔ not this module, and ⛔ not the arm — declares which
    control belongs to which kind.  ``out_of_range``'s control is ``base``, the
    ordinary one-term-per-factor pool, which is why that kind is unaffected.
    """
    with open(path) as fh:
        return json.load(fh)


def control_map_of(registry: dict, invalid_set: str = "c_invalid") -> dict[str, str]:
    """``{C kind: the name of the valid set that controls it}`` from the registry."""
    try:
        spec = registry["sets"][invalid_set]
    except KeyError as exc:  # pragma: no cover - a malformed registry is a hard stop
        raise KeyError(
            f"query-set registry declares no {invalid_set!r} set; ⛔ the arms cannot "
            "know which valid pool controls which C kind"
        ) from exc
    return {str(k): str(v) for k, v in dict(spec.get("controls", {})).items()}


def load_registered_control_requests(
    registry_dir: str | Path, registry: dict, names: Sequence[str]
) -> dict[str, list[dict]]:
    """Load the FILE-backed control sets named in ``names`` (e.g. ``a7_redundant``).

    ⛔ A control that is registered but absent from disk raises — a silent skip is
    "declared not-run reported as null" in code form (R26.7).  ⛔ A control that is
    *derived* rather than file-backed (``base``, which is just the plain valid
    request pool) is **not** loaded here: the arms already hold it as their own
    A / B request lists, and building a second copy would let the two drift.
    """
    registry_dir = Path(registry_dir)
    out: dict[str, list[dict]] = {}
    for name in names:
        spec = registry.get("sets", {}).get(name)
        if spec is None:
            raise KeyError(
                f"{registry_dir / 'query_sets.json'}: control set {name!r} is named by "
                "a C kind but is not registered"
            )
        fname = spec.get("file")
        if not fname:
            continue  # derived (``base``) — held by the caller, never duplicated here
        path = registry_dir / str(fname)
        if not path.exists():
            raise FileNotFoundError(
                f"control set {name!r} is registered at {path} but the file is missing "
                "— ⛔ the A7-controlled column cannot be scored and must not be faked"
            )
        with open(path) as fh:
            out[name] = requests_from_valid_records(json.load(fh)["records"])
    return out


# ---------------------------------------------------------------------------
# ⭐ The kNN arm's embedding: per-factor prototypes, composed additively
# ---------------------------------------------------------------------------


def _extrapolating_index(values: np.ndarray, v: float) -> tuple[int, int, float]:
    """``(i, j, w)`` with ``proto = P[i] + w * (P[j] - P[i])`` for any real ``v``.

    On-grid values give ``w == 0`` (an exact hit).  Off-grid values *inside* the
    range interpolate; values outside extrapolate from the outermost pair, with
    ``|w| > 1``, so the embedding moves **further off-manifold the more invalid
    the request is** — which is the whole point of not clipping.
    """
    n = int(values.size)
    if n == 1:
        return 0, 0, 0.0
    hit = np.flatnonzero(values == v)
    if hit.size:
        return int(hit[0]), int(hit[0]), 0.0
    if v < values[0]:
        i, j = 0, 1
    elif v > values[-1]:
        i, j = n - 2, n - 1
    else:
        j = int(np.searchsorted(values, v))
        i = j - 1
    denom = float(values[j] - values[i])
    w = 0.0 if denom == 0.0 else float((v - values[i]) / denom)
    return i, j, w


@dataclass
class FactorPrototypes:
    """Per-(factor, value) mean train features + the additive composition rule.

    ⭐ **The composition rule, declared:**
    ``x̂(request) = μ_global + Σ_terms (μ_{k,v} − μ_global)``.
    One term per factor reproduces a first-order additive estimate of that
    combination's feature; a ``duplicate_factor`` request contributes **two**
    deviations for one factor and therefore lands between two incompatible
    prototypes; an ``out_of_range`` term contributes an extrapolated deviation
    whose norm grows with the distance outside the grid.

    ⚠ **Byte accounting.**  These prototypes are the kNN arm's *addressing head*.
    ``gen-store-grouping``'s ``store_bytes.json`` excludes the CLU's addressing
    head from its byte count by the same convention, so the two ledgers are
    like-for-like — but :attr:`n_bytes` is reported next to the store's anyway,
    because an unreported head is exactly how a byte-matched comparison gets
    quietly broken.
    """

    names: tuple
    values: tuple
    protos: tuple  # per factor: (C_k, D) float32
    mu_global: np.ndarray  # (D,) float32
    n_rows_fit: int
    n_per_value: tuple

    # -- construction --------------------------------------------------------
    @classmethod
    def fit(
        cls, grid, features: np.ndarray, rows: np.ndarray, min_per_value: int = 1
    ) -> "FactorPrototypes":
        """Mean cached **train** feature per (factor, value).

        ⛔ ``rows`` must be train rows: a prototype fitted on a held-out row would
        leak the very combination B is supposed to test.  The caller owns that
        (the experiment passes its store-train pool), and the count per value is
        recorded so a thin prototype is visible rather than silent.
        """
        X = np.asarray(features, np.float32)
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        fi = grid.factor_indices_of(np.asarray(rows, np.int64))
        if fi.shape[0] != X.shape[0]:
            raise ValueError(f"{fi.shape[0]} rows for {X.shape[0]} feature vectors")
        mu = X.mean(axis=0)
        protos, counts = [], []
        for k in range(grid.n_factors):
            c_k = int(grid.cardinalities[k])
            P = np.zeros((c_k, X.shape[1]), np.float32)
            n_k = np.zeros(c_k, np.int64)
            for c in range(c_k):
                m = fi[:, k] == c
                n_k[c] = int(m.sum())
                # ⛔ A value with no train row falls back to the global mean and
                # says so through ``n_per_value``; it is never silently dropped.
                P[c] = X[m].mean(axis=0) if n_k[c] >= int(min_per_value) else mu
            protos.append(P)
            counts.append(tuple(int(x) for x in n_k))
        return cls(
            names=tuple(grid.names),
            values=tuple(np.asarray(v, np.float64) for v in grid.values),
            protos=tuple(protos),
            mu_global=mu,
            n_rows_fit=int(X.shape[0]),
            n_per_value=tuple(counts),
        )

    # -- properties ----------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.mu_global.shape[0])

    @property
    def n_prototypes(self) -> int:
        return int(sum(p.shape[0] for p in self.protos))

    @property
    def n_bytes(self) -> int:
        """Bytes of the prototype table **as deployed** (float32), + the global mean."""
        return int(
            sum(p.size * p.dtype.itemsize for p in self.protos)
            + self.mu_global.size * self.mu_global.dtype.itemsize
        )

    def byte_ledger(self) -> dict:
        return {
            "n_prototypes": self.n_prototypes,
            "dim": self.dim,
            "dtype": "float32",
            "bytes_formula": "(sum_k C_k + 1) * D * 4",
            "bytes": self.n_bytes,
            "role": (
                "request embedding == the kNN arm's ADDRESSING HEAD; the store's own "
                "byte count excludes the CLU addressing head by the same convention"
            ),
        }

    # -- the embedding -------------------------------------------------------
    def prototype(self, factor_index: int, value: float) -> np.ndarray:
        """One term's prototype, interpolating/extrapolating off-grid values."""
        k = int(factor_index)
        i, j, w = _extrapolating_index(self.values[k], float(value))
        P = self.protos[k]
        if w == 0.0:
            return P[i]
        return P[i] + np.float32(w) * (P[j] - P[i])

    def embed(self, request: Sequence[dict]) -> Embedding:
        """Compose one request.  ⭐ Every term contributes, including a duplicate."""
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
        """How many terms were off-grid, and how far outside — a provenance check.

        ⭐ If this is 0 for the C set, the ``out_of_range`` half of C has been
        silently clipped onto the grid somewhere and the arm is scoring a
        satisfiable request.  Reported, not assumed.
        """
        n_terms = n_off = 0
        max_outside = 0.0
        for req in requests:
            for term in (req["request"] if isinstance(req, dict) else req):
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


# ---------------------------------------------------------------------------
# ⭐ The dense form the LACE arm consumes (same requests, different consumer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseRequests:
    """A batch of requests as fixed-shape arrays, so an energy can be vmapped.

    Every request contributes one term per factor plus **at most one extra term**
    (the ``duplicate_factor`` kind's second value for a factor).  That is enough
    to express both A/B and C without a ragged structure.

    Attributes:
        cls_targets: ``(N, F)`` nearest grid class per factor.  ⚠ For an
            ``out_of_range`` value this is the **nearest** class: LACE's
            categorical head has a **closed vocabulary** and *cannot express* an
            off-grid value at all (R9.7).  That is LACE's structural limitation,
            declared here rather than papered over — the *continuous* head is
            the form that sees the violation.
        val_targets: ``(N, F)`` the requested value, **min-max normalised to
            [0, 1] against that factor's realised grid range** (LACE normalises
            its continuous attributes; an off-grid value lands outside [0, 1]).
        extra_factor: ``(N,)`` factor index of the duplicate term, ``-1`` if none.
        extra_cls / extra_val: that term's targets, in the same two forms.
    """

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
    lo, hi = float(values[0]), float(values[-1])
    span = hi - lo
    return float((v - lo) / span) if span != 0.0 else 0.0


def dense_request_arrays(grid, requests: Sequence[dict]) -> DenseRequests:
    """Pack request dicts into :class:`DenseRequests` (⛔ no request is modified)."""
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
                        f"request {req.get('id')!r} has more than one duplicate term; "
                        "the dense form carries exactly one extra slot"
                    )
                ex_f[i], ex_c[i], ex_v[i] = k, c, nv
            else:
                seen.add(k)
                cls_t[i, k], val_t[i, k] = c, nv
    return DenseRequests(cls_t, val_t, ex_f, ex_c, ex_v, n_f)


def request_labels(requests: Sequence[dict], grid) -> Optional[np.ndarray]:
    """``(N, F)`` probe labels for **valid** requests; ``None`` if any is invalid.

    ⛔ A C request has no ground-truth image, so it has no label and is never
    scored for composition — only for refusal.
    """
    if any((r.get("kind") != "valid") for r in requests):
        return None
    return np.asarray([r["class_indices"] for r in requests], np.int32)
