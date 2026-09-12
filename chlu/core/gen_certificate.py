"""⭐ **The composition certificate** — a refusal criterion DERIVED from the store,
with provably zero invalid-example supervision.

⭐ DIAL DECLARATION — *admission*.  The question is whether a request can be refused
from the store's own geometry, rather than by a head trained to refuse.  Nothing in
this module is fitted to a refusal target, and nothing in it ever sees an invalid
example: every quantity below is a function of (i) the request, (ii) the modes of a
:class:`chlu.core.gen_store.GenStore`, (iii) that store's own addressing over
**valid train rows**, and (iv) the factor grid's realised support.

**The four derived components** (all from quantities the store already exposes):

``reach``
    ``E_reach(R) = sum_t (dmin_t / (s_j / 2))^2`` — the squared distance, in units of
    the factor's own Voronoi radius, from each requested value to the nearest value
    the store has ever been addressed at.  ⛔ **Identically 0** for every on-grid
    request, which is every A request, every B request *and* every
    ``duplicate_factor`` C record.  It is a detector of ``out_of_range`` and nothing
    else, and it is reported as such.
``pair``
    ``chi(t, t') = -sum_k log sum_m q_t[k,m] q_t'[k,m] / pbar[k,m]`` — the
    product-of-experts compatibility of two request terms, measured on the store's
    own per-well mode histograms against its own mode prior.  Two values of the
    *same* factor put their mass on disjoint modes in the wells that carry that
    factor, so the overlap collapses and ``chi`` explodes.  ⭐ This is the only
    component that can see a ``duplicate_factor`` conflict, so the mechanism claim
    lives or dies here.
``curv``
    The composed request is placed at the product-of-experts point
    ``xhat_k = sum_m r[k,m] mode_{k,m}`` and the **Hessian of the store's own soft
    energy** is evaluated there in closed form.  With
    ``Phi(x) = -2 tau^2 log sum_m exp(-||x - mu_m||^2 / 2 tau^2)``,

        ``grad Phi = 2 (x - mubar_w)``,  ``Hess Phi = 2 I - (2 / tau^2) Cov_w(mu)``

    so ``lambda_min = 2 - (2/tau^2) lambda_max(Cov_w)`` exactly — no autodiff, no
    ``(d x d)`` matrix (``lambda_max`` comes from the ``M x M`` Gram).  A request
    whose composed point sits *between* distant modes is on a ridge: ``Cov_w`` is
    large and ``lambda_min`` plunges.  That is the barrier, in the literal sense.
    ⭐ The closed form is verified against finite differences in
    ``tests/test_gen_certificate.py::test_hessian_closed_form_matches_finite_differences``.
``energy``
    ``E_store = sum_k min_m ||xhat_k - mu_{k,m}||^2 / D`` — the store's own
    quantisation energy of the composed point.

⛔ **The supervision rule is asserted, not promised.**  Every threshold and every
normaliser is fitted through :class:`chlu.eval.gen_cert_eval.ValidOnlyFitter`, which
records a hash of exactly the ids it saw and refuses to certify a run in which any C
record id appears.  See :func:`assert_no_decoder_in_path` for the second structural
guard: the certificate path contains **no decoder**, so the superseded-recipe trap
(``dec_recipe="baseline"``) cannot reach a number computed here.

⚠ §7.23: every entry point casts its input to the **store's own parameter dtype**
(``chlu.core.gen_bridge`` is the shipped exemplar); a module that sets
``jax_enable_x64`` before this one cannot change a result here.

Runnable through :mod:`chlu.experiments.exp_gen_certificate`.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .gen_store import GenStore, address_codes

#: The request sources.  ``A`` = seen-combination held-back rows, ``B`` = the
#: held-out valid combinations, ``C`` = the structurally invalid requests, read
#: VERBATIM from ``c_invalid.json`` (R2) and never regenerated.
SOURCES = ("A", "A7", "B", "C")
#: The C kinds, exactly as ``gen-data-splits`` defines them.
C_KINDS = ("duplicate_factor", "out_of_range")
#: ⭐ Components of the derived certificate, in the declared order.  ``reach`` is
#: dimensionless by construction and enters unstandardised; the rest are z-scored on
#: the A set through the audited fitter.
COMPONENTS = ("reach", "pair", "curv", "energy")
#: The arms this module can score.  ``clu`` is the derived certificate; the others
#: are controls and are ⛔ never optional.
ARMS = ("clu", "distance", "distance_reach", "count", "reach")

__all__ = [
    "ARMS",
    "COMPONENTS",
    "C_KINDS",
    "SOURCES",
    "Request",
    "TermStatistics",
    "assert_no_decoder_in_path",
    "certificate_components",
    "compose",
    "composed_point",
    "factor_support",
    "load_c_requests",
    "naive_distance",
    "pair_conflict",
    "reach_energy",
    "redundant_requests",
    "requests_from_rows",
    "score_requests",
    "self_conflict_is_non_positive",
    "store_energy",
    "term_distribution",
    "well_curvature",
]


# ---------------------------------------------------------------------------
# 1. Requests — one representation, identical in form for A, B and C
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """A factor request: an id, and a list of ``(factor_index, value)`` terms.

    ⛔ A request is **not** a row.  A and B requests happen to be realisable and are
    built from the grid's own values; a C request is read verbatim from
    ``c_invalid.json`` and may name the same factor twice or a value off the grid.
    The three sets are represented identically so no arm can tell them apart by
    their container.
    """

    id: str
    source: str
    terms: tuple  # tuple[(factor_index:int, value:float)]
    kind: str = "valid"
    violated_factor: Optional[str] = None
    row: Optional[int] = None

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def factors(self) -> tuple:
        return tuple(int(j) for j, _ in self.terms)


def requests_from_rows(grid, rows, source: str) -> list:
    """Build one request per dataset row from the grid's realised **values**.

    ⛔ Goes row -> factor indices -> grid values (never the cached latent array), the
    same reconstruction path the hold-out assertion uses.
    """
    rows = np.asarray(rows, np.int64)
    vals = grid.values_of(rows)
    out = []
    for i, r in enumerate(rows):
        terms = tuple((int(j), float(vals[i, j])) for j in range(vals.shape[1]))
        out.append(Request(id=f"{source}:{int(r)}", source=str(source), terms=terms,
                           kind="valid", row=int(r)))
    return out


def redundant_requests(requests: Sequence[Request], n_factors: int) -> list:
    """⭐ **The term-count control**, built from VALID requests only.

    Each request is given one extra term that repeats a factor **at the value it
    already has**.  The result is still perfectly satisfiable — it is the same image
    — but it carries the same number of terms as a ``duplicate_factor`` C record.

    ⛔ This is why it exists: the C ``duplicate_factor`` records carry 7 terms while A
    and B carry 6, so *any* term-count-sensitive statistic separates them for free
    (the ``count`` control measures exactly that, at AUROC 1.000).  Scoring C's
    duplicates against **these** requests instead of against A asks the only question
    that matters — does the barrier fire on the **contradiction**, or on the length?

    ⚠ No invalid example is used or synthesised: the construction is a function of a
    valid request alone, which is what keeps the supervision claim intact.
    """
    out = []
    for i, r in enumerate(requests):
        j = int(i % int(n_factors))
        dup_terms = tuple(t for t in r.terms if int(t[0]) == j)
        if not dup_terms:
            continue
        out.append(Request(id=f"A7:{r.id.split(':', 1)[-1]}", source="A7",
                           terms=tuple(r.terms) + (dup_terms[0],),
                           kind="valid_redundant", row=r.row))
    return out


def load_c_requests(path, factor_names: Sequence[str]) -> list:
    """⛔ Read the C set VERBATIM from ``c_invalid.json``.  Never regenerate it.

    The file is single-sourced by ``gen-data-splits`` and byte-identical across every
    arm (R2).  This function validates the schema and the factor-name/-index
    agreement and then copies the records through untouched.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    schema = payload.get("schema", {})
    if int(schema.get("version", -1)) != 1:
        raise ValueError(f"{path}: unexpected c_invalid schema {schema.get('version')!r}")
    names = list(payload["factors"]["names"])
    if names != list(factor_names):
        raise ValueError(
            f"{path}: factor names {names} do not match the loaded grid "
            f"{list(factor_names)} — ⛔ the C set is single-sourced and must not be "
            "re-expressed"
        )
    out = []
    for rec in payload["records"]:
        terms = []
        for t in rec["request"]:
            j = int(t["factor_index"])
            if names[j] != t["factor"]:
                raise ValueError(f"{path}: record {rec['id']} disagrees on factor {j}")
            terms.append((j, float(t["value"])))
        if rec["kind"] not in C_KINDS:
            raise ValueError(f"{path}: unknown C kind {rec['kind']!r}")
        out.append(Request(id=str(rec["id"]), source="C", terms=tuple(terms),
                           kind=str(rec["kind"]),
                           violated_factor=str(rec["violated_factor"])))
    return out


# ---------------------------------------------------------------------------
# 2. The factor support and the reach term
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSupport:
    """The realised value support of each factor, and its Voronoi radius.

    ``values[j]`` is the factor's realised grid (from the file, never a
    ``linspace``); ``radius[j] = median(diff(values[j])) / 2`` is the half-spacing —
    the distance at which a requested value stops being *the* nearest realised value
    and starts being *between* two of them.  ⛔ Nothing here is tuned: both are
    properties of the valid data's own support.
    """

    values: tuple
    radius: tuple
    names: tuple

    def nearest(self, j: int, u: float) -> tuple:
        """``(nearest value index, |u - nearest value|)`` for factor ``j``."""
        v = self.values[int(j)]
        d = np.abs(v - float(u))
        i = int(np.argmin(d))
        return i, float(d[i])


def factor_support(grid, radius_frac: float = 0.5) -> FactorSupport:
    """Build the support from a :class:`chlu.data.gen_splits.FactorGrid`."""
    vals, rad = [], []
    for v in grid.values:
        v = np.asarray(v, np.float64)
        vals.append(v)
        step = float(np.median(np.diff(v))) if v.size > 1 else 1.0
        rad.append(max(float(radius_frac) * step, 1e-12))
    return FactorSupport(tuple(vals), tuple(rad), tuple(grid.names))


def reach_energy(request: Request, support: FactorSupport) -> dict:
    """``sum_t (dmin_t / radius_j)^2`` — 0 for an on-grid request, > 0 off it.

    ⛔ Reported honestly: this component is **identically zero** on A, on B and on
    every ``duplicate_factor`` C record, so it detects ``out_of_range`` and nothing
    else.  It is kept because refusing an unrealisable *value* is a real admission
    decision, not because it carries the mechanism claim.
    """
    per, worst_j, worst = [], -1, 0.0
    for j, u in request.terms:
        _, d = support.nearest(j, u)
        e = (d / support.radius[int(j)]) ** 2
        per.append(e)
        if e > worst:
            worst, worst_j = e, int(j)
    return {"reach": float(np.sum(per)), "reach_max": float(max(per) if per else 0.0),
            "reach_argmax_factor": int(worst_j), "per_term": [float(x) for x in per]}


# ---------------------------------------------------------------------------
# 3. Store-side term statistics — the store's own addressing over VALID rows
# ---------------------------------------------------------------------------


@dataclass
class TermStatistics:
    """Per-``(factor, grid value)`` mode histograms from the store's own addressing.

    ``dist[j][v]`` is ``(K, M)``, row-normalised per well, Laplace-smoothed by
    ``alpha``; ``prior`` is ``(K, M)``, the marginal mode usage.  ``proto[j][v]`` is
    the mean **feature** of the same rows and exists only to feed the naive
    distance-to-store control (R8's kNN-style composition).

    ⛔ Built from valid train rows **only** — the rows the store itself was fitted
    on.  No held-out row, no eval row, and (structurally) no invalid example: a C
    record has no rows at all.
    """

    dist: list
    prior: np.ndarray
    proto: list
    counts: list
    n_rows: int
    n_wells: int
    n_modes: int
    alpha: float
    source_hash: str = ""

    def term(self, j: int, v: int) -> np.ndarray:
        return self.dist[int(j)][int(v)]


def term_distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    """``(K, M)`` counts -> Laplace-smoothed per-well distribution."""
    c = np.asarray(counts, np.float64)
    c = c + float(alpha)
    return c / c.sum(axis=1, keepdims=True)


def build_term_statistics(
    codes: np.ndarray,
    factor_indices: np.ndarray,
    cardinalities: Sequence[int],
    features: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    row_ids: Optional[Sequence[int]] = None,
) -> TermStatistics:
    """Histogram the store's codes conditional on each ``(factor, value)``.

    Args:
        codes: ``(N, K)`` integer mode indices from
            :func:`chlu.core.gen_store.address_codes` — ⭐ the store's own addressing
            head, which is what R8 assigns to the CLU arm.
        factor_indices: ``(N, F)`` factor class indices of the same rows.
        features: optional ``(N, D)``; used ONLY for the naive control's prototypes.
    """
    codes = np.asarray(codes, np.int64)
    fi = np.asarray(factor_indices, np.int64)
    n, k = codes.shape
    m = int(codes.max()) + 1
    m = max(m, 1)
    prior_counts = np.zeros((k, m), np.float64)
    for kk in range(k):
        prior_counts[kk] = np.bincount(codes[:, kk], minlength=m).astype(np.float64)
    prior = term_distribution(prior_counts, alpha)

    dist, proto, counts = [], [], []
    for j, card in enumerate(cardinalities):
        dj, pj, cj = [], [], []
        for v in range(int(card)):
            sel = fi[:, j] == v
            cc = np.zeros((k, m), np.float64)
            for kk in range(k):
                if sel.any():
                    cc[kk] = np.bincount(codes[sel, kk], minlength=m).astype(np.float64)
            dj.append(term_distribution(cc, alpha))
            cj.append(int(sel.sum()))
            if features is not None:
                pj.append(np.asarray(features[sel], np.float64).mean(axis=0)
                          if sel.any() else np.zeros(features.shape[1], np.float64))
        dist.append(dj)
        counts.append(cj)
        proto.append(pj)
    h = ""
    if row_ids is not None:
        ids = np.sort(np.asarray(row_ids, np.int64))
        h = hashlib.sha256(ids.tobytes()).hexdigest()
    return TermStatistics(dist=dist, prior=prior, proto=proto, counts=counts,
                          n_rows=int(n), n_wells=int(k), n_modes=int(m),
                          alpha=float(alpha), source_hash=h)


def term_vectors(request: Request, stats: TermStatistics, support: FactorSupport) -> list:
    """The ``(K, M)`` distribution of each term, snapped to the nearest grid value.

    ⚠ **Snapping is deliberate and is what makes the decomposition honest**: the
    distributional half of the certificate treats an out-of-range value as its
    nearest realisable one, so ``out_of_range`` is caught by ``reach`` *alone* and
    ``duplicate_factor`` by the composition barrier *alone*.  Blending them with a
    kernel would let one component launder the other's credit.
    """
    out = []
    for j, u in request.terms:
        v, _ = support.nearest(j, u)
        out.append(stats.term(j, v))
    return out


# ---------------------------------------------------------------------------
# 4. The composition barrier
# ---------------------------------------------------------------------------


def pair_conflict(q1: np.ndarray, q2: np.ndarray, prior: np.ndarray) -> float:
    """``-sum_k log sum_m q1 q2 / pbar`` — 0 for terms that are independent given the
    store's mode prior, large and positive for terms whose mass is disjoint.

    ⭐ This is the whole mechanism: two values of the same factor address different
    modes in the wells that carry that factor, so the product-of-experts overlap
    collapses.  ⛔ No invalid example is involved in defining it.
    """
    ov = np.sum(np.asarray(q1) * np.asarray(q2) / np.asarray(prior), axis=1)
    return float(-np.sum(np.log(np.maximum(ov, 1e-300))))


def self_conflict_is_non_positive(q: np.ndarray, prior: np.ndarray) -> bool:
    """``chi(t, t) <= 0`` for every term — the identity that makes ``pair`` immune to a
    redundant term.

    By Cauchy-Schwarz, ``sum_m q^2 / pbar >= (sum_m q)^2 / sum_m pbar = 1``, so
    ``chi(t,t) = -sum_k log(...) <= 0``.  ⭐ Repeating a term therefore adds only
    (a) that non-positive self-pair and (b) copies of pair values the request already
    had, so ``max over pairs`` is **unchanged** whenever it was already non-negative.
    That is why ``pair`` — unlike the composed-point components — cannot inherit the C
    set's 7-terms-vs-6 giveaway.  Asserted in
    ``tests/test_gen_certificate.py::test_the_pair_barrier_is_invariant_to_a_redundant_term``.
    """
    return pair_conflict(q, q, prior) <= 1e-12


def compose(qs: Sequence[np.ndarray], prior: np.ndarray) -> tuple:
    """Product-of-experts composition against the prior: ``r ∝ pbar * prod_t (q_t/pbar)``.

    Returns ``(r, log_Z)`` with ``r`` ``(K, M)`` row-normalised.  ``log_Z`` (the
    per-well log normaliser, summed) is the full composition free energy; it is
    reported but ⚠ it grows with the number of terms, so it is **not** the headline
    statistic (the C ``duplicate_factor`` records carry 7 terms against A/B's 6, and
    a term-count-sensitive statistic would launder that giveaway as a result).
    """
    prior = np.asarray(prior, np.float64)
    log_r = np.log(np.maximum(prior, 1e-300))
    for q in qs:
        log_r = log_r + (np.log(np.maximum(np.asarray(q, np.float64), 1e-300))
                         - np.log(np.maximum(prior, 1e-300)))
    mx = log_r.max(axis=1, keepdims=True)
    ex = np.exp(log_r - mx)
    z = ex.sum(axis=1, keepdims=True)
    r = ex / z
    log_z = float(np.sum(np.log(z) + mx))
    return r, log_z


def composed_point(store: GenStore, r: np.ndarray) -> list:
    """``xhat_k = sum_m r[k,m] mode_{k,m}`` — one vector per well, in the store's dtype."""
    dt = np.asarray(store.modes[0]).dtype
    out = []
    for k, m in enumerate(store.modes):
        mu = np.asarray(m, np.float64)
        out.append((np.asarray(r[k], np.float64) @ mu).astype(dt))
    return out


def well_curvature(store: GenStore, xhat: Sequence[np.ndarray], tau: float) -> dict:
    """⭐ Closed-form Hessian of the store's own soft energy at the composed point.

    ``Phi_k(x) = -2 tau^2 log sum_m exp(-||x - mu_m||^2 / (2 tau^2))`` — the same
    ``softmax(-d2 / (2 tau^2))`` occupancy the store is trained with — has

        ``Hess Phi_k = 2 I - (2 / tau^2) Cov_w(mu_k)``

    so ``lambda_min = 2 - (2/tau^2) lambda_max(Cov_w)``.  ``lambda_max`` is taken
    from the ``(M, M)`` Gram of the weighted, centred modes, so no ``(d, d)`` matrix
    is ever formed.  ⚠ A negative ``lambda_min`` means the composed point sits on a
    **ridge between modes** — that is the barrier the certificate reads.
    """
    tau = float(tau)
    lams, lmins, ents = [], [], []
    for k, m in enumerate(store.modes):
        mu = np.asarray(m, np.float64)
        x = np.asarray(xhat[k], np.float64)
        d2 = np.sum((mu - x[None, :]) ** 2, axis=1)
        lw = -d2 / (2.0 * tau * tau)
        lw = lw - lw.max()
        w = np.exp(lw)
        w = w / w.sum()
        mubar = w @ mu
        a = np.sqrt(w)[:, None] * (mu - mubar[None, :])
        gram = a @ a.T
        lam = float(np.linalg.eigvalsh(gram)[-1])
        lams.append(lam)
        lmins.append(2.0 - (2.0 / (tau * tau)) * lam)
        nz = w[w > 0]
        ents.append(float(-(nz * np.log2(nz)).sum()))
    return {
        "lambda_max_cov": [float(x) for x in lams],
        "lambda_min_hess": [float(x) for x in lmins],
        "occupancy_entropy_bits": [float(x) for x in ents],
        "curv": float(max(-x for x in lmins)),
        "curv_mean": float(np.mean([-x for x in lmins])),
    }


def store_energy(store: GenStore, xhat: Sequence[np.ndarray]) -> float:
    """``sum_k min_m ||xhat_k - mu_{k,m}||^2 / D`` — the store's own quantisation energy."""
    tot = 0.0
    for k, m in enumerate(store.modes):
        mu = np.asarray(m, np.float64)
        x = np.asarray(xhat[k], np.float64)
        tot += float(np.min(np.sum((mu - x[None, :]) ** 2, axis=1)))
    return tot / float(store.dim)


def certificate_components(
    request: Request,
    store: GenStore,
    stats: TermStatistics,
    support: FactorSupport,
    tau: float,
) -> dict:
    """Every derived component for one request.  ⛔ No fitting, no threshold, no label
    of validity enters here — this is a deterministic function of the request and the
    store."""
    qs = term_vectors(request, stats, support)
    chis = []
    pair_arg = (-1, -1)
    best = -np.inf
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            c = pair_conflict(qs[i], qs[j], stats.prior)
            chis.append(c)
            if c > best:
                best, pair_arg = c, (i, j)
    r, log_z = compose(qs, stats.prior)
    xhat = composed_point(store, r)
    curv = well_curvature(store, xhat, tau)
    reach = reach_energy(request, support)
    fa = int(request.terms[pair_arg[0]][0]) if pair_arg[0] >= 0 else -1
    fb = int(request.terms[pair_arg[1]][0]) if pair_arg[1] >= 0 else -1
    return {
        "id": request.id,
        "source": request.source,
        "kind": request.kind,
        "violated_factor": request.violated_factor,
        "n_terms": int(request.n_terms),
        "reach": reach["reach"],
        "reach_argmax_factor": reach["reach_argmax_factor"],
        "pair": float(best if np.isfinite(best) else 0.0),
        "pair_mean": float(np.mean(chis)) if chis else 0.0,
        "pair_argmax_factors": [fa, fb],
        "curv": curv["curv"],
        "curv_mean": curv["curv_mean"],
        "energy": store_energy(store, xhat),
        "log_z": float(log_z),
        "composed_entropy_bits": float(
            np.mean([-(p[p > 0] * np.log2(p[p > 0])).sum() for p in r])
        ),
    }


# ---------------------------------------------------------------------------
# 5. The laundering control — plain distance to the store on the same features
# ---------------------------------------------------------------------------


def naive_distance(
    request: Request, store: GenStore, stats: TermStatistics, support: FactorSupport
) -> float:
    """⭐ **The trivial in-house control** (the task's laundering check).

    R8's kNN-style composition — the mean of the per-term **prototype features** —
    followed by a plain distance to the store: ``sum_k min_m ||xbar_k - mu||^2 / D``.
    No product of experts, no prior correction, no curvature, no reach.  If this
    scores as well as the derived certificate, the derivation is decoration over
    naive geometry and the spoke says so.

    ⚠ It is scored on the SAME features and the SAME store, so the comparison is not
    confounded by representation.
    """
    if not stats.proto or not stats.proto[0]:
        raise ValueError(
            "the naive control needs prototype FEATURES — build the term statistics "
            "with features=... (⛔ it must be scored on the same features as the "
            "derived certificate)"
        )
    vecs = []
    for j, u in request.terms:
        v, _ = support.nearest(j, u)
        vecs.append(stats.proto[int(j)][int(v)])
    xbar = np.mean(np.stack(vecs, axis=0), axis=0)
    tot = 0.0
    for k, m in enumerate(store.modes):
        mu = np.asarray(m, np.float64)
        x = xbar[np.asarray(store.well_channels[k], np.int64)]
        tot += float(np.min(np.sum((mu - x[None, :]) ** 2, axis=1)))
    return tot / float(store.dim)


# ---------------------------------------------------------------------------
# 6. Scoring a whole request set
# ---------------------------------------------------------------------------


def score_requests(
    requests: Sequence[Request],
    store: GenStore,
    stats: TermStatistics,
    support: FactorSupport,
    tau: float,
    with_naive: bool = True,
    log: Optional[Callable] = None,
    log_every: int = 256,
) -> list:
    """Per-request rows of every component and every control.  ⛔ Order preserved."""
    rows = []
    for i, req in enumerate(requests):
        row = certificate_components(req, store, stats, support, tau)
        row["n_terms"] = int(req.n_terms)
        if with_naive:
            row["dist"] = naive_distance(req, store, stats, support)
        rows.append(row)
        if log is not None and log_every and (i + 1) % int(log_every) == 0:
            log(f"    scored {i + 1}/{len(requests)} requests")
    return rows


# ---------------------------------------------------------------------------
# 7. The structural guard — no decoder can reach a number computed here
# ---------------------------------------------------------------------------


def assert_no_decoder_in_path() -> dict:
    """⭐ Prove the certificate path contains **no decoder**.

    The wave's standing trap is that ``GenBridgeConfig.dec_recipe`` still defaults to
    the SUPERSEDED ``"baseline"`` recipe, and an arm decoded through it is void while
    looking perfectly normal in the logs.  The certificate never decodes anything, so
    the trap cannot reach it — but that is a claim until it is checked, so it is
    checked: no public entry point here accepts a ``decoder`` argument, and this
    module imports no decoding symbol.

    ⛔ Structural, like ``gen_store.assert_no_bypass``: a future edit that adds a
    decoder argument fails here rather than silently at cell time.
    """
    import chlu.core.gen_certificate as mod

    bad = []
    # leg 1 — runtime signatures.  Catches a monkeypatched or re-bound entry point,
    # which is exactly what the positive-control test does.
    entry = ("certificate_components", "score_requests", "naive_distance",
             "build_term_statistics", "well_curvature", "store_energy")
    for name in entry:
        fn = getattr(mod, name)
        for p in inspect.signature(fn).parameters:
            if "decod" in p.lower() or "probe" in p.lower():
                bad.append(f"signature {name}({p})")
    # leg 2 — the SOURCE, by AST, so a decoder named only in a docstring or a comment
    # cannot trip it and a decoder actually imported or called cannot hide in one.
    tree = ast.parse(inspect.getsource(mod))
    forbidden = ("decod", "dec_recipe", "load_model")
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names] + [a.asname or "" for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        for nm in names:
            low = str(nm).lower()
            if any(f in low for f in forbidden):
                bad.append(f"source reference to {nm!r}")
    if bad:
        raise AssertionError(
            "the certificate path must contain no decoder (the superseded "
            f"dec_recipe='baseline' trap cannot be allowed to reach it); found "
            f"{sorted(set(bad))}"
        )
    return {"decoder_in_certificate_path": False,
            "checked_entry_points": len(entry),
            "legs": ["runtime signatures", "module AST (docstrings and comments excluded)"],
            "note": ("the certificate is a function of the request, the store's modes "
                     "and the store's own addressing over valid train rows; nothing is "
                     "decoded, so no decoder recipe can change a number here")}


# ---------------------------------------------------------------------------
# 8. Addressing helper — codes for the store's own training rows
# ---------------------------------------------------------------------------


def codes_for(store: GenStore, features: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Batched :func:`chlu.core.gen_store.address_codes` (§7.23: cast to the store's dtype)."""
    dt = np.asarray(store.modes[0]).dtype
    x = np.asarray(features, dt).reshape((np.asarray(features).shape[0], -1))
    out = []
    for i in range(0, x.shape[0], int(batch)):
        out.append(address_codes(store, x[i : i + int(batch)]))
    if not out:
        return np.zeros((0, store.n_wells), np.int64)
    return np.concatenate(out, axis=0)
