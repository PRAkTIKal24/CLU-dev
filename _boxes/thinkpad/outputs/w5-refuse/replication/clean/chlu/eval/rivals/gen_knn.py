"""⭐ The **kNN-over-features** competitor arm, byte-matched to the CLU store.

⛔ **Read this before touching anything below.**  This arm is the comparator for
**two** kill rules — (ii) on B and (iv) on C.  ⭐ A weak kNN makes both pass
trivially and lets a dead mechanism through, so a hobbled baseline here does not
cost us a fair fight, **it costs us the truth, in the direction that flatters
us.**  Every choice in this module is made to let the kNN win if it can, and a
kNN that beats the CLU is the control working and gets reported unsoftened.

**What the arm is.**  A request is embedded by the declared R8 rule
(:class:`chlu.core.gen_request.FactorPrototypes` — composed per-factor prototypes
from the cached train features).  The embedding then:

* **generates** by retrieving its ``k`` nearest **store** entries and averaging
  them, and decoding that through THE one shared frozen decoder.  ⭐ Exactly as
  in ``gen_store.read_from_codes``, the *request embedding never reaches the
  decoder* — only the retrieved store content does — so the read channel is
  ``k·log2(n_entries)`` bits and the comparison with the CLU's 30-bit channel is
  structural, not rhetorical;
* **refuses** by thresholding **distance to the store**, with the threshold set
  on **valid data only** — the same rule that binds our own certificate.

**Byte matching (⭐ shrink to match, never grow, never hand-solved).**
:func:`entries_for_budget` inverts the deployed byte formula and floors, so the
arm is always **≤** the store's budget and never over it.  ⛔ There is no dtype
normalisation: a competitor deploying float16 or int8 genuinely fits *more*
entries in the same bytes, and refusing them that would be hobbling.  Three
deployments are swept; each one's realised ledger is reported next to the
store's.

**The second family member, declared not smuggled.**  ⭐ The prototype table is
itself ``(Σ_k C_k + 1)·D·4`` bytes — **466,944 B for Shapes3D, already inside the
524,288 B budget.**  So "decode the composed prototype directly"
(:func:`prototype_direct_read`) is a *legitimate byte-matched competitor*, and
declining to build it would be precisely the hobbling this module exists to
avoid.  It is reported as its own arm with an **unbounded** read channel (the
full ``D``-float vector reaches the decoder) so nobody can mistake it for the
bottlenecked one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: Deployments swept by the arm.  ⛔ Bytes are counted **as deployed**; there is
#: no dtype normalisation, because a real competitor would quantise.
DEPLOYMENTS = ("fp32", "fp16", "int8")
METRICS = ("euclidean", "cosine")
NORMALISERS = ("none", "zscore")


# ---------------------------------------------------------------------------
# Byte ledger
# ---------------------------------------------------------------------------


def deployment_overhead_bytes(deployment: str, dim: int) -> int:
    """Bytes the deployment costs **before** a single entry is stored.

    ``int8`` carries a per-dimension float32 scale and offset; ⛔ those are part
    of what ships, so they are inside the budget, not beside it.
    """
    if deployment in ("fp32", "fp16"):
        return 0
    if deployment == "int8":
        return int(2 * dim * 4)
    raise ValueError(f"unknown deployment {deployment!r} (have {DEPLOYMENTS})")


def bytes_per_entry(deployment: str, dim: int) -> int:
    itemsize = {"fp32": 4, "fp16": 2, "int8": 1}
    if deployment not in itemsize:
        raise ValueError(f"unknown deployment {deployment!r} (have {DEPLOYMENTS})")
    return int(dim * itemsize[deployment])


def entries_for_budget(budget_bytes: int, dim: int, deployment: str) -> int:
    """⭐ **Shrink to match.**  Largest entry count that fits ``budget_bytes``.

    Floored, so the realised ledger is always ``<=`` the store's.  ⛔ Never
    rounded up, never re-solved by hand.
    """
    usable = int(budget_bytes) - deployment_overhead_bytes(deployment, dim)
    if usable <= 0:
        raise ValueError(
            f"deployment {deployment!r} overhead already exceeds the {budget_bytes} B budget"
        )
    n = usable // bytes_per_entry(deployment, dim)
    if n < 1:
        raise ValueError(f"budget {budget_bytes} B holds no {deployment} entry at D={dim}")
    return int(n)


def byte_ledger(n_entries: int, dim: int, deployment: str, budget_bytes: int) -> dict:
    """The realised ledger, reported next to ``store_bytes.json``'s."""
    per = bytes_per_entry(deployment, dim)
    over = deployment_overhead_bytes(deployment, dim)
    total = int(n_entries) * per + over
    return {
        "deployment": deployment,
        "n_entries": int(n_entries),
        "dim": int(dim),
        "bytes_per_entry": per,
        "overhead_bytes": over,
        "bytes_formula": "n_entries * D * itemsize + overhead",
        "realised_bytes": int(total),
        "budget_bytes": int(budget_bytes),
        "slack_bytes": int(int(budget_bytes) - total),
        "within_budget": bool(total <= int(budget_bytes)),
        "bits_per_read_formula": "k * log2(n_entries)",
    }


# ---------------------------------------------------------------------------
# The store: k-means, then quantised as deployed
# ---------------------------------------------------------------------------


def _kmeanspp(X: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (⛔ local to this arm; the CLU store is never imported)."""
    n = int(X.shape[0])
    idx = [int(rng.integers(n))]
    d2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for _ in range(1, int(m)):
        total = float(d2.sum())
        p = (d2 / total) if total > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((X - X[j]) ** 2, axis=1))
    return X[np.asarray(idx, np.int64)].astype(np.float32)


def kmeans_store(
    X: np.ndarray, n_entries: int, seed: int, iters: int = 25, chunk: int = 4096
) -> tuple:
    """Lloyd's algorithm from a k-means++ seed -> ``(centroids, info)``.

    ⭐ k-means, not a random subsample: with only tens-to-hundreds of entries the
    summary quality *is* the arm's ceiling, and a competitor who wanted to win
    would obviously cluster.  (The random subsample is available through
    ``method="subsample"`` and loses; it is not the shipped default.)
    """
    X = np.asarray(X, np.float32)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    n = int(X.shape[0])
    m = min(int(n_entries), n)
    rng = np.random.default_rng(int(seed) * 6151 + 20_260_911)
    C = _kmeanspp(X, m, rng)
    inertia = float("nan")
    for _ in range(int(iters)):
        assign = np.empty(n, np.int64)
        tot = 0.0
        cn = np.sum(C * C, axis=1)
        for i in range(0, n, chunk):
            Xi = X[i : i + chunk]
            d = np.sum(Xi * Xi, axis=1)[:, None] - 2.0 * (Xi @ C.T) + cn[None, :]
            a = np.argmin(d, axis=1)
            assign[i : i + chunk] = a
            tot += float(np.take_along_axis(d, a[:, None], axis=1).sum())
        newC = C.copy()
        for c in range(m):
            sel = assign == c
            if sel.any():
                newC[c] = X[sel].mean(axis=0)
        moved = float(np.max(np.abs(newC - C))) if m else 0.0
        C = newC
        inertia = tot
        if moved == 0.0:
            break
    return C, {"method": "kmeans", "n_entries": int(m), "iters": int(iters),
               "inertia": float(inertia), "n_fit_rows": int(n)}


def subsample_store(X: np.ndarray, n_entries: int, seed: int) -> tuple:
    """The declared weak alternative: a seeded random subsample of train features."""
    X = np.asarray(X, np.float32)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    rng = np.random.default_rng(int(seed) * 7907 + 20_260_911)
    m = min(int(n_entries), int(X.shape[0]))
    sel = rng.permutation(int(X.shape[0]))[:m]
    return X[np.sort(sel)].copy(), {"method": "subsample", "n_entries": int(m),
                                    "n_fit_rows": int(X.shape[0])}


@dataclass
class QuantisedStore:
    """The store **as deployed** — the bytes that would actually ship.

    ⭐ ``values()`` dequantises for arithmetic; the ledger counts the *stored*
    representation, never the dequantised one.
    """

    deployment: str
    payload: np.ndarray
    scale: Optional[np.ndarray]
    offset: Optional[np.ndarray]
    info: dict

    @property
    def n_entries(self) -> int:
        return int(self.payload.shape[0])

    @property
    def dim(self) -> int:
        return int(self.payload.shape[1])

    def values(self) -> np.ndarray:
        if self.deployment == "fp32":
            return np.asarray(self.payload, np.float32)
        if self.deployment == "fp16":
            return np.asarray(self.payload, np.float32)
        return (self.payload.astype(np.float32) * self.scale[None, :]) + self.offset[None, :]

    def deployed_bytes(self) -> int:
        b = int(self.payload.size * self.payload.dtype.itemsize)
        if self.scale is not None:
            b += int(self.scale.size * self.scale.dtype.itemsize)
        if self.offset is not None:
            b += int(self.offset.size * self.offset.dtype.itemsize)
        return b


def quantise(C: np.ndarray, deployment: str, info: dict | None = None) -> QuantisedStore:
    """Quantise centroids to the deployment's representation."""
    C = np.asarray(C, np.float32)
    info = dict(info or {})
    if deployment == "fp32":
        return QuantisedStore("fp32", C.astype(np.float32), None, None, info)
    if deployment == "fp16":
        return QuantisedStore("fp16", C.astype(np.float16), None, None, info)
    if deployment == "int8":
        lo = C.min(axis=0).astype(np.float32)
        hi = C.max(axis=0).astype(np.float32)
        span = np.maximum(hi - lo, 1e-12).astype(np.float32)
        scale = (span / 255.0).astype(np.float32)
        q = np.clip(np.round((C - lo[None, :]) / scale[None, :]), 0, 255).astype(np.uint8)
        return QuantisedStore("int8", q, scale, lo, info)
    raise ValueError(f"unknown deployment {deployment!r} (have {DEPLOYMENTS})")


def build_store(
    X_train: np.ndarray,
    budget_bytes: int,
    deployment: str,
    seed: int,
    method: str = "kmeans",
    iters: int = 25,
) -> QuantisedStore:
    """⭐ Fit a store that fits the byte budget exactly, by shrinking to match."""
    X = np.asarray(X_train, np.float32)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    dim = int(X.shape[1])
    n = entries_for_budget(budget_bytes, dim, deployment)
    C, info = (
        kmeans_store(X, n, seed, iters=iters)
        if method == "kmeans"
        else subsample_store(X, n, seed)
    )
    store = quantise(C, deployment, info)
    got = store.deployed_bytes()
    if got > int(budget_bytes):
        raise AssertionError(
            f"kNN store is {got} B, over the {budget_bytes} B budget — ⛔ shrink, never grow"
        )
    store.info["byte_ledger"] = byte_ledger(store.n_entries, dim, deployment, budget_bytes)
    store.info["deployed_bytes"] = got
    return store


# ---------------------------------------------------------------------------
# Normalisers + metrics (the tuning axes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Normaliser:
    """⛔ Fitted on **train features only** and applied identically to store and query."""

    mode: str
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        if self.mode == "none":
            return X
        return (X - self.mean[None, :]) / self.std[None, :]


def fit_normaliser(X_train: np.ndarray, mode: str) -> Normaliser:
    X = np.asarray(X_train, np.float32)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    if mode not in NORMALISERS:
        raise ValueError(f"unknown normaliser {mode!r} (have {NORMALISERS})")
    mean = X.mean(axis=0).astype(np.float32)
    # ⚠ 71 of 2048 coordinates are DEAD (constant post-ReLU) on this cache; a
    # bare 1/std would divide by zero and poison every distance.
    std = np.maximum(X.std(axis=0), 1e-6).astype(np.float32)
    return Normaliser(mode, mean, std)


def _l2n(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def knn_search(store_values: np.ndarray, Q: np.ndarray, k: int, metric: str) -> tuple:
    """``(idx (N,k), dist (N,k))`` under ``metric``.

    ``cosine`` returns ``1 - cos``; ``euclidean`` returns the L2 distance.  Both
    are *distances* (smaller = closer), so the refusal rule reads identically.
    """
    S = np.asarray(store_values, np.float32)
    Q = np.asarray(Q, np.float32)
    if metric == "cosine":
        D = 1.0 - (_l2n(Q) @ _l2n(S).T)
    elif metric == "euclidean":
        D = np.sqrt(
            np.maximum(
                np.sum(Q * Q, axis=1)[:, None] - 2.0 * (Q @ S.T) + np.sum(S * S, axis=1)[None, :],
                0.0,
            )
        )
    else:
        raise ValueError(f"unknown metric {metric!r} (have {METRICS})")
    kk = min(int(k), S.shape[0])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    d = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(d, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(d, order, axis=1)


def knn_read(store_values_raw: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """⭐ The read: the **mean of the retrieved store entries**, in RAW feature space.

    ⛔ It does not take the query as an argument — the same structural property
    ``gen_store.read_from_codes`` has.  The only channel from request to decoder
    is the ``k`` retrieved indices, i.e. ``k·log2(n_entries)`` bits.
    """
    S = np.asarray(store_values_raw, np.float32)
    return S[np.asarray(idx, np.int64)].mean(axis=1).astype(np.float32)


def bits_per_read(n_entries: int, k: int) -> float:
    return float(int(k) * np.log2(max(int(n_entries), 1)))


def prototype_direct_read(embeddings: np.ndarray) -> np.ndarray:
    """⭐ The **byte-matched prototype arm**: decode the composed prototype itself.

    ⛔ Not a bottlenecked read — the whole ``D``-float vector reaches the decoder,
    so it is reported as its own arm with an ``inf``-bit channel and never in the
    same column as ``knn``'s ``k·log2(n)`` bits.  It is in-budget (the prototype
    table is 466,944 B < 524,288 B on Shapes3D) and it exists because building
    only the weak member of the family would be hobbling.
    """
    return np.asarray(embeddings, np.float32)


# ---------------------------------------------------------------------------
# Refusal: distance to the store, threshold from VALID DATA ONLY
# ---------------------------------------------------------------------------


def refusal_scores(dist: np.ndarray, reduce: str = "min") -> np.ndarray:
    """Refusal score per request (**higher = more likely refused**)."""
    d = np.asarray(dist, np.float64)
    if reduce == "min":
        return d[:, 0].copy()
    if reduce == "mean":
        return d.mean(axis=1)
    raise ValueError(f"unknown reduce {reduce!r} (have 'min', 'mean')")


def threshold_from_valid(scores_valid: np.ndarray, quantile: float) -> float:
    """⭐ The threshold, set on **valid data only** — the rule that binds our certificate.

    ⛔ No C item, and no held-out item, participates: ``scores_valid`` is the
    arm's score on *valid, seen* requests, and the threshold is the ``quantile``
    of that distribution (so ``1 - quantile`` is the false-refusal rate the
    deployer accepts on data they have).
    """
    s = np.asarray(scores_valid, np.float64)
    if s.size == 0:
        raise ValueError("the threshold needs valid scores; ⛔ C is never used for it")
    return float(np.quantile(s, float(quantile)))


def refusal_rates(scores: np.ndarray, threshold: float) -> dict:
    s = np.asarray(scores, np.float64)
    return {
        "n": int(s.size),
        "threshold": float(threshold),
        "refused_fraction": float((s > float(threshold)).mean()) if s.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# The tuning grid (⭐ the budget is reported and must equal every other arm's)
# ---------------------------------------------------------------------------


def tuning_grid(
    ks: Sequence[int] = (1, 3, 5),
    metrics: Sequence[str] = METRICS,
    normalisers: Sequence[str] = NORMALISERS,
    deployments: Sequence[str] = DEPLOYMENTS,
) -> list[dict]:
    """Every (k, metric, normaliser, deployment) trial, in a deterministic order."""
    out = []
    for dep in deployments:
        for norm in normalisers:
            for met in metrics:
                for k in ks:
                    out.append(
                        {"k": int(k), "metric": str(met), "normaliser": str(norm),
                         "deployment": str(dep)}
                    )
    return out
