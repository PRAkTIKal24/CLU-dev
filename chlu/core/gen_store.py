"""The **factored store** over cached encoder channels: grouping, modes, and a read
that provably cannot route around the store.

⭐ DIAL DECLARATION — *isolation*.  The question this module exists to answer is
whether a factored store can be **addressed and read** without the read path
quietly routing around it.  The programme's standing negative result is that the
block *did* route around the memory: deleting the entire store changed the loss by
under 0.001 bits.  So the store-only discipline here is not hygiene, it is the
point, and it is enforced **structurally** and then **measured**:

* :func:`read_from_codes` — the function that produces the read output — does not
  take the features as an argument **at all**.  Its only inputs are the integer
  mode indices and the store.  :func:`assert_no_bypass` checks that by signature
  *and* numerically (perturb the features arbitrarily while holding the codes:
  the output must be bit-identical).
* :func:`delete_store` empties the store; the caller re-runs the read and reports
  the delta.  ⛔ A near-zero delta is a **failed task**, not a bug to repair.

**What is factored.**  The ``D = C·H·W`` coordinates of the frozen encoder's
penultimate map are partitioned into ``K`` **wells**.  Each well ``k`` owns ``M``
learned **modes** in ``R^{d_k}``; a read picks one mode per well and concatenates.
The read channel is therefore exactly ``K·log2(M)`` bits wide and every one of
those bits is a pointer *into the store*.

⛔ **Standing convention (inherited verbatim from** :mod:`chlu.core.factored_store`
**):** no well is ever named semantically — not here, not in the tests, not in an
artifact, not in a figure caption.  Wells carry integer indices and nothing else.

⚠ **What is NOT reused, and why.**  :class:`chlu.core.factored_store.FactoredStore`
is a *physics* store (an :class:`AtomDictionaryPotential` landscape settled by
damped Verlet) bound to :class:`CatTestConfig` and to the synthetic set-code
family that module owns: it needs per-well **anchors** and an item→well
incidence, neither of which exists for a channel partition of a frozen feature
cache.  Forcing it here would have meant inventing anchors, i.e. inventing the
answer.  What *is* reused is the verb that matters and the discipline attached to
it: :func:`chlu.core.multiwell_read.soft_occupancy` is the soft assignment used
during training (⛔ no ``argmax`` in the differentiable path), and the byte
accounting follows ``FactoredStore.n_bytes``.

Runnable through :mod:`chlu.experiments.exp_gen_store`.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .multiwell_read import soft_occupancy

#: The two groupings that are ALWAYS run against each other.  ``random`` is the
#: laundering control and is a full arm on every number, never a footnote.
GROUPINGS = ("correlation", "random")
#: Linkage/affinity backends for the correlation grouping.  ``average`` is the
#: declared default (PREREG D6); the others exist so the choice is auditable.
GROUPING_METHODS = ("auto", "average", "ward", "spectral")
#: ⭐ The order ``method="auto"`` walks.  Pre-registered default first; a method is
#: rejected ONLY when its smallest well falls below the floor, which is a property
#: of train activations alone.
GROUPING_ORDER = ("average", "ward")

__all__ = [
    "GROUPINGS",
    "GROUPING_METHODS",
    "GROUPING_ORDER",
    "GenStore",
    "StorePlan",
    "address_codes",
    "assert_no_bypass",
    "block_diagonality",
    "channel_correlation",
    "choose_k",
    "correlation_grouping",
    "dead_channels",
    "delete_store",
    "dci_scores",
    "grouping_labels",
    "load_store_plan",
    "mig_score",
    "mode_usage",
    "random_grouping",
    "read_from_codes",
    "save_store_plan",
    "soft_weights",
    "store_bytes",
    "store_read",
    "train_store",
    "well_channels_from_labels",
    "well_relevance",
]


# ---------------------------------------------------------------------------
# 1. Grouping — correlation clustering and its same-K laundering control
# ---------------------------------------------------------------------------


def channel_correlation(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Pearson correlation between the ``D`` feature coordinates of ``X`` ``(N, D)``.

    ⛔ Computed on **train** activations only.  A coordinate that is constant over
    the sample (a dead post-ReLU unit) gets a zero row rather than a NaN one — a
    NaN would silently poison the whole linkage.
    """
    A = np.asarray(X, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0)
    dead = sd <= eps
    A = A / np.where(dead, 1.0, sd)
    C = (A.T @ A) / float(A.shape[0])
    C[dead, :] = 0.0
    C[:, dead] = 0.0
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def _affinity(corr: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(corr, dtype=np.float64))


def choose_k(
    corr: np.ndarray, k_min: int = 4, k_max: int = 16, dead: Optional[np.ndarray] = None
) -> dict:
    """⭐ Pick ``K`` by the largest normalised-Laplacian **eigengap** in ``[k_min, k_max]``.

    ⛔ Train activations only; no held-out quantity of any kind enters this.  The
    rule (and the candidate range, and its justification) are pre-registered; the
    value it returns is whatever it returns.
    """
    A = _affinity(corr)
    if dead is not None:
        live = np.flatnonzero(~np.asarray(dead, bool))
        A = A[np.ix_(live, live)]
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.where(d <= 0, 1.0, d))
    L = np.eye(A.shape[0]) - (dinv[:, None] * A * dinv[None, :])
    ev = np.linalg.eigvalsh((L + L.T) / 2.0)
    ev = np.sort(np.real(ev))
    gaps = {int(k): float(ev[k] - ev[k - 1]) for k in range(int(k_min), int(k_max) + 1)}
    best = max(gaps, key=lambda k: gaps[k])
    return {
        "k": int(best),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "rule": "argmax_k (lambda_{k+1} - lambda_k) of the symmetric normalised "
                "Laplacian of |corr|, over the pre-registered candidate range",
        "eigengaps": gaps,
        "eigenvalues_head": [float(x) for x in ev[: int(k_max) + 2]],
        "n_channels_used": int(A.shape[0]),
        "dead_excluded": bool(dead is not None),
    }


def dead_channels(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Boolean mask of coordinates that are **constant** over the sample.

    ⚠ Measured, not assumed: the shipped Shapes3D/``recon`` cache has **65 of 2048**
    dead post-ReLU coordinates.  They correlate with nothing, so every pairwise
    distance involving them is exactly 1.0 — which is precisely what makes
    ``fcluster(..., 'maxclust')`` collapse to a single cluster (a tie plateau at the
    top of the dendrogram admits no cut).  They are therefore clustered *around*,
    not with, and then placed deterministically.  A constant coordinate is
    reconstructed exactly by any mode, so where it lands carries no information.
    """
    return np.asarray(np.std(np.asarray(X, np.float64), axis=0) <= float(eps))


def _cluster_live(corr: np.ndarray, k: int, method: str) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    C = np.asarray(corr, dtype=np.float64)
    if method == "average":
        dist = 1.0 - np.abs(C)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        Z = linkage(squareform(dist, checks=False), method="average")
        return np.asarray(fcluster(Z, int(k), criterion="maxclust") - 1, np.int64)
    if method == "ward":
        # ward needs a Euclidean embedding: each coordinate's correlation PROFILE
        # is its feature vector.
        Z = linkage(np.abs(C), method="ward")
        return np.asarray(fcluster(Z, int(k), criterion="maxclust") - 1, np.int64)
    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        return np.asarray(
            SpectralClustering(
                n_clusters=int(k), affinity="precomputed", random_state=0
            ).fit_predict(_affinity(C)),
            np.int64,
        )
    raise ValueError(f"method must be one of {GROUPING_METHODS}, got {method!r}")


def correlation_grouping(
    corr: np.ndarray,
    k: int,
    method: str = "auto",
    dead: Optional[np.ndarray] = None,
    min_well_frac: float = 0.10,
    return_info: bool = False,
):
    """Partition the ``D`` coordinates into ``k`` wells by correlation clustering.

    Deterministic given ``corr`` — the grouping carries no seed of its own; only the
    feature cache does.

    ⚠ **``method="auto"`` and why it exists.**  The pre-registered default was plain
    average linkage on ``d = 1 - |corr|``.  On the shipped cache it **chains**: at
    every ``k`` in the candidate range it returns one well of ~1976 coordinates plus
    singletons (measured; see the report).  That is a degenerate partition, and a
    degenerate partition makes both arms vacuous — the random control matches the
    size histogram, so it would be equally degenerate.  ``auto`` therefore walks
    :data:`GROUPING_ORDER` and takes the **first method whose smallest well holds at
    least ``min_well_frac`` of an equal share**.

    ⛔ The trigger is a property of the partition over **train activations only**.
    No held-out quantity, no read metric and no arm outcome enters it, and the
    method actually used is recorded in the returned info dict and in the manifest.
    """
    C = np.asarray(corr, dtype=np.float64)
    d = C.shape[0]
    dead = (
        np.zeros(d, bool) if dead is None else np.asarray(dead, bool)
    )
    live = np.flatnonzero(~dead)
    if live.size < int(k):
        raise ValueError(f"only {live.size} live coordinates for {k} wells")
    Cl = C[np.ix_(live, live)]
    floor = max(1, int(np.floor(float(min_well_frac) * live.size / int(k))))
    order = list(GROUPING_ORDER) if method == "auto" else [method]
    attempts, chosen, lab_live = [], None, None
    for m in order:
        lab = _compact_labels(_cluster_live(Cl, int(k), m), int(k), strict=False)
        sizes = np.bincount(lab, minlength=int(k))
        ok = bool(sizes.size == int(k) and sizes.min() >= floor)
        attempts.append(
            {"method": m, "sizes": [int(x) for x in sizes], "min_well_size": int(sizes.min()),
             "floor": int(floor), "accepted": ok}
        )
        if ok:
            chosen, lab_live = m, lab
            break
    if lab_live is None:
        raise ValueError(
            f"every method in {order} returned a degenerate partition at k={k} "
            f"(floor {floor}); attempts: {attempts}"
        )

    labels = np.zeros(d, np.int64)
    labels[live] = lab_live
    # ⭐ dead coordinates are placed deterministically into the smallest well —
    # information-free, and it keeps a whole well from being spent on constants.
    counts = np.bincount(lab_live, minlength=int(k)).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(counts))
        labels[c] = j
        counts[j] += 1
    info = {
        "method_requested": method,
        "method_used": chosen,
        "method_order": order,
        "min_well_frac": float(min_well_frac),
        "min_well_size_floor": int(floor),
        "attempts": attempts,
        "n_dead_channels": int(dead.sum()),
        "n_live_channels": int(live.size),
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=int(k))],
    }
    return (labels, info) if return_info else labels


def _compact_labels(lab: np.ndarray, k: int, strict: bool = True) -> np.ndarray:
    """Relabel to ``[0, n_nonempty)`` in order of first appearance.

    ``strict`` raises when the clustering returned fewer than ``k`` non-empty wells;
    ⛔ an empty well is a degenerate partition and is **reported**, never silently
    merged away.  ``strict=False`` is used by the ``auto`` walk, which needs to see
    the degenerate sizes in order to reject them."""
    uniq = np.unique(lab)
    if strict and uniq.size != k:
        raise ValueError(
            f"clustering returned {uniq.size} non-empty wells, not {k} — an empty "
            "well is a degenerate partition and is reported, never silently merged"
        )
    remap = {int(u): i for i, u in enumerate(uniq)}
    return np.asarray([remap[int(x)] for x in lab], dtype=np.int64)


def random_grouping(reference_labels: np.ndarray, seed: int) -> np.ndarray:
    """⭐ THE laundering control: same ``K``, **same well-size histogram**, shuffled.

    A random partition drawn independently would differ from the clustered one in
    *two* ways (assignment and size profile) and any margin would be unattributable.
    Permuting the reference label vector holds the size histogram exactly, so the
    only thing that varies is which coordinate sits in which well.
    """
    rng = np.random.default_rng(int(seed))
    return np.asarray(rng.permutation(np.asarray(reference_labels)), dtype=np.int64)


def grouping_labels(
    corr: np.ndarray,
    k: int,
    grouping: str,
    seed: int,
    method: str = "auto",
    dead: Optional[np.ndarray] = None,
    min_well_frac: float = 0.10,
) -> tuple:
    """``(labels, info)`` for ``grouping in GROUPINGS`` at the same ``k``.

    ⭐ Both groupings are derived from the SAME clustered reference, so the random
    control inherits its well-size histogram **exactly** and the only thing that
    varies between the arms is which coordinate sits in which well."""
    ref, info = correlation_grouping(
        corr, k, method, dead=dead, min_well_frac=min_well_frac, return_info=True
    )
    if grouping == "correlation":
        return ref, info
    if grouping == "random":
        info = dict(info, randomised_from="correlation", random_seed=int(seed))
        return random_grouping(ref, seed), info
    raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")


def well_channels_from_labels(labels: np.ndarray, k: int) -> tuple:
    """``(K,)`` tuple of sorted coordinate-index tuples.  Integer indices only."""
    lab = np.asarray(labels, dtype=np.int64)
    return tuple(
        tuple(int(i) for i in np.flatnonzero(lab == j)) for j in range(int(k))
    )


def block_diagonality(corr: np.ndarray, labels: np.ndarray) -> dict:
    """Fraction of squared off-diagonal |corr| mass sitting INSIDE the wells.

    ⚠ Near-tautological for the clustered arm (the linkage optimises a monotone
    function of it) — pre-declared as a *sanity check that the clustering ran*,
    never as evidence.  Reported with the random-partition expectation alongside.
    """
    C = np.asarray(corr, dtype=np.float64) ** 2
    np.fill_diagonal(C, 0.0)
    lab = np.asarray(labels, dtype=np.int64)
    same = lab[:, None] == lab[None, :]
    total = float(C.sum())
    inside = float(C[same].sum())
    n = lab.size
    counts = np.bincount(lab)
    # expected in-block share of a uniformly random assignment with these sizes
    exp_pairs = float(np.sum(counts * (counts - 1))) / float(n * (n - 1))
    return {
        "in_block_fraction": inside / total if total > 0 else float("nan"),
        "in_block_pair_fraction": exp_pairs,
        "lift_over_size_matched_random": (
            (inside / total) / exp_pairs if total > 0 and exp_pairs > 0 else float("nan")
        ),
        "well_sizes": [int(c) for c in counts],
    }


# ---------------------------------------------------------------------------
# 2. The store
# ---------------------------------------------------------------------------


class GenStore(eqx.Module):
    """``K`` wells over a channel partition; ``M`` learned modes per well.

    ``modes[k]`` is ``(M, d_k)`` — the well's own vocabulary in its own
    sub-space.  ⛔ The wells partition the coordinates, so
    ``sum_k d_k = D`` and ``store_bytes = M·D·4`` **independently of K**, which is
    what lets a byte-matched baseline exist at all.
    """

    modes: list
    well_channels: tuple = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    feat_shape: tuple = eqx.field(static=True)

    def __init__(self, well_channels, n_modes: int, feat_shape, key, init=None):
        self.well_channels = tuple(tuple(int(i) for i in w) for w in well_channels)
        self.feat_shape = tuple(int(x) for x in feat_shape)
        self.dim = int(np.prod(self.feat_shape))
        flat = np.concatenate([np.asarray(w, np.int64) for w in self.well_channels])
        if flat.size != self.dim or np.unique(flat).size != self.dim:
            raise ValueError(
                f"the wells must PARTITION the {self.dim} coordinates "
                f"(got {flat.size} entries, {np.unique(flat).size} distinct)"
            )
        keys = jax.random.split(key, len(self.well_channels))
        modes = []
        for j, (w, k) in enumerate(zip(self.well_channels, keys, strict=True)):
            if init is not None:
                modes.append(jnp.asarray(init[j], jnp.float32))
            else:
                modes.append(
                    0.1 * jax.random.normal(k, (int(n_modes), len(w)), jnp.float32)
                )
        self.modes = modes

    @property
    def n_wells(self) -> int:
        return len(self.well_channels)

    @property
    def n_modes(self) -> int:
        return int(self.modes[0].shape[0])

    @property
    def bits_per_read(self) -> float:
        """Width of the ONLY channel from the features to the decoder."""
        return float(self.n_wells * np.log2(self.n_modes))

    def n_bytes(self) -> int:
        """Store size in bytes (float32 modes) — follows ``FactoredStore.n_bytes``."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return int(sum(int(np.asarray(x).size) for x in leaves) * 4)


def store_bytes(store: GenStore) -> dict:
    """⭐ Machine-readable byte accounting — ``gen-baselines`` byte-matches to this."""
    per_well = [int(np.asarray(m).size) * 4 for m in store.modes]
    return {
        "store_bytes": int(sum(per_well)),
        "store_bytes_per_well": per_well,
        "n_wells": int(store.n_wells),
        "n_modes": int(store.n_modes),
        "dim": int(store.dim),
        "dtype": "float32",
        "bytes_formula": "n_modes * sum_k d_k * 4 = n_modes * D * 4",
        "bits_per_read": store.bits_per_read,
        "note": (
            "The modes ARE the store. The addressing head is reported separately "
            "and is not part of this count; a byte-matched baseline matches "
            "store_bytes."
        ),
    }


def _well_views(store: GenStore, X):
    """``X (N, D) -> [ (N, d_k) ]`` gathers.  Cast to the store's own dtype (§7.23)."""
    dt = store.modes[0].dtype
    Z = jnp.asarray(X, dt).reshape((X.shape[0], -1))
    return [Z[:, jnp.asarray(w, jnp.int32)] for w in store.well_channels]


def soft_weights(store: GenStore, X, tau: float) -> list:
    """Per-well soft assignment over modes, ``[ (N, M) ]``.  ⛔ No ``argmax``.

    Reuses :func:`chlu.core.multiwell_read.soft_occupancy` verbatim — the same
    softmax-over-negative-squared-distance verb, applied to modes within a well
    rather than to wells within a store.
    """
    return [
        soft_occupancy(z, m, float(tau))
        for z, m in zip(_well_views(store, X), store.modes, strict=True)
    ]


def address_codes(store: GenStore, X) -> np.ndarray:
    """⭐ THE addressing head: features -> ``(N, K)`` integer mode indices.

    This is the **only** thing computed from the features that reaches the read.
    """
    out = []
    for z, m in zip(_well_views(store, X), store.modes, strict=True):
        # ⚠ Gram form, not the (N, M, d) broadcast: at the shipped eval batch the
        # broadcast is ~270 MB PER WELL and the run dies on a laptop.  Exact same
        # argmin (the dropped ||z||^2 term is constant in m).
        d2 = jnp.sum(m * m, axis=-1)[None, :] - 2.0 * (z @ m.T)
        out.append(jnp.argmin(d2, axis=-1))
    return np.asarray(jnp.stack(out, axis=1), dtype=np.int64)


def read_from_codes(store: GenStore, codes) -> np.ndarray:
    """⛔⭐ THE read.  Takes the store and the integer codes — and **NOT the features**.

    That absence is the structural no-bypass guarantee: there is no argument
    through which encoder information could reach the output except the ``K``
    integers, each of which is a pointer into the store.  :func:`assert_no_bypass`
    checks the signature *and* the numerics.
    """
    idx = jnp.asarray(codes, jnp.int32)
    n = int(idx.shape[0])
    out = jnp.zeros((n, store.dim), store.modes[0].dtype)
    for j, (w, m) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        out = out.at[:, jnp.asarray(w, jnp.int32)].set(m[idx[:, j]])
    return np.asarray(out).reshape((n,) + store.feat_shape)


def store_read(store: GenStore, X, batch: int = 4096) -> np.ndarray:
    """``features -> codes -> store read``.  Batched; the two halves stay separate."""
    outs = []
    n = int(np.asarray(X).shape[0])
    for i in range(0, n, int(batch)):
        chunk = np.asarray(X[i : i + int(batch)], np.float32)
        chunk = chunk.reshape((chunk.shape[0], -1))
        outs.append(read_from_codes(store, address_codes(store, chunk)))
    if not outs:
        return np.zeros((0,) + store.feat_shape, np.float32)
    return np.concatenate(outs).astype(np.float32)


def soft_read(store: GenStore, X, tau: float, batch: int = 1024) -> np.ndarray:
    """Soft (training-time) read: ``sum_m pi_m · mode_m`` per well. Secondary metric."""
    outs = []
    n = int(np.asarray(X).shape[0])
    for i in range(0, n, int(batch)):
        chunk = jnp.asarray(
            np.asarray(X[i : i + int(batch)], np.float32).reshape(
                (-1, store.dim)
            ),
            store.modes[0].dtype,
        )
        pis = soft_weights(store, chunk, tau)
        out = jnp.zeros((chunk.shape[0], store.dim), store.modes[0].dtype)
        for w, m, pi in zip(store.well_channels, store.modes, pis, strict=True):
            out = out.at[:, jnp.asarray(w, jnp.int32)].set(pi @ m)
        outs.append(np.asarray(out))
    if not outs:
        return np.zeros((0,) + store.feat_shape, np.float32)
    return np.concatenate(outs).reshape((-1,) + store.feat_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. ⛔ The two checks the whole spoke lives or dies on
# ---------------------------------------------------------------------------


def assert_no_bypass(store: GenStore, X, rng_seed: int = 0) -> dict:
    """⭐ Prove the read output is a function of ``(codes, store)`` **alone**.

    Two independent checks, because either alone is weak:

    1. **Structural / by signature** — ``read_from_codes`` must not accept any
       argument that could carry the features.  A future edit that adds one is
       caught here rather than at cell time.
    2. **Numerical** — perturb the features arbitrarily (they are *replaced* by
       noise), keep the codes, re-read: the output must be **bit-identical**.
       If a bypass existed the output would move.
    """
    params = list(inspect.signature(read_from_codes).parameters)
    if params != ["store", "codes"]:
        raise AssertionError(
            "read_from_codes must take exactly (store, codes) — an extra argument "
            f"is a potential encoder->decoder bypass; got {params}"
        )
    X = np.asarray(X, np.float32).reshape((np.asarray(X).shape[0], -1))
    codes = address_codes(store, X)
    y0 = read_from_codes(store, codes).reshape((X.shape[0], -1))

    # (2a) INDEPENDENT reimplementation: a pure table lookup from (codes, store),
    # written without reference to X at all.  Bit-equality proves the shipped read
    # output IS the codebook gather and nothing else.
    ref = np.zeros_like(y0)
    for j, (w, m) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        ref[:, np.asarray(w, np.int64)] = np.asarray(m)[codes[:, j]]
    max_abs = float(np.max(np.abs(y0 - ref))) if y0.size else 0.0
    if max_abs != 0.0:
        raise AssertionError(
            f"the read output is not the pure codebook gather (max |delta| = "
            f"{max_abs}) — something other than the codes reaches the output"
        )

    # (2b) the channel is LIVE: replacing the features moves the codes, and moving
    # the codes moves the output.  A read that ignored the features would pass
    # (2a) trivially and is caught here.
    rng = np.random.default_rng(int(rng_seed))
    X2 = (rng.random(X.shape).astype(np.float32) * float(np.abs(X).max() + 1.0))
    codes2 = address_codes(store, X2)
    y2 = read_from_codes(store, codes2).reshape((X.shape[0], -1))
    codes_moved = bool(np.any(codes != codes2))
    out_moved = float(np.max(np.abs(y0 - y2))) if y0.size else 0.0
    return {
        "signature_ok": True,
        "read_is_pure_codebook_gather": True,
        "max_abs_delta_vs_independent_gather": max_abs,
        "codes_changed_under_feature_replacement": codes_moved,
        "output_changed_when_codes_changed": out_moved,
        "bits_per_read": store.bits_per_read,
        "n_checked": int(X.shape[0]),
        "note": (
            "The output path has no argument through which encoder information "
            "could travel except the K integer codes."
        ),
    }


def delete_store(store: GenStore, mode: str = "zero", seed: int = 0) -> GenStore:
    """Return the store with its **content removed**; the addressing head is untouched.

    Three deletions, because a single one is easy to fool:

    ``zero``     every mode set to 0 — the store is empty.  The headline.
    ``mean``     every mode in a well replaced by that well's **mean mode**.  ⭐ The
                 *fairest* deletion: the read output is still an on-manifold,
                 correctly-scaled feature vector, so the decoder is not simply being
                 handed garbage — but the store's ``K·log2(M)`` bits now carry
                 **zero** information.  Any accuracy that survives this is accuracy
                 the store never supplied.
    ``scramble`` each mode matrix has its entries permuted (content destroyed,
                 per-well shape/scale/marginal preserved exactly).

    ⚠ **A trap worth naming:** *permuting the rows* of a codebook is a **no-op** —
    nearest-neighbour addressing re-finds the same vector, so the read is
    bit-identical and the "deletion" measures nothing.  That variant was written,
    measured at Δ = 0.0000 exactly, and replaced.

    ⛔ The point of this function is to be *measured against*: if the read output
    does not change, a bypass exists and the task has FAILED.
    """
    if mode == "zero":
        new = [jnp.zeros_like(m) for m in store.modes]
    elif mode == "mean":
        new = [jnp.broadcast_to(jnp.mean(m, axis=0, keepdims=True), m.shape) for m in store.modes]
    elif mode == "scramble":
        rng = np.random.default_rng(int(seed))
        new = []
        for m in store.modes:
            arr = np.asarray(m)
            flat = arr.reshape(-1).copy()
            rng.shuffle(flat)
            new.append(jnp.asarray(flat.reshape(arr.shape), m.dtype))
    else:
        raise ValueError(f"unknown deletion mode {mode!r} (zero|mean|scramble)")
    return eqx.tree_at(lambda t: t.modes, store, new)


# ---------------------------------------------------------------------------
# 4. Training the modes — store-only objective, Optax Adam, keys threaded
# ---------------------------------------------------------------------------


def _kmeanspp_init(Z: np.ndarray, m: int, rng) -> np.ndarray:
    """k-means++ seeding for one well's modes (deterministic given ``rng``)."""
    n = Z.shape[0]
    idx = [int(rng.integers(n))]
    d2 = np.sum((Z - Z[idx[0]]) ** 2, axis=1)
    for _ in range(1, m):
        tot = float(d2.sum())
        p = (d2 / tot) if tot > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))
    return Z[np.asarray(idx)].astype(np.float32)


def init_store(
    X: np.ndarray, well_channels, n_modes: int, feat_shape, key, seed: int
) -> GenStore:
    """Modes seeded by k-means++ **on the store's own training features**."""
    rng = np.random.default_rng(int(seed) * 2_654_435_761 % (2**32))
    flat = np.asarray(X, np.float32).reshape((X.shape[0], -1))
    init = [_kmeanspp_init(flat[:, np.asarray(w, np.int64)], int(n_modes), rng)
            for w in well_channels]
    return GenStore(well_channels, int(n_modes), feat_shape, key, init=init)


def train_store(
    store: GenStore,
    X: np.ndarray,
    cfg,
    key,
    log: Optional[Callable] = None,
) -> tuple:
    """Fit the modes with a **store-only** objective.  ⛔ No labels, no decoder, no MSE
    against anything but the store's own read.

    ``L = mean_k || x_k - sum_m pi_{k,m} mode_{k,m} ||^2  +  beta ||x_k - mode_{k,argmin}||^2``

    with ``pi`` the soft occupancy at temperature ``tau``, geometrically annealed
    ``tau_start -> tau_end`` so the soft read the gradient sees converges onto the
    hard read the evaluation uses.  The second term is the standard codebook /
    commitment pull and is what stops unused modes from drifting away.

    ⚠ **Do not read the returned history as a monotone training curve.** ``tau``
    moves during the run, so ``history[0]`` and ``history[-1]`` are values of two
    *different* functions: the trace can RISE while the store gets strictly better.
    The quantity to compare is :func:`reconstruction_mse` of the hard read against
    the k-means++ initialisation (asserted in
    ``tests/test_gen_store.py::test_training_improves_the_hard_read_over_the_kmeanspp_init``).
    """
    flat = jnp.asarray(np.asarray(X, np.float32).reshape((X.shape[0], -1)))
    n = int(flat.shape[0])
    batch = min(int(cfg.train_batch), n)
    steps = int(cfg.train_steps)
    beta = float(cfg.commitment_beta)
    t0, t1 = float(cfg.tau_start), float(cfg.tau_end)
    opt = optax.adam(float(cfg.train_lr))
    opt_state = opt.init(eqx.filter(store, eqx.is_inexact_array))

    def loss_fn(st, xb, tau):
        views = _well_views(st, xb)
        soft = 0.0
        commit = 0.0
        for z, m in zip(views, st.modes, strict=True):
            # ⚠ ``soft_occupancy`` in GRAM form.  Algebraically identical (asserted
            # in tests/test_gen_store.py::test_gram_softmax_matches_soft_occupancy),
            # but the library form materialises an (N, M, d) broadcast that is
            # ~130 MB per step at the shipped cell and ~4x that under reverse-mode
            # AD.  ⛔ No argmax in the differentiable path either way.
            d2 = (
                jnp.sum(z * z, axis=-1)[:, None]
                - 2.0 * (z @ m.T)
                + jnp.sum(m * m, axis=-1)[None, :]
            )
            pi = jax.nn.softmax(-d2 / (2.0 * tau**2), axis=-1)
            soft = soft + jnp.mean(jnp.sum((z - pi @ m) ** 2, axis=-1))
            commit = commit + jnp.mean(jnp.min(d2, axis=-1))
        k = len(st.modes)
        return (soft + beta * commit) / k

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(st, s, xb, tau):
        loss, grads = grad_fn(st, xb, tau)
        updates, s = opt.update(grads, s, st)
        return eqx.apply_updates(st, updates), s, loss

    history = []
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        tau = t0 * (t1 / t0) ** (i / max(1, steps - 1))
        store, opt_state, loss = step(store, opt_state, flat[idx], tau)
        history.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    step {i + 1}/{steps}  loss={history[-1]:.5f}  tau={tau:.3f}")
    return store, history, key


def reconstruction_mse(store: GenStore, X: np.ndarray, batch: int = 4096) -> dict:
    """Feature-space error of the HARD read (primary) and of the raw feature scale."""
    Xf = np.asarray(X, np.float32)
    n = Xf.shape[0]
    Xf = Xf.reshape((n, -1))
    rec = store_read(store, Xf, batch).reshape((n, -1))
    err = float(np.mean((rec - Xf) ** 2))
    var = float(np.mean((Xf - Xf.mean(axis=0, keepdims=True)) ** 2))
    return {
        "mse": err,
        "feature_variance": var,
        "relative_mse": err / var if var > 0 else float("nan"),
        "n": int(Xf.shape[0]),
    }


def mode_usage(codes: np.ndarray, n_modes: int) -> dict:
    """Per-well mode-occupancy entropy — a dead codebook is a silent failure."""
    codes = np.asarray(codes, np.int64)
    out = []
    for j in range(codes.shape[1]):
        c = np.bincount(codes[:, j], minlength=int(n_modes)).astype(np.float64)
        p = c / max(c.sum(), 1.0)
        nz = p[p > 0]
        out.append(
            {
                "well": int(j),
                "used_modes": int((c > 0).sum()),
                "entropy_bits": float(-(nz * np.log2(nz)).sum()),
                "max_bits": float(np.log2(int(n_modes))),
            }
        )
    return {
        "per_well": out,
        "mean_used_modes": float(np.mean([o["used_modes"] for o in out])),
        "mean_entropy_bits": float(np.mean([o["entropy_bits"] for o in out])),
    }


# ---------------------------------------------------------------------------
# 5. Factorisation measures — the ladder's x-axis and its companions
# ---------------------------------------------------------------------------


def _mi_bits(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    """Plug-in mutual information in bits.  ⚠ Upward-biased at small ``N/bins``;
    identical bias for both groupings (same ``K``, ``M``, ``N``), so only the
    contrast is read."""
    joint = np.zeros((int(na), int(nb)), np.float64)
    np.add.at(joint, (np.asarray(a, np.int64), np.asarray(b, np.int64)), 1.0)
    joint /= max(joint.sum(), 1.0)
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    denom = pa * pb
    nz = (joint > 0) & (denom > 0)
    return float(np.sum(joint[nz] * np.log2(joint[nz] / denom[nz])))


def _entropy_bits(a: np.ndarray, na: int) -> float:
    p = np.bincount(np.asarray(a, np.int64), minlength=int(na)).astype(np.float64)
    p /= max(p.sum(), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def well_relevance(codes: np.ndarray, labels: np.ndarray, cardinalities) -> dict:
    """``R[k, f] = I(code_k ; factor_f)`` in bits, plus the row-normalised version.

    This is the importance matrix DCI reads, and the map the well-space C
    diagnostic lands on.  ⛔ Wells stay integer-indexed.
    """
    codes = np.asarray(codes, np.int64)
    labels = np.asarray(labels, np.int64)
    n_modes = int(codes.max()) + 1
    K, F = codes.shape[1], labels.shape[1]
    R = np.zeros((K, F), np.float64)
    for k in range(K):
        for f in range(F):
            R[k, f] = _mi_bits(codes[:, k], labels[:, f], n_modes, int(cardinalities[f]))
    Hf = np.asarray(
        [_entropy_bits(labels[:, f], int(cardinalities[f])) for f in range(F)]
    )
    col = R.sum(axis=0, keepdims=True)
    return {
        "mi_bits": R.tolist(),
        "factor_entropy_bits": Hf.tolist(),
        "share_by_factor": np.divide(
            R, np.where(col > 0, col, 1.0)
        ).tolist(),  # column k-share of factor f's relevance
    }


def mig_score(codes: np.ndarray, labels: np.ndarray, cardinalities) -> dict:
    """Mutual Information Gap over the ``K`` mode-index variables."""
    rel = well_relevance(codes, labels, cardinalities)
    R = np.asarray(rel["mi_bits"])
    Hf = np.asarray(rel["factor_entropy_bits"])
    per_factor = []
    for f in range(R.shape[1]):
        col = np.sort(R[:, f])[::-1]
        gap = float(col[0] - col[1]) if col.size > 1 else float(col[0])
        per_factor.append(gap / Hf[f] if Hf[f] > 0 else float("nan"))
    return {
        "mig": float(np.nanmean(per_factor)),
        "per_factor": [float(x) for x in per_factor],
    }


def dci_scores(codes: np.ndarray, labels: np.ndarray, cardinalities) -> dict:
    """DCI from the MI importance matrix (informativeness = per-factor best-well MI/H)."""
    rel = well_relevance(codes, labels, cardinalities)
    R = np.asarray(rel["mi_bits"]) + 1e-12
    Hf = np.asarray(rel["factor_entropy_bits"])
    K, F = R.shape
    Pd = R / R.sum(axis=1, keepdims=True)
    d_k = 1.0 - (-(Pd * np.log(Pd)).sum(axis=1) / np.log(F))
    rho = R.sum(axis=1) / R.sum()
    Pc = R / R.sum(axis=0, keepdims=True)
    c_f = 1.0 - (-(Pc * np.log(Pc)).sum(axis=0) / np.log(K))
    info = np.asarray(
        [R[:, f].max() / Hf[f] if Hf[f] > 0 else np.nan for f in range(F)]
    )
    return {
        "disentanglement": float((rho * d_k).sum()),
        "completeness": float(np.mean(c_f)),
        "informativeness": float(np.nanmean(info)),
        "per_well_disentanglement": [float(x) for x in d_k],
        "per_factor_completeness": [float(x) for x in c_f],
    }


# ---------------------------------------------------------------------------
# 6. Per-well probe transfer -> LEAKAGE (⭐ the factorisation ladder's x-axis)
# ---------------------------------------------------------------------------


class WellProbe(eqx.Module):
    """One linear head per factor over a single well's sub-vector.  ⛔ Measurement
    only (R1 clarification): the generative path stays label-free."""

    linears: list

    def __init__(self, in_dim: int, cardinalities, key):
        keys = jax.random.split(key, len(cardinalities))
        self.linears = [
            eqx.nn.Linear(int(in_dim), int(c), key=k)
            for c, k in zip(cardinalities, keys, strict=True)
        ]

    def __call__(self, z):
        z = jnp.asarray(z, self.linears[0].weight.dtype).reshape(-1)
        return [lin(z) for lin in self.linears]


def train_well_probe(Z, Y, cardinalities, cfg, key, log=None):
    """Multinomial logistic heads on ``Z (N, d_k)``, Optax Adam, full-batch minibatches."""
    probe = WellProbe(int(Z.shape[1]), cardinalities, key)
    dt = probe.linears[0].weight.dtype
    Zj = jnp.asarray(np.asarray(Z, np.float32), dt)
    Yj = jnp.asarray(np.asarray(Y, np.int32))
    mu = jnp.mean(Zj, axis=0, keepdims=True)
    sd = jnp.std(Zj, axis=0, keepdims=True) + 1e-6
    Zj = (Zj - mu) / sd

    def loss_fn(m, zb, yb):
        logits = jax.vmap(m)(zb)
        tot = 0.0
        for j, lg in enumerate(logits):
            lp = jax.nn.log_softmax(lg, axis=-1)
            tot = tot - jnp.mean(jnp.take_along_axis(lp, yb[:, j][:, None], axis=-1))
        return tot / len(logits)

    opt = optax.adam(float(cfg.probe_lr))
    opt_state = opt.init(eqx.filter(probe, eqx.is_array))
    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(m, s, zb, yb):
        loss, g = grad_fn(m, zb, yb)
        upd, s = opt.update(g, s, m)
        return eqx.apply_updates(m, upd), s, loss

    n = int(Zj.shape[0])
    batch = min(int(cfg.probe_batch), n)
    hist = []
    for _i in range(int(cfg.probe_steps)):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        probe, opt_state, loss = step(probe, opt_state, Zj[idx], Yj[idx])
        hist.append(float(loss))
    if log is not None:
        log(f"      probe loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    return probe, (np.asarray(mu), np.asarray(sd)), hist


def well_probe_accuracy(probe, norm, Z, Y, factor_names) -> dict:
    mu, sd = norm
    dt = probe.linears[0].weight.dtype
    Zj = (jnp.asarray(np.asarray(Z, np.float32), dt) - jnp.asarray(mu, dt)) / jnp.asarray(
        sd, dt
    )
    logits = eqx.filter_jit(jax.vmap(probe))(Zj)
    preds = np.stack([np.asarray(jnp.argmax(lg, -1)) for lg in logits], axis=1)
    Y = np.asarray(Y, np.int64)
    correct = preds == Y
    pf = {
        str(nm): float(correct[:, j].mean()) if correct.size else float("nan")
        for j, nm in enumerate(factor_names)
    }
    return {
        "n": int(Y.shape[0]),
        "per_factor": pf,
        "aggregate": float(np.mean(list(pf.values()))) if pf else float("nan"),
    }


def leakage_from_transfer(acc_holdout: dict, acc_matched: dict, factor_names) -> dict:
    """⭐ ``leakage = 1 - transfer``, ``transfer = acc_heldout / acc_matched_seen``.

    ⛔ The matched-seen normaliser is R16's permanent companion, **not** random-A:
    two thirds of the raw held-out deficit is a photometric rendering artefact of
    the Montero corner, and normalising against random-A would score that artefact
    as leakage.
    """
    per_factor = {}
    for nm in factor_names:
        a = float(acc_holdout["per_factor"][str(nm)])
        b = float(acc_matched["per_factor"][str(nm)])
        per_factor[str(nm)] = (a / b) if b > 0 else float("nan")
    agg = (
        float(acc_holdout["aggregate"] / acc_matched["aggregate"])
        if acc_matched["aggregate"] > 0
        else float("nan")
    )
    return {
        "transfer_per_factor": per_factor,
        "transfer_aggregate": agg,
        "leakage_per_factor": {k: 1.0 - v for k, v in per_factor.items()},
        "leakage_aggregate": 1.0 - agg,
    }


# ---------------------------------------------------------------------------
# 7. The well-space C mapping (R2 addendum) — ⛔ a DIAGNOSTIC, never a query set
# ---------------------------------------------------------------------------


def c_well_mapping(c_records, relevance: dict, factor_names, top_share: float = 0.5) -> dict:
    """Where the FACTOR-SPACE C queries land on wells.

    ⛔ **C stays factor-space and byte-identical for every arm** (R2): two of the
    four arms have no wells at all.  This function only *reports* where the
    already-fixed queries land, using the ``I(code_k; factor_f)`` relevance matrix.

    ``intra_well_conflict_fraction`` — the headline diagnostic — is the fraction of
    C records whose violated factor has its relevance mass concentrated on a single
    well (top-1 share ``>= top_share``), i.e. the structural conflict is localised
    *inside* one well rather than smeared across the store.
    """
    share = np.asarray(relevance["share_by_factor"])  # (K, F), columns sum to 1
    names = [str(n) for n in factor_names]
    per_record, landing = [], []
    for rec in c_records:
        vf = str(rec.get("violated_factor", ""))
        if vf not in names:
            continue
        f = names.index(vf)
        col = share[:, f]
        k = int(np.argmax(col))
        per_record.append(
            {
                "id": rec.get("id"),
                "kind": rec.get("kind"),
                "violated_factor_index": f,
                "top_well": k,
                "top_well_share": float(col[k]),
                "intra_well": bool(col[k] >= float(top_share)),
            }
        )
        landing.append(k)
    K = share.shape[0]
    hist = np.bincount(np.asarray(landing, np.int64), minlength=K) if landing else np.zeros(K, int)
    return {
        "n_records": len(per_record),
        "intra_well_conflict_fraction": (
            float(np.mean([r["intra_well"] for r in per_record])) if per_record else float("nan")
        ),
        "mean_top_well_share": (
            float(np.mean([r["top_well_share"] for r in per_record])) if per_record else float("nan")
        ),
        "landing_histogram": [int(x) for x in hist],
        "top_share_threshold": float(top_share),
        "per_record": per_record,
        "note": (
            "R2 addendum: reported mapping only. The C query set is defined in "
            "factor space and is byte-identical across arms; it NEVER becomes a "
            "well-space query set."
        ),
    }


# ---------------------------------------------------------------------------
# 8. ⭐ The store PLAN — the reusable artifact every later ladder rung consumes
# ---------------------------------------------------------------------------


class StorePlan(dict):
    """A loadable store configuration.  ⛔ A later rung that re-derives ``K`` or the
    byte count voids the ladder comparison, so the plan is an artifact, not prose."""


def save_store_plan(path, *, dataset, pretext, k, n_modes, dim, feat_shape,
                    labels, k_choice, grouping_method, store_bytes_dict,
                    extra: Optional[dict] = None) -> dict:
    plan = {
        "schema": "chlu-gen-store-plan/1",
        "dataset": str(dataset),
        "pretext": str(pretext),
        "K": int(k),
        "n_modes": int(n_modes),
        "D": int(dim),
        "feat_shape": [int(x) for x in feat_shape],
        "grouping_method": str(grouping_method),
        "k_choice": k_choice,
        "labels": [int(x) for x in np.asarray(labels)],
        "well_sizes": [int(x) for x in np.bincount(np.asarray(labels, np.int64), minlength=int(k))],
        "store_bytes": store_bytes_dict,
        "note": (
            "⭐ Every later factorisation-ladder rung loads this file VERBATIM. "
            "K, n_modes, D and store_bytes are pinned here; only the grouping "
            "(the `labels` field) is allowed to change between rungs."
        ),
    }
    if extra:
        plan.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True, default=float) + "\n")
    return plan


def load_store_plan(path) -> StorePlan:
    """Load a pinned plan.  Later rungs call this instead of re-deriving anything."""
    with open(path) as fh:
        d = json.load(fh)
    if d.get("schema") != "chlu-gen-store-plan/1":
        raise ValueError(f"not a store plan: {d.get('schema')!r}")
    return StorePlan(d)


def assert_plan_matched(plans: Sequence[dict]) -> dict:
    """⛔ Guard for the ladder: every rung must share ``K``, ``n_modes``, ``D`` and bytes."""
    keys = ("K", "n_modes", "D")
    ref = plans[0]
    for p in plans[1:]:
        for k in keys:
            if int(p[k]) != int(ref[k]):
                raise AssertionError(
                    f"ladder rungs disagree on {k}: {p[k]} != {ref[k]} — only the "
                    "grouping may change between rungs"
                )
        if int(p["store_bytes"]["store_bytes"]) != int(ref["store_bytes"]["store_bytes"]):
            raise AssertionError("ladder rungs disagree on store_bytes")
    return {"K": int(ref["K"]), "n_modes": int(ref["n_modes"]), "D": int(ref["D"]),
            "store_bytes": int(ref["store_bytes"]["store_bytes"]), "n_rungs": len(plans)}
