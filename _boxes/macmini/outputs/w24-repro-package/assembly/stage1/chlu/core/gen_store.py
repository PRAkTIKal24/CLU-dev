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
from typing import Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
GROUPINGS = ("correlation", "random")
GROUPING_METHODS = ("auto", "average", "ward", "spectral")
GROUPING_ORDER = ("average", "ward")
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
    def n_modes_per_well(self) -> tuple:
        """``(M_0, ..., M_{K-1})``.  ⭐ The wells may hold DIFFERENT numbers of modes
        (``gen-codebook-allocation``); this is the general accessor and
        :attr:`n_modes` is the uniform special case."""
        return tuple(int(np.asarray(m).shape[0]) for m in self.modes)

    @property
    def n_modes(self) -> int:
        """``M``, for a **uniform** allocation.

        ⛔ **Raises** when the wells hold different ``M_k``: a single ``M`` is then a
        WRONG number, not an approximate one, and it would silently reach a byte
        ledger / a one-hot width / an address book (``gen-codebook-allocation``).
        Use :attr:`n_modes_per_well`.
        """
        ms = self.n_modes_per_well
        if len(set(ms)) != 1:
            raise ValueError(
                f"⛔ this store has a NON-UNIFORM allocation {ms} — there is no single "
                "`n_modes`. Use `store.n_modes_per_well`."
            )
        return int(ms[0])

    @property
    def bits_per_read(self) -> float:
        """Width of the ONLY channel from the features to the decoder.

        ``sum_k log2(M_k)`` — identical to ``K·log2(M)`` for a uniform allocation and
        ⭐ correct for an allocated one, where ``K·log2(M_0)`` would not be.
        """
        return float(sum(np.log2(float(m)) for m in self.n_modes_per_well))

    def n_bytes(self) -> int:
        """Store size in bytes (float32 modes) — follows ``FactoredStore.n_bytes``."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return int(sum(int(np.asarray(x).size) for x in leaves) * 4)


def store_bytes(store: GenStore) -> dict:
    """⭐ Machine-readable byte accounting — ``gen-baselines`` byte-matches to this."""
    per_well = [int(np.asarray(m).size) * 4 for m in store.modes]
    ms = store.n_modes_per_well
    uniform = len(set(ms)) == 1
    return {
        "store_bytes": int(sum(per_well)),
        "store_bytes_per_well": per_well,
        "n_wells": int(store.n_wells),
        # ⛔ ``None`` rather than a lie when the allocation is non-uniform.
        "n_modes": int(ms[0]) if uniform else None,
        "n_modes_per_well": [int(x) for x in ms],
        "uniform_allocation": bool(uniform),
        "dim": int(store.dim),
        "dtype": "float32",
        "bytes_formula": (
            "n_modes * sum_k d_k * 4 = n_modes * D * 4" if uniform
            else "sum_k M_k * d_k * 4  (⭐ a per-well allocation)"
        ),
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
    X: np.ndarray, well_channels, n_modes, feat_shape, key, seed: int
) -> GenStore:
    """Modes seeded by k-means++ **on the store's own training features**.

    ``n_modes`` is either an ``int`` (the uniform allocation, unchanged) or a
    per-well sequence ``(M_0, …, M_{K-1})`` — ⭐ ``gen-codebook-allocation`` spends
    codewords where the distortion is, at constant total bytes.  ⛔ The RNG stream is
    identical in both cases (one ``default_rng`` consumed well-by-well in order), so
    a uniform call is bit-identical to the pre-allocation code path.
    """
    ms = ([int(n_modes)] * len(list(well_channels))
          if np.isscalar(n_modes) or isinstance(n_modes, (int, np.integer))
          else [int(x) for x in n_modes])
    if len(ms) != len(list(well_channels)):
        raise ValueError(
            f"n_modes has {len(ms)} entries but there are "
            f"{len(list(well_channels))} wells"
        )
    if min(ms) < 1:
        raise ValueError(f"every well needs at least one mode, got {ms}")
    rng = np.random.default_rng(int(seed) * 2_654_435_761 % (2**32))
    flat = np.asarray(X, np.float32).reshape((X.shape[0], -1))
    init = [_kmeanspp_init(flat[:, np.asarray(w, np.int64)], int(m), rng)
            for w, m in zip(well_channels, ms, strict=True)]
    return GenStore(well_channels, int(ms[0]), feat_shape, key, init=init)


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


class StorePlan(dict):
    """A loadable store configuration.  ⛔ A later rung that re-derives ``K`` or the
    byte count voids the ladder comparison, so the plan is an artifact, not prose."""


def load_store_plan(path) -> StorePlan:
    """Load a pinned plan.  Later rungs call this instead of re-deriving anything."""
    with open(path) as fh:
        d = json.load(fh)
    if d.get("schema") != "chlu-gen-store-plan/1":
        raise ValueError(f"not a store plan: {d.get('schema')!r}")
    return StorePlan(d)


def adjusted_rand_index(labels_a, labels_b) -> float:
    """Adjusted Rand index between two partitions of the same ``D`` coordinates.

    Chance-corrected: 1.0 for identical partitions (up to relabelling), ~0 for
    independent ones, and it can go negative.  ⭐ This is the statistic R24 rider 1
    asks for, and it is the right one here precisely because well **indices** carry
    no meaning (⛔ standing convention: wells are integer-labelled and nothing
    else), so any agreement measure must be invariant to relabelling.

    Implemented from the contingency table rather than imported so the ladder does
    not acquire a metric whose conventions it cannot see; cross-checked against
    ``sklearn.metrics.adjusted_rand_score`` in the tests.
    """
    a = np.asarray(labels_a, np.int64).ravel()
    b = np.asarray(labels_b, np.int64).ravel()
    if a.size != b.size:
        raise ValueError(f"partitions cover different numbers of coordinates: {a.size} != {b.size}")
    if a.size == 0:
        raise ValueError("an empty partition has no adjusted Rand index")
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((ua.size, ub.size), np.float64)
    np.add.at(cont, (ia, ib), 1.0)

    def _c2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = float(_c2(cont).sum())
    sum_i = float(_c2(cont.sum(axis=1)).sum())
    sum_j = float(_c2(cont.sum(axis=0)).sum())
    n_c2 = float(_c2(np.float64(a.size)))
    expected = sum_i * sum_j / n_c2
    maximum = 0.5 * (sum_i + sum_j)
    if maximum == expected:  # both partitions are trivial (one well, or all singletons)
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


