"""gen store."""

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


def channel_correlation(X: np.ndarray, eps: float = 1e-08) -> np.ndarray:
    A = np.asarray(X, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0)
    dead = sd <= eps
    A = A / np.where(dead, 1.0, sd)
    C = A.T @ A / float(A.shape[0])
    C[dead, :] = 0.0
    C[:, dead] = 0.0
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def _affinity(corr: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(corr, dtype=np.float64))


def dead_channels(X: np.ndarray, eps: float = 1e-08) -> np.ndarray:
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
    min_well_frac: float = 0.1,
    return_info: bool = False,
):
    C = np.asarray(corr, dtype=np.float64)
    d = C.shape[0]
    dead = np.zeros(d, bool) if dead is None else np.asarray(dead, bool)
    live = np.flatnonzero(~dead)
    if live.size < int(k):
        raise ValueError(f"only {live.size} live coordinates for {k} wells")
    Cl = C[np.ix_(live, live)]
    floor = max(1, int(np.floor(float(min_well_frac) * live.size / int(k))))
    order = list(GROUPING_ORDER) if method == "auto" else [method]
    attempts, chosen, lab_live = ([], None, None)
    for m in order:
        lab = _compact_labels(_cluster_live(Cl, int(k), m), int(k), strict=False)
        sizes = np.bincount(lab, minlength=int(k))
        ok = bool(sizes.size == int(k) and sizes.min() >= floor)
        attempts.append(
            {
                "method": m,
                "sizes": [int(x) for x in sizes],
                "min_well_size": int(sizes.min()),
                "floor": int(floor),
                "accepted": ok,
            }
        )
        if ok:
            chosen, lab_live = (m, lab)
            break
    if lab_live is None:
        raise ValueError(
            f"every method in {order} returned a degenerate partition at k={k} (floor {floor}); attempts: {attempts}"
        )
    labels = np.zeros(d, np.int64)
    labels[live] = lab_live
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
    uniq = np.unique(lab)
    if strict and uniq.size != k:
        raise ValueError(
            f"clustering returned {uniq.size} non-empty wells, not {k} — an empty well is a degenerate partition and is reported, never silently merged"
        )
    remap = {int(u): i for i, u in enumerate(uniq)}
    return np.asarray([remap[int(x)] for x in lab], dtype=np.int64)


def random_grouping(reference_labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.asarray(rng.permutation(np.asarray(reference_labels)), dtype=np.int64)


def grouping_labels(
    corr: np.ndarray,
    k: int,
    grouping: str,
    seed: int,
    method: str = "auto",
    dead: Optional[np.ndarray] = None,
    min_well_frac: float = 0.1,
) -> tuple:
    ref, info = correlation_grouping(
        corr, k, method, dead=dead, min_well_frac=min_well_frac, return_info=True
    )
    if grouping == "correlation":
        return (ref, info)
    if grouping == "random":
        info = dict(info, randomised_from="correlation", random_seed=int(seed))
        return (random_grouping(ref, seed), info)
    raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")


def well_channels_from_labels(labels: np.ndarray, k: int) -> tuple:
    """``(K,)`` tuple of sorted coordinate-index tuples.  Integer indices only."""
    lab = np.asarray(labels, dtype=np.int64)
    return tuple((tuple((int(i) for i in np.flatnonzero(lab == j))) for j in range(int(k))))


def block_diagonality(corr: np.ndarray, labels: np.ndarray) -> dict:
    C = np.asarray(corr, dtype=np.float64) ** 2
    np.fill_diagonal(C, 0.0)
    lab = np.asarray(labels, dtype=np.int64)
    same = lab[:, None] == lab[None, :]
    total = float(C.sum())
    inside = float(C[same].sum())
    n = lab.size
    counts = np.bincount(lab)
    exp_pairs = float(np.sum(counts * (counts - 1))) / float(n * (n - 1))
    return {
        "in_block_fraction": inside / total if total > 0 else float("nan"),
        "in_block_pair_fraction": exp_pairs,
        "lift_over_size_matched_random": inside / total / exp_pairs
        if total > 0 and exp_pairs > 0
        else float("nan"),
        "well_sizes": [int(c) for c in counts],
    }


class GenStore(eqx.Module):
    modes: list
    well_channels: tuple = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    feat_shape: tuple = eqx.field(static=True)

    def __init__(self, well_channels, n_modes: int, feat_shape, key, init=None):
        self.well_channels = tuple((tuple((int(i) for i in w)) for w in well_channels))
        self.feat_shape = tuple((int(x) for x in feat_shape))
        self.dim = int(np.prod(self.feat_shape))
        flat = np.concatenate([np.asarray(w, np.int64) for w in self.well_channels])
        if flat.size != self.dim or np.unique(flat).size != self.dim:
            raise ValueError(
                f"the wells must PARTITION the {self.dim} coordinates (got {flat.size} entries, {np.unique(flat).size} distinct)"
            )
        keys = jax.random.split(key, len(self.well_channels))
        modes = []
        for j, (w, k) in enumerate(zip(self.well_channels, keys, strict=True)):
            if init is not None:
                modes.append(jnp.asarray(init[j], jnp.float32))
            else:
                modes.append(0.1 * jax.random.normal(k, (int(n_modes), len(w)), jnp.float32))
        self.modes = modes

    @property
    def n_wells(self) -> int:
        return len(self.well_channels)

    @property
    def n_modes_per_well(self) -> tuple:
        return tuple((int(np.asarray(m).shape[0]) for m in self.modes))

    @property
    def n_modes(self) -> int:
        ms = self.n_modes_per_well
        if len(set(ms)) != 1:
            raise ValueError(
                f"⛔ this store has a NON-UNIFORM allocation {ms} — there is no single `n_modes`. Use `store.n_modes_per_well`."
            )
        return int(ms[0])

    @property
    def bits_per_read(self) -> float:
        return float(sum((np.log2(float(m)) for m in self.n_modes_per_well)))

    def n_bytes(self) -> int:
        """Store size in bytes (float32 modes) — follows ``FactoredStore.n_bytes``."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return int(sum((int(np.asarray(x).size) for x in leaves)) * 4)


def store_bytes(store: GenStore) -> dict:
    per_well = [int(np.asarray(m).size) * 4 for m in store.modes]
    ms = store.n_modes_per_well
    uniform = len(set(ms)) == 1
    return {
        "store_bytes": int(sum(per_well)),
        "store_bytes_per_well": per_well,
        "n_wells": int(store.n_wells),
        "n_modes": int(ms[0]) if uniform else None,
        "n_modes_per_well": [int(x) for x in ms],
        "uniform_allocation": bool(uniform),
        "dim": int(store.dim),
        "dtype": "float32",
        "bytes_formula": "n_modes * sum_k d_k * 4 = n_modes * D * 4"
        if uniform
        else "sum_k M_k * d_k * 4  (⭐ a per-well allocation)",
        "bits_per_read": store.bits_per_read,
        "note": "The modes ARE the store. The addressing head is reported separately and is not part of this count; a byte-matched baseline matches store_bytes.",
    }


def _well_views(store: GenStore, X):
    """``X (N, D) -> [ (N, d_k) ]`` gathers.  Cast to the store's own dtype (§7.23)."""
    dt = store.modes[0].dtype
    Z = jnp.asarray(X, dt).reshape((X.shape[0], -1))
    return [Z[:, jnp.asarray(w, jnp.int32)] for w in store.well_channels]


def address_codes(store: GenStore, X) -> np.ndarray:
    out = []
    for z, m in zip(_well_views(store, X), store.modes, strict=True):
        d2 = jnp.sum(m * m, axis=-1)[None, :] - 2.0 * (z @ m.T)
        out.append(jnp.argmin(d2, axis=-1))
    return np.asarray(jnp.stack(out, axis=1), dtype=np.int64)


def read_from_codes(store: GenStore, codes) -> np.ndarray:
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
    params = list(inspect.signature(read_from_codes).parameters)
    if params != ["store", "codes"]:
        raise AssertionError(
            f"read_from_codes must take exactly (store, codes) — an extra argument is a potential encoder->decoder bypass; got {params}"
        )
    X = np.asarray(X, np.float32).reshape((np.asarray(X).shape[0], -1))
    codes = address_codes(store, X)
    y0 = read_from_codes(store, codes).reshape((X.shape[0], -1))
    ref = np.zeros_like(y0)
    for j, (w, m) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        ref[:, np.asarray(w, np.int64)] = np.asarray(m)[codes[:, j]]
    max_abs = float(np.max(np.abs(y0 - ref))) if y0.size else 0.0
    if max_abs != 0.0:
        raise AssertionError(
            f"the read output is not the pure codebook gather (max |delta| = {max_abs}) — something other than the codes reaches the output"
        )
    rng = np.random.default_rng(int(rng_seed))
    X2 = rng.random(X.shape).astype(np.float32) * float(np.abs(X).max() + 1.0)
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
        "note": "The output path has no argument through which encoder information could travel except the K integer codes.",
    }


def delete_store(store: GenStore, mode: str = "zero", seed: int = 0) -> GenStore:
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
        p = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))
    return Z[np.asarray(idx)].astype(np.float32)


def init_store(X: np.ndarray, well_channels, n_modes, feat_shape, key, seed: int) -> GenStore:
    ms = (
        [int(n_modes)] * len(list(well_channels))
        if np.isscalar(n_modes) or isinstance(n_modes, (int, np.integer))
        else [int(x) for x in n_modes]
    )
    if len(ms) != len(list(well_channels)):
        raise ValueError(
            f"n_modes has {len(ms)} entries but there are {len(list(well_channels))} wells"
        )
    if min(ms) < 1:
        raise ValueError(f"every well needs at least one mode, got {ms}")
    rng = np.random.default_rng(int(seed) * 2654435761 % 2**32)
    flat = np.asarray(X, np.float32).reshape((X.shape[0], -1))
    init = [
        _kmeanspp_init(flat[:, np.asarray(w, np.int64)], int(m), rng)
        for w, m in zip(well_channels, ms, strict=True)
    ]
    return GenStore(well_channels, int(ms[0]), feat_shape, key, init=init)


def train_store(store: GenStore, X: np.ndarray, cfg, key, log: Optional[Callable] = None) -> tuple:
    flat = jnp.asarray(np.asarray(X, np.float32).reshape((X.shape[0], -1)))
    n = int(flat.shape[0])
    batch = min(int(cfg.train_batch), n)
    steps = int(cfg.train_steps)
    beta = float(cfg.commitment_beta)
    t0, t1 = (float(cfg.tau_start), float(cfg.tau_end))
    opt = optax.adam(float(cfg.train_lr))
    opt_state = opt.init(eqx.filter(store, eqx.is_inexact_array))

    def loss_fn(st, xb, tau):
        views = _well_views(st, xb)
        soft = 0.0
        commit = 0.0
        for z, m in zip(views, st.modes, strict=True):
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
        return (eqx.apply_updates(st, updates), s, loss)

    history = []
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        tau = t0 * (t1 / t0) ** (i / max(1, steps - 1))
        store, opt_state, loss = step(store, opt_state, flat[idx], tau)
        history.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 5) == 0):
            log(f"    step {i + 1}/{steps}  loss={history[-1]:.5f}  tau={tau:.3f}")
    return (store, history, key)


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
    pass


def load_store_plan(path) -> StorePlan:
    with open(path) as fh:
        d = json.load(fh)
    if d.get("schema") != "nclu-gen-store-plan/1":
        raise ValueError(f"not a store plan: {d.get('schema')!r}")
    return StorePlan(d)


def adjusted_rand_index(labels_a, labels_b) -> float:
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
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))
