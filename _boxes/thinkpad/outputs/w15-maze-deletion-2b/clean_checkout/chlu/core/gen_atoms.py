"""⭐⭐ **Atom-mean lookup** — the like-for-like control the dynamical read must beat.

``gen-dynamical-read``'s entire case (its ``PREREG.md`` §2) is **a better EXPONENT at
matched bytes**: interpolation over the stored atoms beating piecewise-constant lookup
**over the same stored points**.  ⭐ **Without this module a win could be bought by
storing atoms instead of centres rather than by settling at all.**

So this module builds the rival that stores *the same points* and **simply does not
settle**:

* :func:`select_atoms` — the **atom set**.  Atoms are **actual TRAIN DATA POINTS**
  (⛔ never k-means centres: per-value weights over centres are the shipped codebook
  wearing a different hat).  Selection is **k-medoids** — Lloyd's algorithm, then each
  centroid is snapped to the nearest *real* row inside its own cluster — because the
  standing rule is **never weaken this rival**, and a random subsample is the weak
  member of the family (available as ``method="subsample"``, run only as a declared
  companion).
* :func:`atom_read` — the **read**: retrieve the ``k`` nearest atoms and **average
  them**.  ⛔ The query never reaches the output; only retrieved atom content does, so
  the channel is ``K·k·log2(N)`` bits and the comparison with the store's 30 is
  structural rather than rhetorical.
* :func:`atoms_for_budget` / :func:`atom_byte_ledger` — **matched bytes, asserted**.

⛔ **Matched bytes is the whole point.**  A well ``k`` owns ``d_k`` channels with
``Σ_k d_k = D``, so one atom costs ``D·4`` B however it is sliced and ``N`` atoms/well
cost exactly ``N·D·4``.  At ``D = 2048``: ``N = 64`` → **524,288 B**, which is the
deployed CLU store *exactly* (the store is 64 codewords **per well**); ``N = 512`` →
**4,194,304 B**.  :func:`atoms_for_budget` **refuses** a budget that is not an exact
multiple of ``D·4`` — ⛔ it does not floor silently, because a "matched-bytes" arm that
quietly under-spends is not matched.

⭐ **Two scopes, both byte-identical, both swept, validation picks.**

* ``per_well`` — the like-for-like structure (the landscape settles **per well**); the
  atom set is chosen **independently per well** on that well's own channels, so the
  effective partition is a product code.  ⭐ This is the canonical set that is
  persisted for ``gen-dynamical-read``.
* ``global`` — whole-``D`` nearest-atom lookup over ``N`` whole rows.  The weaker
  member; it is here so that "the control is as strong as it can be" is a **measured**
  statement, ⛔ not an assertion.

⚠ **Randomness.**  Every stream here is NumPy ``default_rng(SeedSequence)`` keyed on
``[root, budget_tag, scope_tag, well]``, which is decorrelated by construction.  ⛔ No
JAX key is derived by lengthening a ``split``: with
``jax_threefry_partitionable=True``, ``split(k, 2)[1] IS split(k, 3)[1]``, so two arms
built that way are **bit-identical rather than decorrelated**.  This module needs no
JAX key at all, and that is deliberate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..eval.rivals.gen_knn import METRICS, Normaliser, knn_search

#: ⭐ ``kmedoid`` is shipped (strongest honest member); ``subsample`` is the declared
#: weak companion and ⛔ is never the default.
ATOM_METHODS = ("kmedoid", "subsample")
#: ⭐ Both cost ``N·D·4`` B.  ⛔ Neither is allowed to be the "default" by omission.
SCOPES = ("per_well", "global")


# ---------------------------------------------------------------------------
# 1. ⛔ The byte ledger — matched bytes, asserted, never floored
# ---------------------------------------------------------------------------


def atoms_for_budget(budget_bytes: int, dim: int, itemsize: int = 4) -> int:
    """Atoms **per well** that spend ``budget_bytes`` **exactly**.

    ⛔ Refuses a budget that is not an exact multiple of ``dim*itemsize``.  Flooring
    here would let a "matched-bytes" arm silently under-spend, which is the one failure
    this control cannot afford: an under-resourced rival hands the dynamical read a
    false win.
    """
    per = int(dim) * int(itemsize)
    b = int(budget_bytes)
    if per <= 0:
        raise ValueError(f"dim*itemsize must be positive, got {per}")
    if b % per != 0:
        raise ValueError(
            f"⛔ budget {b} B is not an exact multiple of D*itemsize = {per} B "
            f"({b / per:.4f} atoms): matched bytes means EXACTLY matched. Choose a "
            "budget on the lattice or state the mismatch."
        )
    n = b // per
    if n < 1:
        raise ValueError(f"budget {b} B holds no atom at D={dim}")
    return int(n)


def atom_byte_ledger(n_atoms: int, well_sizes: Sequence[int], budget_bytes: int,
                     scope: str = "per_well", itemsize: int = 4) -> dict:
    """The realised ledger, in the shape ``store_bytes.json`` reports the store's."""
    d = [int(x) for x in well_sizes]
    D = int(sum(d))
    total = int(n_atoms) * D * int(itemsize)
    return {
        "scope": str(scope),
        "n_atoms_per_well": int(n_atoms),
        "n_wells": len(d),
        "well_sizes": d,
        "dim": D,
        "dtype": "float32",
        "bytes_formula": "N * sum_k d_k * 4 = N * D * 4",
        "realised_bytes": total,
        "budget_bytes": int(budget_bytes),
        "slack_bytes": int(int(budget_bytes) - total),
        "matched_exactly": bool(total == int(budget_bytes)),
        "bits_per_read_formula": "K * k * log2(N)",
        "note": (
            "⭐ an atom costs D*4 B however it is sliced across wells, so N atoms/well "
            "cost exactly N*D*4 — byte-identical to the store's n_modes*D*4"
        ),
    }


def bits_per_read(n_atoms: int, k: int, n_wells: int, scope: str = "per_well") -> float:
    """``K·k·log2(N)`` per-well, ``k·log2(N)`` global."""
    kk = float(int(k) * np.log2(max(int(n_atoms), 1)))
    return kk * float(int(n_wells)) if scope == "per_well" else kk


# ---------------------------------------------------------------------------
# 2. The atom set — ⭐ DATA POINTS, k-medoids, BLAS-fused
# ---------------------------------------------------------------------------


def _sq_dists(X: np.ndarray, C: np.ndarray, xsq: np.ndarray, chunk: int) -> np.ndarray:
    """``(n, m)`` squared distances, chunked (⚠ 16 GB machine)."""
    csq = np.einsum("ij,ij->i", C, C)
    out = np.empty((X.shape[0], C.shape[0]), np.float32)
    for i in range(0, X.shape[0], int(chunk)):
        Xi = X[i : i + int(chunk)]
        out[i : i + int(chunk)] = (
            xsq[i : i + int(chunk), None] - 2.0 * (Xi @ C.T) + csq[None, :]
        )
    return out


def _kmeanspp(X: np.ndarray, m: int, rng: np.random.Generator,
              xsq: np.ndarray) -> np.ndarray:
    """k-means++ seeding, BLAS-fused.

    ⚠ Functionally the same rule as ``gen_knn._kmeanspp``; it is re-implemented here
    only because that one re-materialises an ``(n, d)`` difference array per centre,
    which at ``m = 512`` and ``n = 32,768`` is minutes of pure allocation.  ⛔ The
    rival's own module is **not** edited — nothing that ``knn_proto`` touches moves.
    """
    n = int(X.shape[0])
    j = int(rng.integers(n))
    idx = [j]
    d2 = np.maximum(xsq - 2.0 * (X @ X[j]) + float(xsq[j]), 0.0)
    for _ in range(1, int(m)):
        total = float(d2.sum())
        p = (d2 / total) if total > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, np.maximum(xsq - 2.0 * (X @ X[j]) + float(xsq[j]), 0.0))
    return np.asarray(idx, np.int64)


def _lloyd(X: np.ndarray, C: np.ndarray, iters: int, chunk: int,
           xsq: np.ndarray) -> tuple:
    n, m = int(X.shape[0]), int(C.shape[0])
    assign = np.zeros(n, np.int64)
    inertia = float("nan")
    n_iter = 0
    for it in range(int(iters)):
        Dm = _sq_dists(X, C, xsq, chunk)
        assign = np.argmin(Dm, axis=1)
        inertia = float(np.take_along_axis(Dm, assign[:, None], axis=1).sum())
        del Dm
        newC = C.copy()
        counts = np.bincount(assign, minlength=m)
        for c in np.flatnonzero(counts):
            newC[c] = X[assign == c].mean(axis=0)
        moved = float(np.max(np.abs(newC - C))) if m else 0.0
        C = newC
        n_iter = int(it) + 1
        if moved == 0.0:
            break
    return C, assign, inertia, n_iter


def kmedoid_rows(X: np.ndarray, n_atoms: int, rng: np.random.Generator,
                 iters: int = 25, chunk: int = 4096) -> tuple:
    """⭐ ``n_atoms`` **real row indices**: Lloyd, then snap each centroid to the
    nearest actual row **inside its own cluster**.

    ⛔ The returned rows are ``X``-row indices, so the deployed content is genuinely a
    stored data point — the thing the landscape writes.  ⛔ Never a centroid.

    Empty clusters are back-filled by **farthest-point** from the already-chosen
    medoids, so the realised count is **exactly** ``n_atoms`` distinct rows: a rival
    that silently deploys fewer atoms than its budget buys is a weakened rival.
    """
    X = np.asarray(X, np.float32)
    n = int(X.shape[0])
    m = int(min(int(n_atoms), n))
    xsq = np.einsum("ij,ij->i", X, X)
    seeds = _kmeanspp(X, m, rng, xsq)
    C, assign, inertia, n_iter = _lloyd(X, X[seeds].copy(), int(iters), int(chunk), xsq)

    rows = np.full(m, -1, np.int64)
    csq = np.einsum("ij,ij->i", C, C)
    for c in range(m):
        members = np.flatnonzero(assign == c)
        if members.size == 0:
            continue
        d = xsq[members] - 2.0 * (X[members] @ C[c]) + csq[c]
        rows[c] = int(members[int(np.argmin(d))])
    empty = int(np.count_nonzero(rows < 0))
    if empty:
        chosen = rows[rows >= 0]
        d2 = np.min(_sq_dists(X, X[chosen], xsq, int(chunk)), axis=1)
        d2[chosen] = -1.0
        for c in np.flatnonzero(rows < 0):
            j = int(np.argmax(d2))
            rows[c] = j
            d2 = np.minimum(d2, np.maximum(xsq - 2.0 * (X @ X[j]) + float(xsq[j]), 0.0))
            d2[j] = -1.0
    rows = np.sort(rows)
    if np.unique(rows).size != m:
        raise AssertionError(
            f"⛔ k-medoids produced {np.unique(rows).size} distinct atoms for a budget "
            f"of {m}: the arm would deploy fewer bytes than it is entitled to, which "
            "WEAKENS the rival. ⛔ Stop and report."
        )
    return rows, C, {"method": "kmedoid", "n_atoms": m, "n_pool_rows": n,
                     "lloyd_iters_run": int(n_iter), "lloyd_iters_max": int(iters),
                     "inertia": float(inertia), "empty_clusters_backfilled": empty}


def subsample_rows(X: np.ndarray, n_atoms: int, rng: np.random.Generator) -> tuple:
    """⚠ The **declared weak companion**: a seeded random subsample of real rows."""
    n = int(np.asarray(X).shape[0])
    m = int(min(int(n_atoms), n))
    rows = np.sort(rng.permutation(n)[:m]).astype(np.int64)
    return rows, None, {"method": "subsample", "n_atoms": m, "n_pool_rows": n,
                        "note": "⚠ the weak member of the family, a companion only"}


@dataclass
class AtomSet:
    """The atoms **as deployed** — the bytes that would actually ship.

    ``rows[k]`` are indices into the store-train pool, ``values[k]`` is
    ``(N, d_k) float32`` in **raw** feature space (the read is in raw space, exactly as
    ``gen_knn.knn_read``).  For ``scope="global"`` there is one well covering all ``D``
    channels.
    """

    scope: str
    well_channels: tuple
    rows: tuple
    values: tuple
    info: dict
    #: ⚠ **FIT-TIME ONLY, 0 DEPLOYED BYTES.**  The k-means centroids the medoids were
    #: snapped to.  ⛔ They are **not** an arm — decoding a centroid is the shipped
    #: codebook's mechanism, not this one's.  They exist so the **medoid penalty**
    #: (data point vs centre, at identical N) is a *measured* quantity rather than a
    #: confound between "atoms vs centres" and "my k-means vs their trained store".
    centroids: Optional[tuple] = None

    @property
    def n_atoms(self) -> int:
        return int(self.values[0].shape[0])

    @property
    def n_wells(self) -> int:
        return len(self.values)

    @property
    def well_sizes(self) -> list:
        return [int(v.shape[1]) for v in self.values]

    @property
    def dim(self) -> int:
        return int(sum(self.well_sizes))

    def deployed_bytes(self) -> int:
        return int(sum(v.size * v.dtype.itemsize for v in self.values))

    def sha256(self) -> str:
        h = hashlib.sha256()
        for r, v in zip(self.rows, self.values, strict=True):
            h.update(np.ascontiguousarray(np.asarray(r, np.int64)).tobytes())
            h.update(np.ascontiguousarray(np.asarray(v, np.float32)).tobytes())
        return h.hexdigest()


def select_atoms(X_pool: np.ndarray, well_channels: Sequence[Sequence[int]],
                 n_atoms: int, seed: int, scope: str = "per_well",
                 method: str = "kmedoid", iters: int = 25, chunk: int = 4096,
                 budget_tag: int = 0, log=None) -> AtomSet:
    """⭐ The stored atoms — **the same points the landscape will use**.

    ``per_well``: the set is chosen **independently per well** on that well's own
    channels (a product code).  ``global``: ``n_atoms`` whole rows.  ⛔ Both cost
    ``n_atoms · D · 4`` B; the scope is a structure choice, never a byte advantage.

    ⚠ Streams are ``default_rng([seed, budget_tag, scope_tag, well])`` — decorrelated
    by NumPy's ``SeedSequence``, ⛔ never by lengthening a JAX split.
    """
    if method not in ATOM_METHODS:
        raise ValueError(f"unknown atom method {method!r} (have {ATOM_METHODS})")
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} (have {SCOPES})")
    X = np.asarray(X_pool, np.float32)
    if X.ndim > 2:
        X = X.reshape((X.shape[0], -1))
    D = int(X.shape[1])
    chans = ([tuple(int(i) for i in c) for c in well_channels] if scope == "per_well"
             else [tuple(range(D))])
    got = sorted(i for c in chans for i in c)
    if scope == "per_well" and got != list(range(D)):
        raise AssertionError(
            f"⛔ the wells cover {len(got)} channels, not D={D}: a partial cover would "
            "silently change the byte ledger"
        )
    rows_out, vals_out, cents_out, infos = [], [], [], []
    scope_tag = SCOPES.index(scope)
    for k, c in enumerate(chans):
        Xk = np.ascontiguousarray(X[:, np.asarray(c, np.int64)])
        rng = np.random.default_rng([int(seed), int(budget_tag), int(scope_tag), int(k)])
        if method == "kmedoid":
            rows, cents, info = kmedoid_rows(Xk, int(n_atoms), rng, iters=int(iters),
                                             chunk=int(chunk))
        else:
            rows, cents, info = subsample_rows(Xk, int(n_atoms), rng)
        rows_out.append(rows)
        cents_out.append(None if cents is None else np.ascontiguousarray(cents,
                                                                        np.float32))
        vals_out.append(np.ascontiguousarray(Xk[rows], np.float32))
        infos.append({**info, "well": int(k), "d_k": int(Xk.shape[1])})
        if log is not None:
            log(f"    [atoms] {scope} well {k}: d_k={Xk.shape[1]} -> {rows.size} "
                f"{method} atoms (inertia {info.get('inertia', float('nan')):.4g}, "
                f"{info.get('lloyd_iters_run', 0)} Lloyd iters)")
        del Xk
    return AtomSet(
        scope=str(scope),
        well_channels=tuple(chans),
        rows=tuple(rows_out),
        values=tuple(vals_out),
        centroids=(None if any(c is None for c in cents_out) else tuple(cents_out)),
        info={
            "method": str(method),
            "seed": int(seed),
            "budget_tag": int(budget_tag),
            "per_well": infos,
            "rng": ("numpy default_rng([seed, budget_tag, scope_tag, well]) — "
                    "SeedSequence-decorrelated; ⛔ no JAX split is lengthened"),
            "atoms_are": ("⭐ ACTUAL TRAIN DATA POINTS from the store-train pool. "
                          "⛔ Never k-means centres."),
        },
    )


def save_atoms(atoms: AtomSet, path) -> dict:
    """Persist the atom set so the landscape loads **these** atoms, sha256-verified."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scope": np.asarray(atoms.scope)}
    for k, (r, v, c) in enumerate(
        zip(atoms.rows, atoms.values, atoms.well_channels, strict=True)
    ):
        payload[f"rows_{k}"] = np.asarray(r, np.int64)
        payload[f"values_{k}"] = np.asarray(v, np.float32)
        payload[f"channels_{k}"] = np.asarray(c, np.int64)
    payload["n_wells"] = np.asarray(atoms.n_wells, np.int64)
    payload["info_json"] = np.asarray(json.dumps(atoms.info))
    np.savez_compressed(p, **payload)
    return {"path": str(p), "sha256": atoms.sha256(), "n_atoms": atoms.n_atoms,
            "n_wells": atoms.n_wells, "scope": atoms.scope,
            "deployed_bytes": atoms.deployed_bytes()}


def load_atoms(path) -> AtomSet:
    import json

    z = np.load(path, allow_pickle=False)
    kw = int(z["n_wells"])
    return AtomSet(
        scope=str(z["scope"]),
        well_channels=tuple(tuple(int(i) for i in z[f"channels_{k}"]) for k in range(kw)),
        rows=tuple(np.asarray(z[f"rows_{k}"], np.int64) for k in range(kw)),
        values=tuple(np.asarray(z[f"values_{k}"], np.float32) for k in range(kw)),
        info=json.loads(str(z["info_json"])),
    )


# ---------------------------------------------------------------------------
# 3. The read — ⭐ retrieve, average, and NEVER settle
# ---------------------------------------------------------------------------


def atom_read(atoms: AtomSet, Q: np.ndarray, k: int, metric: str = "euclidean",
              normaliser: Optional[Normaliser] = None,
              use_centroids: bool = False) -> dict:
    """⭐ ``(N, D)`` read = the **mean of the retrieved atoms**, per well, in RAW space.

    ⛔ The query is used **only** to choose indices — it never enters the output, which
    is the same structural property ``gen_store.read_from_codes`` and
    ``gen_knn.knn_read`` have.  The channel is therefore ``K·k·log2(N)`` bits.

    ⭐ Search happens in the **normalised** space (if a normaliser is given) and the
    read in the **raw** space — ``gen_knn``'s convention, unchanged.

    ⚠ ``use_centroids=True`` swaps the deployed atoms for the fit-time **centroids**.
    ⛔ That is **not an arm** — it is the CENTROID CONTROL that isolates the medoid
    penalty at identical ``N``.  It deploys 0 bytes on this arm's path and is always
    labelled as a control, ⛔ never quoted as the atom-mean lookup's number.

    Returns the read, the per-well retrieved indices, the per-well nearest distance,
    and ⭐ the **configuration id** (the tuple of retrieved index sets), which is the
    atom arm's analogue of the store's K-tuple code and is what the distinct-output
    count is computed from.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r} (have {METRICS})")
    Qr = np.asarray(Q, np.float32)
    if Qr.ndim > 2:
        Qr = Qr.reshape((Qr.shape[0], -1))
    Qn = normaliser(Qr) if normaliser is not None else Qr
    n = int(Qr.shape[0])
    D = atoms.dim
    R = np.zeros((n, D), np.float32)
    idxs, d0 = [], []
    if use_centroids and atoms.centroids is None:
        raise ValueError(
            "⛔ this atom set carries no centroids (it was not fitted by k-medoids), "
            "so the centroid control cannot be run on it"
        )
    content = atoms.centroids if use_centroids else atoms.values
    for chan, vals in zip(atoms.well_channels, content, strict=True):
        cidx = np.asarray(chan, np.int64)
        A_raw = vals
        # ⭐ the atoms are normalised on the SAME statistics as the query, sliced to
        # this well's channels — ⛔ never re-fitted per well, which would make the
        # per-well and global scopes incomparable.
        if normaliser is not None and normaliser.mode != "none":
            A_search = ((A_raw - normaliser.mean[cidx][None, :])
                        / normaliser.std[cidx][None, :]).astype(np.float32)
        else:
            A_search = A_raw
        idx, dist = knn_search(A_search, Qn[:, cidx], int(k), str(metric))
        R[:, cidx] = A_raw[idx].mean(axis=1)
        idxs.append(np.asarray(idx, np.int64))
        d0.append(np.asarray(dist, np.float64)[:, 0])
        del A_search
    conf = np.concatenate([np.sort(i, axis=1) for i in idxs], axis=1)
    return {
        "read": R,
        "indices": idxs,
        "nearest_distance": np.stack(d0, axis=1) if d0 else np.zeros((n, 0)),
        "config_id": conf,
        "n_distinct_configs": int(np.unique(conf, axis=0).shape[0]),
        "bits_per_read": bits_per_read(atoms.n_atoms, int(k), atoms.n_wells,
                                       atoms.scope),
        "n": n,
    }


def relative_read_error(read: np.ndarray, target: np.ndarray) -> dict:
    """⭐ The exponent quantity: ``MSE(read, target) / Var(target)``.

    ⛔ Identical in form to ``exp_gen_variance_eq._hard_read_mse``'s ``relative_mse``,
    which is where the centroid quantiser's **0.2783** anchor came from — so the two
    numbers are comparable without a translation step.
    """
    R = np.asarray(read, np.float32).reshape((int(np.asarray(read).shape[0]), -1))
    T = np.asarray(target, np.float32).reshape((int(np.asarray(target).shape[0]), -1))
    err = float(np.mean((R - T) ** 2))
    var = float(np.mean((T - T.mean(axis=0, keepdims=True)) ** 2))
    return {"mse": err, "feature_variance": var,
            "relative_mse": err / var if var > 0 else float("nan"),
            "n": int(T.shape[0])}


def fitted_exponent(points: Sequence[tuple]) -> dict:
    """``d ln ε / d ln N`` through ``(N, ε)`` points — ⭐ **the claim is the exponent**.

    Two points give the exact slope; more are least-squares.  The effective
    participation ratio implied by high-rate theory is ``PR = -2 / slope``.
    """
    pts = [(float(n), float(e)) for n, e in points
           if np.isfinite(e) and e > 0 and n > 0]
    if len(pts) < 2:
        return {"slope": float("nan"), "implied_pr": float("nan"),
                "n_points": len(pts), "NOT_RUN": "fewer than two usable points"}
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "implied_pr": float(-2.0 / slope) if slope != 0 else float("inf"),
            "n_points": len(pts),
            "points": [{"n_atoms": p[0], "relative_mse": p[1]} for p in pts],
            "law": "eps ~ N**slope; high-rate theory gives slope = -2/PR"}
