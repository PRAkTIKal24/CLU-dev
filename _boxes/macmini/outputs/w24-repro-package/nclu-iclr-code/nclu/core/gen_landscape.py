"""gen landscape."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional, Sequence
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ATOM_SELECTIONS = ("random", "kmeanspp", "first")
READ_MODES = ("state", "atom_mixture")
BYPASS_STAGES = ("none", "launch", "settle", "read")
LAUNCH_MODES = ("clamp_atom", "clamp_atom_thresholded", "posterior_mean", "prior_atom")


def arm_key(key, label: str):
    h = hashlib.sha256(str(label).encode("utf-8")).digest()
    return jax.random.fold_in(key, int(np.frombuffer(h[:4], np.uint32)[0]))


def _kmeanspp_rows(Z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = int(Z.shape[0])
    n = int(min(n, m))
    if n <= 0:
        return np.zeros((0,), np.int64)
    idx = [int(rng.integers(m))]
    d2 = np.sum((Z - Z[idx[0]]) ** 2, axis=1)
    for _ in range(1, n):
        tot = float(d2.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            rest = np.setdiff1d(np.arange(m), np.asarray(idx))
            idx.extend(rng.permutation(rest)[: n - len(idx)].tolist())
            break
        j = int(rng.choice(m, p=d2 / tot))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))
    return np.asarray(sorted(set(idx)), np.int64)


def select_atom_rows(Z_train: np.ndarray, n_atoms: int, selection: str, seed: int) -> np.ndarray:
    if selection not in ATOM_SELECTIONS:
        raise ValueError(f"selection must be one of {ATOM_SELECTIONS}, got {selection!r}")
    Z = np.asarray(Z_train, np.float32)
    Z = Z.reshape((Z.shape[0], -1))
    n = int(min(int(n_atoms), Z.shape[0]))
    rng = np.random.default_rng(int(seed) * 9176083 + 20260914)
    if selection == "first":
        return np.arange(n, dtype=np.int64)
    if selection == "random":
        return np.sort(rng.permutation(Z.shape[0])[:n]).astype(np.int64)
    return _kmeanspp_rows(Z, n, rng)


class AtomStore(eqx.Module):
    atoms: list
    well_channels: tuple = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    feat_shape: tuple = eqx.field(static=True)
    atom_rows: tuple = eqx.field(static=True)
    selection: str = eqx.field(static=True)

    def __init__(self, well_channels, feat_shape, Z_train, atom_rows, selection: str = "kmeanspp"):
        self.well_channels = tuple((tuple((int(i) for i in w)) for w in well_channels))
        self.feat_shape = tuple((int(x) for x in feat_shape))
        self.dim = int(np.prod(self.feat_shape))
        flat = np.concatenate([np.asarray(w, np.int64) for w in self.well_channels])
        if flat.size != self.dim or np.unique(flat).size != self.dim:
            raise ValueError(
                f"the wells must PARTITION the {self.dim} coordinates (got {flat.size} entries, {np.unique(flat).size} distinct)"
            )
        rows = np.asarray(atom_rows, np.int64)
        Z = np.asarray(Z_train, np.float32).reshape((np.asarray(Z_train).shape[0], -1))
        if Z.shape[1] != self.dim:
            raise ValueError(f"Z_train has width {Z.shape[1]}, store dim is {self.dim}")
        if rows.size == 0:
            raise ValueError("an atom store with zero atoms is not a store")
        self.atom_rows = tuple((int(r) for r in rows))
        self.selection = str(selection)
        self.atoms = [
            jnp.asarray(Z[np.ix_(rows, np.asarray(w, np.int64))], jnp.float32)
            for w in self.well_channels
        ]

    @property
    def n_wells(self) -> int:
        return len(self.well_channels)

    @property
    def n_atoms(self) -> int:
        return int(self.atoms[0].shape[0])

    @property
    def well_sizes(self) -> list:
        return [len(w) for w in self.well_channels]

    def n_bytes(self) -> int:
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return int(sum((int(np.asarray(x).size) for x in leaves)) * 4)


def build_atom_store(
    well_channels,
    feat_shape,
    Z_train: np.ndarray,
    n_atoms: int,
    selection: str = "kmeanspp",
    seed: int = 0,
) -> AtomStore:
    rows = select_atom_rows(Z_train, int(n_atoms), str(selection), int(seed))
    return AtomStore(well_channels, feat_shape, Z_train, rows, selection=str(selection))


def assert_atoms_are_train_rows(store: AtomStore, Z_train: np.ndarray) -> dict:
    Z = np.asarray(Z_train, np.float32).reshape((np.asarray(Z_train).shape[0], -1))
    rows = np.asarray(store.atom_rows, np.int64)
    worst = 0.0
    for w, a in zip(store.well_channels, store.atoms, strict=True):
        ref = Z[np.ix_(rows, np.asarray(w, np.int64))]
        worst = max(worst, float(np.max(np.abs(np.asarray(a) - ref))) if ref.size else 0.0)
    if worst != 0.0:
        raise AssertionError(
            f"⛔ THE ATOMS ARE NOT STORED TRAIN ROWS (max |Δ| vs Z_train[atom_rows] = {worst}). A landscape whose minima are fitted CENTRES inherits the codebook loss and the arm is void."
        )
    n_distinct = int(np.unique(rows).size)
    collapsed = []
    for k, a in enumerate(store.atoms):
        A = np.asarray(a)
        if A.shape[0] > 1 and float(np.max(np.abs(A - A.mean(axis=0, keepdims=True)))) == 0.0:
            collapsed.append(k)
    if collapsed:
        raise AssertionError(
            f"⛔ wells {collapsed} have atoms identical to their own mean — the store has collapsed to a centre"
        )
    return {
        "atoms_are_train_rows": True,
        "max_abs_delta_vs_train_rows": worst,
        "n_atoms": int(store.n_atoms),
        "n_distinct_atom_rows": n_distinct,
        "atom_selection": str(store.selection),
        "note": "⛔ The minima are NOT the k-means centres: every atom is a verbatim train feature row. Per-value weights over centres would be the old flat count table wearing a different hat.",
    }


def store_byte_footprint(n_atoms: int, well_sizes: Sequence[int]) -> dict:
    d = int(sum((int(x) for x in well_sizes)))
    return {
        "n_atoms_per_well": int(n_atoms),
        "dim": d,
        "well_sizes": [int(x) for x in well_sizes],
        "store_bytes": int(int(n_atoms) * d * 4),
        "dtype": "float32",
        "bytes_formula": "N * sum_k d_k * 4 = N * D * 4",
        "note": "⭐ matched bytes with the codebook at N = M: the landscape gets NO extra points and must win on the EXPONENT alone (PROTOCOL §2).",
    }


def coupling_byte_footprint(rank: int, well_sizes: Sequence[int]) -> dict:
    ds = [int(x) for x in well_sizes]
    K = len(ds)
    total = 0
    pairs = []
    for k in range(K):
        for lo in range(k + 1, K):
            b = int(rank) * (ds[k] + ds[lo]) * 4
            pairs.append({"k": k, "l": lo, "bytes": int(b)})
            total += b
    return {
        "rank": int(rank),
        "n_pairs": len(pairs),
        "coupling_bytes": int(total),
        "per_pair": pairs,
        "dtype": "float32",
        "bytes_formula": "sum_{k<l} r*(d_k + d_l)*4",
        "identity": "= r*(K-1)*D*4 when the wells partition D",
        "derived_free": "the per-well centre m_k and radius s_k are functions of the ATOMS (0 additional bytes)",
        "note": "⛔ EXTRA to store_bytes — a byte-matched rival must be told about it.",
    }


def landscape_byte_footprint(
    n_atoms: int, well_sizes: Sequence[int], rank: int, budget_bytes: int, label: str = ""
) -> dict:
    st = store_byte_footprint(n_atoms, well_sizes)
    cp = coupling_byte_footprint(rank, well_sizes)
    total = int(st["store_bytes"]) + int(cp["coupling_bytes"])
    b = int(budget_bytes)
    return {
        "label": str(label),
        "store": st,
        "coupling": cp,
        "total_bytes": total,
        "budget_bytes": b,
        "multiple_of_budget": total / b if b else float("nan"),
        "d3_rule": "store + coupling <= 2x budget",
        "d3_rule_satisfied": bool(b and total <= 2 * b),
        "d3_rule_binding": bool(b and total == 2 * b),
    }


def decode_width(store: AtomStore, read_mode: str = "state") -> dict:
    d = int(store.dim)
    return {
        "declared_channel_floats": d,
        "declared_channel_bits": int(d * 32),
        "dtype": "float32",
        "parity_with": "knn_proto (also D floats to the decoder)",
        "read_mode": str(read_mode),
        "note": "⛔ NOT K*log2(M): the settled state is continuous. The width is declared at the store's full float width and the empirical spread of the settled states is reported beside it (settled_state_spread).",
    }


def _pair_d2(A: np.ndarray) -> np.ndarray:
    g = (A * A).sum(axis=1)
    return np.maximum(g[:, None] + g[None, :] - 2.0 * (A @ A.T), 0.0)


def atom_bandwidths(store: AtomStore, frac: float = 1.0, rule: str = "nn_median") -> np.ndarray:
    out = []
    for a in store.atoms:
        A = np.asarray(a, np.float64)
        if A.shape[0] < 2:
            out.append(1.0)
            continue
        d2 = _pair_d2(A)
        np.fill_diagonal(d2, np.inf)
        if rule == "nn_median":
            s = float(np.median(np.sqrt(np.min(d2, axis=1))))
        elif rule == "pairwise_median":
            iu = np.triu_indices(A.shape[0], 1)
            s = float(np.median(np.sqrt(np.where(np.isfinite(d2), d2, 0.0)[iu])))
        else:
            raise ValueError(f"unknown bandwidth rule {rule!r}")
        out.append(s if s > 0 else 1.0)
    return np.asarray(out, np.float32) * float(frac)


def atom_centres(store: AtomStore) -> list:
    return [np.asarray(a, np.float64).mean(axis=0) for a in store.atoms]


def atom_radii(store: AtomStore) -> np.ndarray:
    """``s_k`` — the RMS radius of well ``k``'s atoms about ``m_k``.  0 extra bytes."""
    out = []
    for a, m in zip(store.atoms, atom_centres(store), strict=True):
        A = np.asarray(a, np.float64)
        r = float(np.sqrt(np.mean((A - m[None, :]) ** 2)))
        out.append(r if r > 0 else 1.0)
    return np.asarray(out, np.float64)


class LowRankCoupling(eqx.Module):
    u: list
    v: list
    pairs: tuple = eqx.field(static=True)
    centres: list
    radii: jnp.ndarray
    rank: int = eqx.field(static=True)

    def __init__(self, u, v, pairs, centres, radii, rank):
        self.u = [jnp.asarray(x, jnp.float32) for x in u]
        self.v = [jnp.asarray(x, jnp.float32) for x in v]
        self.pairs = tuple(((int(a), int(b)) for a, b in pairs))
        self.centres = [jnp.asarray(x, jnp.float32) for x in centres]
        self.radii = jnp.asarray(radii, jnp.float32)
        self.rank = int(rank)

    def n_bytes(self) -> int:
        return int(sum((int(np.asarray(x).size) for x in self.u + self.v)) * 4)


def fit_low_rank_coupling(
    store: AtomStore, Z_train: np.ndarray, rank: int, normalise: bool = True
) -> LowRankCoupling:
    Z = np.asarray(Z_train, np.float32).reshape((np.asarray(Z_train).shape[0], -1))
    cen = atom_centres(store)
    rad = atom_radii(store)
    blocks = [
        (Z[:, np.asarray(w, np.int64)].astype(np.float64) - cen[k][None, :]) / rad[k]
        for k, w in enumerate(store.well_channels)
    ]
    n = max(int(Z.shape[0]), 1)
    K = store.n_wells
    r = int(rank)
    us, vs, pairs = ([], [], [])
    for k in range(K):
        for lo in range(k + 1, K):
            C = blocks[k].T @ blocks[lo] / n
            rr = int(min(r, min(C.shape)))
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            U, S, Vt = (U[:, :rr], S[:rr], Vt[:rr])
            s0 = float(S[0]) if S.size and S[0] > 0 else 1.0
            sc = np.sqrt(S / (s0 if normalise else 1.0))
            Uk = (U * sc[None, :]).T
            Vl = Vt * sc[:, None]
            if rr < r:
                Uk = np.pad(Uk, ((0, r - rr), (0, 0)))
                Vl = np.pad(Vl, ((0, r - rr), (0, 0)))
            us.append(Uk)
            vs.append(Vl)
            pairs.append((k, lo))
    return LowRankCoupling(us, vs, pairs, cen, rad, r)


def coupling_spectral_norms(coupling: LowRankCoupling) -> np.ndarray:
    out = []
    for U, V in zip(coupling.u, coupling.v, strict=True):
        Un, Vn = (np.asarray(U, np.float64), np.asarray(V, np.float64))
        ev = np.linalg.eigvals(Un @ Un.T @ (Vn @ Vn.T))
        out.append(float(np.sqrt(max(float(np.max(ev.real)), 0.0))))
    return np.asarray(out, np.float64)


def coupling_stability(land: AtomLandscape) -> dict:
    c = land.coupling
    K = int(land.n_wells)
    if c is None or land.coupling_strength == 0.0:
        return {
            "coupling_present": c is not None,
            "coupling_strength": float(land.coupling_strength),
            "bounded_below": True,
            "max_safe_coupling_strength": float("inf"),
            "rule": "no coupling term: the energy is the (coercive) kernel energy",
        }
    norms = coupling_spectral_norms(c)
    rad = np.asarray(c.radii, np.float64)
    d = rad**2 / 2.0
    S = np.zeros((K, K), np.float64)
    for (k, lo), w in zip(c.pairs, norms, strict=True):
        S[k, lo] = S[lo, k] = float(w) / 2.0
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-300))
    rho = float(np.max(np.abs(np.linalg.eigvalsh(dinv[:, None] * S * dinv[None, :]))))
    lam_safe = float("inf") if rho <= 0 else 1.0 / rho
    C = np.diag(d) - float(land.coupling_strength) * S
    min_eig = float(np.min(np.linalg.eigvalsh(C)))
    return {
        "coupling_present": True,
        "coupling_strength": float(land.coupling_strength),
        "rank": int(c.rank),
        "pair_spectral_norms": [float(x) for x in norms],
        "well_radii": [float(x) for x in rad],
        "certificate_min_eigenvalue": min_eig,
        "bounded_below": bool(min_eig >= 0.0),
        "max_safe_coupling_strength": lam_safe,
        "rule": "E >= u^T C u - const with C_kk = s_k^2/2 and C_kl = -lambda*||W_kl||_2/2; coercive iff C >= 0, so lambda* = 1/rho(D^-1/2 S D^-1/2)",
        "note": "⛔ An unbounded-below landscape does not settle, it diverges — measured at E = -1.6e28 on the first smoke. The certificate is computed from the ATOMS, never hand-tuned.",
    }


def safe_coupling_strength(land: AtomLandscape, declared: float, safety_frac: float = 0.9) -> dict:
    cert = coupling_stability(land.with_strengths(coupling_strength=float(declared)))
    lam_star = float(cert["max_safe_coupling_strength"])
    cap = lam_star * float(safety_frac)
    used = float(min(float(declared), cap)) if np.isfinite(cap) else float(declared)
    return {
        "declared_coupling_strength": float(declared),
        "max_safe_coupling_strength": lam_star,
        "safety_frac": float(safety_frac),
        "coupling_strength_used": used,
        "certificate_bound_the_strength": bool(used < float(declared)),
        "certificate": cert,
    }


class AtomAddressBook(eqx.Module):
    log_prior: jnp.ndarray
    log_cond: list

    def __init__(self, log_prior, log_cond):
        self.log_prior = jnp.asarray(log_prior, jnp.float32)
        self.log_cond = [jnp.asarray(x, jnp.float32) for x in log_cond]

    @property
    def n_wells(self) -> int:
        return int(self.log_prior.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.log_prior.shape[1])

    @property
    def n_factors(self) -> int:
        return len(self.log_cond)


def soft_atom_assignment(
    store: AtomStore, Z: np.ndarray, sigma: np.ndarray, batch: int = 4096
) -> list:
    X = np.asarray(Z, np.float32).reshape((np.asarray(Z).shape[0], -1))
    out = [np.zeros((X.shape[0], store.n_atoms), np.float64) for _ in store.atoms]
    for i in range(0, X.shape[0], int(batch)):
        chunk = X[i : i + int(batch)]
        for k, (w, a) in enumerate(zip(store.well_channels, store.atoms, strict=True)):
            z = chunk[:, np.asarray(w, np.int64)].astype(np.float64)
            A = np.asarray(a, np.float64)
            d2 = (z * z).sum(axis=1)[:, None] - 2.0 * (z @ A.T) + (A * A).sum(axis=1)[None, :]
            s = -np.maximum(d2, 0.0) / (2.0 * float(sigma[k]) ** 2)
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            out[k][i : i + chunk.shape[0]] = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    return out


def fit_atom_address_book(
    store: AtomStore,
    Z_train: np.ndarray,
    labels: np.ndarray,
    cardinalities: Sequence[int],
    sigma: np.ndarray,
    smoothing: float = 1.0,
) -> AtomAddressBook:
    pis = soft_atom_assignment(store, Z_train, sigma)
    Y = np.asarray(labels, np.int64)
    K, N = (store.n_wells, store.n_atoms)
    a = float(smoothing)
    prior = np.zeros((K, N), np.float64)
    for k in range(K):
        m = pis[k].sum(axis=0)
        prior[k] = (m + a) / (m.sum() + a * N)
    conds = []
    for f, card in enumerate(cardinalities):
        V = int(card)
        cond = np.zeros((V, K, N), np.float64)
        for v in range(V):
            sel = Y[:, f] == v
            for k in range(K):
                m = pis[k][sel].sum(axis=0) if bool(sel.any()) else np.zeros(N)
                cond[v, k] = (m + a) / (m.sum() + a * N)
        conds.append(np.log(cond))
    return AtomAddressBook(np.log(prior), conds)


def atom_posterior(book: AtomAddressBook, requests: np.ndarray) -> np.ndarray:
    R = np.asarray(requests, np.int64)
    lp = np.asarray(book.log_prior)
    out = np.broadcast_to(lp, (R.shape[0],) + lp.shape).copy()
    for f, lc in enumerate(book.log_cond):
        lc = np.asarray(lc)
        asked = R[:, f] >= 0
        if not asked.any():
            continue
        out[asked] += lc[R[asked, f]] - lp[None, :, :]
    return out.astype(np.float32)


def _softmax_last(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, np.float64)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    return P / np.maximum(P.sum(axis=-1, keepdims=True), 1e-300)


class AtomLandscape(eqx.Module):
    store: AtomStore
    coupling: Optional[LowRankCoupling]
    sigma: jnp.ndarray
    request_strength: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)

    def __init__(
        self,
        store,
        sigma,
        coupling=None,
        request_strength: float = 1.0,
        coupling_strength: float = 1.0,
    ):
        self.store = store
        self.coupling = coupling
        self.sigma = jnp.asarray(np.asarray(sigma, np.float32), jnp.float32)
        if int(self.sigma.shape[0]) != int(store.n_wells):
            raise ValueError(
                f"sigma must have one bandwidth per well ({store.n_wells}), got {self.sigma.shape}"
            )
        self.request_strength = float(request_strength)
        self.coupling_strength = float(coupling_strength)

    @property
    def n_wells(self) -> int:
        return int(self.store.n_wells)

    @property
    def n_atoms(self) -> int:
        return int(self.store.n_atoms)

    def with_strengths(self, request_strength=None, coupling_strength=None) -> "AtomLandscape":
        return AtomLandscape(
            self.store,
            self.sigma,
            self.coupling,
            self.request_strength if request_strength is None else float(request_strength),
            self.coupling_strength if coupling_strength is None else float(coupling_strength),
        )


def _well_slices(store: AtomStore, X):
    idx = [jnp.asarray(w, jnp.int32) for w in store.well_channels]
    Z = X.reshape((X.shape[0], -1))
    return [Z[:, i] for i in idx]


def _well_logk(store: AtomStore, X, sigma_mult):
    out = []
    for z, a, s in zip(_well_slices(store, X), store.atoms, sigma_mult, strict=True):
        d2 = jnp.sum(z * z, axis=-1)[:, None] - 2.0 * (z @ a.T) + jnp.sum(a * a, axis=-1)[None, :]
        out.append(-jnp.maximum(d2, 0.0) / (2.0 * s**2))
    return out


def store_energy(land: AtomLandscape, X, sigma_scale=1.0):
    """``E_store`` — the kernel free energy over the well's atoms."""
    sig = land.sigma * jnp.asarray(sigma_scale, land.sigma.dtype)
    e = jnp.zeros((X.shape[0],), X.dtype)
    for lk, s in zip(_well_logk(land.store, X, sig), sig, strict=True):
        e = e - s**2 * jax.nn.logsumexp(lk, axis=-1)
    return e


def request_energy(land: AtomLandscape, X, log_w, sigma_scale=1.0):
    sig = land.sigma * jnp.asarray(sigma_scale, land.sigma.dtype)
    lks = _well_logk(land.store, X, sig)
    base = jnp.zeros((X.shape[0],), X.dtype)
    for lk, s in zip(lks, sig, strict=True):
        base = base - s**2 * jax.nn.logsumexp(lk, axis=-1)
    total = jnp.zeros((X.shape[0],), X.dtype)
    n_terms = int(log_w.shape[0])
    for t in range(n_terms):
        e = jnp.zeros((X.shape[0],), X.dtype)
        for k, (lk, s) in enumerate(zip(lks, sig, strict=True)):
            e = e - s**2 * jax.nn.logsumexp(lk + log_w[t, :, k, :], axis=-1)
        total = total + (e - base)
    return total


def coupling_energy(land: AtomLandscape, X):
    """``−Σ_{k<l} x̃_kᵀ U_klᵀ V_kl x̃_l`` at rank ``r``.  ``0`` when no coupling."""
    c = land.coupling
    if c is None:
        return jnp.zeros((X.shape[0],), X.dtype)
    zs = _well_slices(land.store, X)
    std = [(z - c.centres[k][None, :]) / c.radii[k] for k, z in enumerate(zs)]
    e = jnp.zeros((X.shape[0],), X.dtype)
    for (k, lo), U, V in zip(c.pairs, c.u, c.v, strict=True):
        e = e - jnp.sum(std[k] @ U.T * (std[lo] @ V.T), axis=-1)
    return e


def total_energy(land: AtomLandscape, X, log_w, sigma_scale=1.0):
    e = store_energy(land, X, sigma_scale)
    if land.request_strength != 0.0 and log_w is not None and (int(log_w.shape[0]) > 0):
        e = e + land.request_strength * request_energy(land, X, log_w, sigma_scale)
    if land.coupling_strength != 0.0 and land.coupling is not None:
        e = e + land.coupling_strength * coupling_energy(land, X)
    return e


def request_experts(book: AtomAddressBook, requests: np.ndarray) -> np.ndarray:
    R = np.asarray(requests, np.int64)
    n, F = R.shape
    K, N = (book.n_wells, book.n_atoms)
    lp = np.asarray(book.log_prior)
    out = np.zeros((F, n, K, N), np.float32)
    for f, lc in enumerate(book.log_cond):
        lc = np.asarray(lc)
        asked = R[:, f] >= 0
        if not asked.any():
            continue
        out[f, asked] = (lc[R[asked, f]] - lp[None, :, :]).astype(np.float32)
    return out


def launch_state(
    land: AtomLandscape,
    book: AtomAddressBook,
    requests: np.ndarray,
    launch_mode: str = "clamp_atom",
    clamp_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    if launch_mode not in LAUNCH_MODES:
        raise ValueError(f"launch_mode must be one of {LAUNCH_MODES}, got {launch_mode!r}")
    store = land.store
    P = _softmax_last(atom_posterior(book, requests))
    prior = _softmax_last(np.asarray(book.log_prior)[None])[0]
    n = int(P.shape[0])
    conf = P.max(axis=-1)
    idx = P.argmax(axis=-1)
    clamped = conf >= float(clamp_threshold)
    x0 = np.zeros((n, store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)
        if launch_mode == "posterior_mean":
            x0[:, cols] = (P[:, k, :] @ A).astype(np.float32)
        elif launch_mode == "prior_atom":
            x0[:, cols] = A[int(np.argmax(prior[k]))][None, :]
        elif launch_mode == "clamp_atom_thresholded":
            take = np.where(clamped[:, k], idx[:, k], int(np.argmax(prior[k])))
            x0[:, cols] = A[take]
        else:
            x0[:, cols] = A[idx[:, k]]
    return {
        "x0": x0,
        "atom_index": idx.astype(np.int64),
        "confidence": conf.astype(np.float32),
        "clamped": clamped,
        "posterior": P.astype(np.float32),
        "launch_mode": str(launch_mode),
        "clamp_threshold": float(clamp_threshold),
        "mean_n_clamped": float(clamped.sum(axis=1).mean()) if clamped.size else 0.0,
        "mean_n_relaxed": float((~clamped).sum(axis=1).mean()) if clamped.size else 0.0,
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
        "n_wells": int(store.n_wells),
    }


class SettleCfg:
    __slots__ = (
        "steps",
        "step_size",
        "sigma_scale_start",
        "sigma_scale_end",
        "temp_start",
        "temp_end",
        "metropolis",
        "trace_every",
    )

    def __init__(
        self,
        steps=1000,
        step_size=0.2,
        sigma_scale_start=4.0,
        sigma_scale_end=1.0,
        temp_start=0.0,
        temp_end=0.0,
        metropolis=True,
        trace_every=1,
    ):
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.sigma_scale_start = float(sigma_scale_start)
        self.sigma_scale_end = float(sigma_scale_end)
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.metropolis = bool(metropolis)
        self.trace_every = int(trace_every)

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__slots__}


def _geom(a: float, b: float, i: int, n: int) -> float:
    a, b = (float(a), float(b))
    if n <= 1 or a <= 0.0 or b <= 0.0:
        return b
    return float(a * (b / a) ** (i / (n - 1)))


@eqx.filter_jit
def _settle_step(land, x, log_w, key, eps, temp, sigma_scale, sigma_ref, metropolis):
    ef = lambda z: jnp.sum(total_energy(land, z, log_w, sigma_scale))
    g = jax.grad(ef)(x)
    mean_f = x - eps * g
    key, nk, uk = jax.random.split(key, 3)
    noise = jax.random.normal(nk, x.shape, x.dtype) * jnp.sqrt(2.0 * eps * temp)
    xp = mean_f + noise
    gp = jax.grad(ef)(xp)
    mean_r = xp - eps * gp
    denom = jnp.maximum(4.0 * eps * temp, jnp.asarray(1e-30, x.dtype))
    logq_f = -jnp.sum((xp - mean_f) ** 2, axis=-1) / denom
    logq_r = -jnp.sum((x - mean_r) ** 2, axis=-1) / denom
    de = total_energy(land, xp, log_w, sigma_scale) - total_energy(land, x, log_w, sigma_scale)
    log_alpha_mh = jnp.nan_to_num(
        -de / jnp.maximum(temp, 1e-30) + logq_r - logq_f, nan=-1e30, posinf=0.0, neginf=-1e30
    )
    log_alpha = jnp.where(temp > 0.0, log_alpha_mh, jnp.where(de < 0.0, 0.0, -1e30))
    u = jax.random.uniform(uk, (x.shape[0],), x.dtype)
    accepted = jnp.log(jnp.maximum(u, jnp.asarray(1e-30, x.dtype))) < log_alpha
    take = accepted if metropolis else jnp.ones_like(accepted)
    x_new = jnp.where(take[:, None], xp, x)
    disp = jnp.sqrt(jnp.mean((x_new - x) ** 2, axis=-1))
    return (
        x_new,
        jnp.exp(jnp.minimum(log_alpha, 0.0)),
        accepted,
        disp,
        key,
        total_energy(land, x_new, log_w, sigma_scale),
        total_energy(land, x_new, log_w, sigma_ref),
    )


def settle_chain(land: AtomLandscape, x0: np.ndarray, log_w, cfg: SettleCfg, key) -> dict:
    x = jnp.asarray(np.asarray(x0, np.float32))
    n_steps = int(cfg.steps)
    lw = None if log_w is None else jnp.asarray(np.asarray(log_w, np.float32))
    sig_ref = jnp.asarray(float(cfg.sigma_scale_end), jnp.float32)
    eps = jnp.asarray(float(cfg.step_size), jnp.float32)
    e_hist, er_hist, a_hist, acc_hist, disp_hist = ([], [], [], [], [])
    sig_hist, temp_hist = ([], [])
    s0 = jnp.asarray(
        _geom(cfg.sigma_scale_start, cfg.sigma_scale_end, 0, max(n_steps, 1)), jnp.float32
    )
    e_hist.append(np.asarray(total_energy(land, x, lw, s0), np.float64))
    er_hist.append(np.asarray(total_energy(land, x, lw, sig_ref), np.float64))
    for i in range(n_steps):
        sig_now = jnp.asarray(
            _geom(cfg.sigma_scale_start, cfg.sigma_scale_end, i, n_steps), jnp.float32
        )
        temp_now = jnp.asarray(
            _geom(cfg.temp_start, cfg.temp_end, i, n_steps)
            if cfg.temp_start > 0
            else float(cfg.temp_end),
            jnp.float32,
        )
        key, sk = jax.random.split(key)
        x, aprob, acc, disp, _k, e_new, er_new = _settle_step(
            land, x, lw, sk, eps, temp_now, sig_now, sig_ref, bool(cfg.metropolis)
        )
        a_hist.append(np.asarray(aprob, np.float64))
        acc_hist.append(np.asarray(acc, bool))
        disp_hist.append(np.asarray(disp, np.float64))
        e_hist.append(np.asarray(e_new, np.float64))
        er_hist.append(np.asarray(er_new, np.float64))
        sig_hist.append(float(sig_now))
        temp_hist.append(float(temp_now))
    n = int(x.shape[0])
    trace = {
        "energy": np.stack(e_hist) if e_hist else np.zeros((0, n)),
        "energy_ref": np.stack(er_hist) if er_hist else np.zeros((0, n)),
        "accept_prob": np.stack(a_hist) if a_hist else np.zeros((0, n)),
        "accepted": np.stack(acc_hist) if acc_hist else np.zeros((0, n), bool),
        "displacement": np.stack(disp_hist) if disp_hist else np.zeros((0, n)),
        "sigma_scale": np.asarray(sig_hist, np.float64),
        "temperature": np.asarray(temp_hist, np.float64),
    }
    return {"x": np.asarray(x, np.float32), "trace": trace}


def convergence_report(trace: dict, window: int = 100, tol: float = 0.001) -> dict:
    E = np.asarray(trace["energy_ref"], np.float64)
    disp = np.asarray(trace["displacement"], np.float64)
    acc = np.asarray(trace["accept_prob"], np.float64)
    took = np.asarray(trace["accepted"])
    if E.shape[0] < 2:
        e0 = float(E[0].mean()) if E.shape[0] else float("nan")
        nan = float("nan")
        return {
            "n_steps": 0,
            "window": 0,
            "tol": float(tol),
            "rule": "zero-step control: no step was taken, so nothing can descend",
            "zero_step": True,
            "frac_still_descending": nan,
            "n_still_descending": 0,
            "n_chains": int(E.shape[1]) if E.ndim > 1 and E.shape[0] else 0,
            "relative_tail_drop": {"mean": nan, "min": nan, "max": nan},
            "energy_first": e0,
            "energy_final": e0,
            "energy_absolute_change": 0.0,
            "energy_relative_drop": 0.0,
            "energy_is_flat": True,
            "mean_accept_prob": nan,
            "accepted_fraction": nan,
            "displacement_step_1": nan,
            "displacement_final_step": nan,
            "displacement_mean": nan,
            "total_displacement": 0.0,
            "verdict": "ZERO-STEP CONTROL (the launch, decoded; no chain was run)",
            "note": "⭐ dreaming's laundering control: it took ESSENTIALLY ALL of that study's apparent gain, so it is never optional.",
        }
    w = int(min(max(window, 2), E.shape[0]))
    tail = E[-w:]
    t = np.arange(w, dtype=np.float64)
    t = t - t.mean()
    slope = (t[:, None] * (tail - tail.mean(axis=0, keepdims=True))).sum(axis=0) / (t * t).sum()
    scale = np.maximum(np.abs(tail).mean(axis=0), 1e-12)
    rel = slope * w / scale
    still = rel < -float(tol)
    frac = float(np.mean(still))
    return {
        "n_steps": int(E.shape[0] - 1),
        "window": w,
        "tol": float(tol),
        "rule": "still descending iff OLS slope of energy_ref over the last `window` steps x window < -tol * mean|E_ref| over that window",
        "frac_still_descending": frac,
        "n_still_descending": int(still.sum()),
        "n_chains": int(E.shape[1]),
        "relative_tail_drop": {
            "mean": float(rel.mean()),
            "min": float(rel.min()),
            "max": float(rel.max()),
        },
        "energy_first": float(E[0].mean()),
        "energy_final": float(E[-1].mean()),
        "energy_absolute_change": float(E[-1].mean() - E[0].mean()),
        "energy_relative_drop": float(
            (E[0].mean() - E[-1].mean()) / max(abs(E[0].mean()), abs(E[-1].mean()), 1e-12)
        ),
        "energy_is_flat": bool(
            float(np.mean(np.abs(E[0] - E[-1]))) / max(float(np.abs(E).mean()), 1e-12) < 1e-06
        ),
        "mean_accept_prob": float(acc.mean()) if acc.size else float("nan"),
        "accepted_fraction": float(took.mean()) if took.size else float("nan"),
        "displacement_step_1": float(disp[0].mean()) if disp.size else float("nan"),
        "displacement_final_step": float(disp[-1].mean()) if disp.size else float("nan"),
        "displacement_mean": float(disp.mean()) if disp.size else float("nan"),
        "total_displacement": float(disp.sum(axis=0).mean()) if disp.size else float("nan"),
        "verdict": "⛔ NOT CONVERGED — chains are still descending at the final step"
        if frac > 0.0
        else "settled (no chain descending in the final window)",
        "note": "⭐ Per-step displacement is reported FROM STEP 1. ⚠ dreaming's acceptance was 0.952 on a chain that barely moved: acceptance alone cannot distinguish settling from sitting still.",
    }


def read_settled(
    land: AtomLandscape, x, read_mode: str = "state", read_sigma_scale: float = 1.0
) -> np.ndarray:
    if read_mode not in READ_MODES:
        raise ValueError(f"read_mode must be one of {READ_MODES}, got {read_mode!r}")
    store = land.store
    X = jnp.asarray(np.asarray(x, np.float32)).reshape((np.asarray(x).shape[0], -1))
    if read_mode == "state":
        return np.asarray(X, np.float32).reshape((X.shape[0],) + store.feat_shape)
    sig = land.sigma * float(read_sigma_scale)
    lks = _well_logk(store, X, sig)
    out = jnp.zeros((X.shape[0], store.dim), jnp.float32)
    for w, a, lk in zip(store.well_channels, store.atoms, lks, strict=True):
        out = out.at[:, jnp.asarray(w, jnp.int32)].set(jax.nn.softmax(lk, axis=-1) @ a)
    return np.asarray(out, np.float32).reshape((X.shape[0],) + store.feat_shape)


def pairwise_separation(x, max_n: int = 512, seed: int = 0, collapse_frac: float = 0.01) -> dict:
    X = np.asarray(x, np.float64).reshape((np.asarray(x).shape[0], -1))
    n0 = int(X.shape[0])
    if n0 < 2:
        return {"n": n0, "note": "fewer than 2 states — no separation"}
    if n0 > int(max_n):
        sel = np.sort(np.random.default_rng(int(seed)).permutation(n0)[: int(max_n)])
        X = X[sel]
    n, d = X.shape
    g = (X * X).sum(1)
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (X @ X.T), 0.0)
    iu = np.triu_indices(n, 1)
    dist = np.sqrt(d2[iu] / max(d, 1))
    mean = float(dist.mean())
    return {
        "n": n,
        "n_total": n0,
        "mean_pairwise_rms_distance": mean,
        "min_pairwise_rms_distance": float(dist.min()),
        "median_pairwise_rms_distance": float(np.median(dist)),
        "frac_pairs_collapsed": float((dist <= float(collapse_frac) * max(mean, 1e-30)).mean()),
        "collapse_frac": float(collapse_frac),
        "all_states_identical": bool(float(dist.max()) == 0.0),
        "note": "⛔ NOT falsifier 2 (that is distinct FACTOR TUPLES read by the EXTERNAL CLASSIFIER off the DECODED IMAGES, PROTOCOL §5.2). ⚠ Also ⛔ not the participation ratio: PR counts DIRECTIONS, not distinct states, and reading PR as distinctness misled this study once.",
    }


def settled_state_spread(x, store: Optional[AtomStore] = None) -> dict:
    X = np.asarray(x, np.float64).reshape((np.asarray(x).shape[0], -1))
    if X.shape[0] < 2:
        return {"n": int(X.shape[0]), "note": "fewer than 2 states — no spread"}
    Xc = X - X.mean(axis=0, keepdims=True)
    var = (Xc * Xc).sum(axis=0) / max(X.shape[0] - 1, 1)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s**2 / max(X.shape[0] - 1, 1)
    pr = float(lam.sum() ** 2 / max(float((lam**2).sum()), 1e-300))
    out = {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "total_variance": float(var.sum()),
        "mean_coordinate_std": float(np.sqrt(np.maximum(var, 0)).mean()),
        "participation_ratio": pr,
        "n_distinct_states": int(np.unique(np.asarray(X, np.float32), axis=0).shape[0]),
        "pairwise": pairwise_separation(X),
        "note": "⛔ n_distinct_states is NOT falsifier 2: continuous states are trivially distinct and that number is worthless. Falsifier 2 is the distinct FACTOR TUPLES the EXTERNAL CLASSIFIER reads off the DECODED IMAGES (PROTOCOL §5.2).",
    }
    if store is not None:
        A = np.concatenate([np.asarray(a, np.float64) for a in store.atoms], axis=1)
        al = np.linalg.svd(A - A.mean(axis=0, keepdims=True), compute_uv=False) ** 2
        al = al / max(A.shape[0] - 1, 1)
        out["atom_participation_ratio"] = float(al.sum() ** 2 / max(float((al**2).sum()), 1e-300))
    return out


def dynamical_read(
    land: AtomLandscape,
    book: AtomAddressBook,
    requests: np.ndarray,
    cfg: SettleCfg,
    key,
    launch_mode: str = "clamp_atom",
    clamp_threshold: float = 0.5,
    read_mode: str = "state",
    *,
    bypass_stage: str = "none",
    bypass_features: Optional[np.ndarray] = None,
    bypass_alpha: float = 0.0,
) -> dict:
    if bypass_stage not in BYPASS_STAGES:
        raise ValueError(f"bypass_stage must be one of {BYPASS_STAGES}, got {bypass_stage!r}")
    R = np.asarray(requests, np.int64)
    lw = request_experts(book, R)
    lau = launch_state(land, book, R, launch_mode, clamp_threshold)
    x0 = lau["x0"]

    def _inject(v):
        B = np.asarray(bypass_features, np.float32).reshape(
            (np.asarray(bypass_features).shape[0], -1)
        )
        if B.shape[0] < v.shape[0]:
            B = np.repeat(B, int(np.ceil(v.shape[0] / B.shape[0])), axis=0)
        return (v + float(bypass_alpha) * B[: v.shape[0], : v.shape[1]]).astype(np.float32)

    if bypass_stage == "launch" and bypass_features is not None:
        x0 = _inject(x0)
    out = settle_chain(land, x0, lw, cfg, key)
    xs = out["x"]
    if bypass_stage == "settle" and bypass_features is not None:
        xs = _inject(xs)
    r = read_settled(land, xs, read_mode)
    if bypass_stage == "read" and bypass_features is not None:
        flat = _inject(np.asarray(r, np.float32).reshape((r.shape[0], -1)))
        r = flat.reshape(r.shape)
    return {
        "launch": lau,
        "x0": np.asarray(x0, np.float32),
        "x_settled": np.asarray(xs, np.float32),
        "read": np.asarray(r, np.float32),
        "trace": out["trace"],
        "bypass_stage": str(bypass_stage),
        "bypass_alpha": float(bypass_alpha),
    }


def save_trace(trace: dict, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        energy=np.asarray(trace["energy"], np.float32),
        energy_ref=np.asarray(trace["energy_ref"], np.float32),
        accept_prob=np.asarray(trace["accept_prob"], np.float32),
        accepted=np.asarray(trace["accepted"], np.bool_),
        displacement=np.asarray(trace["displacement"], np.float32),
        sigma_scale=np.asarray(trace["sigma_scale"], np.float32),
        temperature=np.asarray(trace["temperature"], np.float32),
    )
    return str(path)
