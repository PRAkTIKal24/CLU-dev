"""composition cell."""

from __future__ import annotations
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable
import numpy as np
from nclu.experiments import factor_wells as fw
from nclu.experiments import exp_gen_field_store as fs
from nclu.experiments import exp_gen_read as R
from nclu.core.gen_read import render_span, span_basis, span_coefficients

CELL_PINS = {
    "field_pins_from": "exp_gen_field_store.PINS (selected on val by an earlier study)",
    "atom_seed_stride": int(fs.PINS["atom_seed_stride"]),
    "field_atom_seed": int(fs.PINS["field_atom_seed"]),
    "store_bytes": int(fw.PINS["store_bytes"]),
}
FIELD_ARMS = (
    "oracle_span_field",
    "field_lookup_argmax",
    "oracle_content_pool",
    "oracle_content_oracle",
    "oracle_content_fine",
    "oracle_true_feature",
    "oracle_content_items",
    "pca_tensor",
    "pca_tensor_sum",
    "pca_landscape_removed",
    "pca_landscape_shuffled",
    "pca_random_refit",
    "pca_span_rotated",
    "pca_lookup_argmax",
    "pca_lookup_convex",
)
PCA_ARMS = {
    "pca_tensor": (None, "tensor"),
    "pca_tensor_sum": (None, "tensor_sum"),
    "pca_landscape_removed": ("zero", "tensor"),
    "pca_landscape_shuffled": ("value_permute", "tensor"),
    "pca_random_refit": ("random_span", "tensor"),
    "pca_span_rotated": ("rotate", "tensor"),
    "pca_lookup_argmax": (None, "tensor_argmax"),
    "pca_lookup_convex": (None, "tensor_convex"),
}
PCA_PRE_FIT = ("random_span", "rotate")
FIELD_ARM_CLASS = {
    "oracle_span_field": "NOT AN ARM (ceiling, FIELD store)",
    "field_lookup_argmax": "FLOOR (protocol C.1/C.4, matched-bytes lookup, FIELD atoms)",
    "oracle_content_pool": "NOT AN ARM (ceiling, BEST REALISABLE content @524,288 B)",
    "oracle_content_oracle": "NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, K=5)",
    "oracle_content_fine": "NOT AN ARM (ceiling, ABSOLUTE content @524,288 B, fine K)",
    "oracle_true_feature": "NOT AN ARM (ceiling of EVERY store, any budget)",
    "oracle_content_items": "NOT AN ARM (ceiling, 64 REAL ITEMS @524,288 B)",
    "pca_tensor": "ARM",
    "pca_tensor_sum": "ABLATION (same rank, address summed not multiplied)",
    "pca_landscape_removed": "CONTROL (protocol C.2, atoms zeroed)",
    "pca_landscape_shuffled": "CONTROL (protocol C.2, atom values permuted)",
    "pca_random_refit": "CONTROL (random atoms, address book REFIT)",
    "pca_span_rotated": "CONTROL (predicted NO-OP: the span is unchanged)",
    "pca_lookup_argmax": "FLOOR (protocol C.1/C.4, matched-bytes lookup)",
    "pca_lookup_convex": "FLOOR (protocol C.1, the mixture)",
}
CONTENT_SOURCE = {
    "oracle_content_pool": "pool",
    "oracle_content_oracle": "corners",
    "oracle_content_fine": "corners",
}
CONTENT_PARTITION = {"oracle_content_fine": "fine"}


class FieldSpanStore:
    def __init__(self, field):
        self.field = field
        alive = np.asarray(field.view("atom_alive"), np.uint8) > 0
        self.n_alive = int(alive.sum())
        self.well_channels = [np.asarray(c, np.int64) for c in field.meta["well_channels"]]
        self.modes = [
            np.ascontiguousarray(np.asarray(field.atoms_of_well(k), np.float32)[alive])
            for k in range(field.K)
        ]
        self.dim = int(field.D)

    @property
    def K(self) -> int:
        return len(self.modes)

    def span_ranks(self, tol: float = 1e-06) -> list:
        return [int(U.shape[1]) for U in span_basis(self, tol)]


def _assert_span_is_the_content(store: "FieldSpanStore", field) -> dict:
    """The ceiling must be a property of the DEPLOYED bytes and of nothing else.

    Two assertions, both cheap and both load-bearing:

    1. every exposed mode row is a byte-for-byte view of the ``atoms`` segment -- so the
       subspace being projected onto is the one in the store image, not a re-fit of it;
    2. the total number of exposed mode rows equals the number of live atoms -- so no
       dead atom widens the span.
    """
    A = np.asarray(field.view("atoms"), np.float32)
    alive = np.asarray(field.view("atom_alive"), np.uint8) > 0
    n_rows = int(sum((M.shape[0] for M in store.modes)))
    if n_rows != store.K * int(alive.sum()):
        raise AssertionError(
            f"the span exposes {n_rows} mode rows but the store deploys {store.K * int(alive.sum())} live atom rows"
        )
    for k in range(store.K):
        ref = np.asarray(field.atoms_of_well(k), np.float32)[alive]
        if not np.array_equal(np.asarray(store.modes[k], np.float32), ref):
            raise AssertionError(f"well {k}'s modes are not the store's own atom bytes")
    return {
        "n_live_atoms": int(alive.sum()),
        "n_wells": int(store.K),
        "atoms_segment_bytes": int(A.size * 4),
        "span_ranks": store.span_ranks(),
        "what": "the ceiling is the orthogonal projection onto the row space of the DEPLOYED atom bytes; no refit, no dead atom",
    }


class PCASpanStore:
    def __init__(self, X: np.ndarray, well_channels, n_atoms: int, dim: int):
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(n_atoms)
        self.modes, self.energy = ([], [])
        Xf = np.asarray(X, np.float32)
        for ch in self.well_channels:
            M = np.asarray(Xf[:, ch], np.float64)
            mu = M.mean(axis=0)
            Mc = M - mu
            _, s, Vt = np.linalg.svd(Mc, full_matrices=False)
            r = min(self.n_atoms - 1, Vt.shape[0])
            B = np.vstack([mu[None, :], Vt[:r]]).astype(np.float32)
            if B.shape[0] < self.n_atoms:
                B = np.vstack([B, np.zeros((self.n_atoms - B.shape[0], B.shape[1]), np.float32)])
            self.modes.append(np.ascontiguousarray(B.astype(np.float32)))
            tot = float((s**2).sum())
            self.energy.append(float((s[:r] ** 2).sum() / tot) if tot > 0 else 1.0)

    @property
    def K(self) -> int:
        return len(self.modes)

    def byte_footprint(self) -> dict:
        b = int(sum((M.size for M in self.modes)) * 4)
        return {
            "deployed_bytes": b,
            "budget_bytes": int(CELL_PINS["store_bytes"]),
            "matches_budget": bool(b == int(CELL_PINS["store_bytes"])),
            "param_count": b // 4,
            "what": f"{self.n_atoms} atoms x {self.dim} float32 -- the per-well leading principal subspace, the SAME block a field or round-3 store's atoms occupy",
            "variance_explained_per_well": [round(e, 6) for e in self.energy],
        }


class _ModeStore:
    def __init__(self, modes, well_channels, dim: int, what: str = ""):
        self.modes = [np.ascontiguousarray(np.asarray(M, np.float32)) for M in modes]
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.what = what

    @property
    def K(self) -> int:
        return len(self.modes)

    def byte_footprint(self) -> dict:
        b = int(sum((M.size for M in self.modes)) * 4)
        return {
            "deployed_bytes": b,
            "budget_bytes": int(CELL_PINS["store_bytes"]),
            "matches_budget": bool(b == int(CELL_PINS["store_bytes"])),
            "param_count": b // 4,
            "what": self.what,
        }


def basis_at_fixed_rank(store, ranks) -> list:
    out = []
    for U, r in zip(span_basis(store), [int(r) for r in ranks]):
        U = np.asarray(U, np.float32)
        if U.shape[1] > r:
            U = U[:, :r]
        elif U.shape[1] < r:
            U = np.hstack([U, np.zeros((U.shape[0], r - U.shape[1]), np.float32)])
        out.append(np.ascontiguousarray(U))
    return out


def perturb_modes(store, kind: str, seed: int, what: str = ""):
    rng = np.random.default_rng(int(seed) * 7919 + 20260919)
    out = []
    for G in store.modes:
        Gm = np.asarray(G, np.float32)
        if kind == "zero":
            out.append(np.zeros_like(Gm))
        elif kind == "value_permute":
            out.append(Gm.reshape(-1)[rng.permutation(Gm.size)].reshape(Gm.shape))
        elif kind == "random_span":
            Rn = rng.normal(size=Gm.shape).astype(np.float32)
            Rn *= float(np.linalg.norm(Gm)) / (float(np.linalg.norm(Rn)) + 1e-12)
            out.append(Rn)
        elif kind == "rotate":
            Q, _ = np.linalg.qr(rng.normal(size=(Gm.shape[0], Gm.shape[0])))
            out.append(Q.astype(np.float32) @ Gm)
        else:
            raise ValueError(kind)
    return _ModeStore(
        out, store.well_channels, store.dim, what or f"{kind} control over the arm's own atom block"
    )


class _ItemSpanStore:
    """Content that is WRITTEN ITEMS, at the span interface.  ``A`` rows, ``A x D``
    float32, the same block and the same semantics the field store deploys."""

    def __init__(self, X: np.ndarray, well_channels, dim: int):
        Xf = np.asarray(X, np.float32)
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.n_atoms = int(Xf.shape[0])
        self.modes = [np.ascontiguousarray(Xf[:, c]) for c in self.well_channels]

    @property
    def K(self) -> int:
        return len(self.modes)

    def byte_footprint(self) -> dict:
        b = int(sum((M.size for M in self.modes)) * 4)
        return {
            "deployed_bytes": b,
            "budget_bytes": int(CELL_PINS["store_bytes"]),
            "matches_budget": bool(b == int(CELL_PINS["store_bytes"])),
            "param_count": b // 4,
            "what": f"{self.n_atoms} written items x {self.dim} float32 -- one item in every well, the field store's own layout",
        }


def _content_source_rows(out_dir: Path, source: str, log: Callable) -> np.ndarray:
    cond = fw.load_condition()
    if source == "pool":
        return np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    if source != "corners":
        raise ValueError(source)
    if fw.PINS["condition"] == "val":
        cs = fw.val_corners(cond)
    else:
        cs = fw.nway_corners(cond) + fw.pair_subcorners(cond)
    rows = np.unique(np.concatenate([c["rows"] for c in cs]))
    log(
        f"  optimal content source = the {len(cs)} SCORED corners' own {rows.size} rows (UNREACHABLE bound)"
    )
    return rows


def build_field_for_leg(out_dir: Path, log: Callable = print) -> dict:
    from nclu.core.gen_field import assert_read_request_only, build_field, field_byte_footprint

    out_dir = Path(out_dir)
    d = out_dir / "field"
    done = d / "field.pkl"
    if done.exists():
        log("  field store: already built for this leg, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)
    seed = int(fw.PINS["seed"])
    cond = fw.load_condition()
    grid = cond["grid"]
    parts = fw.build_partitions(out_dir, log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = fs._host_factors(parts["info"], K)
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    F = fw.features_mmap(seed)
    t0 = time.time()
    X = fw.gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    log(f"  field store: pool {pool.size} rows, hosts {hosts}, seed {seed}")
    spec = fs.field_spec(seed)
    field = build_field(X, Y, pool, labels, hosts, cards, spec, log=log)
    del X
    led = field_byte_footprint(field, int(CELL_PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the field store deploys {led['deployed_bytes']} B against the pinned {CELL_PINS['store_bytes']} B footprint -- the byte pin has moved"
        )
    vals = grid.values_of(pool[:64])
    ro = assert_read_request_only(field, vals, grid.values)
    obj = {
        "field": field,
        "seed": seed,
        "condition": str(fw.PINS["condition"]),
        "byte_footprint": led,
        "request_only": ro,
        "manifest": field.manifest(),
        "pool_n_rows": int(pool.size),
        "fit_seconds": time.time() - t0,
        "spec": spec.to_dict() if hasattr(spec, "to_dict") else str(spec),
    }
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items() if k != "field"}, d / "field.json")
    log(
        f"  field store: built in {obj['fit_seconds']:.1f}s; atoms {led['deployed_bytes']:,} B (image {led['total_image_bytes']:,} B) sha {field.sha256()[:16]}"
    )
    return obj


def build_pca_content(out_dir: Path, log: Callable = print) -> dict:
    out_dir = Path(out_dir)
    cache = out_dir / "pca_content.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    fw.build_partitions(out_dir, log=log)
    D = int(fw.PINS["D"])
    n_atoms = int(CELL_PINS["store_bytes"]) // (4 * D)
    labels = np.asarray(dict(np.load(out_dir / "partitions.npz"))["factor"], np.int64)
    K = int(labels.max()) + 1
    wc = [np.flatnonzero(labels == k).astype(np.int64) for k in range(K)]
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    X = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pool)
    store = PCASpanStore(X, wc, n_atoms, D)
    del X
    led = store.byte_footprint()
    if not led["matches_budget"]:
        raise AssertionError(
            f"the principal content deploys {led['deployed_bytes']} B against the pinned {CELL_PINS['store_bytes']} B footprint -- the byte pin has moved"
        )
    obj = {
        "store": store,
        "byte_footprint": led,
        "well_sizes": [int(c.size) for c in wc],
        "n_atoms": n_atoms,
        "n_wells": K,
        "n_pool_rows": int(pool.size),
        "partition": "factor (K=5, inherited label-MI wells)",
        "label_use": "labels align the wells ONCE at fit time and ground the address book; the CONTENT is unsupervised (protocol B.6)",
    }
    with open(cache, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items() if k != "store"}, out_dir / "pca_content.json")
    log(
        f"  principal content: {n_atoms} atoms x {D} float32 = {led['deployed_bytes']:,} B, wells {obj['well_sizes']}, variance explained {[round(e, 4) for e in store.energy]}"
    )
    return obj


def _build_pca_arm(
    arm: str, out_dir: Path, arm_dir: Path, done: Path, t0: float, log: Callable
) -> dict:
    from nclu.core.gen_read import TensorAddress, assert_read_request_only

    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    base = build_pca_content(out_dir, log=log)
    store = base["store"]
    kind, verb = PCA_ARMS[arm]
    if kind in PCA_PRE_FIT:
        store = perturb_modes(store, kind, int(fw.PINS["seed"]))
    basis = span_basis(store)
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    pool = pool[: int(R.READ_PINS["n_addr_train"])]
    tag = kind if kind in PCA_PRE_FIT else ""
    cache = out_dir / (
        f"address_pca_{('sum' if verb == 'tensor_sum' else 'prod')}_R{int(R.READ_PINS['rank'])}_lam{R.READ_PINS['ridge']}_sw{int(R.READ_PINS['sweeps'])}_n{int(R.READ_PINS['n_addr_train'])}"
        + (f"_{tag}" if tag else "")
        + ".pkl"
    )
    if cache.exists():
        with open(cache, "rb") as fh:
            addr = pickle.load(fh)
        log(f"  address book: reused {cache.name}")
    else:
        Y, T = R._pool_span_targets(store, pool, basis, log=log)
        ntr = int(0.9 * Y.shape[0])
        addr = TensorAddress(
            cards,
            rank=int(R.READ_PINS["rank"]),
            ridge=float(R.READ_PINS["ridge"]),
            seed=int(R.READ_PINS["address_seed"]),
        )
        addr.multiplicative = verb != "tensor_sum"
        addr.fit(
            Y[:ntr],
            [t[:ntr] for t in T],
            sweeps=int(R.READ_PINS["sweeps"]),
            log=log,
            Y_val=Y[ntr:],
            targets_val=[t[ntr:] for t in T],
        )
        with open(cache, "wb") as fh:
            pickle.dump(addr, fh)
    obj = {
        "arm": arm,
        "read": verb,
        "kind": "span",
        "store": store,
        "basis": basis,
        "address": addr,
        "byte_footprint": dict(base["byte_footprint"]),
        "address_bytes": addr.byte_footprint(),
        "head_bytes": {
            "head_bytes": 0,
            "note": "this read's address book IS the tensor book, declared beside the store and never netted in",
        },
        "partition": base["partition"],
        "label_use": base["label_use"],
        "content": "per-well principal subspace of the fit pool (unsupervised)",
    }
    vals0 = grid.values_of(pool[:64])
    obj["request_only"] = assert_read_request_only(
        addr, store, vals0, grid.values, basis, rng_seed=int(fw.PINS["seed"])
    )
    if kind is not None and kind not in PCA_PRE_FIT:
        obj["store"] = perturb_modes(store, kind, int(fw.PINS["seed"]))
        obj["perturbation"] = {"kind": kind, "when": "after the address book was fitted"}
        if kind == "zero":
            obj["zeroed"] = True
            obj["basis"] = basis
        else:
            obj["basis"] = basis_at_fixed_rank(
                obj["store"], [np.asarray(U).shape[1] for U in basis]
            )
            obj["basis_rank_note"] = (
                "the control keeps the ARM's address book, so its basis is truncated or zero-padded to the arm's per-well rank; a rank the book never learned to address carries nothing it could use"
            )
    elif kind is not None:
        obj["perturbation"] = {"kind": kind, "when": "BEFORE the address book was fitted"}
    led = obj["store"].byte_footprint() if hasattr(obj["store"], "byte_footprint") else None
    if led is not None and (not led["matches_budget"]):
        raise AssertionError(
            f"{arm} deploys {led['deployed_bytes']} B, not the pinned {CELL_PINS['store_bytes']} B"
        )
    obj["fit_seconds"] = time.time() - t0
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump(
        {k: v for k, v in obj.items() if k not in ("store", "basis", "address")},
        arm_dir / "arm.json",
    )
    log(
        f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; store {obj['byte_footprint']['deployed_bytes']:,} B + address {obj['address_bytes'].get('address_bytes', 0):,} B"
    )
    return obj


def build_arm(arm: str, out_dir: Path, log: Callable = print) -> dict:
    """Dispatch: the field legs here, every pre-existing leg to ``exp_gen_read``."""
    if arm not in FIELD_ARMS:
        return _ORIG_BUILD_ARM(arm, out_dir, log=log)
    out_dir = Path(out_dir)
    arm_dir = out_dir / "arms" / arm
    done = arm_dir / "arm.pkl"
    if done.exists():
        log(f"  arm {arm}: already fitted, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    if arm in PCA_ARMS:
        return _build_pca_arm(arm, out_dir, arm_dir, done, t0, log)
    if arm == "oracle_true_feature":
        obj = {
            "arm": arm,
            "read": "true_feature",
            "kind": "oracle",
            "protos": R._FitFingerprint(()),
            "fingerprint_note": "EMPTY: the identity read deploys no content, so there is nothing a refit could have moved",
            "byte_footprint": {
                "deployed_bytes": None,
                "budget_bytes": int(CELL_PINS["store_bytes"]),
                "param_count": None,
                "what": "NO STORE. This leg returns the requested row's encoder feature unchanged, so it is the ceiling of every store at every byte budget and of the judge path itself.",
            },
            "address_bytes": {"address_bytes": 0},
            "head_bytes": {"head_bytes": 0},
            "declared": "NOT AN ARM (protocol B.3): the identity on the true feature. It bounds this metric, not this store.",
            "fit_seconds": 0.0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump(obj, arm_dir / "arm.json")
        return obj
    if arm == "oracle_content_items":
        fw.build_partitions(out_dir, log=log)
        D = int(fw.PINS["D"])
        n_atoms = int(CELL_PINS["store_bytes"]) // (4 * D)
        labels = np.asarray(dict(np.load(out_dir / "partitions.npz"))["factor"], np.int64)
        K = int(labels.max()) + 1
        wc = [np.flatnonzero(labels == k).astype(np.int64) for k in range(K)]
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        rng = np.random.default_rng(
            int(CELL_PINS["field_atom_seed"])
            + int(CELL_PINS["atom_seed_stride"]) * int(fw.PINS["seed"])
        )
        pick = np.sort(pool[rng.permutation(pool.size)[:n_atoms]])
        Xi = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), pick)
        store = _ItemSpanStore(Xi, wc, D)
        led = store.byte_footprint()
        if not led["matches_budget"]:
            raise AssertionError(
                f"{arm} deploys {led['deployed_bytes']} B against the pinned {CELL_PINS['store_bytes']} B footprint -- the byte pin has moved"
            )
        basis = span_basis(store)
        ranks = [int(U.shape[1]) for U in basis]
        log(
            f"  arm {arm}: {n_atoms} real items x {D} float32, span ranks {ranks} (sum {sum(ranks)}), {led['deployed_bytes']:,} B"
        )
        obj = {
            "arm": arm,
            "read": "oracle_field",
            "kind": "oracle",
            "store": store,
            "basis": basis,
            "byte_footprint": led,
            "protos": R._FitFingerprint(list(store.modes)),
            "address_bytes": {"address_bytes": 0, "what": "a ceiling has no address; protocol B.3"},
            "head_bytes": {"head_bytes": 0},
            "span_audit": {
                "span_ranks": ranks,
                "n_atoms_per_well": n_atoms,
                "n_wells": K,
                "content_source": "64 real pool rows",
                "item_rows": [int(r) for r in pick],
                "total_reachable_dim": int(sum(ranks)),
            },
            "declared": "NOT AN ARM (protocol B.3): the read is handed the true feature. The CONTENT is 64 whole written items, so this is what the store's own semantics can reach.",
            "fit_seconds": time.time() - t0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump(
            {k: v for k, v in obj.items() if k not in ("store", "basis")}, arm_dir / "arm.json"
        )
        return obj
    if arm in CONTENT_SOURCE:
        fw.build_partitions(out_dir, log=log)
        D = int(fw.PINS["D"])
        n_atoms = int(CELL_PINS["store_bytes"]) // (4 * D)
        labels = np.asarray(dict(np.load(out_dir / "partitions.npz"))["factor"], np.int64)
        K = int(labels.max()) + 1
        wc = [np.flatnonzero(labels == k).astype(np.int64) for k in range(K)]
        part_name = "factor (K=5, inherited: the field store's own wells)"
        if CONTENT_PARTITION.get(arm) == "fine":
            order = np.concatenate(wc)
            wc = [
                np.ascontiguousarray(order[s : s + n_atoms]) for s in range(0, order.size, n_atoms)
            ]
            K = len(wc)
            part_name = f"fine (K={K}, the inherited factor wells REFINED into blocks of at most A={n_atoms} channels; same 524,288 B)"
        log(f"  arm {arm}: partition = {part_name}")
        rows = _content_source_rows(out_dir, CONTENT_SOURCE[arm], log)
        X = fw.gather(fw.features_mmap(int(fw.PINS["seed"])), rows)
        store = PCASpanStore(X, wc, n_atoms, D)
        del X
        led = store.byte_footprint()
        if not led["matches_budget"]:
            raise AssertionError(
                f"{arm} deploys {led['deployed_bytes']} B against the pinned {CELL_PINS['store_bytes']} B footprint -- the byte pin has moved"
            )
        basis = span_basis(store)
        ranks = [int(U.shape[1]) for U in basis]
        log(
            f"  arm {arm}: {n_atoms} atoms/well, span ranks {ranks} (sum {sum(ranks)}), {led['deployed_bytes']:,} B, variance explained {[round(e, 4) for e in store.energy]}"
        )
        obj = {
            "arm": arm,
            "read": "oracle_field",
            "kind": "oracle",
            "store": store,
            "basis": basis,
            "byte_footprint": led,
            "protos": R._FitFingerprint(list(store.modes)),
            "address_bytes": {"address_bytes": 0, "what": "a ceiling has no address; protocol B.3"},
            "head_bytes": {"head_bytes": 0},
            "span_audit": {
                "span_ranks": ranks,
                "n_atoms_per_well": n_atoms,
                "n_wells": K,
                "content_source": CONTENT_SOURCE[arm],
                "partition": part_name,
                "well_sizes": [int(c.size) for c in wc],
                "total_reachable_dim": int(sum(ranks)),
                "n_source_rows": int(rows.size),
                "variance_explained_per_well": store.energy,
            },
            "declared": "NOT AN ARM (protocol B.3): the read is handed the true feature. "
            + (
                "The CONTENT is also chosen from the SCORED CORNERS' OWN ROWS, which no deployable store can do -- this is an UNREACHABLE upper bound."
                if CONTENT_SOURCE[arm] == "corners"
                else "The content is the L2-optimal 524,288 B chosen from TRAINING rows only -- the best content a store could actually be built to hold."
            ),
            "fit_seconds": time.time() - t0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump(
            {k: v for k, v in obj.items() if k not in ("store", "basis")}, arm_dir / "arm.json"
        )
        return obj
    built = build_field_for_leg(out_dir, log=log)
    field = built["field"]
    store = FieldSpanStore(field)
    audit = _assert_span_is_the_content(store, field)
    basis = span_basis(store)
    log(
        f"  arm {arm}: {audit['n_live_atoms']} live atoms, span ranks {audit['span_ranks']} (sum {sum(audit['span_ranks'])})"
    )
    obj = {
        "arm": arm,
        "read": "oracle_field" if arm == "oracle_span_field" else "field_argmax",
        "kind": "oracle",
        "protos": R._FitFingerprint(list(store.modes)),
        "store": store,
        "basis": basis,
        "field": field,
        "byte_footprint": {
            "deployed_bytes": int(built["byte_footprint"]["deployed_bytes"]),
            "budget_bytes": int(CELL_PINS["store_bytes"]),
            "param_count": int(built["byte_footprint"]["deployed_bytes"]) // 4,
            "what": "the FIELD store's atoms, the SAME 524,288 B footprint the round-3 store deploys; the interface segments are declared beside it and never netted into it",
            "interface_bytes": int(built["byte_footprint"].get("interface_bytes", 0)),
            "total_image_bytes": int(built["byte_footprint"].get("total_image_bytes", 0)),
        },
        "address_bytes": {
            "address_bytes": 0,
            "what": "a ceiling has no address; protocol B.3 disqualifies it",
        },
        "head_bytes": {"head_bytes": 0},
        "span_audit": audit,
        "field_manifest": {
            "sha256": field.sha256(),
            "n_atoms": int(field.A),
            "n_wells": int(field.K),
        },
        "request_only": built["request_only"],
        "fit_seconds": time.time() - t0,
    }
    if arm == "oracle_span_field":
        obj["declared"] = (
            "NOT AN ARM (protocol B.3): it reads the requested row's own feature. It is the CEILING of the CONTENT dial."
        )
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump(
        {k: v for k, v in obj.items() if k not in ("store", "basis", "field")}, arm_dir / "arm.json"
    )
    return obj


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    """Dispatch: the two field verbs here, everything else to ``exp_gen_read``."""
    verb = obj.get("read", "head")
    if verb == "true_feature":
        rows, oor = R.grid_rows_of(values, grid)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        return (fw.gather(F, rows), oor)
    if verb == "oracle_field":
        rows, oor = R.grid_rows_of(values, grid)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        X = fw.gather(F, rows)
        store, basis = (obj["store"], obj["basis"])
        return (render_span(store, span_coefficients(store, X, basis), basis), oor)
    if verb == "field_argmax":
        from nclu.core.gen_field import field_address

        field = obj["field"]
        V = np.asarray(values, np.float64)
        a = field_address(field, V, grid.values)
        addr = np.asarray(a["mu"], np.float32)
        oor = np.asarray(a["out_of_range"], bool)
        out = np.zeros((V.shape[0], int(field.D)), np.float32)
        alive = np.asarray(field.view("atom_alive"), np.uint8) > 0
        for k in range(field.K):
            ch = np.asarray(field.meta["well_channels"][k], np.int64)
            Ak = np.asarray(field.atoms_of_well(k), np.float32)[alive]
            q = addr[:, ch]
            d2 = np.sum(q * q, 1)[:, None] - 2.0 * (q @ Ak.T) + np.sum(Ak * Ak, 1)[None, :]
            out[:, ch] = Ak[np.argmin(d2, axis=1)]
        return (out, oor)
    return _ORIG_ARM_FEATURES(obj, values, grid)


_ORIG_BUILD_ARM = R.build_arm
_ORIG_ARM_FEATURES = R.arm_features
_ORIG_ARMS = tuple(R.ARMS)


def install() -> dict:
    before = {
        "arms": tuple(R.ARMS),
        "store": dict(R.ARM_STORE),
        "read": dict(R.ARM_READ),
        "cls": dict(R.ARM_CLASS),
    }
    R.ARMS = tuple(_ORIG_ARMS) + FIELD_ARMS
    R.ARM_STORE.update({a: "field" for a in FIELD_ARMS})
    R.ARM_READ.update(
        {
            "oracle_span_field": "oracle_field",
            "field_lookup_argmax": "field_argmax",
            "oracle_content_pool": "oracle_field",
            "oracle_content_oracle": "oracle_field",
            "oracle_content_fine": "oracle_field",
            "oracle_content_items": "oracle_field",
            "oracle_true_feature": "true_feature",
        }
    )
    R.ARM_READ.update({a: v for a, (_k, v) in PCA_ARMS.items()})
    R.ARM_STORE.update({a: "pca" for a in PCA_ARMS})
    R.ARM_CLASS.update(FIELD_ARM_CLASS)
    R.NOT_ARMS = tuple(R.NOT_ARMS) + (
        "oracle_span_field",
        "oracle_content_pool",
        "oracle_content_oracle",
        "oracle_content_fine",
        "oracle_content_items",
        "oracle_true_feature",
    )
    R.build_arm = build_arm
    R.arm_features = arm_features
    for a in before["arms"]:
        if (
            R.ARM_STORE.get(a) != before["store"].get(a)
            or R.ARM_READ.get(a) != before["read"].get(a)
            or R.ARM_CLASS.get(a) != before["cls"].get(a)
        ):
            raise AssertionError(f"installing this study moved pre-existing arm {a!r}")
    if tuple(R.ARMS)[: len(_ORIG_ARMS)] != _ORIG_ARMS:
        raise AssertionError("installing this study reordered exp_gen_read.ARMS")
    return {"added": list(FIELD_ARMS), "n_arms": len(R.ARMS), "pre_existing_unchanged": True}


def _make_artifacts_portable() -> dict:
    import sys

    names = ("FieldSpanStore", "PCASpanStore", "_ItemSpanStore", "_ModeStore")
    g = globals()
    m = sys.modules.get("__main__")
    if __name__ != "__main__" and m is not None and (m is not sys.modules.get(__name__)):
        for n in names:
            setattr(m, n, g[n])
        return {
            "mode": "package classes installed into __main__ (one copy in the process; pre-fix artifacts resolve to it too)",
            "classes": names,
        }
    return {"mode": "no action needed"}


_PORTABILITY = _make_artifacts_portable()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("pins", "prepare", "fit", "run", "table"))
    p.add_argument("--arm", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--max-seconds", type=float, default=2700.0)
    p.add_argument("--primary", default="oracle_span_field")
    p.add_argument("--margin", type=float, default=1.0)
    a = p.parse_args(argv)
    inst = install()
    R.set_condition(a.seed, a.condition)
    _bi = R.bind_instrument(log=lambda m: print(m, flush=True))
    out = R.out_leg(Path(a.out), a.seed, a.condition)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"stage={a.stage} arm={a.arm} out={out} seed={a.seed} cond={a.condition} install={inst}")
    if a.stage == "pins":
        fw._jdump(
            {
                "cell_pins": CELL_PINS,
                "read_pins": R.READ_PINS,
                "fw_pins": fw.PINS,
                "field_pins": {
                    k: list(v) if isinstance(v, tuple) else v for k, v in fs.PINS.items()
                },
                "install": inst,
                "instrument": _bi,
                "bridge_root": os.environ.get("NCLU_ENCODER_ROOT"),
                "feature_cache": R._cache_manifest(int(a.seed)),
            },
            out / "pins.json",
        )
        print(json.dumps({"cell_pins": CELL_PINS, "install": inst}, indent=1, default=str))
        return 0
    if a.stage == "prepare":
        log(json.dumps(R.stage_prepare(out, log=log)))
        return 0
    if a.stage == "fit":
        log(json.dumps(R.stage_fit(a.arm, out, log=log), default=str))
        return 0
    if a.stage == "run":
        log(json.dumps(R.stage_run(a.arm, out, a.max_seconds, log=log), default=str))
        return 0
    if a.stage == "table":
        with R.read_dial():
            t = R.build_table(out, primary=a.primary, margin=a.margin, log=log)
        for name, blk in t["arms"].items():
            r4 = blk.get("r4_comp_nway") or {}
            r3 = blk.get("r3_comp_2way") or {}
            log(
                f"    {name:22s} {blk['class']:40s} r4={r4.get('novel_tuples_per_corner', float('nan')):7.3f} r3={r3.get('novel_tuples_per_corner', float('nan')):7.3f}"
            )
        return 0
    raise SystemExit(f"stage {a.stage} not implemented")


if __name__ == "__main__":
    from nclu.experiments.composition_cell import main as _pkg_main

    raise SystemExit(_pkg_main())
