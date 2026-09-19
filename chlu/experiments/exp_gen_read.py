"""hub/read -- the multi-corner runner for the seven-row scoreboard, with the READ as
the dial.

ONE path, identical for every arm -- the same path wave 3 ran::

    request -> the ADDRESS -> the store's atoms -> THE frozen decoder
            -> image -> THE external classifier -> factor tuple

DIAL DECLARATION -- *how a well's content becomes an answer*.  Every arm here shares the
same encoder, the same frozen feature cache, the same fit pool, the same partition
(``factor_aligned_balanced``, the wave-3 arm), the same mode count, the same 524,288 B
store ledger, the same decoder and the same judge.  ⛔ Not the partition (that is
``hub/placement``), ⛔ not what is stored.  **The map from a request to the coefficients
that address the atoms is the only thing that moves.**

Why the row machinery is IMPORTED and not rewritten
---------------------------------------------------
``exp_gen_factor_wells`` already defines the corners, the judge, the acceptance rule, the
validity rule, the per-corner scores, the paired estimand and the deletion row, and it is
the module that produced wave 3's numbers.  This module **imports it and replaces exactly
two functions inside it** -- ``build_arm`` (what an arm IS) and ``arm_features`` (how an
arm READS) -- under :func:`read_dial`, a context manager that restores them on exit.  So
every row, every corner and every statistic on this hub is computed by the same code as
wave 3's, and the diff between the two hubs is the dial and nothing else.  The
replacement is asserted on entry and on exit (:func:`read_dial` raises if the functions
it is restoring are not the ones it replaced).

Runnable::

    python -m chlu.experiments.exp_gen_read --stage prepare --out DIR [--seed S] [--condition val|test]
    python -m chlu.experiments.exp_gen_read --stage fit   --arm ARM --out DIR
    python -m chlu.experiments.exp_gen_read --stage run   --arm ARM --out DIR
    python -m chlu.experiments.exp_gen_read --stage table --out DIR

Every stage is resumable (one artifact per (stage, arm), skipped when complete) and takes
``--max-seconds``, honoured between corners.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import exp_gen_factor_wells as fw

#: ⭐ Captured AT IMPORT, before :func:`read_dial` ever replaces anything.  Everything in
#: this module that needs wave 3's own builder calls THIS, so the dial can never recurse
#: into itself and the store an arm reads is always wave 3's object.
_FW_BUILD_ARM = fw.build_arm
_FW_ARM_FEATURES = fw.arm_features
from ..core.gen_read import (
    TensorAddress, assert_read_request_only, render_span, span_basis,
    span_coefficients,
)

# ---------------------------------------------------------------------------
# 0. THE PINS OF THE READ -- everything the dial owns, pinned in code
# ---------------------------------------------------------------------------

#: Pinned before any scored run.  ``fw.PINS`` carries everything else and is NOT edited
#: on disk: :func:`set_condition` writes ``seed`` and ``condition`` into it at runtime,
#: which is how the same pinned store recipe is re-fitted on cache generation 2 seeds
#: 0, 1 and 2 without a second copy of the recipe existing anywhere.
READ_PINS = {
    "partition": "balanced",          # the wave-3 factor-aligned, equal-well partition
    "rank": 128,                      # R -- the address width, the read's only capacity dial
    "ridge": 1e-2,                    # the ALS ridge, both half-steps
    "sweeps": 40,
    "address_seed": 0,                # initialisation only; both half-steps are closed form
    "n_addr_train": 131_072,          # the whole pinned fit pool, exactly as the head's gram
    "addr_chunk": 8_192,
    "standardise_targets": False,     # ⛔ MEASURED: standardising costs the hues (LOG 17:35Z)
}

#: Every arm on this hub.  ``span_head`` FIRST -- it is the like-for-like: the SAME store
#: read by the incumbent ridge head, so a margin over it is attributable to the read
#: alone.  ⚠ w4-common section 1: a win over it is an **ABLATION**, not a result.
ARMS = (
    "span_head",            # the incumbent read on the same store (the ABLATION baseline)
    "tensor_addr",          # THE ARM: the product ("tensor") address book
    "tensor_addr_sum",      # ablation: same R, the address SUMMED not multiplied
    "pvm_span",             # the task file's named first mechanism, projected into the span
    "pvm_table",            # the plain per-value-mean table, full D -- the specialist
    "oracle_span",          # ⛔ NOT AN ARM: the ceiling of this dial
    "landscape_removed",    # charter C.2 -- atoms zeroed, must collapse
    "landscape_shuffled",   # charter C.2 -- atom VALUES permuted, must collapse
    "random_span_refit",    # stronger than C.2 asks: random atoms, address book REFIT
    "span_rotated",         # a PREDICTED no-op: what the read actually uses
    "lookup_argmax",        # charter C.1/C.4 -- one atom per well, my address
    "lookup_convex",        # charter C.1 -- the mixture, simplex imposed
    "corr_linear",          # w4-common section 1 -- correlation-split store + linear head
    "knn_proto",            # the prototype table rival
)

#: Which store each arm deploys.  ``None`` = the arm deploys no store.
ARM_STORE = {
    "span_head": "balanced",
    "tensor_addr": "balanced",
    "tensor_addr_sum": "balanced",
    "pvm_span": "balanced",
    "pvm_table": None,
    "oracle_span": "balanced",
    "landscape_removed": "balanced",
    "landscape_shuffled": "balanced",
    "random_span_refit": "balanced",
    "span_rotated": "balanced",
    "lookup_argmax": "balanced",
    "lookup_convex": "balanced",
    "corr_linear": "ward",
    "knn_proto": None,
}

#: Which read verb each arm uses.
ARM_READ = {
    "span_head": "head",
    "tensor_addr": "tensor",
    "tensor_addr_sum": "tensor_sum",
    "pvm_span": "pvm_span",
    "pvm_table": "pvm_table",
    "oracle_span": "oracle",
    "landscape_removed": "tensor",
    "landscape_shuffled": "tensor",
    "random_span_refit": "tensor",
    "span_rotated": "tensor",
    "lookup_argmax": "tensor_argmax",
    "lookup_convex": "tensor_convex",
    "corr_linear": "head",
    "knn_proto": "proto",
}

#: Arms that are declared NOT ARMS and are never quoted as a mechanism.
NOT_ARMS = ("oracle_span",)


def _now() -> float:
    return time.time()


def set_condition(seed: int, condition: str) -> dict:
    """Write ``seed`` and ``condition`` into ``fw.PINS`` at runtime.

    ⛔ This does not edit ``exp_gen_factor_wells`` on disk and it changes no recipe: the
    module's pins are a dict, and ``seed``/``condition`` are exactly the two entries that
    were always meant to move (``fw.PINS['condition']`` already switches the selection
    set).  Every other pin -- the partition, the mode count, the byte ledger, the store
    fit, the head design, the ridge -- is untouched, which is what makes this hub's
    stores bit-comparable with wave 3's.
    """
    if str(condition) not in ("val", "test"):
        raise ValueError(f"condition must be val or test, got {condition!r}")
    fw.PINS["seed"] = int(seed)
    fw.PINS["condition"] = str(condition)
    return {"seed": int(seed), "condition": str(condition)}


def out_leg(base: Path, seed: int, condition: str) -> Path:
    return Path(base) / f"{condition}_seed{int(seed)}"


# ---------------------------------------------------------------------------
# 1. The per-value-mean table -- the specialist this hub must beat or admit to
# ---------------------------------------------------------------------------


class PerValueMeanTable:
    """``mu + sum_f ( mu_{f,v_f} - mu )`` over the fit pool.  Additive, by construction.

    This is the object the target file names: *"address inside each well by the per-value
    average of that well content"*.  It is kept on **every** row of this hub's table
    whether it wins or loses, because if it matches the arm then the store is not earning
    its bytes and this hub says so in those words.
    """

    def __init__(self, cardinalities: Sequence[int]):
        self.cardinalities = [int(c) for c in cardinalities]
        self.mu: Optional[np.ndarray] = None
        self.dev: list = []
        self.n_fit_rows = 0

    def fit(self, Y: np.ndarray, X_iter) -> "PerValueMeanTable":
        """One streaming pass: per-(factor, value) sums, then the deviations."""
        Yi = np.asarray(Y, np.int64)
        tot = None
        sums = None
        cnts = [np.zeros(int(c), np.int64) for c in self.cardinalities]
        n = 0
        s = 0
        for Xc in X_iter:
            Xc = np.asarray(Xc, np.float64)
            e = s + Xc.shape[0]
            if tot is None:
                tot = np.zeros(Xc.shape[1], np.float64)
                sums = [np.zeros((int(c), Xc.shape[1]), np.float64)
                        for c in self.cardinalities]
            tot += Xc.sum(0)
            for f in range(Yi.shape[1]):
                np.add.at(sums[f], Yi[s:e, f], Xc)
                cnts[f] += np.bincount(Yi[s:e, f], minlength=int(self.cardinalities[f]))
            n += Xc.shape[0]
            s = e
        self.mu = tot / max(1, n)
        self.dev = [np.where(cnts[f][:, None] > 0,
                             sums[f] / np.maximum(cnts[f][:, None], 1) - self.mu[None, :],
                             0.0)
                    for f in range(len(self.cardinalities))]
        self.n_fit_rows = int(n)
        return self

    def read(self, values: np.ndarray, grid_values) -> tuple:
        """``(N, F) requested VALUES -> ((N, D), out-of-range mask)``, request-only."""
        from ..core.gen_request_head import one_hot_interpolated
        V = np.asarray(values, np.float64)
        out = np.repeat(self.mu[None, :], V.shape[0], 0).copy()
        oor = np.zeros(V.shape[0], bool)
        for f in range(V.shape[1]):
            Wf, of = one_hot_interpolated(V[:, f], grid_values[f])
            oor |= of
            out += np.asarray(Wf, np.float64) @ self.dev[f]
        return out.astype(np.float32), oor

    def byte_ledger(self) -> dict:
        b = int((self.mu.size + sum(d.size for d in self.dev))) * 4
        return {
            "deployed_bytes": b,
            "what": "the global mean plus one per-(factor,value) mean DEVIATION, full D",
            "note": "no store, no atoms -- this row is the specialist, not an arm of ours",
        }


# ---------------------------------------------------------------------------
# 2. Building an arm:  a store (someone else's dial) + an address (mine)
# ---------------------------------------------------------------------------


def _pool_span_targets(store, pool: np.ndarray, basis, log: Callable = print) -> tuple:
    """Stream the fit pool into per-well span coordinates.  ``(Y, [ (N, r_k) ])``."""
    F = fw.features_mmap(int(fw.PINS["seed"]))
    grid, _ = fw.build_grid()
    n = int(pool.size)
    chunk = int(READ_PINS["addr_chunk"])
    parts = None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        X = fw.gather(F, pool[s:e])
        cs = span_coefficients(store, X, basis)
        if parts is None:
            parts = [np.empty((n, c.shape[1]), np.float32) for c in cs]
        for p, c in zip(parts, cs):
            p[s:e] = c
        del X, cs
    log(f"  span targets: {n} rows -> " + " ".join(str(p.shape) for p in parts))
    return grid.factor_indices_of(pool), parts


#: Perturbations of the LANDSCAPE, and when each is applied.  PRE_FIT ones change what
#: the store IS and the address book is refitted against them; POST_FIT ones keep the
#: fitted address book and break the thing it was fitted to.
PRE_FIT_PERTURB = {
    "random_span_refit": "random_span",
    "span_rotated": "rotate",
}
POST_FIT_PERTURB = {
    "landscape_removed": "zero",
    "landscape_shuffled": "value_permute",
}


def perturb_store(store, kind: str, seed: int):
    """Return a store with the same SHAPE and the same byte ledger, atoms changed.

    ``zero``
        charter C.2 -- the atoms hold nothing.  The bytes are still deployed, so the row
        stays byte-matched and a collapse is attributable to the atoms and not to a
        smaller object.
    ``value_permute``
        charter C.2's informative half, **redefined for a span read**.  Wave 3 permuted
        the MODE INDEX, which a span read is exactly invariant to (permuting the rows of
        ``G`` does not move its left singular vectors), so that control would prove
        nothing here.  This permutes the atom VALUES inside each well's mode matrix: the
        same numbers, the same ledger, the geometry destroyed.
    ``random_span``
        stronger than the charter asks: the atoms are replaced by a random Gaussian
        matrix of the same shape, scaled to the same Frobenius norm, and the address book
        is REFIT.  If the arm survives this, the answer lives in the address book and not
        in the store.
    ``rotate``
        a random orthogonal rotation of the atoms INSIDE each well.  The span is
        mathematically unchanged, so this is a **predicted no-op** and it is reported as
        one: it states that this read uses a well's subspace and not the identity of the
        atoms that span it.
    """
    import jax
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_919)
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
            out.append((Q.astype(np.float32) @ Gm))
        else:
            raise ValueError(kind)
    return type(store)(store.well_channels, int(fw.PINS["n_modes"]), store.feat_shape,
                       jax.random.PRNGKey(0), init=out)


def build_arm(arm: str, out_dir: Path, log: Callable = print) -> dict:
    """Fit one arm: its store (via ``fw.build_arm``, unchanged) and its ADDRESS.

    ⭐ The store is **never refitted here**.  ``fw.build_arm`` is called with the name of
    the wave-3 arm that owns the partition, so the atoms this hub reads are produced by
    the same code, the same pool and the same pinned recipe that produced wave 3's.
    """
    out_dir = Path(out_dir)
    arm_dir = out_dir / "arms" / arm
    done = arm_dir / "arm.pkl"
    if done.exists():
        log(f"  arm {arm}: already fitted, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    t0 = _now()
    verb = ARM_READ[arm]
    store_name = ARM_STORE[arm]

    # -- the arms that deploy no store -------------------------------------
    if arm == "knn_proto":
        obj = _FW_BUILD_ARM("knn_proto", out_dir, log=log)
        obj = dict(obj)
        obj["read"] = "proto"
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        return obj
    if arm == "pvm_table":
        fw.build_partitions(out_dir, log=log)
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        Y = grid.factor_indices_of(pool)
        chunk = int(READ_PINS["addr_chunk"])
        tbl = PerValueMeanTable(cards).fit(
            Y, (fw.gather(F, pool[s:s + chunk]) for s in range(0, pool.size, chunk)))
        obj = {"arm": arm, "read": "pvm_table", "table": tbl,
               "byte_ledger": tbl.byte_ledger(),
               "address_bytes": {"address_bytes": 0,
                                 "note": "the table IS its own address book"},
               "fit_seconds": _now() - t0,
               "prov": {"n_rows_fit": tbl.n_fit_rows}}
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump({k: v for k, v in obj.items() if k != "table"}, arm_dir / "arm.json")
        log(f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; "
            f"{tbl.byte_ledger()['deployed_bytes']:,} B, no store")
        return obj

    # -- the store, from wave 3's own builder, unchanged --------------------
    src = {"balanced": "factor_aligned_balanced", "ward": "basis_head"}[store_name]
    base = _FW_BUILD_ARM(src, out_dir, log=log)
    store, heads, spec = base["store"], base["heads"], base["spec"]

    # -- perturbations applied BEFORE the address book is fitted -----------
    # These change what the store IS, so the address book is refitted against the
    # perturbed atoms.  That is the strong form of the charter C.2 question: if the arm
    # still works when the atoms are replaced, the answer was never in the store.
    if arm in PRE_FIT_PERTURB:
        store = perturb_store(store, PRE_FIT_PERTURB[arm], int(fw.PINS["seed"]))

    basis = span_basis(store)
    obj = {"arm": arm, "read": verb, "store": store, "heads": heads, "spec": spec,
           "basis": basis, "partition": base.get("partition"),
           "byte_ledger": base["byte_ledger"], "store_ledger": base.get("store_ledger"),
           "head_bytes": base["head_bytes"], "labels": base.get("labels"),
           "recon_mse": base.get("recon_mse")}

    if verb in ("tensor", "tensor_sum", "tensor_argmax", "tensor_convex"):
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        pool = pool[: int(READ_PINS["n_addr_train"])]
        # the address book is fitted ONCE per (store, condition, seed) and shared by every
        # arm that reads that store, so an arm's margin is never a second fit.
        tag = PRE_FIT_PERTURB.get(arm)
        cache = out_dir / (f"address_{store_name}_{verb}"
                           + (f"_{tag}" if tag else "") + ".pkl")
        if cache.exists():
            with open(cache, "rb") as fh:
                addr = pickle.load(fh)
            log(f"  address book: reused {cache.name}")
        else:
            Y, T = _pool_span_targets(store, pool, basis, log=log)
            ntr = int(0.9 * Y.shape[0])
            addr = TensorAddress(cards, rank=int(READ_PINS["rank"]),
                                 ridge=float(READ_PINS["ridge"]),
                                 seed=int(READ_PINS["address_seed"]))
            addr.multiplicative = (verb != "tensor_sum")
            addr.fit(Y[:ntr], [t[:ntr] for t in T], sweeps=int(READ_PINS["sweeps"]),
                     log=log, Y_val=Y[ntr:], targets_val=[t[ntr:] for t in T])
            with open(cache, "wb") as fh:
                pickle.dump(addr, fh)
        obj["address"] = addr
        obj["address_bytes"] = addr.byte_ledger()
        vals0 = grid.values_of(pool[:64])
        obj["request_only"] = assert_read_request_only(
            addr, store, vals0, grid.values, basis, rng_seed=int(fw.PINS["seed"]))
    elif verb == "pvm_span":
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        Y = grid.factor_indices_of(pool)
        chunk = int(READ_PINS["addr_chunk"])
        tbl = PerValueMeanTable(cards).fit(
            Y, (fw.gather(F, pool[s:s + chunk]) for s in range(0, pool.size, chunk)))
        obj["table"] = tbl
        obj["address_bytes"] = {
            "address_bytes": tbl.byte_ledger()["deployed_bytes"],
            "note": "the per-value-mean table, used as the address and then projected "
                    "into span(G_k) -- the target file's named first mechanism"}
    else:
        obj["address_bytes"] = dict(base["head_bytes"])

    # -- perturbations applied AFTER the address book is fitted ------------
    if arm in POST_FIT_PERTURB:
        kind = POST_FIT_PERTURB[arm]
        obj["store"] = perturb_store(store, kind, int(fw.PINS["seed"]))
        obj["perturbation"] = {"kind": kind, "when": "after the address book was fitted"}
        if kind == "zero":
            obj["zeroed"] = True
            obj["basis"] = basis
        else:
            obj["basis"] = span_basis(obj["store"])

    obj["fit_seconds"] = _now() - t0
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items()
               if k not in ("store", "heads", "spec", "labels", "basis", "address",
                            "table", "protos")}, arm_dir / "arm.json")
    log(f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; store "
        f"{obj['byte_ledger']['deployed_bytes']:,} B + address "
        f"{obj['address_bytes'].get('address_bytes', 0):,} B")
    return obj


# ---------------------------------------------------------------------------
# 3. The read -- the one function this hub exists to move
# ---------------------------------------------------------------------------


def grid_rows_of(values: np.ndarray, grid) -> tuple:
    """(N, F) requested VALUES -> the dataset row realising them, nearest on the grid.

    Used ONLY by the ``oracle`` non-arm.  shapes3d is a full factorial, so a tuple of
    factor indices IS a row id; an unrequested or off-grid value takes the nearest grid
    index and the row is flagged, which is the most generous reading available to the
    ceiling and therefore the conservative direction for every arm measured against it.
    """
    V = np.asarray(values, np.float64)
    cards = tuple(int(c) for c in grid.cardinalities)
    idx = np.zeros((V.shape[0], len(cards)), np.int64)
    oor = np.zeros(V.shape[0], bool)
    for f in range(len(cards)):
        gv = np.asarray(grid.values[f], np.float64)
        v = V[:, f]
        asked = ~np.isnan(v)
        vv = np.where(asked, v, gv[0])
        idx[:, f] = np.argmin(np.abs(vv[:, None] - gv[None, :]), axis=1)
        oor |= asked & ((v < gv[0]) | (v > gv[-1]))
    return np.ravel_multi_index(tuple(idx.T), cards), oor


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    """(N, F) requested VALUES -> (N, D) features + the out-of-range mask.

    No feature, no row index and no label enters any branch except ``oracle`` -- which is
    DECLARED NOT AN ARM for exactly that reason and is reported as the ceiling only.
    """
    verb = obj.get("read", "head")
    if verb == "proto":
        return _FW_ARM_FEATURES(obj, values, grid)
    if verb == "pvm_table":
        return obj["table"].read(values, grid.values)
    store, basis = obj["store"], obj.get("basis")
    if obj.get("zeroed"):
        # the atoms hold nothing, so the read is identically zero whatever the address
        # says -- which is what charter C.2's "atoms zeroed" means, stated rather than
        # left to an empty SVD basis.
        return (np.zeros((np.asarray(values).shape[0], int(store.dim)), np.float32),
                np.zeros(np.asarray(values).shape[0], bool))
    if verb == "head":
        from ..core.gen_request_head import head_read, request_design
        Phi, oor = request_design(values, grid.values, obj["spec"])
        return head_read(obj["heads"], store, Phi), oor
    if verb == "pvm_span":
        X, oor = obj["table"].read(values, grid.values)
        return render_span(store, span_coefficients(store, X, basis), basis), oor
    if verb == "oracle":
        # NOT AN ARM (charter B.3): it reads the requested row's own feature, so no
        # request-only guarantee holds for it and it is never quoted as a mechanism.  It
        # is the CEILING of this dial: the best point in span(G_k) for this request.
        rows, oor = grid_rows_of(values, grid)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        X = fw.gather(F, rows)
        return render_span(store, span_coefficients(store, X, basis), basis), oor
    addr = obj["address"]
    if verb in ("tensor", "tensor_sum"):
        return addr.read(store, values, grid.values, basis)
    # the two matched-bytes lookups over the SAME atoms, driven by the SAME address
    coeffs, oor = addr.coefficients(values, grid.values)
    out = np.zeros((np.asarray(values).shape[0], int(store.dim)), np.float32)
    for c, w, G, Uk in zip(coeffs, store.well_channels, store.modes, basis):
        Gm = np.asarray(G, np.float32)
        # the coefficients live in the ORTHONORMAL basis; the lookup needs mode weights,
        # so project the addressed point back onto the atoms and take the verb there.
        pts = np.asarray(c, np.float32) @ np.asarray(Uk, np.float32).T      # (N, d_k)
        d2 = (np.sum(pts * pts, 1)[:, None] - 2.0 * (pts @ Gm.T)
              + np.sum(Gm * Gm, 1)[None, :])
        if verb == "tensor_argmax":
            out[:, np.asarray(w, np.int64)] = Gm[np.argmin(d2, axis=1)]
        else:
            p = np.maximum(-d2 + d2.max(1, keepdims=True), 0.0)
            s = p.sum(1, keepdims=True)
            p = np.where(s > 0, p / np.where(s > 0, s, 1.0), 1.0 / p.shape[1])
            out[:, np.asarray(w, np.int64)] = p @ Gm
    return out, oor


# ---------------------------------------------------------------------------
# 4. The dial -- the two functions this hub replaces inside wave 3's runner
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def read_dial():
    """Replace ``fw.build_arm`` and ``fw.arm_features`` for the duration of a run.

    ⭐ Everything else in ``exp_gen_factor_wells`` -- the corners, the judge, the
    acceptance rule, the validity rule, the per-corner scores, the paired estimand, the
    deletion row -- executes unchanged, so this hub's rows are produced by the same code
    as wave 3's and the diff between the hubs is the dial and nothing else.
    """
    fw_build, fw_read = fw.build_arm, fw.arm_features

    def _build(arm, out_dir, log=print):
        if arm in ("factor_aligned_balanced", "basis_head", "knn_proto"):
            return fw_build(arm, out_dir, log=log)     # the STORE, wave 3's own builder
        return build_arm(arm, out_dir, log=log)

    fw.build_arm = _build
    fw.arm_features = arm_features
    try:
        yield {"replaced": ["build_arm", "arm_features"]}
    finally:
        if fw.arm_features is not arm_features:
            raise AssertionError(
                "arm_features was replaced again inside the dial -- the read is not the "
                "only thing that moved")
        fw.build_arm, fw.arm_features = fw_build, fw_read


# ---------------------------------------------------------------------------
# 5. Stages
# ---------------------------------------------------------------------------


def stage_prepare(out_dir: Path, log: Callable = print) -> dict:
    info = fw.build_partitions(out_dir, log=log)
    return {"well_sizes": info["info"]["well_sizes"][READ_PINS["partition"]]}


def stage_fit(arm: str, out_dir: Path, log: Callable = print) -> dict:
    with read_dial():
        obj = build_arm(arm, out_dir, log=log)
    return {"arm": arm, "address_bytes": obj.get("address_bytes"),
            "byte_ledger": obj.get("byte_ledger")}


def stage_run(arm: str, out_dir: Path, max_seconds: float, log: Callable = print) -> dict:
    with read_dial():
        return fw.run_arm(arm, out_dir, max_seconds=max_seconds, log=log)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("prepare", "fit", "run", "table", "pins"))
    p.add_argument("--arm", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--max-seconds", type=float, default=3600.0)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--sweeps", type=int, default=None)
    p.add_argument("--ridge", type=float, default=None)
    return p


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    if a.rank is not None:
        READ_PINS["rank"] = int(a.rank)
    if a.sweeps is not None:
        READ_PINS["sweeps"] = int(a.sweeps)
    if a.ridge is not None:
        READ_PINS["ridge"] = float(a.ridge)
    set_condition(a.seed, a.condition)
    base = Path(a.out)
    out = out_leg(base, a.seed, a.condition)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"stage={a.stage} arm={a.arm} out={out} seed={a.seed} condition={a.condition} "
        f"rank={READ_PINS['rank']} sweeps={READ_PINS['sweeps']}")
    if a.stage == "pins":
        fw._jdump({"read_pins": READ_PINS, "fw_pins": fw.PINS}, out / "pins.json")
        print(json.dumps({"read_pins": READ_PINS}, indent=1, default=str))
        return 0
    if a.stage == "prepare":
        log(json.dumps(stage_prepare(out, log=log)))
        return 0
    if a.stage == "fit":
        log(json.dumps(stage_fit(a.arm, out, log=log), default=str))
        return 0
    if a.stage == "run":
        log(json.dumps(stage_run(a.arm, out, a.max_seconds, log=log), default=str))
        return 0
    raise SystemExit(f"stage {a.stage} not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
