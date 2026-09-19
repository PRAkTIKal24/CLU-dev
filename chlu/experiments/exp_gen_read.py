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
    "knn_proto",            # the prototype table rival, rebuilt inside this harness
) + tuple(f"rival_{n}" for n in (
    # ⭐ THE FULL RIVAL SET (w4-common section 1), spoke A's fitted objects entered
    # through their own callable and run through MY corners by MY judge, so every row on
    # the table is the same path.  A win over one of these is a RESULT; a win over an
    # ablation is an ABLATION and is labelled so.
    "table", "mlp", "cont_mlp", "transformer", "lstm", "lace", "poe_ebm",
))

#: The rival names, in the order they are printed.
RIVAL_NAMES = ("table", "mlp", "cont_mlp", "transformer", "lstm", "lace", "poe_ebm")

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
ARM_STORE.update({f"rival_{n}": None for n in
                  ("table", "mlp", "cont_mlp", "transformer", "lstm", "lace", "poe_ebm")})

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
ARM_READ.update({f"rival_{n}": "rival" for n in
                 ("table", "mlp", "cont_mlp", "transformer", "lstm", "lace", "poe_ebm")})

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


def bind_instrument(log: Callable = print) -> dict:
    """Point the frozen instrument at the right cache without moving any recipe.

    ⭐ Why this exists, stated rather than hidden.  ``fw.classifier_dir`` derives the
    external classifier's location from the FEATURE CACHE's path
    (``CHLU_GEN_BRIDGE_ROOT.parent.parent / "gen-external-classifier"``).  Every scored
    run on this wave uses spoke A's **cache generation 2**, which lives in a different
    tree from the generation-1 cache, so that derivation would send the judge looking for
    a classifier that is not there.  Setting ``CHLU_GEN_CLASSIFIER_ROOT`` rebinds the
    lookup and **nothing else**: the classifier that is loaded is the same file, with the
    same hash, that ``Judge`` asserts against its own manifest on every arm of every row.
    ⛔ If the hash moved, the judge raises before a number exists.
    """
    root = os.environ.get("CHLU_GEN_CLASSIFIER_ROOT")
    if not root:
        return {"classifier_root": "derived from the cache path (generation 1 layout)"}
    base = Path(root)

    def _cdir(seed: int, _base=base):
        return _base / f"seed_{int(seed)}"

    fw.classifier_dir = _cdir
    log(f"  instrument: classifier root rebound to {base}")
    return {"classifier_root": str(base),
            "note": "the classifier FILE is unchanged and is still hash-asserted by Judge"}


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

    # -- the rivals: spoke A's fitted objects, loaded, never refitted here ---
    if arm.startswith("rival_"):
        name = arm[len("rival_"):]
        from ...eval.rivals.w4_api import load_rival, available  # noqa: F401
        try:
            r = load_rival(int(fw.PINS["seed"]), name)
        except Exception as exc:                       # RivalRefusal or not-yet-fitted
            obj = {"arm": arm, "read": "rival", "not_run": f"{type(exc).__name__}: {exc}",
                   "rival": None, "byte_ledger": {"deployed_bytes": 0,
                                                  "what": "DECLARED NOT-RUN"},
                   "address_bytes": {"address_bytes": 0}}
            arm_dir.mkdir(parents=True, exist_ok=True)
            fw._jdump({k: v for k, v in obj.items() if k != "rival"},
                      arm_dir / "NOT_RUN.json")
            log(f"  arm {arm}: DECLARED NOT-RUN -- {obj['not_run']}")
            return obj
        names = list(getattr(r, "names", grid.names))
        if names != [str(x) for x in grid.names]:
            raise AssertionError(
                f"{arm}: rival factor order {names} != the grid's "
                f"{[str(x) for x in grid.names]} -- a silent reorder would score the "
                "rival on a request it never received")
        meta = {}
        mp = (Path(os.environ.get("CHLU_RIVALS_ROOT",
                                  "C:/chlu/outputs/spoke-a-rivals/rivals"))
              / f"seed_{int(fw.PINS['seed'])}" / f"{name}.rival.json")
        if mp.exists():
            with open(mp, encoding="utf-8") as fh:
                meta = json.load(fh)
        obj = {"arm": arm, "read": "rival", "kind": "rival", "rival": r,
               # fw.run_arm fingerprints "the object that must not have been refitted
               # between the two halves of row 7".  A rival exposes no parameters through
               # the callable, so the fingerprint is EMPTY and that is declared here
               # rather than left to look like a passing check.
               "protos": (),
               "fingerprint_note": ("a rival's parameters are not exposed by spoke A's "
                                    "callable, so its row-7 fit fingerprint is empty and "
                                    "proves nothing; the rival has no deletion operation "
                                    "and its row 7 is a dash"),
               "byte_ledger": {"deployed_bytes": int(meta.get("byte_ledger", {})
                                                     .get("bytes", 0)),
                               "what": f"spoke A's fitted {name}, generation 2",
                               "param_count": meta.get("param_count")},
               "address_bytes": {"address_bytes": 0,
                                 "note": "a rival's parameters ARE its address book"},
               "rival_meta": {k: v for k, v in meta.items()
                              if k in ("param_count", "byte_ledger", "offgrid_capable",
                                       "val_mse_full_schedule",
                                       "val_mse_partial_requests", "n_rows_fit")},
               "fit_seconds": 0.0}
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump({k: v for k, v in obj.items() if k != "rival"}, arm_dir / "arm.json")
        log(f"  arm {arm}: loaded spoke A rival '{name}' "
            f"(params={obj['byte_ledger'].get('param_count')})")
        return obj

    # -- the arms that deploy no store -------------------------------------
    if arm == "knn_proto":
        obj = _FW_BUILD_ARM("knn_proto", out_dir, log=log)
        obj = dict(obj)
        obj["read"] = "proto"
        obj.setdefault("kind", "knn_proto")
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
        obj = {"arm": arm, "read": "pvm_table", "kind": "pvm_table", "table": tbl,
               "protos": tuple([tbl.mu] + list(tbl.dev)),
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
    obj = {"arm": arm, "read": verb, "kind": "span",
           "store": store, "heads": heads, "spec": spec,
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
        # ⭐ The key carries every pin the fit depends on.  A cache keyed only by the
        # store and the verb would silently hand a --rank 64 run the rank-128 book.
        cache = out_dir / (
            f"address_{store_name}_{verb}_R{int(READ_PINS['rank'])}"
            f"_lam{READ_PINS['ridge']}_sw{int(READ_PINS['sweeps'])}"
            f"_n{int(READ_PINS['n_addr_train'])}"
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
        # the incumbent read's address book IS its head; name it the same way so the two
        # reads are compared on one ledger line and not on two differently-named ones.
        hb = dict(base["head_bytes"])
        obj["address_bytes"] = {"address_bytes": int(hb.get("head_bytes", 0)),
                                "per_well": hb.get("head_bytes_per_well"),
                                "dtype": hb.get("dtype"),
                                "bytes_formula": hb.get("bytes_formula"),
                                "note": "the incumbent ridge head -- this read's address book"}

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
    if verb == "rival":
        if obj.get("rival") is None:
            raise AssertionError(f"{obj['arm']} is a DECLARED NOT-RUN and has no read")
        feats, oor = obj["rival"](np.asarray(values, np.float64))
        return np.asarray(feats, np.float32), np.asarray(oor, bool)
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
    fw_build, fw_read, fw_arms = fw.build_arm, fw.arm_features, fw.ARMS
    fw_del = fw.delete_value_from_head

    def _build(arm, out_dir, log=print):
        if arm in ("factor_aligned_balanced", "basis_head", "knn_proto"):
            return fw_build(arm, out_dir, log=log)     # the STORE, wave 3's own builder
        return build_arm(arm, out_dir, log=log)

    fw.build_arm = _build
    fw.arm_features = arm_features
    fw.ARMS = ARMS
    fw.delete_value_from_head = lambda o, f, v: (
        delete_value_from_address(o, f, v) if o.get("read", "").startswith("tensor")
        else fw_del(o, f, v))
    try:
        yield {"replaced": ["build_arm", "arm_features", "ARMS",
                            "delete_value_from_head"]}
    finally:
        if fw.arm_features is not arm_features:
            raise AssertionError(
                "arm_features was replaced again inside the dial -- the read is not the "
                "only thing that moved")
        fw.build_arm, fw.arm_features, fw.ARMS = fw_build, fw_read, fw_arms
        fw.delete_value_from_head = fw_del




# ---------------------------------------------------------------------------
# 6. The table -- every arm, every row, and the estimand that decides 3 and 4
# ---------------------------------------------------------------------------

#: How each arm is labelled on every table.  ⛔ w4-common section 1 is mechanical here:
#: a win over a RIVAL is a RESULT, a win over anything else is an ABLATION and says so.
ARM_CLASS = {
    "tensor_addr": "ARM",
    "oracle_span": "NOT AN ARM (ceiling)",
    "span_head": "ABLATION (the deployed read)",
    "corr_linear": "ABLATION (correlation split + linear head)",
    "tensor_addr_sum": "ABLATION (same rank, address summed not multiplied)",
    "pvm_span": "ABLATION (per-value-mean address, projected into the span)",
    "pvm_table": "SPECIALIST (the plain per-value-mean table, no store)",
    "knn_proto": "SPECIALIST (prototype table, rebuilt here)",
    "landscape_removed": "CONTROL (charter C.2, atoms zeroed)",
    "landscape_shuffled": "CONTROL (charter C.2, atom values permuted)",
    "random_span_refit": "CONTROL (random atoms, address book refit)",
    "span_rotated": "CONTROL (predicted NO-OP: the span is unchanged)",
    "lookup_argmax": "FLOOR (charter C.1/C.4, matched-bytes lookup)",
    "lookup_convex": "FLOOR (charter C.1, the mixture)",
}
ARM_CLASS.update({f"rival_{n}": "RIVAL" for n in RIVAL_NAMES})


def _corner_rows(arm_dir: Path, corners, grid, seen_mask) -> list:
    z = np.load(arm_dir / "rows.npz", allow_pickle=True)
    return fw._per_corner(corners, z["corners"].item(), grid, seen_mask)


def build_table(out_dir: Path, primary: str = "tensor_addr", margin: float = 1.0,
                log: Callable = print) -> dict:
    """Every arm on every row, plus the paired estimand, plus the floor checks.

    ⛔ The verdict on rows 3 and 4 is the multi-corner PAIRED estimand and never a single
    cell: the corner is the unit, the statistic is the mean paired difference, and the
    margin was declared in the PREREG before the first held-out number.
    """
    out_dir = Path(out_dir)
    cond = fw.load_condition()
    grid = cond["grid"]
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    n4 = fw.nway_corners(cond)
    n3 = fw.pair_subcorners(cond)

    have, not_run = [], {}
    for a in ARMS:
        d = out_dir / "arms" / a
        if (d / "rows.npz").exists():
            have.append(a)
        elif (d / "NOT_RUN.json").exists():
            with open(d / "NOT_RUN.json", encoding="utf-8") as fh:
                not_run[a] = json.load(fh).get("not_run")
    log(f"  arms with rows: {have}")
    log(f"  declared NOT-RUN: {list(not_run)}")

    per_corner, summary = {}, {}
    for a in have:
        rows = _corner_rows(out_dir / "arms" / a, n3 + n4, grid, seen_mask)
        by_id = {r["corner_id"]: r for r in rows}
        per_corner[a] = by_id
        with open(out_dir / "arms" / a / "arm.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        blk = {"class": ARM_CLASS.get(a, "?"),
               "store_bytes": meta.get("byte_ledger", {}).get("deployed_bytes"),
               "address_bytes": (meta.get("address_bytes") or {}).get("address_bytes"),
               "param_count": meta.get("byte_ledger", {}).get("param_count")}
        for row_id, cs in (("r3_comp_2way", n3), ("r4_comp_nway", n4)):
            v = np.array([by_id[c["corner_id"]]["distinct_valid_novel_tuples"]
                          for c in cs], float)
            j = np.array([by_id[c["corner_id"]]["judge_accuracy"] for c in cs], float)
            blk[row_id] = {"n_corners": int(v.size),
                           "novel_tuples_per_corner": float(v.mean()),
                           "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                           "judge_accuracy": float(j.mean())}
        z = np.load(out_dir / "arms" / a / "rows.npz", allow_pickle=True)
        if "qrows" in z.files:
            q = z["qrows"].item()
            for row_id in ("r1_seen", "r2_unseen_valid", "r5_comp_refusal",
                           "r6_invalid_refusal"):
                if row_id in q:
                    blk[row_id] = {
                        "accept": float(np.asarray(q[row_id]["accept"], bool).mean()),
                        "valid": float(np.asarray(q[row_id]["valid"], bool).mean())}
        summary[a] = blk

    estimands = {}
    if primary in per_corner:
        for other in have:
            if other == primary:
                continue
            for row_id, cs in (("r3_comp_2way", n3), ("r4_comp_nway", n4)):
                ids = [c["corner_id"] for c in cs]
                A = [per_corner[primary][i] for i in ids]
                Bc = [per_corner[other][i] for i in ids]
                estimands[f"{row_id}:{primary}_vs_{other}"] = dict(
                    fw.paired_estimand(A, Bc, "distinct_valid_novel_tuples",
                                       float(margin), primary, other),
                    comparison_class=ARM_CLASS.get(other, "?"),
                    verdict_rule=("a win over a RIVAL is a RESULT; a win over anything "
                                  "else is an ABLATION (w4-common section 1)"))
    table = {"primary": primary, "margin_declared": float(margin),
             "arms": summary, "not_run": not_run, "estimands": estimands,
             "corner_counts": {"r3_comp_2way": len(n3), "r4_comp_nway": len(n4)},
             "labels_note": ("labels are used ONCE at fit time to align wells and at the "
                             "address book, and nowhere else (charter B.6, w4-common 5)"),
             "pins": {"read": READ_PINS, "fw": {k: (list(v) if isinstance(v, tuple) else v)
                                                for k, v in fw.PINS.items()}}}
    fw._jdump(table, out_dir / "READ-TABLE.json")
    fw._jdump(per_corner, out_dir / "READ-PER-CORNER.json")
    log(f"  table written -> {out_dir / 'READ-TABLE.json'}")
    return table



def delete_value_from_address(obj: dict, f: int, v: int) -> dict:
    """Row 7 for a PRODUCT address: delete one (factor, value) from the address book.

    The incumbent's ``fw.delete_value_from_head`` zeroes the head's columns for
    ``(f, v)``.  The product address has no such column: a value is a ROW of ``E_f``, and
    the address is the product of one row per factor.  Zeroing that row makes the whole
    product zero, so the request can no longer reach any atom -- the exact analogue, and
    it is one row of one table instead of a main-effect column plus every pair block that
    contains it.

    ⛔ DECLARED, in the same words the incumbent declares: **this is an operation on the
    INTERFACE, not on atoms.**  Charter B.4 wants deletion to be an operation on atoms and
    under a shared-basis read it cannot be -- every well's atoms are fitted over the whole
    pool and no atom belongs to one value.  Row 7 here measures address-book deletion and
    is reported as exactly that.  It is this hub's reported row, never its claimed one.
    """
    import copy

    if obj.get("read") not in ("tensor", "tensor_sum", "tensor_argmax", "tensor_convex"):
        return fw.delete_value_from_head(obj, f, v)
    addr = copy.deepcopy(obj["address"])
    addr.E[int(f)] = np.array(addr.E[int(f)], np.float64, copy=True)
    addr.E[int(f)][int(v), :] = 0.0
    out = dict(obj)
    out["address"] = addr
    out["deleted"] = {"factor_index": int(f), "value_index": int(v),
                      "n_address_rows_zeroed": 1,
                      "store_modes_touched": False,
                      "mechanism": ("the value's row of E_f is zeroed, so the product "
                                    "address for any request naming it is the zero "
                                    "vector and reaches no atom")}
    return out


def lookup_diagnostics(arm: str, out_dir: Path, n_sample: int = 1024,
                       log: Callable = print) -> dict:
    """Charter C.4: is this read a lookup table wearing a landscape?

    For a sample of the scored corners' requests, per well:

    ``d_near``   the distance from the addressed point to its NEAREST atom;
    ``d_atom``   the mean nearest-neighbour distance BETWEEN atoms, the natural unit;
    ``snapfrac`` the share of requests whose addressed point sits closer to its nearest
                 atom than ``snap_tol`` times ``d_atom``.

    ⛔ An arm whose settled state sits on its nearest atom (``d_near -> 0``) is a lookup
    table and is reported as one.  This is computed on the arm that is actually deployed,
    from the requests that are actually scored, and it is printed whether it flatters the
    arm or not.
    """
    out_dir = Path(out_dir)
    with read_dial():
        obj = build_arm(arm, out_dir, log=log)
    if obj.get("not_run") or obj.get("read") in ("proto", "pvm_table", "rival"):
        return {"arm": arm, "not_applicable": "this arm deploys no atoms"}
    cond = fw.load_condition()
    grid = cond["grid"]
    cs = (fw.val_corners(cond) if fw.PINS["condition"] == "val"
          else fw.pair_subcorners(cond) + fw.nway_corners(cond))
    rows = np.concatenate([c["rows"] for c in cs])
    rng = np.random.default_rng(int(fw.PINS["seed"]))
    rows = rows[rng.permutation(rows.size)[:int(n_sample)]]
    vals = grid.values_of(rows)
    X, _ = arm_features(obj, vals, grid)
    store = obj["store"]
    per = []
    for w, G in zip(store.well_channels, store.modes):
        idx = np.asarray(w, np.int64)
        Gm = np.asarray(G, np.float64)
        P = np.asarray(X, np.float64)[:, idx]
        d2 = (np.sum(P * P, 1)[:, None] - 2.0 * (P @ Gm.T)
              + np.sum(Gm * Gm, 1)[None, :])
        d_near = np.sqrt(np.maximum(d2.min(1), 0.0))
        gg = (np.sum(Gm * Gm, 1)[:, None] - 2.0 * (Gm @ Gm.T)
              + np.sum(Gm * Gm, 1)[None, :])
        np.fill_diagonal(gg, np.inf)
        d_atom = float(np.sqrt(np.maximum(gg.min(1), 0.0)).mean())
        per.append({"d_near_mean": float(d_near.mean()),
                    "d_near_over_d_atom": float(d_near.mean() / max(d_atom, 1e-12)),
                    "d_atom_mean_nn": d_atom,
                    "snapfrac_0p10": float((d_near < 0.10 * d_atom).mean()),
                    "snapfrac_0p25": float((d_near < 0.25 * d_atom).mean())})
    out = {"arm": arm, "n_sample": int(rows.size), "per_well": per,
           "d_near_over_d_atom_mean": float(np.mean([q["d_near_over_d_atom"]
                                                     for q in per])),
           "snapfrac_0p25_mean": float(np.mean([q["snapfrac_0p25"] for q in per])),
           "rule": ("charter C.4 -- d_near/d_atom -> 0 means the read settled ON an atom "
                    "and the arm is a lookup table; it is reported as one")}
    fw._jdump(out, out_dir / "arms" / arm / "lookup_check.json")
    log(f"  {arm}: d_near/d_atom = {out['d_near_over_d_atom_mean']:.4f}, "
        f"snapfrac(0.25) = {out['snapfrac_0p25_mean']:.4f}")
    return out


# ---------------------------------------------------------------------------
# 7. Stages
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
        obj = build_arm(arm, out_dir, log=log)
        if obj.get("not_run"):
            # ⛔ w4-common section 6: a declared NOT-RUN is never a null.  It gets a dash
            # and this footnote on every row, and it is never counted as a win for us.
            return {"arm": arm, "not_run": obj["not_run"], "incomplete": False,
                    "n_done": 0}
        return fw.run_arm(arm, out_dir, max_seconds=max_seconds, log=log)


def _cache_manifest(seed: int) -> dict:
    """The feature cache this leg will actually read, with its declared hash.

    ⛔ w4-common section 2: every scored run is on cache GENERATION 2 and a generation-1
    number is never quoted beside one.  This copies the cache's own
    ``feature_cache.json`` into the leg's pins so the generation a number came from is a
    property of the artifact and not of a memory.
    """
    p = fw.bridge_seed_dir(int(seed)) / "feature_cache.json"
    if not p.exists():
        return {"missing": str(p)}
    with open(p, encoding="utf-8") as fh:
        m = json.load(fh)
    return {k: m.get(k) for k in ("dataset", "dtype", "n_rows", "scope",
                                  "features_sha256", "rows_sha256",
                                  "encoder_param_hash")}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("prepare", "fit", "run", "table", "pins",
                            "lookupcheck"))
    p.add_argument("--arm", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--max-seconds", type=float, default=3600.0)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--sweeps", type=int, default=None)
    p.add_argument("--ridge", type=float, default=None)
    p.add_argument("--primary", default="tensor_addr")
    p.add_argument("--margin", type=float, default=1.0)
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
    _inst = bind_instrument(log=lambda m: print(m, flush=True))
    base = Path(a.out)
    out = out_leg(base, a.seed, a.condition)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"stage={a.stage} arm={a.arm} out={out} seed={a.seed} condition={a.condition} "
        f"rank={READ_PINS['rank']} sweeps={READ_PINS['sweeps']}")
    if a.stage == "pins":
        fw._jdump({"read_pins": READ_PINS, "fw_pins": fw.PINS,
                   "instrument": _inst,
                   "bridge_root": os.environ.get("CHLU_GEN_BRIDGE_ROOT"),
                   "feature_cache": _cache_manifest(int(a.seed))}, out / "pins.json")
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
    if a.stage == "lookupcheck":
        log(json.dumps(lookup_diagnostics(a.arm, out, log=log), default=str))
        return 0
    if a.stage == "table":
        with read_dial():
            t = build_table(out, primary=a.primary, margin=a.margin, log=log)
        for name, blk in t["arms"].items():
            r4 = blk.get("r4_comp_nway", {})
            log(f"    {name:22s} {blk['class']:34s} "
                f"r4={r4.get('novel_tuples_per_corner', float('nan')):7.3f} "
                f"judge={r4.get('judge_accuracy', float('nan')):.4f}")
        return 0
    raise SystemExit(f"stage {a.stage} not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
