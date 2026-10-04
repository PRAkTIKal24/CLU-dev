"""exp gen read."""

from __future__ import annotations
from nclu.paths import run_dir
import contextlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence
import numpy as np
from nclu.experiments import factor_wells as fw

_REFUSALS: dict = {}
_FW_BUILD_ARM = fw.build_arm
_FW_ARM_FEATURES = fw.arm_features
from nclu.core.gen_read import (
    TensorAddress,
    assert_read_request_only,
    render_span,
    span_basis,
    span_coefficients,
)

READ_PINS = {
    "partition": "balanced",
    "rank": 128,
    "ridge": 0.01,
    "sweeps": 40,
    "address_seed": 0,
    "n_addr_train": 131072,
    "addr_chunk": 8192,
    "standardise_targets": False,
}
ARMS = (
    "span_head",
    "tensor_addr",
    "tensor_addr_sum",
    "pvm_span",
    "pvm_table",
    "oracle_span",
    "landscape_removed",
    "landscape_shuffled",
    "random_span_refit",
    "span_rotated",
    "lookup_argmax",
    "lookup_convex",
    "corr_linear",
    "knn_proto",
) + tuple(
    (
        f"rival_{n}"
        for n in ("table", "mlp", "mlp_180k", "cont_mlp", "transformer", "lstm", "lace", "poe_ebm")
    )
)
RIVAL_NAMES = ("table", "mlp", "mlp_180k", "cont_mlp", "transformer", "lstm", "lace", "poe_ebm")
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
ARM_STORE.update({f"rival_{n}": None for n in RIVAL_NAMES})
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
ARM_READ.update({f"rival_{n}": "rival" for n in RIVAL_NAMES})
NOT_ARMS = ("oracle_span",)


def _now() -> float:
    return time.time()


def set_condition(seed: int, condition: str) -> dict:
    if str(condition) not in ("val", "test"):
        raise ValueError(f"condition must be val or test, got {condition!r}")
    fw.PINS["seed"] = int(seed)
    fw.PINS["condition"] = str(condition)
    return {"seed": int(seed), "condition": str(condition)}


def out_leg(base: Path, seed: int, condition: str) -> Path:
    return Path(base) / f"{condition}_seed{int(seed)}"


def bind_instrument(log: Callable = print) -> dict:
    return {"classifier": "weights/ (hash-checked by Judge)"}


class _FitFingerprint:
    def __init__(self, protos):
        self.protos = tuple(protos)


class PerValueMeanTable:
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
                sums = [np.zeros((int(c), Xc.shape[1]), np.float64) for c in self.cardinalities]
            tot += Xc.sum(0)
            for f in range(Yi.shape[1]):
                np.add.at(sums[f], Yi[s:e, f], Xc)
                cnts[f] += np.bincount(Yi[s:e, f], minlength=int(self.cardinalities[f]))
            n += Xc.shape[0]
            s = e
        self.mu = tot / max(1, n)
        self.dev = [
            np.where(
                cnts[f][:, None] > 0,
                sums[f] / np.maximum(cnts[f][:, None], 1) - self.mu[None, :],
                0.0,
            )
            for f in range(len(self.cardinalities))
        ]
        self.n_fit_rows = int(n)
        return self

    def read(self, values: np.ndarray, grid_values) -> tuple:
        """``(N, F) requested VALUES -> ((N, D), out-of-range mask)``, request-only."""
        from nclu.core.gen_request_head import one_hot_interpolated

        V = np.asarray(values, np.float64)
        out = np.repeat(self.mu[None, :], V.shape[0], 0).copy()
        oor = np.zeros(V.shape[0], bool)
        for f in range(V.shape[1]):
            Wf, of = one_hot_interpolated(V[:, f], grid_values[f])
            oor |= of
            out += np.asarray(Wf, np.float64) @ self.dev[f]
        return (out.astype(np.float32), oor)

    def byte_footprint(self) -> dict:
        b = int(self.mu.size + sum((d.size for d in self.dev))) * 4
        return {
            "deployed_bytes": b,
            "what": "the global mean plus one per-(factor,value) mean DEVIATION, full D",
            "note": "no store, no atoms -- this row is the specialist, not an arm of ours",
        }


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
    log(f"  span targets: {n} rows -> " + " ".join((str(p.shape) for p in parts)))
    return (grid.factor_indices_of(pool), parts)


PRE_FIT_PERTURB = {"random_span_refit": "random_span", "span_rotated": "rotate"}
POST_FIT_PERTURB = {"landscape_removed": "zero", "landscape_shuffled": "value_permute"}


def perturb_store(store, kind: str, seed: int):
    import jax

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
    return type(store)(
        store.well_channels,
        int(fw.PINS["n_modes"]),
        store.feat_shape,
        jax.random.PRNGKey(0),
        init=out,
    )


def build_arm(arm: str, out_dir: Path, log: Callable = print) -> dict:
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
    if arm.startswith("rival_"):
        name = arm[len("rival_") :]
        from nclu.eval.baselines.api import load_rival, available

        _rroot = os.environ.get("NCLU_BASELINES_ROOT")
        _rroot = Path(_rroot) if _rroot else None
        try:
            r = load_rival(int(fw.PINS["seed"]), name, root=_rroot)
            if _rroot is not None:
                _want = fw.feature_cache_manifest(int(fw.PINS["seed"])).get("features_sha256")
                _got = (getattr(r, "meta", {}) or {}).get("features_sha256")
                if _want and _got != _want:
                    raise AssertionError(
                        f"{arm}: rival fit on features {_got} but this leg scores features {_want} -- a rival from another feature space is refused"
                    )
        except AssertionError:
            raise
        except Exception as exc:
            obj = {
                "arm": arm,
                "read": "rival",
                "not_run": f"{type(exc).__name__}: {exc}",
                "rival": None,
                "byte_footprint": {"deployed_bytes": 0, "what": "DECLARED NOT-RUN"},
                "address_bytes": {"address_bytes": 0},
                "head_bytes": {"head_bytes": 0},
            }
            arm_dir.mkdir(parents=True, exist_ok=True)
            fw._jdump({k: v for k, v in obj.items() if k != "rival"}, arm_dir / "NOT_RUN.json")
            log(f"  arm {arm}: DECLARED NOT-RUN -- {obj['not_run']}")
            return obj
        r_partial = None
        try:
            r_partial = load_rival(int(fw.PINS["seed"]), f"{name}_dropout", root=_rroot)
        except Exception:
            r_partial = None
        names = list(getattr(r, "names", grid.names))
        if names != [str(x) for x in grid.names]:
            raise AssertionError(
                f"{arm}: rival factor order {names} != the grid's {[str(x) for x in grid.names]} -- a silent reorder would score the rival on a request it never received"
            )
        meta = {}
        root = Path(
            os.environ.get("NCLU_BASELINES_ROOT", str(run_dir("shapes3d/baseline_suite/rivals")))
        )
        for cand in (f"{name}.rival.json", f"{name}.json"):
            mp = root / f"seed_{int(fw.PINS['seed'])}" / cand
            if mp.exists():
                with open(mp, encoding="utf-8") as fh:
                    meta = json.load(fh)
                break
        if not meta.get("param_count"):
            try:
                meta["param_count"] = (
                    (available(int(fw.PINS["seed"])) or {}).get(name, {}).get("param_count")
                )
            except Exception:
                pass
        obj = {
            "arm": arm,
            "read": "rival",
            "kind": "rival",
            "rival": r,
            "rival_partial": r_partial,
            "partial_policy": "the _dropout variant answers a request with an unrequested factor"
            if r_partial is not None
            else "NOT ANSWERABLE: no _dropout variant is fitted, so rows 1/2/5/6 for this rival are a DECLARED NOT-RUN",
            "partial_answerable": bool(r_partial is not None),
            "protos": _FitFingerprint(()),
            "fingerprint_note": "a rival's parameters are not exposed by the earlier baseline set's callable, so its row-7 fit fingerprint is empty and proves nothing; the rival has no deletion operation and its row 7 is a dash",
            "byte_footprint": {
                "deployed_bytes": int(meta.get("byte_footprint", {}).get("bytes", 0)),
                "what": f"the earlier baseline set's fitted {name}, generation 2",
                "param_count": meta.get("param_count"),
            },
            "address_bytes": {
                "address_bytes": 0,
                "note": "a rival's parameters ARE its address book",
            },
            "head_bytes": {"head_bytes": 0, "note": "a rival has no separable addressing head"},
            "rival_meta": {
                k: v
                for k, v in meta.items()
                if k
                in (
                    "param_count",
                    "byte_footprint",
                    "offgrid_capable",
                    "val_mse_full_schedule",
                    "val_mse_partial_requests",
                    "n_rows_fit",
                )
            },
            "fit_seconds": 0.0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        fw._jdump(
            {k: v for k, v in obj.items() if k not in ("rival", "rival_partial", "protos")},
            arm_dir / "arm.json",
        )
        log(
            f"  arm {arm}: loaded the earlier baseline set rival '{name}' (params={obj['byte_footprint'].get('param_count')})"
        )
        return obj
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
            Y, (fw.gather(F, pool[s : s + chunk]) for s in range(0, pool.size, chunk))
        )
        obj = {
            "arm": arm,
            "read": "pvm_table",
            "kind": "pvm_table",
            "table": tbl,
            "protos": _FitFingerprint([tbl.mu] + list(tbl.dev)),
            "byte_footprint": tbl.byte_footprint(),
            "address_bytes": {"address_bytes": 0, "note": "the table IS its own address book"},
            "head_bytes": {"head_bytes": 0, "note": "the table IS its own address book"},
            "fit_seconds": _now() - t0,
            "prov": {"n_rows_fit": tbl.n_fit_rows},
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        fw._jdump({k: v for k, v in obj.items() if k != "table"}, arm_dir / "arm.json")
        log(
            f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; {tbl.byte_footprint()['deployed_bytes']:,} B, no store"
        )
        return obj
    src = {"balanced": "factor_aligned_balanced", "ward": "basis_head"}[store_name]
    base = _FW_BUILD_ARM(src, out_dir, log=log)
    store, heads, spec = (base["store"], base["heads"], base["spec"])
    if arm in PRE_FIT_PERTURB:
        store = perturb_store(store, PRE_FIT_PERTURB[arm], int(fw.PINS["seed"]))
    basis = span_basis(store)
    obj = {
        "arm": arm,
        "read": verb,
        "kind": "span",
        "store": store,
        "heads": heads,
        "spec": spec,
        "basis": basis,
        "partition": base.get("partition"),
        "byte_footprint": base["byte_footprint"],
        "store_footprint": base.get("store_footprint"),
        "head_bytes": base["head_bytes"],
        "labels": base.get("labels"),
        "recon_mse": base.get("recon_mse"),
    }
    if verb in ("tensor", "tensor_sum", "tensor_argmax", "tensor_convex"):
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        pool = pool[: int(READ_PINS["n_addr_train"])]
        tag = PRE_FIT_PERTURB.get(arm)
        cache = out_dir / (
            f"address_{store_name}_{verb}_R{int(READ_PINS['rank'])}_lam{READ_PINS['ridge']}_sw{int(READ_PINS['sweeps'])}_n{int(READ_PINS['n_addr_train'])}"
            + (f"_{tag}" if tag else "")
            + ".pkl"
        )
        if cache.exists():
            with open(cache, "rb") as fh:
                addr = pickle.load(fh)
            log(f"  address book: reused {cache.name}")
        else:
            Y, T = _pool_span_targets(store, pool, basis, log=log)
            ntr = int(0.9 * Y.shape[0])
            addr = TensorAddress(
                cards,
                rank=int(READ_PINS["rank"]),
                ridge=float(READ_PINS["ridge"]),
                seed=int(READ_PINS["address_seed"]),
            )
            addr.multiplicative = verb != "tensor_sum"
            addr.fit(
                Y[:ntr],
                [t[:ntr] for t in T],
                sweeps=int(READ_PINS["sweeps"]),
                log=log,
                Y_val=Y[ntr:],
                targets_val=[t[ntr:] for t in T],
            )
            with open(cache, "wb") as fh:
                pickle.dump(addr, fh)
        obj["address"] = addr
        obj["address_bytes"] = addr.byte_footprint()
        vals0 = grid.values_of(pool[:64])
        obj["request_only"] = assert_read_request_only(
            addr, store, vals0, grid.values, basis, rng_seed=int(fw.PINS["seed"])
        )
    elif verb == "pvm_span":
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        Y = grid.factor_indices_of(pool)
        chunk = int(READ_PINS["addr_chunk"])
        tbl = PerValueMeanTable(cards).fit(
            Y, (fw.gather(F, pool[s : s + chunk]) for s in range(0, pool.size, chunk))
        )
        obj["table"] = tbl
        obj["address_bytes"] = {
            "address_bytes": tbl.byte_footprint()["deployed_bytes"],
            "note": "the per-value-mean table, used as the address and then projected into span(G_k) -- the target file's named first mechanism",
        }
    else:
        hb = dict(base["head_bytes"])
        obj["address_bytes"] = {
            "address_bytes": int(hb.get("head_bytes", 0)),
            "per_well": hb.get("head_bytes_per_well"),
            "dtype": hb.get("dtype"),
            "bytes_formula": hb.get("bytes_formula"),
            "note": "the incumbent ridge head -- this read's address book",
        }
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
    fw._jdump(
        {
            k: v
            for k, v in obj.items()
            if k not in ("store", "heads", "spec", "labels", "basis", "address", "table", "protos")
        },
        arm_dir / "arm.json",
    )
    log(
        f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; store {obj['byte_footprint']['deployed_bytes']:,} B + address {obj['address_bytes'].get('address_bytes', 0):,} B"
    )
    return obj


def grid_rows_of(values: np.ndarray, grid) -> tuple:
    """(N, F) requested VALUES -> the dataset row realising them, nearest on the grid.

    Used ONLY by the ``oracle`` non-arm.  shapes3d is a full factorial, so a tuple of
    factor indices IS a row id; an unrequested or off-grid value takes the nearest grid
    index and the row is flagged, which is the most generous reading available to the
    ceiling and therefore the conservative direction for every arm measured against it.
    """
    V = np.asarray(values, np.float64)
    cards = tuple((int(c) for c in grid.cardinalities))
    idx = np.zeros((V.shape[0], len(cards)), np.int64)
    oor = np.zeros(V.shape[0], bool)
    for f in range(len(cards)):
        gv = np.asarray(grid.values[f], np.float64)
        v = V[:, f]
        asked = ~np.isnan(v)
        vv = np.where(asked, v, gv[0])
        idx[:, f] = np.argmin(np.abs(vv[:, None] - gv[None, :]), axis=1)
        oor |= asked & ((v < gv[0]) | (v > gv[-1]))
    return (np.ravel_multi_index(tuple(idx.T), cards), oor)


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    verb = obj.get("read", "head")
    if verb == "rival":
        if obj.get("rival") is None:
            raise AssertionError(f"{obj['arm']} is a DECLARED NOT-RUN and has no read")
        V = np.asarray(values, np.float64)
        part = np.isnan(V).any(axis=1)
        out = np.zeros((V.shape[0], int(fw.PINS["D"])), np.float32)
        oor = np.zeros(V.shape[0], bool)
        _REFUSALS.setdefault(obj["arm"], {})

        def _ask(fn, rows, tag):
            if fn is None:
                _REFUSALS[obj["arm"]][tag] = "no such variant is fitted"
                oor[rows] = True
                return
            try:
                f, o = fn(V[rows])
            except Exception as exc:
                _REFUSALS[obj["arm"]][tag] = f"{type(exc).__name__}: {str(exc)[:220]}"
                oor[rows] = True
                return
            out[rows] = np.asarray(f, np.float32)
            oor[rows] = np.asarray(o, bool)

        if (~part).any():
            _ask(obj["rival"], ~part, "full_request")
        if part.any():
            _ask(obj.get("rival_partial"), part, "partial_request")
        return (out, oor)
    if verb == "proto":
        return _FW_ARM_FEATURES(obj, values, grid)
    if verb == "pvm_table":
        return obj["table"].read(values, grid.values)
    store, basis = (obj["store"], obj.get("basis"))
    if obj.get("zeroed"):
        return (
            np.zeros((np.asarray(values).shape[0], int(store.dim)), np.float32),
            np.zeros(np.asarray(values).shape[0], bool),
        )
    if verb == "head":
        from nclu.core.gen_request_head import head_read, request_design

        Phi, oor = request_design(values, grid.values, obj["spec"])
        return (head_read(obj["heads"], store, Phi), oor)
    if verb == "pvm_span":
        X, oor = obj["table"].read(values, grid.values)
        return (render_span(store, span_coefficients(store, X, basis), basis), oor)
    if verb == "oracle":
        rows, oor = grid_rows_of(values, grid)
        F = fw.features_mmap(int(fw.PINS["seed"]))
        X = fw.gather(F, rows)
        return (render_span(store, span_coefficients(store, X, basis), basis), oor)
    addr = obj["address"]
    if verb in ("tensor", "tensor_sum"):
        return addr.read(store, values, grid.values, basis)
    coeffs, oor = addr.coefficients(values, grid.values)
    out = np.zeros((np.asarray(values).shape[0], int(store.dim)), np.float32)
    for c, w, G, Uk in zip(coeffs, store.well_channels, store.modes, basis):
        Gm = np.asarray(G, np.float32)
        pts = np.asarray(c, np.float32) @ np.asarray(Uk, np.float32).T
        d2 = np.sum(pts * pts, 1)[:, None] - 2.0 * (pts @ Gm.T) + np.sum(Gm * Gm, 1)[None, :]
        if verb == "tensor_argmax":
            out[:, np.asarray(w, np.int64)] = Gm[np.argmin(d2, axis=1)]
        else:
            p = np.maximum(-d2 + d2.max(1, keepdims=True), 0.0)
            s = p.sum(1, keepdims=True)
            p = np.where(s > 0, p / np.where(s > 0, s, 1.0), 1.0 / p.shape[1])
            out[:, np.asarray(w, np.int64)] = p @ Gm
    return (out, oor)


@contextlib.contextmanager
def read_dial():
    fw_build, fw_read, fw_arms = (fw.build_arm, fw.arm_features, fw.ARMS)
    fw_del = fw.delete_value_from_head

    def _build(arm, out_dir, log=print):
        if arm in ("factor_aligned_balanced", "basis_head", "knn_proto"):
            return fw_build(arm, out_dir, log=log)
        return build_arm(arm, out_dir, log=log)

    fw.build_arm = _build
    fw.arm_features = arm_features
    fw.ARMS = ARMS
    fw.delete_value_from_head = (
        lambda o, f, v: delete_value_from_address(o, f, v)
        if o.get("read", "").startswith("tensor")
        else fw_del(o, f, v)
    )
    try:
        yield {"replaced": ["build_arm", "arm_features", "ARMS", "delete_value_from_head"]}
    finally:
        if fw.arm_features is not arm_features:
            raise AssertionError(
                "arm_features was replaced again inside the dial -- the read is not the only thing that moved"
            )
        fw.build_arm, fw.arm_features, fw.ARMS = (fw_build, fw_read, fw_arms)
        fw.delete_value_from_head = fw_del


ARM_CLASS = {
    "tensor_addr": "ARM",
    "oracle_span": "NOT AN ARM (ceiling)",
    "span_head": "ABLATION (the deployed read)",
    "corr_linear": "ABLATION (correlation split + linear head)",
    "tensor_addr_sum": "ABLATION (same rank, address summed not multiplied)",
    "pvm_span": "ABLATION (per-value-mean address, projected into the span)",
    "pvm_table": "SPECIALIST (the plain per-value-mean table, no store)",
    "knn_proto": "SPECIALIST (prototype table, rebuilt here)",
    "landscape_removed": "CONTROL (protocol C.2, atoms zeroed)",
    "landscape_shuffled": "CONTROL (protocol C.2, atom values permuted)",
    "random_span_refit": "CONTROL (random atoms, address book refit)",
    "span_rotated": "CONTROL (predicted NO-OP: the span is unchanged)",
    "lookup_argmax": "FLOOR (protocol C.1/C.4, matched-bytes lookup)",
    "lookup_convex": "FLOOR (protocol C.1, the mixture)",
}
ARM_CLASS.update({f"rival_{n}": "RIVAL" for n in RIVAL_NAMES})


def _corner_rows(arm_dir: Path, corners, grid, seen_mask) -> list:
    z = np.load(arm_dir / "rows.npz", allow_pickle=True)
    return fw._per_corner(corners, z["corners"].item(), grid, seen_mask)


def build_table(
    out_dir: Path, primary: str = "tensor_addr", margin: float = 1.0, log: Callable = print
) -> dict:
    out_dir = Path(out_dir)
    cond = fw.load_condition()
    grid = cond["grid"]
    if fw.PINS["condition"] == "val":
        fitted = np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond))
    else:
        fitted = cond["train_rows"]
    seen_mask = fw.seen_tuple_mask(grid, fitted)
    if fw.PINS["condition"] == "val":
        n4, n3 = (fw.val_corners(cond), [])
    else:
        n4 = fw.nway_corners(cond)
        n3 = fw.pair_subcorners(cond)
    have, not_run = ([], {})
    for a in ARMS:
        d = out_dir / "arms" / a
        if (d / "rows.npz").exists():
            have.append(a)
        elif (d / "NOT_RUN.json").exists():
            with open(d / "NOT_RUN.json", encoding="utf-8") as fh:
                not_run[a] = json.load(fh).get("not_run")
    log(f"  arms with rows: {have}")
    log(f"  declared NOT-RUN: {list(not_run)}")
    cards = tuple((int(c) for c in grid.cardinalities))
    ceiling = {}
    for c in n3 + n4:
        Y = grid.factor_indices_of(c["rows"])
        ceiling[c["corner_id"]] = int(np.prod([len(np.unique(Y[:, f])) for f in range(len(cards))]))
    per_corner, summary = ({}, {})
    for a in have:
        rows = _corner_rows(out_dir / "arms" / a, n3 + n4, grid, seen_mask)
        by_id = {r["corner_id"]: r for r in rows}
        per_corner[a] = by_id
        with open(out_dir / "arms" / a / "arm.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        blk = {
            "class": ARM_CLASS.get(a, "?"),
            "store_bytes": meta.get("byte_footprint", {}).get("deployed_bytes"),
            "address_bytes": (meta.get("address_bytes") or {}).get("address_bytes"),
            "param_count": meta.get("byte_footprint", {}).get("param_count"),
        }
        for row_id, cs in (("r3_comp_2way", n3), ("r4_comp_nway", n4)):
            v = np.array([by_id[c["corner_id"]]["distinct_valid_novel_tuples"] for c in cs], float)
            j = np.array([by_id[c["corner_id"]]["judge_accuracy"] for c in cs], float)
            if not v.size:
                blk[row_id] = {"n_corners": 0, "note": "not scored under this condition"}
                continue
            cap = np.array([ceiling[c["corner_id"]] for c in cs], float)
            blk[row_id] = {
                "n_corners": int(v.size),
                "novel_tuples_per_corner": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "judge_accuracy": float(j.mean()),
                "ceiling_per_corner": float(cap.mean()),
                "frac_of_ceiling": float((v / cap).mean()),
            }
        rp = out_dir / "arms" / a / "refusals.json"
        if rp.exists():
            with open(rp, encoding="utf-8") as fh:
                blk["rival_refusals"] = json.load(fh)
            if "full_request" in blk["rival_refusals"]:
                blk["query_set_rows"] = (
                    "DECLARED NOT-RUN on the rows this rival declined -- its own reason is in rival_refusals. A dash, never a null."
                )
        if meta.get("partial_answerable") is False:
            blk["query_set_rows"] = (
                "DECLARED NOT-RUN: this rival has no variant fitted for a request with an unrequested factor, and rows 1/2/5/6 carry them. A dash, not a null."
            )
        z = np.load(out_dir / "arms" / a / "rows.npz", allow_pickle=True)
        if "qrows" in z.files and meta.get("partial_answerable") is not False:
            q = z["qrows"].item()
            for row_id in ("r1_seen", "r2_unseen_valid", "r5_comp_refusal", "r6_invalid_refusal"):
                if row_id in q:
                    blk[row_id] = {
                        "accept": float(np.asarray(q[row_id]["accept"], bool).mean()),
                        "valid": float(np.asarray(q[row_id]["valid"], bool).mean()),
                    }
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
                    fw.paired_estimand(
                        A, Bc, "distinct_valid_novel_tuples", float(margin), primary, other
                    ),
                    comparison_class=ARM_CLASS.get(other, "?"),
                    verdict_rule="a win over a RIVAL is a RESULT; a win over anything else is an ABLATION (common section 1)",
                )
    table = {
        "primary": primary,
        "margin_declared": float(margin),
        "condition": fw.PINS["condition"],
        "seed": int(fw.PINS["seed"]),
        "arms": summary,
        "not_run": not_run,
        "estimands": estimands,
        "corner_counts": {"r3_comp_2way": len(n3), "r4_comp_nway": len(n4)},
        "corner_ceilings": {
            "r3_comp_2way": float(np.mean([ceiling[c["corner_id"]] for c in n3])) if n3 else None,
            "r4_comp_nway": float(np.mean([ceiling[c["corner_id"]] for c in n4])) if n4 else None,
            "note": "distinct tuples a corner CAN yield = the product of the distinct values its rows realise per factor; row 3 is 36 and row 4 is 72, so the two rows are not on the same scale",
        },
        "labels_note": "labels are used ONCE at fit time to align wells and at the address book, and nowhere else (protocol B.6, common 5)",
        "pins": {
            "read": READ_PINS,
            "fw": {k: list(v) if isinstance(v, tuple) else v for k, v in fw.PINS.items()},
        },
    }
    fw._jdump(table, out_dir / "READ-TABLE.json")
    fw._jdump(per_corner, out_dir / "READ-PER-CORNER.json")
    log(f"  table written -> {out_dir / 'READ-TABLE.json'}")
    return table


def delete_value_from_address(obj: dict, f: int, v: int) -> dict:
    import copy

    if obj.get("read") not in ("tensor", "tensor_sum", "tensor_argmax", "tensor_convex"):
        return fw.delete_value_from_head(obj, f, v)
    addr = copy.deepcopy(obj["address"])
    addr.E[int(f)] = np.array(addr.E[int(f)], np.float64, copy=True)
    addr.E[int(f)][int(v), :] = 0.0
    out = dict(obj)
    out["address"] = addr
    out["deleted"] = {
        "factor_index": int(f),
        "value_index": int(v),
        "n_address_rows_zeroed": 1,
        "store_modes_touched": False,
        "mechanism": "the value's row of E_f is zeroed, so the product address for any request naming it is the zero vector and reaches no atom",
    }
    return out


def stage_prepare(out_dir: Path, log: Callable = print) -> dict:
    info = fw.build_partitions(out_dir, log=log)
    return {"well_sizes": info["info"]["well_sizes"][READ_PINS["partition"]]}


def stage_fit(arm: str, out_dir: Path, log: Callable = print) -> dict:
    with read_dial():
        obj = build_arm(arm, out_dir, log=log)
    return {
        "arm": arm,
        "address_bytes": obj.get("address_bytes"),
        "byte_footprint": obj.get("byte_footprint"),
    }


def stage_run(arm: str, out_dir: Path, max_seconds: float, log: Callable = print) -> dict:
    done = Path(out_dir) / "arms" / arm / "rows.npz"
    if done.exists():
        log(f"  arm {arm}: rows.npz already on disk, skipping")
        return {"arm": arm, "incomplete": False, "skipped": True}
    with read_dial():
        obj = build_arm(arm, out_dir, log=log)
        if obj.get("not_run"):
            return {"arm": arm, "not_run": obj["not_run"], "incomplete": False, "n_done": 0}
        res = fw.run_arm(arm, out_dir, max_seconds=max_seconds, log=log)
        ref = _REFUSALS.get(arm) or {}
        if ref:
            fw._jdump(ref, Path(out_dir) / "arms" / arm / "refusals.json")
            log(f"  arm {arm}: rival refusals recorded -> {list(ref)}")
        return res


def _cache_manifest(seed: int) -> dict:
    return fw.feature_cache_manifest(int(seed))
