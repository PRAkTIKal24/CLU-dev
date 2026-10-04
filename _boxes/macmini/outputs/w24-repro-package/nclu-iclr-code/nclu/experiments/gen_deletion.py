"""gen deletion."""

from __future__ import annotations
import pickle
import time
import zlib
from pathlib import Path
from typing import Callable
import numpy as np
from nclu.experiments.factor_wells import (
    Judge,
    _jdump,
    accept_rule,
    build_partitions,
    features_mmap,
    feature_cache_manifest,
    fit_pool,
    gather,
    judge_valid,
    load_condition,
    nway_corners,
    pair_subcorners,
    seen_tuple_mask,
    value_presence_bitmap,
)
from nclu.experiments.exp_gen_field_store import PINS as FSPINS, _host_factors, field_spec

PINS = dict(FSPINS)
PINS.update(
    {
        "condition": "value_holdout_scored",
        "seeds": (0, 1, 2),
        "vary_bridge_seed": True,
        "delete_factor": 2,
        "delete_value": 0,
        "hole_pair": ((2, 0), (4, 1)),
        "probe_backgrounds": 256,
        "probe_rng_seed": 20260920,
    }
)
MARGIN_NOVELTY = 1.0
SHAPE_NAMES = {0: "cube", 1: "cylinder", 2: "sphere", 3: "oblong"}
HUE_NAMES = {0: "red", 3: "green", 7: "blue"}


def _now() -> float:
    return time.time()


def assert_generation2(seed: int) -> dict:
    """Refuse to run on a missing or truncated feature cache for ``seed``."""
    man = feature_cache_manifest(int(seed))
    if not man["present"]:
        raise AssertionError(
            f"seed {seed}: no feature cache at {man['file']}. Set NCLU_FEATURES_ROOT (README, Data)."
        )
    if not man["size_ok"]:
        raise AssertionError(f"seed {seed}: feature cache {man['file']} has the wrong size.")
    return man


def assert_out_root(out: Path) -> Path:
    p = Path(out).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_dir(out: Path, seed: int) -> Path:
    return Path(out) / f"seed_{int(seed)}"


def scored_corners(cond) -> dict:
    grid = cond["grid"]
    f, v = (int(PINS["delete_factor"]), int(PINS["delete_value"]))
    dval = float(grid.values[f][v])
    n4, n3 = (nway_corners(cond), pair_subcorners(cond))
    allc = n4 + n3
    deleted, survivor = ([], [])
    for c in allc:
        fi = grid.factor_indices_of(c["rows"])
        holds_red = bool((fi[:, f] == v).any())
        (deleted if holds_red else survivor).append(c)
    return {
        "all": allc,
        "deleted": deleted,
        "survivor": survivor,
        "nway": n4,
        "pair": n3,
        "deleted_value": dval,
    }


def assert_scored_and_heldout(cond, cs: dict, pool: np.ndarray) -> dict:
    te = np.asarray(cond["test_rows"], np.int64)
    tr = np.asarray(cond["train_rows"], np.int64)
    P = np.asarray(pool, np.int64)
    rep = {}
    for name in ("all", "deleted", "survivor"):
        rows = np.concatenate([c["rows"] for c in cs[name]])
        uniq = np.unique(rows)
        if uniq.size != rows.size:
            raise AssertionError(f"{name}: corners overlap ({rows.size - uniq.size} shared rows)")
        n_pool = int(np.intersect1d(uniq, P).size)
        if n_pool:
            raise AssertionError(
                f"{name}: {n_pool} corner rows are inside the fit pool -- not a hold-out"
            )
        rep[name] = {
            "n_corners": len(cs[name]),
            "n_requests": int(rows.size),
            "rows_in_scored_test_index": int(np.intersect1d(uniq, te).size),
            "rows_in_train_index": int(np.intersect1d(uniq, tr).size),
            "rows_in_fit_pool": n_pool,
        }
    if rep["all"]["rows_in_scored_test_index"] != rep["all"]["n_requests"]:
        raise AssertionError("not every corner row is in the scored test index")
    if len(cs["survivor"]) < 49:
        raise AssertionError(
            f"the survivor set has {len(cs['survivor'])} corners; common section 2 requires >= 49 paired corners and forbids a single cell deciding a target"
        )
    rep["statement"] = (
        f"every corner row is in the value-holdout TEST index and none is in any arm's fit pool; the survivor half alone is {len(cs['survivor'])} corners, which meets the >= 49 paired-corner estimand; this run SPENDS run 1/3 of target `deletion`"
    )
    return rep


def conjunction_probes(cond, seed: int) -> dict:
    grid = cond["grid"]
    rng = np.random.default_rng(int(PINS["probe_rng_seed"]) + 101 * int(seed))
    te = np.asarray(cond["test_rows"], np.int64)
    n = int(PINS["probe_backgrounds"])
    bg_rows = te[rng.permutation(te.size)[:n]]
    base = grid.values_of(bg_rows)
    hue_ax, shp_ax = (2, 4)

    def _set(oh: int, sh: int) -> np.ndarray:
        v = np.array(base, np.float64, copy=True)
        v[:, hue_ax] = float(grid.values[hue_ax][oh])
        v[:, shp_ax] = float(grid.values[shp_ax][sh])
        return v

    spec = {
        "red_cylinder": (0, 1, False, "THE HOLE; never seen"),
        "red_sphere": (0, 2, True, "red's other shape, SEEN -- the hole must leave it allowed"),
        "red_oblong": (0, 3, True, "red's other shape, SEEN -- the hole must leave it allowed"),
        "green_cylinder": (3, 1, True, "cylinder's other hue, SEEN -- must stay allowed"),
        "blue_cylinder": (7, 1, True, "cylinder's other hue, SEEN -- must stay allowed"),
        "green_sphere": (3, 2, False, "pair-001: NEVER SEEN, not holed -- must stay allowed"),
        "blue_oblong": (7, 3, False, "pair-002: NEVER SEEN, not holed -- must stay allowed"),
        "red_cube": (0, 0, False, "shape=cube is held out entirely; reported, not a bar"),
    }
    out = {}
    for name, (oh, sh, seen, why) in spec.items():
        out[name] = {
            "values": _set(oh, sh),
            "hue_index": oh,
            "shape_index": sh,
            "seen_in_pool": bool(seen),
            "why": why,
            "label": f"{HUE_NAMES.get(oh, f'hue{oh}')} AND {SHAPE_NAMES[sh]}",
        }
    out["_backgrounds"] = {
        "rows": bg_rows,
        "n": int(bg_rows.size),
        "source": "the value-holdout TEST index",
    }
    return out


def build_field_base(out: Path, seed: int, log: Callable = print) -> dict:
    from nclu.core.gen_field import assert_read_request_only, build_field, field_byte_footprint

    d = seed_dir(out, seed) / "field"
    done = d / "field.pkl"
    if done.exists():
        log(f"  field(seed {seed}): already built, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)
    cache = assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    parts = build_partitions(Path(out) / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = _host_factors(parts["info"], K)
    pool = scored_pool(cond, seed)
    F = features_mmap(int(seed))
    t0 = _now()
    X = gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    field = build_field(X, Y, pool, labels, hosts, cards, field_spec(seed), log=log)
    del X
    led = field_byte_footprint(field, int(PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the field store deploys {led['deployed_bytes']} B against the pinned {PINS['store_bytes']} B footprint -- the byte footprint has moved"
        )
    ro = assert_read_request_only(field, grid.values_of(pool[:64]), grid.values)
    obj = {
        "kind": "field",
        "seed": int(seed),
        "field": field,
        "pool": pool,
        "byte_footprint": led,
        "request_only": ro,
        "cache": cache,
        "manifest": field.manifest(),
        "fit_seconds": _now() - t0,
    }
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({k: v for k, v in obj.items() if k not in ("field", "pool")}, d / "field.json")
    log(
        f"  field(seed {seed}): built in {obj['fit_seconds']:.1f}s; atoms {led['deployed_bytes']:,} B (image {led['total_image_bytes']:,} B)"
    )
    return obj


def scored_pool(cond, seed: int) -> np.ndarray:
    rng_seed = int(PINS["pool_rng_seed"]) + int(PINS["pool_seed_stride"]) * int(seed)
    return fit_pool(cond["train_rows"], int(PINS["n_pool"]), rng_seed)


def field_variants(base: dict, seed: int, log: Callable = print) -> dict:
    from nclu.core.gen_field import _items_holding, decay_items, evict_term, punch_hole

    f0 = base["field"]
    fac, valx = (int(PINS["delete_factor"]), int(PINS["delete_value"]))
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    out = {"base": f0}
    fe = f0.copy()
    ev = evict_term(fe, fac, valx, log=log)
    out["evict_red"] = fe
    fd = f0.copy()
    ids, _, _ = _items_holding(fd, fac, valx)
    dec_half = decay_items(fd, ids, 0.5, log=log)
    fd_half = fd.copy()
    dec_zero = decay_items(fd, ids, 0.0, log=log)
    out["decay_red"] = fd
    out["_decay_half"] = fd_half
    fh_ = f0.copy()
    ho = punch_hole(fh_, hf, hv, hg, hu, log=log)
    out["hole"] = fh_
    rng = np.random.default_rng(int(seed) * 7919 + 20260920)
    fr = f0.copy()
    fr.view("atoms")[:] = 0.0
    out["landscape_removed"] = fr
    fs = f0.copy()
    for k in range(fs.K):
        blk = fs.atoms_of_well(k)
        blk[:] = blk[:, rng.permutation(blk.shape[1])]
    out["landscape_shuffled"] = fs
    out["_ops"] = {
        "evict": {k: v for k, v in ev.items() if k != "permitted_ranges"},
        "decay_half": {k: v for k, v in dec_half.items() if k != "permitted_ranges"},
        "decay_zero": {k: v for k, v in dec_zero.items() if k != "permitted_ranges"},
        "hole": {k: v for k, v in ho.items() if k != "permitted_ranges"},
    }
    return out


def build_tables(out: Path, seed: int, log: Callable = print) -> dict:
    from nclu.core.gen_request import FactorPrototypes

    d = seed_dir(out, seed) / "tables"
    done = d / "tables.pkl"
    if done.exists():
        log(f"  tables(seed {seed}): already built, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)
    assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    pool = scored_pool(cond, seed)
    fac, valx = (int(PINS["delete_factor"]), int(PINS["delete_value"]))
    F = features_mmap(int(seed))
    X = gather(F, pool)
    t0 = _now()
    base = FactorPrototypes.fit(grid, X, pool)
    cost_fit = _now() - t0
    t0 = _now()
    protos = [np.array(P, np.float32, copy=True) for P in base.protos]
    protos[fac][valx] = base.mu_global
    npv = [list(c) for c in base.n_per_value]
    npv[fac][valx] = 0
    removed = FactorPrototypes(
        names=base.names,
        values=base.values,
        protos=tuple(protos),
        mu_global=base.mu_global,
        n_rows_fit=base.n_rows_fit,
        n_per_value=tuple((tuple(c) for c in npv)),
    )
    cost_remove = _now() - t0
    fi = grid.factor_indices_of(pool)
    keep = fi[:, fac] != valx
    kept_rows = pool[keep]
    t0 = _now()
    retrain = FactorPrototypes.fit(grid, X[keep], kept_rows)
    cost_retrain = _now() - t0
    del X
    obj = {
        "kind": "tables",
        "seed": int(seed),
        "base": base,
        "row_removed": removed,
        "retrain": retrain,
        "cost": {
            "fit_full_pool_s": float(cost_fit),
            "row_removal_s": float(cost_remove),
            "retrain_from_scratch_s": float(cost_retrain),
            "rows_in_pool": int(pool.size),
            "rows_dropped": int((~keep).sum()),
            "rows_refitted_by_retrain": int(kept_rows.size),
            "rows_refitted_by_row_removal": 0,
        },
        "n_bytes": int(base.n_bytes) if hasattr(base, "n_bytes") else None,
    }
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump(
        {
            "seed": int(seed),
            "cost": obj["cost"],
            "n_bytes": obj["n_bytes"],
            "n_per_value_deleted_factor_after_removal": list(removed.n_per_value[fac]),
            "n_per_value_deleted_factor_after_retrain": list(retrain.n_per_value[fac]),
        },
        d / "tables.json",
    )
    log(
        f"  tables(seed {seed}): fit {cost_fit:.2f}s; row-removal {cost_remove:.4f}s; RETRAIN-FROM-SCRATCH {cost_retrain:.2f}s on {kept_rows.size:,} rows ({int((~keep).sum()):,} dropped)"
    )
    return obj


class Gate:
    def __init__(
        self, kind: str, grid, bitmap=None, joint=None, label: str = "", deleted_pairs=None
    ):
        self.kind = str(kind)
        self.grid = grid
        self.bitmap = [np.array(b, bool, copy=True) for b in bitmap or []]
        self.joint = None if joint is None else np.array(joint, bool, copy=True)
        self.deleted = None if deleted_pairs is None else np.array(deleted_pairs, bool, copy=True)
        self.label = label or kind

    @classmethod
    def per_factor(cls, grid, fitted_rows) -> "Gate":
        return cls(
            "per_factor",
            grid,
            bitmap=value_presence_bitmap(grid, fitted_rows),
            label="8-byte per-factor presence bitmap",
        )

    @classmethod
    def joint_hue_shape(cls, grid, fitted_rows) -> "Gate":
        fi = grid.factor_indices_of(np.asarray(fitted_rows, np.int64))
        J = np.zeros((int(grid.cardinalities[2]), int(grid.cardinalities[4])), bool)
        J[fi[:, 2], fi[:, 4]] = True
        return cls(
            "joint",
            grid,
            bitmap=value_presence_bitmap(grid, fitted_rows),
            joint=J,
            label="joint object_hue x shape presence gate",
        )

    @classmethod
    def joint_backoff(cls, grid, fitted_rows) -> "Gate":
        cards = (int(grid.cardinalities[2]), int(grid.cardinalities[4]))
        return cls(
            "joint_backoff",
            grid,
            bitmap=value_presence_bitmap(grid, fitted_rows),
            deleted_pairs=np.zeros(cards, bool),
            label="joint hue x shape gate with marginal back-off + a SEPARATE deleted-pair array",
        )

    def copy(self) -> "Gate":
        return Gate(self.kind, self.grid, self.bitmap, self.joint, self.label, self.deleted)

    def pack(self) -> bytes:
        bits = [np.asarray(b, bool).ravel() for b in self.bitmap]
        if self.joint is not None:
            bits.append(self.joint.ravel())
        if self.deleted is not None:
            bits.append(self.deleted.ravel())
        v = np.concatenate(bits) if bits else np.zeros(0, bool)
        return bytes(np.packbits(v))

    def n_bits(self) -> int:
        n = int(sum((int(np.asarray(b).size) for b in self.bitmap)))
        if self.joint is not None:
            n += int(self.joint.size)
        if self.deleted is not None:
            n += int(self.deleted.size)
        return n

    @property
    def n_bytes(self) -> int:
        return len(self.pack())

    def clear_value(self, f: int, v: int) -> "Gate":
        g = self.copy()
        g.bitmap[int(f)][int(v)] = False
        g.label = self.label + f" [{self.grid.names[f]} index {v} cleared]"
        return g

    def delete_conjunction(self, f: int, v: int, g2: int, u: int) -> "Gate":
        out = self.copy()
        if {int(f), int(g2)} == {2, 4}:
            oh, sh = (int(v), int(u)) if int(f) == 2 else (int(u), int(v))
            if self.kind == "joint":
                out.joint[oh, sh] = False
                out.label = self.label + f" [({oh},{sh}) presence bit cleared]"
                return out
            if self.kind == "joint_backoff":
                out.deleted[oh, sh] = True
                out.label = self.label + f" [({oh},{sh}) marked DELETED]"
                return out
        out.label = self.label + " [conjunction NOT EXPRESSIBLE -- 0 bits changed]"
        return out

    def _pair_indices(self, values: np.ndarray) -> tuple:
        gv_h, gv_s = (self.grid.values[2], self.grid.values[4])
        vh, vs = (values[:, 2], values[:, 4])
        ih = np.clip(np.searchsorted(gv_h, np.nan_to_num(vh), side="left"), 0, gv_h.size - 1)
        isx = np.clip(np.searchsorted(gv_s, np.nan_to_num(vs), side="left"), 0, gv_s.size - 1)
        asked = ~np.isnan(vh) & ~np.isnan(vs)
        return (ih, isx, asked)

    def refuse(self, values: np.ndarray, oor: np.ndarray) -> np.ndarray:
        """True where this gate REFUSES the request."""
        dup = np.zeros(values.shape[0], bool)
        ok = accept_rule(values, oor, dup, self.bitmap, self.grid)
        if self.joint is not None:
            ih, isx, asked = self._pair_indices(values)
            ok &= ~asked | self.joint[ih, isx]
        if self.deleted is not None:
            ih, isx, asked = self._pair_indices(values)
            ok &= ~(asked & self.deleted[ih, isx])
        return ~ok


class TableGate:
    def __init__(self, protos, grid, label: str = "table n_per_value gate"):
        self.protos, self.grid, self.label = (protos, grid, label)

    def refuse(self, values: np.ndarray, oor: np.ndarray) -> np.ndarray:
        bm = [np.asarray(c, np.int64) > 0 for c in self.protos.n_per_value]
        dup = np.zeros(values.shape[0], bool)
        return ~accept_rule(values, oor, dup, bm, self.grid)


def _field_features(field, values, grid):
    from nclu.core.gen_field import field_address, field_read

    addr = field_address(field, values, grid.values)
    return (field_read(field, values, grid.values), addr["out_of_range"])


def _field_lookup_features(field, values, grid, mode: str):
    from nclu.core.gen_field import LIFETIME_EPS, field_address

    addr = field_address(field, values, grid.values)
    mu = addr["mu"]
    N, D = mu.shape
    out = np.zeros((N, D), np.float32)
    life = np.asarray(field.view("atom_life"), np.float64)
    alive = (np.asarray(field.view("atom_alive"), np.uint8) > 0) & (life > LIFETIME_EPS)
    ws = np.asarray(field.view("well_sigma"), np.float64)
    nearest = np.full(N, np.inf)
    for k in range(field.K):
        ch = np.asarray(field.meta["well_channels"][k], np.int64)
        A = np.asarray(field.atoms_of_well(k), np.float64)
        M = np.asarray(mu[:, ch], np.float64)
        if not alive.any():
            continue
        idx = np.flatnonzero(alive)
        Ak = A[idx]
        d2 = (M * M).sum(1)[:, None] - 2.0 * M @ Ak.T + (Ak * Ak).sum(1)[None, :]
        d2 = np.maximum(d2, 0.0)
        nearest = np.minimum(nearest, np.sqrt(d2.min(axis=1)))
        if mode == "argmax":
            out[:, ch] = Ak[np.argmin(d2, axis=1)].astype(np.float32)
        else:
            s = max(float(ws[k]), 1e-09)
            w = np.exp(-d2 / (2.0 * s * s)) * life[idx][None, :]
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
            out[:, ch] = (w @ Ak).astype(np.float32)
    return (out, addr["out_of_range"], nearest)


def build_arms(out: Path, seed: int, log: Callable = print) -> dict:
    cond = load_condition()
    grid = cond["grid"]
    base = build_field_base(out, seed, log=log)
    fv = field_variants(base, seed, log=log)
    tabs = build_tables(out, seed, log=log)
    pool = base["pool"]
    fac, valx = (int(PINS["delete_factor"]), int(PINS["delete_value"]))
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    g8 = Gate.per_factor(grid, pool)
    gj = Gate.joint_hue_shape(grid, pool)
    gb = Gate.joint_backoff(grid, pool)
    tg = TableGate(tabs["base"], grid)
    arms = {}

    def _F(field):
        return lambda vals: _field_features(field, vals, grid)

    def _L(field, mode):
        def fn(vals):
            f, o, nd = _field_lookup_features(field, vals, grid, mode)
            return (f, o)

        return fn

    def _T(protos):
        def fn(vals):
            oor = _table_oor(vals, grid)
            return (protos.embed_many(_reqs(vals, grid)), oor)

        return fn

    def _fr(field):
        from nclu.core.gen_field import gap_read

        return lambda vals: gap_read(field, vals, grid.values)

    arms["field/base"] = dict(
        gen=_F(fv["base"]),
        refuse=_fr(fv["base"]),
        family="field",
        role="pre-deletion",
        deletion=None,
        gen_key="field_base",
    )
    arms["field/evict_red"] = dict(
        gen=_F(fv["evict_red"]),
        refuse=_fr(fv["evict_red"]),
        family="field",
        role="DELETION A: evict object_hue=0.0 (red) -- atoms zeroed, no refit",
        deletion="A",
        gen_key="field_evict",
    )
    arms["field/decay_red"] = dict(
        gen=_F(fv["decay_red"]),
        refuse=_fr(fv["decay_red"]),
        family="field",
        role="DELETION A by LIFETIME: lifetimes decayed to zero, then the item is gone",
        deletion="A",
        gen_key="field_decay",
        identical_to="field_evict",
    )
    arms["field/hole"] = dict(
        gen=_F(fv["hole"]),
        refuse=_fr(fv["hole"]),
        family="field",
        role="DELETION B: the coupling hole on red AND cylinder -- ONE byte, no refit",
        deletion="B",
        gen_key="field_hole",
        identical_to="field_base",
    )
    arms["field/landscape_removed"] = dict(
        gen=_F(fv["landscape_removed"]),
        refuse=_fr(fv["landscape_removed"]),
        family="control",
        role="protocol C.2: atoms zeroed -- must collapse",
        deletion=None,
        gen_key="field_removed",
    )
    arms["field/landscape_shuffled"] = dict(
        gen=_F(fv["landscape_shuffled"]),
        refuse=_fr(fv["landscape_shuffled"]),
        family="control",
        role="protocol C.2: each well's channel axis permuted",
        deletion=None,
        gen_key="field_shuffled",
    )
    arms["field/lookup_argmax"] = dict(
        gen=_L(fv["base"], "argmax"),
        refuse=_fr(fv["base"]),
        family="floor",
        role="protocol C.1: matched-bytes lookup, nearest live atom",
        deletion=None,
        gen_key="field_lookup_argmax",
    )
    arms["field/lookup_mean"] = dict(
        gen=_L(fv["base"], "mean"),
        refuse=_fr(fv["base"]),
        family="floor",
        role="protocol C.1: matched-bytes lookup, kernel-weighted atom mean",
        deletion=None,
        gen_key="field_lookup_mean",
    )
    arms["table/base"] = dict(
        gen=_T(tabs["base"]),
        refuse=tg.refuse,
        family="rival",
        role="the table, pre-deletion",
        deletion=None,
        gen_key="table_base",
    )
    arms["table/row_removed"] = dict(
        gen=_T(tabs["row_removed"]),
        refuse=TableGate(tabs["row_removed"], grid).refuse,
        family="rival",
        role="RIVAL: table row-removal (the value's row is gone)",
        deletion="A",
        gen_key="table_row_removed",
    )
    arms["table/retrain"] = dict(
        gen=_T(tabs["retrain"]),
        refuse=TableGate(tabs["retrain"], grid).refuse,
        family="rival",
        role="RIVAL, GOLD STANDARD: retrain from scratch without the value",
        deletion="A",
        gen_key="table_retrain",
    )
    arms["table+gate8/base"] = dict(
        gen=_T(tabs["base"]),
        refuse=g8.refuse,
        family="gate",
        role="the 8-byte per-factor gate, pre-deletion",
        deletion=None,
        gen_key="table_base",
    )
    arms["table+gate8/clear_red"] = dict(
        gen=_T(tabs["base"]),
        refuse=g8.clear_value(fac, valx).refuse,
        family="gate",
        role="RIVAL: the 8-byte gate deletes the VALUE for 1 bit",
        deletion="A",
        gen_key="table_base",
    )
    arms["table+gate8/hole_attempt"] = dict(
        gen=_T(tabs["base"]),
        refuse=g8.delete_conjunction(hf, hv, hg, hu).refuse,
        family="gate",
        role="RIVAL: the 8-byte gate asked for the CONJUNCTION -- ZERO bits change",
        deletion="B",
        gen_key="table_base",
    )
    arms["table+gate8/clear_cylinder"] = dict(
        gen=_T(tabs["base"]),
        refuse=g8.clear_value(hg, hu).refuse,
        family="gate",
        role="the gate's only other approximation to the conjunction: clear the SHAPE",
        deletion="B",
        gen_key="table_base",
    )
    arms["table+joint/base"] = dict(
        gen=_T(tabs["base"]),
        refuse=gj.refuse,
        family="gate",
        role="the joint hue x shape PRESENCE gate, pre-deletion",
        deletion=None,
        gen_key="table_base",
    )
    arms["table+joint/hole"] = dict(
        gen=_T(tabs["base"]),
        refuse=gj.delete_conjunction(hf, hv, hg, hu).refuse,
        family="gate",
        role="RIVAL: the joint PRESENCE gate -- ONE bit means both 'never seen' and 'deleted', so the deletion is a no-op AND every unseen conjunction is refused",
        deletion="B",
        gen_key="table_base",
    )
    arms["table+backoff/base"] = dict(
        gen=_T(tabs["base"]),
        refuse=gb.refuse,
        family="gate",
        role="the GENEROUS joint rival, pre-deletion (unseen pairs back off and are answered)",
        deletion=None,
        gen_key="table_base",
    )
    arms["table+backoff/hole"] = dict(
        gen=_T(tabs["base"]),
        refuse=gb.delete_conjunction(hf, hv, hg, hu).refuse,
        family="gate",
        role="RIVAL, the generous one: marginal back-off + a SEPARATE deleted-pair array. Expected to TIE the field store's hole on the refusal half -- declared in advance",
        deletion="B",
        gen_key="table_base",
    )
    return {
        "arms": arms,
        "field_variants": fv,
        "tables": tabs,
        "gates": {
            "per_factor": g8,
            "joint": gj,
            "joint_backoff": gb,
            "clear_red": g8.clear_value(fac, valx),
            "clear_cylinder": g8.clear_value(hg, hu),
            "hole_attempt": g8.delete_conjunction(hf, hv, hg, hu),
            "joint_hole": gj.delete_conjunction(hf, hv, hg, hu),
            "backoff_hole": gb.delete_conjunction(hf, hv, hg, hu),
        },
        "base": base,
    }


def _reqs(values: np.ndarray, grid) -> list:
    names = list(grid.names)
    out = []
    for row in np.asarray(values, np.float64):
        out.append(
            {
                "request": [
                    {"factor": names[k], "factor_index": int(k), "value": float(row[k])}
                    for k in range(row.size)
                    if not np.isnan(row[k])
                ]
            }
        )
    return out


def _table_oor(values: np.ndarray, grid) -> np.ndarray:
    oor = np.zeros(values.shape[0], bool)
    for f in range(values.shape[1]):
        gv = np.asarray(grid.values[f], np.float64)
        v = values[:, f]
        a = ~np.isnan(v)
        oor |= a & ((v < gv.min() - 1e-12) | (v > gv.max() + 1e-12))
    return oor


_JUDGE: dict = {}


def _judge(seed: int, log: Callable) -> Judge:
    if int(seed) not in _JUDGE:
        _JUDGE[int(seed)] = Judge(int(seed), log=log)
    return _JUDGE[int(seed)]


def _refuse_mask(refuse_fn, values, oor) -> tuple:
    """Normalise the two refusal-verb signatures to ``(refuse, detail)``."""
    try:
        r = refuse_fn(values)
    except TypeError:
        r = refuse_fn(values, oor)
    if isinstance(r, dict):
        det = {
            k: np.asarray(v) if isinstance(v, np.ndarray) else v
            for k, v in r.items()
            if k in ("hole", "empty_basin", "in_gap", "out_of_range")
        }
        return (np.asarray(r["refuse"], bool), det)
    return (np.asarray(r, bool), {})


def score_seed(out: Path, seed: int, max_seconds: float = 7200.0, log: Callable = print) -> dict:
    d = seed_dir(out, seed)
    art = d / "scores.npz"
    if art.exists():
        log(f"  score(seed {seed}): already on disk")
        return {"incomplete": False}
    cache = assert_generation2(seed)
    cond = load_condition()
    grid = cond["grid"]
    cs = scored_corners(cond)
    pool = scored_pool(cond, seed)
    proof = assert_scored_and_heldout(cond, cs, pool)
    probes = conjunction_probes(cond, seed)
    built = build_arms(out, seed, log=log)
    arms = built["arms"]
    J = _judge(int(seed), log)
    shared_bitmap = value_presence_bitmap(grid, pool)
    part = d / "scores_partial.pkl"
    done: dict = {}
    if part.exists():
        with open(part, "rb") as fh:
            done = pickle.load(fh)
        log(f"  score(seed {seed}): resuming with {len(done)} (arm, corner) units done")
    units = []
    for a in arms:
        for c in cs["all"]:
            units.append((a, c["corner_id"]))
        for pname in probes:
            if not pname.startswith("_"):
                units.append((a, f"probe/{pname}"))
    by_id = {c["corner_id"]: c for c in cs["all"]}
    gen_cache: dict = {}
    equalities: dict = {}
    twins = {arms[a]["identical_to"] for a in arms if arms[a].get("identical_to")}
    keys = sorted({arms[a]["gen_key"] for a in arms}, key=lambda k: (0 if k in twins else 1, k))
    rank = {k: i for i, k in enumerate(keys)}
    order = sorted(units, key=lambda u: (u[1], rank[arms[u[0]]["gen_key"]]))
    cur_cid = None
    t0 = _now()
    n_new = 0
    for a, cid in order:
        if (a, cid) in done:
            continue
        if _now() - t0 > float(max_seconds):
            with open(part, "wb") as fh:
                pickle.dump(done, fh)
            log(
                f"  score(seed {seed}): wall-clock cap after {n_new} new units -- state saved, rerun to continue"
            )
            return {"incomplete": True, "n_done": len(done)}
        if cid != cur_cid:
            gen_cache.clear()
            cur_cid = cid
        if cid.startswith("probe/"):
            vals = probes[cid.split("/", 1)[1]]["values"]
            fam = "probe"
        else:
            vals = grid.values_of(np.asarray(by_id[cid]["rows"], np.int64))
            fam = by_id[cid]["family"]
        gk = arms[a]["gen_key"]
        twin = arms[a].get("identical_to")
        if (gk, cid) in gen_cache:
            feats, oor, tup, fh_ = gen_cache[gk, cid]
        else:
            feats, oor = arms[a]["gen"](vals)
            fh_ = int(zlib.crc32(np.ascontiguousarray(feats, np.float32).tobytes()))
            if twin is not None and (twin, cid) in gen_cache:
                tw = gen_cache[twin, cid]
                same = bool(
                    tw[3] == fh_
                    and np.array_equal(np.asarray(feats, np.float32), np.asarray(tw[0], np.float32))
                )
                e = equalities.setdefault(
                    f"{gk} == {twin}", {"n_units": 0, "all_bit_identical": True}
                )
                e["n_units"] += 1
                e["all_bit_identical"] = bool(e["all_bit_identical"] and same)
                if same:
                    tup = tw[2]
                else:
                    tup = J.read_back(feats, int(PINS["eval_batch"]))
                    n_new += 1
            else:
                tup = J.read_back(feats, int(PINS["eval_batch"]))
                n_new += 1
            gen_cache[gk, cid] = (feats if gk in twins else None, oor, tup, fh_)
        refuse, det = _refuse_mask(arms[a]["refuse"], vals, oor)
        shared = accept_rule(vals, oor, np.zeros(vals.shape[0], bool), shared_bitmap, grid)
        accept = shared & ~refuse
        valid = judge_valid(vals, tup, grid) & accept
        done[a, cid] = {
            "n": int(vals.shape[0]),
            "family": fam,
            "own_refuse": refuse,
            "shared_accept": shared,
            "accept": accept,
            "valid": valid,
            "tuples": np.where(accept[:, None], tup, -1),
            "raw_tuples": tup,
            "hole": det.get("hole"),
            "empty_basin": det.get("empty_basin"),
            "in_gap": det.get("in_gap"),
            "feat_crc32": fh_,
        }
        if len(done) % 100 == 0:
            el = _now() - t0
            log(
                f"  score(seed {seed}): {len(done)}/{len(units)} units, {n_new} judge passes, {el:.0f}s elapsed"
            )
    with open(part, "wb") as fh:
        pickle.dump(done, fh)
    payload = {
        "done": np.array(done, dtype=object),
        "generator_equalities": np.array(equalities, dtype=object),
        "proof": np.array(proof, dtype=object),
        "cache": np.array(cache, dtype=object),
        "probes": np.array(
            {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in probes.items()},
            dtype=object,
        ),
        "ops": np.array(built["field_variants"]["_ops"], dtype=object),
        "table_cost": np.array(built["tables"]["cost"], dtype=object),
        "arm_roles": np.array(
            {
                k: {"role": v["role"], "family": v["family"], "deletion": v["deletion"]}
                for k, v in arms.items()
            },
            dtype=object,
        ),
        "corner_sets": np.array(
            {
                "deleted": [c["corner_id"] for c in cs["deleted"]],
                "survivor": [c["corner_id"] for c in cs["survivor"]],
                "nway": [c["corner_id"] for c in cs["nway"]],
                "pair": [c["corner_id"] for c in cs["pair"]],
            },
            dtype=object,
        ),
    }
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art, **payload)
    log(f"  score(seed {seed}): {len(done)} units in {_now() - t0:.1f}s")
    return {"incomplete": False, "n_done": len(done)}


LEARNER_EPOCHS, LEARNER_LR, LEARNER_WD = (20, 0.001, 0.0)


def _per_corner(rec, grid, seen_mask) -> dict:
    cards = tuple((int(c) for c in grid.cardinalities))
    v = np.asarray(rec["valid"], bool)
    t = np.asarray(rec["tuples"], np.int64)
    keep = t[v]
    if keep.size:
        lin = np.ravel_multi_index(tuple(keep.T), cards)
        uniq = np.unique(lin)
        novel = uniq[~seen_mask[uniq]]
    else:
        uniq = np.zeros(0, np.int64)
        novel = np.zeros(0, np.int64)
    return {
        "n_requests": int(rec["n"]),
        "n_accepted": int(np.asarray(rec["accept"]).sum()),
        "refuse_rate": float(np.asarray(rec["own_refuse"]).mean()),
        "judge_accuracy": float(v.mean()) if rec["n"] else 0.0,
        "distinct_valid_tuples": int(uniq.size),
        "distinct_valid_novel_tuples": int(novel.size),
    }


def _paired(a_rows: dict, b_rows: dict, ids, metric: str) -> dict:
    """Paired per-corner difference ``a - b``, with a bootstrap CI.  Never a single cell."""
    dif = np.array([float(a_rows[i][metric]) - float(b_rows[i][metric]) for i in ids], np.float64)
    n = dif.size
    if n == 0:
        return {"NOT_RUN": "no paired corners"}
    rng = np.random.default_rng(20260920)
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(2000)])
    from math import sqrt

    se = float(dif.std(ddof=1) / sqrt(n)) if n > 1 else 0.0
    t = float(dif.mean() / se) if se > 0 else 0.0 if dif.mean() == 0 else float("inf")
    return {
        "n_corners": int(n),
        "mean_delta": float(dif.mean()),
        "median_delta": float(np.median(dif)),
        "sd": float(dif.std(ddof=1)) if n > 1 else 0.0,
        "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        "t": t,
        "n_positive": int((dif > 0).sum()),
        "n_negative": int((dif < 0).sum()),
        "n_zero": int((dif == 0).sum()),
        "max_abs_delta": float(np.abs(dif).max()),
    }
