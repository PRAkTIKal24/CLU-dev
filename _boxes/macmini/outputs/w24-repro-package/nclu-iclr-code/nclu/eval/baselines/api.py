"""api."""

from __future__ import annotations
from nclu.paths import run_dir
import json
import os
import pickle
from pathlib import Path
from typing import Optional
import equinox as eqx
import numpy as np
from nclu.eval.baselines import learners as L

DEFAULT_ROOT = (
    Path(os.environ.get("NCLU_OUT", str(run_dir()))) / "shapes3d/baseline_suite" / "rivals"
)
BASE_FAMILIES = ("mlp", "mlp_180k", "cont_mlp", "cont_mlp_notune", "transformer", "lstm")
LEARNER_FAMILIES = tuple([f for f in BASE_FAMILIES] + [f"{f}_dropout" for f in BASE_FAMILIES])


def base_family(name: str) -> str:
    """The architecture behind a variant name (``mlp_dropout`` -> ``mlp``)."""
    return name[: -len("_dropout")] if name.endswith("_dropout") else name


class RivalRefusal(RuntimeError):
    """This rival cannot express this request. Not a number, and not a zero."""


def _ckpt_dir(seed: int, root: Optional[Path] = None) -> Path:
    d = Path(root or DEFAULT_ROOT) / f"seed_{int(seed)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(
    seed: int,
    protos,
    grid,
    cache_manifest: dict,
    population: Optional[dict] = None,
    root: Optional[Path] = None,
) -> Path:
    """Pickle the fitted prototype table plus what a caller needs to use it."""
    d = _ckpt_dir(seed, root)
    p = d / "table.rival.pkl"
    with open(p, "wb") as fh:
        pickle.dump(
            {
                "kind": "table",
                "seed": int(seed),
                "protos": protos,
                "names": list(grid.names),
                "grid_values": [np.asarray(v, np.float64) for v in grid.values],
                "cardinalities": [int(c) for c in grid.cardinalities],
                "generation": 2,
                "population": population or {},
                "features_sha256": cache_manifest["features_sha256"],
            },
            fh,
        )
    return p


def save_learner(
    seed: int,
    name: str,
    model,
    width: int,
    in_kind: str,
    grid,
    out_dim: int,
    cache_manifest: dict,
    population: Optional[dict] = None,
    root: Optional[Path] = None,
    extra: Optional[dict] = None,
) -> Path:
    d = _ckpt_dir(seed, root)
    meta = {
        "kind": name,
        "seed": int(seed),
        "width": int(width),
        "in_kind": in_kind,
        "out_dim": int(out_dim),
        "names": list(grid.names),
        "cardinalities": [int(c) for c in grid.cardinalities],
        "grid_values": [[float(x) for x in np.asarray(v)] for v in grid.values],
        "param_count": L.count_params(model),
        "generation": 2,
        "population": population or {},
        "features_sha256": cache_manifest["features_sha256"],
    }
    meta.update(dict(extra or {}))
    with open(d / f"{name}.rival.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    p = d / f"{name}.rival.eqx"
    eqx.tree_serialise_leaves(p, model)
    return p


def save_refusal(seed: int, name: str, reason: str, root: Optional[Path] = None) -> Path:
    """Record, as an artifact, that this rival cannot be offered as a callable."""
    d = _ckpt_dir(seed, root)
    p = d / f"{name}.rival.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "kind": name,
                "seed": int(seed),
                "callable": False,
                "NOT_AVAILABLE_AS_CALLABLE": reason,
                "generation": 2,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")
    return p


class RivalRead:
    """``values (N, F) -> (features (N, D) float32, out_of_range (N,) bool)``."""

    def __init__(self, name: str, seed: int, meta: dict, fn, offgrid_ok: bool, partial_ok: bool):
        self.name, self.seed, self.meta = (name, int(seed), meta)
        self._fn = fn
        self.offgrid_ok, self.partial_ok = (bool(offgrid_ok), bool(partial_ok))
        self.names = list(meta["names"])
        self.grid_values = [np.asarray(v, np.float64) for v in meta["grid_values"]]
        self.cardinalities = [int(c) for c in meta["cardinalities"]]
        self.generation = int(meta.get("generation", 2))
        self.population = dict(meta.get("population") or {})
        self.fit_population = self.population.get("fit_population")
        self.n_per_value = self.population.get("n_per_value") or {}

    def __repr__(self) -> str:
        return f"RivalRead({self.name!r}, seed={self.seed}, generation={self.generation}, fit_population={self.fit_population!r}, offgrid={self.offgrid_ok}, partial={self.partial_ok})"

    def like_for_like(self, held_out: dict) -> dict:
        from nclu.eval.baselines.population import assert_not_fitted_on

        if not self.population:
            return {
                "like_for_like": None,
                "verdict": "this checkpoint carries no population record; re-run the backfill or refit",
            }
        return assert_not_fitted_on(self.population, held_out)

    def _analyse(self, values: np.ndarray):
        V = np.asarray(values, np.float64)
        if V.ndim != 2 or V.shape[1] != len(self.names):
            raise ValueError(
                f"values must be (N, {len(self.names)}) in the order {self.names}; got {V.shape}"
            )
        present = ~np.isnan(V)
        oor = np.zeros(V.shape[0], bool)
        offgrid = np.zeros(V.shape, bool)
        for k, gv in enumerate(self.grid_values):
            col, pk = (V[:, k], present[:, k])
            oor |= pk & ((col < gv[0]) | (col > gv[-1]))
            near = np.zeros(V.shape[0], bool)
            near[pk] = np.isin(col[pk], gv)
            offgrid[:, k] = pk & ~near
        return (V, present, oor, offgrid)

    def __call__(self, values: np.ndarray) -> tuple:
        V, present, oor, offgrid = self._analyse(values)
        if offgrid.any() and (not self.offgrid_ok):
            n = int(offgrid.any(axis=1).sum())
            raise RivalRefusal(
                f"{self.name!r} cannot express an off-grid value: {n} of {V.shape[0]} requests carry one. Its vocabulary over the realised grid is closed. This rival gets a dash and a footnote on that row; it does not get a nearest-grid-value number."
            )
        if (~present).any() and (not self.partial_ok):
            n = int((~present).any(axis=1).sum())
            alt = (
                f" Load {self.name}_dropout instead: same architecture, same matched parameter count, trained with request dropout so a partially specified request is in distribution for it."
                if not self.name.endswith("_dropout") and self.name != "table"
                else ""
            )
            raise RivalRefusal(
                f"{self.name!r} was trained on fully specified requests only, and {n} of {V.shape[0]} requests leave a factor unrequested.{alt}"
            )
        feats = self._fn(V, present)
        return (np.asarray(feats, np.float32), oor)


def _table_fn(obj):
    protos = obj["protos"]

    def fn(V, present):
        reqs = []
        for i in range(V.shape[0]):
            reqs.append(
                [
                    {"factor_index": int(f), "value": float(V[i, f])}
                    for f in range(V.shape[1])
                    if present[i, f]
                ]
            )
        return protos.embed_many(reqs)

    return fn


def _learner_fn(model, meta):
    cards = [int(c) for c in meta["cardinalities"]]
    gvs = [np.asarray(v, np.float64) for v in meta["grid_values"]]
    kind = meta["in_kind"]
    if kind == "continuous_raw":

        def fn_raw(V, present):
            X = np.asarray(V, np.float64).copy()
            for k, gv in enumerate(gvs):
                X[:, k] = (X[:, k] - float(gv[0])) / max(float(gv[-1] - gv[0]), 1e-12)
            return L.predict(model, X.astype(np.float32))

        return fn_raw

    def fn(V, present):
        n, f = V.shape
        cls = np.zeros((n, f), np.int32)
        val = np.zeros((n, f), np.float32)
        for k, gv in enumerate(gvs):
            col = np.where(present[:, k], V[:, k], gv[0])
            cls[:, k] = np.argmin(np.abs(col[:, None] - gv[None, :]), axis=1)
            span = max(float(gv[-1] - gv[0]), 1e-12)
            val[:, k] = (col - float(gv[0])) / span
        req = _Dense(cls, val)
        return L.predict(model, L.encode_inputs(kind, req, cards, present))

    return fn


class _Dense:
    def __init__(self, cls_targets, val_targets):
        self.cls_targets, self.val_targets = (cls_targets, val_targets)

    @property
    def n(self) -> int:
        return int(self.cls_targets.shape[0])


def load_rival(seed: int, name: str, root: Optional[Path] = None) -> RivalRead:
    """Load one fitted rival for one seed, as a request-to-features callable.

    Raises:
        RivalRefusal: if this rival is not available as a callable at all (LACE
            and the PoE-EBM are sampler arms whose output is a draw, not a
            function of the request alone).
        FileNotFoundError: if it has not been fitted for this seed yet.
    """
    d = Path(root or DEFAULT_ROOT) / f"seed_{int(seed)}"
    if name == "table":
        p = d / "table.rival.pkl"
        if not p.exists():
            raise FileNotFoundError(f"no fitted table for seed {seed} at {p}")
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
        return RivalRead("table", seed, obj, _table_fn(obj), offgrid_ok=True, partial_ok=True)
    meta_p = d / f"{name}.rival.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"no fitted {name!r} for seed {seed} at {meta_p}")
    with open(meta_p, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("NOT_AVAILABLE_AS_CALLABLE"):
        raise RivalRefusal(f"{name!r}: {meta['NOT_AVAILABLE_AS_CALLABLE']}")
    declared = meta.get("base_family")
    if (
        meta.get("in_kind") != "continuous_raw"
        and name not in LEARNER_FAMILIES
        and (declared not in LEARNER_FAMILIES)
    ):
        raise RivalRefusal(
            f"{name!r} is not exposed as a callable (no builder for base family {base_family(declared or name)!r} and its metadata declares neither a self-describing input kind nor a loadable base_family)"
        )
    import jax

    bs = L.builders(meta["cardinalities"], int(meta["out_dim"]), jax.random.PRNGKey(0))
    if meta.get("in_kind") == "continuous_raw":
        skeleton = L.ContMLPProtocolMatched(
            __import__("jax").random.PRNGKey(0),
            len(meta["cardinalities"]),
            int(meta["width"]),
            int(meta["out_dim"]),
        )
        kind = "continuous_raw"
    else:
        build, kind = bs[base_family(declared or name)]
        skeleton = build(int(meta["width"]))
    model = eqx.tree_deserialise_leaves(d / f"{name}.rival.eqx", skeleton)
    got = L.count_params(model)
    if got != int(meta["param_count"]):
        raise AssertionError(
            f"{name!r} seed {seed}: deserialised {got} parameters, manifest says {meta['param_count']}"
        )
    return RivalRead(
        name,
        seed,
        meta,
        _learner_fn(model, meta),
        offgrid_ok=kind in ("continuous", "continuous_raw"),
        partial_ok=bool(meta.get("partial_ok", name.endswith("_dropout"))),
    )


def discover(seed: int, root: Optional[Path] = None) -> list:
    """Every rival name present for this seed, read off DISK.

    Enumerated rather than listed: a hand-maintained name list silently omits any
    rival built at a new budget, and omitting it from `available()` is how a
    consumer concludes an object does not exist while its checkpoint sits beside
    the ones that do.
    """
    d = Path(root or DEFAULT_ROOT) / f"seed_{int(seed)}"
    names = set()
    if (d / "table.rival.pkl").exists():
        names.add("table")
    for jp in d.glob("*.rival.json"):
        names.add(jp.name[: -len(".rival.json")])
    known = [n for n in ("table",) + LEARNER_FAMILIES + ("lace", "poe_ebm") if n in names]
    return known + sorted(names - set(known))


def available(seed: int, root: Optional[Path] = None) -> dict:
    """What is loadable for this seed, and why anything missing is missing."""
    out = {}
    for name in discover(seed, root) or list(("table",) + LEARNER_FAMILIES + ("lace", "poe_ebm")):
        try:
            r = load_rival(seed, name, root)
            out[name] = {
                "callable": True,
                "offgrid": r.offgrid_ok,
                "partial": r.partial_ok,
                "param_count": r.meta.get("param_count"),
                "fit_population": r.fit_population,
                "values_never_fitted": r.population.get("values_never_fitted", {}),
                "population_caveat": "fitted on the Montero combination pool, which contains EVERY factor value. NOT like-for-like with a value-holdout arm on rows 2 and 5 - call load_rival(...).like_for_like({factor: [values]}) to check.",
            }
        except RivalRefusal as exc:
            out[name] = {"callable": False, "reason": str(exc)}
        except FileNotFoundError as exc:
            out[name] = {"callable": False, "reason": f"not fitted yet: {exc}"}
    return out
