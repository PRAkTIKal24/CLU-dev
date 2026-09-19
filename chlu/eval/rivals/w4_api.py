"""The rival set as a CALLABLE, so a hub can run it through its own corners.

The scored cell measures the rivals on its own two request sets (``B`` and its
matched-seen control). Other hubs score on different ones - the value-holdout
scoreboard's 49 n-way corners and 60 pair sub-corners, for instance. Those cannot
be paired against a ``composition_B`` aggregate, so a rival that exists only as a
number on my request sets is useless to them.

This module ships each fitted rival as a function of a REQUEST, with the contract
``exp_gen_factor_wells.arm_features`` already uses::

    from chlu.eval.rivals.w4_api import load_rival

    r = load_rival(seed=1, name="table")
    feats, out_of_range = r(values)     # values (N, F) float, nan = not requested
                                        # feats (N, D) float32, oor (N,) bool

Factor order is ``grid.names``. ``out_of_range`` is True for a row in which any
REQUESTED value falls outside that factor's realised grid range; an unrequested
(``nan``) factor never sets it.

What each family does with a partial or off-grid request
--------------------------------------------------------
Nothing here guesses on a rival's behalf, and nothing snaps an off-grid value
onto the nearest grid point: a rival scored on a request it did not receive is
worse than a dash.

===============  =================================  ==========================
family           unrequested factor                 off-grid value
===============  =================================  ==========================
``table``        native - the term is omitted       native - interpolates and
                                                    extrapolates
``mlp``          all-zero one-hot block             nearest grid class only,
                                                    so REFUSED
``cont_mlp``     requested-flag channel = 0         native - the value is a
                                                    number
``transformer``  reserved per-factor absent token   nearest grid class only,
                 (``sum(C_k) + k``)                 so REFUSED
``lstm``         same reserved absent token         same, REFUSED
``poe_ebm``      REFUSED - the experts are          REFUSED - closed
                 conditioned on a full assignment   vocabulary
``lace``         on-grid partial only               REFUSED - closed
                                                    categorical vocabulary
===============  =================================  ==========================

A refusal comes back as :class:`RivalRefusal`, never as a silently substituted
number. The three generic token/one-hot families were trained with request
dropout, so an absent factor is in distribution for them rather than a guess.

LACE's and the PoE-EBM's off-grid refusal is the method's own structural limit,
documented at ``chlu.core.gen_request.DenseRequests``: a closed categorical
vocabulary cannot express a hue between two grid points. It is not a limit this
spoke imposed and it is not a weakness we introduced.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import equinox as eqx
import numpy as np

from . import w4_learners as L

#: Where the fitted rivals are written.
DEFAULT_ROOT = (Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs"))
                / "spoke-a-rivals" / "rivals")

BASE_FAMILIES = ("mlp", "mlp_180k", "cont_mlp", "transformer", "lstm")
#: Every loadable learner name: each family in its full-only (primary) and
#: request-dropout variants.
LEARNER_FAMILIES = tuple(
    [f for f in BASE_FAMILIES] + [f"{f}_dropout" for f in BASE_FAMILIES])


def base_family(name: str) -> str:
    """The architecture behind a variant name (``mlp_dropout`` -> ``mlp``)."""
    return name[: -len("_dropout")] if name.endswith("_dropout") else name


class RivalRefusal(RuntimeError):
    """This rival cannot express this request. Not a number, and not a zero."""


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _ckpt_dir(seed: int, root: Optional[Path] = None) -> Path:
    d = Path(root or DEFAULT_ROOT) / f"seed_{int(seed)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(seed: int, protos, grid, cache_manifest: dict,
               population: Optional[dict] = None,
               root: Optional[Path] = None) -> Path:
    """Pickle the fitted prototype table plus what a caller needs to use it."""
    d = _ckpt_dir(seed, root)
    p = d / "table.rival.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"kind": "table", "seed": int(seed), "protos": protos,
                     "names": list(grid.names),
                     "grid_values": [np.asarray(v, np.float64) for v in grid.values],
                     "cardinalities": [int(c) for c in grid.cardinalities],
                     "generation": 2,
                     "population": population or {},
                     "features_sha256": cache_manifest["features_sha256"]}, fh)
    return p


def save_learner(seed: int, name: str, model, width: int, in_kind: str, grid,
                 out_dim: int, cache_manifest: dict,
                 population: Optional[dict] = None,
                 root: Optional[Path] = None) -> Path:
    """Serialise a fitted learner with the metadata needed to rebuild its shape."""
    d = _ckpt_dir(seed, root)
    meta = {"kind": name, "seed": int(seed), "width": int(width),
            "in_kind": in_kind, "out_dim": int(out_dim),
            "names": list(grid.names),
            "cardinalities": [int(c) for c in grid.cardinalities],
            "grid_values": [[float(x) for x in np.asarray(v)] for v in grid.values],
            "param_count": L.count_params(model),
            "generation": 2,
            "population": population or {},
            "features_sha256": cache_manifest["features_sha256"]}
    with open(d / f"{name}.rival.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    p = d / f"{name}.rival.eqx"
    eqx.tree_serialise_leaves(p, model)
    return p


def save_refusal(seed: int, name: str, reason: str,
                 root: Optional[Path] = None) -> Path:
    """Record, as an artifact, that this rival cannot be offered as a callable."""
    d = _ckpt_dir(seed, root)
    p = d / f"{name}.rival.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"kind": name, "seed": int(seed), "callable": False,
                   "NOT_AVAILABLE_AS_CALLABLE": reason, "generation": 2},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p


# ---------------------------------------------------------------------------
# The callable
# ---------------------------------------------------------------------------


class RivalRead:
    """``values (N, F) -> (features (N, D) float32, out_of_range (N,) bool)``."""

    def __init__(self, name: str, seed: int, meta: dict, fn, offgrid_ok: bool,
                 partial_ok: bool):
        self.name, self.seed, self.meta = name, int(seed), meta
        self._fn = fn
        self.offgrid_ok, self.partial_ok = bool(offgrid_ok), bool(partial_ok)
        self.names = list(meta["names"])
        self.grid_values = [np.asarray(v, np.float64) for v in meta["grid_values"]]
        self.cardinalities = [int(c) for c in meta["cardinalities"]]
        self.generation = int(meta.get("generation", 2))
        # WHAT THIS RIVAL'S FIT POOL CONTAINED. A caller gets out_of_range=0 on a
        # value this rival was fitted on, which is correct and says nothing about
        # whether the comparison is like-for-like. The counts are here so the
        # mismatch reaches the call site rather than only a reader of the README.
        self.population = dict(meta.get("population") or {})
        self.fit_population = self.population.get("fit_population")
        self.n_per_value = self.population.get("n_per_value") or {}

    def __repr__(self) -> str:
        return (f"RivalRead({self.name!r}, seed={self.seed}, generation="
                f"{self.generation}, fit_population={self.fit_population!r}, "
                f"offgrid={self.offgrid_ok}, partial={self.partial_ok})")

    def like_for_like(self, held_out: dict) -> dict:
        """Is this rival comparable with an arm that held these values out?

        ``held_out`` is ``{factor_name: [values]}``. Returns a verdict naming any
        value this rival DID fit rows for. Call it before scoring a value-holdout
        row: a clean ``out_of_range = 0`` says the value is inside the grid, NOT
        that the rival is a fair comparison for an arm that never saw it.
        """
        from .w4_population import assert_not_fitted_on

        if not self.population:
            return {"like_for_like": None,
                    "verdict": ("this checkpoint carries no population record; "
                                "re-run the backfill or refit")}
        return assert_not_fitted_on(self.population, held_out)

    def _analyse(self, values: np.ndarray):
        V = np.asarray(values, np.float64)
        if V.ndim != 2 or V.shape[1] != len(self.names):
            raise ValueError(
                f"values must be (N, {len(self.names)}) in the order {self.names}; "
                f"got {V.shape}")
        present = ~np.isnan(V)
        oor = np.zeros(V.shape[0], bool)
        offgrid = np.zeros(V.shape, bool)
        for k, gv in enumerate(self.grid_values):
            col, pk = V[:, k], present[:, k]
            oor |= pk & ((col < gv[0]) | (col > gv[-1]))
            near = np.zeros(V.shape[0], bool)
            near[pk] = np.isin(col[pk], gv)
            offgrid[:, k] = pk & ~near
        return V, present, oor, offgrid

    def __call__(self, values: np.ndarray) -> tuple:
        V, present, oor, offgrid = self._analyse(values)
        if offgrid.any() and not self.offgrid_ok:
            n = int(offgrid.any(axis=1).sum())
            raise RivalRefusal(
                f"{self.name!r} cannot express an off-grid value: {n} of "
                f"{V.shape[0]} requests carry one. Its vocabulary over the "
                "realised grid is closed. This rival gets a dash and a footnote "
                "on that row; it does not get a nearest-grid-value number.")
        if (~present).any() and not self.partial_ok:
            n = int((~present).any(axis=1).sum())
            alt = (f" Load {self.name}_dropout instead: same architecture, same "
                   "matched parameter count, trained with request dropout so a "
                   "partially specified request is in distribution for it."
                   if not self.name.endswith("_dropout") and self.name != "table"
                   else "")
            raise RivalRefusal(
                f"{self.name!r} was trained on fully specified requests only, and "
                f"{n} of {V.shape[0]} requests leave a factor unrequested.{alt}")
        feats = self._fn(V, present)
        return np.asarray(feats, np.float32), oor


def _table_fn(obj):
    protos = obj["protos"]

    def fn(V, present):
        reqs = []
        for i in range(V.shape[0]):
            reqs.append([{"factor_index": int(f), "value": float(V[i, f])}
                         for f in range(V.shape[1]) if present[i, f]])
        return protos.embed_many(reqs)

    return fn


def _learner_fn(model, meta):
    cards = [int(c) for c in meta["cardinalities"]]
    gvs = [np.asarray(v, np.float64) for v in meta["grid_values"]]
    kind = meta["in_kind"]

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
    """The two fields :func:`w4_learners.encode_inputs` reads off a request."""

    def __init__(self, cls_targets, val_targets):
        self.cls_targets, self.val_targets = cls_targets, val_targets

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
        return RivalRead("table", seed, obj, _table_fn(obj),
                         offgrid_ok=True, partial_ok=True)

    meta_p = d / f"{name}.rival.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"no fitted {name!r} for seed {seed} at {meta_p}")
    with open(meta_p, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("NOT_AVAILABLE_AS_CALLABLE"):
        raise RivalRefusal(f"{name!r}: {meta['NOT_AVAILABLE_AS_CALLABLE']}")
    if name not in LEARNER_FAMILIES:
        raise RivalRefusal(f"{name!r} is not exposed as a callable")

    import jax

    bs = L.builders(meta["cardinalities"], int(meta["out_dim"]),
                    jax.random.PRNGKey(0))
    build, kind = bs[base_family(name)]
    skeleton = build(int(meta["width"]))
    model = eqx.tree_deserialise_leaves(d / f"{name}.rival.eqx", skeleton)
    got = L.count_params(model)
    if got != int(meta["param_count"]):
        raise AssertionError(
            f"{name!r} seed {seed}: deserialised {got} parameters, manifest says "
            f"{meta['param_count']}")
    # A full-only learner can ENCODE a partial request but was never trained on
    # one. It refuses, and names the variant that was - so nobody quotes an
    # out-of-distribution number by accident when an in-distribution one exists.
    return RivalRead(name, seed, meta, _learner_fn(model, meta),
                     offgrid_ok=(kind == "continuous"),
                     partial_ok=name.endswith("_dropout"))


def available(seed: int, root: Optional[Path] = None) -> dict:
    """What is loadable for this seed, and why anything missing is missing."""
    out = {}
    for name in ("table",) + LEARNER_FAMILIES + ("lace", "poe_ebm"):
        try:
            r = load_rival(seed, name, root)
            out[name] = {"callable": True, "offgrid": r.offgrid_ok,
                         "partial": r.partial_ok,
                         "param_count": r.meta.get("param_count"),
                         "fit_population": r.fit_population,
                         "values_never_fitted":
                             r.population.get("values_never_fitted", {}),
                         "population_caveat": (
                             "fitted on the Montero combination pool, which "
                             "contains EVERY factor value. NOT like-for-like with a "
                             "value-holdout arm on rows 2 and 5 - call "
                             "load_rival(...).like_for_like({factor: [values]}) to "
                             "check.")}
        except RivalRefusal as exc:
            out[name] = {"callable": False, "reason": str(exc)}
        except FileNotFoundError as exc:
            out[name] = {"callable": False, "reason": f"not fitted yet: {exc}"}
    return out
