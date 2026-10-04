"""hub/refusal-2 -- the LINEAR-READ coercive-component screen.

THE OBJECT (PREREG-hub-refusal-2 section 1), and nothing else:

    S(request) = E_store(x_read),  x_read = the posterior-weighted MEAN of the well's atoms

Every piece is the pinned harness's, CALLED and not re-implemented:
  * ``x_read``  -- ``gen_landscape.launch_state(..., launch_mode="posterior_mean")``'s
    arithmetic, ``gen_landscape.py:1028``, generalised from a dense ``(n, F)`` class-index
    array to one expert per requested TERM so that the two invalid kinds can be expressed
    at all.  The generalisation is the PREDECESSOR's ``refusal_lib.term_experts`` --
    inherited, adopted by explicit reference, not re-derived here.
  * ``E_store`` -- ``gen_landscape.store_energy``, ``gen_landscape.py:895``, at the
    reference bandwidth ``sigma_scale = c.sigma_scale_end``.
  * the landscape -- ``exp_gen_dynamical_read.build_landscape`` via ``refusal_lib``.

WHY THIS IS NOT THE DEAD CRITERION.  Under the ARGMAX read (``clamp_atom``) ``x_read`` is
always exactly a stored atom, so ``d(x_read, nearest atom) == 0`` and the statistic is
definitionally blind.  Under the LINEAR read the per-well weights are a full simplex, so a
request no atom satisfies puts ``x_read`` BETWEEN atoms -- outside every basin.

NOT ONE MANIFEST-PINNED FILE IS EDITED, and nothing under ``scorer/`` is modified: this
module lives beside it and imports it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# The predecessor's library -- inherited fact, imported, never edited.
_SCORER1 = Path(__file__).resolve().parent.parent / "scorer"
if str(_SCORER1) not in sys.path:
    sys.path.insert(0, str(_SCORER1))

CHLU_ROOT = os.environ["CHLU_ROOT"]
if CHLU_ROOT not in sys.path:
    sys.path.insert(0, CHLU_ROOT)

import refusal_lib as R  # noqa: E402

from chlu.config import gen_dynamical_read_config  # noqa: E402
from chlu.core.gen_landscape import (  # noqa: E402
    AtomLandscape,
    _softmax_last,
    atom_posterior,
    launch_state,
    store_energy,
)

#: The pre-registered read.  Both names are the shipped constants, quoted not chosen.
LINEAR_LAUNCH_MODE = "posterior_mean"
ARGMAX_LAUNCH_MODE = "clamp_atom"


# ---------------------------------------------------------------------------
# 1. The posterior over stored atoms, from one expert per requested TERM
# ---------------------------------------------------------------------------


def term_posterior(book, log_w, n_terms) -> np.ndarray:
    """``(T, n, K, N) log-weights -> (n, K, N)`` posterior ``P(atom | request)``.

    Exactly ``launch_state``'s ``_softmax_last(atom_posterior(book, R))`` with the dense
    per-FACTOR sum replaced by the per-TERM sum, which is what lets a request name one
    factor twice or name an off-grid value.  Identical for a valid request; asserted.
    """
    lp = np.asarray(book.log_prior, np.float64)          # (K, N)
    n = int(log_w.shape[1])
    S = np.broadcast_to(lp, (n,) + lp.shape).copy()
    for t in range(int(log_w.shape[0])):
        active = (np.asarray(n_terms) > t)[:, None, None]
        S = S + np.where(active, np.asarray(log_w[t], np.float64), 0.0)
    return _softmax_last(S)


def linear_read(store, P) -> np.ndarray:
    """``x_read[:, cols_k] = P[:, k, :] @ atoms[k]`` -- ``gen_landscape.py:1029``, verbatim."""
    P = np.asarray(P, np.float64)
    n = int(P.shape[0])
    x = np.zeros((n, int(store.dim)), np.float32)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)
        x[:, cols] = (P[:, k, :] @ A).astype(np.float32)
    return x


def assert_reproduces_shipped_linear_read(land, book, requests, P) -> dict:
    """For a VALID row-derived request set, ``linear_read`` must equal the SHIPPED
    ``launch_state(..., "posterior_mean")`` bit for bit.  This is the proof that this hub
    scores the pinned linear read and not a look-alike of its own.
    """
    if any("class_indices" not in r for r in requests):
        return {"skipped": "not a row-derived valid request set"}
    Rq = np.asarray([[int(c) for c in r["class_indices"]] for r in requests], np.int64)
    ref = launch_state(land, book, Rq, LINEAR_LAUNCH_MODE, 0.5)
    got = linear_read(land.store, P)[: Rq.shape[0]]
    x_ref = np.asarray(ref["x0"], np.float32)
    P_ref = np.asarray(_softmax_last(atom_posterior(book, Rq)), np.float64)
    return {
        "x_read_bit_identical": bool(np.array_equal(got, x_ref)),
        "x_read_max_abs_delta": float(np.max(np.abs(got - x_ref))) if x_ref.size else 0.0,
        "posterior_bit_identical": bool(np.array_equal(np.asarray(P, np.float64)[: Rq.shape[0]],
                                                      P_ref)),
        "posterior_max_abs_delta": float(np.max(np.abs(np.asarray(P, np.float64)[: Rq.shape[0]]
                                                       - P_ref))) if P_ref.size else 0.0,
        "n": int(Rq.shape[0]),
    }


# ---------------------------------------------------------------------------
# 2. The statistic, its physics-free control, and the companion rejected in advance
# ---------------------------------------------------------------------------


def margin_control(P) -> np.ndarray:
    """PREREG section 5 control 2 -- the PHYSICS-FREE readout margin.

    ``M = -sum_k log max_a P(a | request, well k)``.  The address book's own confidence,
    with NO atom content, NO kernel, NO bandwidth and NO energy in it.  If ``S`` does not
    beat this the physics is not doing the work and the report says so.
    """
    P = np.asarray(P, np.float64)
    return -np.log(np.maximum(P.max(axis=-1), 1e-300)).sum(axis=-1)


def nearest_atom_distance(store, X) -> dict:
    """PREREG section 6 -- COMPANION REJECTED IN ADVANCE.  Computed, logged, NEVER adopted.

    ``d(x_read, nearest atom)`` (rms over wells) and the atom spread it is divided by.
    """
    X = np.asarray(X, np.float32)
    n = int(X.shape[0])
    d2 = np.zeros(n, np.float64)
    radii = []
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float64)
        Z = np.asarray(X[:, cols], np.float64)
        # (n, N) squared distances, chunked so the (n, N, d) tensor is never formed.
        g = (Z * Z).sum(1)[:, None] + (A * A).sum(1)[None, :] - 2.0 * (Z @ A.T)
        d2 += np.maximum(g.min(axis=1), 0.0)
        radii.append(float(A.std()))
    ref = float(np.sqrt(np.mean(np.asarray(radii) ** 2))) or 1.0
    d = np.sqrt(d2)
    return {"d_nearest": d, "atom_rms_radius": ref, "d_over_spread": d / ref,
            "REJECTED_IN_ADVANCE": "PREREG-hub-refusal-2 section 6 -- logged, never adopted"}


# ---------------------------------------------------------------------------
# 3. One resumable unit
# ---------------------------------------------------------------------------


def linear_unit(ctx: dict, built: dict, set_name: str, requests, arm: str = "linear",
                budget: str = "A", log=R.tlog) -> dict:
    """One unit: the linear read of ``requests``, its ``S``, its controls.

    ``arm``:
      ``linear``                the pre-registered object, full store;
      ``linear_store_removed``  charter section C.2 -- every channel column of every well's
                                atom matrix permuted independently among the N atoms.
                                Byte count unchanged, channel marginals unchanged, no atom
                                a train row.  EXACTLY ONE declared quantity differs.

    NOTE the arm composition that matters and is stated in the PREREG: the shuffle leaves
    the POSTERIOR exactly unchanged (the address book is built from atom LABELS, never
    content), so the only thing that can collapse ``S`` is the atom cloud's GEOMETRY.
    """
    import time

    t0 = time.time()
    config = ctx["config"]
    c = gen_dynamical_read_config(config)
    sig_ref = float(c.sigma_scale_end)

    land = built["land"]
    book = built["book"]
    if "store_removed" in arm:
        land = AtomLandscape(R.shuffled_store(land.store, int(ctx["seed"])),
                             np.asarray(land.sigma), land.coupling,
                             float(land.request_strength), float(land.coupling_strength))

    log_w, n_terms, tinfo = R.term_experts(book, ctx["grid"], requests, R.T_MAX)
    repro_experts = R.assert_reproduces_shipped_experts(book, requests, log_w)
    P = term_posterior(book, log_w, n_terms)
    repro_read = assert_reproduces_shipped_linear_read(land, book, requests, P)

    x_read = linear_read(land.store, P)
    S = np.asarray(store_energy(land, x_read, sig_ref), np.float64)
    M = margin_control(P)
    comp = nearest_atom_distance(land.store, x_read)

    # THE ARGMAX read at the same bandwidth -- the DEAD criterion's read point, kept as a
    # one-line structural check that d(x_read, nearest atom) == 0 there.  Diagnostic only.
    idx = P.argmax(axis=-1)
    x_arg = np.zeros_like(x_read)
    for k, w in enumerate(land.store.well_channels):
        cols = np.asarray(w, np.int64)
        x_arg[:, cols] = np.asarray(land.store.atoms[k], np.float32)[idx[:, k]]
    S_arg = np.asarray(store_energy(land, x_arg, sig_ref), np.float64)
    d_arg = nearest_atom_distance(land.store, x_arg)["d_nearest"]

    return {
        "meta": {
            "seed": int(ctx["seed"]), "budget": str(budget), "set": str(set_name),
            "arm": str(arm), "n": int(len(requests)), "steps": 0,
            "launch_mode": LINEAR_LAUNCH_MODE, "sigma_ref": sig_ref,
            "term_info": tinfo,
            "experts_reproduction": repro_experts,
            "linear_read_reproduction": repro_read,
            "mean_posterior_confidence": float(P.max(axis=-1).mean()),
            "atom_rms_radius": comp["atom_rms_radius"],
            "wall_clock_s": float(time.time() - t0),
        },
        "ids": [str(r["id"]) for r in requests],
        "kinds": [str(r.get("kind", "valid")) for r in requests],
        "n_terms": np.asarray(n_terms),
        "S": S,                              # THE pre-registered statistic
        "M": M,                              # control 2, physics-free
        "d_over_spread": comp["d_over_spread"],   # companion, REJECTED IN ADVANCE
        "S_argmax_read": S_arg,              # diagnostic: the dead read's point
        "d_argmax_read": d_arg,              # diagnostic: must be ~0
        "posterior_conf": P.max(axis=-1),
        "x_read": x_read,
    }


def save_linear_unit(rec: dict, path: Path) -> dict:
    """Persist the moment a unit finishes -- a kill costs ONE UNIT, never the run."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(rec["ids"]), kinds=np.asarray(rec["kinds"]),
        n_terms=rec["n_terms"], S=rec["S"], M=rec["M"],
        d_over_spread=rec["d_over_spread"],
        S_argmax_read=rec["S_argmax_read"], d_argmax_read=rec["d_argmax_read"],
        posterior_conf=rec["posterior_conf"],
        x_read=rec["x_read"].astype(np.float32),
        meta_json=np.asarray(json.dumps(rec["meta"], default=str)),
    )
    return {"path": str(path), "bytes": int(path.stat().st_size)}
