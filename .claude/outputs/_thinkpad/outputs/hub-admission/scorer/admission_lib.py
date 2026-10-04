"""hub/admission -- the ADMISSION GATE, and nothing else.

THE GATE (DIRECTIVE rev. 3 SS1b property 1, registered verbatim in PREREG-1):
    refuse well k iff E_k(x_k) > 0,
    E_k(x) = -sigma_k^2 * log sum_i exp(-||x - a_ki||^2 / (2 sigma_k^2))

NOT RE-IMPLEMENTED.  ``gen_landscape.store_energy`` SUMS over wells, so it is not E_k.
The shipped per-well kernel ``gen_landscape._well_logk`` is CALLED and the per-well
reduction the public API does not expose is applied here -- exactly as the coordinator's
SS0b(ii) prescribes.  :func:`assert_matches_shipped_store_energy` proves numerically that
summing this module's per-well E_k reproduces the shipped ``store_energy``, so this is a
reduction of the shipped quantity and not a rival one.

sigma_k is read off the built landscape (``land.sigma``), i.e. ``atom_bandwidths`` at the
shipped rule ``nn_median`` with ``bandwidth_frac = 1.0``.  It is NEVER recomputed, scaled
or re-ruled here: SS7 forbids scaling sigma_k and this module gives no way to do it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

CHLU_ROOT = os.environ["CHLU_ROOT"]
if CHLU_ROOT not in sys.path:
    sys.path.insert(0, CHLU_ROOT)
_REFUSAL_SCORER = os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer")
if _REFUSAL_SCORER not in sys.path:
    sys.path.insert(0, _REFUSAL_SCORER)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from chlu.core.gen_landscape import (  # noqa: E402
    _softmax_last,
    _well_logk,
    _well_slices,
    store_energy,
)


# ---------------------------------------------------------------------------
# 1. The gate
# ---------------------------------------------------------------------------

def per_well_energy(land, X) -> np.ndarray:
    """``(n, K)`` -- E_k(x_k) per well: the shipped kernel, the per-well reduction."""
    Xj = jnp.asarray(np.asarray(X, np.float32))
    sig = land.sigma
    lk = _well_logk(land.store, Xj, sig)
    cols = [-(jnp.asarray(s, jnp.float32) ** 2) * jax.nn.logsumexp(l, axis=-1)
            for l, s in zip(lk, sig, strict=True)]
    return np.asarray(jnp.stack(cols, axis=1), np.float64)


def per_well_nearest(land, X) -> np.ndarray:
    """``(n, K)`` -- Euclidean distance from ``x_k`` to the nearest atom of well k."""
    Xj = jnp.asarray(np.asarray(X, np.float32))
    out = []
    for z, a in zip(_well_slices(land.store, Xj), land.store.atoms, strict=True):
        d2 = (jnp.sum(z * z, -1)[:, None] - 2.0 * (z @ a.T) + jnp.sum(a * a, -1)[None, :])
        out.append(jnp.sqrt(jnp.maximum(jnp.min(d2, axis=-1), 0.0)))
    return np.asarray(jnp.stack(out, axis=1), np.float64)


def certified_radius(land) -> np.ndarray:
    """``(K,)`` -- the certified SUFFICIENT condition ``sigma_k * sqrt(2 log N)``.

    Beyond it ``E_k > 0`` whatever the atom configuration.  Reported as the CERTIFIED
    FRACTION of refusals.  ``never a second criterion``.
    """
    sig = np.asarray(land.sigma, np.float64)
    N = np.asarray([int(np.asarray(a).shape[0]) for a in land.store.atoms], np.float64)
    return sig * np.sqrt(2.0 * np.log(N))


def gate(land, X) -> dict:
    """The admission decision.  ``accept`` iff NO well refuses."""
    E = per_well_energy(land, X)
    d = per_well_nearest(land, X)
    sig = np.asarray(land.sigma, np.float64)
    rad = certified_radius(land)
    refuse_k = E > 0.0
    certified_k = d >= rad[None, :]
    return {
        "E": E, "d": d, "d_over_sigma": d / sig[None, :],
        "sigma": sig, "certified_radius": rad,
        "refuse_well": refuse_k,
        "refuse": refuse_k.any(axis=1),
        "accept": ~refuse_k.any(axis=1),
        "n_wells_refusing": refuse_k.sum(axis=1),
        "certified_well": certified_k,
        "certified_refusal": certified_k.any(axis=1) & refuse_k.any(axis=1),
    }


def gate_summary(land, X) -> dict:
    g = gate(land, X)
    nref = int(g["refuse"].sum())
    return {
        "n": int(g["accept"].shape[0]),
        "accept_rate": float(g["accept"].mean()),
        "refuse_rate": float(g["refuse"].mean()),
        "certified_fraction_of_refusals": (float(g["certified_refusal"].sum()) / nref
                                           if nref else None),
        "mean_wells_refusing": float(g["n_wells_refusing"].mean()),
        "E_max_over_wells_mean": float(g["E"].max(axis=1).mean()),
        "E_max_over_wells_median": float(np.median(g["E"].max(axis=1))),
        "d_over_sigma_min_mean": float(g["d_over_sigma"].min(axis=1).mean()),
        "d_over_sigma_mean": float(g["d_over_sigma"].mean()),
        "per_well_refuse_rate": [float(x) for x in g["refuse_well"].mean(axis=0)],
        "per_well_E_mean": [float(x) for x in g["E"].mean(axis=0)],
        "per_well_d_over_sigma_mean": [float(x) for x in g["d_over_sigma"].mean(axis=0)],
    }


# ---------------------------------------------------------------------------
# 2. Launch points that can express an INVALID request
# ---------------------------------------------------------------------------
# ``gen_landscape.launch_state`` takes a dense ``(n, F)`` of grid CLASS INDICES, which
# cannot express an off-grid value or a duplicated factor.  ``refusal_lib.term_experts``
# generalises the address book to one expert per requested TERM using the SHIPPED
# ``gen_request._extrapolating_index`` (R8's own rule), and ``refusal_lib.launch_from_terms``
# is the ``clamp_atom`` launch on that tensor.  This is the same construction for the other
# shipped launch modes, so an invalid request reaches every launch point through ONE
# declared generalisation rather than several.

def launch_from_terms_mode(land, book, log_w, n_terms, mode: str = "posterior_mean"):
    """``(n, dim)`` launch point for arbitrary term tensors, in a shipped launch mode."""
    lp = np.asarray(book.log_prior, np.float64)
    n = int(log_w.shape[1])
    S = np.broadcast_to(lp, (n,) + lp.shape).copy()
    for t in range(int(log_w.shape[0])):
        active = (n_terms > t)[:, None, None]
        S = S + np.where(active, np.asarray(log_w[t], np.float64), 0.0)
    P = _softmax_last(S)                     # (n, K, N)
    store = land.store
    x0 = np.zeros((n, store.dim), np.float32)
    idx = P.argmax(axis=-1)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)
        if mode == "posterior_mean":
            x0[:, cols] = (P[:, k, :] @ A).astype(np.float32)
        elif mode == "clamp_atom":
            x0[:, cols] = A[idx[:, k]]
        else:
            raise ValueError(f"unsupported launch mode {mode!r}")
    return {"x0": x0, "posterior": P, "confidence": P.max(axis=-1).astype(np.float32),
            "atom_index": idx.astype(np.int64)}


def atom_cloud_spread(land) -> dict:
    """The OTHER normaliser.  ``hub/refusal-2`` SSC.4 reports distance in "atom spreads";
    this gate is in units of ``sigma_k = nn_median``.  Both are reported so the two
    numbers can be compared instead of guessed at."""
    sig = np.asarray(land.sigma, np.float64)
    rows = []
    for k, a in enumerate(land.store.atoms):
        A = np.asarray(a, np.float64)
        c = A.mean(axis=0)
        rms = float(np.sqrt(((A - c) ** 2).sum(axis=1).mean()))          # cloud RMS radius
        per_ch = float(np.sqrt(A.var(axis=0).mean()))                    # per-channel std
        rows.append({"well": k, "sigma_nn_median": float(sig[k]),
                     "cloud_rms_radius": rms, "per_channel_std": per_ch,
                     "rms_over_sigma": rms / float(sig[k]),
                     "per_channel_std_over_sigma": per_ch / float(sig[k])})
    return {"wells": rows}


# ---------------------------------------------------------------------------
# 3. PREREG-1's mandated self-checks
# ---------------------------------------------------------------------------

def assert_matches_shipped_store_energy(land, X, tol: float = 1e-2) -> dict:
    """sum_k E_k must reproduce the shipped ``store_energy``.  Proof of no re-implementation."""
    E = per_well_energy(land, X)
    ref = np.asarray(store_energy(land, jnp.asarray(np.asarray(X, np.float32))), np.float64)
    got = E.sum(axis=1)
    denom = np.maximum(np.abs(ref), 1.0)
    rel = float(np.max(np.abs(got - ref) / denom)) if ref.size else 0.0
    return {"max_abs_delta": float(np.max(np.abs(got - ref))) if ref.size else 0.0,
            "max_rel_delta": rel, "agrees": bool(rel <= tol),
            "note": "sum_k E_k (this module) vs gen_landscape.store_energy (shipped)"}


def identity_E_le_0_on_live_store(land) -> dict:
    """max_i E_k(a_ki) per well ON THE LIVE STORE.  PREREG-1 requires this before any
    gate number.  A violation is a BUG in this code, not a result."""
    store = land.store
    sig = np.asarray(land.sigma, np.float64)
    wells, worst = [], -np.inf
    for k, a in enumerate(store.atoms):
        A = jnp.asarray(np.asarray(a, np.float32))
        d2 = jnp.sum(A * A, -1)[:, None] - 2.0 * (A @ A.T) + jnp.sum(A * A, -1)[None, :]
        lk = -jnp.maximum(d2, 0.0) / (2.0 * float(sig[k]) ** 2)
        Ek = np.asarray(-(float(sig[k]) ** 2) * jax.nn.logsumexp(lk, axis=-1), np.float64)
        mx = float(Ek.max())
        worst = max(worst, mx)
        wells.append({"well": k, "n_atoms": int(A.shape[0]), "sigma": float(sig[k]),
                      "max_E_at_atoms": mx, "min_E_at_atoms": float(Ek.min()),
                      "certified_radius": float(sig[k] * np.sqrt(2.0 * np.log(A.shape[0])))})
    return {"wells": wells, "max_over_all_wells": worst,
            "identity_E_le_0_holds": bool(worst <= 0.0)}
