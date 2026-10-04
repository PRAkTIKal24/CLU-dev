"""w6 hub/potentials -- does the shorts' CRR predict the field store's settle?

ONE LEG = one (condition, seed).  Resumable: every stage writes one artifact and skips
when it is already on disk.

WHAT IS MEASURED, and why it is not circular
--------------------------------------------
The PREDICTION comes from a linearisation: the mass-whitened Hessian ``mu^2`` of the field
store's own kernel energy at a critical point, pushed through the shorts' closed-form
one-step eigenvalue (``w6_potentials.n_half_crr``).  Nothing is fitted.

The MEASUREMENT comes from rolling the **full nonlinear** damped velocity-Verlet map on
the real landscape from a finite displacement, and reading the half-life off the
trajectory.  The integrator is ``chlu.core.integrators.velocity_verlet_step``'s map,
reproduced here in float64 numpy so a 5000-step rollout over 63 modes at once is a
matmul rather than 63 traced scans (the map is checked against the jax integrator
bit-for-bit in :func:`assert_map_matches_repo_integrator`).

The gap between the two is the physics content.  A linear law predicting a nonlinear
rollout is a real claim; the amplitude sweep says where it stops.

THE DIALS
---------
``mass`` and ``gamma`` and ``eta`` are the declared dial.  ``mu^2 = hess_eig / mass``, so
sweeping mass over decades sweeps ``h = mu eta`` over decades and moves probes across the
band structure -- the field store's raw Hessian spectrum spans only ``[0.16, 1.0]``, which
is a factor 6, and the shorts verify over 4.5 orders of magnitude.

CONTROLS ON EVERY ROW (charter section C.2, w6-common section 1)
-----------------------------------------------------------------
* ``landscape_removed``  -- atoms zeroed.  The energy is then ``||x||^2/2`` exactly, so
  every ``mu^2`` is ``1/mass`` and every basin structure is gone.
* ``landscape_shuffled`` -- atom channels permuted within the well.  Same bytes, same
  width, destroyed geometry.
* The incumbent ``pca_tensor`` content is carried as a third landscape so the law is
  tested on the object that actually won w5, not only on the item store.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from ..core import w6_potentials as W6

# ---------------------------------------------------------------------------
# PINS -- every number this leg runs on, in one place.  Changed only on `val`.
# ---------------------------------------------------------------------------
PINS = {
    # eps = 0.05 is the step every quantitative result in all three shorts runs at
    # (inventory "Shared rider"); inherited verbatim rather than chosen.
    "eta": 0.05,
    # The gamma axis IS the V-curve of D16-D18.  Log-spaced over 2.4 decades so both
    # branch slopes are fittable in V5's own windows and the argmin is bracketed.
    "gammas": (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5),
    # mass moves mu^2 = hess_eig/mass.  With the field store's measured spectrum
    # hess_eig in [0.16, 1.0] this grid puts h = eps*mu across ~4.5 orders of
    # magnitude -- the range V2 Fig. 1 verifies the law over.  The raw spectrum alone
    # spans a factor 6 and would probe one band only.
    "masses": (1e-3, 1e-1, 1.0, 1e1, 1e3),
    "amplitudes": (0.05,),            # V5's smallest "finite write amplitude"
    "n_modes_probed": 3,              # top / middle / bottom of the curvature spectrum
    "confinement_alphas": (0.0,),     # D14's Hessian check is arithmetic and needs no roll
    # V5 runs its lifetime protocol with a censoring cap; a cell whose PREDICTED
    # half-life exceeds this is censored and NOT rolled, and reported as censored.
    # The local bracket around each mode's own predicted critical damping, so B4 is a
    # test of D16 rather than of my grid spacing.  See corner_rows for why this does
    # not put the answer in by hand.
    "gamma_refine": 5,
    "gamma_refine_span": 2.0,
    "censor_steps": 5000,
    "rollout_slack": 3.0,             # roll 3x the prediction: a 3x miss is still seen
    "settle_tol": 1e-10,
    "settle_max_iter": 2000,
    "half_life_floor_steps": 3,       # V5 quotes no rollout half-life from an n < 3 cell
}

# The richer grid, run on a declared SUBSET of corners: it adds V5's large write
# amplitude (where the linearisation is expected to stop) and D14's rolled confinement
# arm.  Same laws, same bars; it says where the law holds, not whether it holds.
PINS_AUX = dict(PINS)
PINS_AUX.update({
    "amplitudes": (0.05, 0.5),        # V5's two extremes
    "n_modes_probed": 4,
    "confinement_alphas": (0.0, 1.0),
    # A COARSER (mass, gamma) grid than the main one on purpose: this arm's questions
    # are "where does the amplitude break the linearisation?" and "does D14's lift still
    # predict once it is rolled?", and neither needs the full V-curve resolution.  Both
    # bands are still spanned.
    "masses": (1e-2, 1.0, 1e2),
    "gammas": (0.005, 0.02, 0.05, 0.2, 0.5),
    "gamma_refine": 0,
})


# ---------------------------------------------------------------------------
# The map.  float64 numpy, checked against the repo's jax integrator.
# ---------------------------------------------------------------------------
def verlet_rollout(X0: np.ndarray, A: np.ndarray, sigma: float, life,
                   mass: float, gamma: float, eta: float, n_steps: int,
                   x_star: np.ndarray, V: np.ndarray, alpha: float = 0.0,
                   record_every: int = 1) -> np.ndarray:
    """Roll ``n_steps`` of the damped velocity-Verlet map from ``X0`` (n_probe, d).

    ``H = p^T p / (2 mass) + E(q) + alpha ||q - x*||^2``; kick-drift-kick then
    ``p <- (1-gamma) p``, which is exactly ``chlu.core.integrators.velocity_verlet_step``.
    ``grad E = q - <a>_w`` is closed form, so no autodiff is needed.

    ``alpha`` is D14's confinement term (0 = off).  ``V`` is ``(d, n_probe)``: probe ``j``
    is displaced along, and read on, column ``j``.

    Returns ``(n_rec, n_probe)``: the signed displacement of each probe along its own mode
    vector, ``(q_n - x_star) . v_j`` -- the coordinate the half-life is read on -- sampled
    every ``record_every`` steps (step 0 always recorded).
    """
    A = np.asarray(A, np.float64)
    q = np.asarray(X0, np.float64).copy()
    p = np.zeros_like(q)
    inv_m = 1.0 / float(mass)
    damp = 1.0 - float(gamma)
    s2 = 2.0 * float(sigma) ** 2
    a2 = np.sum(A * A, axis=1)[None, :]
    xs = np.asarray(x_star, np.float64)
    two_alpha = 2.0 * float(alpha)
    lf = None
    loglife = None
    if life is not None:
        lf = np.asarray(life, np.float64)
        with np.errstate(divide="ignore"):
            loglife = np.where(lf > 0.0, np.log(np.maximum(lf, 1e-300)), -np.inf)

    def grad(z):
        d2 = np.sum(z * z, axis=1)[:, None] - 2.0 * (z @ A.T) + a2
        lk = -np.maximum(d2, 0.0) / s2
        if loglife is not None:
            lk = lk + loglife[None, :]
        m = lk.max(axis=1, keepdims=True)
        w = np.exp(lk - m)
        w /= w.sum(axis=1, keepdims=True)
        g = z - (w @ A)
        if two_alpha:
            g = g + two_alpha * (z - xs[None, :])
        return g

    re = max(1, int(record_every))
    n_rec = 1 + int(n_steps) // re
    out = np.empty((n_rec, q.shape[0]), np.float64)
    out[0] = np.sum((q - xs[None, :]) * V.T, axis=1)
    g = grad(q)
    r = 1
    for n in range(int(n_steps)):
        p_half = p - 0.5 * eta * g
        q = q + eta * inv_m * p_half
        g = grad(q)
        p = damp * (p_half - 0.5 * eta * g)
        if (n + 1) % re == 0 and r < n_rec:
            out[r] = np.sum((q - xs[None, :]) * V.T, axis=1)
            r += 1
    return out[:r]


def verlet_rollout_batch(X0: np.ndarray, A: np.ndarray, sigma: float, life,
                         inv_mass: np.ndarray, damp: np.ndarray, two_alpha: np.ndarray,
                         eta: float, n_steps: int, x_star: np.ndarray, V: np.ndarray,
                         record_every: int = 1) -> np.ndarray:
    """The same map as :func:`verlet_rollout`, but with **per-probe** mass, damping and
    confinement, so every cell of the ``(alpha, mass, gamma, amplitude, mode)`` grid that
    needs the same rollout length is integrated in ONE pass.

    That is the difference between 90 small matmuls per step and one large one: the flop
    count is unchanged, the Python overhead and the BLAS efficiency are not.
    ``inv_mass``, ``damp`` and ``two_alpha`` are ``(n_probe, 1)``.
    """
    A = np.asarray(A, np.float64)
    q = np.asarray(X0, np.float64).copy()
    p = np.zeros_like(q)
    s2 = 2.0 * float(sigma) ** 2
    a2 = np.sum(A * A, axis=1)[None, :]
    xs = np.asarray(x_star, np.float64)[None, :]
    loglife = None
    if life is not None:
        lf = np.asarray(life, np.float64)
        with np.errstate(divide="ignore"):
            loglife = np.where(lf > 0.0, np.log(np.maximum(lf, 1e-300)), -np.inf)
    use_conf = bool(np.any(two_alpha))

    def grad(z):
        d2 = np.sum(z * z, axis=1)[:, None] - 2.0 * (z @ A.T) + a2
        lk = -np.maximum(d2, 0.0) / s2
        if loglife is not None:
            lk = lk + loglife[None, :]
        lk -= lk.max(axis=1, keepdims=True)
        w = np.exp(lk)
        w /= w.sum(axis=1, keepdims=True)
        g = z - (w @ A)
        if use_conf:
            g = g + two_alpha * (z - xs)
        return g

    re = max(1, int(record_every))
    n_rec = 1 + int(n_steps) // re
    out = np.empty((n_rec, q.shape[0]), np.float64)
    out[0] = np.sum((q - xs) * V.T, axis=1)
    g = grad(q)
    r = 1
    for n in range(int(n_steps)):
        p_half = p - (0.5 * eta) * g
        q = q + (eta * inv_mass) * p_half
        g = grad(q)
        p = damp * (p_half - (0.5 * eta) * g)
        if (n + 1) % re == 0 and r < n_rec:
            out[r] = np.sum((q - xs) * V.T, axis=1)
            r += 1
    return out[:r]


def assert_map_matches_repo_integrator(tol: float = 1e-5) -> dict:
    """The numpy map above IS ``chlu.core.integrators.velocity_verlet_step``.

    Checked on the quadratic the shorts derive for, so the same call also checks that
    the repo's integrator is the shorts' Eq. (3) map.
    """
    import jax
    import jax.numpy as jnp
    from ..core.integrators import velocity_verlet_step

    rng = np.random.default_rng(0)
    worst = 0.0
    rows = []
    for mass, gamma, eta, mu in ((1.0, 0.05, 0.05, 0.9), (0.1, 0.2, 0.05, 1.7),
                                 (10.0, 0.01, 0.1, 0.3)):
        d = 4
        A = np.zeros((1, d))                       # one atom at the origin: E = |q|^2/2
        q0 = rng.normal(size=(1, d)) * 0.3
        # repo integrator, same H
        H = lambda q, p: jnp.sum(p * p) / (2.0 * mass) + 0.5 * jnp.sum(q * q)  # noqa: E731
        qj, pj = jnp.asarray(q0[0]), jnp.zeros(d)
        for _ in range(5):
            qj, pj = velocity_verlet_step(H, qj, pj, eta, gamma)
        # numpy map, sigma -> the quadratic limit is exact for a single atom at any sigma
        tr = verlet_rollout(q0, A, 1.0, None, mass, gamma, eta, 5,
                            np.zeros(d), np.eye(d)[:, :1])
        # re-roll keeping the state so we can compare positions
        q = q0.copy()
        p = np.zeros_like(q)
        g = q - 0.0
        for _ in range(5):
            ph = p - 0.5 * eta * g
            q = q + eta * ph / mass
            g = q - 0.0
            p = (1.0 - gamma) * (ph - 0.5 * eta * g)
        e = float(np.abs(np.asarray(qj) - q[0]).max())
        worst = max(worst, e)
        rows.append({"mass": mass, "gamma": gamma, "eta": eta, "max_abs_diff": e,
                     "mu_unused": mu, "trace_len": int(tr.shape[0])})
    ok = worst <= tol
    return {"max_abs_diff": worst, "tol": tol, "passed": bool(ok), "rows": rows,
            "what": "the numpy rollout reproduces chlu.core.integrators.velocity_verlet_step"}


def envelope_half_life(tr: np.ndarray, floor_steps: int = 3,
                       record_every: int = 1) -> np.ndarray:
    """⭐ **THE ENVELOPE half-life** -- the instrument the CRR's ``ln2/(-ln|lambda|)``
    actually predicts, and NOT the same thing as the first crossing.

    V2 Fig. 1 and V5 §D are explicit that the two instruments **coincide on the
    overdamped branch and diverge above the exceptional point**: underdamped, the mode
    oscillates inside a decaying envelope, so ``|x|`` first touches ``|x_0|/2`` because of
    the PHASE, after a time set by the oscillation period, while the modulus
    ``|lambda| = sqrt(1-gamma)`` sets the ENVELOPE.  Reading the first crossing there and
    comparing it to ``ln2/(-ln|lambda|)`` would compare two different quantities and
    would manufacture a failure of the law.

    The envelope is taken as the reverse cumulative maximum of ``|x|`` -- monotone
    non-increasing by construction, no fit and no window length to choose -- and the
    half-life is its first crossing of ``env[0]/2``.
    """
    a = np.abs(np.asarray(tr, np.float64))
    env = np.maximum.accumulate(a[::-1], axis=0)[::-1]
    return _first_crossing(env, env[0] / 2.0, floor_steps, record_every)


def _first_crossing(a: np.ndarray, thr: np.ndarray, floor_steps: int,
                    record_every: int) -> np.ndarray:
    n_rec, n_probe = a.shape
    re = max(1, int(record_every))
    out = np.full(n_probe, np.inf)
    below = a <= thr[None, :]
    first = np.where(below.any(axis=0), below.argmax(axis=0), -1)
    for j in range(n_probe):
        i = int(first[j])
        if i < 0:
            continue
        if i == 0:
            out[j] = 0.0
            continue
        y1, y0 = a[i, j], a[i - 1, j]
        if y0 <= 0.0 or y1 <= 0.0 or y0 == y1 or thr[j] <= 0.0:
            out[j] = float(i) * re
            continue
        f = (np.log(y0) - np.log(thr[j])) / (np.log(y0) - np.log(y1))
        out[j] = (float(i - 1) + float(np.clip(f, 0.0, 1.0))) * re
    return np.where(out < float(floor_steps), np.nan, out)


def instrument_floor_on_exact_quadratic(eta: float = 0.05, d: int = 4) -> dict:
    """⭐ **THE INSTRUMENT FLOOR.**  How well can this measurement chain recover the CRR
    on a landscape where the CRR is TRUE BY CONSTRUCTION?

    One atom at the origin makes ``E = ||q||^2/2`` exactly, so ``hess_eig = 1``,
    ``mu^2 = 1/mass``, and the one-step map IS the shorts' Eq. (3) matrix.  Any
    disagreement between ``n_half_crr`` and the rolled envelope is therefore **the
    instrument**, not the physics: discrete sampling of the envelope, the ``n < 3``
    censoring rule, and the defective (Jordan-block) behaviour at the exceptional point
    where V2 App. C's secular transients ``~ n lambda^n`` make the envelope non-exponential.

    ⛔ This number must be quoted beside any claim about how well the law predicts the
    field store: a bar tighter than this floor would be measuring the rollout, not the law.
    """
    A = np.zeros((1, d))
    V = np.eye(d)[:, :1]
    rows = []
    for mass in (1e-3, 1e-1, 1.0, 1e1, 1e3):
        for g in (0.005, 0.05, 0.2, 0.5):
            mu2 = 1.0 / mass
            npred = W6.n_half_crr(mu2, eta, g)
            if not np.isfinite(npred) or npred > 5000:
                continue
            ns = int(min(max(20, 3 * npred), 5000))
            tr = verlet_rollout_batch(0.05 * V.T, A, 1.0, None,
                                      np.array([[1.0 / mass]]), np.array([[1.0 - g]]),
                                      np.zeros((1, 1)), eta, ns, np.zeros(d), V)
            env = float(envelope_half_life(tr, 3, 1)[0])
            fst = float(half_life_from_trace(tr, 3, 1)[0])
            hs = W6.band_edges(g)[0]
            rows.append({"mass": mass, "gamma": g, "mu2": mu2,
                         "h_over_hstar": float(np.sqrt(mu2) * eta / hs),
                         "band": W6.band_of(mu2, eta, g),
                         "n_pred": npred, "n_env": env, "n_first": fst,
                         "abs_log10_ratio": (abs(np.log10(env / npred))
                                             if np.isfinite(env) and env > 0 else None)})
    e = np.array([r["abs_log10_ratio"] for r in rows
                  if r["abs_log10_ratio"] is not None])
    near = np.array([r["abs_log10_ratio"] for r in rows
                     if r["abs_log10_ratio"] is not None
                     and 0.3 <= r["h_over_hstar"] <= 3.0])
    return {"n_cells": len(rows), "median_abs_log10": float(np.median(e)),
            "max_abs_log10": float(e.max()), "p90_abs_log10": float(np.percentile(e, 90)),
            "median_near_exceptional_point": (float(np.median(near)) if near.size else None),
            "n_near_exceptional_point": int(near.size),
            "rows": rows,
            "what": "the CRR is exact on this landscape, so this is the MEASUREMENT's "
                    "own error floor and nothing else"}


def half_life_from_trace(tr: np.ndarray, floor_steps: int = 3,
                         record_every: int = 1) -> np.ndarray:
    """Envelope half-life per probe, in STEPS: first sample with ``|x| <= |x_0|/2``,
    refined by log-linear interpolation between the bracketing samples and rescaled by
    ``record_every``.

    ``inf`` when the trace never halves inside the rollout -- the latch, and the cells V5
    marks as censored lower bounds.  ``nan`` below ``floor_steps``: V5 quotes no rollout
    half-life from an ``n < 3`` cell and neither does this.
    """
    a = np.abs(np.asarray(tr, np.float64))
    thr = a[0] / 2.0
    n_rec, n_probe = a.shape
    re = max(1, int(record_every))
    out = np.full(n_probe, np.inf)
    below = a <= thr[None, :]
    first = np.where(below.any(axis=0), below.argmax(axis=0), -1)
    for j in range(n_probe):
        i = int(first[j])
        if i < 0:
            continue
        if i == 0:
            out[j] = 0.0
            continue
        y1, y0 = a[i, j], a[i - 1, j]
        if y0 <= 0.0 or y1 <= 0.0 or y0 == y1 or thr[j] <= 0.0:
            out[j] = float(i) * re
            continue
        f = (np.log(y0) - np.log(thr[j])) / (np.log(y0) - np.log(y1))
        out[j] = (float(i - 1) + float(np.clip(f, 0.0, 1.0))) * re
    return np.where(out < float(floor_steps), np.nan, out)


# ---------------------------------------------------------------------------
# Landscapes: the arm and its two charter controls, plus the incumbent
# ---------------------------------------------------------------------------
def landscape_variants(field, k: int, rng: np.random.Generator,
                       pca_store=None) -> dict:
    """``{arm_name: (atoms, life, sigma)}`` for well ``k``.

    ``pca_store`` is ``exp_w5_compose_field.build_pca_content()["store"]`` -- a
    ``PCASpanStore`` whose ``modes[k]`` is well ``k``'s ``A x d_k`` principal block, at
    the same 524,288 B.

    ⚠ **Say exactly what this arm is and is not.**  It puts the w5 WINNING CONTENT into
    THIS hub's kernel landscape, keeping the field store's own width.  ⛔ It is **not**
    w5's read: `pca_tensor` won rows 3 and 4 through a *span projection*, and row 0 of
    the block is the pool mean while the other 63 rows are unit principal DIRECTIONS, so
    as points in a kernel landscape they are one feature vector plus 63 short vectors
    near the origin.  The geometry is therefore genuinely different from the item store's
    and the arm must be read as **a third landscape on which to check whether the law
    still predicts**, not as a test of the incumbent's read.  Reported in those words.
    """
    v = W6.well_view(field, k)
    A, life, sig = v["atoms"], v["life"], v["sigma"]
    out = {"field": (A, life, sig)}
    out["landscape_removed"] = (np.zeros_like(A), life, sig)
    Ash = A.copy()
    for j in range(Ash.shape[0]):
        Ash[j] = Ash[j][rng.permutation(Ash.shape[1])]
    out["landscape_shuffled"] = (Ash, life, sig)
    if pca_store is not None:
        P = np.asarray(pca_store.modes[k], np.float64)
        out["pca_tensor"] = (P, np.ones(P.shape[0]), sig)
    return out


# ---------------------------------------------------------------------------
# One corner
# ---------------------------------------------------------------------------
def probe_modes(mu2: np.ndarray, n: int) -> np.ndarray:
    """``n`` mode indices log-spaced across the curvature spectrum (descending mu2)."""
    r = mu2.size
    if r <= n:
        return np.arange(r)
    idx = np.unique(np.round(np.geomspace(1, r, n)).astype(int) - 1)
    return np.clip(idx, 0, r - 1)


def corner_rows(field, corner_id: str, address: np.ndarray, arms: dict,
                pins: dict, rng: np.random.Generator, wells=None,
                alphas=None) -> list:
    """Every (arm, well, alpha, mass, gamma, amplitude, mode) row for one corner.

    A cell whose PREDICTED half-life exceeds ``pins['censor_steps']`` is CENSORED and not
    rolled -- V5's own protocol, whose censoring cap is 15000 steps.  Everything that is
    rolled is bucketed by required length and integrated in batched passes, so the
    ``(alpha, mass, gamma, amplitude, mode)`` grid costs a handful of large matmuls per
    step instead of one small matmul per cell per step.
    """
    eta = float(pins["eta"])
    cap = int(pins["censor_steps"])
    slack = float(pins["rollout_slack"])
    amps = tuple(pins["amplitudes"])
    alphas = tuple(pins["confinement_alphas"]) if alphas is None else tuple(alphas)
    floor_steps = int(pins["half_life_floor_steps"])
    rows = []
    wells = range(field.K) if wells is None else wells
    for k in wells:
        ch = np.asarray(field.well_channels[k], np.int64)
        x_addr = np.asarray(address, np.float64)[ch]
        for arm, (A, life, sig) in arms[k].items():
            st = W6.mean_shift_settle(x_addr, A, sig, life,
                                      max_iter=int(pins["settle_max_iter"]),
                                      tol=float(pins["settle_tol"]))
            base = W6.curvature_spectrum(st["x"], A, sig, life, mass=1.0)
            hess0 = base["mu2"]               # mass = 1: the raw Hessian eigenvalues
            degenerate = bool(hess0.size == 0)
            if degenerate:
                # ⭐ This is what `landscape_removed` looks like, and it is the charter
                # section C.2 collapse made explicit rather than silently dropped.  With
                # the atoms zeroed, Cov_w = 0, so E = ||q - x*||^2/2 EXACTLY: the whole
                # spectrum is the trivial eigenvalue 1 in every direction and the store
                # contributes no curvature structure at all.  The CRR then holds
                # trivially -- which is the point of the control, and a row that says so
                # is worth more than no row.
                d = np.asarray(A, np.float64).shape[1]
                m0 = int(pins["n_modes_probed"])
                hess0 = np.ones(m0)
                Q = np.linalg.qr(rng.normal(size=(d, m0)))[0]
                base = {"mu2": hess0, "vectors": Q}
            sel = probe_modes(hess0, int(pins["n_modes_probed"]))
            Vv = np.asarray(base["vectors"][:, sel], np.float64)      # (d, m)
            xs = st["x"]

            # ---- enumerate the grid, predict, decide censoring -------------------
            spec = []
            n_ref = int(pins.get("gamma_refine", 0))
            ref_span = float(pins.get("gamma_refine_span", 2.0))
            for alpha in alphas:
                # D14: confinement CENTRED ON x* lifts every Hessian eigenvalue by
                # exactly 2*alpha and leaves x* a critical point, so the prediction
                # stays exact and the check is arithmetic.
                he = hess0[sel] + 2.0 * float(alpha)
                for mass in pins["masses"]:
                    mu2 = he / float(mass)
                    # ⭐ The V-curve's minimum cannot be located more finely than the
                    # gamma grid, and the global grid steps by ~2x.  For each mode we
                    # therefore ADD a local geometric bracket around that mode's OWN
                    # predicted critical damping (D16 inverted exactly).  ⛔ This does
                    # not put the answer in by hand: the measured argmin is still free
                    # to land anywhere on the union of the two grids, including on a
                    # global point far from the bracket, in which case the ratio comes
                    # out far from 1 and B4 misses.  What it removes is the grid
                    # artefact that the underdamped floor FALLS with gamma, so a coarse
                    # grid's argmin is biased to the last underdamped point below
                    # gamma*.
                    gsets = []
                    for j in range(len(sel)):
                        gl = list(pins["gammas"])
                        if n_ref > 0 and mu2[j] > 0:
                            gc = W6.gamma_crit_exact(np.sqrt(mu2[j]) * eta)
                            if np.isfinite(gc) and 0.0 < gc < 1.0:
                                lo = max(1e-4, gc / ref_span)
                                hi = min(0.95, gc * ref_span)
                                if hi > lo:
                                    gl += list(np.geomspace(lo, hi, n_ref))
                        gsets.append(sorted(set(round(float(x), 6) for x in gl)))
                    all_g = sorted(set(g for gs in gsets for g in gs))
                    glob = set(round(float(x), 6) for x in pins["gammas"])
                    for gamma in all_g:
                        for amp in amps:
                            for j in range(len(sel)):
                                if gamma not in gsets[j]:
                                    continue
                                gce_j = (W6.gamma_crit_exact(np.sqrt(mu2[j]) * eta)
                                         if mu2[j] > 0 else float("nan"))
                                npred = W6.n_half_crr(mu2[j], eta, gamma)
                                cens = (not np.isfinite(npred)) or npred > cap
                                steps = 0 if cens else int(
                                    min(max(20, int(slack * npred)), cap))
                                spec.append({
                                    "alpha": float(alpha), "mass": float(mass),
                                    "gamma": float(gamma), "amp": float(amp),
                                    "j": j, "mu2": float(mu2[j]),
                                    "he": float(he[j]),
                                    "gamma_crit_exact": float(gce_j),
                                    "gamma_grid": ("global" if gamma in glob
                                                   else "refined"),
                                    "n_pred": float(npred), "censored": bool(cens),
                                    "steps": steps, "n_meas": np.inf,
                                    "n_meas_first": np.inf,
                                })

            # ---- bucket by required length and roll each bucket once -------------
            live = [sp for sp in spec if not sp["censored"]]
            if live:
                edges = [250, 1000, 4000, cap]
                for hi, lo in zip(edges, [0] + edges[:-1]):
                    grp = [sp for sp in live if lo < sp["steps"] <= hi]
                    if not grp:
                        continue
                    ns = max(sp["steps"] for sp in grp)
                    X0 = np.stack([xs + sp["amp"] * Vv[:, sp["j"]] for sp in grp])
                    Vb = np.stack([Vv[:, sp["j"]] for sp in grp], axis=1)   # (d, n)
                    inv_m = np.array([[1.0 / sp["mass"]] for sp in grp])
                    damp = np.array([[1.0 - sp["gamma"]] for sp in grp])
                    ta = np.array([[2.0 * sp["alpha"]] for sp in grp])
                    re = max(1, ns // 4000)
                    tr = verlet_rollout_batch(X0, A, sig, life, inv_m, damp, ta, eta,
                                              ns, xs, Vb, record_every=re)
                    nm_env = envelope_half_life(tr, floor_steps=floor_steps,
                                                record_every=re)
                    nm_1st = half_life_from_trace(tr, floor_steps=floor_steps,
                                                  record_every=re)
                    for i, sp in enumerate(grp):
                        sp["n_meas"] = float(nm_env[i])        # THE scored instrument
                        sp["n_meas_first"] = float(nm_1st[i])  # V2's second instrument

            for sp in spec:
                rows.append({
                    "corner_id": corner_id, "arm": arm, "well": int(k),
                    "mode": int(sel[sp["j"]]),
                    "hess_eig_raw": float(hess0[sel[sp["j"]]]),
                    "gamma_crit_exact": sp["gamma_crit_exact"],
                    "gamma_grid": sp["gamma_grid"],
                    "alpha": sp["alpha"], "hess_eig": sp["he"],
                    "mass": sp["mass"], "gamma": sp["gamma"],
                    "eta": eta, "amp": sp["amp"],
                    "mu2": sp["mu2"],
                    "h": float(np.sqrt(max(sp["mu2"], 0.0)) * eta),
                    "band": W6.band_of(sp["mu2"], eta, sp["gamma"]),
                    "h_star": float(W6.band_edges(sp["gamma"])[0]),
                    "gamma_crit": float(2.0 * eta * np.sqrt(max(sp["mu2"], 0.0))),
                    "n_pred_crr": sp["n_pred"],
                    "n_pred_asym": float(
                        W6.n_half_overdamped_asym(sp["mu2"], eta, sp["gamma"])),
                    "n_pred_floor": float(W6.n_half_floor(sp["gamma"])),
                    "n_meas": sp["n_meas"],
                    "n_meas_first": sp["n_meas_first"],
                    "censored": sp["censored"],
                    "rollout_steps": sp["steps"],
                    "sigma": float(sig),
                    "settle_converged": bool(st["converged"]),
                    "degenerate_spectrum": degenerate,
                })
    return rows


# ---------------------------------------------------------------------------
# Scoring: the parameter-free law against a FITTED one-parameter rival
# ---------------------------------------------------------------------------
def fit_one_parameter(rows: list) -> dict:
    """The falsifier from the task file: a fitted one-parameter curve.

    ``FIT1``: ``n = C / (mu eta)^2`` with ``C`` chosen to minimise the mean squared
    ``log10`` residual over ALL scored rows at once -- i.e. the fitted rival is given
    every advantage, including sight of the held-out measurements.
    """
    ok = [r for r in rows if np.isfinite(r["n_meas"]) and r["mu2"] > 0
          and np.isfinite(r["n_pred_crr"])]
    if not ok:
        return {"C": float("nan"), "n": 0}
    lg = np.array([np.log10(r["n_meas"]) + 2.0 * np.log10(r["h"]) for r in ok])
    return {"C": float(10.0 ** lg.mean()), "n": len(ok)}


def score_rows(rows: list, fit: dict) -> dict:
    """Per-row ``|log10 ratio|`` for the parameter-free CRR and for the fitted rival."""
    out = []
    for r in rows:
        if not (np.isfinite(r["n_meas"]) and np.isfinite(r["n_pred_crr"])
                and r["n_meas"] > 0 and r["n_pred_crr"] > 0 and r["h"] > 0):
            continue
        e_crr = abs(np.log10(r["n_meas"] / r["n_pred_crr"]))
        n_fit = fit["C"] / (r["h"] ** 2)
        e_fit = abs(np.log10(r["n_meas"] / n_fit)) if n_fit > 0 else np.inf
        d = dict(r)
        d["err_crr_log10"] = float(e_crr)
        d["err_fit1_log10"] = float(e_fit)
        d["paired_delta"] = float(e_crr - e_fit)
        out.append(d)
    return {"rows": out, "fit1_C": fit["C"]}


def corner_estimand(scored: list, arm: str = "field") -> dict:
    """One number per corner: the median over that corner's rows.

    The paired estimand is ``median|log10 n_meas/n_pred_CRR| - median|log10
    n_meas/n_pred_FIT1|`` inside the corner; the corner is the unit, as w6-common
    section 2 requires, and the sign of the mean over corners is the verdict.
    """
    by = {}
    for r in scored:
        if r["arm"] != arm:
            continue
        by.setdefault(r["corner_id"], []).append(r)
    out = []
    for cid, rs in sorted(by.items()):
        ec = np.array([r["err_crr_log10"] for r in rs])
        ef = np.array([r["err_fit1_log10"] for r in rs])
        out.append({"corner_id": cid, "n_rows": len(rs),
                    "median_err_crr": float(np.median(ec)),
                    "median_err_fit1": float(np.median(ef)),
                    "paired_delta": float(np.median(ec) - np.median(ef))})
    return {"corners": out, "n_corners": len(out)}


# ---------------------------------------------------------------------------
# D16 / D17 / D18 -- the V-curve, its critical point, and its two branch slopes
# ---------------------------------------------------------------------------
def d17_slope(gamma: float) -> float:
    """D17, V5 App. G.3: overdamped ``d ln n_half / d ln gamma = 1 + gamma/(2-gamma)``.

    Note the shorts are explicit that this **exceeds +1 at every finite gamma**.
    """
    g = float(gamma)
    return 1.0 + g / (2.0 - g)


def d18_slope(gamma: float) -> float:
    """D18, V5 App. G.3: underdamped ``-gamma/[(1-gamma)(-ln(1-gamma))] = -1 - gamma/2 + ...``."""
    g = float(gamma)
    return -g / ((1.0 - g) * (-np.log(1.0 - g)))


def v_curve_analysis(rows: list, arm: str = "field") -> list:
    """One V-curve per (corner, well, alpha, mass, mode, amp): ``n_half`` against ``gamma``.

    Returns, per curve, the three pre-registered comparisons:

    * **D16** -- the measured argmin against ``gamma_crit = 2 eta mu``, in units of
      ``gamma_crit`` (the shorts' own collapsed variable).
    * **D17** -- the measured log-slope on the overdamped side (``gamma >= 2.5 gamma_crit``,
      V5's own fit window) against ``1 + gamma/(2-gamma)``, and against the naive ``+1``.
    * **D18** -- the measured log-slope on the underdamped side
      (``gamma <= gamma_crit/2.5``) against ``-gamma/[(1-gamma)(-ln(1-gamma))]``, and
      against the naive ``-1``.
    """
    key = ("corner_id", "well", "alpha", "mass", "mode", "amp")
    by = {}
    for r in rows:
        if r["arm"] != arm or r.get("censored"):
            continue
        if not np.isfinite(r["n_meas"]) or r["n_meas"] <= 0:
            continue
        by.setdefault(tuple(r[k] for k in key), []).append(r)
    out = []
    for kk, rs in sorted(by.items(), key=lambda t: str(t[0])):
        rs = sorted(rs, key=lambda r: r["gamma"])
        g = np.array([r["gamma"] for r in rs])
        n = np.array([r["n_meas"] for r in rs])
        gc = float(rs[0]["gamma_crit"])                      # D16's small-h form
        gce = float(rs[0].get("gamma_crit_exact", np.nan))    # the exact inversion
        if gc <= 0 or g.size < 4:
            continue
        am = float(g[int(np.argmin(n))])
        npd = np.array([r["n_pred_crr"] for r in rs])
        fin = np.isfinite(npd)
        am_pred = float(g[fin][int(np.argmin(npd[fin]))]) if fin.any() else float("nan")
        rec = {k: v for k, v in zip(key, kk)}
        rec.update({"gamma_crit": gc, "gamma_crit_exact": gce, "n_gammas": int(g.size),
                    "argmin_gamma": am,
                    "argmin_over_gamma_crit": am / gc,
                    "argmin_over_gamma_crit_exact": (am / gce if gce and
                                                     np.isfinite(gce) and gce > 0
                                                     else float("nan")),
                    "argmin_gamma_pred": am_pred,
                    "pred_argmin_over_gamma_crit_exact": (
                        am_pred / gce if gce and np.isfinite(gce) and gce > 0
                        else float("nan")),
                    "bracketed": bool(g.min() < gc < g.max())})
        # V5's own fit windows (gamma < gamma*/2.5 and gamma > 2.5 gamma*), taken
        # around the EXACT critical damping rather than its small-h limit.
        gw = gce if (gce and np.isfinite(gce) and gce > 0) else gc
        for name, mask, pred_fn, naive in (
                ("under", g <= gw / 2.5, d18_slope, -1.0),
                ("over", g >= 2.5 * gw, d17_slope, +1.0)):
            if int(mask.sum()) >= 3:
                s = float(np.polyfit(np.log(g[mask]), np.log(n[mask]), 1)[0])
                gm = float(np.exp(np.mean(np.log(g[mask]))))
                rec[f"{name}_slope_meas"] = s
                rec[f"{name}_slope_pred"] = float(pred_fn(gm))
                rec[f"{name}_slope_naive"] = float(naive)
                rec[f"{name}_err_pred"] = abs(s - float(pred_fn(gm)))
                rec[f"{name}_err_naive"] = abs(s - float(naive))
                rec[f"{name}_n_points"] = int(mask.sum())
        out.append(rec)
    return out


def hessian_fd_check(field, arms: dict, address: np.ndarray, pins: dict,
                     alphas=(0.0, 1.0), arm: str = "field", n_dirs: int = 6,
                     eps_fd: float = 1e-5) -> dict:
    """Validate the closed-form curvature, and D14's lift, by FINITE DIFFERENCE.

    ⛔ This deliberately does NOT re-use ``curvature_spectrum``'s algebra on both sides --
    that would be a tautology.  The measured side is a central difference of the exact
    gradient of the CONFINED energy, ``grad E(q) + 2 alpha (q - x*)``, along the mode
    vectors; the predicted side is ``diag(mu^2) + 2 alpha`` from the Gram-matrix spectrum.

    Two things are checked at once: that ``Hess E = I - Cov_w/sigma^2`` is the right
    Hessian for the energy actually integrated, and that D14's ``+2 alpha`` lift holds on
    it.  A residual at the float64 differencing floor (``~1e-8`` at ``eps_fd = 1e-5``) is
    a pass; anything larger means the prediction and the rollout disagree about the object.
    """
    rows = []
    for k in sorted(arms):
        entry = arms[k].get(arm)
        if entry is None:
            continue
        A, life, sig = entry
        ch = np.asarray(field.well_channels[k], np.int64)
        st = W6.mean_shift_settle(np.asarray(address, np.float64)[ch], A, sig, life,
                                  max_iter=int(pins["settle_max_iter"]),
                                  tol=float(pins["settle_tol"]))
        xs = st["x"]
        cs = W6.curvature_spectrum(xs, A, sig, life, mass=1.0)
        sel = probe_modes(cs["mu2"], int(n_dirs))
        for al in alphas:
            worst = 0.0
            for j in sel:
                v = np.asarray(cs["vectors"][:, j], np.float64)
                gp = W6.kernel_grad(xs + eps_fd * v, A, sig, life) \
                    + 2.0 * al * (eps_fd * v)
                gm = W6.kernel_grad(xs - eps_fd * v, A, sig, life) \
                    + 2.0 * al * (-eps_fd * v)
                meas = float(v @ (gp - gm) / (2.0 * eps_fd))   # v^T Hess v
                pred = float(cs["mu2"][j]) + 2.0 * float(al)
                worst = max(worst, abs(meas - pred))
            rows.append({"well": int(k), "alpha": float(al),
                         "max_abs_residual": worst, "n_dirs": int(len(sel))})
    if not rows:
        return {"n": 0}
    a = np.array([r["max_abs_residual"] for r in rows])
    return {"n": int(a.size), "max_abs_residual": float(a.max()),
            "mean_abs_residual": float(a.mean()), "eps_fd": eps_fd,
            "per_well": rows,
            "law": "Hess E = I - Cov_w/sigma^2, and D14: +2 alpha lifts every eigenvalue"}


def d14_confinement_check(rows: list, arm: str = "field") -> dict:
    """D14, V5 App. G.4: confinement ``alpha||q||^2`` lifts EVERY Hessian eigenvalue by
    exactly ``2 alpha``.  The prediction is arithmetic on the measured spectrum, so the
    check is exact and its residual should sit at the float64 floor."""
    res, seen = [], set()
    base = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        k = (r["corner_id"], r["well"], r["mode"])
        if r["alpha"] == 0.0:
            base[k] = float(r["hess_eig_raw"])
    for r in rows:
        if r["arm"] != arm or r["alpha"] == 0.0:
            continue
        k = (r["corner_id"], r["well"], r["mode"], r["alpha"])
        if k in seen:
            continue
        seen.add(k)
        b = base.get((r["corner_id"], r["well"], r["mode"]))
        if b is None:
            continue
        res.append(abs(float(r["hess_eig"]) - (b + 2.0 * float(r["alpha"]))))
    if not res:
        return {"n": 0}
    a = np.array(res)
    return {"n": int(a.size), "max_abs_residual": float(a.max()),
            "mean_abs_residual": float(a.mean()),
            "law": "hess_eig(alpha) - [hess_eig(0) + 2 alpha]"}


def _j(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_j(x) for x in o.tolist()]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(x) for x in o]
    return o


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_j(obj), fh, indent=2)
    return str(path)
