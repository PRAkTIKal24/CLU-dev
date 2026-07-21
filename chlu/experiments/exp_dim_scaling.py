"""Experiment DIM-SCALING: is the 8-item ceiling a 2-D ring artifact or CLU's capacity?

w19 (``exp_retrieval``) built a working write -> address -> retrieve loop with
**zero learning** and measured a capacity ceiling of **8 items**. Both the engineer
and the theorist independently warned that 8 must NOT be quoted as CLU's capacity:
it is a property of the 2-D ring's *angular* resolution
(``K_max ~ 0.2 * 2*pi / sigma_theta``), whereas the theorist's packing bound
``(1 + 2R/w)^d`` is **exponential in the address dimension d**.

This module generalizes the ring to a ``d``-dimensional address ball
(``BallRegisterPotential``: K Gaussian item wells farthest-point-packed into a
flat-bottomed ball of radius R) and measures ``K_max`` vs ``d``.

⚠ **EVERYTHING HERE IS HAND-DESIGNED, NOT LEARNED** — the landscape, the site
packing and the addresses. Nothing here is evidence of an emergent capability
(N46 precedent). There is no gradient descent anywhere in this experiment.

Four measurements:

1. ``item1_k_max_vs_dim``  the deliverable: K_max at each d, with the achieved
                           site separation and the packing-bound overlay.
2. ``item2_width_sweep``   separate GEOMETRY from DIMENSION: at fixed d, vary the
                           basin width w and ask whether K_max tracks (1+2R/w)^d.
3. ``item3_regimes``       selectivity alongside K_max -> which of the theorist's
                           three capacity regimes each cell sits in.
4. ``item4_gamma``         does retrieval still REQUIRE dissipation at d > 2, and
                           does the required gamma scale with d?

The w19 fidelity criteria are used **verbatim** so the numbers are comparable:
payload-only linear-codebook read, payload retrieval error, and — mandatory on
every single cell — the **blank-landscape control**. A full-state read scores
1.000 on a blank landscape because it reads the *address* back; **any cell whose
blank control does not pass is not a measurement** and is dropped, not reported.

Runnable directly:
    uv run python -m chlu.experiments.exp_dim_scaling --quick
or via the CLI: ``chlu exp-dim-scaling [--project N] [--seed I] [--quick]``.
"""

import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.memory_potentials import (
    BallRegisterPotential,
    designed_payloads,
    designed_sites,
    site_separation,
)
from chlu.experiments.exp_retrieval import linear_codebook_read, nearest_centroid_read
from chlu.experiments.goldstone_harness import clu_with_potential


# ---------------------------------------------------------------------------
# Model construction / rollout
# ---------------------------------------------------------------------------


def build_ball_model(payloads, centers, cfg, w: Optional[float] = None) -> CHLU:
    """A CLU wired to a hand-designed ``BallRegisterPotential``.

    ``newtonian_learned`` kinetics (as in w19) so the mass vector is actually read
    by the dynamics; here the inertia is left at identity throughout — this
    experiment varies geometry, not mass.
    """
    V = BallRegisterPotential(
        payloads,
        centers,
        R=cfg.R + cfg.wall_margin,
        w=cfg.well_width if w is None else w,
        b=cfg.well_depth,
        kappa=cfg.payload_kappa,
        c_conf=cfg.c_conf,
    )
    dim = V.dim
    return clu_with_potential(
        V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )


def make_ball_queries(key, centers, n_per_item: int, cfg):
    """Noisy queries: ``n_per_item`` jittered addresses around each item site.

    ``x0 = c_k + jitter``, and — the anti-decoration guard, carried over verbatim
    from w19 — the payload channel is ALWAYS launched at ``y(0) = 0`` with
    ``p_y(0) = 0``, so the payload carries no address information and a
    payload-only read cannot be reading the query back.

    ⚠ **The two noise conventions, and why both are reported.** A ``d``-dimensional
    Gaussian jitter with per-axis scale ``sigma`` has NORM ``~ sigma*sqrt(d)``, so
    "the same sigma" means a query that lands progressively further from its site
    as ``d`` grows. That confounds the dimension scaling with a query-precision
    change, and at ``d = 8`` with ``sigma = 0.15`` the typical query already lands
    outside the well's grip. ``cfg.query_noise_mode`` selects:

    ``"fixed_norm"`` (default, the w19-comparable arm)
        ``sigma_d = sigma / sqrt(d)`` so the jitter NORM is ``sigma`` at every
        ``d``. This is the apples-to-apples generalization of w19, whose ring
        address was a 1-D manifold carrying an arc-length jitter of 0.15. Query
        precision is held constant and ONLY the address dimension varies.

    ``"per_axis"``
        Literal ``sigma`` per coordinate, so precision-per-coordinate is held
        constant and the jitter norm grows as ``sqrt(d)``. Also a legitimate
        question (it is what an encoder emitting `d` independently-noisy
        coordinates would produce) and it gives a materially different answer, so
        it is reported as a second arm rather than chosen between.
    """
    K, d = centers.shape
    dim = d + 1
    k_x, k_p = jax.random.split(key, 2)
    n = K * n_per_item
    labels = np.repeat(np.arange(K), n_per_item)

    scale = cfg.query_sigma
    if cfg.query_noise_mode == "fixed_norm":
        scale = cfg.query_sigma / np.sqrt(d)
    elif cfg.query_noise_mode != "per_axis":
        raise ValueError(
            f"unknown query_noise_mode {cfg.query_noise_mode!r} "
            "(expected 'fixed_norm' or 'per_axis')"
        )

    x0 = jnp.repeat(centers, n_per_item, axis=0)
    x0 = x0 + jax.random.normal(k_x, (n, d)) * scale

    Q0 = jnp.zeros((n, dim)).at[:, :d].set(x0)
    P0 = (
        jnp.zeros((n, dim))
        .at[:, :d]
        .set(jax.random.normal(k_p, (n, d)) * cfg.query_sigma_p)
    )
    return Q0, P0, labels


@eqx.filter_jit
def _rollout_read(model, Q0, P0, steps, dt, gamma, d, tail_idx):
    """vmapped rollout returning ONLY what the read needs.

    Returns ``(payload_tail, x_final)``. Keeping the full ``(n, steps, 2*dim)``
    trajectory is what actually bites at large K — at K=2048 x 32 queries x 1200
    steps it is tens of GB — so the tail subsample and the final address are
    extracted inside the jit and the trajectory is never materialized outside it.
    """

    def one(q, p):
        traj = model(q, p, steps, dt, gamma)
        return traj[tail_idx, d], traj[-1, :d]

    return jax.vmap(one)(Q0, P0)


def rollout_read_chunked(model, Q0, P0, cfg, gamma: float, d: int):
    """Chunked ``_rollout_read`` -> (payload_tail (n, n_sub), x_final (n, d)).

    Chunks are zero-padded to a FIXED size so the jit compiles once per (d, K)
    rather than once per ragged tail chunk.
    """
    steps = cfg.steps
    start = int((1.0 - cfg.tail_frac) * steps)
    tail_idx = jnp.asarray(np.linspace(start, steps - 1, cfg.n_subsample).astype(int))

    n = Q0.shape[0]
    chunk = min(cfg.rollout_chunk, n)
    feats, xs = [], []
    for i in range(0, n, chunk):
        q, p = Q0[i : i + chunk], P0[i : i + chunk]
        pad = chunk - q.shape[0]
        if pad > 0:
            q = jnp.concatenate([q, jnp.zeros((pad,) + q.shape[1:])], axis=0)
            p = jnp.concatenate([p, jnp.zeros((pad,) + p.shape[1:])], axis=0)
        f, x = _rollout_read(model, q, p, steps, cfg.dt, gamma, d, tail_idx)
        if pad > 0:
            f, x = f[: chunk - pad], x[: chunk - pad]
        feats.append(np.asarray(f))
        xs.append(np.asarray(x))
    return np.concatenate(feats, axis=0), np.concatenate(xs, axis=0)


# ---------------------------------------------------------------------------
# The measured landscape scales (R and w are MEASURED, not assumed)
# ---------------------------------------------------------------------------


def measure_landscape_scales(cfg, d: int, w: Optional[float] = None, seed: int = 0):
    """Measure ``R`` and ``w`` FROM THE LANDSCAPE, as the task requires.

    The packing bound ``(1 + 2R/w)^d`` is stated in terms of a region radius ``R``
    and a basin width ``w``. Both are read off ``V`` here rather than taken from
    the config, so the bound is evaluated against measured geometry.

    Three scales, on an ISOLATED single well (K=1) so each is a property of the
    well and not of the packing:

    ``w_force_max``
        **The basin width.** The radius along a ray from the site at which the
        restoring force ``|grad V|`` is MAXIMAL. Beyond it the restoring force
        *decreases* — that is the landscape's own definition of the basin edge,
        and it is the scale that competes with a neighbouring well. For a
        Gaussian well it recovers the shape parameter exactly, which is what
        makes it a valid measurement rather than a redefinition.

    ``r_settle``
        Secondary diagnostic: the largest launch offset that still settles back
        at the site **within the rollout**. ⚠ On a flat-bottomed ball an isolated
        well eventually captures from almost anywhere (the Gaussian tail is weak
        but the surroundings are force-free), so this is a *settling-time* scale,
        NOT a basin width — it saturates near ``capture_max_offset`` at every
        ``d`` and must not be substituted into the packing bound. Reported
        because it is the honest operational limit on query jitter.

    ``R_wall``
        The radius at which the confining wall turns on (first radius where
        ``|grad V|`` from the confinement exceeds a small threshold) — recovers
        ``cfg.R``.
    """
    ww = cfg.well_width if w is None else w
    centers = jnp.zeros((1, d))
    V = BallRegisterPotential(
        jnp.ones(1),
        centers,
        R=cfg.R + cfg.wall_margin,
        w=ww,
        b=cfg.well_depth,
        kappa=cfg.payload_kappa,
        c_conf=cfg.c_conf,
    )
    gradV = jax.jit(jax.grad(V))

    # Probe the force along a ray from the site, with the payload channel held at
    # its EQUILIBRIUM y = s(x). Otherwise the payload spring 0.5*kappa*(y-s(x))^2
    # contributes a spurious kappa*s*s' force in the address plane and the
    # measured basin width comes out ~45% too wide (measured: 0.219 vs 0.150).
    # The particle settles onto y = s(x) within a few steps, so this is the
    # effective address-plane potential the geometry question is about.
    def force_at(r):
        x = np.zeros(d, dtype=np.float32)
        x[0] = r
        xj = jnp.asarray(x)
        y = float(V.payload_profile(xj))
        q = jnp.concatenate([xj, jnp.asarray([y], dtype=jnp.float32)])
        return float(jnp.linalg.norm(gradV(q)[:d]))

    rr = np.linspace(1e-3, 2.0 * (cfg.R + cfg.wall_margin), 600)
    fmag = np.array([force_at(r) for r in rr])

    # The profile along the ray is: zero at the well bottom, rising to a peak at
    # the basin edge, decaying to ~0 across the flat ball bottom, then the
    # confining wall. So the basin width is the FIRST LOCAL MAXIMUM scanning
    # outward — not the global argmax (which is the wall for wide wells:
    # measured 1.20 for w=0.30) and not bounded by the global argmin (which is
    # r=0, where the force vanishes by symmetry, for w >= 0.30).
    i_peak = int(np.argmax(fmag))
    for i in range(1, len(fmag) - 1):
        if fmag[i] >= fmag[i - 1] and fmag[i] > fmag[i + 1]:
            i_peak = i
            break
    w_force_max = float(rr[i_peak])

    # --- R: where the confining wall switches on (relu => exactly 0 inside) ---
    i_dead = i_peak + int(np.argmin(fmag[i_peak:]))
    R_wall = float("nan")
    for i in range(i_dead, len(rr)):
        if fmag[i] > 1e-3:
            R_wall = float(rr[i])
            break
    if not np.isfinite(R_wall):
        R_wall = float(cfg.R)

    # --- r_settle: settling-time-limited capture (secondary) ---
    model = clu_with_potential(
        V, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(cfg.capture_n_dirs, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    offsets = np.linspace(
        cfg.capture_max_offset / cfg.capture_n_offsets,
        cfg.capture_max_offset,
        cfg.capture_n_offsets,
    )
    r_settle = 0.0
    detail = []
    for delta in offsets:
        x0 = jnp.asarray(u * delta, dtype=jnp.float32)
        Q0 = jnp.zeros((cfg.capture_n_dirs, d + 1)).at[:, :d].set(x0)
        P0 = jnp.zeros((cfg.capture_n_dirs, d + 1))
        _, x_fin = rollout_read_chunked(model, Q0, P0, cfg, cfg.gamma, d)
        frac = float(np.mean(np.linalg.norm(x_fin, axis=1) < ww))
        detail.append({"offset": float(delta), "frac_captured": frac})
        if frac >= 1.0:
            r_settle = float(delta)
        else:
            break

    return {
        "d": d,
        "well_width_param": float(ww),
        "w_measured_force_max": w_force_max,
        "R_measured_wall": R_wall,
        "r_settle_time_limited": r_settle,
        "r_settle_saturated": bool(r_settle >= offsets[-1] - 1e-9),
        "capture_detail": detail,
    }


# ---------------------------------------------------------------------------
# One cell = one (d, K, gamma, w) retrieval measurement, with its blank control
# ---------------------------------------------------------------------------


def evaluate_cell(
    cfg,
    d: int,
    K: int,
    seed: int = 0,
    gamma: Optional[float] = None,
    w: Optional[float] = None,
):
    """Write K items into a d-dimensional address ball, retrieve them, score.

    Returns the w19 read battery (linear codebook read, nearest centroid, payload
    R^2, payload absolute error), the **physical selectivity** (fraction of queries
    that settled nearest their own site — addressing, independent of any decoder),
    the achieved site separation, and — mandatory — the **blank control**.
    """
    g = cfg.gamma if gamma is None else gamma
    pay = designed_payloads(K, seed=cfg.payload_seed)
    centers = designed_sites(d, K, R=cfg.R, seed=cfg.site_seed)
    sep = site_separation(centers)

    key = jax.random.PRNGKey(seed)
    n_per = int(
        np.clip(
            cfg.max_total_queries // K, cfg.min_query_per_item, cfg.n_query_per_item
        )
    )
    Q0, P0, labels = make_ball_queries(key, centers, n_per, cfg)

    out = {
        "d": d,
        "K": K,
        "gamma": float(g),
        "w": float(cfg.well_width if w is None else w),
        "n_queries": int(Q0.shape[0]),
        "n_query_per_item_effective": n_per,
        "site_sep_min_measured": sep,
        # The MEASURED radius of the region the memory actually occupies (the
        # "R" of the packing bound). Distinct from the confining-wall radius,
        # which sits wall_margin further out so jittered queries never hit it.
        "site_radius_max_measured": float(
            np.linalg.norm(np.asarray(centers), axis=1).max()
        ),
        "site_sep_min_predicted_2RKinvd": float(2.0 * cfg.R * K ** (-1.0 / d)),
        "chance": 1.0 / K,
    }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(Q0.shape[0])
    half = Q0.shape[0] // 2
    tr, te = perm[:half], perm[half:]

    for name, payloads in (("written", pay), ("blank", jnp.zeros_like(pay))):
        model = build_ball_model(payloads, centers, cfg, w=w)
        feats, x_fin = rollout_read_chunked(model, Q0, P0, cfg, g, d)

        acc_cb, r2 = linear_codebook_read(
            feats[tr], labels[tr], feats[te], labels[te], payloads
        )
        acc_nc = nearest_centroid_read(feats[tr], labels[tr], feats[te], labels[te], K)
        rec = {"acc_codebook": acc_cb, "acc_centroid": acc_nc, "payload_r2": r2}

        if name == "blank":
            # ⚠ R^2 is UNDEFINED on the blank arm and must not be reported as a
            # number. The blank codebook is all-zeros, so the regression target
            # is constant: ss_tot = 0 and `linear_codebook_read` returns
            # 1 - 0/1e-12 = 1.000. That is a degenerate 0/0, NOT a leak — and a
            # literal "blank payload_r2 = 1.000" in a results table is precisely
            # the number a reader would mistake for one. Replaced by the
            # meaningful "nothing is stored" check: the magnitude of the payload
            # channel itself, which must sit at ~0 on a blank landscape.
            rec["payload_r2"] = None
            rec["payload_r2_undefined_constant_target"] = True
            rec["payload_abs_level"] = float(np.mean(np.abs(feats[:, -1])))

        if name == "written":
            # PHYSICAL selectivity: did the query settle nearest its OWN site?
            # Decoder-free, so it separates an addressing failure from a
            # read-out failure (w19's `frac_landed_in_correct_well`).
            c = np.asarray(centers)
            d2 = ((x_fin[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            rec["selectivity"] = float(np.mean(np.argmin(d2, axis=1) == labels))
            rec["mean_dist_to_own_site"] = float(
                np.mean(np.linalg.norm(x_fin - c[labels], axis=1))
            )
            end = feats[:, -1]
            rec["payload_abs_err"] = float(
                np.mean(np.abs(end - np.asarray(payloads)[labels]))
            )
        out[name] = rec

    # ⚠ THE LOAD-BEARING GUARD. A read that scores well on a landscape with
    # NOTHING STORED is reading the address back, not the memory. w19's
    # full-state read scored 1.000 on a blank landscape for exactly this reason.
    out["blank_passes"] = bool(
        out["blank"]["acc_codebook"] <= out["chance"] + cfg.blank_margin
    )
    out["retrieved"] = bool(
        out["blank_passes"]
        and out["written"]["acc_codebook"] >= cfg.selectivity_threshold
    )
    return out


def _cell_passes(cell, cfg, criterion: str) -> bool:
    """Does a cell meet the fidelity criterion? The blank control always vetoes.

    ``"codebook"`` (default, **w19 verbatim**)
        payload-only linear-codebook read accuracy. This is the criterion the
        task specifies, and it is the one the headline curve uses.

    ``"selectivity"`` (decoder-free)
        fraction of queries that SETTLED NEAREST THEIR OWN SITE — pure
        addressing, no read-out involved.

    ⚠ **Why both are needed, and why quoting only the first may understate
    capacity.** All K items are read back through ONE SCALAR payload channel
    whose codebook values live in a fixed range ``[-1, 1]``, so their spacing
    shrinks like ``1/K`` and must eventually fall below the payload channel's own
    retrieval error — a **read-out resolution limit that has nothing to do with
    the address space**. If that happens, the codebook read fails while
    ``selectivity`` stays at 1.000 (addressing perfect, scalar channel out of
    resolution), which is the same class of estimator artifact w19 flagged (its
    one-hot probe scored 0.484 at K=4 while the payload was retrieved to 7e-4).
    Whether this regime is actually reached is an EMPIRICAL question answered by
    the run, not an assumption — both criteria are therefore always reported and
    the divergence between them is a measured output.
    """
    if not cell["blank_passes"]:
        return False
    if criterion == "codebook":
        return cell["written"]["acc_codebook"] >= cfg.selectivity_threshold
    if criterion == "selectivity":
        return cell["written"]["selectivity"] >= cfg.selectivity_threshold
    raise ValueError(f"unknown criterion {criterion!r}")


def k_max_for_dim(
    cfg,
    d: int,
    seed: int = 0,
    w: Optional[float] = None,
    verbose=True,
    criterion: str = "codebook",
    k_cap: Optional[int] = None,
):
    """Walk the K ladder until the fidelity criterion breaks -> K_max at this d."""
    cap = cfg.k_cap if k_cap is None else k_cap
    cells = []
    censored = False
    for K in cfg.k_ladder:
        if K > cap:
            censored = True
            break
        cell = evaluate_cell(cfg, d, K, seed=seed, w=w)
        cell["passes_codebook"] = _cell_passes(cell, cfg, "codebook")
        cell["passes_selectivity"] = _cell_passes(cell, cfg, "selectivity")
        cell["passes"] = _cell_passes(cell, cfg, criterion)
        cells.append(cell)
        if verbose:
            print(
                f"  d={d:2d} K={K:5d}  acc={cell['written']['acc_codebook']:.3f} "
                f"sel={cell['written']['selectivity']:.3f} "
                f"blank={cell['blank']['acc_codebook']:.3f} "
                f"({'pass' if cell['blank_passes'] else 'FAIL'}) "
                f"sep={cell['site_sep_min_measured']:.3f}",
                flush=True,
            )
        # Keep walking while EITHER criterion still passes, so one ladder yields
        # K_max under both. Stopping on the stricter one alone would truncate the
        # other and silently report it as censored.
        if not (cell["passes_codebook"] or cell["passes_selectivity"]):
            break
    else:
        censored = True

    def _kmax(flag):
        passing = [c["K"] for c in cells if c[flag]]
        return max(passing, default=0)

    k_max_cb, k_max_sel = _kmax("passes_codebook"), _kmax("passes_selectivity")
    k_top = max(c["K"] for c in cells)
    return {
        "d": d,
        "criterion": criterion,
        "k_max": _kmax(f"passes_{criterion}"),
        "k_max_codebook": k_max_cb,
        "k_max_selectivity": k_max_sel,
        # A cell that ran out of ladder / hit the compute cap without ever
        # failing is CENSORED: k_max is a lower bound, not a measurement.
        "censored": bool(censored and _kmax(f"passes_{criterion}") == k_top),
        "censored_codebook": bool(censored and k_max_cb == k_top),
        "censored_selectivity": bool(censored and k_max_sel == k_top),
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Item 1 — the deliverable: K_max vs d
# ---------------------------------------------------------------------------


def item1_k_max_vs_dim(cfg, seed: int = 0):
    """K_max vs d, with the MEASURED R and w and the packing-bound overlay."""
    rows = []
    for d in cfg.dims:
        scales = measure_landscape_scales(cfg, d, seed=seed)
        res = k_max_for_dim(cfg, d, seed=seed)
        w_meas = scales["w_measured_force_max"]
        # R of the packing bound = the radius of the region the SITES occupy,
        # measured from the site set (NOT the confining-wall radius, which sits
        # wall_margin further out and bounds no memory).
        R_meas = max(
            (c["site_radius_max_measured"] for c in res["cells"]), default=cfg.R
        )
        # The packing bound exactly as the theorist states it, evaluated with the
        # MEASURED R and w (task: "report R and w as measured, not assumed").
        pack = (1.0 + 2.0 * R_meas / max(w_meas, 1e-9)) ** d

        # My pre-registered resolution-floor law: sites must be separated by more
        # than the query noise can bridge AND more than one basin diameter.
        delta_req = max(5.0 * cfg.query_sigma, 2.0 * w_meas)
        # The same law with the jitter NORM substituted for the per-axis sigma.
        # In "per_axis" mode the norm grows as sigma*sqrt(d) (a d-dependence I
        # did NOT register and should have); in the default "fixed_norm" mode the
        # norm is sigma at every d and the two forms coincide. Both are recorded
        # so the prereg can be scored against whichever arm is being read.
        sigma_norm = (
            cfg.query_sigma
            if cfg.query_noise_mode == "fixed_norm"
            else cfg.query_sigma * np.sqrt(d)
        )
        delta_req_sqrtd = max(5.0 * sigma_norm, 2.0 * w_meas)
        rows.append(
            {
                **res,
                **{k: v for k, v in scales.items() if k != "d"},
                "R_assumed": cfg.R,
                "R_sites_measured": float(R_meas),
                "query_noise_mode": cfg.query_noise_mode,
                "query_jitter_norm": float(sigma_norm),
                "packing_bound_1plus2Rovw": float(pack),
                "resolution_floor_delta_req": float(delta_req),
                "prereg_kmax_resolution_law": float((2.0 * R_meas / delta_req) ** d),
                "resolution_floor_delta_req_sqrtd": float(delta_req_sqrtd),
                "kmax_resolution_law_sqrtd": float(
                    (2.0 * R_meas / delta_req_sqrtd) ** d
                ),
            }
        )

    fits = {
        key: _fit_growth(
            np.array([r["d"] for r in rows], dtype=float),
            np.array([max(r[key], 1) for r in rows], dtype=float),
            np.array([bool(r[cens]) for r in rows]),
        )
        for key, cens in (
            ("k_max_codebook", "censored_codebook"),
            ("k_max_selectivity", "censored_selectivity"),
        )
    }
    return {"rows": rows, "fit": fits["k_max_codebook"], "fits": fits}


def _fit_growth(ds, ks, censored=None):
    """Exponential (A^d) vs polynomial (d^alpha) fit of a capacity curve.

    ⚠ **Censored points MUST be excluded.** A cell that hit the compute cap
    without ever failing yields a LOWER BOUND on K_max, not a measurement, and
    including it flattens the curve at exactly the dimensions where growth is
    fastest. Measured consequence of getting this wrong on the shipped run:
    including the censored d=12 and d=16 points (both pinned at the cap, 2048)
    dropped the fitted base from A=2.13 to A=1.51, collapsed the exponential
    R^2 from 0.986 to 0.844, and **inverted the model comparison** so a
    polynomial appeared to fit better -- i.e. it would have reversed the
    headline conclusion of the experiment.
    """
    fit = {}
    ok = ks > 1
    if censored is not None:
        ok = ok & ~np.asarray(censored, dtype=bool)
        fit["n_censored_excluded"] = int(np.sum(censored))
    if ok.sum() >= 2:
        # log K_max = d * log A  (+ intercept): exponential growth test
        A_mat = np.stack([ds[ok], np.ones(ok.sum())], axis=1)
        coef, *_ = np.linalg.lstsq(A_mat, np.log(ks[ok]), rcond=None)
        pred = A_mat @ coef
        ss_res = float(np.sum((np.log(ks[ok]) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ks[ok]) - np.log(ks[ok]).mean()) ** 2))
        fit["exponential_base_A"] = float(np.exp(coef[0]))
        fit["exponential_intercept"] = float(coef[1])
        fit["exponential_r2"] = 1.0 - ss_res / (ss_tot + 1e-12)
        # Competing model: polynomial K_max ~ d^alpha
        B = np.stack([np.log(ds[ok]), np.ones(ok.sum())], axis=1)
        cp, *_ = np.linalg.lstsq(B, np.log(ks[ok]), rcond=None)
        rp = float(np.sum((np.log(ks[ok]) - B @ cp) ** 2))
        fit["polynomial_exponent_alpha"] = float(cp[0])
        fit["polynomial_r2"] = 1.0 - rp / (ss_tot + 1e-12)
        fit["exponential_beats_polynomial"] = bool(ss_res < rp)
        fit["n_points_fitted"] = int(ok.sum())
    return fit


# ---------------------------------------------------------------------------
# Item 2 — separate GEOMETRY from DIMENSION
# ---------------------------------------------------------------------------


def item2_width_sweep(cfg, seed: int = 0):
    """At fixed d, vary the basin width w: does K_max track ``(1+2R/w)^d``?

    If it does not, the packing bound is wrong, and (per the task) that is a more
    important result than the scaling curve.
    """
    rows = []
    for d in cfg.width_sweep_dims:
        for w in cfg.width_sweep:
            scales = measure_landscape_scales(cfg, d, w=w, seed=seed)
            res = k_max_for_dim(cfg, d, seed=seed, w=w, k_cap=cfg.width_sweep_k_cap)
            w_meas = scales["w_measured_force_max"]
            R_meas = max(
                (c["site_radius_max_measured"] for c in res["cells"]), default=cfg.R
            )
            rows.append(
                {
                    "d": d,
                    "w_param": float(w),
                    "w_measured_force_max": w_meas,
                    "k_max": res["k_max"],
                    "censored": res["censored"],
                    "packing_bound_param_w": float((1.0 + 2.0 * cfg.R / w) ** d),
                    "packing_bound_measured_w": float(
                        (1.0 + 2.0 * R_meas / max(w_meas, 1e-9)) ** d
                    ),
                    "prereg_resolution_law": float(
                        (2.0 * R_meas / max(2.0 * w_meas, 5.0 * cfg.query_sigma)) ** d
                    ),
                    "cells": res["cells"],
                }
            )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Item 3 — which capacity regime?
# ---------------------------------------------------------------------------


def _regime(selectivity: float, at_or_below_kmax: bool) -> str:
    """Assign the theorist's three capacity regimes from the measured selectivity.

    1 barrier-protected (selectivity ~1.00, perfect isolation) ·
    2 washboard death zone (selectivity collapses to ~0.5 at INTERMEDIATE packing) ·
    3 continuum register (selectivity recovers to ~0.96 past the merger).

    Regime 2 vs 3 cannot be told apart from a single number — the death zone is
    defined by NON-MONOTONICITY (a dip that later recovers), so the caller must
    look at the sequence. This returns the per-cell label; ``item3_regimes``
    does the sequence-level check.
    """
    if selectivity >= 0.95:
        return "1_barrier_protected" if at_or_below_kmax else "3_continuum_like"
    if selectivity >= 0.7:
        return "degrading"
    return "2_death_zone_candidate"


def item3_regimes(item1, cfg):
    """Selectivity alongside K_max, with regime assignment and the crucial
    non-monotonicity check.

    ⚠ The standing claim is that an EMERGENT CHLU is stuck in regime 2 while
    DESIGNED structure reaches regime 3. This experiment is designed-only, so it
    should reach regime 3. **If designed structure also lands in the death zone
    at higher d, that falsifies the three-regime picture** — which the task says
    is the headline, so it is tested explicitly here rather than eyeballed.
    """
    rows = []
    for r in item1["rows"]:
        sels = [(c["K"], c["written"]["selectivity"]) for c in r["cells"]]
        vals = [s for _, s in sels]
        # A death zone is a DIP followed by a RECOVERY: min occurs strictly
        # before the last measured K and the tail rises materially above it.
        dip = False
        if len(vals) >= 3:
            i = int(np.argmin(vals))
            dip = bool(
                i < len(vals) - 1
                and vals[i] < 0.7
                and max(vals[i + 1 :]) > vals[i] + 0.15
            )
        rows.append(
            {
                "d": r["d"],
                "k_max": r["k_max"],
                "selectivity_by_K": [
                    {
                        "K": K,
                        "selectivity": s,
                        "regime": _regime(s, K <= r["k_max"]),
                    }
                    for K, s in sels
                ],
                "selectivity_at_k_max": next(
                    (s for K, s in sels if K == r["k_max"]), None
                ),
                "selectivity_monotone_non_increasing": bool(
                    all(b <= a + 1e-9 for a, b in zip(vals, vals[1:], strict=False))
                ),
                "death_zone_dip_then_recovery": dip,
            }
        )
    return {
        "rows": rows,
        "any_death_zone": bool(any(r["death_zone_dip_then_recovery"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# Item 4 — does dissipation still gate retrieval at d > 2?
# ---------------------------------------------------------------------------


def item4_gamma(cfg, seed: int = 0):
    """w19 found retrieval REQUIRES dissipation (1.000 at gamma>0, 0.813 at gamma=0).
    Does that hold at d > 2, and does the required gamma scale with d?"""
    rows = []
    for d in cfg.gamma_sweep_dims:
        for g in cfg.gamma_sweep:
            cell = evaluate_cell(cfg, d, cfg.gamma_sweep_K, seed=seed, gamma=g)
            rows.append(
                {
                    "d": d,
                    "gamma": float(g),
                    "K": cfg.gamma_sweep_K,
                    "acc_codebook": cell["written"]["acc_codebook"],
                    "selectivity": cell["written"]["selectivity"],
                    "payload_abs_err": cell["written"]["payload_abs_err"],
                    "blank_acc": cell["blank"]["acc_codebook"],
                    "blank_passes": cell["blank_passes"],
                }
            )
            print(
                f"  gamma d={d:2d} g={g:.3f} acc={rows[-1]['acc_codebook']:.3f} "
                f"sel={rows[-1]['selectivity']:.3f}",
                flush=True,
            )
    # Smallest gamma on the ladder meeting the fidelity criterion, per d.
    gmin = {}
    for d in cfg.gamma_sweep_dims:
        ok = [
            r["gamma"]
            for r in rows
            if r["d"] == d
            and r["blank_passes"]
            and r["acc_codebook"] >= cfg.selectivity_threshold
        ]
        gmin[str(d)] = float(min(ok)) if ok else None
    return {"rows": rows, "gamma_min_by_dim": gmin}


# ---------------------------------------------------------------------------
# Item 5 — ADDRESSING capacity, separated from READ-OUT capacity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figures (local, following the exp_paid_access / exp_retrieval precedent)
# ---------------------------------------------------------------------------


def _plot_all(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []

    it1 = results.get("item1_k_max_vs_dim")
    if it1 and it1["rows"]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        ds = [r["d"] for r in it1["rows"]]
        # ⚠ Censored cells (hit the compute cap without ever failing) are LOWER
        # BOUNDS, not measurements. Plotting them as ordinary points makes the
        # curve appear to saturate at high d, which is an artifact of the cap and
        # the opposite of what the data says. Drawn hollow with up-arrows.
        meas = [(r["d"], r["k_max"]) for r in it1["rows"] if not r["censored"]]
        cens = [(r["d"], r["k_max"]) for r in it1["rows"] if r["censored"]]
        if meas:
            a1.semilogy(*zip(*meas), "o-", lw=2, label="measured $K_{max}$")
        if cens:
            a1.semilogy(
                *zip(*cens),
                "o",
                mfc="none",
                mec="C0",
                label="censored (lower bound, hit $K$ cap)",
            )
            for d_, k_ in cens:
                a1.annotate(
                    "",
                    xy=(d_, k_ * 3.0),
                    xytext=(d_, k_),
                    arrowprops=dict(arrowstyle="->", color="C0", lw=1.2),
                )
        fit = it1.get("fit", {})
        if fit.get("exponential_base_A"):
            A, b = fit["exponential_base_A"], fit["exponential_intercept"]
            xs = np.linspace(min(ds), max(ds), 50)
            a1.semilogy(
                xs,
                np.exp(b) * A**xs,
                "-",
                color="C0",
                alpha=0.4,
                lw=1.0,
                label=f"fit $A^d$, $A$={A:.2f} ($R^2$={fit['exponential_r2']:.3f})",
            )
        pk = [r["packing_bound_1plus2Rovw"] for r in it1["rows"]]
        if all(p is not None for p in pk):
            a1.semilogy(ds, pk, "s--", label=r"packing bound $(1+2R/w)^d$")
        a1.semilogy(
            ds,
            [r["prereg_kmax_resolution_law"] for r in it1["rows"]],
            "^:",
            label="prereg resolution law",
        )
        a1.axhline(8, color="r", ls=":", lw=0.9, label="w19 ring ceiling (8)")
        a1.set_xlabel("address-space dimension $d$")
        a1.set_ylabel("$K_{max}$ (items retrieved at w19 fidelity)")
        a1.set_title("Capacity vs address dimension (DESIGNED landscape)")
        a1.legend(fontsize=7)

        for r in it1["rows"]:
            a2.plot(
                [c["K"] for c in r["cells"]],
                [c["written"]["acc_codebook"] for c in r["cells"]],
                "o-",
                label=f"d={r['d']}",
            )
        a2.axhline(0.9, color="r", ls="--", lw=0.8)
        a2.set_xscale("log", base=2)
        a2.set_xlabel("stored items K")
        a2.set_ylabel("linear codebook read accuracy")
        a2.set_title("Fidelity vs item count, per dimension")
        a2.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(save_dir, "dimscaling_fig1_kmax_vs_dim.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    it2 = results.get("item2_width_sweep")
    if it2 and it2["rows"]:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for d in sorted({r["d"] for r in it2["rows"]}):
            rs = [r for r in it2["rows"] if r["d"] == d]
            ax.semilogy(
                [r["w_param"] for r in rs],
                [max(r["k_max"], 0.5) for r in rs],
                "o-",
                label=f"measured d={d}",
            )
            ax.semilogy(
                [r["w_param"] for r in rs],
                [r["packing_bound_param_w"] for r in rs],
                "--",
                alpha=0.5,
                label=f"$(1+2R/w)^d$ d={d}",
            )
        ax.set_xlabel("basin width $w$")
        ax.set_ylabel("$K_{max}$")
        ax.set_title("Item 2: geometry vs dimension")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(save_dir, "dimscaling_fig2_width_sweep.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    it4 = results.get("item4_gamma")
    if it4 and it4["rows"]:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for d in sorted({r["d"] for r in it4["rows"]}):
            rs = [r for r in it4["rows"] if r["d"] == d]
            ax.plot(
                [r["gamma"] for r in rs],
                [r["acc_codebook"] for r in rs],
                "o-",
                label=f"d={d}",
            )
        ax.set_xlabel(r"friction $\gamma$")
        ax.set_ylabel("codebook read accuracy")
        ax.set_title("Item 4: retrieval requires dissipation, at every $d$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(save_dir, "dimscaling_fig3_gamma.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_dim_scaling(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Run the full dimension-scaling battery and write JSON + figures."""
    config = config or get_default_config()
    cfg = config.experiment_dim_scaling
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "designed_not_learned": True,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "R",
                "well_width",
                "well_depth",
                "payload_kappa",
                "c_conf",
                "dt",
                "gamma",
                "steps",
                "tail_frac",
                "n_subsample",
                "n_query_per_item",
                "query_sigma",
                "query_sigma_p",
                "selectivity_threshold",
                "blank_margin",
                "dims",
                "k_ladder",
                "k_cap",
                "site_seed",
                "payload_seed",
            )
        },
    }

    print("[item 1] K_max vs d", flush=True)
    results["item1_k_max_vs_dim"] = item1_k_max_vs_dim(cfg, seed=seed)
    print("[item 3] regime assignment", flush=True)
    results["item3_regimes"] = item3_regimes(results["item1_k_max_vs_dim"], cfg)
    print("[item 2] basin-width sweep", flush=True)
    results["item2_width_sweep"] = item2_width_sweep(cfg, seed=seed)
    print("[item 4] gamma dependence", flush=True)
    results["item4_gamma"] = item4_gamma(cfg, seed=seed)

    results["figures"] = _plot_all(results, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_dim_scaling_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_dim_scaling
    cfg.dims = [2, 3]
    cfg.k_ladder = [2, 4, 8, 16]
    cfg.k_cap = 16
    cfg.n_query_per_item = 8
    cfg.steps = 200
    cfg.capture_n_dirs = 4
    cfg.capture_n_offsets = 6
    cfg.width_sweep_dims = [2]
    cfg.width_sweep = [0.15, 0.30]
    cfg.gamma_sweep_dims = [2]
    cfg.gamma_sweep = [0.0, 0.02]


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment DIM-SCALING: K_max vs address-space dimension"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed (project-level)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    args = parser.parse_args()

    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        config = get_default_config()
        save_dir, models_dir = "results", None
        os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        apply_quick(config)

    res = run_experiment_dim_scaling(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(json.dumps(res["item1_k_max_vs_dim"]["fit"], indent=2))
    print("metrics ->", res["metrics_path"])


if __name__ == "__main__":
    main()
