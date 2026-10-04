"""Experiment RETRIEVAL: the hand-built write -> address -> retrieve loop (stage 1).

The Head's architectural vision (handover 2026-07-21) is that CLU is an
**addressable dynamical memory**: items are written into the learned landscape
and retrieved by launching a particle of mass ``m`` from an initial latent
``(q0, p0)`` and reading its trajectory. **Nobody had ever run that loop.** This
module runs the smallest thing that could work.

⚠ **EVERYTHING HERE IS HAND-DESIGNED, NOT LEARNED.** The potential is built by
hand (``chlu.core.memory_potentials``), the addresses are hand-picked, and the
only gradient descent anywhere is *on the address* in item 5. This is the right
stage-1 experiment — but a designed demo must never be quoted as an emergent
capability (N46 precedent).

Six measurements:

1. ``item1_selectivity``   two items, two addresses: is retrieval selective
                           under a LINEAR read, and for how long does it survive?
2. ``item2_mass_key``      does MASS alone address? (the vision's load-bearing,
                           least-tested claim)
3. ``item3_write_modes``   permanent / decaying / uncorrelated side by side;
                           half-life of the decaying item + corruption check
                           on its permanent neighbour.
4. ``item4_interference``  selectivity vs item count -> the practical ceiling.
5. ``item5_restructuring`` the learnability crux in its WEAK form: can a
                           DELIBERATELY BAD address be restructured into a
                           working one by descending a retrieval loss?
5b. ``item5b_smoothness``  is the q0 loss surface smooth? gradient norm vs
                           rollout length.

The anti-decoration guard runs through all of it: the payload coordinate is
always launched at ``q2(0) = 0``, so any read-out that recovers the stored value
got it from ``V``, not from the address. Every selectivity number is reported
alongside a **blank-landscape control** (identical dynamics, nothing stored), and
alongside a **payload-only read** that is structurally blind to the address plane.

Runnable directly:
    uv run python -m chlu.experiments.exp_retrieval --quick
or via the CLI: ``chlu exp-retrieval [--project N] [--seed I] [--quick]``.
"""

import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.memory_potentials import (
    RingRegisterPotential,
    ThreeModePotential,
    designed_payloads,
)
from chlu.experiments.goldstone_harness import clu_with_potential, log_mass_for_inertia


# ---------------------------------------------------------------------------
# Model construction / rollout
# ---------------------------------------------------------------------------


def build_ring_model(
    payloads,
    cfg,
    inertia=None,
    dim: int = 3,
) -> CHLU:
    """A CLU wired to a hand-designed ``RingRegisterPotential``.

    ``newtonian_learned`` kinetics so the mass vector is actually read by the
    dynamics (``newtonian_identity`` ignores ``log_mass`` entirely — which is
    why the mass-as-address question cannot even be posed in Exp-A's default
    kinetic mode).
    """
    V = RingRegisterPotential(
        payloads,
        lam=cfg.lam,
        f=cfg.f,
        b=cfg.barrier,
        kappa=cfg.payload_kappa,
        bump_width=cfg.bump_width,
        n_spectator=dim - 3,
    )
    if inertia is None:
        inertia = jnp.ones(dim)
    return clu_with_potential(
        V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.asarray(inertia)
    )


def rollout_batch(model: CHLU, Q0, P0, steps: int, dt: float, gamma: float):
    """Deterministic Verlet rollouts for a batch of addresses -> (n, steps, 2*dim)."""
    return jax.vmap(lambda q, p: model(q, p, steps, dt, gamma))(Q0, P0)


def tail_features(traj, tail_frac: float, n_sub: int, coords=None):
    """Subsample the TAIL of each trajectory and flatten -> (n, n_sub*len(coords)).

    The read deliberately uses only the tail: the early trajectory still carries
    the address (``q0``/``p0``) almost verbatim, so a probe on the full
    trajectory would be reading the query back, not the memory.
    """
    steps = traj.shape[1]
    start = int((1.0 - tail_frac) * steps)
    idx = np.linspace(start, steps - 1, n_sub).astype(int)
    sub = traj[:, idx, :]
    if coords is not None:
        sub = sub[:, :, jnp.asarray(coords)]
    return np.asarray(sub).reshape(sub.shape[0], -1)


# ---------------------------------------------------------------------------
# Linear read-out (the bar: a LINEAR probe, per CAFE's frozen-embedding contract)
# ---------------------------------------------------------------------------


def linear_codebook_read(X_tr, y_tr, X_te, y_te, payloads, ridge: float = 1e-6):
    """LINEAR regression onto the stored value, decoded against the codebook.

    ⚠ Why this exists (an estimator artifact that would otherwise be misread as
    a physics result). The payload-only feature is essentially 1-D (the settled
    ``q2`` ~ a_k) and the stored codebook is deliberately NON-MONOTONE in the
    site index. Ridge-to-one-hot least squares on a 1-D feature fits a
    near-monotone trend per class and therefore CANNOT invert a non-monotone
    code: it scored 0.48 at K=4 while the payload itself was retrieved to
    7e-4 absolute error. That is a decoder failure, not a retrieval failure.

    The natural read for a scalar-valued stored item is a linear map to the
    VALUE followed by nearest-codeword decoding — still strictly linear (plus a
    fixed lookup), and it is what an associative-memory read actually looks
    like. Reported alongside the one-hot probe, never instead of it.
    """
    pay = np.asarray(payloads)
    Xb_tr = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
    Xb_te = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1)
    t_tr = pay[y_tr]
    A = Xb_tr.T @ Xb_tr + ridge * np.eye(Xb_tr.shape[1])
    w = np.linalg.solve(A, Xb_tr.T @ t_tr)
    pred_val = Xb_te @ w
    pred = np.argmin(np.abs(pred_val[:, None] - pay[None, :]), axis=1)
    acc = float(np.mean(pred == y_te))
    ss_res = float(np.sum((pred_val - pay[y_te]) ** 2))
    ss_tot = float(np.sum((pay[y_te] - np.mean(pay[y_te])) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return acc, r2


def nearest_centroid_read(X_tr, y_tr, X_te, y_te, n_class: int):
    """Nearest-class-centroid — also a LINEAR classifier (argmax of
    ``2*c_k.x - |c_k|^2``), but a much better-conditioned estimator than
    least-squares-to-one-hot on a low-rank feature."""
    cents = np.stack(
        [
            X_tr[y_tr == k].mean(axis=0)
            if np.any(y_tr == k)
            else np.zeros(X_tr.shape[1])
            for k in range(n_class)
        ]
    )
    d = ((X_te[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    return float(np.mean(np.argmin(d, axis=1) == y_te))


def linear_probe(X_tr, y_tr, X_te, y_te, n_class: int, ridge: float = 1e-6):
    """Multiclass ridge-to-one-hot linear probe. Returns (accuracy, confusion).

    Deliberately the weakest reasonable decoder: a retrieval that needs a
    nonlinear decoder is much weaker evidence for the vision.
    """
    Xb_tr = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
    Xb_te = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1)
    Y = np.eye(n_class)[y_tr]
    A = Xb_tr.T @ Xb_tr + ridge * np.eye(Xb_tr.shape[1])
    W = np.linalg.solve(A, Xb_tr.T @ Y)
    pred = np.argmax(Xb_te @ W, axis=1)
    acc = float(np.mean(pred == y_te))
    conf = np.zeros((n_class, n_class), dtype=int)
    for t, p in zip(y_te, pred, strict=False):
        conf[t, p] += 1
    return acc, conf


def make_queries(key, K: int, n_per_item: int, cfg, dim: int = 3):
    """Noisy queries: n_per_item jittered addresses for each of the K item sites.

    ``q0[2] = 0`` and ``p0[2] = 0`` ALWAYS — the payload channel carries no
    address information. This is the anti-decoration guard.
    """
    k_th, k_r, k_p = jax.random.split(key, 3)
    n = K * n_per_item
    labels = np.repeat(np.arange(K), n_per_item)
    theta = jnp.asarray(labels, dtype=jnp.float32) * (2.0 * jnp.pi / K)
    theta = theta + jax.random.normal(k_th, (n,)) * cfg.query_sigma_theta
    r = cfg.f + jax.random.normal(k_r, (n,)) * cfg.query_sigma_r

    Q0 = jnp.zeros((n, dim))
    Q0 = Q0.at[:, 0].set(r * jnp.cos(theta)).at[:, 1].set(r * jnp.sin(theta))
    P0 = jnp.zeros((n, dim))
    P0 = P0.at[:, :2].set(jax.random.normal(k_p, (n, 2)) * cfg.query_sigma_p)
    return Q0, P0, labels


def _probe_split(feats, labels, n_class, rng):
    """50/50 split, probe accuracy + confusion."""
    n = feats.shape[0]
    perm = rng.permutation(n)
    half = n // 2
    tr, te = perm[:half], perm[half:]
    return linear_probe(feats[tr], labels[tr], feats[te], labels[te], n_class)


# ---------------------------------------------------------------------------
# Item 1 — minimal demo: two items, two addresses, selective retrieval
# ---------------------------------------------------------------------------


def item1_selectivity(cfg, K: int = 2, seed: int = 0):
    """Two items, two addresses. Linear read, blank control, survival vs tail."""
    key = jax.random.PRNGKey(seed)
    pay = designed_payloads(K, seed=cfg.payload_seed)

    written = build_ring_model(pay, cfg)
    blank = build_ring_model(jnp.zeros_like(pay), cfg)  # live channel, nothing stored

    Q0, P0, labels = make_queries(key, K, cfg.n_query_per_item, cfg)
    out = {"K": K, "payloads": [float(x) for x in pay]}

    for name, model in (("written", written), ("blank", blank)):
        traj = rollout_batch(model, Q0, P0, cfg.steps, cfg.dt, cfg.gamma)
        # (i) full-state tail read, (ii) payload-only tail read (blind to address)
        f_full = tail_features(traj, cfg.tail_frac, cfg.n_subsample)
        f_pay = tail_features(traj, cfg.tail_frac, cfg.n_subsample, coords=[2])
        acc_full, conf_full = _probe_split(
            f_full, labels, K, np.random.default_rng(seed)
        )
        acc_pay, conf_pay = _probe_split(f_pay, labels, K, np.random.default_rng(seed))
        # Better-conditioned LINEAR reads of the same features (see
        # linear_codebook_read for why the one-hot probe alone is misleading here)
        n = f_pay.shape[0]
        perm = np.random.default_rng(seed).permutation(n)
        tr, te = perm[: n // 2], perm[n // 2 :]
        acc_cb, r2_cb = linear_codebook_read(
            f_pay[tr], labels[tr], f_pay[te], labels[te], pay
        )
        acc_nc = nearest_centroid_read(f_pay[tr], labels[tr], f_pay[te], labels[te], K)
        out[name] = {
            "acc_full_state_read": acc_full,
            "acc_payload_only_read": acc_pay,
            "acc_payload_only_codebook_read": acc_cb,
            "payload_regression_r2": r2_cb,
            "acc_payload_only_nearest_centroid": acc_nc,
            "confusion_payload_only": conf_pay.tolist(),
            "confusion_full_state": conf_full.tolist(),
        }
        if name == "written":
            # Survival: payload-only accuracy as a function of WHERE we read.
            steps = traj.shape[1]
            surv = []
            for frac in cfg.survival_fracs:
                i = min(int(frac * steps), steps - 1)
                f = np.asarray(traj[:, i, 2])[:, None]
                a, _ = _probe_split(f, labels, K, np.random.default_rng(seed))
                surv.append({"step": int(i), "acc_payload_only": a})
            out["survival_written"] = surv
            # payload retrieval error at the very end
            end = np.asarray(traj[:, -1, 2])
            err = [
                float(np.mean(np.abs(end[labels == k] - float(pay[k]))))
                for k in range(K)
            ]
            out["payload_abs_err_per_item"] = err
            # Separate the two failure modes: did the query land in the RIGHT
            # WELL (addressing) or did it land right and read out wrong
            # (payload-bump overlap)?
            th_end = np.arctan2(np.asarray(traj[:, -1, 1]), np.asarray(traj[:, -1, 0]))
            th_tgt = labels * (2.0 * np.pi / K)
            d_th = np.abs((th_end - th_tgt + np.pi) % (2 * np.pi) - np.pi)
            spacing = 2.0 * np.pi / K
            out["addressing_mean_abs_dtheta"] = float(np.mean(d_th))
            out["addressing_frac_landed_in_correct_well"] = float(
                np.mean(d_th < 0.5 * spacing)
            )
            out["site_spacing_rad"] = float(spacing)
            out["query_sigma_theta_over_spacing"] = float(
                cfg.query_sigma_theta / spacing
            )

    # gamma=0 (conservative) contrast: does the item ever settle?
    traj0 = rollout_batch(written, Q0, P0, cfg.steps, cfg.dt, 0.0)
    f_pay0 = tail_features(traj0, cfg.tail_frac, cfg.n_subsample, coords=[2])
    a0, _ = _probe_split(f_pay0, labels, K, np.random.default_rng(seed))
    out["gamma0_acc_payload_only"] = a0
    return out


# ---------------------------------------------------------------------------
# Item 2 — does MASS work as an address key?
# ---------------------------------------------------------------------------


def _retrieved_item(traj_tail_q2, payloads):
    """Nearest stored payload to the settled read-out value."""
    d = np.abs(np.asarray(traj_tail_q2)[:, None] - np.asarray(payloads)[None, :])
    return np.argmin(d, axis=1)


def item2_mass_key(cfg, K: int = 8, seed: int = 0):
    """Hold (q0, p0) FIXED, vary ONLY the mass. Do different masses retrieve
    different items?

    Two sweeps:
      (a) SCALAR mass m*I on the address plane — an energy dial (initial
          KE = p0^T M^-1 p0 / 2, so mass is NOT a time reparameterization at
          fixed p0; it is at fixed initial *velocity*).
      (b) MASS VECTOR (m0, m1) — steers the initial velocity direction
          M^-1 p0. Because m_i > 0, the reachable directions are confined to
          the open sign-orthant of p0.
    """
    pay = designed_payloads(K, seed=cfg.payload_seed)
    # One fixed launch address for the whole sweep.
    q0 = jnp.array([cfg.f, 0.0, 0.0])
    p0 = jnp.array([cfg.mass_probe_p, cfg.mass_probe_p, 0.0])

    def retrieve(inertia):
        model = build_ring_model(pay, cfg, inertia=inertia)
        traj = model(q0, p0, cfg.steps, cfg.dt, cfg.gamma)
        tail = traj[int((1.0 - cfg.tail_frac) * cfg.steps) :]
        q2 = float(jnp.mean(tail[:, 2]))
        theta = float(
            jnp.arctan2(jnp.mean(tail[:, 1]), jnp.mean(tail[:, 0])) % (2 * jnp.pi)
        )
        # At very small inertia the launch velocity p0/M is large enough that
        # Verlet goes unstable (observed NaN at m=0.032, |v|=15.8, dt=0.05).
        # Those cells are a NUMERICAL artifact, not a retrieval: flag and
        # exclude them from every capacity count.
        diverged = bool(not np.isfinite(q2) or not np.isfinite(theta))
        item = -1 if diverged else int(_retrieved_item(np.array([q2]), pay)[0])
        ok = bool((not diverged) and abs(q2 - float(pay[item])) < cfg.payload_tol)
        return {
            "q2": q2,
            "theta": theta,
            "item": item,
            "settled": ok,
            "diverged": diverged,
        }

    scalar = []
    for m in np.logspace(cfg.mass_log_lo, cfg.mass_log_hi, cfg.mass_n):
        r = retrieve(jnp.array([m, m, 1.0]))
        r["m"] = float(m)
        scalar.append(r)

    vector = []
    grid = np.logspace(cfg.mass_log_lo, cfg.mass_log_hi, cfg.mass_n_vec)
    for m0 in grid:
        for m1 in grid:
            r = retrieve(jnp.array([m0, m1, 1.0]))
            r["m0"], r["m1"] = float(m0), float(m1)
            vector.append(r)

    def distinct(rs):
        return sorted({r["item"] for r in rs if r["settled"] and not r["diverged"]})

    ds, dv = distinct(scalar), distinct(vector)

    # ⭐ ROBUSTNESS — the measurement that decides whether mass is a USABLE key.
    # Counting distinct reachable items is not enough: a chaotic m -> item map
    # reaches many items and is still worthless as an address, because the
    # address cannot be stored, transmitted or learned to finite precision.
    rng = np.random.default_rng(seed)
    robust = []
    for r in scalar:
        m = r["m"]
        got = []
        for _ in range(cfg.mass_jitter_n):
            mj = m * (1.0 + rng.normal() * cfg.mass_jitter_rel)
            got.append(retrieve(jnp.array([mj, mj, 1.0]))["item"])
        robust.append(
            {
                "m": m,
                "item_nominal": r["item"],
                "items_under_jitter": got,
                "frac_same": float(np.mean([g == r["item"] for g in got])),
            }
        )
    mean_robust = float(np.mean([x["frac_same"] for x in robust]))

    # ⭐ The number that actually answers the question: how many items can be
    # addressed by mass RELIABLY? A mass cell counts only if a 1% mass error
    # still retrieves the same item. (The raw distinct-item count is inflated by
    # a chaotic band near the escape threshold where neighbouring masses land
    # anywhere.)
    robust_items = sorted(
        {
            x["item_nominal"]
            for x in robust
            if x["frac_same"] >= cfg.mass_robust_threshold and x["item_nominal"] >= 0
        }
    )

    # Monotonicity: is the landing angle an ordered dial in 1/m, or scrambled?
    fin = [r for r in scalar if not r["diverged"]]
    logm = np.log([r["m"] for r in fin])
    th = np.array([r["theta"] for r in fin])
    corr = float(np.corrcoef(-logm, th)[0, 1]) if len(fin) > 2 else float("nan")

    return {
        "robustness_rel_jitter": cfg.mass_jitter_rel,
        "robustness_frac_same_item": mean_robust,
        "robustness_detail": robust,
        "robust_distinct_items": robust_items,
        "n_robust_distinct_items": len(robust_items),
        "bits_robust": float(np.log2(max(len(robust_items), 1))),
        "n_diverged_cells": int(sum(r["diverged"] for r in scalar)),
        "monotonicity_corr_neglogm_theta": corr,
        "K": K,
        "payloads": [float(x) for x in pay],
        "launch_q0": [float(x) for x in q0],
        "launch_p0": [float(x) for x in p0],
        "scalar_sweep": scalar,
        "vector_sweep": vector,
        "distinct_items_scalar": ds,
        "distinct_items_vector": dv,
        "n_distinct_scalar": len(ds),
        "n_distinct_vector": len(dv),
        "bits_scalar": float(np.log2(max(len(ds), 1))),
        "bits_vector": float(np.log2(max(len(dv), 1))),
    }


# ---------------------------------------------------------------------------
# Item 3 — the three write modes, side by side
# ---------------------------------------------------------------------------


def item3_write_modes(cfg, seed: int = 0):
    """Permanent + decaying (written NEAR it) + uncorrelated, in one landscape.

    Half-life of the decaying item, and whether writing it corrupts its
    permanent neighbour.
    """
    V = ThreeModePotential(lam=cfg.lam, f=cfg.f, beta=cfg.tm_beta, d=cfg.tm_d)
    model = clu_with_potential(
        V, dim=4, kinetic_mode="newtonian_learned", inertia=jnp.ones(4)
    )

    theta_w = cfg.tm_write_theta  # (a) permanent: the stored angle
    dr = cfg.tm_write_dr  # (b) decaying: radial excursion at the SAME locus
    sign = cfg.tm_write_sign  # (c) uncorrelated: broken-symmetry vacuum

    def run(radial_excursion):
        r = cfg.f + radial_excursion
        q0 = jnp.array([r * np.cos(theta_w), r * np.sin(theta_w), sign * cfg.tm_d, 0.0])
        p0 = jnp.zeros(4)
        return model(q0, p0, cfg.tm_steps, cfg.dt, cfg.gamma)

    traj_w = run(dr)  # with the decaying write
    traj_0 = run(0.0)  # control: permanent item alone

    def theta_of(t):
        return np.unwrap(np.arctan2(np.asarray(t[:, 1]), np.asarray(t[:, 0])))

    th_w, th_0 = theta_of(traj_w), theta_of(traj_0)
    r_w = np.linalg.norm(np.asarray(traj_w[:, :2]), axis=1)

    # (b) half-life of the radial excursion.
    # ⚠ |r - f| OSCILLATES at the radial frequency omega_rad = sqrt(8*lam*f^2)
    # and touches zero every half period, so "first n with |r-f| < dr/2" measures
    # the first zero crossing (~a quarter period), NOT the decay. Measured 6
    # steps that way vs a true half-life of ~69. Take the ENVELOPE: the local
    # maxima of |r - f|, fitted log-linearly.
    env = np.abs(r_w - cfg.f)
    peak_idx = np.where((env[1:-1] >= env[:-2]) & (env[1:-1] >= env[2:]))[0] + 1
    peak_idx = peak_idx[env[peak_idx] > 1e-9]
    n_half = None
    rate = None
    if len(peak_idx) >= 3:
        A = np.polyfit(peak_idx.astype(float), np.log(env[peak_idx]), 1)
        rate = float(-A[0])  # per-step envelope decay rate
        if rate > 0:
            n_half = float(np.log(2.0) / rate)
    # sliding-window (envelope-crossing) cross-check
    period = int(2 * np.pi / np.sqrt(8.0 * cfg.lam * cfg.f**2) / cfg.dt)
    win = max(period, 2)
    roll_max = np.array([env[i : i + win].max() for i in range(max(len(env) - win, 1))])
    below = np.where(roll_max < 0.5 * abs(dr))[0]
    n_half_window = int(below[0]) if len(below) else None

    # (c) uncorrelated item: sign of q2 retained?
    q2_end = float(traj_w[-1, 2])

    return {
        "permanent_theta_written": float(theta_w),
        "permanent_theta_drift_with_decaying_write": float(abs(th_w[-1] - theta_w)),
        "permanent_theta_drift_control": float(abs(th_0[-1] - theta_w)),
        "corruption_delta_theta": float(abs(th_w[-1] - th_0[-1])),
        "decaying_dr_written": float(dr),
        "decaying_dr_final": float(env[-1]),
        "decaying_half_life_steps": n_half,
        "decaying_half_life_steps_window_crosscheck": n_half_window,
        "decaying_envelope_rate_per_step": rate,
        "decaying_half_life_predicted_2ln2_over_gamma": float(
            2.0 * np.log(2.0) / cfg.gamma
        ),
        "mu_rad_sq": float(8.0 * cfg.lam * cfg.f**2),
        "uncorrelated_sign_written": int(sign),
        "uncorrelated_q2_final": q2_end,
        "uncorrelated_sign_retained": bool(np.sign(q2_end) == np.sign(sign)),
        "steps": cfg.tm_steps,
        "gamma": cfg.gamma,
    }


# ---------------------------------------------------------------------------
# Item 4 — interference scaling
# ---------------------------------------------------------------------------


def item4_interference(cfg, seed: int = 0):
    """Selectivity vs item count. Reports the K at which the payload-only read
    drops below ``cfg.selectivity_threshold``."""
    rows = []
    for K in cfg.item_counts:
        r = item1_selectivity(cfg, K=K, seed=seed)
        rows.append(
            {
                "K": K,
                "acc_payload_only": r["written"]["acc_payload_only_read"],
                "acc_payload_codebook": r["written"]["acc_payload_only_codebook_read"],
                "acc_payload_centroid": r["written"][
                    "acc_payload_only_nearest_centroid"
                ],
                "payload_r2": r["written"]["payload_regression_r2"],
                "acc_full_state": r["written"]["acc_full_state_read"],
                "acc_blank_payload_only": r["blank"]["acc_payload_only_read"],
                "acc_blank_payload_codebook": r["blank"][
                    "acc_payload_only_codebook_read"
                ],
                "chance": 1.0 / K,
                "mean_payload_abs_err": float(np.mean(r["payload_abs_err_per_item"])),
                "frac_landed_in_correct_well": r[
                    "addressing_frac_landed_in_correct_well"
                ],
                "mean_abs_dtheta": r["addressing_mean_abs_dtheta"],
                "site_spacing_rad": r["site_spacing_rad"],
                "sigma_theta_over_spacing": r["query_sigma_theta_over_spacing"],
            }
        )
    thr = cfg.selectivity_threshold
    ceiling = None
    for row in rows:
        if row["acc_payload_codebook"] < thr:
            ceiling = row["K"]
            break
    return {
        "threshold": thr,
        "rows": rows,
        "first_K_below_threshold": ceiling,
        "practical_ceiling_items": (
            max([r["K"] for r in rows if r["acc_payload_codebook"] >= thr], default=0)
        ),
        "practical_ceiling_bits": float(
            np.log2(
                max(
                    [r["K"] for r in rows if r["acc_payload_codebook"] >= thr],
                    default=1,
                )
            )
        ),
    }


# ---------------------------------------------------------------------------
# Item 5 — the learnability crux, WEAK form
# ---------------------------------------------------------------------------


def _address_loss_fn(cfg, pay, dim=3):
    """Retrieval loss on the address (log_m, q0[:2], p0[:2]).

    ``q0[2] = p0[2] = 0`` are held FIXED so the optimizer cannot cheat by
    writing the answer into the launch state — the anti-decoration guard
    survives into the learnability test.
    """
    V = RingRegisterPotential(
        pay,
        lam=cfg.lam,
        f=cfg.f,
        b=cfg.barrier,
        kappa=cfg.payload_kappa,
        bump_width=cfg.bump_width,
    )
    base = clu_with_potential(
        V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )
    start = int((1.0 - cfg.tail_frac) * cfg.steps)

    def readout(params):
        model = eqx.tree_at(lambda m: m.log_mass, base, replace=params["log_m"])
        q0 = jnp.concatenate([params["q0"], jnp.zeros(1)])
        p0 = jnp.concatenate([params["p0"], jnp.zeros(1)])
        traj = model(q0, p0, cfg.steps, cfg.dt, cfg.gamma)
        return jnp.mean(traj[start:, 2])

    def loss(params, target):
        return (readout(params) - target) ** 2

    return readout, loss


def item5_restructuring(cfg, K: int = 8, seed: int = 0):
    """⭐ THE PRIMARY NUMBER. Start from a DELIBERATELY BAD address (one that
    retrieves the wrong item), descend the retrieval loss, and ask whether it
    finds a working address.

    This is the Head's WEAK form of the learnability question: not "can we
    recover the exact address by gradient" but "can a bad initial address be
    RESTRUCTURED into a working one over a modest number of steps".
    """
    pay = designed_payloads(K, seed=cfg.payload_seed)
    readout, loss = _address_loss_fn(cfg, pay)
    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    readout = eqx.filter_jit(readout)  # called every GD step for the success trace

    def init_at(site):
        th = 2.0 * np.pi * site / K
        return {
            "log_m": log_mass_for_inertia(jnp.ones(3)),
            "q0": jnp.array([cfg.f * np.cos(th), cfg.f * np.sin(th)]),
            "p0": jnp.zeros(2),
        }

    trials = []
    for target in range(K):
        for offset in cfg.restructure_offsets:
            src = (target + offset) % K
            if src == target:
                continue
            params = init_at(src)
            tgt = float(pay[target])
            opt = optax.adam(cfg.address_lr)
            state = opt.init(params)
            l0 = float(loss(params, tgt))
            hist = []
            first_success = None
            for step in range(cfg.address_steps):
                lv, g = grad_fn(params, tgt)
                updates, state = opt.update(g, state)
                params = eqx.apply_updates(params, updates)
                hist.append(float(lv))
                got = int(_retrieved_item(np.array([float(readout(params))]), pay)[0])
                if got == target and first_success is None:
                    first_success = step
            final = float(readout(params))
            got = int(_retrieved_item(np.array([final]), pay)[0])
            ok = bool(got == target and abs(final - tgt) < cfg.payload_tol)
            trials.append(
                {
                    "target": target,
                    "src_site": src,
                    "offset": int(offset),
                    "antipodal": bool(offset == K // 2),
                    "loss_init": l0,
                    "loss_final": hist[-1],
                    "readout_final": final,
                    "retrieved_item": got,
                    "success": ok,
                    "steps_to_first_success": first_success,
                }
            )

    succ = [t for t in trials if t["success"]]
    anti = [t for t in trials if t["antipodal"]]
    adj = [t for t in trials if abs(t["offset"]) == 1]
    return {
        "K": K,
        "n_trials": len(trials),
        "success_rate": float(len(succ) / max(len(trials), 1)),
        "success_rate_adjacent": float(
            np.mean([t["success"] for t in adj]) if adj else float("nan")
        ),
        "success_rate_antipodal": float(
            np.mean([t["success"] for t in anti]) if anti else float("nan")
        ),
        "median_steps_to_success": (
            float(np.median([t["steps_to_first_success"] for t in succ]))
            if succ
            else None
        ),
        "address_lr": cfg.address_lr,
        "address_steps": cfg.address_steps,
        "trials": trials,
    }


def item5c_annealed_restructuring(cfg, K: int = 8, seed: int = 0):
    """The repair implied by the item-5b diagnosis: ANNEAL gamma during the
    address search.

    The diagnosis: at the gamma that makes the read stable, the settled state is
    independent of the launch point inside a basin, so the address loss is a
    STAIRCASE (flat plateaus, cliffs at separatrices) and gradient descent has
    nothing to descend. At gamma ~ 0 the trajectory keeps circulating and does
    depend on the launch point — the loss is informative but the read is not
    settled.

    So: search at low gamma (informative gradient), evaluate at the operating
    gamma (stable read), interpolating over the GD run. If this fixes the
    restructuring rate, the failure is a *protocol* failure, not a structural
    one. Reported either way.
    """
    pay = designed_payloads(K, seed=cfg.payload_seed)
    g_lo, g_hi = cfg.anneal_gamma_lo, cfg.gamma
    n_steps = cfg.address_steps
    # Pre-build one loss per gamma on the anneal ladder (jit-cached).
    ladder = np.linspace(g_lo, g_hi, cfg.anneal_n_stages)
    stages = []
    for g in ladder:
        ro, lo = _address_loss_fn(_replace(cfg, gamma=float(g)), pay)
        stages.append(
            (eqx.filter_jit(eqx.filter_value_and_grad(lo)), eqx.filter_jit(ro))
        )
    readout_final = stages[-1][1]

    def init_at(site):
        th = 2.0 * np.pi * site / K
        return {
            "log_m": log_mass_for_inertia(jnp.ones(3)),
            "q0": jnp.array([cfg.f * np.cos(th), cfg.f * np.sin(th)]),
            "p0": jnp.zeros(2),
        }

    trials = []
    for target in range(K):
        for offset in cfg.restructure_offsets:
            src = (target + offset) % K
            if src == target:
                continue
            params = init_at(src)
            tgt = float(pay[target])
            opt = optax.adam(cfg.address_lr)
            state = opt.init(params)
            for step in range(n_steps):
                si = min(
                    int(step / n_steps * cfg.anneal_n_stages), cfg.anneal_n_stages - 1
                )
                _, g = stages[si][0](params, tgt)
                updates, state = opt.update(g, state)
                params = eqx.apply_updates(params, updates)
            final = float(readout_final(params))
            got = int(_retrieved_item(np.array([final]), pay)[0])
            trials.append(
                {
                    "target": target,
                    "src_site": src,
                    "offset": int(offset),
                    "readout_final": final,
                    "retrieved_item": got,
                    "success": bool(
                        got == target and abs(final - tgt) < cfg.payload_tol
                    ),
                }
            )
    return {
        "K": K,
        "gamma_lo": g_lo,
        "gamma_hi": g_hi,
        "n_stages": cfg.anneal_n_stages,
        "n_trials": len(trials),
        "success_rate": float(np.mean([t["success"] for t in trials])),
        "success_rate_adjacent": float(
            np.mean([t["success"] for t in trials if abs(t["offset"]) == 1])
        ),
        "trials": trials,
    }


def item5b_smoothness(cfg, K: int = 8, seed: int = 0):
    """Is the q0 retrieval-loss surface smooth or cliffy? And does the gradient
    through the rollout blow up with rollout length (no contraction)?"""
    pay = designed_payloads(K, seed=cfg.payload_seed)
    readout, loss = _address_loss_fn(cfg, pay)
    target = float(pay[0])

    thetas = np.linspace(0.0, 2.0 * np.pi, cfg.smooth_n_theta, endpoint=False)
    curve = []
    for th in thetas:
        params = {
            "log_m": log_mass_for_inertia(jnp.ones(3)),
            "q0": jnp.array([cfg.f * np.cos(th), cfg.f * np.sin(th)]),
            "p0": jnp.zeros(2),
        }
        curve.append(float(loss(params, target)))
    curve = np.array(curve)
    d1 = np.abs(np.diff(curve))
    # "cliffiness": largest single-step jump relative to the median step
    cliff = float(np.max(d1) / (np.median(d1) + 1e-12))

    # gradient norm vs rollout length
    grads = []
    for steps in cfg.smooth_rollout_steps:
        c2 = _replace(cfg, steps=steps)
        ro2, l2 = _address_loss_fn(c2, pay)
        params = {
            "log_m": log_mass_for_inertia(jnp.ones(3)),
            "q0": jnp.array([cfg.f * np.cos(0.9), cfg.f * np.sin(0.9)]),
            "p0": jnp.zeros(2),
        }
        g = eqx.filter_grad(l2)(params, target)
        gn = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree_util.tree_leaves(g))))
        grads.append({"steps": int(steps), "grad_norm": gn})

    # ⭐ THE TENSION. Retrieval NEEDS friction (the particle must settle in a
    # well for the read to be stable), but friction is exactly what erases the
    # trajectory's dependence on its initial condition — which is the address
    # gradient. Measure both sides of it on one axis: settling quality and
    # address-gradient norm as functions of gamma.
    gamma_scan = []
    for g in cfg.smooth_gammas:
        cg = _replace(cfg, gamma=g)
        ro_g, l_g = _address_loss_fn(cg, pay)
        params = {
            "log_m": log_mass_for_inertia(jnp.ones(3)),
            "q0": jnp.array([cfg.f * np.cos(0.9), cfg.f * np.sin(0.9)]),
            "p0": jnp.zeros(2),
        }
        gg = eqx.filter_grad(l_g)(params, target)
        gn = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree_util.tree_leaves(gg))))
        # settling quality at this gamma: payload error from an exact launch
        model = build_ring_model(pay, cg)
        tr = model(jnp.array([cfg.f, 0.0, 0.0]), jnp.zeros(3), cg.steps, cg.dt, g)
        settle_err = float(abs(float(tr[-1, 2]) - float(pay[0])))
        gamma_scan.append(
            {"gamma": float(g), "grad_norm": gn, "settle_payload_err": settle_err}
        )

    return {
        "K": K,
        "thetas": thetas.tolist(),
        "loss_curve": curve.tolist(),
        "gamma_scan_grad_vs_settling": gamma_scan,
        "cliff_ratio_max_over_median_step": cliff,
        "n_local_minima": int(
            np.sum((curve < np.roll(curve, 1)) & (curve < np.roll(curve, -1)))
        ),
        "grad_norm_vs_rollout": grads,
    }


def _replace(cfg, **kw):
    """Shallow copy of a config dataclass with fields overridden."""
    import copy

    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# Figures (local, following the exp_paid_access precedent)
# ---------------------------------------------------------------------------


def _plot_all(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []

    # Fig 1: interference curve
    inter = results.get("item4_interference")
    if inter:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        Ks = [r["K"] for r in inter["rows"]]
        ax.plot(
            Ks,
            [r["acc_payload_only"] for r in inter["rows"]],
            "o-",
            label="payload-only read (written)",
        )
        ax.plot(
            Ks,
            [r["acc_full_state"] for r in inter["rows"]],
            "s--",
            label="full-state read (written)",
        )
        ax.plot(
            Ks,
            [r["acc_blank_payload_only"] for r in inter["rows"]],
            "^:",
            label="blank control",
        )
        ax.plot(Ks, [r["chance"] for r in inter["rows"]], "k:", label="chance")
        ax.axhline(inter["threshold"], color="r", ls="--", lw=0.8, label="threshold")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("stored items K")
        ax.set_ylabel("linear-probe retrieval accuracy")
        ax.set_title("Interference: selectivity vs item count (DESIGNED landscape)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(save_dir, "retrieval_fig1_interference.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # Fig 2: mass sweep
    m2 = results.get("item2_mass_key")
    if m2:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ms = [r["m"] for r in m2["scalar_sweep"]]
        a1.semilogx(ms, [r["item"] for r in m2["scalar_sweep"]], "o-")
        a1.set_xlabel("scalar mass m (address plane)")
        a1.set_ylabel("retrieved item index")
        a1.set_title(
            f"Mass as address key: {m2['n_distinct_scalar']} distinct items\n({m2['bits_scalar']:.2f} bits)"
        )
        g = int(np.sqrt(len(m2["vector_sweep"])))
        img = np.array([r["item"] for r in m2["vector_sweep"]]).reshape(g, g)
        im = a2.imshow(img, origin="lower", cmap="tab20")
        a2.set_xlabel("m1 index")
        a2.set_ylabel("m0 index")
        a2.set_title(f"mass VECTOR sweep: {m2['n_distinct_vector']} distinct items")
        fig.colorbar(im, ax=a2)
        fig.tight_layout()
        p = os.path.join(save_dir, "retrieval_fig2_mass_key.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # Fig 3: q0 loss surface + gradient norm
    sm = results.get("item5b_smoothness")
    if sm:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot(sm["thetas"], sm["loss_curve"], lw=1.0)
        a1.set_xlabel("launch angle theta0")
        a1.set_ylabel("retrieval loss")
        a1.set_title(f"q0 loss surface ({sm['n_local_minima']} local minima)")
        st = [g["steps"] for g in sm["grad_norm_vs_rollout"]]
        gn = [g["grad_norm"] for g in sm["grad_norm_vs_rollout"]]
        a2.loglog(st, gn, "o-")
        a2.set_xlabel("rollout steps")
        a2.set_ylabel("||grad_address loss||")
        a2.set_title("gradient norm vs rollout length")
        fig.tight_layout()
        p = os.path.join(save_dir, "retrieval_fig3_address_landscape.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_retrieval(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Run the full write -> address -> retrieve battery and write JSON + figures."""
    config = config or get_default_config()
    cfg = config.experiment_retrieval
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "designed_not_learned": True,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "lam",
                "f",
                "barrier",
                "payload_kappa",
                "bump_width",
                "dt",
                "gamma",
                "steps",
                "tail_frac",
                "n_subsample",
                "n_query_per_item",
                "query_sigma_theta",
                "query_sigma_r",
                "query_sigma_p",
                "address_lr",
                "address_steps",
                "payload_seed",
                "payload_tol",
            )
        },
    }

    results["item1_selectivity"] = item1_selectivity(cfg, K=2, seed=seed)
    results["item2_mass_key"] = item2_mass_key(cfg, K=cfg.mass_probe_K, seed=seed)
    results["item3_write_modes"] = item3_write_modes(cfg, seed=seed)
    results["item4_interference"] = item4_interference(cfg, seed=seed)
    results["item5_restructuring"] = item5_restructuring(
        cfg, K=cfg.restructure_K, seed=seed
    )
    results["item5c_annealed_restructuring"] = item5c_annealed_restructuring(
        cfg, K=cfg.restructure_K, seed=seed
    )
    results["item5b_smoothness"] = item5b_smoothness(
        cfg, K=cfg.restructure_K, seed=seed
    )

    results["figures"] = _plot_all(results, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_retrieval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment RETRIEVAL: hand-built write/address/retrieve loop"
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

    res = run_experiment_retrieval(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "item5_restructuring"}, indent=2
        )[:4000]
    )


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_retrieval
    cfg.n_query_per_item = 16
    cfg.steps = 300
    cfg.item_counts = [2, 4, 8]
    cfg.mass_jitter_n = 3
    cfg.mass_n = 9
    cfg.mass_n_vec = 5
    cfg.address_steps = 40
    cfg.restructure_offsets = [1, 4]
    cfg.smooth_n_theta = 32
    cfg.smooth_rollout_steps = [50, 100, 200]
    cfg.anneal_n_stages = 3
    cfg.tm_steps = 400


if __name__ == "__main__":
    main()
