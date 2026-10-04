"""Experiment RETRY-COMPUTE: the accuracy-vs-compute curve for CLU retrieval,
done properly (w23 — the novel-property flagship).

⭐ **What this promotes.** w22's `exp_hopfield_capacity` retry was a SINGLE rung
(one boosted re-relaxation on the low-confidence half, +46.9pp at ×1.5 compute).
A referee asks three questions that a single point cannot answer:

  1. is the *curve* real — multiple compute budgets, monotone?
  2. is it the *physics* (the Lorentz-boost-style directed re-launch) or would any
     stochastic-restart heuristic match it?
  3. can a feedforward memory given the SAME extra compute draw the same curve?

This experiment turns the one demo point into a defensible **accuracy-vs-compute
curve with five pre-registered controls**, on the w22 retrieval protocol (Gaussian
noise queries, matched to ``MAGICS-LAB/UHop`` ``memory_retrieval_noise.py``), at
≥2 load levels (M) and ≥2 noise levels (σ). The honest compute unit is the
**relaxation-step count**; wall-clock is secondary.

**The six lines (Item 1 + Item 2 controls):**

- ``clu_gated``      — CLU retry: k boosted re-relaxation rounds, **confidence
                       gated** (retry only reads below a cosine threshold, re-gated
                       each round). The headline: adaptive spend.
- ``ungated_all``    — the SAME boost applied to ALL reads (no gate). Quantifies the
                       gate's contribution (w22: −38pp blank guard says gating is
                       load-bearing).
- ``ensemble``       — k+1 independent CLU settles from the query with random start
                       momenta, keep the best-confidence read. The fair "is it the
                       boost or just k tries?" rival.
- ``random_kick``    — CLU retry with the directed boost replaced by an
                       **equal-energy random** perturbation. Is the boost doing
                       anything a kick doesn't? (N1: test this honestly.)
- ``feedforward_nn`` — the trivial NN baseline given the same budget: k+1 augmented
                       nearest-neighbour reads, majority-vote the identity (TTA).
                       CM-23's "a curve feedforward memories cannot draw" survives
                       ONLY if this is flat-or-worse.
- ``hopfield_ksteps``— the closed-form modern-Hopfield line iterated k+1 steps.

⚠ Compute-axis honesty: CLU methods are placed at their MEASURED relaxation-step
multiplier (gated/kick are **sub-linear** in k — the adaptive-compute advantage).
The feedforward-NN and Hopfield lines are placed at their matched *budget*
multiplier (k+1); mapping one NN/Hopfield read to one CLU settle is **generous to
the baselines** and is stated in the report.

⭐ **w24 extension — the AMBIGUITY regimes (the headroom benchmark).** w23 proved the
mechanism but could not win the benchmark: on masked-pixel MNIST the trivial NN floor
sits at 0.99–1.00 and beats gated retry in every cell (N90). That is a **headroom**
problem — there is nothing for extra compute to buy when the baseline is at ceiling.
The fix must build headroom out of **AMBIGUITY** (several stored items are consistent
with the query, so the surviving evidence does not uniquely identify one) and **NOT**
out of **DESTRUCTION** (full-field Gaussian noise past the σ≈0.4 basin-capture cliff —
the query has left every well and *no* retry can recover it; measured +0 lift there).
Two regimes are added, both selected by the cheap **headroom gate** (``--headroom``)
*before* any ladder is spent:

- ``iid:block``     — contiguous (correlated) occlusion, area fraction ``f``:
                      the surviving crop is consistent with several stored digits.
                      This is the direct fix for "iid-surviving pixels uniquely
                      identify the pattern", which is *why* the NN floor wins today.
- ``crowded:mask``  — the store is a tight nearest-neighbour cluster, so the median
                      spacing falls at/below the packing bound Δ_req ≈ 3.1·max(s,σ_q)
                      and basins genuinely overlap. Ambiguity from GEOMETRY.

Runnable: ``uv run python -m chlu.experiments.exp_retry_compute --quick`` or via the
CLI ``chlu exp-retry-compute [--project N] [--seed I] [--quick] [--headroom]``.
"""

import json
import math
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.experiments.exp_hopfield_capacity import (
    _cosine,
    _median_nn_distance,
    build_clu_memory,
    dropout_query,
    hopfield_retrieve,
    load_patterns,
    noise_query,
)
from chlu.experiments.exp_hopfield_capacity import _softmax as _softmax_act


# ---------------------------------------------------------------------------
# CLU settle with an explicit per-query initial momentum (x64-safe, memory-safe)
# ---------------------------------------------------------------------------


def _settle(model, Q0, P0, steps, dt, gamma, tail_steps, chunk):
    """Damped Verlet from each (q0, p0); read = mean q over the last tail_steps.

    Generalizes ``exp_hopfield_capacity._settle_read`` to accept a launch
    momentum (needed for the boost / kick / ensemble start). x64-safe: launch in
    the ambient float dtype so the lax.scan carry types agree (the w22 landmine).
    NaN-guard: any query whose rollout diverged reads back its launch point.
    """
    tail_steps = int(max(1, tail_steps))
    settle_steps = int(max(1, steps - tail_steps))
    fdtype = jnp.result_type(float)
    Q0 = jnp.asarray(Q0, dtype=fdtype)
    P0 = jnp.asarray(P0, dtype=fdtype)

    def per_query(q0, p0):
        def step_fn(state, _):
            return model.step(state, dt, gamma), None

        (q1, p1), _ = jax.lax.scan(step_fn, (q0, p0), None, length=settle_steps)
        traj = model(q1, p1, tail_steps, dt, gamma)
        dim = q0.shape[0]
        read = jnp.mean(traj[:, :dim], axis=0)
        bad = jnp.any(~jnp.isfinite(read))
        return jnp.where(bad, q0, read)

    f = eqx.filter_jit(jax.vmap(per_query))
    outs = []
    for i in range(0, Q0.shape[0], chunk):
        outs.append(np.asarray(f(Q0[i : i + chunk], P0[i : i + chunk])))
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Label-free confidence + accuracy
# ---------------------------------------------------------------------------


def _confidence_and_nn(reads, patterns):
    """Return (cosine-to-nearest-well, nearest-well-index). Confidence is
    LABEL-FREE (cosine to the retrieved pattern, not to the truth)."""
    R = np.asarray(reads)
    P = np.asarray(patterns)
    d2 = np.sum((R[:, None, :] - P[None, :, :]) ** 2, axis=-1)  # (Nq, M)
    nn = np.argmin(d2, axis=1)
    cos = _cosine(R, P[nn])
    return np.asarray(cos), nn


def _acc(nn, true_idx):
    return float(np.mean(np.asarray(nn) == np.asarray(true_idx)))


def _dt_of(cfg, s):
    return cfg.clu_dt if cfg.clu_dt > 0 else 0.5 * s / np.sqrt(cfg.clu_b)


# ---------------------------------------------------------------------------
# CLU-gated / ungated / random-kick retry ladder (directed boost vs kick vs all)
# ---------------------------------------------------------------------------


def _retry_ladder(
    model, Q, reads0, patterns, true_idx, cfg, dt, threshold, mode, rng
):
    """Confidence-gated, lock-on-retry boosted re-relaxation ladder.

    Returns {k: (identity_acc, compute_mult)} for each k in ``cfg.retry_ladder``.

    Per round we re-relax the ``retry_step_frac`` lowest-confidence *eligible*
    reads with a directed boost (or, in "kick" mode, an equal-energy random
    perturbation) launched from the current settled point toward the query, and
    replace them **unconditionally**. Eligibility = ``(not yet retried) and
    (confidence < threshold)``; a retried read is **locked** so it is never
    retried again. Confidence = cosine to the nearest stored well.

    ⚠ **Why this design (measured, not assumed).** After a damped settle every
    particle sits *in some well*, so cosine-to-nearest is high (≈0.95–1.0) whether
    the well is right or wrong — it is a good *ranking* signal (correct 0.998 vs
    wrong 0.949) but a useless *acceptance* signal (a boost into the RIGHT well
    does not raise cosine above the wrong well it left). So no per-item accept rule
    works; the GATE (retry the low-confidence, wrong-enriched tail) is the
    load-bearing element, exactly w22's finding (−38pp blank guard). Re-gating the
    same reads every round without locking OSCILLATES (boosting an already-correct
    read corrupts it); locking makes the ladder monotone, and the threshold makes
    the gate **auto-stop** spending compute once the low-confidence tail is
    exhausted — the honest adaptive-compute signature.

    mode:
      "gated"   — the eligibility gate above (directed boost). Sub-linear compute.
      "ungated" — retry ALL reads every round, no lock, no threshold (the no-gate
                  control; compute (k+1)×). Quantifies the gate's contribution.
      "kick"    — the SAME gate/lock as "gated" but the directed boost is replaced
                  by an equal-energy random-direction kick (the mechanism control).

    ⚠ JIT-shape-stability: each round settles the FULL population so the vmap batch
    dim is constant (one compile per (M, s)); updates and the compute count are
    restricted to the retried subset. The reported compute (relaxation steps) is
    the honest retried count, not the full-population wall-clock."""
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    chunk = cfg.rollout_chunk
    Nq = Q.shape[0]
    base = Nq * S
    step_n = max(1, int(round(cfg.retry_step_frac * Nq)))

    reads = np.array(reads0, dtype=float)
    locked = np.zeros(Nq, dtype=bool)
    total_steps = base
    out = {0: (_acc(_confidence_and_nn(reads, patterns)[1], true_idx), 1.0)}
    max_k = max(cfg.retry_ladder)
    Qn = np.asarray(Q, dtype=float)

    for j in range(1, max_k + 1):
        cos, _ = _confidence_and_nn(reads, patterns)
        if mode == "ungated":
            cand = np.arange(Nq)  # no gate, no lock
        else:
            eligible = (~locked) & (cos < threshold)
            cm = np.where(eligible, cos, np.inf)
            cand = np.argsort(cm)[:step_n]
            cand = cand[np.isfinite(cm[cand])]  # drop padding when eligible < step_n

        if len(cand) > 0:
            direction = Qn - reads  # full population (constant batch dim)
            if mode == "kick":
                mag = cfg.retry_boost * np.linalg.norm(direction, axis=1, keepdims=True)
                d = rng.normal(size=reads.shape)
                d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
                p0 = mag * d
            else:  # gated / ungated — directed boost toward the query
                p0 = cfg.retry_boost * direction
            new = _settle(model, reads, p0, S, dt, cfg.clu_gamma, tail, chunk)
            reads[cand] = new[cand]  # unconditional within the retried set
            if mode != "ungated":
                locked[cand] = True
            total_steps += len(cand) * S  # honest: only the retried subset charged

        if j in cfg.retry_ladder:
            _, nn_j = _confidence_and_nn(reads, patterns)
            out[j] = (_acc(nn_j, true_idx), total_steps / base)
    return out


# ---------------------------------------------------------------------------
# Ensemble-of-k-reads (k independent random starts, keep best confidence)
# ---------------------------------------------------------------------------


def _ensemble_ladder(model, Q, reads0, patterns, true_idx, cfg, dt, rng):
    """k+1 independent CLU settles from the query with random launch momenta
    (energy-matched to the boost), keep the best-confidence read. Compute (k+1)×."""
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    chunk = cfg.rollout_chunk
    Qn = np.asarray(Q, dtype=float)

    best = np.array(reads0, dtype=float)
    best_cos, best_nn = _confidence_and_nn(best, patterns)
    # per-query energy scale matched to the round-1 boost magnitude
    scale = cfg.retry_boost * np.linalg.norm(Qn - best, axis=1, keepdims=True)

    out = {0: (_acc(best_nn, true_idx), 1.0)}
    max_k = max(cfg.retry_ladder)
    for j in range(1, max_k + 1):
        d = rng.normal(size=Qn.shape)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        p0 = scale * d
        r = _settle(model, Qn, p0, S, dt, cfg.clu_gamma, tail, chunk)
        r_cos, _ = _confidence_and_nn(r, patterns)
        take = r_cos > best_cos
        best[take] = r[take]
        best_cos[take] = r_cos[take]
        if j in cfg.retry_ladder:
            _, nn_j = _confidence_and_nn(best, patterns)
            out[j] = (_acc(nn_j, true_idx), float(j + 1))
    return out


# ---------------------------------------------------------------------------
# Feedforward-NN matched compute (TTA augmentation + majority vote)
# ---------------------------------------------------------------------------


def _majority(votes):
    """Majority vote over axis 0 of an (R, Nq) int array."""
    R, Nq = votes.shape
    out = np.empty(Nq, dtype=int)
    for i in range(Nq):
        vals, counts = np.unique(votes[:, i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _feedforward_ladder(Q, patterns, true_idx, cfg, rng):
    """The trivial feedforward memory given the same budget: k+1 nearest-neighbour
    reads over test-time-augmented queries, majority-vote the identity. Placed at
    the matched budget multiplier (k+1). One NN read ≪ one CLU settle — GENEROUS to
    the baseline (stated in the report)."""
    P = np.asarray(patterns, dtype=float)
    Qn = np.asarray(Q, dtype=float)

    def nn_idx(q):
        d2 = np.sum((q[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        return np.argmin(d2, axis=1)

    votes = [nn_idx(Qn)]
    out = {0: (_acc(votes[0], true_idx), 1.0)}
    max_k = max(cfg.retry_ladder)
    for j in range(1, max_k + 1):
        aug = np.clip(np.abs(Qn + rng.normal(size=Qn.shape) * cfg.ff_aug_sigma), 0, 1)
        votes.append(nn_idx(aug))
        if j in cfg.retry_ladder:
            maj = _majority(np.stack(votes, axis=0))
            out[j] = (_acc(maj, true_idx), float(j + 1))
    return out


# ---------------------------------------------------------------------------
# Hopfield-k-steps (the closed-form line iterated)
# ---------------------------------------------------------------------------


def _hopfield_ladder(Q, patterns, true_idx, cfg):
    """Modern-Hopfield update iterated k+1 steps. beta = auto-sharpened rule
    (β·⟨x,x⟩≈200, floored at the repo β), matching exp_hopfield_capacity."""
    P = np.asarray(patterns, dtype=float)
    self_overlap = float(np.mean(np.sum(P * P, axis=-1)))
    beta = (
        cfg.hopfield_beta_tuned
        if cfg.hopfield_beta_tuned > 0
        else max(cfg.hopfield_beta, 200.0 / (self_overlap + 1e-9))
    )
    out = {}
    for k in cfg.retry_ladder:
        xhat = hopfield_retrieve(patterns, Q, beta, 1.0, k + 1, _softmax_act)
        X = np.asarray(xhat)
        d2 = np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1)
        out[k] = (_acc(nn, true_idx), float(k + 1))
    return out


# ---------------------------------------------------------------------------
# One (M, σ) cell — all six lines
# ---------------------------------------------------------------------------


def block_query(X, frac, key, rescale=True):
    """⭐ w24 — **contiguous (correlated) occlusion**: zero one contiguous region
    covering ``frac`` of each pattern at a random position, and (if ``rescale``)
    scale the survivors by ``1/(1-frac_eff)`` — the SAME survivor convention as
    ``dropout_query`` (``torch.dropout``), so ``block`` and ``mask`` differ ONLY in
    the *correlation structure* of the erasure.

    ⚠ **``rescale=False`` is the PARTIAL-KEY reading, and it is the one that gives
    ambiguity rather than destruction (measured, w24 gate iteration 1).** The
    ``1/(1-f)`` factor is a *training-time* dropout convention (it preserves the
    expected activation); under **iid** erasure the amplification is spread over the
    whole image and roughly unbiased, but under **contiguous** erasure it multiplies
    a surviving *crop* by 2–3.3× and throws the query outside every basin. Measured
    at ``f=0.4, M=128``: first-pass ``0.086`` and NN floor ``0.383`` — both lines
    destroyed, which is precisely the σ≈0.4-cliff failure mode this regime exists to
    avoid. With ``rescale=False`` the query is the stored pattern restricted to the
    surviving coordinates: it stays on the data manifold and the surviving crop is
    consistent with several stored items ⇒ **ambiguity**.

    **Why (the w24 design constraint).** Under iid ``dropout`` the survivors are
    spread over the whole image and *uniquely identify* the stored pattern — which
    is exactly why the trivial NN floor sits at 0.99–1.00 (N90) and there is no
    headroom for test-time compute to buy. A contiguous block leaves a *crop* that
    is consistent with SEVERAL stored items ⇒ **ambiguity**. Crucially the query
    stays inside the data manifold, unlike full-field Gaussian noise past the
    σ≈0.4 cliff, where the query leaves every basin and no retry can recover it.

    Square images (``D`` a perfect square) get a 2-D square block of side
    ``round(√D·√frac)``; otherwise the erasure is a contiguous *segment* of the flat
    vector (still correlated, and defined for any ``D`` — e.g. the synthetic arm).
    """
    drop, frac_eff = _block_drop_mask(int(X.shape[0]), int(X.shape[1]), frac, key)
    keep = (~drop).astype(X.dtype)
    out = X * keep
    return out / (1.0 - frac_eff) if rescale else out


def _block_drop_mask(N, D, frac, key):
    """(drop_mask (N,D) bool, frac_eff) for ``block_query`` — factored out so the
    *true* observed mask can be regenerated from the same key (see
    ``erasure_mask``) instead of being guessed from ``Q != 0``."""
    frac = float(min(max(frac, 0.0), 0.99))
    side = int(round(math.sqrt(D)))
    if side * side == D:
        bs = int(min(side, max(1, round(side * math.sqrt(frac)))))
        k1, k2 = jax.random.split(key)
        r0 = jax.random.randint(k1, (N, 1), 0, side - bs + 1)
        c0 = jax.random.randint(k2, (N, 1), 0, side - bs + 1)
        ax = jnp.arange(side)[None, :]
        rm = (ax >= r0) & (ax < r0 + bs)  # (N, side)
        cm = (ax >= c0) & (ax < c0 + bs)
        drop = (rm[:, :, None] & cm[:, None, :]).reshape(N, D)
        frac_eff = (bs * bs) / float(D)
    else:
        L = int(min(D, max(1, round(frac * D))))
        o = jax.random.randint(key, (N, 1), 0, D - L + 1)
        ax = jnp.arange(D)[None, :]
        drop = (ax >= o) & (ax < o + L)
        frac_eff = L / float(D)
    return drop, frac_eff


def erasure_mask(patterns, query_type, level, key, cfg=None):
    """The TRUE observed mask (N,D) bool, ``True`` = coordinate survived, for the
    erasure protocols; ``None`` for ``noise`` (nothing is erased).

    Regenerated from the SAME key ``make_query`` consumed, so it is the actual mask
    and not the guess ``Q != 0`` (a stored pixel may legitimately be 0). Needed for
    the ``feedforward_nn_masked`` honesty control."""
    X = patterns
    if query_type == "mask":
        return np.asarray(jax.random.bernoulli(key, 1.0 - float(level), X.shape))
    if query_type == "block":
        drop, _ = _block_drop_mask(int(X.shape[0]), int(X.shape[1]), level, key)
        return np.asarray(~drop)
    return None


def survivor_scale(query_type, level, cfg=None):
    """The factor the protocol multiplies surviving coordinates by (1 if none)."""
    if query_type == "mask":
        return 1.0 / (1.0 - float(level))
    if query_type == "block":
        if cfg is not None and not bool(cfg.block_rescale):
            return 1.0
        return 1.0 / (1.0 - float(level))
    return 1.0


def masked_nn_identity(Q, patterns, keep, scale=1.0):
    """⭐ w24 honesty control — nearest neighbour over the **observed coordinates
    only**: ``argmin_i Σ_{j observed} (q_j/scale − ξ_ij)²``.

    Under erasure the zeroed coordinates are *missing*, not *observed as zero*, so
    the harness's full-vector NN is **not** the maximum-likelihood identity rule;
    this one is (it is the MAP decision under the erasure model with a flat prior).
    It is the correct floor to beat before any "CLU wins the benchmark" claim, and
    it is reported whatever the outcome so it cannot be added post-hoc as an excuse.
    Compute-insensitive by construction (closed form, no iterate).

    Expanded ‖(q/scale − ξ)⊙keep‖² so no (N, M, D) tensor is ever materialised."""
    Qn = np.asarray(Q, dtype=np.float64) / float(scale)
    P = np.asarray(patterns, dtype=np.float64)
    K = np.asarray(keep, dtype=np.float64)
    A = K * Qn
    d2 = (
        np.sum(A * Qn, axis=1)[:, None]
        - 2.0 * (A @ P.T)
        + (K @ (P**2).T)
    )
    return np.argmin(d2, axis=1)


def make_query(patterns, query_type, level, key, cfg=None):
    """mask -> torch.dropout(level); noise -> clamp(|x+N(0,level)|,0,1);
    block -> contiguous occlusion of area fraction ``level`` (w24; survivor
    rescaling controlled by ``cfg.block_rescale``)."""
    if query_type == "mask":
        return dropout_query(patterns, level, key)
    if query_type == "noise":
        return noise_query(patterns, level, key)
    if query_type == "block":
        rescale = True if cfg is None else bool(cfg.block_rescale)
        return block_query(patterns, level, key, rescale=rescale)
    raise ValueError(f"unknown query_type {query_type!r}")


def select_store(pool, M, store_mode, cfg):
    """Choose the ``M`` stored patterns for a cell.

    ``"iid"``     — the first ``M`` of the (already randomly subsampled) pool. The
                    w23 default; median-NN spacing sits comfortably above the
                    packing bound (w23 measured slack ≈ 1.08, i.e. *no* slack, but
                    not overlapping either).
    ``"crowded"`` — ⭐ w24: the ``M`` nearest neighbours of an anchor pattern, i.e.
                    a tight cluster, then optionally contracted about its own
                    centroid by ``cfg.crowd_rho`` (``ξ' = c + ρ(ξ − c)``).
                    The ambiguity comes from the STORE GEOMETRY, not from the
                    query — so the query degradation can be kept mild and the
                    failure mode stays "which of these several items is it?"
                    rather than "the query has left the manifold".

                    ⚠ **The nearest-neighbour cluster ALONE does not create
                    ambiguity** (measured, w24 gate iteration 1: NN floor 1.000 at
                    every load and mask level, slack unchanged). The reason is
                    scale invariance: the well width ``s = clu_s_frac·median-NN`` is
                    **store-adaptive**, so a cluster that is k× tighter also gets k×
                    tighter wells, and the NN decision rule has no scale at all. The
                    ``crowd_rho`` contraction is the lever that actually bites,
                    because it shrinks ``median_NN`` (∝ ρ) while leaving the erasure
                    displacement (∝ ‖ξ'‖ ≈ ‖c‖, the centroid norm) untouched — so
                    the ratio ``σ_q / median_NN`` that sets the packing slack grows
                    as ``1/ρ``.
    """
    if store_mode == "iid":
        return pool[:M]
    if store_mode == "crowded":
        P = np.asarray(pool, dtype=np.float64)
        a = int(cfg.crowd_anchor) % P.shape[0]
        d2 = np.sum((P - P[a][None, :]) ** 2, axis=1)
        idx = np.argsort(d2)[: int(M)]
        store = pool[np.asarray(idx)]
        rho = float(getattr(cfg, "crowd_rho", 1.0))
        if rho != 1.0:
            c = jnp.mean(store, axis=0, keepdims=True)
            store = c + rho * (store - c)
        return store
    raise ValueError(f"unknown store_mode {store_mode!r}")


def parse_regimes(cfg):
    """``cfg.regimes`` -> [(store_mode, query_type), …].

    Empty ``regimes`` ⇒ the w23 grid: every ``cfg.query_types`` entry at
    ``store_mode="iid"`` (so the default config reproduces w23 exactly)."""
    if not cfg.regimes:
        return [("iid", qt) for qt in cfg.query_types]
    out = []
    for r in cfg.regimes:
        sm, _, qt = r.partition(":")
        out.append((sm or "iid", qt or "mask"))
    return out


def levels_for(cfg, store_mode, query_type):
    """The degradation levels swept for a regime (config-driven, no magic
    numbers): block -> ``block_fracs``; noise -> ``noise_levels``; a mask on a
    CROWDED store -> ``crowd_mask_fracs`` (kept mild — the geometry is the
    ambiguity); a mask on an iid store -> ``mask_fracs`` (the w23 grid)."""
    if query_type == "block":
        return cfg.block_fracs
    if query_type == "noise":
        return cfg.noise_levels
    if store_mode == "crowded":
        return cfg.crowd_mask_fracs
    return cfg.mask_fracs


def packing_slack(patterns, Q, s):
    """(slack, median_NN, σ_q) with ``slack = median_NN / (3.1·max(s, σ_q))``.

    ``Δ_req ≈ 3.1·max(w, σ_q)`` is the w23 packing-bound rule (``w`` = the CLU well
    width ``s``; ``σ_q`` = the RMS query displacement). **slack < 1 ⇒ the store is
    past the packing bound**, i.e. basins overlap — the R-CROWD design target.

    ⚠ **Unit correction (w24).** ``σ_q`` must be the displacement **norm**
    ``RMS_i‖q_i − ξ_i‖``, in the same units as ``median_NN`` (cf.
    ``exp_dim_scaling``'s ``delta_req_sqrtd``). Coding it as a *per-element* RMS
    understates it by ``√D`` (=28 on MNIST), which pins ``max(s, σ_q) = s`` and makes
    the slack the **tautology** ``1/(3.1·clu_s_frac) = 1.075`` for every store and
    every query level — the value that was reported (as "≈1.08") in w23 and in gate
    iteration 1 for all 14 cells. It is a property of ``clu_s_frac``, not a
    measurement of the store."""
    P = np.asarray(patterns, dtype=float)
    Qn = np.asarray(Q, dtype=float)
    d_nn = float(_median_nn_distance(patterns))
    sigma_q = float(np.sqrt(np.mean(np.sum((Qn - P) ** 2, axis=1))))
    return d_nn / (3.1 * max(float(s), sigma_q) + 1e-12), d_nn, sigma_q


#: deterministic per-protocol seed offset (Python's ``hash`` is process-salted, so
#: it must NOT be used for reproducible PRNG seeding — handover reproducibility rule)
_QT_OFFSET = {"mask": 101, "noise": 202, "block": 303}
#: deterministic per-store-mode seed offset (same rule)
_SM_OFFSET = {"iid": 0, "crowded": 500}


def run_cell(cfg, patterns, query_type, level, seed, store_mode="iid", with_sweep=True):
    M = patterns.shape[0]
    true_idx = np.arange(M)
    qt_off = _QT_OFFSET.get(query_type, 0) + _SM_OFFSET.get(store_mode, 0)
    key = jax.random.PRNGKey(seed + int(1000 * level) + M + qt_off)
    key, kq = jax.random.split(key)
    Q = make_query(patterns, query_type, level, kq, cfg)

    s = cfg.clu_s_frac * _median_nn_distance(patterns)
    dt = _dt_of(cfg, s)
    model = build_clu_memory(patterns, s, cfg)

    # first pass (p0 = 0) — shared by every CLU method
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    reads0 = _settle(
        model,
        np.asarray(Q),
        np.zeros_like(np.asarray(Q)),
        S,
        dt,
        cfg.clu_gamma,
        tail,
        cfg.rollout_chunk,
    )
    first_acc = _acc(_confidence_and_nn(reads0, patterns)[1], true_idx)

    rng = np.random.default_rng(seed + M + int(1000 * level) + qt_off)
    thr = cfg.main_threshold
    lines = {
        "clu_gated": _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, thr, "gated", rng
        ),
        "ungated_all": _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, thr, "ungated", rng
        ),
        "random_kick": _retry_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, thr, "kick", rng
        ),
        "ensemble": _ensemble_ladder(
            model, Q, reads0, patterns, true_idx, cfg, dt, rng
        ),
        "feedforward_nn": _feedforward_ladder(Q, patterns, true_idx, cfg, rng),
        "hopfield_ksteps": _hopfield_ladder(Q, patterns, true_idx, cfg),
    }
    # 7th line (w24) — the ML-optimal erasure floor. Flat by construction (closed
    # form), plotted at the same budget multiplier (k+1) as feedforward_nn.
    keep = erasure_mask(patterns, query_type, level, kq, cfg)
    masked_acc = None
    if keep is not None:
        masked_acc = _acc(
            masked_nn_identity(
                Q, patterns, keep, survivor_scale(query_type, level, cfg)
            ),
            true_idx,
        )
        lines["feedforward_nn_masked"] = {
            k: (masked_acc, float(k + 1)) for k in cfg.retry_ladder
        }

    # threshold sweep (gated only), reported at the full ladder. Skipped on the
    # later seeds of a multi-seed cell (it is 4 extra gated ladders).
    thr_sweep = {}
    if with_sweep:
        for t in cfg.conf_thresholds:
            thr_sweep[f"{t:.2f}"] = _retry_ladder(
                model, Q, reads0, patterns, true_idx, cfg, dt, t, "gated", rng
            )

    slack, d_nn, sigma_q = packing_slack(patterns, Q, s)
    return {
        "M": int(M),
        "store_mode": store_mode,
        "regime": f"{store_mode}:{query_type}",
        "query_type": query_type,
        "level": float(level),
        "seed": int(seed),
        "s": float(s),
        "dt": float(dt),
        "median_nn": d_nn,
        "sigma_q": sigma_q,
        "packing_slack": slack,
        "first_pass_acc": first_acc,
        "nn_floor_masked": masked_acc,
        "lines": {
            name: {str(k): {"acc": v[0], "compute_mult": v[1]} for k, v in d.items()}
            for name, d in lines.items()
        },
        "threshold_sweep": {
            t: {str(k): {"acc": v[0], "compute_mult": v[1]} for k, v in d.items()}
            for t, d in thr_sweep.items()
        },
    }


# ---------------------------------------------------------------------------
# Item 2 — the HEADROOM GATE (run this BEFORE spending a full ladder)
# ---------------------------------------------------------------------------


def headroom_probe(cfg, patterns, query_type, level, seed, store_mode="iid"):
    """⭐ w24 Item 2 — the cheap gate. Measures ONLY what the gate needs: the CLU
    **first-pass** accuracy (one settle, k=0) and the **feedforward-NN floor** (one
    NN read, k=0), plus the packing slack. No ladder, no controls, no τ-sweep.

    A regime passes iff ``first_pass ∈ cfg.headroom_band`` **and**
    ``nn_floor < cfg.headroom_nn_ceiling``. If the NN floor is still at ceiling the
    regime has failed its purpose and the ladder must NOT be run into it — that is
    precisely the w23 failure (8 saturated cells, N90), and this check is ~1/50th
    the cost of finding out the expensive way."""
    M = patterns.shape[0]
    true_idx = np.arange(M)
    qt_off = _QT_OFFSET.get(query_type, 0) + _SM_OFFSET.get(store_mode, 0)
    key = jax.random.PRNGKey(seed + int(1000 * level) + M + qt_off)
    key, kq = jax.random.split(key)
    Q = make_query(patterns, query_type, level, kq, cfg)

    s = cfg.clu_s_frac * _median_nn_distance(patterns)
    dt = _dt_of(cfg, s)
    model = build_clu_memory(patterns, s, cfg)
    S = cfg.clu_steps
    tail = int(max(1, cfg.clu_tail_frac * S))
    Qn = np.asarray(Q)
    reads0 = _settle(
        model, Qn, np.zeros_like(Qn), S, dt, cfg.clu_gamma, tail, cfg.rollout_chunk
    )
    first_acc = _acc(_confidence_and_nn(reads0, patterns)[1], true_idx)

    P = np.asarray(patterns, dtype=float)
    d2 = np.sum((Qn[:, None, :] - P[None, :, :]) ** 2, axis=-1)
    nn_floor = _acc(np.argmin(d2, axis=1), true_idx)
    keep = erasure_mask(patterns, query_type, level, kq, cfg)
    nn_masked = (
        None
        if keep is None
        else _acc(
            masked_nn_identity(
                Q, patterns, keep, survivor_scale(query_type, level, cfg)
            ),
            true_idx,
        )
    )

    lo, hi = float(cfg.headroom_band[0]), float(cfg.headroom_band[1])
    in_band = lo <= first_acc <= hi
    nn_ok = nn_floor < float(cfg.headroom_nn_ceiling)
    slack, d_nn, sigma_q = packing_slack(patterns, Q, s)
    return {
        "regime": f"{store_mode}:{query_type}",
        "store_mode": store_mode,
        "query_type": query_type,
        "M": int(M),
        "level": float(level),
        "seed": int(seed),
        "first_pass_acc": round(first_acc, 4),
        "nn_floor": round(nn_floor, 4),
        "nn_floor_masked": None if nn_masked is None else round(nn_masked, 4),
        "median_nn": round(d_nn, 4),
        "sigma_q": round(sigma_q, 4),
        "s": round(float(s), 4),
        "packing_slack": round(slack, 4),
        "first_pass_in_band": bool(in_band),
        "nn_off_ceiling": bool(nn_ok),
        "passed": bool(in_band and nn_ok),
    }


def run_headroom_gate(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    seed: Optional[int] = None,
):
    """Sweep the gate over every (regime × load × level) cell and report which
    cells have real headroom. Cheap: 1 settle + 1 NN read per cell."""
    config = config or get_default_config()
    cfg = config.experiment_retry_compute
    base = cfg.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for ds in cfg.datasets:
        try:
            pool = load_patterns(ds, cfg.n_data_pool, base)
        except FileNotFoundError as e:
            rows.append({"dataset": ds, "skipped": str(e)})
            continue
        loads = [M for M in cfg.load_grid if M <= pool.shape[0]]
        for store_mode, qt in parse_regimes(cfg):
            for M in loads:
                patterns = select_store(pool, M, store_mode, cfg)
                for lv in levels_for(cfg, store_mode, qt):
                    r = headroom_probe(cfg, patterns, qt, lv, base, store_mode)
                    r["dataset"] = ds
                    rows.append(r)

    out = {
        "seed": base,
        "gate": {
            "band": list(cfg.headroom_band),
            "nn_ceiling": cfg.headroom_nn_ceiling,
            "rule": "pass = first_pass in band AND nn_floor < nn_ceiling",
            "block_rescale": bool(cfg.block_rescale),
            "crowd_rho": float(cfg.crowd_rho),
            "clu_s_frac": float(cfg.clu_s_frac),
            "clu_steps": int(cfg.clu_steps),
        },
        "cells": rows,
        "n_passed": sum(1 for r in rows if r.get("passed")),
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    p = os.path.join(results_dir, "exp_retry_compute_headroom.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    out["metrics_path"] = p
    return out


# ---------------------------------------------------------------------------
# Figure — grid of (M, σ) cells, six lines each
# ---------------------------------------------------------------------------


LINE_STYLE = {
    "clu_gated": ("o-", "#d62728", 2.0),
    "ungated_all": ("s--", "#ff7f0e", 1.2),
    "ensemble": ("^--", "#1f77b4", 1.2),
    "random_kick": ("v--", "#9467bd", 1.2),
    "feedforward_nn": ("D:", "#2ca02c", 1.2),
    "feedforward_nn_masked": ("*-.", "#17becf", 1.4),  # w24 ML-optimal erasure floor
    "hopfield_ksteps": ("x:", "#7f7f7f", 1.2),
}


_LVL_NAME = {"mask": "mask p", "noise": "σ", "block": "block frac"}


def _plot_grid(agg_cells, loads, levels, dataset, regime, save_dir):
    """One panel per (M, level); six lines each, with seed error bars when the
    aggregate carries >1 seed. ``agg_cells`` are aggregate rows (see
    ``aggregate_cells``), so single-seed and multi-seed runs plot identically."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    nrow, ncol = len(loads), len(levels)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False
    )
    qt = regime.split(":")[-1]
    lvl_name = _LVL_NAME.get(qt, "level")
    by_key = {(c["M"], round(c["level"], 4)): c for c in agg_cells}
    for i, M in enumerate(loads):
        for jx, lv in enumerate(levels):
            ax = axes[i][jx]
            c = by_key.get((M, round(lv, 4)))
            if c is None:
                ax.axis("off")
                continue
            for name, (mk, col, lw) in LINE_STYLE.items():
                d = c["lines"].get(name)
                if not d:
                    continue
                ks = sorted(int(k) for k in d)
                xs = [d[str(k)]["compute_mult"] for k in ks]
                ys = [d[str(k)]["acc"] for k in ks]
                es = [d[str(k)].get("acc_std", 0.0) for k in ks]
                ax.errorbar(
                    xs, ys, yerr=es, fmt=mk, color=col, lw=lw, ms=4,
                    capsize=2, elinewidth=0.8, label=name,
                )
            ax.set_title(
                f"M={M}, {lvl_name}={lv} (first {c['first_pass_acc']:.2f}, "
                f"n={c['n_seeds']} seeds)",
                fontsize=8,
            )
            ax.set_xlabel("compute (× first-pass budget)")
            ax.set_ylabel("identity accuracy")
            ax.grid(alpha=0.25)
            if i == 0 and jx == 0:
                ax.legend(fontsize=6)
    fig.suptitle(
        f"Accuracy vs test-time compute — {dataset} / regime {regime} "
        "(CLU-gated retry + 5 controls)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    tag = regime.replace(":", "-")
    p = os.path.join(save_dir, f"retry_compute_grid_{dataset}_{tag}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Multi-seed aggregation (w24 — headline cells require ≥3 seeds)
# ---------------------------------------------------------------------------


def aggregate_cells(cells):
    """Group per-seed cells by (dataset, regime, M, level) and average every line
    rung across seeds. Returns rows carrying ``acc`` (seed mean), ``acc_std``
    (population sd over seeds) and ``compute_mult`` (seed mean), plus the
    per-cell ``margin_vs_nn_pp`` and its seed sd — the quantity Item 4 turns on."""
    groups = {}
    for c in cells:
        if "lines" not in c:
            continue
        k = (c.get("dataset", "?"), c.get("regime", "iid:mask"), c["M"], round(c["level"], 6))
        groups.setdefault(k, []).append(c)

    out = []
    for (ds, regime, M, lv), cs in groups.items():
        names = sorted({n for c in cs for n in c["lines"]})
        lines = {}
        for n in names:
            rungs = sorted({int(k) for c in cs for k in c["lines"][n]})
            lines[n] = {}
            for k in rungs:
                accs = [c["lines"][n][str(k)]["acc"] for c in cs if str(k) in c["lines"][n]]
                comps = [
                    c["lines"][n][str(k)]["compute_mult"] for c in cs if str(k) in c["lines"][n]
                ]
                lines[n][str(k)] = {
                    "acc": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "compute_mult": float(np.mean(comps)),
                }
        # per-seed best-over-ladder margin (gated best − NN best), then mean/sd
        margins, margins_masked = [], []
        for c in cs:
            g = max(v["acc"] for v in c["lines"]["clu_gated"].values())
            f = max(v["acc"] for v in c["lines"]["feedforward_nn"].values())
            margins.append(100.0 * (g - f))
            fm = c["lines"].get("feedforward_nn_masked")
            if fm:
                margins_masked.append(100.0 * (g - max(v["acc"] for v in fm.values())))
        out.append({
            "dataset": ds,
            "regime": regime,
            "M": M,
            "level": lv,
            "n_seeds": len(cs),
            "seeds": [c.get("seed") for c in cs],
            "first_pass_acc": float(np.mean([c["first_pass_acc"] for c in cs])),
            "first_pass_std": float(np.std([c["first_pass_acc"] for c in cs])),
            "packing_slack": float(np.mean([c.get("packing_slack", float("nan")) for c in cs])),
            "margin_vs_nn_pp": float(np.mean(margins)),
            "margin_vs_nn_pp_std": float(np.std(margins)),
            "margin_vs_nn_masked_pp": (
                float(np.mean(margins_masked)) if margins_masked else None
            ),
            "margin_vs_nn_masked_pp_std": (
                float(np.std(margins_masked)) if margins_masked else None
            ),
            "lines": lines,
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_retry_compute(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_retry_compute
    seed = cfg.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)
    seed_list = [seed + i for i in range(max(1, int(cfg.n_seeds)))]
    regimes = parse_regimes(cfg)

    results = {
        "seed": seed,
        "seeds": seed_list,
        "regimes": [f"{sm}:{qt}" for sm, qt in regimes],
        "designed_not_learned": True,
        "protocol": {
            "query": "mask = torch.dropout(p); noise = clamp(|x+N(0,sigma)|,0,1); "
            "block = contiguous occlusion of area frac (survivors scaled 1/(1-f))",
            "store": "iid = pool[:M]; crowded = the M nearest neighbours of an "
            "anchor (past the packing bound, overlapping basins)",
            "compute_unit": "relaxation steps / (Nq*clu_steps) first-pass budget",
            "note": "feedforward_nn and hopfield_ksteps placed at matched budget "
            "multiplier (k+1); one NN/Hopfield read << one CLU settle "
            "(generous to the baselines).",
        },
        "config": {
            k: getattr(cfg, k)
            for k in (
                "datasets",
                "load_grid",
                "noise_levels",
                "query_types",
                "mask_fracs",
                "regimes",
                "block_fracs",
                "block_rescale",
                "crowd_mask_fracs",
                "crowd_anchor",
                "crowd_rho",
                "headroom_band",
                "headroom_nn_ceiling",
                "n_seeds",
                "sweep_seeds",
                "retry_ladder",
                "retry_step_frac",
                "conf_thresholds",
                "main_threshold",
                "retry_boost",
                "clu_s_frac",
                "clu_b",
                "clu_alpha",
                "clu_gamma",
                "clu_steps",
                "clu_kinetic_mode",
                "ff_aug_sigma",
                "n_data_pool",
            )
        },
    }

    all_cells = []
    figs = []
    for si, sd in enumerate(seed_list):
        for ds in cfg.datasets:
            try:
                pool = load_patterns(ds, cfg.n_data_pool, sd)
            except FileNotFoundError as e:
                all_cells.append({"dataset": ds, "skipped": str(e)})
                continue
            loads = [M for M in cfg.load_grid if M <= pool.shape[0]]
            for store_mode, qt in regimes:
                for M in loads:
                    patterns = select_store(pool, M, store_mode, cfg)
                    for lv in levels_for(cfg, store_mode, qt):
                        cell = run_cell(
                            cfg, patterns, qt, lv, sd,
                            store_mode=store_mode,
                            with_sweep=(si < int(cfg.sweep_seeds)),
                        )
                        cell["dataset"] = ds
                        all_cells.append(cell)

    results["cells"] = all_cells
    agg = aggregate_cells(all_cells)
    results["aggregate"] = agg
    for ds in cfg.datasets:
        for store_mode, qt in regimes:
            reg = f"{store_mode}:{qt}"
            rows = [a for a in agg if a["dataset"] == ds and a["regime"] == reg]
            if not rows:
                continue
            figs += _plot_grid(
                rows,
                sorted({a["M"] for a in rows}),
                sorted({a["level"] for a in rows}),
                ds,
                reg,
                save_dir,
            )
    results["figures"] = figs

    # verdict summary: per cell, best gated accuracy over the ladder and the gap of
    # each control's best to it (the honest "does gated dominate at equal compute?"
    # is in the figure; this is the summary scalar).
    verdict = []
    for c in all_cells:
        if "lines" not in c:
            continue
        gated = c["lines"]["clu_gated"]
        g_best = max(v["acc"] for v in gated.values())
        g_at = {k: v for k, v in gated.items()}
        row = {
            "M": c["M"],
            "regime": c.get("regime", "iid:" + c["query_type"]),
            "query_type": c["query_type"],
            "level": c["level"],
            "seed": c.get("seed", seed),
            "first_pass_acc": c["first_pass_acc"],
            "gated_best_acc": round(g_best, 4),
            "gated_lift_pp": round(100.0 * (g_best - c["first_pass_acc"]), 2),
            "gated_compute_at_best": round(
                min(v["compute_mult"] for v in g_at.values() if v["acc"] == g_best), 3
            ),
        }
        for name in c["lines"]:
            if name == "clu_gated":
                continue
            best = max(v["acc"] for v in c["lines"][name].values())
            row[f"{name}_best_acc"] = round(best, 4)
            row[f"{name}_gap_pp"] = round(100.0 * (g_best - best), 2)
        verdict.append(row)
    results["verdict"] = verdict

    # ⭐ Item 4 — the seed-averaged verdict: does CLU-gated beat the NN floor?
    # A cell counts as a WIN only if the mean margin is positive AND exceeds one
    # seed sd (the pre-registered discriminator; single-seed "wins" do not count).
    verdict_agg = []
    for a in agg:
        g = a["lines"]["clu_gated"]
        g_best = max(v["acc"] for v in g.values())
        row = {
            "regime": a["regime"],
            "M": a["M"],
            "level": a["level"],
            "n_seeds": a["n_seeds"],
            "packing_slack": round(a["packing_slack"], 4),
            "first_pass_acc": round(a["first_pass_acc"], 4),
            "gated_best_acc": round(g_best, 4),
            "gated_lift_pp": round(100.0 * (g_best - a["first_pass_acc"]), 2),
            "gated_compute_at_best": round(
                min(v["compute_mult"] for v in g.values() if v["acc"] == g_best), 3
            ),
            "margin_vs_nn_pp": round(a["margin_vs_nn_pp"], 2),
            "margin_vs_nn_pp_std": round(a["margin_vs_nn_pp_std"], 2),
            "beats_nn": bool(
                a["margin_vs_nn_pp"] > 0.0
                and a["margin_vs_nn_pp"] > a["margin_vs_nn_pp_std"]
            ),
        }
        # the ML-optimal erasure floor: a benchmark WIN must clear this too
        mm, ms = a.get("margin_vs_nn_masked_pp"), a.get("margin_vs_nn_masked_pp_std")
        row["margin_vs_nn_masked_pp"] = None if mm is None else round(mm, 2)
        row["margin_vs_nn_masked_pp_std"] = None if ms is None else round(ms, 2)
        row["beats_nn_masked"] = (
            None if mm is None else bool(mm > 0.0 and mm > (ms or 0.0))
        )
        row["benchmark_win"] = bool(
            row["beats_nn"] and (row["beats_nn_masked"] is not False)
        )
        for name in a["lines"]:
            if name == "clu_gated":
                continue
            best = max(v["acc"] for v in a["lines"][name].values())
            row[f"{name}_best_acc"] = round(best, 4)
            row[f"{name}_gap_pp"] = round(100.0 * (g_best - best), 2)
        verdict_agg.append(row)
    results["verdict_agg"] = verdict_agg
    results["item4_any_regime_beats_nn"] = any(r["beats_nn"] for r in verdict_agg)
    results["item4_any_regime_benchmark_win"] = any(
        r["benchmark_win"] for r in verdict_agg
    )

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_retry_compute_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_retry_compute
    cfg.load_grid = [32, 64]
    cfg.noise_levels = [0.3]
    cfg.mask_fracs = [0.5]
    cfg.query_types = ["mask", "noise"]
    cfg.retry_ladder = [0, 1, 2]
    cfg.conf_thresholds = [0.97, 1.0]
    cfg.clu_steps = 40
    cfg.n_data_pool = 200
    cfg.rollout_chunk = 64
    cfg.regimes = []
    cfg.n_seeds = 1
    cfg.sweep_seeds = 1


def apply_ambiguity(config: CHLUConfig) -> None:
    """⭐ w24 preset — the AMBIGUITY (headroom) grid: the two regimes of the
    headroom benchmark at ≥3 seeds, replacing the w23 iid-mask/noise grid.

    The *levels* below are the ones the Item-2 headroom gate selected (gate =
    first-pass ∈ band AND NN floor off ceiling) — a regime is selected for
    HEADROOM, never for whether CLU wins in it (task ⚠ standing trap).

    Gate provenance (seed 0, ``--headroom``, MNIST, ``clu_steps=150``):

    ==============================  =====  ==========  ========  =====
    cell                            first  NN floor    slack     gate
    ==============================  =====  ==========  ========  =====
    iid:block f=0.20 M=128          0.641  0.945       0.341     PASS
    iid:block f=0.20 M=256          0.527  0.902       0.327     PASS
    iid:block f=0.30 M=128/256      0.438/0.254  0.773/0.703     (2nd level, NN off ceiling)
    crowded:mask rho=.25 p=.5 M=128 0.625  0.734       0.063     PASS
    crowded:mask rho=.25 p=.5 M=256 0.684  0.738       0.063     PASS
    crowded:mask rho=.25 p=.3 M=*   0.914/0.930  0.953/0.949     (2nd level)
    ==============================  =====  ==========  ========  =====

    Rejected by the same gate, and why (both are recorded so nobody re-runs them):
    ``block_rescale=False`` (0/6 cells: MNIST ink is central, so an unamplified
    contiguous crop at f≥0.3 destroys *both* lines — first-pass 0.016, NN 0.055 at
    f=0.5) and ``crowd_rho=1.0`` i.e. the plain nearest-neighbour cluster
    (0/4 cells: NN floor 0.992–1.000 — crowding without contraction is neutralised
    by the store-adaptive well width)."""
    cfg = config.experiment_retry_compute
    cfg.regimes = ["iid:block", "crowded:mask"]
    cfg.block_fracs = [0.2, 0.3]
    cfg.block_rescale = True
    cfg.crowd_mask_fracs = [0.3, 0.5]
    cfg.crowd_rho = 0.25
    cfg.load_grid = [128, 256]
    cfg.n_seeds = 3
    cfg.sweep_seeds = 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Retry-compute: accuracy-vs-compute curve, CLU-gated + 5 controls"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--dataset", help="Override datasets (comma-separated)")
    parser.add_argument(
        "--ambiguity",
        action="store_true",
        help="w24 ambiguity/headroom regimes (block occlusion + crowded store)",
    )
    parser.add_argument(
        "--headroom",
        action="store_true",
        help="Item 2 only: the cheap headroom gate (first-pass + NN floor), no ladder",
    )
    args = parser.parse_args()

    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir = str(paths["plots"])
    else:
        config = get_default_config()
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        apply_quick(config)
    if args.ambiguity:
        apply_ambiguity(config)
    if args.dataset:
        config.experiment_retry_compute.datasets = args.dataset.split(",")

    if args.headroom:
        res = run_headroom_gate(config=config, save_dir=save_dir, seed=args.seed)
        print(json.dumps({"gate": res["gate"], "cells": res["cells"],
                          "n_passed": res["n_passed"]}, indent=2)[:8000])
        return

    res = run_experiment_retry_compute(config=config, save_dir=save_dir, seed=args.seed)
    print(
        json.dumps(
            {
                "item4_any_regime_beats_nn": res["item4_any_regime_beats_nn"],
                "verdict_agg": res["verdict_agg"],
                "figures": res["figures"],
            },
            indent=2,
        )[:12000]
    )


if __name__ == "__main__":
    main()
