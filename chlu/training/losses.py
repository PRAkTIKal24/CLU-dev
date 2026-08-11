"""Loss functions for training."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """
    Mean squared error loss.
    
    Args:
        pred: Predictions
        target: Ground truth
    
    Returns:
        MSE loss (scalar)
    """
    return jnp.mean((pred - target) ** 2)


def energy_loss(model, q: jnp.ndarray, p: jnp.ndarray) -> float:
    """
    Energy minimization loss for CHLU sleep phase.
    
    Encourages states to settle into low-energy configurations
    (the "energy valley").
    
    Args:
        model: CHLU model with H(q, p) method
        q: Position batch (batch_size, dim)
        p: Momentum batch (batch_size, dim)
    
    Returns:
        Mean energy (scalar)
    """
    # Compute energy for each state in batch
    energies = jax.vmap(model.H)(q, p)
    
    return jnp.mean(energies)


# ==========================================================================
# ⭐⭐ C2W11 — THE ORGANIZER LOSS PACKAGE (charter §A34.9 (a)-(e))
# ==========================================================================
# The theorist's formalization (`c2w11-loss-package.md` Deliverable 1) is the
# spec these functions implement. Five terms ship; ⛔ **(f) kinetics is a
# declared NOT-RUN this wave** — term (c) exists so that what a kinetics head
# would select on is *defended* rather than assumed (§A34.9).
#
# ⛔⛔ THE COEFFICIENT-ZERO INVARIANT (the C2W2 invariant, pytest-asserted in
# `tests/test_c2w11_organizer.py`): every coefficient at 0 must be
# **BIT-IDENTICAL** to the shipped objective. It is enforced *structurally*
# rather than numerically: each term sits behind a Python-level `if coeff ==
# 0.0` branch on a **static** float, so at zero the term's graph is never
# built, no RNG stream is advanced, and no `0 * NaN` can be injected. (A term
# that consumes RNG at coefficient zero is the usual silent killer —
# §1(a)(iii)(1) of the loss package.)
#
# ⛔ Wells are never named semantically (`PREREG-TierII.md` §2.6): wells,
# channels and features carry integer indices and nothing else.
# ==========================================================================


@dataclass(frozen=True)
class C2W11LossCoeffs:
    """The organizer's coefficients. ⭐ **Every one defaults to 0.0**, so the
    default object reproduces the shipped (spoke-A) objective bit-identically.

    ``lambda_weak`` is **the wave's single ablation** (§A34.6): ONE extra
    coefficient on the organization objective, raced on the same cells. ⛔ It is
    a coefficient, not a new capability — §3.1's one-capability-at-a-time
    doctrine is not stretched.
    """

    lambda_org: float = 0.0    # (a) label-free organization, through the write
    lambda_band: float = 0.0   # (a)'s two-sided d/s hinge
    lambda_share: float = 0.0  # (b) sharing / refresh
    lambda_shape: float = 0.0  # (c) curvature-shape (the w20 defender)
    lambda_read: float = 0.0   # (d) set-level compositional read
    lambda_cal: float = 0.0    # (e) calibration / feature-dropout novelty
    lambda_weak: float = 0.0   # ⭐ the ablation: labels shape well-sharing

    def as_flag_table(self) -> Dict[str, float]:
        """Non-zero coefficients only — the arm's ledger line."""
        return {k: float(v) for k, v in self.__dict__.items() if float(v) != 0.0}

    @property
    def any_live(self) -> bool:
        return bool(self.as_flag_table())


# --------------------------------------------------------------------------
# (a) the label-free organization term  —  the DESIGNED write->phi gradient
# --------------------------------------------------------------------------
def placed_sites(codes: jnp.ndarray, jig: jnp.ndarray, radius: float
                 ) -> jnp.ndarray:
    """``u_j = R e_j + jig_j`` — the placing write's assignment, algebraically.

    ⭐ This is the **designed** write->φ organization channel (§A28.1): a direct
    algebraic path ``dL/du . du/dz_addr (= I) . dz_addr/dphi``. It is neither
    implicit-at-settle nor trajectory, which is why it is the **only** term of
    the package that is live before any well exists (grad norms at init are
    ``1e-10 - 1e-9`` for every other channel and ``O(1)`` only after the write).
    """
    return float(radius) * codes + jig


def placement_centroids(sites: jnp.ndarray, indicators: jnp.ndarray
                        ) -> jnp.ndarray:
    """``(B, N_a) x (N_a, d) -> (B, d)`` the placed set-code centroid ``c(x)``."""
    w = indicators / (jnp.sum(indicators, axis=-1, keepdims=True) + 1e-12)
    return w @ sites


def nt_xent(za: jnp.ndarray, zb: jnp.ndarray, temperature: float = 0.2
            ) -> jnp.ndarray:
    """NT-Xent over two views of the same batch. **Label-free by construction.**

    ⭐ §A34.6, ratified and counter-intuitive: *the organization objective is
    LABEL-FREE; supervision enters ONLY at the read head.* Its measured basis is
    the pass-3 inversion — task supervision built the **worst** address geometry
    beyond 2 SE (``simclr - randconv`` A1 = ``-0.1406 +/- 0.0508``, 0/3 seeds)
    while unfitted ``randconv`` bought the geometry for free.
    """
    a = za / (jnp.linalg.norm(za, axis=-1, keepdims=True) + 1e-12)
    b = zb / (jnp.linalg.norm(zb, axis=-1, keepdims=True) + 1e-12)
    logits = (a @ b.T) / float(temperature)
    lab = jnp.arange(a.shape[0])
    return 0.5 * (
        jnp.mean(-jax.nn.log_softmax(logits, axis=1)[lab, lab])
        + jnp.mean(-jax.nn.log_softmax(logits, axis=0)[lab, lab]))


def band_hinge(d_over_s: jnp.ndarray, lo: float = 2.5, hi: float = 2.9
               ) -> jnp.ndarray:
    """The **two-sided** operating-band hinge on the *measured* ``d/s``.

    ⛔ Two-sided is mandatory (loss package §1(a)(v)): below ``d/s = 2.01`` two
    equal wells **merge into one minimum**; at or above ``4.0`` the settled-point
    organization is exactly nearest-centroid VQ (``D = 0.0000``) and the dividend
    is structurally zero. ``L_org`` is the only term that can control ``d/s``.
    """
    return (jax.nn.relu(lo - d_over_s) ** 2 + jax.nn.relu(d_over_s - hi) ** 2)


def org_loss(codes: jnp.ndarray, jig: jnp.ndarray, radius: float,
             ind_a: jnp.ndarray, ind_b: jnp.ndarray, *,
             temperature: float = 0.2, d_over_s: Optional[jnp.ndarray] = None,
             lambda_band: float = 0.0, lo: float = 2.5, hi: float = 2.9
             ) -> jnp.ndarray:
    """Term **(a)**: ``L_org`` on the placed atom-group centroids + the band."""
    sites = placed_sites(codes, jig, radius)
    ca = placement_centroids(sites, ind_a)
    cb = placement_centroids(sites, ind_b)
    out = nt_xent(ca, cb, temperature)
    if float(lambda_band) != 0.0 and d_over_s is not None:
        out = out + float(lambda_band) * jnp.mean(band_hinge(d_over_s, lo, hi))
    return out


def reach_org_loss(sites: jnp.ndarray, launches: jnp.ndarray,
                   ind: jnp.ndarray, *, beta: float = 8.0,
                   sep_target: float = 0.0, nu_sep: float = 0.0) -> jnp.ndarray:
    """⭐⭐ Term **(a)**, the REACH instantiation — *place the wells where the
    store's own launches can reach them.*

    ``L_reach = E_x sum_{j in A(x)} softmin_c || u_j - q_c(x) ||`` + a two-sided
    separation hinge. **Label-free**: it consumes only the launch geometry
    ``q_c(x)`` (a function of the frozen ``phi``) and the item's own composition;
    ⛔ the target ``y`` is never read.

    ⚠ **REGISTERED AMENDMENT (PREREG.md §A1, filed before the claim cells).** The
    theorist's ``l_metric`` is an InfoNCE on placed centroids. It was implemented,
    run, and is raced here as an ablation — but the *measured* blocker on this
    substrate is not metric structure among items: it is that the matched-filter
    launch head puts only **22.6 %** of its channel picks inside ``A(x)`` while
    the cue-to-well displacement is already **0.0073** (i.e. the placement is
    ALREADY cue-aligned and the deficit is upstream). This term is the
    admissibility functional that attacks the measured blocker: *a needed well
    must lie inside a launch diamond*, which is §A20.3(c)'s own guard written as
    a differentiable objective. Both instantiations are reported.
    """
    d = jnp.linalg.norm(sites[None, None, :, :] - launches[:, :, None, :], axis=-1)
    soft = -(1.0 / float(beta)) * jax.nn.logsumexp(-float(beta) * d, axis=1)
    out = jnp.sum(soft * ind, axis=-1).mean() / jnp.maximum(jnp.sum(ind, -1).mean(), 1.0)
    if float(nu_sep) != 0.0:
        dd = jnp.sum((sites[:, None, :] - sites[None, :, :]) ** 2, -1)
        dd = jnp.sqrt(dd + 1e-12) + 1e9 * jnp.eye(sites.shape[0])
        out = out + float(nu_sep) * jnp.mean(
            jax.nn.relu(float(sep_target) - jnp.min(dd, axis=1)) ** 2)
    return out


def weak_org_loss(codes: jnp.ndarray, jig: jnp.ndarray, radius: float,
                  ind: jnp.ndarray, y: jnp.ndarray, *,
                  temperature: float = 0.2) -> jnp.ndarray:
    """⭐ **THE WAVE'S SINGLE ABLATION** — one extra coefficient, not a capability.

    Labels shape **well-sharing**: the pair target is the downstream target's own
    similarity ``exp(-||y_i - y_j||^2 / 2 tau^2)`` instead of a label-free
    augmentation pair. It tests the **A31.4 inversion at the ORGANIZER level** —
    is label information helpful, neutral, or **harmful** to address geometry
    when it enters the organizer rather than the encoder? ⛔ The prediction is
    registered in this spoke's ``PREREG.md`` (B16) *before* it was run.
    """
    sites = placed_sites(codes, jig, radius)
    c = placement_centroids(sites, ind)
    c = c / (jnp.linalg.norm(c, axis=-1, keepdims=True) + 1e-12)
    logits = (c @ c.T) / float(temperature)
    dy = jnp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    tgt = jax.nn.softmax(-dy / (2.0 * (jnp.mean(dy) + 1e-12)), axis=1)
    logp = jax.nn.log_softmax(logits - 1e9 * jnp.eye(c.shape[0]), axis=1)
    return jnp.mean(-jnp.sum(tgt * logp, axis=1))


# --------------------------------------------------------------------------
# (b) the sharing / refresh term
# --------------------------------------------------------------------------
def refresh_amplitudes(amp: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """⭐ The **STRUCTURAL** refresh: ``amp <- sqrt(amp^2 + delta)``, ``delta >= 0``.

    I1 (refresh-on-rewrite depth monotonicity) then holds as an **IDENTITY, not
    a penalty** — a loss that *rewards* monotonicity can be traded away by any
    other term; a parameterisation cannot (loss package §1(b)). ⛔ A refresh
    **deepens, never moves**: ``c_j`` is frozen during a refresh, because moving
    it is what breaks G-ADDR's designed targets.
    """
    return jnp.sqrt(amp ** 2 + jax.nn.relu(delta))


def share_loss(depth: jnp.ndarray, *, depth_target: float = 0.30,
               depth_prev: Optional[jnp.ndarray] = None, mu: float = 1.0,
               neighbour_min: Optional[jnp.ndarray] = None, nu: float = 0.0,
               log_ratio_max: float = 1.0) -> jnp.ndarray:
    """Term **(b)**: sets *how much* a refresh deepens; never the sign.

    ⚠⚠ The ``nu`` guard is **not optional** (loss package §1(b)(v)): refresh
    manufactures depth heterogeneity, and in the theorist's 1-D toy the shallow
    neighbour's **minimum ceases to exist** at depth ratio 1.60 / 2.22 / 3.27 for
    ``d/s = 2.5 / 2.7 / 2.9``. ``log_ratio_max`` must be set from the **measured**
    annihilation ratio on the learned store, never from O2's formula.
    """
    out = jnp.mean(jax.nn.softplus(float(depth_target) - depth))
    if depth_prev is not None:
        out = out + float(mu) * jnp.mean(jax.nn.relu(depth_prev - depth))
    if float(nu) != 0.0 and neighbour_min is not None:
        out = out + float(nu) * jnp.mean(jax.nn.relu(
            jnp.log(depth + 1e-12) - jnp.log(neighbour_min + 1e-12)
            - float(log_ratio_max)))
    return out


# --------------------------------------------------------------------------
# (c) the curvature-shape term — the w20 defender
# --------------------------------------------------------------------------
def soft_min(lams: jnp.ndarray, beta: float = 50.0) -> jnp.ndarray:
    """``-(1/beta) log sum_i exp(-beta lambda_i)`` — the smooth softest mode.

    The surrogate removes the eigenvalue-crossing non-differentiability;
    ``dlambda_i/dtheta = u_i^T (dH/dtheta) u_i`` (Hellmann-Feynman) is valid
    where ``lambda_i`` is simple.
    """
    return -(1.0 / float(beta)) * jax.nn.logsumexp(-float(beta) * lams, axis=-1)


def shape_loss(lams: jnp.ndarray, depth: jnp.ndarray, capture: jnp.ndarray, *,
               two_alpha: float = 0.10, eps_soft: float = 0.005,
               lambda_stiff_target: float = 1.0, depth_min: float = 0.05,
               sigma_q: float = 0.15, beta: float = 50.0) -> jnp.ndarray:
    """Term **(c)**: pull the softest mode to the floor, keep the second stiff —
    and ⛔ **only at a site that is actually DUG and still CAPTURES**.

    The last two hinges are what make the claim non-vacuous: without them
    ``L_shape`` is minimised by an **undug** site, which reports
    ``lambda_min = 2 alpha`` for free — precisely M8's banked trap (*undug wells
    report ``lambda_min ~ 0.0993`` because ``2 alpha`` is what ``lambda_min``
    reports when nothing was written*).

    ⛔ ``§A4.2 REFUTED the tilt instantiation`` on a learned store (tilt
    monotonically *reduces* ``lambda_min``, +0.099 -> -8.28). The refuted object
    is **not** re-derived here: this is a spectral-shape defender whose claim is
    M8-measurable and whose failure modes are visible.
    """
    lam1 = soft_min(lams, beta)
    lam2 = soft_min(lams[..., 1:], beta)
    return jnp.mean(
        (lam1 - (float(two_alpha) + float(eps_soft))) ** 2
        + jax.nn.relu(float(lambda_stiff_target) - lam2) ** 2
        + jax.nn.relu(float(depth_min) - depth) ** 2
        + jax.nn.relu(float(sigma_q) - capture) ** 2)


# --------------------------------------------------------------------------
# (d) the set-level compositional read loss
# --------------------------------------------------------------------------
def read_loss(pred: jnp.ndarray, y: jnp.ndarray,
              weight: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Term **(d)**: ``E[ l_task( psi({u_f}) , y ) ]``, likelihood weighted by
    captured-vs-scattered particles.

    ⛔ **DeepSets pooling ONLY** — ``AttentionPsi`` is quarantined for trajectory
    input (``AttentionPsiLeakError``); any attention-psi number is a declared
    NOT-RUN this wave.
    """
    per = jnp.sum((pred - y) ** 2, axis=-1)
    if weight is None:
        return jnp.mean(per)
    w = weight / (jnp.mean(weight) + 1e-12)
    return jnp.mean(w * per)


# --------------------------------------------------------------------------
# (e) the calibration term — feature dropout as pseudo-novelty
# --------------------------------------------------------------------------
def cal_loss(logits: jnp.ndarray, novel: jnp.ndarray) -> jnp.ndarray:
    """Term **(e)**: per-channel BCE against the write-side dropout mask.

    ⭐ **Log-loss is STRICTLY PROPER and is the objective. AUROC is the REPORTED
    statistic and is NEVER trained against** (loss package §1(e)) — say so in the
    arm's ledger, which :func:`chlu.core.novelty_read.novelty_ledger` does.

    ⭐ The mask is drawn **independently of the query** and acts on the **WRITE**,
    so ``n_f ⟂ query`` by construction ⇒ a query-only novelty head is *provably*
    at the base rate and any AUROC > 0.5 is store information (``N-e3``: a
    STRUCTURAL kill of the leak, of the same kind as K8).
    """
    return jnp.mean(jnp.logaddexp(0.0, logits) - novel * logits)


def brier(conf: jnp.ndarray, correct: jnp.ndarray) -> jnp.ndarray:
    """The set-answer confidence's Brier score (V2b's proper scoring rule)."""
    return jnp.mean((conf - correct) ** 2)


# --------------------------------------------------------------------------
# the package, assembled
# --------------------------------------------------------------------------
def organizer_total(shipped: jnp.ndarray, coeffs: C2W11LossCoeffs, *,
                    terms: Optional[Dict[str, jnp.ndarray]] = None
                    ) -> jnp.ndarray:
    """``shipped + sum_i lambda_i L_i``, with **structural** coefficient-zero.

    ⛔ Each term is added only when its coefficient is a **non-zero static
    float**, so at zero the term's graph is never built. This is what makes
    ``coeffs = C2W11LossCoeffs()`` bit-identical to the shipped objective rather
    than merely numerically equal (a ``0 * L`` would still trace ``L``, still
    consume its RNG, and could still inject ``0 * NaN``).
    """
    out = shipped
    for name, lam in (("org", coeffs.lambda_org), ("share", coeffs.lambda_share),
                      ("shape", coeffs.lambda_shape), ("read", coeffs.lambda_read),
                      ("cal", coeffs.lambda_cal), ("weak", coeffs.lambda_weak)):
        if float(lam) != 0.0:
            if terms is None or name not in terms:
                raise KeyError(f"coefficient lambda_{name} is live but the term "
                               f"'{name}' was not supplied")
            out = out + float(lam) * terms[name]
    return out
