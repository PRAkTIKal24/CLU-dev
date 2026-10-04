"""**Multiplicity-as-counting-code** — the tier-ii read's *cardinality* iteration.

⛔ **What this module adds, and to which measured failure.**
:mod:`chlu.core.multiwell_read` (iteration 1, charter §A20.3) repaired the read's
**addressing** — ``K0`` 0.0477 → 1.0000, distinct wells occupied 2.20 → 11.27,
``S_eff`` 34–51 (COLLAPSED) → 16.00 (in band) — and left the read's
**expressivity** unrepaired: exact-set 0.0023 (bar 0.02), ``OD_min`` a vacuous
tie, and guard 1 firing (``read − live launder = −0.0016 ± 0.0019``). The named,
measured blocker (``tierii-read-fix`` §4.1) is **CARDINALITY**:

* the read visits ~11.3 wells and has **no mechanism that commits to exactly
  ``F``**; the gated set carries 5.79 ± 0.90 members and never 4;
* the gate discards *correct* wells (``coverage_gated`` 0.053);
* ⭐ and **π-sharpening provably cannot supply the commitment** — ``top-F(π) ==
  A(x)`` measured **0.000 at every** ``β ∈ {1,2,4,8,16}``, because at
  ``depth_ratio = 3`` the settled occupancy's ranking is *depth*-driven, not
  *query*-driven. The information is **not in the settled occupancy**.

**The fix this module implements** (charter §A21's C2W7 row, deliverables 1–3):

1. **Multiplicity-as-counting-code.** The ``k`` particles no longer fan out over
   ``k`` distinct wells: the head allocates them with a **learned multiplicity**
   (well ``j`` receives ``n_j`` particles, ``Σ_j n_j = k`` exactly), and the
   ``F``-commitment is **query-driven** — a **learned stopping rule** on a
   residual matching pursuit decides *how many* wells the query names, and the
   head (never well depth) decides how many particles each gets.
2. **Overlap-as-importance weighting.** The read's answer is a **weighted
   counting code** ``m ∈ R^{N_a}`` (per-well contributions weighted by each
   particle's descent and overlap), replacing binary occupancy + ``noisy_or``.
   ⛔ This changes what a reader consumes and therefore required the reader-class
   re-registration filed as ``AMENDMENT-C2W7`` beside ``ERRATA-TierII.md``.
3. **The batch-level anti-collapse regularizer.** ⛔ It penalises the **MARGINAL**
   (across-inputs) well-usage distribution collapsing onto few wells, and
   **never** per-query concentration — a confident query putting all ``k``
   particles into ``F`` wells *is the design working*. Doctrine §3.3 activation
   order: **monitored first, regularized second** — it ships built and **OFF**
   (``lambda_anticollapse = 0.0``) and turns on only when the launch-collapse
   monitor (:class:`chlu.core.monitors.LaunchCollapseMonitor`) fires.

⛔ **No ``argmax`` anywhere in the latent a reader sees.** The stick-breaking
allocation is a *partition of unity* (``Σ_j beta_ij = 1`` exactly, by
construction) and every launch is a continuous mixture of anchors plus a
continuous residual, exactly as in iteration 1.

⛔ **Claim-form discipline (``PREREG-TierII.md`` §2.6):** no well is named
semantically here, in the tests, or in any artifact. Wells carry integer indices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.factored_store import (CatTestConfig, FactoredStore, FrozenPhi,
                                      exact_set_accuracy, occupancy)
from chlu.core.multiwell_read import (aggregate_occupancy, descent_weight,
                                      query_code, settle_particles,
                                      soft_occupancy)

__all__ = [
    "MultiplicityConfig",
    "MultiplicityHead",
    "mult_head_trainable_spec",
    "counting_code",
    "importance_code",
    "multiplicity_read",
    "multiplicity_launder",
    "marginal_usage",
    "launch_collapse_stat",
    "anticollapse_penalty",
    "count_table_fit",
    "count_table_apply",
    "count_identity_fit",
    "count_identity_apply",
    "fit_readers_mc",
    "apply_reader_mc",
    "score_readers_mc",
    "READERS_MC",
    "READER_CLASS_C2W7",
    "WEIGHT_MODES",
    "read_codes",
    "particle_weights",
    "count_stats",
    "mult_read_loss",
    "train_multiplicity_head",
    "organize_store_mc",
    "mc_hard_vs_soft_gradient",
    "mc_staging_gradient_probe",
    "mc_ledger",
]


# ==========================================================================
# config — lives next to its code (the CatTestConfig / MultiWellReadConfig
# precedent; ⛔ NOT in chlu/config.py, which is the dynamics-experiment schema)
# ==========================================================================
@dataclass
class MultiplicityConfig:
    """Every knob of the multiplicity read, at its **registered** value.

    ⛔ ``k_particles`` is CAPACITY: it is on the byte ledger of every arm and must
    be identical across arms (guard 4, charter §A20.3(c)).
    """

    # -- the k-particle multiplicity head ------------------------------------
    k_particles: int = 12  # ⭐ k (registered; k in {16, 24} declared NOT-RUN)
    f_max: int = 8  # the largest cardinality the head may commit to
    ista_steps: int = 200  # unrolled non-negative ISTA iterations (lax.scan)
    ista_lam: float = 0.05  # sparsity penalty (learned)
    ista_eta: float = 0.30  # step size (learned)
    card_a: float = 0.85  # ⭐ F_hat = clip(a * IPR(x) + b, 1, f_max)
    # ⚠ designed init, tuned on the SEEN split ONLY: `a = 4/mean(IPR)` = 0.85 and
    # `b` is the K0 SLACK. K0's bar (P(>=F distinct) >= 0.90) and the
    # F-commitment pull in opposite directions BY CONSTRUCTION — committing to
    # exactly F wells makes `>= F distinct` fail whenever F_hat < F. Measured on
    # SEEN (3 seeds): b=0.0 -> K0 0.789 / exact 0.143; b=0.3 -> 0.880 / 0.135;
    # b=0.5 -> K0 0.912 / exact 0.122 (⚠ 0.871 on UNSEEN — the SEEN target must
    # carry margin); ⭐ b=0.7 -> K0 0.953 / exact 0.099 (REGISTERED); b=0.9 ->
    # 0.964 / 0.091; b=1.1 -> 0.979 / 0.063; a=1.0,b=0.3 -> 0.956 / 0.078.
    card_b: float = 0.7
    commit_eps: float = 0.005  # commitment sharpness (soft top-F_hat)
    rank_sigma: float = 0.35  # rank-interpolation width for the threshold
    rho: float = 0.25  # continuous-residual injection into the launch point
    # confidence -> per-particle attributes (iteration 1's mechanism, carried).
    # ⚠ WIDER than iteration 1's (0.25 / 0.05): at a multiplicity head a hard
    # confidence gate would delete a whole well from the counting code and the
    # all-or-nothing metric would score the query 0 even with the right set.
    conf_b: float = 0.10
    conf_w: float = 0.15
    log_mass_conf: float = 0.0
    log_mass_unconf: float = 1.5
    gamma_mult_conf: float = 1.0
    gamma_mult_unconf: float = 4.0
    p0_gain: float = 1.0
    learned_p0: bool = True  # ⭐ (e) reach lever only (§A14.1); ablated once

    # -- the read latent ------------------------------------------------------
    occ_tau: float = 0.25  # soft-occupancy temperature (address units)
    payload_ref: float = 0.5  # ||v_j||; sets the descent weight's scale
    descent_gate: bool = True
    weight_mode: str = "both"  # {"both", "descent", "overlap", "none"}
    count_agg: str = "sum"  # ⭐ the dedupe verb, LIVE at multiplicity
    dedupe_pi: str = "noisy_or"  # the carried set-union latent (iteration 1)

    # -- (3) the batch-level anti-collapse regularizer (built OFF) -------------
    lambda_anticollapse: float = 0.0  # ⛔ OFF until the monitor fires
    collapse_band_lo: float = 0.5  # band low = frac * N_a (monitor + penalty)

    # -- training (staged: store first, head second — guard 3) ----------------
    head_steps: int = 60
    # ⚠ tuned on SEEN, seed 0, at the TRAINING settle budget. 3e-2 (iteration 1's
    # value) DESTROYS the mechanism: F_hat saturates at f_max, rho flips sign,
    # occupancy precision falls to chance (0.124 vs 4/32 = 0.125) and the
    # launch-collapse monitor TRIPS (S_marg 6.9 < 16). Measured SEEN scores after
    # 60 steps: lr 1e-3 -> 0.055 (S_marg 28.0) · ⭐ 3e-3 -> 0.063 (21.3) ·
    # 1e-2 -> 0.023 (13.7) · 3e-2 -> 0.000 (6.9); designed (untrained) 0.039.
    head_lr: float = 3e-3
    head_batch: int = 16
    head_settle_address: int = 60  # ⚠ reduced TRAINING settle budget, declared
    head_settle_read: int = 120
    w_occ: float = 1.0  # the counting-code channel
    w_pay: float = 1.0  # the continuous payload channel

    seed: int = 0

    def as_flag_table(self) -> Dict[str, Any]:
        base = MultiplicityConfig()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================================
# (1) the multiplicity head: query-driven cardinality + learned multiplicity
# ==========================================================================
class MultiplicityHead(eqx.Module):
    """``c_tilde -> k`` particles with a **learned counting code**.

    Four stages, all differentiable and all query-driven:

    **(i) the evidence — an unrolled non-negative ISTA over the frozen code
    dictionary.** ``x_0 = 0``; ``x <- relu(x + eta E^T (y - E^T x) - eta lam)``
    with ``y = c_tilde / R``. ⛔ **Greedy pursuit provably cannot do this job here**
    and it was measured: the code's mutual coherence is ``~1/sqrt(d) = 0.35`` while
    OMP's recovery guarantee needs ``< 1/(2F-1) = 0.14``. Measured exact-set
    recovery on the SEEN split at the registered cell: top-``F`` of the raw
    overlaps **0.023**, OMP **0.008**, ⭐ non-negative ISTA **0.19**, exhaustive
    (out-of-class, ``C(32,4)`` decoder) 0.72. Non-negativity is not a trick: every
    true coefficient is ``+1/sqrt(F)`` by the construction of ``phi``.

    **(ii) the cardinality estimate — the inverse participation ratio.**
    ``IPR(x) = (sum_j x_j)^2 / sum_j x_j^2`` is **exactly** ``F`` for a flat
    ``F``-sparse vector, and ``F_hat_0 = clip(a IPR + b, 1, f_max)`` with ``a, b``
    **learned** (designed init ``a = 4 / mean IPR`` measured on SEEN). ⭐ It is a
    function of the *query's own* coefficient profile — never of well depth, which
    is precisely the failure §4.1 measured (``top-F(pi) == A(x)`` was 0.000 at
    every sharpening ``beta`` because the settled occupancy ranks by depth).

    **(iii) the COMMITMENT — a soft top-``F_hat``.** With ``s`` the descending sort
    of ``x``, the per-query threshold is the rank-interpolated midpoint
    ``theta = sum_t w_t (s_t + s_{t+1})/2``, ``w = softmax(-(t+1-F_hat)^2/2 rank_sigma^2)``,
    and ``mask_j = sigmoid((x_j - theta)/commit_eps)``, ``F_committed = sum_j mask_j``.
    ⭐ **This is the F-commitment iteration 1 did not have**, and it is what makes
    the answer *exact* when the set is right: the multiplicity is
    ``n_j = k mask_j / sum_l mask_l`` (so ``sum_j n_j = k`` exactly and each named
    well gets ``k / F_committed`` particles), and the read's importance code lands
    at ``m_j = 1`` on the named wells with **no free parameter to get wrong**.

    **(iv) stick-breaking allocation.** With ``c_j = Σ_{l<j} n_l``, particle ``i``
    owns the unit slice ``[i, i+1)`` of a stick of length ``k`` and

        ``beta_ij = relu(min(i+1, c_j + n_j) - max(i, c_j))``

    ⭐ which is a **partition of unity by construction** (``Σ_j beta_ij = 1``, no
    normalisation, no ``argmax``, no sampling): the slices tile the stick exactly.
    Gradients reach *every* well through the cumulative offsets ``c_j``.

    The launch is then the same continuous mixture iteration 1 registered,
    ``q0_i = Σ_j beta_ij u_j + rho R (c_hat − Σ_j beta_ij e_j) + slot_offset_i``,
    and ``conf_i = sigmoid((Σ_j beta_ij <c_hat, e_j> − b)/w)`` drives the
    per-particle mass / friction / ``p0`` (overlap-as-confidence, carried).
    """

    codes: jnp.ndarray  # (N_a, d) FROZEN query codes
    anchors: jnp.ndarray  # (N_a, d) FROZEN well anchors
    slot_offset: jnp.ndarray  # (k, d) learned
    bias: jnp.ndarray  # (N_a,) learned per-well evidence bias
    log_lam: jnp.ndarray  # ISTA sparsity penalty
    log_eta: jnp.ndarray  # ISTA step size
    card_a: jnp.ndarray  # ⭐ the cardinality commitment's calibration
    card_b: jnp.ndarray
    log_commit_eps: jnp.ndarray
    log_rank_sigma: jnp.ndarray
    rho: jnp.ndarray
    conf_b: jnp.ndarray
    conf_w: jnp.ndarray
    log_mass_c: jnp.ndarray
    log_mass_u: jnp.ndarray
    gam_mult_c: jnp.ndarray
    gam_mult_u: jnp.ndarray
    p0_gain: jnp.ndarray
    k: int = eqx.field(static=True)
    ista_steps: int = eqx.field(static=True)
    f_max: int = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    use_p0: bool = eqx.field(static=True)

    def __init__(self, phi: FrozenPhi, anchors, cfg: CatTestConfig,
                 mc: MultiplicityConfig):
        self.codes = jnp.asarray(phi.codes)
        self.anchors = jnp.asarray(np.asarray(anchors)[:, : cfg.addr_dim],
                                   dtype=jnp.float32)
        self.k = int(mc.k_particles)
        self.ista_steps = int(mc.ista_steps)
        self.f_max = int(mc.f_max)
        self.addr_dim = int(cfg.addr_dim)
        self.payload_dim = int(cfg.payload_dim)
        self.radius = float(cfg.ball_radius)
        self.use_p0 = bool(mc.learned_p0)
        z = jnp.zeros(())
        self.slot_offset = jnp.zeros((self.k, self.addr_dim))
        self.bias = jnp.zeros((int(cfg.n_wells),))
        self.log_lam = z + float(np.log(mc.ista_lam))
        self.log_eta = z + float(np.log(mc.ista_eta))
        self.card_a = z + float(mc.card_a)
        self.card_b = z + float(mc.card_b)
        self.log_commit_eps = z + float(np.log(mc.commit_eps))
        self.log_rank_sigma = z + float(np.log(mc.rank_sigma))
        self.rho = z + float(mc.rho)
        self.conf_b = z + float(mc.conf_b)
        self.conf_w = z + float(mc.conf_w)
        self.log_mass_c = z + float(mc.log_mass_conf)
        self.log_mass_u = z + float(mc.log_mass_unconf)
        self.gam_mult_c = z + float(mc.gamma_mult_conf)
        self.gam_mult_u = z + float(mc.gamma_mult_unconf)
        self.p0_gain = z + float(mc.p0_gain)

    def n_params(self) -> int:
        """Head parameters — ledgered on EVERY arm (the launch protocol is shared)."""
        return int(self.slot_offset.size + self.bias.size + 14)

    # -- (i) the evidence: unrolled non-negative ISTA -------------------------
    def evidence(self, c_noisy: jnp.ndarray) -> jnp.ndarray:
        """``(d,) -> (N_a,)`` non-negative sparse coefficients over the codes."""
        y = c_noisy / self.radius  # true coefficients are +1/sqrt(F)
        lam, eta = jnp.exp(self.log_lam), jnp.exp(self.log_eta)

        def body(x, _):
            g = (y - x @ self.codes) @ self.codes.T
            # ⭐ `bias` is a learned PER-WELL soft-threshold offset: the head may
            # learn that some wells need more evidence than others. Init 0.
            return jax.nn.relu(x + eta * g - eta * (lam + self.bias)), None

        x, _ = jax.lax.scan(body, jnp.zeros((self.codes.shape[0],)), None,
                            length=self.ista_steps)
        return x

    # -- (ii) + (iii): the cardinality commitment ----------------------------
    def commit(self, x: jnp.ndarray):
        """``(N_a,) -> (mask (N_a,), n (N_a,), F_hat (), F_prior ())``."""
        ipr = (jnp.sum(x) ** 2) / (jnp.sum(x ** 2) + 1e-12)
        f_prior = jnp.clip(self.card_a * ipr + self.card_b, 1.0, float(self.f_max))
        s = -jnp.sort(-x)  # descending
        mid = 0.5 * (s[:-1] + s[1:])  # threshold between rank t+1 and t+2
        t = jnp.arange(mid.shape[0], dtype=x.dtype) + 1.0
        sig = jnp.exp(self.log_rank_sigma)
        w = jax.nn.softmax(-((t - f_prior) ** 2) / (2.0 * sig ** 2 + 1e-9))
        theta = jnp.sum(w * mid)
        mask = jax.nn.sigmoid((x - theta) / (jnp.exp(self.log_commit_eps) + 1e-12))
        f_hat = jnp.sum(mask)
        n = float(self.k) * mask / (f_hat + 1e-9)
        return mask, n, f_hat, f_prior

    # -- (iii) allocation + the launch ---------------------------------------
    def allocate(self, n: jnp.ndarray) -> jnp.ndarray:
        """``(N_a,) -> (k, N_a)`` stick-breaking; ⭐ a partition of unity."""
        c = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(n)[:-1]])  # (N_a,)
        lo = jnp.arange(self.k, dtype=n.dtype)[:, None]  # (k, 1)
        hi = lo + 1.0
        return jax.nn.relu(jnp.minimum(hi, c + n) - jnp.maximum(lo, c))

    def __call__(self, c_noisy: jnp.ndarray):
        """``(d,) -> (q0, p0, mass, gamma_mult, conf, beta, n, F_hat)``."""
        d, m, R = self.addr_dim, self.payload_dim, self.radius
        ch = c_noisy / (jnp.linalg.norm(c_noisy) + 1e-12)
        x = self.evidence(c_noisy)
        _mask, n, f_hat, _f_prior = self.commit(x)
        beta = self.allocate(n)  # (k, N_a)

        mix_u = beta @ self.anchors  # (k, d)
        mix_e = beta @ self.codes  # (k, d)
        q_addr = mix_u + self.rho * R * (ch[None, :] - mix_e) + self.slot_offset
        overlap = beta @ (self.codes @ ch)  # (k,) the assigned well's overlap
        conf = jax.nn.sigmoid((overlap - self.conf_b) / (self.conf_w + 1e-6))
        q0 = jnp.concatenate([q_addr, jnp.zeros((self.k, m))], axis=1)

        # (e) learned p0 — the confidence-gated ballistic kick BACK INTO the
        # addressed well; a REACH lever only (§A14.1), never a selectivity claim.
        if self.use_p0:
            p_addr = self.p0_gain * conf[:, None] * (mix_u - q_addr)
        else:
            p_addr = jnp.zeros_like(q_addr)
        p0 = jnp.concatenate([p_addr, jnp.zeros((self.k, m))], axis=1)

        u = 1.0 - conf
        logM = self.log_mass_c + (self.log_mass_u - self.log_mass_c) * u
        mass = jnp.exp(logM)[:, None] * jnp.ones((1, d + m))
        gmult = self.gam_mult_c + (self.gam_mult_u - self.gam_mult_c) * u
        return q0, p0, mass, gmult, conf, beta, n, f_hat


def mult_head_trainable_spec(head: MultiplicityHead):
    """Filter spec selecting ONLY the head's learned leaves.

    ⛔ ``codes``/``anchors`` are the frozen launch geometry shared byte-identically
    by every arm (``PREREG-TierII.md`` §1); they are inexact arrays, so
    ``eqx.is_inexact_array`` alone would hand them to the optimiser.
    """
    spec = jax.tree_util.tree_map(lambda _: False, head)
    for name in ("slot_offset", "bias", "log_lam", "log_eta", "card_a", "card_b",
                 "log_commit_eps", "log_rank_sigma", "rho", "conf_b", "conf_w",
                 "log_mass_c", "log_mass_u", "gam_mult_c", "gam_mult_u",
                 "p0_gain"):
        spec = eqx.tree_at(lambda t, nm=name: getattr(t, nm), spec,
                           replace=jax.tree_util.tree_map(
                               lambda _: True, getattr(head, name)))
    return spec


# ==========================================================================
# (2) the weighted counting code — overlap-as-importance
# ==========================================================================
def particle_weights(w_descent, conf, mode: str = "both"):
    """``a_i`` — the per-particle importance the counting code sums with.

    ⭐ **Overlap-as-importance** (deliverable 2): a particle contributes to its
    well in proportion to *how confident the head was about it* (``conf``, the
    overlap of its assigned well with the query code) **and** *how far it actually
    descended* (``w``, the store's own testimony). ``none`` recovers a plain count.
    """
    if mode == "both":
        return w_descent * conf
    if mode == "descent":
        return w_descent
    if mode == "overlap":
        return conf
    if mode == "none":
        return jnp.ones_like(w_descent)
    raise ValueError(f"unknown weight_mode {mode!r}")


def counting_code(pi, a, mode: str = "sum"):
    """``(B, k, N_w) x (B, k) -> (B, N_w)`` — the **counting** aggregation.

    ⭐ ``sum`` is the *correct* verb once multiplicity exists and is exactly the
    verb iteration 1 measured **INERT** (§6: ``dedupe="sum"`` was bit-identical to
    ``noisy_or`` because successive suppression never let two particles share a
    well). At a multiplicity head the two disagree, and that disagreement is the
    measurement that the dedupe/evolve-unique verb has gone live.
    """
    if mode == "sum":
        return jnp.sum(pi * a[..., None], axis=1)
    return aggregate_occupancy(pi, a, mode)


def importance_code(cnt, f_hat):
    """``m_j = F_hat * cnt_j / Σ_l cnt_l`` — the read's ANSWER.

    ⭐ The normalisation is what turns a *count* into a *membership*: with ``n_j =
    k/F_hat`` particles in each named well, ``m_j -> 1`` on the named wells and
    ``0`` elsewhere, so ``m @ V_table -> Σ_{j in A} v_j`` — the family's target —
    **without any hard set decision anywhere**.
    """
    return jnp.asarray(f_hat)[..., None] * cnt / (jnp.sum(cnt, axis=-1,
                                                          keepdims=True) + 1e-9)


def _model(store: FactoredStore, cfg: CatTestConfig):
    return store.model(cfg)


#: the registered weighting-ablation axis — all four are computed from ONE read
#: pass (they differ only in the per-particle weight ``a_i``), so the ablation
#: costs zero extra settles and every arm carries all four.
WEIGHT_MODES = ("none", "descent", "overlap", "both")


def read_codes(s, w, conf, f_hat, mc: MultiplicityConfig):
    """Every registered weighting/aggregation variant, from one settled read."""
    out = {}
    for mode in WEIGHT_MODES:
        a = particle_weights(w, conf, mode)
        out[f"m__{mode}"] = importance_code(counting_code(s, a, mc.count_agg),
                                            f_hat)
    a0 = particle_weights(w, conf, mc.weight_mode)
    out["m"] = out[f"m__{mc.weight_mode}"]
    out["cnt"] = counting_code(s, a0, mc.count_agg)
    # ⛔ the refuted set-union aggregation, kept as the designed negative: at a
    # multiplicity head `noisy_or` destroys the count (iteration 1 measured the
    # two verbs BIT-IDENTICAL because no two particles ever shared a well).
    out["m__noisy_or"] = importance_code(
        aggregate_occupancy(s, a0, "noisy_or"), f_hat)
    return out


def multiplicity_read(store: FactoredStore, head: MultiplicityHead,
                      phi: FrozenPhi, cfg: CatTestConfig, mc: MultiplicityConfig,
                      indicators, key, *, wells=None, batch: int = 128
                      ) -> Dict[str, np.ndarray]:
    """The full multiplicity read. Returns the latent dict every reader consumes.

    ``z``     (B, k, dim)  continuous settled states
    ``m``     (B, N_w)     ⭐ the **weighted counting code** — the read's answer
    ``cnt``   (B, N_w)     the raw counts (pre-normalisation)
    ``pi``    (B, N_w)     the carried ``noisy_or`` set latent (iteration 1's)
    ``n``     (B, N_w)     the head's launch multiplicity (``Σ_j n_j = k``)
    ``F_hat`` (B,)         the head's per-query cardinality commitment
    ``conf``/``w`` (B, k)  overlap-confidence / descent weight
    ``q0``    (B, k, dim)  the launch states (for the live launder, G1)
    """
    wells = head.anchors if wells is None else jnp.asarray(wells, jnp.float32)
    model = _model(store, cfg)
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B, d = int(ind.shape[0]), int(cfg.addr_dim)
    c_all = query_code(phi, ind, key, float(cfg.query_sigma))

    @eqx.filter_jit
    def _one(c):
        q0, p0, mass, gmult, conf, _beta, n, f_hat = jax.vmap(head)(c)

        def run(q, p, mm, gm):
            q, p = settle_particles(model, q, p, mm,
                                    gm * float(cfg.gamma_address),
                                    int(cfg.address_steps), float(cfg.dt))
            q, p = settle_particles(model, q, jnp.zeros_like(p), mm,
                                    gm * float(cfg.gamma_read),
                                    int(cfg.read_steps), float(cfg.dt))
            return q

        z = jax.vmap(run)(q0, p0, mass, gmult)
        w = (descent_weight(z[..., d:], mc.payload_ref) if mc.descent_gate
             else jnp.ones(z.shape[:2]))
        s = soft_occupancy(z[..., :d], wells, mc.occ_tau)
        out = read_codes(s, w, conf, f_hat, mc)
        out.update({"z": z, "pi": aggregate_occupancy(s, w, mc.dedupe_pi),
                    "conf": conf, "w": w, "n": n, "F_hat": f_hat, "q0": q0})
        return out

    outs: Dict[str, list] = {}
    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        for name, val in _one(c_all[lo:hi]).items():
            outs.setdefault(name, []).append(np.asarray(val))
    return {kk: np.concatenate(v, axis=0) for kk, v in outs.items()}


def multiplicity_launder(head: MultiplicityHead, phi: FrozenPhi,
                         cfg: CatTestConfig, mc: MultiplicityConfig,
                         indicators, key, *, wells=None, batch: int = 512
                         ) -> Dict[str, np.ndarray]:
    """⛔ **Guard 1, RECOMPUTED LIVE on THIS cell's own learned launches** (A1 strong).

    The landscape is deleted (zero settle: the particles never move) and the
    written payload table is retained, so the launder is exactly *"this trained
    head plus a nearest-well table"* — the trivial substitute the store must beat.
    It is scored through the **same re-registered reader class at the same ``k``**
    and it gets the **whole cardinality mechanism** (multiplicity, ``F_hat``,
    importance weighting): the only thing it does not get is the physics.

    ⛔ ``orgdiv-null-arms``' 0.272 and the ``d=8`` 0.695 are **out-of-class
    reference ceilings at their own ``(d, draws)`` noise model** and are never a
    bar; this function computes the bar.
    """
    wells = head.anchors if wells is None else jnp.asarray(wells, jnp.float32)
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B, d = int(ind.shape[0]), int(cfg.addr_dim)
    c_all = query_code(phi, ind, key, float(cfg.query_sigma))

    @eqx.filter_jit
    def _one(c):
        q0, _p0, _m, _g, conf, _beta, n, f_hat = jax.vmap(head)(c)
        w = jnp.ones(q0.shape[:2])  # no descent happened: nothing to gate on
        s = soft_occupancy(q0[..., :d], wells, mc.occ_tau)
        out = read_codes(s, w, conf, f_hat, mc)
        out.update({"z": q0, "pi": aggregate_occupancy(s, w, mc.dedupe_pi),
                    "conf": conf, "w": w, "n": n, "F_hat": f_hat})
        return out

    outs: Dict[str, list] = {}
    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        for name, val in _one(c_all[lo:hi]).items():
            outs.setdefault(name, []).append(np.asarray(val))
    out = {kk: np.concatenate(v, axis=0) for kk, v in outs.items()}
    out["q0"] = out["z"]
    return out


# ==========================================================================
# (3) + (4) the batch-level anti-collapse regularizer and its monitor statistic
# ==========================================================================
def marginal_usage(m_code) -> np.ndarray:
    """``(B, N_w) -> (N_w,)`` the **across-inputs MARGINAL** well-usage law.

    ⛔ This is the only object the anti-collapse machinery is ever allowed to look
    at. **Per-query concentration is CONFIDENCE and is never penalised**: a query
    that puts all ``k`` particles into its ``F`` wells is the design working. What
    collapses is the *marginal* — a learned launch head that sends every input to
    the same handful of wells (the codebook-collapse mode of VQ-VAE).
    """
    mm = jnp.abs(jnp.asarray(m_code))
    p = jnp.mean(mm, axis=0)
    return p / (jnp.sum(p) + 1e-12)


def launch_collapse_stat(m_code, n_wells: Optional[int] = None) -> Dict[str, float]:
    """The launch-collapse row's statistic: **marginal usage perplexity**.

    ``S_marg = exp(H(p_bar))`` — the effective number of wells the head uses
    *across the batch*. ``N_a`` = perfectly uniform marginal, ``1`` = every query
    lands in one well (total codebook collapse). The registered band is
    ``[collapse_band_lo * N_a, N_a]``; below it the monitor trips.
    ⭐ Its designed negative is a head whose allocation is input-independent
    (:func:`chlu.core.monitors.LaunchCollapseMonitor` documents it; the test suite
    asserts it fires).
    """
    p = np.asarray(marginal_usage(m_code))
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum())
    n_w = int(len(p) if n_wells is None else n_wells)
    return {"marginal_perplexity": float(np.exp(h)),
            "marginal_entropy": h,
            "marginal_max": float(p.max()),
            "n_wells": n_w,
            "S_eff_marginal": float(n_w) / max(float(np.exp(h)), 1e-9)}


def anticollapse_penalty(m_code, n_wells: int) -> jnp.ndarray:
    """``log N_a − H(p_bar) >= 0`` — the MARGINAL-collapse penalty (built OFF).

    ⛔ Doctrine §3.3 activation order (monitored first, regularized second): the
    coefficient ``MultiplicityConfig.lambda_anticollapse`` defaults to **0.0** and
    is turned on only after the launch-collapse monitor fires. Both states are
    reported.
    """
    p = marginal_usage(m_code)
    h = -jnp.sum(p * jnp.log(p + 1e-12))
    return jnp.log(float(n_wells)) - h


# ==========================================================================
# the RE-REGISTERED reader class (AMENDMENT-C2W7; every member < N_a m = 256)
# ==========================================================================
def count_table_fit(latent: Dict[str, np.ndarray], y, *, well_payloads
                    ) -> Dict[str, Any]:
    """⭐ **The new class member**: ``yhat = W [m @ V_table, 1]`` — ``(m+1) x m``.

    72 fitted parameters at ``m = 8`` — identical to ``well_table`` and
    ``soft_well_table``, and **< N_a m = 256** (SP-1's storeless bound). It
    consumes the **weighted counting code**, which is what deliverable 2 changed
    about the latent, and it quantises nothing.
    """
    f = np.asarray(latent["m"]) @ np.asarray(well_payloads)
    X = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    w, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return {"kind": "count_table", "w": w, "well_payloads": well_payloads,
            "n_params": int(w.size)}


def count_table_apply(mdl, latent, key: str = "m"):
    f = np.asarray(latent[key]) @ np.asarray(mdl["well_payloads"])
    X = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    return X @ mdl["w"]


def count_identity_fit(latent: Dict[str, np.ndarray], y, *, well_payloads
                       ) -> Dict[str, Any]:
    """⭐ **The ZERO-parameter member**: ``yhat = m @ V_table``. Nothing is fitted.

    ⛔ **Why this member exists, and it is a measurement, not a convenience.** Every
    other member of the class is fitted by **least squares** while the metric is a
    **thresholded** exact-set accuracy. Measured at this cell on the SEEN split:
    the counting code lands its set exactly on ~18 % of queries and, on those, the
    residual is ``0.006`` against ``tol = 0.234`` — but the least-squares fit is
    dominated by the other ~82 %, shrinks its gain to ``diag(W) ~ 0.4``, and drives
    the residual on the GOOD queries to ``0.537 > tol``. ⭐ **The fitted 72-param
    reader scores 0.000 on a latent that the identity decodes at 0.172–0.227**
    (3 seeds), and the 2-parameter gain+bias reader is shrunk just as hard
    (``a ~ 0.5``) and also scores 0.000. The pathology is the *fitting criterion*,
    not the capacity cap — which re-scopes iteration 1 §13.3's question.

    It adds **zero** fitted parameters (the ``knn`` precedent, 0 params) and is
    applied identically to the physics arm, the null and the launder.
    """
    del y
    return {"kind": "count_identity", "well_payloads": well_payloads,
            "n_params": 0}


def count_identity_apply(mdl, latent, key: str = "m"):
    return np.asarray(latent[key]) @ np.asarray(mdl["well_payloads"])


#: ⛔ FROZEN before the first arm ran (``PREREG.md`` §6 / ``AMENDMENT-C2W7``).
READERS_MC = ("sum_linear", "well_table", "knn", "mlp", "soft_well_table",
              "count_table", "count_identity")

#: what each member consumes and how many parameters it fits at ``d=8, m=8``.
#: ⛔ every entry is ``< N_a m = 256`` (SP-1's storeless bound).
READER_CLASS_C2W7 = {
    "sum_linear": ("z (continuous settled states)", 136, "non-quantising"),
    "well_table": ("argmax(z) (hard nearest well)", 72, "QUANTISING (kept)"),
    "knn": ("canonicalised z", 0, "non-parametric"),
    "mlp": ("z", 108, "non-quantising"),
    "soft_well_table": ("pi (noisy_or soft occupancy)", 72, "non-quantising twin"),
    "count_table": ("m (weighted counting code)", 72, "non-quantising, NEW"),
    "count_identity": ("m (weighted counting code)", 0, "non-quantising, NEW"),
}


def fit_readers_mc(latent_seen: Dict[str, np.ndarray], y_seen, *, anchors,
                   well_payloads, seed: int = 0,
                   which: Sequence[str] = READERS_MC) -> Dict[str, Any]:
    """Fit the re-registered reader class on the **SEEN split only**."""
    from chlu.core.multiwell_read import fit_readers_mw

    base = [w for w in which if not w.startswith("count_")]
    out = fit_readers_mw(latent_seen, y_seen, anchors=anchors,
                         well_payloads=well_payloads, seed=seed, which=base)
    if "count_table" in which:
        out["count_table"] = count_table_fit(latent_seen, y_seen,
                                             well_payloads=well_payloads)
    if "count_identity" in which:
        out["count_identity"] = count_identity_fit(latent_seen, y_seen,
                                                   well_payloads=well_payloads)
    return out


def apply_reader_mc(mdl, latent: Dict[str, np.ndarray], key: str = "m"
                    ) -> np.ndarray:
    from chlu.core.multiwell_read import apply_reader_mw

    if mdl["kind"] == "count_table":
        return count_table_apply(mdl, latent, key)
    if mdl["kind"] == "count_identity":
        return count_identity_apply(mdl, latent, key)
    return apply_reader_mw(mdl, latent)


def score_readers_mc(readers, latent, y, tol, key: str = "m") -> Dict[str, float]:
    return {k: exact_set_accuracy(apply_reader_mc(v, latent, key), y, float(tol))
            for k, v in readers.items()}


# ==========================================================================
# diagnostics — R1 / R2 / cardinality (⛔ hard occupancy = DIAGNOSTIC ONLY)
# ==========================================================================
def count_stats(latent: Dict[str, np.ndarray], anchors, subsets, F: int, *,
                gate: float = 0.5) -> Dict[str, float]:
    """The cardinality statistics — R1's own statistic on the counting code.

    ``*_raw``     every launched/settled particle counts (hard nearest well;
                  ⛔ diagnostic only, never a reader input).
    ``*_gated``   the read's asserted set ``{j : m_j >= gate}`` — ⭐ **R1's
                  statistic**, and the object the ``F``-commitment acts on.
    ``F_*``       the head's cardinality commitment and its error against ``F``.
    """
    z = np.asarray(latent["z"])
    occ = occupancy(z, np.asarray(anchors)[:, : z.shape[-1]])
    distinct = np.array([len(np.unique(o)) for o in occ], float)
    prec = np.array([np.isin(o, np.asarray(A)).mean()
                     for o, A in zip(occ, subsets, strict=True)], float)
    cover = np.array([np.isin(np.asarray(A), o).all()
                      for o, A in zip(occ, subsets, strict=True)], float)
    exact = np.array([set(np.unique(o).tolist()) == set(np.asarray(A).tolist())
                      for o, A in zip(occ, subsets, strict=True)], float)
    out = {"distinct_wells_raw": float(distinct.mean()),
           "ge_F_distinct_raw": float((distinct >= F).mean()),
           "occupancy_precision_raw": float(prec.mean()),
           "coverage_raw": float(cover.mean()),
           "exact_set_occupancy_raw": float(exact.mean()),
           "n_wells_ever_occupied": int(len(np.unique(occ)))}
    m = np.asarray(latent["m"])
    sets = [np.flatnonzero(row >= float(gate)) for row in m]
    sz = np.array([len(s) for s in sets], float)
    out.update({
        "distinct_wells_gated": float(sz.mean()),
        "gated_set_size_sd": float(sz.std()),
        "gated_set_is_F": float((sz == F).mean()),
        "ge_F_distinct_gated": float((sz >= F).mean()),
        "occupancy_precision_gated": float(np.mean(
            [np.isin(s, np.asarray(A)).mean() if len(s) else 0.0
             for s, A in zip(sets, subsets, strict=True)])),
        "coverage_gated": float(np.mean(
            [np.isin(np.asarray(A), s).all()
             for s, A in zip(sets, subsets, strict=True)])),
        "exact_set_occupancy_gated": float(np.mean(
            [set(s.tolist()) == set(np.asarray(A).tolist())
             for s, A in zip(sets, subsets, strict=True)])),
        "n_wells_ever_occupied_gated": int(len(np.unique(np.concatenate(sets)))
                                           if any(len(s) for s in sets) else 0),
        "m_gate": float(gate)})
    if "F_hat" in latent:
        f = np.asarray(latent["F_hat"], float)
        out.update({"F_hat_mean": float(f.mean()), "F_hat_sd": float(f.std()),
                    "F_hat_abs_err": float(np.abs(f - F).mean()),
                    "F_hat_within_half": float((np.abs(f - F) <= 0.5).mean())})
    if "n" in latent:
        n = np.asarray(latent["n"], float)
        top = -np.sort(-n, axis=1)[:, :F]
        out.update({"launch_topF_mass": float(top.sum(1).mean()),
                    "launch_multiplicity_max": float(n.max(1).mean())})
        # ⭐ the launch's own top-F set accuracy — the estimator §4.1 measured as
        # BETTER than the settled occupancy's (0.016 vs 0.000). Reported per cell.
        idx = np.argsort(-n, axis=1)[:, :F]
        out["launch_topF_exact_set"] = float(np.mean(
            [set(i.tolist()) == set(np.asarray(A).tolist())
             for i, A in zip(idx, subsets, strict=True)]))
    out.update({k: v for k, v in launch_collapse_stat(
        latent["m"], int(np.asarray(anchors).shape[0])).items()})
    return out


# ==========================================================================
# training — staged (guard 3), soft counting-code signal (guard 2)
# ==========================================================================
def mult_read_loss(store: FactoredStore, head: MultiplicityHead, phi: FrozenPhi,
                   cfg: CatTestConfig, mc: MultiplicityConfig, ind, Y, key, *,
                   wells=None, well_payloads=None, hard: bool = False,
                   w_occ: Optional[float] = None, w_pay: Optional[float] = None,
                   detach_w: bool = False, detach_f: bool = False,
                   lam_ac: Optional[float] = None,
                   address_steps: Optional[int] = None,
                   read_steps: Optional[int] = None):
    """The counting-code objective, differentiable through the settle.

    * ``w_occ`` — ⭐ the **counting-code channel** ``|| m @ V_table − y ||²``. With
      ``hard=True`` the soft occupancy is replaced by an ``argmax`` one-hot whose
      derivative is identically zero — guard 2's designed negative.
    * ``w_pay`` — the **continuous** channel: the importance-weighted mean payload
      block ``F_hat · Σ_i a_i pay(z_i) / Σ_i a_i``, which is what ``sum_linear``
      reads and what only the *landscape* can supply.
    * ``lam_ac`` — the batch-level anti-collapse penalty (**MARGINAL only**),
      default ``mc.lambda_anticollapse`` = 0.0 (built OFF, doctrine §3.3).

    ⛔ ``detach_f`` exists because of a **measured** change from iteration 1: at a
    multiplicity head the counting-code channel has a **second** differentiable
    path — the cardinality scale ``F_hat`` multiplies ``m`` — which survives a
    hard (``argmax``) assignment. With ``F_hat`` live, guard 2's hard/soft ratio
    is **1.00**, i.e. the head still trains under ``argmax``, *through
    cardinality*. Guard 2 asks about the **assignment**, so its probe detaches
    both the magnitude (``detach_w``) and the cardinality (``detach_f``) channels;
    the un-detached ratio is reported beside it as the finding it is.
    """
    wells = head.anchors if wells is None else jnp.asarray(wells, jnp.float32)
    d = int(cfg.addr_dim)
    w_occ = float(mc.w_occ if w_occ is None else w_occ)
    w_pay = float(mc.w_pay if w_pay is None else w_pay)
    lam_ac = float(mc.lambda_anticollapse if lam_ac is None else lam_ac)
    a_steps = int(cfg.address_steps if address_steps is None else address_steps)
    r_steps = int(cfg.read_steps if read_steps is None else read_steps)
    model = _model(store, cfg)
    c = query_code(phi, ind, key, float(cfg.query_sigma))
    q0, p0, mass, gmult, conf, _beta, _n, f_hat = jax.vmap(head)(c)

    def run(q, p, mm, gm):
        q, p = settle_particles(model, q, p, mm, gm * float(cfg.gamma_address),
                                a_steps, float(cfg.dt))
        q, _ = settle_particles(model, q, jnp.zeros_like(p), mm,
                                gm * float(cfg.gamma_read), r_steps, float(cfg.dt))
        return q

    z = jax.vmap(run)(q0, p0, mass, gmult)
    w = (descent_weight(z[..., d:], mc.payload_ref) if mc.descent_gate
         else jnp.ones(z.shape[:2]))
    a = particle_weights(w, conf, mc.weight_mode)
    loss = 0.0
    if w_pay != 0.0:
        y_pay = (f_hat[:, None] * (z[..., d:] * a[..., None]).sum(1)
                 / (a.sum(1)[:, None] + 1e-9))
        loss = loss + w_pay * jnp.mean(jnp.sum((y_pay - Y) ** 2, -1))
    if w_occ != 0.0 and well_payloads is not None:
        s = soft_occupancy(z[..., :d], wells, mc.occ_tau)
        if hard:
            s = jax.nn.one_hot(jnp.argmax(s, axis=-1), wells.shape[0])
        if detach_w:
            # ⛔ guard 2's isolation: with the magnitude channel detached the ONLY
            # path left to the parameters is the assignment. Soft => O(1);
            # hard => EXACTLY zero (argmax has no derivative).
            a = jax.lax.stop_gradient(a)
        cnt = counting_code(s, a, mc.count_agg)
        m_code = importance_code(cnt, jax.lax.stop_gradient(f_hat) if detach_f
                                 else f_hat)
        y_occ = m_code @ jnp.asarray(well_payloads, jnp.float32)
        loss = loss + w_occ * jnp.mean(jnp.sum((y_occ - Y) ** 2, -1))
        if lam_ac != 0.0:
            loss = loss + lam_ac * anticollapse_penalty(m_code, wells.shape[0])
    return loss


def train_multiplicity_head(store: FactoredStore, head: MultiplicityHead,
                            phi: FrozenPhi, cfg: CatTestConfig,
                            mc: MultiplicityConfig, family, key, *, wells=None,
                            steps: Optional[int] = None
                            ) -> Tuple[MultiplicityHead, Dict[str, Any]]:
    """⭐ **Guard 3's ordering: store first, launch head second** (w20/§A20.3(c)).

    The head's gradient on a blank store is ~1 300× weaker than on a written one
    (iteration 1 §8), so this is only ever called on a **written** store;
    :func:`mc_staging_gradient_probe` measures what happens if it is not.
    ⚠ Trained at the reduced ``head_settle_*`` budget, declared beside every
    number it produces (C2W4 read-budget-scoping rule).
    """
    import optax

    ind_all = jnp.asarray(family.indicator(family.seen, cfg.n_wells))
    Y_all = jnp.asarray(family.y_seen, jnp.float32)
    spec = mult_head_trainable_spec(head)
    params, static = eqx.partition(head, spec)
    opt = optax.adam(float(mc.head_lr))
    st = opt.init(params)
    n = int(family.seen.shape[0])

    @eqx.filter_jit
    def _step(params, st, idx, k):
        def loss_fn(p):
            return mult_read_loss(store, eqx.combine(p, static), phi, cfg, mc,
                                  ind_all[idx], Y_all[idx], k, wells=wells,
                                  well_payloads=family.payloads,
                                  address_steps=mc.head_settle_address,
                                  read_steps=mc.head_settle_read)

        val, g = eqx.filter_value_and_grad(loss_fn)(params)
        u, st = opt.update(g, st, params)
        return eqx.apply_updates(params, u), st, val

    hist = []
    for _ in range(int(mc.head_steps if steps is None else steps)):
        key, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.choice(k1, n, (min(int(mc.head_batch), n),), replace=False)
        params, st, val = _step(params, st, idx, k2)
        hist.append(float(val))
    trained = eqx.combine(params, static)
    return trained, {
        "head_loss": hist, "head_loss_first": hist[0] if hist else float("nan"),
        "head_loss_last": hist[-1] if hist else float("nan"),
        "head_settle_budget": [int(mc.head_settle_address),
                               int(mc.head_settle_read)],
        "head_steps": int(mc.head_steps if steps is None else steps),
        "lambda_anticollapse": float(mc.lambda_anticollapse),
        "card_a": [float(head.card_a), float(trained.card_a)],
        "card_b": [float(head.card_b), float(trained.card_b)],
        "ista_lam": [float(jnp.exp(head.log_lam)),
                     float(jnp.exp(trained.log_lam))],
        "rho": [float(head.rho), float(trained.rho)]}


def organize_store_mc(store: FactoredStore, head: MultiplicityHead,
                      phi: FrozenPhi, cfg: CatTestConfig, mc: MultiplicityConfig,
                      family, key, *, wells=None, steps: int = 60,
                      lr: float = 3e-3) -> Tuple[FactoredStore, Dict[str, Any]]:
    """⭐ The PHYSICS organizer: the store trained **through the settle**.

    The organizer-swap null (``null_arms.n1_gradient_placed``) optimises the *same
    parameters* against the *same objective* with a **static** softmax read and no
    dynamics anywhere; only the organizer varies.
    """
    import optax

    ind = jnp.asarray(family.indicator(family.seen, cfg.n_wells))
    Y = jnp.asarray(family.y_seen, jnp.float32)
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, store.V)
    params, static = eqx.partition(store.V, spec)
    opt = optax.adam(float(lr))
    st = opt.init(params)
    n = int(family.seen.shape[0])

    @eqx.filter_jit
    def _step(params, st, idx, k):
        def loss_fn(p):
            s2 = eqx.tree_at(lambda t: t.V, store, eqx.combine(p, static))
            return mult_read_loss(s2, head, phi, cfg, mc, ind[idx], Y[idx], k,
                                  wells=wells, well_payloads=family.payloads,
                                  address_steps=mc.head_settle_address,
                                  read_steps=mc.head_settle_read)

        val, g = eqx.filter_value_and_grad(loss_fn)(params)
        u, st = opt.update(g, st, params)
        return eqx.apply_updates(params, u), st, val

    hist = []
    for _ in range(int(steps)):
        key, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.choice(k1, n, (min(int(mc.head_batch), n),), replace=False)
        params, st, val = _step(params, st, idx, k2)
        hist.append(float(val))
    store = eqx.tree_at(lambda s: s.V, store, eqx.combine(params, static))
    return store, {"organize_loss": hist,
                   "organize_loss_first": hist[0] if hist else float("nan"),
                   "organize_loss_last": hist[-1] if hist else float("nan"),
                   "organize_steps": int(steps), "organize_lr": float(lr),
                   "organize_settle_budget": [int(mc.head_settle_address),
                                              int(mc.head_settle_read)]}


def mc_hard_vs_soft_gradient(store, head, phi, cfg, mc, family, key, *, n: int = 8
                             ) -> Dict[str, float]:
    """⛔ **Guard 2's designed negative.** Hard assignments do not backprop."""
    ind = jnp.asarray(family.indicator(family.seen[:n], cfg.n_wells))
    Y = jnp.asarray(family.y_seen[:n], jnp.float32)
    spec = mult_head_trainable_spec(head)
    params, static = eqx.partition(head, spec)

    def norm(hard, detach_f):
        def loss_fn(p):
            return mult_read_loss(store, eqx.combine(p, static), phi, cfg, mc,
                                  ind, Y, key, hard=hard, w_occ=1.0, w_pay=0.0,
                                  detach_w=True, detach_f=detach_f,
                                  well_payloads=family.payloads,
                                  address_steps=mc.head_settle_address,
                                  read_steps=mc.head_settle_read)

        g = eqx.filter_grad(loss_fn)(params)
        leaves = [x for x in jax.tree_util.tree_leaves(g) if x is not None]
        return float(jnp.sqrt(sum(jnp.sum(jnp.asarray(x) ** 2) for x in leaves)))

    soft, hard = norm(False, True), norm(True, True)
    soft_f, hard_f = norm(False, False), norm(True, False)
    return {"grad_soft": soft, "grad_hard": hard,
            "ratio_hard_over_soft": hard / max(soft, 1e-30),
            # ⭐ the SAME probe with the cardinality channel LIVE — the C2W7
            # finding: F_hat keeps training the head even under argmax.
            "grad_soft_F_live": soft_f, "grad_hard_F_live": hard_f,
            "ratio_hard_over_soft_F_live": hard_f / max(soft_f, 1e-30)}


def mc_staging_gradient_probe(blank, written, head, phi, cfg, mc, family, key, *,
                              n: int = 8) -> Dict[str, float]:
    """⛔ **Guard 3's designed negative.** Un-staged co-training has no gradient."""
    ind = jnp.asarray(family.indicator(family.seen[:n], cfg.n_wells))
    Y = jnp.asarray(family.y_seen[:n], jnp.float32)
    hspec = mult_head_trainable_spec(head)
    hp, hs = eqx.partition(head, hspec)

    def norms(store):
        def loss_head(p):
            return mult_read_loss(store, eqx.combine(p, hs), phi, cfg, mc, ind, Y,
                                  key, well_payloads=family.payloads,
                                  address_steps=mc.head_settle_address,
                                  read_steps=mc.head_settle_read)

        gh = eqx.filter_grad(loss_head)(hp)
        nh = float(jnp.sqrt(sum(jnp.sum(jnp.asarray(x) ** 2)
                                for x in jax.tree_util.tree_leaves(gh))))
        vspec = jax.tree_util.tree_map(eqx.is_inexact_array, store.V)
        vp, vs = eqx.partition(store.V, vspec)

        def loss_store(p):
            s2 = eqx.tree_at(lambda t: t.V, store, eqx.combine(p, vs))
            return mult_read_loss(s2, head, phi, cfg, mc, ind, Y, key,
                                  well_payloads=family.payloads,
                                  address_steps=mc.head_settle_address,
                                  read_steps=mc.head_settle_read)

        gv = eqx.filter_grad(loss_store)(vp)
        nv = float(jnp.sqrt(sum(jnp.sum(jnp.asarray(x) ** 2)
                                for x in jax.tree_util.tree_leaves(gv))))
        return nh, nv

    bh, bv = norms(blank)
    wh, wv = norms(written)
    return {"grad_head_blank": bh, "grad_store_blank": bv,
            "grad_head_written": wh, "grad_store_written": wv,
            "head_ratio_blank_over_written": bh / max(wh, 1e-30),
            "store_ratio_blank_over_written": bv / max(wv, 1e-30)}


# ==========================================================================
# ledger — ⛔ k and the learned head's params are on EVERY arm's row (guard 4)
# ==========================================================================
def mc_ledger(arm: str, cfg: CatTestConfig, mc: MultiplicityConfig, *,
              store_params: int, head: MultiplicityHead, phi_bytes: int,
              reader_params: int = 0, **extra) -> Dict[str, Any]:
    """The row an arm may not be scored without (guard 4's object)."""
    read_flops = int(mc.k_particles * (cfg.address_steps + cfg.read_steps)
                     * cfg.n_atoms * (cfg.dim + 1))
    return {"arm": arm, "k_particles": int(mc.k_particles),
            "ista_steps": int(mc.ista_steps), "f_max": int(mc.f_max),
            "store_params": int(store_params), "store_bytes": int(store_params) * 4,
            "head_params": int(head.n_params()),
            "head_bytes": int(head.n_params()) * 4,
            "phi_bytes": int(phi_bytes), "reader_params": int(reader_params),
            "total_bytes": int(store_params) * 4 + int(head.n_params()) * 4
            + int(phi_bytes),
            "read_flops_per_query": read_flops, **extra}
