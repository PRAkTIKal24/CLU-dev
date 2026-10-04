"""The **multi-well read protocol** — tier ii's read-fix iteration (charter §A20.3).

⛔ **What this module replaces, and why it is a replacement and not a patch.**
The shipped ``P``-particle frozen-offset occupancy read of
:func:`chlu.core.factored_store.multi_particle_read` is **REFUTED at ``P = 4``**
(charter §A20.1, from ``orgdiv-null-arms`` §3): the ``P = 4`` particles occupy
**2.20** distinct wells where the ``F = 4``-term sum needs 4; ``>= F`` distinct
wells are reachable on **5.0 %** of queries; exact-set occupancy is
**0.0000 / 2560**; and the physics settle *lowers* the distinct count further
(2.20 -> 1.70). Two requirements fall out of that measurement and both are
design inputs here (``orgdiv-null-arms`` §12.1):

1. the read must **address ``F`` distinct wells**, and
2. it must **not quantise away the continuous launch coordinate** (quantisation
   is the destructive step: ``N1``/``N2``/``N3`` all replace the launch point
   with a codebook centre, discarding exactly the continuous coordinate the
   0.272 out-of-class ceiling decodes).

**The taxonomy this module implements** (charter §A20.3(f), on the record):

    launch selects the memory  ·  particle attributes select the read style  ·
    well geometry determines the memory mode

* **launch selects the memory** — :class:`LaunchHead` emits ``k`` full particles
  from the (noisy, continuous) set code: ``k`` launch positions, one *inertial
  mass* each (``mass_override``, the shipped Prop-6 per-address mass), one
  *friction* each, and one *initial momentum* each.
* **particle attributes select the read style** — ``conf_i`` (the head's own
  overlap, i.e. overlap-as-confidence) sets mass/friction/``p0``: an unconfident
  particle is heavy and over-damped and **cannot reach the well bottom inside
  the read budget**, so its payload block stays near 0 and it contributes
  nothing to the sum. ⭐ **This is where the continuous launch coordinate
  survives the read**: the launch *position* is necessarily near-discrete (it
  has to address a well), so the continuous code is carried by the attributes
  and by the soft occupancy, never quantised away.
* **well geometry determines the memory mode** — :func:`consolidate_wells`
  merges over-dug minima to the designed budget on mechanical criteria and
  routes spurious shallow wells to the **trash region** ``gamma_phi(q)``
  (:mod:`chlu.core.friction_field`, built in C1 and never used until here).

⛔ **No ``argmax`` appears anywhere in the latent a reader sees.** Hard
occupancy (:func:`chlu.core.factored_store.occupancy`) is computed in this
module for **diagnostics only** (R1/R2, the statistics the refutation is stated
in) and is never an input to a fitted reader.

⛔ **Claim-form discipline (``PREREG-TierII.md`` §2.6, inherited verbatim).** No
well is named semantically anywhere in this module, its tests or its artifacts.
Wells carry integer indices and nothing else.
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
from chlu.core.friction_field import FrictionField

__all__ = [
    "MultiWellReadConfig",
    "LaunchHead",
    "head_trainable_spec",
    "query_code",
    "settle_particles",
    "soft_occupancy",
    "aggregate_occupancy",
    "descent_weight",
    "multiwell_read",
    "launch_only_launder",
    "read_stats",
    "s_effective",
    "find_wells",
    "consolidate_wells",
    "trash_field",
    "soft_well_table_fit",
    "soft_well_table_apply",
    "fit_readers_mw",
    "apply_reader_mw",
    "score_readers_mw",
    "READERS_MW",
    # -- the ZERO-PARAMETER identity twins (C2W7 reconciliation 1) -----------
    "READERS_MW_IDENTITY",
    "READERS_MW_PLUS_IDENTITY",
    "soft_well_identity_fit",
    "soft_well_identity_apply",
    "gated_well_identity_fit",
    "gated_well_identity_apply",
    "read_loss",
    "train_launch_head",
    "organize_store_mw",
    "hard_vs_soft_gradient",
    "staging_gradient_probe",
    "mwr_ledger",
    "assert_k_matched",
]


# ==========================================================================
# config — lives next to its code (the CatTestConfig / NullArmGrid precedent)
# ==========================================================================
@dataclass
class MultiWellReadConfig:
    """Every knob of the multi-well read, at its **registered** value.

    Defaults are ``.claude/outputs/tierii-read-fix/PREREG.md`` §0/§2's registered
    operating point, which was fixed by the **store-free K0 study** before this
    module existed. ⛔ ``k_particles`` is CAPACITY: it is on the byte ledger and
    must be identical on every arm (guard 4).
    """

    # -- the k-particle launch head (PREREG §0: K0 = 1.000 at these values) ---
    k_particles: int = 12  # ⭐ k  (registered; k = 16 is the declared NOT-RUN)
    head_tau: float = 0.002  # slot softmax temperature
    head_kappa: float = 2.0  # successive-suppression strength
    head_rho: float = 0.25  # continuous-residual injection
    head_gain: float = 1.0  # gain on <c_hat, e_j>
    # confidence -> attributes.  conf = the gated OVERLAP, in [0, 1].
    log_mass_conf: float = 0.0  # log M at conf = 1
    log_mass_unconf: float = 1.5  # log M at conf = 0  (heavy => sluggish)
    gamma_mult_conf: float = 1.0  # gamma multiplier at gate = 1
    gamma_mult_unconf: float = 4.0  # at gate = 0 (over-damped => stops short)
    # overlap -> gate:  g = sigmoid((overlap - conf_b) / conf_w).  Designed init:
    # for j in A(x) the overlap <c_hat, e_j> ~ 1/sqrt(F) = 0.5, for j not in A it
    # is ~0 +- 1/sqrt(d); the gate is centred between them.
    conf_b: float = 0.25
    conf_w: float = 0.15
    p0_gain: float = 1.0  # ⭐ lever (e): learned p0, ballistic reach
    learned_p0: bool = True  # the (e) ablation switch

    # -- the read latent ------------------------------------------------------
    occ_tau: float = 0.25  # soft-occupancy temperature (address units)
    dedupe: str = "noisy_or"  # {"noisy_or", "max", "sum"} — the dedupe verb
    payload_ref: float = 1.0  # ||v_j||; sets the descent weight's scale
    descent_gate: bool = True  # weight particles by how far they descended

    # -- (d) consolidate-to-budget + trash-region pruning ---------------------
    consolidate: bool = True
    well_budget: int = 32  # the DESIGNED budget (= N_a)
    merge_radius_frac: float = 0.5  # merge radius = frac * sep (mechanical)
    n_probes: int = 512  # probe cloud for minima discovery
    probe_steps: int = 300
    trash_depth_frac: float = 0.20  # prune below frac * median kept depth
    trash_gamma: float = 0.4  # gamma_phi strength inside a trash horizon
    trash_radius_frac: float = 0.5  # trash horizon radius = frac * sep

    # -- staging (guard 3) ----------------------------------------------------
    head_steps: int = 60
    head_lr: float = 3e-2
    head_batch: int = 32
    head_settle_address: int = 100  # ⚠ reduced TRAINING settle budget, declared
    head_settle_read: int = 200

    # -- bookkeeping ----------------------------------------------------------
    seed: int = 0

    def as_flag_table(self) -> Dict[str, Any]:
        base = MultiWellReadConfig()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================================
# (c) the k-particle learned-launch head
# ==========================================================================
class LaunchHead(eqx.Module):
    """``c_tilde -> k`` full particles ``(q0, p0, mass, gamma)``.

    ``s_j = gain * <c_hat, e_j> + b_j``; slot ``i`` takes a **successive
    suppression** softmax so the ``k`` slots address ``k`` *different* wells:

        ``alpha^(i) = softmax((s - kappa * sum_{l<i} alpha^(l)) / tau)``
        ``q0_i      = sum_j alpha^(i)_j u_j + rho*R*(c_hat - sum_j alpha^(i)_j e_j)``

    ⭐ ``q0_i`` is a **continuous mixture** of anchors plus a continuous residual,
    never an index. ``conf_i = sigmoid((<c_hat, e_selected> - b)/w)`` is
    *overlap-as-confidence*
    (confident => the slots collapse onto unique wells; unfamiliar => scattered
    guesses) and it drives the per-particle mass / friction / ``p0``.

    ⛔ ``codes`` and ``anchors`` are the FROZEN launch geometry (a pure function
    of ``phi`` and the shared placement policy) and are **not** trainable — see
    :func:`head_trainable_spec`, which is what the optimiser is given.
    """

    codes: jnp.ndarray  # (N_a, d) frozen query codes
    anchors: jnp.ndarray  # (N_a, d) frozen well anchors
    slot_offset: jnp.ndarray  # (k, d) learned per-slot offsets, init 0
    bias: jnp.ndarray  # (N_a,) learned per-well bias, init 0
    gain: jnp.ndarray  # scalar
    log_tau: jnp.ndarray
    log_kappa: jnp.ndarray
    rho: jnp.ndarray
    log_mass_c: jnp.ndarray
    log_mass_u: jnp.ndarray
    gam_mult_c: jnp.ndarray
    gam_mult_u: jnp.ndarray
    p0_gain: jnp.ndarray
    conf_b: jnp.ndarray
    conf_w: jnp.ndarray
    k: int = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    use_p0: bool = eqx.field(static=True)

    def __init__(self, phi: FrozenPhi, anchors, cfg: CatTestConfig,
                 mw: MultiWellReadConfig):
        self.codes = jnp.asarray(phi.codes)
        self.anchors = jnp.asarray(np.asarray(anchors)[:, : cfg.addr_dim],
                                   dtype=jnp.float32)
        self.k = int(mw.k_particles)
        self.addr_dim = int(cfg.addr_dim)
        self.payload_dim = int(cfg.payload_dim)
        self.radius = float(cfg.ball_radius)
        self.use_p0 = bool(mw.learned_p0)
        z = jnp.zeros(())
        self.slot_offset = jnp.zeros((self.k, self.addr_dim))
        self.bias = jnp.zeros((int(cfg.n_wells),))
        self.gain = z + float(mw.head_gain)
        self.log_tau = z + float(np.log(mw.head_tau))
        self.log_kappa = z + float(np.log(mw.head_kappa))
        self.rho = z + float(mw.head_rho)
        self.log_mass_c = z + float(mw.log_mass_conf)
        self.log_mass_u = z + float(mw.log_mass_unconf)
        self.gam_mult_c = z + float(mw.gamma_mult_conf)
        self.gam_mult_u = z + float(mw.gamma_mult_unconf)
        self.p0_gain = z + float(mw.p0_gain)
        self.conf_b = z + float(mw.conf_b)
        self.conf_w = z + float(mw.conf_w)

    def n_params(self) -> int:
        """Head parameters, ledgered on EVERY arm (the launch protocol is shared)."""
        return int(self.slot_offset.size + self.bias.size + 11)

    def __call__(self, c_noisy: jnp.ndarray):
        """``(d,) -> (q0 (k,dim), p0 (k,dim), mass (k,dim), gamma_mult (k,), conf (k,))``."""
        d, m, R = self.addr_dim, self.payload_dim, self.radius
        ch = c_noisy / (jnp.linalg.norm(c_noisy) + 1e-12)
        s = self.gain * (self.codes @ ch) + self.bias  # (N_a,)
        tau = jnp.exp(self.log_tau)
        kappa = jnp.exp(self.log_kappa)

        ov = self.codes @ ch  # (N_a,) the RAW overlaps
        used = jnp.zeros_like(s)
        q_list, conf_list, back_list = [], [], []
        for i in range(self.k):
            a = jax.nn.softmax((s - kappa * used) / tau)
            mix_u = a @ self.anchors  # (d,)
            mix_e = a @ self.codes  # (d,)
            q_i = mix_u + self.rho * R * (ch - mix_e) + self.slot_offset[i]
            q_list.append(q_i)
            back_list.append(mix_u - q_i)  # points back INTO the addressed well
            # ⭐ OVERLAP-as-confidence: the *overlap of the addressed well with
            # the query code*, not the slot softmax's own peak. `max_j alpha_ij`
            # is ~1 for every slot at a low temperature (measured: 0.958 +- 0.04
            # across all k) and therefore carries no information about whether
            # this slot's well is actually in A(x); the overlap does (it falls
            # monotonically down the k slots). Confident => the k collapse onto
            # F unique wells; unfamiliar => k scattered guesses.
            conf_list.append(a @ ov)
            used = used + a

        q_addr = jnp.stack(q_list)  # (k, d)
        overlap = jnp.stack(conf_list)  # (k,) raw overlaps
        conf = jax.nn.sigmoid((overlap - self.conf_b) / (self.conf_w + 1e-6))
        back = jnp.stack(back_list)  # (k, d)
        q0 = jnp.concatenate([q_addr, jnp.zeros((self.k, m))], axis=1)

        # ⭐ (e) learned p0: a confidence-gated ballistic kick toward the
        # addressed well — reach across informationally dead inter-well gaps.
        if self.use_p0:
            p_addr = self.p0_gain * conf[:, None] * back
        else:
            p_addr = jnp.zeros_like(q_addr)
        p0 = jnp.concatenate([p_addr, jnp.zeros((self.k, m))], axis=1)

        # attributes: confident -> light + lightly damped; unconfident -> heavy
        # + over-damped (cannot reach the bottom inside the read budget).
        u = 1.0 - conf  # (k,)
        logM = self.log_mass_c + (self.log_mass_u - self.log_mass_c) * u
        mass = jnp.exp(logM)[:, None] * jnp.ones((1, d + m))
        gmult = self.gam_mult_c + (self.gam_mult_u - self.gam_mult_c) * u
        return q0, p0, mass, gmult, conf


def head_trainable_spec(head: LaunchHead):
    """Filter spec selecting ONLY the head's learned leaves.

    ⛔ ``codes``/``anchors`` are the frozen launch geometry, shared byte-identically
    by every arm (``PREREG-TierII.md`` §1). They are inexact arrays, so
    ``eqx.is_inexact_array`` alone would hand them to the optimiser.
    """
    spec = jax.tree_util.tree_map(lambda _: False, head)
    for name in ("slot_offset", "bias", "gain", "log_tau", "log_kappa", "rho",
                 "log_mass_c", "log_mass_u", "gam_mult_c", "gam_mult_u",
                 "p0_gain", "conf_b", "conf_w"):
        spec = eqx.tree_at(lambda t, n=name: getattr(t, n), spec,
                           replace=jax.tree_util.tree_map(
                               lambda _: True, getattr(head, name)))
    return spec


def query_code(phi: FrozenPhi, indicators, key, sigma_q: float) -> jnp.ndarray:
    """``(B, N_a) -> (B, d)`` the noisy set code — ⛔ **ONE draw per query**.

    Registered deviation D7: the shipped launch drew ``sigma_q`` i.i.d. **per
    particle**, so a ``k``-particle head could average the query noise away
    (``sigma/sqrt(k)``) and buy its score from the compute dial. One draw per
    query makes the new protocol see **strictly less** launch information than
    the refuted one.

    ⛔ Called ONCE on the whole split (never per batch): the noisy code a query
    gets must not depend on the batch size, or the read and its launder would not
    be scored on the same query (guard 1 would compare two different problems).
    """
    c = phi.set_code(jnp.asarray(indicators, dtype=jnp.float32))
    if key is None or float(sigma_q) <= 0.0:
        return c
    return c + float(sigma_q) * jax.random.normal(key, c.shape)


# ==========================================================================
# (b) the read — per-particle attributes, no argmax in the latent
# ==========================================================================
def settle_particles(model, q, p, mass, gamma, steps: int, dt: float):
    """``steps`` damped Verlet steps with a **per-particle** mass and friction.

    ``model.step``'s ``mass_override`` is the shipped Prop-6 per-address mass, so
    the physics here is the shipped symplectic Verlet step — only the attributes
    are per-particle. Carries ``(q, p)`` only (no tape): the trajectory stack for
    ``B*k`` particles is ~GB and is never used by this read.

    ⚠ The body is ``jax.checkpoint``-ed. Without it, reverse-mode AD through a
    300-step settle of ``B*k`` particles against ``n_atoms x dim`` atoms stores
    ~25 MB of per-atom residuals PER STEP (~7 GB at the registered cell) and the
    organizer run is **killed by the OS** (measured: the first full run died
    silently at ``[stage] arms``). With it, only the ``(q, p)`` carries are taped
    and the body is recomputed on the backward pass; the forward path is
    unchanged (nothing is recomputed when there is no backward pass).
    """
    @jax.checkpoint
    def body(carry, _):
        qq, pp = carry
        qq, pp = jax.vmap(
            lambda a, b, mm, gg: model.step((a, b), dt, gg, mass_override=mm)
        )(qq, pp, mass, gamma)
        return (qq, pp), None

    (q, p), _ = jax.lax.scan(body, (q, p), None, length=int(steps))
    return q, p


def soft_occupancy(z_addr, wells, tau: float):
    """``(..., d) x (N_w, d) -> (..., N_w)`` soft assignment. ⛔ No ``argmax``."""
    w = jnp.asarray(wells)
    w = w.reshape((1,) * (jnp.ndim(z_addr) - 1) + w.shape)
    d2 = jnp.sum((z_addr[..., None, :] - w) ** 2, axis=-1)
    return jax.nn.softmax(-d2 / (2.0 * float(tau) ** 2), axis=-1)


def descent_weight(z_pay, payload_ref: float):
    """How far a particle descended, in ``[0, 1]``, from its payload-block norm.

    A particle that reached a well bottom carries ``||pay|| ~ payload_radius``;
    one that stalled in the flat inter-well region carries ``~0``. This is the
    **continuous** gate the confidence-modulated attributes act through.

    ⚠ ``sqrt(x^2 + eps)`` rather than ``norm``: the launch pins the payload block
    to exactly 0 (the anti-decoration guard), and ``grad ||.||`` at 0 is NaN.
    """
    n = jnp.sqrt(jnp.sum(z_pay ** 2, axis=-1) + 1e-12)
    return jnp.clip(n / float(payload_ref), 0.0, 1.0)


def aggregate_occupancy(pi, w, mode: str = "noisy_or"):
    """``(B, k, N_w) x (B, k) -> (B, N_w)`` — the **dedupe / evolve-unique** verb.

    ⭐ The refuted read summed the ``P`` particles' payloads, so two particles in
    the same well contributed that well **twice** while two of ``A(x)``'s wells
    went unvisited. A SET union is the right aggregation for a set-valued answer:

    * ``noisy_or``  ``pi_j = 1 - prod_i (1 - w_i pi_ij)``  (registered default)
    * ``max``       ``pi_j = max_i w_i pi_ij``
    * ``sum``       the refuted multiset behaviour, kept as a designed negative
    """
    x = pi * w[..., None]
    if mode == "noisy_or":
        return 1.0 - jnp.prod(1.0 - jnp.clip(x, 0.0, 1.0 - 1e-6), axis=1)
    if mode == "max":
        return jnp.max(x, axis=1)
    if mode == "sum":
        return jnp.sum(x, axis=1)
    raise ValueError(f"unknown dedupe mode {mode!r}")


def _model_with_trash(store: FactoredStore, cfg: CatTestConfig,
                      field: Optional[FrictionField]):
    model = store.model(cfg)
    if field is None:
        return model
    return eqx.tree_at(lambda t: t.friction_field, model, field,
                       is_leaf=lambda x: x is None)


def multiwell_read(store: FactoredStore, head: LaunchHead, phi: FrozenPhi,
                   cfg: CatTestConfig, mw: MultiWellReadConfig,
                   indicators, key, *, wells=None,
                   trash: Optional[FrictionField] = None,
                   batch: int = 128) -> Dict[str, np.ndarray]:
    """The full multi-well read. Returns the latent dict every reader consumes.

    ``z``        (B, k, dim)  the **continuous** settled states
    ``pi``       (B, N_w)     the aggregated **soft** occupancy (dedupe verb)
    ``conf``     (B, k)       the head's overlap-as-confidence
    ``w``        (B, k)       the descent weight (how far each particle got)
    ``q0``       (B, k, dim)  the launch states (for the live launder, G1)
    """
    wells = head.anchors if wells is None else jnp.asarray(wells, jnp.float32)
    model = _model_with_trash(store, cfg, trash)
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B = int(ind.shape[0])
    d = int(cfg.addr_dim)

    c_all = query_code(phi, ind, key, float(cfg.query_sigma))

    @eqx.filter_jit
    def _one_batch(c):
        q0, p0, mass, gmult, conf = jax.vmap(head)(c)

        def run(q, p, mm, gm):
            q, p = settle_particles(model, q, p, mm,
                                    gm * float(cfg.gamma_address),
                                    int(cfg.address_steps), float(cfg.dt))
            q, p = settle_particles(model, q, jnp.zeros_like(p), mm,
                                    gm * float(cfg.gamma_read),
                                    int(cfg.read_steps), float(cfg.dt))
            return q

        z = jax.vmap(run)(q0, p0, mass, gmult)
        w = (descent_weight(z[..., d:], mw.payload_ref) if mw.descent_gate
             else jnp.ones(z.shape[:2]))
        pi = aggregate_occupancy(soft_occupancy(z[..., :d], wells, mw.occ_tau),
                                 w, mw.dedupe)
        return z, pi, conf, w, q0

    outs = {kk: [] for kk in ("z", "pi", "conf", "w", "q0")}
    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        for name, val in zip(("z", "pi", "conf", "w", "q0"),
                             _one_batch(c_all[lo:hi]), strict=True):
            outs[name].append(np.asarray(val))
    return {kk: np.concatenate(v, axis=0) for kk, v in outs.items()}


def launch_only_launder(head: LaunchHead, phi: FrozenPhi, cfg: CatTestConfig,
                        mw: MultiWellReadConfig, indicators, key, *, wells=None,
                        batch: int = 512) -> Dict[str, np.ndarray]:
    """⛔ **Guard 1, RECOMPUTED LIVE** (Advisor amendment A1, binding).

    The launder is *this task's own learned launches* with the **store deleted**,
    scored through the **same reader class at the same ``k``**. "Store deleted"
    is implemented in its **strongest** form: the landscape is gone (zero settle:
    the particles never move) but the written payload table is **retained**, so
    the launder is exactly *"the head plus a nearest-well table"* — the trivial
    substitute this whole programme has to beat.

    ⛔ ``orgdiv-null-arms``' 0.272 is the OLD protocol's out-of-class reference
    ceiling and is never the live bar; this function computes the bar.
    """
    wells = head.anchors if wells is None else jnp.asarray(wells, jnp.float32)
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B, d = int(ind.shape[0]), int(cfg.addr_dim)

    c_all = query_code(phi, ind, key, float(cfg.query_sigma))

    @eqx.filter_jit
    def _one(c):
        q0, _p0, _m, _g, conf = jax.vmap(head)(c)
        w = jnp.ones(q0.shape[:2])  # no descent happened: nothing to gate on
        pi = aggregate_occupancy(soft_occupancy(q0[..., :d], wells, mw.occ_tau),
                                 w, mw.dedupe)
        return q0, pi, conf, w

    outs = {kk: [] for kk in ("z", "pi", "conf", "w")}
    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        z, pi, conf, w = _one(c_all[lo:hi])
        for name, val in zip(("z", "pi", "conf", "w"), (z, pi, conf, w), strict=True):
            outs[name].append(np.asarray(val))
    out = {kk: np.concatenate(v, axis=0) for kk, v in outs.items()}
    out["q0"] = out["z"]
    return out


# ==========================================================================
# diagnostics — R1 / R2 / S_eff (⛔ hard occupancy lives HERE and only here)
# ==========================================================================
def read_stats(z_or_latent, anchors, subsets, F: int, *, gate: float = 0.5
               ) -> Dict[str, float]:
    """The statistics the §A20 refutation is stated in, recomputed on the new read.

    ⛔ Uses a hard nearest-well assignment **for diagnostics only** — it is never
    an input to a fitted reader (that quantisation is the refuted step).

    Two families of numbers, and the distinction is load-bearing:

    * ``*_raw`` — every launched particle counts. Directly comparable to
      ``orgdiv-null-arms`` §3 (which measured 2.202 distinct / 0.050 ``>=F`` /
      0.4106 precision / **0.0000** exact-set at ``P = 4``). ⚠ With ``k > F``
      distinct particles the *raw* exact-set statistic is 0 **by construction**
      (an occupied set of size ``k`` cannot equal a set of size ``F``), so it is
      reported for continuity and is not R1's statistic.
    * ``*_gated`` — the **effective** occupied set ``{j : pi_j >= gate}`` after
      the dedupe verb and the descent gate. ⭐ **This is R1's statistic**: it is
      what the read actually asserts, and it is the thing the refuted protocol
      had no way to express.
    """
    lat = z_or_latent if isinstance(z_or_latent, dict) else {"z": z_or_latent}
    z = np.asarray(lat["z"])
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
    if "pi" in lat:
        pi = np.asarray(lat["pi"])
        sets = [np.flatnonzero(row >= float(gate)) for row in pi]
        sz = np.array([len(s) for s in sets], float)
        out.update({
            "distinct_wells_gated": float(sz.mean()),
            "ge_F_distinct_gated": float((sz >= F).mean()),
            "occupancy_precision_gated": float(np.mean(
                [np.isin(s, np.asarray(A)).mean() if len(s) else 0.0
                 for s, A in zip(sets, subsets, strict=True)])),
            "coverage_gated": float(np.mean(
                [np.isin(np.asarray(A), s).all() for s, A in zip(sets, subsets, strict=True)])),
            "exact_set_occupancy_gated": float(np.mean(
                [set(s.tolist()) == set(np.asarray(A).tolist())
                 for s, A in zip(sets, subsets, strict=True)])),
            "n_wells_ever_occupied_gated": int(len(np.unique(
                np.concatenate(sets)) ) if any(len(s) for s in sets) else 0),
            "pi_gate": float(gate)})
    return out


def s_effective(occ_wells: int, cfg: CatTestConfig) -> float:
    """``S_eff = K*F / #(wells ever occupied)``; registered band ``[S/2, S]``.

    prereg §6 rule 3: outside the band the run is reported **COLLAPSED**, never
    as a null (``orgdiv-cat-test`` §5.2 measured 34.1 / 51.2 / 36.6 vs a band of
    [8, 16] and had to report its arm collapsed).
    """
    return float(cfg.n_items * cfg.f_subset / max(int(occ_wells), 1))


# ==========================================================================
# (d) consolidate-to-budget + trash-region pruning
# ==========================================================================
def find_wells(store: FactoredStore, cfg: CatTestConfig, mw: MultiWellReadConfig,
               key) -> Tuple[np.ndarray, np.ndarray]:
    """Discover the store's **actual** minima by probe descent.

    Over-digging is allowed and expected: this returns whatever minima exist,
    including spurious shallow ones, with their depths. Probes are launched on
    the address shell with the payload block at 0 (the query manifold).
    """
    model = store.model(cfg)
    d, m = int(cfg.addr_dim), int(cfg.payload_dim)
    k_p, _ = jax.random.split(key)
    g = jax.random.normal(k_p, (int(mw.n_probes), d))
    g = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    q0 = jnp.concatenate([g * float(cfg.ball_radius),
                          jnp.zeros((int(mw.n_probes), m))], axis=1)
    mass = jnp.ones_like(q0)
    gam = jnp.full((int(mw.n_probes),), 0.2)
    q, _ = settle_particles(model, q0, jnp.zeros_like(q0), mass, gam,
                            int(mw.probe_steps), float(cfg.dt))
    q = np.asarray(q)
    # ⚠ §7.28 (program-wide ruler): the depth must be CONFINEMENT-SUBTRACTED, or
    # the bowl `alpha*||q||^2` alone reads as a well (1.44x inflation measured on
    # this very store). depth = sum_i A_i^2 exp(-r^2/2s^2) = -(V - alpha ||q||^2).
    V = np.asarray(jax.vmap(store.V)(jnp.asarray(q)))
    depth = -(V - float(cfg.confine) * (q ** 2).sum(-1))
    return q, depth


def consolidate_wells(centers: np.ndarray, depths: np.ndarray, *, budget: int,
                      merge_radius: float, trash_depth_frac: float
                      ) -> Dict[str, Any]:
    """Merge over-dug minima to the designed budget; TRASH the spurious ones.

    Mechanical, measurable criteria only (charter §A20.3(d)):

    1. **merge** — single-linkage at ``merge_radius``; a cluster's centre is its
       depth-weighted mean and its depth is its maximum (two probes in one basin
       are one well);
    2. **truncate to budget** — rank clusters by depth, keep the deepest
       ``budget``;
    3. **prune BELOW budget** — a *controller decision*: clusters shallower than
       ``trash_depth_frac * median(kept depth)`` are pruned even if the budget is
       not full.

    ⛔ Pruned clusters are routed to the trash region, **never merged into a
    meaningful well** — merging a spurious shallow well into a real one would
    move the real well's centre, which is exactly the corruption the trash region
    exists to avoid.
    """
    c = np.asarray(centers, dtype=np.float64)
    dep = np.asarray(depths, dtype=np.float64)
    order = np.argsort(-dep)
    reps: list = []
    members: list = []
    for i in order:
        placed = False
        for r, mem in zip(reps, members, strict=True):
            if np.linalg.norm(c[i] - c[r]) <= merge_radius:
                mem.append(i)
                placed = True
                break
        if not placed:
            reps.append(i)
            members.append([i])
    cl_c = np.stack([(dep[m][:, None] * c[m]).sum(0) / max(dep[m].sum(), 1e-12)
                     for m in members])
    cl_d = np.array([dep[m].max() for m in members])
    cl_n = np.array([len(m) for m in members])
    rank = np.argsort(-cl_d)
    keep = rank[:int(budget)]
    over = rank[int(budget):]
    if len(keep):
        thresh = float(trash_depth_frac) * float(np.median(cl_d[keep]))
        shallow = keep[cl_d[keep] < thresh]
        keep = keep[cl_d[keep] >= thresh]
    else:
        shallow, thresh = np.array([], dtype=int), 0.0
    trashed = np.concatenate([over, shallow]).astype(int)
    return {"kept_centers": cl_c[keep], "kept_depths": cl_d[keep],
            "kept_sizes": cl_n[keep],
            "trashed_centers": cl_c[trashed], "trashed_depths": cl_d[trashed],
            "n_found": int(len(cl_d)), "n_kept": int(len(keep)),
            "n_trashed_over_budget": int(len(over)),
            "n_trashed_shallow": int(len(shallow)),
            "depth_threshold": float(thresh),
            "merge_radius": float(merge_radius)}


def trash_field(trashed_centers: np.ndarray, dim: int, *, radius: float,
                strength: float, gamma_max: float = 0.9,
                width: float = 0.15) -> Optional[FrictionField]:
    """The trash region ``gamma_phi(q)`` — :mod:`chlu.core.friction_field`'s FIRST USE.

    A frozen, compact-gated hole at every pruned well: a particle that falls into
    a trashed basin is damped to a stop and its payload block never reaches a
    written value, so the dedupe verb's descent weight discards it. Outside the
    horizons ``gamma_phi == 0`` **exactly** (``gate="compact"``), so the read is
    bit-identical to the no-trash path everywhere else (F5 Prop-11).
    """
    t = np.asarray(trashed_centers)
    if t.size == 0:
        return None
    if t.shape[1] != dim:
        pad = np.zeros((len(t), dim - t.shape[1]))
        t = np.concatenate([t, pad], axis=1)
    return FrictionField(dim, centers=jnp.asarray(t, jnp.float32),
                         gamma_max=float(gamma_max), width=float(width),
                         init_radius=float(radius), init_strength=float(strength),
                         trainable=False, gate="compact")


# ==========================================================================
# the reader class — the 4 shipped members + the non-quantising twin (D8)
# ==========================================================================
def soft_well_table_fit(latent: Dict[str, np.ndarray], y, *, well_payloads
                        ) -> Dict[str, Any]:
    """R5: ``yhat = W [pi @ V_table, 1]``. ``(m+1) x m`` = 72 fitted params.

    ⭐ The non-quantising twin of ``factored_store._well_table_fit``: identical
    parameter count, identical store dependence, but the assignment is the
    **soft** occupancy instead of an ``argmax``. Keeping BOTH in the class is what
    makes "quantisation is the destructive step" a measurement rather than a
    belief — ``OD_min`` is the min over both.
    """
    f = np.asarray(latent["pi"]) @ np.asarray(well_payloads)
    X = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    w, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return {"kind": "soft_well_table", "w": w, "well_payloads": well_payloads,
            "n_params": int(w.size)}


def soft_well_table_apply(mdl, latent):
    f = np.asarray(latent["pi"]) @ np.asarray(mdl["well_payloads"])
    X = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    return X @ mdl["w"]


READERS_MW = ("sum_linear", "well_table", "knn", "mlp", "soft_well_table")

# ==========================================================================
# ⭐ the ZERO-PARAMETER identity twins (C2W7 reconciliation 1, `reader-fitting-audit`)
# ==========================================================================
# ⛔ Every fitted member above is fitted by **least squares** while the metric is a
# **thresholded** exact-set accuracy; ``c2w7-read-cardinality`` §4 measured a cell
# where that combination scores a 72-parameter reader at 0.0000 on a latent a
# **zero-parameter** reader decodes at 0.0539. The twins below are the unfitted
# forms of the two store-dependent members, so a table can carry fitted AND
# identity columns and nothing is quietly re-based. ⛔ ``READERS_MW`` — the default
# ``which=`` of :func:`fit_readers_mw` — is **unchanged**, so every prior code path
# stays bit-identical.
#
# ⚠ Their standing assumption: the latent is **already in the target's units**.
# ``soft_well_identity`` inherits whatever mass ``pi`` carries (it sums to the
# number of occupied wells, not to ``F``), so it is exact only when the read
# commits to the right cardinality; ``gated_well_identity`` is exact only when the
# gated set is exactly ``A(x)``. Both have no gain with which to fix a scale error.

#: the zero-parameter twins added to the multi-well class.
READERS_MW_IDENTITY = ("soft_well_identity", "gated_well_identity",
                       "well_identity", "sum_identity")

#: the shipped class **plus** its identity twins (the audit's scored class).
READERS_MW_PLUS_IDENTITY = READERS_MW + READERS_MW_IDENTITY


def soft_well_identity_fit(latent: Dict[str, np.ndarray], y, *, well_payloads
                           ) -> Dict[str, Any]:
    """``yhat = pi @ V_table`` — the **0-parameter** twin of ``soft_well_table``."""
    del latent, y
    return {"kind": "soft_well_identity",
            "well_payloads": np.asarray(well_payloads), "n_params": 0}


def soft_well_identity_apply(mdl, latent) -> np.ndarray:
    return np.asarray(latent["pi"]) @ np.asarray(mdl["well_payloads"])


def gated_well_identity_fit(latent: Dict[str, np.ndarray], y, *, well_payloads,
                            gate: float = 0.5) -> Dict[str, Any]:
    """⭐ ``yhat = sum_{j : pi_j >= gate} v_j`` — **0 parameters**.

    The reader of the set the read actually **asserts** — i.e. of R1's own
    statistic (:func:`read_stats`' ``exact_set_occupancy_gated``), and the direct
    analogue of ``multiplicity_read.count_identity``, which is the member that
    exposed the fitting pathology in the first place. Its accuracy is bounded above
    by the gated exact-set occupancy: with payloads on a sphere of radius ``R`` a
    single wrong well costs ``~sqrt(2) R`` of residual against ``tol = 0.25 RMS``,
    i.e. ~3x tol, so it cannot absorb even one substitution.
    """
    del latent, y
    return {"kind": "gated_well_identity",
            "well_payloads": np.asarray(well_payloads), "gate": float(gate),
            "n_params": 0}


def gated_well_identity_apply(mdl, latent) -> np.ndarray:
    pi = np.asarray(latent["pi"])
    return (pi >= float(mdl["gate"])).astype(np.float64) \
        @ np.asarray(mdl["well_payloads"])


def fit_readers_mw(latent_seen: Dict[str, np.ndarray], y_seen, *, anchors,
                   well_payloads, seed: int = 0,
                   which: Sequence[str] = READERS_MW) -> Dict[str, Any]:
    """Fit the reader class on the **SEEN split only** (``PREREG-TierII.md`` §0).

    ⛔ ``which`` defaults to the **shipped** class; the zero-parameter twins of
    ``READERS_MW_IDENTITY`` are opt-in (``which=READERS_MW_PLUS_IDENTITY``) and are
    *added* to the class, never substituted for it.
    """
    from chlu.core.factored_store import fit_readers
    from chlu.core.null_arms import sum_identity_fit, well_identity_fit

    special = ("soft_well_table",) + READERS_MW_IDENTITY
    base = [w for w in which if w not in special]
    out = fit_readers(latent_seen["z"], y_seen, anchors=anchors,
                      well_payloads=well_payloads, seed=seed, which=base)
    if "soft_well_table" in which:
        out["soft_well_table"] = soft_well_table_fit(
            latent_seen, y_seen, well_payloads=well_payloads)
    if "soft_well_identity" in which:
        out["soft_well_identity"] = soft_well_identity_fit(
            latent_seen, y_seen, well_payloads=well_payloads)
    if "gated_well_identity" in which:
        out["gated_well_identity"] = gated_well_identity_fit(
            latent_seen, y_seen, well_payloads=well_payloads)
    if "well_identity" in which:
        out["well_identity"] = well_identity_fit(
            latent_seen["z"], y_seen, anchors=anchors,
            well_payloads=well_payloads)
    if "sum_identity" in which:
        out["sum_identity"] = sum_identity_fit(
            latent_seen["z"], y_seen, addr_dim=int(np.asarray(anchors).shape[1]))
    return out


def apply_reader_mw(mdl, latent: Dict[str, np.ndarray]) -> np.ndarray:
    from chlu.core.null_arms import apply_reader_plus_identity

    if mdl["kind"] == "soft_well_table":
        return soft_well_table_apply(mdl, latent)
    if mdl["kind"] == "soft_well_identity":
        return soft_well_identity_apply(mdl, latent)
    if mdl["kind"] == "gated_well_identity":
        return gated_well_identity_apply(mdl, latent)
    return apply_reader_plus_identity(mdl, latent["z"])


def score_readers_mw(readers, latent, y, tol) -> Dict[str, float]:
    return {k: exact_set_accuracy(apply_reader_mw(v, latent), y, float(tol))
            for k, v in readers.items()}


# ==========================================================================
# training — the soft-occupancy signal, and the STAGED ordering (guards 2, 3)
# ==========================================================================
def read_loss(store: FactoredStore, head: LaunchHead, phi: FrozenPhi,
              cfg: CatTestConfig, mw: MultiWellReadConfig, ind, Y, key, *,
              wells=None, well_payloads=None, hard: bool = False,
              w_occ: float = 1.0, w_pay: float = 1.0, detach_w: bool = False,
              address_steps: Optional[int] = None,
              read_steps: Optional[int] = None):
    """The read objective, differentiable through the settle. Two channels:

    * ``w_pay`` — the **continuous** channel, ``|| sum_i w_i pay(z_i) - y ||^2``
      (what ``sum_linear`` reads);
    * ``w_occ`` — ⭐ the **soft-occupancy** channel, ``|| pi @ V_table - y ||^2``
      (what ``soft_well_table`` reads). This is the training signal guard 2 is
      about: with ``hard=True`` the soft assignment is replaced by an ``argmax``
      one-hot, whose derivative is identically zero, and nothing upstream of the
      assignment can train at all.
    """
    wells = head.anchors if wells is None else jnp.asarray(wells, jnp.float32)
    d = int(cfg.addr_dim)
    a_steps = int(cfg.address_steps if address_steps is None else address_steps)
    r_steps = int(cfg.read_steps if read_steps is None else read_steps)
    model = store.model(cfg)
    c = query_code(phi, ind, key, float(cfg.query_sigma))
    q0, p0, mass, gmult, _conf = jax.vmap(head)(c)

    def run(q, p, mm, gm):
        q, p = settle_particles(model, q, p, mm, gm * float(cfg.gamma_address),
                                a_steps, float(cfg.dt))
        q, _ = settle_particles(model, q, jnp.zeros_like(p), mm,
                                gm * float(cfg.gamma_read), r_steps, float(cfg.dt))
        return q

    z = jax.vmap(run)(q0, p0, mass, gmult)
    w = (descent_weight(z[..., d:], mw.payload_ref) if mw.descent_gate
         else jnp.ones(z.shape[:2]))
    loss = 0.0
    if float(w_pay) != 0.0:
        y_pay = (z[..., d:] * w[..., None]).sum(axis=1)
        loss = loss + float(w_pay) * jnp.mean(jnp.sum((y_pay - Y) ** 2, -1))
    if float(w_occ) != 0.0 and well_payloads is not None:
        s = soft_occupancy(z[..., :d], wells, mw.occ_tau)
        if hard:
            s = jax.nn.one_hot(jnp.argmax(s, axis=-1), wells.shape[0])
        if detach_w:
            # ⛔ guard 2's isolation: with the magnitude channel detached, the
            # ONLY path left to the parameters is the assignment itself. Soft =>
            # O(1); hard => EXACTLY zero (argmax has no derivative).
            w = jax.lax.stop_gradient(w)
        pi = aggregate_occupancy(s, w, mw.dedupe)
        y_occ = pi @ jnp.asarray(well_payloads, jnp.float32)
        loss = loss + float(w_occ) * jnp.mean(jnp.sum((y_occ - Y) ** 2, -1))
    return loss


def train_launch_head(store: FactoredStore, head: LaunchHead, phi: FrozenPhi,
                      cfg: CatTestConfig, mw: MultiWellReadConfig, family, key,
                      *, wells=None) -> Tuple[LaunchHead, Dict[str, Any]]:
    """⭐ **Guard 3: the STAGED ordering.** Store first, launch head second.

    w20's lesson ("free learning erases design") and ``orgdiv-cat-test`` §5.1's
    measurement (gradients ``1e-10`` until wells exist, ``O(1)`` after) both say
    the same thing: the head cannot be trained against a store that has not been
    written. This function is therefore only ever called on a **written** store,
    and :func:`staging_gradient_probe` measures what happens if it is not.
    """
    import optax

    ind_all = jnp.asarray(family.indicator(family.seen, cfg.n_wells))
    Y_all = jnp.asarray(family.y_seen, jnp.float32)
    spec = head_trainable_spec(head)
    params, static = eqx.partition(head, spec)
    opt = optax.adam(float(mw.head_lr))
    st = opt.init(params)
    n = int(family.seen.shape[0])

    @eqx.filter_jit
    def _step(params, st, idx, k):
        def loss_fn(p):
            return read_loss(store, eqx.combine(p, static), phi, cfg, mw,
                             ind_all[idx], Y_all[idx], k, wells=wells,
                             well_payloads=family.payloads,
                             address_steps=mw.head_settle_address,
                             read_steps=mw.head_settle_read)

        val, g = eqx.filter_value_and_grad(loss_fn)(params)
        u, st = opt.update(g, st, params)
        return eqx.apply_updates(params, u), st, val

    hist = []
    for _ in range(int(mw.head_steps)):
        key, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.choice(k1, n, (min(int(mw.head_batch), n),), replace=False)
        params, st, val = _step(params, st, idx, k2)
        hist.append(float(val))
    return eqx.combine(params, static), {
        "head_loss": hist, "head_loss_first": hist[0] if hist else float("nan"),
        "head_loss_last": hist[-1] if hist else float("nan"),
        "head_settle_budget": [int(mw.head_settle_address), int(mw.head_settle_read)],
        "head_steps": int(mw.head_steps)}


def organize_store_mw(store: FactoredStore, head: LaunchHead, phi: FrozenPhi,
                      cfg: CatTestConfig, mw: MultiWellReadConfig, family, key,
                      *, wells=None, steps: int = 60, lr: float = 3e-3
                      ) -> Tuple[FactoredStore, Dict[str, Any]]:
    """⭐ The PHYSICS organizer: the store trained **through the settle**.

    This is the whole tier-ii distinction. The forward pass is the multi-well
    read itself, so every gradient reaching an atom has passed through the damped
    Verlet dynamics; the organizer-swap null (``null_arms.n1_gradient_placed``)
    optimises the *same parameters* against the *same objective* with a **static**
    softmax read and no dynamics anywhere.

    ⚠ Read-budget-scoped (C2W4 standing): training runs at the reduced
    ``head_settle_*`` budget, reported beside every number it produces.
    ⚠ The null gets MORE optimiser steps than this arm (its step is ~100x
    cheaper) — declared, and the direction is conservative for a physics claim.
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
            return read_loss(s2, head, phi, cfg, mw, ind[idx], Y[idx], k,
                             wells=wells, well_payloads=family.payloads,
                             address_steps=mw.head_settle_address,
                             read_steps=mw.head_settle_read)

        val, g = eqx.filter_value_and_grad(loss_fn)(params)
        u, st = opt.update(g, st, params)
        return eqx.apply_updates(params, u), st, val

    hist = []
    for _ in range(int(steps)):
        key, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.choice(k1, n, (min(int(mw.head_batch), n),), replace=False)
        params, st, val = _step(params, st, idx, k2)
        hist.append(float(val))
    store = eqx.tree_at(lambda s: s.V, store, eqx.combine(params, static))
    return store, {"organize_loss": hist,
                   "organize_loss_first": hist[0] if hist else float("nan"),
                   "organize_loss_last": hist[-1] if hist else float("nan"),
                   "organize_steps": int(steps), "organize_lr": float(lr),
                   "organize_settle_budget": [int(mw.head_settle_address),
                                              int(mw.head_settle_read)]}


def hard_vs_soft_gradient(store: FactoredStore, head: LaunchHead, phi: FrozenPhi,
                          cfg: CatTestConfig, mw: MultiWellReadConfig, family,
                          key, *, n: int = 8) -> Dict[str, float]:
    """⛔ **Guard 2's designed negative.** Hard assignments do not backprop.

    Returns ``||dL/dhead||`` under the soft occupancy and under a hard one-hot
    assignment, and their ratio. The guard fires if the ratio is not ~0.
    """
    ind = jnp.asarray(family.indicator(family.seen[:n], cfg.n_wells))
    Y = jnp.asarray(family.y_seen[:n], jnp.float32)
    spec = head_trainable_spec(head)
    params, static = eqx.partition(head, spec)

    def norm(hard):
        def loss_fn(p):
            # ⛔ the OCCUPANCY channel only (w_pay = 0): the guard is about the
            # assignment's differentiability, not about the payload channel that
            # would mask it.
            return read_loss(store, eqx.combine(p, static), phi, cfg, mw, ind, Y,
                             key, hard=hard, w_occ=1.0, w_pay=0.0,
                             detach_w=True, well_payloads=family.payloads,
                             address_steps=mw.head_settle_address,
                             read_steps=mw.head_settle_read)

        g = eqx.filter_grad(loss_fn)(params)
        leaves = [x for x in jax.tree_util.tree_leaves(g) if x is not None]
        return float(jnp.sqrt(sum(jnp.sum(jnp.asarray(x) ** 2) for x in leaves)))

    soft = norm(False)
    hard = norm(True)
    return {"grad_soft": soft, "grad_hard": hard,
            "ratio_hard_over_soft": hard / max(soft, 1e-30)}


def staging_gradient_probe(blank: FactoredStore, written: FactoredStore,
                           head: LaunchHead, phi: FrozenPhi, cfg: CatTestConfig,
                           mw: MultiWellReadConfig, family, key, *, n: int = 8
                           ) -> Dict[str, float]:
    """⛔ **Guard 3's designed negative.** Un-staged co-training has no gradient.

    ``||dL/dhead||`` and ``||dL/dstore||`` on a **blank** (unwritten, flat-init)
    store vs on the **written** one. ``orgdiv-cat-test`` §5.1 measured the store
    side at ``1e-10 -> O(1)``; this reproduces it for the head as well, which is
    what makes "store first, launch head second" a measurement.
    """
    ind = jnp.asarray(family.indicator(family.seen[:n], cfg.n_wells))
    Y = jnp.asarray(family.y_seen[:n], jnp.float32)
    hspec = head_trainable_spec(head)
    hp, hs = eqx.partition(head, hspec)

    def norms(store):
        def loss_head(p):
            return read_loss(store, eqx.combine(p, hs), phi, cfg, mw, ind, Y, key,
                             well_payloads=family.payloads,
                             address_steps=mw.head_settle_address,
                             read_steps=mw.head_settle_read)

        gh = eqx.filter_grad(loss_head)(hp)
        nh = float(jnp.sqrt(sum(jnp.sum(jnp.asarray(x) ** 2)
                                for x in jax.tree_util.tree_leaves(gh))))
        vspec = jax.tree_util.tree_map(eqx.is_inexact_array, store.V)
        vp, vs = eqx.partition(store.V, vspec)

        def loss_store(p):
            s2 = eqx.tree_at(lambda t: t.V, store, eqx.combine(p, vs))
            return read_loss(s2, head, phi, cfg, mw, ind, Y, key,
                             well_payloads=family.payloads,
                             address_steps=mw.head_settle_address,
                             read_steps=mw.head_settle_read)

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
# ledger — ⛔ k is CAPACITY and is on the byte ledger of EVERY arm (guard 4)
# ==========================================================================
def mwr_ledger(arm: str, cfg: CatTestConfig, mw: MultiWellReadConfig, *,
               store_params: int, head: LaunchHead, phi_bytes: int,
               reader_params: int = 0, **extra) -> Dict[str, Any]:
    """The row an arm may not be scored without.

    ⛔ ``k_particles`` is listed explicitly: ``k`` is capacity (it multiplies both
    the read's compute and the number of wells the latent can express), so a
    hidden ``k`` is ledger drift. :func:`assert_k_matched` is the blocking check.
    """
    read_flops = int(mw.k_particles * (cfg.address_steps + cfg.read_steps)
                     * cfg.n_atoms * (cfg.dim + 1))
    return {"arm": arm, "k_particles": int(mw.k_particles),
            "store_params": int(store_params), "store_bytes": int(store_params) * 4,
            "head_params": int(head.n_params()), "head_bytes": int(head.n_params()) * 4,
            "phi_bytes": int(phi_bytes), "reader_params": int(reader_params),
            "total_bytes": int(store_params) * 4 + int(head.n_params()) * 4
            + int(phi_bytes),
            "read_flops_per_query": read_flops, **extra}


def assert_k_matched(ledgers: Sequence[Dict[str, Any]]) -> int:
    """⛔ Blocking: every scored arm carries the SAME ``k``. Guard 4."""
    ks = {int(r["k_particles"]) for r in ledgers}
    if len(ks) != 1:
        raise ValueError(
            f"k is capacity and must be matched across arms; got {sorted(ks)} "
            f"from arms {[r['arm'] for r in ledgers]} (guard 4, charter §A20.3(c))")
    return ks.pop()
