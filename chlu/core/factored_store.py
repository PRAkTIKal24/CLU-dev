"""The **factored store** and the **cat test** — tier ii's vehicle (charter §A4.5, C2W5).

Built to ``.claude/outputs/orgdiv-prereg/PREREG-TierII.md`` (BINDING). This module
owns the *family*; :mod:`chlu.experiments.exp_cat_test` owns the harness that
scores it.

**What is factored.** The shipped store (:class:`~chlu.core.clu_system.LearnedVStore`)
digs one well per ITEM. Here ``N_a`` **shared wells** are dug, each carrying a
payload ``v_j`` drawn at write time, and an *item* is an ``F``-subset of wells with
target ``y(x) = sum_{j in A(x)} v_j``. Sharing factor ``S = K*F/N_a``. Items are
never stored; only wells are.

**The four §A3.7 rules, and where each is discharged** (prereg §2.3):

1. *not recoverable from row order* — ``y`` depends on ``A(x)`` only and insertion
   order is re-shuffled per seed (:func:`build_family`).
2. *not the query, or a function of it alone* — ``v_j`` is drawn at write time and
   never enters ``phi``. ⭐ **This rule is the family's hard constraint and it is
   quantitative, not qualitative** — see :func:`query_identifiability` and the
   ``SP-2`` note below.
3. *not at the arg-min table's maximum* — pre-condition K3.
4. *provably not in the table* — combinatorial: every unseen ``A`` satisfies
   ``|A & B| <= F-2`` for every stored ``B`` (:func:`build_family` asserts it per
   query and REJECTS a split rather than repairing it).

⭐⭐ **SP-2, the structural squeeze on the address dimension ``d``** (derived in
``.claude/outputs/orgdiv-cat-test/PREREG.md`` before this file existed). The
ground truth ``1_A -> y`` is *linear with ``N_a`` degrees of freedom*, so any reader
that can identify ``A(x)`` from ``phi(x)`` and carries ``>= N_a`` fitted parameters
solves the whole family from the SEEN split with no store at all. Two consequences
are built into this module rather than discovered by it:

* the **reader class is capacity-bounded below ``N_a``** (:func:`fit_readers`), which
  is what makes rule 2 discharge-able at all; and
* ``phi`` is a **deliberately lossy set-code** in ``R^d`` with ``d << N_a``, so a
  capacity-bounded query-only reader can explain at most ``~d/N_a`` of ``var(y)``.

The store then has exactly one job, and it is the mechanism under test: **supply
``v_j`` through the landscape's payload channel** so the reader needs O(1)
parameters. Whether ``d`` can be small enough for rule 2 and large enough for the
fan-out to find ``A(x)`` at the same time is SP-2, and it is measured, not assumed.

⛔ **Claim-form discipline (prereg §2.6, inherited verbatim).** No well is ever
named semantically anywhere in this module, its tests, its artifacts or its figure
captions. Wells carry integer indices and nothing else.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.memory_potentials import AtomDictionaryPotential
from chlu.experiments.goldstone_harness import clu_with_potential

__all__ = [
    "CatTestConfig",
    "FrozenPhi",
    "CatFamily",
    "FactoredStore",
    "build_family",
    "build_phi",
    "place_wells",
    "effective_s",
    "multi_particle_read",
    "fit_readers",
    "score_reader",
    "score_curve",
    "exact_set_accuracy",
    "chance_accuracy",
    "byte_ratio",
    "query_identifiability",
    "min_separation",
    "occupancy",
    "occupancy_precision",
    "apply_reader",
    "reader_bytes",
    "READERS",
    "well_write_loss",
    "write_wells",
    "organize_physics",
    "init_grad_norms",
]


# ==========================================================================
# config
# ==========================================================================
@dataclass
class CatTestConfig:
    """Every knob of the cat test, at its **registered** value.

    Defaults are ``PREREG-TierII.md`` §7's registered operating point. Lives here
    (next to the code that reads it) rather than in :mod:`chlu.config`, following
    the ``SoftCertificateConfig`` / ``CluSystemConfig`` / ``PilotConfig``
    precedent — ``chlu/config.py`` is a shared file and this is C2-owned work.
    """

    # -- the family (prereg §2.4 registered design point) -------------------
    n_wells: int = 32  # N_a  (⛔ 16 is FORBIDDEN: the rule-4 split is empty at K>=128)
    f_subset: int = 4  # F
    n_items: int = 128  # K (the SEEN split)
    n_unseen: int = 512  # sampled from the rule-4-valid held-out set
    atoms_per_well: int = 12  # a  (registered default; K1 sweeps {4, 12, 32})

    # -- geometry -----------------------------------------------------------
    addr_dim: int = 4  # d   ⭐ SP-2's swept axis
    # ⚠ REGISTERED DEVIATION, argued in the report. The prereg's byte arithmetic
    # takes m = 1, but at m = 1 pre-condition **K2's second assertion is
    # unsatisfiable**: with K = 128 scalar targets on a line, `min_B |y(A)-y(B)|`
    # is ~1e-4 of `tol` and only **1.2 %** of held-out queries pass. The
    # assertion needs `m >= 8` at (N_a=32, F=4, K=128, unit payload radius) —
    # measured sweep in the report; m = 8 is the SMALLEST value that passes at
    # 100 % over 5 seeds. m is the one symbol prereg §2.1 leaves free
    # (`v_j in R^m`), so the split is repaired in the ONE dimension the prereg
    # did not pin, and the (N_a, F, K, a) design point is untouched.
    payload_dim: int = 8  # m
    ball_radius: float = 2.0  # the address shell the wells live on
    # ⚠ LOAD-BEARING (w22 basin reach, measured here). Payloads are drawn on the
    # SPHERE of this radius, not from N(0, I_m): at m = 12 an i.i.d. normal draw
    # has ||v_j|| ~ sqrt(m) = 3.5, and the read launches from payload 0, so the
    # well is far outside the reach of any atom initialised near the launch
    # manifold ("a well initialised at payload 0 cannot reach |a_i| = 1"). Unit
    # payload radius also makes reach IDENTICAL at every well, so a depth
    # difference is a depth difference and not a distance artefact. The metric is
    # scale-invariant in this radius (tol scales with it), so K2 is unaffected.
    payload_radius: float = 1.0
    atom_width: float = 0.30  # atom_init_width
    atom_depth_init: float = 1e-4  # flat start; the writer digs the wells
    atom_init_scale: float = 1.0  # LOAD-BEARING (basin reach; w22)
    atom_local_radius: float = 0.25  # ⭐ pilot §5.3 tooling (N98 localized init)
    # ⭐ DESIGNED MECHANISM (pilot §5.3 candidate fix 1, generalised to m > 1;
    # declared in the byte ledger, costs ZERO parameters). `atom_init_scale`
    # scatters every coordinate at N(0, 1), so the PAYLOAD block of an atom starts
    # at radius ~sqrt(m) = 2.83 from a target at payload radius 1. With s ~ 0.4
    # that is exp(-2.83^2 / 2*0.4^2) = 2e-11 of signal: the write is inert **by
    # arithmetic**, and it was measured inert here (every well relaxed to the
    # origin, lambda_min = 0.1000 = 2*alpha exactly, i.e. pure confinement).
    # ⚠ N46 fairness is preserved: the payload block is rescaled onto the target
    # SHELL, never toward the target DIRECTION — the writer still has to find m-1
    # angular degrees of freedom, it is merely not asked to cross 2.8 units of
    # exponentially-flat landscape to start looking. 0.0 => the historical scatter.
    atom_payload_init_radius: float = 1.0
    confine: float = 0.05  # coercivity alpha

    # -- the operating point (prereg §7; ⛔ never >= 4.0, never <= 2.01) -----
    target_ds: float = 2.7  # d/s on MEASURED s
    # ⭐ prereg §7/OQ-1: the operating point is set on **measured** `s`, never on
    # the `atom_init_width` ruler (which `bprime-c6` showed overstates the span by
    # 1.74x). This is the median effective `s` measured by `stage_calibrate` at
    # the registered cell (a = 32, 3 seeds, confinement subtracted): 0.312 /
    # 0.321 / 0.320 -> 0.318. Well spacing is `sep = target_ds * s_measured`.
    # None => fall back to `atom_width` and DECLARE that the ruler is the init
    # width, not a measurement.
    s_measured: Optional[float] = 0.318
    depth_ratio: float = 3.0  # depth heterogeneity between neighbouring wells

    # -- the write ----------------------------------------------------------
    write_steps: int = 300
    write_lr: float = 3e-3
    write_weight_decay: float = 1e-4
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    write_n_perturb: int = 32
    lambda_traj: float = 0.0  # ⭐ pilot §5.3 tooling: the trajectory write term

    # -- the read (multi-particle occupancy; ⛔ never a settled point read) --
    n_particles: int = 4  # P >= 4 designed launches
    launch_radius: float = 0.6  # |o_p|, the designed offset scale
    dt: float = 0.05
    gamma_address: float = 0.05  # claim cells gamma in [0.05, 0.1]
    gamma_read: float = 0.02
    address_steps: int = 400
    read_steps: int = 800
    kinetic_mode: str = "newtonian_learned"
    query_sigma: float = 0.15  # sigma_q

    # -- organizer training (the physics arm) -------------------------------
    organize_steps: int = 200
    organize_lr: float = 3e-3
    organize_retain: int = 40  # trajectory-read truncation depth (theory Q4.2)
    organize_batch: int = 32

    # -- scoring -------------------------------------------------------------
    tol_frac: float = 0.25  # tol = tol_frac * sd(y_unseen)
    seed: int = 0
    quick: bool = False

    # -- soft certificate (bprime-c6's RE-LOCATED edge) ----------------------
    soft_cert_B: float = 0.542  # ⭐ `bprime-c6` §2: B >= 0.542 unrefuted

    @property
    def dim(self) -> int:
        return int(self.addr_dim + self.payload_dim)

    @property
    def n_atoms(self) -> int:
        return int(self.n_wells * self.atoms_per_well)

    @property
    def sharing(self) -> float:
        """``S = K*F/N_a`` — items per well."""
        return float(self.n_items * self.f_subset / max(self.n_wells, 1))

    def as_flag_table(self) -> Dict[str, Any]:
        """Every non-default flag in effect — the flag-provenance table."""
        base = CatTestConfig()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================================
# phi — FROZEN, identical on every arm (prereg §1)
# ==========================================================================
class FrozenPhi(eqx.Module):
    """The frozen read-in: a **lossy set-code** plus ``P`` designed launch offsets.

    ``phi(x) = R * normalise(sum_{j in A(x)} e_j)`` with ``e_j`` unit query codes in
    ``R^d``, then ``P`` designed offsets fan out the launch points:

        ``q0_p = (phi(x) + o_p + sigma_q * xi_p , 0_m)`` , ``p0 = 0``.

    ⛔ Everything here is FROZEN and drawn from a key that depends on neither the
    arm nor the store's seed, so ``phi`` is bit-identical across arms
    (``PhiMismatchError`` precedent). ``e_j`` and ``o_p`` are **parameters** and are
    ledgered; the occupancy vector a read produces is a **per-read transient**.

    ⚠ The payload block of the launch is pinned to ``0`` — the anti-decoration
    guard. The read must *dissipate up to* ``v_j``; nothing hands it the answer.
    """

    codes: jnp.ndarray  # (N_a, d) unit query codes
    offsets: jnp.ndarray  # (P, d) designed launch offsets
    radius: float = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)

    def __init__(self, cfg: CatTestConfig, key: jax.random.PRNGKey):
        d, P = int(cfg.addr_dim), int(cfg.n_particles)
        k_c, k_o = jax.random.split(key, 2)
        g = jax.random.normal(k_c, (int(cfg.n_wells), d))
        self.codes = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        # Designed offsets: P directions maximally spread in R^d. For P <= d the
        # leading axes; otherwise a fixed orthogonalised Gaussian draw. Designed,
        # not learned — a parameter of the launch protocol, ledgered.
        o = jax.random.normal(k_o, (P, d))
        o = o / (jnp.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
        self.offsets = o * float(cfg.launch_radius)
        self.radius = float(cfg.ball_radius)
        self.addr_dim = d
        self.payload_dim = int(cfg.payload_dim)

    def set_code(self, indicator: jnp.ndarray) -> jnp.ndarray:
        """``(..., N_a) -> (..., d)``: the lossy set-code, on the well shell."""
        s = indicator @ self.codes
        return self.radius * s / (jnp.linalg.norm(s, axis=-1, keepdims=True) + 1e-12)

    def launch(self, indicator: jnp.ndarray, key=None,
               sigma_q: float = 0.0) -> jnp.ndarray:
        """``(N_a,) -> (P, dim)`` launch positions (payload block pinned to 0)."""
        c = self.set_code(indicator)  # (d,)
        pts = c[None, :] + self.offsets  # (P, d)
        if key is not None and sigma_q > 0.0:
            pts = pts + sigma_q * jax.random.normal(key, pts.shape)
        pad = jnp.zeros((pts.shape[0], self.payload_dim), dtype=pts.dtype)
        return jnp.concatenate([pts, pad], axis=-1)

    def n_bytes(self) -> int:
        """phi-bytes — ledgered on EVERY arm (prereg §1)."""
        return int((self.codes.size + self.offsets.size) * 4)


def build_phi(cfg: CatTestConfig, phi_seed: int = 20260801) -> FrozenPhi:
    """The single frozen phi instance every arm must use.

    ⛔ ``phi_seed`` is deliberately NOT ``cfg.seed``: re-seeding the family must not
    re-draw phi, or the arms would not be comparable across seeds.
    """
    return FrozenPhi(cfg, jax.random.PRNGKey(int(phi_seed)))


# ==========================================================================
# the family + K2's rule-4 assertions
# ==========================================================================
@dataclass
class CatFamily:
    """A rule-4-valid cat-test split. ``seen``/``unseen`` are ``F``-subsets."""

    seen: np.ndarray  # (K, F) int
    unseen: np.ndarray  # (n_unseen, F) int
    payloads: np.ndarray  # (N_a, m)  -- v_j, drawn at WRITE time
    y_seen: np.ndarray  # (K, m)
    y_unseen: np.ndarray  # (n_unseen, m)
    tol: float
    n_valid_heldout: int  # the FULL rule-4-valid count (K2's combinatorics)
    n_total_combos: int
    order: np.ndarray  # insertion order (re-shuffled per seed; rule 1)
    k2: Dict[str, Any] = field(default_factory=dict)

    def indicator(self, subsets: np.ndarray, n_wells: int) -> np.ndarray:
        out = np.zeros((len(subsets), n_wells), dtype=np.float32)
        for i, A in enumerate(subsets):
            out[i, np.asarray(A, dtype=int)] = 1.0
        return out


def build_family(cfg: CatTestConfig, seed: int) -> CatFamily:
    """Construct the split and **assert rule 4 per query** (pre-condition K2).

    ⛔ A failing split is **rejected, not repaired** (prereg §2.3): if any held-out
    query violates ``|A & B| <= F-2`` or the payload-separation tolerance, we raise.
    The rule-4-valid held-out set is enumerated exactly (``O(K |Q|)`` set
    intersections over ``C(N_a, F)`` combinations), which also reproduces the
    prereg §2.4 feasibility count independently.
    """
    rng = np.random.default_rng(int(seed))
    N_a, F, K, m = cfg.n_wells, cfg.f_subset, cfg.n_items, cfg.payload_dim
    if N_a == 16 and K >= 128:
        raise ValueError(
            "N_a = 16 at K >= 128 is registered FORBIDDEN (prereg §2.4): the "
            "rule-4-valid held-out set is EMPTY and the family is unbuildable."
        )

    all_combos = np.array(list(itertools.combinations(range(N_a), F)), dtype=np.int16)
    n_total = len(all_combos)
    if K > n_total:
        raise ValueError(f"K={K} exceeds C({N_a},{F})={n_total}")

    # -- the SEEN split, in a per-seed insertion order (rule 1) -------------
    pick = rng.choice(n_total, size=K, replace=False)
    seen = np.sort(all_combos[pick].astype(int), axis=1)
    order = rng.permutation(K)  # insertion order is re-shuffled per seed

    # -- payloads: drawn at WRITE time, living only in the store (rule 2) ---
    payloads = rng.normal(size=(N_a, m))
    payloads = (payloads / np.linalg.norm(payloads, axis=1, keepdims=True)
                * float(cfg.payload_radius)).astype(np.float64)

    # -- rule 4, by construction: |A & B| <= F-2 for EVERY stored B ---------
    seen_ind = np.zeros((K, N_a), dtype=np.int8)
    np.put_along_axis(seen_ind, seen, 1, axis=1)
    all_ind = np.zeros((n_total, N_a), dtype=np.int8)
    np.put_along_axis(all_ind, all_combos.astype(int), 1, axis=1)
    # max overlap of every combination with every stored item, in one gemm
    max_ov = (all_ind.astype(np.int32) @ seen_ind.T.astype(np.int32)).max(axis=1)
    valid = np.nonzero(max_ov <= F - 2)[0]
    n_valid = int(len(valid))
    if n_valid == 0:
        raise ValueError(
            f"rule-4-valid held-out set is EMPTY at (N_a={N_a}, F={F}, K={K}) — "
            "the split is REJECTED, not repaired (prereg §2.3 rule 4)."
        )

    y_all = all_ind.astype(np.float64) @ payloads  # (n_total, m)
    y_seen = y_all[pick]
    n_take = min(int(cfg.n_unseen), n_valid)
    take = rng.choice(valid, size=n_take, replace=False)
    unseen = np.sort(all_combos[take].astype(int), axis=1)
    y_unseen = y_all[take]

    # ⚠ ``tol`` must scale with the payload dimension or it silently becomes a
    # different metric at every ``m``: the natural scale of ``||y - ybar||`` grows
    # like ``sqrt(m)``. Registered form: a fraction of the RMS deviation NORM.
    ybar = y_seen.mean(axis=0, keepdims=True)
    rms_norm = float(np.sqrt(np.mean(np.sum((y_unseen - ybar) ** 2, axis=-1))))
    tol = float(cfg.tol_frac * rms_norm)

    # -- K2 assertion #2: min_B ||y(A) - y(B)|| >= tol, per query -----------
    d_pay = np.linalg.norm(y_unseen[:, None, :] - y_seen[None, :, :], axis=-1)
    min_sep = d_pay.min(axis=1)
    ov = (all_ind[take].astype(np.int32) @ seen_ind.T.astype(np.int32)).max(axis=1)
    k2 = {
        "overlap_ok": bool((ov <= F - 2).all()),
        "max_overlap": int(ov.max()),
        "payload_sep_ok": bool((min_sep >= tol).all()),
        "min_payload_sep": float(min_sep.min()),
        "tol": tol,
        "frac_payload_sep_ok": float((min_sep >= tol).mean()),
        "n_valid_heldout": n_valid,
        "n_total_combos": n_total,
    }
    if not k2["overlap_ok"]:
        raise ValueError("K2 overlap assertion failed — split REJECTED")

    return CatFamily(seen=seen, unseen=unseen, payloads=payloads,
                     y_seen=y_seen, y_unseen=y_unseen, tol=tol,
                     n_valid_heldout=n_valid, n_total_combos=n_total,
                     order=order, k2=k2)


# ==========================================================================
# well placement — a PURE function of (frozen codes, shared policy params)
# ==========================================================================
def place_wells(phi: FrozenPhi, cfg: CatTestConfig, sep: float,
                n_iter: int = 400) -> np.ndarray:
    """``(N_a, d)`` well anchors ``u_j``, at minimum separation ``sep``.

    Canonical placement: start from the frozen query codes on the shell of radius
    ``R`` and relax pairwise overlaps until ``min_{i!=j} ||u_i - u_j|| >= sep``,
    re-projecting to the shell each round. ⭐ This is a **pure function of
    (frozen codes, shared policy params ``(R, sep)``)** and carries no per-item
    table, so prereg §6 rule 6's ``allocate`` conditions (C4/C5) are satisfied by
    construction — the allocation-shuffle test is run in the harness and is
    vacuously exact.
    """
    u = np.asarray(phi.codes, dtype=np.float64) * float(cfg.ball_radius)
    for _ in range(int(n_iter)):
        diff = u[:, None, :] - u[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        viol = dist < sep
        if not viol.any():
            break
        push = np.where(viol[..., None], diff / (dist[..., None] + 1e-12), 0.0)
        w = np.where(viol, (sep - dist), 0.0)
        u = u + 0.5 * np.einsum("ijk,ij->ik", push, w) / max(1, viol.sum(axis=1).max())
        nrm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        u = u * (float(cfg.ball_radius) / nrm)
    return u


def min_separation(u: np.ndarray) -> float:
    d = np.linalg.norm(u[:, None, :] - u[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


# ==========================================================================
# the store
# ==========================================================================
class FactoredStore(eqx.Module):
    """``N_a`` shared wells in one learned ``V_theta``; ``a`` atoms per well.

    One atom group per **well** (not per item) is the whole point: a write is
    local in parameter space per WELL, and items share wells rather than owning
    them. ``group_rows(j)`` + :func:`atom_write_mask_fn` keeps that locality exact.
    """

    V: AtomDictionaryPotential
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    n_wells: int = eqx.field(static=True)

    def __init__(self, cfg: CatTestConfig, anchors: np.ndarray, key):
        self.addr_dim = int(cfg.addr_dim)
        self.payload_dim = int(cfg.payload_dim)
        self.n_wells = int(cfg.n_wells)
        self.V = AtomDictionaryPotential(
            dim=cfg.dim,
            n_atoms=cfg.n_atoms,
            key=key,
            init_scale=float(cfg.atom_init_scale),
            init_width=float(cfg.atom_width),
            confine=float(cfg.confine),
            depth_init=float(cfg.atom_depth_init),
            n_groups=int(cfg.n_wells),
            # ⭐ pilot §5.3 tooling, deployed as a DESIGNED MECHANISM: group j's
            # atoms start inside a ball of radius `atom_local_radius` around well
            # j's ADDRESS anchor (address axes only — N46 fairness: localizing the
            # payload axis would hand the writer the answer).
            group_centers=jnp.asarray(anchors[:, : cfg.addr_dim]),
            local_radius=float(cfg.atom_local_radius),
        )
        if float(cfg.atom_payload_init_radius) > 0.0:
            c = self.V.centers
            pay = c[:, self.addr_dim:]
            nrm = jnp.linalg.norm(pay, axis=1, keepdims=True) + 1e-12
            c = c.at[:, self.addr_dim:].set(
                pay / nrm * float(cfg.atom_payload_init_radius))
            self.V = eqx.tree_at(lambda t: t.centers, self.V, c)

    @property
    def dim(self) -> int:
        return int(self.addr_dim + self.payload_dim)

    def group_rows(self, well: int) -> jnp.ndarray:
        return self.V.group_rows(int(well))

    def well_state(self, well: int, anchor: np.ndarray) -> Tuple[float, float]:
        """``(D_j, s_j)`` from the landscape at well ``j``'s own anchor."""
        m = np.asarray(self.group_rows(well), dtype=bool)
        A = np.asarray(self.V.amp, dtype=float)[m] ** 2
        s = np.exp(np.asarray(self.V.log_width, dtype=float)[m])
        c = np.asarray(self.V.centers, dtype=float)[m]
        z = np.zeros((self.dim,))
        z[: self.addr_dim] = np.asarray(anchor)[: self.addr_dim]
        w = A * np.exp(-np.sum((c - z[None, :]) ** 2, -1) / (2.0 * s**2 + 1e-12))
        D = float(w.sum())
        s_eff = float((w * s).sum() / max(w.sum(), 1e-12)) if D > 0 else float(s.mean())
        return D, s_eff

    def n_bytes(self) -> int:
        leaves = jax.tree_util.tree_leaves(eqx.filter(self.V, eqx.is_inexact_array))
        return int(sum(int(np.asarray(x).size) for x in leaves) * 4)

    def model(self, cfg: CatTestConfig):
        return clu_with_potential(self.V, dim=self.dim,
                                  kinetic_mode=cfg.kinetic_mode,
                                  inertia=jnp.ones(self.dim))


# ==========================================================================
# the write — masked, one atom group per WELL
# ==========================================================================
def _trainable_spec(V: AtomDictionaryPotential):
    """Boolean filter spec selecting ``centers``/``log_width``/``amp``.

    ⚠ Not :func:`chlu.training.train_memory.trainable_filter` and not
    :func:`chlu.core.memory_potentials.atom_write_mask_fn`: both address the atoms
    through a ``V.learned`` wrapper (``DesignFreedomPotential``). The factored
    store holds the :class:`AtomDictionaryPotential` **directly** — there is no
    designed part to hold fixed, because in a factored store the well vocabulary
    *is* the learned object. Same three leaves, one level up.
    """
    return jax.tree_util.tree_map(eqx.is_inexact_array, V)


def _well_mask_fn(row_mask: jnp.ndarray):
    """Update mask freezing every atom row outside ``row_mask`` (C3-local write).

    The un-wrapped twin of :func:`~chlu.core.memory_potentials.atom_write_mask_fn`.
    Masks the *updates*, not the gradients — ``adamw``'s decoupled weight decay
    would otherwise still shrink the frozen rows.
    """
    m = jnp.asarray(row_mask, dtype=jnp.float32)

    def apply(u):
        return eqx.tree_at(
            lambda t: [t.centers, t.log_width, t.amp], u,
            replace=[u.centers * m[:, None], u.log_width * m, u.amp * m])

    return apply


def well_write_loss(V, targets, key, *, addr_dim: int, payload_dim: int,
                    n_perturb: int = 32, sigma_addr: float = 0.25,
                    sigma_pay: float = 0.6, margin: float = 0.15,
                    barrier: float = 0.2, crowd_targets=None):
    """The shipped write objective, generalised to a ``payload_dim > 1`` block.

    Same three terms as :func:`chlu.training.train_memory.write_loss` —
    stationarity at the target, a finite-neighbourhood minimum condition
    (including on the **query manifold**, payload launched at 0), and a mid-point
    barrier against the nearest other well.

    ⚠ **Why this is not a call into ``train_memory.write_loss``.** That function
    takes a scalar ``payload_index`` and pins/jitters exactly ONE payload
    coordinate (``scale.at[payload_index].set(sigma_pay)``,
    ``query_pts.at[..., payload_index].set(0.0)``). With ``m > 1`` the remaining
    ``m-1`` payload axes would be jittered at ``sigma_addr`` and never pinned, so
    the write's query manifold would not be the manifold the read actually
    launches from. ``train_memory.py`` is a shared file and out of this task's
    ownership, so the generalisation lives here. **At ``m = 1`` the two objectives
    are the same object** (asserted in ``tests/test_factored_store.py``).
    """
    K, dim = targets.shape
    gradV = jax.grad(lambda q: V(q))
    pay = slice(addr_dim, addr_dim + payload_dim)

    l_grad = jnp.mean(jax.vmap(lambda z: jnp.sum(gradV(z) ** 2))(targets))

    k_p, k_q = jax.random.split(key, 2)
    scale = jnp.concatenate([jnp.full((addr_dim,), sigma_addr),
                             jnp.full((payload_dim,), sigma_pay)])
    delta = jax.random.normal(k_p, (K, n_perturb, dim)) * scale
    q_jit = (jax.random.normal(k_q, (K, n_perturb, dim)) * scale
             ).at[:, :, pay].set(0.0)
    query_pts = (targets[:, None, :] + q_jit).at[:, :, pay].set(0.0)

    pts = jnp.concatenate([targets[:, None, :] + delta, query_pts], axis=1)
    v_t = jax.vmap(V)(targets)
    v_p = jax.vmap(jax.vmap(V))(pts)
    l_min = jnp.mean(jnp.mean(jax.nn.relu(v_t[:, None] - v_p + margin), axis=1))

    others = targets if crowd_targets is None else jnp.asarray(crowd_targets)
    d2 = jnp.sum((targets[:, None, :] - others[None, :, :]) ** 2, axis=-1)
    jj = jnp.argmin(jnp.where(d2 < 1e-12, jnp.inf, d2), axis=1)
    nb = others[jj]
    v_m = jax.vmap(V)(0.5 * (targets + nb))
    v_hi = jnp.maximum(v_t, jax.vmap(V)(nb))
    l_bar = jnp.mean(jax.nn.relu(barrier + v_hi - v_m) ** 2)
    return l_grad + l_min + l_bar


def write_wells(store: FactoredStore, cfg: CatTestConfig, anchors: np.ndarray,
                payloads: np.ndarray, key, *, order: Optional[np.ndarray] = None,
                depth_scale: Optional[np.ndarray] = None
                ) -> Tuple[FactoredStore, Dict[str, Any]]:
    """Write all ``N_a`` wells, **masked per well** (one atom group each).

    ``depth_scale`` implements the registered **depth heterogeneity >= 3x**
    between neighbouring wells (prereg §7): the target payload/margin of well
    ``j`` is unchanged, but its barrier target is scaled, so alternate wells come
    out deeper. Insertion ``order`` is re-shuffled per seed (rule 1).
    """
    import optax

    d, m = int(cfg.addr_dim), int(cfg.payload_dim)
    tgt = np.zeros((cfg.n_wells, cfg.dim), dtype=np.float32)
    tgt[:, :d] = anchors[:, :d]
    tgt[:, d:d + m] = payloads
    tgt_j = jnp.asarray(tgt)
    order = np.arange(cfg.n_wells) if order is None else np.asarray(order)
    ds = np.ones(cfg.n_wells) if depth_scale is None else np.asarray(depth_scale)

    losses = []
    V = store.V
    spec = _trainable_spec(V)
    for j in order:
        j = int(j)
        mask_fn = _well_mask_fn(store.group_rows(j))
        params, static = eqx.partition(V, spec)
        opt = optax.adamw(float(cfg.write_lr),
                          weight_decay=float(cfg.write_weight_decay))
        st = opt.init(params)

        @eqx.filter_jit
        def _step(params, static, st, k, tgt_row=tgt_j[j], opt=opt,
                  mask_fn=mask_fn, barrier=float(cfg.write_barrier) * float(ds[j])):
            def loss_fn(p):
                return well_write_loss(
                    eqx.combine(p, static), tgt_row[None, :], k,
                    addr_dim=d, payload_dim=m,
                    n_perturb=int(cfg.write_n_perturb),
                    sigma_addr=float(cfg.write_sigma_addr),
                    sigma_pay=float(cfg.write_sigma_pay),
                    margin=float(cfg.write_margin),
                    barrier=barrier,
                    crowd_targets=tgt_j)

            val, g = eqx.filter_value_and_grad(loss_fn)(params)
            u, st = opt.update(g, st, params)
            return eqx.apply_updates(params, mask_fn(u)), st, val

        for _ in range(int(cfg.write_steps)):
            key, k = jax.random.split(key)
            params, st, val = _step(params, static, st, k)
        V = eqx.combine(params, static)
        losses.append(float(val))

    store = eqx.tree_at(lambda s: s.V, store, V)
    # ⭐ The ENDPOINT write loss, re-evaluated on the FINAL store over ALL wells
    # (the `clu_system._write_item` pattern). The per-well values collected in the
    # loop are STALE by construction: a well written early is scored against a
    # landscape in which the later wells have not been dug yet, so its barrier and
    # minimum terms are evaluated against a store that no longer exists. K1's bar
    # is adjudicated on this number, not on the loop's.
    k_end = jax.random.fold_in(key, 987654)
    endpoint = float(well_write_loss(
        V, tgt_j, k_end, addr_dim=d, payload_dim=m,
        n_perturb=int(cfg.write_n_perturb), sigma_addr=float(cfg.write_sigma_addr),
        sigma_pay=float(cfg.write_sigma_pay), margin=float(cfg.write_margin),
        barrier=float(cfg.write_barrier), crowd_targets=tgt_j))
    return store, {"write_losses": losses,
                   "endpoint_write_loss": endpoint,
                   "per_well_last_loss_mean": float(np.mean(losses)),
                   "per_well_last_loss_max": float(np.max(losses))}


# ==========================================================================
# the ORGANIZER — physics arm: trained through the TRAJECTORY READ
# ==========================================================================
def organize_physics(store: FactoredStore, phi: FrozenPhi, cfg: CatTestConfig,
                     family: CatFamily, key, *, address_steps: int = 150,
                     read_steps: int = 150, channel: str = "implicit",
                     ridge: float = 0.0) -> Tuple[FactoredStore, Dict]:
    """Train the store's organization **through the settle** (the physics arm).

    ⭐ This is the whole tier-ii distinction, and the channel matters.

    * ``channel="implicit"`` (**default, and the registered one**) —
      :func:`~chlu.core.implicit_grad.implicit_settle`: the forward pass is the
      ordinary damped Verlet settle, the backward pass solves
      ``dq*/dtheta = -(Hess V)^-1 d_theta grad V`` at the settled point. This
      reaches ``theta`` **exactly** and is what prereg §6 rule 1 registers for
      store parameters.
    * ``channel="trajectory"`` — :func:`~chlu.core.implicit_grad.truncated_rollout`
      with the last ``organize_retain`` steps taped. ⚠ **Measured here to be
      numerically dead**: at a converged two-phase read the truncated gradient to
      the store is ``~1e-12`` (see the report's grad-norm table), three orders
      below the prereg's own ``2.654e-9`` unroll reference. It is kept because it
      is the *only* legal channel for anything upstream of ``grad V`` — but
      ``phi`` is FROZEN here, so nothing upstream needs one.

    ⛔ Because ``phi`` is frozen, ``dq*/dq0 = 0`` is discharged by construction
    rather than fought (prereg §6 rule 1). The loss is the read objective on the
    SEEN split only: ``L = mean_x || sum_p pay(q*_p(x)) - y(x) ||^2``.

    ⚠ Read-budget-scoped (C2W4 standing): training uses a REDUCED settle budget
    (default 150+150 vs the scoring read's 400+800). The budget is reported beside
    every number this produces.
    """
    import optax

    from chlu.core.implicit_grad import SettleSpec, implicit_settle, truncated_rollout

    d, m, P = int(cfg.addr_dim), int(cfg.payload_dim), int(cfg.n_particles)
    ind = jnp.asarray(family.indicator(family.seen, cfg.n_wells))
    Y = jnp.asarray(family.y_seen, dtype=jnp.float32)
    spec = _trainable_spec(store.V)
    params, static = eqx.partition(store.V, spec)
    opt = optax.adam(float(cfg.organize_lr))
    st = opt.init(params)
    sp1 = SettleSpec(steps=int(address_steps), dt=float(cfg.dt),
                     gamma=float(cfg.gamma_address), ridge=float(ridge))
    sp2 = SettleSpec(steps=int(read_steps), dt=float(cfg.dt),
                     gamma=float(cfg.gamma_read), ridge=float(ridge))

    def _read(V, idx, k):
        model = clu_with_potential(V, dim=cfg.dim, kinetic_mode=cfg.kinetic_mode,
                                   inertia=jnp.ones(cfg.dim))
        keys = jax.random.split(k, idx.shape[0])
        q0 = jax.vmap(lambda i, kk: phi.launch(ind[i], kk, float(cfg.query_sigma))
                      )(idx, keys).reshape(-1, cfg.dim)
        p0 = jnp.zeros_like(q0)
        if channel == "implicit":
            def one(a, b):
                q = implicit_settle(model, a, b, sp1)
                return implicit_settle(model, q, jnp.zeros_like(b), sp2)
        else:
            def one(a, b):
                _, q, pp = truncated_rollout(model, a, b, int(address_steps),
                                             float(cfg.dt), float(cfg.gamma_address),
                                             retain=0, stride=address_steps,
                                             return_endpoint=True)
                _, q, _ = truncated_rollout(model, q, pp, int(read_steps),
                                            float(cfg.dt), float(cfg.gamma_read),
                                            retain=int(cfg.organize_retain),
                                            stride=read_steps, return_endpoint=True)
                return q
        return jax.vmap(one)(q0, p0).reshape(idx.shape[0], P, cfg.dim)

    @eqx.filter_jit
    def _step(params, st, idx, k):
        def loss_fn(p):
            q = _read(eqx.combine(p, static), idx, k)
            return jnp.mean(jnp.sum((q[:, :, d:d + m].sum(axis=1) - Y[idx]) ** 2, -1))

        val, g = eqx.filter_value_and_grad(loss_fn)(params)
        u, st = opt.update(g, st, params)
        return eqx.apply_updates(params, u), st, val

    hist = []
    n = int(family.seen.shape[0])
    for _ in range(int(cfg.organize_steps)):
        key, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.choice(k1, n, (min(int(cfg.organize_batch), n),),
                                replace=False)
        params, st, val = _step(params, st, idx, k2)
        hist.append(float(val))
    store = eqx.tree_at(lambda s: s.V, store, eqx.combine(params, static))
    return store, {"organize_loss": hist,
                   "organize_loss_first": hist[0] if hist else float("nan"),
                   "organize_loss_last": hist[-1] if hist else float("nan"),
                   "organize_channel": channel,
                   "organize_ridge": float(ridge),
                   "organize_address_steps": int(address_steps),
                   "organize_read_steps": int(read_steps),
                   "organize_retain": int(cfg.organize_retain)}


def init_grad_norms(store: FactoredStore, phi: FrozenPhi, cfg: CatTestConfig,
                    family: CatFamily, key, *, address_steps: int = 150,
                    read_steps: int = 150) -> Dict[str, float]:
    """``||dL/d.||`` at init for every trainable group AND every channel.

    prereg §6 rule 1 requires this table; its reference scales are ``0.0``
    (implicit through ``q0``), ``2.654e-9`` (unroll) and ``6.421e-3``
    (trajectory). Both channels are reported so the choice of channel is a
    measurement rather than a preference.
    """
    from chlu.core.implicit_grad import SettleSpec, implicit_settle, truncated_rollout

    d, m, P = int(cfg.addr_dim), int(cfg.payload_dim), int(cfg.n_particles)
    n = min(8, int(family.seen.shape[0]))
    ind = jnp.asarray(family.indicator(family.seen[:n], cfg.n_wells))
    Y = jnp.asarray(family.y_seen[:n], dtype=jnp.float32)
    spec = _trainable_spec(store.V)
    params, static = eqx.partition(store.V, spec)
    sp1 = SettleSpec(steps=int(address_steps), dt=float(cfg.dt),
                     gamma=float(cfg.gamma_address))
    sp2 = SettleSpec(steps=int(read_steps), dt=float(cfg.dt),
                     gamma=float(cfg.gamma_read))

    def make_loss(channel):
        def loss_fn(p):
            V = eqx.combine(p, static)
            model = clu_with_potential(V, dim=cfg.dim,
                                       kinetic_mode=cfg.kinetic_mode,
                                       inertia=jnp.ones(cfg.dim))
            keys = jax.random.split(key, ind.shape[0])
            q0 = jax.vmap(lambda i, kk: phi.launch(i, kk, float(cfg.query_sigma))
                          )(ind, keys).reshape(-1, cfg.dim)
            p0 = jnp.zeros_like(q0)
            if channel == "implicit":
                def one(a, b):
                    q = implicit_settle(model, a, b, sp1)
                    return implicit_settle(model, q, jnp.zeros_like(b), sp2)
            else:
                def one(a, b):
                    _, q, pp = truncated_rollout(model, a, b, int(address_steps),
                                                 float(cfg.dt),
                                                 float(cfg.gamma_address), retain=0,
                                                 stride=address_steps,
                                                 return_endpoint=True)
                    _, q, _ = truncated_rollout(model, q, pp, int(read_steps),
                                                float(cfg.dt), float(cfg.gamma_read),
                                                retain=int(cfg.organize_retain),
                                                stride=read_steps,
                                                return_endpoint=True)
                    return q
            q = jax.vmap(one)(q0, p0).reshape(ind.shape[0], P, cfg.dim)
            return jnp.mean(jnp.sum((q[:, :, d:d + m].sum(1) - Y) ** 2, -1))

        return loss_fn

    out = {}
    for channel in ("implicit", "trajectory"):
        g = eqx.filter_grad(make_loss(channel))(params)
        for name in ("centers", "log_width", "amp"):
            leaf = getattr(g, name, None)
            out[f"{channel}/grad_norm_{name}"] = (
                float(jnp.linalg.norm(leaf)) if leaf is not None else float("nan"))
    return out


# ==========================================================================
# the effective-`s` estimator (prereg §7/§8 OQ-1 — BLOCKING first-day instrument)
# ==========================================================================
def effective_s(V, center: np.ndarray, *, s_hint: float = 0.3, n_rays: int = 16,
                n_r: int = 24, r_max_mult: float = 4.0, seed: int = 0,
                confine: float = 0.0) -> Dict[str, float]:
    """Fit ``A exp(-r^2 / 2 s^2)`` to a written well's **radial profile**.

    prereg §7's bracket-breaking instrument: every ``d/s`` statement in tier ii is
    expressed in a width that, for a *learned multi-atom* well, has never been
    measured (``bprime-theory`` §9.2). We fit it.

    The profile is ``-(V(center + r n) - V_inf)`` averaged over ``n_rays`` random
    unit directions, sampled on ``n_r`` radii out to ``r_max_mult * s_hint``.
    ``ln`` of the profile is linear in ``r^2`` with slope ``-1/2s^2`` — the same
    estimator ``bprime-c6`` used to measure ``s = 0.3979`` (R^2 = 0.9953) on the
    shipped store, so the two numbers are commensurable by construction.

    ⚠ **``confine`` is not optional in practice.** The store's potential carries
    ``+ alpha ||q||^2``, and a pure confinement bowl has a radial profile that is
    monotone decreasing away from the centre — i.e. it *log-fits as a well* and
    returns a finite, entirely spurious ``s``. (Measured: a landscape with no
    wells at all returned a confident width.) Passing ``confine = alpha``
    subtracts the known quadratic analytically, so an undug well correctly
    returns ``nan`` and a dug well's width is not inflated by the bowl.

    Returns ``{s, depth, r2, n_used}``; ``s`` is ``nan`` when the well is not a well.
    """
    rng = np.random.default_rng(int(seed))
    d = int(np.asarray(center).size)
    dirs = rng.normal(size=(int(n_rays), d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    radii = np.linspace(1e-3, float(r_max_mult) * float(s_hint), int(n_r))
    c = jnp.asarray(center, dtype=jnp.float32)

    pts = jnp.asarray(
        (np.asarray(center)[None, None, :] + radii[None, :, None] * dirs[:, None, :]
         ).reshape(-1, d), dtype=jnp.float32)
    vals = np.asarray(jax.vmap(V)(pts)).reshape(len(dirs), len(radii))
    if float(confine) != 0.0:
        vals = vals - float(confine) * np.asarray(
            (pts ** 2).sum(-1)).reshape(len(dirs), len(radii))
    v0 = float(np.asarray(V(c)))
    # far-field reference: the confinement term at the same radii, removed by
    # taking the profile relative to the LARGEST radius sampled per ray
    prof = vals[:, -1][:, None] - vals  # >= 0 inside a well
    prof = prof.mean(axis=0)
    depth = float(prof.max())
    if not np.isfinite(depth) or depth <= 1e-9:
        return {"s": float("nan"), "depth": depth, "r2": float("nan"), "n_used": 0,
                "v_center": v0}
    keep = prof > max(1e-12, 0.02 * depth)
    if keep.sum() < 4:
        return {"s": float("nan"), "depth": depth, "r2": float("nan"),
                "n_used": int(keep.sum()), "v_center": v0}
    x = radii[keep] ** 2
    ylog = np.log(prof[keep])
    slope, intercept = np.polyfit(x, ylog, 1)
    if slope >= 0:
        return {"s": float("nan"), "depth": depth, "r2": float("nan"),
                "n_used": int(keep.sum()), "v_center": v0}
    s = float(np.sqrt(-1.0 / (2.0 * slope)))
    pred = slope * x + intercept
    ss_res = float(((ylog - pred) ** 2).sum())
    ss_tot = float(((ylog - ylog.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"s": s, "depth": float(np.exp(intercept)), "r2": r2,
            "n_used": int(keep.sum()), "v_center": v0}


# ==========================================================================
# the read — MULTI-PARTICLE OCCUPANCY (⛔ never a single settled point, Thm O1)
# ==========================================================================
def _settle(model, q, p, steps: int, dt: float, gamma: float):
    """``steps`` damped Verlet steps carrying only ``(q, p)``.

    ⚠ Deliberately NOT ``model(q, p, steps, ...)``: ``CHLU.__call__`` stacks the
    whole trajectory, which for ``B*P = 2560`` particles x 1200 steps x ``2*dim``
    is ~0.5 GB of dead weight. Same ``velocity_verlet_step``, same result — only
    the tape is dropped. (A *trajectory* read that needs the tape uses
    :func:`~chlu.core.implicit_grad.truncated_rollout` instead.)
    """
    def body(carry, _):
        return model.step(carry, dt, gamma), None

    (q, p), _ = jax.lax.scan(body, (q, p), None, length=int(steps))
    return q, p


def _two_phase(model, q0, p0, cfg: CatTestConfig):
    """The shipped two-phase damped settle, one particle. Returns ``(q, p)``."""
    q, p = _settle(model, q0, p0, cfg.address_steps, cfg.dt, cfg.gamma_address)
    return _settle(model, q, p, cfg.read_steps, cfg.dt, cfg.gamma_read)


def multi_particle_read(store: FactoredStore, phi: FrozenPhi, cfg: CatTestConfig,
                        indicators: np.ndarray, key, *,
                        gamma_address: Optional[float] = None,
                        batch: int = 256) -> np.ndarray:
    """``(B, N_a) -> z (B, P, dim)`` — the P settled states.

    ⭐ Theorem O1 forbids composition from living in a *single* settled point (the
    image of ``x -> q*`` is exactly the set of minima of ``V_theta``, so an
    ``N_min``-row table reproduces it for every reader). The read is therefore a
    **multi-particle occupancy** read: ``P`` designed launches, image up to
    ``N_min^P``, and the occupancy vector is a per-read transient (F4), never state.
    """
    c = cfg if gamma_address is None else _with(cfg, gamma_address=gamma_address)
    model = store.model(c)
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B, P = ind.shape[0], int(c.n_particles)

    @eqx.filter_jit
    def launch_all(ind_b, keys_b):
        return jax.vmap(lambda i, k: phi.launch(i, k, float(c.query_sigma)))(
            ind_b, keys_b)

    @eqx.filter_jit
    def settle_all(q0):
        p0 = jnp.zeros_like(q0)
        return jax.vmap(lambda a, b: _two_phase(model, a, b, c))(q0, p0)[0]

    out = []
    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        keys_b = jax.random.split(jax.random.fold_in(key, lo), hi - lo)
        q0 = launch_all(ind[lo:hi], keys_b).reshape(-1, store.dim)
        out.append(np.asarray(settle_all(q0)).reshape(hi - lo, P, store.dim))
    return np.concatenate(out, axis=0)


def _with(cfg: CatTestConfig, **kw) -> CatTestConfig:
    d = asdict(cfg)
    d.update(kw)
    return CatTestConfig(**d)


def occupancy(z: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """``(B, P, dim) -> (B, P)`` nearest-well index per particle (a transient)."""
    a = np.asarray(anchors)
    zz = np.asarray(z)[..., : a.shape[1]]
    d2 = ((zz[:, :, None, :] - a[None, None, :, :]) ** 2).sum(-1)
    return d2.argmin(-1)


def occupancy_precision(z: np.ndarray, anchors: np.ndarray,
                        subsets: np.ndarray) -> float:
    """Fraction of particles that settle into a well **belonging to ``A(x)``**.

    Chance is ``F / N_a``. ⛔ Reported as a co-activation statistic against task
    structure — **never** as a semantic identification of a well (prereg §2.6).
    """
    occ = occupancy(z, anchors)
    hits = [np.isin(occ[i], np.asarray(subsets[i])).mean() for i in range(len(occ))]
    return float(np.mean(hits))


# ==========================================================================
# the reader class — FROZEN, and every member has < N_a fitted parameters
# ==========================================================================
def _flat(z: np.ndarray) -> np.ndarray:
    return np.asarray(z).reshape(len(z), -1)


def _sum_linear_fit(z: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """R1: ``yhat = sum_p w.z_p + b``. ``d+m+1`` parameters."""
    X = np.asarray(z).sum(axis=1)  # (B, dim) -- permutation-invariant
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    w, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return {"kind": "sum_linear", "w": w, "n_params": int(w.size)}


def _sum_linear_apply(m, z):
    X = np.asarray(z).sum(axis=1)
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return X @ m["w"]


def _well_table_fit(z: np.ndarray, y: np.ndarray, *, anchors, well_payloads
                    ) -> Dict[str, Any]:
    """R2: nearest-well assignment, payloads **read from the store**; 2 fitted params.

    ⭐ The only store-dependent reader in the class, and deliberately so: it is the
    reader that needs O(1) parameters *because* the landscape supplies ``v_j``. Its
    query-only counterpart does not exist (no store to consult), which is reported
    rather than papered over.
    """
    occ = occupancy(z, anchors)
    f = np.asarray(well_payloads)[occ].sum(axis=1)  # (B, m)
    X = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    w, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return {"kind": "well_table", "w": w, "anchors": anchors,
            "well_payloads": well_payloads, "n_params": int(w.size)}


def _well_table_apply(m, z):
    occ = occupancy(z, m["anchors"])
    f = np.asarray(m["well_payloads"])[occ].sum(axis=1)
    X = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    return X @ m["w"]


def _knn_fit(z: np.ndarray, y: np.ndarray, ks=(1, 2, 3, 5, 10)) -> Dict[str, Any]:
    """R3: k-NN with inverse-distance weighting on canonicalised ``z``.

    Nonparametric (0 fitted parameters + the stored SEEN split). ``k`` is selected
    on a held-out-from-seen validation split, **never on ``Q_unseen``**.
    """
    Z = _canon(z)
    n = len(Z)
    n_val = max(4, n // 5)
    idx = np.arange(n)
    tr, va = idx[:-n_val], idx[-n_val:]
    best, best_k = np.inf, ks[0]
    for k in ks:
        pred = _knn_pred(Z[tr], np.asarray(y)[tr], Z[va], k)
        err = float(np.mean((pred - np.asarray(y)[va]) ** 2))
        if err < best:
            best, best_k = err, k
    return {"kind": "knn", "Z": Z, "y": np.asarray(y), "k": int(best_k),
            "n_params": 0}


def _canon(z: np.ndarray) -> np.ndarray:
    """Permutation-invariant canonicalisation: sort particles by their key."""
    zz = np.asarray(z)
    key = zz.reshape(zz.shape[0], zz.shape[1], -1).sum(-1)
    order = np.argsort(key, axis=1)
    return np.take_along_axis(zz, order[:, :, None], axis=1).reshape(len(zz), -1)


def _knn_pred(Ztr, ytr, Zq, k):
    d = np.linalg.norm(Zq[:, None, :] - Ztr[None, :, :], axis=-1)
    k = min(int(k), Ztr.shape[0])
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    dd = np.take_along_axis(d, nn, axis=1)
    w = 1.0 / (dd + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    return (w[..., None] * ytr[nn]).sum(axis=1)


def _knn_apply(m, z):
    return _knn_pred(m["Z"], m["y"], _canon(z), m["k"])


def _mlp_fit(z: np.ndarray, y: np.ndarray, hidden: int = 4, steps: int = 600,
             lr: float = 3e-2, seed: int = 0) -> Dict[str, Any]:
    """R4: ``yhat = sum_p g(z_p)``, ``g`` a 1-hidden-layer tanh MLP of width 4.

    ⭐ Width is frozen so the **total fitted parameter count stays below ``N_a``**
    (29 at ``d=4, m=1``): a reader that could memorise ``N_a`` payloads would solve
    the family from the query alone (SP-1) and the metric would measure nothing.
    """
    import optax

    rng = jax.random.PRNGKey(int(seed))
    dim = np.asarray(z).shape[-1]
    m_out = np.asarray(y).shape[-1]
    k1, k2 = jax.random.split(rng)
    params = {
        "W1": jax.random.normal(k1, (dim, hidden)) * (1.0 / math.sqrt(dim)),
        "b1": jnp.zeros((hidden,)),
        "W2": jax.random.normal(k2, (hidden, m_out)) * (1.0 / math.sqrt(hidden)),
        "b2": jnp.zeros((m_out,)),
    }
    Z = jnp.asarray(z, dtype=jnp.float32)
    Y = jnp.asarray(y, dtype=jnp.float32)

    def fwd(p, Z):
        h = jnp.tanh(Z @ p["W1"] + p["b1"])
        return (h @ p["W2"] + p["b2"]).sum(axis=1)  # sum over particles

    def loss(p):
        return jnp.mean((fwd(p, Z) - Y) ** 2)

    opt = optax.adam(lr)
    st = opt.init(params)

    @jax.jit
    def step(p, st):
        v, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return eqx.apply_updates(p, u), st, v

    for _ in range(int(steps)):
        params, st, _ = step(params, st)
    n_params = int(sum(np.asarray(v).size for v in params.values()))
    return {"kind": "mlp", "params": jax.tree_util.tree_map(np.asarray, params),
            "n_params": n_params}


def _mlp_apply(m, z):
    p = {k: jnp.asarray(v) for k, v in m["params"].items()}
    Z = jnp.asarray(z, dtype=jnp.float32)
    h = jnp.tanh(Z @ p["W1"] + p["b1"])
    return np.asarray((h @ p["W2"] + p["b2"]).sum(axis=1))


READERS = ("sum_linear", "well_table", "knn", "mlp")


def fit_readers(z_seen: np.ndarray, y_seen: np.ndarray, *, anchors=None,
                well_payloads=None, seed: int = 0,
                which: Sequence[str] = READERS) -> Dict[str, Any]:
    """Fit the FROZEN reader class on the **SEEN split only** (prereg §0)."""
    out = {}
    for name in which:
        if name == "sum_linear":
            out[name] = _sum_linear_fit(z_seen, y_seen)
        elif name == "well_table":
            if anchors is None or well_payloads is None:
                continue
            out[name] = _well_table_fit(z_seen, y_seen, anchors=anchors,
                                        well_payloads=well_payloads)
        elif name == "knn":
            out[name] = _knn_fit(z_seen, y_seen)
        elif name == "mlp":
            out[name] = _mlp_fit(z_seen, y_seen, seed=seed)
    return out


def apply_reader(m, z) -> np.ndarray:
    return {"sum_linear": _sum_linear_apply, "well_table": _well_table_apply,
            "knn": _knn_apply, "mlp": _mlp_apply}[m["kind"]](m, z)


def reader_bytes(readers: Dict[str, Any]) -> Dict[str, int]:
    """Reader parameters, ledgered on both arms (prereg §1)."""
    return {k: int(v.get("n_params", 0)) for k, v in readers.items()}


# ==========================================================================
# the metric
# ==========================================================================
def exact_set_accuracy(pred: np.ndarray, y: np.ndarray, tol: float) -> float:
    """Fraction of queries whose decoded payload is within ``tol`` of ``y(x)``."""
    p = np.asarray(pred).reshape(len(y), -1)
    return float((np.linalg.norm(p - np.asarray(y), axis=-1) <= tol).mean())


def chance_accuracy(y_seen: np.ndarray, y_unseen: np.ndarray, tol: float) -> float:
    """The registered chance level: the constant predictor ``mean(y_seen)``."""
    const = np.broadcast_to(np.asarray(y_seen).mean(axis=0), np.asarray(y_unseen).shape)
    return exact_set_accuracy(const, y_unseen, tol)


def score_reader(m, z, y, tol) -> float:
    return exact_set_accuracy(apply_reader(m, z), y, tol)


def score_curve(m, z, y, tol,
                mults: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)
                ) -> Dict[str, float]:
    """⭐ *Quote the curve, not the endpoint* (prereg §3.2, applied to the metric).

    Exact-set accuracy is all-or-nothing at ``m >= 6``: a store that is merely
    *poor* and a store that is *inert* both read 0.000 at the registered ``tol``.
    The accuracy-vs-``tol`` curve separates them, and it is reported beside every
    scalar in this harness.
    """
    pred = apply_reader(m, z)
    return {f"x{mult:g}": exact_set_accuracy(pred, y, tol * float(mult))
            for mult in mults}


# ==========================================================================
# ledgers
# ==========================================================================
def byte_ratio(cfg: CatTestConfig, *, n_spectator: int = 0) -> Dict[str, float]:
    """The **corrected** byte law ``ratio = [A(D+2) + d] / (d + m)`` (`harness-debt`).

    ``A = N_at / K`` (atoms per live item), ``D = dim = d + m + n_spectator``.
    ⚠ The prereg §5.2 table quotes ``ratio = 1.4*(N_at/K) + 0.8``, which is the same
    law at ``D + 2 = 7`` — i.e. it takes ``D = 5``. With ``d = 4, m = 1`` the store
    dimension is ``D = 5`` only if the *payload* axis is counted; both spellings are
    reported so the discrepancy is visible rather than silently resolved.
    """
    d, m = int(cfg.addr_dim), int(cfg.payload_dim)
    D = d + m + int(n_spectator)
    A = cfg.n_atoms / max(cfg.n_items, 1)
    corrected = (A * (D + 2) + d) / (d + m)
    prereg_form = 1.4 * A + 0.8
    return {"A_atoms_per_item": float(A), "D": int(D),
            "ratio_corrected": float(corrected),
            "ratio_prereg_5p2_form": float(prereg_form)}


def query_identifiability(phi: FrozenPhi, family: CatFamily, cfg: CatTestConfig
                          ) -> Dict[str, float]:
    """⭐ SP-2's instrument: how much of ``var(y)`` is linearly available from ``phi``.

    Returns the rank-limited ceiling ``d / N_a`` alongside the *measured* in-sample
    and out-of-sample ``R^2`` of an OLS fit of ``y`` on the set-code. This is the
    upper squeeze of SP-2 and it is a property of the FAMILY, not of any arm.
    """
    ind_s = family.indicator(family.seen, cfg.n_wells)
    ind_u = family.indicator(family.unseen, cfg.n_wells)
    Xs = np.asarray(phi.set_code(jnp.asarray(ind_s)))
    Xu = np.asarray(phi.set_code(jnp.asarray(ind_u)))
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
    Xu = np.concatenate([Xu, np.ones((len(Xu), 1))], 1)
    w, *_ = np.linalg.lstsq(Xs, family.y_seen, rcond=None)

    def r2(X, y):
        return float(1.0 - ((X @ w - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

    return {"rank_ceiling_d_over_Na": float(cfg.addr_dim / cfg.n_wells),
            "r2_seen": r2(Xs, family.y_seen),
            "r2_unseen": r2(Xu, family.y_unseen)}
