"""⭐⭐ **C2W11 spoke B — the physics organizer, the set-level read, the graded-
novelty channel and the anytime curve.**

This module produces the **PHYSICS side** of the wave's three VALUE legs. ⛔ It
adjudicates none of them: `OD`, `OD_min` and the organizer-swap verdict are
computed at the wave review from these artifacts and spoke C's (§A33.1). ⛔ No
paper number, no tier-ii verdict, no full-CLU verdict.

## What is here

* **the training regime (§A34.6, ratified)** — *the organization objective is
  LABEL-FREE; supervision enters ONLY at the READ HEAD.* Its measured basis is
  the pass-3 inversion: task supervision built the **worst** address geometry
  beyond 2 SE (``simclr - randconv`` A1 = ``-0.1406 +/- 0.0508``, 0/3 seeds;
  ``simclr - pca`` = ``-0.1276 +/- 0.0589``, 0/3) while unfitted ``randconv``
  bought the geometry for free ⇒ **a purely supervised organizer is the measured
  wrong bet, and the address block is its own head.**
* **the loss package** (a)-(e) — :mod:`chlu.training.losses`; ⛔ (f) kinetics is
  a declared NOT-RUN.
* **the set-level read** — :class:`chlu.core.psi_readout.ParticleSetPsi`
  (DeepSets pooled; ⛔ attention-psi is quarantined and is a declared NOT-RUN).
* **the graded-novelty read** — :mod:`chlu.core.novelty_read`.
* **the MECHANICS legs this spoke owns** — M3 (per-feature G-ADDR), M7 (the
  curvature-shape term) and M8 (the end-of-training curvature spectrum).
* **the C2W9 traversal trigger** (§7 of ``PREREG-C2W11.md``).

## ⛔ The things that are true here and easy to get wrong

* **The payload repair moved every y-scale.** ``tol = 0.286960063782279`` and
  ``chance_per_seed = [0.0, 0.001953125, 0.0]`` are read from
  :data:`FROZEN` (mirroring ``FROZEN-INTERFACES-C2W11.json``) and never
  hard-coded per cell; ``run1`` artifacts carry the PRE-repair ``0.47827``.
* **The G-DRIFT floor is a DIAGNOSTIC here, not a blocker** (charter ADDENDUM 13,
  Head ruling 2, citing §A13): §A13 explicitly permits table-like inference reads
  on **both** arms, which is the reframe the organizer swap exists for. The floor
  remains BLOCKING where it was built (C2W8's capture gate).
* **Wells are never named semantically** (``PREREG-TierII.md`` §2.6): wells,
  channels and features carry integer indices and nothing else.
* **Depth is not feature importance** (§A23.5 ACTIVE).
* Every ``gamma`` statement is **read-budget-scoped**; the anytime curve is
  quoted as a **curve**, never as an endpoint.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.core.factored_store import (
    CatFamily,
    CatTestConfig,
    FactoredStore,
    _settle,
    _trainable_spec,
    _with,
    build_family,
    build_phi,
    byte_ratio,
    chance_accuracy,
    exact_set_accuracy,
    fit_readers,
    occupancy,
    place_wells,
    reader_bytes,
    resolve_atom_width,
    store_population_spacing,
    write_store,
)
from chlu.core.feature_launch import (
    build_launch_head,
    launch_points,
    occupancy_precision_from_points,
)
from chlu.core.novelty_read import (
    NoveltyHead,
    auroc,
    collapse_statistic,
    ece,
    novelty_input,
    novelty_ledger,
    novelty_negatives_note,
    particle_descriptors,
    psi_input,
)
from chlu.core.psi_readout import ParticleSetPsi, set_psi_param_count
from chlu.core.soft_certificate import capture_radius
from chlu.experiments.exp_c2w11_substrate import (
    _depth_scale,
    _dist,
    _dump,
    _mean_2se,
    _relax_fn,
    _score_all,
    c2w11_config,
)
from chlu.training.losses import (
    C2W11LossCoeffs,
    band_hinge,
    cal_loss,
    org_loss,
    placed_sites,
    read_loss,
    reach_org_loss,
    shape_loss,
    share_loss,
    weak_org_loss,
)

__all__ = [
    "FROZEN",
    "OrganizerConfig",
    "load_frozen",
    "organizer_config",
    "build_organized_cell",
    "train_jig",
    "refine_store",
    "launch_and_settle",
    "fit_set_psi",
    "stage_launch_cap",
    "stage_k5_organized",
    "stage_false_positive_guards",
    "stage_v2",
    "stage_v3",
    "stage_m3",
    "stage_m7_m8",
    "stage_traversal",
    "run_c2w11_organizer",
]


# ==========================================================================
# ⛔ THE FROZEN INTERFACES — taken from the JSON, never re-derived
# ==========================================================================
#: A **mirror** of the fields of ``FROZEN-INTERFACES-C2W11.json`` this module
#: consumes. The JSON lives under ``.claude/`` (gitignored), so tracked code
#: carries the mirror and :func:`load_frozen` **asserts** the two agree whenever
#: the JSON is reachable — a mismatch raises rather than warns.
FROZEN: Dict[str, Any] = {
    "base": "main @ 2e1cdb2 (C2W8 close merged)",
    "family": {"N_a": 32, "F": 4, "K": 128, "m": 8, "a": 12, "d_addr": 4,
               "tol": 0.286960063782279,
               "chance_per_seed": [0.0, 0.001953125, 0.0],
               "chance": 0.0006510416666666666,
               "payload_radius": 0.6, "atom_payload_init_radius": 0.6,
               "s_measured_median": 0.3204208041192687},
    "selected_atom_width": {"atom_width_frac_spacing": 0.37},
    "launch_protocol": {"mode": "feature_factored", "k": 4, "sigma_q": 0.15,
                        "shell_radius": 2.0, "launch_key": 5000},
    "phi": {"phi_seed": 20260801, "phi_bytes": 576,
            "byte_hash": "a2713a0fb155e09f965cb6808720dbb1"},
    "reader_class": ["sum_linear", "well_table", "knn", "mlp",
                     "zero_parameter_identity"],
    "bound_Na_times_m": 256,
    "v3_budget_grid": {"points_total_verlet_steps": [50, 100, 200, 400, 800, 1200],
                       "particles_evolved": 4, "dt": 0.05,
                       "gamma_address": 0.05, "gamma_read": 0.02,
                       "split_rule": "address = round(b/3); read = b - round(b/3)"},
    "k8_structural_split": {"N_a": 32, "F": 4, "K": 24, "m": 8,
                            "sp1_exact_set_unseen_mean": 0.0625},
    "k4_full_psi_bar": 0.05,
    #: the MEASURED SC-6 capture radius at the selected payload radius (spoke A's
    #: payload-reach selection, seeds 100/101/102): 0.8867 / 0.8965 / 0.9414.
    #: ⛔ The jig bound is a fraction of THIS, never of sigma_q (PREREG §P4).
    "capture_radius_median_banked": 0.896484375,
}

#: ⛔ The registered abstain rule for K5 (``GATE-SEMANTICS-C2W11.json``): K5
#: abstains iff BOTH the per-item table AND the arm's best read score sit at or
#: below ``chance + delta``. ``delta`` is NUMERIC AND FIXED — never a judgement
#: call (Head condition (a), 2026-08-10). 0.01 ~ 5.12 metric grains at 1/512.
K5_DELTA = 0.01


def load_frozen(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``FROZEN-INTERFACES-C2W11.json`` when reachable and assert agreement.

    ⛔ The frozen family/launch/phi/reader/budget-grid data are the JSON's, not
    this module's. The mirror exists only so that tracked code does not depend on
    a gitignored file; when both are present they must agree **exactly**.
    """
    cand = [path] if path is not None else []
    env = os.environ.get("CHLU_C2W11_INTERFACES")
    if env:
        cand.append(Path(env))
    here = Path(__file__).resolve()
    for p in list(here.parents)[:5]:
        cand.append(p / ".claude" / "outputs" / "c2w11" /
                    "FROZEN-INTERFACES-C2W11.json")
        cand.append(p.parent / "CHLU" / ".claude" / "outputs" / "c2w11" /
                    "FROZEN-INTERFACES-C2W11.json")
    for p in cand:
        if p is not None and Path(p).is_file():
            j = json.loads(Path(p).read_text())
            fam = j["family"]
            mine = FROZEN["family"]
            for k in ("tol", "chance_per_seed", "payload_radius", "N_a", "F",
                      "K", "m", "a"):
                if fam[k] != mine[k]:
                    raise ValueError(
                        f"FROZEN mirror disagrees with the JSON on family.{k}: "
                        f"{mine[k]!r} != {fam[k]!r}. ⛔ The JSON is authoritative; "
                        "a pre-repair constant is exactly the trap this guards.")
            grid = j["v3_budget_grid"]["points_total_verlet_steps"]
            if grid != FROZEN["v3_budget_grid"]["points_total_verlet_steps"]:
                raise ValueError("⛔ the V3 budget grid disagrees with the JSON — "
                                 "a mismatched axis VOIDS VALUE leg iii")
            out = dict(FROZEN)
            out["source"] = str(p)
            return out
    out = dict(FROZEN)
    out["source"] = "inlined mirror (JSON not reachable from this checkout)"
    return out


# ==========================================================================
# the organizer's own configuration (⛔ chlu/config.py is untouched this wave)
# ==========================================================================
@dataclass(frozen=True)
class OrganizerConfig:
    """Spoke B's knobs. Lives here, next to the code that reads it, following the
    ``CatTestConfig`` / ``SoftCertificateConfig`` / ``PilotConfig`` precedent —
    ⛔ ``chlu/config.py`` is a shared file and **zero C2W11 spokes touch it**.

    ⭐ Every organizer coefficient defaults to **0.0** in
    :class:`~chlu.training.losses.C2W11LossCoeffs`, so ``OrganizerConfig()`` with
    the default coefficients reproduces spoke A's shipped arm bit-identically.
    """

    # -- stage 0: term (a), the jig ----------------------------------------
    jig_steps: int = 300
    jig_lr: float = 0.02
    jig_batch: int = 64
    jig_temperature: float = 0.2
    #: ⭐ term (a)'s instantiation. ``reach`` (registered AMENDMENT §A1) places the
    #: wells where the store's own launches can reach them; ``nt_xent`` is the
    #: theorist's InfoNCE on placed centroids, raced as the ablation.
    org_mode: str = "reach"
    reach_beta: float = 8.0
    reach_nu_sep: float = 1.0
    #: ⛔ the organizer may never move a well out of its own cue's reach: the
    #: per-well jig is clipped to this fraction of the MEASURED capture radius
    #: (the payload-reach trap in mirror image).
    jig_max_frac_capture: float = 0.75
    #: the label-free augmentation: resample this many of the F features
    view_resample: int = 1
    # -- stage 1: terms (b) and (c) on theta --------------------------------
    refine_steps: int = 120
    refine_lr: float = 3e-3
    shape_beta: float = 50.0
    shape_eps_soft: float = 0.005
    shape_stiff_target: float = 1.0
    shape_depth_min: float = 0.15
    shape_capture_dirs: int = 6
    #: ⛔ term (c)'s CAPTURE GUARD (the theorist's N-c3): softness bought by
    #: giving up the basin is NOT a pass. The hinge requires the inward radial
    #: force at ``sigma_q`` to exceed this margin. ⚠ Measured: at margin 0.0 the
    #: guard does NOT bind and (c) degrades per-feature G-ADDR 0.947 -> 0.421 —
    #: which is §A4.2's refutation reproduced on a DIFFERENT instantiation, and
    #: is reported two-sided rather than tuned away.
    shape_capture_margin: float = 0.05
    share_depth_target: float = 0.30
    share_log_ratio_max: float = 0.8  # ln(2.22): the MEASURED annihilation ratio
    # -- stage 2: psi and the novelty head ----------------------------------
    psi_hidden_grid: Tuple[int, ...] = (8, 16, 32)
    psi_hidden: int = 16
    psi_depth: int = 2
    psi_steps: int = 600
    psi_lr: float = 3e-3
    novelty_hidden: int = 16
    novelty_steps: int = 400
    novelty_lr: float = 3e-3
    p_drop: float = 0.25
    n_train_episodes: int = 3
    # -- instruments ---------------------------------------------------------
    m3_n_capture_wells: int = 16
    traversal_threshold: float = 0.20   # ⛔ registered in PREREG.md §P6
    reach_radius_frac_s: float = 2.0
    n_seeds_claim: int = 5

    def as_flag_table(self) -> Dict[str, Any]:
        d = asdict(self)
        base = asdict(OrganizerConfig())
        return {k: v for k, v in d.items() if v != base[k]}


def organizer_config(**kw) -> CatTestConfig:
    """The C2W11 **repaired** substrate at the frozen operating point.

    ⛔ Every value here is read from :data:`FROZEN` — the post-repair payload
    radius (0.60, with ``atom_payload_init_radius`` co-scaled), the SELECTED
    ``atom_width_frac_spacing = 0.37`` (the census refuses any other), the
    feature-factored launch and the placing write.
    """
    f = FROZEN
    base = dict(
        payload_radius=float(f["family"]["payload_radius"]),
        atom_payload_init_radius=float(f["family"]["atom_payload_init_radius"]),
        atom_width_frac_spacing=float(f["selected_atom_width"]["atom_width_frac_spacing"]),
        atom_width_selected_frac=float(f["selected_atom_width"]["atom_width_frac_spacing"]),
        atoms_per_well=int(f["family"]["a"]),
        payload_dim=int(f["family"]["m"]),
        n_wells=int(f["family"]["N_a"]),
        f_subset=int(f["family"]["F"]),
        n_items=int(f["family"]["K"]),
        query_sigma=float(f["launch_protocol"]["sigma_q"]),
    )
    base.update(kw)
    return c2w11_config(**base)


# ==========================================================================
# helpers
# ==========================================================================
class _JiggedCodes:
    """A code carrier for :func:`place_wells`. ⭐ With ``jig = 0`` the codes are
    unchanged **object-for-object**, so the placement — and therefore the whole
    store — is bit-identical to spoke A's ``build_arm``. That identity is the
    coefficient-zero control and it is pytest-asserted."""

    def __init__(self, codes, jig=None, radius: float = 2.0):
        c = np.asarray(codes, dtype=np.float64)
        self.codes = c if jig is None else c + np.asarray(jig, dtype=np.float64) / float(radius)


def _views(rng: np.random.Generator, subsets: np.ndarray, n_wells: int,
           n_resample: int) -> np.ndarray:
    """⭐ The **label-free** augmentation: resample ``n_resample`` of the ``F``
    features of each item. No target, no class, no payload is consulted."""
    out = np.array(subsets, copy=True)
    B, F = out.shape
    for i in range(B):
        pos = rng.choice(F, size=int(n_resample), replace=False)
        for p in pos:
            cand = rng.integers(0, n_wells)
            while cand in out[i]:
                cand = rng.integers(0, n_wells)
            out[i, p] = cand
    return out


def _indicator(subsets: np.ndarray, n_wells: int) -> np.ndarray:
    ind = np.zeros((len(subsets), int(n_wells)), dtype=np.float32)
    for i, s in enumerate(subsets):
        ind[i, np.asarray(s)] = 1.0
    return ind


def launch_and_settle(store: FactoredStore, head, cfg: CatTestConfig,
                      indicators: np.ndarray, key, *,
                      address_steps: Optional[int] = None,
                      read_steps: Optional[int] = None,
                      batch: int = 256, return_mid: bool = False):
    """``(B, N_a) -> (z, q0[, q_mid])``. The launch geometry is returned because
    every per-particle descriptor and every launder needs it.

    ⚠ **Read-budget-scoped** (C2W4 standing): ``address_steps``/``read_steps``
    default to the shipped 400+800 and are varied only by the V3 anytime curve,
    whose budget grid is frozen in ``FROZEN-INTERFACES-C2W11.json``.
    """
    a_s = int(cfg.address_steps if address_steps is None else address_steps)
    r_s = int(cfg.read_steps if read_steps is None else read_steps)
    q0 = launch_points(head, indicators, cfg, key)
    model = store.model(cfg)

    @eqx.filter_jit
    def go(pts):
        p0 = jnp.zeros_like(pts)
        qm, pm = jax.vmap(lambda a, b: _settle(model, a, b, a_s, cfg.dt,
                                               cfg.gamma_address))(pts, p0)
        q, _ = jax.vmap(lambda a, b: _settle(model, a, b, r_s, cfg.dt,
                                             cfg.gamma_read))(qm, jnp.zeros_like(pm))
        return q, qm

    zs, mids = [], []
    flat = jnp.asarray(q0.reshape(-1, store.dim), dtype=jnp.float32)
    for lo in range(0, flat.shape[0], int(batch) * int(cfg.n_particles)):
        hi = min(lo + int(batch) * int(cfg.n_particles), flat.shape[0])
        z, qm = go(flat[lo:hi])
        zs.append(np.asarray(z))
        mids.append(np.asarray(qm))
    z = np.concatenate(zs).reshape(q0.shape)
    if return_mid:
        return z, q0, np.concatenate(mids).reshape(q0.shape)
    return z, q0


# ==========================================================================
# ⭐ STAGE 0 — term (a): the LABEL-FREE organizer, through the write
# ==========================================================================
def train_jig(phi, cfg: CatTestConfig, family: CatFamily, coeffs: C2W11LossCoeffs,
              ocfg: OrganizerConfig, key, *, jig_max: float,
              spacing_ruler: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train the placement jig with **term (a)** — the DESIGNED write->phi channel.

    ⭐ **This is STAGE 0 and it is the only term that is live before the wells
    exist** (§1(a) of the loss package; banked: grad norms ``1e-10 - 1e-9`` at
    init through every other channel, ``O(1)`` only after the write). The channel
    is a *direct algebraic* path through the placing write's assignment
    ``u_j = R e_j + jig_j`` — neither implicit-at-settle nor trajectory.

    ⛔ **Honesty note, carried from the theorist verbatim in substance:** this is
    **not a physics gradient**. ``u = R e + jig`` is algebraic; the physics enters
    only through *what the objective measures* — an admissible, band-compliant,
    shareable placement. The correct statement of the physics-as-trainer bet is
    that the store's own admissibility functionals regularise the placement.
    """
    codes = jnp.asarray(np.asarray(phi.codes, dtype=np.float64), dtype=jnp.float32)
    R = float(cfg.ball_radius)
    if not (float(coeffs.lambda_org) != 0.0 or float(coeffs.lambda_weak) != 0.0):
        # ⛔ COEFFICIENT-ZERO: no optimiser, no RNG draw, no graph. The returned
        # jig is exactly zero, so the placement is bit-identical to spoke A's.
        return (np.zeros((int(cfg.n_wells), int(cfg.addr_dim))),
                {"organizer": "OFF (all organization coefficients 0)",
                 "steps": 0, "loss_first": float("nan"), "loss_last": float("nan"),
                 "jig_max": float(jig_max), "jig_norm_median": 0.0})
    rng = np.random.default_rng(int(np.asarray(jax.random.key_data(key))[-1]) & 0xFFFF)
    seen = np.asarray(family.seen)
    Y = jnp.asarray(family.y_seen, dtype=jnp.float32)
    # ⭐ the REACH instantiation needs the arm's OWN launch geometry, which is a
    # function of the frozen phi alone (no store, no target). Computed once.
    Q = None
    if str(ocfg.org_mode) == "reach":
        head = build_launch_head(phi, cfg)
        ind_all = jnp.asarray(_indicator(seen, cfg.n_wells))
        Q = jnp.asarray(np.asarray(launch_points(head, np.asarray(ind_all), cfg,
                                                 jax.random.fold_in(key, 77)))
                        [..., : int(cfg.addr_dim)], dtype=jnp.float32)
    jig = jnp.zeros((int(cfg.n_wells), int(cfg.addr_dim)))
    opt = optax.adam(float(ocfg.jig_lr))
    st = opt.init(jig)
    lam_org, lam_weak = float(coeffs.lambda_org), float(coeffs.lambda_weak)
    lam_band = float(coeffs.lambda_band)
    ruler = float(spacing_ruler)
    sep_target = float(cfg.target_ds) * ruler

    def total(j, ia, ib, yb, qb):
        out = 0.0
        if lam_org != 0.0:
            if str(ocfg.org_mode) == "reach":
                out = out + lam_org * reach_org_loss(
                    placed_sites(codes, j, R), qb, ia,
                    beta=float(ocfg.reach_beta), sep_target=float(sep_target),
                    nu_sep=float(ocfg.reach_nu_sep))
            else:
                out = out + lam_org * org_loss(
                    codes, j, R, ia, ib, temperature=float(ocfg.jig_temperature))
        if lam_band != 0.0:
            sites = placed_sites(codes, j, R)
            d2 = jnp.sum((sites[:, None, :] - sites[None, :, :]) ** 2, -1)
            d2 = d2 + 1e9 * jnp.eye(sites.shape[0])
            nn = jnp.sqrt(jnp.min(d2, axis=1))
            out = out + lam_band * jnp.mean(band_hinge(nn / ruler))
        if lam_weak != 0.0:
            out = out + lam_weak * weak_org_loss(codes, j, R, ia, yb,
                                                 temperature=float(ocfg.jig_temperature))
        return out

    @eqx.filter_jit
    def step(j, st, ia, ib, yb, qb):
        val, g = eqx.filter_value_and_grad(total)(j, ia, ib, yb, qb)
        u, st = opt.update(g, st, j)
        j = eqx.apply_updates(j, u)
        nrm = jnp.linalg.norm(j, axis=-1, keepdims=True)
        j = j * jnp.minimum(1.0, float(jig_max) / (nrm + 1e-12))  # ⛔ the reach bound
        return j, st, val

    hist = []
    n = len(seen)
    for _ in range(int(ocfg.jig_steps)):
        idx = rng.choice(n, size=min(int(ocfg.jig_batch), n), replace=False)
        sa = seen[idx]
        sb = _views(rng, sa, int(cfg.n_wells), int(ocfg.view_resample))
        ia = jnp.asarray(_indicator(sa, cfg.n_wells))
        ib = jnp.asarray(_indicator(sb, cfg.n_wells))
        qb = (Q[jnp.asarray(idx)] if Q is not None
              else jnp.zeros((len(idx), int(cfg.n_particles), int(cfg.addr_dim))))
        jig, st, val = step(jig, st, ia, ib, Y[jnp.asarray(idx)], qb)
        hist.append(float(val))
    j = np.asarray(jig, dtype=np.float64)
    return j, {"organizer": ("label-free REACH placement (AMENDMENT §A1)"
                            if str(ocfg.org_mode) == "reach" else
                            "label-free NT-Xent on placed set-code centroids")
                            + (" + WEAK-SUPERVISION coefficient" if lam_weak else ""),
               "steps": int(ocfg.jig_steps), "loss_first": hist[0],
               "loss_last": hist[-1], "jig_max": float(jig_max),
               "jig_norm_median": float(np.median(np.linalg.norm(j, axis=-1))),
               "jig_norm_max": float(np.max(np.linalg.norm(j, axis=-1))),
               "jig_bytes": int(j.size * 4), "org_mode": str(ocfg.org_mode),
               "coeffs": coeffs.as_flag_table()}


# ==========================================================================
# ⭐ STAGE 1 — terms (b) and (c) on theta
# ==========================================================================
def _site_spectra(V, sites: jnp.ndarray):
    H = jax.vmap(jax.hessian(lambda q: jnp.reshape(V(q), ())))(sites)
    H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    return jnp.linalg.eigvalsh(H)


def refine_store(store: FactoredStore, cfg: CatTestConfig, ocfg: OrganizerConfig,
                 targets: np.ndarray, coeffs: C2W11LossCoeffs, key
                 ) -> Tuple[FactoredStore, Dict[str, Any]]:
    """Terms **(b)** and **(c)** on ``theta``, at the written sites.

    ⛔ **COEFFICIENT-ZERO IS STRUCTURAL**: with ``lambda_share = lambda_shape =
    0`` this function returns the input store **unchanged** (no optimiser, no
    graph, no RNG), so the coefficient-zero arm is bit-identical to the shipped
    objective — the C2W2 invariant, pytest-asserted.

    ⚠ The capture hinge uses a **declared differentiable surrogate**: the inward
    radial force at ``sigma_q`` along ``shape_capture_dirs`` random directions
    (positive = restoring = captured). The **measured** SC-6 capture radius is
    the bisection instrument and is what M3/M7 report — a surrogate inside a loss
    is not permitted to become the instrument that scores it.
    """
    lam_b, lam_c = float(coeffs.lambda_share), float(coeffs.lambda_shape)
    if lam_b == 0.0 and lam_c == 0.0:
        return store, {"refine": "OFF (lambda_share = lambda_shape = 0)",
                       "steps": 0, "loss_first": float("nan"),
                       "loss_last": float("nan")}
    spec = _trainable_spec(store.V)
    params, static = eqx.partition(store.V, spec)
    opt = optax.adam(float(ocfg.refine_lr))
    st = opt.init(params)
    tg = jnp.asarray(targets, dtype=jnp.float32)
    kd = jax.random.normal(key, (int(ocfg.shape_capture_dirs), tg.shape[-1]))
    kd = kd / (jnp.linalg.norm(kd, axis=-1, keepdims=True) + 1e-12)
    rows = jnp.stack([jnp.asarray(store.group_rows(j), dtype=jnp.float32)
                      for j in range(int(cfg.n_wells))])  # (N_a, n_atoms)

    def loss_fn(p):
        V = eqx.combine(p, static)
        depth = rows @ (V.amp ** 2)                      # (N_a,) depth = sum amp^2
        out = 0.0
        if lam_b != 0.0:
            d2 = jnp.sum((tg[:, None, :] - tg[None, :, :]) ** 2, -1)
            d2 = d2 + 1e9 * jnp.eye(tg.shape[0])
            nb = jnp.argmin(d2, axis=1)
            out = out + lam_b * share_loss(
                depth, depth_target=float(ocfg.share_depth_target),
                neighbour_min=depth[nb], nu=1.0,
                log_ratio_max=float(ocfg.share_log_ratio_max))
        if lam_c != 0.0:
            lams = _site_spectra(V, tg)
            # the declared capture surrogate: inward radial force at sigma_q
            def inward(z):
                g = jax.vmap(jax.grad(lambda q: jnp.reshape(V(q), ())))(
                    z[None, :] + float(cfg.query_sigma) * kd)
                return jnp.mean(-jnp.sum(g * kd, axis=-1))
            cap = jax.vmap(inward)(tg)
            out = out + lam_c * shape_loss(
                lams, depth, cap, two_alpha=2.0 * float(cfg.confine),
                eps_soft=float(ocfg.shape_eps_soft),
                lambda_stiff_target=float(ocfg.shape_stiff_target),
                depth_min=float(ocfg.shape_depth_min),
                sigma_q=float(ocfg.shape_capture_margin),
                beta=float(ocfg.shape_beta))
        return out

    @eqx.filter_jit
    def step(p, st):
        val, g = eqx.filter_value_and_grad(loss_fn)(p)
        u, st = opt.update(g, st, p)
        return eqx.apply_updates(p, u), st, val

    hist = []
    for _ in range(int(ocfg.refine_steps)):
        params, st, val = step(params, st)
        hist.append(float(val))
    store = eqx.tree_at(lambda s: s.V, store, eqx.combine(params, static))
    return store, {"refine": "terms (b)+(c) on theta at the written sites",
                   "steps": int(ocfg.refine_steps), "loss_first": hist[0],
                   "loss_last": hist[-1], "coeffs": coeffs.as_flag_table()}


# ==========================================================================
# the cell: place -> organize -> write -> refine
# ==========================================================================
#: ⭐ the arms this spoke races. ⛔ ``coef0`` is the coefficient-zero CONTROL and
#: must be bit-identical to spoke A's shipped arm; ``weak`` is the wave's SINGLE
#: ablation and is ONE COEFFICIENT, not a new capability (§A34.6 / §3.1).
ARMS: Dict[str, C2W11LossCoeffs] = {
    "coef0": C2W11LossCoeffs(),
    "phys": C2W11LossCoeffs(lambda_org=1.0, lambda_band=0.1, lambda_share=0.0,
                            lambda_shape=1.0, lambda_read=1.0, lambda_cal=1.0),
    # ⭐ the DECOMPOSITION arms: (a) alone is the organizer proper; (c) alone
    # isolates the curvature-shape term. Without them a damaged number cannot be
    # attributed, and attribution is the whole point of a coefficient package.
    "org_only": C2W11LossCoeffs(lambda_org=1.0, lambda_band=0.1, lambda_read=1.0,
                                lambda_cal=1.0),
    "shape_only": C2W11LossCoeffs(lambda_shape=1.0, lambda_read=1.0,
                                  lambda_cal=1.0),
    "weak": C2W11LossCoeffs(lambda_org=1.0, lambda_band=0.1, lambda_share=0.0,
                            lambda_shape=1.0, lambda_read=1.0, lambda_cal=1.0,
                            lambda_weak=1.0),
}


def build_organized_cell(cfg: CatTestConfig, ocfg: OrganizerConfig,
                         family: CatFamily, seed: int, arm: str = "phys", *,
                         phi=None, capture_hint: Optional[float] = None
                         ) -> Dict[str, Any]:
    """One cell of the physics arm: **organize the placement, then write, then
    refine ``theta``** — the staged store-then-launch order (w20, mandatory)."""
    t0 = time.time()
    coeffs = ARMS[arm]
    phi = phi if phi is not None else build_phi(cfg)
    ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
    sep = float(cfg.target_ds * ruler)
    # the jig bound is a fraction of the MEASURED capture radius (PREREG §P4)
    jig_max = float(ocfg.jig_max_frac_capture) * float(
        capture_hint if capture_hint is not None
        else FROZEN["capture_radius_median_banked"])
    jig, org = train_jig(phi, cfg, family, coeffs, ocfg,
                         jax.random.PRNGKey(31000 + int(seed)),
                         jig_max=jig_max, spacing_ruler=float(ruler))
    anchors = place_wells(_JiggedCodes(phi.codes, jig, cfg.ball_radius), cfg, sep=sep)
    winfo = resolve_atom_width(cfg, anchors)
    k_init, k_write, k_ref = jax.random.split(jax.random.PRNGKey(int(seed)), 3)
    store = FactoredStore(cfg, anchors, k_init, atom_width=winfo["atom_width"])
    order = np.random.default_rng(int(seed)).permutation(cfg.n_wells)
    store, wrep = write_store(store, cfg, anchors, family.payloads, k_write,
                              order=order, depth_scale=_depth_scale(cfg),
                              atom_width=winfo["atom_width"])
    tgt = np.zeros((cfg.n_wells, cfg.dim), dtype=np.float32)
    tgt[:, :cfg.addr_dim] = anchors[:, :cfg.addr_dim]
    tgt[:, cfg.addr_dim:cfg.addr_dim + cfg.payload_dim] = family.payloads
    store, rrep = refine_store(store, cfg, ocfg, tgt, coeffs, k_ref)
    return {"store": store, "phi": phi, "anchors": anchors, "targets": tgt,
            "jig": jig, "arm": arm, "sep": sep, "width": winfo, "write": wrep,
            "organize": org, "refine": rrep, "coeffs": coeffs.as_flag_table(),
            "spacing": store_population_spacing(anchors),
            "head": build_launch_head(phi, cfg),
            "bytes": {"store": store.n_bytes(), "phi": phi.n_bytes(),
                      "jig_bytes": int(jig.size * 4),
                      "launch_head_bytes": 0, **byte_ratio(cfg)},
            "wall_s": round(time.time() - t0, 1)}


# ==========================================================================
# psi — the set-level read
# ==========================================================================
def fit_set_psi(u_seen: np.ndarray, w_seen: np.ndarray, y_seen: np.ndarray,
                ocfg: OrganizerConfig, key, *, hidden: Optional[int] = None,
                steps: Optional[int] = None) -> Tuple[ParticleSetPsi, Dict]:
    """Term **(d)** on ``psi``'s **direct** channel (O(1) from step 0 once the
    store exists). ⛔ The store-side implicit channel of (d) is measured and
    reported separately; see :func:`stage_false_positive_guards`."""
    h = int(ocfg.psi_hidden if hidden is None else hidden)
    psi = ParticleSetPsi(u_seen.shape[-1], y_seen.shape[-1], key, hidden=h,
                         depth=int(ocfg.psi_depth))
    opt = optax.adam(float(ocfg.psi_lr))
    st = opt.init(eqx.filter(psi, eqx.is_inexact_array))
    U = jnp.asarray(u_seen, dtype=jnp.float32)
    W = jnp.asarray(w_seen, dtype=jnp.float32)
    Y = jnp.asarray(y_seen, dtype=jnp.float32)

    @eqx.filter_jit
    def step(m, st):
        def lf(mm):
            return read_loss(mm(U, W), Y, weight=jnp.mean(W, axis=1))
        val, g = eqx.filter_value_and_grad(lf)(m)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, val

    hist = []
    for _ in range(int(steps if steps is not None else ocfg.psi_steps)):
        psi, st, val = step(psi, st)
        hist.append(float(val))
    return psi, {"psi_hidden": h, "psi_params": set_psi_param_count(psi),
                 "psi_loss_first": hist[0], "psi_loss_last": hist[-1],
                 "psi_steps": len(hist)}


def _psi_predict(psi: ParticleSetPsi, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(psi(jnp.asarray(u, dtype=jnp.float32),
                          jnp.asarray(w, dtype=jnp.float32)))


def _descriptors(store, z, q0, cfg):
    return particle_descriptors(store.V, z, q0, addr_dim=int(cfg.addr_dim),
                                confine=float(cfg.confine),
                                ball_radius=float(cfg.ball_radius))


def _capture_radii(store, cfg, targets, seed, n_wells: int) -> Dict[str, Any]:
    """The **measured** SC-6 capture radius per well (the bisection instrument)."""
    relax = _relax_fn(store, cfg)
    caps = [float(capture_radius(relax, targets[j], n_dirs=8, r_hi=1.0, steps=8,
                                 tol=0.15, seed=seed)["capture_radius"])
            for j in range(int(n_wells))]
    return {"capture_radii": caps, "capture_median": float(np.median(caps)),
            "capture_distribution": _dist(caps),
            "frac_capture_ge_sigma_q": float(np.mean(np.asarray(caps)
                                                     >= cfg.query_sigma)),
            "n_sites": int(n_wells),
            "instrument": "SC-6 bisection, 8 dirs, r_hi=1.0, tol=0.15"}


def _read_pack(cell, cfg, ocfg, ind, key, *, address_steps=None, read_steps=None):
    """Read -> descriptors -> psi inputs, in one place so no cell drifts."""
    z, q0 = launch_and_settle(cell["store"], cell["head"], cfg, ind, key,
                              address_steps=address_steps, read_steps=read_steps)
    d = _descriptors(cell["store"], z, q0, cfg)
    return {"z": z, "q0": q0, "desc": d, "u": psi_input(d),
            "w": np.asarray(d["capture_w"], dtype=np.float32),
            "n": novelty_input(d)}


def _score_pack(readers, psi, pack, y, tol) -> Dict[str, float]:
    """Every member of the frozen reader class **plus** the ADDED psi member."""
    out = _score_all(readers, pack["z"], y, tol)
    if psi is not None:
        out["deepsets_psi"] = exact_set_accuracy(
            _psi_predict(psi, pack["u"], pack["w"]), y, tol)
    return out


# ==========================================================================
# ⛔⛔ THE BLOCKING FIRST ACT — K5 RE-RUN ON THE **ORGANIZED** ARM
# ==========================================================================
def stage_k5_organized(cfg: CatTestConfig, ocfg: OrganizerConfig,
                       seeds: Sequence[int] = (0, 1, 2), arm: str = "phys",
                       out: Optional[Path] = None) -> Dict[str, Any]:
    """⛔⛔ **K5 on the organized arm — this spoke's FIRST act, and it BLOCKS.**

    K5 is defined on the physics arm **including its training signal**
    (``PREREG-TierII.md`` §4.1/§10). Spoke A ran with ``organizer NOT RUN``,
    which is why it **abstained**; ⛔ **the wave has produced NO K5 verdict** and
    this is the first one.

    **Registered abstain rule** (``delta`` fixed, never a judgement call): K5
    abstains iff **BOTH** the per-item table **AND** the arm's best read score are
    ``<= chance + delta``, ``delta = 0.01``. Otherwise K5 **scores**, and a
    failure is a real verdict: the read must beat the table by **> 0.10** on at
    least one reader.

    ⛔ If it abstains again on a trained arm, that is a **finding** — the
    organizer did not lift the read off the floor — and it is **not** a licence
    to proceed to the VALUE legs.
    """
    cells = []
    for seed in seeds:
        t0 = time.time()
        fam = build_family(cfg, seed=seed)
        cell = build_organized_cell(cfg, ocfg, fam, seed, arm)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        kr = jax.random.PRNGKey(7000 + int(seed))
        ps = _read_pack(cell, cfg, ocfg, ind_s, kr)
        pu = _read_pack(cell, cfg, ocfg, ind_u, jax.random.fold_in(kr, 1))
        readers = fit_readers(ps["z"], fam.y_seen, anchors=cell["anchors"],
                              well_payloads=fam.payloads, seed=seed)
        psi, prep = fit_set_psi(ps["u"], ps["w"], fam.y_seen, ocfg,
                               jax.random.PRNGKey(41000 + int(seed)))
        chance = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
        phys = _score_pack(readers, psi, pu, fam.y_unseen, fam.tol)

        # -- the per-item table launder, through the SAME reader class --------
        code_s = np.asarray(cell["phi"].set_code(jnp.asarray(ind_s)))
        code_u = np.asarray(cell["phi"].set_code(jnp.asarray(ind_u)))
        d_cs = np.linalg.norm(code_u[:, None, :] - code_s[None, :, :], axis=-1)
        nn = d_cs.argmin(1)
        nn_s = np.argmin(np.where(np.eye(len(code_s), dtype=bool), np.inf,
                                  ((code_s[:, None, :] - code_s[None, :, :]) ** 2
                                   ).sum(-1)), axis=1)
        rd_t = fit_readers(ps["z"][nn_s], fam.y_seen, anchors=cell["anchors"],
                           well_payloads=fam.payloads, seed=seed)
        psi_t, _ = fit_set_psi(ps["u"][nn_s], ps["w"][nn_s], fam.y_seen, ocfg,
                              jax.random.PRNGKey(41000 + int(seed)))
        tab = _score_all(rd_t, ps["z"][nn], fam.y_unseen, fam.tol)
        tab["deepsets_psi"] = exact_set_accuracy(
            _psi_predict(psi_t, ps["u"][nn], ps["w"][nn]), fam.y_unseen, fam.tol)

        margins = {k: float(phys[k] - tab.get(k, 0.0)) for k in tab}
        best_read = float(max(phys.values()))
        best_table = float(max(tab.values()))
        abstain = bool(best_read <= chance + K5_DELTA
                       and best_table <= chance + K5_DELTA)
        cells.append({
            "seed": int(seed), "arm": arm, "chance": chance, "tol": fam.tol,
            "physics_scores": phys, "table_scores": tab, "margins": margins,
            "best_margin": float(max(margins.values())),
            "best_read_score": best_read, "best_table_score": best_table,
            "delta": K5_DELTA, "abstains": abstain,
            "K5_PASS": bool(max(margins.values()) > 0.10) if not abstain else None,
            "reader_params": reader_bytes(readers), **prep,
            "organize": cell["organize"], "refine": cell["refine"],
            "bytes": cell["bytes"], "write": {
                k: cell["write"][k] for k in ("endpoint_write_loss",
                                              "grad_norm_at_targets")
                if k in cell["write"]},
            "wall_s": round(time.time() - t0, 1)})
        print(f"[K5-organized] seed={seed} arm={arm} chance={chance:.6f} "
              f"read={best_read:.5f} table={best_table:.5f} "
              f"margin={cells[-1]['best_margin']:+.5f} abstain={abstain} "
              f"({time.time()-t0:.0f}s)", flush=True)
    any_abstain = all(c["abstains"] for c in cells)
    res = {"stage": "k5_organized", "arm": arm, "cells": cells,
           "delta": K5_DELTA,
           "abstains": bool(any_abstain),
           "K5_PASS": (None if any_abstain else
                       bool(all(c["K5_PASS"] for c in cells))),
           "best_read": _mean_2se([c["best_read_score"] for c in cells]),
           "best_table": _mean_2se([c["best_table_score"] for c in cells]),
           "best_margin": _mean_2se([c["best_margin"] for c in cells]),
           "rule": ("abstains iff BOTH the per-item table AND the best read are "
                    "<= chance + 0.01; otherwise K5 SCORES and the bar is a "
                    "> 0.10 margin on >= 1 reader"),
           "consequence_if_abstains": (
               "⛔ the organizer did not lift the read off the floor. That is a "
               "FINDING, and it is NOT a licence to proceed to the VALUE legs.")}
    if out:
        _dump(res, out / "stage_k5_organized.json")
    return res


# ==========================================================================
# ⭐⭐ THE LAUNCH-PRECISION CAP — the wave's central MECHANICS question
# ==========================================================================
def _oracle_launch(cfg: CatTestConfig, anchors: np.ndarray, subsets: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Launch the ``k`` particles at the **needed** wells' address anchors.

    ⛔ **A DIAGNOSTIC CEILING, never a claim**: it hands the read the answer to
    the addressing question and measures only what is left — whether the settle
    and the payload composition work at all. It is the launder in mirror image
    and is labelled DIAGNOSTIC everywhere it appears.
    """
    q0 = np.zeros((len(subsets), int(cfg.n_particles), cfg.dim), dtype=np.float32)
    q0[:, :, :cfg.addr_dim] = (anchors[np.asarray(subsets)][:, :, :cfg.addr_dim]
                               + rng.normal(size=(len(subsets), int(cfg.n_particles),
                                                  int(cfg.addr_dim)))
                               * float(cfg.query_sigma))
    return q0


def _settle_points(store, cfg, q0, *, address_steps=None, read_steps=None,
                   batch: int = 1024):
    a_s = int(cfg.address_steps if address_steps is None else address_steps)
    r_s = int(cfg.read_steps if read_steps is None else read_steps)
    model = store.model(cfg)

    @eqx.filter_jit
    def go(pts):
        p0 = jnp.zeros_like(pts)
        q, p = jax.vmap(lambda a, b: _settle(model, a, b, a_s, cfg.dt,
                                             cfg.gamma_address))(pts, p0)
        q, _ = jax.vmap(lambda a, b: _settle(model, a, b, r_s, cfg.dt,
                                             cfg.gamma_read))(q, jnp.zeros_like(p))
        return q

    flat = jnp.asarray(np.asarray(q0).reshape(-1, cfg.dim), dtype=jnp.float32)
    out = [np.asarray(go(flat[lo:lo + batch])) for lo in range(0, flat.shape[0], batch)]
    return np.concatenate(out).reshape(np.asarray(q0).shape)


def stage_launch_cap(cfg: CatTestConfig, ocfg: OrganizerConfig,
                     seeds: Sequence[int] = (0, 1, 2, 3, 4),
                     arms: Sequence[str] = ("coef0", "phys"),
                     out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐⭐ **Can term (a) move launch-head precision? — the spoke's central question.**

    The task file names it: *"the remaining blocker is LAUNCH-HEAD PRECISION —
    occupancy precision 0.2303, correct-and-distinct 0.92 of 4 ... Training
    φ/placement toward semantic metric structure is loss term (a) — YOURS — and
    it is the named mechanism that could move that number."*

    This stage answers it **with a ceiling, not only with an attempt**. The
    organizer moves wells; it **cannot** move ``phi``. So the best it can do is
    the optimal **bijective re-assignment** of wells to cue sites under the
    label-free co-occurrence matrix ``P[j, c] = E[#times code c is picked |
    feature j present]`` — a linear assignment problem, solved exactly. Reported
    beside it: the achieved precision of every arm, and the ⛔ DIAGNOSTIC
    oracle-addressing read ceiling.
    """
    from scipy.optimize import linear_sum_assignment

    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        head = build_launch_head(phi, cfg)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        picks = np.asarray(jax.vmap(
            lambda i, h=head: h.channels(h.set_code(i)))(jnp.asarray(ind_s)))
        sub_s, sub_u = np.asarray(fam.seen), np.asarray(fam.unseen)
        N = int(cfg.n_wells)
        M = np.zeros((N, N))
        for i in range(len(sub_s)):
            for j in sub_s[i]:
                for c in picks[i]:
                    M[j, c] += 1.0
        cnt = np.array([(sub_s == j).any(1).sum() for j in range(N)], dtype=float)
        P = M / np.maximum(cnt[:, None], 1.0)
        r, c = linear_sum_assignment(-P)
        row = {"seed": int(seed),
               "identity_precision_seen": float(np.mean([P[j, j] for j in range(N)])),
               "best_bijective_assignment": float(P[r, c].mean()),
               "row_max_upper_bound": float(P.max(1).mean()),
               "expected_correct_wells_of_F_at_ceiling": float(
                   P[r, c].mean() * cfg.f_subset),
               "arms": {}}
        for arm in arms:
            cell = build_organized_cell(cfg, ocfg, fam, seed, arm, phi=phi)
            z, q0 = launch_and_settle(cell["store"], cell["head"], cfg, ind_u,
                                      jax.random.PRNGKey(5000 + int(seed)))
            occ_launch = occupancy_precision_from_points(q0, cell["anchors"], sub_u)
            occ_settle = float(np.mean([np.isin(occupancy(z, cell["anchors"])[i],
                                                sub_u[i]).mean()
                                        for i in range(len(sub_u))]))
            yhat = z[:, :, cfg.addr_dim:].sum(1)
            row["arms"][arm] = {
                "launch_occupancy_precision": float(occ_launch),
                "settle_occupancy_precision": occ_settle,
                "zero_param_sum_exact_set": float(exact_set_accuracy(
                    yhat, fam.y_unseen, fam.tol)),
                "jig_norm_median": cell["organize"]["jig_norm_median"],
                "org_mode": cell["organize"].get("org_mode"),
                "coeffs": cell["coeffs"]}
        # ⛔ DIAGNOSTIC: the oracle-addressing ceiling on the coefficient-zero store
        cell0 = build_organized_cell(cfg, ocfg, fam, seed, "coef0", phi=phi)
        rng = np.random.default_rng(1000 + int(seed))
        z_or = _settle_points(cell0["store"], cfg,
                              _oracle_launch(cfg, cell0["anchors"], sub_u, rng))
        err = np.linalg.norm(z_or[:, :, cfg.addr_dim:].sum(1) - fam.y_unseen, axis=1)
        row["DIAGNOSTIC_oracle_addressing"] = {
            "label": "⛔ DIAGNOSTIC ceiling — hands the read the addressing answer",
            "occupancy_precision": float(np.mean(
                [np.isin(occupancy(z_or, cell0["anchors"])[i], sub_u[i]).mean()
                 for i in range(len(sub_u))])),
            "zero_param_sum_exact_set": float((err <= fam.tol).mean()),
            "at_2x_tol": float((err <= 2 * fam.tol).mean()),
            "at_4x_tol": float((err <= 4 * fam.tol).mean()),
            "median_abs_error": float(np.median(err)), "tol": float(fam.tol)}
        cells.append(row)
        print(f"[launch-cap] seed={seed} identity={row['identity_precision_seen']:.4f} "
              f"ceiling={row['best_bijective_assignment']:.4f} "
              f"oracle_exact_set={row['DIAGNOSTIC_oracle_addressing']['zero_param_sum_exact_set']:.4f}",
              flush=True)
    res = {"stage": "launch_cap", "cells": cells,
           "identity_precision": _mean_2se([c["identity_precision_seen"] for c in cells]),
           "assignment_ceiling": _mean_2se([c["best_bijective_assignment"] for c in cells]),
           "row_max_bound": _mean_2se([c["row_max_upper_bound"] for c in cells]),
           "DIAGNOSTIC_oracle_exact_set": _mean_2se(
               [c["DIAGNOSTIC_oracle_addressing"]["zero_param_sum_exact_set"]
                for c in cells]),
           "per_arm_settle_precision": {
               a: _mean_2se([c["arms"][a]["settle_occupancy_precision"]
                             for c in cells]) for a in arms},
           "reading": ("the organizer moves wells, not phi; the bijective "
                       "assignment ceiling is therefore the MOST any placement "
                       "organizer can achieve on this launch head")}
    if out:
        _dump(res, out / "stage_launch_cap.json")
    return res


# ==========================================================================
# ⛔⛔ THE THREE FALSE-POSITIVE GUARDS, IN ONE SECTION (never scattered)
# ==========================================================================
def stage_false_positive_guards(cfg: CatTestConfig, ocfg: OrganizerConfig,
                                seeds: Sequence[int] = (0, 1, 2),
                                arm: str = "phys",
                                out: Optional[Path] = None) -> Dict[str, Any]:
    """⛔⛔ **K4-at-full-ψ · K7-CAP · K8 — together, before any VALUE number.**

    With the G-DRIFT floor demoted to DIAGNOSTIC and K5 moved to this spoke,
    these three carry the **entire false-positive load of the wave**. ⚠ And K4's
    store-only form **ABSTAINED** (vacuous 3/3), so **this is the FIRST
    informative run of that control in this wave and family soundness rests on
    it**. A leak above ``chance + 0.05`` **VOIDS THE FAMILY** — that is what gets
    reported, not a score.

    ⭐ **The ψ budget is SET BY THE MEASURED LEAK** (PREREG §P1): the deployed ψ
    is the largest capacity on the grid whose every leak leg clears. ⛔ Capping ψ
    by fiat would be choosing a design point to make a measurement legible
    (intervention **Error 2**); K8 is the *structural* kill beside the measured one.
    """
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        cell = build_organized_cell(cfg, ocfg, fam, seed, arm, phi=phi)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        kr = jax.random.PRNGKey(7100 + int(seed))
        chance = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
        bar = chance + float(FROZEN["k4_full_psi_bar"])
        ps = _read_pack(cell, cfg, ocfg, ind_s, kr)
        pu = _read_pack(cell, cfg, ocfg, ind_u, jax.random.fold_in(kr, 1))

        # ---- leg 1: BLANK STORE, at full psi capacity ----------------------
        blank = FactoredStore(cfg, cell["anchors"], jax.random.PRNGKey(int(seed)),
                              atom_width=cell["width"]["atom_width"])
        bcell = dict(cell, store=blank)
        bs = _read_pack(bcell, cfg, ocfg, ind_s, jax.random.fold_in(kr, 2))
        bu = _read_pack(bcell, cfg, ocfg, ind_u, jax.random.fold_in(kr, 3))
        # ---- leg 2: QUERY-ONLY (a function of the query alone) -------------
        qs = {"u": np.concatenate(
            [ps["q0"], np.zeros(ps["q0"].shape[:2] + (5,), dtype=np.float32)], -1),
            "w": np.ones(ps["q0"].shape[:2], dtype=np.float32)}
        qu = {"u": np.concatenate(
            [pu["q0"], np.zeros(pu["q0"].shape[:2] + (5,), dtype=np.float32)], -1),
            "w": np.ones(pu["q0"].shape[:2], dtype=np.float32)}
        # ---- leg 3: PERMUTED PAYLOADS --------------------------------------
        rng = np.random.default_rng(int(seed) + 99)
        perm = rng.permutation(int(cfg.n_wells))
        y_s_p = ind_s @ fam.payloads[perm]
        y_u_p = ind_u @ fam.payloads[perm]

        legs_by_cap = {}
        for h in ocfg.psi_hidden_grid:
            kpsi = jax.random.PRNGKey(42000 + int(seed))
            psi_b, rb = fit_set_psi(bs["u"], bs["w"], fam.y_seen, ocfg, kpsi, hidden=h)
            psi_q, _ = fit_set_psi(qs["u"], qs["w"], fam.y_seen, ocfg, kpsi, hidden=h)
            psi_p, _ = fit_set_psi(ps["u"], ps["w"], y_s_p, ocfg, kpsi, hidden=h)
            legs = {
                "blank_store": float(exact_set_accuracy(
                    _psi_predict(psi_b, bu["u"], bu["w"]), fam.y_unseen, fam.tol)),
                "query_only": float(exact_set_accuracy(
                    _psi_predict(psi_q, qu["u"], qu["w"]), fam.y_unseen, fam.tol)),
                "permuted_payloads": float(exact_set_accuracy(
                    _psi_predict(psi_p, pu["u"], pu["w"]), y_u_p, fam.tol))}
            legs["max_leg"] = float(max(legs.values()))
            legs["clears"] = bool(legs["max_leg"] <= bar)
            legs["psi_params"] = int(rb["psi_params"])
            legs_by_cap[int(h)] = legs
        clearing = [h for h in ocfg.psi_hidden_grid if legs_by_cap[int(h)]["clears"]]
        selected = int(max(clearing)) if clearing else None

        # ---- leg 4: the address-leak probe (spoke A's form) ----------------
        leak_full = float(np.mean(
            [np.isin(occupancy(pu["z"], cell["anchors"])[i], np.asarray(fam.unseen)[i]).mean()
             for i in range(len(fam.unseen))]))
        leak_launder = occupancy_precision_from_points(pu["q0"], cell["anchors"],
                                                       np.asarray(fam.unseen))
        # ---- K7-CAP: the SP-1 parameter bound ------------------------------
        readers = fit_readers(ps["z"], fam.y_seen, anchors=cell["anchors"],
                              well_payloads=fam.payloads, seed=seed)
        rp = reader_bytes(readers)
        bound = int(FROZEN["bound_Na_times_m"])
        cells.append({
            "seed": int(seed), "chance": chance, "bar": bar,
            "K4_full_psi_by_capacity": legs_by_cap,
            "psi_capacity_selected_by_measured_leak": selected,
            "K4_full_psi_PASS": bool(selected is not None),
            "address_leak_full": leak_full, "address_leak_launder": leak_launder,
            "address_leak_dividend": float(leak_full - leak_launder),
            "K7_CAP": {"reader_params": rp, "bound_Na_times_m": bound,
                       "every_reader_under_bound": bool(
                           all(int(v) < bound for v in rp.values())),
                       "psi_params": {int(h): legs_by_cap[int(h)]["psi_params"]
                                      for h in ocfg.psi_hidden_grid},
                       "psi_exceeds_bound_by_construction": True,
                       "note": ("psi is NOT a member of the frozen reader class; "
                                "its guard is the MEASURED leak (K4 at full psi) "
                                "plus the K8 structural cell, which is exactly "
                                "why both exist")},
            "novelty_and_psi_ledger": {"psi_bytes_by_capacity": {
                int(h): legs_by_cap[int(h)]["psi_params"] * 4
                for h in ocfg.psi_hidden_grid}, **cell["bytes"]}})
        print(f"[guards] seed={seed} bar={bar:.5f} legs="
              f"{ {h: legs_by_cap[h]['max_leg'] for h in legs_by_cap} } "
              f"psi_selected={selected}", flush=True)
    res = {"stage": "false_positive_guards", "arm": arm, "cells": cells,
           "K4_full_psi_PASS": bool(all(c["K4_full_psi_PASS"] for c in cells)),
           "psi_capacity_selected": [c["psi_capacity_selected_by_measured_leak"]
                                     for c in cells],
           "K8_structural": FROZEN["k8_structural_split"],
           "K8_note": ("the structural facts (rank deficiency, SP-1 cannot "
                       "recover v) are spoke A's and are FROZEN; scoring V1 on "
                       "the K8 cell is a VALUE number and is gated on K5"),
           "family_void": bool(not all(c["K4_full_psi_PASS"] for c in cells))}
    if out:
        _dump(res, out / "stage_guards.json")
    return res


# ==========================================================================
# M3 — per-feature G-ADDR (§A34.8: MECHANICS-ONLY, barred from VALUE duty)
# ==========================================================================
def per_feature_gaddr(z: np.ndarray, picks: np.ndarray, subsets: np.ndarray,
                      anchors: np.ndarray, caps: np.ndarray, addr_dim: int
                      ) -> Dict[str, Any]:
    """Does feature ``f``'s particle resolve to feature ``f``'s well **and** land
    inside that well's **measured** SC-6 capture radius?

    ⛔ ``any_basin`` is reported and **is NOT the leg** (banked: on 4 of 9 pass-3
    cells ``any_basin >= 0.98`` while A1 <= 0.24). ⛔ ``margin_in_SE`` is reported
    beside every boolean (pass 3: ``randconv`` scored 31/31/29 against a 32/128
    threshold — it failed **by ONE read** on 2 of 3 seeds).
    """
    zz = np.asarray(z)[..., :int(addr_dim)]
    a = np.asarray(anchors)[:, :int(addr_dim)]
    occ = ((zz[:, :, None, :] - a[None, None, :, :]) ** 2).sum(-1).argmin(-1)
    hits, tot, basin = [], 0, []
    for i in range(len(subsets)):
        for f in np.asarray(subsets)[i]:
            cs = np.nonzero(np.asarray(picks)[i] == f)[0]
            if len(cs) == 0:
                continue          # no channel asserted f: not an M3 trial
            c = int(cs[0])
            tot += 1
            d = float(np.linalg.norm(zz[i, c] - a[f]))
            hits.append(bool(occ[i, c] == f and d <= float(caps[f])))
    for i in range(len(subsets)):
        for c in range(zz.shape[1]):
            j = int(occ[i, c])
            basin.append(bool(np.linalg.norm(zz[i, c] - a[j]) <= float(caps[j])))
    h = np.asarray(hits, dtype=float)
    score = float(h.mean()) if len(h) else float("nan")
    se = float(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else float("nan")
    return {"per_feature_gaddr": score, "n_trials": int(tot), "se": se,
            "any_basin": float(np.mean(basin)),
            "any_basin_is_not_the_leg": True,
            "frac_features_with_an_asserting_channel": float(
                tot / max(1, len(subsets) * np.asarray(subsets).shape[1]))}


def stage_m3(cfg: CatTestConfig, ocfg: OrganizerConfig,
             seeds: Sequence[int] = (0, 1, 2, 3, 4),
             arms: Sequence[str] = ("coef0", "phys"),
             out: Optional[Path] = None) -> Dict[str, Any]:
    """**M3** — per-feature G-ADDR. ⛔ MECHANICS-ONLY, permanently barred from
    VALUE duty (§A34.8), and its designed negatives are pytest-asserted.

    ⭐ **The G-DRIFT floor travels as a DIAGNOSTIC column here and MAY NOT block
    or fail this leg** (charter ADDENDUM 13, Head ruling 2, citing §A13: table-
    like inference reads are explicitly permitted on BOTH arms, and that
    permission IS the reframe the organizer swap exists for). It remains BLOCKING
    where it was built — C2W8's capture gate.
    """
    chance_m3 = 1.0 / float(cfg.n_wells)
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        row = {"seed": int(seed), "chance": chance_m3, "arms": {}}
        for arm in arms:
            cell = build_organized_cell(cfg, ocfg, fam, seed, arm, phi=phi)
            n_cap = int(min(ocfg.m3_n_capture_wells, cfg.n_wells))
            capinfo = _capture_radii(cell["store"], cfg, cell["targets"], seed,
                                     cfg.n_wells if n_cap >= cfg.n_wells else n_cap)
            caps = np.asarray(capinfo["capture_radii"], dtype=float)
            if len(caps) < cfg.n_wells:   # extend with the measured median
                caps = np.concatenate([caps, np.full(cfg.n_wells - len(caps),
                                                     float(np.median(caps)))])
            z, q0 = launch_and_settle(cell["store"], cell["head"], cfg, ind_u,
                                      jax.random.PRNGKey(5200 + int(seed)))
            hd = cell["head"]
            picks = np.asarray(jax.vmap(
                lambda i, h=hd: h.channels(h.set_code(i)))(jnp.asarray(ind_u)))
            g = per_feature_gaddr(z, picks, np.asarray(fam.unseen), cell["anchors"],
                                  caps, cfg.addr_dim)
            bar = max(4.0 * chance_m3, chance_m3 + 2.0 * (g["se"] or 0.0))
            g.update({"bar": float(bar),
                      "M3_PASS": bool(g["per_feature_gaddr"] >= bar),
                      "margin_in_SE": float((g["per_feature_gaddr"] - bar)
                                            / max(g["se"], 1e-12)),
                      "capture": {k: capinfo[k] for k in
                                  ("capture_median", "frac_capture_ge_sigma_q",
                                   "n_sites", "instrument")},
                      "DIAGNOSTIC_g_drift": {
                          "site_drift_over_spacing": float(np.median(
                              np.linalg.norm(_relax_fn(cell["store"], cfg)(
                                  cell["targets"]) - cell["targets"], axis=-1))
                              / max(cell["spacing"]["median_nn"], 1e-12)),
                          "floor": 0.01,
                          "label": ("⛔ DIAGNOSTIC at tier ii (ADDENDUM 13 ruling "
                                    "2 / §A13) — may not block or fail this leg"),
                          "still_blocking_where_built": "C2W8's capture gate"}})
            row["arms"][arm] = g
        cells.append(row)
        print(f"[M3] seed={seed} " + " ".join(
            f"{a}={row['arms'][a]['per_feature_gaddr']:.4f}"
            f"(any_basin={row['arms'][a]['any_basin']:.3f})" for a in arms), flush=True)
    res = {"stage": "m3", "cells": cells, "chance": chance_m3,
           "per_arm": {a: _mean_2se([c["arms"][a]["per_feature_gaddr"]
                                     for c in cells]) for a in arms},
           "any_basin": {a: _mean_2se([c["arms"][a]["any_basin"] for c in cells])
                         for a in arms},
           "label": "MECHANICS-ONLY (§A34.8), permanently barred from VALUE duty"}
    if out:
        _dump(res, out / "stage_m3.json")
    return res


# ==========================================================================
# M7 / M8 — the curvature-shape term and its END-OF-TRAINING spectrum
# ==========================================================================
def stage_m7_m8(cfg: CatTestConfig, ocfg: OrganizerConfig,
                seeds: Sequence[int] = (0, 1, 2, 3, 4),
                arms: Sequence[str] = ("coef0", "phys"),
                out: Optional[Path] = None) -> Dict[str, Any]:
    """**M7** (does a within-well soft direction survive superposition?) and
    ⭐⭐ **M8** (the end-of-training curvature spectrum — term (c)'s only consumer
    this wave, and therefore MANDATORY, not optional polish).

    ⚠ **The banked trap, marked on the axis rather than described:** undug wells
    report ``lambda_min ~ 0.0993`` **because ``2 alpha`` is what ``lambda_min``
    reports when nothing was written**. A soft direction is therefore defined as
    a ``lambda`` **at or near the floor that is not the floor itself**, AND at a
    site that is dug (``depth >= D_min``) AND still captures (``r >= sigma_q``).

    ⚠ §A4.2 **REFUTED** the tilt instantiation on a learned store. This is
    measured **two-sided**; the shipped confinement floors the soft mode at
    ``2 alpha``, so ``tau_max = Gamma/2 alpha`` travels with every lifetime
    statement (α is the **ceiling**; lowering it breaks the write).
    """
    two_alpha = 2.0 * float(cfg.confine)
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        row = {"seed": int(seed), "two_alpha_floor": two_alpha, "arms": {}}
        for arm in arms:
            cell = build_organized_cell(cfg, ocfg, fam, seed, arm, phi=phi)
            relax = _relax_fn(cell["store"], cfg)
            q_star = relax(cell["targets"])
            V = cell["store"].V

            @eqx.filter_jit
            def spectra(z, V=V):
                H = jax.vmap(jax.hessian(lambda q: jnp.reshape(V(q), ())))(z)
                H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
                w, U = jnp.linalg.eigh(H)
                u2 = U[:, :, 0] ** 2
                pr = (jnp.sum(u2, -1) ** 2) / (jnp.sum(u2 ** 2, -1) + 1e-30)
                return w, pr

            w, pr = spectra(jnp.asarray(q_star, dtype=jnp.float32))
            w = np.asarray(w)
            pr = np.asarray(pr)
            rows = np.stack([np.asarray(cell["store"].group_rows(j))
                             for j in range(int(cfg.n_wells))]).astype(float)
            depth = rows @ np.asarray(cell["store"].V.amp) ** 2
            cap = np.asarray(_capture_radii(cell["store"], cfg, cell["targets"],
                                            seed, int(min(ocfg.m3_n_capture_wells,
                                                          cfg.n_wells)))
                             ["capture_radii"], dtype=float)
            n = len(cap)
            soft = ((w[:n, 0] > two_alpha) & (w[:n, 0] <= two_alpha + 0.02)
                    & (depth[:n] >= float(ocfg.shape_depth_min))
                    & (cap >= float(cfg.query_sigma)))
            row["arms"][arm] = {
                "lambda_min_distribution": _dist(w[:, 0].tolist()),
                "lambda_2nd_distribution": _dist(w[:, 1].tolist()),
                "participation_ratio_distribution": _dist(pr.tolist()),
                "depth_distribution": _dist(depth.tolist()),
                "frac_at_the_2alpha_floor_exactly": float(
                    np.mean(np.abs(w[:, 0] - two_alpha) <= 1e-3)),
                "soft_direction_fraction": float(soft.mean()),
                "n_sites_with_capture_measured": int(n),
                "tau_max_steps": float(-math.log(1.0 - cfg.gamma_address)
                                       / cfg.dt / two_alpha / cfg.dt),
                "coeffs": cell["coeffs"]}
        row["M8_excess_soft_directions_phys_minus_coef0"] = float(
            row["arms"].get("phys", {}).get("soft_direction_fraction", float("nan"))
            - row["arms"].get("coef0", {}).get("soft_direction_fraction", float("nan")))
        row["M8_participation_ratio_excess"] = float(
            row["arms"].get("phys", {}).get(
                "participation_ratio_distribution", {}).get("median", float("nan"))
            - row["arms"].get("coef0", {}).get(
                "participation_ratio_distribution", {}).get("median", float("nan")))
        cells.append(row)
        print(f"[M7/M8] seed={seed} " + " ".join(
            f"{a}: lam_min_med={row['arms'][a]['lambda_min_distribution']['median']:.4f} "
            f"soft={row['arms'][a]['soft_direction_fraction']:.3f} "
            f"PR={row['arms'][a]['participation_ratio_distribution']['median']:.3f}"
            for a in arms), flush=True)
    res = {"stage": "m7_m8", "cells": cells, "two_alpha_floor": two_alpha,
           "soft_direction_fraction": {
               a: _mean_2se([c["arms"][a]["soft_direction_fraction"] for c in cells])
               for a in arms},
           "lambda_min_median": {
               a: _mean_2se([c["arms"][a]["lambda_min_distribution"]["median"]
                             for c in cells]) for a in arms},
           "participation_ratio_median": {
               a: _mean_2se([c["arms"][a]["participation_ratio_distribution"]["median"]
                             for c in cells]) for a in arms},
           "M8_excess": _mean_2se([c["M8_excess_soft_directions_phys_minus_coef0"]
                                   for c in cells]),
           "designed_negative": ("the coefficient-zero arm must show NO EXCESS "
                                 "soft directions over the shipped objective"),
           "floor_note": ("⚠ undug wells report lambda_min ~ 2 alpha = "
                          f"{two_alpha:.4f} BECAUSE that is what an unwritten "
                          "site reports; the floor is marked on the axis")}
    if out:
        _dump(res, out / "stage_m7_m8.json")
    return res


# ==========================================================================
# ⭐ THE C2W9 TRAVERSAL TRIGGER (§7 — this spoke owns the traversal half)
# ==========================================================================
def stage_traversal(cfg: CatTestConfig, ocfg: OrganizerConfig,
                    seeds: Sequence[int] = (0, 1, 2, 3, 4), arm: str = "phys",
                    out: Optional[Path] = None) -> Dict[str, Any]:
    """**In flight: does the evidence point outside the current particle's causal
    diamond?** ⛔ The threshold is registered BEFORE the run (PREREG §P6): at the
    end of the address phase a particle's diamond is the ball of radius
    ``2.0 x s_measured``; a needed well is *unreachable* if it is occupied by no
    particle and lies outside EVERY particle's diamond. The trigger fires iff the
    mean fraction of unreachable needed wells exceeds **0.20** — deliberately the
    same threshold as spoke A's coverage half.
    """
    ruler = float(cfg.s_measured if cfg.s_measured is not None else cfg.atom_width)
    reach = float(ocfg.reach_radius_frac_s) * ruler
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        cell = build_organized_cell(cfg, ocfg, fam, seed, arm)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        z, q0, qm = launch_and_settle(cell["store"], cell["head"], cfg, ind_u,
                                      jax.random.PRNGKey(5300 + int(seed)),
                                      return_mid=True)
        a = np.asarray(cell["anchors"])[:, :cfg.addr_dim]
        mid = np.asarray(qm)[..., :cfg.addr_dim]
        sub = np.asarray(fam.unseen)
        occ = ((mid[:, :, None, :] - a[None, None, :, :]) ** 2).sum(-1).argmin(-1)
        dist = np.linalg.norm(mid[:, :, None, :] - a[None, None, :, :], axis=-1)
        dmin = dist.min(axis=1)                        # (B, N_a)
        needed = np.take_along_axis(dmin, sub, axis=1)  # (B, F)
        visited = np.stack([np.isin(sub[i], occ[i]) for i in range(len(sub))])
        unreachable = (~visited) & (needed > reach)
        frac = unreachable.mean(axis=1)
        per_slot = unreachable.mean(axis=0)
        cells.append({"seed": int(seed), "reach_radius": reach,
                      "mean_frac_needed_wells_unreachable": float(frac.mean()),
                      "frac_queries_with_any_unreachable": float((frac > 0).mean()),
                      "per_slot_frac_unreachable": per_slot.tolist(),
                      "median_distance_to_unvisited_needed_well": float(
                          np.median(needed[~visited])) if (~visited).any() else 0.0,
                      "mean_frac_needed_visited": float(visited.mean()),
                      "n_queries": int(len(sub))})
        print(f"[traversal] seed={seed} unreachable={frac.mean():.4f} "
              f"visited={visited.mean():.4f}", flush=True)
    m = _mean_2se([c["mean_frac_needed_wells_unreachable"] for c in cells])
    fired = bool(m["mean"] > float(ocfg.traversal_threshold))
    res = {"stage": "traversal", "arm": arm, "cells": cells,
           "threshold": float(ocfg.traversal_threshold),
           "mean_frac_unreachable": m, "TRAVERSAL_TRIGGER_FIRED": fired,
           "mode": "TRAVERSAL (in-flight); the COVERAGE half is spoke A's",
           "action": ("append a dated section to "
                      ".claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md"
                      if fired else
                      "⛔ do NOT create the file; an absent trigger is a "
                      "MEASUREMENT, not an omission")}
    if out:
        _dump(res, out / "stage_traversal.json")
    return res


# ==========================================================================
# ⭐⭐ V2 — THE GRADED-NOVELTY READ (leg ii). ⛔ The FLOOR is MECHANICS; the
#     SWAP is VALUE and is never computed here.
# ==========================================================================
def _store_with_unwritten(cell, cfg: CatTestConfig, dropped: np.ndarray, seed: int):
    """Return the cell's store with ``dropped`` wells **never written**.

    The placing write is per-well local (one atom group each), so restoring a
    group's rows from the un-written store is *exactly* "this well was never
    written" — no approximation, and no second write.
    """
    blank = FactoredStore(cfg, cell["anchors"], jax.random.PRNGKey(int(seed)),
                          atom_width=cell["width"]["atom_width"])
    C = np.asarray(cell["store"].V.centers).copy()
    LW = np.asarray(cell["store"].V.log_width).copy()
    AM = np.asarray(cell["store"].V.amp).copy()
    for j in np.asarray(dropped):
        idx = np.nonzero(np.asarray(cell["store"].group_rows(int(j)), dtype=bool))[0]
        C[idx] = np.asarray(blank.V.centers)[idx]
        LW[idx] = np.asarray(blank.V.log_width)[idx]
        AM[idx] = np.asarray(blank.V.amp)[idx]
    V = eqx.tree_at(lambda t: [t.centers, t.log_width, t.amp], cell["store"].V,
                    replace=[jnp.asarray(C, dtype=cell["store"].V.centers.dtype),
                             jnp.asarray(LW, dtype=cell["store"].V.log_width.dtype),
                             jnp.asarray(AM, dtype=cell["store"].V.amp.dtype)])
    return eqx.tree_at(lambda s: s.V, cell["store"], V)


def _novelty_episode(cell, cfg, ocfg, ind, subsets, dropped, key, seed):
    store = _store_with_unwritten(cell, cfg, dropped, seed)
    c2 = dict(cell, store=store)
    pack = _read_pack(c2, cfg, ocfg, ind, key)
    picks = np.asarray(jax.vmap(
        lambda i: cell["head"].channels(cell["head"].set_code(i)))(jnp.asarray(ind)))
    novel = np.isin(picks, np.asarray(dropped)).astype(np.float32)
    pack.update({"picks": picks, "novel": novel, "store": store,
                 "occ": occupancy(pack["z"], cell["anchors"])})
    return pack


def stage_v2(cfg: CatTestConfig, ocfg: OrganizerConfig,
             seeds: Sequence[int] = (0, 1, 2, 3, 4), arm: str = "phys",
             value_legs: bool = False, out: Optional[Path] = None) -> Dict[str, Any]:
    """V2a's **floor** (MECHANICS) + every designed negative; V2b is VALUE.

    ⭐ ``N-e3``, and it is **structural rather than measured**: the dropout mask
    is drawn **independently of the query** and acts on the **WRITE**, so
    ``n_f`` is independent of the query by construction ⇒ a query-only novelty
    head is *provably* at the base rate and any AUROC > 0.5 is store information.
    """
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        cell = build_organized_cell(cfg, ocfg, fam, seed, arm)
        rng = np.random.default_rng(7700 + int(seed))
        n_drop = max(1, int(round(float(ocfg.p_drop) * cfg.n_wells)))
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        # -- training episodes (seen queries, independent masks) --------------
        U, Nv = [], []
        for e in range(int(ocfg.n_train_episodes)):
            dr = rng.choice(cfg.n_wells, size=n_drop, replace=False)
            ep = _novelty_episode(cell, cfg, ocfg, ind_s, fam.seen, dr,
                                  jax.random.PRNGKey(7800 + 10 * seed + e), seed)
            U.append(ep["n"])
            Nv.append(ep["novel"])
        Utr = jnp.asarray(np.concatenate(U), dtype=jnp.float32)
        Ntr = jnp.asarray(np.concatenate(Nv), dtype=jnp.float32)
        head = NoveltyHead(Utr.shape[-1], jax.random.PRNGKey(43000 + int(seed)),
                           hidden=int(ocfg.novelty_hidden))
        opt = optax.adam(float(ocfg.novelty_lr))
        st = opt.init(eqx.filter(head, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, st, Utr=Utr, Ntr=Ntr, opt=opt):
            val, g = eqx.filter_value_and_grad(lambda mm: cal_loss(mm(Utr), Ntr))(m)
            upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
            return eqx.apply_updates(m, upd), st, val

        hist = []
        for _ in range(int(ocfg.novelty_steps)):
            head, st, val = step(head, st)
            hist.append(float(val))

        # -- the EVAL episode (unseen queries, a fresh mask) ------------------
        dr_e = rng.choice(cfg.n_wells, size=n_drop, replace=False)
        ev = _novelty_episode(cell, cfg, ocfg, ind_u, fam.unseen, dr_e,
                              jax.random.PRNGKey(7900 + int(seed)), seed)
        s_log = np.asarray(head(jnp.asarray(ev["n"], dtype=jnp.float32)))
        a_all = auroc(s_log.ravel(), ev["novel"].ravel())
        n_novel_per_query = ev["novel"].sum(1)
        by_count = {}
        for cnt in (0, 1, 2):
            m = n_novel_per_query == cnt
            by_count[int(cnt)] = {"n_queries": int(m.sum()),
                                  "mean_score": float(s_log[m].mean()) if m.any()
                                  else float("nan")}
        # -- designed negatives ----------------------------------------------
        perm = rng.permutation(int(cfg.n_wells))
        cell_p = dict(cell)
        # permuted payloads: rebuild the store with permuted v_j (registered neg.)
        anchors = cell["anchors"]
        tgt_p = cell["targets"].copy()
        tgt_p[:, cfg.addr_dim:cfg.addr_dim + cfg.payload_dim] = fam.payloads[perm]
        store_p, _ = write_store(
            FactoredStore(cfg, anchors, jax.random.PRNGKey(int(seed)),
                          atom_width=cell["width"]["atom_width"]),
            cfg, anchors, fam.payloads[perm], jax.random.PRNGKey(int(seed) + 1),
            depth_scale=_depth_scale(cfg), atom_width=cell["width"]["atom_width"])
        cell_p["store"] = store_p
        ev_p = _novelty_episode(cell_p, cfg, ocfg, ind_u, fam.unseen, dr_e,
                                jax.random.PRNGKey(7900 + int(seed)), seed)
        a_perm = auroc(np.asarray(head(jnp.asarray(ev_p["n"], dtype=jnp.float32))
                                  ).ravel(), ev_p["novel"].ravel())
        cell_b = dict(cell, store=FactoredStore(
            cfg, anchors, jax.random.PRNGKey(int(seed)),
            atom_width=cell["width"]["atom_width"]))
        ev_b = _novelty_episode(cell_b, cfg, ocfg, ind_u, fam.unseen, dr_e,
                                jax.random.PRNGKey(7900 + int(seed)), seed)
        a_blank = auroc(np.asarray(head(jnp.asarray(ev_b["n"], dtype=jnp.float32))
                                   ).ravel(), ev_b["novel"].ravel())
        sh = rng.permutation(ev["novel"].ravel())
        a_shuf = auroc(s_log.ravel(), sh)
        coll = collapse_statistic(ev["occ"], int(cfg.f_subset))
        cells.append({
            "seed": int(seed), "arm": arm,
            "V2a_auroc": float(a_all), "floor": 0.60, "null_below": 0.55,
            "V2a_floor_PASS": bool(a_all > 0.60),
            "V2a_is_a_null": bool(a_all <= 0.55),
            "auroc_by_n_novel_channels": by_count,
            "n_dropped_wells": int(n_drop), "p_drop": float(ocfg.p_drop),
            "n_train_episodes": int(ocfg.n_train_episodes),
            "cal_loss_first": hist[0], "cal_loss_last": hist[-1],
            "DESIGNED_NEGATIVES": {
                "permuted_payloads_REGISTERED": float(a_perm),
                "blank_store": float(a_blank),
                "shuffled_labels": float(a_shuf),
                "note": novelty_negatives_note().strip()[:400]},
            "collapse_statistic": coll,
            "ledger": novelty_ledger(head, psi_params=0,
                                     extra={"episodes": int(ocfg.n_train_episodes)}),
            "V2b_ECE": ("⛔ NOT-RUN: V2b is a VALUE statistic and the K5 stop rule "
                        "applies" if not value_legs else ece(
                            1.0 - 1.0 / (1.0 + np.exp(-s_log.mean(1))),
                            (np.abs(np.asarray(ev["z"])[:, :, cfg.addr_dim:].sum(1)
                                    - fam.y_unseen).max(1) <= fam.tol).astype(float)))})
        print(f"[V2] seed={seed} AUROC={a_all:.4f} (floor 0.60) "
              f"negatives: perm={a_perm:.4f} blank={a_blank:.4f} shuf={a_shuf:.4f} "
              f"collapse={coll['mean_unique_wells']:.2f}/{cfg.f_subset}", flush=True)
    res = {"stage": "v2", "arm": arm, "cells": cells,
           "V2a_auroc": _mean_2se([c["V2a_auroc"] for c in cells]),
           "V2a_floor_PASS": bool(all(c["V2a_floor_PASS"] for c in cells)),
           "negatives": {k: _mean_2se([c["DESIGNED_NEGATIVES"][k] for c in cells])
                         for k in ("permuted_payloads_REGISTERED", "blank_store",
                                   "shuffled_labels")},
           "collapse": _mean_2se([c["collapse_statistic"]["mean_unique_wells"]
                                  for c in cells]),
           "label": ("V2a's > 0.60 FLOOR is MECHANICS (a precondition on the "
                     "VALUE reading); ⛔ the SWAP is VALUE and is not computed "
                     "here — that is the wave review's, from this artifact and "
                     "spoke C's")}
    if out:
        _dump(res, out / "stage_v2.json")
    return res


# ==========================================================================
# V3 — the anytime curve. ⛔ V3-PRIMARY is VALUE (the swap difference) and is
#     NOT computed here; V3-MECHANICS (monotone + non-flat) is MECHANICS.
# ==========================================================================
def stage_v3(cfg: CatTestConfig, ocfg: OrganizerConfig,
             seeds: Sequence[int] = (0, 1, 2, 3, 4), arm: str = "phys",
             out: Optional[Path] = None) -> Dict[str, Any]:
    """The anytime curve at the **frozen** budget grid — the single point of
    coordination with spoke C, who scores the identical grid on the null stores.

    ⛔ **Quote the curve, not the endpoint.** ⭐ **N199 measured a FLAT curve when
    the store carried nothing** — *a memory that carries nothing cannot be read
    better by reading it longer* — so non-flat is positive evidence something
    readable is in there and **FLAT is a MECHANICS FAILURE to be diagnosed, not a
    VALUE number to report on top of.** The ⛔ DIAGNOSTIC oracle-addressed curve
    is emitted beside it precisely so that a flat shipped curve can be
    **attributed** (addressing) rather than merely observed.
    """
    grid = list(FROZEN["v3_budget_grid"]["points_total_verlet_steps"])
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        cell = build_organized_cell(cfg, ocfg, fam, seed, arm)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        sub_u = np.asarray(fam.unseen)
        curve, oracle = {}, {}
        for b in grid:
            a_s = int(round(b / 3))
            r_s = int(b - a_s)
            kr = jax.random.PRNGKey(5400 + int(seed))
            ps = _read_pack(cell, cfg, ocfg, ind_s, kr, address_steps=a_s,
                            read_steps=r_s)
            pu = _read_pack(cell, cfg, ocfg, ind_u, jax.random.fold_in(kr, 1),
                            address_steps=a_s, read_steps=r_s)
            readers = fit_readers(ps["z"], fam.y_seen, anchors=cell["anchors"],
                                  well_payloads=fam.payloads, seed=seed)
            psi, prep = fit_set_psi(ps["u"], ps["w"], fam.y_seen, ocfg,
                                   jax.random.PRNGKey(41000 + int(seed)))
            sc = _score_pack(readers, psi, pu, fam.y_unseen, fam.tol)
            curve[int(b)] = {"scores": sc, "best": float(max(sc.values())),
                             "address_steps": a_s, "read_steps": r_s,
                             "particle_steps": int(b * cfg.n_particles),
                             "settle_occupancy_precision": float(np.mean(
                                 [np.isin(occupancy(pu["z"], cell["anchors"])[i],
                                          sub_u[i]).mean()
                                  for i in range(len(sub_u))]))}
            rng = np.random.default_rng(1000 + int(seed))
            z_or = _settle_points(cell["store"], cfg,
                                  _oracle_launch(cfg, cell["anchors"], sub_u, rng),
                                  address_steps=a_s, read_steps=r_s)
            oracle[int(b)] = float(exact_set_accuracy(
                z_or[:, :, cfg.addr_dim:].sum(1), fam.y_unseen, fam.tol))
        vals = [curve[b]["best"] for b in grid]
        ovals = [oracle[b] for b in grid]
        cells.append({"seed": int(seed), "grid": grid, "curve": curve,
                      "spread": float(max(vals) - min(vals)),
                      "monotone": bool(all(vals[i] <= vals[i + 1] + 1e-12
                                           for i in range(len(vals) - 1))),
                      "DIAGNOSTIC_oracle_addressed_curve": oracle,
                      "DIAGNOSTIC_oracle_spread": float(max(ovals) - min(ovals)),
                      "DIAGNOSTIC_oracle_monotone": bool(
                          all(ovals[i] <= ovals[i + 1] + 1e-12
                              for i in range(len(ovals) - 1)))})
        print(f"[V3] seed={seed} shipped curve={[round(v,4) for v in vals]} "
              f"oracle={[round(v,4) for v in ovals]}", flush=True)
    res = {"stage": "v3", "arm": arm, "grid": grid, "cells": cells,
           "ledger": "particles-evolved x Verlet steps (the ONE shared ledger)",
           "curve_mean": {int(b): _mean_2se([c["curve"][b]["best"] for c in cells])
                          for b in grid},
           "oracle_curve_mean": {
               int(b): _mean_2se([c["DIAGNOSTIC_oracle_addressed_curve"][b]
                                  for c in cells]) for b in grid},
           "spread": _mean_2se([c["spread"] for c in cells]),
           "V3_MECHANICS_non_flat": bool(np.mean([c["spread"] for c in cells])
                                         > 2.0 / float(cfg.n_unseen)),
           "V3_PRIMARY": ("⛔ VALUE — the SWAP DIFFERENCE. Not computed here; the "
                          "grid is emitted so OD_V3(b) is computable at review "
                          "from this artifact and spoke C's"),
           "V3_REPORTED_read_compute": {
               "label": "⛔ a READ-COMPUTE RATIO (mult-adds), not wall-clock, not "
                        "training cost",
               "physics_mult_adds_at_full_budget": int(
                   cfg.n_particles * 1200 * cfg.n_atoms * cfg.dim * 2),
               "banked_reference": "C2W5: 6.88e7 vs N1's 20 480 = 3 360x at a TIE"}}
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        _dump(res, Path(out) / "stage_v3.json")
    return res


# ==========================================================================
# the runner
# ==========================================================================
STAGES = ("launch_cap", "k5", "guards", "m3", "m7_m8", "traversal", "v2", "v3")


def run_c2w11_organizer(project: Optional[str] = None,
                        seeds: Sequence[int] = (), quick: bool = False,
                        out_dir: Optional[str] = None,
                        stages: Sequence[str] = (),
                        arms: Sequence[str] = ()) -> Dict[str, Any]:
    """⛔ **RUN ORDER IS NOT COSMETIC.** ``k5`` is this spoke's registered
    BLOCKING first act and ``launch_cap`` is the cheap mechanism cell that
    interprets it; the VALUE legs (V1, V2's swap, V3-PRIMARY) are gated on K5
    **scoring** rather than abstaining."""
    seeds = tuple(int(s) for s in (seeds or (0, 1, 2, 3, 4)))
    arms = tuple(arms or ("coef0", "phys"))
    stages = tuple(stages or ("launch_cap", "k5"))
    out = Path(out_dir) if out_dir else Path("outputs") / "c2w11_organizer"
    out.mkdir(parents=True, exist_ok=True)
    cfg = organizer_config()
    ocfg = OrganizerConfig()
    if quick:
        cfg = _with(cfg, n_unseen=64, n_items=48)
        ocfg = replace(ocfg, jig_steps=40, refine_steps=20, psi_steps=100,
                       novelty_steps=60, n_train_episodes=1, m3_n_capture_wells=4)
        seeds = seeds[:1]
    res = {"frozen": load_frozen(), "seeds": list(seeds), "arms": list(arms),
           "organizer_flags": ocfg.as_flag_table(), "quick": bool(quick)}
    for st in stages:
        if st == "launch_cap":
            res[st] = stage_launch_cap(cfg, ocfg, seeds, arms, out)
        elif st == "k5":
            res[st] = stage_k5_organized(cfg, ocfg, seeds, "phys", out)
        elif st == "guards":
            res[st] = stage_false_positive_guards(cfg, ocfg, seeds, "phys", out)
        elif st == "m3":
            res[st] = stage_m3(cfg, ocfg, seeds, arms, out)
        elif st == "m7_m8":
            res[st] = stage_m7_m8(cfg, ocfg, seeds, arms, out)
        elif st == "traversal":
            res[st] = stage_traversal(cfg, ocfg, seeds, "phys", out)
        elif st == "v2":
            res[st] = stage_v2(cfg, ocfg, seeds, "phys", False, out)
        elif st == "v3":
            res[st] = {a: stage_v3(cfg, ocfg, seeds, a,
                                   out / a if out else None) for a in arms}
        else:
            raise ValueError(f"unknown stage {st!r}; known: {STAGES}")
    _dump(res, out / "c2w11_organizer_summary.json")
    return res
