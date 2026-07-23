"""Experiment DESIGNED-MECHANISM-LEARNED-CONTENT (w22): is the K=8 wall GEOMETRY
or LEARNING?

Every "primitive" is a **designed mechanism with learned content** (attention's
``softmax(QK)V`` is fixed; ``W_{Q,K,V}`` are learned). The fair CLU configuration is
a fixed atom-dictionary MECHANISM (:class:`AtomDictionaryPotential`) with **learned**
amplitudes/centers/widths, written by the static objective
(:func:`chlu.training.train_memory`). In ``potential-function-class`` that arm scored
0.980 @K=4 and **broke** at 0.741 @K=8 — but K=8 sits at the 2-D ring's own capacity
ceiling (``K_max≈8.4``), so we cannot tell whether the wall is:

* **H-GEOMETRY** — the ring ran out of room; the wall moves up with ``d`` (designed
  capacity is ``4·2^d``). Primitive claim alive.
* **H-LEARNING** — gradient descent cannot fill a landscape past ~8 items regardless
  of room; the wall stays near 8 at every ``d``. Primitive claim in trouble.

This module discriminates them by sweeping the **address dimension** ``d`` (a
``d``-ball address space, sites farthest-point-packed by :func:`designed_sites`,
payload channel at index ``d``) and measuring, at each ``d``, ``K_learned`` = the
largest item count a LEARNED atom dictionary clears at strict 0.9 (leak-immune value
criterion, blank control on every cell), overlaid on ``K_designed`` re-measured on the
IDENTICAL harness with a hand-built :class:`BallRegisterPotential`.

⚠ **The parameter ceiling is a confound (theorist §4.3, ``B_total ≤ P·b_θ``).** The
atom count is **scaled with K** (``n_atoms = atoms_per_item·K``, ``n_groups = K``) so a
plateau is a *learning* failure, not a *parameterization-capacity* failure. ``P`` is
reported per cell.

Items:

1. ``item1_discriminator``  ⭐ ``K_learned`` vs ``d`` and ``K_designed`` vs ``d`` on
    one axis, ≥5 seeds on the learned cells, with the fitted growth of ``K_learned``.
2. ``item2_mass``           per-item learned/assigned mass vs uniform, WITH the
    address-coupling check (Prop F1: mass helps only if ``∂_i∂_j V ≠ 0``).
3. ``item3_interference``   masked vs global cross-write corruption at each ``d``.
4. ``item4_frontier``       the (d,K) performance frontier: where learned content
    matches the designed ceiling and where it falls away.

Runnable directly:
    uv run python -m chlu.experiments.exp_designed_mechanism --quick
or via the CLI: ``chlu exp-designed-mechanism [--project N] [--seed I] [--quick]``.
"""

import copy
import json
import os
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.memory_potentials import (
    BallRegisterPotential,
    DesignFreedomPotential,
    atom_write_mask_fn,
    designed_payloads,
    designed_sites,
    site_separation,
)
from chlu.experiments.exp_retrieval import (
    linear_codebook_read,
    nearest_centroid_read,
)
from chlu.experiments.goldstone_harness import clu_with_potential
from chlu.training.train_memory import train_memory_landscape

# ---------------------------------------------------------------------------
# Geometry: the d-ball address space (address q[:d], payload q[d])
# ---------------------------------------------------------------------------


def ball_setup(d: int, K: int, cfg, payloads=None):
    """Centers (K,d), payloads (K,), targets (K,d+1), min site separation."""
    centers = designed_sites(d, K, R=cfg.R, seed=cfg.site_seed)
    if payloads is None:
        payloads = designed_payloads(K, seed=cfg.payload_seed)
    payloads = jnp.asarray(payloads, dtype=jnp.float32)
    dim = d + 1
    targets = jnp.zeros((K, dim))
    targets = targets.at[:, :d].set(centers).at[:, d].set(payloads)
    return centers, payloads, targets, site_separation(centers)


def _atoms_for(cfg, K: int) -> int:
    """Atom count = ``max(atoms_per_item·K, min_atoms)``.

    The ``·K`` term scales the parameter budget with K so a ``K_learned`` plateau is
    a LEARNING failure, not a parameterization-capacity one (theorist §4.3). The
    ``min_atoms`` FLOOR is load-bearing and separate: a large over-complete
    dictionary also *smooths the write optimization* (potential-function-class used
    896 atoms at every K), so scaling atoms down at small K would STARVE the write
    and inject an optimization artifact (measured: d=4 K=2 with 64 atoms leaves the
    write loss stuck at 0.18 on some seeds). The floor keeps every cell in the
    over-complete regime; the ·K term dominates once K is large.
    """
    return max(cfg.atoms_per_item * K, cfg.min_atoms)


def build_designed_model(centers, payloads, cfg) -> CHLU:
    """CLU wired to a hand-built d-ball register (the designed ceiling arm)."""
    V = BallRegisterPotential(
        payloads,
        centers,
        R=cfg.R + cfg.wall_margin,
        w=cfg.well_width,
        b=cfg.well_depth,
        kappa=cfg.payload_kappa,
        c_conf=cfg.c_conf,
    )
    dim = V.dim
    return clu_with_potential(
        V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )


def build_learned_V(d: int, K: int, cfg, key) -> DesignFreedomPotential:
    """A LEARNED atom dictionary on the (d+1)-dim latent, K groups, K-scaled atoms.

    Returned as a ``DesignFreedomPotential(rung="free_mlp", learned_family="atoms")``
    whose designed part is ``None`` — i.e. a PURE ``AtomDictionaryPotential`` wrapped
    in the ``.learned`` container that ``train_memory.trainable_filter`` and
    ``atom_write_mask_fn`` require (a bare atom dictionary has no ``.learned`` subtree,
    so ``train_memory_landscape`` would train it to a silent no-op).

    Starts FLAT (``depth_init`` tiny, ``A = amp**2``) so the writer digs the wells;
    partitioned into ``K`` contiguous atom blocks so a masked write is local in
    parameter space (one block per item slot). ``.learned`` is the atom dictionary.
    """
    n_atoms = _atoms_for(cfg, K)
    return DesignFreedomPotential(
        rung="free_mlp",
        dim=d + 1,
        payloads=jnp.zeros((K,)),  # unused: the free_mlp rung has designed=None
        key=key,
        learned_family="atoms",
        n_atoms=n_atoms,
        rbf_init_width=cfg.atom_init_width,
        confine=cfg.learned_confine,
        atom_depth_init=cfg.atom_depth_init,
        atom_groups=K,
        atom_init_scale=cfg.atom_init_scale,
    )


def _n_params(V) -> int:
    return int(
        sum(
            x.size
            for x in jax.tree_util.tree_leaves(eqx.filter(V, eqx.is_inexact_array))
        )
    )


def _loss_kwargs(cfg, d: int) -> dict:
    return dict(
        n_perturb=cfg.write_n_perturb,
        sigma_addr=cfg.write_sigma_addr,
        sigma_pay=cfg.write_sigma_pay,
        margin=cfg.write_margin,
        barrier=cfg.write_barrier,
        payload_index=d,  # payload channel is q[d] in the ball geometry
    )


def write_learned(V, targets, cfg, key, d: int, mode: str = "local"):
    """Write ``targets`` into a learned atom dictionary.

    * ``"global"`` — one Adam run over ALL atoms, jointly on all targets (the w20
      write; interferes across items via the shared gradient step).
    * ``"local"``  — one masked single-item write per item; every OTHER item's
      atoms come out bit-identical (the MVC-0 C3-local write, the atom mechanism's
      best operator per ``potential-function-class``).
    """
    targets = jnp.asarray(targets)
    K = targets.shape[0]
    lk = _loss_kwargs(cfg, d)
    if mode == "global":
        V, hist = train_memory_landscape(
            V,
            targets,
            key,
            steps=cfg.write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=lk,
        )
        return V, hist
    if mode != "local":
        raise ValueError(f"unknown write mode {mode!r}")
    hist = []
    for i in range(K):
        key, k = jax.random.split(key)
        mask = V.learned.group_rows(i) if hasattr(V, "learned") else V.group_rows(i)
        V, h = train_memory_landscape(
            V,
            targets[i : i + 1],
            k,
            steps=cfg.local_write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=lk,
            update_mask_fn=atom_write_mask_fn(mask),
        )
        hist.extend(h)
    return V, hist


# ---------------------------------------------------------------------------
# Queries + two-phase retrieval, d-ball geometry
# ---------------------------------------------------------------------------


def make_ball_queries(key, centers, n_per_item: int, cfg):
    """Jittered addresses around each site; payload channel launched at 0 (guard).

    ``fixed_norm`` jitter (``sigma/sqrt(d)`` per axis) so the query NORM is
    ``query_sigma`` at every ``d`` — precision held constant, only the address
    dimension varies (the apples-to-apples generalization of the ring).
    """
    K, d = centers.shape
    dim = d + 1
    k_x, k_p = jax.random.split(key, 2)
    n = K * n_per_item
    labels = np.repeat(np.arange(K), n_per_item)
    scale = cfg.query_sigma / np.sqrt(d)
    x0 = jnp.repeat(jnp.asarray(centers), n_per_item, axis=0)
    x0 = x0 + jax.random.normal(k_x, (n, d)) * scale
    Q0 = jnp.zeros((n, dim)).at[:, :d].set(x0)
    P0 = jnp.zeros((n, dim)).at[:, :d].set(
        jax.random.normal(k_p, (n, d)) * cfg.query_sigma_p
    )
    return Q0, P0, labels


def _two_phase(model, Q0, P0, cfg, d: int, masses=None):
    """query -> [gamma_address relax] -> address -> [gamma_read rollout] -> traj.

    Returns ``(addr_x (n,d), payload_tail (n, n_subsample))``, chunked so the jit
    compiles once and the full trajectory is never materialized outside it.
    ``masses`` (n, dim) supplies a per-query inertial mass (the address-side key);
    ``None`` uses identity mass.
    """
    dim = d + 1
    steps = cfg.read_steps
    start = int((1.0 - cfg.tail_frac) * steps)
    tail_idx = jnp.asarray(np.linspace(start, steps - 1, cfg.n_subsample).astype(int))
    ones = jnp.ones(dim)

    @eqx.filter_jit
    def one(q, p, m):
        tr1 = model(q, p, cfg.address_steps, cfg.dt, cfg.gamma_address, m)
        aq, ap = tr1[-1, :dim], tr1[-1, dim:]
        tr2 = model(aq, ap, cfg.read_steps, cfg.dt, cfg.gamma_read, m)
        return tr2[-1, :d], tr2[tail_idx, d]

    n = Q0.shape[0]
    chunk = min(cfg.rollout_chunk, n)
    xs, feats = [], []
    for i in range(0, n, chunk):
        q, p = Q0[i : i + chunk], P0[i : i + chunk]
        m = ones[None, :].repeat(q.shape[0], axis=0) if masses is None else masses[i : i + chunk]
        pad = chunk - q.shape[0]
        if pad > 0:
            q = jnp.concatenate([q, jnp.zeros((pad,) + q.shape[1:])], axis=0)
            p = jnp.concatenate([p, jnp.zeros((pad,) + p.shape[1:])], axis=0)
            m = jnp.concatenate([m, jnp.ones((pad,) + m.shape[1:])], axis=0)
        x, f = jax.vmap(one)(q, p, m)
        if pad > 0:
            x, f = x[: chunk - pad], f[: chunk - pad]
        xs.append(np.asarray(x))
        feats.append(np.asarray(f))
    return np.concatenate(xs, axis=0), np.concatenate(feats, axis=0)


def score_cell(model, centers, payloads, cfg, d: int, seed: int, masses_fn=None):
    """Run the loop on ONE landscape and score it (the unit of measurement).

    ``masses_fn(labels) -> (n, dim)`` optionally supplies a per-query mass keyed by
    the query's item label (the per-item mass arm); ``None`` = identity mass.

    Returns strict/basin/payload/selectivity + the classification reads (for the
    blank control).
    """
    K = len(payloads)
    dim = d + 1
    key = jax.random.PRNGKey(seed)
    n_per = int(
        np.clip(cfg.max_total_queries // K, cfg.min_query_per_item, cfg.n_query_per_item)
    )
    Q0, P0, labels = make_ball_queries(key, centers, n_per, cfg)
    masses = None
    if masses_fn is not None:
        masses = masses_fn(labels, dim)

    addr_x, feat = _two_phase(model, Q0, P0, cfg, d, masses=masses)
    finite = bool(np.all(np.isfinite(addr_x)) and np.all(np.isfinite(feat)))

    c = np.asarray(centers)
    d2 = ((addr_x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    basin = np.argmin(d2, axis=1)
    basin_ok = basin == labels

    read_val = feat.mean(axis=1)
    pay = np.asarray(payloads)
    err = np.abs(read_val - pay[labels])
    strict_ok = basin_ok & (err < cfg.payload_tol)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(feat.shape[0])
    half = feat.shape[0] // 2
    tr, te = perm[:half], perm[half:]
    acc_cb, _ = linear_codebook_read(feat[tr], labels[tr], feat[te], labels[te], pay)
    acc_nc = nearest_centroid_read(feat[tr], labels[tr], feat[te], labels[te], K)

    return {
        "K": K,
        "finite": finite,
        "basin_success_rate": float(np.mean(basin_ok)),
        "strict_success_rate": float(np.mean(strict_ok)),
        "selectivity": float(np.mean(basin_ok)),
        "payload_abs_err_mean": float(np.mean(err)),
        "acc_payload_codebook_read": float(acc_cb),
        "acc_payload_nearest_centroid": float(acc_nc),
        "chance": 1.0 / K,
        "n_queries": int(feat.shape[0]),
    }


# ---------------------------------------------------------------------------
# One (arm, d, K, seed) cell: written + blank + pass/fail
# ---------------------------------------------------------------------------


def evaluate_arm_cell(arm: str, d: int, K: int, seed: int, cfg):
    """Train (learned) or build (designed) the WRITTEN + BLANK landscape and score.

    Value criterion (leak-immune): mean strict >= pass_strict AND the value blank
    (blank strict <= blank_strict_max) passes. A learned V couples the payload
    channel to the address, so a *classification* read leaks the address on a blank
    landscape — only the value-recovery strict metric is used to gate.
    """
    centers, payloads, targets, sep = ball_setup(d, K, cfg)
    blank_pay = jnp.zeros_like(payloads)
    _, _, blank_targets, _ = ball_setup(d, K, cfg, payloads=blank_pay)
    n_params = 0
    t0 = time.perf_counter()

    if arm == "designed":
        mw = build_designed_model(centers, payloads, cfg)
        mb = build_designed_model(centers, blank_pay, cfg)
    elif arm in ("learned_local", "learned_global"):
        mode = "local" if arm == "learned_local" else "global"
        k_w, k_b = jax.random.split(jax.random.PRNGKey(seed + 7919), 2)
        Vw = build_learned_V(d, K, cfg, k_w)
        Vw, hist = write_learned(Vw, targets, cfg, k_w, d, mode=mode)
        Vb = build_learned_V(d, K, cfg, k_b)
        Vb, _ = write_learned(Vb, blank_targets, cfg, k_b, d, mode=mode)
        n_params = _n_params(Vw)
        mw = clu_with_potential(
            Vw, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
        )
        mb = clu_with_potential(
            Vb, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    write_seconds = time.perf_counter() - t0

    written = score_cell(mw, centers, payloads, cfg, d, seed)
    blank = score_cell(mb, centers, payloads, cfg, d, seed)
    # ⚠ A blank landscape returns ~0 for every item, so it LEGITIMATELY "retrieves"
    # any item whose real payload happens to lie within payload_tol of 0. At large K
    # the linspace(-1,1,K) codebook has several such near-zero values, so a valid
    # blank scores up to this trivial ceiling on strict — gating at a flat
    # blank_strict_max would spuriously disqualify the DESIGNED arm at large K.
    pay_np = np.asarray(payloads)
    trivial_ceiling = float(np.mean(np.abs(pay_np) < cfg.payload_tol))
    blank_ceiling = max(cfg.blank_strict_max, trivial_ceiling + 0.02)
    value_blank_ok = bool(blank["strict_success_rate"] <= blank_ceiling)
    class_blank = max(
        blank["acc_payload_codebook_read"], blank["acc_payload_nearest_centroid"]
    )
    class_blank_ok = bool(class_blank <= blank["chance"] + cfg.blank_margin)
    return {
        "arm": arm,
        "d": d,
        "K": K,
        "seed": seed,
        "site_sep": float(sep),
        "n_learned_params": n_params,
        "param_bits_budget": int(n_params * cfg.bits_per_param),
        "written": written,
        "blank": blank,
        "value_blank_ok": value_blank_ok,
        "classification_blank_ok": class_blank_ok,
        "write_seconds": float(write_seconds),
    }


def _agg(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def k_star_for_arm(arm: str, d: int, cfg, seeds, verbose=True):
    """Walk the K ladder; K_star = largest K clearing strict at every seed's blank.

    A cell passes iff mean strict over seeds >= pass_strict AND the value blank
    passes for EVERY seed (a leaking cell is not a measurement). Stops at the first
    failing K. A cell that runs off the ladder / cap without failing is CENSORED
    (K_star a lower bound).
    """
    ladder = [K for K in cfg.k_ladder if K <= cfg.k_cap]
    per_K, censored = [], False
    for K in ladder:
        cells = [evaluate_arm_cell(arm, d, K, s, cfg) for s in seeds]
        strict = [c["written"]["strict_success_rate"] for c in cells]
        blanks_ok = all(c["value_blank_ok"] for c in cells)
        mean_strict = float(np.mean(strict))
        passes = bool(mean_strict >= cfg.pass_strict and blanks_ok)
        rec = {
            "K": K,
            "strict": _agg(strict),
            "selectivity": _agg([c["written"]["selectivity"] for c in cells]),
            "payload_abs_err": _agg(
                [c["written"]["payload_abs_err_mean"] for c in cells]
            ),
            "n_value_blank_pass": int(sum(c["value_blank_ok"] for c in cells)),
            "blank_strict": _agg([c["blank"]["strict_success_rate"] for c in cells]),
            "n_learned_params": cells[0]["n_learned_params"],
            "site_sep": cells[0]["site_sep"],
            "passes": passes,
            "write_seconds": _agg([c["write_seconds"] for c in cells]),
        }
        per_K.append(rec)
        if verbose:
            print(
                f"  [{arm}] d={d} K={K:5d} strict={mean_strict:.3f}"
                f" blankOK={blanks_ok} P={rec['n_learned_params']}"
                f" sep={rec['site_sep']:.3f} -> {'PASS' if passes else 'fail'}",
                flush=True,
            )
        if not passes:
            break
    else:
        censored = True

    passing = [r["K"] for r in per_K if r["passes"]]
    k_star = max(passing, default=0)
    k_top = max(r["K"] for r in per_K)
    return {
        "arm": arm,
        "d": d,
        "k_star": k_star,
        "censored": bool(censored and k_star == k_top),
        "per_K": per_K,
        "seeds": list(seeds),
    }


def _fit_growth(ds, ks, censored=None):
    """Exponential (A^d) vs polynomial (d^alpha) fit; censored points excluded."""
    ds = np.asarray(ds, float)
    ks = np.asarray(ks, float)
    ok = ks > 1
    fit = {}
    if censored is not None:
        ok = ok & ~np.asarray(censored, dtype=bool)
        fit["n_censored_excluded"] = int(np.sum(censored))
    if ok.sum() >= 2:
        A = np.stack([ds[ok], np.ones(ok.sum())], axis=1)
        coef, *_ = np.linalg.lstsq(A, np.log(ks[ok]), rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((np.log(ks[ok]) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ks[ok]) - np.log(ks[ok]).mean()) ** 2))
        fit["exponential_base_A"] = float(np.exp(coef[0]))
        fit["exponential_intercept"] = float(coef[1])
        fit["exponential_r2"] = 1.0 - ss_res / (ss_tot + 1e-12)
        B = np.stack([np.log(ds[ok]), np.ones(ok.sum())], axis=1)
        cp, *_ = np.linalg.lstsq(B, np.log(ks[ok]), rcond=None)
        rp = float(np.sum((np.log(ks[ok]) - B @ cp) ** 2))
        fit["polynomial_exponent_alpha"] = float(cp[0])
        fit["polynomial_r2"] = 1.0 - rp / (ss_tot + 1e-12)
        fit["exponential_beats_polynomial"] = bool(ss_res < rp)
        fit["n_points_fitted"] = int(ok.sum())
    return fit


# ---------------------------------------------------------------------------
# Item 1 -- ⭐ the discriminator: K_learned vs d, K_designed overlaid
# ---------------------------------------------------------------------------


def item1_discriminator(cfg):
    """K_learned vs d and K_designed vs d on one axis, with the fitted growth."""
    learned_rows, designed_rows = [], []
    for d in cfg.dims:
        designed_rows.append(
            k_star_for_arm("designed", d, cfg, cfg.designed_seeds)
        )
        learned_rows.append(
            k_star_for_arm(cfg.learned_arm, d, cfg, cfg.discriminator_seeds)
        )

    def _fit(rows):
        return _fit_growth(
            [r["d"] for r in rows],
            [max(r["k_star"], 1) for r in rows],
            [r["censored"] for r in rows],
        )

    fit_l = _fit(learned_rows)
    fit_d = _fit(designed_rows)

    # ratio K_learned / K_designed per d (falls under H-GEOMETRY-WEAK, ~const under
    # H-GEOMETRY-STRONG, and collapses hardest under H-LEARNING).
    ratios = []
    for lr, dr in zip(learned_rows, designed_rows, strict=True):
        kd = dr["k_star"]
        ratios.append(
            {
                "d": lr["d"],
                "k_learned": lr["k_star"],
                "k_designed": kd,
                "k_designed_4x2d": int(4 * 2**lr["d"]),
                "ratio_learned_over_designed": (
                    float(lr["k_star"] / kd) if kd > 0 else None
                ),
                "learned_censored": lr["censored"],
                "designed_censored": dr["censored"],
            }
        )

    # Verdict per the pre-registered decision rule.
    A = fit_l.get("exponential_base_A", 1.0)
    kl = [r["k_star"] for r in learned_rows]
    grows = bool(max(kl) - min(kl) >= 4)  # >= +2 ladder rungs over the sweep
    if A >= 1.9:
        verdict = "H-GEOMETRY-STRONG"
    elif A >= 1.3 and fit_l.get("exponential_r2", 0) >= 0.8 and grows:
        verdict = "H-GEOMETRY-WEAK"
    elif A < 1.3 and not grows:
        verdict = "H-LEARNING"
    else:
        verdict = "AMBIGUOUS"

    return {
        "dims": list(cfg.dims),
        "learned_arm": cfg.learned_arm,
        "pass_strict": cfg.pass_strict,
        "discriminator_seeds": list(cfg.discriminator_seeds),
        "designed_seeds": list(cfg.designed_seeds),
        "atoms_per_item": cfg.atoms_per_item,
        "k_learned_vs_d": [
            {"d": r["d"], "k_star": r["k_star"], "censored": r["censored"]}
            for r in learned_rows
        ],
        "k_designed_vs_d": [
            {"d": r["d"], "k_star": r["k_star"], "censored": r["censored"]}
            for r in designed_rows
        ],
        "ratios": ratios,
        "fit_k_learned": fit_l,
        "fit_k_designed": fit_d,
        "verdict": verdict,
        "learned_detail": learned_rows,
        "designed_detail": designed_rows,
    }


# ---------------------------------------------------------------------------
# Item 2 -- does per-item MASS help? (folds in mass-visible-objective + Prop F1)
# ---------------------------------------------------------------------------


def _hessian_coupling(V, centers, d: int):
    """Address-coupling ratio mean|off-diag| / mean|diag| of Hess V at stored sites.

    Prop F1 (relaxation-fiber-capacity): mass is address-side and worth ~0 bits in a
    SEPARABLE well (``∂_i∂_j V = 0``). An isotropic Gaussian atom has a diagonal
    Hessian at its own center, so this ratio being ~0 PREDICTS a mass null.
    Evaluated on the address block ``[:d, :d]`` at each stored site.
    """
    H = eqx.filter_jit(jax.hessian(lambda q: V(q)))
    ratios = []
    for c in np.asarray(centers):
        q = np.zeros(d + 1, dtype=np.float32)
        q[:d] = c
        h = np.asarray(H(jnp.asarray(q)))[:d, :d]
        diag = np.abs(np.diag(h))
        off = np.abs(h - np.diag(np.diag(h)))
        md = float(np.mean(diag)) if diag.size else 0.0
        mo = float(np.sum(off) / max(off.size - d, 1))
        ratios.append(mo / (md + 1e-12))
    return float(np.mean(ratios))


def item2_mass(cfg):
    """Uniform mass (a) vs per-item mass spread (b), at fixed d, WITH the coupling
    check. Per Prop F1, expect ~0 gain unless the atom wells couple coordinates.

    ⚠ Honesty: the write objective is mass-BLIND by construction (kinetic terms
    cancel in a static minimum-digging loss), so the per-item masses are ASSIGNED
    (a geometric spread), not gradient-learned. Whether *any* mass value changes
    retrieval is the load-bearing test — a separable well is mass-invariant at its
    fixed point, so the coupling ratio determines the ceiling on what mass can buy.
    """
    d, K = cfg.mass_dim, cfg.mass_K
    out = []
    for seed in cfg.mass_seeds:
        centers, payloads, targets, _ = ball_setup(d, K, cfg)
        k_w = jax.random.PRNGKey(seed + 111)
        V = build_learned_V(d, K, cfg, k_w)
        V, _ = write_learned(V, targets, cfg, k_w, d, mode="local")
        model = clu_with_potential(
            V, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
        )
        coupling = _hessian_coupling(V, centers, d)

        # arm (a): uniform mass
        a = score_cell(model, centers, payloads, cfg, d, seed, masses_fn=None)

        # arm (b): per-item geometric mass spread, keyed by the query's item label
        spread = np.geomspace(1.0 / cfg.mass_spread, cfg.mass_spread, K).astype(
            np.float32
        )

        def masses_fn(labels, dim, spread=spread):
            m = np.ones((len(labels), dim), dtype=np.float32)
            m[:, :] = spread[labels][:, None]
            return jnp.asarray(m)

        b = score_cell(model, centers, payloads, cfg, d, seed, masses_fn=masses_fn)
        out.append(
            {
                "seed": seed,
                "coupling_ratio_offdiag_over_diag": coupling,
                "uniform_strict": a["strict_success_rate"],
                "permass_strict": b["strict_success_rate"],
                "uniform_payload_err": a["payload_abs_err_mean"],
                "permass_payload_err": b["payload_abs_err_mean"],
                "delta_strict": b["strict_success_rate"] - a["strict_success_rate"],
            }
        )
    return {
        "d": d,
        "K": K,
        "mass_spread": cfg.mass_spread,
        "seeds": list(cfg.mass_seeds),
        "note": "masses ASSIGNED (write objective is mass-blind); coupling ratio "
        "bounds what mass can buy (Prop F1).",
        "rows": out,
        "coupling_ratio": _agg([r["coupling_ratio_offdiag_over_diag"] for r in out]),
        "delta_strict": _agg([r["delta_strict"] for r in out]),
        "mass_helps": bool(
            np.mean([r["delta_strict"] for r in out]) > cfg.mass_help_threshold
        ),
    }


# ---------------------------------------------------------------------------
# Item 3 -- interference across d: masked vs global write
# ---------------------------------------------------------------------------


def _corruption(arm_mode: str, d: int, seed: int, cfg):
    """Write A (K-1 items), read A; write B into a fresh site; re-read A.

    ``arm_mode`` in {"local", "global"}; corruption = change in A's read error.
    """
    K = cfg.interference_K
    centers, payloads, targets, _ = ball_setup(d, K, cfg)
    pay_A = payloads[: K - 1]
    c_A = centers[: K - 1]
    k_w, k_a, k_b = jax.random.split(jax.random.PRNGKey(seed + 333), 3)
    V = build_learned_V(d, K, cfg, k_w)

    # write items 0..K-2
    if arm_mode == "global":
        V, _ = train_memory_landscape(
            V, targets[: K - 1], k_a, steps=cfg.write_steps, lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay, loss_kwargs=_loss_kwargs(cfg, d),
        )
    else:
        for i in range(K - 1):
            k_a, kk = jax.random.split(k_a)
            V, _ = train_memory_landscape(
                V, targets[i : i + 1], kk, steps=cfg.local_write_steps, lr=cfg.write_lr,
                weight_decay=cfg.write_weight_decay, loss_kwargs=_loss_kwargs(cfg, d),
                update_mask_fn=atom_write_mask_fn(V.learned.group_rows(i)),
            )
    V_A = V
    m_A = clu_with_potential(
        V_A, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    before = score_cell(m_A, c_A, pay_A, cfg, d, seed)

    # write item K-1 (B) into its own site/block
    if arm_mode == "global":
        V_B, _ = train_memory_landscape(
            V_A, targets[K - 1 : K], k_b, steps=cfg.interference_write_steps,
            lr=cfg.write_lr, weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg, d),
        )
    else:
        V_B, _ = train_memory_landscape(
            V_A, targets[K - 1 : K], k_b, steps=cfg.interference_write_steps,
            lr=cfg.write_lr, weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg, d),
            update_mask_fn=atom_write_mask_fn(V_A.learned.group_rows(K - 1)),
        )
    m_B = clu_with_potential(
        V_B, dim=d + 1, kinetic_mode="newtonian_learned", inertia=jnp.ones(d + 1)
    )
    after = score_cell(m_B, c_A, pay_A, cfg, d, seed)

    # bit-level check: did the mask freeze the other atoms?
    la = jax.tree_util.tree_leaves(eqx.filter(V_A, eqx.is_inexact_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(V_B, eqx.is_inexact_array))
    delt = [np.asarray(y) - np.asarray(x) for x, y in zip(la, lb, strict=False)]
    n = int(sum(x.size for x in delt))
    nz = int(sum(int(np.count_nonzero(x)) for x in delt))
    return {
        "mode": arm_mode,
        "d": d,
        "seed": seed,
        "read_err_A_before": before["payload_abs_err_mean"],
        "read_err_A_after": after["payload_abs_err_mean"],
        "corruption": abs(
            after["payload_abs_err_mean"] - before["payload_abs_err_mean"]
        ),
        "strict_A_before": before["strict_success_rate"],
        "strict_A_after": after["strict_success_rate"],
        "frac_params_moved_by_B": float(nz / n) if n else 0.0,
    }


def item3_interference(cfg):
    """Masked vs global cross-write corruption at each d (does the write-operator
    advantage survive higher dimensions?)."""
    rows, summary = [], []
    for d in cfg.interference_dims:
        for mode in ("local", "global"):
            got = [_corruption(mode, d, s, cfg) for s in cfg.interference_seeds]
            rows.extend(got)
            summary.append(
                {
                    "d": d,
                    "mode": mode,
                    "corruption": _agg([g["corruption"] for g in got]),
                    "frac_params_moved_by_B": _agg(
                        [g["frac_params_moved_by_B"] for g in got]
                    ),
                    "strict_A_after": _agg([g["strict_A_after"] for g in got]),
                }
            )
    # local-advantage ratio per d
    adv = []
    for d in cfg.interference_dims:
        loc = next(s for s in summary if s["d"] == d and s["mode"] == "local")
        glo = next(s for s in summary if s["d"] == d and s["mode"] == "global")
        lc = max(loc["corruption"]["mean"], 1e-12)
        adv.append(
            {
                "d": d,
                "local_corruption": loc["corruption"]["mean"],
                "global_corruption": glo["corruption"]["mean"],
                "local_advantage_ratio": glo["corruption"]["mean"] / lc,
            }
        )
    return {
        "dims": list(cfg.interference_dims),
        "interference_K": cfg.interference_K,
        "seeds": list(cfg.interference_seeds),
        "summary": summary,
        "local_advantage_by_d": adv,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Item 4 -- the honest performance frontier
# ---------------------------------------------------------------------------


def item4_frontier(item1):
    """Where does learned content MATCH the designed ceiling, and where fall away?

    Derived from item 1: the (d, K) grid of learned strict, with the K where learned
    == designed and the largest K where learned still clears the bar (= K_learned).
    """
    frontier = []
    for lr in item1["learned_detail"]:
        d = lr["d"]
        dr = next(r for r in item1["designed_detail"] if r["d"] == d)
        frontier.append(
            {
                "d": d,
                "k_learned": lr["k_star"],
                "k_designed": dr["k_star"],
                "learned_strict_by_K": [
                    {"K": r["K"], "strict": r["strict"]["mean"], "passes": r["passes"]}
                    for r in lr["per_K"]
                ],
                "matches_designed_up_to_K": (
                    lr["k_star"] if lr["k_star"] >= dr["k_star"] else lr["k_star"]
                ),
                "falls_away_at_K": next(
                    (r["K"] for r in lr["per_K"] if not r["passes"]), None
                ),
            }
        )
    best = max(frontier, key=lambda f: f["k_learned"], default=None)
    return {
        "frontier": frontier,
        "best_learned_cell": (
            {"d": best["d"], "k_learned": best["k_learned"]} if best else None
        ),
    }


# ---------------------------------------------------------------------------
# Figure (local, per the exp_dim_scaling / exp_potential_class precedent)
# ---------------------------------------------------------------------------


def _plot(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    it1 = results.get("item1_discriminator")
    if not it1:
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    dl = [r["d"] for r in it1["k_learned_vs_d"]]
    kl = [max(r["k_star"], 0.5) for r in it1["k_learned_vs_d"]]
    kd = [max(r["k_star"], 0.5) for r in it1["k_designed_vs_d"]]
    a1.semilogy(dl, kl, "o-", lw=2, label="$K_{learned}$ (atom dict, trained)")
    a1.semilogy(dl, kd, "s-", lw=2, label="$K_{designed}$ (re-measured)")
    a1.semilogy(dl, [4 * 2**x for x in dl], "k--", alpha=0.5, label=r"$4\cdot2^d$")
    fit = it1.get("fit_k_learned", {})
    if fit.get("exponential_base_A"):
        A, b = fit["exponential_base_A"], fit["exponential_intercept"]
        xs = np.linspace(min(dl), max(dl), 40)
        a1.semilogy(
            xs, np.exp(b) * A**xs, "-", color="C0", alpha=0.4,
            label=f"fit $A^d$, A={A:.2f} ($R^2$={fit.get('exponential_r2', 0):.2f})",
        )
    a1.axhline(8, color="r", ls=":", lw=0.9, label="ring ceiling (~8)")
    a1.set_xlabel("address dimension $d$")
    a1.set_ylabel("$K$ cleared at strict 0.9")
    a1.set_title(f"Discriminator: {it1['verdict']}")
    a1.legend(fontsize=7)

    it3 = results.get("item3_interference")
    if it3:
        ds = [a["d"] for a in it3["local_advantage_by_d"]]
        a2.semilogy(
            ds, [max(a["global_corruption"], 1e-9) for a in it3["local_advantage_by_d"]],
            "o-", label="global write",
        )
        a2.semilogy(
            ds, [max(a["local_corruption"], 1e-9) for a in it3["local_advantage_by_d"]],
            "s-", label="masked (local) write",
        )
        a2.set_xlabel("address dimension $d$")
        a2.set_ylabel("corruption of A by writing B")
        a2.set_title("Interference: write operator across $d$")
        a2.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(save_dir, "designed_mechanism_fig1.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_designed_mechanism(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_designed_mechanism
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "R", "wall_margin", "well_width", "well_depth", "payload_kappa",
                "c_conf", "site_seed", "payload_seed", "dt", "gamma_address",
                "gamma_read", "address_steps", "read_steps", "tail_frac",
                "n_subsample", "n_query_per_item", "query_sigma", "query_sigma_p",
                "payload_tol", "pass_strict", "blank_strict_max", "blank_margin",
                "atoms_per_item", "atom_init_scale", "atom_init_width",
                "atom_depth_init", "learned_confine", "bits_per_param",
                "write_steps", "local_write_steps", "write_lr", "write_n_perturb",
                "write_sigma_addr", "write_sigma_pay", "write_margin",
                "write_barrier", "dims", "k_ladder", "k_cap", "learned_arm",
            )
        },
    }
    print("[item 1] discriminator: K_learned vs d", flush=True)
    results["item1_discriminator"] = item1_discriminator(cfg)
    print("[item 3] interference across d", flush=True)
    results["item3_interference"] = item3_interference(cfg)
    print("[item 2] mass arm + coupling check", flush=True)
    results["item2_mass"] = item2_mass(cfg)
    results["item4_frontier"] = item4_frontier(results["item1_discriminator"])

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_designed_mechanism_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover
        results["figures"] = []
        results["figure_error"] = repr(exc)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke: same code path, tiny sweeps. NOT shorter on the rollout budget
    than the designed reference needs (a smoke run where the reference fails prints
    a fake scientific negative — w20 lesson)."""
    cfg = config.experiment_designed_mechanism
    cfg.dims = [2, 3]
    cfg.k_ladder = [2, 4, 8]
    cfg.k_cap = 8
    cfg.discriminator_seeds = [0, 1]
    cfg.designed_seeds = [0]
    cfg.atoms_per_item = 8
    cfg.address_steps = 300
    cfg.read_steps = 200
    cfg.write_steps = 60
    cfg.local_write_steps = 40
    cfg.write_n_perturb = 8
    cfg.n_query_per_item = 8
    cfg.interference_dims = [2]
    cfg.interference_seeds = [0]
    cfg.interference_write_steps = 30
    cfg.mass_dim = 2
    cfg.mass_K = 4
    cfg.mass_seeds = [0]


def _replace(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment DESIGNED-MECHANISM-LEARNED-CONTENT: is the K=8 wall "
        "geometry or learning?"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
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

    res = run_experiment_designed_mechanism(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(json.dumps(res["item1_discriminator"]["k_learned_vs_d"], indent=2))
    print(json.dumps(res["item1_discriminator"]["k_designed_vs_d"], indent=2))
    print("verdict:", res["item1_discriminator"]["verdict"])
    print("metrics ->", res["metrics_path"])


if __name__ == "__main__":
    main()
