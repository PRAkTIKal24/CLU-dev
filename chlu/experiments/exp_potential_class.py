"""Experiment POTENTIAL-CLASS (w21, Task D): is the learned-landscape failure
**EXPRESSIVITY** or **SUPPORT STRUCTURE**?

w20 (``exp_learned_memory``) measured that a learned landscape loses everything a
designed one provides: no learned rung clears strict 0.9 at both K=4 and K=8 over
5 seeds, write locality collapses from 0.000 to 2.9e-2 .. 5.0e-1, and one
subsequent write destroys the best rung (strict 1.000 -> 0.000). **Every w20
learned rung used ``PotentialMLP``.** Two hypotheses explain that, and they make
OPPOSITE predictions on one arm:

============  =======================================  ==============================
hypothesis    prediction for a TRANSFORMER potential   prediction for an ATOM dict.
============  =======================================  ==============================
**H-EXPR**    fixes it (more capacity)                 also fixes it
**H-SUPP**    fails, possibly worse (attention is      fixes it (atom writes are
              *more* global than an MLP)               local by construction)
============  =======================================  ==============================

So the **transformer arm is the discriminator**, and this module runs the w20
protocol with the potential's *function class* as the swept variable, everything
else pinned at the w20 values.

Arms (all matched to ``param_target`` learned parameters, tolerance
``param_tol``; the match table is emitted with the results and asserted):

``designed``        rung 0, zero learned parameters -- the ceiling.
``mlp``             the w20 baseline. Global support.
``hopfield``        ⭐ ``V = -(1/b) logsumexp(b <q,k_i>) + a|q|^2`` -- EXACTLY the
                    modern-Hopfield energy (Ramsauer), i.e. attention over a
                    learned memory codebook. Run at two temperatures
                    (``hopfield``/``hopfield_sharp``) at IDENTICAL capacity, which
                    is the clean separation of capacity from support.
``attn``            single-head cross-attention with a learned query projection
                    and a free scalar value head -- so a negative transformer
                    result cannot be dismissed as "not a real attention layer".
``atoms``           the theorist's ``AtomDictionaryPotential`` (MVC-0 substrate),
                    written by an ordinary GLOBAL gradient step.
``atoms_local``     the same class written by K per-item MASKED writes: every
                    other item's atoms come out of the write bit-identical. This
                    is the only arm whose *write operator* is local, and the pair
                    (``atoms``, ``atoms_local``) separates "local basis" from
                    "local write".

⚠ Construction (a) of the task's transformer arm (treat the d coordinates as d
tokens and self-attend) is **NOT** run: at ``dim=3`` a 3-token self-attention is
degenerate, and the task requires only (b). Both (b)-family variants above are
attention over a learned memory codebook.

Measurements (each at >= 5 seeds, each with a blank control over the STRONGEST
read in use -- w20's method finding: a cell without a passing blank is not a
measurement):

1. ``item1_class_sweep``     strict retrieval + blank controls per class, K in {4,8}.
2. ``item2_interference``    ⭐ write A, write B, re-read A. The discriminator.
3. ``item3_support_radius``  perturb theta by ONE write and measure ||grad dV(q)||
                             vs distance from the write site -- the mechanism
                             behind (2), measured rather than assumed.
4. ``item4_ladder_rerun``    w20's design-freedom ladder re-run with the best
                             family: does the minimum-viable-design point move?
5. cost: params, wall-clock, analytic FLOPs/eval per class.

Runnable directly:
    uv run python -m chlu.experiments.exp_potential_class --quick
or via the CLI: ``chlu exp-potential-class [--project N] [--seed I] [--quick]``.
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
from chlu.core.memory_potentials import (
    DESIGN_RUNGS,
    DesignFreedomPotential,
    atom_write_mask_fn,
    designed_payloads,
    ring_sites,
)
from chlu.experiments.exp_learned_memory import (
    _blank_ok,
    _n_params,
    evaluate_cell,
    model_for,
)
from chlu.training.train_memory import train_memory_landscape

# ---------------------------------------------------------------------------
# Arm table: name -> (rung, learned family, write mode)
# ---------------------------------------------------------------------------

#: ``write`` is "global" (one Adam run over all learned parameters, exactly what
#: w20 did) or "local" (one masked single-item write per item -- only meaningful
#: for a family whose parameters are indexed by item, i.e. the atom dictionary).
ARM_TABLE = {
    "designed": dict(rung="designed", family="mlp", write="global", beta=None),
    "mlp": dict(rung="free_mlp", family="mlp", write="global", beta=None),
    "hopfield": dict(rung="free_mlp", family="hopfield", write="global", beta="soft"),
    "hopfield_sharp": dict(
        rung="free_mlp", family="hopfield", write="global", beta="sharp"
    ),
    "attn": dict(rung="free_mlp", family="attn", write="global", beta="attn"),
    "atoms": dict(rung="free_mlp", family="atoms", write="global", beta=None),
    "atoms_local": dict(rung="free_mlp", family="atoms", write="local", beta=None),
}


def arm_spec(arm: str) -> dict:
    if arm not in ARM_TABLE:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARM_TABLE)}")
    return ARM_TABLE[arm]


def build_arm(arm: str, cfg, payloads, key, dim: int = 3, K: Optional[int] = None):
    """A ``DesignFreedomPotential`` for ``arm`` with the w20 geometry.

    ``K`` is the number of item slots; it only matters for ``atoms_local``, where
    the dictionary is partitioned into K contiguous atom blocks (one per item).
    """
    spec = arm_spec(arm)
    beta = {
        "soft": cfg.hopfield_beta_soft,
        "sharp": cfg.hopfield_beta_sharp,
        "attn": cfg.attn_beta,
        None: 0.0,
    }[spec["beta"]]
    key_init = cfg.attn_key_init if spec["family"] == "attn" else cfg.hopfield_key_init
    n_mem = cfg.attn_n_mem if spec["family"] == "attn" else cfg.hopfield_n_mem
    return DesignFreedomPotential(
        rung=spec["rung"],
        dim=dim,
        payloads=payloads,
        key=key,
        lam=cfg.lam,
        f=cfg.f,
        barrier=cfg.barrier,
        payload_kappa=cfg.payload_kappa,
        bump_width=cfg.bump_width,
        hidden=cfg.hidden,
        n_atoms=cfg.n_atoms,
        rbf_init_width=cfg.atom_init_width,
        learned_family=spec["family"],
        n_mem=n_mem,
        beta=beta,
        d_head=cfg.attn_d_head,
        hopfield_confine=cfg.hopfield_confine,
        confine=cfg.learned_confine,
        key_init=key_init,
        atom_depth_init=cfg.atom_depth_init,
        atom_groups=max(1, int(K or 1)),
        atom_init_scale=cfg.atom_init_scale,
    )


def _loss_kwargs(cfg):
    return dict(
        n_perturb=cfg.write_n_perturb,
        sigma_addr=cfg.write_sigma_addr,
        sigma_pay=cfg.write_sigma_pay,
        margin=cfg.write_margin,
        barrier=cfg.write_barrier,
    )


def write_arm(
    V,
    arm: str,
    cfg,
    targets,
    key,
    steps: Optional[int] = None,
    item_ids=None,
):
    """Write ``targets`` into ``V`` under the arm's WRITE OPERATOR.

    * ``write="global"`` -- one Adam run over every learned parameter, jointly on
      all targets. Exactly the w20 write.
    * ``write="local"``  -- one masked Adam run per target, each seeing ONLY its
      own target (no barrier term, no sight of the other items) and updating ONLY
      its own atom block. This is the MVC-0 write: if the substrate really is
      C3-local, a later write must leave earlier items bit-identical, and this is
      the operator that tests it. ``item_ids`` gives the atom block index of each
      target (defaults to ``0..K-1``), so the interference test can write item
      ``K-1`` into its own block without touching blocks ``0..K-2``.

    Returns ``(V, history, n_write_steps)``.
    """
    spec = arm_spec(arm)
    targets = jnp.asarray(targets)
    if getattr(V, "learned", None) is None:  # the designed rung: nothing to write
        return V, [], 0

    if spec["write"] == "global":
        n = cfg.write_steps if steps is None else steps
        V, hist = train_memory_landscape(
            V,
            targets,
            key,
            steps=n,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg),
        )
        return V, hist, n

    if not hasattr(V.learned, "group_rows"):
        raise TypeError(
            f"arm {arm!r} asks for a LOCAL write but its learned family "
            f"({type(V.learned).__name__}) has no atom-block structure; a local "
            "write is only defined for AtomDictionaryPotential."
        )
    n = cfg.local_write_steps if steps is None else steps
    ids = list(range(targets.shape[0])) if item_ids is None else list(item_ids)
    hist, total = [], 0
    for j, gid in enumerate(ids):
        key, k = jax.random.split(key)
        mask = V.learned.group_rows(int(gid))
        V, h = train_memory_landscape(
            V,
            targets[j : j + 1],
            k,
            steps=n,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg),
            update_mask_fn=atom_write_mask_fn(mask),
        )
        hist.extend(h)
        total += n
    return V, hist, total


# ---------------------------------------------------------------------------
# Matched-parameter table
# ---------------------------------------------------------------------------


def flops_per_eval(arm: str, cfg, dim: int = 3) -> int:
    """Analytic multiply-accumulate count for ONE ``V(q)`` evaluation.

    Reported instead of a profiler number because it is exactly reproducible and
    because at ``dim=3`` every arm is bound by one small dense op.
    """
    spec = arm_spec(arm)
    fam = spec["family"] if spec["rung"] != "designed" else "none"
    if fam == "none":
        return 0
    if fam == "mlp":
        h = cfg.hidden
        return dim * h + h * h + h
    if fam == "atoms":
        return cfg.n_atoms * (2 * dim + 2)  # (q-c)^2 + exp + scale
    if fam == "hopfield":
        return cfg.hopfield_n_mem * (dim + 2)  # dot + bias + logsumexp
    return cfg.attn_d_head * dim + cfg.attn_n_mem * (cfg.attn_d_head + 3)


def param_match_table(cfg, dim: int = 3, K: int = 4) -> dict:
    """Learned-parameter count per arm, measured by construction (not by formula).

    ⚠ An unmatched comparison settles nothing, so this is built and checked
    BEFORE any measurement and travels with the results.
    """
    rows = []
    pay = designed_payloads(K, seed=cfg.payload_seed)
    for arm in cfg.potential_classes:
        V = build_arm(arm, cfg, pay, jax.random.PRNGKey(0), dim=dim, K=K)
        n = _n_params(V)
        rel = None if n == 0 else abs(n - cfg.param_target) / cfg.param_target
        rows.append(
            {
                "arm": arm,
                "rung": arm_spec(arm)["rung"],
                "family": V.learned_family,
                "write": arm_spec(arm)["write"],
                "n_learned_params": n,
                "rel_dev_from_target": rel,
                "within_tolerance": (n == 0) or (rel <= cfg.param_tol),
                "flops_per_eval": flops_per_eval(arm, cfg, dim),
            }
        )
    return {
        "param_target": cfg.param_target,
        "param_tol": cfg.param_tol,
        "rows": rows,
        "all_within_tolerance": all(r["within_tolerance"] for r in rows),
    }


# ---------------------------------------------------------------------------
# Item 1 -- per-class fidelity + blank controls, >= 5 seeds
# ---------------------------------------------------------------------------


def _cell(arm, cfg, K, seed, dim=3):
    """Train the WRITTEN and the BLANK landscape for one (arm, K, seed) and score.

    The blank is the identical architecture trained by the identical write with
    ALL payloads zero. Anything a read can still recover from it is address, not
    content (w20: under a learned V a nearest-centroid blank scores 0.992-1.000).
    """
    pay = designed_payloads(K, seed=cfg.payload_seed)
    k_w, k_b, k_tw, k_tb = jax.random.split(jax.random.PRNGKey(seed), 4)
    sites = ring_sites(K, f=cfg.f, dim=dim, payloads=pay)
    blank_pay = jnp.zeros_like(jnp.asarray(pay))
    blank_sites = ring_sites(K, f=cfg.f, dim=dim, payloads=blank_pay)

    t0 = time.perf_counter()
    Vw = build_arm(arm, cfg, pay, k_w, dim=dim, K=K)
    Vw, hist, n_steps = write_arm(Vw, arm, cfg, sites, k_tw)
    write_seconds = time.perf_counter() - t0

    Vb = build_arm(arm, cfg, blank_pay, k_b, dim=dim, K=K)
    Vb, _, _ = write_arm(Vb, arm, cfg, blank_sites, k_tb)

    written = evaluate_cell(model_for(Vw, dim), cfg, pay, seed)
    blank = evaluate_cell(model_for(Vb, dim), cfg, pay, seed)
    class_ok, value_ok = _blank_ok(written, blank, cfg)
    return {
        "arm": arm,
        "K": K,
        "seed": seed,
        "written": written,
        "blank": blank,
        "blank_control_passes_classification": class_ok,
        "blank_control_passes_value": value_ok,
        "n_learned_params": _n_params(Vw),
        "write_loss_initial": hist[0] if hist else None,
        "write_loss_final": hist[-1] if hist else None,
        "write_steps_total": n_steps,
        "write_seconds": write_seconds,
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


def item1_class_sweep(cfg, dim: int = 3):
    """Strict retrieval per class at each K, aggregated over seeds, with blanks."""
    cells, summary = [], []
    for arm in cfg.potential_classes:
        for K in cfg.class_item_counts:
            got = [_cell(arm, cfg, K, s, dim=dim) for s in cfg.class_seeds]
            cells.extend(got)
            strict = [c["written"]["strict_success_rate"] for c in got]
            basin = [c["written"]["basin_success_rate"] for c in got]
            err = [c["written"]["payload_abs_err_mean"] for c in got]
            summary.append(
                {
                    "arm": arm,
                    "K": K,
                    "n_learned_params": got[0]["n_learned_params"],
                    "strict": _agg(strict),
                    "basin": _agg(basin),
                    "payload_abs_err": _agg(err),
                    "blank_strict": _agg(
                        [c["blank"]["strict_success_rate"] for c in got]
                    ),
                    "blank_classification_max": _agg(
                        [
                            max(
                                c["blank"]["acc_payload_codebook_read"],
                                c["blank"]["acc_payload_nearest_centroid"],
                            )
                            for c in got
                        ]
                    ),
                    "n_blank_value_pass": int(
                        sum(c["blank_control_passes_value"] for c in got)
                    ),
                    "n_blank_classification_pass": int(
                        sum(c["blank_control_passes_classification"] for c in got)
                    ),
                    "write_seconds": _agg([c["write_seconds"] for c in got]),
                }
            )

    # PRIMARY criterion, leak-immune: mean strict >= pass_strict at EVERY K, and
    # every cell's VALUE blank control passes. A cell whose value blank fails is
    # not a measurement and disqualifies the arm (reported, not silently dropped).
    verdict = []
    for arm in cfg.potential_classes:
        rows = [s for s in summary if s["arm"] == arm]
        all_blank_ok = all(s["n_blank_value_pass"] == s["strict"]["n"] for s in rows)
        clears = all(s["strict"]["mean"] >= cfg.pass_strict for s in rows)
        verdict.append(
            {
                "arm": arm,
                "clears_bar_at_every_K": bool(clears and all_blank_ok),
                "min_mean_strict_over_K": float(min(s["strict"]["mean"] for s in rows)),
                "value_blank_ok_everywhere": bool(all_blank_ok),
                "classification_blank_ok_everywhere": bool(
                    all(
                        s["n_blank_classification_pass"] == s["strict"]["n"]
                        for s in rows
                    )
                ),
            }
        )
    return {
        "pass_strict": cfg.pass_strict,
        "seeds": list(cfg.class_seeds),
        "item_counts": list(cfg.class_item_counts),
        "summary": summary,
        "verdict": verdict,
        "any_learned_class_clears_bar": bool(
            any(
                v["clears_bar_at_every_K"]
                for v in verdict
                if v["arm"] != cfg.reference_class
            )
        ),
        "cells": cells,
        "w20_baseline": dict(cfg.w20_baseline),
    }


# ---------------------------------------------------------------------------
# Item 2 -- ⭐ cross-write interference (the H-EXPR / H-SUPP discriminator)
# ---------------------------------------------------------------------------


def _write_A_then_B(arm, cfg, seed, dim=3):
    """Write A (K-1 items), snapshot; write B at the fresh site; return both Vs."""
    K = cfg.interference_K
    pay_all = designed_payloads(K, seed=cfg.payload_seed)
    sites_all = ring_sites(K, f=cfg.f, dim=dim, payloads=pay_all)
    k_v, k_a, k_b = jax.random.split(jax.random.PRNGKey(seed), 3)

    V = build_arm(arm, cfg, pay_all, k_v, dim=dim, K=K)
    V_A, _, _ = write_arm(V, arm, cfg, sites_all[:-1], k_a, item_ids=list(range(K - 1)))
    # B goes into the LAST site -- and, for a local write, into its own atom block.
    V_B, _, _ = write_arm(
        V_A,
        arm,
        cfg,
        sites_all[-1:],
        k_b,
        steps=cfg.interference_write_steps,
        item_ids=[K - 1],
    )
    return V_A, V_B, pay_all[:-1], K


def _params_changed(V_before, V_after) -> dict:
    """How much of theta moved, and did ANY frozen parameter move at all.

    The task's adversarial check: an ``atoms_local`` result is only meaningful if
    the mask really froze the other items' atoms, so this is a bit-level
    parameter comparison, not a loss check.
    """
    # Only the LEARNED subtree: the designed part is frozen by construction, and
    # including it would dilute frac_changed with parameters no write can move.
    lb = getattr(V_before, "learned", None)
    la = getattr(V_after, "learned", None)
    if lb is None or la is None:
        return {
            "n_params": 0,
            "n_changed": 0,
            "frac_changed": 0.0,
            "max_abs_delta": 0.0,
        }
    a = jax.tree_util.tree_leaves(eqx.filter(lb, eqx.is_inexact_array))
    b = jax.tree_util.tree_leaves(eqx.filter(la, eqx.is_inexact_array))
    if not a:
        return {
            "n_params": 0,
            "n_changed": 0,
            "frac_changed": 0.0,
            "max_abs_delta": 0.0,
        }
    d = [np.asarray(y) - np.asarray(x) for x, y in zip(a, b, strict=False)]
    n = int(sum(x.size for x in d))
    nz = int(sum(int(np.count_nonzero(x)) for x in d))
    return {
        "n_params": n,
        "n_changed": nz,
        "frac_changed": float(nz / n),
        "max_abs_delta": float(max(float(np.abs(x).max()) for x in d)),
    }


def item2_interference(cfg, dim: int = 3):
    """Write A, then write B, re-read A. Corruption = the change in A's read-out.

    Reference points carried in the output: designed 0.000 (w19/w20) and MLP
    2.9e-2 .. 5.0e-1 (w20). This is the measurement on which H-EXPR and H-SUPP
    make opposite predictions for the attention arms.
    """
    rows, summary = [], []
    for arm in cfg.potential_classes:
        per_seed = []
        for seed in cfg.interference_seeds:
            V_A, V_B, pay_A, K = _write_A_then_B(arm, cfg, seed, dim=dim)
            before = evaluate_cell(
                model_for(V_A, dim), cfg, pay_A, seed, dim=dim, n_sites=K
            )
            after = evaluate_cell(
                model_for(V_B, dim), cfg, pay_A, seed, dim=dim, n_sites=K
            )
            moved = _params_changed(V_A, V_B)
            row = {
                "arm": arm,
                "seed": seed,
                "K_A": int(len(pay_A)),
                "read_err_A_before_B": before["payload_abs_err_mean"],
                "read_err_A_after_B": after["payload_abs_err_mean"],
                "corruption_of_A_by_writing_B": abs(
                    after["payload_abs_err_mean"] - before["payload_abs_err_mean"]
                ),
                "strict_A_before_B": before["strict_success_rate"],
                "strict_A_after_B": after["strict_success_rate"],
                "strict_drop": before["strict_success_rate"]
                - after["strict_success_rate"],
                "theta_moved_by_write_B": moved,
            }
            rows.append(row)
            per_seed.append(row)
        summary.append(
            {
                "arm": arm,
                "corruption": _agg(
                    [r["corruption_of_A_by_writing_B"] for r in per_seed]
                ),
                "strict_drop": _agg([r["strict_drop"] for r in per_seed]),
                "strict_A_before_B": _agg([r["strict_A_before_B"] for r in per_seed]),
                "strict_A_after_B": _agg([r["strict_A_after_B"] for r in per_seed]),
                "frac_theta_moved_by_B": _agg(
                    [r["theta_moved_by_write_B"]["frac_changed"] for r in per_seed]
                ),
            }
        )
    return {
        "seeds": list(cfg.interference_seeds),
        "interference_K": cfg.interference_K,
        "w20_designed_corruption": 0.0,
        "w20_free_mlp_corruption": cfg.w20_baseline["free_mlp_corruption"],
        "rows": rows,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Item 3 -- support radius, MEASURED
# ---------------------------------------------------------------------------


def _sphere(key, n: int, dim: int, r: float):
    v = jax.random.normal(key, (n, dim))
    return r * v / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def item3_support_radius(cfg, dim: int = 3):
    """Perturb theta by a SINGLE write and measure how far the change reaches.

    Primary statistic: ``rms||grad V_after(q) - grad V_before(q)||`` over probe
    points on a sphere of radius ``r`` about the newly written site, normalised
    by its value at the smallest probed radius. The FORCE, not the raw ``V``,
    because ``V`` is defined only up to a constant and only ``grad V`` enters the
    dynamics; ``std(dV)`` over the shell is reported alongside as a secondary,
    offset-free scalar column.

    ⚠ The curve is normalised by its **peak over the radius grid, not by its
    value at the smallest radius**. For a localized write the influence function
    is bump-shaped, not monotone: a Gaussian atom centred exactly on the site has
    ``grad dV = 0`` AT the site and peaks at ``r ~ s``. Normalising at ``r->0``
    therefore divides by a near-zero and reports ratios of 10^3 (observed in the
    smoke run). ``r_10`` is accordingly the first radius **beyond the peak** at
    which the normalised curve falls below ``support_decay_threshold``.

    This is the mechanism behind item 2.
    """
    out = []
    for arm in cfg.potential_classes:
        curves, r10s, r_peaks = [], [], []
        for seed in cfg.support_seeds:
            V_A, V_B, _, K = _write_A_then_B(arm, cfg, seed, dim=dim)
            site_B = np.asarray(ring_sites(K, f=cfg.f, dim=dim))[K - 1]
            gA = eqx.filter_jit(jax.grad(lambda q, m=V_A: m(q)))
            gB = eqx.filter_jit(jax.grad(lambda q, m=V_B: m(q)))
            key = jax.random.PRNGKey(1000 + seed)
            curve = []
            for r in cfg.support_radii:
                key, k = jax.random.split(key)
                pts = jnp.asarray(site_B)[None, :] + _sphere(
                    k, cfg.support_probes_per_radius, dim, float(r)
                )
                dg = jax.vmap(gB)(pts) - jax.vmap(gA)(pts)
                dv = jax.vmap(V_B)(pts) - jax.vmap(V_A)(pts)
                curve.append(
                    {
                        "r": float(r),
                        "rms_grad_dV": float(
                            np.sqrt(np.mean(np.sum(np.asarray(dg) ** 2, axis=1)))
                        ),
                        "std_dV": float(np.std(np.asarray(dv))),
                        "mean_abs_dV": float(np.mean(np.abs(np.asarray(dv)))),
                    }
                )
            gpk = max(c["rms_grad_dV"] for c in curve)
            i_pk = int(np.argmax([c["rms_grad_dV"] for c in curve]))
            for c in curve:
                c["rel_grad_dV"] = float(c["rms_grad_dV"] / gpk) if gpk > 0 else 0.0
            r10 = None
            if gpk > 0:
                for c in curve[i_pk:]:
                    if c["rel_grad_dV"] < cfg.support_decay_threshold:
                        r10 = c["r"]
                        break
            curves.append(curve)
            r10s.append(r10 if r10 is not None else float("nan"))
            r_peaks.append(curve[i_pk]["r"] if gpk > 0 else float("nan"))
        # average the curves across seeds on the shared radius grid
        mean_curve = []
        for i, r in enumerate(cfg.support_radii):
            mean_curve.append(
                {
                    "r": float(r),
                    "rel_grad_dV": float(
                        np.mean([c[i]["rel_grad_dV"] for c in curves])
                    ),
                    "rms_grad_dV": float(
                        np.mean([c[i]["rms_grad_dV"] for c in curves])
                    ),
                    "std_dV": float(np.mean([c[i]["std_dV"] for c in curves])),
                }
            )
        finite_r10 = [x for x in r10s if np.isfinite(x)]
        finite_pk = [x for x in r_peaks if np.isfinite(x)]
        out.append(
            {
                "arm": arm,
                "seeds": list(cfg.support_seeds),
                "threshold": cfg.support_decay_threshold,
                "mean_curve": mean_curve,
                "r10_per_seed": r10s,
                "r10_mean": float(np.mean(finite_r10)) if finite_r10 else None,
                "r10_unresolved_seeds": int(len(r10s) - len(finite_r10)),
                "r_peak_mean": float(np.mean(finite_pk)) if finite_pk else None,
                "rel_at_max_radius": mean_curve[-1]["rel_grad_dV"],
                "abs_peak_rms_grad_dV": float(
                    np.mean([max(c["rms_grad_dV"] for c in cur) for cur in curves])
                ),
            }
        )
    return {"radii": list(cfg.support_radii), "arms": out}


# ---------------------------------------------------------------------------
# Item 4 -- does the design-freedom curve move?
# ---------------------------------------------------------------------------


def _ladder_cell(rung, family, cfg, K, seed, dim=3):
    """One design-freedom rung instantiated with an arbitrary learned family."""
    pay = designed_payloads(K, seed=cfg.payload_seed)
    k_w, k_b, k_tw, k_tb = jax.random.split(jax.random.PRNGKey(seed), 4)
    blank_pay = jnp.zeros_like(jnp.asarray(pay))

    def _build(p, key):
        return DesignFreedomPotential(
            rung=rung,
            dim=dim,
            payloads=p,
            key=key,
            lam=cfg.lam,
            f=cfg.f,
            barrier=cfg.barrier,
            payload_kappa=cfg.payload_kappa,
            bump_width=cfg.bump_width,
            hidden=cfg.hidden,
            n_atoms=cfg.n_atoms,
            rbf_init_width=cfg.atom_init_width,
            learned_family=family,
            n_mem=cfg.attn_n_mem if family == "attn" else cfg.hopfield_n_mem,
            beta=cfg.attn_beta if family == "attn" else cfg.hopfield_beta_sharp,
            d_head=cfg.attn_d_head,
            hopfield_confine=cfg.hopfield_confine,
            confine=cfg.learned_confine,
            key_init=(cfg.attn_key_init if family == "attn" else cfg.hopfield_key_init),
            atom_depth_init=cfg.atom_depth_init,
            atom_groups=K,
            atom_init_scale=cfg.atom_init_scale,
        )

    def _write(V, p, key):
        return train_memory_landscape(
            V,
            ring_sites(K, f=cfg.f, dim=dim, payloads=p),
            key,
            steps=cfg.write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=_loss_kwargs(cfg),
        )[0]

    Vw = _write(_build(pay, k_w), pay, k_tw)
    Vb = _write(_build(blank_pay, k_b), blank_pay, k_tb)
    written = evaluate_cell(model_for(Vw, dim), cfg, pay, seed)
    blank = evaluate_cell(model_for(Vb, dim), cfg, pay, seed)
    class_ok, value_ok = _blank_ok(written, blank, cfg)
    return {
        "rung": rung,
        "family": family,
        "design_freedom": DESIGN_RUNGS.index(rung),
        "K": K,
        "seed": seed,
        "strict": written["strict_success_rate"],
        "basin": written["basin_success_rate"],
        "payload_abs_err": written["payload_abs_err_mean"],
        "blank_strict": blank["strict_success_rate"],
        "blank_classification_max": max(
            blank["acc_payload_codebook_read"], blank["acc_payload_nearest_centroid"]
        ),
        "blank_passes_value": value_ok,
        "blank_passes_classification": class_ok,
        "n_learned_params": _n_params(Vw),
    }


def item4_ladder_rerun(cfg, family: str, dim: int = 3):
    """w20's design-freedom ladder, re-run with ``family`` as the learned class.

    w20's headline was "minimum designed structure = essentially all of it", and
    every one of its learned rungs was an MLP. If a better function class moves
    the minimum-viable-design point, that headline was a property of the class;
    if it does not, the headline hardens into a program-level statement.
    """
    rows, curve = [], []
    for rung in cfg.ladder_rungs:
        per_K = []
        for K in cfg.ladder_item_counts:
            got = [
                _ladder_cell(rung, family, cfg, K, s, dim=dim) for s in cfg.ladder_seeds
            ]
            rows.extend(got)
            per_K.append(
                {
                    "K": K,
                    "strict": _agg([g["strict"] for g in got]),
                    "basin": _agg([g["basin"] for g in got]),
                    "n_blank_value_pass": int(
                        sum(g["blank_passes_value"] for g in got)
                    ),
                    "n_blank_classification_pass": int(
                        sum(g["blank_passes_classification"] for g in got)
                    ),
                }
            )
        curve.append(
            {
                "rung": rung,
                "family": family,
                "design_freedom": DESIGN_RUNGS.index(rung),
                "n_learned_params": rows[-1]["n_learned_params"],
                "per_K": per_K,
                "min_mean_strict": float(min(p["strict"]["mean"] for p in per_K)),
                "value_blank_ok_everywhere": bool(
                    all(p["n_blank_value_pass"] == len(cfg.ladder_seeds) for p in per_K)
                ),
                "passes_value_criterion": bool(
                    all(p["strict"]["mean"] >= cfg.pass_strict for p in per_K)
                    and all(
                        p["n_blank_value_pass"] == len(cfg.ladder_seeds) for p in per_K
                    )
                ),
            }
        )
    passing = [c for c in curve if c["passes_value_criterion"]]
    best = max(passing, key=lambda c: c["design_freedom"]) if passing else None
    return {
        "family": family,
        "seeds": list(cfg.ladder_seeds),
        "item_counts": list(cfg.ladder_item_counts),
        "pass_strict": cfg.pass_strict,
        "curve": curve,
        "rows": rows,
        "minimum_viable_design_rung": best["rung"] if best else None,
        "minimum_viable_design_freedom": best["design_freedom"] if best else None,
        "w20_minimum_viable_design_rung": None,
        "w20_note": (
            "w20 (MLP family): NO learned rung cleared 0.9 at both K=4 and K=8; "
            "skeleton_residual was closest at 0.903+-0.101 / 0.959+-0.043."
        ),
    }


# ---------------------------------------------------------------------------
# Figures (local, following the exp_learned_memory / exp_paid_access precedent)
# ---------------------------------------------------------------------------


def _plot_all(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []
    s1 = results.get("item1_class_sweep")
    s2 = results.get("item2_interference")
    s3 = results.get("item3_support_radius")
    if not (s1 and s2 and s3):
        return paths
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    arms = [v["arm"] for v in s1["verdict"]]

    a = axes[0]
    width = 0.38
    x = np.arange(len(arms))
    for i, K in enumerate(s1["item_counts"]):
        vals = [
            next(
                (
                    s["strict"]["mean"]
                    for s in s1["summary"]
                    if s["arm"] == arm and s["K"] == K
                ),
                np.nan,
            )
            for arm in arms
        ]
        errs = [
            next(
                (
                    s["strict"]["std"]
                    for s in s1["summary"]
                    if s["arm"] == arm and s["K"] == K
                ),
                np.nan,
            )
            for arm in arms
        ]
        a.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=2, label=f"K={K}")
    a.axhline(s1["pass_strict"], color="r", ls="--", lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels(arms, rotation=25, ha="right", fontsize=7)
    a.set_ylabel("strict retrieval")
    a.set_title(f"Fidelity by function class ({len(s1['seeds'])} seeds)")
    a.legend(fontsize=7)

    a = axes[1]
    corr = [
        next(
            (s["corruption"]["mean"] for s in s2["summary"] if s["arm"] == arm), np.nan
        )
        for arm in arms
    ]
    a.bar(x, np.maximum(corr, 1e-8), color="tab:orange")
    a.set_yscale("log")
    a.set_xticks(x)
    a.set_xticklabels(arms, rotation=25, ha="right", fontsize=7)
    a.axhline(
        max(s2["w20_free_mlp_corruption"], 1e-8),
        color="k",
        ls=":",
        lw=0.9,
        label="w20 free_mlp",
    )
    a.set_ylabel("corruption of A by writing B")
    a.set_title("Cross-write interference (log)")
    a.legend(fontsize=7)

    a = axes[2]
    for entry in s3["arms"]:
        r = [c["r"] for c in entry["mean_curve"]]
        y = [max(c["rel_grad_dV"], 1e-12) for c in entry["mean_curve"]]
        if max(y) <= 1e-11:
            continue
        a.plot(r, y, "o-", ms=3, lw=1.0, label=entry["arm"])
    a.set_yscale("log")
    a.set_xlabel("distance from write site")
    a.set_ylabel("||grad dV|| (normalised)")
    a.axhline(s3["arms"][0]["threshold"], color="r", ls="--", lw=0.8)
    a.set_title("Measured support radius")
    a.legend(fontsize=6)

    fig.tight_layout()
    p = os.path.join(save_dir, "potential_class_fig1_class_interference_support.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_potential_class(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Run the matched-parameter check + items 1-4 and write JSON + figures."""
    config = config or get_default_config()
    cfg = config.experiment_potential_class
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "potential_classes": list(cfg.potential_classes),
        "config": {
            k: getattr(cfg, k)
            for k in (
                "lam",
                "f",
                "barrier",
                "payload_kappa",
                "bump_width",
                "dt",
                "gamma_address",
                "gamma_read",
                "address_steps",
                "read_steps",
                "tail_frac",
                "n_subsample",
                "n_query_per_item",
                "query_sigma_theta",
                "query_sigma_p",
                "payload_tol",
                "payload_seed",
                "write_steps",
                "local_write_steps",
                "write_lr",
                "write_margin",
                "write_barrier",
                "write_sigma_addr",
                "write_sigma_pay",
                "hidden",
                "n_atoms",
                "hopfield_n_mem",
                "hopfield_beta_soft",
                "hopfield_beta_sharp",
                "hopfield_confine",
                "attn_n_mem",
                "attn_d_head",
                "attn_beta",
                "atom_depth_init",
                "pass_strict",
                "blank_margin",
                "blank_strict_max",
                "param_target",
                "param_tol",
            )
        },
    }
    # ⚠ matched parameters FIRST: an unmatched comparison settles nothing, so the
    # table is built (and its verdict recorded) before any measurement runs.
    results["param_match"] = param_match_table(
        cfg, K=cfg.class_item_counts[0] if cfg.class_item_counts else 4
    )
    results["item1_class_sweep"] = item1_class_sweep(cfg)
    results["item2_interference"] = item2_interference(cfg)
    results["item3_support_radius"] = item3_support_radius(cfg)

    if cfg.run_ladder_rerun:
        family = cfg.ladder_family
        if family == "auto":
            learned = [
                v
                for v in results["item1_class_sweep"]["verdict"]
                if v["arm"] != cfg.reference_class
            ]
            best_arm = max(learned, key=lambda v: v["min_mean_strict_over_K"])["arm"]
            family = arm_spec(best_arm)["family"]
            results["ladder_family_auto_selected_from"] = best_arm
        results["item4_ladder_rerun"] = item4_ladder_rerun(cfg, family)

    # Metrics are written BEFORE plotting: a plotting bug must never destroy a
    # completed run (it did once in w20).
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_potential_class_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    try:
        results["figures"] = _plot_all(results, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        results["figures"] = []
        results["figure_error"] = repr(exc)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings -- same code path, smaller sweeps.

    ⚠ NOT shorter than this on the rollout budget: the settling budget is what
    makes the DESIGNED reference work, and a smoke run in which the reference
    fails prints what looks like a scientific negative (w20 lesson).
    """
    cfg = config.experiment_potential_class
    cfg.potential_classes = ["designed", "mlp", "atoms_local"]
    cfg.class_item_counts = [2]
    cfg.class_seeds = [0]
    cfg.n_query_per_item = 8
    cfg.address_steps = 400
    cfg.read_steps = 200
    cfg.write_steps = 40
    cfg.local_write_steps = 20
    cfg.write_n_perturb = 8
    cfg.n_atoms = 32
    cfg.hopfield_n_mem = 40
    cfg.attn_n_mem = 20
    cfg.param_tol = 1.0  # sizes are deliberately tiny in quick mode
    cfg.interference_K = 3
    cfg.interference_write_steps = 20
    cfg.interference_seeds = [0]
    cfg.support_radii = [0.1, 0.5, 1.0, 2.0]
    cfg.support_probes_per_radius = 16
    cfg.support_seeds = [0]
    cfg.ladder_rungs = ["free_mlp"]
    cfg.ladder_item_counts = [2]
    cfg.ladder_seeds = [0]


def _replace(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment POTENTIAL-CLASS: is the learned-landscape failure "
        "expressivity or support structure?"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed (project-level)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument(
        "--classes", nargs="+", help="Override the swept potential classes"
    )
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
    if args.classes:
        config.experiment_potential_class.potential_classes = list(args.classes)

    res = run_experiment_potential_class(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(json.dumps(res["param_match"], indent=2))
    print(json.dumps(res["item1_class_sweep"]["verdict"], indent=2))
    print(json.dumps(res["item2_interference"]["summary"], indent=2)[:3000])


if __name__ == "__main__":
    main()
