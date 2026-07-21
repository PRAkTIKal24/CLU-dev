"""Experiment LEARNED-MEMORY: does the w19 write->address->read loop survive LEARNING?

w19 (``exp_retrieval``) demonstrated the full loop with **zero learning**: the
landscape was hand-designed, and its write locality came from designed additive
separability plus an exact symmetry — not from emergence. The program's vision
needs the landscape to be *learned* (designed where design is needed, loss-refined
elsewhere), and **nothing told us the loop survives that transition**. This module
measures it.

Four measurements, all on a **design-freedom ladder** (``DESIGN_RUNGS``) that runs
from the w19 hand-designed landscape to a free MLP:

1. ``item1_write_read``      write K items by training ``V``, then relax a
                             PERTURBED query and ask whether it lands where the
                             writer wrote. Basin-level and STRICT success
                             reported separately (w19/Toy-D showed they diverge),
                             plus retrieval fidelity, durability, and a
                             blank-landscape control on every cell.
2. ``item2_design_freedom``  the fidelity-vs-design-freedom curve and the
                             MINIMUM DESIGNED STRUCTURE that preserves the loop.
3. ``item3_interference``    write A, then write B, then re-read A. Does additive
                             separability survive learning, or was w19's 4.17e-7
                             corruption an artifact of the design?
4. ``item4_gamma_map``       retrieval is TWO-PHASE with two different frictions;
                             a 2-D (gamma_address, gamma_read) map, not a single
                             gamma scan. Tests whether the good region is
                             off-diagonal.

**Anti-decoration guard (inherited from w19, load-bearing).** Every query is
launched with ``q2(0) = p2(0) = 0``, so a read-out that recovers the stored value
got it from ``V``. Every cell carries a **blank-landscape control** (identically
trained, all payloads zero). A full-state read scores 1.000 on a blank landscape
because it reads the address back — so the headline read here is payload-only.
**Any cell without a passing blank control is not a measurement.**

Runnable directly:
    uv run python -m chlu.experiments.exp_learned_memory --quick
or via the CLI: ``chlu exp-learned-memory [--project N] [--seed I] [--quick]``.
"""

import copy
import json
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.memory_potentials import (
    DESIGN_RUNGS,
    DesignFreedomPotential,
    designed_payloads,
    ring_sites,
)
from chlu.experiments.exp_retrieval import (
    linear_codebook_read,
    linear_probe,
    nearest_centroid_read,
    tail_features,
)
from chlu.experiments.goldstone_harness import clu_with_potential
from chlu.training.train_memory import train_memory_landscape


# ---------------------------------------------------------------------------
# Landscape construction + write
# ---------------------------------------------------------------------------


def build_landscape(rung: str, cfg, payloads, key, dim: int = 3):
    """A ``DesignFreedomPotential`` at ``rung`` with the w20 config geometry."""
    return DesignFreedomPotential(
        rung=rung,
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
        residual_scale=cfg.residual_scale,
        rbf_init_width=cfg.rbf_init_width,
    )


def write_items(V, cfg, payloads, key, dim: int = 3, steps: Optional[int] = None):
    """Train ``V`` to hold ``payloads`` at the K ring sites. Returns (V, history)."""
    targets = ring_sites(len(payloads), f=cfg.f, dim=dim, payloads=payloads)
    return train_memory_landscape(
        V,
        targets,
        key,
        steps=cfg.write_steps if steps is None else steps,
        lr=cfg.write_lr,
        weight_decay=cfg.write_weight_decay,
        loss_kwargs=dict(
            n_perturb=cfg.write_n_perturb,
            sigma_addr=cfg.write_sigma_addr,
            sigma_pay=cfg.write_sigma_pay,
            margin=cfg.write_margin,
            barrier=cfg.write_barrier,
        ),
    )


def model_for(V, dim: int = 3) -> CHLU:
    """CLU wired to ``V``. ``newtonian_learned`` so the mass vector is live."""
    return clu_with_potential(
        V, dim=dim, kinetic_mode="newtonian_learned", inertia=jnp.ones(dim)
    )


# ---------------------------------------------------------------------------
# Queries + the TWO-PHASE retrieval loop
# ---------------------------------------------------------------------------


def make_queries(
    key, K: int, n_per_item: int, cfg, dim: int = 3, n_sites: Optional[int] = None
):
    """Perturbed queries near each site. ``q2 = p2 = 0`` ALWAYS (the guard).

    Jitter is Gaussian in q-space (not angular as in w19) so the same query
    distribution is meaningful for rungs that have no ring; the scale is matched
    to w19's arc-length jitter, ``sigma_q = f * sigma_theta``.

    ``n_sites`` is the size of the ring the landscape was built on, which is NOT
    always the number of items being queried: the interference test writes K-1
    items on a K-site ring and then queries only those K-1. Passing K as the ring
    size while querying fewer items would place every query at the wrong angle
    (a harness bug that inflated the designed rung's read error to 0.44).
    """
    k_q, k_p = jax.random.split(key, 2)
    n = K * n_per_item
    labels = np.repeat(np.arange(K), n_per_item)
    sites = ring_sites(n_sites or K, f=cfg.f, dim=dim)[:K]  # payload column left at 0
    Q0 = jnp.asarray(sites)[jnp.asarray(labels)]
    jit = jax.random.normal(k_q, (n, dim)) * (cfg.f * cfg.query_sigma_theta)
    jit = jit.at[:, 2].set(0.0)
    Q0 = (Q0 + jit).at[:, 2].set(0.0)
    P0 = jnp.zeros((n, dim))
    P0 = P0.at[:, :2].set(jax.random.normal(k_p, (n, 2)) * cfg.query_sigma_p)
    return Q0, P0, labels


def two_phase_retrieve(model, Q0, P0, cfg, gamma_address, gamma_read):
    """``query -> [gamma_address relaxation] -> ADDRESS -> [gamma_read rollout] -> traj``.

    The Hub's proposed mechanism: the address of an item is *derived*, not chosen
    — it is wherever the item relaxes to under the dissipative dynamics. Phase 1
    performs that relaxation (no gradient needed); phase 2 is the conservative
    rollout the read happens on (where gradients are safe, Prop 4).

    Returns ``(addr_q, addr_p, traj)`` with traj from phase 2 only.
    """
    dim = Q0.shape[1]

    def relax(q, p):
        tr = model(q, p, cfg.address_steps, cfg.dt, gamma_address)
        return tr[-1, :dim], tr[-1, dim:]

    addr_q, addr_p = jax.vmap(relax)(Q0, P0)
    traj = jax.vmap(lambda q, p: model(q, p, cfg.read_steps, cfg.dt, gamma_read))(
        addr_q, addr_p
    )
    return addr_q, addr_p, traj


def _basin_of(addr_q, n_sites, f):
    """Which site's basin the relaxed address sits in (address plane only).

    Scored against ALL ``n_sites`` of the ring, including sites holding items the
    caller is not querying — landing in a freshly-written neighbour's well is
    exactly the interference we want to see, so it must count as a miss, not be
    hidden by restricting the candidate set.
    """
    sites = np.asarray(ring_sites(n_sites, f=f, dim=addr_q.shape[1]))[:, :2]
    d = ((np.asarray(addr_q)[:, None, :2] - sites[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=1)


def _probe_split(feats, labels, n_class, rng):
    n = feats.shape[0]
    perm = rng.permutation(n)
    half = n // 2
    return linear_probe(
        feats[perm[:half]],
        labels[perm[:half]],
        feats[perm[half:]],
        labels[perm[half:]],
        n_class,
    )


def evaluate_cell(
    model,
    cfg,
    payloads,
    seed,
    gamma_address=None,
    gamma_read=None,
    dim: int = 3,
    with_durability: bool = False,
    n_sites: Optional[int] = None,
):
    """Run the loop on one landscape and score it. The unit of measurement."""
    K = len(payloads)
    n_sites = n_sites or K
    ga = cfg.gamma_address if gamma_address is None else gamma_address
    gr = cfg.gamma_read if gamma_read is None else gamma_read
    key = jax.random.PRNGKey(seed)
    Q0, P0, labels = make_queries(
        key, K, cfg.n_query_per_item, cfg, dim=dim, n_sites=n_sites
    )

    addr_q, _, traj = two_phase_retrieve(model, Q0, P0, cfg, ga, gr)
    finite = bool(np.all(np.isfinite(np.asarray(traj))))

    # --- addressing: did the relaxation land where the writer wrote? ---
    basin = _basin_of(addr_q, n_sites, cfg.f)
    basin_ok = basin == labels

    # --- payload read-out (tail of the phase-2 rollout) ---
    f_pay = tail_features(traj, cfg.tail_frac, cfg.n_subsample, coords=[2])
    f_full = tail_features(traj, cfg.tail_frac, cfg.n_subsample)
    read_val = np.asarray(f_pay).mean(axis=1)
    pay = np.asarray(payloads)
    err = np.abs(read_val - pay[labels])
    strict_ok = basin_ok & (err < cfg.payload_tol)

    rng = np.random.default_rng(seed)
    n = f_pay.shape[0]
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2 :]
    acc_cb, r2_cb = linear_codebook_read(
        f_pay[tr], labels[tr], f_pay[te], labels[te], pay
    )
    acc_nc = nearest_centroid_read(f_pay[tr], labels[tr], f_pay[te], labels[te], K)
    acc_full, _ = _probe_split(f_full, labels, K, np.random.default_rng(seed))

    # ⭐ ADDRESS LEAK. On a BLANK landscape every item stores the same value (0),
    # so any site-to-site spread in the payload read-out is the read seeing the
    # ADDRESS, not the content. This is the quantity that decides whether a
    # blank control passes, so it is reported next to it rather than left to be
    # inferred: the designed landscape has an exactly separable payload spring
    # (spread == 0 when nothing is stored); a learned V couples q2 to (q0,q1)
    # generically and does not.
    site_means = np.array(
        [
            read_val[labels == k].mean() if np.any(labels == k) else np.nan
            for k in range(K)
        ]
    )

    out = {
        "K": K,
        "gamma_address": float(ga),
        "gamma_read": float(gr),
        "finite": finite,
        "read_val_site_means": [float(x) for x in site_means],
        "read_val_site_spread": float(np.nanstd(site_means)),
        "basin_success_rate": float(np.mean(basin_ok)),
        "strict_success_rate": float(np.mean(strict_ok)),
        "payload_abs_err_mean": float(np.mean(err)),
        "payload_abs_err_median": float(np.median(err)),
        "acc_payload_codebook_read": acc_cb,
        "acc_payload_nearest_centroid": acc_nc,
        "payload_regression_r2": r2_cb,
        "acc_full_state_read": acc_full,
        "chance": 1.0 / K,
    }

    if with_durability:
        steps = traj.shape[1]
        surv = []
        for frac in cfg.survival_fracs:
            i = min(int(frac * steps), steps - 1)
            f = np.asarray(traj[:, i, 2])[:, None]
            a, _ = _probe_split(f, labels, K, np.random.default_rng(seed))
            surv.append({"step": int(i), "acc_payload_only": a})
        out["durability"] = surv
        out["durability_drop"] = float(
            surv[0]["acc_payload_only"] - surv[-1]["acc_payload_only"]
        )
    return out


def _blank_ok(written_cell, blank_cell, cfg):
    """Blank control passes iff the blank read is near chance.

    A blank landscape that reads ABOVE chance means the read is seeing the
    address, not the memory: the written number in that cell is not a
    measurement of retrieval (w19: blank 0.469 vs chance 0.500 = passing).
    """
    return bool(
        blank_cell["acc_payload_codebook_read"]
        <= blank_cell["chance"] + cfg.blank_margin
    )


def _cell_pair(rung, cfg, K, seed, dim=3, with_durability=False, train_steps=None):
    """Train the WRITTEN and BLANK landscapes for one rung and score both."""
    pay = designed_payloads(K, seed=cfg.payload_seed)
    key = jax.random.PRNGKey(seed)
    k_w, k_b, k_tw, k_tb = jax.random.split(key, 4)

    Vw = build_landscape(rung, cfg, pay, k_w, dim=dim)
    Vw, hist_w = write_items(Vw, cfg, pay, k_tw, dim=dim, steps=train_steps)
    # BLANK: identical architecture, identical training, NOTHING stored.
    blank_pay = jnp.zeros_like(jnp.asarray(pay))
    Vb = build_landscape(rung, cfg, blank_pay, k_b, dim=dim)
    Vb, hist_b = write_items(Vb, cfg, blank_pay, k_tb, dim=dim, steps=train_steps)

    written = evaluate_cell(
        model_for(Vw, dim), cfg, pay, seed, with_durability=with_durability
    )
    blank = evaluate_cell(model_for(Vb, dim), cfg, pay, seed)
    return (
        {
            "rung": rung,
            "design_freedom": DESIGN_RUNGS.index(rung),
            "written": written,
            "blank": blank,
            "blank_control_passes": _blank_ok(written, blank, cfg),
            "write_loss_initial": hist_w[0] if hist_w else None,
            "write_loss_final": hist_w[-1] if hist_w else None,
            "n_learned_params": _n_params(Vw),
        },
        Vw,
        Vb,
    )


def _n_params(V):
    import equinox as eqx

    if getattr(V, "learned", None) is None:
        return 0
    return int(
        sum(
            x.size
            for x in jax.tree_util.tree_leaves(
                eqx.filter(V.learned, eqx.is_inexact_array)
            )
        )
    )


# ---------------------------------------------------------------------------
# Item 1 — write -> relax consistency, fidelity, durability (per rung)
# ---------------------------------------------------------------------------


def item1_write_read(cfg, seed: int = 0):
    """The decisive test: does relaxation land where the writer wrote, under a
    LEARNED landscape, and does the read still work?"""
    rows = []
    for rung in cfg.rungs:
        for K in cfg.item_counts:
            cell, _, _ = _cell_pair(
                rung, cfg, K, seed, with_durability=(K == cfg.item_counts[0])
            )
            rows.append(cell)
    return {"rows": rows, "w19_baseline": dict(cfg.w19_baseline)}


# ---------------------------------------------------------------------------
# Item 2 — the fidelity-vs-design-freedom curve
# ---------------------------------------------------------------------------


def item2_design_freedom(item1_result, cfg):
    """Collapse item 1 into the curve + the minimum-viable-design point.

    Minimum viable design = the FREEST rung (largest design_freedom) that still
    passes at every measured K: strict success >= threshold, codebook read >=
    threshold, and a PASSING blank control. A rung whose blank control fails is
    disqualified whatever its written score.
    """
    by_rung = {}
    for row in item1_result["rows"]:
        by_rung.setdefault(row["rung"], []).append(row)

    curve = []
    for rung in cfg.rungs:
        rows = by_rung.get(rung, [])
        if not rows:
            continue
        # ⚠ A cell whose blank control fails is NOT A MEASUREMENT — it is neither
        # a pass nor a fail (protocol: "any cell without a passing blank control
        # is not a measurement"). Scoring it as a failure would be just as wrong
        # as scoring it as a success, so disqualified cells are EXCLUDED and
        # counted, and a rung with no surviving cell is reported as unmeasured.
        measured = [r for r in rows if r["blank_control_passes"]]
        passes = bool(measured) and all(
            r["written"]["strict_success_rate"] >= cfg.pass_strict
            and r["written"]["acc_payload_codebook_read"] >= cfg.pass_read
            for r in measured
        )
        curve.append(
            {
                "rung": rung,
                "design_freedom": DESIGN_RUNGS.index(rung),
                "n_learned_params": rows[0]["n_learned_params"],
                "per_K": [
                    {
                        "K": r["written"]["K"],
                        "basin": r["written"]["basin_success_rate"],
                        "strict": r["written"]["strict_success_rate"],
                        "codebook_read": r["written"]["acc_payload_codebook_read"],
                        "blank_read": r["blank"]["acc_payload_codebook_read"],
                        "blank_leak_spread": r["blank"]["read_val_site_spread"],
                        "blank_passes": r["blank_control_passes"],
                        "payload_abs_err": r["written"]["payload_abs_err_mean"],
                    }
                    for r in rows
                ],
                "n_cells": len(rows),
                "n_cells_measured": len(measured),
                "n_cells_disqualified_by_blank": len(rows) - len(measured),
                "Ks_measured": [r["written"]["K"] for r in measured],
                "min_strict": (
                    min(r["written"]["strict_success_rate"] for r in measured)
                    if measured
                    else None
                ),
                "min_codebook_read": (
                    min(r["written"]["acc_payload_codebook_read"] for r in measured)
                    if measured
                    else None
                ),
                "all_blanks_pass": all(r["blank_control_passes"] for r in rows),
                "passes": passes,
            }
        )
    passing = [c for c in curve if c["passes"]]
    return {
        "pass_strict_threshold": cfg.pass_strict,
        "pass_read_threshold": cfg.pass_read,
        "curve": curve,
        "minimum_viable_design_rung": (
            max(passing, key=lambda c: c["design_freedom"])["rung"] if passing else None
        ),
        "minimum_viable_design_freedom": (
            max(c["design_freedom"] for c in passing) if passing else None
        ),
        "loop_survives_learning": bool(any(c["design_freedom"] > 0 for c in passing)),
    }


# ---------------------------------------------------------------------------
# Item 3 — cross-write interference: does additive separability survive learning?
# ---------------------------------------------------------------------------


def item3_interference(cfg, seed: int = 0, dim: int = 3):
    """Write A (K-1 items), re-read A; then write B at a fresh site by CONTINUING
    training, and re-read A again. Corruption = the change in A's read-out.

    w19 measured 4.17e-7 (i.e. exactly zero) from designed additive separability.
    If a learned ``V`` shows interference, that locality was an artifact of the
    design and the "write near without disturbing" vision element is not yet real.
    """
    K = cfg.interference_K
    pay_all = designed_payloads(K, seed=cfg.payload_seed)
    pay_A = pay_all[:-1]  # items written first
    rows = []
    for rung in cfg.rungs:
        key = jax.random.PRNGKey(seed)
        k_v, k_a, k_b = jax.random.split(key, 3)
        V = build_landscape(rung, cfg, pay_all, k_v, dim=dim)

        # --- write A only (targets = the first K-1 sites of the K-site ring) ---
        sites_all = ring_sites(K, f=cfg.f, dim=dim, payloads=pay_all)
        V, _ = train_memory_landscape(
            V,
            sites_all[:-1],
            k_a,
            steps=cfg.write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=dict(
                n_perturb=cfg.write_n_perturb,
                sigma_addr=cfg.write_sigma_addr,
                sigma_pay=cfg.write_sigma_pay,
                margin=cfg.write_margin,
                barrier=cfg.write_barrier,
            ),
        )
        # n_sites=K: the K-1 items live on the FIRST K-1 sites of the K-site
        # ring, and basins are scored against all K (landing in B's fresh well
        # must count as a miss).
        before = evaluate_cell(model_for(V, dim), cfg, pay_A, seed, dim=dim, n_sites=K)

        # --- now write B at the fresh site, CONTINUING from the same params ---
        V2, _ = train_memory_landscape(
            V,
            sites_all[-1:],
            k_b,
            steps=cfg.interference_write_steps,
            lr=cfg.write_lr,
            weight_decay=cfg.write_weight_decay,
            loss_kwargs=dict(
                n_perturb=cfg.write_n_perturb,
                sigma_addr=cfg.write_sigma_addr,
                sigma_pay=cfg.write_sigma_pay,
                margin=cfg.write_margin,
                barrier=cfg.write_barrier,
            ),
        )
        after = evaluate_cell(model_for(V2, dim), cfg, pay_A, seed, dim=dim, n_sites=K)

        rows.append(
            {
                "rung": rung,
                "design_freedom": DESIGN_RUNGS.index(rung),
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
                "codebook_A_before_B": before["acc_payload_codebook_read"],
                "codebook_A_after_B": after["acc_payload_codebook_read"],
            }
        )
    return {
        "w19_designed_corruption": 4.17e-7,
        "codebook_spacing": float(
            np.min(np.diff(np.sort(np.asarray(pay_all)))) if K > 1 else float("nan")
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Item 4 — the 2-D (gamma_address, gamma_read) map
# ---------------------------------------------------------------------------


def item4_gamma_map(cfg, seed: int = 0, dim: int = 3):
    """Retrieval fidelity over the FULL (gamma_address, gamma_read) grid.

    w19 found retrieval needs dissipation (0.813 at gamma=0 -> 1.000 at gamma>0)
    while Prop 5 found dissipation destroys address gradients. The proposed
    resolution is that these are different PHASES wanting different gamma. A
    single gamma scan cannot see that; this map can.
    """
    K = cfg.gamma_map_K
    out = []
    for rung in cfg.gamma_map_rungs:
        pay = designed_payloads(K, seed=cfg.payload_seed)
        key = jax.random.PRNGKey(seed)
        k_v, k_t = jax.random.split(key, 2)
        V = build_landscape(rung, cfg, pay, k_v, dim=dim)
        V, _ = write_items(V, cfg, pay, k_t, dim=dim)
        model = model_for(V, dim)
        cells = []
        for ga in cfg.gamma_address_grid:
            for gr in cfg.gamma_read_grid:
                c = evaluate_cell(
                    model, cfg, pay, seed, gamma_address=ga, gamma_read=gr, dim=dim
                )
                cells.append(
                    {
                        "gamma_address": float(ga),
                        "gamma_read": float(gr),
                        "strict": c["strict_success_rate"],
                        "basin": c["basin_success_rate"],
                        "codebook_read": c["acc_payload_codebook_read"],
                        "payload_abs_err": c["payload_abs_err_mean"],
                        "finite": c["finite"],
                    }
                )
        best = max(cells, key=lambda c: (c["strict"], -c["payload_abs_err"]))
        # Is the good region off-diagonal? Compare the best cell against the
        # best cell that has gamma_read == gamma_address (the "single-gamma"
        # protocol w19 actually ran).
        diag = [c for c in cells if c["gamma_read"] == c["gamma_address"]]
        best_diag = max(diag, key=lambda c: c["strict"]) if diag else None
        # Sensitivity of fidelity to each axis, at the best setting of the other.
        at_best_ga = [c for c in cells if c["gamma_address"] == best["gamma_address"]]
        at_best_gr = [c for c in cells if c["gamma_read"] == best["gamma_read"]]
        out.append(
            {
                "rung": rung,
                "K": K,
                "cells": cells,
                "best": best,
                "best_on_diagonal": best_diag,
                "offdiagonal_gain": (
                    best["strict"] - best_diag["strict"] if best_diag else None
                ),
                "spread_over_gamma_read_at_best_gamma_address": float(
                    max(c["strict"] for c in at_best_ga)
                    - min(c["strict"] for c in at_best_ga)
                ),
                "spread_over_gamma_address_at_best_gamma_read": float(
                    max(c["strict"] for c in at_best_gr)
                    - min(c["strict"] for c in at_best_gr)
                ),
                "strict_at_gamma_address_zero": float(
                    max(
                        (c["strict"] for c in cells if c["gamma_address"] == 0.0),
                        default=float("nan"),
                    )
                ),
            }
        )
    return {"maps": out}


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
    df = results.get("item2_design_freedom")
    if df and df["curve"]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        x = [c["design_freedom"] for c in df["curve"]]
        names = [c["rung"] for c in df["curve"]]
        a1.plot(
            x,
            [c["min_strict"] for c in df["curve"]],
            "o-",
            label="strict success (min over K)",
        )
        a1.plot(
            x,
            [c["min_codebook_read"] for c in df["curve"]],
            "s-",
            label="codebook read (min over K)",
        )
        a1.plot(
            x,
            [max(k["blank_read"] for k in c["per_K"]) for c in df["curve"]],
            "^:",
            label="blank control (max over K)",
        )
        a1.axhline(df["pass_strict_threshold"], color="r", ls="--", lw=0.8)
        a1.set_xticks(x)
        a1.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        a1.set_xlabel("design freedom ->")
        a1.set_ylabel("retrieval")
        a1.set_title("Fidelity vs design freedom")
        a1.legend(fontsize=7)

        gm = results.get("item4_gamma_map")
        if gm and gm["maps"]:
            m = gm["maps"][-1]
            gas = sorted({c["gamma_address"] for c in m["cells"]})
            grs = sorted({c["gamma_read"] for c in m["cells"]})
            img = np.zeros((len(gas), len(grs)))
            for c in m["cells"]:
                img[gas.index(c["gamma_address"]), grs.index(c["gamma_read"])] = c[
                    "strict"
                ]
            im = a2.imshow(
                img, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto"
            )
            a2.set_xticks(range(len(grs)))
            a2.set_xticklabels([f"{g:g}" for g in grs], fontsize=7)
            a2.set_yticks(range(len(gas)))
            a2.set_yticklabels([f"{g:g}" for g in gas], fontsize=7)
            a2.set_xlabel("gamma_read")
            a2.set_ylabel("gamma_address")
            a2.set_title(f"strict retrieval, 2-D gamma map ({m['rung']})")
            fig.colorbar(im, ax=a2)
        fig.tight_layout()
        p = os.path.join(save_dir, "learned_memory_fig1_design_freedom_and_gamma.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_learned_memory(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Run items 1-4 and write JSON + figures."""
    config = config or get_default_config()
    cfg = config.experiment_learned_memory
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "seed": seed,
        "rungs": list(cfg.rungs),
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
                "write_lr",
                "write_margin",
                "write_barrier",
                "write_sigma_addr",
                "write_sigma_pay",
                "hidden",
                "n_atoms",
                "residual_scale",
                "pass_strict",
                "pass_read",
                "blank_margin",
            )
        },
    }
    results["item1_write_read"] = item1_write_read(cfg, seed=seed)
    results["item2_design_freedom"] = item2_design_freedom(
        results["item1_write_read"], cfg
    )
    results["item3_interference"] = item3_interference(cfg, seed=seed)
    results["item4_gamma_map"] = item4_gamma_map(cfg, seed=seed)
    results["figures"] = _plot_all(results, save_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_learned_memory_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps."""
    cfg = config.experiment_learned_memory
    cfg.rungs = ["designed", "free_mlp"]
    cfg.item_counts = [2]
    cfg.n_query_per_item = 8
    cfg.address_steps = 100
    cfg.read_steps = 100
    cfg.write_steps = 40
    cfg.write_n_perturb = 8
    cfg.interference_K = 3
    cfg.interference_write_steps = 20
    cfg.gamma_map_rungs = ["free_mlp"]
    cfg.gamma_map_K = 2
    cfg.gamma_address_grid = [0.0, 0.05]
    cfg.gamma_read_grid = [0.0, 0.05]
    cfg.survival_fracs = [0.1, 0.5, 0.99]


def _replace(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment LEARNED-MEMORY: does the write/address/read loop "
        "survive a LEARNED landscape?"
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

    res = run_experiment_learned_memory(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    print(json.dumps(res["item2_design_freedom"], indent=2)[:4000])


if __name__ == "__main__":
    main()
