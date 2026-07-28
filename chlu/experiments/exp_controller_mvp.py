"""Experiment CONTROLLER-MVP: the minimum viable hand-coded controller (w23).

Every verb in the primitive vision — *decide, add, trash, evict* — is a
controller verb, and until now none was built. This module builds the MVC-0
controller (``chlu/core/controller.py``: admission + placement + eviction/decay,
**no learning anywhere**) on the designed ``AtomStorePotential`` store, and runs
the **N75 rematch**: CLU + controller vs the w21 gru/mlp/attention lines on the
sequential-parametric-write benchmark (K up to 64).

Three items (task ``controller-mvp``):

1. ``rematch``          retention-vs-K, controller ON vs OFF, on the SAME chart as
                        the four w21 primitive lines. The honest accounting is
                        the point: **retention-per-admitted-item** (over the items
                        the controller chose to store) and **retention-per-
                        offered-item** (over all K offered; a refused or evicted
                        item counts as not retained) are DIFFERENT metrics and
                        both are reported. Admitted/K is checked against the disk
                        packing bound (N74: 6.0 ± 0.9 / 16 ≈ 6.1).
2. ``decay_demo``       permanent + leaky wells in one store (w22 per-item
                        retention): a permanent item survives K subsequent writes;
                        a leaky item decays and self-evicts.
3. ``admission_cost``   the price of the admission test per write (O(n_stored)
                        distances + relocation draws; no relaxation, no gradient).

⚠ **Scope (stated before a referee does).** The controller is exercised on the
DESIGNED store only: its spacing certificate is meaningless for a global-support
learned write (N75). On a fixed address space the controller must abstain, so
per-offered retention is capped at ``N_pack / K`` — the abstention-vs-accuracy
trade, the field's oldest trick, stated up front. What the controller CANNOT fix
is in ``clu-controller-spec`` §5 and echoed in the report.

Runnable directly::

    uv run python -m chlu.experiments.exp_controller_mvp --quick

or via the CLI: ``chlu exp-controller-mvp [--project N] [--seed I] [--quick]``.
"""

import copy
import json
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.admission import disk_proposer, min_separation
from chlu.core.controller import (
    Controller,
    packing_bound_disk,
    radius_for_capacity,
)
from chlu.core.memory_potentials import AtomStorePotential, designed_payloads
from chlu.experiments.exp_learned_memory import model_for
from chlu.experiments.exp_sequential_write import (
    evaluate_items,
    sequential_write_primitive,
)

DIM = 3


# ---------------------------------------------------------------------------
# The controller line: offer K items to a designed store, then read them back
# ---------------------------------------------------------------------------


def _new_store(cfg, capacity: int) -> AtomStorePotential:
    return AtomStorePotential(
        dim=DIM,
        capacity=capacity,
        alpha=cfg.atom_alpha,
        s=cfg.atom_width,
        kappa=cfg.payload_kappa,
    )


def _proposer(cfg, radius, dim=DIM):
    return lambda k, n: np.asarray(disk_proposer(radius, dim)(k, n))[:, :2]


def _score(cfg, store, ids, centers, payloads, K_offered, seed):
    """Per-admitted and per-offered retention of a store's LIVE items.

    ``evaluate_items`` scores basin + value recovery per live item. per-admitted
    is the mean over live items; per-offered divides the *number retained* by all
    K offered, so refused/evicted items enter the denominator as zeros — the
    honest accounting the task requires.
    """
    if len(ids) == 0:
        return {
            "n_live": 0,
            "per_admitted": float("nan"),
            "per_offered": 0.0,
            "mean_abs_err": float("nan"),
            "per_item_strict": [],
        }
    sites = np.zeros((len(ids), DIM))
    sites[:, :2] = centers
    ev = evaluate_items(
        model_for(store, DIM),
        sites,
        np.asarray(payloads),
        cfg,
        seed,
        n_query=cfg.n_query_per_item,
        dim=DIM,
    )
    strict = np.array([d["strict"] for d in ev["per_item"]])
    n_retained = float(np.sum(strict))  # each item's strict in [0,1] (query frac)
    return {
        "n_live": len(ids),
        "per_admitted": float(np.mean(strict)),
        "per_offered": float(n_retained / K_offered),
        "mean_abs_err": ev["mean_abs_err"],
        "per_item_strict": [float(s) for s in strict],
        "live_ids": [int(i) for i in ids],
    }


def controller_line(cfg, seed, K, arm, dim=DIM):
    """Offer K items one-at-a-time to the designed store under one controller arm.

    Arms:
      ``off``       — ungated: every offer written at its proposed site (d_safe=0),
                      capacity K, no eviction. Reproduces w21 ``designed_ungated``.
      ``on_fixed``  — gate ON, FIXED disk radius, budget = capacity (the gate
                      refuses collisions; admitted saturates at the packing bound).
      ``on_sized``  — gate ON, disk radius grown so the packing bound >= K (the
                      address space sized to the load; the controller need not
                      abstain).
      ``on_evict``  — gate ON, FIXED radius, finite budget = round(N_pack): a
                      rolling buffer that EVICTS the least-recently-used item when
                      full. Exercises the eviction verb in the rematch itself.
      ``canon_sized``— the w26 rematch cell: same sized geometry, but placement is
                      CANONICAL (PGCP lattice) instead of refuse-and-relocate.
                      Admission is deterministic — every item that finds a free cell is
                      admitted, so ``n_admitted = min(K, n_cells)`` with **zero seed
                      variance** — and the resulting store supports exact deletion.
                      ``cfg.canonical_radius_mult`` inflates the lattice (1.05 => 73
                      cells at K=64, i.e. no abstention at all).
    """
    key = jax.random.PRNGKey(seed)
    k_prop, k_ctrl = jax.random.split(key, 2)
    d_safe = cfg.d_safe_mult * cfg.atom_width
    pay = np.asarray(designed_payloads(K, seed=cfg.payload_seed))

    if arm in ("on_sized", "canon_sized"):
        radius = max(cfg.proposal_radius, radius_for_capacity(K, d_safe))
        if arm == "canon_sized":
            radius *= float(cfg.canonical_radius_mult)
    else:
        radius = cfg.proposal_radius
    n_pack = packing_bound_disk(radius, d_safe)

    proposals = np.asarray(disk_proposer(radius, dim)(k_prop, K))
    proposer = _proposer(cfg, radius, dim)

    if arm == "off":
        ctrl = Controller(_new_store(cfg, K), d_safe=0.0, budget=K,
                          n_candidates=cfg.n_relocation_candidates)
        proposer = None  # ungated never relocates
    elif arm == "on_evict":
        budget = max(1, int(round(n_pack)))
        ctrl = Controller(
            _new_store(cfg, budget), d_safe=d_safe, budget=budget,
            amp=cfg.atom_amp, evict_policy=cfg.evict_policy,
            n_candidates=cfg.n_relocation_candidates,
        )
    elif arm == "canon_sized":
        # canonical placement: the lattice IS the allocator and the admission gate.
        # evict_policy must be set-function (LRU is excluded from the deletion claim).
        ctrl = Controller(
            _new_store(cfg, K), d_safe=d_safe, budget=cfg.budget, amp=cfg.atom_amp,
            evict_policy="depth", placement="canonical", lattice_radius=radius,
        )
        proposer = None  # nothing is ever relocated by search; the lattice decides
    else:  # on_fixed, on_sized
        ctrl = Controller(
            _new_store(cfg, K), d_safe=d_safe, budget=cfg.budget,
            amp=cfg.atom_amp, n_candidates=cfg.n_relocation_candidates,
        )

    for i in range(K):
        k_ctrl, ko = jax.random.split(k_ctrl)
        ctrl.offer(
            item_id=i,
            q_new=proposals[i],
            payload=float(pay[i]),
            key=ko,
            proposer=proposer,
            permanent=False,
        )

    ids, centers, payloads = ctrl.live_items()
    sc = _score(cfg, ctrl.store, ids, centers, payloads, K, seed)
    # achieved min pairwise spacing of the live sites (gate CAN-fire evidence)
    if len(centers) >= 2:
        dd = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        min_spacing = float(dd.min())
    else:
        min_spacing = float("nan")
    return {
        "arm": arm,
        "seed": int(seed),
        "K": int(K),
        "radius": float(radius),
        "d_safe": float(d_safe),
        "packing_bound": float(n_pack),
        "n_cells": (None if ctrl.placer is None else int(ctrl.placer.n_cells)),
        "min_spacing_live": min_spacing,
        "n_admitted": int(sc["n_live"]),
        "stats": dict(ctrl.stats),
        **sc,
    }


# ---------------------------------------------------------------------------
# ITEM 1 — the N75 rematch
# ---------------------------------------------------------------------------


def _mean_std(vals):
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return [float("nan"), float("nan")]
    return [float(np.mean(v)), float(np.std(v))]


def item_rematch(cfg, hcfg, seeds):
    """Retention-vs-K for CLU+controller (ON/OFF) + the four w21 primitive lines."""
    arms = ["off", "on_fixed", "on_evict"]
    if cfg.run_sized_geometry:
        arms.append("on_sized")
    if cfg.run_canonical_placement:
        arms.append("canon_sized")

    controller = {}
    for arm in arms:
        curve = []
        for K in cfg.K_grid:
            runs = [controller_line(cfg, s, K, arm) for s in seeds]
            curve.append({
                "K": int(K),
                "per_admitted": _mean_std([r["per_admitted"] for r in runs]),
                "per_offered": _mean_std([r["per_offered"] for r in runs]),
                "n_admitted": _mean_std([float(r["n_admitted"]) for r in runs]),
                "packing_bound": float(np.mean([r["packing_bound"] for r in runs])),
                "min_spacing_live": _mean_std([r["min_spacing_live"] for r in runs]),
                "radius": float(np.mean([r["radius"] for r in runs])),
                "intervention_rate": _mean_std([
                    (r["stats"]["relocated"] + r["stats"]["refused_spacing"]
                     + r["stats"]["refused_full"]) / max(1, r["stats"]["offered"])
                    for r in runs
                ]),
            })
        controller[arm] = curve

    # --- the four w21 primitive baselines at the rematch point (reuse harness) ---
    primitives = {}
    if cfg.run_primitives:
        for prim in cfg.kv_primitives:
            primitives[prim] = _primitive_curve(cfg, hcfg, prim, seeds)

    return {
        "K_grid": list(cfg.K_grid),
        "d_safe": float(cfg.d_safe_mult * cfg.atom_width),
        "proposal_radius": float(cfg.proposal_radius),
        "packing_bound_fixed": packing_bound_disk(
            cfg.proposal_radius, cfg.d_safe_mult * cfg.atom_width
        ),
        "controller": controller,
        "primitives": primitives,
        "note": (
            "controller lines are on the DESIGNED store (no learning); primitive "
            "lines are PARAMETRIC sequential gradient writes (primitive-harness). "
            "per_offered is the referee-facing metric: refused/evicted items are "
            "zeros in its denominator."
        ),
    }


def _primitive_curve(cfg, hcfg, primitive, seeds):
    """One primitive's retention-vs-K, selecting the LR on a short run first.

    Reuses ``sequential_write_primitive`` (the w21 harness) verbatim; the value it
    reports as ``mean_retention`` is retention-per-offered (every offered item is
    accepted by a baseline), so it is directly comparable to the controller's
    per-offered curve. ``item1_retained`` is the retention of the FIRST item.
    """
    # equal-budget LR selection (one seed, short K), then full run at best LR
    sel = []
    for lr in hcfg.lr_grid:
        r = sequential_write_primitive(
            primitive, cfg, hcfg, seeds[0], lr, n_items=cfg.kv_select_items
        )
        sel.append((lr, r["final_mean_retention"]))
    best_lr = max(sel, key=lambda t: (t[1] if np.isfinite(t[1]) else -1))[0]

    runs = [
        sequential_write_primitive(
            primitive, cfg, hcfg, s, best_lr, n_items=cfg.kv_baseline_K
        )
        for s in seeds
    ]
    n = max((len(r["history"]) for r in runs), default=0)
    mean_ret, item1 = [], []
    for j in range(n):
        mean_ret.append(_mean_std([
            r["history"][j]["mean_retention"] for r in runs if len(r["history"]) > j
        ]))
        item1.append(_mean_std([
            r["history"][j]["item1_retained"] for r in runs if len(r["history"]) > j
        ]))
    return {
        "selected_lr": best_lr,
        "K_at": list(range(1, n + 1)),
        "per_offered_mean_std": mean_ret,  # baselines accept everything => per-offered
        "item1_retention_mean_std": item1,
        "final_per_offered": mean_ret[-1] if mean_ret else [float("nan")] * 2,
    }


# ---------------------------------------------------------------------------
# ITEM 2 — permanent + leaky wells in one store (per-item retention, w22)
# ---------------------------------------------------------------------------


def item_decay_demo(cfg, seed):
    """One permanent item + K-1 leaky items; tick the clock and watch them decay.

    Confirms the third decision rule's decay half: the permanent well (leak=0) is
    untouched by :meth:`Controller.tick` while leaky wells shallow by exp(-leak)
    and self-evict below ``amp_floor``. This is the machinery that gives a store
    permanent AND forgettable content at once.
    """
    K = cfg.decay_demo_K
    d_safe = cfg.d_safe_mult * cfg.atom_width
    radius = max(cfg.proposal_radius, radius_for_capacity(K, d_safe))
    key = jax.random.PRNGKey(seed)
    k_prop, k_ctrl = jax.random.split(key, 2)
    proposals = np.asarray(disk_proposer(radius, DIM)(k_prop, K))
    pay = np.asarray(designed_payloads(K, seed=cfg.payload_seed))
    proposer = _proposer(cfg, radius)

    ctrl = Controller(
        _new_store(cfg, K), d_safe=d_safe, budget=K, amp=cfg.atom_amp,
        leak=cfg.decay_demo_leak, amp_floor=cfg.amp_floor,
        n_candidates=cfg.n_relocation_candidates,
    )
    for i in range(K):
        k_ctrl, ko = jax.random.split(k_ctrl)
        ctrl.offer(
            item_id=i, q_new=proposals[i], payload=float(pay[i]), key=ko,
            proposer=proposer, permanent=(i == 0),  # item 0 is permanent
        )

    # score item-0 (permanent) vs a representative leaky item over decay ticks
    def retention_of(item_id):
        for r in ctrl.records.values():
            if r.item_id == item_id:
                sites = np.zeros((1, DIM))
                sites[0, :2] = r.center
                ev = evaluate_items(
                    model_for(ctrl.store, DIM), sites, np.array([r.payload]),
                    cfg, seed, n_query=cfg.n_query_per_item, dim=DIM,
                )
                return float(ev["per_item"][0]["strict"])
        return 0.0  # evicted

    history = []
    n_ticks = 8
    leaky_id = 1
    for t in range(n_ticks + 1):
        history.append({
            "tick": t,
            "n_live": ctrl.n_live,
            "permanent_item0_retention": retention_of(0),
            "leaky_item1_retention": retention_of(leaky_id),
            "leaky_item1_amp": float(
                next((np.asarray(ctrl.store.amps)[r.slot]
                      for r in ctrl.records.values() if r.item_id == leaky_id), 0.0)
            ),
        })
        ctrl.tick()

    return {
        "K": int(K),
        "leak": float(cfg.decay_demo_leak),
        "amp_floor": float(cfg.amp_floor),
        "predicted_half_life_ticks": float(np.log(2.0) / cfg.decay_demo_leak),
        "history": history,
        "final_stats": dict(ctrl.stats),
    }


# ---------------------------------------------------------------------------
# ITEM 3 — the cost of the admission test per write
# ---------------------------------------------------------------------------


def item_admission_cost(cfg, seed):
    """Wall-time of one admission test vs one relaxation read (the store's read).

    The admission test is O(n_stored) distances plus up to ``n_candidates``
    relocation draws — NO relaxation, NO gradient. This prices it against the one
    unavoidable per-read cost (a two-phase rollout) so the controller's overhead
    is stated, not assumed cheap.
    """
    d_safe = cfg.d_safe_mult * cfg.atom_width
    key = jax.random.PRNGKey(seed)
    proposer = _proposer(cfg, cfg.proposal_radius)
    rows = []
    for n_stored in [1, 4, 16, 64]:
        stored = np.asarray(
            disk_proposer(cfg.proposal_radius, DIM)(jax.random.PRNGKey(seed + n_stored), n_stored)
        )[:, :2]
        q = np.asarray(disk_proposer(cfg.proposal_radius, DIM)(key, 1))[0, :2]

        # admission (worst case: proposal collides => relocation runs)
        from chlu.core.admission import admit_site

        def _adm(q=q, stored=stored, key=key):
            return admit_site(q, stored, d_safe, key=key, proposer=proposer,
                              n_candidates=cfg.n_relocation_candidates)

        t_adm = _time_call(_adm)
        rows.append({
            "n_stored": n_stored,
            "admission_ms": t_adm,
            "n_candidates": cfg.n_relocation_candidates,
        })

    # one relaxation read on a store of 16 items, for scale
    store = _new_store(cfg, 16)
    prop = np.asarray(disk_proposer(cfg.proposal_radius, DIM)(key, 64))
    pay = np.asarray(designed_payloads(16, seed=cfg.payload_seed))
    placed = []
    for c in prop:
        if len(placed) >= 16:
            break
        if min_separation(c[:2], np.stack(placed)[:, :2] if placed else np.zeros((0, 2))) >= d_safe:
            store = store.with_item(c[:2], float(pay[len(placed)]), amp=cfg.atom_amp)
            placed.append(c)
    model = model_for(store, DIM)
    from chlu.experiments.exp_sequential_write import two_phase

    q0 = jnp.zeros((1, DIM)).at[0, :2].set(jnp.asarray(placed[0][:2]))
    p0 = jnp.zeros((1, DIM))
    fn = jax.jit(lambda qq, pp: two_phase(model, qq, pp, cfg, cfg.gamma_address, cfg.gamma_read)[1])
    fn(q0, p0)[0].block_until_ready()
    t_read = _time_call(lambda: fn(q0, p0))

    return {
        "admission_by_n_stored": rows,
        "relaxation_read_ms": t_read,
        "read_steps": int(cfg.address_steps + cfg.read_steps),
        "note": (
            "admission is O(n_stored) + n_candidates draws, no physics; the read "
            "is a two-phase rollout. The controller adds negligible overhead to a "
            "write and none to a read."
        ),
    }


def _time_call(fn, n: int = 20):
    fn()  # warm / compile
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        try:
            jax.block_until_ready(r)  # no-op for numpy/dict results
        except Exception:
            pass
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot(results, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    rm = results.get("rematch")
    if not rm:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"off": "0.5", "on_fixed": "C0", "on_evict": "C2", "on_sized": "C3",
              "canon_sized": "C4"}
    for metric, ax, title in (
        ("per_admitted", axes[0], "retention per ADMITTED item"),
        ("per_offered", axes[1], "retention per OFFERED item"),
    ):
        for arm, curve in rm["controller"].items():
            y = np.array([c[metric] for c in curve])
            ax.errorbar([c["K"] for c in curve], y[:, 0], yerr=y[:, 1],
                        marker="o", ms=4, capsize=2, color=colors.get(arm),
                        label=f"CLU+ctrl:{arm}")
        # primitive lines only make sense on per-offered
        if metric == "per_offered":
            for prim, pc in (rm.get("primitives") or {}).items():
                y = np.array(pc["per_offered_mean_std"])
                ax.errorbar(pc["K_at"], y[:, 0], yerr=y[:, 1], marker="s", ms=2.5,
                            capsize=1.5, ls="--", label=f"{prim} (w21)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (items offered)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=6.5)
    fig.suptitle("N75 rematch: MVC-0 controller on a designed store", fontsize=11)
    fig.tight_layout()
    p = os.path.join(save_dir, "controller_mvp_rematch.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_controller_mvp(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    items: Optional[list] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_controller_mvp
    hcfg = config.experiment_primitive_harness
    seed = config.project.seed if seed is None else seed
    items = items or ["1", "2", "3"]
    os.makedirs(save_dir, exist_ok=True)
    seeds = list(cfg.seeds)

    results = {
        "seed": seed,
        "seeds": seeds,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "atom_width", "atom_alpha", "atom_amp", "payload_kappa",
                "d_safe_mult", "n_relocation_candidates", "proposal_radius",
                "K_grid", "budget", "evict_policy", "leak", "amp_floor",
                "run_sized_geometry", "run_canonical_placement",
                "canonical_radius_mult", "decay_demo_K", "decay_demo_leak",
                "dt", "gamma_address", "gamma_read", "address_steps", "read_steps",
                "n_query_per_item", "payload_tol", "payload_tol_frac",
                "run_primitives", "kv_primitives", "kv_baseline_K",
            )
        },
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_controller_mvp_metrics.json")

    def _dump():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)

    if "1" in items:
        results["rematch"] = item_rematch(cfg, hcfg, seeds)
        _dump()
    if "2" in items:
        results["decay_demo"] = item_decay_demo(cfg, seeds[0])
        _dump()
    if "3" in items:
        results["admission_cost"] = item_admission_cost(cfg, seeds[0])
        _dump()

    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        results["figures"] = []
        results["figure_error"] = repr(exc)
    _dump()
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps.

    ⚠ The relaxation length is what makes the DESIGNED reference arm work; a smoke
    run whose baseline fails prints what looks like a scientific negative (w20
    lesson). Steps are kept full; only the sweep breadth is cut.
    """
    cfg = config.experiment_controller_mvp
    cfg.seeds = [0, 1]
    cfg.K_grid = [2, 8, 16]
    cfg.run_sized_geometry = True
    cfg.n_query_per_item = 8
    cfg.decay_demo_K = 4
    cfg.run_primitives = True
    cfg.kv_primitives = ["mlp", "attention"]
    cfg.kv_baseline_K = 8
    cfg.kv_select_items = 2
    cfg.kv_max_write_steps = 40
    config.experiment_primitive_harness.lr_grid = [1e-3, 3e-3]


def _replace(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment CONTROLLER-MVP: the hand-coded MVC-0 controller "
        "on a designed store, and the N75 rematch."
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed (project-level)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--items", nargs="+", choices=["1", "2", "3"],
                        help="Run only these items (default: all)")
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

    res = run_experiment_controller_mvp(
        config=config, save_dir=save_dir, models_dir=models_dir,
        seed=args.seed, items=args.items,
    )
    print(json.dumps(res.get("rematch", {}).get("controller", {}), indent=2)[:2500])


if __name__ == "__main__":
    main()
