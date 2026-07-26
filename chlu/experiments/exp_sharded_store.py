"""Experiment SHARDED-STORE (w25): the first real N-unit CLU store, and the §5.1
2x2 discriminator — **is the K~=32 write ceiling per-dig?**

**What is being tested.** w23 pinned ``K_learned(d) = min(2^d, K_ceiling ~= 32)``
and w24 showed the ceiling survives every write lever *inside one shared atom pool*.
`lattice-capacity-theory` proved **Prop L2** — a write on disjoint atom groups IS N
independent optimizers (AdamW factorizes elementwise; nothing to synchronize) — and
**Theorem L1** ``K_total = min(K_addr(geometry), sum_r K_write(unit r))``. If the
ceiling is a property of one *dig*, then N units digging K/N valleys each should
clear a wall the monolithic store cannot. If it is not, additivity is refuted.

**The 2x2 is the result, not any single cell** (theorist §5.1):

===================  ==========================  =======================================
cell                 baseline (w23/w24)          discriminates
===================  ==========================  =======================================
d=6 K=64, 2x32       0.855 / 0.858 FAIL          additivity ``min(K_addr, N*32)``
d=4 K=32, 2x16 ⭐     ~0.80-0.83 FAIL             ⭐ the CONTROL: geometry-bound BELOW the
                     (flat over a 16x atom          ceiling, so it must STILL FAIL; if it
                     sweep)                         passes, ``K = 32N`` is unclamped
d=8 K=64, 2x32/4x16  0.883 FAIL (0.9067          additivity is in ``N*K_ceiling``, not in
                     marginal-PASS at 2x atoms)     shard size
d=8 K=256, 8x32      untested                    where geometry takes over
===================  ==========================  =======================================

Both main cells fixed => the ceiling was entirely per-dig, ``K_total = 32N``; only
d=6 fixed => Theorem L1's ``min`` law; **neither => additivity REFUTED** (decision-
grade, reported plainly).

**Arms.** ``monolithic`` is the w23/w24 line and the mandatory laundering control at
every ``K_total``; ``sharded_matched`` splits the SAME total atom budget across N
shards (parameter-matched); ``sharded_per_shard`` gives each shard the budget a
monolithic ``K/N`` store would get (the literal "N independent stores"), whose own
laundering line is ``monolithic_nx`` — the monolithic store at that same total atom
count, which w23 measured getting *worse* with more atoms.

**Everything else is w23's, verbatim**: geometry (``designed_sites`` in a d-ball),
the two-phase read (gamma_address relax -> gamma_read rollout), the write operator
(one Adam dig per shard, ``train_memory.write_loss``), the query jitter, and the
leak-immune value-blank pass criterion.

⚠ **Scope (N89).** The router is ``argmin_r V_r(q)`` — a classical nearest-neighbour
score over stored addresses. The claim is: *the write is additive at zero optimizer
cost and the read stays O(1) in depth because a classical O(N) score suffices to
route.* Never "capacity multiplies by sharding" as a dynamical result.

Runnable directly:
    uv run python -m chlu.experiments.exp_sharded_store --quick
or via the CLI: ``chlu exp-sharded-store [--project N] [--seed I] [--quick]``.
"""

import json
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.admission import D_SAFE_MULT
from chlu.core.controller import Controller, radius_for_capacity
from chlu.core.memory_potentials import (
    AtomStorePotential,
    BallRegisterPotential,
    DesignFreedomPotential,
    designed_payloads,
    site_separation,
)
from chlu.core.shard_store import (
    ShardedRegistry,
    allocate_sites,
    assert_per_shard_query_noise,
    build_sharded_store,
    r2_scores,
    route_from_scores,
    router_scores,
    sharded_two_phase,
)
from chlu.experiments.exp_designed_mechanism import (
    _agg,
    _atoms_for,
    _n_params,
    _replace,
    make_ball_queries,
)
from chlu.training.train_memory import train_memory_landscape

#: Arms this module knows how to build. ``monolithic`` is the w23/w24 line.
ARMS = ("monolithic", "monolithic_nx", "sharded_matched", "sharded_per_shard")


def parse_cell(cell: str):
    """``"6:64:2"`` -> ``(d=6, K=64, n_shards=2)``."""
    parts = str(cell).split(":")
    if len(parts) != 3:
        raise ValueError(f"cell must be 'd:K:n_shards', got {cell!r}")
    d, K, n = (int(x) for x in parts)
    if K % n:
        raise ValueError(f"cell {cell!r}: K={K} is not divisible by n_shards={n}")
    return d, K, n


# ---------------------------------------------------------------------------
# Building a store (monolithic or sharded) on the w23 geometry
# ---------------------------------------------------------------------------


def arm_spec(arm: str, d: int, K: int, n_shards: int, dm):
    """Resolve an arm into ``(n_shards_effective, atoms_per_shard, total_atoms)``.

    The atom budget is the one axis where "sharded vs monolithic" is genuinely
    ambiguous, so both readings are arms rather than a choice made post-hoc.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")
    mono = int(_atoms_for(dm, K, d))  # the w23 budget at this (d, K)
    if arm == "monolithic":
        return 1, mono, mono
    if arm == "monolithic_nx":
        # the laundering line for sharded_per_shard: SAME total atoms, one store
        per = int(_atoms_for(dm, K // n_shards, d))
        return 1, per * n_shards, per * n_shards
    if arm == "sharded_matched":
        per = int(round(mono / n_shards))
        return n_shards, per, per * n_shards
    per = int(_atoms_for(dm, K // n_shards, d))  # sharded_per_shard
    return n_shards, per, per * n_shards


def build_learned_shard(d, n_items, n_atoms, dm, key, group_centers=None,
                        local_radius=0.0):
    """One shard's LEARNED atom dictionary (w23's ``build_learned_V``, per shard).

    ``atom_groups = n_items`` keeps the w23 group structure (one contiguous atom
    block per item slot). ``group_centers`` / ``local_radius`` drive the **N98
    localized init** (build item 2): group j's atoms start in a ball of radius
    ``local_radius`` around item j's ADDRESS site.
    """
    return DesignFreedomPotential(
        rung="free_mlp",
        dim=d + 1,
        payloads=jnp.zeros((n_items,)),
        key=key,
        learned_family="atoms",
        n_atoms=int(n_atoms),
        rbf_init_width=dm.atom_init_width,
        confine=dm.learned_confine,
        atom_depth_init=dm.atom_depth_init,
        atom_groups=int(n_items),
        atom_init_scale=dm.atom_init_scale,
        atom_group_centers=group_centers,
        atom_local_radius=float(local_radius),
    )


def write_shard(V, targets, key, dm):
    """The w23/w24 baseline write operator (ONE global Adam dig), per shard.

    Identical to ``exp_designed_mechanism``'s ``learned_global`` arm — the arm the
    w23 capacity law was measured with. Sharding changes **how many valleys one dig
    must carve**, and nothing else about the operator.
    """
    return train_memory_landscape(
        V,
        jnp.asarray(targets),
        key,
        steps=dm.write_steps,
        lr=dm.write_lr,
        weight_decay=dm.write_weight_decay,
        loss_kwargs=dict(
            n_perturb=dm.write_n_perturb,
            sigma_addr=dm.write_sigma_addr,
            sigma_pay=dm.write_sigma_pay,
            margin=dm.write_margin,
            barrier=dm.write_barrier,
            payload_index=d_of(V),
        ),
    )


def d_of(V) -> int:
    """Address dimension of a shard potential (payload channel is the last coord)."""
    return int(V.dim) - 1


def build_store(arm, d, K, n_shards, seed, dm, ss, payloads=None, verbose=False):
    """Allocate -> partition -> build -> WRITE. Returns the store + bookkeeping.

    Returns ``(store, centers, payloads, groups, info)``. ``store`` is a
    :class:`ShardedStore` for every arm (a monolithic store is the ``N=1`` case, so
    both arms travel the *same* read path and no comparison can be contaminated by
    two different readers).
    """
    n_eff, atoms_per, atoms_total = arm_spec(arm, d, K, n_shards, dm)
    centers, groups, alloc = allocate_sites(
        d, K, n_eff, R=dm.R, seed=dm.site_seed,
        allocation=ss.allocation, partition=ss.partition,
    )
    if payloads is None:
        payloads = np.asarray(designed_payloads(K, seed=dm.payload_seed))
    payloads = np.asarray(payloads, dtype=np.float32)

    local_radius = (
        float(ss.atom_init_local_mult) * float(dm.atom_init_width)
        if ss.atom_init_local
        else 0.0
    )
    key = jax.random.PRNGKey(seed + 7919)
    Vs, hist_last, n_params = [], [], 0
    t0 = time.perf_counter()
    for g in groups:
        key, k_b, k_w = jax.random.split(key, 3)
        tgt = np.zeros((len(g), d + 1), dtype=np.float32)
        tgt[:, :d] = centers[g]
        tgt[:, d] = payloads[g]
        gc = centers[g] if ss.atom_init_local else None
        V = build_learned_shard(
            d, len(g), atoms_per, dm, k_b,
            group_centers=gc, local_radius=local_radius,
        )
        V, hist = write_shard(V, tgt, k_w, dm)
        Vs.append(V)
        n_params += _n_params(V)
        hist_last.append(float(np.mean(hist[-10:])) if hist else float("nan"))
    write_seconds = time.perf_counter() - t0

    store = build_sharded_store(Vs, groups, d=d, kinetic_mode="newtonian_learned")
    info = {
        "arm": arm,
        "n_shards": n_eff,
        "atoms_per_shard": atoms_per,
        "atoms_total": atoms_total,
        "n_learned_params": int(n_params),
        "write_loss_final": float(np.mean(hist_last)),
        "write_seconds": float(write_seconds),
        "atom_init_local_radius": local_radius,
        **alloc,
    }
    if verbose:
        print(f"    built {arm}: {info}", flush=True)
    return store, centers, payloads, groups, info


# ---------------------------------------------------------------------------
# The read + score (w23 metric, extended by routing)
# ---------------------------------------------------------------------------


def score_sharded(store, centers, payloads, dm, ss, seed, router=None, deadband=None):
    """Run the loop on ONE sharded store and score it — the unit of measurement.

    The metric is w23's, unchanged, with the routing decision folded in **as a
    failure mode, not as an excuse**: the predicted item is
    ``(routed shard, nearest center within that shard)``, so a routing miss counts
    exactly like a basin miss in the monolithic arm and the two numbers are directly
    comparable. Abstained queries count as failures in the per-offered metric
    (headline ``deadband=0`` => nothing abstains).
    """
    d, K = store.d, len(payloads)
    router = ss.router if router is None else router
    deadband = ss.abstain_deadband if deadband is None else deadband

    n_per = int(
        np.clip(dm.max_total_queries // K, dm.min_query_per_item, dm.n_query_per_item)
    )
    Q0, P0, labels = make_ball_queries(
        jax.random.PRNGKey(seed), centers, n_per, dm
    )
    # ⚠ §2.3 #10 fairness trap: the jitter must be sigma/sqrt(d_SHARD), never
    # sigma/sqrt(N*d). `make_ball_queries` uses centers.shape[1] == d, so this holds
    # by construction; asserted rather than asserted-in-prose.
    assert_per_shard_query_noise(
        dm.query_sigma, d, store.n_shards, dm.query_sigma / np.sqrt(d)
    )

    t0 = time.perf_counter()
    addr_x, feat = sharded_two_phase(store, Q0, P0, dm)  # (n,N,d), (n,N,n_sub)
    read_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = router_scores(router, store, Q0, addr_x=addr_x)
    route, margin, abstain = route_from_scores(scores, deadband)
    route_seconds = time.perf_counter() - t1

    shard_of = store.item_shard_map(K)
    route_ok = shard_of[labels] == route
    finite = bool(np.all(np.isfinite(addr_x)) and np.all(np.isfinite(feat)))

    # nearest center WITHIN the routed shard -> predicted global item id
    n = len(labels)
    pred = np.empty(n, dtype=int)
    x_sel = addr_x[np.arange(n), route]  # (n, d)
    for r, g in enumerate(store.shard_items):
        m = route == r
        if not np.any(m):
            continue
        gi = np.asarray(g, dtype=int)
        d2 = ((x_sel[m][:, None, :] - centers[gi][None, :, :]) ** 2).sum(-1)
        pred[m] = gi[np.argmin(d2, axis=1)]
    basin_ok = pred == labels

    read_val = feat[np.arange(n), route].mean(axis=1)
    err = np.abs(read_val - np.asarray(payloads)[labels])
    strict_ok = basin_ok & (err < dm.payload_tol) & (~abstain)

    answered = ~abstain
    return {
        "K": int(K),
        "n_shards": int(store.n_shards),
        "router": router,
        "deadband": float(deadband),
        "finite": finite,
        "strict_success_rate": float(np.mean(strict_ok)),
        "basin_success_rate": float(np.mean(basin_ok & answered)),
        "selectivity": float(np.mean(basin_ok & answered)),
        "route_accuracy": float(np.mean(route_ok)),
        "route_margin_mean": float(np.mean(margin[np.isfinite(margin)]))
        if np.any(np.isfinite(margin))
        else float("inf"),
        "abstain_rate": float(np.mean(abstain)),
        "strict_per_answered": float(
            np.mean(strict_ok[answered]) if np.any(answered) else 0.0
        ),
        "payload_abs_err_mean": float(np.mean(err)),
        "n_queries": int(n),
        "read_seconds": float(read_seconds),
        "route_seconds": float(route_seconds),
    }


def evaluate_cell(arm, d, K, n_shards, seed, dm, ss, with_blank=True):
    """Write + score one (arm, cell, seed), with the leak-immune value blank."""
    store, centers, payloads, groups, info = build_store(
        arm, d, K, n_shards, seed, dm, ss
    )
    written = score_sharded(store, centers, payloads, dm, ss, seed)
    out = {
        "arm": arm, "d": d, "K": K, "n_shards_requested": n_shards, "seed": seed,
        **info, "written": written,
    }
    if with_blank == "if_pass":
        with_blank = written["strict_success_rate"] >= dm.pass_strict
    if not with_blank:
        return out
    blank_pay = np.zeros_like(payloads)
    store_b, centers_b, _, _, _ = build_store(
        arm, d, K, n_shards, seed + 101, dm, ss, payloads=blank_pay
    )
    blank = score_sharded(store_b, centers_b, payloads, dm, ss, seed)
    # a blank landscape returns ~0 for every item, so it legitimately "retrieves"
    # any item whose stored value lies within payload_tol of 0 (w23 note).
    trivial = float(np.mean(np.abs(np.asarray(payloads)) < dm.payload_tol))
    out["blank"] = blank
    out["value_blank_ok"] = bool(
        blank["strict_success_rate"] <= max(dm.blank_strict_max, trivial + 0.02)
    )
    return out


def _record(cells, dm):
    strict = [c["written"]["strict_success_rate"] for c in cells]
    blanks = [c["value_blank_ok"] for c in cells if "value_blank_ok" in c]
    mean_strict = float(np.mean(strict))
    return {
        "arm": cells[0]["arm"], "d": cells[0]["d"], "K": cells[0]["K"],
        "n_shards": cells[0]["n_shards"],
        "seeds": [c["seed"] for c in cells],
        "strict": _agg(strict),
        "basin": _agg([c["written"]["basin_success_rate"] for c in cells]),
        "route_accuracy": _agg([c["written"]["route_accuracy"] for c in cells]),
        "payload_abs_err": _agg([c["written"]["payload_abs_err_mean"] for c in cells]),
        "write_loss_final": _agg([c["write_loss_final"] for c in cells]),
        "write_seconds": _agg([c["write_seconds"] for c in cells]),
        "read_seconds": _agg([c["written"]["read_seconds"] for c in cells]),
        "route_seconds": _agg([c["written"]["route_seconds"] for c in cells]),
        "atoms_total": cells[0]["atoms_total"],
        "atoms_per_shard": cells[0]["atoms_per_shard"],
        "n_learned_params": cells[0]["n_learned_params"],
        "union_separation": cells[0]["union_separation"],
        "within_shard_separation_min": cells[0]["within_shard_separation_min"],
        "n_blank_seeds": len(blanks),
        "n_value_blank_pass": int(sum(blanks)),
        "passes": bool(mean_strict >= dm.pass_strict and (all(blanks) if blanks else True)),
    }


def run_cell(arm, d, K, n_shards, dm, ss, seeds, verbose=True):
    cells = []
    for n, s in enumerate(seeds):
        cells.append(
            evaluate_cell(
                arm, d, K, n_shards, s, dm, ss,
                with_blank="if_pass" if n < ss.blank_seeds else False,
            )
        )
    rec = _record(cells, dm)
    if rec["passes"] and rec["n_blank_seeds"] == 0:
        cells[0] = evaluate_cell(arm, d, K, n_shards, seeds[0], dm, ss, with_blank=True)
        rec = _record(cells, dm)
    if verbose:
        print(
            f"  [{arm}] d={d} K={K} N={rec['n_shards']} strict="
            f"{rec['strict']['mean']:.3f}+-{rec['strict']['std']:.3f}"
            f" route={rec['route_accuracy']['mean']:.3f}"
            f" atoms={rec['atoms_total']} sep(union)={rec['union_separation']:.3f}"
            f" sep(shard)={rec['within_shard_separation_min']:.3f}"
            f" -> {'PASS' if rec['passes'] else 'fail'}"
            f" [{rec['write_seconds']['mean']:.0f}s write, "
            f"{rec['read_seconds']['mean']:.0f}s read]",
            flush=True,
        )
    return rec


# ---------------------------------------------------------------------------
# Item 1 -- ⭐ the §5.1 2x2 discriminator
# ---------------------------------------------------------------------------


def item1_discriminator(dm, ss):
    rows = []
    for cell in ss.cells:
        d, K, n = parse_cell(cell)
        print(f"[cell] {cell}", flush=True)
        for arm in ss.arms:
            rows.append(
                {"cell": cell, **run_cell(arm, d, K, n, dm, ss, ss.seeds)}
            )
    # per-cell verdicts: sharded vs its OWN laundering line
    verdicts = []
    for cell in ss.cells:
        got = [r for r in rows if r["cell"] == cell]

        def _g(a, got=got):
            return next((r for r in got if r["arm"] == a), None)

        mono, mono_nx = _g("monolithic"), _g("monolithic_nx")
        sm, sp = _g("sharded_matched"), _g("sharded_per_shard")
        v = {"cell": cell}
        if mono:
            v["monolithic_strict"] = mono["strict"]["mean"]
            v["monolithic_passes"] = mono["passes"]
        if sm and mono:
            v["matched_delta"] = sm["strict"]["mean"] - mono["strict"]["mean"]
            v["matched_strict"] = sm["strict"]["mean"]
            v["matched_passes"] = sm["passes"]
        if sp and mono_nx:
            v["per_shard_delta_vs_nx"] = sp["strict"]["mean"] - mono_nx["strict"]["mean"]
        if sp:
            v["per_shard_strict"] = sp["strict"]["mean"]
            v["per_shard_passes"] = sp["passes"]
        v["sharded_fixes_the_wall"] = bool(
            (sm is not None and sm["passes"]) or (sp is not None and sp["passes"])
        )
        verdicts.append(v)
    return {"rows": rows, "verdicts": verdicts, "cells": list(ss.cells),
            "arms": list(ss.arms), "seeds": list(ss.seeds)}


def item1_verdict(item1, ss):
    """The 2x2 reading (the theorist's decision rule, mechanised)."""
    by_cell = {v["cell"]: v for v in item1["verdicts"]}

    def fixed(cell):
        v = by_cell.get(cell)
        return bool(v and v.get("sharded_fixes_the_wall"))

    main = [c for c in ss.cells if c.startswith(("6:", "8:64"))]
    control = [c for c in ss.cells if c.startswith("4:")]
    any_main = any(fixed(c) for c in main)
    any_control = any(fixed(c) for c in control)
    if any_main and any_control:
        verdict = "CEILING-ENTIRELY-PER-DIG"  # K_total = 32N unclamped
    elif any_main:
        verdict = "THEOREM-L1"  # K = min(K_addr, 32N)
    else:
        verdict = "ADDITIVITY-REFUTED"
    return {
        "verdict": verdict,
        "main_cells_fixed": [c for c in main if fixed(c)],
        "control_cells_fixed": [c for c in control if fixed(c)],
        "note": (
            "CEILING-ENTIRELY-PER-DIG = both the main cells AND the geometry-bound "
            "d=4 control are fixed by sharding (K_total = 32N unclamped, a bigger "
            "and different result); THEOREM-L1 = main cells fixed, control still "
            "fails (K = min(K_addr, N*K_ceiling)); ADDITIVITY-REFUTED = no main "
            "cell is fixed -> R2-route-ii is dead and Theorem L1's conditions must "
            "be re-audited starting at W3 (the theorist's own §1.4)."
        ),
    }


# ---------------------------------------------------------------------------
# Item 2 -- the read-parity check (§5.3): DESIGNED shards, no write
# ---------------------------------------------------------------------------


def build_designed_shards(d, K, n_shards, dm, ss, allocation, seed):
    """N designed ``BallRegisterPotential`` shards on a global/local allocation.

    No write at all: this isolates the READ side, which is exactly what Prop L4 is
    a claim about. A learned store would confound read-side discrimination with the
    write ceiling under test in item 1.
    """
    centers, groups, alloc = allocate_sites(
        d, K, n_shards, R=dm.R, seed=dm.site_seed + seed,
        allocation=allocation, partition=ss.partition,
    )
    payloads = np.asarray(designed_payloads(K, seed=dm.payload_seed))
    Vs = [
        BallRegisterPotential(
            payloads[g], centers[g], R=dm.R + dm.wall_margin, w=dm.well_width,
            b=dm.well_depth, kappa=dm.payload_kappa, c_conf=dm.c_conf,
        )
        for g in groups
    ]
    return build_sharded_store(Vs, groups, d=d), centers, payloads, alloc


def item2_read_parity(dm, ss):
    """Global vs local allocation, N = 1..8, at fixed ``K_total`` (§5.3 / Prop L4)."""
    d, K = ss.parity_d, ss.parity_K
    dmp = _replace(dm, n_query_per_item=ss.parity_n_query_per_item)
    rows = []
    for allocation in ss.parity_allocations:
        for n in ss.parity_shards:
            if K % n:
                continue
            got, seps, seps_in = [], [], []
            for router in ss.routers:
                per_seed = []
                for s in ss.parity_seeds:
                    store, centers, payloads, alloc = build_designed_shards(
                        d, K, n, dmp, ss, allocation, s
                    )
                    seps.append(alloc["union_separation"])
                    seps_in.append(alloc["within_shard_separation_min"])
                    per_seed.append(
                        score_sharded(store, centers, payloads, dmp, ss, s,
                                      router=router, deadband=0.0)
                    )
                got.append(
                    {
                        "router": router,
                        "strict": _agg([g["strict_success_rate"] for g in per_seed]),
                        "route_accuracy": _agg([g["route_accuracy"] for g in per_seed]),
                        "basin": _agg([g["basin_success_rate"] for g in per_seed]),
                    }
                )
            rows.append(
                {
                    "allocation": allocation, "n_shards": n, "d": d, "K": K,
                    "union_separation": float(np.mean(seps)),
                    "within_shard_separation_min": float(np.mean(seps_in)),
                    "by_router": got,
                }
            )
            best = max(g["strict"]["mean"] for g in got)
            print(
                f"  [parity] alloc={allocation} N={n} strict(best router)={best:.3f}"
                f" sep(union)={alloc['union_separation']:.3f}",
                flush=True,
            )
    # parity verdict against the N=1 monolithic line
    def _strict(alloc_name, n, router):
        r = next(
            (x for x in rows if x["allocation"] == alloc_name and x["n_shards"] == n),
            None,
        )
        if r is None:
            return None
        g = next((y for y in r["by_router"] if y["router"] == router), None)
        return None if g is None else g["strict"]["mean"]

    base = _strict(ss.parity_allocations[0], 1, ss.router)
    parity = []
    for allocation in ss.parity_allocations:
        for n in ss.parity_shards:
            v = _strict(allocation, n, ss.router)
            if v is None or base is None:
                continue
            parity.append(
                {"allocation": allocation, "n_shards": n, "strict": v,
                 "delta_vs_monolithic": v - base}
            )
    glob = [p for p in parity if p["allocation"] == "global" and p["n_shards"] > 1]
    return {
        "rows": rows, "parity": parity, "monolithic_strict": base,
        "global_parity_holds": bool(
            glob and max(abs(p["delta_vs_monolithic"]) for p in glob) <= 0.02
        ),
        "max_abs_global_delta": (
            max(abs(p["delta_vs_monolithic"]) for p in glob) if glob else None
        ),
        "note": (
            "PROP L4: sharding relaxes WRITE-side crowding and CONSERVES READ-side "
            "discrimination. Global allocation must retrieve at parity with the "
            "monolithic store; local (per-shard) allocation must degrade with N. If "
            "GLOBAL degrades with N, Prop L4 is wrong and there is an unmodelled "
            "cross-shard channel -> ESCALATE."
        ),
    }


# ---------------------------------------------------------------------------
# Item 3 -- wall-clock: routing vs read (the O(1)-in-depth claim)
# ---------------------------------------------------------------------------


def item3_timing(dm, ss):
    """R2 routing (NO dynamics) vs one two-phase read (1200 Verlet steps).

    The design rule the theorist derived: **the routing statistic must be evaluable
    WITHOUT running the dynamics**, or the read costs O(N) full settles. R2 satisfies
    it; R1 would not, and R1 is also the one that does not work (N97).
    """
    d, K = ss.timing_d, ss.timing_K
    dmt = _replace(dm, n_query_per_item=2, max_total_queries=ss.timing_queries)
    rows = []
    for n in ss.timing_shards:
        if K % n:
            continue
        store, centers, payloads, _ = build_designed_shards(d, K, n, dmt, ss, "global", 0)
        n_q = ss.timing_queries
        Q0, P0, _ = make_ball_queries(
            jax.random.PRNGKey(0), centers, max(1, n_q // K), dmt
        )
        Q0, P0 = Q0[:n_q], P0[:n_q]
        # warm the jit for both paths, then time
        _ = r2_scores(store, Q0)
        _ = sharded_two_phase(store, Q0[:1], P0[:1], dmt)
        t_route, t_read = [], []
        for _rep in range(ss.timing_repeats):
            t0 = time.perf_counter()
            s = r2_scores(store, Q0)
            s.sum()
            t_route.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            x, _f = sharded_two_phase(store, Q0, P0, dmt)
            x.sum()
            t_read.append(time.perf_counter() - t0)
        route_ms = 1e3 * float(np.median(t_route)) / n_q
        read_ms = 1e3 * float(np.median(t_read)) / n_q
        rows.append(
            {
                "n_shards": n, "d": d, "K": K, "n_queries": n_q,
                "route_ms_per_query": route_ms,
                "read_ms_per_query": read_ms,
                "route_over_read": route_ms / max(read_ms, 1e-12),
                "rollout_steps": int(dmt.address_steps + dmt.read_steps),
            }
        )
        print(
            f"  [timing] N={n} route={route_ms:.4f} ms  read={read_ms:.3f} ms  "
            f"ratio={route_ms / max(read_ms, 1e-12):.4f}",
            flush=True,
        )
    mono = rows[0]["read_ms_per_query"] if rows else None
    return {
        "rows": rows,
        "max_route_over_read": max((r["route_over_read"] for r in rows), default=None),
        "read_ms_by_n_shards": {str(r["n_shards"]): r["read_ms_per_query"] for r in rows},
        "read_growth_over_smallest_N": (
            max(r["read_ms_per_query"] for r in rows) / mono if mono else None
        ),
        "note": (
            "Routing is O(N) SCALAR potential evaluations at ONE point and needs no "
            "dynamics; the read is ONE joint rollout that settles every shard "
            "simultaneously (V is separable), so it is O(1) in rollout DEPTH."
        ),
    }


# ---------------------------------------------------------------------------
# Item 4 -- the global allocator (build item 3), as a registry
# ---------------------------------------------------------------------------


def item4_allocator(dm, ss):
    """Global registry vs per-shard registries: what the union separation does.

    Offers ``alloc_demo_K`` items round-robin across ``alloc_demo_shards`` MVC-0
    controllers, once with the spacing test run against the UNION of all shards'
    addresses (the global allocator) and once with each controller seeing only its
    own (the control). The measured quantity is the **union** minimum separation
    against ``d_safe`` — the quantity Theorem L1's condition W4 is about.
    """
    s = float(ss.alloc_demo_s)
    d_safe = D_SAFE_MULT * s
    K, N = int(ss.alloc_demo_K), int(ss.alloc_demo_shards)
    R = float(ss.alloc_demo_radius)
    rng = np.random.default_rng(ss.alloc_demo_seed)
    # one proposal stream, replayed identically for both regimes
    ang = rng.uniform(0, 2 * np.pi, K)
    rad = R * np.sqrt(rng.uniform(0, 1, K))
    proposals = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    out = {}
    for regime, global_alloc in (("global", True), ("local", False)):
        ctls = [
            Controller(
                AtomStorePotential(dim=3, capacity=K, s=s), d_safe=d_safe,
                budget=K, amp=1.0, n_candidates=400,
            )
            for _ in range(N)
        ]
        reg = ShardedRegistry(ctls, global_alloc=global_alloc)
        for i in range(K):
            reg.offer_round_robin(
                i, proposals[i], float(i) / K,
                key=jax.random.PRNGKey(1000 + i),
            )
        union = reg.stored_addresses()
        within = [
            float(site_separation(c.stored_addresses()))
            for c in ctls
            if c.n_live > 1
        ]
        out[regime] = {
            "n_live": int(union.shape[0]),
            "union_separation": float(site_separation(union)),
            "within_shard_separation_min": float(min(within)) if within else float("inf"),
            "d_safe": d_safe,
            "union_respects_d_safe": bool(site_separation(union) >= d_safe - 1e-6),
            "stats": reg.stats(),
        }
        print(
            f"  [allocator] {regime}: live={out[regime]['n_live']} "
            f"union_sep={out[regime]['union_separation']:.3f} (d_safe={d_safe:.3f}) "
            f"respects={out[regime]['union_respects_d_safe']}",
            flush=True,
        )
    return {
        "regimes": out,
        "packing_radius_for_K": radius_for_capacity(K, d_safe),
        "note": (
            "The global allocator is the ONLY global object in a sharded store, and "
            "it is a REGISTRY, not an optimizer: no gradient and no optimizer state "
            "crosses a shard boundary — only the list of where things were written."
        ),
    }


# ---------------------------------------------------------------------------
# Item 5 -- the N98 localized-init ablation (build item 2)
# ---------------------------------------------------------------------------


def item5_init_ablation(dm, ss):
    """Localized atom init ON vs OFF, applied identically to BOTH arms.

    ⚠ Kept OUT of the headline 2x2 on purpose: it is an INITIALISATION, so folding
    it into the headline would confound the sharding effect with an init effect (and
    invite an N46 fairness attack). If it alone lifts the MONOLITHIC cell over the
    bar, that is a finding about w23's initialisation, not about sharding.
    """
    rows = []
    for cell in ss.init_ablation_cells:
        d, K, n = parse_cell(cell)
        for local in (False, True):
            ss_l = _replace(ss, atom_init_local=local)
            for arm in ("monolithic", "sharded_matched"):
                rec = run_cell(arm, d, K, n, dm, ss_l, ss.init_ablation_seeds)
                rows.append({"cell": cell, "atom_init_local": local, **rec})
    deltas = []
    for cell in ss.init_ablation_cells:
        for arm in ("monolithic", "sharded_matched"):
            on = next(
                (r for r in rows if r["cell"] == cell and r["arm"] == arm
                 and r["atom_init_local"]), None
            )
            off = next(
                (r for r in rows if r["cell"] == cell and r["arm"] == arm
                 and not r["atom_init_local"]), None
            )
            if on and off:
                deltas.append(
                    {
                        "cell": cell, "arm": arm,
                        "strict_off": off["strict"]["mean"],
                        "strict_on": on["strict"]["mean"],
                        "delta": on["strict"]["mean"] - off["strict"]["mean"],
                    }
                )
    return {"rows": rows, "deltas": deltas,
            "radius_mult": ss.atom_init_local_mult,
            "note": "address axes only; the payload axis keeps the w23 scatter (N46)"}


# ---------------------------------------------------------------------------
# Item 6 -- the abstention deadband sweep (secondary; never in a headline)
# ---------------------------------------------------------------------------


def item6_deadband(dm, ss):
    """Top-2 deadband sweep on a DESIGNED sharded store (per-offered vs per-answered).

    A product-store routing miss is total (the wrong shard answers with full
    confidence), so a deployed design needs an abstention rule. Reported per-offered
    AND per-answered together, always (N91).
    """
    d, K = ss.parity_d, ss.parity_K
    n = max([x for x in ss.parity_shards if x > 1 and K % x == 0] or [2])
    dmp = _replace(dm, n_query_per_item=ss.parity_n_query_per_item)
    store, centers, payloads, _ = build_designed_shards(
        d, K, n, dmp, ss, ss.allocation, 0
    )
    rows = []
    for db in ss.deadband_sweep:
        got = score_sharded(store, centers, payloads, dmp, ss, 0, deadband=db)
        rows.append(
            {
                "deadband": float(db),
                "strict_per_offered": got["strict_success_rate"],
                "strict_per_answered": got["strict_per_answered"],
                "abstain_rate": got["abstain_rate"],
                "route_accuracy": got["route_accuracy"],
            }
        )
    return {"rows": rows, "d": d, "K": K, "n_shards": n,
            "router": ss.router,
            "note": "per-offered AND per-answered always travel together (N91)"}


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
    it1 = results.get("item1_discriminator")
    it2 = results.get("item2_read_parity")
    if not it1 and not it2:
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    if it1:
        cells = it1["cells"]
        arms = it1["arms"]
        w = 0.8 / max(len(arms), 1)
        for j, arm in enumerate(arms):
            vals, errs = [], []
            for c in cells:
                r = next(
                    (x for x in it1["rows"] if x["cell"] == c and x["arm"] == arm), None
                )
                vals.append(r["strict"]["mean"] if r else np.nan)
                errs.append(r["strict"]["std"] if r else 0.0)
            a1.bar(np.arange(len(cells)) + j * w, vals, width=w, yerr=errs,
                   label=arm, capsize=2)
        a1.axhline(0.9, color="k", ls=":", lw=1.0, label="strict bar 0.9")
        a1.set_xticks(np.arange(len(cells)) + 0.4 - w / 2)
        a1.set_xticklabels(cells, fontsize=8)
        a1.set_ylabel("strict success")
        a1.set_title(
            "sharded vs monolithic: "
            + str(results.get("item1_verdict", {}).get("verdict", ""))
        )
        a1.legend(fontsize=7)
    if it2:
        for allocation in {r["allocation"] for r in it2["rows"]}:
            xs, ys = [], []
            for r in sorted(
                [x for x in it2["rows"] if x["allocation"] == allocation],
                key=lambda x: x["n_shards"],
            ):
                g = next(
                    (y for y in r["by_router"] if y["router"] == "R2"), r["by_router"][0]
                )
                xs.append(r["n_shards"])
                ys.append(g["strict"]["mean"])
            a2.plot(xs, ys, "o-", lw=1.8, label=f"{allocation} allocation")
        a2.axhline(it2["monolithic_strict"] or 1.0, color="k", ls=":", lw=0.9,
                   label="monolithic")
        a2.set_xlabel("number of shards $N$")
        a2.set_ylabel("strict success (designed shards)")
        a2.set_title("read-side: Prop L4")
        a2.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(save_dir, "sharded_store_fig1.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_sharded_store(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    config = config or get_default_config()
    ss = config.experiment_sharded_store
    # geometry / retrieval / write budget are w23's, with the declared read-cost
    # reduction applied identically to every arm.
    dm = _replace(
        config.experiment_designed_mechanism, n_query_per_item=ss.n_query_per_item
    )
    seed = config.project.seed if seed is None else seed
    os.makedirs(save_dir, exist_ok=True)

    bad = [a for a in ss.arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; known: {ARMS}")
    for c in ss.cells:
        parse_cell(c)

    results = {
        "seed": seed,
        "config": {
            "sharded_store": {k: getattr(ss, k) for k in vars(ss)},
            "designed_mechanism": {
                k: getattr(dm, k)
                for k in (
                    "R", "wall_margin", "well_width", "well_depth", "payload_kappa",
                    "c_conf", "site_seed", "payload_seed", "dt", "gamma_address",
                    "gamma_read", "address_steps", "read_steps", "tail_frac",
                    "n_subsample", "n_query_per_item", "max_total_queries",
                    "query_sigma", "query_sigma_p", "payload_tol", "pass_strict",
                    "blank_strict_max", "atoms_per_item", "min_atoms",
                    "min_atoms_base", "min_atoms_c", "atom_init_scale",
                    "atom_init_width", "atom_depth_init", "learned_confine",
                    "write_steps", "write_lr", "write_weight_decay",
                    "write_n_perturb", "write_sigma_addr", "write_sigma_pay",
                    "write_margin", "write_barrier",
                )
            },
        },
    }
    if ss.run_allocator:
        print("[item 4] global allocator (registry) vs per-shard", flush=True)
        results["item4_allocator"] = item4_allocator(dm, ss)
    if ss.run_read_parity:
        print("[item 2] read parity: global vs local allocation", flush=True)
        results["item2_read_parity"] = item2_read_parity(dm, ss)
    if ss.run_timing:
        print("[item 3] wall-clock: routing vs read", flush=True)
        results["item3_timing"] = item3_timing(dm, ss)
    if ss.run_deadband_sweep:
        print("[item 6] abstention deadband sweep", flush=True)
        results["item6_deadband"] = item6_deadband(dm, ss)
    if ss.run_discriminator:
        print("[item 1] the 2x2 discriminator", flush=True)
        results["item1_discriminator"] = item1_discriminator(dm, ss)
        results["item1_verdict"] = item1_verdict(results["item1_discriminator"], ss)
    if ss.run_init_ablation:
        print("[item 5] N98 localized-init ablation", flush=True)
        results["item5_init_ablation"] = item5_init_ablation(dm, ss)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_sharded_store_metrics.json")
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
    """Quick smoke: the same code path, tiny sweeps. NOT a scientific run."""
    ss = config.experiment_sharded_store
    dm = config.experiment_designed_mechanism
    ss.cells = ["2:4:2"]
    ss.arms = ["monolithic", "sharded_matched", "sharded_per_shard", "monolithic_nx"]
    ss.seeds = [0]
    ss.n_query_per_item = 4
    ss.parity_d = 2
    ss.parity_K = 4
    ss.parity_shards = [1, 2]
    ss.parity_seeds = [0]
    ss.parity_n_query_per_item = 4
    ss.timing_d = 2
    ss.timing_K = 4
    ss.timing_shards = [2]
    ss.timing_queries = 8
    ss.timing_repeats = 1
    ss.alloc_demo_K = 8
    ss.alloc_demo_shards = 2
    ss.init_ablation_cells = ["2:4:2"]
    ss.init_ablation_seeds = [0]
    ss.deadband_sweep = [0.0, 0.5]
    dm.atoms_per_item = 8
    dm.min_atoms = 32
    dm.min_atoms_base = 16
    dm.min_atoms_c = 2.0
    dm.address_steps = 200
    dm.read_steps = 150
    dm.write_steps = 30
    dm.write_n_perturb = 8
    dm.max_total_queries = 32


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment SHARDED-STORE: is the K~=32 write ceiling per-dig? "
        "(the first N-unit CLU store + the §5.1 2x2 discriminator)"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--cells", nargs="+", help="Override the cells (d:K:n_shards)")
    parser.add_argument("--arms", nargs="+", help="Override the arms")
    parser.add_argument("--items", nargs="+",
                        help="Subset of items to run (1..6)")
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
    if args.cells:
        config.experiment_sharded_store.cells = list(args.cells)
    if args.arms:
        config.experiment_sharded_store.arms = list(args.arms)
    if args.items:
        ss = config.experiment_sharded_store
        want = set(args.items)
        ss.run_discriminator = "1" in want
        ss.run_read_parity = "2" in want
        ss.run_timing = "3" in want
        ss.run_allocator = "4" in want
        ss.run_init_ablation = "5" in want
        ss.run_deadband_sweep = "6" in want

    res = run_experiment_sharded_store(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )
    if "item1_verdict" in res:
        print(json.dumps(res["item1_verdict"], indent=2))
    print("metrics ->", res["metrics_path"])


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "apply_quick",
    "arm_spec",
    "build_designed_shards",
    "build_learned_shard",
    "build_store",
    "evaluate_cell",
    "item1_discriminator",
    "item1_verdict",
    "item2_read_parity",
    "item3_timing",
    "item4_allocator",
    "item5_init_ablation",
    "item6_deadband",
    "parse_cell",
    "run_cell",
    "run_experiment_sharded_store",
    "score_sharded",
    "write_shard",
]
