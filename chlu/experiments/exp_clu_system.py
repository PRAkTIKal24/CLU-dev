"""Experiment CLU-SYSTEM: the full CLU, every lever live, staged (C2W1).

**The wave's spine** (charter §6.1). Twenty-six waves tested one lever at a time,
each in a configuration chosen to make that lever legible — and legibility was
bought, every time, by turning the other levers off. This experiment runs the
**whole object**: items in a learned ``V_theta``, derived addressing, an admission
policy, per-item lifetimes, a local/masked write, permitted basin interaction, a
two-phase read whose **trajectory** is available to the read-out, a
confidence-gated retry, and controller v0 over the designed verb set — with all
13 anti-collapse monitors running as **loud runtime guards**.

**Acceptance is "does not collapse", not "wins."** No performance claim is made
here and no baseline is run. What is reported is the trip-state of every monitor
at every stage, including the ones that never fired (labelled UNTESTED, never
green), and the dividend as an *instrument reading*.

**Staged activation, not big-bang** (charter §3.1). Every lever starts in its
known productive band — the defaults of :class:`~chlu.core.clu_system.CluSystemConfig`
are the shipped w22-w26 values, not guesses — and exactly one lever is freed per
stage, *inside the running full system*:

======================  ====================================================
stage                   lever freed
======================  ====================================================
``S0_baseline``         (none) learned store + derived addressing + masked
                        write + two-phase read + settled-point psi
``S1_lifetimes``        per-item decay (physical: the item's own atom rows)
``S2_admission``        admission gate + capacity pressure (budget < offered)
``S3_deletion``         an explicit deletion demand mid-stream
``S4_basin``            **basin interaction** — the address ball is shrunk
                        until wells overlap (engineered separability removed;
                        this is the anti-§8.2 stage and the one that matters)
``S5_retry``            confidence-gated retry (the compute dial)
``S6_trajectory``       the trajectory read-out (psi sees the strided buffer)
======================  ====================================================

⛔ **The hard falsifier, carried in code.** If the only configuration in which no
monitor trips is the degenerate one — explicit arrays, engineered separation,
settled-point-only read — then w26's degenerate configuration has been
re-derived and intervention §8.2 has fired. That is a **finding**, reported as
the headline. Moving *toward* the lookup configuration to obtain a clean number
is forbidden, so ``S4`` (basin interaction) is never rolled back to make a
number look better.

Runnable directly::

    uv run python -m chlu.experiments.exp_clu_system --quick

or via the CLI: ``chlu exp-clu-system [--project N] [--seed I] [--quick]``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from chlu.core.clu_controller import GuardViolation
from chlu.core.clu_system import (
    CluSystemConfig,
    build_system,
    settled_point_psi,
    tail_mean_psi,
)
from chlu.core.memory_potentials import designed_payloads, designed_sites
from chlu.eval.dividend import (
    ByteAccount,
    byte_account,
    dividend,
    same_keys_null,
    settle_deleted_launder,
)

STAGES = (
    "S0_baseline",
    "S1_lifetimes",
    "S2_admission",
    "S3_deletion",
    "S4_basin",
    "S5_retry",
    "S6_trajectory",
)


# --------------------------------------------------------------------------
# staged configuration — cumulative, one lever per stage
# --------------------------------------------------------------------------
def stage_config(stage: str, base: CluSystemConfig,
                 n_offer: Optional[int] = None) -> CluSystemConfig:
    """The config for ``stage``: every earlier stage's lever stays on."""
    i = STAGES.index(stage)
    n_offer = int(n_offer or base.capacity + 2)
    cfg = replace(base)
    if i >= 1:  # S1 lifetimes
        cfg = replace(cfg, stage_lifetimes=True, leak=0.02)
    if i >= 2:  # S2 admission + capacity pressure
        cfg = replace(cfg, stage_admission=True, stage_capacity_pressure=True,
                      budget=max(2, base.capacity - 2))
    if i >= 3:  # S3 deletion demand
        cfg = replace(cfg, stage_deletion=True)
    if i >= 4:
        # S4 BASIN INTERACTION — the anti-§8.2 stage. The ball is shrunk until the
        # sites are closer than two well widths, so the wells genuinely overlap.
        # ⚠ The merge certificate `2 s_max + kappa' sigma_q <= sep` CANNOT hold in
        # that regime — permitted basin interaction and the merge certificate are
        # mutually exclusive by construction — so the admission radius is taken
        # DELIBERATELY out of band (`d_safe_override`) and monitor #8/#3 are
        # expected to say so. Leaving the in-band radius here would simply refuse
        # every item after the first and produce a spotless, empty store, which is
        # the degenerate configuration the task forbids settling into.
        R = base.ball_radius * 0.45
        sep_expected = 2.0 * R * n_offer ** (-1.0 / base.addr_dim)
        cfg = replace(cfg, stage_basin_interaction=True, ball_radius=R,
                      d_safe_override=float(0.6 * sep_expected))
    if i >= 5:  # S5 the compute dial
        cfg = replace(cfg, stage_retry=True, retry_max_rounds=2, retry_tau=0.8)
    if i >= 6:  # S6 the trajectory read
        cfg = replace(cfg, stage_trajectory_read=True)
    return cfg


def make_stream(cfg: CluSystemConfig, n_offer: int, seed: int = 0):
    """A nontrivial write stream: farthest-point sites, non-monotone payloads.

    Returns ``(items, sites, payloads)``. The stream carries real capacity
    pressure whenever ``n_offer > cfg.budget`` and a deletion demand when
    ``cfg.stage_deletion``.
    """
    sites = np.asarray(designed_sites(cfg.addr_dim, n_offer, R=cfg.ball_radius,
                                      seed=seed))
    pays = np.asarray(designed_payloads(n_offer, seed=seed))
    items = []
    for i in range(n_offer):
        items.append({"item_id": i, "address": sites[i], "payload": float(pays[i])})
        # ⭐ A COLLISION OFFER halfway through. Without one, a farthest-point
        # stream is genuinely well-separated and the admission gate legitimately
        # never fires — the doctrine's characterised false-trip mode for monitor
        # #3. A stream with "real capacity pressure" must contain a proposal the
        # gate can refuse, so the harness puts one there rather than reading a
        # fire-rate of 0 as a controller failure.
        if i == n_offer // 2:
            items.append({"item_id": 1000 + i, "address": sites[i] * 1.005,
                          "payload": float(-pays[i])})
        # The deletion demand. ⚠ It must name an item that is still LIVE: under
        # capacity pressure the early ids have already been evicted, and a delete
        # of an absent id is a no-op that silently empties the stage of its
        # deletion demand (measured: `deleted=[]` on the first full run).
        if cfg.stage_deletion and i == n_offer - 3:
            items.append({"item_id": i - 1, "delete": True})
    return items, sites, pays


# --------------------------------------------------------------------------
# evaluation (label-side; the monitors are label-free)
# --------------------------------------------------------------------------
def evaluate(system, seed: int = 0) -> dict:
    """Decode accuracy + value error of the full CLU read, and its launders.

    ``decode`` = nearest-stored-payload decoding (never the ``tol`` metric,
    which is vacuous at m>1, N110).
    """
    import jax

    ids, centers, pays = system.codebook()
    if len(ids) == 0:
        return {"n_live": 0}
    cfg = system.cfg
    key = jax.random.PRNGKey(int(seed) + 4242)
    n_per = int(cfg.n_query_per_item)
    labels = np.repeat(np.arange(len(ids)), n_per)
    jit = np.asarray(jax.random.normal(key, (labels.size, cfg.addr_dim))) * cfg.query_sigma
    q0 = np.zeros((labels.size, system.store.dim), dtype=np.float32)
    q0[:, : cfg.addr_dim] = centers[labels] + jit
    queries = q0[:, : cfg.addr_dim]

    t0 = time.time()
    res = system.read(q0)
    read_s = time.time() - t0
    val = np.asarray(res.value).reshape(labels.size, -1)
    dec = np.argmin(np.linalg.norm(val[:, None, :] - pays[None, :, :], axis=-1), axis=1)
    acc = float(np.mean(dec == labels))
    rmse = float(np.sqrt(np.mean(np.sum((val - pays[labels]) ** 2, axis=-1))))

    # --- the launders (same keys, same phi, same admitted set) ---
    l_assign = settle_deleted_launder(centers, pays, queries, metric="assign")
    l_acc = float(np.mean(l_assign == labels))
    null_val = same_keys_null(centers, pays, queries, np.random.default_rng(seed))
    null_dec = np.argmin(
        np.linalg.norm(null_val[:, None, :] - pays[None, :, :], axis=-1), axis=1
    )
    null_acc = float(np.mean(null_dec == labels))

    return {
        "n_live": int(len(ids)),
        "decode_acc": acc,
        "value_rmse": rmse,
        "chance": 1.0 / len(ids),
        "launder_settle_deleted_acc": l_acc,
        "launder_same_keys_null_acc": null_acc,
        "n_steps_per_read": int(res.n_steps),
        "retries": int(res.retries),
        "read_wall_s": read_s,
        "read": res,
        "labels": labels,
        "queries": q0,
    }


def blank_control(cfg: CluSystemConfig, seed: int, live_pays: np.ndarray,
                  q0: np.ndarray, labels: np.ndarray, psi=None) -> dict:
    """Read an identically-configured store with **nothing written into it**."""
    import jax

    blank = build_system(replace(cfg, seed=seed + 991), key=jax.random.PRNGKey(seed + 991),
                         psi=psi, loud=False)
    res = blank.read(q0)
    val = np.asarray(res.value).reshape(labels.size, -1)
    dec = np.argmin(
        np.linalg.norm(val[:, None, :] - live_pays[None, :, :], axis=-1), axis=1
    )
    score = float(np.mean(dec == labels))
    _, counts = np.unique(labels, return_counts=True)
    chance = float(np.max(counts) / labels.size)
    se = float(np.sqrt(max(chance * (1 - chance), 1e-12) / labels.size))
    return {"score": score, "chance": chance, "se": se, "bar": chance + 3 * se,
            "metric": "decode",
            "representation": getattr(psi, "representation", "settled_point")}


def knob_liveness(system, base_eval: dict, knobs: Sequence[str]) -> dict:
    """Monitor #10 tier (b): does each declared read-side dial move an observable?

    Only read/control dials are swept ("declared active this run"); a write-side
    dial would need a re-write and is declared inactive rather than swept, per the
    doctrine's guard on tier (b).
    """
    out = {}
    q0 = base_eval["queries"]
    ref_val = np.asarray(base_eval["read"].value)
    ref_steps = float(base_eval["read"].n_steps)
    noise = max(float(np.std(ref_val)) * 1e-3, 1e-9)
    for knob in knobs:
        cfg = system.cfg
        old = getattr(cfg, knob)
        try:
            if knob == "read_steps":
                setattr(cfg, knob, int(old * 2))
            elif knob == "gamma_read":
                setattr(cfg, knob, float(old) * 4.0)
            elif knob == "address_steps":
                setattr(cfg, knob, int(old * 2))
            elif knob == "traj_stride":
                setattr(cfg, knob, max(1, int(old) // 2))
            else:
                setattr(cfg, knob, float(old) * 2.0)
            res = system.read(q0)
            d_val = float(np.max(np.abs(np.asarray(res.value) - ref_val)))
            d_steps = abs(float(res.n_steps) - ref_steps)
            out[knob] = float(max(d_val / noise, d_steps))
        finally:
            setattr(cfg, knob, old)
    return out


def canary_stream(cfg: CluSystemConfig, seed: int = 0) -> dict:
    """M14: a stream **constructed to require every designed guard**.

    Each guard gets a situation that must fire it: an unreachable payload
    (``admit.reach``), a duplicate address (``admit.merge``), an over-budget
    offer (``admit.budget`` -> ``evict``), a below-threshold utility
    (``admit.priority``), a decay tick, a routing call, a retry beyond budget, an
    anneal schedule and an expand. A guard whose count is 0 afterwards is
    arithmetically vacuous (N74, transplanted to a policy).
    """
    import jax

    # the canary uses the FULL write budget: a guard that cannot fire because the
    # canary's wells were never dug is a property of the canary, not of the policy.
    c = replace(cfg, capacity=4, budget=3, write_steps=cfg.write_steps,
                stage_admission=True, stage_lifetimes=True, leak=0.05,
                retry_max_rounds=1, n_query_per_item=2, seed=seed)
    sysx = build_system(c, key=jax.random.PRNGKey(seed + 31), loud=False)
    sites = np.asarray(designed_sites(c.addr_dim, 4, R=c.ball_radius, seed=seed))
    ctrl = sysx.controller
    ctrl.policy = replace(ctrl.policy, admit_priority_threshold=0.5)
    stream = [
        {"item_id": 0, "address": sites[0], "payload": 0.3, "utility": 1.0},
        {"item_id": 1, "address": sites[1], "payload": 0.4, "utility": 1.0},
        # below the priority threshold -> admit.priority
        {"item_id": 2, "address": sites[2], "payload": 0.2, "utility": 0.1},
        # a duplicate address -> admit.merge (spacing certificate)
        {"item_id": 3, "address": sites[0], "payload": -0.4, "utility": 1.0},
        # an unreachable payload -> admit.reach
        {"item_id": 4, "address": sites[3], "payload": 25.0, "utility": 1.0},
        {"item_id": 5, "address": sites[3], "payload": -0.2, "utility": 1.0},
        {"item_id": 6, "address": -sites[0], "payload": 0.5, "utility": 1.0},
    ]
    sysx.write_stream(stream)
    ctrl.decay(1)
    ctrl.route(sites[0], signal="address")
    ctrl.retry(0.0, round_index=0)
    ctrl.retry(0.0, round_index=99)  # beyond the compute budget -> retry.budget
    ctrl.anneal([2.0, 1.0])
    ctrl.expand(1.2)
    ctrl.place(0, sites[0], lambda_min=-1.0)  # -> place.lambda_min (refused)
    # place.injective — a re-derivation that lands on ANOTHER live item's site
    live_ids = [r.item_id for r in ctrl.allocator.records.values()]
    if len(live_ids) >= 2:
        other = ctrl.allocator.records
        centres = {r.item_id: r.center for r in other.values()}
        a, b = live_ids[0], live_ids[1]
        ctrl.place(a, centres[b], lambda_min=1.0)  # -> place.injective (refused)
    # evict with too little persistence -> evict.persistence
    ctrl.policy = replace(ctrl.policy, evict_persistence_W=3)
    ctrl.evict(0, reason="policy", trips=0)
    # admit.budget — a full store of PERMANENT items is a capacity alarm, never a
    # silent overwrite. Built explicitly: nothing in a healthy stream produces it.
    c2 = replace(c, capacity=2, budget=2)
    sys2 = build_system(c2, key=jax.random.PRNGKey(seed + 77), loud=False)
    s2 = np.asarray(designed_sites(c2.addr_dim, 3, R=c2.ball_radius, seed=seed))
    sys2.write_stream([
        {"item_id": 0, "address": s2[0], "payload": 0.2, "permanent": True},
        {"item_id": 1, "address": s2[1], "payload": -0.2, "permanent": True},
        {"item_id": 2, "address": s2[2], "payload": 0.4},
    ])
    # evict.class_i — an eviction attempted while an INSTRUMENT monitor is
    # tripped must be refused (doctrine §5, consequence 1). The situation is
    # constructed with a blank-store reading above its own bar.
    from chlu.core.monitors import MonitorContext
    sys2.registry.observe(MonitorContext(
        stage="canary", t=0,
        blank={"score": 1.0, "chance": 0.25, "se": 0.01, "metric": "decode"},
        extras={},
    ))
    sys2.controller.evict(0, reason="policy", trips=99)
    for g, v in sys2.controller.guard_fire_counts().items():
        counts_extra = v
        if counts_extra:
            ctrl._guard_counts[g] = ctrl._guard_counts.get(g, 0) + counts_extra
    # the guards that a *healthy* canary cannot fire without breaking a rule
    # are exercised as exceptions instead (they raise, by design):
    for verb, kwargs in (
        ("route", {"signal": "energy"}),
        ("anneal", {}),
        ("evict", {"reason": "lru", "trips": 9}),
        ("expand", {}),
    ):
        try:
            if verb == "route":
                ctrl.route(sites[0], **kwargs)
            elif verb == "anneal":
                ctrl.anneal([2.0, 1.5])
            elif verb == "evict":
                ctrl.evict(0, **kwargs)
            else:
                ctrl.expand(0.5)
        except GuardViolation:
            pass
    counts = ctrl.guard_fire_counts()
    return {"counts": counts, "verbs": ctrl.verb_counts()}


# --------------------------------------------------------------------------
# one stage
# --------------------------------------------------------------------------
#: Remediation arms — "a trip that is cleared by returning the responsible lever
#: to its known productive band is not a failure, it is the ablation table"
#: (task file, acceptance criterion). Each arm names the monitor it targets and
#: the RESTORING VERB it applies; nothing here moves toward the lookup
#: configuration (that is forbidden), and each is a documented band, not a tune.
REMEDIATIONS = {
    # #5 addressing: the annealed read is the MEASURED fix (N109), and its own
    # `readonly` control localises the entire gain to address acquisition.
    "R1_anneal_read": {"base": "S0_baseline", "anneal": [4.0, 2.0, 1.0],
                       "targets": "addressing(#5)", "verb": "anneal"},
    # #8-N3 certificates: re-derive each address by RELAXATION and re-place it,
    # committing only where lambda_min(H) > 0 (never a critical-point solver).
    "R2_place_pass": {"base": "S0_baseline", "place_pass": True,
                      "targets": "certificates(#8-N3)", "verb": "place"},
    # #1 overdamping: gamma_read 0.02 sits below the doctrine's measured
    # convergence band [0.05, 0.5] at N=400; this returns it to the band.
    # #1 overdamping fires only UNDER the annealed read (the schedule leaves the
    # settle less converged), so the arm that tests it must start there and spend
    # the compute dial: same schedule, 2x the read budget.
    "R3_anneal_plus_steps": {"base": "S0_baseline", "anneal": [4.0, 2.0, 1.0],
                             "cfg": {"read_steps": 1600},
                             "targets": "overdamping(#1) under the annealed read",
                             "verb": "retry/anneal(steps)"},
    # #4 blank under a TRAJECTORY read: the raw buffer contains q0 = phi(x)
    # (doctrine I-2), so the same psi is re-run on the store-relative form.
    "R4_traj_store_relative": {"base": "S6_trajectory", "store_relative": True,
                               "targets": "blank(#4)", "verb": "psi representation"},
}


def run_stage(stage: str, base: CluSystemConfig, seed: int, n_offer: int,
              with_knob_sweep: bool = False, label: Optional[str] = None,
              cfg_mutate: Optional[dict] = None, anneal: Optional[list] = None,
              place_pass: bool = False, store_relative: bool = False) -> dict:
    import jax

    cfg = stage_config(stage, replace(base, seed=seed), n_offer=n_offer)
    if cfg_mutate:
        cfg = replace(cfg, **cfg_mutate)
    psi = (tail_mean_psi(cfg.addr_dim, cfg.payload_dim)
           if cfg.stage_trajectory_read
           else settled_point_psi(cfg.addr_dim, cfg.payload_dim))
    system = build_system(cfg, key=jax.random.PRNGKey(seed), psi=psi, loud=True)
    items, sites, pays = make_stream(cfg, n_offer, seed=seed)

    if store_relative:
        # doctrine I-2: the same psi on the STORE-RELATIVE trajectory, so a
        # blank-store read cannot be "a classifier on phi(x)".
        from chlu.core.clu_system import store_relative_trajectory as _srt
        inner = psi
        def psi_rel(traj, state):
            return inner(_srt(traj, state), state)
        psi_rel.representation = "trajectory_tail_store_relative"
        system.psi = psi_rel
        psi = psi_rel

    t0 = time.time()
    wrep = system.write_stream(items, key=jax.random.PRNGKey(seed + 1))
    write_s = time.time() - t0

    if anneal is not None:
        system.controller.anneal(anneal)  # the `anneal` verb sets the read schedule
    if place_pass:
        system.consolidate(place_pass=True)

    ev = evaluate(system, seed=seed)
    ids, centers, live_pays = system.codebook()
    blank = (blank_control(cfg, seed, live_pays, ev["queries"], ev["labels"], psi=psi)
             if ev.get("n_live") else None)

    sweep = None
    if with_knob_sweep and ev.get("n_live"):
        knobs = ["read_steps", "gamma_read", "address_steps"]
        # traj_stride can only move an observable through a psi that reads the
        # trajectory; it is declared ACTIVE only then (doctrine tier-(b) guard:
        # a knob at a no-op value is not a dead knob).
        if cfg.stage_trajectory_read:
            knobs.append("traj_stride")
        sweep = knob_liveness(system, ev, knobs)
    canary = canary_stream(cfg, seed=seed)

    probe = system.self_probe()
    certs = system.certificates()
    extras = {"canary_guard_counts": canary["counts"]}
    if sweep is not None:
        extras["knob_sweep"] = sweep
    readings = system.observe(stage=stage, self_probe=probe, certificates=certs,
                              blank=blank, reads=ev["read"].diagnostics, extras=extras)

    ba: ByteAccount = byte_account(system, centers, live_pays)
    div = dividend(
        ev["decode_acc"], ev["launder_settle_deleted_acc"], metric="decode_acc",
        controls={"same_keys_null": ev["launder_same_keys_null_acc"],
                  "blank_store": (blank["score"] if blank else float("nan")),
                  "blank_bar": (blank["bar"] if blank else float("nan")),
                  "chance": ev["chance"]},
        bytes_account=ba,
        flags={"stage": stage, "seed": seed, **cfg.as_flag_table()},
    )
    Ds, ss = system.well_fits()
    degenerate = int(ev.get("n_live", 0)) < 2
    if degenerate:
        print(f"⛔ DEGENERATE STAGE {stage}: n_live={ev.get('n_live')} — a store with "
              f"fewer than 2 live items cannot support ANY reported metric; a clean "
              f"monitor table here is the empty-store artefact, not an acceptance.")
    return {
        "stage": label or stage,
        "base_stage": stage,
        "seed": seed,
        "degenerate": degenerate,
        "remediation": {"anneal": anneal, "place_pass": place_pass,
                        "cfg_mutate": cfg_mutate, "store_relative": store_relative}
        if (anneal or place_pass or cfg_mutate or store_relative) else None,
        "config_non_default": cfg.as_flag_table(),
        "n_offered": n_offer,
        "admitted": wrep.admitted,
        "refused": wrep.refused,
        "evicted": wrep.evicted,
        "deleted": wrep.deleted,
        "write_losses": wrep.losses,
        "write_wall_s": write_s,
        "eval": {k: v for k, v in ev.items() if k not in ("read", "queries", "labels")},
        "self_probe": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in probe.items()
                       if k in ("acq", "strict", "decode", "chance",
                                "delta_read_basin_conditioned", "retention",
                                "payload_abs", "write_loss")},
        "certificates": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in certs.items()},
        "well_fits": {"D": Ds.tolist(), "s": ss.tolist(),
                      "fairness": float(np.min(Ds) / max(np.max(Ds), 1e-12))},
        "blank": blank,
        "knob_sweep": sweep,
        "canary_guard_counts": canary["counts"],
        "verb_counts": canary["verbs"],
        "dividend": div.as_dict(),
        "monitors": [r.as_dict() for r in readings],
        "trips": [r.name for r in readings if r.tripped],
    }


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_clu_system(
    config=None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    stages: Optional[Sequence[str]] = None,
    quick: bool = False,
    n_offer: Optional[int] = None,
    overrides: Optional[dict] = None,
    remediate: bool = True,
) -> dict:
    """Run the staged full-CLU harness and write the trip-state artifact.

    ``config`` is a :class:`~chlu.config.CHLUConfig` (used only for project
    paths/seed); the harness's own knobs live in :class:`CluSystemConfig`, which
    is read from a ``clu_system:`` block of the project YAML if present. This
    keeps the harness config-driven **without touching ``chlu/config.py``**,
    which C1W27 owns this wave.
    """
    os.makedirs(save_dir, exist_ok=True)
    seed = int(seed if seed is not None else getattr(
        getattr(config, "project", None), "seed", 0) or 0)
    base = CluSystemConfig.from_mapping(overrides or _project_overrides(config))
    base = replace(base, seed=seed)
    stages = list(stages or STAGES)
    n_offer = int(n_offer or base.capacity + 2)
    if quick:
        base = replace(base, capacity=4, write_steps=40, address_steps=100,
                       read_steps=200, n_query_per_item=2, quick=True,
                       min_atoms=64, min_atoms_base=64, atoms_per_item=8)
        n_offer = 5
        stages = stages[:3]
        remediate = False

    results = {
        "seed": seed,
        "stages_run": stages,
        "n_offer": n_offer,
        "base_config": {k: v for k, v in base.__dict__.items()},
        "defaults_are_shipped_band": (
            "atoms_per_item 32 + w23 dimension-aware floor; atom_init_width 0.3; "
            "gamma_address 0.05 / gamma_read 0.02; 400/800 steps; sigma_q 0.15; "
            "write_steps 300 (local masked write); confine 0.05"
        ),
        "results": [],
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_clu_system_metrics.json")

    def _dump():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_json_default)

    for i, stage in enumerate(stages):
        t0 = time.time()
        rec = run_stage(stage, base, seed, n_offer,
                        with_knob_sweep=(i == 0 or stage == "S6_trajectory"))
        rec["stage_wall_s"] = time.time() - t0
        results["results"].append(rec)
        print(f"[{stage}] decode={rec['eval'].get('decode_acc')} "
              f"launder={rec['eval'].get('launder_settle_deleted_acc')} "
              f"dividend={rec['dividend']['dividend']:+.4f} "
              f"trips={rec['trips']} ({rec['stage_wall_s']:.0f}s)")
        _dump()

    if remediate:
        for name, spec in REMEDIATIONS.items():
            if spec["base"] not in stages:
                continue
            t0 = time.time()
            rec = run_stage(spec["base"], base, seed, n_offer, label=name,
                            cfg_mutate=spec.get("cfg"), anneal=spec.get("anneal"),
                            place_pass=bool(spec.get("place_pass")),
                            store_relative=bool(spec.get("store_relative")))
            rec["stage_wall_s"] = time.time() - t0
            rec["targets"] = spec["targets"]
            rec["verb"] = spec["verb"]
            results["results"].append(rec)
            print(f"[{name}] targets={spec['targets']} verb={spec['verb']} "
                  f"decode={rec['eval'].get('decode_acc')} trips={rec['trips']} "
                  f"({rec['stage_wall_s']:.0f}s)")
            _dump()

    results["trip_table"] = _trip_table(results["results"])
    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        results["figures"] = []
        results["figure_error"] = repr(exc)
    _dump()
    results["metrics_path"] = out_path
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _trip_table(records) -> dict:
    """monitor -> {stage: tripped/inapplicable/clear} — the reported artifact."""
    table = {}
    for rec in records:
        for m in rec["monitors"]:
            row = table.setdefault(m["name"], {"mode": m["mode"], "stages": {}})
            row["stages"][rec["stage"]] = (
                "TRIP" if m["tripped"]
                else ("inapplicable" if not m["applicable"] else "clear")
            )
    for row in table.values():
        states = set(row["stages"].values())
        row["ever_tripped"] = "TRIP" in states
        row["untested"] = "TRIP" not in states
    return table


def _plot(results: dict, save_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = results["results"]
    stages = [r["stage"] for r in recs]
    x = np.arange(len(stages))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(x, [r["eval"].get("decode_acc", np.nan) for r in recs], "o-", label="full CLU")
    ax.plot(x, [r["eval"].get("launder_settle_deleted_acc", np.nan) for r in recs],
            "s--", label="settle-deleted launder")
    ax.plot(x, [r["eval"].get("launder_same_keys_null_acc", np.nan) for r in recs],
            "^:", label="same-keys null")
    ax.plot(x, [(r["blank"] or {}).get("score", np.nan) for r in recs], "v:",
            label="blank store")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("decode accuracy")
    ax.set_title("read vs its launders (instrument reading)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    div = [r["dividend"]["dividend"] for r in recs]
    ax.bar(x, div, color=["tab:green" if d > 0 else "tab:red" for d in div])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("dividend (full - launder)")
    ax.set_title("dynamics dividend — <=0 at v0 is the honest start")
    ax.grid(alpha=0.3)

    ax = axes[2]
    names = sorted(results["trip_table"], key=lambda n: results["trip_table"][n]["mode"])
    grid = np.zeros((len(names), len(stages)))
    for i, n in enumerate(names):
        for j, s in enumerate(stages):
            st = results["trip_table"][n]["stages"].get(s, "n/a")
            grid[i, j] = {"TRIP": 2, "clear": 1, "inapplicable": 0}.get(st, 0)
    ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=2)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([f"#{results['trip_table'][n]['mode']} {n}" for n in names],
                       fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=8)
    ax.set_title("monitor trip state (red=TRIP, green=clear, grey=inapplicable)")

    fig.tight_layout()
    p = os.path.join(save_dir, "exp_clu_system_stages.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def _project_overrides(config) -> dict:
    """Read a ``clu_system:`` block from the project YAML, if any."""
    path = getattr(getattr(config, "project", None), "config_path", None)
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return dict(raw.get("clu_system", {}))
    except Exception:
        return {}


def apply_quick(config) -> None:
    """Quick smoke settings (same code path, smaller everything)."""
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Experiment CLU-SYSTEM: the full CLU, every lever live, staged."
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES),
                        help="Run only these stages (default: all)")
    parser.add_argument("--offer", type=int, help="How many items the stream offers")
    parser.add_argument("--no-remediate", action="store_true",
                        help="Skip the remediation arms (the restoring-verb table)")
    args = parser.parse_args()

    config = None
    save_dir, models_dir = "results", None
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        os.makedirs(save_dir, exist_ok=True)

    res = run_experiment_clu_system(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed,
        stages=args.stages, quick=args.quick, n_offer=args.offer,
        remediate=not args.no_remediate,
    )
    print(json.dumps(res["trip_table"], indent=2)[:3000])


if __name__ == "__main__":
    main()
