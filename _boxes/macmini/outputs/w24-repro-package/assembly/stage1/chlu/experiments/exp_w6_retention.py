"""w6 ``hub/retention`` -- DELETION-BY-DECAY on the field store: the COMPOSITION curve.

⛔⛔ READ THIS BEFORE READING A NUMBER THIS MODULE PRODUCES
===========================================================
Wave 5 settled both ends of this hub's axis on the *same object*:

* the **coupling hole** is exact and **free** -- refuses at ``1.0000`` for **1 byte**, 0
  outside, no refit, **0.0000 composition cost**, 101 other never-held conjunctions
  untouched, zero seed spread;
* **instant eviction is NOT free** -- it refuses perfectly but its composition delta is
  ``-2.0899 / -0.9101 / -6.7416`` against a pre-registered margin of ``1.0``, and a fenced
  post-hoc control showed the cost is the *mechanism* (an item store losing 6 of 64 shared
  atoms) rather than the target: evicting 6 RANDOM non-red items cost ``-2.7584``.

Decay is eviction spread over time.  ⭐⭐ **So the question is whether a lifetime buys
anything eviction does not, or simply pays the same composition cost more slowly** -- and a
refusal curve without the composition curve beside it answers nothing.  This module
measures **both curves on one axis**, and the axis is the **law's own**: half-lives.

⛔ THE HALF OF THE ANSWER THAT IS ALREADY CLOSED, IN CLOSED FORM, BEFORE ANY RUN
--------------------------------------------------------------------------------
:mod:`chlu.core.gen_retention` transcribes the shorts' exact-decay law (V2 App. C Eqs.
(3)-(6); V5 App. D) and verifies it against a simulation of its own map.  Two consequences
are **derived, not measured**, and they shape everything below:

1. ``gen_field`` calls an item dead at ``LIFETIME_EPS = 1e-6`` and ``gap_read``'s
   ``empty_basin`` -- what refusal reads -- is exactly that liveness bit.  So a decaying
   item's refusal horizon is ``n_refuse = n_half * ln(1e-6)/ln(1/2) = 19.9316 * n_half``,
   **for every curvature**.
2. Both horizons carry the same ``1/(-ln rho)``, so the spectral mass ``mu^2`` **cancels**:
   in half-life units every lifetime in the family is the same curve ``2**(-t)``.
   ⛔ **No curvature trades refusal against composition.**  ``mu^2`` sets the clock rate.

⭐ That is why the horizon grid here is declared **from the law** (:data:`HORIZONS`) and not
chosen by looking at any data: there is nothing to choose.  The grid brackets the derived
refusal horizon ``19.9316`` on both sides and the run reports where the composition cost
has already converged to instant eviction's.

⭐ THE SECOND LAW THIS HUB TESTS -- V5 App. G.8, "Decay commutes with deletion"
-------------------------------------------------------------------------------
    *"Let the store state be the live key set K with per-item amplitudes {a_i}, and let the
    layout be the canonical placement L(K) ... a function of K and the fixed geometry alone,
    and in particular independent of the amplitudes.  Amplitude decay acts item-wise,
    a_i -> f(a_i), touching neither K nor L(K); deletion removes one key and recanonicalizes,
    touching neither the surviving amplitudes nor their decay clocks.  The two operations act
    on disjoint components of the state (K, {a_i}), so they commute, and the post-delete store
    is byte-identical to the store built from the never-inserted history with the same elapsed
    decay ...  Recency eviction makes L a function of the insertion history rather than of K,
    which is exactly why it is excluded."*

The field store's acceptance property is already byte-level, so this is directly testable.
:func:`commutation_test` tests **both halves separately**, because they are not the same
claim and this hub expects them to come apart:

``A. commutation``     ``evict o decay^n`` vs ``decay^n o evict``, byte-diffed.
``B. never-inserted``  the post-delete store vs a store **built from a pool with the item's
                       rows removed**, decayed the same amount.  ⚠ The field store places
                       its 64 atoms by **greedy span selection** and fits ``term_mu`` on the
                       pool, neither of which is the canonical Blelloch-Golovin placement the
                       paper's guarantee is conditioned on.  ⛔ A failure here is a failure of
                       the **precondition**, not of the law, and is reported in those words.

The arms
--------
Every w5 arm is inherited verbatim by wrapping :func:`exp_gen_delete_row7.build_arms` at
runtime -- ⛔ nothing in that module is edited on disk -- so instant eviction, the hole, the
table row-removal, retrain-from-scratch, the 8-byte gate and the charter section C controls
are on **exactly** the instrument that produced w5's numbers, and this run re-measures
eviction's ``-2.0899/-0.9101/-6.7416`` as an independent replication rather than quoting it.

On top of that, one arm per horizon::

    field/decay@t=<t>   the deleted value's items decayed to lifetime rho**n,
                        n = round(t * n_half), rho from the law at the pinned mu^2.

⚠ CHURN is a **COST LINE only**, one definition, ⛔ never a claim.

Runnable::

    python -m chlu.experiments.exp_w6_retention --stage law      --out DIR
    python -m chlu.experiments.exp_w6_retention --stage commute  --seed S --out DIR
    python -m chlu.experiments.exp_w6_retention --stage dev      --seed S --out DIR
    python -m chlu.experiments.exp_w6_retention --stage score    --seed S --out DIR
    python -m chlu.experiments.exp_w6_retention --stage report   --out DIR
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from . import exp_gen_delete_row7 as R7
from ..core import gen_retention as LAW
_R7_BUILD_ARMS = R7.build_arms
PINS = {
    # -- inherited verbatim: the store, the condition, the targets, the seeds --------
    "inherits": "exp_gen_delete_row7.PINS (which inherits exp_gen_field_store.PINS)",
    "condition": R7.PINS["condition"],
    "seeds": R7.PINS["seeds"],
    "delete_factor": R7.PINS["delete_factor"],     # object_hue
    "delete_value": R7.PINS["delete_value"],       # = 0.0, red
    "hole_pair": R7.PINS["hole_pair"],
    # -- the law's own parameters.  ⛔ BOTH are the papers' declared values, not dials ----
    # V5 "Nomenclature": "Integration step (eps): Fixed at 0.05."  V2 App. F: dt = 0.05.
    "eps": 0.05,
    # V2 Sec. 4.1 / Fig. 1 / Fig. 4 and V5 App. C both quote the floor AT gamma = 0.05.
    "gamma": 0.05,
    # -- the decay clock.  ⭐ Chosen from the LAW's own landmarks, not from data --------
    # The primary lifetime is the paper's curvature-independent FLOOR, 2ln2/(-ln(1-gamma)),
    # which is the SHORTEST half-life the law permits at this gamma -- a landmark, not a fit.
    "primary_half_life_is": "the paper's curvature-independent floor 2 ln2/(-ln(1-gamma))",
    # The invariance check uses a DEEP OVERDAMPED clock at 10x the floor.  ⭐ It needs no
    # judge pass: the claim is that the two produce the SAME lifetime at the same t, which is
    # a byte-level identity on the store, not a scored measurement.
    "invariance_half_life_multiple": 10.0,
    "store_bytes": R7.PINS["store_bytes"],
}
HORIZONS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 19.0,
            LAW.refusal_horizon_half_lives(), 20.5, 22.0, 24.0)
W5_EVICT_COMPOSITION_DELTA = {0: -2.0899, 1: -0.9101, 2: -6.7416}
W5_HOLE_COMPOSITION_DELTA = 0.0
W5_RANDOM_SIZE_MATCHED_DELTA = -2.7584
def _now() -> float:
    return time.time()


def _jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: (
            float(o) if isinstance(o, (np.floating, np.integer)) else
            (o.tolist() if isinstance(o, np.ndarray) else str(o))))


def horizon_schedule(eps: Optional[float] = None, gamma: Optional[float] = None,
                     half_life: Optional[float] = None) -> dict:
    """⭐ The decay schedule: one integer step count per horizon, from the law.

    ``half_life`` defaults to the paper's curvature-independent floor at this ``gamma``.  The
    curvature is then recovered by inverting the **exact** closed form (never the paper's
    asymptotic), the per-step retention factor ``rho`` is read off it, and each horizon ``t``
    becomes an integer number of map steps ``n = round(t * n_half)``.

    ⭐ The refusal step is computed, not searched: ``n_refuse = ceil(ln(eps_life)/ln(rho))``,
    and the two integer steps that bracket it are inserted into the grid so the transition is
    located exactly rather than interpolated.
    """
    eps = float(PINS["eps"] if eps is None else eps)
    gamma = float(PINS["gamma"] if gamma is None else gamma)
    fl = LAW.floor_half_life(gamma)
    nh = float(fl if half_life is None else half_life)
    if nh <= fl * (1.0 - 1e-12):
        raise ValueError(f"half-life {nh} is below the law's floor {fl}")
    if abs(nh - fl) <= fl * 1e-12:
        # the floor is reached by ANY underdamped mode; take one a decade above h*
        mu2 = ((10.0 * LAW.crossover_h(gamma)) / eps) ** 2
        nh = LAW.n_half_closed_form(mu2, eps, gamma)
    else:
        mu2 = LAW.omega2_for_half_life(nh, eps, gamma)
        nh = LAW.n_half_closed_form(mu2, eps, gamma)
    rho = LAW.retention_factor(mu2, eps, gamma)
    n_refuse = int(math.ceil(math.log(LAW.LIFETIME_DEAD_EPS) / math.log(rho)))
    ts = sorted(set(float(t) for t in HORIZONS))
    steps = sorted({0} | {int(round(t * nh)) for t in ts}
                   | {max(0, n_refuse - 1), n_refuse})
    rows = []
    for n in steps:
        life = float(np.float32(rho ** n))      # the store holds float32 lifetimes
        rows.append({
            "n_steps": int(n),
            "t_half_lives": float(n) / nh,
            "lifetime": life,
            "lifetime_f64": float(rho ** n),
            "dead": bool(life <= LAW.LIFETIME_DEAD_EPS),
            "is_refusal_step": bool(n == n_refuse),
        })
    return {
        "eps": eps, "gamma": gamma, "mu2": float(mu2), "h": LAW.h_of(mu2, eps),
        "h_over_h_star": LAW.h_of(mu2, eps) / LAW.crossover_h(gamma),
        "band": LAW.band(mu2, eps, gamma),
        "rho": float(rho), "n_half": float(nh),
        "n_half_floor": float(fl),
        "n_half_paper_asymptotic": LAW.n_half_paper_asymptotic(mu2, eps, gamma),
        "n_refuse_steps": int(n_refuse),
        "n_refuse_half_lives": float(n_refuse) / nh,
        "derived_refusal_horizon_half_lives": LAW.refusal_horizon_half_lives(),
        "lifetime_eps": LAW.LIFETIME_DEAD_EPS,
        "rows": rows,
    }


def arm_name(n: int, t: float) -> str:
    return f"field/decay@n={int(n)}"


def decay_variants(base: dict, seed: int, sched: Optional[dict] = None,
                   log: Callable = print) -> dict:
    """One field image per horizon: the deleted value's items decayed by ``rho**n``.

    ⛔ Nothing refits.  :func:`gen_field.decay_items` moves lifetime bytes, and when an item's
    lifetime crosses ``LIFETIME_EPS`` its content is zeroed too -- which is the whole point:
    past the refusal horizon a decayed store should be **byte-identical to the evicted store**,
    and :func:`byte_properties` asserts exactly that rather than asserting about it.
    """
    from ..core.gen_field import _items_holding, decay_items

    sched = sched or horizon_schedule()
    f0 = base["field"]
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    ids, prov, geom = _items_holding(f0, fac, valx)
    out, ops = {}, {}
    for r in sched["rows"]:
        n = int(r["n_steps"])
        fd = f0.copy()
        if n > 0:
            op = decay_items(fd, ids, float(r["lifetime"]), log=(lambda *_: None))
            ops[n] = {k: v for k, v in op.items() if k != "permitted_ranges"}
        else:
            ops[n] = {"op": "decay_items", "n_items": int(ids.size), "rate": 1.0,
                      "n_died": 0, "died": [], "refit": False}
        out[n] = fd
    log(f"  decay(seed {seed}): {len(out)} horizons over {ids.size} item(s) "
        f"({prov.size} by provenance, {geom.size} by geometry); rho={sched['rho']:.12f}, "
        f"n_half={sched['n_half']:.4f}, refusal step {sched['n_refuse_steps']}")
    return {"fields": out, "_ops": ops, "item_ids": ids.tolist(),
            "n_by_provenance": int(prov.size), "n_by_geometry": int(geom.size),
            "schedule": sched}


def byte_properties(base: dict, dv: dict, fv: dict, log: Callable = print) -> dict:
    """⛔ The exact, mechanical half of this hub.  No judge, no corners, no tolerance.

    Per horizon: how many bytes moved against the undecayed base, in which segments, whether
    any FITTED segment moved (``assert_no_refit``), and -- ⭐ the one that answers the target
    sentence at byte level -- whether the decayed image is **byte-identical to the instantly
    evicted image**.
    """
    from ..core.gen_field import assert_no_refit, byte_diff

    f0, fe = fv["base"], fv["evict_red"]
    rows = []
    for n in sorted(dv["fields"]):
        fd = dv["fields"][n]
        d0 = byte_diff(f0, fd)
        de = byte_diff(fe, fd)
        nr = assert_no_refit(f0, fd)
        rows.append({
            "n_steps": int(n),
            "bytes_changed_vs_base": int(d0["n_changed"]),
            "segments_touched_vs_base": dict(d0["per_segment"]),
            "bytes_differing_from_INSTANT_EVICTION": int(de["n_changed"]),
            "segments_differing_from_eviction": dict(de["per_segment"]),
            "identical_to_instant_eviction": bool(de["n_changed"] == 0),
            "no_refit": bool(all(nr["fitted_segments_byte_identical"].values())),
            "n_live_atoms": int((np.asarray(fd.view("atom_alive"), np.uint8) > 0).sum()),
            "min_lifetime_of_target_items": float(np.asarray(
                fd.view("atom_life"), np.float32)[np.asarray(dv["item_ids"], np.int64)].min())
            if dv["item_ids"] else float("nan"),
        })
    first_ident = next((r["n_steps"] for r in rows if r["identical_to_instant_eviction"]), None)
    return {"rows": rows, "total_image_bytes": int(np.asarray(f0.buf).size),
            "first_step_byte_identical_to_eviction": first_ident,
            "statement": (
                "decay moves ONLY lifetime bytes until an item dies, at which point its "
                "content is zeroed too; the step at which the decayed image becomes "
                f"byte-identical to the instantly evicted image is {first_ident}")}


def invariance_on_the_store(base: dict, seed: int, log: Callable = print) -> dict:
    """⭐⭐ The reparameterisation collapse, ON THE STORE, at byte level.

    The law says ``mu^2`` is a pure time reparameterisation.  The store-level form of that
    claim is: decaying to horizon ``t`` under a **floor-rate** clock and under a **deep
    overdamped** clock ten times slower produces the **same lifetime**, hence the **same
    image**.  ⛔ This needs no judge pass and spends no scored budget: it is a byte identity.
    """
    from ..core.gen_field import _items_holding, byte_diff, decay_items

    s1 = horizon_schedule()
    s2 = horizon_schedule(half_life=float(PINS["invariance_half_life_multiple"])
                          * LAW.floor_half_life(float(PINS["gamma"])))
    f0 = base["field"]
    ids, _, _ = _items_holding(f0, int(PINS["delete_factor"]), int(PINS["delete_value"]))
    rows = []
    # ⚠ The two clocks are matched on STEPS, not on the horizon label: `n2` is the slow
    # clock's nearest integer step to the same elapsed fraction of a half-life.  ⛔ The
    # residual is then purely the slow clock's integer quantisation -- it cannot be removed
    # and it is reported rather than hidden, because a map step is an integer in the paper too.
    for n1 in sorted({int(round(float(t) * s1["n_half"])) for t in HORIZONS}):
        t = float(n1) / s1["n_half"]
        n2 = int(round(float(n1) * s2["n_half"] / s1["n_half"]))
        l1 = float(np.float32(s1["rho"] ** n1))
        l2 = float(np.float32(s2["rho"] ** n2))
        a, b = f0.copy(), f0.copy()
        if n1:
            decay_items(a, ids, l1, log=(lambda *_: None))
        if n2:
            decay_items(b, ids, l2, log=(lambda *_: None))
        d = byte_diff(a, b)
        rows.append({"t_half_lives": float(t), "n_steps_floor_clock": n1,
                     "n_steps_slow_clock": n2,
                     "t_half_lives_slow_clock": float(n2) / s2["n_half"],
                     "lifetime_floor_clock": l1,
                     "lifetime_slow_clock": l2,
                     "lifetime_predicted_2_pow_minus_t": float(2.0 ** -t),
                     "law_abs_dev_floor_clock": abs(l1 - 2.0 ** -t),
                     "abs_lifetime_gap": abs(l1 - l2),
                     "rel_lifetime_gap": abs(l1 - l2) / max(l1, 1e-30),
                     "bytes_differing": int(d["n_changed"]),
                     "images_identical": bool(d["n_changed"] == 0)})
    worst = max(r["abs_lifetime_gap"] for r in rows)
    worst_rel = max(r["rel_lifetime_gap"] for r in rows)
    worst_law = max(r["law_abs_dev_floor_clock"] for r in rows)
    n_ident = sum(1 for r in rows if r["images_identical"])
    log(f"  invariance(seed {seed}): {n_ident}/{len(rows)} horizons byte-identical across "
        f"two clocks ({s1['n_half']:.3f} vs {s2['n_half']:.3f} steps/half-life); worst "
        f"lifetime gap {worst:.3e}")
    return {"clock_floor": {k: s1[k] for k in ("mu2", "h", "band", "rho", "n_half",
                                               "n_refuse_steps")},
            "clock_slow": {k: s2[k] for k in ("mu2", "h", "band", "rho", "n_half",
                                              "n_refuse_steps")},
            "rows": rows, "worst_abs_lifetime_gap": float(worst),
            "worst_rel_lifetime_gap": float(worst_rel),
            "worst_abs_dev_of_floor_clock_from_2_pow_minus_t": float(worst_law),
            "n_horizons_byte_identical": int(n_ident), "n_horizons": len(rows),
            "statement": (
                "the store image depends on the horizon ONLY through the float32 lifetime, and "
                "the law says that lifetime is 2**-t. Two genuinely different curvatures -- one "
                "at the law's floor, one ten times slower and deep in the overdamped band -- "
                "therefore reach the SAME image wherever their integer step counts land on the "
                "same float32 lifetime, and elsewhere differ by the slow clock's step "
                f"quantisation alone (worst relative lifetime gap {worst_rel:.2e}). mu^2 is a "
                "pure time reparameterisation and cannot trade refusal against composition; the "
                "exact, quantisation-free form of that statement is "
                "gen_retention.assert_reparameterisation_invariance, which holds to 4.2e-14.")}


def commutation_test(out: Path, seed: int, log: Callable = print) -> dict:
    """⭐ V5 App. G.8, tested at byte level on the field store.

    ``A. commutation``     ``evict(decay^n(S))`` vs ``decay^n(evict(S))``, for a decay applied
                           to a DISJOINT item set (so the two operations really do act on
                           disjoint components of the state, which is the paper's hypothesis)
                           and, separately, for a decay applied to the SAME set.
    ``B. never-inserted``  the evicted store vs a store **built from a pool with the deleted
                           value's rows removed**.  ⚠ The paper conditions this on *canonical*
                           placement; the field store places greedily and fits its address book
                           on the pool, so a failure here falsifies the PRECONDITION, not the
                           law, and is reported in those words.
    """
    from ..core.gen_field import (
        _items_holding, assert_read_request_only, build_field, byte_diff, decay_items,
        evict_term, field_byte_ledger,
    )
    from .exp_gen_factor_wells import build_partitions, features_mmap, gather, load_condition
    from .exp_gen_field_store import _host_factors, field_spec

    d = R7.assert_out_root(out) / f"seed_{int(seed)}"
    art = d / "commutation.json"
    if art.exists():
        log(f"  commute(seed {seed}): already on disk")
        with open(art, encoding="utf-8") as fh:
            return json.load(fh)

    base = R7.build_field_base(out, seed, log=log)
    f0, pool = base["field"], base["pool"]
    cond = load_condition()
    grid = cond["grid"]
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    sched = horizon_schedule()
    ids, _, _ = _items_holding(f0, fac, valx)
    other = np.setdiff1d(np.arange(int(f0.A), dtype=np.int64), np.asarray(ids, np.int64))
    n_dec = min(int(ids.size), int(other.size))
    rng = np.random.default_rng(20_260_920 + 101 * int(seed))
    disjoint = np.sort(rng.choice(other, size=n_dec, replace=False))

    res = {"seed": int(seed), "schedule": {k: sched[k] for k in
                                           ("mu2", "rho", "n_half", "n_refuse_steps",
                                            "band", "eps", "gamma")},
           "n_target_items": int(ids.size), "n_disjoint_decayed": int(disjoint.size)}

    # -- A. commutation, on a DISJOINT set (the paper's hypothesis, literally) --------
    def _rate(n):
        return float(np.float32(sched["rho"] ** int(n)))

    A = {}
    for label, dec_ids in (("disjoint_item_set", disjoint), ("same_item_set", np.asarray(ids))):
        per_n = []
        for n in (1, 4, 16, 64, int(sched["n_refuse_steps"]),
                  int(sched["n_refuse_steps"]) + 16):
            a = f0.copy()
            decay_items(a, dec_ids, _rate(n), log=(lambda *_: None))
            evict_term(a, fac, valx, log=(lambda *_: None))
            b = f0.copy()
            evict_term(b, fac, valx, log=(lambda *_: None))
            decay_items(b, dec_ids, _rate(n), log=(lambda *_: None))
            dd = byte_diff(a, b)
            per_n.append({"n_steps": int(n), "bytes_differing": int(dd["n_changed"]),
                          "per_segment": dict(dd["per_segment"]),
                          "commutes": bool(dd["n_changed"] == 0)})
        A[label] = {"per_n": per_n,
                    "commutes_at_every_n": bool(all(r["commutes"] for r in per_n))}
    res["A_commutation"] = A
    res["A_statement"] = (
        "evict o decay^n == decay^n o evict, byte for byte, at every tested n"
        if all(v["commutes_at_every_n"] for v in A.values()) else
        "⛔ the two operations DO NOT commute byte-exactly on this store -- see per_segment")

    # -- B. never-inserted history ---------------------------------------------------
    # the same build, on a pool with every row carrying the deleted value removed
    parts = build_partitions(Path(out) / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = _host_factors(parts["info"], K)
    fi = grid.factor_indices_of(np.asarray(pool, np.int64))
    keep = np.asarray(pool, np.int64)[fi[:, fac] != valx]
    F = features_mmap(int(seed))
    X = gather(F, keep)
    Y = grid.factor_indices_of(keep)
    cards = [int(c) for c in grid.cardinalities]
    t0 = _now()
    f_never = build_field(X, Y, keep, labels, hosts, cards, field_spec(seed),
                          log=(lambda *_: None))
    del X
    build_cost = _now() - t0
    f_evict = f0.copy()
    evict_term(f_evict, fac, valx, log=(lambda *_: None))
    dn = byte_diff(f_evict, f_never)
    led = field_byte_ledger(f_never, int(PINS["store_bytes"]))
    res["B_never_inserted"] = {
        "n_pool_rows_before": int(np.asarray(pool).size),
        "n_pool_rows_after": int(keep.size),
        "n_rows_removed": int(np.asarray(pool).size - keep.size),
        "rebuild_seconds": float(build_cost),
        "rebuilt_store_matches_byte_ledger": bool(led["matches_budget"]),
        "bytes_differing_from_evicted_store": int(dn["n_changed"]),
        "total_image_bytes": int(dn["total_bytes"]),
        "fraction_of_image_differing": float(dn["n_changed"]) / max(dn["total_bytes"], 1),
        "per_segment": dict(dn["per_segment"]),
        "byte_identical": bool(dn["n_changed"] == 0),
    }
    res["B_statement"] = (
        "the post-delete store IS byte-identical to a store built from the never-inserted "
        "history" if dn["n_changed"] == 0 else
        "⛔ the post-delete store is NOT byte-identical to a store built from the "
        "never-inserted history. ⚠ This falsifies the PAPER'S PRECONDITION on this object, "
        "not the paper's law: V5 App. G.8 conditions byte-equality on CANONICAL "
        "(Blelloch-Golovin) placement, in which the layout L(K) is a function of the live key "
        "set and the fixed geometry alone. The field store selects its 64 atoms by GREEDY SPAN "
        "SELECTION against the address manifold and fits `term_mu`/`well_sigma`/`coupling` on "
        "the pool, so both its layout AND its fitted interface depend on which rows were "
        "present. The segments that differ are printed above and they say which half is which.")

    _jdump(res, art)
    log(f"  commute(seed {seed}): A={res['A_statement'][:60]}...; "
        f"B byte-identical={res['B_never_inserted']['byte_identical']} "
        f"({res['B_never_inserted']['bytes_differing_from_evicted_store']:,} B differ)")
    return res


def build_arms(out: Path, seed: int, log: Callable = print) -> dict:
    """⭐ ``exp_gen_delete_row7.build_arms`` UNCHANGED, plus the decay horizons.

    ⛔ Nothing in ``exp_gen_delete_row7`` is edited on disk.  Every w5 arm keeps its own code
    path byte for byte, so instant eviction, the hole, the table, retrain-from-scratch, the
    8-byte gate and the charter section C controls are on the instrument that produced w5's
    numbers and this run re-measures them rather than quoting them.
    """
    built = _R7_BUILD_ARMS(out, seed, log=log)
    from ..core.gen_field import gap_read
    cond = R7.load_condition()
    grid = cond["grid"]
    base = built["base"]
    fv = built["field_variants"]
    sched = horizon_schedule()
    dv = decay_variants(base, seed, sched, log=log)
    bp = byte_properties(base, dv, fv, log=log)
    ident = {r["n_steps"]: r["identical_to_instant_eviction"] for r in bp["rows"]}

    def _F(field):
        return lambda vals: R7._field_features(field, vals, grid)

    def _fr(field):
        return lambda vals: gap_read(field, vals, grid.values)

    for r in sched["rows"]:
        n = int(r["n_steps"])
        fd = dv["fields"][n]
        # ⭐ a DECLARED no-op / a DECLARED identity, asserted by the scoring loop's
        # `identical_to` machinery rather than assumed: n = 0 must be the base, and any
        # horizon whose image is byte-identical to the eviction must read identically too.
        twin = "field_base" if n == 0 else ("field_evict" if ident.get(n) else None)
        built["arms"][arm_name(n, r["t_half_lives"])] = dict(
            gen=_F(fd), refuse=_fr(fd), family="field",
            role=(f"DELETION BY DECAY at t = {r['t_half_lives']:.4f} half-lives "
                  f"(n = {n} map steps, lifetime {r['lifetime']:.6g}"
                  f"{', DEAD' if r['dead'] else ''})"),
            deletion="A", gen_key=f"field_decay_n{n}", identical_to=twin)
    built["decay"] = {"schedule": sched, "byte_properties": bp,
                      "item_ids": dv["item_ids"],
                      "n_by_provenance": dv["n_by_provenance"],
                      "n_by_geometry": dv["n_by_geometry"], "_ops": dv["_ops"]}
    built["_decay_fields"] = dv["fields"]
    return built


def stage_law(out: Path, log: Callable = print) -> dict:
    """The law: ``self_check`` against the papers' printed numbers, plus the simulation table."""
    out = R7.assert_out_root(out)
    chk = LAW.self_check(float(PINS["eps"]), float(PINS["gamma"]))
    tab = LAW.verification_table(float(PINS["eps"]), float(PINS["gamma"]))
    sched = horizon_schedule()
    res = {"self_check": chk, "verification_table": tab, "schedule": sched,
           "paper": LAW.PAPER}
    _jdump(res, out / "law.json")
    log(f"  law: PASS. floor {chk['floor_at_gamma_0p05']:.6f} vs the papers' printed "
        f"{LAW.PAPER['floor_steps_at_gamma_0p05']}; overdamped slope "
        f"{chk['overdamped_slope_log_nhalf_vs_log_omega2']:.5f}; onset exponent "
        f"{chk['onset_exponent']:.5f}; refusal horizon "
        f"{chk['refusal_horizon_half_lives']:.4f} half-lives")
    log(f"  schedule: n_half={sched['n_half']:.6f} steps, rho={sched['rho']:.12f}, "
        f"band={sched['band']}, refusal at step {sched['n_refuse_steps']} "
        f"({sched['n_refuse_half_lives']:.4f} half-lives)")
    return res


def stage_dev(out: Path, seed: int, log: Callable = print) -> dict:
    """⭐ UNSCORED. The mechanism, the byte properties, the invariance, and a TIMING probe.

    ⛔ Reads are on the arm's own FIT POOL, never on a held-out corner: this stage exists to
    show the machinery works and to time it, and nothing on it may inform a bar.
    """
    out = R7.assert_out_root(out)
    d = out / f"seed_{int(seed)}"
    base = R7.build_field_base(out, seed, log=log)
    fv = R7.field_variants(base, seed, log=(lambda *_: None))
    sched = horizon_schedule()
    dv = decay_variants(base, seed, sched, log=log)
    bp = byte_properties(base, dv, fv, log=log)
    inv = invariance_on_the_store(base, seed, log=log)

    # -- refusal on FIT-POOL requests carrying the deleted value, and on ones that do not --
    from ..core.gen_field import gap_read
    cond = R7.load_condition()
    grid = cond["grid"]
    pool = np.asarray(base["pool"], np.int64)
    fi = grid.factor_indices_of(pool)
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    hit = pool[fi[:, fac] == valx][:512]
    mis = pool[fi[:, fac] != valx][:512]
    vh, vm = grid.values_of(hit), grid.values_of(mis)
    curve = []
    for n in sorted(dv["fields"]):
        fd = dv["fields"][n]
        rh, rm = gap_read(fd, vh, grid.values), gap_read(fd, vm, grid.values)
        curve.append({"n_steps": int(n),
                      "t_half_lives": float(n) / sched["n_half"],
                      "refuse_deleted_value": float(np.asarray(rh["refuse"]).mean()),
                      "empty_basin_deleted_value": float(np.asarray(rh["empty_basin"]).mean()),
                      "refuse_other_values": float(np.asarray(rm["refuse"]).mean())})

    # -- the timing probe: one judge pass on one corner, so the score stage can be sized --
    t0 = _now()
    J = R7._judge(int(seed), log)
    cs = R7.scored_corners(cond)
    c = cs["survivor"][0]
    vals = grid.values_of(np.asarray(c["rows"], np.int64))
    feats, _ = R7._field_features(base["field"], vals, grid)
    t_feat = _now() - t0
    t1 = _now()
    J.read_back(feats, int(R7.PINS["eval_batch"]))
    t_judge = _now() - t1
    n_units = (len(cs["all"]) + 8)
    n_gen = 10 + len(sched["rows"])
    res = {"schedule": sched, "byte_properties": bp, "invariance": inv,
           "fit_pool_refusal_curve": curve,
           "timing": {"seconds_per_feature_render": t_feat,
                      "seconds_per_judge_pass": t_judge,
                      "n_units_per_generator": int(n_units),
                      "n_generators_estimate": int(n_gen),
                      "estimated_score_seconds": float((t_feat + t_judge) * n_units * n_gen),
                      "n_requests_in_probe_corner": int(vals.shape[0])}}
    _jdump(res, d / "dev.json")
    log(f"  dev(seed {seed}): feature {t_feat:.2f}s + judge {t_judge:.2f}s per unit; "
        f"~{n_units} units x ~{n_gen} generators => "
        f"~{res['timing']['estimated_score_seconds'] / 60:.1f} min/seed")
    log(f"  dev(seed {seed}): first horizon byte-identical to eviction: "
        f"{bp['first_step_byte_identical_to_eviction']}")
    return res


def stage_score(out: Path, seed: int, max_seconds: float = 100_000.0,
                log: Callable = print) -> dict:
    """⭐⭐ THE SCORED STAGE.  ``exp_gen_delete_row7.score_seed`` with the decay arms added.

    ⛔ The scoring loop is w5's, unedited: this function only swaps the arm builder, so every
    number here is on the instrument that produced w5's, and the loop's own ``identical_to``
    assertions apply to the decay arms too.
    """
    out = R7.assert_out_root(out)
    if R7.build_arms is not _R7_BUILD_ARMS:
        raise AssertionError(
            "exp_gen_delete_row7.build_arms is already patched -- refusing to stack wrappers")
    R7.build_arms = lambda o, s, log=print: build_arms(o, s, log=log)
    try:
        return R7.score_seed(out, seed, max_seconds=max_seconds, log=log)
    finally:
        R7.build_arms = _R7_BUILD_ARMS


def _load_scores(out: Path, seed: int):
    z = np.load(Path(out) / f"seed_{int(seed)}" / "scores.npz", allow_pickle=True)
    return {k: z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]
            for k in z.files}


def build_report(out: Path, log: Callable = print) -> dict:
    """The refusal curve and the composition curve, on one axis, against every comparator."""
    out = R7.assert_out_root(out)
    cond = R7.load_condition()
    grid = cond["grid"]
    cs = R7.scored_corners(cond)
    surv = [c["corner_id"] for c in cs["survivor"]]
    dele = [c["corner_id"] for c in cs["deleted"]]
    sched = horizon_schedule()
    seen = R7.seen_tuple_mask(grid, cond["train_rows"])

    rep = {"pins": PINS, "schedule": sched, "per_seed": {},
           "horizons_declared_from_the_law": list(HORIZONS)}
    for seed in PINS["seeds"]:
        p = Path(out) / f"seed_{int(seed)}" / "scores.npz"
        if not p.exists():
            rep["per_seed"][str(seed)] = {"NOT_RUN": f"no scores.npz for seed {seed}"}
            continue
        z = _load_scores(out, seed)
        done = z["done"]
        arms = sorted({a for (a, _) in done})
        rows = {a: {} for a in arms}
        for (a, cid), rec in done.items():
            if cid.startswith("probe/"):
                continue
            rows[a][cid] = R7._per_corner(rec, grid, seen)
        probes = {a: {cid.split("/", 1)[1]: rec for (aa, cid) in done
                      if aa == a and cid.startswith("probe/")
                      for rec in [done[(aa, cid)]]} for a in arms}

        def _ref(a, ids):
            return float(np.mean([rows[a][i]["refuse_rate"] for i in ids])) if ids else float("nan")

        S = {"arms": arms}
        # ---- THE TWO CURVES, on the law's own axis ---------------------------
        curve = []
        for r in sched["rows"]:
            n = int(r["n_steps"])
            a = arm_name(n, r["t_half_lives"])
            if a not in rows:
                continue
            comp = R7._paired(rows[a], rows["field/base"], surv,
                              "distinct_valid_novel_tuples")
            curve.append({
                "n_steps": n, "t_half_lives": r["t_half_lives"],
                "lifetime": r["lifetime"], "dead": r["dead"],
                "refuse_on_deleted_set": _ref(a, dele),
                "refuse_on_survivor_set": _ref(a, surv),
                "composition_mean_delta_vs_base": comp.get("mean_delta"),
                "composition_ci95": comp.get("ci95"),
                "composition_n_corners": comp.get("n_corners"),
                "novel_per_corner": float(np.mean(
                    [rows[a][i]["distinct_valid_novel_tuples"] for i in surv])),
            })
        S["curve"] = curve
        # ---- the comparators, ON THIS RUN's instrument -----------------------
        def _pair(a, b):
            return R7._paired(rows[a], rows[b], surv, "distinct_valid_novel_tuples")

        S["comparators"] = {
            "field/evict_red (INSTANT EVICTION)": {
                **_pair("field/evict_red", "field/base"),
                "refuse_on_deleted_set": _ref("field/evict_red", dele),
                "w5_measured_mean_delta": W5_EVICT_COMPOSITION_DELTA.get(int(seed))},
            "field/hole (THE COUPLING HOLE, 1 byte)": {
                **_pair("field/hole", "field/base"),
                "refuse_on_deleted_set": _ref("field/hole", dele),
                "w5_measured_mean_delta": W5_HOLE_COMPOSITION_DELTA},
            "field/decay_red (w5's decay-to-zero)": {
                **_pair("field/decay_red", "field/base"),
                "refuse_on_deleted_set": _ref("field/decay_red", dele)},
            "table/row_removed": {**_pair("table/row_removed", "table/base"),
                                  "refuse_on_deleted_set": _ref("table/row_removed", dele)},
            "table/retrain (GOLD STANDARD)": {
                **_pair("table/retrain", "table/base"),
                "refuse_on_deleted_set": _ref("table/retrain", dele)},
            "table+gate8/clear_red (THE 8-BYTE GATE)": {
                **_pair("table+gate8/clear_red", "table+gate8/base"),
                "refuse_on_deleted_set": _ref("table+gate8/clear_red", dele)},
            "field/landscape_removed (C.2)": _pair("field/landscape_removed", "field/base"),
            "field/landscape_shuffled (C.2)": _pair("field/landscape_shuffled", "field/base"),
            "field/lookup_argmax (C.1 FLOOR)": _pair("field/lookup_argmax", "field/base"),
            "field/lookup_mean (C.1 FLOOR)": _pair("field/lookup_mean", "field/base"),
        }
        S["dashes"] = {
            "mlp / mlp_180k / cont_mlp / transformer / lstm / lace / poe_ebm":
                "DASH + footnote: no refusal verb -- every learner is a generator that always "
                "emits, so on a deletion row it IS `accept_everything`. w5 established this "
                "mechanically (refuses 0.0000 of the deleted requests). ⛔ A dash is never a "
                "win for us. Deletion is available to them only by retrain-from-scratch, whose "
                "cost is on the cost line."}
        S["generator_equalities"] = z.get("generator_equalities")
        S["table_cost"] = z.get("table_cost")
        S["proof"] = z.get("proof")
        # ---- the convergence question, answered on this seed ------------------
        ev = S["comparators"]["field/evict_red (INSTANT EVICTION)"]["mean_delta"]
        conv = None
        for c in curve:
            if c["composition_mean_delta_vs_base"] is None:
                continue
            if abs(c["composition_mean_delta_vs_base"] - ev) <= 0.1 * max(abs(ev), 1e-9):
                conv = c["t_half_lives"]
                break
        first_ref = next((c["t_half_lives"] for c in curve
                          if c["refuse_on_deleted_set"] >= 0.99), None)
        S["THE_ANSWER"] = {
            "instant_eviction_composition_delta": ev,
            "hole_composition_delta": S["comparators"][
                "field/hole (THE COUPLING HOLE, 1 byte)"]["mean_delta"],
            "first_horizon_within_10pct_of_eviction_cost_half_lives": conv,
            "first_horizon_refusing_at_0.99_half_lives": first_ref,
            "derived_refusal_horizon_half_lives": sched["n_refuse_half_lives"],
            "cost_paid_before_any_refusal_arrives": (
                None if (conv is None or first_ref is None) else bool(conv < first_ref)),
        }
        rep["per_seed"][str(seed)] = S
    _jdump(rep, out / "report.json")
    return rep


def _f(x, nd=4):
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "nan"
    return f"{v:.{nd}f}"


def render_markdown(out: Path, log: Callable = print) -> str:
    """The two curves, the comparators and the bars, as one readable table per seed."""
    out = R7.assert_out_root(out)
    with open(out / "report.json", encoding="utf-8") as fh:
        rep = json.load(fh)
    sched = rep["schedule"]
    L = []
    L.append("# w6 `hub/retention` -- deletion-by-decay on the field store\n")
    L.append("**Scored run 2/3 of the `deletion` target** (1 spent in w5; 3/3 remains "
             "banked). `PREREG-w6-retention-1.md`. Cache generation 2, seeds 0/1/2, "
             "109 held-out corners (20 deleted / **89 survivor**), metric "
             "`distinct_valid_novel_tuples` per corner, paired by corner, margin **1.0**.\n")
    L.append("Labels enter **once at fit time** to align wells and build the address book, "
             "and at the interface. The decay operator itself is **label-free**: it "
             "multiplies `atom_life`, which no label touches.\n")
    L.append("## The clock, declared from the law\n")
    L.append(f"`eps = {sched['eps']}`, `gamma = {sched['gamma']}` (both the papers' own "
             f"declared values) -> `mu^2 = {sched['mu2']:.6g}`, "
             f"`h/h* = {sched['h_over_h_star']:.4g}`, band **{sched['band']}**, "
             f"`rho = {sched['rho']:.12f}`, **`n_half = {sched['n_half']:.6f}` steps** = the "
             f"paper's curvature-independent floor. Derived refusal step "
             f"**{sched['n_refuse_steps']}** = **{sched['n_refuse_half_lives']:.4f} "
             f"half-lives**; the law's `ln(1e-6)/ln(1/2) = "
             f"{sched['derived_refusal_horizon_half_lives']:.4f}`.\n")
    for seed, S in sorted(rep["per_seed"].items()):
        L.append(f"\n## Seed {seed}\n")
        if "NOT_RUN" in S:
            L.append(f"NOT RUN -- {S['NOT_RUN']}. A declared NOT-RUN is never a null.\n")
            continue
        L.append("### THE TWO CURVES, on the law's own axis\n")
        L.append("| step `n` | `t` (half-lives) | lifetime | **refuse (deleted set)** | "
                 "refuse (survivor set) | **composition delta/corner vs base** | CI95 | "
                 "novel/corner |")
        L.append("|---:|---:|---:|---:|---:|---:|:---:|---:|")
        for c in S["curve"]:
            ci = c.get("composition_ci95") or [None, None]
            L.append(f"| {c['n_steps']} | {c['t_half_lives']:.4f} | "
                     f"{c['lifetime']:.4g} | **{_f(c['refuse_on_deleted_set'])}** | "
                     f"{_f(c['refuse_on_survivor_set'])} | "
                     f"**{_f(c['composition_mean_delta_vs_base'])}** | "
                     f"[{_f(ci[0], 3)}, {_f(ci[1], 3)}] | "
                     f"{_f(c['novel_per_corner'], 2)} |")
        L.append("\n### The comparators, RE-MEASURED on this run (not quoted)\n")
        L.append("| comparator | composition delta/corner | CI95 | refuse (deleted set) "
                 "| w5's number |")
        L.append("|---|---:|:---:|---:|---:|")
        for k, v in S["comparators"].items():
            ci = v.get("ci95") or [None, None]
            L.append(f"| `{k}` | {_f(v.get('mean_delta'))} | "
                     f"[{_f(ci[0], 3)}, {_f(ci[1], 3)}] | "
                     f"{_f(v.get('refuse_on_deleted_set'))} | "
                     f"{_f(v.get('w5_measured_mean_delta'))} |")
        a = S["THE_ANSWER"]
        L.append("\n### The answer on this seed\n")
        L.append("- instant eviction's composition delta: "
                 f"**{_f(a['instant_eviction_composition_delta'])}**")
        L.append("- the coupling hole's composition delta: "
                 f"**{_f(a['hole_composition_delta'])}**")
        L.append("- `t_pay` (first horizon within 10 % of eviction's cost): "
                 f"**{_f(a['first_horizon_within_10pct_of_eviction_cost_half_lives'])}** "
                 "half-lives")
        L.append("- `t_refuse` (first horizon refusing at >= 0.99): "
                 f"**{_f(a['first_horizon_refusing_at_0.99_half_lives'])}** half-lives "
                 f"(derived: {_f(a['derived_refusal_horizon_half_lives'])})")
        L.append("- **cost paid before any refusal arrives: "
                 f"{a['cost_paid_before_any_refusal_arrives']}**")
        L.append("\n" + json.dumps(S["dashes"], indent=1))
    md = "\n".join(L) + "\n"
    with open(out / "REPORT.md", "w", encoding="utf-8") as fh:
        fh.write(md)
    log(f"  report -> {out / 'REPORT.md'}")
    return md


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True,
                   choices=("law", "commute", "dev", "score", "report"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seconds", type=float, default=100_000.0)
    return p


def main(argv=None) -> int:
    a = build_arg_parser().parse_args(argv)
    out = Path(a.out)
    if a.stage == "law":
        stage_law(out)
    elif a.stage == "commute":
        commutation_test(out, a.seed)
    elif a.stage == "dev":
        stage_dev(out, a.seed)
    elif a.stage == "score":
        r = stage_score(out, a.seed, max_seconds=a.max_seconds)
        if r.get("incomplete"):
            print("INCOMPLETE -- rerun to continue")
            return 2
    elif a.stage == "report":
        build_report(out)
        print(render_markdown(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
