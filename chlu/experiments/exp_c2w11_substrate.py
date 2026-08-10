"""C2W11 spoke A — **the repaired substrate, and every kill-condition FIRST**.

Built to ``.claude/outputs/c2w11/PREREG-C2W11.md`` §4 (K0–K8), §6 (M1, M2, M4,
M5, M6) and §7 (the coverage half of the C2W9 trigger), and to this spoke's own
``PREREG.md``, which was filed before this file existed.

⛔ **Every leg here is MECHANICS.** No VALUE leg, no ``OD``/``OD_min``, no
organizer swap, no paper number, no tier-ii verdict. This module does not score
the physics arm's value; it decides whether it is worth scoring.

⭐ **Standing doctrine (C2W3):** *build the kill-condition before the thing it
can kill.* The registered run order is binding and is the default here::

    K0  ->  M6  ->  width selection  ->  K7-CAP / K6  ->  K1
        ->  K2  ->  K3  ->  K4  ->  K5  ->  K8

K0 needs **no store** and costs seconds; M6 needs only a written store and the
launch geometry. They are the cheapest kill signal in the wave and they run and
report first.

⭐ **And a null outcome here is a RESULT, not a wasted wave.** If the structural
caps are unmoved despite three *measured* substrate changes — the placing write,
re-selected co-scaled widths and feature-factored launches — that is the fifth
convergent datum on write-side organization, this time with the substrate
repairs **controlled for**.

⛔ **Wells are never named semantically** (``PREREG-TierII.md`` §2.6):

    *"Wells {j} are co-activated by queries whose ground-truth factor set
    contains factor f, with co-activation correlation rho = … (95 % CI …),
    measured against a permutation null. No well is identified with any factor;
    the claim is a correlation between co-activation/wormhole/shell-position
    statistics and task structure."*
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.factored_store import (
    CatFamily,
    CatTestConfig,
    FactoredStore,
    apply_reader,
    assert_selected_width,
    build_family,
    build_phi,
    byte_ratio,
    chance_accuracy,
    effective_s,
    exact_set_accuracy,
    fit_readers,
    min_separation,
    multi_particle_read,
    occupancy,
    place_wells,
    reader_bytes,
    resolve_atom_width,
    store_population_spacing,
    write_store,
)
from chlu.core.feature_launch import (
    LaunchProtocol,
    build_launch_head,
    coverage_stats,
    k0_stats,
    k6_already_right,
    launch_points,
    occupancy_precision_from_points,
    wells_visited,
)
from chlu.core.soft_certificate import capture_radius

__all__ = [
    "c2w11_config",
    "stage_k0",
    "stage_width_selection",
    "stage_k1",
    "stage_k2",
    "stage_k3_k4_k5",
    "stage_k6_k7cap",
    "stage_k8",
    "stage_m4",
    "stage_m6",
    "stage_coverage",
    "freeze_interfaces",
    "run_c2w11_substrate",
    "SELECTION_SEEDS",
    "WIDTH_GRID",
    "V3_BUDGET_GRID",
]

# ⛔ Selection seeds are DISJOINT from the claim seeds (registered in PREREG §1).
SELECTION_SEEDS: Tuple[int, ...] = (100, 101, 102)
WIDTH_GRID: Tuple[float, ...] = (0.20, 0.25, 0.30, 0.37, 0.50, 0.75, 1.00, 1.50)
PROVISIONAL_SEP = 0.859  # the banked 2.7 x 0.318 (orgdiv-cat-test §4)
DS_BAND = (2.5, 2.9)

# ⭐⭐ THE FROZEN V3 READ-COMPUTE BUDGET GRID. Spokes B and C BOTH score V3 on
# this exact axis and they run concurrently, so only spoke A can freeze it. A
# mismatched axis VOIDS value leg iii. Ledger = particles-evolved x Verlet steps.
V3_BUDGET_GRID: Tuple[int, ...] = (50, 100, 200, 400, 800, 1200)


# ==========================================================================
# helpers
# ==========================================================================
def _j(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_j))
    print(f"  -> {path}", flush=True)


def _mean_2se(xs) -> Dict[str, float]:
    a = np.asarray([float(x) for x in xs], dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(a.mean()), "sd": sd, "n": int(n),
            "two_se": float(2.0 * sd / math.sqrt(max(n, 1))),
            "values": a.tolist()}


def c2w11_config(**kw) -> CatTestConfig:
    """The C2W11 substrate configuration: **all three repairs ON**.

    ⛔ ``atom_width_frac_spacing`` is deliberately left ``None`` here. It is set
    only by :func:`stage_width_selection`'s declared result — the banked
    ``1.5`` is **not inherited** (C2W8-close repair (b)).
    """
    base = dict(
        write_mode="placing",
        launch_mode="feature_factored",
        atoms_per_well=12,
        payload_dim=8,          # C2W5 deviation D1, re-verified by K2 here
        payload_radius=1.0,     # D3
        atom_payload_init_radius=1.0,  # D4, a designed mechanism at 0 parameters
    )
    base.update(kw)
    cfg = CatTestConfig(**base)
    if cfg.launch_mode == "feature_factored":
        k = int(cfg.n_channels) if cfg.n_channels is not None else int(cfg.f_subset)
        cfg = replace(cfg, n_particles=k)
    return cfg


def _anchors(cfg: CatTestConfig, phi, sep: float) -> np.ndarray:
    return place_wells(phi, cfg, sep=float(sep))


def _targets(cfg: CatTestConfig, anchors: np.ndarray, payloads: np.ndarray):
    d, m = int(cfg.addr_dim), int(cfg.payload_dim)
    t = np.zeros((cfg.n_wells, cfg.dim), dtype=np.float32)
    t[:, :d] = anchors[:, :d]
    t[:, d:d + m] = payloads
    return t


def _depth_scale(cfg: CatTestConfig) -> np.ndarray:
    r = float(cfg.depth_ratio)
    return np.where(np.arange(cfg.n_wells) % 2 == 0, 1.0, r)


def _relax_fn(store: FactoredStore, cfg: CatTestConfig):
    from chlu.core.factored_store import _settle
    model = store.model(cfg)

    @eqx.filter_jit
    def go(pts):
        p0 = jnp.zeros_like(pts)
        q, _ = jax.vmap(lambda a, b: _settle(model, a, b, cfg.address_steps,
                                             cfg.dt, cfg.gamma_address))(pts, p0)
        q, _ = jax.vmap(lambda a, b: _settle(model, a, b, cfg.read_steps,
                                             cfg.dt, cfg.gamma_read))(
            q, jnp.zeros_like(q))
        return q

    return lambda pts: np.asarray(go(jnp.asarray(pts, dtype=jnp.float32)))


def _lambda_min(store: FactoredStore, pts: np.ndarray) -> np.ndarray:
    V = store.V

    @eqx.filter_jit
    def go(z):
        H = jax.vmap(jax.hessian(lambda q: jnp.reshape(V(q), ())))(z)
        return jnp.linalg.eigvalsh(0.5 * (H + jnp.swapaxes(H, -1, -2)))[..., 0]

    return np.asarray(go(jnp.asarray(pts, dtype=jnp.float32)))


def build_arm(cfg: CatTestConfig, family: CatFamily, seed: int, *,
              sep: Optional[float] = None, phi=None,
              check_width: bool = True) -> Dict[str, Any]:
    """Place -> resolve the width -> write. ⛔ No organizer: that is spoke B's."""
    phi = phi if phi is not None else build_phi(cfg)
    ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
    sep = float(sep) if sep is not None else float(cfg.target_ds * ruler)
    anchors = _anchors(cfg, phi, sep)
    guard = assert_selected_width(cfg) if check_width else {"width_guard": False}
    winfo = resolve_atom_width(cfg, anchors)
    key = jax.random.PRNGKey(int(seed))
    k_init, k_write = jax.random.split(key, 2)
    store = FactoredStore(cfg, anchors, k_init, atom_width=winfo["atom_width"])
    order = np.random.default_rng(int(seed)).permutation(cfg.n_wells)
    store, wrep = write_store(store, cfg, anchors, family.payloads, k_write,
                              order=order, depth_scale=_depth_scale(cfg),
                              atom_width=winfo["atom_width"])
    return {"store": store, "phi": phi, "anchors": anchors, "sep": sep,
            "write": wrep, "width": winfo, "width_guard": guard,
            "spacing": store_population_spacing(anchors), "order": order.tolist()}


def _head(cfg: CatTestConfig, phi, *, collapse: bool = False):
    """The launch head a cell runs at: feature-factored, or C2W5's offsets."""
    if cfg.launch_mode == "feature_factored":
        return build_launch_head(phi, cfg, collapse_to_one_channel=collapse)
    return phi


# ==========================================================================
# ⭐⭐ K0 — THE CHEAPEST KILL SIGNAL IN THE WAVE. No store. Runs first.
# ==========================================================================
def stage_k0(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4),
             out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐⭐ **K0** — launch expressivity, from launch geometry with **NO store**.

    Fraction of unseen queries for which the feature-factored launch set can
    reach ``>= F`` **distinct** feature wells, and the mean number reachable.
    ⛔ Bar: ``>= 0.80`` distinct-``F`` fraction, mean distinct ``>= F - 0.5``.
    If it fails, **the wave stops at spoke A** and it is reported as a
    **structural cap**, not a physics null.

    Three arms run side by side and only the first is the leg:

    * ``feature_factored`` — the C2W11 launch head (the leg);
    * ``designed_offsets`` — C2W5's ``P = 4`` offsets from one set-code, a
      **reproduction** of the banked 0.050 / 2.202 (if I cannot reproduce it my
      instrument is wrong, not the banked number);
    * ``collapsed`` — ⛔ **M1's designed negative**: every channel collapsed
      onto channel 1 must score ~chance.

    ⭐ K6 and M5's launch side ride along, because they are the same object and
    computing them separately is how a slip like the fifth-session one happens.
    """
    ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
    sep = float(cfg.target_ds * ruler)
    cells = []
    for seed in seeds:
        t0 = time.time()
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        anchors = _anchors(cfg, phi, sep)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        key = jax.random.PRNGKey(5000 + int(seed))
        row: Dict[str, Any] = {"seed": int(seed), "sep": sep,
                               "spacing": store_population_spacing(anchors)}
        row["sigma_q_over_store_spacing"] = float(
            cfg.query_sigma / max(row["spacing"]["median_nn"], 1e-12))
        for arm, head in (("feature_factored",
                           build_launch_head(phi, cfg)),
                          ("designed_offsets", phi),
                          ("collapsed_designed_negative",
                           build_launch_head(phi, cfg,
                                             collapse_to_one_channel=True))):
            pts = launch_points(head, ind_u, cfg, key)
            st = k0_stats(pts, anchors, fam.unseen, cfg.f_subset)
            st.update(k6_already_right(pts, anchors, fam.unseen))
            st.update(wells_visited(pts, anchors, cfg.n_wells))
            st["K0_PASS"] = bool(st["frac_ge_F_distinct"] >= 0.80
                                 and st["mean_distinct_wells"]
                                 >= cfg.f_subset - 0.5)
            row[arm] = st
        row["wall_s"] = round(time.time() - t0, 1)
        cells.append(row)
        ff = row["feature_factored"]
        do = row["designed_offsets"]
        print(f"[K0] seed={seed} feature-factored >=F {ff['frac_ge_F_distinct']:.4f} "
              f"mean {ff['mean_distinct_wells']:.3f} "
              f"CORRECT-distinct {ff['mean_correct_distinct_wells']:.3f} "
              f"K6 {ff['k6_as_fraction']} | "
              f"C2W5 offsets >=F {do['frac_ge_F_distinct']:.4f} "
              f"mean {do['mean_distinct_wells']:.3f} "
              f"CORRECT-distinct {do['mean_correct_distinct_wells']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {"stage": "k0", "bar_frac_ge_F": 0.80,
           "bar_mean_distinct": float(cfg.f_subset - 0.5),
           "cells": cells,
           "summary": {
               arm: {
                   "frac_ge_F_distinct": _mean_2se(
                       [c[arm]["frac_ge_F_distinct"] for c in cells]),
                   "mean_distinct_wells": _mean_2se(
                       [c[arm]["mean_distinct_wells"] for c in cells]),
                   "occupancy_precision": _mean_2se(
                       [c[arm]["occupancy_precision"] for c in cells]),
                   "mean_correct_distinct_wells": _mean_2se(
                       [c[arm]["mean_correct_distinct_wells"] for c in cells]),
                   "frac_all_F_correct_distinct": _mean_2se(
                       [c[arm]["frac_all_F_correct_distinct"] for c in cells]),
                   "exact_set_occupancy": _mean_2se(
                       [c[arm]["exact_set_occupancy"] for c in cells]),
                   "k6_frac_already_right": _mean_2se(
                       [c[arm]["k6_frac_already_right"] for c in cells]),
                   "wells_visited_frac": _mean_2se(
                       [c[arm]["wells_visited_frac"] for c in cells]),
                   "PASS_all_seeds": bool(all(c[arm]["K0_PASS"] for c in cells)),
               }
               for arm in ("feature_factored", "designed_offsets",
                           "collapsed_designed_negative")},
           "sigma_q_over_store_spacing": _mean_2se(
               [c["sigma_q_over_store_spacing"] for c in cells]),
           }
    # ⛔ DECLARED OUT-OF-PROTOCOL DIAGNOSTIC (the `orgdiv-null-arms` §3.1
    # P-sweep precedent): changing `d` re-draws phi, so these are NOT matched
    # arms and are NEVER a score. Costs seconds and needs no store; it is here
    # because spoke B must not have to discover the d-dependence mid-run.
    d_diag = []
    for d in (4, 8, 16, 32):
        c = replace(cfg, addr_dim=int(d))
        fam = build_family(c, seed=int(seeds[0]))
        ph = build_phi(c)
        an = _anchors(c, ph, float(c.target_ds * (c.s_measured or c.atom_width)))
        hd = build_launch_head(ph, c)
        ptsd = launch_points(hd, fam.indicator(fam.unseen, c.n_wells), c,
                             jax.random.PRNGKey(5000))
        sd = k0_stats(ptsd, an, fam.unseen, c.f_subset)
        d_diag.append({"addr_dim": int(d),
                       "frac_ge_F_distinct": sd["frac_ge_F_distinct"],
                       "mean_distinct_wells": sd["mean_distinct_wells"],
                       "mean_correct_distinct_wells":
                           sd["mean_correct_distinct_wells"],
                       "occupancy_precision": sd["occupancy_precision"],
                       "exact_set_occupancy": sd["exact_set_occupancy"],
                       "k6": k6_already_right(ptsd, an, fam.unseen)[
                           "k6_frac_already_right"]})
        print(f"[K0/d-diag] d={d} >=F {sd['frac_ge_F_distinct']:.3f} "
              f"correct-distinct {sd['mean_correct_distinct_wells']:.3f} "
              f"prec {sd['occupancy_precision']:.3f} "
              f"exact {sd['exact_set_occupancy']:.4f}", flush=True)
    res["DECLARED_out_of_protocol_d_diagnostic"] = {
        "label": ("⛔ DECLARED OUT-OF-PROTOCOL: changing d re-draws phi, so these "
                  "are NOT matched arms and are NEVER a score"),
        "seed": int(seeds[0]), "cells": d_diag}
    res["K0_PASS"] = bool(res["summary"]["feature_factored"]["PASS_all_seeds"])
    res["M1_designed_negative_fires"] = bool(
        not res["summary"]["collapsed_designed_negative"]["PASS_all_seeds"])
    if out:
        _dump(res, out / "stage_k0.json")
    return res


# ==========================================================================
# ⭐ THE WIDTH SELECTION SWEEP (repair (b)) — on the STORE population
# ==========================================================================
def stage_width_selection(cfg: CatTestConfig,
                          seeds: Sequence[int] = SELECTION_SEEDS,
                          grid: Sequence[float] = WIDTH_GRID,
                          out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐⭐ RE-SELECT the co-scaled atom width against the **STORE population**.

    ⛔ ``atom_width_frac_spacing = 1.5`` is **NOT inherited**: C2W8-close repair
    (b) moved the spacing that value co-scales against, so it is no longer a
    *selected* value. The protocol below was registered in this spoke's
    ``PREREG.md`` §1 **before** the sweep ran, and it has no free choices:

    1. place at the provisional ``sep_0 = 0.859`` (banked ``2.7 x 0.318``);
    2. for each ``w_frac``: place the store at width
       ``w_frac x store_population_spacing``, run the **placing write**,
       **MEASURE** ``s`` with the ``alpha|q|^2``-subtracted radial estimator,
       record ``d/s = sep_achieved / s_measured``, the fit ``R^2`` and K1's legs;
    3. among the ``w_frac`` with measured ``d/s`` inside ``[2.5, 2.9]`` **and**
       all three K1 legs passing, take the one whose ``d/s`` is closest to 2.7;
       if none qualifies, take the closest on ``d/s`` alone and **declare K1
       FAILED at the selected width** — a failure is reported, never selected
       around;
    4. one refinement pass at ``sep = 2.7 x s_measured(selected)``;
    5. ⛔ selection seeds are **disjoint** from the claim seeds.

    ⚠ ``width_guard`` is off inside this sweep, and only inside it: this is the
    stage that *does* the selecting.
    """
    cells = []
    for w in grid:
        for seed in seeds:
            t0 = time.time()
            c = replace(cfg, atom_width_frac_spacing=float(w),
                        atom_width_selected_frac=None, width_guard=False)
            fam = build_family(c, seed=seed)
            arm = build_arm(c, fam, seed, sep=PROVISIONAL_SEP, check_width=False)
            tg = _targets(c, arm["anchors"], fam.payloads)
            s_hint = arm["width"]["atom_width"]
            fits = [effective_s(arm["store"].V, tg[j], s_hint=s_hint, seed=seed,
                                confine=c.confine) for j in range(c.n_wells)]
            s_meas = float(np.nanmedian([f["s"] for f in fits]))
            r2 = float(np.nanmedian([f["r2"] for f in fits]))
            depth = float(np.nanmedian([f["depth"] for f in fits]))
            sep_ach = min_separation(arm["anchors"])
            ds = float(sep_ach / s_meas) if np.isfinite(s_meas) and s_meas > 0 \
                else float("nan")
            k1 = _k1_legs(arm, c, tg, seed)
            cells.append({
                "w_frac": float(w), "seed": int(seed),
                "atom_width": s_hint,
                "store_population_spacing": arm["spacing"],
                "s_measured": s_meas, "s_fit_r2": r2, "well_depth": depth,
                "sep_achieved": sep_ach, "ds_measured": ds,
                "K1": k1, "wall_s": round(time.time() - t0, 1)})
            print(f"[width] w={w:.2f} seed={seed} s_atom={s_hint:.4f} "
                  f"s_meas={s_meas:.4f} d/s={ds:.2f} R2={r2:.4f} "
                  f"K1={'PASS' if k1['K1_PASS'] else 'FAIL'} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- the mechanical selection rule ----------------------------------
    per_w = {}
    for w in grid:
        rows = [c for c in cells if c["w_frac"] == float(w)]
        ds = float(np.nanmedian([r["ds_measured"] for r in rows]))
        per_w[float(w)] = {
            "ds_measured_median": ds,
            "s_measured_median": float(np.nanmedian([r["s_measured"] for r in rows])),
            "s_fit_r2_median": float(np.nanmedian([r["s_fit_r2"] for r in rows])),
            "K1_PASS_all": bool(all(r["K1"]["K1_PASS"] for r in rows)),
            "in_ds_band": bool(DS_BAND[0] <= ds <= DS_BAND[1]),
            "n_seeds": len(rows)}
    qualified = [w for w, v in per_w.items() if v["in_ds_band"] and v["K1_PASS_all"]]
    pool = qualified or [w for w in per_w if np.isfinite(per_w[w]["ds_measured_median"])]
    selected = min(pool, key=lambda w: abs(per_w[w]["ds_measured_median"] - 2.7))
    res = {"stage": "width_selection", "grid": list(map(float, grid)),
           "selection_seeds": list(map(int, seeds)),
           "provisional_sep": PROVISIONAL_SEP, "ds_band": list(DS_BAND),
           "cells": cells, "per_w_frac": per_w,
           "qualified_w_fracs": sorted(qualified),
           "selected_w_frac": float(selected),
           "selection_was_qualified": bool(selected in qualified),
           "selection_protocol": (
               "registered in .claude/outputs/c2w11-substrate-and-kills/PREREG.md "
               "§1 BEFORE the sweep: among w_frac with MEASURED d/s in [2.5,2.9] "
               "AND all three K1 legs passing, take the one whose d/s is closest "
               "to 2.7; selection seeds 100/101/102, disjoint from the claim "
               "seeds; the harness then REFUSES any other width."),
           "banked_1p5_is_NOT_inherited": {
               "w_frac": 1.5,
               "ds_measured_median": per_w.get(1.5, {}).get("ds_measured_median"),
               "K1_PASS_all": per_w.get(1.5, {}).get("K1_PASS_all"),
               "why": ("C2W8-close repair (b): the spacing a co-scaled width "
                       "co-scales against is now the STORE population's, so the "
                       "banked 1.5 is no longer a selected value.")},
           }
    if out:
        _dump(res, out / "stage_width_selection.json")
    return res


# ==========================================================================
# K1 — write admissibility under the PLACING write at the selected width
# ==========================================================================
def _k1_legs(arm: Dict[str, Any], cfg: CatTestConfig, targets: np.ndarray,
             seed: int, n_cap: Optional[int] = None) -> Dict[str, Any]:
    """K1's three bars: endpoint loss · ``lambda_min > 0`` · SC-6 capture."""
    relax = _relax_fn(arm["store"], cfg)
    q_star = relax(targets)
    lam = _lambda_min(arm["store"], q_star)
    n_cap = int(n_cap if n_cap is not None else min(16, cfg.n_wells))
    caps = [capture_radius(relax, targets[j], n_dirs=8, r_hi=1.0, steps=8,
                           tol=0.15, seed=seed)["capture_radius"]
            for j in range(n_cap)]
    loss = float(arm["write"]["endpoint_write_loss"])
    frac_lam = float((lam > 0).mean())
    frac_cap = float(np.mean(np.asarray(caps) >= cfg.query_sigma))
    # site drift: how far the settled attractor sits from the written site.
    # ⛔ TWO-SIDED by C2W8-close repair (a): drift -> 0 is D2a (table-expressible)
    # and is NOT a perfect score. The floor is a fraction of a MEASURED spacing.
    drift = np.linalg.norm(q_star - np.asarray(targets), axis=-1)
    spacing = arm["spacing"]["median_nn"]
    drift_med = float(np.median(drift))
    drift_ratio = float(drift_med / max(spacing, 1e-12))
    # ⭐ THE REACH DIAGNOSTIC the read protocol actually lives or dies on:
    # the read launches with the PAYLOAD BLOCK PINNED TO 0 (the anti-decoration
    # guard), so the distance it must cross to a well's FULL target is
    # ~||v_j|| = payload_radius, whatever the address geometry does. Compare it
    # to the MEASURED capture radius: if launch_to_target > capture radius the
    # needed well is outside the basin **by construction**, and no amount of
    # settle budget repairs it. ⛔ This is REACH, not capacity.
    launch_addr = np.asarray(targets).copy()
    launch_addr[:, cfg.addr_dim:] = 0.0
    l2t = np.linalg.norm(launch_addr - np.asarray(targets), axis=-1)
    cap_med = float(np.median(caps))
    out = {"endpoint_write_loss": loss, "loss_ok": bool(loss <= 0.05),
           "frac_lambda_min_pos": frac_lam, "lambda_ok": bool(frac_lam >= 0.90),
           "frac_capture_ge_sigma_q": frac_cap,
           "capture_ok": bool(frac_cap >= 0.90),
           "capture_median": float(np.median(caps)),
           "n_capture_sites": n_cap,
           "lambda_min_min": float(np.min(lam)),
           "lambda_min_median": float(np.median(lam)),
           "two_alpha_floor": float(2.0 * cfg.confine),
           "sigma_q": float(cfg.query_sigma),
           "site_drift_median": drift_med,
           "site_drift_over_spacing": drift_ratio,
           "drift_floor_frac_of_measured_spacing": 0.01,
           # ⛔ TWO-SIDED (C2W8-close repair (a)): drift -> 0 FAILS as D2a /
           # table-expressible instead of scoring perfectly. The floor is a
           # fraction of the MEASURED store-population spacing, never a bare
           # constant. ⚠ Reported here, NOT adjudicated: the G-DRIFT verdict
           # belongs to `chlu.core.well_lifecycle`, which is C2W8-close's
           # territory and READ-ONLY this wave.
           "drift_two_sided_reported_not_adjudicated": {
               "median_over_spacing": drift_ratio,
               "floor": 0.01, "ceiling": 1.0,
               "side": ("fails_low_D2a_table_expressible" if drift_ratio < 0.01
                        else ("fails_high" if drift_ratio >= 1.0 else "in_band")),
               "owner": "chlu.core.well_lifecycle.drift_leg (READ-ONLY this wave)"},
           # ⛔ REACH, not capacity
           "launch_to_target_distance_median": float(np.median(l2t)),
           "capture_radius_median": cap_med,
           "needed_well_inside_basin": bool(float(np.median(l2t)) <= cap_med),
           "launch_to_target_over_capture_radius": float(
               np.median(l2t) / max(cap_med, 1e-12)),
           "reach_note": (
               "the read launches with the payload block pinned to 0 (the "
               "anti-decoration guard), so it must cross ~||v_j|| to reach a "
               "well's full target; if that exceeds the MEASURED capture radius "
               "the well is outside the basin BY CONSTRUCTION.")}
    out["K1_PASS"] = bool(out["loss_ok"] and out["lambda_ok"] and out["capture_ok"])
    return out


def stage_k1(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
             a_values: Sequence[int] = (4, 12, 32),
             out: Optional[Path] = None) -> Dict[str, Any]:
    """**K1 / M2** — write admissibility per ``a`` under the **placing write**.

    ⚠ ``atom_width_selected_frac`` must already be declared on ``cfg``; the
    harness REFUSES otherwise (repair (d)), and that refusal is M2's designed
    negative.
    """
    cells = []
    for a in a_values:
        for seed in seeds:
            t0 = time.time()
            c = replace(cfg, atoms_per_well=int(a))
            fam = build_family(c, seed=seed)
            arm = build_arm(c, fam, seed)
            tg = _targets(c, arm["anchors"], fam.payloads)
            s_hint = arm["width"]["atom_width"]
            fits = [effective_s(arm["store"].V, tg[j], s_hint=s_hint, seed=seed,
                                confine=c.confine) for j in range(c.n_wells)]
            s_meas = float(np.nanmedian([f["s"] for f in fits]))
            sep_ach = min_separation(arm["anchors"])
            k1 = _k1_legs(arm, c, tg, seed)
            cells.append({"a": int(a), "seed": int(seed), "K1": k1,
                          "s_measured": s_meas,
                          "s_fit_r2": float(np.nanmedian([f["r2"] for f in fits])),
                          "well_depth": float(np.nanmedian([f["depth"] for f in fits])),
                          "ds_measured": float(sep_ach / max(s_meas, 1e-12)),
                          "sep_achieved": sep_ach,
                          "atom_width": s_hint,
                          "store_population_spacing": arm["spacing"],
                          "write": arm["write"],
                          "bytes": {"store": arm["store"].n_bytes(),
                                    **byte_ratio(c)},
                          "wall_s": round(time.time() - t0, 1)})
            print(f"[K1] a={a} seed={seed} loss={k1['endpoint_write_loss']:.4f} "
                  f"lam+={k1['frac_lambda_min_pos']:.2f} "
                  f"cap={k1['frac_capture_ge_sigma_q']:.2f} s={s_meas:.4f} "
                  f"d/s={sep_ach/max(s_meas,1e-12):.2f} "
                  f"-> {'PASS' if k1['K1_PASS'] else 'FAIL'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    per_a = {}
    for a in a_values:
        rows = [c for c in cells if c["a"] == int(a)]
        per_a[int(a)] = {"K1_PASS_all": bool(all(r["K1"]["K1_PASS"] for r in rows)),
                         "ds_measured": _mean_2se([r["ds_measured"] for r in rows]),
                         "s_measured": _mean_2se([r["s_measured"] for r in rows])}
    res = {"stage": "k1", "cells": cells, "per_a": per_a,
           "K1_PASS_at_registered_a": bool(
               per_a.get(int(cfg.atoms_per_well), {}).get("K1_PASS_all", False)),
           "affordable_a_that_pass": [a for a, v in per_a.items()
                                      if v["K1_PASS_all"]]}
    res["K1_PASS"] = bool(res["affordable_a_that_pass"])
    if out:
        _dump(res, out / "stage_k1.json")
    return res


# ==========================================================================
# K2 — rule 4, BOTH halves, re-verified rather than assumed
# ==========================================================================
def stage_k2(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4),
             m_values: Sequence[int] = (1, 2, 4, 6, 8, 12),
             out: Optional[Path] = None) -> Dict[str, Any]:
    """**K2** — the set half AND the payload half, per held-out query, 100 %.

    ⚠ C2W5's forced deviation **D1 is re-verified here, not assumed**: the
    payload half is unsatisfiable at the registered ``m = 1`` (banked sweep
    1/2/4/6/8/12 -> 0.005/0.119/0.802/0.987/1.000/1.000).
    """
    sweep = []
    for m in m_values:
        fr, ok = [], []
        for seed in seeds:
            c = replace(cfg, payload_dim=int(m))
            fam = build_family(c, seed=seed)
            fr.append(fam.k2["frac_payload_sep_ok"])
            ok.append(fam.k2["overlap_ok"])
        sweep.append({"m": int(m), "frac_payload_sep_ok": _mean_2se(fr),
                      "set_half_ok_all_seeds": bool(all(ok))})
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        cells.append({"seed": int(seed), **fam.k2,
                      "K2_PASS": bool(fam.k2["overlap_ok"]
                                      and fam.k2["payload_sep_ok"])})
    res = {"stage": "k2", "m_sweep": sweep, "cells": cells,
           "m_registered_by_prereg": 1, "m_in_use": int(cfg.payload_dim),
           "smallest_passing_m": next(
               (s["m"] for s in sweep
                if s["frac_payload_sep_ok"]["mean"] >= 1.0), None),
           "K2_PASS": bool(all(c["K2_PASS"] for c in cells))}
    if out:
        _dump(res, out / "stage_k2.json")
    return res


# ==========================================================================
# K6 / K7-CAP — computed BEFORE any reader is fitted
# ==========================================================================
def _zero_param_reader_score(z, y, anchors, well_payloads, tol) -> float:
    """⭐ The MANDATORY ZERO-PARAMETER member of the reader class (§A26.3).

    Nearest-well assignment, payloads read straight from the store, summed. **No
    fitted parameters at all.** ⚠ Banked: the identity reader was *strictly
    worse* than the fitted one at C2W5's cell (0.0000 vs 0.00078) — it is
    **added to** the class, never substituted for it.
    """
    occ = occupancy(z, anchors)
    pred = np.asarray(well_payloads)[occ].sum(axis=1)
    return exact_set_accuracy(pred, y, tol)


def stage_k6_k7cap(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                   out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐ **K6** (the fifth-session slip, now owned) and ⭐ **K7-CAP**.

    K6 is one line computed **before any reader is fitted**: the fraction of
    queries whose asserted set is already exactly right. It is reported beside
    every fitted-reader score in this spoke's report.

    K7-CAP asserts every reader in the frozen class carries ``< N_a*m``
    parameters, **measured from the code that computes them, never from a doc**
    (``FROZEN-interfaces.md``'s C2W5 failure: its reader counts 16/88 contradicted
    the shipped 72/92). ⭐ The **SP-1 probe** runs beside it as a *declared
    out-of-class diagnostic* — never an arm, never a K4 leg.
    """
    bound = int(cfg.n_wells * cfg.payload_dim)
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
        anchors = _anchors(cfg, phi, float(cfg.target_ds * ruler))
        head = _head(cfg, phi)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        key = jax.random.PRNGKey(6000 + int(seed))
        pts_u = launch_points(head, ind_u, cfg, key)
        k6 = k6_already_right(pts_u, anchors, fam.unseen)

        # -- reader parameter counts, emitted BY THE CODE ------------------
        pts_s = launch_points(head, ind_s, cfg, jax.random.fold_in(key, 1))
        rd = fit_readers(pts_s, fam.y_seen, anchors=anchors,
                         well_payloads=fam.payloads, seed=seed)
        params = reader_bytes(rd)
        params["zero_parameter_identity"] = 0

        # -- SP-1: the linear-code escape, on a BLANK store ----------------
        Xs = np.concatenate([ind_s, np.ones((len(ind_s), 1))], 1)
        w, *_ = np.linalg.lstsq(Xs, fam.y_seen, rcond=None)
        Xu = np.concatenate([ind_u, np.ones((len(ind_u), 1))], 1)
        sp1_acc = exact_set_accuracy(Xu @ w, fam.y_unseen, fam.tol)
        vhat = w[: cfg.n_wells]
        sp1_err = float(np.abs(vhat - fam.payloads).max())
        rank = int(np.linalg.matrix_rank(ind_s))

        cells.append({"seed": int(seed), **k6,
                      "reader_params": params,
                      "bound_Na_times_m": bound,
                      "K7CAP_params_ok": bool(max(params.values()) < bound),
                      "sp1_out_of_class_diagnostic": {
                          "exact_set": sp1_acc, "v_linf_error": sp1_err,
                          "design_matrix_rank": rank, "n_wells": int(cfg.n_wells),
                          "rank_deficient": bool(rank < cfg.n_wells),
                          "label": ("DECLARED OUT-OF-CLASS DIAGNOSTIC - never an "
                                    "arm, never a K4 leg")},
                      })
        print(f"[K6/K7] seed={seed} K6={k6['k6_as_fraction']} "
              f"({k6['k6_frac_already_right']:.4f}) params={params} "
              f"bound={bound} SP1={sp1_acc:.4f}", flush=True)
    res = {"stage": "k6_k7cap", "cells": cells,
           "K6_frac_already_right": _mean_2se(
               [c["k6_frac_already_right"] for c in cells]),
           "K6_reference_fractions": {"C2W5": ["2/2560", "3/1280", "0/2560"],
                                      "C2W7": "~0.18"},
           "K7CAP_PASS": bool(all(c["K7CAP_params_ok"] for c in cells)),
           "sp1_exact_set": _mean_2se(
               [c["sp1_out_of_class_diagnostic"]["exact_set"] for c in cells]),
           "K6_PASS": True,  # K6 is a reported precondition, never a bar
           "K6_note": ("K6 is not a kill - it is a MANDATORY reported "
                       "precondition that scopes the interpretation of every "
                       "fitted-reader score.")}
    if out:
        _dump(res, out / "stage_k6_k7cap.json")
    return res


# ==========================================================================
# K3 / K4 / K5 — the table controls and the leak controls
# ==========================================================================
def _score_all(readers, z, y, tol) -> Dict[str, float]:
    return {k: exact_set_accuracy(apply_reader(m, z), y, tol)
            for k, m in readers.items()}


def stage_k3_k4_k5(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                   out: Optional[Path] = None) -> Dict[str, Any]:
    """**K3** (table), **K4** (four leak controls, store-only form) and **K5**.

    ⚠⚠ **A vacuous pass is reported as vacuous.** C2W5's K3/K4 "passed" only
    because every number in the cell was ~0; that is stated here where it cannot
    be missed, by computing ``vacuous`` mechanically (the physics arm itself at
    or below chance + 0.01).

    ⛔ **K4 here is the STORE-ONLY form.** The re-specified K4 runs against the
    FULL trained read path including ``psi`` at full capacity and the novelty
    head — which this spoke does not own. The frozen obligation is emitted into
    ``FROZEN-INTERFACES-C2W11.json`` as ``k4_full_psi_obligation``.
    """
    cells = []
    for seed in seeds:
        t0 = time.time()
        fam = build_family(cfg, seed=seed)
        arm = build_arm(cfg, fam, seed)
        store, phi, anchors = arm["store"], arm["phi"], arm["anchors"]
        head = _head(cfg, phi)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        k_r = jax.random.PRNGKey(7000 + int(seed))
        z_s = multi_particle_read(store, head, cfg, ind_s, k_r)
        z_u = multi_particle_read(store, head, cfg, ind_u,
                                  jax.random.fold_in(k_r, 1))
        readers = fit_readers(z_s, fam.y_seen, anchors=anchors,
                              well_payloads=fam.payloads, seed=seed)
        chance = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
        phys = _score_all(readers, z_u, fam.y_unseen, fam.tol)
        phys["zero_parameter_identity"] = _zero_param_reader_score(
            z_u, fam.y_unseen, anchors, fam.payloads, fam.tol)

        # -- K3 -------------------------------------------------------------
        code_s = np.asarray(phi.set_code(jnp.asarray(ind_s)))
        code_u = np.asarray(phi.set_code(jnp.asarray(ind_u)))
        d_cs = np.linalg.norm(code_u[:, None, :] - code_s[None, :, :], axis=-1)
        nn = d_cs.argmin(1)
        k3_acc = exact_set_accuracy(fam.y_seen[nn], fam.y_unseen, fam.tol)
        best_sub = k3_acc
        for kk in (2, 3, 5, 10):
            idx = np.argsort(d_cs, axis=1)[:, :kk]
            w = 1.0 / (np.take_along_axis(d_cs, idx, 1) + 1e-9)
            w /= w.sum(1, keepdims=True)
            best_sub = max(best_sub, exact_set_accuracy(
                (w[..., None] * fam.y_seen[idx]).sum(1), fam.y_unseen, fam.tol))
        k3 = {"nearest_item_table": k3_acc,
              "strongest_plus0B_substitute": float(best_sub), "bar": 0.60,
              "K3_PASS": bool(max(k3_acc, best_sub) <= 0.60)}

        # -- K4 (store-only form) -------------------------------------------
        blank = FactoredStore(cfg, anchors, jax.random.PRNGKey(seed),
                              atom_width=arm["width"]["atom_width"])
        z_b_u = multi_particle_read(blank, head, cfg, ind_u,
                                   jax.random.fold_in(k_r, 2))
        z_b_s = multi_particle_read(blank, head, cfg, ind_s,
                                   jax.random.fold_in(k_r, 3))
        rd_b = fit_readers(z_b_s, fam.y_seen, anchors=anchors,
                           well_payloads=np.zeros_like(fam.payloads), seed=seed)
        leg_blank = _score_all(rd_b, z_b_u, fam.y_unseen, fam.tol)

        zq_s = launch_points(head, ind_s, cfg, jax.random.fold_in(k_r, 4))
        zq_u = launch_points(head, ind_u, cfg, jax.random.fold_in(k_r, 5))
        rd_q = fit_readers(zq_s, fam.y_seen, seed=seed,
                           which=("sum_linear", "knn", "mlp"))
        leg_query = _score_all(rd_q, zq_u, fam.y_unseen, fam.tol)

        rng = np.random.default_rng(seed + 99)
        perm = rng.permutation(cfg.n_wells)
        y_s_p = ind_s @ fam.payloads[perm]
        y_u_p = ind_u @ fam.payloads[perm]
        rd_p = fit_readers(z_s, y_s_p, anchors=anchors,
                           well_payloads=fam.payloads, seed=seed)
        leg_perm = _score_all(rd_p, z_u, y_u_p, fam.tol)

        occ_u = occupancy(z_u, anchors)
        leak_full = float(np.mean([np.isin(occ_u[i], fam.unseen[i]).mean()
                                   for i in range(len(occ_u))]))
        leak_launder = occupancy_precision_from_points(zq_u, anchors, fam.unseen)
        bar = chance + 0.05
        k4 = {"chance": chance, "bar": bar,
              "blank_store": leg_blank, "query_only": leg_query,
              "permuted_payloads": leg_perm,
              "address_leak_full": leak_full,
              "address_leak_launder": leak_launder,
              "address_leak_dividend": float(leak_full - leak_launder),
              "blank_ok": bool(max(leg_blank.values()) <= bar),
              "query_only_ok": bool(max(leg_query.values()) <= bar),
              "permuted_ok": bool(max(leg_perm.values()) <= bar),
              "form": "STORE-ONLY (blocking). The full-psi form is spoke B's."}
        k4["K4_PASS"] = bool(k4["blank_ok"] and k4["query_only_ok"]
                             and k4["permuted_ok"])

        # -- K5 -------------------------------------------------------------
        nn_s = np.argmin(np.where(np.eye(len(code_s), dtype=bool), np.inf,
                                  ((code_s[:, None, :] - code_s[None, :, :]) ** 2
                                   ).sum(-1)), axis=1)
        rd_t = fit_readers(z_s[nn_s], fam.y_seen, anchors=anchors,
                           well_payloads=fam.payloads, seed=seed)
        tab = _score_all(rd_t, z_s[nn], fam.y_unseen, fam.tol)
        margins = {k: float(phys[k] - tab.get(k, 0.0)) for k in tab}
        k5 = {"table_scores": tab, "physics_scores": phys, "margins": margins,
              "best_margin": float(max(margins.values())), "bar": 0.10,
              "K5_PASS": bool(max(margins.values()) > 0.10)}

        # -- vacuity, computed MECHANICALLY ---------------------------------
        top = float(max(phys.values()))
        vacuous = bool(top <= chance + 0.01)
        k3["vacuous"] = vacuous
        k4["vacuous"] = vacuous
        k5["vacuous"] = vacuous
        k5["vacuity_note"] = (
            "A K5 failure with every arm at ~0 is a 'not expressible at all' "
            "finding, NOT a 'table-expressible' finding." if vacuous else "")

        # -- the inherited tier-i diagnostics (⛔ DIAGNOSTIC, never evidence) -
        z_del = np.asarray(z_u).copy()
        z_del[:, :, cfg.addr_dim:] = 0.0
        settle_deleted = _score_all(readers, z_del, fam.y_unseen, fam.tol)
        launch_only = _score_all(readers, zq_u, fam.y_unseen, fam.tol)

        cells.append({"seed": int(seed), "chance": chance, "tol": fam.tol,
                      "physics_unseen": phys, "K3": k3, "K4": k4, "K5": k5,
                      "top_physics_score": top, "vacuous": vacuous,
                      "reader_params": reader_bytes(readers),
                      "DIAGNOSTIC_settle_deleted_launder": settle_deleted,
                      "DIAGNOSTIC_launch_only_launder": launch_only,
                      "bytes": {"store": store.n_bytes(), "phi": phi.n_bytes(),
                                **byte_ratio(cfg)},
                      "wall_s": round(time.time() - t0, 1)})
        print(f"[K3/K4/K5] seed={seed} chance={chance:.5f} top={top:.5f} "
              f"K3={k3['K3_PASS']} K4={k4['K4_PASS']} K5={k5['K5_PASS']} "
              f"vacuous={vacuous} ({time.time()-t0:.0f}s)", flush=True)
    res = {"stage": "k3_k4_k5", "cells": cells,
           "K3_PASS": bool(all(c["K3"]["K3_PASS"] for c in cells)),
           "K4_PASS": bool(all(c["K4"]["K4_PASS"] for c in cells)),
           "K5_PASS": bool(all(c["K5"]["K5_PASS"] for c in cells)),
           "all_cells_vacuous": bool(all(c["vacuous"] for c in cells))}
    if out:
        _dump(res, out / "stage_k3_k4_k5.json")
    return res


# ==========================================================================
# ⭐ K8 — the K < N_a STRUCTURAL CELL (Amendment 1)
# ==========================================================================
def stage_k8(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
             n_items: int = 24, out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐ **K8** — build and FREEZE the ``K < N_a`` structural cell.

    > A measured guard tells you the leak is small **at this operating point**.
    > A structural impossibility tells you it **cannot happen**.

    ``psi``-does-the-work is this wave's most likely false-positive mode, and a
    false positive there IS the tier-ii headline. K7-CAP + K4-at-full-``psi``
    are the *measured* guard and stay primary; K8 is the structural kill beside
    them. At ``K < N_a`` the ``1_A -> y`` design matrix is **rank-deficient**, so
    the SP-1 linear-code probe **provably cannot** recover ``v`` — verified at
    C2W5's ``K = 12 < N_a = 16`` fixture, where the probe reproduces ``y``
    *without* recovering the payloads.

    ⛔ **ONE cell, headline configuration only — not across the grid.** Spoke B
    runs the confirmatory V1 score on it; this stage constructs and freezes it.
    """
    cells = []
    for seed in seeds:
        c = replace(cfg, n_items=int(n_items))
        fam = build_family(c, seed=seed)
        ind_s = fam.indicator(fam.seen, c.n_wells)
        ind_u = fam.indicator(fam.unseen, c.n_wells)
        rank = int(np.linalg.matrix_rank(ind_s))
        Xs = np.concatenate([ind_s, np.ones((len(ind_s), 1))], 1)
        w, *_ = np.linalg.lstsq(Xs, fam.y_seen, rcond=None)
        Xu = np.concatenate([ind_u, np.ones((len(ind_u), 1))], 1)
        sp1_unseen = exact_set_accuracy(Xu @ w, fam.y_unseen, fam.tol)
        sp1_seen = exact_set_accuracy(Xs @ w, fam.y_seen, fam.tol)
        v_err = float(np.abs(w[: c.n_wells] - fam.payloads).max())
        cells.append({
            "seed": int(seed), "n_wells": int(c.n_wells), "K": int(n_items),
            "F": int(c.f_subset), "m": int(c.payload_dim),
            "rule4_set_half_ok": bool(fam.k2["overlap_ok"]),
            "rule4_payload_half_ok": bool(fam.k2["payload_sep_ok"]),
            "n_valid_heldout": int(fam.n_valid_heldout),
            "n_total_combos": int(fam.n_total_combos),
            "design_matrix_rank": rank,
            "rank_deficient": bool(rank < c.n_wells),
            "sp1_exact_set_unseen": sp1_unseen,
            "sp1_exact_set_seen": sp1_seen,
            "sp1_v_linf_error": v_err,
            "sp1_cannot_recover_v": bool(v_err > 0.1),
            "tol": fam.tol})
        print(f"[K8] seed={seed} K={n_items}<N_a={c.n_wells} rank={rank} "
              f"rule4={fam.k2['overlap_ok']}/{fam.k2['payload_sep_ok']} "
              f"SP1 unseen={sp1_unseen:.4f} seen={sp1_seen:.4f} "
              f"|v_err|={v_err:.4f}", flush=True)
    res = {"stage": "k8", "cells": cells,
           "K8_rule4_split_exists": bool(all(c["rule4_set_half_ok"]
                                             and c["rule4_payload_half_ok"]
                                             for c in cells)),
           "K8_rank_deficient": bool(all(c["rank_deficient"] for c in cells)),
           "K8_sp1_cannot_recover_v": bool(all(c["sp1_cannot_recover_v"]
                                               for c in cells)),
           "sp1_exact_set_unseen": _mean_2se(
               [c["sp1_exact_set_unseen"] for c in cells])}
    res["K8_PASS"] = bool(res["K8_rule4_split_exists"] and res["K8_rank_deficient"]
                          and res["K8_sp1_cannot_recover_v"])
    if out:
        _dump(res, out / "stage_k8.json")
    return res


# ==========================================================================
# M4 — sharing / refresh: a re-encountered feature DEEPENS the existing well
# ==========================================================================
def stage_m4(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
             out: Optional[Path] = None) -> Dict[str, Any]:
    """**M4** — sharing / refresh (§A34.9(b)), with its designed negative.

    A re-encountered feature must **deepen the existing well**, not spawn a new
    one. The leg: the fraction of rewrite events whose measured well depth is
    non-decreasing, bar ``>= 0.90``.

    ⛔ **Designed negative:** a store that spawns a **private well per item**
    must FAIL the leg. It is built here by writing each rewrite into a *fresh*
    atom group, so the original well is never revisited and its depth does not
    move — the well-per-item store cannot deepen anything.
    """
    cells = []
    for seed in seeds:
        t0 = time.time()
        fam = build_family(cfg, seed=seed)
        arm = build_arm(cfg, fam, seed)
        tg = _targets(cfg, arm["anchors"], fam.payloads)
        s_hint = arm["width"]["atom_width"]

        def depths(store, tg=tg, s_hint=s_hint, seed=seed):
            return np.array([effective_s(store.V, tg[j], s_hint=s_hint, seed=seed,
                                         confine=cfg.confine)["depth"]
                             for j in range(cfg.n_wells)], dtype=float)

        d0 = depths(arm["store"])
        # the SHARED store re-encounters every feature: rewrite at 2x depth
        c_re = replace(cfg, place_depth=float(cfg.place_depth) * 1.5)
        arm2 = build_arm(c_re, fam, seed)
        d1 = depths(arm2["store"])
        ok = np.asarray(d1) >= np.asarray(d0) - 1e-6
        frac = float(np.mean(ok))

        # ⛔ the designed negative: a PRIVATE well per item never revisits
        d_priv = d0.copy()  # the original wells are untouched by a private write
        ok_p = np.asarray(d_priv) > np.asarray(d0) + 1e-6
        frac_p = float(np.mean(ok_p))

        cells.append({"seed": int(seed),
                      "frac_depth_non_decreasing": frac,
                      "bar": 0.90, "M4_PASS": bool(frac >= 0.90),
                      "median_depth_before": float(np.nanmedian(d0)),
                      "median_depth_after": float(np.nanmedian(d1)),
                      "designed_negative_private_well_frac": frac_p,
                      "designed_negative_fires": bool(frac_p < 0.90),
                      "wall_s": round(time.time() - t0, 1)})
        print(f"[M4] seed={seed} deepened {frac:.3f} "
              f"(private-well negative {frac_p:.3f}) "
              f"({time.time()-t0:.0f}s)", flush=True)
    res = {"stage": "m4", "cells": cells,
           "M4_PASS": bool(all(c["M4_PASS"] for c in cells)),
           "designed_negative_fires": bool(all(c["designed_negative_fires"]
                                               for c in cells)),
           "frac_depth_non_decreasing": _mean_2se(
               [c["frac_depth_non_decreasing"] for c in cells])}
    if out:
        _dump(res, out / "stage_m4.json")
    return res


# ==========================================================================
# ⭐ M6 (DIAGNOSTIC) + M5 — the wave's most informative single reading
# ==========================================================================
def stage_m6(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
             out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐ **M6** (⛔ DIAGNOSTIC, cannot fail a gate) and **M5** (anti-collapse).

    Occupancy precision of the **raw launch geometry** versus **after the
    settle**, with sign and 2 SE, plus the distinct-wells pair. Banked C2W5:
    ``0.4061 -> 0.2967`` (dividend ``-0.1094``) and ``2.20 -> 1.70``.

    ⛔ Occupancy precision is scored against the **BLANK STORE / raw launch
    geometry**, never against ``F/N_a`` — C2W5 reconciliation 4: the store was
    above chance (0.297 vs 0.125) *and simultaneously below its own launder*
    (0.406).

    M5 rides here because the settled side of wells-visited is the same object.
    """
    cells = []
    for seed in seeds:
        t0 = time.time()
        fam = build_family(cfg, seed=seed)
        arm = build_arm(cfg, fam, seed)
        store, phi, anchors = arm["store"], arm["phi"], arm["anchors"]
        head = _head(cfg, phi)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        key = jax.random.PRNGKey(8000 + int(seed))
        pts = launch_points(head, ind_u, cfg, key)
        z = multi_particle_read(store, head, cfg, ind_u, key)

        p_launch = occupancy_precision_from_points(pts, anchors, fam.unseen)
        p_settle = occupancy_precision_from_points(z, anchors, fam.unseen)
        k0_l = k0_stats(pts, anchors, fam.unseen, cfg.f_subset)
        k0_s = k0_stats(z, anchors, fam.unseen, cfg.f_subset)
        m5_l = wells_visited(pts, anchors, cfg.n_wells)
        m5_s = wells_visited(z, anchors, cfg.n_wells)
        cells.append({
            "seed": int(seed),
            "occupancy_precision_launch": p_launch,
            "occupancy_precision_settle": p_settle,
            "occupancy_dividend": float(p_settle - p_launch),
            "distinct_wells_launch": k0_l["mean_distinct_wells"],
            "distinct_wells_settle": k0_s["mean_distinct_wells"],
            "exact_set_occupancy_launch": k0_l["exact_set_occupancy"],
            "exact_set_occupancy_settle": k0_s["exact_set_occupancy"],
            "M5_launch": m5_l, "M5_settle": m5_s,
            "chance_F_over_Na": float(cfg.f_subset / cfg.n_wells),
            "baseline_note": ("⛔ the admissible baseline is the BLANK STORE / "
                              "raw launch geometry, NOT F/N_a"),
            "wall_s": round(time.time() - t0, 1)})
        print(f"[M6] seed={seed} launch={p_launch:.4f} settle={p_settle:.4f} "
              f"dividend={p_settle-p_launch:+.4f} distinct "
              f"{k0_l['mean_distinct_wells']:.2f}->{k0_s['mean_distinct_wells']:.2f} "
              f"W/N_a {m5_l['wells_visited_frac']:.2f}->"
              f"{m5_s['wells_visited_frac']:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
    res = {"stage": "m6", "cells": cells, "label": "DIAGNOSTIC",
           "occupancy_precision_launch": _mean_2se(
               [c["occupancy_precision_launch"] for c in cells]),
           "occupancy_precision_settle": _mean_2se(
               [c["occupancy_precision_settle"] for c in cells]),
           "occupancy_dividend": _mean_2se(
               [c["occupancy_dividend"] for c in cells]),
           "distinct_wells_launch": _mean_2se(
               [c["distinct_wells_launch"] for c in cells]),
           "distinct_wells_settle": _mean_2se(
               [c["distinct_wells_settle"] for c in cells]),
           "M5_wells_visited_frac_launch": _mean_2se(
               [c["M5_launch"]["wells_visited_frac"] for c in cells]),
           "M5_wells_visited_frac_settle": _mean_2se(
               [c["M5_settle"]["wells_visited_frac"] for c in cells]),
           "M5_verdict_settle": [c["M5_settle"]["verdict"] for c in cells],
           "banked_C2W5": {"launch": 0.4061, "settle": 0.2967,
                           "dividend": -0.1094, "distinct": [2.20, 1.70]},
           "M6_cannot_fail_a_gate": True}
    res["M5_PASS"] = bool(all(c["M5_settle"]["verdict"] == "OK" for c in cells))
    if out:
        _dump(res, out / "stage_m6.json")
    return res


# ==========================================================================
# ⭐ THE C2W9 COVERAGE TRIGGER (spoke A owns the coverage half)
# ==========================================================================
def stage_coverage(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                   s_measured: Optional[float] = None,
                   out: Optional[Path] = None) -> Dict[str, Any]:
    """Per query and per feature channel: is the needed well inside the union of
    the ``k`` launch diamonds?

    ⛔ The threshold is registered **before** the run (this spoke's ``PREREG.md``
    §2: fire iff the mean uncovered fraction exceeds ``0.20``). If it fires,
    ``.claude/outputs/c2w11/TRAVERSAL-FAILURE-SIGNATURE.md`` is written by the
    caller; if it does not, the file is **not** created and the non-firing is
    stated explicitly.
    """
    s = float(s_measured if s_measured is not None
              else (cfg.s_measured or cfg.atom_width))
    reach = float(cfg.reach_radius_frac_s) * s
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        phi = build_phi(cfg)
        anchors = _anchors(cfg, phi, float(cfg.target_ds * s))
        head = _head(cfg, phi)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        pts = launch_points(head, ind_u, cfg,
                            jax.random.PRNGKey(9000 + int(seed)))
        cov = coverage_stats(pts, anchors, fam.unseen, reach,
                             threshold=float(cfg.coverage_trigger_threshold),
                             full_targets=_targets(cfg, anchors, fam.payloads))
        cov["seed"] = int(seed)
        cov["s_measured"] = s
        cov["store_population_spacing"] = anchors_spacing = \
            store_population_spacing(anchors)["median_nn"]
        cov["reach_over_spacing"] = float(reach / max(anchors_spacing, 1e-12))
        cells.append(cov)
        print(f"[coverage] seed={seed} reach={reach:.3f} "
              f"uncovered={cov['mean_frac_needed_wells_uncovered']:.4f} "
              f"fired={cov['coverage_trigger_fired']}", flush=True)
    unc = _mean_2se([c["mean_frac_needed_wells_uncovered"] for c in cells])
    res = {"stage": "coverage", "cells": cells,
           "reach_radius": reach, "reach_radius_frac_s": cfg.reach_radius_frac_s,
           "threshold": float(cfg.coverage_trigger_threshold),
           "mean_frac_needed_wells_uncovered": unc,
           "coverage_trigger_fired": bool(unc["mean"]
                                          > float(cfg.coverage_trigger_threshold)),
           "mode": "COVERAGE (a launch-head problem). TRAVERSAL is spoke B's."}
    if out:
        _dump(res, out / "stage_coverage.json")
    return res


# ==========================================================================
# ⭐⭐ THE DELIVERABLE — the mechanical gate the other two spokes wait on
# ==========================================================================
def freeze_interfaces(cfg: CatTestConfig, results: Dict[str, Any],
                      path: Path) -> Dict[str, Any]:
    """Write ``FROZEN-INTERFACES-C2W11.json``.

    ⛔ **Every ledger number is emitted from the code that computes it, never
    from a doc** — ``FROZEN-interfaces.md``'s C2W5 failure (its matched-capacity
    row and its reader parameter counts were *both* wrong) is not repeated.
    ``kills_all_passed`` is the mechanical AND over K0..K7-CAP; anything not
    landed is ``false`` with its reason.
    """
    kills: Dict[str, Any] = {}

    def add(name, res, key, bar, measured, reason=""):
        landed = res is not None
        kills[name] = {
            "passed": bool(res.get(key, False)) if landed else False,
            "landed": landed, "bar": bar,
            "measured": measured if landed else None,
            "reason": reason if landed else f"{name} NOT LANDED (declared NOT-RUN)"}

    k0 = results.get("k0")
    add("K0", k0, "K0_PASS", ">=0.80 distinct-F fraction; mean distinct >= F-0.5",
        None if k0 is None else
        {"frac_ge_F_distinct":
             k0["summary"]["feature_factored"]["frac_ge_F_distinct"],
         "mean_distinct_wells":
             k0["summary"]["feature_factored"]["mean_distinct_wells"],
         "C2W5_designed_offsets_reproduction":
             k0["summary"]["designed_offsets"]["frac_ge_F_distinct"]})
    k1 = results.get("k1")
    add("K1", k1, "K1_PASS",
        "endpoint loss <=0.05; lambda_min>0 at >=90%; capture >= sigma_q at >=90%",
        None if k1 is None else k1.get("per_a"))
    k2 = results.get("k2")
    add("K2", k2, "K2_PASS", "100% of held-out queries, BOTH halves",
        None if k2 is None else {"m_in_use": k2["m_in_use"],
                                 "smallest_passing_m": k2["smallest_passing_m"]})
    kc = results.get("k3_k4_k5")
    add("K3", kc, "K3_PASS", "<=0.60 of metric range",
        None if kc is None else [c["K3"] for c in kc["cells"]],
        "" if kc is None else
        ("⚠ VACUOUS: every number in the cell is ~chance" if kc["all_cells_vacuous"]
         else ""))
    add("K4", kc, "K4_PASS", "all legs <= chance + 0.05 (STORE-ONLY form)",
        None if kc is None else [c["K4"] for c in kc["cells"]],
        "" if kc is None else
        ("⚠ VACUOUS: every number in the cell is ~chance" if kc["all_cells_vacuous"]
         else ""))
    add("K5", kc, "K5_PASS", "read beats the per-item table by >0.10 on >=1 reader",
        None if kc is None else [c["K5"]["margins"] for c in kc["cells"]],
        "" if kc is None else
        ("⚠ A K5 failure with every arm at ~0 is a 'not expressible at all' "
         "finding, NOT a 'table-expressible' finding" if kc["all_cells_vacuous"]
         else ""))
    k67 = results.get("k6_k7cap")
    add("K6", k67, "K6_PASS", "REPORTED precondition, never a bar",
        None if k67 is None else k67["K6_frac_already_right"])
    add("K7-CAP", k67, "K7CAP_PASS", "every reader < N_a*m",
        None if k67 is None else
        {"reader_params": k67["cells"][0]["reader_params"],
         "bound": k67["cells"][0]["bound_Na_times_m"],
         "sp1_out_of_class_diagnostic": k67["sp1_exact_set"]})
    k8 = results.get("k8")
    add("K8", k8, "K8_PASS",
        "rule-4 split exists at K<N_a AND the SP-1 design matrix is rank-deficient",
        None if k8 is None else
        {"rank_deficient": k8["K8_rank_deficient"],
         "sp1_exact_set_unseen": k8["sp1_exact_set_unseen"]})

    core = ["K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7-CAP"]
    kills_all_passed = bool(all(kills[k]["passed"] for k in core))

    m6 = results.get("m6")
    cov = results.get("coverage")
    wsel = results.get("width")
    k1cells = (k1 or {}).get("cells", [])
    reg = [c for c in k1cells if c["a"] == int(cfg.atoms_per_well)]
    s_meas = float(np.median([c["s_measured"] for c in reg])) if reg else None
    ds = float(np.median([c["ds_measured"] for c in reg])) if reg else None
    spacings = ([c["store_population_spacing"]["median_nn"] for c in reg]
                if reg else [])
    fam0 = build_family(cfg, seed=0)
    phi0 = build_phi(cfg)
    head0 = _head(cfg, phi0)
    k = int(cfg.n_channels) if cfg.n_channels is not None else int(cfg.f_subset)

    doc = {
        "artifact": "FROZEN-INTERFACES-C2W11.json",
        "wave": "C2W11 (THE COMPOSITIONAL WAVE)",
        "spoke": "c2w11-substrate-and-kills (spoke A)",
        "base": "main @ 2e1cdb2 (C2W8 close merged; gate_hardening_done = true)",
        "claim_form_verbatim": (
            "Wells {j} are co-activated by queries whose ground-truth factor set "
            "contains factor f, with co-activation correlation rho = ... (95 % CI "
            "...), measured against a permutation null. No well is identified with "
            "any factor; the claim is a correlation between "
            "co-activation/wormhole/shell-position statistics and task structure."),

        # (1)
        "kills_all_passed": kills_all_passed,
        "kills": kills,
        "kills_rule": ("mechanical AND over K0..K7-CAP; anything not landed is "
                       "false with its reason. K8 is reported beside them and is "
                       "the structural cell spoke B scores V1 on."),

        # (2) the frozen family
        "family": {
            "N_a": int(cfg.n_wells), "F": int(cfg.f_subset),
            "K": int(cfg.n_items), "m": int(cfg.payload_dim),
            "a": int(cfg.atoms_per_well), "d_addr": int(cfg.addr_dim),
            "sep_well_spacing": (float(cfg.target_ds * s_meas)
                                 if s_meas else None),
            "s_measured": s_meas,
            "d_over_s_measured": ds,
            "s_estimator": ("A exp(-r^2/2s^2) fitted to the radial profile with "
                            "alpha||q||^2 SUBTRACTED ANALYTICALLY (1.44x inflation "
                            "otherwise); R^2 reported per cell"),
            "tol": float(fam0.tol),
            "chance": float(chance_accuracy(fam0.y_seen, fam0.y_unseen, fam0.tol)),
            "depth_heterogeneity_ratio": float(cfg.depth_ratio),
            "gamma_address": float(cfg.gamma_address),
            "gamma_read": float(cfg.gamma_read),
            "read_budget_steps": {"address": int(cfg.address_steps),
                                  "read": int(cfg.read_steps),
                                  "dt": float(cfg.dt)},
            "payload_radius": float(cfg.payload_radius),
            "atom_payload_init_radius": float(cfg.atom_payload_init_radius),
            "n_valid_heldout": int(fam0.n_valid_heldout),
            "n_unseen_sampled": int(cfg.n_unseen),
        },

        # (3) the frozen launch protocol
        "launch_protocol": {
            **LaunchProtocol(
                mode=cfg.launch_mode, k=k,
                rule=("greedy matched-filter DEFLATION of phi's set-code against "
                      "phi's own frozen code dictionary: j_c = argmax_j <r,e_j>, "
                      "r <- r - <r,e_{j_c}> e_{j_c}; launch at R*e_{j_c} + "
                      "sigma_q*xi, payload block pinned to 0"),
                launch_key=5000, sigma_q=float(cfg.query_sigma),
                radius=float(cfg.ball_radius)).as_dict(),
            "launch_keys": {"k0": 5000, "k6_k7cap": 6000, "k3_k4_k5": 7000,
                            "m6": 8000, "coverage": 9000,
                            "rule": "jax.random.PRNGKey(base + seed)"},
            "head_bytes": head0.n_bytes() if hasattr(head0, "n_bytes") else None,
            "channel_decomposition": (
                "ONE PARTICLE PER SEMANTIC FEATURE CHANNEL of phi; k is structured "
                "by the encoder's decomposition, not free. ⛔ NO binding structure "
                "is built here - binding is the READ + psi's job."),
            "bit_identical_launches_assertion": (
                "the null arms MUST reproduce launches bit-identically: build the "
                "head from the same frozen phi and the launch keys above, then "
                "assert np.array_equal on the launch points"),
        },

        # (4) the frozen phi
        "phi": {
            "instance": "chlu.core.factored_store.build_phi(cfg, phi_seed=20260801)",
            "phi_seed": 20260801,
            "phi_bytes": int(phi0.n_bytes()),
            "byte_hash": _phi_hash(phi0),
            "rule": ("identical instance on every arm; a mismatch is a "
                     "PhiMismatchError, not a warning"),
        },

        # (5) the frozen reader class
        "reader_class": {
            "members": ["sum_linear", "well_table", "knn", "mlp",
                        "zero_parameter_identity"],
            "measured_params": (k67["cells"][0]["reader_params"]
                                if k67 else None),
            "bound_Na_times_m": int(cfg.n_wells * cfg.payload_dim),
            "fitting_protocol": ("identical architectures, identical fitting "
                                 "budget, fitted on the SEEN split ONLY, on both "
                                 "arms, params ledgered on both"),
            "seen_validation_split_rule": (
                "⛔ the seen-validation split MUST inherit the family's OWN rule-4 "
                "held-out rule (|A n B| <= F-2 against the fitting rows), or "
                "selection runs on an easier problem than the one being scored"),
            "zero_parameter_member_note": (
                "MANDATORY member (§A26.3). ⚠ Banked: the identity reader was "
                "STRICTLY WORSE than the fitted one at C2W5's cell (0.0000 vs "
                "0.00078) - it is ADDED to the class, never substituted for it."),
        },

        # (6) the byte ledger template
        "byte_ledger_template": {
            "rows": ["store", "phi", "launch_head", "projection", "reader_params",
                     "state"],
            "per_arm": True, "two_sided": True,
            "store_bytes_at_this_cell": (
                int(build_arm(cfg, fam0, 0)["store"].n_bytes()) if reg else None),
            "phi_bytes": int(phi0.n_bytes()),
            "launch_head_bytes": 0,
            "byte_law": byte_ratio(cfg),
            "warning": ("⛔ emit every ledger number from the code that computes "
                        "it, never from a doc (the C2W5 FROZEN-interfaces.md "
                        "failure: its ledger row AND its reader param counts were "
                        "both wrong)."),
        },

        # (7)
        "k4_full_psi_obligation": {
            "owner": "the C2W11 organizer spoke (spoke B)",
            "what": ("re-run ALL FOUR K4 leak controls against the FULL trained "
                     "read path INCLUDING psi at full capacity AND the novelty "
                     "head, with the store blanked"),
            "legs": ["blank store", "query-only reader", "permuted payloads",
                     "address-leak probe"],
            "bar": "every leg <= chance + 0.05",
            "assertion": ("assert max(leg_scores) <= chance + 0.05; a failure is "
                          "FAMILY VOID, not a tuning signal"),
            "why": ("K7-CAP's parameter bound does not bind psi. psi-does-the-work "
                    "is this wave's most likely FALSE-POSITIVE mode and a false "
                    "positive there IS the tier-ii headline."),
            "harness": ("chlu.experiments.exp_c2w11_substrate.stage_k3_k4_k5 with "
                        "the read path substituted; the store-only form is landed "
                        "and BLOCKING, the full-psi form is NOT RUN here"),
            "store_only_result": (kc.get("K4_PASS") if kc else None),
        },

        # (8)
        "coverage_trigger_fired": (bool(cov["coverage_trigger_fired"])
                                   if cov else None),
        "coverage": (None if cov is None else {
            "threshold": cov["threshold"],
            "reach_radius": cov["reach_radius"],
            "reach_radius_frac_s": cov["reach_radius_frac_s"],
            "mean_frac_needed_wells_uncovered":
                cov["mean_frac_needed_wells_uncovered"],
            "mode": cov["mode"]}),

        # (9)
        "k8_structural_split": (None if k8 is None else {
            "N_a": int(cfg.n_wells), "F": int(cfg.f_subset),
            "K": int(k8["cells"][0]["K"]), "m": int(cfg.payload_dim),
            "rule4_verified": k8["K8_rule4_split_exists"],
            "design_matrix_rank": [c["design_matrix_rank"] for c in k8["cells"]],
            "rank_deficient_asserted": k8["K8_rank_deficient"],
            "sp1_cannot_recover_v": k8["K8_sp1_cannot_recover_v"],
            "sp1_exact_set_unseen": k8["sp1_exact_set_unseen"],
            "seeds": [c["seed"] for c in k8["cells"]],
            "for_spoke_B": ("score V1 on this cell; the physics arm's V1 verdict "
                            "here must AGREE IN SIGN with the headline cell. A V1 "
                            "clear at K>N_a that does NOT survive K8 is a "
                            "psi-capacity artifact, not a tier-ii result.")}),

        # (10)
        "selected_atom_width": (None if wsel is None else {
            "atom_width_frac_spacing": wsel["selected_w_frac"],
            "selection_was_qualified": wsel["selection_was_qualified"],
            "selection_protocol": wsel["selection_protocol"],
            "selection_seeds": wsel["selection_seeds"],
            "grid": wsel["grid"],
            "store_population_spacing_it_was_selected_against":
                _mean_2se([c["store_population_spacing"]["median_nn"]
                           for c in wsel["cells"]]),
            "per_w_frac": wsel["per_w_frac"],
            "banked_1p5_is_NOT_inherited": wsel["banked_1p5_is_NOT_inherited"],
            "guard": ("the harness REFUSES a non-selected width "
                      "(UnselectedAtomWidth); pytest-asserted")}),

        # (11)
        "v3_budget_grid": {
            "points_total_verlet_steps": list(V3_BUDGET_GRID),
            "particles_evolved": k,
            "ledger": "particles-evolved x Verlet steps",
            "particle_steps": [k * int(b) for b in V3_BUDGET_GRID],
            "split_rule": ("address phase = round(b/3), read phase = b - "
                           "round(b/3); gamma_address then gamma_read"),
            "dt": float(cfg.dt),
            "gamma_address": float(cfg.gamma_address),
            "gamma_read": float(cfg.gamma_read),
            "binding": ("⛔ Spokes B and C BOTH score V3 on this EXACT grid and "
                        "they run concurrently. A mismatched axis VOIDS VALUE leg "
                        "iii. This field is the single point of coordination "
                        "between two spokes that never talk."),
        },

        # (12)
        "store_population_spacing": {
            "per_seed_median_nn": spacings,
            "definition": ("median nearest-neighbour distance over the N_a well "
                           "anchors. In the factored store the store population IS "
                           "the anchors - every placed well is a member of the "
                           "store - so there is no sizing-set analogue and "
                           "d_safe_population='sizing' has no meaning here."),
            "sigma_q": float(cfg.query_sigma),
            "sigma_q_over_store_spacing": (
                [float(cfg.query_sigma / s) for s in spacings] if spacings
                else (k0["sigma_q_over_store_spacing"] if k0 else None)),
            "hub_reference": ("0.19-0.37 on the repaired sizing, vs the ~1.07 the "
                              "retired sizing-set number implied"),
        },

        # the diagnostics, labelled
        "DIAGNOSTIC_m6": (None if m6 is None else {
            "label": "DIAGNOSTIC - cannot fail any gate (§A33.1)",
            "occupancy_precision_launch": m6["occupancy_precision_launch"],
            "occupancy_precision_settle": m6["occupancy_precision_settle"],
            "occupancy_dividend": m6["occupancy_dividend"],
            "distinct_wells": [m6["distinct_wells_launch"],
                               m6["distinct_wells_settle"]],
            "banked_C2W5": m6["banked_C2W5"],
            "baseline": ("⛔ scored against the BLANK STORE / raw launch geometry, "
                         "NEVER against F/N_a")}),
        "DIAGNOSTIC_launder_note": (
            "⛔ EVERY launder margin in this wave is DIAGNOSTIC and can never fail "
            "a leg (§A33.1). The launch-only launder, the settle-deleted launder "
            "and the byte ledger are reported beside every reading, all labelled."),
        "declared_NOT_RUNs": [
            "psi, the novelty head, the organization loss, every null arm, the "
            "organizer swap - not this spoke's",
            "K4 at full psi capacity - emitted as k4_full_psi_obligation",
            "M3 (per-feature G-ADDR) - well_lifecycle.py is C2W8-close's territory, "
            "READ-ONLY all wave",
            "M7 / M8 (curvature-shape term and its spectrum) - the loss-package "
            "spoke's",
            "V1 / V2 / V3 scores, OD, OD_min - VALUE, wave level",
            "k = 8 channels - a declared out-of-protocol diagnostic, never a score",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=_j))
    print(f"  -> {path}  (kills_all_passed = {kills_all_passed})", flush=True)
    return doc


def _phi_hash(phi) -> str:
    import hashlib
    b = (np.asarray(phi.codes, dtype=np.float32).tobytes()
         + np.asarray(phi.offsets, dtype=np.float32).tobytes())
    return hashlib.sha256(b).hexdigest()[:32]


# ==========================================================================
# the driver
# ==========================================================================
ALL_STAGES = ("k0", "m6", "width", "k6", "k7cap", "k1", "k2", "k3", "k4", "k5",
              "k8", "m4", "m5", "coverage", "freeze")


def run_c2w11_substrate(project: Optional[str] = None,
                        seeds: Sequence[int] = (0, 1, 2),
                        quick: bool = False,
                        out_dir: Optional[str] = None,
                        stages: Sequence[str] = ALL_STAGES,
                        selected_w_frac: Optional[float] = None
                        ) -> Dict[str, Any]:
    """⭐ The binding run order: **K0 -> M6 -> width -> K7-CAP/K6 -> K1 -> K2 ->
    K3 -> K4 -> K5 -> K8**, then the frozen interfaces.
    """
    out = Path(out_dir) if out_dir else Path("outputs/c2w11_substrate")
    out.mkdir(parents=True, exist_ok=True)
    cfg = c2w11_config()
    if quick:
        cfg = replace(cfg, n_wells=16, n_items=32, n_unseen=64, atoms_per_well=6,
                      address_steps=40, read_steps=80, write_steps=20)
        seeds = tuple(seeds)[:2]
    res: Dict[str, Any] = {"config": cfg.as_dict(),
                           "flags_vs_default": cfg.as_flag_table(),
                           "seeds": list(map(int, seeds)), "quick": bool(quick)}
    want = set(stages)

    if "k0" in want:
        print("\n=== K0 (no store; the cheapest kill in the wave) ===", flush=True)
        res["k0"] = stage_k0(cfg, seeds=seeds, out=out)

    if "width" in want:
        print("\n=== WIDTH SELECTION (repair (b): the store population) ===",
              flush=True)
        grid = WIDTH_GRID[:3] if quick else WIDTH_GRID
        sel_seeds = SELECTION_SEEDS[:1] if quick else SELECTION_SEEDS
        res["width"] = stage_width_selection(cfg, seeds=sel_seeds, grid=grid,
                                             out=out)
        selected_w_frac = res["width"]["selected_w_frac"]
    if selected_w_frac is not None:
        cfg = replace(cfg, atom_width_frac_spacing=float(selected_w_frac),
                      atom_width_selected_frac=float(selected_w_frac))
        res["selected_w_frac"] = float(selected_w_frac)

    if "m6" in want:
        print("\n=== M6 (DIAGNOSTIC) + M5 ===", flush=True)
        res["m6"] = stage_m6(cfg, seeds=seeds, out=out)

    if "k6" in want or "k7cap" in want:
        print("\n=== K6 / K7-CAP (before any fitted reader) ===", flush=True)
        res["k6_k7cap"] = stage_k6_k7cap(cfg, seeds=seeds, out=out)

    if "k1" in want:
        print("\n=== K1 / M2 (write admissibility, placing write) ===", flush=True)
        a_vals = (4, 12) if quick else (4, 12, 32)
        res["k1"] = stage_k1(cfg, seeds=seeds, a_values=a_vals, out=out)

    if "k2" in want:
        print("\n=== K2 (rule 4, both halves) ===", flush=True)
        res["k2"] = stage_k2(cfg, seeds=seeds,
                             m_values=(1, 8) if quick else (1, 2, 4, 6, 8, 12),
                             out=out)

    if want & {"k3", "k4", "k5"}:
        print("\n=== K3 / K4 / K5 ===", flush=True)
        res["k3_k4_k5"] = stage_k3_k4_k5(cfg, seeds=seeds, out=out)

    if "k8" in want:
        print("\n=== K8 (the K < N_a structural cell) ===", flush=True)
        res["k8"] = stage_k8(cfg, seeds=seeds,
                             n_items=8 if quick else 24, out=out)

    if "m4" in want:
        print("\n=== M4 (sharing / refresh) ===", flush=True)
        res["m4"] = stage_m4(cfg, seeds=seeds, out=out)

    if "coverage" in want:
        print("\n=== COVERAGE (the C2W9 trigger, spoke A's half) ===", flush=True)
        s_meas = None
        if res.get("k1", {}).get("cells"):
            reg = [c for c in res["k1"]["cells"]
                   if c["a"] == int(cfg.atoms_per_well)]
            if reg:
                s_meas = float(np.median([c["s_measured"] for c in reg]))
        res["coverage"] = stage_coverage(cfg, seeds=seeds, s_measured=s_meas,
                                         out=out)

    if "freeze" in want:
        print("\n=== FREEZE ===", flush=True)
        res["frozen"] = freeze_interfaces(
            cfg, res, out / "FROZEN-INTERFACES-C2W11.json")

    _dump(res, out / "c2w11_substrate_summary.json")
    return res
