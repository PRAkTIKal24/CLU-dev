"""The tier-ii **cardinality** harness (charter §A21, C2W7) — ``chlu exp-tierii-card``.

Iteration 2 of the read fix. Iteration 1 (:mod:`chlu.core.multiwell_read`,
``chlu exp-tierii-read``) repaired **addressing** and left **expressivity**
unrepaired with a named blocker: the read had no ``F``-commitment. This harness
scores :mod:`chlu.core.multiplicity_read` — multiplicity-as-counting-code,
overlap-as-importance weighting, the batch-level anti-collapse regularizer (built
OFF) and the launch-collapse monitor — against
``.claude/outputs/c2w7-read-cardinality/PREREG.md``, and runs the **organizer
swap** iff the pre-registered gate fires.

Stages (``--stages``):

``k0``           the store-free ``K0`` pre-condition for the **learned** head's
                 designed init (bar 0.90), with the out-of-class launch ceiling
                 reported beside it **with its ``(d, draws)`` noise model**.
``arms``         per seed: write -> **train the launch head** (staged: store
                 first) -> re-check ``K0`` on the TRAINED head -> organize the
                 store through the settle -> read -> fit the re-registered reader
                 class on SEEN -> score ``Q_unseen`` -> the **live** launch-only
                 launder (G1) -> the launch-collapse monitor.
``guards``       the four §A20.3(c) guards + the monitor, each with its designed
                 negative.
``regularizer``  the anti-collapse regularizer OFF (registered) vs ON — ⛔ both
                 states reported (doctrine §3.3: monitored first, regularized
                 second).
``levers``       learned ``p0`` on/off (reach lever) and the ``depth_ratio = 1``
                 **diagnostic axis** (never a claim cell).
``swap``         ⚖ the ORGANIZER SWAP — runs **iff the gate fires** on ``arms``.

⛔ Every tuning decision was taken on the SEEN split. ``Q_unseen`` is read in one
place (``_score``) and never by a fit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.factored_store import (CatTestConfig, FactoredStore, build_family,
                                      build_phi, chance_accuracy, effective_s,
                                      min_separation, occupancy, place_wells,
                                      write_wells)
from chlu.core.monitors import LaunchCollapseMonitor, MonitorContext
from chlu.core.multiplicity_read import (WEIGHT_MODES, MultiplicityConfig,
                                         MultiplicityHead, count_stats,
                                         fit_readers_mc, launch_collapse_stat,
                                         mc_hard_vs_soft_gradient, mc_ledger,
                                         mc_staging_gradient_probe,
                                         multiplicity_launder, multiplicity_read,
                                         organize_store_mc, read_codes,
                                         score_readers_mc,
                                         train_multiplicity_head)
from chlu.core.multiwell_read import assert_k_matched, s_effective
from chlu.experiments.exp_tierii_read import S_MEASURED_D8, registered_cfg

STAGES = ("k0", "arms", "guards", "regularizer", "levers", "swap")

#: ⛔ the pre-registered gate (PREREG §5). All four must clear before the swap.
R1_BAR = 0.02
S_EFF_BAND = (8.0, 16.0)


def registered_mc(**kw) -> MultiplicityConfig:
    """The registered operating point (PREREG §2); every deviation argued there."""
    base: Dict[str, Any] = {}
    base.update(kw)
    return MultiplicityConfig(**base)


# ==========================================================================
# stage k0 — the pre-condition, store-free, adjudicated BEFORE the arms
# ==========================================================================
def _launch_geometry(cfg, mc, head, fam, subsets, seed_offset: int):
    """Store-free: what the head's launches reach before any physics."""
    phi = build_phi(cfg)
    ind = fam.indicator(subsets, cfg.n_wells)
    c = np.asarray(phi.set_code(jnp.asarray(ind)))
    rng = np.random.default_rng(seed_offset)
    cn = jnp.asarray(c + cfg.query_sigma * rng.normal(size=c.shape), jnp.float32)
    q0, _p, _m, _g, conf, _b, n, f_hat = jax.vmap(head)(cn)
    lat = {"z": np.asarray(q0), "m": np.zeros((len(c), cfg.n_wells)),
           "n": np.asarray(n), "F_hat": np.asarray(f_hat),
           "conf": np.asarray(conf)}
    return lat


def stage_k0(cfg: CatTestConfig, mc: MultiplicityConfig, seeds: List[int]
             ) -> Dict[str, Any]:
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, cfg.target_ds * cfg.s_measured)
    head = MultiplicityHead(phi, anchors, cfg, mc)
    rows = []
    for seed in seeds:
        fam = build_family(cfg, seed)
        lat = _launch_geometry(cfg, mc, head, fam, fam.unseen, 1000 + seed)
        st = count_stats({**lat, "m": _launch_m(lat, cfg)}, anchors, fam.unseen,
                         cfg.f_subset)
        rows.append({"seed": seed, **st})
    k0 = float(np.mean([r["ge_F_distinct_raw"] for r in rows]))
    out = {"rows": rows, "bar": 0.90, "K0": k0,
           "verdict": "PASS" if k0 >= 0.90 else "FAIL"}
    for f in ("distinct_wells_raw", "occupancy_precision_raw", "coverage_raw",
              "exact_set_occupancy_raw", "F_hat_mean", "F_hat_within_half",
              "launch_topF_exact_set", "distinct_wells_gated",
              "exact_set_occupancy_gated", "gated_set_is_F"):
        out[f] = float(np.mean([r[f] for r in rows]))
    out["ceiling_out_of_class"] = ceiling_reference(cfg, seeds)
    return out


def _launch_m(lat, cfg) -> np.ndarray:
    """The launch's own importance code (multiplicity, no settle) — for K0 stats."""
    n = np.asarray(lat["n"])
    f = np.asarray(lat["F_hat"])[:, None]
    return f * n / (n.sum(1, keepdims=True) + 1e-9)


def ceiling_reference(cfg: CatTestConfig, seeds, n_draws: int = 1) -> Dict[str, Any]:
    """⛔ **OUT-OF-CLASS reference line, never a bar** — with its noise model.

    Reconciliation 1 of iteration 1: ``0.272`` (``P = 4`` mean of i.i.d. draws) and
    ``0.695`` (``d = 8``, one draw) are ceilings of *different launch models* and
    must always be quoted with ``(d, draws)``. This recomputes the ceiling at THIS
    task's noise model on the same family.
    """
    import itertools

    phi = build_phi(cfg)
    E = np.asarray(phi.codes)
    combos = np.array(list(itertools.combinations(range(cfg.n_wells),
                                                  cfg.f_subset)))
    lib = np.stack([E[c].sum(0) for c in combos])
    lib = cfg.ball_radius * lib / np.linalg.norm(lib, axis=1, keepdims=True)
    accs = []
    for seed in seeds:
        fam = build_family(cfg, seed)
        ind = fam.indicator(fam.unseen, cfg.n_wells)
        c = np.asarray(phi.set_code(jnp.asarray(ind)))
        rng = np.random.default_rng(7000 + seed)
        cn = c + cfg.query_sigma * rng.normal(
            size=(int(n_draws),) + c.shape).mean(0)
        pick = combos[((cn[:, None, :] - lib[None]) ** 2).sum(-1).argmin(1)]
        accs.append(float(np.mean([set(p.tolist()) == set(np.asarray(A).tolist())
                                   for p, A in zip(pick, fam.unseen, strict=True)])))
    return {"value": float(np.mean(accs)), "addr_dim": int(cfg.addr_dim),
            "draws": int(n_draws), "sigma_q": float(cfg.query_sigma),
            "decoder": f"exhaustive over C({cfg.n_wells},{cfg.f_subset})",
            "status": "OUT-OF-CLASS reference line — NEVER an arm bar"}


# ==========================================================================
# the per-seed cell
# ==========================================================================
def _build_seed(cfg: CatTestConfig, mc: MultiplicityConfig, seed: int):
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, cfg.target_ds * cfg.s_measured)
    fam = build_family(cfg, seed)
    depth_scale = np.where(np.arange(cfg.n_wells) % 2 == 0, cfg.depth_ratio, 1.0)
    blank = FactoredStore(cfg, anchors, jax.random.PRNGKey(seed))
    store, wi = write_wells(blank, cfg, anchors, fam.payloads,
                            jax.random.PRNGKey(1000 + seed),
                            depth_scale=depth_scale)
    head = MultiplicityHead(phi, anchors, cfg, mc)
    return phi, anchors, fam, blank, store, head, wi


def _latents(store, head, phi, cfg, mc, fam, key, **kw):
    ind_s = fam.indicator(fam.seen, cfg.n_wells)
    ind_u = fam.indicator(fam.unseen, cfg.n_wells)
    return (multiplicity_read(store, head, phi, cfg, mc, ind_s, key, **kw),
            multiplicity_read(store, head, phi, cfg, mc, ind_u,
                              jax.random.fold_in(key, 1), **kw))


def _score(lat_s, lat_u, fam, anchors, cfg, seed) -> Dict[str, Any]:
    """⛔ The ONE place ``Q_unseen`` is touched. Readers are fitted on SEEN only."""
    rd = fit_readers_mc(lat_s, fam.y_seen, anchors=anchors[:, : cfg.addr_dim],
                        well_payloads=fam.payloads, seed=seed)
    out = {"unseen": score_readers_mc(rd, lat_u, fam.y_unseen, fam.tol),
           "seen": score_readers_mc(rd, lat_s, fam.y_seen, fam.tol),
           "reader_params": {k: int(v.get("n_params", 0)) for k, v in rd.items()}}
    # the registered weighting/aggregation ablation — free (one read pass)
    sub = {k: v for k, v in rd.items() if k.startswith("count_")}
    var = {}
    for mode in list(WEIGHT_MODES) + ["noisy_or"]:
        kk = f"m__{mode}"
        if kk in lat_u:
            var[mode] = score_readers_mc(sub, lat_u, fam.y_unseen, fam.tol, kk)
    out["weight_variants_unseen"] = var
    return out


def _monitor(lat, cfg, mc) -> Dict[str, Any]:
    """The launch-collapse row, observed through the shipped registry protocol."""
    st = launch_collapse_stat(lat["m"], int(cfg.n_wells))
    m = np.asarray(lat["m"])
    p = np.abs(m) / (np.abs(m).sum(1, keepdims=True) + 1e-12)
    st["per_query_perplexity"] = float(np.exp(
        -(p * np.log(p + 1e-12)).sum(1)).mean())
    mon = LaunchCollapseMonitor(band_lo=float(mc.collapse_band_lo))
    r = mon.observe(MonitorContext(stage="arms", extras={"launch_usage": st}))
    return {"stat": st, "reading": r.as_dict()}


def stage_arms(cfg, mc, seeds, *, organize_steps: int = 60) -> Dict[str, Any]:
    rows = []
    for seed in seeds:
        t0 = time.time()
        phi, anchors, fam, blank, store, head0, wi = _build_seed(cfg, mc, seed)
        s_hat = float(np.nanmedian([
            effective_s(store.V, np.concatenate([anchors[j], fam.payloads[j]]),
                        confine=cfg.confine)["s"]
            for j in range(0, cfg.n_wells, 4)]))
        # --- G3's ordering: the store is WRITTEN first, the head trained second
        head, hi = train_multiplicity_head(store, head0, phi, cfg, mc, fam,
                                           jax.random.PRNGKey(300 + seed))
        # --- ⭐ K0 RE-CHECKED on the TRAINED head (a trained head can collapse it)
        lat_g = _launch_geometry(cfg, mc, head, fam, fam.unseen, 1000 + seed)
        k0_tr = count_stats({**lat_g, "m": _launch_m(lat_g, cfg)}, anchors,
                            fam.unseen, cfg.f_subset)
        # --- the PHYSICS organizer: the store trained THROUGH the settle
        phys, oi = organize_store_mc(store, head, phi, cfg, mc, fam,
                                     jax.random.PRNGKey(500 + seed),
                                     steps=int(organize_steps))
        lat_s, lat_u = _latents(phys, head, phi, cfg, mc, fam,
                                jax.random.PRNGKey(2000 + seed))
        sc_p = _score(lat_s, lat_u, fam, anchors, cfg, seed)
        st_u = count_stats(lat_u, anchors, fam.unseen, cfg.f_subset)
        st_s = count_stats(lat_s, anchors, fam.seen, cfg.f_subset)
        # --- guard 1: the LIVE launch-only launder on THIS cell's own head
        lau_s = multiplicity_launder(head, phi, cfg, mc,
                                     fam.indicator(fam.seen, cfg.n_wells),
                                     jax.random.PRNGKey(2000 + seed))
        lau_u = multiplicity_launder(head, phi, cfg, mc,
                                     fam.indicator(fam.unseen, cfg.n_wells),
                                     jax.random.fold_in(
                                         jax.random.PRNGKey(2000 + seed), 1))
        sc_l = _score(lau_s, lau_u, fam, anchors, cfg, seed)
        st_l = count_stats(lau_u, anchors, fam.unseen, cfg.f_subset)
        # --- ⭐ the DESIGNED (untrained) head on the SAME organized store: w20's
        # "free learning erases design" measured at the claim budget, not asserted
        des_s, des_u = _latents(phys, head0, phi, cfg, mc, fam,
                                jax.random.PRNGKey(2000 + seed))
        sc_d = _score(des_s, des_u, fam, anchors, cfg, seed)
        st_d = count_stats(des_u, anchors, fam.unseen, cfg.f_subset)

        readers = sorted(set(sc_p["unseen"]) & set(sc_l["unseen"]))
        g1 = {r: sc_p["unseen"][r] - sc_l["unseen"][r] for r in readers}
        n_par = int(sum(np.asarray(x).size for x in
                        jax.tree_util.tree_leaves(phys.V)))
        led = [mc_ledger("physics", cfg, mc, store_params=n_par, head=head,
                         phi_bytes=phi.n_bytes(), organizer="settle"),
               mc_ledger("launder_La", cfg, mc, store_params=0, head=head,
                         phi_bytes=phi.n_bytes(), organizer="none")]
        rows.append({
            "seed": seed, "s_measured": s_hat,
            "d_over_s": float(cfg.target_ds * cfg.s_measured / max(s_hat, 1e-9)),
            "min_sep": float(min_separation(anchors)),
            "write": wi, "head_train": hi, "organize": oi,
            "K0_trained_head": {k: k0_tr[k] for k in
                                ("ge_F_distinct_raw", "distinct_wells_raw",
                                 "F_hat_mean", "occupancy_precision_raw",
                                 "launch_topF_exact_set")},
            "chance": float(chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)),
            "tol": float(fam.tol),
            "physics": sc_p, "launder_La": sc_l,
            "physics_designed_head": sc_d,
            "stats_unseen_designed_head": st_d,
            "monitor_designed_head": _monitor(des_u, cfg, mc),
            "G1_read_minus_launder": g1, "G1_min": float(min(g1.values())),
            "G1_at_best_reader": float(
                g1[max(readers, key=lambda r: sc_p["unseen"][r])]),
            "stats_unseen_physics": st_u, "stats_seen_physics": st_s,
            "stats_unseen_launder": st_l,
            "monitor_physics": _monitor(lat_u, cfg, mc),
            "monitor_launder": _monitor(lau_u, cfg, mc),
            "S_eff_physics": s_effective(st_u["n_wells_ever_occupied"], cfg),
            "S_eff_physics_gated": s_effective(
                st_u.get("n_wells_ever_occupied_gated", 0), cfg),
            "ledgers": led, "k_matched": assert_k_matched(led),
            "wall_s": float(time.time() - t0)})
        print(f"  [arms] seed {seed}: best {max(sc_p['unseen'].values()):.4f} "
              f"launder {max(sc_l['unseen'].values()):.4f} "
              f"G1min {rows[-1]['G1_min']:+.4f} "
              f"R1 {st_u['exact_set_occupancy_gated']:.4f} "
              f"F {st_u['F_hat_mean']:.2f} "
              f"designed {max(sc_d['unseen'].values()):.4f} "
              f"S_marg {rows[-1]['monitor_physics']['stat']['marginal_perplexity']:.1f}"
              f" ({rows[-1]['wall_s']:.0f}s)", flush=True)
    return {"rows": rows, **_aggregate(rows)}


def _ms(vals) -> Dict[str, float]:
    v = np.asarray(vals, float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "sd": sd,
            "se": float(sd / max(np.sqrt(len(v)), 1)),
            "two_se": float(2 * sd / max(np.sqrt(len(v)), 1))}


def _aggregate(rows) -> Dict[str, Any]:
    readers = sorted(rows[0]["G1_read_minus_launder"])
    out: Dict[str, Any] = {
        "n_seeds": len(rows),
        "G1_min": _ms([r["G1_min"] for r in rows]),
        "G1_at_best_reader": _ms([r["G1_at_best_reader"] for r in rows]),
        "G1_per_reader": {r: _ms([x["G1_read_minus_launder"][r] for x in rows])
                          for r in readers},
        "physics_per_reader": {r: _ms([x["physics"]["unseen"][r] for x in rows])
                               for r in readers},
        "launder_per_reader": {r: _ms([x["launder_La"]["unseen"][r] for x in rows])
                               for r in readers},
        "physics_best": _ms([max(r["physics"]["unseen"].values()) for r in rows]),
        "designed_head_best": _ms([max(r["physics_designed_head"]["unseen"].values())
                                   for r in rows]),
        "designed_head_per_reader": {
            r: _ms([x["physics_designed_head"]["unseen"][r] for x in rows])
            for r in readers},
        "launder_best": _ms([max(r["launder_La"]["unseen"].values())
                             for r in rows]),
        "chance": _ms([r["chance"] for r in rows])}
    for f in ("exact_set_occupancy_gated", "exact_set_occupancy_raw",
              "distinct_wells_raw", "distinct_wells_gated", "gated_set_is_F",
              "occupancy_precision_raw", "occupancy_precision_gated",
              "coverage_gated", "ge_F_distinct_raw", "F_hat_mean",
              "launch_topF_exact_set", "marginal_perplexity"):
        out[f"physics_{f}"] = _ms([r["stats_unseen_physics"][f] for r in rows])
        out[f"launder_{f}"] = _ms([r["stats_unseen_launder"][f] for r in rows])
        out[f"designed_head_{f}"] = _ms([r["stats_unseen_designed_head"][f]
                                         for r in rows])
    out["S_eff"] = _ms([r["S_eff_physics"] for r in rows])
    out["S_eff_gated"] = _ms([r["S_eff_physics_gated"] for r in rows])
    out["K0_trained_head"] = _ms([r["K0_trained_head"]["ge_F_distinct_raw"]
                                  for r in rows])
    var_modes = sorted(rows[0]["physics"]["weight_variants_unseen"])
    out["weight_variants"] = {
        mode: {rdr: {"physics": _ms([r["physics"][
            "weight_variants_unseen"][mode][rdr] for r in rows]),
            "launder": _ms([r["launder_La"]["weight_variants_unseen"][mode][rdr]
                            for r in rows])}
            for rdr in ("count_identity", "count_table")}
        for mode in var_modes}
    out["monitor_tripped_seeds"] = [r["seed"] for r in rows
                                    if r["monitor_physics"]["reading"]["tripped"]]
    out["gate"] = adjudicate_gate(out)
    return out


def adjudicate_gate(agg: Dict[str, Any]) -> Dict[str, Any]:
    """⚖ The pre-registered gate, adjudicated MECHANICALLY (PREREG §5).

    ⛔ Reported, never interpreted here: the tier-ii **verdict** is the Advisor's,
    against raw artifacts.
    """
    r1 = agg["physics_exact_set_occupancy_gated"]
    g1 = agg["G1_min"]
    seff = agg["S_eff"]["mean"]
    checks = {
        "R1_clears": {"stat": "exact_set_occupancy_gated (physics, unseen)",
                      "value": r1["mean"], "two_se": r1["two_se"],
                      "bar": R1_BAR, "clears": bool(r1["mean"] - r1["two_se"]
                                                    > R1_BAR)},
        "G1_clears": {"stat": "read - live launder (worst reader)",
                      "value": g1["mean"], "two_se": g1["two_se"], "bar": 0.0,
                      "clears": bool(g1["mean"] - g1["two_se"] > 0.0)},
        "S_eff_in_band": {"stat": "S_eff", "value": seff, "band": S_EFF_BAND,
                          "clears": bool(S_EFF_BAND[0] <= seff <= S_EFF_BAND[1])},
        "monitor_quiet": {"stat": "launch_collapse trips",
                          "value": len(agg["monitor_tripped_seeds"]),
                          "clears": not agg["monitor_tripped_seeds"]},
    }
    fires = all(c["clears"] for c in checks.values())
    return {"checks": checks, "SWAP_RUNS": bool(fires),
            "verdict": ("GATE FIRES => the organizer swap runs"
                        if fires else
                        "GATE FAILS => NO swap; negatives reported as negatives"),
            "S_eff_label": ("in band [8, 16]"
                            if checks["S_eff_in_band"]["clears"] else "COLLAPSED")}


# ==========================================================================
# stage guards — each guard must FIRE on a designed negative
# ==========================================================================
def stage_guards(cfg, mc, seeds) -> Dict[str, Any]:
    seed = seeds[0]
    phi, anchors, fam, blank, store, head, _ = _build_seed(cfg, mc, seed)
    key = jax.random.PRNGKey(700 + seed)
    g2 = mc_hard_vs_soft_gradient(store, head, phi, cfg, mc, fam, key, n=8)
    g3 = mc_staging_gradient_probe(blank, store, head, phi, cfg, mc, fam, key, n=8)
    cfg_hist = CatTestConfig(**{**cfg.as_dict(), "atom_local_radius": 0.0,
                               "atom_payload_init_radius": 0.0})
    hist_blank = FactoredStore(cfg_hist, anchors, jax.random.PRNGKey(seed))
    g3b = mc_staging_gradient_probe(hist_blank, store, head, phi, cfg, mc, fam,
                                    key, n=8)
    # ⛔ G1's designed negative: a ZERO-step read must equal the launder bitwise
    cfg0 = CatTestConfig(**{**cfg.as_dict(), "address_steps": 0, "read_steps": 0})
    ind_u = fam.indicator(fam.unseen[:64], cfg.n_wells)
    z0 = multiplicity_read(store, head, phi, cfg0, mc, ind_u,
                           jax.random.PRNGKey(11))
    zl = multiplicity_launder(head, phi, cfg0, mc, ind_u, jax.random.PRNGKey(11))
    g1neg = {"max_abs_diff_z": float(np.max(np.abs(z0["z"] - zl["z"]))),
             "max_abs_diff_m": float(np.max(np.abs(z0["m"] - zl["m"]))),
             "bit_identical": bool(np.array_equal(z0["z"], zl["z"]))}
    # guard 4: k is capacity — 2k at the same bytes must move the score
    g4 = {}
    for kk in (mc.k_particles, 2 * mc.k_particles):
        mck = registered_mc(**{**mc.as_dict(), "k_particles": int(kk)})
        h = MultiplicityHead(phi, anchors, cfg, mck)
        ls, lu = _latents(store, h, phi, cfg, mck, fam,
                          jax.random.PRNGKey(2000 + seed))
        sc = _score(ls, lu, fam, anchors, cfg, seed)
        st = count_stats(lu, anchors, fam.unseen, cfg.f_subset)
        g4[f"k={kk}"] = {"unseen": sc["unseen"],
                         "best": max(sc["unseen"].values()),
                         "coverage_raw": st["coverage_raw"],
                         "distinct_raw": st["distinct_wells_raw"],
                         "ledger": mc_ledger("k_probe", cfg, mck, store_params=0,
                                             head=h, phi_bytes=phi.n_bytes())}
    try:
        assert_k_matched([g4[f"k={mc.k_particles}"]["ledger"],
                          g4[f"k={2 * mc.k_particles}"]["ledger"]])
        g4["mismatch_raises"] = False
    except ValueError as e:
        g4["mismatch_raises"] = True
        g4["mismatch_message"] = str(e)
    # ⭐ the launch-collapse monitor's DESIGNED NEGATIVE (N74: a guard that cannot
    # fire is vacuous): an input-independent allocation.
    lat_u = multiplicity_read(store, head, phi, cfg, mc,
                              fam.indicator(fam.unseen, cfg.n_wells),
                              jax.random.PRNGKey(2000 + seed))
    live = _monitor(lat_u, cfg, mc)
    collapsed = np.zeros_like(np.asarray(lat_u["m"]))
    collapsed[:, : cfg.f_subset] = 1.0  # every query -> the same F wells
    neg = _monitor({"m": collapsed}, cfg, mc)
    return {"G2_soft_vs_hard": g2, "G3_staging_designed_blank": g3,
            "G3_staging_historical_blank": g3b,
            "G1_zero_step_equals_launder": g1neg,
            "G4_k_is_capacity": g4,
            "monitor_live": live, "monitor_designed_negative": neg}


# ==========================================================================
# stage regularizer — OFF (registered) vs ON; ⛔ BOTH states reported
# ==========================================================================
def stage_regularizer(cfg, mc, seeds, *, lam_on: float = 1.0) -> Dict[str, Any]:
    out = {}
    for name, lam in (("off_registered", 0.0), ("on", float(lam_on))):
        rows = []
        for seed in seeds:
            m2 = registered_mc(**{**mc.as_dict(), "lambda_anticollapse": lam})
            phi, anchors, fam, _b, store, head0, _ = _build_seed(cfg, m2, seed)
            head, hi = train_multiplicity_head(store, head0, phi, cfg, m2, fam,
                                               jax.random.PRNGKey(300 + seed))
            ls, lu = _latents(store, head, phi, cfg, m2, fam,
                              jax.random.PRNGKey(2000 + seed))
            sc = _score(ls, lu, fam, anchors, cfg, seed)
            st = count_stats(lu, anchors, fam.unseen, cfg.f_subset)
            mo = _monitor(lu, cfg, m2)
            rows.append({"seed": seed, "best_unseen": max(sc["unseen"].values()),
                         "count_identity": sc["unseen"]["count_identity"],
                         "marginal_perplexity": st["marginal_perplexity"],
                         "marginal_max": st["marginal_max"],
                         "per_query_perplexity":
                             mo["stat"]["per_query_perplexity"],
                         "exact_occ_gated": st["exact_set_occupancy_gated"],
                         "monitor_tripped": mo["reading"]["tripped"],
                         "head_loss_last": hi["head_loss_last"]})
        out[name] = {"lambda": lam, "rows": rows, **{
            f: float(np.mean([r[f] for r in rows])) for f in
            ("best_unseen", "count_identity", "marginal_perplexity",
             "marginal_max", "per_query_perplexity", "exact_occ_gated")}}
        print(f"  [regularizer] {name} (lambda={lam}): S_marg "
              f"{out[name]['marginal_perplexity']:.2f} best "
              f"{out[name]['best_unseen']:.4f}", flush=True)
    return out


# ==========================================================================
# stage levers — (e) learned p0 (reach only) + the depth_ratio DIAGNOSTIC axis
# ==========================================================================
def stage_levers(cfg, mc, seeds) -> Dict[str, Any]:
    out = {}
    for name, ckw, mkw in (("p0_on_registered", {}, {}),
                           ("p0_off", {}, dict(learned_p0=False)),
                           ("depth_ratio_1_DIAGNOSTIC", dict(depth_ratio=1.0), {})):
        rows = []
        for seed in seeds:
            c = CatTestConfig(**{**cfg.as_dict(), **ckw})
            m2 = registered_mc(**{**mc.as_dict(), **mkw})
            phi, anchors, fam, _b, store, head, _ = _build_seed(c, m2, seed)
            ls, lu = _latents(store, head, phi, c, m2, fam,
                              jax.random.PRNGKey(2000 + seed))
            sc = _score(ls, lu, fam, anchors, c, seed)
            st = count_stats(lu, anchors, fam.unseen, c.f_subset)
            occ = occupancy(lu["z"], anchors[:, : c.addr_dim])
            inA = np.array([np.isin(o, A) for o, A in
                            zip(occ, fam.unseen, strict=True)])
            rows.append({"seed": seed, "best_unseen": max(sc["unseen"].values()),
                         "count_identity": sc["unseen"]["count_identity"],
                         "distinct_raw": st["distinct_wells_raw"],
                         "precision_raw": st["occupancy_precision_raw"],
                         "precision_gated": st["occupancy_precision_gated"],
                         "coverage_gated": st["coverage_gated"],
                         "exact_occ_gated": st["exact_set_occupancy_gated"],
                         "w_in_A": float(lu["w"][inA].mean()),
                         "w_out_A": float(lu["w"][~inA].mean())})
        out[name] = {"rows": rows, **{
            f: float(np.mean([r[f] for r in rows])) for f in
            ("best_unseen", "count_identity", "distinct_raw", "precision_raw",
             "precision_gated", "coverage_gated", "exact_occ_gated", "w_in_A",
             "w_out_A")}}
        print(f"  [levers] {name}: w_inA {out[name]['w_in_A']:.3f} "
              f"prec_gated {out[name]['precision_gated']:.3f} "
              f"best {out[name]['best_unseen']:.4f}", flush=True)
    return out


# ==========================================================================
# stage swap — ⚖ the ORGANIZER SWAP, iff the gate fires
# ==========================================================================
def stage_swap(cfg, mc, seeds, *, organize_steps: int = 60) -> Dict[str, Any]:
    """The tier-ii control (§A13): same store parameterisation, same objective,
    **static** read, no dynamics — against the physics organizer, with the SAME
    trained launch head FROZEN on both arms (bit-identical launches).
    """
    from chlu.core.multiwell_read import soft_occupancy
    from chlu.core.null_arms import n1_apply, n1_gradient_placed

    rows = []
    for seed in seeds:
        phi, anchors, fam, _b, store, head0, _ = _build_seed(cfg, mc, seed)
        head, _hi = train_multiplicity_head(store, head0, phi, cfg, mc, fam,
                                            jax.random.PRNGKey(300 + seed))
        phys, oi = organize_store_mc(store, head, phi, cfg, mc, fam,
                                     jax.random.PRNGKey(500 + seed),
                                     steps=int(organize_steps))
        lat_s, lat_u = _latents(phys, head, phi, cfg, mc, fam,
                                jax.random.PRNGKey(2000 + seed))
        sc_p = _score(lat_s, lat_u, fam, anchors, cfg, seed)
        c1 = CatTestConfig(**{**cfg.as_dict(), "n_particles": int(mc.k_particles)})
        fit = n1_gradient_placed(c1, fam, anchors, lat_s["q0"], fam.y_seen,
                                 lr=1e-3, tau=1.0, init="written", steps=400,
                                 seed=seed, atoms_per_well=cfg.atoms_per_well)
        n1 = []
        for lat in (lat_s, lat_u):
            z = n1_apply(fit, lat["q0"])
            s = soft_occupancy(jnp.asarray(z[..., : cfg.addr_dim]),
                               jnp.asarray(anchors[:, : cfg.addr_dim],
                                           jnp.float32), mc.occ_tau)
            w = jnp.clip(jnp.linalg.norm(jnp.asarray(z[..., cfg.addr_dim:]),
                                         axis=-1) / mc.payload_ref, 0, 1)
            codes = read_codes(s, w, jnp.asarray(lat["conf"]),
                                jnp.asarray(lat["F_hat"]), mc)
            n1.append({**{k: np.asarray(v) for k, v in codes.items()},
                       "z": np.asarray(z), "q0": lat["q0"], "conf": lat["conf"],
                       "w": np.asarray(w), "n": lat["n"], "F_hat": lat["F_hat"]})
        sc_n = _score(n1[0], n1[1], fam, anchors, cfg, seed)
        readers = sorted(set(sc_p["unseen"]) & set(sc_n["unseen"]))
        od = {r: sc_p["unseen"][r] - sc_n["unseen"][r] for r in readers}
        rows.append({"seed": seed, "physics": sc_p, "null_N1prime": sc_n,
                     "organize": oi, "OD": od, "OD_min": float(min(od.values())),
                     "null_params": int(fit["ledger"]["n_params"])})
        print(f"  [swap] seed {seed}: OD_min {rows[-1]['OD_min']:+.4f}", flush=True)
    readers = sorted(rows[0]["OD"])
    od_min = _ms([r["OD_min"] for r in rows])
    m, t = od_min["mean"], od_min["two_se"]
    return {"rows": rows, "OD_min": od_min,
            "OD_per_reader": {r: _ms([x["OD"][r] for x in rows])
                              for r in readers},
            "F1": ("CLEARS" if m - t > 0.05 else
                   "FIRES" if m + t < 0.05 and abs(m) > 0.05 else "TIE"),
            "note": "⛔ REPORTED, NOT ADJUDICATED — the tier-ii verdict is the "
                    "Advisor's, against raw artifacts."}


# ==========================================================================
# entry point
# ==========================================================================
def run_tierii_cardinality(*, seeds=(0, 1, 2, 3, 4), stages=STAGES, out_dir=None,
                           quick: bool = False, organize_steps: int = 60,
                           k_particles: int = 12, force_swap: bool = False,
                           lam_on: float = 1.0) -> Dict[str, Any]:
    if quick:
        cfg = registered_cfg(n_items=32, n_unseen=32, write_steps=30,
                             address_steps=40, read_steps=60)
        mc = registered_mc(k_particles=int(k_particles), head_steps=2,
                           ista_steps=40, head_settle_address=20,
                           head_settle_read=30)
        seeds, organize_steps = list(seeds)[:1], 2
    else:
        cfg = registered_cfg()
        mc = registered_mc(k_particles=int(k_particles))
    res: Dict[str, Any] = {
        "config": cfg.as_dict(), "mc_config": mc.as_dict(),
        "cfg_non_default": cfg.as_flag_table(), "mc_non_default": mc.as_flag_table(),
        "s_measured_registered": S_MEASURED_D8,
        "seeds": list(seeds), "quick": bool(quick),
        "organize_steps": int(organize_steps)}
    t0 = time.time()
    if "k0" in stages:
        print("[stage] k0", flush=True)
        res["k0"] = stage_k0(cfg, mc, list(seeds))
        print(f"  K0(designed init) = {res['k0']['K0']:.4f} => "
              f"{res['k0']['verdict']}  |  F_hat {res['k0']['F_hat_mean']:.2f} "
              f"| topF-exact {res['k0']['launch_topF_exact_set']:.4f} "
              f"| ceiling(out-of-class) "
              f"{res['k0']['ceiling_out_of_class']['value']:.3f}", flush=True)
        if res["k0"]["verdict"] != "PASS":
            print("  ⛔ K0 FAILS — the protocol is redesigned before anything "
                  "else runs (standing rule).", flush=True)
    if "arms" in stages:
        print("[stage] arms", flush=True)
        res["arms"] = stage_arms(cfg, mc, list(seeds),
                                 organize_steps=organize_steps)
        print(f"  ⚖ GATE: {res['arms']['gate']['verdict']}", flush=True)
    if "guards" in stages:
        print("[stage] guards", flush=True)
        res["guards"] = stage_guards(cfg, mc, list(seeds))
    if "regularizer" in stages:
        print("[stage] regularizer", flush=True)
        res["regularizer"] = stage_regularizer(cfg, mc, list(seeds), lam_on=lam_on)
    if "levers" in stages:
        print("[stage] levers", flush=True)
        res["levers"] = stage_levers(cfg, mc, list(seeds))
    if "swap" in stages:
        fires = bool(res.get("arms", {}).get("gate", {}).get("SWAP_RUNS", False))
        if fires or force_swap:
            print(f"[stage] swap (gate_fires={fires}, forced={force_swap})",
                  flush=True)
            res["swap"] = stage_swap(cfg, mc, list(seeds),
                                     organize_steps=organize_steps)
            res["swap"]["ran_because"] = ("gate fired" if fires
                                          else "FORCED (labelled diagnostic, "
                                               "NOT a claim cell)")
        else:
            res["swap"] = {"ran": False,
                           "why": "⛔ the pre-registered gate did not fire; per "
                                  "the task file NO swap is run and the "
                                  "negatives are reported as negatives."}
            print("[stage] swap SKIPPED — the gate did not fire.", flush=True)
    res["wall_s"] = float(time.time() - t0)
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tierii_cardinality_summary.json").write_text(
            json.dumps(res, indent=1, default=float))
        print(f"[out] {p / 'tierii_cardinality_summary.json'}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="tier-ii multiplicity read (cardinality iteration)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--organize-steps", type=int, default=60)
    ap.add_argument("--k-particles", type=int, default=12)
    ap.add_argument("--lam-on", type=float, default=1.0,
                    help="the anti-collapse coefficient of the ON arm")
    ap.add_argument("--force-swap", action="store_true",
                    help="run the swap even if the gate fails (LABELLED "
                         "diagnostic, never a claim cell)")
    a = ap.parse_args(argv)
    run_tierii_cardinality(seeds=a.seeds, stages=a.stages, out_dir=a.out_dir,
                           quick=a.quick, organize_steps=a.organize_steps,
                           k_particles=a.k_particles, force_swap=a.force_swap,
                           lam_on=a.lam_on)


if __name__ == "__main__":
    main()
