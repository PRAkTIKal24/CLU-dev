"""The tier-ii **read-fix** harness (charter §A20.3) — ``chlu exp-tierii-read``.

Scores the multi-well read protocol of :mod:`chlu.core.multiwell_read` against
``.claude/outputs/tierii-read-fix/PREREG.md`` (filed before this file existed),
and re-runs the **organizer swap** — tier ii's own control (§A13) — against it.

Stages (``--stages``):

``k0``          the store-free K0 pre-condition (bar 0.90) — the launch geometry
                of the REFUTED ``P = 4`` protocol beside the new head's.
``arms``        per seed: write -> organize (physics: **through the settle**;
                null N1': the same parameters, a **static** softmax read) ->
                read -> fit the reader class on SEEN -> score ``Q_unseen`` ->
                ``OD(R)`` per reader and ``OD_min``.
``guards``      the four §A20.3(c) guards, each with its designed negative.
``consolidate`` (d) consolidate-to-budget + trash-region pruning + ``S_eff``.
``levers``      (e) the learned-``p0`` ablation and the depth-heterogeneity
                ablation (the named suspect for the selection failure).

⛔ Every tuning decision in this harness was taken on the SEEN split. ``Q_unseen``
is read in exactly one place (``_score_arm``) and never by a fit.
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
                                      build_phi, effective_s, min_separation,
                                      occupancy, place_wells, write_wells)
from chlu.core.multiwell_read import (LaunchHead, MultiWellReadConfig,
                                      assert_k_matched, consolidate_wells,
                                      find_wells, fit_readers_mw,
                                      hard_vs_soft_gradient, launch_only_launder,
                                      multiwell_read, mwr_ledger,
                                      organize_store_mw, read_stats,
                                      s_effective, score_readers_mw,
                                      staging_gradient_probe,
                                      trash_field)

STAGES = ("k0", "arms", "guards", "consolidate", "levers")

# ⭐ the registered operating point (PREREG §0/§1; every deviation argued there)
# ⭐ measured on the WRITTEN store at d = 8, m = 8, a = 32, payload_radius = 0.5,
# confinement-subtracted (§7.28's program-wide ruler). It is a FIXED POINT of the
# placement loop: sep = 2.7 * s re-measures the same s (0.2879 at sep = 0.7773
# and at sep = 0.7290), so d/s = 2.700 exactly — inside the registered
# soft-certificate band [2.5, 2.9]. ⚠ The first full run used 0.3611 (measured at
# payload_radius = 1.0) and landed at d/s = 3.386, OUT of band; that run is
# retained as a labelled off-band diagnostic and is not a claim cell.
S_MEASURED_D8 = 0.2879


def registered_cfg(**kw) -> CatTestConfig:
    base = dict(atoms_per_well=32, addr_dim=8, s_measured=S_MEASURED_D8,
                payload_radius=0.5, atom_payload_init_radius=0.5, n_unseen=256)
    base.update(kw)
    return CatTestConfig(**base)


def registered_mw(**kw) -> MultiWellReadConfig:
    # ⚠ the TRAINING settle budget (60 + 120) is reduced from the SCORING read
    # budget (400 + 800) and is declared beside every number it produces
    # (C2W4 standing read-budget-scoping rule).
    base = dict(payload_ref=0.5, conf_w=0.05, head_batch=16,
                head_settle_address=60, head_settle_read=120)
    base.update(kw)
    return MultiWellReadConfig(**base)


# ==========================================================================
# stage k0 — the pre-condition, store-free (⛔ adjudicated BEFORE the arms)
# ==========================================================================
def stage_k0(cfg: CatTestConfig, mw: MultiWellReadConfig, seeds: List[int]
             ) -> Dict[str, Any]:
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, cfg.target_ds * cfg.s_measured)
    head = LaunchHead(phi, anchors, cfg, mw)
    rows = []
    for seed in seeds:
        fam = build_family(cfg, seed)
        ind = fam.indicator(fam.unseen, cfg.n_wells)
        c = np.asarray(phi.set_code(jnp.asarray(ind)))
        rng = np.random.default_rng(1000 + seed)
        cn = c + cfg.query_sigma * rng.normal(size=c.shape)
        q0 = np.asarray(jax.vmap(head)(jnp.asarray(cn, jnp.float32))[0])
        new = read_stats({"z": q0}, anchors, fam.unseen, cfg.f_subset)
        # the REFUTED protocol, on the same family (the instrument's own control)
        old = c[:, None, :] + np.asarray(phi.offsets)[None, :, :] \
            + cfg.query_sigma * rng.normal(
                size=(len(c), cfg.n_particles, cfg.addr_dim))
        ref = read_stats({"z": old}, anchors, fam.unseen, cfg.f_subset)
        # ... and the REFUTED protocol at its OWN registered d = 4, which is the
        # cell `orgdiv-null-arms` §3 published (2.202 distinct / 0.050 >= F /
        # 0.4106 precision / 0.0000 exact-set). If this instrument cannot
        # reproduce the number it is refuting, it measures nothing.
        c4 = CatTestConfig(**{**cfg.as_dict(), "addr_dim": 4, "s_measured": 0.318})
        phi4 = build_phi(c4)
        anc4 = place_wells(phi4, c4, c4.target_ds * c4.s_measured)
        fam4 = build_family(c4, seed)
        cc4 = np.asarray(phi4.set_code(jnp.asarray(
            fam4.indicator(fam4.unseen, c4.n_wells))))
        old4 = (cc4[:, None, :] + np.asarray(phi4.offsets)[None, :, :]
                + c4.query_sigma * np.random.default_rng(2000 + seed).normal(
                    size=(len(cc4), c4.n_particles, 4)))
        ref4 = read_stats({"z": old4}, anc4, fam4.unseen, c4.f_subset)
        rows.append({"seed": seed, "new": new, "refuted_P4_at_new_d": ref,
                     "refuted_P4_at_d4_published_cell": ref4})
    def agg(which, field):
        return float(np.mean([r[which][field] for r in rows]))
    return {"rows": rows,
            "K0_new": agg("new", "ge_F_distinct_raw"),
            "K0_refuted_P4_at_new_d": agg("refuted_P4_at_new_d", "ge_F_distinct_raw"),
            "K0_refuted_P4_at_d4": agg("refuted_P4_at_d4_published_cell",
                                       "ge_F_distinct_raw"),
            "distinct_new": agg("new", "distinct_wells_raw"),
            "distinct_refuted_P4_at_new_d": agg("refuted_P4_at_new_d",
                                                "distinct_wells_raw"),
            "distinct_refuted_P4_at_d4": agg("refuted_P4_at_d4_published_cell",
                                             "distinct_wells_raw"),
            "precision_new": agg("new", "occupancy_precision_raw"),
            "precision_refuted_P4_at_new_d": agg("refuted_P4_at_new_d",
                                                 "occupancy_precision_raw"),
            "precision_refuted_P4_at_d4": agg("refuted_P4_at_d4_published_cell",
                                              "occupancy_precision_raw"),
            "exact_occ_refuted_P4_at_d4": agg("refuted_P4_at_d4_published_cell",
                                              "exact_set_occupancy_raw"),
            "coverage_new": agg("new", "coverage_raw"),
            "bar": 0.90,
            "verdict": "PASS" if agg("new", "ge_F_distinct_raw") >= 0.90 else "FAIL"}


# ==========================================================================
# the per-seed cell
# ==========================================================================
def _build_seed(cfg: CatTestConfig, mw: MultiWellReadConfig, seed: int):
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, cfg.target_ds * cfg.s_measured)
    fam = build_family(cfg, seed)
    depth_scale = np.where(np.arange(cfg.n_wells) % 2 == 0, cfg.depth_ratio, 1.0)
    blank = FactoredStore(cfg, anchors, jax.random.PRNGKey(seed))
    store, wi = write_wells(blank, cfg, anchors, fam.payloads,
                            jax.random.PRNGKey(1000 + seed), depth_scale=depth_scale)
    head = LaunchHead(phi, anchors, cfg, mw)
    return phi, anchors, fam, blank, store, head, wi


def _latents(store, head, phi, cfg, mw, fam, key, **kw):
    ind_s = fam.indicator(fam.seen, cfg.n_wells)
    ind_u = fam.indicator(fam.unseen, cfg.n_wells)
    return (multiwell_read(store, head, phi, cfg, mw, ind_s, key, **kw),
            multiwell_read(store, head, phi, cfg, mw, ind_u,
                           jax.random.fold_in(key, 1), **kw))


def _score(lat_s, lat_u, fam, anchors, cfg, seed) -> Dict[str, float]:
    """⛔ The ONE place ``Q_unseen`` is touched. Readers are fitted on SEEN only."""
    rd = fit_readers_mw(lat_s, fam.y_seen, anchors=anchors[:, : cfg.addr_dim],
                        well_payloads=fam.payloads, seed=seed)
    return {"unseen": score_readers_mw(rd, lat_u, fam.y_unseen, fam.tol),
            "seen": score_readers_mw(rd, lat_s, fam.y_seen, fam.tol),
            "reader_params": {k: int(v.get("n_params", 0)) for k, v in rd.items()}}


def _n1_latent(cfg, mw, fam, anchors, lat_s, lat_u, seed, phi):
    """The ORGANIZER SWAP: the same parameters, a **static** read, no dynamics.

    Uses ``null_arms.n1_gradient_placed``'s public API at the SAME ``k`` and on
    the **bit-identical launch points** the physics arm settled from.
    """
    from chlu.core.null_arms import n1_gradient_placed, n1_apply

    c = CatTestConfig(**{**cfg.as_dict(), "n_particles": int(mw.k_particles)})
    fit = n1_gradient_placed(c, fam, anchors, lat_s["q0"], fam.y_seen,
                             lr=1e-3, tau=1.0, init="written", steps=400,
                             seed=seed, atoms_per_well=cfg.atoms_per_well)
    out = []
    for lat in (lat_s, lat_u):
        z = n1_apply(fit, lat["q0"])
        pi = np.asarray(jax.nn.softmax(
            -((jnp.asarray(z)[:, :, None, : cfg.addr_dim]
               - jnp.asarray(anchors[:, : cfg.addr_dim])[None, None]) ** 2).sum(-1)
            / (2.0 * mw.occ_tau ** 2), axis=-1))
        w = np.clip(np.linalg.norm(z[..., cfg.addr_dim:], axis=-1) / mw.payload_ref,
                    0, 1)
        pi = 1.0 - np.prod(1.0 - np.clip(pi * w[..., None], 0, 1 - 1e-6), axis=1)
        out.append({"z": np.asarray(z), "pi": pi, "q0": lat["q0"],
                    "conf": lat["conf"], "w": w})
    return out[0], out[1], fit


def stage_arms(cfg, mw, seeds, *, organize_steps: int = 60) -> Dict[str, Any]:
    rows = []
    for seed in seeds:
        t0 = time.time()
        phi, anchors, fam, blank, store, head, wi = _build_seed(cfg, mw, seed)
        s_hat = float(np.nanmedian([
            effective_s(store.V, np.concatenate([anchors[j], fam.payloads[j]]),
                        confine=cfg.confine)["s"] for j in range(0, cfg.n_wells, 4)]))
        # --- the PHYSICS organizer: through the settle
        phys, oi = organize_store_mw(store, head, phi, cfg, mw, fam,
                                     jax.random.PRNGKey(500 + seed),
                                     steps=int(organize_steps))
        lat_s, lat_u = _latents(phys, head, phi, cfg, mw, fam,
                                jax.random.PRNGKey(2000 + seed))
        sc_p = _score(lat_s, lat_u, fam, anchors, cfg, seed)
        st_u = read_stats(lat_u, anchors, fam.unseen, cfg.f_subset)
        st_s = read_stats(lat_s, anchors, fam.seen, cfg.f_subset)
        # --- the ORGANIZER SWAP (null): same params, static read, no dynamics
        n1_s, n1_u, fit = _n1_latent(cfg, mw, fam, anchors, lat_s, lat_u, seed, phi)
        sc_n = _score(n1_s, n1_u, fam, anchors, cfg, seed)
        st_n = read_stats(n1_u, anchors, fam.unseen, cfg.f_subset)
        # --- guard 1: the live-recomputed launch-only launder
        lau_s = launch_only_launder(head, phi, cfg, mw,
                                    fam.indicator(fam.seen, cfg.n_wells),
                                    jax.random.PRNGKey(2000 + seed))
        lau_u = launch_only_launder(head, phi, cfg, mw,
                                    fam.indicator(fam.unseen, cfg.n_wells),
                                    jax.random.fold_in(jax.random.PRNGKey(2000 + seed), 1))
        sc_l = _score(lau_s, lau_u, fam, anchors, cfg, seed)

        readers = sorted(set(sc_p["unseen"]) & set(sc_n["unseen"]))
        od = {r: sc_p["unseen"][r] - sc_n["unseen"][r] for r in readers}
        g1 = {r: sc_p["unseen"][r] - sc_l["unseen"][r] for r in readers}
        n_par = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda a: a, phys.V))))
        led = [mwr_ledger("physics", cfg, mw, store_params=n_par, head=head,
                          phi_bytes=phi.n_bytes(), organizer="settle"),
               mwr_ledger("null_N1prime", cfg, mw, store_params=fit["ledger"]["n_params"],
                          head=head, phi_bytes=phi.n_bytes(),
                          organizer="gradient_static"),
               mwr_ledger("launder_La", cfg, mw, store_params=0, head=head,
                          phi_bytes=phi.n_bytes(), organizer="none")]
        rows.append({
            "seed": seed, "s_measured": s_hat,
            "d_over_s": float(cfg.target_ds * cfg.s_measured / max(s_hat, 1e-9)),
            "min_sep": float(min_separation(anchors)),
            "write": wi, "organize": oi, "chance": float(
                __import__("chlu.core.factored_store", fromlist=["x"]
                           ).chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)),
            "tol": float(fam.tol),
            "physics": sc_p, "null_N1prime": sc_n, "launder_La": sc_l,
            "OD": od, "OD_min": float(min(od.values())),
            "G1_read_minus_launder": g1,
            "G1_min": float(min(g1.values())),
            "stats_unseen_physics": st_u, "stats_seen_physics": st_s,
            "stats_unseen_null": st_n,
            "S_eff_physics": s_effective(st_u["n_wells_ever_occupied"], cfg),
            "S_eff_physics_gated": s_effective(
                st_u.get("n_wells_ever_occupied_gated", 0), cfg),
            "ledgers": led, "k_matched": assert_k_matched(led),
            "wall_s": float(time.time() - t0)})
        print(f"  [arms] seed {seed}: OD_min {rows[-1]['OD_min']:+.4f} "
              f"G1_min {rows[-1]['G1_min']:+.4f} "
              f"exact_occ {st_u['exact_set_occupancy_gated']:.4f} "
              f"({rows[-1]['wall_s']:.0f}s)", flush=True)
    return {"rows": rows, **_aggregate(rows)}


def _aggregate(rows) -> Dict[str, Any]:
    def ms(vals):
        v = np.asarray(vals, float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        return {"mean": float(v.mean()), "sd": sd,
                "se": float(sd / max(np.sqrt(len(v)), 1)),
                "two_se": float(2 * sd / max(np.sqrt(len(v)), 1))}
    readers = sorted(rows[0]["OD"])
    out = {"n_seeds": len(rows),
           "OD_min": ms([r["OD_min"] for r in rows]),
           "G1_min": ms([r["G1_min"] for r in rows]),
           "OD_per_reader": {r: ms([x["OD"][r] for x in rows]) for r in readers},
           "physics_per_reader": {r: ms([x["physics"]["unseen"][r] for x in rows])
                                  for r in readers},
           "null_per_reader": {r: ms([x["null_N1prime"]["unseen"][r] for x in rows])
                               for r in readers},
           "launder_per_reader": {r: ms([x["launder_La"]["unseen"][r] for x in rows])
                                  for r in readers},
           "chance": ms([r["chance"] for r in rows])}
    for f in ("exact_set_occupancy_gated", "exact_set_occupancy_raw",
              "distinct_wells_raw", "distinct_wells_gated",
              "occupancy_precision_raw", "occupancy_precision_gated",
              "coverage_gated", "ge_F_distinct_raw"):
        out[f"physics_{f}"] = ms([r["stats_unseen_physics"][f] for r in rows])
        out[f"null_{f}"] = ms([r["stats_unseen_null"][f] for r in rows])
    out["S_eff"] = ms([r["S_eff_physics"] for r in rows])
    out["S_eff_gated"] = ms([r["S_eff_physics_gated"] for r in rows])
    band = (out["S_eff"]["mean"] >= 8.0) and (out["S_eff"]["mean"] <= 16.0)
    out["S_eff_verdict"] = "in band [8, 16]" if band else "COLLAPSED"
    bar = 0.05
    m, t = out["OD_min"]["mean"], out["OD_min"]["two_se"]
    out["F1"] = ("CLEARS" if m - t > bar else
                 "FIRES" if m + t < bar and abs(m) > bar else "TIE")
    return out


# ==========================================================================
# stage guards — each guard must FIRE on a designed negative
# ==========================================================================
def stage_guards(cfg, mw, seeds) -> Dict[str, Any]:
    seed = seeds[0]
    phi, anchors, fam, blank, store, head, _ = _build_seed(cfg, mw, seed)
    key = jax.random.PRNGKey(700 + seed)
    g2 = hard_vs_soft_gradient(store, head, phi, cfg, mw, fam, key, n=8)
    g3 = staging_gradient_probe(blank, store, head, phi, cfg, mw, fam, key, n=8)
    # a HISTORICAL-init blank (no localized atoms, no payload-shell init) — the
    # un-designed store the pilot measured at 1e-10
    cfg_hist = CatTestConfig(**{**cfg.as_dict(), "atom_local_radius": 0.0,
                               "atom_payload_init_radius": 0.0})
    hist_blank = FactoredStore(cfg_hist, anchors, jax.random.PRNGKey(seed))
    g3b = staging_gradient_probe(hist_blank, store, head, phi, cfg, mw, fam, key, n=8)
    # guard 4: k is capacity — a 2k arm at the same bytes must score higher
    g4 = {}
    for kk in (mw.k_particles, 2 * mw.k_particles):
        mwk = registered_mw(k_particles=int(kk))
        h = LaunchHead(phi, anchors, cfg, mwk)
        ls, lu = _latents(store, h, phi, cfg, mwk, fam, jax.random.PRNGKey(2000 + seed))
        sc = _score(ls, lu, fam, anchors, cfg, seed)
        st = read_stats(lu, anchors, fam.unseen, cfg.f_subset)
        g4[f"k={kk}"] = {"unseen": sc["unseen"],
                         "best": max(sc["unseen"].values()),
                         "coverage_raw": st["coverage_raw"],
                         "distinct_raw": st["distinct_wells_raw"],
                         "ledger": mwr_ledger("k_probe", cfg, mwk, store_params=0,
                                              head=h, phi_bytes=phi.n_bytes())}
    try:
        assert_k_matched([g4[f"k={mw.k_particles}"]["ledger"],
                          g4[f"k={2 * mw.k_particles}"]["ledger"]])
        g4["mismatch_raises"] = False
    except ValueError as e:
        g4["mismatch_raises"] = True
        g4["mismatch_message"] = str(e)
    return {"G2_soft_vs_hard": g2, "G3_staging_designed_blank": g3,
            "G3_staging_historical_blank": g3b, "G4_k_is_capacity": g4,
            "G4_coverage_gain": float(g4[f"k={2 * mw.k_particles}"]["coverage_raw"]
                                      - g4[f"k={mw.k_particles}"]["coverage_raw"])}


# ==========================================================================
# stage consolidate — (d) + the trash region's first use
# ==========================================================================
def stage_consolidate(cfg, mw, seeds) -> Dict[str, Any]:
    rows = []
    for seed in seeds:
        phi, anchors, fam, _blank, store, head, _ = _build_seed(cfg, mw, seed)
        sep = cfg.target_ds * cfg.s_measured
        c, dep = find_wells(store, cfg, mw, jax.random.PRNGKey(900 + seed))
        con = consolidate_wells(c[:, : cfg.addr_dim], dep, budget=mw.well_budget,
                                merge_radius=mw.merge_radius_frac * sep,
                                trash_depth_frac=mw.trash_depth_frac)
        tf = trash_field(con["trashed_centers"], cfg.dim,
                         radius=mw.trash_radius_frac * sep, strength=mw.trash_gamma)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        lat_off = multiwell_read(store, head, phi, cfg, mw, ind_u,
                                 jax.random.PRNGKey(2000 + seed))
        lat_on = multiwell_read(store, head, phi, cfg, mw, ind_u,
                                jax.random.PRNGKey(2000 + seed),
                                wells=con["kept_centers"], trash=tf)
        st_off = read_stats(lat_off, anchors, fam.unseen, cfg.f_subset)
        st_on = read_stats(lat_on, anchors, fam.unseen, cfg.f_subset)
        # how far the consolidated table is from the designed anchors
        kc = con["kept_centers"]
        dist = (float(np.mean(np.linalg.norm(
            kc[:, None, :] - anchors[None, :, : cfg.addr_dim], axis=-1).min(1)))
            if len(kc) else float("nan"))
        rows.append({"seed": seed,
                     "n_found": con["n_found"], "n_kept": con["n_kept"],
                     "n_trashed_over_budget": con["n_trashed_over_budget"],
                     "n_trashed_shallow": con["n_trashed_shallow"],
                     "depth_threshold": con["depth_threshold"],
                     "merge_radius": con["merge_radius"],
                     "kept_depth_mean": float(np.mean(con["kept_depths"]))
                     if con["n_kept"] else float("nan"),
                     "trashed_depth_mean": float(np.mean(con["trashed_depths"]))
                     if len(con["trashed_depths"]) else float("nan"),
                     "mean_dist_kept_to_designed_anchor": dist,
                     "trash_holes": 0 if tf is None else int(tf.k),
                     "stats_trash_off": st_off, "stats_trash_on": st_on,
                     "S_eff_off": s_effective(st_off["n_wells_ever_occupied"], cfg),
                     "S_eff_on": s_effective(st_on["n_wells_ever_occupied"], cfg)})
        print(f"  [consolidate] seed {seed}: found {con['n_found']} kept "
              f"{con['n_kept']} trashed "
              f"{con['n_trashed_over_budget'] + con['n_trashed_shallow']}", flush=True)
    return {"rows": rows,
            "n_found_mean": float(np.mean([r["n_found"] for r in rows])),
            "n_kept_mean": float(np.mean([r["n_kept"] for r in rows])),
            "n_trashed_mean": float(np.mean(
                [r["n_trashed_over_budget"] + r["n_trashed_shallow"] for r in rows])),
            "S_eff_off_mean": float(np.mean([r["S_eff_off"] for r in rows])),
            "S_eff_on_mean": float(np.mean([r["S_eff_on"] for r in rows]))}


# ==========================================================================
# stage levers — (e) learned p0, and the depth-heterogeneity suspect
# ==========================================================================
def stage_levers(cfg, mw, seeds) -> Dict[str, Any]:
    out = {}
    for name, ckw, mkw in (("p0_on", {}, {}),
                           ("p0_off", {}, dict(learned_p0=False)),
                           ("depth_ratio_1", dict(depth_ratio=1.0), {}),
                           ("dedupe_sum", {}, dict(dedupe="sum")),
                           ("no_descent_gate", {}, dict(descent_gate=False))):
        rows = []
        for seed in seeds:
            c = CatTestConfig(**{**cfg.as_dict(), **ckw})
            m = registered_mw(**{**{"payload_ref": 0.5, "conf_w": 0.05}, **mkw})
            phi, anchors, fam, _b, store, head, _ = _build_seed(c, m, seed)
            ls, lu = _latents(store, head, phi, c, m, fam,
                              jax.random.PRNGKey(2000 + seed))
            sc = _score(ls, lu, fam, anchors, c, seed)
            st = read_stats(lu, anchors, fam.unseen, c.f_subset)
            occ = occupancy(lu["z"], anchors[:, : c.addr_dim])
            inA = np.array([np.isin(o, A) for o, A in
                            zip(occ, fam.unseen, strict=True)])
            rows.append({"seed": seed, "best_unseen": max(sc["unseen"].values()),
                         "unseen": sc["unseen"],
                         "distinct_raw": st["distinct_wells_raw"],
                         "precision_raw": st["occupancy_precision_raw"],
                         "precision_gated": st["occupancy_precision_gated"],
                         "coverage_gated": st["coverage_gated"],
                         "exact_occ_gated": st["exact_set_occupancy_gated"],
                         "w_in_A": float(lu["w"][inA].mean()),
                         "w_out_A": float(lu["w"][~inA].mean())})
        out[name] = {"rows": rows, **{
            f: float(np.mean([r[f] for r in rows])) for f in
            ("best_unseen", "distinct_raw", "precision_raw", "precision_gated",
             "coverage_gated", "exact_occ_gated", "w_in_A", "w_out_A")}}
        print(f"  [levers] {name}: w_inA {out[name]['w_in_A']:.3f} "
              f"prec_gated {out[name]['precision_gated']:.3f} "
              f"best {out[name]['best_unseen']:.4f}", flush=True)
    return out


# ==========================================================================
# entry point
# ==========================================================================
def run_tierii_read(*, seeds=(0, 1, 2, 3, 4), stages=STAGES, out_dir=None,
                    quick: bool = False, organize_steps: int = 60,
                    k_particles: int = 12) -> Dict[str, Any]:
    if quick:
        cfg = registered_cfg(n_items=32, n_unseen=32, write_steps=30,
                             address_steps=40, read_steps=60)
        mw = registered_mw(k_particles=int(k_particles), head_steps=2, n_probes=32,
                           probe_steps=40, head_settle_address=20,
                           head_settle_read=30)
        seeds, organize_steps = list(seeds)[:1], 2
    else:
        cfg = registered_cfg()
        mw = registered_mw(k_particles=int(k_particles))
    res: Dict[str, Any] = {
        "config": cfg.as_dict(), "mw_config": mw.as_dict(),
        "cfg_non_default": cfg.as_flag_table(), "mw_non_default": mw.as_flag_table(),
        "seeds": list(seeds), "quick": bool(quick),
        "organize_steps": int(organize_steps)}
    t0 = time.time()
    if "k0" in stages:
        print("[stage] k0", flush=True)
        res["k0"] = stage_k0(cfg, mw, list(seeds))
        print(f"  K0(new) = {res['k0']['K0_new']:.4f}  "
              f"K0(refuted P=4 @ d=4) = {res['k0']['K0_refuted_P4_at_d4']:.4f}  "
              f"=> {res['k0']['verdict']}", flush=True)
        if res["k0"]["verdict"] != "PASS":
            print("  ⛔ K0 FAILS — per the DIAL declaration the protocol is "
                  "redesigned before anything else runs.", flush=True)
    if "arms" in stages:
        print("[stage] arms", flush=True)
        res["arms"] = stage_arms(cfg, mw, list(seeds), organize_steps=organize_steps)
    if "guards" in stages:
        print("[stage] guards", flush=True)
        res["guards"] = stage_guards(cfg, mw, list(seeds))
    if "consolidate" in stages:
        print("[stage] consolidate", flush=True)
        res["consolidate"] = stage_consolidate(cfg, mw, list(seeds))
    if "levers" in stages:
        print("[stage] levers", flush=True)
        res["levers"] = stage_levers(cfg, mw, list(seeds))
    res["wall_s"] = float(time.time() - t0)
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tierii_read_summary.json").write_text(
            json.dumps(res, indent=1, default=float))
        print(f"[out] {p / 'tierii_read_summary.json'}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="tier-ii multi-well read protocol")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--organize-steps", type=int, default=60)
    ap.add_argument("--k-particles", type=int, default=12)
    a = ap.parse_args(argv)
    run_tierii_read(seeds=a.seeds, stages=a.stages, out_dir=a.out_dir,
                    quick=a.quick, organize_steps=a.organize_steps,
                    k_particles=a.k_particles)


if __name__ == "__main__":
    main()
