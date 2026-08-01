"""⭐ THE CAT TEST — tier ii's spine (C2W5 ``orgdiv-cat-test``).

Harness for :mod:`chlu.core.factored_store`, run in the order
``PREREG-TierII.md`` §10 registers, because **steps 1–3 can kill the wave cheaply
and that is the point**:

1. :func:`stage_calibrate` — the effective-``s`` estimator (OQ-1, BLOCKING) sets
   the operating point on **measured** ``d/s``, then the **K1** write-admissibility
   sweep over ``a in {4, 12, 32}``.
2. :func:`stage_family` — build the family and assert **K2** (rule 4) per query,
   plus the ``m``- and ``d``-sweeps that locate the family's own feasibility
   window (SP-1/SP-2 of ``.claude/outputs/orgdiv-cat-test/PREREG.md``).
3. :func:`stage_controls` — **K3** (nearest-item table), **K4** (the four leak
   controls), **K5** (the per-item table launder) on the physics arm ALONE,
   before any arm is compared.
4. :func:`stage_arm` — the physics arm scored over the frozen reader class and the
   registered ``gamma`` axis, plus the two in-house nulls (N3 fitted
   static-geometric = F5's null, N4 kNN).
5. :func:`stage_deletion` — the deletion curve (two series, never a scalar).

⛔ **The settle-deleted / matched-bytes launder is TIER i's control and is the
WRONG control here.** It is run and reported by :func:`stage_controls` labelled
*"inherited diagnostic"*, never as tier-ii evidence.

⛔ **Claim form (prereg §2.6):** no well is named semantically anywhere — not in
code, not in an artifact, not in a caption.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.factored_store import (
    READERS,
    CatFamily,
    CatTestConfig,
    FactoredStore,
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
    occupancy_precision,
    organize_physics,
    place_wells,
    query_identifiability,
    reader_bytes,
    score_curve,
    score_reader,
    write_wells,
)
from chlu.core.soft_certificate import capture_radius

__all__ = [
    "run_cat_test",
    "stage_calibrate",
    "stage_family",
    "stage_controls",
    "stage_arm",
    "stage_d_sweep",
    "stage_deletion",
    "build_physics_arm",
]


def _j(o):
    """JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(obj), indent=2))
    return path


# ==========================================================================
# helpers shared by the stages
# ==========================================================================
def _relax_fn(store: FactoredStore, cfg: CatTestConfig):
    """``(n, dim) -> (n, dim)`` post-write relaxation (SC-6's ``relax_fn``)."""
    model = store.model(cfg)
    from chlu.core.factored_store import _settle

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


def _targets(cfg: CatTestConfig, anchors: np.ndarray, payloads: np.ndarray):
    d, m = int(cfg.addr_dim), int(cfg.payload_dim)
    t = np.zeros((cfg.n_wells, cfg.dim), dtype=np.float32)
    t[:, :d] = anchors[:, :d]
    t[:, d:d + m] = payloads
    return t


def _depth_scale(cfg: CatTestConfig) -> np.ndarray:
    """Registered depth heterogeneity >= 3x between NEIGHBOURING wells."""
    r = float(cfg.depth_ratio)
    return np.where(np.arange(cfg.n_wells) % 2 == 0, 1.0, r)


def build_physics_arm(cfg: CatTestConfig, family: CatFamily, seed: int, *,
                      sep: Optional[float] = None, organize: bool = True,
                      phi=None) -> Dict[str, Any]:
    """Place -> write -> organize. Returns the arm's store + its reports."""
    phi = phi if phi is not None else build_phi(cfg)
    sep = float(sep) if sep is not None else float(cfg.target_ds * cfg.atom_width)
    anchors = place_wells(phi, cfg, sep=sep)
    key = jax.random.PRNGKey(int(seed))
    k_init, k_write, k_org = jax.random.split(key, 3)
    store = FactoredStore(cfg, anchors, k_init)
    # insertion order is re-shuffled per seed (prereg §2.3 rule 1)
    order = np.random.default_rng(int(seed)).permutation(cfg.n_wells)
    store, wrep = write_wells(store, cfg, anchors, family.payloads, k_write,
                              order=order, depth_scale=_depth_scale(cfg))
    orep: Dict[str, Any] = {}
    if organize:
        store, orep = organize_physics(store, phi, cfg, family, k_org)
    return {"store": store, "phi": phi, "anchors": anchors, "sep": sep,
            "write": wrep, "organize": orep, "order": order.tolist()}


# ==========================================================================
# STAGE 1 — the effective-`s` instrument, the operating point, and K1
# ==========================================================================
def stage_calibrate(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                    a_values: Sequence[int] = (4, 12, 32),
                    out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐ OQ-1 (BLOCKING) + **K1**.

    For each ``a``: place wells at a provisional spacing, write, **measure** the
    effective ``s`` of every written well, re-derive the spacing from measured
    ``d/s``, re-place and re-write, then adjudicate K1's three bars.
    """
    res: Dict[str, Any] = {"a_values": list(a_values), "seeds": list(seeds),
                           "cells": []}
    for a in a_values:
        for seed in seeds:
            t0 = time.time()
            c = replace(cfg, atoms_per_well=int(a))
            fam = build_family(c, seed=seed)
            phi = build_phi(c)
            # pass 1: provisional spacing from the ATOM-WIDTH ruler
            arm = build_physics_arm(c, fam, seed, sep=c.target_ds * c.atom_width,
                                    organize=False, phi=phi)
            tg = _targets(c, arm["anchors"], fam.payloads)
            s1 = [effective_s(arm["store"].V, tg[j], s_hint=c.atom_width, seed=seed,
                              confine=c.confine)
                  for j in range(c.n_wells)]
            s_meas = float(np.nanmedian([x["s"] for x in s1]))
            # pass 2: the operating point set on MEASURED d/s
            sep2 = float(c.target_ds * s_meas) if np.isfinite(s_meas) else None
            arm2 = build_physics_arm(c, fam, seed, sep=sep2, organize=False, phi=phi)
            tg2 = _targets(c, arm2["anchors"], fam.payloads)
            s2 = [effective_s(arm2["store"].V, tg2[j], s_hint=c.atom_width, seed=seed,
                              confine=c.confine)
                  for j in range(c.n_wells)]
            s_fit = np.array([x["s"] for x in s2], dtype=float)
            s_med = float(np.nanmedian(s_fit))
            sep_ach = min_separation(arm2["anchors"])

            # --- K1's three bars -------------------------------------------
            relax = _relax_fn(arm2["store"], c)
            q_star = relax(tg2)
            lam = _lambda_min(arm2["store"], q_star)
            n_cap = min(16, c.n_wells)  # SC-6 bisection is n_dirs relaxations/site
            caps = [capture_radius(relax, tg2[j], n_dirs=8, r_hi=1.0, steps=8,
                                   tol=0.15, seed=seed)["capture_radius"]
                    for j in range(n_cap)]
            frac_lam = float((lam > 0).mean())
            frac_cap = float(np.mean(np.asarray(caps) >= c.query_sigma))
            loss = float(arm2["write"]["endpoint_write_loss"])
            k1 = {"endpoint_write_loss": loss,
                  "loss_ok": bool(loss <= 0.05),
                  "frac_lambda_min_pos": frac_lam, "lambda_ok": bool(frac_lam >= 0.90),
                  "frac_capture_ge_sigma_q": frac_cap,
                  "capture_ok": bool(frac_cap >= 0.90),
                  "n_capture_sites": n_cap,
                  "lambda_min_min": float(np.min(lam)),
                  "lambda_min_median": float(np.median(lam)),
                  "capture_median": float(np.median(caps)),
                  "sigma_q": float(c.query_sigma)}
            k1["K1_PASS"] = bool(k1["loss_ok"] and k1["lambda_ok"] and k1["capture_ok"])
            res["cells"].append({
                "a": int(a), "seed": int(seed),
                "s_pass1_median": s_meas,
                "s_measured_median": s_med,
                "s_measured_iqr": [float(np.nanpercentile(s_fit, 25)),
                                   float(np.nanpercentile(s_fit, 75))],
                "s_fit_r2_median": float(np.nanmedian([x["r2"] for x in s2])),
                "well_depth_median": float(np.nanmedian([x["depth"] for x in s2])),
                "sep_target": sep2, "sep_achieved": sep_ach,
                "ds_measured": float(sep_ach / s_med) if s_med > 0 else float("nan"),
                "ds_atom_width_ruler": float(sep_ach / c.atom_width),
                "K1": k1,
                "bytes": {"store": arm2["store"].n_bytes(), "phi": phi.n_bytes(),
                          **byte_ratio(c)},
                "wall_s": round(time.time() - t0, 1),
            })
            print(f"[K1] a={a} seed={seed} s={s_med:.4f} d/s={sep_ach/max(s_med,1e-9):.2f} "
                  f"loss={loss:.4f} lam+={frac_lam:.2f} cap={frac_cap:.2f} "
                  f"-> {'PASS' if k1['K1_PASS'] else 'FAIL'} ({time.time()-t0:.0f}s)",
                  flush=True)
    if out:
        _dump(res, out / "stage_calibrate.json")
    return res


# ==========================================================================
# STAGE 2 — the family, K2, and the SP-1/SP-2 feasibility sweeps
# ==========================================================================
def stage_family(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4),
                 m_values: Sequence[int] = (1, 2, 4, 6, 8, 12),
                 d_values: Sequence[int] = (4, 8, 16, 24, 32, 48, 64),
                 out: Optional[Path] = None) -> Dict[str, Any]:
    """**K2** at the registered design point + the two structural sweeps."""
    res: Dict[str, Any] = {}

    # -- K2 at the registered point --------------------------------------
    k2_cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        k2_cells.append({"seed": int(seed), **fam.k2,
                         "chance": chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)})
    res["K2"] = {
        "cells": k2_cells,
        "overlap_ok_all": bool(all(c["overlap_ok"] for c in k2_cells)),
        "payload_sep_ok_all": bool(all(c["payload_sep_ok"] for c in k2_cells)),
        "frac_payload_sep_ok_mean": float(np.mean([c["frac_payload_sep_ok"]
                                                   for c in k2_cells])),
        "n_valid_heldout_mean": float(np.mean([c["n_valid_heldout"]
                                               for c in k2_cells])),
        "n_total_combos": k2_cells[0]["n_total_combos"],
        "chance_mean": float(np.mean([c["chance"] for c in k2_cells])),
    }
    res["K2"]["K2_PASS"] = bool(res["K2"]["overlap_ok_all"]
                                and res["K2"]["payload_sep_ok_all"])

    # -- the m-sweep: K2's SECOND assertion vs the payload dimension -------
    m_rows = []
    for m in m_values:
        fr, ms, ch = [], [], []
        for seed in seeds:
            c = replace(cfg, payload_dim=int(m))
            f = build_family(c, seed=seed)
            fr.append(f.k2["frac_payload_sep_ok"])
            ms.append(f.k2["min_payload_sep"] / f.tol)
            ch.append(chance_accuracy(f.y_seen, f.y_unseen, f.tol))
        m_rows.append({"m": int(m), "frac_K2b_pass": float(np.mean(fr)),
                       "min_sep_over_tol": float(np.mean(ms)),
                       "chance": float(np.mean(ch))})
    res["m_sweep"] = m_rows

    # -- ⭐ SP-2: the two squeezes on d ------------------------------------
    d_rows = []
    for d in d_values:
        r2u, prec, ex = [], [], []
        for seed in seeds:
            c = replace(cfg, addr_dim=int(d))
            f = build_family(c, seed=seed)
            phi = build_phi(c, phi_seed=20260801 + int(d))
            r2u.append(query_identifiability(phi, f, c)["r2_unseen"])
            ind = jnp.asarray(f.indicator(f.unseen, c.n_wells))
            sc = np.asarray(phi.set_code(ind)) @ np.asarray(phi.codes).T
            topF = np.argsort(-sc, axis=1)[:, : c.f_subset]
            hit = np.array([np.isin(topF[i], f.unseen[i]).mean()
                            for i in range(len(topF))])
            prec.append(hit.mean())
            ex.append(float((hit == 1.0).mean()))
        d_rows.append({"d": int(d), "rank_ceiling": float(d / cfg.n_wells),
                       "query_only_r2_unseen": float(np.mean(r2u)),
                       "matched_filter_precision": float(np.mean(prec)),
                       "matched_filter_exact_set": float(np.mean(ex))})
    res["d_sweep_SP2"] = d_rows
    res["chance_occupancy_precision"] = float(cfg.f_subset / cfg.n_wells)

    # -- SP-1: the out-of-class 32-dof query-only probe --------------------
    sp1 = []
    for seed in seeds:
        f = build_family(cfg, seed=seed)
        Xs = f.indicator(f.seen, cfg.n_wells)
        Xu = f.indicator(f.unseen, cfg.n_wells)
        w, *_ = np.linalg.lstsq(Xs, f.y_seen, rcond=None)
        sp1.append({"seed": int(seed),
                    "acc_unseen": exact_set_accuracy(Xu @ w, f.y_unseen, f.tol),
                    "v_recovery_linf": float(np.abs(w - f.payloads).max()),
                    "n_params": int(w.size)})
    res["SP1_out_of_class_probe"] = {
        "cells": sp1,
        "acc_unseen_mean": float(np.mean([c["acc_unseen"] for c in sp1])),
        "v_recovery_linf_max": float(np.max([c["v_recovery_linf"] for c in sp1])),
        "note": ("DECLARED OUT-OF-CLASS DIAGNOSTIC (PREREG SP-1): an OLS fit of y "
                 "on the TRUE indicator, N_a*m dof. It is the family's structural "
                 "ceiling, reported and never scored as an arm or as a K4 leg."),
    }
    if out:
        _dump(res, out / "stage_family.json")
    return res


# ==========================================================================
# STAGE 3 — K3 / K4 / K5 on the physics arm ALONE
# ==========================================================================
def _score_all(readers, z, y, tol) -> Dict[str, float]:
    return {k: score_reader(v, z, y, tol) for k, v in readers.items()}


def _curve_all(readers, z, y, tol) -> Dict[str, Dict[str, float]]:
    """Accuracy-vs-tol curve per reader — *quote the curve, not the endpoint*."""
    return {k: score_curve(v, z, y, tol) for k, v in readers.items()}


def stage_controls(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                   out: Optional[Path] = None,
                   arms: Optional[Dict[int, Dict]] = None) -> Dict[str, Any]:
    """**K3**, **K4** (4 legs) and **K5** — all BEFORE any arm comparison."""
    cells = []
    for seed in seeds:
        t0 = time.time()
        fam = build_family(cfg, seed=seed)
        arm = (arms or {}).get(seed) or build_physics_arm(cfg, fam, seed)
        store, phi, anchors = arm["store"], arm["phi"], arm["anchors"]
        k_r = jax.random.PRNGKey(1000 + seed)
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        z_s = multi_particle_read(store, phi, cfg, ind_s, k_r)
        z_u = multi_particle_read(store, phi, cfg, ind_u,
                                  jax.random.fold_in(k_r, 1))
        readers = fit_readers(z_s, fam.y_seen, anchors=anchors,
                              well_payloads=fam.payloads, seed=seed)
        chance = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
        phys = _score_all(readers, z_u, fam.y_unseen, fam.tol)
        phys_seen = _score_all(readers, z_s, fam.y_seen, fam.tol)

        # -- K3: nearest STORED ITEM table (+0 B substitute) ----------------
        code_s = np.asarray(phi.set_code(jnp.asarray(ind_s)))
        code_u = np.asarray(phi.set_code(jnp.asarray(ind_u)))
        nn = np.argmin(((code_u[:, None, :] - code_s[None, :, :]) ** 2).sum(-1), 1)
        k3_acc = exact_set_accuracy(fam.y_seen[nn], fam.y_unseen, fam.tol)
        # the strongest +0 B substitute on the raw item table: IDW over k
        best_sub = k3_acc
        d_cs = np.linalg.norm(code_u[:, None, :] - code_s[None, :, :], axis=-1)
        for k in (2, 3, 5, 10):
            idx = np.argsort(d_cs, axis=1)[:, :k]
            w = 1.0 / (np.take_along_axis(d_cs, idx, 1) + 1e-9)
            w /= w.sum(1, keepdims=True)
            best_sub = max(best_sub, exact_set_accuracy(
                (w[..., None] * fam.y_seen[idx]).sum(1), fam.y_unseen, fam.tol))
        k3 = {"nearest_item_table": k3_acc, "strongest_plus0B_substitute": best_sub,
              "bar": 0.60, "K3_PASS": bool(max(k3_acc, best_sub) <= 0.60)}

        # -- K4: the four leak controls -------------------------------------
        blank = FactoredStore(cfg, anchors, jax.random.PRNGKey(seed))  # never written
        z_blank = multi_particle_read(blank, phi, cfg, ind_u,
                                      jax.random.fold_in(k_r, 2))
        z_blank_s = multi_particle_read(blank, phi, cfg, ind_s,
                                        jax.random.fold_in(k_r, 3))
        rd_blank = fit_readers(z_blank_s, fam.y_seen, anchors=anchors,
                               well_payloads=np.zeros_like(fam.payloads), seed=seed)
        leg_blank = _score_all(rd_blank, z_blank, fam.y_unseen, fam.tol)

        # query-only: the SAME reader class on phi's launch points (no store)
        def _qonly(ind_, phi=phi, seed=seed):
            kq = jax.random.PRNGKey(7 + seed)
            keys = jax.random.split(kq, len(ind_))
            sig = float(cfg.query_sigma)
            launch1 = phi.launch
            L = jax.vmap(lambda ind1, kk: launch1(ind1, kk, sig))(
                jnp.asarray(ind_, dtype=jnp.float32), keys)
            return np.asarray(L)

        zq_s, zq_u = _qonly(ind_s), _qonly(ind_u)
        rd_q = fit_readers(zq_s, fam.y_seen, seed=seed,
                           which=("sum_linear", "knn", "mlp"))
        leg_query = _score_all(rd_q, zq_u, fam.y_unseen, fam.tol)

        # permuted payloads (same keys, v_j permuted)
        rng = np.random.default_rng(seed + 99)
        perm = rng.permutation(cfg.n_wells)
        fam_p = CatFamily(seen=fam.seen, unseen=fam.unseen,
                          payloads=fam.payloads[perm],
                          y_seen=fam.indicator(fam.seen, cfg.n_wells) @ fam.payloads[perm],
                          y_unseen=fam.indicator(fam.unseen, cfg.n_wells) @ fam.payloads[perm],
                          tol=fam.tol, n_valid_heldout=fam.n_valid_heldout,
                          n_total_combos=fam.n_total_combos, order=fam.order)
        rd_perm = fit_readers(z_s, fam_p.y_seen, anchors=anchors,
                              well_payloads=fam.payloads, seed=seed)
        leg_perm = _score_all(rd_perm, z_u, fam_p.y_unseen, fam.tol)

        # address-leak probe: decode item identity from the read's ADDRESS block
        occ_u = occupancy(z_u, anchors)
        leak_full = float(np.mean([np.isin(occ_u[i], fam.unseen[i]).mean()
                                   for i in range(len(occ_u))]))
        occ_q = occupancy(zq_u, anchors)
        leak_launder = float(np.mean([np.isin(occ_q[i], fam.unseen[i]).mean()
                                      for i in range(len(occ_q))]))
        bar = chance + 0.05
        k4 = {"chance": chance, "bar": bar,
              "blank_store": leg_blank, "query_only": leg_query,
              "permuted_payloads": leg_perm,
              "address_leak_full": leak_full,
              "address_leak_launder": leak_launder,
              "address_leak_dividend": leak_full - leak_launder,
              "blank_ok": bool(max(leg_blank.values()) <= bar),
              "query_only_ok": bool(max(leg_query.values()) <= bar),
              "permuted_ok": bool(max(leg_perm.values()) <= bar)}
        k4["K4_PASS"] = bool(k4["blank_ok"] and k4["query_only_ok"]
                             and k4["permuted_ok"])

        # -- K5: the per-item table launder, THROUGH THE SAME READER CLASS --
        nn_s = np.argmin(np.where(np.eye(len(code_s), dtype=bool), np.inf,
                                  ((code_s[:, None, :] - code_s[None, :, :]) ** 2
                                   ).sum(-1)), axis=1)
        z_tab_seen = z_s[nn_s]       # each seen item read as its nearest OTHER row
        z_tab_unseen = z_s[nn]       # each unseen query read as its nearest row
        rd_tab = fit_readers(z_tab_seen, fam.y_seen, anchors=anchors,
                             well_payloads=fam.payloads, seed=seed)
        tab = _score_all(rd_tab, z_tab_unseen, fam.y_unseen, fam.tol)
        margins = {k: phys[k] - tab.get(k, 0.0) for k in phys}
        k5 = {"table_scores": tab, "physics_scores": phys, "margins": margins,
              "best_margin": float(max(margins.values())), "bar": 0.10,
              "K5_PASS": bool(max(margins.values()) > 0.10)}

        # -- inherited tier-i diagnostic: the settle-deleted launder --------
        z_launder = z_u.copy()
        z_launder[:, :, cfg.addr_dim:] = 0.0  # payload channel deleted
        rd_l = fit_readers(z_s, fam.y_seen, anchors=anchors,
                           well_payloads=fam.payloads, seed=seed)
        inherited = _score_all(rd_l, z_launder, fam.y_unseen, fam.tol)

        cells.append({
            "seed": int(seed), "chance": chance, "tol": fam.tol,
            "physics_unseen": phys, "physics_seen": phys_seen,
            "physics_unseen_curve": _curve_all(readers, z_u, fam.y_unseen, fam.tol),
            "occupancy_precision_unseen": occupancy_precision(z_u, anchors,
                                                              fam.unseen),
            "occupancy_precision_seen": occupancy_precision(z_s, anchors, fam.seen),
            "chance_occupancy": float(cfg.f_subset / cfg.n_wells),
            "K3": k3, "K4": k4, "K5": k5,
            "inherited_tier_i_launder": inherited,
            "reader_params": reader_bytes(readers),
            "write": arm["write"], "organize": arm.get("organize", {}),
            "sep": arm["sep"], "wall_s": round(time.time() - t0, 1),
        })
        print(f"[controls] seed={seed} chance={chance:.3f} phys={phys} "
              f"K3={k3['K3_PASS']} K4={k4['K4_PASS']} K5={k5['K5_PASS']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    res = {"cells": cells,
           "K3_PASS_all": bool(all(c["K3"]["K3_PASS"] for c in cells)),
           "K4_PASS_all": bool(all(c["K4"]["K4_PASS"] for c in cells)),
           "K5_PASS_all": bool(all(c["K5"]["K5_PASS"] for c in cells))}
    if out:
        _dump(res, out / "stage_controls.json")
    return res


# ==========================================================================
# STAGE 4 — the physics arm, the gamma axis, and the two in-house nulls
# ==========================================================================
def _n3_static_geometric(z_s, ind_s, ind_u, anchors, phi, cfg, fam, seed):
    """N3: the fitted static-geometric rule ``argmin_j[||z-c_j||^2/2s_j^2 - b_j]``.

    ⭐ This is **F5's null** and it needs no rollout: assignments come from a
    fitted power/Apollonius diagram on the LAUNCH points. It is also the
    "oracle-imitation" null when fitted on the physics arm's own assignments.
    """
    from scipy.optimize import minimize

    P = int(cfg.n_particles)

    def launch_pts(ind_):
        k = jax.random.PRNGKey(31 + seed)
        keys = jax.random.split(k, len(ind_))
        L = jax.vmap(lambda i, kk: phi.launch(jnp.asarray(i, jnp.float32), kk,
                                              float(cfg.query_sigma)))(
            jnp.asarray(ind_, jnp.float32), keys)
        return np.asarray(L)[..., : cfg.addr_dim].reshape(-1, cfg.addr_dim)

    Ls, Lu = launch_pts(ind_s), launch_pts(ind_u)
    c = np.asarray(anchors)

    def assign(L, log_s, b):
        d2 = ((L[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        return np.argmin(d2 / (2.0 * np.exp(log_s)[None, :] ** 2) - b[None, :], 1)

    # fit (log s, b) by coordinate search on the SEEN read objective
    x0 = np.concatenate([np.full(cfg.n_wells, np.log(cfg.atom_width)),
                         np.zeros(cfg.n_wells)])

    def obj(x):
        ls, b = x[: cfg.n_wells], x[cfg.n_wells:]
        j = assign(Ls, ls, b).reshape(-1, P)
        pred = fam.payloads[j].sum(1)
        return float(((pred - fam.y_seen) ** 2).sum())

    r = minimize(obj, x0, method="Powell",
                 options={"maxiter": 2000, "maxfev": 4000, "xtol": 1e-2,
                          "ftol": 1e-2})
    ls, b = r.x[: cfg.n_wells], r.x[cfg.n_wells:]
    j_u = assign(Lu, ls, b).reshape(-1, P)
    z_n3 = np.zeros((len(j_u), P, cfg.dim), dtype=np.float32)
    z_n3[:, :, : cfg.addr_dim] = c[j_u]
    z_n3[:, :, cfg.addr_dim:] = fam.payloads[j_u]
    j_s = assign(Ls, ls, b).reshape(-1, P)
    z_n3_s = np.zeros((len(j_s), P, cfg.dim), dtype=np.float32)
    z_n3_s[:, :, : cfg.addr_dim] = c[j_s]
    z_n3_s[:, :, cfg.addr_dim:] = fam.payloads[j_s]
    return z_n3_s, z_n3, j_s, j_u


def stage_arm(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4),
              gammas: Sequence[float] = (0.02, 0.05, 0.2),
              out: Optional[Path] = None,
              arms: Optional[Dict[int, Dict]] = None) -> Dict[str, Any]:
    """The physics arm over the registered ``gamma`` axis + N3/N4 in-house nulls.

    ⚠ ``gamma = 0.2`` is the **internal VQ-collapse control** (Prop O4's x600
    collapse), never a claim cell. Claim cells are ``gamma in [0.05, 0.1]``.
    """
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        arm = (arms or {}).get(seed) or build_physics_arm(cfg, fam, seed)
        store, phi, anchors = arm["store"], arm["phi"], arm["anchors"]
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        chance = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
        for g in gammas:
            t0 = time.time()
            k = jax.random.PRNGKey(2000 + seed)
            z_s = multi_particle_read(store, phi, cfg, ind_s, k, gamma_address=g)
            z_u = multi_particle_read(store, phi, cfg, ind_u,
                                      jax.random.fold_in(k, 1), gamma_address=g)
            rd = fit_readers(z_s, fam.y_seen, anchors=anchors,
                             well_payloads=fam.payloads, seed=seed)
            phys = _score_all(rd, z_u, fam.y_unseen, fam.tol)
            occ_phys = occupancy(z_u, anchors)

            # -- N3 (F5's null) + F5's agreement statistic -----------------
            z3s, z3u, j3s, j3u = _n3_static_geometric(z_s, ind_s, ind_u, anchors,
                                                      phi, cfg, fam, seed)
            rd3 = fit_readers(z3s, fam.y_seen, anchors=anchors,
                              well_payloads=fam.payloads, seed=seed)
            n3 = _score_all(rd3, z3u, fam.y_unseen, fam.tol)
            f5_agreement = float((j3u == occ_phys).mean())

            # -- N4 kNN on the raw launch code (no store, no training) -----
            code_s = np.asarray(phi.set_code(jnp.asarray(ind_s)))
            code_u = np.asarray(phi.set_code(jnp.asarray(ind_u)))
            d_cs = np.linalg.norm(code_u[:, None, :] - code_s[None, :, :], axis=-1)
            n4_best = 0.0
            for kk in (1, 2, 3, 5, 10):
                idx = np.argsort(d_cs, axis=1)[:, :kk]
                w = 1.0 / (np.take_along_axis(d_cs, idx, 1) + 1e-9)
                w /= w.sum(1, keepdims=True)
                n4_best = max(n4_best, exact_set_accuracy(
                    (w[..., None] * fam.y_seen[idx]).sum(1), fam.y_unseen, fam.tol))

            null_star = {r: max(n3.get(r, 0.0), n4_best) for r in phys}
            od = {r: phys[r] - null_star[r] for r in phys}
            cells.append({
                "seed": int(seed), "gamma_address": float(g),
                "claim_cell": bool(0.05 <= g <= 0.1),
                "read_budget": [int(cfg.address_steps), int(cfg.read_steps)],
                "chance": chance,
                "physics": phys,
                "physics_curve": _curve_all(rd, z_u, fam.y_unseen, fam.tol),
                "N3_static_geometric": n3, "N4_knn_best": n4_best,
                "null_star_inhouse": null_star, "OD_inhouse": od,
                "OD_min_inhouse": float(min(od.values())),
                "F5_assignment_agreement": f5_agreement,
                "F5_fires": bool(f5_agreement >= 0.99),
                "occupancy_precision": occupancy_precision(z_u, anchors, fam.unseen),
                "wall_s": round(time.time() - t0, 1),
            })
            print(f"[arm] seed={seed} g={g} phys={ {k2: round(v,3) for k2,v in phys.items()} } "
                  f"ODmin={min(od.values()):+.3f} F5agree={f5_agreement:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    res = {"cells": cells, "readers": list(READERS)}
    # aggregate per gamma
    agg = {}
    for g in gammas:
        sub = [c for c in cells if c["gamma_address"] == g]
        if not sub:
            continue
        ods = np.array([c["OD_min_inhouse"] for c in sub])
        agg[str(g)] = {
            "n_seeds": len(sub),
            "OD_min_mean": float(ods.mean()),
            "OD_min_sd": float(ods.std(ddof=1)) if len(sub) > 1 else float("nan"),
            "OD_min_2se": float(2 * ods.std(ddof=1) / np.sqrt(len(sub)))
            if len(sub) > 1 else float("nan"),
            "F5_agreement_mean": float(np.mean([c["F5_assignment_agreement"]
                                                for c in sub])),
            "occupancy_precision_mean": float(np.mean([c["occupancy_precision"]
                                                       for c in sub])),
            "physics_mean": {r: float(np.mean([c["physics"][r] for c in sub]))
                             for r in sub[0]["physics"]},
        }
    res["aggregate"] = agg
    if out:
        _dump(res, out / "stage_arm.json")
    return res


# ==========================================================================
# STAGE 4b — ⭐ THE SP-2 SWEEP: the arm scored at every address dimension
# ==========================================================================
def stage_d_sweep(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                  d_values: Sequence[int] = (4, 8, 16, 32),
                  out: Optional[Path] = None) -> Dict[str, Any]:
    """⭐ The decisive table: at every ``d``, the physics arm's *store-side*
    recovery against the *query-side* leak, measured on the SAME split.

    This is the experiment SP-2 predicts is empty: the store needs ``d`` large to
    find ``A(x)`` at all, the query-only control needs ``d`` small to stay at
    chance, and the claim is that no ``d`` satisfies both. It is a property of the
    FAMILY, so it is measured before any organizer swap and it binds every arm.
    """
    cells = []
    for d in d_values:
        c = replace(cfg, addr_dim=int(d))
        for seed in seeds:
            t0 = time.time()
            fam = build_family(c, seed=seed)
            phi = build_phi(c, phi_seed=20260801 + int(d))
            arm = build_physics_arm(c, fam, seed, sep=c.target_ds * 0.40, phi=phi)
            store, anchors = arm["store"], arm["anchors"]
            ind_s = fam.indicator(fam.seen, c.n_wells)
            ind_u = fam.indicator(fam.unseen, c.n_wells)
            k = jax.random.PRNGKey(4000 + seed)
            z_s = multi_particle_read(store, phi, c, ind_s, k)
            z_u = multi_particle_read(store, phi, c, ind_u, jax.random.fold_in(k, 1))
            rd = fit_readers(z_s, fam.y_seen, anchors=anchors,
                             well_payloads=fam.payloads, seed=seed)
            phys = _score_all(rd, z_u, fam.y_unseen, fam.tol)

            def _launch(ind_, phi=phi, c=c, seed=seed):
                kq = jax.random.split(jax.random.PRNGKey(11 + seed), len(ind_))
                sig = float(c.query_sigma)
                lf = phi.launch
                return np.asarray(jax.vmap(lambda i1, kk: lf(i1, kk, sig))(
                    jnp.asarray(ind_, jnp.float32), kq))

            zq_s, zq_u = _launch(ind_s), _launch(ind_u)
            rd_q = fit_readers(zq_s, fam.y_seen, seed=seed,
                               which=("sum_linear", "knn", "mlp"))
            qonly = _score_all(rd_q, zq_u, fam.y_unseen, fam.tol)
            chance = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
            cells.append({
                "d": int(d), "seed": int(seed), "chance": chance,
                "physics": phys, "physics_best": float(max(phys.values())),
                "physics_curve": _curve_all(rd, z_u, fam.y_unseen, fam.tol),
                "query_only": qonly, "query_only_best": float(max(qonly.values())),
                "K4_query_only_bar": chance + 0.05,
                "K4_query_only_ok": bool(max(qonly.values()) <= chance + 0.05),
                "occupancy_precision_unseen": occupancy_precision(z_u, anchors,
                                                                  fam.unseen),
                "occupancy_precision_seen": occupancy_precision(z_s, anchors,
                                                               fam.seen),
                "chance_occupancy": float(c.f_subset / c.n_wells),
                "query_only_r2_unseen": query_identifiability(phi, fam, c)["r2_unseen"],
                "endpoint_write_loss": arm["write"]["endpoint_write_loss"],
                "wall_s": round(time.time() - t0, 1),
            })
            print(f"[d-sweep] d={d} seed={seed} phys_best={max(phys.values()):.3f} "
                  f"qonly_best={max(qonly.values()):.3f} chance={chance:.4f} "
                  f"occ={cells[-1]['occupancy_precision_unseen']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    agg = {}
    for d in d_values:
        sub = [x for x in cells if x["d"] == d]
        agg[str(d)] = {
            "n_seeds": len(sub),
            "physics_best_mean": float(np.mean([x["physics_best"] for x in sub])),
            "physics_best_2se": float(2 * np.std([x["physics_best"] for x in sub],
                                                 ddof=1) / np.sqrt(len(sub)))
            if len(sub) > 1 else float("nan"),
            "query_only_best_mean": float(np.mean([x["query_only_best"]
                                                   for x in sub])),
            "occupancy_precision_mean": float(np.mean(
                [x["occupancy_precision_unseen"] for x in sub])),
            "chance_mean": float(np.mean([x["chance"] for x in sub])),
            "K4_query_only_ok_all": bool(all(x["K4_query_only_ok"] for x in sub)),
        }
    res = {"cells": cells, "aggregate": agg}
    if out:
        _dump(res, out / "stage_d_sweep.json")
    return res


# ==========================================================================
# STAGE 5 — the deletion curve (TWO series; ⛔ never a single scalar)
# ==========================================================================
def stage_deletion(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                   p_values: Sequence[float] = (0.0045, 0.027, 0.036, 0.094),
                   out: Optional[Path] = None,
                   arms: Optional[Dict[int, Dict]] = None) -> Dict[str, Any]:
    """prereg §5.3: exactness on the private fraction + degradation on the shared.

    ⛔ *A single scalar "deletion still works" is inadmissible on a shared
    substrate.* Two series, x-axis ``p`` = private parameter-mass fraction.
    """
    cells = []
    for seed in seeds:
        fam = build_family(cfg, seed=seed)
        arm = (arms or {}).get(seed) or build_physics_arm(cfg, fam, seed)
        store, phi, anchors = arm["store"], arm["phi"], arm["anchors"]
        ind_s = fam.indicator(fam.seen, cfg.n_wells)
        ind_u = fam.indicator(fam.unseen, cfg.n_wells)
        k = jax.random.PRNGKey(3000 + seed)
        z_s = multi_particle_read(store, phi, cfg, ind_s, k)
        rd = fit_readers(z_s, fam.y_seen, anchors=anchors,
                         well_payloads=fam.payloads, seed=seed)
        base = score_reader(rd["sum_linear"],
                            multi_particle_read(store, phi, cfg, ind_u,
                                                jax.random.fold_in(k, 1)),
                            fam.y_unseen, fam.tol)
        for p in p_values:
            n_priv = max(1, int(round(p * cfg.n_atoms)))
            rows = np.zeros(cfg.n_atoms, dtype=bool)
            rows[:n_priv] = True  # the private atom block, deleted byte-exactly
            m = jnp.asarray(rows.astype(np.float32))
            V2 = eqx.tree_at(
                lambda t: [t.centers, t.amp], store.V,
                replace=[store.V.centers * (1 - m)[:, None],
                         store.V.amp * (1 - m)])
            st2 = eqx.tree_at(lambda s: s.V, store, V2)
            z2 = multi_particle_read(st2, phi, cfg, ind_u,
                                     jax.random.fold_in(k, 2))
            after = score_reader(rd["sum_linear"], z2, fam.y_unseen, fam.tol)
            # series 1: exactness on the private fraction (byte equality)
            exact = bool(np.all(np.asarray(V2.amp)[rows] == 0.0)
                         and np.all(np.asarray(V2.centers)[rows] == 0.0))
            # series 2: measured degradation on the SHARED fraction
            cells.append({"seed": int(seed), "p": float(p), "n_private_atoms": n_priv,
                          "private_byte_exact": exact,
                          "shared_read_acc_before": base,
                          "shared_read_acc_after": after,
                          "shared_degradation": float(base - after)})
            print(f"[deletion] seed={seed} p={p} exact={exact} "
                  f"acc {base:.3f} -> {after:.3f}", flush=True)
    res = {"cells": cells}
    if out:
        _dump(res, out / "stage_deletion.json")
    return res


# ==========================================================================
# the runner
# ==========================================================================
def run_cat_test(project: Optional[str] = None, seed: int = 0, quick: bool = False,
                 stages: Sequence[str] = ("family", "calibrate", "controls",
                                          "arm", "d_sweep", "deletion"),
                 out_dir: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the cat test in the registered order (prereg §10)."""
    cfg = CatTestConfig(**dict(overrides or {}))
    if quick:
        cfg = replace(cfg, n_wells=16, f_subset=3, n_items=24, n_unseen=48,
                      atoms_per_well=6, payload_dim=6, write_steps=60,
                      address_steps=80, read_steps=80, organize_steps=10,
                      quick=True)
    out = Path(out_dir) if out_dir else Path("results") / "cat_test"
    out.mkdir(parents=True, exist_ok=True)
    seeds = (0, 1, 2) if quick else (0, 1, 2, 3, 4)
    res: Dict[str, Any] = {"config": cfg.as_dict(),
                           "non_default_flags": cfg.as_flag_table(),
                           "seeds": list(seeds), "quick": bool(quick)}
    if "family" in stages:
        res["family"] = stage_family(cfg, seeds=seeds, out=out)
    if "calibrate" in stages:
        res["calibrate"] = stage_calibrate(cfg, seeds=seeds[:3], out=out)
    arms: Dict[int, Dict] = {}
    if {"controls", "arm", "deletion"} & set(stages):
        for s in seeds:
            fam = build_family(cfg, seed=s)
            arms[s] = build_physics_arm(cfg, fam, s)
    if "controls" in stages:
        res["controls"] = stage_controls(cfg, seeds=seeds, out=out, arms=arms)
    if "arm" in stages:
        res["arm"] = stage_arm(cfg, seeds=seeds, out=out, arms=arms)
    if "d_sweep" in stages:
        res["d_sweep"] = stage_d_sweep(cfg, seeds=seeds[:3], out=out)
    if "deletion" in stages:
        res["deletion"] = stage_deletion(cfg, seeds=seeds[:3], out=out, arms=arms)
    _dump(res, out / "cat_test_summary.json")
    return res
