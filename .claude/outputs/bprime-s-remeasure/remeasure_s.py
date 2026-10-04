"""bprime-s-remeasure: re-estimate `s` on bprime-c6's banked cells under BOTH
conventions (with / without the alpha*||q||^2 subtraction), two estimator families.

Nothing is retrained beyond re-running the DETERMINISTIC shipped write at the same
seed + config; identity with the banked store is asserted by reproducing the banked
`s_fitted_well` / `sep` / `lambda_min` digit-for-digit.

Run:
  cd /tmp/chlu-c6 && PYTHONPATH=/tmp/chlu-c6 /tmp/rmvenv/bin/python remeasure_s.py \
      --radii 0.42 0.55 0.64 0.80 1.00 1.20 --seeds 0 1 2 --out <json>
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.factored_store import effective_s
from chlu.experiments.exp_route3_attribution import _sweep_overrides, _write_and_query


# ---------------------------------------------------------------------------
# E1: CluSystem._well_fit, verbatim, with the confinement coefficient exposed.
# (copied from chlu/core/clu_system.py::_well_fit @ be995ca == d4f56c8 == HEAD;
#  the ONLY edit is `confine` becoming an argument instead of `self.cfg.confine`.)
# ---------------------------------------------------------------------------
def well_fit_variant(system, z, confine, n_dirs=8, seed=0):
    V = system.store.V
    z = np.asarray(z, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    u = rng.normal(size=(int(n_dirs), system.store.dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    radii = np.linspace(0.15, 1.5, 12)
    pts = (z[None, None, :] + radii[None, :, None] * u[:, None, :]).reshape(-1, system.store.dim)
    vals = np.asarray(jax.vmap(V)(jnp.asarray(pts, dtype=jnp.float32)))
    v0 = float(V(jnp.asarray(z)))
    conf = float(confine) * (np.sum(pts ** 2, axis=1) - float(np.sum(z ** 2)))
    y = (vals - conf - v0).reshape(int(n_dirs), radii.size).mean(axis=0)
    best = (float("nan"), float("nan"), np.inf)
    for s_try in np.linspace(0.05, 1.2, 120):
        basis = 1.0 - np.exp(-(radii ** 2) / (2.0 * s_try ** 2))
        denom = float(np.sum(basis * basis))
        if denom <= 0:
            continue
        D = float(np.sum(basis * y) / denom)
        resid = float(np.sum((y - D * basis) ** 2))
        if resid < best[2]:
            best = (D, float(s_try), resid)
    D, s_fit, _ = best
    return max(D, 0.0), s_fit


def z_of(system, c, a):
    z = np.zeros((system.store.dim,), dtype=np.float32)
    z[: system.store.addr_dim] = c
    z[system.store.addr_dim: system.store.addr_dim + system.store.payload_dim] = a
    return z


def agg(v):
    v = [float(x) for x in v if np.isfinite(x)]
    if not v:
        return {"median": float("nan"), "mean": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v),
            "min": float(np.min(v)), "max": float(np.max(v))}


def confinement_cancellation_check(system, q, alpha):
    """Show that the deletion-difference gradient (bprime-c6's `grad_ratio`
    numerator/denominator) is INDEPENDENT of the confinement, numerically."""
    from chlu.experiments.exp_route3_attribution import _deleted_system

    m_full = system.model()
    ids, _, _ = system.codebook()
    slot = system._slot_of(int(ids[0]))
    sysk = _deleted_system(system, slot)
    m_del = sysk.model()

    # direct: grad of each system's V, and of V minus the analytic confinement
    Vf = system.store.V
    Vd = sysk.store.V

    def gfull(x):
        return jax.grad(Vf)(x)

    def gdel(x):
        return jax.grad(Vd)(x)

    def gfull_nc(x):
        return jax.grad(lambda y: Vf(y) - alpha * jnp.sum(y ** 2))(x)

    def gdel_nc(x):
        return jax.grad(lambda y: Vd(y) - alpha * jnp.sum(y ** 2))(x)

    qj = jnp.asarray(q, dtype=jnp.float32)
    d_raw = np.asarray(jax.vmap(gfull)(qj) - jax.vmap(gdel)(qj))
    d_nc = np.asarray(jax.vmap(gfull_nc)(qj) - jax.vmap(gdel_nc)(qj))
    return {"max_abs_diff_of_differences": float(np.max(np.abs(d_raw - d_nc))),
            "bit_identical": bool(np.array_equal(d_raw, d_nc)),
            "mean_norm_raw": float(np.mean(np.linalg.norm(d_raw, axis=-1))),
            "n_points": int(np.asarray(q).shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", nargs="+", type=float,
                    default=[0.42, 0.55, 0.64, 0.80, 1.00, 1.20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="/tmp/remeasure_s.json")
    ap.add_argument("--cancellation-cell", nargs=2, type=float, default=[1.00, 0.0],
                    help="(radius, seed) on which to run the gradient cancellation check")
    args = ap.parse_args()

    t_all = time.time()
    cells = []
    for r in args.radii:
        for sd in args.seeds:
            t0 = time.time()
            cell = _write_and_query("overload", "load1x_shipped", int(sd),
                                    clu_extra=_sweep_overrides(float(r)))
            if cell is None:
                cells.append({"ball_radius": float(r), "seed": int(sd),
                              "degenerate": True})
                continue
            system, ccfg = cell.system, cell.ccfg
            alpha = float(ccfg.confine)
            ids, centers, pays = system.codebook()
            zs = [z_of(system, c, a) for c, a in zip(centers, pays)]

            # --- the banked estimator, called through the SHIPPED entry point --
            Ds_ship, ss_ship = system.well_fits()
            # --- E1, both conventions, item by item ---------------------------
            e1_sub, e1_unsub, d_sub, d_unsub = [], [], [], []
            for z in zs:
                D1, s1 = well_fit_variant(system, z, alpha, seed=ccfg.seed)
                D0, s0 = well_fit_variant(system, z, 0.0, seed=ccfg.seed)
                e1_sub.append(s1)
                e1_unsub.append(s0)
                d_sub.append(D1)
                d_unsub.append(D0)
            # --- E2 (the orgdiv-cat-test estimator), both conventions ---------
            e2_sub, e2_unsub, r2_sub, r2_unsub = [], [], [], []
            for z in zs:
                a = effective_s(system.store.V, z, s_hint=0.3, confine=alpha,
                                seed=int(ccfg.seed))
                b = effective_s(system.store.V, z, s_hint=0.3, confine=0.0,
                                seed=int(ccfg.seed))
                e2_sub.append(a["s"])
                e2_unsub.append(b["s"])
                r2_sub.append(a["r2"])
                r2_unsub.append(b["r2"])

            from chlu.experiments.exp_route3_attribution import _min_sep
            row = {
                "ball_radius": float(r), "seed": int(sd), "alpha_confine": alpha,
                "admissible": bool(cell.admissible), "reason": cell.reason,
                "lambda_min": float(cell.lam_min),
                "endpoint_write_loss_max": float(cell.endpoint_loss),
                "n_live": int(len(ids)), "sep": float(_min_sep(centers)),
                "atom_width": float(ccfg.atom_width),
                "s_fitted_well_shipped_call": float(np.median(ss_ship)),
                "well_depth_median_shipped_call": float(np.median(Ds_ship)),
                "E1_subtracted_per_item": [float(x) for x in e1_sub],
                "E1_unsubtracted_per_item": [float(x) for x in e1_unsub],
                "E1_subtracted": agg(e1_sub), "E1_unsubtracted": agg(e1_unsub),
                "E1_depth_subtracted": agg(d_sub), "E1_depth_unsubtracted": agg(d_unsub),
                "E2_subtracted_per_item": [float(x) for x in e2_sub],
                "E2_unsubtracted_per_item": [float(x) for x in e2_unsub],
                "E2_subtracted": agg(e2_sub), "E2_unsubtracted": agg(e2_unsub),
                "E2_r2_subtracted": agg(r2_sub), "E2_r2_unsubtracted": agg(r2_unsub),
                "E1_matches_shipped_bitwise": bool(
                    np.array_equal(np.asarray(e1_sub, dtype=float),
                                   np.asarray(ss_ship, dtype=float))),
                "wall_s": time.time() - t0,
            }
            if (abs(float(r) - float(args.cancellation_cell[0])) < 1e-9
                    and int(sd) == int(args.cancellation_cell[1])):
                row["gradient_confinement_cancellation"] = (
                    confinement_cancellation_check(
                        system, np.asarray(cell.qs.q0, dtype=np.float32)[:64], alpha))
            cells.append(row)
            print(f"[R={r} s{sd}] adm={cell.admissible} sep={row['sep']:.4f} "
                  f"E1(sub)={row['E1_subtracted']['median']:.6f} "
                  f"E1(unsub)={row['E1_unsubtracted']['median']:.6f} "
                  f"E2(sub)={row['E2_subtracted']['median']:.6f} "
                  f"E2(unsub)={row['E2_unsubtracted']['median']:.6f} "
                  f"[{row['wall_s']:.1f}s]", flush=True)

    out = {"probe": "bprime-s-remeasure: alpha||q||^2 subtraction convention",
           "family": "overload", "arm": "load1x_shipped",
           "radii": args.radii, "seeds": args.seeds,
           "jax": jax.__version__, "wall_s": time.time() - t_all, "cells": cells}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, f"in {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
