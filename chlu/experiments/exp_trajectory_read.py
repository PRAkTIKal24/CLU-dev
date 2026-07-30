"""C2W1 `trainability-spike` — end-to-end gradients and the learned-psi trajectory read.

Four parts, run in the PREREG'd compute order
(``.claude/outputs/trainability-spike/PREREG.md`` §5):

``--part a``
    **A1-A2 gradcheck.** Implicit (DEQ) gradients through the shipped dissipative
    settle vs (a) truncated unroll at the theorist's registered depth and (b)
    re-settled central finite differences, on a controlled toy with a known
    answer (``GaussianWellsPotential``). Also the ``(gamma, dt)``-independence
    check, ``||p*||``, the ridge bias, and the Q3.5 conditioning triple.

``--part e2e``
    **Acceptance half 1.** Gradients flow ``query -> phi -> settle -> psi ->
    loss`` on the real harness store, at a measured wall-clock per training step,
    scored against the budget declared in PREREG §3. Also measures the PREREG §4c
    prediction that the settled-point read sends **no** gradient to ``phi``.

``--part stage0``
    ⭐ **The blocking axis-liveness gate** (Hub amendment, monitor #10). Does the
    strided trajectory carry information the settled point does not, at the
    **healthy** S0 geometry (``sep/sigma_q = 6.83``)? Probes (linear + kNN) on
    matched-capacity feature sets ``q0_only`` / ``endpoints`` / ``full``, over a
    sweep of query ambiguity x ``traj_stride`` x ``gamma_read``. Part B does not
    run unless this passes its registered gate.

``--part b``
    **The point-vs-trajectory ablation** — matched psi family, matched parameter
    count, matched bytes, matched phi, matched seeds — plus the **trajectory
    launder** (``eval/dividend.py``) and the blank-store control, which become
    mandatory the moment a learned psi can see the address block.

Run by module invocation (this wave's CLI file is owned by ``memory-gym-v0``;
**no CLI hook is added**)::

    PYTHONPATH=. python -u -m chlu.experiments.exp_trajectory_read --part a --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np

OUT_DIR = os.environ.get(
    "TRAJ_READ_OUT",
    os.path.join(os.path.expanduser("~"), "Desktop", "CHLU", ".claude", "outputs",
                 "trainability-spike"),
)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _save(name: str, payload: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2)
    return path


# ==========================================================================
# PART A — implicit/DEQ gradients through the settle
# ==========================================================================
def part_a(seed: int = 0, quick: bool = False) -> dict:
    """A1-A2: gradcheck implicit vs truncated unroll vs finite differences.

    Runs in **float64** (enabled here, never at module import — the repo's
    x64-at-import hazard, handover §7.2).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp

    from chlu.core.implicit_grad import (
        SettleSpec,
        implicit_grad,
        implicit_settle,
        ridge_alarm,
        settle_forward,
        settle_telemetry,
        theory_ridge,
        toy_model,
        unroll_grad,
    )

    # -- the controlled toy: 4 Gaussian wells on a ring (the theory's family) --
    n_w, dim = 4, 2
    ang = 2.0 * np.pi * np.arange(n_w) / n_w
    centers = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    amp0 = np.array([1.0, 0.9, 1.1, 0.95])
    s_well, alpha = 0.35, 0.05

    def build(amp):
        return toy_model(centers, jnp.asarray(amp), s=s_well, alpha=alpha)

    q0 = jnp.asarray(centers[0] + np.array([0.2, -0.15]))
    p0 = jnp.zeros_like(q0)

    def loss_fn(q):
        return 0.5 * jnp.sum(q**2)

    model = build(amp0)
    N_settle = 300 if quick else 1500
    spec = SettleSpec(steps=N_settle, dt=0.05, gamma=0.05, ridge=0.0)

    # ---- settle quality: ||p*||, residual, lambda_min (the Q3.5 triple) ----
    q_star, p_star = settle_forward(model, q0, p0, spec)
    tele = settle_telemetry(model, q_star[None, :], centers=centers, ridge=0.0)
    residual = float(tele["residual"][0])
    lam_min = float(tele["lambda_min"][0])
    lam_med = float(tele["lambda_median"][0])
    p_star_norm = float(jnp.linalg.norm(p_star))

    def _flat(tree):
        leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
        return np.concatenate([np.asarray(x).ravel() for x in leaves]) if leaves else np.zeros(0)

    def _amp_grad(tree):
        """The differentiated parameter is ``V.amp``; pull just that leaf."""
        return np.asarray(tree.potential_net.amp)

    # ---- the implicit gradient ----
    t0 = time.time()
    g_imp = _amp_grad(implicit_grad(model, q0, p0, spec, loss_fn))
    t_imp = time.time() - t0

    # ---- finite differences: RE-SETTLE at theta +/- h (the ground truth) ----
    h = 1e-5
    g_fd = np.zeros_like(amp0)
    t0 = time.time()
    for i in range(n_w):
        ap, am = amp0.copy(), amp0.copy()
        ap[i] += h
        am[i] -= h
        qp, _ = settle_forward(build(ap), q0, p0, spec)
        qm, _ = settle_forward(build(am), q0, p0, spec)
        g_fd[i] = float((loss_fn(qp) - loss_fn(qm)) / (2.0 * h))
    t_fd = time.time() - t0

    def relerr(a, b):
        b = np.asarray(b, dtype=float)
        return float(np.linalg.norm(np.asarray(a, dtype=float) - b) / max(np.linalg.norm(b), 1e-300))

    # ---- truncated unroll at the theorist's registered depths ----
    depths = [180, 270, 449] if not quick else [30, 60]
    unroll = {}
    for k in depths:
        t0 = time.time()
        g_k = _amp_grad(unroll_grad(model, q0, p0, spec, loss_fn, retain=k))
        dt_k = time.time() - t0
        unroll[str(k)] = {
            "rel_err_vs_implicit": relerr(g_k, g_imp),
            "rel_err_vs_fd": relerr(g_k, g_fd),
            "wall_s": dt_k,
            "grad": np.asarray(g_k).tolist(),
        }
    # full backprop through the whole settle (the k = N limit)
    g_full = _amp_grad(unroll_grad(model, q0, p0, spec, loss_fn, retain=None))
    unroll["full_%d" % N_settle] = {
        "rel_err_vs_implicit": relerr(g_full, g_imp),
        "rel_err_vs_fd": relerr(g_full, g_fd),
        "grad": np.asarray(g_full).tolist(),
    }

    # ---- (gamma, dt) independence of the implicit answer (theory Q1.3) ----
    # ⚠ The step budget must use the theorist's TWO-branch rho (Q4.2), not the
    # underdamped-only `N = 2 ln(1/tol)/gamma`: at (gamma=0.3, dt=0.02) the well
    # mode is OVERDAMPED (lambda_crit = gamma^2 m / (2(2-gamma) dt^2) = 66.2 >
    # lambda = 8.21) and the naive budget is ~16x too short. Both budgets are
    # reported so the failure of the naive one is on the record.
    def rho_two_branch(gam, dtv, lam, m=1.0):
        lam_crit = gam**2 * m / (2.0 * (2.0 - gam) * dtv**2)
        under = np.sqrt(max(1.0 - gam, 0.0))
        over = 1.0 - (2.0 - gam) * dtv**2 * lam / (2.0 * gam * m)
        return (under if lam > lam_crit else max(over, under)), lam_crit

    grid = []
    for gam in ([0.05] if quick else [0.02, 0.05, 0.1, 0.3]):
        for dtv in ([0.05] if quick else [0.02, 0.05, 0.1]):
            rho, lam_crit = rho_two_branch(gam, dtv, lam_min)
            n_naive = int(np.ceil(2.0 * np.log(1e12) / gam))
            n_needed = int(np.ceil(np.log(1e12) / max(-np.log(rho), 1e-12)))
            for tag, nsteps in (("naive", n_naive), ("two_branch", n_needed)):
                sp = SettleSpec(steps=min(nsteps, 20000), dt=dtv, gamma=gam)
                g = _amp_grad(implicit_grad(model, q0, p0, sp, loss_fn))
                qs, ps = settle_forward(model, q0, p0, sp)
                grid.append({
                    "gamma": gam, "dt": dtv, "budget": tag,
                    "steps": int(min(nsteps, 20000)), "rho": float(rho),
                    "lambda_crit": float(lam_crit),
                    "overdamped": bool(lam_min < lam_crit),
                    "grad": np.asarray(g).tolist(),
                    "p_star_norm": float(jnp.linalg.norm(ps)),
                    "residual": float(jnp.linalg.norm(
                        jax.grad(lambda z: jnp.reshape(model.potential_net(z), ()))(qs)))})

    def _spread(rows):
        if not rows:
            return float("nan")
        G = np.array([r["grad"] for r in rows])
        return float(np.max(np.abs(G - G.mean(axis=0, keepdims=True))) /
                     max(np.max(np.abs(G)), 1e-300))

    spread = _spread([r for r in grid if r["budget"] == "two_branch"])
    spread_naive = _spread([r for r in grid if r["budget"] == "naive"])

    # ---- the ridge (never silently enabled) ----
    lam_r = theory_ridge(0.05, 0.05, 400, 1e-3, 1.0)
    g_ridged = _amp_grad(implicit_grad(model, q0, p0,
                                       replace(spec, ridge=lam_r), loss_fn))
    ridge_bias = relerr(g_ridged, g_imp)

    # ---- d q*/d q0 = 0 (PREREG §4c, the structural claim) ----
    def q_star_of_q0(z):
        return jnp.sum(implicit_settle(model, z, p0, spec) ** 2)

    dq0_implicit = float(jnp.linalg.norm(jax.grad(q_star_of_q0)(q0)))

    def _dq0_unrolled(nsteps: int) -> float:
        sp = replace(spec, steps=int(nsteps))

        def f(z):
            qs, _ = settle_forward(model, z, p0, sp)
            return jnp.sum(qs**2)

        return float(jnp.linalg.norm(jax.grad(f)(q0)))

    dq0_unrolled = _dq0_unrolled(N_settle)
    # Verify the theory's geometric-death law rho^N on the SAME object.
    rho_addr = float(np.sqrt(1.0 - spec.gamma))
    ladder = [50, 100, 200, 400, 800, 1500] if not quick else [50, 100, 200]
    dq0_ladder = []
    for n in ladder:
        v = _dq0_unrolled(n)
        dq0_ladder.append({"N": n, "norm": v, "rho_pow_N": float(rho_addr**n)})

    res = {
        "toy": {"n_wells": n_w, "dim": dim, "s": s_well, "alpha": alpha,
                "amp": amp0.tolist(), "q0": np.asarray(q0).tolist(),
                "N_settle": N_settle, "dtype": "float64"},
        "settle": {"p_star_norm": p_star_norm, "residual": residual,
                   "lambda_min": lam_min, "lambda_median": lam_med,
                   "cond": float(tele["cond"][0]),
                   "basin": int(tele["basin"][0]),
                   "d_nearest": float(tele["d_nearest"][0])},
        "grad_implicit": np.asarray(g_imp).tolist(),
        "grad_fd": g_fd.tolist(),
        "rel_err_implicit_vs_fd": relerr(g_imp, g_fd),
        "wall_s_implicit": t_imp,
        "wall_s_fd_4param": t_fd,
        "unroll": unroll,
        "gamma_dt_grid": grid,
        "gamma_dt_spread": spread,
        "gamma_dt_spread_naive_budget": spread_naive,
        "ridge": {"lambda_ridge": lam_r, "rel_bias_vs_unridged": ridge_bias,
                  "alarm": bool(ridge_alarm(lam_r, [lam_med]))},
        "dq_star_dq0": {"implicit": dq0_implicit, "unrolled_full": dq0_unrolled,
                        "rho_address": rho_addr, "ladder": dq0_ladder},
    }
    return res


# ==========================================================================
# main
# ==========================================================================
def main(argv: Optional[list] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="a", choices=["a", "e2e", "stage0", "b", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    out: Dict[str, Any] = {"part": args.part, "seed": args.seed, "quick": args.quick}
    t0 = time.time()
    if args.part in ("a", "all"):
        out["A"] = part_a(seed=args.seed, quick=args.quick)
        print(json.dumps(_jsonable(out["A"]), indent=2)[:4000])
    out["wall_s"] = time.time() - t0
    path = _save(f"exp_trajectory_read_{args.part}_seed{args.seed}.json", out)
    print(f"\n[exp_trajectory_read] wrote {path}  ({out['wall_s']:.1f} s)")
    return out


if __name__ == "__main__":
    main()
