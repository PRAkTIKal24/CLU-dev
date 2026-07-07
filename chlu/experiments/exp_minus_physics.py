"""Experiment: 'CLU minus the physics' controls (critique G2 / P6).

Runs three identical-capacity arms — CHLU (symplectic + volume-conserving),
BrokenVolumeCHLU (same leapfrog, det J != 1), UnconstrainedTwin (free residual
recurrence, no physics) — through ONE measurement protocol on the SO(2)-
degenerate circle-vacuum task (Experiment-D data), 3+ seeds, and produces the
"which component buys what" table: for each functional metric, CLU vs
broken-volume vs twin, with the delta attributed to (integrator structure |
volume conservation | nothing).

Metrics (all arm-agnostic / geometric so the twin, which has no potential and
hence no spectral masses, is measured the same way):
  - retention-vs-perturbation: half-life of a position kick along the designed
    FLAT (orbit-tangent) and STIFF (radial) directions;
  - latch: coset-angle freeze after a tangential momentum WRITE (F5 §4.1);
  - n_1/2-vs-mu: spectral masses at the settled vacuum (CHLU/broken only; the
    twin reports its step-Jacobian spectral radius as the mu-analog);
  - sleep-erosion susceptibility: does wake-sleep CD invert the vacuum
    (r*->0)? (CHLU/broken; twin sleep is inert by construction);
  - training quality: eval MSE reproducing the constant circle-vacuum orbit;
  - volume: log|det J| of one step at the vacuum (the symplecticity witness).

Runnable:
    uv run python -m chlu.experiments.exp_minus_physics --quick
    chlu exp-minus-physics [--project N] [--seed I] [--quick]

Nomenclature (F5 Def-2): inertial mass M vs spectral mass mu.
"""

import json
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.twins import build_arms
from chlu.data.circle_vacuum import generate_circle_vacuum
from chlu.experiments.goldstone_harness import (
    coset_angle,
    half_life_first_crossing,
    rollout_from,
    settle,
    spectrum_probe,
    step_jacobian,
)
from chlu.training.train import train_chlu

ARMS = ("chlu", "broken_volume", "twin")
#: deterministic per-arm key offsets (Python's hash() is per-process salted)
_ARM_IDX = {"chlu": 0, "broken_volume": 1, "twin": 2}


# ---------------------------------------------------------------------------
# Geometric measurement helpers (uniform across arms)
# ---------------------------------------------------------------------------


def _channel_dirs(q_star, dim):
    """Radial (stiff) and tangential (flat) unit vectors on channel (0, 1)."""
    r = float(jnp.sqrt(q_star[0] ** 2 + q_star[1] ** 2))
    rad = np.zeros(dim)
    tan = np.zeros(dim)
    if r > 1e-9:
        rad[0], rad[1] = float(q_star[0]) / r, float(q_star[1]) / r
        tan[0], tan[1] = -float(q_star[1]) / r, float(q_star[0]) / r
    return r, jnp.asarray(rad), jnp.asarray(tan)


def _position_retention(model, q_star, direction, kick, steps, dt, gamma):
    """Half-life of a position kick projected onto ``direction``."""
    q0 = q_star + kick * direction
    p0 = jnp.zeros_like(q_star)
    traj = rollout_from(model, q0, p0, steps=steps, dt=dt, gamma=gamma)
    dim = q_star.shape[0]
    d = np.asarray((traj[:, :dim] - q_star) @ direction)
    ret = np.abs(d) / max(abs(d[0]), 1e-20)
    return half_life_first_crossing(ret), float(ret[-1]), ret


def _latch(model, q_star, tan, kick, steps, dt, gamma):
    """Tangential momentum WRITE -> coset-angle freeze (F5 §4.1 latch)."""
    p0 = kick * tan
    traj = rollout_from(model, q_star, p0, steps=steps, dt=dt, gamma=gamma)
    dim = q_star.shape[0]
    theta = coset_angle(traj, dim)
    drift = float(abs(theta[-1] - theta[len(theta) // 2]))
    d = np.asarray((traj[:, :dim] - q_star) @ tan)
    return drift, float(d[-1]), theta


def _eval_mse(model, key, cfg, dim, dt, n_eval=32):
    """MSE reproducing the constant circle-vacuum orbit from held-out inits."""
    angles = jax.random.uniform(key, (n_eval,), minval=0.0, maxval=2.0 * jnp.pi)
    q0 = jnp.zeros((n_eval, dim))
    q0 = q0.at[:, 0].set(cfg.circle_radius * jnp.cos(angles))
    q0 = q0.at[:, 1].set(cfg.circle_radius * jnp.sin(angles))
    p0 = jnp.zeros((n_eval, dim))
    steps = cfg.eval_steps

    def roll(q, p):
        return model(q, p, steps, dt, 0.0)

    traj = jax.vmap(roll)(q0, p0)  # (n_eval, steps, 2*dim)
    target = jnp.concatenate([q0, p0], axis=1)[:, None, :]  # constant
    return float(jnp.mean((traj - target) ** 2))


def _measure_arm(name, model, train_data, key, config, dim, dt):
    """All post-training metrics for one arm on one seed.

    Retention/latch are probed from the DATA-manifold point q_data (radius R
    on the channel circle) — a well-defined common origin for every arm, with a
    well-defined coset angle/tangent — rather than each model's settled point
    (which diverges for the volume-breaking arms; that divergence is captured
    separately as the BIBO ``bounded`` diagnostic).
    """
    cfg = config.experiment_minus_physics
    k_eval = jax.random.split(key)[0]

    q_data = train_data[0, 0, :dim]
    # settle diagnostic (BIBO): does a damped rollout stay bounded?
    q_settle, p_settle = settle(
        model, q_data, dt=dt, gamma=cfg.settle_gamma, steps=cfg.settle_steps
    )
    r_star = float(jnp.sqrt(q_settle[0] ** 2 + q_settle[1] ** 2))
    bounded = bool(np.isfinite(r_star) and r_star < cfg.settle_bound * cfg.circle_radius)

    # retention/latch probed from the data-manifold point (common origin)
    _, rad, tan = _channel_dirs(q_data, dim)
    hl_flat, ret_flat_end, _ = _position_retention(
        model, q_data, tan, cfg.probe_kick, cfg.probe_steps, dt, cfg.probe_gamma
    )
    hl_stiff, ret_stiff_end, _ = _position_retention(
        model, q_data, rad, cfg.probe_kick, cfg.probe_steps, dt, cfg.probe_gamma
    )
    latch_drift, latch_dinf, _ = _latch(
        model, q_data, tan, cfg.probe_kick, cfg.probe_steps, dt, cfg.probe_gamma
    )

    # spectral masses at the settled vacuum (only meaningful if bounded and the
    # arm has a potential; twin has no Hamiltonian -> J spectral radius only)
    J = np.asarray(step_jacobian(model, q_data, jnp.zeros(dim), dt, 0.0))
    spec_radius = float(np.max(np.abs(np.linalg.eigvals(J))))
    logdetJ = float(np.log(np.abs(np.linalg.det(J)) + 1e-300))
    if name == "twin" or not bounded:
        mu_sq = np.array([np.nan] * dim)
        flat_mu2 = float("nan")
    else:
        probe = spectrum_probe(model, q_settle)
        mu_sq = np.asarray(probe.mu_sq)
        flat_mu2 = float(mu_sq[0])

    eval_mse = _eval_mse(model, k_eval, cfg, dim, dt)

    return {
        "r_star": r_star,
        "bounded": float(bounded),
        "hl_flat": hl_flat,
        "hl_stiff": hl_stiff,
        "ret_flat_end": ret_flat_end,
        "ret_stiff_end": ret_stiff_end,
        "latch_drift": latch_drift,
        "latch_dinf": latch_dinf,
        "flat_mu2": flat_mu2,
        "mu_sq": mu_sq,
        "spec_radius": spec_radius,
        "logdetJ": logdetJ,
        "eval_mse": eval_mse,
    }


def _train_one(model, train_data, key, config, cfg, dt, sleep: bool):
    """Wake-only (sleep=False) or wake-sleep (sleep=True) train_chlu wrapper."""
    conf = config
    # primary run = wake-only: push sleep beyond reach (epoch-0 event still
    # fires, matching exp-d sleep_mode='off'; inert for the twin).
    conf.training.sleep_frequency = 5 if sleep else 10**9
    model, losses, _ = train_chlu(
        model,
        train_data,
        key=key,
        config=conf,
        epochs=cfg.train_epochs,
        window_size=cfg.seq_len - 1,
        dt=dt,
    )
    return model, float(losses[-1])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_minus_physics(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: Optional[bool] = None,
) -> dict:
    """Run the minus-the-physics G2 controls; returns the results dict."""
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed

    cfg = config.experiment_minus_physics
    if quick:
        cfg.train_epochs = min(cfg.train_epochs, 60)
        cfg.n_seeds = min(cfg.n_seeds, 2)
        cfg.probe_steps = min(cfg.probe_steps, 1500)
        cfg.settle_steps = min(cfg.settle_steps, 1000)
        cfg.n_points = min(cfg.n_points, 128)
        cfg.eval_steps = min(cfg.eval_steps, 200)

    save_dir = config.project.save_dir or "results/"
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    base_seed = config.project.seed
    dim = cfg.dim
    dt = cfg.dt
    seeds = [base_seed + i for i in range(cfg.n_seeds)]

    print("\n" + "=" * 64)
    print("EXPERIMENT: CLU MINUS THE PHYSICS (G2 non-symplectic controls)")
    print("=" * 64)
    print(
        f"arms={ARMS} | dim={dim} hidden={cfg.hidden_dim} "
        f"kinetic={cfg.kinetic_energy_mode} potential={cfg.potential_type}"
    )
    print(f"seeds={seeds} epochs={cfg.train_epochs} erosion={cfg.measure_erosion}")

    per_seed = {arm: [] for arm in ARMS}
    erosion = {arm: [] for arm in ("chlu", "broken_volume")}
    param_report = None

    for s in seeds:
        key = jax.random.PRNGKey(s)
        k_data, k_build, k_train, k_meas, k_ero = jax.random.split(key, 5)
        train_data = generate_circle_vacuum(
            k_data, n_points=cfg.n_points, seq_len=cfg.seq_len,
            dim=dim, radius=cfg.circle_radius,
        )
        arms = build_arms(
            k_build, dim, cfg.hidden_dim, cfg.kinetic_energy_mode,
            cfg.potential_type, rest_mass=config.model.rest_mass,
            c=config.model.speed_of_causality,
        )
        if param_report is None:
            param_report = {"params": arms["params"], "twin_hidden": arms["twin_hidden"]}
            print(f"  param match: {arms['params']} (twin hidden={arms['twin_hidden']})")

        for arm in ARMS:
            kt = jax.random.fold_in(k_train, _ARM_IDX[arm])
            km = jax.random.fold_in(k_meas, _ARM_IDX[arm])
            model, final_loss = _train_one(
                arms[arm], train_data, kt, config, cfg, dt, sleep=False
            )
            m = _measure_arm(arm, model, train_data, km, config, dim, dt)
            m["final_loss"] = final_loss
            per_seed[arm].append(m)
            print(
                f"  [seed {s}] {arm:14s}: mse={m['eval_mse']:.4e} "
                f"logdetJ={m['logdetJ']:+.3f} r*={m['r_star']:.3f} "
                f"hl_flat={m['hl_flat']} hl_stiff={m['hl_stiff']} "
                f"latch_drift={m['latch_drift']:.2e} flat_mu2={m['flat_mu2']:.2e}"
            )

        # --- sleep-erosion susceptibility (CHLU/broken; twin sleep inert) ---
        if cfg.measure_erosion:
            arms_e = build_arms(
                k_build, dim, cfg.hidden_dim, cfg.kinetic_energy_mode,
                cfg.potential_type, rest_mass=config.model.rest_mass,
                c=config.model.speed_of_causality,
            )
            for arm in ("chlu", "broken_volume"):
                kt = jax.random.fold_in(k_ero, _ARM_IDX[arm])
                model, _ = _train_one(
                    arms_e[arm], train_data, kt, config, cfg, dt, sleep=True
                )
                q_star, _ = settle(
                    model, train_data[0, 0, :dim], dt=dt,
                    gamma=cfg.settle_gamma, steps=cfg.settle_steps,
                )
                r_star = float(jnp.sqrt(q_star[0] ** 2 + q_star[1] ** 2))
                if r_star > 1e-6:
                    mu0 = float(np.asarray(spectrum_probe(model, q_star).mu_sq)[0])
                else:
                    mu0 = float("nan")
                erosion[arm].append({"r_star": r_star, "flat_mu2": mu0})
                print(
                    f"  [seed {s}] EROSION {arm:14s}: r*={r_star:.4f} "
                    f"(R={cfg.circle_radius}) flat_mu2={mu0:.3e}"
                )

    summary = _summarize(per_seed, erosion, cfg, param_report)
    _save_outputs(per_seed, erosion, summary, save_dir, results_dir)
    _print_table(summary)
    return {"per_seed": per_seed, "erosion": erosion, "summary": summary}


# ---------------------------------------------------------------------------
# Aggregation / outputs
# ---------------------------------------------------------------------------

_SCALAR_KEYS = (
    "eval_mse", "logdetJ", "r_star", "bounded", "hl_flat", "hl_stiff",
    "ret_flat_end", "ret_stiff_end", "latch_drift", "latch_dinf",
    "flat_mu2", "spec_radius", "final_loss",
)


def _agg(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return {
        "mean": float(a.mean()) if a.size else float("nan"),
        "std": float(a.std()) if a.size else float("nan"),
        "n_finite": int(a.size),
        "n_inf": int(sum(1 for v in vals if not np.isfinite(v))),
        "per_seed": [float(v) for v in vals],
    }


def _summarize(per_seed, erosion, cfg, param_report):
    summary = {
        "arms": list(ARMS),
        "param_report": param_report,
        "config": {
            "dim": cfg.dim, "hidden": cfg.hidden_dim,
            "kinetic": cfg.kinetic_energy_mode, "potential": cfg.potential_type,
            "epochs": cfg.train_epochs, "probe_gamma": cfg.probe_gamma,
            "probe_kick": cfg.probe_kick, "circle_radius": cfg.circle_radius,
        },
        "metrics": {},
        "erosion": {},
    }
    for arm in ARMS:
        summary["metrics"][arm] = {
            k: _agg([m[k] for m in per_seed[arm]]) for k in _SCALAR_KEYS
        }
    for arm in ("chlu", "broken_volume"):
        if erosion[arm]:
            summary["erosion"][arm] = {
                "r_star": _agg([e["r_star"] for e in erosion[arm]]),
                "flat_mu2": _agg([e["flat_mu2"] for e in erosion[arm]]),
            }
    return summary


def _save_outputs(per_seed, erosion, summary, save_dir, results_dir):
    npz = {}
    for arm in ARMS:
        for i, m in enumerate(per_seed[arm]):
            npz[f"{arm}_seed{i}_mu_sq"] = np.asarray(m["mu_sq"])
            for k in _SCALAR_KEYS:
                npz[f"{arm}_seed{i}_{k}"] = np.asarray(m[k], dtype=float)
    np.savez(os.path.join(results_dir, "exp_minus_physics_metrics.npz"), **npz)
    with open(
        os.path.join(results_dir, "exp_minus_physics_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics + summary to {results_dir}/exp_minus_physics_*")


def _print_table(summary):
    """The 'which component buys what' table."""
    m = summary["metrics"]

    def cell(arm, key):
        a = m[arm][key]
        if not np.isfinite(a["mean"]):
            return "  N/A  "
        return f"{a['mean']:.3g}±{a['std']:.2g}"

    print("\n" + "=" * 78)
    print("WHICH COMPONENT BUYS WHAT  (mean±std over seeds)")
    print("=" * 78)
    header = f"{'metric':<16}{'CHLU':>18}{'broken-vol':>18}{'twin':>18}"
    print(header)
    print("-" * 78)
    rows = [
        ("eval_mse", "eval MSE (lower=better fit)"),
        ("logdetJ", "log|det J| (0=volume-cons)"),
        ("bounded", "BIBO: frac settled bounded"),
        ("r_star", "vacuum radius r*"),
        ("hl_flat", "n_1/2 flat (orbit)"),
        ("hl_stiff", "n_1/2 stiff (radial)"),
        ("latch_drift", "latch coset drift"),
        ("flat_mu2", "flat spectral mu^2"),
        ("spec_radius", "step-Jacobian |lambda|max"),
    ]
    for key, _label in rows:
        print(
            f"{key:<16}{cell('chlu', key):>18}"
            f"{cell('broken_volume', key):>18}{cell('twin', key):>18}"
        )
    if summary["erosion"]:
        print("-" * 78)
        print("sleep-erosion (wake-sleep CD): r* -> 0 means the vacuum inverted")
        for arm, blk in summary["erosion"].items():
            print(
                f"  {arm:<14}: r*={blk['r_star']['mean']:.4f}"
                f"±{blk['r_star']['std']:.3f}  "
                f"flat_mu2={blk['flat_mu2']['mean']:.3e}"
            )
    print("=" * 78 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLU minus the physics (G2)")
    parser.add_argument("--project", help="Project name (default ./results)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quick", action="store_true")
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

    run_experiment_minus_physics(
        config=config, save_dir=save_dir, models_dir=models_dir,
        seed=args.seed, quick=args.quick,
    )


if __name__ == "__main__":
    main()
