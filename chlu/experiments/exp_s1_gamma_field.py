"""Experiment S1: the trash-region Pareto pilot (learned friction field).

Thread-1's falsifiable experiment made real: a signal attractor (Figure-8
lemniscate, Exp-A machinery) plus structured noise injected from a localized
off-attractor cluster ("the garbage source"). Compares four damping arms:

    (i)   global gamma        — constant friction sweep (the Pareto trade-off
                                curve: forgetting everything a little)
    (ii)  energy governor     — Exp-B's active brake at the learned energy floor
    (iii) learned gamma_phi   — K contrastively-trained holes (F5 Def-5;
                                wake protects data, sleep damns hallucinations)
    (iv)  oracle fixed hole   — hand-placed at the known noise locus (control:
                                the placement the learned field should find)

Metric: signal-retention vs noise-rejection Pareto.
    retention — trajectory fidelity of a long clean free-run on the attractor
                (coverage of the true curve by the final cycles + kinetic-energy
                survival);
    rejection — decay of injected perturbations at the noise locus (return of
                the state to the attractor + dissipation of excess energy).
Prediction on file (brainstorm Thread 1): (iii) Pareto-dominates (i)/(ii) —
"global friction forgets everything a little; a horizon forgets garbage
completely and memories not at all."

Also reports the C1 comparison (mo-deep-read §5): for each learned hole, the
local spectral masses mu(c_k) (via the goldstone harness spectrum_probe) and
the ratio of the learned strength gamma_k to the critical-damping forgetting
optimum 2*dt*mu — do learned holes find the fast-forgetting regime?
(Measured, not forced: training.friction_field_c1_lambda defaults to 0.)

Design notes / assumptions (documented):
    - Both the base and the learned-field models train with the replay buffer
      partially seeded at the noise locus: the environment EXPOSES the garbage
      source to the sleep phase of every arm; only arm (iii) can convert that
      exposure into friction placement (exposure != placement; arm (iv) gets
      the placement for free — that is what makes it the oracle).
    - Injection draws are shared (same PRNG stream) across all arms and seeds:
      paired comparisons.
    - Kinetic-energy retention uses ||p||^2 as the motion proxy (exact for the
      newtonian_identity default; a monotone proxy otherwise).

Runnable directly (documented script entry):
    uv run python -m chlu.experiments.exp_s1_gamma_field --quick
    uv run python -m chlu.experiments.exp_s1_gamma_field --project myproj
or via the CLI: ``chlu exp-s1 [--project N] [--seed I] [--quick]``.
"""

import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.friction_field import FrictionField
from chlu.data.figure8 import generate_figure8
from chlu.experiments.goldstone_harness import spectrum_probe
from chlu.training.train import train_chlu
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.plotting import plot_gamma_field_landscape, plot_s1_pareto


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def min_dist_to_curve(q_points: jnp.ndarray, curve_q: jnp.ndarray) -> jnp.ndarray:
    """Per-point Euclidean distance to the nearest reference-curve point.

    Args:
        q_points: (N, dim) positions.
        curve_q: (M, dim) reference attractor curve.

    Returns:
        (N,) distances.
    """
    d2 = jnp.sum((q_points[:, None, :] - curve_q[None, :, :]) ** 2, axis=-1)
    return jnp.sqrt(jnp.min(d2, axis=1))


def retention_scores(
    traj: jnp.ndarray, curve_q: jnp.ndarray, steps_per_cycle: int, dim: int = 2
) -> dict:
    """Signal-retention of a clean free-run (higher = better).

    coverage: exp(-mean_curve min-dist to the final two cycles / curve scale)
        — 1 when the rollout still traverses the whole attractor shape, small
        when it froze (a stopped state covers one point) or collapsed.
    ke_ratio: mean ||p||^2 over the last cycle / first cycle, clipped to [0,1]
        — motion survival (0 = the orbit was damped to a halt).
    """
    q_roll = traj[-2 * steps_per_cycle :, :dim]
    cov_dist = float(jnp.mean(min_dist_to_curve(curve_q, q_roll)))
    scale = float(jnp.mean(jnp.linalg.norm(curve_q, axis=1)))
    coverage = float(np.exp(-cov_dist / scale))

    ke = jnp.sum(traj[:, dim:] ** 2, axis=1)
    ke_first = float(jnp.mean(ke[:steps_per_cycle]))
    ke_last = float(jnp.mean(ke[-steps_per_cycle:]))
    ke_ratio = float(np.clip(ke_last / max(ke_first, 1e-12), 0.0, 1.0))
    return {"coverage": coverage, "ke_ratio": ke_ratio}


def rejection_scores(
    rollout_fn,
    model: CHLU,
    injections: tuple,
    curve_q: jnp.ndarray,
    kick_steps: int,
    target_energy: float,
    dim: int = 2,
) -> dict:
    """Noise-rejection of injected perturbations (higher = better).

    For each injected state z0 = (q0, p0) at the noise locus:
        pos: 1 - mean(final-quarter min-dist to curve)/dist(q0)  — did the
             state come back to the attractor?
        energy: 1 - mean(final-quarter excess H)/excess H(z0)    — was the
             injected energy above the learned floor dissipated?
    Both clipped to [0, 1] and averaged over injections.
    """
    q0s, p0s = injections
    tail = max(1, kick_steps // 4)
    pos_scores, energy_scores = [], []
    for q0, p0 in zip(q0s, p0s, strict=True):
        traj = rollout_fn(q0, p0, kick_steps)
        d = min_dist_to_curve(traj[:, :dim], curve_q)
        d0 = float(min_dist_to_curve(q0[None, :], curve_q)[0])
        d_end = float(jnp.mean(d[-tail:]))
        pos_scores.append(float(np.clip(1.0 - d_end / max(d0, 1e-9), 0.0, 1.0)))

        H = jax.vmap(model.H)(traj[:, :dim], traj[:, dim:])
        exc0 = float(model.H(q0, p0)) - target_energy
        exc_end = float(jnp.mean(H[-tail:])) - target_energy
        if exc0 > 1e-6:
            energy_scores.append(float(np.clip(1.0 - exc_end / exc0, 0.0, 1.0)))
    return {
        "rejection_pos": float(np.mean(pos_scores)),
        "rejection_energy": float(np.mean(energy_scores))
        if energy_scores
        else float("nan"),
    }


def field_placement_scores(
    model: CHLU, curve_q: jnp.ndarray, noise_q: jnp.ndarray
) -> dict:
    """Mechanism check: mean gamma_phi on the data curve vs on the noise cluster."""
    field = model.friction_field
    return {
        "gamma_on_curve": float(jnp.mean(jax.vmap(field)(curve_q))),
        "gamma_on_noise": float(jnp.mean(jax.vmap(field)(noise_q))),
    }


def c1_hole_report(model: CHLU, dt: float) -> list:
    """Per-hole C1 comparison: learned gamma_k vs critical damping 2*dt*mu(c_k).

    Spectral masses via the goldstone harness ``spectrum_probe`` (F5 Def-2:
    these are spectral masses mu, not inertial masses M).
    """
    centers, radii, strengths = model.friction_field.hole_params()
    rows = []
    for k in range(centers.shape[0]):
        probe = spectrum_probe(model, centers[k])
        mu = np.sqrt(np.clip(np.asarray(probe.mu_sq), 0.0, None))
        gamma_star = 2.0 * dt * float(np.mean(mu))
        gamma_k = float(strengths[k])
        rows.append(
            {
                "center": np.asarray(centers[k]),
                "radius": float(radii[k]),
                "gamma_k": gamma_k,
                "mu_mean": float(np.mean(mu)),
                "mu_min": float(np.min(mu)),
                "mu_max": float(np.max(mu)),
                "gamma_star_2eps_mu": gamma_star,
                "ratio_gamma_over_star": gamma_k / max(gamma_star, 1e-12),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def run_experiment_s1(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    use_pretrained: Optional[bool] = None,
    seed: Optional[int] = None,
    train_epochs: Optional[int] = None,
    seeds: Optional[list] = None,
) -> dict:
    """Experiment S1: signal-retention vs noise-rejection Pareto (see module docstring).

    Returns a dict with per-arm records, C1 hole reports, and placement
    scores (also written to results/exp_s1_metrics.npz).
    """
    # ------------------------------------------------------------------ config
    if config is None:
        config = get_default_config()

    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed
    if use_pretrained is not None:
        config.experiment_s1.use_pretrained = use_pretrained
    if train_epochs is not None:
        config.experiment_s1.train_epochs = train_epochs
    if seeds is not None:
        config.experiment_s1.seeds = list(seeds)

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "../models")
    cfg = config.experiment_s1
    tcfg = config.training
    dt = cfg.dt
    dim = 2  # Figure-8 position space
    steps_per_cycle = int(2 * np.pi / dt)

    print("\n" + "=" * 60)
    print("EXPERIMENT S1: Trash-Region Pareto (learned friction field)")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print(f"\n[1/4] Data: Figure-8, {cfg.n_train_cycles} train cycles, dt={dt}")
    data_key = jax.random.PRNGKey(config.project.seed)
    train_data = generate_figure8(data_key, n_cycles=cfg.n_train_cycles, dt=dt)
    curve_q = generate_figure8(data_key, n_cycles=1, dt=dt)[:, :dim]
    q0_clean, p0_clean = train_data[-1, :dim], train_data[-1, dim:]

    noise_center = jnp.asarray(cfg.noise_center)
    # Shared injection draws (paired comparisons across arms/seeds)
    inj_key = jax.random.PRNGKey(config.project.seed + 777)
    kq, kp = jax.random.split(inj_key)
    inj_q = noise_center + cfg.noise_q_std * jax.random.normal(
        kq, (cfg.n_injections, dim)
    )
    inj_p = cfg.noise_p_std * jax.random.normal(kp, (cfg.n_injections, dim))
    injections = (inj_q, inj_p)

    # Replay-buffer garbage exposure (same for every trained model)
    n_seed_states = int(cfg.buffer_noise_frac * tcfg.buffer_capacity)
    print(
        f"  Noise locus {np.asarray(noise_center)}, {cfg.n_injections} injections, "
        f"{n_seed_states} buffer seed states ({cfg.buffer_noise_frac:.0%} of buffer)"
    )

    def _negative_seeds(key):
        kq_, kp_ = jax.random.split(key)
        q_seed = noise_center + cfg.noise_q_std * jax.random.normal(
            kq_, (n_seed_states, dim)
        )
        p_seed = cfg.noise_p_std * jax.random.normal(kp_, (n_seed_states, dim))
        return q_seed, p_seed

    # -------------------------------------------------------------- helpers
    def _train_or_load(model, name, train_key, buf_key):
        path = os.path.join(models_dir, f"{name}.pkl")
        if cfg.use_pretrained and os.path.exists(path):
            model, metadata = load_checkpoint(path, model)
            print(f"    loaded {name} (pretrained)")
            return model, metadata.get("target_energy")
        model, losses, target_energy = train_chlu(
            model,
            train_data,
            key=train_key,
            config=config,
            epochs=cfg.train_epochs,
            sleep_steps=cfg.sleep_steps,
            window_size=cfg.window_size,
            dt=dt,
            negative_seed_states=_negative_seeds(buf_key),
        )
        save_checkpoint(
            model,
            path,
            epoch=cfg.train_epochs,
            loss=float(losses[-1]),
            config=config,
            target_energy=target_energy,
        )
        print(f"    trained {name}: final wake loss {float(losses[-1]):.5f}")
        return model, target_energy

    def _rollout_fn(model, gamma=0.0, governor_target=None):
        if governor_target is not None:
            return lambda q0, p0, steps: model.governed_rollout(
                q0, p0, steps, dt, governor_target, cfg.governor_sensitivity
            )
        return lambda q0, p0, steps: model(q0, p0, steps, dt, gamma)

    def _eval_arm(model, target_energy, gamma=0.0, governor_target=None):
        roll = _rollout_fn(model, gamma=gamma, governor_target=governor_target)
        ret = retention_scores(
            roll(q0_clean, p0_clean, cfg.eval_clean_steps), curve_q, steps_per_cycle
        )
        rej = rejection_scores(
            roll, model, injections, curve_q, cfg.eval_kick_steps, target_energy
        )
        return {**ret, **rej}

    # ------------------------------------------------------------ arms loop
    print(f"\n[2/4] Training + evaluating arms over seeds {cfg.seeds}...")
    arm_records, c1_records, placement_records = [], [], []
    heatmap_models = {}  # first-seed learned/oracle models for plotting

    for s in cfg.seeds:
        print(f"\n  --- seed {s} ---")
        skey = jax.random.PRNGKey(s)
        k_model, k_train, k_buf = jax.random.split(skey, 3)

        # Base model (arms i, ii, iv share its V_theta)
        base = CHLU(
            dim=dim,
            hidden=cfg.hidden_dim,
            rest_mass=config.model.rest_mass,
            c=config.model.speed_of_causality,
            kinetic_mode=cfg.kinetic_energy_mode,
            potential_type="mlp",
            key=k_model,
        )
        base, base_E = _train_or_load(base, f"s1_base_seed{s}", k_train, k_buf)

        # (i) global-gamma sweep
        for g in cfg.global_gamma_sweep:
            scores = _eval_arm(base, base_E, gamma=float(g))
            arm_records.append(
                {
                    "arm": "global_gamma",
                    "label": f"gamma={g}",
                    "seed": s,
                    "gamma": float(g),
                    **scores,
                }
            )
            print(f"    (i)  gamma={g}: {scores}")

        # (ii) governor
        scores = _eval_arm(base, base_E, governor_target=base_E)
        arm_records.append(
            {"arm": "governor", "label": "governor", "seed": s, **scores}
        )
        print(f"    (ii) governor: {scores}")

        # (iii) learned gamma_phi, one model per K
        for K in cfg.learned_k_values:
            kf, km, kt, kb = jax.random.split(jax.random.fold_in(skey, 1000 + K), 4)
            field = FrictionField(
                dim,
                k=K,
                gamma_max=tcfg.friction_field_gamma_max,
                width=tcfg.friction_field_width,
                init_radius=tcfg.friction_field_init_radius,
                init_strength=tcfg.friction_field_init_strength,
                init_center_scale=tcfg.friction_field_init_center_scale,
                gate=tcfg.friction_field_gate,
                trainable=True,
                key=kf,
            )
            learned = CHLU(
                dim=dim,
                hidden=cfg.hidden_dim,
                rest_mass=config.model.rest_mass,
                c=config.model.speed_of_causality,
                kinetic_mode=cfg.kinetic_energy_mode,
                potential_type="mlp",
                friction_field=field,
                key=km,
            )
            learned, learned_E = _train_or_load(
                learned, f"s1_learned_k{K}_seed{s}", kt, kb
            )
            scores = _eval_arm(learned, learned_E, gamma=0.0)
            arm_records.append(
                {"arm": f"learned_k{K}", "label": f"learned K={K}", "seed": s, **scores}
            )
            print(f"    (iii) learned K={K}: {scores}")

            placement = field_placement_scores(learned, curve_q, inj_q)
            placement_records.append({"arm": f"learned_k{K}", "seed": s, **placement})
            print(f"          placement: {placement}")
            for row in c1_hole_report(learned, dt):
                c1_records.append({"arm": f"learned_k{K}", "seed": s, **row})
                print(
                    f"          C1 hole @ {np.round(row['center'], 3)}: "
                    f"gamma_k={row['gamma_k']:.4f} vs 2*dt*mu={row['gamma_star_2eps_mu']:.4f} "
                    f"(ratio {row['ratio_gamma_over_star']:.2f})"
                )
            if s == cfg.seeds[0]:
                heatmap_models[f"learned_k{K}"] = learned

        # (iv) oracle fixed hole on the base model
        oracle_field = FrictionField(
            dim,
            gamma_max=tcfg.friction_field_gamma_max,
            width=cfg.oracle_width,  # harder horizon: the frozen control's
            # tail must not reach the curve (it cannot retreat like a learned hole)
            centers=noise_center[None, :],
            init_radius=cfg.oracle_radius,
            init_strength=cfg.oracle_strength,
            gate=tcfg.friction_field_gate,
            trainable=False,
        )
        oracle = eqx.tree_at(
            lambda m: m.friction_field,
            base,
            replace=oracle_field,
            is_leaf=lambda x: x is None,
        )
        scores = _eval_arm(oracle, base_E, gamma=0.0)
        arm_records.append(
            {"arm": "oracle", "label": "oracle hole", "seed": s, **scores}
        )
        print(f"    (iv) oracle hole: {scores}")
        if s == cfg.seeds[0]:
            heatmap_models["oracle"] = oracle

    # ------------------------------------------------------------- outputs
    print("\n[3/4] Plots...")
    for name, model in heatmap_models.items():
        plot_gamma_field_landscape(
            model,
            os.path.join(save_dir, f"exp_s1_field_{name}.png"),
            trajectory=train_data,
            noise_center=np.asarray(noise_center),
            title=f"S1 friction field — {name} (seed {cfg.seeds[0]})",
        )
    plot_s1_pareto(
        arm_records,
        os.path.join(save_dir, "exp_s1_pareto.png"),
        retention_key="coverage",
        rejection_key="rejection_pos",
    )
    plot_s1_pareto(
        arm_records,
        os.path.join(save_dir, "exp_s1_pareto_energy.png"),
        retention_key="ke_ratio",
        rejection_key="rejection_energy",
    )

    print("[4/4] Metrics...")
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "exp_s1_metrics.npz")
    np.savez(
        metrics_path,
        arm=np.array([r["arm"] for r in arm_records]),
        label=np.array([r["label"] for r in arm_records]),
        seed=np.array([r["seed"] for r in arm_records]),
        coverage=np.array([r["coverage"] for r in arm_records]),
        ke_ratio=np.array([r["ke_ratio"] for r in arm_records]),
        rejection_pos=np.array([r["rejection_pos"] for r in arm_records]),
        rejection_energy=np.array([r["rejection_energy"] for r in arm_records]),
        gamma=np.array([r.get("gamma", np.nan) for r in arm_records]),
        c1_arm=np.array([r["arm"] for r in c1_records]),
        c1_seed=np.array([r["seed"] for r in c1_records]),
        c1_center=np.array([r["center"] for r in c1_records])
        if c1_records
        else np.zeros((0, dim)),
        c1_radius=np.array([r["radius"] for r in c1_records]),
        c1_gamma_k=np.array([r["gamma_k"] for r in c1_records]),
        c1_gamma_star=np.array([r["gamma_star_2eps_mu"] for r in c1_records]),
        c1_ratio=np.array([r["ratio_gamma_over_star"] for r in c1_records]),
        c1_mu_mean=np.array([r["mu_mean"] for r in c1_records]),
        placement_arm=np.array([r["arm"] for r in placement_records]),
        placement_seed=np.array([r["seed"] for r in placement_records]),
        gamma_on_curve=np.array([r["gamma_on_curve"] for r in placement_records]),
        gamma_on_noise=np.array([r["gamma_on_noise"] for r in placement_records]),
        noise_center=np.asarray(noise_center),
        dt=dt,
        seeds=np.array(cfg.seeds),
        train_epochs=cfg.train_epochs,
        gamma_max=tcfg.friction_field_gamma_max,
        protect_lambda=tcfg.friction_field_protect_lambda,
        hallu_lambda=tcfg.friction_field_hallu_lambda,
        hallu_gate=tcfg.friction_field_hallu_gate,
        c1_lambda=tcfg.friction_field_c1_lambda,
    )
    print(f"  Saved metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT S1 COMPLETE!")
    print("=" * 60 + "\n")

    return {"arms": arm_records, "c1": c1_records, "placement": placement_records}


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment S1: trash-region Pareto pilot (gamma-field)"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed (project-level)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke mode (1 seed, 60 epochs, short eval)",
    )
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
        config.experiment_s1.train_epochs = 60
        config.experiment_s1.seeds = config.experiment_s1.seeds[:1]
        config.experiment_s1.eval_clean_steps = 1000
        config.experiment_s1.n_injections = 8

    run_experiment_s1(
        config=config, save_dir=save_dir, models_dir=models_dir, seed=args.seed
    )


if __name__ == "__main__":
    main()
