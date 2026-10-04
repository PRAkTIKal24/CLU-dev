"""Experiment D: SO(2) Goldstone memory (V2's core empirical apparatus).

Trains a small CLU on an SO(2)-degenerate task (a circle of attractors in the
(q0, q1) channel plane) and measures the F5 §3.3–§3.4 predictions on the
learned potential: the spectral-mass spectrum at the settled vacuum, per-mode
retention/half-lives under friction, the Goldstone latch (frozen coset angle),
and the exact Noether-charge decay law Q_n = (1-gamma)^n Q_0.

With ``potential_type="so2_invariant"`` the angular flat direction is exact by
construction (designed symmetry); with ``"mlp"`` the question becomes whether
a near-flat direction *emerges* from the degenerate data. The kinetic
isotropy requirement (F5 §4.1: equal channel inertial masses) is enforced via
``tie_channel_mass=True``; switch it off (and use kinetic_energy_mode
"newtonian_learned") for the broken-isotropy falsifiable.

Nomenclature (F5 Def-2): inertial mass M vs spectral mass mu.

Runnable directly (documented script entry):
    uv run python -m chlu.experiments.exp_d_goldstone --quick
    uv run python -m chlu.experiments.exp_d_goldstone --project myproj --tilt-delta 0.01 --tilt-n 2
or via the CLI: ``chlu exp-d [--project N] [--seed I] [--quick] ...``.
"""

import os
import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.data.circle_vacuum import generate_circle_vacuum
from chlu.experiments.goldstone_harness import (
    classify_mode,
    coset_angle,
    fit_decay_rate,
    half_life_first_crossing,
    latch_prediction,
    noether_charge,
    perturb_and_track,
    predicted_half_life,
    rollout_from,
    settle,
    spectrum_probe,
)
from chlu.training.train import train_chlu
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.plotting import plot_goldstone_summary


def run_experiment_d(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    use_pretrained: Optional[bool] = None,
    seed: Optional[int] = None,
    train_epochs: Optional[int] = None,
    potential_type: Optional[str] = None,
    tie_channel_mass: Optional[bool] = None,
    tilt_delta: Optional[float] = None,
    tilt_n: Optional[int] = None,
    sleep_mode: Optional[str] = None,
    anchor_lambda: Optional[float] = None,
) -> dict:
    """
    Experiment D: SO(2) Goldstone memory.

    Protocol:
        1. Generate an SO(2)-degenerate dataset: constant states on a circle
           of radius R in the channel plane (spectators at 0, p = 0).
        2. Train a small CLU with the standard wake–sleep dynamics trainer.
        3. Settle to the learned vacuum, run the spectrum probe
           (W = M_eff^{-1/2} Hess V M_eff^{-1/2} -> spectral masses mu_k^2).
        4. Perturb-and-track: position kicks along the flattest and stiffest
           modes at fixed probe friction -> retention curves and half-lives
           (vs the exact F5 §3.4 predictions).
        5. Latch: momentum kick along the flattest mode -> measured frozen
           displacement vs d_inf = d0 + dt*pc0/gamma; coset angle theta(t).
        6. Noether: tangential kick -> Q(n) vs the exact (1-gamma)^n law.

    Returns a dict of measured numbers (also written to results/exp_d_metrics.npz).
    """
    # ------------------------------------------------------------------ config
    if config is None:
        config = get_default_config()

    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed
    if use_pretrained is not None:
        config.experiment_d.use_pretrained = use_pretrained
    if train_epochs is not None:
        config.experiment_d.train_epochs = train_epochs
    if potential_type is not None:
        config.experiment_d.potential_type = potential_type
    if tie_channel_mass is not None:
        config.experiment_d.tie_channel_mass = tie_channel_mass
    if tilt_delta is not None:
        config.experiment_d.tilt_delta = tilt_delta
    if tilt_n is not None:
        config.experiment_d.tilt_n = tilt_n
    if sleep_mode is not None:
        config.experiment_d.sleep_mode = sleep_mode
    if anchor_lambda is not None:
        config.training.anchor_data_energy_lambda = anchor_lambda

    # sleep_mode="off" => wake-only training (sleep_frequency -> inf), the
    # data-pinned regime that does NOT erode the designed vacuum (v2-full-runs
    # Finding 0). Implemented as a large sleep_frequency so only the (harmless)
    # epoch-0 sleep event can fire — matches the validated wake-only run.
    if config.experiment_d.sleep_mode == "off":
        config.training.sleep_frequency = 10**9

    # Erosion guard (fix-pack-4 item 4): a designed degenerate vacuum trained
    # with an active sleep phase for many epochs and NO anchor is the exact
    # sleep-erosion regime (handover §7.14 / anchor-robustness): the wake–sleep
    # CD sleep phase inverts the SO(2) ring into a local maximum (r*->0). Warn
    # loudly, citing the cure, but do not change behavior (UX guard only).
    if (
        config.experiment_d.sleep_mode != "off"
        and config.experiment_d.train_epochs > 300
        and config.experiment_d.potential_type == "so2_invariant"
        and config.training.anchor_data_energy_lambda == 0.0
    ):
        warnings.warn(
            f"exp-d: sleep_mode='on', train_epochs="
            f"{config.experiment_d.train_epochs} (>300) on a DESIGNED SO(2) "
            "vacuum with NO V(data) anchor (training.anchor_data_energy_lambda"
            "=0.0). This is the sleep-erosion regime: wake–sleep CD inverts the "
            "degenerate ring into a local maximum (r*->0) beyond ~300–600 ep "
            "(handover §7.14 / anchor-robustness P11). Set "
            "anchor_data_energy_lambda>0 (e.g. 10–100), or sleep_mode='off', or "
            "train_epochs<=300 to keep the vacuum intact.",
            RuntimeWarning,
            stacklevel=2,
        )

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "../models")
    seed = config.project.seed
    cfg = config.experiment_d
    dim = cfg.dim
    dt = cfg.dt

    print("\n" + "=" * 60)
    print("EXPERIMENT D: SO(2) Goldstone Memory (V2)")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------ data
    print(
        f"\n[1/4] Generating circle-of-attractors data "
        f"({cfg.n_points} points, R={cfg.circle_radius}, dim={dim})..."
    )
    k1, k2, k3 = jax.random.split(key, 3)
    train_data = generate_circle_vacuum(
        k1,
        n_points=cfg.n_points,
        seq_len=cfg.seq_len,
        dim=dim,
        radius=cfg.circle_radius,
    )
    print(f"  Train data: {train_data.shape}")

    # ------------------------------------------------------------------ model
    print(
        f"  CLU: dim={dim}, potential={cfg.potential_type}, "
        f"kinetic={cfg.kinetic_energy_mode}, tie_channel_mass={cfg.tie_channel_mass}, "
        f"tilt=(delta={cfg.tilt_delta}, n={cfg.tilt_n}), "
        f"spurion=(delta={cfg.spurion_delta}, angle={cfg.spurion_angle})"
    )
    chlu = CHLU(
        dim=dim,
        hidden=cfg.hidden_dim,
        rest_mass=config.model.rest_mass,
        c=config.model.speed_of_causality,
        kinetic_mode=cfg.kinetic_energy_mode,
        potential_type=cfg.potential_type,
        tie_channel_mass=cfg.tie_channel_mass,
        tilt_delta=cfg.tilt_delta,
        tilt_n=cfg.tilt_n,
        spurion_delta=cfg.spurion_delta,
        spurion_angle=cfg.spurion_angle,
        key=k2,
    )

    # ------------------------------------------------------ train or load
    chlu_path = os.path.join(models_dir, "exp_d_chlu.pkl")
    if cfg.use_pretrained and os.path.exists(chlu_path):
        print(f"\n[2/4] Loading pre-trained model from {chlu_path}...")
        chlu, metadata = load_checkpoint(chlu_path, chlu)
        target_energy = metadata.get("target_energy")
        print("  ✓ Model loaded")
    else:
        print(f"\n[2/4] Training CLU ({cfg.train_epochs} epochs, wake–sleep)...")
        # NOTE: epochs/window_size/dt passed explicitly — train_chlu reads
        # config.training.epochs otherwise (the §7.10 quick-mode trap).
        chlu, losses, target_energy = train_chlu(
            chlu,
            train_data,
            key=k3,
            config=config,
            epochs=cfg.train_epochs,
            window_size=cfg.seq_len - 1,
            dt=dt,
        )
        print(f"  Final wake loss: {float(losses[-1]):.6f}")
        save_checkpoint(
            chlu,
            chlu_path,
            epoch=cfg.train_epochs,
            loss=float(losses[-1]),
            config=config,
            target_energy=target_energy,
        )
        print(f"  Saved model to {chlu_path}")

    # ------------------------------------------------------------- measure
    print("\n[3/4] Measuring (settle -> spectrum -> perturb -> charge)...")

    # Settle to the learned vacuum from a training point
    q_data = train_data[0, 0, :dim]
    q_star, p_star = settle(
        chlu, q_data, dt=dt, gamma=cfg.settle_gamma, steps=cfg.settle_steps
    )
    r_star = float(jnp.sqrt(q_star[0] ** 2 + q_star[1] ** 2))

    # Spectrum probe
    probe = spectrum_probe(chlu, q_star)
    mu_sq = np.asarray(probe.mu_sq)
    bands = [classify_mode(float(m2), dt, cfg.probe_gamma) for m2 in mu_sq]
    pred_hl = [predicted_half_life(float(m2), dt, cfg.probe_gamma) for m2 in mu_sq]

    print(f"  Settled point: r* = {r_star:.6f} (data R = {cfg.circle_radius})")
    print(
        f"  |grad V(q*)| = {float(probe.grad_norm):.3e}, |p*| = {float(jnp.linalg.norm(p_star)):.3e}"
    )
    print(f"  Inertial masses M_eff = {np.asarray(probe.M_eff)}")
    print("  Spectral masses mu_k^2 (ascending) and F5 §3.4 bands:")
    for k_mode, (m2, band, hl) in enumerate(zip(mu_sq, bands, pred_hl, strict=True)):
        print(f"    mode {k_mode}: mu^2 = {m2: .6e}  [{band}]  pred n_1/2 = {hl}")

    # Perturb-and-track: flattest and stiffest modes (position kicks)
    res_flat = perturb_and_track(
        chlu,
        probe,
        mode_idx=0,
        kick=cfg.probe_kick,
        kick_type="position",
        steps=cfg.probe_steps,
        dt=dt,
        gamma=cfg.probe_gamma,
    )
    res_stiff = perturb_and_track(
        chlu,
        probe,
        mode_idx=dim - 1,
        kick=cfg.probe_kick,
        kick_type="position",
        steps=cfg.probe_steps,
        dt=dt,
        gamma=cfg.probe_gamma,
    )
    hl_flat = half_life_first_crossing(res_flat["retention"])
    hl_stiff = half_life_first_crossing(res_stiff["retention"])
    print(f"  Measured n_1/2: flattest mode = {hl_flat}, stiffest mode = {hl_stiff}")

    # Latch: momentum kick (the write operation) along the flattest mode
    res_latch = perturb_and_track(
        chlu,
        probe,
        mode_idx=0,
        kick=cfg.probe_kick,
        kick_type="momentum",
        steps=cfg.probe_steps,
        dt=dt,
        gamma=cfg.probe_gamma,
    )
    d_latch = np.asarray(res_latch["d"][:, 0])
    latch_pred = float(
        latch_prediction(d_latch[0], cfg.probe_kick, dt, cfg.probe_gamma)
    )
    latch_meas = float(d_latch[-1])
    latch_freeze = float(abs(d_latch[-1] - d_latch[cfg.probe_steps // 2]))
    theta = coset_angle(res_latch["traj"], dim)
    theta_drift = float(abs(theta[-1] - theta[len(theta) // 2]))
    print(
        f"  Latch: measured d_inf = {latch_meas:.6f} vs predicted {latch_pred:.6f} "
        f"(freeze drift over last half: {latch_freeze:.2e})"
    )
    print(
        f"  Coset angle: theta_final = {theta[-1]:.6f} rad (last-half drift {theta_drift:.2e})"
    )

    # Noether charge: tangential momentum kick on the channel.
    # Guard against a collapsed vacuum (r*->0): the tangent direction that =
    # (-q1, q0)/r* is undefined at the origin, and dividing by r*~0 silently
    # NaN-poisons the whole Noether measurement (v2-full-runs Finding 0: at the
    # eroded 1000-epoch defaults r*->0). Return NaN explicitly with a loud
    # warning instead of propagating a silent NaN.
    r_star_floor = 1e-6
    if r_star < r_star_floor:
        warnings.warn(
            f"Noether metric skipped: settled radius r*={r_star:.3e} < "
            f"{r_star_floor:.0e} (the SO(2) vacuum has collapsed — see "
            "v2-full-runs Finding 0 / handover §7.14; try sleep_mode='off' or "
            "fewer train_epochs). Noether outputs set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        that = jnp.zeros(dim)
        p0_q = cfg.probe_kick * that
        traj_q = rollout_from(
            chlu, q_star, p0_q, steps=cfg.probe_steps, dt=dt, gamma=cfg.probe_gamma
        )
        Q = np.full(cfg.probe_steps + 1, np.nan)
        Q_pred = np.full_like(Q, np.nan)
        noether_max_err = float("nan")
    else:
        that = (
            jnp.zeros(dim).at[0].set(-q_star[1] / r_star).at[1].set(q_star[0] / r_star)
        )
        p0_q = cfg.probe_kick * that
        traj_q = rollout_from(
            chlu, q_star, p0_q, steps=cfg.probe_steps, dt=dt, gamma=cfg.probe_gamma
        )
        Q = np.asarray(noether_charge(traj_q, dim))
        n_arr = np.arange(len(Q))
        Q_pred = (1.0 - cfg.probe_gamma) ** n_arr * Q[0]
        noether_max_err = float(np.max(np.abs(Q - Q_pred)) / abs(Q[0]))
    print(
        f"  Noether decay: max |Q_n - (1-gamma)^n Q_0| / |Q_0| = {noether_max_err:.3e}"
    )

    # ------------------------------------------------------------- outputs
    print("\n[4/4] Writing plots and metrics...")

    retention_curves = {
        f"mode 0 (mu^2={mu_sq[0]:.2e}, {bands[0]})": res_flat["retention"],
        f"mode {dim - 1} (mu^2={mu_sq[-1]:.2e}, {bands[-1]})": res_stiff["retention"],
        "mode 0 momentum write (latch)": np.abs(d_latch)
        / max(np.max(np.abs(d_latch)), 1e-20),
    }
    plot_path = os.path.join(save_dir, "exp_d_goldstone_summary.png")
    plot_goldstone_summary(
        mu_sq, retention_curves, (Q, Q_pred), theta, plot_path, gamma=cfg.probe_gamma
    )

    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "exp_d_metrics.npz")
    np.savez(
        metrics_path,
        mu_sq=mu_sq,
        M_eff=np.asarray(probe.M_eff),
        K=np.asarray(probe.K),
        q_star=np.asarray(q_star),
        r_star=r_star,
        grad_norm=float(probe.grad_norm),
        bands=np.array(bands),
        predicted_half_life=np.array(pred_hl, dtype=float),
        measured_half_life_flat=hl_flat,
        measured_half_life_stiff=hl_stiff,
        retention_flat=np.asarray(res_flat["retention"]),
        retention_stiff=np.asarray(res_stiff["retention"]),
        d_latch=d_latch,
        latch_pred=latch_pred,
        latch_measured=latch_meas,
        latch_freeze_drift=latch_freeze,
        theta=theta,
        noether_Q=Q,
        noether_Q_pred=Q_pred,
        noether_max_err=noether_max_err,
        probe_gamma=cfg.probe_gamma,
        probe_kick=cfg.probe_kick,
        dt=dt,
        seed=seed,
        potential_type=cfg.potential_type,
        kinetic_energy_mode=cfg.kinetic_energy_mode,
        tie_channel_mass=cfg.tie_channel_mass,
        tilt_delta=cfg.tilt_delta,
        tilt_n=cfg.tilt_n,
        spurion_delta=cfg.spurion_delta,
        spurion_angle=cfg.spurion_angle,
        target_energy=np.asarray(
            target_energy if target_energy is not None else np.nan
        ),
    )
    print(f"  Saved metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT D COMPLETE!")
    print("=" * 60 + "\n")

    return {
        "mu_sq": mu_sq,
        "bands": bands,
        "predicted_half_life": pred_hl,
        "measured_half_life_flat": hl_flat,
        "measured_half_life_stiff": hl_stiff,
        "r_star": r_star,
        "grad_norm": float(probe.grad_norm),
        "latch_pred": latch_pred,
        "latch_measured": latch_meas,
        "latch_freeze_drift": latch_freeze,
        "theta_final": float(theta[-1]),
        "theta_drift_last_half": theta_drift,
        "noether_max_err": noether_max_err,
        # Decay rate of the stiffest mode's envelope (compare ln|lambda| =
        # 0.5*ln(1-gamma) for an underdamped mode, F5 §3.3b)
        "stiff_decay_rate_fit": fit_decay_rate(
            res_stiff["retention"], n_fit=min(500, cfg.probe_steps)
        ),
        "q_star": np.asarray(q_star),
        "M_eff": np.asarray(probe.M_eff),
    }


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment D: SO(2) Goldstone memory (V2)"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (100 training epochs)"
    )
    parser.add_argument(
        "--potential-type",
        choices=["so2_invariant", "mlp"],
        help="Designed symmetry (so2_invariant) or emergent (mlp)",
    )
    parser.add_argument(
        "--broken-isotropy",
        action="store_true",
        help="Untie the channel inertial masses (F5 §4.1 falsifiable)",
    )
    parser.add_argument("--tilt-delta", type=float, help="Explicit breaking delta")
    parser.add_argument("--tilt-n", type=int, help="Tilt harmonic n")
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
        config.experiment_d.train_epochs = 100

    run_experiment_d(
        config=config,
        save_dir=save_dir,
        models_dir=models_dir,
        seed=args.seed,
        potential_type=args.potential_type,
        tie_channel_mass=False if args.broken_isotropy else None,
        tilt_delta=args.tilt_delta,
        tilt_n=args.tilt_n,
    )


if __name__ == "__main__":
    main()


# Assumptions documented (this build):
#   - Channel = coordinates (0, 1) by convention; one SO(2) channel.
#   - Channel/spectator split in V is additive (Hessian block-diagonal between
#     channel and spectators by construction for so2_invariant).
#   - The radial profile is fed the invariant r^2 (not r) — every smooth
#     SO(2)-invariant function is smooth in r^2; avoids a cusp at the origin.
#   - Dataset = static points on the circle (zero momentum): the minimal task
#     whose vacuum manifold is a circle. Orbiting variants are follow-up work.
