"""CLU-lattice experiment (V3 first build): scale by mass AND size.

The first working CLU-Net (F5 §7 / Def-4): a joint-Hamiltonian lattice of
CHLU units with position-only coupling on a declared edge list, one global
Verlet step, and DESIGNED inertial-mass banding (Thread-5 + wave-2 verdict:
the mass hierarchy does not emerge — it must be designed in).

Four measurements, in order of importance:

1. **Communication pricing (the acceptance centerpiece, F5 §7.2).** A
   designed 2-unit lattice — Mexican-hat SO(2) channels + a channel spring
   V_c = kappa_c * ||q_1ch - q_2ch||^2 invariant only under SIMULTANEOUS
   rotation. Swept over kappa_c: (a) sync timescale of the relative angle
   (predicted ∝ kappa_c^{-1/2}); (b) relative-information retention
   (overdamped n_1/2 predicted ∝ 1/kappa_c); (c) the diagonal (shared)
   channel stays an exact Goldstone latch at every kappa_c; plus the
   quadratic-order law mu_rel^2 = 4*kappa_c/M and the joint Noether decay
   Q_n = (1-gamma)^n Q_0. One plot: "coupling strength prices communication
   speed against relative-memory lifetime."

2. **Scaling smoke.** N in scaling_sizes chain lattices: joint symplecticity
   ||J^T Omega J - Omega|| at gamma=0 (step_jacobian on the joint state),
   relative energy drift, wall-clock steps/sec (informs CSF3 sizing).

3. **Wormhole slot smoke (skeleton).** A distant pair coupled through the
   smooth energy gate (F5 §7.4): force transmits when the endpoint states
   are aligned, and dies when they are far (gate closed). No top-k logic.

4. **Training smoke (single seed = INDICATIVE ONLY).** A 2-unit lattice
   trained with the standard wake-sleep trainer on a two-timescale composite
   signal (heavy-slow + light-fast reference oscillators, shared stiffness);
   banded-init vs uniform-init inertial masses at matched parameters.

All measurement instruments come from chlu.experiments.goldstone_harness,
applied verbatim to the joint state (the lattice duck-types the CHLU
surface). Nomenclature (F5 Def-2): inertial mass M vs spectral mass mu.

Runnable directly:
    uv run python -m chlu.experiments.exp_lattice --quick
    uv run python -m chlu.experiments.exp_lattice --project myproj --skip-training
or via the CLI: ``chlu exp-lattice [--project N] [--seed I] [--quick] ...``.
"""

import math
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.lattice import (
    CLULattice,
    build_lattice,
    chain_edges,
    channel_spring_coupling,
)
from chlu.data.two_timescale_orbits import generate_two_timescale_orbits
from chlu.experiments.goldstone_harness import (
    MexicanHatPotential,
    clu_with_potential,
    coset_angle,
    half_life_first_crossing,
    latch_prediction,
    noether_charge,
    perturb_and_track,
    predicted_half_life,
    rollout_from,
    spectrum_probe,
    step_jacobian,
)
from chlu.training.train import train_chlu
from chlu.utils.checkpoints import load_checkpoint, save_checkpoint
from chlu.utils.plotting import plot_lattice_pricing

import equinox as eqx


# ---------------------------------------------------------------------------
# Designed instruments
# ---------------------------------------------------------------------------


def designed_so2_pair(
    kappa_c: float,
    hat_lambda: float = 1.0,
    vacuum_radius: float = 1.0,
    channel_inertia: float = 1.0,
) -> CLULattice:
    """
    The designed 2-unit pricing lattice: two dim-2 Mexican-hat SO(2) channels
    (vacuum circles of radius f) joined by a channel spring invariant only
    under simultaneous rotation. Channel inertial masses are equal within and
    across units (kinetic isotropy, F5 §4.1). At the synchronized vacuum the
    exact joint channel spectrum is
        {0 (shared latch), 4*kappa_c/M (relative), 8*lam*f^2/M,
         8*lam*f^2/M + 4*kappa_c/M}.
    """
    hat = MexicanHatPotential(lam=hat_lambda, f=vacuum_radius, k_spec=None)
    inertia = (channel_inertia, channel_inertia)
    units = [
        clu_with_potential(
            hat,
            dim=2,
            kinetic_mode="newtonian_learned",
            inertia=inertia,
            key=jax.random.PRNGKey(i),
        )
        for i in range(2)
    ]
    coupling = channel_spring_coupling(2, 2, kappa_c, channel=(0, 1))
    return CLULattice(units=units, edges=((0, 1),), couplings=(coupling,))


def _pair_on_circle(f: float, theta1: float, theta2: float) -> jnp.ndarray:
    """Joint q with unit 1 at angle theta1 and unit 2 at theta2 on radius f."""
    return jnp.array(
        [
            f * math.cos(theta1),
            f * math.sin(theta1),
            f * math.cos(theta2),
            f * math.sin(theta2),
        ]
    )


def joint_so2_charge(traj: jnp.ndarray, dim: int) -> np.ndarray:
    """Noether charge of the SIMULTANEOUS rotation: Q = sum of per-unit
    channel charges (channels (0,1) and (2,3) of the concatenated state)."""
    return np.asarray(
        noether_charge(traj, dim, channel=(0, 1))
        + noether_charge(traj, dim, channel=(2, 3))
    )


def measure_pricing_at_kappa(cfg, kappa_c: float, dt: float) -> dict:
    """All F5 §7.2 pricing observables for one coupling strength."""
    f = cfg.vacuum_radius
    M = cfg.channel_inertia
    gamma = cfg.probe_gamma
    lattice = designed_so2_pair(kappa_c, cfg.hat_lambda, f, M)
    dim = lattice.dim

    # --- spectrum at the analytic synchronized vacuum -----------------------
    q_star = _pair_on_circle(f, 0.0, 0.0)
    probe = spectrum_probe(lattice, q_star)
    mu_sq = np.asarray(probe.mu_sq)
    mu_rel_sq_pred = 4.0 * kappa_c / M
    # ascending order: [shared ~0, relative 4k/M, radial, radial + 4k/M]
    mu_sym_sq, mu_rel_sq = float(mu_sq[0]), float(mu_sq[1])

    # --- (a) sync timescale: gamma = 0, relative angle Delta0 ---------------
    q0 = _pair_on_circle(f, +cfg.sync_delta0 / 2.0, -cfg.sync_delta0 / 2.0)
    p0 = jnp.zeros(dim)
    traj_sync = rollout_from(
        lattice, q0, p0, steps=cfg.sync_max_steps, dt=dt, gamma=0.0
    )
    delta = coset_angle(traj_sync, dim, channel=(0, 1)) - coset_angle(
        traj_sync, dim, channel=(2, 3)
    )
    crossings = np.nonzero(np.sign(delta) != np.sign(delta[0]))[0]
    sync_steps = float(crossings[0]) if crossings.size else math.inf
    # Quarter period of the relative mode: t = pi / (2 mu_rel)
    sync_pred = math.pi / (2.0 * math.sqrt(mu_rel_sq_pred) * dt)
    # gamma = 0 sanity: joint energy conserved along the sync rollout
    H0 = float(lattice.H(q0, p0))
    H_end = float(lattice.H(traj_sync[-1, :dim], traj_sync[-1, dim:]))
    energy_drift_sync = abs(H_end - H0) / max(abs(H0), 1e-12)

    # --- (b) relative-information retention (overdamped register) -----------
    hl_pred = predicted_half_life(mu_rel_sq, dt, gamma)
    steps_ret = int(min(cfg.max_probe_steps, max(4.0 * hl_pred + 500.0, 1500.0)))
    res_rel = perturb_and_track(
        lattice,
        probe,
        mode_idx=1,
        kick=cfg.probe_kick,
        kick_type="position",
        steps=steps_ret,
        dt=dt,
        gamma=gamma,
    )
    hl_meas = half_life_first_crossing(res_rel["retention"])

    # --- (c) the shared channel stays an exact latch -------------------------
    res_latch = perturb_and_track(
        lattice,
        probe,
        mode_idx=0,
        kick=cfg.probe_kick,
        kick_type="momentum",
        steps=cfg.latch_steps,
        dt=dt,
        gamma=gamma,
    )
    d_shared = np.asarray(res_latch["d"][:, 0])
    latch_pred_val = float(latch_prediction(d_shared[0], cfg.probe_kick, dt, gamma))
    latch_err = abs(float(d_shared[-1]) - latch_pred_val)
    latch_freeze = float(
        abs(d_shared[-1] - d_shared[cfg.latch_steps // 2])
    )  # drift over the last half — 0 means frozen (deadbeat memory)
    # the two units move TOGETHER under the shared write
    th1 = coset_angle(res_latch["traj"], dim, channel=(0, 1))
    th2 = coset_angle(res_latch["traj"], dim, channel=(2, 3))
    shared_rel_leak = float(np.max(np.abs(th1 - th2)))

    # --- joint Noether decay under friction ----------------------------------
    Q = joint_so2_charge(res_latch["traj"], dim)
    n_arr = np.arange(len(Q))
    Q_pred = (1.0 - gamma) ** n_arr * Q[0]
    noether_err = float(np.max(np.abs(Q - Q_pred)) / max(abs(Q[0]), 1e-300))

    return {
        "kappa": kappa_c,
        "mu_sym_sq": mu_sym_sq,
        "mu_rel_sq": mu_rel_sq,
        "mu_rel_sq_pred": mu_rel_sq_pred,
        "sync_steps": sync_steps,
        "sync_pred": sync_pred,
        "energy_drift_sync": energy_drift_sync,
        "half_life": hl_meas,
        "half_life_pred": hl_pred,
        "latch_err": latch_err,
        "latch_freeze": latch_freeze,
        "shared_rel_leak": shared_rel_leak,
        "noether_err": noether_err,
        "grad_norm": float(probe.grad_norm),
    }


# ---------------------------------------------------------------------------
# Scaling / wormhole smokes
# ---------------------------------------------------------------------------


def _symplectic_error(lattice: CLULattice, q, p, dt) -> float:
    """max |J^T Omega J - Omega| for one conservative joint step."""
    D = lattice.dim
    J = np.asarray(step_jacobian(lattice, q, p, dt, gamma=0.0))
    Omega = np.block([[np.zeros((D, D)), np.eye(D)], [-np.eye(D), np.zeros((D, D))]])
    return float(np.max(np.abs(J.T @ Omega @ J - Omega)))


def scaling_smoke(cfg, key: jax.random.PRNGKey) -> list:
    """Chain lattices at N in scaling_sizes: joint symplecticity, energy
    drift at gamma=0, wall-clock steps/sec (post-compile)."""
    results = []
    dt = cfg.dt
    for n in cfg.scaling_sizes:
        key, k_build, k_state = jax.random.split(key, 3)
        lattice = build_lattice(
            k_build,
            unit_dims=[2] * n,
            hidden=cfg.hidden_dim,
            potential_type="mlp",
            kinetic_mode=cfg.kinetic_energy_mode,
            edges=chain_edges(n),
            coupling_type=cfg.coupling_type,
            kappa_c=cfg.kappa_c,
            coupling_dim=cfg.coupling_dim,
            proj_init_scale=cfg.proj_init_scale,
        )
        D = lattice.dim
        kq, kp = jax.random.split(k_state)
        q0 = 0.5 * jax.random.normal(kq, (D,))
        p0 = 0.5 * jax.random.normal(kp, (D,))

        sympl_err = _symplectic_error(lattice, q0, p0, dt)

        roll = eqx.filter_jit(
            lambda m, q, p: m(q, p, steps=cfg.scaling_steps, dt=dt, gamma=0.0)
        )
        traj = roll(lattice, q0, p0)
        traj.block_until_ready()  # compile + first run
        t0 = time.perf_counter()
        traj = roll(lattice, q0, p0)
        traj.block_until_ready()
        wall = time.perf_counter() - t0

        H0 = float(lattice.H(q0, p0))
        H_end = float(lattice.H(traj[-1, :D], traj[-1, D:]))
        drift = abs(H_end - H0) / max(abs(H0), 1e-12)

        results.append(
            {
                "n_units": n,
                "dim": D,
                "symplectic_err": sympl_err,
                "energy_drift": drift,
                "steps_per_sec": cfg.scaling_steps / wall,
                "wall_sec": wall,
            }
        )
    return results


def wormhole_smoke(cfg, key: jax.random.PRNGKey) -> dict:
    """A gated non-adjacent edge (0, N-1) on a 4-chain: the distant pair
    couples when aligned (force transmitted through the smooth gate) and
    decouples when far (gate closed).

    Reports the gate OCCUPANCY <sigma> = dV_wh/dv (the transmitted force
    fraction) alongside the energy: v3-lattice-build checked only the energy
    suppression, which is exactly why the legacy gate's sign-reversing force
    stayed invisible (xy-lattice-theory §6.2)."""
    n = 4
    lattice = build_lattice(
        key,
        unit_dims=[2] * n,
        hidden=cfg.hidden_dim,
        potential_type="mlp",
        kinetic_mode=cfg.kinetic_energy_mode,
        edges=chain_edges(n),
        coupling_type="spring",
        kappa_c=cfg.kappa_c,
        coupling_dim=cfg.coupling_dim,
        proj_init_scale=cfg.proj_init_scale,
        wormhole_edges=((0, n - 1),),
        wormhole_gate_threshold=cfg.wormhole_gate_threshold,
        wormhole_gate_width=cfg.wormhole_gate_width,
        gate_energy_mode=cfg.gate_energy_mode,
    )
    D = lattice.dim
    gate = lattice.couplings[-1]  # the GatedCoupling on (0, 3)
    s0, s3 = lattice.unit_slice(0), lattice.unit_slice(n - 1)

    grad_V = jax.grad(lattice.V)

    # Aligned endpoints: base spring energy ~ 0 => gate open
    q_aligned = (
        jnp.zeros(D)
        .at[s0]
        .set(jnp.array([0.3, -0.2]))
        .at[s3]
        .set(jnp.array([0.3, -0.2]))
    )
    # Move unit 0 only; measure the induced force change on unit 3
    q_shifted = q_aligned.at[s0].set(jnp.array([0.9, 0.4]))
    dforce_3 = float(jnp.max(jnp.abs(grad_V(q_aligned)[s3] - grad_V(q_shifted)[s3])))
    gate_v_aligned = float(gate(q_aligned[s0], q_aligned[s3]))

    # Far endpoints: gate closes smoothly, wormhole energy ~ 0
    q_far = (
        jnp.zeros(D)
        .at[s0]
        .set(jnp.array([40.0, 0.0]))
        .at[s3]
        .set(jnp.array([-40.0, 0.0]))
    )
    gate_v_far = float(gate(q_far[s0], q_far[s3]))

    return {
        "coupled_dforce_on_unit3": dforce_3,
        "wormhole_energy_aligned": gate_v_aligned,
        "wormhole_energy_far": gate_v_far,
        "gate_occupancy_aligned": float(gate.occupancy(q_aligned[s0], q_aligned[s3])),
        "gate_occupancy_far": float(gate.occupancy(q_far[s0], q_far[s3])),
    }


# ---------------------------------------------------------------------------
# Training smoke (banded vs uniform inertial mass; single seed = indicative)
# ---------------------------------------------------------------------------


def _build_training_lattice(cfg, key, banded: bool) -> CLULattice:
    return build_lattice(
        key,
        unit_dims=[2, 2],
        hidden=cfg.hidden_dim,
        potential_type="mlp",
        kinetic_mode=cfg.kinetic_energy_mode,
        mass_scales=list(cfg.banded_mass_scales) if banded else None,
        edges=((0, 1),),
        coupling_type=cfg.coupling_type,
        kappa_c=cfg.train_kappa_c,
        coupling_dim=cfg.coupling_dim,
        proj_init_scale=cfg.proj_init_scale,
    )


def _eval_rollout_mse(
    model: CLULattice, data: jnp.ndarray, steps: int, dt: float
) -> float:
    """Mean rollout MSE from each held-out trajectory's initial state."""
    D = model.dim

    @eqx.filter_jit
    def one(traj):
        pred = model(traj[0, :D], traj[0, D:], steps=steps, dt=dt)
        return jnp.mean((pred - traj[1 : steps + 1]) ** 2)

    return float(jnp.mean(jnp.stack([one(data[i]) for i in range(data.shape[0])])))


def training_smoke(
    cfg, config: CHLUConfig, key: jax.random.PRNGKey, models_dir: str
) -> dict:
    """Train banded-init vs uniform-init 2-unit lattices (identical
    architecture, key, data, budget — only the log_mass INIT differs) on the
    two-timescale composite task. Single seed: indicative only."""
    k_data, k_eval, k_model, k_train = jax.random.split(key, 4)
    train_data = generate_two_timescale_orbits(
        k_data,
        n_traj=cfg.train_n_traj,
        seq_len=cfg.train_seq_len,
        dt=cfg.dt,
        omegas=tuple(cfg.data_omegas),
        masses=tuple(cfg.data_masses),
        radius=cfg.data_radius,
    )
    eval_data = generate_two_timescale_orbits(
        k_eval,
        n_traj=8,
        seq_len=cfg.eval_steps + 1,
        dt=cfg.dt,
        omegas=tuple(cfg.data_omegas),
        masses=tuple(cfg.data_masses),
        radius=cfg.data_radius,
    )

    out = {}
    for label, banded in [("banded", True), ("uniform", False)]:
        model = _build_training_lattice(cfg, k_model, banded)
        n_params = sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )
        ckpt_path = os.path.join(models_dir, f"exp_lattice_{label}.pkl")
        if cfg.use_pretrained and os.path.exists(ckpt_path):
            model, meta = load_checkpoint(ckpt_path, model)
            final_loss = float(meta.get("loss", np.nan))
            print(f"  [{label}] loaded pre-trained model from {ckpt_path}")
        else:
            print(
                f"  [{label}] training ({cfg.train_epochs} epochs, "
                f"{n_params} params, init masses "
                f"{[np.round(np.asarray(m), 3).tolist() for m in model.unit_mass_vectors()]})..."
            )
            # NOTE: epochs/window/dt passed explicitly (handover §7.10 trap:
            # train_chlu reads config.training.epochs otherwise).
            model, losses, _ = train_chlu(
                model,
                train_data,
                key=k_train,
                config=config,
                epochs=cfg.train_epochs,
                window_size=cfg.train_window,
                dt=cfg.dt,
            )
            final_loss = float(losses[-1])
            save_checkpoint(
                model, ckpt_path, epoch=cfg.train_epochs, loss=final_loss, config=config
            )
        eval_mse = _eval_rollout_mse(model, eval_data, cfg.eval_steps, cfg.dt)
        out[label] = {
            "final_wake_loss": final_loss,
            "eval_rollout_mse": eval_mse,
            "n_params": int(n_params),
            "learned_masses": [
                np.asarray(m).tolist() for m in model.unit_mass_vectors()
            ],
        }
        print(
            f"  [{label}] final wake loss = {final_loss:.6f}, "
            f"eval rollout MSE = {eval_mse:.6f}"
        )
    return out


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def run_experiment_lattice(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: bool = False,
    skip_training: bool = False,
) -> dict:
    """Run the CLU-lattice experiment (see module docstring)."""
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "../models")
    seed = config.project.seed
    cfg = config.experiment_lattice
    dt = cfg.dt

    if quick:
        cfg.kappa_sweep = [0.01, 0.1, 0.3]
        cfg.max_probe_steps = 8000
        cfg.sync_max_steps = 2000
        cfg.latch_steps = 1500
        cfg.scaling_sizes = [2, 4]
        cfg.scaling_steps = 500
        cfg.train_epochs = 60

    print("\n" + "=" * 60)
    print("EXPERIMENT LATTICE: CLU-Net first build (V3)")
    print("=" * 60)
    print(
        f"  seed={seed}, dt={dt}, quick={quick}, "
        f"kappa_sweep={list(cfg.kappa_sweep)}, probe_gamma={cfg.probe_gamma}"
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    k_scaling, k_wormhole, k_train = jax.random.split(key, 3)

    # ------------------------------------------------- 1. communication pricing
    print(
        f"\n[1/4] Communication pricing on the designed 2-unit SO(2) lattice "
        f"(f={cfg.vacuum_radius}, M={cfg.channel_inertia}, lam={cfg.hat_lambda})..."
    )
    rows = []
    for kappa_c in cfg.kappa_sweep:
        r = measure_pricing_at_kappa(cfg, float(kappa_c), dt)
        rows.append(r)
        print(
            f"  kappa={kappa_c:<7g} mu_rel^2={r['mu_rel_sq']:.6f} "
            f"(pred {r['mu_rel_sq_pred']:.6f})  sync={r['sync_steps']:.0f} steps "
            f"(pred {r['sync_pred']:.0f})  n_1/2={r['half_life']:.0f} "
            f"(pred {r['half_life_pred']:.0f})  latch freeze={r['latch_freeze']:.2e}  "
            f"Noether err={r['noether_err']:.2e}"
        )

    kappas = np.array([r["kappa"] for r in rows])
    sync_meas = np.array([r["sync_steps"] for r in rows])
    sync_pred = np.array([r["sync_pred"] for r in rows])
    hl_meas = np.array([r["half_life"] for r in rows])
    hl_pred = np.array([r["half_life_pred"] for r in rows])
    mu_rel_meas = np.array([r["mu_rel_sq"] for r in rows])
    mu_rel_pred = np.array([r["mu_rel_sq_pred"] for r in rows])
    latch_freeze = np.array([r["latch_freeze"] for r in rows])
    mu_sym_abs = np.array([abs(r["mu_sym_sq"]) for r in rows])

    # Power-law fits (the pricing laws)
    sync_slope = float(np.polyfit(np.log(kappas), np.log(sync_meas), 1)[0])
    hl_slope = float(np.polyfit(np.log(kappas), np.log(hl_meas), 1)[0])
    print(
        f"  Fitted slopes: sync ∝ kappa^{sync_slope:.3f} (predicted -0.5), "
        f"n_1/2 ∝ kappa^{hl_slope:.3f} (predicted -1)"
    )

    plot_path = os.path.join(save_dir, "exp_lattice_pricing.png")
    plot_lattice_pricing(
        kappas,
        sync_meas,
        sync_pred,
        hl_meas,
        hl_pred,
        mu_rel_meas,
        mu_rel_pred,
        latch_freeze,
        mu_sym_abs,
        plot_path,
        gamma=cfg.probe_gamma,
        slopes=(sync_slope, hl_slope),
    )
    print(f"  Saved pricing plot to {plot_path}")

    # ---------------------------------------------------------- 2. scaling smoke
    print(f"\n[2/4] Scaling smoke (chains, N in {list(cfg.scaling_sizes)})...")
    scaling = scaling_smoke(cfg, k_scaling)
    for s in scaling:
        print(
            f"  N={s['n_units']} (D={s['dim']}): ||J^T O J - O|| = "
            f"{s['symplectic_err']:.2e}, energy drift = {s['energy_drift']:.2e}, "
            f"{s['steps_per_sec']:.0f} steps/s"
        )

    # ---------------------------------------------------------- 3. wormhole smoke
    print("\n[3/4] Wormhole slot smoke (gated edge (0, 3) on a 4-chain)...")
    wh = wormhole_smoke(cfg, k_wormhole)
    print(
        f"  aligned: dF on unit 3 from moving unit 0 = "
        f"{wh['coupled_dforce_on_unit3']:.3e} (V_wh = {wh['wormhole_energy_aligned']:.3e}, "
        f"<sigma> = {wh['gate_occupancy_aligned']:.3f}); "
        f"far: V_wh = {wh['wormhole_energy_far']:.3e}, "
        f"<sigma> = {wh['gate_occupancy_far']:.3e} (gate closed)"
    )

    # --------------------------------------------------------- 4. training smoke
    training = None
    if skip_training:
        print("\n[4/4] Training smoke SKIPPED (--skip-training).")
    else:
        print(
            "\n[4/4] Training smoke: banded vs uniform inertial-mass init "
            "(single seed — indicative only, not a claim)..."
        )
        training = training_smoke(cfg, config, k_train, models_dir)
        b, u = training["banded"], training["uniform"]
        better = (
            "banded" if b["eval_rollout_mse"] < u["eval_rollout_mse"] else "uniform"
        )
        print(
            f"  => {better} wins on eval rollout MSE at this seed "
            f"(banded {b['eval_rollout_mse']:.6f} vs uniform {u['eval_rollout_mse']:.6f}). "
            f"SINGLE SEED = indicative only."
        )

    # -------------------------------------------------------------- metrics out
    results_dir = os.path.join(save_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "exp_lattice_metrics.npz")
    metrics = {
        "kappas": kappas,
        "mu_rel_sq_measured": mu_rel_meas,
        "mu_rel_sq_predicted": mu_rel_pred,
        "mu_sym_sq_abs": mu_sym_abs,
        "sync_steps_measured": sync_meas,
        "sync_steps_predicted": sync_pred,
        "half_life_measured": hl_meas,
        "half_life_predicted": hl_pred,
        "latch_freeze_drift": latch_freeze,
        "latch_err": np.array([r["latch_err"] for r in rows]),
        "shared_rel_leak": np.array([r["shared_rel_leak"] for r in rows]),
        "noether_err": np.array([r["noether_err"] for r in rows]),
        "energy_drift_sync": np.array([r["energy_drift_sync"] for r in rows]),
        "sync_slope": sync_slope,
        "hl_slope": hl_slope,
        "probe_gamma": cfg.probe_gamma,
        "probe_kick": cfg.probe_kick,
        "dt": dt,
        "seed": seed,
        "quick": quick,
        "scaling_n_units": np.array([s["n_units"] for s in scaling]),
        "scaling_symplectic_err": np.array([s["symplectic_err"] for s in scaling]),
        "scaling_energy_drift": np.array([s["energy_drift"] for s in scaling]),
        "scaling_steps_per_sec": np.array([s["steps_per_sec"] for s in scaling]),
        "wormhole_dforce": wh["coupled_dforce_on_unit3"],
        "wormhole_energy_aligned": wh["wormhole_energy_aligned"],
        "wormhole_energy_far": wh["wormhole_energy_far"],
        "wormhole_gate_energy_mode": cfg.gate_energy_mode,
        "wormhole_occupancy_aligned": wh["gate_occupancy_aligned"],
        "wormhole_occupancy_far": wh["gate_occupancy_far"],
    }
    if training is not None:
        for label in ("banded", "uniform"):
            metrics[f"train_{label}_final_wake_loss"] = training[label][
                "final_wake_loss"
            ]
            metrics[f"train_{label}_eval_mse"] = training[label]["eval_rollout_mse"]
            metrics[f"train_{label}_n_params"] = training[label]["n_params"]
    np.savez(metrics_path, **metrics)
    print(f"\n  Saved metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT LATTICE COMPLETE!")
    print("=" * 60 + "\n")

    return {
        "pricing": rows,
        "sync_slope": sync_slope,
        "hl_slope": hl_slope,
        "scaling": scaling,
        "wormhole": wh,
        "training": training,
    }


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CLU-lattice experiment (V3 first build)"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (short sweep, 60 epochs)"
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip the training smoke"
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

    run_experiment_lattice(
        config=config,
        save_dir=save_dir,
        models_dir=models_dir,
        seed=args.seed,
        quick=args.quick,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()


# Assumptions documented (this build):
#   - The pricing measurement runs on a DESIGNED lattice (Mexican-hat SO(2)
#     channels, exact vacuum radius f) — no training in the loop, so the
#     measured laws are attributable to the coupling alone.
#   - "Sync timescale" = first alignment (zero crossing) of the relative
#     angle at gamma=0 from a Delta0 offset ≈ quarter period pi/(2 mu_rel);
#     at Delta0=0.4 the pendulum anharmonicity adds ~+1% (Delta0^2/16) —
#     visible as a small positive bias vs the quadratic prediction.
#   - Retention n_1/2 uses the F5 first-crossing convention on the envelope
#     amplitude of the kicked (relative) mode at overdamped probe_gamma.
#   - The training smoke compares INITIALIZATIONS of log_mass only (banded
#     vs uniform) at bit-identical architecture/params/key/data/budget;
#     single seed => indicative, promoted to a claim only after a seed sweep.
