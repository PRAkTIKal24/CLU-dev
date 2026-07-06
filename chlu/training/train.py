"""PCD (Wake-Sleep) training for CHLU."""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chlu.config import CHLUConfig, get_default_config
from chlu.core.friction_field import (
    c1_regularizer,
    init_adaptive_state,
    maybe_adapt_holes,
)
from chlu.core.regularization import compute_lyapunov_loss
from chlu.training.losses import energy_loss, mse_loss
from chlu.training.replay_buffer import ReplayBuffer


def sample_window(
    key: jax.random.PRNGKey, data: jnp.ndarray, window_size: int
) -> jnp.ndarray:
    """
    Sample a random window from trajectory data.

    Args:
        key: JAX random key
        data: Trajectory data of shape (T, dim)
        window_size: Size of the window to sample

    Returns:
        Window of shape (window_size, dim)
    """
    max_start = len(data) - window_size
    idx = jax.random.randint(key, (), 0, max_start)
    return data[idx : idx + window_size]


def train_chlu(
    model,
    data: jnp.ndarray,
    key: jax.random.PRNGKey,
    config: Optional[CHLUConfig] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    lyapunov_lambda: Optional[float] = None,
    sleep_steps: Optional[int] = None,
    buffer_capacity: Optional[int] = None,
    batch_size: Optional[int] = None,
    dt: Optional[float] = None,
    window_size: Optional[int] = None,
    sleep_temperature: Optional[float] = None,
    langevin_noise: Optional[str] = None,
    negative_seed_states: Optional[tuple] = None,
):
    """
    Train CHLU using Persistent Contrastive Divergence (Wake-Sleep).

    Wake Phase: Supervised learning on data (MSE + Lyapunov regularization)
    Sleep Phase: Unsupervised energy minimization on replay buffer

    Args:
        model: CHLU model
        data: Training data of shape (n_trajectories, T, 2*dim) or (T, 2*dim)
        key: JAX random key
        config: CHLUConfig object (if None, uses defaults)
        epochs: Number of training epochs (overrides config)
        lr: Learning rate (overrides config)
        lyapunov_lambda: Weight for Lyapunov regularization (overrides config)
        sleep_steps: Number of dynamics steps in sleep phase (overrides config)
        buffer_capacity: Replay buffer capacity (overrides config)
        batch_size: Batch size for sleep phase (overrides config)
        dt: Time step for dynamics (overrides config)
        window_size: Window size for sub-sequence sampling (overrides config)
        sleep_temperature: Temperature for Langevin noise during sleep phase (overrides config)
        langevin_noise: Langevin noise scale, "legacy" or "fdt" (overrides config;
            see F5 Prop-9 / TrainingConfig.langevin_noise)
        negative_seed_states: Optional (q, p) arrays of shape (n, dim) written
            into the replay buffer after random init — lets an experiment
            expose a structured "garbage source" to the sleep phase (e.g. the
            S1 noise locus). Default None = unchanged random init.

    If the model carries a ``friction_field`` (trash regions, F5 Def-5), the
    wake loss adds a protection term pushing gamma_phi(q_data) down and the
    sleep loss adds a term pushing gamma_phi up at the evolved negatives
    (one contrastive signal, two fields — brainstorm Thread 1); weights =
    ``training.friction_field_protect_lambda`` / ``_hallu_lambda``, plus the
    optional C1 critical-damping nudge behind ``_c1_lambda``.

    Returns:
        (trained_model, losses): Trained model and loss history
    """
    # Load config with overrides
    if config is None:
        config = get_default_config()

    # Apply overrides
    if epochs is None:
        epochs = config.training.epochs
    if lr is None:
        lr = config.training.learning_rate
    if lyapunov_lambda is None:
        lyapunov_lambda = config.training.lyapunov_lambda
    if sleep_steps is None:
        sleep_steps = config.training.sleep_steps
    if buffer_capacity is None:
        buffer_capacity = config.training.buffer_capacity
    if batch_size is None:
        batch_size = config.training.batch_size
    if dt is None:
        dt = config.training.dt
    if window_size is None:
        # Use experiment A config if available, otherwise use full trajectory
        if hasattr(config, "experiment_a") and hasattr(
            config.experiment_a, "window_size"
        ):
            window_size = config.experiment_a.window_size
        else:
            window_size = None  # Will be set below

    sleep_frequency = config.training.sleep_frequency
    sleep_friction = config.training.sleep_friction
    persistent_sleep_buffer = config.training.persistent_sleep_buffer
    lyapunov_penalty = config.training.lyapunov_penalty
    # Friction-field contrastive weights (inert unless the model carries a field)
    ff_protect_lambda = config.training.friction_field_protect_lambda
    ff_hallu_lambda = config.training.friction_field_hallu_lambda
    ff_c1_lambda = config.training.friction_field_c1_lambda
    ff_hallu_gate = config.training.friction_field_hallu_gate
    ff_lr = config.training.friction_field_lr
    ff_adaptive_k = config.training.friction_field_adaptive_k
    if sleep_temperature is None:
        sleep_temperature = config.training.sleep_temperature
    if langevin_noise is None:
        langevin_noise = config.training.langevin_noise
    clamp_strength = jnp.array(config.training.clamp_strength)
    clamp_ramp = config.training.clamp_ramp

    # Handle data shape
    if data.ndim == 2:
        data = data[None, :, :]  # Add batch dimension

    n_trajectories, T, state_dim = data.shape
    dim = state_dim // 2

    # Set window_size if not provided
    if window_size is None:
        window_size = T  # Use full trajectory if no window specified

    # Initialize optimizer. Two-timescale when the model carries a trainable
    # friction field and friction_field_lr is set: hole centers live in
    # q-space and must travel O(units), which Adam's ~lr/step parameter
    # velocity cannot deliver at the base lr (see
    # TrainingConfig.friction_field_lr).
    params = eqx.filter(model, eqx.is_array)
    if ff_lr is not None and getattr(model, "friction_field", None) is not None:
        # NOTE: must be a label FUNCTION, not a labels pytree — a CHLU-shaped
        # pytree of strings is itself callable (CHLU.__call__), so optax would
        # mistake it for a label fn and call it on the params.
        def _field_labels(tree):
            lbl = jax.tree_util.tree_map(lambda _: "main", tree)
            return eqx.tree_at(
                lambda t: t.friction_field,
                lbl,
                replace=jax.tree_util.tree_map(
                    lambda _: "field", tree.friction_field
                ),
            )

        optimizer = optax.multi_transform(
            {"main": optax.adam(lr), "field": optax.adam(ff_lr)}, _field_labels
        )
    else:
        optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Initialize replay buffer
    k1, k2 = jax.random.split(key)
    buffer = ReplayBuffer(capacity=buffer_capacity, dim=dim)
    buffer.initialize_random(k1, scale=1.0)
    if negative_seed_states is not None:
        q_seed, p_seed = negative_seed_states
        n_seed = min(q_seed.shape[0], buffer_capacity)
        buffer.update((q_seed[:n_seed], p_seed[:n_seed]), jnp.arange(n_seed))

    losses = []

    # Adaptive-K spawning accumulator (only used when the model carries a
    # trainable field and friction_field_adaptive_k is set).
    adapt_active = ff_adaptive_k and getattr(model, "friction_field", None) is not None
    adapt_state = init_adaptive_state(dim) if adapt_active else None

    @eqx.filter_jit
    def wake_step(
        model, opt_state, trajectory, key, epoch, epochs_ramp, clamp_strength
    ):
        """Wake phase: supervised learning on trajectory window."""
        q_true = trajectory[:, :dim]
        p_true = trajectory[:, dim:]

        # Compute clamp_strength annealing outside loss_fn to avoid recomputation
        schedule = epoch / epochs_ramp
        annealed_clamp = clamp_strength * (1 - schedule) + 1.0
        effective_clamp = jnp.where(epoch < epochs_ramp, annealed_clamp, 1.0)

        def loss_fn(model):
            # Wake rollout uses a friction field with STOPPED gradients (same
            # principle as the sleep phase): for an imperfect model, damping
            # near the data manifold suppresses rollout-error growth, so the
            # (clamp-amplified) MSE gradient actively REWARDS on-manifold
            # friction — S1 pilot evidence: learned holes parked ON the
            # attractor at ~1.3x the critical-damping optimum (the
            # fastest-error-settling friction), immune to a 10x protect/hallu
            # rebalance. Field parameters are trained ONLY by the placement
            # terms (protection below, sleep hallucination, optional C1).
            field = getattr(model, "friction_field", None)
            if field is not None:
                frozen_field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)
                model_rollout = eqx.tree_at(
                    lambda m: m.friction_field, model, replace=frozen_field
                )
            else:
                model_rollout = model

            # Run CHLU dynamics from initial state for window_size steps
            q0, p0 = q_true[0], p_true[0]
            pred_trajectory = model_rollout(q0, p0, steps=len(trajectory), dt=dt)

            # Use precomputed clamp strength
            mse = effective_clamp * mse_loss(pred_trajectory, trajectory)

            # Lyapunov regularization (penalty selected via config;
            # "legacy_degenerate" reproduces the old theta-independent loss)
            lyap_loss = compute_lyapunov_loss(
                lambda state: model_rollout.step(state, dt),
                pred_trajectory,
                n_samples=min(10, len(trajectory) // 2),
                penalty=lyapunov_penalty,
            )

            loss = mse + lyapunov_lambda * lyap_loss

            # Trash-region protection (Thread-1 wake term): memories must
            # live in frictionless regions — push gamma_phi(q_data) down.
            # (gamma_phi >= 0, so this term drives friction at data to 0.)
            if field is not None:
                loss = loss + ff_protect_lambda * jnp.mean(jax.vmap(field)(q_true))

            return loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    @eqx.filter_jit
    def sleep_step(
        model, opt_state, q_batch, p_batch, key, sleep_friction=0.0, sleep_temperature=0.0,
        e_ref=0.0, e_scale=1.0,
    ):
        """Sleep phase: energy maximization on buffer samples.

        Buffer sampling is done by the caller (outside the jit) so that persisting
        evolved states back into the buffer doesn't force recompilation. Returns
        the evolved (q, p) states so the caller can update the replay buffer (PCD)
        when ``persistent_sleep_buffer`` is enabled.

        e_ref/e_scale: the current wake window's energy band (max, std) — used
        by the friction-field "energy" hallucination gate (see
        TrainingConfig.friction_field_hallu_gate). Ignored without a field.
        """

        def loss_fn(model):
            # Sleep evolution uses a friction field with STOPPED gradients:
            # the energy-maximization term must not legislate friction (its
            # gradient through the damped dynamics says "less friction keeps
            # hallucinations hot" — exactly backwards vs Thread-1, and it
            # overpowers the hallucination term for hot negatives). Field
            # parameters are trained ONLY by the placement terms (wake
            # protection, sleep hallucination, optional C1).
            field = getattr(model, "friction_field", None)
            if field is not None:
                frozen_field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)
                model_evolve = eqx.tree_at(
                    lambda m: m.friction_field, model, replace=frozen_field
                )
            else:
                model_evolve = model

            # Evolve states for k steps using scan to avoid slow compilation
            if sleep_temperature > 0.0:
                # Stochastic evolution with Langevin noise
                # Split key for each particle in batch
                nonlocal key
                key, *particle_keys = jax.random.split(key, batch_size + 1)
                particle_keys = jnp.array(particle_keys)

                def evolve_single(q, p, particle_key):
                    """Evolve a single (q, p) state for k steps with noise."""
                    def step_fn(carry, _):
                        state, key_state = carry
                        q_s, p_s = state
                        q_next, p_next, new_key = model_evolve.stochastic_step(
                            (q_s, p_s), dt=dt, gamma=sleep_friction,
                            temperature=sleep_temperature, key=key_state,
                            noise_mode=langevin_noise,
                        )
                        return ((q_next, p_next), new_key), None

                    state = (q, p)
                    (final_state, _), _ = jax.lax.scan(
                        step_fn, (state, particle_key), None, length=sleep_steps
                    )
                    return final_state

                q_evolved, p_evolved = jax.vmap(evolve_single)(q_batch, p_batch, particle_keys)
            else:
                # Deterministic evolution
                def evolve_single(q, p):
                    def step_fn(state, _):
                        return model_evolve.step(state, dt, gamma=sleep_friction), None

                    state = (q, p)
                    final_state, _ = jax.lax.scan(step_fn, state, None, length=sleep_steps)
                    return final_state

                q_evolved, p_evolved = jax.vmap(evolve_single)(q_batch, p_batch)

            # Negative sign because we want to *maximize* sleep energy
            sleep_energy = -energy_loss(model, q_evolved, p_evolved)
            loss = sleep_energy

            # Trash-region terms (Thread-1 sleep side): garbage attracts
            # friction — push gamma_phi UP at the PERSISTENT negatives.
            field = getattr(model, "friction_field", None)
            if field is not None:
                # stop_gradient: the term places friction where the negatives
                # ARE; it must not steer the negatives' dynamics themselves.
                q_hallu = jax.lax.stop_gradient(q_evolved)
                gamma_h = jax.vmap(field)(q_hallu)
                if ff_hallu_gate == "energy":
                    # "Persistent" = still above the data energy band after
                    # evolution. Ungated, CD negatives that converge onto the
                    # data manifold drag friction onto it (fights protection;
                    # observed in the S1 smoke run).
                    H_h = jax.lax.stop_gradient(
                        jax.vmap(model.H)(q_evolved, p_evolved)
                    )
                    w = jax.nn.sigmoid((H_h - e_ref) / e_scale)
                else:  # "all": every negative votes
                    w = jnp.ones_like(gamma_h)
                loss = loss - ff_hallu_lambda * jnp.mean(w * gamma_h)
                if ff_c1_lambda > 0.0:
                    # Optional C1 ablation: nudge hole strengths toward the
                    # critical-damping forgetting optimum 2*dt*mu(c_k)
                    # (mo-deep-read §5; default OFF — measure, don't force).
                    loss = loss + ff_c1_lambda * c1_regularizer(model, dt)

            # Return evolved states as aux so the caller can persist them (PCD)
            return loss, (q_evolved, p_evolved)

        (loss, (q_evolved, p_evolved)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss, q_evolved, p_evolved

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training CHLU"):
        k2, k3 = jax.random.split(k2)

        # Sample random trajectory
        traj_idx = jax.random.randint(k2, (), 0, n_trajectories)
        full_trajectory = data[traj_idx]

        # Sample random window from the trajectory
        k3, k4 = jax.random.split(k3)
        trajectory = sample_window(k4, full_trajectory, window_size)

        # Convert epoch values to jax arrays for clamp_strength annealing
        epoch_jax = jnp.array(epoch)
        epochs_ramp_jax = jnp.array(clamp_ramp * epochs)

        k4, k5 = jax.random.split(k4)
        model, opt_state, wake_loss = wake_step(
            model,
            opt_state,
            trajectory,
            k5,
            epoch_jax,
            epochs_ramp_jax,
            clamp_strength,
        )

        # Sleep phase (every few epochs to save compute)
        if epoch % sleep_frequency == 0:
            k5, k6, k7 = jax.random.split(k5, 3)
            # Sample from the buffer outside the jit so that (optional) persistence
            # of evolved states doesn't retrigger compilation.
            q_batch, p_batch, indices = buffer.sample(k7, batch_size)
            # Current wake window's energy band (top + spread) for the
            # friction-field "energy" hallucination gate.
            H_window = jax.vmap(model.H)(trajectory[:, :dim], trajectory[:, dim:])
            e_ref = jnp.max(H_window)
            e_scale = jnp.std(H_window) + 1e-6
            model, opt_state, sleep_loss, q_evolved, p_evolved = sleep_step(
                model,
                opt_state,
                q_batch,
                p_batch,
                k6,
                sleep_friction,
                sleep_temperature,
                e_ref,
                e_scale,
            )
            # Persistent Contrastive Divergence: write evolved negatives back.
            if persistent_sleep_buffer:
                buffer.update((q_evolved, p_evolved), indices)

            # Adaptive-K: spawn a hole where persistent-hallucination density
            # accumulates, prune decayed holes. Uses the SAME energy gate as the
            # sleep training term. Structural edits reset the optimizer state.
            if adapt_active:
                H_h = jax.vmap(model.H)(q_evolved, p_evolved)
                if ff_hallu_gate == "energy":
                    adapt_w = jax.nn.sigmoid((H_h - e_ref) / e_scale)
                else:
                    adapt_w = jnp.ones_like(H_h)
                new_field, adapt_state, changed = maybe_adapt_holes(
                    model.friction_field,
                    q_evolved,
                    adapt_w,
                    adapt_state,
                    spawn_threshold=config.training.friction_field_spawn_threshold,
                    spawn_min_dist=config.training.friction_field_spawn_min_dist,
                    spawn_radius=config.training.friction_field_spawn_radius,
                    spawn_strength=config.training.friction_field_spawn_strength,
                    prune_floor=config.training.friction_field_prune_floor,
                    max_holes=config.training.friction_field_max_k,
                )
                if changed:
                    # eqx.Module is frozen — replace the field leaf via tree_at,
                    # then reinit the optimizer (leaf shapes moved).
                    model = eqx.tree_at(
                        lambda m: m.friction_field, model, new_field
                    )
                    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        losses.append(float(wake_loss))

    # Compute target energy from training data for governor
    # This represents the "normal" energy level learned from clean data
    def compute_energy(traj):
        """Compute mean Hamiltonian over a single trajectory."""
        q = traj[:, :dim]
        p = traj[:, dim:]
        energies = jax.vmap(model.H)(q, p)
        return jnp.mean(energies)

    # Energy across all trajectories
    all_energies_flat = jax.vmap(compute_energy)(data).flatten()

    # 2. Target the "Floor" (1st Percentile)
    # This forces the orbit to the bottom of the learned valley.
    target_energy = float(jnp.percentile(all_energies_flat, 1.0))

    print(f"Mean Energy: {jnp.mean(all_energies_flat)}")
    print(f"Target (Floor): {target_energy}")
    # target_energy = float(jnp.mean(all_energies))

    return model, jnp.array(losses), target_energy
