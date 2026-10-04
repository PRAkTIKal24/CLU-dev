"""Lyapunov stability regularization for CHLU."""

import jax
import jax.numpy as jnp

VALID_LYAPUNOV_PENALTIES = ("none", "max", "sq", "pos", "legacy_degenerate")


def compute_lyapunov_loss(
    step_fn,
    trajectory: jnp.ndarray,
    n_samples: int = 10,
    penalty: str = "max",
) -> jnp.ndarray:
    """
    Compute a Lyapunov-spectrum regularization loss to penalize chaos.

    The loss is built from the log singular values log(sigma_i) of the
    Jacobian of one dynamics step (the local, one-step Lyapunov exponents),
    sampled at ``n_samples`` points along the trajectory.

    Penalty options (selected via ``training.lyapunov_penalty``):
        - "max":  mean over samples of max_i log(sigma_i)
                  (largest local Lyapunov exponent — the chaos-relevant one)
        - "sq":   mean over samples of sum_i log(sigma_i)^2
                  (hyperbolicity / total squeeze magnitude)
        - "pos":  mean over samples of sum_i max(0, log(sigma_i))
                  (total expansion rate; penalizes only expanding directions)
        - "legacy_degenerate": mean over samples of mean_i log(sigma_i).
                  WARNING — provably theta-independent (F5 Prop-5): by exact
                  conformal symplecticity of the dissipative Verlet map,
                  mean_i log(sigma_i) == 0.5 * ln(1 - gamma) identically for
                  any potential/state, so at gamma=0 (the wake phase) this is
                  == 0 with ~zero gradient. Kept only to reproduce historical
                  runs; it cannot penalize chaos even in principle.
        - "none": returns 0.0 without computing any Jacobian (disable).

    Args:
        step_fn: Function (q, p) -> (q_next, p_next)
        trajectory: Trajectory array of shape (T, 2*dim) [q, p]
        n_samples: Number of points to sample from trajectory
        penalty: One of {"none", "max", "sq", "pos", "legacy_degenerate"}

    Returns:
        Lyapunov regularization loss (scalar)
    """
    if penalty not in VALID_LYAPUNOV_PENALTIES:
        raise ValueError(
            f"Unknown lyapunov penalty: {penalty!r}. "
            f"Must be one of {VALID_LYAPUNOV_PENALTIES}."
        )

    if penalty == "none":
        return jnp.asarray(0.0)

    T, state_dim = trajectory.shape
    dim = state_dim // 2

    # Sample timesteps evenly along the trajectory
    indices = jnp.linspace(0, T - 1, n_samples, dtype=jnp.int32)

    def penalty_at_point(idx):
        """Compute the penalty from the step-Jacobian log singular values."""
        state = trajectory[idx]

        # Wrapper for step function that takes flat state
        def step_wrapper(flat_state):
            q_in = flat_state[:dim]
            p_in = flat_state[dim:]
            q_out, p_out = step_fn((q_in, p_in))
            return jnp.concatenate([q_out, p_out])

        # Compute Jacobian and its singular values
        jacobian = jax.jacfwd(step_wrapper)(state)
        singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
        log_sv = jnp.log(singular_values + 1e-8)

        # `penalty` is a static Python string, so this branch resolves at
        # trace time and stays jit-compatible.
        if penalty == "max":
            return jnp.max(log_sv)
        elif penalty == "sq":
            return jnp.sum(log_sv**2)
        elif penalty == "pos":
            return jnp.sum(jnp.maximum(0.0, log_sv))
        else:  # "legacy_degenerate"
            return jnp.mean(log_sv)

    # Mean over sampled trajectory points
    return jnp.mean(jax.vmap(penalty_at_point)(indices))
