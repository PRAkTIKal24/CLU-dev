"""SO(2)-degenerate "circle of attractors" dataset for Experiment D (V2)."""

import jax
import jax.numpy as jnp


def generate_circle_vacuum(
    key: jax.random.PRNGKey,
    n_points: int,
    seq_len: int,
    dim: int = 4,
    radius: float = 1.0,
) -> jnp.ndarray:
    """
    Generate constant trajectories parked on a circle in the (q0, q1) channel
    plane — the simplest dataset whose vacuum manifold is a circle.

    Each sample is a stationary state repeated ``seq_len`` times:

        q_channel   = radius * (cos phi, sin phi),  phi ~ U[0, 2*pi)
        q_spectator = 0   (coords 2..dim-1)
        p           = 0

    Wake–sleep training on this data asks V_theta for a whole circle of
    critical points (an SO(2)-degenerate vacuum): the data fixes *where* the
    vacuum orbit is; whether the angular direction along it is flat (a
    protected Goldstone memory channel, F5 §4.2) is decided by the symmetry
    of the learned potential — exact by construction for
    ``potential_type="so2_invariant"``, emergent (or not) for ``"mlp"``.

    Args:
        key: JAX random key (angles are the only randomness).
        n_points: Number of circle points (= trajectories).
        seq_len: Steps per (constant) trajectory. Must exceed the training
            window size (train_chlu samples windows of length < seq_len).
        dim: Total latent dimension (>= 2); channel = coords (0, 1).
        radius: Circle radius (the designed vacuum decay constant f).

    Returns:
        Trajectories of shape (n_points, seq_len, 2*dim), rows = [q, p].
    """
    if dim < 2:
        raise ValueError(f"generate_circle_vacuum requires dim >= 2, got dim={dim}")

    angles = jax.random.uniform(key, (n_points,), minval=0.0, maxval=2.0 * jnp.pi)

    q = jnp.zeros((n_points, dim))
    q = q.at[:, 0].set(radius * jnp.cos(angles))
    q = q.at[:, 1].set(radius * jnp.sin(angles))
    p = jnp.zeros((n_points, dim))

    state = jnp.concatenate([q, p], axis=-1)  # (n_points, 2*dim)
    return jnp.repeat(state[:, None, :], seq_len, axis=1)
