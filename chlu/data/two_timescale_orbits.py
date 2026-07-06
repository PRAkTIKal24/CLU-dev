"""Two-timescale composite-orbit dataset for the CLU-lattice training smoke.

The simplest "two timescales, two bands" task (Thread-5 / F5 §5.3): a joint
signal whose per-unit channels are circular orbits of REFERENCE isotropic
harmonic oscillators with different inertial masses but a SHARED stiffness:

    unit i channel:  q_i(t) = R (cos(omega_i t + phi), sin(omega_i t + phi)),
                     p_i(t) = M_i * dq_i/dt,      omega_i = sqrt(k_i / M_i).

With the default (data_masses=(4.0, 0.25), data_omegas=(0.5, 2.0)) the
stiffnesses coincide, k_i = M_i * omega_i^2 = 1 for both units: the two
timescales come from the inertial-mass band ALONE, on one shared curvature
scale — exactly the F5 §5 role-3 geometry ("M redistributes a shared
landscape's timescales"). Momenta are physically consistent (p = M dq/dt), so
the mass band is identifiable from the phase-space data: a lattice whose
kinetic masses match the band can fit with the shared simple potential, while
a mass-uniform lattice must first travel in log_mass space — the banding
prior is a learnability head start, not extra capacity (F5 §5 "honest
deflation": constant M is a gauge choice at linear order).

Nomenclature (F5 Def-2): inertial mass M (kinetic), spectral mass mu
(here mu_i = omega_i = sqrt(k_i / M_i)).
"""

from typing import Sequence

import jax
import jax.numpy as jnp


def generate_two_timescale_orbits(
    key: jax.random.PRNGKey,
    n_traj: int,
    seq_len: int,
    dt: float = 0.05,
    omegas: Sequence[float] = (0.5, 2.0),
    masses: Sequence[float] = (4.0, 0.25),
    radius: float = 1.0,
) -> jnp.ndarray:
    """
    Generate joint two-unit circular-orbit trajectories.

    Args:
        key: PRNG key (per-trajectory, per-unit phase offsets are the only
            randomness).
        n_traj: number of trajectories.
        seq_len: steps per trajectory.
        dt: sampling step (the model's integration step in the smoke).
        omegas: per-unit angular frequencies (slow, fast).
        masses: per-unit reference inertial masses M_i (p = M_i * dq/dt).
        radius: orbit radius R for every unit.

    Returns:
        (n_traj, seq_len, 2*D) trajectories with D = 2 * n_units; rows are
        [q_1, ..., q_N, p_1, ..., p_N] in the CLULattice concatenated layout.
    """
    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    if omegas.shape != masses.shape:
        raise ValueError(
            f"omegas {omegas.shape} and masses {masses.shape} must align (one per unit)"
        )
    n_units = omegas.shape[0]

    phases = jax.random.uniform(key, (n_traj, n_units), minval=0.0, maxval=2.0 * jnp.pi)
    t = jnp.arange(seq_len) * dt  # (T,)

    # angle(traj, unit, time) = omega_u * t + phase
    angle = omegas[None, :, None] * t[None, None, :] + phases[:, :, None]

    # Per-unit channel blocks (n_traj, n_units, T, 2)
    q = radius * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    # p = M * dq/dt = M * R * omega * (-sin, cos)
    p_scale = (masses * omegas)[None, :, None, None]
    p = radius * p_scale * jnp.stack([-jnp.sin(angle), jnp.cos(angle)], axis=-1)

    # Concatenate unit blocks: (n_traj, T, 2*n_units) each for q and p
    q_flat = q.transpose(0, 2, 1, 3).reshape(n_traj, seq_len, 2 * n_units)
    p_flat = p.transpose(0, 2, 1, 3).reshape(n_traj, seq_len, 2 * n_units)
    return jnp.concatenate([q_flat, p_flat], axis=-1)
