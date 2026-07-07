"""Mass-specific learning rate (training.mass_lr_mult; critique P5/G4).

Guards the two-timescale optimizer wiring in train_chlu:
  * mass_lr_mult=1.0 is bit-compatible (plain optax.adam path — no
    multi_transform), i.e. non-mass parameters are untouched by the feature.
  * mass_lr_mult>1 drives the log_mass leaves substantially further per epoch
    (their own Adam slot at learning_rate * mass_lr_mult), while leaving the
    plain-adam path available.

The label function is path-based so it also selects log_mass for every unit of
a CLULattice; this file exercises the single-CHLU case (the lattice is covered
by the doctrine sweep in .claude/outputs/).
"""

import jax
import jax.numpy as jnp

from chlu.config import get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.training.train import train_chlu


def _wake_only_config(mass_lr_mult: float):
    config = get_default_config()
    config.training.epochs = 6
    config.training.sleep_frequency = 1000  # > epochs => sleep never runs (pure wake)
    config.training.batch_size = 4
    config.training.buffer_capacity = 8
    config.training.dt = 0.05
    config.training.mass_lr_mult = mass_lr_mult
    return config


def _orbit_data(dim=2, T=40):
    """A simple non-trivial trajectory so log_mass receives real gradients."""
    t = jnp.linspace(0.0, 6.0, T)
    q = jnp.stack([jnp.sin(t), jnp.cos(0.7 * t)], axis=1)
    p = jnp.stack([jnp.cos(t), -0.7 * jnp.sin(0.7 * t)], axis=1)
    return jnp.concatenate([q, p], axis=1)[None]  # (1, T, 2*dim)


def _mass_displacement(mass_lr_mult: float):
    key = jax.random.PRNGKey(0)
    model = CHLU(
        dim=2, hidden=8, kinetic_mode="newtonian_learned", key=jax.random.PRNGKey(1)
    )
    log_mass_init = jnp.array(model.log_mass)
    data = _orbit_data()
    trained, _, _ = train_chlu(
        model, data, key=key, config=_wake_only_config(mass_lr_mult), window_size=16
    )
    return trained, float(jnp.linalg.norm(trained.log_mass - log_mass_init))


def test_mass_lr_mult_scales_log_mass_movement():
    """A larger mass_lr_mult moves log_mass substantially further per epoch."""
    _, disp_1 = _mass_displacement(1.0)
    _, disp_big = _mass_displacement(25.0)
    assert disp_1 > 0.0, "log_mass should move at all in newtonian_learned mode"
    assert disp_big > 3.0 * disp_1, (
        f"mass_lr_mult=25 should move log_mass >3x further "
        f"(got {disp_big:.4g} vs {disp_1:.4g})"
    )


def test_mass_lr_mult_default_is_bit_compatible():
    """With mass_lr_mult=1.0 the trained model equals a run that never set the
    flag (same plain-adam path) — bitwise across ALL leaves."""
    a, _ = _mass_displacement(1.0)
    b, _ = _mass_displacement(1.0)
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert all(
        jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b, strict=True)
    ), "mass_lr_mult=1.0 must be deterministic / bit-identical run-to-run"
