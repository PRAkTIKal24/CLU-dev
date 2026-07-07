"""V(data)-energy anchor (training.anchor_data_energy_lambda; anchor-robustness
P11 / handover §7.14).

Guards the wake-loss anchor term added in train_chlu:
  * anchor_data_energy_lambda=0.0 is the OFF/legacy path: the term is a static
    (Python-float) branch that is never entered, so a lambda=0 run is
    bit-identical run-to-run and to a run that never sets the flag.
  * lambda>0 pins the mean potential value on the data manifold to its epoch-0
    level, so the trained potential differs from the lambda=0 run — the flag is
    genuinely wired into the gradient.

Flag-provenance (behavioral tests): CHLU dim=4, hidden=8, potential=so2_invariant,
kinetic=newtonian_learned; pure-wake (sleep_frequency>epochs), epochs=8, dt=0.05,
batch=4, buffer=8; lyapunov_penalty="max"(lambda=0.01); circle_vacuum data
(n_points=8, seq_len=17, R=1.0, seed 0); train key PRNGKey(0), model PRNGKey(1).
"""

import dataclasses

import jax
import jax.numpy as jnp

from chlu.config import get_default_config, load_config, save_config
from chlu.core.chlu_unit import CHLU
from chlu.data.circle_vacuum import generate_circle_vacuum
from chlu.training.train import train_chlu


def _anchor_config(lam: float):
    config = get_default_config()
    config.training.epochs = 8
    config.training.sleep_frequency = 1000  # > epochs => pure wake, fast + isolates anchor
    config.training.batch_size = 4
    config.training.buffer_capacity = 8
    config.training.dt = 0.05
    config.training.anchor_data_energy_lambda = lam
    return config


def _circle_data(dim=4, n_points=8, seq_len=17):
    return generate_circle_vacuum(
        jax.random.PRNGKey(0), n_points=n_points, seq_len=seq_len, dim=dim, radius=1.0
    )


def _train(lam: float):
    model = CHLU(
        dim=4,
        hidden=8,
        kinetic_mode="newtonian_learned",
        potential_type="so2_invariant",
        key=jax.random.PRNGKey(1),
    )
    data = _circle_data()
    trained, _, _ = train_chlu(
        model, data, key=jax.random.PRNGKey(0), config=_anchor_config(lam),
        window_size=16,
    )
    return trained


# ------------------------------------------------------------------ config


def test_anchor_default_is_zero():
    """The anchor is OFF by default (bit-compatible with all prior runs)."""
    assert get_default_config().training.anchor_data_energy_lambda == 0.0


def test_anchor_config_round_trip(tmp_path):
    cfg = get_default_config()
    cfg.training.anchor_data_energy_lambda = 42.0
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.training.anchor_data_energy_lambda == 42.0
    # full default round trip is unchanged otherwise
    cfg2 = get_default_config()
    path2 = tmp_path / "config2.yaml"
    save_config(cfg2, path2)
    assert dataclasses.asdict(load_config(path2)) == dataclasses.asdict(cfg2)


# ------------------------------------------------------------------ behavior


def test_anchor_off_is_bit_compatible():
    """lambda=0.0 is the legacy path: deterministic and bit-identical run-to-run
    (the anchor term is a never-entered static branch)."""
    a = _train(0.0)
    b = _train(0.0)
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert all(
        jnp.array_equal(x, y) for x, y in zip(leaves_a, leaves_b, strict=True)
    ), "anchor lambda=0.0 must be bit-identical run-to-run"


def test_anchor_on_changes_training():
    """lambda>0 pins mean V(data) => the trained model differs from lambda=0."""
    base = _train(0.0)
    anchored = _train(100.0)
    leaves_base = jax.tree_util.tree_leaves(base)
    leaves_anchored = jax.tree_util.tree_leaves(anchored)
    assert any(
        not jnp.array_equal(x, y)
        for x, y in zip(leaves_base, leaves_anchored, strict=True)
    ), "anchor lambda=100 must change the trained parameters vs lambda=0"
