"""Smoke test for configurable sleep-buffer persistence in dynamics training.

Guards the §7.4 discrepancy fix: `training.persistent_sleep_buffer` toggles
whether evolved sleep states are written back into the replay buffer (true PCD)
or discarded (CD with fresh random negatives, the historical Exp A/B behavior).
"""

import jax
import jax.numpy as jnp

import chlu.training.train as train_mod
from chlu.config import get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.training.train import train_chlu


def _tiny_config(persistent: bool):
    config = get_default_config()
    config.training.epochs = 3
    config.training.sleep_steps = 2
    config.training.sleep_frequency = 1  # run sleep every epoch
    config.training.batch_size = 4
    config.training.buffer_capacity = 8
    config.training.persistent_sleep_buffer = persistent
    config.training.dt = 0.05
    return config


def _run_capturing_buffer(persistent: bool):
    """Run a tiny train_chlu and return (initial_buffer, final_buffer)."""
    captured = {}

    class SpyBuffer(train_mod.ReplayBuffer):
        def initialize_random(self, key, scale=1.0):
            super().initialize_random(key, scale)
            self.initial_snapshot = jnp.array(self.buffer)
            captured["buf"] = self

    orig = train_mod.ReplayBuffer
    train_mod.ReplayBuffer = SpyBuffer
    try:
        key = jax.random.PRNGKey(0)
        model = CHLU(dim=2, hidden=8, key=jax.random.PRNGKey(1))
        # (1 trajectory, T=20, state_dim=4)
        data = jnp.zeros((1, 20, 4))
        train_chlu(
            model, data, key=key, config=_tiny_config(persistent), window_size=10
        )
    finally:
        train_mod.ReplayBuffer = orig

    buf = captured["buf"]
    return buf.initial_snapshot, buf.buffer


def test_sleep_buffer_persists_only_when_enabled():
    """Buffer contents change across epochs iff persistent_sleep_buffer=True."""
    init_off, final_off = _run_capturing_buffer(persistent=False)
    assert jnp.allclose(init_off, final_off), (
        "Buffer must be unchanged when persistent_sleep_buffer=False"
    )

    init_on, final_on = _run_capturing_buffer(persistent=True)
    assert not jnp.allclose(init_on, final_on), (
        "Buffer must change when persistent_sleep_buffer=True"
    )
