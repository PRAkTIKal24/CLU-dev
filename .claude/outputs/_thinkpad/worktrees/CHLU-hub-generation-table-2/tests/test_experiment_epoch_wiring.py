"""Regression tests for --quick / train_epochs wiring (handover §7.10).

run_experiment_a/b set config.experiment_X.train_epochs (what --quick
overrides), but the trainers default to config.training.epochs — so quick
mode silently trained full-length. The experiments must pass the experiment
override explicitly (epochs=...).
"""

import jax
import jax.numpy as jnp
import pytest

import chlu.experiments.exp_a_stability as exp_a_mod
import chlu.experiments.exp_b_noise as exp_b_mod
from chlu.config import get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.training.train import train_chlu


class _StopExperiment(Exception):
    """Raised by a spy trainer to abort the experiment after the calls under test."""


def test_train_chlu_epochs_override_controls_loss_history():
    """train_chlu(epochs=N) trains exactly N epochs even if config says 1000."""
    config = get_default_config()
    config.training.epochs = 1000  # must be ignored when epochs= is passed
    config.training.sleep_frequency = 10_000  # skip sleep phase for speed
    model = CHLU(dim=2, hidden=8, key=jax.random.PRNGKey(1))
    data = jnp.zeros((1, 20, 4))

    _, losses, _ = train_chlu(
        model,
        data,
        key=jax.random.PRNGKey(0),
        config=config,
        epochs=3,
        window_size=10,
    )
    assert len(losses) == 3


def _spy_trainers(monkeypatch, module, captured):
    """Spy train_chlu/train_neural_ode: record the epochs kwarg, then abort."""

    def spy_train_chlu(model, data, key, config=None, **kwargs):
        captured["chlu"] = kwargs.get("epochs")
        # Matches the (model, losses, target_energy) return signature
        return model, jnp.zeros(1), 0.0

    def spy_train_node(model, data, key, config=None, **kwargs):
        captured["node"] = kwargs.get("epochs")
        raise _StopExperiment  # train_lstm call is the identical pattern

    monkeypatch.setattr(module, "train_chlu", spy_train_chlu)
    monkeypatch.setattr(module, "train_neural_ode", spy_train_node)


def test_exp_a_passes_train_epochs_to_trainers(monkeypatch, tmp_path):
    captured = {}
    _spy_trainers(monkeypatch, exp_a_mod, captured)

    config = get_default_config()
    config.experiment_a.train_epochs = 7  # quick-mode-style override
    config.experiment_a.n_train_cycles = 1
    config.experiment_a.n_test_cycles = 1

    with pytest.raises(_StopExperiment):
        exp_a_mod.run_experiment_a(config=config, save_dir=str(tmp_path))

    assert captured["chlu"] == 7
    assert captured["node"] == 7


def test_exp_b_passes_train_epochs_to_trainers(monkeypatch, tmp_path):
    captured = {}
    _spy_trainers(monkeypatch, exp_b_mod, captured)

    config = get_default_config()
    config.experiment_b.train_epochs = 7
    config.experiment_b.n_waves = 10  # keep data generation fast
    config.experiment_b.steps = 64

    with pytest.raises(_StopExperiment):
        exp_b_mod.run_experiment_b(config=config, save_dir=str(tmp_path))

    assert captured["chlu"] == 7
    assert captured["node"] == 7
