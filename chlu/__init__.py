"""
CHLU - Causal Hamiltonian Learning Unit

A JAX/Equinox-based implementation of a dynamical system grounded in
symplectic mechanics with a relativistic Hamiltonian.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    # Single source of truth: the version declared in pyproject.toml.
    __version__ = _pkg_version("chlu")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.2.4"

# Core modules
from chlu.core.chlu_unit import CHLU
from chlu.core.baselines import NeuralODE, LSTMPredictor
from chlu.core.potentials import PotentialMLP

# Training
from chlu.training.train import train_chlu
from chlu.training.train_baselines import train_neural_ode, train_lstm

# Data generators
from chlu.data.figure8 import generate_figure8
from chlu.data.sine_waves import generate_sine_waves, add_noise
from chlu.data.mnist import load_mnist_pca

# Experiments
from chlu.experiments.exp_a_stability import run_experiment_a
from chlu.experiments.exp_b_noise import run_experiment_b
from chlu.experiments.exp_c_dreaming import run_experiment_c

__all__ = [
    "CHLU",
    "NeuralODE",
    "LSTMPredictor",
    "PotentialMLP",
    "train_chlu",
    "train_neural_ode",
    "train_lstm",
    "generate_figure8",
    "generate_sine_waves",
    "add_noise",
    "load_mnist_pca",
    "run_experiment_a",
    "run_experiment_b",
    "run_experiment_c",
]
