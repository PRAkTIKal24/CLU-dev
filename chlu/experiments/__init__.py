"""CHLU experiments for ICLR Workshop paper."""

from chlu.experiments.exp_a_stability import run_experiment_a
from chlu.experiments.exp_b_noise import run_experiment_b
from chlu.experiments.exp_c_dreaming import run_experiment_c
from chlu.experiments.exp_d_goldstone import run_experiment_d
from chlu.experiments.exp_v1_calibration import (
    run_experiment_v1_calibration,
    run_v1_hopfield_regime_map,
)
from chlu.experiments.exp_v1_gate import run_experiment_v1_gate
from chlu.experiments.exp_v1_wormhole import run_experiment_v1_wormhole
from chlu.experiments.exp_lattice import run_experiment_lattice
from chlu.experiments.exp_minus_physics import run_experiment_minus_physics

__all__ = [
    "run_experiment_a",
    "run_experiment_b",
    "run_experiment_c",
    "run_experiment_d",
    "run_experiment_v1_calibration",
    "run_v1_hopfield_regime_map",
    "run_experiment_v1_gate",
    "run_experiment_v1_wormhole",
    "run_experiment_lattice",
    "run_experiment_minus_physics",
]
