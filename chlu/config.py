"""
Central configuration management for CHLU.

This module defines all configurable parameters using dataclasses,
providing type safety and defaults for all experiments, training, and models.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    hidden_dim: int = 64
    rest_mass: float = 1.0
    speed_of_causality: float = 1.0
    log_mass_init_scale: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters for both dynamics and generative training."""

    # Common parameters
    epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 64
    dt: float = 0.05
    buffer_capacity: int = 10000

    # Dynamics training (Experiments A & B)
    lyapunov_lambda: float = 0.01
    # Penalty on the step-Jacobian log singular values (local Lyapunov
    # exponents): "max" (max_i log sigma_i, chaos-relevant, default), "sq"
    # (sum_i (log sigma_i)^2), "pos" (sum_i max(0, log sigma_i)), "none"
    # (disabled), or "legacy_degenerate" (mean_i log sigma_i — provably
    # theta-independent, identically 0.5*ln(1-gamma); reproduction only,
    # see F5 Prop-5).
    lyapunov_penalty: str = "max"
    sleep_steps: int = 500
    clamp_strength: float = 1000.0
    clamp_ramp: float = 0.5
    sleep_frequency: int = 5
    # If True, evolved sleep states are written back into the replay buffer at
    # their sampled indices (true PCD, mirrors train_generative). Default False
    # preserves the current Exp A/B behavior: CD with fresh random negatives.
    persistent_sleep_buffer: bool = False

    # Generative training (Experiment C)
    reinit_prob: float = 0.25  # Probability of resetting chains to noise
    k_steps: int = 100  # Negative phase evolution steps
    clamp_outputs: bool = True  # Enable hard pixel clamping to [-1, 1]
    energy_weight: float = 1.0  # Weight for contrastive energy loss
    input_noise_sigma: float = 0.05  # Gaussian noise std for real data (denoising EBM)

    # Friction (used differently: 0.0 for dynamics, 0.1 for generative)
    sleep_friction: float = 0.0

    # Langevin dynamics (temperature-based noise for exploration)
    sleep_temperature: float = 0.5  # Temperature for sleep phase (0.0 = deterministic)
    # Langevin noise scale: "legacy" = sqrt(2*gamma*T*dt) (historical; violates
    # the discrete FDT — per-mode effective temperature 2*dt*T/((2-gamma)*M_eff),
    # F5 Prop-9) or "fdt" = per-mode sigma_i* = sqrt(M_eff_i*T*gamma*(2-gamma))
    # (exact discrete fluctuation-dissipation; temperatures in energy units).
    # Default "legacy" preserves behavior for existing checkpoints/schedules.
    langevin_noise: str = "legacy"


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A: Stability Test."""

    # Cycle-based parameters for geometry learning
    dt: float = 0.05  # Time step
    n_train_cycles: int = 3  # Train on 3 full cycles
    n_test_cycles: int = 50  # Test on 50 full cycles
    window_size: int = 64  # Window size for sub-sequence sampling
    n_final_cycles_to_plot: int = 2  # Number of final cycles to show in plots
    train_epochs: int = 1000
    use_pretrained: bool = False  # Load pre-trained models if available
    kinetic_energy_mode: str = "newtonian_identity"  # KE calculation mode: newtonian_identity, newtonian_learned, relativistic
    # Note: chlu_dim is always 2 for Figure-8 (not configurable)
    node_dim: int = 4
    hidden_dim: int = 64

    @property
    def steps_per_cycle(self) -> int:
        """Number of steps per cycle (period = 2π)."""
        import math

        return int(2 * math.pi / self.dt)

    @property
    def train_steps(self) -> int:
        """Total training steps."""
        return self.n_train_cycles * self.steps_per_cycle

    @property
    def test_steps(self) -> int:
        """Total test steps."""
        return self.n_test_cycles * self.steps_per_cycle


@dataclass
class ExperimentBConfig:
    """Configuration for Experiment B: Noise Rejection."""

    n_waves: int = 100
    steps: int = 1000
    train_epochs: int = 1000
    use_pretrained: bool = False  # Load pre-trained models if available
    kinetic_energy_mode: str = "newtonian_learned"  # KE calculation mode
    sleep_friction: float = 0.2
    friction_ramp: float = 0.05
    use_governor: bool = True  # Use energy-based governor for dynamic friction
    governor_sensitivity: float = 0.95  # Controls correction speed for governor
    dt: float = 0.05
    sigma_min: float = 0.1
    sigma_max: float = 1.0
    n_sigma: int = 10
    chlu_dim: int = 1
    node_dim: int = 2
    hidden_dim: int = 64


@dataclass
class ExperimentCConfig:
    """Configuration for Experiment C: Dreaming/Generation."""

    pca_dim: int = 784
    train_epochs: int = 500
    use_pretrained: bool = False  # Load pre-trained models if available
    kinetic_energy_mode: str = "relativistic"  # KE calculation mode
    potential_type: str = "conv"  # Potential network type: 'mlp', 'deep_mlp', 'conv'
    n_samples: int = 10000
    dream_steps: int = 1000
    friction: float = 0.3
    dt: float = 0.05
    n_dreams: int = 64
    hidden_dim: int = 1024
    p_train_scale: float = 0.1
    q_noise_scale: float = 1.0
    p_noise_scale: float = 0.1
    snapshot_steps: List[int] = field(default_factory=lambda: [0, 200, 400, 600, 800])

    # Langevin dynamics parameters
    temperature: float = 1.0  # Base temperature for dreaming (0.0 = deterministic)
    temperature_annealing: bool = True  # Enable temperature annealing (cooling)
    temperature_start: float = 1.0  # Starting temperature for annealing
    temperature_end: float = 0.01  # Ending temperature for annealing
    annealing_schedule: str = (
        "exponential"  # Annealing schedule type: 'exponential' or 'linear'
    )

    # Initialization mode parameters
    init_mode: str = "random"  # Initialization mode: 'random' or 'centroid'
    centroid_noise_scale: float = (
        0.5  # Gaussian perturbation scale when using centroid init
    )


@dataclass
class ExperimentDConfig:
    """Configuration for Experiment D: SO(2) Goldstone memory (V2).

    Measures the F5 §3.3–§3.4 mode-budget predictions on a CLU with a
    designed SO(2) channel over coordinates (0, 1): the latch, the half-life
    law, Noether-charge decay, and kinetic isotropy. Nomenclature per F5
    Def-2: inertial mass M vs spectral mass mu — never "mass" unqualified.
    """

    dim: int = 4  # 2 channel dims + (dim-2) curved spectator dims
    hidden_dim: int = 64
    train_epochs: int = 1000
    use_pretrained: bool = False
    kinetic_energy_mode: str = "newtonian_learned"  # learned M => isotropy falsifiable is live
    potential_type: str = "so2_invariant"  # "so2_invariant" (designed) or "mlp" (emergent)
    tie_channel_mass: bool = True  # kinetic isotropy (F5 §4.1); False = broken-isotropy switch
    tilt_delta: float = 0.0  # explicit breaking delta*cos(n*theta) (GMOR probe, F5 §3.3c)
    tilt_n: int = 1
    dt: float = 0.05

    # Dataset: constant trajectories on a circle of attractors (SO(2)-degenerate vacuum)
    n_points: int = 256
    seq_len: int = 65  # training window = seq_len - 1
    circle_radius: float = 1.0

    # Measurement harness defaults
    settle_gamma: float = 0.1
    settle_steps: int = 2000
    probe_gamma: float = 0.05
    probe_steps: int = 4000
    probe_kick: float = 0.1
class ExperimentV1GateConfig:
    """Configuration for the V1 L0 gate experiment: boost-retry cascade on MQAR.

    An energy-based associative memory (CHLU trained with generative PCD on a
    per-episode MQAR dictionary) is queried by governed relaxation; failed
    retrievals are retried with mass-weighted symplectic squeezes S^(M)
    (F5 Def-6/7). Measures Q1 (residual-energy-vs-correctness calibration)
    and Q2 (boost-retry recovery vs matched-compute controls).
    """

    # --- MQAR task (Zoology-legible knobs) ---
    vocab_size: int = 256  # scaled down from Zoology's 8192 for small CLU dims
    embed_dim: int = 16  # per-token embedding; CLU dim = 2 * embed_dim (key||value)
    # embedding norm scale (entries ~ scale/sqrt(embed_dim)); default 2.0 puts
    # entries at ~0.5 so data lives at the scale train_generative's negative
    # chains explore (buffer N(0,1), re-init U(-1,1), clamp [-1,1])
    embed_scale: float = 2.0
    # Difficulty grid: list of [seq_len N, num_kv_pairs] levels
    difficulty_levels: List[List[int]] = field(
        default_factory=lambda: [[64, 4], [64, 8], [128, 16], [128, 32], [256, 64]]
    )
    min_trials_per_level: int = 64  # episodes are added until this many queries
    max_episodes_per_level: int = 8
    gap_distribution: str = "uniform"  # or "powerlaw" ("Based"-style gaps)
    powerlaw_alpha: float = 0.01

    # --- memory model & PCD training (per-episode "write") ---
    kinetic_energy_mode: str = "relativistic"
    potential_type: str = "mlp"  # coercive potential (confinement) — see F5 Prop-10
    hidden_dim: int = 128
    train_epochs: int = 500
    train_lr: float = 1e-3
    train_batch_size: int = 32
    train_k_steps: int = 50
    train_buffer_capacity: int = 256
    train_friction: float = 0.3
    train_temperature: float = 0.3
    train_input_noise_sigma: float = 0.05
    use_pretrained: bool = False

    # --- retrieval / cascade (F5 Def-7, single shell) ---
    # Cue-conditioned retrieval: freeze the key half (q_k = cue, p_k = 0) and
    # relax only the value subspace — the standard associative-memory readout
    # (Hopfield's setting); the frozen-coordinate map is the legitimate
    # sub-system Hamiltonian dynamics for all three kinetic modes.
    clamp_key: bool = True
    dt: float = 0.05
    relax_steps: int = 300  # base governed relaxation n0
    retry_relax_steps: int = 150  # re-relaxation per line-search candidate
    governor_sensitivity: float = 0.95
    retry_budget: int = 3  # B
    zeta_grid: List[float] = field(
        default_factory=lambda: [-0.6, -0.3, -0.15, 0.15, 0.3, 0.6]
    )
    zeta_scale_per_retry: float = 1.5  # retry b uses zeta_grid * scale**b
    n_tau: int = 9  # post-hoc tau sweep = this many quantiles of the residual

    # --- baselines / comparison flags ---
    compare_raw_squeeze: bool = True  # mass-blind S_zeta cascade (F5 says: flag only)
    compare_noise_kick: bool = True  # random p-kick retries at matched energy injection
    hopfield_beta: float = 20.0  # modern-Hopfield softmax inverse temperature


@dataclass
class DataConfig:
    """Data generation and processing parameters."""

    figure8_dt: float = 0.01
    figure8_scale: float = 1.0
    sine_dt: float = 0.01
    sine_freq_min: float = 0.5
    sine_freq_max: float = 2.0
    sine_amp_min: float = 0.5
    sine_amp_max: float = 1.5
    mnist_pca_dim: int = 784
    train_test_split: float = 0.8


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str = "default"
    seed: int = 42
    device: str = "auto"
    save_dir: Optional[str] = None


@dataclass
class CHLUConfig:
    """Master configuration containing all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_a: ExperimentAConfig = field(default_factory=ExperimentAConfig)
    experiment_b: ExperimentBConfig = field(default_factory=ExperimentBConfig)
    experiment_c: ExperimentCConfig = field(default_factory=ExperimentCConfig)
    experiment_d: ExperimentDConfig = field(default_factory=ExperimentDConfig)
    experiment_v1_gate: ExperimentV1GateConfig = field(
        default_factory=ExperimentV1GateConfig
    )
    data: DataConfig = field(default_factory=DataConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)


def get_default_config() -> CHLUConfig:
    """Get default configuration with all parameters set to their defaults."""
    return CHLUConfig()


def load_config(path: Path) -> CHLUConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        CHLUConfig object with values from file
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    def filter_valid_fields(config_class, data_dict):
        """Filter dict to only include fields that exist in the dataclass."""
        if not data_dict:
            return {}
        valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
        filtered = {}
        for k, v in data_dict.items():
            if k in valid_fields:
                # Convert string numbers (including scientific notation) to proper types
                if isinstance(v, str):
                    try:
                        # Try int first (for whole numbers), then float
                        if "." not in v and "e" not in v.lower():
                            filtered[k] = int(v)
                        else:
                            filtered[k] = float(v)
                    except (ValueError, TypeError):
                        filtered[k] = v
                else:
                    filtered[k] = v
        return filtered

    # Reconstruct nested dataclasses with field filtering
    config = CHLUConfig(
        model=ModelConfig(**filter_valid_fields(ModelConfig, data.get("model", {}))),
        training=TrainingConfig(
            **filter_valid_fields(TrainingConfig, data.get("training", {}))
        ),
        experiment_a=ExperimentAConfig(
            **filter_valid_fields(ExperimentAConfig, data.get("experiment_a", {}))
        ),
        experiment_b=ExperimentBConfig(
            **filter_valid_fields(ExperimentBConfig, data.get("experiment_b", {}))
        ),
        experiment_c=ExperimentCConfig(
            **filter_valid_fields(ExperimentCConfig, data.get("experiment_c", {}))
        ),
        experiment_d=ExperimentDConfig(
            **filter_valid_fields(ExperimentDConfig, data.get("experiment_d", {}))
        ),
        experiment_v1_gate=ExperimentV1GateConfig(
            **filter_valid_fields(
                ExperimentV1GateConfig, data.get("experiment_v1_gate", {})
            )
        ),
        data=DataConfig(**filter_valid_fields(DataConfig, data.get("data", {}))),
        project=ProjectConfig(
            **filter_valid_fields(ProjectConfig, data.get("project", {}))
        ),
    )
    return config


def save_config(config: CHLUConfig, path: Path) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: CHLUConfig object to save
        path: Path where to save the YAML file
    """
    # Convert to nested dict
    config_dict = {
        "model": asdict(config.model),
        "training": asdict(config.training),
        "experiment_a": asdict(config.experiment_a),
        "experiment_b": asdict(config.experiment_b),
        "experiment_c": asdict(config.experiment_c),
        "experiment_d": asdict(config.experiment_d),
        "experiment_v1_gate": asdict(config.experiment_v1_gate),
        "data": asdict(config.data),
        "project": asdict(config.project),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
