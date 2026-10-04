"""config."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
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

    epochs: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 64
    dt: float = 0.05
    buffer_capacity: int = 10000
    lyapunov_lambda: float = 0.01
    lyapunov_penalty: str = "max"
    sleep_steps: int = 500
    clamp_strength: float = 1000.0
    clamp_ramp: float = 0.5
    sleep_frequency: int = 5
    persistent_sleep_buffer: bool = False
    mass_lr_mult: float = 1.0
    anchor_data_energy_lambda: float = 0.0
    reinit_prob: float = 0.25
    k_steps: int = 100
    clamp_outputs: bool = True
    energy_weight: float = 1.0
    input_noise_sigma: float = 0.05
    sleep_friction: float = 0.0
    sleep_temperature: float = 0.5
    langevin_noise: str = "legacy"
    friction_field: str = "none"
    friction_field_k: int = 1
    friction_field_gamma_max: float = 0.5
    friction_field_width: float = 0.25
    friction_field_gate: str = "sigmoid"
    friction_field_init_radius: float = 1.0
    friction_field_init_strength: float = 0.15
    friction_field_init_center_scale: float = 1.0
    friction_field_fixed_centers: Optional[List[List[float]]] = None
    friction_field_fixed_radius: float = 0.6
    friction_field_fixed_strength: float = 0.3
    friction_field_protect_lambda: float = 1.0
    friction_field_hallu_lambda: float = 1.0
    friction_field_hallu_gate: str = "energy"
    friction_field_c1_lambda: float = 0.0
    friction_field_lr: Optional[float] = 0.01
    friction_field_adaptive_k: bool = False
    friction_field_max_k: int = 8
    friction_field_spawn_threshold: float = 5.0
    friction_field_spawn_min_dist: float = 0.5
    friction_field_spawn_radius: float = 0.5
    friction_field_spawn_strength: float = 0.15
    friction_field_prune_floor: float = 0.02


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A: Stability Test."""

    dt: float = 0.05
    n_train_cycles: int = 3
    n_test_cycles: int = 50
    window_size: int = 64
    n_final_cycles_to_plot: int = 2
    train_epochs: int = 1000
    use_pretrained: bool = False
    kinetic_energy_mode: str = "newtonian_identity"
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

    n_periods: int = 100
    steps: int = 1000
    train_epochs: int = 1000
    use_pretrained: bool = False
    kinetic_energy_mode: str = "newtonian_learned"
    sleep_friction: float = 0.2
    friction_ramp: float = 0.05
    use_governor: bool = True
    governor_sensitivity: float = 0.95
    dt: float = 0.05
    sigma_min: float = 0.1
    sigma_max: float = 1.0
    n_sigma: int = 10
    nclu_dim: int = 1
    node_dim: int = 2
    hidden_dim: int = 64


@dataclass
class ExperimentCConfig:
    pca_dim: int = 784
    train_epochs: int = 500
    use_pretrained: bool = False
    kinetic_energy_mode: str = "relativistic"
    potential_type: str = "conv"
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
    temperature: float = 1.0
    temperature_annealing: bool = True
    temperature_start: float = 1.0
    temperature_end: float = 0.01
    annealing_schedule: str = "exponential"
    init_mode: str = "random"
    centroid_noise_scale: float = 0.5


@dataclass
class ExperimentDConfig:
    """Configuration for Experiment D: SO(2) Goldstone memory (V2).

    Measures the F5 §3.3–§3.4 mode-budget predictions on a CLU with a
    designed SO(2) channel over coordinates (0, 1): the latch, the half-life
    law, Noether-charge decay, and kinetic isotropy. Nomenclature per F5
    Def-2: inertial mass M vs spectral mass mu — never "mass" unqualified.
    """

    dim: int = 4
    hidden_dim: int = 64
    train_epochs: int = 150
    sleep_mode: str = "on"
    use_pretrained: bool = False
    kinetic_energy_mode: str = "newtonian_learned"
    potential_type: str = "so2_invariant"
    tie_channel_mass: bool = True
    tilt_delta: float = 0.0
    tilt_n: int = 1
    spurion_delta: float = 0.0
    spurion_angle: float = 0.0
    dt: float = 0.05
    n_points: int = 256
    seq_len: int = 65
    circle_radius: float = 1.0
    settle_gamma: float = 0.1
    settle_steps: int = 2000
    probe_gamma: float = 0.05
    probe_steps: int = 4000
    probe_kick: float = 0.1


@dataclass
class ExperimentV1GateConfig:
    vocab_size: int = 256
    embed_dim: int = 16
    embed_scale: float = 2.0
    difficulty_levels: List[List[int]] = field(
        default_factory=lambda: [[64, 4], [64, 8], [128, 16], [128, 32], [256, 64]]
    )
    min_trials_per_level: int = 64
    max_episodes_per_level: int = 8
    gap_distribution: str = "uniform"
    powerlaw_alpha: float = 0.01
    kinetic_energy_mode: str = "relativistic"
    potential_type: str = "mlp"
    hidden_dim: int = 128
    train_epochs: int = 500
    train_lr: float = 0.001
    train_batch_size: int = 32
    train_k_steps: int = 50
    train_buffer_capacity: int = 256
    train_friction: float = 0.3
    train_temperature: float = 0.3
    train_input_noise_sigma: float = 0.05
    use_pretrained: bool = False
    clamp_key: bool = True
    dt: float = 0.05
    relax_steps: int = 300
    retry_relax_steps: int = 150
    governor_sensitivity: float = 0.95
    retry_budget: int = 3
    zeta_grid: List[float] = field(default_factory=lambda: [-0.6, -0.3, -0.15, 0.15, 0.3, 0.6])
    zeta_scale_per_retry: float = 1.5
    n_tau: int = 9
    compare_raw_squeeze: bool = True
    compare_noise_kick: bool = True
    hopfield_beta: float = 20.0
    calib_difficulty_levels: List[List[int]] = field(
        default_factory=lambda: [[128, 16], [128, 24], [128, 32]]
    )
    calib_n_seeds: int = 5
    calib_min_trials_per_level: int = 128
    calib_max_episodes_per_level: int = 8
    calib_probes_per_key: int = 8
    calib_cue_noise_scales: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.3])
    calib_n_impostors: int = 16
    calib_features: str = "r_margin"
    calib_l2: float = 1.0
    calib_fit_all_stages: bool = True
    calib_n_stages: int = 3
    calib_stage_steps: int = 900
    calib_p_exit: float = 0.5
    calib_n_policy_taus: int = 25
    calib_risk_targets: List[float] = field(default_factory=lambda: [0.05, 0.1])
    calib_ltt_delta: float = 0.1
    regime_capacity_levels: List[List[int]] = field(
        default_factory=lambda: [[128, 32], [256, 64], [384, 96]]
    )
    regime_stress_axis: str = "correlation"
    regime_stress_grid: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 0.9])
    regime_n_seeds: int = 3
    regime_episodes_per_cell: int = 1
    regime_n_clusters: int = 8
    regime_base_eval_noise: float = 0.0
    regime_base_correlation: float = 0.0
    regime_comparable_margin: float = 0.03
    hopfield_gate_correlation: float = 0.0
    hopfield_gate_eval_noise: float = 0.0


@dataclass
class ExperimentV1WormholeConfig:
    n_units_values: List[int] = field(default_factory=lambda: [4, 8])
    embed_dim: int = 12
    embed_scale: float = 2.0
    vocab_size: int = 128
    kv_per_unit: int = 3
    n_seeds: int = 2
    trials_per_type: int = 48
    query_cue_noise: float = 0.05
    kinetic_energy_mode: str = "relativistic"
    potential_type: str = "mlp"
    hidden_dim: int = 128
    train_epochs: int = 400
    train_lr: float = 0.001
    train_batch_size: int = 16
    train_k_steps: int = 50
    train_buffer_capacity: int = 128
    train_friction: float = 0.3
    train_temperature: float = 0.3
    train_input_noise_sigma: float = 0.05
    dt: float = 0.05
    relax_steps: int = 250
    route_steps: int = 250
    governor_sensitivity: float = 0.95
    kappa_wormhole: float = 2.0
    kappa_chain: float = 2.0
    gate_z_threshold: float = 0.0
    gate_z_width: float = 0.7
    gate_route_threshold: float = 0.5
    calib_probes_per_key: int = 8
    calib_cue_noise_scales: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.3])
    calib_features: str = "r_margin"
    calib_l2: float = 1.0
    calib_p_route: float = 0.5
    impostor_policy: str = "all_others"
    router_hidden_dim: int = 32
    router_epochs: int = 300
    router_lr: float = 0.003
    router_l2: float = 0.001
    router_p_route: float = 0.5
    workload_mixes: List[List[float]] = field(
        default_factory=lambda: [[0.5, 0.5], [0.8, 0.2], [0.95, 0.05]]
    )
    flops_grad_factor: float = 6.0
    flops_verlet_grads: float = 2.0
    hopfield_beta: float = 20.0


@dataclass
class ExperimentLatticeConfig:
    coupling_type: str = "auto"
    coupling_dim: int = 2
    kappa_c: float = 0.05
    proj_init_scale: float = 0.1
    proj_init_mode: str = "random"
    kinetic_energy_mode: str = "newtonian_learned"
    hidden_dim: int = 32
    dt: float = 0.05
    hat_lambda: float = 1.0
    vacuum_radius: float = 1.0
    channel_inertia: float = 1.0
    kappa_sweep: List[float] = field(default_factory=lambda: [0.003, 0.01, 0.03, 0.1, 0.3])
    sync_delta0: float = 0.4
    sync_max_steps: int = 4000
    probe_gamma: float = 0.2
    probe_kick: float = 0.05
    max_probe_steps: int = 24000
    latch_steps: int = 3000
    scaling_sizes: List[int] = field(default_factory=lambda: [2, 4, 8])
    scaling_steps: int = 2000
    wormhole_gate_threshold: float = 1.0
    wormhole_gate_width: float = 0.25
    gate_energy_mode: str = "free_energy"
    train_epochs: int = 300
    use_pretrained: bool = False
    train_n_traj: int = 64
    train_seq_len: int = 256
    train_window: int = 64
    data_omegas: List[float] = field(default_factory=lambda: [0.5, 2.0])
    data_masses: List[float] = field(default_factory=lambda: [4.0, 0.25])
    data_radius: float = 1.0
    banded_mass_scales: List[float] = field(default_factory=lambda: [4.0, 0.25])
    train_kappa_c: float = 0.01
    eval_steps: int = 255


@dataclass
class ExperimentPaidAccessConfig:
    dim: int = 2
    c: float = 1.0
    rest_mass: float = 1.0
    dt: float = 0.05
    reach_steps: int = 100
    mass_band: List[float] = field(default_factory=lambda: [4.0, 0.25])
    barrier_height: float = 1.0
    basin_curvature: float = 4.0
    basin_distances: List[float] = field(default_factory=lambda: [0.8, 1.6, 2.4, 3.2, 4.0, 5.0])
    gamma: float = 0.1
    relax_steps: int = 400
    init_momentum: float = 1.2
    zeta_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    )
    capture_radius: float = 0.35
    throat_depth: float = 1.5
    latch_radius: float = 3.0
    latch_momentum: float = 0.5
    payoff_latch_samples: int = 16
    payoff_capture_jitter: float = 0.3
    bibo_k: float = 1.0
    bibo_quartic: float = 0.02
    bibo_exit_distances: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 3.6, 4.0, 5.0])
    bibo_init_momentum: float = 0.3
    bibo_steps: int = 2000
    bibo_gamma: float = 0.02
    bibo_escape_radius: float = 20.0
    bibo_margin: float = 0.001
    landing_tol: float = 0.4
    n_seeds: int = 5
    seed0: int = 0
    quick_seeds: int = 2
    quick_distances: int = 3


@dataclass
class ExperimentS1Config:
    dt: float = 0.05
    n_train_cycles: int = 3
    window_size: int = 64
    train_epochs: int = 500
    hidden_dim: int = 64
    kinetic_energy_mode: str = "newtonian_identity"
    use_pretrained: bool = False
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    sleep_steps: int = 100
    noise_center: List[float] = field(default_factory=lambda: [1.5, 1.5])
    noise_q_std: float = 0.15
    noise_p_std: float = 0.6
    buffer_noise_frac: float = 0.3
    eval_clean_steps: int = 2000
    eval_kick_steps: int = 400
    n_injections: int = 16
    global_gamma_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    )
    governor_sensitivity: float = 0.95
    learned_k_values: List[int] = field(default_factory=lambda: [1, 4])
    oracle_radius: float = 0.6
    oracle_strength: float = 0.3
    oracle_width: float = 0.1


@dataclass
class ExperimentKTConfig:
    lam: float = 1.0
    f: float = 1.0
    kappa: float = 0.05
    rstar: float = 1.0
    dt: float = 0.02
    gamma: float = 0.1
    n_walkers: int = 256
    kinetic_mode: str = "newtonian_learned"
    langevin_noise: str = "fdt"
    winding1d_n_values: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    winding1d_tj: float = 1.0
    winding1d_chunks: int = 300
    winding1d_chunks_large: int = 200
    winding1d_chunk_steps: int = 100
    winding1d_seed: int = 31
    winding1d_msd_fit_max: Optional[float] = None
    winding2d_l_values: List[int] = field(default_factory=lambda: [8, 12, 16])
    winding2d_tj_values: List[float] = field(default_factory=lambda: [0.6, 0.7, 1.1, 1.3])
    winding2d_nwalk: int = 24
    winding2d_nmax_below: int = 20000
    winding2d_nmax_above: int = 6000
    winding2d_tkt_over_j: float = 0.9
    winding2d_seed: int = 700
    bridge_l: int = 8
    bridge_tj_values: List[float] = field(default_factory=lambda: [0.7, 0.85, 1.0])
    bridge_chunks: int = 40
    bridge_burn_chunks: int = 10
    bridge_chunk_steps: int = 200
    bridge_seed: int = 7
    bridge_equil_seed: int = 1234
    bridge_equil_sweeps: int = 1500
    reduced_l_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    reduced_tj_values: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]
    )
    reduced_seeds: List[int] = field(default_factory=lambda: [100, 101, 102])
    reduced_nwalk_small: int = 4
    reduced_nwalk_large: int = 2
    reduced_therm_small: int = 1500
    reduced_therm_large: int = 3000
    reduced_meas_small: int = 4000
    reduced_meas_large: int = 6000
    reduced_meas_every: int = 5
    reduced_twist_l_values: List[int] = field(default_factory=lambda: [8, 16])
    reduced_twist_tj_values: List[float] = field(default_factory=lambda: [0.6, 0.8, 0.9, 1.0, 1.1])
    reduced_twist_a: float = 0.2
    reduced_broken_l: int = 16
    reduced_broken_h2: float = 1.0
    reduced_broken_tj_values: List[float] = field(default_factory=lambda: [0.6, 0.8, 0.9, 1.0, 1.2])


@dataclass
class ExperimentMinusPhysicsConfig:
    dim: int = 4
    hidden_dim: int = 64
    potential_type: str = "mlp"
    kinetic_energy_mode: str = "newtonian_learned"
    dt: float = 0.05
    train_epochs: int = 150
    n_seeds: int = 3
    measure_erosion: bool = True
    n_points: int = 256
    seq_len: int = 65
    circle_radius: float = 1.0
    settle_gamma: float = 0.1
    settle_steps: int = 2000
    settle_bound: float = 20.0
    probe_gamma: float = 0.05
    probe_steps: int = 4000
    probe_kick: float = 0.1
    eval_steps: int = 400


@dataclass
class ExperimentRetrievalConfig:
    lam: float = 1.0
    f: float = 1.0
    barrier: float = 0.2
    payload_kappa: float = 1.0
    bump_width: float = 0.05
    payload_seed: int = 0
    dt: float = 0.05
    gamma: float = 0.02
    steps: int = 1200
    tail_frac: float = 0.25
    n_subsample: int = 8
    n_query_per_item: int = 64
    query_sigma_theta: float = 0.15
    query_sigma_r: float = 0.05
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    survival_fracs: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.99])
    mass_probe_K: int = 8
    mass_probe_p: float = 0.5
    mass_log_lo: float = -1.5
    mass_log_hi: float = 1.5
    mass_n: int = 25
    mass_n_vec: int = 9
    mass_jitter_rel: float = 0.01
    mass_jitter_n: int = 8
    mass_robust_threshold: float = 0.9
    tm_beta: float = 1.0
    tm_d: float = 1.0
    tm_write_theta: float = 0.7
    tm_write_dr: float = 0.3
    tm_write_sign: int = 1
    tm_steps: int = 2000
    item_counts: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    selectivity_threshold: float = 0.9
    restructure_K: int = 8
    restructure_offsets: List[int] = field(default_factory=lambda: [1, 2, 4])
    address_lr: float = 0.05
    address_steps: int = 300
    smooth_n_theta: int = 181
    smooth_rollout_steps: List[int] = field(default_factory=lambda: [25, 50, 100, 200, 400, 800])
    smooth_gammas: List[float] = field(
        default_factory=lambda: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    )
    anneal_gamma_lo: float = 0.001
    anneal_n_stages: int = 6


@dataclass
class ExperimentDimScalingConfig:
    R: float = 1.0
    wall_margin: float = 0.5
    well_width: float = 0.15
    well_depth: float = 1.0
    payload_kappa: float = 0.1
    c_conf: float = 10.0
    site_seed: int = 0
    payload_seed: int = 0
    dt: float = 0.05
    gamma: float = 0.02
    steps: int = 1200
    tail_frac: float = 0.25
    n_subsample: int = 8
    rollout_chunk: int = 512
    n_query_per_item: int = 32
    max_total_queries: int = 8192
    min_query_per_item: int = 4
    query_sigma: float = 0.15
    query_noise_mode: str = "fixed_norm"
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    selectivity_threshold: float = 0.9
    blank_margin: float = 0.15
    dims: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16])
    k_ladder: List[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    k_cap: int = 2048
    capture_n_dirs: int = 16
    capture_n_offsets: int = 24
    capture_max_offset: float = 1.0
    width_sweep_dims: List[int] = field(default_factory=lambda: [2, 3, 4])
    width_sweep: List[float] = field(default_factory=lambda: [0.08, 0.15, 0.22, 0.3, 0.45])
    width_sweep_k_cap: int = 512
    gamma_sweep_dims: List[int] = field(default_factory=lambda: [2, 4, 8])
    gamma_sweep: List[float] = field(default_factory=lambda: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    gamma_sweep_K: int = 8


@dataclass
class ExperimentLearnedMemoryConfig:
    lam: float = 1.0
    f: float = 1.0
    barrier: float = 0.2
    payload_kappa: float = 1.0
    bump_width: float = 0.05
    payload_seed: int = 0
    rungs: List[str] = field(
        default_factory=lambda: [
            "designed",
            "skeleton_residual",
            "sites_learned_payload",
            "local_rbf",
            "free_mlp",
        ]
    )
    hidden: int = 64
    n_atoms: int = 24
    residual_scale: float = 0.1
    rbf_init_width: float = 0.3
    write_steps: int = 600
    write_lr: float = 0.003
    write_weight_decay: float = 0.0001
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.0
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    n_query_per_item: int = 32
    query_sigma_theta: float = 0.15
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    item_counts: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    survival_fracs: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.99])
    reference_rung: str = "designed"
    pass_strict: float = 0.9
    pass_read: float = 0.9
    blank_margin: float = 0.15
    blank_strict_max: float = 0.1
    interference_K: int = 4
    interference_write_steps: int = 300
    gamma_map_K: int = 4
    gamma_map_rungs: List[str] = field(
        default_factory=lambda: ["designed", "sites_learned_payload"]
    )
    gamma_address_grid: List[float] = field(default_factory=lambda: [0.0, 0.005, 0.02, 0.05, 0.1])
    gamma_read_grid: List[float] = field(default_factory=lambda: [0.0, 0.005, 0.02, 0.05, 0.1])
    sym_baseline: Dict[str, float] = field(
        default_factory=lambda: {
            "payload_abs_err": 0.000998,
            "codebook_read_K2": 1.0,
            "codebook_read_K8": 0.992,
            "blank_control_K2": 0.469,
            "designed_corruption": 4.17e-07,
        }
    )


@dataclass
class ExperimentPotentialClassConfig:
    lam: float = 1.0
    f: float = 1.0
    barrier: float = 0.2
    payload_kappa: float = 1.0
    bump_width: float = 0.05
    payload_seed: int = 0
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.0
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    n_query_per_item: int = 32
    query_sigma_theta: float = 0.15
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    survival_fracs: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.99])
    blank_margin: float = 0.15
    blank_strict_max: float = 0.1
    pass_strict: float = 0.9
    pass_read: float = 0.9
    reference_rung: str = "designed"
    write_steps: int = 600
    write_lr: float = 0.003
    write_weight_decay: float = 0.0001
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    local_write_steps: int = 600
    potential_classes: List[str] = field(
        default_factory=lambda: [
            "designed",
            "mlp",
            "hopfield",
            "hopfield_sharp",
            "attn",
            "atoms",
            "atoms_local",
        ]
    )
    reference_class: str = "designed"
    baseline_class: str = "mlp"
    param_target: int = 4481
    param_tol: float = 0.05
    hidden: int = 64
    n_atoms: int = 896
    hopfield_n_mem: int = 1120
    attn_n_mem: int = 495
    attn_d_head: int = 8
    hopfield_beta_soft: float = 2.0
    hopfield_beta_sharp: float = 8.0
    hopfield_confine: float = 0.5
    hopfield_key_init: float = 0.1
    attn_beta: float = 1.0
    attn_key_init: float = 0.5
    learned_confine: float = 0.05
    atom_init_width: float = 0.3
    atom_init_scale: float = 1.0
    atom_depth_init: float = 0.0001
    class_item_counts: List[int] = field(default_factory=lambda: [4, 8])
    class_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    interference_K: int = 4
    interference_write_steps: int = 300
    interference_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    support_radii: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    )
    support_probes_per_radius: int = 96
    support_decay_threshold: float = 0.1
    support_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    run_ladder_rerun: bool = True
    ladder_families: List[str] = field(default_factory=lambda: ["atoms", "hopfield"])
    ladder_rungs: List[str] = field(
        default_factory=lambda: ["skeleton_residual", "sites_learned_payload", "free_mlp"]
    )
    ladder_item_counts: List[int] = field(default_factory=lambda: [4, 8])
    ladder_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    matched_baseline: Dict[str, float] = field(
        default_factory=lambda: {
            "designed_strict_K4": 1.0,
            "designed_strict_K8": 0.986,
            "free_mlp_strict_K4": 0.853,
            "free_mlp_strict_K8": 0.599,
            "skeleton_residual_strict_K4": 0.903,
            "skeleton_residual_strict_K8": 0.959,
            "local_rbf_strict_K4": 0.623,
            "local_rbf_strict_K8": 0.348,
            "designed_corruption": 0.0,
            "free_mlp_corruption": 0.353,
            "sites_learned_payload_corruption": 0.495,
        }
    )


@dataclass
class ExperimentDesignedMechanismConfig:
    R: float = 1.0
    wall_margin: float = 0.5
    well_width: float = 0.15
    well_depth: float = 1.0
    payload_kappa: float = 0.1
    c_conf: float = 10.0
    site_seed: int = 0
    payload_seed: int = 0
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.02
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    rollout_chunk: int = 256
    n_query_per_item: int = 32
    max_total_queries: int = 4096
    min_query_per_item: int = 4
    query_sigma: float = 0.15
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    pass_strict: float = 0.9
    blank_strict_max: float = 0.1
    blank_margin: float = 0.15
    atoms_per_item: int = 32
    min_atoms: int = 384
    min_atoms_base: int = 512
    min_atoms_c: float = 1.4142135623730951
    atom_init_scale: float = 1.0
    atom_init_width: float = 0.3
    atom_depth_init: float = 0.0001
    learned_confine: float = 0.05
    bits_per_param: int = 32
    atom_init_local: bool = False
    atom_init_local_mult: float = 2.0
    n_payload_channels: int = 1
    payload_code: str = "linspace"
    payload_launch_sigma: float = 0.0
    payload_obs_sigma: float = 0.0
    pass_metric: str = "decode"
    read_anneal_stages: int = 1
    read_anneal_s0: float = 0.0
    read_anneal_power: float = 1.0
    read_anneal_mode: str = "amplitude"
    read_anneal_phases: str = "both"
    read_anneal_axes: str = "all"
    read_anneal_payload_mult: float = 1.0
    write_steps: int = 600
    local_write_steps: int = 300
    write_lr: float = 0.003
    write_weight_decay: float = 0.0001
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    dims: List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    k_ladder: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128])
    k_cap: int = 128
    learned_arm: str = "learned_global"
    discriminator_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    designed_seeds: List[int] = field(default_factory=lambda: [0])
    mass_dim: int = 3
    mass_K: int = 4
    mass_spread: float = 4.0
    mass_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    mass_help_threshold: float = 0.02
    interference_dims: List[int] = field(default_factory=lambda: [2, 3, 4])
    interference_K: int = 4
    interference_write_steps: int = 300
    interference_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class ExperimentPrimitiveHarnessConfig:
    primitives: List[str] = field(default_factory=lambda: ["mlp", "gru", "ssm", "attention", "clu"])
    d_model: int = 64
    n_layers: int = 2
    target_block_params: int = 40000
    param_tol: float = 0.05
    width_search_lo: int = 4
    width_search_hi: int = 512
    lr_grid: List[float] = field(default_factory=lambda: [0.0003, 0.001, 0.003])
    train_steps: int = 2000
    tune_steps: int = 600
    batch_size: int = 32
    eval_batch: int = 256
    grad_clip: float = 1.0
    n_seeds: int = 3
    mqar_vocab: int = 256
    mqar_seq_lens: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    mqar_kv_fixed: int = 4
    mqar_kv_sweep: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    mqar_seq_len_fixed: int = 128
    adding_seq_len: int = 128
    parity_seq_len: int = 64
    clu_dt: float = 0.1
    clu_gamma: float = 0.05
    clu_steps: int = 1
    clu_hidden: int = 32
    clu_kinetic_mode: str = "newtonian_learned"
    clu_potential_type: str = "mlp"
    clu_read_mode: str = "endpoint"
    clu_write_mode: str = "linear"
    ssm_selective: bool = True
    attn_heads: int = 4
    clu_gamma_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    )
    clu_read_mode_sweep: List[str] = field(default_factory=lambda: ["endpoint", "trajectory"])
    clu_steps_sweep: List[int] = field(default_factory=lambda: [1, 2, 4])
    gw_train_steps: int = 1200
    gw_tune_steps: int = 400
    gw_mqar_gamma: float = 0.0
    gw_extrap_train_T: int = 64
    gw_extrap_mults: List[int] = field(default_factory=lambda: [1, 2, 4])
    gw_extrap_families: List[str] = field(default_factory=lambda: ["adding", "parity"])
    gw_noise_grid: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.4])


@dataclass
class ExperimentSequentialWriteConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    lam: float = 1.0
    f: float = 1.0
    barrier: float = 0.2
    payload_kappa: float = 1.0
    bump_width: float = 0.05
    payload_seed: int = 0
    hidden: int = 64
    n_atoms: int = 24
    residual_scale: float = 0.1
    rbf_init_width: float = 0.3
    write_steps: int = 600
    write_lr: float = 0.003
    write_weight_decay: float = 0.0001
    write_n_perturb: int = 32
    write_sigma_addr: float = 0.25
    write_sigma_pay: float = 0.6
    write_margin: float = 0.15
    write_barrier: float = 0.2
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.0
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    n_query_per_item: int = 32
    n_query_sequential: int = 16
    query_sigma_theta: float = 0.15
    query_sigma_p: float = 0.05
    payload_tol: float = 0.1
    payload_tol_frac: float = 0.35
    d_safe_mult: float = 4.4
    delta_budget: float = 0.1
    c3_chunk_steps: int = 25
    n_relocation_candidates: int = 400
    proposal_radius: float = 2.0
    interference_K: int = 4
    interference_write_steps: int = 300
    gate_rungs: List[str] = field(
        default_factory=lambda: ["designed", "sites_learned_payload", "free_mlp"]
    )
    gate_arms: List[str] = field(
        default_factory=lambda: ["ungated", "gated_spacing", "gated_c3", "anchored"]
    )
    gate_proposals: List[str] = field(default_factory=lambda: ["ring", "disk"])
    n_sequential_items: int = 16
    sequential_write_steps: int = 600
    sequential_rung: str = "free_mlp"
    sequential_arms: List[str] = field(
        default_factory=lambda: [
            "designed_gated",
            "designed_ungated",
            "learned_gated",
            "learned_ungated",
            "learned_anchored",
        ]
    )
    atom_width: float = 0.35
    atom_alpha: float = 0.02
    atom_amp: float = 1.0
    kv_primitives: List[str] = field(default_factory=lambda: ["mlp", "gru", "attention", "clu"])
    kv_n_items: int = 16
    kv_select_items: int = 8
    kv_vocab: int = 128
    kv_key_len: int = 4
    kv_max_write_steps: int = 200
    kv_check_every: int = 1
    kv_extended_items: int = 64
    kv_extended_rescue: bool = True
    cost_K_grid: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])


@dataclass
class ExperimentControllerMvpConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    atom_width: float = 0.35
    atom_alpha: float = 0.02
    atom_amp: float = 1.0
    payload_kappa: float = 1.0
    payload_seed: int = 0
    d_safe_mult: float = 4.4
    n_relocation_candidates: int = 400
    proposal_radius: float = 2.0
    K_grid: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    budget: Optional[int] = None
    evict_policy: str = "staleness"
    leak: float = 0.0
    amp_floor: float = 0.05
    run_sized_geometry: bool = True
    run_canonical_placement: bool = False
    canonical_radius_mult: float = 1.0
    canonical_waitlist: bool = True
    payload_gate: bool = False
    payload_gate_g0: float = 0.05
    payload_gate_eps: float = 1e-06
    decay_demo_K: int = 8
    decay_demo_leak: float = 0.35
    dt: float = 0.05
    gamma_address: float = 0.05
    gamma_read: float = 0.0
    address_steps: int = 400
    read_steps: int = 800
    tail_frac: float = 0.25
    n_subsample: int = 8
    n_query_per_item: int = 16
    query_sigma_theta: float = 0.15
    query_sigma_p: float = 0.05
    f: float = 1.0
    payload_tol: float = 0.1
    payload_tol_frac: float = 0.35
    run_primitives: bool = True
    kv_primitives: List[str] = field(default_factory=lambda: ["mlp", "gru", "attention", "clu"])
    kv_baseline_K: int = 64
    kv_vocab: int = 128
    kv_key_len: int = 4
    kv_max_write_steps: int = 200
    kv_check_every: int = 1
    kv_select_items: int = 8


@dataclass
class ExperimentWriteCeilingConfig:
    arms: List[str] = field(
        default_factory=lambda: [
            "baseline_global",
            "sequential_masked",
            "sequential_free",
            "scale_invariant",
            "crowding_aware",
            "combo",
        ]
    )
    dims: List[int] = field(default_factory=lambda: [4, 5, 6, 8])
    k_ladder: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    k_start: List[int] = field(default_factory=lambda: [16, 32, 32, 32])
    k_cap: int = 128
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    screen_seeds: int = 1
    blank_seeds: int = 1
    n_query_per_item: int = 16
    seq_steps_per_item: int = 300
    seq_passes: int = 1
    scale_sigma_frac: float = 0.3
    scale_width_frac: float = 0.35
    scale_item_agg: str = "sum"
    crowd_weight: float = 1.0
    crowd_d_safe_frac: float = 0.5
    crowd_d_safe_mult: float = 4.4
    crowd_min_agg: str = "max"
    crowd_barrier_pairs: str = "nn"
    run_adequacy_recheck: bool = True
    adequacy_atom_mult: float = 2.0
    adequacy_seeds: int = 2
    run_steps_recheck: bool = True
    steps_recheck_mult: float = 2.0


@dataclass
class ExperimentShardedStoreConfig:
    cells: List[str] = field(default_factory=lambda: ["6:64:2", "4:32:2", "8:64:2", "8:64:4"])
    arms: List[str] = field(
        default_factory=lambda: [
            "monolithic",
            "sharded_matched",
            "sharded_per_shard",
            "monolithic_nx",
        ]
    )
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    n_query_per_item: int = 16
    blank_seeds: int = 1
    allocation: str = "global"
    partition: str = "spread"
    router: str = "router_2"
    routers: List[str] = field(default_factory=lambda: ["router_2", "router_3", "RG"])
    abstain_deadband: float = 0.0
    deadband_sweep: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.2, 1.0])
    atom_init_local: bool = False
    atom_init_local_mult: float = 2.0
    init_ablation_cells: List[str] = field(default_factory=lambda: ["6:64:2"])
    init_ablation_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    parity_d: int = 4
    parity_K: int = 32
    parity_shards: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    parity_allocations: List[str] = field(default_factory=lambda: ["global", "local"])
    parity_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    parity_n_query_per_item: int = 16
    timing_d: int = 6
    timing_K: int = 64
    timing_shards: List[int] = field(default_factory=lambda: [2, 4, 8])
    timing_queries: int = 64
    timing_repeats: int = 3
    alloc_demo_K: int = 24
    alloc_demo_shards: int = 4
    alloc_demo_radius: float = 2.0
    alloc_demo_s: float = 0.35
    alloc_demo_seed: int = 0
    run_discriminator: bool = True
    run_read_parity: bool = True
    run_timing: bool = True
    run_allocator: bool = True
    run_init_ablation: bool = True
    run_deadband_sweep: bool = True


@dataclass
class ExperimentHopfieldCapacityConfig:
    datasets: List[str] = field(default_factory=lambda: ["mnist"])
    load_grid: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    noise_fixed_load: int = 128
    n_data_pool: int = 2000
    mask_p: float = 0.5
    hopfield_beta: float = 1.0
    hopfield_steps: int = 1
    hopfield_beta_tuned: float = 0.0
    hopfield_steps_tuned: int = 2
    activations: List[str] = field(default_factory=lambda: ["softmax", "sparsemax"])
    clu_s_frac: float = 0.3
    clu_b: float = 1.0
    clu_alpha: float = 0.001
    clu_gamma: float = 0.1
    clu_steps: int = 200
    clu_dt: float = 0.0
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"
    retry_enabled: bool = True
    retry_boost: float = 1.5
    retry_conf_frac: float = 0.5
    success_cosine: float = 0.9
    seed: int = 0
    rollout_chunk: int = 256


@dataclass
class ExperimentPhiReadInConfig:
    datasets: List[str] = field(default_factory=lambda: ["mnist"])
    phi_arms: List[str] = field(default_factory=lambda: ["pca", "ae"])
    load_grid: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    noise_fixed_load: int = 128
    n_data_pool: int = 1500
    n_fit_pool: int = 3000
    mask_p: float = 0.5
    phi_dim: int = 32
    ae_hidden: int = 256
    ae_epochs: int = 400
    ae_lr: float = 0.001
    ae_batch: int = 512
    hopfield_beta: float = 1.0
    hopfield_steps: int = 1
    activations: List[str] = field(default_factory=lambda: ["softmax", "sparsemax"])
    clu_s_frac: float = 0.3
    clu_b: float = 1.0
    clu_alpha: float = 0.001
    clu_gamma: float = 0.1
    clu_steps: int = 200
    clu_dt: float = 0.0
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"
    probe_retry_confidence: bool = True
    success_cosine: float = 0.9
    seed: int = 0
    rollout_chunk: int = 256


@dataclass
class ExperimentRetryComputeConfig:
    datasets: List[str] = field(default_factory=lambda: ["mnist"])
    load_grid: List[int] = field(default_factory=lambda: [128, 256])
    noise_levels: List[float] = field(default_factory=lambda: [0.2, 0.3])
    query_types: List[str] = field(default_factory=lambda: ["mask", "noise"])
    mask_fracs: List[float] = field(default_factory=lambda: [0.5, 0.7])
    n_data_pool: int = 2000
    retry_ladder: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    retry_step_frac: float = 0.1
    conf_thresholds: List[float] = field(default_factory=lambda: [0.95, 0.97, 0.99, 1.0])
    main_threshold: float = 0.99
    retry_boost: float = 1.5
    clu_s_frac: float = 0.3
    clu_b: float = 1.0
    clu_alpha: float = 0.001
    clu_gamma: float = 0.1
    clu_steps: int = 150
    clu_dt: float = 0.0
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"
    ff_aug_sigma: float = 0.1
    hopfield_beta: float = 1.0
    hopfield_beta_tuned: float = 0.0
    regimes: List[str] = field(default_factory=list)
    block_fracs: List[float] = field(default_factory=lambda: [0.5, 0.7])
    block_rescale: bool = True
    crowd_mask_fracs: List[float] = field(default_factory=lambda: [0.3, 0.5])
    crowd_anchor: int = 0
    crowd_rho: float = 1.0
    headroom_band: List[float] = field(default_factory=lambda: [0.5, 0.75])
    headroom_nn_ceiling: float = 0.95
    n_seeds: int = 1
    sweep_seeds: int = 1
    seed: int = 0
    rollout_chunk: int = 256


@dataclass
class ExperimentPhiStreamConfig:
    dataset: str = "mnist"
    n_tasks: int = 5
    classes_per_task: int = 2
    phi_regimes: List[str] = field(default_factory=lambda: ["task1_only", "generic_frozen"])
    phi_arms: List[str] = field(default_factory=lambda: ["pca", "ae"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    items_per_task: int = 32
    n_fit_pool: int = 3000
    n_store_region: int = 20000
    mask_p: float = 0.5
    phi_dim: int = 32
    ae_hidden: int = 256
    ae_epochs: int = 400
    ae_lr: float = 0.001
    ae_batch: int = 512
    enc_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    enc_pool: int = 2
    enc_groups: int = 8
    enc_steps: int = 8000
    enc_batch: int = 128
    enc_lr: float = 0.001
    enc_temperature: float = 0.5
    enc_proj_dim: int = 64
    enc_head: str = "pca"
    enc_l2_normalize: bool = True
    enc_aug_crop_pad: int = 4
    enc_aug_zoom_p: float = 0.5
    enc_aug_zoom_size: int = 20
    enc_aug_flip_p: float = 0.5
    enc_aug_color_p: float = 0.8
    enc_aug_color_strength: float = 0.4
    enc_aug_gray_p: float = 0.2
    clu_s_frac: float = 0.3
    clu_b: float = 1.0
    clu_alpha: float = 0.001
    clu_gamma: float = 0.1
    clu_steps: int = 200
    clu_dt: float = 0.0
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"
    s_policy: str = "refit"
    laundering_tie_band: float = 0.03
    success_cosine: float = 0.9
    seed: int = 0
    rollout_chunk: int = 256


@dataclass
class ExperimentClEntryConfig:
    dataset: str = "mnist"
    n_tasks: int = 5
    classes_per_task: int = 2
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    items: List[str] = field(default_factory=lambda: ["entry", "retry", "retention"])
    n_train_per_task: int = 2000
    n_test_per_task: int = 1000
    n_fit_region: int = 10000
    n_fit_pool: int = 3000
    phi_regimes: List[str] = field(default_factory=lambda: ["task1_only", "generic_frozen"])
    phi_arm: str = "pca"
    phi_dim: int = 32
    ae_hidden: int = 256
    ae_epochs: int = 400
    ae_lr: float = 0.001
    ae_batch: int = 512
    enc_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    enc_pool: int = 2
    enc_groups: int = 8
    enc_steps: int = 8000
    enc_batch: int = 128
    enc_lr: float = 0.001
    enc_temperature: float = 0.5
    enc_proj_dim: int = 64
    enc_head: str = "pca"
    enc_l2_normalize: bool = True
    enc_aug_crop_pad: int = 4
    enc_aug_zoom_p: float = 0.5
    enc_aug_zoom_size: int = 20
    enc_aug_flip_p: float = 0.5
    enc_aug_color_p: float = 0.8
    enc_aug_color_strength: float = 0.4
    enc_aug_gray_p: float = 0.2
    memory_items: int = 200
    clu_s_frac: float = 0.2
    d_safe_mult: float = 4.4
    s_policy: str = "refit"
    s_init_items: int = 200
    store_alpha: float = 0.001
    clu_b: float = 1.0
    clu_alpha: float = 0.001
    clu_gamma: float = 0.1
    clu_steps: int = 150
    clu_dt: float = 0.0
    clu_tail_frac: float = 0.1
    clu_kinetic_mode: str = "newtonian_identity"
    rollout_chunk: int = 256
    baselines: List[str] = field(
        default_factory=lambda: [
            "finetune",
            "ewc",
            "si",
            "lwf",
            "er",
            "derpp",
            "icarl",
            "gdumb",
            "joint",
        ]
    )
    backbone: str = "mlp"
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32, 32])
    mlp_width: int = 400
    mlp_depth: int = 2
    baseline_iters: int = 500
    baseline_batch: int = 128
    baseline_lr: float = 0.001
    ewc_lambda: float = 5000.0
    si_c: float = 1.0
    si_xi: float = 0.1
    lwf_alpha: float = 1.0
    lwf_temp: float = 2.0
    lwf_ce_scope: str = "seen"
    fisher_samples: int = 200
    eval_chunk: int = 1024
    tune_baselines: bool = True
    ewc_lambda_grid: List[float] = field(default_factory=lambda: [100.0, 1000.0, 10000.0])
    si_c_grid: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    lwf_alpha_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    derpp_alpha: float = 0.5
    derpp_beta: float = 0.5
    tune_derpp: bool = True
    derpp_alpha_grid: List[float] = field(default_factory=lambda: [0.2, 0.5, 1.0])
    derpp_beta_grid: List[float] = field(default_factory=lambda: [0.5, 1.0])
    frontier_budgets_floats: List[int] = field(
        default_factory=lambda: [6280, 19625, 39250, 78500, 157000, 314000]
    )
    frontier_methods: List[str] = field(default_factory=lambda: ["er", "derpp", "icarl", "gdumb"])
    frontier_fixed_methods: List[str] = field(
        default_factory=lambda: ["finetune", "ewc", "si", "lwf", "joint"]
    )
    frontier_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    frontier_n_test_per_task: int = 0
    frontier_max_clu_items: int = 0
    frontier_tuning_budget: int = 0
    frontier_la_band: float = 0.1
    count_controller_record_floats: bool = True
    bytes_per_float: int = 4
    retry_ladder: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    retry_tau: float = 0.99
    retry_tau_grid: List[float] = field(default_factory=lambda: [0.99, 0.999, 0.9999, 1.0])
    retry_boost: float = 1.5
    retry_step_frac: float = 0.1
    retry_mask_p: float = 0.5
    retry_mask_levels: List[float] = field(default_factory=lambda: [0.5, 0.8])
    retry_mid_task: int = 2
    ff_aug_sigma: float = 0.1
    ticks_per_task: int = 4
    leak_slow: float = 0.0866
    leak_fast: float = 0.3466
    amp_floor: float = 0.05
    permanent_per_task: int = 10
    laundering_tie_band: float = 0.03
    count_phi_param_floats: bool = True
    seed: int = 0


@dataclass
class ExperimentPersistentStoreConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    persistent_store: bool = False
    lifecycle: bool = False
    promote: bool = True
    demote: bool = True
    trash: bool = True
    h_hi: int = 2
    h_lo: int = 1
    window: int = 2
    d_dwell: int = 3
    d_demote: int = 2
    k_streams: int = 3
    trash_criterion: str = "last_k_streams"
    censoring_guard: bool = True
    f_max: float = 0.25
    refresh_monotonic: bool = False
    refresh_max_gain: float = 4.0
    n_regimes: int = 3
    n_classes: int = 4
    n_features: int = 8
    n_per_stream: int = 64
    schedule: List[int] = field(default_factory=lambda: [0, 1, 2, 0, 1, 2])
    drift_free: bool = False
    n_anchors: int = 96
    jitter: float = 0.02
    benchmark_gate_path: str = "runs/benchmark_gate.json"
    decimation_m: int = 1
    wall_target_s: float = 7200.0
    addr_dim: int = 12
    payload_dim: int = 1
    well_budget: int = 64
    capacity: int = 72
    leak: float = 0.02
    payload_scale: float = 8.0
    d_safe_frac: float = 0.88
    atom_width_frac_spacing: float = 1.5
    atom_kernel: str = "wendland"
    atom_kernel_cutoff: float = 2.5
    atom_site_local_init: bool = True
    chunk_size: int = 8
    offers_per_chunk: int = 3
    read_batch: int = 128
    write_steps: int = 40
    read_steps: int = 200
    address_steps: int = 100
    depth_every: int = 2
    quick: bool = False


@dataclass
class ExperimentWellLifecycleConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    dataset: str = "mnist"
    phi_arm: str = "pca"
    phi_regime: str = "task1_only"
    addr_dim: int = 8
    payload_dim: int = 1
    well_budget: int = 8
    capacity: int = 16
    n_offer_per_task: int = 8
    overdig_target: float = 2.0
    payload_scale: float = 9.0
    leak: float = 0.02
    permanent_per_task: int = 1
    ticks_per_task: int = 1
    read_batch: int = 16
    read_every: int = 4
    capture_dirs: int = 16
    capture_bisect_steps: int = 8
    measure_capture: bool = True
    loo_repeats: int = 2
    run_loo: bool = True
    run_gate_addr: bool = True
    gaddr_kappa_q: float = 1.0
    gaddr_n_query_per_item: int = 8
    addr_scale_mult: float = 1.0
    gdrift_floor_frac_spacing: float = 0.01
    gdrift_ceil_frac_spacing: float = 1.0
    d_safe_population: str = "store"
    d_safe_sizing_n: int = 200
    d_safe_population_draws: int = 64
    gaddr_spacing_population: str = "codebook"
    atom_width_selection: Optional[float] = None
    refuse_unselected_atom_width: bool = True
    atom_width_selection_rtol: float = 1e-06
    d_safe_frac: float = 0.88
    write_steps: int = 300
    read_steps: int = 800
    address_steps: int = 400
    n_query_per_item: int = 8
    quick: bool = False


@dataclass
class ExperimentPhiGeometryConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    dataset: str = "cifar10"
    addr_dims: List[int] = field(default_factory=lambda: [8, 12, 16])
    phi_arm_strong: str = "simclr"
    phi_arm_control: str = "randconv"
    phi_arm_reference: str = "pca"
    phi_dim_strong: int = 256
    phi_regime: str = "task1_only"
    enc_steps: int = 8000
    n_fit_region: int = 25000
    n_fit_pool: int = 6000
    projection: str = "pca"
    projection_controls: List[str] = field(default_factory=lambda: ["truncate", "gaussian"])
    n_keys_primary: int = 16
    n_keys_grid: List[int] = field(default_factory=lambda: [16, 64, 200])
    n_offer_per_task: int = 8
    query_sigma: float = 0.15
    d_safe_frac: float = 0.88
    d_safe_sizing_n: int = 200
    go_min_seeds_positive: int = 3
    go_se_multiple: float = 2.0
    depth_probe: bool = True
    depth_probe_dims: List[int] = field(default_factory=lambda: [8, 12, 16])
    depth_probe_writes: int = 6
    depth_probe_seed: int = 0
    depth_inert_tol: float = 1e-06
    capacity: int = 16
    well_budget: int = 8
    write_steps: int = 300
    read_steps: int = 800
    address_steps: int = 400
    n_query_per_item: int = 8
    quick: bool = False


@dataclass
class ExperimentCaptureStrongPhiConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    dataset: str = "cifar10"
    addr_dim: int = 12
    arms: List[str] = field(default_factory=lambda: ["randconv", "simclr", "pca"])
    phi_arm_strong: str = "simclr"
    phi_arm_control: str = "randconv"
    phi_arm_reference: str = "pca"
    phi_dim_strong: int = 256
    phi_regime: str = "task1_only"
    enc_steps: int = 8000
    n_fit_region: int = 25000
    n_fit_pool: int = 6000
    projection: str = "pca"
    atom_width_frac_spacing: Optional[float] = 1.5
    atom_kernel: str = "wendland"
    atom_kernel_cutoff: float = 2.5
    site_local_init: bool = True
    site_local_radius_frac: float = 1.0
    d2a_probe: bool = True
    quick: bool = False


@dataclass
class ExperimentCaptureArmAConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    pilot_seeds: List[int] = field(default_factory=lambda: [7, 8])
    atom_kernel: str = "wendland"
    atom_kernel_cutoff: float = 2.5
    atom_width_frac_spacing: Optional[float] = 0.5
    site_local_init: bool = True
    site_local_radius_frac: float = 1.0
    run_baseline_gaussian: bool = False


@dataclass
class ExperimentCaptureArmBConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    head_hidden: int = 64
    head_layers: int = 2
    width_min: float = 0.3
    width_max: float = 0.8
    depth_min: float = 1.5
    depth_max: float = 3.0
    payload_delta_max: float = 0.05
    center_skip_gain: float = 1.0
    pretrain_steps: int = 600
    pretrain_batch: int = 16
    pretrain_lr: float = 0.003
    pretrain_weight_decay: float = 0.0001
    pretrain_pool: int = 256
    reach_weight: float = 1.0
    reach_rho: float = 2.0
    attr_weight: float = 10.0
    attr_margin: float = 0.15
    crowd_weight: float = 1.0
    run_min_store_cell: bool = True
    min_store_atoms_per_item: int = 1
    quick: bool = False


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
class GenDataConfig:
    data_root: Optional[str] = None
    condition: str = "recomb2element"
    variant: str = "leave1out"
    verify_checksum: bool = True
    holdout_assert: bool = True
    assert_expected_counts: bool = True
    quick_rows: int = 4096
    dsprites_with_images: bool = True
    c_invalid_n_per_kind: Optional[int] = None
    c_invalid_seed: int = 0
    emit_a7: bool = True
    a7_n_requests: Optional[int] = None
    a7_seed: int = 0
    wide_n_cells: int = 12
    wide_seed: int = 0
    wide_include_montero: bool = True


@dataclass
class GenBridgeConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    enc_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    enc_kernel: int = 4
    enc_stride: int = 2
    enc_groups: int = 8
    enc_steps: int = 1500
    enc_batch: int = 128
    enc_lr: float = 0.001
    dec_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    dec_groups: int = 8
    dec_steps: int = 1200
    dec_batch: int = 128
    dec_lr: float = 0.001
    dec_recipe: str = "baseline"
    dec_width_mult: float = 1.0
    dec_blocks_per_stage: int = 1
    dec_loss: str = "mse"
    dec_edge_weight: float = 8.0
    dec_edge_blur: int = 5
    dec_grad_weight: float = 1.0
    dec_lr_schedule: str = "constant"
    dec_warmup_frac: float = 0.05
    probe_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    probe_groups: int = 8
    probe_steps: int = 1500
    probe_batch: int = 128
    probe_lr: float = 0.001
    n_train: int = 8192
    n_eval_a: int = 1024
    eval_batch: int = 256
    cache_scope: str = "full"
    cache_dtype: str = "float16"
    cache_block: int = 8192
    ceiling_floor: float = 0.8
    disc_floor: float = 0.8
    disc_abs_floor: float = 0.75
    disc_min_negatives: int = 20
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    make_plots: bool = True
    verify_cache_checksum: bool = True
    quick_enc_steps: int = 30
    quick_dec_steps: int = 30
    quick_probe_steps: int = 30
    quick_n_train: int = 256
    quick_n_eval_a: int = 64
    quick_cache_scope: str = "used"
    quick_seeds: List[int] = field(default_factory=lambda: [0])


@dataclass
class GenStoreConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    bridge_root: Optional[str] = None
    k_wells: Optional[int] = None
    k_min: int = 4
    k_max: int = 16
    k_choice_seed: int = 0
    grouping_method: str = "auto"
    min_well_frac: float = 0.1
    n_corr: int = 16384
    groupings: List[str] = field(default_factory=lambda: ["correlation", "random"])
    n_modes: int = 64
    n_store_train: int = 32768
    train_steps: int = 1200
    train_batch: int = 128
    train_lr: float = 0.003
    commitment_beta: float = 0.25
    tau_start: float = 1.0
    tau_end: float = 0.1
    n_probe_train: int = 8192
    probe_steps: int = 400
    probe_batch: int = 512
    probe_lr: float = 0.003
    eval_batch: int = 256
    encode_chunk: int = 16384
    delete_modes: List[str] = field(default_factory=lambda: ["zero", "mean", "scramble"])
    c_top_share: float = 0.5
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    make_plots: bool = True
    verify_cache_checksum: bool = True
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_n_corr: int = 1024
    quick_n_store_train: int = 2048
    quick_train_steps: int = 30
    quick_n_probe_train: int = 1024
    quick_probe_steps: int = 30
    quick_n_modes: int = 8
    quick_k_max: int = 6


@dataclass
class GenR0WideConfig:
    condition: str = "recomb2element_wide"
    bridge_runs_root: Optional[str] = None
    montero_store_runs: Optional[str] = None
    store_plan: Optional[str] = None
    dec_recipe: str = "v1_capacity_schedule"
    assert_dec_recipe: bool = True
    gate_tolerance_sigmas: float = 3.0
    underpowered_spread_multiple: float = 3.0
    exclude_shared_corner_from_pooled: bool = True
    shared_corner_cells: List[str] = field(default_factory=lambda: ["montero", "wide-000"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    verify_cache_checksum: bool = True
    make_plots: bool = True


@dataclass
class GenMetricInvarianceConfig:
    relabel_tol: float = 1e-12
    class_count_tol_ratio: float = 0.5
    class_count_min_sign_agreement: int = 5
    configuration_band: float = 3.0
    class_budget: Dict[str, int] = field(default_factory=lambda: {"orientation": 6, "scale": 6})
    partitions: List[str] = field(
        default_factory=lambda: [
            "correlation",
            "random",
            "random_alt",
            "well_permuted",
            "contiguous",
            "balanced_random",
        ]
    )
    random_alt_offset: int = 1000
    negative_control_axis: str = "leak_raw"
    candidate_axis: str = "dci_c"
    companion_axis: str = "leak_cc"
    configurations: List[str] = field(
        default_factory=lambda: ["recomb2element", "recomb2element_wide"]
    )
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class GenDreamConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    bridge_root: Optional[str] = None
    cell_root: Optional[str] = None
    store_plan: Optional[str] = None
    grouping_source: str = "per_seed"
    decoder_source: str = "winner"
    winner_root: Optional[str] = None
    winner_decoder_pattern: str = "decoder_v1_s{seed}.pkl"
    expected_decoder_recipe: str = "v1_capacity_schedule"
    coupling_strength: float = 1.0
    coupling_smoothing: float = 1.0
    address_smoothing: float = 1.0
    clamp_threshold: float = 0.5
    init_mode: str = "centroid"
    sampler: str = "mala"
    samplers: List[str] = field(default_factory=lambda: ["mala", "ula", "uld"])
    steps: int = 200
    step_size: Optional[float] = None
    step_size_init: float = 0.1
    target_accept: float = 0.7
    accept_tolerance: float = 0.1
    tune_rounds: int = 8
    tune_steps: int = 20
    tune_factor: float = 2.0
    n_tune_requests: int = 128
    tau_calibration: str = "mode_distance"
    tau_start: float = 1.0
    tau_end: float = 0.25
    temp_calibration: str = "energy_scale"
    temp_start: float = 0.05
    temp_end: float = 0.005
    uld_dt: float = 0.05
    uld_gamma: float = 0.1
    arms: List[str] = field(default_factory=lambda: ["clu", "store_removed", "no_coupling"])
    variants: List[str] = field(default_factory=lambda: ["full", "zero_step", "prior_mode"])
    request_protocols: List[str] = field(default_factory=lambda: ["full", "partial"])
    request_drop_factors: int = 2
    n_dream: int = 512
    eval_batch: int = 256
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    save_traces: bool = True
    make_plots: bool = True
    use_store_checkpoint: bool = True
    verify_cache_checksum: bool = True
    store_removed_delete_mode: str = "mean"
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_n_dream: int = 64
    quick_steps: int = 10
    quick_tune_rounds: int = 2
    quick_n_store_train: int = 2048
    quick_store_train_steps: int = 30
    quick_arms: List[str] = field(default_factory=lambda: ["clu", "no_coupling"])
    quick_samplers: List[str] = field(default_factory=lambda: ["mala"])
    quick_request_protocols: List[str] = field(default_factory=lambda: ["partial"])


@dataclass
class GenCertificateConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    bridge_root: Optional[str] = None
    store_runs: Optional[str] = None
    c_invalid_path: Optional[str] = None
    groupings: List[str] = field(default_factory=lambda: ["correlation", "random"])
    laplace_alpha: float = 1.0
    reach_radius_frac: float = 0.5
    tau_curvature: Optional[float] = None
    accept_quantile: float = 0.95
    n_requests_a: Optional[int] = None
    term_count_control: bool = True
    save_store: bool = True
    assert_store_bit_identity: bool = True
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    make_plots: bool = True
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_n_store_train: int = 2048
    quick_n_corr: int = 1024
    quick_train_steps: int = 30
    quick_n_modes: int = 8
    quick_n_requests_a: int = 64


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str = "default"
    seed: int = 42
    device: str = "auto"
    save_dir: Optional[str] = None


@dataclass
class NCLUConfig:
    """Master configuration containing all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_a: ExperimentAConfig = field(default_factory=ExperimentAConfig)
    experiment_b: ExperimentBConfig = field(default_factory=ExperimentBConfig)
    experiment_c: ExperimentCConfig = field(default_factory=ExperimentCConfig)
    experiment_d: ExperimentDConfig = field(default_factory=ExperimentDConfig)
    experiment_v1_gate: ExperimentV1GateConfig = field(default_factory=ExperimentV1GateConfig)
    experiment_v1_wormhole: ExperimentV1WormholeConfig = field(
        default_factory=ExperimentV1WormholeConfig
    )
    experiment_lattice: ExperimentLatticeConfig = field(default_factory=ExperimentLatticeConfig)
    experiment_s1: ExperimentS1Config = field(default_factory=ExperimentS1Config)
    experiment_paid_access: ExperimentPaidAccessConfig = field(
        default_factory=ExperimentPaidAccessConfig
    )
    experiment_minus_physics: ExperimentMinusPhysicsConfig = field(
        default_factory=ExperimentMinusPhysicsConfig
    )
    experiment_kt: ExperimentKTConfig = field(default_factory=ExperimentKTConfig)
    experiment_learned_memory: ExperimentLearnedMemoryConfig = field(
        default_factory=ExperimentLearnedMemoryConfig
    )
    experiment_retrieval: ExperimentRetrievalConfig = field(
        default_factory=ExperimentRetrievalConfig
    )
    experiment_dim_scaling: ExperimentDimScalingConfig = field(
        default_factory=ExperimentDimScalingConfig
    )
    experiment_potential_class: ExperimentPotentialClassConfig = field(
        default_factory=ExperimentPotentialClassConfig
    )
    experiment_designed_mechanism: ExperimentDesignedMechanismConfig = field(
        default_factory=ExperimentDesignedMechanismConfig
    )
    experiment_primitive_harness: ExperimentPrimitiveHarnessConfig = field(
        default_factory=ExperimentPrimitiveHarnessConfig
    )
    experiment_sequential_write: ExperimentSequentialWriteConfig = field(
        default_factory=ExperimentSequentialWriteConfig
    )
    experiment_hopfield_capacity: ExperimentHopfieldCapacityConfig = field(
        default_factory=ExperimentHopfieldCapacityConfig
    )
    experiment_phi_read_in: ExperimentPhiReadInConfig = field(
        default_factory=ExperimentPhiReadInConfig
    )
    experiment_retry_compute: ExperimentRetryComputeConfig = field(
        default_factory=ExperimentRetryComputeConfig
    )
    experiment_controller_mvp: ExperimentControllerMvpConfig = field(
        default_factory=ExperimentControllerMvpConfig
    )
    experiment_write_ceiling: ExperimentWriteCeilingConfig = field(
        default_factory=ExperimentWriteCeilingConfig
    )
    experiment_phi_stream: ExperimentPhiStreamConfig = field(
        default_factory=ExperimentPhiStreamConfig
    )
    experiment_cl_entry: ExperimentClEntryConfig = field(default_factory=ExperimentClEntryConfig)
    experiment_sharded_store: ExperimentShardedStoreConfig = field(
        default_factory=ExperimentShardedStoreConfig
    )
    experiment_well_lifecycle: ExperimentWellLifecycleConfig = field(
        default_factory=ExperimentWellLifecycleConfig
    )
    experiment_persistent_store: ExperimentPersistentStoreConfig = field(
        default_factory=ExperimentPersistentStoreConfig
    )
    experiment_phi_geometry: ExperimentPhiGeometryConfig = field(
        default_factory=ExperimentPhiGeometryConfig
    )
    experiment_capture_strong_phi: ExperimentCaptureStrongPhiConfig = field(
        default_factory=ExperimentCaptureStrongPhiConfig
    )
    experiment_capture_arm_a: ExperimentCaptureArmAConfig = field(
        default_factory=ExperimentCaptureArmAConfig
    )
    experiment_capture_armb: ExperimentCaptureArmBConfig = field(
        default_factory=ExperimentCaptureArmBConfig
    )
    data: DataConfig = field(default_factory=DataConfig)
    gen_data: GenDataConfig = field(default_factory=GenDataConfig)
    gen_bridge: GenBridgeConfig = field(default_factory=GenBridgeConfig)
    gen_store: GenStoreConfig = field(default_factory=GenStoreConfig)
    gen_r0_wide: GenR0WideConfig = field(default_factory=GenR0WideConfig)
    gen_metric_invariance: GenMetricInvarianceConfig = field(
        default_factory=GenMetricInvarianceConfig
    )
    gen_dream: GenDreamConfig = field(default_factory=GenDreamConfig)
    gen_certificate: GenCertificateConfig = field(default_factory=GenCertificateConfig)
    gen_baselines: "GenBaselinesConfig" = field(default_factory=lambda: GenBaselinesConfig())
    gen_compose: "GenComposeConfig" = field(default_factory=lambda: GenComposeConfig())
    gen_classifier: "GenClassifierConfig" = field(default_factory=lambda: GenClassifierConfig())
    gen_cell: "GenCellConfig" = field(default_factory=lambda: GenCellConfig())
    gen_bridge_dsprites: "GenBridgeDspritesConfig" = field(
        default_factory=lambda: GenBridgeDspritesConfig()
    )
    gen_encoder_unfreeze: "GenEncoderUnfreezeConfig" = field(
        default_factory=lambda: GenEncoderUnfreezeConfig()
    )
    gen_pretrain: "GenPretrainConfig" = field(default_factory=lambda: GenPretrainConfig())
    project: ProjectConfig = field(default_factory=ProjectConfig)


def get_default_config() -> NCLUConfig:
    """Get default configuration with all parameters set to their defaults."""
    return NCLUConfig()


def load_config(path: Path) -> NCLUConfig:
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
                if isinstance(v, str):
                    try:
                        if "." not in v and "e" not in v.lower():
                            filtered[k] = int(v)
                        else:
                            filtered[k] = float(v)
                    except (ValueError, TypeError):
                        filtered[k] = v
                else:
                    filtered[k] = v
        return filtered

    config = NCLUConfig(
        model=ModelConfig(**filter_valid_fields(ModelConfig, data.get("model", {}))),
        training=TrainingConfig(**filter_valid_fields(TrainingConfig, data.get("training", {}))),
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
            **filter_valid_fields(ExperimentV1GateConfig, data.get("experiment_v1_gate", {}))
        ),
        experiment_v1_wormhole=ExperimentV1WormholeConfig(
            **filter_valid_fields(
                ExperimentV1WormholeConfig, data.get("experiment_v1_wormhole", {})
            )
        ),
        experiment_lattice=ExperimentLatticeConfig(
            **filter_valid_fields(ExperimentLatticeConfig, data.get("experiment_lattice", {}))
        ),
        experiment_s1=ExperimentS1Config(
            **filter_valid_fields(ExperimentS1Config, data.get("experiment_s1", {}))
        ),
        experiment_paid_access=ExperimentPaidAccessConfig(
            **filter_valid_fields(
                ExperimentPaidAccessConfig, data.get("experiment_paid_access", {})
            )
        ),
        experiment_minus_physics=ExperimentMinusPhysicsConfig(
            **filter_valid_fields(
                ExperimentMinusPhysicsConfig, data.get("experiment_minus_physics", {})
            )
        ),
        experiment_kt=ExperimentKTConfig(
            **filter_valid_fields(ExperimentKTConfig, data.get("experiment_kt", {}))
        ),
        experiment_retrieval=ExperimentRetrievalConfig(
            **filter_valid_fields(ExperimentRetrievalConfig, data.get("experiment_retrieval", {}))
        ),
        experiment_dim_scaling=ExperimentDimScalingConfig(
            **filter_valid_fields(
                ExperimentDimScalingConfig, data.get("experiment_dim_scaling", {})
            )
        ),
        experiment_learned_memory=ExperimentLearnedMemoryConfig(
            **filter_valid_fields(
                ExperimentLearnedMemoryConfig, data.get("experiment_learned_memory", {})
            )
        ),
        experiment_potential_class=ExperimentPotentialClassConfig(
            **filter_valid_fields(
                ExperimentPotentialClassConfig, data.get("experiment_potential_class", {})
            )
        ),
        experiment_designed_mechanism=ExperimentDesignedMechanismConfig(
            **filter_valid_fields(
                ExperimentDesignedMechanismConfig, data.get("experiment_designed_mechanism", {})
            )
        ),
        experiment_primitive_harness=ExperimentPrimitiveHarnessConfig(
            **filter_valid_fields(
                ExperimentPrimitiveHarnessConfig, data.get("experiment_primitive_harness", {})
            )
        ),
        experiment_sequential_write=ExperimentSequentialWriteConfig(
            **filter_valid_fields(
                ExperimentSequentialWriteConfig, data.get("experiment_sequential_write", {})
            )
        ),
        experiment_hopfield_capacity=ExperimentHopfieldCapacityConfig(
            **filter_valid_fields(
                ExperimentHopfieldCapacityConfig, data.get("experiment_hopfield_capacity", {})
            )
        ),
        experiment_phi_read_in=ExperimentPhiReadInConfig(
            **filter_valid_fields(ExperimentPhiReadInConfig, data.get("experiment_phi_read_in", {}))
        ),
        experiment_retry_compute=ExperimentRetryComputeConfig(
            **filter_valid_fields(
                ExperimentRetryComputeConfig, data.get("experiment_retry_compute", {})
            )
        ),
        experiment_controller_mvp=ExperimentControllerMvpConfig(
            **filter_valid_fields(
                ExperimentControllerMvpConfig, data.get("experiment_controller_mvp", {})
            )
        ),
        experiment_write_ceiling=ExperimentWriteCeilingConfig(
            **filter_valid_fields(
                ExperimentWriteCeilingConfig, data.get("experiment_write_ceiling", {})
            )
        ),
        experiment_phi_stream=ExperimentPhiStreamConfig(
            **filter_valid_fields(ExperimentPhiStreamConfig, data.get("experiment_phi_stream", {}))
        ),
        experiment_cl_entry=ExperimentClEntryConfig(
            **filter_valid_fields(ExperimentClEntryConfig, data.get("experiment_cl_entry", {}))
        ),
        experiment_sharded_store=ExperimentShardedStoreConfig(
            **filter_valid_fields(
                ExperimentShardedStoreConfig, data.get("experiment_sharded_store", {})
            )
        ),
        experiment_well_lifecycle=ExperimentWellLifecycleConfig(
            **filter_valid_fields(
                ExperimentWellLifecycleConfig, data.get("experiment_well_lifecycle", {})
            )
        ),
        experiment_persistent_store=ExperimentPersistentStoreConfig(
            **filter_valid_fields(
                ExperimentPersistentStoreConfig, data.get("experiment_persistent_store", {})
            )
        ),
        experiment_phi_geometry=ExperimentPhiGeometryConfig(
            **filter_valid_fields(
                ExperimentPhiGeometryConfig, data.get("experiment_phi_geometry", {})
            )
        ),
        experiment_capture_strong_phi=ExperimentCaptureStrongPhiConfig(
            **filter_valid_fields(
                ExperimentCaptureStrongPhiConfig, data.get("experiment_capture_strong_phi", {})
            )
        ),
        experiment_capture_arm_a=ExperimentCaptureArmAConfig(
            **filter_valid_fields(
                ExperimentCaptureArmAConfig, data.get("experiment_capture_arm_a", {})
            )
        ),
        experiment_capture_armb=ExperimentCaptureArmBConfig(
            **filter_valid_fields(
                ExperimentCaptureArmBConfig, data.get("experiment_capture_armb", {})
            )
        ),
        data=DataConfig(**filter_valid_fields(DataConfig, data.get("data", {}))),
        gen_data=GenDataConfig(**filter_valid_fields(GenDataConfig, data.get("gen_data", {}))),
        gen_bridge=GenBridgeConfig(
            **filter_valid_fields(GenBridgeConfig, data.get("gen_bridge", {}))
        ),
        gen_store=GenStoreConfig(**filter_valid_fields(GenStoreConfig, data.get("gen_store", {}))),
        gen_r0_wide=GenR0WideConfig(
            **filter_valid_fields(GenR0WideConfig, data.get("gen_r0_wide", {}))
        ),
        gen_metric_invariance=GenMetricInvarianceConfig(
            **filter_valid_fields(GenMetricInvarianceConfig, data.get("gen_metric_invariance", {}))
        ),
        gen_dream=GenDreamConfig(**filter_valid_fields(GenDreamConfig, data.get("gen_dream", {}))),
        gen_certificate=GenCertificateConfig(
            **filter_valid_fields(GenCertificateConfig, data.get("gen_certificate", {}))
        ),
        gen_baselines=GenBaselinesConfig(
            **filter_valid_fields(GenBaselinesConfig, data.get("gen_baselines", {}))
        ),
        gen_compose=GenComposeConfig(
            **filter_valid_fields(GenComposeConfig, data.get("gen_compose", {}))
        ),
        gen_classifier=GenClassifierConfig(
            **filter_valid_fields(GenClassifierConfig, data.get("gen_classifier", {}))
        ),
        gen_cell=GenCellConfig(**filter_valid_fields(GenCellConfig, data.get("gen_cell", {}))),
        gen_bridge_dsprites=GenBridgeDspritesConfig(
            **filter_valid_fields(GenBridgeDspritesConfig, data.get("gen_bridge_dsprites", {}))
        ),
        gen_encoder_unfreeze=GenEncoderUnfreezeConfig(
            **filter_valid_fields(GenEncoderUnfreezeConfig, data.get("gen_encoder_unfreeze", {}))
        ),
        gen_pretrain=GenPretrainConfig(
            **filter_valid_fields(GenPretrainConfig, data.get("gen_pretrain", {}))
        ),
        project=ProjectConfig(**filter_valid_fields(ProjectConfig, data.get("project", {}))),
    )
    return config


@dataclass
class GenBaselinesConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    arms: List[str] = field(default_factory=lambda: ["knn", "lace"])
    bridge_root: Optional[str] = None
    decoder_source: str = "decoder_lift_v1"
    decoder_lift_dir: Optional[str] = None
    assert_decoder_hash: bool = True
    verify_cache_checksum: bool = True
    encode_chunk: int = 16384
    store_bytes: Optional[int] = None
    include_coupling_bytes: bool = False
    expect_byte_target_bytes: Optional[int] = 524288
    coupling_bytes: Optional[int] = None
    store_runs_root: Optional[str] = None
    store_ceiling_balacc: float = 0.6866
    n_fit: int = 32256
    n_val: int = 512
    eval_batch: int = 256
    knn_ks: List[int] = field(default_factory=lambda: [1, 3, 5])
    knn_metrics: List[str] = field(default_factory=lambda: ["euclidean", "cosine"])
    knn_normalisers: List[str] = field(default_factory=lambda: ["none", "zscore"])
    knn_deployments: List[str] = field(default_factory=lambda: ["fp32", "fp16", "int8"])
    knn_store_method: str = "kmeans"
    knn_kmeans_iters: int = 25
    knn_refusal_reduce: str = "min"
    lace_sigmas: List[float] = field(default_factory=lambda: [0.03, 0.1, 0.3, 1.0])
    lace_forms: List[str] = field(default_factory=lambda: ["categorical", "continuous", "both"])
    lace_shrinkages: List[str] = field(default_factory=lambda: ["none", "ledoit_wolf", "1e-3"])
    lace_n_epochs: int = 50
    lace_batch: int = 128
    lace_lr: float = 0.0001
    lace_warmup_iters: int = 1000
    lace_tune_epochs: int = 6
    lace_tune_n: int = 256
    lace_tune_ld_steps: int = 30
    lace_ld_steps: int = 100
    lace_samplers: List[str] = field(
        default_factory=lambda: ["ode", "ld_published", "ld_corrected"]
    )
    lace_tune_or_not: bool = True
    lace_or_cofs: List[float] = field(default_factory=lambda: [5.0, 20.0, 80.0, 320.0])
    lace_not_nums: List[float] = field(default_factory=lambda: [0.03, 0.1, 0.3, 1.0])
    lace_whitening_exact_tol: float = 0.1
    lace_whitening_exactness_fallback: bool = True
    refusal_valid_quantile: float = 0.95
    score_registered_controls: bool = True
    query_set_index_file: str = "query_sets.json"
    require_registered_controls: bool = True
    control_dedup_sensitivity: bool = True
    persist_per_request_refusal_scores: bool = True
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    make_plots: bool = True
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_n_fit: int = 2048
    quick_n_val: int = 64
    quick_n_eval_a: int = 64
    quick_lace_n_epochs: int = 1
    quick_lace_tune_epochs: int = 1
    quick_lace_tune_n: int = 32
    quick_lace_tune_ld_steps: int = 3
    quick_lace_ld_steps: int = 5
    quick_knn_kmeans_iters: int = 3
    quick_knn_ks: List[int] = field(default_factory=lambda: [1, 3])
    quick_lace_sigmas: List[float] = field(default_factory=lambda: [0.1])
    quick_lace_shrinkages: List[str] = field(default_factory=lambda: ["none"])


@dataclass
class GenComposeConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    arms: List[str] = field(default_factory=lambda: ["read", "clu_wide", "store_removed", "proto"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    bridge_root: Optional[str] = None
    cell_root: Optional[str] = None
    store_plan: Optional[str] = None
    decoder_source: str = "winner"
    winner_root: Optional[str] = None
    winner_decoder_pattern: str = "decoder_v1_s{seed}.pkl"
    expected_decoder_recipe: str = "v1_capacity_schedule"
    verify_cache_checksum: bool = True
    encode_chunk: int = 16384
    grouping_source: str = "per_seed"
    use_store_checkpoint: bool = True
    address_smoothing: float = 1.0
    store_removed_delete_mode: str = "mean"
    n_eval_a: int = 512
    eval_batch: int = 256
    invariance_neg_repeats: int = 2
    invariance_subsample_frac: float = 0.5
    invariance_seed: int = 0
    store_ceiling_balacc: float = 0.6866
    knn_proto_reference_balacc: float = 0.8884
    knn_channel_matched_reference_balacc: float = 0.5206
    expect_deployed_store_bytes: Optional[int] = 524288
    make_plots: bool = True
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_n_eval_a: int = 64
    quick_n_store_train: int = 2048
    quick_store_train_steps: int = 30
    quick_grouping_source: str = "plan"
    quick_arms: List[str] = field(
        default_factory=lambda: ["read", "clu_wide", "store_removed", "proto"]
    )


@dataclass
class GenClassifierConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    clf_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    clf_kernel: int = 5
    clf_stride: int = 2
    clf_groups: int = 8
    clf_hidden: int = 256
    clf_steps: int = 1500
    clf_batch: int = 128
    clf_lr: float = 0.001
    clf_checkpoint_every: int = 250
    n_clf_train: int = 8192
    row_seed_mult: int = 104729
    row_seed_offset: int = 20260913
    render_check_n: int = 64
    exclude_wide_heldout: bool = True
    bridge_root: Optional[str] = None
    cell_root: Optional[str] = None
    decoder_source: str = "winner"
    winner_root: Optional[str] = None
    winner_decoder_pattern: str = "decoder_v1_s{seed}.pkl"
    expected_decoder_recipe: str = "v1_capacity_schedule"
    verify_cache_checksum: bool = True
    eval_batch: int = 256
    gate_factor: str = "shape"
    gate_floor: float = 0.95
    gate_min_negatives: int = 20
    probe_reference_balacc: float = 0.9414
    probe_reference_per_seed: List[float] = field(
        default_factory=lambda: [0.9223484848484849, 0.9364530988274706, 0.9652777777777778]
    )
    wide_reference_balacc: float = 0.9894
    gap_abs_threshold: float = 0.05
    gap_spread_mult: float = 2.0
    make_plots: bool = True
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_n_clf_train: int = 512
    quick_clf_steps: int = 20
    quick_clf_checkpoint_every: int = 10
    quick_render_check_n: int = 8
    quick_verify_cache_checksum: bool = False


@dataclass
class GenCellConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    arms: List[str] = field(
        default_factory=lambda: ["clu", "store_removed", "knn_proto", "lace", "ebm_compose"]
    )
    cell_root: Optional[str] = None
    bridge_root: Optional[str] = None
    decoder_source: str = "winner"
    winner_root: Optional[str] = None
    winner_decoder_pattern: str = "decoder_v1_s{seed}.pkl"
    expected_decoder_recipe: str = "v1_capacity_schedule"
    classifier_root: Optional[str] = None
    classifier_checkpoint_pattern: str = "seed_{seed}/external_classifier.pkl"
    classifier_manifest_pattern: str = "seed_{seed}/classifier_manifest.json"
    classifier_gate_file: str = "classifier_gate.json"
    require_classifier_gate: bool = True
    refusal_scores_file: Optional[str] = "gen-arms-a7-wiring/refusal_scores.json"
    require_refusal_complete: bool = True
    store_plan: Optional[str] = None
    store_checkpoint_root: Optional[str] = None
    store_checkpoint_pattern: str = "runs/{dataset}_{pretext}/seed_{seed}/store_s{seed}.pkl"
    allow_store_training: bool = True
    grouping_source: str = "per_seed"
    address_smoothing: float = 1.0
    store_removed_delete_mode: str = "mean"
    verify_cache_checksum: bool = True
    encode_chunk: int = 2048
    eval_batch: int = 256
    n_eval_a: int = 0
    expect_deployed_store_bytes: Optional[int] = 524288
    addressing_head_bytes_reference: int = 74240
    prototype_table_bytes_reference: int = 475136
    prior_byte_budget: int = 688128
    tuning_trial_budget: int = 36
    lace_selection_source: Optional[str] = "gen-arms-a7-wiring/runs/historical_688128"
    lace_selection_pattern: str = "seed_{seed}/baselines_manifest.json"
    lace_samplers: List[str] = field(
        default_factory=lambda: ["ode", "ld_published", "ld_corrected"]
    )
    lace_primary_sampler: str = "ode"
    lace_n_epochs: int = 50
    lace_batch: int = 128
    lace_lr: float = 0.0001
    lace_warmup_iters: int = 1000
    lace_ld_steps: int = 100
    n_fit: int = 32256
    n_val: int = 512
    ebm_hidden: int = 256
    ebm_depth: int = 2
    ebm_epochs: int = 20
    ebm_batch: int = 128
    ebm_lr: float = 0.001
    ebm_step_sizes: List[float] = field(default_factory=lambda: [0.003, 0.01, 0.03])
    ebm_prior_sigmas: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    ebm_steps_grid: List[int] = field(default_factory=lambda: [50, 200])
    ebm_clip_grid: List[bool] = field(default_factory=lambda: [True, False])
    ebm_tune_n: int = 256
    sampler_noise_repeats: int = 3
    sampler_noise_arms: List[str] = field(default_factory=lambda: ["lace"])
    sampler_noise_seed: int = 0
    clu_probe_reference_per_seed: List[float] = field(
        default_factory=lambda: [0.6609848484848484, 0.6809045226130653, 0.685824958123953]
    )
    knn_proto_probe_reference_per_seed: List[float] = field(
        default_factory=lambda: [0.8762626262626263, 0.8919597989949748, 0.8969849246231156]
    )
    lace_ode_probe_reference_per_seed: List[float] = field(
        default_factory=lambda: [0.7077020202020202, 0.6731574539363484, 0.6582216638749302]
    )
    rival_regression_tolerance: float = 0.05
    make_plots: bool = True
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_arms: List[str] = field(
        default_factory=lambda: ["clu", "store_removed", "knn_proto", "lace", "ebm_compose"]
    )
    quick_n_fit: int = 4096
    quick_n_val: int = 256
    quick_lace_n_epochs: int = 1
    quick_lace_ld_steps: int = 5
    quick_ebm_epochs: int = 1
    quick_ebm_step_sizes: List[float] = field(default_factory=lambda: [0.01])
    quick_ebm_prior_sigmas: List[float] = field(default_factory=lambda: [1.0])
    quick_ebm_steps_grid: List[int] = field(default_factory=lambda: [10])
    quick_ebm_clip_grid: List[bool] = field(default_factory=lambda: [True])
    quick_ebm_tune_n: int = 32
    quick_sampler_noise_repeats: int = 2
    quick_verify_cache_checksum: bool = False


@dataclass
class GenBridgeDspritesConfig:
    dec_recipe: str = "v1_capacity_schedule"
    quick_dec_recipe: str = "baseline"
    expected_decoder_recipe: str = "v1_capacity_schedule"
    cache_block: int = 23040
    gate_factor: str = "shape"
    max_positives: int = 6292
    max_negatives: int = 4096
    max_seen: int = 4096
    min_negatives: int = 20
    disc_floor: float = 0.8
    disc_abs_floor: float = 0.75
    real_instrument: bool = True
    eval_batch: int = 256
    make_plots: bool = True
    quick_max_positives: int = 64
    quick_max_negatives: int = 64
    quick_max_seen: int = 64


@dataclass
class GenEncoderUnfreezeConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    arms: List[str] = field(default_factory=lambda: ["frozen", "unfrozen"])
    seeds: List[int] = field(default_factory=lambda: [0])
    warmup_steps: int = 300
    warmup_batch: int = 64
    n_train: int = 4096
    max_matched_seen: int = 1024
    k_wells: int = 5
    n_modes: int = 64
    grouping_method: str = "auto"
    min_well_frac: float = 0.1
    joint_steps: int = 600
    joint_batch: int = 32
    joint_lr: float = 0.001
    code_grad: str = "straight_through"
    beta_codebook: float = 1.0
    beta_commitment: float = 0.25
    dec_recipe: str = "v1_capacity_schedule"
    expected_decoder_recipe: str = "v1_capacity_schedule"
    quick_dec_recipe: str = "v1_capacity_schedule"
    book_smoothing: float = 1.0
    book_rows: Optional[int] = None
    out_root: Optional[str] = None
    skip_if_exists: bool = True
    persist_code_tuples: bool = True
    make_plots: bool = False
    quick_warmup_steps: int = 40
    quick_joint_steps: int = 60
    quick_joint_batch: int = 16
    quick_warmup_batch: int = 16
    quick_n_train: int = 512
    quick_k_wells: int = 5
    quick_n_modes: int = 64
    quick_max_matched_seen: int = 256
    quick_book_rows: Optional[int] = None
    quick_seeds: List[int] = field(default_factory=lambda: [0])


@dataclass
class GenPretrainConfig:
    dataset: str = "shapes3d"
    variants: List[str] = field(
        default_factory=lambda: [
            "recon_control",
            "random_init",
            "ae_plain",
            "recon_control_2x",
            "ae_deep",
            "mae",
            "ae_hires",
            "random_init_hires",
            "vae",
        ]
    )
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    n_enc_train: int = 4096
    n_probe_train: int = 4096
    n_probe_eval: int = 2048
    include_corner: bool = True
    enc_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    hires_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    enc_kernel: int = 4
    enc_stride: int = 2
    enc_groups: int = 8
    enc_steps: int = 400
    enc_batch: int = 128
    enc_lr: float = 0.001
    dec_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    dec_groups: int = 8
    deep_width_mult: float = 1.0
    deep_blocks_per_stage: int = 2
    mae_mask_ratio: float = 0.5
    mae_patch: int = 8
    mae_loss_on_masked: bool = True
    vae_beta: float = 0.05
    compute_control_mult: int = 2
    probe_kinds: List[str] = field(default_factory=lambda: ["linear", "mlp"])
    probe_hidden: int = 256
    probe_steps: int = 1500
    probe_batch: int = 256
    probe_lr: float = 0.001
    pass_bar: float = 0.9
    bar_factors: List[str] = field(default_factory=lambda: ["scale", "orientation"])
    cache_block: int = 8192
    eval_batch: int = 256
    skip_if_exists: bool = True
    out_dir: Optional[str] = None
    make_plots: bool = True
    quick_variants: List[str] = field(
        default_factory=lambda: [
            "recon_control",
            "random_init",
            "ae_plain",
            "ae_deep",
            "mae",
            "ae_hires",
            "random_init_hires",
            "vae",
        ]
    )
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_enc_steps: int = 12
    quick_probe_steps: int = 12
    quick_n_enc_train: int = 128
    quick_n_probe_train: int = 128
    quick_n_probe_eval: int = 64


@dataclass
class GenDynamicalReadConfig:
    dataset: str = "shapes3d"
    pretext: str = "recon"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    commit_cache_pattern: str = (
        "outputs/gen-commitment-collapse/runs/{dataset}_{pretext}/seed_{seed}"
    )
    store_checkpoint_source: str = (
        "outputs/gen-read-compose/runs/{dataset}_{pretext}/seed_{seed}/store_s{seed}.pkl"
    )
    assert_store_hash: bool = True
    verify_cache_checksum: bool = True
    encode_chunk: int = 8192
    winner_decoder_pattern: str = "outputs/gen-decoder-lift/winner/decoder_v1_s{seed}.pkl"
    winner_hashes_path: str = "outputs/gen-decoder-lift/winner/winner_hashes.json"
    dec_recipe: str = "v1_capacity_schedule"
    assert_decoder_hash: bool = True
    budgets: List[str] = field(default_factory=lambda: ["A", "B"])
    budget_a_atoms: int = 64
    budget_a_rank: int = 16
    budget_a_budget_bytes: int = 524288
    budget_b_atoms: int = 512
    budget_b_rank: int = 128
    budget_b_budget_bytes: int = 4194304
    assert_byte_footprint: bool = True
    atom_selection: str = "kmeanspp"
    bandwidth_frac: float = 1.0
    bandwidth_rule: str = "nn_median"
    address_smoothing: float = 1.0
    request_strength: float = 1.0
    coupling_strength: float = 1.0
    coupling_normalise: bool = True
    enforce_coupling_certificate: bool = True
    coupling_safety_frac: float = 0.9
    divergence_radius_mult: float = 50.0
    n_coupling_fit: int = 8192
    launch_mode: str = "clamp_atom"
    clamp_threshold: float = 0.5
    read_mode: str = "state"
    steps: int = 1000
    step_size: float = 0.2
    sigma_scale_start: float = 4.0
    sigma_scale_end: float = 1.0
    temp_start: float = 0.0
    temp_end: float = 0.0
    metropolis: bool = True
    descent_window: int = 100
    descent_tol: float = 0.001
    request_chunk: int = 256
    arms: List[str] = field(
        default_factory=lambda: [
            "full",
            "request_off",
            "request_off_launch",
            "coupling_off",
            "zero_step",
        ]
    )
    n_store_train: int = 32768
    n_spread_requests: int = 512
    row_seed_mult: int = 7919
    row_seed_offset: int = 20260914
    decode_reads: bool = True
    decode_rows: int = 72
    decode_batch: int = 64
    demonstrate_bypass_failure: bool = True
    bypass_alpha: float = 0.01
    make_plots: bool = True
    skip_if_exists: bool = True
    out_root: Optional[str] = None
    cell_root: Optional[str] = None
    quick_seeds: List[int] = field(default_factory=lambda: [0])
    quick_budgets: List[str] = field(default_factory=lambda: ["A"])
    quick_n_store_train: int = 2048
    quick_n_coupling_fit: int = 1024
    quick_n_spread_requests: int = 128
    quick_steps: int = 25
    quick_descent_window: int = 10
    quick_decode_rows: int = 16
    quick_request_chunk: int = 72
    quick_verify_cache_checksum: bool = False
    quick_assert_byte_footprint: bool = True

    def budget_spec(self, label: str) -> dict:
        lab = str(label).upper()
        if lab == "A":
            return {
                "label": "A",
                "role": "MATCHED",
                "n_atoms": int(self.budget_a_atoms),
                "rank": int(self.budget_a_rank),
                "budget_bytes": int(self.budget_a_budget_bytes),
            }
        if lab == "B":
            return {
                "label": "B",
                "role": "DECLARED LARGER",
                "n_atoms": int(self.budget_b_atoms),
                "rank": int(self.budget_b_rank),
                "budget_bytes": int(self.budget_b_budget_bytes),
            }
        raise ValueError(f"unknown budget {label!r} — PROTOCOL §3 declares A and B only")

    def byte_footprint_expectations(self, label: str) -> dict:
        spec = self.budget_spec(label)
        b = int(spec["budget_bytes"])
        return {
            "store_bytes": b,
            "coupling_bytes": b,
            "total_bytes": 2 * b,
            "multiple_of_budget": 2.0,
        }


def gen_dynamical_read_config(
    config: Optional["NCLUConfig"] = None, **overrides
) -> GenDynamicalReadConfig:
    cfg = getattr(config, "gen_dynamical_read", None) if config is not None else None
    if cfg is None:
        cfg = GenDynamicalReadConfig()
        if config is not None:
            object.__setattr__(config, "gen_dynamical_read", cfg)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


@dataclass
class GenRowSplitConfig:
    mode: str = "stratified"
    tiebreak_jitter: float = 1.0
    seed_mult: int = 6637
    seed_offset: int = 20260914
    assert_balanced: bool = True
    require_all_values: bool = True
    max_count_ratio: float = 1.25
    record_balance: bool = True


def gen_row_split_config(config: Optional["NCLUConfig"] = None, **overrides) -> GenRowSplitConfig:
    cfg = getattr(config, "gen_row_split", None) if config is not None else None
    if cfg is None:
        cfg = GenRowSplitConfig()
        if config is not None:
            object.__setattr__(config, "gen_row_split", cfg)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
